from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import io
import os
import fcntl
import shutil
import tarfile
import subprocess
import re
import gzip
import time
import json
import urllib3
import hashlib
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import RDLogger
from src.utils.rich_progress import progress
from src.utils.structure_utils import MIN_COVERAGE, select_chains, selection_counts
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rdkit.Chem import AllChem
from molvs import standardize_smiles
from hydra import initialize, compose
from multiprocessing.pool import ThreadPool
from hydra.core.global_hydra import GlobalHydra
from rdkit.Chem import (
    MolToMolFile,
    MolFromInchi,
    MolFromMolFile,
    MolFromSmiles,
    MolToSmiles,
)
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

_REACTION_SPECTATORS = {"O", "[H+]"}


TEMP_RANGE_K = (273.0, 373.15)
PH_RANGE = (0.0, 14.0)

# The AlphaFold file server rejects the default python-requests User-Agent.
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) EMULaToR"}

HTTP_RETRIES = 2
HTTP_BUSY_RETRIES = 5  # attempts when the server answers 429 or 5xx
HTTP_BACKOFF = 5  # seconds; doubles after each busy answer
PUBMED_TIMEOUT = 120  # seconds for one efetch batch of up to 10000 ids
AF_THREADS = 4
AF_RETRY_THREADS = 2
AF_RETRY_WAIT = 60
UNIPROT_BATCH = 100
UNIPROT_THREADS = 4
UNIPROT_RETRIES = 3
UNIPROT_TIMEOUT = 60
UNIPROT_RETRY_BATCH = 25
UNIPROT_RETRY_WAIT = 15
UNIPROT_ORG_CHUNK = 50
UNIPROT_EC_ORG_FIELDS = (
    "accession,organism_name,protein_existence,annotation_score,gene_names,lit_pubmed_id,sequence"
)
UNIPROT_EC_ORG_COLUMNS = [
    "reviewed",
    "UniprotID",
    "ECNumber",
    "Organism",
    "protein_existence",
    "annotation_score",
    "genes",
    "pmids",
    "sequence",
]


def null_implausible_conditions(df: pd.DataFrame, temp_col: str, ph_col: str) -> dict[str, int]:
    """Set temperature outside 273-373.15 K and pH outside 0-14 to NaN, in place.

    Rows and labels stay; only the bad condition value is cleared. Returns the count cleared
    per column.
    """
    counts = {}
    for col, (low, high) in ((temp_col, TEMP_RANGE_K), (ph_col, PH_RANGE)):
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ~values.between(low, high)
        df[col] = values.mask(bad, np.nan)
        counts[col] = int(bad.sum())
    return counts


def is_tiny_or_inorganic(smiles) -> bool:
    """True for molecules with at most 3 heavy atoms or no carbon; unparsable SMILES are False."""
    mol = MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() <= 3 or not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


def molecule_keys(smiles, processes=None):
    """Map each unique SMILES to its molecule_key, in parallel (tautomers are slow)."""
    from multiprocessing import Pool

    unique = list(dict.fromkeys(s for s in smiles if isinstance(s, str)))
    with Pool(processes or os.cpu_count()) as pool:
        keys = pool.map(molecule_key, unique, chunksize=64)
    return dict(zip(unique, keys))


def undirected_reaction_key(key):
    """Order the two sides of "A>>B" so a reaction and its reverse share one key."""
    if not isinstance(key, str) or ">>" not in key:
        return key
    left, right = key.split(">>", 1)
    return ">>".join(sorted((left, right)))


def molecule_key(smiles):
    """Stereo-aware identity key: largest organic fragment, neutralized, canonical tautomer."""
    # Salts, charge states and tautomers of one compound share a key; stereo is kept so
    # stereoisomers stay distinct.
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetRemoveSp3Stereo(False)  # default drops stereo next to C=O
        enumerator.SetRemoveBondStereo(False)
        enumerator.SetMaxTautomers(100)  # the default 1000 cap made kcat keys ~9x slower
        enumerator.SetMaxTransforms(100)
        mol = enumerator.Canonicalize(mol)
        key = MolToSmiles(mol)
        return key if MolFromSmiles(key) is not None else None  # e.g. a bare cyclopentadienide
    except Exception:
        return None
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.utils.rich_progress import component_logger


class _BoundChemUtilsMethod:
    """Picklable stand-in for a ChemUtils method; the instance is built on the first call."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return getattr(chem_utils_instance(), self.name)(*args, **kwargs)


_CHEM_UTILS = None


def chem_utils_instance():
    global _CHEM_UTILS
    if _CHEM_UTILS is None:
        _CHEM_UTILS = ChemUtils()
    return _CHEM_UTILS


class LazyChemUtils:
    """Module-level handle that builds ChemUtils (.env, Hydra compose) on first use."""

    # Attribute access and bind() defer construction until something is called, so importing
    # needs no .env or Hydra.
    def bind(self, name):
        return _BoundChemUtilsMethod(name)

    def __getattr__(self, name):
        return getattr(chem_utils_instance(), name)


class ChemUtils:
    def __init__(self, config_path="../../configs", config_name="data_processing"):
        if GlobalHydra.instance().is_initialized():
            cfg = compose(config_name=config_name)
        else:
            with initialize(version_base="1.3", config_path=config_path):
                cfg = compose(config_name=config_name)

        self.KEGG_MOL_URL = cfg.chem_utils.kegg_mol_url
        self.CHEBI_MOL_URL = cfg.chem_utils.chebi_mol_url
        self.PUBCHEM_COMPOUND_URL = cfg.chem_utils.pubchem_compound_url
        self.PUBCHEM_SUBSTANCE_URL = cfg.chem_utils.pubchem_substance_url
        self.PUBCHEM_SYNONYM_URL = cfg.chem_utils.pubchem_synonym_url
        self.PUBCHEM_CID_SMILES_URL = cfg.chem_utils.pubchem_cid_smiles_url
        self.UNIPROT_API_URL = cfg.chem_utils.uniprot_api_url
        self.ALPHAFOLD_API_URL = cfg.chem_utils.alphafold_api_url
        self.ALPHAFOLD_SWISSPROT_URL = cfg.chem_utils.alphafold_swissprot_url
        self.EBI_MMCIF_ARCHIVE_URL = cfg.chem_utils.ebi_mmcif_archive_url
        self.EBI_MMCIF_UPDATED_URL = cfg.chem_utils.ebi_mmcif_updated_url
        self.PUBMED_ENTREZ_EFETCH_URL = cfg.chem_utils.pubmed_entrez_efetch_url

        # Convert config paths to Path objects
        LOG_PATH = Path(cfg.chem_utils.log_dir)
        self.MOL_PATH = {k: Path(v) for k, v in cfg.chem_utils.mol_path.items()}
        self.SDF_PATH = Path(cfg.chem_utils.sdf_path)
        self.MMCIF_PATH = Path(cfg.chem_utils.mmcif_path)
        self.REFERENCE_DOWNLOAD_PATH = Path(cfg.chem_utils.reference_raw_path)
        self.PUBCHEM_CID_SEARCH_PATH = Path(cfg.chem_utils.pubchem_cid_search_path)
        self.AF_DOWNLOAD_PATH = Path(cfg.chem_utils.af_json_path)
        self.AF_PDB_PATH = Path(cfg.chem_utils.af_pdb_path)
        self.AF_BULK_PATH = Path(cfg.chem_utils.af_bulk_path)
        self.ESM_PDB_PATH = Path(cfg.chem_utils.esm_pdb_path)
        self.UNIPROT_ACC_PATH = Path(cfg.chem_utils.uniprot_acc_path)
        self.UNIPROT_TIME_PATH = Path(cfg.chem_utils.uniprot_time_path)
        self.UNIPROT_EC_ACC_PATH = Path(cfg.chem_utils.uniprot_ec_acc_path)
        self.UNIPROT_EC_ORG_ACC_PATH = Path(cfg.chem_utils.uniprot_ec_org_acc_path)

        self.PUBCHEM_CID_SMILES_GZ_FILE_PATH = Path(
            cfg.chem_utils.pubchem_cid_smiles_gz_file_path
        )
        self.PUBCHEM_SYNONYM_GZ_FILE_PATH = Path(
            cfg.chem_utils.pubchem_synonym_gz_file_path
        )
        self.PUBCHEM_NAME_CID_CACHE_FILE_PATH = Path(
            cfg.chem_utils.pubchem_name_cid_cache_file_path
        )
        self.PUBCHEM_CID_SMILES_CACHE_FILE_PATH = Path(
            cfg.chem_utils.pubchem_cid_smiles_cache_file_path
        )
        # Create directories if they don't exist
        for path in [
            LOG_PATH,
            self.SDF_PATH,
            self.MMCIF_PATH,
            self.REFERENCE_DOWNLOAD_PATH,
            self.PUBCHEM_CID_SEARCH_PATH,
            self.AF_DOWNLOAD_PATH,
            self.AF_PDB_PATH,
            self.AF_BULK_PATH,
            self.UNIPROT_ACC_PATH,
            self.UNIPROT_TIME_PATH,
            self.UNIPROT_EC_ACC_PATH,
            self.UNIPROT_EC_ORG_ACC_PATH,
        ] + list(self.MOL_PATH.values()):
            path.mkdir(parents=True, exist_ok=True)

        self.logger = component_logger(
            log_dir=LOG_PATH, log_file_name=cfg.chem_utils.log_file_name
        ).get_logger()

        ROOT_DIR = Path(rootutils.find_root(__file__, indicator=".project-root"))
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            self.logger.warning(
                f".env file not found at {env_path}. Environment variables may not be loaded."
            )
            raise FileNotFoundError(f".env file not found at {env_path}")

    def download(self, url, path, overwrite=False, verbose=False):
        """
        Download a file from a URL to a local path.

        Args:
            url (str): The URL to download from.
            path (Path): The local file path to save the downloaded file.
            overwrite (bool): If True, overwrite existing file. Default is False.
            verbose (bool): If True, log additional info.

        Returns:
            None
        """
        if not overwrite and path.exists():
            if verbose:
                self.logger.info(f"{path} already exists. Skipping download...")
            return
        response = requests.get(url)
        for _ in range(2):  # Retry up to 2 times
            if response.status_code == 404:
                self.logger.warning(f"404 - {url}")
                return
            if response.status_code != 200:
                self.logger.warning(
                    f"Failed to download with error code {response.status_code}. Retrying in 2 seconds..."
                )
                time.sleep(2)
                continue
            if response.headers.get("Content-Type") in [
                "application/gzip",
                "application/x-gzip",
            ]:
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gzip_file:
                    with open(path, "wb") as file:
                        file.write(gzip_file.read())
            else:
                with open(path, "wb") as file:
                    file.write(response.content)
            if verbose:
                self.logger.info(f"Successfully downloaded {path}")
            return
        self.logger.error(f"Failed to download {url} to {path}")
        return

    def get_html(self, url, path, timeout=10):
        """
        Download HTML content from a URL and save to a file.

        Args:
            url (str): The URL to fetch HTML from.
            path (Path): The local file path to save the HTML.

        Returns:
            int | None: 200 if the file was written, 404 if the URL does not exist,
            otherwise None.
        """
        if path.exists():
            return
        errors = 0
        for attempt in range(HTTP_BUSY_RETRIES):
            try:
                response = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
                if response.status_code == 200:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    return 200
                elif response.status_code == 404:
                    self.logger.warning(f"404 - {url}")
                    return 404
                elif response.status_code == 429 or response.status_code >= 500:
                    wait = HTTP_BACKOFF * 2**attempt
                    try:
                        wait = float(response.headers["Retry-After"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    if attempt < HTTP_BUSY_RETRIES - 1:
                        self.logger.warning(
                            f"HTTP {response.status_code} for {url}. Retrying in {wait:g}s..."
                        )
                        time.sleep(wait)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                self.logger.warning(
                    f"Failed to retrieve {url} with error {e}. Retrying..."
                )
                time.sleep(5)
            if errors >= HTTP_RETRIES:
                break
        self.logger.error(f"Failed to retrieve {url}. Max retries exceeded.")
        return

    def canonicalize_smiles(self, smiles):
        """
        Standardize and canonicalize a SMILES string using datamol.

        Applies a series of transformations to standardize SMILES:
        1. Convert SMILES string to molecule object
        2. Fix structural issues (implicit hydrogens, charges)
        3. Sanitize connectivity and valence
        4. Standardize (reionize, normalize, fix stereo; metals stay connected)
        5. Convert back to canonical SMILES representation

        This does not strip salts or other fragments; a multi-fragment SMILES keeps
        all of its fragments.

        Args:
            smiles (str): SMILES string to standardize.

        Returns:
            str: Standardized canonical SMILES, or None if any transformation fails.

        Raises:
            ImportError: If datamol is not installed.
        """
        try:
            import datamol as dm

            dm.disable_rdkit_log()
        except ImportError:
            self.logger.error(
                "datamol is not installed. Please install it to use canonicalize_smiles."
            )
            return None

        # Define transformation pipeline with operation names
        operations = [
            ("Convert SMILES to molecule", lambda m: dm.to_mol(m, ordered=True)),
            ("Fix molecular structure", dm.fix_mol),
            (
                "Sanitize molecule",
                lambda m: dm.sanitize_mol(m, sanifix=True, charge_neutral=False),
            ),
            (
                "Standardize molecule",
                lambda m: dm.standardize_mol(
                    m,
                    disconnect_metals=False,
                    normalize=True,
                    reionize=True,
                    uncharge=False,
                    stereo=True,
                ),
            ),
            (
                "Convert to canonical SMILES",
                lambda m: dm.standardize_smiles(dm.to_smiles(m)),
            ),
        ]

        # RDKit rewrites double-bond "/" "\\" marks after the first pass reorders atoms. Repeat
        # until the output is stable (max 3 passes) and return the last result.
        current = smiles
        for _ in range(3):
            mol = current
            for step_name, operation in operations:
                try:
                    mol = operation(mol)
                except Exception as e:
                    self.logger.warning(f"{step_name} failed for {smiles}: {e}")
                    return None
            if mol == current:
                break
            current = mol

        return mol

    def split_reaction_smiles(self, rxn_smiles):
        """
        Split a reaction SMILES string into stripped LHS/RHS metabolite lists.

        Args:
            rxn_smiles (str): Reaction SMILES string in the form "lhs>>rhs".

        Returns:
            tuple[list[str], list[str]] | tuple[None, None]:
                Stripped metabolite lists for LHS and RHS, or (None, None) if parsing fails.
        """
        if not isinstance(rxn_smiles, str):
            return None, None

        rxn_smiles = rxn_smiles.strip()
        if not rxn_smiles or ">>" not in rxn_smiles:
            return None, None

        lhs, rhs = rxn_smiles.split(">>", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        lhs_mets = [met.strip() for met in lhs.split(".") if met.strip()]
        rhs_mets = [met.strip() for met in rhs.split(".") if met.strip()]
        return lhs_mets, rhs_mets

    def canonicalize_reaction_smiles(self, rxn_smiles, canonical_smiles_map=None):
        """
        Canonicalize a reaction SMILES by canonicalizing each metabolite and
        sorting each side by metabolite length (descending).

        Args:
            rxn_smiles (str): Reaction SMILES string in the form "lhs>>rhs".
            canonical_smiles_map (dict, optional): Mapping from raw metabolite SMILES
                to canonical SMILES. If None, each metabolite is canonicalized on-the-fly.

        Returns:
            str | None: Canonicalized reaction SMILES, or None if parsing/canonicalization fails.
        """
        lhs_mets, rhs_mets = self.split_reaction_smiles(rxn_smiles)
        if lhs_mets is None or rhs_mets is None:
            return None

        def _canonicalize_side(metabolites):
            canonicalized = []
            for met in metabolites:
                canonical_met = (
                    canonical_smiles_map.get(met)
                    if canonical_smiles_map is not None
                    else self.canonicalize_smiles(met)
                )
                if not canonical_met:
                    return None
                canonicalized.append(canonical_met.strip())

            # Break same-length ties on the string itself.
            canonicalized = sorted(canonicalized, key=lambda s: (-len(s), s))
            return ".".join(canonicalized)

        lhs_canonical = _canonicalize_side(lhs_mets)
        rhs_canonical = _canonicalize_side(rhs_mets)
        if lhs_canonical is None or rhs_canonical is None:
            return None

        return f"{lhs_canonical}>>{rhs_canonical}"

    def canonicalize_reaction_smiles_series(self, rxn_smiles_series, logger=None):
        """
        Canonicalize a reaction SMILES collection by canonicalizing unique metabolite
        SMILES first, then reconstructing each reaction.

        Rules:
        1. Strip reaction SMILES.
        2. Split with ">>" into LHS and RHS, then strip both sides.
        3. Split each side with "." and strip each metabolite.
        4. Canonicalize unique metabolite SMILES in parallel.
        5. If any metabolite in a reaction fails canonicalization, that reaction is set to None.
        6. Sort metabolites on each side by length (descending), then join with "." and ">>".

        Args:
            rxn_smiles_series (pd.Series | list): Reaction SMILES values.
            logger: Optional logger to use. Defaults to self.logger.

        Returns:
            tuple[pd.Series, dict]:
                - Canonicalized reaction SMILES series aligned to input index.
                - Stats dictionary for logging/reporting.
        """
        active_logger = logger or self.logger
        series = pd.Series(rxn_smiles_series).copy()

        total_rows = len(series)
        non_null_rows = int(series.notna().sum())

        stripped = series.dropna().astype(str).str.strip()
        stripped = stripped[stripped != ""]
        unique_input_rxn = int(stripped.drop_duplicates().shape[0])

        unique_rxn_smiles = stripped[stripped.str.contains(">>", regex=False)]
        unique_rxn_smiles = unique_rxn_smiles.drop_duplicates().tolist()

        if not unique_rxn_smiles:
            stats = {
                "total_rows": total_rows,
                "non_null_rows": non_null_rows,
                "unique_input_rxn_smiles": unique_input_rxn,
                "unique_valid_format_rxn_smiles": 0,
                "unique_parse_failed_rxn_smiles": 0,
                "unique_metabolite_smiles": 0,
                "unique_metabolite_failed_canonicalization": 0,
                "unique_metabolite_canonicalized": 0,
                "unique_rxn_failed_due_to_metabolite_failure": 0,
                "unique_rxn_failed_during_reassembly": 0,
                "unique_rxn_canonicalized": 0,
                "rows_canonicalized": 0,
                "rows_set_to_none": non_null_rows,
            }
            active_logger.info(
                "Reaction SMILES canonicalization: no valid reaction SMILES to process."
            )
            return pd.Series([None] * total_rows, index=series.index), stats

        parsed_rxn = {}
        parse_failed_rxn = set()
        all_metabolites = []

        for rxn_smiles in unique_rxn_smiles:
            lhs_metabolites, rhs_metabolites = self.split_reaction_smiles(rxn_smiles)
            if lhs_metabolites is None or rhs_metabolites is None:
                parse_failed_rxn.add(rxn_smiles)
                continue
            parsed_rxn[rxn_smiles] = (lhs_metabolites, rhs_metabolites)
            all_metabolites.extend(lhs_metabolites)
            all_metabolites.extend(rhs_metabolites)

        unique_metabolites_df = (
            pd.DataFrame({"smiles": all_metabolites})
            .dropna(subset=["smiles"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if unique_metabolites_df.empty:
            canonical_series = pd.Series([None] * total_rows, index=series.index)
            stats = {
                "total_rows": total_rows,
                "non_null_rows": non_null_rows,
                "unique_input_rxn_smiles": unique_input_rxn,
                "unique_valid_format_rxn_smiles": len(unique_rxn_smiles),
                "unique_parse_failed_rxn_smiles": len(parse_failed_rxn),
                "unique_metabolite_smiles": 0,
                "unique_metabolite_failed_canonicalization": 0,
                "unique_metabolite_canonicalized": 0,
                "unique_rxn_failed_due_to_metabolite_failure": len(parsed_rxn),
                "unique_rxn_failed_during_reassembly": 0,
                "unique_rxn_canonicalized": 0,
                "rows_canonicalized": 0,
                "rows_set_to_none": non_null_rows,
            }
            active_logger.info(
                "Reaction SMILES canonicalization: no metabolite SMILES extracted from reactions."
            )
            return canonical_series, stats

        try:
            from pandarallel import pandarallel

            pandarallel.initialize(nb_workers=os.cpu_count())
            parallel = True
        except ImportError:
            parallel = False
            active_logger.warning(
                "pandarallel is not installed. Install it to speed up canonicalization of large reaction sets. Falling back to single-threaded processing."
            )

        active_logger.info(
            f"Canonicalizing {len(unique_metabolites_df)} unique metabolite SMILES"
            f" ({'in parallel' if parallel else 'single-threaded'})."
        )
        smiles_column = unique_metabolites_df["smiles"]
        unique_metabolites_df["canonical smiles"] = (
            smiles_column.parallel_apply(self.canonicalize_smiles)
            if parallel
            else smiles_column.apply(self.canonicalize_smiles)
        )

        failed_mets = unique_metabolites_df[
            unique_metabolites_df["canonical smiles"].isna()
        ]["smiles"]
        failed_mets_set = set(failed_mets.tolist())

        smiles_map = dict(
            unique_metabolites_df.dropna(subset=["canonical smiles"])[
                ["smiles", "canonical smiles"]
            ].values
        )

        rxn_smiles_map = {}
        rxn_failed_due_to_met = 0
        rxn_failed_during_reassembly = 0

        for rxn_smiles in unique_rxn_smiles:
            if rxn_smiles in parse_failed_rxn:
                rxn_smiles_map[rxn_smiles] = None
                continue

            lhs_metabolites, rhs_metabolites = parsed_rxn[rxn_smiles]
            metabolites = lhs_metabolites + rhs_metabolites
            if any(met in failed_mets_set for met in metabolites):
                rxn_smiles_map[rxn_smiles] = None
                rxn_failed_due_to_met += 1
                continue

            canonical_rxn = self.canonicalize_reaction_smiles(rxn_smiles, smiles_map)
            if canonical_rxn is None:
                rxn_smiles_map[rxn_smiles] = None
                rxn_failed_during_reassembly += 1
                continue

            rxn_smiles_map[rxn_smiles] = canonical_rxn

        stripped_full = series.copy()
        mask = stripped_full.notna()
        stripped_full.loc[mask] = stripped_full.loc[mask].astype(str).str.strip()
        canonical_series = stripped_full.map(rxn_smiles_map)
        canonical_series = canonical_series.where(canonical_series.notna(), None)

        rows_canonicalized = int(canonical_series.notna().sum())
        rows_set_to_none = int(non_null_rows - rows_canonicalized)
        unique_rxn_canonicalized = sum(
            1 for v in rxn_smiles_map.values() if isinstance(v, str) and v
        )

        stats = {
            "total_rows": total_rows,
            "non_null_rows": non_null_rows,
            "unique_input_rxn_smiles": unique_input_rxn,
            "unique_valid_format_rxn_smiles": len(unique_rxn_smiles),
            "unique_parse_failed_rxn_smiles": len(parse_failed_rxn),
            "unique_metabolite_smiles": int(len(unique_metabolites_df)),
            "unique_metabolite_failed_canonicalization": int(len(failed_mets_set)),
            "unique_metabolite_canonicalized": int(
                len(unique_metabolites_df) - len(failed_mets_set)
            ),
            "unique_rxn_failed_due_to_metabolite_failure": int(rxn_failed_due_to_met),
            "unique_rxn_failed_during_reassembly": int(rxn_failed_during_reassembly),
            "unique_rxn_canonicalized": int(unique_rxn_canonicalized),
            "rows_canonicalized": rows_canonicalized,
            "rows_set_to_none": rows_set_to_none,
        }

        active_logger.info("Reaction SMILES canonicalization stats:")
        active_logger.info(
            " - rows: "
            f"total={stats['total_rows']}, non_null={stats['non_null_rows']}, "
            f"canonicalized={stats['rows_canonicalized']}, set_to_none={stats['rows_set_to_none']}"
        )
        active_logger.info(
            " - unique reactions: "
            f"input={stats['unique_input_rxn_smiles']}, valid_format={stats['unique_valid_format_rxn_smiles']}, "
            f"parse_failed={stats['unique_parse_failed_rxn_smiles']}, canonicalized={stats['unique_rxn_canonicalized']}, "
            f"failed_due_to_metabolite={stats['unique_rxn_failed_due_to_metabolite_failure']}, "
            f"failed_during_reassembly={stats['unique_rxn_failed_during_reassembly']}"
        )
        active_logger.info(
            " - unique metabolites: "
            f"total={stats['unique_metabolite_smiles']}, canonicalized={stats['unique_metabolite_canonicalized']}, "
            f"failed={stats['unique_metabolite_failed_canonicalization']}"
        )

        return canonical_series, stats

    def molecule_keys(self, smiles, processes=None):
        """Map each unique SMILES to its molecule_key, in parallel (tautomers are slow)."""
        return molecule_keys(smiles, processes)

    def molecule_key_series(self, smiles, fallback=None):
        """molecule_key per SMILES; a failed key falls back (default: the SMILES) so no row is lost.

        Returns (keys, number of fallbacks).
        """
        # A failed key must not drop the row; the fallback keeps it in its own group.
        smiles = pd.Series(smiles)
        keys = smiles.map(self.molecule_keys(smiles))
        failed = keys.isna() & smiles.notna()
        fallback = smiles if fallback is None else pd.Series(fallback, index=smiles.index)
        return keys.where(~failed, fallback), int(failed.sum())

    def reaction_key(self, rxn_smiles, key_map=None):
        """Equivalence key of a reaction: unmapped, normalized components, spectators dropped, sorted."""
        # Equivalent reactions (maps, charge forms, tautomers, H2O/H+, order) share one key.
        # Orientation and stereo are kept.
        sides = self.split_reaction_smiles(rxn_smiles)
        if None in sides:
            return None
        key_map = key_map if key_map is not None else {}
        out = []
        for mets in sides:
            keys = [key_map[m] if m in key_map else molecule_key(m) for m in mets]
            if None in keys:
                return None
            keys = [k for k in keys if k not in _REACTION_SPECTATORS]
            out.append(".".join(sorted(keys, key=lambda s: (-len(s), s))))
        return ">>".join(out)

    def reaction_key_series(self, rxn_smiles_series):
        """reaction_key for a series, normalizing each unique component once."""
        series = pd.Series(rxn_smiles_series)
        unique = series.dropna().astype(str).str.strip().unique()
        splits = [self.split_reaction_smiles(r) for r in unique]
        mets = [m for sides in splits if None not in sides for s in sides for m in s]
        key_map = self.molecule_keys(mets)
        rxn_map = {r: self.reaction_key(r, key_map) for r in unique}
        return series.map(lambda r: rxn_map.get(r.strip()) if isinstance(r, str) else None)

    def reaction_key_series_with_fallback(self, rxn_smiles_series, logger=None):
        """reaction_key per row; a failed key falls back to the rxn_smiles itself."""
        # A failed key must not drop the row; fall back to the canonical rxn_smiles.
        series = pd.Series(rxn_smiles_series)
        keys = self.reaction_key_series(series)
        failed = keys.isna() & series.notna()
        (logger or self.logger).info(f"rxn_key fell back to rxn_smiles for {int(failed.sum())} rows.")
        return keys.where(~failed, series)

    def get_mol(self, id, url, mol_path):
        """
        Download a molecule file from a URL and save to a path.

        Args:
            id (str): Identifier for the molecule.
            url (str): The URL to download the molecule file from.
            mol_path (Path): The local file path to save the molecule file.

        Returns:
            None
        """
        if mol_path.exists():
            return
        for _ in range(2):  # Retry up to 2 times
            try:
                response = requests.get(url, timeout=60)
                if response.status_code == 200:
                    if len(response.content) > 0:
                        with open(mol_path, "wb") as f:
                            f.write(response.content)
                    else:
                        self.logger.warning(f"{id} - No mol file. Skipping...")
                    break
                self.logger.warning(f"{id} - HTTP {response.status_code}")
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"{id} - Error occurred: {e}")
            time.sleep(5)

    def get_mol_chebi_parallel(self, args):
        """
        Download ChEBI mol files in parallel.

        Args:
            args (list): List of ChEBI IDs.

        Returns:
            None
        """

        def get_mol_chebi(id):
            url = self.CHEBI_MOL_URL.format(id=id[6:])  # Remove "CHEBI:" prefix
            self.get_mol(id, url, self.MOL_PATH["2D"] / f"{id}.mol")

        with progress(
            total=len(args), desc="Downloading Mol files from ChEBI", leave=False
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(get_mol_chebi, args):
                pbar.update()

    def get_mol_kegg(self, id, mol_path):
        """
        Download a KEGG mol file and save to a path.

        Args:
            id (str): KEGG compound ID.
            mol_path (Path): Directory to save the mol file.

        Returns:
            None
        """
        url = self.KEGG_MOL_URL.format(id=id)
        self.get_mol(id, url, mol_path / f"{id}.mol")

    def get_mol_kegg_parallel(self, args):
        """
        Download KEGG mol files in parallel.

        Args:
            args (list): List of KEGG IDs.

        Returns:
            None
        """
        with progress(total=len(args)) as pbar:
            for _ in ThreadPool(5).imap_unordered(self.get_mol_kegg, args):
                pbar.update()

    def get_pubchem_compound(self, cids):
        """
        Fetch compound properties from PubChem for given CIDs.

        Args:
            cids (list): List of PubChem compound IDs.

        Returns:
            dict: Mapping of CID to properties (formula, smiles).
        """
        url = self.PUBCHEM_COMPOUND_URL
        response = requests.post(url, data={"cid": ",".join(str(cid) for cid in cids)})
        if response.status_code != 200:
            self.logger.error(f"Failed to fetch PubChem compound data for CIDs: {cids}")
            return {}
        response_data = response.json()["PropertyTable"]["Properties"]
        result = {}
        for dic in response_data:
            if "CID" in dic:
                result[str(dic.get("CID"))] = {
                    "formula": dic.get("MolecularFormula"),
                    "smiles": dic.get("SMILES"),
                }
        return result

    def get_pubchem_substance(self, sids):
        """
        Fetch substance information from PubChem for given SIDs.

        Args:
            sids (list): List of PubChem substance IDs.

        Returns:
            dict: Mapping of SID to associated CIDs.
        """
        url = self.PUBCHEM_SUBSTANCE_URL
        response = requests.post(url, data={"sid": ",".join(str(sid) for sid in sids)})
        if response.status_code != 200:
            self.logger.error(
                f"Failed to fetch PubChem substance data for SIDs: {sids}"
            )
            return {}
        response_data = response.json()["InformationList"]["Information"]
        result = {}
        for dic in response_data:
            if "SID" in dic:
                result[str(dic.get("SID"))] = {
                    "CID": dic.get("CID", []),
                }
        return result

    def _download_file(self, url, path):
        """Download url to path unless it exists. Resumes a partial download."""
        import subprocess

        if path.exists():
            return
        self.logger.info(f"Downloading {url} to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        part_path = path.with_name(path.name + ".part")
        # A stalled connection times out and retries; -C - resumes from the .part file.
        subprocess.run(
            ["curl", "-L", "-C", "-", "--connect-timeout", "30", "--retry", "10",
             "--retry-all-errors", "-o", str(part_path), url],
            check=True,
        )
        os.replace(part_path, path)

    @staticmethod
    def _scan_pubchem_synonyms(gz_path, keys):
        """
        Stream the synonym file once and collect the CIDs of the wanted names.

        Args:
            gz_path (Path): CID-Synonym-filtered.gz (lines "CID<TAB>synonym").
            keys (set[str]): Lower-case, stripped names to find.

        Returns:
            dict: name -> set of CIDs that list the name as a synonym.
        """
        wanted = {k.encode() for k in keys}
        found = {}
        with gzip.open(gz_path, "rb") as f:
            for line in f:
                cid, _, synonym = line.partition(b"\t")
                synonym = synonym.strip().lower()
                if synonym in wanted:
                    found.setdefault(synonym, set()).add(int(cid))
        return {k.decode(): v for k, v in found.items()}

    def get_pubchem_cids_from_name_parallel(self, compound_names):
        """
        Look up PubChem CIDs for compound names in PubChem's synonym file.

        Names match case-insensitively after stripping. A name with several CIDs
        gets the lowest CID, which is usually the parent compound. Results are cached by name, so a later
        call scans the file only for names not seen before.

        Args:
            compound_names (list): Compound names to look up.

        Returns:
            list: Dictionaries {"compound_name", "cid"}; cid is None for misses.
        """
        cache_path = self.PUBCHEM_NAME_CID_CACHE_FILE_PATH
        cache = self.load_json(cache_path) if cache_path.exists() else {}
        keys = {name.strip().lower() for name in compound_names}
        new_keys = keys - cache.keys()
        multi = 0
        if new_keys:
            self._download_file(self.PUBCHEM_SYNONYM_URL, self.PUBCHEM_SYNONYM_GZ_FILE_PATH)
            self.logger.info(f"Scanning PubChem synonyms for {len(new_keys)} names...")
            found = self._scan_pubchem_synonyms(self.PUBCHEM_SYNONYM_GZ_FILE_PATH, new_keys)
            multi = sum(len(cids) > 1 for cids in found.values())
            cache.update({k: None for k in new_keys})
            cache.update({k: str(min(cids)) for k, cids in found.items()})
            self.save_json(data=cache, file_path=cache_path)
        n_found = sum(cache[k] is not None for k in keys)
        self.logger.info(
            f"PubChem names: {len(keys)} requested, {n_found} found, "
            f"{multi} with several CIDs (among newly scanned names)"
        )
        return [
            {"compound_name": name, "cid": cache[name.strip().lower()]}
            for name in compound_names
        ]

    @staticmethod
    def _scan_pubchem_smiles(gz_path, cids):
        """
        Stream CID-SMILES.gz once and collect the SMILES of the wanted CIDs.

        Args:
            gz_path (Path): CID-SMILES.gz (lines "CID<TAB>SMILES").
            cids (set[str]): CIDs to find.

        Returns:
            dict: cid -> SMILES for the CIDs found in the file.
        """
        wanted = {c.encode() for c in cids}
        found = {}
        with gzip.open(gz_path, "rb") as f:
            for line in f:
                cid, _, smiles = line.partition(b"\t")
                if cid in wanted:
                    found[cid.decode()] = smiles.strip().decode()
                    if len(found) == len(wanted):
                        break
        return found

    def get_pubchem_smiles_from_cids(self, cids):
        """
        Look up SMILES for PubChem CIDs in PubChem's CID-SMILES file.

        Results are cached by CID, so a later call scans the file only for CIDs not
        seen before. The file is downloaded only when it is missing and needed.

        Args:
            cids (list): PubChem compound IDs.

        Returns:
            dict: {cid (str): SMILES or None for CIDs missing from the file}.
        """
        cache_path = self.PUBCHEM_CID_SMILES_CACHE_FILE_PATH
        cache = self.load_json(cache_path) if cache_path.exists() else {}
        keys = {str(c) for c in cids}
        new_keys = keys - cache.keys()
        if new_keys:
            gz_path = self.PUBCHEM_CID_SMILES_GZ_FILE_PATH
            self._download_file(self.PUBCHEM_CID_SMILES_URL, gz_path)
            self.logger.info(f"Scanning PubChem CID-SMILES for {len(new_keys)} CIDs...")
            cache.update({k: None for k in new_keys})
            cache.update(self._scan_pubchem_smiles(gz_path, new_keys))
            self.save_json(data=cache, file_path=cache_path)
        n_found = sum(cache[k] is not None for k in keys)
        self.logger.info(
            f"PubChem CIDs: {len(keys)} requested, {n_found} found, "
            f"{len(keys) - n_found} missing"
        )
        return {k: cache[k] for k in keys}

    def inchi_to_smiles(self, inchi):
        """
        Convert InChI string to SMILES string.

        Args:
            inchi (str): InChI string.

        Returns:
            str: SMILES string or None if conversion fails.
        """
        mol = MolFromInchi(inchi)
        if mol:
            smiles = MolToSmiles(mol, canonical=True)
            try:
                standardized_smiles = standardize_smiles(smiles)
                return standardized_smiles
            except:
                return smiles
        else:
            self.logger.warning(f"Failed to generate mol from InChI: {inchi}")
            return None

    def download_pubchem_sdfs_from_cids_parallel(self, df):
        """
        Download SDF files from PubChem for a list of CIDs in parallel.

        Args:
            df (pd.DataFrame): DataFrame with columns ['smiles_hash', 'smiles', 'CID']

        Returns:
            None
        """

        def download_sdf_from_cid(row):
            cid = row["CID"]
            smiles_hash = row["smiles_hash"]
            sdf_file_path = self.SDF_PATH / f"pubchem_{smiles_hash}.sdf"

            # Skip if already downloaded
            if sdf_file_path.exists():
                self.logger.info(
                    f"SDF file already exists for CID {cid} ({smiles_hash})"
                )
                return

            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/SDF?cid={cid}&record_type=3d"
            for _ in range(2):  # Retry up to 2 times
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        try:
                            # Save the SDF file with pubchem_{smiles_hash} as filename
                            with open(sdf_file_path, "w") as f:
                                f.write(response.text)
                            time.sleep(0.2)
                            return
                        except Exception as e:
                            self.logger.error(f"Failed to save SDF for CID {cid}: {e}")
                            return
                    else:
                        self.logger.warning(
                            f"Failed to download SDF for CID {cid}. Status code: {response.status_code}"
                        )
                        return
                except Exception as e:
                    time.sleep(2)
                    self.logger.error(
                        f"Error downloading SDF for CID {cid}: {e}. Retrying..."
                    )
            self.logger.error(
                f"Failed to download SDF for CID {cid} after multiple attempts."
            )
            return

        # Filter out rows where SDF already exists
        existing_sdf_files = set([f.name for f in self.SDF_PATH.glob("*.sdf")])
        df_filtered = df[
            ~(
                df["smiles_hash"].apply(
                    lambda h: f"pubchem_{h}.sdf" in existing_sdf_files
                )
            )
        ].copy()

        if len(df_filtered) == 0:
            self.logger.info("All SDF files already downloaded. Exiting...")
            return

        self.logger.info(f"Downloading SDF files for {len(df_filtered)} CIDs")

        start_time = time.time()
        last_check_time = time.time()

        with progress(
            total=len(df_filtered),
            desc="Downloading SDF files from PubChem",
            leave=False,
        ) as pbar:
            for _ in ThreadPool(4).imap_unordered(
                lambda row: download_sdf_from_cid(row),
                [row for _, row in df_filtered.iterrows()],
            ):
                pbar.update()

                # Check time every 10 minutes
                current_time = time.time()
                if current_time - last_check_time >= 10 * 60:  # 10 minutes
                    last_check_time = current_time
                    elapsed = current_time - start_time
                    if elapsed >= 6 * 3600:  # 6 hours in seconds
                        self.logger.info(
                            f"6 hours of downloading completed. Taking 30 minute break..."
                        )
                        time.sleep(30 * 60)  # 30 minutes
                        start_time = time.time()  # Reset timer
                        self.logger.info(
                            "30 minute break completed. Resuming downloads..."
                        )

        self.logger.info("SDF download completed.")

    def smiles_to_mol(self, smiles_dict):
        """
        Convert SMILES strings to mol files.

        Args:
            smiles_dict (dict): Mapping of IDs to SMILES strings.

        Returns:
            None
        """
        for id, smiles in progress(
            smiles_dict.items(), desc="Converting SMILES to mol files", leave=False
        ):
            if not (self.MOL_PATH["2D"] / f"{id}.mol").exists():
                mol = MolFromSmiles(smiles)
                if mol:
                    MolToMolFile(mol, self.MOL_PATH["2D"] / f"{id}.mol")
                else:
                    self.logger.warning(f"Failed to generate mol from SMILES for {id}")

    def mol_to_smiles(self, mol_filepath):
        """
        Convert a mol file to SMILES string.

        Args:
            mol_filepath (Path): Path to the mol file.

        Returns:
            str: SMILES string or None if conversion fails.
        """
        mol_filepath = Path(mol_filepath)
        if not mol_filepath.exists():
            self.logger.warning(f"Mol file does not exist: {mol_filepath}")
            return None
        try:
            mol = MolFromMolFile(str(mol_filepath))
            if mol:
                smiles = MolToSmiles(mol, canonical=True)
                try:
                    standardized_smiles = standardize_smiles(smiles)
                    return standardized_smiles
                except:
                    return smiles
            else:
                self.logger.warning(f"Failed to parse mol file: {mol_filepath}")
                return None
        except Exception as e:
            self.logger.error(f"Error converting mol to SMILES for {mol_filepath}: {e}")
            return None

    def parse_equation(self, equation, pattern):
        """
        Parse a reaction equation into substrates and products.

        Args:
            equation (str): The reaction equation string.
            pattern (str): Regex pattern to extract compound IDs.

        Returns:
            tuple: (substrates, direction, products)
        """

        def parse_helper(eqn, pattern):
            eqn = eqn.split(" + ")
            eqn = [
                (
                    re.findall(r"(.*?)" + pattern, elem)[0].strip() or "1",
                    re.findall(pattern, elem)[0],
                )
                for elem in eqn
            ]
            return eqn

        direction = "<=>" if "<=>" in equation else "=>"
        equation = re.sub(r" <=> | => ", " = ", equation)
        substrates, products = equation.split(" = ")
        substrates = parse_helper(substrates, pattern)
        products = parse_helper(products, pattern)
        return substrates, direction, products

    def normalize_ec_token(
        self,
        token,
        *,
        fallback: str | None = None,
        pad_partial: bool = True,
        levels: int = 4,
    ) -> str | None:
        """Normalize one EC token by cleaning characters and optionally padding levels."""
        if token is None or (isinstance(token, float) and np.isnan(token)):
            return fallback

        text = str(token).strip()
        if not text:
            return fallback

        parts = [part.strip() for part in text.split(".") if str(part).strip()]
        if not parts:
            return fallback

        cleaned_parts = []
        for part in parts[:levels]:
            cleaned = re.sub(r"[^A-Za-z0-9-]", "", part)
            cleaned_parts.append(cleaned if cleaned else "-")

        # EC component 4 should be numeric when present; normalize odd suffixes/tokens to '-'.
        if len(cleaned_parts) >= 4:
            fourth_part = cleaned_parts[3]
            if fourth_part != "-" and not fourth_part.isdigit():
                cleaned_parts[3] = "-"

        if pad_partial and len(cleaned_parts) < levels:
            cleaned_parts.extend(["-"] * (levels - len(cleaned_parts)))

        normalized = ".".join(cleaned_parts)
        if not normalized:
            return fallback
        return normalized

    def normalize_ec_collection(
        self,
        value,
        *,
        fallback: str | None = None,
        pad_partial: bool = True,
        levels: int = 4,
        deduplicate: bool = True,
    ) -> list[str | None]:
        """Explode mixed EC strings and normalize each token without dropping rows."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return [fallback]

        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            raw_values = list(value)
        else:
            raw_values = [value]

        tokens: list[str] = []
        for raw in raw_values:
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue

            text = str(raw).strip()
            if not text:
                continue

            # Capture EC-like candidates (1 to 4 dot-separated components) from noisy strings.
            candidates = re.findall(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+){0,3}", text)
            if candidates:
                tokens.extend(candidates)
            else:
                tokens.append(text)

        if not tokens:
            return [fallback]

        normalized = [
            self.normalize_ec_token(
                token,
                fallback=fallback,
                pad_partial=pad_partial,
                levels=levels,
            )
            for token in tokens
        ]
        normalized = [
            token for token in normalized if token is not None or fallback is None
        ]

        if not normalized:
            return [fallback]

        if deduplicate:
            return list(dict.fromkeys(normalized))
        return normalized

    def generate_reaction_SMILES(self, row, smiles_dict):
        """
        Generate a reaction SMILES string from row data and SMILES dictionary.

        Args:
            row (dict): Contains 'substrates' and 'products' lists.
            smiles_dict (dict): Mapping of compound IDs to SMILES.

        Returns:
            str or None: Reaction SMILES string or None if missing data.
        """
        substrate_SMILES = []
        product_SMILES = []
        for sub in row["substrates"]:
            try:
                smiles = smiles_dict[sub[-1]]
            except Exception as e:
                self.logger.warning(f"Missing SMILES for substrate {sub[-1]}: {e}")
                return None
            substrate_SMILES.append(smiles)
        for prod in row["products"]:
            try:
                smiles = smiles_dict[prod[-1]]
            except Exception as e:
                self.logger.warning(f"Missing SMILES for product {prod[-1]}: {e}")
                return None
            product_SMILES.append(smiles)
        if None in substrate_SMILES or None in product_SMILES:
            return None
        return (
            ".".join(sub for sub in substrate_SMILES)
            + ">>"
            + ".".join(prod for prod in product_SMILES)
        )

    def load_json(self, file_path):
        """
        Load JSON data from a file.

        Args:
            file_path (Path or str): Path to the JSON file.

        Returns:
            dict or None: Loaded JSON data, or None on failure.
        """
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None

    def save_json(self, data, file_path):
        """
        Save data as JSON to a file.

        Args:
            data (dict): Data to save.
            file_path (Path or str): Path to the output JSON file.

        Returns:
            None
        """
        try:
            tmp_path = f"{file_path}.{os.getpid()}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, file_path)
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")

    def _search_uniprot_all(self, query, fields, wait=1):
        """
        Fetch every hit of a UniProt search, following the Link rel="next" pages.

        Returns:
            list | None: Result dicts (empty for no hits), or None if a request failed.
        """
        url, params, results = self.UNIPROT_API_URL, {"query": query, "fields": fields, "size": 500}, []
        while url:
            for attempt in range(UNIPROT_RETRIES):
                try:
                    response = requests.get(url=url, params=params, timeout=UNIPROT_TIMEOUT)
                    if response.status_code == 200:
                        break
                    error = f"status code {response.status_code}"
                except requests.RequestException as e:
                    error = str(e)
                time.sleep(wait * 2**attempt)
            else:
                self.logger.warning(f"UniProt request failed for {query[:200]}: {error}")
                return None
            results += response.json()["results"]
            url, params = response.links.get("next", {}).get("url"), None
        return results

    def _search_uniprot_batch(self, accs, fields, wait=1):
        """
        Fetch one batch of accessions with a single OR search.

        Returns:
            dict | None: Mapping of primaryAccession to result, or None if the request failed.
            A malformed accession is dropped and the rest of the batch is still fetched.
            `wait` scales the pause between attempts.
        """
        params = {
            "query": " OR ".join(f"accession:{acc}" for acc in accs),
            "fields": fields,
            "size": 500,
        }
        for attempt in range(UNIPROT_RETRIES):
            try:
                response = requests.get(
                    url=self.UNIPROT_API_URL, params=params, timeout=UNIPROT_TIMEOUT
                )
                if response.status_code == 200:
                    found = {r["primaryAccession"]: r for r in response.json()["results"]}
                    missing = [acc for acc in accs if acc not in found]
                    if missing:
                        self.logger.warning(
                            f"No UniProt data found for {len(missing)} of {len(accs)} accessions."
                        )
                        self.logger.debug(f"Missing UniProt accessions: {missing}")
                    return found
                error = f"status code {response.status_code}"
                if response.status_code == 400:
                    # One malformed accession fails the whole query; isolate it by bisection.
                    if len(accs) == 1:
                        self.logger.warning(f"UniProt rejected accession {accs[0]}.")
                        return {}
                    mid = len(accs) // 2
                    halves = [
                        self._search_uniprot_batch(h, fields, wait) for h in (accs[:mid], accs[mid:])
                    ]
                    if None in halves:
                        return None
                    return {k: v for half in halves for k, v in half.items()}
            except requests.RequestException as e:
                error = str(e)
            time.sleep(wait * 2**attempt)
        self.logger.warning(
            f"UniProt request failed for a batch of {len(accs)} accessions: {error}"
        )
        return None

    def _fetch_uniprot_entries(self, accs, fields, desc):
        """
        Fetch active UniProt entries in batches, following merged accessions.

        A merged accession maps to the entry of its merge target.

        Returns:
            dict: Mapping of requested accession to its UniProt result. Accessions without
                an active entry are omitted.
        """
        entries, merged = {}, {}

        def pull(items, size, wait, pbar=None):
            batches = [items[i : i + size] for i in range(0, len(items), size)]
            found, failed = {}, []
            with ThreadPool(UNIPROT_THREADS) as pool:
                results = pool.imap(lambda b: self._search_uniprot_batch(b, fields, wait), batches)
                for batch, result in zip(batches, results):
                    if result is None:
                        failed += batch
                    else:
                        found |= result
                    if pbar:
                        pbar.update(len(batch))
            return found, failed

        def fetch(items, pbar=None):
            found, failed = pull(items, UNIPROT_BATCH, 1, pbar)
            if failed:
                self.logger.info(f"Retrying {len(failed)} failed UniProt accessions.")
                retried, failed_again = pull(failed, UNIPROT_RETRY_BATCH, UNIPROT_RETRY_WAIT)
                found |= retried
                self.logger.info(
                    f"UniProt retry recovered {len(failed) - len(failed_again)} accessions; "
                    f"{len(failed_again)} still failed."
                )
            return found

        with progress(total=len(accs), desc=desc, leave=False) as pbar:
            found = fetch(list(accs), pbar)
        for acc, data in found.items():
            if data.get("entryType") != "Inactive":
                entries[acc] = data
                continue
            reason = data.get("inactiveReason")
            if isinstance(reason, dict) and reason.get("inactiveReasonType") == "MERGED":
                new_acc = (reason.get("mergeDemergeTo") or [None])[0]
                if new_acc:
                    merged[acc] = new_acc
                    continue
                self.logger.warning(f"Inactive UniProt entry for {acc} with no merge target.")
            else:
                self.logger.warning(f"Inactive UniProt entry for {acc}.")

        found = fetch(sorted(set(merged.values())))
        for acc, new_acc in merged.items():
            data = found.get(new_acc)
            if data and data.get("entryType") != "Inactive":
                entries[acc] = data
                self.logger.info(f"UniProt entry {acc} merged to {new_acc}.")
        return entries

    def _uniprot_acc_record(self, acc, data):
        """Build the cached sequence/EC/PDB dict for one UniProt result."""
        try:
            desc = data["proteinDescription"]
            names = [desc, *desc.get("includes", []), *desc.get("contains", [])]
            ecs = list(
                dict.fromkeys(
                    elem["value"]
                    for item in names
                    for elem in (
                        *item.get("recommendedName", {}).get("ecNumbers", []),
                        *(
                            e
                            for alt in item.get("alternativeNames", [])
                            for e in alt.get("ecNumbers", [])
                        ),
                    )
                )
            )
        except Exception as e:
            ecs = None
            self.logger.warning(f"Failed to extract EC numbers for {acc}: {e}")
        try:
            pdbs = [i["id"] for i in data["uniProtKBCrossReferences"] if i["database"] == "PDB"]
        except Exception as e:
            pdbs = None
            self.logger.warning(f"Failed to extract PDBs for {acc}: {e}")
        try:
            sequence = data["sequence"]["value"]
        except Exception as e:
            sequence = None
            self.logger.warning(f"Failed to extract sequence for {acc}: {e}")
        return {"acc_id": acc, "ecs": ecs, "pdbs": pdbs, "sequence": sequence}

    def get_uniprot_acc_json(self, accs, verbose=False):
        """
        Download UniProt data for a list of accession IDs.

        Args:
            accs (list): List of UniProt accession IDs.
            verbose (bool): Unused; kept for API compatibility.

        Returns:
            dict: Mapping of accession IDs to UniProt data. Accessions with no data are omitted.
        """
        results = {}
        to_fetch = []
        for acc in accs:
            json_file_path = self.UNIPROT_ACC_PATH / f"{acc}.json"
            if json_file_path.exists():
                results[acc] = self.load_json(json_file_path)
            else:
                to_fetch.append(acc)

        fields = "accession,ec,sequence,structure_3d"
        entries = self._fetch_uniprot_entries(
            to_fetch, fields, "Downloading protein data from UniProt"
        )
        for acc, data in entries.items():
            json_data = self._uniprot_acc_record(acc, data)
            self.save_json(json_data, self.UNIPROT_ACC_PATH / f"{acc}.json")
            results[acc] = json_data
        return results

    def get_uniprot_ec_acc_map(self, ecs, verbose=False):
        """
        Map EC numbers to UniProt accession IDs.

        Only full four-level ECs are looked up. A partial EC such as "2.1.1.-" means no EC
        is assigned, not that every enzyme of the class applies, so it maps to None.

        Args:
            ecs (list): List of EC numbers.
            verbose (bool): If True, log warnings.

        Returns:
            dict: Mapping of EC numbers to lists of accession IDs.
        """
        results = {}
        parts = [str(ec).split(".") for ec in ecs]
        results.update({ec: None for ec, p in zip(ecs, parts) if len(p) != 4 or "-" in p})
        ecs = [ec for ec in ecs if ec not in results]

        def process_ec(ec, wait=1):
            """Return True when a UniProt request failed."""
            json_file_path = self.UNIPROT_EC_ACC_PATH / f"{ec}.json"
            if json_file_path.exists():
                results[ec] = self.load_json(json_file_path)
                return

            for reviewed in ("true", "false"):
                response = self._search_uniprot_all(
                    f"ec:{ec} AND active:true AND reviewed:{reviewed}", "accession", wait
                )
                if response is None:
                    results[ec] = None
                    self.logger.error(f"UniProt request failed for EC {ec}; not cached")
                    return True
                if response:
                    break
                if reviewed == "true":
                    self.logger.warning(
                        f"No reviewed UniProt acc_ids found for EC {ec}. Trying unreviewed entries."
                    )

            if not response:
                results[ec] = None
                self.logger.error(f"No UniProt acc_ids found for EC {ec}")
                self.save_json(data=None, file_path=json_file_path)
                return

            acc_ids = [data["primaryAccession"] for data in response]
            self.save_json(acc_ids, json_file_path)
            results[ec] = acc_ids

        with progress(
            total=len(ecs),
            desc="Downloading ec - acc_ids maps from UniProt",
            leave=False,
        ) as pbar:
            failed = []
            for ec, request_failed in zip(ecs, ThreadPool(20).imap(process_ec, ecs)):
                failed += [ec] if request_failed else []
                pbar.update()
        if failed:
            self.logger.info(f"Retrying {len(failed)} failed UniProt EC lookups.")
            with ThreadPool(UNIPROT_THREADS) as pool:
                still_failed = sum(
                    map(bool, pool.starmap(process_ec, [(ec, UNIPROT_RETRY_WAIT) for ec in failed]))
                )
            self.logger.info(
                f"UniProt retry recovered {len(failed) - still_failed} EC lookups; "
                f"{still_failed} still failed."
            )
        return results

    def get_active_acc_id_from_ec_and_org_parallel(self, ec_org_pairs):
        """
        Fetch all active UniProt accession IDs for (EC number, organism) pairs.

        Pairs are grouped by EC and searched in chunks of organisms. A hit belongs to a pair
        when the pair's organism name occurs as a whole-word phrase in the hit's scientific
        name, common name or synonyms, which is how UniProt's organism_name search matches.
        Each pair is cached with every hit, or an empty list when it has none. A failed
        search caches nothing for its pairs.

        Args:
            ec_org_pairs (list of tuples): List of (EC number, organism name) pairs.

        Returns:
            pd.DataFrame: One row per pair and hit with columns reviewed, UniprotID, ECNumber,
                Organism, protein_existence (level 1 to 5), annotation_score, genes (primary,
                synonym, ORF and ordered-locus names), pmids (UniProt PubMed IDs) and sequence.
        """
        records = []
        orgs_by_ec = {}
        for ec, org in dict.fromkeys(map(tuple, ec_org_pairs)):
            json_path = self.UNIPROT_EC_ORG_ACC_PATH / f"{ec} | {org}.json"
            if json_path.exists():
                records += self.load_json(json_path)
            else:
                orgs_by_ec.setdefault(ec, []).append(org)

        def phrase(name):
            return " " + " ".join(re.findall(r"[a-z0-9]+", name.lower())) + " "

        def search(task):
            ec, orgs = task
            query = " OR ".join(f'organism_name:"{org}"' for org in orgs)
            hits = self._search_uniprot_all(
                f"ec:{ec} AND active:true AND ({query})", UNIPROT_EC_ORG_FIELDS
            )
            if hits is None:
                return []
            names = []
            for hit in hits:
                organism = hit["organism"]
                names.append(
                    " ".join(
                        phrase(n)
                        for n in (
                            organism["scientificName"],
                            organism.get("commonName", ""),
                            *organism.get("synonyms", []),
                        )
                    )
                )
            found = []
            for org in orgs:
                pair_records = [
                    {
                        "reviewed": "unreviewed" not in hit["entryType"],
                        "UniprotID": hit["primaryAccession"],
                        "ECNumber": ec,
                        "Organism": org,
                        "protein_existence": int(hit["proteinExistence"][0]),
                        "annotation_score": hit["annotationScore"],
                        "genes": [
                            name["value"]
                            for gene in hit.get("genes", [])
                            for name in (
                                *([gene["geneName"]] if "geneName" in gene else []),
                                *gene.get("synonyms", []),
                                *gene.get("orfNames", []),
                                *gene.get("orderedLocusNames", []),
                            )
                        ],
                        "pmids": [
                            xref["id"]
                            for ref in hit.get("references", [])
                            for xref in ref["citation"].get("citationCrossReferences", [])
                            if xref["database"] == "PubMed"
                        ],
                        "sequence": hit["sequence"]["value"],
                    }
                    for hit, hay in zip(hits, names)
                    if phrase(org) in hay
                ]
                self.save_json(pair_records, self.UNIPROT_EC_ORG_ACC_PATH / f"{ec} | {org}.json")
                found += pair_records
            return found

        tasks = [
            (ec, orgs[i : i + UNIPROT_ORG_CHUNK])
            for ec, orgs in orgs_by_ec.items()
            for i in range(0, len(orgs), UNIPROT_ORG_CHUNK)
        ]
        self.logger.info("Fetching active accession IDs from UniProt...")
        with progress(total=len(tasks), desc="Fetching active accession IDs") as pbar:
            with ThreadPool(UNIPROT_THREADS) as pool:
                for found in pool.imap_unordered(search, tasks):
                    records += found
                    pbar.update()
        self.logger.info("Accession IDs fetching completed.")
        return pd.DataFrame(records, columns=UNIPROT_EC_ORG_COLUMNS)

    def get_uniprot_time_json(self, accs, verbose=False):
        """
        Download UniProt time information for a list of accession IDs.

        Args:
            accs (list): List of UniProt accession IDs.
            verbose (bool): Unused; kept for API compatibility.

        Returns:
            dict: Mapping of accession IDs to UniProt time data, None where UniProt has no data.
        """
        results = {}
        to_fetch = []
        for acc in accs:
            json_file_path = self.UNIPROT_TIME_PATH / f"{acc}.json"
            if json_file_path.exists():
                results[acc] = self.load_json(json_file_path)
            else:
                to_fetch.append(acc)

        fields = "date_created,date_modified,date_sequence_modified,version"
        entries = self._fetch_uniprot_entries(
            to_fetch, fields, "Downloading time data from UniProt"
        )
        for acc in to_fetch:
            if acc not in entries:
                results[acc] = None
                continue
            json_data = self._uniprot_time_record(acc, entries[acc])
            self.save_json(json_data, self.UNIPROT_TIME_PATH / f"{acc}.json")
            results[acc] = json_data
        return results

    def _uniprot_time_record(self, acc, data):
        """Build the cached time dict for one UniProt result."""
        audit_keys = {
            "date_created": "firstPublicDate",
            "date_modified": "lastAnnotationUpdateDate",
            "date_sequence_modified": "lastSequenceUpdateDate",
            "version": "entryVersion",
        }
        record = {"acc_id": acc}
        for name, key in audit_keys.items():
            try:
                record[name] = data["entryAudit"][key]
            except Exception as e:
                record[name] = None
                self.logger.warning(f"Failed to extract {name} for {acc}: {e}")
        return record

    def normalize_uniprot_acc_id(self, acc_id, delimiter="|"):
        if isinstance(
            acc_id, (list, tuple, set, frozenset, np.ndarray, pd.Series, dict)
        ):
            return None

        if acc_id is None or pd.isna(acc_id):
            return None

        acc_id = str(acc_id).strip()
        if not acc_id:
            return None

        if delimiter and delimiter in acc_id:
            acc_id = acc_id.split(delimiter, 1)[0].strip()

        return acc_id or None

    def add_uniprot_date_column(
        self,
        df,
        acc_id_column="uniprot_id",
        output_column="uniprot_date",
        date_field="date_created",
        delimiter="|",
        verbose=False,
    ):
        if acc_id_column not in df.columns:
            raise KeyError(
                f"Column '{acc_id_column}' not found in DataFrame columns: {list(df.columns)}"
            )

        df = df.copy()
        normalized_acc_ids = df[acc_id_column].apply(
            lambda acc_id: self.normalize_uniprot_acc_id(acc_id, delimiter=delimiter)
        )

        if output_column in df.columns:
            existing_dates = pd.to_datetime(df[output_column], errors="coerce")
        else:
            existing_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

        acc_ids_to_fetch = sorted(
            normalized_acc_ids.loc[existing_dates.isna()].dropna().unique().tolist()
        )
        date_map = {}
        if acc_ids_to_fetch:
            self.logger.info(
                f"Fetching UniProt {date_field} timestamps for {len(acc_ids_to_fetch)} unique accession IDs..."
            )
            time_payload = self.get_uniprot_time_json(acc_ids_to_fetch, verbose=verbose)
            date_map = {
                acc_id: pd.to_datetime(
                    (time_payload.get(acc_id) or {}).get(date_field), errors="coerce"
                )
                for acc_id in acc_ids_to_fetch
            }

        mapped_dates = pd.to_datetime(normalized_acc_ids.map(date_map), errors="coerce")
        df[output_column] = existing_dates.where(existing_dates.notna(), mapped_dates)
        df[output_column] = pd.to_datetime(df[output_column], errors="coerce").dt.normalize()

        self.logger.info(
            f"Annotated {df[output_column].notna().sum()} / {len(df)} rows with '{output_column}'."
        )
        return df

    def download_af_archive(self, url, tar):
        """Download `url` to `tar`, resuming a partial file. Raises on failure."""
        if shutil.which("aria2c"):
            subprocess.run(
                ["aria2c", "-c", "-x16", "-s16", "-k1M", "--file-allocation=none",
                 "-d", str(tar.parent), "-o", tar.name, url],
                check=True,
            )
            return
        size = tar.stat().st_size if tar.exists() else 0
        headers = {**HTTP_HEADERS, "Range": f"bytes={size}-"}
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tar, "ab" if r.status_code == 206 else "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    def extract_af_bulk(self, ids, version):
        """
        Extract AlphaFold models from the EBI Swiss-Prot archive into AF_PDB_PATH.

        The archive is downloaded and indexed once; a lock file serializes
        concurrent processes. Returns the set of accessions extracted.
        """
        tar = self.AF_BULK_PATH / f"swissprot_pdb_v{version}.tar"
        index_path = Path(f"{tar}.index.json")
        control = Path(f"{tar}.aria2")
        with open(f"{tar}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not index_path.exists():
                if not tar.exists() or control.exists():
                    try:
                        self.download_af_archive(
                            self.ALPHAFOLD_SWISSPROT_URL.format(version=version), tar
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"AlphaFold Swiss-Prot archive download failed: {e}"
                        )
                        return set()
                index = {}
                with tarfile.open(tar, "r:") as archive:
                    for m in archive:
                        if match := re.fullmatch(
                            r"AF-(.+)-F1-model_v\d+\.pdb\.gz", Path(m.name).name
                        ):
                            index[match.group(1)] = [m.offset_data, m.size]
                tmp = Path(f"{index_path}.tmp")
                tmp.write_text(json.dumps(index))
                os.replace(tmp, index_path)
        index = json.loads(index_path.read_text())
        found = sorted((a for a in ids if a in index), key=lambda a: index[a][0])
        with open(tar, "rb") as f:
            for acc in found:
                offset, size = index[acc]
                f.seek(offset)
                dest = self.AF_PDB_PATH / f"AF-{acc}-F1-model_v{version}.pdb"
                tmp = Path(f"{dest}.{os.getpid()}.tmp")
                tmp.write_bytes(gzip.decompress(f.read(size)))
                os.replace(tmp, dest)
        self.logger.info(
            f"Extracted {len(found)} AlphaFold models from the archive; "
            f"{len(ids) - len(found)} remain for the API."
        )
        return set(found)

    def get_af_pdb(self, args):
        """
        Download AlphaFold PDB files for a list of UniProt IDs.

        Args:
            args (list): List of UniProt IDs.

        Returns:
            None
        """

        def get_canonical_entry(id):
            json_path = self.AF_DOWNLOAD_PATH / f"{id}.json"
            status = self.get_html(
                self.ALPHAFOLD_API_URL.format(id=id), json_path, timeout=30
            )
            if status == 404:
                json_path.write_text("[]")  # AlphaFold has no entry; skip next time
            if not json_path.exists():
                return None
            return next(
                (
                    e
                    for e in self.load_json(json_path)
                    if e["modelEntityId"] == f"AF-{id}-F1"
                ),
                None,
            )

        def download(url):
            path = self.AF_PDB_PATH / url.split("/")[-1]
            self.get_html(url, path, timeout=30)
            return path.exists()

        url_template = version = None

        def get_af_pdb_helper(id):
            json_path = self.AF_DOWNLOAD_PATH / f"{id}.json"
            try:
                ok = (
                    not json_path.exists()
                    and url_template is not None
                    and download(url_template.format(id=id))
                )
                if not ok:
                    entry = get_canonical_entry(id)
                    if entry is None:
                        if not json_path.exists():
                            msg = f"AlphaFold API request failed for {id}."
                        elif not self.load_json(json_path):
                            msg = f"AlphaFold has no entry for {id}."
                        else:
                            msg = f"AlphaFold has no canonical model for {id}."
                        self.logger.warning(msg)
                        return
                    ok = download(entry["pdbUrl"])
                if ok:
                    self.logger.info(f"Downloaded AlphaFold PDB for {id}")
                else:
                    self.logger.warning(f"AlphaFold PDB for {id} was not downloaded.")
            except Exception as e:
                self.logger.error(
                    f"Failed to download AlphaFold PDB for {id} with error {e}"
                )

        def downloaded():
            return {
                m.group(1)
                for f in self.AF_PDB_PATH.glob("*.pdb")
                if (m := re.match(r"AF-([A-Za-z0-9]+)-F1-model_v\d+", f.stem))
            }

        args = set(args) - downloaded()
        # Cached accessions never use the template. An uncached one yields the current
        # file URL pattern and version; other uncached models download directly.
        uncached = [a for a in args if not (self.AF_DOWNLOAD_PATH / f"{a}.json").exists()]
        for id in sorted(uncached):
            if entry := get_canonical_entry(id):
                url_template = entry["pdbUrl"].replace(f"AF-{id}-F1", "AF-{id}-F1")
                version = entry["latestVersion"]
                self.logger.info(f"AlphaFold model version {version}: {url_template}")
                break
        if args and version is not None:
            args -= self.extract_af_bulk(args, version)
        with progress(
            total=len(args), desc="Downloading PDB files from AlphaFold", leave=False
        ) as pbar:
            for _ in ThreadPool(AF_THREADS).imap_unordered(get_af_pdb_helper, args):
                pbar.update()

        have = downloaded()
        failed = [
            id
            for id in args
            if id not in have and not (self.AF_DOWNLOAD_PATH / f"{id}.json").exists()
        ]
        if failed:
            self.logger.info(f"Retrying {len(failed)} failed AlphaFold downloads.")
            time.sleep(AF_RETRY_WAIT)
            with ThreadPool(AF_RETRY_THREADS) as pool:
                pool.map(get_af_pdb_helper, failed)
            still_failed = len(set(failed) - downloaded())
            self.logger.info(
                f"AlphaFold retry recovered {len(failed) - still_failed} accessions; "
                f"{still_failed} still failed."
            )

    def get_ebi_pdb(self, args):
        """
        Download PDB files from EBI for a list of PDB IDs.
        Tries the updated mmCIF first, then the archive mmCIF.

        Args:
            args (list): List of PDB IDs.

        Returns:
            None
        """

        def get_ebi_pdb_helper(id):
            urls_and_paths = [
                (
                    self.EBI_MMCIF_UPDATED_URL.format(id=id.lower()),
                    self.MMCIF_PATH / f"{id.upper()}_updated.cif",
                ),
                (
                    self.EBI_MMCIF_ARCHIVE_URL.format(id=id.lower()),
                    self.MMCIF_PATH / f"{id.upper()}.cif",
                ),
            ]

            for url, path in urls_and_paths:
                self.get_html(url, path, timeout=30)
                if path.exists():
                    return

            self.logger.warning(
                f"Could not download {id} from any source (updated or archive mmCIF)"
            )

        # Collect already downloaded files respecting preference hierarchy
        mmcif_updated = {
            f.stem.replace("_updated", "").upper()
            for f in self.MMCIF_PATH.glob("*_updated.cif")
        }
        mmcif_archive = {
            f.stem.upper()
            for f in self.MMCIF_PATH.glob("*.cif")
            if "_updated" not in f.stem
        }

        # Filter args to only include those that need downloading
        args_to_download = []
        for pdb_id in args:
            pdb_id_upper = pdb_id.upper()
            # Skip if already have any version in preference order
            if pdb_id_upper in mmcif_updated:
                continue
            if pdb_id_upper in mmcif_archive:
                continue
            args_to_download.append(pdb_id)

        if not args_to_download:
            self.logger.info("All requested PDB files already downloaded. Skipping...")
            return

        self.logger.info(f"Downloading {len(args_to_download)} PDB files from EBI")

        with progress(
            total=len(args_to_download), desc="Downloading PDB files from EBI"
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(
                get_ebi_pdb_helper, args_to_download
            ):
                pbar.update()

    def download_pubmed_abstracts_parallel(self, ref_pmids):
        """
        Download PubMed abstracts in parallel for a list of PMIDs.

        Args:
            ref_pmids (list): List of PubMed IDs.

        Returns:
            None
        """

        def download_pubmed_abstracts(ids):
            filepath = self.REFERENCE_DOWNLOAD_PATH / f"{ids[0]} - {ids[-1]}.xml"
            if filepath.exists():
                self.logger.info(f"{filepath} already exists. Skipping download...")
                return
            EMAIL = os.getenv("PUBMED_EMAIL", None)
            API_KEY = os.getenv("PUBMED_API_KEY", None)
            if not EMAIL or not API_KEY:
                self.logger.error(
                    "PUBMED_EMAIL and PUBMED_API_KEY must be set in the environment variables."
                )
                return
            url = self.PUBMED_ENTREZ_EFETCH_URL
            params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
                "email": EMAIL,
                "api_key": API_KEY,
            }
            span = f"{ids[0]} - {ids[-1]}"
            for attempt in range(HTTP_BUSY_RETRIES):
                self.logger.info(
                    f"Downloading PubMed abstracts for {span} (attempt {attempt + 1})"
                )
                try:
                    response = requests.post(url, data=params, timeout=PUBMED_TIMEOUT)
                    if response.status_code == 200:
                        filepath.write_text(response.text)
                        self.logger.info(f"Successfully downloaded PubMed abstracts for {span}")
                        return
                    problem = f"status code: {response.status_code}"
                except requests.RequestException as e:
                    problem = f"error: {e}"
                self.logger.warning(f"Failed to download PubMed abstracts for {span} ({problem}).")
                if attempt < HTTP_BUSY_RETRIES - 1:
                    time.sleep(HTTP_BACKOFF * 2**attempt)
            self.logger.error(
                f"Failed to download PubMed abstracts for {span} after {HTTP_BUSY_RETRIES} attempts."
            )

        ref_pmids = [ref_pmids[i : i + 10000] for i in range(0, len(ref_pmids), 10000)]
        with progress(total=len(ref_pmids)) as pbar:
            for _ in ThreadPool(10).imap_unordered(
                download_pubmed_abstracts, ref_pmids
            ):
                pbar.update()

    def parse_reference_xml(self, xml):
        """
        Parse a PubMed XML file and extract references.

        Args:
            xml (str): Filename of the XML file.

        Returns:
            list: List of reference dicts with pmid, title, abstract, doi.
        """
        self.logger.info(f"Parsing reference XML: {xml}")
        references = []
        try:
            with open(self.REFERENCE_DOWNLOAD_PATH / xml, "r") as f:
                soup = BeautifulSoup(f.read(), "lxml-xml").find_all("PubmedArticle")
            self.logger.info(f"Found {len(soup)} PubmedArticle entries in {xml}")
        except Exception as e:
            self.logger.error(f"Failed to read or parse XML file {xml}: {e}")
            return references

        for elem in soup:
            try:
                title = elem.MedlineCitation.Article.ArticleTitle.text
            except Exception as e:
                title = None
                self.logger.warning(f"Failed to extract title: {e}")
            try:
                abstract = elem.MedlineCitation.Article.Abstract.AbstractText.text
            except Exception as e:
                abstract = None
                self.logger.warning(f"Failed to extract abstract: {e}")
            try:
                pmid = elem.PubmedData.ArticleIdList.find(
                    "ArticleId", {"IdType": "pubmed"}
                ).text
            except Exception as e:
                pmid = None
                self.logger.warning(f"Failed to extract pmid: {e}")
            try:
                doi = elem.PubmedData.ArticleIdList.find(
                    "ArticleId", {"IdType": "doi"}
                ).text
            except Exception as e:
                doi = None
                self.logger.warning(f"Failed to extract doi: {e}")
            references.append(
                {"pmid": pmid, "title": title, "abstract": abstract, "doi": doi}
            )

        self.logger.info(f"Parsed {len(references)} references from {xml}")
        return references

    def parse_all_reference_xmls(self):
        """
        Parse all XML files in REFERENCE_DOWNLOAD_PATH using parse_reference_xml.

        Args:
            None

        Returns:
            list: List of all parsed references from all XML files.
        """
        all_references = []
        xml_files = [f for f in self.REFERENCE_DOWNLOAD_PATH.glob("*.xml")]
        self.logger.info(
            f"Found {len(xml_files)} XML files in {self.REFERENCE_DOWNLOAD_PATH}"
        )
        for xml_file in xml_files:
            references = self.parse_reference_xml(xml_file.name)
            all_references.extend(references)
        self.logger.info(f"Parsed references from {len(xml_files)} XML files.")
        return all_references

    def smiles_to_ecfp(self, smiles, radius=2, n_bits=1024):
        try:
            mol = MolFromSmiles(smiles)
        except Exception as e:
            self.logger.error(f"Error converting SMILES to molecule: {e}")
            return None
        if not mol:
            self.logger.warning(f"Invalid SMILES: {smiles}")
            return None
        generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        ecfp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(ecfp, arr)
        return arr

    def smiles_to_atom_pair_fp(self, smiles, n_bits=1024):
        try:
            mol = MolFromSmiles(smiles)
        except Exception as e:
            self.logger.error(f"Error converting SMILES to molecule: {e}")
            return None
        if not mol:
            self.logger.warning(f"Invalid SMILES: {smiles}")
            return None
        generator = GetAtomPairGenerator(fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def smiles_hash(self, smiles):
        """
        Generate a SHA256 hash for a given SMILES string.

        Args:
            smiles (str): The SMILES string to hash.
        Returns:
            str: The SHA256 hash of the SMILES string.
        """
        return hashlib.sha256(smiles.encode()).hexdigest()

    def mutate_sequence(self, row):
        """
        Mutate the sequence of the protein based on the mutations present in the row.
        """
        import pandas as pd

        if not row["mutations"] or pd.isna(row["mutations"]):
            self.logger.error(f"No mutations found for row index {row.name}")
            row["UniprotID"], row["sequence"] = None, None
            return row
        if not row["sequence"] or pd.isna(row["sequence"]):
            self.logger.error(f"No sequence found for row index {row.name}")
            row["UniprotID"], row["sequence"] = None, None
            return row

        mutations = row["mutations"]
        sequence = row["sequence"]
        uniprot_id = row["UniprotID"]

        for mutation in mutations:
            try:
                old_residue, position, new_residue = (
                    mutation[0],
                    mutation[1:-1],
                    mutation[-1],
                )

                if (
                    not position.isdigit()
                    or int(position) < 1
                    or int(position) > len(sequence)
                ):
                    self.logger.error(
                        f"Invalid position {position} for mutation {mutation} in sequence {sequence}"
                    )
                    row["UniprotID"], row["sequence"] = None, None
                    return row

                position = int(position) - 1  # Convert to 0-based index
                if sequence[position] != old_residue:
                    self.logger.error(
                        f"Mismatch at position {position + 1}: expected {old_residue}, found {sequence[position]}"
                    )
                    row["UniprotID"], row["sequence"] = None, None
                    return row

                sequence = (
                    sequence[:(position)] + new_residue + sequence[(position + 1) :]
                )
                uniprot_id += f"|{mutation}"  # Append mutation to UniProt ID
            except Exception as e:
                self.logger.error(
                    f"Error processing mutation {mutation} for sequence {sequence}: {e}"
                )
                row["UniprotID"], row["sequence"] = None, None
                return row

        row["UniprotID"], row["sequence"] = uniprot_id, sequence
        return row

    def generate_smi_file(self, unique_smiles, smi_file_path):
        """
        Write unique SMILES strings to a .smi file.

        Args:
            unique_smiles (iterable): Collection of SMILES strings to write.
            smi_file_path (str or Path): Path to output .smi file.

        Returns:
            None
        """
        smi_file_path = Path(smi_file_path)
        smi_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(smi_file_path, "w") as f:
                for smi in unique_smiles:
                    f.write(f"{smi}\n")
            self.logger.info(f".smi file saved at {smi_file_path}")
        except Exception as e:
            self.logger.error(f"Error writing .smi file: {e}")
            raise

    def generate_fasta_file(self, unique_sequences_df, fasta_file_path):
        """
        Write unique protein sequences to a FASTA file.

        Args:
            unique_sequences_df (pd.DataFrame): DataFrame with columns 'uniprot_id' and 'sequence'.
            fasta_file_path (str or Path): Path to output .fasta file.

        Returns:
            None
        """
        fasta_file_path = Path(fasta_file_path)
        fasta_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(fasta_file_path, "w") as f:
                for idx, row in unique_sequences_df.iterrows():
                    f.write(f">{row['uniprot_id']}\n{row['sequence']}\n")
            self.logger.info(f".fasta file saved at {fasta_file_path}")
        except Exception as e:
            self.logger.error(f"Error writing .fasta file: {e}")
            raise

    def assign_experimental_and_af_pdbs(self, df):
        """Assign each row experimental mmCIF chains, else one AlphaFold structure.

        A chain of a PDB entry matches the row's UniProt accession at identity >= MIN_IDENTITY;
        the best-covering chain per PDB is selected when its coverage is >= MIN_FALLBACK_COVERAGE.
        A row keeps every PDB whose chain covers >= MIN_COVERAGE. Without such a PDB it keeps
        its best-covering PDB. Rows without a selected chain get one fallback row.
        Experimental rows have the chain and its coverage in `pdb_chain` and `pdb_coverage`.
        A mutant ID such as `P12345|R1Q` is matched through its base accession `P12345`.
        Experimental `pdbs` is the structure key `<PDB>:<base accession>`; fallback `pdbs` is the
        accession.
        """
        self.logger.info("Exploding experimental PDBs...")
        df = df.reset_index(drop=True)
        df["_row_id"] = df.index
        # Explode only the columns the selection needs; the other columns are copied per kept row.
        exploded = (
            df[["_row_id", "uniprot_id", "pdbs"]]
            .explode("pdbs")
            .dropna(subset=["pdbs"])
            .reset_index(drop=True)
        )
        exploded["pdbs"] = exploded["pdbs"].str.upper()
        exploded["_acc"] = exploded["uniprot_id"].str.partition("|")[0]

        unique_pdb_ids = exploded["pdbs"].unique().tolist()
        self.logger.info(f"Found {len(unique_pdb_ids)} unique PDB IDs to download if missing.")
        if unique_pdb_ids:
            self.get_ebi_pdb(unique_pdb_ids)

        pairs = exploded[["pdbs", "_acc"]].drop_duplicates()
        pairs = list(pairs.itertuples(index=False, name=None))
        accs = sorted({acc for _, acc in pairs if isinstance(acc, str)})
        sequences = {
            acc: data["sequence"]
            for acc, data in self.get_uniprot_acc_json(accs).items()
            if data.get("sequence")
        }
        selections = select_chains(pairs, sequences, self.MMCIF_PATH)
        chosen = [
            selections.get(pair, (None, "", 0.0))
            for pair in zip(exploded["pdbs"], exploded["_acc"])
        ]
        exploded["pdb_chain"] = [chain for chain, _, _ in chosen]
        exploded["pdb_coverage"] = [coverage for _, _, coverage in chosen]
        selected = exploded[exploded["pdb_chain"].notna()]
        full = selected[selected["pdb_coverage"] >= MIN_COVERAGE]
        partial = (
            selected[~selected["_row_id"].isin(full["_row_id"])]
            .sort_values("pdb_coverage", ascending=False, kind="stable")
            .drop_duplicates("_row_id")
        )
        kept = pd.concat([full, partial])
        experimental_df = df.iloc[kept["_row_id"].to_numpy()].reset_index(drop=True)
        experimental_df["pdbs"] = (kept["pdbs"] + ":" + kept["_acc"]).to_numpy()
        experimental_df["pdb_chain"] = kept["pdb_chain"].to_numpy()
        experimental_df["pdb_coverage"] = kept["pdb_coverage"].to_numpy()
        experimental_df["pdb_source"] = "PDBe"
        experimental_df["pdb_type"] = "experimental"
        missing_pdb_df = (
            df[~df["_row_id"].isin(experimental_df["_row_id"])]
            .drop(columns=["pdbs"])
            .reset_index(drop=True)
        )
        self.logger.info(
            f"Chain selection for {len(selections)} (PDB, accession) pairs: "
            f"{dict(selection_counts(selections))}. Rows kept: {len(full)} with coverage >= "
            f"{MIN_COVERAGE}, {len(partial)} best partial, {len(missing_pdb_df)} fallback."
        )
        missing_pdb_df["pdbs"] = missing_pdb_df["uniprot_id"]
        missing_pdb_df["pdb_source"] = None
        missing_pdb_df["pdb_type"] = None

        self.logger.info(
            "Attempting to assign AlphaFold predicted PDBs to entries without an accepted experimental chain..."
        )
        self.get_af_pdb(
            sorted(
                missing_pdb_df[
                    ~(
                        missing_pdb_df["pdbs"].isna()
                        | missing_pdb_df["pdbs"].str.contains(r"\|")
                    )
                ][
                    "pdbs"
                ]
                .unique()
                .tolist()
            )
        )

        self.logger.info("Checking existence of AlphaFold PDB files...")
        af_pdb_files = list(self.AF_PDB_PATH.glob("AF-*-F1-model_v*.pdb"))
        af_acc_ids = set(f.name.split("-")[1] for f in af_pdb_files)

        mask = missing_pdb_df["pdbs"].isin(af_acc_ids)
        missing_pdb_df.loc[mask, "pdb_source"] = "AlphaFold"
        missing_pdb_df.loc[mask, "pdb_type"] = "predicted"

        df = pd.concat([experimental_df, missing_pdb_df], ignore_index=True).drop(
            columns=["_row_id"]
        )

        self.logger.info(
            f"Total entries after assigning experimental and predicted PDBs: {df.shape}"
        )

        return df
