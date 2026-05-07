from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import io
import os
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
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rdkit.Chem import AllChem
from zeep import Client, Settings
from zeep.transports import Transport
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
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.utils.tqdmlogger import TqdmLogger


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
        self.PUBCHEM_CID_FROM_NAME_URL = cfg.chem_utils.pubchem_cid_from_name_url
        self.PUBCHEM_CID_SMILES_URL = cfg.chem_utils.pubchem_cid_smiles_url
        self.UNIPROT_API_URL = cfg.chem_utils.uniprot_api_url
        self.ALPHAFOLD_API_URL = cfg.chem_utils.alphafold_api_url
        self.EBI_PDB_URL = cfg.chem_utils.ebi_pdb_url
        self.EBI_MMCIF_ARCHIVE_URL = cfg.chem_utils.ebi_mmcif_archive_url
        self.EBI_MMCIF_UPDATED_URL = cfg.chem_utils.ebi_mmcif_updated_url
        self.PUBMED_ENTREZ_EFETCH_URL = cfg.chem_utils.pubmed_entrez_efetch_url
        self.BRENDA_WSDL_URL = cfg.chem_utils.brenda_wsdl_url
        self.BRENDA_MOL_DOWNLOAD_URL = cfg.chem_utils.brenda_mol_download_url

        # Convert config paths to Path objects
        LOG_PATH = Path(cfg.chem_utils.log_dir)
        self.MOL_PATH = {k: Path(v) for k, v in cfg.chem_utils.mol_path.items()}
        self.SDF_PATH = Path(cfg.chem_utils.sdf_path)
        self.PDB_PATH = Path(cfg.chem_utils.pdb_path)
        self.MMCIF_PATH = Path(cfg.chem_utils.mmcif_path)
        self.PROCESSED_EXP_PDB_PATH = Path(cfg.chem_utils.processed_exp_pdb_path)
        self.REFERENCE_DOWNLOAD_PATH = Path(cfg.chem_utils.reference_raw_path)
        self.PUBCHEM_DOWNLOAD_PATH = Path(cfg.chem_utils.pubchem_raw_path)
        self.PUBCHEM_NAME_SEARCH_PATH = Path(cfg.chem_utils.pubchem_name_search_path)
        self.PUBCHEM_CID_SEARCH_PATH = Path(cfg.chem_utils.pubchem_cid_search_path)
        self.AF_DOWNLOAD_PATH = Path(cfg.chem_utils.af_json_path)
        self.AF_PDB_PATH = Path(cfg.chem_utils.af_pdb_path)
        self.ESM_PDB_PATH = Path(cfg.chem_utils.esm_pdb_path)
        self.UNIPROT_ACC_PATH = Path(cfg.chem_utils.uniprot_acc_path)
        self.UNIPROT_TIME_PATH = Path(cfg.chem_utils.uniprot_time_path)
        self.UNIPROT_EC_ACC_PATH = Path(cfg.chem_utils.uniprot_ec_acc_path)
        self.UNIPROT_EC_ORG_ACC_PATH = Path(cfg.chem_utils.uniprot_ec_org_acc_path)

        self.PUBCHEM_CID_SMILES_RAW_FILE_PATH = Path(
            cfg.chem_utils.pubchem_cid_smiles_raw_file_path
        )
        self.PUBCHEM_CID_SMILES_GZ_FILE_PATH = Path(
            cfg.chem_utils.pubchem_cid_smiles_gz_file_path
        )
        self.PUBCHEM_CID_SMILES_PROCESSED_FILE_PATH = Path(
            cfg.chem_utils.pubchem_cid_smiles_processed_file_path
        )
        # Create directories if they don't exist
        for path in [
            LOG_PATH,
            self.SDF_PATH,
            self.PDB_PATH,
            self.MMCIF_PATH,
            self.REFERENCE_DOWNLOAD_PATH,
            self.PUBCHEM_NAME_SEARCH_PATH,
            self.PUBCHEM_CID_SEARCH_PATH,
            self.AF_DOWNLOAD_PATH,
            self.AF_PDB_PATH,
            self.UNIPROT_ACC_PATH,
            self.UNIPROT_TIME_PATH,
            self.UNIPROT_EC_ACC_PATH,
            self.UNIPROT_EC_ORG_ACC_PATH,
        ] + list(self.MOL_PATH.values()):
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
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
            None
        """
        if path.exists():
            return
        for _ in range(2):  # Retry up to 2 times
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    return
                elif response.status_code == 404:
                    self.logger.warning(f"404 - {url}")
                    return
            except Exception as e:
                self.logger.warning(
                    f"Failed to retrieve {url} with error {e}. Retrying..."
                )
                time.sleep(5)
        else:
            self.logger.error(f"Failed to retrieve {url}. Max retries exceeded.")
        return

    def canonicalize_smiles(self, smiles):
        """
        Standardize and canonicalize a SMILES string using datamol.

        Applies a series of transformations to standardize SMILES:
        1. Convert SMILES string to molecule object
        2. Fix structural issues (implicit hydrogens, charges)
        3. Sanitize connectivity and valence
        4. Standardize (remove salts, reionize, normalize, fix stereo)
        5. Convert back to canonical SMILES representation

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

        # Apply operations sequentially
        mol = smiles
        for step_name, operation in operations:
            try:
                mol = operation(mol)
            except Exception as e:
                self.logger.warning(f"{step_name} failed for {smiles}: {e}")
                return None

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

            canonicalized = sorted(canonicalized, key=len, reverse=True)
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
        except ImportError:
            active_logger.warning(
                "pandarallel is not installed. Install it to speed up canonicalization of large reaction sets. Falling back to single-threaded processing."
            )
            unique_metabolites_df["canonical smiles"] = unique_metabolites_df[
                "smiles"
            ].apply(self.canonicalize_smiles)

        active_logger.info(
            f"Canonicalizing {len(unique_metabolites_df)} unique metabolite SMILES in parallel with {os.cpu_count()} workers."
        )
        unique_metabolites_df["canonical smiles"] = unique_metabolites_df[
            "smiles"
        ].parallel_apply(self.canonicalize_smiles)

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
                response = requests.get(url)
                if response.status_code == 200:
                    if len(response.content) > 0:
                        with open(mol_path, "wb") as f:
                            f.write(response.content)
                    else:
                        self.logger.warning(f"{id} - No mol file. Skipping...")
                    break
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

        with tqdm(
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
        with tqdm(total=len(args)) as pbar:
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

    def get_pubchem_cids_from_name_parallel(self, compound_names):
        """
        Retrieve PubChem CIDs for a list of compound names in parallel.

        Args:
            compound_names (list): List of compound names to search in PubChem.

        Returns:
            list: List of dictionaries containing compound names and their corresponding CIDs.
        """
        results = []

        def get_pubchem_cids_from_name(compound_name):
            cache_path = (
                self.PUBCHEM_NAME_SEARCH_PATH
                / f'{compound_name.replace("/", "|")}.json'
            )
            try:
                if cache_path.exists():
                    try:
                        results.append(self.load_json(cache_path))
                        self.logger.info(f"PubChem: Loaded cache for {compound_name}")
                        return
                    except Exception as e:
                        self.logger.warning(
                            f"PubChem: Failed to load cache for {compound_name}: {e}"
                        )
            except Exception as e:
                self.logger.error(
                    f"PubChem: Error checking cache for {compound_name}: {e}"
                )
                self.logger.error(
                    f"PubChem: Searching online for {compound_name} in PubChem..."
                )

            search_url = self.PUBCHEM_CID_FROM_NAME_URL.format(
                compound_name=compound_name
            )
            for _ in range(2):  # Retry up to 2 times in case of errors
                try:
                    search_response = requests.get(search_url)

                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        if (
                            "IdentifierList" in search_data
                            and "CID" in search_data["IdentifierList"]
                        ):
                            try:
                                cid = str(search_data["IdentifierList"]["CID"][0])
                                result = {"compound_name": compound_name, "cid": cid}
                                results.append(result)
                                try:
                                    self.save_json(data=result, file_path=cache_path)
                                except Exception as e:
                                    self.logger.error(
                                        f"PubChem: Failed to save cache for {compound_name}: {e}"
                                    )
                                self.logger.info(
                                    f"PubChem: Found CID {cid} for {compound_name}"
                                )
                            except Exception as e:
                                result = {"compound_name": compound_name, "cid": None}
                                try:
                                    self.save_json(data=result, file_path=cache_path)
                                except Exception as e:
                                    self.logger.error(
                                        f"PubChem: Failed to save cache for {compound_name}: {e}"
                                    )
                                self.logger.error(
                                    f"PubChem: Error extracting CID for {compound_name}: {e}"
                                )
                                return
                        else:
                            result = {"compound_name": compound_name, "cid": None}
                            results.append(result)
                            try:
                                self.save_json(data=result, file_path=cache_path)
                            except Exception as e:
                                self.logger.error(
                                    f"PubChem: Failed to save cache for {compound_name}: {e}"
                                )
                            self.logger.warning(
                                f"PubChem: No CID found for {compound_name}. Status code: {search_response.status_code}"
                            )
                        return
                except Exception as e:
                    time.sleep(2)  # Wait before retrying
                    self.logger.error(
                        f"PubChem: Error retrieving PubChem CID for {compound_name}: {e}. Retrying..."
                    )
            else:
                result = {"compound_name": compound_name, "cid": None}
                results.append(result)
                try:
                    self.save_json(data=result, file_path=cache_path)
                except Exception as e:
                    self.logger.error(
                        f"PubChem: Failed to save cache for {compound_name}: {e}"
                    )
                self.logger.error(
                    f"PubChem: Failed to retrieve PubChem CID for {compound_name} after multiple attempts. Skipping."
                )
                return None

        with tqdm(
            total=len(compound_names), desc="Retrieving PubChem CIDs", leave=False
        ) as pbar:
            for _ in ThreadPool(5).imap_unordered(
                lambda x: get_pubchem_cids_from_name(x), compound_names
            ):
                pbar.update()
        return results

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

    def download_and_process_pubchem_cids(self):
        """
        Download CID-SMILES mapping from PubChem FTP, extract it,
        canonicalize SMILES using multiprocessing, and save as parquet.

        First checks if processed mapping exists and returns it.
        Then checks if raw mapping exists, otherwise downloads and extracts it.
        Finally processes SMILES with multiprocessing Pool and saves result.

        Returns:
            pd.DataFrame: DataFrame with CID and canonical SMILES columns
        """
        import subprocess

        # Check if processed file exists
        if self.PUBCHEM_CID_SMILES_PROCESSED_FILE_PATH.exists():
            self.logger.info(
                f"Processed CID-SMILES mapping found at {self.PUBCHEM_CID_SMILES_PROCESSED_FILE_PATH}. Loading..."
            )
            return pd.read_parquet(self.PUBCHEM_CID_SMILES_PROCESSED_FILE_PATH)

        # Check if raw file exists, if not download and extract
        if not self.PUBCHEM_CID_SMILES_RAW_FILE_PATH.exists():
            self.logger.info("Raw CID-SMILES mapping not found. Downloading...")

            # Create directory if it doesn't exist
            self.PUBCHEM_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

            # Download using curl
            try:
                subprocess.run(
                    [
                        "curl",
                        "-Lo",
                        str(self.PUBCHEM_CID_SMILES_GZ_FILE_PATH),
                        self.PUBCHEM_CID_SMILES_URL,
                    ],
                    check=True,
                    timeout=600,
                )
                self.logger.info(
                    f"Downloaded CID-SMILES.gz to {self.PUBCHEM_CID_SMILES_GZ_FILE_PATH}"
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to download CID-SMILES.gz: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Error downloading CID-SMILES.gz: {e}")
                raise

            # Extract using gunzip -c and redirect to output file
            try:
                self.logger.info(
                    f"Extracting CID-SMILES.gz to {self.PUBCHEM_CID_SMILES_RAW_FILE_PATH}..."
                )
                with open(self.PUBCHEM_CID_SMILES_RAW_FILE_PATH, "wb") as out_f:
                    subprocess.run(
                        ["gunzip", "-c", str(self.PUBCHEM_CID_SMILES_GZ_FILE_PATH)],
                        check=True,
                        stdout=out_f,
                        stderr=subprocess.PIPE,
                        timeout=120,
                    )
                self.logger.info(
                    f"Extracted CID-SMILES to {self.PUBCHEM_CID_SMILES_RAW_FILE_PATH}"
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to extract CID-SMILES.gz: {e}")
                raise
        else:
            self.logger.info(
                f"Raw CID-SMILES mapping found at {self.PUBCHEM_CID_SMILES_RAW_FILE_PATH}. Loading..."
            )

        # Read the raw file
        self.logger.info("Reading CID-SMILES mapping...")
        pubchem_cids = pd.read_csv(
            self.PUBCHEM_CID_SMILES_RAW_FILE_PATH,
            sep="\t",
            header=None,
            names=["CID", "SMILES"],
        )

        self.logger.info(f"Loaded {len(pubchem_cids)} CID-SMILES pairs")

        from joblib import Parallel, delayed

        # Use physical core count (os.cpu_count() returns thread count)
        nb_workers = os.cpu_count()

        chunk_size = 1000
        chunks = [
            pubchem_cids.iloc[i : i + chunk_size]
            for i in range(0, len(pubchem_cids), chunk_size)
        ]
        self.logger.info(
            f"Processing {len(chunks)} SMILES chunks with {nb_workers} workers"
        )

        def process_chunk(chunk):
            return chunk["SMILES"].apply(self.canonicalize_smiles).values

        results = Parallel(n_jobs=nb_workers, verbose=0, prefer="processes")(
            delayed(process_chunk)(chunk)
            for chunk in tqdm(chunks, desc="Canonicalizing SMILES chunks")
        )
        pubchem_cids["canonical_smiles"] = [
            item for sublist in results for item in sublist
        ]

        # Clean and process
        pubchem_cids = (
            pubchem_cids.dropna(subset=["canonical_smiles"])
            .drop(columns=["SMILES"])
            .rename(columns={"canonical_smiles": "smiles"})
        )

        self.logger.info(f"After cleaning: {len(pubchem_cids)} valid CID-SMILES pairs")

        # Save processed file
        self.logger.info(
            f"Saving processed mapping to {self.PUBCHEM_CID_SMILES_PROCESSED_FILE_PATH}..."
        )
        pubchem_cids.to_parquet(
            self.PUBCHEM_CID_SMILES_PROCESSED_FILE_PATH,
            index=False,
            compression="brotli",
        )

        self.logger.info("CID-SMILES processing completed.")
        return pubchem_cids

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

        with tqdm(
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
        for id, smiles in tqdm(
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
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")

    def get_uniprot_json(self, query, fields, verbose=False):
        """
        Query UniProt for protein data.

        Args:
            query (str): Query string for UniProt.
            fields (str): Fields to retrieve.
            verbose (bool): If True, log warnings.

        Returns:
            list: List of result dicts, or [None] if not found.
        """

        response = requests.get(
            url=self.UNIPROT_API_URL, params={"query": query, "fields": fields}
        )

        if response.status_code != 200:
            msg = f"Request failed with status code {response.status_code} for {query}"
            if verbose:
                self.logger.warning(msg)
            else:
                self.logger.debug(msg)
            return [None]

        results = response.json()["results"]
        if not results:
            msg = f"No protein data found for query {query}"
            if verbose:
                self.logger.warning(msg)
            else:
                self.logger.debug(msg)
            return [None]
        return results

    def get_uniprot_acc_json(self, accs, verbose=False):
        """
        Download UniProt data for a list of accession IDs.

        Args:
            accs (list): List of UniProt accession IDs.
            verbose (bool): If True, log warnings.

        Returns:
            dict: Mapping of accession IDs to UniProt data.
        """
        results = {}

        def process_acc(acc, verbose):
            json_file_path = self.UNIPROT_ACC_PATH / f"{acc}.json"

            if json_file_path.exists():
                results[acc] = self.load_json(json_file_path)
                return

            query = f"accession:{acc}"
            fields = "accession,ec,sequence,structure_3d"
            for _ in range(2):  # Retry up to 2 times
                data = self.get_uniprot_json(query, fields, verbose)
                if not data:
                    results[acc] = None
                    self.logger.warning(f"No UniProt data found for {acc}")
                    return

                data = data[0]

                if not data:
                    results[acc] = None
                    self.logger.warning(f"No UniProt data found for {acc}")
                    return

                # Handle inactive entries
                if data.get("entryType", "") == "Inactive":
                    inactive_reason = data.get("inactiveReason", {})
                    if (
                        isinstance(inactive_reason, dict)
                        and inactive_reason.get("inactiveReasonType") == "MERGED"
                    ):
                        new_acc = inactive_reason.get("mergeDemergeTo", [None])[0]
                        if not new_acc:
                            self.logger.warning(
                                f"Inactive UniProt entry for {acc} with no merge target."
                            )
                            return
                        query = f"accession:{new_acc}"
                        self.logger.info(
                            f"UniProt entry {acc} merged to {new_acc}, Retrying."
                        )
                        continue
                    else:
                        self.logger.warning(f"Inactive UniProt entry for {acc}.")
                        return
                else:
                    break
            try:
                ecs = [
                    elem["value"]
                    for elem in data["proteinDescription"]["recommendedName"][
                        "ecNumbers"
                    ]
                ]
            except Exception as e:
                ecs = None
                self.logger.warning(f"Failed to extract EC numbers for {acc}: {e}")
            try:
                pdbs = tuple(
                    [
                        i["id"]
                        for i in data["uniProtKBCrossReferences"]
                        if i["database"] == "PDB"
                    ]
                )
            except Exception as e:
                pdbs = None
                self.logger.warning(f"Failed to extract PDBs for {acc}: {e}")
            try:
                sequence = data["sequence"]["value"]
            except Exception as e:
                sequence = None
                self.logger.warning(f"Failed to extract sequence for {acc}: {e}")

            json_data = {"acc_id": acc, "ecs": ecs, "pdbs": pdbs, "sequence": sequence}
            self.save_json(json_data, json_file_path)
            results[acc] = json_data
            return

        with tqdm(
            total=len(accs), desc="Downloading protein data from UniProt", leave=False
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(
                lambda x: process_acc(x, verbose), accs
            ):
                pbar.update()
        return results

    def get_uniprot_ec_acc_map(self, ecs, verbose=False):
        """
        Map EC numbers to UniProt accession IDs.

        Args:
            ecs (list): List of EC numbers.
            verbose (bool): If True, log warnings.

        Returns:
            dict: Mapping of EC numbers to lists of accession IDs.
        """
        results = {}

        def process_ec(ec, verbose):
            json_file_path = self.UNIPROT_EC_ACC_PATH / f"{ec}.json"
            if json_file_path.exists():
                results[ec] = self.load_json(json_file_path)
                return

            response = self.get_uniprot_json(
                query=f"ec:{ec} AND active:true AND reviewed:true",
                fields="accession",
                verbose=verbose,
            )
            if response == [None]:
                self.logger.warning(
                    f"No reviewed UniProt acc_ids found for EC {ec}. Trying unreviewed entries."
                )
                response = self.get_uniprot_json(
                    query=f"ec:{ec} AND active:true AND reviewed:false",
                    fields="accession",
                    verbose=verbose,
                )

            if response == [None]:
                results[ec] = None
                self.logger.error(f"No UniProt acc_ids found for EC {ec}")
                self.save_json(data=None, file_path=json_file_path)
                return

            acc_ids = [data["primaryAccession"] for data in response]
            self.save_json(acc_ids, json_file_path)
            results[ec] = acc_ids
            return

        with tqdm(
            total=len(ecs),
            desc="Downloading ec - acc_ids maps from UniProt",
            leave=False,
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(
                lambda x: process_ec(x, verbose), ecs
            ):
                pbar.update()
        return results

    def get_active_acc_id_from_ec_and_org_parallel(self, ec_org_pairs):
        """
        Fetch active UniProt accession IDs for given EC numbers and organisms in parallel.

        Args:
            ec_org_pairs (list of tuples): List of (EC number, organism name) pairs.

        Returns:
            pd.DataFrame: DataFrame containing active UniProt accession IDs with their review status.
        """
        import pandas as pd

        results = {}

        def process_ec_org_pair(ec_org):
            """
            Process a single EC-organism pair to fetch active UniProt accession IDs.

            Args:
                ec_org (tuple): A tuple containing an EC number and an organism name.
            """

            ec, org = ec_org
            json_path = self.UNIPROT_EC_ORG_ACC_PATH / f"{ec} | {org}.json"
            if json_path.exists():
                data = self.load_json(json_path)
                results[ec_org] = (
                    pd.DataFrame(data) if data else None
                )  # Check for empty JSON
                return

            response = self.get_uniprot_json(
                query=f'ec:{ec} AND organism_name:"{org}" AND active:true',
                fields="accession",
            )[0]

            if response:
                acc_ids = pd.DataFrame(response)
                acc_ids = acc_ids[["entryType", "primaryAccession"]]
                acc_ids["ECNumber"] = ec
                acc_ids["Organism"] = org

                # Determine if the entry is reviewed or unreviewed
                acc_ids["entryType"] = acc_ids["entryType"].apply(
                    lambda x: False if "unreviewed" in x else True
                )
                acc_ids.rename(
                    columns={"primaryAccession": "UniprotID", "entryType": "reviewed"},
                    inplace=True,
                )
                results[ec_org] = acc_ids
                self.save_json(
                    acc_ids.to_dict(orient="records"),
                    self.UNIPROT_EC_ORG_ACC_PATH / f"{ec} | {org}.json",
                )
                return

            else:
                self.logger.warning(
                    f"No active accession IDs found for query: {ec} | {org}"
                )
                results[ec_org] = None
                return

        # Use ThreadPool to process EC-organism pairs in parallel
        self.logger.info("Fetching active accession IDs from UniProt...")
        with tqdm(
            total=len(ec_org_pairs), desc="Fetching active accession IDs"
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(process_ec_org_pair, ec_org_pairs):
                pbar.update()

        self.logger.info("Accession IDs fetching completed.")

        # Combine all results into a single DataFrame
        results = pd.concat(results.values(), ignore_index=True)
        return results

    def get_uniprot_time_json(self, accs, verbose=False):
        """
        Download UniProt time information for a list of accession IDs.

        Args:
            accs (list): List of UniProt accession IDs.
            verbose (bool): If True, log warnings.

        Returns:
            dict: Mapping of accession IDs to UniProt time data.
        """
        results = {}

        def process_acc(acc, verbose):
            json_file_path = self.UNIPROT_TIME_PATH / f"{acc}.json"
            if json_file_path.exists():
                results[acc] = self.load_json(json_file_path)
                return

            query = f"accession:{acc}"
            fields = "date_created,date_modified,date_sequence_modified,version"
            for _ in range(2):  # Retry up to 2 times
                data = self.get_uniprot_json(query, fields, verbose)
                if not data:
                    results[acc] = None
                    self.logger.warning(f"No UniProt time data found for {acc}")
                    return

                data = data[0]

                if not data:
                    results[acc] = None
                    self.logger.warning(f"No UniProt time data found for {acc}")
                    return

                # Handle inactive entries
                if data.get("entryType", "") == "Inactive":
                    inactive_reason = data.get("inactiveReason", {})
                    if (
                        isinstance(inactive_reason, dict)
                        and inactive_reason.get("inactiveReasonType") == "MERGED"
                    ):
                        new_acc = inactive_reason.get("mergeDemergeTo", [None])[0]
                        if not new_acc:
                            self.logger.warning(
                                f"Inactive UniProt entry for {acc} with no merge target."
                            )
                            return
                        query = f"accession:{new_acc}"
                        self.logger.info(
                            f"UniProt entry {acc} merged to {new_acc}, Retrying."
                        )
                        continue
                    else:
                        self.logger.warning(f"Inactive UniProt entry for {acc}.")
                        return
                else:
                    break
            try:
                date_created = data["entryAudit"]["firstPublicDate"]
            except Exception as e:
                date_created = None
                self.logger.warning(f"Failed to extract date created for {acc}: {e}")
            try:
                date_modified = data["entryAudit"]["lastAnnotationUpdateDate"]
            except Exception as e:
                date_modified = None
                self.logger.warning(f"Failed to extract date modified for {acc}: {e}")
            try:
                date_sequence_modified = data["entryAudit"]["lastSequenceUpdateDate"]
            except Exception as e:
                date_sequence_modified = None
                self.logger.warning(
                    f"Failed to extract date sequence modified for {acc}: {e}"
                )
            try:
                version = data["entryAudit"]["entryVersion"]
            except Exception as e:
                version = None
                self.logger.warning(f"Failed to extract version for {acc}: {e}")

            json_data = {
                "acc_id": acc,
                "date_created": date_created,
                "date_modified": date_modified,
                "date_sequence_modified": date_sequence_modified,
                "version": version,
            }
            self.save_json(json_data, json_file_path)
            results[acc] = json_data
            return

        with tqdm(
            total=len(accs), desc="Downloading time data from UniProt", leave=False
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(
                lambda x: process_acc(x, verbose), accs
            ):
                pbar.update()
        return results

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

    def get_af_pdb(self, args):
        """
        Download AlphaFold PDB files for a list of UniProt IDs.

        Args:
            args (list): List of UniProt IDs.

        Returns:
            None
        """

        def get_af_pdb_helper(id):
            url = self.ALPHAFOLD_API_URL.format(id=id)
            json_path = self.AF_DOWNLOAD_PATH / f"{id}.json"
            self.get_html(url, json_path, timeout=30)
            if not json_path.exists():
                self.logger.warning(f"AlphaFold doesn't have PDB for {id}.")
                return
            try:
                af_cif_url = self.load_json(json_path)[0]["pdbUrl"]
                af_cif_name = af_cif_url.split("/")[-1]
                self.get_html(af_cif_url, self.AF_PDB_PATH / af_cif_name, timeout=30)
                self.logger.info(f"Downloaded AlphaFold PDB for {id}")
            except Exception as e:
                self.logger.error(
                    f"Failed to download AlphaFold PDB for {id} with error {e}"
                )
                return

        args = set(args) - set(
            [
                re.match(r"AF-([A-Za-z0-9]+)-F1-model_v\d+", f.stem).group(1)
                for f in self.AF_PDB_PATH.glob("*.pdb")
                if re.match(r"AF-([A-Za-z0-9]+)-F1-model_v\d+", f.stem)
            ]
        )
        with tqdm(
            total=len(args), desc="Downloading PDB files from AlphaFold", leave=False
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(get_af_pdb_helper, args):
                pbar.update()

    def get_ebi_pdb(self, args):
        """
        Download PDB files from EBI for a list of PDB IDs.
        Tries formats in order: updated mmCIF > archive mmCIF > PDB

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
                (
                    self.EBI_PDB_URL.format(id=id.lower()),
                    self.PDB_PATH / f"{id.upper()}.pdb",
                ),
            ]

            for url, path in urls_and_paths:
                self.get_html(url, path, timeout=30)
                if path.exists():
                    return

            self.logger.warning(
                f"Could not download {id} from any source (updated mmCIF, archive mmCIF, or PDB)"
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
        pdb_files = {f.stem.upper() for f in self.PDB_PATH.glob("*.pdb")}

        # Filter args to only include those that need downloading
        args_to_download = []
        for pdb_id in args:
            pdb_id_upper = pdb_id.upper()
            # Skip if already have any version in preference order
            if pdb_id_upper in mmcif_updated:
                continue
            if pdb_id_upper in mmcif_archive:
                continue
            if pdb_id_upper in pdb_files:
                continue
            args_to_download.append(pdb_id)

        if not args_to_download:
            self.logger.info("All requested PDB files already downloaded. Skipping...")
            return

        self.logger.info(f"Downloading {len(args_to_download)} PDB files from EBI")

        with tqdm(
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
            try:
                for attempt in range(5):
                    self.logger.info(
                        f"Downloading PubMed abstracts for {ids[0]} - {ids[-1]} (attempt {attempt + 1})"
                    )
                    response = requests.post(url, data=params)
                    if response.status_code == 200:
                        with open(filepath, "w") as f:
                            f.write(response.text)
                        self.logger.info(
                            f"Successfully downloaded PubMed abstracts for {ids[0]} - {ids[-1]}"
                        )
                        return
                    else:
                        self.logger.warning(
                            f"Failed to download PubMed abstracts for {ids[0]} - {ids[-1]} (status code: {response.status_code}). Retrying..."
                        )
                        time.sleep(2)
                self.logger.error(
                    f"Failed to download PubMed abstracts for {ids[0]} - {ids[-1]} after 5 attempts."
                )
                return
            except Exception as e:
                self.logger.error(
                    f"Failed to download PubMed abstracts for {ids[0]} - {ids[-1]} with error {e}"
                )
                return

        ref_pmids = [ref_pmids[i : i + 10000] for i in range(0, len(ref_pmids), 10000)]
        with tqdm(total=len(ref_pmids)) as pbar:
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

    def get_brenda_ligand_structure_id(self, ligands):
        EMAIL = os.getenv("BRENDA_EMAIL", None)
        PASSWORD = os.getenv("BRENDA_PASSWORD", None)
        if not EMAIL or not PASSWORD:
            self.logger.error(
                "BRENDA_EMAIL and BRENDA_PASSWORD must be set in the environment variables."
            )
            return dict(zip(ligands, [None] * len(ligands)))

        PASSWORD = hashlib.sha256(PASSWORD.encode()).hexdigest()
        transport = Transport(timeout=30, operation_timeout=30)
        client = Client(
            wsdl=self.BRENDA_WSDL_URL,
            settings=Settings(strict=False),
            transport=transport,
        )

        def get_ligand_structure_id_helper(sub):
            try:
                param = (EMAIL, PASSWORD, sub)
                result = client.service.getLigandStructureIdByCompoundName(*param)
                return sub, result
            except Exception as e:
                tqdm.write(f"Error fetching ID for {sub}: {e}\n")
                return sub, None

        ligand_id_dict = {}
        with ThreadPool(processes=40) as pool:
            for sub, result in tqdm(
                pool.imap_unordered(get_ligand_structure_id_helper, ligands),
                total=len(ligands),
                desc="Mapping ligand names to BRENDA Group IDs",
            ):
                ligand_id_dict[sub] = result

        return ligand_id_dict

    def download_brenda_mol(self, ligand_ids):
        self.logger.info("Starting download of BRENDA ligand mol files...")
        existing_ligand_ids = {
            f.name[7:].split("_")[0]: f.name
            for f in self.MOL_PATH["2D"].iterdir()
            if f.is_file() and f.name.startswith("BRENDA:") and f.name.endswith(".mol")
        }
        self.logger.info(
            f"Found {len(existing_ligand_ids)} existing BRENDA ligand mol files."
        )

        def download_brenda_mol_helper(ligand_id):
            mol_path = self.MOL_PATH["2D"]
            self.logger.info(f"Downloading mol file for BRENDA:{ligand_id}...")
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Connection": "keep-alive",
                }
                url = self.BRENDA_MOL_DOWNLOAD_URL.format(ligand_id=ligand_id)
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code != 200:
                    self.logger.error(
                        f"Failed to retrieve BRENDA page for ligand_id {ligand_id} with status code {resp.status_code}"
                    )
                    return ligand_id, False

                match = re.search(
                    r'href=["\'](\./molfile\.php\?LigandID=(\d+))["\']', resp.text
                )
                if not match:
                    self.logger.error(
                        f"No molfile download link found for ligand_id {ligand_id}"
                    )
                    return ligand_id, False

                href = match.group(1)
                ligand_id_in_href = match.group(2)

                if not ligand_id_in_href:
                    self.logger.debug(f"Could not extract LigandID from href: {href}")
                    return ligand_id, False

                mol_url = "https://www.brenda-enzymes.org/" + href.replace("./", "")
                self.get_html(
                    mol_url, mol_path / f"BRENDA:{ligand_id}_{ligand_id_in_href}.mol"
                )
                existing_ligand_ids[ligand_id] = (
                    f"BRENDA:{ligand_id}_{ligand_id_in_href}.mol"
                )
                return ligand_id, True
            except Exception as e:
                self.logger.error(f"Failed to download {ligand_id}: {e}")
                return ligand_id, False

        ligand_ids = set(ligand_ids) - set(existing_ligand_ids)
        if not ligand_ids:
            self.logger.info(
                "All BRENDA ligand mol files already exist. Skipping download..."
            )
            return
        self.logger.info(f"Downloading {len(ligand_ids)} BRENDA ligand mol files...")
        with tqdm(
            total=len(ligand_ids), desc="Downloading MOL files from BRENDA"
        ) as pbar:
            for ligand_id, success in ThreadPool(2).imap_unordered(
                download_brenda_mol_helper, ligand_ids
            ):
                pbar.update()
                if not success:
                    self.logger.error(f"Failed to download mol for BRENDA:{ligand_id}")

        return existing_ligand_ids

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
        self.logger.info("Exploding experimental PDBs...")
        df = df.explode("pdbs")
        df_without_pdbs = (
            df[df["pdbs"].isna()].copy().reset_index(drop=True).drop(columns=["pdbs"])
        )
        df = df.dropna(subset=["pdbs"]).copy().reset_index(drop=True)

        unique_pdb_ids = df["pdbs"].str.upper().unique().tolist()
        self.logger.info(
            f"Found {len(unique_pdb_ids)} unique PDB IDs to check for existence and download if missing."
        )
        if unique_pdb_ids:
            self.get_ebi_pdb(unique_pdb_ids)
            self.logger.info(
                "Processing experimental structures by removing non-peptide chains and converting to PDB..."
            )
            self.process_experimental_pdbs(pdb_ids=unique_pdb_ids)

        self.logger.info("Checking existence of processed experimental PDB files...")
        processed_pdb_ids = {
            f.stem.upper() for f in self.PROCESSED_EXP_PDB_PATH.glob("*.pdb")
        }
        df["pdb_exists"] = df["pdbs"].str.upper().isin(processed_pdb_ids)

        experimental_df = df[df["pdb_exists"]].copy().drop(columns=["pdb_exists"])
        experimental_df["pdb_source"] = "PDBe"
        experimental_df["pdb_type"] = "experimental"

        missing_pdb_df = pd.concat(
            [
                df[~df["pdb_exists"]].copy().drop(columns=["pdbs", "pdb_exists"]),
                df_without_pdbs,
            ],
            ignore_index=True,
        )
        missing_pdb_df["pdbs"] = missing_pdb_df["uniprot_id"]

        self.logger.info(
            "Attempting to assign AlphaFold predicted PDBs to entries without a processed experimental PDB..."
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

        df = pd.concat([experimental_df, missing_pdb_df], ignore_index=True)

        self.logger.info(
            f"Total entries after assigning experimental and predicted PDBs: {df.shape}"
        )

        return df

    def process_experimental_pdbs(self, pdb_ids=None):
        """
        Process experimental PDB/mmCIF files and save a PDB containing only peptide chains.
        Tries formats in order: updated mmCIF > archive mmCIF > PDB
        Save processed structures to self.PROCESSED_EXP_PDB_PATH with naming scheme: {pdb_id}.pdb

        Args:
            pdb_ids (list, optional): List of PDB IDs to process. If None, processes all available PDBs.
                Example: ["1abc", "2def", "3ghi"]

        Returns:
            None
        """
        from functools import partial
        from multiprocessing import Pool, cpu_count

        # Collect available files for each PDB ID
        pdb_files = {f.stem.upper(): f for f in self.PDB_PATH.glob("*.pdb")}

        mmcif_updated_files = {}
        mmcif_archive_files = {}
        for f in self.MMCIF_PATH.glob("*.cif"):
            if "_updated" in f.stem:
                pdb_id = f.stem.replace("_updated", "").upper()
                mmcif_updated_files[pdb_id] = f
            else:
                pdb_id = f.stem.upper()
                mmcif_archive_files[pdb_id] = f

        # Build priority: updated mmCIF > archive mmCIF > PDB
        available_files = {}
        all_pdb_ids = set(
            list(mmcif_updated_files.keys())
            + list(mmcif_archive_files.keys())
            + list(pdb_files.keys())
        )

        # Filter to requested PDB IDs if provided
        if pdb_ids:
            pdb_ids_upper = {pid.upper() for pid in pdb_ids}
            all_pdb_ids = all_pdb_ids.intersection(pdb_ids_upper)
            if not all_pdb_ids:
                self.logger.warning(
                    f"None of the requested PDB IDs {pdb_ids} found in available files."
                )
                return

        for pdb_id in all_pdb_ids:
            if pdb_id in mmcif_updated_files:
                available_files[pdb_id] = mmcif_updated_files[pdb_id]
            elif pdb_id in mmcif_archive_files:
                available_files[pdb_id] = mmcif_archive_files[pdb_id]
            elif pdb_id in pdb_files:
                available_files[pdb_id] = pdb_files[pdb_id]

        if not available_files:
            self.logger.info("No PDB/mmCIF files to process.")
            return

        processed_pdb_ids = {
            f.stem.upper() for f in self.PROCESSED_EXP_PDB_PATH.glob("*.pdb")
        }
        already_processed = sorted(set(available_files).intersection(processed_pdb_ids))
        remaining_files = {
            pdb_id: file_path
            for pdb_id, file_path in available_files.items()
            if pdb_id not in processed_pdb_ids
        }

        self.logger.info(
            f"{len(already_processed)} experimental structures are already processed."
        )

        if not remaining_files:
            self.logger.info("All requested experimental structures are already processed.")
            return

        self.logger.info(
            f"Processing {len(remaining_files)} experimental PDB/mmCIF files."
        )

        process_func = partial(
            process_single_pdb,
            output_dir=self.PROCESSED_EXP_PDB_PATH,
            logger=self.logger,
        )

        items = list(remaining_files.items())
        successes = 0
        failures = []
        with tqdm(
            total=len(items),
            desc="Processing experimental PDB/mmCIF files",
            leave=False,
        ) as pbar:
            for result in Pool(processes=cpu_count()).imap_unordered(
                process_func, items
            ):
                if result["success"]:
                    successes += 1
                else:
                    failures.append(result)
                pbar.update()

        self.logger.info(
            f"Processed {successes}/{len(items)} experimental structures into peptide-only PDBs."
        )
        if failures:
            self.logger.warning(
                f"{len(failures)} experimental structures could not be converted to peptide-only PDBs."
            )


def process_single_pdb(pdb_id_and_file, output_dir, logger):
    """
    Save a peptide-only PDB for a PDB/mmCIF input file.
    Non-peptide chains are removed entirely before writing the output PDB.

    Args:
        pdb_id_and_file (tuple): Tuple of (pdb_id, file_path) where file_path can be PDB or mmCIF.
        output_dir (Path): Directory to save processed experimental PDB files.
        logger: Logger instance.

    Returns:
        dict: Processing result with success flag and optional error message.
    """
    from Bio import PDB
    from pathlib import Path

    pdb_id, file_path = pdb_id_and_file
    file_path = Path(file_path)
    output_path = Path(output_dir) / f"{pdb_id}.pdb"

    if output_path.exists():
        return {"pdb_id": pdb_id, "success": True, "error": None}

    class PeptideOnlySelect(PDB.Select):
        def __init__(self, peptide_chain_keys):
            self.peptide_chain_keys = peptide_chain_keys

        def accept_model(self, model):
            model_id = model.get_id()
            return int(
                any(
                    (model_id, chain.get_id()) in self.peptide_chain_keys
                    for chain in model.get_chains()
                )
            )

        def accept_chain(self, chain):
            model = chain.get_parent()
            return int((model.get_id(), chain.get_id()) in self.peptide_chain_keys)

        def accept_residue(self, residue):
            chain = residue.get_parent()
            model = chain.get_parent()
            return int(
                residue.id[0] == " "
                and (model.get_id(), chain.get_id()) in self.peptide_chain_keys
            )

    io = PDB.PDBIO()

    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return {
            "pdb_id": pdb_id,
            "success": False,
            "error": f"File does not exist: {file_path}",
        }

    if file_path.suffix.lower() == ".cif":
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        structure = parser.get_structure(pdb_id, str(file_path))
        ppb = PDB.PPBuilder()
        peptide_chain_keys = set()

        for model in structure:
            model_id = model.get_id()
            for chain in model.get_chains():
                sequence = "".join(
                    str(pp.get_sequence()) for pp in ppb.build_peptides(chain)
                )
                if sequence:
                    peptide_chain_keys.add((model_id, chain.get_id()))

        if not peptide_chain_keys:
            reason = "No peptide chains found"
            logger.warning(f"Skipping {pdb_id}: {reason}")
            output_path.unlink(missing_ok=True)
            return {"pdb_id": pdb_id, "success": False, "error": reason}

        io.set_structure(structure)
        io.save(str(output_path), select=PeptideOnlySelect(peptide_chain_keys))
        logger.debug(f"Saved peptide-only experimental PDB to {output_path}")
        return {"pdb_id": pdb_id, "success": True, "error": None}

    except Exception as e:
        error_msg = str(e)
        logger.warning(
            f"Failed to convert experimental structure {pdb_id} from {file_path.name} to peptide-only PDB: {error_msg}"
        )
        output_path.unlink(missing_ok=True)
        return {"pdb_id": pdb_id, "success": False, "error": error_msg}
