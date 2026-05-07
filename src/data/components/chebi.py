import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import logging
import requests
from pathlib import Path
from rdkit import RDLogger
from tqdm.auto import tqdm
from hydra import initialize, compose
from multiprocessing.pool import ThreadPool
from rdkit.Chem import (
    MolToInchi,
    MolToSmiles,
    MolToInchiKey,
    MolFromMolFile,
)


RDLogger.DisableLog("rdApp.*")
logging.getLogger("zeep.wsdl.bindings.soap").setLevel(logging.ERROR)

from src.utils.tqdmlogger import TqdmLogger
from src.utils.chem_utils import ChemUtils

chem_utils = ChemUtils()
load_json = chem_utils.load_json
save_json = chem_utils.save_json
canonicalize_smiles = chem_utils.canonicalize_smiles


class ChEBIDatasetUtils:
    def __init__(self, config_path="../../../configs", config_name="data_processing"):
        with initialize(version_base="1.3", config_path=config_path):
            cfg = compose(config_name=config_name)

        self.CHEBI_COMPOUNDS_API_URL = cfg.chebi.chebi_compounds_api_url
        self.CHEBI_MOL_FILE_API_URL = cfg.chebi.chebi_mol_file_api_url

        LOG_PATH = Path(cfg.chebi.log_dir)
        self.MOL_PATH = {k: Path(v) for k, v in cfg.chebi.chebi_mol_path.items()}
        self.CHEBI_DOWNLOAD_PATH = Path(cfg.chebi.chebi_raw_path)

        for path in [LOG_PATH, self.CHEBI_DOWNLOAD_PATH] + list(self.MOL_PATH.values()):
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name="chebi.log"
        ).get_logger()

    def download_chebi_mol_parallel(self, metabolite_id_pairs, verbose=True):
        def download_chebi_mol(chebi_id_pair):
            standardized_chebi_id, primary_chebi_id = chebi_id_pair

            mol_file_path = self.MOL_PATH["2D"] / f"{standardized_chebi_id}.mol"
            if mol_file_path.exists():
                return

            try:
                mol_response = requests.get(
                    self.CHEBI_MOL_FILE_API_URL.format(
                        primary_chebi_id=primary_chebi_id
                    )
                )
                mol_response.raise_for_status()
                with open(mol_file_path, "w") as f:
                    f.write(mol_response.text)
            except:
                self.logger.error(
                    f"Failed to fetch MOL files for {standardized_chebi_id}"
                )
                self.logger.warning(
                    f"Will try to generate {standardized_chebi_id} MOL file from SMILES/InChI using RDKit"
                )

        with tqdm(
            total=len(metabolite_id_pairs),
            desc="Downloading ChEBI MOL files",
            disable=not verbose,
        ) as pbar:
            for _ in ThreadPool(20).imap_unordered(
                download_chebi_mol, metabolite_id_pairs
            ):
                pbar.update()

    def download_chebi_json(self, metabolites, verbose=True):
        """
        Download complete ChEBI entity data for a list of metabolites and save JSON + structure files.

        For each metabolite this function:
        - Queries the ChEBI SOAP webservice in batches (50 IDs per request).
        - Saves a JSON representation of the ChEBI entity to CHEBI_DOWNLOAD_PATH.
        - Saves 2D (.mol) and 3D (.sdf) structure files to the corresponding paths in MOL_PATH.
        - Generates SMILES/InChI/InChIKey from saved MOL files when needed.
        - Attempts to standardize SMILES using molvs.
        - Logs progress, warnings and errors via the provided logger.

        Args:
            metabolites (iterable[str]): Iterable of metabolite identifiers to download (e.g. "CHEBI:12345" or similar).
            CHEBI_DOWNLOAD_PATH (pathlib.Path): Directory where ChEBI JSON files will be written.
            MOL_PATH (dict): Mapping with keys "2D" and "3D" to pathlib.Path directories for structure files.
            logger (logging.Logger): Logger used for informational, debug and error messages.
            verbose (bool, optional): If True, show progress and extra log messages. Defaults to True.

        Returns:
            None

        Side effects:
            Creates/updates files under CHEBI_DOWNLOAD_PATH and the directories in MOL_PATH. Errors are logged,
            and failed items are skipped; the function does not raise on per-entity failures.
        """

        pending_metabolites = list(
            {f"CHEBI:{m[6:]}" for m in metabolites}
            - {m.stem for m in self.CHEBI_DOWNLOAD_PATH.glob("*.json")}
        )
        if not pending_metabolites:
            if verbose:
                self.logger.info(
                    "All metabolites have already been downloaded from ChEBI."
                )
            metabolite_id_pairs = []
            for metabolite in metabolites:
                json_path = self.CHEBI_DOWNLOAD_PATH / f"{metabolite}.json"
                entity_dict = load_json(json_path)
                mol_file_path = self.MOL_PATH["2D"] / f"{metabolite}.mol"
                if not mol_file_path.exists():
                    metabolite_id_pairs.append(
                        (entity_dict["chebi_id"], entity_dict["primary_chebi_id"])
                    )

            self.download_chebi_mol_parallel(metabolite_id_pairs, verbose=verbose)
            self.finalize_chebi_jsons(metabolites, verbose=verbose)
            return

        pending_metabolites = [
            pending_metabolites[i : i + 100]
            for i in range(0, len(pending_metabolites), 100)
        ]
        metabolite_id_pairs = []

        for metabolite in tqdm(
            pending_metabolites,
            desc="Downloading ChEBI JSON files",
            disable=not verbose,
        ):
            try:
                response = requests.post(
                    self.CHEBI_COMPOUNDS_API_URL,
                    json={"chebi_ids": metabolite},
                )
                response.raise_for_status()
                response = response.json()
            except Exception as e:
                self.logger.error(
                    f"Failed to fetch ChEBI entities for {metabolite}: {e}"
                )
                continue

            for chebi_id in response:
                entity = response[chebi_id]
                json_path = (
                    self.CHEBI_DOWNLOAD_PATH / f"{entity['standardized_chebi_id']}.json"
                )
                if json_path.exists():
                    if verbose:
                        self.logger.debug(
                            f"Skipping {entity['standardized_chebi_id']}: already exists."
                        )
                    continue

                try:
                    inchi = entity["data"]["default_structure"]["standard_inchi"]
                except:
                    inchi = None
                    if verbose:
                        self.logger.warning(
                            f"No InChI found for {entity['standardized_chebi_id']}"
                        )
                try:
                    inchi_key = entity["data"]["default_structure"][
                        "standard_inchi_key"
                    ]
                except:
                    inchi_key = None
                    if verbose:
                        self.logger.warning(
                            f"No InChI Key found for {entity['standardized_chebi_id']}"
                        )
                try:
                    smiles = canonicalize_smiles(
                        entity["data"]["default_structure"]["smiles"]
                    )
                except:
                    smiles = None
                    if verbose:
                        self.logger.warning(
                            f"No SMILES found for {entity['standardized_chebi_id']}"
                        )
                try:
                    charge = entity["data"]["chemical_data"]["charge"]
                except:
                    charge = None
                    if verbose:
                        self.logger.warning(
                            f"No charge found for {entity['standardized_chebi_id']}"
                        )
                try:
                    mass = entity["data"]["chemical_data"]["mass"]
                except:
                    mass = None
                    if verbose:
                        self.logger.warning(
                            f"No mass found for {entity['standardized_chebi_id']}"
                        )
                try:
                    monoisotopic_mass = entity["data"]["chemical_data"][
                        "monoisotopic_mass"
                    ]
                except:
                    monoisotopic_mass = None
                    if verbose:
                        self.logger.warning(
                            f"No monoisotopic mass found for {entity['standardized_chebi_id']}"
                        )
                try:
                    secondary_ids = entity["data"]["secondary_ids"]
                except:
                    secondary_ids = []
                    if verbose:
                        self.logger.warning(
                            f"No secondary IDs found for {entity['standardized_chebi_id']}"
                        )
                try:
                    synonyms = [i["name"] for i in entity["data"]["names"]["SYNONYM"]]
                except:
                    synonyms = []
                    if verbose:
                        self.logger.warning(
                            f"No synonyms found for {entity['standardized_chebi_id']}"
                        )
                try:
                    iupac_names = [
                        i["name"] for i in entity["data"]["names"]["IUPAC NAME"]
                    ]
                except:
                    iupac_names = []
                    if verbose:
                        self.logger.warning(
                            f"No IUPAC names found for {entity['standardized_chebi_id']}"
                        )
                try:
                    formulae = [entity["data"]["chemical_data"]["formula"]]
                except:
                    formulae = []
                    if verbose:
                        self.logger.warning(
                            f"No formulae found for {entity['standardized_chebi_id']}"
                        )
                try:
                    xrefs = {
                        i["source_name"]: i["accession_number"]
                        for i in entity["data"]["database_accessions"]["MANUAL_X_REF"]
                    }
                except:
                    xrefs = {}
                    if verbose:
                        self.logger.warning(
                            f"No cross-references found for {entity['standardized_chebi_id']}"
                        )
                try:
                    citations = {
                        i["source_name"]: i["accession_number"]
                        for i in entity["data"]["database_accessions"]["CITATION"]
                    }
                except:
                    citations = []
                    if verbose:
                        self.logger.warning(
                            f"No citations found for {entity['standardized_chebi_id']}"
                        )

                entity_dict = {
                    "chebi_id": entity["standardized_chebi_id"],
                    "primary_chebi_id": entity["primary_chebi_id"],
                    "name": entity["data"]["name"],
                    "definition": entity["data"]["definition"],
                    "inchi": inchi,
                    "inchi_key": inchi_key,
                    "smiles": smiles,
                    "charge": charge,
                    "mass": mass,
                    "monoisotopic_mass": monoisotopic_mass,
                    "secondary_ids": secondary_ids,
                    "synonyms": synonyms,
                    "iupac_names": iupac_names,
                    "formulae": formulae,
                    "xrefs": xrefs,
                    "citations": citations,
                    "structure_2d": None,
                    "structure_3d": None,
                }
                try:
                    save_json(entity_dict, json_path)
                    self.logger.info(
                        f"Saved ChEBI JSON for {entity['standardized_chebi_id']}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to save JSON for {entity['standardized_chebi_id']}: {e}"
                    )
                    continue

                mol_file_path = (
                    self.MOL_PATH["2D"] / f"{entity['standardized_chebi_id']}.mol"
                )
                if not mol_file_path.exists():
                    metabolite_id_pairs.append(
                        (
                            entity["standardized_chebi_id"],
                            entity["primary_chebi_id"],
                        )
                    )

        self.download_chebi_mol_parallel(metabolite_id_pairs, verbose=verbose)
        self.finalize_chebi_jsons(metabolites, verbose=verbose)

    def finalize_chebi_jsons(self, metabolites, verbose=True):
        for metabolite in tqdm(
            metabolites, desc="Finalizing ChEBI JSON files", disable=not verbose
        ):
            json_path = self.CHEBI_DOWNLOAD_PATH / f"{metabolite}.json"
            if not json_path.exists():
                if verbose:
                    self.logger.warning(
                        f"ChEBI JSON file not found for {metabolite}, skipping finalization."
                    )
                continue
            mol_file_path = self.MOL_PATH["2D"] / f"{metabolite}.mol"
            if not mol_file_path.exists():
                if verbose:
                    self.logger.warning(
                        f"ChEBI MOL file not found for {metabolite}, cannot finalize JSON."
                    )
                continue

            entity_dict = load_json(json_path)
            if not entity_dict["smiles"]:
                mol = MolFromMolFile(str(mol_file_path))
                if mol:
                    entity_dict["smiles"] = canonicalize_smiles(
                        MolToSmiles(mol, canonical=True)
                    )
                    entity_dict["inchi"] = MolToInchi(mol)
                    entity_dict["inchi_key"] = MolToInchiKey(mol)
                    if verbose:
                        self.logger.debug(
                            f"Generated SMILES/InChI for {entity_dict['chebi_id']} from MOL file."
                        )
                else:
                    if verbose:
                        self.logger.warning(
                            f"Could not parse MOL file for {entity_dict['chebi_id']} to generate SMILES/InChI."
                        )

            try:
                save_json(entity_dict, json_path)
            except Exception as e:
                if verbose:
                    self.logger.error(
                        f"Failed to finalize JSON for {entity_dict['chebi_id']}: {e}"
                    )

    def parse_chebi_jsons(self, chebi_ids, return_as_dict=True, verbose=True):
        """
        Parse saved ChEBI JSON files and optionally restore stored structure files.

        For each provided chebi_id this function:
        - Loads the corresponding JSON file from CHEBI_DOWNLOAD_PATH.
        - If the JSON contains embedded "structure_2d" or "structure_3d" data and the
        corresponding files are missing, writes the .mol/.sdf files to MOL_PATH.
        - Removes "structure_2d" and "structure_3d" keys from the returned metabolite
        representation (they are restored to files instead).
        - Collects results either into a dict keyed by chebi_id or into a list.

        Args:
            chebi_ids (iterable[str]): Iterable of ChEBI IDs (matching the JSON filenames, e.g. "CHEBI:12345").
            CHEBI_DOWNLOAD_PATH (pathlib.Path): Directory containing ChEBI JSON files.
            MOL_PATH (dict): Mapping with keys "2D" and "3D" to pathlib.Path directories for structure files.
            logger (logging.Logger): Logger used for informational, debug and error messages.
            return_as_dict (bool, optional): If True, return a dict mapping chebi_id -> metabolite dict.
                                            If False, return a list of metabolite dicts. Defaults to True.
            verbose (bool, optional): If True, show debug/warning logs. Defaults to True.

        Returns:
            dict or list: A dictionary mapping chebi_id to metabolite data if return_as_dict is True,
                        otherwise a list of metabolite data dictionaries.

        Side effects:
            May write .mol and .sdf files to MOL_PATH when structure data is embedded in the JSON.
            Errors loading/parsing individual JSON files are logged and those entries are skipped.
        """
        metabolites = {} if return_as_dict else []
        for chebi_id in chebi_ids:
            file = self.CHEBI_DOWNLOAD_PATH / f"{chebi_id}.json"
            if not file.exists():
                if verbose:
                    self.logger.warning(
                        f"ChEBI JSON file not found for {chebi_id}, skipping."
                    )
                continue
            try:
                metabolite = load_json(file)
            except Exception as e:
                self.logger.error(f"Failed to load JSON for {chebi_id}: {e}")
                continue

            if (
                metabolite["structure_2d"]
                and not (self.MOL_PATH["2D"] / f"{file.stem}.mol").exists()
            ):
                try:
                    with open(self.MOL_PATH["2D"] / f"{file.stem}.mol", "w") as f:
                        f.write(metabolite["structure_2d"])
                    if verbose:
                        self.logger.debug(f"Restored 2D structure for {chebi_id}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to restore 2D structure for {chebi_id}: {e}"
                    )
            del metabolite["structure_2d"]

            if (
                metabolite["structure_3d"]
                and not (self.MOL_PATH["3D"] / f"{file.stem}.sdf").exists()
            ):
                try:
                    with open(self.MOL_PATH["3D"] / f"{file.stem}.sdf", "w") as f:
                        f.write(metabolite["structure_3d"])
                    if verbose:
                        self.logger.debug(f"Restored 3D structure for {chebi_id}")
                except Exception as e:
                    self.logger.error(
                        f"Failed to restore 3D structure for {chebi_id}: {e}"
                    )
            del metabolite["structure_3d"]

            if return_as_dict:
                metabolites[chebi_id] = metabolite
            else:
                metabolites.append(metabolite)
        return metabolites
