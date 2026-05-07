import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import io
import re
import time
import urllib3
import requests
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from rdkit import RDLogger
from datetime import datetime
from omegaconf import DictConfig
from hydra import initialize, compose
from multiprocessing.pool import ThreadPool
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import (
    MolFromInchi,
    MolToInchi,
    MolToSmiles,
    MolFromSmiles,
    MolToMolFile,
)

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger
from src.data.components.chebi import ChEBIDatasetUtils
from src.data.components.kegg import KEGGDatasetBuilder

chebi_utils = ChEBIDatasetUtils()
parse_chebi_jsons = chebi_utils.parse_chebi_jsons
download_chebi_json = chebi_utils.download_chebi_json

chem_utils = ChemUtils()
load_json = chem_utils.load_json
save_json = chem_utils.save_json
get_mol_kegg = chem_utils.get_mol_kegg
canonicalize_smiles = chem_utils.canonicalize_smiles
get_pubchem_compound = chem_utils.get_pubchem_compound
normalize_ec_collection = chem_utils.normalize_ec_collection


class SABIODatasetBuilder:
    def __init__(self, cfg: DictConfig):
        self.ENZYME_DAT_URL = cfg.sabio_rk.enzyme_dat_url
        self.SABIO_RELEASE_URL = cfg.sabio_rk.sabio_release_url
        self.SABIO_KINETIC_PARAMS_URL = cfg.sabio_rk.sabio_kinetic_param_download_url
        self.SABIO_ENTRYID_LIST_URL = cfg.sabio_rk.sabio_entryid_list_url
        self.SABIO_REACTIONPARTICIPANT_QUERY_URL = (
            cfg.sabio_rk.sabio_reaction_participant_query_url
        )
        self.SABIO_ENTRYID_QUERY_URL = cfg.sabio_rk.sabio_entryid_query_url

        self.SABIO_DATA_PATH = Path(cfg.sabio_rk.sabio_data_path)
        self.SABIO_DOWNLOAD_PATH = Path(cfg.sabio_rk.sabio_raw_path)
        self.SABIO_ENTRIES_PATH = Path(cfg.sabio_rk.sabio_entries_path)
        self.SABIO_COMPOUNDS_PATH = Path(cfg.sabio_rk.sabio_compounds_path)
        self.SABIO_REACTIONS_PATH = Path(cfg.sabio_rk.sabio_reactions_path)
        self.SABIO_KINETIC_LAWS_PATH = Path(cfg.sabio_rk.sabio_kinetic_laws_path)

        self.MOL_PATH = Path(cfg.sabio_rk.mol_path)
        self.KEGG_COMPOUND_PATH = Path(cfg.sabio_rk.kegg_compound_path)

        self.ENZYME_DAT_PATH = Path(cfg.sabio_rk.enzyme_dat_file)
        self.SABIO_VERSION_FILE = Path(cfg.sabio_rk.sabio_version_file)

        self.SABIO_ENTRIES_PARQUET_FILE_PATH = Path(
            cfg.sabio_rk.sabio_entries_parquet_file_path
        )
        self.SABIO_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.sabio_rk.sabio_reactions_parquet_file_path
        )
        self.SABIO_COMPOUNDS_PARQUET_FILE_PATH = Path(
            cfg.sabio_rk.sabio_compounds_parquet_file_path
        )
        self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH = Path(
            cfg.sabio_rk.sabio_kinetic_params_parquet_file_path
        )

        kegg_builder = KEGGDatasetBuilder(cfg)
        self.download_kegg_compound = kegg_builder.download_kegg_compound

        LOG_PATH = Path(cfg.sabio_rk.log_dir)

        for path in [
            LOG_PATH,
            self.MOL_PATH,
            self.SABIO_DATA_PATH,
            self.SABIO_DOWNLOAD_PATH,
            self.SABIO_ENTRIES_PATH,
            self.SABIO_COMPOUNDS_PATH,
            self.SABIO_REACTIONS_PATH,
            self.SABIO_KINETIC_LAWS_PATH,
            self.KEGG_COMPOUND_PATH,
            self.ENZYME_DAT_PATH.parent,
            self.SABIO_VERSION_FILE.parent,
            self.SABIO_ENTRIES_PARQUET_FILE_PATH.parent,
            self.SABIO_REACTIONS_PARQUET_FILE_PATH.parent,
            self.SABIO_COMPOUNDS_PARQUET_FILE_PATH.parent,
            self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.sabio_rk.log_file_name
        ).get_logger()

    def get_latest_sabio_version(self):
        """
        Fetch the latest SABIO version from the server.

        Returns:
            datetime: The latest SABIO version as a datetime object, or None if the request fails.
        """
        try:
            response = requests.get(url=self.SABIO_RELEASE_URL)
            response.raise_for_status()
            return datetime.strptime(response.text.strip(), "%Y-%m-%d")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch SABIO version: {e}")
            return None

    def current_sabio_version(self):
        """
        Get the current SABIO version from the local file.

        Returns:
            datetime: The current SABIO version as a datetime object, or None if the file doesn't exist.
        """
        if self.SABIO_VERSION_FILE.exists():
            with open(self.SABIO_VERSION_FILE, "r") as f:
                return datetime.strptime(f.read().strip(), "%Y-%m-%d")
        return None

    def update_sabio_version(self):
        """
        Update the local SABIO version file with the latest version.
        """
        latest_version = self.get_latest_sabio_version()
        if latest_version:
            with open(self.SABIO_VERSION_FILE, "w") as f:
                f.write(latest_version.strftime("%Y-%m-%d"))
            self.logger.info(f"Updated SABIO version to {latest_version}")
        else:
            self.logger.error("Failed to update SABIO version")

    def download_enzyme_dat(self):
        """
        Download the latest enzyme.dat file using curl.
        """
        self.logger.info(
            f"Downloading the latest enzyme.dat from {self.ENZYME_DAT_URL}..."
        )
        try:
            subprocess.run(
                ["curl", "-o", str(self.ENZYME_DAT_PATH), "-#L", self.ENZYME_DAT_URL],
                check=True,  # Raise an exception if the command fails
            )
            self.logger.info(f"Downloaded enzyme.dat to {self.ENZYME_DAT_PATH}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download enzyme.dat: {e}")
            raise

    def download_SABIO_data(self, url, query, tries=2):
        for attempt in range(tries):
            try:
                response = requests.post(url=url, params=query, verify=False)
                response.raise_for_status()
                if not response.text:
                    self.logger.error(
                        f"Empty response for {query.get('q', query)}"
                    )
                    return pd.DataFrame()
                return pd.read_csv(io.StringIO(response.text), sep="\t", dtype=str)
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt+1}/{tries} failed for query {query}: {e}"
                )
                time.sleep(2)
        self.logger.error(f"Failed to get response from server for {query}")
        return pd.DataFrame()

    def download_SABIO_entries(self, ids, path):
        def download_helper(ids, path):
            valid_fields = [
                "EntryID",
                "KeggReactionID",
                "SabioReactionID",
                "ECNumber",
                "UniProtKB_AC",
                "PubMedID",
                "EnzymeType",
                "ReactomeReactionID",
            ]
            query = {"entryIDs[]": ids, "format": "tsv", "fields[]": valid_fields}
            entries = self.download_SABIO_data(
                url=self.SABIO_ENTRYID_LIST_URL, query=query
            )
            if entries.empty:
                return
            entries = entries.replace({np.nan: None}).to_dict(orient="records")
            for entry in entries:
                save_json(entry, path / f"{entry['EntryID']}.json")

        ids = list(set(ids) - {m.stem for m in path.glob("*.json")})
        if not ids:
            self.logger.info("All entries have already been downloaded from SABIO-RK.")
            return
        ids = [ids[i : i + 250] for i in range(0, len(ids), 250)]
        for _ in tqdm(
            ThreadPool(10).imap_unordered(lambda ids: download_helper(ids, path), ids),
            total=len(ids),
            desc="Downloading SABIO entries",
        ):
            pass

    def parse_sabio_compound(self, compound, path):
        sabio_id = str(compound["SabioCompoundID"])
        if (path / f"{sabio_id}.json").exists():
            return load_json(path / f"{sabio_id}.json")

        name = compound["Name"]
        chebi_ids = (
            sorted(
                {f"CHEBI:{id}" for id in compound["ChebiID"].split(" ") if id.isdigit()}
            )
            if compound["ChebiID"]
            else None
        )
        pubchem_ids = (
            sorted({id for id in compound["PubChemID"].split(" ") if id.isdigit()})
            if compound["PubChemID"]
            else None
        )
        kegg_ids = (
            sorted(
                {
                    id
                    for id in compound["KeggCompoundID"].split(" ")
                    if re.match(r"^[CDG]\d{5}$", id)
                }
            )
            if compound["KeggCompoundID"]
            else None
        )

        smiles, inchi, formula = compound["Smiles"], compound["InChI"], None
        if inchi or smiles:
            mol = MolFromInchi(inchi) if inchi else MolFromSmiles(smiles)
            if mol:
                if not smiles:
                    smiles = canonicalize_smiles(MolToSmiles(mol))
                if not inchi:
                    inchi = MolToInchi(mol)
                formula = CalcMolFormula(mol)
        else:
            if chebi_ids:
                download_chebi_json(chebi_ids, verbose=False)
                for elem in parse_chebi_jsons(chebi_ids, return_as_dict=False):
                    if elem["smiles"]:
                        smiles, inchi, formula = (
                            elem["smiles"],
                            elem["inchi"],
                            elem["formulae"],
                        )
                        break
            if kegg_ids:
                for elem in self.download_kegg_compound(kegg_ids, verbose=False):
                    if elem["smiles"]:
                        smiles, inchi, formula = (
                            elem["smiles"],
                            MolToInchi(MolFromSmiles(elem["smiles"])),
                            elem["formula"],
                        )
                        break
            if pubchem_ids:
                pubchem_data = get_pubchem_compound(pubchem_ids)
                for id in pubchem_ids:
                    if id in pubchem_data:
                        elem = pubchem_data[id]
                        if elem["smiles"]:
                            smiles, inchi, formula = (
                                elem["smiles"],
                                MolToInchi(MolFromSmiles(elem["smiles"])),
                                elem["formula"],
                            )
                            break

        compound_json = {
            "sabio_id": sabio_id,
            "name": name,
            "formula": formula,
            "chebi_id": chebi_ids,
            "pubchem_id": pubchem_ids,
            "kegg_id": kegg_ids,
            "inchi": inchi,
            "smiles": smiles,
        }

        mol_path = self.MOL_PATH / f"SABIO_{sabio_id}.mol"
        if not mol_path.exists() and smiles:
            mol = MolFromSmiles(smiles)
            if mol:
                MolToMolFile(mol, str(mol_path))

        save_json(compound_json, path / f"{sabio_id}.json")
        return compound_json

    def download_SABIO_reactions(self, ids):
        def download_helper(id):
            valid_fields = [
                "Name",
                "Role",
                "SabioCompoundID",
                "ChebiID",
                "PubChemID",
                "KeggCompoundID",
                "InChI",
                "Smiles",
            ]
            query = {"SabioReactionID": f"{id}", "fields[]": valid_fields}
            participants = self.download_SABIO_data(
                url=self.SABIO_REACTIONPARTICIPANT_QUERY_URL, query=query
            )
            if participants.empty:
                return
            participants = participants.replace({np.nan: None}).to_dict(orient="records")
            substrate = {key: [] for key in ["ids", "name", "formula", "smiles"]}
            product = {key: [] for key in ["ids", "name", "formula", "smiles"]}

            for elem in participants:
                compound = self.parse_sabio_compound(
                    compound=elem, path=self.SABIO_COMPOUNDS_PATH
                )
                metabolite = (
                    substrate
                    if elem["Role"] == "Substrate"
                    else product if elem["Role"] == "Product" else None
                )
                metabolite["ids"].append(compound["sabio_id"])
                metabolite["name"].append(compound["name"])
                metabolite["smiles"].append(compound["smiles"])
                metabolite["formula"].append(
                    compound["formula"][0]
                    if isinstance(compound["formula"], list)
                    else compound["formula"] or "Unknown"
                )

            definition = equation = rxn_smiles = None
            if None not in substrate["smiles"] and None not in product["smiles"]:
                definition = (
                    f"{' + '.join(substrate['name'])} <=> {' + '.join(product['name'])}"
                )
                equation = f"{' + '.join(substrate['formula'])} <=> {' + '.join(product['formula'])}"
                rxn_smiles = (
                    f"{'.'.join(substrate['smiles'])}>>{'.'.join(product['smiles'])}"
                )

            save_json(
                {
                    "sabio_id": id,
                    "definition": definition,
                    "equation": equation,
                    "substrate": substrate["ids"],
                    "product": product["ids"],
                    "rxn_smiles": rxn_smiles,
                },
                self.SABIO_REACTIONS_PATH / f"{id}.json",
            )

        ids = list(
            set(ids) - {m.stem for m in self.SABIO_REACTIONS_PATH.glob("*.json")}
        )
        if not ids:
            self.logger.info(
                "All reactions have already been downloaded from SABIO-RK."
            )
            return
        for _ in tqdm(
            ThreadPool(10).imap_unordered(lambda id: download_helper(id), ids),
            total=len(ids),
            desc="Downloading SABIO reactions",
        ):
            pass

    def download_SABIO_kinetic_data(self, url, query, path):
        """
        Download data from the SABIO-RK database using the provided URL and query parameters.

        Args:
            url (str): The URL to fetch data from.
            query (dict): The query parameters for the request.
            path (Path): The file path to save the downloaded data.
        """
        if path.exists():
            try:
                return pd.read_csv(path, sep="\t", dtype=str)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        df = self.download_SABIO_data(url, query)

        if df is None or df.empty:
            self.logger.warning(f"No data for {query['q']}")
            # Cache negative lookups so future runs skip repeated API calls.
            path.touch()
            return pd.DataFrame()
        else:
            self.logger.info(f"Downloaded {len(df)} kinetic laws for {query['q']}")

        df.to_csv(path, sep="\t", index=False)
        return df

    def download_SABIO_kinetic_params_parallel(self, args):
        """
        Download kinetic parameters in parallel using multiple threads.

        Args:
            args (list): A list of EC numbers to download data for.
        """
        results = []

        def download_SABIO_kinetic_params(id):
            """
            Download kinetic parameters for a specific EC number.

            Args:
                id (str): The EC number to download data for.
            """
            valid_fields = [
                "EntryID",
                "Substrate",
                "EnzymeType",
                "PubMedID",
                "Organism",
                "UniprotID",
                "ECNumber",
                "Parameter",
                "Temperature",
            ]
            results.append(
                self.download_SABIO_kinetic_data(
                    url=self.SABIO_KINETIC_PARAMS_URL,
                    query={"fields[]": valid_fields, "q": f"ECNumber:{id}"},
                    path=(self.SABIO_KINETIC_LAWS_PATH / f"{id}.tsv"),
                )
            )

        for _ in tqdm(
            ThreadPool(10).imap_unordered(download_SABIO_kinetic_params, args),
            total=len(args),
            desc="Downloading SABIO kinetic parameters",
        ):
            pass
        return pd.concat(results, ignore_index=True)

    def extract_kinetic_law(self, df, kinetic_law):
        """
        Extract all the kinetic laws corresponding to the given type (kcat, Km, Ki) from the DataFrame.

        Args:
            df (pd.DataFrame): The master DataFrame containing kinetic laws data.
            kinetic_law (str): The type of kinetic law to extract (e.g., "kcat", "Km", "Ki").
            SABIO_compounds (dict, optional): A dictionary mapping substrate names to their SMILES strings. If provided, it will be used to add SMILES information to the extracted kinetic laws.

        Returns:
            pd.DataFrame: A cleaned and processed DataFrame containing the extracted kinetic laws.
        """
        unit = "s^(-1)" if kinetic_law == "kcat" else "M"

        self.logger.info(f"{kinetic_law}: Extracting kinetic law values...")
        try:
            # Filter rows where the parameter type matches the given kinetic law
            kinetic_laws = df[df["parameter.type"] == kinetic_law].copy()

            # Drop rows with incorrect units
            kinetic_laws = kinetic_laws[kinetic_laws["parameter.unit"] == unit]

            # Drop rows with missing start values or Uniprot IDs
            kinetic_laws = (
                kinetic_laws.dropna(subset=["UniprotID"])
                .dropna(
                    subset=["parameter.startValue", "parameter.endValue"], how="all"
                )
                .reset_index(drop=True)
            )
            # If end values are present, replace start values with end values
            kinetic_laws.loc[
                kinetic_laws["parameter.endValue"].notna(), "parameter.startValue"
            ] = kinetic_laws["parameter.endValue"]
        except Exception as e:
            self.logger.error(
                f"{kinetic_law}: Error extracting {kinetic_law} values: {e}"
            )
            raise e
        self.logger.info(f"{kinetic_law}: Extracting {kinetic_law} values completed.")

        if kinetic_law == "kcat":
            kinetic_laws["parameter.associatedSpecies"] = kinetic_laws[
                "Substrate"
            ].str.split(";")
            kinetic_laws = kinetic_laws.explode(
                "parameter.associatedSpecies"
            ).reset_index(drop=True)

        self.logger.info(f"{kinetic_law}: Cleaning kinetic laws...")
        try:
            # Drop unnecessary columns
            kinetic_laws.drop(
                columns=[
                    "Substrate",
                    "PubMedID",
                    "parameter.name",
                    "parameter.type",
                    "parameter.endValue",
                    "parameter.standardDeviation",
                    "parameter.unit",
                ],
                inplace=True,
            )

            # Convert "molar (M)" to "millimolar (mM)" for Km and Ki
            if kinetic_law != "kcat":
                kinetic_laws["parameter.startValue"] = (
                    kinetic_laws["parameter.startValue"].astype(float) * 1000
                )

            # Rename columns for clarity
            kinetic_laws.rename(
                columns={
                    "parameter.startValue": "value",
                    "parameter.associatedSpecies": "substrate",
                    "EnzymeType": "metadata",
                },
                inplace=True,
            )

            kinetic_laws["value"] = kinetic_laws["value"].astype(float)
            kinetic_laws["unit"] = (
                "millimolar (mM)" if kinetic_law != "kcat" else "s^(-1)"
            )
            kinetic_laws["value_type"] = kinetic_law
            kinetic_laws.loc[kinetic_laws["Temperature"] == "-", "Temperature"] = np.nan
            kinetic_laws["Temperature"] = kinetic_laws["Temperature"].astype(float)
        except Exception as e:
            self.logger.error(f"{kinetic_law}: Error cleaning kinetic laws: {e}")
            raise e
        self.logger.info(f"{kinetic_law}: Kinetic laws cleaning completed.")

        self.logger.info(f"{kinetic_law}: Annotating enzyme types and mutations...")
        try:
            # Annotate enzyme types as wildtype or mutant based on metadata
            kinetic_laws["enzymeType"] = np.where(
                kinetic_laws["metadata"].str.contains("wild", case=False, na=False),
                "wildtype",
                "mutant",
            )
            # Extract HGVS mutations for mutant enzymes
            mutations = kinetic_laws.loc[
                kinetic_laws["enzymeType"] == "mutant", "metadata"
            ].str.findall(r"\b[A-Z]\d+[A-Z]\b")
            mutations = mutations.apply(lambda x: x if x else np.nan).apply(
                lambda x: tuple(x) if isinstance(x, list) else x
            )
            kinetic_laws.loc[kinetic_laws["enzymeType"] == "mutant", "mutations"] = (
                mutations
            )
            # Remove mutant rows with where HGVS mutations could not be extracted
            kinetic_laws = kinetic_laws[
                ~(
                    (kinetic_laws["mutations"].isna())
                    & (kinetic_laws["enzymeType"] == "mutant")
                )
            ].reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"{kinetic_law}: Error annotating enzyme types and mutations: {e}"
            )
            raise e
        self.logger.info(
            f"{kinetic_law}: Enzyme types and mutations annotation completed."
        )

        self.logger.info(f"{kinetic_law}: Cleaning Uniprot IDs for kinetic law...")
        try:
            # Split Uniprot IDs into unique tuples
            kinetic_laws["UniprotID"] = kinetic_laws["UniprotID"].apply(
                lambda x: tuple(set(x.split(" ")))
            )
            # Explode Uniprot IDs into separate rows and drop unnecessary columns
            kinetic_laws = (
                kinetic_laws.explode("UniprotID")
                .drop(columns=["EntryID", "metadata"])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        except Exception as e:
            self.logger.error(f"{kinetic_law}: Error cleaning Uniprot IDs: {e}")
            raise e
        self.logger.info(
            f"{kinetic_law}: Cleaning Uniprot IDs for kinetic law completed."
        )
        self.logger.info(f"{kinetic_law}: Kinetic law extraction completed.")
        return kinetic_laws

    def process_kinetic_params(self, SABIO_compounds):
        self.download_enzyme_dat()
        with open(self.ENZYME_DAT_PATH, "r") as f:
            ec_nos = [
                line.split("ID")[-1].strip() for line in f if line.startswith("ID")
            ]

        current_version = self.current_sabio_version()
        latest_version = self.get_latest_sabio_version()

        # If a new version is available, update the data
        if latest_version and current_version and latest_version > current_version:
            self.logger.info("New SABIO version detected. Updating data...")
            try:
                subprocess.run(
                    ["rm", "-rf", str(self.SABIO_KINETIC_LAWS_PATH)], check=True
                )  # Delete the old data
                self.logger.info(f"Deleted folder: {self.SABIO_KINETIC_LAWS_PATH}")
            except subprocess.CalledProcessError as e:
                self.logger.error(
                    f"Failed to delete folder {self.SABIO_KINETIC_LAWS_PATH}: {e}"
                )
                raise e

        # Recreate the directory for kinetic laws
        self.SABIO_KINETIC_LAWS_PATH.mkdir(parents=True, exist_ok=True)

        # Update the version file if necessary
        if not current_version or latest_version > current_version:
            self.update_sabio_version()
        else:
            self.logger.info("SABIO version is up to date.")

        # Download kinetic laws for all EC numbers
        self.logger.info(f"Downloading kinetic laws for {len(ec_nos)} EC numbers...")
        master_df = self.download_SABIO_kinetic_params_parallel(ec_nos)
        self.logger.info(f"Downloaded a total of {len(master_df)} kinetic laws.")

        self.logger.info("Parsing SABIO data...")
        try:
            # Extract associated species from the master DataFrame
            sabio_kinetic_df = pd.concat(
                self.extract_kinetic_law(master_df, kinetic_law)
                for kinetic_law in ["kcat", "Km", "Ki"]
            ).reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Error parsing SABIO data: {e}")
            raise e

        SABIO_compounds = SABIO_compounds[["sabio_id", "name", "smiles"]].rename(
            columns={"name": "substrate", "sabio_id": "substrate_id"}
        )
        SABIO_compounds["substrate_id"] = "SABIO:" + SABIO_compounds["substrate_id"]
        try:
            self.logger.info(f"Adding SMILES for substrates...")
            sabio_kinetic_df = sabio_kinetic_df.merge(
                SABIO_compounds, on="substrate", how="left"
            )
        except Exception as e:
            self.logger.error(f"Error adding SMILES for substrates: {e}")

        self.logger.info(
            f"Kinetic law value types distribution: {sabio_kinetic_df["value_type"].value_counts().to_dict()}"
        )

        try:
            sabio_kinetic_df.to_parquet(
                self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH,
                index=False,
                compression="brotli",
            )
            self.logger.info(
                f"SABIO kinetic params saved to {str(self.SABIO_KINETIC_PARAMS_PARQUET_FILE_PATH)}"
            )
        except Exception as e:
            self.logger.error(f"Error saving SABIO kinetic params: {e}")
            raise e

        return sabio_kinetic_df

    def process_reactions_compounds(self):
        try:
            params = {"format": "txt", "q": "ECNumber:*"}
            self.logger.info("Requesting all SABIO entry IDs.")
            request = requests.get(
                url=self.SABIO_ENTRYID_QUERY_URL, params=params, timeout=300
            )
            request.raise_for_status()
            all_entry_ids = [x for x in request.text.strip().split("\n")]
            self.logger.info(f"Found {len(all_entry_ids)} SABIO entry IDs.")

            self.download_SABIO_entries(all_entry_ids, self.SABIO_ENTRIES_PATH)
        except Exception as e:
            self.logger.error(f"Error downloading SABIO entries: {e}")
            raise

        self.logger.info("Parsing locally cached SABIO entries.")
        SABIO_entries = pd.DataFrame(
            [load_json(f) for f in Path(self.SABIO_ENTRIES_PATH).glob("*.json")]
        ).rename(
            columns={
                "KeggReactionID": "kegg_id",
                "SabioReactionID": "sabio_id",
                "ECNumber": "ec",
                "PubMedID": "pubmed_id",
                "ReactomeReactionID": "reactome_id",
            }
        )

        SABIO_entries["ec"] = SABIO_entries["ec"].apply(
            lambda x: normalize_ec_collection(x, fallback=None)
        )
        SABIO_entries = SABIO_entries.explode("ec").reset_index(drop=True)

        mask = SABIO_entries["EnzymeType"].str.contains("wildtype", na=False)
        SABIO_entries.loc[mask, "mutations"] = None
        SABIO_entries.loc[~mask, "mutations"] = (
            SABIO_entries.loc[~mask, "EnzymeType"]
            .str.findall(r"\b[A-Za-z]\d+[A-Za-z]\b")
            .apply(lambda x: None if x == [] else tuple(x))
        )

        SABIO_entries.drop(columns=["EntryID", "EnzymeType"], inplace=True)

        all_rxn_ids = (
            SABIO_entries["sabio_id"].dropna().drop_duplicates().astype(str).tolist()
        )
        self.logger.info(
            f"Downloading reactions for {len(all_rxn_ids)} unique SABIO reaction IDs."
        )
        self.download_SABIO_reactions(all_rxn_ids)

        SABIO_compounds = pd.DataFrame(
            [load_json(f) for f in Path(self.SABIO_COMPOUNDS_PATH).glob("*.json")]
        )
        SABIO_compounds["formula"] = SABIO_compounds["formula"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        self.logger.info(f"Number of metabolites: {len(SABIO_compounds)}")
        self.logger.info(
            f"Number of metabolites without SMILES: {len(SABIO_compounds[SABIO_compounds['smiles'].isna()])}"
        )

        SABIO_reactions = pd.DataFrame(
            [load_json(f) for f in Path(self.SABIO_REACTIONS_PATH).glob("*.json")]
        )
        self.logger.info(
            f"Number of unique reactions: {SABIO_reactions['sabio_id'].nunique()}"
        )
        self.logger.info(
            f"Number of unique reactions without rxn SMILES: {len(SABIO_reactions[SABIO_reactions['rxn_smiles'].isna()])}"
        )
        SABIO_reactions = SABIO_entries.merge(
            SABIO_reactions, on="sabio_id", how="inner"
        )
        SABIO_reactions["uniprot_id"] = SABIO_reactions["UniProtKB_AC"].apply(
            lambda x: x.split(";") if isinstance(x, str) else []
        )
        SABIO_reactions = (
            SABIO_reactions[
                [
                    "sabio_id",
                    "ec",
                    "definition",
                    "equation",
                    "rxn_smiles",
                    "uniprot_id",
                    "kegg_id",
                    "reactome_id",
                    "pubmed_id",
                    "mutations",
                ]
            ]
            .explode("uniprot_id")
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.logger.info(f"Number of reactions: {len(SABIO_reactions)}")
        self.logger.info(
            f"Number of reactions without rxn SMILES: {len(SABIO_reactions[SABIO_reactions['rxn_smiles'].isna()])}"
        )

        try:
            SABIO_reactions.to_parquet(
                self.SABIO_REACTIONS_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info(
                f"SABIO reactions saved to {str(self.SABIO_REACTIONS_PARQUET_FILE_PATH)}"
            )
        except Exception as e:
            self.logger.error(f"Error saving SABIO reactions: {e}")
            raise e
        try:
            SABIO_compounds.to_parquet(
                self.SABIO_COMPOUNDS_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info(
                f"SABIO compounds saved to {str(self.SABIO_COMPOUNDS_PARQUET_FILE_PATH)}"
            )
        except Exception as e:
            self.logger.error(f"Error saving SABIO compounds: {e}")
            raise e
        try:
            SABIO_entries.to_parquet(
                self.SABIO_ENTRIES_PARQUET_FILE_PATH,
                engine="pyarrow",
                compression="brotli",
                index=False,
            )
            self.logger.info(
                f"SABIO entries saved to {str(self.SABIO_ENTRIES_PARQUET_FILE_PATH)}"
            )
        except Exception as e:
            self.logger.error(f"Error saving SABIO entries: {e}")
            raise e

        return SABIO_reactions, SABIO_compounds, SABIO_entries

    def setup(self):
        self.logger.info("Starting SABIO setup process.")
        SABIO_reactions, SABIO_compounds, SABIO_entries = (
            self.process_reactions_compounds()
        )
        SABIO_kinetic_params = self.process_kinetic_params(SABIO_compounds)

        return SABIO_reactions, SABIO_compounds, SABIO_entries, SABIO_kinetic_params


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs/"):
        cfg = compose(config_name="data_processing")

    builder = SABIODatasetBuilder(cfg)
    reactions_df, compounds_df, entries_df, kinetic_params_df = builder.setup()
