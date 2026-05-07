import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from brendapyrser import BRENDA
from omegaconf import DictConfig
from hydra import initialize, compose

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
mol_to_smiles = chem_utils.mol_to_smiles
download_brenda_mol = chem_utils.download_brenda_mol
get_brenda_ligand_structure_id = chem_utils.get_brenda_ligand_structure_id
get_active_acc_id_from_ec_and_org_parallel = (
    chem_utils.get_active_acc_id_from_ec_and_org_parallel
)


class BRENDADatasetBuilder:
    def __init__(self, cfg: DictConfig):
        self.BRENDA_DUMP_FILE_PATH = Path(cfg.brenda.brenda_dump_file_path)
        self.BRENDA_MOL_PATH = Path(cfg.brenda.brenda_mol_path)

        self.BRENDA_KINETIC_PARAMS_PARQUET_FILE_PATH = Path(
            cfg.brenda.brenda_kinetic_params_parquet_file_path
        )

        LOG_PATH = Path(cfg.brenda.log_dir)

        for path in [
            self.BRENDA_MOL_PATH,
            self.BRENDA_KINETIC_PARAMS_PARQUET_FILE_PATH.parent,
            LOG_PATH,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.brenda.log_file_name
        ).get_logger()

        if not self.BRENDA_DUMP_FILE_PATH.exists():
            self.logger.error(
                f"BRENDA dump file not found at {self.BRENDA_DUMP_FILE_PATH}"
            )
            raise FileNotFoundError(
                f"BRENDA dump file not found at {self.BRENDA_DUMP_FILE_PATH}"
            )

    def make_brenda_parser(self):
        """
        Create and initialize a BRENDA parser instance.

        Returns:
            BRENDA: An instance of the BRENDA parser.

        Raises:
            Exception: If there is an error while creating the parser.
        """
        self.logger.info("Loading BRENDA data into parser...")
        try:
            # Initialize the BRENDA parser with the dump file
            parser = BRENDA(self.BRENDA_DUMP_FILE_PATH)
            self.logger.info("BRENDA data loaded successfully.")
            return parser
        except Exception as e:
            self.logger.error(f"Error while creating BRENDA parser: {e}")
            raise e

    def extract_kinetic_laws(self, parser):
        """
        Extract all kinetic laws (kcat, km, ki) from the BRENDA parser.

        Args:
            parser (BRENDA): An instance of the BRENDA parser containing reaction data.

        Returns:
            tuple: Three DataFrames containing extracted kcat, km, and ki values.
        """
        self.logger.info("Extracting kinetic laws...")

        # Initialize a dictionary to store extracted kinetic laws for kcat, km, and ki
        extracted_kinetic_laws = {"kcat": [], "km": [], "ki": []}

        # Iterate over all reactions in the parser
        for rxn in parser.reactions:
            # Extract kinetic law values for kcat, km, and ki
            kinetic_laws = {
                "kcat": rxn.Kcatvalues,
                "km": rxn.KMvalues,
                "ki": rxn.KIvalues,
            }

            # Process each type of kinetic law (kcat, km, ki)
            for law in kinetic_laws.keys():
                # Iterate over substrates for the current kinetic law
                for sub in kinetic_laws[law].keys():
                    # Iterate over entries for the current substrate
                    for entry in kinetic_laws[law][sub]:
                        # Skip entries without metadata
                        if not entry["meta"]:
                            continue

                        # Extract organisms and metadata from the entry
                        organisms = entry["species"]
                        metadatas = entry["meta"].split(";")

                        # Case 1: Single organism and multiple metadata entries
                        if len(organisms) == 1:
                            try:
                                for i in range(len(metadatas)):
                                    # Check if metadata contains protein references
                                    if not re.search(r"#([\d,]+)#", metadatas[i]):
                                        continue
                                    # Extract protein references from metadata
                                    prot_refs = (
                                        re.search(r"#([\d,]+)#", metadatas[i])
                                        .group(1)
                                        .split(",")
                                    )
                                    for prot_ref in prot_refs:
                                        # Retrieve protein ID from the reaction's protein data
                                        protein_id = (
                                            rxn.proteins[prot_ref]["proteinID"]
                                            if rxn.proteins[prot_ref]["proteinID"]
                                            else None
                                        )
                                        # Append the extracted data to the corresponding kinetic law list
                                        extracted_kinetic_laws[law].append(
                                            (
                                                rxn.ec_number,
                                                sub,
                                                entry["value"],
                                                organisms[0],
                                                metadatas[i],
                                                prot_ref,
                                                protein_id,
                                            )
                                        )
                            except Exception as e:
                                self.logger.error(
                                    f"{law} - Organisms = 1: Error while extracting kinetic laws: {e}"
                                )
                                continue

                        # Case 2: Multiple organisms and a single metadata entry
                        elif len(organisms) > 1 and len(metadatas) == 1:
                            try:
                                # Check if metadata contains protein references
                                if not re.search(r"#([\d,]+)#", metadatas[0]):
                                    continue
                                # Extract protein references from metadata
                                prot_refs = (
                                    re.search(r"#([\d,]+)#", metadatas[0])
                                    .group(1)
                                    .split(",")
                                )
                                # Ensure the number of organisms matches the number of protein references
                                if len(organisms) == len(prot_refs):
                                    for i in range(len(organisms)):
                                        # Retrieve protein ID from the reaction's protein data
                                        protein_id = (
                                            rxn.proteins[prot_refs[i]]["proteinID"]
                                            if rxn.proteins[prot_refs[i]]["proteinID"]
                                            else None
                                        )
                                        # Append the extracted data to the corresponding kinetic law list
                                        extracted_kinetic_laws[law].append(
                                            (
                                                rxn.ec_number,
                                                sub,
                                                entry["value"],
                                                organisms[i],
                                                metadatas[0],
                                                prot_refs[i],
                                                protein_id,
                                            )
                                        )
                            except Exception as e:
                                self.logger.error(
                                    f"{law} - Organisms = {len(organisms)} - Metadatas = 1: Error while extracting kinetic laws: {e}"
                                )
                                continue

                        # Case 3: Multiple organisms and multiple metadata entries
                        elif len(organisms) > 1 and len(metadatas) > 1:
                            # Ensure the number of organisms matches the number of metadata entries
                            if len(organisms) == len(metadatas):
                                try:
                                    for i in range(len(organisms)):
                                        # Check if metadata contains protein references
                                        if not re.search(r"#([\d,]+)#", metadatas[i]):
                                            continue
                                        # Extract protein references from metadata
                                        prot_refs = (
                                            re.search(r"#([\d,]+)#", metadatas[i])
                                            .group(1)
                                            .split(",")
                                        )
                                        for prot_ref in prot_refs:
                                            # Retrieve protein ID from the reaction's protein data
                                            protein_id = (
                                                rxn.proteins[prot_ref]["proteinID"]
                                                if rxn.proteins[prot_ref]["proteinID"]
                                                else None
                                            )
                                            # Append the extracted data to the corresponding kinetic law list
                                            extracted_kinetic_laws[law].append(
                                                (
                                                    rxn.ec_number,
                                                    sub,
                                                    entry["value"],
                                                    organisms[i],
                                                    metadatas[i],
                                                    prot_ref,
                                                    protein_id,
                                                )
                                            )
                                except Exception as e:
                                    self.logger.error(
                                        f"{law} - Organisms = {len(organisms)} - Metadatas = {len(metadatas)}: Error while extracting kinetic laws: {e}"
                                    )
                                    continue

                        # Case 4: No valid combination of organisms and metadata
                        else:
                            pass

        self.logger.info("Kinetic laws extraction completed.")

        # Convert the extracted kinetic laws into DataFrames
        self.logger.info("Converting extracted kinetic laws to DataFrames...")
        # Define column names for the DataFrame
        columns = [
            "ECNumber",
            "substrate",
            "value",
            "Organism",
            "metadata",
            "brenda_protein_id",
            "UniprotID",
        ]
        for law in extracted_kinetic_laws.keys():
            try:
                # Convert the list of extracted data into a DataFrame
                extracted_kinetic_laws[law] = pd.DataFrame(
                    extracted_kinetic_laws[law], columns=columns
                )
                extracted_kinetic_laws[law]["value_type"] = law
            except Exception as e:
                self.logger.error(
                    f"Error while converting {law} values to DataFrame: {e}"
                )
                raise e

        # Return the unified DataFrame containing all kinetic laws
        return pd.concat(list(extracted_kinetic_laws.values())).reset_index(drop=True)

    def clean_kinetic_law_df(self, kinetic_law_df):
        """
        Clean the extracted kinetic law DataFrame by performing the following operations:
        - Extract pH values from metadata.
        - Extract UniProt accession IDs.
        - Extract temperature values from metadata.
        - Annotate enzyme types (wildtype or mutant) and mutations.
        - Remove duplicates and retain the maximum value for each group.
        - Clean up UniProt IDs.

        Args:
            kinetic_law_df (pd.DataFrame): DataFrame containing kinetic law data.
            law (str): The type of kinetic law (e.g., "kcat", "km", or "ki").

        Returns:
            pd.DataFrame: Cleaned DataFrame with processed kinetic law data.
        """
        self.logger.info(f"Cleaning BRENDA kinetic params DataFrame...")

        # Add unit column - BRENDA values are in mM
        self.logger.info(f"Adding unit column...")
        kinetic_law_df["unit"] = "millimolar (mM)"
        kinetic_law_df.loc[kinetic_law_df["value_type"] == "kcat", "unit"] = "s^(-1)"

        # Extract pH values from the metadata column
        self.logger.info(f"Extracting pH values...")
        try:
            # Use regex to extract pH values from the metadata column
            kinetic_law_df["pH"] = kinetic_law_df["metadata"].str.extract(
                r"pH\s*([\d.]+)"
            )
            # Further refine the extracted pH values to ensure numeric format
            kinetic_law_df["pH"] = kinetic_law_df["pH"].str.extract(r"(\d+\.\d+|\d+)")
            kinetic_law_df["pH"] = kinetic_law_df["pH"].astype(float)
        except Exception as e:
            self.logger.error(f"Error while extracting pH values: {e}")
            raise e
        self.logger.info(f"pH values extraction completed.")

        # Extract UniProt accession IDs
        self.logger.info(f"Extracting UniProt accession IDs...")
        try:
            # Extract UniProt IDs using regex
            kinetic_law_df["UniprotID"] = kinetic_law_df["UniprotID"].str.extract(
                r"(\w+)"
            )
        except Exception as e:
            self.logger.error(f"Error while extracting UniProt accession IDs: {e}")
            raise e
        self.logger.info(f"UniProt accession IDs extraction completed.")

        # Extract temperature values from the metadata column
        self.logger.info(f"Extracting temperature values...")
        try:
            # Use regex to extract temperature values (e.g., "37°C")
            kinetic_law_df["Temperature"] = kinetic_law_df["metadata"].str.extract(
                r"(\d+)..C"
            )
            kinetic_law_df["Temperature"] = kinetic_law_df["Temperature"].astype(float)
        except Exception as e:
            self.logger.error(f"Error while extracting temperature values: {e}")
            raise e
        self.logger.info(f"Temperature values extraction completed.")

        # Annotate enzyme types (wildtype or mutant) and extract mutations
        self.logger.info(f"Annotating enzyme types and mutations...")
        try:
            # Identify mutants based on the presence of the word "mutant" in metadata
            kinetic_law_df["enzymeType"] = np.where(
                kinetic_law_df["metadata"].str.contains("mutant", case=False, na=False),
                "mutant",
                "wildtype",
            )
            # Extract mutation details for mutant enzymes
            mutations = (
                kinetic_law_df.loc[kinetic_law_df["enzymeType"] == "mutant", "metadata"]
                .str.extract(r"([A-Z]\d+[A-Z](?:/[A-Z]\d+[A-Z])*)")[0]
                .str.split("/")
                .apply(lambda x: tuple(x) if isinstance(x, list) else x)
            )
            kinetic_law_df.loc[
                kinetic_law_df["enzymeType"] == "mutant", "mutations"
            ] = mutations
        except Exception as e:
            self.logger.error(f"Error while annotating enzyme types and mutations: {e}")
            raise e
        self.logger.info(f"Enzyme types and mutations annotation completed.")

        # Drop unnecessary columns and remove duplicate rows
        self.logger.info(f"Removing unnecessary columns and duplicates...")
        kinetic_law_df = kinetic_law_df.drop(
            columns=["metadata", "brenda_protein_id"]
        ).drop_duplicates()

        # Clean up UniProt IDs by removing unnecessary prefixes
        self.logger.info(f"Cleaning Uniprot IDs...")
        try:
            # Remove "Uniprot" prefix from UniProt IDs, if present
            kinetic_law_df["UniprotID"] = kinetic_law_df["UniprotID"].apply(
                lambda x: x.replace("Uniprot", "") if isinstance(x, str) else x
            )
        except Exception as e:
            self.logger.error(f"Error while cleaning Uniprot IDs: {e}")
            raise e
        self.logger.info(f"Uniprot IDs cleaned.")

        # Return the cleaned DataFrame
        return kinetic_law_df

    def update_kinetic_laws_with_active_acc_id(self, kinetic_law_df):
        """
        Update the kinetic law DataFrame with active UniProt accession IDs for EC-Organism pairs
        that are missing protein IDs.

        Args:
            kinetic_law_df (pd.DataFrame): DataFrame containing kinetic law data.

        Returns:
            pd.DataFrame: Updated DataFrame with active UniProt accession IDs.
        """
        # Identify EC-Organism pairs with missing protein IDs
        self.logger.info(f"Identify EC - Organism pairs with no protein ID...")
        try:
            # Filter rows where UniProtID is null and extract unique EC-Organism pairs
            no_protein_id = kinetic_law_df[kinetic_law_df["UniprotID"].isnull()][
                ["ECNumber", "Organism"]
            ].drop_duplicates()
        except Exception as e:
            self.logger.error(
                f"Error while identifying EC - Organism pairs with no protein ID: {e}"
            )
            raise e
        self.logger.info(f"EC - Organism pairs with no protein ID identified.")

        # Fetch active UniProt accession IDs for the identified EC-Organism pairs
        self.logger.info(f"Fetching active accession IDs from UniProt...")
        try:
            # Fetch active accession IDs in parallel for the EC-Organism pairs
            active_acc_ids = get_active_acc_id_from_ec_and_org_parallel(
                no_protein_id.to_records(index=False).tolist()
            )

            # Separate reviewed and unreviewed accession IDs
            reviewed_acc_ids = (
                active_acc_ids[active_acc_ids["reviewed"] == True]
                .drop(columns=["reviewed"])
                .reset_index(drop=True)
            )
            reviewed_acc_ids = (
                reviewed_acc_ids.groupby(["ECNumber", "Organism"])
                .agg({"UniprotID": tuple})
                .reset_index()
            )
            # Only keep rows where UniprotID tuple has length 1
            reviewed_acc_ids = (
                reviewed_acc_ids[
                    reviewed_acc_ids["UniprotID"].apply(lambda x: len(x) == 1)
                ]
                .explode("UniprotID")
                .reset_index(drop=True)
            )

            unreviewed_acc_ids = (
                active_acc_ids[active_acc_ids["reviewed"] == False]
                .drop(columns=["reviewed"])
                .reset_index(drop=True)
            )
            unreviewed_acc_ids = (
                unreviewed_acc_ids.groupby(["ECNumber", "Organism"])
                .agg({"UniprotID": tuple})
                .reset_index()
            )
            # Only keep rows where UniprotID tuple has length 1
            unreviewed_acc_ids = (
                unreviewed_acc_ids[
                    unreviewed_acc_ids["UniprotID"].apply(lambda x: len(x) == 1)
                ]
                .explode("UniprotID")
                .reset_index(drop=True)
            )
        except Exception as e:
            self.logger.error(
                f"Error while fetching active accession IDs from UniProt: {e}"
            )
            raise e
        self.logger.info(f"Active accession IDs fetching completed.")

        # Merge the fetched accession IDs with the EC-Organism pairs missing protein IDs
        self.logger.info(
            f"Merging active accession IDs with EC-Org pairs missing protein IDs..."
        )
        try:
            # Merge reviewed and unreviewed accession IDs with the EC-Organism pairs
            no_protein_id_single_sequences = pd.merge(
                no_protein_id, reviewed_acc_ids, on=["ECNumber", "Organism"], how="left"
            )
            no_protein_id_single_sequences = pd.merge(
                no_protein_id_single_sequences,
                unreviewed_acc_ids,
                on=["ECNumber", "Organism"],
                how="left",
                suffixes=("_reviewed", "_unreviewed"),
            )

            # Combine reviewed and unreviewed IDs, prioritizing reviewed IDs
            no_protein_id_single_sequences["UniprotID"] = (
                no_protein_id_single_sequences["UniprotID_reviewed"].combine_first(
                    no_protein_id_single_sequences["UniprotID_unreviewed"]
                )
            )

            # Drop intermediate columns used for merging
            no_protein_id_single_sequences = no_protein_id_single_sequences.drop(
                columns=["UniprotID_reviewed", "UniprotID_unreviewed"]
            )
        except Exception as e:
            self.logger.error(
                f"Error while merging active accession IDs with EC-Org pairs: {e}"
            )
            raise e
        self.logger.info(f"Merging active accession IDs with EC-Org pairs completed.")

        # Update the original kinetic law DataFrame with the fetched accession IDs
        self.logger.info(f"Updating kinetic laws with active accession IDs...")
        try:
            # Separate rows with and without UniProt IDs
            kinetic_law_df_with_acc_id = kinetic_law_df[
                kinetic_law_df["UniprotID"].notnull()
            ]
            kinetic_law_df_wo_acc_id = kinetic_law_df[
                kinetic_law_df["UniprotID"].isnull()
            ].drop(columns=["UniprotID"])

            # Merge rows without UniProt IDs with the fetched accession IDs
            kinetic_law_df_wo_acc_id = kinetic_law_df_wo_acc_id.merge(
                no_protein_id_single_sequences, on=["ECNumber", "Organism"], how="left"
            )

            # Combine the updated rows with the original rows that already had UniProt IDs
            kinetic_law_df = pd.concat(
                [kinetic_law_df_with_acc_id, kinetic_law_df_wo_acc_id],
                ignore_index=True,
            )

            # Remove duplicate rows and reset the index
            kinetic_law_df = kinetic_law_df.drop_duplicates().reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error while updating kinetic laws with active accession IDs: {e}"
            )
            raise e
        self.logger.info(f"Kinetic laws updated with active accession IDs.")

        return kinetic_law_df

    def parse_brenda_data(self):
        """
        Parse BRENDA data, extract kinetic laws, clean the data, and save the results.

        This function performs the following steps:
        1. Load the BRENDA data using a parser.
        2. Extract kinetic laws (kcat, km, ki) from the data.
        3. Clean the extracted kinetic law data.
        4. Update the data with active UniProt accession IDs.
        5. Save the cleaned data to parquet files.

        Raises:
            Exception: If any error occurs during the parsing process.
        """

        self.logger.info("Starting BRENDA data parsing...")
        try:
            # Step 1: Load the BRENDA data using a parser
            parser = self.make_brenda_parser()

            # Step 2: Extract kinetic laws (kcat, km, ki) from the data
            kinetic_params_df = self.extract_kinetic_laws(parser)

            # Step 3: Clean the extracted kinetic law data
            kinetic_params_df = self.clean_kinetic_law_df(kinetic_params_df)

            # Step 4: Get ligand IDs for all substrates and download their MOL files from BRENDA
            all_brenda_substrates = kinetic_params_df["substrate"].unique()
            self.logger.info("Fetching ligand IDs for all substrates...")
            substrate_ligand_id_mapping = get_brenda_ligand_structure_id(
                all_brenda_substrates
            )
            kinetic_params_df["substrate_id"] = kinetic_params_df["substrate"].map(
                substrate_ligand_id_mapping
            )
            brenda_ligand_ids = sorted(
                pd.DataFrame(
                    tuple(
                        zip(
                            substrate_ligand_id_mapping.keys(),
                            substrate_ligand_id_mapping.values(),
                        )
                    ),
                    columns=["substrate", "substrate_id"],
                )["substrate_id"]
                .dropna()
                .unique(),
                key=int,
            )
            self.logger.info("Downloading MOL files for BRENDA ligand IDs...")
            ligand_id_mol_path_mapping = download_brenda_mol(brenda_ligand_ids)
            self.logger.info("Generating SMILES from MOL files...")
            ligand_id_smiles_mapping = {
                id: mol_to_smiles(self.BRENDA_MOL_PATH / mol_file)
                for id, mol_file in tqdm(
                    ligand_id_mol_path_mapping.items(), desc="Generating SMILES"
                )
            }
            kinetic_params_df["smiles"] = kinetic_params_df["substrate_id"].map(
                ligand_id_smiles_mapping
            )
            kinetic_params_df["substrate_id"] = (
                "BRENDA:" + kinetic_params_df["substrate_id"]
            )

            # Step 5: Update the data with active UniProt accession IDs
            kinetic_params_df = self.update_kinetic_laws_with_active_acc_id(
                kinetic_params_df
            )
            kinetic_params_df = kinetic_params_df.dropna(
                subset=["UniprotID"]
            ).reset_index(drop=True)
            self.logger.info(
                "Kinetic laws cleaned and updated with active accession IDs."
            )

            # Step 6: Save the cleaned data to parquet files
            self.logger.info("Saving cleaned kinetic laws to parquet files...")
            kinetic_params_df.to_parquet(
                self.BRENDA_KINETIC_PARAMS_PARQUET_FILE_PATH,
                index=False,
                compression="brotli",
            )
            self.logger.info("Cleaned kinetic laws saved successfully.")

            self.logger.info("BRENDA data parsing completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during BRENDA data parsing: {e}")
            raise e


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs/"):
        cfg = compose(config_name="data_processing")
    builder = BRENDADatasetBuilder(cfg)
    builder.parse_brenda_data()
