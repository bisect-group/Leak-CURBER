import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from chemeq import chemeq
from omegaconf import DictConfig
from pandarallel import pandarallel
from hydra import initialize, compose

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
get_af_pdb = chem_utils.get_af_pdb
mutate_sequence = chem_utils.mutate_sequence
get_uniprot_acc_json = chem_utils.get_uniprot_acc_json
add_uniprot_date_column = chem_utils.add_uniprot_date_column
get_uniprot_ec_acc_map = chem_utils.get_uniprot_ec_acc_map
parse_all_reference_xmls = chem_utils.parse_all_reference_xmls
assign_experimental_and_af_pdbs = chem_utils.assign_experimental_and_af_pdbs
download_pubmed_abstracts_parallel = chem_utils.download_pubmed_abstracts_parallel
normalize_ec_collection = chem_utils.normalize_ec_collection

pandarallel.initialize(nb_workers=os.cpu_count())


class ReactionsDatasetBuilder:
    def __init__(self, cfg: DictConfig):
        self.RHEA_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.rhea_reactions_parquet_file_path
        )
        self.KEGG_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.kegg_reactions_parquet_file_path
        )
        self.SABIO_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.sabio_reactions_parquet_file_path
        )
        self.METACYC_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.metacyc_reactions_parquet_file_path
        )

        self.SAVE_1D = cfg.reactions_dataset.save_1d_dataset
        self.SAVE_3D = cfg.reactions_dataset.save_3d_dataset
        if not (self.SAVE_1D or self.SAVE_3D):
            raise ValueError(
                "At least one of reactions_dataset.save_1d_dataset or "
                "reactions_dataset.save_3d_dataset must be true."
            )

        self.UNIFIED_REACTIONS_1D_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.unified_reactions_1d_parquet_file_path
        )
        self.UNIFIED_REACTIONS_3D_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.unified_reactions_3d_parquet_file_path
        )
        self.UNIFIED_REACTIONS_REFERENCES_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.unified_reactions_references_parquet_file_path
        )
        self.UNIFIED_REACTIONS_SEQUENCES_PICKLE_PATH = Path(
            cfg.reactions_dataset.unified_reactions_sequences_pickle_path
        )
        self.UNIFIED_REACTIONS_RXN_SMILES_PICKLE_PATH = Path(
            cfg.reactions_dataset.unified_reactions_rxn_smiles_pickle_path
        )
        self.REACTION_OUTCOME_DATASET_PARQUET_FILE_PATH = Path(
            cfg.reactions_dataset.reaction_outcome_dataset_parquet_file_path
        )

        LOG_PATH = Path(cfg.reactions_dataset.log_dir)

        for path in [
            LOG_PATH,
            self.UNIFIED_REACTIONS_1D_PARQUET_FILE_PATH.parent,
            self.UNIFIED_REACTIONS_3D_PARQUET_FILE_PATH.parent,
            self.UNIFIED_REACTIONS_REFERENCES_PARQUET_FILE_PATH.parent,
            self.UNIFIED_REACTIONS_SEQUENCES_PICKLE_PATH.parent,
            self.UNIFIED_REACTIONS_RXN_SMILES_PICKLE_PATH.parent,
            self.REACTION_OUTCOME_DATASET_PARQUET_FILE_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.reactions_dataset.log_file_name,
        ).get_logger()

        for path in [
            self.RHEA_REACTIONS_PARQUET_FILE_PATH,
            self.KEGG_REACTIONS_PARQUET_FILE_PATH,
            self.SABIO_REACTIONS_PARQUET_FILE_PATH,
            self.METACYC_REACTIONS_PARQUET_FILE_PATH,
        ]:
            if not path.exists():
                self.logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")

    def balance_eqn(self, row):
        """
        Attempt to balance a chemical equation in a DataFrame row.

        Args:
            row (pd.Series): A pandas Series containing at least an 'equation' field.

        Returns:
            pd.Series: The input row with updated 'equation' and 'balance_status' fields.
        """

        eqn = row["equation"]
        if not eqn:
            row["balance_status"] = "undetermined"
            return row

        try:
            chem_eqn = chemeq(eqn.replace("<=>", "="))
        except Exception as e:
            self.logger.error(
                f"Error parsing row {row.name} \t | \t equation {eqn}: {e}"
            )
            row["balance_status"] = "unbalanced-unfixable"
            return row

        if not chem_eqn.is_balanced:
            try:
                chem_eqn.balance()
            except Exception as e:
                self.logger.error(
                    f"Error balancing row {row.name} \t | \t equation {eqn}: {e}"
                )
                row["balance_status"] = "unbalanced-unfixable"
                return row

        row["equation"] = str(chem_eqn).replace("=", "<=>")
        row["balance_status"] = "balanced"
        return row

    def parse_rhea_reactions(self):
        """
        Parse Rhea reactions from parquet file.

        Args:
            None

        Returns:
            pd.DataFrame: DataFrame containing Rhea reactions with additionl 'source' and 'balance_status' columns.
        """

        self.logger.info("Parsing Rhea reactions from parquet file...")
        try:
            rhea_reactions_df = pd.read_parquet(self.RHEA_REACTIONS_PARQUET_FILE_PATH)
        except Exception as e:
            self.logger.error(f"Failed to read Rhea reactions parquet file: {e}")
            raise

        self.logger.info(
            f"Setting 'source' col as 'rhea' and 'balance_status' col as 'balanced'."
        )
        rhea_reactions_df["source"] = "rhea"
        rhea_reactions_df["balance_status"] = "balanced"

        self.logger.info(f"Parsed Rhea reactions.")
        return rhea_reactions_df

    def parse_kegg_reactions(self, rhea_reactions):
        """
        Parse KEGG reactions from parquet file, removing overlaps with Rhea reactions.

        Args:
            rhea_reactions (pd.DataFrame): DataFrame containing Rhea reactions, must have 'kegg_id' and 'rhea_id' columns.

        Returns:
            pd.DataFrame: DataFrame of KEGG reactions not overlapping with Rhea, with UniProt mapping and metadata columns.
        """

        self.logger.info("Parsing KEGG reactions from parquet file...")
        try:
            kegg_reactions_df = pd.read_parquet(self.KEGG_REACTIONS_PARQUET_FILE_PATH)
        except Exception as e:
            self.logger.error(f"Failed to read KEGG reactions parquet file: {e}")
            raise e

        try:
            self.logger.info("Computing RHEA-KEGG overlap...")
            rhea_kegg_overlap = dict(
                rhea_reactions.dropna(subset=["kegg_id"])[["kegg_id", "rhea_id"]]
                .drop_duplicates()
                .values
            )  # kegg_id: rhea_id
            self.logger.info(
                f"Found {len(rhea_kegg_overlap)} KEGG reactions overlapping with Rhea."
            )

            self.logger.info("Slicing KEGG reactions without Rhea overlap...")
            kegg_reactions_no_rhea_overlap = kegg_reactions_df[
                (~kegg_reactions_df["kegg_id"].isin(rhea_kegg_overlap.keys()))
            ].reset_index(drop=True)

            # Remove all rows where rhea_id is present in rhea_reactions['rhea_id']
            kegg_reactions_no_rhea_overlap = kegg_reactions_no_rhea_overlap[
                ~(
                    kegg_reactions_no_rhea_overlap["rhea_id"].notna()
                    & kegg_reactions_no_rhea_overlap["rhea_id"].isin(
                        rhea_reactions["rhea_id"]
                    )
                )
            ].reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error while computing and slicing KEGG reactions without Rhea overlap: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Downloading UniProt EC-acc_id mapping for remaining KEGG reactions..."
            )
            ec_acc_map = get_uniprot_ec_acc_map(
                kegg_reactions_no_rhea_overlap["ec"].dropna().unique()
            )

            self.logger.info("Mapping UniProt IDs for KEGG reactions...")
            kegg_reactions_no_rhea_overlap["uniprot_id"] = (
                kegg_reactions_no_rhea_overlap["ec"].map(ec_acc_map)
            )
            kegg_reactions_no_rhea_overlap = kegg_reactions_no_rhea_overlap.explode(
                "uniprot_id"
            ).reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error while mapping UniProt IDs for remaining KEGG reactions: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Downloading protein sequence data from UniProt for remaining KEGG reactions..."
            )
            uniprot_df = pd.DataFrame(
                get_uniprot_acc_json(
                    kegg_reactions_no_rhea_overlap["uniprot_id"].dropna().unique()
                ).values()
            ).drop(columns=["ecs"])
            self.logger.info("Merging UniProt data into KEGG reactions...")
            kegg_reactions_no_rhea_overlap = kegg_reactions_no_rhea_overlap.merge(
                uniprot_df.rename(columns={"acc_id": "uniprot_id"}),
                on="uniprot_id",
                how="left",
            )
        except Exception as e:
            self.logger.error(f"Error while downloading and merging UniProt data: {e}")
            raise e

        try:
            self.logger.info(
                "Setting 'source' column to 'kegg' and 'balance_status' to 'balanced'."
            )
            kegg_reactions_no_rhea_overlap["source"] = "kegg"
            kegg_reactions_no_rhea_overlap["balance_status"] = "balanced"
        except Exception as e:
            self.logger.error(
                f"Error while setting source and balance_status columns: {e}"
            )
            raise e

        self.logger.info(
            "Finished parsing KEGG reactions and removed overlaps with Rhea."
        )
        return kegg_reactions_no_rhea_overlap

    def handle_sabio_reaction_mutant_seqs(self, df):
        """
        Handle mutant sequences for SABIO reactions.

        Args:
            df (pd.DataFrame): DataFrame containing SABIO reactions with 'mutations' and 'uniprot_id' columns.
        Returns:
            pd.DataFrame: DataFrame with mutated sequences applied where applicable.
        """
        try:
            df.rename(columns={"uniprot_id": "UniprotID"}, inplace=True)

            mask = df["mutations"].notna()
            df.loc[mask, "mutations"] = df.loc[mask, "mutations"].apply(tuple)
            df.loc[mask] = df.loc[mask].apply(mutate_sequence, axis=1)

            df.rename(columns={"UniprotID": "uniprot_id"}, inplace=True)
        except Exception as e:
            self.logger.error(f"Error while handling mutant sequences: {e}")
            raise e
        invalid_mutant_mask = (df["mutations"].notna()) & (df["sequence"].isna())
        if invalid_mutant_mask.any():
            self.logger.info(
                f"Dropping {invalid_mutant_mask.sum()} SABIO reactions with invalid mutant sequences."
            )
        return df[~invalid_mutant_mask].reset_index(drop=True)

    def parse_sabio_reactions(self, rhea_reactions, kegg_reactions):
        """
        Parse SABIO reactions from parquet file, annotate overlaps with KEGG/Rhea, balance equations, and map UniProt IDs.

        Args:
            rhea_reactions (pd.DataFrame): DataFrame containing Rhea reactions, must have 'kegg_id' column.
            kegg_reactions (pd.DataFrame): DataFrame containing KEGG reactions, must have 'kegg_id' column.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - Updated rhea_reactions with SABIO overlap info.
                - Updated kegg_reactions with SABIO overlap and PubMed info.
                - SABIO reactions not overlapping with KEGG, with balanced equations and UniProt mapping.
        """
        try:
            self.logger.info("Parsing SABIO reactions from parquet file...")
            sabio_reactions = pd.read_parquet(self.SABIO_REACTIONS_PARQUET_FILE_PATH)
        except Exception as e:
            self.logger.error(f"Error while reading SABIO reactions: {e}")
            raise e

        try:
            self.logger.info("Computing KEGG-SABIO overlap...")
            kegg_sabio_overlap = dict(
                sabio_reactions.dropna(subset=["kegg_id"])[["kegg_id", "sabio_id"]]
                .drop_duplicates()
                .values
            )  # kegg_id: sabio_id
            self.logger.info(
                f"Found {len(kegg_sabio_overlap)} SABIO reactions overlapping with KEGG."
            )
        except Exception as e:
            self.logger.error(f"Error while computing KEGG-SABIO overlap: {e}")
            raise e

        try:
            self.logger.info("Extracting PUBMED IDs for KEGG-SABIO overlaps...")
            kegg_sabio_overlap_df = sabio_reactions[
                sabio_reactions["kegg_id"].isin(kegg_sabio_overlap.keys())
            ][["sabio_id", "kegg_id", "pubmed_id"]].drop_duplicates()

            kegg_pubmed_dict = dict(
                kegg_sabio_overlap_df.groupby("kegg_id")["pubmed_id"]
                .agg(lambda x: list(x.dropna().unique()))
                .reset_index()
                .values
            )
            self.logger.info(
                f"Found {len(kegg_pubmed_dict)} unique KEGG-PUBMED mappings indirectly via SABIO."
            )

            self.logger.info("Mapping PUBMED IDs to KEGG reactions...")
            kegg_reactions["pubmed_id"] = kegg_reactions["kegg_id"].map(
                kegg_pubmed_dict
            )
        except Exception as e:
            self.logger.error(
                f"Error while extracting PUBMED IDs for KEGG-SABIO overlaps: {e}"
            )
            raise e

        try:
            kegg_sabio_dict = dict(
                kegg_sabio_overlap_df.groupby("kegg_id")["sabio_id"]
                .agg(lambda x: list(x.dropna().unique()))
                .reset_index()
                .values
            )
            self.logger.info("Mapping SABIO IDs to KEGG reactions...")
            kegg_reactions["sabio_id"] = kegg_reactions["kegg_id"].map(kegg_sabio_dict)

            self.logger.info("Mapping SABIO IDs to Rhea reactions...")
            rhea_reactions["sabio_id"] = rhea_reactions["kegg_id"].map(kegg_sabio_dict)
        except Exception as e:
            self.logger.error(
                f"Error while mapping SABIO IDs to KEGG and Rhea reactions: {e}"
            )
            raise e

        try:
            self.logger.info("Slicing SABIO reactions without KEGG overlap...")
            sabio_reactions_no_kegg_overlap = sabio_reactions[
                (~sabio_reactions["kegg_id"].isin(kegg_sabio_overlap.keys()))
            ].reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error while slicing SABIO reactions without KEGG overlap: {e}"
            )
            raise e

        try:
            self.logger.info(
                f"Parallel balancing equations using {os.cpu_count()} workers for SABIO reactions without KEGG overlap..."
            )
            sabio_reactions_no_kegg_overlap = (
                sabio_reactions_no_kegg_overlap.parallel_apply(self.balance_eqn, axis=1)
            )
        except Exception as e:
            self.logger.error(
                f"Error while parallel balancing equations for SABIO reactions without KEGG overlap: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Downloading UniProt EC-acc_id mapping for subset where EC. number is available without a corresponding UniProt acc_id..."
            )
            mask = (
                sabio_reactions_no_kegg_overlap["ec"].notna()
                & (sabio_reactions_no_kegg_overlap["ec"] != "-")
                & sabio_reactions_no_kegg_overlap["uniprot_id"].isna()
            )
            ec_acc_map = get_uniprot_ec_acc_map(
                sabio_reactions_no_kegg_overlap.loc[mask, "ec"].dropna().unique()
            )

            self.logger.info(
                "Mapping UniProt IDs for SABIO reactions without KEGG overlap..."
            )
            sabio_reactions_no_kegg_overlap.loc[mask, "uniprot_id"] = (
                sabio_reactions_no_kegg_overlap.loc[mask, "ec"].map(ec_acc_map)
            )
            sabio_reactions_no_kegg_overlap = sabio_reactions_no_kegg_overlap.explode(
                "uniprot_id"
            ).reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error while mapping UniProt IDs for SABIO reactions without KEGG overlap: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Downloading protein sequence data from UniProt for SABIO reactions without KEGG overlap..."
            )
            uniprot_df = pd.DataFrame(
                get_uniprot_acc_json(
                    sabio_reactions_no_kegg_overlap["uniprot_id"].dropna().unique()
                ).values()
            ).drop(columns=["ecs"])

            self.logger.info(
                "Merging UniProt data into SABIO reactions without KEGG overlap..."
            )
            sabio_reactions_no_kegg_overlap = sabio_reactions_no_kegg_overlap.merge(
                uniprot_df.rename(columns={"acc_id": "uniprot_id"}),
                on="uniprot_id",
                how="left",
            )
        except Exception as e:
            self.logger.error(
                f"Error while merging UniProt data into SABIO reactions without KEGG overlap: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Handling mutant sequences for SABIO reactions without KEGG overlap..."
            )
            sabio_reactions_no_kegg_overlap = self.handle_sabio_reaction_mutant_seqs(
                sabio_reactions_no_kegg_overlap
            )
        except Exception as e:
            self.logger.error(
                f"Error while handling mutant sequences for SABIO reactions without KEGG overlap: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Setting 'source' column to 'sabio' for SABIO reactions without KEGG overlap."
            )
            sabio_reactions_no_kegg_overlap["source"] = "sabio"
        except Exception as e:
            self.logger.error(
                f"Error while setting source column for SABIO reactions without KEGG overlap: {e}"
            )
            raise e

        self.logger.info(
            "Parsed SABIO reactions, annotated overlaps, and balanced equations."
        )
        return rhea_reactions, kegg_reactions, sabio_reactions_no_kegg_overlap

    def add_metacyc_reactions(self, master_reactions_df):
        self.logger.info("Parsing Metacyc reactions from parquet file...")
        try:
            metacyc_reactions = (
                pd.read_parquet(self.METACYC_REACTIONS_PARQUET_FILE_PATH)
                .drop(columns=["ENZRXN-ID", "GIBBS-0"])
                .rename(
                    columns={
                        "UNIQUE-ID": "metacyc_id",
                        "CITATIONS": "pubmed_id",
                        "EC-NUMBER": "ec",
                        "SEQUENCE": "sequence",
                        "ATOM-MAPPED-SMILES": "rxn_smiles",
                        "DEFINITION": "definition",
                        "EQUATION": "equation",
                        "UNIPROT-ID": "uniprot_id",
                        "REACTION-DIRECTION": "direction",
                        "REACTION-BALANCE-STATUS": "balance_status",
                        "ORGANISM": "organism",
                        "SPONTANEOUS?": "spontaneous",
                    }
                )
            )
            self.logger.info(f"Parsed {len(metacyc_reactions)} Metacyc reactions.")
        except Exception as e:
            self.logger.error(f"Failed to read Metacyc reactions parquet file: {e}")

        try:
            self.logger.info("Setting 'source' column to 'metacyc'.")
            metacyc_reactions["source"] = "metacyc"
        except Exception as e:
            self.logger.error(
                f"Error while setting source column for Metacyc reactions: {e}"
            )
            raise e

        # Create overlap dicts for mapping IDs
        try:
            rhea_metacyc_overlap = dict(
                metacyc_reactions.dropna(subset=["rhea_id"])[["metacyc_id", "rhea_id"]]
                .explode("rhea_id")
                .drop_duplicates()
                .values
            )  # metacyc_id: rhea_id
            kegg_metacyc_overlap = dict(
                metacyc_reactions.dropna(subset=["kegg_id"])[["metacyc_id", "kegg_id"]]
                .explode("kegg_id")
                .drop_duplicates()
                .values
            )  # metacyc_id: kegg_id

            self.logger.info("Mapping Metanetx IDs to RHEA reactions...")
            rhea_metanetx_dict = dict(
                metacyc_reactions[
                    (metacyc_reactions["metacyc_id"].isin(rhea_metacyc_overlap.keys()))
                ]
                .explode("rhea_id")[["rhea_id", "metanetx_id"]]
                .values
            )
            mask = master_reactions_df["rhea_id"].notna()
            master_reactions_df.loc[mask, "metanetx_id"] = master_reactions_df.loc[
                mask, "rhea_id"
            ].map(rhea_metanetx_dict)

            self.logger.info(
                "Mapping Metacyc IDs to subset of RHEA reactions that do not have metacyc xrefs yet..."
            )
            rhea_metacyc_dict = dict(
                metacyc_reactions[
                    (metacyc_reactions["metacyc_id"].isin(rhea_metacyc_overlap.keys()))
                ]
                .explode("rhea_id")[["rhea_id", "metacyc_id"]]
                .values
            )
            mask = (master_reactions_df["rhea_id"].notna()) & (
                master_reactions_df["metacyc_id"].isna()
            )
            master_reactions_df.loc[mask, "metacyc_id"] = master_reactions_df.loc[
                mask, "rhea_id"
            ].map(rhea_metacyc_dict)

            self.logger.info("Mapping Metanetx IDs to KEGG reactions...")
            kegg_metanetx_dict = dict(
                metacyc_reactions[
                    (metacyc_reactions["metacyc_id"].isin(kegg_metacyc_overlap.keys()))
                ]
                .explode("kegg_id")[["kegg_id", "metanetx_id"]]
                .values
            )
            mask = (master_reactions_df["kegg_id"].notna()) & (
                master_reactions_df["metanetx_id"].isna()
            )
            master_reactions_df.loc[mask, "metanetx_id"] = master_reactions_df.loc[
                mask, "kegg_id"
            ].map(kegg_metanetx_dict)

            self.logger.info(
                "Mapping Metacyc IDs to subset of KEGG reactions that do not have metacyc xrefs yet..."
            )
            kegg_metacyc_dict = dict(
                metacyc_reactions[
                    (metacyc_reactions["metacyc_id"].isin(kegg_metacyc_overlap.keys()))
                ]
                .explode("kegg_id")[["kegg_id", "metacyc_id"]]
                .values
            )
            mask = (master_reactions_df["kegg_id"].notna()) & (
                master_reactions_df["metacyc_id"].isna()
            )
            master_reactions_df.loc[mask, "metacyc_id"] = master_reactions_df.loc[
                mask, "kegg_id"
            ].map(kegg_metacyc_dict)
        except Exception as e:
            self.logger.error(
                f"Error while mapping Metacyc and MetaNetX IDs to RHEA/KEGG reactions: {e}"
            )
            raise e

        # Remove overlaps with Rhea and KEGG
        try:
            self.logger.info(
                "Slicing Metacyc reactions without Rhea and KEGG overlaps..."
            )
            metacyc_reactions_no_rhea_kegg_overlap = metacyc_reactions[
                (~metacyc_reactions["metacyc_id"].isin(rhea_metacyc_overlap.keys()))
                & (~metacyc_reactions["metacyc_id"].isin(kegg_metacyc_overlap.keys()))
            ].reset_index(drop=True)
        except Exception as e:
            self.logger.error(
                f"Error while slicing Metacyc reactions without Rhea and KEGG overlaps: {e}"
            )
            raise e

        try:
            metacyc_reactions_no_rhea_kegg_overlap["balance_status"] = (
                metacyc_reactions_no_rhea_kegg_overlap["balance_status"]
                .str.lower()
                .str.replace(":", "")
            )
        except Exception as e:
            self.logger.error(
                f"Error while normalizing balance_status values in Metacyc reactions: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Downloading UniProt EC-acc_id mapping for Metacyc reactions with EC but no sequence..."
            )
            mask = (
                metacyc_reactions_no_rhea_kegg_overlap["ec"].notna()
                & metacyc_reactions_no_rhea_kegg_overlap["sequence"].isna()
            )
            ec_acc_map = get_uniprot_ec_acc_map(
                metacyc_reactions_no_rhea_kegg_overlap[mask]["ec"].unique()
            )
            self.logger.info(
                "Mapping UniProt IDs for Metacyc reactions with EC but no sequence..."
            )
            metacyc_reactions_no_rhea_kegg_overlap.loc[mask, "uniprot_id"] = (
                metacyc_reactions_no_rhea_kegg_overlap.loc[mask, "ec"].map(ec_acc_map)
            )
            metacyc_reactions_no_rhea_kegg_overlap = (
                metacyc_reactions_no_rhea_kegg_overlap.explode(
                    "uniprot_id"
                ).reset_index(drop=True)
            )
        except Exception as e:
            self.logger.error(
                f"Error while mapping UniProt IDs for Metacyc reactions with EC but no sequence: {e}"
            )
            raise e

        try:
            self.logger.info(
                "Downloading protein sequence data from UniProt for Metacyc reactions with UniProt IDs..."
            )
            uniprot_df = pd.DataFrame(
                get_uniprot_acc_json(
                    metacyc_reactions_no_rhea_kegg_overlap["uniprot_id"]
                    .dropna()
                    .unique()
                ).values()
            ).drop(columns=["ecs"])

            self.logger.info("Merging UniProt data into Metacyc reactions...")
            metacyc_reactions_no_rhea_kegg_overlap = (
                metacyc_reactions_no_rhea_kegg_overlap.drop(
                    columns=["organism", "sequence"]
                ).merge(
                    uniprot_df.rename(columns={"acc_id": "uniprot_id"}),
                    on="uniprot_id",
                    how="left",
                )
            )
        except Exception as e:
            self.logger.error(
                f"Error while downloading and merging UniProt data for Metacyc reactions: {e}"
            )
            raise e

        try:
            # Add metacyc exclusive reactions to master reactions
            master_reactions_df = pd.concat(
                [master_reactions_df, metacyc_reactions_no_rhea_kegg_overlap],
                ignore_index=True,
            )
            self.logger.info(
                f"Added {len(metacyc_reactions_no_rhea_kegg_overlap)} Metacyc reactions to master reactions DataFrame."
            )
        except Exception as e:
            self.logger.error(
                f"Error while adding Metacyc reactions to master reactions DataFrame: {e}"
            )
            raise e

        return master_reactions_df

    def pickle_all_rxn_smiles(self, master_reactions_df):
        """
        Save the reaction SMILES embeddings to a pickle file.

        Args:
            master_reactions_df (pd.DataFrame): DataFrame containing reactions with 'rxn_smiles' column.

        Outputs:
            Saves a pickle file with unique reaction SMILES to DATA_PATH.
        """
        try:
            self.logger.info("Pickling unique reaction SMILES...")
            uniq_rxn_smiles = (
                master_reactions_df[["rxn_smiles", "source"]]
                .dropna(subset=["rxn_smiles"])
                .drop_duplicates(subset=["rxn_smiles"])
            )
            unmapped_rxn_smiles = uniq_rxn_smiles[
                ~(uniq_rxn_smiles["source"] == "metacyc")
            ]["rxn_smiles"].tolist()
            with open(self.UNIFIED_REACTIONS_RXN_SMILES_PICKLE_PATH, "wb") as f:
                pickle.dump(unmapped_rxn_smiles, f)
            self.logger.info("Pickled unique reaction SMILES.")
        except Exception as e:
            self.logger.error(f"Error while pickling unique reaction SMILES: {e}")
            raise e

    def pickle_all_sequences(self, master_reactions_df):
        """
        Save the protein sequences to a pickle file.

        Args:
            master_reactions_df (pd.DataFrame): DataFrame containing reactions with 'sequence' column.

        Outputs:
            Saves a pickle file with unique protein sequences to DATA_PATH.
        """
        try:
            self.logger.info("Pickling unique protein sequences...")
            with open(self.UNIFIED_REACTIONS_SEQUENCES_PICKLE_PATH, "wb") as f:
                pickle.dump(
                    master_reactions_df[["uniprot_id", "sequence"]]
                    .rename(columns={"uniprot_id": "acc_id"})
                    .dropna()
                    .drop_duplicates()
                    .to_dict(orient="records"),
                    f,
                )
            self.logger.info("Pickled dict of unique protein acc_id: sequences.")
        except Exception as e:
            self.logger.error(f"Error while pickling unique protein sequences: {e}")
            raise e

    def log_stats(self, master_reactions_df):
        rhea_subset = master_reactions_df[master_reactions_df["source"] == "rhea"]
        kegg_subset = master_reactions_df[master_reactions_df["source"] == "kegg"]
        sabio_subset = master_reactions_df[master_reactions_df["source"] == "sabio"]
        metacyc_subset = master_reactions_df[master_reactions_df["source"] == "metacyc"]

        rhea_uniq_subset = rhea_subset.drop_duplicates(subset=["rhea_id"])
        kegg_uniq_subset = kegg_subset.drop_duplicates(subset=["kegg_id"])
        sabio_uniq_subset = sabio_subset.drop_duplicates(subset=["sabio_id"])
        metacyc_uniq_subset = metacyc_subset.drop_duplicates(subset=["metacyc_id"])

        rhea_uniq_rxn = rhea_uniq_subset["rhea_id"].nunique() // 4
        kegg_uniq_rxn = kegg_uniq_subset["kegg_id"].nunique()
        sabio_uniq_rxn = sabio_uniq_subset["sabio_id"].nunique()
        metacyc_uniq_rxn = metacyc_uniq_subset["metacyc_id"].nunique()
        total_uniq_rxn = (
            rhea_uniq_rxn + kegg_uniq_rxn + sabio_uniq_rxn + metacyc_uniq_rxn
        )

        rhea_uniq_no_rxn_smiles = (
            len(rhea_uniq_subset[rhea_uniq_subset["rxn_smiles"].isna()]) // 4
        )
        kegg_uniq_no_rxn_smiles = len(
            kegg_uniq_subset[kegg_uniq_subset["rxn_smiles"].isna()]
        )
        sabio_uniq_no_rxn_smiles = len(
            sabio_uniq_subset[sabio_uniq_subset["rxn_smiles"].isna()]
        )
        metacyc_uniq_no_rxn_smiles = len(
            metacyc_uniq_subset[metacyc_uniq_subset["rxn_smiles"].isna()]
        )
        total_uniq_no_rxn_smiles = (
            rhea_uniq_no_rxn_smiles
            + kegg_uniq_no_rxn_smiles
            + sabio_uniq_no_rxn_smiles
            + metacyc_uniq_no_rxn_smiles
        )

        uniq_ecs = master_reactions_df["ec"].dropna().unique()
        rhea_uniq_ecs = rhea_uniq_subset["ec"].dropna().unique()
        kegg_uniq_ecs = kegg_uniq_subset["ec"].dropna().unique()
        sabio_uniq_ecs = sabio_uniq_subset["ec"].dropna().unique()
        metacyc_uniq_ecs = metacyc_uniq_subset["ec"].dropna().unique()

        self.logger.info(f"Rhea unique EC numbers: {len(rhea_uniq_ecs)}")
        self.logger.info(f"Kegg unique EC numbers: {len(kegg_uniq_ecs)}")
        self.logger.info(f"SABIO unique EC numbers: {len(sabio_uniq_ecs)}")
        self.logger.info(f"Metacyc unique EC numbers: {len(metacyc_uniq_ecs)}")
        self.logger.info(f"Total unique EC numbers: {len(uniq_ecs)}")

        uniq_acc_ids = master_reactions_df["uniprot_id"].dropna().unique()
        rhea_uniq_acc_ids = rhea_subset["uniprot_id"].dropna().unique()
        kegg_uniq_acc_ids = kegg_subset["uniprot_id"].dropna().unique()
        sabio_uniq_acc_ids = sabio_subset["uniprot_id"].dropna().unique()
        metacyc_uniq_acc_ids = metacyc_subset["uniprot_id"].dropna().unique()

        self.logger.info(f"Rhea unique UniProt IDs: {len(rhea_uniq_acc_ids)}")
        self.logger.info(f"Kegg unique UniProt IDs: {len(kegg_uniq_acc_ids)}")
        self.logger.info(f"SABIO unique UniProt IDs: {len(sabio_uniq_acc_ids)}")
        self.logger.info(f"Metacyc unique UniProt IDs: {len(metacyc_uniq_acc_ids)}")
        self.logger.info(f"Total unique UniProt IDs: {len(uniq_acc_ids)}")

        total_rhea_reactions = len(rhea_subset)
        total_kegg_reactions = len(kegg_subset)
        total_sabio_reactions = len(sabio_subset)
        total_metacyc_reactions = len(metacyc_subset)

        rhea_no_rxn_smiles = len(rhea_subset[rhea_subset["rxn_smiles"].isna()]) // 4
        kegg_no_rxn_smiles = len(kegg_subset[kegg_subset["rxn_smiles"].isna()])
        sabio_no_rxn_smiles = len(sabio_subset[sabio_subset["rxn_smiles"].isna()])
        metacyc_no_rxn_smiles = len(metacyc_subset[metacyc_subset["rxn_smiles"].isna()])

        self.logger.info(
            f"Rhea unique reactions without Reaction SMILES: {rhea_uniq_no_rxn_smiles}"
        )
        self.logger.info(f"Rhea unique reactions: {rhea_uniq_rxn}")
        self.logger.info(
            f"Percentage of Rhea unique reactions without Reaction SMILES: {rhea_uniq_no_rxn_smiles / rhea_uniq_rxn * 100:.2f}%"
        )

        self.logger.info(
            f"Rhea reactions without Reaction SMILES: {rhea_no_rxn_smiles}"
        )
        self.logger.info(f"Total Rhea reactions: {total_rhea_reactions}")
        self.logger.info(
            f"Percentage of rhea reactions without Reaction SMILES: {rhea_no_rxn_smiles / total_rhea_reactions * 100:.2f}%"
        )

        self.logger.info(
            f"Kegg unique reactions without Reaction SMILES: {kegg_uniq_no_rxn_smiles}"
        )
        self.logger.info(f"Kegg unique reactions: {kegg_uniq_rxn}")
        self.logger.info(
            f"Percentage of Kegg unique reactions without Reaction SMILES: {kegg_uniq_no_rxn_smiles / kegg_uniq_rxn * 100:.2f}%"
        )

        self.logger.info(
            f"Kegg reactions without Reaction SMILES: {kegg_no_rxn_smiles}"
        )
        self.logger.info(f"Total Kegg reactions: {total_kegg_reactions}")
        self.logger.info(
            f"Percentage of kegg reactions without Reaction SMILES: {kegg_no_rxn_smiles / total_kegg_reactions * 100:.2f}%"
        )

        self.logger.info(
            f"SABIO unique reactions without Reaction SMILES: {sabio_uniq_no_rxn_smiles}"
        )
        self.logger.info(f"SABIO unique reactions: {sabio_uniq_rxn}")
        self.logger.info(
            f"Percentage of SABIO unique reactions without Reaction SMILES: {sabio_uniq_no_rxn_smiles / sabio_uniq_rxn * 100:.2f}%"
        )

        self.logger.info(
            f"SABIO reactions without Reaction SMILES: {sabio_no_rxn_smiles}"
        )
        self.logger.info(f"Total SABIO reactions: {total_sabio_reactions}")
        self.logger.info(
            f"Percentage of sabio reactions without Reaction SMILES: {sabio_no_rxn_smiles / total_sabio_reactions * 100:.2f}%"
        )

        self.logger.info(
            f"Metacyc unique reactions without Reaction SMILES: {metacyc_uniq_no_rxn_smiles}"
        )
        self.logger.info(f"Metacyc unique reactions: {metacyc_uniq_rxn}")
        self.logger.info(
            f"Percentage of Metacyc unique reactions without Reaction SMILES: {metacyc_uniq_no_rxn_smiles / metacyc_uniq_rxn * 100:.2f}%"
        )

        self.logger.info(
            f"Metacyc reactions without Reaction SMILES: {metacyc_no_rxn_smiles}"
        )
        self.logger.info(f"Total Metacyc reactions: {total_metacyc_reactions}")
        self.logger.info(
            f"Percentage of metacyc reactions without Reaction SMILES: {metacyc_no_rxn_smiles / total_metacyc_reactions * 100:.2f}%"
        )

        self.logger.info(
            f"Total unique reactions without Reaction SMILES: {total_uniq_no_rxn_smiles}"
        )
        self.logger.info(f"Total unique reactions: {total_uniq_rxn}")
        self.logger.info(
            f"Percentage of unique reactions without Reaction SMILES: {total_uniq_no_rxn_smiles / total_uniq_rxn * 100:.2f}%"
        )

        self.logger.info(f"Total reactions: {len(master_reactions_df)}")
        self.logger.info(
            f"Total reactions without Reaction SMILES: {len(master_reactions_df[master_reactions_df['rxn_smiles'].isna()])}"
        )
        self.logger.info(
            f"Percentage of total reactions without Reaction SMILES: {len(master_reactions_df[master_reactions_df['rxn_smiles'].isna()]) / len(master_reactions_df) * 100:.2f}%"
        )

        self.logger.info(
            f"Rhea contribution to total reactions: {total_rhea_reactions / len(master_reactions_df) * 100:.2f}%"
        )
        self.logger.info(
            f"Kegg contribution to total reactions: {total_kegg_reactions / len(master_reactions_df) * 100:.2f}%"
        )
        self.logger.info(
            f"SABIO contribution to total reactions: {total_sabio_reactions / len(master_reactions_df) * 100:.2f}%"
        )
        self.logger.info(
            f"Metacyc contribution to total reactions: {total_metacyc_reactions / len(master_reactions_df) * 100:.2f}%"
        )

    def canonicalize_rxn_smiles_column(self, master_reactions_df):
        if "rxn_smiles" not in master_reactions_df.columns:
            self.logger.warning(
                "Column 'rxn_smiles' not found; skipping reaction SMILES canonicalization."
            )
            return master_reactions_df

        canonical_rxn_smiles, _ = chem_utils.canonicalize_reaction_smiles_series(
            master_reactions_df["rxn_smiles"], logger=self.logger
        )
        master_reactions_df = master_reactions_df.copy()
        master_reactions_df["rxn_smiles"] = canonical_rxn_smiles.values
        return master_reactions_df

    def save_reaction_outcome_dataset(self, reactions_df):
        self.logger.info("Building reaction outcome dataset...")
        required_columns = ["rxn_smiles", "uniprot_date"]
        missing_columns = set(required_columns) - set(reactions_df.columns)
        if missing_columns:
            raise KeyError(
                "Cannot build reaction outcome dataset. Missing columns: "
                f"{sorted(missing_columns)}"
            )

        reaction_outcome_df = (
            reactions_df[required_columns]
            .drop_duplicates()
            .dropna(subset=["rxn_smiles"])
            .groupby(["rxn_smiles"])
            .agg({"uniprot_date": "min"})
            .reset_index()
        )
        reaction_outcome_df[["reactants", "products"]] = reaction_outcome_df[
            "rxn_smiles"
        ].str.split(">>", n=1, expand=True)

        self.logger.info(
            f"Reaction outcome dataset rows: {len(reaction_outcome_df)}; "
            f"unique rxn_smiles: {reaction_outcome_df['rxn_smiles'].nunique()}"
        )
        self.logger.info(
            "Saving reaction outcome dataset to "
            f"{self.REACTION_OUTCOME_DATASET_PARQUET_FILE_PATH}..."
        )
        reaction_outcome_df.to_parquet(
            self.REACTION_OUTCOME_DATASET_PARQUET_FILE_PATH,
            index=False,
            compression="brotli",
        )
        self.logger.info("Reaction outcome dataset saved successfully.")
        return reaction_outcome_df

    def setup(self):
        rhea_reactions_df = self.parse_rhea_reactions()
        kegg_reactions_df = self.parse_kegg_reactions(rhea_reactions_df)
        rhea_reactions_df, kegg_reactions_df, sabio_reactions_df = (
            self.parse_sabio_reactions(rhea_reactions_df, kegg_reactions_df)
        )

        try:
            self.logger.info(
                "Unifying de-overlapped reactions from Rhea, KEGG, and SABIO..."
            )
            master_reactions_df = pd.concat(
                [rhea_reactions_df, kegg_reactions_df, sabio_reactions_df],
                ignore_index=True,
            )
        except Exception as e:
            self.logger.error(
                f"Error while unifying reactions from Rhea, KEGG, and SABIO: {e}"
            )
            raise e

        master_reactions_df = self.add_metacyc_reactions(master_reactions_df)

        if "ec" not in master_reactions_df.columns:
            self.logger.warning(
                "Column 'ec' not found; skipping EC normalization/explosion."
            )
        else:
            self.logger.info(
                "Normalizing EC values: exploding mixed strings and padding partial ECs to 4 levels..."
            )
            original_rows = len(master_reactions_df)
            master_reactions_df = master_reactions_df.copy()
            master_reactions_df["ec"] = master_reactions_df["ec"].apply(
                lambda x: normalize_ec_collection(x, fallback="-.-.-.-")
            )
            master_reactions_df = master_reactions_df.explode("ec", ignore_index=True)
            self.logger.info(
                "Normalized/exploded EC column: "
                f"rows {original_rows} -> {len(master_reactions_df)}, "
                f"unique_ec={master_reactions_df['ec'].nunique(dropna=True)}"
            )

        master_reactions_df = self.canonicalize_rxn_smiles_column(master_reactions_df)

        self.logger.info(
            "None-ing empty reaction SMILES in master reactions DataFrame..."
        )
        master_reactions_df.loc[
            (master_reactions_df["rxn_smiles"] == ">>"), "rxn_smiles"
        ] = None

        self.log_stats(master_reactions_df)

        try:
            for col in ["pubmed_id", "reactome_id", "pdbs", "sabio_id", "metanetx_id"]:
                self.logger.info(
                    f"Homogenizing {col} column's data types for master reactions DataFrame..."
                )

                self.logger.info(f"Converting np ndarrays to list of strings...")
                mask = master_reactions_df[col].apply(
                    lambda x: isinstance(x, np.ndarray)
                )
                master_reactions_df.loc[mask, col] = master_reactions_df.loc[
                    mask, col
                ].apply(lambda x: list(map(str, x)))

                self.logger.info(
                    f"Converting non-list, non-array values to list of strings or empty list respectively..."
                )
                mask = master_reactions_df[col].apply(lambda x: not isinstance(x, list))
                master_reactions_df.loc[mask, col] = master_reactions_df.loc[
                    mask, col
                ].apply(lambda x: [str(x)] if x and pd.notna(x) else [])
        except Exception as e:
            self.logger.error(f"Error while homogenizing column data types: {e}")
            raise e

        self.pickle_all_rxn_smiles(master_reactions_df)
        self.pickle_all_sequences(master_reactions_df)
        master_reactions_df = add_uniprot_date_column(
            master_reactions_df, verbose=True
        )
        self.save_reaction_outcome_dataset(master_reactions_df)

        datasets_to_save = []
        if self.SAVE_1D:
            datasets_to_save.append(
                (
                    master_reactions_df.copy(),
                    self.UNIFIED_REACTIONS_1D_PARQUET_FILE_PATH,
                    "1D",
                )
            )

        if self.SAVE_3D:
            annotated_master_reactions_df = assign_experimental_and_af_pdbs(
                master_reactions_df.copy()
            )
            datasets_to_save.append(
                (
                    annotated_master_reactions_df,
                    self.UNIFIED_REACTIONS_3D_PARQUET_FILE_PATH,
                    "3D",
                )
            )

        def _save_dataset(df, path, dataset_kind):
            self.logger.info(
                f"Saving unified {dataset_kind} reactions DataFrame to {path}..."
            )
            df.to_parquet(path, index=False, compression="brotli")
            self.logger.info(
                f"Unified {dataset_kind} reactions DataFrame saved successfully."
            )

        try:
            if len(datasets_to_save) == 1:
                _save_dataset(*datasets_to_save[0])
            else:
                with ThreadPoolExecutor(max_workers=len(datasets_to_save)) as executor:
                    futures = [
                        executor.submit(_save_dataset, df, path, dataset_kind)
                        for df, path, dataset_kind in datasets_to_save
                    ]
                    for future in futures:
                        future.result()
        except Exception as e:
            self.logger.error(
                f"Error while saving unified reactions DataFrame(s) to parquet: {e}"
            )
            raise e

        self.logger.info("Downloading PubMed abstracts...")
        download_pubmed_abstracts_parallel(
            master_reactions_df["pubmed_id"].explode().dropna().unique()
        )
        self.logger.info("PubMed abstracts downloaded successfully.")
        self.logger.info("Parsing all reference XMLs...")
        master_references_df = pd.DataFrame(parse_all_reference_xmls())

        try:
            self.logger.info("Saving master references DataFrame...")
            master_references_df.to_parquet(
                self.UNIFIED_REACTIONS_REFERENCES_PARQUET_FILE_PATH,
                index=False,
                compression="brotli",
            )
            self.logger.info("Master references DataFrame saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving master references DataFrame: {e}")
            raise

        return (
            annotated_master_reactions_df if self.SAVE_3D else master_reactions_df,
            master_references_df,
        )


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
    builder = ReactionsDatasetBuilder(cfg)
    unified_reactions_df, unified_reactions_references_df = builder.setup()
