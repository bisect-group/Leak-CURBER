import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pickle
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig

from src.utils.tqdmlogger import TqdmLogger


class EnzymeRetrievalDatasetBuilder:
    OUTPUT_COLUMNS = ["rxn_smiles", "ec_number", "uniprot_date"]
    PROTEIN_OUTPUT_COLUMNS = OUTPUT_COLUMNS + [
        "sequence",
        "pdbs",
        "pdb_source",
        "pdb_type",
    ]

    def __init__(self, cfg: DictConfig):
        LOG_PATH = Path(cfg.enzyme_retrieval_dataset.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.enzyme_retrieval_dataset.log_file_name,
        ).get_logger()

        self.UNIFIED_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.enzyme_retrieval_dataset.unified_reactions_parquet_file_path
        )
        if not self.UNIFIED_REACTIONS_PARQUET_FILE_PATH.exists():
            self.logger.error(
                f"Input file {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH} does not exist."
            )
            raise FileNotFoundError(
                f"Input file {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH} does not exist."
            )

        self.INCLUDE_PROTEIN_INFORMATION = cfg.enzyme_retrieval_dataset.get(
            "include_protein_information", False
        )
        if self.INCLUDE_PROTEIN_INFORMATION:
            dataset_path_key = (
                "enzyme_retrieval_dataset_with_protein_information_parquet_file_path"
            )
            rxn_smiles_path_key = (
                "enzyme_retrieval_with_protein_information_rxn_smiles_pickle_path"
            )
        else:
            dataset_path_key = "enzyme_retrieval_dataset_parquet_file_path"
            rxn_smiles_path_key = "enzyme_retrieval_rxn_smiles_pickle_path"

        self.ENZYME_RETRIEVAL_DATASET_PARQUET_FILE_PATH = Path(
            cfg.enzyme_retrieval_dataset[dataset_path_key]
        )
        self.ENZYME_RETRIEVAL_RXN_SMILES_PICKLE_PATH = Path(
            cfg.enzyme_retrieval_dataset[rxn_smiles_path_key]
        )

        for path in (
            self.ENZYME_RETRIEVAL_DATASET_PARQUET_FILE_PATH.parent,
            self.ENZYME_RETRIEVAL_RXN_SMILES_PICKLE_PATH.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _as_list(self, value):
        if isinstance(value, (set, tuple, list)):
            return list(value)
        if hasattr(value, "tolist") and not isinstance(value, str):
            return value.tolist()
        if pd.isna(value):
            return []
        return [value]

    def _log_unique_ec_count(self, df, stage):
        unique_ec_count = df["ec_number"].dropna().nunique()
        self.logger.info(f"Unique ec_numbers {stage}: {unique_ec_count}")

    def _drop_missing_required_columns(self, df, required_columns):
        for column in required_columns:
            missing_rows = df[column].isna().sum()
            self.logger.info(f"Rows missing {column}: {missing_rows}")
            df = df.dropna(subset=[column])
            self._log_unique_ec_count(df, f"after dropping rows missing {column}")
        return df

    def setup(self):
        reactions_df = pd.read_parquet(self.UNIFIED_REACTIONS_PARQUET_FILE_PATH)
        input_columns = ["rxn_smiles", "ec", "uniprot_date"]
        dropna_columns = ["rxn_smiles", "ec_number"]
        groupby_columns = ["rxn_smiles", "ec_number"]
        sort_columns = ["ec_number", "rxn_smiles"]
        output_columns = self.OUTPUT_COLUMNS

        if self.INCLUDE_PROTEIN_INFORMATION:
            input_columns.extend(["sequence", "pdbs", "pdb_source", "pdb_type"])
            groupby_columns.extend(["sequence", "pdbs", "pdb_source", "pdb_type"])
            sort_columns.extend(["sequence", "pdbs", "pdb_source", "pdb_type"])
            output_columns = self.PROTEIN_OUTPUT_COLUMNS
            self.logger.info(
                "Including protein sequence, PDB, PDB source, and PDB type information."
            )

        retrieval_df = (
            reactions_df[input_columns]
            .rename(columns={"ec": "ec_number"})
            .reset_index(drop=True)
        )

        retrieval_df["ec_number"] = retrieval_df["ec_number"].apply(self._as_list)
        self._log_unique_ec_count(
            retrieval_df.explode("ec_number"),
            "after normalizing ec_number to lists",
        )

        retrieval_df = retrieval_df.explode("ec_number")
        self._log_unique_ec_count(retrieval_df, "after exploding ec_number")

        if self.INCLUDE_PROTEIN_INFORMATION:
            retrieval_df["pdbs"] = retrieval_df["pdbs"].apply(self._as_list)
            retrieval_df = retrieval_df.explode("pdbs")
            self._log_unique_ec_count(retrieval_df, "after exploding pdbs")

        retrieval_df = self._drop_missing_required_columns(
            retrieval_df, dropna_columns
        )
        self._log_unique_ec_count(
            retrieval_df,
            f"after dropping rows missing any of {dropna_columns}",
        )

        retrieval_df = retrieval_df.drop_duplicates()
        self._log_unique_ec_count(retrieval_df, "after dropping duplicates")

        retrieval_df = retrieval_df.sort_values(sort_columns).reset_index(drop=True)
        self._log_unique_ec_count(retrieval_df, "after sorting")

        retrieval_df = (
            retrieval_df.groupby(groupby_columns, dropna=False)
            .agg({"uniprot_date": "min"})
            .reset_index()
        )
        self._log_unique_ec_count(retrieval_df, "after grouping retrieval pairs")

        retrieval_df = retrieval_df[output_columns]

        self.logger.info(f"Total enzyme retrieval pairs: {len(retrieval_df)}")
        self.logger.info(f"Unique ec_numbers: {retrieval_df['ec_number'].nunique()}")
        self.logger.info(f"Unique rxn_smiles: {retrieval_df['rxn_smiles'].nunique()}")
        if self.INCLUDE_PROTEIN_INFORMATION:
            self.logger.info(f"Unique sequences: {retrieval_df['sequence'].nunique()}")
            self.logger.info(f"Unique PDBs: {retrieval_df['pdbs'].dropna().nunique()}")
            self.logger.info(
                f"Entries with PDB: {retrieval_df['pdbs'].notna().sum()} "
                f"({retrieval_df['pdbs'].notna().mean() * 100:.1f}%)"
            )

        self.logger.info(
            f"Saving retrieval rxn_smiles to {self.ENZYME_RETRIEVAL_RXN_SMILES_PICKLE_PATH}"
        )
        with open(self.ENZYME_RETRIEVAL_RXN_SMILES_PICKLE_PATH, "wb") as f:
            pickle.dump(
                retrieval_df[["rxn_smiles"]]
                .drop_duplicates()
                .dropna()
                .to_dict(orient="records"),
                f,
            )

        self.logger.info(
            f"Saving enzyme retrieval dataset to {self.ENZYME_RETRIEVAL_DATASET_PARQUET_FILE_PATH}"
        )
        retrieval_df.to_parquet(
            self.ENZYME_RETRIEVAL_DATASET_PARQUET_FILE_PATH,
            index=False,
            compression="brotli",
        )


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
    builder = EnzymeRetrievalDatasetBuilder(cfg)
    builder.setup()
