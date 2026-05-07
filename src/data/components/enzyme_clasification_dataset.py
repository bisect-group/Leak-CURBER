import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import pickle
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
add_uniprot_date_column = chem_utils.add_uniprot_date_column


class EnzymeClassificationDatasetBuilder:
    OUTPUT_COLUMNS = [
        "uniprot_id",
        "sequence",
        "ec_number",
        "uniprot_date",
        "pdbs",
        "pdb_source",
        "pdb_type",
    ]

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        LOG_PATH = Path(cfg.enzyme_classification_dataset.log_dir)
        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.enzyme_classification_dataset.log_file_name,
        ).get_logger()

        self.UNIFIED_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.enzyme_classification_dataset.unified_reactions_parquet_file_path
        )
        if not self.UNIFIED_REACTIONS_PARQUET_FILE_PATH.exists():
            self.logger.error(
                f"Input file {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH} does not exist."
            )
            raise FileNotFoundError(
                f"Input file {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH} does not exist."
            )

        self.ENZYME_DATASET_PARQUET_FILE_PATH = Path(
            cfg.enzyme_classification_dataset.enzyme_classification_dataset_parquet_file_path
        )
        self.ENZYME_SEQUENCES_PICKLE_PATH = Path(
            cfg.enzyme_classification_dataset.enzyme_sequences_pickle_path
        )

        for path in (
            self.ENZYME_DATASET_PARQUET_FILE_PATH.parent,
            self.ENZYME_SEQUENCES_PICKLE_PATH.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _as_ec_list(self, value):
        if pd.isna(value):
            return []
        if isinstance(value, (set, tuple, list)):
            return list(value)
        return [value]

    def _run_splits(self):
        from src.data.components.splitters import (
            ECHierarchicalGroupSplitter,
            ProteinSeqMaxFidentSimilaritySplitter,
            ProteinStructMaxLDDTSimilaritySplitter,
            RandomSplitter,
            UniProtTimeBasedSplitter,
        )

        splits_cfg = compose(
            config_name="data_processing",
            overrides=[
                "data/splits_dataset@splits.dataset=enzyme_classification",
                "splits.ec_split_column=ec_number",
                "splits.ec_split_enabled=true",
            ],
        )

        self.logger.info("Running enzyme sequence (fident) splits...")
        ProteinSeqMaxFidentSimilaritySplitter(cfg=splits_cfg).generate_splits()

        self.logger.info("Running enzyme structure (LDDT) splits...")
        ProteinStructMaxLDDTSimilaritySplitter(cfg=splits_cfg).generate_splits()

        self.logger.info("Running EC hierarchical splits (L1-L4)...")
        ECHierarchicalGroupSplitter(cfg=splits_cfg).generate_splits()

        self.logger.info("Running random splits...")
        RandomSplitter(cfg=splits_cfg).generate_splits()

        if splits_cfg.splits.get("time_split_enabled", False):
            self.logger.info("Running UniProt time-based splits...")
            UniProtTimeBasedSplitter(cfg=splits_cfg).generate_splits()

    def setup(self):
        self.logger.info("Setting up enzyme classification dataset...")
        self.logger.info(f"Reading unified reactions from {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH}")
        reactions_df = pd.read_parquet(self.UNIFIED_REACTIONS_PARQUET_FILE_PATH)

        enzyme_df = reactions_df[
            [
                "uniprot_id",
                "sequence",
                "ec",
                "pdbs",
                "pdb_source",
                "pdb_type",
            ]
        ].rename(columns={"ec": "ec_number"})

        self.logger.info("Processing enzyme entries...")
        enzyme_df["ec_number"] = enzyme_df["ec_number"].apply(self._as_ec_list)
        enzyme_df = (
            enzyme_df.explode("ec_number")
            .explode("pdbs")
            .dropna(subset=["uniprot_id", "sequence", "ec_number"])
            .drop_duplicates()
            .sort_values(["uniprot_id", "ec_number"])
            .reset_index(drop=True)
        )

        self.logger.info(
            f"Cleaning duplicate entries and restricting to {self.OUTPUT_COLUMNS} columns..."
        )
        enzyme_df = add_uniprot_date_column(
            (
                enzyme_df.reindex(columns=self.OUTPUT_COLUMNS)
                .drop_duplicates()
                .reset_index(drop=True)
            ),
            verbose=True,
        )

        self.logger.info(f"Total enzyme entries: {len(enzyme_df)}")
        self.logger.info(f"Unique uniprot_ids: {enzyme_df['uniprot_id'].nunique()}")
        self.logger.info(f"Unique ec_numbers: {enzyme_df['ec_number'].nunique()}")
        self.logger.info(f"Unique PDBs: {enzyme_df['pdbs'].dropna().nunique()}")
        self.logger.info(
            f"Entries with PDB: {enzyme_df['pdbs'].notna().sum()} "
            f"({enzyme_df['pdbs'].notna().mean() * 100:.1f}%)"
        )
        self.logger.info(
            f"Entries without PDB: {enzyme_df['pdbs'].isna().sum()} "
            f"({enzyme_df['pdbs'].isna().mean() * 100:.1f}%)"
        )
        self.logger.info(
            f"EC level distribution: "
            + ", ".join(
                f"level-{i}: {(enzyme_df['ec_number'].str.count(r'\.') == i - 1).sum()}"
                for i in range(1, 5)
            )
        )

        self.logger.info(
            f"Saving enzyme sequences to {self.ENZYME_SEQUENCES_PICKLE_PATH}"
        )
        with open(self.ENZYME_SEQUENCES_PICKLE_PATH, "wb") as f:
            pickle.dump(
                enzyme_df[["uniprot_id", "sequence"]]
                .rename(columns={"uniprot_id": "acc_id"})
                .drop_duplicates()
                .dropna()
                .to_dict(orient="records"),
                f,
            )

        self.logger.info(
            f"Saving enzyme dataset to {self.ENZYME_DATASET_PARQUET_FILE_PATH}"
        )
        enzyme_df.to_parquet(
            self.ENZYME_DATASET_PARQUET_FILE_PATH,
            index=False,
            compression="brotli",
        )

        if self.cfg.enzyme_classification_dataset.run_splits:
            self._run_splits()
        else:
            self.logger.info("Skipping splits (run_splits=false).")


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
        builder = EnzymeClassificationDatasetBuilder(cfg)
        builder.setup()
