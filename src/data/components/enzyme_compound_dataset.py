import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from rdkit.Chem import AllChem
from omegaconf import DictConfig
from pandarallel import pandarallel
from rdkit import Chem, DataStructs
from hydra import initialize, compose

from src.utils.chem_utils import ChemUtils
from src.utils.tqdmlogger import TqdmLogger

chem_utils = ChemUtils()
canonicalize_smiles = chem_utils.canonicalize_smiles
add_uniprot_date_column = chem_utils.add_uniprot_date_column


class EnzymeCompoundDatasetBuilder:
    def __init__(self, cfg: DictConfig):
        LOG_PATH = Path(cfg.enzyme_compound_dataset.log_dir)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.enzyme_compound_dataset.log_file_name
        ).get_logger()

        self.RANDOM_SEED = cfg.enzyme_compound_dataset.random_seed
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)

        self.PANDARALLEL_NB_WORKERS = min(
            cfg.enzyme_compound_dataset.pandarallel.nb_workers, os.cpu_count()
        )
        self.PANDARALLEL_PROGRESS_BAR = (
            cfg.enzyme_compound_dataset.pandarallel.progress_bar or False
        )
        self.PANDARALLEL_PROGRESS_BAR_NEGATIVE_PAIRS = (
            cfg.enzyme_compound_dataset.pandarallel.progress_bar_negative_pairs or True
        )

        self.N_NEG = cfg.enzyme_compound_dataset.n_neg
        self.SIM_UPPER = cfg.enzyme_compound_dataset.sim_upper
        self.SIM_LOWER = cfg.enzyme_compound_dataset.sim_lower
        self.SIM_STEP = cfg.enzyme_compound_dataset.sim_step

        self.UNIFIED_REACTIONS_PARQUET_FILE_PATH = Path(
            cfg.enzyme_compound_dataset.unified_reactions_parquet_file_path
        )
        if not self.UNIFIED_REACTIONS_PARQUET_FILE_PATH.exists():
            self.logger.error(
                f"Input file {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH} does not exist."
            )
            raise FileNotFoundError(
                f"Input file {self.UNIFIED_REACTIONS_PARQUET_FILE_PATH} does not exist."
            )

        self.ENZYME_SUBSTRATE_DATASET_PARQUET_FILE_PATH = Path(
            cfg.enzyme_compound_dataset.enzyme_substrate_dataset_parquet_file_path
        )
        self.ENZYME_SUBSTRATE_POSITIVE_DATASET_PARQUET_FILE_PATH = Path(
            cfg.enzyme_compound_dataset.enzyme_substrate_positive_dataset_parquet_file_path
        )
        self.ENZYME_SUBSTRATE_NEGATIVE_DATASET_PARQUET_FILE_PATH = Path(
            cfg.enzyme_compound_dataset.enzyme_substrate_negative_dataset_parquet_file_path
        )
        self.ENZYME_SUBSTRATE_SEQUENCES_PICKLE_PATH = Path(
            cfg.enzyme_compound_dataset.enzyme_substrate_sequences_pickle_path
        )
        self.ENZYME_SUBSTRATE_SMILES_PICKLE_PATH = Path(
            cfg.enzyme_compound_dataset.enzyme_substrate_smiles_pickle_path
        )

        for path in (
            self.ENZYME_SUBSTRATE_DATASET_PARQUET_FILE_PATH.parent,
            self.ENZYME_SUBSTRATE_POSITIVE_DATASET_PARQUET_FILE_PATH.parent,
            self.ENZYME_SUBSTRATE_NEGATIVE_DATASET_PARQUET_FILE_PATH.parent,
            self.ENZYME_SUBSTRATE_SEQUENCES_PICKLE_PATH.parent,
            self.ENZYME_SUBSTRATE_SMILES_PICKLE_PATH.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def extract_substrates(self, rxn_smiles):
        if pd.isna(rxn_smiles):
            return []
        try:
            left = rxn_smiles.split(">>")[0]
            return left.split(".")
        except Exception:
            return []

    def make_substrate_enzyme_df(self, df):
        self.logger.info("Building substrate-enzyme pairs...")
        pair_df = df[
            [
                "ec",
                "rxn_smiles",
                "uniprot_id",
                "sequence",
                "source",
                "rhea_id",
                "kegg_id",
                "metacyc_id",
                "sabio_id",
                "pdbs",
                "pdb_source",
                "pdb_type",
            ]
        ].copy()
        pair_df["smiles"] = pair_df["rxn_smiles"].apply(self.extract_substrates)
        return (
            pair_df.explode("smiles")
            .drop(columns=["rxn_smiles"])
            .dropna(subset=["smiles"])
            .rename(columns={"ec": "enzyme"})
            .reset_index(drop=True)
        )

    def smiles_to_fp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)

    def get_negatives_for_row(self, row, seq2pos_substrates, all_smiles, smiles2fp):
        sequence = row["sequence"]
        substrate = row["smiles"]
        fp = smiles2fp.get(substrate)
        if fp is None or pd.isna(sequence):
            return []
        # Exclude all positive substrates for this sequence
        pos_substrates = seq2pos_substrates.get(sequence, set())
        candidates = [s for s in all_smiles if s not in pos_substrates]
        cands_fps = [(s, smiles2fp[s]) for s in candidates if smiles2fp[s] is not None]
        sims = [(s, DataStructs.FingerprintSimilarity(fp, fp2)) for s, fp2 in cands_fps]
        found = False
        lower = self.SIM_LOWER
        while not found:
            pool = [s for s, sim in sims if lower <= sim <= self.SIM_UPPER]
            if len(pool) >= self.N_NEG:
                found = True
            else:
                lower = max(0, lower - self.SIM_STEP)
                if lower == 0:
                    pool = [s for s, sim in sims if sim <= self.SIM_UPPER]
                    found = True
        return pool

    def build_negative_pairs_pandarallel(self, positive_df):
        pandarallel.initialize(
            progress_bar=self.PANDARALLEL_PROGRESS_BAR_NEGATIVE_PAIRS,
            nb_workers=self.PANDARALLEL_NB_WORKERS,
        )

        all_smiles = positive_df["smiles"].unique()
        smiles2fp = {
            s: self.smiles_to_fp(s)
            for s in tqdm(all_smiles, desc="Computing fingerprints")
        }
        pos_counts = positive_df["smiles"].value_counts().to_dict()
        neg_counts = {s: 0 for s in all_smiles}
        # Build mapping: sequence -> set of positive substrates
        seq2pos_substrates = (
            positive_df.groupby("sequence")["smiles"].apply(set).to_dict()
        )
        candidate_pools = positive_df.parallel_apply(
            lambda row: self.get_negatives_for_row(
                row, seq2pos_substrates, all_smiles, smiles2fp
            ),
            axis=1,
        )

        negative_records = []
        for row, pool in tqdm(
            zip(
                positive_df[["sequence", "uniprot_id"]].itertuples(index=False),
                candidate_pools,
            ),
            desc="Building negative pairs pool",
        ):
            filtered_pool = [
                s for s in pool if neg_counts[s] < self.N_NEG * pos_counts.get(s, 1)
            ]
            sampled = random.sample(filtered_pool, min(self.N_NEG, len(filtered_pool)))
            for neg_smiles in sampled:
                negative_records.append(
                    {
                        "sequence": row.sequence,
                        "uniprot_id": row.uniprot_id,
                        "smiles": neg_smiles,
                        "label": 0,
                    }
                )
                neg_counts[neg_smiles] += 1
        return pd.DataFrame(negative_records)

    def build_classification_dataset(self, positive_pairs):
        positive_pairs = positive_pairs.copy()
        positive_pairs["label"] = 1
        positive_pairs = add_uniprot_date_column(positive_pairs, verbose=True)
        self.logger.info(
            f"Total positive enzyme-substrate pairs: {len(positive_pairs)}"
        )
        self.logger.info(
            f"Total unique enzymes in positive pairs: {positive_pairs['uniprot_id'].nunique()}"
        )
        self.logger.info(
            f"Total unique substrates in positive pairs: {positive_pairs['smiles'].nunique()}"
        )
        self.logger.info(
            f"Saving positive enzyme-substrate pairs to {self.ENZYME_SUBSTRATE_POSITIVE_DATASET_PARQUET_FILE_PATH}"
        )
        positive_pairs.to_parquet(
            self.ENZYME_SUBSTRATE_POSITIVE_DATASET_PARQUET_FILE_PATH
        )

        self.logger.info(
            "Building negative enzyme-substrate pairs using:\n"
            f"- N_NEG={self.N_NEG}\n"
            f"- SIM_UPPER={self.SIM_UPPER}\n"
            f"- SIM_LOWER={self.SIM_LOWER}\n"
            f"- SIM_STEP={self.SIM_STEP}\n"
        )
        negative_pairs = self.build_negative_pairs_pandarallel(positive_pairs).merge(
            positive_pairs[
                [
                    "uniprot_id",
                    "sequence",
                    "uniprot_date",
                    "pdbs",
                    "pdb_source",
                    "pdb_type",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True),
            on=["uniprot_id", "sequence"],
            how="left",
        )
        self.logger.info(
            f"Total negative enzyme-substrate pairs: {len(negative_pairs)}"
        )
        self.logger.info(
            f"Total unique enzymes in negative pairs: {negative_pairs['uniprot_id'].nunique()}"
        )
        self.logger.info(
            f"Total unique substrates in negative pairs: {negative_pairs['smiles'].nunique()}"
        )
        self.logger.info(
            f"Saving negative enzyme-substrate pairs to {self.ENZYME_SUBSTRATE_NEGATIVE_DATASET_PARQUET_FILE_PATH}"
        )
        negative_pairs.to_parquet(
            self.ENZYME_SUBSTRATE_NEGATIVE_DATASET_PARQUET_FILE_PATH,
            index=False,
        )

        substrate_enzyme_df = (
            pd.concat(
                [
                    positive_pairs.drop(
                        columns=[
                            "source",
                            "rhea_id",
                            "kegg_id",
                            "metacyc_id",
                            "sabio_id",
                            "enzyme",
                        ]
                    ),
                    negative_pairs,
                ]
            )
            .drop_duplicates()
            .dropna(subset=["uniprot_id", "sequence", "smiles", "label"])
            .sort_values(["sequence", "label", "smiles"])[
                [
                    "uniprot_id",
                    "sequence",
                    "uniprot_date",
                    "smiles",
                    "label",
                    "pdbs",
                    "pdb_source",
                    "pdb_type",
                ]
            ]
            .reset_index(drop=True)
        )

        self.logger.info(f"Total enzyme-substrate pairs: {len(substrate_enzyme_df)}")
        self.logger.info(
            f"Total unique enzymes in all pairs: {substrate_enzyme_df['uniprot_id'].nunique()}"
        )
        self.logger.info(
            f"Total unique substrates in all pairs: {substrate_enzyme_df['smiles'].nunique()}"
        )

        self.logger.info(
            f"Saving enzyme sequences to {self.ENZYME_SUBSTRATE_SEQUENCES_PICKLE_PATH}"
        )
        with open(self.ENZYME_SUBSTRATE_SEQUENCES_PICKLE_PATH, "wb") as f:
            pickle.dump(
                substrate_enzyme_df[["uniprot_id", "sequence"]]
                .rename(columns={"uniprot_id": "acc_id"})
                .drop_duplicates()
                .dropna()
                .to_dict(orient="records"),
                f,
            )

        self.logger.info(
            f"Saving substrate SMILES to {self.ENZYME_SUBSTRATE_SMILES_PICKLE_PATH}"
        )
        with open(self.ENZYME_SUBSTRATE_SMILES_PICKLE_PATH, "wb") as f:
            pickle.dump(
                substrate_enzyme_df[["smiles"]]
                .drop_duplicates()
                .dropna()
                .to_dict(orient="records"),
                f,
            )

        self.logger.info(
            f"Saving enzyme-substrate pairs to {self.ENZYME_SUBSTRATE_DATASET_PARQUET_FILE_PATH}"
        )
        substrate_enzyme_df.to_parquet(
            self.ENZYME_SUBSTRATE_DATASET_PARQUET_FILE_PATH,
            index=False,
            compression="brotli",
        )

    def setup(self):
        reactions_df = pd.read_parquet(self.UNIFIED_REACTIONS_PARQUET_FILE_PATH)

        positive_pairs = self.make_substrate_enzyme_df(reactions_df).explode("sabio_id")
        self.logger.info(
            "Aggregating enzyme-substrate pairs to unique (smiles, uniprot_id) combinations"
        )
        self.logger.info(
            f"Total enzyme-substrate pairs before aggregation: {len(positive_pairs)}"
        )
        positive_pairs = (
            positive_pairs.groupby(["smiles", "uniprot_id", "sequence", "pdbs"])
            .agg(
                {
                    "enzyme": set,
                    "source": set,
                    "rhea_id": set,
                    "kegg_id": set,
                    "metacyc_id": set,
                    "sabio_id": set,
                    "pdb_source": "first",
                    "pdb_type": "first",
                }
            )
            .reset_index()
        )
        self.logger.info(
            f"Total enzyme-substrate pairs after aggregation: {len(positive_pairs)}"
        )

        self.logger.info("Canonicalizing substrate SMILES...")
        pandarallel.initialize(
            progress_bar=self.PANDARALLEL_PROGRESS_BAR,
            nb_workers=self.PANDARALLEL_NB_WORKERS,
        )
        unique_positive_pair_smiles = positive_pairs[["smiles"]].drop_duplicates()
        unique_positive_pair_smiles["canonical_smiles"] = (
            unique_positive_pair_smiles["smiles"]
            .parallel_apply(canonicalize_smiles)
            .dropna()
        )
        old_size = len(positive_pairs)
        positive_pairs = (
            positive_pairs.merge(unique_positive_pair_smiles, on="smiles", how="inner")
            .dropna(subset=["canonical_smiles"])
            .drop(columns=["smiles"])
            .rename(columns={"canonical_smiles": "smiles"})
        )
        self.logger.info(
            f"Dropped {old_size - len(positive_pairs)} pairs due to invalid SMILES. Total valid enzyme-substrate pairs: {len(positive_pairs)}"
        )

        positive_pairs = positive_pairs.dropna(subset=["sequence"])
        self.build_classification_dataset(positive_pairs)


if __name__ == "__main__":
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")
    builder = EnzymeCompoundDatasetBuilder(cfg)
    builder.setup()
