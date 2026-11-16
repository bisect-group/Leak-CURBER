import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from src.utils.rich_progress import progress
from omegaconf import DictConfig
from multiprocessing import Pool, cpu_count
from src.data.components.splitters.base import BaseSplitter, configured, greedy_fill, order_by_similarity

from rdkit.Chem import MolFromSmiles
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from src.utils.chem_utils import LazyChemUtils, is_tiny_or_inorganic
from src.utils.rich_progress import component_logger

chem_utils = LazyChemUtils()
canonicalize_smiles = chem_utils.bind("canonicalize_smiles")

# Global variable to hold fingerprints for worker processes
_FPS_GLOBAL = None
_FP_RADIUS, _FP_BITS = 2, 1024


def _init_worker(fps):
    """Initialize worker process with shared fingerprints."""
    global _FPS_GLOBAL
    _FPS_GLOBAL = fps


def _compute_max_sim_for_idx(idx):
    """
    Worker function to compute max similarity for a single fingerprint.
    Returns (idx, max_similarity) tuple.
    """
    global _FPS_GLOBAL
    all_sims = BulkTanimotoSimilarity(_FPS_GLOBAL[idx], _FPS_GLOBAL)
    all_sims[idx] = 0.0  # Exclude self-similarity
    return (idx, max(all_sims))


def _smiles_to_fp_worker(args):
    # Send the SMILES and return only the fingerprint, so no Mol objects are pickled.
    idx, smi = args
    return (idx, GetMorganFingerprintAsBitVect(MolFromSmiles(smi), _FP_RADIUS, _FP_BITS))


class SMILESMaxTanimotoSimilaritySplitter(BaseSplitter):
    def __init__(self, cfg: DictConfig):
        smiles_cfg = cfg.splits.dataset.smiles_split

        self.THRESHOLDS = list(
            smiles_cfg.get("similarity_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )

        self.PANDARALLEL_NB_WORKERS = min(
            cfg.splits.pandarallel.nb_workers, cpu_count()
        )
        self.PANDARALLEL_PROGRESS_BAR = cfg.splits.pandarallel.progress_bar or False

        self.TEST_FRAC = configured(cfg.splits.test_frac, 0.1)
        self.VAL_FRAC = configured(cfg.splits.valid_frac, 0.1)

        self.CACHE_OVERWRITE = cfg.splits.get("similarity_cache_overwrite", False)

        self.SMILES_SPLIT_INPUT_DATASET_PARQUET_FILE_PATH = Path(
            smiles_cfg.input_dataset_parquet_file_path
        )
        self.SMILES_SPLIT_OUTPUT_DIR = Path(smiles_cfg.output_dir)
        self.UNIQUE_SMILES_SIMILARITY_PLOT_PATH = Path(smiles_cfg.unique_similarity_plot_path)
        self.FULL_DATASET_SMILES_SIMILARITY_PLOT_PATH = Path(
            smiles_cfg.full_dataset_similarity_plot_path
        )

        LOG_PATH = Path(cfg.splits.log_dir)
        for path in [
            self.SMILES_SPLIT_OUTPUT_DIR,
            self.UNIQUE_SMILES_SIMILARITY_PLOT_PATH.parent,
            self.FULL_DATASET_SMILES_SIMILARITY_PLOT_PATH.parent,
            LOG_PATH,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = component_logger(
            log_dir=LOG_PATH, log_file_name=cfg.splits.log_file_name
        ).get_logger()

        self.MAX_SIM_CACHE_PATH = self.SMILES_SPLIT_OUTPUT_DIR / "max_tanimoto_similarities.tsv"

        if not self.SMILES_SPLIT_INPUT_DATASET_PARQUET_FILE_PATH.exists():
            msg = f"Dataset parquet file not found at {self.SMILES_SPLIT_INPUT_DATASET_PARQUET_FILE_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Loading full dataset...")
        self.full_dataset_df, self.unique_smiles_df = self._clean_full_dataset_df(
            pd.read_parquet(self.SMILES_SPLIT_INPUT_DATASET_PARQUET_FILE_PATH)
        )

    def _clean_full_dataset_df(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            from pandarallel import pandarallel
        except ImportError:
            msg = "pandarallel is not installed. Please install pandarallel to use this functionality."
            self.logger.error(msg)
            raise ImportError(msg)

        pandarallel.initialize(
            progress_bar=self.PANDARALLEL_PROGRESS_BAR,
            nb_workers=self.PANDARALLEL_NB_WORKERS,
        )

        self.logger.info(
            f"Before canonicalization, total unique SMILES count: {df['smiles'].nunique()}",
        )

        unique_smiles_df = df[["smiles"]].drop_duplicates().reset_index(drop=True)
        unique_smiles_df["canonical_smiles"] = unique_smiles_df[
            "smiles"
        ].parallel_apply(canonicalize_smiles)
        self.logger.info(
            f"After canonicalization, invalid SMILES count: {unique_smiles_df['canonical_smiles'].isna().sum()}"
        )

        # Group and fingerprint on molecule_key (desalted, neutralized, canonical
        # tautomer, stereo kept); `smiles` stays the datamol-canonical string for display.
        # A failed key falls back to canonical_smiles instead of dropping the row.
        unique_smiles_df["smiles_key"], n_fallback = chem_utils.molecule_key_series(
            unique_smiles_df["smiles"], fallback=unique_smiles_df["canonical_smiles"]
        )
        self.logger.info(f"molecule_key fell back to canonical SMILES for {n_fallback} SMILES.")
        unique_smiles_df = unique_smiles_df.dropna(subset=["canonical_smiles"])
        self.logger.info(
            f"After dropping invalid SMILES, total unique SMILES count: {unique_smiles_df['canonical_smiles'].nunique()}",
        )

        old_size = len(df)
        # The builders already write smiles_key; drop it so the merge does not suffix it.
        df = df.drop(columns=["smiles_key"], errors="ignore")
        df = (
            df.merge(
                unique_smiles_df,
                on="smiles",
                how="inner",
            )
            .drop(columns=["smiles"])
            .rename(columns={"canonical_smiles": "smiles"})
            .reset_index(drop=True)
        )
        self.logger.info(
            f"Dropped {old_size - len(df)} rows with invalid SMILES from full dataset."
            f" New dataset size: {len(df)} ({100 * (old_size - len(df)) / old_size:.2f} %)"
        )
        unique_smiles_df = (
            unique_smiles_df.drop(columns=["smiles"])
            .rename(columns={"canonical_smiles": "smiles"})
            # Everything is keyed on smiles_key; two keys can share one canonical SMILES, and
            # all their rows must stay. Dedupe on the key only.
            .drop_duplicates(subset="smiles_key")
            .reset_index(drop=True)
        )
        return df, unique_smiles_df

    def _smiles_to_fps(self, smiles: list) -> list:
        with multiprocessing.Pool() as pool:
            results = list(
                progress(
                    pool.imap_unordered(_smiles_to_fp_worker, enumerate(smiles), chunksize=256),
                    total=len(smiles),
                    desc="Generating fingerprints",
                    leave=False,
                )
            )
        # Sort by original index to restore order
        results.sort(key=lambda x: x[0])
        return [fp for idx, fp in results]

    def compute_max_similarities(self):
        """
        Compute max Tanimoto similarity to any other molecule for each molecule.
        This is threshold-independent and only needs to be computed once.
        Uses multiprocessing for parallel computation.

        Args:
            self: SubstrateSmilesSimilaritySplit instance
            n_workers: Number of worker processes. Defaults to cpu_count().

        Returns:
            tuple: (smiles_list, max_sim_to_dataset, smiles_to_row_count)
        """

        n_workers = cpu_count()

        smiles_list = self.unique_smiles_df["smiles_key"].tolist()
        n_unique_smiles = len(smiles_list)
        n_full_dataset = len(self.full_dataset_df)

        fingerprint = self._cache_fingerprint(
            smiles_list, fp="morgan", radius=_FP_RADIUS, n_bits=_FP_BITS
        )
        if not self.CACHE_OVERWRITE:
            loaded = self._load_similarity_cache(
                cache_path=self.MAX_SIM_CACHE_PATH,
                key_column="smiles_key",
                value_column="max_similarity",
                ordered_keys=smiles_list,
                fingerprint=fingerprint,
            )
            if loaded is not None:
                max_sim_to_dataset, _ = loaded
                smiles_to_row_count = self.full_dataset_df["smiles_key"].value_counts().to_dict()
                self.logger.info("Using cached max SMILES tanimoto similarities")
                return smiles_list, max_sim_to_dataset, smiles_to_row_count

        self.logger.info(f"Total unique SMILES: {n_unique_smiles}")
        self.logger.info(f"Full dataset size: {n_full_dataset}")

        # Generate fingerprints
        self.logger.info("Generating fingerprints...")
        fps = self._smiles_to_fps(smiles_list)

        # Compute max similarity to any other molecule for each molecule using multiprocessing
        self.logger.info(
            f"Computing max similarities using {n_workers} workers (this may take a while)..."
        )

        # Initialize result array
        max_sim_to_dataset = np.zeros(n_unique_smiles)

        # Use multiprocessing Pool with imap_unordered
        with Pool(
            processes=n_workers, initializer=_init_worker, initargs=(fps,)
        ) as pool:
            results = pool.imap_unordered(
                _compute_max_sim_for_idx, range(n_unique_smiles)
            )

            for idx, max_sim in progress(
                results,
                total=n_unique_smiles,
                desc="Computing max similarities",
                leave=False,
            ):
                max_sim_to_dataset[idx] = max_sim

        # Build a mapping from SMILES to row count in full dataset
        smiles_to_row_count = self.full_dataset_df["smiles_key"].value_counts().to_dict()

        self._save_similarity_cache(
            cache_path=self.MAX_SIM_CACHE_PATH,
            key_column="smiles_key",
            value_column="max_similarity",
            ordered_keys=smiles_list,
            values=max_sim_to_dataset,
            fingerprint=fingerprint,
        )

        self.logger.info("Max similarities computed successfully.")

        return smiles_list, max_sim_to_dataset, smiles_to_row_count

    def dissimilarity_split(
        self,
        smiles_list: list,
        max_sim_to_dataset: np.ndarray,
        smiles_to_row_count: dict,
        similarity_threshold: float = 0.4,
    ):
        """
        Split based on molecular dissimilarity using precomputed similarities.

        Test set: Most dissimilar molecules (max similarity to any other < threshold)
                up to self.TEST_FRAC of full dataset or until exhausted
        Val set: Next most dissimilar molecules, matching test size
        Train set: Everything else

        Args:
            self: SubstrateSmilesSimilaritySplit instance
            smiles_list: List of SMILES strings
            max_sim_to_dataset: Precomputed max similarity for each molecule
            smiles_to_row_count: Dict mapping SMILES to row count in full dataset
            similarity_threshold: Tanimoto threshold below which molecules are considered dissimilar

        Returns:
            tuple: (unique_smiles_df_splits, full_dataset_df_splits, all_similarities)
        """
        self.logger.info(
            f"Splitting with threshold={similarity_threshold}, max_test_frac={self.TEST_FRAC}"
        )

        n_unique_smiles = len(smiles_list)
        n_full_dataset = len(self.full_dataset_df)
        max_test_rows = int(self.TEST_FRAC * n_full_dataset)

        self.logger.info(f"Max test rows (based on full dataset): {max_test_rows}")

        row_counts = [smiles_to_row_count.get(s, 0) for s in smiles_list]
        # Tiny or inorganic molecules (water, ammonia, ions) are dissimilar to everything;
        # they stay in train and are never test or val candidates.
        ineligible_mask = np.array([is_tiny_or_inorganic(s) for s in smiles_list], dtype=bool)
        self.logger.info(
            f"Ineligible for test/val (<= 3 heavy atoms or no carbon): "
            f"{int(ineligible_mask.sum())} molecules, "
            f"{sum(c for c, bad in zip(row_counts, ineligible_mask) if bad)} rows"
        )

        # Find dissimilar molecules: eligible molecules whose max similarity to any other is below threshold
        dissimilar_mask = ~ineligible_mask & (max_sim_to_dataset < similarity_threshold)
        dissimilar_indices = np.where(dissimilar_mask)[0]
        # Sort dissimilar molecules by their max similarity (most dissimilar first)
        dissimilar_indices = order_by_similarity(
            dissimilar_indices, max_sim_to_dataset, row_counts, smiles_list
        )
        self.logger.info(
            f"Found {len(dissimilar_indices)} dissimilar molecules (max sim < {similarity_threshold})"
        )

        # Greedily add dissimilar molecules to test until we hit max_test_rows
        def row_count_of(idx: int) -> int:
            return smiles_to_row_count.get(smiles_list[idx], 0)

        test_indices, test_row_count = greedy_fill(
            dissimilar_indices, row_count_of, max_test_rows
        )

        self.logger.info(
            f"Assigned {len(test_indices)} dissimilar molecules to test set ({test_row_count} rows)"
        )

        # For val, we want molecules that are dissimilar to the rest
        used_indices = set(test_indices)
        remaining_indices = [
            idx for idx in range(n_unique_smiles) if idx not in used_indices
        ]
        eligible_indices = [idx for idx in remaining_indices if not ineligible_mask[idx]]

        # val is sized from valid_frac, not from the test row count.
        target_val_rows = int(self.VAL_FRAC * n_full_dataset)

        # Sort remaining by their max similarity (most dissimilar first)
        eligible_indices = order_by_similarity(
            eligible_indices, max_sim_to_dataset, row_counts, smiles_list
        )

        # Greedily add molecules to val until we hit target_val_rows
        val_indices, val_row_count = greedy_fill(
            eligible_indices, row_count_of, target_val_rows
        )

        # Everything else goes to train
        val_set = set(val_indices)
        train_indices = list(set(remaining_indices) - val_set)
        train_row_count = n_full_dataset - test_row_count - val_row_count

        self.logger.info(
            f"Test size: {len(test_indices)} SMILES, {test_row_count} rows ({100*test_row_count/n_full_dataset:.1f}%)"
        )
        self.logger.info(
            f"Val size: {len(val_indices)} SMILES, {val_row_count} rows ({100*val_row_count/n_full_dataset:.1f}%)"
        )
        self.logger.info(
            f"Train size: {len(train_indices)} SMILES, {train_row_count} rows ({100*train_row_count/n_full_dataset:.1f}%)"
        )

        # Create split DataFrames
        unique_smiles_df_splits = {
            "train": self.unique_smiles_df.iloc[train_indices].reset_index(drop=True),
            "val": self.unique_smiles_df.iloc[val_indices].reset_index(drop=True),
            "test": self.unique_smiles_df.iloc[test_indices].reset_index(drop=True),
        }

        # Split full dataset
        full_dataset_df_splits = {}
        for split_name, split_unique_smiles_df in unique_smiles_df_splits.items():
            full_dataset_df_splits[split_name] = self.full_dataset_df[
                self.full_dataset_df["smiles_key"].isin(split_unique_smiles_df["smiles_key"])
            ].reset_index(drop=True)

        # Build all_similarities from already computed max_sim_to_dataset
        all_similarities = {
            "test": {smiles_list[idx]: max_sim_to_dataset[idx] for idx in test_indices},
            "val": {smiles_list[idx]: max_sim_to_dataset[idx] for idx in val_indices},
        }

        self.logger.info(
            f"Dissimilarity-based split completed for threshold={similarity_threshold}."
        )

        return unique_smiles_df_splits, full_dataset_df_splits, all_similarities

    def run_splits_across_thresholds(
        self,
    ):
        """
        Run dissimilarity-based splits across multiple thresholds and collect results.
        Computes similarities only once and reuses for all thresholds.

        Args:
            self: SubstrateSmilesSimilaritySplit instance
            n_workers: Number of worker processes for parallel computation

        Returns:
            tuple: (all_unique_smiles_df_splits, all_full_dataset_df_splits, all_similarities)
                Each is a dict keyed by threshold
        """
        # Compute similarities once
        smiles_list, max_sim_to_dataset, smiles_to_row_count = (
            self.compute_max_similarities()
        )

        all_unique_smiles_df_splits = {}
        all_full_dataset_df_splits = {}
        all_similarities = {}

        for threshold in self.THRESHOLDS:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing threshold: {threshold}")
            self.logger.info(f"{'='*50}")

            unique_splits, full_splits, similarities = self.dissimilarity_split(
                smiles_list=smiles_list,
                max_sim_to_dataset=max_sim_to_dataset,
                smiles_to_row_count=smiles_to_row_count,
                similarity_threshold=threshold,
            )

            all_unique_smiles_df_splits[threshold] = unique_splits
            all_full_dataset_df_splits[threshold] = full_splits
            all_similarities[threshold] = similarities

        return all_unique_smiles_df_splits, all_full_dataset_df_splits, all_similarities

    def plot_max_tanimoto_distribution(self, all_similarities):
        values_by_threshold = self._build_value_map_from_similarity_dicts(
            all_similarities
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 101),
            output_path=self.UNIQUE_SMILES_SIMILARITY_PLOT_PATH,
            xlabel="Max Tanimoto Similarity to Train",
            ylabel="Count",
            title_fn=lambda threshold, val_values, test_values: (
                f"Threshold: {threshold}\n"
                f"Val: {len(val_values)}, Test: {len(test_values)}"
            ),
            stats_mode="summary",
            val_label="Val -> Train",
            test_label="Test -> Train",
            threshold_label_fn=lambda threshold: (
                f"Clustering Threshold ({threshold})"
            ),
            legend_loc="upper right",
            hist_range=(0, 1),
        )

    def _plot_full_test_set_tanimoto_distribution(
        self,
        full_dataset_df_splits,
        all_similarities,
        full_dataset_size,
    ):
        values_by_threshold = self._build_value_map_from_split_frames(
            all_splits=full_dataset_df_splits,
            all_similarities=all_similarities,
            key_column="smiles_key",
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 51),
            output_path=self.FULL_DATASET_SMILES_SIMILARITY_PLOT_PATH,
            xlabel="Max Tanimoto Similarity to Train Set",
            ylabel="Number of Test Rows",
            title_fn=lambda threshold, val_values, test_values: (
                f"Full Dataset Tanimoto Similarity\nThreshold {threshold}\n"
                f"Val: {len(val_values)} ({100 * len(val_values) / full_dataset_size:.1f}%), "
                f"Test: {len(test_values)} ({100 * len(test_values) / full_dataset_size:.1f}%)"
            ),
            stats_mode="detailed",
            legend_loc="upper left",
        )

    def generate_splits(self):
        # Run splits across thresholds
        all_unique_splits, all_full_dataset_df_splits, all_similarities = (
            self.run_splits_across_thresholds()
        )

        # Plot unique SMILES distributions
        self.plot_max_tanimoto_distribution(
            all_similarities=all_similarities,
        )

        # Plot full dataset distributions
        self._plot_full_test_set_tanimoto_distribution(
            full_dataset_df_splits=all_full_dataset_df_splits,
            all_similarities=all_similarities,
            full_dataset_size=len(self.full_dataset_df),
        )

        self.logger.info(
            f"Saving full dataset splits for {len(all_full_dataset_df_splits)} thresholds..."
        )
        self.save_threshold_splits(
            all_splits=all_full_dataset_df_splits,
            output_dir=self.SMILES_SPLIT_OUTPUT_DIR,
        )


if __name__ == "__main__":
    # Initialize Hydra config
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")

    # Create splitter instance
    smiles_splitter = SMILESMaxTanimotoSimilaritySplitter(cfg)
    smiles_splitter.generate_splits()
