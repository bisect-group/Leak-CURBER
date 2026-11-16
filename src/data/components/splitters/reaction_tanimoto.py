from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from src.utils.rich_progress import progress, write

from src.data.components.embedders.utils import sha256_short
from src.data.components.splitters.base import (
    configured,
    BaseThresholdedSimilaritySplitter,
    first_present,
    greedy_fill,
    order_by_similarity,
)
from src.utils.rich_progress import component_logger


def _max_drfp_tanimoto(binary: np.ndarray, max_batch_cells: int = 2**25) -> np.ndarray:
    """Max Tanimoto similarity of each binary row to every other row, as float64."""
    # A float32 matrix product per row block is exact for 0/1 data (counts < 2**24).
    # The quotient stays float64 so comparisons with the threshold are exact.
    binary = binary.astype(np.float32, copy=False)
    row_sums = binary.sum(axis=1)
    n_samples = len(binary)
    batch = max(1, max_batch_cells // n_samples)
    max_sims = np.empty(n_samples, dtype=np.float64)
    for start in progress(
        range(0, n_samples, batch),
        desc="Computing max Tanimoto similarities",
        leave=False,
    ):
        stop = min(start + batch, n_samples)
        intersections = binary[start:stop] @ binary.T
        unions = row_sums[start:stop, None] + row_sums[None, :] - intersections
        sims = np.divide(
            intersections,
            unions,
            out=np.zeros(intersections.shape, dtype=np.float64),
            where=unions > 0,
            dtype=np.float64,
        )
        sims[np.arange(stop - start), np.arange(start, stop)] = 0.0
        max_sims[start:stop] = sims.max(axis=1)
    return max_sims


class ReactionDRFPTanimotoSimilaritySplitter(BaseThresholdedSimilaritySplitter):
    """Thresholded splitter for reaction SMILES using DRFP binary fingerprints."""

    EMBEDDING_DTYPE = np.uint8

    def __init__(self, cfg: DictConfig):
        scfg = cfg.splits
        dcfg = scfg.dataset

        def _split_opt(key: str, default=None):
            return dcfg.get(key, scfg.get(key, default))

        self.KEY_COLUMN = str(_split_opt("tanimoto_key_column", "rxn_smiles"))
        self.TEST_FRAC = float(_split_opt("tanimoto_test_frac", configured(scfg.test_frac, 0.1)))
        # val defaults to valid_frac, not to the test row count.
        self.MATCH_VAL_TO_TEST = bool(_split_opt("tanimoto_match_val_to_test", False))
        self.VAL_FRAC = float(_split_opt("tanimoto_val_frac", configured(cfg.splits.valid_frac, 0.1)))
        self.THRESHOLDS = list(
            _split_opt(
                "tanimoto_similarity_thresholds",
                scfg.get("smiles_similarity_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5]),
            )
        )
        self.CACHE_OVERWRITE = bool(scfg.get("similarity_cache_overwrite", False))
        self.EMBEDDING_COLUMN = _split_opt("tanimoto_embedding_col")

        npz_cfg = _split_opt("tanimoto_embeddings_npz", None)
        self.EMBEDDINGS_NPZ = Path(npz_cfg) if npz_cfg else None

        store_cfg = _split_opt("tanimoto_embeddings_store_dir", None)
        self.EMBEDDINGS_STORE_DIR = Path(store_cfg) if store_cfg else None

        self.INPUT_DATASET_PATH = Path(
            first_present(
                dcfg,
                (
                    "tanimoto_split_input_dataset_parquet_file_path",
                    "smiles_split_input_dataset_parquet_file_path",
                    "conformer_split_input_dataset_parquet_file_path",
                ),
            )
        )
        self.OUTPUT_DIR = Path(
            first_present(
                dcfg,
                (
                    "tanimoto_split_output_dir",
                    "smiles_split_output_dir",
                    "conformer_split_output_dir",
                ),
            )
        )
        self.UNIQUE_SIMILARITY_PLOT_PATH = Path(
            first_present(
                dcfg,
                (
                    "unique_tanimoto_similarity_plot_path",
                    "unique_smiles_similarity_plot_path",
                    "unique_conformer_similarity_plot_path",
                ),
            )
        )
        self.FULL_DATASET_SIMILARITY_PLOT_PATH = Path(
            first_present(
                dcfg,
                (
                    "full_dataset_tanimoto_similarity_plot_path",
                    "full_dataset_smiles_similarity_plot_path",
                    "full_dataset_conformer_similarity_plot_path",
                ),
            )
        )

        log_path = Path(scfg.log_dir)
        for path in [
            log_path,
            self.OUTPUT_DIR,
            self.UNIQUE_SIMILARITY_PLOT_PATH.parent,
            self.FULL_DATASET_SIMILARITY_PLOT_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = component_logger(
            log_dir=log_path,
            log_file_name=scfg.log_file_name,
        ).get_logger()

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Loading full dataset...")
        self.full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH).reset_index(
            drop=True
        )
        self.logger.info(f"Full dataset shape: {self.full_dataset_df.shape}")

        if self.KEY_COLUMN not in self.full_dataset_df.columns:
            raise ValueError(
                f"Dataset must contain key column '{self.KEY_COLUMN}' for Tanimoto splitting."
            )

        self.full_dataset_df[self.KEY_COLUMN] = self.full_dataset_df[
            self.KEY_COLUMN
        ].astype("string")
        self.unique_keys = (
            self.full_dataset_df[self.KEY_COLUMN].dropna().astype(str).unique().tolist()
        )

        self._embedding_source, self._embedding_source_path = self._resolve_embedding_source(
            npz_name="drfp_embeddings.npz",
            error_hint=(
                "No Tanimoto embedding source found. Set one of: "
                "splits.dataset.tanimoto_embedding_col, "
                "splits.dataset.tanimoto_embeddings_store_dir, or "
                "splits.dataset.tanimoto_embeddings_npz."
            ),
        )

        self.logger.info(
            f"Unique keys ({self.KEY_COLUMN}): {len(self.unique_keys):,}"
        )
        self.logger.info(
            f"embedding_source={self._embedding_source} ({self._embedding_source_path}), "
            f"thresholds={self.THRESHOLDS}, "
            f"test_frac={self.TEST_FRAC}, "
            f"match_val_to_test={self.MATCH_VAL_TO_TEST}"
        )

        self._max_tanimoto_array: np.ndarray | None = None
        self._max_tanimoto_to_dataset: dict[str, float] | None = None
        self._valid_embedding_mask: np.ndarray | None = None
        self.MAX_SIM_CACHE_PATH = self.OUTPUT_DIR / "max_tanimoto_similarities.tsv"

    def _load_per_key_embeddings_from_column(self) -> dict[str, np.ndarray]:
        direct_embeddings = dict(self._iter_column_vectors())

        self.logger.info(
            f"Loaded {len(direct_embeddings):,} embeddings from dataframe column '{self.EMBEDDING_COLUMN}'."
        )
        return direct_embeddings

    def _load_per_key_embeddings_from_store(self) -> dict[str, np.ndarray]:
        store_dir = self._embedding_source_path
        key_hash_to_key = {sha256_short(key): key for key in self.unique_keys}
        vectors_by_hash = self._read_store_vectors(store_dir, key_hash_to_key)
        vectors_by_key = {
            key_hash_to_key[key_hash]: vec for key_hash, vec in vectors_by_hash.items()
        }
        self.logger.info(
            f"Loaded store embeddings for {len(vectors_by_key):,} dataset keys from {store_dir}."
        )
        return vectors_by_key

    def _load_per_key_embeddings_from_npz(self) -> dict[str, np.ndarray]:
        self.logger.info(f"Loading embeddings from {self._embedding_source_path}...")
        data = np.load(str(self._embedding_source_path), allow_pickle=False)
        direct_embeddings: dict[str, np.ndarray] = {}
        # A set makes each lookup O(1).
        unique_keys = set(self.unique_keys)

        for key in progress(data.files, desc="Reading NPZ keys"):
            if key not in unique_keys:
                continue
            vec = self._vector_from_raw(data[key])
            if vec is None:
                continue
            direct_embeddings[key] = vec

        self.logger.info(
            f"Loaded NPZ embeddings for {len(direct_embeddings):,} dataset keys."
        )
        return direct_embeddings

    def _load_per_key_embeddings(self) -> dict[str, np.ndarray]:
        if self._embedding_source == "column":
            return self._load_per_key_embeddings_from_column()
        if self._embedding_source == "store":
            return self._load_per_key_embeddings_from_store()
        return self._load_per_key_embeddings_from_npz()

    def compute_max_tanimoto_to_dataset(self) -> np.ndarray:
        if self._max_tanimoto_array is not None:
            return self._max_tanimoto_array

        if not self.CACHE_OVERWRITE:
            loaded = self._load_similarity_cache(
                cache_path=self.MAX_SIM_CACHE_PATH,
                key_column=self.KEY_COLUMN,
                value_column="max_tanimoto_similarity",
                ordered_keys=self.unique_keys,
                fingerprint=self._similarity_fingerprint(),
            )
            if loaded is not None:
                values, _ = loaded
                self._max_tanimoto_array = values
                self._max_tanimoto_to_dataset = {
                    key: float(values[i]) for i, key in enumerate(self.unique_keys)
                }
                self._valid_embedding_mask = self._embedding_validity_mask(
                    self._load_per_key_embeddings()
                )
                self.logger.info(
                    f"Using cached Tanimoto similarities from {self.MAX_SIM_CACHE_PATH}"
                )
                return self._max_tanimoto_array

        per_key_embs = self._load_per_key_embeddings()
        n_unique = len(self.unique_keys)
        available = [key for key in self.unique_keys if key in per_key_embs]
        self.logger.info(
            f"Embeddings available for {len(available):,}/{n_unique:,} unique keys."
        )

        if not per_key_embs:
            raise ValueError(
                "No embeddings available for Tanimoto splitting. "
                f"Source={self._embedding_source} at {self._embedding_source_path}."
            )

        emb_dim = next(iter(per_key_embs.values())).shape[0]
        emb_matrix = np.zeros((n_unique, emb_dim), dtype=np.uint8)
        valid_mask = np.zeros(n_unique, dtype=bool)
        key_to_idx = {key: i for i, key in enumerate(self.unique_keys)}

        for key, emb in per_key_embs.items():
            if key not in key_to_idx:
                continue
            idx = key_to_idx[key]
            emb_matrix[idx] = emb
            valid_mask[idx] = True

        valid_indices = np.where(valid_mask)[0]
        valid_embs = emb_matrix[valid_indices]
        self._valid_embedding_mask = self._embedding_validity_mask(per_key_embs)

        if len(valid_indices) == 0:
            raise ValueError(
                "No dataset keys were found in embedding source; cannot compute Tanimoto similarities."
            )

        self.logger.info(
            f"[Tanimoto] Computing pairwise max for {len(valid_indices):,} keys "
            f"(D={emb_dim})..."
        )

        if len(valid_indices) == 1:
            max_sims_valid = np.array([0.0], dtype=np.float64)
        else:
            max_sims_valid = _max_drfp_tanimoto(valid_embs > 0).astype(np.float64)

        max_sims = np.zeros(n_unique, dtype=np.float64)
        max_sims[valid_indices] = max_sims_valid

        self._max_tanimoto_array = max_sims
        self._max_tanimoto_to_dataset = {
            key: float(max_sims[i]) for i, key in enumerate(self.unique_keys)
        }

        self._save_similarity_cache(
            cache_path=self.MAX_SIM_CACHE_PATH,
            key_column=self.KEY_COLUMN,
            value_column="max_tanimoto_similarity",
            ordered_keys=self.unique_keys,
            values=max_sims,
            fingerprint=self._similarity_fingerprint(),
        )

        valid_sims = max_sims[valid_indices]
        self.logger.info("Max Tanimoto similarity statistics (over keys with embeddings):")
        self.logger.info(f"  Min:    {valid_sims.min():.4f}")
        self.logger.info(f"  Max:    {valid_sims.max():.4f}")
        self.logger.info(f"  Mean:   {valid_sims.mean():.4f}")
        self.logger.info(f"  Median: {np.median(valid_sims):.4f}")
        self.logger.info(
            f"  Missing embeddings (forced to train): {int((~valid_mask).sum()):,}"
        )

        return max_sims

    def get_max_tanimoto_for_keys(self, keys: list[str]) -> dict[str, float]:
        if self._max_tanimoto_to_dataset is None:
            self.compute_max_tanimoto_to_dataset()
        return {key: self._max_tanimoto_to_dataset.get(key, 0.0) for key in keys}

    def dissimilarity_based_split(
        self,
        similarity_threshold: float,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
        self.logger.info(
            f"Splitting with Tanimoto threshold={similarity_threshold}, "
            f"max_test_frac={self.TEST_FRAC}"
        )

        max_tanimoto_to_dataset = self.compute_max_tanimoto_to_dataset()

        unique_keys = self.unique_keys
        n_unique = len(unique_keys)
        n_full = len(self.full_dataset_df)
        max_test_rows = int(self.TEST_FRAC * n_full)
        row_count_dict = self.full_dataset_df[self.KEY_COLUMN].value_counts().to_dict()

        # The mask comes from the embeddings (also on warm cache), not from max_sim == 0.
        # All-zero and missing embeddings are always train.
        suspicious_mask = ~self._valid_embedding_mask
        dissimilar_mask = (~suspicious_mask) & (
            max_tanimoto_to_dataset < similarity_threshold
        )
        dissimilar_indices = np.where(dissimilar_mask)[0]
        row_counts = [row_count_dict.get(k, 0) for k in unique_keys]
        dissimilar_indices = order_by_similarity(
            dissimilar_indices, max_tanimoto_to_dataset, row_counts, unique_keys
        )
        self.logger.info(
            f"  {len(dissimilar_indices):,} dissimilar keys (max sim < {similarity_threshold})"
        )

        def row_count_of(idx: int) -> int:
            return row_count_dict.get(unique_keys[idx], 0)

        test_indices, test_row_count = greedy_fill(
            dissimilar_indices, row_count_of, max_test_rows, desc="Filling test set"
        )

        used_indices = set(test_indices)
        remaining_indices = [
            idx
            for idx in range(n_unique)
            if idx not in used_indices and not suspicious_mask[idx]
        ]
        remaining_indices = order_by_similarity(
            remaining_indices, max_tanimoto_to_dataset, row_counts, unique_keys
        )

        target_val_rows = (
            test_row_count if self.MATCH_VAL_TO_TEST else int(self.VAL_FRAC * n_full)
        )
        val_indices, val_row_count = greedy_fill(
            remaining_indices, row_count_of, target_val_rows, desc="Filling val set"
        )

        # Suspicious keys are intentionally excluded from held-out candidates,
        # but they must still be retained in training.  Building training from
        # ``remaining_indices`` would silently drop every suspicious reaction.
        test_set = set(test_indices)
        val_set = set(val_indices)
        train_indices = [
            idx
            for idx in range(n_unique)
            if idx not in test_set and idx not in val_set
        ]
        train_row_count = n_full - test_row_count - val_row_count

        self.logger.info(
            f"  Test set: {len(test_indices):,} keys, {test_row_count:,} rows"
        )
        self.logger.info(
            f"  Val set: {len(val_indices):,} keys, {val_row_count:,} rows"
        )
        self.logger.info(
            f"  Train set: {len(train_indices):,} keys, ~{train_row_count:,} rows"
        )

        splits = {
            "train": self.full_dataset_df[
                self.full_dataset_df[self.KEY_COLUMN].isin([unique_keys[i] for i in train_indices])
            ].reset_index(drop=True),
            "val": self.full_dataset_df[
                self.full_dataset_df[self.KEY_COLUMN].isin([unique_keys[i] for i in val_indices])
            ].reset_index(drop=True),
            "test": self.full_dataset_df[
                self.full_dataset_df[self.KEY_COLUMN].isin([unique_keys[i] for i in test_indices])
            ].reset_index(drop=True),
        }

        test_keys = [unique_keys[i] for i in test_indices]
        val_keys = [unique_keys[i] for i in val_indices]
        train_keys = [unique_keys[i] for i in train_indices]

        if (set(test_keys) & set(val_keys)) or (set(test_keys) & set(train_keys)) or (
            set(val_keys) & set(train_keys)
        ):
            raise RuntimeError("Leakage detected in Tanimoto split.")

        assigned_indices = test_set | val_set | set(train_indices)
        if len(assigned_indices) != n_unique:
            missing_indices = sorted(set(range(n_unique)) - assigned_indices)
            raise RuntimeError(
                "Incomplete Tanimoto split: "
                f"assigned {len(assigned_indices):,} of {n_unique:,} keys; "
                f"examples missing={[unique_keys[i] for i in missing_indices[:5]]}"
            )

        assigned_rows = sum(len(split_df) for split_df in splits.values())
        if assigned_rows != n_full:
            raise RuntimeError(
                "Incomplete Tanimoto split rows: "
                f"assigned {assigned_rows:,} of {n_full:,} rows."
            )

        similarities = {
            "test": self.get_max_tanimoto_for_keys(test_keys),
            "val": self.get_max_tanimoto_for_keys(val_keys),
        }

        return splits, similarities

    def run_splits_across_thresholds(self) -> tuple[dict, dict]:
        self.logger.info("Computing max Tanimoto similarities to full dataset (once)...")
        self.compute_max_tanimoto_to_dataset()

        all_splits = {}
        all_similarities = {}
        for threshold in progress(
            self.THRESHOLDS,
            desc="Processing thresholds",
            unit="threshold",
        ):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Tanimoto threshold: {threshold}")
            self.logger.info(f"{'=' * 50}")

            splits, similarities = self.dissimilarity_based_split(
                similarity_threshold=threshold
            )
            all_splits[threshold] = splits
            all_similarities[threshold] = similarities

        return all_splits, all_similarities

    def plot_unique_distribution(self, all_similarities: dict) -> None:
        values_by_threshold = self._build_value_map_from_similarity_dicts(
            all_similarities
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 101),
            output_path=self.UNIQUE_SIMILARITY_PLOT_PATH,
            xlabel="Max Tanimoto Similarity to Dataset",
            ylabel="Count",
            title_fn=lambda threshold, val_values, test_values: (
                f"Threshold: {threshold}\n"
                f"Val: {len(val_values)}, Test: {len(test_values)}"
            ),
            stats_mode="summary",
            val_label="Val -> Train",
            test_label="Test -> Train",
            threshold_label_fn=lambda threshold: f"Clustering Threshold ({threshold})",
            legend_loc="upper right",
            hist_range=(0, 1),
        )

    def plot_full_distribution(self, all_splits: dict, all_similarities: dict) -> None:
        full_dataset_size = len(self.full_dataset_df)
        values_by_threshold = self._build_value_map_from_split_frames(
            all_splits=all_splits,
            all_similarities=all_similarities,
            key_column=self.KEY_COLUMN,
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 51),
            output_path=self.FULL_DATASET_SIMILARITY_PLOT_PATH,
            xlabel="Max Tanimoto Similarity to Train Set",
            ylabel="Number of Rows",
            title_fn=lambda threshold, val_values, test_values: (
                f"Full Dataset Tanimoto Similarity\nThreshold {threshold}\n"
                f"Val: {len(val_values)} ({100 * len(val_values) / full_dataset_size:.1f}%), "
                f"Test: {len(test_values)} ({100 * len(test_values) / full_dataset_size:.1f}%)"
            ),
            stats_mode="detailed",
            legend_loc="upper left",
        )

    def get_output_dir(self) -> Path:
        return self.OUTPUT_DIR


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")

    splitter = ReactionDRFPTanimotoSimilaritySplitter(cfg=cfg)
    all_splits, all_similarities = splitter.generate_splits()
