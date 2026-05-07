import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from src.utils.tqdmlogger import TqdmLogger
from src.data.components.splitters.base import BaseSplitter

class GroupShuffleUniqueColumnSplitter(BaseSplitter):
    """Leakage-safe grouped splitter using one scalar group value per row."""

    def __init__(self, cfg: DictConfig):
        self.RANDOM_SEED = cfg.splits.get("group_shuffle_random_seed", cfg.splits.random_seed)

        self.TRAIN_FRAC = cfg.splits.train_frac or 0.8
        self.VALID_FRAC = cfg.splits.valid_frac or 0.1
        self.TEST_FRAC = cfg.splits.test_frac or 0.1
        total_frac = self.TRAIN_FRAC + self.VALID_FRAC + self.TEST_FRAC
        if abs(total_frac - 1.0) > 1e-8:
            raise ValueError(
                "train_frac + valid_frac + test_frac must sum to 1.0 for GroupShuffleUniqueColumnSplitter."
            )

        self.UNIQUE_COL = cfg.splits.get("group_shuffle_unique_col")
        if not self.UNIQUE_COL:
            raise ValueError(
                "Missing splits.group_shuffle_unique_col. Provide the unique column to group by."
            )

        self.UNIQUE_COL_DELIMITER = cfg.splits.get("group_shuffle_unique_col_delimiter")

        self.INPUT_DATASET_PATH = Path(
            cfg.splits.dataset.get(
                "group_shuffle_input_dataset_parquet_file_path",
                cfg.splits.dataset.sequence_split.input_dataset_parquet_file_path,
            )
        )

        random_output_dir = Path(cfg.splits.dataset.random_output_dir)
        default_output_dir = random_output_dir.parent / cfg.splits.get(
            "group_shuffle_output_dir_name", "group_shuffle_splits"
        )
        self.OUTPUT_DIR = Path(
            cfg.splits.dataset.get("group_shuffle_output_dir", str(default_output_dir))
        )

        log_path = Path(cfg.splits.log_dir)
        for path in [log_path, self.OUTPUT_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=log_path, log_file_name=cfg.splits.log_file_name
        ).get_logger()

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet file not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

    def _normalize_group_value(self, value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "__MISSING__"

        if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series, dict)):
            raise ValueError(
                f"Column '{self.UNIQUE_COL}' must contain one scalar value per row. "
                "Found list-like/dict value."
            )

        value_str = str(value).strip()
        return value_str if value_str else "__MISSING__"

    def _group_shuffle_split_indices(
        self, groups: np.ndarray, test_size: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(groups))

        try:
            from sklearn.model_selection import GroupShuffleSplit
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for GroupShuffleUniqueColumnSplitter. "
                "Install it with `pip install scikit-learn`."
            ) from exc

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(indices, groups=groups))

        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError("Grouped split failed to create non-empty train/test partitions.")

        return train_idx, test_idx

    def _compute_group_set(self, groups: np.ndarray, idx: np.ndarray) -> set[str]:
        return set(groups[idx].tolist())

    def generate_splits(self):
        self.logger.info("Loading full dataset...")
        full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH).reset_index(drop=True)
        self.logger.info(f"Full dataset shape: {full_dataset_df.shape}")

        if self.UNIQUE_COL not in full_dataset_df.columns:
            raise KeyError(
                f"Column '{self.UNIQUE_COL}' not found in dataset. Available columns: {list(full_dataset_df.columns)}"
            )

        groups = np.array(
            [
                self._normalize_group_value(value)
                for value in full_dataset_df[self.UNIQUE_COL].tolist()
            ],
            dtype=object,
        )

        n_groups = len(np.unique(groups))
        self.logger.info(f"Found {n_groups} unique groups from column '{self.UNIQUE_COL}'.")
        if n_groups < 3:
            raise ValueError(
                "Need at least 3 unique groups to create train/val/test without leakage."
            )

        train_val_idx, test_idx = self._group_shuffle_split_indices(
            groups=groups,
            test_size=self.TEST_FRAC,
            seed=self.RANDOM_SEED,
        )

        remaining_frac = self.TRAIN_FRAC + self.VALID_FRAC
        rel_val_frac = self.VALID_FRAC / remaining_frac

        train_val_groups = groups[train_val_idx]
        train_rel_idx, val_rel_idx = self._group_shuffle_split_indices(
            groups=train_val_groups,
            test_size=rel_val_frac,
            seed=self.RANDOM_SEED + 1,
        )

        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        splits = {
            "train": full_dataset_df.iloc[train_idx].reset_index(drop=True),
            "val": full_dataset_df.iloc[val_idx].reset_index(drop=True),
            "test": full_dataset_df.iloc[test_idx].reset_index(drop=True),
        }

        train_groups = self._compute_group_set(groups, train_idx)
        val_groups = self._compute_group_set(groups, val_idx)
        test_groups = self._compute_group_set(groups, test_idx)

        train_val_overlap = len(train_groups & val_groups)
        train_test_overlap = len(train_groups & test_groups)
        val_test_overlap = len(val_groups & test_groups)

        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise RuntimeError(
                "Leakage detected after grouped split: "
                f"train/val={train_val_overlap}, train/test={train_test_overlap}, val/test={val_test_overlap}"
            )

        total_rows = len(full_dataset_df)
        self.logger.info(
            f"Group-shuffle full dataset splits (unique_col={self.UNIQUE_COL}): "
            f"Train: {splits['train'].shape}, ({100 * splits['train'].shape[0] / total_rows:.2f}%)  "
            f"Val: {splits['val'].shape}, ({100 * splits['val'].shape[0] / total_rows:.2f}%)  "
            f"Test: {splits['test'].shape}, ({100 * splits['test'].shape[0] / total_rows:.2f}%)"
        )
        self.logger.info(
            "Verified zero shared unique-column values across train/val/test."
        )

        self.save_named_splits(splits=splits, output_dir=self.OUTPUT_DIR)
        return splits
