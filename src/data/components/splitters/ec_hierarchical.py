import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.components.splitters.base import BaseSplitter
from src.utils.tqdmlogger import TqdmLogger


class ECHierarchicalGroupSplitter(BaseSplitter):
    """Leakage-safe grouped split by configurable EC hierarchy levels."""

    def __init__(self, cfg: DictConfig):
        ec_cfg = cfg.splits.dataset.get("ec_split")

        self.RANDOM_SEED = (
            ec_cfg.get("random_seed")
            if ec_cfg and ec_cfg.get("random_seed") is not None
            else cfg.splits.get("ec_split_random_seed", cfg.splits.random_seed)
        )

        self.TRAIN_FRAC = cfg.splits.train_frac or 0.8
        self.VALID_FRAC = cfg.splits.valid_frac or 0.1
        self.TEST_FRAC = cfg.splits.test_frac or 0.1
        total_frac = self.TRAIN_FRAC + self.VALID_FRAC + self.TEST_FRAC
        if abs(total_frac - 1.0) > 1e-8:
            raise ValueError(
                "train_frac + valid_frac + test_frac must sum to 1.0 for ECHierarchicalGroupSplitter."
            )

        self.EC_COLUMN = (
            ec_cfg.get("column")
            if ec_cfg and ec_cfg.get("column")
            else cfg.splits.get("ec_split_column")
        )
        if not self.EC_COLUMN:
            raise ValueError("Missing splits.ec_split_column in config.")

        levels = (
            ec_cfg.get("levels")
            if ec_cfg and ec_cfg.get("levels")
            else cfg.splits.get("ec_split_levels", [1, 2, 3, 4])
        )
        if not levels:
            raise ValueError("splits.ec_split_levels cannot be empty.")
        self.EC_LEVELS = sorted({int(level) for level in levels})
        if any(level < 1 or level > 4 for level in self.EC_LEVELS):
            raise ValueError("splits.ec_split_levels must only contain integers from 1 to 4.")

        self.EC_PART_DELIMITER = (
            ec_cfg.get("part_delimiter")
            if ec_cfg and ec_cfg.get("part_delimiter") is not None
            else cfg.splits.get("ec_split_part_delimiter", ".")
        )
        self.EC_MULTIVALUE_DELIMITER = (
            ec_cfg.get("multivalue_delimiter")
            if ec_cfg and ec_cfg.get("multivalue_delimiter") is not None
            else cfg.splits.get("ec_split_multivalue_delimiter")
        )
        self.EC_TEST_FRAC = float(
            ec_cfg.get("test_frac")
            if ec_cfg and ec_cfg.get("test_frac") is not None
            else cfg.splits.get("ec_split_test_frac", self.TEST_FRAC)
        )
        self.EC_MATCH_VAL_TO_TEST = bool(
            ec_cfg.get("match_val_to_test")
            if ec_cfg and ec_cfg.get("match_val_to_test") is not None
            else cfg.splits.get("ec_split_match_val_to_test", True)
        )

        if self.EC_TEST_FRAC <= 0.0 or self.EC_TEST_FRAC >= 0.5:
            raise ValueError("splits.ec_split_test_frac must be in (0, 0.5).")

        default_input_path = cfg.splits.dataset.get("ec_split_input_dataset_parquet_file_path")
        if not default_input_path:
            sequence_cfg = cfg.splits.dataset.get("sequence_split")
            if sequence_cfg:
                default_input_path = sequence_cfg.get("input_dataset_parquet_file_path")
        input_path = (
            ec_cfg.get("input_dataset_parquet_file_path")
            if ec_cfg and ec_cfg.get("input_dataset_parquet_file_path")
            else default_input_path
        )
        if not input_path:
            raise ValueError(
                "Missing EC split input dataset path. "
                "Set splits.dataset.ec_split.input_dataset_parquet_file_path or "
                "provide a dataset sequence_split block."
            )
        self.INPUT_DATASET_PATH = Path(input_path)

        random_output_dir = cfg.splits.dataset.get("random_output_dir")
        default_output_dir = None
        if random_output_dir:
            default_output_dir = str(
                Path(random_output_dir).parent
                / cfg.splits.get("ec_split_output_dir_name", "ec_hierarchy_splits")
            )
        output_dir = (
            ec_cfg.get("output_dir")
            if ec_cfg and ec_cfg.get("output_dir")
            else cfg.splits.dataset.get("ec_split_output_dir", default_output_dir)
        )
        if not output_dir:
            raise ValueError(
                "Missing EC split output directory. "
                "Set splits.dataset.ec_split.output_dir or splits.dataset.random_output_dir."
            )
        self.OUTPUT_DIR = Path(output_dir)

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

    def _normalize_scalar(self, value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""

        if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series, dict)):
            raise ValueError(
                f"Column '{self.EC_COLUMN}' must contain scalar values. "
                "Use splits.ec_split_multivalue_delimiter for string-encoded multi-EC rows."
            )

        value_str = str(value).strip()
        return value_str

    def _extract_primary_ec(self, value: str) -> str:
        if not value:
            return ""
        if self.EC_MULTIVALUE_DELIMITER and self.EC_MULTIVALUE_DELIMITER in value:
            parts = [part.strip() for part in value.split(self.EC_MULTIVALUE_DELIMITER)]
            parts = [part for part in parts if part]
            return parts[0] if parts else ""
        return value

    def _ec_prefix(self, ec: str, level: int) -> str:
        ec = ec.strip()
        if not ec:
            return "__MISSING__"

        parts = [part.strip() for part in ec.split(self.EC_PART_DELIMITER)]
        parts = [part if part else "-" for part in parts]
        if len(parts) < level:
            parts.extend(["-"] * (level - len(parts)))
        return self.EC_PART_DELIMITER.join(parts[:level])

    def _allocate_group_splits_by_rows(
        self,
        *,
        group_row_counts: list[tuple[str, int]],
        target_test_rows: int,
        target_val_rows: int,
    ) -> tuple[set[str], set[str], set[str]]:
        n_groups = len(group_row_counts)
        if n_groups < 3:
            raise ValueError("Need at least 3 groups to allocate train/val/test.")

        # Keep at least 2 groups for val/train while filling test.
        test_groups: list[str] = []
        test_rows = 0
        cursor = 0
        while cursor < n_groups - 2:
            group_name, row_count = group_row_counts[cursor]
            if test_rows + row_count <= target_test_rows:
                test_groups.append(group_name)
                test_rows += row_count
                cursor += 1
            else:
                break

        # Guarantee non-empty test split.
        if not test_groups:
            group_name, row_count = group_row_counts[cursor]
            test_groups.append(group_name)
            test_rows += row_count
            cursor += 1

        # Fill val from the next larger groups until reaching (or slightly exceeding) target.
        # Keep at least 1 group for train.
        val_groups: list[str] = []
        val_rows = 0
        while cursor < n_groups - 1 and val_rows < target_val_rows:
            group_name, row_count = group_row_counts[cursor]
            val_groups.append(group_name)
            val_rows += row_count
            cursor += 1

        # Guarantee non-empty val split while preserving at least one train group.
        if not val_groups and cursor < n_groups - 1:
            group_name, _ = group_row_counts[cursor]
            val_groups.append(group_name)
            cursor += 1

        train_groups = [group_name for group_name, _ in group_row_counts[cursor:]]
        if not train_groups:
            raise RuntimeError(
                "Greedy EC split allocation produced empty train split. "
                "Adjust ec_split_test_frac or dataset grouping distribution."
            )

        return set(train_groups), set(val_groups), set(test_groups)

    def _split_for_level(self, full_dataset_df: pd.DataFrame, level: int) -> dict[str, pd.DataFrame]:
        ec_groups = np.array(
            [
                self._ec_prefix(self._extract_primary_ec(self._normalize_scalar(value)), level)
                for value in full_dataset_df[self.EC_COLUMN].tolist()
            ],
            dtype=object,
        )

        n_groups = len(np.unique(ec_groups))
        self.logger.info(f"EC level L{level}: found {n_groups} unique EC groups.")
        if n_groups < 3:
            raise ValueError(
                f"Need at least 3 unique groups for L{level} to create train/val/test without leakage."
            )

        group_counts_series = pd.Series(ec_groups).value_counts()
        group_row_counts = sorted(
            [(str(group_name), int(row_count)) for group_name, row_count in group_counts_series.items()],
            key=lambda item: (item[1], item[0]),
        )

        target_test_rows = int(round(self.EC_TEST_FRAC * len(full_dataset_df)))
        target_val_rows = (
            target_test_rows
            if self.EC_MATCH_VAL_TO_TEST
            else int(round(self.VALID_FRAC * len(full_dataset_df)))
        )
        train_group_set, val_group_set, test_group_set = self._allocate_group_splits_by_rows(
            group_row_counts=group_row_counts,
            target_test_rows=target_test_rows,
            target_val_rows=target_val_rows,
        )

        train_mask = np.isin(ec_groups, list(train_group_set))
        val_mask = np.isin(ec_groups, list(val_group_set))
        test_mask = np.isin(ec_groups, list(test_group_set))

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        train_groups = set(ec_groups[train_idx].tolist())
        val_groups = set(ec_groups[val_idx].tolist())
        test_groups = set(ec_groups[test_idx].tolist())
        if (train_groups & val_groups) or (train_groups & test_groups) or (val_groups & test_groups):
            raise RuntimeError(
                f"Leakage detected for EC level L{level}: "
                f"train/val={len(train_groups & val_groups)}, "
                f"train/test={len(train_groups & test_groups)}, "
                f"val/test={len(val_groups & test_groups)}"
            )

        self.logger.info(
            f"EC L{level} row-targets: test_target={target_test_rows}, val_target={target_val_rows}; "
            f"actual test={len(test_idx)}, val={len(val_idx)}"
        )

        return {
            "train": full_dataset_df.iloc[train_idx].reset_index(drop=True),
            "val": full_dataset_df.iloc[val_idx].reset_index(drop=True),
            "test": full_dataset_df.iloc[test_idx].reset_index(drop=True),
        }

    def generate_splits(self) -> dict[int, dict[str, pd.DataFrame]]:
        self.logger.info("Loading full dataset for EC hierarchical split...")
        full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH).reset_index(drop=True)
        self.logger.info(f"Full dataset shape: {full_dataset_df.shape}")

        if self.EC_COLUMN not in full_dataset_df.columns:
            raise KeyError(
                f"Column '{self.EC_COLUMN}' not found in dataset. "
                f"Available columns: {list(full_dataset_df.columns)}"
            )

        all_level_splits: dict[int, dict[str, pd.DataFrame]] = {}
        aggregated_named_splits: dict[str, pd.DataFrame] = {}
        total_rows = len(full_dataset_df)
        for level in self.EC_LEVELS:
            splits = self._split_for_level(full_dataset_df=full_dataset_df, level=level)
            all_level_splits[level] = splits
            self.logger.info(
                f"EC L{level} splits: "
                f"Train: {splits['train'].shape}, ({100 * splits['train'].shape[0] / total_rows:.2f}%)  "
                f"Val: {splits['val'].shape}, ({100 * splits['val'].shape[0] / total_rows:.2f}%)  "
                f"Test: {splits['test'].shape}, ({100 * splits['test'].shape[0] / total_rows:.2f}%)"
            )

            for split_name, split_df in splits.items():
                aggregated_named_splits[f"L{level}/{split_name}"] = split_df

        self.save_named_splits(
            splits=aggregated_named_splits, output_dir=self.OUTPUT_DIR
        )

        self.logger.info(
            f"Finished EC hierarchical splits for levels {self.EC_LEVELS} at {self.OUTPUT_DIR}"
        )
        return all_level_splits
