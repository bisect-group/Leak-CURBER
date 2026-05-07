import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import re
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, ListConfig
from src.data.components.splitters.base import BaseSplitter
from src.utils.tqdmlogger import TqdmLogger


class RandomSplitter(BaseSplitter):
    """Splits a dataset randomly into training, validation, and test sets.

    This class provides methods to perform random splits of a dataset based on
    specified proportions for training, validation, and test sets.
    """

    def __init__(self, cfg: DictConfig):
        self.RANDOM_SEED = cfg.splits.random_seed or 42

        self.TRAIN_FRAC = cfg.splits.train_frac or 0.8
        self.VALID_FRAC = cfg.splits.valid_frac or 0.1
        self.TEST_FRAC = cfg.splits.test_frac or 0.1
        total_frac = self.TRAIN_FRAC + self.VALID_FRAC + self.TEST_FRAC
        if abs(total_frac - 1.0) > 1e-8:
            raise ValueError(
                "train_frac + valid_frac + test_frac must sum to 1.0 for RandomSplitter."
            )

        self.UNIQUE_COLS = self._resolve_unique_cols(cfg)
        self.INPUT_DATASET_PATH = self._resolve_input_dataset_path(cfg)
        self.OUTPUT_DIR = self._resolve_output_dir(cfg)

        LOG_PATH = Path(cfg.splits.log_dir)
        for path in [LOG_PATH, self.OUTPUT_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH, log_file_name=cfg.splits.log_file_name
        ).get_logger()

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet file not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

    def _get_random_cfg(self, cfg: DictConfig):
        return cfg.splits.dataset.get("random")

    def _coerce_unique_cols(self, value) -> list[str]:
        if isinstance(value, (list, tuple, ListConfig)):
            columns = [str(column) for column in value]
        else:
            raise TypeError(
                "Random split unique_cols must be configured as a list of strings."
            )

        columns = [column.strip() for column in columns if str(column).strip()]
        if not columns:
            raise ValueError("At least one random split unique column must be configured.")

        duplicate_columns = sorted(
            {column for column in columns if columns.count(column) > 1}
        )
        if duplicate_columns:
            raise ValueError(
                "Duplicate random split unique columns configured: "
                f"{duplicate_columns}"
            )

        return columns

    def _resolve_unique_cols(self, cfg: DictConfig) -> list[str]:
        random_cfg = self._get_random_cfg(cfg)
        if not random_cfg or not random_cfg.get("unique_cols"):
            raise ValueError(
                "Missing random split unique columns. Configure "
                "splits.dataset.random.unique_cols as a list, e.g. ['sequence']."
            )

        return self._coerce_unique_cols(random_cfg.unique_cols)

    def _resolve_input_dataset_path(self, cfg: DictConfig) -> Path:
        dcfg = cfg.splits.dataset
        random_cfg = self._get_random_cfg(cfg)

        if random_cfg and random_cfg.get("input_dataset_parquet_file_path"):
            return Path(random_cfg.input_dataset_parquet_file_path)

        for block_name in ("sequence_split", "ec_split", "smiles_split", "structure_split"):
            block = dcfg.get(block_name)
            if block and block.get("input_dataset_parquet_file_path"):
                return Path(block.input_dataset_parquet_file_path)

        for key in (
            "ec_split_input_dataset_parquet_file_path",
            "tanimoto_split_input_dataset_parquet_file_path",
            "smiles_split_input_dataset_parquet_file_path",
            "conformer_split_input_dataset_parquet_file_path",
        ):
            value = dcfg.get(key)
            if value:
                return Path(value)

        raise ValueError(
            "Missing random split input dataset path. Configure one of "
            "splits.dataset.random.input_dataset_parquet_file_path, "
            "splits.dataset.<split>.input_dataset_parquet_file_path, or one of the "
            "legacy flat split input dataset path keys."
        )

    def _resolve_output_dir(self, cfg: DictConfig) -> Path:
        random_cfg = self._get_random_cfg(cfg)
        if random_cfg and random_cfg.get("output_dir"):
            return Path(random_cfg.output_dir)

        output_dir = cfg.splits.dataset.get("random_output_dir")
        if output_dir:
            return Path(output_dir)

        raise ValueError(
            "Missing random split output directory. Configure "
            "splits.dataset.random.output_dir or splits.dataset.random_output_dir."
        )

    def _normalize_group_value(self, value, unique_col: str) -> str:
        if value is None:
            return "__MISSING__"

        if np.isscalar(value) and pd.isna(value):
            return "__MISSING__"

        if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Series, dict)):
            raise ValueError(
                f"Column '{unique_col}' must contain one scalar value per row. "
                "Found list-like or dict value."
            )

        value_str = str(value).strip()
        return value_str if value_str else "__MISSING__"

    def _grouped_output_dir(self, unique_col: str) -> Path:
        safe_column = re.sub(r"[^A-Za-z0-9_.-]+", "_", unique_col).strip("._-")
        safe_column = safe_column or "column"
        return self.OUTPUT_DIR.parent / f"random_splits_grouped_{safe_column}"

    def _assignment_cost(
        self,
        current_rows: dict[str, int],
        target_rows: dict[str, float],
        split_name: str,
        group_size: int,
    ) -> float:
        cost = 0.0
        for name, target in target_rows.items():
            rows = current_rows[name] + (group_size if name == split_name else 0)
            diff = rows - target
            cost += diff * diff
        return cost

    def _assign_groups_to_splits(
        self, group_sizes: pd.Series
    ) -> tuple[dict[str, set[str]], dict[str, int], dict[str, float]]:
        randomized_groups = group_sizes.sample(frac=1.0, random_state=self.RANDOM_SEED)
        ordered_groups = list(
            randomized_groups.sort_values(ascending=False, kind="stable").items()
        )

        rng = np.random.default_rng(self.RANDOM_SEED)
        target_rows = {
            "train": self.TRAIN_FRAC * self.TOTAL_ROWS,
            "val": self.VALID_FRAC * self.TOTAL_ROWS,
            "test": self.TEST_FRAC * self.TOTAL_ROWS,
        }
        assignments = {split_name: set() for split_name in target_rows}
        current_rows = {split_name: 0 for split_name in target_rows}
        split_names = list(target_rows)

        for index, (group_value, group_size) in enumerate(ordered_groups):
            remaining_groups = len(ordered_groups) - index
            empty_splits = [
                split_name for split_name in split_names if not assignments[split_name]
            ]

            if empty_splits and remaining_groups == len(empty_splits):
                candidate_splits = list(empty_splits)
            else:
                candidate_splits = list(split_names)

            rng.shuffle(candidate_splits)
            best_split = min(
                candidate_splits,
                key=lambda split_name: self._assignment_cost(
                    current_rows=current_rows,
                    target_rows=target_rows,
                    split_name=split_name,
                    group_size=int(group_size),
                ),
            )
            assignments[best_split].add(group_value)
            current_rows[best_split] += int(group_size)

        return assignments, current_rows, target_rows

    def _generate_splits_for_column(
        self, full_dataset_df: pd.DataFrame, unique_col: str
    ) -> dict[str, pd.DataFrame]:
        if unique_col not in full_dataset_df.columns:
            raise KeyError(
                f"Column '{unique_col}' not found in dataset. Available columns: {list(full_dataset_df.columns)}"
            )

        group_values = np.array(
            [
                self._normalize_group_value(value, unique_col)
                for value in full_dataset_df[unique_col].tolist()
            ],
            dtype=object,
        )
        group_sizes = pd.Series(group_values, name=unique_col).value_counts(
            sort=False
        )
        n_groups = len(group_sizes)
        if n_groups < 3:
            raise ValueError(
                "Need at least 3 unique groups to create train/val/test without leakage."
            )

        self.TOTAL_ROWS = len(full_dataset_df)
        self.logger.info(
            f"Starting grouped random split using unique_col='{unique_col}' "
            f"across {n_groups} unique values."
        )
        assignments, assigned_rows, target_rows = self._assign_groups_to_splits(
            group_sizes=group_sizes
        )

        all_assigned_groups = set().union(*assignments.values())
        if len(all_assigned_groups) != n_groups:
            raise RuntimeError(
                "Grouped random split did not assign every unique-column value exactly once."
            )

        if (
            assignments["train"] & assignments["val"]
            or assignments["train"] & assignments["test"]
            or assignments["val"] & assignments["test"]
        ):
            raise RuntimeError(
                "Leakage detected after grouped random split. "
                "Found shared unique-column values across train/val/test."
            )

        splits = {}
        for split_name, split_groups in assignments.items():
            split_mask = pd.Series(group_values).isin(split_groups).to_numpy()
            splits[split_name] = full_dataset_df.loc[split_mask].reset_index(drop=True)

        self.logger.info(
            "Grouped random split target rows: "
            f"train={target_rows['train']:.2f}, "
            f"val={target_rows['val']:.2f}, "
            f"test={target_rows['test']:.2f}"
        )
        self.logger.info(
            "Grouped random split assigned rows by unique value: "
            f"train={assigned_rows['train']}, "
            f"val={assigned_rows['val']}, "
            f"test={assigned_rows['test']}"
        )
        self.logger.info(
            f"Full dataset splits (unique_col={unique_col}): "
            f"Train: {splits['train'].shape}, ({100 * splits['train'].shape[0] / self.TOTAL_ROWS:.2f}%)  "
            f"Val: {splits['val'].shape}, ({100 * splits['val'].shape[0] / self.TOTAL_ROWS:.2f}%)  "
            f"Test: {splits['test'].shape}, ({100 * splits['test'].shape[0] / self.TOTAL_ROWS:.2f}%)"
        )
        self.logger.info(
            "Verified zero shared unique-column values across train/val/test."
        )

        return splits

    def generate_splits(self):
        self.logger.info("Loading full dataset...")
        full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH)
        self.logger.info(f"Full dataset shape: {full_dataset_df.shape}")

        all_splits = {}
        for unique_col in self.UNIQUE_COLS:
            splits = self._generate_splits_for_column(
                full_dataset_df=full_dataset_df, unique_col=unique_col
            )
            output_dir = self._grouped_output_dir(unique_col)
            self.save_named_splits(splits=splits, output_dir=output_dir)
            all_splits[unique_col] = splits

        if len(self.UNIQUE_COLS) == 1:
            return all_splits[self.UNIQUE_COLS[0]]

        return all_splits
