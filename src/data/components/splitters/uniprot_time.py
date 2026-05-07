from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.data.components.splitters.base import BaseSplitter
from src.utils.tqdmlogger import TqdmLogger


class UniProtTimeBasedSplitter(BaseSplitter):
    """Chronological splitter using the UniProt accession first-public date."""

    SUPPORTED_DATE_FIELDS = {
        "date_created",
        "date_modified",
        "date_sequence_modified",
    }
    SUPPORTED_MISSING_DATE_HANDLING = {"train", "val", "test", "drop", "error"}

    def __init__(self, cfg: DictConfig):
        self.TRAIN_FRAC = cfg.splits.train_frac or 0.8
        self.VALID_FRAC = cfg.splits.valid_frac or 0.1
        self.TEST_FRAC = cfg.splits.test_frac or 0.1
        total_frac = self.TRAIN_FRAC + self.VALID_FRAC + self.TEST_FRAC
        if abs(total_frac - 1.0) > 1e-8:
            raise ValueError(
                "train_frac + valid_frac + test_frac must sum to 1.0 for UniProtTimeBasedSplitter."
            )

        self.ACC_ID_COLUMN = cfg.splits.get("time_split_acc_id_column")
        self.ACC_ID_DELIMITER = cfg.splits.get("time_split_acc_id_delimiter", "|")
        self.DATE_FIELD = cfg.splits.get("time_split_date_field", "date_created")
        default_dataset_date_column = (
            "uniprot_date" if self.DATE_FIELD == "date_created" else None
        )
        self.DATASET_DATE_COLUMN = cfg.splits.get(
            "time_split_dataset_date_column", default_dataset_date_column
        )
        if self.DATE_FIELD not in self.SUPPORTED_DATE_FIELDS:
            raise ValueError(
                "splits.time_split_date_field must be one of "
                f"{sorted(self.SUPPORTED_DATE_FIELDS)}."
            )

        self.MISSING_DATE_HANDLING = cfg.splits.get(
            "time_split_missing_date_split", "train"
        )
        if self.MISSING_DATE_HANDLING not in self.SUPPORTED_MISSING_DATE_HANDLING:
            raise ValueError(
                "splits.time_split_missing_date_split must be one of "
                f"{sorted(self.SUPPORTED_MISSING_DATE_HANDLING)}."
            )

        dataset_cfg = cfg.splits.dataset
        time_cfg = dataset_cfg.get("time_split")
        input_dataset_path = (
            time_cfg.get("input_dataset_parquet_file_path")
            if time_cfg and time_cfg.get("input_dataset_parquet_file_path")
            else self._resolve_default_input_path(dataset_cfg)
        )
        output_dir = (
            time_cfg.get("output_dir")
            if time_cfg and time_cfg.get("output_dir")
            else self._resolve_default_output_dir(cfg, dataset_cfg, input_dataset_path)
        )

        self.INPUT_DATASET_PATH = Path(input_dataset_path)
        self.OUTPUT_DIR = Path(output_dir)
        self.UNIQUE_DATE_DISTRIBUTION_PLOT_PATH = Path(
            (
                time_cfg.get("unique_distribution_plot_path")
                if time_cfg and time_cfg.get("unique_distribution_plot_path")
                else str(
                    self.OUTPUT_DIR
                    / cfg.splits.plot_names.unique_uniprot_time_distribution
                )
            )
        )
        self.FULL_DATE_DISTRIBUTION_PLOT_PATH = Path(
            (
                time_cfg.get("full_dataset_distribution_plot_path")
                if time_cfg and time_cfg.get("full_dataset_distribution_plot_path")
                else str(
                    self.OUTPUT_DIR
                    / cfg.splits.plot_names.full_uniprot_time_distribution
                )
            )
        )

        log_path = Path(cfg.splits.log_dir)
        for path in [
            log_path,
            self.OUTPUT_DIR,
            self.UNIQUE_DATE_DISTRIBUTION_PLOT_PATH.parent,
            self.FULL_DATE_DISTRIBUTION_PLOT_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=log_path, log_file_name=cfg.splits.log_file_name
        ).get_logger()

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet file not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

    def _resolve_default_input_path(self, dataset_cfg) -> str:
        for block_name in ("sequence_split", "random", "smiles_split", "structure_split"):
            block = dataset_cfg.get(block_name)
            if block and block.get("input_dataset_parquet_file_path"):
                return block.input_dataset_parquet_file_path

        for key in (
            "ec_split_input_dataset_parquet_file_path",
            "tanimoto_split_input_dataset_parquet_file_path",
            "smiles_split_input_dataset_parquet_file_path",
            "conformer_split_input_dataset_parquet_file_path",
        ):
            value = dataset_cfg.get(key)
            if value:
                return value

        raise ValueError(
            "Missing time split input dataset path. Configure "
            "splits.dataset.time_split.input_dataset_parquet_file_path."
        )

    def _resolve_default_output_dir(
        self,
        cfg: DictConfig,
        dataset_cfg,
        input_dataset_path: str,
    ) -> Path:
        output_dir_name = cfg.splits.get(
            "time_split_output_dir_name", "uniprot_time_splits"
        )
        random_cfg = dataset_cfg.get("random")
        random_output_dir = dataset_cfg.get("random_output_dir")
        if random_cfg and random_cfg.get("output_dir"):
            return Path(random_cfg.output_dir).parent / output_dir_name
        if random_output_dir:
            return Path(random_output_dir).parent / output_dir_name
        return Path(input_dataset_path).parent / output_dir_name

    def _target_row_counts(self, total_rows: int) -> dict[str, int]:
        train_rows = int(self.TRAIN_FRAC * total_rows)
        val_rows = int(self.VALID_FRAC * total_rows)
        test_rows = total_rows - train_rows - val_rows
        return {"train": train_rows, "val": val_rows, "test": test_rows}

    def _build_plot_frame(self, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for split_name, split_df in splits.items():
            frame = pd.DataFrame(
                {
                    "split": split_name,
                    "date": pd.to_datetime(
                        split_df[self.DATASET_DATE_COLUMN], errors="coerce"
                    ).dt.normalize(),
                }
            )
            frames.append(frame)

        plot_df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
        plot_df["year"] = plot_df["date"].dt.year.astype(int)
        return plot_df

    def _plot_overall_year_distribution(self, plot_df: pd.DataFrame, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        if plot_df.empty:
            self.logger.warning(f"Skipping empty UniProt time distribution plot: {output_path}")
            return

        year_counts = plot_df.groupby("year").size().sort_index()
        split_counts = plot_df["split"].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            year_counts.index.astype(str),
            year_counts.values,
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(
            "UniProt Year Distribution\n"
            f"Train: {split_counts.get('train', 0)}, "
            f"Val: {split_counts.get('val', 0)}, "
            f"Test: {split_counts.get('test', 0)}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved UniProt date distribution plot to {output_path}")

    def _plot_split_year_distribution(self, plot_df: pd.DataFrame, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        if plot_df.empty:
            self.logger.warning(f"Skipping empty UniProt time distribution plot: {output_path}")
            return

        split_year_counts = (
            plot_df.groupby(["year", "split"]).size().unstack(fill_value=0).sort_index()
        )
        full_dataset_size = len(plot_df)
        for split_name in ("train", "val", "test"):
            if split_name not in split_year_counts.columns:
                split_year_counts[split_name] = 0
        split_year_counts = split_year_counts[["train", "val", "test"]]

        x = range(len(split_year_counts.index))
        width = 0.25
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = {"train": "#4C78A8", "val": "#F58518", "test": "#E45756"}
        offsets = {"train": -width, "val": 0.0, "test": width}

        for split_name in ("train", "val", "test"):
            ax.bar(
                [idx + offsets[split_name] for idx in x],
                split_year_counts[split_name].to_numpy(),
                width=width,
                label=f"{split_name.title()}",
                color=colors[split_name],
                edgecolor="black",
                linewidth=0.4,
            )

        ax.set_title(
            "Full Dataset UniProt Year Distribution\n"
            f"Train: {split_year_counts['train'].sum()} "
            f"({100 * split_year_counts['train'].sum() / full_dataset_size:.1f}%), "
            f"Val: {split_year_counts['val'].sum()} "
            f"({100 * split_year_counts['val'].sum() / full_dataset_size:.1f}%), "
            f"Test: {split_year_counts['test'].sum()} "
            f"({100 * split_year_counts['test'].sum() / full_dataset_size:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.set_xticks(list(x))
        ax.set_xticklabels(split_year_counts.index.astype(str), rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved UniProt date distribution plot to {output_path}")

    def _save_distribution_plots(self, splits: dict[str, pd.DataFrame]) -> None:
        plot_df = self._build_plot_frame(splits)
        self._plot_overall_year_distribution(
            plot_df,
            self.UNIQUE_DATE_DISTRIBUTION_PLOT_PATH,
        )
        self._plot_split_year_distribution(
            plot_df,
            self.FULL_DATE_DISTRIBUTION_PLOT_PATH,
        )

    def generate_splits(self) -> dict[str, pd.DataFrame]:
        self.logger.info("Loading full dataset for UniProt time-based split...")
        full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH).reset_index(drop=True)
        self.logger.info(f"Full dataset shape: {full_dataset_df.shape}")

        if not self.DATASET_DATE_COLUMN:
            msg = (
                "splits.time_split_dataset_date_column is not configured. "
                "Point it at the dataset timestamp column, e.g. 'uniprot_date'."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        if self.DATASET_DATE_COLUMN not in full_dataset_df.columns:
            msg = (
                f"Dataset column '{self.DATASET_DATE_COLUMN}' not found in "
                f"{self.INPUT_DATASET_PATH}. Available columns: {list(full_dataset_df.columns)}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        all_dates = pd.to_datetime(
            full_dataset_df[self.DATASET_DATE_COLUMN], errors="coerce"
        ).dt.normalize()
        self.logger.info(
            f"Using dataset column '{self.DATASET_DATE_COLUMN}' for "
            f"{int(all_dates.notna().sum())} / {len(full_dataset_df)} rows."
        )
        missing_date_mask = all_dates.isna()
        missing_date_count = int(missing_date_mask.sum())

        if missing_date_count:
            msg = (
                f"{missing_date_count} rows do not have a usable UniProt {self.DATE_FIELD} "
                f"timestamp (handling={self.MISSING_DATE_HANDLING})."
            )
            if self.MISSING_DATE_HANDLING == "error":
                raise ValueError(msg)
            self.logger.warning(msg)

        valid_dates = all_dates.loc[~missing_date_mask]
        if valid_dates.empty:
            raise ValueError(
                f"No rows have a valid UniProt {self.DATE_FIELD} timestamp. Cannot build time split."
            )

        date_group_counts = valid_dates.dt.normalize().value_counts().sort_index()
        self.logger.info(
            f"Found {len(date_group_counts)} unique UniProt {self.DATE_FIELD} dates spanning "
            f"{valid_dates.min().date()} to {valid_dates.max().date()}."
        )

        target_counts = self._target_row_counts(len(full_dataset_df))
        forced_counts = {"train": 0, "val": 0, "test": 0}
        if self.MISSING_DATE_HANDLING in forced_counts:
            forced_counts[self.MISSING_DATE_HANDLING] = missing_date_count

        assigned_date_groups: dict[str, set[pd.Timestamp]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        running_counts = forced_counts.copy()
        for date_value, row_count in date_group_counts.items():
            if running_counts["train"] < target_counts["train"]:
                split_name = "train"
            elif running_counts["val"] < target_counts["val"]:
                split_name = "val"
            else:
                split_name = "test"
            assigned_date_groups[split_name].add(date_value)
            running_counts[split_name] += int(row_count)

        normalized_dates = all_dates.dt.normalize()
        split_masks = {
            "train": normalized_dates.isin(assigned_date_groups["train"]),
            "val": normalized_dates.isin(assigned_date_groups["val"]),
            "test": normalized_dates.isin(assigned_date_groups["test"]),
        }
        if self.MISSING_DATE_HANDLING in split_masks:
            split_masks[self.MISSING_DATE_HANDLING] |= missing_date_mask

        splits = {
            split_name: full_dataset_df.loc[mask].reset_index(drop=True)
            for split_name, mask in split_masks.items()
        }

        if self.MISSING_DATE_HANDLING == "drop":
            assigned_rows = sum(len(split_df) for split_df in splits.values())
            expected_rows = len(full_dataset_df) - missing_date_count
            if assigned_rows != expected_rows:
                raise RuntimeError(
                    f"Time split row accounting mismatch after dropping rows with missing dates: "
                    f"assigned={assigned_rows}, expected={expected_rows}."
                )
        else:
            assigned_rows = sum(len(split_df) for split_df in splits.values())
            if assigned_rows != len(full_dataset_df):
                raise RuntimeError(
                    f"Time split row accounting mismatch: assigned={assigned_rows}, "
                    f"expected={len(full_dataset_df)}."
                )

        for split_name, split_df in splits.items():
            if split_df.empty:
                raise RuntimeError(
                    f"Chronological UniProt time split produced an empty '{split_name}' split. "
                    "Adjust split fractions or use a dataset with more distinct accession dates."
                )

        if (
            split_masks["train"] & split_masks["val"]
        ).any() or (split_masks["train"] & split_masks["test"]).any() or (
            split_masks["val"] & split_masks["test"]
        ).any():
            raise RuntimeError("Leakage detected in UniProt time-based split masks.")

        split_date_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}
        for split_name, split_df in splits.items():
            split_dates = pd.to_datetime(
                split_df[self.DATASET_DATE_COLUMN],
                errors="coerce",
            ).dt.normalize().dropna()
            if split_dates.empty:
                split_date_ranges[split_name] = None
                continue
            split_date_ranges[split_name] = (
                split_dates.min().normalize(),
                split_dates.max().normalize(),
            )

        train_range = split_date_ranges["train"]
        val_range = split_date_ranges["val"]
        test_range = split_date_ranges["test"]
        if train_range and val_range and train_range[1] > val_range[0]:
            raise RuntimeError("Train split contains UniProt dates newer than validation split.")
        if val_range and test_range and val_range[1] > test_range[0]:
            raise RuntimeError("Validation split contains UniProt dates newer than test split.")

        total_rows = len(full_dataset_df)
        self.logger.info(
            f"UniProt time-based splits: "
            f"Train: {splits['train'].shape}, ({100 * len(splits['train']) / total_rows:.2f}%)  "
            f"Val: {splits['val'].shape}, ({100 * len(splits['val']) / total_rows:.2f}%)  "
            f"Test: {splits['test'].shape}, ({100 * len(splits['test']) / total_rows:.2f}%)"
        )
        for split_name, date_range in split_date_ranges.items():
            if date_range is None:
                self.logger.info(f"{split_name.title()} date range: no valid dates in split")
            else:
                self.logger.info(
                    f"{split_name.title()} date range: {date_range[0].date()} -> {date_range[1].date()}"
                )

        self._save_distribution_plots(splits)
        self.save_named_splits(splits=splits, output_dir=self.OUTPUT_DIR)
        return splits
