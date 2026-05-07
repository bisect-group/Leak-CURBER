from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def _write_parquet_task(args: tuple[pd.DataFrame, str]) -> str:
    df, output_path = args
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="brotli", index=False)
    return str(path)


class BaseSplitter(ABC):
    def _load_similarity_cache(
        self,
        *,
        cache_path: Path,
        key_column: str,
        value_column: str,
        ordered_keys: list,
    ) -> tuple[np.ndarray, dict] | None:
        if not cache_path.exists():
            return None

        try:
            cache_df = pd.read_csv(cache_path, sep="\t")
        except Exception as exc:
            self.logger.warning(f"Failed to read cache file {cache_path}: {exc}")
            return None

        if key_column not in cache_df.columns or value_column not in cache_df.columns:
            self.logger.warning(
                f"Cache file {cache_path} missing expected columns "
                f"('{key_column}', '{value_column}'). Ignoring cache."
            )
            return None

        cache_df[value_column] = pd.to_numeric(cache_df[value_column], errors="coerce")
        cache_df = cache_df.dropna(subset=[key_column, value_column])
        sim_map = dict(zip(cache_df[key_column], cache_df[value_column]))
        missing_keys = [key for key in ordered_keys if key not in sim_map]
        if missing_keys:
            sample = ", ".join(str(k) for k in missing_keys[:10])
            raise ValueError(
                f"Cache file {cache_path} is incomplete: missing {len(missing_keys)} "
                f"of {len(ordered_keys)} required keys for '{key_column}'. "
                f"Examples: [{sample}]. Recompute with similarity_cache_overwrite=true."
            )

        values = np.array([float(sim_map[key]) for key in ordered_keys], dtype=float)

        self.logger.info(
            f"Loaded similarity cache from {cache_path} "
            f"({len(sim_map)} cached keys; {len(ordered_keys)} requested)."
        )
        return values, sim_map

    def _save_similarity_cache(
        self,
        *,
        cache_path: Path,
        key_column: str,
        value_column: str,
        ordered_keys: list,
        values: np.ndarray,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({key_column: ordered_keys, value_column: values}).to_csv(
            cache_path, sep="\t", index=False
        )
        self.logger.info(f"Saved similarity cache to {cache_path}")

    def _save_split_tasks_in_parallel(
        self, split_tasks: list[tuple[pd.DataFrame, Path]]
    ) -> None:
        if not split_tasks:
            return

        max_workers = os.cpu_count() or 1
        serialized_tasks = [(df, str(path)) for df, path in split_tasks]
        self.logger.info(
            f"Saving {len(split_tasks)} parquet split files with {max_workers} workers..."
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_write_parquet_task, task): task[1]
                for task in serialized_tasks
            }
            for future in as_completed(futures):
                output_path = future.result()
                self.logger.info(f"Saved split to {output_path}")

    def _build_threshold_split_tasks(
        self, all_splits: dict, output_dir: Path
    ) -> list[tuple[pd.DataFrame, Path]]:
        tasks = []
        for threshold, splits in all_splits.items():
            threshold_dir = output_dir / f"threshold_{threshold}"
            for split_name, split_df in splits.items():
                tasks.append((split_df, threshold_dir / f"{split_name}.parquet"))
        return tasks

    def _build_named_split_tasks(
        self, splits: dict[str, pd.DataFrame], output_dir: Path
    ) -> list[tuple[pd.DataFrame, Path]]:
        return [
            (split_df, output_dir / f"{split_name}.parquet")
            for split_name, split_df in splits.items()
        ]

    def save_threshold_splits(self, all_splits: dict, output_dir: Path) -> None:
        self._save_split_tasks_in_parallel(
            self._build_threshold_split_tasks(all_splits=all_splits, output_dir=output_dir)
        )
        self.logger.info(f"All splits saved to {output_dir}")

    def save_named_splits(self, splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
        self._save_split_tasks_in_parallel(
            self._build_named_split_tasks(splits=splits, output_dir=output_dir)
        )
        self.logger.info(f"All splits saved to {output_dir}")

    def _build_value_map_from_similarity_dicts(self, all_similarities: dict) -> dict:
        return {
            threshold: {
                split_name: np.asarray(
                    list(similarities[split_name].values()), dtype=float
                )
                for split_name in ("val", "test")
            }
            for threshold, similarities in all_similarities.items()
        }

    def _build_value_map_from_split_frames(
        self, all_splits: dict, all_similarities: dict, key_column: str
    ) -> dict:
        values_by_threshold = {}
        for threshold, splits in all_splits.items():
            values_by_threshold[threshold] = {}
            for split_name in ("val", "test"):
                sim_map = all_similarities[threshold][split_name]
                values_by_threshold[threshold][split_name] = (
                    splits[split_name][key_column].map(sim_map).fillna(0.0).to_numpy()
                )
        return values_by_threshold

    def _format_summary_stats_text(
        self, val_values: np.ndarray, test_values: np.ndarray
    ) -> str:
        if len(val_values) == 0 or len(test_values) == 0:
            return ""
        return (
            f"Val Mean: {np.mean(val_values):.3f}\n"
            f"Test Mean: {np.mean(test_values):.3f}\n"
            f"Val Median: {np.median(val_values):.3f}\n"
            f"Test Median: {np.median(test_values):.3f}"
        )

    def _format_detailed_stats_text(
        self, val_values: np.ndarray, test_values: np.ndarray, threshold: float
    ) -> str:
        def _stats(values: np.ndarray) -> dict[str, float | int]:
            total = len(values)
            if total == 0:
                return {
                    "mean": 0.0,
                    "median": 0.0,
                    "max": 0.0,
                    "min": 0.0,
                    "above": 0,
                    "below": 0,
                    "pct_above": 0.0,
                    "pct_below": 0.0,
                }

            above = int(np.sum(values >= threshold))
            below = int(np.sum(values < threshold))
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
                "above": above,
                "below": below,
                "pct_above": 100 * above / total,
                "pct_below": 100 * below / total,
            }

        test_stats = _stats(test_values)
        val_stats = _stats(val_values)
        return (
            f"Test Mean: {test_stats['mean']:.3f}\n"
            f"Test Median: {test_stats['median']:.3f}\n"
            f"Test Max: {test_stats['max']:.3f}\n"
            f"Test Min: {test_stats['min']:.3f}\n"
            f"Test Above: {test_stats['above']} ({test_stats['pct_above']:.1f}%)\n"
            f"Test Below: {test_stats['below']} ({test_stats['pct_below']:.1f}%)\n\n"
            f"Val Mean: {val_stats['mean']:.3f}\n"
            f"Val Median: {val_stats['median']:.3f}\n"
            f"Val Max: {val_stats['max']:.3f}\n"
            f"Val Min: {val_stats['min']:.3f}\n"
            f"Val Above: {val_stats['above']} ({val_stats['pct_above']:.1f}%)\n"
            f"Val Below: {val_stats['below']} ({val_stats['pct_below']:.1f}%)"
        )

    def _plot_threshold_split_distributions(
        self,
        values_by_threshold: dict,
        *,
        bins: np.ndarray,
        output_path: Path,
        xlabel: str,
        ylabel: str,
        title_fn,
        stats_mode: str,
        val_label: str = "Val",
        test_label: str = "Test",
        threshold_label_fn=None,
        legend_loc: str = "upper left",
        hist_range: tuple[float, float] | None = None,
        xlim: tuple[float, float] = (0, 1),
    ) -> None:
        import matplotlib.pyplot as plt

        thresholds = sorted(values_by_threshold.keys())
        n_plots = len(thresholds)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        all_counts = []
        for threshold in thresholds:
            for split_name in ("val", "test"):
                values = values_by_threshold[threshold][split_name]
                counts, _ = np.histogram(values, bins=bins)
                all_counts.extend(counts)

        y_max = max(max(all_counts) if all_counts else 0, 1) * 1.1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, threshold in enumerate(thresholds):
            val_values = values_by_threshold[threshold]["val"]
            test_values = values_by_threshold[threshold]["test"]
            ax = axes[idx]

            hist_kwargs = {"bins": bins, "alpha": 0.5, "edgecolor": "black"}
            if hist_range is not None:
                hist_kwargs["range"] = hist_range

            ax.hist(
                val_values,
                label=val_label,
                color="blue",
                **hist_kwargs,
            )
            ax.hist(
                test_values,
                label=test_label,
                color="red",
                **hist_kwargs,
            )
            ax.axvline(
                threshold,
                color="green",
                linestyle="--",
                linewidth=2,
                label=threshold_label_fn(threshold)
                if threshold_label_fn is not None
                else f"Threshold ({threshold})",
            )
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(
                title_fn(threshold, val_values, test_values),
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlim(*xlim)
            ax.set_ylim(0, y_max)
            ax.legend(loc=legend_loc, fontsize=10)

            if stats_mode == "summary":
                stats_text = self._format_summary_stats_text(val_values, test_values)
            elif stats_mode == "detailed":
                stats_text = self._format_detailed_stats_text(
                    val_values, test_values, threshold
                )
            else:
                raise ValueError(f"Unsupported stats mode: {stats_mode}")

            if stats_text:
                ax.text(
                    0.98 if stats_mode == "detailed" else 0.02,
                    0.97 if stats_mode == "detailed" else 0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    horizontalalignment="right"
                    if stats_mode == "detailed"
                    else "left",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        for idx in range(n_plots, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"Saved plot to {output_path}")


class BaseThresholdedSimilaritySplitter(BaseSplitter, ABC):
    @abstractmethod
    def run_splits_across_thresholds(self) -> tuple[dict, dict]:
        raise NotImplementedError

    @abstractmethod
    def plot_unique_distribution(self, all_similarities: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot_full_distribution(self, all_splits: dict, all_similarities: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_output_dir(self) -> Path:
        raise NotImplementedError

    def generate_splits(self) -> tuple[dict, dict]:
        all_splits, all_similarities = self.run_splits_across_thresholds()
        self.plot_unique_distribution(all_similarities)
        self.plot_full_distribution(all_splits, all_similarities)
        self.save_threshold_splits(all_splits=all_splits, output_dir=self.get_output_dir())
        return all_splits, all_similarities
