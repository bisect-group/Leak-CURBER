from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import ks_2samp

from src.data.components.embedders.store import index_files, load_index
from src.utils.rich_progress import progress


def configured(value, default):
    # Keeps a configured 0 (seed 0, a 0.0 fraction), which "value or default" would replace.
    return default if value is None else value

# Flat split-config keys (older config layout) that can hold the input dataset path.
LEGACY_INPUT_PATH_KEYS = (
    "tanimoto_split_input_dataset_parquet_file_path",
    "smiles_split_input_dataset_parquet_file_path",
    "conformer_split_input_dataset_parquet_file_path",
)


QUANTILES = (1, 5, 25, 50, 75, 95, 99)


def split_shift_stats(splits: dict[str, pd.DataFrame], target: str = "log10_value") -> dict | None:
    # Target distribution per split, its shift from train, and the error of predicting the train mean.
    if "train" not in splits or not all(
        target in df and pd.api.types.is_numeric_dtype(df[target]) for df in splits.values()
    ):
        return None
    values = {name: df[target].dropna().to_numpy(float) for name, df in splits.items()}
    if not all(len(y) for y in values.values()):
        return None
    train = values["train"]
    train_q = dict(zip(QUANTILES, np.percentile(train, QUANTILES)))
    stats: dict = {"target": target, "train_mean": float(train.mean()), "splits": {}, "vs_train": {}}
    for name, y in values.items():
        stats["splits"][name] = {
            "n": len(y),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "quantiles": {f"q{q}": float(v) for q, v in zip(QUANTILES, np.percentile(y, QUANTILES))},
        }
        if name == "train":
            continue
        err = y - train.mean()
        sse = float(err @ err)
        sst = float(((y - y.mean()) ** 2).sum())
        stats["vs_train"][name] = {
            "mean_shift": float(y.mean() - train.mean()),
            "median_shift": float(np.median(y) - np.median(train)),
            "ks": float(ks_2samp(y, train).statistic),
            "share_outside_train_range": float(np.mean((y < train.min()) | (y > train.max()))),
            "share_outside_train_q1_q99": float(np.mean((y < train_q[1]) | (y > train_q[99]))),
            "train_mean_rmse": float(np.sqrt(sse / len(y))),
            "train_mean_mae": float(np.abs(err).mean()),
            "train_mean_r2": float(1 - sse / sst) if sst > 0 else None,
        }
    return stats


def _write_parquet_task(args: tuple[object, str]) -> str:
    df, output_path = args
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Arrow tables use pq.write_table, pandas frames to_parquet, polars frames write_parquet.
    if isinstance(df, pa.Table):
        pq.write_table(df, path, compression="zstd", compression_level=3)
    elif isinstance(df, pd.DataFrame):
        df.to_parquet(path, compression="zstd", compression_level=3, index=False)
    else:
        df.write_parquet(path, compression="zstd", compression_level=3)
    return str(path)


def _frame_nbytes(df: object) -> int:
    if isinstance(df, pa.Table):
        return df.nbytes
    if isinstance(df, pd.DataFrame):
        return int(df.memory_usage(index=False, deep=True).sum())
    return df.estimated_size()


# A worker holds an unpickled copy of its frame plus the parquet encoding buffers.
WORKER_MEMORY_FACTOR = 3
MEMORY_BUDGET_FRACTION = 0.5


def _memory_bounded_workers(frames: list[object], max_workers: int) -> int:
    """Largest worker count whose concurrent frame copies fit the share of free RAM."""
    budget = psutil.virtual_memory().available * MEMORY_BUDGET_FRACTION
    per_worker = WORKER_MEMORY_FACTOR * max(_frame_nbytes(df) for df in frames)
    return max(1, min(max_workers, int(budget // per_worker)))


def _arrow_schema(df: object) -> pa.Schema:
    if isinstance(df, pd.DataFrame):
        return pa.Schema.from_pandas(df, preserve_index=False)
    return df.head(0).to_arrow().schema


def unify_null_columns(frames: list[object]) -> list[object]:
    """Give each all-null column the first non-null Arrow type seen in the sibling frames.

    Frames with an all-null column become Arrow tables; other frames are returned unchanged.
    """
    schemas = [_arrow_schema(df) for df in frames]
    types: dict[str, pa.DataType] = {}
    for schema in schemas:
        for field in schema:
            if not pa.types.is_null(field.type):
                types.setdefault(field.name, field.type)
    unified = []
    for df, schema in zip(frames, schemas):
        null_names = [f.name for f in schema if pa.types.is_null(f.type) and f.name in types]
        if not null_names:
            unified.append(df)
            continue
        table = df if isinstance(df, pa.Table) else (
            pa.Table.from_pandas(df, preserve_index=False)
            if isinstance(df, pd.DataFrame)
            else df.to_arrow()
        )
        for name in null_names:
            table = table.set_column(
                table.schema.get_field_index(name), name, table[name].cast(types[name])
            )
        unified.append(table)
    return unified


def order_by_similarity(indices, sims, row_counts, keys) -> np.ndarray:
    """Return indices sorted by (similarity, row count, key); the sort is stable."""
    # Break similarity ties on count, then key, so input-row order never decides the split.
    indices = np.asarray(indices, dtype=int)
    return indices[
        np.lexsort(
            (np.asarray(keys)[indices], np.asarray(row_counts)[indices], np.asarray(sims)[indices])
        )
    ]


def greedy_fill(
    indices: Iterable[int],
    row_count_of: Callable[[int], int],
    max_rows: int,
    desc: str | None = None,
) -> tuple[list[int], int]:
    """Take keys in order, skipping any key that would pass max_rows.

    Returns the taken indices and their row total.
    """
    # A key that does not fit is skipped and the next one is tried.
    if desc is not None:
        indices = progress(indices, desc=desc, leave=False)
    selected: list[int] = []
    total = 0
    for idx in indices:
        row_count = row_count_of(idx)
        if total + row_count <= max_rows:
            selected.append(idx)
            total += row_count
    return selected, total


def first_present(cfg, keys: Iterable[str]):
    """Return the value of the first key that exists in cfg, else None."""
    for key in keys:
        if key in cfg:
            return cfg[key]
    return None


def find_input_dataset_path(dataset_cfg, block_names: Iterable[str]) -> str | None:
    """Find the input parquet path in split blocks, then in the flat legacy keys."""
    # Only the block order differs per caller.
    for block_name in block_names:
        block = dataset_cfg.get(block_name)
        if block and block.get("input_dataset_parquet_file_path"):
            return block.input_dataset_parquet_file_path
    for key in LEGACY_INPUT_PATH_KEYS:
        value = dataset_cfg.get(key)
        if value:
            return value
    return None


class BaseSplitter(ABC):
    def _embedding_validity_mask(self, per_key_embs: dict) -> np.ndarray:
        # Validity comes from the inputs, never from a zero similarity, so
        # cold and warm-cache runs agree. Keys with no embedding or an all-zero
        # embedding are invalid: they stay in train and are never val/test.
        mask = np.array(
            [
                key in per_key_embs and bool(np.any(per_key_embs[key]))
                for key in self.unique_keys
            ],
            dtype=bool,
        )
        self.logger.info(
            f"Keys forced to train (no or all-zero embedding): {int((~mask).sum()):,}"
        )
        return mask

    @staticmethod
    def _cache_fingerprint(keys: Iterable, **params) -> str:
        """Hash of the sorted input keys and the parameters that change the cached values."""
        key_hash = hashlib.sha256("\n".join(sorted(map(str, keys))).encode()).hexdigest()
        payload = json.dumps({"keys": key_hash, "params": params}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_matches(self, cache_path: Path, fingerprint: str) -> bool:
        # The fingerprint sidecar holds a hash of the inputs and parameters; a mismatch is a cache miss.
        sidecar = cache_path.with_name(cache_path.name + ".fingerprint")
        if sidecar.exists() and sidecar.read_text().strip() == fingerprint:
            return True
        self.logger.info(
            f"Cache {cache_path} has no matching input/parameter fingerprint; recomputing."
        )
        return False

    @staticmethod
    def _write_cache_fingerprint(cache_path: Path, fingerprint: str) -> None:
        cache_path.with_name(cache_path.name + ".fingerprint").write_text(fingerprint)

    def _load_similarity_cache(
        self,
        *,
        cache_path: Path,
        key_column: str,
        value_column: str,
        ordered_keys: list,
        fingerprint: str,
    ) -> tuple[np.ndarray, dict] | None:
        if not cache_path.exists() or not self._cache_matches(cache_path, fingerprint):
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
        # NaN values are kept: LDDT caches use NaN for "no structure".
        cache_df = cache_df.dropna(subset=[key_column])
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
        fingerprint: str,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({key_column: ordered_keys, value_column: values}).to_csv(
            cache_path, sep="\t", index=False
        )
        self._write_cache_fingerprint(cache_path, fingerprint)
        self.logger.info(f"Saved similarity cache to {cache_path}")

    def _save_split_tasks_in_parallel(
        self, split_tasks: list[tuple[object, Path]]
    ) -> None:
        if not split_tasks:
            return

        frames = unify_null_columns([df for df, _ in split_tasks])
        serialized_tasks = [(df, str(path)) for df, (_, path) in zip(frames, split_tasks)]
        # Cap the pool at the task count and at what fits in memory.
        max_workers = _memory_bounded_workers(frames, min(len(split_tasks), os.cpu_count() or 1))
        if max_workers == 1:
            # One worker at a time: write in this process and skip pickling the frame.
            for task in serialized_tasks:
                self.logger.info(f"Saved split to {_write_parquet_task(task)}")
            return
        self.logger.info(
            f"Saving {len(split_tasks)} parquet split files with {max_workers} workers..."
        )

        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_write_parquet_task, task): task[1]
                for task in serialized_tasks
            }
            for future in as_completed(futures):
                output_path = future.result()
                self.logger.info(f"Saved split to {output_path}")

    def _build_threshold_split_tasks(
        self, all_splits: dict, output_dir: Path
    ) -> list[tuple[object, Path]]:
        tasks = []
        for threshold, splits in all_splits.items():
            threshold_dir = output_dir / f"threshold_{threshold}"
            for split_name, split_df in splits.items():
                tasks.append((split_df, threshold_dir / f"{split_name}.parquet"))
        return tasks

    def _build_named_split_tasks(
        self, splits: dict[str, pd.DataFrame], output_dir: Path
    ) -> list[tuple[object, Path]]:
        return [
            (split_df, output_dir / f"{split_name}.parquet")
            for split_name, split_df in splits.items()
        ]

    def _write_split_manifest(self, output_dir: Path) -> None:
        # Write the resolved split config, seed and git commit next to the split files.
        from omegaconf import OmegaConf

        cfg = getattr(self, "cfg", None)  # set by scripts/generate_splits.py
        try:
            # Run in the repo, not the process cwd, so the commit is this code's.
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).resolve().parents[4],
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        manifest = {
            "git_commit": commit,
            "seed": getattr(self, "RANDOM_SEED", None),
            "config": None if cfg is None else OmegaConf.to_container(cfg.splits, resolve=True),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

    def save_threshold_splits(self, all_splits: dict, output_dir: Path) -> None:
        self._save_split_tasks_in_parallel(
            self._build_threshold_split_tasks(all_splits=all_splits, output_dir=output_dir)
        )
        self._write_split_manifest(output_dir)
        self.logger.info(f"All splits saved to {output_dir}")

    def save_named_splits(self, splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
        self._save_split_tasks_in_parallel(
            self._build_named_split_tasks(splits=splits, output_dir=output_dir)
        )
        self._write_split_manifest(output_dir)
        if (stats := split_shift_stats(splits)) is not None:
            (output_dir / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
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
                values = splits[split_name][key_column]
                if isinstance(values, pd.Series):
                    values_by_threshold[threshold][split_name] = (
                        values.map(sim_map).fillna(0.0).to_numpy()
                    )
                else:
                    values_by_threshold[threshold][split_name] = np.asarray(
                        [sim_map.get(value, 0.0) for value in values.to_list()],
                        dtype=float,
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
        # Close the figure after saving so figures do not pile up in memory.
        plt.close(fig)
        self.logger.info(f"Saved plot to {output_path}")


class BaseThresholdedSimilaritySplitter(BaseSplitter, ABC):
    # Each subclass sets its vector dtype and whether 2D arrays are mean-pooled.
    EMBEDDING_DTYPE = np.float32
    MEAN_POOL_2D_EMBEDDINGS = False

    def _similarity_fingerprint(self, **params) -> str:
        """Fingerprint of unique keys, embedding source identity and splitter parameters."""
        if self._embedding_source == "store":
            # meta.json alone does not change when shards are added; add the index stat, including
            # unmerged index parts.
            index = [
                (path.name, path.stat().st_size, path.stat().st_mtime_ns)
                for path in index_files(self._embedding_source_path)
            ]
            source = ((self._embedding_source_path / "meta.json").read_bytes(), index)
        elif self._embedding_source == "npz":
            stat = self._embedding_source_path.stat()
            source = (str(self._embedding_source_path), stat.st_size, stat.st_mtime_ns)
        else:
            source = self.EMBEDDING_COLUMN
        return self._cache_fingerprint(
            self.unique_keys,
            splitter=type(self).__name__,
            key_column=self.KEY_COLUMN,
            embedding_source=self._embedding_source,
            source=source,
            **params,
        )

    def _resolve_embedding_source(self, *, npz_name: str, error_hint: str) -> tuple[str, Path]:
        if self.EMBEDDING_COLUMN and self.EMBEDDING_COLUMN in self.full_dataset_df.columns:
            return "column", self.INPUT_DATASET_PATH

        if self.EMBEDDINGS_STORE_DIR is not None:
            meta_path = self.EMBEDDINGS_STORE_DIR / "meta.json"
            if (
                self.EMBEDDINGS_STORE_DIR.exists()
                and meta_path.exists()
                and index_files(self.EMBEDDINGS_STORE_DIR)
            ):
                return "store", self.EMBEDDINGS_STORE_DIR

        npz_candidates = []
        if self.EMBEDDINGS_NPZ is not None:
            npz_candidates.append(self.EMBEDDINGS_NPZ)
        npz_candidates.append(self.INPUT_DATASET_PATH.parent / npz_name)
        npz_candidates.append(self.INPUT_DATASET_PATH.parent.parent / npz_name)

        for candidate in npz_candidates:
            if candidate is not None and candidate.exists():
                return "npz", candidate

        checked = [str(p) for p in npz_candidates if p is not None]
        if self.EMBEDDINGS_STORE_DIR is not None:
            checked.append(str(self.EMBEDDINGS_STORE_DIR))

        raise FileNotFoundError(f"{error_hint} Checked: " + ", ".join(checked))

    def _vector_from_raw(self, raw_value) -> np.ndarray | None:
        if raw_value is None:
            return None

        value = raw_value
        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            if not stripped:
                return None
            try:
                value = json.loads(stripped)
            except Exception:
                return None

        arr = np.asarray(value, dtype=self.EMBEDDING_DTYPE)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and self.MEAN_POOL_2D_EMBEDDINGS:
            return arr.mean(axis=0)
        return None

    def _iter_column_vectors(self):
        """Yield (key, vector) pairs from the dataframe embedding column."""
        for key, raw_vec in self.full_dataset_df[
            [self.KEY_COLUMN, self.EMBEDDING_COLUMN]
        ].itertuples(index=False):
            if pd.isna(key):
                continue
            vec = self._vector_from_raw(raw_vec)
            if vec is None:
                continue
            yield str(key), vec

    def _read_store_vectors(self, store_dir: Path, candidate_hashes: Iterable[str]) -> dict[str, np.ndarray]:
        """Read the vectors for candidate key hashes from a sharded embedding store."""
        index_df = load_index(store_dir, keep="last")  # this splitter always kept the last row
        if index_df.empty:
            self.logger.warning(f"Embedding store index is empty: {store_dir}")
            return {}

        index_df = index_df.drop_duplicates(subset=["key_hash"], keep="last")
        lookup = {
            row.key_hash: (int(row.shard_id), int(row.row_idx))
            for row in index_df.itertuples(index=False)
        }
        present_hashes = [key_hash for key_hash in candidate_hashes if key_hash in lookup]
        self.logger.info(
            f"Store index rows: {len(index_df):,}; candidate hash matches: {len(present_hashes):,}."
        )
        if not present_hashes:
            self.logger.warning("No dataset-relevant embeddings found in sharded store.")
            return {}

        by_shard: dict[int, list[tuple[str, int]]] = {}
        for key_hash in present_hashes:
            shard_id, row_idx = lookup[key_hash]
            by_shard.setdefault(shard_id, []).append((key_hash, row_idx))

        vectors_by_hash: dict[str, np.ndarray] = {}
        for shard_id, rows in by_shard.items():
            shard_path = store_dir / f"shard_{shard_id:06d}.npy"
            if not shard_path.exists():
                self.logger.warning(f"Missing shard file referenced by index: {shard_path}")
                continue
            arr = np.load(shard_path, mmap_mode="r")
            for key_hash, row_idx in rows:
                if 0 <= row_idx < len(arr):
                    vectors_by_hash[key_hash] = np.asarray(
                        arr[row_idx], dtype=self.EMBEDDING_DTYPE
                    )
                else:
                    self.logger.warning(
                        f"Row index {row_idx} out of bounds for shard {shard_path} "
                        f"(len={len(arr)})."
                    )
        return vectors_by_hash

    def _run_search_with_progress(
        self, cmd: list[str], *, tool: str, tmpdir: Path, out_file: Path
    ) -> float:
        """Run an MMseqs2 or Foldseek search and show the size of its temp files as progress.

        Returns the elapsed seconds. Raises RuntimeError if the tool fails.
        """
        start_time = time.time()
        # Drop stdout and send stderr to a file read after exit, so large output cannot block on a full pipe.
        stderr_file = tempfile.TemporaryFile(mode="w+")
        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=stderr_file, text=True
        )

        last_size = 0
        last_log_time = start_time
        log_interval = 10

        with progress(
            desc=f"{tool} all-vs-all",
            unit="MB",
            unit_scale=False,
            bar_format="{desc}: {elapsed} | {n:.1f} MB processed | {rate_fmt}",
        ) as pbar:
            while process.poll() is None:
                time.sleep(2)

                current_size = 0
                try:
                    for f in tmpdir.rglob("*"):
                        if f.is_file():
                            current_size += f.stat().st_size
                    if out_file.exists():
                        current_size += out_file.stat().st_size
                except Exception:
                    pass

                current_size_mb = current_size / (1024 * 1024)
                delta_mb = current_size_mb - last_size
                if delta_mb > 0:
                    pbar.update(delta_mb)
                    last_size = current_size_mb

                current_time = time.time()
                if current_time - last_log_time > log_interval:
                    elapsed_min = (current_time - start_time) / 60
                    self.logger.info(
                        f"[{tool}] Still running... "
                        f"Elapsed: {elapsed_min:.1f} min, "
                        f"Data processed: {current_size_mb:.1f} MB"
                    )
                    last_log_time = current_time

        process.wait()
        stderr_file.seek(0)
        stderr = stderr_file.read()
        stderr_file.close()
        elapsed = time.time() - start_time

        if process.returncode != 0:
            self.logger.error(f"[{tool} easy-search] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[{tool} easy-search] Error: {stderr}")
            raise RuntimeError(f"{tool} easy-search failed: {stderr}")
        return elapsed

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
