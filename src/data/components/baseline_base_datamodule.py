"""Code shared by the baseline regression and classification datamodules."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.data.components.embedders.store import (
    ShardedEmbeddingReader,
    index_files,
    pending_shard_rows,
)
from src.data.components.embedders.types import CacheMetadata


logger = logging.getLogger(__name__)

# Contiguous row chunks that make up one shuffled training batch.
MIXED_CHUNKS_PER_BATCH = 32


@dataclass(frozen=True)
class ModalitySpec:
    """Lookup details for one embedding modality.

    Each modality has a parquet key column, a read-only embedding store, and a
    zero vector used when a row has no cached embedding for that modality.
    """

    name: str
    key_column: str
    reader: ShardedEmbeddingReader
    zero: np.ndarray


def _to_plain_dict(value) -> dict:
    """Convert optional Hydra/OmegaConf mappings to regular Python dicts."""

    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_digests(root: Path) -> list[list[str]]:
    """Sorted [relative path, blake2b digest] pairs for a store's index files.

    The digest is remembered in a sidecar JSON with the file's size and mtime_ns, so only
    new or changed files are hashed again.
    """
    memo_path = root / ".index_digests.json"
    try:
        memo = json.loads(memo_path.read_text())
    except (OSError, json.JSONDecodeError):
        memo = {}
    files = index_files(root)
    digests, fresh = [], {}
    for path in files:
        rel, stat = str(path.relative_to(root)), path.stat()
        entry = memo.get(rel)
        if not (entry and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns):
            digest = hashlib.blake2b(digest_size=32)
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            entry = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "digest": digest.hexdigest()}
        fresh[rel] = entry
        digests.append([rel, entry["digest"]])
    if fresh != memo:
        tmp_path = memo_path.with_name(f"{memo_path.name}.{os.getpid()}.tmp")
        try:
            tmp_path.write_text(json.dumps(fresh, sort_keys=True))
            os.replace(tmp_path, memo_path)
        except OSError:
            pass  # a read-only store only loses the memo
    return digests


def _is_missing_key(value) -> bool:
    """Return True for null/NaN lookup keys read from parquet."""

    if value is None:
        return True
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _is_missing_key(value.item())
        return value.size == 0
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) == 0
    try:
        missing = np.isnan(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, np.ndarray):
        return bool(missing.size == 0 or missing.all())
    return bool(missing)


def _lookup_embedding(
    spec: ModalitySpec,
    key,
    feature_dtype: np.dtype,
    pooled_cache: Optional[dict[tuple[str, ...], np.ndarray]] = None,
    pooled_cache_size: int = 0,
) -> np.ndarray:
    if _is_missing_key(key):
        return spec.zero
    if isinstance(key, np.ndarray):
        key = key.tolist()
    if isinstance(key, (list, tuple)):
        cache_key = tuple(str(item) for item in key if not _is_missing_key(item))
        if pooled_cache is not None and cache_key in pooled_cache:
            return pooled_cache[cache_key]
        values = [
            np.asarray(value, dtype=np.float32)
            for item in key
            if not _is_missing_key(item)
            for value in [spec.reader.get(str(item))]
            if value is not None
        ]
        if not values:
            return spec.zero
        pooled = np.mean(np.stack(values, axis=0), axis=0).astype(feature_dtype, copy=False)
        if pooled_cache is not None and pooled_cache_size > 0:
            if len(pooled_cache) >= pooled_cache_size:
                pooled_cache.pop(next(iter(pooled_cache)))
            pooled_cache[cache_key] = pooled
        return pooled
    value = spec.reader.get(str(key))
    if value is None:
        return spec.zero
    return np.asarray(value, dtype=feature_dtype)


class BaseParquetFeatureBatchIterableDataset(IterableDataset):
    """Stream parquet rows and turn them into ready-to-train feature batches.

    The source parquet files contain identifiers such as SMILES strings,
    sequences, or accession IDs rather than dense model inputs. This iterable
    reads only the needed columns, looks up each configured modality in its
    sharded embedding store, concatenates those embeddings into ``X``, and
    returns the task label as ``y``.

    The dataset yields complete batches, so callers should use a DataLoader with
    ``batch_size=None``. This avoids materializing the full split in memory.
    Subclasses set ``LABEL_DTYPE`` and implement ``_encode_label``.
    """

    LABEL_DTYPE: type

    def __init__(
        self,
        parquet_path: Path,
        *,
        label_col: str,
        modality_specs: list[ModalitySpec],
        parquet_batch_size: int,
        feature_dtype: str,
        shuffle_rows: bool = False,
        seed: int = 42,
        pooled_list_cache_size: int = 0,
    ) -> None:
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.label_col = label_col
        self.modality_specs = modality_specs
        self.parquet_batch_size = int(parquet_batch_size)
        self.feature_dtype = np.dtype(feature_dtype)
        self.shuffle_rows = bool(shuffle_rows)
        self.seed = int(seed)
        self.pooled_list_cache_size = int(pooled_list_cache_size)
        self._iteration = 0

    def _encode_label(self, value, row_idx: int):
        """Convert one raw parquet label into the value stored in ``y``."""

        raise NotImplementedError

    @property
    def columns(self) -> list[str]:
        """Minimal parquet columns needed to construct labels and embeddings."""

        cols = {self.label_col}
        cols.update(spec.key_column for spec in self.modality_specs)
        return sorted(cols)

    def __len__(self) -> int:
        """Return the number of streamed record batches without loading rows."""

        n_rows = pq.ParquetFile(self.parquet_path).metadata.num_rows
        return int(math.ceil(n_rows / self.parquet_batch_size))

    def __iter__(self):
        """Yield ``(x_batch, y_batch)`` tensors for this worker's parquet batches."""

        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        # The worker seed (new per epoch with fresh workers) plus the iteration counter
        # (persistent or no workers) varies the order each epoch and stays deterministic.
        rng = np.random.default_rng(
            (worker.seed if worker else self.seed + worker_id) + self._iteration
        )
        self._iteration += 1

        parquet_file = pq.ParquetFile(self.parquet_path)
        pooled_cache: dict[tuple[str, ...], np.ndarray] | None = (
            {} if self.pooled_list_cache_size > 0 else None
        )
        # Count missing keys (zero vectors) and zeroed bad values per modality; log the totals
        # when the split is exhausted. Never fails.
        missing = dict.fromkeys((spec.name for spec in self.modality_specs), 0)
        non_finite = dict(missing)
        total_rows = 0
        for batch_idx, batch in enumerate(
            parquet_file.iter_batches(
                batch_size=self.parquet_batch_size,
                columns=self.columns,
            )
        ):
            # IterableDataset replicas exist in each worker. Assign whole
            # parquet record batches round-robin so workers do not duplicate
            # examples.
            if batch_idx % num_workers != worker_id:
                continue

            payload = batch.to_pydict()
            row_count = len(payload[self.label_col])
            order = np.arange(row_count)
            if self.shuffle_rows and row_count > 1:
                # Training shuffles rows within each streamed parquet batch.
                # Validation/test keep parquet order for reproducibility.
                rng.shuffle(order)

            x_batch = np.empty(
                (row_count, sum(spec.zero.shape[0] for spec in self.modality_specs)),
                dtype=self.feature_dtype,
            )
            y_batch = np.empty(row_count, dtype=self.LABEL_DTYPE)

            for output_idx, row_idx in enumerate(order):
                offset = 0
                for spec in self.modality_specs:
                    value = _lookup_embedding(
                        spec,
                        payload[spec.key_column][row_idx],
                        self.feature_dtype,
                        pooled_cache,
                        self.pooled_list_cache_size,
                    )
                    missing[spec.name] += value is spec.zero
                    if not np.isfinite(value).all():
                        non_finite[spec.name] += 1
                        # Cached feature stores should be finite, but sanitize
                        # defensively so a bad embedding does not poison a whole
                        # training run.
                        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                    next_offset = offset + spec.zero.shape[0]
                    x_batch[output_idx, offset:next_offset] = value
                    offset = next_offset
                y_batch[output_idx] = self._encode_label(payload[self.label_col][row_idx], row_idx)

            total_rows += row_count
            yield torch.from_numpy(x_batch), torch.from_numpy(y_batch)

        # Totals of the last full pass; complete only without DataLoader workers.
        self.rows_seen = total_rows
        self.missing_counts = missing
        logger.info(
            "%s (worker %d): %d rows, rows with missing embedding %s, non-finite %s",
            self.parquet_path.name, worker_id, total_rows, missing, non_finite,
        )


class BaseMaterializedFeatureBatchIterableDataset(IterableDataset):
    """Read cached dense features from memmapped ``.npy`` files.

    The dataset yields complete batches so the DataLoader should use
    ``batch_size=None``. Training mixes small contiguous chunks from across the split to
    avoid row-level random access over huge memmaps. Evaluation reads in file order.
    Subclasses set ``LABEL_DTYPE``.
    """

    LABEL_DTYPE: type

    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        *,
        batch_size: int,
        block_rows: int,
        shuffle_blocks: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.batch_size = int(batch_size)
        self.block_rows = max(int(block_rows), int(batch_size))
        self.shuffle_blocks = bool(shuffle_blocks)
        self.seed = int(seed)
        self._iteration = 0
        self.n_rows = int(np.load(self.x_path, mmap_mode="r").shape[0])
        self.chunks_per_batch = min(MIXED_CHUNKS_PER_BATCH, self.batch_size)
        self.chunk_rows = max(self.batch_size // self.chunks_per_batch, 1)

    def __len__(self) -> int:
        if self.shuffle_blocks:
            return math.ceil(math.ceil(self.n_rows / self.chunk_rows) / self.chunks_per_batch)
        return int(math.ceil(self.n_rows / self.batch_size))

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        # All workers share one epoch seed (worker.seed - worker.id, or the iteration
        # counter for persistent or no workers), so they agree on the epoch's order.
        epoch_seed = (worker.seed - worker.id if worker else self.seed) + self._iteration
        self._iteration += 1

        x_array = np.load(self.x_path, mmap_mode="r")
        y_array = np.load(self.y_path, mmap_mode="r")
        if self.shuffle_blocks:
            yield from self._iter_mixed_batches(x_array, y_array, epoch_seed, worker_id, num_workers)
            return

        block_starts = np.arange(0, self.n_rows, self.block_rows)
        for block_position, block_start in enumerate(block_starts):
            if block_position % num_workers != worker_id:
                continue
            block_start = int(block_start)
            block_end = min(block_start + self.block_rows, self.n_rows)
            for start in range(block_start, block_end, self.batch_size):
                end = min(start + self.batch_size, block_end)
                # Copy into a writable contiguous batch. This avoids PyTorch warnings
                # for read-only NumPy views and gives pinned-memory workers a normal tensor.
                x_batch = np.asarray(x_array[start:end]).copy()
                y_batch = np.asarray(y_array[start:end], dtype=self.LABEL_DTYPE).copy()
                yield torch.from_numpy(x_batch), torch.from_numpy(y_batch)

    def _iter_mixed_batches(self, x_array, y_array, epoch_seed: int, worker_id: int, num_workers: int):
        """Yield batches built from contiguous row chunks drawn from random places in the split.

        The split is cut into small chunks, the chunk order is permuted over the whole split
        each epoch, and each batch takes the next ``chunks_per_batch`` chunks. This mixes rows
        across the split while every read stays contiguous.
        """

        chunk_rows, chunks_per_batch = self.chunk_rows, self.chunks_per_batch
        chunk_order = np.random.default_rng(epoch_seed).permutation(math.ceil(self.n_rows / chunk_rows))
        for batch_idx, first in enumerate(range(0, len(chunk_order), chunks_per_batch)):
            if batch_idx % num_workers != worker_id:
                continue
            chunks = [
                slice(int(chunk) * chunk_rows, min((int(chunk) + 1) * chunk_rows, self.n_rows))
                for chunk in chunk_order[first : first + chunks_per_batch]
            ]
            x_batch = np.concatenate([x_array[chunk] for chunk in chunks])
            y_batch = np.concatenate([y_array[chunk] for chunk in chunks]).astype(self.LABEL_DTYPE, copy=False)
            yield torch.from_numpy(x_batch), torch.from_numpy(y_batch)


# Classification and regression datamodules subclass this base and add only endpoint and
# label handling.
class BaselineBaseDataModule(LightningDataModule):
    """Shared embedding-store, split and materialized-cache logic.

    Subclasses call ``save_hyperparameters`` in their own ``__init__`` and define
    the split layout (``_resolve_split_dir``), the dataset classes and the
    task-specific cache metadata (``_metadata_extras``).
    """

    LABEL_DTYPE: type
    MATERIALIZED_DATASET: type[BaseMaterializedFeatureBatchIterableDataset]

    def __init__(
        self,
        *,
        data_root: Path,
        endpoint: str,
        split: str,
        threshold: Optional[str],
        label_col: str,
        modalities: list[str],
        cache_roots: dict[str, str],
        key_columns: dict[str, str],
        batch_size: int,
        parquet_batch_size: Optional[int],
        num_workers: int,
        pin_memory: bool,
        feature_dtype: str,
        shard_cache_size: int,
        materialization_dir: Optional[str],
        materialized_cache_root: Optional[str],
        materialization_max_gb: float,
        use_materialized_features: bool,
        materialize_if_missing: bool,
        refresh_materialized_features: bool,
        materialized_validation: str,
        materialized_feature_dtype: str,
        materialized_block_rows: int,
        materialized_shuffle: str,
        persistent_workers: bool,
        prefetch_factor: Optional[int],
        pooled_list_cache_size: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.endpoint = endpoint
        self.split = split
        self.threshold = threshold
        self.label_col = label_col
        self.modalities = modalities
        self.cache_roots = cache_roots
        self.key_columns = key_columns
        self.batch_size = int(batch_size)
        self.parquet_batch_size = int(parquet_batch_size or batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.feature_dtype = str(feature_dtype)
        self.shard_cache_size = int(shard_cache_size)
        materialization_root = materialized_cache_root or materialization_dir
        self.materialization_dir = Path(materialization_root) if materialization_root else None
        self.materialization_max_gb = float(materialization_max_gb)
        self.use_materialized_features = bool(use_materialized_features)
        self.materialize_if_missing = bool(materialize_if_missing)
        self.refresh_materialized_features = bool(refresh_materialized_features)
        self.materialized_validation = str(materialized_validation)
        self.materialized_feature_dtype = str(materialized_feature_dtype)
        self.materialized_block_rows = int(materialized_block_rows)
        self.materialized_shuffle = str(materialized_shuffle)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = prefetch_factor
        self.pooled_list_cache_size = int(pooled_list_cache_size)
        self.seed = int(seed)

        self.split_dir: Path | None = None
        self.split_paths: dict[str, Path] = {}
        self.readers: dict[str, ShardedEmbeddingReader] = {}
        self.modality_specs: list[ModalitySpec] = []
        self._materialized_info: dict[str, Any] | None = None

    # Hooks -------------------------------------------------------------

    def _resolve_split_dir(self) -> Path:
        raise NotImplementedError

    def _make_parquet_dataset(
        self, split: str, *, shuffle_rows: bool, feature_dtype: str
    ) -> BaseParquetFeatureBatchIterableDataset:
        raise NotImplementedError

    def _metadata_extras(self) -> dict[str, Any]:
        """Task-specific entries of the cache metadata (they enter the fingerprint)."""

        return {}

    def _is_setup(self) -> bool:
        return bool(self.modality_specs)

    def _finish_setup(self) -> None:
        """Run task-specific setup after the modality specs are built."""

    def _existing_materialized_info_if_available(self) -> dict[str, Any] | None:
        return None

    def _write_extra_cache_files(self, tmp_dir: Path, metadata: dict[str, Any]) -> None:
        """Write task-specific files into a new cache directory."""

    # Lightning ---------------------------------------------------------

    def prepare_data(self) -> None:
        """Resolve and validate the train/validation/test parquet paths."""

        self.split_dir = self._resolve_split_dir()
        self.split_paths = {
            split_name: self.split_dir / f"{split_name}.parquet"
            for split_name in ("train", "val", "test")
        }
        for split_name, path in self.split_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{split_name} split file not found: {path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize embedding readers and validate parquet schemas.

        Lightning may call ``setup`` multiple times. The method is idempotent
        after the modality specs have been built.
        """

        if self._is_setup():
            return
        if not self.split_paths:
            self.prepare_data()

        self.readers = {}
        self.modality_specs = []
        for modality in self.modalities:
            if modality not in self.cache_roots:
                raise KeyError(f"Missing cache root for configured modality: {modality}")
            if modality not in self.key_columns:
                raise KeyError(f"Missing key column for configured modality: {modality}")

            # The reader loads cache metadata and a compact key_hash -> shard
            # location index. Actual shard arrays are memory-mapped on demand.
            reader = ShardedEmbeddingReader(
                self.cache_roots[modality],
                shard_cache_size=self.shard_cache_size,
            )
            self.readers[modality] = reader
            self.modality_specs.append(
                ModalitySpec(
                    name=modality,
                    key_column=self.key_columns[modality],
                    reader=reader,
                    zero=np.zeros(reader.embedding_dim, dtype=self.feature_dtype),
                )
            )

        self._validate_required_columns()
        self._finish_setup()

    def get_input_dim(self) -> int:
        """Return the concatenated embedding width expected by the model."""

        self.setup()
        return sum(spec.reader.embedding_dim for spec in self.modality_specs)

    def get_feature_scaling(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-column (mean, std, block id) for the MLP input.

        Each modality block has one scalar mean and std, fitted on the train split only and
        only over rows where the block is not all zero (an all-zero block is a missing embedding).
        The stats are cached next to the materialized train features.
        """
        # Modality scales differ by up to 200x (ECFP std 1, ESM3 std 194). A block with std 0
        # gets std 1 so it is only centred. Zero-filled missing rows are left out of the stats.
        self.setup()
        dims = [spec.reader.embedding_dim for spec in self.modality_specs]
        block_id = np.repeat(np.arange(len(dims)), dims)
        cache_path = None
        if self.use_materialized_features:
            cache_path = Path(self.ensure_materialized_cache()["cache_dir"]) / "input_scaling_v2.npz"
            if cache_path.exists():
                with np.load(cache_path) as saved:
                    return saved["mean"], saved["std"], block_id

        edges = np.concatenate([[0], np.cumsum(dims)])
        total, total_sq, rows = np.zeros(len(dims)), np.zeros(len(dims)), np.zeros(len(dims))
        for x_batch, _ in self.iter_feature_batches("train", feature_dtype="float32"):
            x_batch = x_batch.astype(np.float64, copy=False)
            for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
                block = x_batch[:, start:stop]
                block = block[block.any(axis=1)]
                total[i] += block.sum()
                total_sq[i] += np.square(block).sum()
                rows[i] += block.shape[0]
        n = np.maximum(rows, 1) * np.asarray(dims)
        block_mean = total / n
        block_std = np.sqrt(np.maximum(total_sq / n - block_mean**2, 0.0))
        block_std[block_std == 0] = 1.0
        mean = np.repeat(block_mean, dims).astype(np.float32)
        std = np.repeat(block_std, dims).astype(np.float32)
        if cache_path is not None:
            # Write to a temp file, then replace, so a reader never sees a half-written npz.
            tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            with open(tmp_path, "wb") as handle:
                np.savez(handle, mean=mean, std=std)
            os.replace(tmp_path, cache_path)
        return mean, std, block_id

    def iter_feature_batches(
        self,
        split: str,
        *,
        feature_dtype: Optional[str] = None,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Iterate over one split as NumPy ``(X, y)`` batches.

        This is useful for sklearn-style estimators that consume NumPy arrays
        rather than Torch tensors.
        """

        self.setup()
        dataset = self._make_parquet_dataset(
            split, shuffle_rows=False, feature_dtype=feature_dtype or self.feature_dtype
        )
        for x_batch, y_batch in dataset:
            yield x_batch.numpy(), y_batch.numpy()

    def materialize_split(
        self,
        split: str,
        output_dir: Optional[str | Path] = None,
        *,
        feature_dtype: Optional[str] = None,
    ):
        """Write one split's concatenated features and labels to ``.npy`` files.

        Returns the open memmap arrays and a metadata dict containing the output
        paths and estimated feature-array size. A memory-size guard prevents
        accidentally creating very large materialized feature matrices.
        """

        self.setup()
        split = self._normalize_split_name(split)
        target_dir = Path(output_dir) if output_dir else self.materialization_dir
        if target_dir is None:
            target_dir = Path.cwd() / "materialized_features"
        target_dir.mkdir(parents=True, exist_ok=True)

        n_rows = pq.ParquetFile(self._split_path(split)).metadata.num_rows
        input_dim = self.get_input_dim()
        dtype = np.dtype(feature_dtype or self.feature_dtype)
        estimated_gb = (n_rows * input_dim * dtype.itemsize) / (1024**3)
        if estimated_gb > self.materialization_max_gb:
            raise MemoryError(
                f"Refusing to materialize {split}: estimated X size {estimated_gb:.2f} GiB "
                f"exceeds materialization_max_gb={self.materialization_max_gb:.2f}."
            )

        # open_memmap lets downstream sklearn baselines use dense arrays without
        # keeping the full feature matrix resident in RAM during construction.
        x_path = target_dir / f"{split}_X.npy"
        y_path = target_dir / f"{split}_y.npy"
        x_array = np.lib.format.open_memmap(x_path, mode="w+", dtype=dtype, shape=(n_rows, input_dim))
        y_array = np.lib.format.open_memmap(y_path, mode="w+", dtype=self.LABEL_DTYPE, shape=(n_rows,))

        offset = 0
        for x_batch, y_batch in self.iter_feature_batches(split, feature_dtype=str(dtype)):
            next_offset = offset + x_batch.shape[0]
            x_array[offset:next_offset] = x_batch
            y_array[offset:next_offset] = y_batch
            offset = next_offset
        x_array.flush()
        y_array.flush()
        return x_array, y_array, {"x_path": x_path, "y_path": y_path, "estimated_gb": estimated_gb}

    def missing_embedding_counts(self) -> Optional[dict[str, Any]]:
        """Per-split row and missing-embedding counts recorded in the materialized cache."""

        if not self.use_materialized_features:
            return None
        return self.ensure_materialized_cache().get("missing_embeddings")

    def materialized_cache_status(self) -> dict[str, Any]:
        """Return validation status for this datamodule's materialized cache."""

        metadata = self._materialized_metadata_from_caches(
            load_readers=self.materialized_validation == "full"
        )
        cache_dir = self._materialized_cache_dir(metadata["fingerprint"])
        valid, reason = self._validate_materialized_cache(
            cache_dir,
            metadata,
            policy=self.materialized_validation,
        )
        return {
            "status": "hit" if valid else "missing",
            "reason": reason,
            "cache_dir": str(cache_dir),
            "fingerprint": metadata["fingerprint"],
            "estimated_gb": metadata["estimated_total_gb"],
            "rows": sum(split_info["rows"] for split_info in metadata["splits"].values()),
        }

    def ensure_materialized_cache(self, *, refresh: Optional[bool] = None) -> dict[str, Any]:
        """Validate or build the train/val/test materialized feature cache."""

        refresh = self.refresh_materialized_features if refresh is None else bool(refresh)
        if not refresh:
            info = self._existing_materialized_info_if_available()
            if info is not None:
                self._materialized_info = info
                return info
        metadata = self._materialized_metadata_from_caches(
            load_readers=self.materialized_validation == "full"
        )
        cache_dir = self._materialized_cache_dir(metadata["fingerprint"])
        if not refresh:
            valid, _reason = self._validate_materialized_cache(
                cache_dir,
                metadata,
                policy=self.materialized_validation,
            )
            if valid:
                info = self._materialized_info_from_metadata(cache_dir, metadata, cache_status="hit")
                self._materialized_info = info
                return info
        if not self.materialize_if_missing:
            raise FileNotFoundError(f"Materialized feature cache is missing or stale: {cache_dir}")

        tmp_dir = cache_dir.with_name(f".{cache_dir.name}.tmp.{os.getpid()}")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.setup()
            dtype = np.dtype(self.materialized_feature_dtype)
            input_dim = int(metadata["input_dim"])
            # Rows whose embedding is missing are zero-filled; record how many per split and modality.
            missing_embeddings: dict[str, Any] = {}
            for split_name, split_info in metadata["splits"].items():
                rows = int(split_info["rows"])
                x_array = np.lib.format.open_memmap(
                    tmp_dir / f"{split_name}_X.npy",
                    mode="w+",
                    dtype=dtype,
                    shape=(rows, input_dim),
                )
                y_array = np.lib.format.open_memmap(
                    tmp_dir / f"{split_name}_y.npy",
                    mode="w+",
                    dtype=self.LABEL_DTYPE,
                    shape=(rows,),
                )
                offset = 0
                dataset = self._make_parquet_dataset(split_name, shuffle_rows=False, feature_dtype=str(dtype))
                for x_batch, y_batch in dataset:
                    next_offset = offset + x_batch.shape[0]
                    x_array[offset:next_offset] = x_batch.numpy()
                    y_array[offset:next_offset] = y_batch.numpy()
                    offset = next_offset
                missing_embeddings[split_name] = {"rows": dataset.rows_seen, "missing": dataset.missing_counts}
                if offset != rows:
                    raise RuntimeError(f"Materialized {split_name} wrote {offset} rows, expected {rows}")
                x_array.flush()
                y_array.flush()
                del x_array, y_array
            written = {**metadata, "missing_embeddings": missing_embeddings}
            (tmp_dir / "metadata.json").write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
            self._write_extra_cache_files(tmp_dir, metadata)
            (tmp_dir / ".complete").write_text("ok\n")
            valid, reason = self._validate_materialized_cache(tmp_dir, metadata)
            if not valid:
                raise RuntimeError(f"New materialized cache failed validation: {reason}")
            self._publish_materialized_cache(tmp_dir, cache_dir, metadata, refresh)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

        info = self._materialized_info_from_metadata(cache_dir, metadata, cache_status="built")
        self._materialized_info = info
        return info

    def _publish_materialized_cache(
        self, tmp_dir: Path, cache_dir: Path, metadata: dict[str, Any], refresh: bool
    ) -> None:
        """Move the finished temp cache to `cache_dir` without deleting a valid cache.

        A valid existing cache is kept; an old one moves aside and is deleted only after the
        new one is in place; only "already exists" errors are ignored.
        """
        if not refresh and cache_dir.exists():
            if self._validate_materialized_cache(cache_dir, metadata)[0]:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return
        old_dir = cache_dir.with_name(f".{cache_dir.name}.old.{os.getpid()}")
        if cache_dir.exists():
            cache_dir.rename(old_dir)
        try:
            tmp_dir.rename(cache_dir)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                if old_dir.exists() and not cache_dir.exists():
                    old_dir.rename(cache_dir)
                raise
            shutil.rmtree(tmp_dir, ignore_errors=True)  # another process published first
        shutil.rmtree(old_dir, ignore_errors=True)

    def train_dataloader(self) -> DataLoader:
        """Return streamed training batches with within-batch row shuffling."""

        return self._dataloader("train", shuffle_rows=True)

    def val_dataloader(self) -> DataLoader:
        """Return streamed validation batches in deterministic parquet order."""

        return self._dataloader("val", shuffle_rows=False)

    def test_dataloader(self) -> DataLoader:
        """Return streamed test batches in deterministic parquet order."""

        return self._dataloader("test", shuffle_rows=False)

    def _dataloader(self, split: str, *, shuffle_rows: bool) -> DataLoader:
        self.setup()
        if self.use_materialized_features:
            info = self.ensure_materialized_cache()
            split_info = info["splits"][self._normalize_split_name(split)]
            dataset = self.MATERIALIZED_DATASET(
                Path(split_info["x_path"]),
                Path(split_info["y_path"]),
                batch_size=self.batch_size,
                block_rows=self.materialized_block_rows,
                shuffle_blocks=shuffle_rows and self.materialized_shuffle == "block",
                seed=self.seed,
            )
            return self._build_dataloader(dataset)

        return self._build_dataloader(
            self._make_parquet_dataset(split, shuffle_rows=shuffle_rows, feature_dtype=self.feature_dtype)
        )

    def _build_dataloader(self, dataset: IterableDataset) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": None,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        # The iterable dataset already yields complete feature batches.
        return DataLoader(dataset, **kwargs)

    def _materialized_metadata_from_caches(self, *, load_readers: bool = False) -> dict[str, Any]:
        if not self.split_paths:
            self.prepare_data()
        if load_readers:
            input_dim = int(self.get_input_dim())
            cache_payload = self._cache_metadata_from_readers()
        else:
            cache_payload, input_dim = self._cache_metadata_without_readers()
        dtype = np.dtype(self.materialized_feature_dtype)
        split_payload: dict[str, Any] = {}
        estimated_total_gb = 0.0
        for split_name, path in self.split_paths.items():
            parquet_file = pq.ParquetFile(path)
            rows = int(parquet_file.metadata.num_rows)
            estimated_gb = (rows * input_dim * dtype.itemsize) / (1024**3)
            estimated_total_gb += estimated_gb
            stat = path.stat()
            # The streamed sha256 of each split parquet joins the fingerprint, since size and
            # mtime miss rebuilt or copied splits.
            split_payload[split_name] = {
                "path": str(path.resolve()),
                "rows": rows,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(path),
                "x_shape": [rows, input_dim],
                "y_shape": [rows],
                "estimated_x_gb": estimated_gb,
            }
        if estimated_total_gb > self.materialization_max_gb:
            raise MemoryError(
                f"Refusing to materialize {self.endpoint}/{self.split}: estimated total X size "
                f"{estimated_total_gb:.2f} GiB exceeds materialization_max_gb="
                f"{self.materialization_max_gb:.2f}."
            )

        # Readers ignore _pending_shard.*, so pending rows become zero vectors; warn with the
        # store path and pending row count before materializing.
        for name, info in cache_payload.items():
            if pending := pending_shard_rows(Path(info["root"])):
                warnings.warn(
                    f"Embedding store {name} ({info['root']}) has {pending} unflushed pending rows "
                    "that readers ignore; they become zero vectors.",
                    stacklevel=2,
                )

        payload = {
            "version": 2,
            "endpoint": self.endpoint,
            "split": self.split,
            "threshold": self.threshold,
            "split_dir": str(self.split_dir.resolve()) if self.split_dir else None,
            "label_col": self.label_col,
            "modalities": list(self.modalities),
            "key_columns": {name: self.key_columns[name] for name in self.modalities},
            "caches": cache_payload,
            "splits": split_payload,
            "input_dim": input_dim,
            "feature_dtype": str(dtype),
            "estimated_total_gb": estimated_total_gb,
            **self._metadata_extras(),
        }
        fingerprint_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
        return payload

    def _cache_metadata_from_readers(self) -> dict[str, Any]:
        cache_payload: dict[str, Any] = {}
        for spec in self.modality_specs:
            root = Path(self.cache_roots[spec.name])
            meta_path = root / "meta.json"
            index_stats = [path.stat() for path in index_files(root)]
            cache_payload[spec.name] = {
                "root": str(root.resolve()),
                "key_column": spec.key_column,
                "embedding_dim": int(spec.reader.embedding_dim),
                "meta_sha256": _file_sha256(meta_path),
                "index_size": sum(st.st_size for st in index_stats),
                "index_mtime_ns": max(st.st_mtime_ns for st in index_stats),
                "index_digests": _index_digests(root),
            }
        return cache_payload

    def _cache_metadata_without_readers(self) -> tuple[dict[str, Any], int]:
        cache_payload: dict[str, Any] = {}
        input_dim = 0
        for modality in self.modalities:
            if modality not in self.cache_roots:
                raise KeyError(f"Missing cache root for configured modality: {modality}")
            if modality not in self.key_columns:
                raise KeyError(f"Missing key column for configured modality: {modality}")
            root = Path(self.cache_roots[modality])
            meta_path = root / "meta.json"
            if not root.exists():
                raise FileNotFoundError(f"Embedding store directory not found: {root}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Embedding store metadata not found: {meta_path}")
            if not index_files(root):  # unmerged parts count as an index
                raise FileNotFoundError(f"Embedding store index not found: {root / 'index.parquet'}")
            index_stats = [path.stat() for path in index_files(root)]
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = CacheMetadata(**json.load(handle))
            if metadata.embedding_dim is None:
                raise ValueError(f"Embedding dimension is missing in {meta_path}")
            embedding_dim = int(metadata.embedding_dim)
            input_dim += embedding_dim
            cache_payload[modality] = {
                "root": str(root.resolve()),
                "key_column": self.key_columns[modality],
                "embedding_dim": embedding_dim,
                "meta_sha256": _file_sha256(meta_path),
                "index_size": sum(st.st_size for st in index_stats),
                "index_mtime_ns": max(st.st_mtime_ns for st in index_stats),
                "index_digests": _index_digests(root),
            }
        return cache_payload, input_dim

    def _materialized_cache_dir(self, fingerprint: str) -> Path:
        root = self.materialization_dir or (Path.cwd() / "materialized_features")
        threshold = self.threshold or "direct"
        return root / self.endpoint / self.split / str(threshold) / fingerprint[:16]

    def _validate_materialized_cache(
        self,
        cache_dir: Path,
        expected: dict[str, Any],
        *,
        policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        policy = str(policy or self.materialized_validation)
        if not cache_dir.exists():
            return False, "cache_dir_missing"
        if not (cache_dir / ".complete").exists():
            return False, "complete_marker_missing"
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            return False, "metadata_missing"
        try:
            actual = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            return False, f"metadata_invalid:{exc}"
        if actual.get("fingerprint") != expected["fingerprint"]:
            return False, "fingerprint_mismatch"
        if policy == "trust_complete":
            return True, "fast_hit"
        if policy not in {"metadata", "full"}:
            return False, f"invalid_validation_policy:{policy}"
        dtype = np.dtype(expected["feature_dtype"])
        for split_name, split_info in expected["splits"].items():
            x_path = cache_dir / f"{split_name}_X.npy"
            y_path = cache_dir / f"{split_name}_y.npy"
            if not x_path.exists() or not y_path.exists():
                return False, f"{split_name}_files_missing"
            try:
                x_array = np.load(x_path, mmap_mode="r")
                y_array = np.load(y_path, mmap_mode="r")
            except Exception as exc:
                return False, f"{split_name}_load_failed:{exc}"
            if tuple(x_array.shape) != tuple(split_info["x_shape"]):
                return False, f"{split_name}_x_shape_mismatch"
            if tuple(y_array.shape) != tuple(split_info["y_shape"]):
                return False, f"{split_name}_y_shape_mismatch"
            if np.dtype(x_array.dtype) != dtype:
                return False, f"{split_name}_x_dtype_mismatch"
            if np.dtype(y_array.dtype) != np.dtype(self.LABEL_DTYPE):
                return False, f"{split_name}_y_dtype_mismatch"
        return True, "fast_hit" if policy == "metadata" else "ok"

    def _materialized_info_from_metadata(
        self,
        cache_dir: Path,
        metadata: dict[str, Any],
        *,
        cache_status: str,
    ) -> dict[str, Any]:
        split_files = {}
        for split_name in metadata["splits"]:
            split_files[split_name] = {
                "x_path": str(cache_dir / f"{split_name}_X.npy"),
                "y_path": str(cache_dir / f"{split_name}_y.npy"),
                **metadata["splits"][split_name],
            }
        stored = json.loads((cache_dir / "metadata.json").read_text()) if (cache_dir / "metadata.json").exists() else {}
        return {
            **metadata,
            "missing_embeddings": stored.get("missing_embeddings"),
            "cache_dir": str(cache_dir),
            "cache_status": cache_status,
            "splits": split_files,
        }

    def _validate_required_columns(self) -> None:
        """Ensure all split parquets contain labels and modality lookup keys."""

        required_columns = {self.label_col}
        required_columns.update(spec.key_column for spec in self.modality_specs)
        for split_name, path in self.split_paths.items():
            schema_columns = set(pq.ParquetFile(path).schema_arrow.names)
            missing = required_columns - schema_columns
            if missing:
                raise ValueError(
                    f"{split_name} split {path} is missing required columns: {sorted(missing)}"
                )

    def _split_path(self, split: str) -> Path:
        """Return the parquet path for a normalized split name."""

        split = self._normalize_split_name(split)
        if not self.split_paths:
            self.prepare_data()
        return self.split_paths[split]

    @staticmethod
    def _normalize_split_name(split: str) -> str:
        """Normalize validation aliases and reject unknown split names."""

        split = "val" if split in {"valid", "validation"} else split
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split '{split}'")
        return split

    @staticmethod
    def _has_tvt(path: Path) -> bool:
        """Return True when a directory contains train, val, and test parquets."""

        return all((path / f"{split}.parquet").exists() for split in ("train", "val", "test"))
