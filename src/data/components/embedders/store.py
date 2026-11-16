from __future__ import annotations

import json
import time
import warnings
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import (
    atomic_write_json,
    key_hash_candidates,
    sha256_short,
    utc_timestamp,
)


def pending_shard_rows(root_dir: Path) -> int:
    """Return the row count of an unflushed pending shard (0 if none)."""
    path = Path(root_dir) / "_pending_shard.json"
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return len(json.load(handle).get("keys", []))


def index_files(root_dir: Path | str) -> list[Path]:
    """The index files of a store that exist: index.parquet, then its unmerged parts."""
    root = Path(root_dir)
    base = root / "index.parquet"
    return [*([base] if base.exists() else []), *sorted((root / "index_parts").glob("part_*.parquet"))]


def load_index(
    root_dir: Path | str, files: list[Path] | None = None, columns=None, keep: str = "first"
) -> pd.DataFrame:
    """Read a store's index from index.parquet plus unmerged parts; keep the first row per key.

    A flush writes a part file, not index.parquet, so every reader of an
    index (and every existence check) must use this loader or `index_files`.
    """
    files = index_files(root_dir) if files is None else files
    if not files:
        return pd.DataFrame(columns=columns or ["key_hash", "shard_id", "row_idx"])
    index_df = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
    return index_df.drop_duplicates("key_hash", keep=keep, ignore_index=True)


class ShardedEmbeddingStore:
    def __init__(
        self,
        root_dir: Path,
        metadata: CacheMetadata,
        logger,
        *,
        pending_checkpoint_interval_embeddings: int = 5_000,
        pending_checkpoint_interval_seconds: int = 300,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.metadata = metadata
        self.logger = logger
        self.pending_checkpoint_interval_embeddings = int(
            pending_checkpoint_interval_embeddings
        )
        self.pending_checkpoint_interval_seconds = int(
            pending_checkpoint_interval_seconds
        )

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root_dir / "meta.json"
        self.index_path = self.root_dir / "index.parquet"
        self.failures_path = self.root_dir / "failures.parquet"
        self.index_parts_dir = self.root_dir / "index_parts"
        self.pending_arrays_path = self.root_dir / "_pending_shard.npy"
        self.pending_meta_path = self.root_dir / "_pending_shard.json"
        self._pending_keys: list[str] = []
        self._pending_arrays: list[np.ndarray] = []
        self._pending_bytes = 0
        self._pending_since_checkpoint = 0
        self._last_pending_checkpoint_at = time.monotonic()

        self._write_or_validate_metadata()
        self._compact_index()  # merge parts left by a run that stopped before its final flush
        self._load_pending_checkpoint()

    def get_missing_keys(self, canonical_keys: list[str]) -> list[str]:
        if not canonical_keys:
            return []

        index_df = self._load_index()
        existing_hashes = set()
        if index_df.empty:
            existing_hashes.update(self._pending_key_hashes())
        else:
            existing_hashes.update(index_df["key_hash"])
            existing_hashes.update(self._pending_key_hashes())

        return [
            key
            for key in canonical_keys
            if sha256_short(key) not in existing_hashes
        ]

    def drop_keys(self, canonical_keys: list[str]) -> int:
        """Remove keys from the index so get_missing_keys reports them again.

        Shard rows stay on disk as orphans, and a later put_many writes new rows.
        Returns the number of keys removed. Do not run while another process writes the store.
        """
        before = len(self._load_index())
        self._remove_index_hashes({sha256_short(key) for key in canonical_keys})
        return before - len(self._load_index())

    def put_many(
        self,
        arrays_by_key: dict[str, np.ndarray],
        *,
        flush: bool = False,
    ) -> int:
        if not arrays_by_key:
            return self.flush() if flush else 0

        written = 0
        for canonical_key, raw_array in arrays_by_key.items():
            arr = self._prepare_array(raw_array)
            self._validate_embedding_dim(int(arr.shape[-1]))
            self._pending_keys.append(canonical_key)
            self._pending_arrays.append(arr)
            self._pending_bytes += arr.nbytes
            self._pending_since_checkpoint += 1

            if self._pending_bytes >= self.metadata.max_shard_bytes:
                written += self._flush_shard(force=True)
            elif self._should_checkpoint_pending():
                self.checkpoint_pending(force=True)

        if flush:
            written += self.flush(force=True)
        else:
            self.checkpoint_pending()

        return written

    def flush(self, *, force: bool = True) -> int:
        """Write the pending shard and merge index parts, so index.parquet is complete."""
        written = self._flush_shard(force=force)
        self._compact_index()
        return written

    def _flush_shard(self, *, force: bool = True) -> int:
        if not self._pending_keys:
            return 0
        if not force and self._pending_bytes < self.metadata.max_shard_bytes:
            self.checkpoint_pending(force=True)
            return 0

        batch = np.stack(self._pending_arrays, axis=0)
        shard_id = self._next_shard_id()
        shard_path = self.root_dir / f"shard_{shard_id:06d}.npy"
        tmp_shard_path = shard_path.with_suffix(".npy.tmp")
        with open(tmp_shard_path, "wb") as handle:
            np.save(handle, batch, allow_pickle=False)
        tmp_shard_path.replace(shard_path)

        rows = []
        for row_idx, canonical_key in enumerate(self._pending_keys):
            rows.append(
                {
                    "key_hash": sha256_short(canonical_key),
                    "shard_id": shard_id,
                    "row_idx": row_idx,
                }
            )

        self._write_index_part(rows)
        written = len(rows)
        approx_mb = batch.nbytes / (1024 * 1024)
        self.logger.info(
            f"Wrote shard_{shard_id:06d}.npy with {written} embeddings ({approx_mb:.1f} MiB)"
        )

        self._pending_keys.clear()
        self._pending_arrays.clear()
        self._pending_bytes = 0
        self._pending_since_checkpoint = 0
        self._remove_pending_checkpoint()
        return written

    def checkpoint_pending(self, *, force: bool = False) -> None:
        if not self._pending_keys:
            self._remove_pending_checkpoint()
            return
        if not force and not self._should_checkpoint_pending():
            return

        batch = np.stack(self._pending_arrays, axis=0)
        tmp_arrays_path = self.pending_arrays_path.with_suffix(".npy.tmp")
        with open(tmp_arrays_path, "wb") as handle:
            np.save(handle, batch, allow_pickle=False)
        tmp_arrays_path.replace(self.pending_arrays_path)

        atomic_write_json(
            self.pending_meta_path,
            {
                "keys": self._pending_keys,
                "bytes": self._pending_bytes,
                "count": len(self._pending_keys),
                "updated_at": utc_timestamp(),
            },
        )
        self._pending_since_checkpoint = 0
        self._last_pending_checkpoint_at = time.monotonic()

    def reclaim_last_underfilled_shard(self, canonical_keys: list[str]) -> int:
        if self._pending_keys:
            self.logger.info(
                "Pending shard checkpoint already exists; skipping terminal shard reclaim."
            )
            return 0

        index_df = self._load_index()
        if index_df.empty:
            return 0

        shard_id = int(index_df["shard_id"].max())
        shard_rows = index_df[index_df["shard_id"] == shard_id].sort_values("row_idx")
        shard_path = self.root_dir / f"shard_{shard_id:06d}.npy"
        if not shard_path.exists():
            self.logger.warning(
                f"Cannot reclaim missing terminal shard referenced by index: {shard_path}"
            )
            return 0

        batch = np.load(shard_path, allow_pickle=False)
        if batch.nbytes >= self.metadata.max_shard_bytes:
            return 0
        if len(shard_rows) != batch.shape[0]:
            raise ValueError(
                f"Cannot reclaim {shard_path}: index rows ({len(shard_rows)}) "
                f"do not match shard rows ({batch.shape[0]})."
            )

        key_by_hash = {sha256_short(key): key for key in canonical_keys}
        missing_hashes = sorted(set(shard_rows["key_hash"]) - set(key_by_hash))
        if missing_hashes:
            sample = ", ".join(missing_hashes[:5])
            self.logger.warning(
                "Skipping terminal shard reclaim for shard_%06d.npy: %d of %d shard "
                "hashes are unavailable from the current input keys. Examples: %s. "
                "Missing-key detection will continue; if a strict full-input reclaim is "
                "needed, rerun with an input containing every terminal-shard key. "
                "Temporary workaround: set embeddings.esm3_reclaim_last_underfilled_shard=false.",
                shard_id,
                len(missing_hashes),
                len(shard_rows),
                sample,
            )
            return 0

        self._pending_keys = [key_by_hash[key_hash] for key_hash in shard_rows["key_hash"]]
        self._pending_arrays = [
            np.asarray(batch[int(row_idx)], dtype=self.metadata.storage_dtype)
            for row_idx in shard_rows["row_idx"]
        ]
        self._pending_bytes = int(sum(arr.nbytes for arr in self._pending_arrays))
        self._pending_since_checkpoint = len(self._pending_keys)
        self.checkpoint_pending(force=True)

        self._remove_index_hashes(set(shard_rows["key_hash"]))
        shard_path.unlink()
        self.logger.info(
            f"Reclaimed underfilled shard_{shard_id:06d}.npy with "
            f"{len(self._pending_keys)} embeddings into pending state"
        )
        return len(self._pending_keys)

    def record_failures(
        self,
        failures: list[dict],
    ) -> int:
        if not failures:
            return 0

        stamped_failures = []
        created_at = utc_timestamp()
        for failure in failures:
            stamped_failures.append(
                {
                    "key_hash": (
                        sha256_short(failure["canonical_key"])
                        if failure.get("canonical_key")
                        else None
                    ),
                    "raw_key": failure.get("raw_key"),
                    "canonical_key": failure.get("canonical_key"),
                    "error": failure.get("error"),
                    "created_at": created_at,
                }
            )

        self._append_rows(self.failures_path, stamped_failures)
        self.logger.info(f"Recorded {len(stamped_failures)} failed items")
        return len(stamped_failures)

    def _prepare_array(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=self.metadata.storage_dtype)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D embedding array, got shape {arr.shape}")
        return arr

    def _index_parts(self) -> list[Path]:
        return sorted(self.index_parts_dir.glob("part_*.parquet"))

    def _load_index(self) -> pd.DataFrame:
        return load_index(self.root_dir)

    def _compact_index(self) -> None:
        # A flush writes a small part file; parts are merged into index.parquet at the end of a run or on open.
        # A crash between the merge and the part delete leaves duplicates; the loader drops them.
        if not self._index_parts():
            return
        self._replace_index()

    def _write_index_part(self, rows: list[dict]) -> None:
        self.index_parts_dir.mkdir(exist_ok=True)
        path = self.index_parts_dir / f"part_{rows[0]['shard_id']:06d}.parquet"
        tmp_path = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(
            tmp_path,
            index=False,
            compression="zstd",
            compression_level=3,
        )
        tmp_path.replace(path)
        if len(self._index_parts()) >= 64:
            self._compact_index()

    def _load_pending_checkpoint(self) -> None:
        if not self.pending_meta_path.exists() or not self.pending_arrays_path.exists():
            self._remove_pending_checkpoint()
            return

        import json

        with open(self.pending_meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        keys = [str(key) for key in metadata.get("keys", [])]
        batch = np.load(self.pending_arrays_path, allow_pickle=False)
        if batch.ndim != 2:
            raise ValueError(
                f"Pending shard at {self.pending_arrays_path} must be 2-D, got {batch.shape}."
            )
        if len(keys) != batch.shape[0]:
            raise ValueError(
                f"Pending shard key count ({len(keys)}) does not match rows ({batch.shape[0]})."
            )

        self._pending_keys = keys
        self._pending_arrays = [
            np.asarray(batch[row_idx], dtype=self.metadata.storage_dtype)
            for row_idx in range(batch.shape[0])
        ]
        self._pending_bytes = int(sum(arr.nbytes for arr in self._pending_arrays))
        self._pending_since_checkpoint = 0
        self._last_pending_checkpoint_at = time.monotonic()
        self._remove_index_hashes(self._pending_key_hashes())
        self.logger.info(
            f"Loaded pending shard checkpoint with {len(self._pending_keys)} embeddings "
            f"({self._pending_bytes / (1024 * 1024):.1f} MiB)"
        )

    def _append_rows(self, path: Path, rows: list[dict]) -> None:
        existing_df = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        appended_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
        if path == self.index_path:
            # The reader keeps the first location of a key, so keep the first row per key_hash.
            appended_df = appended_df.drop_duplicates("key_hash", keep="first", ignore_index=True)
        elif path == self.failures_path:
            # Keep the latest row per failed key.
            appended_df = appended_df.drop_duplicates(
                ["key_hash", "raw_key"], keep="last", ignore_index=True
            )
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        appended_df.to_parquet(tmp_path, index=False, compression="zstd", compression_level=3)
        tmp_path.replace(path)

    def _replace_index(self, drop_hashes: frozenset[str] = frozenset()) -> None:
        # Merge index.parquet and the parts read now; delete only those parts, so a part
        # written meanwhile (or by another process) is kept.
        files = index_files(self.root_dir)
        parts = [path for path in files if path != self.index_path]
        index_df = load_index(self.root_dir, files)
        index_df = index_df[~index_df["key_hash"].isin(drop_hashes)].reset_index(drop=True)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        index_df.to_parquet(tmp_path, index=False, compression="zstd", compression_level=3)
        tmp_path.replace(self.index_path)
        for part in parts:
            part.unlink()

    def _remove_index_hashes(self, key_hashes: set[str]) -> None:
        if not key_hashes:
            return
        index_df = self._load_index()
        if index_df["key_hash"].isin(key_hashes).any():
            self._replace_index(frozenset(key_hashes))

    def _next_shard_id(self) -> int:
        shard_ids = []
        for shard_path in self.root_dir.glob("shard_*.npy"):
            try:
                shard_ids.append(int(shard_path.stem.split("_")[-1]))
            except ValueError:
                continue
        return max(shard_ids, default=-1) + 1

    def _pending_key_hashes(self) -> set[str]:
        return {sha256_short(key) for key in self._pending_keys}

    def _should_checkpoint_pending(self) -> bool:
        if not self._pending_keys:
            return False
        if (
            self.pending_checkpoint_interval_embeddings > 0
            and self._pending_since_checkpoint
            >= self.pending_checkpoint_interval_embeddings
        ):
            return True
        return (
            self.pending_checkpoint_interval_seconds > 0
            and time.monotonic() - self._last_pending_checkpoint_at
            >= self.pending_checkpoint_interval_seconds
        )

    def _remove_pending_checkpoint(self) -> None:
        for path in (self.pending_arrays_path, self.pending_meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _write_or_validate_metadata(self) -> None:
        if not self.meta_path.exists():
            atomic_write_json(self.meta_path, self.metadata.to_dict())
            return

        existing = self.meta_path.read_text()
        expected = self.metadata.to_dict()
        if existing.strip() != "":
            import json

            current = json.loads(existing)
            if current != expected:
                expected_with_existing_dim = dict(expected)
                expected_with_existing_dim["embedding_dim"] = current.get("embedding_dim")
                if current == expected_with_existing_dim:
                    self.metadata = CacheMetadata(**current)
                    return
                raise ValueError(
                    f"Existing cache metadata at {self.meta_path} does not match expected metadata."
                )

    def _validate_embedding_dim(self, dim: int) -> None:
        if self.metadata.embedding_dim is None:
            self.metadata = replace(self.metadata, embedding_dim=dim)
            atomic_write_json(self.meta_path, self.metadata.to_dict())
            return

        if self.metadata.embedding_dim != dim:
            raise ValueError(
                f"Embedding dim mismatch for {self.root_dir}: expected {self.metadata.embedding_dim}, got {dim}."
            )


class ShardedEmbeddingReader:
    """Read-only lookup API for sharded embedding stores.

    Stores are expected to contain ``meta.json``, ``index.parquet``, and
    ``shard_*.npy`` files written by :class:`ShardedEmbeddingStore`.
    """

    def __init__(self, root_dir: Path | str, *, shard_cache_size: int = 4) -> None:
        self.root_dir = Path(root_dir)
        self.meta_path = self.root_dir / "meta.json"
        self.index_path = self.root_dir / "index.parquet"
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shard_cache: OrderedDict[int, np.ndarray] = OrderedDict()

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Embedding store directory not found: {self.root_dir}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Embedding store metadata not found: {self.meta_path}")
        if not index_files(self.root_dir):
            raise FileNotFoundError(f"Embedding store index not found: {self.index_path}")

        with open(self.meta_path, "r", encoding="utf-8") as handle:
            self.metadata = CacheMetadata(**json.load(handle))
        if self.metadata.embedding_dim is None:
            raise ValueError(f"Embedding dimension is missing in {self.meta_path}")

        # Keys in _pending_shard.* are not in index.parquet, so this reader returns None for them.
        # Warn when such rows exist. The writers flush the pending shard at the end of a run.
        if pending := pending_shard_rows(self.root_dir):
            warnings.warn(
                f"Embedding store {self.root_dir} has {pending} unflushed pending rows "
                "that this reader ignores; rerun the embedder to flush them.",
                stacklevel=2,
            )

        index_df = load_index(self.root_dir)
        required_columns = {"key_hash", "shard_id", "row_idx"}
        missing_columns = required_columns - set(index_df.columns)
        if missing_columns:
            raise ValueError(
                f"Embedding index {self.index_path} is missing columns: {sorted(missing_columns)}"
            )

        self._locations: dict[str, tuple[int, int]] = {}
        for row in index_df.itertuples(index=False):
            key_hash = str(row.key_hash)
            if key_hash not in self._locations:
                self._locations[key_hash] = (int(row.shard_id), int(row.row_idx))

        for shard_id in sorted({location[0] for location in self._locations.values()}):
            shard_path = self._shard_path(shard_id)
            if not shard_path.exists():
                raise FileNotFoundError(
                    f"Embedding shard referenced by {self.index_path} not found: {shard_path}"
                )

    @property
    def embedding_dim(self) -> int:
        return int(self.metadata.embedding_dim)

    def contains(self, key: str) -> bool:
        return sha256_short(str(key)) in self._locations

    def get(self, key: str) -> np.ndarray | None:
        location = None
        for key_hash in key_hash_candidates(str(key), self.metadata.key_type):
            if (location := self._locations.get(key_hash)) is not None:
                break
        if location is None:
            return None

        shard_id, row_idx = location
        shard = self._load_shard(shard_id)
        if row_idx >= shard.shape[0]:
            raise IndexError(
                f"Index row {row_idx} is out of bounds for {self._shard_path(shard_id)} "
                f"with {shard.shape[0]} rows."
            )
        return np.asarray(shard[row_idx])

    def get_many(self, keys: list[str]) -> list[np.ndarray | None]:
        return [self.get(key) for key in keys]

    def _shard_path(self, shard_id: int) -> Path:
        return self.root_dir / f"shard_{int(shard_id):06d}.npy"

    def _load_shard(self, shard_id: int) -> np.ndarray:
        if shard_id in self._shard_cache:
            shard = self._shard_cache.pop(shard_id)
            self._shard_cache[shard_id] = shard
            return shard

        shard_path = self._shard_path(shard_id)
        if not shard_path.exists():
            raise FileNotFoundError(f"Embedding shard not found: {shard_path}")
        shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        self._shard_cache[shard_id] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard
