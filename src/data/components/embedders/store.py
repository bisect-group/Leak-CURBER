from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import (
    atomic_write_json,
    sha256_short,
    utc_timestamp,
)


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
        self.pending_arrays_path = self.root_dir / "_pending_shard.npy"
        self.pending_meta_path = self.root_dir / "_pending_shard.json"
        self._pending_keys: list[str] = []
        self._pending_arrays: list[np.ndarray] = []
        self._pending_bytes = 0
        self._pending_since_checkpoint = 0
        self._last_pending_checkpoint_at = time.monotonic()

        self._write_or_validate_metadata()
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

    def put_many(
        self,
        arrays_by_key: dict[str, np.ndarray],
        *,
        flush: bool = False,
    ) -> int:
        if not arrays_by_key:
            if flush:
                return self.flush()
            return 0

        written = 0
        for canonical_key, raw_array in arrays_by_key.items():
            arr = self._prepare_array(raw_array)
            self._validate_embedding_dim(int(arr.shape[-1]))
            self._pending_keys.append(canonical_key)
            self._pending_arrays.append(arr)
            self._pending_bytes += arr.nbytes
            self._pending_since_checkpoint += 1

            if self._pending_bytes >= self.metadata.max_shard_bytes:
                written += self.flush(force=True)
            elif self._should_checkpoint_pending():
                self.checkpoint_pending(force=True)

        if flush:
            written += self.flush(force=True)
        else:
            self.checkpoint_pending()

        return written

    def flush(self, *, force: bool = True) -> int:
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

        self._append_rows(self.index_path, rows)
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
            raise ValueError(
                "Cannot reclaim terminal shard because canonical keys are unavailable "
                f"for {len(missing_hashes)} hashes. Examples: {sample}"
            )

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

    def _load_index(self) -> pd.DataFrame:
        if not self.index_path.exists():
            return pd.DataFrame(
                columns=[
                    "key_hash",
                    "shard_id",
                    "row_idx",
                ]
            )
        return pd.read_parquet(self.index_path)

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
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        appended_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)

    def _replace_index(self, index_df: pd.DataFrame) -> None:
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        index_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(self.index_path)

    def _remove_index_hashes(self, key_hashes: set[str]) -> None:
        if not key_hashes or not self.index_path.exists():
            return
        index_df = self._load_index()
        if index_df.empty:
            return
        filtered_df = index_df[~index_df["key_hash"].isin(key_hashes)].reset_index(
            drop=True
        )
        if len(filtered_df) != len(index_df):
            self._replace_index(filtered_df)

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
            updated = CacheMetadata(
                embedder=self.metadata.embedder,
                model_name=self.metadata.model_name,
                version=self.metadata.version,
                storage_dtype=self.metadata.storage_dtype,
                embedding_dim=dim,
                max_shard_bytes=self.metadata.max_shard_bytes,
                key_type=self.metadata.key_type,
                key_field=self.metadata.key_field,
            )
            self.metadata = updated
            atomic_write_json(self.meta_path, self.metadata.to_dict())
            return

        if self.metadata.embedding_dim != dim:
            raise ValueError(
                f"Embedding dim mismatch for {self.root_dir}: expected {self.metadata.embedding_dim}, got {dim}."
            )
