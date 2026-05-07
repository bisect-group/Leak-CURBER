from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.store import ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import load_pickle_items
from src.utils.tqdmlogger import TqdmLogger


class BaseShardEmbedder(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        *,
        input_path: Path,
        log_file_name: str,
        embedder_name: str,
        model_name: str,
        version: str,
        storage_dtype: str,
        key_field: str,
        key_type: str,
        max_shard_bytes: int,
        compute_chunk_size: int,
        pending_checkpoint_interval_embeddings: int = 5_000,
        pending_checkpoint_interval_seconds: int = 300,
    ) -> None:
        self.cfg = cfg
        self.input_path = Path(input_path)
        self.key_field = key_field
        self.compute_chunk_size = compute_chunk_size

        log_dir = Path(cfg.embeddings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TqdmLogger(
            log_dir=log_dir,
            log_file_name=log_file_name,
        ).get_logger()

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        cache_root = Path(cfg.embeddings.embeddings_path)
        store_root = cache_root / embedder_name / model_name / version
        metadata = CacheMetadata(
            embedder=embedder_name,
            model_name=model_name,
            version=version,
            storage_dtype=storage_dtype,
            embedding_dim=None,
            max_shard_bytes=max_shard_bytes,
            key_type=key_type,
            key_field=key_field,
        )
        self.store = ShardedEmbeddingStore(
            store_root,
            metadata,
            self.logger,
            pending_checkpoint_interval_embeddings=pending_checkpoint_interval_embeddings,
            pending_checkpoint_interval_seconds=pending_checkpoint_interval_seconds,
        )

    def embed(self) -> None:
        raw_keys = load_pickle_items(self.input_path, self.key_field)
        self.logger.info(f"Loaded {len(raw_keys)} raw items from {self.input_path}")

        unique_keys: list[str] = []
        seen_keys: set[str] = set()
        failures: list[dict] = []

        for raw_key in tqdm(raw_keys, desc="Preparing input keys", leave=False):
            if raw_key is None:
                failures.append(
                    {
                        "raw_key": raw_key,
                        "canonical_key": None,
                        "error": "Input key is None",
                    }
                )
                continue

            key = str(raw_key)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_keys.append(key)

        self.logger.info(
            f"{len(unique_keys)} unique items after deduplication"
        )

        missing_keys = self.store.get_missing_keys(unique_keys)
        self.logger.info(f"{len(missing_keys)} items missing from cache")

        total_written = 0
        if missing_keys:
            for start in tqdm(
                range(0, len(missing_keys), self.compute_chunk_size),
                desc="Computing cache chunks",
                leave=False,
            ):
                chunk_keys = missing_keys[start : start + self.compute_chunk_size]
                arrays_by_key, compute_failures = self.compute_many(chunk_keys)
                total_written += self.store.put_many(
                    arrays_by_key,
                )
                failures.extend(compute_failures)
            total_written += self.store.put_many({}, flush=True)
        else:
            self.logger.info("No new items to compute")

        self.store.record_failures(failures)
        if total_written:
            self.logger.info(f"Persisted {total_written} embeddings across shard writes")

    @abstractmethod
    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        raise NotImplementedError
