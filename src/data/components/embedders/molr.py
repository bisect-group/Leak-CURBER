from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.base import BaseShardEmbedder


class MolR2DShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.batch_size = cfg.embeddings.molr_batch_size

        super().__init__(
            cfg,
            input_path=cfg.embeddings.molr_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.molr_embeddings_log_file_name,
            embedder_name="molr",
            model_name="2d",
            version="v1",
            storage_dtype="float16",
            key_field="smiles",
            key_type="canonical_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=100_000,
        )

    @staticmethod
    def molr_worker_init():
        from MolR.featurizer import MolEFeaturizer

        global model
        model = MolEFeaturizer()

    @staticmethod
    def molr_worker(smiles_batch):
        try:
            embeddings, _flags = model.transform(smiles_batch)
            return {
                smile: np.asarray(embedding, dtype=np.float16)
                for smile, embedding in zip(smiles_batch, embeddings)
            }
        except Exception:
            return {}

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        chunks = [keys[i : i + self.batch_size] for i in range(0, len(keys), self.batch_size)]
        self.logger.info(
            f"Splitting SMILES into {len(chunks)} MolR batches of up to {self.batch_size}"
        )

        with Pool(
            processes=max(1, (os.cpu_count() or 1) - 1),
            initializer=self.molr_worker_init,
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.molr_worker, chunks),
                    total=len(chunks),
                    desc="MolR 2D embedding batches",
                    leave=False,
                )
            )

        arrays_by_key = {}
        for result in results:
            arrays_by_key.update(result)

        failures = [
            {
                "raw_key": key,
                "canonical_key": key,
                "error": "MolR worker returned no embedding",
            }
            for key in keys
            if key not in arrays_by_key
        ]
        return arrays_by_key, failures
