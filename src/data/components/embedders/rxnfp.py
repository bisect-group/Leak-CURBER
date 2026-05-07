from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.base import BaseShardEmbedder


class RxnFPShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        from rxnfp.transformer_fingerprints import (
            RXNBERTFingerprintGenerator,
            get_default_model_and_tokenizer,
        )

        self.batch_size = cfg.embeddings.rxnfp_batch_size

        super().__init__(
            cfg,
            input_path=cfg.embeddings.rxnfp_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.rxnfp_embeddings_log_file_name,
            embedder_name="rxnfp",
            model_name="default",
            version="v1",
            storage_dtype="float16",
            key_field="rxn_smiles",
            key_type="rxn_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=100_000,
        )

        self.logger.info("Loading RXNFP model")
        model, tokenizer = get_default_model_and_tokenizer()
        self.rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        arrays_by_key = {}
        failures = []

        for start in tqdm(
            range(0, len(keys), self.batch_size),
            desc="Embedding RXNFP batches",
            leave=False,
        ):
            batch = keys[start : start + self.batch_size]
            try:
                batch_fps = self.rxnfp_generator.convert_batch(batch)
                for rxn_smiles, fp in zip(batch, batch_fps):
                    arrays_by_key[rxn_smiles] = np.asarray(fp, dtype=np.float16)
            except Exception as exc:
                failures.extend(
                    {
                        "raw_key": rxn_smiles,
                        "canonical_key": rxn_smiles,
                        "error": str(exc),
                    }
                    for rxn_smiles in batch
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return arrays_by_key, failures
