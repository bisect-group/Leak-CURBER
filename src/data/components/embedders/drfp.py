from __future__ import annotations

import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.base import BaseShardEmbedder


class DRFPShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        from drfp import DrfpEncoder

        self.batch_size = cfg.embeddings.drfp_batch_size
        self.n_folded_length = cfg.embeddings.drfp_n_folded_length
        self.min_radius = cfg.embeddings.drfp_min_radius
        self.radius = cfg.embeddings.drfp_radius
        self.rings = cfg.embeddings.drfp_rings
        self.include_hydrogens = cfg.embeddings.drfp_include_hydrogens
        self.drfp_encoder = DrfpEncoder

        super().__init__(
            cfg,
            input_path=cfg.embeddings.drfp_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.drfp_embeddings_log_file_name,
            embedder_name="drfp",
            model_name="default",
            version="v1",
            storage_dtype="uint8",
            key_field="rxn_smiles",
            key_type="rxn_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=100_000,
        )

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        arrays_by_key = {}
        failures = []

        for start in tqdm(
            range(0, len(keys), self.batch_size),
            desc="Embedding DRFP batches",
            leave=False,
        ):
            batch = keys[start : start + self.batch_size]

            try:
                batch_fps = self.drfp_encoder.encode(
                    batch,
                    n_folded_length=self.n_folded_length,
                    min_radius=self.min_radius,
                    radius=self.radius,
                    rings=self.rings,
                    include_hydrogens=self.include_hydrogens,
                    show_progress_bar=True,
                )
                for rxn_smiles, fp in zip(batch, batch_fps):
                    arrays_by_key[rxn_smiles] = np.asarray(fp, dtype=np.uint8)
            except Exception:
                # Fall back to per-reaction encoding so one invalid reaction does not fail the whole batch.
                for rxn_smiles in batch:
                    try:
                        fp = self.drfp_encoder.encode(
                            rxn_smiles,
                            n_folded_length=self.n_folded_length,
                            min_radius=self.min_radius,
                            radius=self.radius,
                            rings=self.rings,
                            include_hydrogens=self.include_hydrogens,
                            show_progress_bar=False,
                        )[0]
                        arrays_by_key[rxn_smiles] = np.asarray(fp, dtype=np.uint8)
                    except Exception as exc:
                        failures.append(
                            {
                                "raw_key": rxn_smiles,
                                "canonical_key": rxn_smiles,
                                "error": str(exc),
                            }
                        )

        return arrays_by_key, failures