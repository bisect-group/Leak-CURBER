from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from src.utils.rich_progress import progress, write

from src.data.components.embedders.base import BaseShardEmbedder


def drfp_model_name(radius, min_radius, n_folded_length, rings, include_hydrogens) -> str:
    return (
        f"radius{radius}_min{min_radius}_len{n_folded_length}"
        f"_rings{int(rings)}_h{int(include_hydrogens)}"
    )


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

        self.input_dataset_parquet_path = (
            Path(cfg.embeddings.drfp_embeddings_input_dataset_parquet_file_path)
            if cfg.embeddings.get("drfp_embeddings_input_dataset_parquet_file_path")
            else None
        )

        super().__init__(
            cfg,
            input_path=(
                self.input_dataset_parquet_path
                if self.input_dataset_parquet_path is not None
                else cfg.embeddings.drfp_embeddings_input_pkl_path
            ),
            log_file_name=cfg.embeddings.drfp_embeddings_log_file_name,
            embedder_name="drfp",
            # The settings form the model name, as ECFP does, so a changed setting gets a new path.
            model_name=drfp_model_name(
                self.radius,
                self.min_radius,
                self.n_folded_length,
                self.rings,
                self.include_hydrogens,
            ),
            version="v1",
            storage_dtype="uint8",
            key_field="rxn_key",
            key_type="rxn_key",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=100_000,
        )

    def load_raw_keys(self) -> list[str]:
        if self.input_dataset_parquet_path is None:
            return super().load_raw_keys()

        df = pd.read_parquet(self.input_dataset_parquet_path, columns=[self.key_field])
        return df[self.key_field].dropna().astype(str).tolist()

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        arrays_by_key = {}
        failures = []

        for start in progress(
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
