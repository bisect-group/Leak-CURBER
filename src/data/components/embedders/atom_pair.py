from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.components.embedders.base import BaseShardEmbedder
from src.utils.chem_utils import ChemUtils


class SmilesAtomPairShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.n_bits = cfg.embeddings.atom_pair_nbits or 1024
        self.nb_workers = (
            cfg.embeddings.atom_pair_pandarallel.nb_workers
            if cfg.embeddings.atom_pair_pandarallel.nb_workers
            else 1
        )
        self.progress_bar = (
            cfg.embeddings.atom_pair_pandarallel.progress_bar
            if cfg.embeddings.atom_pair_pandarallel.progress_bar is not None
            else False
        )
        self.chem_utils = ChemUtils()

        super().__init__(
            cfg,
            input_path=cfg.embeddings.smiles_atom_pair_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.smiles_atom_pair_embeddings_log_file_name,
            embedder_name="atom_pair",
            model_name=f"nbits{self.n_bits}",
            version="v1",
            storage_dtype="uint8",
            key_field="smiles",
            key_type="canonical_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=250_000,
        )

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        from pandarallel import pandarallel

        pandarallel.initialize(
            progress_bar=self.progress_bar,
            nb_workers=self.nb_workers,
        )

        df = pd.DataFrame({"smiles": keys})
        self.logger.info(f"Generating atom-pair fingerprints for {len(df)} SMILES")
        df["fingerprint"] = df["smiles"].parallel_apply(
            lambda smiles: self.chem_utils.smiles_to_atom_pair_fp(
                smiles,
                n_bits=self.n_bits,
            )
        )

        arrays_by_key = {}
        failures = []
        for smiles, fingerprint in zip(df["smiles"], df["fingerprint"]):
            if fingerprint is None:
                failures.append(
                    {
                        "raw_key": smiles,
                        "canonical_key": smiles,
                        "error": "Fingerprint generation returned None",
                    }
                )
                continue

            arrays_by_key[smiles] = np.asarray(fingerprint, dtype=np.uint8)

        return arrays_by_key, failures
