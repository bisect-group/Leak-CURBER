from __future__ import annotations

import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.base import BaseShardEmbedder


class ClampSmilesShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        self.batch_size = cfg.embeddings.clamp_batch_size
        self.clamp_hp_url = cfg.embeddings.clamp_hp_url
        self.clamp_checkpoint_url = cfg.embeddings.clamp_checkpoint_url
        self.clamp_dir = Path(cfg.embeddings.clamp_dir)
        self.clamp_dir.mkdir(parents=True, exist_ok=True)

        self._download_if_needed(self.clamp_hp_url, self.clamp_dir / "hp.json")
        self._download_if_needed(
            self.clamp_checkpoint_url,
            self.clamp_dir / "checkpoint.pt",
        )

        super().__init__(
            cfg,
            input_path=cfg.embeddings.clamp_smiles_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.clamp_smiles_embeddings_log_file_name,
            embedder_name="clamp",
            model_name="smiles",
            version="v1",
            storage_dtype="float16",
            key_field="smiles",
            key_type="canonical_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=100_000,
        )

    def _download_if_needed(self, url: str, dest: Path) -> None:
        if dest.exists():
            return
        subprocess.run(["curl", "-Lo", str(dest), str(url)], check=True)

    @staticmethod
    def clamp_worker_init(clamp_dir: str):
        import clamp
        import logging

        try:
            log = logging.getLogger("clamp")
            log.setLevel(logging.WARNING)
            log.propagate = False
            for handler in list(log.handlers):
                log.removeHandler(handler)
        except Exception:
            pass

        global model
        model = clamp.CLAMP(device="cpu", path_dir=clamp_dir)
        model.eval()

    @staticmethod
    def clamp_worker(smiles_batch):
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")

        try:
            with torch.no_grad():
                compound_features = model.prepro_smiles(smiles_batch)
                batch_emb = model.compound_encoder(compound_features)
            return {
                smiles: emb.detach().cpu().numpy().astype(np.float16, copy=False)
                for smiles, emb in zip(smiles_batch, batch_emb)
            }
        except Exception:
            return {}

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        chunks = [keys[i : i + self.batch_size] for i in range(0, len(keys), self.batch_size)]
        self.logger.info(
            f"Splitting SMILES into {len(chunks)} CLAMP batches of up to {self.batch_size}"
        )

        with Pool(
            processes=max(1, (os.cpu_count() or 1) - 1),
            initializer=self.clamp_worker_init,
            initargs=(str(self.clamp_dir),),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.clamp_worker, chunks),
                    total=len(chunks),
                    desc="CLAMP embedding batches",
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
                "error": "CLAMP worker returned no embedding",
            }
            for key in keys
            if key not in arrays_by_key
        ]
        return arrays_by_key, failures
