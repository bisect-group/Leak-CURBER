from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import smi_ted
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm
from smi_ted import AutoModel, AutoTokenizer

from src.data.components.embedders.base import BaseShardEmbedder


class SMITEDShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.batch_size = cfg.embeddings.smited_embeddings_batch_size
        self.prefetch_batch_count = max(2, cfg.embeddings.smited_embeddings_prefetch_batches)
        self.tokenize_workers = max(1, cfg.embeddings.smited_embeddings_tokenize_workers)
        self.model_id = cfg.embeddings.smited_embeddings_model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_autocast = bool(
            torch.cuda.is_available() and cfg.embeddings.smited_embeddings_use_autocast
        )
        self.autocast_dtype = self._resolve_autocast_dtype(
            cfg.embeddings.smited_embeddings_autocast_dtype
        )
        self.cpu_workers = max(
            1,
            min(os.cpu_count() or 1, max(self.prefetch_batch_count, self.tokenize_workers)),
        )

        os.environ["TOKENIZERS_PARALLELISM"] = (
            "true" if cfg.embeddings.smited_embeddings_tokenizers_parallelism else "false"
        )

        super().__init__(
            cfg,
            input_path=cfg.embeddings.smited_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.smited_embeddings_log_file_name,
            embedder_name="smited",
            model_name="materials_smi_ted_fork",
            version="v1",
            storage_dtype="float16",
            key_field="smiles",
            key_type="canonical_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=50_000,
        )

        self.logger.info(f"Loading SMI-TED model from {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
        )
        self.model.smi_ted.tokenizer = self.tokenizer
        self.model.smi_ted.set_padding_idx_from_tokenizer()
        self.model.eval()
        self.model.to(self.device)
        self.encoder = self.model.smi_ted.encoder
        self.embedding_head = self.model.smi_ted.decoder.autoencoder.encoder
        self.max_length = int(self.model.smi_ted.max_len)
        self.logger.info(f"SMI-TED model loaded on {self.device}")
        self.logger.info(
            "SMI-TED batching enabled with "
            f"batch_size={self.batch_size}, "
            f"prefetch_batch_count={self.prefetch_batch_count}, "
            f"tokenize_workers={self.tokenize_workers}, "
            f"use_autocast={self.use_autocast}, "
            f"autocast_dtype={self.autocast_dtype}"
        )

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        arrays_by_key: dict[str, np.ndarray] = {}
        failures: list[dict] = []
        batches = [keys[start : start + self.batch_size] for start in range(0, len(keys), self.batch_size)]

        if not batches:
            return arrays_by_key, failures

        self.logger.info(
            f"Splitting SMILES into {len(batches)} SMI-TED batches of up to {self.batch_size}"
        )

        max_in_flight = min(len(batches), self.prefetch_batch_count)
        with ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
            pending = set()
            batch_iter = iter(enumerate(batches))

            for _ in range(max_in_flight):
                try:
                    batch_idx, batch = next(batch_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(self._prepare_batch, batch_idx, batch))

            progress = tqdm(total=len(batches), desc="SMI-TED embedding batches", leave=False)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    _batch_idx, batch, tokenized, batch_failures = future.result()
                    failures.extend(batch_failures)

                    if batch:
                        batch_embeddings, encode_failures = self._encode_batch(batch, tokenized)
                        failures.extend(encode_failures)
                        arrays_by_key.update(batch_embeddings)

                    progress.update(1)

                    try:
                        next_batch_idx, next_batch = next(batch_iter)
                    except StopIteration:
                        continue
                    pending.add(
                        executor.submit(self._prepare_batch, next_batch_idx, next_batch)
                    )

            progress.close()

        return arrays_by_key, failures

    def _prepare_batch(
        self,
        batch_idx: int,
        batch: list[str],
    ) -> tuple[int, list[str], dict[str, torch.Tensor], list[dict]]:
        valid_smiles = []
        failures = []

        for smiles in batch:
            if not smiles:
                failures.append(
                    {
                        "raw_key": smiles,
                        "canonical_key": smiles,
                        "error": "SMILES string is empty",
                    }
                )
                continue
            valid_smiles.append(smiles)

        if not valid_smiles:
            return batch_idx, [], {}, failures

        tokenized = self.tokenizer(
            valid_smiles,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        if "token_type_ids" in tokenized:
            del tokenized["token_type_ids"]

        if self.device == "cuda":
            tokenized = {
                key: value.pin_memory()
                for key, value in tokenized.items()
            }

        return batch_idx, valid_smiles, tokenized, failures

    def _encode_batch(
        self,
        smiles_batch: list[str],
        tokenized: dict[str, torch.Tensor],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        try:
            tokenized = {
                key: value.to(self.device, non_blocking=self.device == "cuda")
                for key, value in tokenized.items()
            }
            with torch.inference_mode():
                if self.use_autocast:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.autocast_dtype,
                    ):
                        token_embeddings = self.encoder(
                            tokenized["input_ids"],
                            tokenized["attention_mask"],
                        )
                        batch_embeddings = self.embedding_head(
                            token_embeddings.view(-1, self.max_length * self.model.smi_ted.n_embd)
                        )
                else:
                    token_embeddings = self.encoder(
                        tokenized["input_ids"],
                        tokenized["attention_mask"],
                    )
                    batch_embeddings = self.embedding_head(
                        token_embeddings.view(-1, self.max_length * self.model.smi_ted.n_embd)
                    )
            batch_embeddings_cpu = (
                batch_embeddings.detach().to(dtype=torch.float16).cpu().numpy()
            )
            return {
                smiles: batch_embeddings_cpu[idx]
                for idx, smiles in enumerate(smiles_batch)
            }, []
        except Exception as exc:
            if len(smiles_batch) == 1:
                smiles = smiles_batch[0]
                return {}, [
                    {
                        "raw_key": smiles,
                        "canonical_key": smiles,
                        "error": str(exc),
                    }
                ]

            midpoint = len(smiles_batch) // 2
            left_tokenized = {
                key: value[:midpoint].contiguous()
                for key, value in tokenized.items()
            }
            right_tokenized = {
                key: value[midpoint:].contiguous()
                for key, value in tokenized.items()
            }
            left_embeddings, left_failures = self._encode_batch(
                smiles_batch[:midpoint],
                left_tokenized,
            )
            right_embeddings, right_failures = self._encode_batch(
                smiles_batch[midpoint:],
                right_tokenized,
            )
            left_embeddings.update(right_embeddings)
            return left_embeddings, left_failures + right_failures

    @staticmethod
    def _resolve_autocast_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float16":
            return torch.float16
        raise ValueError(
            "smited_embeddings_autocast_dtype must be one of: 'float16', 'bfloat16'"
        )
