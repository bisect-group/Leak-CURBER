from __future__ import annotations

import os
import queue
import threading

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.base import BaseShardEmbedder


class ESMCShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.gpu_ids = list(cfg.embeddings.gpu_ids)
        self.max_batch_tokens = 16_384
        self.max_batch_size = 64
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        super().__init__(
            cfg,
            input_path=cfg.embeddings.esmc_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.esmc_embeddings_log_file_name,
            embedder_name="esmc",
            model_name="esmc_600m",
            version="v1",
            storage_dtype="float16",
            key_field="sequence",
            key_type="protein_sequence",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=20_000,
        )

        self.logger.info(f"Using GPUs: {self.gpu_ids}")
        self.logger.info(
            f"ESMC batching enabled with max_batch_tokens={self.max_batch_tokens}, max_batch_size={self.max_batch_size}"
        )

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available for ESMC embedding.")

        sequences_chunks = [keys[i::num_gpus] for i in range(num_gpus)]
        total_batches = sum(
            len(self._make_length_buckets(sorted(chunk, key=len)))
            for chunk in sequences_chunks
            if chunk
        )
        manager = mp.Manager()
        return_dict = manager.dict()
        failure_dict = manager.dict()
        progress_queue = mp.Queue()
        stop_event = threading.Event()
        processes = []
        progress_thread = threading.Thread(
            target=self._consume_progress_updates,
            args=(progress_queue, total_batches, "ESMC total batches", stop_event),
            daemon=True,
        )
        progress_thread.start()

        try:
            for gpu_id in range(num_gpus):
                process = mp.Process(
                    target=self._process_chunk,
                    args=(
                        gpu_id,
                        sequences_chunks[gpu_id],
                        return_dict,
                        failure_dict,
                        progress_queue,
                    ),
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            embeddings: dict[str, np.ndarray] = {}
            failures: list[dict] = []
            for gpu_id in range(num_gpus):
                embeddings.update(return_dict.get(gpu_id, {}))
                failures.extend(failure_dict.get(gpu_id, []))

            return embeddings, failures
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            stop_event.set()
            progress_thread.join()
            progress_queue.close()

    def _process_chunk(
        self,
        gpu_id: int,
        sequences_chunk: list[str],
        return_dict,
        failure_dict,
        progress_queue,
    ) -> None:
        from esm.models.esmc import ESMC
        from esm.sdk.api import LogitsConfig
        from esm.utils.sampling import _BatchedESMProteinTensor

        if not sequences_chunk:
            return_dict[gpu_id] = {}
            failure_dict[gpu_id] = []
            return

        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        client = ESMC.from_pretrained("esmc_600m").to(f"cuda:{gpu_id}")

        embeddings = {}
        failures = []
        sorted_sequences = sorted(sequences_chunk, key=len)
        batches = self._make_length_buckets(sorted_sequences)

        for batch_sequences in tqdm(
            batches,
            position=gpu_id,
            desc=f"GPU {gpu_id} ESMC batches",
            leave=False,
        ):
            try:
                sequence_tokens = client._tokenize(batch_sequences)
                batched = _BatchedESMProteinTensor(sequence=sequence_tokens)
                logits_output = client.logits(
                    batched,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                assert logits_output.embeddings is not None

                for idx, seq in enumerate(batch_sequences):
                    seq_len = len(seq) + 2
                    mean_embedding = (
                        logits_output.embeddings[idx, :seq_len]
                        .mean(dim=0)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float16, copy=False)
                    )
                    embeddings[seq] = mean_embedding
            except Exception as exc:
                failures.extend(
                    {
                        "raw_key": seq,
                        "canonical_key": seq,
                        "error": str(exc),
                    }
                    for seq in batch_sequences
                )
                torch.cuda.empty_cache()
            finally:
                progress_queue.put(1)

        return_dict[gpu_id] = embeddings
        failure_dict[gpu_id] = failures

    def _make_length_buckets(self, sequences: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for seq in sequences:
            seq_tokens = len(seq) + 2
            if current_batch and (
                len(current_batch) >= self.max_batch_size
                or current_tokens + seq_tokens > self.max_batch_tokens
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(seq)
            current_tokens += seq_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    @staticmethod
    def _consume_progress_updates(progress_queue, total: int, desc: str, stop_event) -> None:
        if total <= 0:
            return

        with tqdm(total=total, desc=desc, leave=True, position=0) as progress_bar:
            while True:
                try:
                    increment = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                progress_bar.update(int(increment))
