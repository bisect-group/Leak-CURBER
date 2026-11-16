from __future__ import annotations

import os
import queue
import sys
import threading
import types
from importlib.machinery import ModuleSpec
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig
from src.utils.rich_progress import progress, write

from src.data.components.embedders.base import BaseShardEmbedder
from src.data.components.embedders.utils import load_pickle_items


def _ensure_transformers_vision_imports() -> None:
    try:
        from torchvision.transforms import InterpolationMode  # noqa: F401
    except Exception:
        class _InterpolationMode:
            NEAREST_EXACT = "nearest_exact"
            BOX = "box"
            BILINEAR = "bilinear"
            HAMMING = "hamming"
            BICUBIC = "bicubic"
            LANCZOS = "lanczos"

        torchvision = types.ModuleType("torchvision")
        torchvision.__spec__ = ModuleSpec("torchvision", None)
        transforms = types.ModuleType("torchvision.transforms")
        transforms.__spec__ = ModuleSpec("torchvision.transforms", None)
        transforms.InterpolationMode = _InterpolationMode
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms


def _torch_dtype_from_name(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None or str(dtype_name).lower() in {"", "none", "null"}:
        return None
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported ESMC torch dtype: {dtype_name}")


def _extract_final_hidden_state(output: Any) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden

    hidden_states = getattr(output, "hidden_states", None)
    if isinstance(hidden_states, tuple):
        return hidden_states[-1]
    if torch.is_tensor(hidden_states):
        return hidden_states

    if isinstance(output, dict):
        hidden = output.get("last_hidden_state")
        if hidden is not None:
            return hidden
        hidden_states = output.get("hidden_states")
        if isinstance(hidden_states, tuple):
            return hidden_states[-1]
        if torch.is_tensor(hidden_states):
            return hidden_states

    raise RuntimeError(
        "ESMC output did not contain final hidden states. "
        "Expected 'last_hidden_state' or 'hidden_states'."
    )


def _mean_pool_residues(
    hidden_states: torch.Tensor,
    seq_lengths: list[int],
) -> torch.Tensor:
    # Pool tokens 1..len (residues only); BOS is at 0.
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    lengths = torch.tensor(seq_lengths, device=hidden_states.device).unsqueeze(1)
    mask = ((positions >= 1) & (positions <= lengths)).to(hidden_states.dtype).unsqueeze(-1)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _make_failures(sequences: list[str], error: str) -> list[dict]:
    return [
        {
            "raw_key": seq,
            "canonical_key": seq,
            "error": error,
        }
        for seq in sequences
    ]


def _make_length_buckets(
    sequences: list[str], max_batch_size: int, max_batch_tokens: int
) -> list[list[str]]:
    # `sequences` is sorted by length, so the newest sequence sets the padded width. A sequence
    # whose own padded cost exceeds the budget runs alone.
    batches: list[list[str]] = []
    current_batch: list[str] = []

    for seq in sequences:
        padded_tokens = (len(current_batch) + 1) * (len(seq) + 2)
        if current_batch and (
            len(current_batch) >= max_batch_size or padded_tokens > max_batch_tokens
        ):
            batches.append(current_batch)
            current_batch = []
        current_batch.append(seq)

    if current_batch:
        batches.append(current_batch)

    return batches


def _token_budget(free_bytes: int, bytes_per_token: float, vram_fraction: float) -> int:
    return max(1, int(free_bytes * vram_fraction / bytes_per_token))


def _load_esmc(model_id: str, device_map, torch_dtype_name: str | None, gpu_id: int):
    _ensure_transformers_vision_imports()
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    model_kwargs = {}
    torch_dtype = _torch_dtype_from_name(torch_dtype_name)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if device_map is not None and str(device_map).lower() not in {"", "none", "null"}:
        model_kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, **model_kwargs).eval()
    if "device_map" not in model_kwargs:
        model = model.to(torch.device(f"cuda:{gpu_id}"))
    return tokenizer, model, next(model.parameters()).device


def _embed_batch(batch: list[str], tokenizer, model, model_device) -> torch.Tensor:
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    inputs = {
        name: tensor.to(model_device) for name, tensor in inputs.items() if torch.is_tensor(tensor)
    }
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True)
    return _mean_pool_residues(_extract_final_hidden_state(output), [len(s) for s in batch])


def _calibrated_budget(
    gpu_id: int, tokenizer, model, model_device, vram_fraction: float
) -> int:
    # One forward pass on a short batch measures activation bytes per padded token.
    batch = ["A" * 100] * 4
    torch.cuda.synchronize(model_device)
    baseline = torch.cuda.memory_allocated(model_device)
    torch.cuda.reset_peak_memory_stats(model_device)
    _embed_batch(batch, tokenizer, model, model_device)
    bytes_per_token = (torch.cuda.max_memory_allocated(model_device) - baseline) / (4 * 102)
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(model_device)
    budget = _token_budget(free_bytes, bytes_per_token, vram_fraction)
    write(
        f"GPU {gpu_id} ESMC budget: {budget:,} padded tokens "
        f"({bytes_per_token / 2**20:.2f} MiB/token, {free_bytes / 2**30:.1f} GiB free)"
    )
    return budget


def _embed_with_split(
    batch: list[str],
    run_batch,
    embeddings: dict[str, np.ndarray],
    failures: list[dict],
    gpu_id: int,
) -> None:
    # On CUDA OOM, retry both halves; a single sequence that still fails is recorded.
    try:
        pooled = run_batch(batch)
        for idx, seq in enumerate(batch):
            embeddings[seq] = pooled[idx].detach().float().cpu().numpy().astype(np.float16, copy=False)
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            failures.extend(_make_failures(batch, str(exc)))
            return
        mid = len(batch) // 2
        write(f"GPU {gpu_id} ESMC OOM: splitting batch of {len(batch)} into {mid} + {len(batch) - mid}")
        _embed_with_split(batch[:mid], run_batch, embeddings, failures, gpu_id)
        _embed_with_split(batch[mid:], run_batch, embeddings, failures, gpu_id)
    except Exception as exc:
        failures.extend(_make_failures(batch, str(exc)))
        torch.cuda.empty_cache()


def _embed_sequences(
    gpu_id: int,
    sequences_chunk: list[str],
    tokenizer,
    model,
    model_device,
    progress_queue,
    max_batch_size: int,
    max_batch_tokens: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    embeddings: dict[str, np.ndarray] = {}
    failures: list[dict] = []
    batches = _make_length_buckets(sorted(sequences_chunk, key=len), max_batch_size, max_batch_tokens)

    def run_batch(batch: list[str]) -> torch.Tensor:
        return _embed_batch(batch, tokenizer, model, model_device)

    for batch_sequences in batches:
        _embed_with_split(batch_sequences, run_batch, embeddings, failures, gpu_id)
        progress_queue.put(len(batch_sequences))

    return embeddings, failures


def _esmc_gpu_worker(
    gpu_id: int,
    task_queue,
    result_queue,
    progress_queue,
    model_id: str,
    device_map,
    torch_dtype_name: str | None,
    max_batch_size: int,
    max_batch_tokens: int | None,
    vram_fraction: float,
) -> None:
    # Module-level so the spawn target takes only plain arguments, not `self`. The model loads
    # once per worker. `max_batch_tokens=None` sizes the budget from this GPU's free memory.
    try:
        tokenizer, model, model_device = _load_esmc(model_id, device_map, torch_dtype_name, gpu_id)
        if max_batch_tokens is None:
            max_batch_tokens = _calibrated_budget(
                gpu_id, tokenizer, model, model_device, vram_fraction
            )
    except Exception as exc:
        result_queue.put((gpu_id, None, str(exc)))
        return

    while (sequences_chunk := task_queue.get()) is not None:
        embeddings, failures = _embed_sequences(
            gpu_id,
            sequences_chunk,
            tokenizer,
            model,
            model_device,
            progress_queue,
            max_batch_size,
            max_batch_tokens,
        )
        result_queue.put((gpu_id, embeddings, failures))


class ESMCShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.gpu_ids = list(cfg.embeddings.gpu_ids)
        self.model_id = str(cfg.embeddings.get("esmc_model_id", "biohub/ESMC-6B"))
        self.model_name = str(cfg.embeddings.get("esmc_model_name", "ESMC-6B"))
        self.version = str(cfg.embeddings.get("esmc_version", "hf_v1"))
        tokens = cfg.embeddings.get("esmc_max_batch_tokens", "auto")
        self.max_batch_tokens = None if str(tokens).lower() == "auto" else int(tokens)
        self.vram_fraction = float(cfg.embeddings.get("esmc_vram_fraction", 0.8))
        self.max_batch_size = int(cfg.embeddings.get("esmc_max_batch_size", 64))
        self.device_map = cfg.embeddings.get("esmc_device_map", None)
        self.torch_dtype = cfg.embeddings.get("esmc_torch_dtype", "bfloat16")
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        super().__init__(
            cfg,
            input_path=cfg.embeddings.esmc_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.esmc_embeddings_log_file_name,
            embedder_name="esmc",
            model_name=self.model_name,
            version=self.version,
            storage_dtype="float16",
            key_field="sequence",
            key_type="protein_sequence",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=20_000,
            # dtype and pooling change the vectors but not the path.
            settings=f"dtype={self.torch_dtype};pooling=residues",
        )

        self.logger.info(f"Using GPUs: {self.gpu_ids}")
        self.logger.info(
            f"ESMC Hugging Face model={self.model_id}, version={self.version}, "
            f"max_batch_tokens={self.max_batch_tokens or 'auto'}, vram_fraction={self.vram_fraction}, max_batch_size={self.max_batch_size}, "
            f"dtype={self.torch_dtype}"
        )

    def embed(self) -> None:
        raw_keys = load_pickle_items(self.input_path, self.key_field)
        self.logger.info(f"Loaded {len(raw_keys)} raw items from {self.input_path}")

        unique_keys: list[str] = []
        seen_keys: set[str] = set()
        failures: list[dict] = []

        for raw_key in progress(raw_keys, desc="Preparing input keys", leave=False):
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

        self.logger.info(f"{len(unique_keys)} unique items after deduplication")

        missing_keys = self.store.get_missing_keys(unique_keys)
        self.logger.info(f"{len(missing_keys)} items missing from cache")

        total_written = 0
        if missing_keys:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No CUDA devices available for ESMC embedding.")

            chunk_ranges = list(range(0, len(missing_keys), self.compute_chunk_size))
            self.logger.info(
                f"Computing {len(missing_keys):,} ESMC embeddings across {num_gpus} GPUs."
            )

            progress_queue = mp.get_context("spawn").Queue()
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=self._consume_progress_updates,
                args=(progress_queue, len(missing_keys), "ESMC sequences", stop_event),
                daemon=True,
            )
            progress_thread.start()
            self._start_workers(num_gpus, progress_queue)

            try:
                for start in chunk_ranges:
                    chunk_keys = missing_keys[start : start + self.compute_chunk_size]
                    arrays_by_key, compute_failures = self.compute_many(chunk_keys)
                    total_written += self.store.put_many(arrays_by_key)
                    failures.extend(compute_failures)
                total_written += self.store.put_many({}, flush=True)
            finally:
                self._stop_workers()
                stop_event.set()
                progress_thread.join()
                progress_queue.close()
        else:
            self.logger.info("No new items to compute")

        self.store.record_failures(failures)
        if total_written:
            self.logger.info(f"Persisted {total_written} embeddings across shard writes")

    def _start_workers(self, num_gpus: int, progress_queue) -> None:
        # Each GPU worker loads the model once and reads chunks from its own task queue until it gets None.
        ctx = mp.get_context("spawn")
        self._task_queues = [ctx.Queue() for _ in range(num_gpus)]
        self._result_queue = ctx.Queue()
        self._workers = [
            ctx.Process(
                target=_esmc_gpu_worker,
                args=(
                    gpu_id,
                    self._task_queues[gpu_id],
                    self._result_queue,
                    progress_queue,
                    self.model_id,
                    self.device_map,
                    self.torch_dtype,
                    self.max_batch_size,
                    self.max_batch_tokens,
                    self.vram_fraction,
                ),
            )
            for gpu_id in range(num_gpus)
        ]
        for worker in self._workers:
            worker.start()

    def _stop_workers(self) -> None:
        for task_queue, worker in zip(self._task_queues, self._workers):
            if worker.is_alive():
                task_queue.put(None)
        for worker in self._workers:
            worker.join(timeout=60)
            if worker.is_alive():
                worker.terminate()
                worker.join()

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        num_gpus = len(self._workers)
        for gpu_id, task_queue in enumerate(self._task_queues):
            task_queue.put(keys[gpu_id::num_gpus])

        embeddings: dict[str, np.ndarray] = {}
        failures: list[dict] = []
        for _ in range(num_gpus):
            while True:
                try:
                    gpu_id, chunk_embeddings, chunk_failures = self._result_queue.get(timeout=5)
                    break
                except queue.Empty:
                    if not all(worker.is_alive() for worker in self._workers):
                        raise RuntimeError("An ESMC GPU worker stopped before it returned results.")
            if chunk_embeddings is None:
                raise RuntimeError(f"ESMC GPU worker {gpu_id} failed: {chunk_failures}")
            embeddings.update(chunk_embeddings)
            failures.extend(chunk_failures)
        return embeddings, failures

    @staticmethod
    def _consume_progress_updates(progress_queue, total: int, desc: str, stop_event) -> None:
        if total <= 0:
            return

        with progress(total=total, desc=desc, leave=True, position=0) as progress_bar:
            while True:
                try:
                    increment = progress_queue.get(timeout=0.5)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                progress_bar.update(int(increment))
