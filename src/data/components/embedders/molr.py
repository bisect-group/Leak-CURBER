from __future__ import annotations

import logging
import math
import os
import queue
import sys
import threading
import time
from multiprocessing import get_context
from typing import Any

import numpy as np
from omegaconf import DictConfig
from src.utils.rich_progress import progress

from src.data.components.embedders.base import BaseShardEmbedder


_FEATURE_ENCODER = None


def _graph_worker_init(model_name: str) -> None:
    from MolR.featurizer import load_feature_encoder

    global _FEATURE_ENCODER
    # A fork can inherit the parent's Rich console lock in a held state, so a
    # worker must never log through the parent's Rich handlers.
    logging.getLogger().handlers.clear()
    logging.getLogger("pysmiles").setLevel(logging.ERROR)
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    _FEATURE_ENCODER = load_feature_encoder(model_name)


def _graph_worker(item: tuple[int, str]) -> Any:
    """Build a graph and return its tensors as numpy arrays.

    Torch tensors would cross the pool pipe as shared-memory file descriptors.
    """
    from MolR.featurizer import build_graph_result
    from MolR.graph import Graph

    if _FEATURE_ENCODER is None:
        raise RuntimeError("MolR feature encoder was not initialized in worker.")
    index, smiles = item
    result = build_graph_result(index, smiles, _FEATURE_ENCODER)
    if result.graph is not None:
        g = result.graph
        result.graph = Graph(
            g.feature.numpy(), g.src.numpy(), g.dst.numpy(), g.node_counts.numpy()
        )
    return result


def _restore_graph(result: Any) -> Any:
    import torch
    from MolR.graph import Graph

    if result.graph is not None:
        g = result.graph
        result.graph = Graph(
            *(torch.from_numpy(a) for a in (g.feature, g.src, g.dst, g.node_counts))
        )
    return result


def _split_records(records: list[Any], parts: int) -> list[list[Any]]:
    if parts <= 1:
        return [records]
    return [records[index::parts] for index in range(parts)]


def _molr_inference_worker(
    gpu_id: int | None,
    records: list[Any],
    batch_size: int,
    factor: float,
    output_queue: queue.Queue,
    progress_queue: queue.Queue,
) -> None:
    try:
        import torch
        from MolR.featurizer import MolEFeaturizer, batch_graphs

        model = MolEFeaturizer(gpu=gpu_id)
        model.mole.factor = factor
        device = model.device
        graphs = [record.graph for record in records]
        arrays_by_key: dict[str, np.ndarray] = {}

        with torch.no_grad():
            model.mole.eval()
            for offset in range(0, len(graphs), batch_size):
                graph_batch = batch_graphs(graphs[offset : offset + batch_size]).to(device)
                batch_embeddings = model.mole(graph_batch).detach().cpu().numpy()
                batch_records = records[offset : offset + batch_embeddings.shape[0]]
                for record, embedding in zip(batch_records, batch_embeddings):
                    arrays_by_key[record.smiles] = np.asarray(embedding, dtype=np.float16)
                progress_queue.put(1)
        output_queue.put((arrays_by_key, []))
    except Exception as exc:  # pragma: no cover - exercised in integration smoke.
        output_queue.put(
            (
                {},
                [
                    {
                        "raw_key": record.smiles,
                        "canonical_key": record.smiles,
                        "error": f"MolR inference failed on gpu={gpu_id}: {type(exc).__name__}: {exc}",
                    }
                    for record in records
                ],
            )
        )


class MolR2DShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.batch_size = int(cfg.embeddings.molr_batch_size)
        self.graph_workers = cfg.embeddings.get("molr_graph_workers")
        self.use_gpu = bool(cfg.embeddings.get("molr_use_gpu", True))
        self.gpu_ids = [int(gpu_id) for gpu_id in cfg.embeddings.get("molr_gpu_ids", [0])]
        self.gpu_batch_size = int(cfg.embeddings.get("molr_gpu_batch_size", 2048))
        self.model_name = "gcn_1024"
        self.factor = float(cfg.embeddings.molr_scale_factor)

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

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        started_at = time.time()
        graph_records, graph_failures = self._build_graphs(keys)
        if not graph_records:
            self.logger.info(
                f"MolR graph construction produced no valid graphs; failures={len(graph_failures)}"
            )
            return {}, graph_failures

        arrays_by_key, inference_failures = self._embed_graphs(graph_records)
        failures = [*graph_failures, *inference_failures]
        self.logger.info(
            f"MolR chunk complete: requested={len(keys):,}, graphs={len(graph_records):,}, "
            f"embeddings={len(arrays_by_key):,}, failures={len(failures):,}, "
            f"seconds={time.time() - started_at:.1f}"
        )
        return arrays_by_key, failures

    def _build_graphs(self, keys: list[str]) -> tuple[list[Any], list[dict]]:
        worker_count = (
            int(self.graph_workers) if self.graph_workers else max(1, os.cpu_count() or 1)
        )
        self.logger.info(
            f"MolR graph construction: {len(keys):,} SMILES using {worker_count} CPU workers"
        )
        ctx = get_context("fork")
        records: list[Any] = []
        failures: list[dict] = []
        with ctx.Pool(
            processes=worker_count,
            initializer=_graph_worker_init,
            initargs=(self.model_name,),
        ) as pool:
            iterator = pool.imap_unordered(
                _graph_worker,
                enumerate(keys),
                chunksize=max(1, min(1000, len(keys) // max(worker_count * 4, 1) or 1)),
            )
            for result in progress(
                iterator,
                total=len(keys),
                desc="MolR CPU graph construction",
                leave=True,
                dynamic_ncols=True,
            ):
                if result.graph is None:
                    failures.append(
                        {
                            "raw_key": result.smiles,
                            "canonical_key": result.smiles,
                            "error": result.error or "MolR graph construction failed",
                        }
                    )
                    continue
                records.append(_restore_graph(result))

        records.sort(key=lambda record: record.index)
        self.logger.info(
            f"MolR graph construction complete: graphs={len(records):,}, "
            f"failures={len(failures):,}"
        )
        return records, failures

    def _embed_graphs(self, records: list[Any]) -> tuple[dict[str, np.ndarray], list[dict]]:
        gpu_ids = self._active_gpu_ids()
        worker_count = len(gpu_ids)
        record_splits = [split for split in _split_records(records, worker_count) if split]
        total_batches = sum(math.ceil(len(split) / self.gpu_batch_size) for split in record_splits)
        device_label = "CPU" if gpu_ids == [None] else "GPU"
        self.logger.info(
            f"MolR {device_label} inference: {len(records):,} graphs, "
            f"workers={worker_count}, batch_size={self.gpu_batch_size}, batches={total_batches:,}"
        )

        output_queue: queue.Queue = queue.Queue()
        progress_queue: queue.Queue = queue.Queue()
        threads = []
        for gpu_id, split in zip(gpu_ids, record_splits):
            thread = threading.Thread(
                target=_molr_inference_worker,
                args=(
                    gpu_id,
                    split,
                    self.gpu_batch_size,
                    self.factor,
                    output_queue,
                    progress_queue,
                ),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        with progress(
            total=total_batches,
            desc=f"MolR {device_label} inference batches",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            while any(thread.is_alive() for thread in threads):
                try:
                    progress_queue.get(timeout=0.5)
                    pbar.update(1)
                except queue.Empty:
                    pass
            while True:
                try:
                    progress_queue.get_nowait()
                    pbar.update(1)
                except queue.Empty:
                    break

        for thread in threads:
            thread.join()

        arrays_by_key: dict[str, np.ndarray] = {}
        failures: list[dict] = []
        for _ in threads:
            try:
                worker_arrays, worker_failures = output_queue.get(timeout=30)
                arrays_by_key.update(worker_arrays)
                failures.extend(worker_failures)
            except queue.Empty:
                failures.extend(
                    [
                        {
                            "raw_key": record.smiles,
                            "canonical_key": record.smiles,
                            "error": "MolR inference worker exited without returning results",
                        }
                        for record in records
                        if record.smiles not in arrays_by_key
                    ]
                )
                break
        return arrays_by_key, failures

    def _active_gpu_ids(self) -> list[int | None]:
        if not self.use_gpu:
            return [None]
        try:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                self.logger.warning(
                    "MolR GPU inference requested but CUDA is unavailable; falling back to CPU."
                )
                return [None]
            visible_count = torch.cuda.device_count()
            return [gpu_id for gpu_id in self.gpu_ids if 0 <= gpu_id < visible_count] or [0]
        except Exception as exc:
            self.logger.warning(
                f"Could not inspect CUDA for MolR inference ({exc}); falling back to CPU."
            )
            return [None]
