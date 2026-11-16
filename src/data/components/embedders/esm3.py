from __future__ import annotations

import json
import os
import pickle
import queue
import shutil
import traceback
import warnings
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig
from src.utils.rich_progress import progress
from src.utils.structure_utils import mmcif_file, select_chains, selection_counts

from src.data.components.embedders.base import BaseShardEmbedder
from src.data.components.embedders.utils import (
    PDB_SOURCE_KEY_TYPE,
    PDB_SOURCES,
    load_structure_keys,
    sha256_short,
)


def _suppress_esm3_warnings() -> None:
    warnings.filterwarnings("ignore")


_suppress_esm3_warnings()


_GPU_WORKER_THREADS = 4


def _sequences_match(structure_seq: str, dataset_seq: str) -> bool:
    """Equal, or equal length with "X" in the structure where the dataset has a non-standard letter."""
    return len(structure_seq) == len(dataset_seq) and all(
        s == d or (s == "X" and d not in "ACDEFGHIKLMNPQRSTVWY")
        for s, d in zip(structure_seq, dataset_seq)
    )


def _raw_key(key: str) -> str:
    """Drop the structure source tag; a mutant accession such as `P1|R1Q` keeps its suffix."""
    raw_key, _, source = key.rpartition("|")
    return raw_key if source in PDB_SOURCES else key


def _featurize_chunk(items: list[tuple[str, str, Optional[str]]]) -> list[dict]:
    return [_featurize_esm3_record(item) for item in items]


def _init_featurize_worker() -> None:
    """Keep each featurize worker on one CPU thread."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)


def _featurize_esm3_record(args: tuple[str, str, Optional[str]]) -> dict:
    """Featurize a structure file.

    `chain_id` selects the chain of an experimental mmCIF file. A CIF file without a chain
    id is a predicted ESMFold2 structure and holds one protein chain.
    """
    _suppress_esm3_warnings()
    key, pdb_path, chain_id = args
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        from esm.sdk.api import ESMProtein
        from esm.utils.structure.mmcif_parsing import MmcifWrapper
        from esm.utils.structure.protein_chain import ProteinChain

        if chain_id is None and pdb_path.endswith(".cif"):
            atoms = pdbx.get_structure(pdbx.CIFFile.read(pdb_path), model=1)
            chain = ProteinChain.from_atomarray(atoms[struc.filter_amino_acids(atoms)])
        elif chain_id is None:
            protein = ESMProtein.from_pdb(Path(pdb_path))
            if protein.sequence is None or protein.coordinates is None:
                return {
                    "raw_key": key,
                    "canonical_key": key,
                    "pdb_path": pdb_path,
                    "error": "Parsed protein is missing sequence or coordinates",
                }
            chain = ProteinChain.from_atom37(protein.coordinates, sequence=protein.sequence)
        else:
            chain = ProteinChain.from_mmcif(MmcifWrapper.read(pdb_path), chain_id=chain_id)
        coords, _plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        residue_index = (
            residue_index.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)
        )
        return {
            "key": key,
            "key_hash": sha256_short(key),
            "pdb_path": pdb_path,
            "sequence": chain.sequence,
            "length": len(chain.sequence),
            "coords": coords,
            "residue_index": residue_index,
        }
    except Exception as exc:
        return {
            "raw_key": key,
            "canonical_key": key,
            "pdb_path": pdb_path,
            "error": str(exc),
        }


def _make_length_buckets_from_records(
    records: list[dict],
    *,
    max_batch_tokens: int,
    max_batch_size: int,
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for record in records:
        seq_tokens = int(record["length"]) + 2
        if current_batch and (
            len(current_batch) >= max_batch_size
            or current_tokens + seq_tokens > max_batch_tokens
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(record)
        current_tokens += seq_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def _mean_pool_residues(embeddings: torch.Tensor, length: int) -> torch.Tensor:
    # Pool residues only (tokens 1..length).
    return embeddings[1 : length + 1].mean(dim=0)


def _pool_batch(
    embeddings: torch.Tensor, batch: list[dict]
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Pool each record's residues; rows with NaN or inf become failures."""
    arrays_by_key = {}
    failures = []
    for idx, record in enumerate(batch):
        pooled = (
            _mean_pool_residues(embeddings[idx], int(record["length"]))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float16, copy=False)
        )
        if np.isfinite(pooled).all():
            arrays_by_key[record["key"]] = pooled
        else:
            failures.append(
                {
                    "raw_key": record["key"],
                    "canonical_key": record["key"],
                    "error": "Embedding contains NaN or inf",
                }
            )
    return arrays_by_key, failures


# Elements of the (query - key) distance tensor computed at once in geometric attention.
_GEOM_ATTN_CHUNK_ELEMENTS = 2**27


def _chunked_geometric_attention(self, s, affine, affine_mask, sequence_id, chain_id):
    """`GeometricReasoningOriginalImpl.forward` with queries processed in chunks.

    Softmax runs over keys per query row, so chunking over queries gives the same result
    while the (batch, heads, query, key, 3) distance tensor stays bounded.
    """
    from math import sqrt

    from einops import rearrange
    from torch.nn import functional as F

    if sequence_id is None:
        sequence_id = torch.zeros_like(s[..., 0], dtype=torch.int64)
    attn_bias = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
    attn_bias = attn_bias.unsqueeze(1).float()
    attn_bias = attn_bias.masked_fill(
        ~affine_mask[:, None, None, :], torch.finfo(attn_bias.dtype).min
    )
    chain_id_mask = chain_id.unsqueeze(1) != chain_id.unsqueeze(2)
    attn_bias = attn_bias.masked_fill(chain_id_mask.unsqueeze(1), torch.finfo(s.dtype).min)

    ns = self.s_norm(s)
    vec_rot, vec_dist = self.proj(ns).split(
        [
            self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
            self.v_heads * 2 * 3,
        ],
        dim=-1,
    )
    query_rot, key_rot, value = (
        affine.rot[..., None]
        .apply(rearrange(vec_rot, "... (h c) -> ... h c", c=3))
        .split([self.v_heads, self.v_heads, self.v_heads * self.num_vector_messages], dim=-2)
    )
    query_dist, key_dist = (
        affine[..., None].apply(rearrange(vec_dist, "... (h c) -> ... h c", c=3)).chunk(2, dim=-2)
    )
    query_dist = rearrange(query_dist, "b s h d -> b h s 1 d")
    key_dist = rearrange(key_dist, "b s h d -> b h 1 s d")
    query_rot = rearrange(query_rot, "b s h d -> b h s d")
    key_rot = rearrange(key_rot, "b s h d -> b h d s")
    value = rearrange(value, "b s (h m) d -> b h s (m d)", m=self.num_vector_messages)
    distance_term_weight = rearrange(F.softplus(self.distance_scale_per_head), "h -> h 1 1")
    rotation_term_weight = rearrange(F.softplus(self.rotation_scale_per_head), "h -> h 1 1")

    batch, heads, seq = query_rot.shape[:3]
    chunk = max(1, _GEOM_ATTN_CHUNK_ELEMENTS // (batch * heads * seq * 3))
    outputs = []
    for start in range(0, seq, chunk):
        rows = slice(start, start + chunk)
        distance_term = (query_dist[:, :, rows] - key_dist).norm(dim=-1) / sqrt(3)
        rotation_term = query_rot[:, :, rows].matmul(key_rot) / sqrt(3)
        attn_weight = (
            rotation_term * rotation_term_weight - distance_term * distance_term_weight
        ) + attn_bias[:, :, rows]
        outputs.append(torch.softmax(attn_weight, dim=-1).matmul(value))
    attn_out = torch.cat(outputs, dim=2)

    attn_out = (
        affine.rot[..., None]
        .invert()
        .apply(rearrange(attn_out, "b h s (m d) -> b s (h m) d", m=self.num_vector_messages))
    )
    attn_out = rearrange(attn_out, "b s (h m) d -> b s (h m d)", m=self.num_vector_messages)
    if self.mask_and_zero_frameless:
        attn_out = attn_out.masked_fill(~affine_mask[..., None], 0.0)
    return self.out_proj(attn_out)


def _embed_batch(
    client,
    structure_encoder,
    batch: list[dict],
    device,
    autocast_dtype: Optional[torch.dtype],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Embed one batch of feature records; the structure encoder runs under autocast."""
    from esm.sdk.api import LogitsConfig
    from esm.utils import encoding
    from esm.utils.misc import stack_variable_length_tensors
    from esm.utils.sampling import _BatchedESMProteinTensor

    sequence_tensors = []
    raw_coords = []
    residue_indices = []
    lengths = []

    for record in batch:
        sequence = str(record["sequence"])
        sequence_tensors.append(
            encoding.tokenize_sequence(
                sequence,
                client.tokenizers.sequence,
                add_special_tokens=True,
            )
        )
        raw_coords.append(
            torch.as_tensor(record["coords"], dtype=torch.float32)
        )
        residue_indices.append(
            torch.as_tensor(record["residue_index"], dtype=torch.int64)
        )
        lengths.append(int(record["length"]))

    coords_batch = stack_variable_length_tensors(
        raw_coords,
        constant_value=torch.inf,
    ).to(device)
    residue_index_batch = stack_variable_length_tensors(
        residue_indices,
        constant_value=0,
    ).to(device)

    autocast = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else nullcontext()
    with torch.inference_mode():
        with autocast:
            _, structure_tokens_batch = structure_encoder.encode(
                coords_batch,
                residue_index=residue_index_batch,
            )

        structure_token_tensors = []
        coordinate_tensors = []
        for idx, length in enumerate(lengths):
            structure_tokens = structure_tokens_batch[idx, :length]
            structure_tokens = F.pad(
                structure_tokens,
                (1, 1),
                value=client.tokenizers.structure.mask_token_id,
            )
            structure_tokens[0] = client.tokenizers.structure.bos_token_id
            structure_tokens[-1] = client.tokenizers.structure.eos_token_id
            structure_token_tensors.append(structure_tokens)

            coords_i = raw_coords[idx].to(device)
            coords_i = F.pad(
                coords_i,
                (0, 0, 0, 0, 1, 1),
                value=torch.inf,
            )
            coordinate_tensors.append(coords_i)

        batched = _BatchedESMProteinTensor(
            sequence=stack_variable_length_tensors(
                [seq.to(device) for seq in sequence_tensors],
                constant_value=client.tokenizers.sequence.pad_token_id,
            ),
            structure=stack_variable_length_tensors(
                structure_token_tensors,
                constant_value=client.tokenizers.structure.pad_token_id,
            ),
            coordinates=stack_variable_length_tensors(
                coordinate_tensors,
                constant_value=torch.inf,
            ),
        )

        logits_output = client.logits(
            batched,
            LogitsConfig(return_embeddings=True),
        )
    assert logits_output.embeddings is not None

    return _pool_batch(logits_output.embeddings, batch)


def _embed_with_oom_split(
    client,
    structure_encoder,
    batch: list[dict],
    device,
    autocast_dtype: Optional[torch.dtype],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Embed a batch; on CUDA OOM split it in half and retry down to one record."""
    try:
        return _embed_batch(client, structure_encoder, batch, device, autocast_dtype)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1:
            key = batch[0]["key"]
            error = f"CUDA out of memory for a sequence of length {batch[0]['length']}"
            return {}, [{"raw_key": key, "canonical_key": key, "error": error}]
    half = len(batch) // 2
    arrays: dict[str, np.ndarray] = {}
    failures: list[dict] = []
    for part in (batch[:half], batch[half:]):
        part_arrays, part_failures = _embed_with_oom_split(
            client, structure_encoder, part, device, autocast_dtype
        )
        arrays.update(part_arrays)
        failures.extend(part_failures)
    return arrays, failures


def _esm3_gpu_worker(
    rank: int,
    max_batch_tokens: int,
    max_batch_size: int,
    task_queue,
    result_queue,
    autocast_dtype: Optional[str],
) -> None:
    _suppress_esm3_warnings()
    dtype = None if autocast_dtype in (None, "float32") else getattr(torch, autocast_dtype)
    try:
        from esm.models.esm3 import ESM3
        from esm.utils.constants.models import ESM3_OPEN_SMALL

        from esm.layers.geom_attention import GeometricReasoningOriginalImpl

        GeometricReasoningOriginalImpl.forward = _chunked_geometric_attention
        torch.set_num_threads(_GPU_WORKER_THREADS)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to(f"cuda:{rank}")
        client.eval()
        device = next(client.parameters()).device
        structure_encoder = client.get_structure_encoder()

        while True:
            task = task_queue.get()
            if task is None:
                result_queue.put(("done", rank, 0, None))
                return

            task_id = int(task["task_id"])
            rows = task["rows"]
            if not rows:
                result_queue.put(("task_done", rank, task_id, 0))
                continue

            try:
                records = _load_feature_task_records(task["shard_path"], rows)
                records.sort(key=lambda record: int(record["length"]))
            except Exception as exc:
                failures = [
                    {
                        "raw_key": row["key"],
                        "canonical_key": row["key"],
                        "error": str(exc),
                    }
                    for row in rows
                ]
                result_queue.put(("batch", rank, {}, failures))
                result_queue.put(("task_done", rank, task_id, len(rows)))
                continue

            batches = _make_length_buckets_from_records(
                records,
                max_batch_tokens=max_batch_tokens,
                max_batch_size=max_batch_size,
            )
            processed = 0

            for batch in batches:
                keys = [record["key"] for record in batch]
                processed += len(keys)
                try:
                    arrays_by_key, batch_failures = _embed_with_oom_split(
                        client, structure_encoder, batch, device, dtype
                    )
                    result_queue.put(("batch", rank, arrays_by_key, batch_failures))
                except Exception as exc:
                    failures = [
                        {
                            "raw_key": key,
                            "canonical_key": key,
                            "error": str(exc),
                        }
                        for key in keys
                    ]
                    result_queue.put(("batch", rank, {}, failures))
                    torch.cuda.empty_cache()

            result_queue.put(("task_done", rank, task_id, processed))
    except Exception as exc:
        result_queue.put(
            (
                "fatal",
                rank,
                str(exc),
                traceback.format_exc(),
            )
        )


def _load_feature_task_records(shard_path: str, rows: list[dict]) -> list[dict]:
    with open(shard_path, "rb") as handle:
        shard_records = pickle.load(handle)
    return [shard_records[int(row["row_idx"])] for row in rows]


class _ESM3GpuPool:
    """Forked GPU workers that embed feature shards as the parent submits them."""

    def __init__(
        self,
        store,
        max_batch_tokens: int,
        max_batch_size: int,
        autocast_dtype: Optional[str],
        logger,
    ) -> None:
        self.store = store
        self.autocast_dtype = autocast_dtype
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.logger = logger
        self.processes: list = []
        self.task_queues: list = []
        self.result_queue = None
        self.next_task_id = 0
        self.total_written = 0
        self.failures: list[dict] = []
        self.done_workers = 0

    def start(self) -> None:
        if self.processes:
            return
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available for ESM3 structure embedding.")
        self.logger.info(f"Starting {num_gpus} ESM3 GPU worker(s)")
        ctx = mp.get_context("fork")
        self.result_queue = ctx.Queue(maxsize=1024)
        self.task_queues = [ctx.Queue() for _ in range(num_gpus)]
        for rank in range(num_gpus):
            process = ctx.Process(
                target=_esm3_gpu_worker,
                args=(
                    rank,
                    self.max_batch_tokens,
                    self.max_batch_size,
                    self.task_queues[rank],
                    self.result_queue,
                    self.autocast_dtype,
                ),
            )
            process.start()
            self.processes.append(process)

    def submit_locations(self, locations: Iterable[dict]) -> None:
        by_shard: dict[int, list[dict]] = {}
        for location in locations:
            by_shard.setdefault(int(location["shard_id"]), []).append(location)
        if not by_shard:
            return
        self.start()
        num_gpus = len(self.processes)
        for shard_id, shard_locations in sorted(by_shard.items()):
            shard_locations.sort(key=lambda location: int(location["length"]))
            for rank in range(num_gpus):
                rank_locations = shard_locations[rank::num_gpus]
                if not rank_locations:
                    continue
                self.task_queues[rank].put(
                    {
                        "task_id": self.next_task_id,
                        "shard_id": shard_id,
                        "shard_path": rank_locations[0]["shard_path"],
                        "rows": [
                            {"row_idx": int(location["row_idx"]), "key": location["key"]}
                            for location in rank_locations
                        ],
                    }
                )
                self.next_task_id += 1
            self.logger.info(
                f"Queued ESM3 feature shard {shard_id} "
                f"({len(shard_locations):,} records) for GPU embedding"
            )

    def poll(self) -> None:
        """Handle all available worker messages without blocking."""
        while self.processes:
            try:
                message = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self._handle(message)
        self._check_workers()

    def finish(self) -> tuple[int, list[dict]]:
        """Stop the workers after they drain their tasks."""
        if not self.processes:
            return 0, []
        for task_queue in self.task_queues:
            task_queue.put(None)
        while self.done_workers < len(self.processes):
            try:
                self._handle(self.result_queue.get(timeout=1.0))
            except queue.Empty:
                self._check_workers()
        for process in self.processes:
            process.join()
        self._check_workers()
        self.total_written += self.store.put_many({}, flush=True)
        return self.total_written, self.failures

    def close(self) -> None:
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join()
        for task_queue in self.task_queues:
            task_queue.cancel_join_thread()
            task_queue.close()
        if self.result_queue is not None:
            self.result_queue.close()

    def _handle(self, message: tuple) -> None:
        kind, rank, payload, extra = message
        if kind == "batch":
            self.total_written += self.store.put_many(payload)
            self.failures.extend(extra or [])
        elif kind == "done":
            self.done_workers += 1
        elif kind == "fatal":
            raise RuntimeError(f"GPU worker {rank} failed: {payload}\n{extra}")

    def _check_workers(self) -> None:
        for process in self.processes:
            if process.exitcode not in (None, 0):
                raise RuntimeError(
                    f"GPU worker {process.pid} exited with code {process.exitcode}"
                )


class ESM3FeatureStore:
    def __init__(
        self,
        root_dir: Path,
        records_per_shard: int,
        logger,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.records_per_shard = int(records_per_shard)
        self.logger = logger
        self.shards_dir = self.root_dir / "shards"
        self.manifest_path = self.root_dir / "manifest.parquet"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

    def successful_key_hashes(self) -> set[str]:
        manifest = self._load_manifest()
        if manifest.empty:
            return set()
        return set(manifest.loc[manifest["status"] == "ok", "key_hash"])

    def failed_key_hashes(self) -> set[str]:
        manifest = self._load_manifest()
        if manifest.empty:
            return set()
        return set(manifest.loc[manifest["status"] == "error", "key_hash"].dropna())

    def failure_dicts_for_keys(self, keys: list[str]) -> list[dict]:
        manifest = self._load_manifest()
        if manifest.empty:
            return []
        key_hashes = {sha256_short(key) for key in keys}
        failures = manifest[
            (manifest["status"] == "error") & manifest["key_hash"].isin(key_hashes)
        ].drop_duplicates(subset=["key_hash"], keep="last")
        return [
            {
                "raw_key": row.key,
                "canonical_key": row.key,
                "error": row.error,
            }
            for row in failures.itertuples(index=False)
        ]

    def completed_shards(self) -> list[tuple[int, Path]]:
        shards: list[tuple[int, Path]] = []
        for shard_path in self.shards_dir.glob("feature_shard_*.pkl"):
            shard_id = self._parse_shard_id(shard_path)
            if shard_id is not None and shard_path.name == f"feature_shard_{shard_id:06d}.pkl":
                shards.append((shard_id, shard_path))
        return sorted(shards, key=lambda item: item[0])

    def find_feature_locations_for_keys(self, keys: list[str]) -> dict[str, dict]:
        if not keys:
            return {}

        requested_hashes = {sha256_short(key) for key in keys}
        locations: dict[str, dict] = {}
        shards = self.completed_shards()
        for shard_id, shard_path in progress(
            shards,
            desc="Scanning ESM3 feature shards",
            leave=False,
        ):
            remaining_hashes = requested_hashes - set(locations)
            if not remaining_hashes:
                break
            try:
                with open(shard_path, "rb") as handle:
                    shard_records = pickle.load(handle)
            except Exception as exc:
                self.logger.warning(f"Skipping unreadable ESM3 feature shard {shard_path}: {exc}")
                continue

            for row_idx, record in enumerate(shard_records):
                key = str(record.get("key"))
                key_hash = str(record.get("key_hash") or sha256_short(key))
                if key_hash not in requested_hashes:
                    continue
                locations[key_hash] = {
                    "key": key,
                    "key_hash": key_hash,
                    "length": int(record["length"]),
                    "shard_id": shard_id,
                    "shard_path": str(shard_path),
                    "row_idx": row_idx,
                }

        return locations

    def iter_records_by_shard(
        self,
        locations: Iterable[dict],
        *,
        desc: str = "Loading ESM3 feature shards",
    ):
        by_shard: dict[int, list[dict]] = {}
        for location in locations:
            by_shard.setdefault(int(location["shard_id"]), []).append(location)

        for shard_id, shard_locations in progress(
            sorted(by_shard.items()),
            desc=desc,
            leave=False,
        ):
            shard_path = str(shard_locations[0]["shard_path"])
            rows = [
                {"row_idx": int(location["row_idx"]), "key": location["key"]}
                for location in shard_locations
            ]
            records = _load_feature_task_records(shard_path, rows)
            yield shard_id, records

    def put_records(self, records: list[dict], on_shard=None) -> int:
        if not records:
            return 0

        written = 0
        for start in range(0, len(records), self.records_per_shard):
            shard_records = records[start : start + self.records_per_shard]
            shard_id = self._next_shard_id()
            shard_path = self.shards_dir / f"feature_shard_{shard_id:06d}.pkl"
            tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
            with open(tmp_path, "wb") as handle:
                pickle.dump(shard_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(shard_path)

            rows = []
            for row_idx, record in enumerate(shard_records):
                rows.append(
                    {
                        "key": record["key"],
                        "key_hash": record["key_hash"],
                        "length": int(record["length"]),
                        "pdb_path": record["pdb_path"],
                        "feature_shard": shard_id,
                        "row_idx": row_idx,
                        "status": "ok",
                        "error": None,
                    }
                )
            self._append_manifest_rows(rows)
            if on_shard is not None:
                on_shard(
                    [
                        {
                            "key": row["key"],
                            "length": row["length"],
                            "shard_id": shard_id,
                            "shard_path": str(shard_path),
                            "row_idx": row["row_idx"],
                        }
                        for row in rows
                    ]
                )
            written += len(shard_records)
            self.logger.info(
                f"Wrote ESM3 feature_shard_{shard_id:06d}.pkl with "
                f"{len(shard_records):,} records"
            )

        return written

    def put_failures(self, failures: list[dict]) -> int:
        if not failures:
            return 0
        rows = []
        for failure in failures:
            key = failure.get("canonical_key") or failure.get("raw_key")
            key = str(key) if key is not None else None
            rows.append(
                {
                    "key": key,
                    "key_hash": sha256_short(key) if key is not None else None,
                    "length": None,
                    "pdb_path": failure.get("pdb_path"),
                    "feature_shard": None,
                    "row_idx": None,
                    "status": "error",
                    "error": failure.get("error"),
                }
            )
        self._append_manifest_rows(rows)
        return len(rows)

    def load_records_for_keys(self, keys: list[str]) -> list[dict]:
        locations = self.find_feature_locations_for_keys(keys)
        ordered_hashes = [sha256_short(key) for key in keys]
        ordered_locations = [
            locations[key_hash] for key_hash in ordered_hashes if key_hash in locations
        ]
        records: list[dict] = []
        for _shard_id, shard_records in self.iter_records_by_shard(ordered_locations):
            records.extend(shard_records)

        requested_order = {key_hash: idx for idx, key_hash in enumerate(ordered_hashes)}
        records.sort(key=lambda record: requested_order[record["key_hash"]])
        return records

    def cleanup(self) -> None:
        shutil.rmtree(self.root_dir, ignore_errors=True)

    def _load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame(
                columns=[
                    "key",
                    "key_hash",
                    "length",
                    "pdb_path",
                    "feature_shard",
                    "row_idx",
                    "status",
                    "error",
                ]
            )
        return pd.read_parquet(self.manifest_path)

    def _append_manifest_rows(self, rows: list[dict]) -> None:
        existing_df = (
            pd.read_parquet(self.manifest_path)
            if self.manifest_path.exists()
            else pd.DataFrame()
        )
        appended_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
        tmp_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        appended_df.to_parquet(tmp_path, index=False, compression="zstd", compression_level=3)
        tmp_path.replace(self.manifest_path)

    def _next_shard_id(self) -> int:
        shard_ids = []
        for shard_path in self.shards_dir.glob("feature_shard_*.pkl"):
            shard_id = self._parse_shard_id(shard_path)
            if shard_id is not None:
                shard_ids.append(shard_id)
        return max(shard_ids, default=-1) + 1

    @staticmethod
    def _parse_shard_id(shard_path: Path) -> Optional[int]:
        try:
            return int(shard_path.stem.split("_")[-1])
        except ValueError:
            return None


class ESM3StructureShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        self.gpu_ids = list(cfg.embeddings.gpu_ids)
        self.alphafold_pdb_path = cfg.embeddings.af_pdb_path
        self.esm3_pdb_path = cfg.embeddings.esm_pdb_path
        self.mmcif_path = cfg.embeddings.mmcif_path
        self.uniprot_acc_path = cfg.embeddings.uniprot_acc_path
        self.esmfold2_cif_path = cfg.embeddings.esmfold2_cif_path
        default_featurized_dir = (
            Path(cfg.paths.raw_data_dir) / "esm3_featurized"
            if "paths" in cfg and "raw_data_dir" in cfg.paths
            else Path(cfg.embeddings.embeddings_path).parent.parent
            / "raw"
            / "esm3_featurized"
        )
        self.featurized_dir = Path(
            cfg.embeddings.get(
                "esm3_featurized_dir",
                default_featurized_dir,
            )
        )
        self.keep_featurized_after_success = bool(
            cfg.embeddings.get("esm3_keep_featurized_after_success", True)
        )
        self.reclaim_last_underfilled_shard = bool(
            cfg.embeddings.get("esm3_reclaim_last_underfilled_shard", True)
        )
        self.retry_failed = bool(cfg.embeddings.get("esm3_retry_failed", False))
        self.featurize_workers = int(cfg.embeddings.get("esm3_featurize_workers", 0))
        self.featurize_backend = str(
            cfg.embeddings.get("esm3_featurize_backend", "process")
        ).lower()
        if self.featurize_backend not in {"process", "thread"}:
            raise ValueError(
                "embeddings.esm3_featurize_backend must be 'process' or 'thread'."
            )
        self.feature_records_per_shard = int(
            cfg.embeddings.get("esm3_feature_records_per_shard", 5_000)
        )
        self.autocast_dtype = str(cfg.embeddings.get("esm3_autocast_dtype", "bfloat16"))
        if self.autocast_dtype not in {"bfloat16", "float32"}:
            raise ValueError("embeddings.esm3_autocast_dtype must be 'bfloat16' or 'float32'.")
        self.max_batch_tokens = int(cfg.embeddings.get("esm3_max_batch_tokens", 4_096))
        self.max_batch_size = int(cfg.embeddings.get("esm3_max_batch_size", 8))
        self.parse_workers = max(1, (os.cpu_count() or 1) // max(1, len(self.gpu_ids)))
        self.parse_chunk_size = 256
        self._pdb_index: Optional[dict[str, dict]] = None

        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        super().__init__(
            cfg,
            input_path=cfg.embeddings.esm3_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.esm3_embeddings_log_file_name,
            embedder_name="esm3",
            model_name="open_small_structure",
            version="v1",
            storage_dtype="float16",
            key_field="acc_id",
            key_type=PDB_SOURCE_KEY_TYPE,
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=10_000,
            settings=f"pooling=residues;encoder_dtype={self.autocast_dtype}",
            pending_checkpoint_interval_embeddings=int(
                cfg.embeddings.get("esm3_pending_checkpoint_interval_embeddings", 5_000)
            ),
            pending_checkpoint_interval_seconds=int(
                cfg.embeddings.get("esm3_pending_checkpoint_interval_seconds", 300)
            ),
        )

        self.logger.info(f"Using GPUs: {self.gpu_ids}")
        self.logger.info(
            f"ESM3 batching enabled with max_batch_tokens={self.max_batch_tokens}, "
            f"max_batch_size={self.max_batch_size}, "
            f"encoder_dtype={self.autocast_dtype}"
        )
        self.logger.info(
            f"ESM3 two-phase featurization enabled at {self.featurized_dir} "
            f"with feature_records_per_shard={self.feature_records_per_shard}"
        )

    def embed(self) -> None:
        unique_keys, input_failures = self._load_unique_input_keys()
        self.logger.info(f"Loaded {len(unique_keys)} unique ESM3 input keys")

        if self.reclaim_last_underfilled_shard:
            self.store.reclaim_last_underfilled_shard(unique_keys)

        missing_keys = self.store.get_missing_keys(unique_keys)
        self.logger.info(f"{len(missing_keys)} ESM3 items missing from cache")
        if not missing_keys:
            self.store.record_failures(input_failures)
            self.store.flush()
            self.logger.info("No new ESM3 items to compute")
            return

        feature_store = ESM3FeatureStore(
            self.featurized_dir,
            self.feature_records_per_shard,
            self.logger,
        )
        feature_locations = feature_store.find_feature_locations_for_keys(missing_keys)
        existing_locations = [
            feature_locations[key_hash]
            for key_hash in map(sha256_short, missing_keys)
            if key_hash in feature_locations
        ]
        self.logger.info(
            f"{len(existing_locations)} ESM3 feature records already available for GPU embedding"
        )

        gpu_pool = _ESM3GpuPool(
            self.store,
            self.max_batch_tokens,
            self.max_batch_size,
            self.autocast_dtype,
            self.logger,
        )
        try:
            gpu_pool.submit_locations(existing_locations)
            self.featurize_missing(
                missing_keys,
                feature_store,
                gpu_pool,
                existing_feature_hashes=set(feature_locations),
            )
            total_written, embed_failures = gpu_pool.finish()
        finally:
            gpu_pool.close()
        feature_failures = feature_store.failure_dicts_for_keys(missing_keys)
        if not gpu_pool.next_task_id:
            self.logger.warning("No ESM3 feature records are available for embedding")

        self.store.record_failures(input_failures + feature_failures + embed_failures)
        self.store.flush()
        if total_written:
            self.logger.info(f"Persisted {total_written} ESM3 embeddings")

        if not self.keep_featurized_after_success:
            feature_store.cleanup()
            self.logger.info(f"Removed ESM3 feature checkpoint at {self.featurized_dir}")

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        raise RuntimeError("ESM3StructureShardEmbedder uses two-phase embed().")

    def featurize_missing(
        self,
        missing_keys: list[str],
        feature_store: ESM3FeatureStore,
        gpu_pool: _ESM3GpuPool,
        existing_feature_hashes: Optional[set[str]] = None,
    ) -> int:
        if existing_feature_hashes is None:
            existing_feature_hashes = set(
                feature_store.find_feature_locations_for_keys(missing_keys)
            )
        # Skip keys already recorded as failed unless `embeddings.esm3_retry_failed` is true.
        skip_hashes = set(existing_feature_hashes)
        if not self.retry_failed:
            skip_hashes |= feature_store.failed_key_hashes()
        keys_to_featurize = [
            key for key in missing_keys if sha256_short(key) not in skip_hashes
        ]
        if not keys_to_featurize:
            self.logger.info("All missing ESM3 keys already have feature records")
            return 0

        self.logger.info(
            f"Resolving PDB paths for {len(keys_to_featurize):,} ESM3 keys"
        )
        chains = self._select_experimental_chains(keys_to_featurize)
        esmfold2_keys = {_raw_key(k) for k in keys_to_featurize if k.endswith("|esmfold2")}
        expected_sequences = self._dataset_sequences(esmfold2_keys) if esmfold2_keys else {}
        work_items = []
        missing_pdb_failures = []
        for key in progress(keys_to_featurize, desc="Resolving ESM3 PDB paths", leave=False):
            raw_key = _raw_key(key)
            resolved = self._resolve_pdb(raw_key)
            pdb_path = None if resolved is None else resolved[0]
            error = None
            if pdb_path is None:
                error = "No PDB file found"
            elif ":" in raw_key and chains.get(raw_key) is None:
                error = "No mmCIF chain matches the UniProt sequence"
            elif key.endswith("|esmfold2") and raw_key not in expected_sequences:
                error = "No dataset sequence to check the ESMFold2 structure against"
            if error:
                missing_pdb_failures.append(
                    {
                        "raw_key": key,
                        "canonical_key": key,
                        "pdb_path": pdb_path,
                        "error": error,
                    }
                )
            else:
                work_items.append((key, pdb_path, chains.get(raw_key)))

        feature_store.put_failures(missing_pdb_failures)
        if not work_items:
            return 0

        num_workers = self.featurize_workers or max(1, (os.cpu_count() or 1) - 1)
        self.logger.info(
            f"Featurizing {len(work_items):,} ESM3 PDBs with {num_workers} "
            f"CPU {self.featurize_backend} workers"
        )
        written = 0
        buffered_records: list[dict] = []
        buffered_failures: list[dict] = []
        if self.featurize_backend == "process":
            executor = ProcessPoolExecutor(
                max_workers=num_workers, initializer=_init_featurize_worker
            )
        else:
            executor = ThreadPoolExecutor(max_workers=num_workers)
        gpu_pool.start()
        try:
            for result in progress(
                self._featurize_unordered(executor, work_items, num_workers, gpu_pool),
                total=len(work_items),
                desc="Featurizing ESM3 structures",
            ):
                key = result.get("key")
                if key is not None and key.endswith("|esmfold2"):
                    if not _sequences_match(result["sequence"], expected_sequences[_raw_key(key)]):
                        result = {
                            "raw_key": key,
                            "canonical_key": key,
                            "pdb_path": result["pdb_path"],
                            "error": "ESMFold2 sequence differs from the dataset sequence",
                        }
                if "error" in result:
                    buffered_failures.append(result)
                else:
                    buffered_records.append(result)

                if len(buffered_records) >= self.feature_records_per_shard:
                    written += feature_store.put_records(
                        buffered_records, gpu_pool.submit_locations
                    )
                    buffered_records = []
                if len(buffered_failures) >= self.feature_records_per_shard:
                    feature_store.put_failures(buffered_failures)
                    buffered_failures = []
                gpu_pool.poll()
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown()

        written += feature_store.put_records(buffered_records, gpu_pool.submit_locations)
        feature_store.put_failures(buffered_failures)
        self.logger.info(f"Persisted {written:,} ESM3 feature records")
        return written

    def _featurize_unordered(
        self,
        executor,
        work_items: list[tuple],
        num_workers: int,
        gpu_pool: _ESM3GpuPool,
    ):
        """Yield featurize results as they finish, polling the GPU pool at least once a second.

        A slow structure delays only its own chunk. At most `4 * num_workers` chunks are in flight.
        """
        chunksize = 8 if self.featurize_backend == "process" else 1
        chunks = iter(
            [work_items[i : i + chunksize] for i in range(0, len(work_items), chunksize)]
        )
        pending: set = set()
        while True:
            while len(pending) < 4 * num_workers and (chunk := next(chunks, None)):
                pending.add(executor.submit(_featurize_chunk, chunk))
            if not pending:
                return
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            gpu_pool.poll()
            for future in done:
                yield from future.result()

    def _load_unique_input_keys(self) -> tuple[list[str], list[dict]]:
        raw_keys = load_structure_keys(self.input_path)
        self.logger.info(f"Loaded {len(raw_keys)} raw items from {self.input_path}")

        unique_keys: list[str] = []
        seen_keys: set[str] = set()
        failures: list[dict] = []
        for raw_key in progress(raw_keys, desc="Preparing ESM3 input keys", leave=False):
            if raw_key is None:
                failures.append(
                    {
                        "raw_key": raw_key,
                        "canonical_key": None,
                        "error": "Input key is None",
                    }
                )
                continue
            # The key names the structure source the accession resolves to now.
            resolved = self._resolve_pdb(str(raw_key))
            key = str(raw_key) if resolved is None else f"{raw_key}|{resolved[1]}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_keys.append(key)

        return unique_keys, failures

    def _dataset_sequences(self, raw_keys: set[str]) -> dict[str, str]:
        """Dataset sequence of each structure key found in the input protein records."""
        with open(self.input_path, "rb") as handle:
            payload = pickle.load(handle)
        return {
            key: item["sequence"]
            for item in payload
            if isinstance(item, dict)
            for key in item["pdbs"]
            if key in raw_keys
        }

    def _select_experimental_chains(self, keys: list[str]) -> dict[str, Optional[str]]:
        """Map each `<PDB>:<UniProt>` key to its selected mmCIF chain (None if none fits)."""
        pairs = [
            tuple(raw_key.split(":", 1))
            for raw_key in {_raw_key(key) for key in keys}
            if ":" in raw_key
        ]
        if not pairs:
            return {}
        sequences = {}
        for acc in {acc for _, acc in pairs}:
            path = Path(self.uniprot_acc_path) / f"{acc}.json"
            if path.exists() and (sequence := json.loads(path.read_text()).get("sequence")):
                sequences[acc] = sequence
        selections = select_chains(pairs, sequences, self.mmcif_path)
        self.logger.info(
            f"Selected mmCIF chains for {len(selections):,} (PDB, accession) pairs: "
            f"{dict(selection_counts(selections))}"
        )
        return {f"{pdb}:{acc}": chain for (pdb, acc), (chain, _, _) in selections.items()}

    def _resolve_pdb(self, acc: str) -> Optional[tuple[str, str]]:
        """Return (path, source) for a key; source is alphafold, esmfold2, esm or experimental.

        An accession resolves to a direct file stem, then AlphaFold, ESMFold2 and ESM3, in that order.

        An experimental key is `<PDB>:<UniProt>`; its path is the mmCIF file.
        """
        if ":" in acc:
            path = mmcif_file(self.mmcif_path, acc.partition(":")[0])
            return None if path is None else (str(path), "experimental")
        index = self._get_pdb_index()
        if acc in index["direct"]:
            path, source = index["direct"][acc]
            return str(path), source
        if acc in index["alphafold_alias"]:
            return str(index["alphafold_alias"][acc]), "alphafold"
        if acc in index["esmfold2"]:
            return str(index["esmfold2"][acc]), "esmfold2"
        if acc in index["esm_alias"]:
            return str(index["esm_alias"][acc]), "esm"
        return None

    def _get_pdb_index(self) -> dict[str, dict]:
        if self._pdb_index is not None:
            return self._pdb_index

        alphafold_dir = Path(self.alphafold_pdb_path)
        esm3_dir = Path(self.esm3_pdb_path)

        direct: dict[str, tuple[Path, str]] = {}
        alphafold_alias: dict[str, Path] = {}
        alphafold_versions: dict[str, int] = {}
        esm_alias: dict[str, Path] = {}

        for pdb_path in alphafold_dir.glob("*.pdb"):
            direct[pdb_path.stem] = (pdb_path, "alphafold")
            parsed = self._parse_alphafold_stem(pdb_path.stem)
            if parsed is None:
                continue
            alias, version = parsed
            old_version = alphafold_versions.get(alias)
            if old_version is None or version > old_version:
                alphafold_versions[alias] = version
                alphafold_alias[alias] = pdb_path

        for pdb_path in esm3_dir.glob("*.pdb"):
            direct[pdb_path.stem] = (pdb_path, "esm")
            if pdb_path.stem.startswith("ESM3-open-small-"):
                esm_alias[pdb_path.stem.removeprefix("ESM3-open-small-")] = pdb_path

        esmfold2 = {
            path.stem.removeprefix("ESMFold2-"): path
            for path in Path(self.esmfold2_cif_path).glob("ESMFold2-*.cif")
        }

        self._pdb_index = {
            "esmfold2": esmfold2,
            "direct": direct,
            "alphafold_alias": alphafold_alias,
            "esm_alias": esm_alias,
        }
        self.logger.info(
            "Indexed ESM3 PDB paths: "
            f"{len(direct):,} direct stems, "
            f"{len(alphafold_alias):,} AlphaFold aliases, "
            f"{len(esm_alias):,} ESM aliases, "
            f"{len(esmfold2):,} ESMFold2 structures"
        )
        return self._pdb_index

    @staticmethod
    def _parse_alphafold_stem(stem: str) -> Optional[tuple[str, int]]:
        if not stem.startswith("AF-") or "-model_v" not in stem:
            return None
        try:
            alias = stem.removeprefix("AF-").split("-F1-model_v")[0]
            version = int(stem.split("_v")[-1])
        except (IndexError, ValueError):
            return None
        return alias, version
