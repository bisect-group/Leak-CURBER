from __future__ import annotations

import os
import pickle
import queue
import shutil
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.data.components.embedders.base import BaseShardEmbedder
from src.data.components.embedders.utils import load_pickle_items, sha256_short


def _suppress_esm3_warnings() -> None:
    warnings.filterwarnings("ignore")


_suppress_esm3_warnings()


def _featurize_esm3_record(args: tuple[str, str]) -> dict:
    _suppress_esm3_warnings()
    key, pdb_path = args
    try:
        from esm.sdk.api import ESMProtein
        from esm.utils.structure.protein_chain import ProteinChain

        protein = ESMProtein.from_pdb(Path(pdb_path))
        if protein.sequence is None or protein.coordinates is None:
            return {
                "raw_key": key,
                "canonical_key": key,
                "pdb_path": pdb_path,
                "error": "Parsed protein is missing sequence or coordinates",
            }

        chain = ProteinChain.from_atom37(protein.coordinates, sequence=protein.sequence)
        coords, _plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        residue_index = (
            residue_index.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)
        )
        return {
            "key": key,
            "key_hash": sha256_short(key),
            "pdb_path": pdb_path,
            "sequence": protein.sequence,
            "length": len(protein.sequence),
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


def _esm3_gpu_worker(
    rank: int,
    max_batch_tokens: int,
    max_batch_size: int,
    task_queue,
    result_queue,
) -> None:
    _suppress_esm3_warnings()
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import LogitsConfig
        from esm.utils import encoding
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        from esm.utils.misc import stack_variable_length_tensors
        from esm.utils.sampling import _BatchedESMProteinTensor

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

                    with torch.inference_mode():
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

                    arrays_by_key = {}
                    for idx, key in enumerate(keys):
                        seq_len = lengths[idx] + 2
                        arrays_by_key[key] = (
                            logits_output.embeddings[idx, :seq_len]
                            .mean(dim=0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float16, copy=False)
                        )
                    result_queue.put(("batch", rank, arrays_by_key, []))
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
                finally:
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
        for shard_id, shard_path in tqdm(
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

        for shard_id, shard_locations in tqdm(
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

    def put_records(self, records: list[dict]) -> int:
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
        appended_df.to_parquet(tmp_path, index=False)
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
        self.processed_exp_pdb_path = cfg.embeddings.processed_exp_pdb_path
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
        self.max_batch_tokens = 4_096
        self.max_batch_size = 8
        self.parse_workers = max(1, (os.cpu_count() or 1) // max(1, len(self.gpu_ids)))
        self.parse_chunk_size = 256
        self._pdb_index: Optional[dict[str, dict[str, Path]]] = None

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
            key_type="protein_accession",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=10_000,
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
            f"max_batch_size={self.max_batch_size}"
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
            self.logger.info("No new ESM3 items to compute")
            return

        feature_store = ESM3FeatureStore(
            self.featurized_dir,
            self.feature_records_per_shard,
            self.logger,
        )
        feature_locations = feature_store.find_feature_locations_for_keys(missing_keys)
        self.featurize_missing(
            missing_keys,
            feature_store,
            existing_feature_hashes=set(feature_locations),
        )
        feature_failures = feature_store.failure_dicts_for_keys(missing_keys)

        feature_locations = feature_store.find_feature_locations_for_keys(missing_keys)
        ready_keys = [
            key for key in missing_keys if sha256_short(key) in feature_locations
        ]
        self.logger.info(
            f"{len(ready_keys)} ESM3 feature records ready for GPU embedding; "
            f"{len(feature_failures)} feature failures recorded"
        )

        total_written = 0
        embed_failures: list[dict] = []
        if ready_keys:
            total_written, embed_failures = self.embed_featurized(
                ready_keys,
                feature_store,
                feature_locations=feature_locations,
            )
        else:
            self.logger.warning("No ESM3 feature records are available for embedding")

        self.store.record_failures(input_failures + feature_failures + embed_failures)
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
        existing_feature_hashes: Optional[set[str]] = None,
    ) -> int:
        if existing_feature_hashes is None:
            existing_feature_hashes = set(
                feature_store.find_feature_locations_for_keys(missing_keys)
            )
        keys_to_featurize = [
            key for key in missing_keys if sha256_short(key) not in existing_feature_hashes
        ]
        if not keys_to_featurize:
            self.logger.info("All missing ESM3 keys already have feature records")
            return 0

        self.logger.info(
            f"Resolving PDB paths for {len(keys_to_featurize):,} ESM3 keys"
        )
        work_items = []
        missing_pdb_failures = []
        for key in tqdm(keys_to_featurize, desc="Resolving ESM3 PDB paths", leave=False):
            pdb_path = self._resolve_pdb_path(key)
            if pdb_path is None:
                missing_pdb_failures.append(
                    {
                        "raw_key": key,
                        "canonical_key": key,
                        "pdb_path": None,
                        "error": "No PDB file found",
                    }
                )
            else:
                work_items.append((key, pdb_path))

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
        executor_cls = (
            ProcessPoolExecutor
            if self.featurize_backend == "process"
            else ThreadPoolExecutor
        )
        with executor_cls(max_workers=num_workers) as executor:
            mapped_results = executor.map(
                _featurize_esm3_record,
                work_items,
                chunksize=8 if self.featurize_backend == "process" else 1,
            )
            for result in tqdm(
                mapped_results,
                total=len(work_items),
                desc="Featurizing ESM3 structures",
            ):
                if "error" in result:
                    buffered_failures.append(result)
                else:
                    buffered_records.append(result)

                if len(buffered_records) >= self.feature_records_per_shard:
                    written += feature_store.put_records(buffered_records)
                    buffered_records = []
                if len(buffered_failures) >= self.feature_records_per_shard:
                    feature_store.put_failures(buffered_failures)
                    buffered_failures = []

        written += feature_store.put_records(buffered_records)
        feature_store.put_failures(buffered_failures)
        self.logger.info(f"Persisted {written:,} ESM3 feature records")
        return written

    def embed_featurized(
        self,
        ready_keys: list[str],
        feature_store: ESM3FeatureStore,
        *,
        feature_locations: Optional[dict[str, dict]] = None,
    ) -> tuple[int, list[dict]]:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available for ESM3 structure embedding.")

        if feature_locations is None:
            feature_locations = feature_store.find_feature_locations_for_keys(ready_keys)

        ready_hashes = [sha256_short(key) for key in ready_keys]
        ready_locations = [
            feature_locations[key_hash]
            for key_hash in ready_hashes
            if key_hash in feature_locations
        ]
        if not ready_locations:
            return 0, []

        locations_by_shard: dict[int, list[dict]] = {}
        for location in ready_locations:
            locations_by_shard.setdefault(int(location["shard_id"]), []).append(location)

        total_ready = len(ready_locations)
        self.logger.info(
            f"Embedding {total_ready:,} featurized ESM3 structures "
            f"across {num_gpus} GPU(s)"
        )

        ctx = mp.get_context("fork")
        result_queue = ctx.Queue(maxsize=max(16, num_gpus * 4))
        task_queues = [ctx.Queue(maxsize=1) for _ in range(num_gpus)]
        processes = []
        next_task_id = 0
        try:
            for rank in range(num_gpus):
                process = ctx.Process(
                    target=_esm3_gpu_worker,
                    args=(
                        rank,
                        self.max_batch_tokens,
                        self.max_batch_size,
                        task_queues[rank],
                        result_queue,
                    ),
                )
                process.start()
                processes.append(process)

            total_written = 0
            failures: list[dict] = []
            done_workers = 0
            fatal_errors = []

            def collect_results_until(task_ids: set[int]) -> None:
                nonlocal total_written, done_workers
                completed_task_ids: set[int] = set()
                while completed_task_ids != task_ids:
                    try:
                        message = result_queue.get(timeout=1.0)
                    except queue.Empty:
                        for process in processes:
                            if process.exitcode not in (None, 0):
                                fatal_errors.append(
                                    f"GPU worker {process.pid} exited with "
                                    f"code {process.exitcode}"
                                )
                        if fatal_errors:
                            return
                        continue

                    kind, rank, payload, extra = message
                    if kind == "batch":
                        arrays_by_key = payload
                        batch_failures = extra or []
                        total_written += self.store.put_many(arrays_by_key)
                        failures.extend(batch_failures)
                        feature_progress_bar.update(
                            len(arrays_by_key) + len(batch_failures)
                        )
                    elif kind == "task_done":
                        completed_task_ids.add(int(payload))
                    elif kind == "done":
                        done_workers += 1
                    elif kind == "fatal":
                        fatal_errors.append(f"GPU worker {rank} failed: {payload}\n{extra}")
                        return

            with tqdm(
                total=total_ready,
                desc="Embedding ESM3 features",
                leave=True,
            ) as feature_progress_bar:
                for shard_id, shard_locations in tqdm(
                    sorted(locations_by_shard.items()),
                    desc="Embedding ESM3 feature shards",
                    leave=True,
                ):
                    shard_locations = sorted(
                        shard_locations,
                        key=lambda location: int(location["length"]),
                    )
                    task_ids = set()
                    for rank in range(num_gpus):
                        rank_locations = shard_locations[rank::num_gpus]
                        if not rank_locations:
                            continue
                        task_id = next_task_id
                        next_task_id += 1
                        task_ids.add(task_id)
                        task_queues[rank].put(
                            {
                                "task_id": task_id,
                                "shard_id": shard_id,
                                "shard_path": rank_locations[0]["shard_path"],
                                "rows": [
                                    {
                                        "row_idx": int(location["row_idx"]),
                                        "key": location["key"],
                                    }
                                    for location in rank_locations
                                ],
                            }
                        )

                    collect_results_until(task_ids)
                    if fatal_errors:
                        break

            if fatal_errors:
                raise RuntimeError("\n".join(fatal_errors))

            for task_queue in task_queues:
                task_queue.put(None)

            while done_workers < num_gpus:
                try:
                    message = result_queue.get(timeout=1.0)
                except queue.Empty:
                    for process in processes:
                        if process.exitcode not in (None, 0):
                            raise RuntimeError(
                                f"GPU worker {process.pid} exited with code "
                                f"{process.exitcode}"
                            )
                    continue

                kind, rank, payload, extra = message
                if kind == "done":
                    done_workers += 1
                elif kind == "fatal":
                    raise RuntimeError(f"GPU worker {rank} failed: {payload}\n{extra}")

            for process in processes:
                process.join()
                if process.exitcode not in (0, None):
                    raise RuntimeError(
                        f"GPU worker {process.pid} exited with code {process.exitcode}"
                    )

            total_written += self.store.put_many({}, flush=True)
            return total_written, failures
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            for task_queue in task_queues:
                task_queue.close()
            result_queue.close()

    def _load_unique_input_keys(self) -> tuple[list[str], list[dict]]:
        raw_keys = load_pickle_items(self.input_path, self.key_field)
        self.logger.info(f"Loaded {len(raw_keys)} raw items from {self.input_path}")

        unique_keys: list[str] = []
        seen_keys: set[str] = set()
        failures: list[dict] = []
        for raw_key in tqdm(raw_keys, desc="Preparing ESM3 input keys", leave=False):
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

        return unique_keys, failures

    def _resolve_pdb_path(self, acc: str) -> Optional[str]:
        index = self._get_pdb_index()
        if acc in index["direct"]:
            return str(index["direct"][acc])
        if acc in index["alphafold_alias"]:
            return str(index["alphafold_alias"][acc])
        if acc in index["esm_alias"]:
            return str(index["esm_alias"][acc])
        return None

    def _get_pdb_index(self) -> dict[str, dict[str, Path]]:
        if self._pdb_index is not None:
            return self._pdb_index

        alphafold_dir = Path(self.alphafold_pdb_path)
        esm3_dir = Path(self.esm3_pdb_path)
        processed_exp_pdb_dir = Path(self.processed_exp_pdb_path)

        direct: dict[str, Path] = {}
        alphafold_alias: dict[str, Path] = {}
        alphafold_versions: dict[str, int] = {}
        esm_alias: dict[str, Path] = {}

        for pdb_path in processed_exp_pdb_dir.glob("*.pdb"):
            direct[pdb_path.stem] = pdb_path

        for pdb_path in alphafold_dir.glob("*.pdb"):
            direct[pdb_path.stem] = pdb_path
            parsed = self._parse_alphafold_stem(pdb_path.stem)
            if parsed is None:
                continue
            alias, version = parsed
            old_version = alphafold_versions.get(alias)
            if old_version is None or version > old_version:
                alphafold_versions[alias] = version
                alphafold_alias[alias] = pdb_path

        for pdb_path in esm3_dir.glob("*.pdb"):
            direct[pdb_path.stem] = pdb_path
            if pdb_path.stem.startswith("ESM3-open-small-"):
                esm_alias[pdb_path.stem.removeprefix("ESM3-open-small-")] = pdb_path

        self._pdb_index = {
            "direct": direct,
            "alphafold_alias": alphafold_alias,
            "esm_alias": esm_alias,
        }
        self.logger.info(
            "Indexed ESM3 PDB paths: "
            f"{len(direct):,} direct stems, "
            f"{len(alphafold_alias):,} AlphaFold aliases, "
            f"{len(esm_alias):,} ESM aliases"
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
