"""Multiprocess, multi-GPU atom mapping for reaction SMILES.

The LocalMapper neural model is deliberately used without modifying its library
implementation.  RDKit/DGL graph construction and template extraction run in
CPU pools while persistent GPU workers perform batched inference.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pickle
import re
import signal
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
from typing import Any, Iterable

import pandas as pd
from src.utils.rich_progress import progress, write


_MAP_TAG = re.compile(r":\d+\]")
_PREPROCESS_GRAPH = None
_POST_ACCEPTED_TEMPLATES = None
_POST_MODEL_VERSION = None
_POST_TRIP: str | None = None
_POST_START = 0.0
_POST_START_RSS = 0
# Template extraction can enumerate factorially many atom orders on large reactions and
# reach tens of GB.  A row that passes either limit is reported as failed.
MAX_ROW_RSS_BYTES = 8 << 30
MAX_ROW_SECONDS = 120.0
_GUARD_INTERVAL_SECONDS = 0.5
_RUNAWAY_PREFIX = "RunawayReaction"


class _Runaway(Exception):
    pass


def _cfg(value: Any, key: str, default: Any = None) -> Any:
    if hasattr(value, "get"):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_reusable_row(row: dict[str, Any], model_version: str) -> bool:
    status = row.get("atom_mapping_status")
    return status == "preserved" or (
        status == "mapped" and row.get("atom_mapping_model_version") == model_version
    )


def _reaction_hash(rxn: str) -> str:
    return hashlib.sha256(rxn.encode("utf-8")).hexdigest()


def _unique_non_null_reactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable unique reaction keys without changing the input frame."""
    return (
        frame.loc[frame["rxn_smiles"].notna(), ["rxn_smiles"]]
        .astype({"rxn_smiles": str})
        .drop_duplicates(subset=["rxn_smiles"], keep="first")
        .reset_index(drop=True)
    )


def _base_result(row_id: int, rxn: str) -> dict[str, Any]:
    return {
        "_atom_mapping_row_id": row_id,
        "rxn_smiles": rxn,
        "atom_mapped_rxn_smiles": None,
        "atom_mapping_template": None,
        "atom_mapping_confident": None,
        "atom_mapping_source": None,
        "atom_mapping_status": None,
        "atom_mapping_error": None,
        "atom_mapping_input_hash": _reaction_hash(rxn),
        "atom_mapping_model_version": None,
    }


def _preprocess_init() -> None:
    global _PREPROCESS_GRAPH
    from localmapper.utils import init_featurizer

    _PREPROCESS_GRAPH = init_featurizer()[2]


def _preprocess_one(task: tuple[int, str]) -> dict[str, Any]:
    """Parse one reaction and return either a preserved result or DGL graphs."""
    from rdkit import Chem

    row_id, rxn = task
    result = _base_result(row_id, rxn)
    try:
        sides = rxn.split(">>")
        if len(sides) != 2 or not all(sides):
            raise ValueError("reaction must contain exactly one non-empty >> separator")
        reactant, product = (Chem.MolFromSmiles(side) for side in sides)
        if reactant is None or product is None:
            raise ValueError("RDKit could not parse one reaction side")

        if _MAP_TAG.search(rxn):
            result.update(
                atom_mapped_rxn_smiles=rxn,
                atom_mapping_source="input",
                atom_mapping_status="preserved",
            )
            return {"kind": "result", "result": result}

        return {
            "kind": "graphs",
            "row_id": row_id,
            "rxn": rxn,
            "rgraph": _PREPROCESS_GRAPH(reactant),
            "pgraph": _PREPROCESS_GRAPH(product),
            "r_atoms": reactant.GetNumAtoms(),
            "p_atoms": product.GetNumAtoms(),
        }
    except Exception as exc:  # keep one bad row from stopping a long run
        result.update(atom_mapping_status="failed", atom_mapping_error=f"{type(exc).__name__}: {exc}")
        return {"kind": "result", "result": result}


def _gpu_worker(device_id: int, input_queue, post_queue, model_version: str) -> None:
    import torch
    from localmapper import localmapper

    torch.set_num_threads(1)
    mapper = localmapper(device=f"cuda:{device_id}", model_version=model_version)

    def infer_batch(batch):
        try:
            probabilities = mapper.pred_pxr_graphs(
                [item["rgraph"] for item in batch],
                [item["pgraph"] for item in batch],
            )
            for item, probabilities_for_item in zip(batch, probabilities):
                post_queue.put({"kind": "prediction", "row_id": item["row_id"], "rxn": item["rxn"], "prediction": probabilities_for_item})
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(batch) == 1:
                for item in batch:
                    post_queue.put({"kind": "inference_error", "row_id": item["row_id"], "rxn": item["rxn"], "error": f"{type(exc).__name__}: {exc}"})
                return
            midpoint = max(1, len(batch) // 2)
            torch.cuda.empty_cache()
            infer_batch(batch[:midpoint])
            infer_batch(batch[midpoint:])
        except Exception as exc:
            for item in batch:
                post_queue.put({"kind": "inference_error", "row_id": item["row_id"], "rxn": item["rxn"], "error": f"{type(exc).__name__}: {exc}"})

    while True:
        batch = input_queue.get()
        if batch is None:
            return
        infer_batch(batch)


def _rss_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _guard_tick(signum, frame) -> None:
    """Interrupt the current reaction once it exceeds the memory or time limit.

    Memory is the growth in RSS since the reaction started, so a worker that stays large
    after an earlier reaction does not fail every later one.

    localmapper wraps template extraction in a bare ``except``, which can swallow the
    exception raised here; ``_POST_TRIP`` lets the caller detect the trip afterwards.
    """
    global _POST_TRIP
    growth = _rss_bytes() - _POST_START_RSS
    elapsed = time.monotonic() - _POST_START
    if growth > MAX_ROW_RSS_BYTES or elapsed > MAX_ROW_SECONDS:
        _POST_TRIP = f"{_RUNAWAY_PREFIX}: rss grew {growth / 2**30:.1f} GiB in {elapsed:.0f} s"
        raise _Runaway(_POST_TRIP)


def _postprocess_init(model_version: str) -> None:
    global _POST_ACCEPTED_TEMPLATES, _POST_MODEL_VERSION
    _POST_MODEL_VERSION = model_version
    signal.signal(signal.SIGALRM, _guard_tick)
    from importlib.resources import files

    path = Path(files("localmapper") / "data" / f"templates_{model_version}.pkl")
    with path.open("rb") as handle:
        _POST_ACCEPTED_TEMPLATES = pickle.load(handle)


def _postprocess_one(item: dict[str, Any]) -> dict[str, Any]:
    if item["kind"] == "result":
        return item["result"]
    result = _base_result(item["row_id"], item["rxn"])
    result["atom_mapping_model_version"] = _POST_MODEL_VERSION
    if item["kind"] == "inference_error":
        result.update(atom_mapping_status="failed", atom_mapping_error=item["error"])
        return result

    from localmapper.mapper import prediction2map

    global _POST_TRIP, _POST_START, _POST_START_RSS
    _POST_TRIP, _POST_START, _POST_START_RSS = None, time.monotonic(), _rss_bytes()
    try:
        signal.setitimer(signal.ITIMER_REAL, _GUARD_INTERVAL_SECONDS, _GUARD_INTERVAL_SECONDS)
        mapped = prediction2map(item["rxn"], item["prediction"])
        if _POST_TRIP:
            raise _Runaway(_POST_TRIP)
        confident = mapped["template"] in _POST_ACCEPTED_TEMPLATES
        if not confident:
            mapped = prediction2map(item["rxn"], item["prediction"], neighbor_weight=90)
            if _POST_TRIP:
                raise _Runaway(_POST_TRIP)
            confident = mapped["template"] in _POST_ACCEPTED_TEMPLATES
        result.update(
            atom_mapped_rxn_smiles=mapped["mapped_rxn"],
            atom_mapping_template=mapped["template"],
            atom_mapping_confident=confident,
            atom_mapping_source="localmapper",
            atom_mapping_status="mapped",
        )
    except _Runaway as exc:
        result.update(atom_mapping_status="failed", atom_mapping_error=str(exc))
    except Exception as exc:
        result.update(atom_mapping_status="failed", atom_mapping_error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if _POST_TRIP:
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # give the runaway's freed heap back to the OS
    return result


def _put_while_alive(queue, item, processes) -> None:
    # A blocking put would hang if every GPU worker had died and the queue was full.
    while True:
        try:
            return queue.put(item, timeout=30)
        except Full:
            if not any(process.is_alive() for process in processes):
                raise RuntimeError("All atom-mapping GPU workers stopped; the input queue is full.")


def _queue_iter(queue, sentinel):
    while True:
        item = queue.get()
        if item == sentinel:
            return
        yield item


@dataclass(frozen=True)
class AtomMappingConfig:
    gpu_ids: tuple[int, ...] = (0, 1)
    model_version: str = "202403"
    batch_size: int = 32
    preprocess_workers: int | None = None
    postprocess_workers: int | None = None
    queue_size: int = 4
    checkpoint_interval: int = 1000
    chunksize: int = 1


class AtomMappedReactionGenerator:
    def __init__(self, cfg: Any | None = None, *, options: AtomMappingConfig | None = None):
        cfg = cfg or {}
        emb = _cfg(cfg, "embeddings", cfg)
        gpu_ids = _cfg(emb, "localmapper_gpu_ids", _cfg(emb, "gpu_ids", [0, 1]))
        self.options = options or AtomMappingConfig(
            gpu_ids=tuple(int(x) for x in gpu_ids),
            model_version=str(_cfg(emb, "localmapper_model_version", "202403")),
            batch_size=int(_cfg(emb, "localmapper_batch_size", 32)),
            preprocess_workers=_cfg(emb, "localmapper_preprocess_workers", None),
            postprocess_workers=_cfg(emb, "localmapper_postprocess_workers", None),
            queue_size=int(_cfg(emb, "localmapper_queue_size", 4)),
            checkpoint_interval=int(_cfg(emb, "localmapper_checkpoint_interval", 1000)),
        )
        if not self.options.gpu_ids:
            raise ValueError("At least one GPU ID is required")

    def generate(self, input_path: str | Path, output_path: str | Path | None = None, *, resume: bool = False, limit: int | None = None) -> pd.DataFrame:
        input_path = Path(input_path)
        output_path = Path(output_path or input_path.with_name(f"{input_path.stem}_atom_mapped.parquet"))
        frame = pd.read_parquet(input_path)
        if "rxn_smiles" not in frame.columns:
            raise KeyError(f"Input dataset {input_path} must contain rxn_smiles")
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        frame = frame.reset_index(drop=True)

        mapping_columns = [
            "atom_mapped_rxn_smiles",
            "atom_mapping_template",
            "atom_mapping_confident",
            "atom_mapping_source",
            "atom_mapping_status",
            "atom_mapping_error",
            "atom_mapping_input_hash",
            "atom_mapping_model_version",
        ]

        # Inference is keyed by exact reaction SMILES.  The original frame is
        # retained separately so duplicate rows can be restored after mapping.
        unique_reactions = _unique_non_null_reactions(frame)

        existing: dict[str, dict[str, Any]] = {}
        # The final file wins over the .partial.parquet checkpoint. A "mapped" row is reused
        # only when its model version matches; "failed" and None/NaN (pending) rows are not.
        # "preserved" rows carry no model version and are reused on status alone.
        if resume:
            for resume_path in (output_path.with_suffix(".partial.parquet"), output_path):
                if not resume_path.exists():
                    continue
                prior = pd.read_parquet(resume_path)
                if {"rxn_smiles", "atom_mapping_status"}.issubset(prior.columns):
                    existing.update(
                        {
                            str(row["rxn_smiles"]): row
                            for row in prior.to_dict(orient="records")
                            if _is_reusable_row(row, self.options.model_version)
                        }
                    )

        results_by_rxn: dict[str, dict[str, Any]] = {}
        for rxn in unique_reactions["rxn_smiles"]:
            if rxn in existing:
                cached = {
                    key: existing[rxn].get(key)
                    for key in ["rxn_smiles", *mapping_columns]
                }
                cached["rxn_smiles"] = rxn
                results_by_rxn[rxn] = cached

        tasks = [
            (idx, rxn)
            for idx, rxn in enumerate(unique_reactions["rxn_smiles"])
            if rxn not in results_by_rxn
        ]

        def fanout_results(unique_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
            mapping = pd.DataFrame(
                [
                    {"rxn_smiles": rxn, **{key: result.get(key) for key in mapping_columns}}
                    for rxn, result in unique_results.items()
                ]
            )
            if not mapping.empty and mapping["rxn_smiles"].duplicated().any():
                raise RuntimeError("Unique atom-mapping results contain duplicate reaction keys")

            enriched = frame.drop(columns=[key for key in mapping_columns if key in frame.columns])
            enriched["_atom_mapping_row_order"] = range(len(enriched))
            if mapping.empty:
                for key in mapping_columns:
                    enriched[key] = None
            else:
                enriched = enriched.merge(
                    mapping,
                    on="rxn_smiles",
                    how="left",
                    sort=False,
                    validate="many_to_one",
                )
            null_mask = enriched["rxn_smiles"].isna()
            enriched.loc[null_mask, "atom_mapping_status"] = "not_applicable"
            enriched = enriched.sort_values("_atom_mapping_row_order", kind="stable")
            enriched = enriched.drop(columns=["_atom_mapping_row_order"])
            if len(enriched) != len(frame):
                raise RuntimeError("Atom-mapping fan-out changed the input row count")
            enriched_rxns = enriched["rxn_smiles"].astype("string").reset_index(drop=True)
            frame_rxns = frame["rxn_smiles"].astype("string").reset_index(drop=True)
            if not enriched_rxns.equals(frame_rxns):
                raise RuntimeError("Atom-mapping fan-out changed reaction row order")
            temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
            enriched.to_parquet(
                temporary_output,
                index=False,
                compression="zstd",
                compression_level=3,
            )
            os.replace(temporary_output, output_path)
            return enriched

        if not tasks:
            return fanout_results(results_by_rxn)

        import multiprocessing as mp
        from multiprocessing.pool import Pool

        ctx = mp.get_context("spawn")
        gpu_queues = [ctx.Queue(maxsize=self.options.queue_size) for _ in self.options.gpu_ids]
        post_queue = ctx.Queue(maxsize=max(self.options.queue_size * len(gpu_queues), 4))
        sentinel = ("__ATOM_MAPPING_DONE__", id(post_queue))
        default_cpu_workers = max(1, ((os.cpu_count() or 2) - len(gpu_queues)) // 2)
        pre_workers = self.options.preprocess_workers or default_cpu_workers
        post_workers = self.options.postprocess_workers or default_cpu_workers
        gpu_processes = [ctx.Process(target=_gpu_worker, args=(gpu, queue, post_queue, self.options.model_version)) for gpu, queue in zip(self.options.gpu_ids, gpu_queues)]
        for process in gpu_processes:
            process.start()

        def feed() -> None:
            buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            gpu_index = 0
            buffered = 0
            with ctx.Pool(pre_workers, initializer=_preprocess_init) as pool:
                for item in pool.imap_unordered(_preprocess_one, tasks, chunksize=self.options.chunksize):
                    if item["kind"] == "result":
                        post_queue.put(item)
                        continue
                    key = (item["p_atoms"] // 16, item["r_atoms"] // 16)
                    buckets[key].append(item)
                    buffered += 1
                    if len(buckets[key]) >= self.options.batch_size:
                        _put_while_alive(gpu_queues[gpu_index % len(gpu_queues)], buckets[key][: self.options.batch_size], gpu_processes)
                        del buckets[key][: self.options.batch_size]
                        gpu_index += 1
                        buffered -= self.options.batch_size
                    while buffered > self.options.batch_size * len(gpu_queues) * self.options.queue_size:
                        flush_key = max(buckets, key=lambda k: len(buckets[k]))
                        batch = buckets[flush_key][: self.options.batch_size]
                        del buckets[flush_key][: len(batch)]
                        _put_while_alive(gpu_queues[gpu_index % len(gpu_queues)], batch, gpu_processes)
                        gpu_index += 1
                        buffered -= len(batch)
            for bucket in buckets.values():
                for start in range(0, len(bucket), self.options.batch_size):
                    _put_while_alive(gpu_queues[gpu_index % len(gpu_queues)], bucket[start : start + self.options.batch_size], gpu_processes)
                    gpu_index += 1
            for queue in gpu_queues:
                _put_while_alive(queue, None, gpu_processes)
            for process in gpu_processes:
                process.join()

        feed_errors: list[BaseException] = []

        def run_feed() -> None:
            try:
                feed()
            except BaseException as exc:
                feed_errors.append(exc)
            finally:
                # The sentinel always ends the result loop, so a feeder error cannot hang it.
                post_queue.put(sentinel)

        feeder = threading.Thread(target=run_feed, daemon=True)
        feeder.start()
        runaways = 0
        with ctx.Pool(post_workers, initializer=_postprocess_init, initargs=(self.options.model_version,)) as pool:
            for result in progress(pool.imap_unordered(_postprocess_one, _queue_iter(post_queue, sentinel)), total=len(tasks), desc="Atom mapping"):
                results_by_rxn[result["rxn_smiles"]] = result
                if str(result["atom_mapping_error"]).startswith(_RUNAWAY_PREFIX):
                    runaways += 1
                    write(
                        f"Atom mapping stopped runaway reaction (row {result['_atom_mapping_row_id']}, "
                        f"{len(result['rxn_smiles'])} chars): {result['atom_mapping_error']}"
                    )
                if self.options.checkpoint_interval and len(results_by_rxn) % self.options.checkpoint_interval == 0:
                    pd.DataFrame(results_by_rxn.values()).to_parquet(
                        output_path.with_suffix(".partial.parquet"), index=False,
                        compression="zstd", compression_level=3,
                    )
        feeder.join()
        if runaways:
            write(
                f"Atom mapping left {runaways:,} runaway reactions unmapped; resume retries them."
            )
        if feed_errors:
            raise RuntimeError("Atom mapping stopped early") from feed_errors[0]
        # A lost result would leave NaN rows cached as done; abort instead.
        missing = [rxn for _, rxn in tasks if rxn not in results_by_rxn]
        if missing:
            pd.DataFrame(results_by_rxn.values()).to_parquet(
                output_path.with_suffix(".partial.parquet"), index=False,
                compression="zstd", compression_level=3,
            )
            raise RuntimeError(
                f"Atom mapping returned no result for {len(missing):,} of {len(tasks):,} reactions; "
                "the checkpoint was saved. Rerun with resume."
            )
        return fanout_results(results_by_rxn)
