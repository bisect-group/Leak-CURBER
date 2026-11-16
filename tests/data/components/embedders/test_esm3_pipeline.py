from __future__ import annotations

import logging
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.components.embedders import esm3
from src.data.components.embedders.esm3 import (
    ESM3FeatureStore,
    ESM3StructureShardEmbedder,
    _init_featurize_worker,
    _sequences_match,
)
from src.data.components.embedders.store import ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import PDB_SOURCE_KEY_TYPE, sha256_short

STATE: dict = {}


def _record(key: str) -> dict:
    return {
        "key": key,
        "key_hash": sha256_short(key),
        "pdb_path": "x.pdb",
        "sequence": "AAA",
        "length": 3,
        "coords": np.zeros((3, 37, 3), dtype=np.float32),
        "residue_index": np.arange(3),
    }


def _log(line: str) -> None:
    with open(STATE["log"], "a") as handle:
        handle.write(f"{line} {time.time()}\n")


def _stub_featurize(args: tuple) -> dict:
    key, path, _chain = args
    time.sleep(3 if key.startswith("slow") else 0.1)
    _log(f"feat {key}")
    if key.startswith("bad"):
        return {"raw_key": key, "canonical_key": key, "pdb_path": path, "error": "boom"}
    return _record(key)


def _stub_gpu_worker(
    rank, max_batch_tokens, max_batch_size, task_queue, result_queue, autocast_dtype
) -> None:
    while (task := task_queue.get()) is not None:
        _log(f"embed {task['shard_id']}")
        if STATE.get("crash"):
            raise RuntimeError("GPU worker crashed")
        records = esm3._load_feature_task_records(task["shard_path"], task["rows"])
        arrays = {r["key"]: np.full(4, task["shard_id"], dtype=np.float32) for r in records}
        failures = []
        for key in [k for k in arrays if "embedfail" in k]:
            del arrays[key]
            failures.append({"raw_key": key, "canonical_key": key, "error": "oom"})
        result_queue.put(("batch", rank, arrays, failures))
        result_queue.put(("task_done", rank, task["task_id"], len(records)))
    result_queue.put(("done", rank, 0, None))


@pytest.fixture(autouse=True)
def _stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    STATE.clear()
    STATE["log"] = tmp_path / "events.log"
    STATE["log"].touch()
    monkeypatch.setattr(esm3, "_featurize_esm3_record", _stub_featurize)
    monkeypatch.setattr(esm3, "_esm3_gpu_worker", _stub_gpu_worker)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    signal.signal(signal.SIGALRM, lambda *_: pytest.fail("timeout: embed hung"))
    signal.alarm(60)
    yield
    signal.alarm(0)


def _embedder(tmp_path: Path, keys: list[str]) -> ESM3StructureShardEmbedder:
    embedder = object.__new__(ESM3StructureShardEmbedder)
    embedder.logger = MagicMock()
    embedder.featurized_dir = tmp_path / "features"
    embedder.keep_featurized_after_success = True
    embedder.reclaim_last_underfilled_shard = False
    embedder.retry_failed = False
    embedder.featurize_workers = 1
    embedder.featurize_backend = "thread"
    embedder.feature_records_per_shard = 2
    embedder.max_batch_tokens, embedder.max_batch_size = 4096, 8
    embedder.autocast_dtype = "bfloat16"
    metadata = CacheMetadata(
        embedder="t",
        model_name="m",
        version="v1",
        storage_dtype="float32",
        embedding_dim=None,
        max_shard_bytes=1 << 20,
        key_type=PDB_SOURCE_KEY_TYPE,
        key_field="acc_id",
    )
    embedder.store = ShardedEmbeddingStore(tmp_path / "store", metadata, logging.getLogger("t"))
    embedder._load_unique_input_keys = lambda: (keys, [])
    embedder._select_experimental_chains = lambda _keys: {}
    embedder._resolve_pdb = lambda raw: None if raw.startswith("nopdb") else ("x.pdb", "alphafold")
    embedder.recorded = []
    embedder.store.record_failures = embedder.recorded.extend
    return embedder


def _events(kind: str) -> list[tuple[str, float]]:
    rows = [line.split() for line in STATE["log"].read_text().splitlines()]
    return [(r[1], float(r[2])) for r in rows if r[0] == kind]


def test_embedding_starts_before_featurization_ends(tmp_path: Path) -> None:
    keys = [f"k{i}|alphafold" for i in range(6)]
    embedder = _embedder(tmp_path, keys)
    embedder.embed()

    last_featurized = max(t for _, t in _events("feat"))
    assert min(t for _, t in _events("embed")) < last_featurized
    assert embedder.store.get_missing_keys(keys) == []


def test_every_key_ends_in_store_or_failures_once(tmp_path: Path) -> None:
    good = [f"k{i}|alphafold" for i in range(5)]
    keys = good + ["bad1|alphafold", "nopdb1|alphafold", "embedfail1|alphafold"]
    embedder = _embedder(tmp_path, keys)
    embedder.embed()

    failed = [f["canonical_key"] for f in embedder.recorded]
    stored = [k for k in keys if k not in embedder.store.get_missing_keys(keys)]
    assert sorted(failed) == sorted(["bad1|alphafold", "nopdb1|alphafold", "embedfail1|alphafold"])
    assert sorted(stored) == sorted(good)
    assert len(set(failed) | set(stored)) == len(keys) == len(failed) + len(stored)


def test_restart_reuses_feature_shards_and_skips_embedded_keys(tmp_path: Path) -> None:
    keys = [f"k{i}|alphafold" for i in range(4)]
    embedder = _embedder(tmp_path, keys)
    feature_store = ESM3FeatureStore(embedder.featurized_dir, 2, embedder.logger)
    feature_store.put_records([_record(k) for k in keys[:2]])
    embedder.store.put_many({keys[0]: np.full(4, 99.0, dtype=np.float32)}, flush=True)

    embedder.embed()

    assert sorted(k for k, _ in _events("feat")) == keys[2:]
    assert embedder.store.get_missing_keys(keys) == []
    assert embedder.recorded == []


def _esmfold2_embedder(tmp_path: Path, keys: list[str], sequences: dict[str, str]):
    embedder = _embedder(tmp_path, keys)
    embedder.input_path = tmp_path / "proteins.pkl"
    records = [{"pdbs": [acc], "sequence": sequence} for acc, sequence in sequences.items()]
    embedder.input_path.write_bytes(pickle.dumps(records))
    embedder._resolve_pdb = lambda raw: ("x.cif", "esmfold2")
    return embedder


def test_esmfold2_sequence_mismatch_is_a_failure(tmp_path: Path) -> None:
    keys = ["a|esmfold2", "b|esmfold2", "c|esmfold2", "m|R1Q|esmfold2"]
    embedder = _esmfold2_embedder(tmp_path, keys, {"a": "AAA", "b": "CCC", "m|R1Q": "AAA"})
    embedder.embed()

    errors = {f["canonical_key"]: f["error"] for f in embedder.recorded}
    assert errors == {
        "b|esmfold2": "ESMFold2 sequence differs from the dataset sequence",
        "c|esmfold2": "No dataset sequence to check the ESMFold2 structure against",
    }
    assert embedder.store.get_missing_keys(keys) == ["b|esmfold2", "c|esmfold2"]


@pytest.mark.parametrize(
    ("structure", "dataset", "expected"),
    [
        ("AXA", "AUA", True),
        ("AXXA", "ABZA", True),
        ("AXA", "ACA", False),
        ("AXA", "AUAA", False),
    ],
)
def test_sequences_match_only_for_x_at_non_standard_residues(structure, dataset, expected) -> None:
    assert _sequences_match(structure, dataset) is expected


def test_retry_embeds_only_keys_with_a_new_esmfold2_structure(tmp_path: Path) -> None:
    done = "k0|alphafold"
    embedder = _esmfold2_embedder(tmp_path, [done, "a", "b"], {"a": "AAA", "b": "AAA"})
    embedder._resolve_pdb = lambda raw: ("x.pdb", "alphafold") if raw == "k0" else None
    del embedder.store.record_failures
    embedder.embed()
    failures = embedder.store.failures_path
    assert sorted(pd.read_parquet(failures)["canonical_key"]) == ["a", "b"]

    # An ESMFold2 structure for "a" appears; "b" still has none.
    embedder.retry_failed = True
    embedder._load_unique_input_keys = lambda: ([done, "a|esmfold2", "b"], [])
    embedder._resolve_pdb = lambda raw: ("x.cif", "esmfold2") if raw == "a" else None
    STATE["log"].write_text("")
    embedder.embed()

    assert sorted(k for k, _ in _events("feat")) == ["a|esmfold2"]
    assert embedder.store.get_missing_keys(["a|esmfold2", done]) == []
    failed = pd.read_parquet(failures)["canonical_key"]
    assert failed.is_unique and "b" in set(failed) and "a|esmfold2" not in set(failed)


def _pending_files(embedder: ESM3StructureShardEmbedder) -> list[Path]:
    return list(embedder.store.root_dir.glob("_pending_shard.*"))


def test_reclaim_without_feature_records_still_flushes_to_the_index(tmp_path: Path) -> None:
    keys = [f"k{i}|alphafold" for i in range(3)]
    embedder = _embedder(tmp_path, keys)
    embedder.reclaim_last_underfilled_shard = True
    embedder.store.put_many({k: np.ones(4, dtype=np.float32) for k in keys}, flush=True)

    embedder.embed()

    assert not _pending_files(embedder)
    assert len(embedder.store._load_index()) == len(keys)
    assert embedder.store.get_missing_keys(keys) == []


def test_reclaim_with_work_refills_the_terminal_shard(tmp_path: Path) -> None:
    old, new = ["k0|alphafold", "k1|alphafold"], ["k2|alphafold", "k3|alphafold"]
    embedder = _embedder(tmp_path, old + new)
    embedder.reclaim_last_underfilled_shard = True
    embedder.store.put_many({k: np.ones(4, dtype=np.float32) for k in old}, flush=True)

    embedder.embed()

    assert not _pending_files(embedder)
    assert len(embedder.store._load_index()) == 4
    assert embedder.store.get_missing_keys(old + new) == []
    assert sorted(k for k, _ in _events("feat")) == new


def test_results_drain_while_one_featurize_item_is_slow(tmp_path: Path) -> None:
    keys = ["slow1|alphafold"] + [f"k{i}|alphafold" for i in range(4)]
    embedder = _embedder(tmp_path, keys)
    embedder.featurize_workers = 2
    put_many = embedder.store.put_many

    def logged_put_many(arrays, **kwargs):
        if arrays:
            _log("put x")
        return put_many(arrays, **kwargs)

    embedder.store.put_many = logged_put_many
    embedder.embed()

    slow_done = dict(_events("feat"))["slow1|alphafold"]
    assert min(t for _, t in _events("put")) < slow_done - 1
    assert embedder.store.get_missing_keys(keys) == []


@pytest.mark.parametrize("featurize_new_keys", [True, False])
def test_gpu_worker_crash_raises(tmp_path: Path, featurize_new_keys: bool) -> None:
    STATE["crash"] = True
    keys = [f"k{i}|alphafold" for i in range(4)]
    embedder = _embedder(tmp_path, keys)
    if not featurize_new_keys:
        ESM3FeatureStore(embedder.featurized_dir, 2, embedder.logger).put_records(
            [_record(k) for k in keys]
        )
    with pytest.raises(RuntimeError, match="exited with code"):
        embedder.embed()


def test_nan_or_inf_rows_are_failures_and_not_stored() -> None:
    embeddings = torch.ones(3, 5, 4)  # BOS, 3 residues, EOS
    embeddings[1, 2, 0] = float("nan")
    embeddings[2, 1, 3] = float("inf")
    batch = [_record(f"k{i}|alphafold") for i in range(3)]

    arrays, failures = esm3._pool_batch(embeddings, batch)

    assert list(arrays) == ["k0|alphafold"]
    assert [(f["canonical_key"], f["error"]) for f in failures] == [
        ("k1|alphafold", "Embedding contains NaN or inf"),
        ("k2|alphafold", "Embedding contains NaN or inf"),
    ]


def _oom_above(limit: int, big_key: str | None = None):
    def forward(client, encoder, batch, device, autocast_dtype):
        if len(batch) > limit or any(r["key"] == big_key for r in batch):
            raise torch.cuda.OutOfMemoryError("oom")
        return {r["key"]: np.zeros(4, dtype=np.float16) for r in batch}, []

    return forward


def test_oom_splits_the_batch_and_stores_every_record_once(monkeypatch) -> None:
    sizes: list[int] = []
    forward = _oom_above(2)

    def recording(client, encoder, batch, device, autocast_dtype):
        sizes.append(len(batch))
        return forward(client, encoder, batch, device, autocast_dtype)

    monkeypatch.setattr(esm3, "_embed_batch", recording)
    batch = [_record(f"k{i}|alphafold") for i in range(7)]

    arrays, failures = esm3._embed_with_oom_split(None, None, batch, None, None)

    assert sorted(arrays) == sorted(r["key"] for r in batch)
    assert failures == []
    assert sizes[:3] == [7, 3, 1]  # halves recurse until a batch fits


def test_single_record_oom_is_a_failure(monkeypatch) -> None:
    monkeypatch.setattr(esm3, "_embed_batch", _oom_above(8, big_key="big|alphafold"))
    batch = [_record("a|alphafold"), _record("big|alphafold"), _record("b|alphafold")]

    arrays, failures = esm3._embed_with_oom_split(None, None, batch, None, None)

    assert sorted(arrays) == ["a|alphafold", "b|alphafold"]
    assert len(failures) == 1 and failures[0]["canonical_key"] == "big|alphafold"
    assert "CUDA out of memory" in failures[0]["error"]


def test_chunked_geometric_attention_matches_the_library(monkeypatch) -> None:
    from esm.layers.geom_attention import GeometricReasoningOriginalImpl
    from esm.utils.structure.affine3d import Affine3D

    torch.manual_seed(0)
    module = GeometricReasoningOriginalImpl(c_s=16, v_heads=4)
    for p in (module.distance_scale_per_head, module.rotation_scale_per_head):
        torch.nn.init.normal_(p)
    length = 11
    s = torch.randn(2, length, 16)
    affine = Affine3D.from_graham_schmidt(*torch.randn(3, 2, length, 3))
    mask = torch.ones(2, length, dtype=torch.bool)
    chain_id = torch.zeros(2, length, dtype=torch.int64)

    expected = module(s, affine, mask, None, chain_id)
    monkeypatch.setattr(esm3, "_GEOM_ATTN_CHUNK_ELEMENTS", 2 * 4 * length * 3 * 3)  # 3 rows
    actual = esm3._chunked_geometric_attention(module, s, affine, mask, None, chain_id)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_encoder_dtype_option_resolves_and_sets_the_store_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str
) -> None:
    from tests.helpers.builders import embedder_config

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "")
    keys = {
        "gpu_ids": [0],
        "af_pdb_path": str(tmp_path),
        "esm_pdb_path": str(tmp_path),
        "mmcif_path": str(tmp_path),
        "uniprot_acc_path": str(tmp_path),
        "esmfold2_cif_path": str(tmp_path),
        "esm3_embeddings_log_file_name": "esm3.log",
    }
    cfg = embedder_config(tmp_path, [], **keys, esm3_autocast_dtype=dtype)
    cfg.embeddings.esm3_embeddings_input_pkl_path = cfg.embeddings.input_pkl_path

    embedder = ESM3StructureShardEmbedder(cfg)

    assert embedder.autocast_dtype == dtype
    assert embedder.store.metadata.settings == f"pooling=residues;encoder_dtype={dtype}"

    cfg.embeddings.esm3_autocast_dtype = "float16"
    with pytest.raises(ValueError, match="esm3_autocast_dtype"):
        ESM3StructureShardEmbedder(cfg)


def test_featurize_workers_use_one_thread() -> None:
    with ProcessPoolExecutor(1, initializer=_init_featurize_worker) as executor:
        assert executor.submit(torch.get_num_threads).result() == 1
