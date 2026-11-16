from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.components.embedders.store import ShardedEmbeddingReader, ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata
from src.data.components.embedders.utils import sha256_short
from tests.helpers.builders import (
    make_store,
    store_metadata,
    write_cache,
    write_store_files,
)


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True
    logger.setLevel(logging.INFO)
    return logger


def test_reclaim_skips_when_terminal_shard_keys_are_not_in_current_input(
    tmp_path: Path,
    caplog,
) -> None:
    root = tmp_path / "cache"
    write_store_files(
        root,
        {
            0: ["current_embedded"],
            1: ["historical_a", "historical_b"],
        },
    )
    original_index = pd.read_parquet(root / "index.parquet")
    store = ShardedEmbeddingStore(root, store_metadata(), _logger("test.reclaim.skip"))

    with caplog.at_level(logging.WARNING, logger="test.reclaim.skip"):
        reclaimed = store.reclaim_last_underfilled_shard(["current_embedded", "current_missing"])

    assert reclaimed == 0
    assert "Skipping terminal shard reclaim for shard_000001.npy" in caplog.text
    assert "2 of 2 shard hashes are unavailable" in caplog.text
    assert (root / "shard_000001.npy").exists()
    assert not (root / "_pending_shard.npy").exists()
    assert not (root / "_pending_shard.json").exists()
    pd.testing.assert_frame_equal(
        pd.read_parquet(root / "index.parquet"),
        original_index,
    )
    assert store.get_missing_keys(["current_embedded", "current_missing"]) == ["current_missing"]


def test_reclaim_moves_terminal_shard_to_pending_when_keys_are_available(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    write_store_files(
        root,
        {
            0: ["stable_a"],
            1: ["terminal_a", "terminal_b"],
        },
    )
    store = ShardedEmbeddingStore(root, store_metadata(), _logger("test.reclaim.success"))

    reclaimed = store.reclaim_last_underfilled_shard(["stable_a", "terminal_a", "terminal_b"])

    assert reclaimed == 2
    assert not (root / "shard_000001.npy").exists()
    assert (root / "_pending_shard.npy").exists()
    pending_meta = json.loads((root / "_pending_shard.json").read_text())
    assert pending_meta["keys"] == ["terminal_a", "terminal_b"]
    index_df = pd.read_parquet(root / "index.parquet")
    assert set(index_df["key_hash"]) == {sha256_short("stable_a")}
    assert store.get_missing_keys(["stable_a", "terminal_a", "terminal_b"]) == []


def test_store_rejects_other_pooling_settings(tmp_path: Path) -> None:
    store = make_store(tmp_path, "dtype=bf16;pooling=residues")
    store.put_many({"a": np.ones(2, dtype=np.float32)}, flush=True)
    make_store(tmp_path, "dtype=bf16;pooling=residues")  # same settings reopen
    with pytest.raises(ValueError, match="does not match"):
        make_store(tmp_path, "dtype=fp16;pooling=residues")
    with pytest.raises(ValueError, match="does not match"):
        make_store(tmp_path)  # a store written without settings is not reused


def test_index_has_one_row_per_key(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.put_many({"a": np.ones(2, dtype=np.float32)}, flush=True)
    store._pending_keys, store._pending_arrays = ["a"], [np.full(2, 5.0, dtype=np.float32)]
    store._pending_bytes = 8
    store.flush()
    index = pd.read_parquet(tmp_path / "index.parquet")
    assert index["key_hash"].tolist() == [sha256_short("a")]


def test_reader_warns_about_pending_shard(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.put_many({"a": np.ones(2, dtype=np.float32)}, flush=True)
    store.put_many({"b": np.ones(2, dtype=np.float32)})
    store.checkpoint_pending(force=True)
    with pytest.warns(UserWarning, match="unflushed pending rows"):
        ShardedEmbeddingReader(tmp_path)
    store.flush()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ShardedEmbeddingReader(tmp_path)


def test_failures_are_not_duplicated(tmp_path: Path) -> None:
    metadata = CacheMetadata(
        embedder="t",
        model_name="m",
        version="v1",
        storage_dtype="float32",
        embedding_dim=None,
        max_shard_bytes=1 << 20,
        key_type="t",
        key_field="t",
    )
    store = ShardedEmbeddingStore(tmp_path, metadata, logging.getLogger("t"))
    failure = {"canonical_key": "k", "raw_key": "K", "error": "boom"}
    store.record_failures([failure])
    store.record_failures([failure])
    assert len(pd.read_parquet(store.failures_path)) == 1


def test_size_flushes_write_index_parts_and_the_final_flush_merges_them(tmp_path: Path) -> None:
    store = make_store(tmp_path, shard_bytes=16)  # two float32 vectors of dim 2 fill a shard
    for i in range(6):
        store.put_many({f"k{i}": np.full(2, i, dtype=np.float32)})
    assert list((tmp_path / "index_parts").glob("part_*.parquet"))
    assert not (tmp_path / "index.parquet").exists()
    assert store.get_missing_keys([f"k{i}" for i in range(6)]) == []  # parts are visible
    store.put_many({}, flush=True)
    assert not list((tmp_path / "index_parts").glob("part_*.parquet"))
    assert len(pd.read_parquet(tmp_path / "index.parquet")) == 6
    # A reopen after a crash merges parts that were never compacted.
    store.put_many({"late": np.ones(2, dtype=np.float32)})
    store._flush_shard()
    assert list((tmp_path / "index_parts").glob("part_*.parquet"))
    make_store(tmp_path, shard_bytes=16)
    assert len(pd.read_parquet(tmp_path / "index.parquet")) == 7


def test_readers_see_keys_that_exist_only_in_index_parts(tmp_path: Path) -> None:
    from src.data.components.embedders.store import ShardedEmbeddingReader, index_files, load_index

    store = make_store(tmp_path, shard_bytes=16)
    for i in range(4):  # size flushes write parts; index.parquet is never written
        store.put_many({f"k{i}": np.full(2, i, dtype=np.float32)})
    assert not (tmp_path / "index.parquet").exists()
    assert index_files(tmp_path)
    assert len(load_index(tmp_path)) == 4
    reader = ShardedEmbeddingReader(tmp_path)
    assert reader.get("k3").tolist() == [3.0, 3.0]


def test_merge_deletes_only_the_parts_it_read(tmp_path: Path) -> None:
    from src.data.components.embedders.store import load_index

    store = make_store(tmp_path, shard_bytes=16)
    for i in range(2):
        store.put_many({f"k{i}": np.full(2, i, dtype=np.float32)})
    late = tmp_path / "index_parts" / "part_999999.parquet"
    real_load = load_index

    def load_then_add_part(root, files=None, columns=None):
        frame = real_load(root, files, columns)
        pd.DataFrame({"key_hash": ["late"], "shard_id": [0], "row_idx": [0]}).to_parquet(late)
        return frame

    import src.data.components.embedders.store as store_module

    store_module.load_index = load_then_add_part
    try:
        store._replace_index()
    finally:
        store_module.load_index = real_load
    assert late.exists()  # written after the merge read its parts, so it stays
    assert len(pd.read_parquet(tmp_path / "index.parquet")) == 2


def test_reader_lookup_returns_the_cached_vector(tmp_path: Path) -> None:
    cache = write_cache(tmp_path / "cache", "smited", {"C": [1.0, 2.0]})
    reader = ShardedEmbeddingReader(cache)

    assert reader.embedding_dim == 2
    assert reader.contains("C")
    assert np.allclose(reader.get("C"), np.asarray([1.0, 2.0], dtype=np.float32))
    assert reader.get("missing") is None


def test_drop_keys_makes_them_missing_and_keeps_the_others(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    write_store_files(root, {0: ["a", "b"], 1: ["c"]})
    store = ShardedEmbeddingStore(root, store_metadata(), _logger("test.drop"))

    assert store.drop_keys(["b", "c", "never_stored"]) == 2

    assert store.get_missing_keys(["a", "b", "c"]) == ["b", "c"]
    assert list(pd.read_parquet(root / "index.parquet")["key_hash"]) == [sha256_short("a")]
