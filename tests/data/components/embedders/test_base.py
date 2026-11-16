from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.components.embedders.base import BaseShardEmbedder
from src.data.components.embedders.store import ShardedEmbeddingReader
from tests.helpers.builders import embedder_config


class _LengthEmbedder(BaseShardEmbedder):
    """Embeds a key as a 3-vector of its length; keys starting with `bad` fail."""

    def __init__(self, cfg, chunk_size: int = 100) -> None:
        super().__init__(
            cfg,
            input_path=cfg.embeddings.input_pkl_path,
            log_file_name="length.log",
            embedder_name="length",
            model_name="m",
            version="v1",
            storage_dtype="float32",
            key_field="name",
            key_type="name",
            max_shard_bytes=1 << 20,
            compute_chunk_size=chunk_size,
        )
        self.chunks: list[list[str]] = []

    def compute_many(self, keys):
        self.chunks.append(list(keys))
        arrays = {k: np.full(3, len(k), dtype=np.float32) for k in keys if not k.startswith("bad")}
        failures = [
            {"raw_key": k, "canonical_key": k, "error": "bad key"}
            for k in keys
            if k.startswith("bad")
        ]
        return arrays, failures


def _reader(embedder: BaseShardEmbedder) -> ShardedEmbeddingReader:
    return ShardedEmbeddingReader(embedder.store.root_dir)


def test_store_lives_under_embedder_model_and_version(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, ["a"]))
    assert embedder.store.root_dir == tmp_path / "emb" / "length" / "m" / "v1"


def test_missing_input_file_is_an_error(tmp_path) -> None:
    cfg = embedder_config(tmp_path, ["a"])
    cfg.embeddings.input_pkl_path = str(tmp_path / "absent.pkl")
    with pytest.raises(FileNotFoundError, match="absent.pkl"):
        _LengthEmbedder(cfg)


def test_embed_deduplicates_keys_and_skips_none(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, ["aa", None, "b", "aa"]))
    embedder.embed()
    assert embedder.chunks == [["aa", "b"]]
    reader = _reader(embedder)
    assert reader.get("aa").tolist() == [2.0, 2.0, 2.0]
    assert reader.get("b").tolist() == [1.0, 1.0, 1.0]


def test_none_keys_are_recorded_as_failures(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, ["a", None]))
    embedder.embed()
    failures = pd.read_parquet(embedder.store.failures_path)
    assert failures["error"].tolist() == ["Input key is None"]
    assert failures["canonical_key"].isna().all()


def test_second_embed_computes_only_missing_keys(tmp_path) -> None:
    cfg = embedder_config(tmp_path, ["a", "bb"])
    _LengthEmbedder(cfg).embed()
    embedder = _LengthEmbedder(embedder_config(tmp_path, ["a", "bb", "ccc"]))
    embedder.embed()
    assert embedder.chunks == [["ccc"]]
    unchanged = _LengthEmbedder(embedder_config(tmp_path, ["a", "bb", "ccc"]))
    unchanged.embed()
    assert unchanged.chunks == []


def test_missing_keys_are_computed_in_chunks(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, ["a", "b", "c", "d", "e"]), chunk_size=2)
    embedder.embed()
    assert embedder.chunks == [["a", "b"], ["c", "d"], ["e"]]


def test_failed_keys_are_recorded_and_not_stored(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, ["ok", "bad1"]))
    embedder.embed()
    failures = pd.read_parquet(embedder.store.failures_path)
    assert failures["raw_key"].tolist() == ["bad1"]
    assert failures["error"].tolist() == ["bad key"]
    assert not _reader(embedder).contains("bad1")


def test_keys_are_read_from_dict_items_by_key_field(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, [{"name": "xx"}, {"name": "y"}]))
    assert embedder.load_raw_keys() == ["xx", "y"]


def test_dict_items_without_the_key_field_are_an_error(tmp_path) -> None:
    embedder = _LengthEmbedder(embedder_config(tmp_path, [{"other": "xx"}]))
    with pytest.raises(ValueError, match="must contain 'name'"):
        embedder.load_raw_keys()
