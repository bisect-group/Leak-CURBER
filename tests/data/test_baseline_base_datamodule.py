from __future__ import annotations

import errno
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.components.baseline_base_datamodule import (
    BaselineBaseDataModule,
    BaseMaterializedFeatureBatchIterableDataset,
)


class _Dataset(BaseMaterializedFeatureBatchIterableDataset):
    LABEL_DTYPE = np.float32


def _epoch_labels(dataset) -> list[float]:
    return torch.cat([y for _, y in dataset]).tolist()


def test_rows_are_shuffled_inside_a_block_with_a_new_order_each_epoch(tmp_path: Path) -> None:
    n = 40
    np.save(tmp_path / "x.npy", np.arange(n, dtype=np.float32)[:, None])
    np.save(tmp_path / "y.npy", np.arange(n, dtype=np.float32))

    def make(shuffle: bool) -> _Dataset:
        return _Dataset(
            tmp_path / "x.npy",
            tmp_path / "y.npy",
            batch_size=8,
            block_rows=n,
            shuffle_blocks=shuffle,
            seed=0,
        )

    dataset = make(True)
    first, second = _epoch_labels(dataset), _epoch_labels(dataset)
    assert sorted(first) == sorted(second) == list(range(n))
    assert first != list(range(n)) and first != second
    assert _epoch_labels(make(True)) == first  # deterministic from the seed
    assert _epoch_labels(make(False)) == list(range(n))  # eval order is untouched


def test_features_stay_aligned_with_labels(tmp_path: Path) -> None:
    np.save(tmp_path / "x.npy", np.arange(20, dtype=np.float32)[:, None])
    np.save(tmp_path / "y.npy", np.arange(20, dtype=np.float32))
    dataset = _Dataset(
        tmp_path / "x.npy",
        tmp_path / "y.npy",
        batch_size=6,
        block_rows=20,
        shuffle_blocks=True,
        seed=3,
    )
    for x, y in dataset:
        assert torch.equal(x[:, 0], y)


@pytest.fixture
def datamodule(monkeypatch):
    module = object.__new__(BaselineBaseDataModule)
    valid = {"result": False}
    monkeypatch.setattr(
        module,
        "_validate_materialized_cache",
        lambda *a, **k: (valid["result"], ""),
        raising=False,
    )
    return module, valid


def _cache(path: Path, text: str) -> Path:
    path.mkdir(parents=True)
    (path / "marker").write_text(text)
    return path


def test_publish_replaces_a_stale_cache(tmp_path: Path, datamodule) -> None:
    module, _ = datamodule
    cache, tmp = _cache(tmp_path / "c", "stale"), _cache(tmp_path / "t", "fresh")
    module._publish_materialized_cache(tmp, cache, {}, refresh=False)
    assert (cache / "marker").read_text() == "fresh"
    assert not tmp.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["c"]


def test_publish_keeps_an_already_valid_cache(tmp_path: Path, datamodule) -> None:
    module, valid = datamodule
    valid["result"] = True
    cache, tmp = _cache(tmp_path / "c", "theirs"), _cache(tmp_path / "t", "mine")
    module._publish_materialized_cache(tmp, cache, {}, refresh=False)
    assert (cache / "marker").read_text() == "theirs"
    assert not tmp.exists()


def test_publish_keeps_the_existing_cache_on_a_real_error(
    tmp_path: Path, datamodule, monkeypatch
) -> None:
    module, _ = datamodule
    cache, tmp = _cache(tmp_path / "c", "stale"), _cache(tmp_path / "t", "fresh")
    real_rename = Path.rename

    def failing_rename(self, target):
        if self == tmp:
            raise OSError(errno.EIO, "disk error")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="disk error"):
        module._publish_materialized_cache(tmp, cache, {}, refresh=False)
    assert (cache / "marker").read_text() == "stale"


def test_index_digests_reuse_the_memo_for_unchanged_files(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import os

    from src.data.components import baseline_base_datamodule as module

    index = tmp_path / "index.parquet"
    index.write_bytes(b"aaaa")
    first = module._index_digests(tmp_path)
    assert first[0][0] == "index.parquet"

    # An unchanged file reuses the memo: no new hashing.
    def forbidden(*args, **kwargs):
        raise AssertionError("rehashed an unchanged file")

    monkeypatch.setattr(hashlib, "blake2b", forbidden)
    assert module._index_digests(tmp_path) == first
    monkeypatch.undo()

    # A same-size edit with a new mtime changes the digest.
    index.write_bytes(b"aaab")
    os.utime(index, ns=(0, index.stat().st_mtime_ns + 10**9))
    assert module._index_digests(tmp_path) != first

    # A new part file is hashed and listed with its relative path.
    (tmp_path / "index_parts").mkdir()
    (tmp_path / "index_parts" / "part_000001.parquet").write_bytes(b"zz")
    assert [rel for rel, _ in module._index_digests(tmp_path)] == [
        "index.parquet",
        "index_parts/part_000001.parquet",
    ]


def test_shuffled_batches_mix_rows_across_the_whole_split(tmp_path: Path) -> None:
    n = 4096
    np.save(tmp_path / "x.npy", np.arange(n, dtype=np.float32)[:, None])
    np.save(tmp_path / "y.npy", np.arange(n, dtype=np.float32))
    dataset = _Dataset(
        tmp_path / "x.npy", tmp_path / "y.npy", batch_size=256, block_rows=512, shuffle_blocks=True, seed=0
    )
    batches = list(dataset)
    assert len(batches) == len(dataset) == 16
    assert sorted(torch.cat([y for _, y in batches]).tolist()) == list(range(n))
    for x, y in batches:
        assert len(y) == 256 and torch.equal(x[:, 0], y)
        assert y.max() - y.min() > 2 * 512  # spans several blocks
