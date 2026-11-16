from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from src.data.components.splitters import base
from src.data.components.splitters.base import (
    BaseSplitter,
    BaseThresholdedSimilaritySplitter,
    _write_parquet_task,
    configured,
    find_input_dataset_path,
    first_present,
    greedy_fill,
    order_by_similarity,
)


def test_write_parquet_task_accepts_polars_and_pandas(tmp_path: Path) -> None:
    _write_parquet_task((pl.DataFrame({"a": [1, 2]}), str(tmp_path / "pl.parquet")))
    _write_parquet_task((pd.DataFrame({"a": [1, 2]}), str(tmp_path / "pd.parquet")))
    assert pd.read_parquet(tmp_path / "pl.parquet")["a"].tolist() == [1, 2]
    assert pd.read_parquet(tmp_path / "pd.parquet")["a"].tolist() == [1, 2]


def test_greedy_fill_skips_a_group_that_does_not_fit() -> None:
    counts = {0: 2, 1: 9, 2: 3, 3: 2}
    taken, total = greedy_fill([0, 1, 2, 3], counts.__getitem__, 7)
    assert taken == [
        0,
        2,
        3,
    ]  # the 9-row group does not fit, and the later groups still fill the budget
    assert total == 7


def test_order_by_similarity_breaks_ties_on_row_count_then_key() -> None:
    keys = np.array(["d", "c", "b", "a"])
    sims = np.array([0.1, 0.1, 0.1, 0.0])
    counts = np.array([1, 1, 5, 9])
    assert order_by_similarity(range(4), sims, counts, keys).tolist() == [3, 1, 0, 2]


def test_split_manifest_is_written(tmp_path: Path) -> None:
    splitter = object.__new__(BaseSplitter)
    splitter.RANDOM_SEED = 7
    splitter.cfg = OmegaConf.create({"splits": {"x": 1}})
    splitter._write_split_manifest(tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "split_manifest.json").read_text())
    assert manifest["seed"] == 7
    assert manifest["config"] == {"x": 1}


def test_configured_keeps_zero() -> None:
    assert configured(0.0, 0.1) == 0.0
    assert configured(None, 0.1) == 0.1


class _StubSplitter(BaseThresholdedSimilaritySplitter):
    def run_splits_across_thresholds(self):
        raise NotImplementedError

    def plot_unique_distribution(self, all_similarities):
        raise NotImplementedError

    def plot_full_distribution(self, all_splits, all_similarities):
        raise NotImplementedError

    def get_output_dir(self):
        raise NotImplementedError


def _stub(tmp_path: Path, **attrs) -> _StubSplitter:
    stub = _StubSplitter.__new__(_StubSplitter)
    stub.logger = logging.getLogger("stub")
    stub.KEY_COLUMN = "key"
    stub.EMBEDDING_COLUMN = None
    stub.EMBEDDINGS_STORE_DIR = None
    stub.EMBEDDINGS_NPZ = None
    stub.INPUT_DATASET_PATH = tmp_path / "data" / "in.parquet"
    stub.full_dataset_df = pd.DataFrame(
        {"key": ["a", "b", None, "c"], "emb": [None, None, None, None]}
    )
    for name, value in attrs.items():
        setattr(stub, name, value)
    return stub


def test_vector_from_raw_dtype_and_pooling(tmp_path: Path) -> None:
    cosine = _stub(tmp_path)  # float32, no pooling
    assert cosine._vector_from_raw("[1, 2, 3]").dtype == np.float32
    assert cosine._vector_from_raw([[1, 2], [3, 4]]) is None
    assert cosine._vector_from_raw(None) is None
    assert cosine._vector_from_raw("  ") is None
    assert cosine._vector_from_raw("not json") is None
    assert cosine._vector_from_raw(5) is None
    pooled = _stub(tmp_path, MEAN_POOL_2D_EMBEDDINGS=True)
    assert np.array_equal(pooled._vector_from_raw([[1, 2], [3, 4]]), [2.0, 3.0])
    drfp = _stub(tmp_path, EMBEDDING_DTYPE=np.uint8)
    vec = drfp._vector_from_raw(json.dumps([0, 1, 1]))
    assert vec.dtype == np.uint8 and vec.tolist() == [0, 1, 1]


def test_iter_column_vectors_skips_missing(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"key": ["a", None, "b", "c"], "emb": [[1.0, 2.0], [3.0, 4.0], None, "[5, 6]"]}
    )
    stub = _stub(tmp_path, EMBEDDING_COLUMN="emb", full_dataset_df=df)
    out = dict(stub._iter_column_vectors())
    assert sorted(out) == ["a", "c"]
    assert out["c"].tolist() == [5.0, 6.0]


def test_read_store_vectors_and_source_resolution(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    np.save(store / "shard_000000.npy", np.array([[1, 2], [3, 4]], dtype=np.float32))
    np.save(store / "shard_000001.npy", np.array([[5, 6]], dtype=np.float32))
    pd.DataFrame(
        [
            {"key_hash": "h0", "shard_id": 0, "row_idx": 0},
            {"key_hash": "h1", "shard_id": 0, "row_idx": 1},
            {"key_hash": "h2", "shard_id": 1, "row_idx": 0},
            {"key_hash": "h1", "shard_id": 0, "row_idx": 0},  # later duplicate wins
            {"key_hash": "hbad", "shard_id": 0, "row_idx": 9},  # out of bounds
            {"key_hash": "hmissing", "shard_id": 7, "row_idx": 0},  # shard file absent
        ]
    ).to_parquet(store / "index.parquet", index=False)
    (store / "meta.json").write_text("{}")

    stub = _stub(tmp_path, EMBEDDINGS_STORE_DIR=store)
    vectors = stub._read_store_vectors(store, ["h0", "h1", "h2", "hbad", "hmissing", "hunknown"])
    assert sorted(vectors) == ["h0", "h1", "h2"]
    assert vectors["h1"].tolist() == [1.0, 2.0]
    assert vectors["h2"].tolist() == [5.0, 6.0]
    assert vectors["h0"].dtype == np.float32

    assert stub._resolve_embedding_source(npz_name="x.npz", error_hint="none") == ("store", store)
    stub.EMBEDDING_COLUMN = "emb"
    assert stub._resolve_embedding_source(npz_name="x.npz", error_hint="none")[0] == "column"

    npz = tmp_path / "data" / "x.npz"
    npz.parent.mkdir()
    np.savez(npz, a=np.ones(2))
    npz_stub = _stub(tmp_path)
    assert npz_stub._resolve_embedding_source(npz_name="x.npz", error_hint="none") == ("npz", npz)
    with pytest.raises(FileNotFoundError, match="nothing here"):
        _stub(tmp_path)._resolve_embedding_source(
            npz_name="missing.npz", error_hint="nothing here."
        )


@pytest.mark.parametrize(
    "row_counts, order, max_rows, expected",
    [
        ({0: 3, 1: 3, 2: 3, 3: 3}, [2, 0, 3, 1], 9, ([2, 0, 3], 9)),  # ties: input order decides
        ({0: 5, 1: 9, 2: 1}, [0, 1, 2], 8, ([0, 2], 6)),  # skips the key that does not fit
        ({0: 4, 1: 4}, [0, 1], 8, ([0, 1], 8)),  # exact fit
        ({0: 4}, [0], 0, ([], 0)),  # nothing fits
        ({}, [], 5, ([], 0)),
    ],
)
def test_greedy_fill_returns_selected_indices_and_total(
    row_counts, order, max_rows, expected
) -> None:
    assert greedy_fill(order, row_counts.__getitem__, max_rows) == expected


@pytest.mark.parametrize(
    "max_rows, expected",
    [
        (0, ([], 0)),
        (4, ([0, 1], 4)),
        (10, ([4, 0, 1], 9)),
        (25, ([4, 0, 2, 1, 5, 3], 22)),
        (1000, ([4, 0, 2, 1, 5, 3], 22)),
    ],
)
def test_greedy_fill_with_row_counts_from_an_array(max_rows, expected) -> None:
    sequence_row_count = np.array([3, 1, 7, 2, 5, 4], dtype=np.int64)
    assert (
        greedy_fill([4, 0, 2, 1, 5, 3], lambda idx: int(sequence_row_count[idx]), max_rows)
        == expected
    )


@pytest.mark.parametrize(
    "max_rows, expected",
    [
        (0, ([2], 0)),
        (4, ([2, 4, 1], 3)),
        (10, ([2, 4, 3], 10)),
        (25, ([2, 4, 3, 1, 0], 18)),
        (1000, ([2, 4, 3, 1, 0], 18)),
    ],
)
def test_greedy_fill_treats_keys_missing_from_the_counts_as_empty(max_rows, expected) -> None:
    unique_keys = ["k0", "k1", "k2", "k3", "k4"]
    row_count_dict = {"k0": 6, "k1": 2, "k3": 9, "k4": 1}
    assert (
        greedy_fill([2, 4, 3, 1, 0], lambda idx: row_count_dict.get(unique_keys[idx], 0), max_rows)
        == expected
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({}, None),
        ({"a_key": "A"}, "A"),
        ({"b_key": "B", "c_key": "C"}, "B"),
        ({"c_key": "C"}, "C"),
        ({"a_key": None, "b_key": "B"}, None),  # a present-but-None value still wins
        ({"a_key": "A", "b_key": "B", "c_key": "C"}, "A"),
        ({"b_key": None, "c_key": "C"}, None),
    ],
)
def test_first_present_returns_the_first_key_in_the_config(raw, expected) -> None:
    assert first_present(OmegaConf.create(raw), ("a_key", "b_key", "c_key")) == expected


_BLOCK_PATH = "input_dataset_parquet_file_path"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({}, None),
        ({"sequence_split": {_BLOCK_PATH: "seq.parquet"}}, "seq.parquet"),
        (
            {"sequence_split": {_BLOCK_PATH: None}, "smiles_split": {_BLOCK_PATH: "smi.parquet"}},
            "smi.parquet",
        ),
        (
            {"sequence_split": None, "structure_split": {_BLOCK_PATH: "struct.parquet"}},
            "struct.parquet",
        ),
        (
            {"smiles_split": {}, "tanimoto_split_input_dataset_parquet_file_path": "tan.parquet"},
            "tan.parquet",
        ),
        (
            {
                "smiles_split_input_dataset_parquet_file_path": None,
                "conformer_split_input_dataset_parquet_file_path": "conf.parquet",
            },
            "conf.parquet",
        ),
        (
            {
                "structure_split": {_BLOCK_PATH: "s.parquet"},
                "smiles_split_input_dataset_parquet_file_path": "flat.parquet",
            },
            "s.parquet",
        ),
        ({"sequence_split": {_BLOCK_PATH: ""}}, None),
    ],
)
def test_find_input_dataset_path_prefers_blocks_then_flat_keys(raw, expected) -> None:
    blocks = ("sequence_split", "smiles_split", "structure_split")
    assert find_input_dataset_path(OmegaConf.create(raw), blocks) == expected


def test_write_parquet_task_uses_zstd(tmp_path: Path) -> None:
    for name, frame in (("pl", pl.DataFrame({"a": [1, 2]})), ("pd", pd.DataFrame({"a": [1, 2]}))):
        path = Path(_write_parquet_task((frame, str(tmp_path / f"{name}.parquet"))))
        assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


def _save(tmp_path: Path, frames: dict[str, object]) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.parquet" for name in frames}
    splitter = BaseSplitter.__new__(BaseSplitter)
    splitter.logger = logging.getLogger("test")
    splitter._save_split_tasks_in_parallel([(df, paths[n]) for n, df in frames.items()])
    return paths


def test_save_split_tasks_gives_null_polars_columns_the_sibling_type(tmp_path: Path) -> None:
    train = pl.DataFrame({"a": [1, 2], "mutations": [["x"], ["y"]]})
    empty = train.clear()
    all_null = pl.DataFrame({"a": [3], "mutations": [None]}, schema=train.schema | {"mutations": pl.Null})
    paths = _save(tmp_path, {"train": train, "empty": empty, "null": all_null})
    for name in ("empty", "null"):
        assert pl.read_parquet_schema(paths[name]) == train.schema


@pytest.mark.parametrize("values, dtype", [(["x", "y"], pl.String), ([["x"], ["y"]], pl.List(pl.String))])
def test_save_split_tasks_gives_null_pandas_columns_the_sibling_type(
    tmp_path: Path, values, dtype
) -> None:
    train = pd.DataFrame({"a": [1, 2], "m": values})
    test = pd.DataFrame({"a": [3], "m": [None]})
    paths = _save(tmp_path, {"train": train, "test": test})
    assert pl.read_parquet_schema(paths["test"])["m"] == dtype
    assert pl.read_parquet_schema(paths["train"]) == pl.read_parquet_schema(paths["test"])
    assert len(pd.read_parquet(paths["test"])) == 1


def test_save_split_tasks_leaves_frames_without_null_columns_unchanged(tmp_path: Path) -> None:
    pdf = pd.DataFrame({"a": [1, 2], "m": ["x", "y"]}, index=[5, 6])
    plf = pl.DataFrame({"a": [1, 2], "m": [["x"], ["y"]]})
    paths = _save(tmp_path, {"pd": pdf, "pl": plf})
    assert pl.read_parquet(paths["pl"]).equals(plf)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["pd"]), pdf.reset_index(drop=True))


def test_save_split_tasks_writes_in_process_when_memory_allows_one_worker(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(base, "MEMORY_BUDGET_FRACTION", 0.0)
    monkeypatch.setattr(base, "ProcessPoolExecutor", None)  # a pool would fail here
    paths = _save(tmp_path, {"train": pd.DataFrame({"a": [1, 2]}), "test": pd.DataFrame({"a": [3]})})
    assert {n: len(pd.read_parquet(p)) for n, p in paths.items()} == {"train": 2, "test": 1}


def test_memory_bounded_workers_caps_by_available_memory(monkeypatch) -> None:
    frame = pd.DataFrame({"a": np.zeros(1000)})
    need = base.WORKER_MEMORY_FACTOR * frame.memory_usage(index=False).sum()
    free = 2 * need / base.MEMORY_BUDGET_FRACTION
    monkeypatch.setattr(base.psutil, "virtual_memory", lambda: type("M", (), {"available": free})())
    assert base._memory_bounded_workers([frame] * 3, 9) == 2


def test_save_named_splits_writes_shift_stats(tmp_path: Path) -> None:
    splitter = object.__new__(BaseSplitter)
    splitter.logger = logging.getLogger("test")
    frames = {
        "train": pd.DataFrame({"log10_value": [0.0, 1.0, 2.0, 3.0]}),
        "val": pd.DataFrame({"log10_value": [2.0, 3.0]}),
        "test": pd.DataFrame({"log10_value": [4.0, 5.0]}),
    }
    splitter.save_named_splits(frames, tmp_path)
    stats = json.loads((tmp_path / "split_stats.json").read_text())
    assert stats["splits"]["train"]["n"] == 4
    assert stats["splits"]["train"]["quantiles"]["q50"] == 1.5
    val, test = stats["vs_train"]["val"], stats["vs_train"]["test"]
    assert val["mean_shift"] == 1.0
    assert val["ks"] == 0.5
    assert val["share_outside_train_range"] == 0.0
    assert test["share_outside_train_range"] == 1.0
    assert test["median_shift"] == 3.0
    assert test["train_mean_rmse"] == pytest.approx(np.sqrt(0.5 * (2.5**2 + 3.5**2)))
    assert test["train_mean_mae"] == 3.0
    assert test["train_mean_r2"] == pytest.approx(1 - 18.5 / 0.5)
    assert val["train_mean_r2"] == pytest.approx(1 - (0.25 + 2.25) / 0.5)


def test_save_named_splits_skips_stats_without_a_numeric_target(tmp_path: Path) -> None:
    splitter = object.__new__(BaseSplitter)
    splitter.logger = logging.getLogger("test")
    splitter.save_named_splits({"train": pd.DataFrame({"a": [1]}), "test": pd.DataFrame({"a": [2]})}, tmp_path)
    assert not (tmp_path / "split_stats.json").exists()
