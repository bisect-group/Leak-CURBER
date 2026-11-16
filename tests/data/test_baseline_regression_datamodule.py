from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.data.baseline_regression_datamodule import (
    BaselineRegressionDataModule,
    MaterializedFeatureBatchIterableDataset,
)
from tests.helpers.builders import regression_datamodule, write_cache, write_tvt


def test_direct_split_and_test_dataloader(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path, labels=(10.0, 20.0, 30.0))
    dm.setup()

    assert dm.get_input_dim() == 3
    assert len(dm.train_dataloader()) == 1
    x_batch, y_batch = next(iter(dm.test_dataloader()))
    assert x_batch.shape == (1, 3)
    assert y_batch.item() == 30.0


def test_threshold_split_resolution(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    threshold_dir = data_root / "baselines_ki" / "substrate_splits" / "threshold_0.6"
    write_tvt(threshold_dir)
    smited = write_cache(tmp_path / "embeddings" / "smited", "smited", {"C": [1.0]})

    dm = BaselineRegressionDataModule(
        data_root=str(data_root),
        endpoint="ki",
        split="substrate_splits",
        modalities=["smited"],
        cache_roots={"smited": str(smited)},
        batch_size=1,
        pin_memory=False,
    )
    dm.prepare_data()

    assert dm.split_dir == threshold_dir


def test_missing_row_embedding_zero_pads(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    split_dir = data_root / "baselines_ki" / "random_splits_grouped_smiles"
    write_tvt(split_dir)
    for split_name in ("train", "val", "test"):
        pd.DataFrame(
            {
                "smiles": ["missing"],
                "sequence": ["SEQ"],
                "pdb_ids": [["P12345"]],
                "log10_value": [1.0],
            }
        ).to_parquet(split_dir / f"{split_name}.parquet", index=False)
    smited = write_cache(tmp_path / "embeddings" / "smited", "smited", {"C": [1.0, 2.0]})
    esmc = write_cache(tmp_path / "embeddings" / "esmc", "esmc", {"SEQ": [3.0]})

    dm = BaselineRegressionDataModule(
        data_root=str(data_root),
        endpoint="ki",
        split="random_splits_grouped_smiles",
        modalities=["smited", "esmc"],
        cache_roots={"smited": str(smited), "esmc": str(esmc)},
        batch_size=1,
        pin_memory=False,
    )
    x_batch, _ = next(iter(dm.train_dataloader()))

    assert torch.allclose(x_batch, torch.tensor([[0.0, 0.0, 3.0]]))


def test_missing_cache_fails(tmp_path: Path) -> None:
    dm = regression_datamodule(
        tmp_path,
        cache_roots={"smited": str(tmp_path / "missing")},
        modalities=["smited"],
    )

    with pytest.raises(FileNotFoundError):
        dm.setup()


def test_missing_esm3_key_column_fails(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    split_dir = data_root / "baselines_ki" / "random_splits_grouped_smiles"
    write_tvt(split_dir, include_pdb_ids=False)
    esm3 = write_cache(tmp_path / "embeddings" / "esm3", "esm3", {"P12345": [1.0]})

    dm = BaselineRegressionDataModule(
        data_root=str(data_root),
        endpoint="ki",
        split="random_splits_grouped_smiles",
        modalities=["esm3"],
        cache_roots={"esm3": str(esm3)},
        batch_size=1,
        pin_memory=False,
    )

    with pytest.raises(ValueError, match="pdb_ids"):
        dm.setup()


def test_esm3_list_lookup_mean_pools_available_embeddings(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    split_dir = data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        pd.DataFrame(
            {
                "pdb_ids": [["1AAA", "missing", "2BBB"]],
                "log10_value": [1.0],
            }
        ).to_parquet(split_dir / f"{split_name}.parquet", index=False)
    esm3 = write_cache(
        tmp_path / "embeddings" / "esm3",
        "esm3",
        {
            "1AAA": [1.0, 3.0],
            "2BBB": [5.0, 7.0],
        },
    )

    dm = BaselineRegressionDataModule(
        data_root=str(data_root),
        endpoint="ki",
        split="random_splits_grouped_smiles",
        modalities=["esm3"],
        cache_roots={"esm3": str(esm3)},
        batch_size=1,
        pin_memory=False,
    )
    x_batch, _ = next(iter(dm.train_dataloader()))

    assert torch.allclose(x_batch, torch.tensor([[3.0, 5.0]]))


def test_esm3_list_lookup_zero_when_all_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    split_dir = data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        pd.DataFrame(
            {
                "pdb_ids": [["missing"]],
                "log10_value": [1.0],
            }
        ).to_parquet(split_dir / f"{split_name}.parquet", index=False)
    esm3 = write_cache(tmp_path / "embeddings" / "esm3", "esm3", {"1AAA": [1.0, 3.0]})

    dm = BaselineRegressionDataModule(
        data_root=str(data_root),
        endpoint="ki",
        split="random_splits_grouped_smiles",
        modalities=["esm3"],
        cache_roots={"esm3": str(esm3)},
        batch_size=1,
        pin_memory=False,
    )
    x_batch, _ = next(iter(dm.train_dataloader()))

    assert torch.allclose(x_batch, torch.tensor([[0.0, 0.0]]))


def test_esm3_empty_numpy_list_key_zero_pads(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    split_dir = data_root / "baselines_ki" / "random_splits_grouped_smiles"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        pd.DataFrame(
            {
                "pdb_ids": [[]],
                "log10_value": [1.0],
            }
        ).to_parquet(split_dir / f"{split_name}.parquet", index=False)
    esm3 = write_cache(tmp_path / "embeddings" / "esm3", "esm3", {"1AAA": [1.0, 3.0]})

    dm = BaselineRegressionDataModule(
        data_root=str(data_root),
        endpoint="ki",
        split="random_splits_grouped_smiles",
        modalities=["esm3"],
        cache_roots={"esm3": str(esm3)},
        batch_size=1,
        pin_memory=False,
    )

    original_iter_batches = pq.ParquetFile.iter_batches

    def iter_batches_with_numpy_lists(self, *args, **kwargs):
        for batch in original_iter_batches(self, *args, **kwargs):
            payload = batch.to_pydict()
            payload["pdb_ids"] = [np.asarray(value) for value in payload["pdb_ids"]]
            yield pa.RecordBatch.from_pydict(payload)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", iter_batches_with_numpy_lists)
    x_batch, _ = next(iter(dm.train_dataloader()))

    assert torch.allclose(x_batch, torch.tensor([[0.0, 0.0]]))


def test_materialized_cache_reuses_valid_features(tmp_path: Path, monkeypatch) -> None:
    dm = regression_datamodule(
        tmp_path,
        materialized_cache_root=str(tmp_path / "materialized"),
        materialized_feature_dtype="float16",
        use_materialized_features=True,
        materialization_max_gb=1.0,
    )

    first = dm.ensure_materialized_cache()
    second = dm.ensure_materialized_cache()

    assert first["cache_status"] == "built"
    assert second["cache_status"] == "hit"
    assert Path(first["cache_dir"], ".complete").exists()

    def fail_iter(*_args, **_kwargs):
        raise AssertionError("valid materialized cache should not rebuild")

    monkeypatch.setattr(dm, "iter_feature_batches", fail_iter)
    third = dm.ensure_materialized_cache()
    assert third["cache_status"] == "hit"


def test_materialized_metadata_validation_does_not_load_embedding_readers(
    tmp_path: Path, monkeypatch
) -> None:
    dm = regression_datamodule(
        tmp_path,
        materialized_cache_root=str(tmp_path / "materialized"),
        materialized_feature_dtype="float16",
        materialized_validation="metadata",
        use_materialized_features=True,
        materialization_max_gb=1.0,
    )
    first = dm.ensure_materialized_cache()

    def fail_setup(*_args, **_kwargs):
        raise AssertionError("metadata validation should not initialize embedding readers")

    monkeypatch.setattr(dm, "setup", fail_setup)
    second = dm.ensure_materialized_cache()

    assert first["cache_status"] == "built"
    assert second["cache_status"] == "hit"
    assert second["cache_dir"] == first["cache_dir"]


def test_materialized_trust_complete_skips_npy_header_validation(tmp_path: Path) -> None:
    dm = regression_datamodule(
        tmp_path,
        materialized_cache_root=str(tmp_path / "materialized"),
        materialized_feature_dtype="float16",
        materialized_validation="metadata",
        use_materialized_features=True,
        materialization_max_gb=1.0,
    )
    first = dm.ensure_materialized_cache()
    Path(first["splits"]["train"]["x_path"]).unlink()

    assert dm.materialized_cache_status()["status"] == "missing"

    dm.materialized_validation = "trust_complete"
    status = dm.materialized_cache_status()

    assert status["status"] == "hit"
    assert status["reason"] == "fast_hit"


def test_materialized_dataloader_yields_float16_features(tmp_path: Path) -> None:
    dm = regression_datamodule(
        tmp_path,
        materialized_cache_root=str(tmp_path / "materialized"),
        materialized_feature_dtype="float16",
        use_materialized_features=True,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    x_batch, y_batch = next(iter(dm.train_dataloader()))

    assert x_batch.dtype == torch.float16
    assert y_batch.dtype == torch.float32
    assert x_batch.shape == (1, 3)


def test_materialized_block_shuffle_covers_each_row_once(tmp_path: Path) -> None:
    x_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.arange(12, dtype=np.float16).reshape(12, 1), allow_pickle=False)
    np.save(y_path, np.arange(12, dtype=np.float32), allow_pickle=False)
    dataset = MaterializedFeatureBatchIterableDataset(
        x_path,
        y_path,
        batch_size=2,
        block_rows=4,
        shuffle_blocks=True,
        seed=1,
    )

    first_epoch = [int(value) for _x, y in dataset for value in y.tolist()]
    second_epoch = [int(value) for _x, y in dataset for value in y.tolist()]

    assert sorted(first_epoch) == list(range(12))
    assert sorted(second_epoch) == list(range(12))
    assert first_epoch != second_epoch


def test_scaling_uses_non_missing_rows_of_train_only(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path, use_materialized_features=False)
    root = tmp_path / "data" / "baselines_ki" / "random_splits_grouped_smiles"
    for name in ("val", "test"):  # val/test must not change the stats
        frame = pd.read_parquet(root / f"{name}.parquet")
        frame["smiles"] = "MISSING"
        frame.to_parquet(root / f"{name}.parquet", index=False)
    train = pd.concat([pd.read_parquet(root / "train.parquet")] * 3, ignore_index=True)
    train.loc[0, "smiles"] = "MISSING"  # a missing row must not enter the mean
    train.to_parquet(root / "train.parquet", index=False)
    mean, std, block_id = dm.get_feature_scaling()
    # smited block [1, 2] -> mean 1.5, std 0.5. esmc block [3] -> mean 3, std 0 -> 1.
    np.testing.assert_allclose(mean, [1.5, 1.5, 3.0])
    np.testing.assert_allclose(std, [0.5, 0.5, 1.0])
    assert block_id.tolist() == [0, 0, 1]


def test_materialized_fingerprint_tracks_split_content(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path)
    dm.prepare_data()
    before = dm._materialized_metadata_from_caches()["fingerprint"]
    path = dm.split_paths["train"]
    stat = path.stat()
    data = bytearray(path.read_bytes())
    data[4] ^= 0xFF  # same size, different content
    path.write_bytes(bytes(data))
    import os

    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert dm._materialized_metadata_from_caches()["fingerprint"] != before


def test_materialized_metadata_warns_on_pending_shard(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path)
    dm.prepare_data()
    (tmp_path / "embeddings" / "smited" / "_pending_shard.json").write_text(
        json.dumps({"keys": ["a", "b"]})
    )
    with pytest.warns(UserWarning, match="smited.*2 unflushed pending rows"):
        dm._materialized_metadata_from_caches()


def test_materialized_cache_records_missing_embeddings(tmp_path: Path) -> None:
    dm = regression_datamodule(
        tmp_path,
        materialized_cache_root=str(tmp_path / "materialized"),
        use_materialized_features=True,
        materialization_max_gb=1.0,
    )
    val_path = tmp_path / "data" / "baselines_ki" / "random_splits_grouped_smiles" / "val.parquet"
    pd.DataFrame({"smiles": ["Z"], "sequence": ["SEQ"], "pdb_ids": [["P12345"]], "log10_value": [2.0]}).to_parquet(
        val_path, index=False
    )

    info = dm.ensure_materialized_cache()

    expected = {
        "train": {"rows": 1, "missing": {"smited": 0, "esmc": 0}},
        "val": {"rows": 1, "missing": {"smited": 1, "esmc": 0}},
        "test": {"rows": 1, "missing": {"smited": 0, "esmc": 0}},
    }
    assert info["missing_embeddings"] == expected
    assert json.loads((Path(info["cache_dir"]) / "metadata.json").read_text())["missing_embeddings"] == expected
    assert dm.ensure_materialized_cache()["missing_embeddings"] == expected
    assert dm.missing_embedding_counts() == expected
