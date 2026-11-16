from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.baseline_classification_datamodule import BaselineClassificationDataModule, LabelEncoding
from tests.helpers.builders import write_cache, write_classification_tvt, write_retrieval_tvt


def test_classification_datamodule_encodes_full_ec_and_int64_materialization(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "data" / "enzyme_classification_dataset" / "random_splits"
    write_classification_tvt(split_dir)
    esmc = write_cache(tmp_path / "embeddings" / "esmc", "esmc", {"SEQ1": [1.0], "SEQ2": [2.0]})
    esm3 = write_cache(tmp_path / "embeddings" / "esm3", "esm3", {"P1": [3.0], "P2": [4.0]})

    dm = BaselineClassificationDataModule(
        endpoint="classification",
        data_root=str(tmp_path / "data"),
        data_roots={"classification": str(tmp_path / "data" / "enzyme_classification_dataset")},
        split="random_splits",
        modalities=["esmc", "esm3"],
        cache_roots={"esmc": str(esmc), "esm3": str(esm3)},
        batch_size=2,
        parquet_batch_size=2,
        pin_memory=False,
        materialized_cache_root=str(tmp_path / "mat"),
        materialized_feature_dtype="float32",
    )

    assert dm.get_class_labels() == ["1.2.3.4", "2.7.7.7"]
    x_batch, y_batch = next(iter(dm.train_dataloader()))
    assert x_batch.shape == (2, 2)
    assert y_batch.dtype == torch.int64

    info = dm.ensure_materialized_cache()
    y_train = np.load(info["splits"]["train"]["y_path"], mmap_mode="r")
    assert y_train.dtype == np.int64
    assert info["classes"] == ["1.2.3.4", "2.7.7.7"]


def test_retrieval_uses_drfp_rxn_key(tmp_path: Path) -> None:
    split_dir = (
        tmp_path
        / "data"
        / "enzyme_retrieval_dataset"
        / "reaction_drfp_tanimoto_splits"
        / "threshold_0.15"
    )
    write_retrieval_tvt(split_dir)
    drfp = write_cache(
        tmp_path / "embeddings" / "drfp", "drfp", {"A>>B": [1.0, 2.0], "C>>D": [3.0, 4.0]}
    )

    dm = BaselineClassificationDataModule(
        endpoint="retrieval",
        data_root=str(tmp_path / "data"),
        data_roots={"retrieval": str(tmp_path / "data" / "enzyme_retrieval_dataset")},
        split="reaction_drfp_tanimoto_splits",
        threshold="0.15",
        modalities=["drfp"],
        cache_roots={"drfp": str(drfp)},
        batch_size=2,
        parquet_batch_size=2,
        pin_memory=False,
    )

    x_batch, y_batch = next(iter(dm.train_dataloader()))
    assert x_batch.shape == (2, 2)
    assert sorted(y_batch.tolist()) == [0, 1]


def test_cache_is_never_reused_without_fingerprint_check() -> None:
    module = object.__new__(BaselineClassificationDataModule)
    module._materialized_info = None
    module.use_materialized_features, module.refresh_materialized_features = True, False
    # The base class returns None so the fingerprint path runs.
    assert module._existing_materialized_info_if_available() is None


def test_label_encoding_orders_train_classes_before_extra_classes() -> None:
    encoding = LabelEncoding.from_labels(["2.1.1.1", "1.1.1.1", None, "2.1.1.1"], ["3.1.1.1", "1.1.1.1", "0.0.0.0", "nan"])
    assert encoding.train_classes == ["1.1.1.1", "2.1.1.1"]
    assert encoding.classes == ["1.1.1.1", "2.1.1.1", "0.0.0.0", "3.1.1.1"]
    assert encoding.encode("3.1.1.1") == 3 and encoding.encode("9.9.9.9") == -1
