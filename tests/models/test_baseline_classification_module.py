from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.baseline_classification_datamodule import BaselineClassificationDataModule
from src.models.baseline_classification_module import (
    BaselineClassificationEstimatorModule,
    BaselineClassificationLightningModule,
    _classification_scores,
    _top_k_accuracy,
    _top_k_hit_rate,
    classification_metrics,
)
from tests.helpers.builders import logged_metrics, write_cache, write_classification_tvt


def test_top_k_accuracy_drops_rows_without_a_valid_class() -> None:
    scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.2, 0.3], [0.1, 0.2, 0.7]])
    targets = np.array([0, 1, 7, -1, 0])  # 7 is out of range, -1 is unlabeled
    assert _top_k_accuracy(scores, targets, 3, k=1) == pytest.approx(2 / 3)


def _classification_module(num_classes: int) -> BaselineClassificationLightningModule:
    module = BaselineClassificationLightningModule(
        input_dim=2,
        num_classes=num_classes,
        hidden_dims=[2],
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        metrics=["accuracy", "macro_f1", "balanced_accuracy", "top_5_accuracy"],
    )
    module.log = MagicMock()
    return module


def test_epoch_macro_f1_equals_full_set_metric_across_batches() -> None:
    num_classes = 6
    module = _classification_module(num_classes)
    g = torch.Generator().manual_seed(0)
    logits = [torch.randn(5, num_classes, generator=g), torch.randn(7, num_classes, generator=g)]
    targets = [torch.tensor([0, 0, 1, 1, 2]), torch.tensor([3, 3, 3, 4, 5, 0, 1])]
    module._step = MagicMock(
        side_effect=[(torch.tensor(0.0), lg, t) for lg, t in zip(logits, targets)]
    )
    for _ in range(2):
        module._run_step(None, "val")
    module.on_validation_epoch_end()

    all_logits, all_targets = torch.cat(logits).numpy(), torch.cat(targets).numpy()
    expected = classification_metrics(
        all_logits.argmax(1),
        all_targets,
        class_labels=module.class_labels,
        metrics=module.metric_names,
        scores=all_logits,
    )
    logged = logged_metrics(module, "val/")
    for name, value in expected.items():
        assert np.isclose(logged[f"val/{name}"], value), name


def test_ec_level_exact_match_metrics() -> None:
    metrics = classification_metrics(
        np.asarray([0, 1, 2]),
        np.asarray([0, 2, 1]),
        class_labels=["1.2.3.4", "1.2.9.9", "2.7.7.7"],
        metrics=[
            "exact_match",
            "ec_l1_exact_match",
            "ec_l2_exact_match",
            "ec_l3_exact_match",
            "ec_l4_exact_match",
        ],
    )

    assert metrics["exact_match"] == pytest.approx(1 / 3)
    assert metrics["ec_l1_exact_match"] == pytest.approx(1 / 3)
    assert metrics["ec_l2_exact_match"] == pytest.approx(1 / 3)
    assert metrics["ec_l3_exact_match"] == pytest.approx(1 / 3)
    assert metrics["ec_l4_exact_match"] == pytest.approx(1 / 3)


def test_estimator_wrapper_predict_and_predict_proba(tmp_path: Path) -> None:
    class DummyClassifier:
        def fit(self, x, y):
            self.classes_ = np.asarray(sorted(set(y.tolist())), dtype=np.int64)
            return self

        def predict(self, x):
            return np.zeros((x.shape[0],), dtype=np.int64)

        def predict_proba(self, x):
            out = np.zeros((x.shape[0], 2), dtype=np.float32)
            out[:, 0] = 0.9
            out[:, 1] = 0.1
            return out

    split_dir = tmp_path / "data" / "enzyme_classification_dataset" / "random_splits"
    write_classification_tvt(split_dir)
    test_path = split_dir / "test.parquet"
    pd.read_parquet(test_path).assign(uniprot_id="q1").to_parquet(test_path, index=False)
    esmc = write_cache(tmp_path / "embeddings" / "esmc", "esmc", {"SEQ1": [1.0], "SEQ2": [2.0]})

    dm = BaselineClassificationDataModule(
        endpoint="classification",
        data_root=str(tmp_path / "data"),
        data_roots={"classification": str(tmp_path / "data" / "enzyme_classification_dataset")},
        split="random_splits",
        modalities=["esmc"],
        cache_roots={"esmc": str(esmc)},
        batch_size=2,
        parquet_batch_size=2,
        pin_memory=False,
        use_materialized_features=False,
    )
    wrapper = BaselineClassificationEstimatorModule(
        DummyClassifier(),
        save_model=False,
        save_predictions=True,
        metrics=["accuracy", "top_5_accuracy", "exact_match", "ec_l1_exact_match"],
    )

    metrics = wrapper.fit_evaluate(dm, tmp_path / "out")

    assert "val/accuracy" in metrics
    assert "test/top_5_accuracy" in metrics
    assert (tmp_path / "out" / "baseline_classifier" / "classes.json").exists()
    assert (tmp_path / "out" / "baseline_classifier" / "test_predictions.npy").exists()


def test_macro_f1_counts_classes_that_occur_only_in_predictions() -> None:
    from sklearn.metrics import f1_score

    targets = np.array([0, 0, 1, 1])
    preds = np.array([0, 2, 1, 2])
    scores = _classification_scores(preds, targets, num_classes=3)
    assert scores["macro_f1"] == pytest.approx(f1_score(targets, preds, average="macro"))
    assert scores["weighted_f1"] == pytest.approx(f1_score(targets, preds, average="weighted"))


def test_loss_batch_size_survives_a_compiled_net() -> None:
    module = object.__new__(BaselineClassificationLightningModule)
    module.num_classes = 3
    module.net = object()  # a torch.compile wrapper cannot be indexed
    assert module._loss_batch_size(torch.tensor([0, 2, 3, -1])) == 2
    assert module._loss_batch_size(torch.tensor([-1])) == 1


def test_chunked_top_k_matches_the_full_matrix() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((23, 9)).astype(np.float32)
    targets = np.append(rng.integers(0, 9, 22), 12)  # the last class has no score column

    class Estimator:
        def predict_proba(self, x):
            return scores[x[:, 0].astype(int)]

    module = object.__new__(BaselineClassificationEstimatorModule)
    module.estimator, module.predict_batch_rows = Estimator(), 4
    x = np.arange(23, dtype=np.float32)[:, None]
    chunked, chunked_scores = module._predict_top_k_in_chunks(x)
    full = np.argsort(-scores, axis=1)[:, :5]
    assert [set(a) for a in chunked] == [set(b) for b in full]
    assert np.array_equal(np.sort(chunked_scores, axis=1), np.sort(scores, axis=1)[:, -5:])
    assert _top_k_hit_rate(chunked, targets, 9) == _top_k_hit_rate(full, targets, 9)


def test_classification_scores_skip_invalid_targets_and_count_out_of_range_predictions() -> None:
    preds = np.array([0, 1, 2, 2, 5, 7, -1, 1, 3, 0, 2, 4])
    targets = np.array([0, 1, 1, 2, 5, 5, 2, -1, 6, 0, 2, 1])
    assert _classification_scores(preds, targets, num_classes=4) == {
        "accuracy": 0.625,
        "balanced_accuracy": 0.6666666666666666,
        "macro_f1": 0.7222222222222222,
        "weighted_f1": 0.6875,
    }


def test_classification_scores_are_zero_without_valid_targets() -> None:
    scores = _classification_scores(np.array([0, 1]), np.array([-1, 9]), num_classes=3)
    assert scores == dict.fromkeys(
        ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"), 0.0
    )


def test_classification_logits_are_float32_under_bf16_autocast() -> None:
    module = _classification_module(5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = module.net[0](torch.randn(3, 2))
        logits = module(torch.randn(3, 2))
    assert hidden.dtype == torch.bfloat16
    assert logits.dtype == torch.float32
