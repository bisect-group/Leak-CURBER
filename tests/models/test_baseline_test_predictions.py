from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.baseline_classification_datamodule import BaselineClassificationDataModule
from src.models.components.baseline_base_module import TEST_PREDICTIONS_FILE
from src.models.baseline_classification_module import (
    BaselineClassificationEstimatorModule,
    BaselineClassificationLightningModule,
    save_ec_predictions,
)
from src.models.baseline_regression_module import (
    BaselineRegressionEstimatorModule,
    BaselineRegressionLightningModule,
    regression_metrics,
)
from src.utils.ec_level_accuracy import ec_level_accuracy_from_predictions
from tests.helpers.builders import regression_datamodule, write_cache, write_classification_tvt

OPTIMIZER = lambda params: torch.optim.Adam(params, lr=1e-3)  # noqa: E731


def _trainer(root: Path) -> L.Trainer:
    root.mkdir()
    return L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(root),
    )


def _classification_datamodule(tmp_path: Path, endpoint: str = "classification"):
    split_dir = tmp_path / "data" / "random_splits"
    write_classification_tvt(split_dir)
    test_path = split_dir / "test.parquet"
    test = pd.DataFrame(
        {
            "sequence": ["SEQ1", "SEQ2", "SEQ2"],
            "pdb_ids": [["P1"], ["P2"], ["P2"]],
            "ec_number": ["1.2.3.4", "2.7.7.7", "1.2.3.4"],
            "uniprot_id": ["q1", "q2", "q2"],
        }
    )
    test.to_parquet(test_path, index=False)
    esmc = write_cache(tmp_path / "embeddings" / "esmc", "esmc", {"SEQ1": [1.0], "SEQ2": [2.0]})
    return BaselineClassificationDataModule(
        endpoint=endpoint,
        data_root=str(tmp_path / "data"),
        split="random_splits",
        modalities=["esmc"],
        cache_roots={"esmc": str(esmc)},
        batch_size=2,
        parquet_batch_size=2,
        pin_memory=False,
    )


def test_regression_mlp_saved_predictions_reproduce_logged_metrics(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path)
    model = BaselineRegressionLightningModule(
        input_dim=dm.get_input_dim(), hidden_dims=[4], optimizer=OPTIMIZER
    )
    trainer = _trainer(tmp_path / "run")
    trainer.test(model, datamodule=dm)

    saved = pd.read_parquet(tmp_path / "run" / TEST_PREDICTIONS_FILE)
    assert list(saved.columns) == ["y_true", "y_pred"]
    recomputed = regression_metrics(saved["y_pred"], saved["y_true"], model.metric_names)
    for name, value in recomputed.items():
        assert trainer.callback_metrics[f"test/{name}"].item() == pytest.approx(value, rel=1e-6), (
            name
        )


def test_classification_mlp_saved_predictions_reproduce_logged_metrics(tmp_path: Path) -> None:
    dm = _classification_datamodule(tmp_path)
    model = BaselineClassificationLightningModule(
        input_dim=dm.get_input_dim(),
        num_classes=dm.get_num_classes(),
        class_labels=dm.get_class_labels(),
        hidden_dims=[4],
        optimizer=OPTIMIZER,
    )
    trainer = _trainer(tmp_path / "run")
    trainer.test(model, datamodule=dm)

    saved = pd.read_parquet(tmp_path / "run" / TEST_PREDICTIONS_FILE)
    assert list(saved.columns) == ["query", "ec_number", "top_ec", "top_score"]
    assert saved["query"].tolist() == ["q1", "q2", "q2"]
    assert all(
        len(ecs) == 2 and len(scores) == 2
        for ecs, scores in zip(saved["top_ec"], saved["top_score"])
    )
    assert all(scores[0] >= scores[1] for scores in saved["top_score"])
    accuracy = np.mean([ecs[0] == true for ecs, true in zip(saved["top_ec"], saved["ec_number"])])
    top_5 = np.mean([true in ecs[:5] for ecs, true in zip(saved["top_ec"], saved["ec_number"])])
    assert trainer.callback_metrics["test/accuracy"].item() == pytest.approx(accuracy)
    assert trainer.callback_metrics["test/top_5_accuracy"].item() == pytest.approx(top_5)
    accuracy = ec_level_accuracy_from_predictions(tmp_path / "run" / TEST_PREDICTIONS_FILE)
    assert accuracy["n_queries"] == 2 and accuracy["n_rows"] == 3
    written = json.loads((tmp_path / "run" / "ec_level_accuracy.json").read_text())
    assert written == {**accuracy, "query_column": "uniprot_id"}
    assert not (tmp_path / "run" / "ec_level_accuracy_no_b.json").exists()


def test_estimators_write_test_predictions(tmp_path: Path) -> None:
    class Classifier:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.ones(x.shape[0], dtype=np.int64)

        def predict_proba(self, x):
            return np.tile(np.array([0.1, 0.9], dtype=np.float32), (x.shape[0], 1))

    dm = _classification_datamodule(tmp_path)
    wrapper = BaselineClassificationEstimatorModule(Classifier(), save_model=False)
    metrics = wrapper.fit_evaluate(dm, tmp_path / "cls")
    saved = pd.read_parquet(tmp_path / "cls" / TEST_PREDICTIONS_FILE)
    labels = dm.get_class_labels()
    assert [list(ecs) for ecs in saved["top_ec"]] == [[labels[1], labels[0]]] * 3
    assert metrics["test/accuracy"] == np.mean(
        [ecs[0] == true for ecs, true in zip(saved["top_ec"], saved["ec_number"])]
    )

    class Regressor:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.full(x.shape[0], 2.0)

    reg_dm = regression_datamodule(tmp_path / "reg")
    metrics = BaselineRegressionEstimatorModule(Regressor(), save_model=False).fit_evaluate(
        reg_dm, tmp_path / "reg_out"
    )
    saved = pd.read_parquet(tmp_path / "reg_out" / TEST_PREDICTIONS_FILE)
    assert (
        regression_metrics(saved["y_pred"], saved["y_true"], ["mse"])["mse"] == metrics["test/mse"]
    )


def test_estimator_classification_writes_only_ec_level_accuracy(tmp_path: Path) -> None:
    class Classifier:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(x.shape[0], dtype=np.int64)

        def predict_proba(self, x):
            return np.tile(np.array([0.9, 0.1], dtype=np.float32), (x.shape[0], 1))

    dm = _classification_datamodule(tmp_path)
    BaselineClassificationEstimatorModule(Classifier(), save_model=False).fit_evaluate(dm, tmp_path / "cls")
    assert (tmp_path / "cls" / "ec_level_accuracy.json").exists()
    assert not (tmp_path / "cls" / "ec_level_accuracy_no_b.json").exists()


def test_retrieval_predictions_write_both_ec_level_accuracy_files(tmp_path: Path) -> None:
    test_path = tmp_path / "test.parquet"
    pd.DataFrame({"rxn_smiles": ["A>>B", "C>>D"], "ec_number": ["1.1.1.1", "3.4.23.B9"]}).to_parquet(test_path)
    dm = SimpleNamespace(endpoint="retrieval", label_col="ec_number", _split_path=lambda split: test_path)
    labels = ["1.1.1.1", "3.4.23.B9"]
    out = tmp_path / "run"
    out.mkdir()
    save_ec_predictions(
        out / TEST_PREDICTIONS_FILE, dm, np.array([[1, 0], [1, 0]]), np.array([[9.0, 1.0], [9.0, 1.0]]), labels
    )

    predictions = out / TEST_PREDICTIONS_FILE
    expected = ec_level_accuracy_from_predictions(predictions)
    expected_no_b = ec_level_accuracy_from_predictions(predictions, drop_b=True)
    assert json.loads((out / "ec_level_accuracy.json").read_text()) == {**expected, "query_column": "rxn_smiles"}
    assert json.loads((out / "ec_level_accuracy_no_b.json").read_text()) == expected_no_b
    assert expected_no_b["n_queries"] == 1 and expected_no_b["k=1"]["level_4_accuracy"] == 100.0
    assert expected["n_queries"] == 2 and expected["k=1"]["level_4_accuracy"] == 50.0


def test_regression_runs_write_no_ec_level_accuracy(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path)
    model = BaselineRegressionLightningModule(input_dim=dm.get_input_dim(), hidden_dims=[4], optimizer=OPTIMIZER)
    _trainer(tmp_path / "run").test(model, datamodule=dm)
    assert not list((tmp_path / "run").glob("ec_level_accuracy*"))
