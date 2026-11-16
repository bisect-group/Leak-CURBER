from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from src.models.baseline_regression_module import (
    BaselineRegressionEstimatorModule,
    BaselineRegressionLightningModule,
    regression_metrics,
)
from tests.helpers.builders import logged_metrics, regression_datamodule


def test_epoch_r2_equals_full_set_metric_across_batches(tmp_path: Path) -> None:
    module = BaselineRegressionLightningModule(
        input_dim=2,
        hidden_dims=[2],
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        metrics=["r2", "pcc"],
    )
    module.log = MagicMock()
    module.trainer = MagicMock(default_root_dir=str(tmp_path))
    preds = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 11.0])]
    targets = [torch.tensor([1.5, 2.5, 2.0]), torch.tensor([10.0, 13.0])]
    module._step = MagicMock(
        side_effect=[(torch.tensor(0.0), p, t) for p, t in zip(preds, targets)]
    )
    for _ in range(2):
        module._run_step(None, "test")
    module.on_test_epoch_end()

    expected = regression_metrics(
        torch.cat(preds).numpy(), torch.cat(targets).numpy(), ["r2", "pcc"]
    )
    logged = logged_metrics(module, "test/")
    for name, value in expected.items():
        assert np.isclose(logged[f"test/{name}"], value), name


def test_lightning_module_fast_steps(tmp_path: Path) -> None:
    dm = regression_datamodule(tmp_path)
    model = BaselineRegressionLightningModule(
        input_dim=dm.get_input_dim(),
        hidden_dims=[4],
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )
    batch = next(iter(dm.train_dataloader()))

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_regression_metrics_include_configurable_pcc_and_scc() -> None:
    preds = np.asarray([1.0, 2.0, 3.0, 4.0])
    targets = np.asarray([1.1, 1.9, 3.2, 3.8])

    metrics = regression_metrics(preds, targets, ["rmse", "pcc", "scc"])

    assert set(metrics) == {"rmse", "pcc", "scc"}
    assert np.isfinite(metrics["pcc"])
    assert np.isfinite(metrics["scc"])


def test_estimator_honors_configured_metrics(tmp_path: Path) -> None:
    from sklearn.linear_model import Ridge

    dm = regression_datamodule(tmp_path)
    model = BaselineRegressionEstimatorModule(
        Ridge(alpha=1.0),
        save_model=False,
        save_predictions=False,
        metrics=["r2", "scc"],
    )

    metrics = model.fit_evaluate(dm, tmp_path / "outputs")

    assert "test/r2" in metrics
    assert "test/scc" in metrics
    assert "test/rmse" not in metrics
    assert "test/loss" in metrics


def test_lightning_module_preserves_feature_dtype_by_default(tmp_path: Path) -> None:
    model = BaselineRegressionLightningModule(
        input_dim=2,
        hidden_dims=[2],
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
    )
    seen: dict[str, torch.dtype] = {}

    def fake_forward(x: torch.Tensor) -> torch.Tensor:
        seen["dtype"] = x.dtype
        return torch.zeros(x.shape[0], dtype=torch.float32)

    model.forward = fake_forward  # type: ignore[method-assign]
    batch = (torch.ones(2, 2, dtype=torch.float16), torch.ones(2, dtype=torch.float32))

    loss = model.training_step(batch, 0)

    assert seen["dtype"] == torch.float16
    assert torch.isfinite(loss)


def test_sklearn_estimator_fit_evaluate(tmp_path: Path) -> None:
    from sklearn.linear_model import Ridge

    dm = regression_datamodule(tmp_path)
    model = BaselineRegressionEstimatorModule(Ridge(alpha=1.0), save_model=False)

    metrics = model.fit_evaluate(dm, tmp_path / "outputs")

    assert "test/rmse" in metrics
    assert (tmp_path / "outputs" / "baseline_estimator" / "metrics.json").exists()


def test_estimator_uses_shared_materialized_cache_without_job_local_copy(tmp_path: Path) -> None:
    from sklearn.linear_model import Ridge

    dm = regression_datamodule(
        tmp_path,
        materialized_cache_root=str(tmp_path / "shared_materialized"),
        materialized_feature_dtype="float32",
        use_materialized_features=True,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    model = BaselineRegressionEstimatorModule(
        Ridge(alpha=1.0), save_model=False, save_predictions=False
    )
    output_dir = tmp_path / "outputs"

    metrics = model.fit_evaluate(dm, output_dir)

    assert "val/r2" in metrics
    assert not (output_dir / "materialized_features").exists()
    metadata = json.loads((output_dir / "baseline_estimator" / "metadata.json").read_text())
    assert str(tmp_path / "shared_materialized") in metadata["train"]["x_path"]


def test_estimator_predicts_in_configured_chunks() -> None:
    class ChunkRecorder:
        def __init__(self) -> None:
            self.chunks: list[int] = []

        def predict(self, x: np.ndarray) -> np.ndarray:
            self.chunks.append(int(x.shape[0]))
            return np.arange(x.shape[0], dtype=np.float32)

    estimator = ChunkRecorder()
    model = BaselineRegressionEstimatorModule(estimator, predict_batch_rows=3)

    preds = model._predict_in_chunks(np.zeros((8, 2), dtype=np.float32))

    assert estimator.chunks == [3, 3, 2]
    assert preds.shape == (8,)


def _scaled_module():
    from src.models.baseline_regression_module import BaselineRegressionLightningModule

    return BaselineRegressionLightningModule(
        input_dim=3, hidden_dims=[2], optimizer=lambda params: torch.optim.Adam(params, lr=1e-3)
    )


def test_module_standardizes_present_blocks_and_keeps_missing_blocks_zero() -> None:
    module = _scaled_module()
    module.set_input_scaling(
        np.array([1.0, 1.0, 3.0]), np.array([2.0, 2.0, 1.0]), np.array([0, 0, 1])
    )
    out = module._cast_input(torch.tensor([[3.0, 1.0, 4.0], [0.0, 0.0, 4.0], [3.0, 1.0, 0.0]]))
    assert out.tolist() == [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    assert {"input_mean", "input_std", "input_block_id"} <= set(module.state_dict())


def test_checkpoint_without_scaling_buffers_loads_as_identity() -> None:
    module = _scaled_module()
    state = {k: v for k, v in module.state_dict().items() if not k.startswith("input_")}
    checkpoint = {"state_dict": state}
    module.on_load_checkpoint(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    x = torch.tensor([[3.0, 1.0, 4.0]])
    assert module._cast_input(x).tolist() == x.tolist()


def test_regression_output_is_float32_under_bf16_autocast() -> None:
    module = BaselineRegressionLightningModule(input_dim=4, hidden_dims=[8], optimizer=None)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden = module.net[0](torch.randn(3, 4))
        preds = module(torch.randn(3, 4))
    assert hidden.dtype == torch.bfloat16
    assert preds.dtype == torch.float32
