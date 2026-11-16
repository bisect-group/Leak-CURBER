from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.components.baseline_base_module import (
    BaselineEstimatorModule,
    BaselineLightningModule,
    TEST_PREDICTIONS_FILE,
    Fp32Linear,
    normalize_metric_names,
)


def build_mlp(
    input_dim: int,
    *,
    hidden_dims: list[int],
    dropout: float = 0.1,
    activation: str = "relu",
) -> nn.Module:
    """Build a simple feed-forward regressor ending in one scalar output.

    Args:
        input_dim: Width of the concatenated embedding vector produced by the
            baseline regression datamodule.
        hidden_dims: Ordered hidden-layer widths.
        dropout: Dropout probability inserted after each activation. Set to
            ``0`` to disable dropout.
        activation: Name of the activation function used after hidden layers.

    Returns:
        A ``torch.nn.Sequential`` network whose final layer has output size 1
        and runs in float32 under autocast.
    """

    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    if activation not in activations:
        raise ValueError(f"Unsupported activation '{activation}'. Valid: {sorted(activations)}")

    layers: list[nn.Module] = []
    previous_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        # Each hidden block is Linear -> activation -> optional dropout.
        layers.append(nn.Linear(previous_dim, int(hidden_dim)))
        layers.append(activations[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        previous_dim = int(hidden_dim)
    layers.append(Fp32Linear(previous_dim, 1))
    return nn.Sequential(*layers)


DEFAULT_REGRESSION_METRICS = ("mse", "rmse", "mae", "r2", "pearson", "pcc", "scc")


def _normalize_metric_names(metrics: Optional[list[str] | tuple[str, ...]]) -> list[str]:
    return normalize_metric_names(metrics, DEFAULT_REGRESSION_METRICS, "regression")


def _finite_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def regression_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[list[str] | tuple[str, ...]] = None,
) -> dict[str, float]:
    """Compute common scalar regression metrics from predictions and targets.

    Inputs are flattened so callers can pass ``(N,)`` or ``(N, 1)`` arrays. The
    returned names intentionally omit split prefixes; callers add prefixes such
    as ``train/`` or ``test/`` depending on where the metrics are logged.
    """

    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if preds.shape != targets.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {targets.shape}")

    selected = _normalize_metric_names(metrics)
    errors = preds - targets
    mse = float(np.mean(errors**2))
    values: dict[str, float] = {}
    if "mse" in selected:
        values["mse"] = mse
    if "rmse" in selected:
        values["rmse"] = float(np.sqrt(mse))
    if "mae" in selected:
        values["mae"] = float(np.mean(np.abs(errors)))
    if "r2" in selected:
        target_var = float(np.sum((targets - targets.mean()) ** 2))
        values["r2"] = 0.0 if target_var == 0 else float(1.0 - np.sum(errors**2) / target_var)
    if "pearson" in selected or "pcc" in selected:
        pearson = _finite_corrcoef(preds, targets)
        if "pearson" in selected:
            values["pearson"] = pearson
        if "pcc" in selected:
            values["pcc"] = pearson
    if "scc" in selected:
        pred_ranks = pd.Series(preds).rank(method="average").to_numpy(dtype=np.float64)
        target_ranks = pd.Series(targets).rank(method="average").to_numpy(dtype=np.float64)
        values["scc"] = _finite_corrcoef(pred_ranks, target_ranks)
    return values


def save_regression_predictions(path: Path, targets: np.ndarray, preds: np.ndarray) -> None:
    """Write one ``y_true``/``y_pred`` row per test row, in test-split order."""

    pd.DataFrame({"y_true": np.asarray(targets), "y_pred": np.asarray(preds)}).to_parquet(path, index=False)


class BaselineRegressionLightningModule(BaselineLightningModule):
    """Torch/Lightning baseline regressor over concatenated embeddings.

    This module is intended for the streaming batches produced by
    ``BaselineRegressionDataModule``. The built-in architecture is a configurable
    MLP that predicts one scalar regression target per row. Optimizer and
    scheduler factories are normally provided by Hydra config.
    """

    def __init__(
        self,
        input_dim: int,
        architecture: str = "mlp",
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        optimizer=None,
        scheduler=None,
        compile: bool = False,
        input_cast_dtype: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize the Lightning regression module.

        Args:
            input_dim: Feature width returned by ``datamodule.get_input_dim()``.
            architecture: Model architecture name. Currently only ``"mlp"`` is
                implemented.
            hidden_dims: Hidden-layer widths for the MLP. Defaults to
                ``[1024, 512, 256]``.
            dropout: Dropout probability after hidden activations.
            activation: Hidden activation name passed to ``build_mlp``.
            optimizer: Callable/factory that accepts ``params=...`` and returns
                a Torch optimizer.
            scheduler: Optional callable/factory that accepts
                ``optimizer=...`` and returns a scheduler.
            compile: Whether to run ``torch.compile`` on the network during
                fit setup.
            input_cast_dtype: Optional explicit feature tensor cast. ``None``
                preserves the dataloader dtype and lets Lightning autocast
                control model compute precision.
            metrics: Regression metric names to log for train/val/test.
            **kwargs: Extra Hydra/Lightning keys. Private keys beginning with
                ``_`` are ignored before hyperparameters are saved.
        """

        for key in [k for k in list(kwargs) if k.startswith("_")]:
            kwargs.pop(key, None)
        hidden_dims = [int(dim) for dim in list(hidden_dims)] if hidden_dims is not None else None
        super().__init__(optimizer, scheduler, _normalize_metric_names(metrics), int(input_dim))
        self.save_hyperparameters(ignore=["optimizer", "scheduler"], logger=False)

        if architecture != "mlp":
            raise ValueError("Only architecture='mlp' is built in; add a factory for custom Torch models.")
        self.net = build_mlp(
            int(input_dim),
            hidden_dims=hidden_dims or [1024, 512, 256],
            dropout=dropout,
            activation=activation,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one scalar value per input row."""

        return self.net(x).squeeze(-1)

    def _step(self, batch):
        """Shared batch logic for train/validation/test steps."""

        x, targets = batch
        x = self._cast_input(x)
        targets = targets.float()
        preds = self.forward(x)
        loss = self.criterion(preds, targets)
        return loss, preds.detach(), targets.detach()

    def _save_test_predictions(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        save_regression_predictions(
            Path(self.trainer.default_root_dir) / TEST_PREDICTIONS_FILE, targets.numpy(), preds.numpy()
        )

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        return regression_metrics(preds.float().cpu().numpy(), targets.float().cpu().numpy(), self.metric_names)


class BaselineRegressionEstimatorModule(BaselineEstimatorModule):
    """Hydra-instantiated wrapper for sklearn and XGBoost regressors.

    Splits are materialized to memmapped ``.npy`` arrays, the estimator is
    trained, all splits are evaluated, and results go under ``baseline_estimator``.
    """

    ARTIFACTS_DIR_NAME = "baseline_estimator"
    PREDICT_DTYPE = np.float32

    def __init__(
        self,
        estimator,
        backend: str = "sklearn",
        save_model: bool = True,
        save_predictions: bool = True,
        predict_batch_rows: int = 32768,
        metrics: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Create the estimator wrapper.

        Args:
            estimator: A scikit-learn/XGBoost-style object with ``fit`` and
                ``predict`` methods.
            backend: Descriptive backend name stored in metadata.
            save_model: Whether to save the fitted estimator with ``joblib`` if
                joblib is installed.
            save_predictions: Whether to write ``*_predictions.npy`` files.
            predict_batch_rows: Maximum number of rows passed to
                ``estimator.predict`` at once. This avoids very large GPU
                prediction allocations for XGBoost on large splits.
            metrics: Regression metric names to write for train/val/test.
            **kwargs: Extra Hydra keys. Private keys beginning with ``_`` are
                ignored.
        """

        super().__init__(
            estimator, backend, save_model, save_predictions, predict_batch_rows, _normalize_metric_names(metrics)
        )

    def _fit(self, datamodule, artifacts_dir, x_train, y_train) -> None:
        self.estimator.fit(x_train, y_train)

    def _save_test_predictions(self, datamodule, path, x_test, preds, y_test) -> None:
        save_regression_predictions(path, y_test, preds)

    def _evaluate_split(self, preds, x_split, y_split):
        split_metrics = regression_metrics(preds, y_split, self.metric_names)
        # Use MSE as a loss-equivalent value for parity with the Torch MLP.
        loss_metrics = split_metrics if "mse" in split_metrics else regression_metrics(preds, y_split, ["mse"])
        return split_metrics, loss_metrics["mse"]
