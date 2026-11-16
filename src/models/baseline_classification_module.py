from __future__ import annotations

import json
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
from src.models.baseline_regression_module import build_mlp
from src.utils.ec_level_accuracy import QUERY_COLUMN, write_ec_level_accuracy

TEST_TOP_K = 50


DEFAULT_CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "top_5_accuracy",
    "exact_match",
    "ec_l1_exact_match",
    "ec_l2_exact_match",
    "ec_l3_exact_match",
    "ec_l4_exact_match",
)


def _normalize_metric_names(metrics: Optional[list[str] | tuple[str, ...]]) -> list[str]:
    return normalize_metric_names(metrics, DEFAULT_CLASSIFICATION_METRICS, "classification")


def _ec_prefix(label: str, level: int) -> str:
    parts = str(label).split(".")
    if len(parts) < level:
        parts.extend(["-"] * (level - len(parts)))
    return ".".join(parts[:level])


def _labels_for_indices(indices: np.ndarray, class_labels: list[str]) -> list[str]:
    labels = []
    for value in np.asarray(indices).reshape(-1):
        idx = int(value)
        labels.append(class_labels[idx] if 0 <= idx < len(class_labels) else "")
    return labels


def _classification_scores(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    valid_mask = (targets >= 0) & (targets < num_classes)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    if targets.size == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    accuracy = float(np.mean(preds == targets))
    # Per-class counts via bincount. Out-of-range predictions count for no class.
    support = np.bincount(targets, minlength=num_classes)
    true_positive = np.bincount(targets[preds == targets], minlength=num_classes)
    predicted = np.bincount(preds[(preds >= 0) & (preds < num_classes)], minlength=num_classes)
    # Macro-F1 averages over the union of target and predicted classes, as sklearn does.
    # Balanced accuracy uses classes with support only.
    labels = (support > 0) | (predicted > 0)
    tp, supports, predicted = true_positive[labels], support[labels], predicted[labels]
    precision = np.divide(tp, predicted, out=np.zeros(tp.shape), where=predicted > 0)
    recalls = np.divide(tp, supports, out=np.zeros(tp.shape), where=supports > 0)
    denominator = precision + recalls
    f1s = np.divide(2.0 * precision * recalls, denominator, out=np.zeros(tp.shape), where=denominator > 0)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls[supports > 0])),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=supports)) if supports.sum() else 0.0,
    }


def classification_metrics(
    pred_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    class_labels: list[str],
    metrics: Optional[list[str] | tuple[str, ...]] = None,
    scores: Optional[np.ndarray] = None,
    top_k_indices: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute scalar classification metrics for encoded EC labels.

    ``top_k_indices`` (top-5 class indices per row) replaces ``scores`` for top-5 accuracy.
    """

    selected = _normalize_metric_names(metrics)
    pred_indices = np.asarray(pred_indices, dtype=np.int64).reshape(-1)
    target_indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if pred_indices.shape != target_indices.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {pred_indices.shape} vs {target_indices.shape}"
        )

    values: dict[str, float] = {}
    score_values = _classification_scores(
        pred_indices,
        target_indices,
        num_classes=len(class_labels),
    )
    for name in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
        if name in selected:
            values[name] = score_values[name]

    pred_labels = _labels_for_indices(pred_indices, class_labels)
    target_labels = _labels_for_indices(target_indices, class_labels)
    valid_pairs = [(pred, target) for pred, target in zip(pred_labels, target_labels) if target]
    if "exact_match" in selected:
        values["exact_match"] = (
            float(np.mean([pred == target for pred, target in valid_pairs])) if valid_pairs else 0.0
        )
    for level in range(1, 5):
        name = f"ec_l{level}_exact_match"
        if name in selected:
            values[name] = (
                float(
                    np.mean(
                        [
                            _ec_prefix(pred, level) == _ec_prefix(target, level)
                            for pred, target in valid_pairs
                        ]
                    )
                )
                if valid_pairs
                else 0.0
            )

    if "top_5_accuracy" in selected:
        values["top_5_accuracy"] = (
            _top_k_hit_rate(top_k_indices, target_indices, len(class_labels))
            if top_k_indices is not None
            else _top_k_accuracy(scores, target_indices, len(class_labels), k=5)
        )
    return values


def _top_k_accuracy(scores: Optional[np.ndarray], targets: np.ndarray, num_classes: int, *, k: int) -> float:
    if scores is None:
        return 0.0
    scores = np.asarray(scores)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != targets.shape[0]:
        return 0.0
    return _top_k_hit_rate(_top_k_indices(scores, k), targets, num_classes)


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), scores.shape[1])
    return np.argpartition(scores, -k, axis=1)[:, -k:]


def _top_k_hit_rate(top_k_indices: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    # Same rows as the other metrics: targets outside the class range are dropped.
    valid = (targets >= 0) & (targets < num_classes)
    if not valid.any():
        return 0.0
    return float((top_k_indices[valid] == targets[valid, None]).any(axis=1).mean())


def save_ec_predictions(
    path: Path,
    datamodule,
    top_indices: np.ndarray,
    top_scores: np.ndarray,
    class_labels: list[str],
) -> None:
    """Write one row per test row: query id, true EC and the top-k ECs with scores, best first.
    Also write the EC level accuracy files next to it."""

    columns = [QUERY_COLUMN[datamodule.endpoint], datamodule.label_col]
    rows = pd.read_parquet(datamodule._split_path("test"), columns=columns)
    if len(rows) != len(top_indices):
        raise ValueError(f"Row mismatch: {len(top_indices)} predictions vs {len(rows)} test rows")
    order = np.argsort(-top_scores, axis=1, kind="stable")
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    pd.DataFrame(
        {
            "query": rows.iloc[:, 0].to_numpy(),
            "ec_number": rows.iloc[:, 1].to_numpy(),
            "top_ec": np.asarray(class_labels)[top_indices].tolist(),
            "top_score": top_scores.tolist(),
        }
    ).to_parquet(path, index=False)
    write_ec_level_accuracy(path, datamodule.endpoint)


class BaselineClassificationLightningModule(BaselineLightningModule):
    """Torch MLP classifier for full EC-number prediction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        architecture: str = "mlp",
        hidden_dims: Optional[list[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        optimizer=None,
        scheduler=None,
        compile: bool = False,
        input_cast_dtype: Optional[str] = None,
        metrics: Optional[list[str]] = None,
        class_labels: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        for key in [k for k in list(kwargs) if k.startswith("_")]:
            kwargs.pop(key, None)
        hidden_dims = [int(dim) for dim in list(hidden_dims)] if hidden_dims is not None else None
        super().__init__(optimizer, scheduler, _normalize_metric_names(metrics), int(input_dim))
        self.class_labels = list(class_labels or [str(i) for i in range(int(num_classes))])
        self.save_hyperparameters(ignore=["optimizer", "scheduler"], logger=False)

        if architecture != "mlp":
            raise ValueError("Only architecture='mlp' is built in; add a factory for custom Torch models.")
        self.net = build_mlp(
            int(input_dim),
            hidden_dims=hidden_dims or [1024, 512, 256],
            dropout=dropout,
            activation=activation,
        )
        last_layer = self.net[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("Expected build_mlp to end with nn.Linear.")
        self.net[-1] = Fp32Linear(last_layer.in_features, int(num_classes))
        # Kept as an int; a torch.compile wrapper cannot be indexed as self.net[-1].
        self.num_classes = int(num_classes)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def _step(self, batch):
        x, targets = batch
        x = self._cast_input(x)
        targets = targets.long()
        logits = self.forward(x)
        loss_targets = targets.clone()
        loss_targets[(loss_targets < 0) | (loss_targets >= logits.shape[1])] = -1
        # CrossEntropyLoss(ignore_index=-1) is NaN when no target in the batch is valid;
        # use a zero loss that keeps the graph instead.
        if (loss_targets >= 0).any():
            loss = self.criterion(logits, loss_targets)
        else:
            loss = logits.sum() * 0.0
        return loss, logits.detach(), targets.detach()

    def _loss_batch_size(self, targets) -> int:
        # Weight the epoch-mean loss by the rows that count in the loss (at least 1 so the logger accepts it).
        valid = (targets >= 0) & (targets < self.num_classes)
        return max(int(valid.sum()), 1)

    def _metric_inputs(self, logits: torch.Tensor, targets: torch.Tensor):
        # Keep only the top-5 class indices (sorted, so column 0 is the argmax), not the full
        # score matrix, so epoch buffers stay small.
        top_k = logits.float().topk(min(5, logits.shape[1]), dim=1).indices
        return top_k.cpu(), targets.cpu()

    def _prediction_inputs(self, logits: torch.Tensor, targets: torch.Tensor):
        top = logits.float().topk(min(TEST_TOP_K, logits.shape[1]), dim=1)
        return top.indices.cpu(), top.values.cpu(), targets.cpu()

    def _save_test_predictions(self, top_indices: torch.Tensor, top_scores: torch.Tensor, targets: torch.Tensor) -> None:
        save_ec_predictions(
            Path(self.trainer.default_root_dir) / TEST_PREDICTIONS_FILE,
            self.trainer.datamodule,
            top_indices.numpy(),
            top_scores.numpy(),
            self.class_labels,
        )

    def _compute_metrics(self, top_k: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        top_k_np = top_k.numpy()
        return classification_metrics(
            top_k_np[:, 0],
            targets.numpy(),
            class_labels=self.class_labels,
            metrics=self.metric_names,
            top_k_indices=top_k_np,
        )


class BaselineClassificationEstimatorModule(BaselineEstimatorModule):
    """Hydra wrapper for sklearn, XGBoost, and cuML classifiers."""

    ARTIFACTS_DIR_NAME = "baseline_classifier"
    PREDICT_DTYPE = np.int64

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
        super().__init__(
            estimator, backend, save_model, save_predictions, predict_batch_rows, _normalize_metric_names(metrics)
        )

    def _fit(self, datamodule, artifacts_dir, x_train, y_train) -> None:
        self._class_labels = list(datamodule.get_class_labels())
        (artifacts_dir / "classes.json").write_text(json.dumps(self._class_labels, indent=2) + "\n")
        train_mask = (y_train >= 0) & (y_train < int(datamodule.get_num_classes()))
        self.estimator.fit(x_train[train_mask], y_train[train_mask])

    def _save_test_predictions(self, datamodule, path, x_test, preds, y_test) -> None:
        top_indices, top_scores = self._predict_top_k_in_chunks(x_test, TEST_TOP_K)
        save_ec_predictions(path, datamodule, top_indices, top_scores, self._class_labels)

    def _evaluate_split(self, preds, x_split, y_split):
        top_k = self._predict_top_k_in_chunks(x_split)
        split_metrics = classification_metrics(
            preds,
            y_split,
            class_labels=self._class_labels,
            metrics=self.metric_names,
            top_k_indices=top_k[0] if top_k is not None else None,
        )
        return split_metrics, 1.0 - split_metrics.get("accuracy", 0.0)

    def _metadata_extras(self, artifacts_dir: Path) -> dict[str, str]:
        return {"classes": str(artifacts_dir / "classes.json")}

    def _predict_top_k_in_chunks(self, x_split: np.ndarray, k: int = 5) -> Optional[tuple[np.ndarray, np.ndarray]]:
        # Each chunk is reduced to its top-k indices right away, so memory stays at one chunk
        # of scores.
        method_name = "predict_proba" if hasattr(self.estimator, "predict_proba") else None
        if method_name is None and hasattr(self.estimator, "decision_function"):
            method_name = "decision_function"
        if method_name is None:
            return None
        indices: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        method = getattr(self.estimator, method_name)
        for start in range(0, int(x_split.shape[0]), self.predict_batch_rows):
            stop = min(start + self.predict_batch_rows, int(x_split.shape[0]))
            # Low-precision scores tie across many classes; rank in float32.
            score_chunk = np.asarray(method(x_split[start:stop]), dtype=np.float32)
            if score_chunk.ndim == 1:
                score_chunk = np.stack([-score_chunk, score_chunk], axis=1)
            indices.append(_top_k_indices(score_chunk, k))
            scores.append(np.take_along_axis(score_chunk, indices[-1], axis=1))
        return (np.concatenate(indices), np.concatenate(scores)) if indices else None
