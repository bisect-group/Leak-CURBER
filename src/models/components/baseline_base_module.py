"""Code shared by the baseline regression and classification modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import torch

TEST_PREDICTIONS_FILE = "test_predictions.parquet"

INPUT_CAST_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class Fp32Linear(torch.nn.Linear):
    """Output layer that runs in float32 even under autocast, so loss and metrics see fp32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


def normalize_metric_names(
    metrics: Optional[list[str] | tuple[str, ...]],
    default: tuple[str, ...],
    kind: str,
) -> list[str]:
    names = list(metrics or default)
    valid = set(default)
    invalid = sorted({name for name in names if name not in valid})
    if invalid:
        raise ValueError(f"Unsupported {kind} metric(s): {invalid}. Valid: {sorted(valid)}")
    return names


# Classification and regression modules subclass this base and override only task-specific
# pieces (loss, metrics).
class BaselineLightningModule(L.LightningModule):
    """Shared Lightning steps for the baseline MLPs.

    Subclasses must call ``save_hyperparameters`` in their own ``__init__`` so
    the saved checkpoint hyperparameters keep the subclass signature. They also
    define ``net``, ``criterion``, ``_step`` and ``_compute_metrics``.
    """

    def __init__(self, optimizer, scheduler, metric_names: list[str], input_dim: int) -> None:
        super().__init__()
        # Per-column input standardization. Buffers are saved in the checkpoint, so eval
        # and test use the train-split stats. The defaults (0, 1) leave the input unchanged.
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))
        # Column -> modality block, so an all-zero (missing) block stays 0 after scaling.
        self.register_buffer("input_block_id", torch.zeros(input_dim, dtype=torch.long))
        self._n_blocks = 1
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.metric_names = metric_names
        self._metric_buffers: dict[str, list[tuple[torch.Tensor, ...]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self._test_outputs: list[tuple[torch.Tensor, ...]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        """Optionally compile the network before training starts."""

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, "test")

    def _loss_batch_size(self, targets) -> int:
        """Weight of this batch in the epoch-mean loss."""
        return targets.shape[0]

    def _run_step(self, batch, stage: str):
        loss, outputs, targets = self._step(batch)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=self._loss_batch_size(targets),
        )
        # Batch-averaged metrics (macro-F1, R2, ...) are biased. Buffer compact
        # CPU inputs per batch and compute each metric once over the whole epoch.
        self._metric_buffers[stage].append(self._metric_inputs(outputs, targets))
        if stage == "test":
            self._test_outputs.append(self._prediction_inputs(outputs, targets))
        return loss

    def _metric_inputs(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compact detached CPU tensors to buffer for epoch-level metrics."""

        return outputs.float().cpu(), targets.float().cpu()

    def _prediction_inputs(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """CPU tensors buffered over the test split and saved as test predictions."""

        return self._metric_inputs(outputs, targets)

    def _save_test_predictions(self, *inputs: torch.Tensor) -> None:
        raise NotImplementedError

    def _log_epoch_metrics(self, stage: str) -> None:
        buffer = self._metric_buffers[stage]
        if not buffer:
            return
        inputs = [torch.cat(parts) for parts in zip(*buffer)]
        buffer.clear()
        for name, value in self._compute_metrics(*inputs).items():
            self.log(f"{stage}/{name}", value)

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        if self._test_outputs:
            self._save_test_predictions(*[torch.cat(parts) for parts in zip(*self._test_outputs)])
            self._test_outputs.clear()
        self._log_epoch_metrics("test")

    def set_input_scaling(self, mean: np.ndarray, std: np.ndarray, block_id: np.ndarray) -> None:
        self.input_mean.copy_(torch.as_tensor(mean))
        self.input_std.copy_(torch.as_tensor(std))
        self.input_block_id.copy_(torch.as_tensor(block_id))
        self._n_blocks = int(self.input_block_id.max()) + 1

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Checkpoints from before input scaling lack these buffers; identity values are right
        # for those unscaled runs.
        state = checkpoint["state_dict"]
        for name in ("input_mean", "input_std", "input_block_id"):
            state.setdefault(name, getattr(self, name).detach().clone())
        self._n_blocks = int(state["input_block_id"].max()) + 1

    def _cast_input(self, x: torch.Tensor) -> torch.Tensor:
        # A missing embedding is an all-zero block. Scaling would turn it into
        # -mean/std, so blocks that are all zero before scaling stay 0.
        raw = x.float()
        present = torch.zeros(raw.shape[0], self._n_blocks, device=raw.device)
        present.index_add_(1, self.input_block_id, (raw != 0).float())
        scaled = (raw - self.input_mean) / self.input_std
        x = torch.where((present > 0)[:, self.input_block_id], scaled, 0.0).to(x.dtype)
        cast = self.hparams.input_cast_dtype
        if cast is None:
            return x
        if cast not in INPUT_CAST_DTYPES:
            raise ValueError(f"Unsupported input_cast_dtype '{cast}'. Valid: {sorted(INPUT_CAST_DTYPES)}")
        return x.to(dtype=INPUT_CAST_DTYPES[cast])

    def configure_optimizers(self) -> dict[str, Any]:
        """Instantiate the optimizer and optional epoch scheduler for Lightning."""

        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# Shared base of the classification and regression estimator modules.
class BaselineEstimatorModule:
    """Hydra-instantiated wrapper for sklearn, XGBoost and cuML estimators.

    Non-Torch estimators expect dense arrays rather than streamed Torch batches.
    The wrapper asks the datamodule to materialize each split to memmapped
    ``.npy`` arrays, fits the estimator, evaluates all splits and writes
    metrics, predictions and artifacts under ``ARTIFACTS_DIR_NAME``.
    """

    ARTIFACTS_DIR_NAME: str
    PREDICT_DTYPE: type

    def __init__(
        self,
        estimator,
        backend: str,
        save_model: bool,
        save_predictions: bool,
        predict_batch_rows: int,
        metric_names: list[str],
    ) -> None:
        self.estimator = estimator
        self.backend = backend
        self.save_model = bool(save_model)
        self.save_predictions = bool(save_predictions)
        self.predict_batch_rows = max(1, int(predict_batch_rows))
        self.metric_names = metric_names

    def _fit(self, datamodule, artifacts_dir: Path, x_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    def _evaluate_split(
        self, preds: np.ndarray, x_split: np.ndarray, y_split: np.ndarray
    ) -> tuple[dict[str, float], float]:
        """Return the split metrics and the loss-equivalent value."""

        raise NotImplementedError

    def _save_test_predictions(self, datamodule, path: Path, x_test: np.ndarray, preds: np.ndarray, y_test: np.ndarray) -> None:
        raise NotImplementedError

    def _metadata_extras(self, artifacts_dir: Path) -> dict[str, Any]:
        return {}

    def fit_evaluate(self, datamodule, output_dir: str | Path) -> dict[str, float]:
        """Fit the estimator, evaluate train/val/test and return flat ``split/name`` metrics."""

        output_dir = Path(output_dir)
        artifacts_dir = output_dir / self.ARTIFACTS_DIR_NAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        split_arrays = self._open_materialized_splits(datamodule, output_dir)
        x_train, y_train, _train_info = split_arrays["train"]
        self._fit(datamodule, artifacts_dir, x_train, y_train)

        metrics: dict[str, float] = {}
        predictions_payload: dict[str, str] = {}
        for split, (x_split, y_split, _split_info) in split_arrays.items():
            preds = self._predict_in_chunks(x_split)
            split_metrics, loss = self._evaluate_split(preds, x_split, y_split)
            metrics.update({f"{split}/{name}": value for name, value in split_metrics.items()})
            metrics[f"{split}/loss"] = loss
            if split == "test":
                self._save_test_predictions(datamodule, output_dir / TEST_PREDICTIONS_FILE, x_split, preds, y_split)

            if self.save_predictions:
                pred_path = artifacts_dir / f"{split}_predictions.npy"
                np.save(pred_path, preds, allow_pickle=False)
                predictions_payload[split] = str(pred_path)

        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        metadata = {
            "backend": self.backend,
            "predict_batch_rows": self.predict_batch_rows,
            **self._metadata_extras(artifacts_dir),
            **{split: {key: str(value) for key, value in split_arrays[split][2].items()} for split in ("train", "val", "test")},
            "predictions": predictions_payload,
        }
        (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

        if self.save_model:
            try:
                import joblib

                joblib.dump(self.estimator, artifacts_dir / "model.joblib")
            except ImportError:
                # joblib is optional. Metrics and predictions are still useful
                # even when model serialization is unavailable.
                pass
        return metrics

    def _open_materialized_splits(
        self, datamodule, output_dir: Path
    ) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
        """Open train/val/test arrays, preferring the datamodule's shared cache."""

        if getattr(datamodule, "use_materialized_features", False):
            info = datamodule.ensure_materialized_cache()
            return {
                split: (
                    np.load(split_info["x_path"], mmap_mode="r"),
                    np.load(split_info["y_path"], mmap_mode="r"),
                    split_info,
                )
                for split, split_info in info["splits"].items()
            }

        materialized_dir = output_dir / "materialized_features"
        return {split: datamodule.materialize_split(split, materialized_dir) for split in ("train", "val", "test")}

    def _predict_in_chunks(self, x_split: np.ndarray) -> np.ndarray:
        """Predict a split without forcing the estimator to stage all rows at once."""

        n_rows = int(x_split.shape[0])
        if n_rows == 0:
            return np.empty((0,), dtype=self.PREDICT_DTYPE)
        chunks = [
            np.asarray(self.estimator.predict(x_split[start : start + self.predict_batch_rows]), dtype=self.PREDICT_DTYPE).reshape(-1)
            for start in range(0, n_rows, self.predict_batch_rows)
        ]
        return np.concatenate(chunks, axis=0)
