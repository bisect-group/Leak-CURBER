from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule


class OrchestratorProgressCallback(Callback):
    """Writes lightweight progress artifacts for the external orchestrator."""

    def __init__(
        self,
        output_dir: str | Path,
        stage: str = "manual",
        endpoint: str = "",
        split: str = "",
        threshold: str = "",
        seed: str = "",
        job_id: str = "",
        optimized_metric: str = "val/r2",
        mode: str = "max",
        status_interval_sec: float = 2.0,
        batch_interval: int = 25,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.stage = stage
        self.endpoint = endpoint
        self.split = split
        self.threshold = threshold
        self.seed = seed
        self.job_id = job_id
        self.optimized_metric = str(optimized_metric or "val/r2")
        self.mode = mode
        self.status_interval_sec = float(status_interval_sec)
        self.batch_interval = max(1, int(batch_interval))
        self.state_path = self.output_dir / "progress_state.json"
        self.events_path = self.output_dir / "progress_events.jsonl"
        self.started_at = time.time()
        self.last_write = 0.0
        self.best_optimized_metric: float | None = None
        self.train_batch_idx: int | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._write(trainer, "fit_start", force=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.train_batch_idx = None
        self._write(trainer, "train_epoch_start", force=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.train_batch_idx = int(batch_idx)
        if batch_idx % self.batch_interval == 0:
            self._write(trainer, "train_batch_end")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._update_best(trainer)
        self._write(trainer, "validation_epoch_end", force=True)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._update_best(trainer)
        self._write(trainer, "train_epoch_end", force=True)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._update_best(trainer)
        self._write(trainer, "fit_end", force=True, status="complete")

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        self._write(
            trainer,
            "exception",
            force=True,
            status="failed",
            error=f"{exception.__class__.__name__}: {exception}",
        )

    def _write(
        self,
        trainer: Trainer,
        event: str,
        *,
        force: bool = False,
        status: str = "running",
        error: str = "",
    ) -> None:
        if not trainer.is_global_zero:
            return
        now = time.time()
        if not force and now - self.last_write < self.status_interval_sec:
            return
        self.last_write = now
        state = self._state(trainer, event, status=status, error=error)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(state, sort_keys=True) + "\n")
        tmp_path.replace(self.state_path)
        with open(self.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(state, sort_keys=True) + "\n")

    def _state(self, trainer: Trainer, event: str, *, status: str, error: str) -> dict[str, Any]:
        metrics = trainer.callback_metrics
        return {
            "event": event,
            "status": status,
            "error": error,
            "stage": self.stage,
            "endpoint": self.endpoint,
            "split": self.split,
            "threshold": self.threshold,
            "seed": self.seed,
            "job_id": self.job_id,
            "hydra_job_num": self._hydra_job_num(),
            "current_epoch": int(getattr(trainer, "current_epoch", 0)),
            "max_epochs": int(getattr(trainer, "max_epochs", 0) or 0),
            "train_batch_idx": self.train_batch_idx,
            "train_num_batches": self._num_training_batches(trainer),
            "optimized_metric": self.optimized_metric,
            "current_optimized_metric": self._metric(metrics, self.optimized_metric),
            "best_optimized_metric": self.best_optimized_metric,
            "current_val_r2": self._metric(metrics, "val/r2"),
            "current_val_loss": self._metric(metrics, "val/loss"),
            "best_val_r2": self.best_optimized_metric if self.optimized_metric == "val/r2" else None,
            "elapsed_sec": round(time.time() - self.started_at, 3),
            "updated_at": time.time(),
        }

    def _update_best(self, trainer: Trainer) -> None:
        value = self._metric(trainer.callback_metrics, self.optimized_metric)
        if value is None:
            return
        better = max if self.mode == "max" else min
        self.best_optimized_metric = value if self.best_optimized_metric is None else better(self.best_optimized_metric, value)

    @staticmethod
    def _metric(metrics: dict[str, Any], name: str) -> float | None:
        value = metrics.get(name)
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _num_training_batches(trainer: Trainer) -> int | None:
        value = getattr(trainer, "num_training_batches", None)
        if value is None:
            return None
        try:
            if value == float("inf"):
                return None
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def _hydra_job_num(self) -> str:
        if self.output_dir.name.startswith("trial_"):
            return self.output_dir.name.removeprefix("trial_")
        try:
            from hydra.core.hydra_config import HydraConfig

            hydra_cfg = HydraConfig.get()
            return str(hydra_cfg.job.num)
        except Exception:
            return ""
