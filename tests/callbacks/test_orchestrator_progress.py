from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.callbacks.orchestrator_progress import OrchestratorProgressCallback


def test_best_metric_follows_mode(tmp_path: Path) -> None:
    callback = OrchestratorProgressCallback(tmp_path, optimized_metric="val/loss", mode="min")
    for value in (3.0, 1.0, 2.0):
        trainer = SimpleNamespace(callback_metrics={"val/loss": torch.tensor(value)})
        callback._update_best(trainer)
    assert callback.best_optimized_metric == 1.0
