from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from src.train import best_checkpoint_score, rebuild_wandb_logger, setup_sweep_logger
from tests.helpers.builders import compose_config


def _trainer(monitor: str, best):
    return SimpleNamespace(
        checkpoint_callback=SimpleNamespace(monitor=monitor, best_model_score=best)
    )


def test_best_checkpoint_score_is_used_only_for_the_monitored_metric() -> None:
    best = torch.tensor(0.7)
    assert best_checkpoint_score(_trainer("val/r2", best), "val/r2") is best
    assert best_checkpoint_score(_trainer("val/loss", best), "val/r2") is None
    assert best_checkpoint_score(SimpleNamespace(checkpoint_callback=None), "val/r2") is None


def test_rebuilt_logger_keeps_entity_and_offline() -> None:
    cfg = OmegaConf.create(
        {
            "logger": {
                "wandb": {
                    "_target_": "builtins.dict",
                    "entity": "team",
                    "offline": True,
                    "project": "p",
                }
            }
        }
    )
    kwargs = rebuild_wandb_logger(cfg, name="n", group="g")
    assert kwargs == {"entity": "team", "offline": True, "project": "p", "name": "n", "group": "g"}


def test_setup_sweep_logger_rebuilds_from_the_wandb_config() -> None:
    from lightning.pytorch.loggers import WandbLogger

    cfg = OmegaConf.create(
        {
            "wandb_group": "grp",
            "wandb_name": "run",
            "logger": {"wandb": {"_target_": "builtins.dict", "entity": "team"}},
        }
    )
    hydra_cfg = SimpleNamespace(mode=SimpleNamespace(name="RUN"), job=SimpleNamespace(num=0))
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=hydra_cfg):
        (logger,) = setup_sweep_logger(cfg, [MagicMock(spec=WandbLogger)])
    assert logger["entity"] == "team" and logger["group"] == "grp" and logger["name"] == "run"


def test_rebuild_wandb_logger_accepts_new_keys_on_a_composed_config() -> None:
    cfg = compose_config("train", ["logger=wandb"])
    cfg.logger.wandb._target_ = "builtins.dict"
    cfg.logger.wandb.save_dir = "out"  # the default interpolates a Hydra runtime value
    kwargs = rebuild_wandb_logger(cfg, name="n", group="g")
    assert kwargs["name"] == "n" and kwargs["group"] == "g"


def _early_stopping_trainer(monitor: str, mode: str, best):
    from lightning.pytorch.callbacks import EarlyStopping

    callback = EarlyStopping(monitor=monitor, mode=mode)
    callback.best_score = best
    return SimpleNamespace(checkpoint_callback=None, early_stopping_callbacks=[callback])


def test_best_score_falls_back_to_early_stopping_best() -> None:
    best = torch.tensor(0.8)
    assert best_checkpoint_score(_early_stopping_trainer("val/r2", "max", best), "val/r2") is best
    assert (
        best_checkpoint_score(_early_stopping_trainer("val/loss", "min", best), "val/r2") is None
    )


def test_early_stopping_best_score_tracks_max_over_epochs() -> None:
    from lightning.pytorch.callbacks import EarlyStopping

    callback = EarlyStopping(monitor="val/r2", mode="max", patience=5)
    for value in (0.2, 0.7, 0.5):
        callback._evaluate_stopping_criteria(torch.tensor(value))
    assert callback.best_score.item() == pytest.approx(0.7)
