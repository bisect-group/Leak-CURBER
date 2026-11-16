from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.callbacks.warmup_loader import WarmupLoaderCallback


class _FitLoop:
    def __init__(self) -> None:
        self._combined_loader = "loader"
        self.setups = 0

    def setup_data(self) -> None:
        self.setups += 1


def _trainer(epoch: int, **hparams):
    datamodule = SimpleNamespace(hparams=SimpleNamespace(**hparams))
    return SimpleNamespace(datamodule=datamodule, current_epoch=epoch, fit_loop=_FitLoop())


def _run(trainer) -> None:
    WarmupLoaderCallback().on_train_epoch_start(trainer, None)


def test_loader_is_reloaded_once_when_the_warmup_ends() -> None:
    trainer = _trainer(3, use_warmup_loader=True, warmup_epochs=3)
    _run(trainer)
    assert trainer.fit_loop._combined_loader is None
    assert trainer.fit_loop.setups == 1


@pytest.mark.parametrize("epoch", [0, 2, 4])
def test_loader_is_kept_in_all_other_epochs(epoch: int) -> None:
    trainer = _trainer(epoch, use_warmup_loader=True, warmup_epochs=3)
    _run(trainer)
    assert trainer.fit_loop._combined_loader == "loader"
    assert trainer.fit_loop.setups == 0


@pytest.mark.parametrize(
    "hparams",
    [
        {"use_warmup_loader": False, "warmup_epochs": 3},
        {"use_warmup_loader": True, "warmup_epochs": 0},
        {},
    ],
)
def test_loader_is_kept_when_warmup_is_off_or_has_no_epochs(hparams) -> None:
    trainer = _trainer(0, **hparams)
    trainer.current_epoch = hparams.get("warmup_epochs", 0)
    _run(trainer)
    assert trainer.fit_loop.setups == 0


def test_a_trainer_without_a_datamodule_is_ignored() -> None:
    trainer = _trainer(3, use_warmup_loader=True, warmup_epochs=3)
    trainer.datamodule = None
    _run(trainer)
    assert trainer.fit_loop.setups == 0
