from __future__ import annotations

import logging

import pytest
from omegaconf import OmegaConf

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

_TARGET = "lightning.pytorch.callbacks.EarlyStopping"


def test_empty_callback_config_gives_no_callbacks(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert instantiate_callbacks(None) == []
    assert "No callback configs found" in caplog.text


def test_callbacks_are_built_from_entries_with_a_target() -> None:
    cfg = OmegaConf.create(
        {
            "early_stopping": {"_target_": _TARGET, "monitor": "val/loss", "patience": 4},
            "note": "ignored",
            "no_target": {"patience": 1},
        }
    )
    callbacks = instantiate_callbacks(cfg)
    assert [type(c).__name__ for c in callbacks] == ["EarlyStopping"]
    assert (callbacks[0].monitor, callbacks[0].patience) == ("val/loss", 4)


def test_callback_config_must_be_a_dict_config() -> None:
    with pytest.raises(TypeError, match="Callbacks config must be a DictConfig"):
        instantiate_callbacks([{"_target_": _TARGET}])


def test_empty_logger_config_gives_no_loggers(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert instantiate_loggers({}) == []
    assert "No logger configs found" in caplog.text


def test_loggers_are_built_from_entries_with_a_target(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "csv": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": str(tmp_path),
                "name": "run",
            },
            "skipped": {"flag": True},
        }
    )
    loggers = instantiate_loggers(cfg)
    assert [type(lg).__name__ for lg in loggers] == ["CSVLogger"]
    assert loggers[0].name == "run"


def test_logger_config_must_be_a_dict_config() -> None:
    with pytest.raises(TypeError, match="Logger config must be a DictConfig"):
        instantiate_loggers([{"_target_": "x"}])
