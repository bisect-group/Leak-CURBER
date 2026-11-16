from __future__ import annotations

import logging
import warnings

import pytest
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import utils
from src.utils.utils import extras, get_metric_value, task_wrapper


@pytest.fixture(autouse=True)
def rank_zero(monkeypatch):
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)


@pytest.fixture
def restore_warning_filters():
    with warnings.catch_warnings():
        yield


def test_extras_does_nothing_without_an_extras_section(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        extras(OmegaConf.create({}))
    assert "Extras config not found" in caplog.text


def test_extras_ignores_warnings_when_asked(restore_warning_filters) -> None:
    warnings.simplefilter("default")
    extras(OmegaConf.create({"extras": {"ignore_warnings": True}}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("noise")
    assert caught == []


def test_extras_leaves_warnings_alone_by_default(restore_warning_filters) -> None:
    warnings.simplefilter("default")
    extras(OmegaConf.create({"extras": {"ignore_warnings": False}}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("noise")
    assert len(caught) == 1


def test_extras_enforces_tags_and_prints_the_config(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        utils.rich_utils,
        "enforce_tags",
        lambda cfg, save_to_file: calls.append(("tags", save_to_file)),
    )
    monkeypatch.setattr(
        utils.rich_utils,
        "print_config_tree",
        lambda cfg, resolve, save_to_file: calls.append(("tree", resolve, save_to_file)),
    )
    extras(OmegaConf.create({"extras": {"enforce_tags": True, "print_config": True}}))
    assert calls == [("tags", True), ("tree", True, True)]


def test_task_wrapper_returns_the_task_results(tmp_path) -> None:
    @task_wrapper
    def task(cfg):
        return {"acc": 1.0}, {"model": "m"}

    cfg = OmegaConf.create({"paths": {"output_dir": str(tmp_path)}})
    assert task(cfg) == ({"acc": 1.0}, {"model": "m"})


def test_task_wrapper_reraises_task_errors_and_still_logs_the_output_dir(tmp_path, caplog) -> None:
    @task_wrapper
    def task(cfg):
        raise RuntimeError("boom")

    cfg = OmegaConf.create({"paths": {"output_dir": str(tmp_path)}})
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="boom"):
        task(cfg)
    assert f"Output dir: {tmp_path}" in caplog.text


def test_task_wrapper_closes_an_active_wandb_run(monkeypatch, tmp_path) -> None:
    finished = []
    pytest.importorskip("wandb")
    monkeypatch.setattr("wandb.run", object())
    monkeypatch.setattr("wandb.finish", lambda: finished.append(True))

    @task_wrapper
    def task(cfg):
        return {}, {}

    task(OmegaConf.create({"paths": {"output_dir": str(tmp_path)}}))
    assert finished == [True]


def test_task_wrapper_leaves_wandb_alone_without_a_run(monkeypatch, tmp_path) -> None:
    pytest.importorskip("wandb")
    monkeypatch.setattr("wandb.run", None)
    monkeypatch.setattr("wandb.finish", lambda: pytest.fail("finish called"))

    @task_wrapper
    def task(cfg):
        return {}, {}

    task(OmegaConf.create({"paths": {"output_dir": str(tmp_path)}}))


def test_metric_value_is_none_without_a_metric_name() -> None:
    assert get_metric_value({"acc": 1.0}, None) is None
    assert get_metric_value({"acc": 1.0}, "") is None


def test_metric_value_reads_tensors_and_floats() -> None:
    assert get_metric_value({"acc": torch.tensor(0.5)}, "acc") == 0.5
    assert get_metric_value({"acc": "0.25"}, "acc") == 0.25


def test_unknown_metric_name_is_an_error_naming_it() -> None:
    with pytest.raises(Exception, match="Metric value not found! <metric_name=loss>"):
        get_metric_value({"acc": 1.0}, "loss")
