from __future__ import annotations

from pathlib import Path

import pytest

from omegaconf import OmegaConf

from src.utils import hydra_resolvers  # noqa: F401  (registers `hydra_or`)


def test_paths_resolve_outside_hydra_main(monkeypatch, tmp_path: Path) -> None:
    from hydra import compose, initialize

    monkeypatch.setenv("EMULATOR_PROJECT_ROOT", str(tmp_path))
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="data_processing")
    paths = OmegaConf.to_container(cfg.paths, resolve=True)
    assert paths["output_dir"] == f"{tmp_path}/logs/outputs"
    assert paths["work_dir"]


@pytest.fixture
def hydra_runtime():
    """Set `HydraConfig` like a running @hydra.main app and reset it afterwards."""
    from hydra.core.hydra_config import HydraConfig

    def set_runtime(**runtime) -> None:
        HydraConfig.instance().cfg = OmegaConf.create({"hydra": {"runtime": runtime}})

    yield set_runtime
    HydraConfig.instance().cfg = None


def test_hydra_or_returns_the_current_directory_outside_a_hydra_run(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"out": "${hydra_or:runtime.output_dir}"})
    assert cfg.out == str(tmp_path)


def test_hydra_or_returns_the_given_default_outside_a_hydra_run() -> None:
    cfg = OmegaConf.create({"out": "${hydra_or:runtime.output_dir,fallback}"})
    assert cfg.out == "fallback"


def test_hydra_or_reads_the_runtime_value_inside_a_hydra_run(hydra_runtime) -> None:
    hydra_runtime(output_dir="/runs/1")
    cfg = OmegaConf.create({"out": "${hydra_or:runtime.output_dir,fallback}"})
    assert cfg.out == "/runs/1"


def test_hydra_or_uses_the_default_for_a_key_the_run_does_not_set(
    hydra_runtime, monkeypatch, tmp_path
) -> None:
    hydra_runtime(output_dir="/runs/1")
    monkeypatch.chdir(tmp_path)
    assert OmegaConf.create({"a": "${hydra_or:runtime.choices_x,fallback}"}).a == "fallback"
    assert OmegaConf.create({"a": "${hydra_or:runtime.choices_x}"}).a == str(tmp_path)
