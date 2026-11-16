from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import optuna
import pytest
from omegaconf import OmegaConf
from optuna.distributions import CategoricalDistribution, FloatDistribution

from hydra_plugins.emulator_optuna4_sweeper.optuna4_sweeper import (
    OPTUNA_TRIAL_NUMBER_TOKEN,
    Optuna4Sweeper,
    _build_trial_overrides,
    _flatten_params_config,
    _replace_trial_number_token,
    create_params_from_overrides,
)


def test_optuna4_parser_supports_log_interval_and_list_choices() -> None:
    distributions, fixed = create_params_from_overrides(
        [
            "optimizer.lr=tag(log, interval(1e-5, 1e-3))",
            "model.hidden_dims=choice([512,256],[1024,512,256])",
            "model.activation=relu",
        ]
    )

    assert isinstance(distributions["optimizer.lr"], FloatDistribution)
    assert distributions["optimizer.lr"].log
    assert isinstance(distributions["model.hidden_dims"], CategoricalDistribution)
    assert distributions["model.hidden_dims"].choices == ("[512,256]", "[1024,512,256]")
    assert fixed["model.activation"] == "relu"


def test_optuna4_replaces_trial_number_token() -> None:
    value = f"/tmp/hpo/trials/trial_{OPTUNA_TRIAL_NUMBER_TOKEN}"

    assert _replace_trial_number_token(value, 12) == "/tmp/hpo/trials/trial_12"


def test_optuna4_flattens_nested_params_config() -> None:
    params = OmegaConf.create(
        {
            "optimizer": {"lr": "choice(1e-5,5e-5)"},
            "model": {"hidden_dims": "choice([512,256],[1024,512,256])"},
            "scheduler.eta_min": "choice(1e-7,1e-6)",
        }
    )

    assert _flatten_params_config(params) == [
        "optimizer.lr=choice(1e-5,5e-5)",
        "model.hidden_dims=choice([512,256],[1024,512,256])",
        "scheduler.eta_min=choice(1e-7,1e-6)",
    ]


def test_optuna4_trial_overrides_win_over_fixed_defaults() -> None:
    overrides = _build_trial_overrides(
        fixed_params={
            "model.estimator.n_estimators": "200",
            "paths.output_dir": f"/tmp/trials/trial_{OPTUNA_TRIAL_NUMBER_TOKEN}",
        },
        trial_params={"model.estimator.n_estimators": "600"},
        trial_number=7,
    )

    fixed_index = overrides.index("model.estimator.n_estimators=200")
    trial_index = overrides.index("++model.estimator.n_estimators=600")
    assert fixed_index < trial_index
    assert overrides[overrides.index("paths.output_dir=/tmp/trials/trial_7")]
    assert overrides[-1] == "+optuna_trial_number=7"


def test_optuna4_marks_stale_running_trials_failed(tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="stale", storage=storage, direction="maximize")
    stale = study.ask()
    study.tell(study.ask(), 0.1)

    import sqlite3

    with sqlite3.connect(tmp_path / "study.db") as con:
        con.execute(
            "update trials set datetime_start='2000-01-01 00:00:00.000000' where number=?",
            (stale.number,),
        )

    sweeper = Optuna4Sweeper(
        sampler=optuna.samplers.TPESampler(seed=1),
        storage=storage,
        study_name="stale",
        stale_trial_timeout_sec=1,
    )
    repaired = optuna.load_study(study_name="stale", storage=storage)

    assert sweeper._fail_stale_trials(repaired) == 1
    states = {trial.number: trial.state for trial in repaired.get_trials(deepcopy=False)}
    assert states[stale.number] == optuna.trial.TrialState.FAIL


def test_fail_trial_uses_the_public_storage_api(tmp_path: Path) -> None:
    import optuna

    from hydra_plugins.emulator_optuna4_sweeper.optuna4_sweeper import fail_trial

    url = f"sqlite:///{tmp_path / 's.db'}"
    storage = optuna.storages.get_storage(url)
    study = optuna.create_study(storage=storage, study_name="s")
    trial = study.ask()
    fail_trial(storage, "s", trial.number, {"reason": "stale"})
    frozen = optuna.load_study(study_name="s", storage=storage).trials[0]
    assert frozen.state == optuna.trial.TrialState.FAIL
    assert frozen.user_attrs["reason"] == "stale"


def test_sweeper_fails_trials_in_an_in_memory_study() -> None:
    import optuna

    from hydra_plugins.emulator_optuna4_sweeper.optuna4_sweeper import Optuna4Sweeper, fail_trial

    sweeper = Optuna4Sweeper(sampler=optuna.samplers.RandomSampler(seed=0), storage=None)
    assert sweeper._get_storage() is sweeper._get_storage()  # one object, no per-call opening
    study = optuna.create_study(storage=sweeper._get_storage(), study_name="mem")
    trial = study.ask()
    fail_trial(sweeper._get_storage(), "mem", trial.number, {"reason": "x"})
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def _sweeper(tmp_path: Path, launcher, **kwargs):
    import optuna

    from hydra_plugins.emulator_optuna4_sweeper.optuna4_sweeper import Optuna4Sweeper

    sweeper = Optuna4Sweeper(
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{tmp_path / 's.db'}",
        study_name="s",
        params=OmegaConf.create({"x": "interval(0, 1)"}),
        **kwargs,
    )
    sweeper.config = OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path / "out")}}})
    sweeper.hydra_context = MagicMock()
    sweeper.launcher = launcher
    return sweeper


def _launcher(values):
    launcher = MagicMock()
    launcher.launch.side_effect = lambda *a, **k: [SimpleNamespace(return_value=next(values))]
    return launcher


def test_nan_results_count_as_failures(tmp_path: Path) -> None:
    import itertools

    sweeper = _sweeper(
        tmp_path,
        _launcher(itertools.repeat(float("nan"))),
        n_trials=5,
        max_consecutive_failed_trials=2,
    )
    with pytest.raises(RuntimeError, match="2 consecutive failed trials"):
        sweeper.sweep([])


def test_resumed_study_reseeds_the_sampler(tmp_path: Path) -> None:
    import itertools

    import optuna

    first = _sweeper(tmp_path, _launcher(iter([0.5])), n_trials=1)
    first.sweep([])
    second = _sweeper(tmp_path, _launcher(itertools.repeat(0.7)), n_trials=2)
    second.sampler = MagicMock(wraps=second.sampler)
    second.sweep([])
    second.sampler.reseed_rng.assert_called_once()
    assert len(optuna.load_study(study_name="s", storage=second.storage).trials) == 2
