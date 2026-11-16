from __future__ import annotations

import contextlib
import fcntl
import logging
import math
import os
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlparse

import optuna
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import (
    ChoiceSweep,
    IntervalSweep,
    Override,
    RangeSweep,
    Transformer,
)
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState


log = logging.getLogger(__name__)
OPTUNA_TRIAL_NUMBER_TOKEN = "__OPTUNA_TRIAL_NUMBER__"


def _stringify_override_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stringify_override_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_stringify_override_value(val)}" for key, val in value.items()) + "}"
    return str(value)


def _replace_trial_number_token(value: str, trial_number: int) -> str:
    return value.replace(OPTUNA_TRIAL_NUMBER_TOKEN, str(trial_number))


def _build_trial_overrides(
    fixed_params: Mapping[str, Any],
    trial_params: Mapping[str, Any],
    trial_number: int,
) -> tuple[str, ...]:
    fixed = {
        name: _replace_trial_number_token(str(value), trial_number)
        if OPTUNA_TRIAL_NUMBER_TOKEN in str(value)
        else value
        for name, value in fixed_params.items()
    }
    trial = {
        name: _replace_trial_number_token(str(value), trial_number)
        if OPTUNA_TRIAL_NUMBER_TOKEN in str(value)
        else value
        for name, value in trial_params.items()
    }
    # Hydra applies later overrides last. Keep fixed task/config values first
    # and append sampled Optuna values after them so HPO parameters win over
    # defaults from hydra.common / pipeline profiles.
    return tuple(
        [f"{name}={_stringify_override_value(value)}" for name, value in fixed.items()]
        + [f"++{name}={_stringify_override_value(value)}" for name, value in trial.items()]
        + [f"+optuna_trial_number={trial_number}"]
    )


def _distribution_from_override(override: Override) -> Any:
    if not override.is_sweep_override():
        return override.get_value_element_as_str()

    value = override.value()
    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        choices = [
            _stringify_override_value(choice)
            for choice in override.sweep_iterator(transformer=Transformer.encode)
        ]
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        assert isinstance(value, RangeSweep)
        assert value.start is not None
        assert value.stop is not None
        if value.shuffle:
            choices = [
                _stringify_override_value(choice)
                for choice in override.sweep_iterator(transformer=Transformer.encode)
            ]
            return CategoricalDistribution(choices)
        step = value.step if value.step is not None else 1
        if any(isinstance(v, float) for v in (value.start, value.stop, step)):
            return FloatDistribution(float(value.start), float(value.stop), step=float(step))
        return IntDistribution(int(value.start), int(value.stop), step=int(step))

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        assert value.start is not None
        assert value.end is not None
        log_scale = "log" in value.tags
        if isinstance(value.start, int) and isinstance(value.end, int):
            return IntDistribution(int(value.start), int(value.end), log=log_scale)
        return FloatDistribution(float(value.start), float(value.end), log=log_scale)

    raise NotImplementedError(f"Unsupported Optuna sweep override: {override}")


def create_params_from_overrides(arguments: list[str]) -> tuple[dict[str, BaseDistribution], dict[str, Any]]:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(arguments)
    distributions: dict[str, BaseDistribution] = {}
    fixed_params: dict[str, Any] = {}
    for override in parsed:
        param_name = override.get_key_element()
        value = _distribution_from_override(override)
        if isinstance(value, BaseDistribution):
            distributions[param_name] = value
        else:
            fixed_params[param_name] = value
    return distributions, fixed_params


def _flatten_params_config(params: Any, prefix: str = "") -> list[str]:
    if OmegaConf.is_config(params):
        params = OmegaConf.to_container(params, resolve=True)
    if not isinstance(params, Mapping):
        return []

    overrides: list[str] = []
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) or OmegaConf.is_config(value):
            overrides.extend(_flatten_params_config(value, name))
        else:
            overrides.append(f"{name}={_stringify_override_value(value)}")
    return overrides


def fail_trial(
    storage: optuna.storages.BaseStorage, study_name: str, trial_number: int, user_attrs: dict
) -> None:
    """Annotate a running trial and mark it FAIL through Optuna's public storage API.

    The caller passes the storage object it already uses (from `optuna.storages.get_storage`),
    so an in-memory study works and no storage is opened per call.
    """
    study_id = storage.get_study_id_from_name(study_name)
    trial_id = storage.get_trial_id_from_study_id_trial_number(study_id, trial_number)
    try:
        for key, value in user_attrs.items():
            storage.set_trial_user_attr(trial_id, key, value)
    except Exception:
        log.exception("Could not annotate trial %s before failing it.", trial_number)
    storage.set_trial_state_values(trial_id, state=TrialState.FAIL, values=None)


def _sqlite_lock_path(storage: Optional[str], study_name: Optional[str]) -> Optional[Path]:
    if not storage or not str(storage).startswith("sqlite:///"):
        return None
    parsed = urlparse(str(storage))
    path = Path(parsed.path)
    lock_name = f".{study_name or 'optuna_study'}.lock"
    return path.parent / lock_name


def _orchestrator_parent_alive() -> bool:
    parent_pid_text = os.environ.get("ORCHESTRATOR_PARENT_PID")
    if not parent_pid_text:
        return True
    try:
        parent_pid = int(parent_pid_text)
    except ValueError:
        return True
    if parent_pid <= 1:
        return True
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def _study_lock(storage: Optional[str], study_name: Optional[str]):
    lock_path = _sqlite_lock_path(storage, study_name)
    if lock_path is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Optuna4Sweeper(Sweeper):
    def __init__(
        self,
        sampler: Any,
        direction: str = "maximize",
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 50,
        n_jobs: int = 1,
        max_consecutive_failed_trials: int = 5,
        stale_trial_timeout_sec: int = 7200,
        stale_trial_check_interval_sec: int = 300,
        fail_stale_running_trials: bool = True,
        params: Optional[DictConfig] = None,
    ) -> None:
        self.sampler = sampler
        self.direction = direction
        self.storage = storage
        self._storage_obj: Optional[optuna.storages.BaseStorage] = None
        self.study_name = study_name
        self.n_trials = int(n_trials)
        self.n_jobs = max(1, int(n_jobs))
        self.max_consecutive_failed_trials = max(1, int(max_consecutive_failed_trials))
        self.stale_trial_timeout_sec = int(stale_trial_timeout_sec)
        self.stale_trial_check_interval_sec = int(stale_trial_check_interval_sec)
        self.fail_stale_running_trials = bool(fail_stale_running_trials)
        self.params = params or OmegaConf.create({})
        self.job_idx = 0
        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.launcher = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.hydra_context = hydra_context
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config,
            hydra_context=hydra_context,
            task_function=task_function,
        )

    def _get_storage(self) -> optuna.storages.BaseStorage:
        # One storage object for the study and for fail_trial (None gives an in-memory one).
        if self._storage_obj is None:
            self._storage_obj = optuna.storages.get_storage(self.storage)
        return self._storage_obj

    def sweep(self, arguments: list[str]) -> None:
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.launcher is not None

        params_conf = _flatten_params_config(self.params)
        params_conf.extend(arguments)
        distributions, fixed_params = create_params_from_overrides(params_conf)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self._get_storage(),
            sampler=self.sampler,
            direction=self.direction,
            load_if_exists=True,
        )
        log.info(f"Optuna 4 study: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Direction: {self.direction}")
        consecutive_failures = 0
        last_stale_check_at = 0.0
        if study.trials:
            # A fresh RNG stream keeps a resumed run from repeating earlier random start points
            # (so a resumed run is not bit-reproducible).
            self.sampler.reseed_rng()

        while True:
            trial = None
            with _study_lock(self.storage, self.study_name):
                study = optuna.load_study(
                    study_name=study.study_name,
                    storage=self._get_storage(),
                    sampler=self.sampler,
                )
                if not _orchestrator_parent_alive():
                    failed_count = self._fail_active_trials(
                        study,
                        "Optuna worker stopped because orchestrator parent process disappeared.",
                    )
                    if failed_count:
                        log.warning("Marked %s active Optuna trial(s) as FAIL after parent death.", failed_count)
                    raise KeyboardInterrupt("orchestrator parent process disappeared")
                now_monotonic = time.monotonic()
                if (
                    self.fail_stale_running_trials
                    and self.stale_trial_timeout_sec > 0
                    and now_monotonic - last_stale_check_at >= self.stale_trial_check_interval_sec
                ):
                    stale_count = self._fail_stale_trials(study)
                    if stale_count:
                        log.warning("Marked %s stale Optuna RUNNING/WAITING trial(s) as FAIL.", stale_count)
                    last_stale_check_at = now_monotonic
                completed_count = len(
                    study.get_trials(
                        deepcopy=False,
                        states=(TrialState.COMPLETE,),
                    )
                )
                running_count = len(
                    study.get_trials(
                        deepcopy=False,
                        states=(TrialState.RUNNING, TrialState.WAITING),
                    )
                )
                if completed_count >= self.n_trials:
                    break
                if completed_count + running_count < self.n_trials:
                    trial = study.ask(distributions)

            if trial is None:
                # Other concurrent workers already own enough running trials to
                # reach the completed-trial target. Wait for one to finish or
                # fail before asking for more.
                time.sleep(5.0)
                continue

            overrides: Sequence[Sequence[str]] = [
                _build_trial_overrides(fixed_params, trial.params, trial.number)
            ]
            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
            self.job_idx += len(returns)

            ret = returns[0]
            try:
                value = float(ret.return_value)
                # A non-finite result is a failed trial and counts toward max_consecutive_failed_trials.
                if not math.isfinite(value):
                    raise ValueError(f"Trial returned a non-finite value: {value}")
                if not self._tell_if_running(trial.number, value=value, state=TrialState.COMPLETE):
                    consecutive_failures += 1
                    continue
                consecutive_failures = 0
            except Exception as exc:
                if not self._tell_if_running(trial.number, value=None, state=TrialState.FAIL):
                    consecutive_failures += 1
                    continue
                consecutive_failures += 1
                log.exception(
                    "Optuna trial %s failed (%s/%s consecutive failures); continuing.",
                    trial.number,
                    consecutive_failures,
                    self.max_consecutive_failed_trials,
                )
                if consecutive_failures >= self.max_consecutive_failed_trials:
                    raise RuntimeError(
                        f"Optuna worker reached {consecutive_failures} consecutive failed trials"
                    ) from exc

        self._save_results(study)

    def _tell_if_running(self, trial_number: int, *, value: Optional[float], state: TrialState) -> bool:
        with _study_lock(self.storage, self.study_name):
            study = optuna.load_study(
                study_name=self.study_name,
                storage=self._get_storage(),
                sampler=self.sampler,
            )
            frozen = next(
                (trial for trial in study.get_trials(deepcopy=False) if trial.number == trial_number),
                None,
            )
            if frozen is None:
                log.warning("Optuna trial %s disappeared before tell(%s).", trial_number, state.name)
                return False
            if frozen.state not in {TrialState.RUNNING, TrialState.WAITING}:
                log.warning(
                    "Optuna trial %s is already %s before tell(%s); leaving existing state unchanged.",
                    trial_number,
                    frozen.state.name,
                    state.name,
                )
                return False
            if state == TrialState.COMPLETE:
                study.tell(frozen.number, value, state=state)
            else:
                study.tell(frozen.number, state=state)
            return True

    def _fail_stale_trials(self, study: optuna.Study) -> int:
        threshold = datetime.now()
        stale_count = 0
        for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING)):
            started_at = frozen.datetime_start
            if started_at is None:
                continue
            age_sec = (threshold - started_at.replace(tzinfo=None)).total_seconds()
            if age_sec < self.stale_trial_timeout_sec:
                continue
            message = (
                f"Marked failed by Optuna4Sweeper after {age_sec:.0f}s without completion "
                f"(stale_trial_timeout_sec={self.stale_trial_timeout_sec})."
            )
            try:
                # `Study.tell()` goes through the regular trial-finalization
                # path and can fail on old RUNNING rows from interrupted
                # workers. For stale recovery we only need to atomically move
                # the row out of RUNNING/WAITING so new trials can be asked.
                fail_trial(
                    self._get_storage(),
                    study.study_name,
                    frozen.number,
                    {"stale_failure_reason": message, "stale_failure_at": datetime.now().isoformat()},
                )
                stale_count += 1
            except Exception:
                log.exception("Could not mark stale trial %s as FAIL.", frozen.number)
        return stale_count

    def _fail_active_trials(self, study: optuna.Study, message: str) -> int:
        failed_count = 0
        for frozen in study.get_trials(deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING)):
            try:
                fail_trial(
                    self._get_storage(),
                    study.study_name,
                    frozen.number,
                    {
                        "orchestrator_parent_death_reason": message,
                        "orchestrator_parent_death_at": datetime.now().isoformat(),
                    },
                )
                failed_count += 1
            except Exception:
                log.exception("Could not mark active trial %s as FAIL after parent death.", frozen.number)
        return failed_count

    def _save_results(self, study: optuna.Study) -> None:
        completed = [
            trial
            for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if trial.value is not None
        ]
        if not completed:
            log.warning("No completed Optuna trials; skipping optimization_results.yaml")
            return
        best_trial = study.best_trial
        results = {
            "name": "optuna4",
            "study_name": study.study_name,
            "best_params": dict(best_trial.params),
            "best_value": best_trial.value,
            "best_trial_number": best_trial.number,
            "n_completed_trials": len(completed),
        }
        assert self.config is not None
        output_path = Path(str(self.config.hydra.sweep.dir)) / "optimization_results.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(results), output_path)
        log.info(f"Best trial {best_trial.number}: value={best_trial.value}, params={best_trial.params}")
