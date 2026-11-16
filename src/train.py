from typing import Any, Dict, List, Optional, Tuple

import json
import os
from pathlib import Path
import signal
import threading
import time
import hydra
import lightning as L
import rootutils
import torch
from typing import Optional
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
    create_exp_name,
)

log = RankedLogger(__name__, rank_zero_only=True)
_ORCHESTRATOR_WATCHDOG_STARTED = False


def _parent_process_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fail_orchestrator_optuna_trials(reason: str) -> None:
    study_name = os.environ.get("ORCHESTRATOR_OPTUNA_STUDY_NAME")
    storage = os.environ.get("ORCHESTRATOR_OPTUNA_STORAGE")
    if not study_name or not storage:
        return
    try:
        import optuna
        from optuna.trial import TrialState

        from hydra_plugins.emulator_optuna4_sweeper.optuna4_sweeper import fail_trial

        storage = optuna.storages.get_storage(storage)  # opened once, reused for every trial
        study = optuna.load_study(study_name=study_name, storage=storage)
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING)):
            fail_trial(
                storage,
                study_name,
                trial.number,
                {
                    "orchestrator_parent_death_reason": reason,
                    "orchestrator_parent_death_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
    except Exception:
        pass


def _start_orchestrator_watchdog() -> None:
    global _ORCHESTRATOR_WATCHDOG_STARTED
    if _ORCHESTRATOR_WATCHDOG_STARTED:
        return
    parent_pid_text = os.environ.get("ORCHESTRATOR_PARENT_PID")
    if not parent_pid_text:
        return
    try:
        parent_pid = int(parent_pid_text)
    except ValueError:
        return
    if parent_pid <= 1:
        return
    _ORCHESTRATOR_WATCHDOG_STARTED = True

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if _parent_process_alive(parent_pid):
                continue
            reason = f"orchestrator parent pid {parent_pid} disappeared"
            _fail_orchestrator_optuna_trials(reason)
            try:
                os.killpg(os.getpgrp(), signal.SIGTERM)
            except Exception:
                pass
            os._exit(130)

    threading.Thread(target=watch, name="orchestrator-parent-watchdog", daemon=True).start()


def _metric_to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def save_metric_artifact(metric_dict: Dict[str, Any], output_dir: str | Path, **metadata: Any) -> None:
    payload = {key: _metric_to_python(value) for key, value in metric_dict.items()}
    payload.update(metadata)
    path = Path(output_dir) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _missing_embedding_metadata(datamodule: LightningDataModule) -> Dict[str, Any]:
    """Per-split missing-embedding counts for metrics.json, when the datamodule records them."""
    counts = datamodule.missing_embedding_counts() if hasattr(datamodule, "missing_embedding_counts") else None
    return {} if counts is None else {"missing_embeddings": counts}


def rebuild_wandb_logger(cfg: DictConfig, **overrides: Any) -> Logger:
    """Instantiate the configured W&B logger again, replacing only the given kwargs."""
    # The struct-mode logger node rejects new keys such as `name`, so open it for the merge. Merge into the config: `config` is also instantiate()'s first parameter name.
    with open_dict(cfg.logger.wandb):
        wandb_cfg = OmegaConf.merge(cfg.logger.wandb, overrides)
    return hydra.utils.instantiate(wandb_cfg, _convert_="all")


def best_checkpoint_score(trainer: Trainer, metric_name: Optional[str]) -> Optional[Any]:
    """Return the best score over epochs of ``metric_name``.

    Uses the checkpoint callback when it monitors the metric, else an early stopping callback
    that does (HPO runs without checkpoints). Returns None when neither does.
    """
    checkpoint = trainer.checkpoint_callback
    if checkpoint is not None and metric_name and checkpoint.monitor == metric_name:
        return checkpoint.best_model_score
    for callback in getattr(trainer, "early_stopping_callbacks", []):
        if metric_name and callback.monitor == metric_name:
            return callback.best_score
    return None


def setup_sweep_logger(cfg: DictConfig, logger_list: list) -> list:
    """Setup WandB logger with Optuna sweep information.

    Args:
        cfg: Hydra config
        logger_list: List of instantiated loggers

    Returns:
        Updated list of loggers
    """
    from hydra.core.hydra_config import HydraConfig

    # Get Hydra runtime config
    hydra_cfg = HydraConfig.get()
    explicit_group = cfg.get("wandb_group")
    explicit_name = cfg.get("wandb_name")
    explicit_tags = list(cfg.get("wandb_tags") or [])
    explicit_job_type = cfg.get("wandb_job_type")

    # Check if we're in multirun mode
    if hydra_cfg.mode.name != "MULTIRUN":
        log.info(
            f"Not in multirun mode (mode={hydra_cfg.mode.name}), skipping sweep logger setup"
        )
        job_num = None
        sweep_name = None
    else:
        log.info("Detected multirun mode, setting up sweep logger...")
        job_num = cfg.get("optuna_trial_number", hydra_cfg.job.num)
        sweep_name = "optuna_sweep"
        if cfg.get("hparams_search"):
            sweep_name = (
                cfg.hparams_search.get("hydra", {})
                .get("sweeper", {})
                .get("study_name", "optuna_sweep")
            )

    if not (explicit_group or explicit_name or explicit_tags or explicit_job_type or job_num is not None):
        return logger_list

    base_name = explicit_name or sweep_name or "wandb_run"
    run_name = f"{base_name}_trial_{job_num}" if job_num is not None else base_name

    log.info(f"Sweep name: {sweep_name}")
    log.info(f"Trial number: {job_num}")
    log.info(f"Run name: {run_name}")

    # Collect current hyperparameters
    sweep_params = {
        "optimizer.lr": cfg.get("optimizer", {}).get("lr"),
        "data.batch_size": cfg.get("data", {}).get("batch_size"),
        "model.num_hid_layers": cfg.get("model", {}).get("num_hid_layers"),
        "data._difficulty": cfg.get("data", {}).get("_difficulty"),
    }

    # Remove None values
    sweep_params = {k: v for k, v in sweep_params.items() if v is not None}

    log.info(f"Sweep params: {sweep_params}")

    # Update WandB logger
    updated_loggers = []
    for logger in logger_list:
        if isinstance(logger, WandbLogger):
            log.info("Found WandB logger, updating with sweep information...")
            # Rebuild from cfg.logger.wandb and override only the sweep fields (name, group, tags, job_type, config).
            new_logger = rebuild_wandb_logger(
                cfg,
                name=run_name,
                group=explicit_group or sweep_name,
                tags=explicit_tags or ["optuna_sweep", sweep_name],
                job_type=explicit_job_type or "hparam_search",
                config={
                    "sweep_name": sweep_name,
                    "trial_number": job_num,
                    **sweep_params,
                },
            )
            updated_loggers.append(new_logger)
            log.info(f"Updated WandB logger with group={sweep_name}, name={run_name}")
        else:
            updated_loggers.append(logger)

    return updated_loggers


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Set input_dim from datamodule before instantiating model
    if "input_dim" in cfg.model and (
        cfg.model.get("input_dim") is None or str(cfg.model.input_dim) == "???"
    ):
        input_dim = datamodule.get_input_dim()
        log.info(f"Setting model input_dim to {input_dim} from datamodule")
        cfg.model.input_dim = input_dim

    if "num_classes" in cfg.model and (
        cfg.model.get("num_classes") is None or str(cfg.model.num_classes) == "???"
    ):
        if not hasattr(datamodule, "get_num_classes"):
            raise AttributeError(
                f"Datamodule <{cfg.data._target_}> does not expose get_num_classes()"
            )
        num_classes = datamodule.get_num_classes()
        log.info(f"Setting model num_classes to {num_classes} from datamodule")
        cfg.model.num_classes = num_classes

    if "class_labels" in cfg.model and cfg.model.get("class_labels") is None:
        if hasattr(datamodule, "get_class_labels"):
            cfg.model.class_labels = datamodule.get_class_labels()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    if hasattr(model, "set_input_scaling"):
        # Standardize MLP inputs with stats fitted on the train split only.
        model.set_input_scaling(*datamodule.get_feature_scaling())

    if not isinstance(model, LightningModule):
        log.info("Detected non-Lightning baseline model; using estimator fit/evaluate path.")
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metric_dict = model.fit_evaluate(datamodule, output_dir)
        save_metric_artifact(
            metric_dict, output_dir, best_ckpt_path=None, **_missing_embedding_metadata(datamodule)
        )
        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
        }
        return metric_dict, object_dict

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    logger = setup_sweep_logger(cfg, logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    # create experiment name. Use it for runs and huggingface hub
    cfg_hash = create_exp_name(cfg)
    log.info(f"Experiment name: {cfg_hash}")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        # Only these loads of our own checkpoints (resume, test) use weights_only=False.
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    # Train/val values come from the last epoch but test values from the best checkpoint.
    # Record the source of each group so the file does not read as one model state.
    save_metric_artifact(
        metric_dict,
        cfg.paths.output_dir,
        best_ckpt_path=ckpt_path if cfg.get("test") else None,
        train_val_metrics_from="last_epoch",
        test_metrics_from=("best_checkpoint" if ckpt_path else "current_weights") if cfg.get("test") else None,
        **_missing_embedding_metadata(datamodule),
    )

    # callback_metrics holds the last epoch; report the best score over epochs instead when a
    # checkpoint or early stopping callback monitors the optimized metric.
    # Done after the artifact is saved, so metrics.json keeps the last-epoch values.
    best_score = best_checkpoint_score(trainer, cfg.get("optimized_metric"))
    if best_score is not None:
        metric_dict[cfg.optimized_metric] = best_score

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    _start_orchestrator_watchdog()
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
