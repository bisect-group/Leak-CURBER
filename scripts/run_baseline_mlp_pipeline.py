from __future__ import annotations

import argparse
import atexit
import copy
import csv
import fcntl
import hashlib
import json
import os
import pickle
import queue
import re
import select
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import optuna
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from optuna.trial import TrialState
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.embedders.store import pending_shard_rows  # noqa: E402
from src.data.components.embedders.store import index_files, load_index  # noqa: E402
from src.data.components.embedders.utils import key_hash_candidates  # noqa: E402

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover - rich is optional at runtime.
    Console = None
    Group = None
    Live = None
    Panel = None
    Table = None
    Text = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"
EMBEDDINGS_SCRIPT = REPO_ROOT / "scripts" / "run_embeddings.py"

ENDPOINTS = ["ec50", "ic50", "kcat", "kd", "ki", "km"]
SPLITS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "substrate_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "conformer_cosine_splits",
]
MODALITIES = ["smited", "ecfp", "molr", "esmc", "esm3"]
MODALITY_KEY_COLUMNS = {
    "smited": "smiles",
    "ecfp": "smiles",
    "molr": "smiles",
    "esmc": "sequence",
    "esm3": "pdb_ids",
}
MODALITY_INPUT_OVERRIDES = {
    "smited": "embeddings.smited_embeddings_input_pkl_path",
    "ecfp": "embeddings.smiles_ecfp_embeddings_input_pkl_path",
    "molr": "embeddings.molr_embeddings_input_pkl_path",
    "esmc": "embeddings.esmc_embeddings_input_pkl_path",
    "esm3": "embeddings.esm3_embeddings_input_pkl_path",
}
GPU_EMBEDDERS = {"smited", "esmc", "esm3"}
EMBEDDING_DATASET_CONFIG = "kinetic_params_ki"
HPO_PROGRESS_POLL_SEC = 10.0
JOB_DASHBOARD_REFRESH_SEC = 1.0
LOG_TAIL_LINES = 80
OPTUNA_TRIAL_NUMBER_TOKEN = "__OPTUNA_TRIAL_NUMBER__"
STREAM_DISPLAY_LINES = 14
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

_ACTIVE_ARGS: Optional[argparse.Namespace] = None
_CONSOLE = Console() if Console is not None else None
_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_SHUTDOWN_CLEANUP_DONE = False
_SHUTDOWN_CLEANUP_ARMED = False
_RUN_LOCK_HANDLE: Optional[Any] = None


# PipelineTask holds the task-specific hooks so one orchestrator serves all tasks;
# main(task) selects which hooks run.
@dataclass(frozen=True)
class PipelineTask:
    """Task-specific pieces of the orchestrator . Other tasks pass their own instance to main()."""

    modalities: list[str]
    modality_key_columns: dict[str, str]
    modality_input_overrides: dict[str, str]
    gpu_embedders: set[str]
    embedding_dataset_config: str
    pipeline_arg_defaults: Callable[[], dict[str, Any]]
    cache_roots: Callable[..., Any]
    resolve_split_dir: Callable[..., Any]
    split_threshold: Callable[..., Any]
    discover_split_specs: Callable[..., Any]
    discover_final_jobs: Callable[..., Any]
    study_name_for_endpoint: Callable[..., Any]
    build_base_overrides: Callable[..., Any]
    build_hpo_jobs: Callable[..., Any]
    materialization_specs_for_hpo: Callable[..., Any]
    materialization_specs_for_final: Callable[..., Any]
    materialization_specs_for_final_endpoint: Callable[..., Any]
    materialization_status_row: Callable[..., Any]
    required_columns_for_modalities: Callable[..., Any]
    preflight: Callable[..., Any]
    # A task whose HPO split depends on the endpoint (classification "auto") sets this.
    hpo_split_for_endpoint: Optional[Callable[..., str]] = None


_TASK: PipelineTask


@dataclass(frozen=True)
class PipelineJob:
    job_id: str
    stage: str
    endpoint: str
    split: Optional[str]
    threshold: Optional[str]
    seed: Optional[int]
    worker_index: int
    command: list[str]
    output_dir: Path
    modality: Optional[str] = None
    requires_gpu: bool = True
    stream_output: bool = False


def register_process(proc: subprocess.Popen) -> None:
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(proc)


def unregister_process(proc: subprocess.Popen) -> None:
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESSES.discard(proc)


def terminate_process_tree(proc: subprocess.Popen, *, grace_sec: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
    deadline = time.monotonic() + grace_sec
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        proc.kill()


def terminate_active_processes() -> None:
    with _ACTIVE_PROCESS_LOCK:
        procs = list(_ACTIVE_PROCESSES)
    for proc in procs:
        terminate_process_tree(proc)


ENDPOINT_DIRS = {"classification": "enzyme_classification", "retrieval": "enzyme_retrieval"}
DIR_ENDPOINTS = {folder: endpoint for endpoint, folder in ENDPOINT_DIRS.items()}


def run_root_for(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / "_runs" / str(args.pipeline_id)


def final_dir(
    output_root: Path, endpoint: str, split: str, threshold: Optional[str], seed: Any
) -> Path:
    return (
        Path(output_root)
        / ENDPOINT_DIRS.get(endpoint, endpoint)
        / split
        / (threshold or "all")
        / f"seed_{seed}"
    )


def final_job_metrics(
    output_root: Path, endpoints: Optional[Iterable[str]] = None
) -> list[tuple[str, str, Optional[str], str, Path]]:
    """Return (endpoint, split, threshold, seed, metrics_path) for finished finals under output_root."""
    wanted = None if endpoints is None else set(endpoints)
    found = []
    for path in sorted(Path(output_root).glob("*/*/*/seed_*/metrics.json")):
        folder, split, threshold, seed_dir = path.relative_to(output_root).parts[:4]
        endpoint = DIR_ENDPOINTS.get(folder, folder)
        if folder.startswith("_") or (wanted is not None and endpoint not in wanted):
            continue
        found.append(
            (
                endpoint,
                split,
                None if threshold == "all" else threshold,
                seed_dir.removeprefix("seed_"),
                path,
            )
        )
    return found


def final_endpoint_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        Path(args.output_root) / ENDPOINT_DIRS.get(endpoint, endpoint)
        for endpoint in args.endpoints
    ]


def acquire_run_lock(run_root: Path) -> None:
    global _RUN_LOCK_HANDLE
    lock_path = run_root / ".orchestrator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"Another orchestrator appears to be running for {run_root}. "
            f"Stop it before starting a second run."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\nstarted_at={datetime.now().isoformat()}\n")
    handle.flush()
    _RUN_LOCK_HANDLE = handle


def scoped_child_process_groups(args: argparse.Namespace) -> set[int]:
    pipeline_id_value = getattr(args, "pipeline_id", None)
    output_root = getattr(args, "output_root", None)
    endpoints = list(getattr(args, "endpoints", []) or [])
    patterns: list[str] = []
    if pipeline_id_value and output_root is not None:
        patterns.append(str(run_root_for(args)))
        patterns.append(str(pipeline_id_value))
    for endpoint in endpoints:
        try:
            patterns.append(_TASK.study_name_for_endpoint(args, endpoint))
        except Exception:
            continue
    patterns = [pattern for pattern in dict.fromkeys(patterns) if pattern]
    if not patterns:
        return set()

    current_pid = os.getpid()
    current_pgid = os.getpgrp()
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid,pgid,args"], text=True)
    except Exception:
        return set()

    pgids: set[int] = set()
    for row in rows.splitlines()[1:]:
        parts = row.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            pgid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        if pid == current_pid or pgid == current_pgid:
            continue
        if "src/train.py" not in command:
            continue
        if any(pattern in command for pattern in patterns):
            pgids.add(pgid)
    return pgids


def terminate_scoped_child_processes(
    reason: str = "orchestrator cleanup", *, grace_sec: float = 5.0
) -> int:
    args = _ACTIVE_ARGS
    if args is None:
        return 0
    pgids = scoped_child_process_groups(args)
    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except OSError:
            continue
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        remaining = scoped_child_process_groups(args) & pgids
        if not remaining:
            break
        time.sleep(0.1)
    remaining = scoped_child_process_groups(args) & pgids
    for pgid in sorted(remaining):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except OSError:
            continue
    if pgids:
        log_step(f"{reason}: terminated {len(pgids)} scoped child process group(s)")
    return len(pgids)


def fail_active_optuna_trials(reason: str = "orchestrator interrupted") -> int:
    args = _ACTIVE_ARGS
    if args is None or getattr(args, "dry_run", False) or getattr(args, "skip_hpo", False):
        return 0
    pipeline_id_value = getattr(args, "pipeline_id", None)
    output_root = getattr(args, "output_root", None)
    endpoints = list(getattr(args, "endpoints", []) or [])
    if not pipeline_id_value or output_root is None or not endpoints:
        return 0

    changed = 0
    for endpoint in endpoints:
        try:
            study_name = _TASK.study_name_for_endpoint(args, endpoint)
            storage = optuna_storage_url(run_root_for(args), study_name)
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            continue
        storage_backend = study._storage
        for trial in study.get_trials(
            deepcopy=False, states=(TrialState.RUNNING, TrialState.WAITING)
        ):
            try:
                storage_backend.set_trial_user_attr(
                    trial._trial_id, "orchestrator_stop_reason", reason
                )
                storage_backend.set_trial_user_attr(
                    trial._trial_id, "orchestrator_stop_at", datetime.now().isoformat()
                )
                storage_backend.set_trial_state_values(
                    trial._trial_id, state=TrialState.FAIL, values=None
                )
                changed += 1
            except Exception:
                log_step(
                    f"Could not mark Optuna trial {trial.number} in study {study_name} as failed"
                )
    if changed:
        log_step(f"Marked {changed} active Optuna RUNNING/WAITING trial(s) as FAIL")
    return changed


def cleanup_interrupted_run(reason: str) -> None:
    global _SHUTDOWN_CLEANUP_DONE
    if not _SHUTDOWN_CLEANUP_ARMED:
        return
    if _SHUTDOWN_CLEANUP_DONE:
        return
    _SHUTDOWN_CLEANUP_DONE = True
    terminate_active_processes()
    terminate_scoped_child_processes(reason)
    fail_active_optuna_trials(reason)


def cleanup_orphaned_run_workers(reason: str) -> None:
    terminated = terminate_scoped_child_processes(reason)
    failed = fail_active_optuna_trials(reason)
    if terminated or failed:
        log_step(
            f"{reason}: cleaned {terminated} process group(s), failed {failed} active Optuna trial(s)"
        )


def handle_termination_signal(signum: int, _frame: Any) -> None:
    log_step(f"Received signal {signum}; terminating active child processes")
    cleanup_interrupted_run(f"orchestrator received signal {signum}")
    raise KeyboardInterrupt


def pipeline_arg_defaults() -> dict[str, Any]:
    return {
        "output_root": REPO_ROOT / "outputs/baselines/MLP",
        "pipeline_id": None,
        "data_root": REPO_ROOT / "outputs/baselines/MLP/data/regression",
        "embeddings_root": REPO_ROOT / "data/intermediate/embeddings",
        "endpoints": list(ENDPOINTS),
        "splits": list(SPLITS),
        "seeds": [666, 777, 888],
        "gpus": None,
        "max_slots_per_gpu": 1,
        "hpo_workers_per_endpoint": 1,
        "hpo_sampler_seed_base": 42,
        "hpo_sampler_seed_strategy": "worker_offset",
        "n_trials": 50,
        "max_epochs": 200,
        "final_max_epochs": None,
        "final_params_source": "hpo_csv",
        "final_best_params_path": None,
        # Reusing old final successes is opt-in, so stale results stay out by default.
        "final_reuse_existing_successes": False,
        "hpo_checkpoints": False,
        "final_checkpoints": True,
        "retry_failed": 0,
        "job_timeout_sec": 0,
        "skip_embeddings": False,
        "skip_materialization": False,
        "skip_hpo": False,
        "skip_final": False,
        "refresh_materialized_features": False,
        "materialized_feature_dtype": "float16",
        "materialized_cache_root": None,
        "materialized_block_rows": 65536,
        "materialized_validation": "metadata",
        "refresh_embedding_superset": False,
        "stream_all_jobs": False,
        "modalities": list(MODALITIES),
        "wandb_project": "EMULaToR",
        "wandb_entity": "bisect-group",
        "wandb_offline": False,
        "skip_wandb_check": False,
        "dry_run": False,
        "preflight_only": False,
        "skip_preflight": False,
        "fail_fast": False,
        "progress_backend": "rich",
        "hpo_trial_log": False,
        "no_progress": False,
        "hpo_split": "enzyme_structure_splits",
        "run_label": "mlp",
        "study_prefix": "baseline_mlp",
        "run_tags": ["baseline_regression", "mlp"],
        "hydra_profile": {},
        "storage_profile": default_storage_profile(),
        "cleanup_disabled": False,
        "cleanup_dry_run": False,
        "cleanup_existing_run": False,
        "run_config": None,
        "print_resolved_run_config": False,
    }


def default_storage_profile() -> dict[str, Any]:
    return {
        "shared_materialized_cache": {
            "enabled": True,
            "max_total_gb": 500,
            "max_entry_gb": 75,
            "prune_policy": "lru",
            "min_free_gb": 500,
            "fail_if_exceeded": False,
        },
        "job_artifacts": {
            "delete_job_materialized_features": True,
            "delete_hpo_predictions": True,
            "delete_hpo_models": False,
        },
        "checkpoints": {
            "prune_hpo_checkpoints": True,
            "keep_hpo_best_checkpoint": False,
            "keep_hpo_last_checkpoint": False,
            "prune_final_checkpoints": False,
            "keep_final_best_checkpoint": True,
            "keep_final_last_checkpoint": False,
            "max_checkpoint_total_gb": 150,
        },
        "logs": {
            "max_run_root_gb": 2000,
            "fail_if_exceeded": False,
            "write_disk_report": True,
        },
    }


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _profile_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _profile_to_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    run = profile.get("run") or {}
    resources = profile.get("resources") or {}
    stages = profile.get("stages") or {}
    wandb = profile.get("wandb") or {}
    ui = profile.get("ui") or {}
    hydra = profile.get("hydra") or {}
    storage = profile.get("storage") or {}

    for source, mapping in (
        (
            run,
            {
                "pipeline_id": "pipeline_id",
                "output_root": "output_root",
                "data_root": "data_root",
                "embeddings_root": "embeddings_root",
                "endpoints": "endpoints",
                "splits": "splits",
                "seeds": "seeds",
                "modalities": "modalities",
                "label": "run_label",
                "study_prefix": "study_prefix",
                "tags": "run_tags",
            },
        ),
        (
            resources,
            {
                "gpus": "gpus",
                "max_slots_per_gpu": "max_slots_per_gpu",
                "hpo_workers_per_endpoint": "hpo_workers_per_endpoint",
                "retry_failed": "retry_failed",
                "job_timeout_sec": "job_timeout_sec",
            },
        ),
        (
            wandb,
            {
                "project": "wandb_project",
                "entity": "wandb_entity",
                "offline": "wandb_offline",
                "skip_check": "skip_wandb_check",
            },
        ),
        (
            ui,
            {
                "progress_backend": "progress_backend",
                "hpo_trial_log": "hpo_trial_log",
                "stream_all_jobs": "stream_all_jobs",
                "no_progress": "no_progress",
            },
        ),
    ):
        for source_key, dest_key in mapping.items():
            if source_key in source:
                defaults[dest_key] = source[source_key]

    embeddings = stages.get("embeddings") or {}
    if "enabled" in embeddings:
        defaults["skip_embeddings"] = not bool(embeddings["enabled"])
    if "refresh_superset" in embeddings:
        defaults["refresh_embedding_superset"] = bool(embeddings["refresh_superset"])

    materialization = stages.get("materialization") or {}
    if "enabled" in materialization:
        defaults["skip_materialization"] = not bool(materialization["enabled"])
    if "refresh" in materialization:
        defaults["refresh_materialized_features"] = bool(materialization["refresh"])
    if "feature_dtype" in materialization:
        defaults["materialized_feature_dtype"] = materialization["feature_dtype"]
    if "cache_root" in materialization:
        defaults["materialized_cache_root"] = materialization["cache_root"]
    if "block_rows" in materialization:
        defaults["materialized_block_rows"] = materialization["block_rows"]
    if "validation" in materialization:
        defaults["materialized_validation"] = materialization["validation"]

    hpo = stages.get("hpo") or {}
    if "enabled" in hpo:
        defaults["skip_hpo"] = not bool(hpo["enabled"])
    if "split" in hpo:
        defaults["hpo_split"] = hpo["split"]
    if "n_trials" in hpo:
        defaults["n_trials"] = hpo["n_trials"]
    if "max_epochs" in hpo:
        defaults["max_epochs"] = hpo["max_epochs"]
    if "sampler_seed_base" in hpo:
        defaults["hpo_sampler_seed_base"] = hpo["sampler_seed_base"]
    if "sampler_seed_strategy" in hpo:
        defaults["hpo_sampler_seed_strategy"] = hpo["sampler_seed_strategy"]
    if "checkpoints" in hpo:
        defaults["hpo_checkpoints"] = bool(hpo["checkpoints"])

    final = stages.get("final") or {}
    if "enabled" in final:
        defaults["skip_final"] = not bool(final["enabled"])
    if "max_epochs" in final:
        defaults["final_max_epochs"] = final["max_epochs"]
    if "params_source" in final:
        defaults["final_params_source"] = final["params_source"]
    if "best_params_path" in final:
        defaults["final_best_params_path"] = final["best_params_path"]
    if "reuse_existing_successes" in final:
        defaults["final_reuse_existing_successes"] = bool(final["reuse_existing_successes"])
    if "checkpoints" in final:
        defaults["final_checkpoints"] = bool(final["checkpoints"])

    cleanup = stages.get("cleanup") or {}
    if "enabled" in cleanup:
        defaults["cleanup_disabled"] = not bool(cleanup["enabled"])
    if "dry_run" in cleanup:
        defaults["cleanup_dry_run"] = bool(cleanup["dry_run"])

    common_data = (hydra.get("common") or {}).get("data") or {}
    if "modalities" in common_data and "modalities" not in defaults:
        defaults["modalities"] = common_data["modalities"]

    defaults["hydra_profile"] = hydra
    if storage:
        defaults["storage_profile"] = deep_update(default_storage_profile(), storage)
    return defaults


def load_run_config_defaults(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    cfg = OmegaConf.load(path)
    profile = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(profile, dict):
        raise ValueError(f"Run config must be a mapping: {path}")
    defaults = _profile_to_defaults(profile)
    defaults["run_config"] = path
    return defaults


def build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline regression HPO and final training pipeline."
    )
    parser.add_argument(
        "--run_config",
        type=Path,
        default=defaults["run_config"],
        help="YAML run profile for orchestrator and Hydra options.",
    )
    parser.add_argument(
        "--print_resolved_run_config",
        action=argparse.BooleanOptionalAction,
        default=defaults["print_resolved_run_config"],
        help="Print the merged run profile and exit.",
    )
    parser.add_argument("--output_root", type=Path, default=defaults["output_root"])
    parser.add_argument("--pipeline_id", type=str, default=defaults["pipeline_id"])
    parser.add_argument("--data_root", type=Path, default=defaults["data_root"])
    parser.add_argument("--embeddings_root", type=Path, default=defaults["embeddings_root"])
    parser.add_argument("--endpoints", nargs="+", default=defaults["endpoints"])
    parser.add_argument("--splits", nargs="+", default=defaults["splits"])
    parser.add_argument("--seeds", nargs="+", type=int, default=defaults["seeds"])
    parser.add_argument("--gpus", nargs="+", default=defaults["gpus"])
    parser.add_argument("--max_slots_per_gpu", type=int, default=defaults["max_slots_per_gpu"])
    parser.add_argument(
        "--hpo_workers_per_endpoint", type=int, default=defaults["hpo_workers_per_endpoint"]
    )
    parser.add_argument(
        "--hpo_sampler_seed_base", type=int, default=defaults["hpo_sampler_seed_base"]
    )
    parser.add_argument(
        "--hpo_sampler_seed_strategy",
        choices=["worker_offset", "fixed", "none"],
        default=defaults["hpo_sampler_seed_strategy"],
    )
    parser.add_argument("--n_trials", type=int, default=defaults["n_trials"])
    parser.add_argument("--max_epochs", type=int, default=defaults["max_epochs"])
    parser.add_argument("--final_max_epochs", type=int, default=defaults["final_max_epochs"])
    parser.add_argument(
        "--final_params_source",
        choices=["fixed", "hpo_csv"],
        default=defaults["final_params_source"],
        help="When HPO is skipped, choose whether final jobs use fixed Hydra config or endpoint params from a best-params CSV.",
    )
    parser.add_argument(
        "--final_best_params_path", type=Path, default=defaults["final_best_params_path"]
    )
    parser.add_argument(
        "--hpo_checkpoints",
        "--hpo-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=defaults["hpo_checkpoints"],
    )
    parser.add_argument(
        "--final_checkpoints",
        "--final-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=defaults["final_checkpoints"],
    )
    parser.add_argument(
        "--final_reuse_existing_successes",
        "--final-reuse-existing-successes",
        action=argparse.BooleanOptionalAction,
        default=defaults["final_reuse_existing_successes"],
        help="For final jobs, skip job IDs already marked success in final_status.csv.",
    )
    parser.add_argument("--retry_failed", type=int, default=defaults["retry_failed"])
    parser.add_argument("--job_timeout_sec", type=int, default=defaults["job_timeout_sec"])
    parser.add_argument(
        "--skip_embeddings",
        "--skip-embeddings",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_embeddings"],
    )
    parser.add_argument(
        "--skip_materialization",
        "--skip-materialization",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_materialization"],
    )
    parser.add_argument(
        "--skip_hpo",
        "--skip-hpo",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_hpo"],
    )
    parser.add_argument(
        "--skip_final",
        "--skip-final",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_final"],
    )
    parser.add_argument(
        "--refresh_materialized_features",
        "--refresh-materialized-features",
        action=argparse.BooleanOptionalAction,
        default=defaults["refresh_materialized_features"],
    )
    parser.add_argument(
        "--materialized_feature_dtype",
        choices=["float16", "float32"],
        default=defaults["materialized_feature_dtype"],
    )
    parser.add_argument(
        "--materialized_cache_root", type=Path, default=defaults["materialized_cache_root"]
    )
    parser.add_argument(
        "--materialized_block_rows", type=int, default=defaults["materialized_block_rows"]
    )
    parser.add_argument(
        "--materialized_validation",
        choices=["metadata", "full", "trust_complete"],
        default=defaults["materialized_validation"],
        help="Validation policy for materialized feature cache hits.",
    )
    parser.add_argument(
        "--refresh_embedding_superset",
        "--refresh-embedding-superset",
        action=argparse.BooleanOptionalAction,
        default=defaults["refresh_embedding_superset"],
        help="Recompute the deduplicated embedding input superset even when a matching cache exists.",
    )
    parser.add_argument(
        "--stream_all_jobs",
        "--stream-all-jobs",
        action=argparse.BooleanOptionalAction,
        default=defaults["stream_all_jobs"],
    )
    parser.add_argument(
        "--modalities", nargs="+", default=defaults["modalities"], choices=_TASK.modalities
    )
    parser.add_argument("--wandb_project", type=str, default=defaults["wandb_project"])
    parser.add_argument("--wandb_entity", type=str, default=defaults["wandb_entity"])
    parser.add_argument(
        "--wandb_offline",
        "--wandb-offline",
        action=argparse.BooleanOptionalAction,
        default=defaults["wandb_offline"],
    )
    parser.add_argument(
        "--skip_wandb_check",
        "--skip-wandb-check",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_wandb_check"],
    )
    parser.add_argument(
        "--dry_run",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=defaults["dry_run"],
    )
    parser.add_argument(
        "--preflight_only",
        "--preflight-only",
        action=argparse.BooleanOptionalAction,
        default=defaults["preflight_only"],
    )
    parser.add_argument(
        "--skip_preflight",
        "--skip-preflight",
        action=argparse.BooleanOptionalAction,
        default=defaults["skip_preflight"],
    )
    parser.add_argument(
        "--fail_fast",
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=defaults["fail_fast"],
    )
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=not defaults["cleanup_disabled"],
        help="Enable configured cleanup of job-local materialized features and HPO artifacts.",
    )
    parser.add_argument(
        "--cleanup_dry_run",
        "--cleanup-dry-run",
        action=argparse.BooleanOptionalAction,
        default=defaults["cleanup_dry_run"],
        help="Report cleanup candidates without deleting them.",
    )
    parser.add_argument(
        "--cleanup_existing_run",
        "--cleanup-existing-run",
        action=argparse.BooleanOptionalAction,
        default=defaults["cleanup_existing_run"],
        help="Run cleanup/reporting on the configured run root and exit.",
    )
    parser.add_argument(
        "--progress_backend",
        choices=["rich", "plain"],
        default=defaults["progress_backend"],
        help="Terminal progress renderer. Rich gives live HPO/final dashboards when available.",
    )
    parser.add_argument(
        "--hpo_trial_log",
        "--hpo-trial-log",
        action=argparse.BooleanOptionalAction,
        default=defaults["hpo_trial_log"],
        help="Print concise HPO trial completion events in addition to the live endpoint summary.",
    )
    parser.add_argument(
        "--no_progress",
        "--no-progress",
        action=argparse.BooleanOptionalAction,
        default=defaults["no_progress"],
        help="Disable terminal progress bars.",
    )
    parser.add_argument("--hpo_split", type=str, default=defaults["hpo_split"])
    parser.set_defaults(
        hydra_profile=defaults["hydra_profile"],
        run_label=defaults["run_label"],
        study_prefix=defaults["study_prefix"],
        run_tags=defaults["run_tags"],
        storage_profile=defaults["storage_profile"],
    )
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run_config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _TASK.pipeline_arg_defaults()
    profile_defaults = load_run_config_defaults(_profile_path(pre_args.run_config))
    defaults.update(profile_defaults)
    parser = build_arg_parser(defaults)
    args = parser.parse_args()
    if args.run_config is not None:
        args.run_config = _profile_path(str(args.run_config))
    for path_field in (
        "output_root",
        "data_root",
        "embeddings_root",
        "materialized_cache_root",
        "final_best_params_path",
    ):
        value = getattr(args, path_field, None)
        if value is not None and not isinstance(value, Path):
            value = Path(value)
        if isinstance(value, Path) and not value.is_absolute():
            value = REPO_ROOT / value
        setattr(args, path_field, value)
    if args.gpus is not None:
        args.gpus = [str(gpu) for gpu in args.gpus]
    args.cleanup_disabled = not bool(getattr(args, "cleanup", True))
    args.endpoints = [str(endpoint) for endpoint in args.endpoints]
    args.splits = [str(split) for split in args.splits]
    args.modalities = [str(modality) for modality in args.modalities]
    return args


def log_step(message: str) -> None:
    text = f"[pipeline] {message}"
    if rich_enabled(_ACTIVE_ARGS):
        assert _CONSOLE is not None and Text is not None
        _CONSOLE.print(Text(text, style="cyan"))
        return
    print(text, flush=True)


def progress_enabled(args: argparse.Namespace) -> bool:
    return (
        not getattr(args, "no_progress", False)
        and getattr(args, "progress_backend", "rich") != "plain"
    )


def plain_progress(args: Optional[argparse.Namespace]) -> bool:
    return bool(args is not None and getattr(args, "progress_backend", "rich") == "plain")


def rich_enabled(args: Optional[argparse.Namespace]) -> bool:
    return bool(
        args is not None
        and progress_enabled(args)
        and getattr(args, "progress_backend", "rich") == "rich"
        and _CONSOLE is not None
        and Table is not None
        and Live is not None
    )


def progress_iter(
    args: argparse.Namespace,
    iterable: Iterable[Any],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
) -> Iterable[Any]:
    if not progress_enabled(args):
        return iterable
    if rich_enabled(args):
        assert _CONSOLE is not None

        def rich_generator() -> Iterable[Any]:
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=_CONSOLE,
                transient=True,
            ) as progress:
                task_id = progress.add_task(desc, total=total)
                for item in iterable:
                    yield item
                    progress.update(task_id, advance=1)

        return rich_generator()
    return iterable


def render_summary_table(
    title: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]
) -> None:
    if not rows:
        return
    if rich_enabled(_ACTIVE_ARGS):
        assert _CONSOLE is not None and Table is not None
        table = Table(title=title)
        for header, _key in columns:
            table.add_column(header)
        for row in rows:
            table.add_row(*(str(row.get(key, "")) for _header, key in columns))
        _CONSOLE.print(table)
        return
    log_step(title)
    for row in rows:
        values = ", ".join(f"{header}={row.get(key, '')}" for header, key in columns)
        log_step(values)


class StreamDisplayState:
    """Keeps a small sanitized terminal view of a streaming subprocess."""

    def __init__(self, max_lines: int = STREAM_DISPLAY_LINES) -> None:
        self.max_lines = max_lines
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.current = ""
        self.carriage_line = ""

    def feed(self, text: str) -> None:
        for char in text:
            if char == "\r":
                self.carriage_line = sanitize_stream_line(self.current)
                self.current = ""
                continue
            if char == "\n":
                line = sanitize_stream_line(self.current)
                if self.carriage_line:
                    self.lines.append(self.carriage_line)
                    self.carriage_line = ""
                if line:
                    self.lines.append(line)
                self.current = ""
                continue
            self.current += char

    def render_lines(self) -> list[str]:
        lines = list(self.lines)
        if self.carriage_line:
            lines.append(self.carriage_line)
        current = sanitize_stream_line(self.current)
        if current:
            lines.append(current)
        return lines[-self.max_lines :]


def sanitize_stream_line(line: str) -> str:
    line = ANSI_RE.sub("", line)
    line = line.replace("\t", " ")
    line = "".join(char for char in line if char.isprintable() or char == " ")
    return line.strip()


def append_stream_display(
    stream_states: Optional[dict[str, StreamDisplayState]],
    job_id: str,
    text: str,
    lock: Optional[threading.Lock] = None,
) -> None:
    if stream_states is None:
        return
    if lock is None:
        state = stream_states.setdefault(job_id, StreamDisplayState())
        state.feed(text)
        return
    with lock:
        state = stream_states.setdefault(job_id, StreamDisplayState())
        state.feed(text)


def pipeline_id() -> str:
    return datetime.now().strftime("baseline_mlp_%Y%m%d_%H%M%S")


def cache_roots(embeddings_root: Path) -> dict[str, Path]:
    return {
        "smited": embeddings_root / "smited/materials_smi_ted_fork/v1",
        "ecfp": embeddings_root / "ecfp/radius2_nbits1024/v1",
        "molr": embeddings_root / "molr/2d/v1",
        "esmc": embeddings_root / "esmc/ESMC-6B/hf_v1",
        "esm3": embeddings_root / "esm3/open_small_structure/v1",
    }


def enabled_modalities(args: argparse.Namespace) -> list[str]:
    return list(args.modalities or _TASK.modalities)


def required_columns_for_modalities(modalities: Iterable[str]) -> set[str]:
    columns = {"log10_value"}
    columns.update(_TASK.modality_key_columns[modality] for modality in modalities)
    return columns


def resolve_split_dir(
    data_root: Path, endpoint: str, split: str, threshold: Optional[str] = None
) -> Path:
    split_root = data_root / f"baselines_{endpoint}" / split
    if has_tvt(split_root):
        return split_root
    if threshold is not None:
        threshold_name = (
            threshold if threshold.startswith("threshold_") else f"threshold_{threshold}"
        )
        threshold_dir = split_root / threshold_name
        if has_tvt(threshold_dir):
            return threshold_dir
        raise FileNotFoundError(f"Missing train/val/test under {threshold_dir}")
    candidates = [
        path for path in sorted(split_root.glob("threshold_*")) if path.is_dir() and has_tvt(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple threshold folders under {split_root}; pass an explicit threshold."
        )
    raise FileNotFoundError(f"No train/val/test parquets found under {split_root}")


def has_tvt(path: Path) -> bool:
    return all((path / f"{split}.parquet").exists() for split in ("train", "val", "test"))


def split_threshold(split_dir: Path) -> Optional[str]:
    return split_dir.name if split_dir.name.startswith("threshold_") else None


def discover_split_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for endpoint in args.endpoints:
        for split in args.splits:
            split_dir = _TASK.resolve_split_dir(args.data_root, endpoint, split)
            specs.append(
                {
                    "endpoint": endpoint,
                    "split": split,
                    "threshold": _TASK.split_threshold(split_dir),
                    "split_dir": split_dir,
                }
            )
    return specs


def embedding_superset_fingerprint(
    args: argparse.Namespace,
    split_specs: list[dict[str, Any]],
    key_columns: list[str],
) -> dict[str, Any]:
    files = []
    for spec in split_specs:
        split_dir = Path(spec["split_dir"])
        for tvt in ("train", "val", "test"):
            path = split_dir / f"{tvt}.parquet"
            stat = path.stat()
            files.append(
                {
                    "endpoint": spec["endpoint"],
                    "split": spec["split"],
                    "threshold": spec["threshold"],
                    "tvt": tvt,
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    # Key on each store's meta.json hash (records model/version) and warn on pending rows,
    # which readers cannot see.
    stores = {}
    for modality, root in _TASK.cache_roots(args.embeddings_root).items():
        if modality not in enabled_modalities(args):
            continue
        meta_path = root / "meta.json"
        stores[modality] = (
            hashlib.sha256(meta_path.read_bytes()).hexdigest() if meta_path.exists() else None
        )
        if pending := pending_shard_rows(root):
            log_step(
                f"WARNING: store {modality} ({root}) has {pending} pending rows invisible to readers."
            )
    return {
        "version": 2,
        "stores": stores,
        "data_root": str(args.data_root.resolve()),
        "endpoints": list(args.endpoints),
        "splits": list(args.splits),
        "modalities": enabled_modalities(args),
        "key_columns": key_columns,
        "files": files,
    }


def embedding_superset_cache_key(fingerprint: dict[str, Any]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def embedding_superset_cache_dir(args: argparse.Namespace, cache_key: str) -> Path:
    return args.output_root / "_embedding_superset_cache" / cache_key


def embedding_input_paths(input_dir: Path) -> dict[str, Path]:
    return {
        "smiles": input_dir / "smiles.pkl",
        "sequence": input_dir / "sequences.pkl",
        "pdb_ids": input_dir / "pdb_ids.pkl",
    }


def load_embedding_superset(
    input_paths: dict[str, Path], key_columns: list[str]
) -> dict[str, list[str]]:
    keys_by_column = {}
    for column in key_columns:
        with open(input_paths[column], "rb") as handle:
            keys_by_column[column] = pickle.load(handle)
    return keys_by_column


def write_embedding_input_manifest(
    run_root: Path,
    input_paths: dict[str, Path],
    keys_by_column: dict[str, list[str]],
    cache_key: str,
    cache_dir: Path,
    cache_status: str,
) -> None:
    manifest_rows = [
        {
            "key_column": column,
            "unique_keys": len(keys),
            "input_path": str(input_paths[column]),
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "cache_status": cache_status,
        }
        for column, keys in sorted(keys_by_column.items())
    ]
    write_dict_rows(run_root / "embedding_input_manifest.csv", manifest_rows)


def build_embedding_superset(
    args: argparse.Namespace, run_root: Path
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    split_specs = _TASK.discover_split_specs(args)
    key_columns = sorted(
        {_TASK.modality_key_columns[modality] for modality in enabled_modalities(args)}
    )
    fingerprint = embedding_superset_fingerprint(args, split_specs, key_columns)
    cache_key = embedding_superset_cache_key(fingerprint)
    input_dir = embedding_superset_cache_dir(args, cache_key)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_paths = embedding_input_paths(input_dir)
    metadata_path = input_dir / "metadata.json"

    cache_complete = metadata_path.exists() and all(
        input_paths[column].exists() for column in key_columns
    )
    if cache_complete and not args.refresh_embedding_superset:
        keys_by_column = load_embedding_superset(input_paths, key_columns)
        write_embedding_input_manifest(
            run_root, input_paths, keys_by_column, cache_key, input_dir, "hit"
        )
        counts = ", ".join(
            f"{column}={len(keys):,}" for column, keys in sorted(keys_by_column.items())
        )
        log_step(f"Embedding superset cache hit: {cache_key} ({counts})")
        return input_paths, keys_by_column

    if args.refresh_embedding_superset and cache_complete:
        log_step(f"Refreshing embedding superset cache: {cache_key}")
    else:
        log_step(f"Embedding superset cache miss: {cache_key}")

    values_by_column: dict[str, set[str]] = {column: set() for column in key_columns}

    scan_tasks = [(spec, tvt) for spec in split_specs for tvt in ("train", "val", "test")]
    log_step(
        f"Scanning {len(scan_tasks)} parquet file(s) for embedding keys "
        f"across columns: {', '.join(key_columns)}"
    )
    for spec, tvt in progress_iter(
        args, scan_tasks, desc="Embedding superset", total=len(scan_tasks), unit="file"
    ):
        split_dir = Path(spec["split_dir"])
        path = split_dir / f"{tvt}.parquet"
        table = pq.read_table(path, columns=key_columns)
        frame = table.to_pandas()
        for column in key_columns:
            add_key_values(values_by_column[column], frame[column])

    keys_by_column = {column: sorted(values) for column, values in values_by_column.items()}
    for column, keys in keys_by_column.items():
        with open(input_paths[column], "wb") as handle:
            pickle.dump(keys, handle)

    metadata_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "fingerprint": fingerprint,
                "counts": {column: len(keys) for column, keys in sorted(keys_by_column.items())},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_embedding_input_manifest(
        run_root,
        input_paths,
        keys_by_column,
        cache_key,
        input_dir,
        "refresh" if args.refresh_embedding_superset else "miss",
    )
    counts = ", ".join(
        f"{column}={len(keys):,}" for column, keys in sorted(keys_by_column.items())
    )
    log_step(f"Embedding superset cached at {input_dir}: {counts}")
    return input_paths, keys_by_column


def add_key_values(output: set[str], values: Iterable[Any]) -> None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, (list, tuple, set, np.ndarray)):
            for item in value:
                if item is not None and str(item):
                    output.add(str(item))
            continue
        if str(value):
            output.add(str(value))


def existing_embedding_count(
    cache_root: Path,
    keys: list[str],
    args: Optional[argparse.Namespace] = None,
    desc: Optional[str] = None,
) -> int:
    if not index_files(cache_root) or not keys:
        return 0
    index_df = load_index(cache_root, columns=["key_hash"])
    existing_hashes = set(index_df["key_hash"].astype(str))
    key_type = json.loads((cache_root / "meta.json").read_text()).get("key_type", "")
    iterator = keys
    if args is not None and progress_enabled(args) and len(keys) >= 100_000:
        iterator = progress_iter(
            args,
            keys,
            desc=desc or "Checking cache",
            total=len(keys),
            unit="key",
        )
    # ESM3 keys carry the PDB source, so a key counts if any source variant exists.
    return sum(
        1
        for key in iterator
        if any(h in existing_hashes for h in key_hash_candidates(str(key), key_type))
    )


def hydra_list(values: Iterable[Any]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def modality_command(
    modality: str, input_path: Path, args: Optional[argparse.Namespace] = None
) -> list[str]:
    overrides = [f"{_TASK.modality_input_overrides[modality]}={input_path}"]
    if modality == "esmc":
        overrides.extend(
            [
                "embeddings.esmc_model_id=biohub/ESMC-6B",
                "embeddings.esmc_model_name=ESMC-6B",
                "embeddings.esmc_version=hf_v1",
            ]
        )
    if modality == "molr" and args is not None:
        visible_gpu_count = len(args.gpus or [])
        if visible_gpu_count:
            overrides.extend(
                [
                    "embeddings.molr_use_gpu=true",
                    f"embeddings.molr_gpu_ids={hydra_list(range(visible_gpu_count))}",
                ]
            )
    base_command = [
        sys.executable,
        str(EMBEDDINGS_SCRIPT),
        "--datasets",
        _TASK.embedding_dataset_config,
        "--embedders",
        modality,
        "--overrides",
        *overrides,
    ]
    return base_command


def build_embedding_jobs(
    args: argparse.Namespace,
    run_root: Path,
    input_paths: dict[str, Path],
    keys_by_column: dict[str, list[str]],
) -> list[PipelineJob]:
    jobs = []
    plan_rows = []
    roots = _TASK.cache_roots(args.embeddings_root)
    modalities = enabled_modalities(args)
    for modality in progress_iter(
        args, modalities, desc="Embedding plan", total=len(modalities), unit="modality"
    ):
        key_column = _TASK.modality_key_columns[modality]
        keys = keys_by_column.get(key_column, [])
        cache_root = roots[modality]
        existing_count = existing_embedding_count(
            cache_root, keys, args=args, desc=f"{modality} cache"
        )
        missing_count = max(0, len(keys) - existing_count)
        command = modality_command(modality, input_paths[key_column], args)
        status = "ready" if missing_count else "complete"
        device_class = "gpu" if modality in _TASK.gpu_embedders else "cpu"
        plan_rows.append(
            {
                "modality": modality,
                "key_column": key_column,
                "total_unique_keys": len(keys),
                "existing_keys": existing_count,
                "missing_keys": missing_count,
                "cache_root": str(cache_root),
                "input_path": str(input_paths[key_column]),
                "status": status,
                "device_class": device_class,
                "command": " ".join(command),
            }
        )
        if missing_count == 0:
            continue
        jobs.append(
            PipelineJob(
                job_id=f"embed_{modality}",
                stage="embed",
                endpoint="",
                split=None,
                threshold=None,
                seed=None,
                worker_index=0,
                command=command,
                output_dir=run_root / "embeddings" / modality,
                modality=modality,
                requires_gpu=modality in _TASK.gpu_embedders,
                stream_output=True,
            )
        )
    write_dict_rows(run_root / "embedding_plan.csv", plan_rows)
    render_summary_table(
        "Embedding plan",
        [
            {
                **row,
                "total_unique_keys": f"{int(row['total_unique_keys']):,}",
                "existing_keys": f"{int(row['existing_keys']):,}",
                "missing_keys": f"{int(row['missing_keys']):,}",
            }
            for row in plan_rows
        ],
        [
            ("modality", "modality"),
            ("key", "key_column"),
            ("total", "total_unique_keys"),
            ("existing", "existing_keys"),
            ("missing", "missing_keys"),
            ("status", "status"),
            ("device", "device_class"),
        ],
    )
    return jobs


def discover_final_jobs(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for endpoint in args.endpoints:
        for split in args.splits:
            split_dir = _TASK.resolve_split_dir(args.data_root, endpoint, split)
            for seed in args.seeds:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "split": split,
                        "threshold": _TASK.split_threshold(split_dir),
                        "seed": int(seed),
                        "output_dir": final_dir(
                            args.output_root,
                            endpoint,
                            split,
                            _TASK.split_threshold(split_dir),
                            seed,
                        ),
                    }
                )
    return rows


def preflight(args: argparse.Namespace, run_root: Path) -> None:
    issues: list[dict[str, Any]] = []
    modalities = enabled_modalities(args)
    required_columns = _TASK.required_columns_for_modalities(modalities)

    package_rows = []
    for package in ("optuna", "wandb", "sqlalchemy", "pyarrow", "pytest"):
        try:
            __import__(package)
            package_rows.append({"check": f"package:{package}", "status": "ok", "detail": ""})
        except ImportError as exc:
            issues.append({"kind": "missing_package", "package": package, "error": str(exc)})
            package_rows.append(
                {"check": f"package:{package}", "status": "missing", "detail": str(exc)}
            )

    wandb_status = "skipped" if args.skip_wandb_check or args.wandb_offline else "ok"
    wandb_detail = "offline/check skipped" if args.skip_wandb_check or args.wandb_offline else ""
    if not args.skip_wandb_check and not args.wandb_offline:
        netrc = Path.home() / ".netrc"
        if not os.environ.get("WANDB_API_KEY") and (
            not netrc.exists() or "api.wandb.ai" not in netrc.read_text(errors="ignore")
        ):
            issues.append(
                {
                    "kind": "wandb_auth",
                    "message": "WANDB_API_KEY or ~/.netrc api.wandb.ai entry not found.",
                }
            )
            wandb_status = "missing"
            wandb_detail = "WANDB_API_KEY or ~/.netrc api.wandb.ai entry not found"
    package_rows.append({"check": "wandb_auth", "status": wandb_status, "detail": wandb_detail})

    if args.skip_embeddings:
        for name, root in _TASK.cache_roots(args.embeddings_root).items():
            if name not in modalities:
                continue
            if not (root / "meta.json").exists():
                issues.append(
                    {"kind": "missing_cache", "modality": name, "path": str(root / "meta.json")}
                )
            if not index_files(root):  # parts count as an index
                issues.append(
                    {
                        "kind": "missing_cache",
                        "modality": name,
                        "path": str(root / "index.parquet"),
                    }
                )

    schema_tasks: list[tuple[str, str, Path, str]] = []
    for endpoint in args.endpoints:
        for split in args.splits:
            try:
                split_dir = _TASK.resolve_split_dir(args.data_root, endpoint, split)
            except Exception as exc:
                issues.append(
                    {
                        "kind": "missing_split",
                        "endpoint": endpoint,
                        "split": split,
                        "error": str(exc),
                    }
                )
                continue
            for tvt in ("train", "val", "test"):
                schema_tasks.append((endpoint, split, split_dir, tvt))

    for endpoint, split, split_dir, tvt in progress_iter(
        args,
        schema_tasks,
        desc="Preflight schemas",
        total=len(schema_tasks),
        unit="file",
    ):
        path = split_dir / f"{tvt}.parquet"
        schema = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(required_columns - schema)
        if missing:
            issues.append(
                {
                    "kind": "missing_columns",
                    "endpoint": endpoint,
                    "split": split,
                    "threshold": _TASK.split_threshold(split_dir),
                    "tvt": tvt,
                    "path": str(path),
                    "missing": ",".join(missing),
                }
            )

    report_path = run_root / "preflight_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows(report_path, issues)
    summary_rows = [
        {"check": "endpoints", "status": len(args.endpoints), "detail": ",".join(args.endpoints)},
        {"check": "splits", "status": len(args.splits), "detail": ",".join(args.splits)},
        {"check": "modalities", "status": len(modalities), "detail": ",".join(modalities)},
        {"check": "schema_files", "status": len(schema_tasks), "detail": ""},
        {"check": "issues", "status": len(issues), "detail": str(report_path)},
    ]
    render_summary_table(
        "Preflight summary",
        [*package_rows, *summary_rows],
        [("check", "check"), ("status", "status"), ("detail", "detail")],
    )
    if issues:
        raise RuntimeError(f"Preflight failed with {len(issues)} issue(s). See {report_path}")
    log_step("Preflight passed")


def hydra_profile(args: argparse.Namespace) -> dict[str, Any]:
    profile = getattr(args, "hydra_profile", {}) or {}
    return dict(profile)


def hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{hydra_value(item)}" for key, item in value.items()) + "}"
    return str(value)


def hydra_override_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.replace("\\", "\\\\").replace('"', '\\"')
        if text == "" or any(char in text for char in "{}[],:= \t\n"):
            return f'"{text}"'
    return hydra_value(value)


def flatten_hydra_mapping(prefix: str, values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, dict):
        return [f"++{prefix}={hydra_override_value(values)}"]
    overrides: list[str] = []
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            overrides.extend(flatten_hydra_mapping(path, value))
        else:
            overrides.append(f"++{path}={hydra_override_value(value)}")
    return overrides


def profile_raw_overrides(args: argparse.Namespace, scope: str) -> list[str]:
    raw = (hydra_profile(args).get("raw_overrides") or {}).get(scope) or []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def profile_config_group_overrides(args: argparse.Namespace) -> list[str]:
    groups = hydra_profile(args).get("config_groups")
    if not groups:
        groups = {
            "data": "baseline_regression",
            "model": "baseline_regression",
            "scheduler": "cosine_warm_restarts",
            "trainer": "gpu",
            "logger": "wandb",
        }
    overrides: list[str] = []
    for key, value in groups.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            overrides.append(f"{key}=[{','.join(str(item) for item in value)}]")
        else:
            overrides.append(f"{key}={value}")
    return overrides


def stage_config_group_overrides(args: argparse.Namespace, stage: str) -> list[str]:
    overrides = profile_config_group_overrides(args)
    replacement: Optional[str] = None
    groups = (hydra_profile(args).get("config_groups") or {}).get("callbacks")
    if isinstance(groups, (list, tuple)):
        callback_groups = [str(item) for item in groups]
        if stage == "hpo" and not getattr(args, "hpo_checkpoints", False):
            callback_groups = [item for item in callback_groups if item != "model_checkpoint"]
            replacement = f"callbacks=[{','.join(callback_groups)}]"
        elif (
            stage == "final"
            and getattr(args, "final_checkpoints", True)
            and "model_checkpoint" not in callback_groups
        ):
            callback_groups.append("model_checkpoint")
            replacement = f"callbacks=[{','.join(callback_groups)}]"
        elif stage == "final" and not getattr(args, "final_checkpoints", True):
            callback_groups = [item for item in callback_groups if item != "model_checkpoint"]
            replacement = f"callbacks=[{','.join(callback_groups)}]"
    if replacement is None:
        return overrides
    return [item for item in overrides if not item.startswith("callbacks=")] + [replacement]


def profile_common_overrides(args: argparse.Namespace) -> list[str]:
    common = hydra_profile(args).get("common") or {}
    return flatten_hydra_mapping("", common)


def profile_stage_overrides(args: argparse.Namespace, stage: str) -> list[str]:
    stage_cfg = hydra_profile(args).get(stage) or {}
    ignored = {
        "hparams_search",
        "optimized_metric",
        "direction",
        "search_space",
        "params_source",
        "best_params_path",
        "reuse_existing_successes",
        "checkpoints",
    }
    fixed = {key: value for key, value in stage_cfg.items() if key not in ignored}
    return flatten_hydra_mapping("", fixed)


def profile_hpo_overrides(args: argparse.Namespace) -> list[str]:
    hpo_cfg = hydra_profile(args).get("hpo") or {}
    overrides: list[str] = ["test=false"]
    if not getattr(args, "hpo_checkpoints", False):
        # Without this Lightning adds a default ModelCheckpoint that writes every trial's weights.
        overrides.append("++trainer.enable_checkpointing=false")
    hparams_search = hpo_cfg.get("hparams_search", "baseline_regression_optuna4")
    if hparams_search:
        overrides.append(f"hparams_search={hparams_search}")
    direction = hpo_cfg.get("direction")
    if direction:
        overrides.append(f"hydra.sweeper.direction={direction}")
    search_space = hpo_cfg.get("search_space") or {}
    if search_space:
        overrides.append("~hydra.sweeper.params")
        for key, value in search_space.items():
            text = hydra_value(value).replace("\\", "\\\\").replace('"', '\\"')
            overrides.append(f'++hydra.sweeper.params.{key}="{text}"')
    overrides.extend(profile_stage_overrides(args, "hpo"))
    return overrides


def hpo_sampler_seed_override(args: argparse.Namespace, worker_index: int) -> list[str]:
    strategy = str(getattr(args, "hpo_sampler_seed_strategy", "worker_offset") or "none")
    seed_base = getattr(args, "hpo_sampler_seed_base", None)
    if strategy == "none" or seed_base is None:
        return []
    if strategy == "worker_offset":
        seed = int(seed_base) + int(worker_index)
    elif strategy == "fixed":
        seed = int(seed_base)
    else:
        raise ValueError(f"Unsupported HPO sampler seed strategy: {strategy}")
    return [f"hydra.sweeper.sampler.seed={seed}"]


def profile_final_overrides(args: argparse.Namespace) -> list[str]:
    return profile_stage_overrides(args, "final")


def optimized_metric(args: argparse.Namespace) -> str:
    hpo_cfg = hydra_profile(args).get("hpo") or {}
    return str(hpo_cfg.get("optimized_metric", "val/r2"))


def current_progress_metric(state: dict[str, Any]) -> Any:
    return state.get("current_optimized_metric", state.get("current_val_r2"))


def best_progress_metric(state: dict[str, Any]) -> Any:
    return state.get("best_optimized_metric", state.get("best_val_r2"))


def run_tags(args: argparse.Namespace, *extra: Any) -> list[str]:
    base = list(
        getattr(args, "run_tags", None)
        or ["baseline_regression", getattr(args, "run_label", "mlp")]
    )
    for item in extra:
        if item is None:
            continue
        text = str(item)
        if text not in base:
            base.append(text)
    return base


def tag_override(args: argparse.Namespace, *extra: Any) -> str:
    return "+wandb_tags=[" + ",".join(run_tags(args, *extra)) + "]"


def hpo_split(args: argparse.Namespace, endpoint: Optional[str] = None) -> str:
    # Resolve a task-specific "auto" split per endpoint.
    if endpoint is not None and _TASK.hpo_split_for_endpoint is not None:
        return _TASK.hpo_split_for_endpoint(args, endpoint)
    return str(getattr(args, "hpo_split", "enzyme_structure_splits"))


def hpo_study_suffix(args: argparse.Namespace) -> str:
    split = hpo_split(args)
    if split == "enzyme_structure_splits":
        return "structure"
    return split


def data_fingerprint(
    args: argparse.Namespace, endpoint: str, split_name: str, threshold: Optional[str] = None
) -> str:
    """Short hash of the split files and embedding indexes (path, size, mtime; no file reads)."""
    # Study names and job ids carry this hash, so a rebuilt file does not reuse old trials or metrics.
    if not hasattr(
        args, "data_root"
    ):  # best_trial_summary_for_endpoint can pass an empty Namespace
        return "nodata"
    try:
        split_dir = _TASK.resolve_split_dir(args.data_root, endpoint, split_name, threshold)
    except FileNotFoundError:  # a final-only run can lack the HPO split
        return "nodata"
    roots = _TASK.cache_roots(args.embeddings_root)
    paths = [split_dir / f"{name}.parquet" for name in ("train", "val", "test")]
    for m in enabled_modalities(args):
        if m in roots:
            paths += index_files(roots[m])  # index.parquet plus unmerged parts
    parts = [
        f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}"
        for p in paths
        if p.exists()
        for st in [p.stat()]
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def study_name_for_endpoint(args: argparse.Namespace, endpoint: str) -> str:
    fingerprint = data_fingerprint(args, endpoint, hpo_split(args))
    return f"{getattr(args, 'study_prefix', 'baseline_mlp')}_{endpoint}_{hpo_study_suffix(args)}_{fingerprint}"


def build_base_overrides(
    args: argparse.Namespace, endpoint: str, output_dir: Path, stage: str
) -> list[str]:
    group = f"{args.pipeline_id}/{endpoint}"
    modalities = ",".join(enabled_modalities(args))
    max_epochs = (
        args.final_max_epochs
        if stage == "final" and args.final_max_epochs is not None
        else args.max_epochs
    )
    use_materialized = not args.skip_materialization
    # Pass the data root and each enabled modality's cache root to every child run.
    roots = _TASK.cache_roots(args.embeddings_root)
    cache_root_overrides = [
        f"data.cache_roots.{modality}={roots[modality].resolve()}"
        for modality in enabled_modalities(args)
        if modality in roots
    ]
    return [
        *stage_config_group_overrides(args, stage),
        *profile_common_overrides(args),
        f"data.data_root={Path(args.data_root).resolve()}",
        *cache_root_overrides,
        f"data.modalities=[{modalities}]",
        f"data.endpoint={endpoint}",
        f"trainer.max_epochs={max_epochs}",
        f"paths.output_dir={output_dir}",
        f"data.use_materialized_features={str(use_materialized).lower()}",
        "data.materialize_if_missing=true",
        "data.refresh_materialized_features=false",
        f"data.materialized_cache_root={args.materialized_cache_root}",
        f"data.materialized_feature_dtype={args.materialized_feature_dtype}",
        f"data.materialized_block_rows={args.materialized_block_rows}",
        f"data.materialized_validation={args.materialized_validation}",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        f"logger.wandb.project={args.wandb_project}",
        f"logger.wandb.entity={args.wandb_entity}",
        f"logger.wandb.offline={str(args.wandb_offline).lower()}",
        f"+wandb_group={group}",
        f"+wandb_job_type={stage}",
        tag_override(args, stage, endpoint),
        f"++optimized_metric={optimized_metric(args)}",
        *profile_raw_overrides(args, "common"),
    ]


def build_hpo_jobs(args: argparse.Namespace, run_root: Path) -> list[PipelineJob]:
    optuna_dir = run_root / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[PipelineJob] = []
    for endpoint in args.endpoints:
        hpo_split_name = hpo_split(args)
        hpo_split_dir = _TASK.resolve_split_dir(args.data_root, endpoint, hpo_split_name)
        hpo_threshold = _TASK.split_threshold(hpo_split_dir)
        study_name = _TASK.study_name_for_endpoint(args, endpoint)
        storage = optuna_storage_url(run_root, study_name)
        for worker_index in range(args.hpo_workers_per_endpoint):
            output_dir = run_root / "hpo" / endpoint / f"worker_{worker_index}"
            trial_output_dir = f"{output_dir}/trials/trial_{OPTUNA_TRIAL_NUMBER_TOKEN}"
            overrides = _TASK.build_base_overrides(args, endpoint, Path(trial_output_dir), "hpo")
            overrides.extend(
                [
                    *profile_hpo_overrides(args),
                    f"data.split={hpo_split_name}",
                    f"data.threshold={hpo_threshold}" if hpo_threshold else "data.threshold=null",
                    f"trainer.max_epochs={args.max_epochs}",
                    # Each worker gets its own sweep dir.
                    f"hydra.sweep.dir={output_dir}/multirun",
                    f"hydra.sweeper.study_name={study_name}",
                    f"hydra.sweeper.storage={storage}",
                    f"hydra.sweeper.n_trials={args.n_trials}",
                    f"+wandb_name=hpo_{endpoint}_worker_{worker_index}",
                    *profile_raw_overrides(args, "hpo"),
                    *hpo_sampler_seed_override(args, worker_index),
                ]
            )
            command = [sys.executable, str(TRAIN_SCRIPT), "-m", *overrides]
            jobs.append(
                PipelineJob(
                    job_id=f"hpo_{endpoint}_worker_{worker_index}_{study_name.rsplit('_', 1)[-1]}",
                    stage="hpo",
                    endpoint=endpoint,
                    split=hpo_split_name,
                    threshold=hpo_threshold,
                    seed=None,
                    worker_index=worker_index,
                    command=command,
                    output_dir=output_dir,
                )
            )
    return jobs


def optuna_storage_url(run_root: Path, study_name: str) -> str:
    storage_path = run_root / "optuna" / f"{study_name}.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path}"


def best_params_for_endpoint(run_root: Path, endpoint: str) -> dict[str, Any]:
    return best_trial_summary_for_endpoint(run_root, endpoint)["best_params"]


def final_best_params_path(args: argparse.Namespace, run_root: Path) -> Path:
    path = getattr(args, "final_best_params_path", None)
    if path is None:
        return run_root / "hpo_best_params.csv"
    return Path(path)


def load_best_params_csv(path: Path, endpoints: Iterable[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Final best-params CSV does not exist: {path}")
    rows = read_dict_rows(path)
    if not rows:
        raise ValueError(f"Final best-params CSV is empty: {path}")
    params_by_endpoint: dict[str, dict[str, Any]] = {}
    for row in rows:
        endpoint = row.get("endpoint")
        if not endpoint:
            continue
        params_by_endpoint[str(endpoint)] = {
            key: value
            for key, value in row.items()
            if key != "endpoint" and value not in {"", None}
        }
    missing = [endpoint for endpoint in endpoints if endpoint not in params_by_endpoint]
    if missing:
        raise KeyError(f"Best-params CSV {path} is missing endpoint(s): {', '.join(missing)}")
    return {endpoint: params_by_endpoint[endpoint] for endpoint in endpoints}


def best_trial_summary_for_endpoint(run_root: Path, endpoint: str) -> dict[str, Any]:
    study_name = _TASK.study_name_for_endpoint(_ACTIVE_ARGS or argparse.Namespace(), endpoint)
    storage = optuna_storage_url(run_root, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trial
    return {
        "endpoint": endpoint,
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": dict(best_trial.params),
    }


def endpoint_hpo_completed(args: argparse.Namespace, run_root: Path, endpoint: str) -> bool:
    try:
        study_name = _TASK.study_name_for_endpoint(args, endpoint)
        storage = optuna_storage_url(run_root, study_name)
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception:
        return False
    completed = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return len(completed) >= int(args.n_trials)


def upsert_best_params_csv(run_root: Path, rows: list[dict[str, Any]]) -> None:
    path = run_root / "hpo_best_params.csv"
    existing = {row.get("endpoint"): row for row in read_dict_rows(path) if row.get("endpoint")}
    for row in rows:
        endpoint = row.get("endpoint")
        if endpoint:
            existing[endpoint] = row
    write_dict_rows(path, [existing[endpoint] for endpoint in sorted(existing)])


def best_params_overrides(params: dict[str, Any]) -> list[str]:
    return [f"++{key}={stringify_hydra_value(value)}" for key, value in sorted(params.items())]


def stringify_hydra_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(stringify_hydra_value(item) for item in value) + "]"
    return str(value)


def build_final_jobs(
    args: argparse.Namespace,
    run_root: Path,
    params_by_endpoint: Optional[dict[str, dict[str, Any]]] = None,
    *,
    write_best_params_csv: bool = True,
) -> list[PipelineJob]:
    jobs: list[PipelineJob] = []
    best_rows = []
    best_dir = run_root / "best_hparams"
    best_dir.mkdir(parents=True, exist_ok=True)
    for endpoint in args.endpoints:
        params = (params_by_endpoint or {}).get(endpoint)
        if params is None:
            summary = best_trial_summary_for_endpoint(run_root, endpoint)
            params = summary["best_params"]
        else:
            summary = {
                "endpoint": endpoint,
                "study_name": _TASK.study_name_for_endpoint(args, endpoint),
                "best_trial_number": None,
                "best_value": None,
                "best_params": params,
            }
        (best_dir / f"{endpoint}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        best_rows.append({"endpoint": endpoint, **params})
        param_overrides = best_params_overrides(params)
        for spec in _TASK.discover_final_jobs(args, run_root):
            if spec["endpoint"] != endpoint:
                continue
            output_dir = Path(spec["output_dir"])
            threshold = spec["threshold"]
            seed = int(spec["seed"])
            split = spec["split"]
            overrides = _TASK.build_base_overrides(args, endpoint, output_dir, "final")
            overrides.extend(
                [
                    *profile_final_overrides(args),
                    f"data.split={split}",
                    f"data.threshold={threshold}" if threshold else "data.threshold=null",
                    f"seed={seed}",
                    f"+wandb_name=final_{endpoint}_{split}_{threshold or 'direct'}_seed_{seed}",
                    tag_override(args, "final", endpoint, split, f"seed_{seed}"),
                    *param_overrides,
                    *profile_raw_overrides(args, "final"),
                ]
            )
            jobs.append(
                PipelineJob(
                    job_id=(
                        f"final_{endpoint}_{split}_{threshold or 'direct'}_seed_{seed}"
                        f"_{data_fingerprint(args, endpoint, split, threshold)}"
                    ),
                    stage="final",
                    endpoint=endpoint,
                    split=split,
                    threshold=threshold,
                    seed=seed,
                    worker_index=0,
                    command=[sys.executable, str(TRAIN_SCRIPT), *overrides],
                    output_dir=output_dir,
                )
            )
    if write_best_params_csv:
        write_dict_rows(run_root / "hpo_best_params.csv", best_rows)
    return jobs


def build_final_jobs_for_endpoint(
    args: argparse.Namespace, run_root: Path, endpoint: str
) -> tuple[list[PipelineJob], list[dict[str, Any]]]:
    summary = best_trial_summary_for_endpoint(run_root, endpoint)
    scoped_args = copy.copy(args)
    scoped_args.endpoints = [endpoint]
    jobs = build_final_jobs(
        scoped_args,
        run_root,
        params_by_endpoint={endpoint: summary["best_params"]},
        write_best_params_csv=False,
    )
    return jobs, [{"endpoint": endpoint, **summary["best_params"]}]


def materialization_specs_for_hpo(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for endpoint in args.endpoints:
        split_name = hpo_split(args)
        split_dir = _TASK.resolve_split_dir(args.data_root, endpoint, split_name)
        specs.append(
            {
                "endpoint": endpoint,
                "split": split_name,
                "threshold": _TASK.split_threshold(split_dir),
            }
        )
    return specs


def materialization_specs_for_final(
    args: argparse.Namespace, run_root: Path
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, Optional[str]]] = set()
    specs: list[dict[str, Any]] = []
    for spec in _TASK.discover_final_jobs(args, run_root):
        key = (spec["endpoint"], spec["split"], spec["threshold"])
        if key in seen:
            continue
        seen.add(key)
        specs.append({"endpoint": key[0], "split": key[1], "threshold": key[2]})
    return specs


def materialization_specs_for_final_endpoint(
    args: argparse.Namespace, run_root: Path, endpoint: str
) -> list[dict[str, Any]]:
    return [
        spec
        for spec in _TASK.materialization_specs_for_final(args, run_root)
        if spec["endpoint"] == endpoint
    ]


def run_materialization_stage(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    run_root: Path,
    status_path: Path,
    *,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    if args.skip_materialization:
        rows = [
            {
                **spec,
                "stage": "materialize",
                "status": "skipped",
                "reason": "skip_materialization",
            }
            for spec in specs
        ]
        write_dict_rows(status_path, rows)
        return rows

    if args.dry_run:
        rows = [_TASK.materialization_status_row(args, spec, dry_run=True) for spec in specs]
        write_dict_rows(status_path, rows)
        if not quiet:
            render_materialization_summary(args, rows, title="Materialization plan")
        return rows

    rows: list[dict[str, Any]] = []
    prior_rows = latest_materialization_rows(read_dict_rows(status_path))
    iterator = specs
    for spec in iterator:
        prior = prior_rows.get(materialization_key(spec))
        row = _TASK.materialization_status_row(
            args,
            spec,
            dry_run=False,
            prior_row=prior,
        )
        rows.append(row)
        write_dict_rows(status_path, rows)
        if row["status"] not in {"hit", "built"}:
            raise RuntimeError(f"Materialization failed for {spec}: {row.get('reason')}")
        if not rich_enabled(args):
            log_step(
                "Materialized "
                f"{spec['endpoint']}/{spec['split']}/{spec.get('threshold') or 'direct'}: "
                f"{row['status']} ({float(row['estimated_gb']):.2f} GiB)"
            )
    if not quiet:
        render_materialization_summary(args, rows, title="Materialization summary")
    return rows


def materialization_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("endpoint") or ""),
        str(row.get("split") or ""),
        str(row.get("threshold") or ""),
    )


def latest_materialization_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        latest[materialization_key(row)] = row
    return latest


def materialization_status_row(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    dry_run: bool,
    prior_row: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from src.data.baseline_regression_datamodule import BaselineRegressionDataModule

    dm = BaselineRegressionDataModule(
        data_root=str(args.data_root),
        endpoint=spec["endpoint"],
        split=spec["split"],
        threshold=spec.get("threshold"),
        modalities=enabled_modalities(args),
        cache_roots=_TASK.cache_roots(args.embeddings_root),
        batch_size=4096,
        parquet_batch_size=4096,
        num_workers=0,
        pin_memory=False,
        use_materialized_features=True,
        materialize_if_missing=not dry_run,
        refresh_materialized_features=args.refresh_materialized_features and not dry_run,
        materialized_cache_root=str(args.materialized_cache_root),
        materialized_feature_dtype=args.materialized_feature_dtype,
        materialized_block_rows=args.materialized_block_rows,
        materialized_validation=args.materialized_validation,
        materialization_max_gb=10_000.0,
        pooled_list_cache_size=100000,
        seed=666,
    )
    started_at = time.monotonic()
    try:
        if (
            not dry_run
            and not args.refresh_materialized_features
            and prior_row is not None
            and prior_row.get("status") in {"hit", "built", "fast_hit"}
        ):
            status = dm.materialized_cache_status()
            if status.get("status") == "hit":
                elapsed_sec = time.monotonic() - started_at
                return {
                    **spec,
                    "stage": "materialize",
                    "status": "hit",
                    "reason": "fast_hit",
                    "elapsed_sec": f"{elapsed_sec:.3f}",
                    "rows": status.get("rows") or prior_row.get("rows", ""),
                    "estimated_gb": status.get("estimated_gb")
                    or prior_row.get("estimated_gb", ""),
                    "cache_dir": status["cache_dir"],
                    "feature_dtype": args.materialized_feature_dtype,
                    "validation": args.materialized_validation,
                }
        if dry_run:
            status = dm.materialized_cache_status()
        else:
            status = dm.ensure_materialized_cache(refresh=args.refresh_materialized_features)
        elapsed_sec = time.monotonic() - started_at
        cache_status = status.get("cache_status") or status.get("status")
        return {
            **spec,
            "stage": "materialize",
            "status": cache_status,
            "reason": status.get("reason", "ok"),
            "elapsed_sec": f"{elapsed_sec:.3f}",
            "rows": status.get("rows")
            or sum(split_info["rows"] for split_info in status["splits"].values()),
            "estimated_gb": status.get("estimated_gb") or status.get("estimated_total_gb"),
            "cache_dir": status["cache_dir"],
            "feature_dtype": args.materialized_feature_dtype,
            "validation": args.materialized_validation,
        }
    except Exception as exc:
        elapsed_sec = time.monotonic() - started_at
        return {
            **spec,
            "stage": "materialize",
            "status": "failed",
            "reason": str(exc),
            "elapsed_sec": f"{elapsed_sec:.3f}",
            "rows": "",
            "estimated_gb": "",
            "cache_dir": "",
            "feature_dtype": args.materialized_feature_dtype,
            "validation": args.materialized_validation,
        }


def render_materialization_summary(
    args: argparse.Namespace, rows: list[dict[str, Any]], *, title: str
) -> None:
    if not rich_enabled(args) or Table is None or _CONSOLE is None:
        return
    table = Table(title=title)
    for column in (
        "endpoint",
        "split",
        "threshold",
        "status",
        "reason",
        "sec",
        "rows",
        "GiB",
        "dtype",
        "cache",
    ):
        table.add_column(column)
    for row in rows:
        estimated = row.get("estimated_gb")
        table.add_row(
            str(row.get("endpoint", "")),
            str(row.get("split", "")),
            str(row.get("threshold") or "direct"),
            str(row.get("status", "")),
            str(row.get("reason", "")),
            str(row.get("elapsed_sec", "")),
            str(row.get("rows", "")),
            "" if estimated in {"", None} else f"{float(estimated):.2f}",
            str(row.get("feature_dtype", "")),
            str(row.get("cache_dir", "")),
        )
    _CONSOLE.print(table)


def storage_profile(args: argparse.Namespace) -> dict[str, Any]:
    return deep_update(default_storage_profile(), getattr(args, "storage_profile", {}) or {})


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024**3)


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() or item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


def delete_path(path: Path, *, dry_run: bool) -> tuple[int, str]:
    size = path_size_bytes(path)
    if not path.exists():
        return 0, "missing"
    if dry_run:
        return size, "dry_run"
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size, "deleted"


def cleanup_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "cleanup", True))


def cleanup_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "cleanup_dry_run", False) or getattr(args, "dry_run", False))


def cleanup_job_output(args: argparse.Namespace, job: PipelineJob) -> list[dict[str, Any]]:
    if not cleanup_enabled(args):
        return []
    cfg = storage_profile(args)
    artifact_cfg = cfg.get("job_artifacts", {})
    dry_run = cleanup_dry_run(args)
    rows: list[dict[str, Any]] = []

    candidates: list[tuple[Path, str, str]] = []
    if artifact_cfg.get("delete_job_materialized_features", True):
        candidates.append(
            (
                job.output_dir / "materialized_features",
                "job_materialized_features",
                "job-local materialized cache",
            )
        )
        if job.stage == "hpo":
            candidates.extend(
                (path, "job_materialized_features", "hpo trial-local materialized cache")
                for path in job.output_dir.glob("trials/trial_*/materialized_features")
            )
    if job.stage == "hpo" and artifact_cfg.get("delete_hpo_predictions", True):
        candidates.extend(
            (path, "hpo_predictions", "hpo predictions")
            for path in job.output_dir.glob("trials/trial_*/baseline_estimator/*_predictions.npy")
        )
    if job.stage == "hpo" and artifact_cfg.get("delete_hpo_models", False):
        candidates.extend(
            (path, "hpo_model", "hpo fitted model")
            for path in job.output_dir.glob("trials/trial_*/baseline_estimator/model.joblib")
        )

    seen: set[Path] = set()
    for path, kind, reason in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        size, action = delete_path(path, dry_run=dry_run)
        rows.append(
            {
                "stage": job.stage,
                "job_id": job.job_id,
                "endpoint": job.endpoint,
                "split": job.split,
                "threshold": job.threshold,
                "seed": job.seed,
                "kind": kind,
                "path": str(path),
                "bytes": size,
                "gib": f"{bytes_to_gib(size):.4f}",
                "action": action,
                "reason": reason,
            }
        )
    return rows


def cleanup_manifest_rows_from_job_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        cleanup_payload = row.get("cleanup_manifest")
        if not cleanup_payload:
            continue
        try:
            payload = json.loads(cleanup_payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            output.extend(item for item in payload if isinstance(item, dict))
    return output


def append_cleanup_manifest(run_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = run_root / "cleanup_manifest.csv"
    existing = read_dict_rows(path)
    write_dict_rows(path, existing + rows)


def write_disk_report(args: argparse.Namespace, run_root: Path, name: str) -> list[dict[str, Any]]:
    cfg = storage_profile(args)
    if not cfg.get("logs", {}).get("write_disk_report", True):
        return []
    targets = [run_root]
    if args.materialized_cache_root is not None:
        targets.append(Path(args.materialized_cache_root))
    rows: list[dict[str, Any]] = []
    for target in targets:
        if not target.exists():
            continue
        for path in sorted(target.iterdir() if target.is_dir() else [target]):
            size = path_size_bytes(path)
            rows.append(
                {
                    "path": str(path),
                    "kind": "dir" if path.is_dir() else "file",
                    "bytes": size,
                    "gib": f"{bytes_to_gib(size):.4f}",
                }
            )
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    write_dict_rows(run_root / f"disk_usage_{name}.csv", rows)
    return rows


def render_disk_summary(
    args: argparse.Namespace, rows: list[dict[str, Any]], *, title: str
) -> None:
    if not rows:
        return
    render_summary_table(
        title,
        rows[:10],
        [("GiB", "gib"), ("kind", "kind"), ("path", "path")],
    )


def check_min_free_space(args: argparse.Namespace) -> None:
    cfg = storage_profile(args).get("shared_materialized_cache", {})
    min_free_gb = cfg.get("min_free_gb")
    if min_free_gb in (None, "") or args.materialized_cache_root is None:
        return
    root = Path(args.materialized_cache_root)
    root.mkdir(parents=True, exist_ok=True)
    free_gb = bytes_to_gib(shutil.disk_usage(root).free)
    if free_gb >= float(min_free_gb):
        return
    message = f"Free disk under {root} is {free_gb:.1f} GiB, below storage min_free_gb={float(min_free_gb):.1f}"
    if cfg.get("fail_if_exceeded", False):
        raise RuntimeError(message)
    log_step(f"WARNING: {message}")


def complete_materialized_cache_entries(cache_root: Path) -> list[dict[str, Any]]:
    if not cache_root.exists():
        return []
    entries = []
    for metadata_path in cache_root.glob("*/*/*/*/metadata.json"):
        entry_dir = metadata_path.parent
        if not (entry_dir / ".complete").exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        size = path_size_bytes(entry_dir)
        entries.append(
            {
                "path": entry_dir,
                "bytes": size,
                "mtime": entry_dir.stat().st_mtime,
                "endpoint": metadata.get("endpoint", ""),
                "split": metadata.get("split", ""),
                "threshold": metadata.get("threshold", ""),
                "fingerprint": metadata.get("fingerprint", entry_dir.name),
            }
        )
    return entries


def enforce_shared_cache_budget(
    args: argparse.Namespace,
    run_root: Path,
    protected_cache_dirs: Iterable[str | Path],
) -> list[dict[str, Any]]:
    if args.materialized_cache_root is None or not cleanup_enabled(args):
        return []
    cfg = storage_profile(args).get("shared_materialized_cache", {})
    if not cfg.get("enabled", True):
        return []
    cache_root = Path(args.materialized_cache_root)
    entries = complete_materialized_cache_entries(cache_root)
    protected = {Path(path).resolve() for path in protected_cache_dirs if path}
    rows: list[dict[str, Any]] = []

    max_entry_gb = cfg.get("max_entry_gb")
    for entry in entries:
        gib = bytes_to_gib(int(entry["bytes"]))
        if max_entry_gb not in (None, "") and gib > float(max_entry_gb):
            rows.append(
                {
                    "stage": "storage",
                    "job_id": "",
                    "endpoint": entry["endpoint"],
                    "split": entry["split"],
                    "threshold": entry["threshold"],
                    "seed": "",
                    "kind": "shared_materialized_cache",
                    "path": str(entry["path"]),
                    "bytes": entry["bytes"],
                    "gib": f"{gib:.4f}",
                    "action": "warn",
                    "reason": f"entry exceeds max_entry_gb={max_entry_gb}",
                }
            )

    max_total_gb = cfg.get("max_total_gb")
    if max_total_gb in (None, ""):
        append_cleanup_manifest(run_root, rows)
        return rows
    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    target_bytes = int(float(max_total_gb) * 1024**3)
    if total_bytes <= target_bytes:
        append_cleanup_manifest(run_root, rows)
        return rows

    dry_run = cleanup_dry_run(args)
    candidates = sorted(entries, key=lambda entry: float(entry["mtime"]))
    for entry in candidates:
        entry_path = Path(entry["path"]).resolve()
        if entry_path in protected:
            continue
        if total_bytes <= target_bytes:
            break
        size, action = delete_path(entry_path, dry_run=dry_run)
        total_bytes -= size
        rows.append(
            {
                "stage": "storage",
                "job_id": "",
                "endpoint": entry["endpoint"],
                "split": entry["split"],
                "threshold": entry["threshold"],
                "seed": "",
                "kind": "shared_materialized_cache",
                "path": str(entry_path),
                "bytes": size,
                "gib": f"{bytes_to_gib(size):.4f}",
                "action": action,
                "reason": f"shared cache total exceeded max_total_gb={max_total_gb}",
            }
        )

    append_cleanup_manifest(run_root, rows)
    if total_bytes > target_bytes and cfg.get("fail_if_exceeded", False):
        raise RuntimeError(
            f"Shared materialized cache remains above max_total_gb={max_total_gb} after pruning protected entries."
        )
    return rows


def protected_checkpoint_names(stage: str, cfg: dict[str, Any]) -> set[str]:
    protected: set[str] = set()
    if stage == "hpo":
        if cfg.get("keep_hpo_last_checkpoint", False):
            protected.add("last.ckpt")
        # Best checkpoint names are project-specific epoch_*.ckpt files. They
        # are intentionally not protected unless the user opts in.
        if not cfg.get("keep_hpo_best_checkpoint", False):
            return protected
    if stage == "final":
        if cfg.get("keep_final_last_checkpoint", False):
            protected.add("last.ckpt")
        if not cfg.get("keep_final_best_checkpoint", True):
            return protected
    return protected


def checkpoint_cleanup_candidates(
    args: argparse.Namespace, run_root: Path, stage: str, cfg: dict[str, Any]
) -> list[Path]:
    roots = final_endpoint_dirs(args) if stage == "final" else [run_root / stage]
    protected = protected_checkpoint_names(stage, cfg)
    candidates = []
    for path in (path for root in roots if root.exists() for path in root.glob("**/*.ckpt")):
        if path.name in protected:
            continue
        if (
            stage == "final"
            and cfg.get("keep_final_best_checkpoint", True)
            and path.name != "last.ckpt"
        ):
            continue
        candidates.append(path)
    return candidates


def prune_checkpoints(
    args: argparse.Namespace, run_root: Path, stage: str
) -> list[dict[str, Any]]:
    if not cleanup_enabled(args):
        return []
    cfg = storage_profile(args).get("checkpoints", {})
    prune_enabled = (stage == "hpo" and cfg.get("prune_hpo_checkpoints", True)) or (
        stage == "final" and cfg.get("prune_final_checkpoints", False)
    )
    if not prune_enabled:
        return []
    dry_run = cleanup_dry_run(args)
    rows: list[dict[str, Any]] = []
    for path in checkpoint_cleanup_candidates(args, run_root, stage, cfg):
        size, action = delete_path(path, dry_run=dry_run)
        rows.append(
            {
                "stage": stage,
                "job_id": "",
                "endpoint": "",
                "split": "",
                "threshold": "",
                "seed": "",
                "kind": "checkpoint",
                "path": str(path),
                "bytes": size,
                "gib": f"{bytes_to_gib(size):.4f}",
                "action": action,
                "reason": f"{stage} checkpoint pruning enabled",
            }
        )
    append_cleanup_manifest(run_root, rows)
    return rows


def prune_checkpoint_budget(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    if not cleanup_enabled(args):
        return []
    cfg = storage_profile(args).get("checkpoints", {})
    max_gb = cfg.get("max_checkpoint_total_gb")
    if max_gb in (None, ""):
        return []
    checkpoint_paths = [
        path
        for root in [run_root, *final_endpoint_dirs(args)]
        for path in root.glob("**/*.ckpt")
        if path.is_file()
    ]
    total = sum(path_size_bytes(path) for path in checkpoint_paths)
    target = int(float(max_gb) * 1024**3)
    if total <= target:
        return []
    eligible = []
    for stage in ("hpo", "final"):
        eligible.extend(checkpoint_cleanup_candidates(args, run_root, stage, cfg))
    eligible = sorted(set(eligible), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    rows: list[dict[str, Any]] = []
    dry_run = cleanup_dry_run(args)
    for path in eligible:
        if total <= target:
            break
        size, action = delete_path(path, dry_run=dry_run)
        total -= size
        rows.append(
            {
                "stage": "storage",
                "job_id": "",
                "endpoint": "",
                "split": "",
                "threshold": "",
                "seed": "",
                "kind": "checkpoint",
                "path": str(path),
                "bytes": size,
                "gib": f"{bytes_to_gib(size):.4f}",
                "action": action,
                "reason": f"checkpoint total exceeded max_checkpoint_total_gb={max_gb}",
            }
        )
    append_cleanup_manifest(run_root, rows)
    return rows


def cleanup_existing_run(args: argparse.Namespace, run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not cleanup_enabled(args):
        return rows
    cfg = storage_profile(args)
    dry_run = cleanup_dry_run(args)
    if cfg.get("job_artifacts", {}).get("delete_job_materialized_features", True):
        for path in (
            path
            for root in [run_root, *final_endpoint_dirs(args)]
            for path in root.glob("**/materialized_features")
        ):
            # Only remove job-local materialized dirs, never the shared cache.
            if path.name != "materialized_features":
                continue
            size, action = delete_path(path, dry_run=dry_run)
            rows.append(
                {
                    "stage": "existing",
                    "job_id": "",
                    "endpoint": "",
                    "split": "",
                    "threshold": "",
                    "seed": "",
                    "kind": "job_materialized_features",
                    "path": str(path),
                    "bytes": size,
                    "gib": f"{bytes_to_gib(size):.4f}",
                    "action": action,
                    "reason": "existing run job-local materialized cache",
                }
            )
    append_cleanup_manifest(run_root, rows)
    rows.extend(prune_checkpoints(args, run_root, "hpo"))
    rows.extend(prune_checkpoints(args, run_root, "final"))
    rows.extend(prune_checkpoint_budget(args, run_root))
    return rows


def cleanup_summary(
    args: argparse.Namespace, rows: list[dict[str, Any]], *, title: str = "Cleanup summary"
) -> None:
    if not rows:
        log_step(f"{title}: no cleanup candidates")
        return
    total = sum(
        int(row.get("bytes", 0) or 0)
        for row in rows
        if row.get("action") in {"deleted", "dry_run"}
    )
    log_step(
        f"{title}: {len(rows)} item(s), {bytes_to_gib(total):.2f} GiB {'candidate' if cleanup_dry_run(args) else 'processed'}"
    )
    render_summary_table(
        title,
        sorted(rows, key=lambda row: int(row.get("bytes", 0) or 0), reverse=True)[:20],
        [("GiB", "gib"), ("action", "action"), ("kind", "kind"), ("path", "path")],
    )


def latest_rows_by_job_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = row.get("job_id")
        if job_id:
            latest[str(job_id)] = row
    return latest


def latest_rows_for_jobs(
    rows: Iterable[dict[str, Any]], jobs: Iterable[PipelineJob]
) -> list[dict[str, Any]]:
    latest = latest_rows_by_job_id(rows)
    return [latest[job.job_id] for job in jobs if job.job_id in latest]


def jobs_succeeded(rows: Iterable[dict[str, Any]], jobs: Iterable[PipelineJob]) -> bool:
    latest = latest_rows_by_job_id(rows)
    return all(latest.get(job.job_id, {}).get("status") == "success" for job in jobs)


def count_successful_jobs(rows: Iterable[dict[str, Any]], jobs: Iterable[PipelineJob]) -> int:
    latest = latest_rows_by_job_id(rows)
    return sum(1 for job in jobs if latest.get(job.job_id, {}).get("status") == "success")


def run_jobs(
    args: argparse.Namespace,
    jobs: list[PipelineJob],
    status_path: Path,
) -> list[dict[str, Any]]:
    stage = jobs[0].stage if jobs else status_path.stem.replace("_status", "")
    if args.dry_run:
        log_step(f"Dry run: planned {len(jobs)} {stage} job(s); status target: {status_path}")
        return [
            {
                **asdict(job),
                "command": " ".join(job.command),
                "status": "planned",
                "gpu": None,
                "returncode": None,
            }
            for job in jobs
        ]

    previous_rows = read_dict_rows(status_path)
    reuse_successes = stage != "final" or bool(
        getattr(args, "final_reuse_existing_successes", False)
    )
    latest_previous = latest_rows_by_job_id(previous_rows)
    successful_job_ids = (
        {job_id for job_id, row in latest_previous.items() if row.get("status") == "success"}
        if reuse_successes
        else set()
    )
    rows: list[dict[str, Any]] = list(previous_rows)
    pending_jobs = [job for job in jobs if job.job_id not in successful_job_ids]
    if not pending_jobs:
        log_step(f"{stage}: all {len(jobs)} job(s) already succeeded")
        return rows
    log_step(
        f"Running {len(pending_jobs)} pending {stage} job(s) "
        f"({len(successful_job_ids)} already complete)"
    )
    if rich_enabled(args):
        return run_jobs_rich(args, jobs, pending_jobs, rows, status_path, stage)

    return run_jobs_plain(args, jobs, pending_jobs, rows, status_path, stage)


def run_jobs_plain(
    args: argparse.Namespace,
    jobs: list[PipelineJob],
    pending_jobs: list[PipelineJob],
    rows: list[dict[str, Any]],
    status_path: Path,
    stage: str,
) -> list[dict[str, Any]]:
    gpu_slots = [(str(gpu), slot) for gpu in args.gpus for slot in range(args.max_slots_per_gpu)]
    initial_row_count = len(rows)
    work_queue: queue.Queue[PipelineJob] = queue.Queue()
    for job in pending_jobs:
        work_queue.put(job)

    lock = threading.Lock()

    def worker(gpu_id: str, slot_index: int) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            row = execute_job(args, job, gpu_id, slot_index)
            with lock:
                rows.append(row)
                write_dict_rows(status_path, rows)
            if args.fail_fast and row["status"] != "success":
                while not work_queue.empty():
                    try:
                        work_queue.get_nowait()
                    except queue.Empty:
                        break
            work_queue.task_done()

    threads = []
    for gpu_id, slot_index in gpu_slots:
        thread = threading.Thread(target=worker, args=(gpu_id, slot_index), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    successes = count_successful_jobs(rows, jobs)
    log_step(f"{stage}: {successes}/{len(jobs)} job(s) successful")
    print_failed_job_tails(
        [row for row in latest_rows_for_jobs(rows, jobs) if row.get("status") != "success"]
    )
    append_cleanup_manifest(
        status_path.parent, cleanup_manifest_rows_from_job_rows(rows[initial_row_count:])
    )
    return rows


def run_jobs_rich(
    args: argparse.Namespace,
    jobs: list[PipelineJob],
    pending_jobs: list[PipelineJob],
    rows: list[dict[str, Any]],
    status_path: Path,
    stage: str,
) -> list[dict[str, Any]]:
    assert _CONSOLE is not None and Live is not None
    gpu_slots = [(str(gpu), slot) for gpu in args.gpus for slot in range(args.max_slots_per_gpu)]
    initial_row_count = len(rows)
    work_queue: queue.Queue[PipelineJob] = queue.Queue()
    for job in pending_jobs:
        work_queue.put(job)

    lock = threading.Lock()
    active_jobs: dict[str, dict[str, Any]] = {}
    stream_states: dict[str, StreamDisplayState] = {}
    trial_counts: dict[str, int] = {}
    hpo_snapshots: dict[str, dict[str, Any]] = {}
    last_hpo_poll = 0.0

    def worker(gpu_id: str, slot_index: int) -> None:
        while True:
            try:
                job = work_queue.get_nowait()
            except queue.Empty:
                return
            gpu_label = job_gpu_label(args, job, gpu_id)
            with lock:
                active_jobs[job.job_id] = {
                    "job": job,
                    "gpu": gpu_label,
                    "slot": slot_index,
                    "started_at": time.time(),
                    "attempt": 1,
                    "status": "running",
                }
            row = execute_job(
                args,
                job,
                gpu_id,
                slot_index,
                stream_states=stream_states if job.stream_output else None,
                stream_lock=lock,
            )
            with lock:
                active_jobs.pop(job.job_id, None)
                rows.append(row)
                write_dict_rows(status_path, rows)
            if args.fail_fast and row["status"] != "success":
                while not work_queue.empty():
                    try:
                        work_queue.get_nowait()
                    except queue.Empty:
                        break
            work_queue.task_done()

    threads = []
    for gpu_id, slot_index in gpu_slots:
        thread = threading.Thread(target=worker, args=(gpu_id, slot_index), daemon=True)
        thread.start()
        threads.append(thread)

    with Live(
        render_job_dashboard(
            args, stage, jobs, pending_jobs, rows, active_jobs, hpo_snapshots, stream_states
        ),
        console=_CONSOLE,
        refresh_per_second=2,
        transient=False,
    ) as live:
        while any(thread.is_alive() for thread in threads):
            now = time.time()
            with lock:
                if stage == "hpo" and now - last_hpo_poll >= HPO_PROGRESS_POLL_SEC:
                    hpo_snapshots = collect_hpo_progress(args, jobs)
                    if args.hpo_trial_log:
                        emit_hpo_trial_events(hpo_snapshots, trial_counts)
                    last_hpo_poll = now
                live.update(
                    render_job_dashboard(
                        args,
                        stage,
                        jobs,
                        pending_jobs,
                        rows,
                        active_jobs,
                        hpo_snapshots,
                        stream_states,
                    )
                )
            time.sleep(JOB_DASHBOARD_REFRESH_SEC)
        for thread in threads:
            thread.join()
        with lock:
            if stage == "hpo":
                hpo_snapshots = collect_hpo_progress(args, jobs)
            live.update(
                render_job_dashboard(
                    args,
                    stage,
                    jobs,
                    pending_jobs,
                    rows,
                    active_jobs,
                    hpo_snapshots,
                    stream_states,
                )
            )

    successes = count_successful_jobs(rows, jobs)
    log_step(f"{stage}: {successes}/{len(jobs)} job(s) successful")
    print_failed_job_tails(
        [row for row in latest_rows_for_jobs(rows, jobs) if row.get("status") != "success"]
    )
    append_cleanup_manifest(
        status_path.parent, cleanup_manifest_rows_from_job_rows(rows[initial_row_count:])
    )
    return rows


def endpoint_ready_for_finals(
    args: argparse.Namespace,
    run_root: Path,
    endpoint: str,
    hpo_jobs: list[PipelineJob],
    hpo_rows: list[dict[str, Any]],
) -> bool:
    latest = latest_rows_by_job_id(hpo_rows)
    endpoint_jobs = [job for job in hpo_jobs if job.endpoint == endpoint]
    return (
        bool(endpoint_jobs)
        and all(latest.get(job.job_id, {}).get("status") == "success" for job in endpoint_jobs)
        and endpoint_hpo_completed(args, run_root, endpoint)
    )


def run_hpo_with_streaming_finals(
    args: argparse.Namespace,
    run_root: Path,
    hpo_jobs: list[PipelineJob],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[PipelineJob], list[dict[str, Any]]]:
    """Run HPO first, but enqueue endpoint final jobs as each endpoint study completes."""

    if args.dry_run:
        final_jobs = build_final_jobs(
            args,
            run_root,
            params_by_endpoint={endpoint: {} for endpoint in args.endpoints},
        )
        hpo_rows = [
            {**asdict(job), "command": " ".join(job.command), "status": "planned"}
            for job in hpo_jobs
        ]
        final_rows = [
            {**asdict(job), "command": " ".join(job.command), "status": "planned"}
            for job in final_jobs
        ]
        write_dict_rows(run_root / "hpo_status.csv", hpo_rows)
        write_dict_rows(run_root / "planned_final_jobs.csv", job_plan_rows(final_jobs))
        write_dict_rows(run_root / "final_status.csv", final_rows)
        return (hpo_rows, [], final_jobs, final_rows)

    hpo_status_path = run_root / "hpo_status.csv"
    final_status_path = run_root / "final_status.csv"
    materialization_status_path = run_root / "materialization_final_status.csv"
    planned_final_path = run_root / "planned_final_jobs.csv"

    hpo_rows = read_dict_rows(hpo_status_path)
    final_rows = read_dict_rows(final_status_path)
    materialization_rows = read_dict_rows(materialization_status_path)
    final_jobs: list[PipelineJob] = []
    planned_final_by_id: dict[str, PipelineJob] = {}
    latest_hpo = latest_rows_by_job_id(hpo_rows)
    latest_final = latest_rows_by_job_id(final_rows)
    hpo_success_ids = {
        job_id for job_id, row in latest_hpo.items() if row.get("status") == "success"
    }
    final_success_ids = (
        {job_id for job_id, row in latest_final.items() if row.get("status") == "success"}
        if bool(getattr(args, "final_reuse_existing_successes", False))
        else set()
    )

    hpo_queue: queue.Queue[PipelineJob] = queue.Queue()
    final_queue: queue.Queue[PipelineJob] = queue.Queue()
    for job in hpo_jobs:
        if job.job_id not in hpo_success_ids:
            hpo_queue.put(job)

    endpoints = list(args.endpoints)
    endpoints_final_enqueued: set[str] = set()
    endpoints_final_failed: set[str] = set()
    endpoints_materializing: set[str] = set()
    hpo_pending_remaining = sum(1 for job in hpo_jobs if job.job_id not in hpo_success_ids)
    hpo_done = hpo_pending_remaining == 0
    stop_requested = False
    lock = threading.Lock()
    active_jobs: dict[str, dict[str, Any]] = {}
    stream_states: dict[str, StreamDisplayState] = {}
    trial_counts: dict[str, int] = {}
    hpo_snapshots: dict[str, dict[str, Any]] = {}
    last_hpo_poll = 0.0

    def enqueue_finals_for_endpoint(endpoint: str) -> None:
        nonlocal materialization_rows
        with lock:
            if endpoint in endpoints_final_enqueued or endpoint in endpoints_materializing:
                return
            if not endpoint_ready_for_finals(args, run_root, endpoint, hpo_jobs, hpo_rows):
                return
            endpoints_materializing.add(endpoint)

        log_step(f"Endpoint {endpoint}: HPO complete; preparing final materialized caches")
        rows = run_materialization_stage(
            args,
            _TASK.materialization_specs_for_final_endpoint(args, run_root, endpoint),
            run_root,
            materialization_status_path,
            quiet=True,
        )
        ok = all(row["status"] in {"hit", "built", "skipped"} for row in rows)
        jobs: list[PipelineJob] = []
        best_rows: list[dict[str, Any]] = []
        if ok:
            jobs, best_rows = build_final_jobs_for_endpoint(args, run_root, endpoint)

        with lock:
            materialization_rows.extend(rows)
            write_dict_rows(materialization_status_path, materialization_rows)
            endpoints_materializing.discard(endpoint)
            if not ok:
                endpoints_final_failed.add(endpoint)
                return
            upsert_best_params_csv(run_root, best_rows)
            new_jobs = [
                job
                for job in jobs
                if job.job_id not in final_success_ids and job.job_id not in planned_final_by_id
            ]
            for job in jobs:
                planned_final_by_id[job.job_id] = job
            final_jobs[:] = list(planned_final_by_id.values())
            write_dict_rows(planned_final_path, job_plan_rows(final_jobs))
            for job in new_jobs:
                final_queue.put(job)
            endpoints_final_enqueued.add(endpoint)
            log_step(f"Endpoint {endpoint}: queued {len(new_jobs)} final job(s)")

    for endpoint in endpoints:
        if endpoint_ready_for_finals(args, run_root, endpoint, hpo_jobs, hpo_rows):
            enqueue_finals_for_endpoint(endpoint)

    def mark_unrecoverable_endpoints_if_hpo_done() -> None:
        with lock:
            if not hpo_done:
                return
            for endpoint in endpoints:
                if endpoint in endpoints_final_enqueued or endpoint in endpoints_final_failed:
                    continue
                if not endpoint_ready_for_finals(args, run_root, endpoint, hpo_jobs, hpo_rows):
                    endpoints_final_failed.add(endpoint)
                    materialization_rows.append(
                        {
                            "endpoint": endpoint,
                            "split": hpo_split(args, endpoint),
                            "threshold": "",
                            "stage": "materialize",
                            "status": "failed",
                            "reason": "HPO workers finished but Optuna study did not reach completed-trial target",
                        }
                    )
                    write_dict_rows(materialization_status_path, materialization_rows)
                    log_step(
                        f"Endpoint {endpoint}: HPO workers finished but study is incomplete; finals not queued"
                    )

    mark_unrecoverable_endpoints_if_hpo_done()

    def next_job() -> Optional[PipelineJob]:
        try:
            return hpo_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            return final_queue.get_nowait()
        except queue.Empty:
            return None

    def worker(gpu_id: str, slot_index: int) -> None:
        nonlocal hpo_done, hpo_pending_remaining, stop_requested
        while True:
            with lock:
                if stop_requested:
                    return
                done = (
                    hpo_done
                    and final_queue.empty()
                    and hpo_queue.empty()
                    and len(endpoints_final_enqueued | endpoints_final_failed) >= len(endpoints)
                )
                if done:
                    return
            job = next_job()
            if job is None:
                time.sleep(1.0)
                continue
            gpu_label = job_gpu_label(args, job, str(gpu_id))
            with lock:
                active_jobs[job.job_id] = {
                    "job": job,
                    "gpu": gpu_label,
                    "slot": slot_index,
                    "started_at": time.time(),
                    "attempt": 1,
                    "status": "running",
                }
            try:
                row = execute_job(
                    args,
                    job,
                    str(gpu_id),
                    slot_index,
                    stream_states=stream_states if job.stream_output else None,
                    stream_lock=lock,
                )
            except Exception as exc:
                row = {
                    "job_id": job.job_id,
                    "stage": job.stage,
                    "endpoint": job.endpoint,
                    "split": job.split,
                    "threshold": job.threshold,
                    "seed": job.seed,
                    "gpu": str(gpu_id),
                    "slot": slot_index,
                    "status": "failed",
                    "returncode": "exception",
                    "seconds": 0,
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.output_dir / "pipeline_stdout_stderr.log"),
                    "command": " ".join(job.command),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                stop_requested = True
            if job.stage == "hpo":
                with lock:
                    active_jobs.pop(job.job_id, None)
                    hpo_rows.append(row)
                    write_dict_rows(hpo_status_path, hpo_rows)
                    hpo_pending_remaining = max(0, hpo_pending_remaining - 1)
                    hpo_done = hpo_pending_remaining == 0
                    if args.fail_fast and row["status"] != "success":
                        stop_requested = True
                if row["status"] == "success":
                    try:
                        enqueue_finals_for_endpoint(job.endpoint)
                    except Exception as exc:
                        with lock:
                            endpoints_final_failed.add(job.endpoint)
                            materialization_rows.append(
                                {
                                    "endpoint": job.endpoint,
                                    "split": hpo_split(args, job.endpoint),
                                    "threshold": job.threshold or "",
                                    "stage": "materialize",
                                    "status": "failed",
                                    "reason": f"{exc.__class__.__name__}: {exc}",
                                }
                            )
                            write_dict_rows(materialization_status_path, materialization_rows)
                mark_unrecoverable_endpoints_if_hpo_done()
            else:
                with lock:
                    active_jobs.pop(job.job_id, None)
                    final_rows.append(row)
                    write_dict_rows(final_status_path, final_rows)
                    if args.fail_fast and row["status"] != "success":
                        stop_requested = True
            if job.stage == "hpo":
                hpo_queue.task_done()
            else:
                final_queue.task_done()

    log_step(
        f"Running {hpo_pending_remaining} pending HPO job(s); finals will stream per completed endpoint"
    )
    gpu_slots = [(str(gpu), slot) for gpu in args.gpus for slot in range(args.max_slots_per_gpu)]
    threads = []
    for gpu_id, slot_index in gpu_slots:
        thread = threading.Thread(target=worker, args=(gpu_id, slot_index), daemon=True)
        thread.start()
        threads.append(thread)

    if rich_enabled(args):
        assert _CONSOLE is not None and Live is not None
        with Live(
            render_streaming_pipeline_dashboard(
                args,
                hpo_jobs,
                hpo_rows,
                final_jobs,
                final_rows,
                active_jobs,
                hpo_snapshots,
                stream_states,
            ),
            console=_CONSOLE,
            refresh_per_second=2,
            transient=False,
        ) as live:
            while any(thread.is_alive() for thread in threads):
                now = time.time()
                with lock:
                    if now - last_hpo_poll >= HPO_PROGRESS_POLL_SEC:
                        hpo_snapshots = collect_hpo_progress(args, hpo_jobs)
                        if args.hpo_trial_log:
                            emit_hpo_trial_events(hpo_snapshots, trial_counts)
                        last_hpo_poll = now
                    live.update(
                        render_streaming_pipeline_dashboard(
                            args,
                            hpo_jobs,
                            hpo_rows,
                            final_jobs,
                            final_rows,
                            active_jobs,
                            hpo_snapshots,
                            stream_states,
                        )
                    )
                time.sleep(JOB_DASHBOARD_REFRESH_SEC)
            for thread in threads:
                thread.join()
            with lock:
                hpo_snapshots = collect_hpo_progress(args, hpo_jobs)
                live.update(
                    render_streaming_pipeline_dashboard(
                        args,
                        hpo_jobs,
                        hpo_rows,
                        final_jobs,
                        final_rows,
                        active_jobs,
                        hpo_snapshots,
                        stream_states,
                    )
                )
    else:
        while any(thread.is_alive() for thread in threads):
            time.sleep(JOB_DASHBOARD_REFRESH_SEC)
        for thread in threads:
            thread.join()

    append_cleanup_manifest(
        run_root, cleanup_manifest_rows_from_job_rows([*hpo_rows, *final_rows])
    )
    print_failed_job_tails(
        [row for row in latest_rows_for_jobs(hpo_rows, hpo_jobs) if row.get("status") != "success"]
    )
    print_failed_job_tails(
        [
            row
            for row in latest_rows_for_jobs(final_rows, final_jobs)
            if row.get("status") != "success"
        ]
    )
    return hpo_rows, materialization_rows, final_jobs, final_rows


def job_gpu_label(args: argparse.Namespace, job: PipelineJob, gpu_id: str) -> str:
    if job.stage == "embed" and job.modality == "molr":
        return ",".join(str(gpu) for gpu in (args.gpus or [])) or "cpu"
    if job.requires_gpu:
        return str(gpu_id)
    return "cpu"


def collect_hpo_progress(
    args: argparse.Namespace, jobs: list[PipelineJob]
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for endpoint in sorted({job.endpoint for job in jobs if job.endpoint}):
        study_name = _TASK.study_name_for_endpoint(args, endpoint)
        storage = optuna_storage_url(run_root_for(args), study_name)
        snapshot = {
            "endpoint": endpoint,
            "completed": 0,
            "running": 0,
            "failed": 0,
            "waiting": 0,
            "target": args.n_trials,
            "best_value": "",
            "last_trial": "",
            "error": "",
        }
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            trials = study.get_trials(deepcopy=False)
            snapshot["completed"] = sum(
                1 for trial in trials if trial.state == TrialState.COMPLETE
            )
            snapshot["running"] = sum(1 for trial in trials if trial.state == TrialState.RUNNING)
            snapshot["failed"] = sum(1 for trial in trials if trial.state == TrialState.FAIL)
            snapshot["waiting"] = sum(1 for trial in trials if trial.state == TrialState.WAITING)
            completed_trials = [
                trial
                for trial in trials
                if trial.state == TrialState.COMPLETE and trial.value is not None
            ]
            if completed_trials:
                best = max(completed_trials, key=lambda trial: float(trial.value))
                latest = max(completed_trials, key=lambda trial: trial.number)
                snapshot["best_value"] = f"{float(best.value):.5g}"
                snapshot["last_trial"] = f"#{latest.number}={float(latest.value):.5g}"
        except Exception as exc:
            snapshot["error"] = exc.__class__.__name__
        snapshots[endpoint] = snapshot
    return snapshots


def emit_hpo_trial_events(
    snapshots: dict[str, dict[str, Any]], trial_counts: dict[str, int]
) -> None:
    metric_name = optimized_metric(_ACTIVE_ARGS) if _ACTIVE_ARGS is not None else "val/r2"
    for endpoint, snapshot in sorted(snapshots.items()):
        completed = int(snapshot.get("completed") or 0)
        previous = trial_counts.get(endpoint, 0)
        if completed > previous:
            log_step(
                f"HPO {endpoint}: {completed}/{snapshot.get('target')} completed, "
                f"study best {metric_name}={snapshot.get('best_value') or 'n/a'}, "
                f"study last={snapshot.get('last_trial') or 'n/a'}"
            )
        trial_counts[endpoint] = completed


def render_job_dashboard(
    args: argparse.Namespace,
    stage: str,
    jobs: list[PipelineJob],
    pending_jobs: list[PipelineJob],
    rows: list[dict[str, Any]],
    active_jobs: dict[str, dict[str, Any]],
    hpo_snapshots: dict[str, dict[str, Any]],
    stream_states: Optional[dict[str, StreamDisplayState]] = None,
) -> Any:
    assert Group is not None and Panel is not None and Table is not None
    now = time.time()
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = sum(1 for row in rows if row.get("status") not in {"success", "planned"})
    completed = len(rows)
    total = len(jobs)
    title = f"{stage.upper()} progress: {completed}/{total} complete, {success} success, {failed} failed, {len(active_jobs)} running"
    metric_name = optimized_metric(args)

    if stage == "hpo":
        endpoint_table = Table(title="HPO endpoint progress")
        for column in (
            "endpoint",
            "gpu",
            "worker",
            "active trial",
            "epoch",
            "batch",
            f"current {metric_name}",
            f"study best {metric_name}",
            "trials",
            "run",
            "fail",
            "study last",
            "elapsed",
            "log",
        ):
            endpoint_table.add_column(column)
        latest_by_job_id = latest_rows_by_job_id(rows)
        for job in jobs:
            active = active_jobs.get(job.job_id)
            row = latest_by_job_id.get(job.job_id)
            snapshot = hpo_snapshots.get(job.endpoint, {})
            progress_state = load_progress_state_for_job(job) if active else {}
            elapsed = (
                elapsed_text(now - active["started_at"])
                if active
                else elapsed_text(float(row.get("seconds", 0) or 0))
                if row
                else ""
            )
            endpoint_table.add_row(
                job.endpoint,
                str(active.get("gpu", "")) if active else str(row.get("gpu", "")) if row else "",
                "running" if active else str(row.get("status", "queued")) if row else "queued",
                progress_state.get("hydra_job_num", "") if active else "",
                epoch_progress_text(progress_state),
                batch_progress_text(progress_state),
                metric_text(current_progress_metric(progress_state)),
                metric_text(best_progress_metric(progress_state) or snapshot.get("best_value")),
                f"{snapshot.get('completed', 0)}/{snapshot.get('target', args.n_trials)}",
                str(snapshot.get("running", 0)),
                str(snapshot.get("failed", 0)),
                str(snapshot.get("last_trial", "")),
                elapsed,
                compact_path(job.output_dir / "pipeline_stdout_stderr.log"),
            )
        return wrap_stream_panel(stage, stream_states, Panel(title), endpoint_table)

    active_table = Table(title=f"{stage} active jobs")
    for column in (
        "gpu",
        "slot",
        "job",
        "endpoint",
        "split",
        "threshold",
        "seed",
        "epoch",
        "batch",
        f"current {metric_name}",
        f"best {metric_name}",
        "elapsed",
        "attempt",
    ):
        active_table.add_column(column)
    if active_jobs:
        for active in sorted(
            active_jobs.values(), key=lambda item: (str(item["gpu"]), int(item["slot"]))
        ):
            job = active["job"]
            progress_state = load_progress_state_for_job(job)
            active_table.add_row(
                str(active["gpu"]),
                str(active["slot"]),
                job.job_id,
                job.endpoint,
                job.split or "",
                job.threshold or "",
                "" if job.seed is None else str(job.seed),
                epoch_progress_text(progress_state),
                batch_progress_text(progress_state),
                metric_text(current_progress_metric(progress_state)),
                metric_text(best_progress_metric(progress_state)),
                elapsed_text(now - active["started_at"]),
                str(active.get("attempt", 1)),
            )
    else:
        active_table.add_row("", "", "no active jobs", "", "", "", "", "", "", "", "", "", "")

    if stage == "final":
        return wrap_stream_panel(
            stage,
            stream_states,
            Panel(title),
            active_table,
            render_final_seed_summary(jobs, rows, active_jobs),
        )
    return wrap_stream_panel(stage, stream_states, Panel(title), active_table)


def render_streaming_pipeline_dashboard(
    args: argparse.Namespace,
    hpo_jobs: list[PipelineJob],
    hpo_rows: list[dict[str, Any]],
    final_jobs: list[PipelineJob],
    final_rows: list[dict[str, Any]],
    active_jobs: dict[str, dict[str, Any]],
    hpo_snapshots: dict[str, dict[str, Any]],
    stream_states: Optional[dict[str, StreamDisplayState]] = None,
) -> Any:
    assert Group is not None and Panel is not None
    hpo_active = {
        job_id: active for job_id, active in active_jobs.items() if active["job"].stage == "hpo"
    }
    final_active = {
        job_id: active for job_id, active in active_jobs.items() if active["job"].stage == "final"
    }
    hpo_dashboard = render_job_dashboard(
        args,
        "hpo",
        hpo_jobs,
        hpo_jobs,
        hpo_rows,
        hpo_active,
        hpo_snapshots,
        None,
    )
    final_dashboard = render_job_dashboard(
        args,
        "final",
        final_jobs,
        final_jobs,
        final_rows,
        final_active,
        hpo_snapshots,
        None,
    )
    hpo_success = count_successful_jobs(hpo_rows, hpo_jobs)
    final_success = count_successful_jobs(final_rows, final_jobs)
    title = (
        "Endpoint-streaming pipeline: "
        f"HPO {hpo_success}/{len(hpo_jobs)} success, "
        f"final {final_success}/{len(final_jobs)} success, "
        f"{len(hpo_active)} HPO running, {len(final_active)} final running"
    )
    return wrap_stream_panel(
        "pipeline",
        stream_states,
        Panel(title),
        hpo_dashboard,
        final_dashboard,
    )


def wrap_stream_panel(
    stage: str,
    stream_states: Optional[dict[str, StreamDisplayState]],
    *renderables: Any,
) -> Any:
    if not stream_states:
        return Group(*renderables)
    output_lines: list[str] = []
    for job_id, state in sorted(stream_states.items()):
        lines = state.render_lines()
        if not lines:
            continue
        output_lines.extend(lines)
    if not output_lines:
        return Group(*renderables)
    text = "\n".join(output_lines[-STREAM_DISPLAY_LINES:])
    return Group(
        Panel(text or "waiting for subprocess output", title=f"{stage} subprocess output"),
        *renderables,
    )


def load_progress_state_for_job(job: PipelineJob) -> dict[str, Any]:
    if job.stage == "hpo":
        candidates = list((job.output_dir / "trials").glob("*/progress_state.json"))
    else:
        candidates = [job.output_dir / "progress_state.json"]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {}
    path = max(existing, key=lambda item: item.stat().st_mtime)
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    state["_progress_path"] = str(path)
    if not state.get("hydra_job_num") and path.parent.name != job.output_dir.name:
        state["hydra_job_num"] = path.parent.name
    return state


def epoch_progress_text(state: dict[str, Any]) -> str:
    if not state:
        return ""
    current = state.get("current_epoch")
    max_epochs = state.get("max_epochs")
    if current is None:
        return ""
    try:
        current_int = int(current) + 1
        max_int = int(max_epochs) if max_epochs not in (None, "", -1) else 0
    except (TypeError, ValueError):
        return str(current)
    if max_int > 0:
        return f"{current_int}/{max_int} {mini_bar(current_int, max_int)}"
    return str(current_int)


def batch_progress_text(state: dict[str, Any]) -> str:
    if not state:
        return ""
    batch_idx = state.get("train_batch_idx")
    total = state.get("train_num_batches")
    if batch_idx is None or total in (None, "", 0):
        return ""
    try:
        done = min(int(batch_idx) + 1, int(total))
        total_int = int(total)
    except (TypeError, ValueError):
        return ""
    return f"{done}/{total_int} {mini_bar(done, total_int)}"


def mini_bar(done: int, total: int, width: int = 10) -> str:
    if total <= 0:
        return ""
    filled = max(0, min(width, int(round(width * done / total))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def metric_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def render_final_seed_summary(
    jobs: list[PipelineJob],
    rows: list[dict[str, Any]],
    active_jobs: dict[str, dict[str, Any]],
) -> Any:
    assert Table is not None
    row_by_id = {row.get("job_id"): row for row in rows}
    active_ids = set(active_jobs)
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for job in jobs:
        key = (job.endpoint, "" if job.seed is None else str(job.seed))
        bucket = buckets.setdefault(key, {"total": 0, "success": 0, "failed": 0, "running": 0})
        bucket["total"] += 1
        if job.job_id in active_ids:
            bucket["running"] += 1
            continue
        row = row_by_id.get(job.job_id)
        if row and row.get("status") == "success":
            bucket["success"] += 1
        elif row and row.get("status") not in {"planned", None, ""}:
            bucket["failed"] += 1
    table = Table(title="Final seed summary")
    for column in ("endpoint", "seed", "done/total", "running", "failed"):
        table.add_column(column)
    for (endpoint, seed), bucket in sorted(buckets.items()):
        table.add_row(
            endpoint,
            seed,
            f"{bucket['success']}/{bucket['total']}",
            str(bucket["running"]),
            str(bucket["failed"]),
        )
    return table


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def compact_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_failed_job_tails(rows: list[dict[str, Any]]) -> None:
    failed_rows = [row for row in rows if row.get("status") not in {"success", "planned"}]
    for row in failed_rows:
        log_path = Path(str(row.get("log_path", "")))
        log_step(
            f"{row.get('stage')} job failed: {row.get('job_id')} "
            f"returncode={row.get('returncode')} log={log_path}"
        )
        if log_path.exists():
            print_log_tail(log_path, LOG_TAIL_LINES)


def print_log_tail(log_path: Path, line_count: int) -> None:
    lines = log_path.read_text(errors="replace").splitlines()[-line_count:]
    text = "\n".join(lines) if lines else "(empty log)"
    if rich_enabled(_ACTIVE_ARGS):
        assert _CONSOLE is not None and Panel is not None
        _CONSOLE.print(Panel(text, title=f"Last {line_count} lines: {compact_path(log_path)}"))
        return
    print(f"\n--- Last {line_count} lines: {log_path} ---\n{text}\n", flush=True)


def execute_job(
    args: argparse.Namespace,
    job: PipelineJob,
    gpu_id: str,
    slot_index: int,
    stream_states: Optional[dict[str, StreamDisplayState]] = None,
    stream_lock: Optional[threading.Lock] = None,
) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "pipeline_stdout_stderr.log"
    env = os.environ.copy()
    gpu_label = job_gpu_label(args, job, gpu_id)
    if job.stage == "embed" and job.modality == "molr":
        visible_gpus = [str(gpu) for gpu in (args.gpus or [])]
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpus)
        env["MOLR_CUDA_VISIBLE_DEVICES"] = gpu_label
    elif job.requires_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if args.wandb_offline:
        env["WANDB_MODE"] = "offline"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["ORCHESTRATOR_STAGE"] = job.stage
    env["ORCHESTRATOR_ENDPOINT"] = job.endpoint or ""
    env["ORCHESTRATOR_SPLIT"] = job.split or ""
    env["ORCHESTRATOR_THRESHOLD"] = job.threshold or ""
    env["ORCHESTRATOR_SEED"] = "" if job.seed is None else str(job.seed)
    env["ORCHESTRATOR_JOB_ID"] = job.job_id
    env["ORCHESTRATOR_PARENT_PID"] = str(os.getpid())
    if job.stage == "hpo":
        for item in job.command:
            if item.startswith("hydra.sweeper.study_name="):
                env["ORCHESTRATOR_OPTUNA_STUDY_NAME"] = item.split("=", 1)[1]
            elif item.startswith("hydra.sweeper.storage="):
                env["ORCHESTRATOR_OPTUNA_STORAGE"] = item.split("=", 1)[1]

    attempts = args.retry_failed + 1
    start = time.time()
    last_returncode = None
    status = "failed"
    for attempt in range(1, attempts + 1):
        try:
            with open(log_path, "ab") as log_handle:
                log_handle.write(
                    f"\n\n### attempt {attempt} gpu={gpu_label} slot={slot_index}\n".encode()
                )
                if job.stream_output or args.stream_all_jobs:
                    last_returncode = run_streaming_subprocess(
                        args, job, env, log_handle, stream_states, stream_lock
                    )
                else:
                    last_returncode = run_logged_subprocess(
                        args,
                        job.command,
                        env,
                        log_handle,
                    )
            # A final job must also leave its metrics.json (read by aggregate()); otherwise it failed.
            if last_returncode == 0 and (
                job.stage != "final" or (job.output_dir / "metrics.json").exists()
            ):
                status = "success"
                break
        except subprocess.TimeoutExpired:
            last_returncode = "timeout"
            status = "timeout"
    cleanup_rows = cleanup_job_output(args, job)
    cleanup_bytes = sum(int(row.get("bytes", 0) or 0) for row in cleanup_rows)
    return {
        "job_id": job.job_id,
        "stage": job.stage,
        "endpoint": job.endpoint,
        "split": job.split,
        "threshold": job.threshold,
        "seed": job.seed,
        "gpu": gpu_label,
        "slot": slot_index,
        "status": status,
        "returncode": last_returncode,
        "seconds": round(time.time() - start, 3),
        "output_dir": str(job.output_dir),
        "log_path": str(log_path),
        "command": " ".join(job.command),
        "cleanup_bytes": cleanup_bytes,
        "cleanup_gib": f"{bytes_to_gib(cleanup_bytes):.4f}",
        "cleanup_manifest": json.dumps(cleanup_rows, sort_keys=True) if cleanup_rows else "",
    }


def run_logged_subprocess(
    args: argparse.Namespace,
    command: list[str],
    env: dict[str, str],
    log_handle,
) -> int:
    proc = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    register_process(proc)
    started_at = time.monotonic()
    timed_out = False
    try:
        while True:
            returncode = proc.poll()
            if returncode is not None:
                return int(returncode)
            if args.job_timeout_sec and time.monotonic() - started_at > args.job_timeout_sec:
                timed_out = True
                terminate_process_tree(proc)
                break
            time.sleep(0.2)
    except BaseException:
        terminate_process_tree(proc)
        raise
    finally:
        unregister_process(proc)
    if timed_out:
        raise subprocess.TimeoutExpired(command, args.job_timeout_sec)
    return int(proc.wait())


def run_streaming_subprocess(
    args: argparse.Namespace,
    job: PipelineJob,
    env: dict[str, str],
    log_handle,
    stream_states: Optional[dict[str, StreamDisplayState]] = None,
    stream_lock: Optional[threading.Lock] = None,
) -> int:
    prefix = f"[{job.stage}/{job.modality or job.job_id}]"
    use_rich_capture = rich_enabled(args) and stream_states is not None
    proc = subprocess.Popen(
        job.command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    register_process(proc)
    assert proc.stdout is not None
    timed_out = False
    started_at = time.monotonic()
    at_line_start = True
    try:
        while True:
            if args.job_timeout_sec and time.monotonic() - started_at > args.job_timeout_sec:
                timed_out = True
                terminate_process_tree(proc)
                break
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            chunk = proc.stdout.read(1)
            if chunk == "":
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            if at_line_start:
                prefixed = f"{prefix} "
                if use_rich_capture:
                    append_stream_display(stream_states, job.job_id, prefixed, stream_lock)
                else:
                    print(prefixed, end="", flush=True)
                log_handle.write(prefixed.encode("utf-8", errors="replace"))
                at_line_start = False
            if use_rich_capture:
                append_stream_display(stream_states, job.job_id, chunk, stream_lock)
            else:
                print(chunk, end="", flush=True)
            log_handle.write(chunk.encode("utf-8", errors="replace"))
            log_handle.flush()
            if chunk in {"\n", "\r"}:
                at_line_start = True
    except BaseException:
        terminate_process_tree(proc)
        raise
    finally:
        unregister_process(proc)
    returncode = proc.wait()
    if timed_out:
        raise subprocess.TimeoutExpired(job.command, args.job_timeout_sec)
    return int(returncode)


def aggregate(
    output_root: Path,
    run_root: Path,
    endpoints: Iterable[str],
    seeds: Optional[Iterable[int]] = None,
) -> None:
    metric_rows = []
    # Keep configured seeds only.
    allowed = None if seeds is None else {str(seed) for seed in seeds}
    for endpoint, split, threshold, seed, metrics_path in final_job_metrics(
        output_root, endpoints
    ):
        if allowed is not None and seed not in allowed:
            continue
        with open(metrics_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metric_rows.append(
            {
                **payload,
                "endpoint": endpoint,
                "split": split,
                "threshold": threshold,
                "seed": seed,
                "metrics_path": str(metrics_path),
            }
        )
    write_dict_rows(run_root / "all_tvt_metrics.csv", metric_rows)
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        metric_cols = [
            col for col in df.columns if "/" in col and pd.api.types.is_numeric_dtype(df[col])
        ]
        group_cols = ["endpoint", "split", "threshold"]
        agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
        agg.columns = [
            "_".join([str(part) for part in col if part != ""]).strip("_") for col in agg.columns
        ]
        agg.to_csv(run_root / "aggregate_tvt_metrics.csv", index=False)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_dict_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def job_plan_rows(jobs: Iterable[PipelineJob]) -> list[dict[str, Any]]:
    return [{**asdict(job), "command": " ".join(job.command)} for job in jobs]


def to_plain_config(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_config(item) for item in value]
    return value


def resolved_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return to_plain_config(
        {
            "run": {
                "pipeline_id": args.pipeline_id,
                "output_root": args.output_root,
                "data_root": args.data_root,
                "embeddings_root": args.embeddings_root,
                "endpoints": args.endpoints,
                "splits": args.splits,
                "seeds": args.seeds,
                "modalities": enabled_modalities(args),
                "label": getattr(args, "run_label", "mlp"),
                "study_prefix": getattr(args, "study_prefix", "baseline_mlp"),
                "tags": run_tags(args),
            },
            "resources": {
                "gpus": args.gpus,
                "max_slots_per_gpu": args.max_slots_per_gpu,
                "hpo_workers_per_endpoint": args.hpo_workers_per_endpoint,
                "retry_failed": args.retry_failed,
                "job_timeout_sec": args.job_timeout_sec,
            },
            "stages": {
                "embeddings": {
                    "enabled": not args.skip_embeddings,
                    "refresh_superset": args.refresh_embedding_superset,
                },
                "materialization": {
                    "enabled": not args.skip_materialization,
                    "refresh": args.refresh_materialized_features,
                    "feature_dtype": args.materialized_feature_dtype,
                    "block_rows": args.materialized_block_rows,
                    "validation": args.materialized_validation,
                    "cache_root": args.materialized_cache_root,
                },
                "hpo": {
                    "enabled": not getattr(args, "skip_hpo", False),
                    "split": (
                        hpo_split(args)
                        if _TASK.hpo_split_for_endpoint is None
                        else {endpoint: hpo_split(args, endpoint) for endpoint in args.endpoints}
                    ),
                    "n_trials": args.n_trials,
                    "max_epochs": args.max_epochs,
                    "sampler_seed_base": args.hpo_sampler_seed_base,
                    "sampler_seed_strategy": args.hpo_sampler_seed_strategy,
                    "checkpoints": args.hpo_checkpoints,
                },
                "final": {
                    "enabled": not getattr(args, "skip_final", False),
                    "max_epochs": args.final_max_epochs,
                    "params_source": args.final_params_source,
                    "best_params_path": args.final_best_params_path,
                    "reuse_existing_successes": args.final_reuse_existing_successes,
                    "checkpoints": args.final_checkpoints,
                },
                "cleanup": {
                    "enabled": cleanup_enabled(args),
                    "dry_run": cleanup_dry_run(args),
                },
            },
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "offline": args.wandb_offline,
                "skip_check": args.skip_wandb_check,
            },
            "ui": {
                "progress_backend": args.progress_backend,
                "hpo_trial_log": args.hpo_trial_log,
                "stream_all_jobs": args.stream_all_jobs,
                "no_progress": args.no_progress,
            },
            "storage": storage_profile(args),
            "hydra": hydra_profile(args),
        }
    )


def write_resolved_run_config(args: argparse.Namespace, run_root: Path) -> None:
    payload = resolved_run_config(args)
    (run_root / "run_config_resolved.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(payload)), encoding="utf-8"
    )


REGRESSION_TASK = PipelineTask(
    modalities=MODALITIES,
    modality_key_columns=MODALITY_KEY_COLUMNS,
    modality_input_overrides=MODALITY_INPUT_OVERRIDES,
    gpu_embedders=GPU_EMBEDDERS,
    embedding_dataset_config=EMBEDDING_DATASET_CONFIG,
    pipeline_arg_defaults=pipeline_arg_defaults,
    cache_roots=cache_roots,
    resolve_split_dir=resolve_split_dir,
    split_threshold=split_threshold,
    discover_split_specs=discover_split_specs,
    discover_final_jobs=discover_final_jobs,
    study_name_for_endpoint=study_name_for_endpoint,
    build_base_overrides=build_base_overrides,
    build_hpo_jobs=build_hpo_jobs,
    materialization_specs_for_hpo=materialization_specs_for_hpo,
    materialization_specs_for_final=materialization_specs_for_final,
    materialization_specs_for_final_endpoint=materialization_specs_for_final_endpoint,
    materialization_status_row=materialization_status_row,
    required_columns_for_modalities=required_columns_for_modalities,
    preflight=preflight,
)
_TASK = REGRESSION_TASK


# Single entry point for all tasks. Regression is the default; other scripts pass
# their own PipelineTask, which is stored in _TASK and read by the hook call sites.
def main(task: PipelineTask = REGRESSION_TASK) -> None:
    global _ACTIVE_ARGS, _SHUTDOWN_CLEANUP_ARMED, _TASK
    _TASK = task
    args = parse_args()
    _ACTIVE_ARGS = args
    args.pipeline_id = args.pipeline_id or pipeline_id()
    if args.gpus is None:
        args.gpus = discover_gpus()
    if args.materialized_cache_root is None:
        args.materialized_cache_root = args.output_root / "_cache" / "materialized_features"
    if args.print_resolved_run_config:
        print(OmegaConf.to_yaml(OmegaConf.create(resolved_run_config(args))), end="")
        return
    run_root = run_root_for(args)
    run_root.mkdir(parents=True, exist_ok=True)
    acquire_run_lock(run_root)
    _SHUTDOWN_CLEANUP_ARMED = True
    atexit.register(cleanup_interrupted_run, "orchestrator process exiting")
    if not args.dry_run:
        cleanup_orphaned_run_workers("startup cleanup")
    write_resolved_run_config(args, run_root)
    log_step(f"Pipeline id: {args.pipeline_id}")
    log_step(f"Run root: {run_root}")
    log_step(f"GPUs: {', '.join(str(gpu) for gpu in args.gpus)}; dry_run={args.dry_run}")
    before_rows = write_disk_report(args, run_root, "before")
    render_disk_summary(args, before_rows, title="Disk usage before")

    if args.cleanup_existing_run:
        log_step("Cleanup existing run requested")
        cleanup_rows = cleanup_existing_run(args, run_root)
        cleanup_summary(args, cleanup_rows, title="Existing-run cleanup")
        after_rows = write_disk_report(args, run_root, "after")
        render_disk_summary(args, after_rows, title="Disk usage after")
        return

    if not args.skip_preflight:
        _TASK.preflight(args, run_root)
    if args.preflight_only:
        print(f"Preflight passed: {run_root}")
        return

    embedding_jobs: list[PipelineJob] = []
    embedding_rows: list[dict[str, Any]] = []
    if not args.skip_embeddings:
        log_step("Stage 0: building embedding input supersets")
        input_paths, keys_by_column = build_embedding_superset(args, run_root)
        log_step("Stage 0: planning embedding jobs")
        embedding_jobs = build_embedding_jobs(args, run_root, input_paths, keys_by_column)
        write_dict_rows(run_root / "planned_embedding_jobs.csv", job_plan_rows(embedding_jobs))
        log_step(f"Stage 0: {len(embedding_jobs)} embedding job(s) need compute")
        if args.dry_run:
            embedding_rows = run_jobs(args, embedding_jobs, run_root / "embedding_status.csv")
        else:
            non_molr_jobs = [job for job in embedding_jobs if job.modality != "molr"]
            molr_jobs = [job for job in embedding_jobs if job.modality == "molr"]
            embedding_rows = []
            if non_molr_jobs:
                embedding_rows = run_jobs(args, non_molr_jobs, run_root / "embedding_status.csv")
            if molr_jobs:
                log_step("Stage 0: running MolR embedding with all allowed GPUs")
                embedding_rows = run_jobs(args, molr_jobs, run_root / "embedding_status.csv")
        if not args.dry_run and not jobs_succeeded(embedding_rows, embedding_jobs):
            raise RuntimeError(
                f"One or more embedding jobs failed. See {run_root / 'embedding_status.csv'}"
            )
    else:
        log_step("Stage 0: embedding generation skipped")

    hpo_materialization_rows: list[dict[str, Any]] = []
    hpo_jobs: list[PipelineJob] = []
    hpo_rows: list[dict[str, Any]] = []
    final_materialization_rows: list[dict[str, Any]] = []
    final_jobs: list[PipelineJob] = []
    final_rows: list[dict[str, Any]] = []
    if not getattr(args, "skip_hpo", False):
        log_step("Stage 0.5: preparing HPO materialized feature caches")
        check_min_free_space(args)
        hpo_materialization_rows = run_materialization_stage(
            args,
            _TASK.materialization_specs_for_hpo(args),
            run_root,
            run_root / "materialization_hpo_status.csv",
        )
        if not args.dry_run and any(
            row["status"] not in {"hit", "built", "skipped"} for row in hpo_materialization_rows
        ):
            raise RuntimeError(
                f"One or more HPO materialization jobs failed. See {run_root / 'materialization_hpo_status.csv'}"
            )
        enforce_shared_cache_budget(
            args,
            run_root,
            [row.get("cache_dir", "") for row in hpo_materialization_rows],
        )

        log_step("Stage 1: planning HPO jobs")
        hpo_jobs = _TASK.build_hpo_jobs(args, run_root)
        write_dict_rows(run_root / "planned_hpo_jobs.csv", job_plan_rows(hpo_jobs))
        log_step(f"Stage 1: {len(hpo_jobs)} HPO job(s)")
        if not getattr(args, "skip_final", False):
            log_step("Stage 1/3: running HPO with endpoint-streaming finals")
            hpo_rows, final_materialization_rows, final_jobs, final_rows = (
                run_hpo_with_streaming_finals(
                    args,
                    run_root,
                    hpo_jobs,
                )
            )
            if not args.dry_run and not jobs_succeeded(hpo_rows, hpo_jobs):
                raise RuntimeError(
                    f"One or more HPO jobs failed. See {run_root / 'hpo_status.csv'}"
                )
            if not args.dry_run and any(
                row["status"] not in {"hit", "built", "skipped"}
                for row in final_materialization_rows
            ):
                raise RuntimeError(
                    f"One or more final materialization jobs failed. See {run_root / 'materialization_final_status.csv'}"
                )
            if not args.dry_run and not jobs_succeeded(final_rows, final_jobs):
                write_dict_rows(
                    run_root / "runs_status.csv",
                    embedding_rows
                    + hpo_materialization_rows
                    + hpo_rows
                    + final_materialization_rows
                    + final_rows,
                )
                raise RuntimeError(
                    f"One or more final jobs failed. See {run_root / 'final_status.csv'}"
                )
            if not args.dry_run:
                cleanup_summary(
                    args, prune_checkpoints(args, run_root, "hpo"), title="HPO checkpoint cleanup"
                )
                final_cleanup = prune_checkpoints(args, run_root, "final")
                final_cleanup.extend(prune_checkpoint_budget(args, run_root))
                cleanup_summary(args, final_cleanup, title="Final/storage checkpoint cleanup")
        else:
            hpo_rows = run_jobs(args, hpo_jobs, run_root / "hpo_status.csv")
            if not args.dry_run and not jobs_succeeded(hpo_rows, hpo_jobs):
                raise RuntimeError(
                    f"One or more HPO jobs failed. See {run_root / 'hpo_status.csv'}"
                )
            if not args.dry_run:
                cleanup_summary(
                    args, prune_checkpoints(args, run_root, "hpo"), title="HPO checkpoint cleanup"
                )
    else:
        log_step("Stage 1: HPO skipped")
        for skipped_path in (run_root / "planned_hpo_jobs.csv", run_root / "hpo_status.csv"):
            if not skipped_path.exists():
                write_dict_rows(skipped_path, [])

    if getattr(args, "skip_final", False):
        log_step("Stage 3: final training skipped")
        write_dict_rows(run_root / "planned_final_jobs.csv", [])
        write_dict_rows(run_root / "final_status.csv", [])
    elif getattr(args, "skip_hpo", False) or not hpo_jobs:
        log_step("Stage 2: preparing final materialized feature caches")
        check_min_free_space(args)
        final_materialization_rows = run_materialization_stage(
            args,
            _TASK.materialization_specs_for_final(args, run_root),
            run_root,
            run_root / "materialization_final_status.csv",
        )
        if not args.dry_run and any(
            row["status"] not in {"hit", "built", "skipped"} for row in final_materialization_rows
        ):
            raise RuntimeError(
                f"One or more final materialization jobs failed. See {run_root / 'materialization_final_status.csv'}"
            )
        enforce_shared_cache_budget(
            args,
            run_root,
            [
                row.get("cache_dir", "")
                for row in [*hpo_materialization_rows, *final_materialization_rows]
            ],
        )

        log_step("Stage 3: planning final training jobs")
        write_best_params_csv = True
        if getattr(args, "skip_hpo", False):
            write_best_params_csv = False
            if args.final_params_source == "fixed":
                log_step("Stage 3: using fixed final Hydra config; no HPO best params required")
                params_by_endpoint = {endpoint: {} for endpoint in args.endpoints}
            elif args.final_params_source == "hpo_csv":
                best_params_path = final_best_params_path(args, run_root)
                log_step(f"Stage 3: loading final best params from {best_params_path}")
                params_by_endpoint = load_best_params_csv(best_params_path, args.endpoints)
            else:
                raise ValueError(f"Unsupported final params source: {args.final_params_source}")
        elif args.dry_run:
            params_by_endpoint = {endpoint: {} for endpoint in args.endpoints}
        else:
            params_by_endpoint = None
        final_jobs = build_final_jobs(
            args,
            run_root,
            params_by_endpoint=params_by_endpoint,
            write_best_params_csv=write_best_params_csv,
        )
        write_dict_rows(run_root / "planned_final_jobs.csv", job_plan_rows(final_jobs))
        log_step(f"Stage 3: {len(final_jobs)} final job(s)")
        final_rows = run_jobs(args, final_jobs, run_root / "final_status.csv")
        if not args.dry_run and not jobs_succeeded(final_rows, final_jobs):
            write_dict_rows(
                run_root / "runs_status.csv",
                embedding_rows
                + hpo_materialization_rows
                + hpo_rows
                + final_materialization_rows
                + final_rows,
            )
            raise RuntimeError(
                f"One or more final jobs failed. See {run_root / 'final_status.csv'}"
            )
        if not args.dry_run:
            final_cleanup = prune_checkpoints(args, run_root, "final")
            final_cleanup.extend(prune_checkpoint_budget(args, run_root))
            cleanup_summary(args, final_cleanup, title="Final/storage checkpoint cleanup")
    else:
        log_step("Stage 3: final training handled by endpoint-streaming scheduler")

    write_dict_rows(
        run_root / "planned_runs.csv", job_plan_rows([*embedding_jobs, *hpo_jobs, *final_jobs])
    )
    write_dict_rows(
        run_root / "runs_status.csv",
        embedding_rows
        + hpo_materialization_rows
        + hpo_rows
        + final_materialization_rows
        + final_rows,
    )

    if not args.dry_run and final_rows:
        log_step("Stage 4: aggregating metrics")
        aggregate(args.output_root, run_root, args.endpoints, args.seeds)
    after_rows = write_disk_report(args, run_root, "after")
    render_disk_summary(args, after_rows, title="Disk usage after")
    log_step(f"Pipeline complete: {run_root}")


def discover_gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return ["0"]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return gpus or ["0"]


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_termination_signal)
    signal.signal(signal.SIGTERM, handle_termination_signal)
    try:
        main()
    except KeyboardInterrupt:
        cleanup_interrupted_run("orchestrator interrupted by KeyboardInterrupt")
        raise
