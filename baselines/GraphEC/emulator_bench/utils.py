from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence


BASELINE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = BASELINE_ROOT / "emulator_bench"
DEFAULT_DATASET_ROOT = BASELINE_ROOT.parents[1] / "data/processed/datasets/enzyme_classification_dataset"
DEFAULT_RUNS_ROOT = PACKAGE_ROOT / "runs"
DEFAULT_CACHE_ROOT = PACKAGE_ROOT / "cache/enzyme_classification_dataset"
SPLIT_NAMES = ("train", "val", "test")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path, *, base: str | Path = BASELINE_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def read_json(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path: str | Path, data: object) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def split_group_slug(split_group: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", split_group.strip("/")).replace("/", "_")


def safe_id(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.|:-]+", "_", text)
    return text[:180] or "missing_id"


def metadata_path_for_split(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    return Path(runs_root) / split_group_slug(split_group) / "metadata.json"


def load_run_metadata(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> dict:
    path = metadata_path_for_split(split_group, runs_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return read_json(path)


def seed_run_root_for_split(
    split_group: str, seed: int, runs_root: str | Path = DEFAULT_RUNS_ROOT
) -> Path:
    return Path(runs_root) / split_group_slug(split_group) / "seeds" / str(seed)


def seed_train_metadata_path_for_split(
    split_group: str, seed: int, runs_root: str | Path = DEFAULT_RUNS_ROOT
) -> Path:
    return seed_run_root_for_split(split_group, seed, runs_root) / "train.json"


def seed_results_root_for_split(
    split_group: str, seed: int, runs_root: str | Path = DEFAULT_RUNS_ROOT
) -> Path:
    return seed_run_root_for_split(split_group, seed, runs_root) / "results"


def conda_python(env_name: str) -> list[str]:
    if env_name in {"", "current", "none"}:
        return [sys.executable]
    executable = Path(sys.executable).resolve()
    env_root = executable.parent.parent
    if env_root.name == env_name and env_root.parent.name == "envs":
        return [str(executable)]
    return ["conda", "run", "-n", env_name, "python"]


def shell_join(parts: Sequence[str | Path]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in parts)


def find_spooler(explicit: str | None = None) -> str:
    candidates = [explicit] if explicit else ["ts", "tsp"]
    for candidate in candidates:
        if not candidate:
            continue
        found = shutil.which(candidate)
        if found:
            return found
        candidate_path = resolve_path(candidate)
        if candidate_path.exists() and os.access(candidate_path, os.X_OK):
            return str(candidate_path)
    raise FileNotFoundError("Could not find task-spooler. Install it or pass --spooler-bin.")


def submit_ts_job(
    command: str,
    *,
    label: str,
    log_name: str,
    depends_on: Iterable[str] | None = None,
    gpus: int | None = None,
    spooler_bin: str | None = None,
) -> str:
    spooler = find_spooler(spooler_bin)
    args = [spooler, "-L", label, "-O", log_name]
    if gpus is not None and gpus > 0:
        args.extend(["-G", str(gpus)])
    depends = [str(job_id) for job_id in (depends_on or []) if str(job_id)]
    if depends:
        args.extend(["-W", ",".join(depends)])
    args.extend(["bash", "-lc", command])
    output = subprocess.check_output(args, text=True).strip()
    job_id = output.splitlines()[-1].strip()
    print(f"[emulator_bench] queued {label}: job {job_id}", flush=True)
    return job_id


def wait_for_ts_jobs(
    job_ids: Sequence[str],
    *,
    spooler_bin: str | None = None,
    poll_seconds: float = 10.0,
) -> None:
    from tqdm import tqdm

    spooler = find_spooler(spooler_bin)
    remaining = {str(job_id) for job_id in job_ids}
    with tqdm(total=len(remaining), desc="ts jobs", unit="job") as progress:
        while remaining:
            finished = []
            for job_id in sorted(remaining):
                status = subprocess.check_output([spooler, "-s", job_id], text=True).strip()
                lowered = status.lower()
                if "finished" in lowered:
                    finished.append(job_id)
                elif "failed" in lowered or "error" in lowered:
                    raise RuntimeError(f"task-spooler job {job_id} failed: {status}")
            for job_id in finished:
                remaining.remove(job_id)
                progress.update(1)
            if remaining:
                time.sleep(poll_seconds)


@contextmanager
def baseline_cwd():
    old_cwd = Path.cwd()
    os.chdir(BASELINE_ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def import_task_module(task_dir: str, module_name: str):
    """Import a GraphEC task module that uses local absolute imports like `from data import *`."""
    task_path = BASELINE_ROOT / task_dir
    if not task_path.exists():
        raise FileNotFoundError(task_path)
    for stale_name in ("data", "model", "utils", "label_diffusion"):
        sys.modules.pop(stale_name, None)
    sys.path.insert(0, str(task_path))
    try:
        __import__(module_name)
        return sys.modules[module_name]
    finally:
        try:
            sys.path.remove(str(task_path))
        except ValueError:
            pass


def choose_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return "bf16"
    if torch.cuda.is_available():
        return "fp16"
    return "fp32"


def autocast_kwargs(precision: str) -> dict:
    import torch

    if precision == "bf16":
        return {"enabled": True, "dtype": torch.bfloat16}
    if precision == "fp16":
        return {"enabled": True, "dtype": torch.float16}
    return {"enabled": False}


def link_or_copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    ensure_dir(target.parent)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def graph_data_dirs() -> dict[str, Path]:
    return {
        "structures": BASELINE_ROOT / "Data/Structures",
        "prottrans": BASELINE_ROOT / "Data/ProtTrans",
        "dssp": BASELINE_ROOT / "Data/DSSP",
    }


def cache_dirs(cache_root: str | Path) -> dict[str, Path]:
    root = Path(cache_root)
    return {
        "structures": ensure_dir(root / "Structures"),
        "prottrans": ensure_dir(root / "ProtTrans"),
        "dssp": ensure_dir(root / "DSSP"),
        "active_sites": ensure_dir(root / "ActiveSites"),
        "fastas": ensure_dir(root / "fastas"),
        "tmp": ensure_dir(root / "tmp"),
    }


def ensure_graphec_data_links(keys: Iterable[str], cache_root: str | Path) -> None:
    dirs = cache_dirs(cache_root)
    graph_dirs = graph_data_dirs()
    for directory in graph_dirs.values():
        ensure_dir(directory)
    for key in keys:
        for suffix in (".tensor", ".pdb"):
            source = dirs["structures"] / f"{key}{suffix}"
            if source.exists():
                link_or_copy(source, graph_dirs["structures"] / f"{key}{suffix}")
        for name in ("prottrans", "dssp"):
            source = dirs[name] / f"{key}.tensor"
            if source.exists():
                link_or_copy(source, graph_dirs[name] / f"{key}.tensor")


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
