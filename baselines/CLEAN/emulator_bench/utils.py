from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence


BASELINE_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = BASELINE_ROOT / "app"
APP_DATA_DIR = APP_DIR / "data"
APP_RESULTS_DIR = APP_DIR / "results"
EMULATOR_DIR = BASELINE_ROOT / "emulator_bench"
DEFAULT_RUNS_ROOT = EMULATOR_DIR / "runs"
DEFAULT_CACHE_ROOT = EMULATOR_DIR / "cache" / "esm1b"


def resolve_path(path: str | Path, *, base: Path = BASELINE_ROOT) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    value = value.strip().replace(os.sep, "__").replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    if not value:
        raise ValueError("Cannot build a slug from an empty value")
    return value


def split_group_slug(split_group: str) -> str:
    return slugify(split_group)


def seed_run_root_for_split(
    split_group: str,
    seed: int,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    return Path(runs_root) / split_group_slug(split_group) / "seeds" / str(seed)


def seed_train_metadata_path_for_split(
    split_group: str,
    seed: int,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    return seed_run_root_for_split(split_group, seed, runs_root) / "train.json"


def seed_results_root_for_split(
    split_group: str,
    seed: int,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    return seed_run_root_for_split(split_group, seed, runs_root) / "results"


def seed_run_root(metadata: dict, seed: int) -> Path:
    return Path(metadata["run_root"]) / "seeds" / str(seed)


def seed_train_metadata_path(metadata: dict, seed: int) -> Path:
    return seed_run_root(metadata, seed) / "train.json"


def seed_results_root(metadata: dict, seed: int) -> Path:
    return seed_run_root(metadata, seed) / "results"


def cache_key_for_entry(entry: str) -> str:
    return slugify(entry)


def write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def conda_python(env_name: str) -> list[str]:
    if env_name == "current":
        return [sys.executable]
    return ["conda", "run", "-n", env_name, "python"]


def shell_join(command: Sequence[str | Path]) -> str:
    return shlex.join([str(part) for part in command])


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    printable = shell_join(command)
    print(f"[emulator_bench] running: {printable}", flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


@contextmanager
def pushd(path: str | Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def add_clean_to_path() -> None:
    clean_src = APP_DIR / "src"
    clean_src_str = str(clean_src)
    if clean_src_str not in sys.path:
        sys.path.insert(0, clean_src_str)


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
    raise FileNotFoundError(
        "Could not find task-spooler. Install it or pass --spooler-bin. "
        "This repository uses the command name 'ts'."
    )


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
    if gpus is not None:
        if gpus < 0:
            raise ValueError(f"gpus must be >= 0, got {gpus}")
        if gpus > 0:
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
                if "finished" in status.lower():
                    finished.append(job_id)
                elif "failed" in status.lower() or "error" in status.lower():
                    raise RuntimeError(f"task-spooler job {job_id} failed: {status}")
            for job_id in finished:
                remaining.remove(job_id)
                progress.update(1)
            if remaining:
                time.sleep(poll_seconds)


def metadata_path_for_split(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    return Path(runs_root) / split_group_slug(split_group) / "metadata.json"


def load_run_metadata(split_group: str, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> dict:
    path = metadata_path_for_split(split_group, runs_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return read_json(path)


def env_with_cuda_visible(device: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env
