from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


BASELINE_ROOT = Path(__file__).resolve().parents[1]
EMULATOR_DIR = BASELINE_ROOT / "emulator_bench"
DEFAULT_DATASET_ROOT = BASELINE_ROOT / "../../data/processed/datasets/enzyme_classification_dataset"
DEFAULT_RUNS_ROOT = EMULATOR_DIR / "runs"
DEFAULT_CACHE_ROOT = EMULATOR_DIR / "cache" / "tokens"
DEFAULT_VOCAB_PATH = EMULATOR_DIR / "vocab" / "ec_vocab.json"
TOKENIZER_PATH = BASELINE_ROOT / "utils" / "tokenizer.pickle"
DEFAULT_MODEL_DIMENSION = 1024
DEFAULT_SEEDS = (1234, 2345, 3456)
SPLIT_NAMES = ("train", "val", "test")


def resolve_path(path, base=BASELINE_ROOT):
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value):
    value = str(value).strip().replace(os.sep, "__").replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._")
    if not value:
        raise ValueError("Cannot build a slug from an empty value")
    return value


def split_group_slug(split_group):
    return slugify(split_group)


def cache_key_for_entry(entry):
    return slugify(entry)


def metadata_path_for_split(split_group, runs_root=DEFAULT_RUNS_ROOT):
    return Path(runs_root) / split_group_slug(split_group) / "metadata.json"


def split_run_root(split_group, runs_root=DEFAULT_RUNS_ROOT):
    return Path(runs_root) / split_group_slug(split_group)


def seed_run_root_for_split(split_group, seed, runs_root=DEFAULT_RUNS_ROOT):
    return split_run_root(split_group, runs_root) / "seeds" / str(seed)


def seed_train_metadata_path_for_split(split_group, seed, runs_root=DEFAULT_RUNS_ROOT):
    return seed_run_root_for_split(split_group, seed, runs_root) / "train.json"


def seed_results_root_for_split(split_group, seed, runs_root=DEFAULT_RUNS_ROOT):
    return seed_run_root_for_split(split_group, seed, runs_root) / "results"


def seed_run_root(metadata, seed):
    return Path(metadata["run_root"]) / "seeds" / str(seed)


def seed_train_metadata_path(metadata, seed):
    return seed_run_root(metadata, seed) / "train.json"


def seed_results_root(metadata, seed):
    return seed_run_root(metadata, seed) / "results"


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def load_run_metadata(split_group, runs_root=DEFAULT_RUNS_ROOT):
    path = metadata_path_for_split(split_group, runs_root)
    if not path.exists():
        raise FileNotFoundError("Missing run metadata: {}".format(path))
    return read_json(path)


def resolve_model_max_tokens(max_tokens=None, model_dimension=DEFAULT_MODEL_DIMENSION):
    model_dimension = int(model_dimension)
    if model_dimension <= 1:
        raise ValueError("model_dimension must be > 1")
    if max_tokens is None:
        resolved = model_dimension
    else:
        resolved = int(max_tokens)
    if resolved <= 1:
        raise ValueError("max_tokens must be > 1")
    if resolved > model_dimension:
        raise ValueError(
            "max_tokens={} exceeds the model positional limit {}".format(
                resolved,
                model_dimension,
            )
        )
    return resolved


def conda_python(env_name):
    if env_name == "current":
        return [sys.executable]
    return ["conda", "run", "-n", env_name, "python"]


def shell_join(command):
    return shlex.join([str(part) for part in command])


def run_command(command, cwd=None, env=None):
    print("[emulator_bench] running: {}".format(shell_join(command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def find_ts(explicit=None):
    candidates = [explicit] if explicit else ["ts"]
    for candidate in candidates:
        if not candidate:
            continue
        found = shutil.which(candidate)
        if found:
            return found
        candidate_path = resolve_path(candidate)
        if candidate_path.exists() and os.access(str(candidate_path), os.X_OK):
            return str(candidate_path)
    raise FileNotFoundError("Could not find task-spooler command 'ts'. Pass --spooler-bin.")


def submit_ts_job(
    command,
    label,
    log_name,
    depends_on=None,
    gpus=0,
    spooler_bin=None,
):
    spooler = find_ts(spooler_bin)
    args = [spooler, "-L", label, "-O", log_name]
    if gpus is not None and int(gpus) > 0:
        args.extend(["-G", str(int(gpus))])
    depends = [str(job_id) for job_id in (depends_on or []) if str(job_id)]
    if depends:
        args.extend(["-W", ",".join(depends)])
    args.extend(["bash", "-lc", command])
    output = subprocess.check_output(args, text=True).strip()
    job_id = output.splitlines()[-1].strip()
    print("[emulator_bench] queued {}: job {}".format(label, job_id), flush=True)
    return job_id


def wait_for_ts_jobs(job_ids, spooler_bin=None, poll_seconds=10.0):
    from tqdm import tqdm

    spooler = find_ts(spooler_bin)
    remaining = {str(job_id) for job_id in job_ids}
    with tqdm(total=len(remaining), desc="ts jobs", unit="job") as progress:
        while remaining:
            finished = []
            for job_id in sorted(remaining):
                status = subprocess.check_output([spooler, "-s", job_id], text=True).strip()
                lower = status.lower()
                if "finished" in lower:
                    info = subprocess.check_output([spooler, "-i", job_id], text=True).strip()
                    first_line = info.splitlines()[0] if info else ""
                    if "exit code 0" not in first_line:
                        raise RuntimeError(
                            "task-spooler job {} finished unsuccessfully: {}".format(
                                job_id,
                                first_line,
                            )
                        )
                    finished.append(job_id)
                elif "failed" in lower or "error" in lower or "skipped" in lower:
                    raise RuntimeError("task-spooler job {} failed: {}".format(job_id, status))
            for job_id in finished:
                remaining.remove(job_id)
                progress.update(1)
            if remaining:
                time.sleep(poll_seconds)


def with_repo_prefix(command, cuda_visible_devices=None):
    cmd = shell_join(command)
    if cuda_visible_devices is not None:
        cmd = "env CUDA_VISIBLE_DEVICES={} {}".format(
            shell_join([cuda_visible_devices]),
            cmd,
        )
    return "cd {} && {}".format(shell_join([BASELINE_ROOT]), cmd)


def choose_precision(requested="auto"):
    if requested != "auto":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            return "32"
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        return "16-mixed"
    except Exception:
        return "32"


def lightning_precision_arg(precision):
    """Return a PyTorch Lightning precision value compatible with Lightning 1.9."""
    if precision in {"32", "fp32", 32}:
        return 32
    if precision in {"16", "16-mixed", "fp16", 16}:
        return 16
    if precision in {"bf16", "bf16-mixed"}:
        try:
            import pytorch_lightning as pl

            major = int(str(pl.__version__).split(".", 1)[0])
        except Exception:
            major = 1
        return "bf16-mixed" if major >= 2 else "bf16"
    raise ValueError("Unsupported precision: {}".format(precision))


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = "{}{}".format(prefix, key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=name + "."))
        else:
            flat[name] = value
    return flat
