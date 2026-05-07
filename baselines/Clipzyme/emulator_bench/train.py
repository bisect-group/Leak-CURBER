from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

from .utils import (
    BASELINE_ROOT,
    DEFAULT_RUNS_ROOT,
    choose_clipzyme_precision,
    ensure_dir,
    existing_train_metadata_checkpoint,
    find_checkpoint,
    load_run_metadata,
    run_command,
    seed_run_root,
    seed_train_metadata_path,
    write_json,
)


BASE_TRAIN_CONFIG = BASELINE_ROOT / "configs" / "train" / "clip_esm.json"


def _one(value):
    return [value]


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def build_train_config(
    metadata: dict,
    *,
    seed: int,
    epochs: int | None,
    precision: str,
    available_gpus: str,
    checkpoint_path: str | None,
    num_workers: int,
    batch_size: int | None,
    accumulate_grad_batches: int | None,
    clip_freeze_esm: bool,
) -> dict:
    import json

    base = json.loads(BASE_TRAIN_CONFIG.read_text())
    config = copy.deepcopy(base)
    config["script"] = "main"
    config["available_gpus"] = [available_gpus]
    hp = config["cartesian_hyperparams"]

    seed_root = seed_run_root(metadata["split_group"], seed, Path(metadata["run_root"]).parent)
    checkpoint_dir = ensure_dir(seed_root / "checkpoints")
    inference_dir = ensure_dir(seed_root / "inference")
    torch_hub_dir = ensure_dir(Path(metadata["cache_root"]) / "torch_hub")

    hp.update(
        {
            "dataset_file_path": _one(_abs(metadata["clipzyme_files"]["dataset_json"])),
            "ec2uniprot_path": _one(_abs(metadata["clipzyme_files"]["ec2uniprot"])),
            "uniprot2sequence_path": _one(_abs(metadata["clipzyme_files"]["uniprot2sequence"])),
            "from_checkpoint": _one(bool(checkpoint_path)),
            "checkpoint_path": _one(_abs(checkpoint_path)) if checkpoint_path else _one(None),
            "remove_duplicate_reactions": _one(False),
            "use_mapped_reaction": _one(True),
            "assign_splits": _one(False),
            "split_seed": _one(int(seed)),
            "seed": _one(int(seed)),
            "max_protein_length": _one(int(metadata["max_sequence_length"])),
            "precision": _one(precision),
            "max_epochs": _one(20 if epochs is None else int(epochs)),
            "checkpoint_dir": _one(_abs(checkpoint_dir)),
            "checkpoint_save_last": _one(True),
            "pretrained_hub_dir": _one(_abs(torch_hub_dir)),
            "logger_name": _one("tensorboard"),
            "logger_tags": _one(f"clipzyme emulator {metadata['run_slug']} seed{seed}"),
            "project_name": _one("emulator_clipzyme"),
            "workspace": _one("emulator"),
            "inference_dir": _one(_abs(inference_dir)),
            "gpus": _one(1),
            "num_workers": _one(int(num_workers)),
            "dev": _one(False),
            "test": _one(False),
            "train": _one(True),
            "lr": _one(1e-4),
            "weight_decay": _one(0.05),
        }
    )
    if batch_size is not None:
        hp["batch_size"] = _one(int(batch_size))
    if accumulate_grad_batches is not None:
        hp["accumulate_grad_batches"] = _one(int(accumulate_grad_batches))
    if clip_freeze_esm:
        hp["clip_freeze_esm"] = _one(True)
    if not checkpoint_path:
        hp.pop("checkpoint_path", None)
        hp["from_checkpoint"] = _one(False)
    hp.pop("dataset_cache_path", None)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Clipzyme on an Leak-CURBER split group")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--available-gpus", default="${CUDA_VISIBLE_DEVICES:-0}")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--clip-freeze-esm", action="store_true")
    return parser.parse_args()


def _args_files(seed_root: Path) -> list[Path]:
    return sorted((seed_root / "logs").glob("*.args"), key=lambda path: path.stat().st_mtime)


def _checkpoint_from_args_file(args_files: list[Path]) -> Path | None:
    if not args_files:
        return None
    try:
        saved_args = pickle.load(args_files[-1].open("rb"))
        model_path = saved_args.get("model_path")
    except Exception:
        return None
    if model_path and Path(model_path).exists():
        return Path(model_path)
    return None


def _latest_experiment_log(seed_root: Path) -> Path | None:
    logs = sorted((seed_root / "logs").glob("*.txt"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def _tail_text(path: Path, *, max_lines: int = 80) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


def _missing_checkpoint_error(checkpoint_dir: Path, seed_root: Path) -> FileNotFoundError:
    latest_log = _latest_experiment_log(seed_root)
    message = f"No checkpoint found under {checkpoint_dir}"
    if latest_log is not None:
        message += (
            f". Latest Clipzyme experiment log: {latest_log}\n"
            f"Last lines from failed experiment:\n{_tail_text(latest_log)}"
        )
    return FileNotFoundError(message)


def write_train_metadata(
    *,
    metadata: dict,
    seed: int,
    epochs: int,
    precision: str,
    config_path: str | Path,
    checkpoint: str | Path,
    checkpoint_dir: str | Path,
    seed_root: str | Path,
    runs_root: str | Path,
) -> dict:
    seed_root = Path(seed_root)
    args_files = _args_files(seed_root)
    train_metadata = {
        "split_group": metadata["split_group"],
        "run_slug": metadata["run_slug"],
        "seed": int(seed),
        "epochs": int(epochs),
        "precision": precision,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_dir": str(checkpoint_dir),
        "args_files": [str(path) for path in args_files],
        "seed_run_root": str(seed_root),
    }
    canonical_path = seed_train_metadata_path(metadata["split_group"], seed, runs_root)
    write_json(canonical_path, train_metadata)
    write_json(Path(metadata["run_root"]) / f"train_seed{seed}.json", train_metadata)
    return train_metadata


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    precision = choose_clipzyme_precision(args.precision)
    metadata = load_run_metadata(args.split_group, args.runs_root)
    seed_root = seed_run_root(args.split_group, args.seed, args.runs_root)
    ensure_dir(seed_root / "configs")
    ensure_dir(seed_root / "logs")

    config = build_train_config(
        metadata,
        seed=args.seed,
        epochs=args.epochs,
        precision=precision,
        available_gpus=args.available_gpus,
        checkpoint_path=args.checkpoint_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        clip_freeze_esm=args.clip_freeze_esm,
    )
    config_path = seed_root / "configs" / "train_config.json"
    write_json(config_path, config)
    checkpoint_dir = seed_root / "checkpoints"

    checkpoint = existing_train_metadata_checkpoint(args.split_group, args.seed, args.runs_root)
    if checkpoint is None:
        try:
            checkpoint = find_checkpoint(checkpoint_dir)
        except FileNotFoundError:
            checkpoint = None
    if checkpoint is not None:
        args_checkpoint = _checkpoint_from_args_file(_args_files(seed_root))
        if args_checkpoint is not None:
            checkpoint = args_checkpoint
        write_train_metadata(
            metadata=metadata,
            seed=args.seed,
            epochs=config["cartesian_hyperparams"]["max_epochs"][0],
            precision=precision,
            config_path=config_path,
            checkpoint=checkpoint,
            checkpoint_dir=checkpoint_dir,
            seed_root=seed_root,
            runs_root=args.runs_root,
        )
        print(f"[emulator_bench] checkpoint already exists, skipping train: {checkpoint}", flush=True)
        return

    command = [
        sys.executable,
        "scripts/dispatcher.py",
        "-c",
        str(config_path),
        "-l",
        str(seed_root / "logs"),
    ]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    run_command(command, cwd=BASELINE_ROOT, env=env)

    try:
        checkpoint = find_checkpoint(checkpoint_dir)
    except FileNotFoundError as exc:
        raise _missing_checkpoint_error(checkpoint_dir, seed_root) from exc
    args_checkpoint = _checkpoint_from_args_file(_args_files(seed_root))
    if args_checkpoint is not None:
        checkpoint = args_checkpoint
    write_train_metadata(
        metadata=metadata,
        seed=args.seed,
        epochs=config["cartesian_hyperparams"]["max_epochs"][0],
        precision=precision,
        config_path=config_path,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        seed_root=seed_root,
        runs_root=args.runs_root,
    )
    print(f"[emulator_bench] checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
