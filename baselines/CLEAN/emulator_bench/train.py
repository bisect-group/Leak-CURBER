from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from .utils import (
    APP_DIR,
    DEFAULT_RUNS_ROOT,
    ensure_dir,
    conda_python,
    load_run_metadata,
    run_command,
    seed_run_root,
    seed_train_metadata_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLEAN through the native triplet script")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--env-name", default="clean")
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--adaptive-rate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def choose_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if not torch.cuda.is_available():
            return "fp32"
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    except Exception:
        return "fp32"


def main() -> None:
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.runs_root)
    train_data = metadata["clean_data"]["train"]
    model_name = args.model_name or f"{train_data}_triplet_seed{args.seed}"
    precision = choose_precision(args.precision)

    command = [
        *conda_python(args.env_name),
        "train-triplet.py",
        "--training_data",
        train_data,
        "--model_name",
        model_name,
        "--epoch",
        str(args.epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--hidden_dim",
        str(args.hidden_dim),
        "--out_dim",
        str(args.out_dim),
        "--adaptive_rate",
        str(args.adaptive_rate),
        "--precision",
        precision,
        "--seed",
        str(args.seed),
    ]
    if args.no_progress:
        command.append("--no-progress")

    env = os.environ.copy()
    clean_src = str(APP_DIR / "src")
    env["PYTHONPATH"] = (
        clean_src if not env.get("PYTHONPATH") else f"{clean_src}{os.pathsep}{env['PYTHONPATH']}"
    )
    run_command(command, cwd=APP_DIR, env=env)

    checkpoint = APP_DIR / "data" / "model" / f"{model_name}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished but checkpoint was not found: {checkpoint}")

    run_root = seed_run_root(metadata, args.seed)
    copied_checkpoint = ensure_dir(run_root / "checkpoints") / checkpoint.name
    shutil.copy2(checkpoint, copied_checkpoint)

    train_metadata = {
        "split_group": args.split_group,
        "train_data": train_data,
        "model_name": model_name,
        "checkpoint": str(copied_checkpoint),
        "native_checkpoint": str(checkpoint),
        "epochs": args.epochs,
        "precision": precision,
        "seed": args.seed,
        "seed_run_root": str(run_root),
    }
    canonical_train_json = seed_train_metadata_path(metadata, args.seed)
    write_json(canonical_train_json, train_metadata)
    write_json(
        Path(metadata["run_root"]) / f"train_seed{args.seed}.json",
        {
            **train_metadata,
            "canonical_train_metadata": str(canonical_train_json),
        },
    )
    print(f"[emulator_bench] checkpoint: {copied_checkpoint}", flush=True)
    print(f"[emulator_bench] native checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
