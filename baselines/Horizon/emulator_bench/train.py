from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .utils import (
    BASELINE_ROOT,
    DEFAULT_RUNS_ROOT,
    choose_precision,
    ensure_dir,
    existing_train_metadata_checkpoint,
    find_checkpoint,
    load_run_metadata,
    run_command,
    seed_run_root,
    seed_train_metadata_path,
    write_json,
    write_yaml,
)


def _abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def build_train_config(metadata: dict, *, seed: int, epochs: int | None, precision: str) -> dict:
    seed_root = seed_run_root(metadata["split_group"], seed, Path(metadata["run_root"]).parent)
    checkpoint_dir = seed_root / "checkpoints"
    log_dir = seed_root / "logs"
    max_epochs = 100 if epochs is None else int(epochs)
    return {
        "seed": int(seed),
        "logging": {
            "log_dir": str(log_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "save_every_n_epochs": 10,
            "log_every_n_steps": 1,
        },
        "data": {
            "train_pairs_path": _abs(metadata["baseline_files"]["train"]["pairs"]),
            "test_pairs_path": _abs(metadata["baseline_files"]["val"]["pairs"]),
            "train_reactions_path": _abs(metadata["baseline_files"]["train"]["reactions"]),
            "test_reactions_path": _abs(metadata["baseline_files"]["val"]["reactions"]),
            "protein_embeds_path": _abs(metadata["baseline_files"]["protein_embeds"]),
            "train_batch_size": 16384,
            "retrieval_batch_size": 128,
            "num_workers": 20,
            "pin_memory": False,
            "rdkit_fp_dim": 1024,
            "drfp_dim": 1024,
            "standardize_reactions": True,
            "standardize_hypervalent": True,
            "standardize_remove_hs": True,
            "standardize_kekulize": False,
            "standardize_uncharge": True,
            "standardize_metals": True,
        },
        "model": {
            "name": "DualContrastiveModel",
            "query_encoder_dims": [2048, 4096, 4096, 512],
            "target_encoder_dims": [1024, 4096, 4096, 512],
            "embedding_dim": 512,
        },
        "training": {
            "max_epochs": max_epochs,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "precision": precision,
            "loss": {
                "name": "FullBatchMLNCELoss",
                "beta": 10.0,
                "learn_beta": False,
                "beta_min": 0.01,
                "beta_max": 100.0,
            },
            "metrics": {"top_k": [1, 10, 100, 1000]},
            "check_val_every_n_epoch": 10,
            "enable_progress_bar": True,
        },
    }


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
    train_metadata = {
        "split_group": metadata["split_group"],
        "run_slug": metadata["run_slug"],
        "seed": int(seed),
        "epochs": int(epochs),
        "precision": precision,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_dir": str(checkpoint_dir),
        "seed_run_root": str(seed_root),
    }
    canonical_path = seed_train_metadata_path(metadata["split_group"], seed, runs_root)
    write_json(canonical_path, train_metadata)
    write_json(Path(metadata["run_root"]) / f"train_seed{seed}.json", train_metadata)
    return train_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Horizon on Leak-CURBER split data")
    parser.add_argument("--split-group", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_run_metadata(args.split_group, args.runs_root)
    if "protein_embeds" not in metadata["baseline_files"]:
        raise FileNotFoundError("Missing protein HDF5 path in metadata. Run cache_features first.")
    precision = choose_precision(args.precision)
    config = build_train_config(
        metadata,
        seed=args.seed,
        epochs=args.epochs,
        precision=precision,
    )
    seed_root = seed_run_root(args.split_group, args.seed, args.runs_root)
    ensure_dir(seed_root / "configs")
    config_path = seed_root / "configs" / "train.yaml"
    write_yaml(config_path, config)
    checkpoint_dir = config["logging"]["checkpoint_dir"]

    checkpoint = existing_train_metadata_checkpoint(args.split_group, args.seed, args.runs_root)
    if checkpoint is None:
        try:
            checkpoint = find_checkpoint(checkpoint_dir)
        except FileNotFoundError:
            checkpoint = None
    if checkpoint is not None:
        write_train_metadata(
            metadata=metadata,
            seed=args.seed,
            epochs=config["training"]["max_epochs"],
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
        "train.py",
        "--config",
        str(config_path),
        "--seed",
        str(args.seed),
    ]
    run_command(command, cwd=BASELINE_ROOT)

    checkpoint = find_checkpoint(checkpoint_dir)
    write_train_metadata(
        metadata=metadata,
        seed=args.seed,
        epochs=config["training"]["max_epochs"],
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
