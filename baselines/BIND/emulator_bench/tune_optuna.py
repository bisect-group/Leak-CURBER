import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_HIDDEN_STATE_INDICES, discover_split_jobs, normalize_threshold_args
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_regression_tvt.py"


def metric_direction(metric):
    return "minimize" if metric in {"rmse", "mse", "mae", "loss"} else "maximize"


def sqlite_path_from_storage(storage):
    if not storage or not storage.startswith("sqlite:///"):
        return None
    parsed = urlparse(storage)
    raw_path = unquote(parsed.path or "")
    return Path(raw_path) if raw_path else None


def sqlite_has_optuna_schema(db_path):
    with sqlite3.connect(str(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    return "version_info" in tables


def prepare_optuna_storage(args):
    db_path = sqlite_path_from_storage(args.storage)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return
    if args.reset_storage:
        db_path.unlink()
        return
    if not sqlite_has_optuna_schema(db_path):
        raise RuntimeError(
            "Optuna storage exists but does not contain a valid Optuna schema: "
            f"{db_path}. Use a new --storage path or rerun with --reset_storage."
        )


def suggest_hparams(trial, args):
    batch_choices = [size for size in [1, 2, 4, 8, 16, 32, 64] if size <= int(args.effective_batch_size) and int(args.effective_batch_size) % size == 0]
    if not batch_choices:
        batch_choices = [1]
    batch_size = int(trial.suggest_categorical("batch_size", batch_choices))
    grad_accumulation_steps = max(1, int(args.effective_batch_size) // batch_size)
    return {
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
        "min_lr": trial.suggest_float("min_lr", 1e-7, 5e-5, log=True),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.0, 0.5, 1.0, 2.0]),
        "patience": trial.suggest_categorical("patience", [0, 2, 3, 4]),
        "scheduler": "cosine",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "amsgrad": False,
        "lr_decay_factor": 0.5,
        "lr_decay_patience": 2,
        "min_delta": 0.0,
    }


def run_trial_job(job, seed, hparams, args, trial_number):
    trial_root = (
        Path(job["value_root"])
        / "optuna_runs"
        / args.study_name
        / f"trial_{trial_number}"
        / job["split_group"]
        / job["split_name"]
        / f"seed_{seed}"
    )
    metric_file = trial_root / f"final_results_{args.eval_split}.csv"
    if not metric_file.exists() or args.overwrite_runs:
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--train_path",
            job["train_path"],
            "--val_path",
            job["val_path"],
            "--test_path",
            job["test_path"],
            "--value_type",
            job["value_type"],
            "--target_head",
            job["value_type"],
            "--embeddings_dir",
            args.embeddings_dir,
            "--out_dir",
            str(trial_root),
            "--task_name",
            f"optuna_trial_{trial_number}_{job['value_type']}_{job['split_group']}_{job['split_name']}_seed{seed}",
            "--sequence_col",
            args.sequence_col,
            "--smiles_col",
            args.smiles_col,
            "--target_col",
            args.target_col,
            "--protein_model_name",
            args.protein_model_name,
            "--hidden_state_indices",
            *[str(index) for index in args.hidden_state_indices],
            "--protein_length_cap",
            str(args.protein_length_cap),
            "--long_sequence_strategy",
            args.long_sequence_strategy,
            "--cross_attention_dropout",
            str(args.cross_attention_dropout),
            "--leaky_relu_negative_slope",
            str(args.leaky_relu_negative_slope),
            "--batch_size",
            str(hparams["batch_size"]),
            "--grad_accumulation_steps",
            str(hparams["grad_accumulation_steps"]),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(hparams["lr"]),
            "--weight_decay",
            str(hparams["weight_decay"]),
            "--beta1",
            str(hparams["beta1"]),
            "--beta2",
            str(hparams["beta2"]),
            "--eps",
            str(hparams["eps"]),
            "--scheduler",
            hparams["scheduler"],
            "--lr_decay_factor",
            str(hparams["lr_decay_factor"]),
            "--lr_decay_patience",
            str(hparams["lr_decay_patience"]),
            "--min_lr",
            str(hparams["min_lr"]),
            "--clip_grad",
            str(hparams["clip_grad"]),
            "--huber_delta",
            str(args.huber_delta),
            "--patience",
            str(hparams["patience"]),
            "--min_delta",
            str(hparams["min_delta"]),
            "--monitor_metric",
            args.metric,
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--sharing_strategy",
            args.sharing_strategy,
            "--protein_cache_items",
            str(args.protein_cache_items),
            "--seed",
            str(seed),
        ]
        if args.pin_memory:
            cmd.append("--pin_memory")
        if args.persistent_workers:
            cmd.append("--persistent_workers")
        if args.preload_proteins:
            cmd.append("--preload_proteins")
        if args.preload_ligands:
            cmd.append("--preload_ligands")
        if args.lazy_ligands:
            cmd.append("--lazy_ligands")
        if args.torch_compile:
            cmd.append("--torch_compile")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    metrics = pd.read_csv(metric_file).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError(f"Metric `{args.metric}` not found in {metric_file}")
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune non-architectural BIND retraining hyperparameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--value_type", type=str, required=True)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--cross_attention_dropout", type=float, default=0.1)
    parser.add_argument("--leaky_relu_negative_slope", type=float, default=0.05)
    parser.add_argument("--effective_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--sharing_strategy", choices=["file_descriptor", "file_system"], default="file_system")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--preload_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")
    parser.add_argument("--max_residues", type=int, default=4096)
    parser.add_argument("--max_batch", type=int, default=4)
    parser.add_argument("--sliding_window_stride", type=int, default=896)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--huber_delta", type=float, default=2.0)
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.hidden_state_indices = [int(index) for index in args.hidden_state_indices]
    args.value_type = args.value_type.lower()
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.study_name is None:
        args.study_name = f"bind_{args.value_type}_optuna"
    if args.storage is None:
        args.storage = "sqlite:///%s" % (Path(args.base_dir) / args.value_type / "optuna_studies" / f"{args.study_name}.db")

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    jobs = discover_split_jobs(
        Path(args.base_dir),
        value_types=[args.value_type],
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}/{args.value_type}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=metric_direction(args.metric),
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        load_if_exists=True,
    )

    trial_rows = []

    def objective(trial):
        hparams = suggest_hparams(trial, args)
        trial.set_user_attr("resolved_hparams", hparams)
        metrics = []
        for job in jobs:
            for seed in args.seeds:
                metrics.append(run_trial_job(job, seed, hparams, args, trial.number))
        score = float(np.mean(metrics)) if metrics else float("nan")
        trial_rows.append({"trial": trial.number, **hparams, "metric": score})
        return score

    import numpy as np

    study.optimize(objective, n_trials=args.n_trials)

    out_dir = Path(args.base_dir) / args.value_type / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_hparams = dict(study.best_trial.user_attrs.get("resolved_hparams", dict(study.best_params)))
    best_payload = {
        "study_name": args.study_name,
        "value_type": args.value_type,
        "metric": args.metric,
        "direction": metric_direction(args.metric),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_hparams": best_hparams,
    }
    with open(out_dir / f"{args.study_name}_best_hparams.json", "w") as handle:
        json.dump(best_payload, handle, indent=2, sort_keys=True)
    pd.DataFrame(trial_rows).to_csv(out_dir / f"{args.study_name}_trials.csv", index=False)
    print(json.dumps(best_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
