import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_SPLIT_GROUPS, REPO_ROOT, discover_split_jobs, normalize_threshold_args
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


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
        raise RuntimeError(f"Optuna storage exists but is not an Optuna DB: {db_path}. Use --reset_storage or a new path.")


def suggest_hparams(trial, args):
    batch_size = int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    final_batch = trial.suggest_categorical("final_batch_size", [512, 1024, 2048, 4096])
    warm_batch = max(batch_size, min(final_batch, 512))
    return {
        "batch_size": batch_size,
        "batch_size_schedule": f"1:{batch_size},20:{warm_batch},60:{final_batch}",
        "lr": trial.suggest_float("lr", 1e-5, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "min_lr": trial.suggest_float("min_lr", 1e-8, 1e-5, log=True),
        "lr_warmup_epochs": trial.suggest_int("lr_warmup_epochs", 0, 15),
        "lr_warmup_start_factor": trial.suggest_float("lr_warmup_start_factor", 0.05, 0.5),
        "loss_alpha": trial.suggest_float("loss_alpha", 0.0, 0.7),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.5, 1.0, 2.0, 5.0]),
        "patience": trial.suggest_categorical("patience", [10, 15, 20, 30]),
        "scheduler": "cosine",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "amsgrad": False,
        "lr_decay_factor": 0.5,
        "lr_decay_patience": 5,
        "min_delta": 0.0,
    }


def run_trial_job(job, seed, hparams, args, trial_number):
    trial_root = Path(job["root_dir"]) / "structurefree_optuna_runs" / f"trial_{trial_number}" / job["split_group"] / job["split_name"] / f"seed_{seed}"
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
            "--embeddings_dir",
            args.embeddings_dir,
            "--out_dir",
            str(trial_root),
            "--task_name",
            f"optuna_trial_{trial_number}_{job['split_group']}_{job['split_name']}_seed{seed}",
            "--sequence_col",
            args.sequence_col,
            "--smiles_col",
            args.smiles_col,
            "--target_col",
            args.target_col,
            "--protein_model_name",
            args.protein_model_name,
            "--molecule_model_name",
            args.molecule_model_name,
            "--max_protein_length",
            str(args.max_protein_length),
            "--max_molecule_length",
            str(args.max_molecule_length),
            "--hidden_sizes",
            args.hidden_sizes,
            "--dropout",
            str(args.dropout),
            "--batch_size",
            str(hparams["batch_size"]),
            "--batch_size_schedule",
            hparams["batch_size_schedule"],
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
            "--lr_warmup_epochs",
            str(hparams["lr_warmup_epochs"]),
            "--lr_warmup_start_factor",
            str(hparams["lr_warmup_start_factor"]),
            "--loss_alpha",
            str(hparams["loss_alpha"]),
            "--clip_grad",
            str(hparams["clip_grad"]),
            "--patience",
            str(hparams["patience"]),
            "--min_delta",
            str(hparams["min_delta"]),
            "--val_every",
            str(args.val_every),
            "--monitor_metric",
            args.metric,
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--embedding_cache_items",
            str(args.embedding_cache_items),
            "--seed",
            str(seed),
        ]
        for enabled, flag in [
            (args.pin_memory, "--pin_memory"),
            (args.persistent_workers, "--persistent_workers"),
            (args.preload_embeddings, "--preload_embeddings"),
            (args.torch_compile, "--torch_compile"),
        ]:
            if enabled:
                cmd.append(flag)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    metrics = pd.read_csv(metric_file).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError(f"Metric `{args.metric}` not found in {metric_file}")
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune retraining-safe StructureFree-DTA hyperparameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--molecule_model_name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--max_protein_length", type=int, default=1024)
    parser.add_argument("--max_molecule_length", type=int, default=128)
    parser.add_argument("--protein_batch_size", type=int, default=64)
    parser.add_argument("--molecule_batch_size", type=int, default=256)
    parser.add_argument("--embedding_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--hidden_sizes", type=str, default="1024,768,512,256,1")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_embeddings", action="store_true")
    parser.add_argument("--embedding_cache_items", type=int, default=20000)
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mse", "mae", "loss", "r2_score", "pearson", "spearman", "ci"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="structurefree_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = f"sqlite:///{Path(args.base_dir) / 'optuna_studies' / (args.study_name + '.db')}"

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    study = optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    def objective(trial):
        hparams = suggest_hparams(trial, args)
        scores = []
        for job in jobs:
            for seed in args.seeds:
                scores.append(run_trial_job(job, seed, hparams, args, trial.number))
        trial.set_user_attr("n_jobs", len(jobs))
        trial.set_user_attr("n_scores", len(scores))
        trial.set_user_attr("batch_size_schedule", hparams["batch_size_schedule"])
        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=args.n_trials)

    out_dir = Path(args.base_dir) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / f"{args.study_name}_trials.csv", index=False)
    with open(out_dir / f"{args.study_name}_best_hparams.json", "w") as handle:
        json.dump(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "direction": study.direction.name.lower(),
                "best_trial_number": int(study.best_trial.number),
                "best_value": float(study.best_value),
                "best_hparams": dict(study.best_params) | {"batch_size_schedule": study.best_trial.user_attrs.get("batch_size_schedule")},
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
