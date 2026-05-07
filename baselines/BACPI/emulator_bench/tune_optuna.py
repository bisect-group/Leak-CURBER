import argparse
import json
import os
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

from emulator_bench.common import DEFAULT_SPLIT_GROUPS, discover_split_jobs, normalize_threshold_args, resolve_base_dir
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
        raise RuntimeError(
            "Optuna storage exists but does not contain a valid Optuna schema: %s. "
            "Use a new --storage path or rerun with --reset_storage." % db_path
        )


def suggest_hparams(trial, args):
    batch_size = int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [16, 24, 32, 48, 64])
    return {
        "batch_size": batch_size,
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-6, 1e-5, 1e-4]),
        "step_size": trial.suggest_categorical("step_size", [5, 8, 10, 12, 15]),
        "gamma": trial.suggest_categorical("gamma", [0.3, 0.5, 0.7, 0.85]),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.0, 1.0, 2.0, 5.0]),
        "patience": trial.suggest_categorical("patience", [0, 4, 8, 12]),
        "scheduler": "step",
        "min_lr": 1e-6,
        "min_delta": 0.0,
        "amsgrad": True,
    }


def run_trial_job(job, seed, hparams, args, trial_number):
    trial_root = (
        Path(job["root_dir"])
        / "bacpi_optuna_runs"
        / ("trial_%s" % trial_number)
        / job["split_group"]
        / job["split_name"]
        / ("seed_%s" % seed)
    )
    metric_file = trial_root / ("final_results_%s.csv" % args.eval_split)
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
            str(args.embeddings_dir),
            "--out_dir",
            str(trial_root),
            "--task_name",
            "optuna_trial_%s_%s_%s_seed%s" % (trial_number, job["split_group"], job["split_name"], seed),
            "--sequence_col",
            args.sequence_col,
            "--smiles_col",
            args.smiles_col,
            "--target_col",
            args.target_col,
            "--lr",
            str(hparams["lr"]),
            "--weight_decay",
            str(hparams["weight_decay"]),
            "--step_size",
            str(hparams["step_size"]),
            "--gamma",
            str(hparams["gamma"]),
            "--batch_size",
            str(hparams["batch_size"]),
            "--num_epochs",
            str(args.num_epochs),
            "--scheduler",
            hparams["scheduler"],
            "--min_lr",
            str(hparams["min_lr"]),
            "--grad_clip",
            str(hparams["grad_clip"]),
            "--patience",
            str(hparams["patience"]),
            "--min_delta",
            str(hparams["min_delta"]),
            "--val_every",
            str(args.val_every),
            "--gat_dim",
            str(args.gat_dim),
            "--num_head",
            str(args.num_head),
            "--dropout",
            str(args.dropout),
            "--alpha",
            str(args.alpha),
            "--comp_dim",
            str(args.comp_dim),
            "--prot_dim",
            str(args.prot_dim),
            "--latent_dim",
            str(args.latent_dim),
            "--window",
            str(args.window),
            "--layer_cnn",
            str(args.layer_cnn),
            "--layer_out",
            str(args.layer_out),
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--bucket_size_multiplier",
            str(args.bucket_size_multiplier),
            "--pad_to_multiple",
            str(args.pad_to_multiple),
            "--seed",
            str(seed),
            "--monitor_metric",
            args.metric,
            "--amsgrad",
        ]
        if args.skip_smiles_validity_check:
            cmd.append("--skip_smiles_validity_check")
        if args.pin_memory:
            cmd.append("--pin_memory")
        else:
            cmd.append("--no_pin_memory")
        if args.persistent_workers:
            cmd.append("--persistent_workers")
        else:
            cmd.append("--no_persistent_workers")
        if args.preload_features:
            cmd.append("--preload_features")
        else:
            cmd.append("--no_preload_features")
        if args.torch_compile:
            cmd.append("--torch_compile")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    metrics = pd.read_csv(metric_file).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError("Metric `%s` not found in %s" % (args.metric, metric_file))
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune retraining-safe BACPI hyperparameters with Optuna.")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--base_root", type=str, default=None)
    parser.add_argument("--value_type", type=str, default="ki")
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--bucket_size_multiplier", type=int, default=50)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--preload_features", dest="preload_features", action="store_true")
    parser.add_argument("--no_preload_features", dest="preload_features", action="store_false")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--cache_overwrite_vocabs", action="store_true")
    parser.add_argument("--clean_splits_in_place", dest="clean_splits_in_place", action="store_true")
    parser.add_argument("--no_clean_splits_in_place", dest="clean_splits_in_place", action="store_false")
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "mae", "mse", "loss"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="bacpi_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")

    parser.add_argument("--gat_dim", type=int, default=50)
    parser.add_argument("--num_head", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--comp_dim", type=int, default=80)
    parser.add_argument("--prot_dim", type=int, default=80)
    parser.add_argument("--latent_dim", type=int, default=80)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--layer_cnn", type=int, default=3)
    parser.add_argument("--layer_out", type=int, default=3)
    parser.set_defaults(pin_memory=True, persistent_workers=True, preload_features=True, clean_splits_in_place=True)
    args = parser.parse_args()

    args.base_dir = resolve_base_dir(args.base_dir, args.base_root, args.value_type)
    args.embeddings_dir = Path(args.embeddings_dir).expanduser().resolve() if args.embeddings_dir else args.base_dir / "embeddings"
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = "sqlite:///%s" % (args.base_dir / "optuna_studies" / ("%s.db" % args.study_name))

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError("No split jobs found in %s" % args.base_dir)

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
        return float(sum(scores) / len(scores))

    study.optimize(objective, n_trials=args.n_trials)
    best_payload = {
        "study_name": args.study_name,
        "storage": args.storage,
        "value_type": args.value_type,
        "metric": args.metric,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_hparams": dict(study.best_params),
    }
    best_path = args.base_dir / "optuna_studies" / ("%s_best.json" % args.study_name)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, "w") as handle:
        json.dump(best_payload, handle, indent=2, sort_keys=True)
    print("Saved best Optuna payload to %s" % best_path, flush=True)


if __name__ == "__main__":
    main()
