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

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    normalize_threshold_args,
)
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
            "Optuna storage exists but does not contain a valid Optuna schema: "
            "%s. Use a new --storage path or rerun with --reset_storage." % db_path
        )


def suggest_hparams(trial, args):
    """Only optimization-side hyperparameters; architecture stays fixed to preserve result validity."""
    batch_size = int(args.batch_size) if args.batch_size is not None else trial.suggest_categorical("batch_size", [64, 128, 192, 256, 384])
    return {
        "batch_size": batch_size,
        "lr": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "cosine_t0": trial.suggest_categorical("cosine_t0", [5, 8, 10, 15, 20]),
        "clip_grad": trial.suggest_categorical("clip_grad", [0.5, 1.0, 2.0, 5.0]),
        "scheduler": "cosine_warm_restarts",
        "amsgrad": True,
    }


def run_trial_job(job, seed, hparams, args, trial_number):
    trial_root = (
        Path(job["root_dir"])
        / "emosaic_optuna_runs"
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
            args.embeddings_dir,
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
            "--num_layer",
            str(args.num_layer),
            "--emb_dim",
            str(args.emb_dim),
            "--gnn_type",
            args.gnn_type,
            "--max_length",
            str(args.max_length),
            "--prot_input_dim",
            str(args.prot_input_dim),
            "--dropout",
            str(args.dropout),
            "--batch_size",
            str(hparams["batch_size"]),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(hparams["lr"]),
            "--weight_decay",
            str(hparams["weight_decay"]),
            "--scheduler",
            hparams["scheduler"],
            "--cosine_t0",
            str(hparams["cosine_t0"]),
            "--clip_grad",
            str(hparams["clip_grad"]),
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
            "--protein_cache_items",
            str(args.protein_cache_items),
            "--seed",
            str(seed),
        ]
        if hparams.get("amsgrad", True):
            cmd.append("--amsgrad")
        else:
            cmd.append("--no-amsgrad")
        if args.pin_memory:
            cmd.append("--pin_memory")
        if args.persistent_workers:
            cmd.append("--persistent_workers")
        else:
            cmd.append("--no-persistent_workers")
        if args.preload_proteins:
            cmd.append("--preload_proteins")
        if args.lazy_ligands:
            cmd.append("--lazy_ligands")
        if args.torch_compile:
            cmd.append("--torch_compile")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    metrics = pd.read_csv(metric_file).iloc[0].to_dict()
    if args.metric not in metrics:
        raise RuntimeError("Metric `%s` not found in %s" % (args.metric, metric_file))
    return float(metrics[args.metric])


def main():
    parser = argparse.ArgumentParser(description="Tune retraining-safe hparams of the eMOSAIC bench with Optuna.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--max_length", type=int, default=700)
    parser.add_argument("--prot_input_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--long_seq_stride", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="emosaic_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    if args.storage is None:
        args.storage = "sqlite:///%s" % (Path(args.base_dir) / "optuna_studies" / (args.study_name + ".db"))

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
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

    out_dir = Path(args.base_dir) / "optuna_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / ("%s_trials.csv" % args.study_name), index=False)
    best_json = out_dir / ("%s_best_hparams.json" % args.study_name)
    with open(best_json, "w") as handle:
        json.dump(
            {
                "study_name": args.study_name,
                "storage": args.storage,
                "direction": study.direction.name.lower(),
                "best_trial_number": int(study.best_trial.number),
                "best_value": float(study.best_value),
                "best_hparams": dict(study.best_params),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
