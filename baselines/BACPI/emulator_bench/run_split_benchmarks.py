import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    normalize_threshold_args,
    resolve_base_dir,
    split_sizes,
)
CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    print("Preparing BACPI caches at %s" % args.embeddings_dir, flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        str(args.base_dir),
        "--embeddings_dir",
        str(args.embeddings_dir),
        "--value_type",
        *[str(value) for value in (args.value_type if isinstance(args.value_type, (list, tuple)) else [args.value_type])],
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--radius",
        str(args.radius),
        "--ngram",
        str(args.ngram),
        "--fp_radius",
        str(args.fp_radius),
        "--nbits",
        str(args.nbits),
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.cache_overwrite:
        cmd.append("--overwrite")
    if args.cache_overwrite_vocabs:
        cmd.append("--overwrite_vocabs")
    if getattr(args, "clean_splits_in_place", True):
        cmd.append("--clean_splits_in_place")
    else:
        cmd.append("--no_clean_splits_in_place")
    if getattr(args, "skip_smiles_validity_check", False):
        cmd.append("--skip_smiles_validity_check")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def maybe_load_hparams(args):
    if not args.hparams_json:
        return args
    with open(args.hparams_json, "r") as handle:
        payload = json.load(handle)
    hparams = payload.get("best_hparams", payload.get("defaults", payload))
    for key in ["batch_size", "lr", "weight_decay", "step_size", "gamma", "grad_clip", "patience", "min_delta", "scheduler", "min_lr", "amsgrad"]:
        if key in hparams:
            setattr(args, key, hparams[key])
    return args


def train_one(job, seed, args):
    result_root = Path(job["root_dir"]) / args.results_dirname / ("seed_%s" % seed)
    metric_path = result_root / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return result_root
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
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
        str(result_root),
        "--task_name",
        "%s_%s_seed%s" % (job["split_group"], job["split_name"], seed),
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--step_size",
        str(args.step_size),
        "--gamma",
        str(args.gamma),
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--scheduler",
        args.scheduler,
        "--min_lr",
        str(args.min_lr),
        "--grad_clip",
        str(args.grad_clip),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
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
        args.monitor_metric,
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
    if args.amsgrad:
        cmd.append("--amsgrad")
    else:
        cmd.append("--no_amsgrad")
    if args.torch_compile:
        cmd.append("--torch_compile")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return result_root


def aggregate_summary(run_rows):
    frame = pd.DataFrame(run_rows)
    if frame.empty:
        return frame
    numeric_cols = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    grouped = frame.groupby(["split_group", "split_name", "difficulty"], dropna=False)[numeric_cols].agg(["mean", "std"])
    grouped.columns = ["%s_%s" % (column, stat) for column, stat in grouped.columns]
    return grouped.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Run the BACPI emulator bench across discovered split jobs.")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--base_root", type=str, default=None)
    parser.add_argument("--value_type", type=str, default="ki")
    parser.add_argument("--embeddings_dir", type=str, default=None)
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--split_groups", nargs="+", default=DEFAULT_SPLIT_GROUPS)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--cache_overwrite_vocabs", action="store_true")
    parser.add_argument("--clean_splits_in_place", dest="clean_splits_in_place", action="store_true")
    parser.add_argument("--no_clean_splits_in_place", dest="clean_splits_in_place", action="store_false")
    parser.add_argument("--skip_smiles_validity_check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--scheduler", choices=["step", "cosine", "none"], default="step")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_every", type=int, default=1)

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

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--bucket_size_multiplier", type=int, default=50)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--monitor_metric", choices=["rmse", "pearson", "spearman", "mae", "mse", "loss"], default="rmse")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--preload_features", dest="preload_features", action="store_true")
    parser.add_argument("--no_preload_features", dest="preload_features", action="store_false")
    parser.add_argument("--amsgrad", dest="amsgrad", action="store_true")
    parser.add_argument("--no_amsgrad", dest="amsgrad", action="store_false")
    parser.set_defaults(pin_memory=True, persistent_workers=True, preload_features=True, amsgrad=True, clean_splits_in_place=True)
    args = parser.parse_args()

    args.base_dir = resolve_base_dir(args.base_dir, args.base_root, args.value_type)
    args.embeddings_dir = Path(args.embeddings_dir).expanduser().resolve() if args.embeddings_dir else args.base_dir / "embeddings"
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args = maybe_load_hparams(args)
    print("Running BACPI split benchmarks from %s" % args.base_dir, flush=True)
    maybe_cache_embeddings(args)

    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError("No split jobs found in %s" % args.base_dir)

    run_dirs = []
    run_rows = []
    for job in jobs:
        for seed in args.seeds:
            run_dir = train_one(job, seed, args)
            run_dirs.append(run_dir)
            metrics = pd.read_csv(Path(run_dir) / "final_results_test.csv").iloc[0].to_dict()
            row = {
                "split_group": job["split_group"],
                "split_name": job["split_name"],
                "difficulty": job["difficulty"],
                "seed": seed,
                "run_dir": str(run_dir),
            }
            row.update(split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])))
            row.update(metrics)
            run_rows.append(row)

    runs_frame = pd.DataFrame(run_rows)
    runs_path = args.base_dir / "bacpi_summary_runs.csv"
    runs_frame.to_csv(runs_path, index=False)
    summary_frame = aggregate_summary(run_rows)
    summary_path = args.base_dir / "bacpi_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    print("Saved run summary to %s" % runs_path, flush=True)
    print("Saved aggregate summary to %s" % summary_path, flush=True)


if __name__ == "__main__":
    main()
