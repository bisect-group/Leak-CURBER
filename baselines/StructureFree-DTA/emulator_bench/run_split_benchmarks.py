import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
    REPO_ROOT,
    discover_split_jobs,
    normalize_threshold_args,
    split_sizes,
    summarize_seed_runs,
)


CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def maybe_cache_embeddings(args):
    if args.skip_cache:
        return
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--protein_model_name",
        args.protein_model_name,
        "--molecule_model_name",
        args.molecule_model_name,
        "--max_protein_length",
        str(args.max_protein_length),
        "--max_molecule_length",
        str(args.max_molecule_length),
        "--protein_batch_size",
        str(args.protein_batch_size),
        "--molecule_batch_size",
        str(args.molecule_batch_size),
        "--embedding_dtype",
        args.embedding_dtype,
        "--device",
        args.cache_device,
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.cache_overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def maybe_load_hparams(args):
    if not args.hparams_json:
        return args
    with open(args.hparams_json, "r") as handle:
        payload = json.load(handle)
    hparams = payload.get("best_hparams", payload)
    for key in [
        "batch_size",
        "batch_size_schedule",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "scheduler",
        "lr_decay_factor",
        "lr_decay_patience",
        "min_lr",
        "lr_warmup_epochs",
        "lr_warmup_start_factor",
        "loss_alpha",
        "clip_grad",
        "patience",
        "min_delta",
    ]:
        if key in hparams:
            setattr(args, key, hparams[key])
    if hparams.get("amsgrad"):
        args.amsgrad = True
    return args


def train_one(job, seed, args):
    result_root = Path(job["root_dir"]) / args.results_dirname / f"seed_{seed}"
    metric_path = result_root / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return result_root
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
        str(result_root),
        "--task_name",
        f"{job['split_group']}_{job['split_name']}_seed{seed}",
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
        str(args.batch_size),
        "--batch_size_schedule",
        str(args.batch_size_schedule),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--beta1",
        str(args.beta1),
        "--beta2",
        str(args.beta2),
        "--eps",
        str(args.eps),
        "--scheduler",
        args.scheduler,
        "--lr_decay_factor",
        str(args.lr_decay_factor),
        "--lr_decay_patience",
        str(args.lr_decay_patience),
        "--min_lr",
        str(args.min_lr),
        "--lr_warmup_epochs",
        str(args.lr_warmup_epochs),
        "--lr_warmup_start_factor",
        str(args.lr_warmup_start_factor),
        "--loss_alpha",
        str(args.loss_alpha),
        "--clip_grad",
        str(args.clip_grad),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--val_every",
        str(args.val_every),
        "--monitor_metric",
        args.monitor_metric,
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
        (args.amsgrad, "--amsgrad"),
    ]:
        if enabled:
            cmd.append(flag)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return result_root


def main():
    parser = argparse.ArgumentParser(description="Run StructureFree-DTA emulator bench across Leak-CURBER split groups.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--molecule_model_name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--max_protein_length", type=int, default=1024)
    parser.add_argument("--max_molecule_length", type=int, default=128)
    parser.add_argument("--protein_batch_size", type=int, default=64)
    parser.add_argument("--molecule_batch_size", type=int, default=256)
    parser.add_argument("--embedding_dtype", choices=["float16", "float32"], default="float16")

    parser.add_argument("--hidden_sizes", type=str, default="1024,768,512,256,1")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_schedule", type=str, default="1:64,20:512,60:2048")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--lr_warmup_epochs", type=int, default=10)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.1)
    parser.add_argument("--loss_alpha", type=float, default=0.5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--monitor_metric", choices=["rmse", "mse", "mae", "loss", "r2_score", "pearson", "spearman", "ci"], default="rmse")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_embeddings", action="store_true")
    parser.add_argument("--embedding_cache_items", type=int, default=20000)
    parser.add_argument("--torch_compile", action="store_true")
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args = maybe_load_hparams(args)
    maybe_cache_embeddings(args)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    summary_rows = []
    for job in tqdm(jobs, desc="Benchmark jobs", unit="job"):
        for seed in args.seeds:
            out_dir = train_one(job, seed, args)
            test_metrics = pd.read_csv(out_dir / "final_results_test.csv").iloc[0].to_dict()
            val_metrics = pd.read_csv(out_dir / "final_results_val.csv").iloc[0].to_dict()
            row = {
                "split_group": job["split_group"],
                "split_name": job["split_name"],
                "difficulty": job["difficulty"],
                "seed": int(seed),
                "run_dir": str(out_dir),
            }
            row.update(split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])))
            for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            summary_rows.append(row)

    base_dir = Path(args.base_dir)
    pd.DataFrame(summary_rows).to_csv(base_dir / "structurefree_summary_runs.csv", index=False)
    metric_cols = [col for col in pd.DataFrame(summary_rows).columns if col.startswith("test_")]
    summarize_seed_runs(summary_rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(base_dir / "structurefree_summary_thresholds.csv", index=False)
    summarize_seed_runs(summary_rows, ["split_group"], metric_cols).to_csv(base_dir / "structurefree_summary_by_split_group.csv", index=False)


if __name__ == "__main__":
    main()
