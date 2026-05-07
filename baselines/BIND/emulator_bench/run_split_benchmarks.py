import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HIDDEN_STATE_INDICES,
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_VALUE_TYPES,
    discover_split_jobs,
    normalize_threshold_args,
    split_sizes,
    summarize_seed_runs,
)

CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"
TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_regression_tvt.py"


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
        "--device",
        args.cache_device,
        "--protein_model_name",
        args.protein_model_name,
        "--hidden_state_indices",
        *[str(index) for index in args.hidden_state_indices],
        "--protein_length_cap",
        str(args.protein_length_cap),
        "--long_sequence_strategy",
        args.long_sequence_strategy,
        "--max_residues",
        str(args.max_residues),
        "--max_batch",
        str(args.max_batch),
        "--sliding_window_stride",
        str(args.sliding_window_stride),
        "--protein_dtype",
        args.protein_dtype,
    ]
    value_types = getattr(args, "value_types", None)
    if value_types is None and getattr(args, "value_type", None):
        value_types = [args.value_type]
    if value_types:
        cmd.extend(["--value_types", *value_types])
    split_groups = getattr(args, "split_groups", None)
    if split_groups:
        cmd.extend(["--split_groups", *split_groups])
    thresholds = getattr(args, "thresholds", None)
    if thresholds:
        cmd.extend(["--thresholds", *thresholds])
    if getattr(args, "cache_overwrite", False):
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
        "grad_accumulation_steps",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "amsgrad",
        "scheduler",
        "lr_decay_factor",
        "lr_decay_patience",
        "min_lr",
        "clip_grad",
        "patience",
        "min_delta",
    ]:
        if key in hparams:
            setattr(args, key, hparams[key])
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
        "--value_type",
        job["value_type"],
        "--target_head",
        job["value_type"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(result_root),
        "--task_name",
        f"{job['value_type']}_{job['split_group']}_{job['split_name']}_seed{seed}",
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
        str(args.batch_size),
        "--grad_accumulation_steps",
        str(args.grad_accumulation_steps),
        "--epochs",
        str(args.epochs),
        "--train_budget_mode",
        args.train_budget_mode,
        "--paper_effective_epochs",
        str(args.paper_effective_epochs),
        "--max_optimizer_steps",
        str(args.max_optimizer_steps),
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
        "--clip_grad",
        str(args.clip_grad),
        "--huber_delta",
        str(args.huber_delta),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--monitor_metric",
        args.monitor_metric,
        "--validate_every_epochs",
        str(args.validate_every_epochs),
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
        "--results_dirname",
        args.results_dirname,
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
    if args.amsgrad:
        cmd.append("--amsgrad")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return result_root


def train_one_on_device(job, seed, args, gpu_id=None):
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
        "--value_type",
        job["value_type"],
        "--target_head",
        job["value_type"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(result_root),
        "--task_name",
        f"{job['value_type']}_{job['split_group']}_{job['split_name']}_seed{seed}",
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
        str(args.batch_size),
        "--grad_accumulation_steps",
        str(args.grad_accumulation_steps),
        "--epochs",
        str(args.epochs),
        "--train_budget_mode",
        args.train_budget_mode,
        "--paper_effective_epochs",
        str(args.paper_effective_epochs),
        "--max_optimizer_steps",
        str(args.max_optimizer_steps),
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
        "--clip_grad",
        str(args.clip_grad),
        "--huber_delta",
        str(args.huber_delta),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--monitor_metric",
        args.monitor_metric,
        "--validate_every_epochs",
        str(args.validate_every_epochs),
        "--device",
        "cuda:0" if gpu_id is not None and str(args.device).startswith("cuda") else args.device,
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
        "--results_dirname",
        args.results_dirname,
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
    if args.amsgrad:
        cmd.append("--amsgrad")
    env = os.environ.copy()
    if gpu_id is not None and str(args.device).startswith("cuda"):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return result_root


def _run_parallel_jobs(jobs, args):
    experiments = []
    for job_index, job in enumerate(jobs, start=1):
        for seed in args.seeds:
            experiments.append((job_index, len(jobs), job, seed))

    work_queue = queue.Queue()
    for item in experiments:
        work_queue.put(item)

    rows = []
    rows_lock = threading.Lock()

    def worker(gpu_id, slot_index):
        while True:
            try:
                job_index, total_jobs, job, seed = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                print(
                    f"[GPU {gpu_id} slot {slot_index}] Running job {job_index}/{total_jobs}: "
                    f"{job['value_type']} | {job['split_group']} | {job['split_name']} | seed {seed}",
                    flush=True,
                )
                result_root = train_one_on_device(job, seed, args, gpu_id=gpu_id)
                test_metrics = pd.read_csv(result_root / "final_results_test.csv").iloc[0].to_dict()
                row = {
                    "value_type": job["value_type"],
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "seed": int(seed),
                    "gpu_id": str(gpu_id),
                    "slot_index": int(slot_index),
                    "run_dir": str(result_root),
                    **split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])),
                    **test_metrics,
                }
            except Exception as exc:
                row = {
                    "value_type": job["value_type"],
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "seed": int(seed),
                    "gpu_id": str(gpu_id),
                    "slot_index": int(slot_index),
                    "status": "failed",
                    "error": str(exc),
                }
            finally:
                work_queue.task_done()
            with rows_lock:
                rows.append(row)

    threads = []
    for gpu_id in args.gpus:
        for slot_index in range(int(args.runs_per_gpu)):
            thread = threading.Thread(target=worker, args=(str(gpu_id), slot_index), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run the BIND emulator bench across Leak-CURBER split groups.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--value_types", nargs="+", default=DEFAULT_VALUE_TYPES)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")
    parser.add_argument("--max_residues", type=int, default=4096)
    parser.add_argument("--max_batch", type=int, default=4)
    parser.add_argument("--sliding_window_stride", type=int, default=896)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")

    parser.add_argument("--cross_attention_dropout", type=float, default=0.1)
    parser.add_argument("--leaky_relu_negative_slope", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--train_budget_mode", choices=["paper_steps", "fixed_epochs", "fixed_optimizer_steps"], default="fixed_epochs")
    parser.add_argument("--paper_effective_epochs", type=float, default=10.365942049524907)
    parser.add_argument("--max_optimizer_steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--lr_decay_patience", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--huber_delta", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--monitor_metric", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"], default="rmse")
    parser.add_argument("--validate_every_epochs", type=int, default=1)
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
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--runs_per_gpu", type=int, default=1)
    args = parser.parse_args()

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args = maybe_load_hparams(args)

    maybe_cache_embeddings(args)
    jobs = discover_split_jobs(Path(args.base_dir), value_types=args.value_types, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    total_runs = len(jobs) * len(args.seeds)
    print(
        f"Discovered {len(jobs)} split jobs across {len(args.value_types)} value types. "
        f"Running {total_runs} total job/seed combinations.",
        flush=True,
    )

    if args.gpus:
        rows = _run_parallel_jobs(jobs, args)
    else:
        rows = []
        for job_index, job in enumerate(tqdm(jobs, desc="Split jobs", unit="job"), start=1):
            for seed in args.seeds:
                print(
                    f"Running job {job_index}/{len(jobs)}: "
                    f"{job['value_type']} | {job['split_group']} | {job['split_name']} | seed {seed}",
                    flush=True,
                )
                result_root = train_one(job, seed, args)
                test_metrics = pd.read_csv(result_root / "final_results_test.csv").iloc[0].to_dict()
                rows.append(
                    {
                        "value_type": job["value_type"],
                        "split_group": job["split_group"],
                        "split_name": job["split_name"],
                        "difficulty": job["difficulty"],
                        "seed": int(seed),
                        "run_dir": str(result_root),
                        **split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])),
                        **test_metrics,
                    }
                )

    summary_dir = Path(args.base_dir) / "bench_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(summary_dir / "all_seed_runs.csv", index=False)

    summary_df = summarize_seed_runs(
        rows,
        group_cols=["value_type", "split_group", "split_name", "difficulty"],
        metric_cols=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"],
    )
    summary_df.to_csv(summary_dir / "summary_by_split.csv", index=False)
    print(f"Saved aggregate results to {summary_dir}")


if __name__ == "__main__":
    main()
