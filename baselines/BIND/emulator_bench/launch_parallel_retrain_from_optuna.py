import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HIDDEN_STATE_INDICES,
    DEFAULT_VALUE_TYPES,
    discover_split_jobs,
    normalize_threshold_args,
    summarize_seed_runs,
)
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_regression_tvt.py"


def default_hparams_json(base_dir: str, value_type: str) -> Path:
    return Path(base_dir) / value_type / "optuna_studies" / f"bind_{value_type}_optuna_best_hparams.json"


def load_best_hparams(args, value_type: str):
    if args.hparams_json_template:
        hparams_path = Path(args.hparams_json_template.format(value_type=value_type))
    else:
        hparams_path = default_hparams_json(args.base_dir, value_type)
    if not hparams_path.exists():
        raise FileNotFoundError(f"Best-hparams file not found for {value_type}: {hparams_path}")
    with open(hparams_path, "r") as handle:
        payload = json.load(handle)
    return payload.get("best_hparams", payload)


def resolve_training_hparams(raw_hparams, args):
    def choose(key, fallback):
        override = getattr(args, key)
        if override is not None:
            return override
        return raw_hparams.get(key, fallback)

    return {
        "batch_size": int(choose("batch_size", 64)),
        "grad_accumulation_steps": int(choose("grad_accumulation_steps", 4)),
        "lr": float(choose("lr", 1e-4)),
        "weight_decay": float(choose("weight_decay", 1e-3)),
        "beta1": float(choose("beta1", 0.9)),
        "beta2": float(choose("beta2", 0.999)),
        "eps": float(choose("eps", 1e-8)),
        "scheduler": str(choose("scheduler", "cosine")),
        "lr_decay_factor": float(choose("lr_decay_factor", 0.5)),
        "lr_decay_patience": int(choose("lr_decay_patience", 2)),
        "min_lr": float(choose("min_lr", 0.0)),
        "clip_grad": float(choose("clip_grad", 0.0)),
        "patience": int(choose("patience", 0)),
        "min_delta": float(choose("min_delta", 0.0)),
        "amsgrad": bool(raw_hparams.get("amsgrad", False) or args.amsgrad),
    }


def build_experiments(jobs, seeds, output_dirname, per_value_hparams):
    experiments = []
    for job in jobs:
        for seed in seeds:
            run_dir = Path(job["value_root"]) / output_dirname / job["split_group"] / job["split_name"] / f"seed_{seed}"
            experiments.append(
                {
                    "value_type": job["value_type"],
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "train_path": job["train_path"],
                    "val_path": job["val_path"],
                    "test_path": job["test_path"],
                    "seed": int(seed),
                    "run_dir": run_dir,
                    "hparams": per_value_hparams[job["value_type"]],
                }
            )
    return experiments


def train_command(exp, args, device):
    hparams = exp["hparams"]
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        exp["train_path"],
        "--val_path",
        exp["val_path"],
        "--test_path",
        exp["test_path"],
        "--value_type",
        exp["value_type"],
        "--target_head",
        exp["value_type"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(exp["run_dir"]),
        "--task_name",
        f"{exp['value_type']}_{exp['split_group']}_{exp['split_name']}_seed{exp['seed']}",
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
        args.monitor_metric,
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--sharing_strategy",
        args.sharing_strategy,
        "--protein_cache_items",
        str(args.protein_cache_items),
        "--seed",
        str(exp["seed"]),
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
    if hparams["amsgrad"]:
        cmd.append("--amsgrad")
    return cmd


def run_experiment(exp, args, gpu_id):
    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    metric_path = exp["run_dir"] / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return {
            "status": "skipped_exists",
            "gpu_id": str(gpu_id),
            "run_dir": str(exp["run_dir"]),
            "value_type": exp["value_type"],
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0" if args.device.startswith("cuda") else args.device
    cmd = train_command(exp, args, device)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return {
        "status": "completed",
        "gpu_id": str(gpu_id),
        "run_dir": str(exp["run_dir"]),
        "value_type": exp["value_type"],
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
    }


def run_parallel(experiments, args):
    work_queue = queue.Queue()
    for exp in experiments:
        work_queue.put(exp)

    results = []
    result_lock = threading.Lock()

    def worker(gpu_id, slot_index):
        while True:
            try:
                exp = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_experiment(exp, args, gpu_id)
                result["slot_index"] = int(slot_index)
            except Exception as exc:
                result = {
                    "status": "failed",
                    "gpu_id": str(gpu_id),
                    "slot_index": int(slot_index),
                    "run_dir": str(exp["run_dir"]),
                    "value_type": exp["value_type"],
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                    "error": str(exc),
                }
            with result_lock:
                results.append(result)
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for slot_index in range(args.trials_per_gpu):
            thread = threading.Thread(target=worker, args=(str(gpu_id), slot_index), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrain all requested BIND split jobs in parallel from the best Optuna hyperparameters.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--value_types", nargs="+", default=DEFAULT_VALUE_TYPES)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--output_dirname", type=str, default="retrain_from_optuna")
    parser.add_argument("--hparams_json_template", type=str, default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--cross_attention_dropout", type=float, default=0.1)
    parser.add_argument("--leaky_relu_negative_slope", type=float, default=0.05)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")
    parser.add_argument("--max_residues", type=int, default=4096)
    parser.add_argument("--max_batch", type=int, default=4)
    parser.add_argument("--sliding_window_stride", type=int, default=896)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--huber_delta", type=float, default=2.0)
    parser.add_argument("--monitor_metric", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"], default="rmse")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accumulation_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lr_decay_factor", type=float, default=None)
    parser.add_argument("--lr_decay_patience", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--amsgrad", action="store_true")
    args = parser.parse_args()

    args.hidden_state_indices = [int(index) for index in args.hidden_state_indices]
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)

    maybe_cache_embeddings(args)
    jobs = discover_split_jobs(
        Path(args.base_dir),
        value_types=args.value_types,
        split_groups=args.split_groups,
        thresholds=args.thresholds,
    )
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    per_value_hparams = {
        value_type: resolve_training_hparams(load_best_hparams(args, value_type), args)
        for value_type in sorted({job["value_type"] for job in jobs})
    }
    experiments = build_experiments(jobs, seeds=args.seeds, output_dirname=args.output_dirname, per_value_hparams=per_value_hparams)
    results = run_parallel(experiments, args)

    summary_dir = Path(args.base_dir) / "bench_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs_df = pd.DataFrame(results)
    runs_df.to_csv(summary_dir / "parallel_retrain_runs.csv", index=False)

    metric_rows = []
    for exp in experiments:
        metric_path = exp["run_dir"] / "final_results_test.csv"
        if metric_path.exists():
            metrics = pd.read_csv(metric_path).iloc[0].to_dict()
            metric_rows.append(
                {
                    "value_type": exp["value_type"],
                    "split_group": exp["split_group"],
                    "split_name": exp["split_name"],
                    "difficulty": exp["difficulty"],
                    "seed": exp["seed"],
                    **metrics,
                }
            )

    if metric_rows:
        summary_df = summarize_seed_runs(
            metric_rows,
            group_cols=["value_type", "split_group", "split_name", "difficulty"],
            metric_cols=["rmse", "pearson", "spearman", "r2_score", "mae", "mse", "loss"],
        )
        summary_df.to_csv(summary_dir / "parallel_retrain_summary.csv", index=False)
    print(f"Saved parallel retrain summaries to {summary_dir}")


if __name__ == "__main__":
    main()
