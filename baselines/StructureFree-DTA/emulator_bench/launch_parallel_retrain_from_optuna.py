import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import optuna
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_SPLIT_GROUPS, REPO_ROOT, discover_split_jobs, normalize_threshold_args, summarize_seed_runs
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings


TRAIN_SCRIPT = REPO_ROOT / "emulator_bench" / "train_single_target_tvt.py"


def load_best_hparams(args):
    if args.hparams_json:
        with open(args.hparams_json, "r") as handle:
            payload = json.load(handle)
        return payload.get("best_hparams", payload)
    if not args.storage:
        raise ValueError("Provide either --hparams_json or --storage.")
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best = dict(study.best_params)
    best["batch_size_schedule"] = study.best_trial.user_attrs.get("batch_size_schedule")
    return best


def choose_hparam(raw, args, key, fallback):
    override = getattr(args, key)
    if override is not None:
        return override
    return raw.get(key, fallback)


def resolve_training_hparams(raw, args):
    batch_size = int(choose_hparam(raw, args, "batch_size", 64))
    batch_size_schedule = choose_hparam(raw, args, "batch_size_schedule", raw.get("batch_size_schedule") or f"1:{batch_size},20:512,60:2048")
    return {
        "batch_size": batch_size,
        "batch_size_schedule": str(batch_size_schedule),
        "lr": float(choose_hparam(raw, args, "lr", 5e-5)),
        "weight_decay": float(choose_hparam(raw, args, "weight_decay", 1e-5)),
        "beta1": float(choose_hparam(raw, args, "beta1", 0.9)),
        "beta2": float(choose_hparam(raw, args, "beta2", 0.999)),
        "eps": float(choose_hparam(raw, args, "eps", 1e-8)),
        "scheduler": str(choose_hparam(raw, args, "scheduler", "cosine")),
        "lr_decay_factor": float(choose_hparam(raw, args, "lr_decay_factor", 0.5)),
        "lr_decay_patience": int(choose_hparam(raw, args, "lr_decay_patience", 5)),
        "min_lr": float(choose_hparam(raw, args, "min_lr", 1e-7)),
        "lr_warmup_epochs": int(choose_hparam(raw, args, "lr_warmup_epochs", 10)),
        "lr_warmup_start_factor": float(choose_hparam(raw, args, "lr_warmup_start_factor", 0.1)),
        "loss_alpha": float(choose_hparam(raw, args, "loss_alpha", 0.5)),
        "clip_grad": float(choose_hparam(raw, args, "clip_grad", 1.0)),
        "patience": int(choose_hparam(raw, args, "patience", 20)),
        "min_delta": float(choose_hparam(raw, args, "min_delta", 0.0)),
        "amsgrad": bool(args.amsgrad or raw.get("amsgrad", False)),
    }


def build_experiments(jobs, seeds, output_root):
    experiments = []
    for job in jobs:
        for seed in seeds:
            experiments.append(
                {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "train_path": job["train_path"],
                    "val_path": job["val_path"],
                    "test_path": job["test_path"],
                    "seed": int(seed),
                    "run_dir": output_root / job["split_group"] / job["split_name"] / f"seed_{seed}",
                }
            )
    return experiments


def legacy_random_run_dir(exp, output_root):
    if exp["split_group"] == "random_splits_grouped_sequence":
        return output_root / "random_splits" / "random" / f"seed_{exp['seed']}"
    return None


def train_command(exp, args, hparams, device):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train_path",
        exp["train_path"],
        "--val_path",
        exp["val_path"],
        "--test_path",
        exp["test_path"],
        "--embeddings_dir",
        args.embeddings_dir,
        "--out_dir",
        str(exp["run_dir"]),
        "--task_name",
        f"{exp['split_group']}_{exp['split_name']}_seed{exp['seed']}",
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
        args.monitor_metric,
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--embedding_cache_items",
        str(args.embedding_cache_items),
        "--seed",
        str(exp["seed"]),
    ]
    for enabled, flag in [
        (args.pin_memory, "--pin_memory"),
        (args.persistent_workers, "--persistent_workers"),
        (args.preload_embeddings, "--preload_embeddings"),
        (args.torch_compile, "--torch_compile"),
        (hparams["amsgrad"], "--amsgrad"),
    ]:
        if enabled:
            cmd.append(flag)
    return cmd


def run_experiment(exp, args, hparams, gpu_id):
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / "retrain_from_optuna"
    legacy_run_dir = legacy_random_run_dir(exp, output_root)
    if legacy_run_dir is not None and (legacy_run_dir / "final_results_test.csv").exists() and not args.overwrite:
        return {
            "status": "skipped_legacy_random_exists",
            "gpu_id": str(gpu_id),
            "run_dir": str(legacy_run_dir),
            "split_group": exp["split_group"],
            "split_name": exp["split_name"],
            "difficulty": exp["difficulty"],
            "seed": exp["seed"],
        }

    exp["run_dir"].mkdir(parents=True, exist_ok=True)
    metric_path = exp["run_dir"] / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        status = "skipped_exists"
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = "cuda:0" if args.device.startswith("cuda") else args.device
        subprocess.run(train_command(exp, args, hparams, device), check=True, cwd=str(REPO_ROOT), env=env)
        status = "completed"
    return {
        "status": status,
        "gpu_id": str(gpu_id),
        "run_dir": str(exp["run_dir"]),
        "split_group": exp["split_group"],
        "split_name": exp["split_name"],
        "difficulty": exp["difficulty"],
        "seed": exp["seed"],
    }


def run_parallel(experiments, args, hparams):
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
                result = run_experiment(exp, args, hparams, gpu_id)
                result["slot_index"] = int(slot_index)
            except Exception as exc:
                result = {
                    "status": "failed",
                    "gpu_id": str(gpu_id),
                    "slot_index": int(slot_index),
                    "run_dir": str(exp["run_dir"]),
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
    parser = argparse.ArgumentParser(description="Retrain StructureFree-DTA splits in parallel from best Optuna hparams.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--output_root", type=str, default=None)
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
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--study_name", type=str, default="structurefree_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--hparams_json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--batch_size_schedule", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lr_decay_factor", type=float, default=None)
    parser.add_argument("--lr_decay_patience", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_start_factor", type=float, default=None)
    parser.add_argument("--loss_alpha", type=float, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--amsgrad", action="store_true")
    parser.add_argument("--monitor_metric", choices=["rmse", "mse", "mae", "loss", "r2_score", "pearson", "spearman", "ci"], default="rmse")
    args = parser.parse_args()

    if args.trials_per_gpu <= 0:
        raise ValueError("--trials_per_gpu must be positive")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    hparams = resolve_training_hparams(load_best_hparams(args), args)
    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")
    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / "retrain_from_optuna"
    output_root.mkdir(parents=True, exist_ok=True)
    results = run_parallel(build_experiments(jobs, args.seeds, output_root), args, hparams)

    summary_rows = []
    for result in results:
        row = dict(result)
        if result["status"] != "failed":
            run_dir = Path(result["run_dir"])
            for prefix in ["train", "val", "test"]:
                metrics_path = run_dir / f"final_results_{prefix}.csv"
                if metrics_path.exists():
                    for key, value in pd.read_csv(metrics_path).iloc[0].to_dict().items():
                        row[f"{prefix}_{key}"] = value
        summary_rows.append(row)
    runs_df = pd.DataFrame(summary_rows)
    runs_df.to_csv(output_root / "retrain_summary_runs.csv", index=False)
    metric_cols = [col for col in runs_df.columns if col.startswith("test_")]
    summarize_seed_runs(summary_rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(output_root / "retrain_summary_thresholds.csv", index=False)
    summarize_seed_runs(summary_rows, ["split_group"], metric_cols).to_csv(output_root / "retrain_summary_by_split_group.csv", index=False)


if __name__ == "__main__":
    main()
