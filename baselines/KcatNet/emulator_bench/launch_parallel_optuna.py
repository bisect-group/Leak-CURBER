import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.tune_optuna import (
    _metric_direction,
    maybe_cache_embeddings,
    prepare_optuna_storage,
)
from emulator_bench.common import normalize_threshold_args


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def _split_trials(total_trials: int, num_workers: int) -> list[int]:
    base = total_trials // num_workers
    remainder = total_trials % num_workers
    return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]


def _resolve_total_trials(args) -> int:
    if args.trials_per_gpu is not None:
        if args.trials_per_gpu <= 0:
            raise ValueError("--trials_per_gpu must be a positive integer")
        return args.trials_per_gpu * len(args.gpus)
    if args.n_trials is None or args.n_trials <= 0:
        raise ValueError("Provide a positive --n_trials or --trials_per_gpu")
    return args.n_trials


def _build_worker_cmd(args, gpu_id: str, worker_index: int, worker_trials: int) -> list[str]:
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--config_path",
        args.config_path,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--epochs",
        str(args.epochs),
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--cache_device",
        "cuda:0" if args.cache_device.startswith("cuda") else args.cache_device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--val_every",
        str(args.val_every),
        "--protein_cache_items",
        str(args.protein_cache_items),
        "--metric",
        args.metric,
        "--eval_split",
        args.eval_split,
        "--n_trials",
        str(worker_trials),
        "--sampler_seed",
        str(args.sampler_seed + worker_index),
        "--study_name",
        args.study_name,
        "--storage",
        args.storage,
        "--skip_cache",
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.persistent_workers:
        cmd.append("--persistent_workers")
    if args.pin_memory:
        cmd.append("--pin_memory")
    if args.preload_proteins:
        cmd.append("--preload_proteins")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    if args.overwrite_runs:
        cmd.append("--overwrite_runs")
    if args.prot_t5_model:
        cmd.extend(["--prot_t5_model", args.prot_t5_model])
    if args.protein_dtype:
        cmd.extend(["--protein_dtype", args.protein_dtype])
    cmd.extend(["--prot_t5_max_residues", str(args.prot_t5_max_residues)])
    cmd.extend(["--prot_t5_max_seq_len", str(args.prot_t5_max_seq_len)])
    cmd.extend(["--prot_t5_max_batch", str(args.prot_t5_max_batch)])
    cmd.extend(["--ligand_batch_size", str(args.ligand_batch_size)])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch multiple single-GPU Optuna workers that share one study/storage.")
    parser.add_argument("--gpus", nargs="+", required=True, help="Visible GPU ids to assign one worker each, e.g. --gpus 0 1")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(REPO_ROOT / "config_KcatNet.json"))
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument(
        "--all_thresholds",
        action="store_true",
        help="Use every discovered threshold under the requested split groups.",
    )
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=256)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--prot_t5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--prot_t5_max_residues", type=int, default=4000)
    parser.add_argument("--prot_t5_max_seq_len", type=int, default=1000)
    parser.add_argument("--prot_t5_max_batch", type=int, default=32)
    parser.add_argument("--ligand_batch_size", type=int, default=256)
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "pearson", "spearman", "r2_score", "mae", "mse"])
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=None, help="Total trials to distribute across all GPU workers.")
    parser.add_argument("--trials_per_gpu", type=int, default=None, help="Trials assigned to each GPU worker.")
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="kcatnet_optuna")
    parser.add_argument("--storage", type=str, required=True, help="Shared Optuna storage. Required for parallel workers.")
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--stagger_seconds", type=float, default=3.0, help="Delay between worker launches to reduce DB startup contention.")
    args = parser.parse_args()

    if not args.gpus:
        raise ValueError("At least one GPU id must be provided via --gpus")

    args.thresholds = None if args.all_thresholds else normalize_threshold_args(args.thresholds, args.threshold)
    if args.device.startswith("cuda") and len(args.gpus) < 1:
        raise ValueError("CUDA device requested but no GPU ids were provided")
    total_trials = _resolve_total_trials(args)

    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    optuna.create_study(
        direction=_metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    trial_splits = _split_trials(total_trials, len(args.gpus))
    workers = [(gpu_id, worker_trials) for gpu_id, worker_trials in zip(args.gpus, trial_splits) if worker_trials > 0]
    if not workers:
        raise RuntimeError("No worker received any trials. Increase --n_trials/--trials_per_gpu or reduce the number of --gpus.")

    procs: list[tuple[str, int, subprocess.Popen]] = []
    try:
        for worker_index, (gpu_id, worker_trials) in enumerate(workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd = _build_worker_cmd(args, str(gpu_id), worker_index, worker_trials)
            print(f"Launching worker {worker_index} on GPU {gpu_id} for {worker_trials} trials")
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            procs.append((str(gpu_id), worker_trials, proc))
            if worker_index < len(workers) - 1 and args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)

        failed = False
        for gpu_id, worker_trials, proc in procs:
            return_code = proc.wait()
            if return_code != 0:
                failed = True
                print(f"Worker on GPU {gpu_id} failed after being assigned {worker_trials} trials with exit code {return_code}")
        if failed:
            raise RuntimeError("One or more Optuna worker processes failed")
    finally:
        for _, _, proc in procs:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
