import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from emulator_bench.common import REPO_ROOT, normalize_threshold_args
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings
from emulator_bench.tune_optuna import metric_direction, prepare_optuna_storage


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def split_trials(total_trials, num_workers):
    base = total_trials // num_workers
    remainder = total_trials % num_workers
    return [base + (1 if idx < remainder else 0) for idx in range(num_workers)]


def worker_cmd(args, worker_trials, worker_index):
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
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
        "--protein_batch_size",
        str(args.protein_batch_size),
        "--molecule_batch_size",
        str(args.molecule_batch_size),
        "--embedding_dtype",
        args.embedding_dtype,
        "--hidden_sizes",
        args.hidden_sizes,
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--val_every",
        str(args.val_every),
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--cache_device",
        "cuda:0" if args.cache_device.startswith("cuda") else args.cache_device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--embedding_cache_items",
        str(args.embedding_cache_items),
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
    for enabled, flag in [
        (args.persistent_workers, "--persistent_workers"),
        (args.pin_memory, "--pin_memory"),
        (args.preload_embeddings, "--preload_embeddings"),
        (args.torch_compile, "--torch_compile"),
        (args.overwrite_runs, "--overwrite_runs"),
    ]:
        if enabled:
            cmd.append(flag)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch parallel single-GPU Optuna workers for StructureFree-DTA.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
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
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="structurefree_optuna")
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--stagger_seconds", type=float, default=3.0)
    args = parser.parse_args()

    if args.trials_per_gpu <= 0:
        raise ValueError("--trials_per_gpu must be positive")
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    slots = [(str(gpu), slot) for gpu in args.gpus for slot in range(args.trials_per_gpu)]
    trial_counts = split_trials(args.n_trials, len(slots))
    processes = []
    try:
        for worker_index, ((gpu_id, slot_index), worker_trials) in enumerate(zip(slots, trial_counts)):
            if worker_trials <= 0:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd = worker_cmd(args, worker_trials, worker_index)
            print(f"Launching Optuna worker {worker_index} on GPU {gpu_id} slot {slot_index} for {worker_trials} trials", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            processes.append((gpu_id, slot_index, worker_trials, proc))
            if worker_index < len(slots) - 1 and args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)
        failed = False
        for gpu_id, slot_index, worker_trials, proc in processes:
            return_code = proc.wait()
            if return_code != 0:
                failed = True
                print(f"Worker on GPU {gpu_id} slot {slot_index} failed after {worker_trials} trials with exit code {return_code}", flush=True)
        if failed:
            raise RuntimeError("One or more parallel Optuna workers failed.")
    finally:
        for _gpu_id, _slot_index, _worker_trials, proc in processes:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
