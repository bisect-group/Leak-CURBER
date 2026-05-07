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

from emulator_bench.run_split_benchmarks import maybe_cache_embeddings
from emulator_bench.tune_optuna import metric_direction, prepare_optuna_storage


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def split_trials(total_trials, num_workers):
    base = total_trials // num_workers
    remainder = total_trials % num_workers
    return [base + (1 if index < remainder else 0) for index in range(num_workers)]


def worker_cmd(args, worker_trials, worker_index):
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        str(args.base_dir),
        "--embeddings_dir",
        str(args.embeddings_dir),
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--target_col",
        args.target_col,
        "--radius",
        str(args.radius),
        "--ngram",
        str(args.ngram),
        "--fp_radius",
        str(args.fp_radius),
        "--nbits",
        str(args.nbits),
        "--num_epochs",
        str(args.num_epochs),
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--bucket_size_multiplier",
        str(args.bucket_size_multiplier),
        "--pad_to_multiple",
        str(args.pad_to_multiple),
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
    if args.overwrite_runs:
        cmd.append("--overwrite_runs")
    if args.clean_splits_in_place:
        cmd.append("--clean_splits_in_place")
    else:
        cmd.append("--no_clean_splits_in_place")
    if args.skip_smiles_validity_check:
        cmd.append("--skip_smiles_validity_check")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch parallel single-GPU Optuna workers for the BACPI bench.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--split_groups", nargs="+", default=None)
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
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default="bacpi_optuna")
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--reset_storage", action="store_true")
    parser.add_argument("--stagger_seconds", type=float, default=3.0)
    parser.set_defaults(pin_memory=True, persistent_workers=True, preload_features=True, clean_splits_in_place=True)
    args = parser.parse_args()

    if args.trials_per_gpu <= 0:
        raise ValueError("--trials_per_gpu must be a positive integer")

    args.base_dir = Path(args.base_dir).expanduser().resolve()
    args.embeddings_dir = Path(args.embeddings_dir).expanduser().resolve()
    args.thresholds = args.thresholds or ([args.threshold] if args.threshold else None)
    maybe_cache_embeddings(args)
    prepare_optuna_storage(args)
    optuna.create_study(
        direction=metric_direction(args.metric),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
    )

    gpu_worker_slots = []
    for gpu_id in args.gpus:
        for slot_index in range(args.trials_per_gpu):
            gpu_worker_slots.append((str(gpu_id), slot_index))

    worker_trial_counts = split_trials(args.n_trials, len(gpu_worker_slots))
    processes = []
    try:
        for worker_index, ((gpu_id, slot_index), worker_trials) in enumerate(zip(gpu_worker_slots, worker_trial_counts)):
            if worker_trials <= 0:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = worker_cmd(args, worker_trials, worker_index)
            print("Launching Optuna worker %s on GPU %s slot %s for %s trials" % (worker_index, gpu_id, slot_index, worker_trials), flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
            processes.append((gpu_id, slot_index, worker_trials, proc))
            if worker_index < len(gpu_worker_slots) - 1 and args.stagger_seconds > 0:
                time.sleep(args.stagger_seconds)

        failed = False
        for gpu_id, slot_index, worker_trials, proc in processes:
            return_code = proc.wait()
            if return_code != 0:
                failed = True
                print(
                    "Worker on GPU %s slot %s failed after %s trials with exit code %s"
                    % (gpu_id, slot_index, worker_trials, return_code),
                    flush=True,
                )
        if failed:
            raise RuntimeError("One or more parallel Optuna workers failed.")
    finally:
        for _gpu_id, _slot_index, _worker_trials, proc in processes:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
