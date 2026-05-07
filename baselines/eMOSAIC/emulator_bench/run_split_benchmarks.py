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
    DEFAULT_RESULTS_DIRNAME,
    DEFAULT_SPLIT_GROUPS,
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
        "--device",
        args.cache_device,
        "--max_seq_len",
        str(args.max_length),
        "--long_seq_stride",
        str(args.long_seq_stride),
        "--chunk_size",
        str(args.chunk_size),
        "--protein_dtype",
        args.protein_dtype,
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
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "amsgrad",
        "scheduler",
        "cosine_t0",
        "clip_grad",
    ]:
        if key in hparams:
            setattr(args, key, hparams[key])
    return args


def train_one(job, seed, args, gpu_id=None):
    result_root = Path(job["root_dir"]) / args.results_dirname / f"seed_{seed}"
    metric_path = result_root / "final_results_test.csv"
    if metric_path.exists() and not args.overwrite:
        return result_root

    if gpu_id is not None:
        device = "cuda:0"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        device = args.device
        env = None

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
        str(args.batch_size),
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
        "--cosine_t0",
        str(args.cosine_t0),
        "--clip_grad",
        str(args.clip_grad),
        "--val_every",
        str(args.val_every),
        "--device",
        device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
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
    else:
        cmd.append("--no-persistent_workers")
    if args.preload_proteins:
        cmd.append("--preload_proteins")
    if args.lazy_ligands:
        cmd.append("--lazy_ligands")
    if args.torch_compile:
        cmd.append("--torch_compile")
    if args.amsgrad:
        cmd.append("--amsgrad")
    else:
        cmd.append("--no-amsgrad")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    return result_root


def _collect_row(job, seed, out_dir, args):
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
            row["%s_%s" % (prefix, key)] = value
    return row


def run_parallel(jobs, args):
    work_queue = queue.Queue()
    for job in jobs:
        for seed in args.seeds:
            work_queue.put((job, seed))

    summary_rows = []
    result_lock = threading.Lock()

    def worker(gpu_id):
        while True:
            try:
                job, seed = work_queue.get_nowait()
            except queue.Empty:
                return
            print(f"[GPU {gpu_id}] {job['split_group']} | {job['split_name']} | threshold {job['difficulty']} | seed {seed}", flush=True)
            try:
                out_dir = train_one(job, seed, args, gpu_id=gpu_id)
                row = _collect_row(job, seed, out_dir, args)
            except Exception as exc:
                row = {
                    "split_group": job["split_group"],
                    "split_name": job["split_name"],
                    "difficulty": job["difficulty"],
                    "seed": int(seed),
                    "run_dir": str(Path(job["root_dir"]) / args.results_dirname / ("seed_%s" % seed)),
                    "status": "failed",
                    "error": str(exc),
                }
            with result_lock:
                summary_rows.append(row)
            work_queue.task_done()

    threads = []
    for gpu_id in args.gpus:
        for _ in range(args.trials_per_gpu):
            t = threading.Thread(target=worker, args=(str(gpu_id),), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Run the eMOSAIC emulator bench across Leak-CURBER split groups.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--results_dirname", type=str, default=DEFAULT_RESULTS_DIRNAME)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU IDs for parallel execution. If omitted, runs sequentially on --device.")
    parser.add_argument("--trials_per_gpu", type=int, default=1, help="Concurrent training jobs per GPU when --gpus is set.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_device", type=str, default="cuda:0")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--cache_overwrite", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hparams_json", type=str, default=None)

    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")

    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--max_length", type=int, default=700)
    parser.add_argument("--prot_input_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler", choices=["none", "cosine_warm_restarts"], default="cosine_warm_restarts")
    parser.add_argument("--cosine_t0", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--preload_proteins", action="store_true")
    parser.add_argument("--protein_cache_items", type=int, default=512)
    parser.add_argument("--lazy_ligands", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")

    parser.add_argument("--long_seq_stride", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    if args.trials_per_gpu <= 0:
        raise ValueError("--trials_per_gpu must be a positive integer")

    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args = maybe_load_hparams(args)
    maybe_cache_embeddings(args)

    jobs = discover_split_jobs(Path(args.base_dir), split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs found in {args.base_dir}")

    total_jobs = len(jobs) * len(args.seeds)
    print(f"Found {len(jobs)} splits × {len(args.seeds)} seed(s) = {total_jobs} total jobs")

    if args.gpus:
        summary_rows = run_parallel(jobs, args)
    else:
        summary_rows = []
        for job in tqdm(jobs, desc="Benchmark jobs", unit="job"):
            for seed in args.seeds:
                tqdm.write(f"  {job['split_group']} | {job['split_name']} | threshold {job['difficulty']} | seed {seed}")
                out_dir = train_one(job, seed, args)
                summary_rows.append(_collect_row(job, seed, out_dir, args))

    base = Path(args.base_dir)
    runs_df = pd.DataFrame(summary_rows)
    runs_df.to_csv(base / "emosaic_summary_runs.csv", index=False)
    metric_cols = [col for col in runs_df.columns if col.startswith("test_")]
    summarize_seed_runs(summary_rows, ["split_group", "split_name", "difficulty"], metric_cols).to_csv(
        base / "emosaic_summary_thresholds.csv", index=False,
    )
    summarize_seed_runs(summary_rows, ["split_group"], metric_cols).to_csv(
        base / "emosaic_summary_by_split_group.csv", index=False,
    )


if __name__ == "__main__":
    main()
