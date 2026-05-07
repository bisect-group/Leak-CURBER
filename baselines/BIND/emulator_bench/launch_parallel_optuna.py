import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_HIDDEN_STATE_INDICES
from emulator_bench.run_split_benchmarks import maybe_cache_embeddings
from emulator_bench.tune_optuna import sqlite_path_from_storage


TUNE_SCRIPT = REPO_ROOT / "emulator_bench" / "tune_optuna.py"


def build_worker_command(args):
    cmd = [
        sys.executable,
        str(TUNE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--value_type",
        args.value_type,
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
        "--cross_attention_dropout",
        str(args.cross_attention_dropout),
        "--leaky_relu_negative_slope",
        str(args.leaky_relu_negative_slope),
        "--effective_batch_size",
        str(args.effective_batch_size),
        "--epochs",
        str(args.epochs),
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
        "--num_workers",
        str(args.num_workers),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--sharing_strategy",
        args.sharing_strategy,
        "--protein_cache_items",
        str(args.protein_cache_items),
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
        "--huber_delta",
        str(args.huber_delta),
        "--metric",
        args.metric,
        "--eval_split",
        args.eval_split,
        "--n_trials",
        str(args.n_trials),
        "--sampler_seed",
        str(args.sampler_seed),
        "--study_name",
        args.study_name,
        "--storage",
        args.storage,
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.seeds:
        cmd.extend(["--seeds", *[str(seed) for seed in args.seeds]])
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
    if args.skip_cache:
        cmd.append("--skip_cache")
    if args.cache_overwrite:
        cmd.append("--cache_overwrite")
    if args.overwrite_runs:
        cmd.append("--overwrite_runs")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch multiple single-GPU Optuna workers against one shared BIND study.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--trials_per_gpu", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--value_type", type=str, required=True)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[666])
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--target_col", type=str, default="log10_value")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--cross_attention_dropout", type=float, default=0.1)
    parser.add_argument("--leaky_relu_negative_slope", type=float, default=0.05)
    parser.add_argument("--effective_batch_size", type=int, default=256)
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
    parser.add_argument("--overwrite_runs", action="store_true")
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")
    parser.add_argument("--max_residues", type=int, default=4096)
    parser.add_argument("--max_batch", type=int, default=4)
    parser.add_argument("--sliding_window_stride", type=int, default=896)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--huber_delta", type=float, default=2.0)
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--reset_storage", action="store_true")
    args = parser.parse_args()

    args.hidden_state_indices = [int(index) for index in args.hidden_state_indices]
    args.thresholds = args.thresholds or ([args.threshold] if args.threshold else None)
    if args.study_name is None:
        args.study_name = f"bind_{args.value_type}_optuna"
    if args.storage is None:
        args.storage = "sqlite:///%s" % (Path(args.base_dir) / args.value_type / "optuna_studies" / f"{args.study_name}.db")

    maybe_cache_embeddings(args)

    db_path = sqlite_path_from_storage(args.storage)
    if db_path is not None and args.reset_storage and db_path.exists():
        db_path.unlink()

    worker_cmd = build_worker_command(args)
    failures = []
    lock = threading.Lock()

    def worker(gpu_id, slot_index):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            subprocess.run(worker_cmd, check=True, cwd=str(REPO_ROOT), env=env)
        except Exception as exc:
            with lock:
                failures.append({"gpu_id": str(gpu_id), "slot_index": int(slot_index), "error": str(exc)})

    threads = []
    for gpu_id in args.gpus:
        for slot_index in range(int(args.trials_per_gpu)):
            thread = threading.Thread(target=worker, args=(gpu_id, slot_index), daemon=True)
            thread.start()
            threads.append(thread)
    for thread in threads:
        thread.join()

    if failures:
        raise RuntimeError(f"One or more Optuna workers failed: {failures}")


if __name__ == "__main__":
    main()
