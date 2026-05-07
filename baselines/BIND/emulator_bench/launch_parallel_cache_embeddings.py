import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_EMBEDDINGS_DIR, DEFAULT_HIDDEN_STATE_INDICES, DEFAULT_VALUE_TYPES


CACHE_SCRIPT = REPO_ROOT / "emulator_bench" / "cache_embeddings.py"


def build_worker_command(args, shard_index: int):
    cmd = [
        sys.executable,
        str(CACHE_SCRIPT),
        "--base_dir",
        args.base_dir,
        "--embeddings_dir",
        args.embeddings_dir,
        "--value_types",
        *args.value_types,
        "--sequence_col",
        args.sequence_col,
        "--smiles_col",
        args.smiles_col,
        "--device",
        "cuda:0" if args.device.startswith("cuda") else args.device,
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
        "--writer_threads",
        str(args.writer_threads),
        "--max_pending_writes",
        str(args.max_pending_writes),
        "--num_shards",
        str(args.num_shards),
        "--shard_index",
        str(shard_index),
    ]
    if args.split_groups:
        cmd.extend(["--split_groups", *args.split_groups])
    if args.thresholds:
        cmd.extend(["--thresholds", *args.thresholds])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch multiple GPU workers to build the shared BIND cache by hash sharding proteins and ligands.")
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--value_types", nargs="+", default=DEFAULT_VALUE_TYPES)
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--hidden_state_indices", nargs="+", type=int, default=DEFAULT_HIDDEN_STATE_INDICES)
    parser.add_argument("--protein_length_cap", type=int, default=2048)
    parser.add_argument("--long_sequence_strategy", choices=["drop", "direct", "sliding_window"], default="direct")
    parser.add_argument("--max_residues", type=int, default=8192)
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--sliding_window_stride", type=int, default=896)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--writer_threads", type=int, default=max(4, min(16, os.cpu_count() or 4)))
    parser.add_argument("--max_pending_writes", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.hidden_state_indices = [int(index) for index in args.hidden_state_indices]
    args.thresholds = args.thresholds or ([args.threshold] if args.threshold else None)
    args.num_shards = len(args.gpus)

    failures = []
    lock = threading.Lock()

    def worker(gpu_id, shard_index):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = build_worker_command(args, shard_index=shard_index)
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
        except Exception as exc:
            with lock:
                failures.append({"gpu_id": str(gpu_id), "shard_index": int(shard_index), "error": str(exc)})

    threads = []
    for shard_index, gpu_id in enumerate(args.gpus):
        thread = threading.Thread(target=worker, args=(gpu_id, shard_index), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    if failures:
        raise RuntimeError(f"One or more cache workers failed: {failures}")


if __name__ == "__main__":
    main()
