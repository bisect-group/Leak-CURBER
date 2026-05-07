import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_SPLIT_GROUPS,
    discover_split_jobs,
    ligand_cache_path,
    normalize_sequence,
    normalize_threshold_args,
    protein_cache_path,
    read_table,
    save_json,
)
from emulator_bench.feature_pipeline import (
    embed_sequences,
    ligand_cache_item,
    load_esmfold_model,
    protein_cache_item,
    resolve_amp_dtype,
    save_ligand_pt,
    save_protein_npz,
)


def _collect_unique_values(jobs, sequence_col: str, smiles_col: str, max_seq_len: int):
    """Collect unique sequences (truncated to max_seq_len) and SMILES across all split files."""
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[split_key]))
            if sequence_col not in frame.columns or smiles_col not in frame.columns:
                raise ValueError(f"Expected columns `{sequence_col}` and `{smiles_col}` in {job[split_key]}")
            for value in frame[sequence_col].astype(str):
                normalized = normalize_sequence(value)
                if max_seq_len and len(normalized) > max_seq_len:
                    normalized = normalized[:max_seq_len]
                sequences.add(normalized)
            smiles_values.update(str(value) for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def cache_proteins_single_gpu(device_str, sequences, args):
    """Embed and cache a list of (already-truncated) sequences on one GPU."""
    pending = [
        seq
        for seq in sequences
        if args.overwrite or not protein_cache_path(args.embeddings_dir, seq).exists()
    ]
    if not pending:
        return {"proteins_total": len(sequences), "proteins_written": 0}

    device = torch.device(device_str)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    print(f"[{device_str}] Caching {len(pending)} proteins | precision: {precision_mode}", flush=True)
    model = load_esmfold_model(device, chunk_size=args.chunk_size)

    written = 0
    iterator = tqdm(pending, desc=f"[{device_str}] ESMFold embeddings", unit="seq")
    for sequence in iterator:
        # sequence is already truncated; pass max_seq_len large enough so embed_sequences
        # doesn't try to chunk it further (it's at most max_seq_len long).
        embedded = embed_sequences(
            model,
            [sequence],
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=args.max_seq_len,
            long_seq_stride=args.long_seq_stride,
        )
        save_protein_npz(
            protein_cache_path(args.embeddings_dir, sequence),
            protein_cache_item(sequence, embedded[sequence], protein_dtype=args.protein_dtype),
        )
        written += 1
        iterator.set_postfix(written=written, remaining=len(pending) - written)

    return {"proteins_total": len(sequences), "proteins_written": written}


def _worker_main():
    """Entry point when this script is re-invoked as a worker subprocess."""
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--_worker", action="store_true")
    parser.add_argument("--worker_sequences_file", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, default=700)
    parser.add_argument("--long_seq_stride", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.embeddings_dir = Path(args.embeddings_dir)

    with open(args.worker_sequences_file, "rb") as f:
        sequences = pickle.load(f)

    cache_proteins_single_gpu(args.device, sequences, args)


def cache_proteins_multi_gpu(args, sequences, gpus):
    """Split sequence list across GPUs and launch one subprocess per GPU."""
    import pickle
    import tempfile

    pending = [
        seq
        for seq in sequences
        if args.overwrite or not protein_cache_path(args.embeddings_dir, seq).exists()
    ]
    if not pending:
        print("Protein cache is already complete.")
        return {"proteins_total": len(sequences), "proteins_written": 0}

    n_gpus = len(gpus)
    chunk_size = math.ceil(len(pending) / n_gpus)
    chunks = [pending[i * chunk_size : (i + 1) * chunk_size] for i in range(n_gpus)]
    chunks = [c for c in chunks if c]  # drop empty tail chunks

    tmp_dir = Path(tempfile.mkdtemp(prefix="emosaic_cache_"))
    procs = []
    for i, (gpu_id, chunk) in enumerate(zip(gpus, chunks)):
        seq_file = tmp_dir / f"seqs_gpu{i}.pkl"
        with open(seq_file, "wb") as f:
            pickle.dump(chunk, f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, __file__,
            "--_worker",
            "--worker_sequences_file", str(seq_file),
            "--device", "cuda:0",
            "--embeddings_dir", str(args.embeddings_dir),
            "--max_seq_len", str(args.max_seq_len),
            "--long_seq_stride", str(args.long_seq_stride),
            "--chunk_size", str(args.chunk_size),
            "--protein_dtype", args.protein_dtype,
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        print(f"[GPU {gpu_id}] Launching worker for {len(chunk)} sequences", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    for proc in procs:
        proc.wait()

    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        raise RuntimeError(f"Worker subprocesses for GPU indices {failed} exited with errors.")

    # Clean up temp files
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    return {"proteins_total": len(sequences), "proteins_written": len(pending)}


def cache_ligands(args, smiles_values):
    pending = [
        smiles
        for smiles in smiles_values
        if args.overwrite or not ligand_cache_path(args.embeddings_dir, smiles).exists()
    ]
    if not pending:
        print("Ligand cache is already complete.")
        return {"ligands_total": len(smiles_values), "ligands_written": 0}

    written = 0
    iterator = tqdm(pending, desc="Caching ligand graphs", unit="smiles")
    for smiles in iterator:
        item = ligand_cache_item(smiles)
        save_ligand_pt(ligand_cache_path(args.embeddings_dir, smiles), item)
        written += 1
        iterator.set_postfix(written=written, remaining=len(pending) - written)

    return {"ligands_total": len(smiles_values), "ligands_written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache reusable eMOSAIC protein (ESMFold) embeddings and ligand graphs.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="GPU IDs for multi-GPU caching (e.g. --gpus 0 1 2 3). "
                             "If omitted, uses --device for single-GPU caching.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_len", type=int, default=700)
    parser.add_argument("--long_seq_stride", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=128, help="ESMFold internal chunk size; lower reduces peak memory.")
    parser.add_argument("--protein_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_proteins", action="store_true", help="Skip ESMFold embedding; only (re-)cache ligand graphs.")
    parser.add_argument("--skip_ligands", action="store_true", help="Skip ligand graph caching; only (re-)cache protein embeddings.")
    # Worker-only args (parsed by _worker_main, accepted here to avoid argparse errors)
    parser.add_argument("--worker_sequences_file", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args._worker:
        _worker_main()
        return

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")

    started = time.time()
    sequences, smiles_values = _collect_unique_values(
        jobs,
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
        max_seq_len=args.max_seq_len,
    )
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique normalized sequences (truncated to {args.max_seq_len} residues): {len(sequences)}")
    print(f"Unique SMILES: {len(smiles_values)}")

    if args.skip_proteins:
        protein_stats = {"proteins_total": len(sequences), "proteins_written": 0, "skipped": True}
    elif args.gpus and len(args.gpus) > 1:
        protein_stats = cache_proteins_multi_gpu(args, sequences, args.gpus)
    else:
        device_str = f"cuda:{args.gpus[0]}" if args.gpus else args.device
        protein_stats = cache_proteins_single_gpu(device_str, sequences, args)

    if args.skip_ligands:
        ligand_stats = {"ligands_total": len(smiles_values), "ligands_written": 0, "skipped": True}
    else:
        ligand_stats = cache_ligands(args, smiles_values)

    manifest = {
        "cache_version": 2,
        "base_dir": str(args.base_dir),
        "embeddings_dir": str(args.embeddings_dir),
        "sequence_col": args.sequence_col,
        "smiles_col": args.smiles_col,
        "split_groups": [job["split_group"] for job in jobs],
        "thresholds": args.thresholds,
        "protein_dtype": args.protein_dtype,
        "protein_model": "esmfold_v1 (structure-module s_s)",
        "protein_max_seq_len": int(args.max_seq_len),
        "protein_long_seq_stride": int(args.long_seq_stride),
        "protein_chunk_size": int(args.chunk_size) if args.chunk_size else None,
        "protein_cache": protein_stats,
        "ligand_cache": ligand_stats,
        "sequences_truncated_to_max": int(args.max_seq_len),
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.embeddings_dir / "manifest.json", manifest)
    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
