import argparse
import sys
import time
from pathlib import Path

import pandas as pd
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
    embedding_cache_path,
    normalize_sequence,
    normalize_threshold_args,
    read_table,
    save_json,
)
from emulator_bench.feature_pipeline import embed_text_batch, load_encoder, resolve_amp_dtype, save_embedding


def _collect_unique_values(jobs, sequence_col: str, smiles_col: str):
    sequences = set()
    smiles_values = set()
    for job in jobs:
        for split_key in ("train_path", "val_path", "test_path"):
            frame = read_table(Path(job[split_key]))
            if sequence_col not in frame.columns or smiles_col not in frame.columns:
                raise ValueError(f"Expected columns `{sequence_col}` and `{smiles_col}` in {job[split_key]}. Found: {list(frame.columns)}")
            sequences.update(normalize_sequence(value) for value in frame[sequence_col].astype(str))
            smiles_values.update(str(value).strip() for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)


def _worker_fn(rank: int, device_str: str, pending_chunk: list, kind: str, model_name: str, max_length: int, batch_size: int, embeddings_dir: Path, embedding_dtype: str) -> int:
    device = torch.device(device_str)
    autocast_dtype, precision_mode = resolve_amp_dtype(device)
    print(f"[{device_str}] Loading {model_name} | precision: {precision_mode}", flush=True)
    tokenizer, model = load_encoder(model_name, device)
    written = 0
    for start in tqdm(range(0, len(pending_chunk), batch_size), desc=f"[{device_str}] {kind}", unit="batch", position=rank, leave=True):
        batch_values = pending_chunk[start : start + batch_size]
        embeddings = embed_text_batch(batch_values, tokenizer, model, device, max_length=max_length, autocast_dtype=autocast_dtype)
        for value, embedding in zip(batch_values, embeddings):
            save_embedding(
                embedding_cache_path(embeddings_dir, kind, value, model_name, max_length),
                embedding,
                dtype=embedding_dtype,
            )
            written += 1
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return written


def _worker_fn_star(args_tuple):
    return _worker_fn(*args_tuple)


def cache_kind(args, values, kind: str, model_name: str, max_length: int, batch_size: int):
    pending = [
        value
        for value in values
        if args.overwrite or not embedding_cache_path(args.embeddings_dir, kind, value, model_name, max_length).exists()
    ]
    if not pending:
        print(f"{kind.capitalize()} cache is already complete.")
        return {f"{kind}_total": len(values), f"{kind}_written": 0}

    devices = args.devices
    n_devices = len(devices)
    chunks = [pending[i::n_devices] for i in range(n_devices)]
    chunk_sizes = [len(c) for c in chunks]
    print(f"Caching {len(pending)} {kind} with {model_name} across {n_devices} device(s): {devices} | chunks: {chunk_sizes}", flush=True)

    if n_devices == 1:
        written = _worker_fn(0, devices[0], chunks[0], kind, model_name, max_length, batch_size, args.embeddings_dir, args.embedding_dtype)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        worker_args = [
            (rank, devices[rank], chunks[rank], kind, model_name, max_length, batch_size, args.embeddings_dir, args.embedding_dtype)
            for rank in range(n_devices)
        ]
        with ctx.Pool(processes=n_devices) as pool:
            results = pool.map(_worker_fn_star, worker_args)
        written = sum(results)

    return {f"{kind}_total": len(values), f"{kind}_written": written}


def main():
    parser = argparse.ArgumentParser(description="Cache one-time ESM2 and ChemBERTa mean-pooled embeddings for StructureFree-DTA splits.")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--split_groups", nargs="+", default=None)
    parser.add_argument("--threshold", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", default=None)
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--smiles_col", type=str, default="smiles")
    parser.add_argument("--protein_model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--molecule_model_name", type=str, default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--max_protein_length", type=int, default=1024)
    parser.add_argument("--max_molecule_length", type=int, default=128)
    parser.add_argument("--protein_batch_size", type=int, default=64)
    parser.add_argument("--molecule_batch_size", type=int, default=256)
    parser.add_argument("--embedding_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--devices", nargs="+", default=None, help="One or more devices, e.g. --devices cuda:0 cuda:1. Overrides --device.")
    parser.add_argument("--device", type=str, default=None, help="Single device (legacy). Use --devices for multi-GPU.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Resolve devices: --devices wins over --device; fall back to cuda:0 or cpu
    if args.devices is None:
        if args.device is not None:
            args.devices = [args.device]
        elif torch.cuda.is_available():
            args.devices = ["cuda:0"]
        else:
            args.devices = ["cpu"]

    args.base_dir = Path(args.base_dir)
    args.embeddings_dir = Path(args.embeddings_dir)
    args.embeddings_dir.mkdir(parents=True, exist_ok=True)
    args.thresholds = normalize_threshold_args(args.thresholds, args.threshold)

    started = time.time()
    jobs = discover_split_jobs(args.base_dir, split_groups=args.split_groups, thresholds=args.thresholds)
    if not jobs:
        raise FileNotFoundError(f"No split jobs discovered in {args.base_dir}")
    sequences, smiles_values = _collect_unique_values(jobs, args.sequence_col, args.smiles_col)
    print(f"Discovered {len(jobs)} split jobs")
    print(f"Unique proteins: {len(sequences)}")
    print(f"Unique molecules: {len(smiles_values)}")

    protein_stats = cache_kind(args, sequences, "proteins", args.protein_model_name, args.max_protein_length, args.protein_batch_size)
    molecule_stats = cache_kind(args, smiles_values, "molecules", args.molecule_model_name, args.max_molecule_length, args.molecule_batch_size)

    save_json(
        args.embeddings_dir / "manifest.json",
        {
            "cache_version": 1,
            "base_dir": str(args.base_dir),
            "embeddings_dir": str(args.embeddings_dir),
            "sequence_col": args.sequence_col,
            "smiles_col": args.smiles_col,
            "split_groups": [job["split_group"] for job in jobs],
            "thresholds": args.thresholds,
            "protein_model_name": args.protein_model_name,
            "molecule_model_name": args.molecule_model_name,
            "max_protein_length": int(args.max_protein_length),
            "max_molecule_length": int(args.max_molecule_length),
            "embedding_dtype": args.embedding_dtype,
            "devices": args.devices,
            "protein_cache": protein_stats,
            "molecule_cache": molecule_stats,
            "elapsed_seconds": time.time() - started,
        },
    )
    pd.DataFrame(jobs).to_csv(args.embeddings_dir / "discovered_jobs.csv", index=False)
    print(f"Saved cache manifest to {args.embeddings_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
