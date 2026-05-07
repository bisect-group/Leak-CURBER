from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset_adapter import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    SPLIT_NAMES,
    prepare_split_group,
    select_split_groups,
)
from .utils import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNS_ROOT,
    ensure_dir,
    metadata_path_for_split,
    read_json,
    resolve_path,
    sha256_text,
    write_json,
)


def protein_cache_path(cache_root: str | Path, protein_id: str) -> Path:
    shard = protein_id.split("_", 1)[-1][:2]
    return Path(cache_root) / "proteins" / shard / f"{protein_id}.npy"


def deterministic_vector(sequence: str, dim: int) -> np.ndarray:
    seed = int(sha256_text(sequence)[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def _load_protein_table(metadata: dict) -> pd.DataFrame:
    frames = []
    for split in SPLIT_NAMES:
        frame = pd.read_csv(metadata["manifests"][split])
        frames.append(frame.loc[:, ["protein_id", "sequence", "sequence_sha256"]])
    proteins = pd.concat(frames, ignore_index=True).drop_duplicates("protein_id")
    return proteins.sort_values("protein_id").reset_index(drop=True)


def _save_vector(path: Path, vector: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, vector.astype(np.float32))


def _load_vector(path: Path, expected_dim: int) -> np.ndarray:
    vector = np.load(path).astype(np.float32)
    if vector.shape != (expected_dim,):
        raise ValueError(f"{path} has shape {vector.shape}; expected {(expected_dim,)}")
    return vector


def _prepare_prott5_sequences(sequences: list[str]) -> list[str]:
    prepared = []
    for sequence in sequences:
        normalized = sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
        prepared.append(" ".join(normalized))
    return prepared


def embed_missing_with_prott5(
    missing: pd.DataFrame,
    *,
    cache_root: Path,
    model_name: str,
    model_revision: str | None,
    device: str,
    embedding_dim: int,
    batch_size: int,
) -> int:
    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise ImportError(
            "ProtT5 embedding generation requires transformers and sentencepiece. "
            "Install the emulator optional dependencies or use --embedding-source cache-only "
            "with a prepopulated cache."
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    revision_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer = T5Tokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        **revision_kwargs,
    )
    model_kwargs = {
        **revision_kwargs,
        "use_safetensors": True,
    }
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    try:
        model = T5EncoderModel.from_pretrained(model_name, **model_kwargs).to(device)
    except Exception as exc:
        revision_text = f" revision {model_revision!r}" if model_revision else ""
        raise RuntimeError(
            f"Could not load {model_name}{revision_text} with safetensors. "
            "The current horizon environment has torch<2.6, so recent "
            "transformers releases refuse to load pytorch_model.bin files. "
            "Use a safetensors revision/cache, prepopulate the protein cache and "
            "run with --embedding-source cache-only, or upgrade torch in the "
            "horizon environment to >=2.6."
        ) from exc
    model.eval()

    written = 0
    records = missing.to_dict("records")
    for start in tqdm(range(0, len(records), batch_size), desc="ProtT5 embeddings"):
        batch = records[start : start + batch_size]
        sequences = [record["sequence"] for record in batch]
        tokenized = tokenizer(
            _prepare_prott5_sequences(sequences),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            outputs = model(**tokenized)
        hidden = outputs.last_hidden_state.detach().float().cpu()
        for idx, record in enumerate(batch):
            seq_len = len(record["sequence"])
            vector = hidden[idx, :seq_len].mean(dim=0).numpy().astype(np.float32)
            if vector.shape != (embedding_dim,):
                raise ValueError(
                    f"ProtT5 produced vector shape {vector.shape}; expected {(embedding_dim,)}"
                )
            _save_vector(protein_cache_path(cache_root, record["protein_id"]), vector)
            written += 1
    return written


def populate_protein_cache(
    proteins: pd.DataFrame,
    *,
    cache_root: Path,
    embedding_source: str,
    allow_deterministic_embeddings: bool,
    embedding_dim: int,
    protein_model_name: str,
    protein_model_revision: str | None,
    device: str,
    sequence_batch_size: int,
) -> dict:
    cache_root = ensure_dir(cache_root)
    missing_records = []
    hits = 0
    for row in tqdm(
        proteins.itertuples(index=False),
        total=len(proteins),
        desc="protein cache scan",
    ):
        path = protein_cache_path(cache_root, row.protein_id)
        if path.exists():
            _load_vector(path, embedding_dim)
            hits += 1
            continue
        missing_records.append(
            {"protein_id": row.protein_id, "sequence": row.sequence}
        )

    if missing_records and embedding_source == "cache-only":
        raise FileNotFoundError(
            f"{len(missing_records)} protein embeddings are missing from {cache_root}"
        )

    written = 0
    if missing_records and embedding_source == "deterministic":
        if not allow_deterministic_embeddings:
            raise ValueError(
                "Deterministic embeddings are for smoke tests only. "
                "Pass --allow-deterministic-embeddings to acknowledge this."
            )
        for record in tqdm(missing_records, desc="deterministic embeddings"):
            vector = deterministic_vector(record["sequence"], embedding_dim)
            _save_vector(protein_cache_path(cache_root, record["protein_id"]), vector)
            written += 1
    elif missing_records and embedding_source == "prott5":
        written = embed_missing_with_prott5(
            pd.DataFrame(missing_records),
            cache_root=cache_root,
            model_name=protein_model_name,
            model_revision=protein_model_revision,
            device=device,
            embedding_dim=embedding_dim,
            batch_size=sequence_batch_size,
        )

    return {
        "cache_hits": int(hits),
        "cache_misses": int(len(missing_records)),
        "cache_written": int(written),
        "embedding_source": embedding_source,
        "embedding_dim": int(embedding_dim),
    }


def write_hdf5_view(
    proteins: pd.DataFrame,
    *,
    cache_root: Path,
    output_path: Path,
    embedding_dim: int,
) -> None:
    ensure_dir(output_path.parent)
    ids = proteins["protein_id"].astype(str).tolist()
    vectors = np.zeros((len(ids), embedding_dim), dtype=np.float32)
    for idx, protein_id in enumerate(tqdm(ids, desc="materialize HDF5")):
        vectors[idx] = _load_vector(protein_cache_path(cache_root, protein_id), embedding_dim)
    with h5py.File(output_path, "w") as handle:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("ids", data=np.array(ids, dtype=object), dtype=string_dtype)
        handle.create_dataset("vectors", data=vectors, compression="gzip")


def prepare_and_cache_group(
    group,
    *,
    dataset_root: str | Path,
    runs_root: str | Path,
    cache_root: str | Path,
    label_policy: str,
    max_sequence_length: int,
    limit_per_split: int | None,
    validate_fingerprints: bool,
    embedding_source: str,
    allow_deterministic_embeddings: bool,
    embedding_dim: int,
    protein_model_name: str,
    protein_model_revision: str | None,
    device: str,
    sequence_batch_size: int,
) -> dict:
    cache_root = resolve_path(cache_root)
    metadata = prepare_split_group(
        group,
        dataset_root=dataset_root,
        runs_root=runs_root,
        label_policy=label_policy,
        max_sequence_length=max_sequence_length,
        limit_per_split=limit_per_split,
        validate_fingerprints=validate_fingerprints,
    )
    proteins = _load_protein_table(metadata)
    cache_stats = populate_protein_cache(
        proteins,
        cache_root=cache_root,
        embedding_source=embedding_source,
        allow_deterministic_embeddings=allow_deterministic_embeddings,
        embedding_dim=embedding_dim,
        protein_model_name=protein_model_name,
        protein_model_revision=protein_model_revision,
        device=device,
        sequence_batch_size=sequence_batch_size,
    )

    hdf5_path = Path(metadata["run_root"]) / "horizyn_data" / "prots_t5.h5"
    write_hdf5_view(
        proteins,
        cache_root=cache_root,
        output_path=hdf5_path,
        embedding_dim=embedding_dim,
    )
    metadata["baseline_files"]["protein_embeds"] = str(hdf5_path)
    metadata["cache"] = {
        "cache_root": str(cache_root),
        "protein_manifest": str(Path(metadata["run_root"]) / "manifests" / "proteins.csv"),
        **cache_stats,
    }
    protein_manifest = proteins.loc[:, ["protein_id", "sequence_sha256"]].copy()
    protein_manifest["cache_path"] = protein_manifest["protein_id"].map(
        lambda protein_id: str(protein_cache_path(cache_root, protein_id))
    )
    protein_manifest.to_csv(metadata["cache"]["protein_manifest"], index=False)
    write_json(metadata_path_for_split(group.name, runs_root), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Horizon protein features for Leak-CURBER")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--skip-fingerprint-validation", action="store_true")
    parser.add_argument(
        "--embedding-source",
        choices=["prott5", "cache-only", "deterministic"],
        default="prott5",
    )
    parser.add_argument("--allow-deterministic-embeddings", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--protein-model-name", default="Rostlab/prot_t5_xl_half_uniref50-enc")
    parser.add_argument(
        "--protein-model-revision",
        default="refs/pr/1",
        help=(
            "Model revision used for ProtT5. The default points at the "
            "safetensors conversion PR because transformers refuses PyTorch "
            "weight files with torch<2.6."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sequence-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = select_split_groups(args.dataset_root, args.split_group)
    for group in groups:
        metadata = prepare_and_cache_group(
            group,
            dataset_root=args.dataset_root,
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            label_policy=args.label_policy,
            max_sequence_length=args.max_sequence_length,
            limit_per_split=args.limit_per_split,
            validate_fingerprints=not args.skip_fingerprint_validation,
            embedding_source=args.embedding_source,
            allow_deterministic_embeddings=args.allow_deterministic_embeddings,
            embedding_dim=args.embedding_dim,
            protein_model_name=args.protein_model_name,
            protein_model_revision=args.protein_model_revision,
            device=args.device,
            sequence_batch_size=args.sequence_batch_size,
        )
        print(
            f"[emulator_bench] cached {group.name}: "
            f"{metadata['baseline_files']['protein_embeds']} "
            f"(hits={metadata['cache']['cache_hits']}, "
            f"misses={metadata['cache']['cache_misses']}, "
            f"written={metadata['cache']['cache_written']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
