from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from .dataset_adapter import load_manifest, prepare_split_group, select_split_groups
from .utils import (
    APP_DATA_DIR,
    APP_DIR,
    DEFAULT_CACHE_ROOT,
    DEFAULT_RUNS_ROOT,
    add_clean_to_path,
    ensure_dir,
    pushd,
    resolve_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CLEAN inputs and shared ESM cache")
    parser.add_argument(
        "--dataset-root",
        default="../../data/processed/datasets/enzyme_classification_dataset",
        help="Leak-CURBER enzyme classification dataset root",
    )
    parser.add_argument("--split-group", action="append", help="Split group to prepare")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--esm-model", default="esm1b_t33_650M_UR50S")
    parser.add_argument("--repr-layer", type=int, default=33)
    parser.add_argument("--toks-per-batch", type=int, default=2048)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-esm", action="store_true")
    parser.add_argument("--skip-distance-map", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_by_tokens(items: list[tuple[str, str, Path]], toks_per_batch: int):
    sorted_items = sorted(items, key=lambda item: len(item[1]))
    batch: list[tuple[str, str, Path]] = []
    max_len = 0
    for item in sorted_items:
        seq_len = len(item[1]) + 2
        if batch and max(max_len, seq_len) * (len(batch) + 1) > toks_per_batch:
            yield batch
            batch = []
            max_len = 0
        batch.append(item)
        max_len = max(max_len, seq_len)
    if batch:
        yield batch


def load_esm_model(model_name: str, device: torch.device):
    from esm import pretrained

    print(f"[emulator_bench] loading ESM model {model_name} on {device}", flush=True)
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)
    return model, alphabet


def extract_esm_features(
    items: list[tuple[str, str, Path]],
    *,
    model_name: str,
    repr_layer: int,
    toks_per_batch: int,
    device: torch.device,
    max_seq_length: int,
) -> None:
    if not items:
        return

    model, alphabet = load_esm_model(model_name, device)
    max_positions = getattr(getattr(model, "args", None), "max_positions", None)
    special_tokens = int(alphabet.prepend_bos) + int(alphabet.append_eos)
    esm_truncation_length = max_seq_length
    if max_positions is not None:
        esm_truncation_length = min(max_seq_length, int(max_positions) - special_tokens)
    if esm_truncation_length < max_seq_length:
        print(
            "[emulator_bench] ESM input capped at "
            f"{esm_truncation_length} residues because {model_name} exposes "
            f"max_positions={max_positions} including special tokens",
            flush=True,
        )

    batch_converter = alphabet.get_batch_converter(esm_truncation_length)
    batches = list(batch_by_tokens(items, toks_per_batch))
    for batch in tqdm(batches, desc="ESM batches", unit="batch"):
        labels = [item[0] for item in batch]
        sequences = [item[1] for item in batch]
        output_paths = [item[2] for item in batch]
        _, _, toks = batch_converter(list(zip(labels, sequences)))
        toks = toks.to(device)
        with torch.no_grad():
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
        reps = out["representations"][repr_layer].cpu()

        for label, sequence, output_path, representation in zip(
            labels, sequences, output_paths, reps
        ):
            truncate_len = min(esm_truncation_length, len(sequence))
            mean_rep = representation[1 : truncate_len + 1].mean(0).clone()
            ensure_dir(output_path.parent)
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            torch.save({"label": label, "mean_representations": {repr_layer: mean_rep}}, tmp_path)
            tmp_path.replace(output_path)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    label = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if label is not None:
                records.append((label, "".join(chunks)))
            label = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if label is not None:
        records.append((label, "".join(chunks)))
    return records


def link_or_copy_cache(cache_file: Path, app_esm_file: Path) -> None:
    ensure_dir(app_esm_file.parent)
    if app_esm_file.exists() or app_esm_file.is_symlink():
        return
    try:
        relative_target = os.path.relpath(cache_file, start=app_esm_file.parent)
        app_esm_file.symlink_to(relative_target)
    except OSError:
        shutil.copy2(cache_file, app_esm_file)


def collect_unique_cache_items(metadata: dict, cache_root: Path) -> tuple[list[tuple[str, str, Path]], int]:
    seen: dict[str, tuple[str, str, Path]] = {}
    hits = 0
    for split_name in ("train", "val", "test"):
        manifest = load_manifest(metadata, split_name)
        for row in manifest.itertuples(index=False):
            cache_file = cache_root / f"{row.cache_key}.pt"
            if cache_file.exists():
                hits += 1
            else:
                seen[row.cache_key] = (row.Entry, row.Sequence, cache_file)
    return list(seen.values()), hits


def materialize_app_esm_links(metadata: dict, cache_root: Path) -> None:
    app_esm_root = APP_DATA_DIR / "esm_data"
    for split_name in ("train", "val", "test"):
        manifest = load_manifest(metadata, split_name)
        for row in tqdm(
            manifest.itertuples(index=False),
            total=len(manifest),
            desc=f"{metadata['split_group']} {split_name} cache links",
            leave=False,
        ):
            cache_file = cache_root / f"{row.cache_key}.pt"
            if not cache_file.exists():
                raise FileNotFoundError(f"Missing shared cache file: {cache_file}")
            link_or_copy_cache(cache_file, app_esm_root / f"{row.Entry}.pt")


def extract_masked_singleton_embeddings(
    train_data_name: str,
    *,
    model_name: str,
    repr_layer: int,
    toks_per_batch: int,
    device: torch.device,
    max_seq_length: int,
) -> None:
    add_clean_to_path()
    from CLEAN.utils import mutate_single_seq_ECs

    with pushd(APP_DIR):
        masked_fasta_name = mutate_single_seq_ECs(train_data_name)
    masked_fasta = APP_DATA_DIR / f"{masked_fasta_name}.fasta"
    if not masked_fasta.exists():
        return

    app_esm_root = APP_DATA_DIR / "esm_data"
    missing = []
    for label, sequence in read_fasta(masked_fasta):
        output_path = app_esm_root / f"{label}.pt"
        if not output_path.exists():
            missing.append((label, sequence, output_path))
    if missing:
        print(
            f"[emulator_bench] extracting {len(missing)} masked singleton-positive ESM features",
            flush=True,
        )
        extract_esm_features(
            missing,
            model_name=model_name,
            repr_layer=repr_layer,
            toks_per_batch=toks_per_batch,
            device=device,
            max_seq_length=max_seq_length,
        )


def compute_distance_map(train_data_name: str) -> None:
    add_clean_to_path()
    from CLEAN.utils import compute_esm_distance

    ensure_dir(APP_DATA_DIR / "distance_map")
    with pushd(APP_DIR):
        compute_esm_distance(train_data_name)


def main() -> None:
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    runs_root = resolve_path(args.runs_root)
    cache_root = resolve_path(args.cache_root)
    ensure_dir(runs_root)
    ensure_dir(cache_root)
    ensure_dir(APP_DATA_DIR / "esm_data")

    groups = select_split_groups(dataset_root, args.split_group)
    device = choose_device(args.device)
    completed = []
    for group in groups:
        print(f"[emulator_bench] preparing split group {group.name}", flush=True)
        metadata = prepare_split_group(
            group,
            dataset_root=dataset_root,
            runs_root=runs_root,
            label_policy=args.label_policy,
            max_sequence_length=args.max_seq_length,
            limit_per_split=args.limit_per_split,
        )
        metadata["cache_root"] = str(cache_root)
        metadata["esm_model"] = args.esm_model
        metadata["repr_layer"] = args.repr_layer
        metadata["toks_per_batch"] = args.toks_per_batch

        if not args.skip_esm:
            missing_items, hit_count = collect_unique_cache_items(metadata, cache_root)
            print(
                f"[emulator_bench] shared ESM cache hits={hit_count}, "
                f"misses={len(missing_items)}",
                flush=True,
            )
            extract_esm_features(
                missing_items,
                model_name=args.esm_model,
                repr_layer=args.repr_layer,
                toks_per_batch=args.toks_per_batch,
                device=device,
                max_seq_length=args.max_seq_length,
            )
            materialize_app_esm_links(metadata, cache_root)
            extract_masked_singleton_embeddings(
                metadata["clean_data"]["train"],
                model_name=args.esm_model,
                repr_layer=args.repr_layer,
                toks_per_batch=args.toks_per_batch,
                device=device,
                max_seq_length=args.max_seq_length,
            )

        if not args.skip_distance_map:
            compute_distance_map(metadata["clean_data"]["train"])

        write_json(Path(metadata["run_root"]) / "metadata.json", metadata)
        completed.append(metadata)

    write_json(runs_root / "last_cache_run.json", {"runs": completed})


if __name__ == "__main__":
    main()
