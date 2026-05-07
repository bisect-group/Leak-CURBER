from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .dataset_adapter import (
    build_ec_vocab,
    prepare_split_group,
    select_split_groups,
    sequence_sha256,
)
from .utils import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_RUNS_ROOT,
    DEFAULT_VOCAB_PATH,
    TOKENIZER_PATH,
    ensure_dir,
    read_json,
    resolve_model_max_tokens,
    resolve_path,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare HITEC token caches and manifests")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", help="Split group to prepare")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--vocab-path", default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--rebuild-vocab", action="store_true")
    parser.add_argument(
        "--model-dimension",
        type=int,
        default=DEFAULT_MODEL_DIMENSION,
        help="HITEC transformer hidden size and positional limit.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Token length after BOS/truncation/padding. Defaults to --model-dimension.",
    )
    parser.add_argument("--limit-per-split", type=int, default=None)
    return parser.parse_args()


def load_tokenizer():
    import pickle

    with Path(TOKENIZER_PATH).open("rb") as handle:
        return pickle.load(handle)


def tokenize_sequence(sequence, tokenizer, max_tokens):
    token_ids = tokenizer.texts_to_sequences([str(sequence).lower()])[0]
    token_ids = [22] + token_ids
    token_ids = token_ids[:max_tokens]
    if len(token_ids) < max_tokens:
        token_ids += [0] * (max_tokens - len(token_ids))
    return torch.tensor(token_ids, dtype=torch.long)


def _load_cache_index(index_path):
    if index_path.exists():
        return read_json(index_path)
    return {}


def _validate_existing_cache(cache_path, cache_key, seq_hash, max_tokens):
    cached = torch.load(cache_path, map_location="cpu")
    if cached.get("sequence_sha256") != seq_hash:
        raise ValueError(
            "Cache key conflict for {}: {} != {}".format(
                cache_key,
                cached.get("sequence_sha256"),
                seq_hash,
            )
        )
    if int(cached.get("max_tokens", -1)) != int(max_tokens):
        raise ValueError(
            "Cache token length conflict for {}: {} != {}".format(
                cache_key,
                cached.get("max_tokens"),
                max_tokens,
            )
        )
    tokens = cached.get("tokens")
    if tokens is None or int(tokens.numel()) != int(max_tokens):
        raise ValueError("Cached token tensor for {} is not length {}".format(cache_key, max_tokens))


def populate_token_cache(metadata, cache_root, max_tokens):
    import pandas as pd

    tokenizer = load_tokenizer()
    cache_root = ensure_dir(cache_root)
    index_path = cache_root / "cache_index.json"
    cache_index = _load_cache_index(index_path)
    completed = {}
    hits = 0
    misses = 0

    for split_name, manifest_path in metadata["manifests"].items():
        df = pd.read_csv(manifest_path)
        token_paths = []
        iterator = tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="{} {} tokens".format(metadata["split_group"], split_name),
            unit="seq",
        )
        for row in iterator:
            cache_key = str(row.cache_key)
            cache_path = cache_root / "{}.pt".format(cache_key)
            seq_hash = sequence_sha256(str(row.Sequence))
            existing = cache_index.get(cache_key)
            if existing:
                if existing.get("sequence_sha256") != seq_hash:
                    raise ValueError(
                        "Cache key conflict for {}: {} != {}".format(
                            cache_key,
                            existing.get("sequence_sha256"),
                            seq_hash,
                        )
                    )
                if int(existing.get("max_tokens", -1)) != int(max_tokens):
                    raise ValueError(
                        "Cache token length conflict for {}: {} != {}".format(
                            cache_key,
                            existing.get("max_tokens"),
                            max_tokens,
                        )
                    )
            if cache_path.exists():
                _validate_existing_cache(cache_path, cache_key, seq_hash, max_tokens)
                hits += 1
            else:
                token_tensor = tokenize_sequence(str(row.Sequence), tokenizer, max_tokens)
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                torch.save(
                    {
                        "cache_key": cache_key,
                        "entry": str(row.Entry),
                        "tokens": token_tensor,
                        "sequence_sha256": seq_hash,
                        "max_tokens": int(max_tokens),
                    },
                    tmp_path,
                )
                tmp_path.replace(cache_path)
                misses += 1
            cache_index[cache_key] = {
                "sequence_sha256": seq_hash,
                "max_tokens": int(max_tokens),
                "path": str(cache_path),
            }
            token_paths.append(str(cache_path))

        df["token_cache_path"] = token_paths
        df.to_csv(manifest_path, index=False)
        completed[split_name] = {
            "rows": int(len(df)),
            "manifest": str(manifest_path),
        }

    write_json(index_path, cache_index)
    return {"cache_hits": int(hits), "cache_misses": int(misses), "splits": completed}


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    runs_root = resolve_path(args.runs_root)
    cache_root = resolve_path(args.cache_root)
    vocab_path = resolve_path(args.vocab_path)
    ensure_dir(runs_root)
    ensure_dir(cache_root)
    ensure_dir(vocab_path.parent)
    max_tokens = resolve_model_max_tokens(args.max_tokens, args.model_dimension)

    vocab = build_ec_vocab(dataset_root, vocab_path=vocab_path, rebuild=args.rebuild_vocab)
    groups = select_split_groups(dataset_root, args.split_group)
    completed = []
    for group in groups:
        print("[emulator_bench] preparing split group {}".format(group.name), flush=True)
        metadata = prepare_split_group(
            group,
            dataset_root=dataset_root,
            vocab=vocab,
            runs_root=runs_root,
            cache_root=cache_root,
            max_tokens=max_tokens,
            model_dimension=args.model_dimension,
            limit_per_split=args.limit_per_split,
        )
        cache_stats = populate_token_cache(metadata, cache_root, max_tokens)
        metadata["cache_root"] = str(cache_root)
        metadata["vocab_path"] = str(vocab_path)
        metadata["cache_stats"] = cache_stats
        write_json(Path(metadata["run_root"]) / "metadata.json", metadata)
        completed.append(metadata)

    write_json(runs_root / "cache_summary.json", completed)
    print("[emulator_bench] prepared {} split group(s)".format(len(completed)), flush=True)


if __name__ == "__main__":
    main()
