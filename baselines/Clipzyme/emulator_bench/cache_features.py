from __future__ import annotations

import argparse

from .dataset_adapter import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
    prepare_split_group,
    select_split_groups,
)
from .utils import DEFAULT_CACHE_ROOT, DEFAULT_DATASET_ROOT, DEFAULT_RUNS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Clipzyme Leak-CURBER features")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--atom-map-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = select_split_groups(args.dataset_root, args.split_group)
    for group in groups:
        metadata = prepare_split_group(
            group,
            dataset_root=args.dataset_root,
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            label_policy=args.label_policy,
            max_sequence_length=args.max_sequence_length,
            limit_per_split=args.limit_per_split,
            atom_map_batch_size=args.atom_map_batch_size,
        )
        cache_stats = metadata["atom_map_cache"]
        print(
            "[emulator_bench] cache complete for "
            f"{group.name}: {metadata['clipzyme_rows']} Clipzyme rows "
            f"(atom-map hits={cache_stats['cache_hits']}, "
            f"misses={cache_stats['cache_misses']}, "
            f"failed_cached={cache_stats['failed_cached']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
