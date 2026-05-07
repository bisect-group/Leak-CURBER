from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_adapter import DEFAULT_MAX_SEQUENCE_LENGTH, select_split_groups
from .utils import (
    BASELINE_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNS_ROOT,
    conda_python,
    existing_train_metadata_checkpoint,
    find_spooler,
    seed_results_root,
    seed_run_root,
    seed_train_metadata_path,
    shell_join,
    split_group_slug,
    submit_ts_job,
    wait_for_ts_jobs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue Clipzyme Leak-CURBER cache/train/eval with ts")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--env-name", default="clipzyme")
    parser.add_argument("--spooler-bin", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--atom-map-batch-size", type=int, default=32)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--available-gpus", default="${CUDA_VISIBLE_DEVICES:-0}")
    parser.add_argument("--gpus-per-job", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--clip-freeze-esm", action="store_true")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--skip-native-eval", action="store_true")
    parser.add_argument("--skip-care", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def with_repo_prefix(command: list[str], cuda_visible_devices: str | None) -> str:
    prefix = f"cd {shell_join([BASELINE_ROOT])}"
    cmd = shell_join(command)
    if cuda_visible_devices is not None:
        cmd = f"env CUDA_VISIBLE_DEVICES={shell_join([cuda_visible_devices])} {cmd}"
    return f"{prefix} && {cmd}"


def main() -> None:
    args = parse_args()
    if args.gpus_per_job < 0:
        raise ValueError(f"--gpus-per-job must be >= 0, got {args.gpus_per_job}")
    find_spooler(args.spooler_bin)
    groups = select_split_groups(args.dataset_root, args.split_group)
    seeds = args.seed if args.seed else [42, 43, 44]
    jobs = []

    for group in groups:
        split_slug = split_group_slug(group.name)
        cache_command = [
            *conda_python(args.env_name),
            "-m",
            "emulator_bench.cache_features",
            "--dataset-root",
            args.dataset_root,
            "--split-group",
            group.name,
            "--runs-root",
            args.runs_root,
            "--cache-root",
            args.cache_root,
            "--label-policy",
            args.label_policy,
            "--max-sequence-length",
            str(args.max_sequence_length),
            "--atom-map-batch-size",
            str(args.atom_map_batch_size),
        ]
        if args.limit_per_split is not None:
            cache_command.extend(["--limit-per-split", str(args.limit_per_split)])
        cache_job = submit_ts_job(
            with_repo_prefix(cache_command, args.cuda_visible_devices),
            label=f"clipzyme-cache-{split_slug}",
            log_name=f"clipzyme-cache-{split_slug}.log",
            gpus=args.gpus_per_job,
            spooler_bin=args.spooler_bin,
        )

        for seed in seeds:
            train_command = [
                *conda_python(args.env_name),
                "-m",
                "emulator_bench.train",
                "--split-group",
                group.name,
                "--runs-root",
                args.runs_root,
                "--epochs",
                str(args.epochs),
                "--precision",
                args.precision,
                "--seed",
                str(seed),
                "--available-gpus",
                args.available_gpus,
                "--num-workers",
                str(args.num_workers),
            ]
            if args.batch_size is not None:
                train_command.extend(["--batch-size", str(args.batch_size)])
            if args.accumulate_grad_batches is not None:
                train_command.extend(
                    ["--accumulate-grad-batches", str(args.accumulate_grad_batches)]
                )
            if args.clip_freeze_esm:
                train_command.append("--clip-freeze-esm")
            eval_command = [
                *conda_python(args.env_name),
                "-m",
                "emulator_bench.evaluate",
                "--split-group",
                group.name,
                "--runs-root",
                args.runs_root,
                "--eval-split",
                args.eval_split,
                "--seed",
                str(seed),
                "--available-gpus",
                args.available_gpus,
                "--num-workers",
                str(args.num_workers),
            ]
            if args.skip_native_eval:
                eval_command.append("--skip-native")
            if args.skip_care:
                eval_command.append("--skip-care")
            slug = f"{split_slug}-seed{seed}"
            checkpoint = existing_train_metadata_checkpoint(group.name, seed, args.runs_root)
            train_job = None
            train_skipped = checkpoint is not None
            if train_skipped:
                print(
                    "[emulator_bench] skipping queued train for "
                    f"{group.name} seed {seed}; checkpoint exists: {checkpoint}",
                    flush=True,
                )
            else:
                train_job = submit_ts_job(
                    with_repo_prefix(train_command, args.cuda_visible_devices),
                    label=f"clipzyme-train-{slug}",
                    log_name=f"clipzyme-train-{slug}.log",
                    depends_on=[cache_job],
                    gpus=args.gpus_per_job,
                    spooler_bin=args.spooler_bin,
                )
            eval_job = submit_ts_job(
                with_repo_prefix(eval_command, args.cuda_visible_devices),
                label=f"clipzyme-eval-{slug}",
                log_name=f"clipzyme-eval-{slug}.log",
                depends_on=[train_job or cache_job],
                gpus=args.gpus_per_job,
                spooler_bin=args.spooler_bin,
            )
            jobs.append(
                {
                    "split_group": group.name,
                    "seed": int(seed),
                    "cache_job": cache_job,
                    "train_job": train_job,
                    "train_skipped": train_skipped,
                    "existing_checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "eval_job": eval_job,
                    "expected_outputs": {
                        "seed_run_root": str(seed_run_root(group.name, seed, args.runs_root)),
                        "train_metadata": str(
                            seed_train_metadata_path(group.name, seed, args.runs_root)
                        ),
                        "results_root": str(seed_results_root(group.name, seed, args.runs_root)),
                    },
                }
            )

    write_json(Path(args.runs_root) / "queued_jobs.json", jobs)
    if args.wait:
        wait_for_ts_jobs([job["eval_job"] for job in jobs], spooler_bin=args.spooler_bin)


if __name__ == "__main__":
    main()
