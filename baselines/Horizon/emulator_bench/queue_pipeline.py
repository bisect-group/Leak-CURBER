from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Queue Horizon Leak-CURBER cache/train/eval with ts")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--env-name", default="horizon")
    parser.add_argument("--spooler-bin", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-sequence-length", type=int, default=5000)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--skip-fingerprint-validation", action="store_true")
    parser.add_argument(
        "--embedding-source",
        choices=["prott5", "cache-only", "deterministic"],
        default="prott5",
    )
    parser.add_argument("--allow-deterministic-embeddings", action="store_true")
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gpus-per-job", type=int, default=1)
    parser.add_argument("--seed", type=int, action="append")
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
    seeds = args.seed if args.seed else [42, 43, 44]
    jobs = []

    for split_group in args.split_group:
        split_slug = split_group_slug(split_group)
        cache_command = [
            *conda_python(args.env_name),
            "-m",
            "emulator_bench.cache_features",
            "--dataset-root",
            args.dataset_root,
            "--split-group",
            split_group,
            "--runs-root",
            args.runs_root,
            "--cache-root",
            args.cache_root,
            "--label-policy",
            args.label_policy,
            "--max-sequence-length",
            str(args.max_sequence_length),
            "--embedding-source",
            args.embedding_source,
        ]
        if args.allow_deterministic_embeddings:
            cache_command.append("--allow-deterministic-embeddings")
        if args.limit_per_split is not None:
            cache_command.extend(["--limit-per-split", str(args.limit_per_split)])
        if args.skip_fingerprint_validation:
            cache_command.append("--skip-fingerprint-validation")

        cache_job = submit_ts_job(
            with_repo_prefix(cache_command, args.cuda_visible_devices),
            label=f"horizon-cache-{split_slug}",
            log_name=f"horizon-cache-{split_slug}.log",
            gpus=args.gpus_per_job,
            spooler_bin=args.spooler_bin,
        )

        for seed in seeds:
            train_command = [
                *conda_python(args.env_name),
                "-m",
                "emulator_bench.train",
                "--split-group",
                split_group,
                "--runs-root",
                args.runs_root,
                "--epochs",
                str(args.epochs),
                "--precision",
                args.precision,
                "--seed",
                str(seed),
            ]
            eval_command = [
                *conda_python(args.env_name),
                "-m",
                "emulator_bench.evaluate",
                "--split-group",
                split_group,
                "--runs-root",
                args.runs_root,
                "--eval-split",
                args.eval_split,
                "--seed",
                str(seed),
            ]
            checkpoint = existing_train_metadata_checkpoint(split_group, seed, args.runs_root)
            train_job = None
            train_skipped = checkpoint is not None
            if train_skipped:
                print(
                    "[emulator_bench] skipping queued train for "
                    f"{split_group} seed {seed}; checkpoint exists: {checkpoint}",
                    flush=True,
                )
            else:
                train_job = submit_ts_job(
                    with_repo_prefix(train_command, args.cuda_visible_devices),
                    label=f"horizon-train-{split_slug}-seed{seed}",
                    log_name=f"horizon-train-{split_slug}-seed{seed}.log",
                    depends_on=[cache_job],
                    gpus=args.gpus_per_job,
                    spooler_bin=args.spooler_bin,
                )
            eval_job = submit_ts_job(
                with_repo_prefix(eval_command, args.cuda_visible_devices),
                label=f"horizon-eval-{split_slug}-seed{seed}",
                log_name=f"horizon-eval-{split_slug}-seed{seed}.log",
                depends_on=[train_job or cache_job],
                gpus=args.gpus_per_job,
                spooler_bin=args.spooler_bin,
            )
            jobs.append(
                {
                    "split_group": split_group,
                    "seed": int(seed),
                    "cache_job": cache_job,
                    "train_job": train_job,
                    "train_skipped": train_skipped,
                    "existing_checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "eval_job": eval_job,
                    "expected_outputs": {
                        "seed_run_root": str(seed_run_root(split_group, seed, args.runs_root)),
                        "train_metadata": str(
                            seed_train_metadata_path(split_group, seed, args.runs_root)
                        ),
                        "results_root": str(seed_results_root(split_group, seed, args.runs_root)),
                    },
                }
            )

    write_json(Path(args.runs_root) / "queued_jobs.json", jobs)
    if args.wait:
        wait_for_ts_jobs([job["eval_job"] for job in jobs], spooler_bin=args.spooler_bin)


if __name__ == "__main__":
    main()
