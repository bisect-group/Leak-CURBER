from __future__ import annotations

import argparse
from pathlib import Path

from .utils import (
    BASELINE_ROOT,
    DEFAULT_RUNS_ROOT,
    conda_python,
    find_spooler,
    seed_results_root_for_split,
    seed_run_root_for_split,
    seed_train_metadata_path_for_split,
    shell_join,
    submit_ts_job,
    split_group_slug,
    wait_for_ts_jobs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue CLEAN Leak-CURBER cache/train/eval with ts")
    parser.add_argument(
        "--dataset-root",
        default="../../data/processed/datasets/enzyme_classification_dataset",
    )
    parser.add_argument("--split-group", action="append", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--env-name", default="current")
    parser.add_argument("--spooler-bin", default=None)
    parser.add_argument("--epochs", type=int, default=7000)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument(
        "--gpus-per-job",
        type=int,
        default=1,
        help="Number of GPUs to request from task-spooler per queued job. Use 0 for CPU/debug.",
    )
    parser.add_argument("--seed", type=int, action="append", help="Random seed for CLEAN training")
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
    if args.cuda_visible_devices is not None and args.gpus_per_job > 0:
        print(
            "[emulator_bench] warning: --cuda-visible-devices overrides the GPU "
            "selected by task-spooler; prefer TS_VISIBLE_DEVICES to restrict auto scheduling",
            flush=True,
        )
    find_spooler(args.spooler_bin)
    jobs = []
    seeds = args.seed if args.seed else [1234]
    gpus_per_job = args.gpus_per_job

    for split_group in args.split_group:
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
            "--label-policy",
            args.label_policy,
            "--max-seq-length",
            str(args.max_seq_length),
        ]
        if args.limit_per_split is not None:
            cache_command.extend(["--limit-per-split", str(args.limit_per_split)])

        split_slug = split_group_slug(split_group)
        cache_job = submit_ts_job(
            with_repo_prefix(cache_command, args.cuda_visible_devices),
            label=f"clean-cache-{split_slug}",
            log_name=f"clean-cache-{split_slug}.log",
            gpus=gpus_per_job,
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
                "--env-name",
                args.env_name,
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

            slug = f"{split_slug}_seed{seed}"
            seed_run_root = seed_run_root_for_split(split_group, seed, args.runs_root)
            expected_outputs = {
                "seed_run_root": str(seed_run_root),
                "train_metadata": str(
                    seed_train_metadata_path_for_split(split_group, seed, args.runs_root)
                ),
                "checkpoint_dir": str(seed_run_root / "checkpoints"),
                "results_root": str(
                    seed_results_root_for_split(split_group, seed, args.runs_root)
                ),
            }
            train_job = submit_ts_job(
                with_repo_prefix(train_command, args.cuda_visible_devices),
                label=f"clean-train-{slug}",
                log_name=f"clean-train-{slug}.log",
                depends_on=[cache_job],
                gpus=gpus_per_job,
                spooler_bin=args.spooler_bin,
            )
            eval_job = submit_ts_job(
                with_repo_prefix(eval_command, args.cuda_visible_devices),
                label=f"clean-eval-{slug}",
                log_name=f"clean-eval-{slug}.log",
                depends_on=[train_job],
                gpus=gpus_per_job,
                spooler_bin=args.spooler_bin,
            )
            jobs.append(
                {
                    "split_group": split_group,
                    "seed": seed,
                    "gpus_per_job": gpus_per_job,
                    "cache_job": cache_job,
                    "train_job": train_job,
                    "eval_job": eval_job,
                    "expected_outputs": expected_outputs,
                }
            )

    write_json(Path(args.runs_root) / "queued_jobs.json", jobs)
    if args.wait:
        wait_for_ts_jobs(
            [job["eval_job"] for job in jobs],
            spooler_bin=args.spooler_bin,
        )


if __name__ == "__main__":
    main()
