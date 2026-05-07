from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .utils import (
    BASELINE_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNS_ROOT,
    conda_python,
    find_spooler,
    seed_results_root_for_split,
    seed_run_root_for_split,
    seed_train_metadata_path_for_split,
    shell_join,
    split_group_slug,
    submit_ts_job,
    wait_for_ts_jobs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue GraphEC Leak-CURBER cache/train with ts and optionally run eval directly"
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", required=True)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--env-name", default="graphec")
    parser.add_argument("--spooler-bin", default=None)
    parser.add_argument("--execution-mode", choices=["ts", "direct"], default="ts")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--max-seq-length", type=int, default=1022)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--prottrans-model-path", default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--eval-mode", choices=["direct", "ts"], default="direct")
    parser.add_argument("--eval-cuda-device", default="0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--cache-gpus", type=int, default=1)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--eval-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--skip-feature-generation", action="store_true")
    return parser.parse_args()


def with_repo_prefix(command: list[str], cuda_visible_devices: str | None) -> str:
    cmd = shell_join(command)
    if cuda_visible_devices is not None:
        cmd = f"env CUDA_VISIBLE_DEVICES={shell_join([cuda_visible_devices])} {cmd}"
    return f"cd {shell_join([BASELINE_ROOT])} && {cmd}"


def main() -> None:
    args = parse_args()
    seeds = args.seed if args.seed else [0]
    if args.execution_mode == "ts":
        find_spooler(args.spooler_bin)
    jobs = []
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
            "--cache-root",
            args.cache_root,
            "--max-seq-length",
            str(args.max_seq_length),
        ]
        if args.limit_per_split is not None:
            cache_command.extend(["--limit-per-split", str(args.limit_per_split)])
        if args.prottrans_model_path:
            cache_command.extend(["--prottrans-model-path", args.prottrans_model_path])
        if args.skip_feature_generation:
            cache_command.append("--skip-feature-generation")
        split_slug = split_group_slug(split_group)
        cache_invocation = with_repo_prefix(cache_command, args.cuda_visible_devices)
        cache_job = None
        if args.execution_mode == "ts":
            cache_job = submit_ts_job(
                cache_invocation,
                label=f"graphec-cache-{split_slug}",
                log_name=f"graphec-cache-{split_slug}.log",
                gpus=args.cache_gpus,
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
                "--cache-root",
                args.cache_root,
                "--env-name",
                args.env_name,
                "--seed",
                str(seed),
                "--epochs",
                str(args.epochs),
                "--folds",
                str(args.folds),
                "--batch-size",
                str(args.batch_size),
                "--precision",
                args.precision,
            ]
            eval_command = [
                *conda_python(args.env_name),
                "-m",
                "emulator_bench.evaluate",
                "--split-group",
                split_group,
                "--runs-root",
                args.runs_root,
                "--cache-root",
                args.cache_root,
                "--seed",
                str(seed),
                "--eval-split",
                args.eval_split,
                "--batch-size",
                str(args.batch_size),
                "--precision",
                args.precision,
            ]
            slug = f"{split_slug}_seed{seed}"
            train_invocation = with_repo_prefix(train_command, args.cuda_visible_devices)
            train_job = None
            if args.execution_mode == "ts":
                train_job = submit_ts_job(
                    train_invocation,
                    label=f"graphec-train-{slug}",
                    log_name=f"graphec-train-{slug}.log",
                    depends_on=[cache_job],
                    gpus=args.train_gpus,
                    spooler_bin=args.spooler_bin,
                )
            direct_eval_command = with_repo_prefix(eval_command, args.eval_cuda_device)
            eval_job = None
            if args.execution_mode == "ts" and args.eval_mode == "ts":
                eval_job = submit_ts_job(
                    with_repo_prefix(eval_command, args.cuda_visible_devices),
                    label=f"graphec-eval-{slug}",
                    log_name=f"graphec-eval-{slug}.log",
                    depends_on=[train_job],
                    gpus=args.eval_gpus,
                    spooler_bin=args.spooler_bin,
                )
            elif args.execution_mode == "ts":
                print(
                    f"[emulator_bench] direct eval after train job {train_job}: {direct_eval_command}",
                    flush=True,
                )
            jobs.append(
                {
                    "split_group": split_group,
                    "seed": seed,
                    "execution_mode": args.execution_mode,
                    "cache_job": cache_job,
                    "train_job": train_job,
                    "eval_job": eval_job,
                    "cache_command": cache_invocation,
                    "train_command": train_invocation,
                    "eval_mode": args.eval_mode,
                    "direct_eval_command": direct_eval_command,
                    "expected_outputs": {
                        "seed_run_root": str(seed_run_root_for_split(split_group, seed, args.runs_root)),
                        "train_metadata": str(
                            seed_train_metadata_path_for_split(split_group, seed, args.runs_root)
                        ),
                        "results_root": str(seed_results_root_for_split(split_group, seed, args.runs_root)),
                    },
                }
            )
    write_json(Path(args.runs_root) / "queued_jobs.json", jobs)
    if args.wait:
        if args.execution_mode == "ts" and args.eval_mode == "ts":
            wait_for_ts_jobs([job["eval_job"] for job in jobs], spooler_bin=args.spooler_bin)
        elif args.execution_mode == "ts":
            wait_for_ts_jobs([job["train_job"] for job in jobs], spooler_bin=args.spooler_bin)
            for job in jobs:
                subprocess.run(["bash", "-lc", job["direct_eval_command"]], check=True)
        else:
            for job in jobs:
                subprocess.run(["bash", "-lc", job["cache_command"]], check=True)
                subprocess.run(["bash", "-lc", job["train_command"]], check=True)
                subprocess.run(["bash", "-lc", job["direct_eval_command"]], check=True)


if __name__ == "__main__":
    main()
