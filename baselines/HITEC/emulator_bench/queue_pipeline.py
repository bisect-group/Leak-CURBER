from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_adapter import select_split_groups
from .utils import (
    BASELINE_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SEEDS,
    DEFAULT_VOCAB_PATH,
    conda_python,
    ensure_dir,
    seed_results_root_for_split,
    seed_run_root_for_split,
    seed_train_metadata_path_for_split,
    split_group_slug,
    submit_ts_job,
    wait_for_ts_jobs,
    with_repo_prefix,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Queue HITEC Leak-CURBER cache/train/eval with ts")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append", help="Split group to run")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--vocab-path", default=str(DEFAULT_VOCAB_PATH))
    parser.add_argument("--env-name", default="hitec")
    parser.add_argument("--spooler-bin", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--seed", type=int, action="append", help="Random seed")
    parser.add_argument("--model-dimension", type=int, default=DEFAULT_MODEL_DIMENSION)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--eval-split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--rank-limit", type=int, default=50)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--cache-gpus", type=int, default=0)
    parser.add_argument("--train-gpus", type=int, default=1)
    parser.add_argument("--eval-gpus", type=int, default=1)
    parser.add_argument("--include-inter-stage", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    groups = select_split_groups(args.dataset_root, args.split_group)
    split_names = [group.name for group in groups]
    seeds = args.seed if args.seed else list(DEFAULT_SEEDS)
    eval_batch_size = args.eval_batch_size or args.batch_size
    logs_dir = ensure_dir(Path(args.runs_root) / "logs")

    if args.include_inter_stage:
        print(
            "[emulator_bench] warning: --include-inter-stage is accepted but not queued; "
            "Leak-CURBER metrics use HITEC infer-mode predictions.",
            flush=True,
        )

    cache_command = [
        *conda_python(args.env_name),
        "-m",
        "emulator_bench.cache_features",
        "--dataset-root",
        args.dataset_root,
        "--runs-root",
        args.runs_root,
        "--cache-root",
        args.cache_root,
        "--vocab-path",
        args.vocab_path,
        "--model-dimension",
        str(args.model_dimension),
    ]
    if args.max_tokens is not None:
        cache_command.extend(["--max-tokens", str(args.max_tokens)])
    for split_group in split_names:
        cache_command.extend(["--split-group", split_group])
    if args.limit_per_split is not None:
        cache_command.extend(["--limit-per-split", str(args.limit_per_split)])

    cache_slug = "{}groups".format(len(split_names)) if len(split_names) > 1 else split_group_slug(split_names[0])
    cache_job = submit_ts_job(
        with_repo_prefix(cache_command, args.cuda_visible_devices),
        label="hitec-cache-{}".format(cache_slug),
        log_name=str(logs_dir / "hitec-cache-{}.log".format(cache_slug)),
        gpus=args.cache_gpus,
        spooler_bin=args.spooler_bin,
    )

    jobs = []
    for split_group in split_names:
        for seed in seeds:
            slug = "{}_seed{}".format(split_group_slug(split_group), seed)
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
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--precision",
                args.precision,
                "--seed",
                str(seed),
                "--model-dimension",
                str(args.model_dimension),
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
                "--batch-size",
                str(eval_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--rank-limit",
                str(args.rank_limit),
                "--seed",
                str(seed),
            ]

            train_job = submit_ts_job(
                with_repo_prefix(train_command, args.cuda_visible_devices),
                label="hitec-train-{}".format(slug),
                log_name=str(logs_dir / "hitec-train-{}.log".format(slug)),
                depends_on=[cache_job],
                gpus=args.train_gpus,
                spooler_bin=args.spooler_bin,
            )
            eval_job = submit_ts_job(
                with_repo_prefix(eval_command, args.cuda_visible_devices),
                label="hitec-eval-{}".format(slug),
                log_name=str(logs_dir / "hitec-eval-{}.log".format(slug)),
                depends_on=[train_job],
                gpus=args.eval_gpus,
                spooler_bin=args.spooler_bin,
            )
            jobs.append(
                {
                    "split_group": split_group,
                    "seed": int(seed),
                    "cache_job": cache_job,
                    "train_job": train_job,
                    "eval_job": eval_job,
                    "expected_outputs": {
                        "seed_run_root": str(seed_run_root_for_split(split_group, seed, args.runs_root)),
                        "train_metadata": str(
                            seed_train_metadata_path_for_split(split_group, seed, args.runs_root)
                        ),
                        "checkpoint_dir": str(
                            seed_run_root_for_split(split_group, seed, args.runs_root) / "checkpoints"
                        ),
                        "results_root": str(seed_results_root_for_split(split_group, seed, args.runs_root)),
                    },
                }
            )

    queue_summary = {
        "baseline_root": str(BASELINE_ROOT),
        "dataset_root": args.dataset_root,
        "split_groups": split_names,
        "seeds": seeds,
        "cache_job": cache_job,
        "jobs": jobs,
    }
    write_json(Path(args.runs_root) / "queued_jobs.json", queue_summary)
    if args.wait:
        wait_for_ts_jobs([job["eval_job"] for job in jobs], spooler_bin=args.spooler_bin)


if __name__ == "__main__":
    main()
