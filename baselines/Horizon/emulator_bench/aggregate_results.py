from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import DEFAULT_RUNS_ROOT, ensure_dir, read_json


METRIC_SECTIONS = ("primary_native_horizon", "care_task2", "supplemental_ec_ranking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Horizon Leak-CURBER seed metrics")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--long-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def iter_metric_files(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    return sorted(root.glob("*/seeds/*/results/*/metrics.json"))


def flatten_scalar_metrics(data: dict, prefix: str = "") -> Iterable[tuple[str, float]]:
    for key, value in data.items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield metric_name, float(value)
        elif isinstance(value, dict):
            yield from flatten_scalar_metrics(value, metric_name)


def metric_context(metric_path: Path) -> tuple[str, str, str]:
    seed = metric_path.parents[2].name
    split_group = metric_path.parents[4].name
    eval_split = metric_path.parent.name
    return split_group, seed, eval_split


def collect_metric_rows(runs_root: str | Path) -> list[dict]:
    rows = []
    for metric_path in iter_metric_files(runs_root):
        metrics = read_json(metric_path)
        path_split_group, path_seed, path_eval_split = metric_context(metric_path)
        split_group = str(metrics.get("split_group", path_split_group))
        seed = str(metrics.get("seed", path_seed))
        eval_split = str(metrics.get("eval_split", path_eval_split))
        ec_candidate_splits = metrics.get("ec_candidate_splits", [])
        if isinstance(ec_candidate_splits, list):
            ec_candidate_splits = ";".join(str(split) for split in ec_candidate_splits)
        else:
            ec_candidate_splits = str(ec_candidate_splits)
        ec_scoring = str(metrics.get("ec_scoring", ""))
        score_similarity = str(metrics.get("score_similarity", ""))
        direction_aggregation = str(metrics.get("direction_aggregation", ""))
        max_rank_columns = str(metrics.get("max_rank_columns", ""))
        result_name = str(metrics.get("result_name", path_eval_split))

        for section in METRIC_SECTIONS:
            section_metrics = metrics.get(section)
            if not isinstance(section_metrics, dict):
                continue
            for metric, value in flatten_scalar_metrics(section_metrics, section):
                rows.append(
                    {
                        "split_group": split_group,
                        "seed": seed,
                        "eval_split": eval_split,
                        "result_name": result_name,
                        "ec_candidate_splits": ec_candidate_splits,
                        "ec_scoring": ec_scoring,
                        "score_similarity": score_similarity,
                        "direction_aggregation": direction_aggregation,
                        "max_rank_columns": max_rank_columns,
                        "metric": metric,
                        "value": value,
                        "metrics_path": str(metric_path),
                    }
                )
    return rows


def write_aggregate_csvs(
    rows: list[dict],
    *,
    long_csv: str | Path,
    summary_csv: str | Path,
) -> tuple[Path, Path]:
    if not rows:
        raise FileNotFoundError("No seed metric files found under runs-root")

    long_path = Path(long_csv)
    summary_path = Path(summary_csv)
    ensure_dir(long_path.parent)
    ensure_dir(summary_path.parent)

    long_df = pd.DataFrame(rows).sort_values(
        [
            "split_group",
            "eval_split",
            "result_name",
            "ec_candidate_splits",
            "ec_scoring",
            "score_similarity",
            "direction_aggregation",
            "max_rank_columns",
            "metric",
            "seed",
        ]
    )
    summary_df = (
        long_df.groupby(
            [
                "split_group",
                "eval_split",
                "result_name",
                "ec_candidate_splits",
                "ec_scoring",
                "score_similarity",
                "direction_aggregation",
                "max_rank_columns",
                "metric",
            ]
        )["value"]
        .agg(count="count", mean="mean", std="std")
        .reset_index()
    )
    summary_df["std"] = summary_df["std"].fillna(0.0)

    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return long_path, summary_path


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    long_csv = args.long_csv or runs_root / "aggregated_seed_metrics_long.csv"
    summary_csv = args.summary_csv or runs_root / "aggregated_seed_metrics_summary.csv"

    rows = collect_metric_rows(runs_root)
    long_path, summary_path = write_aggregate_csvs(
        rows,
        long_csv=long_csv,
        summary_csv=summary_csv,
    )
    print(f"[emulator_bench] long metrics: {long_path}", flush=True)
    print(f"[emulator_bench] summary metrics: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
