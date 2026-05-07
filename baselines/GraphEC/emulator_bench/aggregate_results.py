from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import DEFAULT_RUNS_ROOT, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate GraphEC Leak-CURBER metrics")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--long-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def flatten(data: dict, prefix: str = ""):
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from flatten(value, name)
        elif isinstance(value, (int, float)) and value is not None:
            yield name, float(value)


def context_from_path(path: Path) -> tuple[str, str, str]:
    eval_split = path.name.removesuffix("_metrics.json")
    seed = path.parents[1].name
    split_group = path.parents[3].name
    return split_group, seed, eval_split


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    rows = []
    for metrics_path in sorted(runs_root.glob("*/seeds/*/results/*_metrics.json")):
        metrics = read_json(metrics_path)
        split_group, seed, eval_split = context_from_path(metrics_path)
        for section in ("graphec", "care_task1", "supplemental"):
            values = metrics.get(section, {})
            if not isinstance(values, dict):
                continue
            for metric, value in flatten(values, section):
                rows.append(
                    {
                        "split_group": metrics.get("split_group", split_group),
                        "seed": metrics.get("seed", seed),
                        "eval_split": metrics.get("eval_split", eval_split),
                        "metric": metric,
                        "value": value,
                        "metrics_path": str(metrics_path),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No metric JSON files found under {runs_root}")
    long_df = pd.DataFrame(rows).sort_values(["split_group", "eval_split", "metric", "seed"])
    summary_df = (
        long_df.groupby(["split_group", "eval_split", "metric"], as_index=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    long_csv = Path(args.long_csv) if args.long_csv else runs_root / "aggregated_seed_metrics_long.csv"
    summary_csv = (
        Path(args.summary_csv) if args.summary_csv else runs_root / "aggregated_seed_metrics_summary.csv"
    )
    long_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[emulator_bench] long metrics: {long_csv}", flush=True)
    print(f"[emulator_bench] summary metrics: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
