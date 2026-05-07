from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .utils import DEFAULT_RUNS_ROOT, ensure_dir, flatten_dict, read_json


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate HITEC Leak-CURBER metrics")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def collect_metric_rows(runs_root):
    rows = []
    for metrics_path in sorted(Path(runs_root).glob("*/seeds/*/results/*_metrics.json")):
        metrics = read_json(metrics_path)
        row = flatten_dict(metrics)
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    return rows


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    rows = collect_metric_rows(runs_root)
    if not rows:
        raise FileNotFoundError("No metrics JSON files found under {}".format(runs_root))

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv) if args.output_csv else runs_root / "aggregate_metrics.csv"
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"seed", "metrics_path"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    group_cols = [col for col in ["split_group", "eval_split"] if col in df.columns]
    if group_cols and numeric_cols:
        summary = df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
        summary.columns = [
            ".".join([str(part) for part in col if str(part)])
            if isinstance(col, tuple)
            else str(col)
            for col in summary.columns
        ]
    else:
        summary = pd.DataFrame()

    summary_csv = Path(args.summary_csv) if args.summary_csv else runs_root / "aggregate_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("[emulator_bench] aggregate metrics: {}".format(output_csv), flush=True)
    print("[emulator_bench] aggregate summary: {}".format(summary_csv), flush=True)


if __name__ == "__main__":
    main()
