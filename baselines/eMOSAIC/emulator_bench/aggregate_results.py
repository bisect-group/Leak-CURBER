"""
Aggregate eMOSAIC results across all splits and seeds.
Computes mean and variance for each metric across seeds, per split type, for Train/Val/Test.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

METRICS = ["mae", "mse", "rmse", "r2_score", "pearson", "spearman", "loss"]
SPLITS = {"train": "final_train_metrics", "val": "final_val_metrics", "test": "final_test_metrics"}

DATA_DIR = Path(os.environ.get("PROJECT_ROOT", ".")) / "data" / "processed" / "baselines" / "eMOSAIC"


def find_emosaic_results_dirs(base: Path) -> list[Path]:
    return sorted(base.rglob("emosaic_results"))


def get_split_label(results_dir: Path) -> str:
    """Build a human-readable label from the path above emosaic_results."""
    parts = results_dir.relative_to(DATA_DIR).parts[:-1]  # drop 'emosaic_results'
    return "/".join(parts)


def load_seed_metrics(results_dir: Path) -> dict[str, dict[str, list[float]]]:
    """
    Returns {split: {metric: [values across seeds]}}
    """
    accumulated: dict[str, dict[str, list[float]]] = {
        split: {m: [] for m in METRICS} for split in SPLITS
    }

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        summary_path = seed_dir / "run_summary.json"
        if not summary_path.exists():
            print(f"  WARNING: missing {summary_path}")
            continue
        with open(summary_path) as f:
            data = json.load(f)
        for split, key in SPLITS.items():
            metrics_block = data.get(key, {})
            for m in METRICS:
                if m in metrics_block:
                    accumulated[split][m].append(metrics_block[m])

    return accumulated


def main():
    results_dirs = find_emosaic_results_dirs(DATA_DIR)
    if not results_dirs:
        print("No emosaic_results directories found.")
        return

    rows = []
    for rd in results_dirs:
        label = get_split_label(rd)
        print(f"Processing: {label}")
        seed_metrics = load_seed_metrics(rd)

        for split in SPLITS:
            for metric in METRICS:
                values = seed_metrics[split][metric]
                if not values:
                    mean, var, n = float("nan"), float("nan"), 0
                else:
                    arr = np.array(values)
                    mean = float(np.mean(arr))
                    var = float(np.var(arr, ddof=1) if len(arr) > 1 else 0.0)
                    n = len(arr)
                rows.append({
                    "split_type": label,
                    "tvt": split,
                    "metric": metric,
                    "mean": mean,
                    "var": var,
                    "n_seeds": n,
                })

    df = pd.DataFrame(rows)

    # Wide format: one row per (split_type, tvt), columns for each metric mean/var
    wide_rows = []
    for (split_type, tvt), grp in df.groupby(["split_type", "tvt"], sort=False):
        row = {"split_type": split_type, "tvt": tvt}
        for _, r in grp.iterrows():
            row[f"{r['metric']}_mean"] = r["mean"]
            row[f"{r['metric']}_var"] = r["var"]
        row["n_seeds"] = grp["n_seeds"].iloc[0]
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)

    # Order columns nicely
    metric_cols = [f"{m}_{stat}" for m in METRICS for stat in ("mean", "var")]
    col_order = ["split_type", "tvt"] + metric_cols + ["n_seeds"]
    wide_df = wide_df[[c for c in col_order if c in wide_df.columns]]

    out_long = DATA_DIR / "aggregated_results_long.csv"
    out_wide = DATA_DIR / "aggregated_results_wide.csv"
    df.to_csv(out_long, index=False)
    wide_df.to_csv(out_wide, index=False)

    print(f"\nSaved long-form results to:  {out_long}")
    print(f"Saved wide-form results to:  {out_wide}")
    print("\n--- Wide-form preview ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(wide_df.to_string(index=False))


if __name__ == "__main__":
    main()
