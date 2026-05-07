import argparse
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import DEFAULT_BASE_DIR, summarize_seed_runs

DISPLAY_METRICS = ["rmse", "mae", "r2_score", "pearson", "spearman", "ci"]


def normalize_split_labels(base_dir: Path, split_group: str, split_name: str) -> tuple[str, str]:
    grouped_sequence = Path(base_dir) / "random_splits_grouped_sequence"
    if split_group == "random_splits" and split_name == "random" and grouped_sequence.exists():
        return "random_splits_grouped_sequence", "sequence"
    return split_group, split_name


def scan_runs(output_root: Path, split: str, base_dir: Path) -> list:
    rows = []
    for result_file in sorted(output_root.rglob(f"final_results_{split}.csv")):
        # expected: output_root/{split_group}/{split_name}/seed_{seed}/final_results_{split}.csv
        try:
            seed_dir = result_file.parent
            split_name = seed_dir.parent.name
            split_group = seed_dir.parent.parent.name
            seed = int(seed_dir.name.replace("seed_", ""))
        except (ValueError, IndexError):
            continue
        split_group, split_name = normalize_split_labels(base_dir, split_group, split_name)
        try:
            metrics = pd.read_csv(result_file).iloc[0].to_dict()
        except Exception:
            continue
        rows.append({"split_group": split_group, "split_name": split_name, "seed": seed, **metrics})
    return rows


def scan_tvt_sizes(output_root: Path, base_dir: Path) -> pd.DataFrame:
    seen = set()
    rows = []
    for seed_dir in sorted(output_root.rglob("seed_*")):
        if not seed_dir.is_dir():
            continue
        split_name = seed_dir.parent.name
        split_group = seed_dir.parent.parent.name
        split_group, split_name = normalize_split_labels(base_dir, split_group, split_name)
        key = (split_group, split_name)
        if key in seen:
            continue
        sizes = {}
        for prefix in ("train", "val", "test"):
            pred_file = seed_dir / f"pred_label_{prefix}.csv"
            if pred_file.exists():
                try:
                    sizes[f"n_{prefix}"] = len(pd.read_csv(pred_file))
                except Exception:
                    sizes[f"n_{prefix}"] = None
            else:
                sizes[f"n_{prefix}"] = None
        if any(v is not None for v in sizes.values()):
            n_total = sum(v for v in sizes.values() if v is not None)
            rows.append({"split_group": split_group, "split_name": split_name, **sizes, "n_total": n_total})
            seen.add(key)
    return pd.DataFrame(rows).sort_values(["split_group", "split_name"]).reset_index(drop=True) if rows else pd.DataFrame()


def format_mean_std(mean: float, var: float, decimals: int = 4) -> str:
    std = math.sqrt(max(var, 0.0))
    fmt = f".{decimals}f"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def print_table(summary: pd.DataFrame, metric_cols: list, group_cols: list) -> None:
    mean_cols = [f"{m}_mean" for m in metric_cols if f"{m}_mean" in summary.columns]
    if not mean_cols:
        print("No metrics found.")
        return

    id_cols = list(group_cols) + ["n_seeds"]
    display = summary[id_cols].copy()
    for m in metric_cols:
        mean_col, var_col = f"{m}_mean", f"{m}_var"
        if mean_col in summary.columns:
            display[m] = summary.apply(
                lambda r: format_mean_std(r[mean_col], r.get(var_col, 0.0)), axis=1
            )

    col_widths = {col: max(len(col), display[col].astype(str).str.len().max()) for col in display.columns}
    header = "  ".join(col.ljust(col_widths[col]) for col in display.columns)
    print(header)
    print("-" * len(header))
    for _, row in display.iterrows():
        print("  ".join(str(row[col]).ljust(col_widths[col]) for col in display.columns))


def main():
    parser = argparse.ArgumentParser(description="Aggregate StructureFree-DTA retrain results across seeds.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Root directory of retrain outputs (default: base_dir/retrain_from_optuna)")
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--group_by", choices=["split_group", "split_name"], default="split_group",
                        help="Granularity of aggregation")
    parser.add_argument("--metrics", nargs="+", default=DISPLAY_METRICS,
                        help="Metrics to display (default: rmse mae r2_score pearson spearman ci)")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save aggregated metrics CSV")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else Path(args.base_dir) / "retrain_from_optuna"
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    base_dir = Path(args.base_dir)
    rows = scan_runs(output_root, args.split, base_dir)
    if not rows:
        raise FileNotFoundError(f"No final_results_{args.split}.csv files found under {output_root}")

    all_metric_cols = [col for col in pd.DataFrame(rows).columns
                       if col not in ("split_group", "split_name", "seed")]
    requested = [m for m in args.metrics if m in all_metric_cols]

    group_cols = ["split_group"] if args.group_by == "split_group" else ["split_group", "split_name"]
    summary = summarize_seed_runs(rows, group_cols, requested)

    print(f"\nResults ({args.split}) — mean ± std across seeds\n")
    print_table(summary, requested, group_cols)

    metrics_save = Path(args.save) if args.save else output_root / f"aggregate_{args.split}_{args.group_by}.csv"
    metrics_save.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_save, index=False)
    print(f"\nSaved metrics to {metrics_save}")

    tvt_df = scan_tvt_sizes(output_root, base_dir)
    if not tvt_df.empty:
        tvt_save = output_root / "split_tvt_sizes.csv"
        tvt_df.to_csv(tvt_save, index=False)
        print(f"Saved TVT sizes to {tvt_save}\n")
        col_widths = {col: max(len(col), tvt_df[col].astype(str).str.len().max()) for col in tvt_df.columns}
        header = "  ".join(col.ljust(col_widths[col]) for col in tvt_df.columns)
        print(header)
        print("-" * len(header))
        for _, row in tvt_df.iterrows():
            print("  ".join(str(row[col]).ljust(col_widths[col]) for col in tvt_df.columns))
    else:
        print("No pred_label files found for TVT sizes.")


if __name__ == "__main__":
    main()
