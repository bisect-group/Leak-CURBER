"""
Aggregate BIND baseline results across seeds.

Walks all bind_results directories under the BIND root, reads
final_results_{train,val,test}.csv for every seed, and outputs
mean + variance per metric per TVT split.

Output: aggregate_bind_results.csv saved to BIND_ROOT.
"""

import os
from pathlib import Path

import pandas as pd

BIND_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")) / "data" / "processed" / "baselines" / "BIND"
SPLITS = ("train", "val", "test")
METRICS = ("mae", "mse", "rmse", "r2_score", "pearson", "spearman", "loss")


def parse_path(seed_dir: Path) -> dict:
    """Extract dataset / split_type / threshold from a seed_* path."""
    # seed_dir: .../BIND/<dataset>/<split_type>/[threshold_X/]bind_results/seed_N
    parts = seed_dir.relative_to(BIND_ROOT).parts
    # parts[0] = dataset, parts[1] = split_type,
    # parts[2] = threshold_X or 'bind_results', parts[-2] = 'bind_results'
    dataset = parts[0]
    split_type = parts[1]
    threshold = parts[2] if parts[2] != "bind_results" else None
    seed = seed_dir.name  # e.g. seed_666
    return dict(dataset=dataset, split_type=split_type, threshold=threshold, seed=seed)


def load_seed_results(seed_dir: Path) -> dict[str, pd.Series] | None:
    """Return {split: metrics_series} for one seed dir, or None if incomplete."""
    results = {}
    for split in SPLITS:
        fpath = seed_dir / f"final_results_{split}.csv"
        if not fpath.exists():
            return None
        df = pd.read_csv(fpath)
        results[split] = df.iloc[0]
    return results


def main():
    rows = []

    for seed_dir in sorted(BIND_ROOT.rglob("bind_results/seed_*")):
        if not seed_dir.is_dir():
            continue

        meta = parse_path(seed_dir)
        split_results = load_seed_results(seed_dir)
        if split_results is None:
            print(f"  [skip] incomplete: {seed_dir.relative_to(BIND_ROOT)}")
            continue

        for split, series in split_results.items():
            row = {**meta, "tvt_split": split}
            for metric in METRICS:
                if metric in series.index:
                    row[metric] = series[metric]
            rows.append(row)

    if not rows:
        print("No complete results found.")
        return

    df = pd.DataFrame(rows)

    group_keys = ["dataset", "split_type", "threshold", "tvt_split"]
    agg = (
        df.groupby(group_keys, dropna=False)[list(METRICS)]
        .agg(["mean", "var"])
    )
    # Flatten MultiIndex columns: (metric, stat) -> metric_mean / metric_var
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_seeds"] = df.groupby(group_keys, dropna=False).size()
    agg = agg.reset_index()

    out_path = BIND_ROOT / "aggregate_bind_results.csv"
    agg.to_csv(out_path, index=False)
    print(f"Saved {len(agg)} rows to {out_path}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
