"""Feature-free R2 references on the P-seq (enzyme sequence) test split of each regression subtask.

Training-mean: predict the mean training log10 target.
Expected random: predictions drawn from the empirical training-target distribution; the exact
expected R2 is 1 - (E[(y - mu)^2] + Var_train) / Var_test, so no seeds are involved.
"""

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data/processed/datasets"
# subtask -> (dataset folder, P-seq threshold)
SUBTASKS = {
    "k_cat": ("kinetic_dataset/kcat", 0.01),
    "K_M": ("kinetic_dataset/km", 0.05),
    "K_i": ("kinetic_dataset/ki", 0.05),
    "K_d": ("binding_affinity_dataset/kd", 0.01),
    "EC50": ("binding_affinity_dataset/ec50", 0.01),
    "IC50": ("binding_affinity_dataset/ic50", 0.01),
}


def log10_target(path: Path) -> pl.Series:
    return pl.read_parquet(path, columns=["log10_value"])["log10_value"].drop_nulls()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/feature_free_reference.csv")
    out = parser.parse_args().out
    rows = []
    for name, (folder, threshold) in SUBTASKS.items():
        split = DS / folder / "enzyme_sequence_splits" / f"threshold_{threshold}"
        train, test = log10_target(split / "train.parquet"), log10_target(split / "test.parquet")
        mse_mean = ((test - train.mean()) ** 2).mean()
        var_test = test.var(ddof=0)
        rows.append(
            {
                "subtask": name,
                "threshold": threshold,
                "n_train": len(train),
                "n_test": len(test),
                "train_mean": train.mean(),
                "test_mean": test.mean(),
                "train_sd": train.std(ddof=0),
                "test_sd": test.std(ddof=0),
                "training_mean_r2": 1 - mse_mean / var_test,
                "training_mean_mae": (test - train.mean()).abs().mean(),
                "expected_random_r2": 1 - (mse_mean + train.var(ddof=0)) / var_test,
            }
        )
    df = pl.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out, float_precision=4)
    with pl.Config(tbl_cols=-1, tbl_rows=-1):
        print(df)


if __name__ == "__main__":
    main()
