import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import AFFINITY_VALUE_TYPES, DEFAULT_BASE_ROOT, discover_split_jobs, is_random_split_group, split_sizes


DEFAULT_RUNS_DIR = "default_hparams_original_retrain_runs"
DEFAULT_METRIC_FILE = "final_results_test.csv"
GROUP_COLUMNS = ["value_type", "split_group", "split_name", "difficulty"]
RUN_COLUMNS = GROUP_COLUMNS + ["seed", "train_size", "val_size", "test_size", "total_size", "run_dir"]
BACKFILL_METRICS = ("mae", "r2")


def normalize_value_types(value_types):
    normalized = []
    seen = set()
    for value_type in value_types:
        value_type = str(value_type).strip().lower()
        if not value_type:
            continue
        if value_type not in AFFINITY_VALUE_TYPES:
            raise ValueError(
                "Unsupported value_type `%s`. Expected one of %s."
                % (value_type, ", ".join(AFFINITY_VALUE_TYPES))
            )
        if value_type not in seen:
            seen.add(value_type)
            normalized.append(value_type)
    if not normalized:
        raise ValueError("At least one value type is required.")
    return normalized


def parse_seed(seed_name):
    if seed_name.startswith("seed_"):
        seed_value = seed_name[len("seed_") :]
    else:
        seed_value = seed_name
    try:
        return int(seed_value)
    except ValueError:
        return seed_value


def split_info_lookup(base_dir):
    lookup = {}
    for job in discover_split_jobs(base_dir):
        key = (job["split_group"], job["split_name"])
        info = {"difficulty": job["difficulty"]}
        try:
            info.update(split_sizes(Path(job["train_path"]), Path(job["val_path"]), Path(job["test_path"])))
        except Exception:
            pass
        lookup[key] = info
    return lookup


def fallback_difficulty(split_group, split_name):
    if is_random_split_group(split_group) and split_name == "random":
        return "random"
    if split_group == "uniprot_time_splits":
        return split_name
    if split_name.startswith("threshold_"):
        return "single"
    return split_name


def find_metric_paths(run_root, metric_file):
    return sorted(run_root.glob("*/*/seed_*/%s" % metric_file))


def infer_prediction_path(metric_path):
    filename = metric_path.name
    if filename.startswith("final_results_") and filename.endswith(".csv"):
        split_name = filename[len("final_results_") : -len(".csv")]
        return metric_path.with_name("pred_label_%s.csv" % split_name)
    return None


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def prediction_metrics(prediction_path):
    frame = pd.read_csv(prediction_path)
    missing_columns = [column for column in ("y_true", "y_pred") if column not in frame.columns]
    if missing_columns:
        raise ValueError("Missing required columns %s in %s" % (missing_columns, prediction_path))

    y_true = frame["y_true"].to_numpy(dtype=np.float64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {metric: float("nan") for metric in BACKFILL_METRICS}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    total_sum_squares = float(np.sum(np.square(y_true - np.mean(y_true))))
    if total_sum_squares == 0.0:
        r2 = 0.0 if float(np.sum(np.square(y_true - y_pred))) > 0.0 else 1.0
    else:
        residual_sum_squares = float(np.sum(np.square(y_true - y_pred)))
        r2 = 1.0 - (residual_sum_squares / total_sum_squares)
    return {"mae": round(mae, 6), "r2": round(r2, 6)}


def backfill_prediction_metrics(metrics, metric_path):
    prediction_path = infer_prediction_path(metric_path)
    if prediction_path is None or not prediction_path.exists():
        return metrics

    computed = prediction_metrics(prediction_path)
    for metric_name, metric_value in computed.items():
        if metric_name not in metrics or is_missing(metrics[metric_name]):
            metrics[metric_name] = metric_value
    return metrics


def read_metric_row(metric_path, value_type, lookup):
    relative_parts = metric_path.relative_to(metric_path.parents[3]).parts
    split_group, split_name, seed_name = relative_parts[:3]
    metrics = pd.read_csv(metric_path).iloc[0].to_dict()
    metrics = backfill_prediction_metrics(metrics, metric_path)
    info = lookup.get((split_group, split_name), {})
    row = {
        "value_type": value_type,
        "split_group": split_group,
        "split_name": split_name,
        "difficulty": info.get("difficulty", fallback_difficulty(split_group, split_name)),
        "seed": parse_seed(seed_name),
        "train_size": info.get("train_size"),
        "val_size": info.get("val_size"),
        "test_size": info.get("test_size"),
        "total_size": sum(info[k] for k in ("train_size", "val_size", "test_size") if info.get(k) is not None) or None,
        "run_dir": str(metric_path.parent),
    }
    row.update(metrics)
    return row


def aggregate_runs(runs_frame):
    if runs_frame.empty:
        return pd.DataFrame()
    numeric_cols = [
        column
        for column in runs_frame.columns
        if pd.api.types.is_numeric_dtype(runs_frame[column])
    ]
    summary = runs_frame.groupby(GROUP_COLUMNS, dropna=False)[numeric_cols].agg(["mean", "std"])
    summary.columns = ["%s_%s" % (column, stat) for column, stat in summary.columns]
    return summary.reset_index()


def write_outputs(runs_frame, run_root, write_empty):
    if runs_frame.empty and not write_empty:
        return None, None
    run_root.mkdir(parents=True, exist_ok=True)
    runs_path = run_root / "summary_runs.csv"
    summary_path = run_root / "summary.csv"
    runs_frame.to_csv(runs_path, index=False)
    aggregate_runs(runs_frame).to_csv(summary_path, index=False)
    return runs_path, summary_path


def aggregate_value_type(base_root, value_type, runs_dir, metric_file):
    base_dir = base_root / value_type
    run_root = base_dir / runs_dir
    if not run_root.exists():
        return run_root, pd.DataFrame()

    lookup = split_info_lookup(base_dir)
    rows = []
    for metric_path in find_metric_paths(run_root, metric_file):
        rows.append(read_metric_row(metric_path, value_type, lookup))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return run_root, frame

    metric_columns = [column for column in frame.columns if column not in RUN_COLUMNS]
    return run_root, frame[RUN_COLUMNS + metric_columns].sort_values(
        ["value_type", "split_group", "split_name", "seed"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate BACPI retrain seed metrics into summary_runs.csv and summary.csv."
    )
    parser.add_argument("--base_root", type=str, default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--value_types", nargs="+", default=list(AFFINITY_VALUE_TYPES))
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--metric_file", type=str, default=DEFAULT_METRIC_FILE)
    parser.add_argument("--combined_prefix", type=str, default=None)
    parser.add_argument("--no_per_value", action="store_true")
    parser.add_argument("--no_combined", action="store_true")
    parser.add_argument("--write_empty", action="store_true")
    args = parser.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    value_types = normalize_value_types(args.value_types)
    combined_prefix = args.combined_prefix or args.runs_dir

    all_frames = []
    for value_type in value_types:
        run_root, runs_frame = aggregate_value_type(
            base_root=base_root,
            value_type=value_type,
            runs_dir=args.runs_dir,
            metric_file=args.metric_file,
        )
        if not args.no_per_value:
            runs_path, summary_path = write_outputs(runs_frame, run_root, args.write_empty)
            if runs_path is not None:
                print(
                    "%s: wrote %s runs to %s and %s"
                    % (value_type, len(runs_frame), runs_path, summary_path),
                    flush=True,
                )
            else:
                print("%s: no completed metric files under %s" % (value_type, run_root), flush=True)
        elif runs_frame.empty:
            print("%s: no completed metric files under %s" % (value_type, run_root), flush=True)

        if not runs_frame.empty:
            all_frames.append(runs_frame)

    if not args.no_combined:
        combined_frame = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
        if not combined_frame.empty or args.write_empty:
            runs_path = base_root / ("%s_summary_runs.csv" % combined_prefix)
            summary_path = base_root / ("%s_summary.csv" % combined_prefix)
            combined_frame.to_csv(runs_path, index=False)
            aggregate_runs(combined_frame).to_csv(summary_path, index=False)
            print(
                "combined: wrote %s runs to %s and %s"
                % (len(combined_frame), runs_path, summary_path),
                flush=True,
            )
        else:
            print("combined: no completed metric files found", flush=True)


if __name__ == "__main__":
    main()
