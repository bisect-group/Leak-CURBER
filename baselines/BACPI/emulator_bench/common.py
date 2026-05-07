import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


DEFAULT_MODEL_NAME = "BACPI"
DEFAULT_BASE_ROOT = Path(os.environ.get("PROJECT_ROOT", ".")) / "data" / "processed" / "baselines" / DEFAULT_MODEL_NAME
DEFAULT_VALUE_TYPE = "ki"
DEFAULT_RESULTS_DIRNAME = "bacpi_results"
DEFAULT_SPLIT_GROUPS = [
    "random_splits",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "substrate_splits",
    "conformer_cosine_splits",
]
AFFINITY_VALUE_TYPES = ("ki", "kd", "ec50", "ic50")
RANDOM_SPLIT_GROUP_ALIAS = "random_splits"
RANDOM_SPLIT_GROUP_PREFIX = "random_splits_grouped_"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_base_dir(
    base_dir: Optional[str] = None,
    base_root: Optional[str] = None,
    value_type: str = DEFAULT_VALUE_TYPE,
) -> Path:
    if base_dir:
        return Path(base_dir).expanduser().resolve()
    root = Path(base_root or DEFAULT_BASE_ROOT).expanduser().resolve()
    normalized = str(value_type).strip().lower()
    if normalized not in AFFINITY_VALUE_TYPES:
        raise ValueError("Unsupported value_type `%s`. Expected one of %s." % (value_type, ", ".join(AFFINITY_VALUE_TYPES)))
    return root / normalized


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def load_json(path: Path) -> Dict:
    with open(path, "r") as handle:
        return json.load(handle)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported table format: %s" % path)


def write_table(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent(path)
    suffix = path.suffix.lower()
    tmp_path = Path(str(path) + ".tmp")
    if suffix == ".parquet":
        frame.to_parquet(tmp_path, index=False)
    elif suffix == ".csv":
        frame.to_csv(tmp_path, index=False)
    else:
        raise ValueError("Unsupported table format: %s" % path)
    tmp_path.replace(path)


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns %s in %s" % (missing, path))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_fast_torch_math() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(device_index)
    if major >= 8:
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}

    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        if math.isnan(pearson):
            pearson = 0.0
    try:
        from scipy import stats

        spearman = float(stats.spearmanr(y_true, y_pred).statistic)
        if math.isnan(spearman):
            spearman = 0.0
    except Exception:
        true_ranks = np.argsort(np.argsort(y_true))
        pred_ranks = np.argsort(np.argsort(y_pred))
        if np.std(true_ranks) == 0 or np.std(pred_ranks) == 0:
            spearman = 0.0
        else:
            spearman = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    return {
        "rmse": round(rmse, 6),
        "pearson": round(pearson, 6),
        "spearman": round(spearman, 6),
    }


def split_sizes(train_path: Path, val_path: Path, test_path: Path) -> Dict[str, int]:
    return {
        "train_size": int(len(read_table(train_path))),
        "val_size": int(len(read_table(val_path))),
        "test_size": int(len(read_table(test_path))),
    }


def _threshold_value(name: str) -> float:
    try:
        return float(name.split("threshold_")[-1])
    except Exception:
        return math.inf


def _difficulty_labels_for_thresholds(names: List[str]) -> Dict[str, str]:
    ordered = sorted(names, key=_threshold_value)
    if len(ordered) == 1:
        return {ordered[0]: "single"}
    if len(ordered) == 2:
        return {ordered[0]: "hard", ordered[1]: "easy"}
    if len(ordered) == 3:
        return {ordered[0]: "hard", ordered[1]: "medium", ordered[2]: "easy"}
    return {name: "rank_%s" % (idx + 1) for idx, name in enumerate(ordered)}


def normalize_threshold_args(
    thresholds: Optional[Iterable[str]] = None,
    threshold: Optional[str] = None,
) -> Optional[List[str]]:
    values: List[str] = []
    if thresholds is not None:
        values.extend([str(value) for value in thresholds if str(value).strip()])
    if threshold is not None and str(threshold).strip():
        values.append(str(threshold))
    if not values:
        return None
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = directory / ("%s%s" % (stem, suffix))
        if candidate.exists():
            return candidate
    return None


def is_random_split_group(split_group: str) -> bool:
    return split_group == RANDOM_SPLIT_GROUP_ALIAS or split_group.startswith(RANDOM_SPLIT_GROUP_PREFIX)


def expand_split_groups(base_dir: Path, split_groups: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    seen = set()
    grouped_random_dirs: Optional[List[str]] = None

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            expanded.append(name)

    for split_group in split_groups:
        split_group = str(split_group)
        if split_group != RANDOM_SPLIT_GROUP_ALIAS:
            add(split_group)
            continue

        if grouped_random_dirs is None:
            grouped_random_dirs = sorted(
                child.name
                for child in Path(base_dir).glob("%s*" % RANDOM_SPLIT_GROUP_PREFIX)
                if child.is_dir()
            )

        if grouped_random_dirs:
            for grouped_split_group in grouped_random_dirs:
                add(grouped_split_group)
        elif (Path(base_dir) / RANDOM_SPLIT_GROUP_ALIAS).exists():
            add(RANDOM_SPLIT_GROUP_ALIAS)

    return expanded


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    split_groups = expand_split_groups(Path(base_dir), split_groups or DEFAULT_SPLIT_GROUPS)
    threshold_filter = list(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []

    for split_group in split_groups:
        group_dir = Path(base_dir) / split_group
        if not group_dir.exists():
            continue

        train_path = _find_split_file(group_dir, "train")
        val_path = _find_split_file(group_dir, "val")
        test_path = _find_split_file(group_dir, "test")
        if train_path and val_path and test_path:
            if is_random_split_group(split_group):
                split_name = "random"
                difficulty = "random"
            else:
                split_name = split_group
                difficulty = split_group
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": split_name,
                    "difficulty": difficulty,
                    "root_dir": str(group_dir),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
            continue

        candidate_dirs = []
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir():
                continue
            if threshold_filter is not None and child.name not in threshold_filter:
                continue
            if child.name.startswith("threshold_") or child.name in {"easy", "medium", "hard"}:
                candidate_dirs.append(child)

        threshold_names = [child.name for child in candidate_dirs if child.name.startswith("threshold_")]
        threshold_difficulties = _difficulty_labels_for_thresholds(threshold_names)
        for child in candidate_dirs:
            train_path = _find_split_file(child, "train")
            val_path = _find_split_file(child, "val")
            test_path = _find_split_file(child, "test")
            if not (train_path and val_path and test_path):
                continue
            difficulty = threshold_difficulties.get(child.name, child.name)
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "difficulty": difficulty,
                    "root_dir": str(child),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
    return jobs


def resolve_single_split_job(base_dir: Path, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    threshold_filter = None if is_random_split_group(split_group) or split_group == "uniprot_time_splits" else normalize_threshold_args(threshold=threshold)
    jobs = discover_split_jobs(base_dir, split_groups=[split_group], thresholds=threshold_filter)
    if not jobs:
        detail = "%s/%s" % (split_group, threshold) if threshold else split_group
        raise FileNotFoundError("No split job discovered for %s in %s" % (detail, base_dir))
    if split_group == RANDOM_SPLIT_GROUP_ALIAS and len(jobs) > 1:
        available = ", ".join(job["split_group"] for job in jobs)
        raise ValueError(
            "Multiple grouped random split jobs found for %s. Specify one of: %s"
            % (split_group, available)
        )
    if is_random_split_group(split_group) or split_group == "uniprot_time_splits":
        return jobs[0]
    if threshold is None:
        available = ", ".join(job["split_name"] for job in jobs)
        raise ValueError("Multiple thresholded jobs found for %s. Specify --threshold. Available: %s" % (split_group, available))
    matching = [job for job in jobs if job["split_name"] == threshold]
    if not matching:
        available = ", ".join(job["split_name"] for job in jobs)
        raise FileNotFoundError("Threshold `%s` not found for %s. Available: %s" % (threshold, split_group, available))
    return matching[0]


def summarize_seed_runs(run_dirs: Iterable[Path], metric_filename: str = "final_results_test.csv") -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        metric_path = Path(run_dir) / metric_filename
        if not metric_path.exists():
            continue
        metrics = pd.read_csv(metric_path).iloc[0].to_dict()
        row = {"run_dir": str(run_dir)}
        row.update(metrics)
        parts = Path(run_dir).parts
        if len(parts) >= 3:
            row["seed"] = parts[-1]
            row["split_name"] = parts[-2]
            row["split_group"] = parts[-3]
        rows.append(row)
    return pd.DataFrame(rows)
