import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch


DEFAULT_MODEL_NAME = "BIND"
DEFAULT_BASE_DIR = Path(os.environ.get("PROJECT_ROOT", ".")) / "data" / "processed" / "baselines" / DEFAULT_MODEL_NAME
DEFAULT_EMBEDDINGS_DIR = DEFAULT_BASE_DIR / "embeddings"
DEFAULT_RESULTS_DIRNAME = "bind_results"
DEFAULT_VALUE_TYPES = ["ki", "kd", "ic50", "ec50"]
HEAD_ORDER = ["ki", "ic50", "kd", "ec50"]
DEFAULT_HIDDEN_STATE_INDICES = [0, 10, 20, 30]
SPECIAL_CHILDREN = {"embeddings", "optuna_studies", "retrain_from_optuna", "bench_summaries"}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split()).replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")


def truncate_sequence(sequence: str, max_length: Optional[int] = None) -> str:
    normalized = normalize_sequence(sequence)
    if max_length is None:
        return normalized
    limit = int(max_length)
    if limit <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    return normalized[:limit]


def canonical_value_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("p", "", 1) if str(value).strip().lower().startswith("p") else str(value).strip().lower()
    mapping = {
        "ki": "ki",
        "kd": "kd",
        "ic50": "ic50",
        "ec50": "ec50",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported value_type: {value}")
    return mapping[normalized]


def model_cache_tag(model_name: str, layer_indices: Sequence[int]) -> str:
    safe_model = str(model_name).replace("/", "_").replace("-", "_")
    layer_tag = "-".join(str(int(index)) for index in layer_indices)
    return f"{safe_model}_layers_{layer_tag}"


def protein_cache_path(embeddings_dir: Path, sequence: str, model_name: str, layer_indices: Sequence[int]) -> Path:
    key = stable_hash(normalize_sequence(sequence))
    return Path(embeddings_dir) / "proteins" / model_cache_tag(model_name, layer_indices) / key[:2] / f"{key}.npz"


def ligand_cache_path(embeddings_dir: Path, smiles: str) -> Path:
    key = stable_hash(str(smiles).strip())
    return Path(embeddings_dir) / "ligands" / "graph_v1" / key[:2] / f"{key}.npz"


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


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_parent(path)
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


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
    return {name: f"rank_{idx}" for idx, name in enumerate(ordered, start=1)}


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
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def discover_value_types(base_dir: Path, value_types: Optional[Iterable[str]] = None) -> List[str]:
    base_dir = Path(base_dir)
    if value_types:
        return [canonical_value_type(value) for value in value_types]

    discovered = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir() or child.name in SPECIAL_CHILDREN:
            continue
        try:
            discovered.append(canonical_value_type(child.name))
        except ValueError:
            continue
    if not discovered:
        raise FileNotFoundError(f"No value_type directories found under {base_dir}")
    return discovered


def discover_split_groups(value_root: Path, split_groups: Optional[Iterable[str]] = None) -> List[str]:
    value_root = Path(value_root)
    if not value_root.exists():
        return []
    if split_groups:
        return [str(group) for group in split_groups]
    groups = []
    for child in sorted(value_root.iterdir()):
        if child.is_dir() and child.name not in SPECIAL_CHILDREN:
            groups.append(child.name)
    return groups


def discover_split_jobs(
    base_dir: Path,
    value_types: Optional[Iterable[str]] = None,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    threshold_filter = list(thresholds) if thresholds is not None else None
    jobs: List[Dict[str, str]] = []

    for value_type in discover_value_types(base_dir, value_types=value_types):
        value_root = Path(base_dir) / value_type
        if not value_root.exists():
            continue
        for split_group in discover_split_groups(value_root, split_groups=split_groups):
            group_dir = value_root / split_group
            if not group_dir.exists():
                continue

            train_path = _find_split_file(group_dir, "train")
            val_path = _find_split_file(group_dir, "val")
            test_path = _find_split_file(group_dir, "test")
            if train_path and val_path and test_path:
                jobs.append(
                    {
                        "value_type": value_type,
                        "value_root": str(value_root),
                        "split_group": split_group,
                        "split_name": split_group,
                        "difficulty": split_group,
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
                        "value_type": value_type,
                        "value_root": str(value_root),
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


def resolve_single_split_job(base_dir: Path, value_type: str, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    canonical = canonical_value_type(value_type)
    threshold_filter = None if threshold is None else normalize_threshold_args(threshold=threshold)
    jobs = discover_split_jobs(base_dir, value_types=[canonical], split_groups=[split_group], thresholds=threshold_filter)
    if not jobs:
        detail = f"{canonical}/{split_group}/{threshold}" if threshold else f"{canonical}/{split_group}"
        raise FileNotFoundError(f"No split job discovered for {detail} in {base_dir}")
    if threshold is None and len(jobs) == 1:
        return jobs[0]
    if threshold is None:
        available = ", ".join(job["split_name"] for job in jobs)
        raise ValueError(f"Multiple thresholded jobs found for {canonical}/{split_group}. Specify --threshold. Available: {available}")
    matching = [job for job in jobs if job["split_name"] == threshold]
    if not matching:
        available = ", ".join(job["split_name"] for job in jobs)
        raise FileNotFoundError(f"Threshold `{threshold}` not found for {canonical}/{split_group}. Available: {available}")
    return matching[0]


def split_sizes(train_path: Path, val_path: Path, test_path: Path) -> Dict[str, float]:
    train_size = len(read_table(train_path))
    val_size = len(read_table(val_path))
    test_size = len(read_table(test_path))
    total = train_size + val_size + test_size
    if total == 0:
        return {
            "train_size": 0,
            "val_size": 0,
            "test_size": 0,
            "train_ratio": 0.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
        }
    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_size / total,
        "val_ratio": val_size / total,
        "test_ratio": test_size / total,
    }


def summarize_seed_runs(rows: List[Dict], group_cols: Iterable[str], metric_cols: Iterable[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    runs_df = pd.DataFrame(rows)
    out_rows = []
    for keys, group in runs_df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for col in runs_df.columns:
            if col in row or col in metric_cols or col == "seed":
                continue
            if col.endswith("_dir"):
                continue
            values = group[col].dropna()
            if len(values) > 0:
                row[col] = values.iloc[0]
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            metric_values = group[metric].dropna()
            if len(metric_values) == 0:
                continue
            row[f"{metric}_mean"] = float(metric_values.mean())
            row[f"{metric}_var"] = float(metric_values.var(ddof=1)) if len(metric_values) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "r2_score": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

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
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score,
        "pearson": pearson,
        "spearman": spearman,
    }
