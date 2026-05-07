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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "StructureFree-DTA"
DEFAULT_BASE_DIR = Path(os.environ.get("PROJECT_ROOT", ".")) / "data" / "processed" / "baselines" / DEFAULT_MODEL_NAME
DEFAULT_EMBEDDINGS_DIR = DEFAULT_BASE_DIR / "embeddings"
DEFAULT_RESULTS_DIRNAME = "structurefree_results"
RANDOM_SPLIT_GROUP_PREFIX = "random_splits_grouped_"
LEGACY_RANDOM_SPLIT_GROUP = "random_splits"
DEFAULT_SPLIT_GROUPS = [
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "conformer_cosine_splits",
    "substrate_splits",
    "uniprot_time_splits",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(str(sequence).strip().upper().split()).replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")


def cache_key(value: str, model_name: str, max_length: int, pooling: str = "mean") -> str:
    return stable_hash(f"{str(value).strip()}|{model_name}|maxlen={int(max_length)}|pool={pooling}")


def embedding_cache_path(embeddings_dir: Path, kind: str, value: str, model_name: str, max_length: int, pooling: str = "mean") -> Path:
    normalized = normalize_sequence(value) if kind == "proteins" else str(value).strip()
    key = cache_key(normalized, model_name=model_name, max_length=max_length, pooling=pooling)
    return Path(embeddings_dir) / kind / stable_hash(model_name)[:10] / key[:2] / f"{key}.npz"


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}. Available columns: {list(df.columns)}")


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


def normalize_threshold_args(thresholds: Optional[Iterable[str]] = None, threshold: Optional[str] = None) -> Optional[List[str]]:
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


def is_random_split_group(split_group: str) -> bool:
    return split_group == LEGACY_RANDOM_SPLIT_GROUP or split_group.startswith(RANDOM_SPLIT_GROUP_PREFIX)


def _random_split_name(split_group: str) -> str:
    if split_group.startswith(RANDOM_SPLIT_GROUP_PREFIX):
        return split_group.removeprefix(RANDOM_SPLIT_GROUP_PREFIX)
    return "random"


def _has_direct_tvt_split(directory: Path) -> bool:
    return all(_find_split_file(directory, stem) for stem in ("train", "val", "test"))


def _default_split_groups(base_dir: Path) -> List[str]:
    base = Path(base_dir)
    random_groups = [
        child.name
        for child in sorted(base.glob(f"{RANDOM_SPLIT_GROUP_PREFIX}*"))
        if child.is_dir() and _has_direct_tvt_split(child)
    ]
    if not random_groups and _has_direct_tvt_split(base / LEGACY_RANDOM_SPLIT_GROUP):
        random_groups = [LEGACY_RANDOM_SPLIT_GROUP]
    return random_groups + list(DEFAULT_SPLIT_GROUPS)


def _expand_split_groups(base_dir: Path, split_groups: Optional[Iterable[str]]) -> List[str]:
    if split_groups is None:
        return _default_split_groups(base_dir)

    expanded: List[str] = []
    seen = set()
    for split_group in split_groups:
        split_group = str(split_group)
        matches = []
        if any(char in split_group for char in "*?["):
            matches = [child.name for child in sorted(Path(base_dir).glob(split_group)) if child.is_dir()]
        for value in matches or [split_group]:
            if value not in seen:
                seen.add(value)
                expanded.append(value)
    return expanded


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    split_groups = _expand_split_groups(base_dir, split_groups)
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
            split_name = _random_split_name(split_group) if is_random_split_group(split_group) else split_group.replace("_splits", "")
            difficulty = split_name
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
            jobs.append(
                {
                    "split_group": split_group,
                    "split_name": child.name,
                    "difficulty": threshold_difficulties.get(child.name, child.name),
                    "root_dir": str(child),
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                }
            )
    return jobs


def resolve_single_split_job(base_dir: Path, split_group: str, threshold: Optional[str] = None) -> Dict[str, str]:
    threshold_filter = None if is_random_split_group(split_group) or split_group in {"group_shuffle_splits", "uniprot_time_splits"} else normalize_threshold_args(threshold=threshold)
    jobs = discover_split_jobs(base_dir, split_groups=[split_group], thresholds=threshold_filter)
    if not jobs:
        detail = f"{split_group}/{threshold}" if threshold else split_group
        raise FileNotFoundError(f"No split job discovered for {detail} in {base_dir}")
    if threshold is None and len(jobs) > 1 and not is_random_split_group(split_group) and split_group not in {"group_shuffle_splits", "uniprot_time_splits"}:
        available = ", ".join(job["split_name"] for job in jobs)
        raise ValueError(f"Multiple jobs found for {split_group}. Specify --threshold. Available: {available}")
    return jobs[0] if threshold is None else next(job for job in jobs if job["split_name"] == threshold)


def split_sizes(train_path: Path, val_path: Path, test_path: Path) -> Dict[str, float]:
    train_size = len(read_table(train_path))
    val_size = len(read_table(val_path))
    test_size = len(read_table(test_path))
    total = train_size + val_size + test_size
    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_ratio": train_size / total if total else 0.0,
        "val_ratio": val_size / total if total else 0.0,
        "test_ratio": test_size / total if total else 0.0,
    }


def concordance_index(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return 0.5
    order = np.argsort(y_true, kind="mergesort")
    yt = y_true[order]
    yp = y_pred[order]
    ranks = {value: idx + 1 for idx, value in enumerate(sorted(set(yp)))}
    bit = np.zeros(len(ranks) + 2, dtype=np.int64)

    def add(index: int, value: int) -> None:
        while index < len(bit):
            bit[index] += value
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(bit[index])
            index -= index & -index
        return total

    permissible = 0
    concordant = 0
    previous = 0
    start = 0
    while start < len(yt):
        end = start + 1
        while end < len(yt) and yt[end] == yt[start]:
            end += 1
        for pred in yp[start:end]:
            rank = ranks[pred]
            concordant += prefix(rank - 1)
            permissible += previous
        for pred in yp[start:end]:
            add(ranks[pred], 1)
            previous += 1
        start = end
    return float(concordant / permissible) if permissible else 0.5


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {key: float("nan") for key in ["mae", "mse", "rmse", "r2_score", "pearson", "spearman", "ci"]}
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
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
        spearman = float(np.corrcoef(true_ranks, pred_ranks)[0, 1]) if np.std(true_ranks) and np.std(pred_ranks) else 0.0
    return {
        "mae": float(np.mean(np.abs(residual))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2_score": 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "pearson": pearson,
        "spearman": spearman,
        "ci": concordance_index(y_true, y_pred),
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
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values):
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_var"] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        out_rows.append(row)
    return pd.DataFrame(out_rows)
