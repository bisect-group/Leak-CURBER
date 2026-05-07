from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_rank_start(df: pd.DataFrame) -> int:
    for index, column in enumerate(df.columns):
        if str(column) == "0":
            return index
    raise ValueError("CARE results DataFrame does not contain rank column '0'")


def _rank_columns(df: pd.DataFrame) -> list:
    columns = []
    for column in df.columns:
        if str(column).isdigit():
            columns.append(column)
    return sorted(columns, key=lambda value: int(str(value)))


def ec_prefixes(label: str) -> list[str]:
    prefixes = []
    parts = str(label).strip().split(".")
    for part in parts[:4]:
        normalized = part.strip()
        if not normalized or normalized == "-" or normalized.lower().startswith("n"):
            break
        prefixes.append(".".join(parts[: len(prefixes) + 1]))
    return prefixes


def _true_prefixes(ec_value: object) -> list[list[str]]:
    parsed = []
    for raw in str(ec_value).replace(",", ";").split(";"):
        label = raw.strip()
        if not label or label.lower() in {"nan", "none", "null"}:
            continue
        prefixes = ec_prefixes(label)
        if prefixes:
            parsed.append(prefixes)
    return parsed


def get_accuracy_level(predicted_ecs: Iterable[str], true_ecs: Iterable[str]) -> list[int]:
    predicted = [str(ec) for ec in predicted_ecs if pd.notna(ec)]
    true = [str(ec) for ec in true_ecs if pd.notna(ec)]
    if not predicted:
        predicted = ["0.0.0.0"]
    levels = []
    for true_ec in true:
        true_split = true_ec.split(".")
        counters = []
        for predicted_ec in predicted:
            if predicted_ec.count(".") != 3:
                predicted_ec = "0.0.0.0"
            predicted_split = predicted_ec.split(".")
            counter = 0
            for predicted_part, true_part in zip(predicted_split, true_split):
                if predicted_part == true_part:
                    counter += 1
                else:
                    break
            counters.append(counter)
        levels.append(int(np.max(counters)) if counters else 0)
    return levels


def average_accuracy(levels: list[int], level: int) -> float:
    if not levels:
        return 0.0
    return float(np.mean([1 if value >= level else 0 for value in levels]))


def compute_care_metrics(care_df: pd.DataFrame, k_values: tuple[int, ...] = (1, 20)) -> dict:
    rank_cols = _rank_columns(care_df)
    if not rank_cols:
        raise ValueError("CARE results DataFrame does not contain rank columns")
    ranked = care_df.copy()
    ranked.loc[:, rank_cols] = ranked.loc[:, rank_cols].fillna("0.0.0.0")
    metrics = {}
    for k in k_values:
        rows = []
        for _, row in ranked.iterrows():
            true_ecs = str(row["EC number"]).split(";")
            predicted = list(row[rank_cols[:k]])
            rows.append(get_accuracy_level(predicted, true_ecs))
        metrics[f"k={k}"] = {
            f"level_{level}_accuracy": round(
                float(np.mean([average_accuracy(levels, level) for levels in rows])) * 100.0,
                4,
            )
            for level in (4, 3, 2, 1)
        }
        metrics[f"k={k}"].update(
            {
                f"level_{level}_support": int(len(rows))
                for level in (4, 3, 2, 1)
            }
        )
    return metrics


def compute_supplemental_ranking_metrics(
    care_df: pd.DataFrame,
    *,
    hit_ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict:
    rank_cols = _rank_columns(care_df)
    if not rank_cols:
        raise ValueError("CARE results DataFrame does not contain rank columns")

    row_reciprocal_ranks = []
    label_reciprocal_ranks = []
    row_hits = {k: [] for k in hit_ks}
    label_hits = {k: [] for k in hit_ks}
    for _, row in care_df.iterrows():
        ranked_ecs = [str(row[col]) for col in rank_cols if pd.notna(row[col])]
        first_ranks = []
        for prefixes in _true_prefixes(row["EC number"]):
            depth = len(prefixes)
            true_prefix = prefixes[-1]
            first_rank = None
            for rank, ec_number in enumerate(ranked_ecs, start=1):
                pred_prefixes = ec_prefixes(ec_number)
                if len(pred_prefixes) >= depth and pred_prefixes[depth - 1] == true_prefix:
                    first_rank = rank
                    break
            first_ranks.append(first_rank)
            label_reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
            for k in hit_ks:
                label_hits[k].append(first_rank is not None and first_rank <= k)

        row_first_rank = min((rank for rank in first_ranks if rank is not None), default=None)
        row_reciprocal_ranks.append(0.0 if row_first_rank is None else 1.0 / row_first_rank)
        for k in hit_ks:
            row_hits[k].append(row_first_rank is not None and row_first_rank <= k)

    row_metrics = {
        "mrr": round(float(np.mean(row_reciprocal_ranks)), 6) if row_reciprocal_ranks else 0.0,
        **{
            f"hit@{k}": round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in row_hits.items()
        },
    }
    label_metrics = {
        "mrr": round(float(np.mean(label_reciprocal_ranks)), 6)
        if label_reciprocal_ranks
        else 0.0,
        **{
            f"hit@{k}": round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in label_hits.items()
        },
    }
    return {
        "rank_columns": int(len(rank_cols)),
        "row": row_metrics,
        "label_weighted": label_metrics,
        "mrr": row_metrics["mrr"],
        **{f"hit@{k}": row_metrics[f"hit@{k}"] for k in hit_ks},
    }


def compute_supplemental_classification_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    if y_true.size == 0:
        return {
            average: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for average in ("micro", "macro", "weighted", "samples")
        }
    metrics = {}
    for average in ("micro", "macro", "weighted", "samples"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true.astype(int),
            predictions.astype(int),
            average=average,
            zero_division=0,
        )
        metrics[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    metrics["classes"] = int(y_true.shape[1])
    metrics["rows"] = int(y_true.shape[0])
    return metrics


def multilabel_targets(labels: list[str], label_to_idx: dict[str, int]) -> np.ndarray:
    target = np.zeros(len(label_to_idx), dtype=np.float32)
    for label in labels:
        if label in label_to_idx:
            target[label_to_idx[label]] = 1.0
    return target


def compute_graphec_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    if y_true.size == 0:
        return {}
    pred = (scores >= threshold).astype(int)
    flat_true = y_true.reshape(-1)
    flat_pred = pred.reshape(-1)
    flat_scores = scores.reshape(-1)
    metrics = {
        "micro_precision": float(precision_score(flat_true, flat_pred, zero_division=0)),
        "micro_recall": float(recall_score(flat_true, flat_pred, zero_division=0)),
        "micro_f1": float(f1_score(flat_true, flat_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(flat_true, flat_pred)) if len(np.unique(flat_true)) > 1 else 0.0,
        "exact_match_accuracy": float(np.mean(np.all(pred == y_true, axis=1))),
    }
    if len(np.unique(flat_true)) > 1:
        metrics["micro_auc"] = float(roc_auc_score(flat_true, flat_scores))
        metrics["micro_aupr"] = float(average_precision_score(flat_true, flat_scores))
    else:
        metrics["micro_auc"] = None
        metrics["micro_aupr"] = None
    macro_aucs = []
    macro_auprs = []
    for column in range(y_true.shape[1]):
        if len(np.unique(y_true[:, column])) > 1:
            macro_aucs.append(roc_auc_score(y_true[:, column], scores[:, column]))
            macro_auprs.append(average_precision_score(y_true[:, column], scores[:, column]))
    metrics["macro_auc"] = float(np.mean(macro_aucs)) if macro_aucs else None
    metrics["macro_aupr"] = float(np.mean(macro_auprs)) if macro_auprs else None
    return metrics


def write_ranked_outputs(
    manifest_csv: str | Path,
    scores: np.ndarray,
    labels: list[str],
    output_csv: str | Path,
    *,
    top_k: int = 50,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    rank_rows = []
    max_k = min(top_k, len(labels))
    for row_scores in scores:
        ranked_indices = np.argsort(row_scores)[::-1][:max_k]
        rank_rows.append([labels[index] for index in ranked_indices])
    rank_df = pd.DataFrame(rank_rows, columns=[str(index) for index in range(max_k)])
    care_df = pd.concat(
        [manifest.loc[:, ["Entry", "EC number", "Sequence"]].reset_index(drop=True), rank_df],
        axis=1,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    care_df.to_csv(output_csv, index=False)
    return care_df


def compact_evaluation_summary(all_metrics, *, train_metadata=None) -> dict:
    metrics_list = list(all_metrics)
    first = metrics_list[0] if metrics_list else {}
    fold_results = (train_metadata or {}).get("fold_results", [])
    summary = {
        "split_group": first.get("split_group"),
        "seed": first.get("seed"),
        "fold_checkpoints": [fold.get("checkpoint") for fold in fold_results],
        "eval_splits": [metrics.get("eval_split") for metrics in metrics_list],
        "metrics_files": {},
        "care_ranked_csvs": {},
        "prediction_artifacts": {},
        "overview": {},
    }
    for metrics in metrics_list:
        split = metrics["eval_split"]
        artifacts = metrics.get("artifacts", {})
        scores_path = artifacts.get("scores_npz")
        metrics_file = None
        if scores_path:
            metrics_file = str(Path(scores_path).with_name(f"{split}_metrics.json"))
        summary["metrics_files"][split] = metrics_file
        summary["care_ranked_csvs"][split] = artifacts.get("care_ranked_csv")
        summary["prediction_artifacts"][split] = {
            key: value for key, value in artifacts.items() if key != "care_ranked_csv"
        }
        summary["overview"][split] = {
            "graphec.micro_f1": metrics.get("graphec", {}).get("micro_f1"),
            "care_task1.k=1.level_4_accuracy": metrics.get("care_task1", {})
            .get("k=1", {})
            .get("level_4_accuracy"),
            "care_task1.k=20.level_4_accuracy": metrics.get("care_task1", {})
            .get("k=20", {})
            .get("level_4_accuracy"),
            "supplemental.ranking.row.mrr": metrics.get("supplemental", {})
            .get("ranking", {})
            .get("row", {})
            .get("mrr"),
        }
    return summary
