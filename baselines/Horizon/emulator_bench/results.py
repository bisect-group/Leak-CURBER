from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


CARE_TASK2_K_VALUES = (1, 3, 5, 10, 20, 30, 40, 50)
SUPPLEMENTAL_HIT_KS = (1, 3, 5, 10, 20, 50)


def rank_columns(df: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in df.columns if str(column).isdigit()]
    return sorted(columns, key=lambda value: int(value))


def split_ec_labels(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    labels = []
    for raw in str(value).replace(",", ";").split(";"):
        label = raw.strip()
        if label and label.lower() not in {"nan", "none", "null", "-", "-.-.-.-"}:
            labels.append(label)
    return sorted(set(labels))


def ec_prefixes(label: str) -> list[str]:
    parts = [part.strip() for part in str(label).split(".")]
    prefixes = []
    for idx, part in enumerate(parts[:4]):
        if not part or part == "-":
            break
        prefixes.append(".".join(parts[: idx + 1]))
    return prefixes


def get_accuracy_level(predicted_ecs: list[str], true_ecs: list[str]) -> list[int]:
    if not predicted_ecs:
        predicted_ecs = ["0.0.0.0"]
    levels = []
    for true_ec in true_ecs:
        true_parts = str(true_ec).split(".")
        counters = []
        for predicted_ec in predicted_ecs:
            predicted = str(predicted_ec)
            if predicted.count(".") != 3:
                predicted = "0.0.0.0"
            predicted_parts = predicted.split(".")
            counter = 0
            for predicted_part, true_part in zip(predicted_parts, true_parts):
                if predicted_part == true_part:
                    counter += 1
                else:
                    break
            counters.append(counter)
        levels.append(int(np.max(counters)) if counters else 0)
    return levels


def _average_accuracy(levels: list[int], level: int) -> float:
    if not levels:
        return 0.0
    return float(np.mean([1 if value >= level else 0 for value in levels]))


def compute_care_task2_metrics(
    care_df: pd.DataFrame,
    *,
    k_values: tuple[int, ...] = CARE_TASK2_K_VALUES,
) -> dict:
    ranks = rank_columns(care_df)
    if not ranks:
        raise ValueError("CARE Task 2 DataFrame does not contain rank columns")
    metrics = {}
    for k in k_values:
        rows = []
        for _, row in care_df.iterrows():
            true_ecs = split_ec_labels(row["EC number"])
            predicted = [
                str(row[column])
                for column in ranks[:k]
                if pd.notna(row[column]) and str(row[column]).strip()
            ]
            rows.append(get_accuracy_level(predicted, true_ecs))
        metrics[f"k={k}"] = {
            f"level_{level}_accuracy": round(
                float(np.mean([_average_accuracy(levels, level) for levels in rows])) * 100.0,
                1,
            )
            for level in (4, 3, 2, 1)
        }
        metrics[f"k={k}"].update(
            {f"level_{level}_support": int(len(rows)) for level in (4, 3, 2, 1)}
        )
    return metrics


def compute_supplemental_ec_metrics(
    care_df: pd.DataFrame,
    *,
    hit_ks: tuple[int, ...] = SUPPLEMENTAL_HIT_KS,
) -> dict:
    ranks = rank_columns(care_df)
    if not ranks:
        raise ValueError("CARE Task 2 DataFrame does not contain rank columns")

    row_rr = []
    label_rr = []
    average_precisions = []
    row_hits = {k: [] for k in hit_ks}
    label_hits = {k: [] for k in hit_ks}

    for _, row in care_df.iterrows():
        predictions = [
            str(row[column])
            for column in ranks
            if pd.notna(row[column]) and str(row[column]).strip()
        ]
        true_labels = set(split_ec_labels(row["EC number"]))
        first_ranks = []
        precision_hits = 0
        precision_sum = 0.0
        for rank, predicted in enumerate(predictions, start=1):
            if predicted in true_labels:
                precision_hits += 1
                precision_sum += precision_hits / rank
        average_precisions.append(
            precision_sum / len(true_labels) if true_labels else 0.0
        )

        for true_label in true_labels:
            first_rank = None
            for rank, predicted in enumerate(predictions, start=1):
                if predicted == true_label:
                    first_rank = rank
                    break
            first_ranks.append(first_rank)
            label_rr.append(0.0 if first_rank is None else 1.0 / first_rank)
            for k in hit_ks:
                label_hits[k].append(first_rank is not None and first_rank <= k)

        row_first_rank = min((rank for rank in first_ranks if rank is not None), default=None)
        row_rr.append(0.0 if row_first_rank is None else 1.0 / row_first_rank)
        for k in hit_ks:
            row_hits[k].append(row_first_rank is not None and row_first_rank <= k)

    row_metrics = {
        "mrr": round(float(np.mean(row_rr)), 6) if row_rr else 0.0,
        "map": round(float(np.mean(average_precisions)), 6) if average_precisions else 0.0,
        **{
            f"hit@{k}": round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in row_hits.items()
        },
    }
    label_metrics = {
        "mrr": round(float(np.mean(label_rr)), 6) if label_rr else 0.0,
        **{
            f"hit@{k}": round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in label_hits.items()
        },
    }
    return {
        "rank_columns": int(len(ranks)),
        "rows": int(len(care_df)),
        "row": row_metrics,
        "label_weighted": label_metrics,
        "mrr": row_metrics["mrr"],
        "map": row_metrics["map"],
        **{f"hit@{k}": row_metrics[f"hit@{k}"] for k in hit_ks},
    }


def load_care_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

