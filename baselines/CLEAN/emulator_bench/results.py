from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def _rank_columns(df):
    columns = []
    for column in df.columns:
        if str(column).isdigit():
            columns.append(column)
    return sorted(columns, key=lambda value: int(str(value)))


def _empty_prf_metrics() -> dict:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def get_accuracy_level(predicted_ecs, true_ecs):
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


def average_accuracy(levels, level):
    if not levels:
        return 0.0
    return float(np.mean([1 if value >= level else 0 for value in levels]))


def ec_prefixes(label):
    prefixes = []
    parts = str(label).strip().split(".")
    for part in parts[:4]:
        normalized = part.strip()
        if not normalized or normalized == "-" or normalized.lower().startswith("n"):
            break
        prefixes.append(".".join(parts[: len(prefixes) + 1]))
    return prefixes


def _split_labels(ec_value):
    labels = []
    for raw in str(ec_value).replace(",", ";").split(";"):
        label = raw.strip()
        if label and label.lower() not in {"nan", "none", "null"}:
            labels.append(label)
    return labels


def _true_prefixes(ec_value):
    parsed = []
    for label in _split_labels(ec_value):
        prefixes = ec_prefixes(label)
        if prefixes:
            parsed.append(prefixes)
    return parsed


def compute_care_metrics(care_df: pd.DataFrame, k_values: tuple[int, ...] = (1, 20)) -> dict:
    """Compute CARE Task 1 metrics with CLEAN's notebook-compatible truncation policy."""
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
            # CARE's original notebook special-cases CLEAN/random by truncating the top-k
            # list to the number of true EC labels for that row.
            predicted = predicted[: len(true_ecs)]
            levels = get_accuracy_level(predicted, true_ecs)
            rows.append(levels)

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
        predictions = [str(row[col]) for col in rank_cols if pd.notna(row[col])]
        true_prefix_lists = _true_prefixes(row["EC number"])
        first_ranks = []
        for prefixes in true_prefix_lists:
            depth = len(prefixes)
            true_prefix = prefixes[-1]
            first_rank = None
            for rank, pred in enumerate(predictions, start=1):
                pred_prefixes = ec_prefixes(pred)
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
        # Backward-compatible aliases for existing aggregate scripts and notebooks.
        "mrr": row_metrics["mrr"],
        **{f"hit@{k}": row_metrics[f"hit@{k}"] for k in hit_ks},
    }


def compute_supplemental_classification_metrics(true_labels, predicted_labels) -> dict:
    labels = sorted(
        {
            str(label)
            for row in list(true_labels) + list(predicted_labels)
            for label in row
            if pd.notna(label)
        }
    )
    if not labels:
        return {average: _empty_prf_metrics() for average in ("micro", "macro", "weighted", "samples")}

    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(predicted_labels)
    metrics = {}
    for average in ("micro", "macro", "weighted", "samples"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        metrics[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    metrics["classes"] = int(len(labels))
    metrics["rows"] = int(len(y_true))
    return metrics


def parse_clean_maxsep_csv(maxsep_csv: str | Path) -> pd.DataFrame:
    rows = []
    with Path(maxsep_csv).open() as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            entry = raw_row[0]
            for rank, pred_ec_dist in enumerate(raw_row[1:]):
                if ":" not in pred_ec_dist or "/" not in pred_ec_dist:
                    raise ValueError(f"Malformed CLEAN maxsep prediction: {pred_ec_dist}")
                ec_number = pred_ec_dist.split(":", 1)[1].split("/", 1)[0]
                distance = pred_ec_dist.rsplit("/", 1)[1]
                rows.append(
                    {
                        "Entry": entry,
                        "rank": rank,
                        "predicted_ec": ec_number,
                        "distance": float(distance),
                    }
                )
    return pd.DataFrame(rows, columns=["Entry", "rank", "predicted_ec", "distance"])


def compact_evaluation_summary(all_metrics, *, checkpoint=None, model_name=None) -> dict:
    metrics_list = list(all_metrics)
    first = metrics_list[0] if metrics_list else {}
    summary = {
        "split_group": first.get("split_group"),
        "run_slug": first.get("run_slug"),
        "seed": first.get("seed"),
        "checkpoint": str(checkpoint or first.get("checkpoint")),
        "model_name": model_name or first.get("model_name"),
        "eval_splits": [metrics.get("eval_split") for metrics in metrics_list],
        "metrics_files": {},
        "care_ranked_csvs": {},
        "native_artifacts": {},
        "overview": {},
    }
    for metrics in metrics_list:
        split = metrics["eval_split"]
        seed_run_root = Path(metrics["seed_run_root"])
        artifacts = metrics.get("artifacts", {})
        summary["metrics_files"][split] = str(seed_run_root / "results" / f"{split}_metrics.json")
        summary["care_ranked_csvs"][split] = artifacts.get("care_ranked_csv")
        summary["native_artifacts"][split] = {
            key: value
            for key, value in artifacts.items()
            if key not in {"care_ranked_csv", "external_care_ranked_csv"}
        }
        summary["overview"][split] = {
            "native_clean.weighted_f1": metrics.get("native_clean", {}).get("weighted_f1"),
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


def write_care_ranked_csv(
    eval_dist_df: pd.DataFrame,
    clean_eval_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    clean_df = pd.read_csv(clean_eval_csv, sep="\t")
    rank_rows = []
    for entry in clean_df["Entry"]:
        if entry not in eval_dist_df.columns:
            raise KeyError(f"Missing evaluation distances for Entry={entry}")
        rank_rows.append(list(eval_dist_df[entry].sort_values(ascending=True).index))

    max_ranks = max((len(row) for row in rank_rows), default=0)
    rank_df = pd.DataFrame(rank_rows, columns=[str(i) for i in range(max_ranks)])
    care_df = pd.concat([clean_df.reset_index(drop=True), rank_df], axis=1)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    care_df.to_csv(output_csv, index=False)
    return care_df
