from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from tqdm import tqdm

from .dataset_adapter import _ec_sort_key, ec_prefixes, parse_label_set


def sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-logits))


def choose_f1_thresholds(probs, targets, masks):
    n_classes = targets.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    stats = {"classes": int(n_classes), "no_observed": 0, "no_positive": 0}

    for col in tqdm(range(n_classes), desc="validation thresholds", unit="class"):
        observed = masks[:, col] > 0
        if not np.any(observed):
            stats["no_observed"] += 1
            thresholds[col] = 1.0
            continue
        y_true = targets[observed, col].astype(np.int32)
        y_score = probs[observed, col]
        if y_true.sum() == 0:
            stats["no_positive"] += 1
            thresholds[col] = 1.0
            continue
        precision, recall, candidate_thresholds = precision_recall_curve(y_true, y_score)
        if len(candidate_thresholds) == 0:
            thresholds[col] = 0.5
            continue
        f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
        thresholds[col] = float(candidate_thresholds[int(np.argmax(f1))])
    return thresholds, stats


def _binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _empty_prf_metrics():
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _average_metrics(y_true, y_pred, average):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _masked_macro_weighted_metrics(targets, predictions, masks, weighted):
    per_class = []
    weights = []
    for col in range(targets.shape[1]):
        observed = masks[:, col] > 0
        if not np.any(observed):
            continue
        y_true = targets[observed, col].astype(int)
        y_pred = predictions[observed, col].astype(int)
        per_class.append(_binary_metrics(y_true, y_pred))
        weights.append(float(y_true.sum()))
    if not per_class:
        return _empty_prf_metrics()
    if weighted and np.sum(weights) > 0:
        weights_array = np.asarray(weights, dtype=np.float64)
        weights_array = weights_array / weights_array.sum()
        return {
            metric: float(
                np.sum([item[metric] * weight for item, weight in zip(per_class, weights_array)])
            )
            for metric in ("precision", "recall", "f1")
        }
    return {
        metric: float(np.mean([item[metric] for item in per_class]))
        for metric in ("precision", "recall", "f1")
    }


def compute_supplemental_classification_metrics(targets, predictions, masks=None):
    target_int = np.asarray(targets).astype(int)
    pred_int = np.asarray(predictions).astype(int)
    if target_int.size == 0:
        return {average: _empty_prf_metrics() for average in ("micro", "macro", "weighted", "samples")}

    if masks is None:
        return {
            average: _average_metrics(target_int, pred_int, average)
            for average in ("micro", "macro", "weighted", "samples")
        }

    mask_bool = np.asarray(masks) > 0
    if not np.any(mask_bool):
        return {average: _empty_prf_metrics() for average in ("micro", "macro", "weighted", "samples")}

    masked_target = target_int * mask_bool.astype(int)
    masked_pred = pred_int * mask_bool.astype(int)
    return {
        "micro": _binary_metrics(target_int[mask_bool], pred_int[mask_bool]),
        "macro": _masked_macro_weighted_metrics(target_int, pred_int, mask_bool, weighted=False),
        "weighted": _masked_macro_weighted_metrics(target_int, pred_int, mask_bool, weighted=True),
        "samples": _average_metrics(masked_target, masked_pred, "samples"),
        "observed_targets": int(mask_bool.sum()),
        "classes": int(target_int.shape[1]),
        "rows": int(target_int.shape[0]),
    }


def _macro_metrics(targets, predictions, masks, start, end):
    per_class = []
    for col in range(start, end):
        observed = masks[:, col] > 0
        if not np.any(observed):
            continue
        per_class.append(
            _binary_metrics(targets[observed, col].astype(int), predictions[observed, col].astype(int))
        )
    if not per_class:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "observed_classes": 0}
    return {
        "precision": float(np.mean([item["precision"] for item in per_class])),
        "recall": float(np.mean([item["recall"] for item in per_class])),
        "f1": float(np.mean([item["f1"] for item in per_class])),
        "observed_classes": int(len(per_class)),
    }


def compute_hitec_metrics(logits, targets, masks, thresholds, vocab):
    probs = sigmoid(logits)
    predictions = (probs >= thresholds.reshape(1, -1)).astype(np.int32)
    target_int = targets.astype(np.int32)
    mask_bool = masks > 0

    observed_target = target_int[mask_bool]
    observed_pred = predictions[mask_bool]
    micro = _binary_metrics(observed_target, observed_pred)

    exact_rows = []
    for row_idx in range(target_int.shape[0]):
        row_mask = mask_bool[row_idx]
        if not np.any(row_mask):
            continue
        exact_rows.append(bool(np.all(target_int[row_idx, row_mask] == predictions[row_idx, row_mask])))

    per_level = {}
    level_starts = vocab["level_starts"]
    output_dims = vocab["output_dims"]
    for level_idx, start in enumerate(level_starts):
        end = start + output_dims[level_idx]
        level_mask = mask_bool[:, start:end]
        if np.any(level_mask):
            level_micro = _binary_metrics(
                target_int[:, start:end][level_mask],
                predictions[:, start:end][level_mask],
            )
        else:
            level_micro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        per_level["level_{}".format(level_idx + 1)] = {
            "micro": level_micro,
            "macro": _macro_metrics(target_int, predictions, masks, start, end),
        }

    return {
        "micro": micro,
        "macro": _macro_metrics(target_int, predictions, masks, 0, target_int.shape[1]),
        "exact_match_accuracy": float(np.mean(exact_rows)) if exact_rows else 0.0,
        "rows": int(target_int.shape[0]),
        "observed_targets": int(mask_bool.sum()),
        "per_level": per_level,
    }


def rank_full_ecs(probs, vocab, rank_limit=50):
    l4_start = vocab["level_starts"][3]
    l4_labels = list(vocab["levels"][3])
    l4_probs = probs[:, l4_start : l4_start + len(l4_labels)]
    limit = len(l4_labels) if rank_limit is None or rank_limit <= 0 else min(rank_limit, len(l4_labels))
    ranked_rows = []
    for row in tqdm(l4_probs, desc="CARE ranks", unit="row"):
        top = np.argsort(-row)[:limit]
        ranked_rows.append([l4_labels[index] for index in top])
    return ranked_rows


def write_care_ranked_csv(manifest, probs, vocab, output_csv, rank_limit=50):
    metadata_cols = [
        col
        for col in [
            "Entry",
            "Original Entry",
            "EC number",
            "Sequence",
            "Original Sequence Length",
            "Sequence Length",
            "uniprot_date",
            "pdbs",
            "pdb_source",
            "pdb_type",
            "pdb_count",
        ]
        if col in manifest.columns
    ]
    ranked_rows = rank_full_ecs(probs, vocab, rank_limit=rank_limit)
    rank_df = pd.DataFrame(
        ranked_rows,
        columns=[str(index) for index in range(len(ranked_rows[0]) if ranked_rows else 0)],
    )
    care_df = pd.concat([manifest.loc[:, metadata_cols].reset_index(drop=True), rank_df], axis=1)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    care_df.to_csv(output_csv, index=False)
    return care_df


def _rank_columns(df):
    columns = []
    for column in df.columns:
        if str(column).isdigit():
            columns.append(column)
    return sorted(columns, key=lambda value: int(str(value)))


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


def _true_prefixes(ec_value):
    parsed = parse_label_set(ec_value)
    return [item["prefixes"] for item in parsed if item["prefixes"]]


def compute_care_metrics(care_df, k_values=(1, 20)):
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
            levels = get_accuracy_level(predicted, true_ecs)
            rows.append(levels)

        metrics["k={}".format(k)] = {
            "level_{}_accuracy".format(level): round(
                float(np.mean([average_accuracy(levels, level) for levels in rows])) * 100.0,
                4,
            )
            for level in (4, 3, 2, 1)
        }
        metrics["k={}".format(k)].update(
            {
                "level_{}_support".format(level): int(len(rows))
                for level in (4, 3, 2, 1)
            }
        )
    return metrics


def compute_supplemental_ranking_metrics(care_df, hit_ks=(1, 3, 5, 10, 20)):
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
            "hit@{}".format(k): round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in row_hits.items()
        },
    }
    label_metrics = {
        "mrr": round(float(np.mean(label_reciprocal_ranks)), 6)
        if label_reciprocal_ranks
        else 0.0,
        **{
            "hit@{}".format(k): round(float(np.mean(values)) * 100.0, 4) if values else 0.0
            for k, values in label_hits.items()
        },
    }

    return {
        "rank_columns": int(len(rank_cols)),
        "row": row_metrics,
        "label_weighted": label_metrics,
        "mrr": row_metrics["mrr"],
        **{"hit@{}".format(k): row_metrics["hit@{}".format(k)] for k in hit_ks},
    }


def sorted_label_string(labels):
    return ";".join(sorted(labels, key=_ec_sort_key))


def compact_evaluation_summary(all_metrics, *, checkpoint=None, thresholds=None) -> dict:
    metrics_list = list(all_metrics)
    first = metrics_list[0] if metrics_list else {}
    summary = {
        "split_group": first.get("split_group"),
        "run_slug": first.get("run_slug"),
        "seed": first.get("seed"),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "thresholds": thresholds,
        "eval_splits": [metrics.get("eval_split") for metrics in metrics_list],
        "metrics_files": {},
        "care_ranked_csvs": {},
        "prediction_artifacts": {},
        "overview": {},
    }
    for metrics in metrics_list:
        split = metrics["eval_split"]
        artifacts = metrics.get("artifacts", {})
        logits_path = artifacts.get("logits_npz")
        metrics_file = None
        if logits_path:
            metrics_file = str(Path(logits_path).with_name(f"{split}_metrics.json"))
        summary["metrics_files"][split] = metrics_file
        summary["care_ranked_csvs"][split] = artifacts.get("care_ranked_csv")
        summary["prediction_artifacts"][split] = {
            key: value
            for key, value in artifacts.items()
            if key not in {"care_ranked_csv", "external_care_ranked_csv"}
        }
        summary["overview"][split] = {
            "hitec.micro.f1": metrics.get("hitec", {}).get("micro", {}).get("f1"),
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
