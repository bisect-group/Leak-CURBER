"""EC level accuracy for ranked EC predictions: top-k accuracy at each EC level."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EC_LEVELS = (1, 2, 3, 4)
MISS_EC = "0.0.0.0"
KS = (1, 3, 5, 10, 20, 30, 40, 50)
# Test-prediction column that identifies a query, per endpoint.
FULL_EC = re.compile(r"\d+\.\d+\.\d+\.\d+")  # same pattern as the CREEP wrapper
QUERY_COLUMN = {"classification": "uniprot_id", "retrieval": "rxn_smiles"}


def matched_levels(true_ec: str, pred_ec: str) -> int:
    """Length of the common EC prefix; a malformed or empty prediction counts as a miss."""
    if pred_ec.count(".") != 3:
        pred_ec = MISS_EC
    matched = 0
    for pred_part, true_part in zip(pred_ec.split("."), true_ec.split(".")):
        if pred_part != true_part:
            break
        matched += 1
    return matched


def ec_level_accuracy(
    true_ecs: list[list[str]],
    ranked_ecs: list[list[str]],
    ks: tuple[int, ...] = (1,),
) -> dict[str, dict[str, float]]:
    """Per query: for each true EC take the best prefix match among the top-k predictions, then
    the share of true ECs reaching level n. Report the mean over queries x 100, unrounded."""
    result: dict[str, dict[str, float]] = {}
    for k in ks:
        scores = np.zeros((len(true_ecs), len(EC_LEVELS)))
        for i, (trues, ranked) in enumerate(zip(true_ecs, ranked_ecs)):
            top = [str(ec) for ec in ranked[:k] if isinstance(ec, str) and ec] or [MISS_EC]
            best = np.array([max(matched_levels(t, p) for p in top) for t in trues])
            if best.size:
                scores[i] = [(best >= n).mean() for n in EC_LEVELS]
        means = scores.mean(axis=0) * 100.0 if len(scores) else np.zeros(len(EC_LEVELS))
        result[f"k={k}"] = {f"level_{n}_accuracy": float(m) for n, m in zip(EC_LEVELS, means)}
    return result


def ec_level_accuracy_from_predictions(path: str | Path, drop_b: bool = False) -> dict:
    """EC level accuracy from a classification/retrieval ``test_predictions.parquet``.

    Rows of one query share inputs, so each EC keeps its best score over the query's rows.
    With ``drop_b``, ECs that are not four numeric levels (preliminary B-ECs) leave the true ECs
    and the ranked lists, and a query left without a true EC is not counted.
    """
    frame = pd.read_parquet(path)
    n_rows = len(frame)
    frame = frame[frame["ec_number"].notna() & (frame["ec_number"] != "")]
    if drop_b:
        frame = frame[frame["ec_number"].map(FULL_EC.fullmatch).notna()]
    true_ecs, ranked_ecs = [], []
    for _, rows in frame.groupby("query", sort=False):
        best: dict[str, float] = {}
        for ecs, scores in zip(rows["top_ec"], rows["top_score"]):
            for ec, score in zip(ecs, scores):
                if drop_b and not FULL_EC.fullmatch(ec):
                    continue
                best[ec] = max(score, best.get(ec, score))
        ranked_ecs.append(sorted(best, key=best.get, reverse=True)[: max(KS)])
        true_ecs.append(sorted(set(rows["ec_number"])))
    return {
        **ec_level_accuracy(true_ecs, ranked_ecs, KS),
        "n_queries": len(true_ecs),
        "n_rows": n_rows,
        **({"b_ecs_dropped": True} if drop_b else {}),
    }


def write_ec_level_accuracy(predictions_path: str | Path, endpoint: str) -> None:
    """Write ``ec_level_accuracy.json`` next to the predictions. Retrieval also gets
    ``ec_level_accuracy_no_b.json``, scored without preliminary B-ECs, as the paper reports."""
    predictions_path = Path(predictions_path)
    payload = ec_level_accuracy_from_predictions(predictions_path)
    payload["query_column"] = QUERY_COLUMN[endpoint]
    (predictions_path.parent / "ec_level_accuracy.json").write_text(json.dumps(payload, indent=2) + "\n")
    if endpoint == "retrieval":
        no_b = ec_level_accuracy_from_predictions(predictions_path, drop_b=True)
        (predictions_path.parent / "ec_level_accuracy_no_b.json").write_text(json.dumps(no_b, indent=2) + "\n")
