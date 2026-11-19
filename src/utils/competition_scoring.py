"""Competition-contract EC scorer: first prediction only, fractional multi-label credit.

``ec_level_accuracy`` scores *ranked* predictions (top-k) and reports the share of
true ECs reaching each level.  The released enzyme-classification competition
scored differently, and a submission can pass one scorer while failing the other:

* one EC per unique input is scored - the first in the list; later entries are
  accepted by the output contract but never change the score;
* credit at level ``L`` is ``matches / n`` where ``n`` is the number of recorded
  ECs for that input, then averaged over unique inputs;
* identical input sequences are grouped and scored once, under the first id.

The two definitions agree at ``k=1`` on identical inputs, which
``tests/utils/test_competition_scoring.py`` asserts, so this module is a
contract checker rather than a competing metric.  It exists because a model can
rank the right EC second and still score zero on the competition, and because
the published catalogue includes multi-label inputs where a naive top-1 accuracy
over rows would be wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

EC_LEVELS = (1, 2, 3, 4)
MAX_PREDICTIONS = 10


class SubmissionError(ValueError):
    """Raised when a submission violates the competition output contract."""


def ec_prefix(ec: str, level: int) -> tuple[str, ...]:
    """First ``level`` components of an EC number, ignoring empty components."""
    parts = [part.strip() for part in str(ec).split(".")]
    while parts and parts[-1] == "":
        parts.pop()
    return tuple(parts[:level])


def is_scorable_prediction(ec: str) -> bool:
    """True only for a four-component numeric EC.

    ``ec_level_accuracy`` replaces any other prediction with ``MISS_EC`` before
    comparing, and the evaluator rejects such an entry outright, so a malformed
    prediction earns nothing at any level here too.
    """
    return len(str(ec).split(".")) == 4 and all(part.isdigit() for part in str(ec).split("."))


def prepare_unique_inputs(
    frame: pd.DataFrame, input_columns: Sequence[str] = ("sequence",)
) -> pd.DataFrame:
    """Group identical inputs and keep the first id, as the evaluator does."""
    columns = ["id", *input_columns]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SubmissionError("missing input columns: " + ", ".join(missing))
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise SubmissionError("input ids must be present and unique")
    return frame[columns].drop_duplicates(list(input_columns), keep="first").reset_index(drop=True)


def validate_submission(
    predictions: Mapping[str, Sequence[str]],
    expected_ids: Iterable[str],
    vocabulary: Sequence[str] | set[str],
) -> None:
    """Apply the evaluator's rejection rules and report every violation at once."""
    expected = list(expected_ids)
    if len(set(expected)) != len(expected):
        raise SubmissionError("expected_ids contains duplicates")
    expected_set = set(expected)
    allowed = set(vocabulary)
    problems: list[str] = []

    missing = expected_set - set(predictions)
    extra = set(predictions) - expected_set
    if missing:
        problems.append(f"{len(missing)} missing ids, e.g. {sorted(missing)[:5]}")
    if extra:
        problems.append(f"{len(extra)} extra ids, e.g. {sorted(extra)[:5]}")

    for key, ranked in predictions.items():
        if key not in expected_set:
            continue
        if not isinstance(ranked, (list, tuple)):
            problems.append(f"{key}: prediction is not a list")
            continue
        if len(ranked) > MAX_PREDICTIONS:
            problems.append(f"{key}: {len(ranked)} predictions exceeds {MAX_PREDICTIONS}")
        seen: set[str] = set()
        for ec in ranked:
            if not isinstance(ec, str) or not ec:
                problems.append(f"{key}: empty or non-string prediction")
                continue
            if ec in seen:
                problems.append(f"{key}: duplicate prediction {ec}")
            seen.add(ec)
            if ec not in allowed:
                problems.append(f"{key}: {ec} not in vocabulary")

    if problems:
        head = "; ".join(problems[:20])
        tail = f" (+{len(problems) - 20} more)" if len(problems) > 20 else ""
        raise SubmissionError(head + tail)


def score_competition(
    predictions: Mapping[str, Sequence[str]],
    references: Mapping[str, Sequence[str]],
    levels: Sequence[int] = EC_LEVELS,
) -> dict:
    """Score first-prediction ECs with fractional credit over recorded ECs.

    Returns one fraction per level, averaged over unique inputs, plus the number
    of inputs scored.  Inputs present in ``references`` but absent from
    ``predictions`` raise, because the evaluator rejects such a run outright.
    """
    missing = set(references) - set(predictions)
    if missing:
        raise SubmissionError(f"{len(missing)} reference ids have no prediction")

    totals = {level: 0.0 for level in levels}
    for key, labels in references.items():
        recorded = []
        for label in labels:
            text = str(label)
            if text not in recorded:
                recorded.append(text)
        if not recorded:
            raise SubmissionError(f"{key}: empty reference label list")

        ranked = predictions[key]
        first = str(ranked[0]) if ranked else None
        if first is not None and not is_scorable_prediction(first):
            first = None
        for level in levels:
            if first is None:
                continue
            prefix = ec_prefix(first, level)
            hits = sum(1 for label in recorded if ec_prefix(label, level) == prefix)
            totals[level] += hits / len(recorded)

    n_inputs = len(references)
    if n_inputs == 0:
        raise SubmissionError("no inputs to score")
    return {
        **{f"level_{level}_score": totals[level] / n_inputs for level in levels},
        "n_inputs": n_inputs,
    }


def references_from_frame(
    frame: pd.DataFrame, id_column: str = "id", label_column: str = "ec_number"
) -> dict[str, list[str]]:
    """Group a long-form (input, label) frame into per-input label lists."""
    references: dict[str, list[str]] = {}
    for key, label in zip(frame[id_column], frame[label_column]):
        references.setdefault(key, []).append(str(label))
    return references


def read_submission(path: str | Path) -> dict[str, list[str]]:
    """Read JSON Lines output, enforcing exactly the keys ``id`` and ``prediction``."""
    predictions: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SubmissionError(f"line {line_number}: blank line")
            record = json.loads(line)
            if set(record) != {"id", "prediction"}:
                raise SubmissionError(f"line {line_number}: keys must be exactly id, prediction")
            if record["id"] in predictions:
                raise SubmissionError(f"line {line_number}: duplicate id {record['id']}")
            if not isinstance(record["prediction"], list):
                raise SubmissionError(f"line {line_number}: prediction must be a list")
            predictions[record["id"]] = record["prediction"]
    return predictions


def write_submission(predictions: Mapping[str, Sequence[str]], path: str | Path) -> Path:
    """Write JSON Lines with exactly the keys ``id`` and ``prediction``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for key, ranked in predictions.items():
            handle.write(json.dumps({"id": key, "prediction": list(ranked)}) + "\n")
    return target
