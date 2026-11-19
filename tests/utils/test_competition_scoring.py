"""Tests for the competition-contract scorer, including agreement with ec_level_accuracy."""

from __future__ import annotations

import pytest

from src.utils.competition_scoring import (
    SubmissionError,
    ec_prefix,
    prepare_unique_inputs,
    read_submission,
    references_from_frame,
    score_competition,
    validate_submission,
    write_submission,
)
from src.utils.ec_level_accuracy import ec_level_accuracy

MULTI_LABEL_QUERIES = [
    (["1.1.1.1"], ["1.1.1.1", "2.2.2.2"]),
    (["1.1.1.1", "3.4.5.6"], ["1.1.2.9", "3.4.5.7", "3.4.5.6"]),
    (["2.7.1.1"], ["2.7.9.9", "2.1.1.1"]),
    (["4.1.1.1"], ["5.1.1.1", "6.1.1.1"]),
    (["4.1.1.1"], []),
    (["1.2.3.4", "1.2.3.5"], ["1.2.3", "1.2.3.5"]),
]


def test_published_multi_label_example():
    """Recorded 1.1.1.1 and 1.1.1.2; predicting 1.1.1.1 scores 0.5 / 1 / 1 / 1."""
    got = score_competition({"a": ["1.1.1.1"]}, {"a": ["1.1.1.1", "1.1.1.2"]})
    assert got["level_4_score"] == 0.5
    assert got["level_3_score"] == 1.0
    assert got["level_2_score"] == 1.0
    assert got["level_1_score"] == 1.0


def test_only_the_first_prediction_counts():
    references = {"a": ["2.7.7.48"]}
    assert score_competition({"a": ["1.1.1.1", "2.7.7.48"]}, references)["level_4_score"] == 0.0
    assert score_competition({"a": ["2.7.7.48", "1.1.1.1"]}, references)["level_4_score"] == 1.0


def test_empty_prediction_scores_zero_and_missing_prediction_raises():
    assert score_competition({"a": []}, {"a": ["1.1.1.1"]})["level_4_score"] == 0.0
    with pytest.raises(SubmissionError, match="no prediction"):
        score_competition({"a": ["1.1.1.1"]}, {"a": ["1.1.1.1"], "b": ["2.7.7.48"]})


def test_repeated_reference_labels_are_not_double_counted():
    got = score_competition({"a": ["1.1.1.1"]}, {"a": ["1.1.1.1", "1.1.1.1", "1.1.1.2"]})
    assert got["level_4_score"] == 0.5


def test_partial_reference_does_not_earn_full_level_4_credit():
    assert ec_prefix("1.1.1.", 4) == ("1", "1", "1")
    got = score_competition({"a": ["1.1.1.9"]}, {"a": ["1.1.1."]})
    assert got["level_4_score"] == 0.0
    assert got["level_3_score"] == 1.0


def test_agrees_with_ec_level_accuracy_at_k1():
    references = [query[0] for query in MULTI_LABEL_QUERIES]
    predictions = [query[1] for query in MULTI_LABEL_QUERIES]

    contract = score_competition(
        {index: ranked for index, ranked in enumerate(predictions)},
        {index: labels for index, labels in enumerate(references)},
    )
    ranked = ec_level_accuracy(references, predictions, ks=(1,))["k=1"]
    for level in (1, 2, 3, 4):
        assert contract[f"level_{level}_score"] == pytest.approx(
            ranked[f"level_{level}_accuracy"] / 100.0
        )


def test_validate_submission_rejects_the_documented_failures():
    vocabulary = {"1.1.1.1"}
    with pytest.raises(SubmissionError, match="missing ids"):
        validate_submission({}, ["a"], vocabulary)
    with pytest.raises(SubmissionError, match="extra ids"):
        validate_submission({"a": ["1.1.1.1"], "b": ["1.1.1.1"]}, ["a"], vocabulary)
    with pytest.raises(SubmissionError, match="not in vocabulary"):
        validate_submission({"a": ["9.9.9.9"]}, ["a"], vocabulary)
    with pytest.raises(SubmissionError, match="duplicate prediction"):
        validate_submission({"a": ["1.1.1.1", "1.1.1.1"]}, ["a"], vocabulary)
    with pytest.raises(SubmissionError, match="exceeds 10"):
        validate_submission({"a": ["1.1.1.1"] * 11}, ["a"], vocabulary)


def test_prepare_unique_inputs_keeps_first_id():
    import pandas as pd

    frame = pd.DataFrame(
        {"id": ["i0", "i1", "i2"], "sequence": ["AAA", "BBB", "AAA"]}
    )
    grouped = prepare_unique_inputs(frame)
    assert list(grouped["id"]) == ["i0", "i1"]


def test_references_from_frame_groups_labels():
    import pandas as pd

    frame = pd.DataFrame(
        {"id": ["a", "a", "b"], "ec_number": ["1.1.1.1", "1.1.1.2", "2.7.7.48"]}
    )
    assert references_from_frame(frame) == {
        "a": ["1.1.1.1", "1.1.1.2"],
        "b": ["2.7.7.48"],
    }


def test_jsonl_round_trip_and_contract(tmp_path):
    predictions = {"v_1": ["1.1.1.1"], "v_2": ["2.7.7.48"]}
    path = write_submission(predictions, tmp_path / "out.jsonl")
    assert read_submission(path) == predictions

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id":"a","prediction":["1.1.1.1"]}\n\n', encoding="utf-8")
    with pytest.raises(SubmissionError, match="blank line"):
        read_submission(bad)

    bad.write_text('{"id":"a","prediction":["1.1.1.1"],"extra":1}\n', encoding="utf-8")
    with pytest.raises(SubmissionError, match="keys must be exactly"):
        read_submission(bad)
