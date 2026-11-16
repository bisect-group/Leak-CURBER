from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.ec_level_accuracy import (
    QUERY_COLUMN,
    ec_level_accuracy,
    ec_level_accuracy_from_predictions,
    matched_levels,
)

ROOT = Path(__file__).resolve().parents[2]


def _load(path: str, name: str):
    if not (ROOT / path).is_file():
        pytest.skip(f"{path} is missing (submodule not checked out)")
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        pytest.skip(f"cannot import {path}")
    return module


QUERIES = [
    (["1.1.1.1"], ["1.1.1.1", "2.2.2.2"]),
    (["1.1.1.1", "3.4.5.6"], ["1.1.2.9", "3.4.5.7", "3.4.5.6"]),
    (["2.7.1.1"], ["2.7.9.9", "2.1.1.1"]),
    (["4.1.1.1"], ["5.1.1.1", "6.1.1.1"]),
    (["4.1.1.1"], []),
    (["1.2.3.4", "1.2.3.5"], ["1.2.3", "1.2.3.5"]),
]


def _horizyn_get_accuracy_level(predicted_ecs, true_ecs):
    # Oracle copied from baselines/Horizon/emulator_bench/results.py (importing that module is heavy).
    if not predicted_ecs:
        predicted_ecs = ["0.0.0.0"]
    levels = []
    for true_ec in true_ecs:
        true_parts = true_ec.split(".")
        counters = []
        for predicted in predicted_ecs:
            if predicted.count(".") != 3:
                predicted = "0.0.0.0"
            counter = 0
            for predicted_part, true_part in zip(predicted.split("."), true_parts):
                if predicted_part != true_part:
                    break
                counter += 1
            counters.append(counter)
        levels.append(max(counters))
    return levels


def _reference_scores(module_get_levels, ks):
    out = {}
    for k in ks:
        per_level = {n: [] for n in (1, 2, 3, 4)}
        for trues, ranked in QUERIES:
            levels = module_get_levels([e for e in ranked[:k] if e], trues)
            for n in per_level:
                per_level[n].append(np.mean([v >= n for v in levels]) if levels else 0.0)
        out[f"k={k}"] = {
            f"level_{n}_accuracy": float(np.mean(v) * 100) for n, v in per_level.items()
        }
    return out


@pytest.mark.parametrize("source", ["graphec", "horizyn"])
def test_matches_reference_get_accuracy_level(source):
    if source == "graphec":
        oracle = _load(
            "baselines/GraphEC/emulator_bench/results.py", "graphec_results"
        ).get_accuracy_level
    else:
        oracle = _horizyn_get_accuracy_level
    ks = (1, 2, 3)
    expected = _reference_scores(oracle, ks)
    actual = ec_level_accuracy([q[0] for q in QUERIES], [q[1] for q in QUERIES], ks)
    for k in expected:
        assert actual[k] == pytest.approx(expected[k])


def test_known_values():
    got = ec_level_accuracy(
        [["1.1.1.1"], ["1.1.1.1", "2.2.2.2"]], [["1.1.9.9"], ["2.2.2.2"]], (1,)
    )
    assert got["k=1"] == {
        "level_1_accuracy": 75.0,
        "level_2_accuracy": 75.0,
        "level_3_accuracy": 25.0,
        "level_4_accuracy": 25.0,
    }


def test_miss_and_malformed_predictions():
    assert matched_levels("1.1.1.1", "1.1.1") == 0
    assert matched_levels("1.1.1.1", "1.1.1.9") == 3
    got = ec_level_accuracy([["1.1.1.1"]], [[]], (1, 5))
    assert got["k=5"]["level_1_accuracy"] == 0.0


def test_no_truncation_to_true_ec_count():
    got = ec_level_accuracy([["1.1.1.1"]], [["2.2.2.2", "1.1.1.1"]], (1, 2))
    assert got["k=1"]["level_4_accuracy"] == 0.0
    assert got["k=2"]["level_4_accuracy"] == 100.0


def test_predictions_file_groups_rows_into_queries_and_merges_by_max_score(tmp_path):
    frame = pd.DataFrame(
        {
            "query": ["a", "b", "a", "c"],
            "ec_number": ["1.1.1.1", "2.2.2.2", "3.3.3.3", ""],
            "top_ec": [
                ["2.2.2.2", "3.3.3.3"],
                ["1.1.1.1", "3.3.3.3"],
                ["3.3.3.3", "1.1.1.1"],
                ["1.1.1.1"],
            ],
            "top_score": [[5.0, 3.0], [9.0, 1.0], [9.0, 0.0], [0.0]],
        }
    )
    frame.to_parquet(tmp_path / "p.parquet", index=False)
    got = ec_level_accuracy_from_predictions(tmp_path / "p.parquet")
    assert (got["n_queries"], got["n_rows"]) == (2, 4)
    # Query a ranks 3.3.3.3 (9.0) first, so its 3.3.3.3 is right at k=1 and 1.1.1.1 is not.
    assert got["k=1"]["level_4_accuracy"] == 25.0
    assert got["k=5"]["level_4_accuracy"] == 50.0


def test_query_columns():
    assert QUERY_COLUMN == {"classification": "uniprot_id", "retrieval": "rxn_smiles"}


def test_drop_b_removes_preliminary_ecs_from_truth_and_rankings(tmp_path):
    frame = pd.DataFrame(
        {
            "query": ["a", "a", "b", "c"],
            "ec_number": ["1.1.1.1", "3.4.23.B9", "3.4.23.B9", "2.2.2.2"],
            "top_ec": [
                ["3.4.23.B9", "2.2.2.2"],
                ["3.4.23.B9", "1.1.1.1"],
                ["1.1.1.1"],
                ["3.4.23.B9", "2.2.2.2"],
            ],
            "top_score": [[9.0, 5.0], [9.0, 1.0], [1.0], [9.0, 5.0]],
        }
    )
    frame.to_parquet(tmp_path / "p.parquet", index=False)
    got = ec_level_accuracy_from_predictions(tmp_path / "p.parquet", drop_b=True)
    # Query b has only a B-EC and is not counted; the B-EC leaves the ranked lists, so 2.2.2.2 is top-1.
    assert (got["n_queries"], got["n_rows"], got["b_ecs_dropped"]) == (2, 4, True)
    assert got["k=1"]["level_4_accuracy"] == 50.0
    assert got["k=5"]["level_4_accuracy"] == 50.0
    assert "b_ecs_dropped" not in ec_level_accuracy_from_predictions(tmp_path / "p.parquet")
