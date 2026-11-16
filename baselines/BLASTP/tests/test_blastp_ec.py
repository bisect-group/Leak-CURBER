import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

MODULE = Path(__file__).parents[1] / "emulator_bench" / "run_blastp_ec.py"
spec = importlib.util.spec_from_file_location("blastp_ec", MODULE)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
CLEAN_RESULTS = Path(__file__).parents[2] / "CLEAN/emulator_bench/results.py"


def test_rank_by_best_bitscore_then_frequency_then_string():
    labels = {"a": ["1.1.1.1", "2.2.2.2"], "b": ["3.3.3.3", "1.1.1.1"], "c": ["0.0.0.1"]}
    freq = Counter({"1.1.1.1": 5, "2.2.2.2": 9, "3.3.3.3": 9, "0.0.0.1": 1})
    ranked = baseline.rank_ecs([("a", 100.0), ("b", 120.0), ("c", 100.0)], labels, freq)
    # 1.1.1.1 and 3.3.3.3 share the best hit (120); the more frequent wins, ties by string.
    assert ranked == ["3.3.3.3", "1.1.1.1", "2.2.2.2", "0.0.0.1"]
    assert baseline.rank_ecs([("a", 100.0)], labels, freq) == ["2.2.2.2", "1.1.1.1"]
    assert baseline.rank_ecs(
        [("a", 1.0)] * 3, {"a": [f"1.1.1.{i}" for i in range(30)]}, Counter(), depth=20
    )[:2] == ["1.1.1.0", "1.1.1.1"]


def test_no_hit_is_a_miss():
    assert baseline.rank_ecs([], {}, Counter()) == []
    queries = pd.DataFrame({"Entry": ["Q"], "protein_id": ["x"], "ec_labels": [["1.1.1.1"]]})
    metrics = baseline.care_metrics(baseline.care_frame(queries, [[]]))
    assert metrics["k=1"]["level_1_accuracy"] == 0 and metrics["k=20"]["level_4_support"] == 1


def test_full_ec_rule_keeps_preliminary_and_drops_partial(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2", "P3"],
            "sequence": ["AAA", "BBB", "CCC"],
            "ec_number": ["3.4.23.B9;1.1.-.-", "1.2.-.-", "1.1.1.1"],
        }
    ).to_parquet(path)
    frame = baseline.partition_frame(path)
    assert dict(zip(frame.Entry, frame.ec_labels)) == {"P1": ["3.4.23.B9"], "P3": ["1.1.1.1"]}


def test_split_discovery_finds_grouped_random_and_ignores_backups(tmp_path):
    for rel in (
        "random_splits_grouped_sequence",
        "random_splits",
        "backup_random_splits_grouped_sequence",
        "uniprot_time_splits",
        "enzyme_sequence_splits/threshold_0.385",
        "enzyme_sequence_splits/backup_threshold_0.4",
        "enzyme_structure_splits/threshold_0.855",
    ):
        (tmp_path / rel).mkdir(parents=True)
        for part in baseline.PARTS:
            (tmp_path / rel / f"{part}.parquet").touch()
    found = [str(p.relative_to(tmp_path)) for p in baseline.discover_splits(tmp_path)]
    assert found == [
        "random_splits_grouped_sequence",
        "uniprot_time_splits",
        "enzyme_sequence_splits/threshold_0.385",
        "enzyme_structure_splits/threshold_0.855",
    ]


def test_split_dir_flat_is_all_and_threshold_is_kept(tmp_path):
    flat = baseline.split_dir(tmp_path, Path("uniprot_time_splits"))
    assert flat == tmp_path / "enzyme_classification/uniprot_time_splits/all"
    threshold = baseline.split_dir(tmp_path, Path("enzyme_sequence_splits/threshold_0.385"))
    assert threshold == tmp_path / "enzyme_classification/enzyme_sequence_splits/threshold_0.385"


def test_search_is_care_diamond_with_k50(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(baseline, "run", lambda command, log, **kwargs: calls.append(command))
    args = type(
        "A", (), {"diamond": "diamond", "threads_per_run": 1, "tmpdir": tmp_path, "reporter": None}
    )
    baseline.search(
        tmp_path / "q.fasta",
        tmp_path / "db",
        tmp_path / "hits.tsv",
        args,
        tmp_path / "log",
        "fixture",
    )
    command = calls[0]
    assert command[:3] == ["diamond", "blastp", "-k"] and command[3] == "50"
    assert not {"--query-cover", "--evalue", "-e", "--sensitive", "--more-sensitive"} & set(
        command
    )


def test_empty_hits_file_gives_empty_frame(tmp_path):
    assert baseline.read_hits(tmp_path / "missing.tsv").empty


def test_metrics_equal_graphec_function_on_fixture():
    queries = pd.DataFrame(
        {
            "Entry": ["a", "b"],
            "protein_id": ["x", "y"],
            "ec_labels": [["1.1.1.1", "2.3.4.5"], ["3.4.23.B9"]],
        }
    )
    frame = baseline.care_frame(queries, [["1.1.1.9", "2.3.4.5"], []])
    spec = importlib.util.spec_from_file_location("graphec_results", baseline.METRICS)
    graphec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphec)
    metrics = baseline.care_metrics(frame)
    assert metrics == graphec.compute_care_metrics(frame)
    assert (
        metrics["k=1"]["level_3_accuracy"] == 25.0 and metrics["k=20"]["level_4_accuracy"] == 25.0
    )


def test_clean_function_differs_only_by_k20_truncation():
    spec = importlib.util.spec_from_file_location("clean_results", CLEAN_RESULTS)
    clean = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clean)
    queries = pd.DataFrame({"Entry": ["a"], "protein_id": ["x"], "ec_labels": [["1.1.1.1"]]})
    frame = baseline.care_frame(queries, [["9.9.9.9", "1.1.1.1"]])
    assert clean.compute_care_metrics(frame)["k=1"] == baseline.care_metrics(frame)["k=1"]
