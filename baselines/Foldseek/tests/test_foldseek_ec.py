import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

MODULE = Path(__file__).parents[1] / "emulator_bench" / "run_foldseek_ec.py"
spec = importlib.util.spec_from_file_location("foldseek_ec", MODULE)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_selection_follows_the_lddt_split_rule(tmp_path):
    for name in ("mmcif/1AAA_updated.cif", "mmcif/2BBB.cif", "alphafold/AF-P3-F1-model_v1.pdb"):
        touch(tmp_path / name)
    touch(tmp_path / "alphafold/AF-P3-F1-model_v2.pdb")
    touch(tmp_path / "esmfold2_pdb/ESMFold2-P4.pdb")
    touch(tmp_path / "esm/ESM3-open-small-P4.pdb")
    touch(tmp_path / "esm/ESM3-open-small-P5.pdb")
    master = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P1", "P1", "P2", "P3", "P4", "P5", "P6", "P7"],
            "sequence": ["AA", "AA", "AA", "BB", "CC", "DD", "EE", "FF", "GG"],
            "pdbs": ["1AAA:P1", "2BBB:P1", "P1", "9ZZZ:P2", "P3", "P4", "P5", "P6", "7XXX:P7"],
            "pdb_type": ["experimental", "experimental", None, "experimental"]
            + [None] * 4
            + ["experimental"],
            "pdb_chain": ["B", "A", None, "A"] + [None] * 4 + ["A"],
            "pdb_coverage": [0.5, 0.9, None, 0.9] + [None] * 5,
        }
    )
    path = tmp_path / "master.parquet"
    master.to_parquet(path)
    chosen = baseline.select_structures(path, tmp_path).set_index("Entry")
    # P1: best experimental coverage wins over an AlphaFold-less accession and a lower coverage.
    assert chosen.loc["P1", ["pdbs", "source", "hit"]].tolist() == [
        "2BBB:P1",
        "experimental",
        "2BBB_A",
    ]
    assert chosen.loc["P3", ["source", "hit"]].tolist() == ["alphafold", "P3"]
    assert chosen.loc["P3", "path"].endswith("AF-P3-F1-model_v2.pdb")
    assert chosen.loc["P4", "source"] == "esmfold2"
    assert chosen.loc["P5", "source"] == "esm3"
    # No located file: absent (P2 mmCIF missing, P6 and P7 have none).
    assert set(chosen.index) == {"P1", "P3", "P4", "P5"}


def test_hit_names_and_lookup_keys(tmp_path):
    selection = pd.DataFrame({"hit": ["1AAA_A", "P3"], "source": ["experimental", "alphafold"]})
    names = baseline.hit_names(selection)
    assert names == {
        "1AAA_A": "1AAA_A",
        "P3": "P3",
        "P3.pdb": "P3",
        "P3_A": "P3",
        "P3.pdb_A": "P3",
    }
    (tmp_path / "all.lookup").write_text(
        "0\t1AAA_A\t1AAA.cif\n1\t1AAA_B\t1AAA.cif\n2\tP3.pdb_A\tP3.pdb\n"
    )
    assert baseline.read_lookup(tmp_path / "all", names) == {"1AAA_A": 0, "P3": 2}


def test_top_hits_keeps_fifty_by_bitscore_then_name(tmp_path):
    rows = [f"q\tt{i:02d}\t{100 - i // 2}.0" for i in range(60)]
    path = tmp_path / "hits.tsv"
    path.write_text("\n".join(rows) + "\n")
    names = {"q": "q", **{f"t{i:02d}": f"t{i:02d}" for i in range(60)}}
    hits = baseline.top_hits(path, names)["q"]
    assert len(hits) == 50 and hits[0] == ("t00", 100.0) and hits[1] == ("t01", 100.0)
    assert baseline.top_hits(tmp_path / "missing.tsv", names) == {}


def test_search_uses_foldseek_defaults(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(baseline.blastp, "run", lambda command, *a: calls.append(command))
    args = SimpleNamespace(foldseek="foldseek", threads=2, tmpdir=tmp_path, reporter=None)
    baseline.search(
        tmp_path / "q", tmp_path / "t", tmp_path / "h.tsv", args, tmp_path / "log", "x"
    )
    command = calls[0]
    assert command[:2] == ["foldseek", "easy-search"]
    assert command[command.index("--format-output") + 1] == "query,target,bits"
    flags = {"--max-seqs", "-c", "--cov-mode", "-e", "-s", "--alignment-type", "--gpu"}
    assert not flags & set(command)


def test_no_structure_and_no_hit_are_misses(tmp_path, monkeypatch):
    split = tmp_path / "data" / "uniprot_time_splits"
    split.mkdir(parents=True)
    frames = {
        "train": (["T1", "T2", "T3"], ["AAA", "BBB", "CCC"], ["1.1.1.1", "2.2.2.2", "3.3.3.3"]),
        "val": (["V"], ["VVV"], ["1.1.1.1"]),
        "test": (
            ["Q1", "Q2", "Q3", "Q4"],
            ["DDD", "EEE", "FFF", "GGG"],
            ["1.1.1.1", "2.2.2.2", "1.1.-.-", "3.4.23.B9;9.9.9.9"],
        ),
    }
    for part, (ids, seqs, ecs) in frames.items():
        pd.DataFrame({"uniprot_id": ids, "sequence": seqs, "ec_number": ecs}).to_parquet(
            split / f"{part}.parquet"
        )
    selection = pd.DataFrame(
        {
            "Entry": ["T1", "T2", "Q1", "Q4"],
            "Sequence": ["AAA", "BBB", "DDD", "GGG"],
            "source": ["experimental", "experimental", "alphafold", "esm3"],
            "hit": ["1AAA_A", "1BBB_A", "Q1", "Q4"],
            "key": [1, 2, 3, 4],
        }
    )
    names = {h: h for h in selection.hit}
    monkeypatch.setattr(baseline, "subdb", lambda *a: None)

    def fake_search(query_db, target_db, hits, *a):
        hits.write_text("Q1\t1BBB_A\t40.0\nQ1\t1AAA_A\t60.0\n")

    monkeypatch.setattr(baseline, "search", fake_search)
    args = SimpleNamespace(
        dataset_dir=tmp_path / "data",
        baseline_dir=tmp_path / "out",
        foldseek_version="v",
        force=False,
        metrics_module=baseline.blastp.METRICS,
        reporter=None,
        tmpdir=tmp_path,
    )
    summary = baseline.evaluate_split(split, selection, names, tmp_path / "db", "sig", args)
    assert (summary["queries"], summary["hits"], summary["no_hit"], summary["no_structure"]) == (
        3,
        1,
        1,
        1,
    )
    assert summary["train_proteins"] == 3 and summary["train_with_structure"] == 2
    assert summary["structure_sources"]["train"] == {"experimental": 2}
    frame = pd.read_parquet(
        tmp_path / "out/enzyme_classification/uniprot_time_splits/all/test_predictions.parquet"
    )
    assert frame.set_index("Entry")["0"].to_dict() == {"Q1": "1.1.1.1", "Q2": None, "Q4": None}
    assert frame.set_index("Entry")["1"]["Q1"] == "2.2.2.2"
    # Q3 has only a partial EC and is not a query; Q2 (no structure) and Q4 (no hit) are misses.
    assert summary["care_task1"]["k=1"]["level_4_accuracy"] == 33.3333
    assert (
        json.loads(
            (
                tmp_path / "out/enzyme_classification/uniprot_time_splits/all/complete.json"
            ).read_text()
        )["parameters"]["structure_selection"]
        == "sig"
    )


def test_metrics_equal_graphec_function_and_ranking_is_blastps():
    spec = importlib.util.spec_from_file_location("graphec_results", baseline.blastp.METRICS)
    graphec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphec)
    queries = pd.DataFrame(
        {"Entry": ["a"], "protein_id": ["x"], "ec_labels": [["1.1.1.1", "3.4.23.B9"]]}
    )
    frame = baseline.blastp.care_frame(queries, [["1.1.1.9", "3.4.23.B9"]])
    assert baseline.blastp.care_metrics(frame) == graphec.compute_care_metrics(frame)
    assert baseline.blastp.is_full_ec("3.4.23.B9") and not baseline.blastp.is_full_ec("1.1.-.-")
