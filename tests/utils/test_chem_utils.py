from __future__ import annotations

import io
import json
import gzip
import tarfile
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import requests

from src.utils.chem_utils import (
    ChemUtils,
    LazyChemUtils,
    null_implausible_conditions,
)


def test_reaction_canonicalization_ignores_component_order() -> None:
    identity = {s: s for s in ("CC", "CO", "CN", "CCO")}
    chem = ChemUtils()
    forward = chem.canonicalize_reaction_smiles("CC.CO.CN>>CCO", identity)
    backward = chem.canonicalize_reaction_smiles("CN.CO.CC>>CCO", identity)
    assert forward == backward == "CC.CN.CO>>CCO"


def test_molecule_key_merges_twins_and_keeps_stereo() -> None:
    from src.utils.chem_utils import molecule_key

    assert molecule_key("N") == molecule_key("[NH4+]")
    assert molecule_key("CC(=O)[O-].[Na+]") == molecule_key("CC(=O)O")
    assert molecule_key("OC(C)=C") == molecule_key("CC(C)=O")
    assert molecule_key("C[C@H](N)C(=O)O") != molecule_key("C[C@@H](N)C(=O)O")


def test_molecule_key_is_none_when_key_does_not_reparse() -> None:
    from src.utils.chem_utils import molecule_key

    ferrocene = "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1"
    assert molecule_key(ferrocene) is None
    assert ChemUtils().molecule_key_series([ferrocene], fallback=["canon"])[0].tolist() == ["canon"]


def test_reaction_key_ignores_maps_order_and_spectators() -> None:
    chem = ChemUtils()
    key = chem.reaction_key("CC.CN>>CCN")
    assert chem.reaction_key("[CH3:1][CH3:2].[CH3:3][NH2:4]>>CCN") == key
    assert chem.reaction_key("CN.CC>>CCN") == key
    assert chem.reaction_key("CC.CN.O>>CCN.[H+]") == key
    assert chem.reaction_key("CCN>>CC.CN") != key  # orientation is kept
    assert chem.reaction_key("C[C@H](N)C(=O)O>>C") != chem.reaction_key("C[C@@H](N)C(=O)O>>C")


@pytest.mark.slow
def test_key_failures_fall_back_instead_of_dropping_rows() -> None:
    chem = ChemUtils()
    keys, n_fallback = chem.molecule_key_series(["N", "[NH4+]", "not_a_smiles"])
    assert keys.tolist() == ["N", "N", "not_a_smiles"] and n_fallback == 1
    keys, _ = chem.molecule_key_series(["bad"], fallback=["canon"])
    assert keys.tolist() == ["canon"]
    rxn = pd.Series(["CC.CN>>CCN", "bad>>CCN", None])
    out = chem.reaction_key_series_with_fallback(rxn, logger=MagicMock())
    assert out.tolist() == ["CC.CN>>CCN", "bad>>CCN", None]


def test_importing_modules_does_not_build_chem_utils() -> None:
    import src.utils.chem_utils as module

    assert isinstance(module.LazyChemUtils(), LazyChemUtils)
    # The data modules were imported above; nothing may have built a ChemUtils at import.
    assert module._CHEM_UTILS is None or isinstance(module._CHEM_UTILS, ChemUtils)
    import src.data.components.splitters.random  # noqa: F401
    import src.data.components.splitters.smiles_tanimoto as smiles_tanimoto

    assert isinstance(smiles_tanimoto.chem_utils, LazyChemUtils)
    pickle.dumps(smiles_tanimoto.chem_utils.bind("canonicalize_smiles"))  # picklable for multiprocessing


def test_canonicalize_falls_back_to_serial_without_pandarallel(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "pandarallel", None
    )  # makes `import pandarallel` raise ImportError
    utils = object.__new__(ChemUtils)
    utils.logger = MagicMock()
    # Must not reach parallel_apply after the serial fallback.
    canonical, _stats = utils.canonicalize_reaction_smiles_series(
        pd.Series(["CCO>>CC=O", "OCC>>CC=O"])
    )
    assert canonical.iloc[0] == canonical.iloc[1] and canonical.notna().all()


def test_bound_chem_utils_method_pickles() -> None:
    bound = LazyChemUtils().bind("canonicalize_smiles")
    assert pickle.loads(pickle.dumps(bound)).name == "canonicalize_smiles"


def test_canonicalize_smiles_is_idempotent() -> None:
    stub = SimpleNamespace(logger=MagicMock())
    once = ChemUtils.canonicalize_smiles(stub, r"C(=C/c1ccccc1)\c1ccccc1")
    assert once is not None
    assert ChemUtils.canonicalize_smiles(stub, once) == once


def test_implausible_conditions_become_nan() -> None:
    df = pd.DataFrame(
        {"t": [298.0, 1937.0, 10.0, np.nan], "ph": [7.0, 60.0, -1.0, 14.0], "y": [1, 2, 3, 4]}
    )
    counts = null_implausible_conditions(df, "t", "ph")
    assert counts == {"t": 2, "ph": 2}
    assert df["t"].isna().tolist() == [False, True, True, True]
    assert df["ph"].isna().tolist() == [False, True, True, False]
    assert df["y"].tolist() == [1, 2, 3, 4]


def test_boiling_point_assay_temperature_is_kept() -> None:
    df = pd.DataFrame({"t": [373.15, 373.2], "ph": [7.0, 7.0]})
    null_implausible_conditions(df, "t", "ph")
    assert df["t"].isna().tolist() == [False, True]


def test_save_json_writes_valid_json_and_leaves_no_temp_file(tmp_path) -> None:
    path = tmp_path / "a.json"
    ChemUtils.save_json(SimpleNamespace(logger=MagicMock()), {"x": 1}, path)

    assert json.loads(path.read_text()) == {"x": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["a.json"]


def _uniprot_entry(acc: str, with_ec: bool) -> dict:
    name = {"ecNumbers": [{"value": "1.1.1.1"}]} if with_ec else {}
    return {
        "proteinDescription": {"recommendedName": name},
        "uniProtKBCrossReferences": [],
        "sequence": {"value": "MK"},
    }


def _result(acc: str, **extra) -> dict:
    return {"primaryAccession": acc, **_uniprot_entry(acc, True), **extra}


def _uniprot_chem(tmp_path, monkeypatch, handler):
    """ChemUtils whose UniProt search calls `handler(query) -> list | int (status)`."""
    chem = ChemUtils()
    chem.logger = MagicMock()
    monkeypatch.setattr(chem, "UNIPROT_ACC_PATH", tmp_path)
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", lambda s: None)
    queries: list[str] = []

    def fake_get(url, params, timeout):
        queries.append(params["query"])
        out = handler(params["query"])
        status = out if isinstance(out, int) else 200
        return SimpleNamespace(
            status_code=status, json=lambda: {"results": [] if isinstance(out, int) else out}
        )

    monkeypatch.setattr("src.utils.chem_utils.requests.get", fake_get)
    return chem, queries


def _accs_in(query: str) -> list[str]:
    return [q.removeprefix("accession:") for q in query.split(" OR ")]


def test_uniprot_acc_json_omits_accessions_without_data(tmp_path, monkeypatch) -> None:
    chem, _ = _uniprot_chem(tmp_path, monkeypatch, lambda q: [_result("P1")])
    out = chem.get_uniprot_acc_json(["P1", "P2"])
    assert list(out) == ["P1"] and out["P1"]["ecs"] == ["1.1.1.1"]


def test_uniprot_entry_without_ec_numbers_gives_empty_list_and_no_warning(
    tmp_path, monkeypatch
) -> None:
    entry = {"primaryAccession": "P1", **_uniprot_entry("P1", False)}
    chem, _ = _uniprot_chem(tmp_path, monkeypatch, lambda q: [entry])
    assert chem.get_uniprot_acc_json(["P1"])["P1"]["ecs"] == []
    chem.logger.warning.assert_not_called()


def test_uniprot_acc_json_fresh_pdbs_match_cached_list(tmp_path, monkeypatch) -> None:
    refs = [{"database": "PDB", "id": "1ABC"}, {"database": "PDB", "id": "2DEF"}]
    chem, _ = _uniprot_chem(
        tmp_path, monkeypatch, lambda q: [_result("P1", uniProtKBCrossReferences=refs)]
    )
    fresh = chem.get_uniprot_acc_json(["P1"])["P1"]["pdbs"]
    assert fresh == ["1ABC", "2DEF"] and isinstance(fresh, list)
    assert chem.get_uniprot_acc_json(["P1"])["P1"]["pdbs"] == fresh


def test_uniprot_acc_json_batches_requests(tmp_path, monkeypatch) -> None:
    chem, queries = _uniprot_chem(
        tmp_path, monkeypatch, lambda q: [_result(a) for a in _accs_in(q)]
    )
    accs = [f"P{i}" for i in range(150)]
    assert len(chem.get_uniprot_acc_json(accs)) == 150
    assert sorted(len(_accs_in(q)) for q in queries) == [50, 100]


def test_uniprot_acc_json_skips_cached_accessions(tmp_path, monkeypatch) -> None:
    chem, queries = _uniprot_chem(
        tmp_path, monkeypatch, lambda q: [_result(a) for a in _accs_in(q)]
    )
    (tmp_path / "P1.json").write_text(json.dumps({"acc_id": "P1", "sequence": "CACHED"}))
    out = chem.get_uniprot_acc_json(["P1", "P2"])
    assert out["P1"]["sequence"] == "CACHED"
    assert queries == ["accession:P2"]


def test_uniprot_acc_json_missing_accession_is_omitted_and_not_cached(
    tmp_path, monkeypatch
) -> None:
    chem, _ = _uniprot_chem(tmp_path, monkeypatch, lambda q: [_result("P1")])
    assert list(chem.get_uniprot_acc_json(["P1", "P2"])) == ["P1"]
    assert not (tmp_path / "P2.json").exists()
    chem.logger.warning.assert_called_once()


def test_uniprot_acc_json_failed_request_caches_nothing(tmp_path, monkeypatch) -> None:
    chem, queries = _uniprot_chem(tmp_path, monkeypatch, lambda q: 500)
    assert chem.get_uniprot_acc_json(["P1", "P2"]) == {}
    assert len(queries) == 6  # three attempts in the main pass and three in the retry pass
    assert list(tmp_path.iterdir()) == []


def test_uniprot_acc_json_retry_pass_uses_small_batches(
    tmp_path, monkeypatch
) -> None:
    def handler(query):
        accs = _accs_in(query)
        return 500 if len(accs) > 25 else [_result(a) for a in accs]

    chem, queries = _uniprot_chem(tmp_path, monkeypatch, handler)
    accs = [f"P{i}" for i in range(100)]
    assert len(chem.get_uniprot_acc_json(accs)) == 100
    assert sorted({len(_accs_in(q)) for q in queries}) == [25, 100]
    chem.logger.info.assert_any_call("UniProt retry recovered 100 accessions; 0 still failed.")
    assert len(list(tmp_path.iterdir())) == 100


def test_uniprot_acc_json_retry_pass_waits_longer_and_logs_remaining_failures(tmp_path, monkeypatch) -> None:
    chem, _ = _uniprot_chem(tmp_path, monkeypatch, lambda q: 500)
    waits: list[float] = []
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", waits.append)
    chem.get_uniprot_acc_json(["P1"])
    assert waits == [1, 2, 4, 15, 30, 60]
    chem.logger.info.assert_any_call("UniProt retry recovered 0 accessions; 1 still failed.")


def test_uniprot_acc_json_merged_accession_uses_target_data(tmp_path, monkeypatch) -> None:
    merged = {
        "primaryAccession": "OLD",
        "entryType": "Inactive",
        "inactiveReason": {"inactiveReasonType": "MERGED", "mergeDemergeTo": ["NEW"]},
    }

    def handler(query):
        return [merged] if "OLD" in query else [_result("NEW", sequence={"value": "NEWSEQ"})]

    chem, queries = _uniprot_chem(tmp_path, monkeypatch, handler)
    out = chem.get_uniprot_acc_json(["OLD"])
    assert queries == ["accession:OLD", "accession:NEW"]
    assert out["OLD"]["acc_id"] == "OLD" and out["OLD"]["sequence"] == "NEWSEQ"
    assert not (tmp_path / "NEW.json").exists()


def test_uniprot_acc_json_cache_file_format(tmp_path, monkeypatch) -> None:
    refs = [
        {"database": "PDB", "id": "1ABC"},
        {"database": "Other", "id": "x"},
        {"database": "PDB", "id": "2XYZ"},
    ]
    chem, _ = _uniprot_chem(
        tmp_path, monkeypatch, lambda q: [_result("P1", uniProtKBCrossReferences=refs)]
    )
    chem.get_uniprot_acc_json(["P1"])
    expected = {"acc_id": "P1", "ecs": ["1.1.1.1"], "pdbs": ["1ABC", "2XYZ"], "sequence": "MK"}
    assert json.loads((tmp_path / "P1.json").read_text()) == expected


def test_uniprot_acc_json_malformed_accession_does_not_fail_batch(tmp_path, monkeypatch) -> None:
    def handler(query):
        if "BAD" in _accs_in(query):
            return 400
        return [_result(a) for a in _accs_in(query)]

    chem, _ = _uniprot_chem(tmp_path, monkeypatch, handler)
    assert sorted(chem.get_uniprot_acc_json(["P1", "BAD", "P2", "P3"])) == ["P1", "P2", "P3"]


def _ec(*values: str) -> dict:
    return {"recommendedName": {"ecNumbers": [{"value": v} for v in values]}}


def _alt(*values: str) -> dict:
    return {"ecNumbers": [{"value": v} for v in values]}


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ({"includes": [{}, _ec("2.4.2.9")]}, ["2.4.2.9"]),
        (
            {"contains": [_ec("3.4.22.29"), {}, _ec("3.4.21.98", "3.6.1.15")]},
            ["3.4.22.29", "3.4.21.98", "3.6.1.15"],
        ),
        (
            {
                **_ec("1.1.1.1"),
                "includes": [_ec("2.2.2.2", "1.1.1.1")],
                "contains": [_ec("3.3.3.3", "2.2.2.2")],
            },
            ["1.1.1.1", "2.2.2.2", "3.3.3.3"],
        ),
        ({"recommendedName": {}, "includes": [{}], "contains": []}, []),
        ({"alternativeNames": [_alt("3.1.4.1"), {}, _alt("3.6.1.9")]}, ["3.1.4.1", "3.6.1.9"]),
        (
            {
                "alternativeNames": [_alt("9.9.9.9")],
                **_ec("1.1.1.1"),
                "includes": [{"alternativeNames": [_alt("2.2.2.2")], **_ec("3.3.3.3")}],
                "contains": [{"alternativeNames": [_alt("4.4.4.4", "1.1.1.1")]}],
            },
            ["1.1.1.1", "9.9.9.9", "3.3.3.3", "2.2.2.2", "4.4.4.4"],
        ),
    ],
)
def test_uniprot_acc_record_collects_ecs_from_includes_and_contains(
    tmp_path, monkeypatch, description, expected
) -> None:
    chem, _ = _uniprot_chem(tmp_path, monkeypatch, lambda q: [])
    record = chem._uniprot_acc_record(
        "P1",
        {
            "proteinDescription": description,
            "uniProtKBCrossReferences": [],
            "sequence": {"value": "MK"},
        },
    )
    assert record["ecs"] == expected
    chem.logger.warning.assert_not_called()


def _paged_chem(tmp_path, monkeypatch, handler):
    """ChemUtils whose UniProt search calls `handler(url, params) -> (results, next_url) | int`."""
    chem = ChemUtils()
    chem.logger = MagicMock()
    monkeypatch.setattr(chem, "UNIPROT_EC_ACC_PATH", tmp_path / "ec")
    monkeypatch.setattr(chem, "UNIPROT_EC_ORG_ACC_PATH", tmp_path / "ecorg")
    (tmp_path / "ec").mkdir()
    (tmp_path / "ecorg").mkdir()
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", lambda s: None)
    calls: list[tuple[str, dict | None]] = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        out = handler(url, params)
        if isinstance(out, int):
            return SimpleNamespace(status_code=out, links={}, json=lambda: {})
        results, next_url = out
        links = {"next": {"url": next_url}} if next_url else {}
        return SimpleNamespace(status_code=200, links=links, json=lambda: {"results": results})

    monkeypatch.setattr("src.utils.chem_utils.requests.get", fake_get)
    return chem, calls


def _hit(acc: str, name: str, reviewed: bool = True, extra: dict | None = None, **organism) -> dict:
    kind = "reviewed (Swiss-Prot)" if reviewed else "unreviewed (TrEMBL)"
    return {
        "primaryAccession": acc,
        "entryType": f"UniProtKB {kind}",
        "organism": {"scientificName": name, **organism},
        "proteinExistence": "1: Evidence at protein level",
        "annotationScore": 1.0,
        "sequence": {"value": "MK"},
        **(extra or {}),
    }


def test_search_uniprot_all_follows_next_links(tmp_path, monkeypatch) -> None:
    pages = {"u0": ([{"n": 1}], "u1"), "u1": ([{"n": 2}], "u2"), "u2": ([{"n": 3}], None)}
    chem, calls = _paged_chem(tmp_path, monkeypatch, lambda url, p: pages["u0" if p else url])
    chem.UNIPROT_API_URL = "u0"
    assert chem._search_uniprot_all("q", "accession") == [{"n": 1}, {"n": 2}, {"n": 3}]
    assert [c[0] for c in calls] == ["u0", "u1", "u2"]
    assert calls[0][1]["size"] == 500 and calls[1][1] is None


def test_search_uniprot_all_retries_then_gives_up(tmp_path, monkeypatch) -> None:
    chem, calls = _paged_chem(tmp_path, monkeypatch, lambda url, p: 500)
    assert chem._search_uniprot_all("q", "accession") is None
    assert len(calls) == 3
    chem.logger.warning.assert_called_once()


def test_ec_acc_map_pages_reviewed_hits_and_caches_list(tmp_path, monkeypatch) -> None:
    def handler(url, params):
        if params and "reviewed:true" in params["query"]:
            return [{"primaryAccession": "R1"}], "next"
        return [{"primaryAccession": "R2"}], None

    chem, calls = _paged_chem(tmp_path, monkeypatch, handler)
    assert chem.get_uniprot_ec_acc_map(["1.1.1.1"]) == {"1.1.1.1": ["R1", "R2"]}
    assert json.loads((tmp_path / "ec" / "1.1.1.1.json").read_text()) == ["R1", "R2"]
    assert len(calls) == 2


def test_ec_acc_map_skips_partial_ecs_and_looks_up_full_ones(tmp_path, monkeypatch) -> None:
    chem, calls = _paged_chem(tmp_path, monkeypatch, lambda url, params: ([{"primaryAccession": "R1"}], None))
    ecs = ["2.1.1.-", "2.-.-.-", "2.7.11", "2.7.11.n1"]
    assert chem.get_uniprot_ec_acc_map(ecs) == {
        "2.1.1.-": None,
        "2.-.-.-": None,
        "2.7.11": None,
        "2.7.11.n1": ["R1"],
    }
    assert [c[1]["query"] for c in calls] == ["ec:2.7.11.n1 AND active:true AND reviewed:true"]


def test_ec_acc_map_falls_back_to_unreviewed_and_caches_null_without_hits(
    tmp_path, monkeypatch
) -> None:
    def handler(url, params):
        if "1.1.1.1" in params["query"] and "reviewed:false" in params["query"]:
            return [{"primaryAccession": "U1"}], None
        return [], None

    chem, _ = _paged_chem(tmp_path, monkeypatch, handler)
    out = chem.get_uniprot_ec_acc_map(["1.1.1.1", "2.2.2.2"])
    assert out == {"1.1.1.1": ["U1"], "2.2.2.2": None}
    assert (tmp_path / "ec" / "2.2.2.2.json").read_text() == "null"


def test_ec_acc_map_failed_request_is_not_cached(tmp_path, monkeypatch) -> None:
    chem, _ = _paged_chem(tmp_path, monkeypatch, lambda url, p: 500)
    assert chem.get_uniprot_ec_acc_map(["1.1.1.1"]) == {"1.1.1.1": None}
    assert list((tmp_path / "ec").iterdir()) == []


def test_ec_acc_map_retry_pass_recovers_failed_lookups(tmp_path, monkeypatch) -> None:
    waits: list[float] = []
    state = {"calls": 0}

    def handler(url, params):
        state["calls"] += 1
        return 500 if state["calls"] <= 3 else ([{"primaryAccession": "R1"}], None)

    chem, _ = _paged_chem(tmp_path, monkeypatch, handler)
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", waits.append)
    assert chem.get_uniprot_ec_acc_map(["1.1.1.1"]) == {"1.1.1.1": ["R1"]}
    assert json.loads((tmp_path / "ec" / "1.1.1.1.json").read_text()) == ["R1"]
    assert waits[:3] == [1, 2, 4]
    chem.logger.info.assert_any_call("UniProt retry recovered 1 EC lookups; 0 still failed.")


def test_ec_org_batches_by_ec_and_assigns_hits_to_pairs(tmp_path, monkeypatch) -> None:
    hits = {
        "1.1.1.1": [
            _hit("P1", "Homo sapiens", True, commonName="Human"),
            _hit("P2", "Homo sapiens", False),
            _hit("P3", "Escherichia coli (strain K12)", True),
            _hit("P4", "Bos indicus x Bos taurus", False),
            _hit(
                "P5",
                "Saccharomyces pastorianus",
                False,
                synonyms=["Saccharomyces cerevisiae x Saccharomyces eubayanus"],
            ),
            _hit("P6", "Escherichia colinus", False),
        ],
        "2.2.2.2": [_hit("P7", "Homo sapiens")],
    }
    chem, calls = _paged_chem(
        tmp_path, monkeypatch, lambda url, p: (hits[p["query"].split()[0][3:]], None)
    )
    pairs = [
        ("1.1.1.1", "Homo sapiens"),
        ("1.1.1.1", "Escherichia coli"),
        ("1.1.1.1", "Bos taurus"),
        ("1.1.1.1", "Saccharomyces cerevisiae"),
        ("1.1.1.1", "Rattus norvegicus"),
        ("2.2.2.2", "Homo sapiens"),
    ]
    df = chem.get_active_acc_id_from_ec_and_org_parallel(pairs)
    assert len(calls) == 2
    query = next(c[1]["query"] for c in calls if c[1]["query"].startswith("ec:1.1.1.1"))
    assert query == (
        'ec:1.1.1.1 AND active:true AND (organism_name:"Homo sapiens" OR '
        'organism_name:"Escherichia coli" OR organism_name:"Bos taurus" OR '
        'organism_name:"Saccharomyces cerevisiae" OR organism_name:"Rattus norvegicus")'
    )
    got = {(r.ECNumber, r.Organism): set() for r in df.itertuples()}
    for r in df.itertuples():
        got[r.ECNumber, r.Organism].add((r.UniprotID, r.reviewed))
    assert got == {
        ("1.1.1.1", "Homo sapiens"): {("P1", True), ("P2", False)},
        ("1.1.1.1", "Escherichia coli"): {("P3", True)},
        ("1.1.1.1", "Bos taurus"): {("P4", False)},
        ("1.1.1.1", "Saccharomyces cerevisiae"): {("P5", False)},
        ("2.2.2.2", "Homo sapiens"): {("P7", True)},
    }


def test_ec_org_records_carry_ranking_and_matching_fields(tmp_path, monkeypatch) -> None:
    extra = {
        "proteinExistence": "2: Evidence at transcript level",
        "annotationScore": 3.5,
        "genes": [
            {
                "geneName": {"value": "ADH5"},
                "synonyms": [{"value": "ADHX"}],
                "orfNames": [{"value": "ORF1"}],
                "orderedLocusNames": [{"value": "b0001"}],
            },
            {"orfNames": [{"value": "ORF2"}]},
        ],
        "references": [
            {
                "citation": {
                    "citationCrossReferences": [
                        {"database": "PubMed", "id": "111"},
                        {"database": "DOI", "id": "10.1/x"},
                    ]
                }
            },
            {"citation": {}},
        ],
        "sequence": {"value": "MKV"},
    }
    chem, calls = _paged_chem(
        tmp_path, monkeypatch, lambda url, p: ([_hit("P1", "Homo sapiens", extra=extra)], None)
    )
    df = chem.get_active_acc_id_from_ec_and_org_parallel([("1.1.1.1", "Homo sapiens")])
    assert calls[0][1]["fields"] == (
        "accession,organism_name,protein_existence,annotation_score,gene_names,"
        "lit_pubmed_id,sequence"
    )
    expected = {
        "reviewed": True,
        "UniprotID": "P1",
        "ECNumber": "1.1.1.1",
        "Organism": "Homo sapiens",
        "protein_existence": 2,
        "annotation_score": 3.5,
        "genes": ["ADH5", "ADHX", "ORF1", "b0001", "ORF2"],
        "pmids": ["111"],
        "sequence": "MKV",
    }
    assert df.to_dict("records") == [expected]
    record = json.loads((tmp_path / "ecorg" / "1.1.1.1 | Homo sapiens.json").read_text())
    assert record == [expected]


def test_ec_org_caches_empty_list_and_reuses_cache(tmp_path, monkeypatch) -> None:
    chem, calls = _paged_chem(tmp_path, monkeypatch, lambda url, p: ([], None))
    df = chem.get_active_acc_id_from_ec_and_org_parallel([("1.1.1.1", "Nowhere")])
    assert df.empty and list(df.columns)[:4] == ["reviewed", "UniprotID", "ECNumber", "Organism"]
    assert (tmp_path / "ecorg" / "1.1.1.1 | Nowhere.json").read_text() == "[]"
    chem.get_active_acc_id_from_ec_and_org_parallel([("1.1.1.1", "Nowhere")])
    assert len(calls) == 1


def test_ec_org_failed_request_caches_nothing(tmp_path, monkeypatch) -> None:
    chem, _ = _paged_chem(tmp_path, monkeypatch, lambda url, p: 500)
    df = chem.get_active_acc_id_from_ec_and_org_parallel([("1.1.1.1", "Homo sapiens")])
    assert df.empty
    assert list((tmp_path / "ecorg").iterdir()) == []


def test_ec_org_chunks_organisms_and_pages(tmp_path, monkeypatch) -> None:
    def handler(url, params):
        if params:
            return [_hit("P1", "Org0")], "next"
        return [_hit("P2", "Org0")], None

    chem, calls = _paged_chem(tmp_path, monkeypatch, handler)
    pairs = [("1.1.1.1", f"Org{i}") for i in range(51)]
    df = chem.get_active_acc_id_from_ec_and_org_parallel(pairs)
    assert [c[1] is None for c in calls].count(False) == 2
    assert sorted(df["UniprotID"]) == ["P1", "P2"] and set(df["Organism"]) == {"Org0"}


def _synonym_chem(tmp_path, monkeypatch):
    """ChemUtils with a tiny synonym file; returns it and the list of scanned name sets."""
    import gzip

    lines = "5\tAcetate\n5\tacetic acid\n9\tAcetate\n7\tGlucose\n8\tothers\n"
    gz_path = tmp_path / "syn.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(lines)
    chem = ChemUtils()
    chem.logger = MagicMock()
    monkeypatch.setattr(chem, "PUBCHEM_SYNONYM_GZ_FILE_PATH", gz_path)
    monkeypatch.setattr(chem, "PUBCHEM_NAME_CID_CACHE_FILE_PATH", tmp_path / "cache.json")
    scans: list[set] = []
    scan = ChemUtils._scan_pubchem_synonyms

    def spy(path, keys):
        scans.append(set(keys))
        return scan(path, keys)

    monkeypatch.setattr(chem, "_scan_pubchem_synonyms", spy)
    return chem, scans


def test_pubchem_name_lookup_is_case_insensitive_and_takes_lowest_cid(
    tmp_path, monkeypatch
) -> None:
    chem, _ = _synonym_chem(tmp_path, monkeypatch)
    out = chem.get_pubchem_cids_from_name_parallel([" ACETATE ", "glucose", "nothing"])
    assert out == [
        {"compound_name": " ACETATE ", "cid": "5"},
        {"compound_name": "glucose", "cid": "7"},
        {"compound_name": "nothing", "cid": None},
    ]


def test_pubchem_name_lookup_cache_scans_only_new_names(tmp_path, monkeypatch) -> None:
    chem, scans = _synonym_chem(tmp_path, monkeypatch)
    chem.get_pubchem_cids_from_name_parallel(["acetate", "nothing"])
    chem.get_pubchem_cids_from_name_parallel(["Acetate", "nothing"])
    assert scans == [{"acetate", "nothing"}]

    out = chem.get_pubchem_cids_from_name_parallel(["acetate", "glucose"])
    assert scans[1:] == [{"glucose"}]
    assert [o["cid"] for o in out] == ["5", "7"]


def test_pubchem_synonym_download_resumes_and_renames(tmp_path, monkeypatch) -> None:
    chem = ChemUtils()
    chem.logger = MagicMock()
    gz_path = tmp_path / "syn.gz"
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("data")

    monkeypatch.setattr("subprocess.run", fake_run)
    chem._download_file("http://x/syn.gz", gz_path)
    chem._download_file("http://x/syn.gz", gz_path)

    assert len(calls) == 1
    assert "-C" in calls[0] and calls[0][calls[0].index("-C") + 1] == "-"
    assert calls[0][calls[0].index("-o") + 1] == f"{gz_path}.part"
    assert [f.name for f in tmp_path.iterdir()] == ["syn.gz"]


def _smiles_chem(tmp_path, monkeypatch):
    """ChemUtils with a tiny CID-SMILES file; returns it and the list of scanned CID sets."""
    import gzip

    gz_path = tmp_path / "cid_smiles.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write("1\tCC(=O)O\n2\tC1CCCCC1\n3\tCCO\n")
    chem = ChemUtils()
    chem.logger = MagicMock()
    monkeypatch.setattr(chem, "PUBCHEM_CID_SMILES_GZ_FILE_PATH", gz_path)
    monkeypatch.setattr(chem, "PUBCHEM_CID_SMILES_CACHE_FILE_PATH", tmp_path / "cache.json")
    scans: list[set] = []
    scan = ChemUtils._scan_pubchem_smiles

    def spy(path, cids):
        scans.append(set(cids))
        return scan(path, cids)

    monkeypatch.setattr(chem, "_scan_pubchem_smiles", spy)
    return chem, scans


def test_pubchem_smiles_lookup_returns_smiles_and_none_for_misses(
    tmp_path, monkeypatch
) -> None:
    chem, _ = _smiles_chem(tmp_path, monkeypatch)
    assert chem.get_pubchem_smiles_from_cids(["1", 3, "99"]) == {
        "1": "CC(=O)O",
        "3": "CCO",
        "99": None,
    }


def test_pubchem_smiles_lookup_cache_scans_only_new_cids(tmp_path, monkeypatch) -> None:
    chem, scans = _smiles_chem(tmp_path, monkeypatch)
    chem.get_pubchem_smiles_from_cids(["1", "99"])
    chem.get_pubchem_smiles_from_cids(["1", "99"])
    assert scans == [{"1", "99"}]

    out = chem.get_pubchem_smiles_from_cids(["1", "2"])
    assert scans[1:] == [{"2"}]
    assert out == {"1": "CC(=O)O", "2": "C1CCCCC1"}


def _structure_chem(tmp_path: Path, looked_up: list | None = None):
    from tests.utils.test_structure_utils import UNIPROT

    chem = object.__new__(ChemUtils)
    chem.logger = MagicMock()
    chem.MMCIF_PATH = tmp_path
    chem.AF_PDB_PATH = tmp_path / "af"
    chem.AF_PDB_PATH.mkdir()
    (chem.AF_PDB_PATH / "AF-P2-F1-model_v4.pdb").write_text("")
    chem.get_ebi_pdb = lambda ids: None
    chem.get_af_pdb = lambda ids: None

    def uniprot(accs):
        if looked_up is not None:
            looked_up.extend(accs)
        return {a: {"sequence": UNIPROT} for a in accs}

    chem.get_uniprot_acc_json = uniprot
    return chem


def test_row_keeps_every_full_coverage_pdb(tmp_path: Path) -> None:
    from tests.utils.test_structure_utils import UNIPROT, write_cif

    write_cif(tmp_path / "1GOOD.cif", {"AAA": UNIPROT}, {"AAA": "P1"})
    write_cif(tmp_path / "2GOOD.cif", {"A": UNIPROT[:-20]}, {"A": "P1"})  # coverage 0.9
    write_cif(tmp_path / "3HALF.cif", {"A": UNIPROT[:100]}, {"A": "P1"})
    df = pd.DataFrame({"uniprot_id": ["P1"], "pdbs": [["1good", "2GOOD", "3HALF"]]})
    out = _structure_chem(tmp_path).assign_experimental_and_af_pdbs(df)

    assert sorted(out["pdbs"]) == ["1GOOD:P1", "2GOOD:P1"]
    assert set(out["pdb_type"]) == {"experimental"}
    assert out.set_index("pdbs").loc["1GOOD:P1", "pdb_chain"] == "AAA"
    assert out.set_index("pdbs").loc["2GOOD:P1", "pdb_coverage"] == 0.9


def test_row_without_full_coverage_keeps_its_best_partial_pdb(tmp_path: Path) -> None:
    from tests.utils.test_structure_utils import UNIPROT, write_cif

    write_cif(tmp_path / "1LOW.cif", {"A": UNIPROT[:80]}, {"A": "P1"})
    write_cif(tmp_path / "2MID.cif", {"A": UNIPROT[:140]}, {"A": "P1"})
    write_cif(tmp_path / "3TINY.cif", {"A": UNIPROT[:40]}, {"A": "P1"})  # below 0.3
    df = pd.DataFrame({"uniprot_id": ["P1"], "pdbs": [["1LOW", "2MID", "3TINY"]]})
    out = _structure_chem(tmp_path).assign_experimental_and_af_pdbs(df)

    assert out["pdbs"].tolist() == ["2MID:P1"]
    assert out["pdb_coverage"].tolist() == [0.7]


def test_each_row_without_a_chain_gets_exactly_one_fallback_row(tmp_path: Path) -> None:
    from tests.utils.test_structure_utils import UNIPROT, write_cif

    write_cif(tmp_path / "1TINY.cif", {"A": UNIPROT[:40]}, {"A": "P2"})
    write_cif(tmp_path / "2TINY.cif", {"A": UNIPROT[:50]}, {"A": "P2"})
    write_cif(tmp_path / "3GOOD.cif", {"A": UNIPROT}, {"A": "P1"})
    df = pd.DataFrame(
        {
            "uniprot_id": ["P2", "P2", "P1", "P3"],
            "pdbs": [["1TINY", "2TINY"], ["1TINY"], ["3GOOD"], None],
            "value": [1, 2, 3, 4],
        }
    )
    out = _structure_chem(tmp_path).assign_experimental_and_af_pdbs(df)

    assert out.groupby("value").size().tolist() == [1, 1, 1, 1]
    by_value = out.set_index("value")
    assert by_value.loc[1, "pdb_type"] == "predicted" and by_value.loc[2, "pdb_type"] == "predicted"
    assert by_value.loc[3, "pdb_type"] == "experimental"
    assert by_value.loc[4, "pdb_source"] != "PDBe"
    assert "_row_id" not in out.columns


def test_mutant_id_uses_the_base_accession(tmp_path: Path) -> None:
    from tests.utils.test_structure_utils import UNIPROT, write_cif

    write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT}, {"A": "P1"})
    looked_up: list = []
    df = pd.DataFrame({"uniprot_id": ["P1|R2Q"], "pdbs": [["1ABC"]]})
    out = _structure_chem(tmp_path, looked_up).assign_experimental_and_af_pdbs(df)

    assert looked_up == ["P1"]
    assert out["uniprot_id"].tolist() == ["P1|R2Q"]
    assert out["pdbs"].tolist() == ["1ABC:P1"]
    assert out["pdb_chain"].tolist() == ["A"] and out["pdb_type"].tolist() == ["experimental"]


def test_get_html_sends_user_agent(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, text="ok")

    monkeypatch.setattr("src.utils.chem_utils.requests.get", fake_get)
    path = tmp_path / "x.html"
    ChemUtils.get_html(SimpleNamespace(logger=MagicMock()), "http://x", path)
    assert "User-Agent" in calls[0]["headers"]
    assert path.read_text() == "ok"


_AF_P1_ENTRY = {
    "modelEntityId": "AF-P1-F1",
    "latestVersion": 6,
    "pdbUrl": "http://x/AF-P1-F1-model_v6.pdb",
}


def test_get_af_pdb_warns_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", lambda s: None)
    (tmp_path / "pdb").mkdir()
    chem = SimpleNamespace(
        logger=MagicMock(),
        ALPHAFOLD_API_URL="http://x/{id}",
        AF_DOWNLOAD_PATH=tmp_path,
        AF_PDB_PATH=tmp_path / "pdb",
        get_html=lambda url, path, timeout: (
            path.write_text("{}") if path.suffix == ".json" else None
        ),
        load_json=lambda path: [_AF_P1_ENTRY],
        extract_af_bulk=lambda ids, version: set(),
    )
    ChemUtils.get_af_pdb(chem, ["P1"])
    assert not any("Downloaded" in str(c) for c in chem.logger.info.call_args_list)
    assert "was not downloaded" in str(chem.logger.warning.call_args)


def _http_get(monkeypatch, statuses):
    sleeps, calls = [], []
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", sleeps.append)

    def fake_get(url, headers, timeout):
        calls.append(url)
        return SimpleNamespace(
            status_code=statuses[min(len(calls), len(statuses)) - 1],
            text="ok",
            headers={},
        )

    monkeypatch.setattr("src.utils.chem_utils.requests.get", fake_get)
    return calls, sleeps


def test_get_html_waits_and_retries_on_429(tmp_path, monkeypatch) -> None:
    calls, sleeps = _http_get(monkeypatch, [429, 200])
    path = tmp_path / "a.html"
    ChemUtils.get_html(SimpleNamespace(logger=MagicMock()), "http://x", path)
    assert len(calls) == 2
    assert sleeps == [5]
    assert path.read_text() == "ok"


def test_get_html_gives_up_after_repeated_429(tmp_path, monkeypatch) -> None:
    calls, sleeps = _http_get(monkeypatch, [429])
    path = tmp_path / "a.html"
    chem = SimpleNamespace(logger=MagicMock())
    ChemUtils.get_html(chem, "http://x", path)
    assert len(calls) == 5
    assert sleeps == [5, 10, 20, 40]
    assert not path.exists()
    assert "Max retries exceeded" in str(chem.logger.error.call_args)


def test_get_af_pdb_retry_pass_downloads_failed_file(tmp_path, monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", sleeps.append)
    (tmp_path / "pdb").mkdir()
    attempts = []
    api_calls = []

    def get_html(url, path, timeout):
        if path.suffix == ".json":
            api_calls.append(path)
            if len(api_calls) > 2:
                path.write_text("{}")
        else:
            attempts.append(path)
            path.write_text("x")

    chem = SimpleNamespace(
        logger=MagicMock(),
        ALPHAFOLD_API_URL="http://x/{id}",
        AF_DOWNLOAD_PATH=tmp_path,
        AF_PDB_PATH=tmp_path / "pdb",
        get_html=get_html,
        load_json=lambda path: [_AF_P1_ENTRY],
        extract_af_bulk=lambda ids, version: set(),
    )
    ChemUtils.get_af_pdb(chem, ["P1"])
    assert len(api_calls) == 3 and len(attempts) == 1
    assert sleeps and (tmp_path / "pdb" / "AF-P1-F1-model_v6.pdb").exists()
    assert "retry recovered 1 accessions; 0 still failed." in str(
        chem.logger.info.call_args
    )


def _af_chem(tmp_path, entries, files):
    """ChemUtils stand-in; `files` holds the URLs that exist, `entries` maps id to API JSON."""
    (tmp_path / "pdb").mkdir()
    fetched = []

    def get_html(url, path, timeout):
        if path.exists():
            return
        fetched.append(url)
        if path.suffix == ".json":
            acc = path.stem
            if acc in entries:
                path.write_text(json.dumps(entries[acc]))
        elif url in files:
            path.write_text("x")

    chem = SimpleNamespace(
        logger=MagicMock(),
        ALPHAFOLD_API_URL="http://api/{id}",
        AF_DOWNLOAD_PATH=tmp_path,
        AF_PDB_PATH=tmp_path / "pdb",
        get_html=get_html,
        load_json=lambda path: json.loads(path.read_text()),
        extract_af_bulk=lambda ids, version: set(),
    )
    return chem, fetched


def _af_entry(acc, version=7, entity=None):
    return {
        "modelEntityId": entity or f"AF-{acc}-F1",
        "latestVersion": version,
        "pdbUrl": f"http://files/{entity or f'AF-{acc}-F1'}-model_v{version}.pdb",
    }


def test_get_af_pdb_uses_one_api_call_and_direct_urls(tmp_path) -> None:
    accs = ["P3", "P1", "P2"]
    entries = {"P1": [_af_entry("P1")]}
    files = {f"http://files/AF-{a}-F1-model_v7.pdb" for a in accs}
    chem, fetched = _af_chem(tmp_path, entries, files)
    ChemUtils.get_af_pdb(chem, accs)
    assert [u for u in fetched if u.startswith("http://api/")] == ["http://api/P1"]
    assert sorted(f.name for f in (tmp_path / "pdb").glob("*.pdb")) == [
        f"AF-{a}-F1-model_v7.pdb" for a in ["P1", "P2", "P3"]
    ]


def test_get_af_pdb_lookup_skips_cached_accessions(tmp_path) -> None:
    files = {f"http://files/AF-{a}-F1-model_v8.pdb" for a in ["P2", "P3"]}
    chem, fetched = _af_chem(tmp_path, {"P2": [_af_entry("P2", version=8)]}, files)
    (tmp_path / "P1.json").write_text("[]")
    ChemUtils.get_af_pdb(chem, ["P1", "P2", "P3"])
    assert "http://api/P1" not in fetched
    assert fetched.count("http://api/P2") == 1
    assert "http://files/AF-P3-F1-model_v8.pdb" in fetched
    assert (tmp_path / "P1.json").read_text() == "[]"


def test_get_af_pdb_all_cached_makes_no_request(tmp_path) -> None:
    chem, fetched = _af_chem(tmp_path, {}, set())
    chem.extract_af_bulk = MagicMock()
    (tmp_path / "P1.json").write_text("[]")
    (tmp_path / "P2.json").write_text("[]")
    ChemUtils.get_af_pdb(chem, ["P1", "P2"])
    assert not [u for u in fetched if u.startswith("http://api/")]
    chem.extract_af_bulk.assert_not_called()


def test_get_af_pdb_falls_back_to_api_and_picks_canonical(tmp_path) -> None:
    entries = {
        "P1": [_af_entry("P1")],
        "P2": [
            _af_entry("P2", entity="AF-0000000365777370"),
            _af_entry("P2", entity="AF-P2-4-F1"),
            _af_entry("P2", version=8),
        ],
    }
    files = {"http://files/AF-P1-F1-model_v7.pdb", "http://files/AF-P2-F1-model_v8.pdb"}
    chem, fetched = _af_chem(tmp_path, entries, files)
    ChemUtils.get_af_pdb(chem, ["P1", "P2"])
    assert "http://files/AF-P2-F1-model_v7.pdb" in fetched
    assert (tmp_path / "pdb" / "AF-P2-F1-model_v8.pdb").exists()
    assert not list((tmp_path / "pdb").glob("*VR3D*"))
    assert len(list((tmp_path / "pdb").glob("*.pdb"))) == 2


def test_get_af_pdb_warns_without_canonical_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", lambda s: None)
    entries = {
        "P1": [
            _af_entry("P1", entity="AF-0000000365777370"),
            _af_entry("P1", entity="AF-P1-4-F1"),
        ]
    }
    chem, _ = _af_chem(tmp_path, entries, set())
    ChemUtils.get_af_pdb(chem, ["P1"])
    assert not list((tmp_path / "pdb").glob("*.pdb"))
    assert "no canonical model for P1" in str(chem.logger.warning.call_args)


def _bulk_chem(tmp_path, download=None):
    """Stand-in with a two-member Swiss-Prot tar of gzip PDBs already in place."""
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    (tmp_path / "pdb").mkdir(exist_ok=True)
    with tarfile.open(bulk / "swissprot_pdb_v7.tar", "w") as tar:
        for acc in ["P1", "P9"]:
            data = gzip.compress(f"MODEL {acc}".encode())
            info = tarfile.TarInfo(f"AF-{acc}-F1-model_v7.pdb.gz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return SimpleNamespace(
        logger=MagicMock(),
        ALPHAFOLD_SWISSPROT_URL="http://ftp/swissprot_pdb_v{version}.tar",
        AF_BULK_PATH=bulk,
        AF_PDB_PATH=tmp_path / "pdb",
        download_af_archive=download or MagicMock(),
    )


def test_extract_af_bulk_extracts_and_indexes(tmp_path) -> None:
    chem = _bulk_chem(tmp_path)
    assert ChemUtils.extract_af_bulk(chem, {"P1", "P2"}, 7) == {"P1"}
    assert (tmp_path / "pdb" / "AF-P1-F1-model_v7.pdb").read_text() == "MODEL P1"
    assert not (tmp_path / "pdb" / "AF-P2-F1-model_v7.pdb").exists()
    index = json.loads((tmp_path / "bulk" / "swissprot_pdb_v7.tar.index.json").read_text())
    assert sorted(index) == ["P1", "P9"]
    chem.download_af_archive.assert_not_called()
    assert "Extracted 1 AlphaFold models" in str(chem.logger.info.call_args)


def test_extract_af_bulk_reuses_saved_index(tmp_path, monkeypatch) -> None:
    chem = _bulk_chem(tmp_path)
    ChemUtils.extract_af_bulk(chem, {"P1"}, 7)
    opened = MagicMock(side_effect=AssertionError)
    monkeypatch.setattr("src.utils.chem_utils.tarfile.open", opened)
    assert ChemUtils.extract_af_bulk(chem, {"P9"}, 7) == {"P9"}
    opened.assert_not_called()


def test_extract_af_bulk_downloads_when_control_file_present(tmp_path) -> None:
    chem = _bulk_chem(tmp_path)
    (tmp_path / "bulk" / "swissprot_pdb_v7.tar.aria2").write_text("")
    ChemUtils.extract_af_bulk(chem, {"P1"}, 7)
    chem.download_af_archive.assert_called_once_with(
        "http://ftp/swissprot_pdb_v7.tar", tmp_path / "bulk" / "swissprot_pdb_v7.tar"
    )


def test_extract_af_bulk_download_failure_falls_back(tmp_path) -> None:
    chem = _bulk_chem(tmp_path, MagicMock(side_effect=OSError("down")))
    (tmp_path / "bulk" / "swissprot_pdb_v7.tar").unlink()
    assert ChemUtils.extract_af_bulk(chem, {"P1"}, 7) == set()
    assert "download failed" in str(chem.logger.warning.call_args)


def test_get_af_pdb_sends_only_unextracted_to_api(tmp_path) -> None:
    entries = {"P1": [_af_entry("P1")]}
    files = {"http://files/AF-P2-F1-model_v7.pdb"}
    chem, fetched = _af_chem(tmp_path, entries, files)
    bulk = _bulk_chem(tmp_path)
    chem.__dict__.update(
        {k: v for k, v in vars(bulk).items() if k not in ("logger", "AF_PDB_PATH")}
    )
    chem.extract_af_bulk = lambda ids, version: ChemUtils.extract_af_bulk(
        chem, ids, version
    )
    ChemUtils.get_af_pdb(chem, ["P1", "P2"])
    assert (tmp_path / "pdb" / "AF-P1-F1-model_v7.pdb").read_text() == "MODEL P1"
    assert "http://files/AF-P2-F1-model_v7.pdb" in fetched
    assert "http://files/AF-P1-F1-model_v7.pdb" not in fetched


def test_get_af_pdb_reports_failed_api_request(tmp_path) -> None:
    chem, _ = _af_chem(tmp_path, {}, set())
    ChemUtils.get_af_pdb(chem, ["P1"])
    assert "API request failed for P1" in str(chem.logger.warning.call_args)
    assert "no canonical" not in str(chem.logger.warning.call_args)


def _af_requests(fetched, acc):
    return [u for u in fetched if u.endswith(f"/{acc}") or f"-{acc}-" in u]


def test_get_af_pdb_caches_api_404_as_empty_entry(tmp_path) -> None:
    entries = {a: [_af_entry(a)] for a in ["P0", "P1"]}
    files = {f"http://files/AF-{a}-F1-model_v7.pdb" for a in ["P0", "P1"]}
    chem, fetched = _af_chem(tmp_path, entries, files)
    base = chem.get_html

    def get_html(url, path, timeout):
        if path.stem == "P2" and not path.exists():
            fetched.append(url)
            return 404
        return base(url, path, timeout)

    chem.get_html = get_html
    ChemUtils.get_af_pdb(chem, ["P1", "P2"])
    assert (tmp_path / "P2.json").read_text() == "[]"
    assert "AlphaFold has no entry for P2." in str(chem.logger.warning.call_args)
    fetched.clear()
    chem.logger.reset_mock()
    ChemUtils.get_af_pdb(chem, ["P0", "P2"])
    assert _af_requests(fetched, "P2") == []
    assert "no entry for P2" in str(chem.logger.warning.call_args)


def test_get_af_pdb_cached_non_canonical_json_makes_no_request(tmp_path) -> None:
    chem, fetched = _af_chem(
        tmp_path, {"P0": [_af_entry("P0")]}, {"http://files/AF-P0-F1-model_v7.pdb"}
    )
    (tmp_path / "P1.json").write_text(
        json.dumps([_af_entry("P1", entity="AF-P1-4-F1")])
    )
    ChemUtils.get_af_pdb(chem, ["P0", "P1"])  # P0 supplies the URL template
    assert _af_requests(fetched, "P1") == []
    assert "no canonical model for P1" in str(chem.logger.warning.call_args)


def test_get_af_pdb_retry_skips_accessions_with_cached_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", lambda s: None)
    chem, fetched = _af_chem(tmp_path, {"P2": []}, set())
    ChemUtils.get_af_pdb(chem, ["P1", "P2"])
    assert len(_af_requests(fetched, "P1")) == 3
    assert len(_af_requests(fetched, "P2")) == 1
    assert "Retrying 1 failed" in str(chem.logger.info.call_args_list)


def test_get_af_pdb_lists_download_folder_a_bounded_number_of_times(tmp_path, monkeypatch) -> None:
    accs = [f"P{i}" for i in range(20)]
    files = {f"http://files/AF-{a}-F1-model_v7.pdb" for a in accs}
    chem, _ = _af_chem(tmp_path, {"P0": [_af_entry("P0")]}, files)
    listings = []
    real_glob = Path.glob

    def counting_glob(self, pattern):
        listings.append(self)
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", counting_glob)
    ChemUtils.get_af_pdb(chem, accs)
    assert len([p for p in listings if p == tmp_path / "pdb"]) == 2


def test_pubmed_batch_is_retried_after_a_request_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PUBMED_EMAIL", "a@b.c")
    monkeypatch.setenv("PUBMED_API_KEY", "key")
    monkeypatch.setattr("src.utils.chem_utils.time.sleep", lambda s: None)
    replies = [
        requests.ConnectionError("Response ended prematurely"),
        SimpleNamespace(status_code=200, text="<xml/>"),
    ]

    def fake_post(url, data, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("src.utils.chem_utils.requests.post", fake_post)
    chem = object.__new__(ChemUtils)
    chem.logger = MagicMock()
    chem.PUBMED_ENTREZ_EFETCH_URL = "https://pubmed.test"
    chem.REFERENCE_DOWNLOAD_PATH = tmp_path
    chem.download_pubmed_abstracts_parallel(["1", "2"])
    assert (tmp_path / "1 - 2.xml").read_text() == "<xml/>"


def test_undirected_reaction_key_ignores_direction() -> None:
    from src.utils.chem_utils import undirected_reaction_key

    assert undirected_reaction_key("A>>B") == undirected_reaction_key("B>>A")
    assert undirected_reaction_key("CC") == "CC"
    assert undirected_reaction_key(None) is None
