from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from src.data.components.generators import atom_mapped_reaction
from src.data.components.generators.atom_mapped_reaction import (
    AtomMappedReactionGenerator,
    AtomMappingConfig,
    _postprocess_init,
    _postprocess_one,
    _preprocess_init,
    _preprocess_one,
    _unique_non_null_reactions,
)


@pytest.fixture
def fake_localmapper(monkeypatch, tmp_path):
    """Stand in for the localmapper package: a featurizer tuple and an accepted-templates file."""
    package, utils = ModuleType("localmapper"), ModuleType("localmapper.utils")
    utils.init_featurizer = lambda: (None, None, lambda mol: mol)
    package.utils = utils
    monkeypatch.setitem(sys.modules, "localmapper", package)
    monkeypatch.setitem(sys.modules, "localmapper.utils", utils)
    (tmp_path / "data").mkdir()
    with (tmp_path / "data" / "templates_202403.pkl").open("wb") as handle:
        pickle.dump(set(), handle)
    monkeypatch.setattr("importlib.resources.files", lambda name: tmp_path)
    for name in ("_PREPROCESS_GRAPH", "_POST_ACCEPTED_TEMPLATES", "_POST_MODEL_VERSION"):
        monkeypatch.setattr(atom_mapped_reaction, name, None)


def test_unique_reaction_extraction_is_stable_and_excludes_nulls():
    frame = pd.DataFrame({"rxn_smiles": ["B>>C", None, "A>>B", "B>>C", float("nan")]})
    assert _unique_non_null_reactions(frame)["rxn_smiles"].tolist() == ["B>>C", "A>>B"]


def test_resume_fanout_preserves_duplicate_rows_and_nulls(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    frame = pd.DataFrame({"rxn_smiles": ["A>>B", "A>>B", None]})
    frame.to_parquet(input_path, index=False)
    pd.DataFrame(
        {
            "rxn_smiles": ["A>>B"],
            "atom_mapped_rxn_smiles": ["[A:1]>>[B:1]"],
            "atom_mapping_template": [None],
            "atom_mapping_confident": [True],
            "atom_mapping_source": ["localmapper"],
            "atom_mapping_status": ["mapped"],
            "atom_mapping_error": [None],
            "atom_mapping_input_hash": ["hash"],
            "atom_mapping_model_version": ["202403"],
        }
    ).to_parquet(output_path, index=False)

    output = AtomMappedReactionGenerator(options=AtomMappingConfig(gpu_ids=(0,))).generate(
        input_path, output_path, resume=True
    )
    assert output["rxn_smiles"].tolist() == ["A>>B", "A>>B", None]
    assert output["atom_mapped_rxn_smiles"].tolist()[:2] == ["[A:1]>>[B:1]"] * 2
    assert output["atom_mapping_status"].tolist() == ["mapped", "mapped", "not_applicable"]


def test_resume_fanout_ignores_string_dtype_changes(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    frame = pd.DataFrame({"rxn_smiles": pd.Series(["A>>B", "C>>D"], dtype="string")})
    frame.to_parquet(input_path, index=False)
    pd.DataFrame(
        {
            "rxn_smiles": ["A>>B", "C>>D"],
            "atom_mapping_status": ["mapped", "mapped"],
            # Rows are reused only when their model version matches.
            "atom_mapping_model_version": ["202403", "202403"],
        }
    ).to_parquet(output_path, index=False)

    output = AtomMappedReactionGenerator(options=AtomMappingConfig(gpu_ids=(0,))).generate(
        input_path, output_path, resume=True
    )

    assert output["rxn_smiles"].tolist() == ["A>>B", "C>>D"]


def test_preprocess_preserves_parseable_input_mapping(fake_localmapper):
    _preprocess_init()
    rxn = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
    item = _preprocess_one((7, rxn))
    assert item["kind"] == "result"
    assert item["result"]["atom_mapped_rxn_smiles"] == rxn
    assert item["result"]["atom_mapping_source"] == "input"
    assert item["result"]["atom_mapping_status"] == "preserved"


def test_preprocess_reports_rdkit_failure(fake_localmapper):
    _preprocess_init()
    item = _preprocess_one((3, "not-a-smiles>>still-not-a-smiles"))
    assert item["kind"] == "result"
    assert item["result"]["atom_mapping_status"] == "failed"
    assert "RDKit" in item["result"]["atom_mapping_error"]


def test_postprocess_returns_a_preprocessed_result_unchanged(fake_localmapper):
    _postprocess_init("202403")
    result = {"_atom_mapping_row_id": 0, "rxn_smiles": "A>>B", "atom_mapping_status": "preserved"}
    assert _postprocess_one({"kind": "result", "result": result}) == result


def test_postprocess_sets_the_model_version_on_an_inference_error(fake_localmapper):
    _postprocess_init("202403")
    item = {"kind": "inference_error", "row_id": 0, "rxn": "A>>B", "error": "boom"}
    result = _postprocess_one(item)
    assert result["atom_mapping_model_version"] == "202403"
    assert (result["atom_mapping_status"], result["atom_mapping_error"]) == ("failed", "boom")


def test_resume_loads_the_partial_checkpoint(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "in.parquet", tmp_path / "out.parquet"
    pd.DataFrame({"rxn_smiles": ["A>>B"]}).to_parquet(input_path, index=False)
    pd.DataFrame(
        {
            "rxn_smiles": ["A>>B"],
            "atom_mapped_rxn_smiles": ["[A:1]>>[B:1]"],
            "atom_mapping_status": ["mapped"],
            "atom_mapping_model_version": ["202403"],
        }
    ).to_parquet(output_path.with_suffix(".partial.parquet"), index=False)
    out = AtomMappedReactionGenerator(options=AtomMappingConfig(gpu_ids=(0,))).generate(
        input_path, output_path, resume=True
    )
    assert out["atom_mapping_status"].tolist() == ["mapped"]


def test_preserved_rows_are_reused_without_a_model_version() -> None:
    from src.data.components.generators.atom_mapped_reaction import _is_reusable_row

    row = {"atom_mapping_status": "preserved", "atom_mapping_model_version": None}
    assert _is_reusable_row(row, "202403")
    assert _is_reusable_row(
        {"atom_mapping_status": "mapped", "atom_mapping_model_version": "202403"}, "202403"
    )
    assert not _is_reusable_row(
        {"atom_mapping_status": "mapped", "atom_mapping_model_version": "old"}, "202403"
    )
    assert not _is_reusable_row(
        {"atom_mapping_status": "failed", "atom_mapping_model_version": "202403"}, "202403"
    )


@pytest.fixture
def fake_mapper(monkeypatch, fake_localmapper):
    mapper = ModuleType("localmapper.mapper")
    monkeypatch.setitem(sys.modules, "localmapper.mapper", mapper)
    return mapper


def test_postprocess_reports_a_reaction_over_the_time_limit_as_failed(fake_mapper, monkeypatch):
    def slow_swallowing_mapper(rxn, prediction, neighbor_weight=10):
        try:
            while True:
                time.sleep(0.01)
        except:  # noqa: E722  localmapper swallows every exception in template extraction
            return {"mapped_rxn": rxn, "template": None}

    fake_mapper.prediction2map = slow_swallowing_mapper
    monkeypatch.setattr(atom_mapped_reaction, "MAX_ROW_SECONDS", 0.3)
    monkeypatch.setattr(atom_mapped_reaction, "_GUARD_INTERVAL_SECONDS", 0.05)
    _postprocess_init("202403")
    result = _postprocess_one(
        {"kind": "prediction", "row_id": 0, "rxn": "A>>B", "prediction": None}
    )
    assert result["atom_mapping_status"] == "failed"
    assert result["atom_mapping_error"].startswith("RunawayReaction")
    assert result["atom_mapped_rxn_smiles"] is None


def test_postprocess_maps_a_normal_reaction_under_the_guard(fake_mapper):
    fake_mapper.prediction2map = lambda rxn, prediction, neighbor_weight=10: {
        "mapped_rxn": "[A:1]>>[B:1]",
        "template": "T",
    }
    _postprocess_init("202403")
    result = _postprocess_one(
        {"kind": "prediction", "row_id": 0, "rxn": "A>>B", "prediction": None}
    )
    assert result["atom_mapping_status"] == "mapped"
    assert result["atom_mapped_rxn_smiles"] == "[A:1]>>[B:1]"


def test_postprocess_maps_a_normal_reaction_in_a_worker_that_is_already_large(
    fake_mapper, monkeypatch
):
    fake_mapper.prediction2map = lambda rxn, prediction, neighbor_weight=10: {
        "mapped_rxn": "[A:1]>>[B:1]",
        "template": "T",
    }
    # The worker starts above the limit; only growth during the row counts.
    monkeypatch.setattr(atom_mapped_reaction, "MAX_ROW_RSS_BYTES", 1)
    monkeypatch.setattr(atom_mapped_reaction, "_GUARD_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(atom_mapped_reaction, "_rss_bytes", lambda: 10 << 30)
    _postprocess_init("202403")
    result = _postprocess_one(
        {"kind": "prediction", "row_id": 0, "rxn": "A>>B", "prediction": None}
    )
    assert result["atom_mapping_status"] == "mapped"
