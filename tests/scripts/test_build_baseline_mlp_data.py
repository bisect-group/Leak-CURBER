from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.helpers.builders import load_script

builder = load_script("build_baseline_mlp_data")


def _dirs(tmp_path: Path, files: list[str]) -> dict[str, Path]:
    dirs = {name: tmp_path / name for name in ("af", "esmfold2", "esm", "mmcif")}
    for directory in dirs.values():
        directory.mkdir()
    for name in files:
        (tmp_path / name).write_text("")
    return {
        "af_dir": dirs["af"],
        "esmfold2_dir": dirs["esmfold2"],
        "esm_dir": dirs["esm"],
        "mmcif_dir": dirs["mmcif"],
    }


def _candidates(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "row_key": pl.String,
            "pdbs": pl.String,
            "pdb_type": pl.String,
            "pdb_chain": pl.String,
            "pdb_coverage": pl.Float64,
        },
        orient="row",
    )


def test_one_structure_per_sequence_in_source_order(tmp_path: Path) -> None:
    dirs = _dirs(
        tmp_path,
        [
            "mmcif/1AAA_updated.cif",
            "mmcif/2BBB.cif",
            "af/AF-A2-F1-model_v3.pdb",
            "af/AF-A2-F1-model_v4.pdb",
            "esmfold2/ESMFold2-E3.pdb",
            "esm/ESM3-open-small-E3.pdb",
            "esm/ESM3-open-small-S4.pdb",
        ],
    )
    candidates = _candidates(
        [
            ("exp", "1AAA:P1", "experimental", "A", 0.9),
            ("exp", "2BBB:P1", "experimental", "B", 0.95),
            ("exp", "A2", None, None, None),
            ("af", "A2", "predicted", None, None),
            ("fold", "E3", None, None, None),
            ("esm", "S4", None, None, None),
            ("none", "N5", None, None, None),
            ("nofile", "9ZZZ:P6", "experimental", "A", 1.0),
        ]
    )
    chosen = builder.choose_structures(candidates, **dirs).sort("row_key")
    assert chosen.rows() == [
        ("af", "A2|alphafold", "alphafold", None),
        ("esm", "S4|esm", "esm3", None),
        ("exp", "2BBB:P1|experimental", "experimental", "B"),
        ("fold", "E3|esmfold2", "esmfold2", None),
    ]


def test_equal_coverage_picks_the_smallest_structure_id(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, ["mmcif/3CCC.cif", "mmcif/1AAA.cif"])
    candidates = _candidates(
        [
            ("s", "3CCC:P1", "experimental", "A", 1.0),
            ("s", "1AAA:P1", "experimental", "C", 1.0),
        ]
    )
    assert builder.choose_structures(candidates, **dirs)["pdb_id"].to_list() == ["1AAA:P1|experimental"]


def test_experimental_beats_predicted_even_with_low_coverage(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, ["mmcif/1AAA.cif", "af/AF-A2-F1-model_v1.pdb"])
    candidates = _candidates(
        [("s", "1AAA:P1", "experimental", "A", 0.35), ("s", "A2", "predicted", None, None)]
    )
    assert builder.choose_structures(candidates, **dirs)["pdb_id"].to_list() == ["1AAA:P1|experimental"]


def test_rows_of_one_sequence_choose_from_their_own_lists(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, ["mmcif/1AAA.cif", "mmcif/2BBB.cif"])
    candidates = _candidates(
        [
            ("r1", "1AAA:P1", "experimental", "A", 0.9),
            ("r2", "2BBB:P1", "experimental", "B", 0.85),
        ]
    )
    chosen = builder.choose_structures(candidates, **dirs).sort("row_key")
    assert chosen["pdb_id"].to_list() == ["1AAA:P1|experimental", "2BBB:P1|experimental"]


def test_alphafold_isoform_models_are_not_used(tmp_path: Path) -> None:
    dirs = _dirs(
        tmp_path,
        [
            "af/AF-Q1-2-F1-model_v6.pdb",
            "af/AF-Q2-F1-model_v4.pdb",
            "esmfold2/ESMFold2-Q1.pdb",
        ],
    )
    candidates = _candidates([("r1", "Q1", None, None, None), ("r2", "Q2", None, None, None)])
    chosen = builder.choose_structures(candidates, **dirs).sort("row_key")
    assert chosen["pdb_id"].to_list() == ["Q1|esmfold2", "Q2|alphafold"]


def test_pdb_ids_is_a_one_element_list_or_empty() -> None:
    frame = pl.DataFrame({"a": ["x", "x", "y", None], "y": [1.0, 2.0, 3.0, 4.0]})
    hashes = frame.select(pl.struct(["a", "y"]).hash())["a"]
    selection = pl.DataFrame({"row_key": [hashes[0]], "pdb_id": ["A2|alphafold"]})
    out = builder.with_pdb_ids(frame, selection, ["a", "y"])
    assert out["pdb_ids"].to_list() == [["A2|alphafold"], [], [], []]
    assert out.columns == ["a", "y", "pdb_ids"]


def _write_source(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    source = tmp_path / "source"
    (source / "random_splits_grouped_sequence").mkdir(parents=True)
    for part in builder.PARTS:
        pl.DataFrame({"sequence": ["S1", "S1"], "y": [1.0, 2.0], "pdbs": [["a"], ["b"]]}).write_parquet(
            source / "random_splits_grouped_sequence" / f"{part}.parquet"
        )
    source_3d = tmp_path / "source_3d.parquet"
    pl.DataFrame(
        rows,
        schema_overrides={c: pl.String for c in ("pdb_type", "pdb_chain", "pdb_source")}
        | {"pdb_coverage": pl.Float64},
    ).write_parquet(source_3d)
    return source, source_3d


def _row(y: float, pdbs: str, pdb_type: str | None) -> dict:
    return {
        "sequence": "S1", "y": y, "pdbs": pdbs, "pdb_type": pdb_type,
        "pdb_chain": None, "pdb_coverage": None, "pdb_source": None,
    }


def test_build_dataset_matches_rows_one_to_one_and_keeps_existing_outputs(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, ["af/AF-A2-F1-model_v1.pdb", "esm/ESM3-open-small-E3.pdb"])
    source, source_3d = _write_source(tmp_path, [_row(1.0, "A2", "predicted"), _row(2.0, "E3", None)])
    out = tmp_path / "out"
    splits = ["random_splits_grouped_sequence"]

    builder.build_dataset(source, source_3d, out, splits, False, dirs)
    train = out / splits[0] / "train.parquet"
    assert pl.read_parquet(train)["pdb_ids"].to_list() == [["A2|alphafold"], ["E3|esm"]]

    stamp = train.stat().st_mtime_ns
    builder.build_dataset(source, source_3d, out, splits, False, dirs)
    assert train.stat().st_mtime_ns == stamp


def test_build_dataset_rejects_rows_without_a_3d_row(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, [])
    source, source_3d = _write_source(tmp_path, [_row(1.0, "A2", None)])
    with pytest.raises(ValueError, match="no row"):
        builder.build_dataset(source, source_3d, tmp_path / "out", ["random_splits_grouped_sequence"], False, dirs)


def test_locate_structures_keeps_isoform_models_by_default(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, ["af/AF-Q1-2-F1-model_v6.pdb"])
    assert "Q1" in builder.locate_structures([], ["Q1"], **dirs)
    assert "Q1" not in builder.locate_structures([], ["Q1"], canonical_alphafold_only=True, **dirs)


def test_predicted_fallback_order_is_alphafold_esmfold2_esm3(tmp_path: Path) -> None:
    dirs = _dirs(
        tmp_path,
        [
            "af/AF-Q1-F1-model_v4.pdb",
            "esmfold2/ESMFold2-Q1.pdb",
            "esm/ESM3-open-small-Q1.pdb",
            "esmfold2/ESMFold2-Q2.pdb",
            "esm/ESM3-open-small-Q2.pdb",
            "esm/ESM3-open-small-Q3.pdb",
        ],
    )
    candidates = _candidates([(f"r{i}", f"Q{i}", "predicted", None, None) for i in (1, 2, 3, 4)])
    chosen = builder.choose_structures(candidates, **dirs).sort("row_key")
    assert chosen["pdb_id"].to_list() == ["Q1|alphafold", "Q2|esmfold2", "Q3|esm"]


def test_row_whose_3d_candidates_all_lack_files_gets_no_structure(tmp_path: Path) -> None:
    dirs = _dirs(tmp_path, ["af/AF-A2-F1-model_v1.pdb"])
    source, source_3d = _write_source(tmp_path, [_row(1.0, "A2", "predicted"), _row(2.0, "9ZZZ", "predicted")])
    out = tmp_path / "out"
    splits = ["random_splits_grouped_sequence"]
    builder.build_dataset(source, source_3d, out, splits, False, dirs)
    assert pl.read_parquet(out / splits[0] / "train.parquet")["pdb_ids"].to_list() == [["A2|alphafold"], []]
