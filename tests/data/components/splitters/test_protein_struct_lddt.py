from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from src.data.components.splitters.protein_struct_lddt import (
    mean_plddt,
    FOLDSEEK_SEARCH_ARGS,
    ProteinStructMaxLDDTSimilaritySplitter,
)
from tests.helpers.builders import SilentLogger


def _splitter_for_source(source_path: Path):
    splitter = object.__new__(ProteinStructMaxLDDTSimilaritySplitter)
    splitter.MIN_MEAN_PLDDT = 70.0
    splitter.DROP_TO_1D_DATASET_PATH = source_path
    splitter.logger = SilentLogger()
    return splitter


def test_companion_1d_source_preserves_rows_values_and_schema(tmp_path: Path) -> None:
    source_path = tmp_path / "source_1d.parquet"
    source = pl.DataFrame(
        {
            "sequence": ["train", "val", "test", "train", None],
            "value": [1.0, 2.0, 3.0, 9.0, 4.0],
            "uniprot_date": [
                datetime(2020, 1, 1),
                None,
                datetime(2020, 1, 3),
                datetime(2020, 1, 1),
                None,
            ],
            "pdbs": [["1abc"], ["2abc"], ["3abc"], ["1abc"], None],
        }
    )
    source.write_parquet(source_path)

    splits = _splitter_for_source(source_path)._materialize_companion_1d_splits(
        train_sequences={"train"}, val_sequences={"val"}, test_sequences={"test"}
    )

    assert sum(frame.height for frame in splits.values()) == source.height
    assert splits["train"].to_dicts() == [
        source.row(0, named=True),
        source.row(3, named=True),
        source.row(4, named=True),
    ]
    assert splits["val"].to_dicts() == [source.row(1, named=True)]
    assert splits["test"].to_dicts() == [source.row(2, named=True)]
    assert all(frame.schema == source.schema for frame in splits.values())
    assert splits["train"].get_column("uniprot_date").dtype == pl.Datetime


def test_companion_1d_source_reports_missing_sequence_coverage(tmp_path: Path) -> None:
    source_path = tmp_path / "source_1d.parquet"
    pl.DataFrame({"sequence": ["known", "missing"]}).write_parquet(source_path)

    with pytest.raises(ValueError, match="1 sequences, 1 rows.*missing"):
        _splitter_for_source(source_path)._materialize_companion_1d_splits(
            train_sequences={"known"}, val_sequences=set(), test_sequences=set()
        )


def test_drop_to_1d_split_uses_every_companion_source_row(tmp_path: Path) -> None:
    source_path = tmp_path / "source_1d.parquet"
    source = pl.DataFrame(
        {
            "sequence": ["a", "a", "b", "c", None],
            "value": [1.0, 9.0, 2.0, 3.0, 4.0],
        }
    )
    source.write_parquet(source_path)
    splitter = _splitter_for_source(source_path)
    splitter.DROP_TO_1D = True
    splitter.TEST_FRAC = splitter.VAL_FRAC = 0.4
    splitter.compute_max_lddt_to_dataset = lambda: None
    splitter._get_grouped_dataset_df = lambda: pl.DataFrame(
        {
            "sequence": ["a", "b", "c"],
            "lddt": [0.05, 0.1, 0.2],
            "has_structure": [True, True, True],
            "has_any_structure": [True, True, True],
        }
    )

    splits, _ = splitter.dissimilarity_based_split(0.15)

    assert sum(frame.height for frame in splits.values()) == source.height
    assert splits["test"].get_column("value").to_list() == [1.0, 9.0]
    assert splits["val"].get_column("value").to_list() == [2.0]
    assert splits["train"].get_column("value").to_list() == [3.0, 4.0]


def _lddt_splitter() -> ProteinStructMaxLDDTSimilaritySplitter:
    splitter = object.__new__(ProteinStructMaxLDDTSimilaritySplitter)
    splitter.logger = MagicMock()
    splitter.GPU = False
    splitter.TOOL_VERSION = "v1"
    splitter.MIN_MEAN_PLDDT = 70.0
    return splitter


def test_lddt_is_symmetric_and_missing_structure_is_nan(tmp_path: Path) -> None:
    splitter = _lddt_splitter()
    splitter.CACHE_OVERWRITE = True
    splitter.unique_structures = ["p1", "p2", "p3", "p4"]
    splitter.MAX_SIM_CACHE_PATH = tmp_path / "lddt.tsv"
    splitter._max_lddt_array = splitter._max_lddt_to_dataset = None
    splitter.full_dataset_df = pl.DataFrame({"structure_id": ["p1"], "sequence": ["S1"]})
    splitter._generate_pdb_paths = lambda df: {"p1": 1, "p2": 1, "p4": 1}  # p3 has no file
    splitter._pdb_file_stats = lambda: []
    splitter._selected_chains = lambda: []
    splitter._foldseek_all_vs_all = lambda: pl.LazyFrame(
        {"query": ["p1"], "target": ["p2"], "lddt": [0.8]}
    )
    values = splitter.compute_max_lddt_to_dataset()
    # p2 is only a target; p4 has a structure but no hit (stays 0.0); p3 is unknown (NaN).
    assert values[:2].tolist() == [0.8, 0.8]
    assert values[3] == 0.0
    assert np.isnan(values[2])

    # The NaN survives a warm-cache reload.
    warm = _lddt_splitter()
    warm.CACHE_OVERWRITE = False
    warm.unique_structures = splitter.unique_structures
    warm.full_dataset_df = splitter.full_dataset_df
    warm.MAX_SIM_CACHE_PATH = splitter.MAX_SIM_CACHE_PATH
    warm._pdb_file_stats = lambda: []
    warm._selected_chains = lambda: []
    warm._max_lddt_array = warm._max_lddt_to_dataset = None
    reloaded = warm.compute_max_lddt_to_dataset()
    assert np.isnan(reloaded[2]) and reloaded[0] == 0.8


def test_sequences_without_structure_or_hit_always_go_to_train(tmp_path: Path) -> None:
    source_path = tmp_path / "source_1d.parquet"
    pl.DataFrame({"sequence": list("abcde"), "value": [1.0, 2.0, 3.0, 4.0, 5.0]}).write_parquet(
        source_path
    )
    splitter = _lddt_splitter()
    splitter.DROP_TO_1D, splitter.DROP_TO_1D_DATASET_PATH = True, source_path
    splitter.TEST_FRAC, splitter.VAL_FRAC = 0.5, 0.2
    splitter.compute_max_lddt_to_dataset = lambda: None
    splitter._get_grouped_dataset_df = lambda: pl.DataFrame(
        {
            "sequence": list("abcde"),
            "lddt": [0.0, 0.1, 0.2, 0.9, 0.0],
            # "a" has no structure and "e" has a structure but no hit; both score as most dissimilar.
            "has_structure": [False, True, True, True, True],
            "has_any_structure": [False, True, True, True, True],
        }
    )
    splits, _ = splitter.dissimilarity_based_split(0.15)
    assert splits["test"].get_column("sequence").to_list() == ["b"]
    assert splits["val"].get_column("sequence").to_list() == ["c"]
    assert sorted(splits["train"].get_column("sequence").to_list()) == ["a", "d", "e"]
    logged = " ".join(str(c.args[0]) for c in splitter.logger.info.call_args_list)
    assert "without a structure (forced to train): 1" in logged
    assert "with no Foldseek hit (forced to train): 1" in logged


def test_lddt_fingerprint_changes_when_a_structure_file_changes(tmp_path: Path) -> None:
    pdb = tmp_path / "p1.pdb"
    pdb.write_text("ATOM")
    splitter = _lddt_splitter()
    splitter.unique_structures, splitter.full_dataset_df = ["p1"], None
    splitter._selected_chains = lambda: []
    splitter._pdb_source_paths = lambda df: ({"p1": pdb}, {}, {})
    before = splitter._similarity_fingerprint()
    splitter._pdb_stats = None
    assert splitter._similarity_fingerprint() == before
    pdb.write_text("ATOM CHANGED")
    splitter._pdb_stats = None
    assert splitter._similarity_fingerprint() != before


def test_lddt_fingerprint_changes_with_the_chain_selection() -> None:
    splitter = _lddt_splitter()
    splitter.unique_structures = ["1ABC:P1"]
    splitter._pdb_stats = []

    def fingerprint(chain: str) -> str:
        splitter.full_dataset_df = pl.DataFrame(
            {"pdb_type": ["experimental"], "structure_id": ["1ABC:P1"], "pdb_chain": [chain]}
        )
        return splitter._similarity_fingerprint()

    assert fingerprint("A") == fingerprint("A")
    assert fingerprint("A") != fingerprint("B")


def _hit_splitter() -> ProteinStructMaxLDDTSimilaritySplitter:
    splitter = _lddt_splitter()
    # 1ABC holds P1 (chain AAA) and P2 (chain B); chain C of 1ABC is not selected.
    splitter.full_dataset_df = pl.DataFrame(
        {
            "pdb_type": ["experimental", "experimental", "predicted", None],
            "structure_id": ["1ABC:P1", "1ABC:P2", "P3", "MONOMER-1|Akk_muc"],
            "pdbs": ["1ABC:P1", "1ABC:P2", "P3", "MONOMER-1|Akk_muc"],
            "pdb_chain": ["AAA", "B", None, None],
        }
    )
    splitter._pdb_paths = dict.fromkeys(splitter.full_dataset_df["structure_id"], Path("f"))
    splitter._low_confidence_structures = lambda: set()
    return splitter


def test_foldseek_hits_map_to_structures_and_drop_other_chains(tmp_path: Path) -> None:
    tsv = tmp_path / "hits.tsv"
    rows = [
        ("1ABC_AAA", "1ABC_B", "0.9"),  # two selected chains of one entry
        ("1ABC_AAA", "1ABC_C", "0.8"),  # unselected chain
        ("1ABC_B", "P3_A", "0.7"),  # accession with the chain suffix
        ("P3", "MONOMER-1|Akk_muc_A", "0.6"),  # underscores in the accession, no extension
        ("P3.pdb", "MONOMER-1|Akk_muc.pdb_A", "0.5"),  # extension spellings
        ("P3_A", "OTHER_A", "0.4"),  # file that is not in the dataset
    ]
    tsv.write_text("".join("\t".join(row) + "\n" for row in rows))

    out = _hit_splitter()._read_foldseek_all_vs_all_tsv(tsv, raw_format=True).collect()

    assert list(zip(out["query"], out["target"], out["lddt"])) == [
        ("1ABC:P1", "1ABC:P2", 0.9),
        ("1ABC:P2", "P3", 0.7),
        ("P3", "MONOMER-1|Akk_muc", 0.6),
        ("P3", "MONOMER-1|Akk_muc", 0.5),
    ]
    # The normalized cache holds structure ids and reads back unchanged.
    normalized = tmp_path / "normalized.tsv"
    out.write_csv(normalized, separator="\t")
    assert _hit_splitter()._read_foldseek_all_vs_all_tsv(normalized).collect().equals(out)


def test_symlinks_share_one_cif_per_entry(tmp_path: Path) -> None:
    cif, pdb = tmp_path / "1ABC_updated.cif", tmp_path / "AF-P3.pdb"
    cif.write_text("")
    pdb.write_text("")
    splitter = _lddt_splitter()
    splitter.ALL_PDB_SYMLINK_PATH = tmp_path / "links"
    splitter.ALL_PDB_SYMLINK_PATH.mkdir(exist_ok=True)
    splitter.ALL_VS_ALL_CACHE_PATH = tmp_path / "cache.tsv"
    paths: dict = {}
    for sid, source in (("1ABC:P1", cif), ("1ABC:P2", cif), ("P3", pdb)):
        splitter._create_pdb_symlink(sid, source, paths)
    assert sorted(p.name for p in splitter.ALL_PDB_SYMLINK_PATH.iterdir()) == ["1ABC.cif", "P3.pdb"]
    assert paths["1ABC:P1"] == paths["1ABC:P2"]


def test_grouped_rows_keep_the_chain_of_each_structure() -> None:
    splitter = _hit_splitter()
    splitter.DROP_TO_1D, splitter._grouped_dataset_df = False, None
    splitter.full_dataset_df = splitter.full_dataset_df.with_columns(
        sequence=pl.Series(["s1", "s1", "s2", "s3"]),
        pdb_source=pl.Series(["PDBe", "PDBe", "AlphaFold", None]),
    )
    splitter._max_lddt_to_dataset = {"1ABC:P1": 0.5, "1ABC:P2": 0.7, "P3": 0.2}
    grouped = splitter._get_grouped_dataset_df()
    row = grouped.filter(pl.col("sequence") == "s1").row(0, named=True)
    assert row["lddt"] == 0.7
    assert {(r["pdbs"], r["pdb_chain"]) for r in row["pdb_records"]} == {("1ABC:P1", "AAA"), ("1ABC:P2", "B")}
    restored = splitter._explode_grouped_split_df(grouped)
    assert {"pdbs", "pdb_chain", "pdb_type", "pdb_source"} <= set(restored.columns)


def _cmd_splitter(tmp_path: Path, gpu: bool) -> ProteinStructMaxLDDTSimilaritySplitter:
    splitter = object.__new__(ProteinStructMaxLDDTSimilaritySplitter)
    splitter.unique_structures = ["a", "b"]
    splitter.FOLDSEEK_BIN = Path("/x/foldseek")
    splitter.GPU = gpu
    splitter.SEARCH_QUERY_CHUNK_SIZE = None
    splitter.TOOL_VERSION = "v1"
    splitter.MIN_MEAN_PLDDT = 70.0
    splitter.CACHE_OVERWRITE = True
    splitter.logger = MagicMock()
    splitter.FOLDSEEK_ALL_PATH = tmp_path / "all"
    splitter.FOLDSEEK_TMPDIR = tmp_path / "tmp"
    splitter.ALL_PDB_SYMLINK_PATH = tmp_path / "links"
    splitter.ALL_PDB_SYMLINK_PATH.mkdir(exist_ok=True)
    splitter.ALL_VS_ALL_CACHE_PATH = tmp_path / "cache.tsv"
    splitter._pdb_file_stats = lambda: {}
    splitter._selected_chains = lambda: {}
    # 1ABC has chains A and B but only B is selected; the predicted P3 is named `<stem>_A`.
    splitter._hit_structure_ids = lambda: pl.DataFrame({"hit": ["1ABC_B", "P3_A"]})
    return splitter


LOOKUP = "0\t1ABC_A\t0\n1\t1ABC_B\t0\n2\tP3_A\t1\n"


def _run_all_vs_all(splitter, lookup: str = LOOKUP) -> list[list[str]]:
    """Return every foldseek command; the search stops after its command is built."""
    cmds: list[list[str]] = []

    def run(cmd, **_kwargs):
        cmds.append(cmd)
        if cmd[1] == "createdb":
            Path(f"{cmd[3]}.lookup").write_text(lookup)
        return MagicMock(returncode=0)

    def stop(cmd, **_kwargs):
        cmds.append(cmd)
        raise StopIteration

    splitter._run_search_with_progress = stop
    with patch("subprocess.run", side_effect=run):
        with pytest.raises(StopIteration):
            splitter._foldseek_all_vs_all()
    return cmds


def test_cpu_mode_builds_index_and_searches_without_gpu(tmp_path: Path) -> None:
    cmds = _run_all_vs_all(_cmd_splitter(tmp_path, gpu=False))
    sub = str(tmp_path / "all" / "selectedDB")
    assert [c[1] for c in cmds] == ["createdb", "createsubdb", "createindex", "easy-search"]
    assert all(c[0] == "/x/foldseek" for c in cmds)
    assert cmds[2][2] == sub
    assert cmds[3][2:4] == [sub, sub]
    assert "--gpu" not in cmds[3]


def test_gpu_mode_pads_database_and_searches_it_with_gpu(tmp_path: Path) -> None:
    cmds = _run_all_vs_all(_cmd_splitter(tmp_path, gpu=True))
    db = str(tmp_path / "all" / "selectedDB")
    assert [c[1] for c in cmds] == ["createdb", "createsubdb", "makepaddedseqdb", "easy-search"]
    assert cmds[2][2:] == [db, f"{db}_pad"]
    assert cmds[3][2:4] == [db, f"{db}_pad"]
    assert cmds[3][-2:] == ["--gpu", "1"]


def test_only_selected_chains_enter_the_subset_database(tmp_path: Path) -> None:
    splitter = _cmd_splitter(tmp_path, gpu=False)
    cmds = _run_all_vs_all(splitter)
    full = str(tmp_path / "all" / "foldseekDB")
    keys_file = tmp_path / "all" / "selected_keys.tsv"
    assert cmds[1][2:] == [str(keys_file), full, str(tmp_path / "all" / "selectedDB")]
    assert keys_file.read_text() == "1\n2\n"


def test_no_selected_chain_in_the_database_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No Foldseek chain"):
        _run_all_vs_all(_cmd_splitter(tmp_path, gpu=False), lookup="0\tOTHER_A\t0\n")


def test_fingerprint_depends_on_gpu_mode_and_version(tmp_path: Path) -> None:
    cpu = _cmd_splitter(tmp_path, gpu=False)._similarity_fingerprint()
    assert cpu != _cmd_splitter(tmp_path, gpu=True)._similarity_fingerprint()
    other = _cmd_splitter(tmp_path, gpu=False)
    other.TOOL_VERSION = "v2"
    assert cpu != other._similarity_fingerprint()


def test_missing_binary_raises_with_install_hint(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "splits": {
                "test_frac": 0.1, "valid_frac": 0.1, "log_dir": str(tmp_path / "log"),
                "log_file_name": "t.log", "foldseek_bin": str(tmp_path / "nope"),
                "af_pdb_path": "af", "esm_pdb_path": "esm", "esmfold2_pdb_path": "esmfold2", "mmcif_path": "cif",
                "dataset": {"structure_split": {
                    "input_dataset_parquet_file_path": str(tmp_path / "d.parquet"),
                    "output_dir": str(tmp_path / "out"),
                    "unique_similarity_plot_path": str(tmp_path / "p" / "u.png"),
                    "full_dataset_similarity_plot_path": str(tmp_path / "p" / "f.png"),
                }},
            }
        }
    )
    with pytest.raises(EnvironmentError, match="bash scripts/install_external_tools.sh"):
        ProteinStructMaxLDDTSimilaritySplitter(cfg)


def _grouped_splitter(**extra) -> ProteinStructMaxLDDTSimilaritySplitter:
    splitter = _hit_splitter()
    splitter.DROP_TO_1D, splitter._grouped_dataset_df = False, None
    splitter.full_dataset_df = splitter.full_dataset_df.with_columns(
        sequence=pl.Series(["s1", "s1", "s2", "s3"]),
        pdb_source=pl.Series(["PDBe", "PDBe", "AlphaFold", None]),
        **{k: pl.Series(v) for k, v in extra.items()},
    )
    splitter._max_lddt_to_dataset = {"1ABC:P1": 0.5, "1ABC:P2": 0.7, "P3": 0.2}
    return splitter


def test_pdb_coverage_is_not_a_group_key() -> None:
    splitter = _grouped_splitter(pdb_coverage=[0.9, 0.5, 1.0, None])
    grouped = splitter._get_grouped_dataset_df()
    assert grouped.height == splitter.full_dataset_df["sequence"].n_unique()


def test_drop_to_1d_grouping_ignores_pdb_coverage() -> None:
    splitter = _grouped_splitter(pdb_coverage=[0.9, 0.5, 1.0, None])
    splitter.DROP_TO_1D = True
    assert splitter._get_grouped_dataset_df().height == 3


def test_exploded_split_keeps_pdb_coverage_per_pdb_record() -> None:
    splitter = _grouped_splitter(pdb_coverage=[0.9, 0.5, 1.0, 0.8])
    restored = splitter._explode_grouped_split_df(splitter._get_grouped_dataset_df())
    coverage = {(r.pdbs, r.pdb_coverage) for r in restored.itertuples()}
    assert coverage == {("1ABC:P1", 0.9), ("1ABC:P2", 0.5), ("P3", 1.0), ("MONOMER-1|Akk_muc", 0.8)}


def test_grouping_works_without_pdb_coverage_column() -> None:
    splitter = _grouped_splitter()
    grouped = splitter._get_grouped_dataset_df()
    restored = splitter._explode_grouped_split_df(grouped)
    assert grouped.height == 3
    assert "pdb_coverage" not in restored.columns


def _built_splitter(tmp_path: Path, name: str) -> ProteinStructMaxLDDTSimilaritySplitter:
    tool = tmp_path / "foldseek"
    tool.write_text("#!/bin/sh\necho v1\n")
    tool.chmod(0o755)
    data = tmp_path / "d.parquet"
    pl.DataFrame({"pdbs": ["P1", "P2"], "pdb_chain": [None, None]}).write_parquet(data)
    out = tmp_path / name
    cfg = OmegaConf.create(
        {
            "splits": {
                "test_frac": 0.1, "valid_frac": 0.1, "log_dir": str(tmp_path / "log"),
                "log_file_name": "t.log", "foldseek_bin": str(tool),
                "af_pdb_path": "af", "esm_pdb_path": "esm", "esmfold2_pdb_path": "esmfold2", "mmcif_path": "cif",
                "dataset": {"structure_split": {
                    "input_dataset_parquet_file_path": str(data),
                    "output_dir": str(out),
                    "unique_similarity_plot_path": str(out / "u.png"),
                    "full_dataset_similarity_plot_path": str(out / "f.png"),
                }},
            }
        }
    )
    return ProteinStructMaxLDDTSimilaritySplitter(cfg)


def test_scratch_dirs_are_per_split_and_cache_paths_stay_in_output_dir(tmp_path: Path) -> None:
    a, b = _built_splitter(tmp_path, "a"), _built_splitter(tmp_path, "b")
    scratch = ("FOLDSEEK_ALL_PATH", "FOLDSEEK_TMPDIR", "ALL_PDB_SYMLINK_PATH")
    assert {getattr(a, n) for n in scratch}.isdisjoint({getattr(b, n) for n in scratch})
    assert all(getattr(a, n).parent == a.OUTPUT_DIR for n in scratch)
    assert a.ALL_VS_ALL_CACHE_PATH == a.OUTPUT_DIR / "foldseek_all_vs_all_lddt.tsv"
    assert a.MAX_SIM_CACHE_PATH == a.OUTPUT_DIR / "max_lddt_similarities.tsv"


def _same_sequence_splitter(tmp_path: Path, rows: list, hits: list) -> np.ndarray:
    splitter = _lddt_splitter()
    splitter.CACHE_OVERWRITE = True
    splitter.unique_structures = ["a1", "a2", "b", "c"]
    splitter.MAX_SIM_CACHE_PATH = tmp_path / "lddt.tsv"
    splitter._max_lddt_array = splitter._max_lddt_to_dataset = None
    splitter.full_dataset_df = pl.DataFrame(rows, schema=["structure_id", "sequence"], orient="row")
    splitter._generate_pdb_paths = lambda df: dict.fromkeys(splitter.unique_structures, 1)
    splitter._pdb_file_stats = lambda: []
    splitter._selected_chains = lambda: []
    splitter._foldseek_all_vs_all = lambda: pl.LazyFrame(
        hits, schema=["query", "target", "lddt"], orient="row"
    )
    return splitter.compute_max_lddt_to_dataset()


def test_hits_between_structures_of_one_sequence_are_dropped(tmp_path: Path) -> None:
    rows = [("a1", "S1"), ("a2", "S1"), ("b", "S2"), ("c", None)]
    hits = [("a1", "a2", 1.0), ("a1", "b", 0.3), ("b", "c", 0.5)]
    # a1/a2 share S1, so only the 0.3 hit to S2 counts; a null sequence never matches.
    assert _same_sequence_splitter(tmp_path, rows, hits).tolist() == [0.3, 0.0, 0.5, 0.5]


def test_structure_shared_by_two_sequences_drops_hits_to_both(tmp_path: Path) -> None:
    rows = [("a1", "S1"), ("a1", "S2"), ("a2", "S1"), ("b", "S2"), ("c", "S3")]
    hits = [("a1", "a2", 0.9), ("a1", "b", 0.8), ("a2", "b", 0.7), ("a2", "c", 0.2)]
    # a1 shares S1 with a2 and S2 with b; a2 and b share no sequence.
    assert _same_sequence_splitter(tmp_path, rows, hits).tolist() == [0.0, 0.7, 0.7, 0.2]


def test_max_cache_fingerprint_changes_with_the_structure_sequence_mapping() -> None:
    splitter = _lddt_splitter()
    splitter.unique_structures = ["a", "b"]
    splitter._pdb_file_stats = lambda: []
    splitter._selected_chains = lambda: []
    fingerprints = []
    for sequences in (["AAA", "BBB"], ["AAA", "AAA"]):
        splitter.full_dataset_df = pl.DataFrame({"structure_id": ["a", "b"], "sequence": sequences})
        fingerprints.append(splitter._max_similarity_fingerprint())
    assert fingerprints[0] != fingerprints[1]


def test_only_the_max_cache_fingerprint_marks_same_sequence_exclusion() -> None:
    splitter = _lddt_splitter()
    splitter.unique_structures = ["a"]
    splitter.full_dataset_df = pl.DataFrame({"structure_id": ["a"], "sequence": ["AAA"]})
    splitter._pdb_file_stats = lambda: []
    splitter._selected_chains = lambda: []
    assert splitter._max_similarity_fingerprint() != splitter._similarity_fingerprint()
    assert splitter._similarity_fingerprint() == splitter._cache_fingerprint(
        ["a"],
        search=FOLDSEEK_SEARCH_ARGS,
        max_mode="symmetric",
        missing="nan_missing",
        structure_files=[],
        chains=[],
        gpu=False,
        version="v1",
        search_db="selected_chains",
    )


def test_predicted_structures_fall_back_from_alphafold_to_esmfold2_to_esm3(tmp_path: Path) -> None:
    dirs = {name: tmp_path / name for name in ("af", "esmfold2", "esm", "cif")}
    for d in dirs.values():
        d.mkdir()
    (dirs["af"] / "AF-A1-F1-model_v4.pdb").write_text("")
    (dirs["esmfold2"] / "ESMFold2-A1.pdb").write_text("")
    (dirs["esmfold2"] / "ESMFold2-B2.pdb").write_text("")
    (dirs["esm"] / "ESM3-open-small-B2.pdb").write_text("")
    (dirs["esm"] / "ESM3-open-small-C3.pdb").write_text("")
    splitter = _lddt_splitter()
    splitter.logger = MagicMock()
    splitter.AF_PDB_PATH, splitter.ESMFOLD2_PDB_PATH = dirs["af"], dirs["esmfold2"]
    splitter.ESM_PDB_PATH, splitter.MMCIF_PATH = dirs["esm"], dirs["cif"]
    df = pl.DataFrame(
        {
            "structure_id": ["A1", "B2", "C3", "D4", "1ABC:P9"],
            "pdb_type": ["predicted", "predicted", None, None, "experimental"],
        }
    )
    sources, missing, requested = splitter._pdb_source_paths(df)
    assert {k: v.parent.name for k, v in sources.items()} == {
        "A1": "af", "B2": "esmfold2", "C3": "esm",
    }
    assert missing == {"experimental": 1, "predicted": 1}
    assert requested == {"experimental": 1, "predicted": 4}


def test_esmfold2_symlink_is_named_by_structure_id(tmp_path: Path) -> None:
    source = tmp_path / "ESMFold2-P3.pdb"
    source.write_text("")
    splitter = _lddt_splitter()
    splitter.ALL_PDB_SYMLINK_PATH = tmp_path / "links"
    splitter.ALL_PDB_SYMLINK_PATH.mkdir()
    paths: dict = {}
    splitter._create_pdb_symlink("P3", source, paths)
    assert paths["P3"].name == "P3.pdb"

    # Foldseek names the hit `<file stem>_<chain>`, as observed for a real ESMFold2 PDB.
    tsv = tmp_path / "hits.tsv"
    tsv.write_text("P3_A\t1ABC_B\t0.5\n")
    out = _hit_splitter()._read_foldseek_all_vs_all_tsv(tsv, raw_format=True).collect()
    assert out.rows() == [("P3", "1ABC:P2", 0.5)]


def test_search_uses_shorter_chain_coverage_and_the_fingerprint_follows_it(monkeypatch) -> None:
    assert FOLDSEEK_SEARCH_ARGS[:4] == ["--cov-mode", "2", "-c", "0.8"]
    splitter = _lddt_splitter()
    splitter.unique_structures = ["a"]
    splitter._pdb_file_stats = lambda: []
    splitter._selected_chains = lambda: []
    before = splitter._similarity_fingerprint()
    monkeypatch.setattr(
        "src.data.components.splitters.protein_struct_lddt.FOLDSEEK_SEARCH_ARGS",
        ["--cov-mode", "0", *FOLDSEEK_SEARCH_ARGS[2:]],
    )
    assert splitter._similarity_fingerprint() != before


PDB_LINE = "ATOM  {:>5} {} ALA A{:>4}    {:8.3f}{:8.3f}{:8.3f}  1.00{:6.2f}           C\n"


def _write_pdb(path: Path, scores: list[float]) -> Path:
    lines = []
    for i, score in enumerate(scores, 1):
        lines.append(PDB_LINE.format(i * 2 - 1, " N  ", i, 0, 0, 0, 99.0))
        lines.append(PDB_LINE.format(i * 2, " CA ", i, 1, 1, 1, score))
    path.write_text("".join(lines))
    return path


def _write_cif(path: Path, scores: list[float]) -> Path:
    rows = "".join(
        f"ATOM {i} C CA ALA A {i} 1.0 1.0 1.0 {score}\nATOM {i + 100} N N ALA A {i} 0.0 0.0 0.0 99.0\n"
        for i, score in enumerate(scores, 1)
    )
    path.write_text(
        "data_t\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        f"_atom_site.B_iso_or_equiv\n{rows}"
    )
    return path


def test_mean_plddt_reads_ca_atoms_and_scales_unit_values(tmp_path: Path) -> None:
    assert mean_plddt(_write_pdb(tmp_path / "a.pdb", [40.0, 60.0])) == pytest.approx(50.0)
    assert mean_plddt(_write_pdb(tmp_path / "b.pdb", [0.5, 0.7])) == pytest.approx(60.0)
    assert mean_plddt(_write_cif(tmp_path / "a.cif", [80.0, 90.0])) == pytest.approx(85.0)
    assert mean_plddt(_write_cif(tmp_path / "b.cif", [0.25, 0.75])) == pytest.approx(50.0)


def test_unknown_plddt_is_nan(tmp_path: Path) -> None:
    constant = _write_pdb(tmp_path / "constant.pdb", [1.0, 1.0])
    no_ca = tmp_path / "no_ca.pdb"
    no_ca.write_text(PDB_LINE.format(1, " N  ", 1, 0, 0, 0, 99.0))
    malformed = _write_pdb(tmp_path / "malformed.pdb", [40.0, 60.0])
    malformed.write_text(malformed.read_text().replace(" 40.00", "      "))
    assert all(np.isnan(mean_plddt(p)) for p in (constant, no_ca, malformed))


def _plddt_splitter(tmp_path: Path, files: dict, experimental=()) -> ProteinStructMaxLDDTSimilaritySplitter:
    splitter = _lddt_splitter()
    splitter.PLDDT_CACHE_PATH = tmp_path / "plddt.tsv"
    splitter.MIN_MEAN_PLDDT = 70.0
    ids = [*files, *experimental]
    splitter.full_dataset_df = pl.DataFrame(
        {
            "structure_id": ids,
            "pdb_type": ["predicted"] * len(files) + ["experimental"] * len(experimental),
        }
    )
    sources = {**files, **{sid: tmp_path / "missing.cif" for sid in experimental}}
    splitter._pdb_source_paths = lambda df: (sources, {}, {})
    return splitter


def test_plddt_is_read_for_predicted_files_only_and_cached(tmp_path: Path) -> None:
    low = _write_pdb(tmp_path / "low.pdb", [30.0, 50.0])
    high = _write_pdb(tmp_path / "high.pdb", [80.0, 100.0])
    splitter = _plddt_splitter(tmp_path, {"LOW": low, "HIGH": high}, experimental=["1ABC:P1"])
    # The experimental source does not exist; reading it would fail.
    assert splitter._predicted_mean_plddt() == {"LOW": 40.0, "HIGH": 90.0}
    assert splitter._low_confidence_structures() == {"LOW"}

    # An unchanged file set reuses the cache, whatever the cache holds.
    cached = pl.read_csv(splitter.PLDDT_CACHE_PATH, separator="\t")
    cached.with_columns(mean_plddt=pl.lit(10.0)).write_csv(splitter.PLDDT_CACHE_PATH, separator="\t")
    assert splitter._predicted_mean_plddt() == {"LOW": 10.0, "HIGH": 10.0}

    # A changed file recomputes every value.
    _write_pdb(low, [90.0, 100.0, 95.0])
    assert splitter._predicted_mean_plddt() == {"LOW": 95.0, "HIGH": 90.0}


def test_unknown_plddt_is_low_confidence_and_survives_the_cache(tmp_path: Path) -> None:
    files = {
        "CONST": _write_pdb(tmp_path / "c.pdb", [1.0, 1.0]),
        "OK": _write_pdb(tmp_path / "ok.pdb", [80.0, 100.0]),
    }
    splitter = _plddt_splitter(tmp_path, files)
    assert splitter._low_confidence_structures() == {"CONST"}
    assert splitter._low_confidence_structures() == {"CONST"}  # read back from the cache
    assert (tmp_path / "plddt.tsv.fingerprint").exists()


def test_no_predicted_files_gives_an_empty_result(tmp_path: Path) -> None:
    splitter = _plddt_splitter(tmp_path, {}, experimental=["1ABC:P1"])
    assert splitter._predicted_mean_plddt() == {}
    assert splitter._low_confidence_structures() == set()


def test_plddt_cache_from_another_reader_misses(tmp_path: Path) -> None:
    files = {"A": _write_pdb(tmp_path / "a.pdb", [80.0, 100.0])}
    splitter = _plddt_splitter(tmp_path, files)
    splitter._predicted_mean_plddt()
    sidecar = tmp_path / "plddt.tsv.fingerprint"
    sidecar.write_text(
        splitter._cache_fingerprint(
            files,
            files=[("A", str(files["A"]), files["A"].stat().st_size, files["A"].stat().st_mtime_ns)],
        )
    )
    pl.DataFrame({"structure_id": ["A"], "mean_plddt": [1.0]}).write_csv(splitter.PLDDT_CACHE_PATH, separator="\t")
    assert splitter._predicted_mean_plddt() == {"A": 90.0}


def test_low_confidence_structure_is_unknown_for_eligibility_and_sort(tmp_path: Path) -> None:
    splitter = _plddt_splitter(tmp_path, {})
    splitter.full_dataset_df = pl.DataFrame(
        {
            "sequence": ["a", "b", "b", "c"],
            "structure_id": ["A", "B_LOW", "B_OK", "C_EXP"],
            "pdbs": ["A", "B_LOW", "B_OK", "C_EXP"],
            "pdb_type": ["predicted", "predicted", "predicted", "experimental"],
            "pdb_source": ["AlphaFold"] * 3 + ["PDBe"],
            "pdb_chain": [None, None, None, "A"],
        }
    )
    splitter._low_confidence_structures = lambda: {"A", "B_LOW"}
    # B_LOW hits something closely (0.9); B_OK is dissimilar (0.2). A is dissimilar but low confidence.
    splitter._max_lddt_to_dataset = {"A": 0.05, "B_LOW": 0.9, "B_OK": 0.2, "C_EXP": 0.1}
    splitter.DROP_TO_1D, splitter.DROP_TO_1D_DATASET_PATH = True, None
    splitter._grouped_dataset_df = None
    grouped = splitter._get_grouped_dataset_df().sort("sequence")
    assert grouped["has_structure"].to_list() == [False, True, True]
    assert grouped["has_any_structure"].to_list() == [True, True, True]
    # The sort lDDT of "b" is its eligible structure only.
    assert grouped["lddt"].to_list() == [0.0, 0.2, 0.1]

    splitter.TEST_FRAC, splitter.VAL_FRAC = 1.0, 0.0
    splitter.compute_max_lddt_to_dataset = lambda: None
    splits, _ = splitter.dissimilarity_based_split(0.5)
    assert sorted(splits["test"]["sequence"].to_list()) == ["b", "c"]
    assert splits["train"]["sequence"].to_list() == ["a"]
    logged = " ".join(str(c.args[0]) for c in splitter.logger.info.call_args_list)
    assert "(mean pLDDT < 70, forced to train): 1 (1 rows)" in logged


def test_min_mean_plddt_config_defaults_to_70(tmp_path: Path) -> None:
    assert _built_splitter(tmp_path, "out").MIN_MEAN_PLDDT == 70.0


def _shared_cfg_splitter(tmp_path: Path, name: str, own: list, shared: list[Path] | None):
    tool = tmp_path / "foldseek"
    tool.write_text("#!/bin/sh\necho v1\n")
    tool.chmod(0o755)
    data = tmp_path / f"{name}.parquet"
    pl.DataFrame(
        {
            "pdbs": own,
            "pdb_type": ["experimental"] * len(own),
            "pdb_chain": ["A"] * len(own),
            "sequence": [f"S{i}" for i in range(len(own))],
        }
    ).write_parquet(data)
    out = tmp_path / f"{name}_out"
    struct = {
        "input_dataset_parquet_file_path": str(data),
        "output_dir": str(out),
        "unique_similarity_plot_path": str(out / "u.png"),
        "full_dataset_similarity_plot_path": str(out / "f.png"),
    }
    if shared:
        struct |= {
            "shared_search_inputs": [str(path) for path in shared],
            "shared_search_dir": str(tmp_path / "shared"),
        }
    cfg = OmegaConf.create(
        {
            "splits": {
                "test_frac": 0.1, "valid_frac": 0.1, "log_dir": str(tmp_path / "log"),
                "log_file_name": "t.log", "foldseek_bin": str(tool),
                "af_pdb_path": "af", "esm_pdb_path": "esm", "esmfold2_pdb_path": "esmfold2",
                "mmcif_path": "cif", "dataset": {"structure_split": struct},
            }
        }
    )
    splitter = ProteinStructMaxLDDTSimilaritySplitter(cfg)
    splitter._pdb_file_stats = lambda: []
    return splitter


def _shared_pair(tmp_path: Path):
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    for path, ids in zip(paths, (["1ABC:P1", "2ABC:P2"], ["2ABC:P2", "3ABC:P3"])):
        pl.DataFrame(
            {"pdbs": ids, "pdb_type": "experimental", "pdb_chain": "A", "sequence": [f"S{i}" for i in ids]}
        ).write_parquet(path)
    a = _shared_cfg_splitter(tmp_path, "a", ["1ABC:P1", "2ABC:P2"], paths)
    b = _shared_cfg_splitter(tmp_path, "b", ["2ABC:P2", "3ABC:P3"], paths)
    return a, b


def test_shared_inputs_with_different_chains_for_one_structure_raise(tmp_path: Path) -> None:
    paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    for path, chain in zip(paths, ("A", "B")):
        pl.DataFrame({"pdbs": ["1ABC:P1"], "pdb_type": "experimental", "pdb_chain": chain}).write_parquet(path)
    with pytest.raises(ValueError, match="1ABC:P1"):
        _shared_cfg_splitter(tmp_path, "a", ["1ABC:P1"], paths)


def test_shared_search_fingerprint_is_identical_for_both_datasets(tmp_path: Path) -> None:
    a, b = _shared_pair(tmp_path)
    assert a.search_structures == ["1ABC:P1", "2ABC:P2", "3ABC:P3"]
    assert a._similarity_fingerprint() == b._similarity_fingerprint()
    assert a.ALL_VS_ALL_CACHE_PATH == b.ALL_VS_ALL_CACHE_PATH == tmp_path / "shared" / "foldseek_all_vs_all_lddt.tsv"
    assert a._max_similarity_fingerprint() != b._max_similarity_fingerprint()
    assert a.MAX_SIM_CACHE_PATH.parent == a.OUTPUT_DIR


def test_unset_shared_search_keeps_the_own_dataset_fingerprint(tmp_path: Path) -> None:
    own = _shared_cfg_splitter(tmp_path, "own", ["1ABC:P1", "2ABC:P2"], None)
    assert own.search_df is own.full_dataset_df
    assert own._similarity_fingerprint() == own._cache_fingerprint(
        own.unique_structures,
        search=FOLDSEEK_SEARCH_ARGS,
        max_mode="symmetric",
        missing="nan_missing",
        structure_files=[],
        chains=own._selected_chains(),
        gpu=False,
        version="v1",
        search_db="selected_chains",
    )
    assert own._max_similarity_fingerprint() == own._cache_fingerprint(
        [own._similarity_fingerprint()],
        self_hits="same_sequence_excluded",
        structure_sequences=own._cache_fingerprint(
            f"{sid}\t{seq}" for sid, seq in own.full_dataset_df.select("structure_id", "sequence").rows()
        ),
    )


def test_shared_hits_are_filtered_to_the_datasets_own_structures(tmp_path: Path) -> None:
    a, _ = _shared_pair(tmp_path)
    a.CACHE_OVERWRITE = True
    a.full_dataset_df = a.full_dataset_df.with_columns(sequence=pl.col("structure_id"))
    a._generate_pdb_paths = lambda df: dict.fromkeys(a.search_structures, 1)
    # 3ABC:P3 only belongs to the other dataset; its hits must not reach the max.
    a._foldseek_all_vs_all = lambda: pl.LazyFrame(
        [("1ABC:P1", "3ABC:P3", 0.9), ("2ABC:P2", "3ABC:P3", 0.8), ("1ABC:P1", "2ABC:P2", 0.2)],
        schema=["query", "target", "lddt"], orient="row",
    )
    assert a.compute_max_lddt_to_dataset().tolist() == [0.2, 0.2]


def test_second_dataset_reuses_the_shared_all_vs_all_tsv(tmp_path: Path) -> None:
    a, b = _shared_pair(tmp_path)
    a.ALL_VS_ALL_CACHE_PATH.parent.mkdir(parents=True)
    pl.DataFrame({"query": ["1ABC:P1"], "target": ["2ABC:P2"], "lddt": ["0.5"]}).write_csv(
        a.ALL_VS_ALL_CACHE_PATH, separator="\t"
    )
    a._write_cache_fingerprint(a.ALL_VS_ALL_CACHE_PATH, a._similarity_fingerprint())
    with patch("subprocess.run") as run:
        hits = b._foldseek_all_vs_all().collect()
    run.assert_not_called()
    assert hits.rows() == [("1ABC:P1", "2ABC:P2", 0.5)]


def test_shared_hit_names_map_to_structure_ids_from_both_inputs(tmp_path: Path) -> None:
    a, _ = _shared_pair(tmp_path)
    a._pdb_paths = dict.fromkeys(["1ABC:P1", "3ABC:P3", "P12345"], 1)
    hits = a._hit_structure_ids()
    assert set(hits.rows()) >= {("1ABC:P1", "1ABC_A"), ("3ABC:P3", "3ABC_A")}
    assert set(hits.rows()) >= {("P12345", f"P12345{s}") for s in ("", ".pdb", "_A", ".pdb_A")}
    assert "2ABC:P2" not in hits["structure_id"].to_list()


@pytest.mark.parametrize("chunk_size", [None, 2, 5])
def test_search_is_not_chunked_when_chunk_size_covers_all_chains(tmp_path: Path, chunk_size) -> None:
    splitter = _cmd_splitter(tmp_path, gpu=False)  # two chains are selected
    splitter.SEARCH_QUERY_CHUNK_SIZE = chunk_size
    cmds = _run_all_vs_all(splitter)
    assert [c[1] for c in cmds] == ["createdb", "createsubdb", "createindex", "easy-search"]


def test_chunked_search_runs_one_easy_search_per_query_chunk(tmp_path: Path) -> None:
    splitter = _cmd_splitter(tmp_path, gpu=False)
    splitter.SEARCH_QUERY_CHUNK_SIZE = 1
    splitter._run_search_with_progress = lambda cmd, **_: 0.0
    cmds: list[list[str]] = []

    def run(cmd, **_kwargs):
        cmds.append(cmd)
        if cmd[1] == "createdb":
            Path(f"{cmd[3]}.lookup").write_text(LOOKUP)
        if cmd[1] == "createsubdb":
            Path(f"{cmd[4]}.lookup").write_text(Path(cmd[2]).read_text())
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=run):
        splitter._foldseek_all_vs_all()
    subs = [c for c in cmds if c[1] == "createsubdb"]
    assert len(subs) == 3  # the selection, then one query chunk per chain
    assert all(c[3] == str(tmp_path / "all" / "foldseekDB") for c in subs[1:])


AF_DIR = Path(os.environ.get("EMULATOR_AF_DIR", "/nonexistent"))


@pytest.mark.skipif(
    shutil.which("foldseek") is None or not AF_DIR.is_dir(), reason="foldseek missing or EMULATOR_AF_DIR unset or not a directory"
)
def test_chunked_search_gives_the_same_hits_as_one_search(tmp_path: Path) -> None:
    links = tmp_path / "links"
    links.mkdir()
    sources = sorted(AF_DIR.glob("*.pdb"), key=lambda p: p.stat().st_size)[:24]
    for src in sources:
        shutil.copy(src, links / src.name)
    splitter = _cmd_splitter(tmp_path, gpu=False)
    splitter.FOLDSEEK_BIN = Path(shutil.which("foldseek"))
    splitter.SEARCH_QUERY_CHUNK_SIZE = 7
    splitter._hit_structure_ids = lambda: pl.DataFrame({"hit": [f"{src.stem}_A" for src in sources]})
    splitter._foldseek_create_database()
    work = tmp_path / "all"
    db = work / "selectedDB"
    args = ["--format-output", "query,target,lddt", *FOLDSEEK_SEARCH_ARGS]
    whole = work / "whole.tsv"
    subprocess.run(
        [str(splitter.FOLDSEEK_BIN), "easy-search", str(db), str(db), str(whole), str(tmp_path / "t"), *args],
        check=True, capture_output=True,
    )
    chunked = work / "chunked.tsv"
    splitter._foldseek_search_in_chunks(7, db, chunked, args)
    assert whole.read_text()
    assert sorted(chunked.read_text().splitlines()) == sorted(whole.read_text().splitlines())
    assert not list(work.glob("chunkDB*"))


def test_raw_foldseek_hits_give_the_hand_computed_max_lddt(tmp_path: Path) -> None:
    splitter = _lddt_splitter()
    splitter.CACHE_OVERWRITE = True
    splitter.unique_structures = ["a1", "a2", "b"]
    splitter.MAX_SIM_CACHE_PATH = tmp_path / "lddt.tsv"
    splitter._max_lddt_array = splitter._max_lddt_to_dataset = None
    splitter.full_dataset_df = pl.DataFrame(
        [("a1", "S1"), ("a2", "S1"), ("b", "S2"), ("x", "S3")],
        schema=["structure_id", "sequence"], orient="row",
    )
    splitter._generate_pdb_paths = lambda df: dict.fromkeys(splitter.unique_structures, 1)
    splitter._pdb_file_stats = lambda: []
    splitter._selected_chains = lambda: []
    splitter._hit_structure_ids = lambda: pl.DataFrame(
        {"structure_id": ["a1", "a2", "b", "x"], "hit": ["A1_A", "A2_A", "B_A", "X_A"]}
    )
    rows = [
        ("A1_A", "A2_A", "0.99"),  # same sequence: dropped
        ("A1_A", "B_A", "2.500E-01"),  # scientific notation
        ("B_A", "A2_A", "0.4"),
        ("A1_A", "A1_A", "1.0"),  # self pair: dropped
        ("A1_A", "X_A", "0.9"),  # structure outside the dataset: dropped
        ("A1_A", "UNKNOWN_A", "0.8"),  # not a dataset structure: dropped
    ]
    raw = tmp_path / "raw.tsv"
    raw.write_text("".join("\t".join(row) + "\n" for row in rows))
    splitter._foldseek_all_vs_all = lambda: splitter._read_foldseek_all_vs_all_tsv(raw, raw_format=True)

    assert splitter.compute_max_lddt_to_dataset().tolist() == [0.25, 0.4, 0.4]


def _chunked_run(tmp_path: Path, searched: list[str], prep=None, fingerprint: str = "fp"):
    """Run the chunked all-vs-all with a stub search; the hit of a chunk is its query key."""
    splitter = _cmd_splitter(tmp_path, gpu=False)
    splitter.SEARCH_QUERY_CHUNK_SIZE = 1
    splitter._similarity_fingerprint = lambda: fingerprint
    cmds: list[list[str]] = []

    def search(cmd, out_file, **_):
        key = Path(cmd[2]).with_name("chunk_keys.tsv").read_text().strip()
        searched.append(key)
        if key != "1":  # chunk of key 1 has no hits
            Path(out_file).write_text(f"hit{key}\n")
        return 0.0

    def run(cmd, **_kwargs):
        cmds.append(cmd)
        if cmd[1] == "createdb":
            Path(f"{cmd[3]}.lookup").write_text(LOOKUP)
        if cmd[1] == "createsubdb":
            Path(f"{cmd[4]}.lookup").write_text(Path(cmd[2]).read_text())
        return MagicMock(returncode=0)

    splitter._run_search_with_progress = search
    if prep:
        prep(splitter)
    # The normalization step reads the output; stopping there keeps the work dir for inspection.
    with patch("subprocess.run", side_effect=run), patch.object(
        splitter, "_read_foldseek_all_vs_all_tsv", side_effect=RuntimeError("stop")
    ):
        with pytest.raises(RuntimeError, match="stop"):
            splitter._foldseek_all_vs_all()
    return splitter, cmds


def test_chunked_search_writes_one_file_per_chunk_and_concatenates_in_order(tmp_path: Path) -> None:
    searched: list[str] = []
    splitter, _ = _chunked_run(tmp_path, searched)
    work = splitter.FOLDSEEK_ALL_PATH
    assert searched == ["1", "2"]
    assert (work / "all_vs_all.tsv").read_text() == "hit2\n"
    assert (work / "chunks" / "chunk_001.tsv").read_text() == ""  # no hits still gives a file
    assert not list((work / "chunks").glob("*.part"))


def _finished_chunks(splitter, **manifest_changes) -> None:
    """Pretend an earlier run finished chunk 1 and then stopped."""
    chunks = splitter.FOLDSEEK_ALL_PATH / "chunks"
    (chunks / "chunk_001.tsv").write_text("old1\n")
    (chunks / "chunk_002.tsv").unlink()
    manifest = json.loads((chunks / "manifest.json").read_text()) | manifest_changes
    (chunks / "manifest.json").write_text(json.dumps(manifest))


def _second_run(tmp_path: Path, changes: dict, fingerprint: str = "fp"):
    first: list[str] = []
    splitter, _ = _chunked_run(tmp_path, first)
    _finished_chunks(splitter, **changes)
    searched: list[str] = []
    # Reuse the work dir of the first run: _cmd_splitter has the same paths.
    splitter, cmds = _chunked_run(tmp_path, searched, fingerprint=fingerprint)
    return splitter, cmds, searched


def test_matching_manifest_reuses_database_and_skips_finished_chunks(tmp_path: Path) -> None:
    splitter, cmds, searched = _second_run(tmp_path, {})
    assert "createdb" not in [c[1] for c in cmds]
    assert searched == ["2"]
    assert (splitter.FOLDSEEK_ALL_PATH / "all_vs_all.tsv").read_text() == "old1\nhit2\n"


@pytest.mark.parametrize(
    "changes",
    [
        {"similarity_fingerprint": "other"},
        {"chunk_size": 5},
        {"search_args": ["-x"]},
        {"selected_keys_sha256": "0"},
        {"selected_lookup_sha256": "0"},
    ],
)
def test_manifest_mismatch_rebuilds_and_removes_old_chunks(tmp_path: Path, changes: dict) -> None:
    splitter, cmds, searched = _second_run(tmp_path, changes)
    assert "createdb" in [c[1] for c in cmds]
    assert searched == ["1", "2"]
    assert (splitter.FOLDSEEK_ALL_PATH / "all_vs_all.tsv").read_text() == "hit2\n"
    assert (splitter.FOLDSEEK_ALL_PATH / "chunks" / "chunk_001.tsv").read_text() == ""


def test_changed_fingerprint_rebuilds(tmp_path: Path) -> None:
    first: list[str] = []
    splitter, _ = _chunked_run(tmp_path, first)
    _, cmds = _chunked_run(tmp_path, [], fingerprint="changed")
    assert "createdb" in [c[1] for c in cmds]
