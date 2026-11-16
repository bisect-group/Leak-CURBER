from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.components.splitters.base import BaseThresholdedSimilaritySplitter
from src.data.components.splitters.protein_seq_fident import (
    MMSEQS_SEARCH_ARGS,
    ProteinSeqMaxFidentSimilaritySplitter,
    fident_fingerprint,
)


def _fident_splitter(tmp_path: Path, calls: list) -> ProteinSeqMaxFidentSimilaritySplitter:
    splitter = object.__new__(ProteinSeqMaxFidentSimilaritySplitter)
    splitter.unique_sequences = ["a", "b", "c"]
    splitter.SEQUENCE_COL = "sequence"
    splitter.CACHE_OVERWRITE = False
    splitter.MAX_SIM_CACHE_PATH = tmp_path / "max.tsv"
    splitter._max_fident_array = splitter._max_fident_to_dataset = None
    splitter.logger = MagicMock()
    splitter.GPU = False
    splitter.TOOL_VERSION = "v1"

    def fake_all_vs_all():
        calls.append(1)
        # Only one direction is reported: id 0 ("a") is the query, id 1 ("b") is the target.
        return pd.DataFrame({"query": [0], "target": [1], "fident": [0.9]})

    splitter._mmseqs_all_vs_all = fake_all_vs_all
    return splitter


def test_max_fident_counts_hits_where_sequence_is_the_target(tmp_path: Path) -> None:
    splitter = _fident_splitter(tmp_path, [])
    assert splitter.compute_max_fident_to_dataset().tolist() == [0.9, 0.9, 0.0]


def test_cache_without_matching_fingerprint_is_recomputed(tmp_path: Path) -> None:
    calls: list = []
    _fident_splitter(tmp_path, calls).compute_max_fident_to_dataset()
    assert len(calls) == 1

    _fident_splitter(tmp_path, calls).compute_max_fident_to_dataset()  # warm: reused
    assert len(calls) == 1

    changed = _fident_splitter(tmp_path, calls)
    changed.unique_sequences = ["a", "b", "z"]  # different inputs: miss, overwrite
    changed.compute_max_fident_to_dataset()
    assert len(calls) == 2

    tmp_path.joinpath("max.tsv.fingerprint").unlink()  # a cache without a fingerprint
    _fident_splitter(tmp_path, calls).compute_max_fident_to_dataset()
    assert len(calls) == 3


def test_fingerprint_depends_on_keys_and_params_not_order() -> None:
    fingerprint = ProteinSeqMaxFidentSimilaritySplitter._cache_fingerprint
    assert fingerprint(["a", "b"], x=1) == fingerprint(["b", "a"], x=1)
    assert fingerprint(["a", "b"], x=1) != fingerprint(["a", "b"], x=2)
    assert fingerprint(["a", "b"], x=1) != fingerprint(["a", "c"], x=1)


def _cmd_splitter(tmp_path: Path, gpu: bool) -> ProteinSeqMaxFidentSimilaritySplitter:
    splitter = object.__new__(ProteinSeqMaxFidentSimilaritySplitter)
    splitter.unique_sequences = ["a", "b"]
    splitter.MMSEQS_BIN = Path("/x/mmseqs")
    splitter.GPU = gpu
    splitter.TOOL_VERSION = "v1"
    splitter.CACHE_OVERWRITE = True
    splitter.logger = MagicMock()
    splitter.MMSEQS_ALL_PATH = tmp_path / "all"
    splitter.MMSEQS_TMPDIR = tmp_path / "tmp"
    splitter.FASTA_PATH = tmp_path / "in.fasta"
    splitter.ALL_VS_ALL_CACHE_PATH = tmp_path / "cache.tsv"
    return splitter


def _run_all_vs_all(splitter) -> list[list[str]]:
    """Return every mmseqs command; the search stops after its command is built."""
    cmds: list[list[str]] = []

    def stop(cmd, **_kwargs):
        cmds.append(cmd)
        raise StopIteration

    splitter._write_fasta = lambda: None
    splitter._run_search_with_progress = stop
    with patch("subprocess.run", side_effect=lambda cmd, **_: cmds.append(cmd) or MagicMock(returncode=0)):
        with pytest.raises(StopIteration):
            splitter._mmseqs_all_vs_all()
    return cmds


def test_cpu_mode_builds_index_and_searches_without_gpu(tmp_path: Path) -> None:
    cmds = _run_all_vs_all(_cmd_splitter(tmp_path, gpu=False))
    assert [c[1] for c in cmds] == ["createdb", "createindex", "easy-search"]
    assert all(c[0] == "/x/mmseqs" for c in cmds)
    assert cmds[2][3] == str(tmp_path / "all" / "mmseqsDB")
    assert "--gpu" not in cmds[2]


def test_gpu_mode_pads_database_and_searches_it_with_gpu(tmp_path: Path) -> None:
    cmds = _run_all_vs_all(_cmd_splitter(tmp_path, gpu=True))
    db = str(tmp_path / "all" / "mmseqsDB")
    assert [c[1] for c in cmds] == ["createdb", "makepaddedseqdb", "easy-search"]
    assert cmds[1][2:] == [db, f"{db}_pad"]
    assert cmds[2][3] == f"{db}_pad"
    assert cmds[2][-2:] == ["--gpu", "1"]


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
                "log_file_name": "t.log", "mmseqs_bin": str(tmp_path / "nope"),
                "dataset": {"sequence_split": {
                    "input_dataset_parquet_file_path": str(tmp_path / "d.parquet"),
                    "output_dir": str(tmp_path / "out"),
                    "unique_similarity_plot_path": str(tmp_path / "p" / "u.png"),
                    "full_dataset_similarity_plot_path": str(tmp_path / "p" / "f.png"),
                }},
            }
        }
    )
    with pytest.raises(EnvironmentError, match="bash scripts/install_external_tools.sh"):
        ProteinSeqMaxFidentSimilaritySplitter(cfg)


def _built_splitter(tmp_path: Path, name: str) -> ProteinSeqMaxFidentSimilaritySplitter:
    tool = tmp_path / "mmseqs"
    tool.write_text("#!/bin/sh\necho v1\n")
    tool.chmod(0o755)
    data = tmp_path / "d.parquet"
    pd.DataFrame({"sequence": ["AAA", "CCC"]}).to_parquet(data)
    out = tmp_path / name
    cfg = OmegaConf.create(
        {
            "splits": {
                "test_frac": 0.1, "valid_frac": 0.1, "log_dir": str(tmp_path / "log"),
                "log_file_name": "t.log", "mmseqs_bin": str(tool),
                "dataset": {"sequence_split": {
                    "input_dataset_parquet_file_path": str(data),
                    "output_dir": str(out),
                    "unique_similarity_plot_path": str(out / "u.png"),
                    "full_dataset_similarity_plot_path": str(out / "f.png"),
                }},
            }
        }
    )
    return ProteinSeqMaxFidentSimilaritySplitter(cfg)


def test_scratch_dirs_are_per_split_and_cache_paths_stay_in_output_dir(tmp_path: Path) -> None:
    a, b = _built_splitter(tmp_path, "a"), _built_splitter(tmp_path, "b")
    assert {a.MMSEQS_ALL_PATH, a.MMSEQS_TMPDIR}.isdisjoint({b.MMSEQS_ALL_PATH, b.MMSEQS_TMPDIR})
    assert a.MMSEQS_ALL_PATH.parent == a.MMSEQS_TMPDIR.parent == a.OUTPUT_DIR
    assert a.ALL_VS_ALL_CACHE_PATH == a.OUTPUT_DIR / "mmseqs_all_vs_all_fident.tsv"
    assert a.MAX_SIM_CACHE_PATH == a.OUTPUT_DIR / "max_fident_similarities.tsv"
    assert a._similarity_fingerprint() == b._similarity_fingerprint()


def _stubbed_search_splitter(tmp_path: Path, sequences: list[str], hits: str) -> tuple:
    """Splitter whose MMseqs2 search writes `hits` (integer-id TSV); returns it and a call list."""
    splitter = _cmd_splitter(tmp_path, gpu=False)
    splitter.unique_sequences = sequences
    splitter.SEQUENCE_COL = "sequence"
    splitter.CACHE_OVERWRITE = False
    splitter.MAX_SIM_CACHE_PATH = tmp_path / "max.tsv"
    splitter._max_fident_array = splitter._max_fident_to_dataset = None
    calls: list = []

    def search(cmd, *, out_file, **_kwargs):
        calls.append(cmd)
        Path(out_file).write_text(hits)
        return 0.0

    splitter._run_search_with_progress = search
    return splitter, calls


def test_fasta_ids_are_sorted_index_independent_of_dataset_order(tmp_path: Path) -> None:
    fastas = []
    for name, order in (("x", ["C", "A", "B"]), ("y", ["B", "C", "A"])):
        splitter = _cmd_splitter(tmp_path / name, gpu=False)
        splitter.unique_sequences = order
        splitter.FASTA_PATH = tmp_path / f"{name}.fasta"
        splitter._write_fasta()
        fastas.append(splitter.FASTA_PATH.read_text())
    assert fastas[0] == fastas[1] == ">0\nA\n>1\nB\n>2\nC\n"


def test_all_vs_all_cache_stores_integer_ids_and_round_trips(tmp_path: Path) -> None:
    # Dataset order differs from sorted order: ids 0, 1, 2 are "a", "b", "c".
    hits = "0\t0\t1.0\n1\t1\t1.0\n0\t1\t0.9\n2\t1\t0.4\n"
    splitter, calls = _stubbed_search_splitter(tmp_path, ["c", "a", "b"], hits)
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        first = splitter.compute_max_fident_to_dataset()
    assert first.tolist() == [0.4, 0.9, 0.9]  # c, a, b
    cache = pd.read_csv(splitter.ALL_VS_ALL_CACHE_PATH, sep="\t")
    assert cache.columns.tolist() == ["query", "target", "fident"]
    assert cache.to_dict("list") == {"query": [0, 2], "target": [1, 1], "fident": [0.9, 0.4]}

    splitter._max_fident_array = splitter._max_fident_to_dataset = None
    splitter.MAX_SIM_CACHE_PATH.unlink()  # force the all-vs-all cache to be read
    assert splitter.compute_max_fident_to_dataset().tolist() == first.tolist()
    assert len(calls) == 1


def test_sequence_keyed_cache_with_old_fingerprint_is_a_miss(tmp_path: Path) -> None:
    splitter, calls = _stubbed_search_splitter(tmp_path, ["a", "b"], "0\t1\t0.7\n")
    old = BaseThresholdedSimilaritySplitter._cache_fingerprint(
        ["a", "b"], search=MMSEQS_SEARCH_ARGS, max_mode="symmetric", gpu=False, version="v1"
    )
    assert old != fident_fingerprint(["a", "b"], False, "v1")
    splitter.ALL_VS_ALL_CACHE_PATH.write_text("query\ttarget\tfident\na\tb\t0.5\n")
    splitter._write_cache_fingerprint(splitter.ALL_VS_ALL_CACHE_PATH, old)
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        assert splitter.compute_max_fident_to_dataset().tolist() == [0.7, 0.7]
    assert len(calls) == 1
