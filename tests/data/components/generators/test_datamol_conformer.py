from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_conformer_shard_folder_is_created(tmp_path: Path) -> None:
    pytest.importorskip("datamol")
    from src.data.components.generators.datamol_conformer import Datamol3DConformerGenerator

    generator = object.__new__(Datamol3DConformerGenerator)
    generator.DATAMOL_SDF_PATH = tmp_path / "datamol"
    generator.NUM_CONFORMERS, generator.RANDOM_SEED, generator.OPTIMIZE = 1, 0, False
    generator.TIMEOUT_S = 600
    generator.logger = MagicMock()
    assert generator._generate_conformer_datamol(("abcd1234", "CCO")) == ("abcd1234", "ok")
    assert list((tmp_path / "datamol" / "abcd").glob("*.sdf"))


def test_sdf_write_is_atomic(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("datamol")
    from rdkit import Chem

    from src.data.components.generators.datamol_conformer import Datamol3DConformerGenerator

    gen = object.__new__(Datamol3DConformerGenerator)
    gen.DATAMOL_SDF_PATH = tmp_path
    gen.NUM_CONFORMERS, gen.RANDOM_SEED, gen.OPTIMIZE = 1, 0, False
    gen.TIMEOUT_S = 600
    gen.logger = logging.getLogger("t")

    assert gen._generate_conformer_datamol(("abcd1", "CCO")) == ("abcd1", "ok")
    assert [p.name for p in tmp_path.glob("*/*")] == ["datamol_abcd1_conf0.sdf"]

    def broken_write(self, *args, **kwargs):
        raise OSError("killed")

    monkeypatch.setattr(Chem.SDWriter, "write", broken_write)
    assert gen._generate_conformer_datamol(("efgh2", "CCO")) == ("efgh2", "failed")
    assert list(tmp_path.glob("efgh/*")) == []


SLOW_SMILES = "CC(C)Oc1ccc(NC(=O)[C@@H]2C[C@@H]3CC[C@@H]2N(S(=O)(=O)c2ccccc2F)C3)cc1"


def _generator(tmp_path: Path, **settings):
    from src.data.components.generators.datamol_conformer import Datamol3DConformerGenerator

    gen = object.__new__(Datamol3DConformerGenerator)
    gen.DATAMOL_SDF_PATH = tmp_path
    gen.OUTCOMES_PATH = tmp_path / "outcomes.tsv"
    gen.NUM_CONFORMERS, gen.RANDOM_SEED, gen.OPTIMIZE = 10, 0, True
    gen.MAX_ATTEMPTS, gen.TIMEOUT_S = 1000, 600
    gen.logger = MagicMock()
    for name, value in settings.items():
        setattr(gen, name, value)
    return gen


def test_slow_molecule_times_out_and_writes_no_sdf(tmp_path: Path) -> None:
    pytest.importorskip("datamol")
    gen = _generator(tmp_path, TIMEOUT_S=2)
    assert gen._generate_conformer_datamol(("abcd1", SLOW_SMILES)) == ("abcd1", "timeout")
    assert list(tmp_path.glob("*/*")) == []


def test_resume_skips_attempts_with_the_same_settings_only(tmp_path: Path) -> None:
    gen = _generator(tmp_path)
    gen.OUTCOMES_PATH.write_text(
        f"h1\ttimeout\t{gen._settings_id}\nh2\tfailed\tother\nh3\tpartial\t{gen._settings_id}\nh4\tfai"
    )
    assert gen._read_attempted() == {"h1", "h3"}
    assert _generator(tmp_path, TIMEOUT_S=900)._read_attempted() == set()


def test_generate_records_incomplete_outcomes_and_skips_them_on_rerun(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pandas")
    from src.data.components.generators import datamol_conformer as module

    class SerialPool:
        def __init__(self, processes=None): ...
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def imap_unordered(self, fn, items): return map(fn, items)

    monkeypatch.setattr(module, "Pool", SerialPool)
    gen = _generator(tmp_path, NUM_CONFORMERS=1)
    calls = []

    def fake(args):
        calls.append(args[0])
        return (args[0], {"a" * 64: "ok", "b" * 64: "timeout"}[args[0]])

    gen._generate_conformer_datamol = fake
    smiles = {"a" * 64: "C", "b" * 64: "CC"}
    assert gen.generate(smiles) == (["a" * 64], ["b" * 64])
    assert gen.OUTCOMES_PATH.read_text() == f"{'b' * 64}\ttimeout\t{gen._settings_id}\n"

    calls.clear()
    gen.generate(smiles)
    assert calls == ["a" * 64]
