from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

from scripts.generate_conformers import build_smiles_dict, parse_args


def _pickle(path: Path, items: list) -> Path:
    path.write_bytes(pickle.dumps(items))
    return path


def test_smiles_dict_maps_sha256_to_each_unique_smiles_key(tmp_path: Path) -> None:
    records = [
        {"smiles": "[O-]", "smiles_key": "O"},
        {"smiles": "C", "smiles_key": "C"},
        {"smiles": "O", "smiles_key": "O"},
    ]
    path = _pickle(tmp_path / "s.pkl", records)
    assert build_smiles_dict(path) == {
        hashlib.sha256(s.encode()).hexdigest(): s for s in ("C", "O")
    }


def test_input_path_defaults_to_the_configured_superset() -> None:
    argv, sys.argv = sys.argv, ["x"]
    try:
        assert parse_args().input_path is None
        sys.argv = ["x", "--input-path", "s.pkl"]
        assert parse_args().input_path == "s.pkl"
    finally:
        sys.argv = argv


def test_configured_input_is_the_superset_pickle() -> None:
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="data_processing")
    path = cfg.embeddings.datamol_conformer_input_pkl_path
    assert Path(path).name == "all_canonical_smiles.pkl"
    assert cfg.embeddings.datamol_conformer_timeout_s == 600
    assert "unimol_input_dataset_parquet_file_path" not in cfg.embeddings


def test_main_composes_the_real_config_and_runs_the_generator(tmp_path: Path, monkeypatch) -> None:
    from scripts import generate_conformers as gc

    calls = []

    class FakeGenerator:
        def __init__(self, cfg) -> None:
            calls.append(cfg)

        def generate(self, smiles_dict) -> None:
            calls.append(smiles_dict)

    path = _pickle(tmp_path / "s.pkl", [{"smiles_key": "C"}])
    monkeypatch.setattr(gc, "Datamol3DConformerGenerator", FakeGenerator)
    monkeypatch.setattr(sys, "argv", ["x", "--input-path", str(path)])
    gc.main()
    assert calls[0].embeddings.datamol_conformer_timeout_s == 600
    assert list(calls[1].values()) == ["C"]
