from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.components.embedders.atom_pair import SmilesAtomPairShardEmbedder
from src.data.components.embedders.store import ShardedEmbeddingReader
from tests.helpers.builders import embedder_config, run_pandarallel_serially


def _embedder(tmp_path, items, monkeypatch, **overrides) -> SmilesAtomPairShardEmbedder:
    run_pandarallel_serially(monkeypatch)
    settings = {
        "atom_pair_nbits": 64,
        "atom_pair_pandarallel": {"nb_workers": 1, "progress_bar": False},
        "smiles_atom_pair_embeddings_log_file_name": "atom_pair.log",
        **overrides,
    }
    cfg = embedder_config(tmp_path, items, **settings)
    cfg.embeddings.smiles_atom_pair_embeddings_input_pkl_path = cfg.embeddings.input_pkl_path
    return SmilesAtomPairShardEmbedder(cfg)


def test_model_name_carries_the_bit_count(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch, atom_pair_nbits=128)
    assert embedder.store.root_dir == tmp_path / "emb" / "atom_pair" / "nbits128" / "v1"


def test_unset_bit_count_defaults_to_1024(tmp_path, monkeypatch) -> None:
    assert _embedder(tmp_path, ["C"], monkeypatch, atom_pair_nbits=None).n_bits == 1024


def test_fingerprints_are_uint8_arrays_of_the_configured_width(tmp_path, monkeypatch) -> None:
    arrays, failures = _embedder(tmp_path, ["C"], monkeypatch).compute_many(["CCO", "CC(=O)O"])
    assert failures == []
    for fingerprint in arrays.values():
        assert fingerprint.shape == (64,)
        assert fingerprint.dtype == np.uint8
        assert fingerprint.any()


def test_fingerprints_are_deterministic_and_differ_between_molecules(
    tmp_path, monkeypatch
) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch)
    first, _ = embedder.compute_many(["CCO", "CC(=O)O"])
    second, _ = embedder.compute_many(["CCO", "CC(=O)O"])
    np.testing.assert_array_equal(first["CCO"], second["CCO"])
    assert not np.array_equal(first["CCO"], first["CC(=O)O"])


def test_invalid_smiles_are_reported_as_failures(tmp_path, monkeypatch) -> None:
    arrays, failures = _embedder(tmp_path, ["C"], monkeypatch).compute_many(
        ["CCO", "not_a_smiles"]
    )
    assert list(arrays) == ["CCO"]
    assert [f["raw_key"] for f in failures] == ["not_a_smiles"]
    assert failures[0]["error"] == "Fingerprint generation returned None"


def test_embed_stores_fingerprints_and_records_invalid_smiles(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["CCO", "not_a_smiles"], monkeypatch)
    embedder.embed()
    assert ShardedEmbeddingReader(embedder.store.root_dir).get("CCO").shape == (64,)
    assert pd.read_parquet(embedder.store.failures_path)["raw_key"].tolist() == ["not_a_smiles"]
