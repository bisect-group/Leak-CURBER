from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.components.embedders.ecfp import SmilesECFPShardEmbedder
from src.data.components.embedders.store import ShardedEmbeddingReader
from tests.helpers.builders import embedder_config, run_pandarallel_serially


def _embedder(tmp_path, items, monkeypatch, **overrides) -> SmilesECFPShardEmbedder:
    run_pandarallel_serially(monkeypatch)
    settings = {
        "ecfp_radius": 2,
        "ecfp_nbits": 64,
        "ecfp_pandarallel": {"nb_workers": 1, "progress_bar": False},
        "smiles_ecfp_embeddings_log_file_name": "ecfp.log",
        **overrides,
    }
    cfg = embedder_config(tmp_path, items, **settings)
    cfg.embeddings.smiles_ecfp_embeddings_input_pkl_path = cfg.embeddings.input_pkl_path
    return SmilesECFPShardEmbedder(cfg)


def test_model_name_carries_radius_and_bit_count(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch, ecfp_radius=3, ecfp_nbits=128)
    assert embedder.store.root_dir == tmp_path / "emb" / "ecfp" / "radius3_nbits128" / "v1"


def test_unset_radius_and_bits_default_to_2_and_1024(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch, ecfp_radius=None, ecfp_nbits=None)
    assert (embedder.radius, embedder.n_bits) == (2, 1024)


def test_fingerprints_are_uint8_bit_vectors_of_the_configured_width(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch)
    arrays, failures = embedder.compute_many(["CCO", "c1ccccc1"])
    assert failures == []
    assert list(arrays) == ["CCO", "c1ccccc1"]
    for fingerprint in arrays.values():
        assert fingerprint.shape == (64,)
        assert fingerprint.dtype == np.uint8
        assert set(np.unique(fingerprint)) <= {0, 1}
        assert fingerprint.any()


def test_fingerprints_are_deterministic_and_differ_between_molecules(
    tmp_path, monkeypatch
) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch)
    first, _ = embedder.compute_many(["CCO", "c1ccccc1"])
    second, _ = embedder.compute_many(["CCO", "c1ccccc1"])
    np.testing.assert_array_equal(first["CCO"], second["CCO"])
    assert not np.array_equal(first["CCO"], first["c1ccccc1"])


def test_invalid_smiles_are_reported_as_failures(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch)
    arrays, failures = embedder.compute_many(["CCO", "not_a_smiles"])
    assert list(arrays) == ["CCO"]
    assert failures == [
        {
            "raw_key": "not_a_smiles",
            "canonical_key": "not_a_smiles",
            "error": "Fingerprint generation returned None",
        }
    ]


def test_pandarallel_starts_once_across_chunks(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["C"], monkeypatch)
    calls = []
    monkeypatch.setattr(
        "pandarallel.pandarallel.initialize", lambda **kwargs: calls.append(kwargs)
    )
    embedder.compute_many(["C"])
    embedder.compute_many(["CC"])
    assert calls == [{"progress_bar": False, "nb_workers": 1}]


def test_embed_stores_fingerprints_and_records_invalid_smiles(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, ["CCO", "not_a_smiles", "CCO"], monkeypatch)
    embedder.embed()
    reader = ShardedEmbeddingReader(embedder.store.root_dir)
    assert reader.get("CCO").shape == (64,)
    assert not reader.contains("not_a_smiles")
    failures = pd.read_parquet(embedder.store.failures_path)
    assert failures["raw_key"].tolist() == ["not_a_smiles"]
