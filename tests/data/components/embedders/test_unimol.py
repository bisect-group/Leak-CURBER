from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import AllChem

from src.data.components.embedders.store import ShardedEmbeddingReader
from src.data.components.embedders import unimol
from src.data.components.embedders.unimol import UnimolSDFEmbedder, UnimolSDFShardEmbedder

HASH_A, HASH_B = "aaaa1111", "bbbb2222"


def _embedder(tmp_path, **embeddings) -> UnimolSDFShardEmbedder:
    cfg = OmegaConf.create(
        {
            "paths": {"raw_data_dir": str(tmp_path / "raw")},
            "embeddings": {
                "log_dir": str(tmp_path / "logs"),
                "unimol_log_file_name": "unimol.log",
                "unimol_sdf_dir": str(tmp_path / "sdf"),
                "unimol_embeddings_store_dir": str(tmp_path / "store"),
                "datamol_num_conformers": 2,
                **embeddings,
            },
        }
    )
    return UnimolSDFShardEmbedder(cfg)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def test_sdf_paths_are_sharded_by_the_first_four_hash_characters(tmp_path) -> None:
    embedder = _embedder(tmp_path)
    assert embedder._pubchem_sdf_path(HASH_A) == tmp_path / "sdf/pubchem/aaaa/pubchem_aaaa1111.sdf"
    assert (
        embedder._datamol_sdf_path(HASH_A, 1)
        == tmp_path / "sdf/datamol/aaaa/datamol_aaaa1111_conf1.sdf"
    )


def test_stems_per_batch_defaults_to_one_million(tmp_path) -> None:
    assert _embedder(tmp_path).stems_per_batch == 1_000_000
    assert _embedder(tmp_path, unimol_stems_per_batch=5).stems_per_batch == 5


def test_missing_input_pickle_is_an_error(tmp_path) -> None:
    embedder = _embedder(tmp_path, datamol_conformer_input_pkl_path=str(tmp_path / "x.pkl"))
    with pytest.raises(FileNotFoundError, match="x.pkl"):
        embedder.resolve_input_path()


def test_input_override_wins_over_the_configured_path(tmp_path) -> None:
    configured = _touch(tmp_path / "configured.pkl")
    override = _touch(tmp_path / "override.pkl")
    embedder = _embedder(tmp_path, datamol_conformer_input_pkl_path=str(configured))
    assert embedder.resolve_input_path(str(override)) == override
    assert embedder.resolve_input_path() == configured


def test_superset_smiles_keys_are_hashed_unique_and_sorted_by_hash(tmp_path) -> None:
    path = _pickle(
        tmp_path / "s.pkl",
        [{"smiles": "[O-]", "smiles_key": "O"}, {"smiles": "C", "smiles_key": "C"}, {"smiles": "O", "smiles_key": "O"}],
    )
    result = _embedder(tmp_path)._load_smiles_df(path)
    hashes = {s: hashlib.sha256(s.encode()).hexdigest() for s in ("C", "O")}
    expected = sorted(hashes.items(), key=lambda item: item[1])
    assert result["smiles_key"].tolist() == [s for s, _ in expected]
    assert result["smiles_hash"].tolist() == [h for _, h in expected]


def test_only_existing_sdf_files_within_the_conformer_count_become_stems(tmp_path) -> None:
    embedder = _embedder(tmp_path)
    pubchem = _touch(embedder._pubchem_sdf_path(HASH_A))
    conf0 = _touch(embedder._datamol_sdf_path(HASH_A, 0))
    _touch(embedder._datamol_sdf_path(HASH_A, 2))  # beyond datamol_num_conformers
    frame = pd.DataFrame({"smiles_hash": [HASH_A, HASH_B]})
    assert embedder._build_relevant_sdf_stems(frame) == {
        pubchem.stem: pubchem,
        conf0.stem: conf0,
    }


def test_two_dimensional_embeddings_are_mean_pooled_to_float16(tmp_path) -> None:
    prepared = _embedder(tmp_path)._prepare_store_arrays(
        {"a": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([5.0, 6.0])}
    )
    assert prepared["a"].tolist() == [2.0, 3.0]
    assert prepared["b"].tolist() == [5.0, 6.0]
    assert all(v.dtype == np.float16 for v in prepared.values())


def test_embeddings_with_more_than_two_dimensions_are_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unexpected embedding shape for key 'a'"):
        _embedder(tmp_path)._prepare_store_arrays({"a": np.zeros((2, 2, 2))})


def test_items_are_batched_with_their_batch_index(tmp_path) -> None:
    batches = list(_embedder(tmp_path)._iter_batches(["a", "b", "c", "d", "e"], 2))
    assert batches == [(0, ["a", "b"]), (1, ["c", "d"]), (2, ["e"])]


def test_new_store_uses_the_unimol_metadata(tmp_path) -> None:
    store = _embedder(tmp_path)._build_store()
    meta = json.loads((tmp_path / "store" / "meta.json").read_text())
    assert store.root_dir == tmp_path / "store"
    assert (meta["embedder"], meta["model_name"], meta["storage_dtype"]) == (
        "unimol_sdf",
        "unimolv2",
        "float16",
    )


def _pickle(path, items) -> str:
    path.write_bytes(pickle.dumps(items))
    return str(path)


def _dataset_with_sdf(tmp_path, embedder, smiles) -> tuple[str, list[str]]:
    hashes = [hashlib.sha256(s.encode()).hexdigest() for s in smiles]
    for smi_hash in hashes:
        _touch(embedder._pubchem_sdf_path(smi_hash))
    return _pickle(tmp_path / "s.pkl", [{"smiles": s, "smiles_key": s} for s in smiles]), sorted(hashes)


def test_embed_processes_missing_stems_in_batches(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path, unimol_stems_per_batch=1)
    path, (hash_a, hash_b) = _dataset_with_sdf(tmp_path, embedder, ["C", "O"])
    seen = []

    def fake_process(batch_index, batch_stems, **_kwargs):
        seen.append((batch_index, batch_stems))
        return len(batch_stems)

    monkeypatch.setattr(embedder, "_process_batch", fake_process)
    embedder.embed(input_override=path)
    assert seen == [(0, [f"pubchem_{hash_a}"]), (1, [f"pubchem_{hash_b}"])]


def test_embed_skips_stems_that_the_store_already_holds(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path)
    path, (hash_a, hash_b) = _dataset_with_sdf(tmp_path, embedder, ["C", "O"])
    store = embedder._build_store()
    store.put_many({f"pubchem_{hash_a}": np.ones(2, dtype=np.float16)}, flush=True)
    seen = []
    monkeypatch.setattr(
        embedder,
        "_process_batch",
        lambda batch_index, batch_stems, **_: seen.append(batch_stems) or 0,
    )
    embedder.embed(input_override=path)
    assert seen == [[f"pubchem_{hash_b}"]]


def test_embed_does_nothing_when_the_store_is_complete(tmp_path, monkeypatch) -> None:
    embedder = _embedder(tmp_path)
    path, (hash_a,) = _dataset_with_sdf(tmp_path, embedder, ["C"])
    embedder._build_store().put_many(
        {f"pubchem_{hash_a}": np.ones(2, dtype=np.float16)}, flush=True
    )
    monkeypatch.setattr(embedder, "_process_batch", lambda **_: pytest.fail("batch was processed"))
    embedder.embed(input_override=path)
    assert ShardedEmbeddingReader(tmp_path / "store").contains(f"pubchem_{hash_a}")


def _gpu_embedder(tmp_path, gpu_ids=(0, 1)) -> UnimolSDFEmbedder:
    cfg = OmegaConf.create(
        {
            "embeddings": {
                "unimol_gpu_ids": gpu_ids if isinstance(gpu_ids, int) else list(gpu_ids),
                "unimol_batch_size": 8,
                "unimol_model_size": "84m",
                "log_dir": str(tmp_path / "logs"),
            }
        }
    )
    return UnimolSDFEmbedder(cfg)


def test_gpu_ids_become_strings_and_a_single_id_becomes_a_list(tmp_path) -> None:
    assert _gpu_embedder(tmp_path, (0, 3)).GPU_IDS == ["0", "3"]
    assert _gpu_embedder(tmp_path, 2).GPU_IDS == ["2"]


def test_init_creates_the_log_directory(tmp_path) -> None:
    embedder = _gpu_embedder(tmp_path)
    assert (tmp_path / "logs").is_dir()
    assert (embedder.BATCH_SIZE, embedder.MODEL_SIZE) == (8, "84m")


def test_embed_by_atom_count_batches_one_atom_count_and_keeps_order() -> None:
    feats = [{"src_tokens": [0] * n, "id": i} for i, n in enumerate([5, 3, 5, 4, 3, 5])]
    seen: list[list[int]] = []

    def infer(subset):
        seen.append([len(f["src_tokens"]) for f in subset])
        return [f["id"] for f in subset]

    assert unimol._embed_by_atom_count(feats, infer) == [0, 1, 2, 3, 4, 5]
    assert all(len(set(sizes)) == 1 for sizes in seen)
    assert sorted(len(sizes) for sizes in seen) == [1, 2, 3]


def test_featurize_never_deletes_inputs(tmp_path: Path) -> None:
    assert not hasattr(unimol, "_delete_invalid_sdf")
    missing = tmp_path / "gone.sdf"
    stem, feats, valid, error = unimol._featurize_sdf(str(missing))
    assert (stem, feats, valid) == ("gone", [], 0)
    assert "Unreadable SDF" in error


def _write_sdfs(tmp_path: Path, smiles_by_stem: dict[str, list[str]]) -> list[Path]:
    paths = []
    for stem, smiles in smiles_by_stem.items():
        path = tmp_path / f"{stem}.sdf"
        writer = Chem.SDWriter(str(path))
        for smi in smiles:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(mol, randomSeed=7)
            writer.write(mol)
        writer.close()
        paths.append(path)
    return paths


def _same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return np.array_equal(np.asarray(a), np.asarray(b))


def test_in_memory_featurization_matches_the_old_checkpoint_path(tmp_path: Path) -> None:
    """Old path: Phase 0 sets max_atoms to the next power of two of the largest molecule, so never crops."""
    from unimol_tools.data.conformer import mol2unimolv2

    paths = _write_sdfs(
        tmp_path, {"a": ["CCO", "c1ccccc1O"], "b": ["CC(=O)OC1=CC=CC=C1C(=O)O"], "c": ["C"]}
    )
    reference = {}
    for path in paths:
        for i, mol in enumerate(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)):
            reference[(path.stem, i)] = mol2unimolv2(mol, max_atoms=16, remove_hs=True)

    cfg = OmegaConf.create(
        {"embeddings": {"unimol_num_workers": 2, "log_dir": str(tmp_path / "logs")}}
    )
    data = unimol.UnimolSDFFeaturizer(cfg).featurize(paths)

    assert set(data) == {"feats_sorted", "stems_sorted", "recidx_sorted"}
    keys = list(zip(data["stems_sorted"], data["recidx_sorted"]))
    assert sorted(keys) == sorted(reference)
    sizes = [len(f["src_tokens"]) for f in data["feats_sorted"]]
    assert sizes == sorted(sizes)
    for key, feat in zip(keys, data["feats_sorted"]):
        assert _same(feat, reference[key])
