from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from src.data.components.embedders.store import ShardedEmbeddingReader, ShardedEmbeddingStore
from src.data.components.embedders.types import CacheMetadata


def _esm3_embedder(tmp_path: Path, with_esm_file: bool):
    from src.data.components.embedders.esm3 import ESM3StructureShardEmbedder

    dirs = {name: tmp_path / name for name in ("af", "esm", "exp", "esmfold2")}
    for path in dirs.values():
        path.mkdir(exist_ok=True)
    (dirs["af"] / "AF-X1-F1-model_v4.pdb").write_text("")
    if with_esm_file:
        (dirs["esm"] / "X1.pdb").write_text("")
    embedder = object.__new__(ESM3StructureShardEmbedder)
    embedder.alphafold_pdb_path, embedder.esm3_pdb_path = dirs["af"], dirs["esm"]
    embedder.mmcif_path, embedder._pdb_index = dirs["exp"], None
    embedder.esmfold2_cif_path = dirs["esmfold2"]
    embedder.logger = MagicMock()
    return embedder


def test_mean_pool_pools_residues_only() -> None:
    from src.data.components.embedders.esm3 import _mean_pool_residues

    tokens = torch.tensor([[100.0], [1.0], [3.0], [100.0], [100.0]])  # BOS, 2 residues, EOS, pad
    assert _mean_pool_residues(tokens, 2).tolist() == [2.0]


def test_key_names_the_pdb_source(tmp_path: Path) -> None:
    only_af = _esm3_embedder(tmp_path, with_esm_file=False)._resolve_pdb("X1")
    with_esm = _esm3_embedder(tmp_path, with_esm_file=True)._resolve_pdb("X1")
    assert only_af[1] == "alphafold"
    assert with_esm[1] == "esm"
    assert _esm3_embedder(tmp_path, True)._resolve_pdb("nope") is None


def test_reader_finds_vectors_by_accession(tmp_path: Path) -> None:
    from src.data.components.embedders.utils import PDB_SOURCE_KEY_TYPE

    metadata = CacheMetadata(
        embedder="t",
        model_name="m",
        version="v1",
        storage_dtype="float32",
        embedding_dim=None,
        max_shard_bytes=1 << 20,
        key_type=PDB_SOURCE_KEY_TYPE,
        key_field="acc_id",
    )
    store = ShardedEmbeddingStore(tmp_path, metadata, logging.getLogger("t"))
    store.put_many({"X1|esm": np.full(2, 1.0, dtype=np.float32)}, flush=True)
    reader = ShardedEmbeddingReader(tmp_path)
    assert reader.get("X1").tolist() == [1.0, 1.0]
    assert reader.get("X1|esm").tolist() == [1.0, 1.0]
    assert reader.get("X1|alphafold") is None and reader.get("X2") is None


def test_experimental_key_resolves_to_the_selected_chain(tmp_path: Path) -> None:
    from tests.utils.test_structure_utils import UNIPROT, write_cif

    embedder = _esm3_embedder(tmp_path, with_esm_file=False)
    cif = write_cif(embedder.mmcif_path / "1ABC_updated.cif", {"A": UNIPROT[::-1], "BB": UNIPROT})
    embedder.uniprot_acc_path = tmp_path / "acc"
    embedder.uniprot_acc_path.mkdir()
    (embedder.uniprot_acc_path / "P1.json").write_text(json.dumps({"sequence": UNIPROT}))

    keys = ["1ABC:P1|experimental", "1ABC:P2|experimental", "X1|alphafold"]
    chains = embedder._select_experimental_chains(keys)

    # P1 matches by sequence; P2 has no cached UniProt sequence, so no chain is selected.
    assert chains == {"1ABC:P1": "BB", "1ABC:P2": None}
    assert embedder._resolve_pdb("1ABC:P1") == (str(cif), "experimental")
    assert embedder._resolve_pdb("9XYZ:P1") is None


def test_experimental_key_hash_has_one_candidate() -> None:
    from src.data.components.embedders.utils import PDB_SOURCE_KEY_TYPE, key_hash_candidates, sha256_short

    assert key_hash_candidates("1ABC:P1", PDB_SOURCE_KEY_TYPE) == [sha256_short("1ABC:P1|experimental")]
    assert len(key_hash_candidates("P1", PDB_SOURCE_KEY_TYPE)) == 4
    assert key_hash_candidates("P12345|R1Q", PDB_SOURCE_KEY_TYPE) == [
        sha256_short(f"P12345|R1Q|{source}") for source in ("esm", "alphafold", "experimental", "esmfold2")
    ]
    for key in ("P12345|R1Q|esm", "P12345|esm"):
        assert key_hash_candidates(key, PDB_SOURCE_KEY_TYPE) == [sha256_short(key)]


def test_featurize_reads_the_selected_mmcif_chain(tmp_path: Path) -> None:
    from src.data.components.embedders.esm3 import _featurize_esm3_record
    from tests.utils.test_structure_utils import UNIPROT, write_cif

    cif = write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT[::-1], "BB": UNIPROT})
    record = _featurize_esm3_record(("1ABC:P1|experimental", str(cif), "BB"))
    assert record.get("error") is None, record
    assert record["sequence"] == UNIPROT
    assert record["coords"].shape == (len(UNIPROT), 37, 3)


def test_structure_keys_follow_the_pdb_type() -> None:
    from src.data.components.embedders.utils import structure_keys

    experimental = {"uniprot_id": ["P1", "P1|R2Q"], "pdbs": ["1ABC:P1", "2DEF:P1"], "pdb_type": "experimental"}
    predicted = {"uniprot_id": ["P3"], "pdbs": ["P3", "Q9"], "pdb_type": "predicted"}
    unresolved = {"uniprot_id": ["P4|K1A"], "pdbs": ["P4|K1A"], "pdb_type": None}

    assert structure_keys(experimental) == ["1ABC:P1", "2DEF:P1"]
    assert structure_keys(predicted) == ["P3", "Q9"]
    assert structure_keys(unresolved) == ["P4|K1A"]


def test_embedder_input_keys_come_from_protein_records(tmp_path: Path) -> None:
    import pickle

    embedder = _esm3_embedder(tmp_path, with_esm_file=False)
    embedder.input_path = tmp_path / "all_proteins.pkl"
    records = [
        {"sequence": "A", "uniprot_id": ["P1"], "pdbs": ["1ABC:P1"], "pdb_type": "experimental"},
        {"sequence": "B", "uniprot_id": ["X1"], "pdbs": ["X1"], "pdb_type": "predicted"},
        {"sequence": "B", "uniprot_id": ["X9"], "pdbs": ["X9"], "pdb_type": None},
    ]
    embedder.input_path.write_bytes(pickle.dumps(records))
    (embedder.mmcif_path / "1ABC_updated.cif").write_text("")

    keys, failures = embedder._load_unique_input_keys()

    # X9 has no structure file, so its key stays bare.
    assert keys == ["1ABC:P1|experimental", "X1|alphafold", "X9"]
    assert failures == []
