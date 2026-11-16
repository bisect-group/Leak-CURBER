from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock

from src.data.components.embedders.esm3 import ESM3StructureShardEmbedder, _featurize_esm3_record

SEQUENCE = "MSG"
THREE_LETTER = {"M": "MET", "S": "SER", "G": "GLY"}
BACKBONE = (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"))
CIF_HEADER = (
    "data_test\n#\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
    "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
    "_atom_site.label_entity_id\n_atom_site.label_seq_id\n_atom_site.pdbx_PDB_ins_code\n"
    "_atom_site.auth_seq_id\n_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
    "_atom_site.auth_atom_id\n_atom_site.B_iso_or_equiv\n_atom_site.Cartn_x\n"
    "_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.pdbx_PDB_model_num\n_atom_site.id\n"
)


def write_esmfold2_cif(path: Path, sequence: str = SEQUENCE) -> Path:
    """Write a backbone-only single-chain CIF like the ESMFold2 generator output."""
    lines, atom_id = [], 0
    for res_id, residue in enumerate(sequence, start=1):
        for atom, element in BACKBONE:
            atom_id += 1
            x = 3.8 * res_id + 0.5 * atom_id % 1.4
            name = THREE_LETTER[residue]
            lines.append(
                f"ATOM {element} {atom} . {name} A 1 {res_id} . {res_id} {name} A {atom} "
                f"90.0 {x:.3f} {0.3 * atom_id:.3f} {0.1 * res_id:.3f} 1 {atom_id}"
            )
    path.write_text(CIF_HEADER + "\n".join(lines) + "\n")
    return path


def _embedder(tmp_path: Path) -> ESM3StructureShardEmbedder:
    dirs = {name: tmp_path / name for name in ("af", "esm", "exp", "esmfold2")}
    for path in dirs.values():
        path.mkdir()
    embedder = object.__new__(ESM3StructureShardEmbedder)
    embedder.alphafold_pdb_path, embedder.esm3_pdb_path = dirs["af"], dirs["esm"]
    embedder.mmcif_path, embedder.esmfold2_cif_path = dirs["exp"], dirs["esmfold2"]
    embedder._pdb_index = None
    embedder.logger = MagicMock()
    return embedder


def test_predicted_source_order_is_alphafold_esmfold2_esm3(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path)
    for acc in ("X1", "X2"):
        (embedder.esmfold2_cif_path / f"ESMFold2-{acc}.cif").write_text("")
    for acc in ("X1", "X2", "X3"):
        (embedder.esm3_pdb_path / f"ESM3-open-small-{acc}.pdb").write_text("")
    (embedder.alphafold_pdb_path / "AF-X1-F1-model_v4.pdb").write_text("")

    assert embedder._resolve_pdb("X1")[1] == "alphafold"
    assert embedder._resolve_pdb("X2") == (
        str(embedder.esmfold2_cif_path / "ESMFold2-X2.cif"),
        "esmfold2",
    )
    assert embedder._resolve_pdb("X3")[1] == "esm"
    assert embedder._resolve_pdb("X4") is None


def test_esmfold2_keys_carry_the_source_tag(tmp_path: Path) -> None:
    embedder = _embedder(tmp_path)
    (embedder.esmfold2_cif_path / "ESMFold2-X2.cif").write_text("")
    embedder.input_path = tmp_path / "proteins.pkl"
    embedder.input_path.write_bytes(pickle.dumps([{"pdbs": ["X2"]}, {"pdbs": ["X3"]}]))

    keys, failures = embedder._load_unique_input_keys()

    assert keys == ["X2|esmfold2", "X3"] and failures == []


def test_featurize_reads_an_esmfold2_cif(tmp_path: Path) -> None:
    cif = write_esmfold2_cif(tmp_path / "ESMFold2-X2.cif")

    record = _featurize_esm3_record(("X2|esmfold2", str(cif), None))

    assert record["sequence"] == SEQUENCE and record["length"] == 3
    assert record["coords"].shape == (3, 37, 3)
    assert record["residue_index"].tolist() == [1, 2, 3]
