from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.structure_utils import (
    chain_records,
    collapse_structures_to_1d,
    mmcif_file,
    select_chain,
    select_chains,
    selection_counts,
)

THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
    "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
    "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}  # fmt: skip
UNIPROT = "".join(random.Random(0).choices(sorted(THREE), k=200))


def write_cif(
    path: Path, chains: dict[str, str], accessions: dict[str, str] | None = None
) -> Path:
    """Write a CA-only mmCIF with one residue per letter; `accessions` maps chain -> UniProt."""
    lines = ["data_TEST", "#"]
    if accessions:
        lines += ["loop_", "_struct_ref.id", "_struct_ref.db_name", "_struct_ref.pdbx_db_accession"]
        refs = {acc: i for i, acc in enumerate(dict.fromkeys(accessions.values()), 1)}
        lines += [f"{i} UNP {acc}" for acc, i in refs.items()]
        lines += ["#", "loop_", "_struct_ref_seq.align_id", "_struct_ref_seq.ref_id"]
        lines += ["_struct_ref_seq.pdbx_strand_id"]
        lines += [f"{n} {refs[acc]} {chain}" for n, (chain, acc) in enumerate(accessions.items(), 1)]
        lines.append("#")
    lines += ["loop_", "_entity.id", "_entity.type"]
    lines += [f"{n} polymer" for n in range(1, len(chains) + 1)]
    lines += ["#", "loop_", "_entity_poly.entity_id", "_entity_poly.pdbx_strand_id"]
    lines += ["_entity_poly.pdbx_seq_one_letter_code_can"]
    lines += [f"{n} {chain} {seq}" for n, (chain, seq) in enumerate(chains.items(), 1)]
    lines += ["#", "loop_", "_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.seq_id"]
    lines += ["_pdbx_poly_seq_scheme.auth_seq_num", "_pdbx_poly_seq_scheme.pdb_strand_id"]
    lines += [
        f"{chain} {i} {i} {chain}" for chain, seq in chains.items() for i in range(1, len(seq) + 1)
    ]
    lines += [
        "#", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_comp_id", "_atom_site.label_asym_id",
        "_atom_site.label_seq_id", "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x",
        "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id", "_atom_site.pdbx_PDB_model_num",
    ]  # fmt: skip
    atom = 0
    for chain, sequence in chains.items():
        for number, letter in enumerate(sequence, 1):
            atom += 1
            lines.append(
                f"ATOM {atom} C CA {THREE[letter]} {chain} {number} ? {3.8 * number:.1f} 0.0 0.0 "
                f"{number} {chain} 1"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def mutate(sequence: str, every: int) -> str:
    return "".join("W" if i % every == 0 and c != "W" else c for i, c in enumerate(sequence))


def test_chain_records_read_accessions_and_observed_sequence(tmp_path: Path) -> None:
    cif = write_cif(tmp_path / "1ABC.cif", {"AAA": UNIPROT}, {"AAA": "P1"})
    assert chain_records(cif) == {"AAA": {"accessions": {"P1"}, "sequence": UNIPROT}}


def test_accession_match_selects_the_mapped_chain(tmp_path: Path) -> None:
    cif = write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT, "B": UNIPROT}, {"A": "P2", "B": "P1"})
    assert select_chain(cif, "P1", UNIPROT) == "B"


def test_isoform_root_matches(tmp_path: Path) -> None:
    cif = write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT, "B": UNIPROT}, {"A": "P2", "B": "P1-2"})
    assert select_chain(cif, "P1", UNIPROT) == "B"


def test_sequence_match_without_accession_match(tmp_path: Path) -> None:
    write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT[::-1], "C": UNIPROT}, {"A": "P2"})
    selections = select_chains([("1ABC", "P1")], {"P1": UNIPROT}, tmp_path)
    assert selections == {("1ABC", "P1"): ("C", "sequence", 1.0)}


def test_fragment_below_coverage_is_rejected(tmp_path: Path) -> None:
    cif = write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT[:50]}, {"A": "P1"})
    assert select_chain(cif, "P1", UNIPROT) is None


def test_fragment_above_the_floor_is_selected_with_its_coverage(tmp_path: Path) -> None:
    write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT[:100]}, {"A": "P1"})
    assert select_chains([("1ABC", "P1")], {"P1": UNIPROT}, tmp_path, workers=1) == {
        ("1ABC", "P1"): ("A", "accession", 0.5)
    }


def test_ortholog_below_identity_is_rejected(tmp_path: Path) -> None:
    cif = write_cif(tmp_path / "1ABC.cif", {"A": mutate(UNIPROT, 5)}, {"A": "P1"})
    assert select_chain(cif, "P1", UNIPROT) is None


def test_point_mutant_and_tag_are_accepted(tmp_path: Path) -> None:
    mutant = "MHHHHHH" + UNIPROT[:50] + "W" + UNIPROT[51:]
    cif = write_cif(tmp_path / "1ABC.cif", {"A": mutant}, {"A": "P1"})
    assert select_chain(cif, "P1", UNIPROT) == "A"


def test_homodimer_keeps_the_best_chain_then_chain_order(tmp_path: Path) -> None:
    same = write_cif(tmp_path / "1AAA.cif", {"B": UNIPROT, "A": UNIPROT}, {"A": "P1", "B": "P1"})
    assert select_chain(same, "P1", UNIPROT) == "A"
    shorter = write_cif(
        tmp_path / "1BBB.cif", {"A": UNIPROT[:-30], "B": UNIPROT}, {"A": "P1", "B": "P1"}
    )
    assert select_chain(shorter, "P1", UNIPROT) == "B"


def test_select_chains_counts_missing_and_reuses_the_cache(tmp_path: Path) -> None:
    write_cif(tmp_path / "1ABC_updated.cif", {"A": UNIPROT}, {"A": "P1"})
    write_cif(tmp_path / "2ABC.cif", {"A": UNIPROT[:50]}, {"A": "P1"})
    assert mmcif_file(tmp_path, "1abc").name == "1ABC_updated.cif"
    pairs = [("1ABC", "P1"), ("2ABC", "P1"), ("3ABC", "P1"), ("1ABC", "P9")]
    sequences = {"P1": UNIPROT}

    first = select_chains(pairs, sequences, tmp_path, workers=1)
    assert first == {
        ("1ABC", "P1"): ("A", "accession", 1.0),
        ("2ABC", "P1"): (None, "rejected", 0.0),
        ("3ABC", "P1"): (None, "missing", 0.0),
        ("1ABC", "P9"): (None, "rejected", 0.0),
    }
    assert dict(selection_counts(first)) == {"accession": 1, "rejected": 2, "missing": 1}
    # The cache name carries the thresholds and the entries carry the coverage.
    cache = json.loads((tmp_path / "chain_selection_id90_cov30.json").read_text())
    assert cache["1ABC:P1"] == ["A", "accession", 1.0]

    # Without the file, a cached pair still resolves: nothing is read or aligned again.
    (tmp_path / "1ABC_updated.cif").unlink()
    assert select_chains(pairs, sequences, tmp_path, workers=1) == first


def test_unreadable_file_gets_error_basis_and_is_not_cached(tmp_path: Path) -> None:
    (tmp_path / "1ABC.cif").write_text("not an mmcif file")
    sequences = {"P1": UNIPROT}

    result = select_chains([("1ABC", "P1")], sequences, tmp_path, workers=1)
    assert result == {("1ABC", "P1"): (None, "error", 0.0)}
    assert dict(selection_counts(result)) == {"error": 1}

    write_cif(tmp_path / "1ABC.cif", {"A": UNIPROT}, {"A": "P1"})
    assert select_chains([("1ABC", "P1")], sequences, tmp_path, workers=1) == {
        ("1ABC", "P1"): ("A", "accession", 1.0)
    }


def test_collapse_structures_to_1d_dedups_and_empties() -> None:
    df = pd.DataFrame({"k": ["a", "a", "a", "b"], "pdbs": ["1ABC", "1ABC", "2XYZ", np.nan]})
    out = collapse_structures_to_1d(df, ["k"]).set_index("k")["pdbs"]
    assert out["a"] == ["1ABC", "2XYZ"]
    assert out["b"] == []
