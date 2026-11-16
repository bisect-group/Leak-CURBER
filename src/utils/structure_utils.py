"""Choose the mmCIF chain that holds a given UniProt protein."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import biotite.structure.info as struc_info
import biotite.structure.io.pdbx as pdbx
import pandas as pd
from Bio.Align import PairwiseAligner

MIN_IDENTITY = 0.9
MIN_COVERAGE = 0.8  # a PDB with at least this coverage is always kept
MIN_FALLBACK_COVERAGE = 0.3  # lowest coverage of a selected chain
MAX_WORKERS = 32

# Selection result per (PDB ID, accession): the chain, how it was found and its coverage.
Pair = tuple[str, str]
Selection = tuple[str | None, str, float]  # basis: accession, sequence, rejected, missing, error


def mmcif_file(mmcif_dir: Path, pdb_id: str) -> Path | None:
    for name in (f"{pdb_id.upper()}_updated.cif", f"{pdb_id.upper()}.cif"):
        if (path := Path(mmcif_dir) / name).exists():
            return path
    return None


def uniprot_root(accession: str) -> str:
    """Drop the isoform suffix: P12345-2 -> P12345."""
    return re.sub(r"-\d+$", "", accession.strip())


@lru_cache(maxsize=None)
def _one_letter(res_name: str) -> str:
    return struc_info.one_letter_code(res_name) or "X"


def chain_records(cif_path: Path) -> dict[str, dict]:
    """Return {author chain: {"accessions": set, "sequence": str}} for protein chains.

    The sequence holds the residues with coordinates; accessions come from the UNP
    entries of `_struct_ref` and `_struct_ref_seq`.
    """
    cif = pdbx.CIFFile.read(cif_path)
    block = cif.block
    accessions: dict[str, set[str]] = {}
    if "struct_ref" in block and "struct_ref_seq" in block:
        ref = block["struct_ref"]
        ref_ids = ref["id"].as_array(str)
        isoforms = (
            ref["pdbx_db_isoform"].as_array(str)
            if "pdbx_db_isoform" in ref
            else [""] * len(ref_ids)
        )
        unp = {
            ref_id: {uniprot_root(v) for v in (acc, isoform) if v not in ("?", ".", "")}
            for ref_id, db, acc, isoform in zip(
                ref_ids,
                ref["db_name"].as_array(str),
                ref["pdbx_db_accession"].as_array(str),
                isoforms,
            )
            if db == "UNP"
        }
        seq = block["struct_ref_seq"]
        for ref_id, strands in zip(
            seq["ref_id"].as_array(str), seq["pdbx_strand_id"].as_array(str)
        ):
            for strand in strands.split(","):
                accessions.setdefault(strand.strip(), set()).update(unp.get(ref_id, ()))

    atoms = pdbx.get_structure(cif, model=1, use_author_fields=True)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    starts = struc.get_residue_starts(atoms)
    chain_ids = atoms.chain_id[starts]
    names = atoms.res_name[starts]
    records = {}
    for chain in dict.fromkeys(chain_ids.tolist()):
        sequence = "".join(_one_letter(n) for n in names[chain_ids == chain])
        records[chain] = {"accessions": accessions.get(chain, set()), "sequence": sequence}
    return records


def _aligner() -> PairwiseAligner:
    aligner = PairwiseAligner(mode="global")
    aligner.match_score, aligner.mismatch_score = 1, -1
    aligner.open_gap_score, aligner.extend_gap_score = -2, -0.5
    aligner.end_gap_score = 0  # tags and unmodelled ends are not penalised
    return aligner


def identity_and_coverage(uniprot_sequence: str, chain_sequence: str) -> tuple[float, float]:
    """Identity over aligned pairs and fraction of the UniProt sequence aligned."""
    alignment = _aligner().align(uniprot_sequence, chain_sequence)[0]
    pairs = identical = 0
    for (t0, t1), (q0, q1) in zip(*alignment.aligned):
        pairs += t1 - t0
        identical += sum(a == b for a, b in zip(uniprot_sequence[t0:t1], chain_sequence[q0:q1]))
    return (identical / pairs if pairs else 0.0), pairs / len(uniprot_sequence)


def _select(
    records: dict[str, dict],
    accession: str,
    uniprot_sequence: str,
    min_identity: float,
    min_coverage: float,
) -> Selection:
    root = uniprot_root(accession)
    candidates = [c for c, r in records.items() if root in r["accessions"]]
    basis = "accession"
    if not candidates:
        candidates, basis = list(records), "sequence"
    scores: dict[str, tuple[float, float]] = {}  # chains with equal sequences share a score
    accepted = []
    for chain in candidates:
        sequence = records[chain]["sequence"]
        # A chain shorter than the coverage threshold cannot pass.
        if len(sequence) < min_coverage * len(uniprot_sequence):
            continue
        if sequence not in scores:
            scores[sequence] = identity_and_coverage(uniprot_sequence, sequence)
        identity, coverage = scores[sequence]
        if identity >= min_identity and coverage >= min_coverage:
            accepted.append((-coverage, -identity, chain))
    if not accepted:
        return None, "rejected", 0.0
    neg_coverage, _, chain = min(accepted)
    return chain, basis, -neg_coverage


def select_chain(
    cif_path: Path,
    accession: str,
    uniprot_sequence: str,
    min_identity: float = MIN_IDENTITY,
    min_coverage: float = MIN_FALLBACK_COVERAGE,
) -> str | None:
    """Chain of the file that holds the accession, or None when no chain is accepted."""
    return _select(
        chain_records(cif_path), accession, uniprot_sequence, min_identity, min_coverage
    )[0]


def _select_file(args: tuple[str, Path, list[tuple[str, str]], float, float]):
    pdb_id, path, items, min_identity, min_coverage = args
    try:
        records = chain_records(path)
    except Exception:
        return [((pdb_id, acc), (None, "error", 0.0)) for acc, _ in items]
    return [
        ((pdb_id, acc), _select(records, acc, seq, min_identity, min_coverage))
        for acc, seq in items
    ]


def select_chains(
    pairs: Iterable[Pair],
    uniprot_sequences: dict[str, str],
    mmcif_dir: Path,
    min_identity: float = MIN_IDENTITY,
    min_coverage: float = MIN_FALLBACK_COVERAGE,
    workers: int | None = None,
) -> dict[Pair, Selection]:
    """Select the best-covering chain for each unique (PDB ID, accession) pair.

    A chain needs `min_identity` and `min_coverage`. Results are cached in the mmCIF
    directory per threshold pair, so a rerun aligns nothing. Pairs without a file or without a UniProt sequence are not cached.
    Pairs whose file cannot be parsed get the basis "error" and are not cached.
    """
    mmcif_dir = Path(mmcif_dir)
    cache_path = mmcif_dir / (
        f"chain_selection_id{round(min_identity * 100)}_cov{round(min_coverage * 100)}.json"
    )
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    results: dict[Pair, Selection] = {}
    todo: dict[str, list[tuple[str, str]]] = {}
    for pdb_id, acc in dict.fromkeys((p.upper(), a) for p, a in pairs):
        if f"{pdb_id}:{acc}" in cache:
            results[pdb_id, acc] = tuple(cache[f"{pdb_id}:{acc}"])
        elif acc not in uniprot_sequences:
            results[pdb_id, acc] = (None, "rejected", 0.0)
        else:
            todo.setdefault(pdb_id, []).append((acc, uniprot_sequences[acc]))

    jobs = []
    for pdb_id, items in todo.items():
        if (path := mmcif_file(mmcif_dir, pdb_id)) is None:
            results.update(((pdb_id, acc), (None, "missing", 0.0)) for acc, _ in items)
        else:
            jobs.append((pdb_id, path, items, min_identity, min_coverage))

    computed = {}
    if jobs:
        workers = min(workers or os.cpu_count() or 1, MAX_WORKERS, len(jobs))
        with get_context("fork").Pool(workers) as pool:
            for batch in pool.imap_unordered(_select_file, jobs, chunksize=8):
                computed.update(batch)
        cache.update(
            {f"{p}:{a}": list(sel) for (p, a), sel in computed.items() if sel[1] != "error"}
        )
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(cache))
        tmp_path.replace(cache_path)
    return results | computed


def selection_counts(selections: dict[Pair, Selection]) -> Counter:
    """Counts of pairs per basis: accession, sequence, rejected, missing, error."""
    return Counter(basis for _, basis, _ in selections.values())



def collapse_structures_to_1d(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """One row per `key_cols` combination; `pdbs` lists its structure ids, [] when it has none."""
    pdbs = (
        df.dropna(subset=["pdbs"])
        .drop_duplicates([*key_cols, "pdbs"])
        .groupby(key_cols, dropna=False, sort=False)["pdbs"]
        .agg(list)
        .reset_index()
    )
    out = df[key_cols].drop_duplicates().merge(pdbs, on=key_cols, how="left")
    out["pdbs"] = out["pdbs"].map(lambda ids: ids if isinstance(ids, list) else [])
    return out
