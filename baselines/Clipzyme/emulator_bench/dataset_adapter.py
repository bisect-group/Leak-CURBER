from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNS_ROOT,
    ensure_dir,
    metadata_path_for_split,
    resolve_path,
    sha256_text,
    split_group_slug,
    write_json,
)


REQUIRED_COLUMNS = {"rxn_smiles", "ec_number", "sequence"}
SPLIT_NAMES = ("train", "val", "test")
CLIPZYME_SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}
DEFAULT_MAX_SEQUENCE_LENGTH = 650


@dataclass(frozen=True)
class SplitGroup:
    name: str
    path: Path


def discover_split_groups(dataset_root: str | Path) -> list[SplitGroup]:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    groups = []
    for train_file in sorted(root.rglob("train.parquet")):
        parent = train_file.parent
        if all((parent / f"{split}.parquet").exists() for split in SPLIT_NAMES):
            groups.append(SplitGroup(name=parent.relative_to(root).as_posix(), path=parent))
    if not groups:
        raise FileNotFoundError(f"No train/val/test parquet split groups found under {root}")
    return groups


def select_split_groups(dataset_root: str | Path, requested: list[str] | None) -> list[SplitGroup]:
    groups = discover_split_groups(dataset_root)
    if not requested:
        return groups
    by_name = {group.name: group for group in groups}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(
            f"Unknown split group(s): {missing}. Available: {sorted(by_name.keys())}"
        )
    return [by_name[name] for name in requested]


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text.lower() in {"", "nan", "none", "null"}


def _split_label_values(value: object) -> list[str]:
    if _is_missing_value(value):
        return []
    if isinstance(value, (list, tuple, set)):
        labels = []
        for item in value:
            labels.extend(_split_label_values(item))
        return labels
    return [part.strip() for part in re.split(r"[;,]", str(value)) if part.strip()]


def normalize_ec_labels(value: object, policy: str = "remove") -> list[str]:
    normalized = []
    for label in _split_label_values(value):
        lowered = label.lower()
        if lowered in {"nan", "none", "null", "-", "-.-.-.-"}:
            continue
        parts = [part.strip() for part in label.split(".")]
        has_missing = any(part in {"", "-"} for part in parts)
        if policy == "remove":
            if has_missing or len(parts) != 4:
                continue
            normalized.append(".".join(parts))
        elif policy == "truncate":
            kept = []
            for part in parts:
                if part in {"", "-"}:
                    break
                kept.append(part)
            if kept:
                normalized.append(".".join(kept))
        elif policy == "keep":
            normalized.append(label)
        else:
            raise ValueError(f"Unsupported label policy: {policy}")
    return sorted(set(normalized))


def _normalize_sequence(value: object, max_sequence_length: int) -> tuple[str, int, bool]:
    sequence = re.sub(r"\s+", "", str(value)).upper()
    original_length = len(sequence)
    if max_sequence_length > 0 and original_length > max_sequence_length:
        return sequence[:max_sequence_length], original_length, True
    return sequence, original_length, False


def _normalize_reaction(value: object) -> str:
    return str(value).strip()


def _valid_reaction(value: str) -> bool:
    if not value or value.lower() in {"nan", "none", "null"}:
        return False
    parts = value.split(">>")
    return len(parts) == 2 and bool(parts[0].strip()) and bool(parts[1].strip())


def reaction_id_for_smiles(smiles: str) -> str:
    return f"rxn_{sha256_text(smiles)[:20]}"


def protein_id_for_sequence(sequence: str) -> str:
    return f"seq_{sha256_text(sequence)[:20]}"


def record_id_for_content(reaction_smiles: str, sequence: str, ec_number: str) -> str:
    joined = f"{reaction_smiles}\t{sequence}\t{ec_number}"
    return f"rec_{sha256_text(joined)[:24]}"


def _validate_columns(path: Path, columns: list[str]) -> None:
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_clipzyme_records(
    parquet_path: str | Path,
    *,
    split_name: str,
    label_policy: str = "remove",
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    parquet_path = Path(parquet_path)
    raw_df = pd.read_parquet(parquet_path)
    _validate_columns(parquet_path, list(raw_df.columns))
    raw_rows = len(raw_df)

    records = []
    missing_required_rows = 0
    invalid_reaction_rows = 0
    empty_sequence_rows = 0
    dropped_label_rows = 0
    truncated_rows = 0

    columns = ["rxn_smiles", "sequence", "ec_number"]
    source_df = raw_df.loc[:, columns]
    limit_sampling = False
    if limit is not None and raw_rows > limit:
        candidate_count = min(raw_rows, max(limit * 20, limit))
        sampled_indices = np.linspace(0, raw_rows - 1, candidate_count, dtype=int)
        sampled_indices = sorted(set(int(idx) for idx in sampled_indices))
        source_df = source_df.iloc[sampled_indices]
        limit_sampling = True
    rows_scanned = 0
    for row in tqdm(
        source_df.itertuples(index=True),
        total=len(source_df),
        desc=f"{split_name} rows",
        leave=False,
    ):
        rows_scanned += 1
        row_idx = int(row.Index)
        rxn_value = row.rxn_smiles
        sequence_value = row.sequence
        ec_value = row.ec_number
        if any(_is_missing_value(value) for value in (rxn_value, sequence_value, ec_value)):
            missing_required_rows += 1
            continue
        reaction_smiles = _normalize_reaction(rxn_value)
        if not _valid_reaction(reaction_smiles):
            invalid_reaction_rows += 1
            continue
        sequence, original_sequence_length, truncated = _normalize_sequence(
            sequence_value,
            max_sequence_length,
        )
        if not sequence:
            empty_sequence_rows += 1
            continue
        labels = normalize_ec_labels(ec_value, policy=label_policy)
        if not labels:
            dropped_label_rows += 1
            continue
        if truncated:
            truncated_rows += 1

        reaction_id = reaction_id_for_smiles(reaction_smiles)
        protein_id = protein_id_for_sequence(sequence)
        for ec_number in labels:
            records.append(
                {
                    "record_id": record_id_for_content(reaction_smiles, sequence, ec_number),
                    "reaction_id": reaction_id,
                    "protein_id": protein_id,
                    "reaction_smiles": reaction_smiles,
                    "sequence": sequence,
                    "ec_number": ec_number,
                    "reaction_sha256": sha256_text(reaction_smiles),
                    "sequence_sha256": sha256_text(sequence),
                    "original_sequence_length": original_sequence_length,
                    "sequence_length": len(sequence),
                    "sequence_truncated": truncated,
                    "source_row": int(row_idx),
                }
            )

    before_dedup = len(records)
    if records:
        df = pd.DataFrame(records)
        dedup_df = (
            df.drop_duplicates(["reaction_smiles", "sequence", "ec_number"])
            .sort_values(["reaction_id", "protein_id", "ec_number"])
            .reset_index(drop=True)
        )
    else:
        dedup_df = pd.DataFrame(
            columns=[
                "record_id",
                "reaction_id",
                "protein_id",
                "reaction_smiles",
                "sequence",
                "ec_number",
                "reaction_sha256",
                "sequence_sha256",
                "original_sequence_length",
                "sequence_length",
                "sequence_truncated",
                "source_row",
            ]
        )

    rows_after_dedup = len(dedup_df)
    if limit is not None:
        dedup_df = dedup_df.head(limit).copy()

    stats = {
        "split": split_name,
        "raw_rows": int(raw_rows),
        "rows_scanned": int(rows_scanned),
        "limit_sampling": bool(limit_sampling),
        "missing_required_rows": int(missing_required_rows),
        "invalid_reaction_rows": int(invalid_reaction_rows),
        "empty_sequence_rows": int(empty_sequence_rows),
        "dropped_label_rows": int(dropped_label_rows),
        "truncated_rows_before_dedup": int(truncated_rows),
        "rows_after_filter": int(before_dedup),
        "rows_after_dedup": int(rows_after_dedup),
        "rows_after_limit": int(len(dedup_df)),
        "duplicate_exact_rows": int(before_dedup - rows_after_dedup),
        "unique_reactions": int(dedup_df["reaction_id"].nunique()) if not dedup_df.empty else 0,
        "unique_proteins": int(dedup_df["protein_id"].nunique()) if not dedup_df.empty else 0,
        "unique_ec_labels": int(dedup_df["ec_number"].nunique()) if not dedup_df.empty else 0,
        "truncated_sequences_after_dedup": int(dedup_df["sequence_truncated"].sum())
        if not dedup_df.empty
        else 0,
        "max_original_sequence_length": int(dedup_df["original_sequence_length"].max())
        if not dedup_df.empty
        else 0,
        "max_sequence_length": int(max_sequence_length),
        "label_policy": label_policy,
        "limit": limit,
    }
    return dedup_df, stats


def atom_map_cache_path(cache_root: str | Path, reaction_sha256: str) -> Path:
    return Path(cache_root) / "atom_maps" / reaction_sha256[:2] / f"{reaction_sha256}.json"


def _write_atom_map_cache(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_atom_map_cache(path: Path) -> dict:
    return json.loads(path.read_text())


def _is_standalone_hydrogen_component(component: str) -> bool:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(component)
    if mol is None or mol.GetNumAtoms() != 1:
        return False
    return mol.GetAtomWithIdx(0).GetAtomicNum() == 1


def _split_reaction_components(
    reaction: str,
    *,
    drop_standalone_hydrogen: bool = True,
) -> tuple[list[str], list[str]]:
    reactants, products = reaction.split(">>")
    split_reactants = [part for part in reactants.split(".") if part]
    split_products = [part for part in products.split(".") if part]
    if drop_standalone_hydrogen:
        split_reactants = [
            part for part in split_reactants if not _is_standalone_hydrogen_component(part)
        ]
        split_products = [
            part for part in split_products if not _is_standalone_hydrogen_component(part)
        ]
    return split_reactants, split_products


def _sanitize_mapped_reaction(mapped_reaction: str) -> str:
    reactants, products = _split_reaction_components(mapped_reaction)
    return ".".join(reactants) + ">>" + ".".join(products)


def _reaction_has_complete_atom_maps(mapped_reaction: str) -> bool:
    from rdkit import Chem

    for side in mapped_reaction.split(">>"):
        mol = Chem.MolFromSmiles(side)
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            if not atom.HasProp("molAtomMapNumber"):
                return False
            try:
                if atom.GetIntProp("molAtomMapNumber") <= 0:
                    return False
            except Exception:
                return False
    return True


def _clipzyme_graph_components(mapped_reaction: str) -> tuple[list[str], list[str], str]:
    reactants, products = _split_reaction_components(mapped_reaction)
    reactants = sorted(reactants)
    products = sorted(products)
    products = [product for product in products if product not in reactants]
    return reactants, products, ".".join(reactants) + ">>" + ".".join(products)


def _atom_map_numbers(components: list[str]) -> set[int]:
    from rdkit import Chem

    joined = ".".join(components)
    mol = Chem.MolFromSmiles(joined)
    if mol is None:
        raise ValueError(f"RDKit could not parse mapped SMILES: {joined}")
    atom_maps = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("molAtomMapNumber"):
            raise ValueError(f"SMILES contains an unmapped atom: {joined}")
        atom_maps.append(atom.GetIntProp("molAtomMapNumber"))
    if len(atom_maps) != len(set(atom_maps)):
        raise ValueError(f"SMILES contains duplicate atom-map numbers: {joined}")
    return set(atom_maps)


def _compute_bond_changes(mapped_reaction: str) -> list[list[object]]:
    from clipzyme.utils.wln_processing import get_bond_changes

    bond_changes = get_bond_changes(mapped_reaction)
    return [[str(u), str(v), float(t)] for u, v, t in sorted(bond_changes)]


def _validate_mapped_reaction(
    mapped_reaction: str,
) -> tuple[bool, list[list[object]], str | None, str | None]:
    if not _valid_reaction(mapped_reaction):
        return False, [], "mapped reaction is malformed", None
    try:
        sanitized = _sanitize_mapped_reaction(mapped_reaction)
        if not _valid_reaction(sanitized):
            return False, [], "mapped reaction has no non-hydrogen reactants or products", sanitized
        if not _reaction_has_complete_atom_maps(sanitized):
            return False, [], "mapped reaction has unmapped atoms", sanitized
        bond_changes = _compute_bond_changes(sanitized)
        if not bond_changes:
            return False, [], "mapped reaction has no computable bond changes", sanitized
        graph_reactants, graph_products, graph_reaction = _clipzyme_graph_components(sanitized)
        if not graph_reactants or not graph_products:
            return False, [], "mapped reaction has no Clipzyme graph products", sanitized
        reactant_maps = _atom_map_numbers(graph_reactants)
        product_maps = _atom_map_numbers(graph_products)
        if reactant_maps != product_maps:
            return (
                False,
                [],
                "mapped reactant/product atom-map sets differ after Clipzyme filtering",
                sanitized,
            )
        from clipzyme.utils.pyg import from_mapped_smiles

        reactant_graph, atom_map2new_index = from_mapped_smiles(
            ".".join(graph_reactants),
            encode_no_edge=True,
        )
        product_graph, _ = from_mapped_smiles(
            ".".join(graph_products),
            encode_no_edge=True,
        )
        if reactant_graph is None or product_graph is None:
            return False, [], "Clipzyme graph conversion returned no graph", sanitized
        if reactant_graph.x.shape[0] != product_graph.x.shape[0]:
            return (
                False,
                [],
                (
                    "Clipzyme graph node count mismatch "
                    f"({reactant_graph.x.shape[0]} reactant vs {product_graph.x.shape[0]} product)"
                ),
                sanitized,
            )
        for u, v, _ in bond_changes:
            atom_map2new_index[int(u)]
            atom_map2new_index[int(v)]
        return True, bond_changes, None, sanitized
    except Exception as exc:
        return False, [], f"mapped reaction validation failed: {exc}", None


def populate_atom_map_cache(
    reactions: pd.DataFrame,
    *,
    cache_root: str | Path,
    batch_size: int = 32,
) -> dict[str, dict]:
    cache_root = Path(cache_root)
    reaction_rows = (
        reactions.loc[:, ["reaction_sha256", "reaction_smiles"]]
        .drop_duplicates("reaction_sha256")
        .sort_values("reaction_sha256")
        .reset_index(drop=True)
    )
    cached: dict[str, dict] = {}
    missing = []
    hits = 0
    failed_cached = 0
    for row in tqdm(
        reaction_rows.itertuples(index=False),
        total=len(reaction_rows),
        desc="atom-map cache scan",
    ):
        path = atom_map_cache_path(cache_root, row.reaction_sha256)
        if path.exists():
            payload = _read_atom_map_cache(path)
            mapped_source = payload.get("raw_mapped_reaction") or payload.get("mapped_reaction")
            if mapped_source:
                ok, bond_changes, error, sanitized = _validate_mapped_reaction(
                    mapped_source
                )
                payload.update(
                    {
                        "raw_mapped_reaction": payload.get(
                            "raw_mapped_reaction", payload.get("mapped_reaction")
                        ),
                        "mapped_reaction": sanitized or payload.get("mapped_reaction"),
                        "bond_changes": bond_changes,
                        "status": "ok" if ok else "failed",
                        "error": error,
                    }
                )
                _write_atom_map_cache(path, payload)
            elif payload.get("status") == "ok":
                payload.update(
                    {
                        "bond_changes": [],
                        "status": "failed",
                        "error": "atom-map cache entry has no mapped reaction",
                    }
                )
                _write_atom_map_cache(path, payload)
            cached[row.reaction_sha256] = payload
            hits += 1
            if payload.get("status") != "ok":
                failed_cached += 1
        else:
            missing.append({"reaction_sha256": row.reaction_sha256, "reaction_smiles": row.reaction_smiles})

    if missing:
        try:
            from rxnmapper import RXNMapper
        except ImportError as exc:
            raise ImportError(
                "RXNMapper is required for unmapped Leak-CURBER reactions. "
                "Install rxnmapper in the clipzyme environment before running cache_features."
            ) from exc

        mapper = RXNMapper()
        for start in tqdm(range(0, len(missing), batch_size), desc="RXNMapper batches"):
            batch = missing[start : start + batch_size]
            smiles_batch = [record["reaction_smiles"] for record in batch]
            try:
                mapped_batch = mapper.get_attention_guided_atom_maps(smiles_batch)
            except Exception as exc:
                mapped_batch = [
                    {"mapped_rxn": None, "confidence": None, "error": f"RXNMapper failed: {exc}"}
                    for _ in batch
                ]
            for record, mapped in zip(batch, mapped_batch):
                mapped_rxn = mapped.get("mapped_rxn")
                confidence = mapped.get("confidence")
                if mapped_rxn:
                    ok, bond_changes, error, sanitized = _validate_mapped_reaction(mapped_rxn)
                else:
                    ok, bond_changes, error, sanitized = (
                        False,
                        [],
                        mapped.get("error") or "RXNMapper returned no mapping",
                        None,
                    )
                payload = {
                    "reaction_sha256": record["reaction_sha256"],
                    "reaction_smiles": record["reaction_smiles"],
                    "raw_mapped_reaction": mapped_rxn,
                    "mapped_reaction": sanitized or mapped_rxn,
                    "confidence": confidence,
                    "bond_changes": bond_changes,
                    "status": "ok" if ok else "failed",
                    "error": error,
                }
                path = atom_map_cache_path(cache_root, record["reaction_sha256"])
                _write_atom_map_cache(path, payload)
                cached[record["reaction_sha256"]] = payload

    return {
        "payloads": cached,
        "stats": {
            "unique_reactions": int(len(reaction_rows)),
            "cache_hits": int(hits),
            "cache_misses": int(len(missing)),
            "failed_cached": int(failed_cached),
            "failed_total": int(sum(1 for payload in cached.values() if payload.get("status") != "ok")),
            "cache_root": str(cache_root),
        },
    }


def _ec_prefix(label: str, level: int) -> str:
    parts = str(label).split(".")
    return ".".join(parts[:level]) if len(parts) >= level else str(label)


def materialize_clipzyme_inputs(
    split_frames: dict[str, pd.DataFrame],
    *,
    run_root: str | Path,
    cache_payloads: dict[str, dict],
) -> dict:
    run_root = Path(run_root)
    manifest_root = ensure_dir(run_root / "manifests")
    clipzyme_root = ensure_dir(run_root / "clipzyme_inputs")

    all_entries = []
    ec2uniprot: dict[str, set[str]] = {}
    uniprot2sequence: dict[str, str] = {}
    split_stats = {}
    manifest_paths = {}

    for split, frame in split_frames.items():
        usable_rows = []
        map_failed_rows = 0
        for record in tqdm(
            frame.to_dict("records"),
            total=len(frame),
            desc=f"{split} Clipzyme rows",
        ):
            payload = cache_payloads.get(record["reaction_sha256"])
            if not payload or payload.get("status") != "ok":
                map_failed_rows += 1
                continue
            mapped_reaction = payload["mapped_reaction"]
            raw_reactants, raw_products = _split_reaction_components(record["reaction_smiles"])
            mapped_reactants, mapped_products = _split_reaction_components(mapped_reaction)
            protein_id = record["protein_id"]
            ec_number = record["ec_number"]
            uniprot2sequence[protein_id] = record["sequence"]
            ec2uniprot.setdefault(ec_number, set()).add(protein_id)
            entry = {
                "rxnid": record["record_id"],
                "quality": 1.0,
                "ec": ec_number,
                "reactants": raw_reactants,
                "products": raw_products,
                "mapped_reactants": mapped_reactants,
                "mapped_products": mapped_products,
                "protein_refs": repr([protein_id]),
                "protein_db": "uniprot",
                "rule_id": record["reaction_id"],
                "split": CLIPZYME_SPLIT_MAP[split],
                "bond_changes": payload["bond_changes"],
            }
            all_entries.append(entry)
            usable = dict(record)
            usable.update(
                {
                    "mapped_reaction": mapped_reaction,
                    "atom_map_confidence": payload.get("confidence"),
                    "clipzyme_split": CLIPZYME_SPLIT_MAP[split],
                    "EC3": _ec_prefix(ec_number, 3),
                    "EC2": _ec_prefix(ec_number, 2),
                    "EC1": _ec_prefix(ec_number, 1),
                }
            )
            usable_rows.append(usable)

        manifest = pd.DataFrame(usable_rows)
        if manifest.empty:
            raise ValueError(
                f"{split} produced no usable Clipzyme rows after atom mapping and bond-change checks"
            )
        manifest_path = manifest_root / f"{split}.csv"
        manifest.to_csv(manifest_path, index=False)
        manifest_paths[split] = str(manifest_path)
        split_stats[split] = {
            "rows_before_mapping": int(len(frame)),
            "rows_after_mapping": int(len(manifest)),
            "dropped_mapping_or_bond_change_rows": int(map_failed_rows),
            "unique_reactions_after_mapping": int(manifest["reaction_id"].nunique()),
            "unique_proteins_after_mapping": int(manifest["protein_id"].nunique()),
            "unique_ec_labels_after_mapping": int(manifest["ec_number"].nunique()),
        }

    dataset_json_path = clipzyme_root / "enzymemap_emulator.json"
    dataset_json_path.write_text(json.dumps(all_entries, indent=2, sort_keys=True) + "\n")
    ec2uniprot_path = clipzyme_root / "ec2uniprot.p"
    uniprot2sequence_path = clipzyme_root / "uniprot2sequence.p"
    pickle.dump({key: sorted(value) for key, value in ec2uniprot.items()}, ec2uniprot_path.open("wb"))
    pickle.dump(uniprot2sequence, uniprot2sequence_path.open("wb"))

    return {
        "dataset_json": str(dataset_json_path),
        "ec2uniprot": str(ec2uniprot_path),
        "uniprot2sequence": str(uniprot2sequence_path),
        "manifests": manifest_paths,
        "mapping_stats": split_stats,
        "clipzyme_rows": int(len(all_entries)),
        "unique_proteins": int(len(uniprot2sequence)),
        "unique_ec_labels": int(len(ec2uniprot)),
    }


def prepare_split_group(
    group: SplitGroup,
    *,
    dataset_root: str | Path,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    label_policy: str = "remove",
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    limit_per_split: int | None = None,
    atom_map_batch_size: int = 32,
) -> dict:
    dataset_root = resolve_path(dataset_root)
    run_slug = split_group_slug(group.name)
    run_root = Path(runs_root) / run_slug
    split_frames = {}
    raw_stats = {}

    for split in SPLIT_NAMES:
        frame, stats = load_clipzyme_records(
            group.path / f"{split}.parquet",
            split_name=f"{group.name}/{split}",
            label_policy=label_policy,
            max_sequence_length=max_sequence_length,
            limit=limit_per_split,
        )
        if frame.empty:
            raise ValueError(
                f"{group.name}/{split} produced no usable rows after filtering. "
                "Check EC labels, reaction SMILES, and sequence fields."
            )
        split_frames[split] = frame
        raw_stats[split] = stats

    all_records = pd.concat(split_frames.values(), ignore_index=True)
    cache_result = populate_atom_map_cache(
        all_records,
        cache_root=cache_root,
        batch_size=atom_map_batch_size,
    )
    clipzyme_files = materialize_clipzyme_inputs(
        split_frames,
        run_root=run_root,
        cache_payloads=cache_result["payloads"],
    )

    metadata = {
        "dataset_root": str(dataset_root),
        "split_group": group.name,
        "run_slug": run_slug,
        "run_root": str(run_root),
        "cache_root": str(Path(cache_root)),
        "label_policy": label_policy,
        "max_sequence_length": int(max_sequence_length),
        "limit_per_split": limit_per_split,
        "stats": raw_stats,
        "atom_map_cache": cache_result["stats"],
        "clipzyme_files": {
            "dataset_json": clipzyme_files["dataset_json"],
            "ec2uniprot": clipzyme_files["ec2uniprot"],
            "uniprot2sequence": clipzyme_files["uniprot2sequence"],
        },
        "manifests": clipzyme_files["manifests"],
        "mapping_stats": clipzyme_files["mapping_stats"],
        "clipzyme_rows": clipzyme_files["clipzyme_rows"],
        "unique_proteins": clipzyme_files["unique_proteins"],
        "unique_ec_labels": clipzyme_files["unique_ec_labels"],
    }
    write_json(metadata_path_for_split(group.name, runs_root), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Clipzyme inputs from Leak-CURBER splits")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--atom-map-batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = select_split_groups(args.dataset_root, args.split_group)
    for group in groups:
        metadata = prepare_split_group(
            group,
            dataset_root=args.dataset_root,
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            label_policy=args.label_policy,
            max_sequence_length=args.max_sequence_length,
            limit_per_split=args.limit_per_split,
            atom_map_batch_size=args.atom_map_batch_size,
        )
        print(f"[emulator_bench] prepared {group.name}: {metadata['run_root']}", flush=True)


if __name__ == "__main__":
    main()
