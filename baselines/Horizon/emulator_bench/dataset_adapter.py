from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import (
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
DEFAULT_MAX_SEQUENCE_LENGTH = 5000


@dataclass(frozen=True)
class SplitGroup:
    name: str
    path: Path


def discover_split_groups(dataset_root: str | Path) -> list[SplitGroup]:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    groups: list[SplitGroup] = []
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
        raw_parts = []
        for item in value:
            raw_parts.extend(_split_label_values(item))
        return raw_parts
    return [part.strip() for part in re.split(r"[;,]", str(value)) if part.strip()]


def normalize_ec_labels(value: object, policy: str = "remove") -> list[str]:
    normalized: list[str] = []
    for label in _split_label_values(value):
        lowered = label.lower()
        if lowered in {"nan", "none", "null", "-", "-.-.-.-"}:
            continue
        parts = [part.strip() for part in label.split(".")]
        has_missing = any(part in {"", "-"} for part in parts)
        if policy == "remove":
            if has_missing:
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


def _filter_horizon_featurizable_reactions(records: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, dict]:
    if records.empty:
        return records, {"invalid_fingerprint_reactions": 0, "invalid_fingerprint_rows": 0}

    from horizyn.datasets.base import BaseDataset
    from horizyn.datasets.fingerprints import (
        DRFPFingerprintDataset,
        RDKitPlusFingerprintDataset,
    )

    reactions = records.loc[:, ["reaction_id", "reaction_smiles"]].drop_duplicates()
    augmented_keys = []
    augmented_data = []
    key_to_reaction_id = {}
    for row in reactions.itertuples(index=False):
        forward_key = f"{row.reaction_id}_f"
        reverse_key = f"{row.reaction_id}_r"
        parts = str(row.reaction_smiles).split(">>")
        augmented_keys.append(forward_key)
        augmented_data.append({"reaction_smiles": row.reaction_smiles})
        key_to_reaction_id[forward_key] = row.reaction_id
        if len(parts) == 2:
            augmented_keys.append(reverse_key)
            augmented_data.append({"reaction_smiles": f"{parts[1]}>>{parts[0]}"})
            key_to_reaction_id[reverse_key] = row.reaction_id

    reaction_dataset = BaseDataset(keys=augmented_keys, array_data=augmented_data)
    rdkit_fp = RDKitPlusFingerprintDataset(
        reaction_dataset=reaction_dataset,
        vec_dim=1024,
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=True,
        standardize=True,
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )
    drfp_fp = DRFPFingerprintDataset(
        reaction_dataset=reaction_dataset,
        vec_dim=1024,
        radius=3,
        rings=True,
        standardize=True,
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )

    invalid_reactions = set()
    for key in tqdm(augmented_keys, desc=f"{split_name} fingerprints", leave=False):
        reaction_id = key_to_reaction_id[key]
        if reaction_id in invalid_reactions:
            continue
        try:
            _ = rdkit_fp[key]
            _ = drfp_fp[key]
        except Exception as exc:
            print(
                f"[emulator_bench] dropping reaction {reaction_id}; "
                f"Horizon fingerprint generation failed: {exc}",
                flush=True,
            )
            invalid_reactions.add(reaction_id)

    if not invalid_reactions:
        return records, {"invalid_fingerprint_reactions": 0, "invalid_fingerprint_rows": 0}
    before = len(records)
    filtered = records[~records["reaction_id"].isin(invalid_reactions)].copy()
    return filtered, {
        "invalid_fingerprint_reactions": int(len(invalid_reactions)),
        "invalid_fingerprint_rows": int(before - len(filtered)),
    }


def reaction_id_for_smiles(smiles: str) -> str:
    return f"rxn_{sha256_text(smiles)[:20]}"


def protein_id_for_sequence(sequence: str) -> str:
    return f"seq_{sha256_text(sequence)[:20]}"


def pair_id_for_ids(reaction_id: str, protein_id: str) -> str:
    joined = f"{reaction_id}\t{protein_id}"
    return f"pair_{sha256_text(joined)[:20]}"


def _validate_columns(path: Path, columns: list[str]) -> None:
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_horizon_records(
    parquet_path: str | Path,
    *,
    split_name: str,
    label_policy: str = "remove",
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    limit: int | None = None,
    validate_fingerprints: bool = True,
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
    for row in tqdm(
        raw_df.loc[:, columns].itertuples(index=False),
        total=raw_rows,
        desc=f"{split_name} rows",
        leave=False,
    ):
        rxn_value, sequence_value, ec_value = row
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

        reaction_sha = sha256_text(reaction_smiles)
        sequence_sha = sha256_text(sequence)
        reaction_id = reaction_id_for_smiles(reaction_smiles)
        protein_id = protein_id_for_sequence(sequence)
        records.append(
            {
                "pr_id": pair_id_for_ids(reaction_id, protein_id),
                "reaction_id": reaction_id,
                "protein_id": protein_id,
                "reaction_smiles": reaction_smiles,
                "sequence": sequence,
                "ec_number": ";".join(labels),
                "reaction_sha256": reaction_sha,
                "sequence_sha256": sequence_sha,
                "original_sequence_length": original_sequence_length,
                "sequence_length": len(sequence),
                "sequence_truncated": truncated,
            }
        )

    before_dedup = len(records)
    if records:
        df = pd.DataFrame(records)
        dedup_df = (
            df.groupby(["reaction_id", "protein_id"], as_index=False)
            .agg(
                {
                    "pr_id": "first",
                    "reaction_smiles": "first",
                    "sequence": "first",
                    "ec_number": lambda values: ";".join(
                        sorted(
                            {
                                label
                                for value in values
                                for label in str(value).split(";")
                                if label
                            }
                        )
                    ),
                    "reaction_sha256": "first",
                    "sequence_sha256": "first",
                    "original_sequence_length": "max",
                    "sequence_length": "max",
                    "sequence_truncated": "max",
                }
            )
            .sort_values(["reaction_id", "protein_id"])
            .reset_index(drop=True)
        )
    else:
        dedup_df = pd.DataFrame(
            columns=[
                "reaction_id",
                "protein_id",
                "pr_id",
                "reaction_smiles",
                "sequence",
                "ec_number",
                "reaction_sha256",
                "sequence_sha256",
                "original_sequence_length",
                "sequence_length",
                "sequence_truncated",
            ]
        )

    rows_after_pair_dedup = len(dedup_df)
    if limit is not None:
        dedup_df = dedup_df.head(limit).copy()

    rows_after_limit = len(dedup_df)
    fingerprint_stats = {"invalid_fingerprint_reactions": 0, "invalid_fingerprint_rows": 0}
    if validate_fingerprints:
        dedup_df, fingerprint_stats = _filter_horizon_featurizable_reactions(
            dedup_df,
            split_name,
        )

    stats = {
        "split": split_name,
        "raw_rows": int(raw_rows),
        "missing_required_rows": int(missing_required_rows),
        "invalid_reaction_rows": int(invalid_reaction_rows),
        "empty_sequence_rows": int(empty_sequence_rows),
        "dropped_label_rows": int(dropped_label_rows),
        "truncated_rows_before_dedup": int(truncated_rows),
        "rows_after_filter": int(before_dedup),
        "rows_after_pair_dedup": int(rows_after_pair_dedup),
        "rows_after_limit": int(rows_after_limit),
        "rows_after_dedup": int(len(dedup_df)),
        "duplicate_pair_rows": int(before_dedup - rows_after_pair_dedup),
        **fingerprint_stats,
        "unique_reactions": int(dedup_df["reaction_id"].nunique()) if not dedup_df.empty else 0,
        "unique_proteins": int(dedup_df["protein_id"].nunique()) if not dedup_df.empty else 0,
        "unique_ec_labels": int(dedup_df["ec_number"].str.split(";").explode().nunique())
        if not dedup_df.empty
        else 0,
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


def write_pairs_csv(records: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    records.loc[:, ["pr_id", "reaction_id", "protein_id"]].to_csv(path, index=False)


def write_reactions_csv(records: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    reactions = (
        records.loc[:, ["reaction_id", "reaction_smiles"]]
        .drop_duplicates()
        .sort_values("reaction_id")
    )
    reactions.to_csv(path, index=False)


def write_manifest(records: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    records.to_csv(path, index=False)


def prepare_split_group(
    group: SplitGroup,
    *,
    dataset_root: str | Path,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    label_policy: str = "remove",
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    limit_per_split: int | None = None,
    validate_fingerprints: bool = True,
) -> dict:
    dataset_root = resolve_path(dataset_root)
    run_slug = split_group_slug(group.name)
    run_root = Path(runs_root) / run_slug
    manifest_root = run_root / "manifests"
    data_root = run_root / "horizyn_data"

    metadata: dict = {
        "dataset_root": str(dataset_root),
        "split_group": group.name,
        "run_slug": run_slug,
        "run_root": str(run_root),
        "label_policy": label_policy,
        "max_sequence_length": int(max_sequence_length),
        "limit_per_split": limit_per_split,
        "baseline_files": {},
        "manifests": {},
        "stats": {},
    }

    for split in SPLIT_NAMES:
        records, stats = load_horizon_records(
            group.path / f"{split}.parquet",
            split_name=f"{group.name}/{split}",
            label_policy=label_policy,
            max_sequence_length=max_sequence_length,
            limit=limit_per_split,
            validate_fingerprints=validate_fingerprints,
        )
        if records.empty:
            raise ValueError(
                f"{group.name}/{split} produced no usable rows after filtering. "
                "Check EC labels, reaction SMILES, and sequence fields."
            )

        manifest_path = manifest_root / f"{split}.csv"
        pairs_path = data_root / f"{split}_pairs.csv"
        reactions_path = data_root / f"{split}_rxns.csv"
        write_manifest(records, manifest_path)
        write_pairs_csv(records, pairs_path)
        write_reactions_csv(records, reactions_path)
        metadata["manifests"][split] = str(manifest_path)
        metadata["baseline_files"][split] = {
            "pairs": str(pairs_path),
            "reactions": str(reactions_path),
        }
        metadata["stats"][split] = stats

    write_json(metadata_path_for_split(group.name, runs_root), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Horizon inputs from Leak-CURBER splits")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split-group", action="append")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--label-policy", choices=["remove", "truncate", "keep"], default="remove")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--skip-fingerprint-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = select_split_groups(args.dataset_root, args.split_group)
    for group in groups:
        metadata = prepare_split_group(
            group,
            dataset_root=args.dataset_root,
            runs_root=args.runs_root,
            label_policy=args.label_policy,
            max_sequence_length=args.max_sequence_length,
            limit_per_split=args.limit_per_split,
            validate_fingerprints=not args.skip_fingerprint_validation,
        )
        print(f"[emulator_bench] prepared {group.name}: {metadata['run_root']}", flush=True)


if __name__ == "__main__":
    main()
