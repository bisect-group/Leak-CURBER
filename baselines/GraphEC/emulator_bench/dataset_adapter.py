from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUNS_ROOT,
    SPLIT_NAMES,
    ensure_dir,
    resolve_path,
    safe_id,
    split_group_slug,
    write_json,
)


REQUIRED_COLUMNS = {"uniprot_id", "sequence", "ec_number"}
OPTIONAL_COLUMNS = ("uniprot_date", "pdbs", "pdb_source", "pdb_type")
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class SplitGroup:
    name: str
    path: Path


def discover_split_groups(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> list[SplitGroup]:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    groups = []
    for train_file in sorted(root.rglob("train.parquet")):
        parent = train_file.parent
        if all((parent / f"{split}.parquet").exists() for split in SPLIT_NAMES):
            groups.append(SplitGroup(parent.relative_to(root).as_posix(), parent))
    if not groups:
        raise FileNotFoundError(f"No train/val/test split groups found under {root}")
    return groups


def select_split_groups(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    requested: list[str] | None = None,
) -> list[SplitGroup]:
    groups = discover_split_groups(dataset_root)
    if not requested:
        return groups
    by_name = {group.name: group for group in groups}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown split group(s): {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in requested]


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def split_labels(value: object) -> list[str]:
    labels = []
    for raw in str(value).split(";"):
        label = raw.strip()
        if label and label.lower() != "nan":
            labels.append(label)
    return labels


def is_full_ec(label: str) -> bool:
    return "-" not in label and label.count(".") == 3


def full_ec_labels(value: object) -> list[str]:
    return sorted({label for label in split_labels(value) if is_full_ec(label)})


def normalize_sequence(value: object, max_sequence_length: int) -> tuple[str, int]:
    sequence = "".join(str(value).split()).upper()
    return sequence[:max_sequence_length], len(sequence)


def _aggregate_text(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values.dropna() if str(value)})
    return ";".join(cleaned)


def load_split_records(
    parquet_path: str | Path,
    *,
    split_name: str,
    max_sequence_length: int,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    parquet_path = Path(parquet_path)
    raw_df = pd.read_parquet(parquet_path)
    missing = REQUIRED_COLUMNS.difference(raw_df.columns)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    raw_rows = len(raw_df)
    if limit is not None:
        raw_df = raw_df.head(limit).copy()
    working_cols = ["uniprot_id", "sequence", "ec_number"] + [
        column for column in OPTIONAL_COLUMNS if column in raw_df.columns
    ]
    df = raw_df.loc[:, working_cols].copy()
    df = df.dropna(subset=["uniprot_id", "sequence", "ec_number"])
    missing_required_rows = raw_rows - len(df)

    normalized = df["sequence"].map(lambda value: normalize_sequence(value, max_sequence_length))
    df["Sequence"] = normalized.map(lambda item: item[0])
    df["Original Sequence Length"] = normalized.map(lambda item: item[1])
    df["Sequence Length"] = df["Sequence"].str.len()
    invalid_residue_rows = int(
        df["Sequence"].map(lambda sequence: not set(sequence).issubset(CANONICAL_AA)).sum()
    )
    df = df[df["Sequence"].map(lambda sequence: set(sequence).issubset(CANONICAL_AA))].copy()
    df["Original Entry"] = df["uniprot_id"].astype(str)
    df["labels"] = df["ec_number"].map(full_ec_labels)
    df["partial_label_count"] = df["ec_number"].map(
        lambda value: sum(1 for label in split_labels(value) if "-" in label)
    )
    df["non_full_label_count"] = df["ec_number"].map(
        lambda value: sum(1 for label in split_labels(value) if not is_full_ec(label))
    )
    rows_with_no_full_label = int((df["labels"].map(len) == 0).sum())
    df = df[(df["labels"].map(len) > 0) & (df["Sequence Length"] > 0)].copy()

    exploded_rows = []
    for row in tqdm(df.to_dict("records"), desc=f"{split_name} full EC labels", leave=False):
        for label in row["labels"]:
            item = {
                "Original Entry": row["Original Entry"],
                "EC number": label,
                "Sequence": row["Sequence"],
                "Original Sequence Length": row["Original Sequence Length"],
                "Sequence Length": row["Sequence Length"],
            }
            for column in OPTIONAL_COLUMNS:
                if column in row:
                    item[column] = row[column]
            exploded_rows.append(item)

    if exploded_rows:
        exploded = pd.DataFrame(exploded_rows)
    else:
        exploded = pd.DataFrame(
            columns=[
                "Original Entry",
                "EC number",
                "Sequence",
                "Original Sequence Length",
                "Sequence Length",
            ]
        )

    before_dedup_rows = len(exploded)
    if exploded.empty:
        records = exploded.copy()
    else:
        agg = {
            "EC number": lambda labels: ";".join(sorted(set(map(str, labels)))),
            "Original Sequence Length": "max",
            "Sequence Length": "max",
        }
        for column in OPTIONAL_COLUMNS:
            if column in exploded.columns:
                agg[column] = _aggregate_text
        records = (
            exploded.groupby(["Original Entry", "Sequence"], as_index=False)
            .agg(agg)
            .sort_values(["Original Entry", "Sequence"])
            .reset_index(drop=True)
        )

    if limit is not None:
        records = records.head(limit).copy()

    records["sequence_sha256"] = records["Sequence"].map(sequence_hash)
    stats = {
        "split": split_name,
        "raw_rows": int(raw_rows),
        "missing_required_rows": int(missing_required_rows),
        "invalid_residue_rows": invalid_residue_rows,
        "rows_with_no_full_label": rows_with_no_full_label,
        "rows_after_full_label_filter": int(len(df)),
        "rows_after_label_explosion": int(before_dedup_rows),
        "rows_after_dedup": int(len(records)),
        "unique_entries": int(records["Original Entry"].nunique()) if not records.empty else 0,
        "unique_sequences": int(records["Sequence"].nunique()) if not records.empty else 0,
        "unique_full_ec_labels": int(records["EC number"].str.split(";").explode().nunique())
        if not records.empty
        else 0,
        "truncated_sequences": int(
            (records["Original Sequence Length"] > records["Sequence Length"]).sum()
        )
        if not records.empty
        else 0,
        "max_sequence_length": max_sequence_length,
        "limit": limit,
    }
    return records, stats


def assign_cache_keys(records_by_split: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    entry_to_sequences: dict[str, set[str]] = {}
    for records in records_by_split.values():
        for _, row in records.iterrows():
            entry_to_sequences.setdefault(str(row["Original Entry"]), set()).add(
                str(row["Sequence"])
            )

    updated = {}
    for split_name, records in records_by_split.items():
        records = records.copy()
        keys = []
        for _, row in records.iterrows():
            entry = str(row["Original Entry"])
            base = safe_id(entry)
            if len(entry_to_sequences.get(entry, set())) > 1:
                base = f"{base}__{str(row['sequence_sha256'])[:12]}"
            keys.append(base)
        records["Entry"] = keys
        records["cache_key"] = keys
        updated[split_name] = records
    return updated


def write_fasta(records: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for row in records.itertuples(index=False):
            handle.write(f">{row.Entry}\n{row.Sequence}\n")
    return path


def build_vocab(records_by_split: dict[str, pd.DataFrame]) -> dict:
    labels = set()
    split_label_counts = {}
    for split_name, records in records_by_split.items():
        split_labels_set = set()
        for value in records.get("EC number", pd.Series(dtype=str)).astype(str):
            split_labels_set.update(split_labels(value))
        labels.update(split_labels_set)
        split_label_counts[split_name] = len(split_labels_set)
    label_to_idx = {label: index for index, label in enumerate(sorted(labels))}
    return {
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(index): label for label, index in label_to_idx.items()},
        "num_labels": len(label_to_idx),
        "source": "full EC labels observed across train, val, and test after per-file dedup",
        "split_unique_label_counts": split_label_counts,
    }


def prepare_split_group(
    group: SplitGroup,
    *,
    dataset_root: str | Path,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    max_sequence_length: int = 1022,
    limit_per_split: int | None = None,
) -> dict:
    dataset_root = resolve_path(dataset_root)
    run_slug = split_group_slug(group.name)
    run_root = Path(runs_root) / run_slug
    manifest_root = ensure_dir(run_root / "manifests")
    fasta_root = ensure_dir(run_root / "fastas")
    vocab_root = ensure_dir(run_root / "vocab")

    records_by_split = {}
    stats = {}
    for split_name in SPLIT_NAMES:
        records, split_stats = load_split_records(
            group.path / f"{split_name}.parquet",
            split_name=f"{group.name}/{split_name}",
            max_sequence_length=max_sequence_length,
            limit=limit_per_split,
        )
        if records.empty:
            raise ValueError(f"No usable full-EC rows remain for {group.name}/{split_name}")
        records_by_split[split_name] = records
        stats[split_name] = split_stats

    records_by_split = assign_cache_keys(records_by_split)
    vocab = build_vocab(records_by_split)
    vocab_path = write_json(vocab_root / "ec_vocab.json", vocab)

    manifests = {}
    fastas = {}
    for split_name, records in records_by_split.items():
        ordered_cols = [
            "Entry",
            "Original Entry",
            "EC number",
            "Sequence",
            "Original Sequence Length",
            "Sequence Length",
            "cache_key",
            "sequence_sha256",
        ]
        ordered_cols += [column for column in OPTIONAL_COLUMNS if column in records.columns]
        manifest_path = manifest_root / f"{split_name}.csv"
        records.loc[:, ordered_cols].to_csv(manifest_path, index=False)
        fasta_path = write_fasta(records, fasta_root / f"{split_name}.fasta")
        manifests[split_name] = str(manifest_path)
        fastas[split_name] = str(fasta_path)

    metadata = {
        "dataset_root": str(dataset_root),
        "split_group": group.name,
        "run_slug": run_slug,
        "run_root": str(run_root),
        "max_sequence_length": max_sequence_length,
        "limit_per_split": limit_per_split,
        "manifests": manifests,
        "fastas": fastas,
        "vocab_path": str(vocab_path),
        "vocab_size": vocab["num_labels"],
        "stats": stats,
    }
    metadata_path = write_json(run_root / "metadata.json", metadata)
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def prepare_dataset(
    *,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    split_groups: list[str] | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    max_sequence_length: int = 1022,
    limit_per_split: int | None = None,
) -> list[dict]:
    selected = select_split_groups(dataset_root, split_groups)
    return [
        prepare_split_group(
            group,
            dataset_root=dataset_root,
            runs_root=runs_root,
            max_sequence_length=max_sequence_length,
            limit_per_split=limit_per_split,
        )
        for group in selected
    ]
