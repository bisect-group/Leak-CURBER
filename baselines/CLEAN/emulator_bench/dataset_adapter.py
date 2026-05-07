from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import (
    APP_DATA_DIR,
    DEFAULT_RUNS_ROOT,
    cache_key_for_entry,
    ensure_dir,
    resolve_path,
    split_group_slug,
    write_json,
)


REQUIRED_COLUMNS = {"uniprot_id", "sequence", "ec_number"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitGroup:
    name: str
    path: Path


def discover_split_groups(dataset_root: str | Path) -> list[SplitGroup]:
    root = resolve_path(dataset_root, base=Path.cwd())
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


def _split_labels(value: object) -> list[str]:
    labels = []
    for raw in str(value).split(";"):
        label = raw.strip()
        if label and label.lower() != "nan":
            labels.append(label)
    return labels


def normalize_labels(value: object, policy: str) -> list[str]:
    labels = _split_labels(value)
    normalized = []
    for label in labels:
        has_missing = "-" in label
        if policy == "remove" and has_missing:
            continue
        if policy == "truncate" and has_missing:
            parts = [part for part in label.split(".") if part and part != "-"]
            if not parts:
                continue
            normalized.append(".".join(parts))
            continue
        if policy in {"remove", "keep"} or (policy == "truncate" and not has_missing):
            normalized.append(label)
            continue
        raise ValueError(f"Unsupported label policy: {policy}")
    return sorted(set(normalized))


def _validate_columns(path: Path, columns: list[str]) -> None:
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def load_clean_records(
    parquet_path: str | Path,
    *,
    split_name: str,
    label_policy: str = "remove",
    max_sequence_length: int = 1024,
    limit: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    parquet_path = Path(parquet_path)
    raw_df = pd.read_parquet(parquet_path)
    _validate_columns(parquet_path, list(raw_df.columns))
    raw_rows = len(raw_df)

    df = raw_df.loc[:, ["uniprot_id", "sequence", "ec_number"]].copy()
    df = df.dropna(subset=["uniprot_id", "sequence", "ec_number"])
    missing_required_rows = raw_rows - len(df)

    df["Entry"] = df["uniprot_id"].astype(str).map(cache_key_for_entry)
    df["Original Entry"] = df["uniprot_id"].astype(str)
    df["Sequence"] = df["sequence"].astype(str).str.replace(r"\s+", "", regex=True).str[:max_sequence_length]
    df["Original Sequence Length"] = df["sequence"].astype(str).str.replace(
        r"\s+", "", regex=True
    ).str.len()
    df["Sequence Length"] = df["Sequence"].str.len()
    df["labels"] = df["ec_number"].map(lambda value: normalize_labels(value, label_policy))

    partial_or_missing_labels = int((df["labels"].map(len) == 0).sum())
    df = df[df["labels"].map(len) > 0].copy()
    df = df[df["Sequence Length"] > 0].copy()

    exploded = []
    for row in tqdm(
        df.to_dict("records"),
        total=len(df),
        desc=f"{split_name} labels",
        leave=False,
    ):
        for label in row["labels"]:
            exploded.append(
                {
                    "Entry": row["Entry"],
                    "Original Entry": row["Original Entry"],
                    "EC number": label,
                    "Sequence": row["Sequence"],
                    "Original Sequence Length": row["Original Sequence Length"],
                    "Sequence Length": row["Sequence Length"],
                }
            )

    if exploded:
        clean_df = pd.DataFrame(exploded)
    else:
        clean_df = pd.DataFrame(
            columns=[
                "Entry",
                "Original Entry",
                "EC number",
                "Sequence",
                "Original Sequence Length",
                "Sequence Length",
            ]
        )

    before_dedup_rows = len(clean_df)
    if clean_df.empty:
        dedup_df = clean_df
    else:
        dedup_df = (
            clean_df.groupby(["Entry", "Original Entry", "Sequence"], as_index=False)
            .agg(
                {
                    "EC number": lambda labels: ";".join(sorted(set(map(str, labels)))),
                    "Original Sequence Length": "max",
                    "Sequence Length": "max",
                }
            )
            .sort_values(["Entry", "Sequence"])
            .reset_index(drop=True)
        )

    if limit is not None:
        dedup_df = dedup_df.head(limit).copy()

    dedup_df["cache_key"] = dedup_df["Entry"].map(cache_key_for_entry)
    dedup_df["sequence_sha256"] = dedup_df["Sequence"].map(_sequence_sha256)

    stats = {
        "split": split_name,
        "raw_rows": raw_rows,
        "missing_required_rows": missing_required_rows,
        "rows_after_label_policy": int(len(df)),
        "rows_after_label_explosion": int(before_dedup_rows),
        "rows_after_dedup": int(len(dedup_df)),
        "dropped_label_rows": partial_or_missing_labels,
        "unique_entries": int(dedup_df["Entry"].nunique()) if not dedup_df.empty else 0,
        "unique_sequences": int(dedup_df["Sequence"].nunique()) if not dedup_df.empty else 0,
        "unique_labels": int(dedup_df["EC number"].str.split(";").explode().nunique())
        if not dedup_df.empty
        else 0,
        "truncated_sequences": int(
            (dedup_df["Original Sequence Length"] > dedup_df["Sequence Length"]).sum()
        )
        if not dedup_df.empty
        else 0,
        "max_sequence_length": max_sequence_length,
        "label_policy": label_policy,
        "limit": limit,
    }
    return dedup_df, stats


def write_clean_tsv(records: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    records.loc[:, ["Entry", "EC number", "Sequence"]].to_csv(path, sep="\t", index=False)


def write_fasta(records: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for row in records.itertuples(index=False):
            handle.write(f">{row.Entry}\n{row.Sequence}\n")


def prepare_split_group(
    group: SplitGroup,
    *,
    dataset_root: str | Path,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    label_policy: str = "remove",
    max_sequence_length: int = 1024,
    limit_per_split: int | None = None,
) -> dict:
    dataset_root = resolve_path(dataset_root, base=Path.cwd())
    run_slug = split_group_slug(group.name)
    run_root = Path(runs_root) / run_slug
    manifest_root = run_root / "manifests"
    data_prefix = f"emulator_{run_slug}"

    metadata: dict = {
        "dataset_root": str(dataset_root),
        "split_group": group.name,
        "run_slug": run_slug,
        "run_root": str(run_root),
        "label_policy": label_policy,
        "max_sequence_length": max_sequence_length,
        "clean_data": {},
        "baseline_files": {},
        "manifests": {},
        "stats": {},
    }

    for split_name in SPLIT_NAMES:
        records, stats = load_clean_records(
            group.path / f"{split_name}.parquet",
            split_name=f"{group.name}/{split_name}",
            label_policy=label_policy,
            max_sequence_length=max_sequence_length,
            limit=limit_per_split,
        )
        if records.empty:
            raise ValueError(f"{group.name}/{split_name} has no records after filtering")

        clean_data_name = f"{data_prefix}_{split_name}"
        clean_tsv = APP_DATA_DIR / f"{clean_data_name}.csv"
        clean_fasta = APP_DATA_DIR / f"{clean_data_name}.fasta"
        manifest_csv = manifest_root / f"{split_name}.csv"

        write_clean_tsv(records, clean_tsv)
        write_fasta(records, clean_fasta)
        ensure_dir(manifest_csv.parent)
        records.to_csv(manifest_csv, index=False)

        metadata["clean_data"][split_name] = clean_data_name
        metadata["baseline_files"][split_name] = {
            "csv": str(clean_tsv),
            "fasta": str(clean_fasta),
        }
        metadata["manifests"][split_name] = str(manifest_csv)
        metadata["stats"][split_name] = stats

    write_json(run_root / "metadata.json", metadata)
    return metadata


def load_manifest(metadata: dict, split_name: str) -> pd.DataFrame:
    path = Path(metadata["manifests"][split_name])
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return pd.read_csv(path)
