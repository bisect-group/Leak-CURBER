from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .utils import (
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_RUNS_ROOT,
    DEFAULT_VOCAB_PATH,
    SPLIT_NAMES,
    cache_key_for_entry,
    ensure_dir,
    resolve_path,
    split_group_slug,
    write_json,
)


REQUIRED_COLUMNS = {"uniprot_id", "sequence", "ec_number"}
OPTIONAL_METADATA_COLUMNS = ["uniprot_date", "pdbs", "pdb_source", "pdb_type", "pdb_count"]
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYX")


@dataclass(frozen=True)
class SplitGroup:
    name: str
    path: Path


def discover_split_groups(dataset_root):
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError("Dataset root does not exist: {}".format(root))

    groups = []
    for train_file in sorted(root.rglob("train.parquet")):
        parent = train_file.parent
        if all((parent / "{}.parquet".format(split)).exists() for split in SPLIT_NAMES):
            groups.append(SplitGroup(parent.relative_to(root).as_posix(), parent))

    if not groups:
        raise FileNotFoundError("No train/val/test split groups found under {}".format(root))
    return groups


def select_split_groups(dataset_root, requested=None):
    groups = discover_split_groups(dataset_root)
    if not requested:
        return groups
    by_name = {group.name: group for group in groups}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(
            "Unknown split group(s): {}. Available: {}".format(missing, sorted(by_name))
        )
    return [by_name[name] for name in requested]


def normalize_sequence(sequence):
    text = "".join(str(sequence).split()).upper()
    return "".join(char if char in AMINO_ACIDS else "X" for char in text)


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def split_ec_labels(value):
    if pd.isna(value):
        return []
    labels = []
    for raw in str(value).replace(",", ";").split(";"):
        label = raw.strip()
        if label and label.lower() not in {"nan", "none", "null"}:
            labels.append(label)
    return labels


def ec_prefixes(label):
    prefixes = []
    parts = str(label).strip().split(".")
    for part in parts[:4]:
        normalized = part.strip()
        if not normalized or normalized == "-" or normalized.lower().startswith("n"):
            break
        prefixes.append(".".join(parts[: len(prefixes) + 1]))
    return prefixes


def parse_label_set(value):
    raw_labels = split_ec_labels(value)
    parsed = []
    for label in raw_labels:
        prefixes = ec_prefixes(label)
        if prefixes:
            parsed.append({"label": label, "prefixes": prefixes, "depth": len(prefixes)})
    return parsed


def _validate_columns(path, columns):
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError("{} is missing required columns: {}".format(path, sorted(missing)))


def _first_non_null(values):
    for value in values:
        if pd.notna(value):
            return value
    return None


def _parquet_columns(path):
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).head(0).columns)


def _parquet_row_count(path):
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _read_parquet_columns(path, columns, limit=None):
    if limit is None:
        return pd.read_parquet(path, columns=columns)

    try:
        import pyarrow.parquet as pq

        frames = []
        raw_target = max(int(limit) * 256, 4096)
        rows_read = 0
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=min(raw_target, 65536), columns=columns):
            frame = batch.to_pandas()
            frames.append(frame)
            rows_read += len(frame)
            if rows_read >= raw_target:
                break
        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
    except Exception:
        return pd.read_parquet(path, columns=columns).head(max(int(limit) * 256, 4096))


def build_ec_vocab(dataset_root, vocab_path=DEFAULT_VOCAB_PATH, rebuild=False):
    dataset_root = resolve_path(dataset_root)
    vocab_path = resolve_path(vocab_path)
    if vocab_path.exists() and not rebuild:
        return json.loads(vocab_path.read_text())

    root_file = dataset_root / "enzyme_classification_dataset.parquet"
    if not root_file.exists():
        raise FileNotFoundError("Missing dataset root parquet: {}".format(root_file))

    df = pd.read_parquet(root_file, columns=["ec_number"])
    level_sets = [set(), set(), set(), set()]
    total_items = 0
    usable_items = 0
    max_depth_counts = {"1": 0, "2": 0, "3": 0, "4": 0}

    for value in tqdm(df["ec_number"], desc="building EC vocabulary", unit="row"):
        for item in parse_label_set(value):
            total_items += 1
            prefixes = item["prefixes"]
            usable_items += 1
            max_depth_counts[str(len(prefixes))] += 1
            for index, prefix in enumerate(prefixes):
                level_sets[index].add(prefix)

    levels = [sorted(values, key=_ec_sort_key) for values in level_sets]
    starts = []
    cursor = 0
    for values in levels:
        starts.append(cursor)
        cursor += len(values)

    vocab = {
        "dataset_root": str(dataset_root),
        "source": str(root_file),
        "levels": levels,
        "level_starts": starts,
        "output_dims": [len(values) for values in levels],
        "total_outputs": cursor,
        "label_items": total_items,
        "usable_label_items": usable_items,
        "max_depth_counts": max_depth_counts,
        "policy": "masked_prefix",
    }
    write_json(vocab_path, vocab)
    return vocab


def _ec_sort_key(value):
    key = []
    for part in str(value).split("."):
        try:
            key.append((0, int(part)))
        except ValueError:
            key.append((1, part))
    return key


def label_targets(ec_number, vocab):
    level_maps = [
        {label: idx for idx, label in enumerate(labels)}
        for labels in vocab["levels"]
    ]
    target_indices = [set(), set(), set(), set()]
    observed_levels = set()
    usable_labels = []
    dropped_labels = []

    for item in parse_label_set(ec_number):
        prefixes = item["prefixes"]
        usable_labels.append(item["label"])
        for level_index, prefix in enumerate(prefixes):
            observed_levels.add(level_index)
            local_index = level_maps[level_index].get(prefix)
            if local_index is None:
                dropped_labels.append(prefix)
                continue
            target_indices[level_index].add(vocab["level_starts"][level_index] + local_index)

    return {
        "target_indices": [sorted(values) for values in target_indices],
        "observed_levels": sorted(observed_levels),
        "usable_labels": sorted(set(usable_labels), key=_ec_sort_key),
        "dropped_labels": sorted(set(dropped_labels), key=_ec_sort_key),
    }


def load_hitec_records(
    parquet_path,
    split_name,
    vocab,
    max_tokens=DEFAULT_MODEL_DIMENSION,
    limit=None,
):
    parquet_path = Path(parquet_path)
    source_columns = _parquet_columns(parquet_path)
    _validate_columns(parquet_path, source_columns)

    columns = ["uniprot_id", "sequence", "ec_number"] + [
        col for col in OPTIONAL_METADATA_COLUMNS if col in source_columns
    ]
    raw_df = _read_parquet_columns(parquet_path, columns=columns, limit=limit)
    source_raw_rows = _parquet_row_count(parquet_path)
    raw_rows_read = len(raw_df)
    raw_rows = source_raw_rows if source_raw_rows is not None else raw_rows_read
    df = raw_df.loc[:, columns].copy()
    df = df.dropna(subset=["uniprot_id", "sequence", "ec_number"])
    rows_after_required_drop = len(df)
    missing_required_rows = raw_rows_read - rows_after_required_drop

    df["Original Entry"] = df["uniprot_id"].astype(str)
    df["Entry"] = df["Original Entry"].map(cache_key_for_entry)
    df["Original Sequence"] = df["sequence"].map(normalize_sequence)
    df["Original Sequence Length"] = df["Original Sequence"].str.len()
    residue_limit = max_tokens - 1
    if residue_limit <= 0:
        raise ValueError("max_tokens must be > 1")
    df["Sequence"] = df["Original Sequence"].str[:residue_limit]
    df["Sequence Length"] = df["Sequence"].str.len()
    df = df[df["Sequence Length"] > 0].copy()

    before_exact_dedup = len(df)
    df = df.drop_duplicates(subset=["Original Entry", "Sequence", "ec_number"]).copy()

    aggregations = {
        "Entry": "first",
        "ec_number": lambda values: ";".join(
            sorted(set(label for value in values for label in split_ec_labels(value)), key=_ec_sort_key)
        ),
        "Original Sequence Length": "max",
        "Sequence Length": "max",
        "uniprot_id": "first",
    }
    for col in OPTIONAL_METADATA_COLUMNS:
        if col in df.columns:
            aggregations[col] = _first_non_null

    grouped = (
        df.groupby(["Original Entry", "Sequence"], as_index=False)
        .agg(aggregations)
        .sort_values(["Entry", "Sequence"])
        .reset_index(drop=True)
    )

    grouped_for_processing = grouped.head(limit).copy() if limit is not None else grouped
    rows = []
    no_label_rows = 0
    dropped_vocab_prefixes = 0
    for row in tqdm(
        grouped_for_processing.to_dict("records"),
        total=len(grouped_for_processing),
        desc="{} labels".format(split_name),
        leave=False,
    ):
        targets = label_targets(row["ec_number"], vocab)
        if not any(targets["target_indices"]) or not targets["observed_levels"]:
            no_label_rows += 1
            continue
        if targets["dropped_labels"]:
            dropped_vocab_prefixes += len(targets["dropped_labels"])
        row["EC number"] = ";".join(targets["usable_labels"])
        row["target_indices"] = json.dumps(targets["target_indices"], separators=(",", ":"))
        row["observed_levels"] = json.dumps(targets["observed_levels"], separators=(",", ":"))
        row["cache_key"] = row["Entry"]
        row["sequence_sha256"] = sequence_sha256(row["Sequence"])
        row["token_cache_path"] = ""
        rows.append(row)

    records = pd.DataFrame(rows)
    stats = {
        "split": split_name,
        "raw_rows": int(raw_rows),
        "raw_rows_read": int(raw_rows_read),
        "raw_read_limited": bool(limit is not None),
        "missing_required_rows": int(missing_required_rows),
        "rows_after_required_drop": int(rows_after_required_drop),
        "rows_before_exact_dedup": int(before_exact_dedup),
        "rows_after_exact_dedup": int(len(df)),
        "rows_after_sequence_aggregation": int(len(grouped)),
        "rows_considered_after_limit": int(len(grouped_for_processing)),
        "rows_after_label_filter": int(len(records)),
        "dropped_no_label_rows": int(no_label_rows),
        "dropped_vocab_prefixes": int(dropped_vocab_prefixes),
        "unique_entries": int(records["Entry"].nunique()) if not records.empty else 0,
        "truncated_sequences": int(
            (records["Original Sequence Length"] > records["Sequence Length"]).sum()
        )
        if not records.empty
        else 0,
        "max_tokens": int(max_tokens),
        "residue_limit": int(max_tokens - 1),
        "limit": limit,
    }
    return records, stats


def write_manifest(records, path):
    path = Path(path)
    ensure_dir(path.parent)
    records.to_csv(path, index=False)


def prepare_split_group(
    group,
    dataset_root,
    vocab,
    runs_root=DEFAULT_RUNS_ROOT,
    cache_root=None,
    max_tokens=DEFAULT_MODEL_DIMENSION,
    model_dimension=DEFAULT_MODEL_DIMENSION,
    limit_per_split=None,
):
    dataset_root = resolve_path(dataset_root)
    run_slug = split_group_slug(group.name)
    run_root = Path(runs_root) / run_slug
    manifest_root = run_root / "manifests"
    metadata = {
        "dataset_root": str(dataset_root),
        "split_group": group.name,
        "run_slug": run_slug,
        "run_root": str(run_root),
        "cache_root": str(cache_root) if cache_root else None,
        "model_dimension": int(model_dimension),
        "max_tokens": int(max_tokens),
        "max_supported_tokens": int(model_dimension),
        "vocab_path": str(DEFAULT_VOCAB_PATH),
        "output_dims": vocab["output_dims"],
        "level_starts": vocab["level_starts"],
        "manifests": {},
        "stats": {},
    }

    for split_name in SPLIT_NAMES:
        records, stats = load_hitec_records(
            group.path / "{}.parquet".format(split_name),
            split_name="{}/{}".format(group.name, split_name),
            vocab=vocab,
            max_tokens=max_tokens,
            limit=limit_per_split,
        )
        if records.empty:
            raise ValueError("{}/{} has no rows after filtering".format(group.name, split_name))
        manifest_csv = manifest_root / "{}.csv".format(split_name)
        write_manifest(records, manifest_csv)
        metadata["manifests"][split_name] = str(manifest_csv)
        metadata["stats"][split_name] = stats

    write_json(run_root / "metadata.json", metadata)
    return metadata


def load_manifest(metadata, split_name):
    path = Path(metadata["manifests"][split_name])
    if not path.exists():
        raise FileNotFoundError("Missing manifest: {}".format(path))
    return pd.read_csv(path)
