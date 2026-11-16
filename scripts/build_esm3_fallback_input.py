"""Write the protein records that ESMFold2 failed to fold, as input for ESM3 structure generation.

A record is a failure when its sequence hash is in failures.parquet or in a failure line of
.esmfold2_rows/*.jsonl, and none of its accessions is in manifest.parquet or has an ESMFold2 cif.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from src.data.components.embedders.utils import sha256_short


def failed_sequence_hashes(esmfold2_dir: Path) -> set[str]:
    failures_path = esmfold2_dir / "failures.parquet"
    hashes = (
        set(pd.read_parquet(failures_path, columns=["sequence_hash"])["sequence_hash"])
        if failures_path.exists()
        else set()
    )
    for path in (esmfold2_dir / ".esmfold2_rows").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a kill
            if entry["kind"] == "failure":
                hashes.add(entry["row"]["sequence_hash"])
    return hashes


def folded_acc_ids(esmfold2_dir: Path) -> set[str]:
    manifest_path = esmfold2_dir / "manifest.parquet"
    folded = (
        set(pd.read_parquet(manifest_path, columns=["acc_id"])["acc_id"])
        if manifest_path.exists()
        else set()
    )
    return folded | {
        path.stem.removeprefix("ESMFold2-") for path in esmfold2_dir.glob("ESMFold2-*.cif")
    }


def build_fallback_records(esmfold2_dir: Path, input_pkl: Path) -> list[dict]:
    with open(input_pkl, "rb") as handle:
        records = pickle.load(handle)
    failed = failed_sequence_hashes(esmfold2_dir)
    folded = folded_acc_ids(esmfold2_dir)
    return [
        record
        for record in records
        if sha256_short(record["sequence"]) in failed and folded.isdisjoint(record["uniprot_id"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--esmfold2-dir", type=Path, default=Path("data/intermediate/esmfold2_cif")
    )
    parser.add_argument(
        "--input-pkl", type=Path, default=Path("data/intermediate/esmfold2_long_proteins.pkl")
    )
    parser.add_argument("--output-pkl", type=Path, required=True)
    args = parser.parse_args()

    records = build_fallback_records(args.esmfold2_dir, args.input_pkl)
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as handle:
        pickle.dump(records, handle)
    lengths = [len(record["sequence"]) for record in records]
    span = f"; lengths {min(lengths):,} to {max(lengths):,}" if lengths else ""
    print(f"Wrote {len(records):,} records to {args.output_pkl}{span}")


if __name__ == "__main__":
    main()
