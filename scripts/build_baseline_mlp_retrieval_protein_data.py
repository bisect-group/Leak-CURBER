"""Build the MLP retrieval data on the protein-based splits of Horizyn, CLIPZyme and CREEP.

Reads data/processed/datasets/enzyme_retrieval_dataset/with_protein_information/<split>[/threshold_*],
drops rows without a protein sequence, and keeps one row per (rxn_smiles, ec_number).

Layout: <out_root>/enzyme_retrieval_dataset/<split>[/threshold_*]/{train,val,test}.parquet
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
from pathlib import Path

import polars as pl

from scripts.build_baseline_mlp_data import DATASETS, PARTS, REPO_ROOT, RETRIEVAL_DATASET, write_parquet

SOURCE_DIR = DATASETS / RETRIEVAL_DATASET / "with_protein_information"
SPLITS = ["random_splits_grouped_rxn_key", "uniprot_time_splits", "reaction_drfp_tanimoto_splits"]
THRESHOLDS = {"reaction_drfp_tanimoto_splits": "threshold_0.425"}
COLUMNS = ["rxn_key", "rxn_smiles", "ec_number", "uniprot_date"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out_root", type=Path, default=REPO_ROOT / "outputs/baselines/MLP/data/classification_protein"
    )
    args = parser.parse_args()
    for split in SPLITS:
        relative = Path(split) / THRESHOLDS.get(split, "")
        for part in PARTS:
            frame = (
                pl.read_parquet(SOURCE_DIR / relative / f"{part}.parquet")
                .drop_nulls("sequence")
                .select(COLUMNS)
                .unique(["rxn_smiles", "ec_number"], keep="first", maintain_order=True)
            )
            write_parquet(frame, args.out_root / RETRIEVAL_DATASET / relative / f"{part}.parquet")
            print(f"{relative}/{part}: {frame.height:,} rows, {frame['rxn_smiles'].n_unique():,} reactions")


if __name__ == "__main__":
    main()
