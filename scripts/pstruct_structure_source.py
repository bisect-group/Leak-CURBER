"""Structure source of each P-struct (enzyme_structure_splits) partition, row-based and protein-based.

The structure of a row is the one the structure-based baselines use (`choose_structures` of
build_baseline_mlp_data.py): the row's best experimental chain, else AlphaFold, ESMFold2, ESM3.
A protein is a unique sequence; it takes the best source among its rows. A predicted structure has
low pLDDT when its mean pLDDT is not >= 70, the split's rule (`predicted_mean_plddt.tsv`).
Percentages of the sources and of "none" (no structure file) are of all rows or proteins; the
low-pLDDT percentage is of the predicted-structure rows or proteins.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
from pathlib import Path

import polars as pl

from scripts.build_baseline_mlp_data import (
    DATASETS,
    INTERMEDIATE,
    PARTS,
    RETRIEVAL_DATASET,
    SOURCE_RANK,
    STORE_SOURCES,
    choose_structures,
    read_3d,
    regression_source,
    with_pdb_ids,
)
from src.data.components.splitters.protein_struct_lddt import PREDICTED_SOURCES

MIN_MEAN_PLDDT = 70
SOURCES = [*SOURCE_RANK, "none"]
# subtask -> (dataset directory, 3D parquet, P-struct threshold)
CLASSIFICATION = DATASETS / "enzyme_classification_dataset"
RETRIEVAL = DATASETS / RETRIEVAL_DATASET / "with_protein_information"
SUBTASKS = {
    "k_cat": (*regression_source("kcat"), 0.625),
    "K_M": (*regression_source("km"), 0.69),
    "K_i": (*regression_source("ki"), 0.705),
    "K_d": (*regression_source("kd"), 0.545),
    "EC50": (*regression_source("ec50"), 0.445),
    "IC50": (*regression_source("ic50"), 0.695),
    "classification": (CLASSIFICATION, CLASSIFICATION / "enzyme_classification_dataset_3d.parquet", 0.855),
    "retrieval": (RETRIEVAL, RETRIEVAL / "enzyme_retrieval_dataset_3d.parquet", 0.86),
}


def source_table(subtask: str, dataset_dir: Path, source_3d: Path, threshold: float, dirs: dict[str, Path]) -> pl.DataFrame:
    split_dir = dataset_dir / "enzyme_structure_splits"
    plddt = pl.read_csv(split_dir / "predicted_mean_plddt.tsv", separator="\t", schema_overrides={"structure_id": pl.String})
    low = set(plddt.filter(~(pl.col("mean_plddt") >= MIN_MEAN_PLDDT))["structure_id"])
    frame_3d, key_columns = read_3d(source_3d)
    selection = choose_structures(frame_3d.select("row_key", "pdbs", "pdb_type", "pdb_chain", "pdb_coverage"), **dirs)
    rows = []
    for part in PARTS:
        frame = with_pdb_ids(
            pl.read_parquet(split_dir / f"threshold_{threshold}" / f"{part}.parquet"), selection, key_columns
        ).with_columns(key=pl.col("pdb_ids").list.first())
        frame = frame.with_columns(
            source=pl.col("key").str.extract(r"\|([a-z0-9]+)$", 1).replace_strict(STORE_SOURCES, default="none"),
            low=pl.col("key").str.extract(r"^(.*)\|", 1).is_in(low),
        ).with_columns(rank=pl.col("source").replace_strict({s: i for i, s in enumerate(SOURCES)}))
        proteins = frame.sort("rank").unique("sequence", keep="first")  # best source per protein
        row = {"subtask": subtask, "partition": part, "n_rows": frame.height, "n_proteins": proteins.height}
        for level, data in (("row", frame), ("protein", proteins)):
            predicted = data.filter(pl.col("source").is_in(PREDICTED_SOURCES))
            for s in SOURCES:
                row[f"pct_{s}_{level}"] = 100 * data.filter(pl.col("source") == s).height / data.height
            row[f"pct_low_plddt_{level}"] = 100 * predicted["low"].sum() / max(predicted.height, 1)
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_csv", type=Path, default=Path("outputs/pstruct_structure_source.csv"))
    args = parser.parse_args()
    dirs = {
        "af_dir": INTERMEDIATE / "alphafold",
        "esmfold2_dir": INTERMEDIATE / "esmfold2_pdb",
        "esm_dir": INTERMEDIATE / "esm",
        "mmcif_dir": INTERMEDIATE / "mmcif",
    }
    table = pl.concat(source_table(name, *spec, dirs) for name, spec in SUBTASKS.items())
    table = table.with_columns(pl.col(pl.Float64).round(1))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.out_csv)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=300):
        print(table)


if __name__ == "__main__":
    main()
