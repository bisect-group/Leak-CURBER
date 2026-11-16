"""Build the MLP baseline input data under outputs/baselines/MLP/data.

Copies the split parquets of the six regression endpoints and of enzyme classification, and adds
`pdb_ids`: one ESM3 structure key per row, as a list of length 1, or [] without a structure.
A row is matched to its 3D rows (the candidate list of `assign_experimental_and_af_pdbs`) by all
its non-structure columns. The structure is the first available of: the row's experimental chain
with the highest coverage (ties: smallest structure id), AlphaFold (highest version, canonical
`AF-<accession>-F1` models only), ESMFold2, ESM3. The files are located by the structure
split's `locate_structures`.

Layout:
  <out_root>/regression/baselines_<endpoint>/<split>[/threshold_*]/{train,val,test}.parquet
  <out_root>/classification/enzyme_classification_dataset/<split>[/threshold_*]/...
  <out_root>/classification/enzyme_retrieval_dataset/<split>[/threshold_*]/... (plain copy)

Existing outputs are kept; `--overwrite` rebuilds them. `--verify` checks the keys against the
ESM3 store and the experimental chains against `select_chains`, and writes `structure_report.csv`.
"""

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

from src.data.components.embedders.store import load_index
from src.data.components.embedders.utils import sha256_short
from src.data.components.splitters.protein_struct_lddt import locate_structures
from src.utils.structure_utils import select_chains

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = REPO_ROOT / "data/processed/datasets"
INTERMEDIATE = REPO_ROOT / "data/intermediate"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs/baselines/MLP/data"
ESM3_STORE = INTERMEDIATE / "embeddings/esm3/open_small_structure/v1"
UNIPROT_ACC_DIR = REPO_ROOT / "data/raw/uniprot/acc"

REGRESSION_FAMILIES = {
    "kcat": "kinetic_dataset",
    "km": "kinetic_dataset",
    "ki": "kinetic_dataset",
    "kd": "binding_affinity_dataset",
    "ic50": "binding_affinity_dataset",
    "ec50": "binding_affinity_dataset",
}
REGRESSION_SPLITS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "substrate_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
    "conformer_cosine_splits",
]
CLASSIFICATION_DATASET = "enzyme_classification_dataset"
CLASSIFICATION_SPLITS = [
    "random_splits_grouped_sequence",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
]
RETRIEVAL_DATASET = "enzyme_retrieval_dataset"
RETRIEVAL_SPLITS = [
    "random_splits_grouped_rxn_key",
    "uniprot_time_splits",
    "enzyme_structure_splits",
    "reaction_drfp_tanimoto_splits",
]
PARTS = ("train", "val", "test")
# Preference order of the structure sources, and the suffix of their ESM3 store keys.
SOURCE_RANK = {"experimental": 0, "alphafold": 1, "esmfold2": 2, "esm3": 3}
STORE_SUFFIX = {"experimental": "experimental", "alphafold": "alphafold", "esmfold2": "esmfold2", "esm3": "esm"}
STORE_SOURCES = {suffix: source for source, suffix in STORE_SUFFIX.items()}
STRUCTURE_COLUMNS = ["pdbs", "pdb_source", "pdb_type", "pdb_chain", "pdb_coverage"]
CANDIDATE_COLUMNS = ["row_key", "pdbs", "pdb_type", "pdb_chain", "pdb_coverage"]


def choose_structures(candidates: pl.DataFrame, **dirs: Path) -> pl.DataFrame:
    """One structure per row from the row's own candidates.

    `candidates` holds `row_key`, `pdbs`, `pdb_type`, `pdb_chain` and `pdb_coverage`: the rows
    of the 3D parquet. Returns `row_key`, `pdb_id` (ESM3 store key), `source` and `chain`; rows
    without a structure file are absent. `dirs` are the arguments of `locate_structures` after the ids.
    Isoform and fragment AlphaFold models are not used.
    """
    candidates = candidates.drop_nulls("pdbs").unique()
    experimental = candidates["pdb_type"] == "experimental"
    located = locate_structures(
        sorted(candidates.filter(experimental)["pdbs"].unique()),
        sorted(candidates.filter(~experimental.fill_null(False))["pdbs"].unique()),
        canonical_alphafold_only=True,
        **dirs,
    )
    found = pl.DataFrame(
        {"pdbs": list(located), "source": [source for _, source in located.values()]},
        schema={"pdbs": pl.String, "source": pl.String},
    )
    return (
        candidates.join(found, on="pdbs")
        .with_columns(
            rank=pl.col("source").replace_strict(SOURCE_RANK, return_dtype=pl.Int8),
            coverage=pl.col("pdb_coverage").fill_null(0.0),
        )
        .sort(["row_key", "rank", "coverage", "pdbs"], descending=[False, False, True, False])
        .unique("row_key", keep="first", maintain_order=True)
        .select(
            "row_key",
            pl.format("{}|{}", "pdbs", pl.col("source").replace_strict(STORE_SUFFIX)).alias("pdb_id"),
            "source",
            pl.col("pdb_chain").alias("chain"),
        )
    )


def with_pdb_ids(frame: pl.DataFrame, selection: pl.DataFrame, key_columns: list[str]) -> pl.DataFrame:
    """`frame` plus `pdb_ids`: a one-element list with the row's structure key, else [].

    A row is identified by its `key_columns` (every column of the 3D row except the structure ones).
    """
    keyed = frame.with_columns(row_key=pl.struct(key_columns).hash())
    joined = keyed.join(selection.select("row_key", "pdb_id"), on="row_key", how="left")
    return joined.with_columns(
        pdb_ids=pl.when(pl.col("pdb_id").is_null())
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.concat_list("pdb_id"))
    ).drop("pdb_id", "row_key")


def split_dirs(dataset_dir: Path, splits: list[str]) -> list[Path]:
    """Directories under the split families that hold train, val and test."""
    found = []
    for split in splits:
        root = dataset_dir / split
        found += [d for d in [root, *sorted(root.glob("threshold_*"))] if all((d / f"{p}.parquet").exists() for p in PARTS)]
    return found


def write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.write_parquet(tmp, compression="zstd", compression_level=3)
    tmp.replace(path)


def read_3d(source_3d: Path) -> tuple[pl.DataFrame, list[str]]:
    """The 3D rows with `row_key`, and the columns that identify a row."""
    frame = pl.read_parquet(source_3d)
    key_columns = [c for c in frame.columns if c not in STRUCTURE_COLUMNS]
    return frame.with_columns(row_key=pl.struct(key_columns).hash()), key_columns


def build_dataset(
    source_dir: Path, source_3d: Path, out_dir: Path, splits: list[str], overwrite: bool, dirs: dict[str, Path]
) -> None:
    targets = {
        (split_dir, part): out_dir / split_dir.relative_to(source_dir) / f"{part}.parquet"
        for split_dir in split_dirs(source_dir, splits)
        for part in PARTS
    }
    if not overwrite:
        targets = {k: v for k, v in targets.items() if not v.exists()}
    if not targets:
        return
    frame_3d, key_columns = read_3d(source_3d)
    selection = choose_structures(frame_3d.select(CANDIDATE_COLUMNS), **dirs)
    print(f"{out_dir.name}: {selection.height:,} of {frame_3d['row_key'].n_unique():,} rows have a structure")
    row_keys = frame_3d["row_key"].unique()
    for (split_dir, part), target in targets.items():
        frame = pl.read_parquet(split_dir / f"{part}.parquet")
        unmatched = frame.select((~pl.struct(key_columns).hash().is_in(row_keys.implode())).sum()).item()
        if unmatched:
            raise ValueError(f"{split_dir / part}: {unmatched} rows have no row in {source_3d.name}")
        write_parquet(with_pdb_ids(frame, selection, key_columns), target)


def copy_dataset(source_dir: Path, out_dir: Path, splits: list[str], overwrite: bool) -> None:
    for split_dir in split_dirs(source_dir, splits):
        for part in PARTS:
            target = out_dir / split_dir.relative_to(source_dir) / f"{part}.parquet"
            if overwrite or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(split_dir / f"{part}.parquet", target)


def regression_source(endpoint: str) -> tuple[Path, Path]:
    source_dir = DATASETS / REGRESSION_FAMILIES[endpoint] / endpoint
    (source_3d,) = source_dir.glob("*_3d.parquet")
    return source_dir, source_3d


def build_all(out_root: Path, endpoints: list[str], overwrite: bool, dirs: dict[str, Path]) -> None:
    for endpoint in endpoints:
        source_dir, source_3d = regression_source(endpoint)
        build_dataset(
            source_dir, source_3d, out_root / "regression" / f"baselines_{endpoint}", REGRESSION_SPLITS, overwrite, dirs
        )
    classification_dir = DATASETS / CLASSIFICATION_DATASET
    build_dataset(
        classification_dir,
        classification_dir / f"{CLASSIFICATION_DATASET}_3d.parquet",
        out_root / "classification" / CLASSIFICATION_DATASET,
        CLASSIFICATION_SPLITS,
        overwrite,
        dirs,
    )
    copy_dataset(
        DATASETS / RETRIEVAL_DATASET, out_root / "classification" / RETRIEVAL_DATASET, RETRIEVAL_SPLITS, overwrite
    )


def verify(out_root: Path, endpoints: list[str], store: Path, mmcif_dir: Path, uniprot_dir: Path) -> pl.DataFrame:
    """Per dataset and split family: rows, proteins, source counts, keys missing from the store."""
    store_hashes = set(load_index(store, columns=["key_hash"])["key_hash"])
    datasets = [
        (f"regression/{e}", out_root / "regression" / f"baselines_{e}", REGRESSION_SPLITS, regression_source(e)[1])
        for e in endpoints
    ]
    classification_dir = DATASETS / CLASSIFICATION_DATASET
    datasets.append(
        (
            "classification",
            out_root / "classification" / CLASSIFICATION_DATASET,
            CLASSIFICATION_SPLITS,
            classification_dir / f"{CLASSIFICATION_DATASET}_3d.parquet",
        )
    )
    rows = []
    for name, dataset_dir, splits, source_3d in datasets:
        frames = {
            split_dir: pl.concat(
                [pl.read_parquet(split_dir / f"{p}.parquet", columns=["sequence", "pdb_ids"]) for p in PARTS]
            ).with_columns(key=pl.col("pdb_ids").list.first())
            for split_dir in split_dirs(dataset_dir, splits)
        }
        keys = set().union(*(frame["key"].drop_nulls() for frame in frames.values()))
        missing_keys = {key for key in keys if sha256_short(key) not in store_hashes}
        expected = dict(
            pl.read_parquet(source_3d, columns=["pdbs", "pdb_chain"]).filter(pl.col("pdb_chain").is_not_null()).unique().rows()
        )
        pairs = [tuple(key.removesuffix("|experimental").split(":", 1)) for key in keys if key.endswith("|experimental")]
        sequences = {}
        for acc in {acc for _, acc in pairs}:
            path = uniprot_dir / f"{acc}.json"
            if path.exists() and (sequence := json.loads(path.read_text()).get("sequence")):
                sequences[acc] = sequence
        chains = select_chains(pairs, sequences, mmcif_dir)
        mismatched = sum(chains[pdb.upper(), acc][0] != expected.get(f"{pdb}:{acc}") for pdb, acc in pairs)
        for split_dir, frame in frames.items():
            by_source = (
                frame.with_columns(
                    source=pl.col("key").str.extract(r"\|([a-z0-9]+)$", 1).replace_strict(STORE_SOURCES, default="none")
                )
                .group_by("source")
                .agg(rows=pl.len(), proteins=pl.col("sequence").n_unique())
            )
            counts = {s: (0, 0) for s in [*SOURCE_RANK, "none"]} | {
                r["source"]: (r["rows"], r["proteins"]) for r in by_source.iter_rows(named=True)
            }
            rows.append(
                {
                    "dataset": name,
                    "split": str(split_dir.relative_to(dataset_dir)),
                    "rows": frame.height,
                    "proteins": frame["sequence"].n_unique(),
                    **{f"{s}_rows": c[0] for s, c in counts.items()},
                    **{f"{s}_proteins": c[1] for s, c in counts.items()},
                    "keys_missing_from_store": len(missing_keys & set(frame["key"].drop_nulls())),
                    "rows_missing_from_store": frame.filter(pl.col("key").is_in(missing_keys)).height,
                    "chain_mismatches_dataset": mismatched,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--endpoints", nargs="+", default=list(REGRESSION_FAMILIES), choices=list(REGRESSION_FAMILIES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--alphafold_dir", type=Path, default=INTERMEDIATE / "alphafold")
    parser.add_argument("--esmfold2_dir", type=Path, default=INTERMEDIATE / "esmfold2_pdb")
    parser.add_argument("--esm_dir", type=Path, default=INTERMEDIATE / "esm")
    parser.add_argument("--mmcif_dir", type=Path, default=INTERMEDIATE / "mmcif")
    parser.add_argument("--esm3_store", type=Path, default=ESM3_STORE)
    parser.add_argument("--uniprot_acc_dir", type=Path, default=UNIPROT_ACC_DIR)
    args = parser.parse_args()

    dirs = {
        "af_dir": args.alphafold_dir,
        "esmfold2_dir": args.esmfold2_dir,
        "esm_dir": args.esm_dir,
        "mmcif_dir": args.mmcif_dir,
    }
    build_all(args.out_root, args.endpoints, args.overwrite, dirs)
    if args.verify:
        report = verify(args.out_root, args.endpoints, args.esm3_store, args.mmcif_dir, args.uniprot_acc_dir)
        report.write_csv(args.out_root / "structure_report.csv")
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
            print(report.filter(pl.col("split") == "random_splits_grouped_sequence").drop("split"))


if __name__ == "__main__":
    main()
