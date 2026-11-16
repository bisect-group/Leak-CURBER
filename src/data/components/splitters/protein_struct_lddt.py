import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import multiprocessing
import time
import os
import math
import hashlib
import json
import shutil
import subprocess
import numpy as np
import polars as pl
import biotite.structure.io.pdbx as pdbx
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from src.utils.rich_progress import progress, write
from omegaconf import DictConfig
from src.utils.rich_progress import component_logger
from src.utils.structure_utils import mmcif_file
from src.data.components.splitters.base import (
    configured,
    BaseThresholdedSimilaritySplitter,
    greedy_fill,
)


# Threshold-independent search settings; part of the cache fingerprint.
FOLDSEEK_SEARCH_ARGS = [
    # Cov-mode 2 covers the shorter chain: a partial experimental chain (one domain, under 80%
    # of the protein) still hits the full-length structures that contain it. Pairs of
    # full-length chains still need 80% coverage.
    "--cov-mode", "2", "-c", "0.8", "--max-seqs", "1000000000",
    "-s", "9.5", "--alignment-mode", "3", "--alignment-type", "2",
    # Every hit is named <file>_<chain>, also for single-chain files.
    "--chain-name-mode", "1",
]


PREDICTED_SOURCES = ("alphafold", "esmfold2", "esm3")


def locate_structures(
    experimental_ids: list[str],
    predicted_ids: list[str],
    *,
    af_dir: Path,
    esmfold2_dir: Path,
    esm_dir: Path,
    mmcif_dir: Path,
    canonical_alphafold_only: bool = False,
) -> dict[str, tuple[Path, str]]:
    """Structure file and source of each id that has one.

    An experimental id `<PDB>:<accession>` is its mmCIF file. A predicted id is an accession; it
    falls back from the highest AlphaFold version to ESMFold2 to ESM3. Ids without a file are absent.
    `canonical_alphafold_only` skips isoform and fragment models (`AF-<accession>-<n>-F1-model_v*.pdb`).
    """
    af_versions: dict[str, tuple[int, Path]] = {}
    for path in Path(af_dir).glob("AF-*-F1-model_v*.pdb"):
        parts = path.stem.split("-")
        if len(parts) >= 3 and not (canonical_alphafold_only and len(parts) != 4):
            version = int(parts[-1].split("_v")[-1])
            if parts[1] not in af_versions or version > af_versions[parts[1]][0]:
                af_versions[parts[1]] = (version, path)
    predicted_locators = {
        "alphafold": lambda acc: af_versions[acc][1] if acc in af_versions else None,
        "esmfold2": lambda acc: Path(esmfold2_dir) / f"ESMFold2-{acc}.pdb",
        "esm3": lambda acc: Path(esm_dir) / f"ESM3-open-small-{acc}.pdb",
    }
    located = {}
    for sid in experimental_ids:
        path = mmcif_file(mmcif_dir, sid.partition(":")[0])
        if path is not None and path.exists():
            located[sid] = (path, "experimental")
    for acc in predicted_ids:
        for source, locate in predicted_locators.items():
            path = locate(acc)
            if path is not None and path.exists():
                located[acc] = (path, source)
                break
    return located


def mean_plddt(path: Path) -> float:
    """Mean CA B-factor of a predicted structure on a 0-100 scale.

    NaN means unknown: no CA atoms, a constant B-factor (a default value, not a pLDDT), or an
    unreadable file.
    """
    try:
        if path.suffix == ".cif":
            atoms = pdbx.CIFFile.read(path).block["atom_site"]
            names = atoms["label_atom_id" if "label_atom_id" in atoms else "auth_atom_id"]
            scores = atoms["B_iso_or_equiv"].as_array(float)[names.as_array(str) == "CA"]
        else:
            with open(path) as handle:
                scores = np.array(
                    [float(line[60:66]) for line in handle if line.startswith("ATOM") and line[12:16] == " CA "]
                )
    except Exception:
        return math.nan
    if scores.size == 0 or np.ptp(scores) == 0:
        return math.nan
    mean = float(scores.mean())
    return mean * 100 if mean <= 1.0 else mean


class ProteinStructMaxLDDTSimilaritySplitter(BaseThresholdedSimilaritySplitter):
    """Splits a dataset based on protein structure LDDT similarity using Foldseek.

    This class provides methods to split a dataset into training, validation,
    and test sets based on the structural similarity (LDDT score) of proteins.

    The approach:
    1. Compute max LDDT similarity of each protein to ALL other proteins in dataset
    2. Proteins with low max similarity (structurally dissimilar) go to test first and validation second
    3. This ensures test proteins are structurally distinct from training set

    LDDT (Local Distance Difference Test) is a superposition-free score that
    evaluates local distance differences of all atoms in a model. It ranges
    from 0 to 1, where 1 indicates perfect structural similarity.

    This class uses Foldseek for fast structure comparison, which requires a
    structure file for each protein. It supports experimental mmCIF chains (the
    chain in `pdb_chain`, one structure per `<PDB>:<base UniProt accession>`), AlphaFold-predicted
    structures, and ESM-predicted structures (one structure per accession).

    Datasets with nearly the same proteins can share one Foldseek all-vs-all. Set
    `structure_split.shared_search_inputs` (the 3D parquets whose structures form the search set)
    and `structure_split.shared_search_dir` identically for each of them. The search then runs
    over the union of those structures in `shared_search_dir`, and the first dataset to run
    caches the TSV there under a fingerprint of the union. Later datasets reuse it, keep only
    the hits between their own structures, and drop same-sequence hits by their own sequences.
    The max-lDDT and pLDDT caches and the split outputs stay in each dataset's `output_dir`.
    The datasets must run one after another: concurrent runs would write the shared directory
    at the same time.
    """

    # Union of the shared search inputs; None when the dataset searches only its own structures.
    _shared_search_df: pl.DataFrame | None = None

    def __init__(self, cfg: DictConfig):
        struct_cfg = cfg.splits.dataset.structure_split

        self.TEST_FRAC = configured(cfg.splits.test_frac, 0.1)
        self.VAL_FRAC = configured(cfg.splits.valid_frac, 0.1)
        self.CACHE_OVERWRITE = cfg.splits.get("similarity_cache_overwrite", False)
        self.DROP_TO_1D = bool(struct_cfg.get("drop_to_1d", False))
        self.MIN_MEAN_PLDDT = float(struct_cfg.get("min_mean_plddt", 70))
        self.THRESHOLDS = list(
            struct_cfg.get("similarity_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )

        # PDB source paths
        self.AF_PDB_PATH = Path(cfg.splits.af_pdb_path)
        self.ESMFOLD2_PDB_PATH = Path(cfg.splits.esmfold2_pdb_path)
        self.ESM_PDB_PATH = Path(cfg.splits.esm_pdb_path)
        self.MMCIF_PATH = Path(cfg.splits.mmcif_path)

        # Foldseek binary and mode
        self.FOLDSEEK_BIN = Path(cfg.splits.foldseek_bin)
        self.GPU = bool(cfg.splits.get("foldseek_gpu", False))
        # Query chunks bound the Foldseek temp disk; the hits are identical to one search.
        self.SEARCH_QUERY_CHUNK_SIZE = struct_cfg.get("search_query_chunk_size")

        self.INPUT_DATASET_PATH = Path(struct_cfg.input_dataset_parquet_file_path)
        drop_to_1d_path = struct_cfg.get("drop_to_1d_dataset_parquet_file_path")
        self.DROP_TO_1D_DATASET_PATH = (
            Path(drop_to_1d_path) if drop_to_1d_path else None
        )
        self.OUTPUT_DIR = Path(struct_cfg.output_dir)

        # Foldseek scratch and PDB symlink directories are per split, so split runs can overlap,
        # unless the search is shared; then they live in the shared directory.
        shared_inputs = [Path(p) for p in struct_cfg.get("shared_search_inputs") or []]
        search_dir = Path(struct_cfg.shared_search_dir) if shared_inputs else self.OUTPUT_DIR
        self.FOLDSEEK_ALL_PATH = search_dir / "foldseek_work"
        self.FOLDSEEK_TMPDIR = search_dir / "foldseek_tmp"
        self.ALL_PDB_SYMLINK_PATH = search_dir / "pdb_symlinks"
        self.UNIQUE_STRUCTURE_SIMILARITY_PLOT_PATH = Path(struct_cfg.unique_similarity_plot_path)
        self.FULL_DATASET_STRUCTURE_SIMILARITY_PLOT_PATH = Path(
            struct_cfg.full_dataset_similarity_plot_path
        )

        LOG_PATH = Path(cfg.splits.log_dir)

        # Create all necessary directories
        for path in [
            LOG_PATH,
            self.OUTPUT_DIR,
            self.UNIQUE_STRUCTURE_SIMILARITY_PLOT_PATH.parent,
            self.FULL_DATASET_STRUCTURE_SIMILARITY_PLOT_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = component_logger(
            log_dir=LOG_PATH,
            log_file_name=cfg.splits.log_file_name,
        ).get_logger()

        if not (self.FOLDSEEK_BIN.is_file() and os.access(self.FOLDSEEK_BIN, os.X_OK)):
            msg = (
                f"Foldseek binary not found or not executable at {self.FOLDSEEK_BIN}. "
                "Run: bash scripts/install_external_tools.sh"
            )
            self.logger.error(msg)
            raise EnvironmentError(msg)
        self.TOOL_VERSION = subprocess.run(
            [str(self.FOLDSEEK_BIN), "version"], capture_output=True, text=True
        ).stdout.strip()
        self.logger.info(
            f"Foldseek binary: {self.FOLDSEEK_BIN}, version: {self.TOOL_VERSION}, "
            f"mode: {'GPU' if self.GPU else 'CPU'}"
        )

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet file not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        if (
            self.DROP_TO_1D
            and self.DROP_TO_1D_DATASET_PATH is not None
            and not self.DROP_TO_1D_DATASET_PATH.exists()
        ):
            msg = (
                "1D dataset parquet file not found at "
                f"{self.DROP_TO_1D_DATASET_PATH}"
            )
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Loading full dataset...")
        self.full_dataset_df = pl.read_parquet(self.INPUT_DATASET_PATH)
        self.logger.info(f"Full dataset shape: {self.full_dataset_df.shape}")

        if "pdb_chain" not in self.full_dataset_df.columns:
            msg = "Dataset lacks column 'pdb_chain'; rebuild the 3D dataset."
            self.logger.error(msg)
            raise ValueError(msg)
        # `pdbs` is the structure key: `<PDB>:<base UniProt accession>` for experimental rows,
        # else the accession.
        self.full_dataset_df = self.full_dataset_df.with_columns(
            pl.col("pdbs").alias("structure_id")
        )
        self.unique_structures = (
            self.full_dataset_df.get_column("structure_id")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )
        self.logger.info(f"Unique structures: {len(self.unique_structures)}")
        if shared_inputs:
            self._shared_search_df = (
                pl.concat(
                    [
                        pl.read_parquet(path, columns=["pdbs", "pdb_type", "pdb_chain"])
                        for path in shared_inputs
                    ]
                )
                .select("pdbs", pl.col("pdbs").alias("structure_id"), "pdb_type", "pdb_chain")
                .drop_nulls("structure_id")
                .unique()
            )
            clashes = (
                self._shared_search_df.filter(pl.col("pdb_type") == "experimental")
                .group_by("structure_id")
                .agg(pl.col("pdb_chain").n_unique().alias("n"))
                .filter(pl.col("n") > 1)
            )
            if clashes.height:
                raise ValueError(
                    f"{clashes.height} experimental structures map to several chains across "
                    f"shared_search_inputs, e.g. {clashes['structure_id'].head(3).to_list()}"
                )
            self.logger.info(
                f"Shared search over {len(self.search_structures)} structures from "
                f"{len(shared_inputs)} datasets, in {search_dir}"
            )
        self.logger.info(f"drop_to_1d={self.DROP_TO_1D}")
        if self.DROP_TO_1D_DATASET_PATH is not None:
            self.logger.info(
                "Using companion 1D source for split output: "
                f"{self.DROP_TO_1D_DATASET_PATH}"
            )
        elif self.DROP_TO_1D:
            self.logger.warning(
                "drop_to_1d has no companion 1D source; falling back to projection "
                "of the 3D dataset. Exact source-row recovery is unavailable."
            )

        # Will be computed lazily and cached for reuse
        # Only store max LDDT per protein (O(n) memory instead of O(n²) for pairwise)
        self._pdb_paths = None
        self._max_lddt_to_dataset = None  # dict: protein_id -> max_lddt
        self._max_lddt_array = None  # array in same order as self.unique_structures
        self._grouped_dataset_df = None
        self.ALL_VS_ALL_CACHE_PATH = search_dir / "foldseek_all_vs_all_lddt.tsv"
        self.MAX_SIM_CACHE_PATH = self.OUTPUT_DIR / "max_lddt_similarities.tsv"
        self.PLDDT_CACHE_PATH = self.OUTPUT_DIR / "predicted_mean_plddt.tsv"

    @property
    def search_df(self) -> pl.DataFrame:
        """Structures that Foldseek searches: the dataset's own, or the union of the shared inputs."""
        if self._shared_search_df is None:
            return self.full_dataset_df
        return self._shared_search_df

    @property
    def search_structures(self) -> list:
        if self._shared_search_df is None:
            return self.unique_structures
        return sorted(self._shared_search_df["structure_id"].unique())

    def _similarity_fingerprint(self) -> str:
        # "symmetric" marks the max over query and target hits; "nan_missing" marks NaN for no structure.
        return self._cache_fingerprint(
            self.search_structures,
            search=FOLDSEEK_SEARCH_ARGS,
            max_mode="symmetric",
            missing="nan_missing",
            structure_files=self._pdb_file_stats(),
            chains=self._selected_chains(),
            gpu=self.GPU,
            version=self.TOOL_VERSION,
            search_db="selected_chains",
        )

    def _max_similarity_fingerprint(self) -> str:
        # Hits between structures that share a sequence are excluded from the max.
        # A shared search covers more structures than the dataset; its own structures are part of the key.
        own = {"own_structures": self.unique_structures} if self._shared_search_df is not None else {}
        pairs = self.full_dataset_df.select("structure_id", "sequence").drop_nulls().unique().rows()
        return self._cache_fingerprint(
            [self._similarity_fingerprint()],
            self_hits="same_sequence_excluded",
            structure_sequences=self._cache_fingerprint(f"{sid}\t{seq}" for sid, seq in pairs),
            **own,
        )

    def _selected_chains(self) -> list:
        """Experimental structure ids with their chains; the selection is part of the cache key."""
        return (
            self.search_df.filter(pl.col("pdb_type") == "experimental")
            .select("structure_id", "pdb_chain")
            .unique()
            .sort("structure_id")
            .rows()
        )

    def _create_pdb_symlink(
        self, structure_id: str, source_path: Path, pdb_paths: dict
    ) -> None:
        """Link a structure file; one `<PDB>.cif` link serves every accession of the entry.

        Experimental ids are `<PDB>:<accession>`; a predicted structure is linked as `<id>.<ext>`.
        """
        name = (
            f"{structure_id.partition(':')[0]}.cif"
            if ":" in structure_id
            else f"{structure_id}{source_path.suffix}"
        )
        symlink_path = self.ALL_PDB_SYMLINK_PATH / name
        if not symlink_path.exists():
            symlink_path.symlink_to(source_path.resolve())
        pdb_paths[structure_id] = symlink_path

    def _pdb_source_paths(self, df: pl.DataFrame) -> tuple[dict, dict, dict]:
        """Locate each structure file on disk without touching the symlink directory.

        Returns (structure id -> source path, missing counts, requested counts) by structure kind.
        """
        experimental_pdbs = sorted(
            df.filter(pl.col("pdb_type") == "experimental")
            .get_column("structure_id")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )
        # Every non-experimental structure falls back from AlphaFold to ESMFold2 to ESM3.
        predicted_pdbs = sorted(
            df.filter(pl.col("pdb_type").fill_null("") != "experimental")
            .get_column("structure_id")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )

        self.logger.info("Indexing AlphaFold structures...")
        located = locate_structures(
            experimental_pdbs,
            predicted_pdbs,
            af_dir=self.AF_PDB_PATH,
            esmfold2_dir=self.ESMFOLD2_PDB_PATH,
            esm_dir=self.ESM_PDB_PATH,
            mmcif_dir=self.MMCIF_PATH,
        )
        used_counts = dict.fromkeys(["experimental", *PREDICTED_SOURCES], 0)
        for _, source in located.values():
            used_counts[source] += 1
        n_predicted_found = sum(used_counts[source] for source in PREDICTED_SOURCES)
        self.logger.info(f"Structure files used by source: {used_counts}")
        return (
            {sid: path for sid, (path, _) in located.items()},
            {
                "experimental": len(experimental_pdbs) - used_counts["experimental"],
                "predicted": len(predicted_pdbs) - n_predicted_found,
            },
            {"experimental": len(experimental_pdbs), "predicted": len(predicted_pdbs)},
        )

    def _pdb_file_stats(self) -> list:
        # The LDDT cache fingerprint must change when a structure file changes.
        # Key on every source file's size and mtime_ns (computed once per run).
        if getattr(self, "_pdb_stats", None) is None:
            sources, _, _ = self._pdb_source_paths(self.search_df)
            self._pdb_stats = [
                (pdb, path.stat().st_size, path.stat().st_mtime_ns)
                for pdb, path in sorted(sources.items())
            ]
        return self._pdb_stats

    def _predicted_mean_plddt(self) -> dict:
        """Mean pLDDT of every predicted structure; experimental files are not read.

        The values do not depend on the cut-off. They are cached per file path, size and mtime.
        """
        sources, _, _ = self._pdb_source_paths(self.full_dataset_df)
        experimental = set(
            self.full_dataset_df.filter(pl.col("pdb_type") == "experimental")
            .get_column("structure_id")
            .to_list()
        )
        files = {sid: path for sid, path in sorted(sources.items()) if sid not in experimental}
        fingerprint = self._cache_fingerprint(
            files,
            files=[(sid, str(p), p.stat().st_size, p.stat().st_mtime_ns) for sid, p in files.items()],
            reader="ca_bfactor_v2",
        )
        if self.PLDDT_CACHE_PATH.exists() and self._cache_matches(self.PLDDT_CACHE_PATH, fingerprint):
            cached = pl.read_csv(self.PLDDT_CACHE_PATH, separator="\t", schema_overrides={"structure_id": pl.Utf8})
            return dict(zip(cached["structure_id"], cached["mean_plddt"]))
        if not files:
            return {}
        self.logger.info(f"Reading mean pLDDT of {len(files):,} predicted structures...")
        with ProcessPoolExecutor(
            min(len(files), os.cpu_count() or 1), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            means = list(pool.map(mean_plddt, files.values(), chunksize=64))
        result = dict(zip(files, means))
        pl.DataFrame(
            {"structure_id": list(result), "mean_plddt": means},
            schema={"structure_id": pl.Utf8, "mean_plddt": pl.Float64},
        ).write_csv(self.PLDDT_CACHE_PATH, separator="\t")
        self._write_cache_fingerprint(self.PLDDT_CACHE_PATH, fingerprint)
        return result

    def _low_confidence_structures(self) -> set:
        return {
            sid for sid, mean in self._predicted_mean_plddt().items() if not mean >= self.MIN_MEAN_PLDDT
        }

    def _generate_pdb_paths(self, df: pl.DataFrame) -> dict:
        """
        Generate symlinks to PDB files for all proteins in the given DataFrame.
        Supports experimental mmCIF files, AlphaFold, ESMFold2 and ESM3 predicted structures.
        Symlinks are written to self.ALL_PDB_SYMLINK_PATH.

        Args:
            df: DataFrame containing at minimum 'structure_id' and 'pdb_type' columns.

        Returns:
            Dictionary mapping structure id -> symlink Path for every structure located on disk.
        """
        shutil.rmtree(self.ALL_PDB_SYMLINK_PATH, ignore_errors=True)
        self.ALL_PDB_SYMLINK_PATH.mkdir(parents=True, exist_ok=True)

        sources, missing_counts, totals = self._pdb_source_paths(df)
        pdb_paths = {}
        for sid, source in progress(sources.items(), desc="Preparing PDB paths", leave=False):
            self._create_pdb_symlink(sid, source, pdb_paths)

        for kind, label in (
            ("experimental", "experimental structures"),
            ("predicted", "predicted-structure proteins (AlphaFold, ESMFold2, ESM3)"),
        ):
            if missing_counts[kind] > 0 and totals[kind] > 0:
                pct = 100 * missing_counts[kind] / totals[kind]
                self.logger.warning(f"No PDB found for {missing_counts[kind]} {label} ({pct:.1f}%)")

        self.logger.info(f"Generated {len(pdb_paths)} PDB symlinks")
        return pdb_paths

    def _foldseek_create_database(self) -> int:
        """
        Create a Foldseek database from the PDB symlinks in self.ALL_PDB_SYMLINK_PATH,
        subset it to the selected chains, and index the subset; return the number of selected
        chains. The databases are written to self.FOLDSEEK_ALL_PATH ('foldseekDB' and 'selectedDB') and the temporary index
        scratch space is self.FOLDSEEK_TMPDIR (cleaned up on completion).
        """
        full_db = self.FOLDSEEK_ALL_PATH / "foldseekDB"
        db_path = self.FOLDSEEK_ALL_PATH / "selectedDB"

        shutil.rmtree(self.FOLDSEEK_ALL_PATH, ignore_errors=True)
        self.FOLDSEEK_ALL_PATH.mkdir(parents=True, exist_ok=True)

        n_files = len(list(self.ALL_PDB_SYMLINK_PATH.iterdir()))
        self.logger.info(
            f"[Foldseek createdb] Creating database from {n_files} structure files..."
        )
        self.logger.info(f"[Foldseek createdb] Input: {str(self.ALL_PDB_SYMLINK_PATH)}")
        self.logger.info(f"[Foldseek createdb] Output: {str(full_db)}")

        cmd = [
            str(self.FOLDSEEK_BIN),
            "createdb",
            str(self.ALL_PDB_SYMLINK_PATH),
            str(full_db),
            "--chain-name-mode", "1",
        ]

        start_time = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[Foldseek createdb] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[Foldseek createdb] Error: {out.stderr}")
            raise RuntimeError(f"Foldseek createdb failed: {out.stderr}")
        self.logger.info(f"[Foldseek createdb] Completed in {elapsed:.1f}s")

        # Only the selected chain of each structure is searched, so hits are never spent on
        # chains that are dropped. createsubdb keeps the keys and the hit names.
        lookup = pl.read_csv(
            f"{full_db}.lookup", separator="\t", has_header=False,
            new_columns=["key", "name", "file"], schema_overrides={"name": pl.Utf8},
        )
        keys = lookup.filter(pl.col("name").is_in(self._hit_structure_ids()["hit"]))["key"]
        self.logger.info(
            f"[Foldseek createsubdb] {lookup.height} chains in the database, {keys.len()} selected"
        )
        if keys.is_empty():
            raise RuntimeError("No Foldseek chain matches a selected structure")
        keys_path = self.FOLDSEEK_ALL_PATH / "selected_keys.tsv"
        keys_path.write_text("".join(f"{key}\n" for key in keys))
        out = subprocess.run(
            [str(self.FOLDSEEK_BIN), "createsubdb", str(keys_path), str(full_db), str(db_path)],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            self.logger.error(f"[Foldseek createsubdb] Error: {out.stderr}")
            raise RuntimeError(f"Foldseek createsubdb failed: {out.stderr}")

        start_time = time.time()
        if self.GPU:
            self.logger.info("[Foldseek makepaddedseqdb] Creating padded database for GPU search...")
            cmd = [str(self.FOLDSEEK_BIN), "makepaddedseqdb", str(db_path), f"{db_path}_pad"]
            step = "makepaddedseqdb"
        else:
            self.logger.info("[Foldseek createindex] Creating index for faster searching...")
            shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)
            self.FOLDSEEK_TMPDIR.mkdir(parents=True, exist_ok=True)
            cmd = [str(self.FOLDSEEK_BIN), "createindex", str(db_path), str(self.FOLDSEEK_TMPDIR)]
            step = "createindex"
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[Foldseek {step}] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[Foldseek {step}] Error: {out.stderr}")
            raise RuntimeError(f"Foldseek {step} failed: {out.stderr}")

        self.logger.info(f"[Foldseek {step}] Completed in {elapsed:.1f}s")
        shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)
        return keys.len()

    def _foldseek_search_args(self) -> list[str]:
        return ["--format-output", "query,target,lddt", *FOLDSEEK_SEARCH_ARGS,
                *(["--gpu", "1"] if self.GPU else [])]

    def _chunk_manifest(self, chunk_size: int, search_args: list[str]) -> dict:
        """Identity of the database and search that the finished chunk files belong to."""
        sha = lambda name: hashlib.sha256((self.FOLDSEEK_ALL_PATH / name).read_bytes()).hexdigest()
        return {
            "similarity_fingerprint": self._similarity_fingerprint(),
            "chunk_size": int(chunk_size),
            "search_args": search_args,
            "selected_keys_sha256": sha("selected_keys.tsv"),
            "selected_lookup_sha256": sha("selectedDB.lookup"),
        }

    def _reusable_database_size(self, chunk_size, search_args: list[str]) -> int | None:
        """Number of selected chains if the database and chunks on disk match this run, else None."""
        manifest_path = self.FOLDSEEK_ALL_PATH / "chunks" / "manifest.json"
        if not (chunk_size and manifest_path.is_file()):
            return None
        try:
            matches = json.loads(manifest_path.read_text()) == self._chunk_manifest(chunk_size, search_args)
        except FileNotFoundError:
            matches = False
        if not matches:
            self.logger.info("[Foldseek] Chunk manifest differs from this run; rebuilding the database")
            return None
        n_keys = len((self.FOLDSEEK_ALL_PATH / "selected_keys.tsv").read_text().split())
        n_done = len(list((manifest_path.parent).glob("chunk_*.tsv")))
        self.logger.info(
            f"[Foldseek] Reusing the database and {n_done} finished query chunks ({n_keys} chains)"
        )
        return n_keys

    def _foldseek_search_in_chunks(
        self, chunk_size: int, target_db: Path, out_file: Path, search_args: list[str]
    ) -> float:
        """Search the selected chains as queries, chunk_size at a time, against the whole target.

        Each chunk is a subset of foldseekDB (createsubdb keeps the keys and copies the _ss, _ca and _h
        companions), so the hits are the same as one search; only the prefilter temp files shrink.
        A finished chunk is chunks/chunk_NNN.tsv, renamed from a temp name once complete, and a rerun
        skips it. The chunk files are concatenated in order into out_file.
        """
        keys = (self.FOLDSEEK_ALL_PATH / "selected_keys.tsv").read_text().split()
        chunks = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]
        chunks_dir = self.FOLDSEEK_ALL_PATH / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunk_db = self.FOLDSEEK_ALL_PATH / "chunkDB"
        chunk_keys = self.FOLDSEEK_ALL_PATH / "chunk_keys.tsv"
        elapsed = 0.0
        for i, chunk in enumerate(chunks, 1):
            done = chunks_dir / f"chunk_{i:03d}.tsv"
            if done.exists():
                self.logger.info(f"[Foldseek easy-search] Query chunk {i}/{len(chunks)} is finished; skipping")
                continue
            self.logger.info(f"[Foldseek easy-search] Query chunk {i}/{len(chunks)} ({len(chunk)} chains)")
            chunk_out = done.with_suffix(".tsv.part")
            chunk_out.unlink(missing_ok=True)
            chunk_keys.write_text("".join(f"{key}\n" for key in chunk))
            sub = subprocess.run(
                [str(self.FOLDSEEK_BIN), "createsubdb", str(chunk_keys),
                 str(self.FOLDSEEK_ALL_PATH / "foldseekDB"), str(chunk_db)],
                capture_output=True, text=True,
            )
            if sub.returncode != 0:
                raise RuntimeError(f"Foldseek createsubdb failed: {sub.stderr}")
            shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)
            cmd = [
                str(self.FOLDSEEK_BIN), "easy-search", str(chunk_db), str(target_db),
                str(chunk_out), str(self.FOLDSEEK_TMPDIR), *search_args,
            ]
            elapsed += self._run_search_with_progress(
                cmd, tool="Foldseek", tmpdir=self.FOLDSEEK_TMPDIR, out_file=chunk_out
            )
            shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)
            for f in self.FOLDSEEK_ALL_PATH.glob("chunkDB*"):
                f.unlink()
            chunk_out.touch()
            chunk_out.rename(done)
        with open(out_file, "wb") as dst:
            for i in range(1, len(chunks) + 1):
                with open(chunks_dir / f"chunk_{i:03d}.tsv", "rb") as src:
                    shutil.copyfileobj(src, dst)
        return elapsed

    def _foldseek_all_vs_all(self) -> pl.LazyFrame:
        """
        Run all-vs-all structure comparison using Foldseek easy-search against the
        database built from self.ALL_PDB_SYMLINK_PATH.  Results are saved to
        self.OUTPUT_DIR / 'foldseek_all_vs_all.tsv' for reproducibility.

        lddt values are read as strings then coerced to float so that
        scientific-notation entries (e.g. '2.190E-01') are handled correctly.
        Self-hits are removed before returning.

        Returns:
            LazyFrame with columns: query (str), target (str), lddt (float64).
        """
        self.logger.info("=" * 60)
        self.logger.info("[Foldseek] Starting all-vs-all structure comparison")
        self.logger.info("=" * 60)

        if (
            not self.CACHE_OVERWRITE
            and self.ALL_VS_ALL_CACHE_PATH.exists()
            and self._cache_matches(
                self.ALL_VS_ALL_CACHE_PATH, self._similarity_fingerprint()
            )
        ):
            self.logger.info(
                f"[Foldseek] Reusing cached all-vs-all TSV: {self.ALL_VS_ALL_CACHE_PATH}"
            )
            return self._read_foldseek_all_vs_all_tsv(self.ALL_VS_ALL_CACHE_PATH)

        search_args = self._foldseek_search_args()
        chunk_size = self.SEARCH_QUERY_CHUNK_SIZE
        n_structures = self._reusable_database_size(chunk_size, search_args)
        if n_structures is None:
            n_structures = self._foldseek_create_database()
        db_path = self.FOLDSEEK_ALL_PATH / "selectedDB"
        if self.GPU:
            db_path = Path(f"{db_path}_pad")
        out_file = self.FOLDSEEK_ALL_PATH / "all_vs_all.tsv"

        # Count chains for estimation
        n_comparisons_estimate = n_structures * (n_structures - 1)

        self.logger.info(f"[Foldseek easy-search] Starting all-vs-all comparison...")
        self.logger.info(f"[Foldseek easy-search] Structures: {n_structures}")
        self.logger.info(
            f"[Foldseek easy-search] Max possible comparisons: {n_comparisons_estimate:,}"
        )
        self.logger.info(f"[Foldseek easy-search] Parameters:")
        self.logger.info(f"    --cov-mode 2 (coverage of the shorter chain)")
        self.logger.info(f"    -c 0.8 (min 80% alignment coverage of the shorter chain)")
        self.logger.info(f"    -s 9.5 (high sensitivity)")
        self.logger.info(f"    --alignment-mode 3 (global alignment)")
        self.logger.info(f"    --alignment-type 2 (3Di+AA alignment)")
        self.logger.info(
            f"[Foldseek easy-search] This may take a while for large datasets..."
        )

        selected_db = self.FOLDSEEK_ALL_PATH / "selectedDB"
        try:
            if chunk_size and n_structures > chunk_size:
                manifest = self._chunk_manifest(chunk_size, search_args)
                (self.FOLDSEEK_ALL_PATH / "chunks").mkdir(parents=True, exist_ok=True)
                (self.FOLDSEEK_ALL_PATH / "chunks" / "manifest.json").write_text(json.dumps(manifest))
                elapsed = self._foldseek_search_in_chunks(
                    int(chunk_size), db_path, out_file, search_args
                )
            else:
                cmd = [
                    str(self.FOLDSEEK_BIN), "easy-search", str(selected_db), str(db_path),
                    str(out_file), str(self.FOLDSEEK_TMPDIR), *search_args,
                ]
                elapsed = self._run_search_with_progress(
                    cmd, tool="Foldseek", tmpdir=self.FOLDSEEK_TMPDIR, out_file=out_file
                )
        finally:
            # Cleanup temp directories
            shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)

        self.logger.info(
            f"[Foldseek easy-search] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)"
        )

        if not (out_file.exists() and out_file.stat().st_size):
            self.logger.warning("[Foldseek] No output file - no alignments found")
            return pl.LazyFrame(
                schema={"query": pl.Utf8, "target": pl.Utf8, "lddt": pl.Float64}
            )

        self.ALL_VS_ALL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            f"[Foldseek] Normalizing raw results to {self.ALL_VS_ALL_CACHE_PATH}..."
        )
        self._read_foldseek_all_vs_all_tsv(out_file, raw_format=True).sink_csv(
            self.ALL_VS_ALL_CACHE_PATH, separator="\t", engine="streaming"
        )
        shutil.rmtree(self.ALL_PDB_SYMLINK_PATH, ignore_errors=True)
        shutil.rmtree(self.FOLDSEEK_ALL_PATH, ignore_errors=True)
        self._write_cache_fingerprint(
            self.ALL_VS_ALL_CACHE_PATH, self._similarity_fingerprint()
        )

        results_df = self._read_foldseek_all_vs_all_tsv(self.ALL_VS_ALL_CACHE_PATH)
        summary = results_df.select(
            pl.len().alias("n"),
            pl.col("lddt").min().alias("min"),
            pl.col("lddt").max().alias("max"),
            pl.col("lddt").mean().alias("mean"),
        ).collect(engine="streaming").row(0, named=True)
        self.logger.info("[Foldseek] Results summary:")
        self.logger.info(f"    Total pairwise comparisons: {summary['n']:,}")
        if summary["n"]:
            self.logger.info(
                f"    LDDT range: [{summary['min']:.3f}, {summary['max']:.3f}]"
            )
            self.logger.info(f"    LDDT mean: {summary['mean']:.3f}")
        self.logger.info("=" * 60)
        return results_df

    def _read_foldseek_all_vs_all_tsv(
        self, tsv_path: Path, raw_format: bool = False
    ) -> pl.LazyFrame:
        """Lazily scan a Foldseek hit TSV (raw, or the normalized cache) into query/target/lddt."""
        columns = {"query": pl.Utf8, "target": pl.Utf8, "lddt": pl.Utf8}
        if raw_format:
            hits = pl.scan_csv(tsv_path, separator="\t", has_header=False, schema=columns)
        else:
            hits = pl.scan_csv(tsv_path, separator="\t", schema_overrides=columns)
            if not set(columns).issubset(hits.collect_schema().names()):
                self.logger.warning(
                    f"Cached file {tsv_path} is missing required columns {sorted(columns)}. "
                    "Attempting to read it as a raw Foldseek TSV."
                )
                return self._read_foldseek_all_vs_all_tsv(tsv_path, raw_format=True)

        hits = hits.with_columns(
            pl.col("lddt").cast(pl.Float64, strict=False),
            pl.col("query").str.strip_chars(),
            pl.col("target").str.strip_chars(),
        )
        if raw_format:
            hit_map = self._hit_structure_ids().lazy()
            # Hits on files or chains that are not a dataset structure have no entry and drop out.
            for side in ("query", "target"):
                hits = (
                    hits.join(
                        hit_map.rename({"hit": side, "structure_id": f"{side}_id"}),
                        on=side,
                    )
                    .drop(side)
                    .rename({f"{side}_id": side})
                )
        return (
            hits.select("query", "target", "lddt")
            .drop_nulls()
            .filter(pl.col("query") != pl.col("target"))
        )

    def _hit_structure_ids(self) -> pl.DataFrame:
        """Foldseek hit name -> structure id, for the structures that have a file.

        An experimental structure is its selected chain, named `<PDB>_<chain>`. An AlphaFold or
        ESM structure is named by its accession, with the optional `.pdb` extension and the
        `_A` chain; spelling it out keeps accessions that contain underscores intact.
        """
        available = pl.Series("structure_id", list(self._pdb_paths), dtype=pl.Utf8)
        experimental = (
            self.search_df.filter(pl.col("pdb_type") == "experimental")
            .select(
                "structure_id",
                pl.format(
                    "{}_{}", pl.col("pdbs").str.split(":").list.first(), "pdb_chain"
                ).alias("hit"),
            )
            .unique()
            .filter(pl.col("structure_id").is_in(available))
        )
        predicted = pl.DataFrame({"structure_id": available}).filter(
            ~pl.col("structure_id").is_in(experimental.get_column("structure_id"))
        )
        return pl.concat(
            [experimental]
            + [
                predicted.select("structure_id", (pl.col("structure_id") + suffix).alias("hit"))
                for suffix in ("", ".pdb", "_A", ".pdb_A")
            ]
        )

    def _drop_same_sequence_hits(self, hits: pl.LazyFrame, index: pl.LazyFrame) -> pl.LazyFrame:
        """Drop hits whose query and target structures share any dataset sequence.

        `hits` holds structure indices `qi`, `ti` (see `index`: structure_id -> i).
        """
        by_sequence = (
            self.full_dataset_df.lazy()
            .select("structure_id", "sequence")
            .drop_nulls()
            .unique()
            .join(index, left_on="structure_id", right_on="id")
        )
        same_sequence = (
            by_sequence.join(by_sequence, on="sequence")
            .select(pl.col("i").alias("qi"), pl.col("i_right").alias("ti"))
            .unique()
        )
        return hits.join(same_sequence, on=["qi", "ti"], how="anti")

    def compute_max_lddt_to_dataset(self) -> np.ndarray:
        """
        Compute max LDDT similarity of each protein to ALL other proteins in dataset.
        This is computed once and cached.

        NOTE on LDDT direction:
        - Higher LDDT (closer to 1.0) = MORE similar structures
        - Lower LDDT (closer to 0.0) = MORE dissimilar structures

        For a challenging test set, we want proteins with LOW max LDDT
        (structurally dissimilar from the rest of the dataset).

        Memory optimization: Only stores max LDDT per protein (O(n)), not all
        pairwise comparisons (O(n²)).

        Returns:
            Array of max LDDT similarities for each protein (same order as self.unique_structures)
        """
        if self._max_lddt_array is not None:
            self.logger.info("Using cached max LDDT similarities")
            return self._max_lddt_array

        if not self.CACHE_OVERWRITE:
            loaded = self._load_similarity_cache(
                cache_path=self.MAX_SIM_CACHE_PATH,
                key_column="structure_id",
                value_column="max_similarity",
                ordered_keys=self.unique_structures,
                fingerprint=self._max_similarity_fingerprint(),
            )
            if loaded is not None:
                values, sim_map = loaded
                self._max_lddt_array = values
                self._max_lddt_to_dataset = sim_map
                self.logger.info(
                    f"Using cached max LDDT similarities from {self.MAX_SIM_CACHE_PATH}"
                )
                return self._max_lddt_array

        n = len(self.unique_structures)
        self.logger.info(
            f"Computing max LDDT similarity for {n} proteins to all others..."
        )

        # Generate PDB paths for all proteins
        self._pdb_paths = self._generate_pdb_paths(self.search_df)

        # Only include proteins with available PDBs
        available_ids = list(self._pdb_paths.keys())
        self.logger.info(
            f"PDB structures available for {len(available_ids)}/{n} proteins"
        )

        # Hits are reduced to integer indices of the dataset's own structures before any
        # aggregation: string keys keep the whole scanned file alive and memory grows with it.
        index = pl.LazyFrame(
            {"id": self.unique_structures, "i": pl.Series(range(n), dtype=pl.UInt32)},
            schema_overrides={"id": pl.Utf8},
        )
        hits = (
            self._foldseek_all_vs_all()
            .join(index.rename({"id": "query", "i": "qi"}), on="query")
            .join(index.rename({"id": "target", "i": "ti"}), on="target")
            .select("qi", "ti", "lddt")
        )
        hits = self._drop_same_sequence_hits(hits, index)

        # Max LDDT per protein, symmetric in query and target. Batches keep memory bounded.
        self.logger.info("Computing max LDDT per protein...")
        max_by_index = pl.DataFrame(schema={"i": pl.UInt32, "lddt": pl.Float64})
        for batch in hits.collect_batches(chunk_size=5_000_000):
            max_by_index = (
                pl.concat(
                    [
                        max_by_index,
                        batch.select(pl.col("qi").alias("i"), "lddt"),
                        batch.select(pl.col("ti").alias("i"), "lddt"),
                    ]
                )
                .group_by("i")
                .agg(pl.col("lddt").max())
            )
        max_lddt_to_dataset = {
            self.unique_structures[i]: lddt
            for i, lddt in zip(max_by_index["i"].to_list(), max_by_index["lddt"].to_list())
        }

        # Build array in same order as unique_structures
        max_similarities = np.zeros(n)
        for i, uid in progress(
            enumerate(self.unique_structures),
            total=n,
            desc="Building max LDDT array",
            leave=False,
        ):
            # A missing structure is unknown, not dissimilar. Store NaN so the split
            # (also from a warm cache) can force it to train. No-hit structures stay 0.0.
            max_similarities[i] = (
                max_lddt_to_dataset.get(uid, 0.0) if uid in self._pdb_paths else np.nan
            )

        self._max_lddt_array = max_similarities
        # Keep the in-memory map consistent with the persisted cache: proteins with
        # no Foldseek hit are treated as 0.0 similarity, not left missing/NaN.
        self._max_lddt_to_dataset = dict(zip(self.unique_structures, max_similarities))

        self._save_similarity_cache(
            cache_path=self.MAX_SIM_CACHE_PATH,
            key_column="structure_id",
            value_column="max_similarity",
            ordered_keys=self.unique_structures,
            values=max_similarities,
            fingerprint=self._max_similarity_fingerprint(),
        )

        # Log statistics
        self.logger.info("Max LDDT similarity statistics:")
        self.logger.info(f"  Min: {np.nanmin(max_similarities):.3f}")
        self.logger.info(f"  Max: {np.nanmax(max_similarities):.3f}")
        self.logger.info(f"  Mean: {np.nanmean(max_similarities):.3f}")
        self.logger.info(f"  Median: {np.nanmedian(max_similarities):.3f}")
        self.logger.info(f"  Proteins with max_lddt=0: {np.sum(max_similarities == 0)}")
        self.logger.info(f"  Proteins without structure (NaN): {np.isnan(max_similarities).sum()}")

        return max_similarities

    def get_max_lddt_for_proteins(self, protein_ids: list) -> dict:
        """
        Get max LDDT to dataset for given proteins from cache.

        This returns the max LDDT of each protein to ALL other proteins in the
        dataset (not just train). This is memory-efficient as we only store
        O(n) max values, not O(n²) pairwise comparisons.

        Args:
            protein_ids: List of protein IDs to get max LDDT for

        Returns:
            Dictionary mapping protein_id -> max_lddt_to_dataset
        """
        if self._max_lddt_to_dataset is None:
            msg = "Max LDDT to dataset not computed yet. Call compute_max_lddt_to_dataset() first."
            self.logger.error(msg)
            raise RuntimeError(msg)

        similarities = {}
        for uid in protein_ids:
            similarities[uid] = self._max_lddt_to_dataset.get(uid, 0.0)

        # Log statistics
        sim_values = list(similarities.values())
        if sim_values:
            self.logger.info(
                f"Max LDDT to dataset statistics for {len(protein_ids)} proteins:"
            )
            self.logger.info(f"  Min: {np.min(sim_values):.3f}")
            self.logger.info(f"  Max: {np.max(sim_values):.3f}")
            self.logger.info(f"  Mean: {np.mean(sim_values):.3f}")
            self.logger.info(f"  Median: {np.median(sim_values):.3f}")

        return similarities

    def _get_grouped_dataset_df(self) -> pl.DataFrame:
        if self._grouped_dataset_df is not None:
            return self._grouped_dataset_df

        if self._max_lddt_to_dataset is None:
            self.compute_max_lddt_to_dataset()

        # NaN LDDT marks a missing structure. Keep that as has_any_structure=False and
        # use lddt 0.0 so max() is unchanged; rows without a pdb id have no structure either.
        # A low-confidence predicted structure is unknown: it is not an eligible structure
        # and its lDDT does not count, but its hits still count for other proteins.
        similarity_df = pl.DataFrame(
            {
                "structure_id": list(self._max_lddt_to_dataset.keys()),
                "lddt": list(self._max_lddt_to_dataset.values()),
            },
            schema={"structure_id": pl.Utf8, "lddt": pl.Float64},
        ).with_columns(has_any_structure=pl.col("lddt").is_not_nan())
        grouped_input_df = self.full_dataset_df.join(
            similarity_df, on="structure_id", how="left"
        ).with_columns(
            pl.col("lddt").fill_nan(0.0).fill_null(0.0),
            pl.col("has_any_structure").fill_null(False),
        ).with_columns(
            has_structure=pl.col("has_any_structure")
            & ~pl.col("structure_id").is_in(list(self._low_confidence_structures()))
        ).with_columns(lddt=pl.when("has_structure").then("lddt").otherwise(0.0))
        pdb_cols = {"pdbs", "pdb_source", "pdb_type", "pdb_chain", "pdb_coverage", "structure_id"}
        # __index_level_0__ is a pandas-generated artifact index; including it as a
        # group key would make every row its own group, defeating deduplication.
        index_cols = {"__index_level_0__"}
        group_cols = [
            col
            for col in grouped_input_df.columns
            if col not in pdb_cols | index_cols | {"lddt", "has_structure", "has_any_structure"}
        ]

        if self.DROP_TO_1D:
            # A companion 1D source is materialized after sequence assignment.
            # Keep this 3D grouping solely for Foldseek/LDDT assignment and legacy
            # configurations that do not provide a recoverable 1D source.
            grouped_df = grouped_input_df.group_by(group_cols, maintain_order=True).agg(
                pl.col("lddt").max().alias("lddt"),
                pl.col("has_structure").any(),
                pl.col("has_any_structure").any(),
            )
        else:
            record_cols = [
                col for col in ("pdbs", "pdb_source", "pdb_type", "pdb_chain", "pdb_coverage")
                if col in grouped_input_df.columns
            ]
            grouped_df = (
                grouped_input_df.with_columns(pl.struct(record_cols).alias("pdb_record"))
                .group_by(group_cols, maintain_order=True)
                .agg(
                    pl.col("lddt").max().alias("lddt"),
                    pl.col("has_structure").any(),
                    pl.col("has_any_structure").any(),
                    pl.col("pdbs")
                    .drop_nulls()
                    .unique(maintain_order=True)
                    .alias("pdbs"),
                    pl.col("pdb_source")
                    .drop_nulls()
                    .unique(maintain_order=True)
                    .alias("pdb_source"),
                    pl.col("pdb_type")
                    .drop_nulls()
                    .unique(maintain_order=True)
                    .alias("pdb_type"),
                    pl.col("pdbs").drop_nulls().n_unique().alias("pdb_count"),
                    pl.col("pdb_record")
                    .drop_nulls()
                    .unique(maintain_order=True)
                    .alias("pdb_records"),
                )
            )

        self._grouped_dataset_df = grouped_df
        return self._grouped_dataset_df

    def _explode_grouped_split_df(self, split_df):
        split_df = split_df.drop("lddt", "has_structure", "has_any_structure").unique(maintain_order=True)
        if self.DROP_TO_1D or split_df.is_empty():
            return split_df.to_pandas()

        metadata_cols = [
            col
            for col in ["pdbs", "pdb_source", "pdb_type", "pdb_chain", "pdb_coverage"]
            if col in split_df.columns
        ]
        exploded_df = (
            split_df.drop(metadata_cols)
            .explode("pdb_records")
            .unnest("pdb_records")
        )
        return exploded_df.to_pandas()

    def _materialize_companion_1d_splits(
        self,
        *,
        train_sequences: set,
        val_sequences: set,
        test_sequences: set,
    ) -> dict[str, pl.DataFrame]:
        """Split the original 1D rows without deduplicating or aggregating them."""
        if self.DROP_TO_1D_DATASET_PATH is None:
            return {}

        source_df = pl.read_parquet(self.DROP_TO_1D_DATASET_PATH).with_row_index(
            "_source_row_index"
        )
        if "sequence" not in source_df.columns:
            raise ValueError(
                "Companion 1D dataset must contain a 'sequence' column: "
                f"{self.DROP_TO_1D_DATASET_PATH}"
            )

        assigned_sequences = train_sequences | val_sequences | test_sequences
        missing_sequences = (
            source_df.filter(
                pl.col("sequence").is_not_null()
                & ~pl.col("sequence").is_in(list(assigned_sequences))
            )
            .get_column("sequence")
            .unique(maintain_order=True)
            .to_list()
        )
        if missing_sequences:
            sample = ", ".join(map(str, missing_sequences[:5]))
            missing_rows = source_df.filter(
                pl.col("sequence").is_in(missing_sequences)
            ).height
            raise ValueError(
                "Companion 1D dataset contains non-null sequences absent from the "
                f"3D assignment dataset: {len(missing_sequences)} sequences, "
                f"{missing_rows} rows. Examples: [{sample}]"
            )

        split_masks = {
            "train": pl.col("sequence").is_null()
            | pl.col("sequence").is_in(list(train_sequences)),
            "val": pl.col("sequence").is_in(list(val_sequences)),
            "test": pl.col("sequence").is_in(list(test_sequences)),
        }
        indexed_splits = {
            name: source_df.filter(mask) for name, mask in split_masks.items()
        }
        combined = pl.concat(list(indexed_splits.values()))
        if combined.height != source_df.height or combined.get_column(
            "_source_row_index"
        ).n_unique() != source_df.height:
            raise RuntimeError(
                "Companion 1D split invariant failed: source rows were lost or "
                "assigned more than once."
            )

        source_schema = source_df.drop("_source_row_index").schema
        splits = {
            name: frame.drop("_source_row_index") for name, frame in indexed_splits.items()
        }
        if any(frame.schema != source_schema for frame in splits.values()):
            raise RuntimeError("Companion 1D split invariant failed: schema changed.")
        return splits

    def dissimilarity_based_split(self, similarity_threshold) -> tuple[dict, dict]:
        """
        Split dataset based on protein structure dissimilarity using max LDDT.

        NOTE on LDDT direction:
        - Higher LDDT = MORE similar (1.0 = identical)
        - Lower LDDT = MORE dissimilar (0.0 = completely different)

        Algorithm:
        1. Compute max LDDT of each protein to ALL other proteins
        2. Proteins with max LDDT < threshold are "dissimilar" (structurally dissimilar)
        3. Assign dissimilar proteins to test/val first (most dissimilar = lowest max LDDT)

        This creates a challenging test set where test proteins are structurally
        distinct from the training set.

        Args:
            similarity_threshold: Proteins with max LDDT below this are dissimilar

        Returns:
            tuple: (splits_dict, similarities_dict)
                splits_dict: {"train": df, "val": df, "test": df}
                similarities_dict: {"test": {id: max_lddt}, "val": {id: max_lddt}}
        """

        self.logger.info(
            f"Splitting with similarity_threshold={similarity_threshold}, "
            f"max_test_frac={self.TEST_FRAC}"
        )

        self.compute_max_lddt_to_dataset()
        grouped_df = self._get_grouped_dataset_df()

        if self.DROP_TO_1D:
            sequence_similarity_df = (
                grouped_df.group_by("sequence", maintain_order=True)
                .agg(
                    pl.col("lddt").max().alias("lddt"),
                    pl.col("has_structure").any(),
                    pl.col("has_any_structure").any(),
                    pl.len().alias("row_count"),
                )
                .sort(["lddt", "row_count", "sequence"])
            )
        else:
            sequence_similarity_df = (
                grouped_df.group_by("sequence", maintain_order=True)
                .agg(
                    pl.col("lddt").max().alias("lddt"),
                    pl.col("has_structure").any(),
                    pl.col("has_any_structure").any(),
                    pl.col("pdb_records").list.len().sum().alias("row_count"),
                )
                .sort(["lddt", "row_count", "sequence"])
            )

        sequence_similarity_df = sequence_similarity_df.filter(
            pl.col("sequence").is_not_null()
        )
        sequences = sequence_similarity_df.get_column("sequence").to_list()
        sequence_lddt = sequence_similarity_df.get_column("lddt").to_numpy()
        sequence_row_count = sequence_similarity_df.get_column("row_count").to_numpy()
        # A missing structure, a structure with no Foldseek hit (lddt 0.0) and a
        # low-confidence predicted structure are all unknown, not dissimilar. They always go
        # to train, never val/test; counts logged apart.
        has_structure = sequence_similarity_df.get_column("has_structure").to_numpy()
        has_any_structure = sequence_similarity_df.get_column("has_any_structure").to_numpy()
        no_hit = has_structure & (sequence_lddt == 0.0)
        low_confidence = has_any_structure & ~has_structure
        unknown = ~has_structure | no_hit
        for label, mask in (
            ("without a structure (forced to train)", ~has_any_structure),
            (
                "with only low-confidence predicted structures "
                f"(mean pLDDT < {self.MIN_MEAN_PLDDT:g}, forced to train)",
                low_confidence,
            ),
            ("with no Foldseek hit (forced to train)", no_hit),
        ):
            self.logger.info(
                f"Sequences {label}: {int(mask.sum()):,} "
                f"({int(sequence_row_count[mask].sum()):,} rows)"
            )
        sequence_to_lddt = dict(zip(sequences, sequence_lddt))

        n_unique = len(sequences)
        n_full = (
            len(grouped_df)
            if self.DROP_TO_1D
            else int(grouped_df["pdb_records"].list.len().sum())
        )
        max_test_rows = int(self.TEST_FRAC * n_full)

        self.logger.info(f"Max test rows: {max_test_rows}")
        self.logger.info(f"Grouped dataset rows: {n_full}")

        eligible = np.isfinite(sequence_lddt)

        # Find dissimilar sequences (max LDDT below threshold)
        dissimilar_mask = sequence_lddt < similarity_threshold
        dissimilar_indices = np.where(eligible & ~unknown & dissimilar_mask)[0]

        self.logger.info(
            f"Found {len(dissimilar_indices)} dissimilar sequences "
            f"(max LDDT < {similarity_threshold})"
        )

        def row_count_of(idx: int) -> int:
            return int(sequence_row_count[idx])

        test_indices, test_row_count = greedy_fill(
            dissimilar_indices, row_count_of, max_test_rows, desc="Filling test set"
        )

        self.logger.info(
            f"Test set: {len(test_indices)} sequences, {test_row_count} rows"
        )

        # Phase 2: Fill val set to match test size
        used_indices = set(test_indices)
        remaining_indices = [
            idx for idx in range(n_unique) if idx not in used_indices and not unknown[idx]
        ]

        # val is sized from valid_frac, not from the test row count.
        target_val_rows = int(self.VAL_FRAC * n_full)
        val_indices, val_row_count = greedy_fill(
            remaining_indices, row_count_of, target_val_rows, desc="Filling val set"
        )

        self.logger.info(f"Val set: {len(val_indices)} sequences, {val_row_count} rows")

        # Phase 3: Train = everything else
        val_set = set(val_indices)
        train_indices = [
            idx for idx in range(n_unique) if idx not in used_indices and idx not in val_set
        ]
        train_row_count = n_full - test_row_count - val_row_count

        self.logger.info(
            f"Train set: {len(train_indices)} sequences, {train_row_count} rows"
        )

        test_sequences = {sequences[i] for i in test_indices}
        val_sequences = {sequences[i] for i in val_indices}
        train_sequences = {sequences[i] for i in train_indices}

        # Create split DataFrames
        if self.DROP_TO_1D and self.DROP_TO_1D_DATASET_PATH is not None:
            splits = self._materialize_companion_1d_splits(
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                test_sequences=test_sequences,
            )
        else:
            grouped_splits = {
                "train": grouped_df.filter(
                    pl.col("sequence").is_null()
                    | pl.col("sequence").is_in(list(train_sequences))
                ),
                "val": grouped_df.filter(pl.col("sequence").is_in(list(val_sequences))),
                "test": grouped_df.filter(pl.col("sequence").is_in(list(test_sequences))),
            }
            splits = {
                split_name: self._explode_grouped_split_df(split_df)
                for split_name, split_df in grouped_splits.items()
            }

        # Verify no overlap
        assert len(test_sequences & val_sequences) == 0, "Test and val overlap!"
        assert len(test_sequences & train_sequences) == 0, "Test and train overlap!"
        assert len(val_sequences & train_sequences) == 0, "Val and train overlap!"

        self.logger.info("Getting max LDDT to dataset for test sequences...")
        test_similarities = {
            seq: sequence_to_lddt[seq] for seq in sorted(test_sequences)
        }

        self.logger.info("Getting max LDDT to dataset for val sequences...")
        val_similarities = {seq: sequence_to_lddt[seq] for seq in sorted(val_sequences)}

        similarities = {
            "test": test_similarities,
            "val": val_similarities,
        }

        # Log summary (use actual post-explode total since pdb_records can have more
        # entries than pdb_count when a PDB ID appears under multiple source/type combos)
        total_rows = sum(len(df) for df in splits.values())
        for split_name, split_df in splits.items():
            pct = 100 * len(split_df) / total_rows if total_rows > 0 else 0.0
            self.logger.info(f"{split_name}: {len(split_df)} rows ({pct:.1f}%)")

        return splits, similarities

    def run_splits_across_thresholds(self) -> tuple[dict, dict]:
        """
        Run LDDT-based splits across multiple similarity thresholds.

        The max LDDT to all other proteins is computed ONCE, then splits are
        generated for each threshold (proteins with max_lddt < threshold go to test/val).

        Returns:
            tuple: (all_splits, all_similarities) - both dicts keyed by threshold
        """

        self.logger.info(
            "Computing max LDDT similarities to full dataset (one time)..."
        )
        self.compute_max_lddt_to_dataset()

        all_splits = {}
        all_similarities = {}

        for threshold in progress(
            self.THRESHOLDS,
            desc="Processing thresholds",
            unit="threshold",
        ):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing similarity threshold: {threshold}")
            self.logger.info(f"{'='*50}")

            splits, similarities = self.dissimilarity_based_split(
                similarity_threshold=threshold,
            )

            all_splits[threshold] = splits
            all_similarities[threshold] = similarities

        return all_splits, all_similarities

    def plot_lddt_distribution(self, all_similarities: dict) -> None:
        values_by_threshold = self._build_value_map_from_similarity_dicts(
            all_similarities
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 21),
            output_path=self.UNIQUE_STRUCTURE_SIMILARITY_PLOT_PATH,
            xlabel="Max LDDT to Dataset",
            ylabel="Count",
            title_fn=lambda threshold, val_values, test_values: (
                f"Threshold: {threshold}\n"
                f"Val: {len(val_values)}, Test: {len(test_values)}"
            ),
            stats_mode="summary",
            legend_loc="upper right",
            hist_range=(0, 1),
        )

    def plot_full_dataset_lddt_distribution(
        self, all_splits: dict, all_similarities: dict
    ) -> None:
        grouped_df = self._get_grouped_dataset_df()
        full_dataset_size = (
            len(grouped_df)
            if self.DROP_TO_1D
            else int(grouped_df["pdb_records"].list.len().sum())
        )
        values_by_threshold = self._build_value_map_from_split_frames(
            all_splits=all_splits,
            all_similarities=all_similarities,
            key_column="sequence",
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 21),
            output_path=self.FULL_DATASET_STRUCTURE_SIMILARITY_PLOT_PATH,
            xlabel="Max LDDT to Dataset",
            ylabel="Number of Rows",
            title_fn=lambda threshold, val_values, test_values: (
                f"Full Dataset LDDT Similarity\nThreshold {threshold}\n"
                f"Val: {len(val_values)} ({100 * len(val_values) / full_dataset_size:.1f}%), "
                f"Test: {len(test_values)} ({100 * len(test_values) / full_dataset_size:.1f}%)"
            ),
            stats_mode="detailed",
            legend_loc="upper left",
        )

    def plot_unique_distribution(self, all_similarities: dict) -> None:
        self.plot_lddt_distribution(all_similarities)

    def plot_full_distribution(self, all_splits: dict, all_similarities: dict) -> None:
        self.plot_full_dataset_lddt_distribution(all_splits, all_similarities)

    def get_output_dir(self) -> Path:
        return self.OUTPUT_DIR


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")

    splitter = ProteinStructMaxLDDTSimilaritySplitter(cfg=cfg)
    all_splits, all_similarities = splitter.generate_splits()
