import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import time
import shutil
import subprocess
import numpy as np
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from src.utils.tqdmlogger import TqdmLogger
from src.data.components.splitters.base import BaseThresholdedSimilaritySplitter


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

    This class uses Foldseek for fast structure comparison, which requires
    PDB files for each protein. It supports experimental PDB chain structures,
    AlphaFold-predicted structures, and ESM-predicted structures.
    """

    def __init__(self, cfg: DictConfig):
        struct_cfg = cfg.splits.dataset.structure_split

        self.TEST_FRAC = cfg.splits.test_frac or 0.1
        self.CACHE_OVERWRITE = cfg.splits.get("similarity_cache_overwrite", False)
        self.DROP_TO_1D = bool(struct_cfg.get("drop_to_1d", False))
        self.THRESHOLDS = list(
            struct_cfg.get("similarity_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )

        # PDB source paths
        self.AF_PDB_PATH = Path(cfg.splits.af_pdb_path)
        self.ESM_PDB_PATH = Path(cfg.splits.esm_pdb_path)
        self.PROCESSED_EXP_PDB_PATH = Path(cfg.splits.processed_exp_pdb_path)

        # Foldseek working directories
        self.FOLDSEEK_ALL_PATH = Path(cfg.splits.all_foldseek_path)
        self.FOLDSEEK_TMPDIR = Path(cfg.splits.foldseek_tmpdir)

        # PDB symlink directories
        self.ALL_PDB_SYMLINK_PATH = Path(cfg.splits.all_pdb_symlink_path)

        self.INPUT_DATASET_PATH = Path(struct_cfg.input_dataset_parquet_file_path)
        self.OUTPUT_DIR = Path(struct_cfg.output_dir)
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

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.splits.log_file_name,
        ).get_logger()

        # Check for foldseek installation
        if shutil.which("foldseek") is None:
            msg = "Foldseek installation not found. Visit https://github.com/steineggerlab/foldseek to install"
            self.logger.error(msg)
            raise EnvironmentError(msg)

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet file not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Loading full dataset...")
        self.full_dataset_df = pl.read_parquet(self.INPUT_DATASET_PATH)
        self.logger.info(f"Full dataset shape: {self.full_dataset_df.shape}")

        # Get unique PDB IDs
        self.unique_pdbs = (
            self.full_dataset_df.get_column("pdbs")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )
        self.logger.info(f"Unique PDB IDs: {len(self.unique_pdbs)}")
        self.logger.info(f"drop_to_1d={self.DROP_TO_1D}")

        # Will be computed lazily and cached for reuse
        # Only store max LDDT per protein (O(n) memory instead of O(n²) for pairwise)
        self._pdb_paths = None
        self._max_lddt_to_dataset = None  # dict: protein_id -> max_lddt
        self._max_lddt_array = None  # array in same order as self.unique_pdbs
        self._grouped_dataset_df = None
        self.ALL_VS_ALL_CACHE_PATH = self.OUTPUT_DIR / "foldseek_all_vs_all_lddt.tsv"
        self.MAX_SIM_CACHE_PATH = self.OUTPUT_DIR / "max_lddt_similarities.tsv"

    def _create_pdb_symlink(
        self, pdb_id: str, source_path: Path, pdb_paths: dict
    ) -> bool:
        """Create a symlink for a PDB file. Returns True if successful."""
        if not source_path.exists():
            return False
        symlink_path = self.ALL_PDB_SYMLINK_PATH / f"{pdb_id}.pdb"
        symlink_path.symlink_to(source_path)
        pdb_paths[pdb_id] = symlink_path
        return True

    def _generate_pdb_paths(self, df: pl.DataFrame) -> dict:
        """
        Generate symlinks to PDB files for all proteins in the given DataFrame.
        Supports experimental chain PDBs, AlphaFold structures, and ESM-predicted structures.
        Symlinks are written to self.ALL_PDB_SYMLINK_PATH.

        Args:
            df: DataFrame containing at minimum 'pdbs', 'pdb_type', and 'pdb_source' columns.

        Returns:
            Dictionary mapping pdb_id -> symlink Path for every structure located on disk.
        """
        shutil.rmtree(self.ALL_PDB_SYMLINK_PATH, ignore_errors=True)
        self.ALL_PDB_SYMLINK_PATH.mkdir(parents=True, exist_ok=True)

        experimental_pdbs = sorted(
            df.filter(pl.col("pdb_type") == "experimental")
            .get_column("pdbs")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )
        alphafold_pdbs = sorted(
            df.filter(
                (pl.col("pdb_type") == "predicted")
                & (pl.col("pdb_source") == "AlphaFold")
            )
            .get_column("pdbs")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )
        esm_pdbs = sorted(
            df.filter(pl.col("pdb_type").is_null())
            .get_column("pdbs")
            .drop_nulls()
            .unique(maintain_order=True)
            .to_list()
        )

        # Index AlphaFold structures and keep highest version for each acc_id
        self.logger.info("Indexing AlphaFold structures...")
        all_af_pdbs = list(self.AF_PDB_PATH.glob("AF-*-F1-model_v*.pdb"))

        af_pdb_map = {}
        for pdb_file in all_af_pdbs:
            parts = pdb_file.stem.split("-")
            if len(parts) >= 3:
                acc_id = parts[1]
                version = int(parts[-1].split("_v")[-1])
                if acc_id not in af_pdb_map:
                    af_pdb_map[acc_id] = []
                af_pdb_map[acc_id].append((version, pdb_file))

        af_highest = {
            acc_id: max(versions, key=lambda x: x[0])[1]
            for acc_id, versions in af_pdb_map.items()
        }

        pdb_paths = {}
        missing_counts = {"experimental": 0, "alphafold": 0, "esm": 0}

        # Process experimental PDBs
        for pdb in tqdm(
            experimental_pdbs, desc="Preparing experimental PDB paths", leave=False
        ):
            if not self._create_pdb_symlink(
                pdb, self.PROCESSED_EXP_PDB_PATH / f"{pdb}.pdb", pdb_paths
            ):
                missing_counts["experimental"] += 1

        # Process AlphaFold PDBs
        for acc_id in tqdm(
            alphafold_pdbs, desc="Preparing alphafold PDB paths", leave=False
        ):
            if acc_id not in af_highest:
                missing_counts["alphafold"] += 1
                continue
            if not self._create_pdb_symlink(acc_id, af_highest[acc_id], pdb_paths):
                missing_counts["alphafold"] += 1

        # Process ESM PDBs
        for acc_id in tqdm(esm_pdbs, desc="Preparing ESM PDB paths", leave=False):
            if not self._create_pdb_symlink(
                acc_id, self.ESM_PDB_PATH / f"ESM3-open-small-{acc_id}.pdb", pdb_paths
            ):
                missing_counts["esm"] += 1

        # Log missing PDBs
        if missing_counts["experimental"] > 0 and len(experimental_pdbs) > 0:
            pct = 100 * missing_counts["experimental"] / len(experimental_pdbs)
            self.logger.warning(
                f"No PDB found for {missing_counts['experimental']} experimental PDBs ({pct:.1f}%)"
            )
        if missing_counts["alphafold"] > 0 and len(alphafold_pdbs) > 0:
            pct = 100 * missing_counts["alphafold"] / len(alphafold_pdbs)
            self.logger.warning(
                f"No PDB found for {missing_counts['alphafold']} AlphaFold proteins ({pct:.1f}%)"
            )
        if missing_counts["esm"] > 0 and len(esm_pdbs) > 0:
            pct = 100 * missing_counts["esm"] / len(esm_pdbs)
            self.logger.warning(
                f"No PDB found for {missing_counts['esm']} ESM proteins ({pct:.1f}%)"
            )

        self.logger.info(f"Generated {len(pdb_paths)} PDB symlinks")
        return pdb_paths

    def _foldseek_create_database(self) -> None:
        """
        Create a Foldseek database and index from the PDB symlinks in
        self.ALL_PDB_SYMLINK_PATH.  The database is written to
        self.FOLDSEEK_ALL_PATH / 'foldseekDB' and the temporary index
        scratch space is self.FOLDSEEK_TMPDIR (cleaned up on completion).
        """
        db_path = self.FOLDSEEK_ALL_PATH / "foldseekDB"

        shutil.rmtree(self.FOLDSEEK_ALL_PATH, ignore_errors=True)
        self.FOLDSEEK_ALL_PATH.mkdir(parents=True, exist_ok=True)

        # Count PDB files
        pdb_files = list(self.ALL_PDB_SYMLINK_PATH.glob("*.pdb"))
        n_pdbs = len(pdb_files)
        self.logger.info(
            f"[Foldseek createdb] Creating database from {n_pdbs} PDB files..."
        )
        self.logger.info(f"[Foldseek createdb] Input: {str(self.ALL_PDB_SYMLINK_PATH)}")
        self.logger.info(f"[Foldseek createdb] Output: {str(db_path)}")

        cmd = [
            "foldseek",
            "createdb",
            str(self.ALL_PDB_SYMLINK_PATH),
            str(db_path),
        ]

        start_time = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[Foldseek createdb] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[Foldseek createdb] Error: {out.stderr}")
            raise RuntimeError(f"Foldseek createdb failed: {out.stderr}")
        self.logger.info(f"[Foldseek createdb] Completed in {elapsed:.1f}s")

        self.logger.info(
            f"[Foldseek createindex] Creating index for faster searching..."
        )
        start_time = time.time()

        shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)
        self.FOLDSEEK_TMPDIR.mkdir(parents=True, exist_ok=True)

        cmd = ["foldseek", "createindex", str(db_path), str(self.FOLDSEEK_TMPDIR)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[Foldseek createindex] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[Foldseek createindex] Error: {out.stderr}")
            raise RuntimeError(f"Foldseek createindex failed: {out.stderr}")

        self.logger.info(f"[Foldseek createindex] Completed in {elapsed:.1f}s")
        shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)

    def _foldseek_all_vs_all(self) -> pl.DataFrame:
        """
        Run all-vs-all structure comparison using Foldseek easy-search against the
        database built from self.ALL_PDB_SYMLINK_PATH.  Results are saved to
        self.OUTPUT_DIR / 'foldseek_all_vs_all.tsv' for reproducibility.

        lddt values are read as strings then coerced to float so that
        scientific-notation entries (e.g. '2.190E-01') are handled correctly.
        Self-hits are removed before returning.

        Returns:
            DataFrame with columns: query (str), target (str), lddt (float64).
        """
        self.logger.info("=" * 60)
        self.logger.info("[Foldseek] Starting all-vs-all structure comparison")
        self.logger.info("=" * 60)

        if not self.CACHE_OVERWRITE and self.ALL_VS_ALL_CACHE_PATH.exists():
            self.logger.info(
                f"[Foldseek] Reusing cached all-vs-all TSV: {self.ALL_VS_ALL_CACHE_PATH}"
            )
            results_df = self._read_foldseek_all_vs_all_tsv(self.ALL_VS_ALL_CACHE_PATH)
            return results_df

        # Create database
        self._foldseek_create_database()
        db_path = self.FOLDSEEK_ALL_PATH / "foldseekDB"
        out_file = self.FOLDSEEK_ALL_PATH / "all_vs_all.tsv"

        # Count structures for estimation
        n_structures = len(list(self.ALL_PDB_SYMLINK_PATH.glob("*.pdb")))
        n_comparisons_estimate = n_structures * (n_structures - 1)

        self.logger.info(f"[Foldseek easy-search] Starting all-vs-all comparison...")
        self.logger.info(f"[Foldseek easy-search] Structures: {n_structures}")
        self.logger.info(
            f"[Foldseek easy-search] Max possible comparisons: {n_comparisons_estimate:,}"
        )
        self.logger.info(f"[Foldseek easy-search] Parameters:")
        self.logger.info(f"    --cov-mode 0 (coverage of query and target)")
        self.logger.info(f"    -c 0.8 (min 80% alignment coverage)")
        self.logger.info(f"    -s 9.5 (high sensitivity)")
        self.logger.info(f"    --alignment-mode 3 (global alignment)")
        self.logger.info(f"    --alignment-type 2 (3Di+AA alignment)")
        self.logger.info(
            f"[Foldseek easy-search] This may take a while for large datasets..."
        )

        try:
            cmd = [
                "foldseek",
                "easy-search",
                str(self.ALL_PDB_SYMLINK_PATH),
                str(db_path),
                str(out_file),
                str(self.FOLDSEEK_TMPDIR),
                "--cov-mode",
                "0",  # coverage of query and target
                "-c",
                "0.8",  # at least 80% alignment coverage
                "--max-seqs",
                "1000000000",  # No limit
                "--format-output",
                "query,target,lddt",
                "-s",
                "9.5",  # high sensitivity
                "--alignment-mode",
                "3",  # global alignment
                "--alignment-type",
                "2",  # Use 3Di+AA for alignment
            ]

            # Run subprocess in background and monitor progress
            start_time = time.time()
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Monitor progress by watching tmpdir size and output file
            # Foldseek writes intermediate results to tmpdir, then converts to TSV at the end
            last_size = 0
            last_log_time = start_time
            log_interval = 10

            with tqdm(
                desc="Foldseek all-vs-all",
                unit="MB",
                unit_scale=False,
                bar_format="{desc}: {elapsed} | {n:.1f} MB processed | {rate_fmt}",
            ) as pbar:
                while process.poll() is None:  # While process is running
                    time.sleep(2)  # Check every 2 seconds

                    # Calculate total size of tmpdir + output file
                    current_size = 0
                    try:
                        # Check tmpdir for intermediate files
                        for f in self.FOLDSEEK_TMPDIR.rglob("*"):
                            if f.is_file():
                                current_size += f.stat().st_size
                        # Also check output file
                        if out_file.exists():
                            current_size += out_file.stat().st_size
                    except Exception:
                        pass

                    # Update progress (in MB)
                    current_size_mb = current_size / (1024 * 1024)
                    delta_mb = current_size_mb - last_size
                    if delta_mb > 0:
                        pbar.update(delta_mb)
                        last_size = current_size_mb

                    # Periodic log message
                    current_time = time.time()
                    if current_time - last_log_time > log_interval:
                        elapsed_min = (current_time - start_time) / 60
                        self.logger.info(
                            f"[Foldseek] Still running... "
                            f"Elapsed: {elapsed_min:.1f} min, "
                            f"Data processed: {current_size_mb:.1f} MB"
                        )
                        last_log_time = current_time

            # Get return code and stderr
            _, stderr = process.communicate()
            elapsed = time.time() - start_time

            if process.returncode != 0:
                self.logger.error(f"[Foldseek easy-search] FAILED after {elapsed:.1f}s")
                self.logger.error(f"[Foldseek easy-search] Error: {stderr}")
                raise RuntimeError(f"Foldseek easy-search failed: {stderr}")
        finally:
            # Cleanup temp directories
            shutil.rmtree(self.FOLDSEEK_TMPDIR, ignore_errors=True)

        self.logger.info(
            f"[Foldseek easy-search] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)"
        )

        if not out_file.exists():
            self.logger.warning("[Foldseek] No output file - no alignments found")
            return pl.DataFrame(
                schema={"query": pl.Utf8, "target": pl.Utf8, "lddt": pl.Float64}
            )

        self.ALL_VS_ALL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_file, self.ALL_VS_ALL_CACHE_PATH)
        self.logger.info(
            f"[Foldseek] Copied raw all-vs-all TSV to {self.ALL_VS_ALL_CACHE_PATH}"
        )
        self.logger.info(
            f"[Foldseek] Reading raw results from {self.ALL_VS_ALL_CACHE_PATH}..."
        )
        results_df = self._read_foldseek_all_vs_all_tsv(
            self.ALL_VS_ALL_CACHE_PATH, raw_format=True
        )

        self.logger.info(f"[Foldseek] Results summary:")
        self.logger.info(f"    Total pairwise comparisons: {len(results_df):,}")
        self.logger.info(
            f"    Unique queries: {results_df.get_column('query').n_unique():,}"
        )
        self.logger.info(
            f"    Unique targets: {results_df.get_column('target').n_unique():,}"
        )
        self.logger.info(
            f"    LDDT range: "
            f"[{results_df.get_column('lddt').min():.3f}, {results_df.get_column('lddt').max():.3f}]"
        )
        self.logger.info(
            f"    LDDT mean: {results_df.get_column('lddt').mean():.3f}"
        )
        self.logger.info("=" * 60)

        shutil.rmtree(self.ALL_PDB_SYMLINK_PATH, ignore_errors=True)
        shutil.rmtree(self.FOLDSEEK_ALL_PATH, ignore_errors=True)

        results_df.write_csv(self.ALL_VS_ALL_CACHE_PATH, separator="\t")
        self.logger.info(
            f"[Foldseek] Saved normalized all-vs-all TSV to {self.ALL_VS_ALL_CACHE_PATH}"
        )

        return results_df

    def _read_foldseek_all_vs_all_tsv(
        self, tsv_path: Path, raw_format: bool = False
    ) -> pl.DataFrame:
        if raw_format:
            results_df = pl.read_csv(
                tsv_path,
                separator="\t",
                has_header=False,
                new_columns=["query", "target", "lddt"],
                schema_overrides={
                    "query": pl.Utf8,
                    "target": pl.Utf8,
                    "lddt": pl.Utf8,
                },
            )
        else:
            results_df = pl.read_csv(
                tsv_path,
                separator="\t",
                schema_overrides={
                    "query": pl.Utf8,
                    "target": pl.Utf8,
                    "lddt": pl.Utf8,
                },
            )
            required_cols = {"query", "target", "lddt"}
            if not required_cols.issubset(set(results_df.columns)):
                self.logger.warning(
                    f"Cached file {tsv_path} is missing required columns {sorted(required_cols)}. "
                    "Attempting to read it as a raw Foldseek TSV."
                )
                return self._read_foldseek_all_vs_all_tsv(tsv_path, raw_format=True)

        invalid_tokens = ["", "nan", "None"]
        results_df = (
            results_df.with_columns(
                pl.col("lddt").cast(pl.Float64, strict=False),
                pl.col("query")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.split("_")
                .list.first()
                .alias("query"),
                pl.col("target")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.split("_")
                .list.first()
                .alias("target"),
            )
            .with_columns(
                pl.when(pl.col("query").is_in(invalid_tokens))
                .then(None)
                .otherwise(pl.col("query"))
                .alias("query"),
                pl.when(pl.col("target").is_in(invalid_tokens))
                .then(None)
                .otherwise(pl.col("target"))
                .alias("target"),
            )
            .drop_nulls(["query", "target", "lddt"])
            .filter(pl.col("query") != pl.col("target"))
        )

        return results_df

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
            Array of max LDDT similarities for each protein (same order as self.unique_pdbs)
        """
        if self._max_lddt_array is not None:
            self.logger.info("Using cached max LDDT similarities")
            return self._max_lddt_array

        if not self.CACHE_OVERWRITE:
            loaded = self._load_similarity_cache(
                cache_path=self.MAX_SIM_CACHE_PATH,
                key_column="pdbs",
                value_column="max_similarity",
                ordered_keys=self.unique_pdbs,
            )
            if loaded is not None:
                values, sim_map = loaded
                self._max_lddt_array = values
                self._max_lddt_to_dataset = sim_map
                self.logger.info(
                    f"Using cached max LDDT similarities from {self.MAX_SIM_CACHE_PATH}"
                )
                return self._max_lddt_array

        n = len(self.unique_pdbs)
        self.logger.info(
            f"Computing max LDDT similarity for {n} proteins to all others..."
        )

        # Generate PDB paths for all proteins
        self._pdb_paths = self._generate_pdb_paths(self.full_dataset_df)

        # Only include proteins with available PDBs
        available_ids = list(self._pdb_paths.keys())
        self.logger.info(
            f"PDB structures available for {len(available_ids)}/{n} proteins"
        )

        # Run all-vs-all comparison
        results_df = self._foldseek_all_vs_all()

        # Compute max LDDT for each protein - only store max (O(n) memory)
        # Group by query and find max LDDT
        self.logger.info("Computing max LDDT per protein...")
        max_lddt_df = results_df.group_by("query").agg(pl.col("lddt").max().alias("lddt"))
        max_lddt_to_dataset = dict(
            zip(
                max_lddt_df.get_column("query").to_list(),
                max_lddt_df.get_column("lddt").to_list(),
            )
        )

        # Build array in same order as unique_pdbs
        max_similarities = np.zeros(n)
        for i, uid in tqdm(
            enumerate(self.unique_pdbs),
            total=n,
            desc="Building max LDDT array",
            leave=False,
        ):
            max_similarities[i] = max_lddt_to_dataset.get(uid, 0.0)

        self._max_lddt_array = max_similarities
        # Keep the in-memory map consistent with the persisted cache: proteins with
        # no Foldseek hit are treated as 0.0 similarity, not left missing/NaN.
        self._max_lddt_to_dataset = dict(zip(self.unique_pdbs, max_similarities))

        self._save_similarity_cache(
            cache_path=self.MAX_SIM_CACHE_PATH,
            key_column="pdbs",
            value_column="max_similarity",
            ordered_keys=self.unique_pdbs,
            values=max_similarities,
        )

        # Log statistics
        self.logger.info("Max LDDT similarity statistics:")
        self.logger.info(f"  Min: {max_similarities.min():.3f}")
        self.logger.info(f"  Max: {max_similarities.max():.3f}")
        self.logger.info(f"  Mean: {max_similarities.mean():.3f}")
        self.logger.info(f"  Median: {np.median(max_similarities):.3f}")
        self.logger.info(f"  Proteins with max_lddt=0: {np.sum(max_similarities == 0)}")

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

        similarity_df = pl.DataFrame(
            {
                "pdbs": list(self._max_lddt_to_dataset.keys()),
                "lddt": list(self._max_lddt_to_dataset.values()),
            }
        )
        grouped_input_df = (
            self.full_dataset_df.join(similarity_df, on="pdbs", how="left").with_columns(
                pl.col("lddt").fill_null(0.0)
            )
        )
        if "uniprot_date" in grouped_input_df.columns:
            grouped_input_df = grouped_input_df.with_columns(
                pl.col("uniprot_date").fill_null("NO_DATE")
            )

        pdb_cols = {"pdbs", "pdb_source", "pdb_type"}
        # __index_level_0__ is a pandas-generated artifact index; including it as a
        # group key would make every row its own group, defeating deduplication.
        index_cols = {"__index_level_0__"}
        group_cols = [col for col in grouped_input_df.columns if col not in pdb_cols | index_cols | {"lddt"}]

        if self.DROP_TO_1D:
            # Exclude measurement columns from group keys so that multiple
            # experimental values for the same (smiles, sequence) pair are
            # collapsed to one row — matching how the 1D dataset was built
            # (max value per pair).
            value_cols = {c for c in ["value", "log10_value"] if c in grouped_input_df.columns}
            group_cols_1d = [c for c in group_cols if c not in value_cols]
            agg_exprs = [pl.col("lddt").max().alias("lddt")]
            if "value" in grouped_input_df.columns:
                agg_exprs.append(pl.col("value").max().alias("value"))
            grouped_df = grouped_input_df.group_by(
                group_cols_1d, maintain_order=True
            ).agg(agg_exprs)
        else:
            grouped_df = (
                grouped_input_df.with_columns(
                    pl.struct("pdbs", "pdb_source", "pdb_type").alias("pdb_record")
                )
                .group_by(group_cols, maintain_order=True)
                .agg(
                    pl.col("lddt").max().alias("lddt"),
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
        if "uniprot_date" in split_df.columns:
            split_df = split_df.with_columns(
                pl.when(pl.col("uniprot_date") == "NO_DATE")
                .then(None)
                .otherwise(pl.col("uniprot_date"))
                .alias("uniprot_date")
            )
        split_df = split_df.drop("lddt").unique(maintain_order=True)
        if "value" in split_df.columns:
            split_df = split_df.with_columns(
                pl.col("value").cast(pl.Float64, strict=False).log10().alias("log10_value")
            )
        if self.DROP_TO_1D or split_df.is_empty():
            return split_df.to_pandas()

        metadata_cols = [
            col for col in ["pdbs", "pdb_source", "pdb_type"] if col in split_df.columns
        ]
        exploded_df = (
            split_df.drop(metadata_cols)
            .explode("pdb_records")
            .unnest("pdb_records")
        )
        return exploded_df.to_pandas()

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
                    pl.len().alias("row_count"),
                )
                .sort(["lddt", "row_count", "sequence"])
            )
        else:
            sequence_similarity_df = (
                grouped_df.group_by("sequence", maintain_order=True)
                .agg(
                    pl.col("lddt").max().alias("lddt"),
                    pl.col("pdb_records").list.len().sum().alias("row_count"),
                )
                .sort(["lddt", "row_count", "sequence"])
            )

        sequences = sequence_similarity_df.get_column("sequence").to_list()
        sequence_lddt = sequence_similarity_df.get_column("lddt").to_numpy()
        sequence_row_count = sequence_similarity_df.get_column("row_count").to_numpy()
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
        dissimilar_indices = np.where(eligible & dissimilar_mask)[0]

        self.logger.info(
            f"Found {len(dissimilar_indices)} dissimilar sequences "
            f"(max LDDT < {similarity_threshold})"
        )

        test_indices = []
        test_row_count = 0

        for idx in tqdm(
            dissimilar_indices,
            desc="Filling test set",
            leave=False,
        ):
            row_count = int(sequence_row_count[idx])
            if test_row_count + row_count <= max_test_rows:
                test_indices.append(idx)
                test_row_count += row_count
            else:
                break

        self.logger.info(
            f"Test set: {len(test_indices)} sequences, {test_row_count} rows"
        )

        # Phase 2: Fill val set to match test size
        used_indices = set(test_indices)
        remaining_indices = [idx for idx in range(n_unique) if idx not in used_indices]

        target_val_rows = test_row_count
        val_indices = []
        val_row_count = 0

        for idx in tqdm(
            remaining_indices,
            desc="Filling val set",
            leave=False,
        ):
            row_count = int(sequence_row_count[idx])
            if val_row_count + row_count <= target_val_rows:
                val_indices.append(idx)
                val_row_count += row_count
            else:
                break

        self.logger.info(f"Val set: {len(val_indices)} sequences, {val_row_count} rows")

        # Phase 3: Train = everything else
        val_set = set(val_indices)
        train_indices = list(set(remaining_indices) - val_set)
        train_row_count = n_full - test_row_count - val_row_count

        self.logger.info(
            f"Train set: {len(train_indices)} sequences, {train_row_count} rows"
        )

        test_sequences = {sequences[i] for i in test_indices}
        val_sequences = {sequences[i] for i in val_indices}
        train_sequences = {sequences[i] for i in train_indices}

        # Create split DataFrames
        grouped_splits = {
            "train": grouped_df.filter(pl.col("sequence").is_in(list(train_sequences))),
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

        for threshold in tqdm(
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
