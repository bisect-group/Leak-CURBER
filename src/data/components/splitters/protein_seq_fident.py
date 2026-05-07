import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from src.utils.tqdmlogger import TqdmLogger
from src.data.components.splitters.base import BaseThresholdedSimilaritySplitter


class ProteinSeqMaxFidentSimilaritySplitter(BaseThresholdedSimilaritySplitter):
    """Splits a dataset based on protein sequence fident similarity using MMseqs2.

    This class provides methods to split a dataset into training, validation,
    and test sets based on the sequence similarity (fident score) of proteins.

    The approach:
    1. Compute max fident similarity of each sequence to ALL other sequences in dataset
    2. Sequences with low max similarity (sequentially dissimilar) go to test first and validation second
    3. This ensures test sequences are sequentially distinct from the training set

    fident (fraction of identical residues) is the fraction of identical residue pairs
    in the gapped alignment region. It ranges from 0 to 1, where 1 indicates identical
    sequences. Unlike Hamming similarity, fident accounts for insertions and deletions
    via gapped alignment and uses BLOSUM62-based alignment scoring.

    This class mirrors the structure of ProteinMaxLDDTSimilaritySplitter, replacing
    Foldseek's structural comparison with MMseqs2's sequence comparison. It is
    intended to produce a "medium" split sitting between the hard structural split
    (lDDT) and the easy random split.

    This class uses MMseqs2 for fast sequence comparison, which requires a FASTA
    file of all unique sequences.
    """

    def __init__(self, cfg: DictConfig):
        seq_cfg = cfg.splits.dataset.sequence_split

        self.TEST_FRAC = cfg.splits.test_frac or 0.1
        self.CACHE_OVERWRITE = cfg.splits.get("similarity_cache_overwrite", False)
        self.THRESHOLDS = list(
            seq_cfg.get("similarity_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )

        self.SEQUENCE_COL = cfg.splits.get("sequence_col", "sequence")

        # MMseqs2 working directories
        self.MMSEQS_ALL_PATH = Path(cfg.splits.all_mmseqs2_path)
        self.MMSEQS_TMPDIR = Path(cfg.splits.mmseqs2_tmpdir)

        self.INPUT_DATASET_PATH = Path(seq_cfg.input_dataset_parquet_file_path)
        self.OUTPUT_DIR = Path(seq_cfg.output_dir)
        self.UNIQUE_SEQUENCE_SIMILARITY_PLOT_PATH = Path(seq_cfg.unique_similarity_plot_path)
        self.FULL_DATASET_SEQUENCE_SIMILARITY_PLOT_PATH = Path(
            seq_cfg.full_dataset_similarity_plot_path
        )

        LOG_PATH = Path(cfg.splits.log_dir)

        # Create all necessary directories
        for path in [
            LOG_PATH,
            self.OUTPUT_DIR,
            self.UNIQUE_SEQUENCE_SIMILARITY_PLOT_PATH.parent,
            self.FULL_DATASET_SEQUENCE_SIMILARITY_PLOT_PATH.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.logger = TqdmLogger(
            log_dir=LOG_PATH,
            log_file_name=cfg.splits.log_file_name,
        ).get_logger()

        # Check for MMseqs2 installation
        if shutil.which("mmseqs") is None:
            msg = "MMseqs2 installation not found. Visit https://github.com/soedinglab/MMseqs2 to install."
            self.logger.error(msg)
            raise EnvironmentError(msg)

        if not self.INPUT_DATASET_PATH.exists():
            msg = f"Dataset parquet file not found at {self.INPUT_DATASET_PATH}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.logger.info("Loading full dataset...")
        self.full_dataset_df = pd.read_parquet(self.INPUT_DATASET_PATH)
        self.logger.info(f"Full dataset shape: {self.full_dataset_df.shape}")

        # Get unique sequences (drop NaN — these are assigned to train unconditionally)
        self.unique_sequences = (
            self.full_dataset_df[self.SEQUENCE_COL].dropna().unique().tolist()
        )
        self.logger.info(f"Unique sequences: {len(self.unique_sequences)}")

        # MMseqs2 uses integer IDs in the FASTA; maintain a bidirectional mapping
        # seq_id  (str "0", "1", ...) <-> sequence (str)
        self._id_to_seq: dict[str, str] = {
            str(i): seq for i, seq in enumerate(self.unique_sequences)
        }
        self._seq_to_id: dict[str, str] = {
            seq: str(i) for i, seq in enumerate(self.unique_sequences)
        }

        # Will be computed lazily and cached for reuse
        self._max_fident_to_dataset = None  # dict: sequence -> max_fident
        self._max_fident_array = None  # array in same order as self.unique_sequences

        # Paths for FASTA input, all-vs-all raw results, and max-sim cache
        self.FASTA_PATH = self.OUTPUT_DIR / "unique_sequences.fasta"
        self.ALL_VS_ALL_CACHE_PATH = self.OUTPUT_DIR / "mmseqs_all_vs_all_fident.tsv"
        self.MAX_SIM_CACHE_PATH = self.OUTPUT_DIR / "max_fident_similarities.tsv"

    # ------------------------------------------------------------------
    # FASTA helpers
    # ------------------------------------------------------------------

    def _write_fasta(self) -> Path:
        """
        Write all unique sequences to a FASTA file at self.FASTA_PATH.
        Each sequence is assigned an integer ID (its index in self.unique_sequences)
        so that MMseqs2 query/target IDs can be mapped back to sequences.

        Returns:
            Path to the written FASTA file.
        """
        self.logger.info(
            f"Writing {len(self.unique_sequences)} sequences to {self.FASTA_PATH}..."
        )
        with open(self.FASTA_PATH, "w") as f:
            for seq_id, seq in self._id_to_seq.items():
                f.write(f">{seq_id}\n{seq}\n")
        self.logger.info("FASTA written.")
        return self.FASTA_PATH

    # ------------------------------------------------------------------
    # MMseqs2 database and search
    # ------------------------------------------------------------------

    def _mmseqs_create_database(self) -> None:
        """
        Create an MMseqs2 sequence database and index from self.FASTA_PATH.
        The database is written to self.MMSEQS_ALL_PATH / 'mmseqsDB'.
        Mirrors _foldseek_create_database exactly.
        """
        db_path = self.MMSEQS_ALL_PATH / "mmseqsDB"

        shutil.rmtree(self.MMSEQS_ALL_PATH, ignore_errors=True)
        self.MMSEQS_ALL_PATH.mkdir(parents=True, exist_ok=True)

        n_seqs = len(self.unique_sequences)
        self.logger.info(
            f"[MMseqs2 createdb] Creating database from {n_seqs} sequences..."
        )
        self.logger.info(f"[MMseqs2 createdb] Input:  {self.FASTA_PATH}")
        self.logger.info(f"[MMseqs2 createdb] Output: {db_path}")

        cmd = ["mmseqs", "createdb", str(self.FASTA_PATH), str(db_path)]

        start_time = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[MMseqs2 createdb] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[MMseqs2 createdb] Error: {out.stderr}")
            raise RuntimeError(f"MMseqs2 createdb failed: {out.stderr}")
        self.logger.info(f"[MMseqs2 createdb] Completed in {elapsed:.1f}s")

        self.logger.info("[MMseqs2 createindex] Creating index for faster searching...")
        shutil.rmtree(self.MMSEQS_TMPDIR, ignore_errors=True)
        self.MMSEQS_TMPDIR.mkdir(parents=True, exist_ok=True)

        cmd = ["mmseqs", "createindex", str(db_path), str(self.MMSEQS_TMPDIR)]

        start_time = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[MMseqs2 createindex] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[MMseqs2 createindex] Error: {out.stderr}")
            raise RuntimeError(f"MMseqs2 createindex failed: {out.stderr}")

        self.logger.info(f"[MMseqs2 createindex] Completed in {elapsed:.1f}s")
        shutil.rmtree(self.MMSEQS_TMPDIR, ignore_errors=True)

    def _mmseqs_all_vs_all(self) -> pd.DataFrame:
        """
        Run all-vs-all sequence comparison using MMseqs2 easy-search.
        Results are saved to self.ALL_VS_ALL_CACHE_PATH for reproducibility.

        fident values are read as strings then coerced to float.
        Self-hits are removed before returning.

        Directly mirrors _foldseek_all_vs_all, with lddt -> fident and
        --alignment-type dropped (not applicable to sequence search).

        Returns:
            DataFrame with columns: query (str), target (str), fident (float64).
            query and target are the original sequence strings (not integer IDs).
        """
        self.logger.info("=" * 60)
        self.logger.info("[MMseqs2] Starting all-vs-all sequence comparison")
        self.logger.info("=" * 60)

        if not self.CACHE_OVERWRITE and self.ALL_VS_ALL_CACHE_PATH.exists():
            self.logger.info(
                f"[MMseqs2] Reusing cached all-vs-all TSV: {self.ALL_VS_ALL_CACHE_PATH}"
            )
            results_df = pd.read_csv(self.ALL_VS_ALL_CACHE_PATH, sep="\t")
            required_cols = {"query", "target", "fident"}
            if not required_cols.issubset(results_df.columns):
                raise ValueError(
                    f"Cached file {self.ALL_VS_ALL_CACHE_PATH} is missing required columns "
                    f"{sorted(required_cols)}"
                )
            results_df["fident"] = pd.to_numeric(results_df["fident"], errors="coerce")
            results_df = results_df.dropna(
                subset=["query", "target", "fident"]
            ).reset_index(drop=True)
            results_df = results_df[results_df["query"] != results_df["target"]]
            return results_df

        # Write FASTA and build database
        self._write_fasta()
        self._mmseqs_create_database()

        db_path = self.MMSEQS_ALL_PATH / "mmseqsDB"
        out_file = self.MMSEQS_ALL_PATH / "all_vs_all.tsv"

        n_seqs = len(self.unique_sequences)
        n_comparisons_estimate = n_seqs * (n_seqs - 1)

        self.logger.info("[MMseqs2 easy-search] Starting all-vs-all comparison...")
        self.logger.info(f"[MMseqs2 easy-search] Sequences:              {n_seqs}")
        self.logger.info(
            f"[MMseqs2 easy-search] Max possible comparisons: {n_comparisons_estimate:,}"
        )
        self.logger.info("[MMseqs2 easy-search] Parameters:")
        self.logger.info(
            "    --cov-mode 0      (bidirectional coverage of query and target)"
        )
        self.logger.info("    -c 0.8            (min 80% alignment coverage)")
        self.logger.info("    -s 9.5            (high sensitivity)")
        self.logger.info("    --alignment-mode 3 (global alignment)")
        self.logger.info(
            "[MMseqs2 easy-search] This may take a while for large datasets..."
        )

        try:
            cmd = [
                "mmseqs",
                "easy-search",
                str(self.FASTA_PATH),  # query: same FASTA -> all-vs-all
                str(db_path),  # target: prebuilt database
                str(out_file),
                str(self.MMSEQS_TMPDIR),
                "--cov-mode",
                "0",
                "-c",
                "0.8",
                "--max-seqs",
                "1000000000",
                "--format-output",
                "query,target,fident",
                "-s",
                "9.5",
                "--alignment-mode",
                "3",
            ]

            start_time = time.time()
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            last_size = 0
            last_log_time = start_time
            log_interval = 10

            with tqdm(
                desc="MMseqs2 all-vs-all",
                unit="MB",
                unit_scale=False,
                bar_format="{desc}: {elapsed} | {n:.1f} MB processed | {rate_fmt}",
            ) as pbar:
                while process.poll() is None:
                    time.sleep(2)

                    current_size = 0
                    try:
                        for f in self.MMSEQS_TMPDIR.rglob("*"):
                            if f.is_file():
                                current_size += f.stat().st_size
                        if out_file.exists():
                            current_size += out_file.stat().st_size
                    except Exception:
                        pass

                    current_size_mb = current_size / (1024 * 1024)
                    delta_mb = current_size_mb - last_size
                    if delta_mb > 0:
                        pbar.update(delta_mb)
                        last_size = current_size_mb

                    current_time = time.time()
                    if current_time - last_log_time > log_interval:
                        elapsed_min = (current_time - start_time) / 60
                        self.logger.info(
                            f"[MMseqs2] Still running... "
                            f"Elapsed: {elapsed_min:.1f} min, "
                            f"Data processed: {current_size_mb:.1f} MB"
                        )
                        last_log_time = current_time

            _, stderr = process.communicate()
            elapsed = time.time() - start_time

            if process.returncode != 0:
                self.logger.error(f"[MMseqs2 easy-search] FAILED after {elapsed:.1f}s")
                self.logger.error(f"[MMseqs2 easy-search] Error: {stderr}")
                raise RuntimeError(f"MMseqs2 easy-search failed: {stderr}")

        finally:
            shutil.rmtree(self.MMSEQS_TMPDIR, ignore_errors=True)

        self.logger.info(
            f"[MMseqs2 easy-search] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)"
        )

        if not out_file.exists():
            self.logger.warning("[MMseqs2] No output file — no alignments found")
            return pd.DataFrame(columns=["query", "target", "fident"])

        self.logger.info(f"[MMseqs2] Reading results from {out_file}...")
        results_df = pd.read_csv(
            out_file,
            sep="\t",
            header=None,
            names=["query", "target", "fident"],
            dtype={"query": str, "target": str, "fident": str},
        )
        results_df["fident"] = pd.to_numeric(results_df["fident"], errors="coerce")

        # MMseqs2 query/target IDs are the integer index strings in the FASTA.
        # Map them back to the original sequence strings
        results_df["query"] = results_df["query"].map(self._id_to_seq)
        results_df["target"] = results_df["target"].map(self._id_to_seq)

        results_df = results_df.dropna(
            subset=["query", "target", "fident"]
        ).reset_index(drop=True)

        # Remove self-hits
        results_df = results_df[results_df["query"] != results_df["target"]]

        self.logger.info("[MMseqs2] Results summary:")
        self.logger.info(f"    Total pairwise comparisons: {len(results_df):,}")
        self.logger.info(f"    Unique queries:  {results_df['query'].nunique():,}")
        self.logger.info(f"    Unique targets:  {results_df['target'].nunique():,}")
        self.logger.info(
            f"    fident range: [{results_df['fident'].min():.3f}, "
            f"{results_df['fident'].max():.3f}]"
        )
        self.logger.info(f"    fident mean:  {results_df['fident'].mean():.3f}")
        self.logger.info("=" * 60)

        shutil.rmtree(self.MMSEQS_ALL_PATH, ignore_errors=True)

        results_df.to_csv(self.ALL_VS_ALL_CACHE_PATH, sep="\t", index=False)

        return results_df

    # ------------------------------------------------------------------
    # Max-fident computation (mirrors compute_max_lddt_to_dataset)
    # ------------------------------------------------------------------

    def compute_max_fident_to_dataset(self) -> np.ndarray:
        """
        Compute max fident similarity of each sequence to ALL other sequences in dataset.
        This is computed once and cached.

        NOTE on fident direction:
        - Higher fident (closer to 1.0) = MORE similar sequences
        - Lower fident (closer to 0.0) = MORE dissimilar sequences

        For a challenging test set, we want sequences with LOW max fident
        (sequentially dissimilar from the rest of the dataset).

        Memory optimization: Only stores max fident per sequence (O(n)), not all
        pairwise comparisons (O(n²)).

        Returns:
            Array of max fident similarities for each sequence
            (same order as self.unique_sequences).
        """
        if self._max_fident_array is not None:
            self.logger.info("Using cached max fident similarities")
            return self._max_fident_array

        if not self.CACHE_OVERWRITE:
            loaded = self._load_similarity_cache(
                cache_path=self.MAX_SIM_CACHE_PATH,
                key_column=self.SEQUENCE_COL,
                value_column="max_similarity",
                ordered_keys=self.unique_sequences,
            )
            if loaded is not None:
                values, sim_map = loaded
                self._max_fident_array = values
                self._max_fident_to_dataset = sim_map
                self.logger.info(
                    f"Using cached max fident similarities from {self.MAX_SIM_CACHE_PATH}"
                )
                return self._max_fident_array

        n = len(self.unique_sequences)
        self.logger.info(
            f"Computing max fident similarity for {n} sequences to all others..."
        )

        # Run all-vs-all comparison
        results_df = self._mmseqs_all_vs_all()

        # Compute max fident per sequence — only store max (O(n) memory)
        self.logger.info("Computing max fident per sequence...")
        max_fident_df = results_df.groupby("query")["fident"].max().reset_index()
        max_fident_to_dataset = dict(
            zip(max_fident_df["query"], max_fident_df["fident"])
        )

        # Build array in same order as unique_sequences
        max_similarities = np.zeros(n)
        for i, seq in tqdm(
            enumerate(self.unique_sequences),
            total=n,
            desc="Building max fident array",
            leave=False,
        ):
            max_similarities[i] = max_fident_to_dataset.get(seq, 0.0)

        self._max_fident_array = max_similarities
        # Keep the in-memory map consistent with the persisted cache: sequences with
        # no MMseqs2 hit are treated as 0.0 similarity, not left missing.
        self._max_fident_to_dataset = dict(zip(self.unique_sequences, max_similarities))

        self._save_similarity_cache(
            cache_path=self.MAX_SIM_CACHE_PATH,
            key_column=self.SEQUENCE_COL,
            value_column="max_similarity",
            ordered_keys=self.unique_sequences,
            values=max_similarities,
        )

        self.logger.info("Max fident similarity statistics:")
        self.logger.info(f"  Min:    {max_similarities.min():.3f}")
        self.logger.info(f"  Max:    {max_similarities.max():.3f}")
        self.logger.info(f"  Mean:   {max_similarities.mean():.3f}")
        self.logger.info(f"  Median: {np.median(max_similarities):.3f}")
        self.logger.info(
            f"  Sequences with max_fident=0: {np.sum(max_similarities == 0)}"
        )

        return max_similarities

    def get_max_fident_for_sequences(self, sequences: list) -> dict:
        """
        Get max fident to dataset for given sequences from cache.

        Args:
            sequences: List of sequence strings to get max fident for

        Returns:
            Dictionary mapping sequence -> max_fident_to_dataset
        """
        if self._max_fident_to_dataset is None:
            msg = (
                "Max fident to dataset not computed yet. "
                "Call compute_max_fident_to_dataset() first."
            )
            self.logger.error(msg)
            raise RuntimeError(msg)

        similarities = {
            seq: self._max_fident_to_dataset.get(seq, 0.0) for seq in sequences
        }

        sim_values = list(similarities.values())
        if sim_values:
            self.logger.info(
                f"Max fident to dataset statistics for {len(sequences)} sequences:"
            )
            self.logger.info(f"  Min:    {np.min(sim_values):.3f}")
            self.logger.info(f"  Max:    {np.max(sim_values):.3f}")
            self.logger.info(f"  Mean:   {np.mean(sim_values):.3f}")
            self.logger.info(f"  Median: {np.median(sim_values):.3f}")

        return similarities

    def dissimilarity_based_split(self, similarity_threshold) -> tuple[dict, dict]:
        """
        Split dataset based on protein sequence dissimilarity using max fident.

        NOTE on fident direction:
        - Higher fident = MORE similar (1.0 = identical sequences)
        - Lower fident  = MORE dissimilar (0.0 = no identical aligned residues)

        Algorithm:
        1. Compute max fident of each sequence to ALL other sequences
        2. Sequences with max fident < threshold are "dissimilar"
        3. Assign dissimilar sequences to test/val first (most dissimilar = lowest max fident)

        NaN sequences are placed in the training set unconditionally.

        Args:
            similarity_threshold: Sequences with max fident below this are dissimilar

        Returns:
            tuple: (splits_dict, similarities_dict)
                splits_dict: {"train": df, "val": df, "test": df}
                similarities_dict: {"test": {seq: max_fident}, "val": {seq: max_fident}}
        """
        self.logger.info(
            f"Splitting with similarity_threshold={similarity_threshold}, "
            f"max_test_frac={self.TEST_FRAC}"
        )

        max_fident_to_dataset = self.compute_max_fident_to_dataset()

        row_count_series = (
            self.full_dataset_df[self.full_dataset_df[self.SEQUENCE_COL].notna()][
                self.SEQUENCE_COL
            ]
            .value_counts()
        )
        sequence_similarity_df = pd.DataFrame(
            {
                self.SEQUENCE_COL: self.unique_sequences,
                "fident": max_fident_to_dataset,
            }
        )
        sequence_similarity_df["row_count"] = (
            sequence_similarity_df[self.SEQUENCE_COL]
            .map(row_count_series)
            .fillna(0)
            .astype(int)
        )
        sequence_similarity_df = sequence_similarity_df.sort_values(
            ["fident", "row_count", self.SEQUENCE_COL],
            kind="mergesort",
        ).reset_index(drop=True)

        sequences = sequence_similarity_df[self.SEQUENCE_COL].tolist()
        sequence_fident = sequence_similarity_df["fident"].to_numpy()
        sequence_row_count = sequence_similarity_df["row_count"].to_numpy()
        n_unique = len(sequences)
        n_full = len(self.full_dataset_df)
        max_test_rows = int(self.TEST_FRAC * n_full)

        self.logger.info(f"Max test rows: {max_test_rows}")

        eligible = np.isfinite(sequence_fident)

        # Find dissimilar sequences (max fident below threshold)
        dissimilar_mask = sequence_fident < similarity_threshold
        dissimilar_indices = np.where(eligible & dissimilar_mask)[0]

        self.logger.info(
            f"Found {len(dissimilar_indices)} dissimilar sequences "
            f"(max fident < {similarity_threshold})"
        )

        # Phase 1: Fill test set with most dissimilar sequences
        test_indices = []
        test_row_count = 0

        for idx in tqdm(dissimilar_indices, desc="Filling test set", leave=False):
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

        for idx in tqdm(remaining_indices, desc="Filling val set", leave=False):
            row_count = int(sequence_row_count[idx])
            if val_row_count + row_count <= target_val_rows:
                val_indices.append(idx)
                val_row_count += row_count
            else:
                break

        self.logger.info(f"Val set: {len(val_indices)} sequences, {val_row_count} rows")

        # Phase 3: Train = everything else
        val_set = set(val_indices)
        train_indices = [idx for idx in remaining_indices if idx not in val_set]
        train_row_count = n_full - test_row_count - val_row_count

        self.logger.info(
            f"Train set: {len(train_indices)} sequences, {train_row_count} rows"
        )

        # Convert indices to sequence strings
        test_seqs = {sequences[i] for i in test_indices}
        val_seqs = {sequences[i] for i in val_indices}
        train_seqs = {sequences[i] for i in train_indices}

        # Create split DataFrames (NaN sequences -> train, mirrors LDDT splitter)
        nan_mask = self.full_dataset_df[self.SEQUENCE_COL].isna()
        splits = {
            "train": pd.concat(
                [
                    self.full_dataset_df[
                        self.full_dataset_df[self.SEQUENCE_COL].isin(train_seqs)
                    ].reset_index(drop=True),
                    self.full_dataset_df[nan_mask].reset_index(drop=True),
                ],
                ignore_index=True,
            ),
            "val": self.full_dataset_df[
                self.full_dataset_df[self.SEQUENCE_COL].isin(val_seqs)
            ].reset_index(drop=True),
            "test": self.full_dataset_df[
                self.full_dataset_df[self.SEQUENCE_COL].isin(test_seqs)
            ].reset_index(drop=True),
        }

        if nan_mask.any():
            self.logger.info(f"Added {nan_mask.sum()} NaN-sequence rows to train set")

        # Verify no overlap
        assert len(test_seqs & val_seqs) == 0, "Test and val overlap!"
        assert len(test_seqs & train_seqs) == 0, "Test and train overlap!"
        assert len(val_seqs & train_seqs) == 0, "Val and train overlap!"

        self.logger.info("Getting max fident to dataset for test sequences...")
        test_similarities = self.get_max_fident_for_sequences(list(test_seqs))

        self.logger.info("Getting max fident to dataset for val sequences...")
        val_similarities = self.get_max_fident_for_sequences(list(val_seqs))

        similarities = {
            "test": test_similarities,
            "val": val_similarities,
        }

        for split_name, split_df in splits.items():
            pct = 100 * len(split_df) / n_full
            self.logger.info(f"{split_name}: {len(split_df)} rows ({pct:.1f}%)")

        return splits, similarities

    def run_splits_across_thresholds(self) -> tuple[dict, dict]:
        """
        Run fident-based splits across multiple similarity thresholds.

        Max fident to all other sequences is computed ONCE, then splits are
        generated for each threshold. Mirrors run_splits_across_thresholds in
        ProteinMaxLDDTSimilaritySplitter.

        Returns:
            tuple: (all_splits, all_similarities) — both dicts keyed by threshold
        """
        self.logger.info(
            "Computing max fident similarities to full dataset (one time)..."
        )
        self.compute_max_fident_to_dataset()

        all_splits = {}
        all_similarities = {}

        for threshold in tqdm(
            self.THRESHOLDS, desc="Processing thresholds", unit="threshold"
        ):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"Processing similarity threshold: {threshold}")
            self.logger.info(f"{'=' * 50}")

            splits, similarities = self.dissimilarity_based_split(
                similarity_threshold=threshold,
            )

            all_splits[threshold] = splits
            all_similarities[threshold] = similarities

        return all_splits, all_similarities

    def plot_fident_distribution(self, all_similarities: dict) -> None:
        values_by_threshold = self._build_value_map_from_similarity_dicts(
            all_similarities
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 21),
            output_path=self.UNIQUE_SEQUENCE_SIMILARITY_PLOT_PATH,
            xlabel="Max fident to Dataset",
            ylabel="Count",
            title_fn=lambda threshold, val_values, test_values: (
                f"Threshold: {threshold}\n"
                f"Val: {len(val_values)}, Test: {len(test_values)}"
            ),
            stats_mode="summary",
            legend_loc="upper right",
            hist_range=(0, 1),
        )

    def plot_full_dataset_fident_distribution(
        self, all_splits: dict, all_similarities: dict
    ) -> None:
        full_dataset_size = len(self.full_dataset_df)
        values_by_threshold = self._build_value_map_from_split_frames(
            all_splits=all_splits,
            all_similarities=all_similarities,
            key_column=self.SEQUENCE_COL,
        )
        self._plot_threshold_split_distributions(
            values_by_threshold=values_by_threshold,
            bins=np.linspace(0, 1, 21),
            output_path=self.FULL_DATASET_SEQUENCE_SIMILARITY_PLOT_PATH,
            xlabel="Max fident to Dataset",
            ylabel="Number of Rows",
            title_fn=lambda threshold, val_values, test_values: (
                f"Full Dataset fident Similarity\nThreshold {threshold}\n"
                f"Val: {len(val_values)} ({100 * len(val_values) / full_dataset_size:.1f}%), "
                f"Test: {len(test_values)} ({100 * len(test_values) / full_dataset_size:.1f}%)"
            ),
            stats_mode="detailed",
            legend_loc="upper left",
        )

    def plot_unique_distribution(self, all_similarities: dict) -> None:
        self.plot_fident_distribution(all_similarities)

    def plot_full_distribution(self, all_splits: dict, all_similarities: dict) -> None:
        self.plot_full_dataset_fident_distribution(all_splits, all_similarities)

    def get_output_dir(self) -> Path:
        return self.OUTPUT_DIR


if __name__ == "__main__":
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(config_name="data_processing")

    splitter = ProteinSeqMaxFidentSimilaritySplitter(cfg=cfg)
    all_splits, all_similarities = splitter.generate_splits()
