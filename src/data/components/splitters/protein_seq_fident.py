import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.rich_progress import progress, write
from omegaconf import DictConfig
from src.utils.rich_progress import component_logger
from src.data.components.splitters.base import (
    configured,
    BaseThresholdedSimilaritySplitter,
    greedy_fill,
)


# Threshold-independent search settings; part of the cache fingerprint.
MMSEQS_SEARCH_ARGS = [
    "--cov-mode", "0", "-c", "0.8", "--max-seqs", "1000000000",
    "-s", "9.5", "--alignment-mode", "3",
]


def fident_fingerprint(sequences: list[str], gpu: bool, version: str) -> str:
    """Fingerprint of the all-vs-all cache; ids are indices into the sorted sequences."""
    return BaseThresholdedSimilaritySplitter._cache_fingerprint(
        sequences,
        search=MMSEQS_SEARCH_ARGS,
        max_mode="symmetric",
        gpu=gpu,
        version=version,
        cache_ids="sorted_index",
    )


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

        self.TEST_FRAC = configured(cfg.splits.test_frac, 0.1)
        self.VAL_FRAC = configured(cfg.splits.valid_frac, 0.1)
        self.CACHE_OVERWRITE = cfg.splits.get("similarity_cache_overwrite", False)
        self.THRESHOLDS = list(
            seq_cfg.get("similarity_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        )

        self.SEQUENCE_COL = cfg.splits.get("sequence_col", "sequence")

        self.MMSEQS_BIN = Path(cfg.splits.mmseqs_bin)
        self.GPU = bool(cfg.splits.get("mmseqs_gpu", False))

        self.INPUT_DATASET_PATH = Path(seq_cfg.input_dataset_parquet_file_path)
        self.OUTPUT_DIR = Path(seq_cfg.output_dir)

        # MMseqs2 scratch directories are per split, so split runs can overlap.
        self.MMSEQS_ALL_PATH = self.OUTPUT_DIR / "mmseqs2_work"
        self.MMSEQS_TMPDIR = self.OUTPUT_DIR / "mmseqs2_tmp"
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

        self.logger = component_logger(
            log_dir=LOG_PATH,
            log_file_name=cfg.splits.log_file_name,
        ).get_logger()

        if not (self.MMSEQS_BIN.is_file() and os.access(self.MMSEQS_BIN, os.X_OK)):
            msg = (
                f"MMseqs2 binary not found or not executable at {self.MMSEQS_BIN}. "
                "Run: bash scripts/install_external_tools.sh"
            )
            self.logger.error(msg)
            raise EnvironmentError(msg)
        self.TOOL_VERSION = subprocess.run(
            [str(self.MMSEQS_BIN), "version"], capture_output=True, text=True
        ).stdout.strip()
        self.logger.info(
            f"MMseqs2 binary: {self.MMSEQS_BIN}, version: {self.TOOL_VERSION}, "
            f"mode: {'GPU' if self.GPU else 'CPU'}"
        )

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

        # Will be computed lazily and cached for reuse
        self._max_fident_to_dataset = None  # dict: sequence -> max_fident
        self._max_fident_array = None  # array in same order as self.unique_sequences

        # Paths for FASTA input, all-vs-all raw results, and max-sim cache
        self.FASTA_PATH = self.OUTPUT_DIR / "unique_sequences.fasta"
        self.ALL_VS_ALL_CACHE_PATH = self.OUTPUT_DIR / "mmseqs_all_vs_all_fident.tsv"
        self.MAX_SIM_CACHE_PATH = self.OUTPUT_DIR / "max_fident_similarities.tsv"

    def _similarity_fingerprint(self) -> str:
        return fident_fingerprint(self.unique_sequences, self.GPU, self.TOOL_VERSION)

    # ------------------------------------------------------------------
    # FASTA helpers
    # ------------------------------------------------------------------

    def _write_fasta(self) -> Path:
        """
        Write all unique sequences to a FASTA file at self.FASTA_PATH.
        The FASTA id of a sequence is its index in sorted(self.unique_sequences),
        so ids do not depend on the dataset row order.

        Returns:
            Path to the written FASTA file.
        """
        self.logger.info(
            f"Writing {len(self.unique_sequences)} sequences to {self.FASTA_PATH}..."
        )
        with open(self.FASTA_PATH, "w") as f:
            for seq_id, seq in enumerate(sorted(self.unique_sequences)):
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

        cmd = [str(self.MMSEQS_BIN), "createdb", str(self.FASTA_PATH), str(db_path)]

        start_time = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[MMseqs2 createdb] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[MMseqs2 createdb] Error: {out.stderr}")
            raise RuntimeError(f"MMseqs2 createdb failed: {out.stderr}")
        self.logger.info(f"[MMseqs2 createdb] Completed in {elapsed:.1f}s")

        if self.GPU:
            self.logger.info("[MMseqs2 makepaddedseqdb] Creating padded database for GPU search...")
            cmd = [str(self.MMSEQS_BIN), "makepaddedseqdb", str(db_path), f"{db_path}_pad"]
            step = "makepaddedseqdb"
        else:
            self.logger.info("[MMseqs2 createindex] Creating index for faster searching...")
            shutil.rmtree(self.MMSEQS_TMPDIR, ignore_errors=True)
            self.MMSEQS_TMPDIR.mkdir(parents=True, exist_ok=True)
            cmd = [str(self.MMSEQS_BIN), "createindex", str(db_path), str(self.MMSEQS_TMPDIR)]
            step = "createindex"

        start_time = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if out.returncode != 0:
            self.logger.error(f"[MMseqs2 {step}] FAILED after {elapsed:.1f}s")
            self.logger.error(f"[MMseqs2 {step}] Error: {out.stderr}")
            raise RuntimeError(f"MMseqs2 {step} failed: {out.stderr}")

        self.logger.info(f"[MMseqs2 {step}] Completed in {elapsed:.1f}s")
        shutil.rmtree(self.MMSEQS_TMPDIR, ignore_errors=True)

    def _read_hits(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep="\t",
            dtype={"query": np.int32, "target": np.int32, "fident": np.float64},
            **kwargs,
        )

    def _mmseqs_all_vs_all(self) -> pd.DataFrame:
        """
        Run all-vs-all sequence comparison using MMseqs2 easy-search.
        Results are saved to self.ALL_VS_ALL_CACHE_PATH for reproducibility.

        Self-hits are removed before returning.

        Directly mirrors _foldseek_all_vs_all, with lddt -> fident and
        --alignment-type dropped (not applicable to sequence search).

        Returns:
            DataFrame with columns: query (int32), target (int32), fident (float64).
            query and target are indices into sorted(self.unique_sequences);
            the cache file stores the same integer ids.
        """
        self.logger.info("=" * 60)
        self.logger.info("[MMseqs2] Starting all-vs-all sequence comparison")
        self.logger.info("=" * 60)

        if (
            not self.CACHE_OVERWRITE
            and self.ALL_VS_ALL_CACHE_PATH.exists()
            and self._cache_matches(
                self.ALL_VS_ALL_CACHE_PATH, self._similarity_fingerprint()
            )
        ):
            self.logger.info(
                f"[MMseqs2] Reusing cached all-vs-all TSV: {self.ALL_VS_ALL_CACHE_PATH}"
            )
            return self._read_hits(self.ALL_VS_ALL_CACHE_PATH)

        # Write FASTA and build database
        self._write_fasta()
        self._mmseqs_create_database()

        db_path = self.MMSEQS_ALL_PATH / "mmseqsDB"
        if self.GPU:
            db_path = Path(f"{db_path}_pad")
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
                str(self.MMSEQS_BIN),
                "easy-search",
                str(self.FASTA_PATH),  # query: same FASTA -> all-vs-all
                str(db_path),  # target: prebuilt database
                str(out_file),
                str(self.MMSEQS_TMPDIR),
                "--format-output",
                "query,target,fident",
                *MMSEQS_SEARCH_ARGS,
                *(["--gpu", "1"] if self.GPU else []),
            ]

            elapsed = self._run_search_with_progress(
                cmd, tool="MMseqs2", tmpdir=self.MMSEQS_TMPDIR, out_file=out_file
            )

        finally:
            shutil.rmtree(self.MMSEQS_TMPDIR, ignore_errors=True)

        self.logger.info(
            f"[MMseqs2 easy-search] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)"
        )

        if not out_file.exists():
            self.logger.warning("[MMseqs2] No output file — no alignments found")
            return pd.DataFrame(columns=["query", "target", "fident"]).astype(
                {"query": np.int32, "target": np.int32, "fident": np.float64}
            )

        self.logger.info(f"[MMseqs2] Reading results from {out_file}...")
        results_df = self._read_hits(
            out_file, header=None, names=["query", "target", "fident"]
        )
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
        self._write_cache_fingerprint(
            self.ALL_VS_ALL_CACHE_PATH, self._similarity_fingerprint()
        )

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
                fingerprint=self._similarity_fingerprint(),
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
        # Max over both query and target columns (symmetric), indexed by sorted-sequence id.
        max_by_id = (
            pd.concat(
                [
                    results_df.groupby("query")["fident"].max(),
                    results_df.groupby("target")["fident"].max().rename_axis("query"),
                ]
            )
            .groupby(level=0)
            .max()
        )
        by_sorted_id = np.zeros(n)
        by_sorted_id[max_by_id.index.to_numpy()] = max_by_id.to_numpy()

        # Build array in same order as unique_sequences
        sorted_id = {seq: i for i, seq in enumerate(sorted(self.unique_sequences))}
        max_similarities = by_sorted_id[[sorted_id[seq] for seq in self.unique_sequences]]

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
            fingerprint=self._similarity_fingerprint(),
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
        remaining_indices = [idx for idx in range(n_unique) if idx not in used_indices]

        # val is sized from valid_frac, not from the test row count.
        target_val_rows = int(self.VAL_FRAC * n_full)
        val_indices, val_row_count = greedy_fill(
            remaining_indices, row_count_of, target_val_rows, desc="Filling val set"
        )

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

        for threshold in progress(
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
