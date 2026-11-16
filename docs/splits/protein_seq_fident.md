# Protein Sequence Identity (fident) Split

This document describes `ProteinSeqMaxFidentSimilaritySplitter` in `src/data/components/splitters/protein_seq_fident.py`.

## Overview

The splitter uses MMseqs2 to find, for each unique sequence, its highest sequence identity (`fident`) to any other sequence in the dataset. Sequences with low maximum identity go to test. The split is made once for each threshold in `similarity_thresholds`.

`fident` is the fraction of identical residues in the gapped alignment. It goes from 0 to 1. Gaps are allowed, so insertions and deletions are handled.

This is a sequence-based split. The structure-based split is in `protein_lddt.md`.

---

## Processing Pipeline

### Step 1: Load unique sequences
1. Read the input parquet file.
2. Take the unique non-null values of `sequence` (key `splits.sequence_col`).
3. Rows with a null sequence always go to train.

### Step 2: All-vs-all search
1. Write a FASTA file `unique_sequences.fasta`. The FASTA id is the index of the sequence in the sorted unique sequences, so the ids do not depend on row order.
2. Build an MMseqs2 database. CPU mode also builds an index. GPU mode (`mmseqs_gpu: true`) builds a padded database.
3. Run `mmseqs easy-search` of the FASTA file against the database, with output `query,target,fident`. The fixed arguments are in `MMSEQS_SEARCH_ARGS`:

| Argument | Value | Meaning |
|----------|-------|---------|
| `--cov-mode` | `0` | Coverage of both query and target |
| `-c` | `0.8` | At least 80% coverage |
| `--max-seqs` | `1000000000` | No practical limit on hits |
| `-s` | `9.5` | High sensitivity |
| `--alignment-mode` | `3` | Alignment mode 3 |

4. Remove self hits (`query == target`).
5. Save the hits to `mmseqs_all_vs_all_fident.tsv`. Delete the MMseqs2 work folders.

### Step 3: Maximum identity
1. For each sequence, take the maximum `fident` over all hits where it is the query or the target.
2. A sequence with no hit gets `0.0`.
3. Save the result to `max_fident_similarities.tsv` (columns `sequence`, `max_similarity`).

### Step 4: Split for each threshold
1. Sort sequences by `fident`, then row count, then sequence. This makes the order independent of row order.
2. **Test**: take sequences with `max fident < threshold`, lowest first. Fill with `greedy_fill` up to `int(test_frac * n_rows)` rows. A sequence that does not fit is skipped, and the next one is tried.
3. **Val**: take from all sequences that are not in test, in the same order. Fill up to `int(valid_frac * n_rows)` rows. Val comes from all sequences, so it is in-distribution. Only test is below the threshold.
4. **Train**: all other sequences, and the rows with a null sequence.
5. Check that the three sets share no sequence.

### Step 5: Save
`save_threshold_splits` writes `threshold_<t>/{train,val,test}.parquet` (zstd, level 3). The splitter also writes two distribution plots and a JSON file with the resolved config, seed and git commit.

---

## Caches

Both TSV files have a `.fingerprint` file next to them. The fingerprint is a hash of the sorted sequences, `MMSEQS_SEARCH_ARGS`, GPU mode and the MMseqs2 version. If the fingerprint does not match, the cache is computed again.

Set `splits.similarity_cache_overwrite: true` to ignore the caches. A max-similarity cache that lacks some sequences raises `ValueError`.

---

## Configuration

Seed and fractions are in `configs/data/splits.yaml`. The `sequence_split` block is in the dataset files in `configs/data/splits_dataset/`. Its input path points to a dataset file listed in `configs/data/split_inputs/*.yaml` or in a builder yaml.

| Key | Default | Description |
|-----|---------|-------------|
| `splits.test_frac` | `0.05` | Maximum test rows as a fraction of all rows |
| `splits.valid_frac` | `0.05` | Val rows as a fraction of all rows |
| `splits.similarity_cache_overwrite` | `false` | Ignore the caches |
| `splits.sequence_col` | `sequence` | Sequence column |
| `splits.mmseqs_bin` | `${paths.mmseqs_bin}` | MMseqs2 binary (`scripts/install_external_tools.sh` installs it) |
| `splits.mmseqs_gpu` | `false` | Run the prefilter on the visible CUDA devices |
| `splits.dataset.sequence_split.input_dataset_parquet_file_path` | | Input parquet file |
| `splits.dataset.sequence_split.output_dir` | | Output folder. Caches and scratch folders are here |
| `splits.dataset.sequence_split.similarity_thresholds` | `[0.1 ... 0.9]` | Thresholds to run. This is the code default. Each dataset file sets its own list |
| `splits.dataset.sequence_split.unique_similarity_plot_path` | | Plot of the maximum identity per sequence |
| `splits.dataset.sequence_split.full_dataset_similarity_plot_path` | | Plot of the maximum identity per row |

---

## Notes

1. The dataset files in `configs/data/splits_dataset/` set the thresholds. See `overview.md` for how they were chosen.
2. `test_frac` is a maximum. A low threshold can give fewer test rows.
3. MMseqs2 finds only hits with 80% coverage. A sequence with no such hit has maximum identity 0 and is a test candidate.
