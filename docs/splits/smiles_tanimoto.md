# SMILES Tanimoto Similarity Split

This document describes `SMILESMaxTanimotoSimilaritySplitter` in `src/data/components/splitters/smiles_tanimoto.py`.

## Overview

The splitter makes train/val/test splits from the chemical dissimilarity of molecules. For each unique molecule, it finds the highest Tanimoto similarity to any other molecule. Molecules with low maximum similarity go to test. The split is made once for each threshold in `similarity_thresholds`.

All rows of one molecule stay in one split. Row counts, not molecule counts, set the sizes.

---

## Processing Pipeline

### Step 1: Load and clean (`_clean_full_dataset_df`)
1. Take the unique `smiles` values. Canonicalise them in parallel with pandarallel.
2. Compute `smiles_key` for each SMILES with `molecule_key`. The key joins salt, charge and tautomer forms of one compound. Stereo stays. If the key fails, the canonical SMILES is the key.
3. Drop rows whose SMILES is invalid.
4. In the data, replace `smiles` with the canonical SMILES. The `smiles_key` column is rebuilt.
5. Keep one row for each `smiles_key` in the unique molecule table.

All grouping, fingerprints and row counts use `smiles_key`.

### Step 2: Fingerprints (`_smiles_to_fps`)
Morgan fingerprint, radius 2, 1024 bits (`_FP_RADIUS`, `_FP_BITS`), one for each unique `smiles_key`. A multiprocessing pool builds them.

### Step 3: Maximum similarity (`compute_max_similarities`)
1. For each molecule, `BulkTanimotoSimilarity` against all fingerprints. The self value is set to 0. The maximum is stored.
2. Save the values to `max_tanimoto_similarities.tsv` (columns `smiles_key`, `max_similarity`) in the output folder, with a `.fingerprint` file. The fingerprint hashes the keys and the fingerprint settings.
3. A later run reads the file if the fingerprint matches. Set `splits.similarity_cache_overwrite: true` to compute it again.

### Step 4: Split for one threshold (`dissimilarity_split`)
1. `max_test_rows = int(test_frac * n_rows)`. `target_val_rows = int(valid_frac * n_rows)`.
2. Tiny or inorganic molecules never go to test or val. `is_tiny_or_inorganic` is true for molecules with at most 3 heavy atoms or no carbon (water, ammonia, ions). They stay in train.
3. **Test**: eligible molecules with `max_sim < threshold`. `order_by_similarity` sorts them by similarity, then row count, then key. `greedy_fill` takes them in that order. A molecule that does not fit is skipped, and the next one is tried.
4. **Val**: all eligible molecules that are not in test, in the same order. `greedy_fill` fills up to `target_val_rows`. Val is in-distribution: only test is below the threshold.
5. **Train**: all other molecules.

### Step 5: Plots and save
1. `plot_max_tanimoto_distribution` plots the unique-molecule histograms. `_plot_full_test_set_tanimoto_distribution` plots the row-weighted histograms.
2. `BaseSplitter.save_threshold_splits` writes `threshold_<t>/{train,val,test}.parquet` with zstd compression (level 3). It also writes a JSON file with the resolved config, seed and git commit.

---

## Configuration

Seed and fractions are in `configs/data/splits.yaml`. The `smiles_split` block is in the dataset files in `configs/data/splits_dataset/`. Its input path points to a dataset file listed in `configs/data/split_inputs/*.yaml` or in a builder yaml.

| Key | Default | Description |
|-----|---------|-------------|
| `splits.test_frac` | `0.05` | Maximum test rows as a fraction of all rows |
| `splits.valid_frac` | `0.05` | Val rows as a fraction of all rows |
| `splits.similarity_cache_overwrite` | `false` | Ignore the similarity cache |
| `splits.pandarallel.nb_workers` | `112` | Workers for canonicalisation. The code uses `min(nb_workers, cpu_count())` |
| `splits.pandarallel.progress_bar` | `false` | Show the pandarallel progress bar |
| `splits.dataset.smiles_split.input_dataset_parquet_file_path` | | Input parquet file with a `smiles` column |
| `splits.dataset.smiles_split.output_dir` | | Output folder for `threshold_*` folders and the cache |
| `splits.dataset.smiles_split.similarity_thresholds` | `[0.1 ... 0.9]` | Thresholds to run. This is the code default. Each dataset file sets its own list |
| `splits.dataset.smiles_split.unique_similarity_plot_path` | | Plot for unique molecules |
| `splits.dataset.smiles_split.full_dataset_similarity_plot_path` | | Plot for all rows |
| `splits.log_dir`, `splits.log_file_name` | | Log location |

---

## Output

```
<output_dir>/
  threshold_0.1/{train,val,test}.parquet
  threshold_0.2/...
  max_tanimoto_similarities.tsv
```

The output files keep all input columns. `smiles` holds the canonical SMILES and `smiles_key` holds the grouping key.

---

## Notes

1. Rows with the same `smiles_key` always stay in one split.
2. `test_frac` is a maximum. A low threshold can give fewer test rows.
3. Test molecules are only dissimilar to the whole dataset, so they are dissimilar to train too.
