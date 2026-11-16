# Random Grouped Split

This document describes `RandomSplitter` in `src/data/components/splitters/random.py`.

## Overview

The splitter makes one train/val/test split for each column in `unique_cols`. All rows with the same group value go to the same split, so no group appears in two splits. Group sizes decide the row fractions.

---

## Processing Pipeline

For each column in `unique_cols`:

1. Read the input parquet file. Raise `KeyError` if the column is missing.
2. Build one group value for each row:
   - `smiles`: the `molecule_key` of the SMILES. Salt, charge and tautomer forms share a key. Stereo stays. If the key fails, the raw SMILES is used.
   - `rxn_key`: `undirected_reaction_key`. A reaction and its reverse reaction share one group.
   - Other columns: the value as a stripped string. Missing or empty values become `__MISSING__`. A list or dict value raises `ValueError`.
3. Count rows for each group. Raise `ValueError` if there are fewer than 3 groups.
4. Shuffle the groups with `random_state=seed`. The group index is sorted first, so row order does not change the result.
5. Assign each group, one at a time, to the split with the lowest cost. The cost is the sum of squared differences between the split row counts (after adding the group) and the target row counts (`frac * total_rows`). Equal costs are broken at random with the same seed.
6. If a split is still empty and as many groups remain as empty splits, only the empty splits can take the group. This keeps all three splits non-empty.
7. Check that every group has one split and that no group is in two splits.
8. Save the splits.

---

## Configuration

Seed and fractions come from `configs/data/splits.yaml`. Paths and columns come from the `random` block in the dataset files in `configs/data/splits_dataset/`.

| Key | Default | Description |
|-----|---------|-------------|
| `splits.random_seed` | `42` | Seed for the shuffle and for tie breaks |
| `splits.train_frac` | `0.9` | Train fraction |
| `splits.valid_frac` | `0.05` | Validation fraction |
| `splits.test_frac` | `0.05` | Test fraction |
| `splits.random_unique_cols` | `[smiles, sequence]` | Columns to group by |
| `splits.dataset.random.unique_cols` | `${splits.random_unique_cols}` | Group columns the splitter reads |
| `splits.dataset.random.input_dataset_parquet_file_path` | the sequence split input | Input parquet file |
| `splits.dataset.random.output_dir` | `<dataset path>/random_splits` | Only the parent folder is used (see Output) |
| `splits.log_dir`, `splits.log_file_name` | | Log location |

The three fractions must sum to 1 (`ValueError` otherwise).

---

## Output

For each column, files are saved to a sibling of `output_dir`:

```
<parent of output_dir>/random_splits_grouped_<column>/
  train.parquet
  val.parquet
  test.parquet
  split_stats.json     # only when the data has a numeric log10_value column
```

The folder also holds a JSON file with the resolved config, the seed and the git commit. Files use zstd compression (level 3).

`generate_splits()` returns the splits for one column, or a dict keyed by column when there are several.

---

## Notes

1. The split is random but repeatable. The same input, seed and fractions give the same split.
2. The row fractions are close to the targets, not exact. Large groups cause the difference.
3. Only the group columns are checked for leakage. Similar but different sequences or molecules can be in different splits.
