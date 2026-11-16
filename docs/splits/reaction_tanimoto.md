# Reaction DRFP Tanimoto Split

This document describes `ReactionDRFPTanimotoSimilaritySplitter` in `src/data/components/splitters/reaction_tanimoto.py`.

## Overview

The splitter uses DRFP reaction fingerprints. For each unique reaction key, it finds the highest Tanimoto similarity to any other reaction. Reactions with low maximum similarity go to test. The split is made once for each threshold.

The DRFP vectors must already exist. This splitter does not compute them.

---

## Processing Pipeline

1. Read the input parquet file. The key column (`rxn_key` in the dataset files) must exist.
2. Load one binary DRFP vector for each unique key from one source. The splitter checks the sources in this order:
   1. A dataframe column (`tanimoto_embedding_col`)
   2. A sharded embedding store (`tanimoto_embeddings_store_dir`, with `meta.json` and an index)
   3. An NPZ file (`tanimoto_embeddings_npz`, else `drfp_embeddings.npz` next to the input file or in its parent folder)

   The splitter stops with an error if it finds no source.
3. Compute the maximum Tanimoto similarity of each vector to all other vectors. The splitter uses exact matrix products in blocks. The self value is set to 0.
4. Save the values to `max_tanimoto_similarities.tsv` (columns: key column, `max_tanimoto_similarity`) in the output folder, with a `.fingerprint` file. The fingerprint hashes the keys, the embedding source and the splitter name. Set `splits.similarity_cache_overwrite: true` to compute the values again.
5. For each threshold:
   - Keys with no vector or an all-zero vector always go to train.
   - **Test**: keys with `max_sim < threshold`, sorted by `order_by_similarity` (similarity, row count, key). `greedy_fill` fills up to `int(test_frac * n_rows)` rows. A key that does not fit is skipped, and the next one is tried.
   - **Val**: all other keys that have a vector and are not in test, in the same order. The target is `int(val_frac * n_rows)` rows. Val is in-distribution: only test is below the threshold. If `tanimoto_match_val_to_test` is true, the target is the test row count instead.
   - **Train**: all other keys.
6. Check that no key is in two splits and that all rows are assigned.
7. Plot the distributions. Save `threshold_<t>/{train,val,test}.parquet` with `save_threshold_splits` (zstd, level 3).

---

## Configuration

The keys are set in the dataset files in `configs/data/splits_dataset/` (for example `reaction_outcome_dataset.yaml` and `enzyme_retrieval.yaml`). Each key is read from `splits.dataset.<key>` first, then from `splits.<key>`. The splitter runs only if `tanimoto_split_input_dataset_parquet_file_path` is set.

| Key | Default | Description |
|-----|---------|-------------|
| `tanimoto_split_input_dataset_parquet_file_path` | | Input parquet file |
| `tanimoto_split_output_dir` | | Output folder |
| `unique_tanimoto_similarity_plot_path` | | Plot for unique keys |
| `full_dataset_tanimoto_similarity_plot_path` | | Plot for all rows |
| `tanimoto_key_column` | `rxn_smiles` | Column with the reaction key |
| `tanimoto_similarity_thresholds` | `[0.1 ... 0.5]` | Thresholds to run. Dataset files set their own |
| `tanimoto_test_frac` | `splits.test_frac` | Maximum test rows as a fraction of all rows |
| `tanimoto_val_frac` | `splits.valid_frac` | Val rows as a fraction of all rows |
| `tanimoto_match_val_to_test` | `false` | Use the test row count as the val target |
| `tanimoto_embedding_col` | none | Dataframe column with the vectors |
| `tanimoto_embeddings_store_dir` | none | Embedding store folder (`embeddings.drfp_embeddings_store_dir` in the dataset files) |
| `tanimoto_embeddings_npz` | none | NPZ file with the vectors |
| `splits.similarity_cache_overwrite` | `false` | Ignore the cache |

---

## Notes

1. The embeddings are DRFP bit vectors. The splitter treats every value above 0 as 1.
2. Unlike the SMILES splitter, this splitter has no tiny-molecule filter.
3. `test_frac` is a maximum. A low threshold can give fewer test rows.
