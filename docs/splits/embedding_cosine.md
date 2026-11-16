# Embedding Cosine Split

This document describes `EmbeddingCosineSimilaritySplitter` in `src/data/components/splitters/embedding_cosine.py`.

## Overview

The splitter uses pre-computed embeddings. For each unique key, it finds the highest cosine similarity to any other key. Keys with low maximum similarity go to test. The split is made once for each threshold.

The dataset files in `configs/data/splits_dataset/` use it for two cases:
- Molecule conformers: UniMol embeddings, key column `smiles_hash` (`conformer_cosine_splits`).
- Reactions: RXNFP embeddings, key column `rxn_smiles` (`reaction_cosine_splits`).

The splitter does not compute embeddings.

---

## Processing Pipeline

1. Read the input parquet file. The key column must exist.
2. Load one vector for each unique key from one source. The splitter checks the sources in this order:
   1. A dataframe column (`embedding_col`)
   2. A sharded embedding store (`embeddings_store_dir`, with `meta.json` and an index)
   3. An NPZ file (`embeddings_npz`, else `unimol_sdf_embeddings.npz` next to the input file or in its parent folder)
3. Find the embedding of each key with the lookup mode (`embedding_lookup_mode`):
   - `direct`: the embedding key is the dataset key.
   - `conformer` (default): the direct key, else `pubchem_<key>`, else the mean of `datamol_<key>_conf<i>` for `i < datamol_num_conformers` (default 10). A 2D embedding (several conformers) is mean-pooled.
   - `templates`: keys built from `embedding_key_templates`. `{key}` is the dataset key and `{i}` runs over the conformers.
4. Normalise the vectors and compute the maximum cosine similarity of each key to all other keys in blocks of `batch_size` (default 512). The self value is excluded.
5. Save the values to `max_cosine_similarities.tsv` in the output folder, with a `.fingerprint` file. The fingerprint hashes the keys, the embedding source, the lookup mode, the number of conformers and the templates. Set `splits.similarity_cache_overwrite: true` to compute the values again.
6. For each threshold:
   - Keys with no embedding or an all-zero embedding always go to train.
   - If the key column is `smiles_hash`, tiny or inorganic molecules (`is_tiny_or_inorganic`) also stay in train.
   - **Test**: remaining keys with `max_cosine < threshold`, sorted by `order_by_similarity`. `greedy_fill` fills up to `int(test_frac * n_rows)` rows. A key that does not fit is skipped.
   - **Val**: all other eligible keys that are not in test, in the same order. The target is `int(val_frac * n_rows)` rows. Val is in-distribution: only test is below the threshold. If `match_val_to_test` is true, the target is the test row count instead.
   - **Train**: all other keys.
7. Plot the distributions. Save `threshold_<t>/{train,val,test}.parquet` with `save_threshold_splits` (zstd, level 3).

---

## Configuration

The keys are in the `cosine_split` block of the dataset files in `configs/data/splits_dataset/`. The block runs only when `enabled` is not false.

| Key | Default | Description |
|-----|---------|-------------|
| `splits.dataset.cosine_split.input_dataset_parquet_file_path` | | Input parquet file |
| `splits.dataset.cosine_split.output_dir` | | Output folder |
| `splits.dataset.cosine_split.unique_similarity_plot_path` | | Plot for unique keys |
| `splits.dataset.cosine_split.full_dataset_similarity_plot_path` | | Plot for all rows |
| `splits.dataset.cosine_split.key_column` | `smiles_hash` | Column with the key |
| `splits.dataset.cosine_split.embedding_lookup_mode` | `conformer` | `direct`, `conformer` or `templates` |
| `splits.dataset.cosine_split.embeddings_store_dir` | none | Embedding store folder |
| `splits.dataset.cosine_split.embeddings_npz` | none | NPZ file |
| `splits.dataset.cosine_split.embedding_col` | none | Dataframe column |
| `splits.dataset.cosine_split.embedding_key_templates` | `[]` | Templates for the `templates` mode |
| `splits.dataset.cosine_split.datamol_num_conformers` | `embeddings.datamol_num_conformers`, else 10 | Conformers per molecule |
| `splits.dataset.cosine_split.similarity_thresholds` | `[0.5 ... 0.95]` | Thresholds to run. Dataset files set their own |
| `splits.dataset.cosine_split.test_frac` | `splits.test_frac` | Maximum test rows. Must be in (0, 0.5) |
| `splits.dataset.cosine_split.val_frac` | `splits.valid_frac` | Val rows. Must be in (0, 0.5) unless `match_val_to_test` is true |
| `splits.dataset.cosine_split.match_val_to_test` | `false` | Use the test row count as the val target |
| `splits.dataset.cosine_split.batch_size` | `512` | Rows per cosine block |
| `splits.similarity_cache_overwrite` | `false` | Ignore the cache |

---

## Notes

1. UniMol cosine values are close to 1.0. The thresholds in the dataset files for molecule conformers use a 0.001 grid.
2. `test_frac` is a maximum. A low threshold can give fewer test rows.
