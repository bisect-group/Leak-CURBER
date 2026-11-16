# Protein LDDT-Based Splitting

This document describes the structure-based dataset splitting using LDDT scores implemented in [src/data/components/splitters/protein_struct_lddt.py](../../src/data/components/splitters/protein_struct_lddt.py).

## Overview

`ProteinStructMaxLDDTSimilaritySplitter` creates train/validation/test splits based on protein structure dissimilarity. It:

1. Prepares protein structures from multiple sources (experimental mmCIF chains, AlphaFold, ESMFold2, ESM3)
2. Uses Foldseek `easy-search` for rapid all-vs-all structure comparison
3. Computes the **maximum** LDDT similarity of each protein to all other proteins in the dataset
4. Generates splits across multiple LDDT thresholds — proteins with `max_lddt < threshold` are assigned to test/val first
5. Saves separate parquet split files for every threshold

Row counts (not protein counts) are used to enforce the test/val size budget, so the target fraction applies to dataset rows.

Rows with `NaN` PDB IDs cannot be structurally split and are always assigned to train. Sequences with no usable structure are also forced to train (see Step 5).

---

## Processing Pipeline

### Step 1: Prepare PDB Symlinks (`_generate_pdb_paths`)
1. Classify proteins by `pdb_type` column: `"experimental"` or any other value (predicted)
   - A structure is `<PDB>:<base UniProt accession>` for experimental rows (the chain in `pdb_chain`; a mutant ID `P1|R2Q` has the base accession `P1`) and the accession for the others
2. `locate_structures` finds the file of each structure. A predicted protein uses the first source that has a file:
   1. AlphaFold: the highest-version `AF-<acc>-*F1-model_v*.pdb` file in `af_pdb_path/`
   2. ESMFold2: `{esmfold2_pdb_path}/ESMFold2-{acc}.pdb`
   3. ESM3: `{esm_pdb_path}/ESM3-open-small-{acc}.pdb`
3. Create symlinks under `pdb_symlinks/` (in `<output_dir>`, or in `shared_search_dir` for a shared search):
   - Experimental: the mmCIF file of the entry in `mmcif_path`, linked once per entry as `{pdb_id}.cif`
   - Predicted: the file found in item 2
4. Log missing counts for experimental and predicted structures

### Step 2: Build Foldseek Database (`_foldseek_create_database`)
1. Run `foldseek createdb --chain-name-mode 1` on the symlink directory → `<output_dir>/foldseek_work/foldseekDB`
2. Keep the keys of the selected chains from `foldseekDB.lookup` (the `pdb_chain` of each experimental structure, chain `A` of each predicted one) in `selected_keys.tsv`, then run `foldseek createsubdb` → `selectedDB`. The search runs on these chains only. Hits on other chains are dropped anyway, so searching them would only waste hits. The run fails if no chain is selected
3. Run `foldseek createindex` on `selectedDB` using `<output_dir>/foldseek_tmp` as scratch space (cleaned up on completion). In GPU mode (`foldseek_gpu: true`), run `foldseek makepaddedseqdb` → `selectedDB_pad` instead

### Step 3: All-vs-All Comparison (`_foldseek_all_vs_all`)
1. Run `foldseek easy-search` with `selectedDB` as query and target (`selectedDB_pad` as target in GPU mode) and these parameters:
   - `--cov-mode 2` — coverage of the shorter chain, so a partial chain still hits full-length structures
   - `-c 0.8` — minimum 80% alignment coverage
   - `-s 9.5` — high sensitivity
   - `--alignment-mode 3` — global alignment
   - `--alignment-type 2` — 3Di+AA alignment
   - `--max-seqs 1000000000` — no result limit
   - `--chain-name-mode 1` — every hit is named `<file>_<chain>`
   - `--format-output query,target,lddt`
2. Monitor progress by polling tmpdir + output file sizes
3. Read output TSV with `lddt` parsed as string then coerced to `float64` via `pd.to_numeric` (handles scientific notation such as `2.190E-01`)
4. Map hit names to structure ids and remove self-hits. `<PDB>_<chain>` maps to `<PDB>:<base UniProt accession>` only for the chain in `pdb_chain`; hits on other chains or files are dropped. An accession hit may carry the `.pdb` extension and the `_A` chain
5. Save results to `foldseek_all_vs_all_lddt.tsv` (in `<output_dir>`, or in `shared_search_dir` for a shared search) with a `.fingerprint` file. A later run reuses it when the fingerprint matches
6. Clean up the database directories

### Step 4: Compute Max LDDT per Protein (`compute_max_lddt_to_dataset`)
1. Keep only hits between structures of the dataset.
2. Drop hits whose query and target share any dataset sequence (`_drop_same_sequence_hits`). The same sequence in two structures is not a similar protein. This rule always applies.
3. Take the maximum `lddt` over hits where the structure is the query or the target. Batches of 5 million hits keep memory low.
4. Build a NumPy array in `unique_structures` order. A structure with a file but no hit gets `0.0`. A structure with no file gets `NaN`.
5. Save the array to `<output_dir>/max_lddt_similarities.tsv` (columns `structure_id`, `max_similarity`) with a `.fingerprint` file

### Step 5: Split Across Thresholds (`run_splits_across_thresholds` → `dissimilarity_based_split`)
Rows are grouped by sequence. Each sequence takes the highest `lddt` of its structures.

Some sequences are unknown, not dissimilar. They always go to train and never to val or test:
- The sequence has no structure file.
- The sequence has only predicted structures with mean pLDDT below `min_mean_plddt` (default 70). The Foldseek hits of such a structure still count for other proteins.
- The sequence has a structure but no Foldseek hit (`max_lddt == 0.0`).

For each threshold value:
1. Find known sequences with `max_lddt < threshold` ("dissimilar" sequences)
2. Sort by max LDDT, then row count, then sequence (most dissimilar first)
3. **Test**: `greedy_fill` adds sequences until the row count reaches `int(test_frac * n_rows)`. A sequence that does not fit is skipped, and the next one is tried
4. **Val**: take from all known sequences that are not in test, in the same order. Fill up to `int(valid_frac * n_rows)` rows. Val is in-distribution: only test is below the threshold
5. **Train**: all remaining sequences, the unknown sequences and the rows with a null sequence
6. Assert no overlap across splits

### Step 6: Visualize and Save (`generate_splits`)
1. `plot_lddt_distribution` — one subplot per threshold, histograms of val/test max LDDT for unique proteins
2. `plot_full_dataset_lddt_distribution` — same but weighted by row count
3. Save each split as `OUTPUT_DIR/threshold_{t}/{train,val,test}.parquet` (zstd level 3 compressed). A JSON file with the resolved config, seed and git commit is saved in `OUTPUT_DIR`

---

## Configuration

Seed and fractions are in `configs/data/splits.yaml`. Each dataset file in `configs/data/splits_dataset/` sets the `structure_split` block. Its input paths point to dataset files listed in `configs/data/split_inputs/*.yaml` or in a builder yaml.

| Key | Type | Description |
|-----|------|-------------|
| `splits.test_frac` | float | Maximum test rows as a fraction of all rows (`0.05` in `splits.yaml`) |
| `splits.valid_frac` | float | Val rows as a fraction of all rows (`0.05`) |
| `splits.similarity_cache_overwrite` | bool | Default `false`. If `true`, ignore the caches |
| `splits.af_pdb_path` | str | Directory with AlphaFold PDB files (`AF-*-F1-model_v*.pdb`) |
| `splits.esmfold2_pdb_path` | str | Directory with ESMFold2 PDB files (`ESMFold2-{acc}.pdb`) |
| `splits.esm_pdb_path` | str | Directory with ESM3 PDB files (`ESM3-open-small-{acc}.pdb`) |
| `splits.mmcif_path` | str | Directory with experimental mmCIF files |
| `splits.foldseek_bin` | str | Path to the Foldseek binary (`paths.foldseek_bin`) |
| `splits.foldseek_gpu` | bool | Default `false`. If `true`, build a padded database and search with `--gpu 1` on the visible CUDA devices (`CUDA_VISIBLE_DEVICES`) |
| `splits.log_dir`, `splits.log_file_name` | str | Log location |
| `splits.dataset.structure_split.input_dataset_parquet_file_path` | str | Input 3D dataset parquet |
| `splits.dataset.structure_split.output_dir` | str | Root output directory for split parquets and caches |
| `splits.dataset.structure_split.similarity_thresholds` | list[float] | LDDT thresholds to run (default `[0.1, …, 0.9]` if not set) |
| `splits.dataset.structure_split.unique_similarity_plot_path` | str | Plot of the unique-protein LDDT histograms |
| `splits.dataset.structure_split.full_dataset_similarity_plot_path` | str | Plot of the full-dataset LDDT histograms |
| `splits.dataset.structure_split.min_mean_plddt` | float | Default `70`. A predicted structure (AlphaFold, ESMFold2, ESM3) with a lower mean CA pLDDT counts as unknown. A structure without CA atoms, with a constant B-factor, or with an unreadable file also counts as unknown. Mean pLDDT values are cached in `<output_dir>/predicted_mean_plddt.tsv` |
| `splits.dataset.structure_split.drop_to_1d` | bool | Default `false`. Save 1D rows (no structure columns) in the splits |
| `splits.dataset.structure_split.drop_to_1d_dataset_parquet_file_path` | str | Optional 1D parquet that the 1D splits are made from. It must exist when `drop_to_1d` is set |
| `splits.dataset.structure_split.search_query_chunk_size` | int | Default `None`: one `easy-search` over all selected chains. If set and smaller than the number of selected chains, the selected chains are searched as queries in chunks of this size against the whole selected database. Each chunk is a `createsubdb` subset of `foldseekDB` (this copies `_ss`, `_ca` and `_h`), so the hits are the same and the cache fingerprint ignores it. Each chunk deletes its temp files, so the prefilter temp disk is bounded per chunk. A crashed run restarts the search from chunk 1 |
| `splits.dataset.structure_split.shared_search_inputs` | list[str] | Optional. 3D parquets whose structures form one shared search set (enzyme classification and retrieval list both). The createdb, the all-vs-all search, its TSV and fingerprint use the union and live in `shared_search_dir`; the first dataset to run computes them and later datasets hit the cache. Each dataset then keeps only the hits between its own structures and drops same-sequence hits by its own sequences. The max-lDDT and pLDDT caches and the splits stay in `<output_dir>`. Run the datasets one after another. When unset, the search covers only the dataset's own structures in `<output_dir>` |
| `splits.dataset.structure_split.shared_search_dir` | str | Required with `shared_search_inputs`. Directory for the shared Foldseek scratch folders, `foldseek_all_vs_all_lddt.tsv` and its fingerprint |

Foldseek uses `<output_dir>/foldseek_tmp` as scratch space. The folder is deleted after use.

---

## Methods

### `ProteinStructMaxLDDTSimilaritySplitter`

#### `__init__(self, cfg: DictConfig)`
- Reads all config paths and parameters
- Creates output/log directories
- Checks that `foldseek_bin` is an executable file, reads its version and logs the binary, version and mode
- Loads the full dataset parquet and adds a `structure_id` column and extracts `unique_structures`
- Initialises lazy-computed cache attributes (`_pdb_paths`, `_max_lddt_to_dataset`, `_max_lddt_array`)

#### `_create_pdb_symlink(self, structure_id, source_path, pdb_paths) → None`
Creates a symlink `{ALL_PDB_SYMLINK_PATH}/{pdb_id}.cif` (mmCIF, shared by all accessions of the entry) or `{structure_id}.pdb`, and records it in `pdb_paths`.

#### `_generate_pdb_paths(self, df) → dict`
Builds PDB symlinks for all proteins in `df`. Returns `{structure_id: symlink_path}` for every structure found on disk.

#### `_foldseek_create_database(self) → None`
Runs `foldseek createdb` on the symlink directory, `foldseek createsubdb` for the selected chains, then `foldseek createindex` (`makepaddedseqdb` in GPU mode) on the subset. Returns the number of selected chains. Deletes the tmpdir when it ends.

#### `_foldseek_all_vs_all(self) → pl.LazyFrame`
Runs `foldseek easy-search` (or reads the cached TSV), reads results as `(query, target, lddt)`, coerces `lddt` to `float64`, removes self-hits, saves the TSV, and returns a lazy frame.

#### `compute_max_lddt_to_dataset(self) → np.ndarray`
Computes and caches `max_lddt` per protein over the hits, without same-sequence hits. Returns an array aligned to `self.unique_structures`. Subsequent calls return the cached array.

#### `get_max_lddt_for_proteins(self, protein_ids) → dict`
Looks up `{protein_id: max_lddt}` from the cached dictionary. Raises `RuntimeError` if called before `compute_max_lddt_to_dataset`.

#### `dissimilarity_based_split(self, similarity_threshold) → tuple[dict, dict]`
Produces one set of train/val/test splits for a given threshold. Returns `(splits_dict, similarities_dict)` where `splits_dict` maps split name → DataFrame and `similarities_dict` maps `"test"`/`"val"` → `{pdb_id: max_lddt}`.

#### `run_splits_across_thresholds(self) → tuple[dict, dict]`
Calls `compute_max_lddt_to_dataset` once, then calls `dissimilarity_based_split` for each value in `self.THRESHOLDS`. Returns `(all_splits, all_similarities)` keyed by threshold.

#### `plot_lddt_distribution(self, all_similarities) → None`
Saves a multi-panel histogram (`UNIQUE_STRUCTURE_SIMILARITY_PLOT_PATH`) showing the max LDDT distribution of unique val/test proteins per threshold.

#### `plot_full_dataset_lddt_distribution(self, all_splits, all_similarities) → None`
Saves a multi-panel histogram (`FULL_DATASET_STRUCTURE_SIMILARITY_PLOT_PATH`) showing the same distributions weighted by row count.

#### `generate_splits(self) → tuple[dict, dict]`
Top-level entry point. Runs threshold sweep, generates both plots, and saves parquet files to `OUTPUT_DIR/threshold_{t}/{train,val,test}.parquet`.

---

## Data Schema

### Input Dataset (parquet)
| Column | Type | Description |
|--------|------|-------------|
| `pdbs` | str or NaN | structure key: `<PDB>:<base accession>` for experimental rows, else the UniProt accession |
| `uniprot_id` | str | UniProt accession of the row |
| `pdb_chain` | str or NaN | mmCIF chain selected for `pdbs`; NaN unless `pdb_type` is `"experimental"` |
| `pdb_coverage` | float or NaN | Fraction of the UniProt sequence the selected chain covers; NaN unless `pdb_type` is `"experimental"` |
| `pdb_type` | str or NaN | `"experimental"` or another value for predicted structures |
| `pdb_source` | str | Source of the structure |
| `sequence` | str | Protein sequence. Splits are made on this column |
| ... | ... | Any other columns — preserved unchanged in output |

Rows where `pdbs` is NaN cannot be split structurally and are always assigned to train.

### Output Split Files
```
OUTPUT_DIR/
  threshold_0.1/
    train.parquet
    val.parquet
    test.parquet
  threshold_0.2/
    ...
  max_lddt_similarities.tsv ← max LDDT per structure (cache)
  predicted_mean_plddt.tsv  ← mean pLDDT per predicted structure (cache)
  foldseek_all_vs_all_lddt.tsv ← raw Foldseek output (cache; in shared_search_dir for a shared search)
```

### Foldseek TSV (Intermediate)
| query | target | lddt |
|-------|--------|------|
| 4HHB:P69905 | P67890 | 0.821 |
| 4HHB:P69905 | Q98765 | 2.190E-01 |

`lddt` is in [0, 1]; values may be in scientific notation.

---

## LDDT (Local Distance Difference Test)

LDDT measures structural similarity based on local distance patterns:

$$\text{LDDT} = \frac{1}{L} \sum_{i=1}^{L} \frac{1}{|\text{neighbors}(i)|} \sum_{j \in \text{neighbors}(i)} \mathbb{1}[|\Delta d_{ij}| < \tau]$$

Where:
- $L$: number of residues
- $\Delta d_{ij}$: difference in distance between residues $i, j$ in query vs target
- $\tau$: distance threshold (typically 4 Å)
- Score range: **0 to 1** (1 = identical structure). Foldseek outputs values already in this range — no normalisation is needed.

---

## Foldseek Integration

### Command Used
```bash
# Stage 1: create database from the structure symlink directory
foldseek createdb {output_dir}/pdb_symlinks {output_dir}/foldseek_work/foldseekDB --chain-name-mode 1
foldseek createsubdb {output_dir}/foldseek_work/selected_keys.tsv \
    {output_dir}/foldseek_work/foldseekDB {output_dir}/foldseek_work/selectedDB
foldseek createindex {output_dir}/foldseek_work/selectedDB {output_dir}/foldseek_tmp
# GPU mode: foldseek makepaddedseqdb {output_dir}/foldseek_work/selectedDB {output_dir}/foldseek_work/selectedDB_pad

# Stage 2: all-vs-all search
foldseek easy-search {output_dir}/foldseek_work/selectedDB {output_dir}/foldseek_work/selectedDB \
    {output_dir}/foldseek_work/all_vs_all.tsv {output_dir}/foldseek_tmp \
    --cov-mode 2 -c 0.8 --max-seqs 1000000000 \
    --format-output query,target,lddt \
    -s 9.5 --alignment-mode 3 --alignment-type 2 --chain-name-mode 1
# GPU mode: use selectedDB_pad as target and add --gpu 1
```

`foldseek` stands for the binary in `foldseek_bin`. Install it with `bash scripts/install_external_tools.sh`.

### Chain selection
The dataset builder chooses the chain of a PDB entry for each UniProt accession (`src/utils/structure_utils.py`):
1. Candidates are the chains that `_struct_ref_seq` maps to the accession (the isoform root matches). If none match, all protein chains are candidates.
2. A candidate is a match when, aligned to the UniProt canonical sequence, identity is at least 90%. A mutant ID such as `P1|R2Q` is compared through its base accession `P1` and the wild-type sequence.
3. Per (PDB, accession), the matching chain with the highest coverage, then identity, then chain ID order is selected. Its coverage must be at least 30%.
4. Per data row, if any PDB has a selected chain with coverage of at least 80%, the row keeps every such PDB (one row each). Otherwise it keeps the single PDB with the highest coverage.
5. A row without a selected chain gets exactly one fallback row: a predicted structure instead (AlphaFold, ESMFold2 or ESM3).

Results are cached in `{mmcif_path}/chain_selection_id90_cov30.json`. Each entry holds the chain, the basis and the coverage.

---

## Memory Optimization

Only the **max LDDT per protein** is stored, not the full N×N pairwise matrix.

| Approach | Memory (100k proteins) |
|----------|------------------------|
| Full pairwise matrix | ~40 GB |
| Max per protein (this impl.) | ~400 KB |

---

## Split Strategy

**Rationale**: evaluate the model on structurally novel proteins to test generalisation.

- **Test**: known sequences with the **lowest** `max_lddt` below the threshold, up to `test_frac` of the rows
- **Val**: from all known sequences not in test, lowest `max_lddt` first, up to `valid_frac` of the rows. Val is in-distribution
- **Train**: all remaining sequences, the unknown sequences (no structure, low-confidence predicted structure only, no Foldseek hit) and all NaN-pdb rows

The budget is enforced in **row counts** (not protein counts), so that proteins appearing in many rows do not disproportionately inflate a split.

---

## Performance Notes

1. **Foldseek runtime**: dominant cost; ~70 minutes for ~90k structures in testing.
2. **Memory**: O(N) after Foldseek results are aggregated.
3. **Disk**: symlinks avoid duplicating structure files; the Foldseek DB and tmpdir are cleaned up automatically.
4. **Thresholds**: max LDDT is computed once; all threshold sweeps reuse the cache.

---

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| `foldseek_bin` missing or not executable | `EnvironmentError` with the path and the install hint, raised in `__init__` |
| Input parquet not found | `FileNotFoundError` raised in `__init__` |
| Structure file missing for a protein | Warning is logged; `max_lddt` is `NaN` and the sequence goes to train |
| Foldseek non-zero exit code | `RuntimeError` raised with stderr |
| Unparseable lddt value | Coerced to `NaN` via `pd.to_numeric(errors='coerce')`; excluded from `groupby max` |
| No Foldseek output file | Empty result; all proteins with a structure get `max_lddt = 0.0` and go to train |
