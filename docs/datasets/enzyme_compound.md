# Enzyme Compound Dataset

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes `EnzymeCompoundDatasetBuilder` in `src/data/components/enzyme_compound_dataset.py`.

## Overview

The builder makes an enzyme-substrate pair dataset from the unified 3D reactions file. It writes:
- Positive pairs: a substrate that a reaction of the enzyme uses (`label = 1`).
- Negative pairs: a substrate that is similar to a positive substrate, but is not known for that enzyme sequence (`label = 0`).

`builder.setup()` runs the whole pipeline.

---

## Processing Pipeline

### Step 1: Load reactions (`setup`)
Read these columns from `unified_reactions_3d_parquet_file_path`: `ec`, `rxn_smiles`, `uniprot_id`, `sequence`, `source`, `rhea_id`, `kegg_id`, `metacyc_id`, `sabio_id`, `pdbs`, `pdb_source`, `pdb_type`.

### Step 2: Make substrate-enzyme rows (`make_substrate_enzyme_df`)
1. `extract_substrates` takes the left side of `rxn_smiles` (before `>>`) and splits it on `.`.
2. Explode so that each substrate is one row. Drop rows with no SMILES. Rename `ec` to `enzyme`.
3. Explode `sabio_id`.

### Step 3: Aggregate
Group by (`smiles`, `uniprot_id`, `sequence`, `pdbs`). For `enzyme`, `source`, `rhea_id`, `kegg_id`, `metacyc_id` and `sabio_id`, collect the values in a set. For `pdb_source` and `pdb_type`, take the first value.

### Step 4: Canonicalise
Canonicalise the unique SMILES with `canonicalize_smiles` in parallel (pandarallel). Drop pairs with an invalid SMILES. Drop pairs with no `sequence`.

### Step 5: Positive pairs (`build_classification_dataset`)
1. Set `label = 1` and add the `uniprot_date` column.
2. Save the pairs to `enzyme_substrate_positive_dataset.parquet`.

### Step 6: Negative pairs (`build_negative_pairs_pandarallel`)
1. Compute a Morgan fingerprint (radius 2, 1024 bits) for each unique substrate.
2. For each unique (`sequence`, `smiles`) pair, `get_negatives_for_row` builds a pool of negative candidates:
   - Candidates are all substrates in the dataset that are not a positive substrate of that sequence.
   - The score is the Tanimoto similarity of the candidate to the positive substrate of the row.
   - The pool holds candidates with `sim_lower <= similarity <= sim_upper`.
   - If the pool has fewer than `n_neg` candidates, the lower bound goes down by `sim_step` and the pool is built again. When the lower bound reaches 0, the pool holds all candidates with `similarity <= sim_upper`.
3. For each positive row, `n_neg` negatives are sampled from the pool of the row with `random.sample`. A candidate substrate can be a negative at most `n_neg * (number of positive rows of that substrate)` times.
4. Merge `uniprot_date`, `pdbs`, `pdb_source` and `pdb_type` back by (`uniprot_id`, `sequence`).
5. Save the pairs to `enzyme_substrate_negative_dataset.parquet`.

### Step 7: Final dataset
1. Combine positive and negative pairs. Drop duplicate rows and rows with a missing key field. Sort by `sequence`, `label`, `smiles`.
2. Save `enzyme_substrate_dataset.parquet` with the columns `uniprot_id`, `sequence`, `uniprot_date`, `smiles`, `label`, `pdbs`, `pdb_source`, `pdb_type`.
3. Save `enzyme_substrate_sequences.pkl` (list of `acc_id`, `sequence` records) and `enzyme_substrate_smiles.pkl` (list of `smiles` records). The embedding steps read these files.

All parquet files use zstd compression (level 3).

---

## Configuration

The builder keys are in `configs/data/enzyme_compound_dataset.yaml`. The path keys that the split configs read (`enzyme_compound_dataset_path`, `enzyme_substrate_dataset_parquet_file_path`) are in `configs/data/split_inputs/enzyme_compound_dataset.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `log_dir`, `log_file_name` | `${paths.log_dir}`, `enzyme_compound_dataset.log` | Log location |
| `unified_reactions_3d_parquet_file_path` | from `reactions_dataset` | Input 3D unified reactions |
| `unified_reactions_sequences_pickle_path`, `unified_reactions_rxn_smiles_pickle_path` | from `reactions_dataset` | Reaction pickles from the reactions builder |
| `enzyme_substrate_positive_dataset_parquet_file_path` | `<dataset path>/enzyme_substrate_positive_dataset.parquet` | Positive pairs |
| `enzyme_substrate_negative_dataset_parquet_file_path` | `<dataset path>/enzyme_substrate_negative_dataset.parquet` | Negative pairs |
| `enzyme_substrate_dataset_parquet_file_path` | `<dataset path>/enzyme_substrate_dataset.parquet` | Final dataset |
| `enzyme_substrate_sequences_pickle_path`, `enzyme_substrate_smiles_pickle_path` | `<dataset path>/enzyme_substrate_{sequences,smiles}.pkl` | Sequence and SMILES lists |
| `random_seed` | `42` | Seed for `random` and `numpy` |
| `pandarallel.nb_workers` | `112` | Workers (the code uses `min(nb_workers, cpu_count())`) |
| `pandarallel.progress_bar` | `false` | Progress bar for canonicalisation |
| `pandarallel.progress_bar_negative_pairs` | `true` | Progress bar for the negative pool |
| `n_neg` | `3` | Negatives for each positive row |
| `sim_upper` | `0.95` | Maximum similarity of a negative to the positive substrate |
| `sim_lower` | `0.7` | Starting minimum similarity. It goes down by `sim_step` when the pool is too small |
| `sim_step` | `0.2` | Step for the lower bound |

---

## Output schema

| Column | Type | Description |
|--------|------|-------------|
| `uniprot_id` | str | UniProt accession of the enzyme |
| `sequence` | str | Amino acid sequence |
| `uniprot_date` | date | UniProt creation date of the accession |
| `smiles` | str | Canonical SMILES of the substrate |
| `label` | int | 1 for a positive pair, 0 for a negative pair |
| `pdbs` | str | Structure key (`<PDB>:<base accession>` for experimental structures, else the accession) |
| `pdb_source`, `pdb_type` | str | Source and type of the structure |

The positive file also has `enzyme`, `source`, `rhea_id`, `kegg_id`, `metacyc_id` and `sabio_id` (as sets).

---

## Notes

1. Negatives are close to a known substrate (similarity from 0.7 up to 0.95 at first). They are hard negatives, not random molecules.
2. A negative is not a proven non-substrate. It is a molecule with no known reaction for that sequence.
3. The splits use `configs/data/splits_dataset/enzyme_substrate.yaml`. This dataset is not part of the default `splits.datasets` list.
