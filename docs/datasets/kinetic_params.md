# Kinetic Parameters Dataset Generation

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document provides a comprehensive description of the kinetic parameters dataset generation pipeline implemented in `src/data/components/kinetic_params_dataset.py`.

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Configuration](#configuration)
4. [Pipeline Steps](#pipeline-steps)
   - [Step 1: Read Kinetic Law Data](#step-1-read-kinetic-law-data)
   - [Step 2: Unify BRENDA and SABIO](#step-2-unify-brenda-and-sabio)
   - [Step 3: Unify with Metacyc](#step-3-unify-with-metacyc)
   - [Step 4: Unify with BindingDB](#step-4-unify-with-bindingdb)
   - [Step 5: Validate SMILES](#step-5-validate-smiles)
   - [Step 6: Aggregate by Kinetic Type](#step-6-aggregate-by-kinetic-type)
   - [Step 7: Final Cleaning](#step-7-final-cleaning)
   - [Step 8: Save Outputs](#step-8-save-outputs)
5. [Output Files](#output-files)
6. [Data Schema](#data-schema)

---

## Overview

The `KineticParamsDatasetBuilder` class processes kinetic parameter data from multiple sources (BRENDA, SABIO, Metacyc, and BindingDB) to create a unified kinetic parameter dataset containing:

- **kcat** (Catalytic rate constant / turnover number)
- **Km** (Michaelis constant / substrate affinity)
- **Ki** (Inhibition constant)

The pipeline unifies data from multiple databases, handles enzyme mutations, canonicalizes SMILES strings, fetches protein sequences from UniProt, and aggregates duplicate measurements.

---

## Data Sources

The pipeline integrates kinetic parameter data from four major sources:

### 1. SABIO-RK Database
- **Data Includes**: Kcat, Km, Ki
- **Input Format**: Pre-processed parquet file

### 2. BRENDA Database
- **Data Includes**: Kcat, Km, Ki
- **Input Format**: Pre-processed parquet file

### 3. Metacyc Database
- **Data Includes**: Kcat, Km
- **Input Format**: Pre-processed parquet file

### 4. BindingDB
- **Data Includes**: Ki
- **Input Format**: Pre-processed parquet file
- **Note**: BindingDB data is incorporated for Ki measurements not present in other sources

---

## Configuration

The pipeline is configured with Hydra. The builder keys are in `configs/data/kinetic_params_dataset.yaml`. The output paths that the split configs read (`kinetic_dataset_path` and `dataset_paths`) are in `configs/data/split_inputs/kinetic_params_dataset.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sabio_kinetic_params_parquet_file_path` | from `sabio_rk` | Path to SABIO kinetic parameters |
| `brenda_kinetic_params_parquet_file_path` | from `brenda` | Path to BRENDA kinetic parameters |
| `include_metacyc` | `false` | Merge Metacyc kinetic parameters when `true`. When `false`, the Metacyc parquet is not read and need not exist. |
| `metacyc_kinetic_params_parquet_file_path` | from `metacyc` | Path to Metacyc kinetic parameters (used only when `include_metacyc` is `true`) |
| `bindingdb_kinetic_params_parquet_file_path` | `<bindingdb data path>/ki.parquet` | Path to BindingDB Ki measurements |
| `save_1d_dataset` | `true` | Save the 1D parquet files |
| `save_3d_dataset` | `true` | Save the 3D parquet files. At least one of the two must be `true` |
| `keep_temperature` | `false` | Keep `Temperature` in the dataset and in the duplicate-merge key. When `false`, the column is dropped before merging |
| `keep_ph` | `false` | Same for `pH` |
| `value_types` | `[unified, kcat, km, ki]` | Files to save. Allowed values: `unified`, `kcat`, `km`, `ki` |
| `unified_kinetic_params_smiles_pickle_path` | `<kinetic_dataset_path>/unified_kinetic_params_smiles.pkl` | Output path for the unique SMILES pickle |
| `unified_kinetic_params_sequences_pickle_path` | `<kinetic_dataset_path>/unified_kinetic_params_sequences.pkl` | Output path for the protein sequences pickle |
| `unified_kinetic_params_1d_parquet_file_path`, `unified_kinetic_params_3d_parquet_file_path` | `dataset_paths.unified.parquet_1d`, `.parquet_3d` | Output paths of the unified dataset |
| `unified_kinetic_params_parquet_file_path` | the 3D path | Alias of the 3D path |
| `dataset_paths.<type>.{dir,parquet_1d,parquet_3d}` | see Output Files | Output paths for `unified`, `kcat`, `km` and `ki` (in `split_inputs`) |
| `log_dir`, `log_file_name` | `${paths.log_dir}`, `unify_kinetic_params.log` | Log location |

---

## Pipeline Steps

### Step 1: Read Kinetic Law Data

**Method**: `_read_kinetic_law_dfs()`

**Description**: Reads kinetic parameter DataFrames from SABIO and BRENDA parquet files.

**Process**:
1. Read SABIO kinetic parameters from parquet
2. Add `source` column with value "sabio"
3. Read BRENDA kinetic parameters from parquet
4. Add `source` column with value "brenda"

**Output**: List of DataFrames [brenda_df, sabio_df] with source attribution.

---

### Step 2: Unify BRENDA and SABIO

**Method**: `_unify_brenda_sabio()`

**Description**: Combines BRENDA and SABIO data, retrieving missing SMILES from PubChem and protein sequences from UniProt.

#### 2.1 Concatenate DataFrames

**Action**: Combine BRENDA and SABIO DataFrames into a single unified DataFrame.

#### 2.2 Search PubChem for Missing SMILES

**Process**:
1. Identify substrates with missing SMILES strings (where `smiles` is NaN)
2. Call `get_pubchem_cids_from_name_parallel()` to find PubChem Compound IDs (CIDs) for substrate names. It scans PubChem's `CID-Synonym-filtered.gz` once (downloaded when missing) and matches names case-insensitively. A name with several CIDs gets the lowest CID. Results are cached in `name_to_cid.json`, so later runs scan only new names.
3. Extract unique CIDs

**Output**: List of PubChem CIDs for substrates without SMILES.

#### 2.3 Fetch SMILES from PubChem

**Method**: `get_pubchem_smiles_from_cids()`

**Description**: Retrieves SMILES strings for compounds with known PubChem CIDs.

**Process**:
1. Scan PubChem's `CID-SMILES.gz` once (downloaded when missing) for the wanted CIDs
2. Create mapping of CID -> SMILES (None for CIDs not in the file)
3. Cache the mapping in `cid_to_smiles.json`, so later runs scan only new CIDs

**Action**: Update DataFrame by adding retrieved SMILES for rows where `smiles` column was NaN, using the substrate name to map to the SMILES.

**Note**: If a CID is not in the file for a substrate, the entry will have a missing SMILES and will be dropped in later steps.

#### 2.4 Fetch Protein Sequences

**Method**: `get_uniprot_acc_json()`

**Description**: Retrieves protein sequences from UniProt for all unique UniProt IDs.

**Process**:
1. Extract unique UniProt IDs from the `UniprotID` column
2. Call `get_uniprot_acc_json()` to fetch sequences from UniProt API
3. Extract `acc_id` and `sequence` columns
4. Rename columns and merge into main DataFrame on `UniprotID`

**Output**: DataFrame with protein sequences added.

#### 2.5 Apply Mutations

**Method**: `mutate_sequence()`

**Description**: For enzyme variants/mutants, applies specified mutations to the wildtype protein sequence.

**Process**:
1. Identify rows with non-null `mutations` column
2. Convert mutations to tuple format for processing
3. For each row with mutations:
   - Apply the specified mutations to the protein sequence
   - Update the `sequence` column with the mutated sequence
4. Keep wildtype sequences unchanged (where `mutations` is null)

**Note**: This allows the model to learn sequence-specific kinetic parameters for both wildtype and mutant enzymes.

---

### Step 3: Unify with Metacyc

**Method**: `_unify_metacyc()`

**Description**: Integrates Metacyc kinetic parameters by standardizing column names and combining with BRENDA/SABIO data. This runs only when `include_metacyc` is `true` (default `false`). When it is `false`, the step logs that Metacyc is excluded and keeps the BRENDA/SABIO rows. The step still cleans and renames the `uniprot_id` column, so the output schema does not change.

#### 3.1 Prepare Metacyc DataFrame

**Process**:
1. Add missing columns with null/default values:
   - `Temperature`: None
   - `pH`: None
   - `mutations`: None
   - `pdbs`: None
   - `enzymeType`: "wildtype"
   - `source`: "metacyc"
2. Select relevant columns to match BRENDA/SABIO schema
3. Ensure column ordering is consistent

#### 3.2 Concatenate and Clean

**Actions**:
1. Concatenate BRENDA/SABIO unified DataFrame with prepared Metacyc DataFrame
2. Drop rows with missing SMILES or sequence
3. Remove duplicate rows based on: `smiles`, `sequence`, `value`, `value_type`
4. Standardize UniProt IDs by replacing "/" with "_"
5. Rename `UniprotID` to `uniprot_id`

**Output**: Unified DataFrame combining BRENDA, SABIO, and Metacyc data.

---

### Step 4: Unify with BindingDB

**Method**: `_unify_bindingdb()`

**Description**: Integrates BindingDB Ki measurements by column selection and renaming.

#### 4.1 Prepare BindingDB DataFrame

**Process**:
1. Select relevant columns from BindingDB Ki data
2. Rename columns to match unified schema:
   - `organism` -> `Organism`
   - `temp` -> `Temperature`
   - `ligand` -> `substrate`
   - `ligand_id` -> `substrate_id`
3. Convert `value_type` to lowercase (e.g., "KI" -> "ki")
4. Set defaults:
   - `enzymeType`: "wildtype"
   - `source`: "bindingdb"
   - `ECNumber`: None
   - `mutations`: None
5. Parse `pdbs` column: convert comma-separated strings to lists

#### 4.2 Concatenate and Clean

**Actions**:
1. Concatenate the unified BRENDA/SABIO/Metacyc DataFrame with prepared BindingDB DataFrame
2. Drop rows with missing SMILES or sequence
3. Remove duplicates based on: `smiles`, `sequence`, `value`, `value_type`

**Output**: Unified DataFrame combining all four sources (BRENDA, SABIO, Metacyc, BindingDB).

#### 4.3 Keep Enzyme Rows

**Method**: `_keep_enzyme_rows()`

The builder keeps only rows whose target is an enzyme. This rule applies to kcat, Km and Ki rows. A row stays if one of these is true:
- It has an `ECNumber` from its source.
- The UniProt entry for its `uniprot_id` lists at least one EC number. Partial ECs such as `3.4.-.-` count.

BRENDA and SABIO rows always have an `ECNumber`, so they stay. BindingDB rows have no `ECNumber`, so only those with a UniProt EC stay. Most dropped BindingDB targets are receptors, such as GPCRs. The builder looks up UniProt only for rows without a source EC. It does not fill `ECNumber` for BindingDB rows. It logs the row counts before and after, and the rows dropped by value type and source.

---

### Step 5: Validate SMILES

**Method**: `_clean_unified_df()`

**Description**: Validates and canonicalizes all SMILES strings using parallel processing.

#### 5.1 Canonicalize SMILES

**Process**:
1. Extract unique SMILES from the dataset
2. Apply `canonicalize_smiles()` function in parallel using `pandarallel`
3. Drop entries with invalid SMILES (canonicalization returns `None`)
4. Merge canonical SMILES back to main DataFrame
5. Replace original SMILES column with canonical version

**Parallelization**: Uses `pandarallel` with the number of workers equal to the CPU count.

**Logging**: Reports number of entries dropped for each kinetic type (kcat, km, ki) due to invalid SMILES.

---

### Step 6: Aggregate by Kinetic Type

**Method**: `_clean_kinetic_law_df()`

**Description**: Cleans and aggregates kinetic data separately for each measurement type (kcat, km, ki) with type-specific aggregation strategies.

#### 6.1 Filter Non-Positive Values

**Action**: Remove entries where `value <= 0` (invalid kinetic measurements).

#### 6.2 Log Transform

**Action**: Create `log10_value = np.log10(value)`. Drop rows where it is not finite.

#### 6.3 Assay Conditions

**Units**: All sources give temperature in Kelvin.

**Actions**:
1. Fill a missing `Temperature` with 298.15 K (25 C). Fill a missing `pH` with 7.0.
2. `null_implausible_conditions` sets a temperature outside 273 to 373.15 K, or a pH outside 0 to 14, to null. The row stays.
3. Drop the `Temperature` column unless `keep_temperature` is `true`. Drop `pH` unless `keep_ph` is `true`. Both are `false` by default, so rows that differ only in temperature or pH merge in the next step.

#### 6.4 Type-Specific Aggregation

**Grouping Columns** (defines what constitutes a "unique" measurement):
- `smiles`
- `enzymeType`
- `Organism`
- `uniprot_id`
- `sequence`
- `Temperature` (only if `keep_temperature`)
- `pH` (only if `keep_ph`)
- `unit`

Rows are sorted by the group columns, `log10_value` and `source` first, so the result does not depend on row order.

##### For kcat (Turnover Number)

**Strategy**: Takes the row with the maximum `log10_value` in each group (highest catalytic efficiency). The row keeps its own `value`.

**Rationale**: kcat values vary significantly with experimental conditions. Taking the maximum represents the best-measured catalytic rate.

##### For km and ki (Substrate/Inhibitor Affinity)

**Aggregation Method**: Average of log-transformed values. Then `value = 10 ** log10_value`, with full precision (no rounding).

**Aggregation Details**:

| Column Type | Aggregation | Notes |
|-------------|-------------|-------|
| `log10_value` | `mean` | Average of log-transformed values |
| `source` | join with "\|" | Sorted, pipe-separated list of source databases |
| `uniprot_match` | join with "\|" | Sorted labels that tell how the UniProt ID was matched. Null if no row has a label |
| `pdbs` | first non-null | Takes first non-null PDB structure if available |
| Metadata columns | `first` | Takes first value |

**Metadata Columns** (aggregated with `first`):
- `ECNumber`
- `substrate`
- `substrate_id`
- `mutations`

#### 6.5 Annotate Value Type

**Action**: Set `value_type` to `"kcat"`, `"km"` or `"ki"`.

---

### Step 7: Final Cleaning

**Description**: Standardizes IDs and adds the SMILES key and hash.

#### 7.1 Standardize UniProt IDs

**Action**: Replace spaces with underscores in UniProt IDs.

#### 7.2 Add SMILES Key and Hash

**Process**:
1. Extract unique SMILES from the dataset
2. Compute `smiles_key` with `molecule_key` (salt, charge and tautomer forms of one compound share a key; stereo is kept). If the key fails, the SMILES is used. `smiles_key` is the SMILES that the embedding steps use.
3. Compute `smiles_hash` from `smiles_key` in parallel with `pandarallel`
4. Merge `smiles_key` and `smiles_hash` back into the main DataFrame
5. Sort the rows in a stable order

#### 7.3 Pickles and UniProt dates

The builder writes the two pickles (Step 8.2 and 8.3). Then `add_uniprot_date_column` adds `uniprot_date`.

---

### Step 8: Save Outputs

**Description**: Saves all processed data to disk.

#### 8.1 Kinetic Parameters Datasets

**Format**: Parquet with zstd compression (level 3)

**Contents**: For each entry in `value_types`, the builder saves a 1D and a 3D file (if `save_1d_dataset` and `save_3d_dataset` are on). `unified` holds kcat, km and ki rows. The other types hold only their own rows.

- **1D**: one row per measurement. No structure columns, except the `pdbs` list from the source.
- **3D**: made with `assign_experimental_and_af_pdbs`. Each row gets experimental structure chains (`pdbs` as `<PDB>:<base accession>`, `pdb_chain`, `pdb_coverage`, `pdb_source`, `pdb_type`). A row without a selected chain gets one fallback row with a predicted structure. A row can appear several times in the 3D file.

#### 8.2 Protein Sequences Pickle

**Method**: `_pickle_all_sequences()`

**Format**: Python pickle

**Contents**: List of dictionaries with keys:
- `acc_id`: UniProt accession ID or UniProt ID with mutation annotations
- `sequence`: Protein amino acid sequence (wildtype or mutated)

**Note**: Each unique (UniProt ID, sequence) pair is saved, allowing downstream embedding pipelines to process both wildtype and mutant sequences.

#### 8.3 Ligand SMILES Pickle

**Method**: `_pickle_all_smiles()`

**Format**: Python pickle

**Contents**: List of dictionaries with keys:
- `substrate`: Substrate/ligand name
- `smiles`: Canonical SMILES string

**Note**: Unique (substrate, SMILES) pairs ensure proper mapping between substrate identifiers and chemical structures.

---

## Output Files

`<kinetic_dataset_path>` is `paths.kinetic_dataset_path`.

| File | Format | Description |
|------|--------|-------------|
| `<kinetic_dataset_path>/{unified,kcat,km,ki}/<type>_kinetic_params_1d.parquet` | Parquet | 1D dataset for each value type (for example `unified/unified_kinetic_params_1d.parquet`) |
| `<kinetic_dataset_path>/{unified,kcat,km,ki}/<type>_kinetic_params_3d.parquet` | Parquet | 3D dataset for each value type |
| `<kinetic_dataset_path>/unified_kinetic_params_sequences.pkl` | Pickle | Protein sequences (wildtype and mutant variants) |
| `<kinetic_dataset_path>/unified_kinetic_params_smiles.pkl` | Pickle | Substrate names and SMILES |

---

## Data Schema

### Final Kinetic Parameters Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `smiles` | string | Canonical SMILES of the substrate |
| `smiles_key` | string | `molecule_key` of the SMILES. Grouping and embedding key |
| `smiles_hash` | string | Hash of `smiles_key` |
| `enzymeType` | string | "wildtype" or "mutant" (for enzyme variants) |
| `Organism` | string | Source organism of the enzyme |
| `uniprot_id` | string | UniProt accession ID (spaces replaced with underscores) |
| `uniprot_match` | string | How the UniProt ID was matched (from BRENDA). Null if not labelled |
| `uniprot_date` | date | UniProt creation date of the accession |
| `sequence` | string | Amino acid sequence (wildtype or with applied mutations) |
| `Temperature` | float | Temperature in Kelvin. Only present if `keep_temperature` is `true`. Missing values are 298.15 K |
| `pH` | float | pH of the measurement. Only present if `keep_ph` is `true`. Missing values are 7.0 |
| `unit` | string | Unit of measurement (depends on kinetic type) |
| `value_type` | string | Type of kinetic parameter ("kcat", "km", or "ki") |
| `ECNumber` | string | Enzyme Commission (EC) number |
| `substrate` | string | Substrate/ligand name |
| `substrate_id` | string | Standardized substrate identifier |
| `mutations` | tuple | Amino acid mutations (None for wildtype) |
| `pdbs` | list or string | 1D: PDB IDs from the source, or null. 3D: structure key |
| `pdb_source`, `pdb_type`, `pdb_chain`, `pdb_coverage` | string, string, string, float | 3D files only: structure source, type, selected chain and its coverage |
| `source` | string | Source database, or a sorted pipe-separated list if rows merged |
| `log10_value` | float | Log10 of the kinetic parameter value |
| `value` | float | Kinetic parameter value |

### Unit Standards

| Kinetic Type | Unit | Description |
|--------------|------|-------------|
| `kcat` | `s^-1` | Catalytic rate constant (turnover number) |
| `km` | `mM` | Michaelis constant (Millimolar concentration units) |
| `ki` | `mM` | Inhibition constant (Millimolar concentration units) |

---

## Notes

1. **Mutation Handling**: Wildtype and mutant enzyme variants are treated as separate entries with distinct sequences, allowing the model to learn how mutations affect kinetic parameters.

2. **SMILES Canonicalization**: Ensures consistent molecular representation across the dataset, critical for proper train/test splitting and model training.

3. **Type-Specific Aggregation**: 
   - **kcat**: Maximum value taken (best measured catalytic rate)
   - **km/ki**: Mean of log-transformed values (handles wide range of affinities)

4. **Sequence Requirement**: Entries without retrievable protein sequences from UniProt are excluded, as sequences are required for downstream modeling.

5. **PubChem Integration**: Missing SMILES strings are automatically retrieved from PubChem using substrate names (matched offline in PubChem's synonym file), expanding dataset coverage.

6. **Log-Space Aggregation**: When multiple measurements exist for the same enzyme-substrate pair, values are averaged in log space to handle the typically log-normal distribution of biochemical parameters.

7. **Source Attribution**: Each entry is tagged with its source database (BRENDA, SABIO, Metacyc, BindingDB) or pipe-separated sources if data was aggregated during cleaning.

8. **PubChem Name Matching**: The name lookup matches exact synonyms. Some protein-bound substrate names, such as `acetyl-[acyl-carrier protein]`, `dodecanoyl-[acp]`, `thioredoxin` and `holocytochrome c`, match a small-molecule CID in PubChem. The current build has about 11 such names. Their SMILES are probably wrong.
