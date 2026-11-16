# Binding Affinity Dataset Generation

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document provides a comprehensive description of the binding affinity dataset generation pipeline implemented in `src/data/components/binding_affinity_dataset.py`.

## Table of Contents

1. [Overview](#overview)
2. [Data Source](#data-source)
3. [Configuration](#configuration)
4. [Pipeline Steps](#pipeline-steps)
   - [Step 1: Read BindingDB Data](#step-1-read-bindingdb-data)
   - [Step 2: Extract Single-Chain Entries](#step-2-extract-single-chain-entries)
   - [Step 3: Extract Multi-Chain Entries](#step-3-extract-multi-chain-entries)
   - [Step 4: Flatten Multi-Chain Entries](#step-4-flatten-multi-chain-entries)
   - [Step 5: Concatenate and Clean Entries](#step-5-concatenate-and-clean-entries)
   - [Step 6: Data Type Cleaning](#step-6-data-type-cleaning) (includes the placeholder filter)
   - [Step 7: Column Renaming](#step-7-column-renaming)
   - [Step 8: SMILES Canonicalization](#step-8-smiles-canonicalization)
   - [Step 9: Fetch Protein Sequences](#step-9-fetch-protein-sequences)
   - [Step 10: Split by Affinity Type](#step-10-split-by-affinity-type)
   - [Step 11: Aggregate and Clean](#step-11-aggregate-and-clean)
   - [Step 12: Save Outputs](#step-12-save-outputs)
5. [Output Files](#output-files)
6. [Data Schema](#data-schema)

---

## Overview

The `BindingAffinityDatasetBuilder` class processes raw BindingDB data to create a clean, unified binding affinity dataset containing:

- **Ki** (Inhibition constant)
- **Kd** (Dissociation constant)
- **EC50** (Half maximal effective concentration)
- **IC50** (Half maximal inhibitory concentration)

The pipeline handles multi-chain protein complexes, canonicalizes SMILES strings, fetches protein sequences from UniProt, and aggregates duplicate measurements.

---

## Data Source

**BindingDB** - A public database of measured binding affinities between proteins and drug-like molecules.

- **Download URL**: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
- **File Required**: Extracted "All Data" TSV file (e.g., `BindingDB_All_YYYYMM_tsv.zip`)
- **File Format**: Tab-separated values (TSV)

---

## Configuration

The pipeline is configured with Hydra. The builder keys are in `configs/data/binding_affinity_dataset.yaml`. The output paths that the split configs read (`binding_affinity_dataset_path` and `dataset_paths`) are in `configs/data/split_inputs/binding_affinity_dataset.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bindingdb_raw_path` | `paths.bindingdb_raw_path` | Folder with the raw BindingDB file |
| `bindingdb_tsv_file_path` | `<bindingdb_raw_path>/BindingDB_All.tsv` | Path to the raw BindingDB TSV file |
| `bindingdb_data_path` | `paths.bindingdb_data_path` | Output folder for the per-type files (`ki.parquet`, `kd.parquet`, ...) |
| `save_1d_dataset` | `true` | Save the 1D parquet files |
| `save_3d_dataset` | `true` | Save the 3D parquet files. At least one of the two must be `true` |
| `keep_temperature` | `false` | Keep `temp` in the dataset and in the duplicate-merge key. When `false`, the column is dropped before merging |
| `keep_ph` | `false` | Same for `pH` |
| `value_types` | `[unified, kd, ec50, ic50]` | Files to save. Allowed values: `unified`, `kd`, `ec50`, `ic50` |
| `binding_affinity_smiles_pickle_path` | `<binding_affinity_dataset_path>/binding_affinity_smiles.pkl` | Output path for the unique SMILES pickle |
| `binding_affinity_sequences_pickle_path` | `<binding_affinity_dataset_path>/binding_affinity_sequences.pkl` | Output path for the protein sequences pickle |
| `unified_binding_affinity_1d_dataset_parquet_file_path`, `unified_binding_affinity_3d_dataset_parquet_file_path` | `dataset_paths.unified.parquet_1d`, `.parquet_3d` | Output paths of the unified dataset |
| `binding_affinity_dataset_parquet_file_path` | the 3D path | Alias of the 3D path |
| `dataset_paths.<type>.{dir,parquet_1d,parquet_3d}` | see Output Files | Output paths for `unified`, `kd`, `ec50` and `ic50` (in `split_inputs`) |
| `pandarallel.nb_workers` | `112` | Workers for SMILES canonicalization (the code uses `min(nb_workers, cpu_count())`) |
| `pandarallel.progress_bar` | `true` | Show the progress bar |
| `log_dir`, `log_file_name` | `${paths.log_dir}`, `binding_affinity_dataset.log` | Log location |

---

## Pipeline Steps

### Step 1: Read BindingDB Data

**Method**: `_read_bindingdb_data()`

**Description**: Reads the raw BindingDB TSV file while dropping unnecessary columns to reduce memory usage.

**Columns Dropped**:

- `BindingDB Reactant_set_id`
- `Ligand InChI`
- `Ligand InChI Key`
- `Article DOI`
- `Authors`
- `Date of publication`
- `Date in BindingDB`
- `Institution`
- `Link to Ligand in BindingDB`
- `Link to Target in BindingDB`
- `Link to Ligand-Target Pair in BindingDB`
- `Ligand HET ID in PDB`
- `DrugBank ID of Ligand`
- `IUPHAR_GRAC ID of Ligand`

For each chain index `i` from 1 to 50, the following are also dropped:
- `BindingDB Target Chain Sequence {i}`
- `UniProt (SwissProt) Recommended Name of Target Chain {i}`
- `UniProt (SwissProt) Entry Name of Target Chain {i}`
- `UniProt (SwissProt) Secondary ID(s) of Target Chain {i}`
- `UniProt (SwissProt) Alternative ID(s) of Target Chain {i}`
- `UniProt (TrEMBL) Submitted Name of Target Chain {i}`
- `UniProt (TrEMBL) Entry Name of Target Chain {i}`
- `UniProt (TrEMBL) Primary ID of Target Chain {i}`
- `UniProt (TrEMBL) Secondary ID(s) of Target Chain {i}`
- `UniProt (TrEMBL) Alternative ID(s) of Target Chain {i}`

**Note**: `Curation/DataSource`, `PubChem AID` and `Patent Number` are kept at this step. Step 6.5 uses them and then drops them.

**Output**: DataFrame with all remaining columns from BindingDB.

---

### Step 2: Extract Single-Chain Entries

**Description**: Filters entries where the target protein has exactly one chain.

**Filter Condition**:
```
Number of Protein Chains in Target (>1 implies a multichain complex) == 1
```

**Actions**:
1. Filter rows where chain count equals 1
2. Drop chain-specific columns for chains 2-50 (only chain 1 data is relevant):
   - `UniProt (SwissProt) Primary ID of Target Chain {2-50}`
   - `PDB ID(s) of Target Chain {2-50}`

**Output**: DataFrame containing only single-chain protein entries.

---

### Step 3: Extract Multi-Chain Entries

**Description**: Filters entries where the target protein has more than one chain (protein complexes).

**Filter Condition**:
```
Number of Protein Chains in Target (>1 implies a multichain complex) > 1
```

**Output**: DataFrame containing only multi-chain protein complex entries.

---

### Step 4: Flatten Multi-Chain Entries

**Method**: `_explode_multichain_with_metadata()`

**Description**: Converts multi-chain complex entries into multiple single-chain entries. Each chain in a complex becomes its own row, preserving all metadata from the original entry.

**Process**:
1. For each row in the multi-chain DataFrame:
   - Iterate through chain indices 1 to 50
   - For each chain index, check if chain data exists (non-null values)
   - If chain data exists, create a new row with:
     - All metadata columns from the original row
     - Chain-specific data moved to "Chain 1" columns

**Chain-Specific Fields Processed**:
- `UniProt (SwissProt) Primary ID of Target Chain {i}`
- `PDB ID(s) of Target Chain {i}`

**Example**:
```
Original row with 3 chains:
  - UniProt Chain 1: P12345
  - UniProt Chain 2: P67890
  - UniProt Chain 3: P11111

Becomes 3 rows:
  Row 1: UniProt Chain 1: P12345
  Row 2: UniProt Chain 1: P67890
  Row 3: UniProt Chain 1: P11111
```

**Output**: DataFrame with one row per chain, all chain data in "Chain 1" columns.

---

### Step 5: Concatenate and Clean Entries

**Description**: Combines single-chain and flattened multi-chain entries into a unified DataFrame.

**Actions**:
1. Concatenate single-chain and multi-chain DataFrames
2. Drop columns:
   - `Number of Protein Chains in Target (>1 implies a multichain complex)`
   - `kon (M-1-s-1)` (association rate constant)
   - `koff (s-1)` (dissociation rate constant)
3. Drop rows where either is null:
   - `Ligand SMILES`
   - `UniProt (SwissProt) Primary ID of Target Chain 1`

**Output**: Unified DataFrame with valid SMILES and UniProt IDs.

---

### Step 6: Data Type Cleaning

**Description**: Cleans and standardizes various columns.

#### 6.1 Temperature and pH Cleaning

**Column**: `Temp (C)`

**Actions**:
1. Extract the number from strings like "25 °C" and add 273.15. The result is in Kelvin, for example `25 °C` -> `298.15`. The column keeps its name `Temp (C)` until Step 7 renames it to `temp`.
2. Fill a missing temperature with 298.15 K (25 C).
3. Fill a missing `pH` with 7.0.
4. `null_implausible_conditions` sets a temperature outside 273 to 373.15 K, or a pH outside 0 to 14, to null. The row stays.

#### 6.2 ID Columns Cleaning

**Columns**: `PMID`, `PubChem CID`, `PubChem SID`

**Action**: Convert to string representation of integers (remove decimal points)

```python
str(int(x)) if pd.notna(x) else x
```

#### 6.3 Ligand Name Cleaning

**Column**: `BindingDB Ligand Name`

**Action**: Take only the first part before "::" separator

```python
x.split("::")[0]
```

#### 6.4 Affinity Value Cleaning

**Method**: `_to_float()`

**Columns**: `Ki (nM)`, `Kd (nM)`, `EC50 (nM)`, `IC50 (nM)`

**Actions**:
1. Remove commas and whitespace.
2. A value that starts with `>` or `<` is a qualified value. It becomes null (the log counts these values).
3. Strip a leading `=` and convert to float. A value that does not convert becomes null.

**Output**: All affinity values as floats in nanomolar (nM) units.

#### 6.5 Placeholder Values

**Method**: `_drop_placeholder_values()`

**Description**: Sets binned or placeholder affinities to null. The row stays for its other affinity types. For each rule and column, the log shows how many values the rule removed. Only values above 0 are tested.

| Rule | Rows | Columns | Values removed |
|------|------|---------|----------------|
| `patent_bin_midpoint` | `Curation/DataSource` is `US Patent` or `WIPO` | Ki, Kd, EC50, IC50 | Values `5.5 * 10^n` (the midpoint of a bin) |
| `tyra_kd_10nm` | Patent `US20240109865` | Kd | 10 nM |
| `aid_1914_ec50_150000nm` | PubChem AID `aid1914` | EC50 | 150000 nM |
| `aid_857_858` | PubChem AID `aid857` or `aid858` | Ki, Kd, EC50, IC50 | All values |

After the rules, the builder drops `Curation/DataSource`, `PubChem AID` and `Patent Number`.

---

### Step 7: Column Renaming

**Description**: Renames columns to standardized, shorter names.

| Original Column | New Name |
|-----------------|----------|
| `Ligand SMILES` | `smiles` |
| `BindingDB MonomerID` | `monomer_id` |
| `BindingDB Ligand Name` | `ligand` |
| `Target Name` | `target` |
| `Target Source Organism According to Curator or DataSource` | `organism` |
| `Ki (nM)` | `ki` |
| `IC50 (nM)` | `ic50` |
| `Kd (nM)` | `kd` |
| `EC50 (nM)` | `ec50` |
| `Temp (C)` | `temp` |
| `BindingDB Entry DOI` | `doi` |
| `PMID` | `pmid` |
| `PDB ID(s) for Ligand-Target Complex` | `ligand_target_complex_pdbs` |
| `PubChem CID` | `pubchem_cid` |
| `PubChem SID` | `pubchem_sid` |
| `ChEBI ID of Ligand` | `chebi_id` |
| `ChEMBL ID of Ligand` | `chembl_id` |
| `KEGG ID of Ligand` | `kegg_id` |
| `ZINC ID of Ligand` | `zinc_id` |
| `UniProt (SwissProt) Primary ID of Target Chain 1` | `uniprot_id` |
| `PDB ID(s) of Target Chain 1` | `pdbs` |

---

### Step 8: SMILES Canonicalization & Hashing

**Description**: Cleans and converts all SMILES strings to their canonical form using RDKit and generates SHA256 hashes of the canonicalized SMILES.

#### 8.1 Pipe Character Cleaning

**Issue**: Some SMILES strings in BindingDB contain pipe characters (`|`) followed by non SMILES data.

**Action**: Extract only the SMILES and discard anything after the pipe character:
```python
mask = bindingdb_df["smiles"].str.contains(r"\|")
bindingdb_df.loc[mask, "smiles"] = (
    bindingdb_df.loc[mask, "smiles"].str.split("|").str[0].str.strip()
)
```

**Example**: `"O=C1CCCC2=C1C1(CCS(=O)(=O)C1)N=C(Nc1nc3ccccc3o1)N2 |c:5,t:16|"` -> `"O=C1CCCC2=C1C1(CCS(=O)(=O)C1)N=C(Nc1nc3ccccc3o1)N2"`

#### 8.2 Canonicalization & Hashing

**Process**:
1. Extract unique SMILES from the dataset
2. Apply `canonicalize_smiles()` function in parallel using `pandarallel`
3. Drop entries with invalid SMILES (canonicalization returns `None`)
4. Compute `smiles_key` with `molecule_key` (salt, charge and tautomer forms of one compound share a key; stereo is kept). If the key fails, the canonical SMILES is used. `smiles_key` is the SMILES that the embedding steps use.
5. Apply `smiles_hash()` to `smiles_key` in parallel using `pandarallel`
6. Merge canonical SMILES, `smiles_key` and `smiles_hash` back to main DataFrame
7. Replace original SMILES column with canonical version

**Parallelization**: Uses `pandarallel` with configurable number of workers.

**Logging**: Reports number of entries dropped for each affinity type (Ki, Kd, EC50, IC50) due to invalid SMILES.

---

### Step 9: Fetch Protein Sequences

**Description**: Retrieves protein sequences from UniProt for all unique UniProt IDs.

**Process**:
1. Split multi-ID entries (space-separated UniProt IDs) into separate rows by exploding the DataFrame
2. Extract unique UniProt IDs from the exploded DataFrame
3. Call `get_uniprot_acc_json()` to fetch sequences from UniProt API
4. Extract `acc_id` and `sequence` columns
5. Merge sequences into main DataFrame on `uniprot_id`
6. Drop entries where sequence could not be retrieved

**Note**: Entries with multiple space-separated UniProt IDs are expanded so each ID becomes its own row, allowing all protein-ligand pairs to be captured.

**Logging**: Reports number of entries dropped due to missing protein sequences.

---

### Step 10: Split by Affinity Type

**Description**: First, the builder sets `ligand_id` for each row: `PUBCHEM:C<CID>`, else `CHEBI:<id>`, else `KEGG:<id>`, else `CHEMBL:<id>`, else `ZINC:<id>`, else the ligand name. Then it creates a separate DataFrame for each affinity type. All values are converted from nanomolar (nM) to millimolar (mM) using the conversion factor: 1 nM = 1e-6 mM.

**Affinity Types Processed**:
- **ki**: Inhibition constant
- **kd**: Dissociation constant
- **ec50**: Half maximal effective concentration
- **ic50**: Half maximal inhibitory concentration

**Process for Each Type**:
1. Filter rows where the affinity type column is not null
2. Drop columns for other affinity types
3. Rename the affinity column to `value`
4. Add `value_type` column with the affinity type name
5. Convert `value` from nM to mM: `value = value * 1e-6`
6. Add `unit` column with value "millimolar (mM)"

**Output**: Four DataFrames (ki, kd, ec50, ic50), one per affinity type, saved as individual parquet files.

**Note**: Only Kd, EC50, and IC50 measurements are included in the unified binding affinity dataset (the code concatenates the `kd`, `ec50` and `ic50` frames). Ki measurements are saved as `ki.parquet` in `bindingdb_data_path`. The kinetic parameters builder reads that file.

---

### Step 11: Aggregate and Clean

**Method**: `_clean_binding_affinity_df()`

**Description**: Aggregates duplicate measurements and cleans the data.

#### 11.1 Filter Non-Positive Values

**Action**: Remove entries where `value <= 0` (invalid affinity measurements)

#### 11.2 Drop Conditions and Log Transform

**Actions**:
1. Drop `temp` unless `keep_temperature` is `true`. Drop `pH` unless `keep_ph` is `true`. Both are `false` by default, so rows that differ only in temperature or pH merge in the next step.
2. Create `log10_value = np.log10(value)`.

#### 11.3 Aggregate Duplicates

**Grouping Columns** (defines what constitutes a "unique" measurement):
- `smiles`
- `smiles_hash`
- `smiles_key`
- `organism`
- `temp` (only if `keep_temperature`)
- `pH` (only if `keep_ph`)
- `sequence`
- `uniprot_id`
- `value_type`

**Aggregation Strategy**:

| Column Type | Aggregation | Notes |
|-------------|-------------|-------|
| `log10_value` | `mean` | Average of log-transformed values |
| Metadata columns | `first` (skip NaN) | Takes first non-null value |

**Metadata Columns** (aggregated with `first`):
- `monomer_id`
- `ligand`
- `ligand_id`
- `target`
- `doi`
- `pmid`
- `ligand_target_complex_pdbs`
- `pubchem_cid`
- `pubchem_sid`
- `chebi_id`
- `chembl_id`
- `kegg_id`
- `zinc_id`
- `pdbs`
- `unit`

#### 11.4 Back-Transform Value

**Action**: Convert log10 value back to linear scale, with full precision (no rounding):
```python
value = 10 ** log10_value
```

**Logging**: Reports number of entries removed during cleaning for each affinity type.

---

### Step 12: Save Outputs

**Description**: Saves all processed data to disk. All parquet files use zstd compression (level 3).

#### 12.1 Individual Affinity Type Files

**Files** (in `bindingdb_data_path`):
- `ki.parquet`
- `kd.parquet`
- `ec50.parquet`
- `ic50.parquet`

**Contents**: Individual affinity type DataFrames before aggregation (values in mM).

#### 12.2 Binding Affinity Datasets

**Files**: for each entry in `value_types`, a 1D file and a 3D file (if `save_1d_dataset` and `save_3d_dataset` are on). `unified` holds Kd, EC50 and IC50 rows. The other types hold only their own rows.

**Contents**: Cleaned and aggregated dataset with `uniprot_date` added (Ki is excluded).
- **1D**: one row per aggregated measurement. `pdbs` is a list of PDB IDs from BindingDB (empty if none).
- **3D**: made with `assign_experimental_and_af_pdbs`. Each row gets experimental structure chains (`pdbs` as `<PDB>:<base accession>`, `pdb_chain`, `pdb_coverage`, `pdb_source`, `pdb_type`). A row without a selected chain gets one fallback row with a predicted structure. A row can appear several times in the 3D file.

#### 12.3 Protein Sequences Pickle

**Format**: Python pickle

**File**: `{binding_affinity_sequences_pickle_path}`

**Contents**: List of dictionaries with keys:
- `acc_id`: UniProt accession ID
- `sequence`: Protein amino acid sequence

#### 12.4 Ligand SMILES Pickle

**Format**: Python pickle

**File**: `{binding_affinity_smiles_pickle_path}`

**Contents**: List of unique canonical SMILES strings.

---

## Output Files

`<binding_affinity_dataset_path>` is `paths.binding_affinity_dataset_path`.

| File | Format | Description |
|------|--------|-------------|
| `<bindingdb_data_path>/{ki,kd,ec50,ic50}.parquet` | Parquet | Measurements of one type (before aggregation) |
| `<binding_affinity_dataset_path>/{unified,kd,ec50,ic50}/<type>_binding_affinity_dataset_1d.parquet` | Parquet | 1D dataset for each value type (for example `unified/unified_binding_affinity_dataset_1d.parquet`) |
| `<binding_affinity_dataset_path>/{unified,kd,ec50,ic50}/<type>_binding_affinity_dataset_3d.parquet` | Parquet | 3D dataset for each value type |
| `<binding_affinity_dataset_path>/binding_affinity_sequences.pkl` | Pickle | Protein sequences |
| `<binding_affinity_dataset_path>/binding_affinity_smiles.pkl` | Pickle | Unique ligand SMILES |

---

## Data Schema

### Final Binding Affinity Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `smiles` | string | Canonical SMILES of the ligand |
| `smiles_key` | string | `molecule_key` of the SMILES. Grouping and embedding key |
| `smiles_hash` | string | Hash of `smiles_key` |
| `organism` | string | Source organism of the target protein |
| `temp` | float | Temperature in Kelvin. Only present if `keep_temperature` is `true`. Missing values are 298.15 K |
| `pH` | float | pH of the measurement. Only present if `keep_ph` is `true`. Missing values are 7.0 |
| `sequence` | string | Amino acid sequence of the target protein |
| `uniprot_id` | string | UniProt accession ID |
| `value_type` | string | Type of affinity measurement (kd, ec50, ic50) |
| `monomer_id` | string | BindingDB monomer identifier |
| `ligand` | string | Ligand name |
| `ligand_id` | string | Standardized ligand identifier (priority: PubChem CID > ChEBI > KEGG > ChEMBL > ZINC, falls back to ligand name) |
| `target` | string | Target protein name |
| `doi` | string | DOI of the source publication |
| `pmid` | string | PubMed ID of the source publication |
| `ligand_target_complex_pdbs` | string | PDB IDs of ligand-target complexes |
| `pubchem_cid` | string | PubChem Compound ID |
| `pubchem_sid` | string | PubChem Substance ID |
| `chebi_id` | string | ChEBI ID |
| `chembl_id` | string | ChEMBL ID |
| `kegg_id` | string | KEGG ID |
| `zinc_id` | string | ZINC database ID |
| `pdbs` | list or string | 1D: PDB IDs of the target protein (list). 3D: structure key |
| `pdb_source`, `pdb_type`, `pdb_chain`, `pdb_coverage` | string, string, string, float | 3D files only: structure source, type, selected chain and its coverage |
| `uniprot_date` | date | UniProt creation date of the accession |
| `log10_value` | float | Log10 of the affinity value (aggregated mean) |
| `value` | float | Affinity value in millimolar (mM) (back-transformed from log10, not rounded) |
| `unit` | string | Unit of measurement ("millimolar (mM)") |

---

## Notes

1. **Unit Standardization**: All affinity values are converted to millimolar (mM) using the conversion 1 nM = 1e-6 mM.

2. **Multi-Chain Handling**: Each chain in a protein complex is treated as a separate entry, allowing the model to learn individual chain-ligand interactions.

3. **Duplicate Aggregation**: When multiple measurements exist for the same ligand-protein pair under identical conditions, values are averaged in log space to handle the typically log-normal distribution of binding affinities.

4. **Ki Exclusion**: Ki measurements are saved separately as `ki.parquet` and are not in the unified binding affinity dataset (which contains only Kd, EC50, and IC50). The kinetic parameters builder uses the Ki file.

5. **Placeholder and qualified values**: Values with a `>` or `<` qualifier, and binned or placeholder values (Step 6.5), are removed before aggregation.

6. **SMILES Canonicalization**: Ensures consistent molecular representation across the dataset, critical for proper train/test splitting and model training.

7. **Sequence Requirement**: Entries without retrievable protein sequences from UniProt are excluded, as sequences are required for downstream modeling.
