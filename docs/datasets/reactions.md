# Unified Reactions Dataset

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the unified reactions dataset pipeline implemented in `src/data/components/reactions_dataset.py`.

## Overview

The reactions dataset combines and deduplicates reactions from five major biochemical databases:

1. **Rhea**: Manually curated reactions from UniProt
2. **KEGG**: KEGG Reaction database
3. **SABIO-RK**: SABIO reaction kinetic database
4. **MetaCyc**: MetaCyc pathway reactions
5. **BRENDA**: BRENDA reactions with structures (`brenda_reactions.parquet`)

The pipeline:
- Unifies reactions across sources
- Removes duplicates using hierarchical priority (Rhea > KEGG > SABIO > MetaCyc > BRENDA)
- Balances chemical equations
- Associates enzyme information with reactions
- Handles mutant variant information
- Adds a reaction key (`rxn_key`) as the reaction identity
- Saves a 1D file, a 3D file and a reaction outcome file

---

## Processing Pipeline

### Step 1: Parse Rhea Reactions
Read rhea_reactions.parquet. Add source='rhea' and balance_status='balanced'.

### Step 2: Parse KEGG Reactions & Remove Rhea Overlap
1. Read kegg_reactions.parquet
2. Compute rhea_kegg_overlap dict from rhea_reactions[kegg_id] to rhea_id
3. Filter KEGG to exclude: rows where kegg_id in overlap dict OR rhea_id in rhea_reactions[rhea_id]
4. Map UniProt IDs on EC, explode uniprot_id
5. Fetch sequences and merge with KEGG data on uniprot_id
6. Add source='kegg', balance_status='balanced'
7. Return KEGG without overlap to Rhea

### Step 3: Parse SABIO Reactions & Update Overlaps
1. Read sabio_reactions.parquet
2. Compute kegg_sabio_overlap dict from sabio_reactions[kegg_id] to sabio_id
3. Extract pubmed_id/sabio_id from overlaps, map back to kegg_reactions(UPDATE)
4. Map sabio_id back to rhea_reactions via kegg dict (UPDATE)
5. Filter SABIO to exclude where kegg_id in overlap dict
6. Parallel balance equations
7. Map UniProt IDs where EC exists but uniprot_id missing
8. Fetch sequences and merge with SABIO data
9. Handle mutants: apply mutate_sequence(), drop invalid rows (mutation failed)
10. Add source='sabio'
11. Return (updated rhea_reactions, updated kegg_reactions, sabio_reactions_no_overlap)

### Step 4: Unify Rhea, KEGG, SABIO
Concatenate into master_reactions_df.

### Step 5: Add MetaCyc Reactions & Cross-Reference Mapping
Runs only when `include_metacyc` is `true` (default `false`). When it is `false`, the step logs that MetaCyc is excluded and keeps `metacyc_id` from the Rhea cross-references (null if absent). The MetaCyc statistics in `log_stats` are skipped when there are no MetaCyc rows.
1. Read metacyc_reactions.parquet, rename columns to standardized names
2. Compute rhea_metacyc_overlap and kegg_metacyc_overlap dicts
3. Map metanetx_id to master_reactions via rhea_id and kegg_id
4. Map metacyc_id to master_reactions where applicable
5. Filter MetaCyc to exclude rhea and kegg overlaps
6. Normalize `balance_status` column (convert to lowercase, remove colons)
7. Map UniProt IDs where EC but no sequence
8. Fetch sequences and merge into MetaCyc data
9. Concatenate to master_reactions_df, add source='metacyc'
10. Return updated master_reactions_df

### Step 5b: Normalize EC Numbers
Read from `setup()`. Each `ec` value is normalized with `normalize_ec_collection` (fallback `-.-.-.-`). Mixed strings are exploded to one row per EC. Partial ECs are padded to four levels (for example `1.1.-` becomes `1.1.-.-`).

### Step 5c: Canonicalize Reactions and Compute `rxn_key`
1. `canonicalize_rxn_smiles_column` canonicalizes `rxn_smiles`
2. `rxn_smiles` equal to `>>` (no substrates or products) is set to null
3. `rxn_key` is computed with `reaction_key_series_with_fallback`. The key removes atom maps, normalizes the components (`molecule_key`), drops spectators (water, proton) and sorts the components. Orientation and stereo stay. If the key fails, the row gets its canonical `rxn_smiles` as key. `rxn_smiles` stays for display. `rxn_key` is the identity for deduplication, splits and DRFP

### Step 5d: Add BRENDA Reactions
1. Read brenda_reactions.parquet. Drop rows with no `rxn_smiles`, then canonicalize it and compute `rxn_key` (tautomer-robust, so amides from InChI match)
2. Keep IUBMB and literature reactions. Carry the `is_iubmb` flag and the `brenda_rxn_id`. Both are null for other sources
3. A BRENDA reaction whose `rxn_key` exists already keeps the existing row. Only its new (reaction, UniProt) pairs are added, as enzyme rows that copy the reaction columns of the existing row and keep its `source`
4. Other BRENDA reactions become new rows with source='brenda'. Their formula equation comes from `rxn_smiles`, and `balance_equations` sets `balance_status`. Unbalanced reactions are kept
5. Fetch UniProt sequences and PDBs for the added accessions

### Step 6: Log Statistics
Log counts per source: unique reactions, unique EC/UniProt, reactions with/without rxn_smiles, percentages, contributions.

### Step 7: Homogenize Data Types
For [pubmed_id, reactome_id, pdbs, sabio_id, metanetx_id]: convert numpy arrays to lists, scalars to [str(x)], nulls to [].

### Step 8: Pickle Unique Reaction SMILES
Extract unique rxn_smiles (non-metacyc sources), pickle to file.

### Step 9: Pickle Unique Sequences
Extract unique (uniprot_id, sequence) pairs as {acc_id, sequence} dicts, pickle to file.

### Step 10: Add UniProt Dates and Atom Maps
1. `add_uniprot_date_column` adds `uniprot_date`
2. If `atom_mapping` is `true`, `add_atom_mapping_columns` maps each unique `rxn_smiles` once with LocalMapper and adds `atom_mapped_rxn_smiles`, `atom_mapping_status` and `atom_mapping_confident`. Results are cached in `atom_mapping_cache_parquet_file_path`. Failed reactions are tried again on the next run and listed in `atom_mapping_failures_parquet_file_path`. The step needs the `localmapper` package. When `atom_mapping` is `false`, the step is skipped

### Step 11: Save Reaction Outcome Dataset
`save_reaction_outcome_dataset` keeps one row for each `rxn_key`: the row with the smallest `rxn_smiles`, with the earliest `uniprot_date` of the key. Columns: `rxn_key`, `rxn_smiles`, `uniprot_date`, the atom-mapping columns (if present), and `reactants` and `products` (the two sides of `rxn_smiles`). It is saved to `reaction_outcome_dataset_parquet_file_path` with zstd level 3.

### Step 12: Save Reactions Parquet
1. 1D (`save_1d_dataset`): `master_reactions_df` is saved to `unified_reactions_1d.parquet` (zstd level 3).
2. 3D (`save_3d_dataset`): `assign_experimental_and_af_pdbs` adds experimental structure chains (`pdbs` as `<PDB>:<base accession>`, `pdb_chain`, `pdb_coverage`, `pdb_source`, `pdb_type`). A row without a selected chain gets one fallback row with a predicted structure. That function downloads the missing AlphaFold files (`get_af_pdb`). The result is saved to `unified_reactions_3d.parquet`.

At least one of the two must be `true`. If both are `true`, the two files are saved in parallel.

### Step 13: Download and Parse References
Download all unique pubmed_ids in parallel. Parse all reference XMLs. Save `master_references_df` to `unified_reactions_references.parquet`.

---

## Configuration

The builder keys are in `configs/data/reactions_dataset.yaml`. The output paths that the split configs read are in `configs/data/split_inputs/reactions_dataset.yaml`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `rhea_reactions_parquet_file_path` | str | Path to input Rhea reactions parquet |
| `kegg_reactions_parquet_file_path` | str | Path to input KEGG reactions parquet |
| `sabio_reactions_parquet_file_path` | str | Path to input SABIO reactions parquet |
| `include_metacyc` | bool | Merge MetaCyc reactions when `true`. Default `false`: the MetaCyc parquet is not read and need not exist. |
| `metacyc_reactions_parquet_file_path` | str | Path to input MetaCyc reactions parquet (used only when `include_metacyc` is `true`) |
| `brenda_reactions_parquet_file_path` | str | Path to input BRENDA reactions parquet |
| `balance_cache_dir` | str | Folder for the balanced-equation cache. Default `<intermediate_data_dir>/balance_eqn` |
| `save_1d_dataset`, `save_3d_dataset` | bool | Default `true` for both. At least one must be `true` |
| `atom_mapping` | bool | Default `false`. Add LocalMapper atom-mapping columns when `true` |
| `atom_mapping_cache_parquet_file_path` | str | Cache with one row for each unique `rxn_smiles` |
| `atom_mapping_failures_parquet_file_path` | str | Rows of the cache that failed |
| `unified_reactions_references_parquet_file_path` | str | Output path for references |
| `unified_reactions_sequences_pickle_path` | str | Output path for sequence pickle |
| `unified_reactions_rxn_smiles_pickle_path` | str | Output path for rxn_smiles pickle |
| `reactions_dataset_path`, `reaction_outcome_dataset_path` | str | Output folders (in `split_inputs`) |
| `unified_reactions_1d_parquet_file_path` | str | Output path for the 1D unified reactions: `<reactions_dataset_path>/unified_reactions_1d.parquet` (in `split_inputs`) |
| `unified_reactions_3d_parquet_file_path` | str | Output path for the 3D unified reactions: `<reactions_dataset_path>/unified_reactions_3d.parquet` (in `split_inputs`) |
| `reaction_outcome_dataset_parquet_file_path` | str | Output path for the reaction outcome dataset: `<reaction_outcome_dataset_path>/reaction_outcome_dataset.parquet` (in `split_inputs`) |
| `log_dir`, `log_file_name` | str | Log location (`unify_reactions.log`) |

---

## Methods

### `ReactionsDatasetBuilder`

#### `__init__(cfg: DictConfig)`
Store the input parquet paths and the output paths. Create output directories. Initialize logger. Check that the Rhea, KEGG, SABIO and BRENDA files exist, and the MetaCyc file if `include_metacyc` is `true` (raise FileNotFoundError if missing).

#### `setup() -> Tuple[pd.DataFrame, pd.DataFrame]`
Orchestrate full pipeline: parse rhea → parse kegg (remove rhea overlap) → parse sabio (remove kegg overlap, balance, update rhea/kegg) → concat master → add metacyc (remove rhea+kegg overlap) → normalize and explode EC → canonicalize `rxn_smiles` → compute `rxn_key` → add brenda → log stats → homogenize types → pickle smiles/sequences → add `uniprot_date` → atom mapping (optional) → save reaction outcome dataset → save 1D and 3D parquet → download PubMed/references. Return (3D or 1D `master_reactions_df`, `master_references_df`).

#### `parse_rhea_reactions() -> pd.DataFrame`
Read rhea_reactions.parquet. Set source='rhea', balance_status='balanced'. Return rhea_df.

#### `parse_kegg_reactions(rhea_reactions: pd.DataFrame) -> pd.DataFrame`
1. Read kegg_reactions.parquet
2. Compute rhea_kegg_overlap dict from rhea_reactions[kegg_id]
3. Filter kegg: exclude kegg_id in overlap dict, exclude rhea_id in rhea_reactions[rhea_id]
4. Map UniProt IDs via get_uniprot_ec_acc_map(ec), explode, fetch sequences via get_uniprot_acc_json()
5. Set source='kegg', balance_status='balanced'
6. Return kegg_reactions_no_rhea_overlap

#### `parse_sabio_reactions(rhea_reactions: pd.DataFrame, kegg_reactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
1. Read sabio_reactions.parquet
2. Compute kegg_sabio_overlap dict
3. Extract pubmed_id/sabio_id from overlaps, map to kegg_reactions (UPDATE), map sabio_id to rhea_reactions (UPDATE)
4. Filter sabio: exclude kegg_id in overlap dict
5. Balance equations via balance_equations() (cached per unique equation)
6. Map UniProt IDs where EC exists but uniprot_id missing
7. Fetch sequences, merge
8. Handle mutants: call handle_sabio_reaction_mutant_seqs()
9. Set source='sabio'
10. Return (updated rhea_reactions, updated kegg_reactions, sabio_reactions_no_kegg_overlap)

#### `balance_equations(df: pd.DataFrame) -> pd.DataFrame`
Balance the 'equation' column. Each unique equation is balanced once by `balance_equation` and the result is stored in `data/intermediate/balance_eqn/balance_eqn_v<BALANCE_CACHE_VERSION>.parquet` (config key `balance_cache_dir`). Later runs balance only equations missing from the cache. Raise `BALANCE_CACHE_VERSION` when the balancing logic changes. Rows without an equation get balance_status='undetermined'.

#### `balance_equation(eqn: str) -> Tuple[str, str]`
Parse via chemeq(eqn.replace("<=>", "=")). If not balanced, call balance(). Return the balanced equation and 'balanced'. On error, return the input equation and 'unbalanced-unfixable'.

#### `handle_sabio_reaction_mutant_seqs(df: pd.DataFrame) -> pd.DataFrame`
Apply mutate_sequence() parallely to rows with mutations. Drop rows where mutations.notna() and sequence.isna(). Return df.

#### `add_metacyc_reactions(master_reactions_df: pd.DataFrame) -> pd.DataFrame`
1. Read metacyc_reactions.parquet, rename columns (UNIQUE-ID→metacyc_id, ATOM-MAPPED-SMILES→rxn_smiles, EC-NUMBER→ec, SEQUENCE→sequence, etc.)
2. Compute rhea_metacyc_overlap and kegg_metacyc_overlap dicts
3. Map metanetx_id to master_reactions via rhea_id and kegg_id
4. Map metacyc_id where applicable
5. Filter metacyc: exclude rhea_metacyc_overlap and kegg_metacyc_overlap keys
6. Normalize balance_status (.lower().replace(":", ""))
7. Map UniProt IDs where EC but no sequence
8. Fetch sequences, merge
9. Concatenate to master_reactions_df, set source='metacyc'
10. Return updated master_reactions_df

#### `add_brenda_reactions(master_reactions_df: pd.DataFrame) -> pd.DataFrame`
Merge BRENDA by `rxn_key`: add new UniProt pairs to matched reactions, add unmatched reactions as source='brenda', drop rows without `rxn_smiles`, set `balance_status` with `balance_equations`. See Step 5b.

#### `pickle_all_rxn_smiles(master_reactions_df: pd.DataFrame) -> None`
Extract unique rxn_smiles (where source != 'metacyc'), pickle to UNIFIED_REACTIONS_RXN_SMILES_PICKLE_PATH.

#### `pickle_all_sequences(master_reactions_df: pd.DataFrame) -> None`
Extract unique (uniprot_id→acc_id, sequence) pairs as list of dicts, pickle to UNIFIED_REACTIONS_SEQUENCES_PICKLE_PATH.

#### `canonicalize_rxn_smiles_column(master_reactions_df: pd.DataFrame) -> pd.DataFrame`
Canonicalize the `rxn_smiles` column with `canonicalize_reaction_smiles_series`.

#### `homogenize_list_columns(master_reactions_df: pd.DataFrame) -> pd.DataFrame`
Make the list-valued ID columns hold lists of strings (see Step 7).

#### `add_atom_mapping_columns(reactions_df: pd.DataFrame) -> pd.DataFrame`
Map each unique `rxn_smiles` once and join the mapping columns back (see Step 10).

#### `save_reaction_outcome_dataset(reactions_df: pd.DataFrame) -> pd.DataFrame`
Save one row for each `rxn_key` (see Step 11).

#### `log_stats(master_reactions_df: pd.DataFrame) -> None`
Log per-source stats: unique reactions (rhea dividing id counts by 4), unique EC/UniProt, reactions with/without rxn_smiles, percentages, source contributions.

---

## Data Schema

### Output: Unified Reactions (master_reactions_df)
| Column | Type | Description |
|--------|------|-------------|
| rhea_id | list[str] or str | Rhea reaction ID(s) |
| kegg_id | str | KEGG reaction ID |
| sabio_id | list[str] | SABIO reaction ID(s) |
| metacyc_id | str | MetaCyc reaction ID |
| source | str | Primary source (rhea/kegg/sabio/metacyc/brenda) |
| ec | str or list[str] | EC number(s) |
| rxn_smiles | str | Canonical reaction SMILES (for display) |
| rxn_key | str | Reaction identity key (see Step 5c) |
| uniprot_date | date | UniProt creation date of the accession |
| definition | str | Reaction definition with compound names |
| equation | str | Reaction equation with formulas |
| direction | str | Reaction direction (REVERSIBLE, LEFT-TO-RIGHT, RIGHT-TO-LEFT) |
| uniprot_id | str | UniProt accession ID |
| sequence | str | Protein sequence |
| organism | str | Source organism |
| species | str | NCBI taxonomy |
| gene_name | str | Gene name |
| mutations | tuple or str | Mutation info if mutant |
| balance_status | str | balanced/unbalanced-unfixable/undetermined |
| pubmed_id | list[str] | PubMed IDs |
| reactome_id | list[str] | Reactome IDs |
| pdbs | list[str] | PDB IDs |
| metanetx_id | list[str] | MetaNetX reaction IDs |
| spontaneous | bool or str | Spontaneity (MetaCyc) |
| brenda_rxn_id | str | BRENDA reaction ID of a BRENDA-derived row, else null |
| is_iubmb | bool | True for an IUBMB BRENDA reaction, False for a literature one, null for other rows |
| atom_mapped_rxn_smiles, atom_mapping_status, atom_mapping_confident | str, str, bool | Only when `atom_mapping` is `true` |
| pdb_chain, pdb_coverage, pdb_source, pdb_type | str, float, str, str | 3D file only. `pdbs` is the structure key there |

---

## Deduplication Strategy

### Incremental Overlap Removal
Databases processed sequentially, each removing overlaps with previously processed ones:

1. **KEGG vs Rhea**: Compute overlap dict from rhea_reactions[kegg_id]. Filter kegg_reactions to exclude both direct kegg_id matches AND rhea_id matches in Rhea.
2. **SABIO vs KEGG**: Compute overlap dict from sabio_reactions[kegg_id]. Map pubmed_id/sabio_id back to kegg and rhea reactions (enrichment). Filter sabio_reactions to exclude kegg_id matches.
3. **MetaCyc vs Rhea+KEGG**: Compute overlap dicts from metacyc_reactions[rhea_id] and [kegg_id]. Map metanetx_id/metacyc_id back (enrichment). Filter metacyc to exclude both rhea and kegg overlaps.

### Enrichment via Cross-Reference Mapping
All overlap information is mapped back to previous DataFrames as new columns (e.g., sabio_id added to kegg/rhea).

---

## Chemical Equation Balancing

Applied to SABIO reactions without KEGG overlap and to new BRENDA reactions via `balance_equations()`:

1. Take the unique 'equation' values (formula expressions) that the cache does not hold
2. Parse via chemeq(eqn.replace("<=>", "="))
3. Check if balanced; if not, call balance()
4. Store the balanced equation with balance_status='balanced'
5. On any error, store the input equation with balance_status='unbalanced-unfixable'
6. Append the results to the balance cache and map them to all rows

Example:
```
Input:  C6H12O6 <=> C2H5OH + CO2
Parsed: C6H12O6 = C2H5OH + CO2
Balanced: 1*C6H12O6 = 2*C2H5OH + 2*CO2
Output: 1*C6H12O6 <=> 2*C2H5OH + 2*CO2
```

---

## UniProt Sequence Mapping

Whenever a source provides EC numbers without UniProt IDs:

1. Call get_uniprot_ec_acc_map(ec_list) to get dict mapping EC to list[uniprot_ids]
2. Map EC to UniProt IDs, explode to 1 row per ID
3. Call get_uniprot_acc_json(uniprot_ids) to fetch {acc_id, sequence, organism, ...}
4. Merge on uniprot_id to add sequence, organism columns

Applied in: KEGG parse (all reactions), SABIO parse (reactions missing uniprot_id but having EC), MetaCyc parse (same).

---

## Mutant Sequence Handling

Only for SABIO reactions with mutations field:

1. Rename uniprot_id to UniprotID (for ChemUtils compatibility)
2. Apply mutate_sequence() row-wise: takes mutations tuple, applies to sequence
3. Rename UniprotID back to uniprot_id
4. Drop rows where mutations.notna() but sequence.isna() (invalid mutations)

---

## Implementation Details

### Rhea Reactions Special Case
Rhea represents each reaction as 4 rows (different structural variants with same reaction ID). The log_stats() method divides rhea_id counts by 4 to get true unique reactions.

### SABIO-KEGG Overlap Enrichment
When SABIO reactions overlap with KEGG:
- Extract all pubmed_id values from overlapping SABIO entries
- Map back to kegg_reactions via kegg_id (many-to-many)
- This gives KEGG reactions additional PubMed references from literature-curated SABIO
- Also map sabio_id list back to kegg and rhea reactions for cross-reference tracking

### Balance Status Values
- balanced: Equation successfully balanced via chemeq
- unbalanced-unfixable: Chemeq parsing or balancing failed
- undetermined: No equation provided (rhea_reactions default)

### Null Reaction SMILES Handling
Empty rxn_smiles=">>" (no substrates or products) are set to None for consistency.

### Pickle Outputs
- **rxn_smiles pickle**: List of unique reaction SMILES strings, excluding MetaCyc sources (MetaCyc SMILES are atom-mapped)
- **sequences pickle**: List of dicts {acc_id: ..., sequence: ...} for unique UniProt IDs, used for embedding generation

### PubMed & Reference Download
- All unique pubmed_ids extracted and downloaded in parallel via download_pubmed_abstracts_parallel()
- Abstracts stored by ID, retrievable by pubmed_id column
- All reference XMLs from upstream builders parsed via parse_all_reference_xmls() to create master_references_df

### Structures (3D file)
- `assign_experimental_and_af_pdbs` selects experimental chains and, for rows without one, adds a predicted-structure row. It downloads the missing AlphaFold files (`get_af_pdb`) for those rows.
- `setup()` has no separate AlphaFold download step.

### Data Type Homogenization
Columns [pubmed_id, reactome_id, pdbs, sabio_id, metanetx_id] standardized to list[str] format:
- numpy ndarrays converted to list of strings
- scalar values converted to [str(value)] if not null, [] if null
- Already-lists preserved as-is

This ensures consistency across different source databases which may return arrays, scalars, or lists.
