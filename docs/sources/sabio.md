# SABIO-RK Kinetic Parameters Database

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the SABIO-RK database integration and kinetic parameter extraction pipeline implemented in `src/data/components/sabio.py`.

## Overview

The `SABIODatasetBuilder` class downloads and processes reactions, compounds, and kinetic parameters from the SABIO-RK Export API. Key workflow:

1. **Download**: Fetch kinetic law entries, reactions, and compounds as paged JSON caches
2. **Entries**: Parse entry metadata (EC number, UniProt accessions, mutations)
3. **Compounds**: Parse compounds, with fallback to ChEBI/KEGG/PubChem for missing SMILES
4. **Reactions**: Build reaction definitions and reaction SMILES, joined with entries
5. **Kinetic Parameters**: Extract kcat, Km, Ki from the entries with normalized units
6. **Output**: Save reactions, compounds, entries, and kinetic parameters as parquet files

---

## Processing Pipeline

### Step 1: Download Paged Data
**Method**: `process_reactions_compounds()` -> `download_pages(endpoint, folder, query)`

**Process**:
1. Download three endpoints under `sabio_api_url`:
   - `kinlaw-entry/json` with query `ECNumber:*` -> `SABIO_ENTRIES_PATH`
   - `reaction` -> `SABIO_REACTIONS_PATH`
   - `compound` -> `SABIO_COMPOUNDS_PATH`
2. Request the total count first and write it to `total_count.txt` in the folder
3. If the total changed since the last run, discard the cached pages
4. Download each page of `page_size` rows to `page_NNNN.json` and skip pages that exist

**Method**: `request_json(endpoint, params)`

- Waits `request_interval` seconds before each request (limit: 60 requests per minute)
- Retries up to `max_retries` times with exponential backoff on HTTP 429, 5xx, and network errors
- Raises an error if the reply is not JSON

**Output**: `page_NNNN.json` files in each folder. `read_pages(folder)` yields their rows in order.

### Step 2: Parse Entries
**Method**: `parse_entry(entry)`

**Process**:
1. Read the EC number, reaction ID, PubMed ID, and KEGG and Reactome reaction IDs
2. Collect the UniProtKB accessions from the `kinlaw_entry` links (semicolon-joined)
3. Extract mutations from `mutant_spec` with regex `\b[A-Za-z]\d+[A-Za-z]\b`; wildtype entries have none
4. Normalize EC numbers with `normalize_ec_collection()` and explode to one row per EC number

**Output**: SABIO_entries DataFrame (columns: kegg_id, sabio_id, ec, UniProtKB_AC, pubmed_id, reactome_id, mutations)

### Step 3: Parse Compounds
**Method**: `parse_sabio_compound(compound)`

**Process** (for each compound used by a reaction that has an entry):
1. Extract SABIO ID, name, ChEBI IDs, PubChem IDs, KEGG IDs from the external links
2. **SMILES acquisition** (in priority order):
   - If SMILES or InChI is provided: build the mol with RDKit and fill the missing one of the two
   - Else if ChEBI IDs are available: `download_chebi_json()` and `parse_chebi_jsons()`
   - Else if KEGG IDs are available: `download_kegg_compound()`
   - Else if PubChem IDs are available: `get_pubchem_compound()`
3. Compute the molecular formula from the mol if available
4. Write a MOL file `SABIO_{sabio_id}.mol` to `mol_path` if SMILES is available

**Output**: SABIO_compounds DataFrame (columns: sabio_id, name, formula, chebi_id, pubchem_id, kegg_id, inchi, smiles)

### Step 4: Build Reactions
**Method**: `parse_sabio_reaction(reaction, compounds)`

**Process**:
1. Split the `compound_in_reaction` participants by role (Substrate or Product)
2. If all participants have SMILES, build the definition, equation, and reaction SMILES
3. Merge with entries on `sabio_id`
4. Split `UniProtKB_AC` into a list and explode to one row per protein per reaction
5. Select columns: sabio_id, ec, definition, equation, rxn_smiles, uniprot_id, kegg_id, reactome_id, pubmed_id, mutations
6. Drop duplicate rows

**Output**: SABIO_reactions DataFrame (one row per reaction-protein combination)

### Step 5: Extract Kinetic Parameters
**Method**: `process_kinetic_params(SABIO_compounds)` -> `parse_kinetic_params(entry)`

**Process** (for each cached entry):
1. Read the UniProt accessions and the EC number
2. Set enzymeType: "wildtype" if the entry is wildtype, else "mutant"
   - Mutations come from `mutant_spec` with regex `\b[A-Z]\d+[A-Z]\b`
   - Skip mutant entries with no parsed mutation
3. Take the temperature from the experimental conditions and convert Celsius to Kelvin
4. For each parameter of type kcat, Km, or Ki:
   - Use `n_end_value` if present, otherwise `n_start_value`
   - Keep the parameter only if the normalized unit (`unit.n_name`) is "s^(-1)" for kcat or "M" for Km/Ki
   - kcat: one row for each substrate of the reaction
   - Km/Ki: one row for the substrate named in the parameter species key; multiply the value by 1000 (M -> mM)
   - One row for each UniProt accession
5. Drop duplicate rows

### Step 6: Add SMILES and Save
**Method**: `process_kinetic_params()` (final steps)

**Process**:
1. Join with SABIO_compounds on substrate name
2. Add columns: substrate_id (prefixed with "SABIO:"), smiles
3. Sort rows for a stable order
4. Save to `sabio_kinetic_params_parquet_file_path`

**Output**: Final kinetic parameters DataFrame with SMILES

---

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `sabio_rk.sabio_api_url` | str | SABIO-RK Export API base URL |
| `sabio_rk.page_size` | int | Rows per page (the API maximum, 1000) |
| `sabio_rk.request_interval` | float | Seconds to wait before each request |
| `sabio_rk.max_retries` | int | Attempts per request |
| `sabio_rk.sabio_data_path` | str | Root directory for processed SABIO data |
| `sabio_rk.sabio_raw_path` | str | Directory for raw downloads |
| `sabio_rk.sabio_entries_path` | str | Directory for entry pages |
| `sabio_rk.sabio_compounds_path` | str | Directory for compound pages |
| `sabio_rk.sabio_reactions_path` | str | Directory for reaction pages |
| `sabio_rk.mol_path` | str | Directory for MOL structure files |
| `sabio_rk.kegg_compound_path` | str | Directory for KEGG compound files |
| `sabio_rk.sabio_entries_parquet_file_path` | str | Output: entries parquet |
| `sabio_rk.sabio_reactions_parquet_file_path` | str | Output: reactions parquet |
| `sabio_rk.sabio_compounds_parquet_file_path` | str | Output: compounds parquet |
| `sabio_rk.sabio_kinetic_params_parquet_file_path` | str | Output: kinetic params parquet |
| `sabio_rk.log_dir` | str | Directory for log files |
| `sabio_rk.log_file_name` | str | Log file name |

---

## Main Methods

### `SABIODatasetBuilder`

#### `__init__(cfg: DictConfig)`
1. Store all configuration paths and API settings from the Hydra config
2. Initialize KEGGDatasetBuilder for KEGG compound fallback
3. Create all output directories
4. Initialize logger

#### `setup() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]`
1. Call `process_reactions_compounds()` -> returns SABIO_reactions, SABIO_compounds, SABIO_entries
2. Call `process_kinetic_params(SABIO_compounds)` -> returns SABIO_kinetic_params
3. Return all four DataFrames

#### `request_json(endpoint, params)`
GET one JSON reply with pacing and retries (see Step 1).

#### `download_pages(endpoint, folder, query=None)`
Cache all pages of an endpoint (see Step 1).

#### `read_pages(folder)`
Yield the rows of all cached pages, sorted by page file name.

#### `parse_sabio_compound(compound) -> dict`
Parse one compound and get its SMILES (see Step 3).

#### `parse_sabio_reaction(reaction, compounds) -> dict | None`
Build one reaction row from its participants (see Step 4). Returns None if the reaction has no substrate or product.

#### `process_reactions_compounds() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
1. Download the entry, reaction, and compound pages
2. Parse entries and explode EC numbers
3. Parse compounds of the reactions that have an entry
4. Build reactions and merge with entries
5. Save SABIO_reactions, SABIO_compounds, SABIO_entries to parquet

#### `process_kinetic_params(SABIO_compounds: pd.DataFrame) -> pd.DataFrame`
Extract, join with compounds, and save kinetic parameters (see Steps 5 and 6).

### Module functions

- `parse_entry(entry)`: entry-level annotation row
- `parse_kinetic_params(entry)`: normalized kcat, Km, and Ki rows for one entry

---

## Data Schema

### Output: Kinetic Parameters
| Column | Type | Description |
|--------|------|-------------|
| ECNumber | str | Enzyme Commission number |
| substrate | str | Substrate name/compound |
| value | float | Kinetic parameter value (normalized) |
| Organism | str | Source organism |
| UniprotID | str | UniProt accession ID |
| Temperature | float | Temperature in K (NaN if not available) |
| enzymeType | str | "wildtype" or "mutant" |
| mutations | tuple | Tuple of mutations (e.g., ("A123B", "C456D")) or NaN |
| unit | str | "s^(-1)" for kcat, "millimolar (mM)" for Km/Ki |
| value_type | str | "kcat", "Km", or "Ki" |
| substrate_id | str | SABIO substrate ID with prefix (e.g., "SABIO:12345") |
| smiles | str | Canonical SMILES of substrate (or NaN) |

### Output: Compounds
| Column | Type | Description |
|--------|------|-------------|
| sabio_id | str | SABIO compound ID |
| name | str | Compound name |
| formula | str | Molecular formula |
| chebi_id | list | ChEBI IDs (if available) |
| pubchem_id | list | PubChem IDs (if available) |
| kegg_id | list | KEGG IDs (if available) |
| inchi | str | InChI string |
| smiles | str | Canonical SMILES |
| charge | int | Formal charge |
| mass | float | Molecular mass |
| monoisotopic_mass | float | Monoisotopic mass |
| secondary_ids | list | Alternative SABIO IDs |
| synonyms | list | Common names/synonyms |
| iupac_names | list | IUPAC names |
| formulae | list | Molecular formulae |
| xrefs | dict | Cross-references to other databases |
| citations | dict | Citation links |

### Output: Reactions
| Column | Type | Description |
|--------|------|-------------|
| sabio_id | str | SABIO reaction ID |
| ec | str | EC number |
| definition | str | Reaction definition/name |
| equation | str | Balanced chemical equation |
| rxn_smiles | str | Reaction SMILES (SMILES>>products) |
| uniprot_id | list | UniProt IDs of enzymes (exploded to one per row) |
| kegg_id | str | KEGG reaction ID (if available) |
| reactome_id | str | Reactome reaction ID (if available) |
| pubmed_id | str | PubMed reference ID |
| mutations | tuple | Enzyme mutations (if mutant) |

### Output: Entries
| Column | Type | Description |
|--------|------|-------------|
| UniProtKB_AC | str | UniProt accession IDs (semicolon-separated) |
| SabioReactionID | str | SABIO reaction ID |
| KeggReactionID | str | KEGG reaction ID |
| ECNumber | str | EC number |
| PubMedID | str | PubMed ID |
| ReactomeReactionID | str | Reactome reaction ID |
| mutations | tuple | Extracted mutations (or None) |

---

## Unit Conversions

### kcat (Turnover Number)
- **Standard unit**: s^(-1) (per second)
- **SABIO requirement**: The normalized unit must be "s^(-1)". Values with other units are filtered out

### Km, Ki (Binding Constants)
- **Standard unit**: mM (millimolar)
- **SABIO requirement**: The normalized unit must be "M" (molar). Values with other units are filtered out
- Molar is converted to millimolar.

---

## Error Handling

1. **Missing SMILES**: Fall back to ChEBI, KEGG, then PubChem
2. **Invalid Units**: Skip the parameter
3. **API Errors**: Retry with exponential backoff, then raise
4. **Non-JSON Reply**: Raise an error

---

## Notes

1. **Compound SMILES Enrichment**: Falls back through ChEBI, KEGG, PubChem when SABIO/InChI unavailable

2. **Substrate Name String Matching**: Final merge uses substrate NAME (not ID) to join kinetic params with compounds

3. **Organism**: The organism comes from the kinetic law entry

4. **SMILES Carryover**: Compounds without SMILES retain NaN in final output; rows not filtered

5. **Output Row Duplication**: From multiple measurements per enzyme-substrate, multiple UniProts per entry, and kcat substrate explosion
