# KEGG Database Integration

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the KEGG data download and processing pipeline implemented in `src/data/components/kegg.py`.

## Overview

The KEGG module downloads reactions and compounds from KEGG REST API, enriches with structures, and generates reaction SMILES:

1. Downloads reaction list (tab-separated IDs) from KEGG API
2. Parses all reactions: ENTRY, EQUATION (metabolite IDs), EC, PMID, DBLINKS (Rhea mapping)
3. Identifies and replaces glycan IDs (G*) with compound IDs (C*) by modifying JSON files
4. Downloads all compounds referenced in reactions: ENTRY, NAME, FORMULA, MOL file → SMILES
5. Fills missing SMILES by querying PubChem via compound cross-references
6. Generates reaction SMILES from substrate/product SMILES with stoichiometry
7. Generates equations with molecular formulae
8. Explodes EC and Rhea ID columns to one-to-many mapping
9. Saves two independent parquet files: reactions + compounds

---

## Processing Pipeline

### Step 1: Download KEGG Reaction List
Query KEGG REST API, returns tab-separated IDs on each line. Extract reaction IDs. Download all reactions (batched in groups of 10, processed with ThreadPool(2)).

### Step 2: Parse Reactions
For each reaction text block from API:
- Extract ENTRY: reaction ID (e.g., "R00001")
- Extract EQUATION: text like "C00001 + C00002 => C00003" (uses => for LEFT-TO-RIGHT, <=> for REVERSIBLE)
- Parse equation via `parse_equation()` utility → substrates [[[coeff, id], ...], ...], direction, products [...]
- Extract ENZYME: EC numbers as space-separated list
- Extract DBLINKS: find "RHEA:" entries, format as "RHEA:{ID}"
- Extract REFERENCE: find all PMID and DOI tags
Return dict with kegg_id, definition, substrates, direction, products, ec, pmids, dois, rhea_id

### Step 3: Identify & Download Glycans
Extract all unique glycan IDs (IDs starting with "G") from substrates/products across all reactions. Download glycan metadata: ENTRY (glycan ID) and REMARK "Same as: C*" (compound ID mapping). Create dict {G_id: C_id}.

### Step 4: Replace Glycan IDs with Compound IDs
For each reaction JSON file in KEGG_REACTION_PATH:
- For each substrate and product tuple: if the ID starts with "G", replace with mapped compound ID
- Save updated JSON
Reload all reaction JSONs into DataFrame.

### Step 5: Download Compounds
Extract all unique compound IDs (starting with "C") from final reaction substrates/products. Download compound metadata:
- Download MOL file via `get_mol_kegg()` utility
- Convert MOL → SMILES using RDKit
- Extract NAME, FORMULA (or calculate from MOL)
- Extract DBLINKS (PubChem, etc.)
Return list of dicts with kegg_id, name, formula, smiles, dblinks

### Step 6: Fill Missing SMILES with PubChem
For compounds with missing SMILES:
- Extract PubChem ID from dblinks["PubChem"]
- Query PubChem API to get SMILES + formula
- Update compounds DataFrame

### Step 7: Generate Reaction SMILES
For each reaction:
- Map substrate/product compound IDs to SMILES using smiles_dict
- Call `generate_reaction_SMILES()` utility → "SMILES.SMILES>>SMILES.SMILES"
Store in rxn_smiles column.

### Step 8: Generate Equation with Formulae
For each reaction:
- Look up molecular formula from compounds DataFrame (kegg_id → formula)
- Combine with stoichiometry: "coeff formula + coeff formula = ..."
Store in equation column (or None if atleast one missing formula).

### Step 9: Explode & Reshape
- Explode ec column: one EC per row (multiple ECs become multiple rows)
- Explode rhea_id column: one Rhea ID per row
- Drop columns: substrates, products, dois
- Rename: pmids → pubmed_id

### Step 10: Output
Save two independent parquet files:
- kegg_reactions: all reaction data points after explosion of EC and Rhea ID
- kegg_compounds: deduplicated compound metadata

---

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `kegg.kegg_reaction_list_url` | str | KEGG REST API for reaction list  |
| `kegg.kegg_mol_path` | str | Directory to cache downloaded MOL files |
| `kegg.kegg_data_path` | str | Directory for data files (may be unused legacy) |
| `kegg.kegg_glycan_path` | str | Directory to cache parsed glycan JSONs |
| `kegg.kegg_compound_path` | str | Directory to cache parsed compound JSONs |
| `kegg.kegg_reaction_path` | str | Directory to cache parsed reaction JSONs |
| `kegg.kegg_reactions_parquet_file_path` | str | Output parquet file for reactions |
| `kegg.kegg_compounds_parquet_file_path` | str | Output parquet file for compounds |
| `kegg.log_dir` | str | Directory for log files |
| `kegg.log_file_name` | str | Log file name |

---

## Methods

### `KEGGDatasetBuilder`

#### `__init__(cfg: DictConfig)`
Store all config paths, create directories, initialize logger.

#### `setup() -> Tuple[pd.DataFrame, pd.DataFrame]`
Main orchestration method: Downloads reaction list → downloads reactions → identifies/downloads/replaces glycans → downloads compounds → fills missing SMILES from PubChem → generates rxn_smiles → generates equations → explodes columns → saves parquet files.

Returns: (reactions_df, compounds_df)

#### `download_kegg_json(ids: list, path: Path, _type: str)`
Download batch of KEGG entries (reactions/compounds/glycans) via single REST API call. Parse using type-specific function (parse_kegg_reaction/parse_kegg_compound/parse_kegg_glycan). Save each as individual JSON file.

#### `download_kegg_data(ids: list, path: Path, _type: str, verbose: bool) -> list[dict]`
Wrapper for batch downloading. Check for existing files locally. Only download missing IDs. Batch requests in groups of 10, process with ThreadPool(2). Load all JSONs from disk. Returns list of dicts.

#### `download_kegg_reaction/compound/glycan(ids: list, verbose: bool) -> list[dict]`
Thin wrappers around `download_kegg_data()` with type-specific path and type parameter.

#### `parse_kegg_reaction(reaction: str) -> dict`
Parse single reaction text block from KEGG API. Extract ENTRY, EQUATION, ENZYME (EC), REFERENCE (PMID/DOI), DBLINKS (RHEA). Use `parse_equation()` to split equation into substrates/direction/products. Return dict: {kegg_id, rhea_id, definition, substrates, direction, products, ec, pmids, dois}. Substrates/products format: [[[coeff, id], ...], ...]

#### `parse_kegg_compound(compound: str) -> dict`
Parse compound text block. Extract ENTRY, NAME, FORMULA. Download MOL file via `get_mol_kegg()`. Convert MOL to SMILES + formula via RDKit. Parse DBLINKS. Return dict: {kegg_id, name, formula, smiles, dblinks}

#### `parse_kegg_glycan(glycan: str) -> dict`
Parse glycan text block. Extract ENTRY (glycan ID) and REMARK "Same as:" (compound ID). Return dict: {kegg_id, compound_id}

#### `fill_missing_smiles(row: pd.Series, pubchem_data: dict) -> pd.Series`
For compound with missing SMILES: look up PubChem ID in row["PubChem"]. Retrieve SMILES + formula from pubchem_data dict. Update compound JSON file and MOL file. Update row values and return.

#### `generate_equation(row: pd.Series, formula_dict: dict) -> pd.Series`
Generate equation with molecular formulae. For substrate/product tuples: extract only those with IDs starting with "G" (glycan entries). Look up formula from formula_dict. Combine: "coeff formula + coeff formula = ...". Return None if no glycan entries or missing data.

---

## Data Schema

### Output: Reactions DataFrame
| Column | Type | Description |
|--------|------|-------------|
| kegg_id | str | KEGG reaction ID (e.g., R00001) |
| definition | str | User-provided reaction description |
| substrates | list | Format: [[[coeff, id], [coeff, id], ...]] (substrate groups with stoichiometry) |
| direction | str | LEFT-TO-RIGHT, REVERSIBLE, or None |
| products | list | Format: [[[coeff, id], [coeff, id], ...]] (product groups with stoichiometry) |
| ec | str | Single EC number (after explosion, one per row) |
| pubmed_id | list | PubMed IDs (after parsing REFERENCE blocks with PMID tags) |
| rxn_smiles | str | Reaction SMILES (e.g., CCO.CC>>CCOC) or NaN if any substrate/product missing SMILES |
| equation | str | Equation with molecular formulae for glycan entries only (or NaN) |
| rhea_id | str | Single Rhea ID (after explosion, one per row) |

**Note**: After explosion on ec and rhea_id, one unique kegg_id may appear multiple times (one row per EC per Rhea ID combination).

### Output: Compounds DataFrame
| Column | Type | Description |
|--------|------|-------------|
| kegg_id | str | KEGG compound ID (C00001) |
| name | list | List of alternative names |
| formula | str | Molecular formula (from KEGG or calculated from MOL file) |
| smiles | str | SMILES structure (from KEGG MOL or PubChem fallback) or NaN |
| dblinks | dict | Cross-references to other databases (PubChem, PubMed, ChEBI, etc.) or None |

---

## Glycan ID Replacement Details

**Why glycans matter**: Glycan IDs (G*) represent polysaccharide structures that KEGG doesn't provide structures for. Each glycan maps to a compound ID (C*) representing the core chemical entity.

**Replacement process**:
1. Extract glycan IDs from initial reactions: `pd.concat([substrates, products]).explode().str[-1]` filters for starting with "G"
2. Query KEGG glycan database: look for "REMARK Same as: C*" line
3. Build mapping dict: {G00001: C00002, ...}
4. Update reaction JSONs in place: loop through KEGG_REACTION_PATH/*.json, replace last element of substrate/product tuples
5. Reload reactions from disk after modification

**Filtering**: Only glycans with compound_id mapping are retained. Glycans without mapping are left in situ (logged warning).

---

## Error Handling

1. **Missing SMILES**: Compound attempts PubChem lookup via dblinks["PubChem"]. Falls through if not available. `smiles` remains NaN.
2. **Missing PubChem CID**: Logged warning. Compound stays with NaN smiles.
3. **Invalid MOL → SMILES**: RDKit conversion fails silently. Falls back to checking KEGG SMILES field or skipping.
4. **PubChem API failures**: Logged but no retry. Affected compounds remain NaN.
5. **Glycan ID not in mapping**: Warning logged. Glycan ID remains unchanged in reaction.
6. **Reaction SMILES generation fails**: If any substrate/product has NaN SMILES, entire rxn_smiles is NaN.
7. **Equation generation fails**: Any exception logged. equation column set to None.

---

## API Integration & Data Format

### KEGG REST API
- **Base URL**: `https://rest.kegg.jp/`
- **List endpoint**: `/rest/list/reaction` returns tab-separated "ID\tname" on each line (split by "\n")
- **Get endpoint**: `/rest/get/{id1}+{id2}+...` (batch up to 10) returns text blocks separated by "///\n"
- **Response format**: Each entry is 4-5 newline-separated fields: ENTRY, NAME, DEFINITION, EQUATION, ENZYME, DBLINKS, REFERENCE, etc.
- **Batch limits**: Code batches by 10 entries per request, uses ThreadPool(2) for parallel downloading

### Text Entry Format (Example)
```
ENTRY R00001
NAME alcohol dehydrogenase
DEFINITION  C00001 + NAD+ <=> C00002 + NADH + H+
EQUATION     C00001 + NAD+ <=> C00002 + NADH + H+
ENZYME       1.1.1.1
DBLINKS      RHEA:10000 RHEA:10001 ...
REFERENCE
>>> [PMID:12345678]
>>> DOI: 10.1234/...     
```

### PubChem API
- **Base URL**: Used via `get_pubchem_compound()` chem_utils function
- **Usage**: Pass list of CIDs, returns dict with {cid: {"smiles": ..., "formula": ...}}

---

## Key Implementation Details

1. **Substrates/Products Structure**: Tuples of [coefficient, ID] grouped in lists. Example: `[[[\"2\", \"C00001\"], [\"1\", \"C00002\"]]]` = 2 molecules C00001 + 1 molecule C00002

2. **Glycan Replacement is In-Place**: Modifies JSON files on disk, then reloads DataFrame. Not reversible without re-download.

3. **Equation Generation Only for Glycans**: The `generate_equation()` method filters substrates/products for IDs starting with "G". This creates glycan-only equations. Non-glycan reactions may have NaN equation.

4. **Stoichiometry in SMILES**: Coefficient multiplies SMILES representation. Example: 2 C00001 (SMILES: "C") → "C.C"

5. **RDKit MOL File Creation**: 
   - KEGG MOL files downloaded via `get_mol_kegg()`
   - Converted to SMILES via `MolFromMolFile()` → `MolToSmiles()`
   - For PubChem fallback: SMILES converted back to MOL via `MolFromSmiles()` → `MolToMolFile()`

6. **Column Order After Processing**: Final reactions_df columns are NOT in original order due to adding rxn_smiles/equation, dropping substrates/products/dois, renaming pmids.

7. **Two Independent Output Files**: reactions.parquet and compounds.parquet are separate. No attempt to join them; compounds are reference data.

8. **Logging Statistics**:
   - Number of metabolites total
   - Number without SMILES (before PubChem fill)
   - Number of unique reactions
   - Number of reactions without rxn_smiles
   - After explosion: total number of reaction rows (may be >>unique kegg_ids due to multiple EC/Rhea per reaction)

9. **Compound Name is LIST**: Extracted via multi-split pattern (`;` or `\n+` prefixes). Result stored as list, not single string.

10. **DBLINKS Dictionary**: Parsed from KEGG compound, keys = database names (PubChem, ChEBI, PubMed, etc.), values = IDs or comma-separated ID lists.
