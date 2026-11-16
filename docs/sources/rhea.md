# Rhea Reaction Database Integration

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the Rhea database parsing and integration pipeline implemented in `src/data/components/rhea.py`.

## Overview

The `RheaDatasetBuilder` class downloads and parses biochemical reactions from the Rhea database. Key workflow:

1. **Downloads**: Rhea reactions.txt file (///-delimited entries) and Rhea-SwissProt mapping file (HTTP, not FTP)
2. **ChEBI extraction**: Regex-based extraction of CHEBI:xxxxx IDs from reaction entries
3. **ChEBI enrichment**: Download ChEBI JSON data for all metabolites in reactions
4. **Entry parsing**: Parse each 4-line Rhea entry into structured reaction dict with SMILES, direction, EC numbers
5. **Xref download**: Query Rhea API (returns 4 rows per unique rhea_id; each row is a distinct variant with different xref data)
6. **Value propagation & selective nulling**: groupby(index//4) propagates columns (ec, go_id, pubmed_id, xrefs) across all 4 rows; then selectively null specific xrefs and override direction for row variants
7. **UniProt mapping**: Read and merge Rhea-UniProt file (separate from entries) to add enzyme accessions
8. **Bidirectional EC enrichment**: Map EC→UniProt and UniProt→EC to fill missing enzyme data, add organism metadata
9. **PubMed integration**: Download abstracts and parse reference XMLs into structured data
10. **Output**: Save reactions and references to parquet files

---

## Processing Pipeline

### Step 1: Download Rhea Data
HTTP downloads (not FTP): reactions.txt (///-delimited 4-line entries) and Rhea-UniProt TSV

### Step 2: Extract and Download ChEBI
Use regex `CHEBI:\d+` to extract all ChEBI IDs from entry text. Download ChEBI JSON data.

### Step 3: Parse Entries
For each entry: extract rhea_id, definition, chebi_ids, EC. Parse metabolites with `split_coefficients()`. Get SMILES/formulae from ChEBI JSONs.

### Step 4: Download Cross-References  
Query `download_rhea_xref("")` with empty string, forcing it to return xrefs for all Rhea IDs. Returns 4 rows per unique rhea_id (each row is a directional reaction variant).

### Step 5: Propagate Values & Create Variants
For each group of 4 rows (groupby(index//4)): 
- Propagate ec, go_id, pubmed_id, and xref columns from first (index 0) row (directionless, master reaction) to all 4
- **KEGG**: Keep only in row 3 (index 3 - bidirectional reaction). Null in rows 0,1,2 (index % 4 != 3)
- **Reactome**: Keep in rows 0,1,2. Null only in row 3 (index % 4 == 3)
- **M-CSA**: Keep only in rows 1,2 (Forward and backward direction). Null in rows 0,3 (index % 4 == 0 or 3)
- **Direction**: Override row 2 (index % 4 == 2) to "RIGHT-TO-LEFT"

### Step 6: Merge Rhea-UniProt File
Read separate Rhea-UniProt TSV. Prefix rhea_ids with "RHEA:", aggregate uniprot_ids into lists, merge onto reactions.

### Step 7: Bidirectional EC-UniProt Enrichment
**Phase 1 - EC -> UniProt**: Explode(ec). Find rows with EC but no UniProt. Query UniProt API for each EC. Map back to df.
**Phase 2 - UniProt→EC**: Explode(uniprot_id). Find rows with UniProt but no EC. Query UniProt JSON for each protein. Extract EC numbers. Map back & explode(ec) again.
**Phase 3 - Metadata**: Query full UniProt JSON for all uniprot_ids. Merge additional columns (organism, ...).

### Step 8: PubMed Integration
Extract all unique PubMed IDs from the dataframe. Download abstracts in parallel. Parse all reference XMLs into structured DataFrame (separate from reactions).

### Step 9: Save to Parquet
Save reactions DataFrame and references DataFrame as two independent parquet files.

---

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `rhea.rhea_url` | str | Rhea API URL for querying cross-references |
| `rhea.rhea_reactions_url` | str | URL to download reactions.txt file |
| `rhea.rhea_uniprot_url` | str | URL to download Rhea-SwissProt mapping file |
| `rhea.rhea_reactions_file_path` | str | Local path to cache reactions.txt |
| `rhea.rhea_uniprot_file_path` | str | Local path to cache Rhea-UniProt mapping |
| `rhea.rhea_reactions_parquet_file_path` | str | Output path for reactions parquet |
| `rhea.rhea_references_parquet_file_path` | str | Output path for references parquet |
| `rhea.log_dir` | str | Directory for log files |
| `rhea.log_file_name` | str | Log file name |

---

## Methods

### `RheaDatasetBuilder`

#### `__init__(cfg: DictConfig)`
Store URLs and paths from config. Create directories. Initialize logger.

#### `setup() -> Tuple[pd.DataFrame, pd.DataFrame]`
Main entry point: download files, extract ChEBI, parse entries, merge xrefs, deduplicate, merge UniProt, enrich EC bidirectionally, download PubMed, save to parquet.

Returns: (reactions_df, references_df)

#### `extract_chebi_ids_from_rhea(rhea_entries: list, verbose: bool = False) -> set`
Use regex CHEBI:\d+ to extract unique ChEBI IDs from entry text.

Returns: Set of CHEBI ID strings

#### `parse_rhea_entry(entry: str, keep_chebi_ids: bool = False) -> dict or None`
Parse 4-line entry (ID/DE/CC/EC lines split by regex): extract rhea_id, direction (=> or <=>), definition, chebi_ids. Parse metabolites with `split_coefficients()` regex. Download & lookup ChEBI JSONs for SMILES/formulae. Build rxn_smiles and equation. Returns None if: (1) chebi_ids contains commas (macromolecules), or (2) missing ChEBI JSON data.

Returns: Dict with keys [rhea_id, direction, definition, chebi_ids, equation, rxn_smiles, ec], or None if filtered

#### `download_rhea_xref(id: str) -> pd.DataFrame`
Query Rhea API with empty string to get all reactions. Returns multiple rows per rhea_id with different xref combinations.

---

## Data Schema

### Output: Reactions DataFrame
| Column | Type | Description |
|--------|------|-------------|
| rhea_id | str | Rhea ID with RHEA: prefix |
| direction | str | LEFT-TO-RIGHT, REVERSIBLE, or RIGHT-TO-LEFT (varies per row) |
| definition | str | Reaction equation with metabolite names |
| chebi_ids | str | ChEBI mapped equation |
| equation | str | Equation with molecular formulae (or NaN) |
| rxn_smiles | str | Reaction SMILES with . and >> (or NaN) |
| ec | str | EC number (one per row after explosion) |
| go_id | str | Gene Ontology ID |
| pubmed_id | tuple or NaN | PubMed IDs as tuple (propagated across 4-row group, separated file for abstracts) |
| ecocyc_id, metacyc_id, kegg_id, reactome_id, m_csa_id | str/tuple or None | Cross-references (variant-specific: KEGG→row 3 only, Reactome→rows 0-2, M-CSA→rows 1-2) |
| uniprot_id | str | UniProt accession, exploded to one per row |
| organism | str | Organism name from UniProt |
| Other fields | Various | UniProt metadata columns |

**Key directional note**: 4 rows per unique rhea_id. Each row represents a reaction variant from Rhea API with different direction/cross-reference assignments. This is NOT duplicate data - these are distinct variants returned by the Rhea database.

### Output: References DataFrame
| Column | Type | Description |
|--------|------|-------------|
| pubmed_id | str | PubMed ID |
| title | str | Publication title |
| authors | str | Author list (comma-separated) |
| journal | str | Journal name |
| year | int | Publication year |

**Important**: References DataFrame is INDEPENDENT from reactions DataFrame. PubMed metadata is stored separately. To link back to reactions, join on pubmed_id column.

---

## ChEBI Integration

ChEBI (Chemical Entities of Biological Interest) IDs uniquely identify metabolites. Rhea uses ChEBI extensively:

### ChEBI Parsing
- Format: `CHEBI:12345` or embedded in SMILES
- Example reaction: `CHEBI:15377 + CHEBI:58349 -> CHEBI:13389`

### Benefits
1. **Standardization**: One ID per unique chemical entity
2. **Cross-Linking**: ChEBI links to PubChem, DrugBank, etc.
3. **Structure Data**: Can retrieve SMILES/InChI from ChEBI

---

## EC to UniProt Mapping

### Process
1. Query UniProt for EC number
2. Retrieve all proteins with that function
3. Filter by organism if needed
4. Store accession IDs

### Reverse Mapping
- Also build reverse: UniProt ID → EC numbers
- Useful for enzyme annotation downstream

---

## Cross-Reference Assignment (4-Row Variants)

Rhea API returns 4 rows per reaction. These represent different ways of representing the same reaction with access to different databases:

| Database | Rows Included | Rows Nulled |
|----------|---------------|-------------|
| KEGG | Row 3 only | Rows 0,1,2 |
| MetaCyc | Rows 0,1,2 | Row 3 |
| Reactome | Rows 0,1,2 | Row 3 |
| M-CSA | Rows 1,2 | Rows 0,3 |
| EcoCyc | All 4 rows | None |

---

## Filtering & Error Handling

### Entry-Level Filtering
1. **Macromolecules**: Skip entries where chebi_ids contains commas (indicator of protein/RNA in reaction)
2. **Missing ChEBI data**: Skip if ChEBI JSON cannot be found for any metabolite
3. **Direction mismatch**: Entries missing => or <=> arrows → direction set to None

### Downstream Handling
4. **Missing SMILES**: rxn_smiles is NaN if any metabolite missing SMILES
5. **Missing formulae**: equation is NaN if any metabolite missing formula
6. **Missing enzymes**: ec column is NaN/None if reaction has no EC numbers
7. **EC -> UniProt mapping**: If EC not found in UniProt, uniprot_id remains NaN
8. **UniProt -> EC mapping**: If UniProt has no EC, ec remains NaN

---
