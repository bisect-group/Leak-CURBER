# ChEBI Dataset Utilities

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the ChEBI dataset utilities implemented in `src/data/components/chebi.py`.

## Overview

The `ChEBIDatasetUtils` class provides utilities for downloading, parsing, and managing chemical structure data from ChEBI (Chemical Entities of Biological Interest).

Functions include:
- Downloading ChEBI molecular structures (2D and 3D)
- Parsing ChEBI metadata (SMILES, InChI, chemical properties)
- Generating molecular representations (SMILES, InChI, InChI Key)
- Managing cached molecular data

---

## Data Source

**ChEBI** - A freely available database of small chemical compounds.

- **API URL**: ChEBI SOAP webservice and REST API
- **Molecular Structures**: 2D (MOL) and 3D (SDF) files available

---

## Configuration

| Parameter | Description |
|-----------|-------------|
| `chebi_compounds_api_url` | URL for ChEBI compounds API |
| `chebi_mol_file_api_url` | URL for ChEBI MOL file downloads |
| `chebi_mol_path` | Dictionary with "2D" and "3D" paths for structure files |
| `chebi_raw_path` | Path for downloaded JSON files |
| `log_dir` | Directory for log files |

---

## Main Methods

### 1. Download ChEBI Molecular Structures

**Method**: `download_chebi_mol_parallel(metabolite_id_pairs, verbose=True)`

**Description**: Downloads MOL files in parallel for metabolites.

**Process**:
1. For each metabolite, check if MOL file already exists
2. Query the ChEBI REST API with the numeric ChEBI ID (`{chebi_number}` in `chebi_mol_file_api_url`)
3. Skip empty or non-MOL responses with a warning
4. Save to disk
5. If download fails, log an error; SMILES/InChI are generated later from cached data

**Parameters**:
- `metabolite_id_pairs`: List of tuples (standardized_chebi_id, primary_chebi_id)
- `verbose`: Show progress bar

**Parallelization**: Uses ThreadPool with 20 workers for concurrent downloads

### 2. Download ChEBI JSON Data

**Method**: `download_chebi_json(metabolites, verbose=True)`

**Description**: Downloads complete ChEBI entity data including structure information.

**Process**:

1. **Check for already-downloaded metabolites**:
   - Identify which metabolites have existing JSON files in `chebi_raw_path`
   - If ALL metabolites are already downloaded, skip to step 5 (MOL download + finalization), avoiding redundant API calls

2. **Batch pending metabolites**:
   - Split pending metabolites into batches of 100 IDs each minimizing API requests to ChEBI SOAP webservice

3. **Query ChEBI API for each batch**:
   - Send POST request to ChEBI compounds API with batch of IDs
   - Receive JSON response with complete entity data

4. **Extract and save entity data** (each field extraction individually wrapped in try-except):
   - **Basic info**: Name, definition
   - **Chemical structures**: InChI, SMILES (canonicalized), InChI Key
   - **Chemical properties**: Formal charge, molecular mass, monoisotopic mass
   - **Identifiers**: Secondary IDs, synonyms, IUPAC names, molecular formulae
   - **Cross-references**: Links to ChEMBL, PubChem, KEGG, etc. (from manual xrefs and citations)
   - Save JSON file for each metabolite

5. **Download MOL files**:
   - Accumulate pairs of (standardized_chebi_id, primary_chebi_id) across all batches
   - Call `download_chebi_mol_parallel()` to fetch all MOL files in parallel (ThreadPool with 20 workers)

6. **Finalize JSON files**:
   - Call `finalize_chebi_jsons()` to generate missing SMILES/InChI from MOL files using RDKit

**Output**: JSON files in `chebi_raw_path`, MOL/SDF files in `mol_path`

### 3. Finalize ChEBI JSON Files

**Method**: `finalize_chebi_jsons(metabolites, verbose=True)`

**Description**: Generates missing SMILES/InChI from MOL files using RDKit.

**Process**:
1. For each metabolite JSON without SMILES:
   - Load corresponding MOL file
   - Parse with RDKit
   - Generate SMILES using canonical representation
   - Generate InChI and InChI Key
   - Update and save JSON

**Output**: Complete JSON files with all molecular representations

### 4. Parse Saved ChEBI JSON Files

**Method**: `parse_chebi_jsons(chebi_ids, return_as_dict=True, verbose=True)`

**Description**: Loads and parses previously downloaded ChEBI JSON files.

**Process**:
1. Load JSON for each ChEBI ID
4. Return as dictionary or list based on parameter

**Parameters**:
- `chebi_ids`: Iterable of ChEBI IDs to parse
- `return_as_dict`: If True, return dict keyed by chebi_id; if False, return list
- `verbose`: Show debug logs

**Process**:
1. For each ChEBI ID:
   - Attempt to load JSON file; if missing, log warning and skip
2. Collect results into dict or list based on `return_as_dict` parameter

**Output**: 
- Dictionary or list of metabolite data (depending on `return_as_dict`)
- Restored MOL/SDF files written to appropriate directories if embedded data was present

---

## Data Schema (ChEBI JSON)

| Field | Type | Description |
|-------|------|-------------|
| `chebi_id` | string | Standardized ChEBI ID |
| `primary_chebi_id` | string | Primary ChEBI ID |
| `name` | string | Preferred ChEBI name |
| `definition` | string | Definition/description |
| `inchi` | string | Standard InChI string |
| `inchi_key` | string | InChI Key for compound identification |
| `smiles` | string | Canonical SMILES string |
| `charge` | int | Formal charge |
| `mass` | float | Molecular mass (mass number) |
| `monoisotopic_mass` | float | Monoisotopic mass |
| `secondary_ids` | list | Alternative ChEBI identifiers |
| `synonyms` | list | Common names/synonyms |
| `iupac_names` | list | IUPAC chemical names |
| `formulae` | list | Molecular formulae |
| `xrefs` | dict | Cross-references to ChEMBL, PubChem, KEGG, etc. (manual xrefs from ChEBI) |
| `citations` | dict | Citation database links (from ChEBI citations) |

---

## Notes

1. **Early Return Optimization**: If all requested metabolites are already in cache, `download_chebi_json()` skips API calls and proceeds directly to MOL file downloads and finalization. This avoids redundant ChEBI API requests.

2. **Field Extraction Robustness**: Each metadata field extraction (SMILES, InChI, charge, mass, etc.) is individually wrapped in try-except to gracefully handle missing or malformed API responses. Missing fields are logged as warnings and set to None or empty collections.

3. **Batching Strategy**: Pending metabolites are pre-split into batches of 100 IDs each (before the loop) to minimize ChEBI API requests. All MOL downloads are then accumulated across batches and downloaded together in parallel.

4. **Parallelization**: MOL file downloads use ThreadPool(20) for concurrent HTTP requests without blocking. JSON API calls are batched but sequential (one batch at a time) to avoid overwhelming the ChEBI API.

5. **Cache Strategy**: Previously downloaded ChEBI JSON data is cached in `chebi_raw_path`. Files are only re-fetched from the API if missing from cache.

6. **Structure Files Workflow**: 2D (MOL) and 3D (SDF) structure files are downloaded separately from the JSON metadata via the ChEBI MOL file API. If a MOL file cannot be downloaded, SMILES/InChI are generated from cached data using RDKit during the finalization step.

7. **Cross-references**: ChEBI JSON includes mappings to other chemical databases (PubChem, ChEMBL, KEGG, etc.) via manual xrefs and citations fields extracted from the API response.

8. **SMILES Canonicalization**: SMILES strings from ChEBI are standardized using RDKit's canonical representation to ensure consistent molecular representation.
