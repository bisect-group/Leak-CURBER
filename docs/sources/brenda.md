# BRENDA Dataset Building

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the BRENDA dataset building process implemented in `src/data/components/brenda.py`.

## Overview

The `BRENDADatasetBuilder` class processes raw BRENDA kinetic data to extract and clean enzyme kinetic parameters including:

- **kcat** (Catalytic rate constant / turnover number)
- **Km** (Michaelis constant)
- **Ki** (Inhibition constant)

The pipeline reads the BRENDA JSON release, extracts kinetic laws with organism and protein metadata, and gets substrate structures from the BRENDA SPARQL endpoint, and retrieves UniProt accessions for entries that have none.

---

## Data Source

**BRENDA** - A comprehensive enzyme information database with kinetic parameters from published literature.

- **Data Format**: BRENDA 2026.1 JSON release (CC BY 4.0)
- **File Source**: `brenda_json_file_path` in configuration (`data/raw/brenda/brenda_2026_1.json`)
- **Download**: Manual. Accept the licence at https://www.brenda-enzymes.org/download.php, then extract the JSON into `data/raw/brenda/`. The builder raises `FileNotFoundError` if the file is missing.

---

## Configuration

| Parameter | Description |
|-----------|-------------|
| `brenda_json_file_path` | Path to the BRENDA JSON release |
| `brenda_sparql_path` | Cache folder for SPARQL result pages (`data/raw/brenda/sparql/all`, shared with `brenda_reactions`) |
| `sparql_url`, `page_size`, `request_interval`, `request_timeout`, `max_retries` | SPARQL client settings, taken from `brenda_reactions` |
| `brenda_kinetic_params_parquet_file_path` | Output path for kinetic parameters |
| `log_dir` | Directory for log files |
| `log_file_name` | Name of the log file |

---

## Pipeline Steps

### Step 1: Load the BRENDA JSON

**Method**: `load_brenda_json()`

**Description**: Reads the JSON file and returns its `data` mapping (EC number to enzyme entry).

### Step 2: Extract Kinetic Laws

**Method**: `extract_kinetic_laws(data)`

**Description**: Extracts kcat (`turnover_number`), Km (`km_value`) and Ki (`ki_value`) entries from each enzyme.

**Process**:
1. Skip entries without a comment
2. Split each value string with `parse_value()`: the text in braces is the substrate, the rest is the number
   - A range ("a-b") gives its mean
   - An unparsable number gives -999
3. Skip entries with the substrate "more"
4. Create one row for each protein reference of the entry. Skip a protein record with more than one accession: it is an enzyme complex, and its value belongs to the whole complex. The log shows how many entries were dropped
5. Take the organism and the UniProt `accessions` (space-joined) from the protein record
6. Create DataFrame with columns: ECNumber, substrate, value, Organism, metadata, brenda_protein_id, UniprotID, value_type
7. Add the helper columns `row_pmids` (PubMed IDs of the entry's references), `protein_pmids` (PubMed IDs of the protein record's references) and `protein_comment`. Step 5 uses them. Step 6 drops them

### Step 3: Clean Kinetic Parameters

**Method**: `clean_kinetic_law_df(kinetic_law_df)`

**Description**: Cleans extracted kinetic law data by extracting metadata, annotating enzyme types, and removing duplicates. **This step occurs BEFORE substrate SMILES retrieval (Step 4).**

**Actions**:

#### 3.1 Add Unit Column
- Km and Ki values are in millimolar (mM): unit "millimolar (mM)"
- kcat values are in s^(-1): unit "s^(-1)"

#### 3.2 Extract pH Values
- Uses regex to extract pH from metadata: `pH\s*([\d.]+)`
- Convert to numeric format

#### 3.3 Extract Temperature Values
- Uses regex to extract temperature: `(\d+(?:\.\d+)?)\s*Â?°\s*C`
- Converts Celsius to Kelvin (e.g., "37°C" → 310.15)

#### 3.4 Annotate Enzyme Types and Mutations
- **Wildtype**: Default enzyme type
- **Mutant**: Identified by presence of "mutant" keyword in metadata
- **Mutations**: Extracted using regex. Regex only considers mutations encoded using HGVS notation (e.g., "A123B" or "A123B/C456D"). Mutant rows with no parsed mutation are dropped.

#### 3.5 Clean UniProt IDs
- Keep the first valid UniProt accession in the `UniprotID` text, or NaN if none

#### 3.6 Remove Duplicates and Drop Protein IDs
- Drop the `brenda_protein_id` column (no longer needed). `metadata` and the helper columns stay for Step 5
- Drop exact duplicate rows across all remaining columns. Step 6 drops duplicates again after it removes the helper columns
- Substrate column is preserved for later SMILES mapping

### Step 4: Get Substrate Structures and Generate SMILES

**Method**: `add_substrate_structures(kinetic_law_df)`

**Description**: Matches each substrate name to a compound of the BRENDA SPARQL endpoint (`https://sparql.dsmz.de/api/brenda`) and converts its InChI to SMILES. **Occurs after cleaning (Step 3) but before UniProt enrichment (Step 5).**

**Process**:
1. Download three paged queries with the client in `brenda_sparql.py`: `labels` (compound id, rdfs:label), `synonyms` (compound id, d3o:hasSynonym), and `compounds` (compound id, InChI, InChIKey). Pages are cached in `brenda_sparql_path/<query>/page_NNNN.json`. The `compounds` cache is shared with `brenda_reactions`.
2. For each substrate name, the candidates are the compounds with a structure whose label or synonym equals the name exactly (case-sensitive). `pick_structure()` chooses one:
   - If any candidate has the name as its label, only those candidates stay.
   - Candidates with the same InChIKey are not ambiguous. The lowest compound id is used.
   - If the InChIKeys differ but share the first block (same skeleton), the candidate with the best stereo score wins. The score is 1 if the InChI has a `/t` or `/b` layer, minus the number of `?` in those layers. A tied best score gives no structure.
   - If the skeletons differ, the name has no structure.
3. Names with no candidate have no structure. There is no name lookup fallback.
4. `substrate_id` is `"BRENDA:" + <SPARQL compound id>`. These ids are not the ligand ids of the BRENDA website or SOAP service.
5. `smiles` is the canonical SMILES of the InChI (via `inchi_to_smiles()`), or NaN if RDKit cannot read it.
6. The log shows how many names have a structure, match no compound, and are ambiguous (resolved by label, resolved by stereo score, or unresolved).

### Step 5: Fetch Missing UniProt Accession IDs

**Method**: `update_kinetic_laws_with_active_acc_id(kinetic_law_df)`

**Description**: Retrieves UniProt accession IDs for EC-Organism pairs missing protein identifiers.

**Process**:
1. Identify rows with missing UniProt IDs
2. Extract unique EC-Organism pairs
3. Query UniProt for active accession IDs for those pairs (in parallel)
4. Keep the reviewed (SwissProt) hits of each pair, or its unreviewed (TrEMBL) hits if it has no reviewed hit
5. Pick one hit for each row with the rules below, and write the rule name to `uniprot_match`

**Pick rules** (each rule narrows the candidates; the first rule that leaves one candidate sets `uniprot_match`):
1. `unique`: the pair has one candidate
2. `same_sequence`: all candidates have the same sequence. The best ranked one is used
3. `pubmed`: keep the candidates whose UniProt PubMed IDs include a PubMed ID of the row's references. If none match, use the protein record's references
4. `name`: keep the candidates with a gene name (primary, synonym, ORF or ordered-locus) that is a whole token, ignoring case, in the row comment. If none match, use the protein comment
5. `ranked`: use the best candidate only if every other candidate has at least 90% sequence identity to it. Rank by protein existence level (1 is best), then annotation score (higher is better), then the lowest accession. Identity is the matches of a global BLOSUM62 alignment (gap open -10, extend -0.5) divided by the longer length

A row that no rule resolves gets no UniProt ID. A row with a BRENDA accession gets `uniprot_match` `brenda`. The log shows the rows per `uniprot_match` value and the rows left without an ID.

**Output**: Kinetic parameters DataFrame with UniProt IDs for all entries

### Step 6: Filter and Save Results

**Process**:
1. Remove any rows still missing UniProt IDs: `.dropna(subset=["UniprotID"])`
   - Only rows with active Uniprot accession IDs are kept
2. Save to parquet file: `brenda_kinetic_params.parquet`
3. Compression: zstd level 3

---

## Output Files

| File | Format | Description |
|------|--------|-------------|
| `brenda_kinetic_params.parquet` | Parquet | Cleaned kinetic parameters with SMILES and enzyme metadata from BRENDA |

---

## Data Schema (Output)

| Column | Type | Description |
|--------|------|-------------|
| `ECNumber` | string | Enzyme Commission number (e.g., "1.1.1.1") |
| `substrate` | string | Substrate name as given in BRENDA |
| `substrate_id` | string | BRENDA SPARQL compound ID with prefix (e.g., "BRENDA:5"), or NaN |
| `smiles` | string | Canonical SMILES generated from the BRENDA InChI, or NaN if the substrate has no structure |
| `value` | float | Kinetic parameter value (mM for Km and Ki, s⁻¹ for kcat) |
| `Organism` | string | Source organism for the measurement |
| `pH` | float | pH of assay (if available) |
| `Temperature` | float | Temperature in Kelvin (if available) |
| `unit` | string | "millimolar (mM)" for Km and Ki, "s^(-1)" for kcat |
| `UniprotID` | string | UniProt accession ID of the enzyme |
| `enzymeType` | string | "wildtype" or "mutant" |
| `mutations` | tuple | Amino acid mutations (for mutants only, e.g., ("A123B", "C456D")) |
| `value_type` | string | Type of kinetic parameter: "kcat", "km", or "ki" |
| `uniprot_match` | string | How `UniprotID` was set: "brenda", "unique", "same_sequence", "pubmed", "name" or "ranked" |

---

## Notes

1. **Protein Rows**: Each protein reference of an entry gives one row. The organism and accessions come from the protein record in the JSON.

2. **Accessions**: UniProt accessions come from the JSON protein `accessions`. Step 5 fills the missing ones. Proteins with more than one accession (enzyme complexes) give no row, so Step 5 cannot fill them. A value measured on a complex does not belong to one chain. One row per chain would count the measurement several times and could leak it across splits.

3. **Metadata Extraction Timing**: Temperature, pH, enzyme type, and mutations are extracted from metadata strings during Step 3 (Clean) using regex patterns:
   - pH: `pH\s*([\d.]+)`
   - Temperature: `(\d+(?:\.\d+)?)\s*Â?°\s*C` (requires the degree sign)
   - Mutations: `([A-Z]\d+[A-Z](?:/[A-Z]\d+[A-Z])*)`  (HGVS notation, handles multiple mutations)
   - The metadata column is kept for the UniProt pick (Step 5) and dropped in Step 6

4. **Substrate SMILES**: Substrate names are matched to BRENDA SPARQL compounds by exact label or synonym (see Step 4). About 91% of names (96% of rows) get a structure. Missing SMILES remain as NaN in the output.

5. **UniProt Mapping**: For EC-Organism pairs without UniProt IDs, the pipeline queries UniProt REST API in parallel to find active accession IDs.

6. **SwissProt Priority**: When multiple UniProt accession IDs exist for the same EC-Organism pair, reviewed entries (SwissProt) are prioritized over unreviewed entries (TrEMBL). The pick rules in Step 5 then choose one accession for each row. After this enrichment, Step 6 removes any remaining rows without a UniProt ID.

7. **Kinetic Types**: The pipeline keeps three types of kinetic parameters:
   - **kcat**: Catalytic rate constant (turnover number), in s⁻¹
   - **Km**: Michaelis constant (substrate affinity)
   - **Ki**: Inhibition constant (inhibitor affinity)

8. **Units**: Km and Ki values are in millimolar (mM). kcat values are in s⁻¹.

9. **Duplicate Handling**: Exact duplicate rows (across all columns) are removed. This can occur when the same measurement appears in BRENDA multiple times.

10. **Enzyme Type Detection**: Mutant enzymes are identified during Step 3 (Clean) by checking if the word "mutant" appears (case-insensitive) in the metadata string. All other enzymes default to "wildtype".

11. **Substrate ID Format**: `substrate_id` is the string `"BRENDA:<compound id>"`, or NaN when the name has no structure.

12. **Final Filtering**: After UniProt enrichment (Step 5), rows lacking UniProt IDs are removed via `.dropna(subset=["UniprotID"])`. This ensures only enzyme records with identifiers are in the final output, filtering out entries where UniProt mapping failed.
