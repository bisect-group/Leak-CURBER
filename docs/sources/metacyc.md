# MetaCyc Database Integration

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the MetaCyc data parsing pipeline implemented in `src/data/components/metacyc.py`.

## Overview

The MetaCyc module parses local MetaCyc flat files in attribute-value (AV) format. Architecture:

1. **Input**: 7 local MetaCyc .dat files (enzymatic reactions, proteins, reactions, compounds, protein sequences, protein-seq-ID mapping, atom-mapped SMILES)
2. **Parse & Preprocess**: Convert each AV file to DataFrame, handle S-expressions, extract annotations
3. **Integration**: Link reactions → enzymatic reactions → proteins → UniProt sequences
4. **Kinetic Parameters**: Extract KCAT and KM per enzymatic reaction per substrate per protein
5. **Output**: 3 parquet files (reactions with protein info, compounds, kinetic parameters)

---

## Processing Pipeline

### Step 1: Validate Input Files
Check that all 7 required MetaCyc files exist. Log error if missing.

### Step 2: Parse Compounds
Read compounds.dat via `parse_attribute_value_file()`. Handle annotations. Clean columns (remove CREDITS, INSTANCE-NAME-TEMPLATE, SYNONYMS, COMMENT). Validate SMILES via RDKit.MolFromSmiles(). Drop rows with invalid/NaN SMILES. Generate MOL files. Return (compounds_df, compound_smiles_mapping, compound_formula_mapping).

### Step 3: Parse Atom-Mapped SMILES
Read atom-mappings-smiles.dat into DataFrame with UNIQUE-ID and ATOM-MAPPED-SMILES columns.

### Step 4: Parse Reactions
Read reactions.dat via `parse_attribute_value_file()`. Clean many columns (drop 20 columns like ATOM-MAPPINGS, CREDITS, ENZYME-NOT-USED, etc.). Strip annotations. Merge with atom-mapped SMILES. Clean EC-NUMBER (via `clean_EC_NUM()`) and DBLINKS (via `clean_DBLINKS()`). Clean LEFT/RIGHT sides (via `clean_reaction_side()`). Generate reaction SMILES (via `generate_reaction_smiles()`). Augment with DEFINITION/EQUATION and extract kegg_id/metanetx_id/rhea_id (via `augment_rxn()`). Create enz_2_rxn_map by exploding ENZYMATIC-REACTION. Return (reactions_df, enz_2_rxn_map).

### Step 5: Parse Proteins and Sequences
Read proteins.dat, protseq.fasta (merged into DataFrame), and protein-seq-ids.lisp (parsed with sexpdata). Merge proteins with protseq. Explode CATALYZES. Merge with enz_2_rxn_map. Clean DBLINKS (extract UniProt IDs only). Parse protein-seq-ids.lisp S-expressions to get alternate accessions. Fetch UniProt sequences via `get_uniprot_acc_json()` for primary IDs. For missing sequences, try alternates via `try_alternate_accs()`. Return (proteins_df, protein_seq_ids_df).

### Step 6: Parse Enzymatic Reactions
Read enzrxns.dat via `parse_attribute_value_file()`. Extract KCAT, KM, and other kinetic fields. Merge with proteins_df. Return enzrxns_df with kinetic data ready for expansion.

### Step 7: Merge Reactions with Proteins
Inner merge reactions_df with proteins_df to link reactions with enzyme info (UNIQUE-ID, SEQUENCE, ORGANISM).

### Step 8: Extract Kinetic Parameters
For each kinetic type (KCAT, KM):
  - Call `process_kinetic_df()` which:
    - Filters enzrxns_df to rows with non-null kinetic_type column
    - Expands kinetic dict (via `expand_kinetic_df()`) into value/substrate/citations
    - Merges with enz_2_rxn_map to link to reactions
    - Merges with compounds_df to get substrate SMILES
    - Merges with protein_seq_ids_df to get sequences/organism
    - Drops rows with missing values
  - Returns kinetic params DataFrame with substrate-level granularity

### Step 9: Output
Save 3 parquet files: reactions.parquet, compounds.parquet, kinetic_params.parquet

---

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `metacyc.metacyc_enzrxns_file_path` | str | Path to enzrxns.dat file |
| `metacyc.metacyc_protseq_file_path` | str | Path to protseq.fasta file |
| `metacyc.metacyc_proteins_file_path` | str | Path to proteins.dat file |
| `metacyc.metacyc_reactions_file_path` | str | Path to reactions.dat file |
| `metacyc.metacyc_compounds_file_path` | str | Path to compounds.dat file |
| `metacyc.metacyc_protein_seq_ids_file_path` | str | Path to protein-seq-ids.lisp file |
| `metacyc.metacyc_atom_mapped_smiles_file_path` | str | Path to atom-mappings-smiles.dat file |
| `metacyc.mol_path` | str | Directory to cache generated MOL files |
| `metacyc.metacyc_reactions_parquet_file_path` | str | Output path for reactions parquet |
| `metacyc.metacyc_compounds_parquet_file_path` | str | Output path for compounds parquet |
| `metacyc.metacyc_kinetic_params_parquet_file_path` | str | Output path for kinetic parameters parquet |
| `metacyc.log_dir` | str | Log directory |
| `metacyc.log_file_name` | str | Log file name |

---

## Methods

### `MetaCycDatasetBuilder`

#### `__init__(cfg: DictConfig)`
Store paths for 7 input files and 3 output files. Create output directories. Initialize logger. Validate all 7 input files exist (raise FileNotFoundError if missing).

#### `setup() -> None`
Main orchestration: parse_compounds → parse_reactions → parse_proteins → parse_enzrxns → merge reactions+proteins → extract kinetic params (KCAT, KM) → save 3 parquet files.

#### `parse_attribute_value_file(text: str) -> list[dict]`
Parse MetaCyc AV format. Handle entry delimiters (//), continuation lines (/), annotation groups (^), multi-line values. Return entries as list of dicts with attribute values as list[{"value": ..., "annotations": ...}].

#### `split_columns_by_annotations(df: pd.DataFrame) -> Tuple[list, list]`
Identify DataFrame columns with/without annotation dicts. Return (cols_with_annotations, cols_without_annotations).

#### `strip_annotation_keys(df: pd.DataFrame, cols_without_annotations: list) -> pd.DataFrame`
Simplify specified columns from {"value": x, ...} to scalars/lists.

#### `read_fasta(file_path) -> list[tuple]`
Parse FASTA file. Header format: >key|name|organism. Return list of (key, name, organism, sequence) tuples.

#### `clean_EC_NUM(ec_num: list) -> str or None`
Extract EC number from list of dicts. Log warning if multiple or invalid format.

#### `clean_DBLINKS(dblinks: list or str) -> list[tuple]`
Parse DBLINKS pattern (DB "ID") to extract list of (db, id) tuples using regex.

#### `clean_reaction_side(participants: list) -> list[tuple]`
Extract (coefficient, compound_id) tuples. Coeff defaults to 1 if missing.

#### `generate_reaction_smiles(row: pd.Series, compound_smiles_mapping: dict) -> pd.Series`
Generate atom-mapped reaction SMILES. If already present, keep it. Otherwise build from LEFT/RIGHT using SMILES lookup, replicate by coefficient, return "SMILES.SMILES>>SMILES.SMILES" or None if any missing.

#### `augment_rxn(row: pd.Series, compound_formula_mapping: dict) -> pd.Series`
Generate DEFINITION and EQUATION. Parse DBLINKS for kegg_id, metanetx_id, rhea_id lists.

#### `parse_compounds() -> Tuple[pd.DataFrame, dict, dict]`
Parse compounds.dat, validate SMILES via RDKit, drop invalid/NaN. Return (compounds_df, compound_smiles_mapping, compound_formula_mapping).

#### `parse_reactions(compound_smiles_mapping: dict, compound_formula_mapping: dict) -> Tuple[pd.DataFrame, pd.DataFrame]`
Parse reactions.dat and atom-mappings-smiles.dat. Clean, merge, generate SMILES/definitions. Extract enz_2_rxn_map. Return (reactions_df, enz_2_rxn_map).

#### `parse_proteins(enz_2_rxn_map: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`
Parse proteins.dat, protseq.fasta, protein-seq-ids.lisp. Merge, fetch UniProt sequences, try alternates. Return (proteins_df, protein_seq_ids_df).

#### `parse_enzrxns(proteins_df: pd.DataFrame) -> pd.DataFrame`
Parse enzrxns.dat. Extract single values. Merge with proteins_df. Return enzrxns_df with kinetic data.

#### `process_kinetic_df(enzrxns_df: pd.DataFrame, kinetic_type: str, enz_2_rxn_map: pd.DataFrame, compounds_df: pd.DataFrame, protein_seq_ids_df: pd.DataFrame) -> pd.DataFrame`
Extract kinetic parameters of type kinetic_type. Expand, merge with enz_2_rxn_map, compounds_df, protein_seq_ids_df. Drop missing. Return kinetic params with substrates, sequences, organisms.

#### `expand_kinetic_df(row: pd.Series, type: str) -> pd.Series`
Expand kinetic parameter dict into value, substrate, citations columns.

#### `sexp_to_str(obj) -> str or list`
Recursively convert S-expression objects to strings. Used for protein-seq-ids.lisp.

#### `try_alternate_accs(row: pd.Series, results: dict) -> pd.Series`
For missing SEQUENCE, try alternate IDs from SEQ-ID list. Return updated row with valid sequence.

---

## Data Schema

### Input: MetaCyc Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| UNIQUE-ID | str | Unique identifier |
| COMMON-NAME | str | Human-readable name |
| LEFT | list | Substrates with stoichiometry |
| RIGHT | list | Products with stoichiometry |
| KINETIC-DATA | str | S-expression with kinetic parameters |
| REACTION-DIRECTION | str | REVERSIBLE, LEFT-TO-RIGHT, or RIGHT-TO-LEFT |
| DBLINKS | list | Cross-references to other databases |

### Output: Reactions (reactions.parquet)
| Column | Type | Description |
|--------|------|-------------|
| UNIQUE-ID | str | MetaCyc reaction ID |
| EC-NUMBER | str | EC number from reaction |
| EQUATION | str | Reaction equation with compound formulas |
| DEFINITION | str | Reaction definition with compound names |
| LEFT | list[tuple] | [(coeff, compound_id), ...] substrates |
| RIGHT | list[tuple] | [(coeff, compound_id), ...] products |
| REACTION-DIRECTION | str | REVERSIBLE, LEFT-TO-RIGHT, or RIGHT-TO-LEFT |
| ATOM-MAPPED-SMILES | str | Atom-mapped reaction SMILES |
| kegg_id | list[str] | KEGG reaction cross-references |
| metanetx_id | list[str] | MetaNetX reaction cross-references |
| rhea_id | list[str] | RHEA reaction cross-references |
| UNIQUE-ID (from proteins) | str | Associated enzyme UNIQUE-ID |
| SEQUENCE | str | Protein sequence from UniProt |

### Output: Compounds (compounds.parquet)
| Column | Type | Description |
|--------|------|-------------|
| UNIQUE-ID | str | MetaCyc compound ID |
| COMMON-NAME | str | Compound name |
| SMILES | str | SMILES string (RDKit-validated) |
| CHEMICAL-FORMULA | str | Chemical formula |
| MONOISOTOPIC-MW | float | Monoisotopic molecular weight |
| INCHI | str | INCHI string |
| MOLECULAR-WEIGHT | float | Standard molecular weight |
| GIBBS-0 | float | Gibbs free energy of formation |
| kegg_id | str | KEGG compound ID |
| chebi_id | str | ChEBI compound ID |

### Output: Kinetic Parameters (kinetic_params_KCAT.parquet, kinetic_params_KM.parquet)
| Column | Type | Description |
|--------|------|-------------|
| ENZ-ID | str | Enzyme UNIQUE-ID |
| RXN-ID | str | Reaction UNIQUE-ID |
| SUBSTRATE | str | Substrate compound ID |
| KCAT or KM | float | Kinetic parameter value |
| CITATIONS | str or list | Reference citations |
| SEQUENCE | str | Protein sequence |
| ORGANISM | str | Source organism |

---

## Attribute-Value Format

### Example Structure
```
#
UNIQUE-ID - RXN-12345
COMMON-NAME - Glucose-6-phosphatase reaction
LEFT - CPD-1234@1 | CPD-5678@2
RIGHT - CPD-9999@1 | CPD-8888@1
REACTION-DIRECTION - REVERSIBLE
KINETIC-DATA - (...)
DBLINKS - (KEGG-REACTION "R00001") (BRENDA "1.1.1.1")
#
```

---

## S-Expression Parsing

MetaCyc uses S-expressions (Lisp-like syntax) for complex data:

```
(KINETIC-DATA
 (ENTRY "...")
 (ENTRY "...")
)
```

Key features:
- Parentheses denote nested structures
- Quoted strings for values
- Attributes as first element of list

---

## Error Handling

### File Not Found
All 7 input files must exist. FileNotFoundError raised in `__init__()` if any missing.

### Invalid SMILES
`parse_compounds()` validates via RDKit.MolFromSmiles(). Drops rows with invalid/NaN SMILES. Logs warning if >10% compounds have invalid SMILES.

### Malformed AV Format Entries
`parse_attribute_value_file()` handles continuation lines (/) and annotation groups (^). Skips entries with missing UNIQUE-ID. Warns if entry lacks critical attributes (LEFT, RIGHT for reactions).

### Dead UniProt Accessions
`parse_proteins()` tries primary ID first, falls back to alternates via `try_alternate_accs()`. If all fail, sequence=None (row retained). Log warning with organism name when sequence not found.

### Missing Kinetic Data
Enzrxns entries without KINETIC-DATA are skipped. `process_kinetic_df()` drops rows where value is NaN.

### S-Expression Parsing Failures
`sexp_to_str()` wrapped in try-except for malformed protein-seq-ids.lisp. Falls back to empty dict if parsing fails.

### Annotation Dict Handling
`split_columns_by_annotations()` identifies columns with {"value": ..., "annotations": ...} dicts. `strip_annotation_keys()` converts to scalars; annotations discarded.

---

## Implementation Details

### AV Format Parsing
Entry delimiters: // (double slash on separate line). Continuation: values on next line prefixed with /. Annotations: attribute values followed by ^annotationgroup with structured dict. Multi-line parsing: accumulate until next / or // delimiter.

### Stoichiometry & Coefficients
Participants encoded as COMPOUND-ID@COEFF (e.g., CPD-1234@2). Coeff defaults to 1 if omitted. LEFT/RIGHT lists delimited by |. `parse_attribute_value_file()` returns list[{"value": x, "annotations": {...}}].

### Reaction SMILES Generation
`generate_reaction_smiles()` replicates atoms by stoichiometry. Format: [C].[C].[C]>>[C].[C] (replicated atoms on each side). Returns None if any substrate/product SMILES missing.

### UniProt Sequence Strategy
Primary: protein-seq-ids.lisp → UniProt direct fetch. Fallback: `try_alternate_accs()` with SEQ-ID list. Final fallback: protseq.fasta match by organism. Result: sequence or None.

### Enzyme-Reaction Linking
enz_2_rxn_map built during `parse_reactions()` by exploding ENZYMATIC-REACTION field. Links enz_id → reactions by ENZRXN-ID. Used in `process_kinetic_df()` to join kinetic params with reactions.

### Unit Handling
Units from KINETIC-DATA kept as-is (e.g., "µM", "s-1"). No conversion to SI units. Stored in original_unit for external normalization if needed.

### Kinetic Parameters Structure
Different types (KCAT, KM, etc.) processed separately. One type per `process_kinetic_df()` call. `expand_kinetic_df()` extracts value/substrate/citations from S-expression. Substrate-level granularity: one row per substrate per enzyme per reaction.

### Enzyme-Substrate Granularity
Kinetic measurements indexed by substrate_compound_id (specific compound). One measurement row = one enzyme + reaction + substrate + param type combo. Enables multi-substrate reactions with independent kcat/km per substrate.

### SMILES Validation
RDKit.MolFromSmiles() validates all compound SMILES. Invalid/NaN dropped silently. Generates MOL files via RDKit.MolToMolFile() for visual inspection.
