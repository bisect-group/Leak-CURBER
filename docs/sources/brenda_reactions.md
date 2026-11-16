# BRENDA Reactions

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

This document describes the BRENDA reaction extraction in `src/data/components/brenda_reactions.py`.

## Overview

The `BRENDAReactionsBuilder` class reads reactions with structures from the BRENDA SPARQL endpoint (`https://sparql.dsmz.de/api/brenda`). It writes `brenda_reactions.parquet` next to the other processed BRENDA files. A later step can merge this file into the unified reactions dataset.

The endpoint is a prototype with no documented rate limit. The builder sends one request at a time.

---

## Processing Pipeline

### Step 1: Download Paged Query Results
**Method**: `download_pages(name)` -> `fetch_pages(name, query)` -> `request_rows(query)`. The last two, `parse_tsv`, and `inchi_to_smiles` are shared with `brenda.py` and live in `src/data/components/brenda_sparql.py`.

Five small queries run in sequence. Each one uses `ORDER BY` and `LIMIT`/`OFFSET` paging with `page_size` rows per page.

| Query | Rows | Content |
|-------|------|---------|
| `reactions` | about 167,000 | Reaction, EC number, IUBMB flag |
| `participants` | about 581,000 | Reaction, role (Substrate or Product), ligand id, label |
| `compounds` | about 175,000 | Ligand id, InChI, InChIKey |
| `enzymes` | about 155,000 | Reaction, EC number, UniProt accession, organism, taxonomy id |
| `pubmed` | about 344,000 | Reaction, PubMed id |

A full extraction needs about 75 requests.

- Each page is saved to `brenda_sparql_path/<ec or all>/<query>/page_NNNN.json` through a temporary file and `os.replace`.
- Cached pages are skipped, so an interrupted run resumes. The endpoint has no release marker. Delete a query folder to refresh it.
- A request waits `request_interval` seconds first. It retries with exponential backoff on HTTP 429, 5xx, and network errors. It raises an error after `max_retries` attempts.
- Replies use the TSV format. The JSON reply of the endpoint is malformed on paged queries.
- The `enzymes` query keeps only enzymes that have a UniProt accession and whose EC number equals the reaction EC number. Enzyme nodes without UniProt are merged across ECs and organisms, so they are not trusted.

### Step 2: Convert Structures
**Function**: `inchi_to_smiles(inchi)`

1. Read the InChI with RDKit `MolFromInchi`
2. Write SMILES and standardize it with `canonicalize_smiles`, as the other sources do
3. A compound with no InChI or no readable InChI has no structure

Generic compounds such as "an alcohol" have no structure. RDKit can return a tautomer from an InChI, for example an amide as `N=C(O)`. The `molecule_key` function in `chem_utils` removes this difference.

### Step 3: Build Reactions
**Function**: `build_reactions(reactions, participants, smiles_by_cid, enzymes, pubmed, logger)`

1. Sort the participants of each reaction by ligand id, so the output is stable
2. Skip reactions without a substrate or without a product
3. Build `definition` as `A + B = C + D` from the compound labels
4. Build `rxn_smiles` as `substrates>>products` only if every participant has a structure and no participant is the placeholder `?` (an unknown compound). Otherwise `rxn_smiles` is None
5. Split enzyme complexes and clean the UniProt ids (see below), then join the enzymes on reaction and EC number. Drop rows with no usable `rxn_smiles` (None, empty, or `>>`), then join the PubMed ids on reaction. This keeps the placeholder reaction `S/89`, with its many PubMed ids, out of the output
6. One row for each reaction, EC number, and UniProt accession. A reaction with no enzyme has one row with no `uniprot_id`
7. Sort by `ec`, `brenda_rxn_id`, and `uniprot_id`, then save to parquet

BRENDA stores a heteromeric complex as one identifier that joins accessions with `and`, for example `P50213 and O43837 and P51553`. The builder reads each such identifier as one complex of several chains. It writes one row per chain accession, because each chain takes part in the reaction. The link that the chains act together is not kept. It decodes the identifier, splits it on `and` in any letter case, and also handles separators that a line wrap damaged. An isoform suffix such as `-2` is removed. Identifiers that are not UniProt accessions, for example GenBank or RefSeq ids, are dropped. The log shows the number of complex rows split, the chain rows made, and the rows dropped.

Reactions with `I/N` ids are IUBMB reactions. Reactions with `S/N` ids are substrate and product pairs from the literature. Both are kept. The reactions have no stoichiometry, so the builder does not check balance. The PubMed ids belong to the reaction and are not specific to one enzyme.

---

## Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `brenda_reactions.sparql_url` | str | SPARQL endpoint |
| `brenda_reactions.page_size` | int | Rows per page |
| `brenda_reactions.request_interval` | float | Seconds to wait before each request |
| `brenda_reactions.request_timeout` | int | Seconds before a request times out |
| `brenda_reactions.max_retries` | int | Attempts per request |
| `brenda_reactions.ec` | str or null | Restrict to one EC number; null extracts all |
| `brenda_reactions.brenda_sparql_path` | str | Directory for cached pages |
| `brenda_reactions.brenda_reactions_parquet_file_path` | str | Output parquet |
| `brenda_reactions.log_dir` | str | Directory for log files |
| `brenda_reactions.log_file_name` | str | Log file name |

Run it with `python src/data/components/brenda_reactions.py`.

---

## Data Schema

| Column | Type | Description |
|--------|------|-------------|
| brenda_rxn_id | str | BRENDA reaction id, for example `I/29164` or `S/139551` |
| ec | str | EC number |
| is_iubmb | bool | True for an IUBMB reaction |
| definition | str | Names, `A + B = C + D` |
| rxn_smiles | str | Canonical reaction SMILES |
| uniprot_id | str | UniProt accession of one chain, or None. A complex has one row for each chain |
| organism | str | Organism of the enzyme, or None |
| taxonomy_id | Int64 | NCBI taxonomy id, or null |
| pubmed_id | tuple | Sorted PubMed ids, or None |
