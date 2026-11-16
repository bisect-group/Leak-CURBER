# Enzyme retrieval and classification datasets

The public Leak-CURBER release does not include the code for this step. This document describes what it does.

Both builders read the unified reactions parquet files. Each reads only the columns it needs.

## Inputs

| Builder | Config key | File |
|---------|------------|------|
| Retrieval, `include_protein_information: false` | `unified_reactions_1d_parquet_file_path` | 1D: one row per reaction row |
| Retrieval, `include_protein_information: true` | `unified_reactions_3d_parquet_file_path` | 3D: one row per reaction row and structure |
| Classification | `unified_reactions_3d_parquet_file_path` | 3D |
| Enzyme compound | `unified_reactions_3d_parquet_file_path` | 3D |

## Outputs

- Retrieval: `rxn_key`, `rxn_smiles`, `ec_number`, `uniprot_date`. With protein information, it adds
  `sequence`, `pdbs`, `pdb_source`, `pdb_type`, `pdb_chain` and `pdb_coverage`.
- Classification: `uniprot_id`, `sequence`, `ec_number`, `uniprot_date`, `pdbs`, `pdb_source`,
  `pdb_type`, `pdb_chain` and `pdb_coverage`.

`pdbs` is one structure key per row: `<PDB>:<base accession>` for experimental structures, else the accession. Each pair of (reaction or enzyme, EC number, structure) is one row.

## Full EC numbers only

Both builders keep only full EC numbers. `_filter_partial_ec_numbers` drops every row whose `ec_number` has `-` in any of its four fields (for example `1.1.1.-`). The builders also drop rows with no EC number or with a first field outside 1 to 7.

The rule is on by default. `include_partial_ec_numbers: false` is set in `configs/data/enzyme_classification_dataset.yaml` and `configs/data/enzyme_retrieval_dataset.yaml`. Set it to `true` to keep partial ECs.
