# Foldseek EC-classification baseline

This structure baseline follows the Foldseek baseline of the CARE paper. It
searches test structures against the **training structures** of each split with
Foldseek and transfers the ECs of the hits. Test structures and labels never
enter the reference database. The run is deterministic. It has no seeds,
learned parameters, or GPU use.

## Run

From the repository root:

```bash
conda run --no-capture-output -n leakcurber python baselines/Foldseek/emulator_bench/run_foldseek_ec.py \
  --threads 8
```

The runner uses the pinned binary `external/foldseek/bin/foldseek` (install it
with `bash scripts/install_external_tools.sh`). Use `--foldseek`,
`--source-dataset`, `--baseline-dir`, `--intermediate-dir` (holds `mmcif`,
`alphafold`, `esmfold2_pdb`, `esm`), `--tmpdir`, and `--metrics-module` to
change paths. `--only <text>` runs the splits whose path contains the text. The
runner writes its progress messages to `<baseline-dir>/logs/run_<time>.log`.

## Method

- Splits: `random_splits_grouped_sequence`, `uniprot_time_splits`,
  `enzyme_sequence_splits/threshold_*`, `enzyme_structure_splits/threshold_*`.
  Other folders (for example `backup_*` and `ec_hierarchy_splits`) are ignored.
- Queries: the same set as the BLASTP and CLEAN baselines. One query per
  `(Entry, Sequence)`. Preliminary ECs (for example `3.4.23.B9`) stay. Partial
  ECs (with `-`) are dropped, and proteins left without a label are not
  queries. The code comes from `baselines/BLASTP/emulator_bench/run_blastp_ec.py`.
- Structures: one per protein, chosen as the lDDT structure split chooses them.
  `locate_structures` in `src/data/components/splitters/protein_struct_lddt.py`
  finds the files: the experimental mmCIF chain of the 3D dataset
  (`pdb_chain`), else the highest AlphaFold version, else ESMFold2, else ESM3.
  A protein with several structure files takes the best source, then the
  highest PDB coverage, then the lowest structure id. Training proteins without
  a structure are not in the database. Test proteins without a structure stay
  queries and count as misses. Predicted structures with a low pLDDT stay in the
  database.
- Database: Foldseek `createdb --chain-name-mode 1` over all selected files.
  `createsubdb` then keeps only the selected chain of each structure, once for
  each split's training proteins and once for its test queries.
- Search: `foldseek easy-search` with the default settings (including the
  default `--max-seqs`), no coverage filter. Only `--format-output
  query,target,bits` and `--threads` are set. CARE used `--max-seqs 1`. Here the
  top 50 hits per query, by bitscore then target name, enter the ranking.
- Ranking: each EC takes the best bitscore of any hit that carries it. ECs sort
  by bitscore, then by train frequency (high first), then by EC string. The
  first 20 are saved as rank columns `0`-`19`. No hit, or no structure, gives
  an empty list and counts as a miss.
- Metric: the CARE task-1 metric (`compute_care_metrics` from
  `baselines/GraphEC/emulator_bench/results.py`, loaded by path). Levels 4-1,
  k=1 and k=20, times 100. Validation is not searched.

## Outputs

```text
<baseline-dir>/
  dataset/ and dataset_materialization.complete.json
  structure_selection.parquet        # one selected structure per protein
  structures/{links,db,logs}         # files, global Foldseek DB, createdb log
  enzyme_classification/
    <split_group>/<threshold_x|all>/   # "all" for splits without thresholds
      db/{train,test}*, logs/{subdb_train,subdb_test,search_test}.log
      test_hits.tsv, test_predictions.parquet, test_metrics.json, complete.json
    aggregate_metrics.json
  logs/run_<time>.log
```

`test_metrics.json` holds `care_task1` (`k=1`, `k=20`), the query count, hit
coverage, `no_hit` (structure but no hit), `no_structure` (test queries with no
structure), and structure-source counts. Completion markers store the Foldseek
version, the input fingerprint, and a fingerprint of the structure selection.
Reruns reuse completed splits. `--force` rebuilds them.
