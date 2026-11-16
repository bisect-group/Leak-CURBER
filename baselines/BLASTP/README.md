# BLASTP EC-classification baseline

This sequence-only baseline follows the BLAST baseline of the CARE paper. It
searches test proteins against the **training partition** of each split with
DIAMOND blastp and transfers the ECs of the hits. Test sequences and labels
never enter the reference database. The run is deterministic. It has no seeds,
learned parameters, structures, or GPU use.

## Run

From the repository root:

```bash
conda run --no-capture-output -n leakcurber python baselines/BLASTP/emulator_bench/run_blastp_ec.py \
  --threads-per-run 8 --total-threads 8
```

The runner uses the pinned DIAMOND binary `external/diamond-2.1.9/diamond`
(install it with `bash scripts/install_external_tools.sh`). Use `--diamond`,
`--source-dataset`, `--baseline-dir`, `--tmpdir`, and `--metrics-module` to
change paths.

## Method

- Splits: `random_splits_grouped_sequence`, `uniprot_time_splits`,
  `enzyme_sequence_splits/threshold_*`, `enzyme_structure_splits/threshold_*`.
  Other folders (for example `backup_*`) are ignored.
- Queries: one per `(Entry, Sequence)`, built as the CLEAN baseline builds
  them. Preliminary ECs (for example `3.4.23.B9`) stay. Partial ECs (with
  `-`) are dropped, and proteins left without a label are not queries.
- Search: `diamond blastp -k 50` with the default sensitivity and default
  E-value (0.001), as in CARE. No query-coverage filter.
- Ranking: each EC takes the best bitscore of any hit that carries it. ECs
  sort by bitscore, then by train frequency (high first), then by EC string.
  The first 20 are saved as rank columns `0`-`19`. No hit gives an empty list
  and counts as a miss.
- Metric: the CARE task-1 metric (`compute_care_metrics` from
  `baselines/GraphEC/emulator_bench/results.py`, loaded by path). Levels 4-1,
  k=1 and k=20, times 100. CLEAN's copy also cuts the top-k list to the number
  of true ECs, so its k=20 values can differ. Validation is not searched.

## Outputs

```text
<baseline-dir>/
  dataset/ and dataset_materialization.complete.json
  enzyme_classification/
    <split_group>/<threshold_x|all>/   # "all" for splits without thresholds
      train|test.fasta, db/train.dmnd, logs/{makedb,search_test}.log
      test_hits.tsv, test_predictions.parquet, test_metrics.json, complete.json
    aggregate_metrics.json
```

`test_metrics.json` holds `care_task1` (`k=1`, `k=20`), query count, and hit
coverage. Completion markers store the DIAMOND version and input fingerprint.
Reruns reuse completed splits. `--force` rebuilds them.
