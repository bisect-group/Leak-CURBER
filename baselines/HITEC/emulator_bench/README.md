# HITEC Leak-CURBER Adapter

This wrapper adapts HITEC to the Leak-CURBER enzyme classification dataset layout.
It keeps HITEC sequence-only: PDB columns are retained as manifest metadata but are
not model inputs.

## Dataset Mapping

Expected dataset root:

```bash
../../data/processed/datasets/enzyme_classification_dataset
```

Each split group must contain `train.parquet`, `val.parquet`, and `test.parquet`.
The wrapper discovers the released grouped random split directories plus
sequence, structure, time, and EC hierarchy split groups automatically.

Required columns are `uniprot_id`, `sequence`, and `ec_number`. Optional columns
such as `pdbs`, `pdb_source`, and `pdb_type` are retained after per-file
deduplication for traceability.

## Cache And Manifests

`cache_features.py` writes:

- `emulator_bench/vocab/ec_vocab.json`
- `emulator_bench/cache/tokens/<uniprot_id>.pt`
- `emulator_bench/runs/<split_group>/manifests/{train,val,test}.csv`
- `emulator_bench/runs/<split_group>/metadata.json`

Sequences are uppercased, unknown residues are mapped to `X`, BOS token `22` is
prepended, and tensors are truncated/padded to the model's supported positional
length. For HITEC's documented config this defaults to `1024`; use
`--model-dimension` to change the model limit and optional `--max-tokens` to use a
shorter token length. Partial EC labels use masked-prefix training, so `3.2.2.-`
observes levels 1-3 and masks level 4.

## Commands

Prepare one split group:

```bash
conda run -n hitec python -m emulator_bench.cache_features \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits_grouped_sequence
```

Train one seed:

```bash
conda run -n hitec python -m emulator_bench.train \
  --split-group random_splits_grouped_sequence \
  --seed 1234 \
  --epochs 80 \
  --precision auto
```

Evaluate one seed:

```bash
conda run -n hitec python -m emulator_bench.evaluate \
  --split-group random_splits_grouped_sequence \
  --seed 1234 \
  --eval-split test
```

Queue cache, train, and evaluate with `ts`:

```bash
conda run -n hitec python -m emulator_bench.queue_pipeline \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits_grouped_sequence \
  --seed 1234 \
  --epochs 80 \
  --wait
```

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n hitec python -m emulator_bench.queue_pipeline \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits_grouped_sequence \
  --epochs 1 \
  --seed 1234 \
  --limit-per-split 16 \
  --cuda-visible-devices 3 \
  --wait
```

Aggregate metrics:

```bash
conda run -n hitec python -m emulator_bench.aggregate_results
```

## Metrics

Primary HITEC-style metrics include micro/macro precision, recall, F1, exact-match
accuracy, and per-level micro/macro metrics. Validation predictions select
per-class F1-maximizing thresholds; test metrics reuse those thresholds.

The evaluator also writes CARE Task 1 ranked CSVs with metadata columns followed
by rank columns `0,1,2,...`. CARE hierarchical accuracy is reported at levels 1-4
for `k=1` and `k=20`. Supplemental ranking metrics include MRR and hit@1, hit@3,
hit@5, hit@10, and hit@20.

## Notes

- Environment name: `hitec`.
- Queue command: `ts`.
- Default seeds: `1234`, `2345`, `3456`.
- Default real-run epochs: `80`.
- Default rank output columns: top 50 full EC labels, enough for CARE `k=20`.
