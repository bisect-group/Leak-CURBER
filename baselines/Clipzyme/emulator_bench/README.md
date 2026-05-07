# Clipzyme Leak-CURBER Retrieval Wrapper

This wrapper adapts `baselines/Clipzyme` to
`data/processed/datasets/enzyme_retrieval_dataset` without replacing Clipzyme's
native Lightning trainer.

## Data Mapping

- Split discovery recursively finds folders containing `train.parquet`,
  `val.parquet`, and `test.parquet`.
- Required parquet columns are `rxn_smiles`, `sequence`, and `ec_number`.
- `val` is written as Clipzyme `dev`.
- Rows with missing or partial EC labels are removed by default.
- Protein sequences are truncated to Clipzyme's 650-aa limit before IDs and
  manifests are generated.
- Each split file is deduplicated by normalized `rxn_smiles`, truncated
  `sequence`, and normalized `ec_number`.
- The dataset has no stable common IDs, so pseudo IDs are SHA-256 content hashes.

## Cache And Generated Files

`cache_features.py` writes:

- shared RXNMapper atom-map cache under `emulator_bench/cache/atom_maps/`;
- split manifests under `emulator_bench/runs/<split_slug>/manifests/`;
- Clipzyme JSON and pickle inputs under
  `emulator_bench/runs/<split_slug>/clipzyme_inputs/`;
- split metadata under `emulator_bench/runs/<split_slug>/metadata.json`.

RXNMapper failures, incomplete atom maps, non-computable bond changes, malformed
reactions, mapped reactions that are incompatible with Clipzyme's native
reactant/product graph tensor shapes, missing sequences, and incomplete EC labels
are logged in metadata and removed.

## Commands

Use the provided environment:

```bash
conda run -n clipzyme python -m emulator_bench.cache_features \
  --dataset-root ../../data/processed/datasets/enzyme_retrieval_dataset \
  --split-group random_splits
```

Train through Clipzyme's native dispatcher:

```bash
conda run -n clipzyme python -m emulator_bench.train \
  --split-group random_splits \
  --seed 42 \
  --epochs 20 \
  --precision bf16
```

Evaluate the checkpoint produced by training:

```bash
conda run -n clipzyme python -m emulator_bench.evaluate \
  --split-group random_splits \
  --seed 42 \
  --eval-split test
```

Queue all discovered split groups and default seeds `42,43,44` through
task-spooler:

```bash
conda run -n clipzyme python -m emulator_bench.queue_pipeline \
  --dataset-root ../../data/processed/datasets/enzyme_retrieval_dataset \
  --epochs 20 \
  --precision bf16
```

The queue wrapper defaults `--available-gpus` to `${CUDA_VISIBLE_DEVICES:-0}` so
task-spooler GPU assignment is preserved inside Clipzyme's native dispatcher.
Pass a concrete GPU only when you intentionally want to bypass task-spooler GPU
selection.

Reruns are incremental. The cache stage is still submitted so missing atom-map
entries are generated, but existing per-reaction JSON cache entries are
validated and reused. Queue submission skips a seed's train job when its
`train_seed.json` points to an existing checkpoint; evaluation then depends on
the refreshed cache job instead of a retrain job.

Required smoke-test shape:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n clipzyme python -m emulator_bench.queue_pipeline \
  --split-group random_splits \
  --seed 42 \
  --epochs 1 \
  --limit-per-split 64 \
  --batch-size 1 \
  --accumulate-grad-batches 16 \
  --clip-freeze-esm \
  --eval-split test \
  --gpus-per-job 0
```

For full benchmark runs, omit `--limit-per-split`, `--batch-size`,
`--accumulate-grad-batches`, and `--clip-freeze-esm` unless the target GPU memory
requires those overrides.
The repository-level `queue_multiple_seeds.sh` targets the local 22 GB A10
task-spooler setup and therefore passes
`--batch-size 1 --accumulate-grad-batches 16`. This keeps the previous
single-GPU optimizer update size from the default `batch_size=8` and
`accumulate_grad_batches=2`, but the smaller microbatch still changes CLIP's
in-batch negative set. If trainable ESM still exceeds GPU memory, add
`--clip-freeze-esm` and report that run separately.

## Metrics

Native Clipzyme metrics are preserved through dispatcher evaluation:
`clip_loss`, `clip_accuracy`, and `clip_quantile`.

The wrapper also emits CARE Task 2 outputs under
`emulator_bench/runs/<split_slug>/seeds/<seed>/results/<split>/`:

- `care_task2_ranked_ec.csv` with metadata columns plus rank columns
  `0,1,2,...`;
- `care_task2_metrics.json` with level-1/2/3/4 accuracy at
  `k = 1,3,5,10,20,30,40,50`;
- `supplemental_metrics.json` with exact-EC MRR, MAP, hit@k, mean rank, median
  rank, and row/rank counts.

## Environment Notes

The `clipzyme` environment must include parquet, atom-mapping, and
`pkg_resources` support:

```bash
conda run -n clipzyme python -m pip install 'setuptools<81' pyarrow rxnmapper tensorboard
```
