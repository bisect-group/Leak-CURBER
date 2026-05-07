# GraphEC Leak-CURBER Adapter

This wrapper adapts GraphEC to `data/processed/datasets/enzyme_classification_dataset`.
GraphEC's repository exposes EC-number inference but no native training loop, so the
adapter adds a thin PyTorch fallback trainer around the unchanged GraphEC model.

## Dataset Mapping

Each split group must contain `train.parquet`, `val.parquet`, and `test.parquet`.
Required columns are `uniprot_id`, `sequence`, and `ec_number`.

The adapter deduplicates each split file by `uniprot_id` and truncated sequence,
aggregates full EC labels, and drops partial labels such as `3.2.2.-` because
GraphEC predicts full EC classes. The output vocabulary is expanded from all full
EC labels observed across train, validation, and test for each split group.

Sequences are capped at 1022 residues by default, matching the maximum length in
GraphEC's shipped EC training set. Rows containing non-canonical amino acids are
dropped because GraphEC's residue encoder supports only the 20 canonical residues.

GraphEC uses ESMFold-predicted structures, so dataset PDB columns are retained
only as manifest provenance and are not used as model inputs.

## Commands

Prepare one split group and populate the shared feature cache:

```bash
conda run -n graphec python -m emulator_bench.cache_features \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits \
  --prottrans-model-path /path/to/Prot-T5-XL-U50
```

Train one seed:

```bash
conda run -n graphec python -m emulator_bench.train \
  --split-group random_splits \
  --seed 0 \
  --epochs 35 \
  --precision auto
```

Evaluate one seed:

```bash
conda run -n graphec python -m emulator_bench.evaluate \
  --split-group random_splits \
  --seed 0 \
  --eval-split test
```

Queue cache and train with task-spooler, then run evaluation directly on CUDA 0
when `--wait` is used:

```bash
conda run -n graphec python -m emulator_bench.queue_pipeline \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits \
  --seed 0 \
  --execution-mode ts \
  --epochs 35 \
  --prottrans-model-path /path/to/Prot-T5-XL-U50 \
  --eval-mode direct \
  --eval-cuda-device 0 \
  --wait
```

One-epoch smoke test directly on GPU 0, without waiting on `ts`:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n graphec python -m emulator_bench.queue_pipeline \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits \
  --seed 0 \
  --execution-mode direct \
  --epochs 1 \
  --limit-per-split 8 \
  --cuda-visible-devices 0 \
  --eval-cuda-device 0 \
  --prottrans-model-path /path/to/Prot-T5-XL-U50 \
  --wait
```

Aggregate metrics:

```bash
conda run -n graphec python -m emulator_bench.aggregate_results
```

## Outputs

- Manifests: `emulator_bench/runs/<split_group>/manifests/{train,val,test}.csv`.
- FASTA views: `emulator_bench/runs/<split_group>/fastas/{train,val,test}.fasta`.
- Expanded EC vocab: `emulator_bench/runs/<split_group>/vocab/ec_vocab.json`.
- Shared feature cache: `emulator_bench/cache/enzyme_classification_dataset/`.
- Seed checkpoints: `emulator_bench/runs/<split_group>/seeds/<seed>/checkpoints/fold*.pt`.
- Metrics and ranked outputs: `emulator_bench/runs/<split_group>/seeds/<seed>/results/`.

CARE-ranked CSVs keep `Entry`, `EC number`, and `Sequence`, then append rank
columns `0,1,2,...`. Metrics include GraphEC-style AUC, AUPR, F1, MCC, CARE Task
1 hierarchical accuracy for `k=1` and `k=20`, and supplemental MRR/hit@k.

## Notes

- Environment name: `graphec`.
- Queue command: `ts` for cache/train in `--execution-mode ts`. Smoke tests can bypass the
  queue entirely with `--execution-mode direct`, and evaluation defaults to direct execution on
  CUDA 0.
- Default real-run seeds in `queue_multiple_seeds.sh`: `0 1 2`.
- Label diffusion is enabled during evaluation by default and uses GraphEC's
  bundled Diamond executable.
