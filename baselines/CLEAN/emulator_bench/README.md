# CLEAN Leak-CURBER Adapter

This wrapper adapts `data/processed/datasets/enzyme_classification_dataset` to CLEAN's
protein-to-EC workflow.

It creates CLEAN TSV/FASTA views with columns `Entry`, `EC number`, and `Sequence`,
deduplicated per split file by `uniprot_id` and truncated sequence. Partial EC labels
containing `-` are removed by default because CLEAN's full EC training and CARE Task 1
evaluation expect supported EC labels.

The wrapper stores sequences truncated to 1024 residues in the CLEAN views. ESM-1b
has `max_positions=1024` including BOS/EOS special tokens, so the feature extractor
caps model input at 1022 residues for that model and logs the cap.

## Commands

Prepare cache and CLEAN inputs for one split group:

```bash
conda run -n clean python -m emulator_bench.cache_features \
  --dataset-root ../../data/processed/datasets/enzyme_classification_dataset \
  --split-group random_splits
```

Train with the native triplet script:

```bash
conda run -n clean python -m emulator_bench.train \
  --split-group random_splits \
  --env-name clean \
  --epochs 7000 \
  --precision auto
```

Evaluate the produced checkpoint:

```bash
conda run -n clean python -m emulator_bench.evaluate \
  --split-group random_splits \
  --eval-split both
```

Queue cache, train, and evaluation through this repository's task-spooler command
name, `ts`. By default each queued cache, train, and eval job requests one GPU
with task-spooler's native GPU scheduler:

```bash
conda run -n clean python -m emulator_bench.queue_pipeline \
  --split-group random_splits \
  --env-name clean \
  --spooler-bin ts \
  --gpus-per-job 1
```

To restrict the scheduler to a subset of GPUs, start or invoke task-spooler with
`TS_VISIBLE_DEVICES`, for example `TS_VISIBLE_DEVICES=1,2,3 ts -S 3`, then queue
the pipeline normally. Use `--gpus-per-job 0` only for CPU/debug runs.

One-epoch smoke test on GPU 3 with reduced data:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n clean python -m emulator_bench.cache_features \
  --split-group random_splits \
  --limit-per-split 32
CUDA_VISIBLE_DEVICES=3 conda run -n clean python -m emulator_bench.train \
  --split-group random_splits \
  --epochs 1 \
  --model-name emulator_random_splits_smoke
CUDA_VISIBLE_DEVICES=3 conda run -n clean python -m emulator_bench.evaluate \
  --split-group random_splits \
  --model-name emulator_random_splits_smoke \
  --eval-split test
```

Aggregate completed seed-wise metrics:

```bash
conda run -n clean python -m emulator_bench.aggregate_results \
  --runs-root emulator_bench/runs
```

## Outputs

- Shared ESM cache: `emulator_bench/cache/esm1b/<uniprot_id>.pt`.
- Split manifests: `emulator_bench/runs/<split_group>/manifests/{train,val,test}.csv`.
- CLEAN views: `app/data/emulator_<split_group>_{train,val,test}.csv`.
- Native distance maps: `app/data/distance_map/emulator_<split_group>_train*.pkl`.
- Native CLEAN checkpoints: `app/data/model/<model_name>.pth`.
- Seed checkpoints: `emulator_bench/runs/<split_group>/seeds/<seed>/checkpoints/<model_name>.pth`.
- Seed train metadata: `emulator_bench/runs/<split_group>/seeds/<seed>/train.json`.
- CARE-ranked CSVs: `emulator_bench/runs/<split_group>/seeds/<seed>/results/*_results_df.csv`.
- Metrics JSON: `emulator_bench/runs/<split_group>/seeds/<seed>/results/*_metrics.json`.
- Aggregates: `emulator_bench/runs/aggregated_seed_metrics_{long,summary}.csv`.

The CARE-ranked CSV schema keeps `Entry`, `EC number`, and `Sequence`, then appends
rank columns `0,1,2,...` from nearest to farthest EC cluster center. Metrics include
CLEAN native weighted precision, recall, F1, AUC, exact-match accuracy, CARE Task 1
hierarchical accuracy at levels 4 through 1 for `k=1` and `k=20`, and supplemental
MRR/hit@k.

The root-level `emulator_bench/runs/<split_group>/train_seed<seed>.json` file is
kept as a compatibility pointer, but new training and evaluation artifacts are
seed-scoped under `seeds/<seed>/`.

Native CLEAN training is preserved. The only native edit is to `app/train-triplet.py`
to expose CUDA AMP precision, training progress bars, and one-epoch-safe checkpoint
cleanup.
