# emulator_bench

This folder adds an emulator-bench style workflow to KcatNet without modifying the original repo code paths.

What this adds:

- Shared reusable embedding cache for the Leak-CURBER KcatNet splits
- Cached train-split PNA degree histograms reused across seeds and Optuna trials
- Explicit train/val/test training script
- Direct parquet/CSV split loading
- Benchmark runner across `random_splits`, `enzyme_sequence_splits`, and `substrate_splits`
- Optuna tuner for optimization hyperparameters
- Single-checkpoint prediction script

## Data layout

Default split root:

- `<PROJECT_ROOT>/data/processed/baselines/KcatNet`

Default cache root:

- `<PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings`

Expected columns in the split files:

- `sequence`
- `smiles`
- `log10_value`

## Cache embeddings once

This scans the split tree, deduplicates normalized protein sequences and raw SMILES, and writes reusable cache files under the shared embeddings directory.

```bash
python emulator_bench/cache_embeddings.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --device cuda:0
```

Useful knobs:

- `--prot_t5_model`
- `--protein_dtype float16|float32`
- `--overwrite`

Default behavior stores cached protein embeddings in `float16` to keep the cache size practical.

You can also restrict caching to one thresholded split family with `--split_groups ... --threshold threshold_0.3`.

## Train one explicit TVT split

```bash
python emulator_bench/train_single_target_tvt.py \
  --train_path <PROJECT_ROOT>/data/processed/baselines/KcatNet/random_splits/train.parquet \
  --val_path <PROJECT_ROOT>/data/processed/baselines/KcatNet/random_splits/val.parquet \
  --test_path <PROJECT_ROOT>/data/processed/baselines/KcatNet/random_splits/test.parquet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --out_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/random_splits/kcatnet_results/seed_666 \
  --device cuda:0
```

Or resolve the split by split group and threshold:

```bash
python emulator_bench/train_single_target_tvt.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --split_group enzyme_sequence_splits \
  --threshold threshold_0.3 \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --out_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/enzyme_sequence_splits/threshold_0.3/kcatnet_results/seed_666 \
  --device cuda:0
```

Outputs include:

- `bestmodel.pth`
- `checkpoint_last.pt`
- `logfile.csv`
- `results_val.csv`
- `results_test.csv`
- `final_results_val.csv`
- `final_results_test.csv`
- `pred_label_val.csv`
- `pred_label_test.csv`
- `run_summary.csv`

## Run the full benchmark sweep

```bash
python emulator_bench/run_split_benchmarks.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --split_groups random_splits enzyme_sequence_splits substrate_splits \
  --seeds 666 \
  --device cuda:0 \
  --cache_device cuda:0
```

To run only one threshold:

```bash
python emulator_bench/run_split_benchmarks.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --split_groups enzyme_sequence_splits \
  --threshold threshold_0.3 \
  --seeds 666 \
  --device cuda:0
```

To reuse tuned hyperparameters:

```bash
python emulator_bench/run_split_benchmarks.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --hparams_json <PROJECT_ROOT>/data/processed/baselines/KcatNet/optuna_studies/kcatnet_optuna_best_hparams.json \
  --seeds 666 \
  --device cuda:0
```

Per split outputs are written under:

- `<split_root>/kcatnet_results/seed_<seed>`

Aggregate summaries are written to the base directory:

- `kcatnet_summary_runs.csv`
- `kcatnet_summary_thresholds.csv`
- `kcatnet_summary_by_split_group.csv`
- `kcatnet_summary_ranked.csv`

## Predict from a trained checkpoint

```bash
python emulator_bench/predict_single_target.py \
  --input_path <PROJECT_ROOT>/data/processed/baselines/KcatNet/random_splits/test.parquet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --ckpt_path <PROJECT_ROOT>/data/processed/baselines/KcatNet/random_splits/kcatnet_results/seed_666/bestmodel.pth \
  --out_csv /tmp/kcatnet_predictions.csv \
  --device cuda:0
```

Or resolve the input split directly:

```bash
python emulator_bench/predict_single_target.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --split_group substrate_splits \
  --threshold threshold_0.4 \
  --split test \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --ckpt_path <PROJECT_ROOT>/data/processed/baselines/KcatNet/substrate_splits/threshold_0.4/kcatnet_results/seed_666/bestmodel.pth \
  --out_csv /tmp/kcatnet_predictions.csv \
  --device cuda:0
```

## Tune hyperparameters with Optuna

```bash
python emulator_bench/tune_optuna.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --split_groups random_splits enzyme_sequence_splits substrate_splits \
  --batch_size 256 \
  --metric rmse \
  --eval_split val \
  --epochs 40 \
  --n_trials 20 \
  --device cuda:0 \
  --cache_device cuda:0
```

To tune only one threshold:

```bash
python emulator_bench/tune_optuna.py \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --split_groups enzyme_sequence_splits \
  --threshold threshold_0.3 \
  --metric rmse \
  --eval_split val \
  --n_trials 20 \
  --device cuda:0
```

Outputs:

- `optuna_studies/<study_name>_best_hparams.json`
- `optuna_studies/<study_name>_trials.csv`

## Launch parallel Optuna workers across multiple GPUs

This keeps training single-GPU and runs multiple independent Optuna workers against the same shared study.

```bash
python emulator_bench/launch_parallel_optuna.py \
  --gpus 0 1 \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --split_groups enzyme_sequence_splits \
  --threshold threshold_0.09 \
  --metric rmse \
  --eval_split val \
  --epochs 25 \
  --val_every 2 \
  --n_trials 30 \
  --batch_size 192 \
  --num_workers 16 \
  --pin_memory \
  --storage sqlite:///<PROJECT_ROOT>/data/processed/baselines/KcatNet/kcatnet_enz_seq_0_09.db
```

Notes:

- `--n_trials` is the total trial budget across all listed GPUs.
- Each worker gets one GPU via its own `CUDA_VISIBLE_DEVICES`.
- Training remains single-GPU per trial, which is usually cleaner for KcatNet than DDP.
- `--storage` is required so the workers can coordinate through one shared Optuna study.

The tuner currently searches these optimization hyperparameters:

- `lr`
- `weight_decay`
- `min_lr`
- `lr_warmup_epochs`
- `lr_warmup_start_factor`
- `clip_grad`
- `patience`

If you omit `--batch_size`, the tuner searches batch size over `8`, `16`, and `32`.
If you pass `--batch_size`, it is held fixed and removed from the search space.

The tuner keeps the scheduler fixed to:

- `scheduler=cosine`

with a linear warmup start before cosine annealing.

## Retrain from best Optuna trial across many seeds/splits on multiple GPUs

Use this when an Optuna sweep has already finished and you want to:

- load the best trial hyperparameters from the Optuna study/storage,
- retrain on one or more split groups and thresholds,
- run one experiment per GPU in parallel,
- save train/val/test prediction CSVs with ground truth,
- save train/val/test metrics including `r2`, `pcc`, `scc`, `mse`, `rmse`, `mae`,
- keep checkpoints and logs under one structured run tree.

```bash
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 2 3 \
  --base_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet \
  --embeddings_dir <PROJECT_ROOT>/data/processed/baselines/KcatNet/embeddings \
  --study_name kcatnet_optuna \
  --storage sqlite:///<PROJECT_ROOT>/data/processed/baselines/KcatNet/kcatnet_enz_seq_0_09.db \
  --split_groups enzyme_sequence_splits substrate_splits \
  --thresholds threshold_0.09 threshold_0.3 \
  --seeds 101 202 303 404 \
  --epochs 80 \
  --num_workers 8 \
  --pin_memory
```

Default output root:

- `<PROJECT_ROOT>/data/processed/baselines/KcatNet/retrain_from_optuna/<study_name>`

Per-run layout:

- `<output_root>/<split_group>/<split_name>/seed_<seed>/train/` (includes `bestmodel.pth` and `checkpoint_last.pt`)
- `<output_root>/<split_group>/<split_name>/seed_<seed>/predictions/` (train/val/test predictions CSVs)
- `<output_root>/<split_group>/<split_name>/seed_<seed>/metrics/tvt_metrics_long.csv`
- `<output_root>/<split_group>/<split_name>/seed_<seed>/logs/`

Global summaries written once at output root:

- `selected_hparams.json`
- `planned_runs.csv`
- `runs_status.csv`
- `all_tvt_metrics.csv`
- `aggregate_tvt_metrics.csv`
- `aggregate_test_metrics_ranked.csv`

## Notes

- The bench keeps the core KcatNet architecture from `models/model_kcat.py`.
- The train script computes the PNA degree histogram from the train split only.
- That train-only PNA degree histogram is cached under `<embeddings_dir>/metadata/pna_degrees` and reused across repeated runs on the same train split.
- The shared cache is external to this repo so multiple split runs can reuse it.
- `random_splits` is treated as a single benchmark job.
- Thresholded split directories are mapped to `easy`, `medium`, and `hard` by threshold order, with larger thresholds treated as easier.
