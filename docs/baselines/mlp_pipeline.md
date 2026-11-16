# Baseline MLP Pipeline Options

This pipeline has two layers of controls:

- Hydra controls the training job itself: data, model, optimizer, scheduler, callbacks, logging, and search space.
- `scripts/run_baseline_mlp_pipeline.py` controls orchestration: endpoint queueing, GPU slots, retries, embedding generation, and W&B mode.

## Hydra Configs To Edit

| What | File | Common options |
| --- | --- | --- |
| HPO search space | `configs/hparams_search/baseline_regression_optuna4.yaml` | `hydra.sweeper.params`, `n_trials`, sampler settings, `optimized_metric` |
| Model defaults | `configs/model/baseline_regression.yaml` | `hidden_dims`, `dropout`, `activation`, optimizer/scheduler factories |
| Data and modalities | `configs/data/baseline_regression.yaml` | `endpoint`, `split`, `threshold`, `batch_size`, `modalities`, cache roots/key columns |
| Trainer behavior | `configs/trainer/default.yaml`, `configs/trainer/gpu.yaml` | `max_epochs`, precision, `check_val_every_n_epoch`, accelerator/devices |
| Callbacks | `configs/callbacks/default.yaml` | checkpoint monitor, early stopping patience, model summary, progress callback |
| W&B logging | `configs/logger/wandb.yaml` | project, entity, save dir, offline/online behavior |

## Orchestrator CLI Options

For reproducible runs, prefer a YAML run profile:

```bash
python scripts/run_baseline_mlp_pipeline.py \
  --run_config configs/pipeline/baseline_mlp_optuna100.yaml
```

The orchestrator merges defaults, then the YAML profile, then explicit CLI flags. This means a profile can hold the stable run definition while CLI flags remain useful for quick changes:

```bash
python scripts/run_baseline_mlp_pipeline.py \
  --run_config configs/pipeline/baseline_mlp_optuna100.yaml \
  --pipeline_id mlp_optuna100_retry \
  --gpus 1 2 \
  --n_trials 20
```

Useful profile files:

| Profile | Purpose |
| --- | --- |
| `configs/pipeline/baseline_mlp_optuna100.yaml` | Torch MLP HPO/final run with 100 Optuna trials per endpoint. |
| `configs/pipeline/baseline_sklearn_ridge_optuna100.yaml` | Ridge estimator run with a Ridge-specific search space. |
| `configs/pipeline/baseline_sklearn_random_forest_optuna100.yaml` | RandomForest estimator run with a RandomForest-specific search space. |
| `configs/pipeline/baseline_xgboost_cpu_optuna100.yaml` | XGBoost on CPU for the small endpoints: ec50, kcat, kd, ki, km. Five concurrent jobs. |
| `configs/pipeline/baseline_xgboost_optuna100.yaml` | XGBoost on GPU for the IC50 endpoint only. |

The classification and retrieval endpoints use `scripts/run_baseline_classification_pipeline.py`. It reuses the same orchestrator and takes the same `--run_config` option. Its endpoints are `classification` and `retrieval`.

| Classification and retrieval profile | Purpose |
| --- | --- |
| `configs/pipeline/baseline_classification_mlp_optuna100.yaml` | MLP, both endpoints. |
| `configs/pipeline/baseline_classification_mlp_protein_ret.yaml` | MLP, retrieval only, with protein inputs. |
| `configs/pipeline/baseline_classification_rapids_random_forest_optuna100.yaml` | RAPIDS RandomForest, both endpoints. |
| `configs/pipeline/baseline_classification_sklearn_logistic_regression_optuna100.yaml` | Logistic regression, retrieval only. |
| `configs/pipeline/baseline_classification_xgboost_optuna100.yaml` | XGBoost, retrieval only. |

Inside a profile, fixed Hydra values go under `hydra.common`, HPO search space goes under `hydra.hpo.search_space`, and advanced manual dotlist entries go under `hydra.raw_overrides`.

Run the full pipeline from the repo root:

```bash
python scripts/run_baseline_mlp_pipeline.py \
  --gpus 1 2 3 \
  --max_slots_per_gpu 1 \
  --hpo_workers_per_endpoint 1 \
  --n_trials 50 \
  --max_epochs 200 \
  --wandb_project EMULaToR
```

Useful orchestration flags:

| Flag | Effect |
| --- | --- |
| `--gpus 1 2 3` | GPUs available to the queue. Each subprocess sees its assigned GPU as `cuda:0`, except MolR embedding, which can see all allowed GPUs. |
| `--max_slots_per_gpu N` | Number of concurrent training jobs per GPU. The profile sets the value (for example 4 in the MLP profile, 1 in the GPU XGBoost profile). Raise it only if GPU utilization is low and memory is safe. |
| `--hpo_workers_per_endpoint N` | Concurrent Optuna workers per endpoint sharing the same endpoint SQLite study. |
| `--n_trials N` | Global trial cap per endpoint study. |
| `--max_epochs N` | Trainer max epochs for HPO and final runs. |
| `--skip_embeddings` | Skip embedding generation and require caches to already exist. |
| `--refresh_embedding_superset` | Recompute the deduplicated embedding input supersets instead of reusing the cache. |
| `--wandb_offline` | Run W&B in offline mode. |
| `--pipeline_id ID` | Resume or continue an existing pipeline run directory. |
| `--progress_backend rich\|plain` | Choose terminal progress renderer. Rich is the default dashboard. |
| `--stream_all_jobs` | Debug mode for streaming subprocess logs; default dashboards keep HPO/final logs file-only. |

## Direct Hydra Multiruns

Direct Hydra runs are useful for small experiments:

```bash
python src/train.py -m \
  data=baseline_regression \
  model=baseline_regression \
  trainer=gpu \
  logger=wandb \
  scheduler=cosine_warm_restarts \
  data.endpoint=ki \
  data.split=enzyme_structure_splits \
  data.modalities='[smited,ecfp,molr,esmc,esm3]' \
  trainer.max_epochs=20 \
  model.dropout=0.0,0.1 \
  optimizer.lr=1e-4,3e-4
```

For Optuna through the local Optuna 4 sweeper:

```bash
python src/train.py -m \
  data=baseline_regression \
  model=baseline_regression \
  trainer=gpu \
  logger=wandb \
  hparams_search=baseline_regression_optuna4 \
  data.endpoint=ki \
  data.split=enzyme_structure_splits \
  hydra.sweeper.n_trials=10 \
  hydra.sweeper.storage=sqlite:///logs/manual_ki_optuna.db
```

Use direct multiruns for quick local checks. Use the orchestrator for the full endpoint-parallel HPO plus `6 x 7 x 3` final training schedule, because Hydra's default launcher does not manage GPU slots across endpoints.
