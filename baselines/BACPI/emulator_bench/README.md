# emulator_bench

This bench adds a Leak-CURBER-style retraining workflow to `BACPI` without modifying the original `code/` model implementation.

It is designed for BACPI affinity split trees under:

- `<PROJECT_ROOT>/data/processed/baselines/BACPI/<value_type>`

Supported `value_type` folders:

- `ki`
- `kd`
- `ec50`
- `ic50`

## What The Model Uses

Inputs per sample:

- Compound `smiles`
- Protein `sequence`
- Affinity target, defaulting to `log10_value`

Bench feature path:

- Compounds use BACPI's native RDKit graph featurization:
  - atom-environment graph fingerprint ids from the original `atom_features` pipeline
  - adjacency matrix with self-loops
  - Morgan fingerprint bits with radius `2` and `1024` bits
- Proteins use BACPI's native sequence featurization:
  - amino-acid 3-gram tokenization with the original `-SEQ=` framing

Model path:

- The bench imports the original `BACPI` module from [code/model.py](<PROJECT_ROOT>/baselines/BACPI/code/model.py)
- The default architecture remains the original affinity configuration:
  - `gat_dim=50`
  - `num_head=3`
  - `dropout=0.1`
  - `alpha=0.1`
  - `comp_dim=80`
  - `prot_dim=80`
  - `latent_dim=80`
  - `window=5`
  - `layer_cnn=3`
  - `layer_out=3`

## How Embeddings Are Cached

BACPI does not use external pretrained embeddings, so this bench caches the expensive native input featurization instead of recomputing it every run.

Global cache layout:

- `embeddings/vocabs/featurizer.pkl`
  - frozen BACPI vocabularies for atom symbols, bonds, graph fingerprints, edges, and protein 3-grams
- `embeddings/compounds/<hash-prefix>/<hash>.npz`
  - one file per unique canonical SMILES
  - stores compound graph fingerprint ids, adjacency matrix, and Morgan fingerprint bits
- `embeddings/proteins/<hash-prefix>/<hash>.npz`
  - one file per unique normalized sequence
  - stores the 3-gram token ids used by BACPI
- `embeddings/manifest.json`
  - records cache settings, token counts, and discovered split coverage

Shared multi-target cache:

- When you pass multiple `value_type` values, the bench now defaults to one shared cache under `<PROJECT_ROOT>/data/processed/baselines/BACPI/embeddings_shared`
- That shared cache is built from the union of all requested split trees, so BACPI featurization work for repeated compounds and proteins is only done once across `ki`, `kd`, `ec50`, and `ic50`

Cache behavior:

- Features are computed once per unique compound or protein and reused across train, validation, test, Optuna trials, and multi-GPU retrains
- Shared-cache mode saves additional compute across value types because many compounds and protein sequences overlap between targets
- Split files with invalid SMILES are cleaned in place during cache building by default, so the TVT filtering cost is paid once and later runs reuse the cleaned files
- Cache rebuilding is incremental by default: existing compound and protein `.npz` files are left in place and only missing cache entries are written
- If new split data introduces compound/protein features that were not in an existing vocabulary, the cache builder extends the vocabulary while preserving existing token IDs
- The train wrapper refuses to run if required cache files are missing, so repeated RDKit and tokenization work does not silently creep back in
- The train path can preload the cache into CPU memory to reduce I/O stalls

## Bench Scripts

Core scripts:

- `cache_embeddings.py`
  - scans the requested split jobs
  - builds the frozen BACPI featurizer vocabularies
  - caches all unique compounds and proteins once
- `train_single_target_tvt.py`
  - trains BACPI on one explicit train/val/test split
  - writes `bestmodel.pth`, `final_results_train.csv`, `final_results_val.csv`, `final_results_test.csv`, and val/test prediction CSVs
- `predict_single_target.py`
  - evaluates a saved checkpoint on one split file

Benchmark and tuning scripts:

- `run_split_benchmarks.py`
  - sequential runner across discovered split jobs and seeds
- `tune_optuna.py`
  - tunes only retraining-safe optimization hyperparameters
- `launch_parallel_optuna.py`
  - launches multiple single-GPU Optuna workers against one shared study
- `launch_parallel_retrain_from_optuna.py`
  - retrains many split jobs in parallel across multiple GPUs from the best Optuna result

## Enhancements In This Bench

Compared with the original repo flow, this bench adds:

- Explicit train/val/test split loading from parquet or CSV
- One-time reusable feature caching for compounds and proteins
- Automatic mixed precision:
  - `bf16` on Ampere-or-newer CUDA devices
  - `fp16` on older CUDA devices
- TF32 enabled for CUDA matmul and cuDNN where available
- Pinned-memory DataLoaders, worker prefetching, and persistent workers
- Length-aware bucketed batching to reduce padding waste
- No extra batch-shape rounding by default; `pad_to_multiple` now defaults to `1` so batches are padded only to the natural max length inside the batch
- Optional cache preloading so GPUs spend less time waiting on disk
- Best-checkpoint validation selection and final train/val/test metric dumps
- Optuna tuning restricted to retraining-safe hyperparameters:
  - `batch_size`
  - `lr`
  - `weight_decay`
  - `step_size`
  - `gamma`
  - `grad_clip`
  - `patience`
- Multi-GPU parallel retraining from the best Optuna result

## Notes

- The bench targets BACPI affinity tasks only for `Ki`, `Kd`, `EC50`, and `IC50`
- The default target is `log10_value`; pass `--target_col value` if you want the raw split column instead
- The cache is specific to the split universe used when `cache_embeddings.py` runs. If you expand to new split files with unseen compounds or proteins, rebuild the cache or point the run at a different `embeddings/` directory
- If you run multi-value retraining without specifying `--embeddings_dir`, the launcher will automatically use the shared cache path
- Rows with invalid RDKit-parsable SMILES are dropped automatically during cache building, training, and prediction instead of crashing the run
- If cache building sees invalid SMILES, it overwrites the split file with the cleaned rows unless you pass `--no_clean_splits_in_place`
- Use `conda run -n mldb ...` for all commands in this repo
