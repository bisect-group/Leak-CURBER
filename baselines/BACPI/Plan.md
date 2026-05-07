## 1. Baseline Summary From README

- Baseline repository: `BACPI`
- Task addressed by the baseline: compound-protein interaction prediction and compound-protein binding affinity prediction with the BACPI GAT + CNN + bi-directional attention model
- README and docs inspected: `README.md`, `code/main.py`, `code/data_process.py`, `code/data_prepare.py`, `code/model.py`, `code/utils.py`
- Recommended training command from docs: `python main.py -task affinity -dataset Kd`
- Recommended evaluation command from docs: no separate evaluator is documented; `code/main.py` trains and prints dev/test metrics during and after training
- Default config or hyperparameter source: CLI defaults in `code/main.py`
- Notes about any doc or code mismatch:
  - The README documents only `train.txt` and `test.txt`, but `code/main.py` internally creates a random dev split from train data via `split_data(train_data, 0.1)`.
  - The original preprocessing rebuilds RDKit graph features, Morgan fingerprints, and protein 3-gram encodings every run.
  - The original preprocessing writes `fingerprint_dict` to `atom_dict`; this is intentional because the model embeds graph fingerprint ids rather than raw atom symbols.

## 2. Expected Input Format

- Raw file format(s):
  - Original baseline: headerless `train.txt` / `test.txt` files with `smiles,sequence,label`
  - Leak-CURBER splits: `.parquet` or `.csv`
- Required columns or fields:
  - `smiles`
  - `sequence`
  - target column, defaulting to `log10_value`
- Label or target fields:
  - BACPI affinity bench will default to `log10_value`
  - `value` remains usable through `--target_col`
- Identifier fields:
  - No identifier is required by the model
  - `smiles_hash` can exist in Leak-CURBER files but is not required
- Any assumptions about train, validation, and test partitioning:
  - The bench will require explicit `train`, `val`, and `test` split files
  - The split tree follows `<PROJECT_ROOT>/data/processed/baselines/BACPI/<value_type>/<split_group>/...`

## 3. Featurization and Preprocessing Path

- Native preprocessing entrypoint: `code/data_process.py::training_data_process`
- Native featurization functions, scripts, or modules:
  - `create_atoms`
  - `create_ijbonddict`
  - `atom_features`
  - `create_adjacency`
  - `get_fingerprints`
  - `split_sequence`
- Required intermediate artifacts:
  - Frozen BACPI featurizer vocabularies for atoms, bonds, graph fingerprints, edges, and protein 3-grams
  - Cached per-unique compound feature files
  - Cached per-unique protein feature files
- Where cached features will be stored:
  - `<PROJECT_ROOT>/data/processed/baselines/BACPI/<value_type>/embeddings/`
  - `embeddings/compounds/<hash-prefix>/<hash>.npz`
  - `embeddings/proteins/<hash-prefix>/<hash>.npz`
  - `embeddings/vocabs/featurizer.pkl`
  - `embeddings/manifest.json`
- Cache format for train, validation, and test:
  - The bench will cache unique compounds and proteins once globally per `value_type`
  - Split files will be read directly and mapped to cached keys at runtime so no repeated RDKit or sequence tokenization happens across train, validation, test, Optuna trials, or parallel retrains

## 4. Training and Evaluation Entrypoints

- Training entrypoint: new `emulator_bench/train_single_target_tvt.py`
- Evaluation entrypoint: training wrapper will evaluate the best validation checkpoint on train/val/test and write CSV metrics; a prediction wrapper will also be added for standalone checkpoint evaluation
- Checkpoint location or discovery rule:
  - Default output under `<split_root>/bacpi_results/seed_<seed>/`
  - Best checkpoint at `bestmodel.pth`
  - Best state dict at `bestmodel_state_dict.pth`
- Default settings that must remain unchanged:
  - BACPI model architecture in `code/model.py`
  - Original affinity defaults from `code/main.py` unless explicitly overridden:
    - `lr=5e-4`
    - `step_size=10`
    - `gamma=0.5`
    - `batch_size=16`
    - `num_epochs=20`
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
- Exact wrapper-to-baseline command mapping:
  - The bench will import `BACPI` from `code/model.py`
  - The bench will preserve BACPI’s loss, optimizer family, and scheduler defaults while replacing only the data plumbing and launcher layer needed for explicit train/val/test splits, AMP, reusable caching, and multi-job orchestration

## 5. Dataset and Split Mapping

- User dataset path: `<PROJECT_ROOT>/data/processed/baselines/BACPI/<value_type>`
- User split definition path or files:
  - Examples already present:
    - `ki/random_splits/train.parquet`
    - `ki/enzyme_sequence_splits/threshold_0.05/train.parquet`
    - `kd/enzyme_structure_splits/threshold_0.7/test.parquet`
    - `ec50/substrate_splits/threshold_0.55/val.parquet`
    - `ic50/uniprot_time_splits/test.parquet`
- Mapping from user schema to baseline schema:
  - `smiles` -> BACPI compound graph + Morgan fingerprint featurization
  - `sequence` -> BACPI protein 3-gram tokenization
  - `log10_value` by default -> BACPI affinity regression label
- Mapping from user train, validation, and test splits to baseline expectations:
  - The wrapper will use the explicit Leak-CURBER train, validation, and test files directly instead of creating a random validation subset from train
  - Vocabularies will be frozen from the discovered dataset universe for a given `value_type`, then reused for all requested split jobs
- Assumptions or blockers:
  - Affinity tasks only for this bench (`Ki`, `Kd`, `EC50`, `IC50`)
  - Input schema must contain valid SMILES and protein sequences
  - No blocker remains because the BACPI split tree already exists locally

## 6. Files To Add Under `emulator_bench/`

- `emulator_bench/README.md`: documents the bench, inputs, cache layout, AMP policy, commands, and enhancements
- `emulator_bench/common.py`: shared path resolution, split discovery, metrics, serialization helpers, and AMP selection
- `emulator_bench/feature_pipeline.py`: BACPI-native featurizer vocabulary builder and reusable per-unique cache generation
- `emulator_bench/dataset.py`: cached dataset, preloading, collation, and bucketed batching
- `emulator_bench/modeling.py`: thin BACPI model builder around `code/model.py`
- `emulator_bench/cache_embeddings.py`: one-time cache builder for compounds and proteins
- `emulator_bench/train_single_target_tvt.py`: explicit train/val/test retraining wrapper with AMP and checkpointing
- `emulator_bench/predict_single_target.py`: standalone checkpoint evaluation wrapper
- `emulator_bench/run_split_benchmarks.py`: sequential runner across discovered split jobs
- `emulator_bench/tune_optuna.py`: Optuna tuning limited to retraining-safe optimization hyperparameters
- `emulator_bench/launch_parallel_optuna.py`: multi-GPU parallel Optuna worker launcher
- `emulator_bench/launch_parallel_retrain_from_optuna.py`: multi-GPU parallel retrain launcher from the best Optuna result
- `emulator_bench/default_hparams_original.json`: frozen copy of BACPI’s original affinity defaults
- `emulator_bench/__init__.py`: package marker

## 7. Minimal Edits Outside `emulator_bench/`

- Required external edits:
  - Add root `commands.txt`
  - Add root `Plan.md`
- Why each edit is unavoidable:
  - `commands.txt` is requested by the user as a quick command reference
  - `Plan.md` is required by the adaptation workflow
- Why the edit does not change baseline behavior:
  - Both files are documentation only
  - No existing BACPI source files need to be modified

## 8. Exact Execution Plan

1. Activate or invoke the provided environment with `conda run -n mldb`.
2. Inspect the dataset and validate schema and split membership under `<PROJECT_ROOT>/data/processed/baselines/BACPI/<value_type>`.
3. Build and persist BACPI featurizer vocabularies from the requested split universe.
4. Cache unique compound and protein features once under `embeddings/`.
5. Launch BACPI retraining on explicit train/val/test splits through `emulator_bench/train_single_target_tvt.py`.
6. Evaluate the best checkpoint on train, validation, and test and write metrics plus predictions.
7. Run Optuna only over retraining-safe optimization hyperparameters, then launch parallel multi-GPU retraining from the best study result.
8. Run a small-sample smoke test with `CUDA_VISIBLE_DEVICES=3`.
9. Record outputs, metrics, cache locations, and command examples in `emulator_bench/README.md` and `commands.txt`.
