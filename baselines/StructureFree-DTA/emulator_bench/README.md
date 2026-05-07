# emulator_bench

This bench adds split-driven retraining wrappers to `StructureFree-DTA` for Leak-CURBER-style train/val/test trees under:

- `<PROJECT_ROOT>/data/processed/baselines/StructureFree-DTA`

Override that path with `--base_dir` if your baseline folder has a different name.

## Inputs

Each split file can be parquet or CSV. Defaults are:

- Protein FASTA sequence: `sequence`
- Molecule SMILES: `smiles`
- Regression target: `log10_value`

Use `--sequence_col`, `--smiles_col`, and `--target_col` if the actual columns differ. The original target repo used `Molecule Sequence`, `Protein Sequence`, and `Binding Affinity`; the bench defaults are set for Leak-CURBER baseline files.

## Embeddings

The original model embeds inputs with:

- Protein encoder: `facebook/esm2_t6_8M_UR50D`, capped at 1024 tokens
- Molecule encoder: `DeepChem/ChemBERTa-77M-MLM`, capped at 128 tokens
- Pooling: attention-mask-aware mean pooling over the final hidden state

This bench computes those encoder embeddings once and then trains only the StructureFree fusion/regression head. The cached model keeps the original Residual Inception blocks and MLP regressor, but skips repeated transformer forward passes during every epoch, Optuna trial, and seed rerun.

## Cache Layout

Embeddings are stored under:

- `embeddings/proteins/<model-hash>/<hash-prefix>/<hash>.npz`
- `embeddings/molecules/<model-hash>/<hash-prefix>/<hash>.npz`

Cache keys include the raw input, encoder model name, max token length, and pooling mode. Existing files are never recomputed unless `cache_embeddings.py --overwrite` is passed. A cache manifest is written to `embeddings/manifest.json`.

## Scripts

- `cache_embeddings.py`: scans selected split groups and builds the one-time ESM2/ChemBERTa cache
- `train_single_target_tvt.py`: trains one explicit train/val/test job from cached embeddings
- `run_split_benchmarks.py`: runs discovered split jobs sequentially and writes summary CSVs
- `tune_optuna.py`: tunes retraining-safe optimizer/schedule hyperparameters
- `launch_parallel_optuna.py`: runs multiple Optuna workers across GPUs with shared SQLite storage
- `launch_parallel_retrain_from_optuna.py`: retrains many split/seed jobs across GPUs from the best Optuna result
- `commands.txt`: copy-ready command examples

## Enhancements

- One-time encoder embedding cache for all split jobs
- Automatic CUDA AMP: bf16 on Ampere-or-newer GPUs, fp16 on older CUDA GPUs, fp32 on CPU
- TF32 enabled for CUDA matmul/cuDNN
- Large progressive batch schedule support, defaulting to `1:64,20:512,60:2048`
- Cosine schedule with warmup, AdamW, gradient clipping, and early stopping
- Optional `torch.compile`
- DataLoader controls for pinned memory, persistent workers, prefetching, and embedding preloading
- Optuna tuning restricted to optimizer, loss, scheduler, clipping, patience, and batch schedule parameters, leaving encoder choice and head architecture fixed by default
- Multi-GPU parallel Optuna and multi-GPU retraining launchers

## Outputs

Each run writes:

- `bestmodel.pth`
- `checkpoint_last.pt`
- `logfile.csv`
- `pred_label_train.csv`, `pred_label_val.csv`, `pred_label_test.csv`
- `final_results_train.csv`, `final_results_val.csv`, `final_results_test.csv`
- `run_summary.json`

Metrics include MSE, RMSE, MAE, R2, Pearson, Spearman, CI, and training loss.
