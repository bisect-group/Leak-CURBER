# emulator_bench

This bench adds a split-driven retraining workflow to `BIND` without touching the original `train_graph.py` path.

It is designed for Leak-CURBER-style train/val/test trees under:

- `<PROJECT_ROOT>/data/processed/baselines/BIND/<value_type>`

Supported value types:

- `ki`
- `kd`
- `ic50`
- `ec50`

## What The Model Uses

Inputs per sample:

- Protein sequence from the split file column `sequence`
- Ligand SMILES from the split file column `smiles`
- Target value from `log10_value` by default

Bench input encoding:

- Protein input uses cached hidden states from `facebook/esm2_t33_650M_UR50D`
- The released code path cross-attends to hidden-state indices `0`, `10`, `20`, and `30`, so the bench caches those exact tensors by default
- Ligand input uses the repo's original atom/bond featurisation path from `loading.py` and caches the resulting molecular graph once

Default training mode:

- One retrain per value type, using the original graph trunk and only the matching regression head as the supervised target
- This is the fastest and safest setup for Leak-CURBER because the split trees are already separated by head and the data sizes are very different across `ki`, `kd`, `ic50`, and `ec50`

## How Embeddings Are Cached

Shared cache root:

- `<PROJECT_ROOT>/data/processed/baselines/BIND/embeddings`

Protein cache:

- One file per normalized sequence
- Stored under `embeddings/proteins/<model-and-layer-tag>/<hash-prefix>/<hash>.npz`
- Cache payload stores only the four hidden states the released BIND code actually consumes, plus sequence length
- Cache is shared across all four value types, so `ki`, `kd`, `ic50`, and `ec50` never recompute the same sequence

Ligand cache:

- One file per SMILES
- Stored under `embeddings/ligands/graph_v1/<hash-prefix>/<hash>.npz`
- Cache payload stores node features, edge index, and edge features from the repo's original graph builder
- This means graph parsing and featurisation also happen only once

Cache behavior:

- Embeddings and graphs are only computed when the cache file does not already exist
- Repeated training runs, Optuna trials, and multi-GPU retrains all reuse the same shared cache
- The bench now truncates every protein to the first `2048` residues before caching or training
- Long proteins are then passed directly into `facebook/esm2_t33_650M_UR50D` with no sliding-window rescue logic

## Original Settings Found In The Repo

Released training script:

- `train_graph.py` sets `iterations = 100000`, `learning_rate = 1e-4`, `batch_size = 1`, `grad_accumulation_steps = 256`
- `train_graph.py` uses `AdamW(..., weight_decay=1e-3)`, `HuberLoss(delta=2.0)`, cosine annealing, and AMP
- `train_graph.py` loads `facebook/esm2_t33_650M_UR50D` and keeps it frozen by calling `.eval()`
- The bench currently defaults to a plain `11`-epoch loop for retraining, while still exposing paper-style step-budget flags if you want them
- The current bench throughput preset keeps that same effective batch size `256` but defaults to `batch_size = 64` and `grad_accumulation_steps = 4` to reduce tiny micro-batch overhead on large modern GPUs

Released model definition:

- `model_graph.py` defines five `GATv2Conv` blocks, four cross-attention blocks, `LCMAggregation`, four regression heads, plus one classifier head
- `model_graph.py` cross-attends to `hidden_states[0]`, `hidden_states[10]`, `hidden_states[20]`, and `hidden_states[30]`
- `cross_attention_graph.py` defaults the cross-attention dropout to `0.1`
- This bench uses `leaky_relu_negative_slope = 0.05` to match the paper-reported activation setting during retraining

Paper-vs-code note:

- The paper text says proteins longer than `2048` aa were removed and layers `1`, `11`, `21`, `31` were used
- The released HuggingFace ESM-2 config exposes `max_position_embeddings = 1026`, which gives a practical `1024`-residue limit for the unmodified released code
- This bench is intentionally configured for your retraining workflow to keep all rows by truncating sequences to the first `2048` residues, then passing them directly into ESM-2
- Because the model itself still has a native context limit of about `1024` residues, very long truncated sequences may fail unless the upstream stack tolerates them

## What This Bench Changes

Compared with the original repo training flow, this bench adds:

- Explicit train/val/test split loading from Leak-CURBER parquet or CSV files
- One shared model-level cache for protein hidden states and ligand graphs across all four value types
- Automatic mixed precision:
  `bf16` on Ampere-or-newer CUDA devices, otherwise `fp16`
- TF32 enabled for CUDA matmul and cuDNN where available
- Flash / memory-efficient scaled-dot-product attention enabled where PyTorch supports it
- Full-split deterministic TVT training instead of repeated random sampling from one monolithic JSON
- Optuna restricted to retraining-safe optimisation hyperparameters while keeping the released architecture fixed
- Multi-GPU parallel retraining from the best per-value Optuna result
- Removal of the decoy classifier and temperature parameter from the retrain path, while preserving the released regression trunk

## Bench Scripts

Core scripts:

- `cache_embeddings.py`: scans the split tree and builds reusable protein and ligand caches once
- `launch_parallel_cache_embeddings.py`: shards proteins and ligands by hash and fills the shared cache across multiple GPUs safely
- `train_regression_tvt.py`: trains one value-type regression model on one explicit train/val/test split
- `run_split_benchmarks.py`: benchmark runner across discovered split jobs, with optional multi-GPU parallel execution across seeds and split jobs

Tuning and parallel execution:

- `tune_optuna.py`: tunes only optimisation hyperparameters for one value type
- `launch_parallel_optuna.py`: launches multiple single-GPU Optuna workers against one shared study for one value type
- `launch_parallel_retrain_from_optuna.py`: loads the best Optuna hparams per value type and retrains many split jobs in parallel across multiple GPUs

Parallel retrain note:

- `launch_parallel_retrain_from_optuna.py` accepts multiple value types in one run
- The launcher loads `bind_<value_type>_optuna_best_hparams.json` for each requested value type unless you override the template
- Example: one command can drain `ki`, `kd`, `ic50`, and `ec50` split jobs across all visible GPU slots

## Recommended Strategy

For your use case, separate checkpoints per value type are the default and recommended choice:

- The Leak-CURBER directory tree is already split per value type
- Each head has a very different dataset size, so separate Optuna studies let the training schedule adapt cleanly
- This avoids introducing extra multi-task weighting choices that would make comparison to the released regression heads harder

The trunk architecture is still shared in code:

- Every per-head retrain uses the same released BIND graph trunk
- Only the supervised regression head changes by value type

## Outputs

Each training run writes:

- `bestmodel.pt`
- `bestmodel_state_dict.pth`
- `checkpoint_last.pt`
- `final_results_train.csv`
- `final_results_val.csv`
- `final_results_test.csv`
- `pred_label_train.csv`
- `pred_label_val.csv`
- `pred_label_test.csv`
- `logfile.csv`
- `run_summary.json`
