# emulator_bench

Split-driven retraining workflow for the eMOSAIC **TrustAffinity** regression path. It wraps the
existing `code/BindingAffinityModule` model so Leak-CURBER-style split trees can be retrained,
tuned, and benchmarked without touching the original code.

Default base directory follows the Leak-CURBER baseline convention:

- `<PROJECT_ROOT>/data/processed/baselines/eMOSAIC`

Pass `--base_dir` to point elsewhere.

## What the model uses

Inputs per sample (columns in the split parquet / csv):

- Protein sequence from `sequence`
- Ligand SMILES from `smiles`
- Target (pKi) from `log10_value` by default (override via `--target_col`)

Model graph (paper-faithful, uncertainty-quantification module excluded):

- **Protein sequence module**: ESMFold-v1 (loaded via `torch.hub`, `facebookresearch/esm:main`) → structure-module single representation `s_s`
  (dim 384, per-residue). Padded to `max_length=700` with a binary mask; passed through a 5-stage
  2D ResNet (`ResnetEncoderModel` from `code/BindingAffinityModule/models/resnet.py`). The flattened
  ResNet output is the "refined protein embedding".
- **Ligand module**: RDKit → 2D molecular graph (atom/bond one-hots from
  `models/ligand_graph_features.py`) → 5-layer-by-default GIN (`GNN` from `models/model_Yang.py`) →
  sum-pooling for a ligand-level embedding.
- **PLI module**: `AttentivePooling` cross-attention between chem and protein embeddings →
  concatenation → 3-layer MLP regressor (matches the original `BindingModel2`).

No uncertainty-quantification head is built — this bench trains the binding-affinity regressor
only, matching the "regression part" of the paper.

## How embeddings are cached (one-time, reusable)

Protein cache:

- Key: sha256 of the normalized sequence (`U`/`Z`/`O`/`B` → `X`).
- Path: `embeddings/proteins/<hash_prefix>/<hash>.npz`.
- Payload: per-residue ESMFold `s_s` embedding + length. Stored as `float16` by default.
- Long sequences (> 700 residues) are truncated to the first 700 residues at both cache and
  train time (normalized sequence is sliced before hashing, so the truncated form is cached).

Ligand cache:

- Key: sha256 of the SMILES string.
- Path: `embeddings/ligands/<hash_prefix>/<hash>.pt` (pickled dict of numpy arrays).
- Payload: `x`, `edge_index`, `edge_attr`, `num_nodes` — the pytorch-geometric graph descriptor.

Cache behavior:

- A sample is recomputed only if its cache file is missing (or `--overwrite` is passed).
- All downstream scripts — training, Optuna trials, multi-GPU retrain, inference — hit the same
  cache, so ESMFold runs at most once per unique sequence, ever.

## Bench scripts

- `cache_embeddings.py`: discovers split files under `--base_dir`, builds and saves all protein
  and ligand caches. Run this before training.
- `train_single_target_tvt.py`: trains on one explicit train/val/test split (or one discovered
  split group). Writes `bestmodel.pth`, `logfile.csv`, `final_results_{train,val,test}.csv`,
  `pred_label_{train,val,test}.csv`, and `run_summary.json`.
- `run_split_benchmarks.py`: sequentially runs training across discovered split groups/thresholds
  (random, enzyme_sequence, substrate, group_shuffle, plus threshold subfolders).
- `tune_optuna.py`: Optuna study over retraining-safe hparams only (see below). Averages the
  chosen metric across requested split groups × seeds as the objective.
- `launch_parallel_optuna.py`: launches multiple single-GPU Optuna workers against a shared SQLite
  study. Supports `--trials_per_gpu` for concurrent workers per GPU.
- `launch_parallel_retrain_from_optuna.py`: drains a queue of split jobs across N GPU-slots and
  retrains each from the best Optuna hparams. Supports `--trials_per_gpu` for concurrent retrains
  per GPU.
- `predict_single_target.py`: runs inference from a saved checkpoint on any parquet/csv with the
  same schema.

## What's tuned vs. what's fixed

To keep results comparable with the original paper, Optuna tunes only optimization knobs:

- `lr`, `weight_decay`, `cosine_t0`, `clip_grad`, `batch_size`

Architecture stays locked:

- `num_layer=2`, `emb_dim=300`, `gnn_type=gin`, `max_length=700`, `prot_input_dim=384`
- ResNet block sizes `[16, 32, 64, 32, 16]` and depths `[2, 2, 2, 2, 2]`
- 3-layer MLP head `combined_dim → 1024 → 512 → 128 → 1`

## Enhancements in this bench

Compared with the original `code/BindingAffinityModule/main.py`:

- Explicit train/val/test split loading from parquet or CSV (no hardcoded paths).
- **One-time-only** ESMFold embedding cache + one-time RDKit graph cache. ESMFold is the
  bottleneck; the bench runs it once per unique protein across all retrains and trials.
- Per-sequence cache key makes the cache survive split reshuffles — no recomputation when moving
  from random_splits to substrate_splits.
- Automatic mixed precision: **bf16** on Ampere-and-newer CUDA devices, **fp16** on older CUDA,
  fp32 on CPU. `GradScaler` only turns on in the fp16 branch.
- TF32 enabled on matmul and cuDNN where supported; `cudnn.benchmark=True` for fastest kernels.
- pyg `DataLoader` with `pin_memory`, `prefetch_factor=4`, `persistent_workers`, and in-memory
  protein LRU + fully preloaded ligand graphs — keeps the GPU fed so CUDA stays pegged.
- Optional `torch.compile` of the full model graph.
- Early stopping + best-checkpoint restoration before the final eval pass.
- Multi-GPU Optuna tuning and multi-GPU parallel retraining from best hparams, including
  `--trials_per_gpu` for multiple concurrent processes per GPU.
- Sequences over max_seq_len are **truncated** (not dropped) at both caching and train time.

## Notes

- If your split column names differ, pass `--sequence_col`, `--smiles_col`, `--target_col`.
- If your baseline folder isn't named `eMOSAIC`, pass `--base_dir`.
- If an existing checkpoint was produced with a different `max_length` or `prot_input_dim`, the
  wrapper honours the checkpoint's settings when running `predict_single_target.py`.
- The bench intentionally does not touch the uncertainty-quantification (AnomalyDetection) path.
