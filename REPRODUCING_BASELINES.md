# Reproducing Baseline Results

This document provides step-by-step instructions for reproducing all baseline results reported in the paper.
Each baseline is a published state-of-the-art model retrained from scratch on the Leak-CURBER
benchmark dataset and splits using the `emulator_bench/` adapter scripts added to each baseline
directory.

---

## Prerequisites

### 1. Obtain the code

Either extract the `05_code/` directory from the dataverse archive, or, if the project is also
mirrored as a git repository with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

All Python dependencies (including those required by every baseline's `emulator_bench/` workflow)
are declared in the root package. Install once from the project root:

```bash
pip install -e .
```

> Do **not** install each baseline separately. The root `pip install -e .` covers every workflow
> in `baselines/*/emulator_bench/`.

### 3. Hardware

A CUDA-capable GPU is strongly recommended. Embedding-cache steps run ESM, ProtT5, or ESMFold
encoders on GPU. Training can run on CPU but will be orders of magnitude slower.
All commands below assume at least one GPU (`cuda:0`); adjust `--gpus` / `--device` flags as needed.

### 4. Dataset

Download the benchmark dataset archive from the dataverse record associated with this submission.
After extraction the top-level layout is:

```
<DATASET_ROOT>/
  kinetic_params_dataset/
    kcat/
    km/
    ki/
  binding_affinity_dataset/
    ec50/
    ic50/
    kd/
```

Each value-type folder contains the same set of split-group subdirectories:

| Subdirectory | Type | Contents |
|---|---|---|
| `random_splits_grouped_sequence/` | flat | `train.parquet`, `val.parquet`, `test.parquet` |
| `random_splits_grouped_smiles/` | flat | `train.parquet`, `val.parquet`, `test.parquet` |
| `enzyme_sequence_splits/threshold_X.XX/` | thresholded | `train.parquet`, `val.parquet`, `test.parquet` |
| `enzyme_structure_splits/threshold_X.XX/` | thresholded | `train.parquet`, `val.parquet`, `test.parquet` |
| `substrate_splits/threshold_X.XX/` | thresholded | `train.parquet`, `val.parquet`, `test.parquet` |
| `uniprot_time_splits/` | flat | `train.parquet`, `val.parquet`, `test.parquet` |
| `conformer_cosine_splits/threshold_X.XX/` | thresholded | `train.parquet`, `val.parquet`, `test.parquet` |

Every parquet file has at minimum the columns `sequence` (protein sequence string),
`smiles` (substrate SMILES string), and `log10_value` (log₁₀-transformed kinetic or
affinity measurement, the regression target).

Set a convenience variable for the rest of this document:

```bash
export DATASET_ROOT=/path/to/benchmark_datasets
```

---

## Baseline–Task Matrix

| Baseline        | kcat | Km | Ki | EC50 | IC50 |
|-----------------|:----:|:--:|:--:|:----:|:----:|
| KcatNet         |  ✓   |  ✓ |    |      |      |
| CataPro         |  ✓   |    |    |      |      |
| EITLEM-Kinetics |  ✓   |  ✓ |    |      |      |
| DEKP            |      |  ✓ |    |      |      |
| eMOSAIC         |      |    |  ✓ |      |      |
| BIND            |      |    |  ✓ |  ✓   |  ✓   |
| BACPI           |      |    |  ✓ |  ✓   |  ✓   |

**Important**: every command below must be executed from the respective baseline's root directory
(e.g. `baselines/KcatNet/`), because the scripts are referenced by relative path as
`emulator_bench/<script>.py`.

---

## KcatNet (kcat, Km)

**Directory**: `baselines/KcatNet/`  
**Default hyperparameters**: `emulator_bench/kcatnet_repo_defaults_hparams.json`  
**Protein features**: ProtT5-XL-UniRef50 sequence embeddings  
**Ligand features**: Morgan fingerprint (radius 2, 1024 bits)

### Step 1 — Build the shared embedding cache (once per value type)

The cache step encodes every unique protein sequence and substrate SMILES that appears across all
split files. It writes reusable binary caches so training jobs do not re-encode on every run.

```bash
cd baselines/KcatNet

# kcat
python emulator_bench/cache_embeddings.py \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/kcat \
  --embeddings_dir /tmp/kcatnet_kcat_embeddings \
  --device cuda:0

# Km
python emulator_bench/cache_embeddings.py \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/km \
  --embeddings_dir /tmp/kcatnet_km_embeddings \
  --device cuda:0
```

Key options (defaults match paper):

| Flag | Default | Description |
|---|---|---|
| `--prot_t5_model` | `Rostlab/prot_t5_xl_uniref50` | HuggingFace model ID |
| `--protein_dtype` | `float16` | Cache precision (saves ~2× disk space) |
| `--overwrite` | off | Pass to force-rebuild existing cache entries |

### Step 2 — Retrain across all split groups

The launcher discovers every split job (all thresholds under thresholded split groups), assigns
them round-robin to the listed GPUs, and runs `train → predict → aggregate` for each job.

```bash
# kcat
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 2 3 \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/kcat \
  --embeddings_dir /tmp/kcatnet_kcat_embeddings \
  --hparams_json emulator_bench/kcatnet_repo_defaults_hparams.json \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                 enzyme_sequence_splits enzyme_structure_splits \
                 substrate_splits uniprot_time_splits conformer_cosine_splits \
  --seeds 666 \
  --epochs 80 \
  --num_workers 4 \
  --pin_memory \
  --persistent_workers

# Km (identical flags, different data root and embeddings dir)
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 2 3 \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/km \
  --embeddings_dir /tmp/kcatnet_km_embeddings \
  --hparams_json emulator_bench/kcatnet_repo_defaults_hparams.json \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                 enzyme_sequence_splits enzyme_structure_splits \
                 substrate_splits uniprot_time_splits conformer_cosine_splits \
  --seeds 666 \
  --epochs 80 \
  --num_workers 4 \
  --pin_memory \
  --persistent_workers
```

Results land under `<base_dir>/<split_group>/[threshold_X.XX/]kcatnet_results/seed_666/`.
A rolled-up `aggregate_tvt_metrics.csv` is written to the launcher output root.

---

## CataPro (kcat)

**Directory**: `baselines/CataPro/`  
**Default hyperparameters**: loaded automatically from `emulator_bench/default_hparams.json`  
**Protein features**: ProtT5-XL-UniRef50 + ESM-1b embeddings (built inline, no separate cache step)  
**Ligand features**: MolT5 SMILES embeddings + MACCS keys (built inline)

CataPro builds all molecular and protein features on the fly during the first pass over each split.
No separate caching step is needed; features are cached internally in `emulator_bench/.cache_embeddings/`
on the first run and reused on subsequent runs.

### Retrain across all split groups

```bash
cd baselines/CataPro

python emulator_bench/launch_parallel_bench.py \
  --value_root $DATASET_ROOT/kinetic_params_dataset/kcat \
  --gpus 0 1 2 3 \
  --runs_per_gpu 2 \
  --seeds 666
```

`default_hparams.json` is loaded automatically. Passing `--hparams_json` overrides individual keys.
Results land under `<value_root>/<split_group>/[threshold_X.XX/]catapro_results/seed_666/`.

---

## EITLEM-Kinetics (kcat, Km)

**Directory**: `baselines/EITLEM-Kinetics/`  
**Default hyperparameters**: `emulator_bench/default_hparams_kcat_original.json` (kcat),
`emulator_bench/default_hparams_km_original.json` (Km)  
**Protein features**: ESM-1b sequence embeddings  
**Ligand features**: MACCS keys fingerprint

### Step 1 — Build the shared embedding cache (once per value type)

```bash
cd baselines/EITLEM-Kinetics

# kcat
python emulator_bench/cache_embeddings.py \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/kcat \
  --embeddings_dir /tmp/eitlem_kcat_embeddings \
  --device cuda:0 \
  --mol_type MACCSKeys

# Km
python emulator_bench/cache_embeddings.py \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/km \
  --embeddings_dir /tmp/eitlem_km_embeddings \
  --device cuda:0 \
  --mol_type MACCSKeys
```

### Step 2 — Retrain across all split groups

```bash
# kcat — published hyperparameters from original paper
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 \
  --trials_per_gpu 2 \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/kcat \
  --embeddings_dir /tmp/eitlem_kcat_embeddings \
  --predictor_type kcat \
  --mol_type MACCSKeys \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                 enzyme_sequence_splits enzyme_structure_splits \
                 substrate_splits uniprot_time_splits conformer_cosine_splits \
  --seeds 666 \
  --epochs 100 \
  --patience 0 \
  --hparams_json emulator_bench/default_hparams_kcat_original.json \
  --num_workers 4 \
  --persistent_workers \
  --pin_memory

# Km — same structure, Km predictor type and Km hparams
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 \
  --trials_per_gpu 2 \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/km \
  --embeddings_dir /tmp/eitlem_km_embeddings \
  --predictor_type km \
  --mol_type MACCSKeys \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                 enzyme_sequence_splits enzyme_structure_splits \
                 substrate_splits uniprot_time_splits conformer_cosine_splits \
  --seeds 666 \
  --epochs 100 \
  --patience 0 \
  --hparams_json emulator_bench/default_hparams_km_original.json \
  --num_workers 4 \
  --persistent_workers \
  --pin_memory
```

`--predictor_type` selects the kcat or Km regression head. The embeddings cache is shared across
both heads (only protein sequences and SMILES are encoded; the same cache can serve both tasks if
the sequence sets overlap, but separate cache dirs are used here for clarity).

---

## DEKP (Km)

**Directory**: `baselines/DEKP/`  
**Default hyperparameters**: `emulator_bench/configs/fine_tune_defaults.json`  
**Protein features**: ProtT5 pooled embeddings (`t5`)  
**Ligand features**: SMILES Transformer (`trfm`); the pretrained model weights
(`emulator_bench/trfm_12_23000.pkl`) are bundled with the baseline — no separate download required  
**Protein structure**: not required. The `trfm,t5` feature set is sequence-only;
PDB structure directories are optional and can be omitted.

### Step 1 — Build the shared embedding cache

```bash
cd baselines/DEKP

python emulator_bench/cache_embeddings.py \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/km \
  --cache_dir /tmp/dekp_km_embeddings \
  --feature_list trfm,t5 \
  --device cuda:0
```

### Step 2 — Retrain across all split groups

```bash
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 \
  --jobs_per_gpu 2 \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/km \
  --cache_dir /tmp/dekp_km_embeddings \
  --hparams_json emulator_bench/configs/fine_tune_defaults.json \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                 enzyme_sequence_splits enzyme_structure_splits \
                 substrate_splits uniprot_time_splits conformer_cosine_splits \
  --seeds 3407 \
  --num_workers 8 \
  --pin_memory \
  --persistent_workers
```

---

## eMOSAIC (Ki)

**Directory**: `baselines/eMOSAIC/`  
**Default hyperparameters**: `emulator_bench/default_hparams_original.json`  
**Protein features**: ESMFold structure embeddings (runs ESMFold once per unique sequence; this is the
most GPU-intensive cache step across all baselines — budget ~1–4 hours depending on dataset size
and sequence length distribution)  
**Ligand features**: RDKit Morgan fingerprint

### Step 1 — Build the shared embedding cache

Sequences longer than `--max_seq_len` are truncated, not skipped.

```bash
cd baselines/eMOSAIC

python -u emulator_bench/cache_embeddings.py \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/ki \
  --embeddings_dir /tmp/emosaic_ki_embeddings \
  --device cuda:0 \
  --max_seq_len 700 \
  --chunk_size 64 \
  --protein_dtype float16
```

To parallelise the ESMFold passes across multiple GPUs, replace `--device cuda:0` with
`--gpus 0 1` (the launcher shards the unique sequence list across workers).

### Step 2 — Retrain across all split groups

```bash
python -u emulator_bench/launch_parallel_retrain_from_optuna.py \
  --gpus 0 1 \
  --trials_per_gpu 2 \
  --base_dir $DATASET_ROOT/kinetic_params_dataset/ki \
  --embeddings_dir /tmp/emosaic_ki_embeddings \
  --hparams_json emulator_bench/default_hparams_original.json \
  --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                 enzyme_sequence_splits enzyme_structure_splits \
                 substrate_splits uniprot_time_splits conformer_cosine_splits \
  --seeds 666 \
  --epochs 50 \
  --pin_memory \
  --persistent_workers \
  --preload_proteins
```

`--preload_proteins` loads all protein embeddings for a split into CPU RAM before training,
which eliminates repeated disk I/O during the epoch loop.

---

## BIND (Ki, EC50, IC50)

**Directory**: `baselines/BIND/`  
**Default hyperparameters**: `emulator_bench/default_hparams_bind_original.json`  
**Protein features**: ESM-2 (650M) hidden states at layers 0, 10, 20, 30  
**Ligand features**: BIND's native atom/bond graph featurization

BIND uses a unified data root where each supported value type is a subdirectory. Because Ki lives
under `kinetic_params_dataset/` and EC50/IC50 live under `binding_affinity_dataset/` in the
dataverse layout, you must assemble this structure before running.

### Step 0 — Assemble the unified data root

```bash
export BIND_DIR=/tmp/BIND_data
mkdir -p $BIND_DIR

ln -s $DATASET_ROOT/kinetic_params_dataset/ki    $BIND_DIR/ki
ln -s $DATASET_ROOT/binding_affinity_dataset/ec50 $BIND_DIR/ec50
ln -s $DATASET_ROOT/binding_affinity_dataset/ic50 $BIND_DIR/ic50

export BIND_EMB=/tmp/bind_embeddings
```

### Step 1 — Build the shared embedding cache

The cache is shared across all value types (protein sequences overlap across Ki/EC50/IC50).

```bash
cd baselines/BIND

python emulator_bench/launch_parallel_cache_embeddings.py \
  --base_dir $BIND_DIR \
  --embeddings_dir $BIND_EMB \
  --value_types ki ec50 ic50 \
  --protein_length_cap 2048 \
  --long_sequence_strategy direct \
  --gpus 0 \
  --max_residues 12288 \
  --max_batch 8 \
  --writer_threads 16 \
  --max_pending_writes 256
```

`--protein_length_cap 2048` truncates sequences above this length before encoding.
Use `--gpus 0 1 2 3` to shard the unique sequence list across multiple GPUs.

### Step 2 — Retrain across all split groups

```bash
python emulator_bench/launch_parallel_retrain_from_optuna.py \
  --base_dir $BIND_DIR \
  --embeddings_dir $BIND_EMB \
  --value_types ki ec50 ic50 \
  --hparams_json emulator_bench/default_hparams_bind_original.json \
  --protein_length_cap 2048 \
  --long_sequence_strategy direct \
  --gpus 0 1 2 3 \
  --trials_per_gpu 1 \
  --num_workers 8 \
  --pin_memory \
  --persistent_workers
```

The launcher iterates over all three value types and all split groups automatically.
Results land under `$BIND_DIR/<value_type>/<split_group>/[threshold_X.XX/]bind_results/`.

---

## BACPI (Ki, EC50, IC50)

**Directory**: `baselines/BACPI/`  
**Default hyperparameters**: `emulator_bench/default_hparams_original.json`  
**Protein features**: amino-acid 3-gram tokenization (no large pretrained model required)  
**Ligand features**: RDKit atom-environment graph + Morgan fingerprint (radius 2, 1024 bits)

BACPI computes all features inline during training. No separate cache step is required.

### Step 0 — Assemble the unified data root

```bash
export BACPI_DIR=/tmp/BACPI_data
mkdir -p $BACPI_DIR

ln -s $DATASET_ROOT/kinetic_params_dataset/ki    $BACPI_DIR/ki
ln -s $DATASET_ROOT/binding_affinity_dataset/ec50 $BACPI_DIR/ec50
ln -s $DATASET_ROOT/binding_affinity_dataset/ic50 $BACPI_DIR/ic50
```

### Retrain across all split groups

```bash
cd baselines/BACPI

for VALUE_TYPE in ki ec50 ic50; do
  python emulator_bench/launch_parallel_retrain_from_optuna.py \
    --gpus 0 1 2 3 \
    --trials_per_gpu 1 \
    --base_dir $BACPI_DIR \
    --value_type $VALUE_TYPE \
    --split_groups random_splits_grouped_sequence random_splits_grouped_smiles \
                   enzyme_sequence_splits enzyme_structure_splits \
                   substrate_splits uniprot_time_splits conformer_cosine_splits \
    --seeds 666 \
    --hparams_json emulator_bench/default_hparams_original.json \
    --skip_smiles_validity_check \
    --num_workers 4
done
```

Results land under `$BACPI_DIR/<value_type>/<split_group>/[threshold_X.XX/]bacpi_results/`.

---

## Output Structure

Every baseline writes the same standardised output layout inside each split job directory:

```
<split_job_dir>/
  train/
    bestmodel.pth           ← checkpoint with best validation loss
    checkpoint_last.pt
  predictions/
    train_predictions.csv
    val_predictions.csv
    test_predictions.csv
    *_predictions_metrics.csv
  metrics/
    tvt_metrics_long.csv    ← per-split metrics (RMSE, MAE, R², PCC, SCC)
    tvt_metrics_wide.csv
  logs/
    train.log
    predict_*.log
```

The parallel launcher additionally writes at its output root:

```
planned_runs.csv            ← all discovered jobs
runs_status.csv             ← per-job success/failure
all_tvt_metrics.csv         ← concatenation of every job's metrics
aggregate_tvt_metrics.csv   ← mean ± std across seeds per split group
```

These aggregate files are the primary artefacts used to populate the paper's results tables.

---

## Common Troubleshooting

**Out-of-memory during cache step**: reduce `--max_batch` (KcatNet/BIND) or `--chunk_size`
(eMOSAIC), or switch to `--protein_dtype float32` to diagnose precision-related issues.

**"Split group not found"**: verify that the value-type folder passed to `--base_dir` (or
assembled via symlinks for BIND/BACPI) contains the split-group subdirectories listed in the
dataset layout table above.

**Slow disk I/O during training**: pass `--pin_memory --persistent_workers --preload_proteins`
(where supported) to reduce DataLoader stall time.

**Single GPU**: replace `--gpus 0 1 2 3` with `--gpus 0`. Each split job runs sequentially on
that GPU; results are identical, only wall-clock time increases.
