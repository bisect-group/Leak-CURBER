# Reproducing Baseline Results

This guide covers every baseline in this checkout that has a Leak-CURBER
`emulator_bench/` adapter. It gives complete commands for the submitted
subtasks, all applicable benchmark split axes, and seeds 666, 777, and 888.

The commands keep the downloaded Dataverse files read-only. Generated data,
feature caches, checkpoints, predictions, and summaries are written below
`RUN_ROOT` or `CACHE_ROOT`. Some upstream classification adapters create small
compatibility symlinks inside their baseline checkout. These links point to the
generated run or cache data; they do not copy the Dataverse data into the
repository.

## Contents

- [Coverage](#coverage)
- [One-time setup](#one-time-setup)
- [Regression baselines](#regression-baselines)
- [Enzyme-classification baselines](#enzyme-classification-baselines)
- [Enzyme-retrieval baselines](#enzyme-retrieval-baselines)
- [Result checks](#result-checks)
- [Troubleshooting](#troubleshooting)

## Coverage

### Task-first baseline matrix

“Reported” identifies the Leak-CURBER results reproduced by this guide.
“Adapter support” lists additional native targets accepted by the checked-in
adapter. Additional targets are not part of the commands below unless they were
reported in the submission.

| Task family | Baseline | Reported subtasks | Adapter support |
|---|---|---|---|
| Kinetic regression | [KcatNet](baselines/KcatNet/emulator_bench/README.md) | kcat | kcat, Km |
| Kinetic regression | [CataPro](baselines/CataPro/emulator_bench/README.md) | kcat, Km | kcat, Km, Ki |
| Kinetic regression | [EITLEM-Kinetics](baselines/EITLEM-Kinetics/emulator_bench/README.md) | kcat, Km | kcat, Km, kcat/Km |
| Kinetic regression | [DEKP](baselines/DEKP/emulator_bench/README.md) | Km | Km |
| Kinetic regression | [eMOSAIC](baselines/eMOSAIC/emulator_bench/README.md) | Ki | Ki |
| Kinetic and affinity regression | [BIND](baselines/BIND/emulator_bench/README.md) | Ki, Kd, EC50 | Ki, Kd, EC50, IC50 |
| Kinetic and affinity regression | [BACPI](baselines/BACPI/emulator_bench/README.md) | Ki, Kd, EC50, IC50 | Ki, Kd, EC50, IC50 |
| Affinity regression | [StructureFree-DTA](baselines/StructureFree-DTA/emulator_bench/README.md) | Kd | One regression target per run |
| Enzyme classification | [CLEAN](baselines/CLEAN/emulator_bench/README.md) | EC classification | EC classification |
| Enzyme classification | [GraphEC](baselines/GraphEC/emulator_bench/README.md) | EC classification | EC classification |
| Enzyme classification | [HITEC](baselines/HITEC/emulator_bench/README.md) | EC classification | EC classification |
| Enzyme retrieval | [Clipzyme](baselines/Clipzyme/emulator_bench/README.md) | Reaction-to-EC retrieval | Reaction-to-enzyme and reaction-to-EC retrieval |
| Enzyme retrieval | [Horizon](baselines/Horizon/emulator_bench/README.md) | Reaction-to-EC retrieval | Reaction-to-enzyme and reaction-to-EC retrieval |

`baselines/CARE/` is an upstream reference checkout. It does not contain a
Leak-CURBER `emulator_bench/` adapter and is therefore not included below.

The submitted reaction-outcome baselines were Root-aligned SMILES, MEGAN, and
Chemformer. Their adapters are not present in this checkout. This guide does not
claim that those results can be reproduced from this release.

### Evaluation axes

The regression sweeps cover seven axes:

| Display name | Released directory |
|---|---|
| GR-seq | `random_splits_grouped_sequence` |
| GR-SMILES | `random_splits_grouped_smiles` |
| P-seq | `enzyme_sequence_splits/threshold_*` |
| P-struct | `enzyme_structure_splits/threshold_*` |
| Mol-2D | `substrate_splits/threshold_*` |
| Mol-3D | `conformer_cosine_splits/threshold_*` |
| Time | `uniprot_time_splits` |

Enzyme classification uses GR-seq, P-seq, P-struct, and Time. Enzyme retrieval
uses those four axes plus `reaction_drfp_tanimoto_splits`, shown as DRFP in the
paper. Each adapter discovers the released threshold directory under a split
family. Do not substitute a different threshold.

## One-time setup

### 1. Define paths

Set these paths in every new shell:

```bash
export PROJECT_ROOT=/path/to/cloned/Leak-CURBER
export DATAVERSE_ROOT=/path/to/downloaded/Leak-CURBER_dataverse

# Use the full benchmark for paper-result reproduction.
export DATASET_ROOT="$DATAVERSE_ROOT/01_core_benchmark_datasets"

# Use this instead for a pipeline check on the bundled sample.
# export DATASET_ROOT="$PROJECT_ROOT/00-sample__benchmark_datasets"

export RUN_ROOT=/path/to/writable/leakcurber_runs
export CACHE_ROOT=/path/to/writable/leakcurber_caches
export GPU=0

mkdir -p "$RUN_ROOT" "$CACHE_ROOT"
```

The Dataverse layout is:

```text
DATAVERSE_ROOT/
├── 00-sample__benchmark_datasets/
└── 01_core_benchmark_datasets/
    ├── kinetic_params_dataset/
    │   ├── kcat/
    │   ├── km/
    │   └── ki/
    ├── binding_affinity_dataset/
    │   ├── kd/
    │   ├── ec50/
    │   └── ic50/
    ├── enzyme_classification_dataset/
    ├── enzyme_retrieval_dataset/
    └── reaction_outcome_dataset/
```

The commands below require only the first four task-family directories. The
reaction-outcome directory is shown for completeness.

### 2. Create environments

The regression adapters were integrated and validated in the release project
environment:

```bash
cd "$PROJECT_ROOT"
conda env create --name leakcurber --file environment.yml
```

If the environment already exists, use:

```bash
conda env update --name leakcurber --file environment.yml --prune
```

CataPro also has a smaller, independently verified environment:

```bash
conda create --name leakcurber-catapro python=3.12 -y
conda run -n leakcurber-catapro pip install \
  torch transformers==4.57.6 sentencepiece protobuf \
  numpy pandas pyarrow scikit-learn rdkit tqdm
```

The classification and retrieval models use incompatible upstream dependency
stacks. Create their environments from the checked-in upstream specifications:

| Baseline | Environment command or source |
|---|---|
| CLEAN | Python 3.10, PyTorch 1.11, and [`app/requirements.txt`](baselines/CLEAN/app/requirements.txt) |
| GraphEC | Python 3.8 and the versions in its [System requirement](baselines/GraphEC/README.md#system-requirement), including PyTorch Geometric, ESMFold, and ProtTrans |
| HITEC | Python 3.9 and [`requirements.txt`](baselines/HITEC/requirements.txt) |
| Clipzyme | `conda env create --file baselines/Clipzyme/environment.yml`, then `pip install -e baselines/Clipzyme` |
| Horizon | Python 3.10, then `pip install -e 'baselines/Horizon[emulator]'` |

The commands in the relevant sections assume environment names `clean`,
`graphec`, `hitec`, `clipzyme`, and `horizon`. Change `--name` or `conda run -n`
consistently if you use different names.

### 3. Stage regression data without copying it

The regression runners write results beside their split trees. Define this
helper once. It creates a writable directory tree and links every released file
into it. It refuses to replace an existing conflicting path.

```bash
stage_dataset() {
  python - "$1" "$2" <<'PY'
from pathlib import Path
import os
import sys

source = Path(sys.argv[1]).expanduser().resolve()
target = Path(sys.argv[2]).expanduser().resolve()
if not source.is_dir():
    raise SystemExit(f"Dataset directory does not exist: {source}")

target.mkdir(parents=True, exist_ok=True)
for item in sorted(source.rglob("*")):
    relative = item.relative_to(source)
    output = target / relative
    if item.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        continue
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() and output.resolve() == item.resolve():
        continue
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to replace staged path: {output}")
    output.symlink_to(item.resolve())

print(f"Staged read-only inputs: {source} -> {target}")
PY
}
```

Define the shared sweep arguments:

```bash
REGRESSION_SPLITS=(
  random_splits_grouped_sequence
  random_splits_grouped_smiles
  enzyme_sequence_splits
  enzyme_structure_splits
  substrate_splits
  conformer_cosine_splits
  uniprot_time_splits
)

CLASSIFICATION_SPLITS=(
  random_splits_grouped_sequence
  enzyme_sequence_splits/threshold_0.4
  enzyme_structure_splits/threshold_0.9
  uniprot_time_splits
)

RETRIEVAL_SPLITS=(
  random_splits_grouped_sequence
  reaction_drfp_tanimoto_splits/threshold_0.15
  enzyme_sequence_splits/threshold_0.4
  enzyme_structure_splits/threshold_0.9
  uniprot_time_splits
)

SEEDS=(666 777 888)
```

## Regression baselines

Run each section from a shell in which the path variables, arrays, and
`stage_dataset` function above are defined.

### KcatNet: kcat

Setup references: [adapter README](baselines/KcatNet/emulator_bench/README.md)
and [complete command reference](baselines/KcatNet/commands.txt).

```bash
KCATNET_DATA="$RUN_ROOT/data/KcatNet/kcat"
KCATNET_CACHE="$CACHE_ROOT/KcatNet/kcat"
stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/kcat" \
  "$KCATNET_DATA"

cd "$PROJECT_ROOT/baselines/KcatNet"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/cache_embeddings.py \
  --base_dir "$KCATNET_DATA" \
  --embeddings_dir "$KCATNET_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --device cuda:0

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/run_split_benchmarks.py \
  --base_dir "$KCATNET_DATA" \
  --embeddings_dir "$KCATNET_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --seeds "${SEEDS[@]}" \
  --hparams_json emulator_bench/kcatnet_repo_defaults_hparams.json \
  --epochs 80 \
  --skip_cache \
  --device cuda:0 \
  --num_workers 4 \
  --pin_memory \
  --persistent_workers
```

Primary outputs:

- `$KCATNET_DATA/kcatnet_summary_runs.csv`
- `$KCATNET_DATA/kcatnet_summary_by_split_group.csv`
- `<split>/kcatnet_results/seed_<seed>/final_results_test.csv`

### CataPro: kcat and Km

Setup references: [adapter README](baselines/CataPro/emulator_bench/README.md)
and [command reference](baselines/CataPro/emulator_bench/commands.txt).

`launch_parallel_bench.py` performs cache generation, training, evaluation, and
aggregation. The first run downloads
`Rostlab/prot_t5_xl_uniref50` and
`laituan245/molt5-base-smiles2caption`.

```bash
CATAPRO_DATA="$RUN_ROOT/data/CataPro"
CATAPRO_CACHE="$CACHE_ROOT/CataPro"
stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/kcat" \
  "$CATAPRO_DATA/kcat"
stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/km" \
  "$CATAPRO_DATA/km"

cd "$PROJECT_ROOT/baselines/CataPro"

conda run --no-capture-output -n leakcurber-catapro \
  python emulator_bench/launch_parallel_bench.py \
  --value_root "$CATAPRO_DATA/kcat" "$CATAPRO_DATA/km" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --gpus "$GPU" \
  --runs_per_gpu 1 \
  --feature_gpu "$GPU" \
  --seeds "${SEEDS[@]}" \
  --cache_dir "$CATAPRO_CACHE" \
  --prot_batch_size 8 \
  --hparams_json emulator_bench/default_hparams.json
```

Primary outputs for each value type:

- `$CATAPRO_DATA/<value_type>/catapro_summary_runs.csv`
- `$CATAPRO_DATA/<value_type>/catapro_summary_by_split_group.csv`
- `<split>/catapro_results/seed_<seed>/final_results_test.csv`

The root [README](README.md) contains a separately verified 900/50/50-row
CataPro P-seq workflow.

### EITLEM-Kinetics: kcat and Km

Setup references:
[adapter README](baselines/EITLEM-Kinetics/emulator_bench/README.md) and
[command reference](baselines/EITLEM-Kinetics/commands.txt).

Run the same pipeline once for each reported target:

```bash
cd "$PROJECT_ROOT/baselines/EITLEM-Kinetics"

for VALUE_TYPE in kcat km; do
  EITLEM_DATA="$RUN_ROOT/data/EITLEM-Kinetics/$VALUE_TYPE"
  EITLEM_CACHE="$CACHE_ROOT/EITLEM-Kinetics/$VALUE_TYPE"

  stage_dataset \
    "$DATASET_ROOT/kinetic_params_dataset/$VALUE_TYPE" \
    "$EITLEM_DATA"

  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
    python emulator_bench/cache_embeddings.py \
    --base_dir "$EITLEM_DATA" \
    --embeddings_dir "$EITLEM_CACHE" \
    --split_groups "${REGRESSION_SPLITS[@]}" \
    --mol_type MACCSKeys \
    --device cuda:0

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/run_split_benchmarks.py \
    --base_dir "$EITLEM_DATA" \
    --embeddings_dir "$EITLEM_CACHE" \
    --split_groups "${REGRESSION_SPLITS[@]}" \
    --predictor_type "$VALUE_TYPE" \
    --mol_type MACCSKeys \
    --seeds "${SEEDS[@]}" \
    --hparams_json "emulator_bench/default_hparams_${VALUE_TYPE}_original.json" \
    --skip_cache \
    --device cuda:0
done
```

Primary outputs:

- `<task-root>/eitlem_summary_runs.csv`
- `<task-root>/eitlem_summary_by_split_group.csv`
- `<split>/eitlem_results/seed_<seed>/final_results_test.csv`

### DEKP: Km

Setup references: [adapter README](baselines/DEKP/emulator_bench/README.md)
and [command reference](baselines/DEKP/emulator_bench/commands.txt).

The submitted model uses DEKP’s native TRFM ligand representation and ProtT5
protein representation. The checked-in TRFM weights and vocabulary are used.

```bash
DEKP_DATA="$RUN_ROOT/data/DEKP/km"
DEKP_CACHE="$CACHE_ROOT/DEKP/km"
stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/km" \
  "$DEKP_DATA"

cd "$PROJECT_ROOT/baselines/DEKP"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/cache_embeddings.py \
  --base_dir "$DEKP_DATA" \
  --dataset_df_path "$DEKP_DATA/km_kinetic_params_3d.parquet" \
  --cache_dir "$DEKP_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --feature_list trfm,t5 \
  --device cuda:0

  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
    python emulator_bench/run_split_benchmarks.py \
  --base_dir "$DEKP_DATA" \
  --cache_dir "$DEKP_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --feature_list trfm,t5 \
  --seeds "${SEEDS[@]}" \
  --epochs 80 \
  --skip_cache \
  --device cuda:0 \
  --num_workers 8 \
  --pin_memory \
  --persistent_workers
```

Primary outputs are the per-seed `final_results_test.csv` files under
`dekp_results/` and the summary CSVs written at the task root.

### eMOSAIC: Ki

Setup references: [adapter README](baselines/eMOSAIC/emulator_bench/README.md),
[command reference](baselines/eMOSAIC/emulator_bench/commands.txt), and the
upstream [environment recipe](baselines/eMOSAIC/environment/postInstall).

```bash
EMOSAIC_DATA="$RUN_ROOT/data/eMOSAIC/ki"
EMOSAIC_CACHE="$CACHE_ROOT/eMOSAIC/ki"
stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/ki" \
  "$EMOSAIC_DATA"

cd "$PROJECT_ROOT/baselines/eMOSAIC"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/cache_embeddings.py \
  --base_dir "$EMOSAIC_DATA" \
  --embeddings_dir "$EMOSAIC_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --device cuda:0 \
  --max_seq_len 700 \
  --protein_dtype float16

conda run --no-capture-output -n leakcurber \
  python emulator_bench/run_split_benchmarks.py \
  --base_dir "$EMOSAIC_DATA" \
  --embeddings_dir "$EMOSAIC_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --seeds "${SEEDS[@]}" \
  --hparams_json emulator_bench/default_hparams_original.json \
  --epochs 50 \
  --skip_cache \
  --device cuda:0 \
  --pin_memory \
  --persistent_workers \
  --preload_proteins
```

Primary outputs are the per-seed test metrics under `emosaic_results/` and the
summary CSVs at the task root.

### BIND: Ki, Kd, and EC50

Setup references: [adapter README](baselines/BIND/emulator_bench/README.md) and
[command reference](baselines/BIND/commands.txt).

The wrapper trains one native BIND regression head for each value type. The
following command intentionally excludes IC50 because the submitted BIND
evaluation reported Ki, Kd, and EC50 only.

```bash
BIND_DATA="$RUN_ROOT/data/BIND"
BIND_CACHE="$CACHE_ROOT/BIND"

stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/ki" \
  "$BIND_DATA/ki"
stage_dataset \
  "$DATASET_ROOT/binding_affinity_dataset/kd" \
  "$BIND_DATA/kd"
stage_dataset \
  "$DATASET_ROOT/binding_affinity_dataset/ec50" \
  "$BIND_DATA/ec50"

cd "$PROJECT_ROOT/baselines/BIND"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/cache_embeddings.py \
  --base_dir "$BIND_DATA" \
  --embeddings_dir "$BIND_CACHE" \
  --value_types ki kd ec50 \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --protein_length_cap 2048 \
  --long_sequence_strategy direct \
  --device cuda:0 \
  --max_batch 4

conda run --no-capture-output -n leakcurber \
  python emulator_bench/run_split_benchmarks.py \
  --base_dir "$BIND_DATA" \
  --embeddings_dir "$BIND_CACHE" \
  --value_types ki kd ec50 \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --seeds "${SEEDS[@]}" \
  --hparams_json emulator_bench/default_hparams_bind_original.json \
  --protein_length_cap 2048 \
  --long_sequence_strategy direct \
  --skip_cache \
  --gpus "$GPU" \
  --runs_per_gpu 1 \
  --num_workers 6 \
  --prefetch_factor 2 \
  --sharing_strategy file_system \
  --pin_memory \
  --persistent_workers
```

`run_split_benchmarks.py` accepts `--hparams_json`.
`launch_parallel_retrain_from_optuna.py` does not; that launcher instead accepts
`--hparams_json_template`.

Primary outputs:

- `$BIND_DATA/bench_summaries/all_seed_runs.csv`
- `$BIND_DATA/bench_summaries/summary_by_split.csv`
- `<value_type>/<split>/bind_results/seed_<seed>/final_results_test.csv`

### BACPI: Ki, Kd, EC50, and IC50

Setup references: [adapter README](baselines/BACPI/emulator_bench/README.md) and
[command reference](baselines/BACPI/commands.txt).

BACPI normally removes invalid SMILES from split files during cache generation.
The staged tree is writable, but `--no_clean_splits_in_place` keeps its linked
input parquets unchanged. Invalid rows are still excluded by the model pipeline.

```bash
BACPI_DATA="$RUN_ROOT/data/BACPI"
BACPI_CACHE="$CACHE_ROOT/BACPI"

stage_dataset \
  "$DATASET_ROOT/kinetic_params_dataset/ki" \
  "$BACPI_DATA/ki"
stage_dataset \
  "$DATASET_ROOT/binding_affinity_dataset/kd" \
  "$BACPI_DATA/kd"
stage_dataset \
  "$DATASET_ROOT/binding_affinity_dataset/ec50" \
  "$BACPI_DATA/ec50"
stage_dataset \
  "$DATASET_ROOT/binding_affinity_dataset/ic50" \
  "$BACPI_DATA/ic50"

cd "$PROJECT_ROOT/baselines/BACPI"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python emulator_bench/cache_embeddings.py \
  --base_root "$BACPI_DATA" \
  --value_type ki kd ec50 ic50 \
  --embeddings_dir "$BACPI_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --no_clean_splits_in_place

for VALUE_TYPE in ki kd ec50 ic50; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
    python emulator_bench/run_split_benchmarks.py \
    --base_root "$BACPI_DATA" \
    --value_type "$VALUE_TYPE" \
    --embeddings_dir "$BACPI_CACHE" \
    --split_groups "${REGRESSION_SPLITS[@]}" \
    --seeds "${SEEDS[@]}" \
    --hparams_json emulator_bench/default_hparams_original.json \
    --num_epochs 20 \
    --skip_cache \
    --no_clean_splits_in_place \
    --device cuda:0 \
    --num_workers 4 \
    --pin_memory \
    --persistent_workers \
    --preload_features
done

conda run --no-capture-output -n leakcurber \
  python emulator_bench/aggregate_retrain_metrics.py \
  --base_root "$BACPI_DATA" \
  --value_types ki kd ec50 ic50 \
  --runs_dir bacpi_results
```

Primary outputs:

- `$BACPI_DATA/bacpi_results_summary.csv`
- `<value_type>/bacpi_results/summary.csv`
- `<value_type>/<split>/bacpi_results/seed_<seed>/final_results_test.csv`

### StructureFree-DTA: Kd

Setup references:
[adapter README](baselines/StructureFree-DTA/emulator_bench/README.md) and
[command reference](baselines/StructureFree-DTA/emulator_bench/commands.txt).

```bash
STRUCTUREFREE_DATA="$RUN_ROOT/data/StructureFree-DTA/kd"
STRUCTUREFREE_CACHE="$CACHE_ROOT/StructureFree-DTA/kd"
stage_dataset \
  "$DATASET_ROOT/binding_affinity_dataset/kd" \
  "$STRUCTUREFREE_DATA"

cd "$PROJECT_ROOT/baselines/StructureFree-DTA"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python -m emulator_bench.cache_embeddings \
  --base_dir "$STRUCTUREFREE_DATA" \
  --embeddings_dir "$STRUCTUREFREE_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --device cuda:0

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n leakcurber \
  python -m emulator_bench.run_split_benchmarks \
  --base_dir "$STRUCTUREFREE_DATA" \
  --embeddings_dir "$STRUCTUREFREE_CACHE" \
  --split_groups "${REGRESSION_SPLITS[@]}" \
  --seeds "${SEEDS[@]}" \
  --hparams_json emulator_bench/default_hparams.json \
  --epochs 150 \
  --skip_cache \
  --device cuda:0 \
  --preload_embeddings \
  --pin_memory \
  --persistent_workers

conda run --no-capture-output -n leakcurber \
  python -m emulator_bench.aggregate_results \
  --base_dir "$STRUCTUREFREE_DATA" \
  --split test \
  --group_by split_group \
  --save "$STRUCTUREFREE_DATA/structurefree_summary.csv"
```

Primary outputs:

- `$STRUCTUREFREE_DATA/structurefree_summary.csv`
- `<split>/structurefree_results/seed_<seed>/final_results_test.csv`

## Enzyme-classification baselines

These adapters accept the released classification directory directly and keep
generated state in explicit run and cache roots. The commands run each stage
sequentially so that task-spooler is not required.

### CLEAN

Setup reference: [adapter README](baselines/CLEAN/emulator_bench/README.md).

```bash
CLEAN_RUNS="$RUN_ROOT/CLEAN"
CLEAN_CACHE="$CACHE_ROOT/CLEAN"
CLASSIFICATION_DATA="$DATASET_ROOT/enzyme_classification_dataset"

cd "$PROJECT_ROOT/baselines/CLEAN"

for SPLIT_GROUP in "${CLASSIFICATION_SPLITS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n clean \
    python -m emulator_bench.cache_features \
    --dataset-root "$CLASSIFICATION_DATA" \
    --split-group "$SPLIT_GROUP" \
    --runs-root "$CLEAN_RUNS" \
    --cache-root "$CLEAN_CACHE"

  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n clean \
      python -m emulator_bench.train \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$CLEAN_RUNS" \
      --env-name clean \
      --epochs 7000 \
      --precision auto \
      --seed "$SEED"

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n clean \
      python -m emulator_bench.evaluate \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$CLEAN_RUNS" \
      --eval-split test \
      --device cuda \
      --seed "$SEED"
  done
done

conda run --no-capture-output -n clean \
  python -m emulator_bench.aggregate_results \
  --runs-root "$CLEAN_RUNS"
```

Primary outputs:

- `$CLEAN_RUNS/aggregated_seed_metrics_long.csv`
- `$CLEAN_RUNS/aggregated_seed_metrics_summary.csv`
- `<split>/seeds/<seed>/results/`

### GraphEC

Setup reference: [adapter README](baselines/GraphEC/emulator_bench/README.md).
Set `PROTTRANS_MODEL_PATH` if the ProtTrans checkpoint is not in the location
expected by the upstream repository.

```bash
GRAPHEC_RUNS="$RUN_ROOT/GraphEC"
GRAPHEC_CACHE="$CACHE_ROOT/GraphEC"
CLASSIFICATION_DATA="$DATASET_ROOT/enzyme_classification_dataset"

cd "$PROJECT_ROOT/baselines/GraphEC"

GRAPHEC_MODEL_ARGS=()
if [[ -n "${PROTTRANS_MODEL_PATH:-}" ]]; then
  GRAPHEC_MODEL_ARGS=(
    --prottrans-model-path "$PROTTRANS_MODEL_PATH"
  )
fi

for SPLIT_GROUP in "${CLASSIFICATION_SPLITS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n graphec \
    python -m emulator_bench.cache_features \
    --dataset-root "$CLASSIFICATION_DATA" \
    --split-group "$SPLIT_GROUP" \
    --runs-root "$GRAPHEC_RUNS" \
    --cache-root "$GRAPHEC_CACHE" \
    "${GRAPHEC_MODEL_ARGS[@]}"

  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n graphec \
      python -m emulator_bench.train \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$GRAPHEC_RUNS" \
      --cache-root "$GRAPHEC_CACHE" \
      --env-name graphec \
      --seed "$SEED" \
      --epochs 35 \
      --folds 5 \
      --batch-size 32 \
      --precision auto

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n graphec \
      python -m emulator_bench.evaluate \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$GRAPHEC_RUNS" \
      --cache-root "$GRAPHEC_CACHE" \
      --seed "$SEED" \
      --eval-split test \
      --batch-size 32 \
      --precision auto
  done
done

conda run --no-capture-output -n graphec \
  python -m emulator_bench.aggregate_results \
  --runs-root "$GRAPHEC_RUNS"
```

Primary outputs:

- `$GRAPHEC_RUNS/aggregated_seed_metrics_long.csv`
- `$GRAPHEC_RUNS/aggregated_seed_metrics_summary.csv`
- `<split>/seeds/<seed>/results/`

### HITEC

Setup reference: [adapter README](baselines/HITEC/emulator_bench/README.md).

```bash
HITEC_RUNS="$RUN_ROOT/HITEC"
HITEC_CACHE="$CACHE_ROOT/HITEC"
HITEC_VOCAB="$HITEC_CACHE/ec_vocab.json"
CLASSIFICATION_DATA="$DATASET_ROOT/enzyme_classification_dataset"

cd "$PROJECT_ROOT/baselines/HITEC"

for SPLIT_GROUP in "${CLASSIFICATION_SPLITS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n hitec \
    python -m emulator_bench.cache_features \
    --dataset-root "$CLASSIFICATION_DATA" \
    --split-group "$SPLIT_GROUP" \
    --runs-root "$HITEC_RUNS" \
    --cache-root "$HITEC_CACHE" \
    --vocab-path "$HITEC_VOCAB"

  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n hitec \
      python -m emulator_bench.train \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$HITEC_RUNS" \
      --env-name hitec \
      --epochs 80 \
      --batch-size 2 \
      --seed "$SEED" \
      --precision auto

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n hitec \
      python -m emulator_bench.evaluate \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$HITEC_RUNS" \
      --seed "$SEED" \
      --eval-split test \
      --device cuda \
      --rank-limit 50
  done
done

conda run --no-capture-output -n hitec \
  python -m emulator_bench.aggregate_results \
  --runs-root "$HITEC_RUNS"
```

Primary outputs:

- `$HITEC_RUNS/aggregate_metrics.csv`
- `$HITEC_RUNS/aggregate_summary.csv`
- `<split>/seeds/<seed>/results/`

## Enzyme-retrieval baselines

### Clipzyme

Setup reference: [adapter README](baselines/Clipzyme/emulator_bench/README.md).
The microbatch and gradient-accumulation values preserve the submitted effective
batch size while fitting a 22–24 GB GPU.

```bash
CLIPZYME_RUNS="$RUN_ROOT/Clipzyme"
CLIPZYME_CACHE="$CACHE_ROOT/Clipzyme"
RETRIEVAL_DATA="$DATASET_ROOT/enzyme_retrieval_dataset"

cd "$PROJECT_ROOT/baselines/Clipzyme"

for SPLIT_GROUP in "${RETRIEVAL_SPLITS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n clipzyme \
    python -m emulator_bench.cache_features \
    --dataset-root "$RETRIEVAL_DATA" \
    --split-group "$SPLIT_GROUP" \
    --runs-root "$CLIPZYME_RUNS" \
    --cache-root "$CLIPZYME_CACHE"

  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n clipzyme \
      python -m emulator_bench.train \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$CLIPZYME_RUNS" \
      --seed "$SEED" \
      --epochs 20 \
      --precision bf16 \
      --available-gpus 0 \
      --batch-size 1 \
      --accumulate-grad-batches 16

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n clipzyme \
      python -m emulator_bench.evaluate \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$CLIPZYME_RUNS" \
      --seed "$SEED" \
      --eval-split test \
      --available-gpus 0
  done
done

conda run --no-capture-output -n clipzyme \
  python - "$CLIPZYME_RUNS" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

root = Path(sys.argv[1]).resolve()
rows = []
for path in sorted(root.glob("*/seeds/*/results/test/care_task2_metrics.json")):
    metrics = pd.json_normalize(json.loads(path.read_text()), sep=".").iloc[0].to_dict()
    rows.append(
        {
            "split_group": path.parents[4].name,
            "seed": int(path.parents[2].name),
            **metrics,
        }
    )

if not rows:
    raise SystemExit(f"No Clipzyme test metrics found under {root}")

runs = pd.DataFrame(rows).sort_values(["split_group", "seed"])
runs.to_csv(root / "aggregated_seed_metrics_long.csv", index=False)
metric_columns = [column for column in runs if column not in {"split_group", "seed"}]
summary = runs.groupby("split_group")[metric_columns].agg(["mean", "std"])
summary.columns = ["_".join(column) for column in summary.columns]
summary.reset_index().to_csv(
    root / "aggregated_seed_metrics_summary.csv",
    index=False,
)
PY
```

Clipzyme writes native retrieval metrics and CARE-style EC-ranked outputs below
`$CLIPZYME_RUNS/<split>/seeds/<seed>/results/`. The final command writes
`aggregated_seed_metrics_long.csv` and
`aggregated_seed_metrics_summary.csv` because this adapter does not provide a
separate aggregate CLI.

### Horizon

Setup reference: [adapter README](baselines/Horizon/emulator_bench/README.md).

```bash
HORIZON_RUNS="$RUN_ROOT/Horizon"
HORIZON_CACHE="$CACHE_ROOT/Horizon"
RETRIEVAL_DATA="$DATASET_ROOT/enzyme_retrieval_dataset"

cd "$PROJECT_ROOT/baselines/Horizon"

for SPLIT_GROUP in "${RETRIEVAL_SPLITS[@]}"; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n horizon \
    python -m emulator_bench.cache_features \
    --dataset-root "$RETRIEVAL_DATA" \
    --split-group "$SPLIT_GROUP" \
    --runs-root "$HORIZON_RUNS" \
    --cache-root "$HORIZON_CACHE" \
    --embedding-source prott5 \
    --device cuda:0

  for SEED in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n horizon \
      python -m emulator_bench.train \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$HORIZON_RUNS" \
      --seed "$SEED" \
      --epochs 100 \
      --precision bf16

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n horizon \
      python -m emulator_bench.evaluate \
      --split-group "$SPLIT_GROUP" \
      --runs-root "$HORIZON_RUNS" \
      --seed "$SEED" \
      --eval-split test \
      --device cuda
  done
done

conda run --no-capture-output -n horizon \
  python -m emulator_bench.aggregate_results \
  --runs-root "$HORIZON_RUNS"
```

Primary outputs:

- `$HORIZON_RUNS/aggregated_seed_metrics_long.csv`
- `$HORIZON_RUNS/aggregated_seed_metrics_summary.csv`
- `<split>/seeds/<seed>/results/`

## Result checks

### Regression

Every completed regression run must contain a best checkpoint, test
predictions, and test metrics. File names vary slightly by adapter, but
`final_results_test.csv` is common:

```bash
find "$RUN_ROOT/data" -type f -name final_results_test.csv -print
find "$RUN_ROOT/data" -type f \
  \( -name bestmodel.pth -o -name 'best*.pt' -o -name 'best*.ckpt' \) \
  -print
```

The regression metric files report MSE, RMSE, MAE, R2, Pearson correlation, and
Spearman correlation. Confirm that each reported split has exactly three
completed seed rows in its aggregate summary.

### Classification and retrieval

Each seed directory must contain training metadata with a checkpoint path and
an evaluation results directory:

```bash
find "$RUN_ROOT" -type f -name train_seed.json -print
find "$RUN_ROOT" -type d -name results -print
```

Classification evaluation reports exact-match accuracy at EC Levels 4, 3, 2,
and 1. Retrieval evaluation reports the same EC-level accuracies from ranked
reaction-query predictions. Use the aggregate CSVs named in each section for
the mean and standard deviation across seeds.

### Restart behavior

The adapters reuse completed caches. Most runners also skip a completed seed
when its expected checkpoint and metric files are present. Do not pass
`--overwrite`, `--cache_overwrite`, or equivalent flags unless you intend to
replace generated artifacts.

## Troubleshooting

**No jobs discovered:** Check that `DATASET_ROOT` points directly to
`00-sample__benchmark_datasets` or `01_core_benchmark_datasets`. For regression,
also check that the staged task root contains the split directories rather than
another nested task-name directory.

**CUDA out of memory during feature generation:** Reduce the relevant encoder
batch argument. Common controls are CataPro `--prot_batch_size`, BIND
`--max_batch`, DEKP `--prot_t5_max_batch`, eMOSAIC `--chunk_size`, and
StructureFree-DTA `--protein_batch_size`.

**CUDA out of memory during training:** Reduce the model’s batch size. Preserve
the effective batch size with gradient accumulation when the adapter supports
it. Clipzyme’s documented `1 × 16` configuration is the tested example.

**Model download fails:** The first cache run can download pretrained weights
from Hugging Face or an upstream model host. Confirm internet access and set the
standard Hugging Face cache variables if the default home directory is not
writable.

**Permission error in Dataverse data:** Do not make the downloaded record
writable. Re-run `stage_dataset` with a new empty directory under `RUN_ROOT` and
point the regression adapter to that staged tree.

**A baseline environment cannot import its upstream modules:** Use the
baseline-specific environment source linked in [Create environments](#2-create-environments).
The upstream projects require incompatible PyTorch, PyTorch Geometric, and
Lightning versions, so one environment is not guaranteed to run all 13
adapters.
