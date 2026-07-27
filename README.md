# Leak-CURBER

Code, configs, and baseline adapters for the **Leak-CURBER** benchmark: a
suite of leakage-aware splits for kinetic-parameter prediction, binding
affinity prediction, enzyme classification, enzyme retrieval, and reaction
outcome.

This repository is one part of the Leak-CURBER Dataverse release. The complete
processed benchmark and a reviewer sample are available in a separate
Dataverse record. A small reproducible **sample** of the core benchmark is also
bundled in-tree under
[`00-sample__benchmark_datasets/`](00-sample__benchmark_datasets/) so the
baseline adapters and data utilities can be smoke-tested before downloading the
full release.

## Repository layout

```
Leak-CURBER/
├── src/                            Dataset, feature, splitting, and utility code
│   ├── data/components/            Dataset builders and data-processing components
│   │   ├── embedders/              Protein, molecule, and reaction embedders
│   │   ├── generators/             Structure and conformer generation helpers
│   │   └── splitters/              Leakage-aware split generators
│   └── utils/                      Logging, instantiation, chemistry helpers
├── configs/                        Hydra configuration trees
│   ├── data/                       Dataset and split-generation configs
│   │   └── splits_dataset/         Task-specific split configs
│   ├── paths/                      Path resolution (driven by env vars)
│   └── data_processing.yaml        Aggregates all data-processing configs
├── baselines/                      Third-party baselines + Leak-CURBER adapters
│   ├── BACPI/  BIND/  CARE/  CataPro/  CLEAN/  Clipzyme/  DEKP/
│   ├── EITLEM-Kinetics/  eMOSAIC/  GraphEC/  HITEC/  Horizon/
│   └── KcatNet/  StructureFree-DTA/
├── 00-sample__benchmark_datasets/  Bundled sample of the full benchmark (see its README)
├── environment.yml                 Conda environment spec (env name: leakcurber)
├── .env.example                    Template for local env-var overrides
├── .project-root                   Marker used by rootutils to anchor paths
├── REPRODUCING_BASELINES.md        Step-by-step instructions per baseline
└── README.md                       This file
```

Baseline directories contain copied upstream projects plus local Leak-CURBER
adapter code where available. Upstream README files describe the original
projects and may reference their original datasets, environments, or download
locations. For Leak-CURBER usage, prefer each `emulator_bench/README.md`.

## Setup

### 1. Install dependencies

Create a conda environment (Python >= 3.12 recommended) and install from
[`requirements.txt`](requirements.txt):

```bash
conda create -n leakcurber python=3.12 -y
conda activate leakcurber
pip install -r requirements.txt
```

`requirements.txt` lists the direct top-level dependencies and pulls four
patched forks (`chemeq`, `unimol_tools`, `clamp`, `rxnfp`) from
`anonymous.4open.science`.

If you already have the project environment used for validation, activate it
instead:

```bash
conda activate mldb
```

### 2. Set required environment variables

The code repository and Dataverse download are separate directories. After
downloading and extracting the Dataverse record, its top-level layout is:

```text
DATAVERSE_ROOT/
├── 00-sample__benchmark_datasets/
│   ├── binding_affinity_dataset/
│   ├── enzyme_classification_dataset/
│   ├── enzyme_retrieval_dataset/
│   ├── kinetic_params_dataset/
│   └── reaction_outcome_dataset/
└── 01_core_benchmark_datasets/
    ├── binding_affinity_dataset/
    ├── enzyme_classification_dataset/
    ├── enzyme_retrieval_dataset/
    ├── kinetic_params_dataset/
    └── reaction_outcome_dataset/
```

Set the code, Dataverse, and full-dataset roots explicitly:

```bash
export PROJECT_ROOT=/absolute/path/to/cloned/Leak-CURBER
export DATAVERSE_ROOT=/absolute/path/to/downloaded/Leak-CURBER_dataverse
export DATASET_ROOT="$DATAVERSE_ROOT/01_core_benchmark_datasets"
```

Required:

| Variable          | Purpose                                                                    |
|-------------------|----------------------------------------------------------------------------|
| `PROJECT_ROOT`    | Absolute path to the cloned code repository. |
| `DATAVERSE_ROOT`  | Absolute path to the extracted Dataverse download containing the `00` and `01` directories above. |
| `DATASET_ROOT`    | Dataset tier used by a baseline command. Point it at the bundled sample or `01_core_benchmark_datasets`. |

The Dataverse record intentionally excludes the 44 GB shared embedding store.
When a baseline requires cached features, its adapter generates them from the
released protein, molecule, or reaction inputs through the baseline's native
feature pipeline. This avoids a large mandatory download and preserves
model-specific preprocessing.

Optional (only needed for the matching feature):

- `PUBMED_API_KEY`, `BRENDA_PASSWORD` — used by [src/utils/chem_utils.py](src/utils/chem_utils.py) for raw-source ingestion only.

### 3. Anonymization placeholder

One model identifier is intentionally hidden during anonymous review:

- [configs/data/embeddings.yaml](configs/data/embeddings.yaml):
  `smited_embeddings_model_id: <your-hf-namespace>/materials-smi-ted-fork`.
  This value is needed only to regenerate SMI-TED embeddings. It does not
  affect use of the released benchmark or the CataPro example below. The
  de-anonymized public release will provide the public model identifier.

## Running baselines

For step-by-step reproduction of each baseline result, see
**[REPRODUCING_BASELINES.md](REPRODUCING_BASELINES.md)**.

That document covers the seven regression baselines:

| Baseline        | Tasks                |
|-----------------|----------------------|
| KcatNet         | kcat, Km             |
| CataPro         | kcat                 |
| EITLEM-Kinetics | kcat, Km             |
| DEKP            | Km                   |
| eMOSAIC         | Ki                   |
| BIND            | Ki, EC50, IC50       |
| BACPI           | Ki, EC50, IC50       |

The remaining baselines target enzyme classification, enzyme retrieval, and
reaction-outcome tasks. CLEAN, Clipzyme, GraphEC, HITEC, Horizon, and
StructureFree-DTA include Leak-CURBER adapters under
`baselines/<NAME>/emulator_bench/`. CARE is included as an upstream reference
baseline in this checkout, but it does not have an `emulator_bench/` adapter.

Quick orientation:

1. `baselines/<NAME>/README.md` — upstream baseline's environment setup and model description.
2. `baselines/<NAME>/emulator_bench/README.md` — Leak-CURBER adapter usage, when an adapter is present.
3. `baselines/<NAME>/commands.txt` (when present) — verbatim commands from our runs.
4. `baselines/<NAME>/queue_multiple_seeds.sh` (when present) — multi-seed sweep script.

## Verified end-to-end example: CataPro k<sub>cat</sub> P-seq

This example performs feature generation, training, test evaluation, and result
aggregation for one originally submitted baseline. It uses the bundled P-seq
sample, which contains 900 training rows, 50 validation rows, and 50 test rows.
The example writes all generated files to a temporary directory. It does not
modify the code repository or downloaded Dataverse files.

### 1. Create the CataPro environment

```bash
conda create -n leakcurber-catapro python=3.12 -y
conda activate leakcurber-catapro

python -m pip install \
  torch \
  "transformers==4.57.6" \
  sentencepiece \
  protobuf \
  numpy \
  pandas \
  pyarrow \
  scikit-learn \
  rdkit \
  tqdm
```

A CUDA-capable GPU and internet access are required. During the first run,
CataPro downloads `Rostlab/prot_t5_xl_uniref50` and
`laituan245/molt5-base-smiles2caption` from Hugging Face.

### 2. Prepare a writable run directory

Run these commands from the cloned repository:

```bash
cd "$PROJECT_ROOT"

export DATASET_ROOT="$PROJECT_ROOT/00-sample__benchmark_datasets"
export CATAPRO_SOURCE="$DATASET_ROOT/kinetic_params_dataset/kcat/enzyme_sequence_splits/threshold_0.01"
export CATAPRO_RUN_ROOT
CATAPRO_RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/leakcurber-catapro-kcat-pseq.XXXXXX")"

mkdir -p "$CATAPRO_RUN_ROOT/enzyme_sequence_splits/threshold_0.01"
cp "$CATAPRO_SOURCE/train.parquet" \
   "$CATAPRO_SOURCE/val.parquet" \
   "$CATAPRO_SOURCE/test.parquet" \
   "$CATAPRO_RUN_ROOT/enzyme_sequence_splits/threshold_0.01/"

export HF_HOME="$CATAPRO_RUN_ROOT/huggingface"
```

### 3. Run the complete CataPro workflow

The checked-in `default_hparams.json` supplies CataPro's native settings,
including its 150-epoch training budget. The protein batch size below controls
feature-generation memory only. It does not change the training configuration.

```bash
cd "$PROJECT_ROOT/baselines/CataPro"

CUDA_VISIBLE_DEVICES=0 python emulator_bench/run_split_benchmarks.py \
  --value_root "$CATAPRO_RUN_ROOT" \
  --split_groups enzyme_sequence_splits \
  --thresholds threshold_0.01 \
  --cache_dir "$CATAPRO_RUN_ROOT/.cache_embeddings" \
  --prot_batch_size 8 \
  --seeds 666 \
  --device cuda:0 \
  --primary_metric R2 \
  --higher_is_better
```

The primary test metrics are written to:

```text
$CATAPRO_RUN_ROOT/enzyme_sequence_splits/threshold_0.01/catapro_results/seed_666/final_results_test.csv
```

The aggregate result is written to:

```text
$CATAPRO_RUN_ROOT/catapro_summary.csv
```

The corresponding run-level index, including seed 666 and threshold 0.01, is
written to `$CATAPRO_RUN_ROOT/catapro_summary_runs.csv`.

Verify the run and print the test metrics:

```bash
export CATAPRO_RESULT_DIR="$CATAPRO_RUN_ROOT/enzyme_sequence_splits/threshold_0.01/catapro_results/seed_666"
export CATAPRO_FINAL="$CATAPRO_RESULT_DIR/final_results_test.csv"
export CATAPRO_PREDICTIONS="$CATAPRO_RESULT_DIR/pred_label_test.csv"
export CATAPRO_SUMMARY="$CATAPRO_RUN_ROOT/catapro_summary.csv"
export CATAPRO_RUNS="$CATAPRO_RUN_ROOT/catapro_summary_runs.csv"

test -s "$CATAPRO_FINAL"
test -s "$CATAPRO_PREDICTIONS"
test -s "$CATAPRO_SUMMARY"
test -s "$CATAPRO_RUNS"

python - "$CATAPRO_FINAL" "$CATAPRO_PREDICTIONS" "$CATAPRO_SUMMARY" "$CATAPRO_RUNS" <<'PY'
import sys
import numpy as np
import pandas as pd

metrics = pd.read_csv(sys.argv[1])
predictions = pd.read_csv(sys.argv[2])
summary = pd.read_csv(sys.argv[3])
runs = pd.read_csv(sys.argv[4])
required = {"PCC", "SCC", "R2", "RMSE", "MSE", "MAE"}
missing = required.difference(metrics.columns)
assert not missing, f"Missing metric columns: {sorted(missing)}"
assert np.isfinite(metrics[list(required)].to_numpy()).all(), "Test metrics must be finite"
assert len(predictions) == 50, f"Expected 50 test predictions, found {len(predictions)}"
assert summary["n_seeds"].tolist() == [1]
assert runs["seed"].astype(int).tolist() == [666]
assert runs["threshold"].tolist() == ["threshold_0.01"]
print(metrics.to_string(index=False))
PY
```

This sampled-data result verifies the complete execution path. It is not a
paper result.

### 4. Use the complete Dataverse dataset

For the full k<sub>cat</sub> P-seq evaluation, prepare a new temporary run
directory from the `01` dataset tier:

```bash
export DATASET_ROOT="$DATAVERSE_ROOT/01_core_benchmark_datasets"
export CATAPRO_SOURCE="$DATAVERSE_ROOT/01_core_benchmark_datasets/kinetic_params_dataset/kcat/enzyme_sequence_splits/threshold_0.01"
export CATAPRO_RUN_ROOT
CATAPRO_RUN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/leakcurber-catapro-kcat-pseq-full.XXXXXX")"

mkdir -p "$CATAPRO_RUN_ROOT/enzyme_sequence_splits/threshold_0.01"
cp "$CATAPRO_SOURCE/train.parquet" \
   "$CATAPRO_SOURCE/val.parquet" \
   "$CATAPRO_SOURCE/test.parquet" \
   "$CATAPRO_RUN_ROOT/enzyme_sequence_splits/threshold_0.01/"

export HF_HOME="$CATAPRO_RUN_ROOT/huggingface"
```

Then run Step 3. The first run builds the native CataPro feature cache. Later
runs can reuse that cache.

## Other bundled-sample workflows

The sample under [`00-sample__benchmark_datasets/`](00-sample__benchmark_datasets/)
mirrors the task layout of the full release with ~50-900 rows per split. It is
a drop-in stand-in for smoke-testing baseline adapters before downloading the
full data.

Point `--base_dir`, `--value_root`, `--dataset-root`, or the relevant config
field at the matching directory under `$PROJECT_ROOT/00-sample__benchmark_datasets`
instead of the full dataverse path. See
[`00-sample__benchmark_datasets/README.md`](00-sample__benchmark_datasets/README.md)
and `SAMPLE_MANIFEST.json` for full provenance.

## Licenses

Baseline licenses are inherited from the copied upstream projects where license
files are present in the corresponding `baselines/<NAME>/` directory. CLEAN
ships its license as a PDF, and a few copied baseline directories in this
checkout do not expose a top-level `LICENSE*` file. The Leak-CURBER adapter code
inside each `emulator_bench/` folder, the `src/` tree, and the configs are
released under the dataverse's top-level license.
