# Leak-CURBER

Code, configs, and baseline adapters for the **Leak-CURBER** benchmark: a
suite of leakage-aware splits for kinetic-parameter prediction, binding
affinity prediction, enzyme classification, enzyme retrieval, and reaction
outcome.

This repository is one part of the Leak-CURBER dataverse release. The full
benchmark datasets, raw sources, intermediates, and Croissant metadata live
outside this anonymous code tree. A small reproducible **sample** of the core
benchmark is bundled in-tree under
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

Configs and baseline adapters resolve paths through environment variables.
Set these explicitly in your shell or in your local environment manager:

```bash
export PROJECT_ROOT=/absolute/path/to/Leak-CURBER
export DATAVERSE_ROOT=/absolute/path/to/Leak-CURBER_dataverse
export DATASET_ROOT=$PROJECT_ROOT/00-sample__benchmark_datasets
```

Required:

| Variable          | Purpose                                                                    |
|-------------------|----------------------------------------------------------------------------|
| `PROJECT_ROOT`    | Absolute path to your working copy (this directory). Configs anchor every dataset and embedding path against this. |
| `DATAVERSE_ROOT`  | Absolute path to the parent dataverse root. Used by configs that read full-release data and intermediates outside this anonymous code tree. |
| `DATASET_ROOT`    | Convenience variable used by the baseline examples. Point it at the bundled sample or the full benchmark dataset directory. |

Optional (only needed for the matching feature):

- `PUBMED_API_KEY`, `BRENDA_PASSWORD` — used by [src/utils/chem_utils.py](src/utils/chem_utils.py) for raw-source ingestion only.

### 3. Placeholders to fill in

One value was left as a placeholder during anonymization:

- [configs/data/embeddings.yaml](configs/data/embeddings.yaml): `smited_embeddings_model_id: <your-hf-namespace>/materials-smi-ted-fork` — set to your HuggingFace fork of `materials.smi-ted`, or upstream if compatible.

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

## Trying it on the bundled sample

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
