# Leak-CURBER

Code, configs, and baseline adapters for the **Leak-CURBER** benchmark — a
suite of leakage-aware splits for kinetic-parameter prediction, binding
affinity prediction, enzyme classification, enzyme retrieval, and reaction
outcome.

This repository is one part of the Leak-CURBER dataverse release. The full
benchmark datasets, raw sources, intermediates, and Croissant metadata live
in sibling directories of the dataverse root (see [Dataverse layout](#dataverse-layout)
below). A small reproducible **sample** of the core benchmark is bundled
in-tree under [`00-sample__benchmark_datasets/`](00-sample__benchmark_datasets/)
so the code can be exercised end-to-end without downloading the full release.

## Repository layout

```
Leak-CURBER/
├── src/                            Custom dataset, model, and utility code
│   ├── data/components/            Dataset builders, embedders, generators, splitters
│   ├── models/                     Model definitions (MLP, FlexMoE, …)
│   └── utils/                      Logging, instantiation, chemistry helpers
├── configs/                        Hydra configuration trees
│   ├── data/                       Dataset configs (kinetic params, binding affinity,
│   │   └── splits_dataset/         splits, classification, retrieval, reactions)
│   ├── model/                      Model configs (mlp, flexmoe)
│   ├── trainer/                    Lightning Trainer presets
│   ├── logger/                     Loggers (wandb, neptune, comet, aim, …)
│   ├── paths/                      Path resolution (driven by env vars)
│   ├── experiment/                 Experiment overlays
│   ├── hparams_search/             Optuna sweeps
│   ├── train.yaml                  Default train config
│   ├── eval.yaml                   Default eval config
│   ├── data_processing.yaml        Aggregates all data-processing configs
│   └── flexmoe_train.yaml          FlexMoE-specific train config
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

Each baseline directory under `baselines/` is an unmodified copy of the upstream
project (with its own README and license) plus an `emulator_bench/` subdirectory
containing the Leak-CURBER adapter scripts.


## Setup

### 1. Install dependencies

Create a conda env (Python ≥ 3.12 recommended) and install from
[`requirements.txt`](requirements.txt):

```bash
conda create -n leakcurber python=3.12 -y
conda activate leakcurber
pip install -r requirements.txt
```

`requirements.txt` lists the direct top-level dependencies and pulls four
patched forks (`chemeq`, `unimol_tools`, `clamp`, `rxnfp`) from
`anonymous.4open.science`.

### 2. Set required environment variables

Configs and baseline adapters resolve paths through environment variables.
Copy the template and fill it in:

```bash
cp .env.example .env
# edit .env with your values
```

Required:

| Variable          | Purpose                                                                    |
|-------------------|----------------------------------------------------------------------------|
| `PROJECT_ROOT`    | Absolute path to your working copy (this directory). Configs anchor every dataset and embedding path against this. |
| `DATAVERSE_ROOT`  | Absolute path to the parent dataverse root. Used by [configs/data/embeddings.yaml](configs/data/embeddings.yaml) to locate `01_core_benchmark_datasets/`. |

Optional (only needed for the matching feature):

- `WANDB_ENTITY` — `entity` is `null` in [configs/logger/wandb.yaml](configs/logger/wandb.yaml); set this or edit the config.
- `NEPTUNE_API_TOKEN`, `COMET_API_TOKEN` — referenced by the corresponding logger configs.
- `PUBMED_API_KEY`, `BRENDA_PASSWORD` — used by [src/utils/chem_utils.py](src/utils/chem_utils.py) for raw-source ingestion only.

### 3. Placeholders to fill in

Two values were left as placeholders during anonymization:

- [configs/data/embeddings.yaml](configs/data/embeddings.yaml): `smited_embeddings_model_id: <your-hf-namespace>/materials-smi-ted-fork` — set to your HuggingFace fork of `materials.smi-ted`, or upstream if compatible.
- [configs/logger/wandb.yaml](configs/logger/wandb.yaml): `entity: null` — set to your W&B team if logging to W&B.

## Running the main code

The `src/` and `configs/` trees follow the
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
convention. The expected entrypoint is a `train.py` at the project root that
imports from `src/`.

> **Note**: this dataverse release ships configs and library code only. The
> `train.py` driver script is not bundled here; provide your own (a few dozen
> lines that load the Hydra config and instantiate the Lightning Trainer) or
> wire your own runner against the `src/` modules.

Once `train.py` is in place, typical invocations follow Hydra conventions:

```bash
python train.py                                       # default composition (data=mlp, model=mlp, logger=wandb)
python train.py model=flexmoe trainer=gpu             # override individual fields
python train.py experiment=example                    # use an experiment preset
python train.py -m hparams_search=mlp_optuna          # Optuna sweep
```

See [configs/train.yaml](configs/train.yaml) for the default composition and
[configs/experiment/](configs/experiment/) for example overlays.

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

The remaining seven baselines (CARE, CLEAN, Clipzyme, GraphEC, HITEC, Horizon,
StructureFree-DTA) target enzyme classification, enzyme retrieval, and
reaction-outcome tasks. See each baseline's own
`baselines/<NAME>/emulator_bench/README.md` for usage.

Quick orientation for any baseline:

1. `baselines/<NAME>/README.md` — upstream baseline's environment setup and model description.
2. `baselines/<NAME>/emulator_bench/README.md` — Leak-CURBER adapter usage.
3. `baselines/<NAME>/commands.txt` (when present) — verbatim commands from our runs.
4. `baselines/<NAME>/queue_multiple_seeds.sh` (when present) — multi-seed sweep script.

## Trying it on the bundled sample

The sample under [`00-sample__benchmark_datasets/`](00-sample__benchmark_datasets/)
mirrors the layout of `01_core_benchmark_datasets/` (the full release) with
~50–900 rows per split. It is a drop-in stand-in for smoke-testing the
training pipelines and baseline adapters before downloading the full data.

Point `--base_dir` (or the relevant config field) at
`$PROJECT_ROOT/00-sample__benchmark_datasets/<task>/<value_type>` instead of
the full dataverse path. See
[`00-sample__benchmark_datasets/README.md`](00-sample__benchmark_datasets/README.md)
and `SAMPLE_MANIFEST.json` for full provenance.

## Licenses

Each baseline retains its upstream license (see `baselines/<NAME>/LICENSE*`).
The Leak-CURBER adapter code inside each `emulator_bench/` folder, the `src/`
tree, and the configs are released under the dataverse's top-level license.
