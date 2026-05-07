# 05_code — Leak-CURBER code

This directory contains the modeling code, configs, and baseline adapters that
accompany the **Leak-CURBER** benchmark dataverse. It is one part of a larger
release; see the dataverse root for the benchmark datasets, raw sources,
intermediates, and metadata.

## Layout

```
05_code/
├── src/                  Custom dataset, model, and utility code (Hydra + Lightning)
│   ├── data/             Dataset builders, embedders, generators, splitters
│   ├── models/           Model definitions (e.g. MLP, FlexMoE)
│   └── utils/            Logging, instantiation, chemistry helpers
├── configs/              Hydra configuration trees
│   ├── data/             Dataset configs (kinetic params, binding affinity, splits, …)
│   ├── model/            Model configs (mlp, flexmoe)
│   ├── trainer/          Lightning Trainer presets
│   ├── logger/           Loggers (wandb, neptune, comet, aim, …)
│   ├── paths/            Path resolution (driven by env vars — see below)
│   ├── experiment/       Experiment overlays
│   ├── hparams_search/   Optuna sweeps
│   ├── train.yaml        Top-level train entrypoint config
│   ├── eval.yaml         Top-level eval entrypoint config
│   ├── data_processing.yaml   Aggregates all data-processing configs
│   └── flexmoe_train.yaml     FlexMoE-specific train config
└── baselines/            Third-party baselines + Leak-CURBER adapters
    ├── BACPI/            (each baseline keeps its upstream README, code, license,
    ├── BIND/              and adds an `emulator_bench/` directory with our
    ├── CARE/              adapter scripts that wire the baseline to Leak-CURBER
    ├── CataPro/           split files)
    ├── CLEAN/
    ├── Clipzyme/
    ├── DEKP/
    ├── EITLEM-Kinetics/
    ├── eMOSAIC/
    ├── GraphEC/
    ├── HITEC/
    ├── Horizon/
    ├── KcatNet/
    └── StructureFree-DTA/
```

## Required environment variables

The configs and baseline adapters resolve paths through environment variables.
Set these before running any code:

| Variable          | Purpose                                                               |
|-------------------|-----------------------------------------------------------------------|
| `PROJECT_ROOT`    | Absolute path to your local working copy of this project (the parent of `data/`, `baselines/`, etc. as expected by the configs). |
| `DATAVERSE_ROOT`  | Absolute path to the Leak-CURBER dataverse root (used by [configs/data/embeddings.yaml](configs/data/embeddings.yaml) to locate `01_core_benchmark_datasets/`). |

Optional, only if you use that logger:
- `WANDB_ENTITY` (the W&B entity is left as `null` in [configs/logger/wandb.yaml](configs/logger/wandb.yaml); set this env var or edit the config)
- `NEPTUNE_API_TOKEN`, `COMET_API_TOKEN`
- `PUBMED_API_KEY`, `BRENDA_PASSWORD` (used by [src/utils/chem_utils.py](src/utils/chem_utils.py) for raw-source ingestion only)

Example:

```bash
export PROJECT_ROOT=/path/to/your/working/copy
export DATAVERSE_ROOT=/path/to/Leak-CURBER_dataverse
```

## Placeholders to fill in before running

A small number of identifiers were stripped during anonymization. Replace
these with values appropriate to your environment:

- [configs/data/embeddings.yaml](configs/data/embeddings.yaml): `smited_embeddings_model_id: <your-hf-namespace>/materials-smi-ted-fork` — set to your HuggingFace fork of `materials.smi-ted`, or upstream if compatible.
- [configs/logger/wandb.yaml](configs/logger/wandb.yaml): `entity: null` — set to your W&B team if logging to W&B.

## Running the main code

The `src/` and `configs/` trees follow the
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
convention. Once you place your training entry script (`train.py`) at the
project root that imports from `src/`, typical invocations look like:

```bash
# Train with default config (data=mlp, model=mlp, logger=wandb)
python train.py

# Override individual fields
python train.py model=flexmoe trainer=gpu logger=tensorboard

# Use an experiment preset
python train.py experiment=example

# Optuna hyperparameter search
python train.py -m hparams_search=mlp_optuna
```

Hydra resolves configs against the directories under [configs/](configs/); see
[configs/train.yaml](configs/train.yaml) for the default composition.

## Running baselines

Each baseline directory under [baselines/](baselines/) is an unmodified copy of
the upstream project (with its own README, code, and license) plus an
**`emulator_bench/`** subdirectory containing the Leak-CURBER adapter scripts
we added on top.

Typical baseline workflow (specifics differ per baseline — see the per-baseline
`emulator_bench/README.md`):

1. Read `baselines/<NAME>/README.md` for the upstream baseline's environment setup.
2. Read `baselines/<NAME>/emulator_bench/README.md` for adapter usage.
3. Optionally consult `baselines/<NAME>/commands.txt` for verbatim commands we ran.
4. Set `PROJECT_ROOT` and run the bench scripts (`cache_*.py`, `train_*.py`,
   `run_split_benchmarks.py`, `tune_optuna.py`, `aggregate_*.py`, etc.).

Most baselines expose a `queue_multiple_seeds.sh` for running multi-seed sweeps.

## Notes

- This is the code release; benchmark data, raw sources, intermediates, and
  Croissant metadata live in sibling directories (`00-sample__benchmark_datasets/`,
  `01_core_benchmark_datasets/`, `02_raw_sources/`, `03_intermediate/`,
  `04_croissant_metadata/`).
- Each baseline retains its upstream license. The Leak-CURBER adapter code
  inside each `emulator_bench/` folder is released under the dataverse's
  top-level license.
- A training entry script (`train.py`) at the project root is expected by the
  configs but is not included in this directory; it lives at the project root
  alongside `src/` and `configs/` in the working layout.
