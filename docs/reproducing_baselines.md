# Reproducing the baseline results

This file explains how to rerun every baseline of the Leak-CURBER paper from this repository.

Each baseline lives in its own submodule under `baselines/`. Each submodule has one script, `emulator_bench/run.sh`. The script builds the feature cache, trains and tests all splits for seeds 666, 777 and 888, and writes a summary table. Finished runs are skipped, so you can stop and restart it.

Two baseline families have no `run.sh`. The reaction outcome models (syntheseus) use a command list. The internal references (MLP, XGBoost, BLASTP, Foldseek) use scripts in this repository.

## Coverage

All paths are relative to the repository root. The submodule name can differ from the paper name.

| Baseline (paper name) | Subtasks | Submodule |
|---|---|---|
| KcatNet | kcat | `baselines/KcatNet` |
| EITLEM-Kinetics | kcat, Km | `baselines/EITLEM-Kinetics` |
| CataPro | kcat, Km | `baselines/CataPro` |
| DEKP | Km (kcat in the appendix) | `baselines/DEKP` |
| DLKcat | kcat | `baselines/DLKcat` |
| UniKP | kcat, Km | `baselines/UniKP` |
| MPEK | kcat, Km, pH and temperature study | `baselines/mpek` |
| CatPred | kcat, Km, Ki | `baselines/CatPred` |
| eMOSAIC-T | Ki | `baselines/eMOSAIC` |
| BIND | Ki, Kd, EC50, IC50 | `baselines/BIND` |
| BACPI | Ki, Kd, EC50, IC50 | `baselines/BACPI` |
| Structure-Free DTA | Kd | `baselines/StructureFree-DTA` |
| CLEAN | EC classification | `baselines/CLEAN` |
| GraphEC | EC classification | `baselines/GraphEC` |
| HIT-EC | EC classification | `baselines/HITEC` |
| CREEP | enzyme retrieval | `baselines/CARE/CREEP` |
| CLIPZyme | enzyme retrieval | `baselines/Clipzyme` |
| Horizyn | enzyme retrieval | `baselines/Horizon` |
| Root-aligned SMILES, MEGAN, Chemformer | reaction outcome (retro and forward) | `baselines/syntheseus` |
| MLP, XGBoost | all regression and function subtasks | `scripts/`, `configs/pipeline/` |
| BLASTP, Foldseek | EC classification | `baselines/BLASTP`, `baselines/Foldseek` |

The main paper figures show 17 baselines. The appendix tables add DLKcat, UniKP, MPEK, CatPred and the internal references.

## Setup

### Code and environment

```bash
git clone --recurse-submodules https://github.com/bisect-group/Leak-CURBER.git
cd Leak-CURBER
bash scripts/setup_env.sh            # add --tools for MMseqs2, Foldseek, BLAST and DIAMOND
```

The script creates the `leakcurber` env, installs the packages and this repository, and writes `.env`. Use `-n NAME` to choose another env name. The manual steps are:

```bash
conda env create -f environment.yaml
conda activate leakcurber
pip install --no-deps -r requirements-nodeps.txt
pip install --no-deps -e .
cp .env.example .env                 # then set EMULATOR_PROJECT_ROOT to this folder
bash scripts/install_external_tools.sh
```

The file `environment.yaml` pins the versions of our `leakcurber` env. In our runs, `PY` was that env's `python` for most baselines. Some baselines need their own env. The sections below name them.

Never run `pip install fair-esm` in this env. It overwrites the `esm` package that ESM3 needs. KcatNet and EITLEM-Kinetics unpack a pinned copy of fair-esm by themselves.

### Data

Each `run.sh` reads the full split folders under `data/processed/datasets/` and writes to `outputs/baselines/<submodule name>/`. The dataset folders are `kinetic_dataset`, `binding_affinity_dataset`, `enzyme_classification_dataset`, `enzyme_retrieval_dataset` and `reaction_outcome_dataset`. `DATA_ROOT` means a different folder for each submodule (for example `kinetic_dataset/kcat` for KcatNet). Run `run.sh --dry-run` to see the paths it will use.

The public release (Dataverse, [doi:10.7910/DVN/CZW47K](https://doi.org/10.7910/DVN/CZW47K), version 3.0) does not hold these folders. It has no full datasets, no test sets and none of the other split groups, because they hold leaderboard test rows. With the release alone, you can train on the leaderboard train and validation rows in `leaderboard_train_val/Leak-CURBER-<task>/` and get test scores through the leaderboard. These rows are also on Hugging Face (`bisectgroup/Leak-CURBER-<task>`). The embeddings are on Hugging Face too (`bisectgroup/Leak-CURBER-embeddings`).

What the release does give are the inputs that `run.sh` cannot build. Every archive is a standalone 7z file. Run these commands from the repository root:

```bash
R=/path/to/Leak-CURBER_dataverse
mkdir -p data/intermediate/sdf/datamol data/intermediate/esm data/intermediate/esmfold2_pdb data/intermediate/esmfold2_cif \
  data/processed/datasets/enzyme_retrieval_dataset
for f in "$R"/conformers/*.7z; do 7z x "$f" -odata/intermediate/sdf/datamol; done
for f in "$R"/structures/esm3_generated/*.7z; do 7z x "$f" -odata/intermediate/esm; done
for f in "$R"/structures/esmfold2_pdb/*.7z; do 7z x "$f" -odata/intermediate/esmfold2_pdb; done
for f in "$R"/structures/esmfold2_cif/*.7z; do 7z x "$f" -odata/intermediate/esmfold2_cif; done
python "$R"/structures/id_lists/download_structures.py --out data/intermediate
cp "$R"/structures/experimental_selected_chains/chain_selection_*.json data/intermediate/mmcif/
cp "$R"/atom_maps/reaction_atom_mappings.parquet data/processed/datasets/enzyme_retrieval_dataset/
```

| Release folder | Local path | Used by |
|---|---|---|
| `conformers/` | `data/intermediate/sdf/datamol/<shard>/` | Uni-Mol conformers (the archives hold `<shard>/datamol_*.sdf`) |
| `structures/esm3_generated/` | `data/intermediate/esm/` | ESM3 fallback structures |
| `structures/esmfold2_pdb/` | `data/intermediate/esmfold2_pdb/` | P-struct split, GraphEC, MLP |
| `structures/esmfold2_cif/` | `data/intermediate/esmfold2_cif/` | ESM3 embedder |
| `structures/id_lists/` | `data/intermediate/mmcif/` and `data/intermediate/alphafold/` | RCSB mmCIF files and AlphaFold DB models, downloaded by `download_structures.py --out data/intermediate` |
| `structures/experimental_selected_chains/` (JSON files) | `data/intermediate/mmcif/` | chain selection cache |
| `atom_maps/` | `data/processed/datasets/enzyme_retrieval_dataset/` | CLIPZyme |

The `experimental_selected_chains_*.7z` archives hold the selected chains as PDB files, for reference. No baseline reads them. The code takes experimental structures from the mmCIF files and selects the chain itself.

The `checkpoints/` folder holds baseline checkpoints for the leaderboard splits. Its `README.md` lists the models and seeds. They are not needed for the commands below.

### External tools

```bash
bash scripts/install_external_tools.sh   # mmseqs, foldseek, BLAST, DIAMOND into external/
```

### Run a baseline

```bash
baselines/KcatNet/emulator_bench/run.sh --dry-run   # print the commands only
baselines/KcatNet/emulator_bench/run.sh             # run all steps
baselines/KcatNet/emulator_bench/run.sh train       # run one step
```

The steps are `cache`, `train` and `aggregate`. `all` is the default. `train` also tests. A few submodules add `prepare` or `rescore`, or drop `cache`. The sections below say which.

Set these environment variables to change a run: `PY`, `GPUS` (a space-separated list), `SEEDS`, `RUNS_PER_GPU`, `DATA_ROOT`, `OUT_ROOT` and `EMULATOR_ROOT`. `DRY_RUN=1` has the same effect as `--dry-run`.

Always set `GPUS`. The defaults differ: CataPro and DLKcat use `0 1 2 3`, CatPred uses `1 2 3`, and most others use `0`. MPEK, MPEK assay and UniKP take one GPU id.

## Kinetic and binding regression

The fork README has the details for each baseline. Its "Reproduce" section matches the `run.sh` header.

### KcatNet

```bash
baselines/KcatNet/emulator_bench/run.sh
```

Covers kcat. `DATA_ROOT` is the `kcat` folder. [README](../baselines/KcatNet/emulator_bench/README.md)

### EITLEM-Kinetics

```bash
baselines/EITLEM-Kinetics/emulator_bench/run.sh
```

Covers kcat and Km. The cache holds ESM-1v features. [README](../baselines/EITLEM-Kinetics/emulator_bench/README.md)

### CataPro

```bash
baselines/CataPro/emulator_bench/run.sh
```

Covers kcat and Km. The first run downloads ProtT5 and MolT5 from Hugging Face. [README](../baselines/CataPro/emulator_bench/README.md)

### DEKP

```bash
baselines/DEKP/emulator_bench/run.sh
```

Covers Km and kcat. The steps are `prepare`, `train` and `all`. `train` builds the caches. DEKP trains with `--amp` (bf16). [README](../baselines/DEKP/emulator_bench/README.md)

### DLKcat

```bash
baselines/DLKcat/emulator_bench/run.sh
```

Covers kcat. The `rescore` step exists only for old runs. A new run does not need it, and `all` skips it. [README](../baselines/DLKcat/emulator_bench/README.md)

### UniKP

```bash
baselines/UniKP/emulator_bench/run.sh
```

Covers kcat and Km with Extra Trees on CPU. `JOBS` and `MODEL_N_JOBS` set the CPU use. `GPUS` matters only for ProtT5 vectors that are missing. UniKP reuses the ProtT5 cache of CataPro when it exists. [README](../baselines/UniKP/emulator_bench/README.md)

### MPEK

```bash
git -C baselines/mpek submodule update --init Mole-BERT
baselines/mpek/emulator_bench/run.sh                  # kcat and Km
baselines/mpek/assay_emulator_bench/run.sh            # pH and temperature study
```

Both scripts have a `prepare` step that checks the model assets. The assay script uses the cache of the first script, so run the first script first. Results go to `outputs/baselines/mpek` and `outputs/baselines/MPEK_assay`. [README](../baselines/mpek/emulator_bench/README.md), [assay README](../baselines/mpek/assay_emulator_bench/README.md)

### CatPred

```bash
baselines/CatPred/emulator_bench/run.sh
```

Covers kcat, Km and Ki. The script caches ESM-2 features and ProGRES structure vectors. Ki time splits run through `train_run.py`. [README](../baselines/CatPred/emulator_bench/README.md)

### eMOSAIC-T

```bash
baselines/eMOSAIC/emulator_bench/run.sh
```

Covers Ki, with `--amp` (bf16). The paper runs used 3 runs per GPU on an 80 GB GPU. This only changes speed. The default is 1. [README](../baselines/eMOSAIC/emulator_bench/README.md)

### BIND and BACPI

```bash
baselines/BIND/emulator_bench/run.sh
baselines/BACPI/emulator_bench/run.sh
```

Both cover Ki, Kd, EC50 and IC50. BIND caches in two passes (Ki, Kd and EC50, then IC50). BACPI keeps the IC50 cache in its own folder `embeddings_ic50/`. [BIND README](../baselines/BIND/emulator_bench/README.md), [BACPI README](../baselines/BACPI/emulator_bench/README.md)

### Structure-Free DTA

```bash
baselines/StructureFree-DTA/emulator_bench/run.sh
```

Covers Kd. It has no `cache` step. [README](../baselines/StructureFree-DTA/emulator_bench/README.md)

## Enzyme function

### CLEAN

```bash
PY=/path/to/leakcurber/bin/python ESM_PY=/path/to/clipzyme/bin/python baselines/CLEAN/emulator_bench/run.sh
```

`ESM_PY` runs the ESM-1b cache with the original fair-esm. In our runs it was the `clipzyme` env. [README](../baselines/CLEAN/emulator_bench/README.md)

### GraphEC

```bash
baselines/GraphEC/emulator_bench/run.sh
```

The script needs DIAMOND 2.0.14 at `external/diamond-2.0.14/diamond` (or `DIAMOND_BIN`) and ProtT5 (`PROTTRANS_MODEL_PATH` or the Hugging Face id). Missing structures come from ESMFold. `rescore` scores the test rows that hold non-canonical residues. `all` runs it, and the paper tables use it. [README](../baselines/GraphEC/emulator_bench/README.md)

### HIT-EC

```bash
baselines/HITEC/emulator_bench/run.sh
```

The steps are `prepare`, `train`, `aggregate` and `all`. The batch size is 64. This is our choice, because the authors did not publish it. [README](../baselines/HITEC/emulator_bench/README.md)

### CREEP

```bash
export PATH="$PWD/external/mmseqs/bin:$PATH"
PY=/path/to/creep/bin/python baselines/CARE/CREEP/emulator_bench/run.sh
```

The `prepare` step downloads `ec2go`. The encoders stay frozen, and the candidate pool is all proteins of the split. [README](../baselines/CARE/CREEP/emulator_bench/README.md)

### CLIPZyme

Two inputs must exist before the first run. The script does not make them. Take the atom maps from `atom_maps/` in the release (see [Data](#data)), or generate them:

```bash
# LocalMapper atom maps
python scripts/generate_atom_mapped_reactions.py \
  --input-path data/processed/datasets/enzyme_retrieval_dataset/enzyme_retrieval_dataset.parquet \
  --output-path data/processed/datasets/enzyme_retrieval_dataset/reaction_atom_mappings.parquet
```

The ESM-2 weights must be in `<OUT_ROOT>/weights/torch_hub`. The script never downloads them. Then run:

```bash
PY=/path/to/clipzyme/bin/python baselines/Clipzyme/emulator_bench/run.sh
```

The encoder stays frozen (`--frozen_esm --amp --fast`). [README](../baselines/Clipzyme/emulator_bench/README.md)

### Horizyn

```bash
baselines/Horizon/emulator_bench/run.sh
```

The ProtT5 weights `Rostlab/prot_t5_xl_half_uniref50-enc` must be in the Hugging Face cache. Horizyn trains with bf16. Set `AMP=0` for fp32. The `rescore` step scores without preliminary ECs, and `all` runs it. The paper rows come from `summary_no_b.csv`. [README](../baselines/Horizon/emulator_bench/README.md)

## Reaction outcome

The syntheseus submodule has no `run.sh`. Zero-shot models run released checkpoints and have no seeds. The fine-tuned runs use seeds. The README explains the method. `commands.txt` lists every command in order: corpora, checkpoints, curated retro split files, zero-shot runs, fine-tuning and summaries. Run the commands from `baselines/syntheseus`, in the env `syntheseus_modern`.

```bash
cd baselines/syntheseus
python -m emulator_bench.launch --gpus 0 --workers 4 --procs_per_gpu 4 --dry_run
```

Remove `--dry_run` for the full zero-shot run. The paper models are `Chemformer`, `MEGAN` and `RootAligned`. See [README](../baselines/syntheseus/emulator_bench/README.md) and [commands.txt](../baselines/syntheseus/emulator_bench/commands.txt).

## Internal references

### MLP and XGBoost

Both use the same orchestrator. Each has its own Optuna search space, and each runs 3 seeds. Build the input data first. The command keeps existing files. Add `--overwrite` to rebuild.

```bash
python scripts/build_baseline_mlp_data.py --verify
python scripts/build_baseline_mlp_retrieval_protein_data.py   # protein-based retrieval splits
```

The data goes to `outputs/baselines/MLP/data/`. Each row gets one ESM3 structure key. The rule is:

1. The experimental chain with the highest coverage. A tie goes to the smallest PDB id.
2. Else the AlphaFold model (`AF-<accession>-F1`, highest version), then ESMFold2, then ESM3.
3. Else no key. The row gets a zero vector, and so does a key that is not in the ESM3 store.

`--verify` writes `structure_report.csv`. `metrics.json` lists the count of zero-vector rows per split in `missing_embeddings`.

Regression runs use `scripts/run_baseline_mlp_pipeline.py`. Classification and retrieval runs use `scripts/run_baseline_classification_pipeline.py`. Pick the profile in `configs/pipeline/`:

```bash
python scripts/run_baseline_mlp_pipeline.py --run_config configs/pipeline/baseline_mlp_optuna100.yaml
python scripts/run_baseline_mlp_pipeline.py --run_config configs/pipeline/baseline_xgboost_cpu_optuna100.yaml
python scripts/run_baseline_mlp_pipeline.py --run_config configs/pipeline/baseline_xgboost_optuna100.yaml
python scripts/run_baseline_classification_pipeline.py --run_config configs/pipeline/baseline_classification_mlp_optuna100.yaml
python scripts/run_baseline_classification_pipeline.py --run_config configs/pipeline/baseline_classification_mlp_protein_ret.yaml
python scripts/run_baseline_classification_pipeline.py --run_config configs/pipeline/baseline_classification_xgboost_optuna100.yaml --n_trials 30
```

The two XGBoost regression profiles split the work. The CPU profile covers EC50, kcat, Kd, Ki and Km. The GPU profile covers IC50. XGBoost retrieval used 30 Optuna trials. The profile says 100. Other flags and profile fields are in [baselines/mlp_pipeline.md](baselines/mlp_pipeline.md).

Each classification and retrieval run writes `ec_level_accuracy.json` next to `test_predictions.parquet`. Retrieval runs also write `ec_level_accuracy_no_b.json`, which is scored without preliminary B-ECs. The paper reports this file for retrieval.

### BLASTP and Foldseek

Both are deterministic. They have no seeds and no GPU use. They search the test proteins against the training partition of each classification split and transfer the ECs of the hits.

```bash
conda run --no-capture-output -n leakcurber python baselines/BLASTP/emulator_bench/run_blastp_ec.py \
  --threads-per-run 8 --total-threads 8
conda run --no-capture-output -n leakcurber python baselines/Foldseek/emulator_bench/run_foldseek_ec.py --threads 8
```

BLASTP uses `external/diamond-2.1.9/diamond`. Foldseek uses `external/foldseek/bin/foldseek` and the structures in `data/intermediate`. `--force` rebuilds finished splits. [BLASTP README](../baselines/BLASTP/README.md), [Foldseek README](../baselines/Foldseek/README.md)

## Check the results

- Run `aggregate` again after any run. Each split row must show 3 seeds.
- Regression tables report MSE, RMSE, MAE, R2, Pearson and Spearman correlation.
- Classification and retrieval tables report EC level accuracy at levels 4, 3, 2 and 1. The level 4 value is the exact EC match.
- Do not pass `--overwrite` or similar flags to a fork script unless you want to replace finished results.

## Troubleshooting

No jobs found. Check that `DATA_ROOT` points to the folder this submodule expects. Use `--dry-run` to print it.

Out of memory during features. Lower the encoder batch size. The common flags are CataPro `--prot_batch_size`, BIND `--max_batch`, DEKP `--prot_t5_max_batch`, eMOSAIC `--chunk_size` and Structure-Free DTA `--protein_batch_size`.

Out of memory during training. Lower `RUNS_PER_GPU`. If that is not enough, lower the batch size in the fork. Keep the effective batch size with gradient accumulation where the fork supports it.

A model download fails. The first cache run can download weights from Hugging Face. Check the network. Set `HF_HOME` if your home folder is read-only.

An import fails. The upstream projects need different PyTorch, PyTorch Geometric and Lightning versions. One env cannot run all baselines. Use the env named in the fork README.
