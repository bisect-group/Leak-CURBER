# Leak-CURBER

Leak-CURBER is a leakage-controlled benchmark for machine learning models of enzymatic reactions. It is a NeurIPS 2026 Evaluations & Datasets paper: [*Leak-CURBER: A Leakage-Controlled Multimodal Evaluation Benchmark for Enzymatic Reaction Tasks*](https://neurips.cc/virtual/2026/loc/sydney/poster/139384).

The benchmark merges data from 10 biochemical databases. It covers 4 tasks and 10 subtasks:

- Reaction outcome: forward synthesis and retrosynthesis.
- Enzyme function: EC classification and enzyme retrieval.
- Kinetic parameters: kcat, Km and Ki.
- Protein-ligand binding: Kd, EC50 and IC50.

The splits control overlap between train and test along protein sequence, protein structure, molecule structure, reaction similarity and annotation time. Models that look strong on random splits often fail on these splits.

This repository holds the code to retrain the baselines, create embeddings and run the splitters, with the configs for each. The paper tables come from it.

## Data

| What | Where |
|---|---|
| Leaderboard train and validation rows, with the inputs that the baselines need (conformers, structures, atom maps) | Harvard Dataverse, [doi:10.7910/DVN/CZW47K](https://doi.org/10.7910/DVN/CZW47K), version 3.0 |
| Leaderboard train and validation rows, one repository for each subtask | Hugging Face, `bisectgroup/Leak-CURBER-<task>` |
| Precomputed embeddings | Hugging Face, `bisectgroup/Leak-CURBER-embeddings` |

On Dataverse the rows are in `leaderboard_train_val/Leak-CURBER-<task>/`. The Hugging Face repositories hold the same rows, with train and validation merged. A small sample of every file is in [`sample_data/`](sample_data/).

## Setup

```bash
git clone --recurse-submodules https://github.com/bisect-group/Leak-CURBER.git
cd Leak-CURBER
bash scripts/setup_env.sh
conda activate leakcurber
```

The script creates the `leakcurber` env, installs the packages and this repository, and writes `.env` with `EMULATOR_PROJECT_ROOT` set to this folder. Use `-n NAME` to choose another env name. Add `--tools` to also install the external search tools (MMseqs2, Foldseek, BLAST, DIAMOND) into `external/`. `PUBMED_EMAIL` and `PUBMED_API_KEY` in `.env` are only for PubMed lookups in the chemistry utilities.

The manual steps are:

```bash
conda env create -f environment.yaml
conda activate leakcurber
pip install --no-deps -r requirements-nodeps.txt
pip install --no-deps -e .
cp .env.example .env                       # then set EMULATOR_PROJECT_ROOT to this folder
bash scripts/install_external_tools.sh     # optional: MMseqs2, Foldseek, BLAST, DIAMOND
```

Do not run `pip install fair-esm` in this env. It overwrites the `esm` package that ESM3 needs.

## Repository layout

```
Leak-CURBER/
├── src/                  Embedders, structure and conformer generators, splitters, models, utilities
├── configs/              Hydra configs: paths, split inputs, split settings, baseline pipelines
├── scripts/              Embedding, split, MLP/XGBoost and classification pipeline scripts
├── baselines/            One submodule per baseline, plus BLASTP and Foldseek
├── docs/                 Dataset construction and split documents
├── figures/              Figure scripts and result tables
├── tests/                Unit tests
├── sample_data/                     Small sample of the released data
├── docs/reproducing_baselines.md    Step-by-step baseline instructions
├── environment.yaml                 Conda environment
└── requirements-nodeps.txt          Packages that install without dependencies
```

## Retrain the baselines

[docs/reproducing_baselines.md](docs/reproducing_baselines.md) lists all baselines, their subtasks and their submodules. It also explains how to put the released inputs in place.

Each baseline submodule has one script, `baselines/<name>/emulator_bench/run.sh`. For example:

```bash
baselines/KcatNet/emulator_bench/run.sh --dry-run   # print the commands
baselines/KcatNet/emulator_bench/run.sh             # build features, train, test, summarize
```

The reaction outcome models (`baselines/syntheseus`), the MLP and XGBoost references, BLASTP and Foldseek use their own commands. They are in `docs/reproducing_baselines.md`.

The release holds the leaderboard train and validation rows only. A full rerun of the paper tables needs the withheld test rows (see below). With the release alone, you can train on the released rows and get test scores from the leaderboard.

## Splits

[docs/splits/overview.md](docs/splits/overview.md) describes the split families, how to run `scripts/generate_splits.py` and how the thresholds were chosen. Each family has its own document in `docs/splits/`. The splitters are in `src/data/components/splitters/`.

## Leaderboard

Each subtask has a public leaderboard. The boards are in the Hugging Face collection [bisectgroup/leak-curber](https://huggingface.co/collections/bisectgroup/leak-curber). A board scores the most difficult leakage-controlled split of its subtask. The score is relative to a naive reference: 0 equals the reference and 1 is perfect. The paper (appendix on the leaderboard protocol) gives the full method.

Train only on the released train and validation rows. The test rows are hidden.

## What is not released

The code that builds the datasets from the source databases is not in this repository. The full datasets and the test sets are not released either. The builders and the deterministic splitters together would recreate the hidden test rows of the leaderboard.

The build steps are described in prose in `docs/` (one document for each source database and dataset). The embedders, generators, splitters, baseline code and configs are released.

## Citation

Please cite the paper. The file [CITATION.cff](CITATION.cff) has the entry. GitHub shows a "Cite this repository" button for it.

## License

- Code: MIT, see [LICENSE](LICENSE).
- Data: CC BY-NC-SA 4.0 (Dataverse and Hugging Face).
- Baselines: each submodule under `baselines/` keeps the license of its upstream project.
