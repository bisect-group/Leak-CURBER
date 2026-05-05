<div align="center">

# Bisect Template Python

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![pixi](https://img.shields.io/badge/pixi-enabled-blue)](https://pixi.sh/)
[![wandb](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=white)](https://wandb.ai/)
[![dvc](https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white)](https://dvc.org/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](#license)

A clean template to kickstart your deep learning project 🚀⚡🔥<br>
Built with PyTorch Lightning, Hydra, and Pixi for modern ML development.

_Suggestions are always welcome!_

</div>

## Main Technologies

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

[Pixi](https://pixi.sh/) - a modern, fast package manager built on top of conda that provides reproducible environments across platforms. Pixi uses conda-forge packages and supports PyPI packages, making dependency management simple and reliable.

[Weights & Biases](https://wandb.ai/) - experiment tracking and model management platform for machine learning. Track hyperparameters, system metrics, and model artifacts with ease.

[DVC](https://dvc.org/) - data version control system for ML projects. Version control large files and create reproducible data pipelines.

<br>

## Main Ideas

- [**Rapid Experimentation**](#your-superpowers): thanks to hydra command line superpowers
- [**Minimal Boilerplate**](#how-it-works): thanks to automating pipelines with config instantiation
- [**Main Configs**](#main-config): allow you to specify default training configuration
- [**Experiment Configs**](#experiment-config): allow you to override chosen hyperparameters and version control experiments
- [**Workflow**](#workflow): comes down to 4 simple steps
- [**Experiment Tracking**](#experiment-tracking): Tensorboard, W&B, Neptune, Comet, MLFlow and CSVLogger
- [**Logs**](#logs): all logs (checkpoints, configs, etc.) are stored in a dynamically generated folder structure
- [**Hyperparameter Search**](#hyperparameter-search): simple search is effortless with Hydra plugins like Optuna Sweeper
- [**Tests**](#tests): generic, easy-to-adapt smoke tests for speeding up the development
- [**Continuous Integration**](#continuous-integration): automatically test and lint your repo with Github Actions
- [**Best Practices**](#best-practices): a couple of recommended tools, practices and standards

<br>

## Project Structure

The directory structure of this template looks like this:

```
├── configs/                  <- Hydra configuration files
│   ├── callbacks/               <- Callbacks configs (early stopping, checkpointing, etc.)
│   ├── data/                    <- Data configs (datasets, dataloaders)
│   ├── data_collator/           <- Data collator configs (for sequence tasks)
│   ├── debug/                   <- Debugging configs (profiling, overfitting tests)
│   ├── experiment/              <- Experiment configs (complete training setups)
│   ├── extras/                  <- Extra utilities configs
│   ├── generation/              <- Generation configs (for language models)
│   ├── hparams_search/          <- Hyperparameter search configs (Optuna)
│   ├── hydra/                   <- Hydra framework configs
│   ├── local/                   <- Local machine-specific configs (excluded from git)
│   ├── logger/                  <- Logger configs (wandb, tensorboard, etc.)
│   ├── model/                   <- Model architecture configs
│   ├── optimizer/               <- Optimizer configs (Adam, SGD, etc.)
│   ├── paths/                   <- Project paths configs
│   ├── scheduler/               <- Learning rate scheduler configs
│   ├── tokenizer/               <- Tokenizer configs (for NLP tasks)
│   ├── trainer/                 <- Trainer configs (GPU, CPU, DDP)
│   ├── eval.yaml                <- Main config for evaluation
│   └── train.yaml               <- Main config for training
│
├── data/                     <- Project data (DVC tracked)
│   ├── raw/                     <- The original, immutable data dump
│   ├── intermediate/            <- Intermediate data that has been transformed
│   └── processed/               <- The final, canonical data sets for modeling
│
├── logs/                     <- Logs generated by hydra and lightning loggers
│
├── notebooks/                <- Jupyter notebooks for exploration and analysis
│                                Naming: number-initials-description.ipynb
│
├── scripts/                  <- Shell scripts for various tasks
│   └── schedule.sh              <- Job scheduling script
│
├── src/                      <- Source code
│   ├── data/                    <- Data loading and processing
│   │   ├── components/          <- Reusable data components
│   │   └── mnist_datamodule.py  <- Example MNIST datamodule
│   ├── models/                  <- Model architectures
│   │   ├── components/          <- Reusable model components
│   │   ├── mnist_module.py      <- Example MNIST Lightning module
│   │   └── simple_dense_net.py  <- Example neural network
│   ├── utils/                   <- Utility functions
│   │   ├── instantiators.py     <- Hydra instantiation utilities
│   │   ├── logging_utils.py     <- Logging utilities
│   │   ├── pylogger.py          <- Python logger setup
│   │   ├── rich_utils.py        <- Rich formatting utilities
│   │   └── utils.py             <- General utilities
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests/                    <- Test suite
│   ├── helpers/                 <- Test helper functions
│   ├── test_configs.py          <- Test configuration files
│   ├── test_datamodules.py      <- Test data modules
│   ├── test_eval.py             <- Test evaluation pipeline
│   ├── test_sweeps.py           <- Test hyperparameter sweeps
│   └── test_train.py            <- Test training pipeline
│
├── .env.example              <- Example environment variables file
├── .gitignore                <- Files to ignore in git
├── .pre-commit-config.yaml   <- Pre-commit hooks configuration
├── environment.yaml          <- Conda environment file
├── Makefile                  <- Makefile with useful commands
├── pixi.toml                 <- Pixi project configuration (recommended)
├── pyproject.toml            <- Python project configuration
├── setup.py                  <- Setup script for package installation
└── README.md                 <- This file
```

<br>

## 🚀  Quickstart

### Using Pixi (Recommended)

[Pixi](https://pixi.sh/) is a modern package manager that provides reproducible environments and cross-platform compatibility.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash
# Or use: curl -fsSL https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz | tar -xzC ~/.local/bin

# Clone project
git clone https://github.com/bisect-group/bisect_template_python
cd bisect_template_python

# Install dependencies with pixi (creates environment automatically)
pixi install

# Activate the environment
pixi shell

# Run training with default configuration
pixi run python src/train.py
```

**Important:** Update the CUDA wheels in `pixi.toml` according to your system:
- For CUDA 11.8: change `cu126` to `cu118` in the `extra-index-urls`
- For CUDA 12.1: change `cu126` to `cu121` in the `extra-index-urls`
- For CPU-only: remove the `extra-index-urls` and use CPU versions

### Verify Installation

Template contains an example with MNIST classification. When running `python src/train.py` you should see training progress with metrics being logged.

**Quick test run:**
```bash
# Quick training run (1 epoch, debug mode)
python src/train.py debug=default trainer.max_epochs=1
```

## ⚡  Your Superpowers

<details>
<summary><b>Override any config parameter from command line</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python train.py trainer=mps
```
</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>

<!-- deepspeed support still in beta
<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>

```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like W&B or Tensorboard</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

> **Note**: Lightning provides convenient integrations with most popular logging frameworks. Learn more [here](#experiment-tracking).

> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.

> **Note**: Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.

</details>

<details>
<summary><b>Train model with chosen experiment config</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Experiment configs are placed in [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Evaluate checkpoint on test dataset</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Note**: Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) doesn't require you to add any boilerplate to your code, everything is defined in a [single config file](configs/hparams_search/mnist_optuna.yaml).

> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> **Note**: This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not implemented in this template.

</details>

<!-- <details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details> -->

<details>
<summary><b>Use Hydra tab completion</b></summary>

> **Note**: Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Read the [docs](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

```bash
pre-commit run -a
```

> **Note**: Apply pre-commit hooks to do things like auto-formatting code and configs, performing code analysis or removing output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

Update pre-commit hook versions in `.pre-commit-config.yaml` with:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Run tests</b></summary>

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

</details>

<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["mnist","experiment_X"]
```

> **Note**: You might need to escape the bracket characters in your shell with `python train.py tags=\["mnist","experiment_X"\]`.

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>

## 📊 Experiment Tracking with Weights & Biases

We **strongly recommend** using [Weights & Biases](https://wandb.ai/) for experiment tracking, as it provides:

- **Real-time metrics visualization** during training
- **Hyperparameter tracking** and comparison across runs
- **Model artifact storage** and versioning
- **Collaborative experiment sharing** with your team
- **Advanced features** like sweeps, reports, and model registry

### Setup W&B

1. **Create a W&B account** at [wandb.ai](https://wandb.ai/)
2. **Install and login:**
   ```bash
   # Already included in pixi.toml
   wandb login
   ```
3. **Configure your project** in `configs/logger/wandb.yaml`:
   ```yaml
   wandb:
     project: "your_project_name"
     entity: "your_wandb_team_name"  # optional
   ```
4. **Run with W&B logging:**
   ```bash
   python src/train.py logger=wandb
   ```

Example W&B dashboard features you'll get:
- Loss curves and metrics over time
- System metrics (GPU utilization, memory usage)
- Hyperparameter importance analysis
- Model comparison tables
- Interactive plots and custom visualizations

## 📁 Data Version Control with DVC

For data versioning and pipeline management, we recommend using [DVC](https://dvc.org/):

### Setup DVC

```bash
# Initialize DVC in your project (already included in pixi.toml)
dvc init

# Add your data directory to DVC tracking
dvc add data/raw/your_dataset
dvc add data/processed/

# Commit the .dvc files to git
git add data/raw/your_dataset.dvc data/processed.dvc .gitignore
git commit -m "Add data to DVC tracking"

# Set up remote storage (choose one):
# AWS S3
dvc remote add myremote s3://my-bucket/dvc-storage
# Google Cloud Storage  
dvc remote add myremote gs://my-bucket/dvc-storage
# Azure Blob Storage
dvc remote add myremote azure://my-container/dvc-storage

# Push data to remote storage
dvc push
```

### DVC Benefits

- **Version control large files** without storing them in git
- **Reproducible data pipelines** with `dvc.yaml`
- **Collaborate on datasets** with team members
- **Track data lineage** and transformations
- **Efficient storage** with deduplication and compression

### Recommended Data Organization

```
data/
├── raw/                    <- Original, immutable data (DVC tracked)
│   ├── dataset_v1/
│   └── external_source/
├── intermediate/           <- Intermediate data (DVC tracked)  
│   ├── cleaned/
│   └── features/
└── processed/              <- Final datasets for modeling (DVC tracked)
    ├── train/
    ├── val/
    └── test/
```

<br>

## How It Works

All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: src.models.mnist_model.MNISTLitModule
lr: 0.001
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  lin1_size: 256
  lin2_size: 256
  lin3_size: 256
  output_size: 10
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models! Every time you create a new one, just specify its module path and parameters in appropriate config file. <br>

Switch between models and datamodules with command line arguments:

```bash
python train.py model=mnist
```

Example pipeline managing the instantiation logic: [src/train.py](src/train.py).

<br>

## Main Config

Location: [configs/train.yaml](configs/train.yaml) <br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python train.py`.<br>

<details>
<summary><b>Show main project config</b></summary>

```yaml
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - data: mnist.yaml
  - model: mnist.yaml
  - callbacks: default.yaml
  - logger: null # set logger here or use command line (e.g. `python train.py logger=csv`)
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml

  # experiment configs allow for version control of specific hyperparameters
  # e.g. best hyperparameters for given model and datamodule
  - experiment: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

# task name, determines output directory path
task_name: "train"

# tags to help you identify your experiments
# you can overwrite this in experiment configs
# overwrite from command line with `python train.py tags="[first_tag, second_tag]"`
# appending lists from command line is currently not supported :(
# https://github.com/facebookresearch/hydra/issues/1547
tags: ["dev"]

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# simply provide checkpoint path to resume training
ckpt_path: null

# seed for random number generators in pytorch, numpy and python.random
seed: null
```

</details>

<br>

## Experiment Config

Location: [configs/experiment](configs/experiment)<br>
Experiment configs allow you to overwrite parameters from main config.<br>
For example, you can use them to version control best hyperparameters for each combination of model and dataset.

<details>
<summary><b>Show example experiment config</b></summary>

```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /data: mnist.yaml
  - override /model: mnist.yaml
  - override /callbacks: default.yaml
  - override /trainer: default.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["mnist", "simple_dense_net"]

seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

data:
  batch_size: 64

logger:
  wandb:
    tags: ${tags}
    group: "mnist"
```

</details>

<br>

## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [data/mnist_datamodule.py](src/data/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:
   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```

**Experiment design**

_Say you want to execute many runs to plot how accuracy changes in respect to batch size._

1. Execute the runs with some config parameter that allows you to identify them easily, like tags:

   ```bash
   python train.py -m logger=csv data.batch_size=16,32,64,128 tags=["batch_size_exp"]
   ```

2. Write a script or notebook that searches over the `logs/` folder and retrieves csv logs from runs containing given tags in config. Plot the results.

<br>

## Logs

Hydra creates new output directory for every executed run.

Default logging structure:

```
├── logs
│   ├── task_name
│   │   ├── runs                        # Logs generated by single runs
│   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   ├── csv                     # Csv logs
│   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   └── ...
│   │   │
│   │   └── multiruns                   # Logs generated by multiruns
│   │       ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │       │   ├──1                        # Multirun job number
│   │       │   ├──2
│   │       │   └── ...
│   │       └── ...
│   │
│   └── debugs                          # Logs generated when debugging config is attached
│       └── ...
```

</details>

You can change this structure by modifying paths in [hydra configuration](configs/hydra).

<br>

## Experiment Tracking

PyTorch Lightning supports many popular logging frameworks: [Weights&Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger.

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the [docs](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_module.py).

<br>

## 🧪 Tests

This template comes with a comprehensive test suite implemented with `pytest` to ensure code quality and prevent regressions.

### Running Tests

```bash
# Run all tests
pytest

# Run all tests with coverage report
pytest --cov=src --cov-report=html

# Run tests from specific file
pytest tests/test_train.py

# Run all tests except the ones marked as slow
pytest -k "not slow"

# Run tests with verbose output
pytest -v

# Run tests in parallel (install pytest-xdist first)
pytest -n auto
```

### Using Pixi for Tests

```bash
# Run tests in pixi environment
pixi run pytest

# Run with coverage
pixi run pytest --cov=src --cov-report=html

# Run specific test categories
pixi run pytest -m "not slow"
```

### Test Categories

The test suite covers multiple aspects of your ML pipeline:

**Core Functionality Tests:**
- `test_train.py` - Training pipeline validation
- `test_eval.py` - Evaluation pipeline validation  
- `test_datamodules.py` - Data loading and processing
- `test_configs.py` - Configuration file validation
- `test_sweeps.py` - Hyperparameter sweep functionality

**Test Scenarios Include:**
- ✅ **Smoke tests** - Run 1 train/val/test step without errors
- ✅ **Integration tests** - Full pipeline with checkpointing and resuming
- ✅ **Configuration tests** - All config combinations are valid
- ✅ **Distributed training** - DDP simulation on CPU
- ✅ **Memory tests** - Training on small data batches
- ✅ **Speed tests** - Fast training runs for CI/CD

### Conditional Testing with @RunIf

The template includes a `@RunIf` decorator for conditional test execution:

```python
@RunIf(min_gpus=1)
def test_gpu_training():
    """Only runs if GPU is available."""
    pass

@RunIf(min_torch_version="1.12")
def test_new_torch_feature():
    """Only runs with newer PyTorch versions."""
    pass
```

### Adding Your Own Tests

1. **Create test files** in the `tests/` directory following the `test_*.py` naming convention
2. **Use fixtures** from `conftest.py` for common setup
3. **Test your models** with small datasets for fast execution
4. **Mock external dependencies** (APIs, databases) for reliability
5. **Add markers** for categorizing tests (slow, integration, unit)

### Continuous Integration

Tests automatically run on:
- **Every push** to main branch
- **Pull requests** with modified files only
- **Scheduled runs** to catch dependency issues

**Pro tip:** Run tests locally before pushing to catch issues early!

```bash
# Quick test run before committing
pytest -x --ff  # Stop on first failure, run failed tests first
```

<br>

## Hyperparameter Search

You can define hyperparameter search by adding new config file to [configs/hparams_search](configs/hparams_search).

<details>
<summary><b>Show example hyperparameter search config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
optimized_metric: "val/acc_best"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      data.batch_size: choice(32, 64, 128, 256)
      model.net.lin1_size: choice(64, 128, 256)
      model.net.lin2_size: choice(64, 128, 256)
      model.net.lin3_size: choice(32, 64, 128, 256)
```

</details>

Next, execute it with: `python train.py -m hparams_search=mnist_optuna`

Using this approach doesn't require adding any boilerplate to code, everything is defined in a single config file. The only necessary thing is to return the optimized metric value from the launch file.

You can use different optimization frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/).

The `optimization_results.yaml` will be available under `logs/task_name/multirun` folder.

This approach doesn't support resuming interrupted search and advanced techniques like prunning - for more sophisticated search and workflows, you should probably write a dedicated optimization task (without multirun feature).

<br>

## Continuous Integration

Template comes with CI workflows implemented in Github Actions:

- `.github/workflows/test.yaml`: running all tests with pytest
- `.github/workflows/code-quality-main.yaml`: running pre-commits on main branch for all files
- `.github/workflows/code-quality-pr.yaml`: running pre-commits on pull requests for modified files only

<br>

## Distributed Training

Lightning supports multiple ways of doing distributed training. The most common one is DDP, which spawns separate process for each GPU and averages gradients between them. To learn about other approaches read the [lightning docs](https://lightning.ai/docs/pytorch/latest/advanced/speed.html).

You can run DDP on mnist example with 4 GPUs like this:

```bash
python train.py trainer=ddp
```

> **Note**: When using DDP you have to be careful how you write your models - read the [docs](https://lightning.ai/docs/pytorch/latest/advanced/speed.html).

<br>

## Accessing Datamodule Attributes In Model

The simplest way is to pass datamodule attribute directly to model on initialization:

```python
# ./src/train.py
datamodule = hydra.utils.instantiate(config.data)
model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
```

> **Note**: Not a very robust solution, since it assumes all your datamodules have `some_param` attribute available.

Similarly, you can pass a whole datamodule config as an init parameter:

```python
# ./src/train.py
model = hydra.utils.instantiate(config.model, dm_conf=config.data, _recursive_=False)
```

You can also pass a datamodule config parameter to your model through variable interpolation:

```yaml
# ./configs/model/my_model.yaml
_target_: src.models.my_module.MyLitModule
lr: 0.01
some_param: ${data.some_param}
```

Another approach is to access datamodule in LightningModule directly through Trainer:

```python
# ./src/models/mnist_module.py
def on_train_start(self):
  self.some_param = self.trainer.datamodule.some_param
```

> **Note**: This only works after the training starts since otherwise trainer won't be yet available in LightningModule.

<br>

## Best Practices

<details>
<summary><b>Use Miniconda</b></summary>

It's usually unnecessary to install full anaconda environment, miniconda should be enough (weights around 80MB).

Big advantage of conda is that it allows for installing packages without requiring certain compilers or libraries to be available in the system (since it installs precompiled binaries), so it often makes it easier to install some dependencies e.g. cudatoolkit for GPU support.

It also allows you to access your environments globally which might be more convenient than creating new local environment for every project.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Update conda:

```bash
conda update -n base -c defaults conda
```

Create new conda environment:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:

```bash
pip install pre-commit
```

Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
pre-commit install
```

After that your code will be automatically reformatted on every new commit.

To reformat all files in the project use command:

```bash
pre-commit run -a
```

To update hook versions in [.pre-commit-config.yaml](.pre-commit-config.yaml) use:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.<br>

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `train.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy


class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).<br>

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn’t have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```

2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def on_train_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def on_validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def on_test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Version control your data and models with DVC</b></summary>

Use [DVC](https://dvc.org) to version control big files, like your data or trained ML models.<br>
To initialize the dvc repository:

```bash
dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
dvc add data/MNIST
```

DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:

```bash
git add data/MNIST.dvc data/.gitignore
git commit -m "Add raw data"
```

</details>

<details>
<summary><b>Support installing project as a package</b></summary>

It allows other people to easily use your modules in their own projects.
Change name of the `src` folder to your project name and complete the `setup.py` file.

Now your project can be installed from local files:

```bash
pip install -e .
```

Or directly from git repository:

```bash
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```

So any file can be easily imported into any other file like so:

```python
from project_name.models.mnist_module import MNISTLitModule
from project_name.data.mnist_datamodule import MNISTDataModule
```

</details>

<details>
<summary><b>Keep local configs out of code versioning</b></summary>

Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file [configs/local/default.yaml](configs/local/) can be created which is automatically loaded but not tracked by Git.

For example, you can use it for a SLURM cluster config:

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

<br>

## Resources

This template was inspired by:

- [PyTorchLightning/deep-learning-project-template](https://github.com/PyTorchLightning/deep-learning-project-template)
- [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
- [lucmos/nn-template](https://github.com/lucmos/nn-template)

Other useful repositories:

- [jxpress/lightning-hydra-template-vertex-ai](https://github.com/jxpress/lightning-hydra-template-vertex-ai) - lightning-hydra-template integration with Vertex AI hyperparameter tuning and custom training job

</details>

<br>

## 🔬 Academic Lab Workflow

### Pull Request Guidelines

This template is designed for collaborative academic research. Follow these guidelines when contributing:

#### Before Creating a Pull Request

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/experiment-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Update your environment** to ensure reproducibility:
   ```bash
   pixi install  # Update dependencies
   pixi run pre-commit install  # Install pre-commit hooks
   ```

3. **Run tests locally** before pushing:
   ```bash
   pixi run pytest  # Run all tests
   pixi run pre-commit run --all-files  # Check code formatting
   ```

#### Pull Request Templates

**Use the existing PR template** located at `.github/PULL_REQUEST_TEMPLATE.md` when creating pull requests. This template will automatically populate when you create a new PR.

For academic-specific contributions, we provide specialized templates based on your contribution type:

**📊 New Experiment:** `.github/PULL_REQUEST_TEMPLATE/experiment.md`
- Use when adding new experiment configurations
- Includes sections for research questions, methodology, and results
- Focuses on reproducibility and academic rigor

**🧠 Model Architecture:** `.github/PULL_REQUEST_TEMPLATE/model.md`  
- Use when proposing new models or architectural changes
- Includes performance analysis and ablation studies
- Emphasizes theoretical justification and baselines

**📁 Dataset/Data Processing:** `.github/PULL_REQUEST_TEMPLATE/dataset.md`
- Use when adding new datasets or data processing pipelines
- Includes ethical considerations and DVC integration
- Focuses on data quality and documentation

**🐛 Bug Fix:** `.github/PULL_REQUEST_TEMPLATE/bugfix.md`
- Use when fixing bugs or issues
- Includes root cause analysis and testing verification
- Emphasizes regression prevention

**To use a specific template:**
1. Create your PR normally
2. Replace the default template content with the appropriate specialized template
3. Or manually copy from the template files in `.github/PULL_REQUEST_TEMPLATE/`

#### Academic-Specific Guidelines

**For New Experiments:**
- Create a new experiment config in `configs/experiment/`
- Document hyperparameters and rationale in the PR description
- Include preliminary results or motivation
- Tag relevant lab members for review

**For Model Changes:**
- Provide comparison with baseline models
- Document performance implications
- Include ablation studies if applicable
- Update model documentation

**For Data Changes:**
- Use DVC to version control datasets
- Document data preprocessing steps
- Ensure ethical considerations are addressed
- Update data documentation

#### Code Review Process

1. **Author responsibilities:**
   - Ensure code is well-documented and tested
   - Provide clear commit messages
   - Respond to reviewer feedback promptly

2. **Reviewer responsibilities:**
   - Focus on code quality, reproducibility, and scientific rigor
   - Check experimental methodology
   - Verify computational efficiency
   - Ensure adherence to lab standards

### Lab-Specific Setup

#### Setting Up for Collaboration

1. **Configure Git for the lab:**
   ```bash
   git config user.name "Your Name"
   git config user.email "your.email@university.edu"
   ```

2. **Set up W&B for the lab:**
   ```bash
   wandb login
   # Use lab entity in configs/logger/wandb.yaml
   ```

3. **Configure DVC remote** (ask lab admin for credentials):
   ```bash
   dvc remote add lab-storage s3://lab-bucket/project-name
   dvc remote default lab-storage
   ```

#### Lab Computing Resources

**For Multi-GPU Training:**
```bash
# Distributed training
python src/train.py trainer=ddp trainer.devices=4
```

#### Experiment Tracking Best Practices

1. **Use consistent naming conventions:**
   ```bash
   python src/train.py experiment=paper_baseline tags=["paper","baseline","v1.0"]
   ```

2. **Document hyperparameter choices:**
   - Link to papers or previous experiments
   - Explain deviation from standard practices
   - Include computational resource requirements

3. **Share results with the lab:**
   - Use W&B reports for presenting results
   - Tag experiments with paper/project names
   - Include reproduction instructions

#### Research Reproducibility

**Version Control Everything:**
- Code: Git
- Data: DVC
- Environment: Pixi/conda
- Experiments: W&B
- Papers: Git (LaTeX) or Overleaf

**Document Computational Requirements:**
```yaml
# Add to experiment configs
compute_requirements:
  min_gpu_memory: "8GB"
  estimated_runtime: "4 hours"
  recommended_gpus: 1
  cpu_cores: 8
```

**Create Reproducible Experiment Scripts:**
```bash
# Example: reproduce paper results
pixi run python src/train.py experiment=paper_main_results seed=42
```

### Troubleshooting Common Issues

**Environment Issues:**
```bash
# Reset environment
pixi install --force-reinstall

# Check CUDA compatibility
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

**Data Access Issues:**
```bash
# Sync data from DVC remote
dvc pull

# Check data integrity
dvc status
```

<br>
<br>
<br>

---

## 🎯 Quick Start Template

When starting your own project using this template, follow these steps:

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://pixi.sh/"><img alt="Pixi" src="https://img.shields.io/badge/pixi-enabled-blue"></a>
<a href="https://wandb.ai/"><img alt="W&B" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=white"></a>
<a href="https://dvc.org/"><img alt="DVC" src="https://img.shields.io/badge/DVC-13ADC7?logo=dvc&logoColor=white"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

Brief description of what your project does and its main contributions.

## Installation

Follow the installation instructions above, using Pixi (recommended) or alternative methods.

## How to run

Train model with default configuration:

```bash
# Using Pixi (recommended)
pixi run python src/train.py trainer=cpu    # CPU training
pixi run python src/train.py trainer=gpu    # GPU training

# Using direct Python
python src/train.py trainer=cpu
python src/train.py trainer=gpu
```

Train with specific experiment configuration:

```bash
pixi run python src/train.py experiment=experiment_name
```

Override parameters from command line:

```bash
pixi run python src/train.py trainer.max_epochs=20 data.batch_size=64 logger=wandb
```

## �📝 Citation

```bibtex
@article{your_paper_2024,
  title={Your Paper Title},
  author={Your Name and Collaborators},
  journal={Your Journal/Conference},
  year={2024}
}
```
