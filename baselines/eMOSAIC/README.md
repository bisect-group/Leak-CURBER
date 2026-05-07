## Overview

This repository contains the code for training and testing our proposed eMOSAIC model for protein–ligand binding affinity prediction and uncertainty quantification.

*TrustAffinity* represents the baseline binding affinity prediction model, while *eMOSAIC* denotes the application of the uncertainty quantification framework ("eMOSAIC") to TrustAffinity.

## Repository Structure
- `data/` – Datasets, data splits, input embeddings, and pretrained model weights.
- `code/BindingAffinityModule/` – Training and inference code of *TrustAffinityNet* for binding affinity prediction.  
- `code/AnomalyDetection/` – Training and inference code of *eMOSAIC* for uncertainty quantification.  
- `environment/` – Requirements for setting up the environment.  
- `results/` – The output files for both the binding affinity predictions and the uncertainty quantification tasks, including detailed log files, tabular summaries, and visual summaries (`results/figures/`).

## Usage

### 1. Train *TrustAffinityNet*

Run from `code/BindingAffinityModule/`:

```bash
python main.py
```

### 2. Train *eMOSAIC*

Run from `code/AnomalyDetection/`:

```bash
python main.py
```

### 3. Inference: Binding Affinity Prediction with Uncertainty Quantification

To predict binding affinity and quantify uncertainty for given protein–ligand pairs, run from `code/`:

```bash
python predict_pki_uncertainty.py \
  --smiles_list "Cc1cc(Oc2ccc(/C=C3\\SC(=O)N([C@@H](Cc4ccccc4)C(=O)O)C3=O)cc2)cc(C)c1Cl, \
                 Cc1cc(Oc2ccc(/C=C3\\SC(=O)N([C@@H](Cc4ccccc4)C(=O)O)C3=O)cc2)cc(C)c1Cl, \
                 COC(=O)c1cccc(COc2ccc3[nH]c(SCC(=O)c4ccc(O)c(O)c4)nc3c2)c1" \
  --uniprot_ids "Q07817, Q07820, P47871" \
  --data_split scaffold \
  --num_clusters 50 \
  --iters 10 \
  --scaling True \
  --seed 42 \
  --checkpoint_dir "/results/exp08-02-2025-05-02-20/"
```
To reproduce the binding affinity prediction and uncertainty quantification results reported in our manuscript, run from `code/`:

```bash
python reproducible_run.py
```

## Pretrained Models and Reproducibility

For full reproducibility, including access to pretrained models and complete input files, we provide a ready-to-run Code Ocean capsule:

https://codeocean.com/capsule/2486685/tree/v1

The capsule contains all pretrained checkpoints, processed datasets, and scripts required to reproduce the binding affinity prediction and uncertainty quantification results reported in our manuscript.

## Citation
If you find our model and code helpful in your work, please consider citing us:
```
@article{badkul2025multimodal,
  title={Multimodal out-of-distribution individual uncertainty quantification enhances binding affinity prediction for polypharmacology},
  author={Badkul, Amitesh and Xie, Li and Zhang, Shuo and Xie, Lei},
  journal={Nature Machine Intelligence},
  pages={1--11},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
