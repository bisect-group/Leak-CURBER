# KcatNet Repo Audit and Retraining Notes

## Summary

This repo predicts **log10(kcat)** from an **enzyme amino-acid sequence** plus a **substrate SMILES**.

In the current codebase, the active model is a **protein-graph + molecule feature fusion regressor**:

- The protein side uses **ProtT5 per-residue embeddings** plus **ESM2 per-residue embeddings**.
- The protein graph is built from the **ESM2 predicted contact map**.
- The molecule side uses:
  - a learned embedding of per-atom categorical IDs,
  - a 43-d handcrafted per-atom feature vector,
  - a 1024-d global SMILES Transformer embedding.
- The final output is a single scalar in **log10(kcat)** space.

The released checkpoint in `RESULT/model_KcatNet.pt` has about **6.05M parameters**.

## Dataset and Input Schema

The provided splits are:

- `Dataset/KcatNet_traindf.pkl`: 9407 rows
- `Dataset/KcatNet_validdf.pkl`: 1175 rows
- `Dataset/KcatNet_testdf.pkl`: 1175 rows

Each split contains these columns:

- `Pro_seq`: protein sequence
- `Smile`: substrate SMILES
- `label`: regression target
- `ECNumber`: metadata
- `Uni_ID`: metadata

Only these columns are consumed by the active training path:

- `Pro_seq`
- `Smile`
- `label`

`label` is already in **log10(kcat)** space, not raw kcat. This is clear from:

- the label range in the bundled data, which includes negative values, and
- inference converting predictions back via `10 ** prediction` in `pred_kcat.py`.

If your CSV contains raw kcat values, you should convert them to `log10(kcat)` before training if you want the retrained model to be comparable to the original setup.

## Embedding Pipeline

### Protein pipeline

Source files:

- `utils/protein_init.py`
- `utils/Kcat_Dataset.py`

Effective behavior:

- Each unique protein sequence is embedded once and cached in a dictionary.
- Sequences longer than 1000 residues are truncated to the **first 1000 aa**.
- ProtT5 and ESM2 are both used at the **per-residue** level.

ProtT5 branch:

- Loaded in `get_T5_model()`.
- The code uses a **hardcoded local model path**: `/home/pantong/Code/ProtT5/`.
- The tokenizer/model are not loaded from a public HF name in the current repo version.
- `get_embeddings()` returns a per-residue embedding of size **1024**.
- These embeddings are stored as `token_representation`.

ESM2 branch:

- Loaded via `esm2_t33_650M_UR50D`.
- `esm_extract(..., layer=33, approach='last', dim=1280)` uses the **last layer** representation.
- This produces a per-residue embedding of size **1280**.
- These embeddings are stored as `token_representation_esm`.

Protein graph construction:

- The ESM2 predicted contact probability map is thresholded at **0.5**.
- Extra sequence edges are added for residue offsets:
  - `i <-> i+1`
  - `i <-> i+2`
- Self loops are then added.
- Edge weights are the contact probabilities, with the injected sequence edges filled with `0.5`, and self loops filled with `1`.

Important note:

- `config_KcatNet.json` names are semantically swapped:
  - `prot_in_channels = 1280` is actually the **ESM2** input size.
  - `prot_evo_channels = 1024` is actually the **ProtT5** input size.

### Molecule pipeline

Source files:

- `utils/ligand_init.py`
- `utils/pretrain_trfm.py`
- `utils/Kcat_Dataset.py`

The repo creates several molecule objects, but the active forward path only uses:

- `atom_idx`
- `atom_feature`
- `total_fea`

#### 1. Atom categorical ID

Each atom gets a coarse categorical ID derived from RDKit atom class grouping:

- `B`
- `C`
- `N`
- `O`
- `P`
- `S`
- `Se`
- `halogen`
- `metal`
- `0` for unknown/unmatched

This is stored as `atom_idx` and fed into an embedding table.

#### 2. Atom handcrafted feature vector

Each atom also gets a **43-d** feature vector from `atom_features()`:

- atom degree: 11-way one-hot
- total hydrogens: 11-way one-hot
- implicit valence: 11-way one-hot
- hybridization: 6-way one-hot
- aromatic flag: 1 dim
- chirality/can-be-chiral flags: 3 dims

This is stored as `atom_feature`.

#### 3. Global SMILES embedding

The repo loads a pretrained SMILES Transformer:

- `TrfmSeq2seq(45, 256, 45, 4)`
- weights from `utils/trfm_12_23000.pkl`
- vocabulary from `utils/vocab.pkl`

Tokenization details:

- custom SMILES splitter in `split()`
- max token length is **220**
- overlong SMILES are truncated to **first 109 + last 109** tokens
- `SOS` and `EOS` are added

Encoding details:

- `TrfmSeq2seq.encode()` returns a **1024-d** vector
- it is the concatenation of:
  - mean-pooled encoder output
  - max-pooled encoder output
  - first-token output from last encoder layer
  - first-token output from the penultimate encoder layer

This 1024-d vector is stored as `total_fea`.

### Molecule structures that are computed but not actually used

The repo also computes:

- `bond_feature`
- junction-tree outputs:
  - `tree_edge_index`
  - `atom2clique_index`
  - `x_clique`

But these are **not consumed** by `EnzMolDataset` or by `KcatNet.forward()`.

So, in the committed code, the molecule branch is **not doing bond-based message passing**. It is using atom-wise features plus attention-style pooling plus a global SMILES embedding.

## What Actually Goes Into the Model

From `utils/Kcat_Dataset.py`, each sample passed to the model contains:

- molecule:
  - `mol_x`: atom class IDs
  - `mol_x_feat`: 43-d atom feature matrix
  - `mol_total_fea`: 1024-d global SMILES embedding
- protein:
  - `prot_node_prot5`: ProtT5 residue embeddings, 1024-d
  - `prot_node_esm`: ESM2 residue embeddings, 1280-d
  - `prot_edge_index`
  - `prot_edge_weight`
- target:
  - `reg_y`

Stored but unused in the active forward path:

- `prot_seq`
- `prot_node_pos`
- `prot_num_nodes`
- molecule bond/junction-tree features

## Model Architecture

Source files:

- `models/model_kcat.py`
- `models/layers.py`
- `models/Mol_pool.py`
- `models/protein_pool.py`
- `models/pna.py`

### High-level structure

The architecture has **3 repeated interaction blocks** (`total_layer = 3`).

Initial encoders:

- Protein:
  - ProtT5 `1024 -> 400`
  - ESM2 `1280 -> 400`
  - concatenate to `800`
  - project to hidden size `800 -> 200`
- Protein edge weights:
  - scalar contact weights are expanded with an RBF embedding to **200 dims**
- Molecule:
  - atom type embedding: `20 -> 200`
  - atom handcrafted feature projection: `43 -> 200`
  - sum those two to get atom hidden states
  - global SMILES embedding: `1024 -> 200 -> 200`

Per interaction block:

1. Apply `GraphNorm` to protein residue states and atom states.
2. Run a **protein PNAConv** block on the residue graph.
3. Max-pool residue states to get one graph-level protein feature.
4. Predict soft cluster assignments with a small GCN and apply **dense mincut pooling**.
5. Update atom states and pool atoms with `MotifPool` to get one graph-level molecule feature.
6. Concatenate pooled molecule feature with the 1024-d SMILES embedding branch and project back to hidden size.
7. Run `InterConv` between:
   - one molecule graph-level node
   - the set of pooled protein clusters
8. Use the interaction attention to:
   - pool cluster features
   - send cluster information back to residues
   - compute residue attention scores
   - pool a second graph-level protein feature

Final fusion:

- concatenate across all 3 blocks:
  - pooled molecule features
  - residue max-pooled features
  - residue attention-pooled features
  - cluster pooled features
- compress each stack back to 200 dims
- concatenate to a final **800-d** fused vector
- MLP head:
  - `800 -> 512 -> 128 -> 1`

### Final architecture config in this repo

From `config_KcatNet.json` and the active forward path:

- `mol_in_channels = 43`
- `prot_in_channels = 1280` for ESM2
- `prot_evo_channels = 1024` for ProtT5
- `hidden_channels = 200`
- `aggregators = [mean, min, max, std]`
- `scalers = [identity, amplification, linear]`
- `pre_layers = 2`
- `post_layers = 1`
- `total_layer = 3`
- `K = [3, 10, 30]`
- `heads = 5`
- `dropout = 0`
- `dropout_attn_score = 0.2`

Active classifier head:

- `Linear(800, 512)`
- `Linear(512, 128)`
- `Linear(128, 1)`

Released checkpoint size:

- about **6,053,679** parameters

## Training Objective, Metrics, and Actual Optimization Behavior

Source files:

- `train.py`
- `utils/trainer.py`
- `utils/metrics.py`
- `models/model_kcat.py`

### Loss

Training uses:

- `0.9 * MSE(regression)`
- `0.1 * mincut clustering loss`

The target and prediction are both in **log10(kcat)** space during training and evaluation.

### Evaluation metrics

Validation/test metrics are computed on the log-scale predictions:

- RMSE
- Pearson
- Spearman
- R2
- MAE

### Actual training settings

CLI defaults in `train.py`:

- seed: `666`
- epochs: `80`
- batch size: `16`

### Important code-path caveat: optimizer config is not what it looks like

There are two separate sources of optimizer behavior:

1. `config_KcatNet.json`
2. `models/model_kcat.py`, where the optimizer is created inside the model

What actually happens in the current repo:

- The optimizer object is created inside the model as:
  - `AdamW(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)`
- `weight_decay` is **not set there**, so PyTorch AdamW default applies.
- In current PyTorch, that default is typically **0.01**, not the `0.0001` shown in `config_KcatNet.json`.

The trainer then does this:

- it reuses `self.model.optimizer`
- it does **not** rebuild the optimizer from config
- it does **not** pass `schedule_lr=config['optimizer']['schedule_lr']`

So the effective behavior is:

- LR scheduling is actually **enabled**, because `Trainer(... schedule_lr=True)` is the default.
- LR follows cosine decay from:
  - start: `1e-4`
  - end: `0`
- even though `config_KcatNet.json` says `"schedule_lr": false`

Practical conclusion:

- If you want strict code-level comparability to the released training path, keep this behavior as-is.
- If you "fix" the optimizer/config mismatch, your retrain is cleaner, but it is no longer a strict reproduction of this repo's training code.

## Other Repo Caveats That Matter for Retraining

### Checkpoint naming mismatch

`train.py` saves:

- `RESULT/model.pt`

`pred_kcat.py` loads:

- `RESULT/model_KcatNet.pt`

So the shipped checkpoint naming does not match the current training script.

### ProtT5 path is hardcoded

`utils/protein_init.py` assumes ProtT5 lives at:

- `/home/pantong/Code/ProtT5/`

You will need to patch this for your environment before retraining, unless you mirror that path.

### Some modules/features are defined but unused

Unused or inactive in the current forward path:

- molecule bond features
- molecule junction-tree features
- `atom_type_encoder`
- `seq_embed_evo2`
- `atom_type_embed2`
- `atom_feat_embed2`
- protein positional features

For comparability, follow the **active forward path**, not the unused code.

## What You Should Keep Fixed for a Comparable Retrain

If your goal is "same model, new TVT split", keep these fixed:

- Target definition:
  - train on `log10(kcat)`, not raw kcat
- Input columns:
  - `Pro_seq`, `Smile`, `label`
- Protein preprocessing:
  - same truncation to first 1000 residues
  - same ESM2 model: `esm2_t33_650M_UR50D`
  - same ProtT5 model family and embedding extraction
- Protein graph construction:
  - contact threshold `0.5`
  - add `i+-1` and `i+-2` sequence edges
  - add self loops
- Molecule preprocessing:
  - same RDKit atom feature definition
  - same SMILES tokenization
  - same vocab file
  - same pretrained SMILES Transformer weights
  - same 220-token truncation rule
- Architecture:
  - hidden size `200`
  - 3 interaction blocks
  - heads `5`
  - PNA aggregators/scalers unchanged
  - cluster counts `K = [3, 10, 30]`
  - same classifier head
- Objective:
  - `0.9 * MSE + 0.1 * mincut loss`
- Validation model selection:
  - select checkpoint by best validation RMSE

Also keep fixed if you want the closest possible training reproduction:

- epochs `80`
- batch size `16`
- gradient clip `1`
- cosine LR schedule from `1e-4` to `0`
- AdamW optimizer behavior from the current code path

## What Should Be Recomputed on Your New Split

These are not "tunable" hyperparameters. They are split-dependent statistics and should be recomputed from your new train split:

- `protein_dict`
- `ligand_dict`
- `degree.pt` / `prot_deg` for PNA degree scaling

Do **not** reuse the old `degree.pt` from the bundled split if you want a proper retrain on your own TVT partition.

## What You Can Tune While Still Staying Reasonably Comparable

If by "comparable" you mean same model family, not bit-for-bit reproduction, these are acceptable to tune and still call it a KcatNet-style retrain:

- random seed
- number of epochs
- early stopping patience
- learning rate
- minimum learning rate
- weight decay
- batch size

If you tune these, report them explicitly, because the current repo already has optimizer/config inconsistencies.

## What You Should Not Tune If You Want Strong Comparability

Avoid changing these unless you are willing to call the retrain a modified model:

- pretrained protein encoders
- pretrained SMILES encoder
- sequence truncation rules
- SMILES truncation rules
- protein graph construction rule
- hidden size
- number of blocks
- number of heads
- PNA aggregators/scalers
- cluster sizes `K`
- loss weighting between regression and clustering
- label transform

## Recommended Practical Strategy for Your CSV TVT Splits

If the goal is a fair comparison against the original repo:

1. Keep the active architecture and representation pipeline exactly the same.
2. Convert your CSV splits to the same schema:
   - `Pro_seq`
   - `Smile`
   - `label`
3. Make sure `label = log10(kcat)`.
4. Recompute:
   - protein embeddings
   - ligand embeddings
   - PNA degree histogram
5. Retrain with:
   - seed `666`
   - batch size `16`
   - epochs `80`
   - same loss
   - same effective optimizer behavior
6. Report metrics in log-space the same way the repo does.

If you want a slightly better-tuned retrain while remaining broadly comparable:

- keep the architecture and feature pipeline fixed
- tune only:
  - learning rate
  - weight decay
  - epochs / early stopping
  - seed

## Bottom line

This repo's effective model is:

- **protein graph from ESM2 contacts**
- **ProtT5 + ESM2 residue features**
- **atom features + atom type embedding + global SMILES Transformer embedding**
- **3 rounds of protein clustering and protein-molecule interaction**
- **final regression in log10(kcat) space**

For the most defensible comparison to the original results, keep the representation pipeline and architecture fixed, recompute split-dependent artifacts on your own train split, and only tune optimization settings lightly.
