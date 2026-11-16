# Dataset splits

`scripts/generate_splits.py` makes the train/val/test splits for the datasets. Each split family has its own splitter class and document.

## Split families

| Family (`--splits`) | Splitter | Document |
|---------------------|----------|----------|
| `random` | `RandomSplitter` | [random.md](random.md) |
| `smiles` | `SMILESMaxTanimotoSimilaritySplitter` | [smiles_tanimoto.md](smiles_tanimoto.md) |
| `sequence` | `ProteinSeqMaxFidentSimilaritySplitter` | [protein_seq_fident.md](protein_seq_fident.md) |
| `structure` | `ProteinStructMaxLDDTSimilaritySplitter` | [protein_lddt.md](protein_lddt.md) |
| `cosine` | `EmbeddingCosineSimilaritySplitter` | [embedding_cosine.md](embedding_cosine.md) |
| `reaction_tanimoto` | `ReactionDRFPTanimotoSimilaritySplitter` | [reaction_tanimoto.md](reaction_tanimoto.md) |
| `time` | `UniProtTimeBasedSplitter` | [uniprot_time.md](uniprot_time.md) |

The code is in `src/data/components/splitters/`. The shared helpers (`greedy_fill`, `order_by_similarity`, cache and save code) are in `base.py`.

## Usage

```bash
python scripts/generate_splits.py                                   # datasets in splits.datasets, all families
python scripts/generate_splits.py --datasets kinetic_params_kcat    # one dataset
python scripts/generate_splits.py --splits sequence structure       # some families
python scripts/generate_splits.py --skip-splits cosine time         # all except some families
python scripts/generate_splits.py --similarity-cache-overwrite      # ignore the similarity caches
python scripts/generate_splits.py --list-splits                     # print the family names
```

- `--datasets` takes names of the files in `configs/data/splits_dataset/`. Without it, the script uses the `datasets` list in `configs/data/splits.yaml`.
- For each dataset, the script runs the families in the order of the table above. It skips a family when the dataset file has no config for it.
- The split output goes to the folders set in the dataset file (`threshold_<t>/{train,val,test}.parquet` for similarity splits).
- Each split folder also holds a JSON file with the resolved config, the seed and the git commit.

## Configuration files

- `configs/data/splits.yaml`: seed, fractions, cache switch, tool paths, plot names and the dataset list.
- `configs/data/splits_dataset/*.yaml`: one file for each dataset. Each file sets the input and output paths and the thresholds of each family.
- `configs/data/split_inputs/*.yaml`: the dataset paths that the split files read. These files are always loaded. The builder yamls in `configs/data/` hold the other builder keys. `configs/data/builders.yaml` loads them if it exists.

## Rules

| Rule | How it is set |
|------|---------------|
| Seed and fractions come from `configs/data/splits.yaml`: `random_seed 42`, `train_frac 0.9`, `valid_frac 0.05`, `test_frac 0.05`. | Config. Only the random split uses the seed. The similarity splitters are deterministic. |
| In similarity splits, test rows are at most `test_frac * n_rows` and val rows are at most `valid_frac * n_rows`. | Code. `greedy_fill` skips a key that does not fit and tries the next. |
| Each similarity split uses one threshold: the toughest threshold that still gives about 5% test rows. | Config. Each dataset file lists the chosen threshold. The code does not search for it. The code only fills test up to `test_frac`. |
| Val stays in-distribution. Only test is below the threshold. | Code. Val takes the lowest-similarity keys that are left after test, with no threshold. |
| Unknown items always go to train: no structure, low-confidence predicted structure only, no Foldseek hit, no or all-zero embedding, tiny or inorganic molecule. | Code. See the document of each family. |
| Unified datasets (`kinetic_params_unified`, `binding_affinity_unified`, `reactions_unified`) are never split. | Code. `scripts/generate_splits.py` skips every dataset name that ends with `_unified`. |
| Classification and retrieval use full EC numbers only. | Config default, applied in the builders. `include_partial_ec_numbers: false`. See [../datasets/enzyme_retrieval_classification.md](../datasets/enzyme_retrieval_classification.md). |
| All rows of one group stay in one split. | Code. The group is the sequence, the molecule key, the reaction key or the structure. |
