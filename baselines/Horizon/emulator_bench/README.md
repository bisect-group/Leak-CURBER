# Horizon Leak-CURBER Adapter

This directory adapts Horizon to `data/processed/datasets/enzyme_retrieval_dataset`
without replacing Horizon's native model, loss, trainer, or evaluator.

## Dataset Mapping

Each split group must contain `train.parquet`, `val.parquet`, and `test.parquet`.
Nested groups such as `ec_hierarchy_splits/L1` are discovered automatically.
The adapter maps:

- `rxn_smiles` -> Horizon `reaction_smiles`
- `sequence` -> a protein target embedding key
- `ec_number` -> CARE Task 2 EC labels and metrics

Rows with missing EC labels are removed by default. Use
`--label-policy truncate` to keep the known EC prefix for partially missing
labels. Sequences are truncated to `--max-sequence-length 5000` by default,
matching the ProtT5 sequence cap used for Horizon's protein embeddings.
Duplicate reaction-protein pairs are collapsed within each split file and their
EC labels are merged. Horizon RDKit+ and DRFP fingerprint generation is also
validated during materialization; reactions that the native Horizon featurizer
cannot process are dropped and counted in `metadata.json`.

The dataset does not expose stable reaction or protein IDs, so the adapter uses
content hashes:

- `rxn_<sha256(rxn_smiles)>`
- `seq_<sha256(truncated_sequence)>`
- `pair_<sha256(reaction_id, protein_id)>`

## Artifacts

Generated files are written under `emulator_bench/runs/<split_slug>/`:

- `manifests/{train,val,test}.csv`
- `horizyn_data/{train,val,test}_pairs.csv`
- `horizyn_data/{train,val,test}_rxns.csv`
- `horizyn_data/prots_t5.h5`
- `metadata.json`

The shared protein vector cache lives under `emulator_bench/cache/proteins/`.
Real runs should use ProtT5 vectors through `--embedding-source prott5` or a
prepopulated cache via `--embedding-source cache-only`. `--protein-model-revision`
defaults to `refs/pr/1`, the Hugging Face safetensors conversion for
`Rostlab/prot_t5_xl_half_uniref50-enc`; this avoids the current
`transformers`/`torch<2.6` refusal to load `pytorch_model.bin` weights. The
deterministic embedding source is for smoke tests only and is marked explicitly
in metadata.

## Commands

Install adapter dependencies in the provided environment:

```bash
conda run -n horizon python -m pip install -e '.[emulator]'
```

Cache features for one split group:

```bash
conda run -n horizon python -m emulator_bench.cache_features \
  --dataset-root ../../data/processed/datasets/enzyme_retrieval_dataset \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --cache-root emulator_bench/cache \
  --embedding-source prott5 \
  --protein-model-revision refs/pr/1 \
  --label-policy remove \
  --max-sequence-length 5000
```

Train one seed with Horizon's native trainer:

```bash
conda run -n horizon python -m emulator_bench.train \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --seed 42 \
  --precision bf16
```

Evaluate the produced checkpoint:

```bash
conda run -n horizon python -m emulator_bench.evaluate \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --seed 42 \
  --eval-split test
```

Converted EC retrieval ranks EC labels by aggregating scores over proteins from
the train split only by default. This keeps test EC annotations out of the EC
candidate mapping while reusing the trained Horizon reaction/protein encoders.
Use `--reuse-native-metrics` to recompute only the converted EC ranking when
`native_metrics.json` already exists.

CARE-style EC retrieval uses EC centroids instead of max-over-protein scores,
cosine similarity, the forward reaction direction, and all split manifests as
EC references, matching CARE Task 2's EC-prototype ranking more closely:

```bash
conda run -n horizon python -m emulator_bench.evaluate \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --seed 42 \
  --eval-split test \
  --reuse-native-metrics \
  --results-tag care_style \
  --ec-scoring centroid \
  --score-similarity cosine \
  --direction-aggregation forward \
  --max-rank-columns 50 \
  --ec-candidate-split train \
  --ec-candidate-split val \
  --ec-candidate-split test
```

Aggregate all seed metrics under a runs root:

```bash
conda run -n horizon python -m emulator_bench.aggregate_results \
  --runs-root emulator_bench/runs
```

Queue cache, train, and evaluate with task-spooler:

```bash
conda run -n horizon python -m emulator_bench.queue_pipeline \
  --dataset-root ../../data/processed/datasets/enzyme_retrieval_dataset \
  --split-group random_splits_grouped_sequence \
  --env-name horizon \
  --precision bf16 \
  --seed 42 --seed 43 --seed 44
```

Reruns are incremental. The cache stage is still submitted so missing protein
vectors are generated, but existing protein `.npy` cache entries are validated
and reused. Queue submission skips a seed's train job when its `train_seed.json`
points to an existing checkpoint; evaluation then depends on the refreshed cache
job instead of a retrain job.

Smoke-test mode keeps the workflow identical but limits rows and uses explicit
non-benchmark embeddings:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n horizon python -m emulator_bench.cache_features \
  --dataset-root ../../data/processed/datasets/enzyme_retrieval_dataset \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --cache-root emulator_bench/cache \
  --embedding-source deterministic \
  --allow-deterministic-embeddings \
  --limit-per-split 8

CUDA_VISIBLE_DEVICES=3 conda run -n horizon python -m emulator_bench.train \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --seed 42 \
  --epochs 1 \
  --precision bf16

CUDA_VISIBLE_DEVICES=3 conda run -n horizon python -m emulator_bench.evaluate \
  --split-group random_splits_grouped_sequence \
  --runs-root emulator_bench/runs \
  --seed 42 \
  --eval-split test
```

## Metrics

Primary metrics come from Horizon's native evaluator:
Top-1, Top-10, Top-100, Top-1000 hit rate, R-precision, and average precision.

Supplemental benchmark artifacts are written per seed and split:

- `results/<split>/care_task2_ranked.csv`
- `results/<split>/native_metrics.json`
- `results/<split>/metrics.json`

Tagged runs write the same files under `results/<split>__<tag>/`.

The aggregate command writes `aggregated_seed_metrics_long.csv` and
`aggregated_seed_metrics_summary.csv` in the runs root by default, grouping
summary rows by `ec_candidate_splits`, EC scoring mode, score similarity, and
direction aggregation so train-only converted metrics are not averaged with
CARE-style, all-split, or smoke-test outputs.

The CARE Task 2 CSV keeps reaction metadata and appends numeric rank columns
`0,1,2,...` containing EC numbers. CARE Task 2 metrics include EC-prefix
accuracy at levels 1 through 4 for k in `[1,3,5,10,20,30,40,50]`. Additional
supplemental exact-EC metrics include MRR, MAP, and hit rates. The
`ec_candidate_splits` field records which manifest splits were used to build
the protein-to-EC candidate map. Use `--max-rank-columns 50` when only the
implemented CARE k-values are needed; omit it to emit a full all-EC ranking.

Horizon's native Lightning trainer is sufficient; no PyTorch Lightning fallback
was added beyond the baseline's existing trainer.
