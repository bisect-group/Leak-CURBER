# Sample Data

This directory is a reproducible sample of the Leak-CURBER benchmark datasets
bundled with the anonymous repository for smoke tests.

- Global seed: `20260507`
- Unsplit dataset parquets: `1000` rows each
- Split parquets: `train=900`, `val=50`, `test=50`
- Parquet compression: Brotli
- Provenance column: `sample_source_row_index`, the zero-based physical row number in the source parquet
- PNG, FASTA, and PKL companion files are copied as-is
- `*.7z*` archive shards are intentionally excluded
- `max_*similarities.tsv` files are sampled by matching sampled split keys when possible, otherwise by seeded TSV row sampling
- `embeddings/` contains deterministic smoke-test subsets of the full sharded embedding stores

## Bundled task directories

```
00-sample__benchmark_datasets/
├── kinetic_params_dataset/
│   ├── kcat/
│   ├── km/
│   └── ki/
├── binding_affinity_dataset/
│   ├── ec50/
│   ├── ic50/
│   └── kd/
├── enzyme_classification_dataset/
├── enzyme_retrieval_dataset/
├── reaction_outcome_dataset/
└── embeddings/
    ├── drfp/default/v1/
    ├── esm3/open_small_structure/v1/
    ├── esmc/esmc_600m/v1/
    ├── smited/materials_smi_ted_fork/v1/
    └── unimol_sdf/unimolv2/v1/
```

The sample embedding stores preserve the full-release store format:
`meta.json`, `index.parquet`, optional `failures.parquet`, and `shard_*.npy`.
They are capped at 1000 sample-matching embeddings per store and are intended
for format and pipeline smoke tests, not full sample coverage.

Full provenance is recorded in `SAMPLE_MANIFEST.json`. That manifest preserves
generation-time path names from the sampling job, including `00-sample_data`;
in this released checkout those files live under
`00-sample__benchmark_datasets/`.
