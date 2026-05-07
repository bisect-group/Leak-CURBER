# Sample Data

This directory is a reproducible sample of `01_core_benchmark`.

- Global seed: `20260507`
- Unsplit dataset parquets: `1000` rows each
- Split parquets: `train=900`, `val=50`, `test=50`
- Parquet compression: Brotli
- Provenance column: `sample_source_row_index`, the zero-based physical row number in the source parquet
- PNG, FASTA, and PKL companion files are copied as-is
- `*.7z*` archive shards are intentionally excluded
- `max_*similarities.tsv` files are sampled by matching sampled split keys when possible, otherwise by seeded TSV row sampling

Full provenance is recorded in `SAMPLE_MANIFEST.json`.
