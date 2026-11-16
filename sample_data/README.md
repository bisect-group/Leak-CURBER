# Sample benchmark data

Each file here is a small random sample of one released Leak-CURBER file.
The rows come from the released leaderboard train and validation rows only.
No test rows are included.

- Source: Harvard Dataverse, doi:10.7910/DVN/CZW47K, folder `leaderboard_train_val/`.
  The same rows are on Hugging Face as `bisectgroup/Leak-CURBER-<task>`.
- Sample size: up to 64 rows per file, drawn with seed 42. Smaller files are copied whole.
- Columns are the released columns. Row order follows the source file.
- Format: Parquet, zstd compression.
- Layout: `Leak-CURBER-<task>/data/<path in the source>`.

Rebuild or verify with `release/make_samples.py` (`--check` tests that every sample row is in the source).
