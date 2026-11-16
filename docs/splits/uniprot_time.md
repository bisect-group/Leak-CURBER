# UniProt Time Split

This document describes `UniProtTimeBasedSplitter` in `src/data/components/splitters/uniprot_time.py`.

## Overview

The splitter orders rows by the date of the UniProt accession. Old accessions go to train, newer ones to val, and the newest to test. There is no similarity threshold.

The dataset must have a date column. The builders write `uniprot_date` (the UniProt `date_created` field).

---

## Processing Pipeline

1. Read the input parquet file. The date column must exist.
2. Parse the dates and normalise them to days. Rows with no usable date follow `time_split_missing_date_split` (default `train`). The other values are `val`, `test`, `drop` and `error`.
3. Count the rows for each date, oldest first.
4. Assign each date, oldest first, to train until train has `int(train_frac * n_rows)` rows. Then assign to val until val has `int(valid_frac * n_rows)` rows. The remaining dates go to test. Rows with a missing date count toward the split they are assigned to.
5. Check that no split is empty and that train dates are not newer than val dates, and val dates are not newer than test dates.
6. Make groups so that one group stays in one split. A group that occurs in several splits moves with all its rows to the earliest of them. The group keys are:
   - `time_split.group_cols` if set. A `rxn_key` column uses `undirected_reaction_key`, so a reaction and its reverse share one group.
   - Else the pair (`sequence`, `smiles_key` or `smiles`), if both columns exist.
   - Else no grouping.
7. Save two year-distribution plots and `train.parquet`, `val.parquet`, `test.parquet` with `save_named_splits` (zstd, level 3). A `split_stats.json` file is saved if the data has a numeric `log10_value` column.

---

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `splits.train_frac`, `splits.valid_frac`, `splits.test_frac` | `0.9`, `0.05`, `0.05` | Must sum to 1 |
| `splits.time_split_date_field` | `date_created` | `date_created`, `date_modified` or `date_sequence_modified` |
| `splits.time_split_dataset_date_column` | `uniprot_date` | Date column in the dataset. The default applies only when the field is `date_created` |
| `splits.time_split_missing_date_split` | `train` | `train`, `val`, `test`, `drop` or `error` |
| `splits.time_split_output_dir_name` | `uniprot_time_splits` | Folder name next to the random split folder, used when `time_split.output_dir` is not set |
| `splits.time_split_enabled` | `false` | Read by the classification builder only. `scripts/generate_splits.py` does not read it |
| `splits.dataset.time_split.input_dataset_parquet_file_path` | | Input parquet file. Else the input of another split block |
| `splits.dataset.time_split.output_dir` | | Output folder |
| `splits.dataset.time_split.group_cols` | none | Columns that make the groups |
| `splits.dataset.time_split.enabled` | `true` | Set to `false` to skip this split |

`scripts/generate_splits.py` runs the splitter for a dataset when it has a `time_split` block or a `sequence_split` block.

---

## Notes

1. The row fractions are close to the targets. Dates with many rows and the group move cause the difference.
2. The split needs enough distinct dates to fill all three splits.
