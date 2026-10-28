# Leak-CURBER project page

Static site. It has no build step.

## Preview

```
python -m http.server
```

Open http://localhost:8000.

## Refresh the data

```
python scripts/sync_data.py && python scripts/sync_data.py --check
```

Data sources:

- k_cat, K_M, K_i, K_d and EC50 come from the longtables `tab:kinetic_detailed_combined` and `tab:binding_detailed_combined` in the camera-ready tex.
- IC50, EC classification, retrieval, forward synthesis and retrosynthesis come from the result CSVs. IC50 also gets the tex rows its CSV lacks, such as `Original,BIND`.

The sync command writes `data/results/*.csv` and `index.json`. It also fills the stat cards in `data/facts.json` from those CSVs. Each card is a spec: task, metric, method, `from_split`, `to_splits` and an optional `paper_reported` flag.

`--check` exits with a non-zero code when one of these fails:

- The scale numbers in `facts.json` match Table 1 of the paper, and the paper says 10 databases, ten subtasks and seventeen baselines.
- Every CSV parses and has Split and Method columns.
- The five tex-sourced CSVs equal a fresh parse of the tex, and the IC50 CSV has every tex row.
- Every stat card and the Horizyn note equal the values in `data/results/`.

Set the paths with arguments or env vars: `--src` or `LC_RESULTS_SRC` (result CSV folder), `--tex` or `LC_TEX` (camera-ready tex). Run `python scripts/sync_data.py -h` for all options.

## Layout

```
index.html            page
static/css/site.css   tokens, light and dark theme, layout
static/css/figures.css  figure and explorer styles
static/js/facts.js    fills numbers and stat cards from data/facts.json, theme toggle, BibTeX copy
static/fonts/         Inter and Source Serif 4 (woff2) with their licences
static/js/figures.js  figures
static/js/explorer.js results explorer
static/js/splits-demo.js  split animation
static/img/           logos, overview figure, favicon
data/facts.json       every number shown outside the explorer
data/results/         result CSVs and index.json
scripts/sync_data.py  data sync and check
```
