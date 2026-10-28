#!/usr/bin/env python3
"""Build data/results/ and the stat cards in data/facts.json; --check verifies them.

k_cat, K_M, K_i, K_d and EC50 come from the camera-ready tex longtables. The other tasks
come from the result CSVs. IC50 also gets the tex rows that its CSV lacks.
Paths come from arguments, then the env vars LC_RESULTS_SRC and LC_TEX, then the defaults.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("LC_RESULTS_SRC", "/home/adhil/github/EMULaToR/figures/data"))
TEX = Path(os.environ.get("LC_TEX", "/home/adhil/github/Leak-CURBER-NeurIPS-2026/camera_ready.tex"))

Rows = list[list[str]]
HEADER = ["Split", "Method"] + [f"{m}_{s}" for m in ("RMSE", "MAE", "R2", "PCC", "SCC") for s in ("mean", "std")]
TABLES = ("tab:kinetic_detailed_combined", "tab:binding_detailed_combined")
TASKS = {  # section title in the tex -> CSV name
    r"k_{\mathrm{cat}}$ prediction": "kcat",
    r"K_M$ prediction": "km",
    r"K_i$ prediction": "ki",
    r"K_d$ prediction": "kd",
    r"\mathrm{EC}_{50}$ prediction": "ec50",
    r"\mathrm{IC}_{50}$ prediction": "ic50",
}
TEX_ONLY = ("kcat", "km", "ki", "kd", "ec50")
METHODS = {"EITLEM-Kinetics": "EITLEM"}
WRAPPER = re.compile(r"\\[A-Za-z]+\{([^{}]*)\}")
SECTION = re.compile(r"\\textbf\{\$(.+)\}\}\s*\\\\")


def clean(cell: str) -> str:
    cell = cell.strip()
    while (new := WRAPPER.sub(r"\1", cell)) != cell:
        cell = new
    return cell.replace("$", "").strip()


def value(cell: str) -> list[str]:
    cell = clean(cell)
    if cell in ("", "--"):
        return ["", ""]
    mean, _, std = cell.partition(r"\pm")
    return [mean.strip(), std.strip()]


def parse_tex(tex: str) -> dict[str, Rows]:
    """Return the rows of every longtable section, keyed by task name."""
    out: dict[str, Rows] = {}
    for label in TABLES:
        start = tex.index(r"\label{" + label + "}")
        body = tex[start : tex.index(r"\end{longtable}", start)]
        body = body[body.index(r"\endlastfoot") :]
        task, split = "", ""
        for line in body.splitlines():
            if r"\multicolumn{7}{@{}l}{\textbf" in line:
                m = SECTION.search(line)
                task = TASKS.get(m.group(1), "") if m else ""
                continue
            cells = line.rstrip().removesuffix(r"\\").split("&")
            if len(cells) != 7 or not task:
                continue
            split = clean(cells[0]) or split
            method = clean(cells[1])
            out.setdefault(task, []).append([split, METHODS.get(method, method)] + [v for c in cells[2:] for v in value(c)])
    return out


def read_rows(path: Path) -> Rows:
    with path.open(newline="") as f:
        return list(csv.reader(f))[1:]


def write_rows(path: Path, rows: Rows, header: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def merge_missing(rows: Rows, extra: Rows) -> Rows:
    """Add the extra rows whose (Split, Method) key is absent, after the last row of the same split."""
    have = {(r[0], r[1]) for r in rows}
    for row in extra:
        if (row[0], row[1]) in have:
            continue
        idx = max((i for i, r in enumerate(rows) if r[0] == row[0]), default=-1)
        rows.insert(idx + 1, row)
    return rows


def describe(path: Path) -> dict:
    """Return the metric columns, splits and methods of one result CSV."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []
    return {
        "file": path.name,
        "metrics": [c[:-5] for c in cols if c.endswith("_mean")],
        "splits": list(dict.fromkeys(r["Split"] for r in rows)),
        "methods": list(dict.fromkeys(r["Method"] for r in rows)),
        "rows": len(rows),
    }


def cell(results: Path, task: str, metric: str, method: str, split: str) -> str:
    """Format 'mean ± std' for one result row, with a true minus sign."""
    with (results / f"{task}.csv").open(newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["Split"] == split and r["Method"] == method)
    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
    return (f"{mean} ± {std}" if std else mean).replace("-", "−")


def fill_card(card: dict, results: Path) -> dict:
    """Return the card with from, to, note and source filled from data/results."""
    methods = card["method"] if isinstance(card["method"], list) else [card["method"]]
    get = lambda split: " / ".join(cell(results, card["task"], card["metric"], m, split) for m in methods)
    out = dict(card, source=f"{card['task']}.csv")
    out["from"] = f"{card['from_split']} {get(card['from_split'])}"
    out["to"] = [f"{s} {get(s)}" for s in card["to_splits"]]
    if card.get("paper_reported"):
        mean = cell(results, card["task"], card["metric"], methods[0], "Original")
        out["note"] = f"R² {mean} reported in the original paper"
    return out


def fill_note(note: dict, results: Path) -> dict:
    """Return the note spec with its text filled from data/results: the range of means over its splits."""
    means = [float(cell(results, note["task"], note["metric"], note["method"], s).split(" ")[0].replace("−", "-")) for s in note["splits"]]
    names = ", ".join(note["splits"][:-1]) + " and " + note["splits"][-1]
    return dict(note, text=f"{note['method']} scores {min(means)} to {max(means)}% at EC level 4 on the {names} splits.")


def sync(src: Path, tex: Path, out: Path, facts_path: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.csv"):
        old.unlink()
    tables = parse_tex(tex.read_text())
    for f in sorted(src.glob("*.csv")):
        if f.stem in TEX_ONLY:
            continue
        if f.stem == "ic50":
            rows = merge_missing(read_rows(f), tables["ic50"])
            write_rows(out / f.name, rows, HEADER)
        else:
            shutil.copy2(f, out / f.name)
    for task in TEX_ONLY:
        write_rows(out / f"{task}.csv", tables[task], HEADER)
    tasks = [describe(f) for f in sorted(out.glob("*.csv"))]
    (out / "index.json").write_text(json.dumps({"tasks": tasks}, indent=1) + "\n")

    facts = json.loads(facts_path.read_text())
    for key in ("regression", "retrieval"):
        facts["stats"][key] = [fill_card(c, out) for c in facts["stats"][key]]
    facts["stats"]["note"] = fill_note(facts["stats"]["note"], out)
    facts_path.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} CSV files to {out} and stat cards to {facts_path}")


def check(tex: Path, facts_path: Path, out: Path) -> list[str]:
    errors: list[str] = []
    text = tex.read_text()
    facts = json.loads(facts_path.read_text())
    scale = facts["stats"]["scale"]

    row = re.search(r"\\textbf\{Leak-CURBER\} & \\textbf\{([\d,]+)\} & \\textbf\{([\d,]+)\} & \\textbf\{([\d,]+)\}", text)
    if not row:
        errors.append("Table 1 Leak-CURBER row not found")
    else:
        paper = dict(zip(("reactions", "enzymes", "pairs"), (int(g.replace(",", "")) for g in row.groups())))
        errors += [f"{k}: facts {scale[k]} != paper {v}" for k, v in paper.items() if scale[k] != v]
    for phrase in ("10 biochemical databases", "ten subtasks", "seventeen"):
        if phrase not in text:
            errors.append(f'phrase "{phrase}" not found in paper')
    if (scale["databases"], scale["subtasks"], scale["baselines"]) != (10, 10, 17):
        errors.append("databases/subtasks/baselines must be 10/10/17")

    files = sorted(out.glob("*.csv"))
    if not files:
        errors.append(f"no CSV files in {out}")
    for f in files:
        try:
            info = describe(f)
        except (KeyError, csv.Error) as e:
            errors.append(f"{f.name}: cannot parse ({e!r}); Split and Method columns required")
            continue
        if not info["rows"] or not info["metrics"]:
            errors.append(f"{f.name}: no rows or no metric columns")

    tables = parse_tex(text)
    for task in TEX_ONLY:
        if not (out / f"{task}.csv").exists() or read_rows(out / f"{task}.csv") != tables[task]:
            errors.append(f"{task}.csv differs from a fresh parse of the tex")
    if (out / "ic50.csv").exists():
        have = {(r[0], r[1]) for r in read_rows(out / "ic50.csv")}
        errors += [f"ic50.csv lacks tex row {r[0]},{r[1]}" for r in tables["ic50"] if (r[0], r[1]) not in have]

    for key in ("regression", "retrieval"):
        for card in facts["stats"][key]:
            try:
                ok = fill_card(card, out) == card
            except (KeyError, StopIteration):
                ok = False
            if not ok:
                errors.append(f"stat card {key} / {card.get('label')} does not match data/results")
    if fill_note(facts["stats"]["note"], out) != facts["stats"]["note"]:
        errors.append("stats.note does not match data/results")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify facts.json and the CSVs instead of writing them")
    p.add_argument("--src", type=Path, default=SRC, help="folder with the source CSV files (LC_RESULTS_SRC)")
    p.add_argument("--tex", type=Path, default=TEX, help="paper camera_ready.tex (LC_TEX)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "results")
    p.add_argument("--facts", type=Path, default=ROOT / "data" / "facts.json")
    a = p.parse_args()
    if not a.check:
        sync(a.src, a.tex, a.out, a.facts)
        return 0
    errors = check(a.tex, a.facts, a.out)
    print("\n".join(errors) if errors else "all checks passed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
