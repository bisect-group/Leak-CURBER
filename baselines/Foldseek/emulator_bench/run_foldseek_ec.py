#!/usr/bin/env python3
"""Run a deterministic, train-only Foldseek EC-transfer baseline (CARE Foldseek search, top-50 hits)."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/processed/datasets/enzyme_classification_dataset"
BASELINE = ROOT / "outputs/baselines/Foldseek"
FOLDSEEK = ROOT / "external/foldseek/bin/foldseek"
INTERMEDIATE = ROOT / "data/intermediate"
MASTER = "enzyme_classification_dataset_3d.parquet"
SOURCE_RANK = {"experimental": 0, "alphafold": 1, "esmfold2": 2, "esm3": 3}
# Foldseek names a predicted structure's chain <file>[.pdb][_A]; an experimental one <PDB>_<chain>.
PREDICTED_SUFFIXES = ("", ".pdb", "_A", ".pdb_A")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


blastp = load_module(ROOT / "baselines/BLASTP/emulator_bench/run_blastp_ec.py", "blastp_ec")
MAX_HITS, MAX_RANKS, PARTS = blastp.MAX_HITS, blastp.MAX_RANKS, blastp.PARTS


def select_structures(master, dirs):
    """One structure per (Entry, Sequence), chosen as the lDDT split does.

    The split's `locate_structures` finds each structure file: an experimental mmCIF, else the
    highest AlphaFold version, else ESMFold2, else ESM3. A protein with several located structures
    takes the best source, then the highest PDB coverage, then the lowest structure id.
    """
    sys.path.insert(0, str(ROOT))
    from src.data.components.splitters.protein_struct_lddt import locate_structures

    rows = pd.read_parquet(
        master, columns=["uniprot_id", "sequence", "pdbs", "pdb_type", "pdb_chain", "pdb_coverage"]
    ).dropna(subset=["uniprot_id", "sequence", "pdbs"])
    experimental = rows.pdb_type == "experimental"
    located = locate_structures(
        sorted(rows.pdbs[experimental].unique()),
        sorted(rows.pdbs[~experimental].unique()),
        af_dir=dirs / "alphafold",
        esmfold2_dir=dirs / "esmfold2_pdb",
        esm_dir=dirs / "esm",
        mmcif_dir=dirs / "mmcif",
    )
    rows = rows[rows.pdbs.isin(located)].copy()
    rows["Entry"] = rows.uniprot_id.astype(str).map(blastp.slugify)
    rows["Sequence"] = rows.sequence.astype(str).str.replace(r"\s+", "", regex=True)
    rows["source"] = rows.pdbs.map(lambda s: located[s][1])
    rows["path"] = rows.pdbs.map(lambda s: str(located[s][0]))
    rows["rank"] = rows.source.map(SOURCE_RANK)
    rows = rows.sort_values(
        ["rank", "pdb_coverage", "pdbs"], ascending=[True, False, True], na_position="last"
    ).drop_duplicates(["Entry", "Sequence"])
    is_experimental = rows.source == "experimental"
    rows["hit"] = rows.pdbs
    rows.loc[is_experimental, "hit"] = (
        rows.pdbs[is_experimental].str.partition(":")[0] + "_" + rows.pdb_chain[is_experimental]
    )
    return rows[["Entry", "Sequence", "pdbs", "source", "hit", "path"]].sort_values(
        ["Entry", "Sequence"], ignore_index=True
    )


def selection_fingerprint(selection):
    h = hashlib.sha256(b"foldseek-structure-selection-v1")
    for pdbs, source, hit, path in sorted(
        set(zip(selection.pdbs, selection.source, selection.hit, selection.path))
    ):
        stat = Path(path).stat()
        h.update(f"{pdbs}:{source}:{hit}:{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()


def link_structures(selection, folder):
    """Symlink each selected structure file; one `<PDB>.cif` link serves all its chains."""
    folder.mkdir(parents=True, exist_ok=True)
    for pdbs, path in set(zip(selection.pdbs, selection.path)):
        name = f"{pdbs.partition(':')[0]}.cif" if ":" in pdbs else f"{pdbs}{Path(path).suffix}"
        link = folder / name
        if not link.is_symlink():
            link.symlink_to(path)


def hit_names(selection):
    """Foldseek chain name -> canonical hit name, for every selected structure."""
    names = {}
    for hit, source in zip(selection.hit, selection.source):
        for suffix in PREDICTED_SUFFIXES if source != "experimental" else ("",):
            names[hit + suffix] = hit
    return names


def read_lookup(database, names):
    """Database key of each selected chain, by canonical hit name."""
    lookup = pd.read_csv(
        f"{database}.lookup",
        sep="\t",
        header=None,
        names=["key", "name", "file"],
        dtype={"name": "string"},
    )
    lookup = lookup[lookup.name.isin(names)]
    return dict(zip(lookup.name.map(names), lookup.key))


def prepare_structures(args):
    out = args.baseline_dir / "structures"
    selection = select_structures(args.source_dataset / MASTER, args.intermediate_dir)
    selection.to_parquet(args.baseline_dir / "structure_selection.parquet", index=False)
    sig = selection_fingerprint(selection) + f"|{args.foldseek_version}"
    marker, database = out / "db.complete.json", out / "db" / "all"
    if (
        args.force
        or not marker.exists()
        or json.loads(marker.read_text()).get("fingerprint") != sig
    ):
        database.parent.mkdir(parents=True, exist_ok=True)
        link_structures(selection, out / "links")
        blastp.run(
            [
                args.foldseek,
                "createdb",
                str(out / "links"),
                str(database),
                "--chain-name-mode",
                "1",
                "--threads",
                str(args.threads),
            ],
            out / "logs" / "createdb.log",
            args.reporter,
            "Building Foldseek structure DB",
        )
        blastp.atomic_json(marker, {"fingerprint": sig, "structures": len(selection)})
    names = hit_names(selection)
    selection["key"] = selection.hit.map(read_lookup(database, names))
    return selection, names, database, sig


def subdb(keys, database, target, folder, log, args, description):
    folder.mkdir(parents=True, exist_ok=True)
    key_file = folder / f"{target.name}.keys"
    key_file.write_text("".join(f"{k}\n" for k in sorted(keys)))
    blastp.run(
        [args.foldseek, "createsubdb", str(key_file), str(database), str(target)],
        log,
        args.reporter,
        description,
    )


def search(query_db, target_db, hits, args, log, name):
    """Foldseek easy-search with default settings; only the output columns are chosen."""
    shutil.rmtree(args.tmpdir / f"search_{name}", ignore_errors=True)
    command = [
        args.foldseek,
        "easy-search",
        str(query_db),
        str(target_db),
        str(hits),
        str(args.tmpdir / f"search_{name}"),
        "--format-output",
        "query,target,bits",
        "--threads",
        str(args.threads),
    ]
    blastp.run(command, log, args.reporter, f"Foldseek test search: {name}")
    shutil.rmtree(args.tmpdir / f"search_{name}", ignore_errors=True)


def top_hits(path, names):
    """Top MAX_HITS (target hit, bitscore) per query hit, by bitscore then target name."""
    hits = blastp.read_hits(path)
    hits["protein_id"] = hits.protein_id.map(names)
    hits["target_protein_id"] = hits.target_protein_id.map(names)
    hits = hits.dropna().sort_values(
        ["protein_id", "bitscore", "target_protein_id"], ascending=[True, False, True]
    )
    return {
        q: list(zip(g.target_protein_id, g.bitscore))
        for q, g in hits.groupby("protein_id").head(MAX_HITS).groupby("protein_id")
    }


def evaluate_split(split, selection, names, database, selection_sig, args):
    relative = split.relative_to(args.dataset_dir)
    name = str(relative).replace("/", "__")
    out = blastp.split_dir(args.baseline_dir, relative)
    out.mkdir(parents=True, exist_ok=True)
    complete, logs = out / "complete.json", out / "logs"
    params = {
        "k": MAX_HITS,
        "ranks": MAX_RANKS,
        "foldseek_version": args.foldseek_version,
        "search": "foldseek easy-search default settings, default max-seqs, no coverage filter",
        "inputs": blastp.fingerprint([split / f"{p}.parquet" for p in PARTS]),
        "structure_selection": selection_sig,
    }
    if (
        not args.force
        and complete.exists()
        and json.loads(complete.read_text()).get("parameters") == params
    ):
        return json.loads(complete.read_text())
    train, test = (
        blastp.partition_frame(split / f"{p}.parquet").merge(
            selection[["Entry", "Sequence", "source", "hit", "key"]], how="left"
        )
        for p in ("train", "test")
    )
    shutil.rmtree(out / "db", ignore_errors=True)
    train_total = len(train)
    train, test = train[train.key.notna()], test.assign(has_structure=test.key.notna())
    subdb(
        train.key.astype(int),
        database,
        out / "db" / "train",
        out / "db",
        logs / "subdb_train.log",
        args,
        f"Train subdatabase: {name}",
    )
    subdb(
        test.key[test.has_structure].astype(int),
        database,
        out / "db" / "test",
        out / "db",
        logs / "subdb_test.log",
        args,
        f"Test subdatabase: {name}",
    )
    hitfile = out / "test_hits.tsv"
    search(
        out / "db" / "test", out / "db" / "train", hitfile, args, logs / "search_test.log", name
    )
    grouped = top_hits(hitfile, names)
    train_labels = train.groupby("hit").ec_labels.agg(
        lambda x: sorted({ec for e in x for ec in e})
    )
    frequency = Counter(ec for x in train.ec_labels for ec in x)
    ranked = [
        blastp.rank_ecs(grouped.get(h, []), train_labels, frequency)
        for h in test.hit.where(test.has_structure)
    ]
    frame = blastp.care_frame(test, ranked)
    frame.to_parquet(out / "test_predictions.parquet", index=False)
    hits = sum(map(bool, ranked))
    metrics = {
        "care_task1": blastp.care_metrics(frame, args.metrics_module),
        "queries": len(test),
        "hits": hits,
        "no_hit": int(test.has_structure.sum()) - hits,
        "no_structure": int((~test.has_structure).sum()),
        "hit_coverage": hits / len(test) if len(test) else None,
        "train_proteins": train_total,
        "train_with_structure": len(train),
        "structure_sources": {
            "train": train.source.value_counts().to_dict(),
            "test": test.source.value_counts().to_dict(),
        },
    }
    blastp.atomic_json(out / "test_metrics.json", metrics)
    summary = {
        "split": str(relative),
        "seeds": "not_applicable_deterministic",
        "parameters": params,
        **metrics,
    }
    blastp.atomic_json(complete, summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dataset", default=SOURCE)
    p.add_argument("--baseline-dir", default=BASELINE)
    p.add_argument("--intermediate-dir", default=INTERMEDIATE)
    p.add_argument("--foldseek", default=str(FOLDSEEK))
    p.add_argument("--metrics-module", default=str(blastp.METRICS))
    p.add_argument("--tmpdir", default=tempfile.gettempdir())
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--only", help="run only splits whose name contains this text")
    p.add_argument("--force", action="store_true")
    return p


def main():
    args = parser().parse_args()
    for key in ("source_dataset", "baseline_dir", "intermediate_dir", "tmpdir"):
        setattr(args, key, Path(getattr(args, key)).resolve())
    (args.baseline_dir / "logs").mkdir(parents=True, exist_ok=True)
    sys.stderr = (args.baseline_dir / "logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.log").open(
        "a", buffering=1
    )
    args.reporter = blastp.Reporter(disabled=True)
    args.tmpdir.mkdir(parents=True, exist_ok=True)
    args.foldseek_version = blastp.executable_version(args.foldseek)
    args.dataset_dir = blastp.materialize_dataset(
        args.source_dataset, args.baseline_dir, args.force, args.reporter
    )
    splits = [
        s for s in blastp.discover_splits(args.dataset_dir) if not args.only or args.only in str(s)
    ]
    selection, names, database, sig = prepare_structures(args)
    summaries = []
    for split in splits:
        summaries.append(evaluate_split(split, selection, names, database, sig, args))
        args.reporter.message(f"completed {split.relative_to(args.dataset_dir)}")
    summaries.sort(key=lambda r: r["split"])
    blastp.atomic_json(
        args.baseline_dir / "enzyme_classification" / "aggregate_metrics.json", {"runs": summaries}
    )
    args.reporter.summary(summaries)


if __name__ == "__main__":
    main()
