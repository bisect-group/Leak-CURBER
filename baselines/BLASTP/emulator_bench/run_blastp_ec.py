#!/usr/bin/env python3
"""Run a deterministic, train-only DIAMOND blastp EC-transfer baseline (CARE BLAST search, top-50 hits)."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/processed/datasets/enzyme_classification_dataset"
BASELINE = ROOT / "outputs/baselines/BLASTP"
DIAMOND = ROOT / "external/diamond-2.1.9/diamond"
METRICS = ROOT / "baselines/GraphEC/emulator_bench/results.py"
SPLIT_GROUPS = (
    "random_splits_grouped_sequence",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "enzyme_structure_splits",
)
THRESHOLD_GROUPS = SPLIT_GROUPS[2:]
MAX_HITS, MAX_RANKS = 50, 20
PARTS = ("train", "val", "test")
HIT_COLUMNS = ["protein_id", "target_protein_id", "bitscore"]


class Reporter:
    def __init__(self, disabled=False):
        self.enabled = not disabled and sys.stderr.isatty()
        self.console = Console(stderr=True)

    def message(self, value):
        if not self.enabled:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {value}", file=sys.stderr, flush=True)

    @contextmanager
    def bar(self, text, total):
        if not self.enabled:
            self.message(f"{text}: starting ({total:,} items)")
            yield lambda advance=1, detail=None: None
            self.message(f"{text}: complete")
            return
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task(text, total=total)
            yield lambda advance=1, detail=None: progress.update(
                task, advance=advance, description=detail or text
            )

    @contextmanager
    def activity(self, text):
        if not self.enabled:
            self.message(f"{text}: starting")
            try:
                yield
            finally:
                self.message(f"{text}: complete")
        else:
            with self.console.status(text, spinner="dots"):
                yield

    def summary(self, rows):
        for r in rows:
            k1, k20 = r["care_task1"]["k=1"], r["care_task1"]["k=20"]
            self.message(
                f"{r['split']}: n={r['queries']} hit={r['hit_coverage']:.4f} k1 L4/L3/L2/L1="
                + "/".join(f"{k1[f'level_{i}_accuracy']:.2f}" for i in (4, 3, 2, 1))
                + f" k20 L4={k20['level_4_accuracy']:.2f}"
            )


@contextmanager
def null_bar():
    yield lambda advance=1, detail=None: None


@contextmanager
def null_activity():
    yield


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf8", delete=False) as f:
        json.dump(data, f, indent=2, sort_keys=True)
        tmp = Path(f.name)
    tmp.replace(path)


def fingerprint(paths, extra=""):
    h = hashlib.sha256(extra.encode())
    for path in sorted(map(Path, paths)):
        stat = path.stat()
        h.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()


def protein_id(uniprot, sequence):
    return hashlib.sha1(f"{uniprot}\0{sequence}".encode()).hexdigest()[:20]


def slugify(value):
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())).strip("._")


def is_full_ec(label):
    return "-" not in label and label.count(".") == 3


def discover_splits(dataset):
    folders = [
        f
        for g in SPLIT_GROUPS
        for f in (
            sorted((dataset / g).glob("threshold_*")) if g in THRESHOLD_GROUPS else [dataset / g]
        )
    ]
    return [f for f in folders if all((f / f"{part}.parquet").is_file() for part in PARTS)]


def source_files(source):
    splits = discover_splits(source)
    if not splits:
        raise FileNotFoundError(
            f"No supported complete train/val/test split configurations found in {source}"
        )
    return [split / f"{part}.parquet" for split in splits for part in PARTS]


def copy_atomic(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        shutil.copy2(source, tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)


def materialize_dataset(source, baseline, force=False, reporter=None):
    source, baseline = Path(source).resolve(), Path(baseline).resolve()
    files = source_files(source)
    rel = [x.relative_to(source) for x in files]
    sig = fingerprint(files, "blastp-baseline-inputs-v2")
    copied = baseline / "dataset"
    marker = baseline / "dataset_materialization.complete.json"
    expected = [copied / x for x in rel]
    if (
        not force
        and marker.exists()
        and all(x.is_file() for x in expected)
        and json.loads(marker.read_text()).get("fingerprint") == sig
    ):
        if reporter:
            reporter.message("Materializing dataset: reusing completed baseline-local copy")
        return copied
    if copied.exists():
        wanted = set(rel)
        for stale in copied.rglob("*.parquet"):
            if stale.relative_to(copied) not in wanted:
                stale.unlink()
    with reporter.bar("Materializing dataset", len(files)) if reporter else null_bar() as advance:
        for src, relative in zip(files, rel):
            copy_atomic(src, copied / relative)
            advance()
    atomic_json(
        marker, {"fingerprint": sig, "source_dataset": str(source), "files": [str(x) for x in rel]}
    )
    return copied


def partition_frame(path):
    """One row per (Entry, Sequence) with its full-EC labels, as CLEAN builds its queries."""
    frame = pd.read_parquet(path, columns=["uniprot_id", "sequence", "ec_number"]).dropna()
    frame["Entry"] = frame.uniprot_id.astype(str).map(slugify)
    frame["Sequence"] = frame.sequence.astype(str).str.replace(r"\s+", "", regex=True)
    frame["ec"] = frame.ec_number.astype(str).str.split(";")
    frame = frame.explode("ec")
    frame["ec"] = frame.ec.str.strip()
    frame = frame[frame.ec.map(is_full_ec) & (frame.Sequence != "")]
    frame = (
        frame.groupby(["Entry", "Sequence"], sort=True)
        .ec.agg(lambda x: sorted(set(x)))
        .rename("ec_labels")
        .reset_index()
    )
    frame["protein_id"] = [protein_id(u, s) for u, s in zip(frame.Entry, frame.Sequence)]
    return frame


def write_fasta(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf8", delete=False) as f:
        for row in frame.itertuples(index=False):
            f.write(f">{row.protein_id}\n{row.Sequence}\n")
        tmp = Path(f.name)
    tmp.replace(path)


def rank_ecs(hits, labels, frequency, depth=MAX_RANKS):
    """Rank ECs by best hit bitscore; ties by train frequency (high first), then EC string."""
    best = {}
    for target, score in hits:
        for ec in labels[target]:
            best[ec] = max(score, best.get(ec, score))
    return sorted(best, key=lambda ec: (-best[ec], -frequency[ec], ec))[:depth]


def care_frame(queries, ranked):
    frame = queries[["Entry", "protein_id"]].copy()
    frame["EC number"] = queries.ec_labels.map(";".join)
    for i in range(MAX_RANKS):
        frame[str(i)] = [r[i] if i < len(r) else None for r in ranked]
    return frame


def care_metrics(frame, module_path=METRICS):
    spec = importlib.util.spec_from_file_location("care_results", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_care_metrics(frame)


def run(command, log, reporter=None, description=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with (
        reporter.activity(f"{description or 'Running command'} (log: {log})")
        if reporter
        else null_activity()
    ):
        with log.open("w") as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)


def executable_version(executable):
    return subprocess.run(
        [str(executable), "version"], check=True, text=True, capture_output=True
    ).stdout.strip()


def build_database(fasta, database, args, log, name):
    if not args.force and Path(f"{database}.dmnd").exists():
        return
    database.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            args.diamond,
            "makedb",
            "--in",
            str(fasta),
            "--db",
            str(database),
            "--ignore-warnings",
            "--threads",
            str(args.threads_per_run),
        ],
        log,
        reporter=args.reporter,
        description=f"Building DIAMOND train DB: {name}",
    )


def search(fasta, database, hits, args, log, name):
    """CARE's DIAMOND call with default sensitivity and E-value, except -k MAX_HITS."""
    command = [
        args.diamond,
        "blastp",
        "-k",
        str(MAX_HITS),
        "--query",
        str(fasta),
        "--db",
        str(database),
        "--out",
        str(hits),
        "--outfmt",
        "6",
        "qseqid",
        "sseqid",
        "bitscore",
        "--threads",
        str(args.threads_per_run),
        "--tmpdir",
        str(args.tmpdir),
        "--quiet",
    ]
    run(command, log, reporter=args.reporter, description=f"DIAMOND test search: {name}")


def read_hits(path):
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame(
            {c: pd.Series(dtype="string" if c != "bitscore" else float) for c in HIT_COLUMNS}
        )
    return pd.read_csv(
        path,
        sep="\t",
        names=HIT_COLUMNS,
        dtype={"protein_id": "string", "target_protein_id": "string"},
    )


def split_dir(baseline_dir, relative):
    """<endpoint>/<split_group>/<threshold_x>, or <split_group>/all for splits without thresholds."""
    group = relative if len(relative.parts) > 1 else relative / "all"
    return baseline_dir / "enzyme_classification" / group


def evaluate_split(split, args):
    relative = split.relative_to(args.dataset_dir)
    name = str(relative).replace("/", "__")
    out = split_dir(args.baseline_dir, relative)
    out.mkdir(parents=True, exist_ok=True)
    complete = out / "complete.json"
    params = {
        "k": MAX_HITS,
        "ranks": MAX_RANKS,
        "diamond_version": executable_version(args.diamond),
        "search": "diamond blastp default sensitivity, default evalue, no coverage filter",
        "inputs": fingerprint([split / f"{p}.parquet" for p in PARTS]),
    }
    if (
        not args.force
        and complete.exists()
        and json.loads(complete.read_text()).get("parameters") == params
    ):
        return json.loads(complete.read_text())
    train, test = (partition_frame(split / f"{p}.parquet") for p in ("train", "test"))
    for part, frame in (("train", train), ("test", test)):
        write_fasta(frame, out / f"{part}.fasta")
    db = out / "db" / "train"
    logs = out / "logs"
    build_database(out / "train.fasta", db, args, logs / "makedb.log", name)
    hitfile = out / "test_hits.tsv"
    search(out / "test.fasta", db, hitfile, args, logs / "search_test.log", name)
    labels = dict(zip(train.protein_id, train.ec_labels))
    frequency = Counter(ec for x in train.ec_labels for ec in x)
    grouped = {
        q: list(zip(g.target_protein_id, g.bitscore))
        for q, g in read_hits(hitfile).groupby("protein_id")
    }
    ranked = [rank_ecs(grouped.get(q, []), labels, frequency) for q in test.protein_id]
    frame = care_frame(test, ranked)
    frame.to_parquet(out / "test_predictions.parquet", index=False)
    hits = sum(map(bool, ranked))
    metrics = {
        "care_task1": care_metrics(frame, args.metrics_module),
        "queries": len(test),
        "hits": hits,
        "no_hit": len(test) - hits,
        "hit_coverage": hits / len(test) if len(test) else None,
        "train_proteins": len(train),
    }
    atomic_json(out / "test_metrics.json", metrics)
    summary = {
        "split": str(relative),
        "seeds": "not_applicable_deterministic",
        "parameters": params,
        **metrics,
    }
    atomic_json(complete, summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dataset", default=SOURCE)
    p.add_argument("--baseline-dir", default=BASELINE)
    p.add_argument("--diamond", default=str(DIAMOND))
    p.add_argument("--metrics-module", default=str(METRICS))
    p.add_argument("--tmpdir", default=tempfile.gettempdir())
    p.add_argument("--threads-per-run", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--total-threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if (
        args.threads_per_run < 1
        or args.total_threads < 1
        or args.threads_per_run > args.total_threads
    ):
        raise SystemExit("--threads-per-run must be between 1 and --total-threads")
    if not Path(args.diamond).is_file():
        raise SystemExit(f"Executable not found: {args.diamond}")
    args.reporter = Reporter(args.no_progress)
    args.baseline_dir = Path(args.baseline_dir)
    args.tmpdir = Path(args.tmpdir)
    args.tmpdir.mkdir(parents=True, exist_ok=True)
    args.dataset_dir = materialize_dataset(
        args.source_dataset, args.baseline_dir, args.force, args.reporter
    )
    if args.prepare_only:
        return
    splits = discover_splits(args.dataset_dir)
    summaries = []
    with args.reporter.bar("Evaluating splits", len(splits)) as advance:
        with ThreadPoolExecutor(
            max_workers=max(1, args.total_threads // args.threads_per_run)
        ) as pool:
            futures = {pool.submit(evaluate_split, split, args): split for split in splits}
            for future in as_completed(futures):
                summaries.append(future.result())
                advance(
                    detail=f"Evaluating splits: completed {futures[future].relative_to(args.dataset_dir)}"
                )
    summaries.sort(key=lambda r: r["split"])
    atomic_json(
        args.baseline_dir / "enzyme_classification" / "aggregate_metrics.json", {"runs": summaries}
    )
    args.reporter.summary(summaries)


if __name__ == "__main__":
    main()
