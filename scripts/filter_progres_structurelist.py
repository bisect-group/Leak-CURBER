#!/usr/bin/env python3
"""Filter a progres structure list by removing entries that fail graph construction.

The input format matches `progres embed -l`:
- column 1: structure file path
- column 2: domain id
- column 3+: optional note text
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass

import rootutils
from progres import progres as p

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.rich_progress import progress  # noqa: E402


@dataclass
class RejectRecord:
    line_no: int
    file_path: str
    domain_id: str
    reason: str
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a progres structure list, excluding rows that fail coordinate "
            "parsing or graph construction."
        )
    )
    parser.add_argument(
        "-i",
        "--input-list",
        required=True,
        help="Input structure list file passed to progres embed -l",
    )
    parser.add_argument(
        "-o",
        "--output-list",
        required=True,
        help="Output filtered structure list",
    )
    parser.add_argument(
        "--bad-list",
        default=None,
        help=(
            "Optional TSV file for excluded rows (default: <output-list>.excluded.tsv)"
        ),
    )
    parser.add_argument(
        "-f",
        "--fileformat",
        default="guess",
        choices=["guess", "pdb", "mmcif", "mmtf", "coords"],
        help="File format for progres.read_coords (default: guess)",
    )
    parser.add_argument(
        "--min-ca",
        type=int,
        default=3,
        help="Minimum number of C-alpha coordinates required (default: 3)",
    )
    parser.add_argument(
        "--skip-graph-check",
        action="store_true",
        help="Only check parsed C-alpha count, skip full coords_to_graph check",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N rows (default: 1000)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable Rich progress output even when stderr is not a terminal",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help=(
            "Number of worker processes. Uses multiprocessing Pool.imap_unordered "
            "when >1 (default: cpu_count-1)."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help="Chunksize passed to Pool.imap_unordered (default: 64)",
    )
    return parser.parse_args()


def count_lines(file_path: str) -> int:
    total = 0
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def split_input_line(raw_line: str) -> tuple[str, str, str] | None:
    cols = raw_line.rstrip("\n").split(None, 2)
    if len(cols) < 2:
        return None
    file_path = cols[0]
    domain_id = cols[1]
    note = cols[2] if len(cols) > 2 else ""
    return file_path, domain_id, note


def reject_to_tsv_line(reject: RejectRecord) -> str:
    details = reject.details.replace("\t", " ").replace("\n", " ")
    return (
        f"{reject.line_no}\t{reject.file_path}\t{reject.domain_id}\t"
        f"{reject.reason}\t{details}\n"
    )


def evaluate_line(task: tuple[int, str, str, int, bool]) -> tuple[int, str | None, RejectRecord | None]:
    line_no, raw_line, fileformat, min_ca, skip_graph_check = task

    stripped = raw_line.strip()
    if not stripped:
        return (
            line_no,
            None,
            RejectRecord(
                line_no=line_no,
                file_path="",
                domain_id="",
                reason="malformed_line",
                details="empty or whitespace-only line",
            ),
        )

    parsed = split_input_line(raw_line)
    if parsed is None:
        return (
            line_no,
            None,
            RejectRecord(
                line_no=line_no,
                file_path="",
                domain_id="",
                reason="malformed_line",
                details="needs at least 2 columns",
            ),
        )

    file_path, domain_id, note = parsed

    if not os.path.isfile(file_path):
        return (
            line_no,
            None,
            RejectRecord(
                line_no=line_no,
                file_path=file_path,
                domain_id=domain_id,
                reason="missing_file",
                details="not found on disk",
            ),
        )

    try:
        coords = p.read_coords(file_path, fileformat=fileformat, res_range=None)
        n_ca = len(coords)
        if n_ca < min_ca:
            return (
                line_no,
                None,
                RejectRecord(
                    line_no=line_no,
                    file_path=file_path,
                    domain_id=domain_id,
                    reason="too_few_ca",
                    details=f"n_ca={n_ca}",
                ),
            )

        if not skip_graph_check:
            p.coords_to_graph(coords)

        if note:
            keep_line = f"{file_path} {domain_id} {note}"
        else:
            keep_line = f"{file_path} {domain_id}"
        return line_no, keep_line, None
    except Exception as exc:  # noqa: BLE001
        return (
            line_no,
            None,
            RejectRecord(
                line_no=line_no,
                file_path=file_path,
                domain_id=domain_id,
                reason="progres_error",
                details=repr(exc),
            ),
        )


def main() -> int:
    args = parse_args()

    if args.bad_list is None:
        args.bad_list = args.output_list + ".excluded.tsv"

    kept = 0
    rejected = 0
    total = 0

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")

    use_progress = not args.no_progress and (sys.stderr.isatty() or args.progress)

    pbar = None
    if use_progress:
        total_rows = count_lines(args.input_list)
        pbar = progress(
            total=total_rows,
            desc="Filtering structurelist",
            unit="row",
        )
        pbar.__enter__()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_list)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.bad_list)), exist_ok=True)

    with open(args.input_list, "r", encoding="utf-8") as inf, open(
        args.output_list, "w", encoding="utf-8"
    ) as outf, open(args.bad_list, "w", encoding="utf-8") as badf:
        badf.write("line_no\tfile_path\tdomain_id\treason\tdetails\n")

        tasks = (
            (line_no, raw_line, args.fileformat, args.min_ca, args.skip_graph_check)
            for line_no, raw_line in enumerate(inf, start=1)
        )

        if args.workers == 1:
            results = map(evaluate_line, tasks)
        else:
            print(
                (
                    f"Using multiprocessing with workers={args.workers}, "
                    f"chunksize={args.chunksize} (output order kept)"
                ),
                file=sys.stderr,
            )
            pool = mp.Pool(processes=args.workers)
            # imap keeps the output in line_no order.
            results = pool.imap(evaluate_line, tasks, chunksize=args.chunksize)

        try:
            for _, keep_line, reject in results:
                total += 1
                if pbar is not None:
                    pbar.update(1)

                if keep_line is not None:
                    outf.write(keep_line + "\n")
                    kept += 1
                else:
                    badf.write(reject_to_tsv_line(reject))
                    rejected += 1

                if pbar is not None and args.progress_every > 0 and total % args.progress_every == 0:
                    pbar.set_postfix(kept=kept, rejected=rejected)
                elif pbar is None and args.progress_every > 0 and total % args.progress_every == 0:
                    print(
                        f"processed={total} kept={kept} rejected={rejected}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            if args.workers > 1:
                pool.close()
                pool.join()

    if pbar is not None:
        pbar.set_postfix(kept=kept, rejected=rejected)
        pbar.close()

    print(
        f"Done. processed={total} kept={kept} rejected={rejected}",
        file=sys.stderr,
    )
    print(f"Filtered list: {args.output_list}", file=sys.stderr)
    print(f"Excluded rows: {args.bad_list}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
