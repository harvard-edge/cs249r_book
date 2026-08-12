#!/usr/bin/env python3
"""
run_scale_lane.py — move ``fmt(x / MILLION, suffix='M')`` into the guarded
``fmt_count`` formatter with the same BYTE-IDENTICAL gate as the percent lane.

    cell:  X = fmt(x / MILLION, …, suffix='M')  → X = fmt_count(x, scale='M', …)

The scale division and the glyph were two facts a reader had to keep in sync;
fmt_count declares the scale once and applies it (and guards counts >= 0). Only
the clean division case is auto-migrated — and only when the divisor's
magnitude matches the glyph, so the rendered value is unchanged. Pre-scaled
values, lowercase 'k', spaced glyphs, fmt_int and prefix= are queued for a
source-level decision (keep the raw count, declare scale=).

Reuses run_percent_lane's generic engine: all-at-once byte-identical accept,
else per-edit bisect keeping only the provably safe sites.

Usage::

    python3 book/tools/audit/fmt/run_scale_lane.py --write <qmd> ...
    python3 book/tools/audit/fmt/run_scale_lane.py --write --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codemod_fmt import scan_scale  # noqa: E402
from run_percent_lane import lane_process  # noqa: E402

SCALE_DUP_CODES = ("scale_dup",)
QUEUE_OUT = HERE / "scale_adjudication_queue.txt"


def process(path: Path) -> tuple[str, str]:
    return lane_process(path, scan_scale, "fmt_count", SCALE_DUP_CODES, QUEUE_OUT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("qmd", nargs="*")
    ap.add_argument("--all", action="store_true", help="all chapters under contents/")
    ap.add_argument("--write", action="store_true", help="required to actually apply (else lists targets)")
    args = ap.parse_args()

    files = [Path(p) for p in args.qmd]
    if args.all:
        files = sorted(Path("book/quarto/contents").rglob("*.qmd"))
    if not files:
        ap.error("pass qmd file(s) or --all")

    if not args.write:
        print("DRY: would process (use --write):")
        for f in files:
            edits, q = scan_scale(f)
            if edits:
                print(f"  {len(edits):3d} edits  ({len(q)} queued)  {str(f).split('contents/')[-1]}")
        return 0

    results: dict[str, list[str]] = {"PASS": [], "PART": [], "SKIP": [], "FAIL": [], "NOOP": []}
    for f in files:
        if not f.is_file():
            continue
        verdict, detail = process(f)
        results[verdict].append(str(f).split("contents/")[-1])
        if verdict != "NOOP":
            print(f"[{verdict}] {str(f).split('contents/')[-1]}\n    {detail}")
    print("\n=== summary ===")
    for v in ("PASS", "PART", "SKIP", "FAIL"):
        print(f"  {v}: {len(results[v])}")
        for f in results[v]:
            print(f"      {f}")
    return 1 if results["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
