#!/usr/bin/env python3
"""
run_time_lane.py — migrate exact time suffixes to fmt_time.

This lane handles the mechanically provable shapes:

    fmt(x, ..., suffix=" ms")       -> fmt_time(x, "millisecond", ...)
    fmt(x, ..., suffix=" seconds")  -> fmt_time(x, "second", ..., style="word")
    fmt_int(x, suffix=" hours")     -> fmt_time(round(x), "hour", precision=0, ...)

The source uses full unit names so the argument is visibly a unit; compact vs
word rendering is controlled by ``style``. The lane accepts a file only when
every exported value and every visible prose preview is byte-identical.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codemod_fmt import scan_time  # noqa: E402
from run_percent_lane import lane_process  # noqa: E402

TIME_DUP_CODES = ("unit_dup", "unit_word_dup")
QUEUE_OUT = HERE / "time_adjudication_queue.txt"


def process(path: Path) -> tuple[str, str]:
    return lane_process(path, scan_time, "fmt_time", TIME_DUP_CODES, QUEUE_OUT)


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
            edits, q = scan_time(f)
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
