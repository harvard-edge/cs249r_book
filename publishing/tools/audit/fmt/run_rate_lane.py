#!/usr/bin/env python3
"""
run_rate_lane.py — migrate exact service-rate suffixes to fmt_rate.

This lane covers counted service throughputs such as ``tokens/s``, ``img/s``,
``req/s``, ``samples/s``, ``images/s``, and ``FPS``. Physical rates such as
``GB/s`` and ``TFLOP/s`` remain ``fmt_qty`` work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codemod_fmt import scan_rate  # noqa: E402
from run_percent_lane import lane_process  # noqa: E402

RATE_DUP_CODES = ("unit_dup", "unit_word_dup")
QUEUE_OUT = HERE / "rate_adjudication_queue.txt"


def process(path: Path) -> tuple[str, str]:
    return lane_process(path, scan_rate, "fmt_rate", RATE_DUP_CODES, QUEUE_OUT)


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
            edits, q = scan_rate(f)
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
