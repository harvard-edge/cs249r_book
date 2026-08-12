#!/usr/bin/env python3
"""Compatibility wrapper for Binder-native margin geometry checks.

Core implementation lives in ``book/cli/checks/margin_geometry.py``. Keep this
script so older notes and shell history that call ``book/tools/audit`` still
work, but do not put Binder-critical check logic here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from book.cli.checks.margin_geometry import scan_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=0)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    summary = scan_pdf(args.pdf, first=args.first, last=args.last)
    counts = summary.counts

    last = args.last or summary.page_count
    print(f"Scanned pages {args.first}-{last} of {summary.page_count} | {args.pdf}")
    print(f"  overlaps:        {counts.get('overlap', 0)}")
    print(f"  overflow-bottom: {counts.get('overflow-bottom', 0)}")
    print(f"  overflow-top:    {counts.get('overflow-top', 0)}")
    print(f"  TOTAL findings:  {len(summary.findings)}\n")
    for finding in summary.findings:
        print(
            f"  p{finding.page:<4} {finding.issue:<16} "
            f"{finding.side:<5} {finding.detail}"
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(f) for f in summary.findings], indent=2)
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
