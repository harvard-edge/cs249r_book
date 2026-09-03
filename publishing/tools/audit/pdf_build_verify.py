#!/usr/bin/env python3
"""Thin CLI wrapper — canonical entry point is `./binder check pdf`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "book"))

from cli.commands._pdf_checks import (  # noqa: E402
    default_pdf_path,
    format_failure_report,
    verify_pdf,
)

BOOK_DIR = REPO_ROOT / "book" / "quarto"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify a built volume PDF for render defects.")
    parser.add_argument("--vol1", action="store_true", help="Verify Volume I default PDF path")
    parser.add_argument("--vol2", action="store_true", help="Verify Volume II default PDF path")
    parser.add_argument("--pdf", type=Path, help="Explicit PDF path to verify")
    parser.add_argument("--log", type=Path, help="Optional Quarto render log to scan for warnings")
    parser.add_argument("--quiet", action="store_true", help="Only print output when issues are found")
    args = parser.parse_args(argv)

    volumes: list[tuple[str, Path]] = []
    if args.pdf:
        volumes.append(("custom PDF", args.pdf.resolve()))
    elif args.vol1:
        volumes.append(("Volume I", default_pdf_path(BOOK_DIR, "vol1")))
    elif args.vol2:
        volumes.append(("Volume II", default_pdf_path(BOOK_DIR, "vol2")))
    else:
        parser.error("specify --vol1, --vol2, or --pdf")

    overall_ok = True
    for label, pdf_path in volumes:
        issues = verify_pdf(pdf_path, log_path=args.log)
        if issues:
            overall_ok = False
            print(format_failure_report(label, pdf_path, issues))
            if len(volumes) > 1:
                print()
        elif not args.quiet:
            print(f"PDF validation passed for {label}: {pdf_path}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
