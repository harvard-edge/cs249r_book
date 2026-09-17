#!/usr/bin/env python3
"""Reject unresolved Quarto cross-references in generated website pages."""

from html import unescape
import os
from pathlib import Path
import re
import sys


UNRESOLVED = re.compile(r"\?@(?:lst|fig|tbl|sec|eq)-[A-Za-z0-9_-]+")


def main() -> int:
    output = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_build"))
    if not output.is_dir():
        print(f"Rendered reference check: missing output directory {output}", file=sys.stderr)
        return 1
    pages = sorted(output.rglob("*.html"))
    if not pages:
        print(f"Rendered reference check: no HTML pages in {output}", file=sys.stderr)
        return 1
    failures = []
    for page in pages:
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for reference in UNRESOLVED.findall(unescape(line)):
                failures.append(f"{page}:{number}: unresolved {reference}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Rendered reference check passed ({len(pages)} HTML pages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
