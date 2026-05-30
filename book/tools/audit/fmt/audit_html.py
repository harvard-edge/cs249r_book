#!/usr/bin/env python3
"""
audit_html.py
=============

Scan rendered Quarto HTML for spurious ``*.0`` in narrative prose (final check).

Strips code cells and Quarto chrome, then flags numbers like ``739,726.0`` or
``153.0 percent`` that should have been formatted as integers. Uses the same
false-positive filters as ``audit_prose.py`` (via ``spurious_zero``).

Usage::

    python3 book/tools/audit/fmt/audit_html.py path/to/chapter.html

Exit code 0 prints ``CLEAN``; exit code 1 lists findings.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book.cli.checks.currency_style import audit_rendered_file
from spurious_zero import find_spurious_zeros

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "audit_html.py requires beautifulsoup4 (pip install beautifulsoup4)"
    ) from exc


def audit_html(file_path: Path) -> list[dict[str, str]]:
    issues = [
        {
            "value": issue.code,
            "context": issue.context,
        }
        for issue in audit_rendered_file(file_path)
    ]

    soup = BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser")
    content = soup.find("main") or soup.body
    if not content:
        return issues

    for tag in content(["script", "style", "pre", "code"]):
        tag.decompose()
    for tag in content.find_all(class_=re.compile(r"cell-code|quarto-appendix-contents|quarto-figure")):
        tag.decompose()

    text = content.get_text(separator=" ")
    issues.extend(
        {"value": value, "context": context}
        for value, context in find_spurious_zeros(text)
    )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, help="Rendered chapter .html file")
    args = parser.parse_args()

    if not args.html.is_file():
        print(f"File not found: {args.html}", file=sys.stderr)
        return 1

    issues = audit_html(args.html)
    if not issues:
        print("CLEAN")
        return 0

    for issue in issues:
        print(f"FOUND: {issue['value']} | Context: ...{issue['context']}...")
    return 1


if __name__ == "__main__":
    sys.exit(main())
