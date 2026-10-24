#!/usr/bin/env python3
"""
Post-render step: number front-matter figures and tables P.1, P.2, ...

Figures and tables in front matter (preface, prerequisites) sit before
Chapter 1 and have no chapter number. The PDF numbers them P.1, P.2, ...
(shared/tex/header-includes.tex). Quarto's HTML output numbers them 1, 2, ...
and offers no supported prefix option (2026-10-13: its crossref schema accepts
only a numbering style, a label list fails at render time, and user filters
cannot reach its number formatter). This script rewrites the rendered HTML of
every page under a frontmatter/ folder so captions and in-page references
match the PDF. It is idempotent: an already-prefixed number does not match.
"""

import os
import re
import sys
from pathlib import Path

# Caption prefix ("Figure&nbsp;1:"), also repeated in the lightbox title
# attribute, and cross-reference link text ("figure&nbsp;1").
CAPTION = re.compile(r'(>\s*|title=")(Figure|Table)&nbsp;(\d+)(:)')
XREF = re.compile(r'(class="quarto-xref">)(Figure|figure|Table|table)&nbsp;(\d+)(<)')


def relabel(html_path: Path) -> int:
    text = html_path.read_text(encoding="utf-8")
    count = 0

    def sub(m: re.Match) -> str:
        nonlocal count
        count += 1
        return f"{m.group(1)}{m.group(2)}&nbsp;P.{m.group(3)}{m.group(4)}"

    new_text = XREF.sub(sub, CAPTION.sub(sub, text))
    if count:
        html_path.write_text(new_text, encoding="utf-8")
    return count


def main() -> None:
    if len(sys.argv) > 1:
        build_dir = Path(sys.argv[1])
    elif os.environ.get("QUARTO_PROJECT_OUTPUT_DIR"):
        build_dir = Path(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])
    else:
        build_dir = Path(__file__).resolve().parents[2] / "_build"
    if not build_dir.is_dir():
        sys.exit(f"[frontmatter-labels] build directory not found: {build_dir}")

    total = files = 0
    for html_file in sorted(build_dir.rglob("*.html")):
        if "frontmatter" not in html_file.parts:
            continue
        n = relabel(html_file)
        if n:
            total += n
            files += 1
    print(f"[frontmatter-labels] Relabeled {total} front-matter float number(s) in {files} file(s).")


if __name__ == "__main__":
    main()
