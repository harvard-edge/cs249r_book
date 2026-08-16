#!/usr/bin/env python3
"""Scan a built HTML volume for defects a reader would see on the page.

Source-level checks cannot catch these: each one is something that looks fine in
the .qmd and only goes wrong on the way to the page. Every pattern here was a
real reader-visible defect found in the 2026-08-16 render audit.

  literal-python     an inline `{python}` ref that lost a backtick, so the raw
                     expression prints instead of its value
  unresolved-ref     a ?@ref Quarto could not resolve
  traceback          a Python traceback rendered into the page
  offset-directive   a PDF-only [offset=NNmm] sidenote directive leaking into
                     HTML because it was stripped only in the LaTeX branch
  latex-macro        a LaTeX-only macro such as \\mbox whose CONTENT is silently
                     dropped by the HTML writer, truncating the sentence
  doubled-unit       a self-labeling formatter value ("128 GPUs") followed by a
                     redundant prose noun ("accelerators")

Math inside \\( \\), \\[ \\], and pseudocode containers is excluded: that markup
is rendered client-side and raw LaTeX there is expected.

Usage:
    check_rendered_html.py <built-html-dir> [...]
Exit status is 1 when any defect is found.
"""
from __future__ import annotations

import html as htmllib
import re
import sys
from pathlib import Path

DROP = re.compile(r"<(script|style|pre|code)\b.*?</\1>", re.S | re.I)
PSEUDO = re.compile(r'<[^>]*class="[^"]*pseudocode[^"]*".*?</div>', re.S | re.I)
MATH = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$", re.S)
TAG = re.compile(r"<[^>]+>")

UNITS = (
    r"(?:GPUs?|accelerators?|devices?|tokens?|steps?|parameters?|nodes?|workers?"
    r"|participants?|clusters?|micro-batches?|false wakes?)"
)

CHECKS: list[tuple[str, re.Pattern[str]]] = [
    ("literal-python", re.compile(r"\{python\}")),
    ("unresolved-ref", re.compile(r"\?@[a-zA-Z]+-[\w-]+")),
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("offset-directive", re.compile(r"\[offset=[-0-9.]+mm\]")),
    ("latex-macro", re.compile(r"\\(?:mbox|hbox|vbox|phantom|hspace|vspace)\b")),
    # Only a PLURAL first unit indicates a self-labeling formatter value
    # ("128 GPUs accelerators"). A singular first unit is an ordinary compound
    # noun where the unit modifies the noun ("64 GPU workers", "8 GPU nodes"),
    # which is correct English and must not be flagged.
    ("doubled-unit", re.compile(rf"\b\d[\d,\.]*\s+{UNITS}s\s+{UNITS}\b", re.I)),
]


def visible_text(raw: str) -> str:
    t = DROP.sub(" ", raw)
    t = PSEUDO.sub(" ", t)
    t = MATH.sub(" ", t)
    t = TAG.sub(" ", t)
    return htmllib.unescape(t)


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]]
    if not roots:
        print("usage: check_rendered_html.py <built-html-dir> [...]", file=sys.stderr)
        return 2

    findings: list[tuple[str, str, str]] = []
    n_files = 0
    for root in roots:
        for f in sorted(root.rglob("*.html")):
            if "_files" in str(f):
                continue
            n_files += 1
            text = visible_text(f.read_text(errors="ignore"))
            for name, pat in CHECKS:
                for m in pat.finditer(text):
                    s = max(0, m.start() - 60)
                    e = min(len(text), m.end() + 50)
                    findings.append((name, str(f), " ".join(text[s:e].split())))

    if not findings:
        print(f"OK: {n_files} rendered pages, no reader-visible defects")
        return 0

    by_kind: dict[str, list] = {}
    for kind, f, ctx in findings:
        by_kind.setdefault(kind, []).append((f, ctx))
    for kind, items in by_kind.items():
        print(f"\n{kind}  ({len(items)})")
        seen = set()
        for f, ctx in items:
            key = (f, ctx[:60])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {f}")
            print(f"      ...{ctx}...")
    print(f"\n{len(findings)} reader-visible defect(s) across {n_files} pages")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
