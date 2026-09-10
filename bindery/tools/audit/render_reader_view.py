#!/usr/bin/env python3
"""Extract the reader's view of a rendered chapter.

A reviewer needs to see what a reader sees, not the markup. This flattens one
built HTML page into plain text that preserves the things a rendering or prose
defect actually shows up in: heading structure, paragraph text, list items,
table cells, captions, callout bodies, and math rendered back to its LaTeX
source. Navigation chrome, scripts, and styles are dropped.

Usage:
    render_reader_view.py <built.html> [--section N] [--list]

    --list        print the section outline with indices and line counts
    --section N   print only section N (use --list first)
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

DROP_BLOCK = re.compile(
    r"<(script|style|head|nav|footer)\b.*?</\1>", re.S | re.I
)
# site chrome that is not chapter content
DROP_ATTR = re.compile(
    r'<(?:div|aside|section)[^>]*\b(?:id|class)="[^"]*'
    r"(?:sidebar|navbar|toc|footer|announcement|quarto-secondary-nav|"
    r'margin-sidebar|nav-page|cookie)[^"]*"[^>]*>',
    re.I,
)


def strip_chrome(doc: str) -> str:
    doc = DROP_BLOCK.sub(" ", doc)
    # keep it simple: remove obvious nav containers by cutting to <main> if present
    m = re.search(r"<main\b.*?</main>", doc, re.S | re.I)
    if m:
        doc = m.group(0)
    return doc


def mathml_to_tex(doc: str) -> str:
    """Replace MathJax/MathML output with the LaTeX it came from."""
    doc = re.sub(
        r'<span class="math[^"]*">\s*(.*?)\s*</span>',
        lambda m: " " + re.sub(r"<[^>]+>", "", m.group(1)) + " ",
        doc,
        flags=re.S,
    )
    return doc


BLOCK_END = re.compile(
    r"</(p|div|li|tr|h1|h2|h3|h4|h5|h6|figcaption|caption|blockquote|pre|table)>",
    re.I,
)
CELL_SEP = re.compile(r"</(td|th)>", re.I)
HEADING = re.compile(r"<(h[1-6])\b[^>]*>(.*?)</\1>", re.S | re.I)


def to_text(doc: str) -> str:
    doc = mathml_to_tex(doc)
    # mark headings so the outline survives flattening
    doc = HEADING.sub(
        lambda m: f"\n\n@@H{m.group(1)[1]}@@ " + re.sub(r"<[^>]+>", "", m.group(2)).strip() + "\n",
        doc,
    )
    doc = CELL_SEP.sub(" | ", doc)
    doc = BLOCK_END.sub("\n", doc)
    doc = re.sub(r"<br\s*/?>", "\n", doc, flags=re.I)
    doc = re.sub(r"<[^>]+>", "", doc)
    doc = html.unescape(doc)
    doc = re.sub(r"[ \t]+", " ", doc)
    doc = re.sub(r"\n\s*\n\s*\n+", "\n\n", doc)
    return "\n".join(ln.strip() for ln in doc.splitlines()).strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    cur_title, buf = "(front)", []
    for line in text.splitlines():
        m = re.match(r"@@H([1-6])@@ (.*)", line)
        if m and m.group(1) in "123":
            if buf:
                parts.append((cur_title, "\n".join(buf).strip()))
            cur_title, buf = f"H{m.group(1)}: {m.group(2)}", []
        else:
            buf.append(re.sub(r"@@H([1-6])@@ ", lambda x: "#" * int(x.group(1)) + " ", line))
    if buf:
        parts.append((cur_title, "\n".join(buf).strip()))
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--section", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    raw = Path(a.path).read_text(errors="ignore")
    text = to_text(strip_chrome(raw))
    secs = split_sections(text)

    if a.list:
        print(f"{len(secs)} sections in {a.path}")
        for i, (t, body) in enumerate(secs):
            print(f"  [{i:2d}] {len(body.splitlines()):5d} lines  {t[:90]}")
        return 0

    if a.section is not None:
        if not 0 <= a.section < len(secs):
            print(f"section {a.section} out of range (0..{len(secs)-1})", file=sys.stderr)
            return 1
        t, body = secs[a.section]
        print(f"===== [{a.section}] {t} =====\n{body}")
        return 0

    for i, (t, body) in enumerate(secs):
        print(f"\n===== [{i}] {t} =====\n{body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
