#!/usr/bin/env python3
"""Generate books/shared/STRUCTURE.md from the per-volume Quarto PDF configs.

The PDF config is the canonical chapter order ("Chapter order is
canonical"). This script derives the reading order from it so the manifest can
never disagree with what actually builds. Never hand-edit STRUCTURE.md.

    python3 binder/tools/scripts/structure/gen_structure.py [--check]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
QUARTO = REPO / "books"
CONTENTS = QUARTO
OUT = CONTENTS / "shared" / "STRUCTURE.md"

VOLUMES = {
    "vol1": "Volume I: Introduction to Machine Learning Systems",
    "vol2": "Volume II: Machine Learning Systems at Scale",
    "vol3": "Volume III: Agentic Machine Learning Systems",
    "vol4": "Volume IV: Physical AI Systems",
}

QMD_RE = re.compile(r"(?<![\w/.-])(vol[0-9])/([A-Za-z0-9_./-]+\.qmd)")


def entries_for(vol: str) -> list[str]:
    """Reading order for one volume, from its PDF config, first occurrence wins."""
    cfg = QUARTO / "config" / f"_quarto-pdf-{vol}.yml"
    if not cfg.exists():
        return []
    seen: list[str] = []
    for line in cfg.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # Only render/chapter/appendix list items, not navbar hrefs or comments.
        if not stripped.startswith("- ") or stripped.startswith("#"):
            continue
        m = QMD_RE.search(stripped)
        if m and m.group(1) == vol:
            rel = m.group(2)
            if rel not in seen:
                seen.append(rel)
    return seen


def classify(rel: str) -> str:
    top = rel.split("/", 1)[0]
    if top == "frontmatter":
        return "frontmatter"
    if top == "backmatter":
        return "backmatter"
    if top == "parts":
        return "part"
    if rel == "index.qmd":
        return "index"
    return "chapter"


def render() -> str:
    lines = [
        "# Content Structure",
        "",
        "<!-- GENERATED FILE. Do not edit by hand. -->",
        "<!-- Regenerate: python3 binder/tools/scripts/structure/gen_structure.py -->",
        "",
        "Chapter order is derived from each volume's Quarto PDF config",
        "(`books/config/_quarto-pdf-<vol>.yml`), which is the canonical",
        "reading order. This file is a readable summary of that order.",
        "",
        "## Shared skeleton",
        "",
        "Every volume uses the same layout:",
        "",
        "```",
        "books/<vol>/",
        "├── index.qmd            volume home",
        "├── README.md            volume readme",
        "├── frontmatter/         *.qmd",
        "├── parts/               <name>_principles.qmd + summaries.yml",
        "├── <chapter>/           <chapter>.qmd + images/",
        "└── backmatter/          references.qmd, appendix_*.qmd, glossary/",
        "```",
        "",
        "The load-bearing invariant is `<chapter>/<chapter>.qmd`: a chapter directory",
        "and its main file share a name. `binder/cli/commands/build.py` relies on it,",
        "and `binder/tests/test_content_structure.py` enforces it.",
        "",
    ]
    for vol, title in VOLUMES.items():
        rels = entries_for(vol)
        if not rels:
            continue
        chapters = [r for r in rels if classify(r) == "chapter"]
        lines += [f"## {title} (`{vol}`)", ""]
        lines += [f"{len(chapters)} chapters. Reading order:", "", "| # | Kind | Path |", "|---:|---|---|"]
        n = 0
        for rel in rels:
            kind = classify(rel)
            if kind == "chapter":
                n += 1
                num = str(n)
            else:
                num = ""
            lines.append(f"| {num} | {kind} | `{rel}` |")
        lines.append("")
    # Exactly one trailing newline, matching the end-of-file-fixer hook.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if STRUCTURE.md is stale")
    args = ap.parse_args()
    new = render()
    if args.check:
        old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if old != new:
            print("STRUCTURE.md is stale. Regenerate with:", file=sys.stderr)
            print("  python3 binder/tools/scripts/structure/gen_structure.py", file=sys.stderr)
            return 1
        print("STRUCTURE.md is current.")
        return 0
    OUT.write_text(new, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
