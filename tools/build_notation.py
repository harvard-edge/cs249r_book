#!/usr/bin/env python3
"""Emit LaTeX macros for MLSysBook from MLSysIM Notation registry."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mlsysim"))
from mlsysim.notation import Notation

# Retired names (each entry's `aliases:`) still compile, expanding to the
# canonical macro, so old chapters render while the strict check reports them.
COMPATIBILITY_ALIASES = Notation.aliases()

tex_lines = [
    "% AUTO-GENERATED NOTATION MACROS FROM MLSYSIM",
    "% Do not edit manually. Edit the registry in mlsysim/mlsysim/notation/data/ instead.",
    "",
]

for s in Notation.list():
    tex_lines.append(f"\\newcommand{{{s.macro}}}{{{s.latex}}}")

for alias, canonical in COMPATIBILITY_ALIASES.items():
    tex_lines.append(f"\\newcommand{{{alias}}}{{{canonical}}}")

# \Glyph{B}: a letter mentioned as a letter ("the book writes B for batch
# size"), not used as a quantity. Renders its argument unchanged; the strict
# check treats the argument like \text{}.
tex_lines.append("\\newcommand{\\Glyph}[1]{#1}")

macros_tex_path = PROJECT_ROOT / "books/shared/_includes/_macros.tex"
macros_tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

html_lines = [
    "<!-- AUTO-GENERATED NOTATION MACROS FROM MLSYSIM -->",
    "<div style=\"display: none;\">",
    "$$",
]
for s in Notation.list():
    html_lines.append(f"\\newcommand{{{s.macro}}}{{{s.latex}}}")
for alias, canonical in COMPATIBILITY_ALIASES.items():
    html_lines.append(f"\\newcommand{{{alias}}}{{{canonical}}}")
html_lines.append("\\newcommand{\\Glyph}[1]{#1}")
html_lines.extend(["$$", "</div>", ""])

macros_html_path = PROJECT_ROOT / "books/shared/_includes/_macros.html"
macros_html_path.write_text("\n".join(html_lines), encoding="utf-8")

print(f"Emitted {len(Notation)} LaTeX macros to _macros.tex and _macros.html.")
