#!/usr/bin/env python3
"""Emit LaTeX macros for MLSysBook from MLSysIM Notation registry."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mlsysim"))
from mlsysim.notation import Notation

tex_lines = [
    "% AUTO-GENERATED NOTATION MACROS FROM MLSYSIM",
    "% Do not edit manually. Edit mlsysim/notation/data/symbols.yaml instead.",
    "",
]

for s in Notation.list():
    tex_lines.append(f"\\newcommand{{{s.macro}}}{{{s.latex}}}")

macros_tex_path = PROJECT_ROOT / "books/shared/_includes/_macros.tex"
macros_tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

html_lines = [
    "<!-- AUTO-GENERATED NOTATION MACROS FROM MLSYSIM -->",
    "<div style=\"display: none;\">",
    "$$",
]
for s in Notation.list():
    html_lines.append(f"\\newcommand{{{s.macro}}}{{{s.latex}}}")
html_lines.extend(["$$", "</div>", ""])

macros_html_path = PROJECT_ROOT / "books/shared/_includes/_macros.html"
macros_html_path.write_text("\n".join(html_lines), encoding="utf-8")

print(f"Emitted {len(Notation)} LaTeX macros to _macros.tex and _macros.html.")
