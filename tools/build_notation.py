#!/usr/bin/env python3
import sys
from pathlib import Path

# Always resolve paths relative to the project root (parent of tools dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to sys.path so we can import mlsysim
sys.path.insert(0, str(PROJECT_ROOT / "mlsysim"))
from mlsysim.notation import SYMBOLS

# Generate formatted LaTeX macros
macros = set()
tex_lines = []
data_variables = list(SYMBOLS._all_symbols.values())

for var in data_variables:
    macro = var.macro
    latex = var.latex
    if macro not in macros:
        tex_lines.append(f"\\newcommand{{{macro}}}{{{latex}}}")
        macros.add(macro)

with open(PROJECT_ROOT / "books/shared/_includes/_macros.tex", "w") as out:
    out.write("% AUTO-GENERATED NOTATION MACROS\n")
    out.write("\n".join(tex_lines) + "\n")

with open(PROJECT_ROOT / "books/shared/_includes/_macros.html", "w") as out:
    out.write("<!-- AUTO-GENERATED NOTATION MACROS -->\n")
    out.write("<div style=\"display: none;\">\n$$\n")
    out.write("\n".join(tex_lines) + "\n")
    out.write("$$\n</div>\n")

print(f"Generated {len(macros)} macros into _macros.tex and _macros.html.")
