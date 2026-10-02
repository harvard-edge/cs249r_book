import yaml
import re
import os

with open("books/shared/_partials/notation_registry_clean.yml") as f:
    data = yaml.safe_load(f)

macros = set()
tex_lines = []
for var in data["variables"]:
    macro = var["macro"]
    latex = var["latex"]
    if macro not in macros:
        tex_lines.append(f"\\newcommand{{{macro}}}{{{latex}}}")
        macros.add(macro)

with open("books/shared/_partials/_macros.tex", "w") as out:
    out.write("% AUTO-GENERATED NOTATION MACROS\n")
    out.write("\n".join(tex_lines) + "\n")

with open("books/shared/_partials/_macros.html", "w") as out:
    out.write("<!-- AUTO-GENERATED NOTATION MACROS -->\n")
    out.write("<div style=\"display: none;\">\n$$\n")
    out.write("\n".join(tex_lines) + "\n")
    out.write("$$\n</div>\n")

print(f"Generated {len(macros)} macros.")
