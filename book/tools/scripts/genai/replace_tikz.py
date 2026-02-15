
import os

filepath = "book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd"

with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
in_fig_div = False
in_tikz_block = False
replaced = False

for line in lines:
    if "::: {#fig-collective-comm" in line:
        in_fig_div = True
        new_lines.append(line)
        continue
    
    if in_fig_div:
        if "```{.tikz}" in line:
            in_tikz_block = True
            if not replaced:
                new_lines.append("![](images/png/comm_primitives.png)\n")
                replaced = True
            continue
        
        if in_tikz_block:
            if "```" in line.strip() and line.strip() == "```":
                in_tikz_block = False
            continue
        
        if line.strip() == ":::":
            in_fig_div = False
            new_lines.append(line)
            continue
            
        new_lines.append(line)
    else:
        new_lines.append(line)

with open(filepath, "w") as f:
    f.writelines(new_lines)

print("Replaced TikZ block with image reference.")
