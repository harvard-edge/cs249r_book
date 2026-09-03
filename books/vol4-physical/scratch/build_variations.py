import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

scratch_dir = "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch"
base_dir = "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book"

# Read the full tex file generated during quarto render
with open(os.path.join(base_dir, "Physical-AI.tex"), "r", encoding="utf-8") as f:
    full_tex = f.read()

# Extract preamble and chapter 1 + frontmatter content up to end of chapter 1
# Chapter 1 ends before Chapter 2 \chapter{The Physical Body}
ch1_cut = full_tex.split(r"\chapter{The Physical Body}")[0]
ch1_content = ch1_cut + r"\end{document}"

variants = [
    ("var1_burnt_copper", "B34726", "Burnt Copper / Kinetic Rust", True),
    ("var2_imperial_indigo", "3E2A68", "Deep Imperial Indigo", True),
    ("var3_emerald_pine", "1A4D3E", "Emerald Pine / Forest", True),
    ("var4_obsidian_steel", "133B4F", "Obsidian Steel / Deep Marine", True),
    ("var5_eth_teal", "007A87", "Original ETH Teal / Petrol", True),
    ("var6_copper_pure", "B34726", "Burnt Copper (Trilogy Pure Minimal)", False),
]

for name, hex_color, label, with_subtitle in variants:
    tex_path = os.path.join(scratch_dir, f"{name}.tex")
    pdf_path = os.path.join(scratch_dir, f"{name}.pdf")
    
    # Replace accentcolor definition
    custom_tex = ch1_content.replace(
        r"\definecolor{accentcolor}{HTML}{007A87}",
        rf"\definecolor{{accentcolor}}{{HTML}}{{{hex_color}}}"
    )
    
    # Replace cover subtitle if requested
    if with_subtitle:
        target_systems = r"\fontsize{50pt}{56pt}\rmfamily\selectfont Systems"
        replacement_systems = r"\fontsize{50pt}{56pt}\rmfamily\selectfont Systems\\[0.40em]{\fontsize{13.5pt}{17pt}\rmfamily\itshape\color{accentcolor}That Sense and Act}"
        custom_tex = custom_tex.replace(target_systems, replacement_systems, 1)
        
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(custom_tex)
        
    print(f"Compiling {name} ({label})...")
    cmd = ["lualatex", "-interaction=nonstopmode", f"{name}.tex"]
    subprocess.run(cmd, cwd=scratch_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Run second pass for hyperref and geometry
    subprocess.run(cmd, cwd=scratch_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Extract Cover (p1), Preface (p11), and Chapter 1 opener (p24)
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1", f"{name}.pdf", f"{name}_cover"], cwd=scratch_dir)
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "11", "-l", "11", f"{name}.pdf", f"{name}_preface"], cwd=scratch_dir)
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "24", "-l", "24", f"{name}.pdf", f"{name}_ch1"], cwd=scratch_dir)

print("All variations compiled!")
