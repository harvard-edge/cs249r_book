from PIL import Image, ImageDraw, ImageFont
import os

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

im_c4 = Image.open(f"{brain_dir}/verified_physical_ai_candidate4_cover.png")
im_c1 = Image.open(f"{scratch_dir}/final_v4_torus_out-1.png")

# Render Candidate 5 with lowered baseline
import subprocess
tex_c5 = """\\documentclass[10pt, letterpaper, twoside, openright]{scrbook}
\\usepackage{geometry}
\\geometry{paperwidth=8in, paperheight=10in, margin=0pt}
\\usepackage{fontspec}
\\setmainfont{TeX Gyre Pagella}
\\usepackage{xcolor}
\\usepackage{tikz}
\\usepackage{graphicx}

\\definecolor{ink}{HTML}{1A202C}
\\definecolor{softink}{HTML}{4A5568}
\\definecolor{accentcolor}{HTML}{1A4D3E}

\\newlength{\\titlelen}
\\newlength{\\leftm}
\\newlength{\\rightm}

\\begin{document}
\\thispagestyle{empty}
\\null

\\settowidth{\\titlelen}{{\\fontsize{44pt}{44pt}\\selectfont Machine Learning Systems}}
\\setlength{\\leftm}{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}
\\setlength{\\rightm}{\\dimexpr\\leftm + \\titlelen\\relax}

\\begin{tikzpicture}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  \\node[anchor=center, inner sep=0pt] at ([xshift=0.55in, yshift=1.50in]current page.center) {%
    \\includegraphics[width=0.96\\paperwidth]{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-dexterous-hand-art.png}%
  };

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {%
    {\\fontsize{23.5pt}{27.5pt}\\rmfamily\\selectfont\\color{ink}Physical AI}%
  };

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {%
    {\\fontsize{44pt}{48pt}\\rmfamily\\selectfont\\color{ink}Machine Learning Systems}%
  };

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {%
    {\\fontsize{21.5pt}{25pt}\\rmfamily\\selectfont\\color{ink}Vijay Janapa Reddi}%
  };

\\end{tikzpicture}
\\newpage
\\end{document}
"""
with open(f"{scratch_dir}/cand5_lowered.tex", "w") as f:
    f.write(tex_c5)
subprocess.run(["lualatex", "-interaction=nonstopmode", "cand5_lowered.tex"], cwd=scratch_dir, check=True)
subprocess.run(["lualatex", "-interaction=nonstopmode", "cand5_lowered.tex"], cwd=scratch_dir, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "150", "cand5_lowered.pdf", f"{scratch_dir}/cand5_lowered_out"], cwd=scratch_dir, check=True)

im_c5 = Image.open(f"{scratch_dir}/cand5_lowered_out-1.png")

w, h = im_c4.size
thumb_w = 360
thumb_h = int(h * (thumb_w / w))

grid = Image.new('RGB', (thumb_w * 3 + 80, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 12)

draw.text((25, 18), 'FINAL 3 CONTENDERS: Cand 4 (Gimbal) vs Cand 1 (Torus) vs Cand 5 (Hand)', font=font_title, fill=(20, 20, 20))

panels = [
    (im_c4, '1. Candidate 4: Gyroscopic Multi-Axis Gimbal (Recommended ★★★)', (26, 77, 62)),
    (im_c1, '2. Candidate 1: Kinetic Feedback Torus Loop (Runner-up ★★)', (40, 90, 80)),
    (im_c5, '3. Candidate 5: Dexterous Robotic Hand & Tactile Array (★)', (70, 70, 70)),
]

for idx, (img, label, col) in enumerate(panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3 if '★★★' in label else 2)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

out_file = os.path.join(brain_dir, 'contact_sheet_user_final_three_comparison.png')
grid.save(out_file)
print("Final 3 comparison contact sheet saved!")
