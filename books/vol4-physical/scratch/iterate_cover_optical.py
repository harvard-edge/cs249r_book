import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

tex_template = r"""\documentclass[10pt, letterpaper, twoside, openright]{scrbook}
\usepackage{geometry}
\geometry{paperwidth=8in, paperheight=10in, margin=0pt}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{graphicx}

\definecolor{ink}{HTML}{1A202C}
\definecolor{softink}{HTML}{4A5568}
\definecolor{accentcolor}{HTML}{1A4D3E}

\newlength{\mlsysTitleLen}
\newlength{\mlsysLeftM}
\newlength{\mlsysRightM}

\newcommand{\drawOptCover}[3]{% #1=ArtXShift, #2=ArtWidth, #3=Label
\newpage\thispagestyle{empty}\null
\settowidth{\mlsysTitleLen}{{\fontsize{43.5pt}{43.5pt}\selectfont Machine Learning Systems}}%
\setlength{\mlsysLeftM}{0.5\dimexpr\paperwidth - \mlsysTitleLen\relax}%
\setlength{\mlsysRightM}{\dimexpr\mlsysLeftM + \mlsysTitleLen\relax}%
\begin{tikzpicture}[remember picture,overlay]
  \fill[white] (current page.south west) rectangle (current page.north east);

  % Floating 3D sculpture artwork in upper canvas with optical xshift
  \node[anchor=center, inner sep=0pt] at ([xshift=#1, yshift=1.95in]current page.center) {%
    \includegraphics[width=#2]{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png}%
  };

  % Left-aligned Subtitle at \mlsysLeftM
  \node[anchor=south west, inner sep=0pt] at ([xshift=\mlsysLeftM, yshift=3.40in]current page.south west) {%
    {\fontsize{22pt}{26pt}\rmfamily\selectfont\color{ink}Physical AI}%
  };

  % Single-Line Main Title at \mlsysLeftM
  \node[anchor=south west, inner sep=0pt] at ([xshift=\mlsysLeftM, yshift=2.65in]current page.south west) {%
    {\fontsize{43.5pt}{47pt}\rmfamily\selectfont\color{ink}Machine Learning Systems}%
  };

  % Single-Line Right-aligned Author at \mlsysRightM
  \node[anchor=south east, inner sep=0pt] at ([xshift=\mlsysRightM, yshift=1.40in]current page.south west) {%
    {\fontsize{21pt}{25pt}\rmfamily\selectfont\color{ink}Vijay Janapa Reddi}%
  };

\end{tikzpicture}\mbox{}\vfill
}

\begin{document}
\drawOptCover{0.00in}{0.83\paperwidth}{Shift 0: Geometric Center (xshift=0.0in)}
\drawOptCover{0.18in}{0.83\paperwidth}{Shift 1: Optical Right Bias (xshift=+0.18in)}
\drawOptCover{0.28in}{0.83\paperwidth}{Shift 2: Balanced Optical Center (xshift=+0.28in) ★}
\drawOptCover{0.38in}{0.83\paperwidth}{Shift 3: Strong Right Bias (xshift=+0.38in)}
\end{document}
"""

with open(f"{scratch_dir}/test_opt_covers.tex", "w") as f:
    f.write(tex_template)

subprocess.run(["lualatex", "-interaction=nonstopmode", "test_opt_covers.tex"], cwd=scratch_dir, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "150", "test_opt_covers.pdf", f"{scratch_dir}/test_opt_out"], cwd=scratch_dir, check=True)

# Build contact sheet comparing against Ghostty reference
ref_img = Image.open("/var/folders/92/bhy05ch15kn2_msznf84g_sc0000gn/T/images/Ghostty 2026-09-02 16.03.08.png")

rendered_pages = [
    (Image.open(f"{scratch_dir}/test_opt_out-1.png"), "Shift 0: xshift=0.0in (Geometric Center)"),
    (Image.open(f"{scratch_dir}/test_opt_out-2.png"), "Shift 1: xshift=+0.18in (Slight Right)"),
    (Image.open(f"{scratch_dir}/test_opt_out-3.png"), "Shift 2: xshift=+0.28in (Optical Balance) ★"),
    (Image.open(f"{scratch_dir}/test_opt_out-4.png"), "Shift 3: xshift=+0.38in (Strong Right)"),
]

w, h = rendered_pages[0][0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 5
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 15), 'OPTICAL CENTERING ITERATIONS: Sculpture X-Shift Calibration', font=font_title, fill=(20, 20, 20))

# Paste Reference First
ref_resized = ref_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
grid.paste(ref_resized, (20, 60))
draw.rectangle([20, 60, 20 + thumb_w, 60 + thumb_h], outline=(165, 28, 48), width=3)
draw.text((25, 42), "TARGET REFERENCE (Vol 1)", font=font_label, fill=(165, 28, 48))

for idx, (img, label) in enumerate(rendered_pages):
    x = 20 + (idx + 1) * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    color = (26, 77, 62) if "★" in label else (70, 70, 70)
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=color, width=2 if "★" not in label else 3)
    draw.text((x + 5, y - 18), label, font=font_label, fill=color)

out_contact = os.path.join(brain_dir, "contact_sheet_cover_optical_centering.png")
grid.save(out_contact)
print("Optical centering contact sheet saved successfully!")
