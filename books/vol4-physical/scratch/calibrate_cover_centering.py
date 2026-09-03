import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

# In TeX Gyre Pagella:
# We will create a TeX template that measures the natural width of "Machine Learning Systems"
# and automatically centers the title block (so Left Margin == Right Margin),
# aligns the Subtitle to the Left Margin, and aligns the Author to the Right Margin!

tex_template = r"""\documentclass[10pt, letterpaper, oneside]{article}
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

\newlength{\titlewidth}
\newlength{\titlemargin}
\newlength{\rightmarginpos}

\newcommand{\drawSymmetricCover}[4]{% #1=TitleFontSize, #2=ArtWidth, #3=ArtYShift, #4=VariantLabel
\newpage\thispagestyle{empty}%
\settowidth{\titlewidth}{{\fontsize{#1}{#1}\selectfont Machine Learning Systems}}%
\setlength{\titlemargin}{0.5\dimexpr\paperwidth - \titlewidth\relax}%
\setlength{\rightmarginpos}{\dimexpr\paperwidth - \titlemargin\relax}%
\begin{tikzpicture}[remember picture,overlay]
  \fill[white] (current page.south west) rectangle (current page.north east);

  % Floating 3D Sculpture Art (Horizontally Centered)
  \node[anchor=center, inner sep=0pt] at ([yshift=#3]current page.center) {%
    \includegraphics[width=#2]{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png}%
  };

  % Left-aligned Subtitle at exact left margin of Title
  \node[anchor=south west, inner sep=0pt] at ([xshift=\titlemargin, yshift=3.35in]current page.south west) {%
    {\fontsize{22pt}{26pt}\rmfamily\selectfont\color{ink}Physical AI}%
  };

  % Centered Title (Starts at \titlemargin, ends at \rightmarginpos)
  \node[anchor=south west, inner sep=0pt] at ([xshift=\titlemargin, yshift=2.65in]current page.south west) {%
    {\fontsize{#1}{#1}\selectfont\color{ink}Machine Learning Systems}%
  };

  % Right-aligned Author Block at exact right margin of Title
  \node[anchor=south east, inner sep=0pt] at ([xshift=\rightmarginpos, yshift=1.40in]current page.south west) {%
    {\fontsize{20pt}{24pt}\rmfamily\selectfont\color{ink}Vijay Janapa Reddi}%
  };

  % Subtle measurement guide lines for debugging symmetry (light cyan)
  %\draw[cyan, thin] (\titlemargin, 0) -- (\titlemargin, 10in);
  %\draw[cyan, thin] (\rightmarginpos, 0) -- (\rightmarginpos, 10in);

\end{tikzpicture}\mbox{}\vfill
}

\begin{document}

% Var A: 39pt Title, 0.80\paperwidth Art
\drawSymmetricCover{39pt}{0.80\paperwidth}{1.85in}{Var A: 39pt Title (Exact Optical Center)}

% Var B: 41pt Title, 0.82\paperwidth Art
\drawSymmetricCover{41pt}{0.82\paperwidth}{1.90in}{Var B: 41pt Title (Balanced Spread)}

% Var C: 42pt Title, 0.84\paperwidth Art
\drawSymmetricCover{42pt}{0.84\paperwidth}{1.95in}{Var C: 42pt Title (Wide Canvas)}

% Var D: 40pt Title, 0.82\paperwidth Art
\drawSymmetricCover{40pt}{0.82\paperwidth}{1.90in}{Var D: 40pt Title (Reference Standard)}

\end{document}
"""

tex_file = os.path.join(scratch_dir, "cover_symmetric_test.tex")
with open(tex_file, "w") as f:
    f.write(tex_template)

subprocess.run(["lualatex", "-interaction=nonstopmode", "cover_symmetric_test.tex"], cwd=scratch_dir, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "150", "cover_symmetric_test.pdf", f"{scratch_dir}/cover_sym_out"], cwd=scratch_dir, check=True)

# Build contact sheet comparing against Ghostty reference
ref_img = Image.open("/var/folders/92/bhy05ch15kn2_msznf84g_sc0000gn/T/images/Ghostty 2026-09-02 16.03.08.png")

rendered_pages = [
    (Image.open(f"{scratch_dir}/cover_sym_out-1.png"), "Var A: 39pt (Symmetric Margins 0.088w)"),
    (Image.open(f"{scratch_dir}/cover_sym_out-2.png"), "Var B: 41pt (Symmetric Margins 0.065w) ★"),
    (Image.open(f"{scratch_dir}/cover_sym_out-3.png"), "Var C: 42pt (Symmetric Margins 0.052w)"),
    (Image.open(f"{scratch_dir}/cover_sym_out-4.png"), "Var D: 40pt (Symmetric Margins 0.076w) ★"),
]

w, h = rendered_pages[0][0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 5
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 15), 'COVER REFINEMENT: Symmetric Alignment & Exact Right-Margin Author Docking', font=font_title, fill=(20, 20, 20))

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

out_contact = os.path.join(brain_dir, "contact_sheet_cover_symmetric_refinement.png")
grid.save(out_contact)
print("Symmetric cover refinement contact sheet saved successfully!")
