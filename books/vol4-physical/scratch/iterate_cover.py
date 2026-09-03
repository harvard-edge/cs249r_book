import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

variations = [
    {
        "id": "var1",
        "title_size": "44pt", "title_lead": "48pt",
        "sub_size": "22pt", "sub_lead": "26pt",
        "author_size": "20pt", "author_lead": "24pt",
        "art_width": "0.80\\paperwidth", "art_y": "1.85in",
        "sub_y": "0.380\\paperwidth",
        "title_y": "0.295\\paperwidth",
        "author_y": "0.150\\paperwidth",
        "left_m": "0.080\\paperwidth", "right_m": "0.920\\paperwidth",
        "desc": "Var 1: 44pt Title (Margins 0.08/0.92)"
    },
    {
        "id": "var2",
        "title_size": "46pt", "title_lead": "50pt",
        "sub_size": "22pt", "sub_lead": "26pt",
        "author_size": "21pt", "author_lead": "25pt",
        "art_width": "0.82\\paperwidth", "art_y": "1.95in",
        "sub_y": "0.385\\paperwidth",
        "title_y": "0.295\\paperwidth",
        "author_y": "0.140\\paperwidth",
        "left_m": "0.075\\paperwidth", "right_m": "0.925\\paperwidth",
        "desc": "Var 2: 46pt Title (Margins 0.075/0.925) ★"
    },
    {
        "id": "var3",
        "title_size": "48pt", "title_lead": "52pt",
        "sub_size": "23pt", "sub_lead": "27pt",
        "author_size": "22pt", "author_lead": "26pt",
        "art_width": "0.85\\paperwidth", "art_y": "2.05in",
        "sub_y": "0.395\\paperwidth",
        "title_y": "0.300\\paperwidth",
        "author_y": "0.135\\paperwidth",
        "left_m": "0.070\\paperwidth", "right_m": "0.930\\paperwidth",
        "desc": "Var 3: 48pt Full-Span Title, 0.85w Art"
    },
    {
        "id": "var4",
        "title_size": "45pt", "title_lead": "49pt",
        "sub_size": "22pt", "sub_lead": "26pt",
        "author_size": "21pt", "author_lead": "25pt",
        "art_width": "0.78\\paperwidth", "art_y": "1.80in",
        "sub_y": "0.375\\paperwidth",
        "title_y": "0.290\\paperwidth",
        "author_y": "0.145\\paperwidth",
        "left_m": "0.082\\paperwidth", "right_m": "0.918\\paperwidth",
        "desc": "Var 4: 45pt Title, 0.78w Art, Compact Margins"
    }
]

tex_content = r"""\documentclass[10pt, letterpaper, oneside]{article}
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

\newcommand{\drawCover}[9]{%
\newpage\thispagestyle{empty}%
\begin{tikzpicture}[remember picture,overlay]
  \fill[white] (current page.south west) rectangle (current page.north east);

  % Floating Art
  \node[anchor=center, inner sep=0pt] at ([yshift=#2]current page.center) {%
    \includegraphics[width=#1]{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png}%
  };

  % Left-aligned Subtitle
  \node[anchor=south west, inner sep=0pt] at ([xshift=#6, yshift=#3]current page.south west) {%
    {\fontsize{22pt}{26pt}\rmfamily\selectfont\color{ink}Physical AI}%
  };

  % Single-Line Main Title (Left aligned)
  \node[anchor=south west, inner sep=0pt] at ([xshift=#6, yshift=#4]current page.south west) {%
    {\fontsize{#8}{#9}\rmfamily\selectfont\color{ink}Machine Learning Systems}%
  };

  % Single-Line Author (Right aligned to #7)
  \node[anchor=south east, inner sep=0pt] at ([xshift=#7, yshift=#5]current page.south west) {%
    {\fontsize{21pt}{25pt}\rmfamily\selectfont\color{ink}Vijay Janapa Reddi}%
  };

\end{tikzpicture}\mbox{}\vfill
}

\begin{document}
"""

for v in variations:
    tex_content += f"\\drawCover{{{v['art_width']}}}{{{v['art_y']}}}{{{v['sub_y']}}}{{{v['title_y']}}}{{{v['author_y']}}}{{{v['left_m']}}}{{{v['right_m']}}}{{{v['title_size']}}}{{{v['title_lead']}}}\n"

tex_content += r"\end{document}"

tex_file = os.path.join(scratch_dir, "cover_fast_test.tex")
with open(tex_file, "w") as f:
    f.write(tex_content)

subprocess.run(["lualatex", "-interaction=nonstopmode", "cover_fast_test.tex"], cwd=scratch_dir, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "150", "cover_fast_test.pdf", f"{scratch_dir}/cover_fast_out"], cwd=scratch_dir, check=True)

# Build contact sheet comparing against Ghostty reference
ref_img = Image.open("/var/folders/92/bhy05ch15kn2_msznf84g_sc0000gn/T/images/Ghostty 2026-09-02 16.03.08.png")

rendered_pages = [
    (Image.open(f"{scratch_dir}/cover_fast_out-1.png"), "Var 1: 44pt Title (Margins 0.08/0.92)"),
    (Image.open(f"{scratch_dir}/cover_fast_out-2.png"), "Var 2: 46pt Title (Margins 0.075/0.925) ★"),
    (Image.open(f"{scratch_dir}/cover_fast_out-3.png"), "Var 3: 48pt Title (Margins 0.070/0.930)"),
    (Image.open(f"{scratch_dir}/cover_fast_out-4.png"), "Var 4: 45pt Title (Margins 0.082/0.918)"),
]

w, h = rendered_pages[0][0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 5
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 15), 'COVER ITERATIONS: Calibrated to Reference Geometry & Single-Line Layout', font=font_title, fill=(20, 20, 20))

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

out_contact = os.path.join(brain_dir, "contact_sheet_cover_fast_iterations.png")
grid.save(out_contact)
print("Cover fast iterations contact sheet saved successfully!")
