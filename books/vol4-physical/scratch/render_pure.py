import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'

colors = [
    ('emerald', '1A4D3E', 'A. Deep Emerald Pine (#1A4D3E) — 180° Complement to Crimson', (26, 77, 62)),
    ('indigo', '3E2A68', 'B. Imperial Indigo (#3E2A68) — Deep Royal Navy-Purple', (62, 42, 104)),
    ('obsidian', '133B4F', 'C. Obsidian Steel (#133B4F) — CNC Industrial Blue-Grey', (19, 59, 79)),
    ('petrol', '007A87', 'D. ETH Petrol / Teal (#007A87) — Original Swiss Technical', (0, 122, 135)),
]

tex_template = r'''\documentclass{scrbook}
\usepackage{geometry}
\geometry{paperwidth=8in, paperheight=10in, margin=0in}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{calc, positioning}
\setmainfont{STIX Two Text}
\setsansfont{Avenir Next}

\definecolor{ink}{HTML}{1A202C}
\definecolor{softink}{HTML}{4A5568}
\definecolor{theme}{HTML}{HEXVAL}

\begin{document}
\pagestyle{empty}

\noindent\null
\begin{tikzpicture}[remember picture, overlay]
  \node[anchor=north, inner sep=0pt] at ([yshift=-0.65in]current page.north) {%
    \includegraphics[width=6.0in, height=5.5in, keepaspectratio]{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png}%
  };
  \node[anchor=south west, inner sep=0pt] at ([xshift=0.875in, yshift=3.35in]current page.south west) {%
    {\fontsize{26pt}{30pt}\rmfamily\selectfont\color{softink}Physical AI}%
  };
  \node[anchor=north west, inner sep=0pt] at ([xshift=0.875in, yshift=3.15in]current page.south west) {%
    \linespread{1.05}\selectfont%
    {\fontsize{50pt}{56pt}\rmfamily\selectfont\color{ink}%
     \begin{tabular}{@{}l@{}}
       \fontsize{50pt}{56pt}\rmfamily\selectfont Machine Learning\\
       \fontsize{50pt}{56pt}\rmfamily\selectfont Systems
     \end{tabular}%
    }%
  };
  \node[anchor=south east, inner sep=0pt] at ([xshift=-0.875in, yshift=0.85in]current page.south east) {%
    {\fontsize{18pt}{22pt}\rmfamily\selectfont\color{ink}%
     \begin{tabular}{@{}r@{}}
       Vijay\\ Janapa Reddi
     \end{tabular}%
    }%
  };
\end{tikzpicture}
\end{document}
'''

imgs = []
for tag, hexval, label, col in colors:
    tex_src = tex_template.replace('HEXVAL', hexval)
    tex_path = os.path.join(scratch_dir, f'pure_cov_{tag}.tex')
    with open(tex_path, 'w') as f:
        f.write(tex_src)
    subprocess.run(['lualatex', '-interaction=nonstopmode', f'pure_cov_{tag}.tex'], cwd=scratch_dir, stdout=subprocess.DEVNULL)
    subprocess.run(['lualatex', '-interaction=nonstopmode', f'pure_cov_{tag}.tex'], cwd=scratch_dir, stdout=subprocess.DEVNULL)
    subprocess.run(['pdftoppm', '-png', '-r', '150', '-f', '1', '-l', '1', f'pure_cov_{tag}.pdf', f'pure_img_{tag}'], cwd=scratch_dir)
    img_path = os.path.join(scratch_dir, f'pure_img_{tag}-1.png')
    img = Image.open(img_path)
    imgs.append((img, label, col))

# Create 2x2 contact sheet
w, h = imgs[0][0].size
thumb_w, thumb_h = int(w * 0.45), int(h * 0.45)
grid = Image.new('RGB', (thumb_w * 2 + 50, thumb_h * 2 + 100), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 26)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 14)

draw.text((20, 15), 'PURE TRILOGY MINIMALIST COVERS (Exact Volume 1 & 2 Sizing)', font=font_title, fill=(20, 20, 20))

for idx, (img, label, col) in enumerate(imgs):
    r = idx // 2
    c = idx % 2
    x = 20 + c * (thumb_w + 15)
    y = 55 + r * (thumb_h + 20)
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

grid.save('/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e/contact_sheet_pure_minimal_covers.png')
print('Pure minimal covers sheet saved successfully!')
