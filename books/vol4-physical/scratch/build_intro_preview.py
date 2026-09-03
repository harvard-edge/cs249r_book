import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
book_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

with open(os.path.join(book_dir, 'tex/theme-colors-physical.tex'), 'r', encoding='utf-8') as f:
    theme_colors = f.read()

with open(os.path.join(book_dir, 'tex/preamble.tex'), 'r', encoding='utf-8') as f:
    preamble = f.read()

with open(os.path.join(book_dir, 'tex/cover.tex'), 'r', encoding='utf-8') as f:
    cover = f.read()

# Fix cover image path to absolute
cover = cover.replace('assets/covers/', '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/')

with open(os.path.join(book_dir, 'Physical-AI.tex'), 'r', encoding='utf-8') as f:
    full_tex = f.read()

frontmatter_start = full_tex.find('\\bookmarksetup{startatroot}\n\n\\chapter*{Preface}')
if frontmatter_start == -1:
    frontmatter_start = full_tex.find('\\chapter*{Preface}')

ch2_start = full_tex.find('\\chapter{\\texorpdfstring{The Physical')

extracted_content = full_tex[frontmatter_start:ch2_start]

# Fix figure paths to absolute
extracted_content = extracted_content.replace('index_files/mediabag/chapters/', '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/chapters/')
extracted_content = extracted_content.replace('index_files/mediabag/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/chapters/', '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/chapters/')
extracted_content = extracted_content.replace('chapters/', '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/chapters/')

doc_template = r'''\documentclass[
  10pt,
  letterpaper,
  twoside,
  openright,
  numbers=noenddot,
  headings=standardclasses
]{scrbook}

\usepackage{geometry}
\geometry{
  paperwidth=8in,
  paperheight=10in,
  inner=0.875in,
  outer=1.75in,
  top=0.875in,
  bottom=0.875in,
  marginparwidth=1.25in,
  marginparsep=0.25in,
  headheight=14pt,
  headsep=0.25in,
  footskip=0.4in
}

\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{calc, positioning}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{tcolorbox}
\tcbuselibrary{skins, breakable}
\usepackage{hyperref}
\usepackage{bookmark}
\usepackage{sidenotes}
\usepackage{caption}

''' + theme_colors + '\n' + preamble + '\n' + cover + r'''

\begin{document}
\frontmatter
\maketitle

\tableofcontents

''' + extracted_content + r'''

\end{document}
'''

tex_path = os.path.join(scratch_dir, 'physical_intro_preview.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(doc_template)

print('Running LuaLaTeX pass 1...')
subprocess.run(['lualatex', '-interaction=nonstopmode', 'physical_intro_preview.tex'], cwd=scratch_dir, stdout=subprocess.DEVNULL)
print('Running LuaLaTeX pass 2...')
subprocess.run(['lualatex', '-interaction=nonstopmode', 'physical_intro_preview.tex'], cwd=scratch_dir, stdout=subprocess.DEVNULL)

subprocess.run(['pdftoppm', '-png', '-r', '150', 'physical_intro_preview.pdf', 'preview_page'], cwd=scratch_dir)

# Assemble updated 2x4 walkthrough
page_indices = [
    ('preview_page-01.png', '1. Book Cover (Pure Minimalist Symmetry)'),
    ('preview_page-02.png', '2. Interior Title Page (MIT Press Typography)'),
    ('preview_page-05.png', '3. Table of Contents (Canonical MLSysBook Layout)'),
    ('preview_page-11.png', '4. Preface Opener (Lettrine Drop Cap & Accents)'),
    ('preview_page-23.png', '5. Chapter 1: The Causal Boundary (Opener & Locator)'),
    ('preview_page-24.png', '6. Chapter 1: Section 1.1 Core Systems Narrative'),
    ('preview_page-25.png', '7. Chapter 1: Section 1.5 The Four Pillars Table'),
    ('preview_page-27.png', '8. Chapter 1: Three Archetypes & Multi-Rate Coupling'),
]

imgs = []
for fname, label in page_indices:
    p = os.path.join(scratch_dir, fname)
    if os.path.exists(p):
        imgs.append((Image.open(p), label))

w, h = imgs[0][0].size
thumb_w, thumb_h = int(w * 0.40), int(h * 0.40)

cols, rows = 4, 2
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h * rows + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 28)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)

draw.text((25, 15), 'PHYSICAL AI SYSTEMS — Complete Walkthrough: Cover, Frontmatter, Preface & Chapter 1', font=font_title, fill=(20, 20, 20))

for idx, (img, label) in enumerate(imgs):
    r = idx // cols
    c = idx % cols
    x = 20 + c * (thumb_w + 20)
    y = 60 + r * (thumb_h + 30)
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(26, 77, 62), width=2)
    draw.text((x + 5, y - 18), label, font=font_label, fill=(26, 77, 62))

grid.save(os.path.join(brain_dir, 'contact_sheet_intro_walkthrough.png'))
print('Updated walkthrough contact sheet saved!')
