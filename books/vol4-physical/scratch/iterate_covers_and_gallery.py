import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

# Template for rendering individual covers
def render_cover_tex(name, subtitle, title, author, art_path, art_width, art_x, art_y, sub_y, title_y, author_y):
    tex = f"""\\documentclass[10pt, letterpaper, twoside, openright]{{scrbook}}
\\usepackage{{geometry}}
\\geometry{{paperwidth=8in, paperheight=10in, margin=0pt}}
\\usepackage{{fontspec}}
\\setmainfont{{TeX Gyre Pagella}}
\\usepackage{{xcolor}}
\\usepackage{{tikz}}
\\usepackage{{graphicx}}

\\definecolor{{ink}}{{HTML}}{{1A202C}}
\\definecolor{{softink}}{{HTML}}{{4A5568}}
\\definecolor{{accentcolor}}{{HTML}}{{1A4D3E}}

\\newlength{{\\titlelen}}
\\newlength{{\\leftm}}
\\newlength{{\\rightm}}

\\begin{{document}}
\\thispagestyle{{empty}}%
\\null
\\settowidth{{\\titlelen}}{{\\fontsize{{43.5pt}}{{43.5pt}}\\selectfont {title}}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  % Artwork node
  \\node[anchor=center, inner sep=0pt] at ([xshift={art_x}, yshift={art_y}]current page.center) {{%
    \\includegraphics[width={art_width}]{{{art_path}}}%
  }};

  % Subtitle at exact left margin of Title
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift={sub_y}]current page.south west) {{%
    {{\\fontsize{{22pt}}{{26pt}}\\rmfamily\\selectfont\\color{{ink}}{subtitle}}}%
  }};

  % Single-Line Main Title
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift={title_y}]current page.south west) {{%
    {{\\fontsize{{43.5pt}}{{47pt}}\\rmfamily\\selectfont\\color{{ink}}{title}}}%
  }};

  % Single-Line Right-aligned Author docked under right edge of Title
  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift={author_y}]current page.south west) {{%
    {{\\fontsize{{21pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}{author}}}%
  }};

\\end{{tikzpicture}}%
\\newpage
\\end{{document}}
"""
    tex_path = f"{scratch_dir}/{name}.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{name}.pdf", f"{scratch_dir}/{name}_out"], cwd=scratch_dir, check=True)
    return f"{scratch_dir}/{name}_out-1.png"

# Render Volume 1 Cover
vol1_img = render_cover_tex(
    "vol1_cover_render",
    "Introduction to",
    "Machine Learning Systems",
    "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol1.png",
    "0.92\\paperwidth", "1.24in", "2.15in", "3.95in", "3.30in", "1.70in"
)

# Render Volume 2 Cover
vol2_img = render_cover_tex(
    "vol2_cover_render",
    "At Scale",
    "Machine Learning Systems",
    "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol2.png",
    "0.90\\paperwidth", "1.24in", "2.15in", "3.95in", "3.30in", "1.70in"
)

# Render 4 Variations of Volume 3 Physical AI Cover:
# Option A: Large Right-Offset Artwork (Matching Vol 1 & Vol 2)
v3_optA = render_cover_tex(
    "v3_optA",
    "Physical AI",
    "Machine Learning Systems",
    "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png",
    "0.92\\paperwidth", "1.15in", "2.15in", "3.95in", "3.30in", "1.70in"
)

# Option B: Moderate Right-Offset Artwork
v3_optB = render_cover_tex(
    "v3_optB",
    "Physical AI",
    "Machine Learning Systems",
    "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png",
    "0.86\\paperwidth", "0.60in", "2.10in", "3.95in", "3.30in", "1.70in"
)

# Option C: Centered Large Artwork
v3_optC = render_cover_tex(
    "v3_optC",
    "Physical AI",
    "Machine Learning Systems",
    "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png",
    "0.84\\paperwidth", "0.15in", "2.05in", "3.95in", "3.30in", "1.70in"
)

# Option D: Extra Large Bleed Artwork
v3_optD = render_cover_tex(
    "v3_optD",
    "Physical AI",
    "Machine Learning Systems",
    "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png",
    "0.96\\paperwidth", "1.30in", "2.20in", "3.95in", "3.30in", "1.70in"
)

# Build 3-Book Trilogy Showcase Contact Sheet
im_v1 = Image.open(vol1_img)
im_v2 = Image.open(vol2_img)
im_v3 = Image.open(v3_optA)

w, h = im_v1.size
thumb_w = 360
thumb_h = int(h * (thumb_w / w))

trilogy_grid = Image.new('RGB', (thumb_w * 3 + 80, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(trilogy_grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 12)

draw.text((30, 15), 'THE MACHINE LEARNING SYSTEMS TRILOGY: Harmonized Cover Architecture', font=font_title, fill=(20, 20, 20))

t_panels = [
    (im_v1, 'VOLUME I: Introduction to (Harvard Crimson)', (165, 28, 48)),
    (im_v2, 'VOLUME II: At Scale (ETH Zurich Blue)', (31, 64, 122)),
    (im_v3, 'VOLUME III: Physical AI (Deep Emerald Pine)', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(t_panels):
    x = 25 + idx * (thumb_w + 15)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    trilogy_grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

trilogy_file = os.path.join(brain_dir, 'contact_sheet_trilogy_covers.png')
trilogy_grid.save(trilogy_file)

# Build Physical AI Artwork Position Iterations Contact Sheet
im_optA = Image.open(v3_optA)
im_optB = Image.open(v3_optB)
im_optC = Image.open(v3_optC)
im_optD = Image.open(v3_optD)

v3_grid = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_v3 = ImageDraw.Draw(v3_grid)
draw_v3.text((30, 15), 'PHYSICAL AI COVER: Artwork Scaling & Right-Offset Iterations (At y=3.95in Baseline)', font=font_title, fill=(20, 20, 20))

v3_panels = [
    (im_optA, 'Option A: 0.92w Art, Right Shift +1.15in (Matches Vol 1 Offset) ★', (26, 77, 62)),
    (im_optB, 'Option B: 0.86w Art, Moderate Right Shift +0.60in', (70, 70, 70)),
    (im_optC, 'Option C: 0.84w Art, Centered +0.15in', (70, 70, 70)),
    (im_optD, 'Option D: 0.96w Art, Large Bleed Right Shift +1.30in', (70, 70, 70)),
]

for idx, (img, label, col) in enumerate(v3_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    v3_grid.paste(resized, (x, y))
    draw_v3.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3 if '★' in label else 2)
    draw_v3.text((x + 5, y - 18), label, font=font_label, fill=col)

v3_file = os.path.join(brain_dir, 'contact_sheet_physical_cover_options.png')
v3_grid.save(v3_file)

print('Trilogy covers and Physical AI options contact sheets generated successfully!')
