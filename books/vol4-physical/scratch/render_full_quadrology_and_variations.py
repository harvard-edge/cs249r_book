import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

# Template for calibrated covers using the MIT Press Gold Standard baseline (sub=2.95in, title=2.30in, author=1.35in)
def render_cover_tex(name, subtitle, title, author, art_path, art_width, art_x, art_y, accent_hex="1A4D3E"):
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
\\definecolor{{accentcolor}}{{HTML}}{{{accent_hex}}}

\\newlength{{\\titlelen}}
\\newlength{{\\leftm}}
\\newlength{{\\rightm}}

\\begin{{document}}
\\thispagestyle{{empty}}
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

  % Subtitle at exact left margin of Title (MIT Press Calibrated y=2.95in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.95in]current page.south west) {{%
    {{\\fontsize{{23pt}}{{27pt}}\\rmfamily\\selectfont\\color{{ink}}{subtitle}}}%
  }};

  % Single-Line Main Title (MIT Press Calibrated y=2.30in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.30in]current page.south west) {{%
    {{\\fontsize{{43.5pt}}{{47pt}}\\rmfamily\\selectfont\\color{{ink}}{title}}}%
  }};

  % Single-Line Right-aligned Author docked under right edge of Title (MIT Press Calibrated y=1.35in)
  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=1.35in]current page.south west) {{%
    {{\\fontsize{{21.5pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}{author}}}%
  }};

\\end{{tikzpicture}}
\\newpage
\\end{{document}}
"""
    tex_path = f"{scratch_dir}/{name}.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{name}.pdf", f"{scratch_dir}/{name}_out"], cwd=scratch_dir, check=True)
    return f"{scratch_dir}/{name}_out-1.png"

# 1. Volume I: Introduction to (Single Machine Foundation)
cov_v1 = render_cover_tex(
    "quad_v1", "Introduction to", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol1.png",
    "1.02\\paperwidth", "1.10in", "1.92in", "A51C30"
)

# 2. Volume II: At Scale (Distributed Fleet / Constellation)
cov_v2 = render_cover_tex(
    "quad_v2", "At Scale", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol2.png",
    "0.98\\paperwidth", "1.10in", "1.92in", "1F407A"
)

# 3. Volume III: Agentic AI Systems (Autonomous Multi-Agent Orchestration)
agentic_art = "/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e/cover_agentic_mesh_1788358983986.jpg"
cov_v3_agentic = render_cover_tex(
    "quad_v3_agentic", "Agentic", "Machine Learning Systems", "Vijay Janapa Reddi",
    agentic_art,
    "0.94\\paperwidth", "0.75in", "1.95in", "C05621"
)

# 4. Volume IV: Physical AI (Embodied Perception, Real-Time Control & Hardware)
cov_v4_phys = render_cover_tex(
    "quad_v4_phys", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png",
    "0.96\\paperwidth", "1.15in", "1.95in", "1A4D3E"
)

# Physical AI Alternative Art 1: Articulated Mechanical Ring
phys_linkage_art = "/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e/cover_physical_linkage_1788359003600.jpg"
cov_phys_var2 = render_cover_tex(
    "quad_phys_var2", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    phys_linkage_art,
    "0.92\\paperwidth", "0.50in", "1.90in", "1A4D3E"
)

# Physical AI Alternative Art 2: Gyroscopic Multi-Axis Gimbal
cov_phys_var3 = render_cover_tex(
    "quad_phys_var3", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-gyroscopic-gimbal-art.png",
    "0.90\\paperwidth", "1.05in", "1.90in", "1A4D3E"
)

# Physical AI Alternative Art 3: Kinodynamic S-Curve
cov_phys_var4 = render_cover_tex(
    "quad_phys_var4", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-scurve-runnerup-art.png",
    "0.94\\paperwidth", "1.10in", "1.90in", "1A4D3E"
)

# Assemble Master 4-Book Tetralogy Showcase Contact Sheet
im1 = Image.open(cov_v1)
im2 = Image.open(cov_v2)
im3 = Image.open(cov_v3_agentic)
im4 = Image.open(cov_v4_phys)

w, h = im1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

grid_4 = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_4 = ImageDraw.Draw(grid_4)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw_4.text((25, 15), 'THE COMPLETE MACHINE LEARNING SYSTEMS SERIES (MIT Press Calibrated)', font=font_title, fill=(20, 20, 20))

tetralogy_panels = [
    (im1, 'VOL I: Introduction to (Harvard Crimson #A51C30)', (165, 28, 48)),
    (im2, 'VOL II: At Scale (ETH Zurich Blue #1F407A)', (31, 64, 122)),
    (im3, 'AGENTIC: Autonomous Agents (Warm Amber #C05621) ★', (192, 86, 33)),
    (im4, 'PHYSICAL AI: Sensing & Acting (Deep Emerald #1A4D3E) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(tetralogy_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_4.paste(resized, (x, y))
    draw_4.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_4.text((x + 5, y - 18), label, font=font_label, fill=col)

tetralogy_out = os.path.join(brain_dir, 'contact_sheet_mlsys_quadrology_series.png')
grid_4.save(tetralogy_out)

# Assemble Physical AI Artwork Variations Contact Sheet
im_p1 = Image.open(cov_v4_phys)
im_p2 = Image.open(cov_phys_var2)
im_p3 = Image.open(cov_phys_var3)
im_p4 = Image.open(cov_phys_var4)

grid_phys = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_p = ImageDraw.Draw(grid_phys)
draw_p.text((25, 15), 'PHYSICAL AI: Concept Artwork Variations (Harmonized MIT Press Baseline)', font=font_title, fill=(20, 20, 20))

phys_variations = [
    (im_p1, 'Concept A: Kinetic Feedback Torus Loop (Recommended ★)', (26, 77, 62)),
    (im_p2, 'Concept B: Articulated Mechanical Origami Linkage', (70, 70, 70)),
    (im_p3, 'Concept C: Gyroscopic Multi-Axis Gimbal', (70, 70, 70)),
    (im_p4, 'Concept D: Kinodynamic S-Curve Actuator Ribbon', (70, 70, 70)),
]

for idx, (img, label, col) in enumerate(phys_variations):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_phys.paste(resized, (x, y))
    draw_p.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3 if '★' in label else 2)
    draw_p.text((x + 5, y - 18), label, font=font_label, fill=col)

phys_out = os.path.join(brain_dir, 'contact_sheet_physical_ai_art_variations.png')
grid_phys.save(phys_out)

print("Quadrology and Physical AI art variations contact sheets generated successfully!")
