import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'

# Master Lowered Baseline Template (sub=2.40in, title=1.62in, author=0.78in)
def render_cover(name, subtitle, title, author, art_path, art_width, art_x, art_y, accent_hex="1A4D3E"):
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

\\settowidth{{\\titlelen}}{{\\fontsize{{44pt}}{{44pt}}\\selectfont {title}}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  % Artwork node
  \\node[anchor=center, inner sep=0pt] at ([xshift={art_x}, yshift={art_y}]current page.center) {{%
    \\includegraphics[width={art_width}]{{{art_path}}}%
  }};

  % Subtitle (MIT Press Exact Match y=2.40in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {{%
    {{\\fontsize{{23.5pt}}{{27.5pt}}\\rmfamily\\selectfont\\color{{ink}}{subtitle}}}%
  }};

  % Single-Line Main Title (MIT Press Exact Match y=1.62in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {{%
    {{\\fontsize{{44pt}}{{48pt}}\\rmfamily\\selectfont\\color{{ink}}{title}}}%
  }};

  % Single-Line Right-aligned Author (MIT Press Exact Match y=0.78in)
  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {{%
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

# 1. Volume I: Introduction to (Harvard Crimson)
p_v1 = render_cover(
    "final_v1", "Introduction to", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol1.png",
    "1.06\\paperwidth", "1.15in", "1.50in", "A51C30"
)

# 2. Volume II: Scaling (ETH Zurich Blue)
p_v2 = render_cover(
    "final_v2", "Scaling", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol2.png",
    "1.02\\paperwidth", "1.15in", "1.50in", "1F407A"
)

# 3. Volume III: Agentic (Warm Amber / Bronze)
art_agentic = f"{brain_dir}/cover_agentic_mesh_1788358983986.jpg"
p_v3 = render_cover(
    "final_v3", "Agentic", "Machine Learning Systems", "Vijay Janapa Reddi",
    art_agentic,
    "0.98\\paperwidth", "0.75in", "1.52in", "C05621"
)

# 4. Volume IV: Physical AI - Concept 1 (Möbius Ouroboros Loop) ★
art_mobius = f"{brain_dir}/cover_phys_mobius_ouroboros_1788361216490.jpg"
p_v4_mobius = render_cover(
    "final_v4_mobius", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    art_mobius,
    "0.98\\paperwidth", "0.60in", "1.50in", "1A4D3E"
)

# 4b. Volume IV: Physical AI - Concept 2 (Robotic Mechanical Flower)
art_flower = f"{brain_dir}/cover_phys_mechanical_flower_1788361264670.jpg"
p_v4_flower = render_cover(
    "final_v4_flower", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    art_flower,
    "0.98\\paperwidth", "0.75in", "1.50in", "1A4D3E"
)

# 4c. Volume IV: Physical AI - Concept 3 (Kinetic Torus Loop)
art_torus = f"{assets_dir}/cover-physical-loop-recommended-art.png"
p_v4_torus = render_cover(
    "final_v4_torus", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi",
    art_torus,
    "1.02\\paperwidth", "1.15in", "1.52in", "1A4D3E"
)

# Assemble Master 4-Volume Tetralogy Showcase Contact Sheet
im1 = Image.open(p_v1)
im2 = Image.open(p_v2)
im3 = Image.open(p_v3)
im4 = Image.open(p_v4_mobius)

w, h = im1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

grid = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'THE MACHINE LEARNING SYSTEMS SERIES (MIT Press Exact Baseline)', font=font_title, fill=(20, 20, 20))

tetralogy_panels = [
    (im1, 'VOL I: Introduction to (Harvard Crimson #A51C30)', (165, 28, 48)),
    (im2, 'VOL II: Scaling (ETH Zurich Blue #1F407A)', (31, 64, 122)),
    (im3, 'AGENTIC: Autonomous Agents (Warm Amber #C05621)', (192, 86, 33)),
    (im4, 'PHYSICAL AI: Sensing & Acting (Deep Emerald #1A4D3E) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(tetralogy_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

out_master = os.path.join(brain_dir, 'contact_sheet_master_quadrology_lowered_baseline.png')
grid.save(out_master)

# Assemble Physical AI Top 3 Artworks on Lowered Baseline Contact Sheet
im_p_mobius = Image.open(p_v4_mobius)
im_p_flower = Image.open(p_v4_flower)
im_p_torus = Image.open(p_v4_torus)

grid_p = Image.new('RGB', (thumb_w * 3 + 80, thumb_h + 120), (245, 245, 247))
draw_p = ImageDraw.Draw(grid_p)
draw_p.text((25, 18), 'PHYSICAL AI: TOP 3 ARTWORKS ON MIT PRESS LOWERED BASELINE', font=font_title, fill=(20, 20, 20))

p_panels = [
    (im_p_mobius, 'Concept 1: Möbius Ouroboros (Tendon & Sensor Loop) ★★★', (26, 77, 62)),
    (im_p_flower, 'Concept 2: Robotic Mechanical Flower (Vol 1 Twin) ★★', (40, 90, 80)),
    (im_p_torus, 'Concept 3: Kinetic Torus Loop (Continuous Ring) ★', (70, 70, 70)),
]

for idx, (img, label, col) in enumerate(p_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_p.paste(resized, (x, y))
    draw_p.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3 if '★★★' in label else 2)
    draw_p.text((x + 5, y - 18), label, font=font_label, fill=col)

out_p = os.path.join(brain_dir, 'contact_sheet_physical_ai_top3_lowered_baseline.png')
grid_p.save(out_p)

print("Master quadrology and Physical AI top 3 contact sheets generated successfully!")
