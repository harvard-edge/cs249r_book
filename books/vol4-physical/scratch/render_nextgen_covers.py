import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'

# New Next-Gen Artworks
art_mobius = f"{brain_dir}/cover_phys_mobius_ouroboros_1788361216490.jpg"
art_gyro = f"{brain_dir}/cover_phys_dual_gyro_1788361241925.jpg"
art_mech_flower = f"{brain_dir}/cover_phys_mechanical_flower_1788361264670.jpg"
art_loop_v2 = f"{brain_dir}/cover_phys_kinetic_loop_v2_1788361285892.jpg"
art_orig_torus = f"{assets_dir}/cover-physical-loop-recommended-art.png"
art_orig_gimbal = f"{assets_dir}/cover-physical-gyroscopic-gimbal-art.png"

nextgen_list = [
    {
        "id": "cover_ng1_mobius",
        "label": "1. Möbius Ouroboros (Pleated Origami + Titanium Tendons) ★",
        "art": art_mobius,
        "width": "0.94\\paperwidth", "x": "0.55in", "y": "1.92in"
    },
    {
        "id": "cover_ng2_flower",
        "label": "2. Robotic Mechanical Flower (Vol 1 Evolutionary Twin) ★",
        "art": art_mech_flower,
        "width": "0.94\\paperwidth", "x": "0.75in", "y": "1.92in"
    },
    {
        "id": "cover_ng3_loopv2",
        "label": "3. Precision Bearing Torus & Emerald Core",
        "art": art_loop_v2,
        "width": "0.92\\paperwidth", "x": "0.50in", "y": "1.90in"
    },
    {
        "id": "cover_ng4_gyro",
        "label": "4. Spherical Dual-Rate Gyro Gimbal",
        "art": art_gyro,
        "width": "0.90\\paperwidth", "x": "0.10in", "y": "1.90in"
    },
    {
        "id": "cover_ng5_orig_torus",
        "label": "5. Previous Cand 1: Kinetic Torus Loop",
        "art": art_orig_torus,
        "width": "0.96\\paperwidth", "x": "1.15in", "y": "1.95in"
    },
    {
        "id": "cover_ng6_orig_gimbal",
        "label": "6. Previous Cand 4: Gyroscopic Multi-Axis Gimbal",
        "art": art_orig_gimbal,
        "width": "0.90\\paperwidth", "x": "1.05in", "y": "1.90in"
    }
]

def render_cover(c):
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
\\thispagestyle{{empty}}
\\null

\\settowidth{{\\titlelen}}{{\\fontsize{{43.5pt}}{{43.5pt}}\\selectfont Machine Learning Systems}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  \\node[anchor=center, inner sep=0pt] at ([xshift={c['x']}, yshift={c['y']}]current page.center) {{%
    \\includegraphics[width={c['width']}]{{{c['art']}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.95in]current page.south west) {{%
    {{\\fontsize{{23pt}}{{27pt}}\\rmfamily\\selectfont\\color{{ink}}Physical AI}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.30in]current page.south west) {{%
    {{\\fontsize{{43.5pt}}{{47pt}}\\rmfamily\\selectfont\\color{{ink}}Machine Learning Systems}}%
  }};

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=1.35in]current page.south west) {{%
    {{\\fontsize{{21.5pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}Vijay Janapa Reddi}}%
  }};

\\end{{tikzpicture}}
\\newpage
\\end{{document}}
"""
    tex_path = f"{scratch_dir}/{c['id']}.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{c['id']}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{c['id']}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{c['id']}.pdf", f"{scratch_dir}/{c['id']}_out"], cwd=scratch_dir, check=True)
    return f"{scratch_dir}/{c['id']}_out-1.png"

rendered_paths = [render_cover(c) for c in nextgen_list]
rendered_imgs = [Image.open(p) for p in rendered_paths]

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 3
rows = 2
grid_w = thumb_w * cols + (cols + 1) * 20
grid_h = thumb_h * rows + (rows + 1) * 50 + 60

master_grid = Image.new('RGB', (grid_w, grid_h), (245, 245, 247))
draw = ImageDraw.Draw(master_grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'PHYSICAL AI: NEXT-GENERATION BESPOKE COVER ARTWORKS', font=font_title, fill=(20, 20, 20))

for idx, (c, img) in enumerate(zip(nextgen_list, rendered_imgs)):
    r = idx // cols
    col_idx = idx % cols
    x = 20 + col_idx * (thumb_w + 20)
    y = 70 + r * (thumb_h + 45)

    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    master_grid.paste(resized, (x, y))

    border_color = (26, 77, 62) if "★" in c["label"] else (90, 90, 90)
    border_width = 3 if "★" in c["label"] else 1
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=border_color, width=border_width)
    draw.text((x + 5, y - 18), c["label"], font=font_label, fill=border_color)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_physical_ai_nextgen_bespoke.png')
master_grid.save(master_contact_path)

print("Next-Gen bespoke covers contact sheet saved successfully!")
