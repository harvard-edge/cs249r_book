import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

hybrids = [
    {
        "id": "gyro_hyb1",
        "label": "Hybrid 1: Concentric Bearing Gimbal & Linkages ★",
        "art": f"{brain_dir}/cover_phys_gyro_robot_hybrid1_1788361791629.jpg",
        "width": "0.96\\paperwidth", "x": "0.55in", "y": "1.52in"
    },
    {
        "id": "gyro_hyb2",
        "label": "Hybrid 2: Multi-Axis Gimbal & Robotic Hands",
        "art": f"{brain_dir}/cover_phys_gyro_robot_hybrid2_1788361812306.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "gyro_hyb3",
        "label": "Hybrid 3: Interlocking Gyro-Ring & Sensor Lattice ★★★",
        "art": f"{brain_dir}/cover_phys_gyro_robot_hybrid3_1788361836431.jpg",
        "width": "0.96\\paperwidth", "x": "0.55in", "y": "1.52in"
    },
    {
        "id": "gyro_hyb4",
        "label": "Hybrid 4: Planetary Gear Gyro-Blossom ★★",
        "art": f"{brain_dir}/cover_phys_gyro_robot_hybrid4_1788361857917.jpg",
        "width": "0.94\\paperwidth", "x": "0.50in", "y": "1.52in"
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

\\settowidth{{\\titlelen}}{{\\fontsize{{44pt}}{{44pt}}\\selectfont Machine Learning Systems}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  \\node[anchor=center, inner sep=0pt] at ([xshift={c['x']}, yshift={c['y']}]current page.center) {{%
    \\includegraphics[width={c['width']}]{{{c['art']}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {{%
    {{\\fontsize{{23.5pt}}{{27.5pt}}\\rmfamily\\selectfont\\color{{ink}}Physical AI}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {{%
    {{\\fontsize{{44pt}}{{48pt}}\\rmfamily\\selectfont\\color{{ink}}Machine Learning Systems}}%
  }};

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {{%
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

rendered_paths = [render_cover(c) for c in hybrids]
rendered_imgs = [Image.open(p) for p in rendered_paths]

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 4
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'PHYSICAL AI: GYROSCOPIC GIMBAL + ROBOTIC LINKAGE HYBRIDS (Lowered Baseline)', font=font_title, fill=(20, 20, 20))

for idx, (c, img) in enumerate(zip(hybrids, rendered_imgs)):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))

    border_color = (26, 77, 62) if "★" in c["label"] else (90, 90, 90)
    border_width = 3 if "★★★" in c["label"] else 2
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=border_color, width=border_width)
    draw.text((x + 5, y - 18), c["label"], font=font_label, fill=border_color)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_gyro_robot_hybrids.png')
grid.save(master_contact_path)

print("Gyro-Robot hybrids contact sheet saved successfully!")
