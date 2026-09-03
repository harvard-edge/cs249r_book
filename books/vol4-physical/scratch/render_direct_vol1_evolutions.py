import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

evolutions = [
    {
        "id": "v1_evo1",
        "label": "Evolution 1: Tendon & Robotic Articulation",
        "art": f"{brain_dir}/vol1_direct_evolution_1_1788361996173.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "v1_evo2",
        "label": "Evolution 2: Planetary Bearing Gear Core & Linkages ★",
        "art": f"{brain_dir}/vol1_direct_evolution_2_1788362026684.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "v1_evo3",
        "label": "Evolution 3: 3D Gyro-Gimbal & Micro-Actuators ★★",
        "art": f"{brain_dir}/vol1_direct_evolution_3_1788362081356.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "v1_evo4",
        "label": "Evolution 4: Gimbal Core + Robot Fingers + Emerald Network ★★★",
        "art": f"{brain_dir}/vol1_direct_evolution_4_1788362108455.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
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

rendered_paths = [render_cover(c) for c in evolutions]
rendered_imgs = [Image.open(p) for p in rendered_paths]

# Add Volume 1 for direct benchmark
im_v1 = Image.open(f"{scratch_dir}/final_v1_out-1.png")

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 5
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'PHYSICAL AI: DIRECT EVOLUTIONS FROM VOLUME 1 ORIGINAL ARTWORK', font=font_title, fill=(20, 20, 20))

all_panels = [(im_v1, "ORIGINAL VOL 1: Introduction to (Benchmark)", (165, 28, 48))] + [
    (img, c["label"], (26, 77, 62) if "★" in c["label"] else (70, 70, 70))
    for c, img in zip(evolutions, rendered_imgs)
]

for idx, (img, label, col) in enumerate(all_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))

    border_width = 3 if ("★★★" in label or "ORIGINAL" in label) else 2
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=border_width)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_vol1_direct_evolutions.png')
grid.save(master_contact_path)

print("Vol 1 direct evolutions contact sheet saved successfully!")
