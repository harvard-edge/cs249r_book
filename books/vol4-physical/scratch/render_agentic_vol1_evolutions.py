import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

agentic_evolutions = [
    {
        "id": "agentic_evo1",
        "label": "Agentic 1: Autonomous Multi-Agent Mesh ★★★",
        "art": f"{brain_dir}/vol1_agentic_evolution_1_1788363106392.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "agentic_evo2",
        "label": "Agentic 2: Gold Crystalline Facet Lattice ★★",
        "art": f"{brain_dir}/vol1_agentic_evolution_2_1788363138143.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "agentic_evo3",
        "label": "Agentic 3: Gold Thread Deliberation Core & Amber Nodes ★",
        "art": f"{brain_dir}/vol1_agentic_evolution_3_1788363172399.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "agentic_evo4",
        "label": "Agentic 4: Swarm Polyhedral Agent Constellation",
        "art": f"{brain_dir}/vol1_agentic_evolution_4_1788363228540.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    }
]

def render_agentic_cover(c):
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
\\definecolor{{accentcolor}}{{HTML}}{{C05621}}

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
    {{\\fontsize{{23.5pt}}{{27.5pt}}\\rmfamily\\selectfont\\color{{ink}}Agentic}}%
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

rendered_paths = [render_agentic_cover(c) for c in agentic_evolutions]
rendered_imgs = [Image.open(p) for p in rendered_paths]

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 4
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'AGENTIC ML SYSTEMS: DIRECT EVOLUTIONS FROM VOLUME 1 ARTWORK', font=font_title, fill=(20, 20, 20))

for idx, (c, img) in enumerate(zip(agentic_evolutions, rendered_imgs)):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))

    border_color = (192, 86, 33) if "★" in c["label"] else (90, 90, 90)
    border_width = 3 if "★★★" in c["label"] else 2
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=border_color, width=border_width)
    draw.text((x + 5, y - 18), c["label"], font=font_label, fill=border_color)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_agentic_vol1_evolutions.png')
grid.save(master_contact_path)

# Build Master 4-Volume Tetralogy (Vol 1, Vol 2, Agentic 1, Physical AI Cand 4)
im_v1 = Image.open(f"{scratch_dir}/final_v1_out-1.png")
im_v2 = Image.open(f"{scratch_dir}/final_v2_out-1.png")
im_ag1 = rendered_imgs[0]
im_phys_cand4 = Image.open(f"{brain_dir}/verified_physical_ai_candidate4_cover.png")

grid_tetralogy = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_tet = ImageDraw.Draw(grid_tetralogy)
draw_tet.text((25, 18), 'THE COMPLETE 4-BOOK MACHINE LEARNING SYSTEMS SERIES (MIT Press Master Baseline)', font=font_title, fill=(20, 20, 20))

tet_panels = [
    (im_v1, 'VOL I: Introduction to (Harvard Crimson #A51C30)', (165, 28, 48)),
    (im_v2, 'VOL II: Scaling (ETH Zurich Blue #1F407A)', (31, 64, 122)),
    (im_ag1, 'AGENTIC: Autonomous Agents (Warm Amber #C05621) ★', (192, 86, 33)),
    (im_phys_cand4, 'PHYSICAL AI: Systems Architecture (Deep Emerald #1A4D3E) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(tet_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_tetralogy.paste(resized, (x, y))
    draw_tet.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_tet.text((x + 5, y - 18), label, font=font_label, fill=col)

out_tet = os.path.join(brain_dir, 'contact_sheet_final_mlsys_tetralogy_series.png')
grid_tetralogy.save(out_tet)

print("Agentic evolutions and master tetralogy contact sheets saved successfully!")
