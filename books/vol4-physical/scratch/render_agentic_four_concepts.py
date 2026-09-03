import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

concepts = [
    {
        "id": "ag_c1_tree",
        "label": "Concept 1: Recursive Self-Reflection & Tree of Thoughts ★★★",
        "art": f"{brain_dir}/cover_agentic_tree_of_thoughts_1788364767098.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "ag_c2_tools",
        "label": "Concept 2: Modular Tool-Use & Crystalline Prisms ★★",
        "art": f"{brain_dir}/cover_agentic_tool_prisms_1788364416892.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "ag_c3_swarm",
        "label": "Concept 3: Multi-Agent Orchestration Lattice ★",
        "art": f"{brain_dir}/cover_agentic_swarm_mesh_1788364815515.jpg",
        "width": "0.96\\paperwidth", "x": "0.50in", "y": "1.52in"
    },
    {
        "id": "ag_c4_nuclei",
        "label": "Concept 4: Multi-Nuclei Autonomous Agent Council",
        "art": f"{brain_dir}/cover_agentic_multi_nuclei_1788364339776.jpg",
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
\\definecolor{{accentcolor}}{{HTML}}{{6B21A8}}

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

rendered_paths = [render_cover(c) for c in concepts]
rendered_imgs = [Image.open(p) for p in rendered_paths]

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 4
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'AGENTIC ML SYSTEMS: HOW WE CAPTURE AGENTS VISUALLY (Vol 1 Evolution)', font=font_title, fill=(20, 20, 20))

for idx, (c, img) in enumerate(zip(concepts, rendered_imgs)):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))

    border_color = (107, 33, 168) if "★" in c["label"] else (90, 90, 90)
    border_width = 3 if "★★★" in c["label"] else 2
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=border_color, width=border_width)
    draw.text((x + 5, y - 18), c["label"], font=font_label, fill=border_color)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_agentic_four_distinct_concepts.png')
grid.save(master_contact_path)

print("Agentic four distinct concepts contact sheet saved successfully!")
