import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'

candidates = [
    {
        "id": "phys_cand1_torus",
        "title_label": "Candidate 1: Kinetic Feedback Torus Loop ★",
        "art_path": f"{assets_dir}/cover-physical-loop-recommended-art.png",
        "art_width": "0.96\\paperwidth", "art_x": "1.15in", "art_y": "1.95in",
        "desc": "Closed kinetic origami torus with emerald cybernetic sensor network"
    },
    {
        "id": "phys_cand2_linkage",
        "title_label": "Candidate 2: Articulated Mechanical Origami Linkage",
        "art_path": f"{brain_dir}/cover_physical_linkage_1788359003600.jpg",
        "art_width": "0.92\\paperwidth", "art_x": "0.50in", "art_y": "1.90in",
        "desc": "Articulated robotic linkages with pleated fans and feedback wiring"
    },
    {
        "id": "phys_cand3_scurve",
        "title_label": "Candidate 3: Kinodynamic S-Curve Actuator Ribbon",
        "art_path": f"{assets_dir}/cover-physical-scurve-runnerup-art.png",
        "art_width": "0.94\\paperwidth", "art_x": "1.10in", "art_y": "1.90in",
        "desc": "Explosive dynamic kinematic ribbon with motor linkages and node field"
    },
    {
        "id": "phys_cand4_gimbal",
        "title_label": "Candidate 4: Gyroscopic Multi-Axis Gimbal",
        "art_path": f"{assets_dir}/cover-physical-gyroscopic-gimbal-art.png",
        "art_width": "0.90\\paperwidth", "art_x": "1.05in", "art_y": "1.90in",
        "desc": "Precision dual-ring gimbal mechanism with hemisphere sensor graph"
    },
    {
        "id": "phys_cand5_hand",
        "title_label": "Candidate 5: Dexterous Robotic Hand & Tactile Array",
        "art_path": f"{assets_dir}/cover-physical-dexterous-hand-art.png",
        "art_width": "0.92\\paperwidth", "art_x": "1.00in", "art_y": "1.90in",
        "desc": "Dexterous compliant hand with tactile sensor lattice"
    },
    {
        "id": "phys_cand6_causal",
        "title_label": "Candidate 6: Causal Loop Dynamic Sculpture",
        "art_path": f"{assets_dir}/cover_concept_3_causal_loop.png",
        "art_width": "0.90\\paperwidth", "art_x": "0.85in", "art_y": "1.90in",
        "desc": "Symmetric dual-arm causal feedback loop"
    },
    {
        "id": "phys_cand7_phase",
        "title_label": "Candidate 7: Phase Space Dynamic Manifold",
        "art_path": f"{assets_dir}/cover_concept_1_phase_space.png",
        "art_width": "0.92\\paperwidth", "art_x": "0.90in", "art_y": "1.90in",
        "desc": "Multi-rate phase space manifold with geometric origami core"
    },
    {
        "id": "phys_cand8_dualspeed",
        "title_label": "Candidate 8: Dual-Speed Brain & Reflex Mechanism",
        "art_path": f"{assets_dir}/cover_concept_2_dual_speed.png",
        "art_width": "0.92\\paperwidth", "art_x": "0.90in", "art_y": "1.90in",
        "desc": "Dual-speed proposal-permission architecture in kinetic form"
    }
]

for c in candidates:
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

  % Artwork node
  \\node[anchor=center, inner sep=0pt] at ([xshift={c['art_x']}, yshift={c['art_y']}]current page.center) {{%
    \\includegraphics[width={c['art_width']}]{{{c['art_path']}}}%
  }};

  % Subtitle at exact left margin of Title (MIT Press Calibrated y=2.95in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.95in]current page.south west) {{%
    {{\\fontsize{{23pt}}{{27pt}}\\rmfamily\\selectfont\\color{{ink}}Physical AI}}%
  }};

  % Single-Line Main Title (MIT Press Calibrated y=2.30in)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.30in]current page.south west) {{%
    {{\\fontsize{{43.5pt}}{{47pt}}\\rmfamily\\selectfont\\color{{ink}}Machine Learning Systems}}%
  }};

  % Single-Line Right-aligned Author docked under right edge of Title (MIT Press Calibrated y=1.35in)
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

# Build 2-Row x 4-Column Master Gallery Contact Sheet
rendered_imgs = [Image.open(f"{scratch_dir}/{c['id']}_out-1.png") for c in candidates]

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 4
rows = 2
grid_w = thumb_w * cols + (cols + 1) * 20
grid_h = thumb_h * rows + (rows + 1) * 50 + 60

master_grid = Image.new('RGB', (grid_w, grid_h), (245, 245, 247))
draw = ImageDraw.Draw(master_grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'PHYSICAL AI: MASTER COVER CANDIDATE GALLERY (MIT Press Locked Geometry)', font=font_title, fill=(20, 20, 20))

for idx, (c, img) in enumerate(zip(candidates, rendered_imgs)):
    r = idx // cols
    col_idx = idx % cols
    x = 20 + col_idx * (thumb_w + 20)
    y = 70 + r * (thumb_h + 45)

    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    master_grid.paste(resized, (x, y))

    border_color = (26, 77, 62) if "★" in c["title_label"] else (90, 90, 90)
    border_width = 3 if "★" in c["title_label"] else 1
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=border_color, width=border_width)
    draw.text((x + 5, y - 18), c["title_label"], font=font_label, fill=border_color)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_physical_ai_all_candidates_master.png')
master_grid.save(master_contact_path)

print("Master Physical AI candidates contact sheet saved successfully!")
