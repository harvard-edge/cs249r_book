import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess, os

brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'
vol_assets = '/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers'

def clean_to_pure_white(in_p, out_p, bg_thresh=235):
    if os.path.exists(out_p):
        return out_p
    img = Image.open(in_p).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (r + g + b) / 3.0
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    mask = (brightness > bg_thresh) & (diff < 15)
    factor = np.clip((brightness - bg_thresh) / (255.0 - bg_thresh), 0.0, 1.0)
    for c in range(3):
        arr[:,:,c] = np.where(mask, arr[:,:,c] * (1 - factor) + 255.0 * factor, arr[:,:,c])
    out = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    out.save(out_p, quality=98)
    return out_p

# Clean additional candidate images
clean_to_pure_white(f"{brain_dir}/cover_agentic_branching_trajectory_1788371185416.jpg", f"{brain_dir}/agentic_branching_pw.png", 230)
clean_to_pure_white(f"{brain_dir}/cover_agentic_armillary_spiral_1788368482646.jpg", f"{brain_dir}/agentic_armillary_pw.png", 230)
clean_to_pure_white(f"{brain_dir}/cover_agentic_tri_swarm_1788368559356.jpg", f"{brain_dir}/agentic_tri_swarm_pw.png", 230)
clean_to_pure_white(f"{assets_dir}/cover-physical-dexterous-hand-art.png", f"{brain_dir}/phys_hand_pw.png", 240)
clean_to_pure_white(f"{assets_dir}/cover-physical-scurve-runnerup-art.png", f"{brain_dir}/phys_scurve_pw.png", 240)

catalog_items = [
    # Volume 1
    {
        "vol": "VOLUME I: FOUNDATION",
        "id": "cat_v1_baseline",
        "rank": "Rank 1 (Locked Baseline)",
        "sub": "Introduction to",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "A51C30",
        "label": "VOL 1 #1: Single Compute Unit (Crimson)",
        "art": f"{vol_assets}/cover-image-white-vol1.png"
    },
    # Volume 2
    {
        "vol": "VOLUME II: SCALING",
        "id": "cat_v2_highres",
        "rank": "Rank 1 (Recommended)",
        "sub": "Scaling",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1F407A",
        "label": "VOL 2 #1: Replicated Fleet Constellation (High-Res) ★",
        "art": f"{brain_dir}/v2_scaling_highres_pw.png"
    },
    {
        "vol": "VOLUME II: SCALING",
        "id": "cat_v2_classic",
        "rank": "Rank 2 (Classic Galaxy)",
        "sub": "Scaling",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1F407A",
        "label": "VOL 2 #2: Replicated Network Galaxy Classic",
        "art": f"{brain_dir}/v2_repeated_network_pw.png"
    },
    {
        "vol": "VOLUME II: SCALING",
        "id": "cat_v2_clean_grid",
        "rank": "Rank 3 (Cluster Grid)",
        "sub": "Scaling",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1F407A",
        "label": "VOL 2 #3: Distributed Cluster Grid",
        "art": f"{brain_dir}/cover_vol2_scaling_pure_white.png"
    },
    # Volume 3: Agentic
    {
        "vol": "VOLUME III: AGENTIC",
        "id": "cat_v3_spawning",
        "rank": "Rank 1 (Locked & Preferred)",
        "sub": "Agentic",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "6B21A8",
        "label": "AGENTIC #1: Spawning Decision Trajectory [LOCKED] ★★★",
        "art": f"{brain_dir}/agentic_waypoint_arc_pw.png"
    },
    {
        "vol": "VOLUME III: AGENTIC",
        "id": "cat_v3_armillary",
        "rank": "Rank 2 (Celestial Navigation)",
        "sub": "Agentic",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "6B21A8",
        "label": "AGENTIC #2: 3D Titanium Armillary Sphere ★★",
        "art": f"{brain_dir}/agentic_armillary_pw.png"
    },
    {
        "vol": "VOLUME III: AGENTIC",
        "id": "cat_v3_infinity",
        "rank": "Rank 3 (Horizontal Loop)",
        "sub": "Agentic",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "6B21A8",
        "label": "AGENTIC #3: Horizontal Decision Waypoints ★",
        "art": f"{brain_dir}/cover_agentic_waypoints_pure_white.png"
    },
    {
        "vol": "VOLUME III: AGENTIC",
        "id": "cat_v3_branching",
        "rank": "Rank 4 (Decision Tree)",
        "sub": "Agentic",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "6B21A8",
        "label": "AGENTIC #4: Branching Decision Tree Trajectory",
        "art": f"{brain_dir}/agentic_branching_pw.png"
    },
    # Volume 4: Physical AI
    {
        "vol": "VOLUME IV: PHYSICAL AI",
        "id": "cat_v4_robot_gears",
        "rank": "Rank 1 (Recommended)",
        "sub": "Physical AI",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1A4D3E",
        "label": "PHYSICAL AI #1: Robotic Linkages, Gears & Vol 1 Colors ★★★",
        "art": f"{brain_dir}/phys_exact_net1_pw.png"
    },
    {
        "vol": "VOLUME IV: PHYSICAL AI",
        "id": "cat_v4_gimbal",
        "rank": "Rank 2 (3-Axis Kinematics)",
        "sub": "Physical AI",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1A4D3E",
        "label": "PHYSICAL AI #2: 3-Axis Gyroscopic Gimbal & Gearbox ★★",
        "art": f"{brain_dir}/cover_physical_gimbal_pure_white.png"
    },
    {
        "vol": "VOLUME IV: PHYSICAL AI",
        "id": "cat_v4_torus_classic",
        "rank": "Rank 3 (Continuous Loop)",
        "sub": "Physical AI",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1A4D3E",
        "label": "PHYSICAL AI #3: Kinetic Feedback Torus Loop (Classic) ★",
        "art": f"{brain_dir}/physical_torus_pw.png"
    },
    {
        "vol": "VOLUME IV: PHYSICAL AI",
        "id": "cat_v4_harmonic",
        "rank": "Rank 4 (Harmonic Drive)",
        "sub": "Physical AI",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1A4D3E",
        "label": "PHYSICAL AI #4: Harmonic Drive Gearboxes & Scissors",
        "art": f"{brain_dir}/phys_harmonic_vol1col_pw.png"
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
\\definecolor{{accentcolor}}{{HTML}}{{{c['color']}}}

\\newlength{{\\titlelen}}
\\newlength{{\\leftm}}
\\newlength{{\\rightm}}

\\begin{{document}}
\\thispagestyle{{empty}}
\\null

\\settowidth{{\\titlelen}}{{\\fontsize{{44pt}}{{44pt}}\\selectfont {c['title']}}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  \\node[anchor=center, inner sep=0pt] at ([xshift=0.50in, yshift=1.52in]current page.center) {{%
    \\includegraphics[width=0.96\\paperwidth]{{{c['art']}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {{%
    {{\\fontsize{{23.5pt}}{{27.5pt}}\\rmfamily\\selectfont\\color{{ink}}{c['sub']}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {{%
    {{\\fontsize{{44pt}}{{48pt}}\\rmfamily\\selectfont\\color{{ink}}{c['title']}}}%
  }};

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {{%
    {{\\fontsize{{21.5pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}{c['author']}}}%
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

rendered_paths = [render_cover(c) for c in catalog_items]
rendered_imgs = [Image.open(p) for p in rendered_paths]

# Build 3 separate Volume Catalog sheets and 1 Master Grand Catalog
# 1. Master Grand Catalog: 4 columns x 3 rows
w, h = rendered_imgs[0].size
thumb_w = 320
thumb_h = int(h * (thumb_w / w))

# Grid: 4 columns (Vol 1, Vol 2, Vol 3, Vol 4)
# Row 0: Top Recommended across all 4 volumes
# Row 1: Second-Ranked options
# Row 2: Third-Ranked options
# Let's map items into a 4x3 matrix:
matrix = [
    # Column 0: Vol 1 (only 1 item, repeat or leave clean)
    [catalog_items[0], None, None],
    # Column 1: Vol 2 (3 items)
    [catalog_items[1], catalog_items[2], catalog_items[3]],
    # Column 2: Vol 3 Agentic (3 items)
    [catalog_items[4], catalog_items[5], catalog_items[6]],
    # Column 3: Vol 4 Physical AI (3 items)
    [catalog_items[8], catalog_items[9], catalog_items[10]],
]

cols = 4
rows = 3
grid_grand = Image.new('RGB', (thumb_w * cols + (cols + 1) * 25, thumb_h * rows + (rows + 1) * 45 + 80), (245, 245, 247))
draw_grand = ImageDraw.Draw(grid_grand)
font_head = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 26)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)
font_vol_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 13)

draw_grand.text((30, 20), 'MACHINE LEARNING SYSTEMS SERIES: MASTER DESIGN CATALOG (VOL I – IV)', font=font_head, fill=(20, 20, 20))

col_headers = [
    ("VOLUME I: SINGLE-NODE", (165, 28, 48)),
    ("VOLUME II: SCALING", (31, 64, 122)),
    ("VOLUME III: AGENTIC", (107, 33, 168)),
    ("VOLUME IV: PHYSICAL AI", (26, 77, 62)),
]

for c_idx in range(cols):
    hx = 25 + c_idx * (thumb_w + 25)
    draw_grand.text((hx + 5, 65), col_headers[c_idx][0], font=font_vol_title, fill=col_headers[c_idx][1])

for r_idx in range(rows):
    for c_idx in range(cols):
        item = matrix[c_idx][r_idx]
        if item is None:
            continue
        # Find index in catalog_items
        idx = catalog_items.index(item)
        img = rendered_imgs[idx]
        
        x = 25 + c_idx * (thumb_w + 25)
        y = 95 + r_idx * (thumb_h + 45)
        
        resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        grid_grand.paste(resized, (x, y))
        
        bcol = tuple(int(item['color'][i:i+2], 16) for i in (0, 2, 4))
        bwidth = 3 if "★" in item['label'] else 2
        draw_grand.rectangle([x, y, x + thumb_w, y + thumb_h], outline=bcol, width=bwidth)
        draw_grand.text((x + 5, y - 16), item['label'], font=font_label, fill=bcol)

out_grand = os.path.join(brain_dir, 'contact_sheet_master_grand_catalog.png')
grid_grand.save(out_grand)

# Also create Row 1 "Top Recommended Series Lineup"
grid_top = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_top = ImageDraw.Draw(grid_top)
draw_top.text((25, 18), 'TOP RECOMMENDED SERIES LINEUP (The Definitive 4-Volume Tetralogy)', font=font_head, fill=(20, 20, 20))

top_four = [
    (catalog_items[0], rendered_imgs[0]),
    (catalog_items[1], rendered_imgs[1]),
    (catalog_items[4], rendered_imgs[4]),
    (catalog_items[8], rendered_imgs[8]),
]

for idx, (item, img) in enumerate(top_four):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_top.paste(resized, (x, y))
    bcol = tuple(int(item['color'][i:i+2], 16) for i in (0, 2, 4))
    draw_top.rectangle([x, y, x + thumb_w, y + thumb_h], outline=bcol, width=3)
    draw_top.text((x + 5, y - 18), item['label'], font=font_label, fill=bcol)

out_top = os.path.join(brain_dir, 'contact_sheet_top_recommended_tetralogy.png')
grid_top.save(out_top)

print("Master Grand Catalog and Top Recommended Series Lineup saved successfully!")
