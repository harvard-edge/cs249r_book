import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess, os

brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'
vol_assets = '/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers'

def clean_to_pure_white(in_p, out_p, bg_thresh=235):
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

# 1. Process High-Res Scaling Vol 2
clean_to_pure_white(f"{brain_dir}/cover_vol2_scaling_highres_1788375487703.jpg", f"{brain_dir}/v2_scaling_highres_pw.png", bg_thresh=230)

# 2. Process Physical AI Candidates with Vol 1 Colors
clean_to_pure_white(f"{brain_dir}/cover_phys_vol1_colors_hybrid1_1788407266509.jpg", f"{brain_dir}/phys_hybrid1_pw.png", bg_thresh=230)
clean_to_pure_white(f"{brain_dir}/cover_phys_vol1_colors_gimbal_1788407335342.jpg", f"{brain_dir}/phys_gimbal_vol1col_pw.png", bg_thresh=230)
clean_to_pure_white(f"{brain_dir}/cover_phys_vol1_colors_harmonic_1788407362215.jpg", f"{brain_dir}/phys_harmonic_vol1col_pw.png", bg_thresh=230)

# Physical AI Candidates
phys_candidates = [
    {
        "id": "phys_cand_hybrid1",
        "label": "1. Articulated Robotic Finger Linkages & Gears (Vol 1 Palette) ★★★",
        "art": f"{brain_dir}/phys_hybrid1_pw.png",
        "desc": "Robotic linkages, planetary gears, and cable tendons interwoven with Volume 1's exact crimson/sapphire/cyan palette."
    },
    {
        "id": "phys_cand_harmonic",
        "label": "2. Harmonic Drive Gearboxes & Scissor Mechanisms (Vol 1 Palette) ★★",
        "art": f"{brain_dir}/phys_harmonic_vol1col_pw.png",
        "desc": "Strain-wave harmonic drives and scissor linkages in a continuous closed loop with crimson/sapphire/amber nodes."
    },
    {
        "id": "phys_cand_gimbal_vol1",
        "label": "3. 3-Axis Gyroscopic Gimbal & Bearings (Vol 1 Palette) ★",
        "art": f"{brain_dir}/phys_gimbal_vol1col_pw.png",
        "desc": "Multi-axis rotational gimbal rings with bearings and planetary gear trains in Volume 1's rich color palette."
    }
]

def render_tex(c_id, sub, title, author, art, color):
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
\\definecolor{{accentcolor}}{{HTML}}{{{color}}}

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

  \\node[anchor=center, inner sep=0pt] at ([xshift=0.50in, yshift=1.52in]current page.center) {{%
    \\includegraphics[width=0.96\\paperwidth]{{{art}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {{%
    {{\\fontsize{{23.5pt}}{{27.5pt}}\\rmfamily\\selectfont\\color{{ink}}{sub}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {{%
    {{\\fontsize{{44pt}}{{48pt}}\\rmfamily\\selectfont\\color{{ink}}{title}}}%
  }};

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {{%
    {{\\fontsize{{21.5pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}{author}}}%
  }};

\\end{{tikzpicture}}
\\newpage
\\end{{document}}
"""
    tex_path = f"{scratch_dir}/{c_id}.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{c_id}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{c_id}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{c_id}.pdf", f"{scratch_dir}/{c_id}_out"], cwd=scratch_dir, check=True)
    return f"{scratch_dir}/{c_id}_out-1.png"

# Render Physical AI candidates
phys_rendered = [render_tex(c['id'], "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi", c['art'], "1A4D3E") for c in phys_candidates]
phys_imgs = [Image.open(p) for p in phys_rendered]

# Build Physical AI comparison contact sheet
w, h = phys_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

grid_phys = Image.new('RGB', (thumb_w * 3 + 80, thumb_h + 120), (245, 245, 247))
draw_phys = ImageDraw.Draw(grid_phys)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw_phys.text((25, 18), 'PHYSICAL AI CANDIDATES: CLOSED-LOOP + EMBODIED ROBOTICS (Vol 1 Color Palette)', font=font_title, fill=(20, 20, 20))

for idx, (c, img) in enumerate(zip(phys_candidates, phys_imgs)):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_phys.paste(resized, (x, y))
    bcol = (26, 77, 62)
    draw_phys.rectangle([x, y, x + thumb_w, y + thumb_h], outline=bcol, width=3)
    draw_phys.text((x + 5, y - 18), c["label"], font=font_label, fill=bcol)

out_phys = os.path.join(brain_dir, 'contact_sheet_physical_ai_vol1_colors_candidates.png')
grid_phys.save(out_phys)

# Render New High-Res Scaling Vol 2
p_v2_highres = render_tex("v2_highres", "Scaling", "Machine Learning Systems", "Vijay Janapa Reddi", f"{brain_dir}/v2_scaling_highres_pw.png", "1F407A")
im_v2_highres = Image.open(p_v2_highres)

# Master Tetralogy Sheet with Candidate 1 (Robotic Finger Linkages & Gears)
im_v1 = Image.open(f"{scratch_dir}/sys_v1_out-1.png")
im_agentic = Image.open(f"{scratch_dir}/sys_v3_waypoint_out-1.png")
im_phys_top = phys_imgs[0]

grid_master = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_master = ImageDraw.Draw(grid_master)
draw_master.text((25, 18), 'MASTER 4-VOLUME MACHINE LEARNING SYSTEMS SERIES (Harmonized Color & Physical Architecture)', font=font_title, fill=(20, 20, 20))

tet_panels = [
    (im_v1, 'VOL I: Introduction to (Harvard Crimson #A51C30)', (165, 28, 48)),
    (im_v2_highres, 'VOL II: Scaling [High-Res Render] (ETH Blue #1F407A) ★', (31, 64, 122)),
    (im_agentic, 'AGENTIC: Spawning Trajectory (Royal Amethyst #6B21A8) [LOCKED]', (107, 33, 168)),
    (im_phys_top, 'PHYSICAL AI: Robotic Linkages & Gears (Deep Emerald #1A4D3E) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(tet_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_master.paste(resized, (x, y))
    draw_master.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_master.text((x + 5, y - 18), label, font=font_label, fill=col)

out_master = os.path.join(brain_dir, 'contact_sheet_master_harmonized_tetralogy.png')
grid_master.save(out_master)

print("Physical AI candidates and master tetralogy contact sheet generated successfully!")
