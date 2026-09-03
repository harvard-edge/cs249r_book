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

# Clean new images to pure white
clean_to_pure_white(f"{brain_dir}/cover_phys_vol1_exact_net_colors1_1788407657069.jpg", f"{brain_dir}/phys_exact_net1_pw.png", bg_thresh=230)
clean_to_pure_white(f"{brain_dir}/cover_phys_vol1_exact_net_colors2_1788407683764.jpg", f"{brain_dir}/phys_exact_net2_pw.png", bg_thresh=230)

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

p_phys1 = render_tex("phys_exact1", "Physical AI", "Machine Learning Systems", "Vijay Janapa Reddi", f"{brain_dir}/phys_exact_net1_pw.png", "1A4D3E")
im_phys1 = Image.open(p_phys1)

# Master Tetralogy with the exact neural network colors
im_v1 = Image.open(f"{scratch_dir}/sys_v1_out-1.png")
im_v2 = Image.open(f"{scratch_dir}/v2_highres_out-1.png")
im_agentic = Image.open(f"{scratch_dir}/sys_v3_waypoint_out-1.png")

w, h = im_v1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

grid_master = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_master = ImageDraw.Draw(grid_master)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw_master.text((25, 18), 'OFFICIAL MACHINE LEARNING SYSTEMS TETRALOGY (Exact Vol 1 Neural Network Colors)', font=font_title, fill=(20, 20, 20))

tet_panels = [
    (im_v1, 'VOL I: Introduction to (Harvard Crimson #A51C30)', (165, 28, 48)),
    (im_v2, 'VOL II: Scaling [Replicated Fleet] (ETH Blue #1F407A)', (31, 64, 122)),
    (im_agentic, 'AGENTIC: Spawning Trajectory (Royal Amethyst #6B21A8) [LOCKED]', (107, 33, 168)),
    (im_phys1, 'PHYSICAL AI: Exact Vol 1 Neural Filigree & Robotics (Deep Emerald #1A4D3E) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(tet_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_master.paste(resized, (x, y))
    draw_master.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_master.text((x + 5, y - 18), label, font=font_label, fill=col)

out_master = os.path.join(brain_dir, 'contact_sheet_official_tetralogy_exact_net_colors.png')
grid_master.save(out_master)

# Also create side-by-side comparison of Vol 1 and Physical AI to inspect network color match
grid_pair = Image.new('RGB', (thumb_w * 2 + 60, thumb_h + 120), (245, 245, 247))
draw_pair = ImageDraw.Draw(grid_pair)
draw_pair.text((25, 18), 'SIDE-BY-SIDE INSPECTION: VOLUME 1 vs PHYSICAL AI (Exact Neural Network Color Match)', font=font_title, fill=(20, 20, 20))

pair_panels = [
    (im_v1, 'VOL I: Original Neural Network (Crimson/Indigo/Cyan Gradient)', (165, 28, 48)),
    (im_phys1, 'PHYSICAL AI: Exact Vol 1 Network Colors + Robotic Finger Linkages & Gears', (26, 77, 62))
]

for idx, (img, label, col) in enumerate(pair_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_pair.paste(resized, (x, y))
    draw_pair.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_pair.text((x + 5, y - 18), label, font=font_label, fill=col)

out_pair = os.path.join(brain_dir, 'contact_sheet_v1_vs_physical_exact_color_match.png')
grid_pair.save(out_pair)

print("Exact neural network color match contact sheets generated successfully!")
