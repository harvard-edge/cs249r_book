import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

def render_cover_tex(name, subtitle, title, author, art_path, art_width, art_x, art_y, accent_hex='1F407A'):
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

\\settowidth{{\\titlelen}}{{\\fontsize{{43.5pt}}{{43.5pt}}\\selectfont {title}}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  \\node[anchor=center, inner sep=0pt] at ([xshift={art_x}, yshift={art_y}]current page.center) {{%
    \\includegraphics[width={art_width}]{{{art_path}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.95in]current page.south west) {{%
    {{\\fontsize{{23pt}}{{27pt}}\\rmfamily\\selectfont\\color{{ink}}{subtitle}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.30in]current page.south west) {{%
    {{\\fontsize{{43.5pt}}{{47pt}}\\rmfamily\\selectfont\\color{{ink}}{title}}}%
  }};

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=1.35in]current page.south west) {{%
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

# Re-render Vol 2 with 'Scaling'
cov_v2 = render_cover_tex(
    "quad_v2_scaling", "Scaling", "Machine Learning Systems", "Vijay Janapa Reddi",
    "/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol2.png",
    "0.98\\paperwidth", "1.10in", "1.92in", "1F407A"
)

im_v1 = Image.open(f"{scratch_dir}/quad_v1_out-1.png")
im_v2 = Image.open(cov_v2)
im_c1 = Image.open(f"{scratch_dir}/phys_cand1_torus_out-1.png")
im_c4 = Image.open(f"{scratch_dir}/phys_cand4_gimbal_out-1.png")
im_c3 = Image.open(f"{scratch_dir}/phys_cand3_scurve_out-1.png")

w, h = im_v1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

grid = Image.new('RGB', (thumb_w * 5 + 120, thumb_h + 130), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'TOP CONTENDER RANKING: Physical AI vs Volume 1 & Volume 2 Trilogic Context', font=font_title, fill=(20, 20, 20))

panels = [
    (im_v1, 'VOL I: Introduction to (Harvard Crimson)', (165, 28, 48), 2),
    (im_v2, 'VOL II: Scaling (ETH Zurich Blue)', (31, 64, 122), 2),
    (im_c1, 'RANK 1: Candidate 1 (Torus Feedback Loop) ★★★', (26, 77, 62), 4),
    (im_c4, 'RANK 2: Candidate 4 (Gyroscopic Gimbal) ★★', (40, 90, 80), 3),
    (im_c3, 'RANK 3: Candidate 3 (Kinodynamic S-Curve) ★', (70, 70, 70), 2),
]

for idx, (img, label, col, bwidth) in enumerate(panels):
    x = 20 + idx * (thumb_w + 20)
    y = 65
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=bwidth)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

out_file = os.path.join(brain_dir, 'contact_sheet_top_contender_ranking.png')
grid.save(out_file)
print("Updated top contender ranking contact sheet saved!")
