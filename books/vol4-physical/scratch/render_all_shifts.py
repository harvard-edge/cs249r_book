import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

shifts = [
    ("shift0", "0.00in", "Shift 0: xshift=0.0in (Geometric Center)"),
    ("shift1", "0.15in", "Shift 1: xshift=+0.15in (Subtle Right)"),
    ("shift2", "0.22in", "Shift 2: xshift=+0.22in (Optimal Balance) ★"),
    ("shift3", "0.32in", "Shift 3: xshift=+0.32in (Pronounced Right)"),
]

for name, xshift, label in shifts:
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
\\thispagestyle{{empty}}%
\\null
\\settowidth{{\\titlelen}}{{\\fontsize{{43.5pt}}{{43.5pt}}\\selectfont Machine Learning Systems}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  % Floating 3D sculpture artwork in upper canvas with optical xshift
  \\node[anchor=center, inner sep=0pt] at ([xshift={xshift}, yshift=1.95in]current page.center) {{%
    \\includegraphics[width=0.83\\paperwidth]{{/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers/cover-physical-loop-recommended-art.png}}%
  }};

  % Left-aligned Subtitle at leftm
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=3.40in]current page.south west) {{%
    {{\\fontsize{{22pt}}{{26pt}}\\rmfamily\\selectfont\\color{{ink}}Physical AI}}%
  }};

  % Single-Line Main Title at leftm (Symmetric!)
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.65in]current page.south west) {{%
    {{\\fontsize{{43.5pt}}{{47pt}}\\rmfamily\\selectfont\\color{{ink}}Machine Learning Systems}}%
  }};

  % Single-Line Right-aligned Author at rightm (Aligned under right edge of "Systems")
  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=1.40in]current page.south west) {{%
    {{\\fontsize{{21pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}Vijay Janapa Reddi}}%
  }};

\\end{{tikzpicture}}%
\\newpage
\\end{{document}}
"""
    with open(f"{scratch_dir}/{name}.tex", "w") as f:
        f.write(tex)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{name}.pdf", f"{scratch_dir}/{name}_out"], cwd=scratch_dir, check=True)

# Build 5-column contact sheet: Reference + 4 shifts
ref_img = Image.open("/var/folders/92/bhy05ch15kn2_msznf84g_sc0000gn/T/images/Ghostty 2026-09-02 16.03.08.png")

rendered = [
    (Image.open(f"{scratch_dir}/shift0_out-1.png"), "Shift 0: xshift=0.0in (Geometric)"),
    (Image.open(f"{scratch_dir}/shift1_out-1.png"), "Shift 1: xshift=+0.15in (Subtle Right)"),
    (Image.open(f"{scratch_dir}/shift2_out-1.png"), "Shift 2: xshift=+0.22in (Optimal Balance) ★"),
    (Image.open(f"{scratch_dir}/shift3_out-1.png"), "Shift 3: xshift=+0.32in (Pronounced Right)"),
]

w, h = rendered[0][0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 5
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 15), 'OPTICAL CENTERING COMPARISON: Calibrating Sculpture Visual Weight', font=font_title, fill=(20, 20, 20))

ref_resized = ref_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
grid.paste(ref_resized, (20, 60))
draw.rectangle([20, 60, 20 + thumb_w, 60 + thumb_h], outline=(165, 28, 48), width=3)
draw.text((25, 42), "TARGET REFERENCE (Vol 1)", font=font_label, fill=(165, 28, 48))

for idx, (img, label) in enumerate(rendered):
    x = 20 + (idx + 1) * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    color = (26, 77, 62) if "★" in label else (70, 70, 70)
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=color, width=3 if "★" in label else 2)
    draw.text((x + 5, y - 18), label, font=font_label, fill=color)

out_contact = os.path.join(brain_dir, "contact_sheet_cover_optical_centering_final.png")
grid.save(out_contact)
print("Optical centering final contact sheet saved!")
