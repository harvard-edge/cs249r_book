import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
ref_img = Image.open('/var/folders/92/bhy05ch15kn2_msznf84g_sc0000gn/T/images/Ghostty 2026-09-02 19.22.02.png')

variations = [
    {
        "id": "v1_base1",
        "title_size": "43.5pt", "title_lead": "47pt",
        "sub_size": "23pt", "sub_lead": "27pt",
        "author_size": "21.5pt", "author_lead": "25pt",
        "art_width": "1.00\\paperwidth", "art_x": "1.05in", "art_y": "1.90in",
        "sub_y": "2.95in", "title_y": "2.30in", "author_y": "1.35in",
        "desc": "Var 1: Art 1.00w, Sub 2.95in, Title 2.30in, Author 1.35in"
    },
    {
        "id": "v1_base2",
        "title_size": "44pt", "title_lead": "48pt",
        "sub_size": "23.5pt", "sub_lead": "27.5pt",
        "author_size": "22pt", "author_lead": "26pt",
        "art_width": "1.04\\paperwidth", "art_x": "1.15in", "art_y": "1.95in",
        "sub_y": "3.05in", "title_y": "2.38in", "author_y": "1.40in",
        "desc": "Var 2: Art 1.04w, Sub 3.05in, Title 2.38in, Author 1.40in ★"
    },
    {
        "id": "v1_base3",
        "title_size": "44.5pt", "title_lead": "48.5pt",
        "sub_size": "24pt", "sub_lead": "28pt",
        "author_size": "22pt", "author_lead": "26pt",
        "art_width": "1.08\\paperwidth", "art_x": "1.25in", "art_y": "2.00in",
        "sub_y": "3.15in", "title_y": "2.45in", "author_y": "1.45in",
        "desc": "Var 3: Art 1.08w, Sub 3.15in, Title 2.45in, Author 1.45in"
    },
    {
        "id": "v1_base4",
        "title_size": "43pt", "title_lead": "47pt",
        "sub_size": "23pt", "sub_lead": "27pt",
        "author_size": "21pt", "author_lead": "25pt",
        "art_width": "0.96\\paperwidth", "art_x": "0.95in", "art_y": "1.85in",
        "sub_y": "2.85in", "title_y": "2.22in", "author_y": "1.30in",
        "desc": "Var 4: Art 0.96w, Sub 2.85in, Title 2.22in, Author 1.30in"
    }
]

for v in variations:
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

\\newlength{{\\titlelen}}
\\newlength{{\\leftm}}
\\newlength{{\\rightm}}

\\begin{{document}}
\\thispagestyle{{empty}}
\\null

\\settowidth{{\\titlelen}}{{\\fontsize{{{v['title_size']}}}{{{v['title_size']}}}\\selectfont Machine Learning Systems}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  % Massive artwork matching gold standard reference
  \\node[anchor=center, inner sep=0pt] at ([xshift={v['art_x']}, yshift={v['art_y']}]current page.center) {{%
    \\includegraphics[width={v['art_width']}]{{/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol1.png}}%
  }};

  % Subtitle at exact left margin of Title
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift={v['sub_y']}]current page.south west) {{%
    {{\\fontsize{{{v['sub_size']}}}{{{v['sub_lead']}}}\\rmfamily\\selectfont\\color{{ink}}Introduction to}}%
  }};

  % Single-Line Main Title
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift={v['title_y']}]current page.south west) {{%
    {{\\fontsize{{{v['title_size']}}}{{{v['title_lead']}}}\\rmfamily\\selectfont\\color{{ink}}Machine Learning Systems}}%
  }};

  % Single-Line Right-aligned Author docked under right edge of Title
  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift={v['author_y']}]current page.south west) {{%
    {{\\fontsize{{{v['author_size']}}}{{{v['author_lead']}}}\\rmfamily\\selectfont\\color{{ink}}Vijay Janapa Reddi}}%
  }};

\\end{{tikzpicture}}
\\newpage
\\end{{document}}
"""
    tex_path = f"{scratch_dir}/{v['id']}.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    # Run twice for TikZ coordinates
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{v['id']}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{v['id']}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{v['id']}.pdf", f"{scratch_dir}/{v['id']}_out"], cwd=scratch_dir, check=True)

# Build 5-column calibration contact sheet
rendered = [
    (Image.open(f"{scratch_dir}/{v['id']}_out-1.png"), v["desc"])
    for v in variations
]

w, h = rendered[0][0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 5
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 15), 'VOL 1 MIT PRESS GOLD STANDARD CALIBRATION: Baseline & Artwork Geometry', font=font_title, fill=(20, 20, 20))

ref_resized = ref_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
grid.paste(ref_resized, (20, 60))
draw.rectangle([20, 60, 20 + thumb_w, 60 + thumb_h], outline=(165, 28, 48), width=3)
draw.text((25, 42), "MIT PRESS GOLD STANDARD (Ref)", font=font_label, fill=(165, 28, 48))

for idx, (img, label) in enumerate(rendered):
    x = 20 + (idx + 1) * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    color = (26, 77, 62) if "★" in label else (70, 70, 70)
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=color, width=3 if "★" in label else 2)
    draw.text((x + 5, y - 18), label, font=font_label, fill=color)

out_contact = os.path.join(brain_dir, "contact_sheet_vol1_gold_standard_calibration.png")
grid.save(out_contact)
print("Gold standard calibration contact sheet saved successfully!")
