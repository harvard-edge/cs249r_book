import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
ref_img = Image.open('/var/folders/92/bhy05ch15kn2_msznf84g_sc0000gn/T/images/Ghostty 2026-09-02 20.32.47.png')

variations = [
    {
        "id": "v1_low1",
        "title_size": "44pt", "title_lead": "48pt",
        "sub_size": "23.5pt", "sub_lead": "27.5pt",
        "author_size": "21.5pt", "author_lead": "25pt",
        "art_width": "1.06\\paperwidth", "art_x": "1.15in", "art_y": "1.50in",
        "sub_y": "2.40in", "title_y": "1.62in", "author_y": "0.78in",
        "desc": "Var 1: Sub 2.40in, Title 1.62in, Author 0.78in (Target Match) ★"
    },
    {
        "id": "v1_low2",
        "title_size": "44pt", "title_lead": "48pt",
        "sub_size": "23.5pt", "sub_lead": "27.5pt",
        "author_size": "21.5pt", "author_lead": "25pt",
        "art_width": "1.04\\paperwidth", "art_x": "1.10in", "art_y": "1.55in",
        "sub_y": "2.48in", "title_y": "1.70in", "author_y": "0.85in",
        "desc": "Var 2: Sub 2.48in, Title 1.70in, Author 0.85in"
    },
    {
        "id": "v1_low3",
        "title_size": "44.5pt", "title_lead": "48.5pt",
        "sub_size": "24pt", "sub_lead": "28pt",
        "author_size": "22pt", "author_lead": "26pt",
        "art_width": "1.08\\paperwidth", "art_x": "1.20in", "art_y": "1.45in",
        "sub_y": "2.32in", "title_y": "1.55in", "author_y": "0.72in",
        "desc": "Var 3: Sub 2.32in, Title 1.55in, Author 0.72in"
    },
    {
        "id": "v1_low4",
        "title_size": "43.5pt", "title_lead": "47.5pt",
        "sub_size": "23pt", "sub_lead": "27pt",
        "author_size": "21pt", "author_lead": "25pt",
        "art_width": "1.02\\paperwidth", "art_x": "1.05in", "art_y": "1.60in",
        "sub_y": "2.55in", "title_y": "1.78in", "author_y": "0.92in",
        "desc": "Var 4: Sub 2.55in, Title 1.78in, Author 0.92in"
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

  % Artwork node lowered to give huge presence and soft contact shadow
  \\node[anchor=center, inner sep=0pt] at ([xshift={v['art_x']}, yshift={v['art_y']}]current page.center) {{%
    \\includegraphics[width={v['art_width']}]{{/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers/cover-image-transparent-vol1.png}}%
  }};

  % Subtitle lowered
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift={v['sub_y']}]current page.south west) {{%
    {{\\fontsize{{{v['sub_size']}}}{{{v['sub_lead']}}}\\rmfamily\\selectfont\\color{{ink}}Introduction to}}%
  }};

  % Single-Line Main Title lowered
  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift={v['title_y']}]current page.south west) {{%
    {{\\fontsize{{{v['title_size']}}}{{{v['title_lead']}}}\\rmfamily\\selectfont\\color{{ink}}Machine Learning Systems}}%
  }};

  % Single-Line Right-aligned Author lowered
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

draw.text((25, 15), 'LOWERED TEXT BASELINE CALIBRATION (Matching MIT Press Ref Exactly)', font=font_title, fill=(20, 20, 20))

ref_resized = ref_img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
grid.paste(ref_resized, (20, 60))
draw.rectangle([20, 60, 20 + thumb_w, 60 + thumb_h], outline=(165, 28, 48), width=3)
draw.text((25, 42), "MIT PRESS REFERENCE (Ghostty)", font=font_label, fill=(165, 28, 48))

for idx, (img, label) in enumerate(rendered):
    x = 20 + (idx + 1) * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    color = (26, 77, 62) if "★" in label else (70, 70, 70)
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=color, width=3 if "★" in label else 2)
    draw.text((x + 5, y - 18), label, font=font_label, fill=color)

out_contact = os.path.join(brain_dir, "contact_sheet_lowered_baseline_calibration.png")
grid.save(out_contact)
print("Lowered baseline calibration contact sheet saved!")
