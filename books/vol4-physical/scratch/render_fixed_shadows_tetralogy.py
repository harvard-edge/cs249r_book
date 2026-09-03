import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
vol_assets = '/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers'

books = [
    {
        "id": "fixed_v1",
        "sub": "Introduction to",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "A51C30",
        "label": "VOL I: Single Model (Harvard Crimson #A51C30)",
        "art": f"{vol_assets}/cover-image-white-vol1.png"
    },
    {
        "id": "fixed_v2",
        "sub": "Scaling",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1F407A",
        "label": "VOL II: Replicated Fleet (ETH Blue #1F407A)",
        "art": f"{brain_dir}/v2_scaling_highres_fixed.png"
    },
    {
        "id": "fixed_v3",
        "sub": "Agentic",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "6B21A8",
        "label": "AGENTIC: Spawning Trajectory (Royal Amethyst #6B21A8) [LOCKED]",
        "art": f"{brain_dir}/agentic_waypoint_arc_fixed.png"
    },
    {
        "id": "fixed_v4",
        "sub": "Physical AI",
        "title": "Machine Learning Systems",
        "author": "Vijay Janapa Reddi",
        "color": "1A4D3E",
        "label": "PHYSICAL AI: Breathy Robotic Finger Loop (Deep Emerald #1A4D3E) ★",
        "art": f"{brain_dir}/phys_breathy1_fixed.png"
    }
]

def render_book(b):
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
\\definecolor{{accentcolor}}{{HTML}}{{{b['color']}}}

\\newlength{{\\titlelen}}
\\newlength{{\\leftm}}
\\newlength{{\\rightm}}

\\begin{{document}}
\\thispagestyle{{empty}}
\\null

\\settowidth{{\\titlelen}}{{\\fontsize{{44pt}}{{44pt}}\\selectfont {b['title']}}}
\\setlength{{\\leftm}}{{0.5\\dimexpr\\paperwidth - \\titlelen\\relax}}
\\setlength{{\\rightm}}{{\\dimexpr\\leftm + \\titlelen\\relax}}

\\begin{{tikzpicture}}[remember picture,overlay]
  \\fill[white] (current page.south west) rectangle (current page.north east);

  \\node[anchor=center, inner sep=0pt] at ([xshift=0.50in, yshift=1.52in]current page.center) {{%
    \\includegraphics[width=0.96\\paperwidth]{{{b['art']}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=2.40in]current page.south west) {{%
    {{\\fontsize{{23.5pt}}{{27.5pt}}\\rmfamily\\selectfont\\color{{ink}}{b['sub']}}}%
  }};

  \\node[anchor=south west, inner sep=0pt] at ([xshift=\\leftm, yshift=1.62in]current page.south west) {{%
    {{\\fontsize{{44pt}}{{48pt}}\\rmfamily\\selectfont\\color{{ink}}{b['title']}}}%
  }};

  \\node[anchor=south east, inner sep=0pt] at ([xshift=\\rightm, yshift=0.78in]current page.south west) {{%
    {{\\fontsize{{21.5pt}}{{25pt}}\\rmfamily\\selectfont\\color{{ink}}{b['author']}}}%
  }};

\\end{{tikzpicture}}
\\newpage
\\end{{document}}
"""
    tex_path = f"{scratch_dir}/{b['id']}.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{b['id']}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["lualatex", "-interaction=nonstopmode", f"{b['id']}.tex"], cwd=scratch_dir, check=True)
    subprocess.run(["pdftoppm", "-png", "-r", "150", f"{b['id']}.pdf", f"{scratch_dir}/{b['id']}_out"], cwd=scratch_dir, check=True)
    return f"{scratch_dir}/{b['id']}_out-1.png"

rendered_paths = [render_book(b) for b in books]
rendered_imgs = [Image.open(p) for p in rendered_paths]

w, h = rendered_imgs[0].size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

cols = 4
grid = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'MACHINE LEARNING SYSTEMS TETRALOGY (Neutral Shadows & Pure White Verified)', font=font_title, fill=(20, 20, 20))

for idx, (b, img) in enumerate(zip(books, rendered_imgs)):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))

    border_color = tuple(int(b['color'][i:i+2], 16) for i in (0, 2, 4))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=border_color, width=3)
    draw.text((x + 5, y - 18), b["label"], font=font_label, fill=border_color)

master_contact_path = os.path.join(brain_dir, 'contact_sheet_perfect_neutral_shadows_tetralogy.png')
grid.save(master_contact_path)

# Also create a high-zoom crop of the shadows across all 4 books to verify zero color cast!
crop_h = 160
crop_w = thumb_w
shadow_strip = Image.new('RGB', (thumb_w * cols + (cols + 1) * 20, crop_h + 80), (245, 245, 247))
draw_strip = ImageDraw.Draw(shadow_strip)
draw_strip.text((25, 18), 'ZOOM INSPECTION: GROUND SHADOWS ACROSS ALL 4 BOOKS (Zero Cyan/Green Glow)', font=font_title, fill=(20, 20, 20))

for idx, (b, img) in enumerate(zip(books, rendered_imgs)):
    x = 20 + idx * (thumb_w + 20)
    y = 50
    # Crop the shadow region from the full-res render
    # full h is ~1500, shadow is around y=850 to 1100
    orig_w, orig_h = img.size
    shadow_crop = img.crop((int(orig_w * 0.1), int(orig_h * 0.58), int(orig_w * 0.9), int(orig_h * 0.76)))
    resized_crop = shadow_crop.resize((thumb_w, crop_h), Image.Resampling.LANCZOS)
    shadow_strip.paste(resized_crop, (x, y))
    draw_strip.rectangle([x, y, x + thumb_w, y + crop_h], outline=(100, 100, 100), width=1)
    draw_strip.text((x + 5, y - 16), b["sub"], font=font_label, fill=(40, 40, 40))

out_strip = os.path.join(brain_dir, 'contact_sheet_shadow_zoom_inspection.png')
shadow_strip.save(out_strip)

print("Shadow inspection contact sheets saved successfully!")
