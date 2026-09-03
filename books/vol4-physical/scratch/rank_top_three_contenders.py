import subprocess, os
from PIL import Image, ImageDraw, ImageFont

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

im_v1 = Image.open(f"{scratch_dir}/quad_v1_out-1.png")
im_v2 = Image.open(f"{scratch_dir}/quad_v2_out-1.png")
im_c1 = Image.open(f"{scratch_dir}/phys_cand1_torus_out-1.png")
im_c4 = Image.open(f"{scratch_dir}/phys_cand4_gimbal_out-1.png")
im_c3 = Image.open(f"{scratch_dir}/phys_cand3_scurve_out-1.png")

w, h = im_v1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

# Build 5-column comparison grid: [Vol 1, Vol 2, Rank 1: Torus, Rank 2: Gimbal, Rank 3: S-Curve]
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
print("Top contender ranking contact sheet saved!")
