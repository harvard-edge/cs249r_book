from PIL import Image, ImageDraw, ImageFont
import os

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

im_v1 = Image.open(f"{scratch_dir}/final_v1_out-1.png")
im_v2 = Image.open(f"{scratch_dir}/final_v2_out-1.png")
im_torus = Image.open(f"{scratch_dir}/final_v4_torus_out-1.png")
im_gimbal = Image.open(f"{brain_dir}/verified_physical_ai_candidate4_cover.png")
im_armillary = Image.open(f"{scratch_dir}/ag_armillary_3d_out-1.png")

w, h = im_v1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

# Scenario A: Torus for Physical AI
grid_a = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_a = ImageDraw.Draw(grid_a)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 22)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw_a.text((25, 18), 'SERIES LINEUP A: Physical AI = Kinetic Torus Loop | Agentic = 3D Armillary', font=font_title, fill=(20, 20, 20))

panels_a = [
    (im_v1, 'VOL I: Introduction to (Crimson #A51C30)', (165, 28, 48)),
    (im_v2, 'VOL II: Scaling (ETH Zurich Blue #1F407A)', (31, 64, 122)),
    (im_armillary, 'AGENTIC: 3D Armillary Sphere (Royal Amethyst)', (107, 33, 168)),
    (im_torus, 'PHYSICAL AI: Kinetic Torus Loop (Deep Emerald) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(panels_a):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_a.paste(resized, (x, y))
    draw_a.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_a.text((x + 5, y - 18), label, font=font_label, fill=col)

out_a = os.path.join(brain_dir, 'contact_sheet_series_lineup_torus_physical.png')
grid_a.save(out_a)

# Scenario B: Gimbal for Physical AI
grid_b = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw_b = ImageDraw.Draw(grid_b)
draw_b.text((25, 18), 'SERIES LINEUP B: Physical AI = Gyro Gimbal | Agentic = 3D Armillary', font=font_title, fill=(20, 20, 20))

panels_b = [
    (im_v1, 'VOL I: Introduction to (Crimson #A51C30)', (165, 28, 48)),
    (im_v2, 'VOL II: Scaling (ETH Zurich Blue #1F407A)', (31, 64, 122)),
    (im_armillary, 'AGENTIC: 3D Armillary Sphere (Royal Amethyst)', (107, 33, 168)),
    (im_gimbal, 'PHYSICAL AI: Gyro Multi-Axis Gimbal (Deep Emerald) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(panels_b):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid_b.paste(resized, (x, y))
    draw_b.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw_b.text((x + 5, y - 18), label, font=font_label, fill=col)

out_b = os.path.join(brain_dir, 'contact_sheet_series_lineup_gimbal_physical.png')
grid_b.save(out_b)

print("Both series lineups saved successfully!")
