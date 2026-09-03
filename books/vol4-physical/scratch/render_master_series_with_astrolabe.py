from PIL import Image, ImageDraw, ImageFont
import os

scratch_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/scratch'
brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

im_v1 = Image.open(f"{scratch_dir}/final_v1_out-1.png")
im_v2 = Image.open(f"{scratch_dir}/final_v2_out-1.png")
im_agentic = Image.open(f"{scratch_dir}/ag_astrolabe_out-1.png")
im_phys = Image.open(f"{brain_dir}/verified_physical_ai_candidate4_cover.png")

w, h = im_v1.size
thumb_w = 340
thumb_h = int(h * (thumb_w / w))

grid = Image.new('RGB', (thumb_w * 4 + 100, thumb_h + 120), (245, 245, 247))
draw = ImageDraw.Draw(grid)
font_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 24)
font_label = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 11)

draw.text((25, 18), 'THE MACHINE LEARNING SYSTEMS TETRALOGY (MIT Press Calibrated Series)', font=font_title, fill=(20, 20, 20))

tet_panels = [
    (im_v1, 'VOL I: Introduction to (Harvard Crimson #A51C30)', (165, 28, 48)),
    (im_v2, 'VOL II: Scaling (ETH Zurich Blue #1F407A)', (31, 64, 122)),
    (im_agentic, 'AGENTIC: Autonomous Navigation & Tools (Amethyst #6B21A8) ★', (107, 33, 168)),
    (im_phys, 'PHYSICAL AI: Systems Architecture (Deep Emerald #1A4D3E) ★', (26, 77, 62)),
]

for idx, (img, label, col) in enumerate(tet_panels):
    x = 20 + idx * (thumb_w + 20)
    y = 60
    resized = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    grid.paste(resized, (x, y))
    draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=col, width=3)
    draw.text((x + 5, y - 18), label, font=font_label, fill=col)

out_tet = os.path.join(brain_dir, 'contact_sheet_master_series_with_astrolabe.png')
grid.save(out_tet)
print("Master series with Astrolabe saved successfully!")
