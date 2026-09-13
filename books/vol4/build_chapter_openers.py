#!/usr/bin/env python3
"""
Generate high-resolution labeled and unlabeled isometric blueprint chapter openers
for Volume 4 Part I and Part II (Chapters 02 - 07).
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageChops

BASE = os.path.dirname(os.path.abspath(__file__))
BRAIN = '/Users/VJ/.gemini/antigravity-cli/brain/6702a6d9-86c1-48b2-93d2-f2c949a6e416'

FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'

font = ImageFont.truetype(FONT_PATH, 16)

def create_isometric_grid(w=1400, h=800, spacing=50, color=(226, 235, 242)):
    grid_img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    slope = math.tan(math.radians(30))
    for c in range(-h * 2, h * 2, spacing):
        x1, y1 = 0, c
        x2, y2 = w, int(slope * w + c)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
        x2_n, y2_n = w, int(-slope * w + c)
        draw.line([(x1, y1), (x2_n, y2_n)], fill=color, width=1)
    return grid_img

def draw_pill(draw, text, cx, cy, border_color='#1A4D3E', text_color='#1A4D3E'):
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 16)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    px, py = 12, 6
    bx0, by0 = cx - tw//2 - px, cy - th//2 - py
    bx1, by1 = cx + tw//2 + px, cy + th//2 + py
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill='white', outline=border_color, width=2)
    draw.text((cx - tw//2, cy - th//2 - 1), text, fill=text_color, font=font)
    return (bx0, by0, bx1, by1)

def draw_leader(draw, p_box, p_target, color='#1A4D3E'):
    draw.line([p_box, p_target], fill=color, width=2)
    r = 3.5
    draw.ellipse([p_target[0]-r, p_target[1]-r, p_target[0]+r, p_target[1]+r], fill=color)

def process_chapter(slug, base_filename, labels):
    base_path = os.path.join(BRAIN, base_filename)
    im_base = Image.open(base_path).convert('RGB')
    
    # 1400 x 800 canvas
    canvas = Image.new('RGB', (1400, 800), (255, 255, 255))
    dx = (1400 - im_base.width) // 2
    dy = (800 - im_base.height) // 2
    canvas.paste(im_base, (dx, dy))
    
    # Blend with isometric grid
    grid = create_isometric_grid(1400, 800)
    unlabeled = ImageChops.multiply(canvas, grid)
    
    labeled = unlabeled.copy()
    draw = ImageDraw.Draw(labeled)
    
    for item in labels:
        text = item['text']
        px, py = item['pill']
        tx = dx + item['targ'][0]
        ty = dy + item['targ'][1]
        border = item.get('border', '#1C4E4F')
        tcolor = item.get('color', '#1C4E4F')
        
        # Determine anchor point on pill closest to target
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = (bbox[2] - bbox[0]) // 2 + 12
        if tx < px:
            anchor = (px - tw, py)
        else:
            anchor = (px + tw, py)
            
        draw_leader(draw, anchor, (tx, ty), color=border)
        draw_pill(draw, text, px, py, border_color=border, text_color=tcolor)
        
    # Directories
    ch_dir = os.path.join(BASE, slug)
    png_dir = os.path.join(ch_dir, 'images', 'png')
    webp_dir = os.path.join(ch_dir, 'images', 'webp')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(webp_dir, exist_ok=True)
    
    prefix = slug.split('_')[1] # body, brain, nervous, data, training, evaluation
    
    unlabeled_png = os.path.join(png_dir, f'cover_{prefix}_blueprint.png')
    unlabeled_webp = os.path.join(webp_dir, f'cover_{prefix}_blueprint.webp')
    labeled_png = os.path.join(png_dir, f'cover_{prefix}_blueprint_labeled_print.png')
    labeled_webp = os.path.join(webp_dir, f'cover_{prefix}_blueprint_labeled.webp')
    
    unlabeled.save(unlabeled_png, 'PNG')
    unlabeled.save(unlabeled_webp, 'WEBP', quality=95)
    labeled.save(labeled_png, 'PNG')
    labeled.save(labeled_webp, 'WEBP', quality=95)
    
    labeled.save(os.path.join(BRAIN, f'final_{slug}_labeled.png'))
    print(f'Processed {slug} -> {labeled_png}')

# Chapter 02: The Body
labels_body = [
    {'text': 'stator windings', 'pill': (320, 160), 'targ': (595, 225)},
    {'text': 'rotor inertia', 'pill': (470, 75), 'targ': (520, 230)},
    {'text': 'planetary gearset', 'pill': (1080, 180), 'targ': (780, 410)},
    {'text': 'optical encoder', 'pill': (1140, 360), 'targ': (880, 440)},
    {'text': 'output drive', 'pill': (1100, 550), 'targ': (980, 520)},
    {'text': 'heatsink radiator', 'pill': (320, 570), 'targ': (550, 550)},
]

# Chapter 03: The Brain
labels_brain = [
    {'text': 'token lattice', 'pill': (1120, 140), 'targ': (770, 180)},
    {'text': 'npu tensor core', 'pill': (280, 260), 'targ': (670, 355)},
    {'text': 'stacked hbm dies', 'pill': (260, 430), 'targ': (550, 380)},
    {'text': 'weights streaming bus', 'pill': (1120, 370), 'targ': (800, 430)},
    {'text': 'heat dissipation', 'pill': (340, 620), 'targ': (605, 575)},
]

# Chapter 04: The Nervous System
labels_nervous = [
    {'text': 'real-time mcu', 'pill': (280, 160), 'targ': (510, 240)},
    {'text': 'jitter monitor', 'pill': (1100, 160), 'targ': (740, 260)},
    {'text': 'deterministic bus', 'pill': (240, 350), 'targ': (380, 330)},
    {'text': 'seqlock mailbox', 'pill': (1100, 520), 'targ': (870, 485)},
    {'text': 'reflex filter', 'pill': (360, 570), 'targ': (580, 410), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 05: Physical Data
labels_data = [
    {'text': 'haptic teleop', 'pill': (140, 370), 'targ': (380, 370)},
    {'text': 'stereo vision', 'pill': (670, 90), 'targ': (670, 300)},
    {'text': 'micro-lidar', 'pill': (940, 240), 'targ': (705, 365)},
    {'text': 'calibration target', 'pill': (780, 710), 'targ': (800, 510)},
    {'text': 'deterministic logger', 'pill': (1180, 180), 'targ': (1080, 390)},
    {'text': 'timestamp sync', 'pill': (1260, 680), 'targ': (1200, 560), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 06: Policy Training (Sim-to-Real)
labels_training = [
    {'text': 'virtual twin', 'pill': (160, 200), 'targ': (298, 344), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'friction cones', 'pill': (600, 80), 'targ': (718, 314), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'actuator lag', 'pill': (620, 710), 'targ': (668, 484), 'border': '#1A4D3E', 'color': '#1A4D3E'},
    {'text': 'parameter ranges', 'pill': (380, 690), 'targ': (568, 514), 'border': '#1A4D3E', 'color': '#1A4D3E'},
    {'text': 'contact mismatch', 'pill': (1220, 260), 'targ': (1098, 434), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'physical plant', 'pill': (990, 740), 'targ': (978, 614), 'border': '#1A4D3E', 'color': '#1A4D3E'},
]

# Chapter 07: Closed-Loop Evaluation & Metrology
labels_evaluation = [
    {'text': 'mocap array', 'pill': (530, 70), 'targ': (530, 160)},
    {'text': 'tracking constellation', 'pill': (360, 220), 'targ': (660, 310)},
    {'text': 'dynamometer plate', 'pill': (690, 690), 'targ': (690, 480)},
    {'text': 'confidence ledger', 'pill': (1150, 180), 'targ': (1020, 310)},
    {'text': 'safety envelope', 'pill': (240, 560), 'targ': (350, 440), 'border': '#9B2226', 'color': '#9B2226'},
]

if __name__ == '__main__':
    process_chapter('02_body', 'vol4_ch02_body_base_1789242698098.jpg', labels_body)
    process_chapter('03_brain', 'vol4_ch03_brain_base_1789242713788.jpg', labels_brain)
    process_chapter('04_nervous', 'vol4_ch04_nervous_base_1789242728948.jpg', labels_nervous)
    process_chapter('05_data', 'vol4_ch05_data_clean_1789245664081.jpg', labels_data)
    process_chapter('06_training', 'test_ch06_gemini_opt2_1789246590824.jpg', labels_training)
    process_chapter('07_evaluation', 'vol4_ch07_eval_vibrant_1789292406547.jpg', labels_evaluation)
    print('All 6 chapter openers updated successfully.')
