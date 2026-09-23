#!/usr/bin/env python3
"""
Generate publication-quality labeled and unlabeled blueprint covers for Chapter 09: Agent Security.
"""

import argparse
import os
import sys
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG_DIR = os.path.join(REPO_ROOT, 'books', 'vol3', '09_security', 'images', 'png')
OUT_WEBP_DIR = os.path.join(REPO_ROOT, 'books', 'vol3', '09_security', 'images', 'webp')

os.makedirs(OUT_PNG_DIR, exist_ok=True)
os.makedirs(OUT_WEBP_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Render Chapter 09 blueprint cover")
parser.add_argument('--src', default=os.path.join(OUT_PNG_DIR, 'cover_security_blueprint.png'),
                    help="Path to source clean blueprint image")
args, _ = parser.parse_known_args()

# 1. Load source image
src = Image.open(args.src).convert('RGB')
w, h = src.size

# Target size 1400x800
if (w, h) == (1400, 800):
    canvas = src.copy()
else:
    canvas = Image.new('RGB', (1400, 800), (255, 255, 255))
    scale_fit = min(1380 / w, 780 / h)
    nw = int(w * scale_fit)
    nh = int(h * scale_fit)
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    offset_x = (1400 - nw) // 2
    offset_y = (800 - nh) // 2
    canvas.paste(resized, (offset_x, offset_y))

# Save unlabeled base
unlabeled_png = os.path.join(OUT_PNG_DIR, 'cover_security_blueprint.png')
unlabeled_webp = os.path.join(OUT_WEBP_DIR, 'cover_security_blueprint.webp')
canvas.save(unlabeled_png, 'PNG')
canvas.save(unlabeled_webp, 'WEBP', quality=95)
print(f"Saved {unlabeled_png}")

# 2. Render labeled version with 2x supersampling (2800x1600)
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'

labels = [
    {
        'text': 'quarantined reader chamber',
        'pill': (210, 220),
        'targ': (380, 270),
        'border': '#0090B0',
        'color': '#0090B0'
    },
    {
        'text': 'untrusted observation payload',
        'pill': (210, 460),
        'targ': (450, 350),
        'border': '#9B2226',
        'color': '#9B2226'
    },
    {
        'text': 'reference monitor barrier',
        'pill': (700, 75),
        'targ': (765, 215),
        'border': '#2EC4B6',
        'color': '#2EC4B6'
    },
    {
        'text': 'cryptographic capability vault',
        'pill': (1160, 200),
        'targ': (940, 320),
        'border': '#0090B0',
        'color': '#0090B0'
    },
    {
        'text': 'ephemeral credential tokens',
        'pill': (1160, 400),
        'targ': (1005, 415),
        'border': '#D97706',
        'color': '#D97706'
    },
    {
        'text': 'dynamic taint tracking conduit',
        'pill': (1160, 600),
        'targ': (960, 480),
        'border': '#2EC4B6',
        'color': '#2EC4B6'
    }
]

out_w, out_h = 1400, 800
scale = 2
w_sup, h_sup = out_w * scale, out_h * scale

overlay = Image.new('RGBA', (w_sup, h_sup), (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

font = ImageFont.truetype(FONT_PATH, 28 * scale)

pad_x = 22 * scale
min_half_w = 45 * scale
pill_h = 40 * scale
radius = 10 * scale
line_w = 2 * scale
dot_r = 3.5 * scale
safe_margin = 24 * scale

for item in labels:
    text = item['text']
    px = item['pill'][0] * scale
    py = item['pill'][1] * scale
    tx = item['targ'][0] * scale
    ty = item['targ'][1] * scale
    border = item['border']
    tcolor = item['color']

    tw = draw.textlength(text, font=font)
    half_w = max(int(tw / 2 + pad_x), min_half_w)
    bx0 = px - half_w
    by0 = py - pill_h // 2
    bx1 = px + half_w
    by1 = py + pill_h // 2

    # Canvas edge clamping
    if bx0 < safe_margin:
        shift = safe_margin - bx0
        bx0 += shift
        bx1 += shift
        px += shift
    if bx1 > w_sup - safe_margin:
        shift = bx1 - (w_sup - safe_margin)
        bx0 -= shift
        bx1 -= shift
        px -= shift
    if by0 < safe_margin:
        shift = safe_margin - by0
        by0 += shift
        by1 += shift
        py += shift
    if by1 > h_sup - safe_margin:
        shift = by1 - (h_sup - safe_margin)
        by0 -= shift
        by1 -= shift
        py -= shift

    # Anchor point
    if tx < bx0:
        anchor = (bx0, py)
    elif tx > bx1:
        anchor = (bx1, py)
    elif ty < by0:
        anchor = (px, by0)
    elif ty > by1:
        anchor = (px, by1)
    else:
        anchor = (bx1, py)

    # Leader line
    draw.line([anchor, (tx, ty)], fill=border, width=line_w)
    # Target dot
    draw.ellipse([tx - dot_r, ty - dot_r, tx + dot_r, ty + dot_r], fill=border)
    # Pill container
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=radius, fill=(255, 255, 255, 255), outline=border, width=line_w)
    # Centered text
    draw.text((px, py), text, fill=tcolor, font=font, anchor='mm')

overlay_down = overlay.resize((out_w, out_h), Image.Resampling.LANCZOS)
final_im = canvas.copy()
final_im.paste(overlay_down, (0, 0), overlay_down)

labeled_png = os.path.join(OUT_PNG_DIR, 'cover_security_blueprint_labeled_print.png')
labeled_webp = os.path.join(OUT_WEBP_DIR, 'cover_security_blueprint_labeled.webp')
final_im.save(labeled_png, 'PNG')
final_im.save(labeled_webp, 'WEBP', quality=95)
print(f"Saved {labeled_png} and {labeled_webp}")
