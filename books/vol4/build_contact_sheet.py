#!/usr/bin/env python3
"""
Generate the master contact sheet for Volume 4 Chapter Openers (Chapters 01 - 17 + Blueprint Standard Card).
"""

import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'

font_header = ImageFont.truetype(FONT_PATH, 32)
font_sub = ImageFont.truetype(FONT_PATH, 18)
font_card_title = ImageFont.truetype(FONT_PATH, 16)
font_card_sub = ImageFont.truetype(FONT_PATH, 12)
font_std_title = ImageFont.truetype(FONT_PATH, 18)
font_std_body = ImageFont.truetype(FONT_PATH, 13)

chapters = [
    ('01_boundary', 'Chapter 01 — The Causal Boundary', 'Part I: The Machine Anatomy'),
    ('02_body', 'Chapter 02 — The Physical Body', 'Part I: The Machine Anatomy'),
    ('03_brain', 'Chapter 03 — The Learned Brain', 'Part I: The Machine Anatomy'),
    ('04_nervous', 'Chapter 04 — Real-Time Nervous System', 'Part I: The Machine Anatomy'),
    ('05_data', 'Chapter 05 — Physical Data & Ingestion', 'Part II: Teaching the Machine'),
    ('06_training', 'Chapter 06 — Policy Training (Sim-to-Real)', 'Part II: Teaching the Machine'),
    ('07_evaluation', 'Chapter 07 — Closed-Loop Evaluation', 'Part II: Teaching the Machine'),
    ('08_perception', 'Chapter 08 — Sensor Perception', 'Part III: Running the Machine'),
    ('09_memory', 'Chapter 09 — Spatial Memory', 'Part III: Running the Machine'),
    ('10_intent', 'Chapter 10 — Grounded Intent', 'Part III: Running the Machine'),
    ('11_planning', 'Chapter 11 — Trajectory Planning', 'Part III: Running the Machine'),
    ('12_enforcement', 'Chapter 12 — Safety Enforcement', 'Part III: Running the Machine'),
    ('13_placement', 'Chapter 13 — Silicon Placement', 'Part III: Running the Machine'),
    ('14_intervention', 'Chapter 14 — Supervisory Intervention', 'Part IV: Governing the Machine'),
    ('15_verification', 'Chapter 15 — Adversarial Verification', 'Part IV: Governing the Machine'),
    ('16_release', 'Chapter 16 — Deployment Release', 'Part IV: Governing the Machine'),
    ('17_frontier', 'Chapter 17 — The Epistemic Frontier', 'Part IV: Governing the Machine'),
]

def build_contact_sheet(output_path):
    card_w, card_h = 720, 411
    pad_x, pad_y = 40, 50
    header_h = 130
    cols, rows = 3, 6

    total_w = pad_x * 2 + cols * card_w + (cols - 1) * pad_x
    total_h = header_h + rows * (card_h + 50) + pad_y

    sheet = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    draw.text((pad_x, 30), 'VOLUME 4: PHYSICAL AI SYSTEMS', fill='#1A4D3E', font=font_header)
    draw.text((pad_x, 75), 'Master Chapter Opener Blueprint Atlas — All 17 Chapters (Parts I - IV)', fill='#4A5568', font=font_sub)
    draw.line([(pad_x, 105), (total_w - pad_x, 105)], fill='#CBD5E1', width=2)

    for idx, (slug, title, part) in enumerate(chapters):
        c = idx % cols
        r = idx // cols
        x = pad_x + c * (card_w + pad_x)
        y = header_h + r * (card_h + 50)

        draw.text((x, y), title.upper().split(' — ')[0], fill='#718096', font=font_card_sub)
        draw.text((x + 95, y), '— ' + title.split(' — ')[1], fill='#1A202C', font=font_card_title)
        draw.text((x, y + 20), part, fill='#718096', font=font_card_sub)

        prefix = slug.split('_')[1]
        img_path = os.path.join(BASE, slug, 'images', 'png', f'cover_{prefix}_blueprint_labeled_print.png')
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_thumb = img.resize((card_w, card_h), Image.Resampling.LANCZOS)
            sheet.paste(img_thumb, (x, y + 40))
            draw.rectangle([x, y + 40, x + card_w, y + 40 + card_h], outline='#E2E8F0', width=1)

    # 18th Card: Blueprint Standard Specification
    c = 17 % cols
    r = 17 // cols
    x = pad_x + c * (card_w + pad_x)
    y = header_h + r * (card_h + 50) + 40

    draw.rounded_rectangle([x, y, x + card_w, y + card_h], radius=10, fill='#FFFFFF', outline='#1A4D3E', width=2)

    cx = x + 35
    cy = y + 35
    draw.text((cx, cy), 'THE PHYSICAL AI BLUEPRINT STANDARD', fill='#1A4D3E', font=font_std_title)
    cy += 38

    draw.text((cx, cy), 'Canonical 4-Role Semantic Color Contract:', fill='#2D3748', font=font_std_body)
    cy += 24
    draw.text((cx + 10, cy), '• Cyan (#00D4FF): Deliberative planning, simulation & compute', fill='#4A5568', font=font_std_body)
    cy += 20
    draw.text((cx + 10, cy), '• Mint Green (#2EC4B6): Observation telemetry, sensors & MCU enclaves', fill='#4A5568', font=font_std_body)
    cy += 20
    draw.text((cx + 10, cy), '• Warm Amber (#F77F00): Power, torque, dynamic forces & leases', fill='#4A5568', font=font_std_body)
    cy += 20
    draw.text((cx + 10, cy), '• Harvard Crimson (#9B2226): Non-negotiable physical constraints & safety', fill='#4A5568', font=font_std_body)
    cy += 35

    draw.text((cx, cy), 'Key Architecture Features:', fill='#2D3748', font=font_std_body)
    cy += 24
    draw.text((cx + 10, cy), '• Pure white page integration with faint 30 deg blueprint grid', fill='#4A5568', font=font_std_body)
    cy += 20
    draw.text((cx + 10, cy), '• 100% textless base rasters (zero AI typography artifacts)', fill='#4A5568', font=font_std_body)
    cy += 20
    draw.text((cx + 10, cy), '• Crisp vector-composited pill badge keyword callouts', fill='#4A5568', font=font_std_body)
    cy += 20
    draw.text((cx + 10, cy), '• Dual delivery: print-ready PNG and web-optimized WebP', fill='#4A5568', font=font_std_body)

    sheet.save(output_path, 'PNG')
    print(f'Master contact sheet saved to {output_path}')

if __name__ == '__main__':
    out = os.path.join(BASE, 'vol4_master_contact_sheet.png')
    build_contact_sheet(out)
