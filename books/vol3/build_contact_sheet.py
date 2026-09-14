#!/usr/bin/env python3
"""
Generate the master contact sheet for Volume 3 Chapter Openers (All 18 Chapters).
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

chapters = [
    ('01_introduction', 'Chapter 01 — Introduction', 'The Emerging Agentic Systems Stack'),
    ('02_processor', 'Chapter 02 — The Stochastic Processor Contract', 'Part I: The Stochastic Processor'),
    ('03_deliberation', 'Chapter 03 — Test-Time Deliberation', 'Part I: The Stochastic Processor'),
    ('04_working_sets', 'Chapter 04 — Dynamic Context Working Sets', 'Part II: The Context Memory Hierarchy'),
    ('05_virtual_memory', 'Chapter 05 — Paged Attention Memory', 'Part II: The Context Memory Hierarchy'),
    ('06_episodic_memory', 'Chapter 06 — Episodic Trajectory Memory', 'Part II: The Context Memory Hierarchy'),
    ('07_checkpointing', 'Chapter 07 — Trajectory State Checkpointing', 'Part II: The Context Memory Hierarchy'),
    ('08_actuation', 'Chapter 08 — Environment Tool Actuation', 'Part III: The Actuation Boundary'),
    ('09_virtualization', 'Chapter 09 — Agent Execution Sandboxing', 'Part III: The Actuation Boundary'),
    ('10_interrupts', 'Chapter 10 — Asynchronous Agent Interrupts', 'Part III: The Actuation Boundary'),
    ('11_scheduling', 'Chapter 11 — Trajectory Cluster Scheduling', 'Part IV: The Policy Compiler'),
    ('12_data_flywheel', 'Chapter 12 — Autonomous Trajectory Flywheels', 'Part IV: The Policy Compiler'),
    ('13_sft', 'Chapter 13 — Supervised Trajectory Fine-Tuning', 'Part IV: The Policy Compiler'),
    ('14_rlvr', 'Chapter 14 — RL with Verifiable Rewards', 'Part IV: The Policy Compiler'),
    ('15_multi_agent', 'Chapter 15 — Multi-Agent Distributed Consensus', 'Part V: The Distributed Fleet'),
    ('16_observability', 'Chapter 16 — Agent Trajectory Observability', 'Part V: The Distributed Fleet'),
    ('17_tokenomics', 'Chapter 17 — Agent System Tokenomics', 'Part V: The Distributed Fleet'),
    ('18_conclusion', 'Chapter 18 — The Autonomous Frontier', 'Conclusion & Systems Invariants'),
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

    draw.text((pad_x, 30), 'VOLUME 3: AGENTIC MACHINE LEARNING SYSTEMS', fill='#1A4D3E', font=font_header)
    draw.text((pad_x, 75), 'Master Chapter Opener Blueprint Atlas — All 18 Chapters (Parts I - V)', fill='#4A5568', font=font_sub)
    draw.line([(pad_x, 105), (total_w - pad_x, 105)], fill='#CBD5E1', width=2)

    for idx, (slug, title, part) in enumerate(chapters):
        c = idx % cols
        r = idx // cols
        x = pad_x + c * (card_w + pad_x)
        y = header_h + r * (card_h + 50)

        parts_title = title.split(' — ')
        draw.text((x, y), parts_title[0].upper(), fill='#718096', font=font_card_sub)
        draw.text((x + 95, y), '— ' + parts_title[1], fill='#1A202C', font=font_card_title)
        draw.text((x, y + 20), part, fill='#718096', font=font_card_sub)

        parts_slug = slug.split('_', 1)
        prefix = parts_slug[1] if len(parts_slug) > 1 else parts_slug[0]
        img_path = os.path.join(BASE, slug, 'images', 'png', f'cover_{prefix}_blueprint_labeled_print.png')
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_thumb = img.resize((card_w, card_h), Image.Resampling.LANCZOS)
            sheet.paste(img_thumb, (x, y + 40))
            draw.rectangle([x, y + 40, x + card_w, y + 40 + card_h], outline='#E2E8F0', width=1)

    sheet.save(output_path, 'PNG')
    print(f'Master contact sheet saved to {output_path}')

if __name__ == '__main__':
    out = os.path.join(BASE, 'vol3_master_contact_sheet.png')
    build_contact_sheet(out)
