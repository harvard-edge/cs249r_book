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
    
    # Clean pure white background (preserve base lighting and plinth shadow without synthetic full-canvas grid)
    unlabeled = canvas.copy()
    
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
    {'text': 'stator windings', 'pill': (300, 520), 'targ': (620, 455)},
    {'text': 'planetary gearset', 'pill': (240, 390), 'targ': (590, 385)},
    {'text': 'harmonic drive transmission', 'pill': (260, 260), 'targ': (560, 275)},
    {'text': 'optical encoder disc', 'pill': (440, 100), 'targ': (675, 155)},
    {'text': 'kinematic arm linkage', 'pill': (880, 80), 'targ': (820, 210)},
    {'text': 'stopping envelope', 'pill': (1180, 150), 'targ': (1110, 210), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'contact friction cones', 'pill': (1220, 310), 'targ': (1100, 310), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'parallel jaw gripper', 'pill': (1200, 460), 'targ': (1030, 310)},
]

# Chapter 03: The Brain
labels_brain = [
    {'text': 'multimodal perception rays', 'pill': (240, 140), 'targ': (420, 190)},
    {'text': 'cognitive token lattice', 'pill': (260, 360), 'targ': (640, 270), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'beveled glass housing', 'pill': (680, 70), 'targ': (760, 210)},
    {'text': 'proposal aperture', 'pill': (1120, 360), 'targ': (800, 360)},
    {'text': 'prospective action chunk', 'pill': (1160, 600), 'targ': (990, 510), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'thermal dissipation fins', 'pill': (440, 700), 'targ': (760, 520)},
]

# Chapter 04: The Nervous System
labels_nervous = [
    {'text': 'deterministic 1 khz clock', 'pill': (440, 80), 'targ': (710, 220)},
    {'text': 'jitter-bound envelope', 'pill': (960, 90), 'targ': (830, 160)},
    {'text': 'seqlock proposal mailbox', 'pill': (200, 480), 'targ': (420, 400), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'cbf reflex filter', 'pill': (420, 660), 'targ': (690, 372), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'shielded fieldbus ring', 'pill': (1180, 210), 'targ': (1050, 260)},
    {'text': 'machined heatsink fins', 'pill': (1180, 440), 'targ': (850, 400)},
    {'text': 'sensor feedback return', 'pill': (960, 720), 'targ': (750, 560)},
]

# Chapter 05: Physical Data
labels_data = [
    {'text': 'bilateral haptic master', 'pill': (240, 260), 'targ': (440, 360), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'overhead sensor mast', 'pill': (560, 80), 'targ': (690, 130)},
    {'text': 'stereo vision cone', 'pill': (880, 160), 'targ': (760, 240)},
    {'text': '6-dof follower arm', 'pill': (1160, 260), 'targ': (960, 340)},
    {'text': 'manipulation workpiece', 'pill': (1180, 520), 'targ': (816, 457)},
    {'text': 'action stream conduit', 'pill': (240, 580), 'targ': (580, 500), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'timestamp sync logger', 'pill': (700, 740), 'targ': (710, 600), 'border': '#9B2226', 'color': '#9B2226'},
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
    {'text': 'optical mocap tower', 'pill': (200, 180), 'targ': (348, 275)},
    {'text': 'retroreflective constellation', 'pill': (480, 80), 'targ': (618, 290), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'dynamometer baseplate', 'pill': (260, 640), 'targ': (570, 440), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'safety perimeter rail', 'pill': (190, 480), 'targ': (380, 480), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'confidence ledger cylinder', 'pill': (1160, 520), 'targ': (875, 520), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'tracking ray cone', 'pill': (1160, 240), 'targ': (850, 310)},
]

# Chapter 08: Sensor Perception
labels_perception = [
    {'text': 'stereo optical baseline', 'pill': (360, 70), 'targ': (620, 140)},
    {'text': 'micro-lidar point cloud', 'pill': (880, 80), 'targ': (720, 230), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'metrology step artifact', 'pill': (200, 260), 'targ': (550, 360)},
    {'text': 'extrinsic calibration frame', 'pill': (220, 500), 'targ': (635, 415), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'latency drift vector', 'pill': (480, 720), 'targ': (690, 525), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'spatial covariance ellipsoid', 'pill': (1160, 600), 'targ': (850, 480), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'edge preprocessor box', 'pill': (1160, 340), 'targ': (980, 400)},
]

# Chapter 09: Spatial Memory
labels_memory = [
    {'text': 'octree voxel grid', 'pill': (180, 320), 'targ': (341, 438)},
    {'text': 'occlusion barrier', 'pill': (560, 70), 'targ': (568, 224)},
    {'text': 'spatial belief decay', 'pill': (820, 90), 'targ': (752, 240), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'temporal lease ledger', 'pill': (1220, 160), 'targ': (1079, 278), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'stale token eviction', 'pill': (1160, 710), 'targ': (958, 567), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 10: Grounded Intent
labels_intent = [
    {'text': 'task token prism', 'pill': (200, 80), 'targ': (364, 135), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'deliberative projection', 'pill': (480, 240), 'targ': (508, 264)},
    {'text': 'reachability manifold', 'pill': (460, 690), 'targ': (608, 504)},
    {'text': 'tolerance cylinder', 'pill': (700, 710), 'targ': (694, 434), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'countdown lease ring', 'pill': (880, 160), 'targ': (694, 294), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'expiration tripwire', 'pill': (1180, 160), 'targ': (1003, 239), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 11: Trajectory Planning
labels_planning = [
    {'text': 'actuator origin', 'pill': (240, 140), 'targ': (368, 224)},
    {'text': 'executed spline', 'pill': (380, 520), 'targ': (534, 285), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'prospective chunk', 'pill': (700, 680), 'targ': (795, 341), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'continuity calipers', 'pill': (650, 100), 'targ': (648, 204)},
    {'text': 'stopping suffix', 'pill': (1160, 360), 'targ': (1034, 478), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'terminal buffer pad', 'pill': (1160, 720), 'targ': (1068, 624)},
]

# Chapter 12: Safety Enforcement
labels_enforcement = [
    {'text': 'neural proposal flow', 'pill': (220, 80), 'targ': (396, 208), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'cbf-qp projection filter', 'pill': (744, 80), 'targ': (732, 284)},
    {'text': 'admissible permission gate', 'pill': (1120, 140), 'targ': (937, 274), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'actuator motor drive', 'pill': (1200, 260), 'targ': (1048, 204)},
    {'text': 'unsafe vector sump', 'pill': (1020, 730), 'targ': (868, 624), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'emergency interlock', 'pill': (420, 680), 'targ': (568, 464), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 13: Silicon Placement
labels_placement = [
    {'text': '3d memory stacks', 'pill': (140, 100), 'targ': (218, 294)},
    {'text': 'cognitive npu cluster', 'pill': (180, 280), 'targ': (368, 344), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'thermal dissipation fins', 'pill': (360, 60), 'targ': (468, 144)},
    {'text': 'deterministic mcu enclave', 'pill': (1160, 260), 'targ': (925, 356)},
    {'text': 'shared crossbar bus', 'pill': (380, 720), 'targ': (538, 464), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'qos isolation trench', 'pill': (720, 750), 'targ': (588, 564), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 14: Supervisory Intervention
labels_intervention = [
    {'text': 'autonomous policy', 'pill': (180, 260), 'targ': (335, 353), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'haptic master console', 'pill': (550, 70), 'targ': (568, 154)},
    {'text': 'kinetic authority arbiter', 'pill': (850, 220), 'targ': (738, 344), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'unified drive shaft', 'pill': (1150, 320), 'targ': (1038, 404)},
    {'text': 'intervention event logger', 'pill': (460, 720), 'targ': (657, 575), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 15: Adversarial Verification
labels_verification = [
    {'text': 'virtual simulation crucible', 'pill': (300, 680), 'targ': (455, 559), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'processor-in-the-loop', 'pill': (420, 240), 'targ': (588, 474)},
    {'text': 'hardware-in-the-loop dyno', 'pill': (760, 680), 'targ': (748, 331), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'fault injection needles', 'pill': (1160, 120), 'targ': (926, 187), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'surveillance telemetry', 'pill': (880, 60), 'targ': (768, 144)},
]

# Chapter 16: Deployment Release
labels_release = [
    {'text': 'evidence telemetry pipelines', 'pill': (280, 680), 'targ': (473, 549)},
    {'text': 'gsn argument hierarchy', 'pill': (480, 220), 'targ': (691, 379), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'deployment authorization seal', 'pill': (722, 60), 'targ': (710, 175), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'certified envelope threshold', 'pill': (1160, 320), 'targ': (875, 410), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'revocation interlock', 'pill': (1120, 680), 'targ': (888, 524), 'border': '#9B2226', 'color': '#9B2226'},
]

# Chapter 17: The Epistemic Frontier
labels_frontier = [
    {'text': 'tripartite physical stack', 'pill': (200, 140), 'targ': (390, 253), 'border': '#0090B0', 'color': '#0090B0'},
    {'text': 'actuator body housing', 'pill': (340, 680), 'targ': (528, 454)},
    {'text': 'containment barrier', 'pill': (1180, 220), 'targ': (949, 307), 'border': '#9B2226', 'color': '#9B2226'},
    {'text': 'exploratory horizon prism', 'pill': (880, 60), 'targ': (869, 179), 'border': '#D97706', 'color': '#D97706'},
    {'text': 'unobserved reality chasm', 'pill': (1200, 680), 'targ': (1088, 524)},
]

if __name__ == '__main__':
    process_chapter('02_body', 'vol4_body_anatomy_1789296638170.jpg', labels_body)
    process_chapter('03_brain', 'vol4_brain_cognitive_1789296655524.jpg', labels_brain)
    process_chapter('04_nervous', 'vol4_nervous_bridge_1789296675456.jpg', labels_nervous)
    process_chapter('05_data', 'vol4_ch05_teleop_cell_1789298090568.jpg', labels_data)
    process_chapter('06_training', 'test_ch06_gemini_opt2_1789246590824.jpg', labels_training)
    process_chapter('07_evaluation', 'vol4_ch07_mocap_cell_1789299882214.jpg', labels_evaluation)
    process_chapter('08_perception', 'vol4_ch08_opt_b_bench_1789300399870.jpg', labels_perception)
    process_chapter('09_memory', 'vol4_ch09_memory_base_1789293246965.jpg', labels_memory)
    process_chapter('10_intent', 'vol4_ch10_intent_base_1789293267152.jpg', labels_intent)
    process_chapter('11_planning', 'vol4_ch11_planning_base_1789293290135.jpg', labels_planning)
    process_chapter('12_enforcement', 'vol4_ch12_enforcement_base_1789293313920.jpg', labels_enforcement)
    process_chapter('13_placement', 'vol4_ch13_placement_base_1789293408508.jpg', labels_placement)
    process_chapter('14_intervention', 'vol4_ch14_intervention_base_1789294696657.jpg', labels_intervention)
    process_chapter('15_verification', 'vol4_ch15_verification_base_1789294730403.jpg', labels_verification)
    process_chapter('16_release', 'vol4_ch16_release_base_1789294762541.jpg', labels_release)
    process_chapter('17_frontier', 'vol4_ch17_frontier_base_1789294795242.jpg', labels_frontier)
    print('All 16 chapter openers generated successfully.')
