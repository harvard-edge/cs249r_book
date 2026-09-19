#!/usr/bin/env python3
"""
annotate_publication_hardware_plates.py
======================================
Consolidated generator for all Volume IV publication-quality annotated hardware plates.
Enforces strict styling:
- Crisp rectangular callout cards (rx=0, no rounded bubble boxes)
- Dark slate body (#0F172AF0 or #0F172AFF) with subtle border (#334155)
- Category header bar with 3px top accent stripe
- Clear 4-element pedagogical structure per card:
    * Component: physical hardware component name
    * Role: role in cyber-physical architecture
    * Invariant / Governing Law: physical law, timing invariant, or safety constraint
    * Breaking Point: stress concentration, physical failure mode, thermal limit, or timing deadline
- High-visibility leader lines with anchor dot at card edge and crisp, prominent ARROWHEAD
  pointing directly at the specific target hardware component.
- Clean text wrapping and layout calculation ensuring zero text clipping across all cards.
- Idempotent and deterministic regeneration across all chapters.
"""

import math
import os
import sys
from PIL import Image, ImageDraw, ImageFont

FONT_HEAD_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
if not os.path.exists(FONT_HEAD_PATH):
    FONT_HEAD_PATH = "/System/Library/Fonts/Helvetica.ttc"


def get_fonts(scale=1.0):
    s = scale
    f_head = ImageFont.truetype(FONT_HEAD_PATH, max(12, int(20 * s)))
    f_label = ImageFont.truetype(FONT_HEAD_PATH, max(11, int(15 * s)))
    f_body = ImageFont.truetype(FONT_HEAD_PATH, max(10, int(13.5 * s)))
    return f_head, f_label, f_body


def draw_clean_card(draw, x, y, w, min_h, title, lines, scale=1.0, accent_color='#38BDF8',
                    header_bg=(30, 58, 138, 250), body_bg=(15, 23, 42, 255)):
    """
    Draws a crisp rectangular callout card with header stripe, accent line,
    and cleanly wrapped body text so no words are clipped.
    Returns the actual height of the drawn card.
    """
    s = scale
    f_head, f_label, f_body = get_fonts(scale)

    head_h = int(36 * s)
    line_spacing = int(22 * s)
    pad_x = int(14 * s)
    max_w = w - pad_x * 2

    # Pre-calculate layout and wrap lines if needed
    formatted_rows = []
    for label, text, is_alert in lines:
        prefix = label + ": "
        p_bbox = draw.textbbox((0, 0), prefix, font=f_label)
        p_w = p_bbox[2] - p_bbox[0]

        words = text.split(' ')
        text_lines = []
        cur_line = []

        for word in words:
            test_line = ' '.join(cur_line + [word])
            b = draw.textbbox((0, 0), test_line, font=f_body)
            avail_w = max_w - p_w if len(text_lines) == 0 else max_w - int(14 * s)
            if (b[2] - b[0]) <= avail_w:
                cur_line.append(word)
            else:
                if cur_line:
                    text_lines.append(' '.join(cur_line))
                    cur_line = [word]
                else:
                    text_lines.append(word)
                    cur_line = []
        if cur_line:
            text_lines.append(' '.join(cur_line))

        formatted_rows.append((label, text_lines, is_alert, p_w))

    total_text_lines = sum(len(tl) for _, tl, _, _ in formatted_rows)
    needed_h = head_h + int(12 * s) + total_text_lines * line_spacing + int(12 * s)
    h = max(min_h, needed_h)

    # 1. Base card rectangle: CRISP RECTANGLE (rx=0, NO ROUNDED BUBBLES)
    draw.rectangle([x, y, x + w, y + h], fill=body_bg, outline='#334155', width=max(1, int(1.5 * s)))

    # 2. Header bar
    draw.rectangle([x, y, x + w, y + head_h], fill=header_bg)

    # 3. Top accent stripe (3px)
    draw.line([(x, y), (x + w, y)], fill=accent_color, width=max(2, int(3 * s)))

    # 4. Header title text
    draw.text((x + pad_x, y + int(7 * s)), title, font=f_head, fill='#FFFFFF')

    # 5. Body lines
    cur_y = y + head_h + int(10 * s)
    for label, text_lines, is_alert, p_w in formatted_rows:
        prefix = label + ": "
        label_color = '#F87171' if is_alert else '#94A3B8'
        text_color = '#FECACA' if is_alert else '#E2E8F0'

        for idx, t_line in enumerate(text_lines):
            if idx == 0:
                draw.text((x + pad_x, cur_y), prefix, font=f_label, fill=label_color)
                draw.text((x + pad_x + p_w, cur_y), t_line, font=f_body, fill=text_color)
            else:
                draw.text((x + pad_x + int(14 * s), cur_y), t_line, font=f_body, fill=text_color)
            cur_y += line_spacing

    return h


def draw_leader(draw, card_pt, target_pt, scale=1.0, color='#38BDF8', width=2.8):
    """
    Draws a high-visibility leader line originating at the card edge and pointing
    with a bold, prominent ARROWHEAD directly at the physical hardware component.
    """
    s = scale
    lw = max(2, int(width * s))
    start = card_pt
    end = target_pt

    # 1. Anchor dot at card edge
    r_base = int(5 * s)
    draw.ellipse([start[0] - r_base, start[1] - r_base, start[0] + r_base, start[1] + r_base],
                 fill=color, outline='#0F172A', width=max(1, int(1.5 * s)))

    # 2. Leader line with dark halo for high contrast against light or busy backgrounds
    draw.line([start, end], fill=(15, 23, 42, 230), width=lw + int(2.5 * s))
    draw.line([start, end], fill=color, width=lw)

    # 3. Prominent sharp arrowhead at target_pt pointing directly AT the physical component
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    head_len = int(22 * s)
    a1 = angle + math.pi * 5 / 6
    a2 = angle - math.pi * 5 / 6
    p1 = (end[0] + head_len * math.cos(a1), end[1] + head_len * math.sin(a1))
    p2 = (end[0] + head_len * math.cos(a2), end[1] + head_len * math.sin(a2))

    # Dark shadow behind arrowhead
    p1_s = (end[0] + (head_len + 3 * s) * math.cos(a1), end[1] + (head_len + 3 * s) * math.sin(a1))
    p2_s = (end[0] + (head_len + 3 * s) * math.cos(a2), end[1] + (head_len + 3 * s) * math.sin(a2))
    draw.polygon([end, p1_s, p2_s], fill=(15, 23, 42, 255))
    draw.polygon([end, p1, p2], fill=color)


# ==============================================================================
# PLATE 1: Chapter 01 - NTSB Tempe Test Vehicle & Sensor Suite
# ==============================================================================
def annotate_ntsb_vehicle():
    src = 'books/vol4/01_boundary/images/jpg/fig01_real_ntsb_tempe_vehicle.jpg'
    dst = 'books/vol4/01_boundary/images/jpg/fig01_real_ntsb_tempe_vehicle_annotated.jpg'
    if not os.path.exists(src):
        return
    orig = Image.open(src).convert('RGBA')
    ow, oh = orig.size

    # Wipe the old legend on the original with white
    draw_orig = ImageDraw.Draw(orig)
    draw_orig.rectangle([0, 0, 560, 360], fill=(255, 255, 255, 255))

    target_w = 2160
    target_h = 1350
    car_x = (target_w - ow) // 2
    car_y = 60

    canvas = Image.new('RGBA', (target_w, target_h), (255, 255, 255, 255))
    canvas.paste(orig, (car_x, car_y))

    w, h = target_w, target_h
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w_top = int(480 * scale)
    card_h_top = int(160 * scale)
    card_w_bot = int(580 * scale)
    card_h_bot = int(160 * scale)

    # Card 1: Roof LiDAR (Top Left)
    c1_x, c1_y = int(30 * scale), int(25 * scale)
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w_top, card_h_top, '1. Roof 64-Beam LiDAR (Puck)', [
        ('Component', 'Spinning pulsed laser array (10-20 Hz metric scan)', False),
        ('Role', 'Direct 3D spatial range without photometric ambiguity', False),
        ('Invariant', 'Continuous contact maintained from 82 m to impact', False),
        ('Breaking Point', 'Classification thrashing erased kinematic track history', True)
    ], scale=scale * 0.88, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 right edge directly to roof LiDAR puck
    draw_leader(draw, (c1_x + card_w_top, c1_y + h1 // 2), (car_x + 960, car_y + 160), scale=scale, color='#38BDF8')

    # Card 2: Forward Camera & Radar (Top Right)
    c2_x, c2_y = w - card_w_top - int(30 * scale), int(25 * scale)
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w_top, card_h_top, '2. Forward Vision & Long-Range Radar', [
        ('Component', 'Bifocal optical cameras + 77 GHz millimetric radar', False),
        ('Role', 'Semantic classification and Doppler radial velocity', False),
        ('Invariant', 'Direct Doppler velocity avoids numerical differentiation', False),
        ('Breaking Point', 'Class oscillation toggled between vehicle, bicycle, unknown', True)
    ], scale=scale * 0.88, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 left edge directly to forward camera enclosure on roof rack
    draw_leader(draw, (c2_x, c2_y + h2 // 2), (car_x + 1040, car_y + 230), scale=scale, color='#F43F5E')

    # Card 3: Suppressed Factory AEB (Bottom Left)
    c3_x, c3_y = int(30 * scale), h - card_h_bot - int(35 * scale)
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w_bot, card_h_bot, '3. Suppressed Factory AEB (Missing Safety Referee)', [
        ('Component', 'Factory autonomous emergency braking on chassis CAN bus', False),
        ('Role', 'Deterministic hardware-isolated collision avoidance backstop', False),
        ('Invariant', 'Physical stopping envelope: d_stop = d_lag + d_brake = 46.0 m', False),
        ('Breaking Point', 'Disabled to avoid CAN bus conflict; eliminated safety referee', True)
    ], scale=scale * 0.88, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 3 top edge directly to front bumper radar and OEM braking sensor
    draw_leader(draw, (c3_x + card_w_bot // 2, c3_y), (car_x + 305, car_y + 570), scale=scale, color='#34D399')

    # Card 4: Trunk ADS Compute Rack (Bottom Right)
    c4_x, c4_y = w - card_w_bot - int(30 * scale), h - card_h_bot - int(35 * scale)
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w_bot, card_h_bot, '4. Trunk ADS Compute Rack (1.2 s Suppression Delay)', [
        ('Component', 'High-power heterogeneous compute cluster running autonomy stack', False),
        ('Role', 'Executes multi-modal perception, object tracking & path planning', False),
        ('Invariant', 'Unguided drift distance: d_drift = v0 * t_delay = 23.0 meters', False),
        ('Breaking Point', 'Deliberate 1.2 s delay suppressed braking until 3.8 m from impact', True)
    ], scale=scale * 0.88, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 top edge directly to rear trunk compute bay
    draw_leader(draw, (c4_x + card_w_bot // 2, c4_y), (car_x + 1475, car_y + 470), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(canvas, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 2: Chapter 02 - Broken Gear Tooth Macro-Fracture
# ==============================================================================
def annotate_gear_fracture():
    src = 'books/vol4/02_body/images/jpg/fig02_real_broken_gear_tooth.jpg'
    dst = 'books/vol4/02_body/images/jpg/fig02_real_broken_gear_tooth_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = int(570 * scale)
    card_h = int(210 * scale)

    # Card 1: Intact Gear Tooth
    h1 = draw_clean_card(draw, int(680 * scale), int(30 * scale), card_w, card_h, '1. Intact Gear Tooth (Drive Flank)', [
        ('Component', 'Involute profile spur gear tooth in meshing contact', False),
        ('Role', 'Transmits continuous normal driving torque via contact flank', False),
        ('Normal State', 'Designed for smooth kinematic rolling-sliding contact', False),
        ('Breaking Point', 'Cyclic pitting & spalling under extreme Hertzian stress', True)
    ], scale=scale, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 left edge directly to intact tooth flank
    draw_leader(draw, (int(680 * scale), int(30 * scale) + h1 // 2), (int(540 * scale), int(130 * scale)), scale=scale, color='#38BDF8')

    # Card 2: Tooth Root Fillet
    h2 = draw_clean_card(draw, int(680 * scale), int(270 * scale), card_w, card_h, '2. Tooth Root Fillet (Stress Concentration)', [
        ('Component', 'Concave fillet transition from tooth flank to gear hub rim', False),
        ('Role', 'Distributes cantilever bending moment into structural hub', False),
        ('Governing Law', 'Peak bending stress: sigma_b = (Kv * Wt) / (b * m * Y)', False),
        ('Breaking Point', 'Fatigue micro-crack initiates here under cyclic shock bending', True)
    ], scale=scale, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 left edge directly to tooth root fillet
    draw_leader(draw, (int(680 * scale), int(270 * scale) + h2 // 2), (int(540 * scale), int(370 * scale)), scale=scale, color='#F43F5E')

    # Card 3: Sheared Tooth Fragment
    h3 = draw_clean_card(draw, int(680 * scale), int(510 * scale), card_w, card_h, '3. Sheared Tooth Fragment (Catastrophic Fracture)', [
        ('Component', 'Severed gear tooth detached along root fracture plane', False),
        ('Role', 'Macro-scale fatigue cleavage and brittle shear detachment', False),
        ('Physical Cause', 'Reflected inertia prevents backdriving during sudden impact', False),
        ('Breaking Point', 'Brittle fracture occurs when shock shear stress exceeds tau_ult', True)
    ], scale=scale, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 3 left edge directly to sheared fracture plane
    draw_leader(draw, (int(680 * scale), int(510 * scale) + h3 // 2), (int(520 * scale), int(480 * scale)), scale=scale, color='#F59E0B')

    # Card 4: Reflected Inertia Invariant
    h4 = draw_clean_card(draw, int(680 * scale), int(750 * scale), card_w, card_h, '4. Reflected Inertia Invariant (J_ref = N^2 * J_m)', [
        ('Systems Law', 'Reflected rotor inertia scales with gear ratio squared: N^2', False),
        ('Implication', 'N=50 -> 2500x rotor inertia; N=100 -> 10,000x rotor inertia', False),
        ('Dynamic Trap', 'External impact cannot backdrive motor within 1 ms reflex window', False),
        ('Breaking Point', 'Kinetic energy cannot regenerate; dissipates as fatal metal shear', True)
    ], scale=scale, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 4 left edge directly to gear hub / web
    draw_leader(draw, (int(680 * scale), int(750 * scale) + h4 // 2), (int(280 * scale), int(780 * scale)), scale=scale, color='#34D399')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 3: Chapter 02 - Strain Wave Gear Assembly
# ==============================================================================
def annotate_strain_wave():
    src = 'books/vol4/02_body/images/jpg/fig02_real_strain_wave_gear.jpg'
    dst = 'books/vol4/02_body/images/jpg/fig02_real_strain_wave_gear_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Card 1: Wave Generator (Top Left: x=40, y=40)
    card_w1 = int(540 * scale)
    card_h1 = int(200 * scale)
    h1 = draw_clean_card(draw, int(40 * scale), int(40 * scale), card_w1, card_h1, '1. Wave Generator (High-Speed Input)', [
        ('Component', 'Elliptical steel cam with thin-race flexible bearing', False),
        ('Role', 'Couples to motor rotor; imposes continuous traveling wave', False),
        ('Invariant', 'Input velocity omega_in = N * omega_out', False),
        ('Breaking Point', 'Bearing raceway spalling and cage wear under sustained RPM', True)
    ], scale=scale, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 bottom edge directly to wave generator elliptical bearing
    draw_leader(draw, (int(40 * scale) + card_w1 // 2, int(40 * scale) + h1), (450, 450), scale=scale, color='#38BDF8')

    # Card 2: Flexspline (Right Middle: x=680, y=420)
    card_w2 = int(560 * scale)
    card_h2 = int(210 * scale)
    h2 = draw_clean_card(draw, int(680 * scale), int(420 * scale), card_w2, card_h2, '2. Flexspline (Flexible Toothed Cup)', [
        ('Component', 'Thin-walled flexible alloy cup with external gear teeth', False),
        ('Role', 'Deflects radially to engage circular spline at major axis', False),
        ('Compliance', 'Introduces torsional elasticity (k_s) and non-minimum phase lag', False),
        ('Breaking Point', 'High-cycle bending fatigue at tooth roots & cup diaphragm shear', True)
    ], scale=scale, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 top-middle edge directly to flexspline toothed cup wall
    draw_leader(draw, (int(680 * scale) + card_w2 // 2, int(420 * scale)), (1150, 320), scale=scale, color='#F43F5E')

    # Card 3: Circular Spline (Bottom Left: x=40, y=1050)
    card_w3 = int(550 * scale)
    card_h3 = int(200 * scale)
    h3 = draw_clean_card(draw, int(40 * scale), int(1050 * scale), card_w3, card_h3, '3. Circular Spline (Rigid Ring Gear)', [
        ('Component', 'Rigid outer steel ring with internal teeth (Z_c = Z_f + 2)', False),
        ('Role', 'Grounded to actuator casing or delivers high-torque joint output', False),
        ('Reduction Ratio', 'N = Z_f / (Z_c - Z_f) = 50:1 to 160:1 in a single compact stage', False),
        ('Breaking Point', 'Internal tooth flank wear and mounting bolt shear at peak stall', True)
    ], scale=scale, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 3 top edge directly to circular spline internal teeth
    draw_leader(draw, (int(40 * scale) + card_w3 // 2, int(1050 * scale)), (520, 950), scale=scale, color='#34D399')

    # Card 4: Transmission Invariant (Bottom Right: x=680, y=1050)
    card_w4 = int(580 * scale)
    card_h4 = int(200 * scale)
    h4 = draw_clean_card(draw, int(680 * scale), int(1050 * scale), card_w4, card_h4, '4. Reflected Inertia Invariant (J_ref = N^2 * J_m)', [
        ('Systems Law', 'Reflected rotor inertia scales quadratically: J_ref = N^2 * J_rotor', False),
        ('Dynamic Trap', 'N=100 -> 10,000x rotor inertia; 87% torque consumed spinning rotor', False),
        ('Backdrivability', 'Extreme reflected inertia prevents backdriving; zero impact yield', False),
        ('Breaking Point', 'Step torque commands cause destructive resonant chattering', True)
    ], scale=scale, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 top edge directly to circular spline mounting flange
    draw_leader(draw, (int(680 * scale) + card_w4 // 2, int(1050 * scale)), (1100, 1150), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 4: Chapter 02 - Burnt Motor Stator & Thermal Runaway
# ==============================================================================
def annotate_burnt_stator():
    src = 'books/vol4/02_body/images/jpg/fig02_real_burnt_motor_stator.jpg'
    dst = 'books/vol4/02_body/images/jpg/fig02_real_burnt_motor_stator_annotated.jpg'
    clean_base = '/tmp/clean_stator_v4.jpg'
    if not os.path.exists(src):
        return

    base_path = clean_base if os.path.exists(clean_base) else src
    im = Image.open(base_path).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = 1260
    card_h = 420
    s_card = scale * 0.90

    # Card 1: Stator Copper Phase Windings (Top-Left)
    c1_x, c1_y = 50, 50
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w, card_h, '1. Stator Copper Phase Windings (Heat Source)', [
        ('Component', 'Magnet wire copper bundles packed into laminated iron slots', False),
        ('Role', 'Carries 3-phase AC current synthesizing rotating stator magnetic field', False),
        ('Invariant', 'Joule heating loss: P_loss(t) = I^2(t) * R(T) scales with copper temp', False),
        ('Breaking Point', 'Continuous stall current drives rapid adiabatic temp rise (>20 C/s)', True)
    ], scale=s_card, accent_color='#38BDF8', header_bg=(30, 58, 138, 255), body_bg=(15, 23, 42, 255))
    # Arrow points from Card 1 bottom edge directly down to copper winding bundle
    draw_leader(draw, (c1_x + 400, c1_y + h1), (480, 820), scale=scale, color='#38BDF8')

    # Card 2: Laminated Stator Core (Top-Right)
    c2_x, c2_y = 1360, 50
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w, card_h, '2. Laminated Stator Core (Iron Stack)', [
        ('Component', 'Thin silicon-steel laminations suppressing eddy current losses', False),
        ('Role', 'Lumped thermal capacitance (C_th) & conductive dissipation (theta_JA)', False),
        ('Invariant', 'Thermal time constant tau = theta_JA * C_th (10-15 min) buffers bursts', False),
        ('Breaking Point', 'Core saturation accelerates heating; dissipation lags current spikes', True)
    ], scale=s_card, accent_color='#34D399', header_bg=(6, 78, 59, 255), body_bg=(15, 23, 42, 255))
    # Arrow points from Card 2 bottom edge directly down-left to stator iron core teeth
    draw_leader(draw, (c2_x + 400, c2_y + h2), (1650, 900), scale=scale, color='#34D399')

    # Card 3: Dielectric Slot Liners & Phase Barriers (Bottom-Left)
    c3_x, c3_y = 50, 1530
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w, card_h, '3. Dielectric Slot Liners & Phase Barriers', [
        ('Component', 'Nomex paper & polyimide enamel coating each copper turn', False),
        ('Role', 'Dielectric isolation barrier preventing inter-turn & phase shorts', False),
        ('Invariant', 'Arrhenius aging law: insulation lifetime halves every 10 C rise', False),
        ('Breaking Point', 'Exceeding 155 C Class F rating melts enamel, inducing arc flash short', True)
    ], scale=s_card, accent_color='#F43F5E', header_bg=(159, 18, 57, 255), body_bg=(15, 23, 42, 255))
    # Arrow points from Card 3 top edge directly up-left to Nomex slot liner insulation
    draw_leader(draw, (c3_x + 600, c3_y), (420, 1280), scale=scale, color='#F43F5E')

    # Card 4: Rotor Bore & Bearing End-Bell (Bottom-Right)
    c4_x, c4_y = 1360, 1530
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w, card_h, '4. Rotor Bore & Bearing End-Bell', [
        ('Component', 'Air gap, sealed cartridge bearing & permanent magnet rotor cavity', False),
        ('Role', 'Maintains sub-millimeter magnetic air gap; structural rotor alignment', False),
        ('Invariant', 'Thermal conduction across air gap is poor (k_air = 0.026 W/(m*K))', False),
        ('Breaking Point', 'Bearing grease breakdown & rotor NdFeB demagnetization >120 C', True)
    ], scale=s_card, accent_color='#F59E0B', header_bg=(180, 83, 9, 255), body_bg=(15, 23, 42, 255))
    # Arrow points from Card 4 top edge directly up-right to bearing carrier end-bell
    draw_leader(draw, (c4_x + 400, c4_y), (2550, 750), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 5: Chapter 04 - Industrial Beckhoff EtherCAT Controller
# ==============================================================================
def annotate_ethercat_controller():
    src = 'books/vol4/04_nervous/images/jpg/fig04_real_industrial_ethercat_controller.jpg'
    dst = 'books/vol4/04_nervous/images/jpg/fig04_real_industrial_ethercat_controller_annotated.jpg'
    if not os.path.exists(src):
        return
    orig = Image.open(src).convert('RGBA')
    ow, oh = orig.size

    target_w = ow
    target_h = oh + 500

    canvas = Image.new('RGBA', (target_w, target_h), (248, 250, 252, 255))
    canvas.paste(orig, (0, 0))

    w, h = target_w, target_h
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = int(560 * scale)
    card_h = int(160 * scale)

    # Card 1: Embedded PC Module (Top-Left)
    c1_x, c1_y = int(35 * scale), int(35 * scale)
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w, card_h, '1. Embedded PC Module (TwinCAT Kernel)', [
        ('Component', 'x86 dual-core processor running TwinCAT 3 real-time kernel', False),
        ('Role', 'Executes 1000 Hz motion loops; isolates RT control from OS jitter', False),
        ('Invariant', 'Cyclic jitter bound: J_cycle < 1.0 us via dedicated core reservation', False),
        ('Breaking Point', 'Unbounded heap allocation causes non-deterministic deadline miss', True)
    ], scale=scale * 0.88, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 bottom edge directly to x86 CPU heatsink
    draw_leader(draw, (c1_x + card_w // 2, c1_y + h1), (2150, 1850), scale=scale, color='#38BDF8')

    # Card 2: Modular E-Bus I/O & Drive Slices (Top-Right)
    c2_x, c2_y = w - card_w - int(35 * scale), int(35 * scale)
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w, card_h, '2. Modular E-Bus I/O & Drive Slices', [
        ('Component', 'DIN-rail digital/analog I/O slices (EL1008 inputs, EL2809 outputs)', False),
        ('Role', 'Directly reads joint encoders & drives motor inverter PWM gates', False),
        ('Invariant', '2.5 kV galvanic isolation barrier prevents high-voltage ground loops', False),
        ('Breaking Point', 'Inductive back-EMF spike exceeds avalanche rating of optocouplers', True)
    ], scale=scale * 0.88, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 bottom edge directly to top of terminal slices array
    draw_leader(draw, (c2_x + card_w // 2, c2_y + h2), (4200, 2050), scale=scale, color='#F43F5E')

    # Card 3: EtherCAT Master Port & DC (Bottom-Left)
    c3_x, c3_y = int(35 * scale), h - card_h - int(35 * scale)
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w, card_h, '3. EtherCAT Master & Distributed Clocks (DC)', [
        ('Component', '100BASE-TX shielded RJ-45 port with hardware Distributed Clocks', False),
        ('Role', 'Processes cyclic Process Data Objects (PDOs) on-the-fly at 100 Mbps', False),
        ('Invariant', 'Sub-100 ns multi-axis synchronization via hardware PLL & SYNC0 pulses', False),
        ('Breaking Point', 'Shield grounding fault or packet loss triggers frame CRC discard', True)
    ], scale=scale * 0.88, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 3 top edge directly to yellow RJ-45 cable and EtherCAT port
    draw_leader(draw, (c3_x + card_w // 2, c3_y), (1050, 2150), scale=scale, color='#34D399')

    # Card 4: Fieldbus Watchdog & Fail-Safe STO (Bottom-Right)
    c4_x, c4_y = w - card_w - int(35 * scale), h - card_h - int(35 * scale)
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w, card_h, '4. Fieldbus Watchdog & Fail-Safe STO Machine', [
        ('Component', 'Hardware watchdog timer monitoring cyclic EtherCAT SyncManager', False),
        ('Role', 'De-energizes motor phases if cyclic PDO packet stream is interrupted', False),
        ('Invariant', 'Failsafe response deadline: t_timeout <= 1.0 ms under IEC 61508 SIL-3', False),
        ('Breaking Point', 'Severed Ethernet cable trips Safe Torque Off, engaging holding brakes', True)
    ], scale=scale * 0.88, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 top edge directly to red STO safety terminal slice
    draw_leader(draw, (c4_x + card_w // 2, c4_y), (3400, 2450), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(canvas, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 6: Chapter 08 - Autonomous Sensor Suite Plate
# ==============================================================================
def annotate_sensor_suite():
    src = 'books/vol4/08_perception/images/jpg/fig08_real_sensor_suite.jpg'
    dst = 'books/vol4/08_perception/images/jpg/fig08_real_sensor_suite_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Place cards on the blurred left background and bottom right to keep sensor pod completely clear
    card_w_left = int(510 * scale)
    card_h_left = int(175 * scale)

    # Card 1: Top Perimeter LiDAR Puck (Top Left)
    c1_x, c1_y = int(30 * scale), int(30 * scale)
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w_left, card_h_left, '1. Top Perimeter LiDAR Puck (Direct 3D Range)', [
        ('Component', 'Spinning pulsed ToF laser array (905 nm / 1550 nm eye-safe)', False),
        ('Role', 'Direct 3D spatial range measurement at 10-20 Hz; metric point cloud', False),
        ('Invariant', 'Immune to optical shadow, ambient illumination & headlight bloom', False),
        ('Breaking Point', 'Atmospheric attenuation in dense fog/spray; sparse returns >150 m', True)
    ], scale=scale * 0.90, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 right edge directly to top spinning LiDAR puck window
    draw_leader(draw, (c1_x + card_w_left, c1_y + h1 // 2), (1600, 480), scale=scale, color='#38BDF8')

    # Card 2: High-Dynamic-Range (HDR) Cameras (Left Middle)
    c2_x, c2_y = int(30 * scale), int(230 * scale)
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w_left, card_h_left, '2. Multi-Exposure HDR Cameras (Semantic Vision)', [
        ('Component', 'Multi-exposure automotive CMOS image sensors (>120 dB dynamic range)', False),
        ('Role', 'Serialized raw pixel streams via GMSL2/FPD-Link; semantic classification', False),
        ('Latency Path', 'Exposure midpoint lag (t_exp/2 = 8 ms) + SerDes serialization', False),
        ('Breaking Point', 'Direct solar saturation, lens droplet glare, and nighttime photon noise', True)
    ], scale=scale * 0.90, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 right edge directly across to camera aperture lens
    draw_leader(draw, (c2_x + card_w_left, c2_y + h2 // 2), (1850, 1070), scale=scale, color='#F43F5E')

    # Card 3: Downward Proximity LiDAR (Left Bottom)
    c3_x, c3_y = int(30 * scale), int(430 * scale)
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w_left, card_h_left, '3. Downward Proximity LiDAR (Blind-Zone Defense)', [
        ('Component', 'Angled solid-state ToF sensor covering curb and wheel blind zones', False),
        ('Role', 'Direct millimeter range to ground plane & low obstacles <30 cm height', False),
        ('Safety Invariant', 'Feeds independent hardware reflex stopping gate on MCU', False),
        ('Breaking Point', 'Mud/slush splattering on optical aperture drops range gate to zero', True)
    ], scale=scale * 0.90, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 3 right edge directly to downward proximity sensor horn
    draw_leader(draw, (c3_x + card_w_left, c3_y + h3 // 2), (1420, 1340), scale=scale, color='#34D399')

    # Card 4: Rigid Machined Bracket & Sync Bus (Bottom Right)
    card_w_right = int(520 * scale)
    c4_x, c4_y = w - card_w_right - int(30 * scale), h - int(190 * scale) - int(30 * scale)
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w_right, int(190 * scale), '4. Rigid Billet Mount & Sub-Microsecond Sync Bus', [
        ('Component', 'CNC-machined aluminum mounting arm with PTP / IEEE 1588 sync', False),
        ('Role', 'Preserves invariant SE(3) extrinsic baseline: T_lidar^cam; GPIO strobes <1 us', False),
        ('Thermal Drift', 'Thermal expansion & road vibration induce extrinsic baseline tilt', False),
        ('Breaking Point', '1.15 deg vibration error causes 200 mm spatial error at 10 m range', True)
    ], scale=scale * 0.90, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 top edge directly to aluminum mounting arm
    draw_leader(draw, (c4_x + card_w_right // 2, c4_y), (2300, 1150), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 7: Chapter 13 - Heterogeneous Autonomous SoC Board (Tesla FSD HW3)
# ==============================================================================
def annotate_fsd_board():
    src = 'books/vol4/13_placement/images/jpg/fig13_real_fsd_board.jpg'
    dst = 'books/vol4/13_placement/images/jpg/fig13_real_fsd_board_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = int(580 * scale)
    card_h = int(145 * scale)

    # Card 1: Primary Autonomous SoC Node A (Top Left)
    c1_x, c1_y = int(25 * scale), int(20 * scale)
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w, card_h, '1. Primary Autonomous SoC (Node A)', [
        ('Component', '12-core ARM Cortex-A72 CPU + Dual NPU accelerators (73.7 TOPS)', False),
        ('Role', 'Executes perception neural models and candidate trajectory generation', False),
        ('Memory Bandwidth', '128-bit LPDDR4 memory interface providing 68 GB/s bandwidth', False),
        ('Breaking Point', 'Crossbar contention during camera DMA spikes tail latency to >50 ms', True)
    ], scale=scale * 0.88, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 bottom edge directly to SoC Node A
    draw_leader(draw, (c1_x + card_w // 2, c1_y + h1), (825, 565), scale=scale, color='#38BDF8')

    # Card 2: Redundant Mirror SoC Node B (Top Right)
    c2_x, c2_y = w - card_w - int(25 * scale), int(20 * scale)
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w, card_h, '2. Redundant Mirror SoC (Node B)', [
        ('Component', 'Physically independent, identical compute node on separate silicon', False),
        ('Role', 'Isolated power rail, oscillator, and reset domain; lockstep voting', False),
        ('Redundancy', 'Fail-operational design: assumes control if Node A watchdog lapses', False),
        ('Breaking Point', 'Shared spatial hazards (coolant leak, severe physical chassis intrusion)', True)
    ], scale=scale * 0.88, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 2 bottom edge directly to SoC Node B
    draw_leader(draw, (c2_x + card_w // 2, c2_y + h2), (1475, 565), scale=scale, color='#34D399')

    # Card 3: Multi-Phase PDN (Bottom Left)
    c3_x, c3_y = int(25 * scale), h - card_h - int(25 * scale)
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w, card_h, '3. Multi-Phase Power Delivery (PDN)', [
        ('Component', 'Multi-phase buck regulators, power inductors & decoupling arrays', False),
        ('Role', 'Supplies low-voltage high-current (0.8V @ 80A) to tensor cores', False),
        ('Transient Invariant', 'Suppresses inductive voltage droop: Delta_V = L * (dI/dt)', False),
        ('Breaking Point', 'Unbuffered NPU activation bursts collapse V_core below reset threshold', True)
    ], scale=scale * 0.88, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 3 top edge directly up to PDN buck regulator inductors
    draw_leader(draw, (c3_x + card_w // 2, c3_y), (350, 600), scale=scale, color='#F59E0B')

    # Card 4: GMSL2 SerDes Camera Hub (Bottom Right)
    c4_x, c4_y = w - card_w - int(25 * scale), h - card_h - int(25 * scale)
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w, card_h, '4. GMSL2 SerDes Camera Deserializer Hub', [
        ('Component', '8x FAKRA coaxial high-speed automotive camera deserializer inputs', False),
        ('Role', 'Broadcasts uncompressed camera video streams simultaneously to both SoCs', False),
        ('Zero-Copy DMA', 'Direct memory transfer into accelerator SRAM buffers (<5 ms ingress)', False),
        ('Breaking Point', 'Deserializer lock loss under severe EMI causes frame drops & blindness', True)
    ], scale=scale * 0.88, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 4 top edge directly up-right to FAKRA SerDes deserializer connectors
    draw_leader(draw, (c4_x + card_w // 2, c4_y), (1980, 600), scale=scale, color='#F43F5E')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 8: Chapter 14 - Industrial ABB Robot Teach Pendant
# ==============================================================================
def annotate_teach_pendant():
    src = 'books/vol4/14_intervention/images/jpg/fig14_real_teach_pendant.jpg'
    dst = 'books/vol4/14_intervention/images/jpg/fig14_real_teach_pendant_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = int(520 * scale)
    card_h = int(145 * scale)

    # Card 1: 3-Position Enabling Grip (Top-Left)
    c1_x, c1_y = int(30 * scale), int(20 * scale)
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w, card_h, '1. 3-Position Enabling Device (Deadman)', [
        ('Component', 'Ergonomic spring-loaded 3-position liveman switch on rear handle', False),
        ('Role', 'ISO 10218-1 / ISO 13849-1 mandate: manual authority is held, not granted', False),
        ('Invariant', 'Position 1 (released) = Safe Stop; Pos 2 = Enable; Pos 3 (squeeze) = Safe Stop', False),
        ('Breaking Point', 'Involuntary panic squeeze or grip release asserts Safe Torque Off (STO)', True)
    ], scale=scale * 0.88, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 right edge directly to deadman grip rear handle contour
    draw_leader(draw, (c1_x + card_w, c1_y + h1 // 2), (1150, 1400), scale=scale, color='#38BDF8')

    # Card 2: Hardwired Mushroom E-Stop Button (Top-Right)
    c2_x, c2_y = w - card_w - int(30 * scale), int(20 * scale)
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w, card_h, '2. Hardwired Emergency Stop (Mushroom Button)', [
        ('Component', 'Dual-channel normally-closed mushroom E-stop wired to safety relays', False),
        ('Role', 'Direct physical disconnection of actuator power; bypasses all software', False),
        ('Invariant', 'Hardware Category 0/1 stop latency bounded to T <= 1.0 ms', False),
        ('Breaking Point', 'Slapping button forces instant galvanic disconnection to spring brakes', True)
    ], scale=scale * 0.88, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 bottom edge directly down to red mushroom E-stop button
    draw_leader(draw, (c2_x + card_w // 2, c2_y + h2), (3363, 590), scale=scale, color='#F43F5E')

    # Card 3: Safety Umbilical & Lease Bus (Bottom-Left)
    c3_x, c3_y = int(30 * scale), h - card_h - int(30 * scale)
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w, card_h, '3. Safety Umbilical & Shielded Lease Bus', [
        ('Component', 'Reinforced multi-conductor umbilical carrying dual safety channels', False),
        ('Role', 'Streams cyclic keep-alive heartbeats renewing temporal intervention lease', False),
        ('Invariant', 'Temporal lease expiration window bounded to tau_lease in [50, 100] ms', False),
        ('Breaking Point', 'Severed umbilical or dropped packet drops lease, asserting safe stop', True)
    ], scale=scale * 0.88, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 3 right edge directly to thick orange safety umbilical cable
    draw_leader(draw, (c3_x + card_w, c3_y + h3 // 2), (650, 2100), scale=scale, color='#34D399')

    # Card 4: 3-Axis Proportional Jog Joystick (Bottom-Right)
    c4_x, c4_y = w - card_w - int(30 * scale), h - card_h - int(30 * scale)
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w, card_h, '4. 3-Axis Proportional Jog Joystick', [
        ('Component', 'Hall-effect proportional joystick for manual Cartesian / joint jogging', False),
        ('Role', 'Translates operator manual intent into bounded velocity setpoints', False),
        ('Invariant', 'Speed clamped by downstream safety enforcer: v_tcp <= 250 mm/s (T1 mode)', False),
        ('Breaking Point', 'Excessive manual deflection clamped; enforcer preserves kinematic barrier', True)
    ], scale=scale * 0.88, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 top edge directly up to black proportional joystick knob
    draw_leader(draw, (c4_x + card_w // 2, c4_y), (3272, 1220), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 9: Chapter 15 - NASA Full-Scale HIL Avionics Testbed
# ==============================================================================
def annotate_hil_testbed():
    src = 'books/vol4/15_verification/images/jpg/fig15_real_hil_avionics_testbed.jpg'
    dst = 'books/vol4/15_verification/images/jpg/fig15_real_hil_avionics_testbed_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = int(550 * scale)
    card_h = int(145 * scale)

    # Card 1: Flight-Geometry Structural Ring (Top-Left)
    c1_x, c1_y = int(30 * scale), int(25 * scale)
    h1 = draw_clean_card(draw, c1_x, c1_y, card_w, card_h, '1. Flight-Geometry Structural Avionics Ring', [
        ('Component', 'Full-scale circular airframe mockup matching flight harness geometry', False),
        ('Role', 'Replicates authentic parasitic inductance, capacitance, and ground impedance', False),
        ('Invariant', 'Eliminates simulation reality gap: wire delay tau = L / v_prop and cross-talk', False),
        ('Breaking Point', 'Injected transient verifies noise floor on real harness avoids false trip', True)
    ], scale=scale * 0.84, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 bottom edge directly to structural avionics bulkhead ring
    draw_leader(draw, (c1_x + card_w // 2, c1_y + h1), (1050, 750), scale=scale, color='#38BDF8')

    # Card 2: Production Actuator Control Units (Top-Right)
    c2_x, c2_y = w - card_w - int(30 * scale), int(25 * scale)
    h2 = draw_clean_card(draw, c2_x, c2_y, card_w, card_h, '2. Production Actuator Control Units (ACUs)', [
        ('Component', 'Flight-spec embedded ECUs running low-level servo loops (1-10 kHz)', False),
        ('Role', 'Drives high-power hydraulic servo valves and electromechanical actuators', False),
        ('Invariant', 'Hard real-time execution floor: failsafe fallback must trigger within 1.0 ms', False),
        ('Breaking Point', 'Hydraulic pressure drop or shaft jam tests whether ACU opens bypass valve', True)
    ], scale=scale * 0.84, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 2 bottom edge directly down to production ACU ECU chassis
    draw_leader(draw, (c2_x + card_w // 2, c2_y + h2), (2300, 1200), scale=scale, color='#F43F5E')

    # Card 3: Heavy-Gauge Power & Fieldbus Wire Harness (Bottom-Left)
    c3_x, c3_y = int(30 * scale), h - card_h - int(25 * scale)
    h3 = draw_clean_card(draw, c3_x, c3_y, card_w, card_h, '3. Heavy-Gauge Power & Fieldbus Wire Harness', [
        ('Component', 'Flight-grade twisted-shielded serial buses & high-current 28V DC lines', False),
        ('Role', 'Distributes real electrical power and time-triggered communication frames', False),
        ('Invariant', 'Suppresses inductive switching droop: Delta_V = L_harness * (dI/dt)', False),
        ('Breaking Point', 'Simulated actuator step load stresses DC bus; verifies zero reset brownouts', True)
    ], scale=scale * 0.84, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 3 top edge directly up to heavy-gauge wire harness bundle
    draw_leader(draw, (c3_x + card_w // 2, c3_y), (1100, 1150), scale=scale, color='#34D399')

    # Card 4: Real-Time Fault Injection & Breakout Interface (Bottom-Right)
    c4_x, c4_y = w - card_w - int(30 * scale), h - card_h - int(25 * scale)
    h4 = draw_clean_card(draw, c4_x, c4_y, card_w, card_h, '4. Real-Time Fault Injection & Breakout Interface', [
        ('Component', 'Hardware breakout fixtures & programmable fault insertion units (FIUs)', False),
        ('Role', 'Injects pin-level open circuits, shorts to ground, and bit-level bus errors', False),
        ('Invariant', 'Falsification engine: verifies safety cases under physical degradation', False),
        ('Breaking Point', 'Deterministic bit-flip validates that enforcer isolates rogue commands', True)
    ], scale=scale * 0.84, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 top edge directly up-left to central breakout interface fixture
    draw_leader(draw, (c4_x + card_w // 2, c4_y), (1550, 1150), scale=scale, color='#F59E0B')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


# ==============================================================================
# PLATE 10: Chapter 17 - Humanoid Locomotion Across Unstable Disaster Rubble
# ==============================================================================
def annotate_disaster_rubble():
    src = 'books/vol4/17_frontier/images/jpg/fig17_real_humanoid_disaster_rubble.jpg'
    dst = 'books/vol4/17_frontier/images/jpg/fig17_real_humanoid_disaster_rubble_annotated.jpg'
    if not os.path.exists(src):
        return
    im = Image.open(src).convert('RGBA')
    w, h = im.size
    scale = w / 1280.0
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    card_w = 2080
    card_h = 460
    card_x = 60

    # Card 1: Sensor Head (Top Left)
    c1_y = 60
    h1 = draw_clean_card(draw, card_x, c1_y, card_w, card_h, '1. MultiSense SL Sensor Head & Roll-Cage', [
        ('Component', 'Stereo optical cameras & spinning Hokuyo planar LiDAR scanner', False),
        ('Role', 'Builds 3D point-cloud elevation map for footstep candidate planning', False),
        ('Latency', 'Exteroceptive pipeline period tau_vision >= 100 ms', False),
        ('Breaking Point', 'Exteroceptive blind spot: cannot measure internal rubble shear strength', True)
    ], scale=scale * 0.88, accent_color='#38BDF8', header_bg=(30, 58, 138, 250))
    # Arrow points from Card 1 right edge directly to MultiSense sensor head
    draw_leader(draw, (card_x + card_w, c1_y + h1 // 2), (3100, 450), scale=scale, color='#38BDF8')

    # Card 2: Torso Power & Flight Computer
    c2_y = 640
    h2 = draw_clean_card(draw, card_x, c2_y, card_w, card_h, '2. Torso Hydraulic Power & Flight Computer', [
        ('Component', 'Onboard 21 MPa hydraulic pump, accumulator & real-time core', False),
        ('Role', 'Executes 1000 Hz balance state estimator (1.0 ms deadline)', False),
        ('Safety Invariant', 'Enforces Center-of-Mass momentum bounds without waiting for vision', False),
        ('Breaking Point', 'Hydraulic fluid pressure droop under multi-joint demand saturation', True)
    ], scale=scale * 0.88, accent_color='#34D399', header_bg=(6, 78, 59, 250))
    # Arrow points from Card 2 right edge directly to red torso hydraulic power pack
    draw_leader(draw, (card_x + card_w, c2_y + h2 // 2), (3550, 1160), scale=scale, color='#34D399')

    # Card 3: High-Bandwidth Actuators
    c3_y = 1220
    h3 = draw_clean_card(draw, card_x, c3_y, card_w, card_h, '3. High-Bandwidth Hydraulic Servo Actuators', [
        ('Component', 'Low-inertia hydraulic linear servo cylinders at hips & knees', False),
        ('Role', 'Rapid force control bandwidth (>50 Hz); up to 800 N*m joint torque', False),
        ('Shock Absorption', 'Fluid compressibility mechanically absorbs sudden ground impact', False),
        ('Breaking Point', 'Seal blow-by and valve saturation during violent unmodeled trips', True)
    ], scale=scale * 0.88, accent_color='#A855F7', header_bg=(88, 28, 135, 250))
    # Arrow points from Card 3 right edge directly to knee hydraulic cylinder actuator
    draw_leader(draw, (card_x + card_w, c3_y + h3 // 2), (3100, 2250), scale=scale, color='#A855F7')

    # Card 4: Compliant Foot & 6-Axis F/T Sensor
    c4_y = 1800
    h4 = draw_clean_card(draw, card_x, c4_y, card_w, card_h, '4. Compliant Foot & 6-Axis Sole F/T Sensor', [
        ('Component', '6-axis force/torque load cell + passive elastomeric sole damper', False),
        ('Role', 'Measures Ground Reaction Forces (GRF); detects slip in <5 ms', False),
        ('Morphology', 'Rubber sole passively conforms to cinderblock edges before loop runs', False),
        ('Breaking Point', 'Point-contact shear overload on sharp edge causes sudden ankle rollover', True)
    ], scale=scale * 0.88, accent_color='#F59E0B', header_bg=(180, 83, 9, 250))
    # Arrow points from Card 4 right edge directly to robot foot sole and load cell
    draw_leader(draw, (card_x + card_w, c4_y + h4 // 2), (3500, 2850), scale=scale, color='#F59E0B')

    # Card 5: Unobservable Ground Collapse Interface
    c5_y = 2380
    h5 = draw_clean_card(draw, card_x, c5_y, card_w, card_h, '5. Unobservable Rubble Collapse (Epistemic Limit)', [
        ('Component', 'Loose hollow cinderblocks, crushed gravel & crumbling masonry', False),
        ('Role', 'Latent physical state: brittle fracture, tilt & aggregate shifting', False),
        ('Indistinguishability', 'Solid stone and hollow crumbly block appear identical to cameras', False),
        ('Breaking Point', 'Time-to-harm t_harm <= 30 ms; demands passive mechanics & reflex veto', True)
    ], scale=scale * 0.88, accent_color='#F43F5E', header_bg=(159, 18, 57, 250))
    # Arrow points from Card 5 right edge directly to cinderblock fracture cavity
    draw_leader(draw, (card_x + card_w, c5_y + h5 // 2), (2800, 2850), scale=scale, color='#F43F5E')

    final = Image.alpha_composite(im, overlay).convert('RGB')
    final.save(dst, quality=95)
    print(f"Generated {dst}")


def main():
    print("Annotating all Volume IV publication hardware plates...")
    annotate_ntsb_vehicle()
    annotate_gear_fracture()
    annotate_strain_wave()
    annotate_burnt_stator()
    annotate_ethercat_controller()
    annotate_sensor_suite()
    annotate_fsd_board()
    annotate_teach_pendant()
    annotate_hil_testbed()
    annotate_disaster_rubble()
    print("All Volume IV hardware plates successfully annotated.")


if __name__ == '__main__':
    main()
