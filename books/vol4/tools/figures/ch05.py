"""
book/tools/figures/ch05.py
Figures for Chapter 5: Data & Demonstration Provenance.
Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import os
import math
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig05_compounding_error_flywheel():
    """
    Figure 5.1: Covariate Shift Compounding Error Flywheel & Trajectory Phase Space.
    Shows the O(T^2 epsilon) quadratic error compounding in naive behavioral cloning
    vs O(T epsilon) linear bounded corridor in DAgger / Corrective Aggregation,
    along with trajectory divergence, the lag window [t_div, t_takeover], and episode cutting.
    """
    W = 1000
    H = 580
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    
    # Background Card
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">COVARIATE SHIFT COMPOUNDING ERROR FLYWHEEL &amp; RECOVERY DYNAMICS</text>')
    svg.append(f'<text x="{W/2}" y="45" class="subtitle">Quadratic Divergence O(T²ε) in Naive Behavioral Cloning vs Linear Bounded Tube O(Tε) in DAgger Corrective Aggregation</text>')

    # ==========================================
    # LEFT PANEL: The Two Feedback Mechanisms (Flywheels)
    # ==========================================
    lx = 24
    lw = 460
    
    # 1. Top Left: Naive Behavioral Cloning (Fatal Compounding Loop)
    ty = 68
    th = 240
    svg.append(f'<rect x="{lx}" y="{ty}" width="{lw}" height="{th}" rx="8" fill="{CORAL}" fill-opacity="0.03" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{lx}" y="{ty}" width="{lw}" height="26" rx="8" fill="{CORAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{lx+14}" y="{ty+18}" font-size="10" font-weight="700" fill="{CORAL}">✕ NAIVE BEHAVIORAL CLONING (OPEN-LOOP DRIFT FLYWHEEL)</text>')
    svg.append(f'<text x="{lx+lw-14}" y="{ty+18}" font-size="9.5" font-weight="700" fill="{CRIMSON}" text-anchor="end">Error: O(T²ε)</text>')

    # 4 Cyclic Boxes for BC
    # Box 1: Small Single-step error
    b1_x, b1_y, bw, bh = lx + 16, ty + 38, 195, 42
    svg.append(f'<rect x="{b1_x}" y="{b1_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{b1_x+8}" y="{b1_y+16}" font-size="8.5" font-weight="700" fill="{INK}">1. Single-Step Perturbation ε</text>')
    svg.append(f'<text x="{b1_x+8}" y="{b1_y+30}" font-size="7.5" fill="{SLATE}">Friction / torque / sensor noise</text>')

    # Arrow 1->2
    svg.append(f'<line x1="{b1_x+bw}" y1="{b1_y+21}" x2="{b1_x+bw+30}" y2="{b1_y+21}" stroke="{CORAL}" stroke-width="1.3" marker-end="url(#arr-coral)"/>')

    # Box 2: Out of distribution state
    b2_x, b2_y = lx + 245, ty + 38
    svg.append(f'<rect x="{b2_x}" y="{b2_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{b2_x+8}" y="{b2_y+16}" font-size="8.5" font-weight="700" fill="{CORAL}">2. State Exits Support</text>')
    svg.append(f'<text x="{b2_x+8}" y="{b2_y+30}" font-size="7.5" fill="{SLATE}">s_(t+1) ∉ supp(d^π*)</text>')

    # Arrow 2->3 (down)
    svg.append(f'<line x1="{b2_x+bw/2}" y1="{b2_y+bh}" x2="{b2_x+bw/2}" y2="{b2_y+bh+22}" stroke="{CORAL}" stroke-width="1.3" marker-end="url(#arr-coral)"/>')

    # Box 3: Ungrounded action
    b3_x, b3_y = lx + 245, ty + 102
    svg.append(f'<rect x="{b3_x}" y="{b3_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{b3_x+8}" y="{b3_y+16}" font-size="8.5" font-weight="700" fill="{CORAL}">3. Arbitrary Action â</text>')
    svg.append(f'<text x="{b3_x+8}" y="{b3_y+30}" font-size="7.5" fill="{SLATE}">Zero constraint from ERM loss</text>')

    # Arrow 3->4 (left)
    svg.append(f'<line x1="{b3_x}" y1="{b3_y+21}" x2="{b3_x-30}" y2="{b3_y+21}" stroke="{CORAL}" stroke-width="1.3" marker-end="url(#arr-coral)"/>')

    # Box 4: Compounding error
    b4_x, b4_y = lx + 16, ty + 102
    svg.append(f'<rect x="{b4_x}" y="{b4_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<text x="{b4_x+8}" y="{b4_y+16}" font-size="8.5" font-weight="700" fill="{CRIMSON}">4. Amplified Deviation</text>')
    svg.append(f'<text x="{b4_x+8}" y="{b4_y+30}" font-size="7.5" fill="{SLATE}">Pushed further off-manifold</text>')

    # Arrow 4->1 (up, loop)
    svg.append(f'<line x1="{b4_x+bw/2}" y1="{b4_y}" x2="{b4_x+bw/2}" y2="{b1_y+bh}" stroke="{CRIMSON}" stroke-width="1.3" marker-end="url(#arr-crimson)"/>')

    # BC Summary equation box
    svg.append(f'<rect x="{lx+16}" y="{ty+156}" width="{lw-32}" height="{72}" rx="5" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{lx+24}" y="{ty+174}" font-size="8.5" font-weight="700" fill="{CRIMSON}">Mathematical Compounding Mechanism:</text>')
    svg.append(f'<text x="{lx+24}" y="{ty+193}" font-size="8" font-family="monospace" fill="{INK}">E[Error_BC] ≤ ∑ t·ε = [T(T+1)/2]·ε = O(T²ε)</text>')
    svg.append(f'<text x="{lx+24}" y="{ty+211}" font-size="7.5" fill="{SLATE}">Example (T = 500 steps, ε = 0.01) ⇒ Cumulative Error Factor = 2,500×</text>')

    # 2. Bottom Left: DAgger / Corrective Aggregation (Stabilizing Loop)
    by = 320
    bh_p = 240
    svg.append(f'<rect x="{lx}" y="{by}" width="{lw}" height="{bh_p}" rx="8" fill="{TEAL}" fill-opacity="0.03" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{lx}" y="{by}" width="{lw}" height="26" rx="8" fill="{TEAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{lx+14}" y="{by+18}" font-size="10" font-weight="700" fill="{PETROL}">✓ CORRECTIVE AGGREGATION / DAgger (STABILIZING FLYWHEEL)</text>')
    svg.append(f'<text x="{lx+lw-14}" y="{by+18}" font-size="9.5" font-weight="700" fill="{TEAL}" text-anchor="end">Error: O(Tε)</text>')

    # 4 Cyclic Boxes for DAgger
    d1_x, d1_y = lx + 16, by + 38
    svg.append(f'<rect x="{d1_x}" y="{d1_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+16}" font-size="8.5" font-weight="700" fill="{INK}">1. Policy Visits s ~ d_policy</text>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+30}" font-size="7.5" fill="{SLATE}">Learner visits perturbed state</text>')

    svg.append(f'<line x1="{d1_x+bw}" y1="{d1_y+21}" x2="{d1_x+bw+30}" y2="{d1_y+21}" stroke="{TEAL}" stroke-width="1.3" marker-end="url(#arr-teal)"/>')

    d2_x, d2_y = lx + 245, by + 38
    svg.append(f'<rect x="{d2_x}" y="{d2_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{d2_x+8}" y="{d2_y+16}" font-size="8.5" font-weight="700" fill="{PETROL}">2. Expert Labels π*(s)</text>')
    svg.append(f'<text x="{d2_x+8}" y="{d2_y+30}" font-size="7.5" fill="{SLATE}">Recovery action demonstrated</text>')

    svg.append(f'<line x1="{d2_x+bw/2}" y1="{d2_y+bh}" x2="{d2_x+bw/2}" y2="{d2_y+bh+22}" stroke="{TEAL}" stroke-width="1.3" marker-end="url(#arr-teal)"/>')

    d3_x, d3_y = lx + 245, by + 102
    svg.append(f'<rect x="{d3_x}" y="{d3_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{d3_x+8}" y="{d3_y+16}" font-size="8.5" font-weight="700" fill="{TEAL}">3. Aggregate to Dataset D</text>')
    svg.append(f'<text x="{d3_x+8}" y="{d3_y+30}" font-size="7.5" fill="{SLATE}">supp(D) ⊇ recovery corridor</text>')

    svg.append(f'<line x1="{d3_x}" y1="{d3_y+21}" x2="{d3_x-30}" y2="{d3_y+21}" stroke="{TEAL}" stroke-width="1.3" marker-end="url(#arr-teal)"/>')

    d4_x, d4_y = lx + 16, by + 102
    svg.append(f'<rect x="{d4_x}" y="{d4_y}" width="{bw}" height="{bh}" rx="5" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{d4_x+8}" y="{d4_y+16}" font-size="8.5" font-weight="700" fill="{PETROL}">4. Closed-Loop Recovery</text>')
    svg.append(f'<text x="{d4_x+8}" y="{d4_y+30}" font-size="7.5" fill="{SLATE}">Restoring torque restores state</text>')

    svg.append(f'<line x1="{d4_x+bw/2}" y1="{d4_y}" x2="{d4_x+bw/2}" y2="{d1_y+bh}" stroke="{PETROL}" stroke-width="1.3" marker-end="url(#arr-petrol)"/>')

    # DAgger Summary equation box
    svg.append(f'<rect x="{lx+16}" y="{by+156}" width="{lw-32}" height="{72}" rx="5" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{lx+24}" y="{by+174}" font-size="8.5" font-weight="700" fill="{PETROL}">Mathematical Linear Bounded Corridor:</text>')
    svg.append(f'<text x="{lx+24}" y="{by+193}" font-size="8" font-family="monospace" fill="{INK}">E[Error_CA] ≤ T·ε = O(Tε)  [Ross &amp; Bagnell 2011]</text>')
    svg.append(f'<text x="{lx+24}" y="{by+211}" font-size="7.5" fill="{SLATE}">Example (T = 500 steps, ε = 0.01) ⇒ Cumulative Error Factor = 5.0× (500× reduction!)</text>')

    # ==========================================
    # RIGHT PANEL: Phase Space & Episode Cut Point
    # ==========================================
    rx = 504
    rw = 472
    ry = 68
    rh = 492
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="8" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="26" rx="8" fill="{NAVY}" fill-opacity="0.08"/>')
    svg.append(f'<text x="{rx+14}" y="{ry+18}" font-size="10" font-weight="700" fill="{NAVY}">TRAJECTORY DIVERGENCE &amp; INTERVENTION CUT BOUNDARY</text>')

    # Coordinate Plot inside Right Panel
    px = rx + 55
    py = ry + 165
    pw = 390
    ph = 145

    # 1. Hazard / E-Stop Boundary (Top Zone)
    svg.append(f'<rect x="{px}" y="{py-115}" width="{pw}" height="26" fill="{CORAL}" fill-opacity="0.1"/>')
    svg.append(f'<line x1="{px}" y1="{py-89}" x2="{px+pw}" y2="{py-89}" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="4,2"/>')
    svg.append(f'<text x="{px+8}" y="{py-99}" font-size="7.5" font-weight="700" fill="{CORAL}">HARDWARE HAZARD / E-STOP BOUNDARY (+12.0 mm)</text>')

    # 2. Nominal Envelope Corridor [-1.5mm, +1.5mm] -> y-range [py-16, py+16]
    svg.append(f'<rect x="{px}" y="{py-16}" width="{pw}" height="32" fill="{TEAL}" fill-opacity="0.08" stroke="{TEAL}" stroke-width="0.8" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{px+pw-8}" y="{py-6}" font-size="7" font-weight="700" fill="{TEAL}" text-anchor="end">Nominal Operating Corridor [±1.5 mm]</text>')

    # Axes
    svg.append(f'<line x1="{px}" y1="{py}" x2="{px+pw+15}" y2="{py}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{px+pw+15}" y="{py+14}" font-size="7.5" font-weight="600" fill="{SLATE}" text-anchor="end">Time t (s) →</text>')

    svg.append(f'<line x1="{px}" y1="{py+45}" x2="{px}" y2="{py-115}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{px-35}" y="{py-40}" font-size="8" font-weight="600" fill="{SLATE}" transform="rotate(-90 {px-35} {py-40})" text-anchor="middle">Tracking Error e(t) (mm) →</text>')

    # Nominal Trajectory (Center Line)
    svg.append(f'<line x1="{px}" y1="{py}" x2="{px+pw}" y2="{py}" stroke="{PETROL}" stroke-width="2"/>')
    svg.append(f'<text x="{px+10}" y="{py-4}" font-size="7.5" font-weight="700" fill="{PETROL}">π* Nominal Target</text>')

    # Key Timestamps X coordinates
    t_div_x = px + 85       # t_div = 1.2s
    t_take_x = px + 170     # t_takeover = 2.0s (Δt = 800ms human delay)
    t_rec_x = px + 280      # t_recovery = 3.5s

    # Shaded Reaction Delay Region [t_div, t_takeover]
    svg.append(f'<rect x="{t_div_x}" y="{py-89}" width="{t_take_x-t_div_x}" height="{89+35}" fill="{CORAL}" fill-opacity="0.12"/>')
    svg.append(f'<line x1="{t_div_x}" y1="{py-89}" x2="{t_div_x}" y2="{py+40}" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,3"/>')
    svg.append(f'<line x1="{t_take_x}" y1="{py-89}" x2="{t_take_x}" y2="{py+40}" stroke="{PURPLE}" stroke-width="1.2" stroke-dasharray="3,3"/>')

    # Naive BC Divergent Path (Quadratic runaway to hazard)
    bc_path = f"M {px} {py} L {t_div_x} {py} Q {t_take_x-10} {py-42} {t_take_x+55} {py-89}"
    svg.append(f'<path d="{bc_path}" fill="none" stroke="{CORAL}" stroke-width="2.2" stroke-dasharray="5,2"/>')
    svg.append(f'<text x="{t_take_x+60}" y="{py-76}" font-size="7.5" font-weight="700" fill="{CORAL}">Naive BC: O(T²) Runaway</text>')

    # Human Takeover & Expert Recovery Path
    rec_path = f"M {t_div_x} {py} Q {t_take_x-5} {py-46} {t_take_x} {py-48} Q {t_take_x+50} {py-52} {t_rec_x} {py}"
    svg.append(f'<path d="{rec_path}" fill="none" stroke="{TEAL}" stroke-width="2.5"/>')

    # Mark Key Nodes
    # 1. Divergence Point t_div
    svg.append(f'<circle cx="{t_div_x}" cy="{py}" r="4.5" fill="{CORAL}" stroke="{BG_WHITE}" stroke-width="1.5"/>')
    svg.append(f'<text x="{t_div_x}" y="{py+28}" font-size="8" font-weight="700" fill="{CORAL}" text-anchor="middle">t_div</text>')
    svg.append(f'<text x="{t_div_x}" y="{py+38}" font-size="6.5" fill="{MUTED}" text-anchor="middle">(Divergence)</text>')

    # 2. Takeover Point t_takeover
    svg.append(f'<circle cx="{t_take_x}" cy="{py-48}" r="5" fill="{PURPLE}" stroke="{BG_WHITE}" stroke-width="1.5"/>')
    svg.append(f'<text x="{t_take_x}" y="{py-58}" font-size="7.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">t_takeover (Override)</text>')
    svg.append(f'<text x="{t_take_x}" y="{py+28}" font-size="8" font-weight="700" fill="{PURPLE}" text-anchor="middle">t_take</text>')
    svg.append(f'<text x="{t_take_x}" y="{py+38}" font-size="6.5" fill="{MUTED}" text-anchor="middle">(Intervention)</text>')

    # 3. Recovery Complete t_rec
    svg.append(f'<circle cx="{t_rec_x}" cy="{py}" r="4.5" fill="{TEAL}" stroke="{BG_WHITE}" stroke-width="1.5"/>')
    svg.append(f'<text x="{t_rec_x}" y="{py+28}" font-size="8" font-weight="700" fill="{TEAL}" text-anchor="middle">t_rec</text>')
    svg.append(f'<text x="{t_rec_x}" y="{py+38}" font-size="6.5" fill="{MUTED}" text-anchor="middle">(Recovered)</text>')

    # Callout Bracket for [t_div, t_takeover] lag window
    svg.append(f'<line x1="{t_div_x+2}" y1="{py+54}" x2="{t_take_x-2}" y2="{py+54}" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<polyline points="{t_div_x+2},{py+51} {t_div_x+2},{py+57}" fill="none" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<polyline points="{t_take_x-2},{py+51} {t_take_x-2},{py+57}" fill="none" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{(t_div_x+t_take_x)/2}" y="{py+66}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">Reaction Delay (300-800 ms)</text>')

    # Bottom Instructions: The 3 Rules of Episode Curation
    iy = ry + 275
    svg.append(f'<rect x="{rx+12}" y="{iy}" width="{rw-24}" height="{195}" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+22}" y="{iy+18}" font-size="9" font-weight="700" fill="{NAVY}">THE 3 DATA CURATION LAWS FOR PHYSICAL INTERVENTIONS:</text>')

    rules = [
        ("1. Truncate Autonomous Rollout at t_div (NOT t_takeover):", "Cut the autonomous trajectory where the policy drifted off-manifold. Prevents neural networks from learning that approaching a hazard precedes task success.", CORAL),
        ("2. Discard Corrupted Drift Interval [t_div, t_takeover]:", "Purge the 30-80 execution frames logged while human reaction latency delayed takeover. These transitions contain diverging actions falsely logged as nominal.", CRIMSON),
        ("3. Ingest Recovery Demonstration [t_takeover, t_rec] into DAgger Buffer:", "Treat the human rescue as corrective supervision initialized on the learner-induced distribution d_policy, collapsing error compounding from O(T²) to O(T).", TEAL)
    ]
    for idx, (rhdr, rtxt, rcol) in enumerate(rules):
        ry_pos = iy + 36 + idx * 52
        svg.append(f'<circle cx="{rx+26}" cy="{ry_pos-3}" r="3.5" fill="{rcol}"/>')
        svg.append(f'<text x="{rx+36}" y="{ry_pos}" font-size="8" font-weight="700" fill="{INK}">{rhdr}</text>')
        # Multi-line text wrapping for description
        if idx == 0:
            svg.append(f'<text x="{rx+36}" y="{ry_pos+13}" font-size="7.2" fill="{SLATE}">Cut rollout where policy drifted. Prevents neural networks from</text>')
            svg.append(f'<text x="{rx+36}" y="{ry_pos+24}" font-size="7.2" fill="{SLATE}">learning that approaching a hazard precedes goal completion.</text>')
        elif idx == 1:
            svg.append(f'<text x="{rx+36}" y="{ry_pos+13}" font-size="7.2" fill="{SLATE}">Purge 30-80 frames logged during human reaction delay. These</text>')
            svg.append(f'<text x="{rx+36}" y="{ry_pos+24}" font-size="7.2" fill="{SLATE}">transitions contain divergent commands falsely logged as nominal.</text>')
        else:
            svg.append(f'<text x="{rx+36}" y="{ry_pos+13}" font-size="7.2" fill="{SLATE}">Treat rescue as expert feedback on policy distribution d_policy,</text>')
            svg.append(f'<text x="{rx+36}" y="{ry_pos+24}" font-size="7.2" fill="{SLATE}">expanding support to recovery corridor and bounding error to O(Tε).</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/05-data/figures/fig05_compounding_error_flywheel.svg", "\n".join(svg))



def gen_fig05_collector_coverage_ledger():
    """
    Figure 5.2: Empirical State-Action Occupancy vs External Scenario Ledger.
    Illustrates how cautious teleoperation leaves 98.7% of the required operational grid unvisited
    despite low training/validation loss, compared with exploratory collection and relational ledger joins.
    """
    W = 1000
    H = 540
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)

    # Background Card
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">COLLECTOR EMPIRICAL OCCUPANCY VS SCENARIO LEDGER SUPPORT</text>')
    svg.append(f'<text x="{W/2}" y="45" class="subtitle">Why Low Validation Loss on Held-Out Demonstrations Conceals Critical Operational Blind Spots</text>')

    # 3 Column Cards
    # Col 1: Cautious Collector
    c1_x = 24
    cw = 295
    cy = 68
    ch = 450

    svg.append(f'<rect x="{c1_x}" y="{cy}" width="{cw}" height="{ch}" rx="8" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{c1_x}" y="{cy}" width="{cw}" height="26" rx="8" fill="{CORAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c1_x+cw/2}" y="{cy+18}" font-size="10" font-weight="700" fill="{CORAL}" text-anchor="middle">COLLECTOR A: CAUTIOUS TELEOP</text>')

    # Grid 1: Cautious (Concentrated in 8 cells)
    g1_x = c1_x + 42
    g1_y = cy + 42
    gw = 210
    gh = 150

    # Draw grid background
    svg.append(f'<rect x="{g1_x}" y="{g1_y}" width="{gw}" height="{gh}" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    
    # 6x5 grid subdivisions representing 30x20
    for i in range(1, 6):
        svg.append(f'<line x1="{g1_x + i*gw/6}" y1="{g1_y}" x2="{g1_x + i*gw/6}" y2="{g1_y+gh}" stroke="{BORDER}" stroke-width="0.5" stroke-dasharray="2,2"/>')
    for j in range(1, 5):
        svg.append(f'<line x1="{g1_x}" y1="{g1_y + j*gh/5}" x2="{g1_x+gw}" y2="{g1_y + j*gh/5}" stroke="{BORDER}" stroke-width="0.5" stroke-dasharray="2,2"/>')

    # Hatching for 592 unvisited cells
    svg.append(f'<rect x="{g1_x}" y="{g1_y}" width="{gw}" height="{gh}" fill="{CORAL}" fill-opacity="0.04"/>')

    # Highlight only center 8 cells (dense hotspot)
    spot_x = g1_x + gw/2 - 16
    spot_y = g1_y + gh/2 - 12
    svg.append(f'<rect x="{spot_x}" y="{spot_y}" width="32" height="24" rx="3" fill="{CORAL}" fill-opacity="0.9" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<text x="{spot_x+16}" y="{spot_y+15}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">10⁶ pts</text>')

    # Axis Labels & Ticks for Grid 1
    svg.append(f'<text x="{g1_x}" y="{g1_y+gh+12}" font-size="6.5" fill="{MUTED}" text-anchor="middle">-15°</text>')
    svg.append(f'<text x="{g1_x+gw/2}" y="{g1_y+gh+12}" font-size="6.5" fill="{MUTED}" text-anchor="middle">0°</text>')
    svg.append(f'<text x="{g1_x+gw}" y="{g1_y+gh+12}" font-size="6.5" fill="{MUTED}" text-anchor="middle">+15°</text>')
    svg.append(f'<text x="{g1_x+gw/2}" y="{g1_y+gh+24}" font-size="7.5" font-weight="600" fill="{SLATE}" text-anchor="middle">Contact Angle θ (deg)</text>')

    svg.append(f'<text x="{g1_x-6}" y="{g1_y+6}" font-size="6.5" fill="{MUTED}" text-anchor="end">20N</text>')
    svg.append(f'<text x="{g1_x-6}" y="{g1_y+gh/2+3}" font-size="6.5" fill="{MUTED}" text-anchor="end">10N</text>')
    svg.append(f'<text x="{g1_x-6}" y="{g1_y+gh}" font-size="6.5" fill="{MUTED}" text-anchor="end">0N</text>')
    svg.append(f'<text x="{g1_x-22}" y="{g1_y+gh/2}" font-size="7.5" font-weight="600" fill="{SLATE}" transform="rotate(-90 {g1_x-22} {g1_y+gh/2})" text-anchor="middle">Normal Force F_N (N)</text>')

    # Metrics Card 1
    my1 = g1_y + gh + 36
    svg.append(f'<rect x="{c1_x+12}" y="{my1}" width="{cw-24}" height="195" rx="6" fill="{CORAL}" fill-opacity="0.04" stroke="{CORAL}" stroke-width="0.8"/>')
    
    m1_items = [
        ("Total Recorded Transitions:", "1,000,000 samples @ 100 Hz"),
        ("Scenario Grid Coverage:", "8 / 600 cells (1.3%)"),
        ("Zero-Count Blind Spots:", "592 / 600 cells (98.7%!)"),
        ("Held-Out Validation Loss:", "L_val = 0.001 rad (Deceptive)"),
        ("Compounding Drift Risk:", "CATASTROPHIC at ±2.5°"),
        ("Ingestion Policy Decision:", "REFUSE / REQUIRE DAgger")
    ]
    for idx, (lbl, val) in enumerate(m1_items):
        svg.append(f'<text x="{c1_x+20}" y="{my1+18+idx*29}" font-size="8" font-weight="700" fill="{INK}">{lbl}</text>')
        col_val = CRIMSON if "98.7" in val or "CATASTROPHIC" in val or "REFUSE" in val else SLATE
        svg.append(f'<text x="{c1_x+20}" y="{my1+30+idx*29}" font-size="7.5" font-weight="600" fill="{col_val}">{val}</text>')


    # Col 2: Exploratory Collector
    c2_x = 352
    svg.append(f'<rect x="{c2_x}" y="{cy}" width="{cw}" height="{ch}" rx="8" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{c2_x}" y="{cy}" width="{cw}" height="26" rx="8" fill="{TEAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c2_x+cw/2}" y="{cy+18}" font-size="10" font-weight="700" fill="{PETROL}" text-anchor="middle">COLLECTOR B: EXPLORATORY POLICY</text>')

    # Grid 2: Exploratory (Broad coverage)
    g2_x = c2_x + 42
    g2_y = cy + 42

    svg.append(f'<rect x="{g2_x}" y="{g2_y}" width="{gw}" height="{gh}" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    for i in range(1, 6):
        svg.append(f'<line x1="{g2_x + i*gw/6}" y1="{g2_y}" x2="{g2_x + i*gw/6}" y2="{g2_y+gh}" stroke="{BORDER}" stroke-width="0.5" stroke-dasharray="2,2"/>')
    for j in range(1, 5):
        svg.append(f'<line x1="{g2_x}" y1="{g2_y + j*gh/5}" x2="{g2_x+gw}" y2="{g2_y + j*gh/5}" stroke="{BORDER}" stroke-width="0.5" stroke-dasharray="2,2"/>')

    # Broad multi-cell coverage blobs
    svg.append(f'<rect x="{g2_x+12}" y="{g2_y+10}" width="{gw-24}" height="{gh-20}" rx="6" fill="{TEAL}" fill-opacity="0.25" stroke="{TEAL}" stroke-width="1"/>')
    svg.append(f'<rect x="{g2_x+35}" y="{g2_y+25}" width="{gw-70}" height="{gh-50}" rx="4" fill="{TEAL}" fill-opacity="0.45"/>')
    svg.append(f'<text x="{g2_x+gw/2}" y="{g2_y+gh/2+4}" font-size="7.5" font-weight="700" fill="{NAVY}" text-anchor="middle">510 Visited Cells (~1,961 pts/cell)</text>')

    # Axis Labels & Ticks for Grid 2
    svg.append(f'<text x="{g2_x}" y="{g2_y+gh+12}" font-size="6.5" fill="{MUTED}" text-anchor="middle">-15°</text>')
    svg.append(f'<text x="{g2_x+gw/2}" y="{g2_y+gh+12}" font-size="6.5" fill="{MUTED}" text-anchor="middle">0°</text>')
    svg.append(f'<text x="{g2_x+gw}" y="{g2_y+gh+12}" font-size="6.5" fill="{MUTED}" text-anchor="middle">+15°</text>')
    svg.append(f'<text x="{g2_x+gw/2}" y="{g2_y+gh+24}" font-size="7.5" font-weight="600" fill="{SLATE}" text-anchor="middle">Contact Angle θ (deg)</text>')

    svg.append(f'<text x="{g2_x-6}" y="{g2_y+6}" font-size="6.5" fill="{MUTED}" text-anchor="end">20N</text>')
    svg.append(f'<text x="{g2_x-6}" y="{g2_y+gh/2+3}" font-size="6.5" fill="{MUTED}" text-anchor="end">10N</text>')
    svg.append(f'<text x="{g2_x-6}" y="{g2_y+gh}" font-size="6.5" fill="{MUTED}" text-anchor="end">0N</text>')
    svg.append(f'<text x="{g2_x-22}" y="{g2_y+gh/2}" font-size="7.5" font-weight="600" fill="{SLATE}" transform="rotate(-90 {g2_x-22} {g2_y+gh/2})" text-anchor="middle">Normal Force F_N (N)</text>')

    # Metrics Card 2
    my2 = g2_y + gh + 36
    svg.append(f'<rect x="{c2_x+12}" y="{my2}" width="{cw-24}" height="195" rx="6" fill="{TEAL}" fill-opacity="0.04" stroke="{TEAL}" stroke-width="0.8"/>')
    
    m2_items = [
        ("Total Recorded Transitions:", "1,000,000 samples @ 100 Hz"),
        ("Scenario Grid Coverage:", "510 / 600 cells (85.0%)"),
        ("Unmonitored Edge Cells:", "90 / 600 cells (15.0%)"),
        ("Held-Out Validation Loss:", "L_val = 0.012 rad (Realistic)"),
        ("Compounding Drift Risk:", "BOUNDED by recovery data"),
        ("Ingestion Policy Decision:", "ADMIT / PROCEED TO TRAINING")
    ]
    for idx, (lbl, val) in enumerate(m2_items):
        svg.append(f'<text x="{c2_x+20}" y="{my2+18+idx*29}" font-size="8" font-weight="700" fill="{INK}">{lbl}</text>')
        col_val = TEAL if "85.0" in val or "BOUNDED" in val or "ADMIT" in val else SLATE
        svg.append(f'<text x="{c2_x+20}" y="{my2+30+idx*29}" font-size="7.5" font-weight="600" fill="{col_val}">{val}</text>')


    # Col 3: The Scenario Ledger Relational Join
    c3_x = 680
    c3_w = 295
    svg.append(f'<rect x="{c3_x}" y="{cy}" width="{c3_w}" height="{ch}" rx="8" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{c3_x}" y="{cy}" width="{c3_w}" height="26" rx="8" fill="{NAVY}" fill-opacity="0.1"/>')
    svg.append(f'<text x="{c3_x+c3_w/2}" y="{cy+18}" font-size="10" font-weight="700" fill="{NAVY}" text-anchor="middle">SCENARIO LEDGER RELATIONAL JOIN</text>')

    # Flowchart / Table Schema Join
    jy = cy + 38
    svg.append(f'<text x="{c3_x+14}" y="{jy+12}" font-size="8.5" font-weight="700" fill="{NAVY}">External Specification S_req:</text>')
    
    # Box: S_req
    svg.append(f'<rect x="{c3_x+14}" y="{jy+18}" width="{c3_w-28}" height="42" rx="4" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{c3_x+22}" y="{jy+34}" font-size="7.5" font-weight="700" fill="{SLATE}">Operational Envelope Grid: 600 Bins</text>')
    svg.append(f'<text x="{c3_x+22}" y="{jy+48}" font-size="7" fill="{MUTED}">θ ∈ [-15°, +15°], F_N ∈ [0, 20 N], μ ∈ [0.1, 0.4]</text>')

    # Relational Join Operator (Circle)
    svg.append(f'<circle cx="{c3_x+c3_w/2}" cy="{jy+76}" r="11" fill="{BLUE}" stroke="{BG_WHITE}" stroke-width="1.5"/>')
    svg.append(f'<text x="{c3_x+c3_w/2}" y="{jy+80}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">⋈</text>')

    # Box: Empirical Log d^π_col
    svg.append(f'<rect x="{c3_x+14}" y="{jy+94}" width="{c3_w-28}" height="42" rx="4" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{c3_x+22}" y="{jy+110}" font-size="7.5" font-weight="700" fill="{SLATE}">Empirical Dataset Visitation: N_cell</text>')
    svg.append(f'<text x="{c3_x+22}" y="{jy+124}" font-size="7" fill="{MUTED}">Count occurrences per discretized scenario cell</text>')

    # Down arrow
    svg.append(f'<line x1="{c3_x+c3_w/2}" y1="{jy+138}" x2="{c3_x+c3_w/2}" y2="{jy+156}" stroke="{NAVY}" stroke-width="1.3" marker-end="url(#arr-navy)"/>')

    # Output Decision Matrix
    dy = jy + 162
    svg.append(f'<rect x="{c3_x+12}" y="{dy}" width="{c3_w-24}" height="225" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{c3_x+20}" y="{dy+18}" font-size="8.5" font-weight="700" fill="{INK}">Support Classification Rules:</text>')

    decisions = [
        ("N_cell ≥ 500:", "Supported Nominal Regime", "Policy trained under dense supervision.", "Admit directly into training cluster.", TEAL),
        ("0 < N_cell < 50:", "High-Variance Regime", "Insufficient sample count; wide epistemic variance.", "Condition: require targeted recollection.", AMBER),
        ("N_cell = 0:", "Zero-Support Blind Spot", "Zero empirical constraints from training data.", "Refuse: trigger autonomous safety gate.", CORAL)
    ]
    for idx, (cond, cat, exp1, exp2, dcol) in enumerate(decisions):
        d_pos = dy + 36 + idx * 60
        svg.append(f'<circle cx="{c3_x+24}" cy="{d_pos-3}" r="3.5" fill="{dcol}"/>')
        svg.append(f'<text x="{c3_x+34}" y="{d_pos}" font-size="8" font-weight="700" fill="{dcol}">{cond} <tspan font-weight="600" fill="{INK}">{cat}</tspan></text>')
        svg.append(f'<text x="{c3_x+34}" y="{d_pos+14}" font-size="7" fill="{SLATE}">{exp1}</text>')
        svg.append(f'<text x="{c3_x+34}" y="{d_pos+25}" font-size="7" font-weight="600" fill="{dcol}">{exp2}</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/05-data/figures/fig05_collector_coverage_ledger.svg", "\n".join(svg))


def run_all():
    print("Generating Chapter 5 Data figures...")
    gen_fig05_compounding_error_flywheel()
    gen_fig05_collector_coverage_ledger()
    print("Chapter 5 figures generated successfully.")

if __name__ == "__main__":
    run_all()


