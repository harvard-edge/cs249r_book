"""
book/tools/figures/ch09.py
Figures for Chapter 9: Memory & Temporal Belief Dynamics.
Pure vector SVG, unclipped typography, Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import os
import subprocess
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig09_uncertainty_growth():
    W = 920
    H = 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">TEMPORAL BELIEF UNCERTAINTY GROWTH &amp; INVALIDATION HORIZON</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Bounding Covariance Expansion E(t) = E(0) + 3σ_v t + ½ a_dist t² Against Mechanical Task Clearance E_max</text>')

    # -------------------------------------------------------------
    # LEFT PANEL: Dynamic Error Growth Graph
    # -------------------------------------------------------------
    gx = 40
    gy = 68
    gw = 530
    gh = 310

    svg.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2"/>')

    # Plot area inside left panel
    px = gx + 55
    py = gy + 30
    pw = 450
    ph = 230

    # Axes ranges: t from 0 to 140 ms, E from 0.0 to 1.6 mm
    def map_x(t_ms):
        return px + (t_ms / 140.0) * pw

    def map_y(e_mm):
        return (py + ph) - (e_mm / 1.6) * ph

    # Grid lines and labels
    # Horizontal grid (E values: 0.0, 0.4, 0.8, 1.0, 1.2, 1.6 mm)
    for e_val in [0.0, 0.4, 0.8, 1.0, 1.2, 1.6]:
        y_pos = map_y(e_val)
        is_emax = (e_val == 1.0)
        stroke_col = CORAL if is_emax else BORDER
        stroke_w = "1.5" if is_emax else "0.8"
        dash = ' stroke-dasharray="4,3"' if is_emax else ' stroke-dasharray="2,2"'
        svg.append(f'<line x1="{px}" y1="{y_pos}" x2="{px+pw}" y2="{y_pos}" stroke="{stroke_col}" stroke-width="{stroke_w}"{dash}/>')
        svg.append(f'<text x="{px-8}" y="{y_pos+3.5}" font-size="8.5" font-weight="{"700" if is_emax else "500"}" fill="{"#DC2626" if is_emax else SLATE}" text-anchor="end">{e_val:.1f}</text>')

    # Vertical grid (t values: 0, 20, 40, 60, 80, 100, 120, 140 ms)
    for t_val in [0, 20, 40, 60, 80, 100, 120, 140]:
        x_pos = map_x(t_val)
        svg.append(f'<line x1="{x_pos}" y1="{py}" x2="{x_pos}" y2="{py+ph}" stroke="{BORDER}" stroke-width="0.8" stroke-dasharray="2,2"/>')
        svg.append(f'<text x="{x_pos}" y="{py+ph+14}" font-size="8.5" fill="{SLATE}" text-anchor="middle">{t_val}</text>')

    # Axes Lines
    svg.append(f'<line x1="{px}" y1="{py+ph}" x2="{px+pw+15}" y2="{py+ph}" stroke="{INK}" stroke-width="1.3" marker-end="url(#arr-navy)"/>')
    svg.append(f'<line x1="{px}" y1="{py+ph}" x2="{px}" y2="{py-10}" stroke="{INK}" stroke-width="1.3" marker-end="url(#arr-navy)"/>')
    svg.append(f'<text x="{px+pw+20}" y="{py+ph+3}" font-size="9" font-weight="700" fill="{INK}">Δt (ms)</text>')
    svg.append(f'<text x="{px}" y="{py-16}" font-size="9" font-weight="700" fill="{INK}" text-anchor="middle">Error Bound E(t) [mm]</text>')

    # Color zones / shading under curve
    # Points for E(t) = 0.1 + 12.0*(t/1000) + 25.0*(t/1000)^2
    # At t=0: E=0.10
    # At t=30: E = 0.10 + 0.36 + 0.0225 = 0.4825 mm (Operational threshold)
    # At t=66: E = 0.10 + 0.792 + 0.1089 = 1.0009 mm (Expiry threshold)
    # At t=100: E = 0.10 + 1.20 + 0.25 = 1.55 mm

    # Shaded Area under nominal curve:
    # 1. Valid Zone (0 to 30 ms)
    pts_valid = [f"{map_x(0)},{map_y(0)}"]
    for t_step in range(0, 31, 2):
        t_sec = t_step / 1000.0
        e_val = 0.10 + 12.0 * t_sec + 25.0 * (t_sec ** 2)
        pts_valid.append(f"{map_x(t_step)},{map_y(e_val)}")
    pts_valid.append(f"{map_x(30)},{map_y(0)}")
    svg.append(f'<polygon points="{" ".join(pts_valid)}" fill="{TEAL}" fill-opacity="0.18"/>')

    # 2. Degraded Zone (30 to 66 ms)
    pts_deg = [f"{map_x(30)},{map_y(0)}"]
    for t_step in range(30, 67, 2):
        t_sec = t_step / 1000.0
        e_val = 0.10 + 12.0 * t_sec + 25.0 * (t_sec ** 2)
        pts_deg.append(f"{map_x(t_step)},{map_y(e_val)}")
    pts_deg.append(f"{map_x(66)},{map_y(0)}")
    svg.append(f'<polygon points="{" ".join(pts_deg)}" fill="{AMBER}" fill-opacity="0.18"/>')

    # 3. Expired Zone (66 to 102 ms)
    pts_exp = [f"{map_x(66)},{map_y(0)}"]
    for t_step in range(66, 103, 2):
        t_sec = t_step / 1000.0
        e_val = 0.10 + 12.0 * t_sec + 25.0 * (t_sec ** 2)
        pts_exp.append(f"{map_x(t_step)},{map_y(e_val)}")
    pts_exp.append(f"{map_x(102)},{map_y(0)}")
    svg.append(f'<polygon points="{" ".join(pts_exp)}" fill="{CORAL}" fill-opacity="0.18"/>')

    # Curve 1: Nominal Error Growth E(t)
    curve_pts = []
    for t_step in range(0, 105, 2):
        t_sec = t_step / 1000.0
        e_val = 0.10 + 12.0 * t_sec + 25.0 * (t_sec ** 2)
        if e_val <= 1.6:
            curve_pts.append(f"{map_x(t_step):.1f},{map_y(e_val):.1f}")
    svg.append(f'<polyline points="{" ".join(curve_pts)}" fill="none" stroke="{NAVY}" stroke-width="2.5"/>')

    # Curve 2: High Dynamics Curve (v=1.5 m/s, payload=5kg, E_exp at 22 ms)
    fast_pts = []
    for t_step in range(0, 30, 2):
        t_sec = t_step / 1000.0
        e_val = 0.10 + 38.0 * t_sec + 120.0 * (t_sec ** 2)
        if e_val <= 1.6:
            fast_pts.append(f"{map_x(t_step):.1f},{map_y(e_val):.1f}")
    svg.append(f'<polyline points="{" ".join(fast_pts)}" fill="none" stroke="{PURPLE}" stroke-width="1.6" stroke-dasharray="4,2"/>')
    svg.append(f'<text x="{map_x(18)+4}" y="{map_y(1.35)}" font-size="7.5" font-weight="700" fill="{PURPLE}">High Dynamics (1.5 m/s)</text>')

    # Curve 3: Low Dynamics / Static Clamped (v=0.2 m/s, E_exp at 140 ms)
    slow_pts = []
    for t_step in range(0, 141, 4):
        t_sec = t_step / 1000.0
        e_val = 0.10 + 6.0 * t_sec + 3.0 * (t_sec ** 2)
        if e_val <= 1.6:
            slow_pts.append(f"{map_x(t_step):.1f},{map_y(e_val):.1f}")
    svg.append(f'<polyline points="{" ".join(slow_pts)}" fill="none" stroke="{PETROL}" stroke-width="1.6" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{map_x(105)}" y="{map_y(0.70)}" font-size="7.5" font-weight="700" fill="{PETROL}">Slow Motion (0.2 m/s)</text>')

    # Key Marker Points on Nominal Curve:
    # 1. t0: Last constraining measurement
    x0, y0 = map_x(0), map_y(0.10)
    svg.append(f'<circle cx="{x0}" cy="{y0}" r="4.5" fill="{TEAL}" stroke="{INK}" stroke-width="1.2"/>')
    svg.append(f'<text x="{x0+8}" y="{y0-8}" font-size="8" font-weight="700" fill="{TEAL}">t₀: Evidence Latch (E₀ = 0.10 mm)</text>')

    # 2. Expiry Horizon crossing: t=66 ms, E=1.0 mm
    x_exp, y_exp = map_x(66), map_y(1.0)
    svg.append(f'<line x1="{x_exp}" y1="{py}" x2="{x_exp}" y2="{py+ph}" stroke="{CORAL}" stroke-width="1.8" stroke-dasharray="4,2"/>')
    svg.append(f'<circle cx="{x_exp}" cy="{y_exp}" r="5" fill="{CORAL}" stroke="{INK}" stroke-width="1.2"/>')
    
    # Expiry Callout Tag
    svg.append(f'<rect x="{x_exp-55}" y="{y_exp-34}" width="110" height="22" rx="4" fill="{CORAL}"/>')
    svg.append(f'<text x="{x_exp}" y="{y_exp-20}" font-size="8.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">EXPIRY: t_exp = 66 ms</text>')

    # Task Clearance E_max Label
    svg.append(f'<rect x="{px+160}" y="{map_y(1.0)-18}" width="175" height="15" rx="3" fill="{CORAL}" fill-opacity="0.12" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{px+165}" y="{map_y(1.0)-7}" font-size="8" font-weight="700" fill="{CORAL}">E_max = 1.0 mm (Task Clearance)</text>')

    # Reacquisition Path at t=45 ms
    x_reacq = map_x(45)
    e_reacq = 0.10 + 12.0 * 0.045 + 25.0 * (0.045 ** 2) # ~0.69 mm
    y_reacq = map_y(e_reacq)
    svg.append(f'<circle cx="{x_reacq}" cy="{y_reacq}" r="4" fill="{BLUE}" stroke="{INK}" stroke-width="1"/>')
    svg.append(f'<line x1="{x_reacq}" y1="{y_reacq}" x2="{x_reacq}" y2="{map_y(0.10)}" stroke="{BLUE}" stroke-width="1.5" stroke-dasharray="2,2" marker-end="url(#arr-blue)"/>')
    svg.append(f'<text x="{x_reacq+6}" y="{y_reacq+14}" font-size="7.5" font-weight="700" fill="{BLUE}">Sensor Reacquisition</text>')
    svg.append(f'<text x="{x_reacq+6}" y="{y_reacq+23}" font-size="7" fill="{MUTED}">Resets clock to t₀\'</text>')

    # -------------------------------------------------------------
    # BOTTOM LIFECYCLE STATE BAR (Inside Left Panel)
    # -------------------------------------------------------------
    l_bar_y = gy + gh + 14
    l_bar_w = gw
    svg.append(f'<rect x="{gx}" y="{l_bar_y}" width="{l_bar_w}" height="42" rx="6" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    
    # Valid zone segment
    w_val = (30 / 140.0) * l_bar_w
    svg.append(f'<rect x="{gx}" y="{l_bar_y}" width="{w_val}" height="42" rx="6 0 0 6" fill="{TEAL}" fill-opacity="0.2" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{gx + w_val/2}" y="{l_bar_y+16}" font-size="8.5" font-weight="700" fill="{TEAL}" text-anchor="middle">VALID [0–30 ms]</text>')
    svg.append(f'<text x="{gx + w_val/2}" y="{l_bar_y+30}" font-size="7.5" fill="{INK}" text-anchor="middle">Full Authority</text>')

    # Degraded zone segment
    w_deg = ((66 - 30) / 140.0) * l_bar_w
    svg.append(f'<rect x="{gx+w_val}" y="{l_bar_y}" width="{w_deg}" height="42" fill="{AMBER}" fill-opacity="0.2" stroke="{AMBER}" stroke-width="1.2"/>')
    svg.append(f'<text x="{gx + w_val + w_deg/2}" y="{l_bar_y+16}" font-size="8.5" font-weight="700" fill="{AMBER}" text-anchor="middle">DEGRADED [30–66 ms]</text>')
    svg.append(f'<text x="{gx + w_val + w_deg/2}" y="{l_bar_y+30}" font-size="7.5" fill="{INK}" text-anchor="middle">Slowdown / Widen Buffer</text>')

    # Expired zone segment
    w_exp = l_bar_w - (w_val + w_deg)
    svg.append(f'<rect x="{gx+w_val+w_deg}" y="{l_bar_y}" width="{w_exp}" height="42" rx="0 6 6 0" fill="{CORAL}" fill-opacity="0.2" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{gx + w_val + w_deg + w_exp/2}" y="{l_bar_y+16}" font-size="8.5" font-weight="700" fill="{CORAL}" text-anchor="middle">EXPIRED [&gt;66 ms]</text>')
    svg.append(f'<text x="{gx + w_val + w_deg + w_exp/2}" y="{l_bar_y+30}" font-size="7.5" fill="{INK}" text-anchor="middle">Action Perms Revoked</text>')

    # -------------------------------------------------------------
    # RIGHT PANEL 1: Governing Decay Formulation & Empirical Tails
    # -------------------------------------------------------------
    rx = 590
    ry = 68
    rw = 290
    rh1 = 175

    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh1}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="26" rx="8 8 0 0" fill="{NAVY}"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+17}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">GOVERNING DECAY LAW</text>')

    # Formula box
    svg.append(f'<rect x="{rx+12}" y="{ry+34}" width="{rw-24}" height="32" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+48}" font-size="8.5" font-weight="700" fill="{NAVY}" text-anchor="middle">E(t) = E(0) + 3σ_v t + ½ a_dist t²</text>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+60}" font-size="7.5" fill="{MUTED}" text-anchor="middle">t_exp = inf {{ t ≥ 0 : E(t) ≥ E_max }}</text>')

    law_points = [
        ("Initial Sensor Bound E(0):", "0.10 mm at P99.9 metrology latch"),
        ("Linear Velocity Drift 3σ_v:", "12.0 mm/s (unobserved creep)"),
        ("Disturbance Accel a_dist:", "50.0 mm/s² (friction/backlash)"),
        ("Empirical Characterization:", "Fitted from staged P99 dropouts")
    ]
    cur_y = ry + 78
    for title_txt, val_txt in law_points:
        svg.append(f'<text x="{rx+14}" y="{cur_y}" font-size="8" font-weight="700" fill="{INK}">• {title_txt}</text>')
        svg.append(f'<text x="{rx+22}" y="{cur_y+11}" font-size="7.5" fill="{SLATE}">{val_txt}</text>')
        cur_y += 22

    # -------------------------------------------------------------
    # RIGHT PANEL 2: The Silent Frozen Transform Hazard
    # -------------------------------------------------------------
    ry2 = ry + rh1 + 12
    rh2 = 180

    svg.append(f'<rect x="{rx}" y="{ry2}" width="{rw}" height="{rh2}" rx="8" fill="{CRIMSON}" fill-opacity="0.04" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{rx}" y="{ry2}" width="{rw}" height="26" rx="8 8 0 0" fill="{CRIMSON}"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry2+17}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">THE FROZEN TRANSFORM TRAP</text>')

    trap_items = [
        ("The Software Illusion:", "Middleware republishes cached pose with fresh OS clock (t_pub = t_now)."),
        ("Deceptive Freshness:", "Downstream age check (t_now - t_pub &lt; 2 ms) passes with zero warnings."),
        ("Physical Reality:", "True evidence age Δt &gt; 66 ms; physical error E(t) &gt;&gt; E_max (crash hazard)."),
        ("Architectural Invariant:", "Age MUST bind strictly to acquisition epoch t₀, never to publish time.")
    ]
    cur_ty = ry2 + 38
    for hdr, desc in trap_items:
        svg.append(f'<text x="{rx+12}" y="{cur_ty}" font-size="8" font-weight="700" fill="{CORAL}">✕ {hdr}</text>')
        svg.append(f'<text x="{rx+20}" y="{cur_ty+11}" font-size="7.5" fill="{SLATE}">{desc}</text>')
        cur_ty += 23

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/09-memory/figures/fig09_uncertainty_growth.svg", "\n".join(svg))


def gen_fig09_frame_staleness_error():
    W = 920
    H = 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">COORDINATE TRANSFORM STALENESS &amp; LEVER-ARM ERROR PROPAGATION</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Rigid-Body Kinematic Divergence: e_tool ≈ v · Δt + ℓ · ω · Δt Under Stale Coordinate Frames</text>')

    # -------------------------------------------------------------
    # LEFT PANEL: Geometric Kinematics Diagram
    # -------------------------------------------------------------
    lx = 40
    ly = 68
    lw = 460
    lh = 368

    svg.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="26" rx="8 8 0 0" fill="{NAVY}"/>')
    svg.append(f'<text x="{lx+lw/2}" y="{ly+17}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">PHYSICAL ARM DYNAMICS &amp; LEVER ARM DIVERGENCE</text>')

    # Coordinate Origin / Base
    bx0 = lx + 80
    by0 = ly + 310

    # Base translation motion: v = 0.80 m/s, delta t = 40 ms -> e_trans = 3.20 cm
    bx_stale = bx0
    by_stale = by0

    bx_true = bx0 + 60
    by_true = by0

    # Arm Link geometry: length = 180 px (representing 0.60 m)
    import math
    stale_rad = math.radians(52)
    true_rad = math.radians(52 - 14) # rotated forwards

    link_len = 175
    tx_stale = bx_stale + link_len * math.cos(stale_rad)
    ty_stale = by_stale - link_len * math.sin(stale_rad)

    tx_true = bx_true + link_len * math.cos(true_rad)
    ty_true = by_true - link_len * math.sin(true_rad)

    # Base ground plane
    svg.append(f'<line x1="{lx+30}" y1="{by0+20}" x2="{lx+lw-30}" y2="{by0+20}" stroke="{BORDER_DARK}" stroke-width="1.5"/>')
    for g_tick in range(lx+40, lx+lw-30, 20):
        svg.append(f'<line x1="{g_tick}" y1="{by0+20}" x2="{g_tick-8}" y2="{by0+28}" stroke="{BORDER}" stroke-width="1"/>')

    # Base Translation Vector
    svg.append(f'<line x1="{bx_stale}" y1="{by0+10}" x2="{bx_true}" y2="{by0+10}" stroke="{BLUE}" stroke-width="2.5" marker-end="url(#arr-blue)"/>')
    svg.append(f'<text x="{(bx_stale+bx_true)/2}" y="{by0-2}" font-size="8" font-weight="700" fill="{BLUE}" text-anchor="middle">e_trans = v · Δt = 3.20 cm</text>')

    # Stale Arm (Ghosted / Dashed)
    svg.append(f'<line x1="{bx_stale}" y1="{by_stale}" x2="{tx_stale}" y2="{ty_stale}" stroke="{MUTED}" stroke-width="3.5" stroke-dasharray="5,3" stroke-linecap="round"/>')
    svg.append(f'<circle cx="{bx_stale}" cy="{by_stale}" r="7" fill="{MUTED}" fill-opacity="0.3" stroke="{MUTED}" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{tx_stale}" cy="{ty_stale}" r="5" fill="{MUTED}" stroke="{MUTED}" stroke-width="1.5"/>')
    svg.append(f'<text x="{tx_stale-10}" y="{ty_stale-10}" font-size="8.5" font-weight="700" fill="{MUTED}">Stale Pose (t₀)</text>')

    # True Arm (Solid)
    svg.append(f'<line x1="{bx_true}" y1="{by_true}" x2="{tx_true}" y2="{ty_true}" stroke="{NAVY}" stroke-width="4.5" stroke-linecap="round"/>')
    svg.append(f'<circle cx="{bx_true}" cy="{by_true}" r="8" fill="{NAVY}" stroke="{INK}" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{tx_true}" cy="{ty_true}" r="6" fill="{PETROL}" stroke="{INK}" stroke-width="1.5"/>')
    svg.append(f'<text x="{tx_true+12}" y="{ty_true-8}" font-size="8.5" font-weight="700" fill="{NAVY}">True Pose (t_now)</text>')

    # Lever Arm Annotation
    mid_arm_x = (bx_true + tx_true) / 2
    mid_arm_y = (by_true + ty_true) / 2
    svg.append(f'<text x="{mid_arm_x-18}" y="{mid_arm_y-14}" font-size="8" font-weight="700" fill="{NAVY}">Link ℓ = 0.60 m</text>')
    svg.append(f'<text x="{bx_true+22}" y="{by_true-24}" font-size="7.5" font-weight="700" fill="{BRONZE}">ω = 1.50 rad/s</text>')

    # Angular Sweep Arc
    svg.append(f'<path d="M {bx_true+45} {by_true-30} A 50 50 0 0 1 {bx_true+52} {by_true-15}" fill="none" stroke="{BRONZE}" stroke-width="1.5" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<text x="{bx_true+62}" y="{by_true-20}" font-size="7.5" font-weight="700" fill="{BRONZE}">Δθ = 0.06 rad</text>')

    # Error vectors at Tool Tip
    # 1. Translation offset from Stale Tip
    tx_trans = tx_stale + 60
    ty_trans = ty_stale
    svg.append(f'<line x1="{tx_stale}" y1="{ty_stale}" x2="{tx_trans}" y2="{ty_trans}" stroke="{BLUE}" stroke-width="1.5" stroke-dasharray="2,2"/>')

    # 2. Rotational arc offset from translation point to True Tip
    svg.append(f'<line x1="{tx_trans}" y1="{ty_trans}" x2="{tx_true}" y2="{ty_true}" stroke="{BRONZE}" stroke-width="1.5" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{tx_trans+6}" y="{ty_trans+12}" font-size="7.5" font-weight="700" fill="{BRONZE}">e_rot ≈ ℓΔθ = 3.60 cm</text>')

    # 3. Total Error Vector (Stale to True)
    svg.append(f'<line x1="{tx_stale}" y1="{ty_stale}" x2="{tx_true}" y2="{ty_true}" stroke="{CORAL}" stroke-width="2.5" marker-end="url(#arr-coral)"/>')
    svg.append(f'<rect x="{(tx_stale+tx_true)/2-65}" y="{(ty_stale+ty_true)/2-32}" width="130" height="20" rx="4" fill="{CORAL}"/>')
    svg.append(f'<text x="{(tx_stale+tx_true)/2}" y="{(ty_stale+ty_true)/2-18}" font-size="8.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">e_tool = 6.80 cm (68 mm)</text>')

    # Workpiece / Fixture Collision Boundary
    fix_x = tx_stale + 30
    svg.append(f'<rect x="{fix_x}" y="{ty_true-30}" width="40" height="80" rx="2" fill="{CRIMSON}" fill-opacity="0.15" stroke="{CRIMSON}" stroke-width="1.5"/>')
    svg.append(f'<text x="{fix_x+20}" y="{ty_true-35}" font-size="7.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">Fixture Plane</text>')
    svg.append(f'<text x="{fix_x+20}" y="{ty_true+60}" font-size="7" fill="{MUTED}" text-anchor="middle">Tol: ±0.5 mm</text>')

    # Bottom Callout inside left panel
    svg.append(f'<rect x="{lx+10}" y="{ly+lh-48}" width="{lw-20}" height="38" rx="5" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{lx+lw/2}" y="{ly+lh-30}" font-size="8" font-weight="700" fill="{CORAL}" text-anchor="middle">CRITICAL COLLISION: 68.0 mm Error vs 0.5 mm Mechanical Clearance (136× Breach)</text>')
    svg.append(f'<text x="{lx+lw/2}" y="{ly+lh-18}" font-size="7.5" fill="{SLATE}" text-anchor="middle">Uncompensated 40 ms lag drives end-effector into solid fixture during trajectory execution.</text>')

    # -------------------------------------------------------------
    # RIGHT PANEL: Mathematical Breakdown & Failure Comparison
    # -------------------------------------------------------------
    rx = 520
    ry = 68
    rw = 360
    rh1 = 145

    # Card 1: Kinematic Arithmetic Derivation
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh1}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="26" rx="8 8 0 0" fill="{NAVY}"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+17}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">KINEMATIC LEVER-ARM ARITHMETIC</text>')

    math_lines = [
        ("Translational Drift:", "e_trans = v · Δt = 0.80 m/s × 0.040 s = 3.20 cm"),
        ("Rotational Arc Deflection:", "e_rot ≈ ℓ · (ω · Δt) = 0.60 m × (1.50 × 0.040) = 3.60 cm"),
        ("Total Tool-Point Offset:", "e_tool ≈ e_trans + e_rot = 3.20 + 3.60 = 6.80 cm (68.0 mm)"),
        ("Task Safety Clearance:", "ε_tol = 0.50 mm (Exceeded by 136× within 40 ms!)")
    ]
    cur_my = ry + 42
    for label_m, val_m in math_lines:
        svg.append(f'<text x="{rx+12}" y="{cur_my}" font-size="8" font-weight="700" fill="{INK}">• {label_m}</text>')
        svg.append(f'<text x="{rx+20}" y="{cur_my+11}" font-size="7.5" fill="{SLATE}">{val_m}</text>')
        cur_my += 24

    # Card 2: Loud vs Silent Failure Taxonomy
    ry2 = ry + rh1 + 12
    rh2 = 211

    svg.append(f'<rect x="{rx}" y="{ry2}" width="{rw}" height="{rh2}" rx="8" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{rx}" y="{ry2}" width="{rw}" height="26" rx="8 8 0 0" fill="{PURPLE}"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry2+17}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">LOUD VS SILENT TRANSFORM FAILURES</text>')

    # Split comparison blocks
    # Block A: Loud Failure (Safe)
    by_a = ry2 + 34
    svg.append(f'<rect x="{rx+10}" y="{by_a}" width="{rw-20}" height="66" rx="5" fill="{TEAL}" fill-opacity="0.06" stroke="{TEAL}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+18}" y="{by_a+16}" font-size="8.5" font-weight="700" fill="{TEAL}">✓ LOUD FAILURE: MISSING / DROPPED TRANSFORM</text>')
    svg.append(f'<text x="{rx+18}" y="{by_a+30}" font-size="7.5" fill="{SLATE}">• Sensor link drops; frame tree records null edge.</text>')
    svg.append(f'<text x="{rx+18}" y="{by_a+42}" font-size="7.5" fill="{SLATE}">• Matrix traversal raises FRAME_NOT_FOUND exception.</text>')
    svg.append(f'<text x="{rx+18}" y="{by_a+54}" font-size="7.5" font-weight="600" fill="{INK}">• Result: Deterministic emergency stop / safe position hold.</text>')

    # Block B: Silent Failure (Destructive)
    by_b = by_a + 72
    svg.append(f'<rect x="{rx+10}" y="{by_b}" width="{rw-20}" height="70" rx="5" fill="{CORAL}" fill-opacity="0.06" stroke="{CORAL}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+18}" y="{by_b+16}" font-size="8.5" font-weight="700" fill="{CORAL}">✕ SILENT FAILURE: FROZEN / REPUBLISHED MATRIX</text>')
    svg.append(f'<text x="{rx+18}" y="{by_b+30}" font-size="7.5" fill="{SLATE}">• Middleware republishes cached pose with fresh OS timestamp.</text>')
    svg.append(f'<text x="{rx+18}" y="{by_b+42}" font-size="7.5" fill="{SLATE}">• Downstream linear algebra evaluates with zero exceptions.</text>')
    svg.append(f'<text x="{rx+18}" y="{by_b+54}" font-size="7.5" font-weight="600" fill="{CORAL}">• Result: Trajectory planner drives tool into solid fixture.</text>')

    # Bottom Invariant Rule
    svg.append(f'<rect x="{rx+10}" y="{by_b+76}" width="{rw-20}" height="22" rx="3" fill="{NAVY}"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{by_b+90}" font-size="7.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">RULE: Transforms MUST carry immutable evidence epoch t_e.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/09-memory/figures/fig09_frame_staleness_error.svg", "\n".join(svg))


def run_all():
    print("Generating Chapter 9 Figures...")
    gen_fig09_uncertainty_growth()
    gen_fig09_frame_staleness_error()
    print("✓ Chapter 9 Figures generated successfully!")

if __name__ == "__main__":
    run_all()
