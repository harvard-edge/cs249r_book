"""
book/tools/figures/ch02.py
Figures for Chapter 2: Body (The Five Physical Columns).
Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import os
import math
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig02_latency_waterfall():
    """
    Fig 2.X: End-to-End Latency Waterfall & Measurement Freshness Pipeline.
    Photons striking CMOS -> DMA -> Policy Inference -> Action Chunk Spline ->
    Current Loop -> Motor Torque Acceleration -> Physical Displacement vs Clearance Collapse.
    """
    W = 960
    H = 620
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    
    # Header
    svg.append(f'<text x="{W/2}" y="28" class="title">END-TO-END SENSE-TO-ACTUATION LATENCY WATERFALL &amp; FRESHNESS BUDGET</text>')
    svg.append(f'<text x="{W/2}" y="45" class="subtitle">Cumulative Wall-Clock Decomposition: From Photon Transduction to Torque Generation vs. the Freshness Deadline</text>')

    # -------------------------------------------------------------
    # TOP: 8-STAGE SYSTEM ARCHITECTURE & TIMING CARDS
    # -------------------------------------------------------------
    stages = [
        ("1. TRANSDUCTION", "CMOS Exposure", "Photons striking well;\nmid-exposure timestamp", NAVY, "11.2", "16.6"),
        ("2. READOUT/DMA", "MIPI &amp; PCIe DMA", "Zero-copy ring buffer;\ncrossbar DMA ingest", BLUE, "1.4", "3.2"),
        ("3. TOKENIZER", "Vision Backbone", "ViT patch embedding;\nspatial token tensor", BLUE, "7.0", "10.5"),
        ("4. POLICY VLA", "Action Chunking", "Diffusion / ACT model;\n16-step trajectory spline", BRONZE, "11.5", "13.6"),
        ("5. MEMORY IPC", "Shared SRAM TCM", "Lock-free seqlock;\nMPU-to-MCU mailbox", PURPLE, "0.6", "2.1"),
        ("6. REAL-TIME QP", "1 kHz MCU Filter", "Active-set CBF QP;\nstopping envelope check", PETROL, "0.2", "0.4"),
        ("7. FIELDBUS", "EtherCAT Frame", "Cyclic sync torque;\nPWM gate driver enable", PETROL, "1.0", "1.0"),
        ("8. COIL TORQUE", "Stator L/R Rise", "Current buildup \u03c4_e;\nrotor torque J\u03b8\u0308 = \u03c4_m", CRIMSON, "2.3", "3.0")
    ]

    card_w = 106
    card_h = 142
    gap = 8
    start_x = (W - (8 * card_w + 7 * gap)) / 2
    top_y = 65

    for idx, (tag, name, desc, col, p50, p99) in enumerate(stages):
        x = start_x + idx * (card_w + gap)
        
        # Card container
        svg.append(f'<rect x="{x}" y="{top_y}" width="{card_w}" height="{card_h}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        # Header banner
        svg.append(f'<rect x="{x}" y="{top_y}" width="{card_w}" height="20" rx="6" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{top_y+14}" font-size="7.5" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{top_y+34}" font-size="9" font-weight="700" fill="{INK}" text-anchor="middle">{name}</text>')
        
        # Description lines
        lines = desc.split("\n")
        svg.append(f'<text x="{x+card_w/2}" y="{top_y+48}" font-size="7.2" fill="{SLATE}" text-anchor="middle">{lines[0]}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{top_y+59}" font-size="7.2" fill="{MUTED}" text-anchor="middle">{lines[1]}</text>')

        # Divider
        svg.append(f'<line x1="{x+6}" y1="{top_y+70}" x2="{x+card_w-6}" y2="{top_y+70}" stroke="{BORDER}" stroke-width="0.8"/>')

        # Percentile metrics
        svg.append(f'<rect x="{x+6}" y="{top_y+76}" width="{card_w-12}" height="26" rx="3" fill="{TEAL}" fill-opacity="0.08" stroke="{TEAL}" stroke-width="0.7"/>')
        svg.append(f'<text x="{x+10}" y="{top_y+87}" font-size="7" font-weight="700" fill="{MUTED}">P50 Median</text>')
        svg.append(f'<text x="{x+card_w-10}" y="{top_y+97}" font-size="9" font-weight="700" fill="{PETROL}" text-anchor="end">{p50} ms</text>')

        svg.append(f'<rect x="{x+6}" y="{top_y+106}" width="{card_w-12}" height="26" rx="3" fill="{CRIMSON}" fill-opacity="0.08" stroke="{CRIMSON}" stroke-width="0.7"/>')
        svg.append(f'<text x="{x+10}" y="{top_y+117}" font-size="7" font-weight="700" fill="{MUTED}">P99 Tail</text>')
        svg.append(f'<text x="{x+card_w-10}" y="{top_y+127}" font-size="9" font-weight="700" fill="{CRIMSON}" text-anchor="end">{p99} ms</text>')

        # Arrow connector between cards
        if idx < 7:
            ax1 = x + card_w + 1
            ax2 = ax1 + gap - 2
            ay = top_y + card_h / 2
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{col}" stroke-width="1.3" marker-end="url(#arr-navy)"/>')

    # -------------------------------------------------------------
    # MIDDLE & BOTTOM: CUMULATIVE LATENCY WATERFALL & TIME AXIS
    # -------------------------------------------------------------
    waterfall_box_y = 222
    waterfall_box_h = 380
    
    # Left container: Cumulative Timing Waterfall (width = 570)
    lw = 575
    svg.append(f'<rect x="{start_x}" y="{waterfall_box_y}" width="{lw}" height="{waterfall_box_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{start_x+16}" y="{waterfall_box_y+22}" class="section-hdr">CUMULATIVE SENSE-TO-ACTUATION DELAY WATERFALL (\u0394t)</text>')
    svg.append(f'<text x="{start_x+16}" y="{waterfall_box_y+36}" font-size="8.5" fill="{MUTED}">Direct Empirical Sum vs. Measured Tail Convolution on Physical Hardware</text>')

    wf_start_x = start_x + 160
    wf_max_ms = 80.0
    wf_plot_w = lw - 185
    wf_base_y = waterfall_box_y + 60
    row_h = 24

    # Gridlines and Ticks
    for t_mark in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
        gx = wf_start_x + (t_mark / wf_max_ms) * wf_plot_w
        svg.append(f'<line x1="{gx}" y1="{wf_base_y}" x2="{gx}" y2="{wf_base_y + 8 * row_h + 50}" stroke="{BORDER}" stroke-width="0.8" stroke-dasharray="2,2"/>')
        svg.append(f'<text x="{gx}" y="{wf_base_y + 8 * row_h + 62}" font-size="8" fill="{SLATE}" text-anchor="middle">{t_mark} ms</text>')

    # Time axis line
    svg.append(f'<line x1="{wf_start_x}" y1="{wf_base_y + 8 * row_h + 50}" x2="{wf_start_x + wf_plot_w + 10}" y2="{wf_base_y + 8 * row_h + 50}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{wf_start_x + wf_plot_w + 14}" y="{wf_base_y + 8 * row_h + 53}" font-size="8" font-weight="700" fill="{SLATE}">Time</text>')

    # Freshness Deadline Vertical Marker at 40.0 ms
    deadline_ms = 40.0
    dx = wf_start_x + (deadline_ms / wf_max_ms) * wf_plot_w
    svg.append(f'<line x1="{dx}" y1="{wf_base_y-10}" x2="{dx}" y2="{wf_base_y + 8 * row_h + 50}" stroke="{CORAL}" stroke-width="2" stroke-dasharray="4,3"/>')
    svg.append(f'<rect x="{dx-55}" y="{wf_base_y-18}" width="110" height="16" rx="3" fill="{CORAL}"/>')
    svg.append(f'<text x="{dx}" y="{wf_base_y-7}" font-size="7.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">FRESHNESS LIMIT: 40.0 ms</text>')

    # Waterfall Rows (Stage 1 to 8)
    st_durations = [11.2, 1.4, 7.0, 11.5, 0.6, 0.2, 1.0, 2.3]
    st_names = [
        "1. CMOS Exposure",
        "2. MIPI/PCIe DMA",
        "3. Vision Backbone",
        "4. Policy Chunking",
        "5. Shared SRAM IPC",
        "6. MCU Barrier QP",
        "7. Fieldbus Command",
        "8. Stator Current Rise"
    ]
    st_cols = [NAVY, BLUE, BLUE, BRONZE, PURPLE, PETROL, PETROL, CRIMSON]

    cum_time = 0.0
    for idx in range(8):
        ry = wf_base_y + idx * row_h
        dur = st_durations[idx]
        svg.append(f'<text x="{wf_start_x-8}" y="{ry+14}" font-size="8" font-weight="600" fill="{INK}" text-anchor="end">{st_names[idx]}</text>')

        # Bar
        bx = wf_start_x + (cum_time / wf_max_ms) * wf_plot_w
        bw = (dur / wf_max_ms) * wf_plot_w
        col = st_cols[idx]
        svg.append(f'<rect x="{bx}" y="{ry+2}" width="{max(bw, 2)}" height="16" rx="2" fill="{col}" fill-opacity="0.85" stroke="{col}" stroke-width="0.8"/>')
        if bw > 25:
            svg.append(f'<text x="{bx+bw/2}" y="{ry+13}" font-size="7.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">+{dur}ms</text>')
        else:
            svg.append(f'<text x="{bx+bw+3}" y="{ry+13}" font-size="7" font-weight="700" fill="{col}">+{dur}</text>')
        cum_time += dur

    # Bottom Comparison Bars: Nominal vs Tail Latency
    comp_y = wf_base_y + 8 * row_h + 10
    
    # 1. Nominal P50 Total = 35.0 ms
    p50_total = 35.0
    p50_w = (p50_total / wf_max_ms) * wf_plot_w
    svg.append(f'<rect x="{wf_start_x}" y="{comp_y}" width="{p50_w}" height="16" rx="3" fill="{TEAL}" fill-opacity="0.85" stroke="{TEAL}" stroke-width="1"/>')
    svg.append(f'<text x="{wf_start_x-8}" y="{comp_y+12}" font-size="8" font-weight="700" fill="{PETROL}" text-anchor="end">Empirical P50 (35.0 ms)</text>')
    svg.append(f'<text x="{wf_start_x+p50_w/2}" y="{comp_y+12}" font-size="7.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">Nominal Loop: SAFE (\u2264 \u0394t_fresh)</text>')

    # 2. Tail P99.9 Total = 68.4 ms
    p99_y = comp_y + 20
    p99_total = 68.4
    p99_w = (p99_total / wf_max_ms) * wf_plot_w
    svg.append(f'<rect x="{wf_start_x}" y="{p99_y}" width="{p99_w}" height="16" rx="3" fill="{CORAL}" fill-opacity="0.85" stroke="{CORAL}" stroke-width="1"/>')
    svg.append(f'<text x="{wf_start_x-8}" y="{p99_y+12}" font-size="8" font-weight="700" fill="{CRIMSON}" text-anchor="end">Empirical P99.9 (68.4 ms)</text>')
    svg.append(f'<text x="{wf_start_x+p99_w/2}" y="{p99_y+12}" font-size="7.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">TAIL SPIKE: BREACHES FRESHNESS (+28.4 ms STALE)</text>')

    # -------------------------------------------------------------
    # RIGHT CONTAINER: PHYSICAL CONSEQUENCES / CLEARANCE COLLAPSE
    # -------------------------------------------------------------
    rx = start_x + lw + 12
    rw = W - rx - start_x
    svg.append(f'<rect x="{rx}" y="{waterfall_box_y}" width="{rw}" height="{waterfall_box_h}" rx="8" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{rx}" y="{waterfall_box_y}" width="{rw}" height="28" rx="8" fill="{NAVY}" fill-opacity="0.08"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{waterfall_box_y+18}" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">PHYSICAL IMPACT: CLEARANCE COLLAPSE</text>')

    # Scenario explanation
    svg.append(f'<text x="{rx+12}" y="{waterfall_box_y+44}" font-size="8.5" font-weight="700" fill="{INK}">Blind Coasting Travel: d_react = v\u2080 \u00b7 \u0394t</text>')
    svg.append(f'<text x="{rx+12}" y="{waterfall_box_y+58}" font-size="7.8" fill="{SLATE}">Operating velocity v\u2080 = 1.2 m/s; clearance D_clear = 20.0 cm</text>')

    # Graphical Tracks for Nominal vs Tail
    track_y1 = waterfall_box_y + 80
    track_w = rw - 24
    
    cm_scale = (track_w - 40) / 20.0  # px per cm

    # Track 1: Nominal (P50 = 35 ms => d_react = 4.2 cm, d_brake = 7.2 cm)
    svg.append(f'<rect x="{rx+12}" y="{track_y1}" width="{track_w}" height="55" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+18}" y="{track_y1+14}" font-size="8" font-weight="700" fill="{PETROL}">CASE A: Nominal P50 (\u0394t = 35.0 ms)</text>')
    
    # Reaction bar (4.2 cm)
    r1_w = 4.2 * cm_scale
    svg.append(f'<rect x="{rx+18}" y="{track_y1+20}" width="{r1_w}" height="18" rx="2" fill="{BLUE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{rx+18+r1_w/2}" y="{track_y1+32}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">d_react 4.2cm</text>')
    
    # Braking bar (7.2 cm)
    b1_w = 7.2 * cm_scale
    svg.append(f'<rect x="{rx+18+r1_w}" y="{track_y1+20}" width="{b1_w}" height="18" rx="2" fill="{BRONZE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{rx+18+r1_w+b1_w/2}" y="{track_y1+32}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Braking 7.2cm</text>')

    # Clearance margin (8.6 cm)
    m1_w = (20.0 - 4.2 - 7.2) * cm_scale
    svg.append(f'<rect x="{rx+18+r1_w+b1_w}" y="{track_y1+20}" width="{m1_w}" height="18" rx="2" fill="{TEAL}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{rx+18+r1_w+b1_w+m1_w/2}" y="{track_y1+32}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Margin +8.6cm \u2713</text>')

    svg.append(f'<text x="{rx+18}" y="{track_y1+48}" font-size="7" font-weight="700" fill="{PETROL}">Total d_stop = 11.4 cm \u2264 20.0 cm (Certified Safe Operation)</text>')

    # Track 2: Tail Latency Spike (P99.9 = 68.4 ms => d_react = 8.2 cm, d_brake = 7.2 cm)
    track_y2 = track_y1 + 65
    svg.append(f'<rect x="{rx+12}" y="{track_y2}" width="{track_w}" height="55" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+18}" y="{track_y2+14}" font-size="8" font-weight="700" fill="{CRIMSON}">CASE B: Tail Latency P99.9 (\u0394t = 68.4 ms)</text>')
    
    # Reaction bar (8.2 cm)
    r2_w = 8.2 * cm_scale
    svg.append(f'<rect x="{rx+18}" y="{track_y2+20}" width="{r2_w}" height="18" rx="2" fill="{CORAL}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{rx+18+r2_w/2}" y="{track_y2+32}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">d_react 8.2cm (+4.0cm!)</text>')
    
    # Braking bar (7.2 cm)
    b2_w = 7.2 * cm_scale
    svg.append(f'<rect x="{rx+18+r2_w}" y="{track_y2+20}" width="{b2_w}" height="18" rx="2" fill="{BRONZE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{rx+18+r2_w+b2_w/2}" y="{track_y2+32}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Braking 7.2cm</text>')

    # Clearance margin (4.6 cm)
    m2_w = (20.0 - 8.2 - 7.2) * cm_scale
    svg.append(f'<rect x="{rx+18+r2_w+b2_w}" y="{track_y2+20}" width="{m2_w}" height="18" rx="2" fill="{AMBER}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{rx+18+r2_w+b2_w+m2_w/2}" y="{track_y2+32}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Margin 4.6cm \u26a0</text>')

    svg.append(f'<text x="{rx+18}" y="{track_y2+48}" font-size="7" font-weight="700" fill="{CRIMSON}">Total d_stop = 15.4 cm (Safety buffer eroded by 46.5%)</text>')

    # Summary callout box inside right container
    callout_y = track_y2 + 65
    callout_h = waterfall_box_h - (callout_y - waterfall_box_y) - 12
    svg.append(f'<rect x="{rx+12}" y="{callout_y}" width="{track_w}" height="{callout_h}" rx="5" fill="{NAVY}" fill-opacity="0.05" stroke="{NAVY}" stroke-width="1"/>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+16}" font-size="8" font-weight="700" fill="{NAVY}">FIRST-PRINCIPLES SYSTEMS IMPLICATION</text>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+30}" font-size="7.5" fill="{SLATE}">\u2022 Latency is not an elastic software metric;</text>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+43}" font-size="7.5" fill="{SLATE}">  it directly maps into unguided physical travel.</text>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+56}" font-size="7.5" fill="{SLATE}">\u2022 P99.9 tail latency dictates true safe speed limits,</text>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+69}" font-size="7.5" fill="{SLATE}">  not average computational throughput (P50).</text>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+82}" font-size="7.5" fill="{SLATE}">\u2022 Stale observations (\u0394t &gt; \u0394t_fresh) inject phase</text>')
    svg.append(f'<text x="{rx+18}" y="{callout_y+95}" font-size="7.5" fill="{SLATE}">  lag that destabilizes high-gain feedback loops.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/02-body/figures/fig02_latency_waterfall.svg", "\n".join(svg))


def gen_fig02_stopping_distance():
    """
    Fig 2.X: Defended Dynamic Stopping Envelope: Reaction Lag vs. Kinetic Braking Distance.
    Visualizing how computational latency produces linear reaction travel while physical
    deceleration produces quadratic kinetic braking distance.
    """
    W = 960
    H = 590
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    
    # Header
    svg.append(f'<text x="{W/2}" y="28" class="title">DEFENDED DYNAMIC STOPPING ENVELOPE: REACTION LAG VS. KINETIC BRAKING</text>')
    svg.append(f'<text x="{W/2}" y="45" class="subtitle">Linear Sensitivity to Computational Latency vs. Quadratic Sensitivity to Velocity: d_stop(v\u2080, \u0394t) = v\u2080\u0394t + v\u2080\u00b2/(2a) + \u03b4_overhead \u2264 D_clear</text>')

    # -------------------------------------------------------------
    # TOP EQUATION DECOMPOSITION CARD
    # -------------------------------------------------------------
    eq_y = 60
    eq_w = W - 40
    eq_h = 75
    svg.append(f'<rect x="20" y="{eq_y}" width="{eq_w}" height="{eq_h}" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1" filter="url(#shadow)"/>')
    
    # 3 Equation Term Columns
    col_w = (eq_w - 30) / 3
    
    # Term 1: Reaction Travel
    svg.append(f'<rect x="30" y="{eq_y+8}" width="{col_w}" height="{eq_h-16}" rx="4" fill="{BLUE}" fill-opacity="0.08" stroke="{BLUE}" stroke-width="1"/>')
    svg.append(f'<text x="{30+col_w/2}" y="{eq_y+26}" font-size="10.5" font-weight="700" fill="{BLUE}" text-anchor="middle">1. LINEAR REACTION TRAVEL</text>')
    svg.append(f'<text x="{30+col_w/2}" y="{eq_y+42}" font-size="11.5" font-weight="700" fill="{INK}" text-anchor="middle">d_react = v\u2080 \u00b7 \u0394t_delay</text>')
    svg.append(f'<text x="{30+col_w/2}" y="{eq_y+56}" font-size="8" fill="{MUTED}" text-anchor="middle">Unguided coasting during compute latency</text>')

    # Term 2: Kinetic Braking
    svg.append(f'<rect x="{35+col_w}" y="{eq_y+8}" width="{col_w}" height="{eq_h-16}" rx="4" fill="{BRONZE}" fill-opacity="0.08" stroke="{BRONZE}" stroke-width="1"/>')
    svg.append(f'<text x="{35+col_w+col_w/2}" y="{eq_y+26}" font-size="10.5" font-weight="700" fill="{BRONZE}" text-anchor="middle">2. QUADRATIC BRAKING DISTANCE</text>')
    svg.append(f'<text x="{35+col_w+col_w/2}" y="{eq_y+42}" font-size="11.5" font-weight="700" fill="{INK}" text-anchor="middle">d_brake = v\u2080\u00b2 / (2 \u00b7 a_brake)</text>')
    svg.append(f'<text x="{35+col_w+col_w/2}" y="{eq_y+56}" font-size="8" fill="{MUTED}" text-anchor="middle">Work-energy dissipation: E_k = \u00bd m v\u2080\u00b2</text>')

    # Term 3: Clearance Inequality
    svg.append(f'<rect x="{40+2*col_w}" y="{eq_y+8}" width="{col_w}" height="{eq_h-16}" rx="4" fill="{PETROL}" fill-opacity="0.08" stroke="{PETROL}" stroke-width="1"/>')
    svg.append(f'<text x="{40+2*col_w+col_w/2}" y="{eq_y+26}" font-size="10.5" font-weight="700" fill="{PETROL}" text-anchor="middle">3. MAX PERMITTED VELOCITY</text>')
    svg.append(f'<text x="{40+2*col_w+col_w/2}" y="{eq_y+42}" font-size="10" font-weight="700" fill="{INK}" text-anchor="middle">v_max = -a\u0394t + \u221a((a\u0394t)\u00b2 + 2aD_avail)</text>')
    svg.append(f'<text x="{40+2*col_w+col_w/2}" y="{eq_y+56}" font-size="8" fill="{MUTED}" text-anchor="middle">Defended clearance bound: d_stop \u2264 D_clear</text>')

    # -------------------------------------------------------------
    # MAIN CONTENT: 2 PANELS
    # Panel 1 (Left): Mathematical Scaling Curves (d vs v0)
    # Panel 2 (Right): Multi-Scenario Track Stopping Realities
    # -------------------------------------------------------------
    main_y = 145
    main_h = 370
    panel_w = (W - 50) / 2

    # LEFT PANEL: Scaling Curves
    lx = 20
    svg.append(f'<rect x="{lx}" y="{main_y}" width="{panel_w}" height="{main_h}" rx="8" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{lx}" y="{main_y}" width="{panel_w}" height="24" rx="8" fill="{NAVY}" fill-opacity="0.08"/>')
    svg.append(f'<text x="{lx+panel_w/2}" y="{main_y+16}" font-size="9" font-weight="700" fill="{NAVY}" text-anchor="middle">ASYMMETRIC PARAMETER SENSITIVITY CURVES</text>')

    # Plot Axes inside Left Panel
    ax_x0 = lx + 55
    ax_y0 = main_y + main_h - 45
    ax_w = panel_w - 75
    ax_h = main_h - 90

    # Axes lines
    svg.append(f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0+ax_w+10}" y2="{ax_y0}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<line x1="{ax_x0}" y1="{ax_y0}" x2="{ax_x0}" y2="{ax_y0-ax_h-10}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{ax_x0+ax_w+12}" y="{ax_y0+4}" font-size="8" font-weight="700" fill="{SLATE}">Speed v\u2080 (m/s)</text>')
    svg.append(f'<text x="{ax_x0-10}" y="{ax_y0-ax_h-12}" font-size="8" font-weight="700" fill="{SLATE}" text-anchor="end">Distance d (m)</text>')

    # X-axis ticks (0.0 to 2.5 m/s)
    v_max_plot = 2.5
    d_max_plot = 1.6
    for v_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
        tx = ax_x0 + (v_val / v_max_plot) * ax_w
        svg.append(f'<line x1="{tx}" y1="{ax_y0}" x2="{tx}" y2="{ax_y0+4}" stroke="{SLATE}" stroke-width="1"/>')
        svg.append(f'<text x="{tx}" y="{ax_y0+14}" font-size="7.5" fill="{MUTED}" text-anchor="middle">{v_val:.1f}</text>')
        svg.append(f'<line x1="{tx}" y1="{ax_y0}" x2="{tx}" y2="{ax_y0-ax_h}" stroke="{BORDER}" stroke-width="0.6" stroke-dasharray="2,2"/>')

    # Y-axis ticks (0.0 to 1.6 m)
    for d_val in [0.4, 0.8, 1.2, 1.6]:
        ty = ax_y0 - (d_val / d_max_plot) * ax_h
        svg.append(f'<line x1="{ax_x0-4}" y1="{ty}" x2="{ax_x0}" y2="{ty}" stroke="{SLATE}" stroke-width="1"/>')
        svg.append(f'<text x="{ax_x0-8}" y="{ty+3}" font-size="7.5" fill="{MUTED}" text-anchor="end">{d_val:.1f} m</text>')
        svg.append(f'<line x1="{ax_x0}" y1="{ty}" x2="{ax_x0+ax_w}" y2="{ty}" stroke="{BORDER}" stroke-width="0.6" stroke-dasharray="2,2"/>')

    # Plot Curves:
    pts_react = []
    pts_brake = []
    pts_total = []
    for step in range(51):
        v = (step / 50.0) * v_max_plot
        d_r = v * 0.080
        d_b = (v**2) / 4.0
        d_t = d_r + d_b + 0.10
        
        px = ax_x0 + (v / v_max_plot) * ax_w
        py_r = ax_y0 - (d_r / d_max_plot) * ax_h
        py_b = ax_y0 - (d_b / d_max_plot) * ax_h
        py_t = ax_y0 - min(d_t / d_max_plot, 1.05) * ax_h
        
        pts_react.append(f"{px:.1f},{py_r:.1f}")
        pts_brake.append(f"{px:.1f},{py_b:.1f}")
        if d_t <= d_max_plot * 1.05:
            pts_total.append(f"{px:.1f},{py_t:.1f}")

    svg.append(f'<polyline points="{" ".join(pts_react)}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>')
    svg.append(f'<polyline points="{" ".join(pts_brake)}" fill="none" stroke="{BRONZE}" stroke-width="1.8"/>')
    svg.append(f'<polyline points="{" ".join(pts_total)}" fill="none" stroke="{NAVY}" stroke-width="2.4"/>')

    # Fixed Clearance Line at D_clear = 1.0 m
    clear_y = ax_y0 - (1.0 / d_max_plot) * ax_h
    svg.append(f'<line x1="{ax_x0}" y1="{clear_y}" x2="{ax_x0+ax_w}" y2="{clear_y}" stroke="{CORAL}" stroke-width="1.8" stroke-dasharray="4,3"/>')
    svg.append(f'<rect x="{ax_x0+ax_w-130}" y="{clear_y-14}" width="130" height="13" rx="2" fill="{CORAL}"/>')
    svg.append(f'<text x="{ax_x0+ax_w-65}" y="{clear_y-4}" font-size="7.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">CLEARANCE D_clear = 1.0 m</text>')

    # Curve Labels
    svg.append(f'<text x="{ax_x0+110}" y="{ax_y0-22}" font-size="8" font-weight="700" fill="{BLUE}">d_react \u221d v\u2080 (Linear)</text>')
    svg.append(f'<text x="{ax_x0+ax_w-50}" y="{ax_y0-85}" font-size="8" font-weight="700" fill="{BRONZE}" text-anchor="end">d_brake \u221d v\u2080\u00b2 (Quadratic)</text>')
    svg.append(f'<text x="{ax_x0+ax_w-90}" y="{clear_y-20}" font-size="8.5" font-weight="700" fill="{NAVY}">Total d_stop Envelope</text>')

    # Asymmetric Scaling Callout text inside plot
    svg.append(f'<rect x="{ax_x0+10}" y="{main_y+35}" width="180" height="52" rx="4" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="0.8"/>')
    svg.append(f'<text x="{ax_x0+16}" y="{main_y+48}" font-size="7.5" font-weight="700" fill="{INK}">THE ASYMMETRY PRINCIPLE:</text>')
    svg.append(f'<text x="{ax_x0+16}" y="{main_y+60}" font-size="7" fill="{SLATE}">\u2022 2\u00d7 Delay (\u0394t) \u2192 2\u00d7 Reaction (+17% d_stop)</text>')
    svg.append(f'<text x="{ax_x0+16}" y="{main_y+72}" font-size="7" fill="{CRIMSON}">\u2022 2\u00d7 Speed (v\u2080) \u2192 4\u00d7 Braking (+267% d_stop!)</text>')


    # RIGHT PANEL: Physical Multi-Scenario Track Realities
    rx = lx + panel_w + 10
    svg.append(f'<rect x="{rx}" y="{main_y}" width="{panel_w}" height="{main_h}" rx="8" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{rx}" y="{main_y}" width="{panel_w}" height="24" rx="8" fill="{NAVY}" fill-opacity="0.08"/>')
    svg.append(f'<text x="{rx+panel_w/2}" y="{main_y+16}" font-size="9" font-weight="700" fill="{NAVY}" text-anchor="middle">PHYSICAL SCENARIO TRAJECTORIES (D_clear = 60.0 cm)</text>')

    track_x0 = rx + 20
    track_w = panel_w - 40
    track_scale = (track_w - 60) / 60.0  # px per cm

    # Scenario 1: Nominal Case
    s1_y = main_y + 35
    svg.append(f'<rect x="{track_x0}" y="{s1_y}" width="{track_w}" height="70" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{track_x0+8}" y="{s1_y+14}" font-size="8" font-weight="700" fill="{PETROL}">CASE 1: Nominal Speed &amp; Latency (v\u2080 = 1.0 m/s, \u0394t = 50 ms)</text>')
    
    # 5.0 cm react + 25.0 cm brake + 10.0 cm margin = 40.0 cm
    w1_r = 5.0 * track_scale
    w1_b = 25.0 * track_scale
    w1_m = (60.0 - 40.0) * track_scale
    
    svg.append(f'<rect x="{track_x0+8}" y="{s1_y+22}" width="{w1_r}" height="20" rx="2" fill="{BLUE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w1_r/2}" y="{s1_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">5cm</text>')
    
    svg.append(f'<rect x="{track_x0+8+w1_r}" y="{s1_y+22}" width="{w1_b}" height="20" rx="2" fill="{BRONZE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w1_r+w1_b/2}" y="{s1_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Braking 25.0 cm</text>')

    svg.append(f'<rect x="{track_x0+8+w1_r+w1_b}" y="{s1_y+22}" width="{w1_m}" height="20" rx="2" fill="{TEAL}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w1_r+w1_b+w1_m/2}" y="{s1_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Margin +20.0 cm \u2713</text>')

    svg.append(f'<text x="{track_x0+8}" y="{s1_y+58}" font-size="7.5" font-weight="700" fill="{PETROL}">Total Stop d_stop = 40.0 cm \u2264 60.0 cm (CERTIFIED SAFE)</text>')

    # Scenario 2: Tail Latency Spike
    s2_y = s1_y + 80
    svg.append(f'<rect x="{track_x0}" y="{s2_y}" width="{track_w}" height="70" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{track_x0+8}" y="{s2_y+14}" font-size="8" font-weight="700" fill="{AMBER}">CASE 2: Tail Latency Spike (v\u2080 = 1.0 m/s, \u0394t = 200 ms)</text>')
    
    # 20.0 cm react + 25.0 cm brake + 5.0 cm margin = 55.0 cm
    w2_r = 20.0 * track_scale
    w2_b = 25.0 * track_scale
    w2_m = (60.0 - 55.0) * track_scale
    
    svg.append(f'<rect x="{track_x0+8}" y="{s2_y+22}" width="{w2_r}" height="20" rx="2" fill="{AMBER}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w2_r/2}" y="{s2_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Reaction 20.0 cm (+15cm unguided!)</text>')
    
    svg.append(f'<rect x="{track_x0+8+w2_r}" y="{s2_y+22}" width="{w2_b}" height="20" rx="2" fill="{BRONZE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w2_r+w2_b/2}" y="{s2_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Braking 25cm</text>')

    svg.append(f'<rect x="{track_x0+8+w2_r+w2_b}" y="{s2_y+22}" width="{w2_m}" height="20" rx="2" fill="{CORAL}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w2_r+w2_b+w2_m/2}" y="{s2_y+35}" font-size="6.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">+5cm</text>')

    svg.append(f'<text x="{track_x0+8}" y="{s2_y+58}" font-size="7.5" font-weight="700" fill="{AMBER}">Total Stop d_stop = 55.0 cm (CLEARANCE MARGIN COMPRESSED TO 5 cm)</text>')

    # Scenario 3: High Speed + Tail Latency (Collision)
    s3_y = s2_y + 80
    svg.append(f'<rect x="{track_x0}" y="{s3_y}" width="{track_w}" height="70" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{track_x0+8}" y="{s3_y+14}" font-size="8" font-weight="700" fill="{CRIMSON}">CASE 3: Speed Increase + Tail Latency (v\u2080 = 1.5 m/s, \u0394t = 200 ms)</text>')
    
    # 30.0 cm react + 56.25 cm brake = 86.25 cm > 60 cm (Breach = +26.25 cm)
    w3_r = 30.0 * track_scale
    w3_b = (60.0 - 30.0) * track_scale  # capped at barrier for visual
    w3_breach = 26.25 * track_scale
    
    svg.append(f'<rect x="{track_x0+8}" y="{s3_y+22}" width="{w3_r}" height="20" rx="2" fill="{CORAL}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w3_r/2}" y="{s3_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Reaction 30.0 cm</text>')
    
    svg.append(f'<rect x="{track_x0+8+w3_r}" y="{s3_y+22}" width="{w3_b}" height="20" rx="2" fill="{BRONZE}" fill-opacity="0.85"/>')
    svg.append(f'<text x="{track_x0+8+w3_r+w3_b/2}" y="{s3_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">Braking 30cm</text>')

    # Barrier impact breach bar extending past 60 cm
    svg.append(f'<rect x="{track_x0+8+w3_r+w3_b}" y="{s3_y+22}" width="{min(w3_breach, track_w-w3_r-w3_b-16)}" height="20" rx="2" fill="{CRIMSON}"/>')
    svg.append(f'<text x="{track_x0+8+w3_r+w3_b+w3_breach/2}" y="{s3_y+35}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">IMPACT (+26.3cm)</text>')

    svg.append(f'<text x="{track_x0+8}" y="{s3_y+58}" font-size="7.5" font-weight="700" fill="{CRIMSON}">Total Stop d_stop = 86.3 cm &gt; 60.0 cm (CATASTROPHIC IMPACT BREACH!)</text>')

    # Physical Clearance Barrier Line across all 3 tracks
    barrier_x = track_x0 + 8 + 60.0 * track_scale
    svg.append(f'<line x1="{barrier_x}" y1="{s1_y+10}" x2="{barrier_x}" y2="{s3_y+65}" stroke="{CORAL}" stroke-width="2.2" stroke-dasharray="4,3"/>')
    svg.append(f'<rect x="{barrier_x-40}" y="{s1_y-2}" width="80" height="14" rx="2" fill="{CORAL}"/>')
    svg.append(f'<text x="{barrier_x}" y="{s1_y+8}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">WALL 60.0 cm</text>')

    # Bottom Invariant Bar
    svg.append(f'<rect x="20" y="525" width="{W-40}" height="45" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="543" font-size="8.5" font-weight="700" fill="{NAVY}" text-anchor="middle">FIRST-PRINCIPLES PHYSICAL AI DESIGN LAW</text>')
    svg.append(f'<text x="{W/2}" y="558" font-size="8" fill="{SLATE}" text-anchor="middle">Software models propose speed setpoints; the Body enforces quadratic kinetic momentum. Safe operation requires budgeting P99.9 latency and defending physical clearance.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/02-body/figures/fig02_stopping_distance.svg", "\n".join(svg))


def run_all():
    print("Generating Chapter 2 Vector Diagrams...")
    gen_fig02_latency_waterfall()
    gen_fig02_stopping_distance()
    print("✓ Chapter 2 Diagrams generated successfully.")

if __name__ == "__main__":
    run_all()
