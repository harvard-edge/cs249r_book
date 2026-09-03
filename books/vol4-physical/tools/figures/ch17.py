"""
book/tools/figures/ch17.py
Figures for Chapter 17: Frontier — What This Method Cannot Settle, and What Would Change That.
Harvard Crimson & ETH Zurich Semantic Palette.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch17_closed_loop_synthesis():
    W = 1040
    H = 680
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE CLOSED CAUSAL LOOP &amp; EPISTEMIC BOUNDARY SYNTHESIS</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Synthesizing All 17 Chapters Across 4 Parts: Sensory Transduction, Cognitive Deliberation, Real-Time Invariant Enforcement, and Physical Dynamics</text>')

    # -------------------------------------------------------------
    # TOP STRIP: The 4 Parts of the Whole Curriculum (Chapters 1-17)
    # -------------------------------------------------------------
    card_w = 236
    gap = 14
    start_x = (W - (4 * card_w + 3 * gap)) / 2

    part_boxes = [
        ("PART I: FOUNDATIONS (Ch 1–4)", "Body, Brain &amp; Nervous System", NAVY, [
            "Ch 1: The Causal Boundary",
            "Ch 2: Physical Body &amp; Dynamics",
            "Ch 3: Learned Brain &amp; Latency",
            "Ch 4: Nervous System &amp; Clocks"
        ]),
        ("PART II: DATA &amp; LEARNING (Ch 5–7)", "Demonstrations, Training &amp; Bounds", BLUE, [
            "Ch 5: Sensor Demonstration Data",
            "Ch 6: Imitation &amp; RL Training",
            "Ch 7: Evaluation &amp; Rare Events",
            "Finite-Sample Exposure Bounds"
        ]),
        ("PART III: ONLINE RUNTIME (Ch 8–13)", "Perception to Edge Placement", BRONZE, [
            "Ch 8: Perception · Ch 9: Memory",
            "Ch 10: Intent · Ch 11: Planning",
            "Ch 12: Real-Time Safety Filter",
            "Ch 13: Heterogeneous Placement"
        ]),
        ("PART IV: GOVERNANCE (Ch 14–17)", "Verification, Release &amp; Frontier", PURPLE, [
            "Ch 14: Shared Intervention",
            "Ch 15: Empirical Stress Testing",
            "Ch 16: Formal Safety Release",
            "Ch 17: The Epistemic Frontier"
        ])
    ]

    for i, (p_tag, p_title, col, chs) in enumerate(part_boxes):
        px = start_x + i * (card_w + gap)
        py = 62
        ph = 88
        svg.append(f'<rect x="{px}" y="{py}" width="{card_w}" height="{ph}" rx="6" fill="{BG_LIGHT}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{px}" y="{py}" width="{card_w}" height="20" rx="6" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{px+card_w/2}" y="{py+14}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle">{p_tag}</text>')
        svg.append(f'<text x="{px+card_w/2}" y="{py+33}" font-size="9" font-weight="700" fill="{INK}" text-anchor="middle">{p_title}</text>')
        for c_idx, ch_txt in enumerate(chs):
            svg.append(f'<text x="{px+card_w/2}" y="{py+47+c_idx*12}" font-size="7.8" fill="{SLATE}" text-anchor="middle">{ch_txt}</text>')

    # -------------------------------------------------------------
    # MAIN ARCHITECTURE: 3 Top Blocks + Causal Interface
    # -------------------------------------------------------------
    top_y = 168
    box_h = 195

    # 1. BRAIN (System 2 / Linux MPU / High-Capacity Learned Policies)
    bx = start_x
    bw = 295
    svg.append(f'<rect x="{bx}" y="{top_y}" width="{bw}" height="{box_h}" rx="8" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.4" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{bx}" y="{top_y}" width="{bw}" height="24" rx="8" fill="{NAVY}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{bx+bw/2}" y="{top_y+16}" font-size="10" font-weight="700" fill="{NAVY}" text-anchor="middle">THE BRAIN: COGNITIVE DELIBERATION</text>')
    svg.append(f'<text x="{bx+12}" y="{top_y+40}" font-size="11" font-weight="700" fill="{INK}">High-Capacity Stochastic Policy (MPU)</text>')
    svg.append(f'<text x="{bx+12}" y="{top_y+54}" font-size="8.5" font-weight="600" fill="{BLUE}">Cadence: 20–50 Hz · Linux / NPU · Latency τ_infer</text>')
    
    brain_items = [
        "VLM Semantic Goal Grounding (Ch 10)",
        "SE(3) Dynamic Kinematic Frame Tree (Ch 9)",
        "Diffusion / ACT Action Chunks H=16 (Ch 11)",
        "Expiring Intent Lease Issuance t_expire (Ch 10)",
        "Epistemic Uncertainty &amp; Silent Drift (Ch 7, 17)"
    ]
    for idx, item in enumerate(brain_items):
        svg.append(f'<text x="{bx+12}" y="{top_y+74+idx*17}" font-size="8.5" fill="{SLATE}">• {item}</text>')

    svg.append(f'<rect x="{bx+10}" y="{top_y+box_h-26}" width="{bw-20}" height="18" rx="4" fill="{NAVY}" fill-opacity="0.08"/>')
    svg.append(f'<text x="{bx+bw/2}" y="{top_y+box_h-13}" font-size="8" font-weight="700" fill="{NAVY}" text-anchor="middle">EMITS: Candidate Action Trajectory û_t:t+H</text>')

    # 2. CAUSAL AUTHORITY INTERFACE (Center Column)
    cx = bx + bw + 30
    cw = 320
    svg.append(f'<rect x="{cx}" y="{top_y}" width="{cw}" height="{box_h}" rx="8" fill="{BG_LIGHT}" stroke="{PURPLE}" stroke-width="1.3" stroke-dasharray="4,3"/>')
    svg.append(f'<rect x="{cx}" y="{top_y}" width="{cw}" height="24" rx="8" fill="{PURPLE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{cx+cw/2}" y="{top_y+16}" font-size="9.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">PROPOSAL–PERMISSION CAUSAL INTERFACE</text>')
    svg.append(f'<text x="{cx+cw/2}" y="{top_y+38}" font-size="10.5" font-weight="700" fill="{INK}" text-anchor="middle">Hardware Architectural Separation (Ch 13, 17)</text>')

    svg.append(f'<line x1="{cx+15}" y1="{top_y+48}" x2="{cx+cw-15}" y2="{top_y+48}" stroke="{BORDER}" stroke-width="1"/>')

    iface_rows = [
        ("Shared SRAM TCM", "Zero-copy lock-free seqlock mailbox"),
        ("Dead-Man Leases", "Lease t_expire bounds MPU stalls ≤ 50 ms"),
        ("Invariant Projection", "u_t* = argmin ||u - û||² s.t. h(x) ≥ 0"),
        ("Authority Hierarchy", "Reflex Stop > Human Override > Policy")
    ]
    for r_idx, (r_title, r_desc) in enumerate(iface_rows):
        svg.append(f'<text x="{cx+14}" y="{top_y+66+r_idx*24}" font-size="8.5" font-weight="700" fill="{PURPLE}">▸ {r_title}:</text>')
        svg.append(f'<text x="{cx+14}" y="{top_y+78+r_idx*24}" font-size="8" fill="{SLATE}">{r_desc}</text>')

    # 3. NERVOUS SYSTEM (System 1 / RTOS MCU / Real-Time Safety Enforcer)
    nx = cx + cw + 30
    nw = 295
    svg.append(f'<rect x="{nx}" y="{top_y}" width="{nw}" height="{box_h}" rx="8" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.4" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{nx}" y="{top_y}" width="{nw}" height="24" rx="8" fill="{PETROL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{nx+nw/2}" y="{top_y+16}" font-size="10" font-weight="700" fill="{PETROL}" text-anchor="middle">THE NERVOUS SYSTEM: REFLEX ENFORCER</text>')
    svg.append(f'<text x="{nx+12}" y="{top_y+40}" font-size="11" font-weight="700" fill="{INK}">Deterministic Invariant Filter (MCU)</text>')
    svg.append(f'<text x="{nx+12}" y="{top_y+54}" font-size="8.5" font-weight="600" fill="{PETROL}">Cadence: 1000 Hz · FreeRTOS / Cortex-M · Hard Real-Time</text>')

    nervous_items = [
        "Real-Time Active-Set QP Solver: h(x) ≥ 0 (Ch 12)",
        "Dynamic Stopping Clearance: d_stop(v) ≤ d_gap (Ch 2)",
        "Lock-Free Inter-Core Mailbox &amp; Lease Check (Ch 4)",
        "Shared Intervention &amp; C² Bumpless Takeover (Ch 14)",
        "Hardware Watchdog &amp; ISO 13849 Safety Relay (Ch 15)"
    ]
    for idx, item in enumerate(nervous_items):
        svg.append(f'<text x="{nx+12}" y="{top_y+74+idx*17}" font-size="8.5" fill="{SLATE}">• {item}</text>')

    svg.append(f'<rect x="{nx+10}" y="{top_y+box_h-26}" width="{nw-20}" height="18" rx="4" fill="{PETROL}" fill-opacity="0.08"/>')
    svg.append(f'<text x="{nx+nw/2}" y="{top_y+box_h-13}" font-size="8" font-weight="700" fill="{PETROL}" text-anchor="middle">PERMITS: Verified Motor Commands u_t*</text>')

    # Connectors between top blocks
    # Brain -> Interface
    svg.append(f'<line x1="{bx+bw}" y1="{top_y+95}" x2="{cx}" y2="{top_y+95}" stroke="{NAVY}" stroke-width="2" marker-end="url(#arr-navy)"/>')
    svg.append(f'<rect x="{bx+bw+2}" y="{top_y+78}" width="26" height="14" rx="2" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="0.7"/>')
    svg.append(f'<text x="{bx+bw+15}" y="{top_y+88}" font-size="7" font-weight="700" fill="{NAVY}" text-anchor="middle">û_t</text>')

    # Interface -> MCU
    svg.append(f'<line x1="{cx+cw}" y1="{top_y+95}" x2="{nx}" y2="{top_y+95}" stroke="{PETROL}" stroke-width="2" marker-end="url(#arr-petrol)"/>')
    svg.append(f'<rect x="{cx+cw+2}" y="{top_y+78}" width="26" height="14" rx="2" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="0.7"/>')
    svg.append(f'<text x="{cx+cw+15}" y="{top_y+88}" font-size="7" font-weight="700" fill="{PETROL}" text-anchor="middle">Gate</text>')

    # -------------------------------------------------------------
    # BOTTOM HALF: SENSORY TRANSDUCTION & PHYSICAL PLANT
    # -------------------------------------------------------------
    bot_y = 390
    bot_h = 170
    bot_w = (W - 2 * start_x - 30) / 2

    # 4. SENSORY TRANSDUCTION (Bottom Left)
    sx = start_x
    svg.append(f'<rect x="{sx}" y="{bot_y}" width="{bot_w}" height="{bot_h}" rx="8" fill="{BLUE}" fill-opacity="0.04" stroke="{BLUE}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{sx}" y="{bot_y}" width="{bot_w}" height="24" rx="8" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{sx+bot_w/2}" y="{bot_y+16}" font-size="10" font-weight="700" fill="{BLUE}" text-anchor="middle">SENSORY TRANSDUCTION &amp; INFORMATION FRESHNESS</text>')
    svg.append(f'<text x="{sx+14}" y="{bot_y+40}" font-size="10.5" font-weight="700" fill="{INK}">Photons / Forces → Digital Bits (The Sensing Boundary)</text>')

    trans_items = [
        "Transduction: MIPI CSI-2 CMOS, IMU gyros, optical wheel encoders (Ch 8)",
        "Zero-Copy DMA: Ring buffers, hardware PTP nanosecond timestamps",
        "Information Age Freshness: τ_total = τ_readout + τ_DMA + τ_infer + τ_act",
        "Spatial Displacement Lag: Unchecked motion d_lag = v · τ_total (Ch 2, 17)",
        "Endogenous Feedback: Action u_t shifts pose, shaping future observations y_t+1"
    ]
    for idx, item in enumerate(trans_items):
        svg.append(f'<text x="{sx+14}" y="{bot_y+58+idx*17}" font-size="8.5" fill="{SLATE}">• {item}</text>')

    # 5. PHYSICAL PLANT & CONTINUOUS DYNAMICS (Bottom Right)
    px = sx + bot_w + 30
    svg.append(f'<rect x="{px}" y="{bot_y}" width="{bot_w}" height="{bot_h}" rx="8" fill="{CRIMSON}" fill-opacity="0.04" stroke="{CRIMSON}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{px}" y="{bot_y}" width="{bot_w}" height="24" rx="8" fill="{CRIMSON}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{px+bot_w/2}" y="{bot_y+16}" font-size="10" font-weight="700" fill="{CRIMSON}" text-anchor="middle">THE PHYSICAL PLANT &amp; NEWTONIAN REALITY (W_t → W_t+1)</text>')
    svg.append(f'<text x="{px+14}" y="{bot_y+40}" font-size="10.5" font-weight="700" fill="{INK}">Continuous Mechanics, Conservation Laws &amp; Irreversibility</text>')

    phys_items = [
        "Inertial Momentum: Kinetic energy E_k = 1/2 m v² (No software rollback!)",
        "Tire/Surface Coulomb Friction Limits: F_friction ≤ μ N (Ch 2, 17)",
        "Non-Smooth Contact Dynamics: Painlevé paradoxes &amp; brittle yield (Ch 2)",
        "Actuator Dissipation: Thermal Joule heating P_loss = I²R, winding limits",
        "Physical State Update: W_t+1 = Dynamics(W_t, u_t*, Disturbances)"
    ]
    for idx, item in enumerate(phys_items):
        svg.append(f'<text x="{px+14}" y="{bot_y+58+idx*17}" font-size="8.5" fill="{SLATE}">• {item}</text>')

    # Connectors for Closed Loop
    # MCU down to Physical Plant
    svg.append(f'<line x1="{nx+nw/2}" y1="{top_y+box_h}" x2="{nx+nw/2}" y2="{bot_y}" stroke="{CRIMSON}" stroke-width="2" marker-end="url(#arr-crimson)"/>')
    svg.append(f'<rect x="{nx+nw/2-55}" y="{top_y+box_h+7}" width="110" height="14" rx="2" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="0.8"/>')
    svg.append(f'<text x="{nx+nw/2}" y="{top_y+box_h+17}" font-size="7.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">Actuation Torques u_t*</text>')

    # Physical Plant left to Transduction (Environment interaction)
    svg.append(f'<line x1="{px}" y1="{bot_y+bot_h/2}" x2="{sx+bot_w}" y2="{bot_y+bot_h/2}" stroke="{BLUE}" stroke-width="2" marker-end="url(#arr-blue)"/>')
    svg.append(f'<rect x="{sx+bot_w+4}" y="{bot_y+bot_h/2-14}" width="22" height="28" rx="2" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="0.8"/>')
    svg.append(f'<text x="{sx+bot_w+15}" y="{bot_y+bot_h/2-3}" font-size="7" font-weight="700" fill="{BLUE}" text-anchor="middle">W_t</text>')
    svg.append(f'<text x="{sx+bot_w+15}" y="{bot_y+bot_h/2+9}" font-size="6.5" fill="{SLATE}" text-anchor="middle">State</text>')

    # Transduction up to Brain (Vision/DMA)
    svg.append(f'<line x1="{sx+100}" y1="{bot_y}" x2="{bx+100}" y2="{top_y+box_h}" stroke="{BLUE}" stroke-width="2" marker-end="url(#arr-blue)"/>')
    svg.append(f'<rect x="{sx+45}" y="{top_y+box_h+7}" width="110" height="14" rx="2" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="0.8"/>')
    svg.append(f'<text x="{sx+100}" y="{top_y+box_h+17}" font-size="7.5" font-weight="700" fill="{BLUE}" text-anchor="middle">Sensor Stream y_t (DMA)</text>')

    # Transduction up to MCU (Fast Proprioception 1 kHz)
    svg.append(f'<path d="M {sx+bot_w-60} {bot_y} L {sx+bot_w-60} {top_y+box_h+14} L {nx+50} {top_y+box_h+14} L {nx+50} {top_y+box_h}" fill="none" stroke="{PETROL}" stroke-width="1.6" stroke-dasharray="4,2" marker-end="url(#arr-petrol)"/>')
    svg.append(f'<rect x="{cx+cw/2-60}" y="{top_y+box_h+7}" width="120" height="14" rx="2" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{cx+cw/2}" y="{top_y+box_h+17}" font-size="7.2" font-weight="700" fill="{PETROL}" text-anchor="middle">1 kHz Fast Odometry / Encoders</text>')

    # -------------------------------------------------------------
    # BOTTOM BANNER: CHAPTER 17 EPISTEMIC FRONTIER SYNTHESIS
    # -------------------------------------------------------------
    fy = 580
    fw = W - 2 * start_x
    fh = 82
    svg.append(f'<rect x="{start_x}" y="{fy}" width="{fw}" height="{fh}" rx="8" fill="{BG_LIGHT}" stroke="{CORAL}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{start_x}" y="{fy}" width="{fw}" height="22" rx="8" fill="{CORAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{W/2}" y="{fy+15}" font-size="9.5" font-weight="700" fill="{CORAL}" text-anchor="middle">CHAPTER 17 FRONTIER SYNTHESIS: THE BOUNDARY OF VERIFIABLE KNOWLEDGE</text>')

    col_w_f = (fw - 40) / 3
    
    # Col 1: Detectorless Gaps
    f1_x = start_x + 15
    svg.append(f'<text x="{f1_x}" y="{fy+38}" font-size="9" font-weight="700" fill="{INK}">1. Detectorless Blind Spots</text>')
    svg.append(f'<text x="{f1_x}" y="{fy+52}" font-size="8" fill="{SLATE}">States S_A &amp; S_B produce identical telemetry prior to</text>')
    svg.append(f'<text x="{f1_x}" y="{fy+65}" font-size="8" fill="{SLATE}">deadline t_deadline (internal voids, friction drops, drift).</text>')

    # Col 2: Astronomical Exposure Wall
    f2_x = start_x + 15 + col_w_f + 10
    svg.append(f'<text x="{f2_x}" y="{fy+38}" font-size="9" font-weight="700" fill="{INK}">2. Astronomical Exposure Limits</text>')
    svg.append(f'<text x="{f2_x}" y="{fy+52}" font-size="8" fill="{SLATE}">Proving p ≤ 10⁻⁹/h takes n ≥ 3.0×10⁹ h (342k yrs).</text>')
    svg.append(f'<text x="{f2_x}" y="{fy+65}" font-size="8" fill="{SLATE}">Empirical sampling cannot certify open-world safety.</text>')

    # Col 3: Architectural Containment
    f3_x = start_x + 15 + 2 * (col_w_f + 10)
    svg.append(f'<text x="{f3_x}" y="{fy+38}" font-size="9" font-weight="700" fill="{INK}">3. Structural Containment Mandate</text>')
    svg.append(f'<text x="{f3_x}" y="{fy+52}" font-size="8" fill="{SLATE}">Where evidence ends, nervous system clamps authority</text>')
    svg.append(f'<text x="{f3_x}" y="{fy+65}" font-size="8" fill="{SLATE}">(v_clamp, force limits) to render the unknown harmless.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/17-frontier/figures/fig17_closed_loop_synthesis.svg", "\n".join(svg))


def gen_ch17_epistemic_limits():
    W = 1040
    H = 490
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE DUAL EPISTEMIC BOUNDARIES: DETECTOR DEADLINES &amp; EXPOSURE WALLS</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Why Empirical Testing Alone Cannot Settle Open-World Safety, Demanding Architectural Authority Limits and Closure Tests</text>')

    panel_w = 316
    panel_gap = 18
    panel_y = 66
    panel_h = 400
    start_x = (W - (3 * panel_w + 2 * panel_gap)) / 2

    # -------------------------------------------------------------
    # PANEL 1: OBSERVATIONAL INDISTINGUISHABILITY & DETECTOR DEADLINES
    # -------------------------------------------------------------
    p1_x = start_x
    svg.append(f'<rect x="{p1_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{p1_x}" y="{panel_y}" width="{panel_w}" height="26" rx="8" fill="{NAVY}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p1_x+panel_w/2}" y="{panel_y+17}" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">1. OBSERVABILITY DEADLINE LIMIT</text>')
    
    svg.append(f'<text x="{p1_x+12}" y="{panel_y+44}" font-size="10.5" font-weight="700" fill="{INK}">Identical Pre-Harm Telemetry</text>')
    svg.append(f'<text x="{p1_x+12}" y="{panel_y+58}" font-size="8.5" fill="{MUTED}">States S_A (Safe) vs S_B (Hazard) indistinguishable</text>')

    # Timeline graph inside Panel 1
    gy = panel_y + 70
    gw = panel_w - 24
    gh = 160
    svg.append(f'<rect x="{p1_x+12}" y="{gy}" width="{gw}" height="{gh}" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    
    # Axes
    svg.append(f'<line x1="{p1_x+30}" y1="{gy+gh-30}" x2="{p1_x+gw-5}" y2="{gy+gh-30}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{p1_x+gw-5}" y="{gy+gh-16}" font-size="8" fill="{SLATE}">Time t</text>')
    svg.append(f'<line x1="{p1_x+30}" y1="{gy+gh-30}" x2="{p1_x+30}" y2="{gy+15}" stroke="{SLATE}" stroke-width="1.2"/>')
    svg.append(f'<text x="{p1_x+34}" y="{gy+22}" font-size="8" fill="{SLATE}">Telemetry y(t)</text>')

    # Identical Telemetry Line
    svg.append(f'<line x1="{p1_x+30}" y1="{gy+55}" x2="{p1_x+170}" y2="{gy+55}" stroke="{BLUE}" stroke-width="2"/>')
    svg.append(f'<text x="{p1_x+38}" y="{gy+48}" font-size="8" font-weight="700" fill="{BLUE}">y_A(t) ≡ y_B(t) = y₀</text>')

    # Deadline & Fracture markers (spaced apart cleanly)
    svg.append(f'<line x1="{p1_x+105}" y1="{gy+18}" x2="{p1_x+105}" y2="{gy+gh-30}" stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{p1_x+105}" y="{gy+14}" font-size="7.5" font-weight="700" fill="{AMBER}" text-anchor="middle">t_deadline</text>')

    svg.append(f'<line x1="{p1_x+170}" y1="{gy+18}" x2="{p1_x+170}" y2="{gy+gh-30}" stroke="{CRIMSON}" stroke-width="1.2" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{p1_x+170}" y="{gy+14}" font-size="7.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">t_harm (38 ms)</text>')

    # Divergence curve post t_harm with clear label above
    svg.append(f'<text x="{p1_x+225}" y="{gy+35}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">Fracture / Slip</text>')
    svg.append(f'<path d="M {p1_x+170} {gy+55} Q {p1_x+210} {gy+55} {p1_x+260} {gy+42}" fill="none" stroke="{CORAL}" stroke-width="2"/>')

    # Sensing + Processing latency window
    svg.append(f'<rect x="{p1_x+30}" y="{gy+gh-48}" width="200" height="15" rx="2" fill="{PURPLE}" fill-opacity="0.15" stroke="{PURPLE}" stroke-width="0.8"/>')
    svg.append(f'<text x="{p1_x+130}" y="{gy+gh-37}" font-size="7.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">Detection Latency τ_detect = 64 ms</text>')

    p1_desc = [
        ("State S_A vs S_B:", "Nominal surface vs oil slick / internal void"),
        ("Pre-Contact:", "Zero optical or tactile surface signature"),
        ("Late Detection:", "Harm occurs before reflex engages (38 ms < 64 ms)"),
        ("Physical Theorem:", "Identical y(t) implies safe intervention cannot"),
        ("Epistemic Result:", "exceed prior without structural force clamping")
    ]
    for idx, (dt, dd) in enumerate(p1_desc):
        svg.append(f'<text x="{p1_x+12}" y="{gy+gh+18+idx*16}" font-size="8" font-weight="700" fill="{INK}">• {dt}</text>')
        svg.append(f'<text x="{p1_x+115}" y="{gy+gh+18+idx*16}" font-size="8" fill="{SLATE}">{dd}</text>')

    # -------------------------------------------------------------
    # PANEL 2: THE ASTRONOMICAL EXPOSURE SCALING WALL
    # -------------------------------------------------------------
    p2_x = start_x + panel_w + panel_gap
    svg.append(f'<rect x="{p2_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{p2_x}" y="{panel_y}" width="{panel_w}" height="26" rx="8" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p2_x+panel_w/2}" y="{panel_y+17}" font-size="9.5" font-weight="700" fill="{BLUE}" text-anchor="middle">2. ASTRONOMICAL EXPOSURE WALL</text>')

    svg.append(f'<text x="{p2_x+12}" y="{panel_y+44}" font-size="10.5" font-weight="700" fill="{INK}">n ≥ ln(1/α) / p at 95% Confidence</text>')
    svg.append(f'<text x="{p2_x+12}" y="{panel_y+58}" font-size="8.5" fill="{MUTED}">Required failure-free hours grows reciprocally with 1/p</text>')

    # Exposure Table / Scaling Rows inside Panel 2
    ey = panel_y + 70
    ew = panel_w - 24
    
    tiers = [
        ("p ≤ 10⁻²/h", "Consumer App", "n ≥ 300 h", "12.5 days", TEAL, "Tractable (Unit Test)"),
        ("p ≤ 10⁻⁴/h", "Industrial Cobot", "n ≥ 3.0×10⁴ h", "3.4 years", BLUE, "Affordable (Bench Rig)"),
        ("p ≤ 10⁻⁶/h", "Autonomous Transit", "n ≥ 3.0×10⁶ h", "342 years", BRONZE, "Fleet Limit (Correlated)"),
        ("p ≤ 10⁻⁹/h", "SIL-4 / ASIL-D", "n ≥ 3.0×10⁹ h", "342,000 yrs", CRIMSON, "LOGICALLY INSUFFICIENT")
    ]

    for idx, (p_rate, app_name, n_hrs, n_yrs, col, status) in enumerate(tiers):
        ry = ey + idx * 48
        svg.append(f'<rect x="{p2_x+12}" y="{ry}" width="{ew}" height="42" rx="4" fill="{col}" fill-opacity="0.06" stroke="{col}" stroke-width="1"/>')
        svg.append(f'<text x="{p2_x+20}" y="{ry+16}" font-size="9" font-weight="700" fill="{col}">{p_rate} ({app_name})</text>')
        svg.append(f'<text x="{p2_x+ew-8}" y="{ry+16}" font-size="8.5" font-weight="700" fill="{INK}" text-anchor="end">{n_hrs}</text>')
        svg.append(f'<text x="{p2_x+20}" y="{ry+32}" font-size="8" fill="{MUTED}">Time: {n_yrs}</text>')
        svg.append(f'<text x="{p2_x+ew-8}" y="{ry+32}" font-size="8" font-weight="600" fill="{col}" text-anchor="end">{status}</text>')

    # Barrier Bar
    by = ey + 4 * 48 + 8
    svg.append(f'<rect x="{p2_x+12}" y="{by}" width="{ew}" height="84" rx="5" fill="{CRIMSON}" fill-opacity="0.08" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<text x="{p2_x+panel_w/2}" y="{by+18}" font-size="8.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">WHY FLEET SAMPLING FAILS AT THE TAIL:</text>')
    
    fleet_limits = [
        "1. Correlated updates &amp; identical software reset trial clock",
        "2. Safety drivers censor crashes → missing counterfactuals",
        "3. Simulators test only pre-programmed physics disturbances",
        "4. Ultra-reliability demands structural containment!"
    ]
    for idx, fl in enumerate(fleet_limits):
        svg.append(f'<text x="{p2_x+20}" y="{by+34+idx*13}" font-size="7.5" fill="{SLATE}">{fl}</text>')

    # -------------------------------------------------------------
    # PANEL 3: THE 4-PART RESIDUAL-CLAIMS REGISTER & CLOSURE GATE
    # -------------------------------------------------------------
    p3_x = start_x + 2 * (panel_w + panel_gap)
    svg.append(f'<rect x="{p3_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="{BG_WHITE}" stroke="{PURPLE}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{p3_x}" y="{panel_y}" width="{panel_w}" height="26" rx="8" fill="{PURPLE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{p3_x+panel_w/2}" y="{panel_y+17}" font-size="9.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">3. RESIDUAL-CLAIMS REGISTER</text>')

    svg.append(f'<text x="{p3_x+12}" y="{panel_y+44}" font-size="10.5" font-weight="700" fill="{INK}">Converting Caveats to Engineering Work</text>')
    svg.append(f'<text x="{p3_x+12}" y="{panel_y+58}" font-size="8.5" fill="{MUTED}">4 Mandatory Fields for Every Open Assumption</text>')

    reg_y = panel_y + 70
    reg_w = panel_w - 24

    reg_steps = [
        ("FIELD 1: UNSUPPORTED CLAIM", "Brake distance ≤ 0.49 m on concrete\nMissing Observable: Surface friction μ < 0.15", NAVY),
        ("FIELD 2: OPERATIONAL CONTAINMENT", "Structural velocity clamp v ≤ 0.9 m/s\nKinetic energy Ek reduced 83% (1089 J → 182 J)", CRIMSON),
        ("FIELD 3: ACCOUNTABLE OWNER", "Lead Drivetrain Engineer owns parameter lock;\nAccepts operational throughput penalty", PURPLE),
        ("FIELD 4: FALSIFIABLE CLOSURE TEST", "Integrate polarimetric sensor + 1 kHz slip reflex;\nValidated across N=1000 wet floor trials", PETROL)
    ]

    for idx, (f_title, f_desc, col) in enumerate(reg_steps):
        sy = reg_y + idx * 56
        svg.append(f'<rect x="{p3_x+12}" y="{sy}" width="{reg_w}" height="50" rx="4" fill="{BG_LIGHT}" stroke="{col}" stroke-width="1"/>')
        svg.append(f'<text x="{p3_x+20}" y="{sy+14}" font-size="8.5" font-weight="700" fill="{col}">{f_title}</text>')
        for d_idx, dl in enumerate(f_desc.split("\n")):
            svg.append(f'<text x="{p3_x+20}" y="{sy+27+d_idx*12}" font-size="7.5" fill="{SLATE}">{dl}</text>')

    # Resolution Gate Box at bottom of Panel 3
    gy_b = reg_y + 4 * 56 + 8
    svg.append(f'<rect x="{p3_x+12}" y="{gy_b}" width="{reg_w}" height="52" rx="4" fill="{PURPLE}" fill-opacity="0.08" stroke="{PURPLE}" stroke-width="1.1"/>')
    svg.append(f'<text x="{p3_x+panel_w/2}" y="{gy_b+15}" font-size="8.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">FALSIFICATION &amp; VERDICT GATE</text>')
    svg.append(f'<text x="{p3_x+panel_w/2}" y="{gy_b+30}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">Relocated Assumption? → REJECT / STOP</text>')
    svg.append(f'<text x="{p3_x+panel_w/2}" y="{gy_b+43}" font-size="7.5" font-weight="700" fill="{TEAL}" text-anchor="middle">Independent Evidence? → RETIRE GAP &amp; LIFT CLAMP</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/17-frontier/figures/fig17_epistemic_limits.svg", "\n".join(svg))


def run_all():
    print("Generating Chapter 17 Figures...")
    gen_ch17_closed_loop_synthesis()
    gen_ch17_epistemic_limits()

if __name__ == "__main__":
    run_all()
