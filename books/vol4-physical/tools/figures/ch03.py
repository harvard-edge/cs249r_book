"""
book/tools/figures/ch03.py
Figures for Chapter 3: Brain (What a Learned Component Gives You, and What It Costs)
Author: Physical AI Systems Team
Harvard Crimson & ETH Zurich Academic Semantic Palette
"""

import os
import subprocess
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig03_two_speed_brain():
    """
    Figure 3.1: The Two-Speed Brain Architecture & Proposal-Permission Privilege Boundary.
    Contrasts Slow System 2 Cognitive Foundation Model / VLM Intent Proposer (1-5 Hz)
    against Fast System 1 Trajectory & Reflex Enforcer (100-1000 Hz) across the strict
    hardware/software isolation boundary.
    """
    W = 940
    H = 540
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    
    # Background
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    
    # Title Banner
    svg.append(f'<text x="{W/2}" y="28" class="title">THE TWO-SPEED BRAIN ARCHITECTURE &amp; PRIVILEGE BOUNDARY</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Slow Deliberative System 2 (1–5 Hz) ⟷ Fast Real-Time System 1 (100–1000 Hz) with Proposal–Permission Gating</text>')

    # -------------------------------------------------------------
    # 1. TOP TIER: UNTRUSTED DELIBERATIVE PROPOSAL REALM (MPU / GPU)
    # -------------------------------------------------------------
    tier1_y = 66
    tier1_h = 175
    tier1_w = W - 48
    svg.append(f'<rect x="24" y="{tier1_y}" width="{tier1_w}" height="{tier1_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2" filter="url(#shadow)"/>')
    
    # Header tag for Top Tier
    svg.append(f'<rect x="24" y="{tier1_y}" width="{tier1_w}" height="24" rx="8" fill="{BLUE}" fill-opacity="0.10"/>')
    svg.append(f'<text x="36" y="{tier1_y+16}" font-size="9" font-weight="700" fill="{NAVY}">UNTRUSTED DELIBERATIVE COGNITIVE REALM · APPLICATION PROCESSOR (HOST LINUX MPU / ACCELERATOR, ~35W TDP)</text>')
    svg.append(f'<text x="{W-36}" y="{tier1_y+16}" font-size="8.5" font-weight="700" fill="{AMBER}" text-anchor="end">AUTHORITY: UNPRIVILEGED CANDIDATE PROPOSALS ONLY</text>')

    # Left Card inside Tier 1: System 2 VLM Intent Proposer
    c1_x = 38
    c1_y = tier1_y + 32
    c1_w = 415
    c1_h = 130
    svg.append(f'<rect x="{c1_x}" y="{c1_y}" width="{c1_w}" height="{c1_h}" rx="6" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{c1_x}" y="{c1_y}" width="{c1_w}" height="22" rx="6" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c1_x+10}" y="{c1_y+15}" font-size="9" font-weight="700" fill="{BLUE}">SYSTEM 2: COGNITIVE FOUNDATION MODEL / VLM</text>')
    svg.append(f'<text x="{c1_x+c1_w-10}" y="{c1_y+15}" font-size="8.5" font-weight="700" fill="{BLUE}" text-anchor="end">1 – 5 Hz · Latency: 200–1000 ms</text>')
    
    c1_lines = [
        ("Substrate:", "Multi-modal Transformer (7B–70B weights) on NPU/GPU"),
        ("Function:", "Open-world semantic grounding, object affordance &amp; goal logic"),
        ("Artifact Emitted:", "Expiring Intent Lease L_intent = (G, V_scope, t_expire)"),
        ("Failure Profile:", "Semantic hallucinations, P99 tail spikes, OOD unreliability")
    ]
    for idx, (lbl, val) in enumerate(c1_lines):
        ly = c1_y + 38 + idx * 22
        svg.append(f'<text x="{c1_x+10}" y="{ly}" font-size="8.5" font-weight="700" fill="{INK}">• {lbl}</text>')
        svg.append(f'<text x="{c1_x+98}" y="{ly}" font-size="8.5" fill="{SLATE}">{val}</text>')

    # Arrow from System 2 to System 1.5
    svg.append(f'<line x1="{c1_x+c1_w}" y1="{c1_y+c1_h/2}" x2="{c1_x+c1_w+36}" y2="{c1_y+c1_h/2}" stroke="{BRONZE}" stroke-width="2" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<rect x="{c1_x+c1_w+4}" y="{c1_y+c1_h/2-18}" width="30" height="13" rx="3" fill="{BG_WHITE}" stroke="{BRONZE}" stroke-width="0.8"/>')
    svg.append(f'<text x="{c1_x+c1_w+19}" y="{c1_y+c1_h/2-9}" font-size="7.5" font-weight="700" fill="{BRONZE}" text-anchor="middle">Lease</text>')

    # Right Card inside Tier 1: System 1.5 Trajectory Generator
    c2_x = 490
    c2_y = tier1_y + 32
    c2_w = 412
    c2_h = 130
    svg.append(f'<rect x="{c2_x}" y="{c2_y}" width="{c2_w}" height="{c2_h}" rx="6" fill="{BG_WHITE}" stroke="{BRONZE}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{c2_x}" y="{c2_y}" width="{c2_w}" height="22" rx="6" fill="{BRONZE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c2_x+10}" y="{c2_y+15}" font-size="9" font-weight="700" fill="{BRONZE}">SYSTEM 1.5: GENERATIVE TRAJECTORY PLANNER</text>')
    svg.append(f'<text x="{c2_x+c2_w-10}" y="{c2_y+15}" font-size="8.5" font-weight="700" fill="{BRONZE}" text-anchor="end">20 – 50 Hz · Horizon H=16</text>')

    c2_lines = [
        ("Substrate:", "Diffusion Policy / Action Chunking Transformer (ACT)"),
        ("Function:", "Synthesizes smooth C² continuous spline waypoints A_{t:t+H}"),
        ("Artifact Emitted:", "Candidate Action Chunk Proposal p_t = {q_k, q̇_k, τ_k}"),
        ("Seam Behavior:", "Recedes on arrival; decelerates to hold if next chunk delays")
    ]
    for idx, (lbl, val) in enumerate(c2_lines):
        ly = c2_y + 38 + idx * 22
        svg.append(f'<text x="{c2_x+10}" y="{ly}" font-size="8.5" font-weight="700" fill="{INK}">• {lbl}</text>')
        svg.append(f'<text x="{c2_x+96}" y="{ly}" font-size="8.5" fill="{SLATE}">{val}</text>')

    # -------------------------------------------------------------
    # 2. MIDDLE BOUNDARY: PROPOSAL-PERMISSION PRIVILEGE BOUNDARY
    # -------------------------------------------------------------
    b_y = 262
    svg.append(f'<line x1="24" y1="{b_y}" x2="{W-24}" y2="{b_y}" stroke="{CRIMSON}" stroke-width="2" stroke-dasharray="6,4"/>')
    
    # Boundary Badge in center
    b_badge_w = 460
    b_badge_h = 24
    b_badge_x = (W - b_badge_w) / 2
    svg.append(f'<rect x="{b_badge_x}" y="{b_y - b_badge_h/2}" width="{b_badge_w}" height="{b_badge_h}" rx="12" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1.5" filter="url(#shadow)"/>')
    svg.append(f'<circle cx="{b_badge_x+14}" cy="{b_y}" r="4" fill="{CRIMSON}"/>')
    svg.append(f'<text x="{W/2+4}" y="{b_y+4}" font-size="9" font-weight="700" fill="{CRIMSON}" text-anchor="middle">PROPOSAL–PERMISSION PRIVILEGE BOUNDARY (NO DIRECT ACTUATOR ACCESS)</text>')

    # Downward proposal arrow across boundary (Dashed)
    prop_x = 696
    svg.append(f'<line x1="{prop_x}" y1="{tier1_y+tier1_h}" x2="{prop_x}" y2="{b_y+30}" stroke="{CRIMSON}" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arr-crimson)"/>')
    svg.append(f'<rect x="{prop_x-64}" y="{b_y+6}" width="128" height="18" rx="4" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1"/>')
    svg.append(f'<text x="{prop_x}" y="{b_y+18}" font-size="8" font-weight="700" fill="{CRIMSON}" text-anchor="middle">Unverified Proposal p_t</text>')

    # -------------------------------------------------------------
    # 3. LOWER TIER: TRUSTED HARD REAL-TIME ENFORCEMENT REALM (MCU)
    # -------------------------------------------------------------
    tier2_y = 302
    tier2_h = 160
    tier2_w = W - 48
    svg.append(f'<rect x="24" y="{tier2_y}" width="{tier2_w}" height="{tier2_h}" rx="8" fill="{PETROL}" fill-opacity="0.04" stroke="{PETROL}" stroke-width="1.3" filter="url(#shadow)"/>')
    
    # Header tag for Bottom Tier
    svg.append(f'<rect x="24" y="{tier2_y}" width="{tier2_w}" height="24" rx="8" fill="{PETROL}" fill-opacity="0.15"/>')
    svg.append(f'<text x="36" y="{tier2_y+16}" font-size="9" font-weight="700" fill="{PETROL}">TRUSTED REAL-TIME SAFETY REALM · DUAL LOCK-STEP MCU (BARE-METAL / FreeRTOS, &lt; 1.5W TDP, 0 DYNAMIC MALLOC)</text>')
    svg.append(f'<text x="{W-36}" y="{tier2_y+16}" font-size="8.5" font-weight="700" fill="{PETROL}" text-anchor="end">AUTHORITY: SOLE PERMISSION TO ENERGIZE ACTUATORS</text>')

    # Card 1 in Lower Tier: Real-time Invariant & Barrier Enforcer
    e1_x = 38
    e1_y = tier2_y + 32
    e1_w = 460
    e1_h = 116
    svg.append(f'<rect x="{e1_x}" y="{e1_y}" width="{e1_w}" height="{e1_h}" rx="6" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{e1_x}" y="{e1_y}" width="{e1_w}" height="20" rx="6" fill="{PETROL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{e1_x+10}" y="{e1_y+14}" font-size="9" font-weight="700" fill="{PETROL}">SYSTEM 1: DETERMINISTIC BARRIER ENFORCER &amp; SAFETY GATE</text>')
    svg.append(f'<text x="{e1_x+e1_w-10}" y="{e1_y+14}" font-size="8.5" font-weight="700" fill="{PETROL}" text-anchor="end">1000 Hz · Jitter &lt; 5 µs</text>')

    e1_items = [
        ("Control Barrier Function:", "Solves Active-Set QP: min ||u - p_t||² s.t. ḣ(x) + α(h(x)) ≥ 0"),
        ("Dynamic Stopping Distance:", "Evaluates d_stop(v_t) = v_t·t_react + v_t²/(2 a_max) ≤ D_clear"),
        ("Temporal &amp; Lease Check:", "Rejects p_t if t &gt; t_expire or CRC/timing deadline missed"),
        ("Dual Execution Verdict:", "Passes safe command u_t = permit(p_t) OR triggers Safe-Stop ⊥")
    ]
    for idx, (lbl, val) in enumerate(e1_items):
        ly = e1_y + 34 + idx * 20
        svg.append(f'<text x="{e1_x+10}" y="{ly}" font-size="8" font-weight="700" fill="{INK}">• {lbl}</text>')
        svg.append(f'<text x="{e1_x+135}" y="{ly}" font-size="8" fill="{SLATE}">{val}</text>')

    # Card 2 in Lower Tier: Actuator Motor Drives & Power Stages
    e2_x = 520
    e2_y = tier2_y + 32
    e2_w = 382
    e2_h = 116
    svg.append(f'<rect x="{e2_x}" y="{e2_y}" width="{e2_w}" height="{e2_h}" rx="6" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{e2_x}" y="{e2_y}" width="{e2_w}" height="20" rx="6" fill="{NAVY}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{e2_x+10}" y="{e2_y+14}" font-size="9" font-weight="700" fill="{NAVY}">ACTUATION POWER STAGE &amp; PHYSICAL BODY</text>')
    svg.append(f'<text x="{e2_x+e2_w-10}" y="{e2_y+14}" font-size="8.5" font-weight="700" fill="{CRIMSON}" text-anchor="end">Irreversible State (W_t → W_{{t+1}})</text>')

    e2_items = [
        ("PWM Gate Driver Registers:", "Direct H-Bridge gate drive, L/R electrical current rise"),
        ("Electromagnetic Torque:", "Delivers mechanical torque τ = k_t · I into gearboxes"),
        ("Physical Conservation Laws:", "Momentum p=mv · Joule Heat I²R · Friction dissipation"),
        ("Irreversible Boundary:", "Once current energizes stator, mechanical motion is committed")
    ]
    for idx, (lbl, val) in enumerate(e2_items):
        ly = e2_y + 34 + idx * 20
        svg.append(f'<text x="{e2_x+10}" y="{ly}" font-size="8" font-weight="700" fill="{INK}">• {lbl}</text>')
        svg.append(f'<text x="{e2_x+140}" y="{ly}" font-size="8" fill="{SLATE}">{val}</text>')

    # Horizontal connection inside Tier 2 from Enforcer to Actuators
    svg.append(f'<line x1="{e1_x+e1_w}" y1="{e1_y+40}" x2="{e2_x}" y2="{e1_y+40}" stroke="{TEAL}" stroke-width="2.5" marker-end="url(#arr-teal)"/>')
    svg.append(f'<rect x="{e1_x+e1_w+4}" y="{e1_y+26}" width="54" height="15" rx="3" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{e1_x+e1_w+31}" y="{e1_y+37}" font-size="7.5" font-weight="700" fill="{TEAL}" text-anchor="middle">u_t (Safe)</text>')

    # Refusal / Safe-Stop E-Stop branch
    svg.append(f'<line x1="{e1_x+e1_w}" y1="{e1_y+84}" x2="{e2_x}" y2="{e1_y+84}" stroke="{CORAL}" stroke-width="2" stroke-dasharray="4,2" marker-end="url(#arr-coral)"/>')
    svg.append(f'<rect x="{e1_x+e1_w+4}" y="{e1_y+70}" width="54" height="15" rx="3" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{e1_x+e1_w+31}" y="{e1_y+81}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">⊥ (E-Stop)</text>')

    # -------------------------------------------------------------
    # 4. CLOSED-LOOP SENSORY FEEDBACK
    # -------------------------------------------------------------
    # Bottom Invariant Bar
    svg.append(f'<rect x="24" y="474" width="{tier2_w}" height="52" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="492" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">THE BEDROCK PRINCIPLE OF THE TWO-SPEED ARCHITECTURE</text>')
    svg.append(f'<text x="{W/2}" y="508" font-size="8.5" fill="{SLATE}" text-anchor="middle">High-level neural models propose rich, open-world intentions over extended horizons without physical privilege;</text>')
    svg.append(f'<text x="{W/2}" y="520" font-size="8.5" fill="{SLATE}" text-anchor="middle">Hard real-time deterministic enforcers audit every proposal at microsecond cadences, retaining exclusive permission to energize motors.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-brain/figures/fig03_two_speed_brain.svg", "\n".join(svg))

def gen_fig03_five_handoffs():
    """
    Figure 3.2: The Five Handoffs Architecture & Rejection Interfaces.
    Upper Register: Canonical 5-stage chain (Observe -> Estimate -> Intend -> Plan -> Enforce)
    with explicit Rejection Gates at every interface.
    Lower Register: Collapsed End-to-End Latent Pipeline exposing the 4 Lost Inspection Points.
    """
    W = 940
    H = 560
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    
    # Background
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    
    # Title Banner
    svg.append(f'<text x="{W/2}" y="28" class="title">THE FIVE HANDOFFS: ARCHITECTURAL CHAIN &amp; REJECTION GATES</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Modular Functional Contracts with Interface Rejection Tests vs. Collapsed End-to-End Policy</text>')

    # -------------------------------------------------------------
    # REGISTER A: CANONICAL FIVE HANDOFFS CHAIN (MODULAR & AUDITABLE)
    # -------------------------------------------------------------
    ra_y = 62
    ra_h = 240
    ra_w = W - 40
    svg.append(f'<rect x="20" y="{ra_y}" width="{ra_w}" height="{ra_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="20" y="{ra_y}" width="{ra_w}" height="22" rx="8" fill="{NAVY}" fill-opacity="0.10"/>')
    svg.append(f'<text x="32" y="{ra_y+15}" font-size="9" font-weight="700" fill="{NAVY}">PANEL A: CANONICAL MODULAR CHAIN · EXPLICIT INTERFACE CONTRACTS &amp; REJECTION GATES</text>')
    svg.append(f'<text x="{W-32}" y="{ra_y+15}" font-size="8.5" font-weight="700" fill="{TEAL}" text-anchor="end">PROPOSALS (DASHED) → PERMITTED ACTUATION (SOLID)</text>')

    stages = [
        ("1. OBSERVE", "Sensory Evidence", "Photons, IMU, Encoders\nTimestamped O_t + CRC\nHardware error flags", "RG-1: Desync / Stale\nDrop if delay &gt; 15 ms", NAVY),
        ("2. ESTIMATE", "Metric State Belief", "Spatial x̂_t ∈ SE(3)\nVelocity v_t, Covariance Σ_t\nTracked frame tree", "RG-2: Drift / Age\nReject if age &gt; 50 ms", BLUE),
        ("3. INTEND", "Semantic Goal Lease", "Target goal state G\nBounding volume V_scope\nExpiring lease t_expire", "RG-3: Expired Lease\nExpire if t &gt; t_expire", PURPLE),
        ("4. PLAN", "Trajectory Chunk", "Action chunk A_{t:t+H}\n16 waypoints, C² spline\nSmooth jerk envelope", "RG-4: Infeasibility\nReject if q̈ &gt; a_max", BRONZE),
        ("5. ENFORCE", "Real-Time Barrier", "1 kHz QP Solver h(x)≥0\nStopping check d_stop\nSole permission gate", "RG-5: Barrier Breach\nVETO → Safe-Stop ⊥", PETROL)
    ]

    card_w = 160
    gap = 18
    start_x = (W - (5 * card_w + 4 * gap)) / 2

    for i, (s_num, s_title, s_emit, s_rej, col) in enumerate(stages):
        x = start_x + i * (card_w + gap)
        y = ra_y + 32
        h = 194

        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{h}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="20" rx="6" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+14}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle">{s_num}</text>')
        
        svg.append(f'<text x="{x+card_w/2}" y="{y+32}" font-size="9.5" font-weight="700" fill="{INK}" text-anchor="middle">{s_title}</text>')
        
        # Emitted artifact lines
        cur_ly = y + 48
        for l in s_emit.split("\n"):
            svg.append(f'<text x="{x+8}" y="{cur_ly}" font-size="7.8" fill="{SLATE}">• {l}</text>')
            cur_ly += 14

        # Divider line above Rejection Gate
        svg.append(f'<line x1="{x+6}" y1="{y+98}" x2="{x+card_w-6}" y2="{y+98}" stroke="{BORDER}" stroke-width="0.8"/>')

        # Rejection Gate Box
        r_box_y = y + 106
        svg.append(f'<rect x="{x+6}" y="{r_box_y}" width="{card_w-12}" height="76" rx="4" fill="{CRIMSON}" fill-opacity="0.06" stroke="{CRIMSON}" stroke-width="0.9"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{r_box_y+13}" font-size="7.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">REJECTION GATE</text>')
        for r_idx, rl in enumerate(s_rej.split("\n")):
            svg.append(f'<text x="{x+10}" y="{r_box_y+28+r_idx*14}" font-size="7.5" font-weight="600" fill="{CRIMSON if r_idx>0 else INK}">{rl}</text>')

        # Inter-stage arrows
        if i < 4:
            ax1 = x + card_w + 1
            ax2 = ax1 + gap - 2
            ay = y + 54
            is_prop = (i >= 2)
            arrow_col = BRONZE if is_prop else BLUE
            dash = ' stroke-dasharray="4,2"' if is_prop else ''
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{arrow_col}" stroke-width="1.6"{dash} marker-end="url(#arr-{"bronze" if is_prop else "blue"})"/>')

    # Enforce stage final solid command output arrow to actuators
    enf_x = start_x + 4 * (card_w + gap) + card_w
    svg.append(f'<text x="{start_x+4*(card_w+gap)+card_w/2}" y="{ra_y+ra_h-8}" font-size="8" font-weight="700" fill="{TEAL}" text-anchor="middle">✓ ONLY ENFORCE COMMANDS ACTUATORS</text>')

    # -------------------------------------------------------------
    # REGISTER B: COLLAPSED END-TO-END LATENT PIPELINE
    # -------------------------------------------------------------
    rb_y = 316
    rb_h = 160
    rb_w = W - 40
    svg.append(f'<rect x="20" y="{rb_y}" width="{rb_w}" height="{rb_h}" rx="8" fill="{BG_LIGHT}" stroke="{AMBER}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<rect x="20" y="{rb_y}" width="{rb_w}" height="22" rx="8" fill="{AMBER}" fill-opacity="0.15"/>')
    svg.append(f'<text x="32" y="{rb_y+15}" font-size="9" font-weight="700" fill="{AMBER}">PANEL B: COLLAPSED END-TO-END POLICY · OPAQUE LATENT STATE &amp; LOST INSPECTION INTERFACES</text>')
    svg.append(f'<text x="{W-32}" y="{rb_y+15}" font-size="8.5" font-weight="700" fill="{CRIMSON}" text-anchor="end">HIGH RISK: ZERO INTERMEDIATE REJECTION GATES</text>')

    # Large Black Box Policy
    bb_x = 38
    bb_y = rb_y + 32
    bb_w = W - 76
    bb_h = 116
    svg.append(f'<rect x="{bb_x}" y="{bb_y}" width="{bb_w}" height="{bb_h}" rx="6" fill="{BG_WHITE}" stroke="{AMBER}" stroke-width="1.2"/>')
    
    # Input on left of black box
    svg.append(f'<rect x="{bb_x+10}" y="{bb_y+16}" width="140" height="84" rx="4" fill="{BG_LIGHT}" stroke="{NAVY}" stroke-width="1"/>')
    svg.append(f'<text x="{bb_x+80}" y="{bb_y+34}" font-size="8.5" font-weight="700" fill="{NAVY}" text-anchor="middle">RAW SENSORY INGEST</text>')
    svg.append(f'<text x="{bb_x+80}" y="{bb_y+52}" font-size="8" fill="{SLATE}" text-anchor="middle">Camera RGB Pixels</text>')
    svg.append(f'<text x="{bb_x+80}" y="{bb_y+68}" font-size="8" fill="{SLATE}" text-anchor="middle">&amp; Natural Language</text>')
    svg.append(f'<text x="{bb_x+80}" y="{bb_y+86}" font-size="7.5" font-weight="600" fill="{MUTED}" text-anchor="middle">Input Tensor O_t</text>')

    # Arrow into Latent Model
    svg.append(f'<line x1="{bb_x+152}" y1="{bb_y+58}" x2="{bb_x+180}" y2="{bb_y+58}" stroke="{NAVY}" stroke-width="2" marker-end="url(#arr-navy)"/>')

    # Black Box Center
    box_w = 490
    box_x = bb_x + 184
    svg.append(f'<rect x="{box_x}" y="{bb_y+12}" width="{box_w}" height="{bb_h-24}" rx="5" fill="{INK}" fill-opacity="0.04" stroke="{BORDER_DARK}" stroke-width="1" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{box_x+box_w/2}" y="{bb_y+30}" font-size="9" font-weight="700" fill="{INK}" text-anchor="middle">BLACK-BOX END-TO-END VLA / DIFFUSION POLICY (OPAQUE LATENT MAP)</text>')
    
    # 4 Lost Inspection Points
    lost_pts = [
        ("Lost Sensory Check", "No CRC / desync gate"),
        ("Lost State Audit", "No SE(3) covariance Σ_t"),
        ("Lost Lease Expiry", "No timeout t_expire"),
        ("Lost Kinematic Gate", "No jerk / singular check")
    ]
    lp_w = 110
    lp_gap = 10
    lp_start_x = box_x + (box_w - (4 * lp_w + 3 * lp_gap)) / 2
    for lp_idx, (lp_t, lp_d) in enumerate(lost_pts):
        lx = lp_start_x + lp_idx * (lp_w + lp_gap)
        ly = bb_y + 42
        svg.append(f'<rect x="{lx}" y="{ly}" width="{lp_w}" height="42" rx="4" fill="{CORAL}" fill-opacity="0.08" stroke="{CORAL}" stroke-width="0.9"/>')
        svg.append(f'<text x="{lx+lp_w/2}" y="{ly+16}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">⚠ {lp_t}</text>')
        svg.append(f'<text x="{lx+lp_w/2}" y="{ly+30}" font-size="7" fill="{SLATE}" text-anchor="middle">{lp_d}</text>')

    # Arrow out of Black Box
    svg.append(f'<line x1="{box_x+box_w+2}" y1="{bb_y+58}" x2="{box_x+box_w+26}" y2="{bb_y+58}" stroke="{CRIMSON}" stroke-width="2" stroke-dasharray="4,2" marker-end="url(#arr-crimson)"/>')

    # Output on right
    out_x = box_x + box_w + 30
    out_w = bb_w - (box_x + box_w + 30 - bb_x) - 10
    svg.append(f'<rect x="{out_x}" y="{bb_y+16}" width="{out_w}" height="84" rx="4" fill="{CRIMSON}" fill-opacity="0.08" stroke="{CRIMSON}" stroke-width="1"/>')
    svg.append(f'<text x="{out_x+out_w/2}" y="{bb_y+34}" font-size="8" font-weight="700" fill="{CRIMSON}" text-anchor="middle">UNAUDITED MOTOR PROPOSAL</text>')
    svg.append(f'<text x="{out_x+out_w/2}" y="{bb_y+50}" font-size="7.5" fill="{SLATE}" text-anchor="middle">Direct Joint Torques / Voltages</text>')
    svg.append(f'<text x="{out_x+out_w/2}" y="{bb_y+66}" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">Enforcer Must Catch ALL Errors</text>')
    svg.append(f'<text x="{out_x+out_w/2}" y="{bb_y+82}" font-size="7" fill="{MUTED}" text-anchor="middle">At 1 kHz without context</text>')

    # -------------------------------------------------------------
    # BOTTOM INVARIANT BAR
    # -------------------------------------------------------------
    svg.append(f'<rect x="20" y="488" width="{ra_w}" height="56" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="506" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">ARCHITECTURAL TAKEAWAY: COLLAPSING STAGES CONCEALS BUT DOES NOT ELIMINATE HAZARDS</text>')
    svg.append(f'<text x="{W/2}" y="522" font-size="8.5" fill="{SLATE}" text-anchor="middle">Modular contracts allow each boundary to refuse invalid inputs before they consume compute or threaten physical safety.</text>')
    svg.append(f'<text x="{W/2}" y="534" font-size="8.5" fill="{SLATE}" text-anchor="middle">An end-to-end model transfers the entire burden of sensory validation, state estimation, and lease audit onto low-level reflex filters.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-brain/figures/fig03_five_handoffs.svg", "\n".join(svg))

def run_all():
    gen_fig03_two_speed_brain()
    gen_fig03_five_handoffs()

if __name__ == "__main__":
    run_all()

def gen_fig03_vla_architecture():
    """
    Figure 3.3: Vision-Language-Action (VLA) Model Architecture & Action Tokenization Pipeline.
    Details multimodal ingestion, visual patch projection, autoregressive transformer backbone,
    discrete action binning vs. continuous diffusion action chunking, and edge hardware execution.
    """
    W = 960
    H = 560
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="8" stroke="{BORDER}" stroke-width="1"/>')

    # Header
    svg.append(f'<text x="{W/2}" y="26" class="title">VISION-LANGUAGE-ACTION (VLA) ARCHITECTURE &amp; PROPOSAL PIPELINE</text>')
    svg.append(f'<text x="{W/2}" y="42" class="subtitle">Multimodal Ingestion · Visual Patch Tokenization · Autoregressive Transformer Backbone · Action Tokenization Paradigms</text>')

    # =========================================================================
    # STAGE 1: MULTIMODAL SENSORY INGESTION & TOKENIZATION
    # =========================================================================
    s1_x = 24
    s1_y = 60
    s1_w = 210
    s1_h = 420
    svg.append(f'<rect x="{s1_x}" y="{s1_y}" width="{s1_w}" height="{s1_h}" rx="6" fill="{BG_LIGHT}" stroke="{BLUE}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{s1_x}" y="{s1_y}" width="{s1_w}" height="24" rx="6" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{s1_x+10}" y="{s1_y+16}" font-size="9.5" font-weight="700" fill="{BLUE}">1. MULTIMODAL INGESTION</text>')

    # 1.1 Overhead Camera Card
    svg.append(f'<rect x="{s1_x+10}" y="{s1_y+34}" width="{s1_w-20}" height="75" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+50}" font-size="9" font-weight="700" fill="{INK}">Overhead RGB Camera</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+64}" font-size="8" fill="{SLATE}">224 x 224 x 3 @ 30 Hz</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+78}" font-size="8" fill="{MUTED}">ViT Patch: 14x14 -&gt; 256 tokens</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+92}" font-size="7.5" font-weight="600" fill="{BLUE}">Linear Projection z_vis</text>')

    # 1.2 Wrist Camera Card
    svg.append(f'<rect x="{s1_x+10}" y="{s1_y+118}" width="{s1_w-20}" height="75" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+134}" font-size="9" font-weight="700" fill="{INK}">Wrist RGB Camera</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+148}" font-size="8" fill="{SLATE}">224 x 224 x 3 (Gripper View)</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+162}" font-size="8" fill="{MUTED}">Fine-contact visual alignment</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+176}" font-size="7.5" font-weight="600" fill="{BLUE}">Linear Projection z_wrist</text>')

    # 1.3 Natural Language Instruction
    svg.append(f'<rect x="{s1_x+10}" y="{s1_y+202}" width="{s1_w-20}" height="80" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+218}" font-size="9" font-weight="700" fill="{PURPLE}">Task Instruction (Text)</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+232}" font-size="8" fill="{SLATE}">"pick red cup, place on tray"</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+246}" font-size="8" fill="{MUTED}">BPE Tokenizer / SentencePiece</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+260}" font-size="7.5" font-weight="600" fill="{PURPLE}">Text Tokens: [t_1, t_2, ..., t_L]</text>')

    # 1.4 Proprioceptive State
    svg.append(f'<rect x="{s1_x+10}" y="{s1_y+290}" width="{s1_w-20}" height="70" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+306}" font-size="9" font-weight="700" fill="{TEAL}">Proprioceptive State</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+320}" font-size="8" fill="{SLATE}">Joint Angles q, Velocities q_dot</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+334}" font-size="8" fill="{MUTED}">Gripper state g in [0, 1]</text>')
    svg.append(f'<text x="{s1_x+18}" y="{s1_y+348}" font-size="7.5" font-weight="600" fill="{TEAL}">State Token: z_state</text>')

    # =========================================================================
    # STAGE 2: AUTOREGRESSIVE MULTIMODAL TRANSFORMER BACKBONE
    # =========================================================================
    s2_x = 260
    s2_y = 60
    s2_w = 320
    s2_h = 420
    svg.append(f'<rect x="{s2_x}" y="{s2_y}" width="{s2_w}" height="{s2_h}" rx="6" fill="{BG_LIGHT}" stroke="{AMBER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{s2_x}" y="{s2_y}" width="{s2_w}" height="24" rx="6" fill="{AMBER}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s2_x+10}" y="{s2_y+16}" font-size="9.5" font-weight="700" fill="{AMBER}">2. VLA TRANSFORMER BACKBONE (7B–55B)</text>')

    # Transformer Details Card
    svg.append(f'<rect x="{s2_x+12}" y="{s2_y+34}" width="{s2_w-24}" height="140" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+50}" font-size="9" font-weight="700" fill="{INK}">Multimodal Fusion &amp; Self-Attention</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+66}" font-size="8" fill="{SLATE}">Concatenated Tokens: [z_vis | z_wrist | z_text | z_state]</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+80}" font-size="8" fill="{SLATE}">32–80 Transformer Layers · Multi-Head Attention</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+94}" font-size="8" fill="{MUTED}">Web-scale pretraining + Robot Trajectory co-fine-tuning</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+108}" font-size="8" fill="{MUTED}">Exemplars: OpenVLA (7B), RT-2 (55B), PaLM-E (562B)</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+124}" font-size="7.5" font-weight="700" fill="{AMBER}">Memory Bound: Streams ~7 GB weights per action step</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+138}" font-size="7.5" fill="{MUTED}">Inference Latency: 150–350 ms on Edge NPU (3–5 Hz)</text>')

    # Edge Silicon Execution Card
    svg.append(f'<rect x="{s2_x+12}" y="{s2_y+184}" width="{s2_w-24}" height="110" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+200}" font-size="9" font-weight="700" fill="{NAVY}">Edge Silicon Resource Footprint</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+216}" font-size="8" fill="{SLATE}">Precision: INT4 / FP8 Quantized Weights</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+230}" font-size="8" fill="{SLATE}">Memory: 5.6 GB DRAM (Weights) + 1.2 GB KV Cache</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+244}" font-size="8" fill="{SLATE}">Bus Bandwidth: 204.8 GB/s LPDDR5 Memory Wall</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+258}" font-size="8" fill="{CRIMSON}">Thermal Draw: 35W–50W sustained on SoC</text>')
    svg.append(f'<text x="{s2_x+20}" y="{s2_y+272}" font-size="7.5" font-weight="600" fill="{NAVY}">Hardware: NVIDIA Jetson AGX Orin / Apple Silicon</text>')

    # Unprivileged Banner
    svg.append(f'<rect x="{s2_x+12}" y="{s2_y+304}" width="{s2_w-24}" height="55" rx="4" fill="{AMBER}" fill-opacity="0.10" stroke="{AMBER}" stroke-width="1"/>')
    svg.append(f'<text x="{s2_x+s2_w/2}" y="{s2_y+322}" font-size="8.5" font-weight="700" fill="{AMBER}" text-anchor="middle">UNPRIVILEGED CANDIDATE PROPOSALS</text>')
    svg.append(f'<text x="{s2_x+s2_w/2}" y="{s2_y+336}" font-size="7.5" fill="{SLATE}" text-anchor="middle">Outputs are stochastic samples a_hat ~ pi_theta(o_t)</text>')
    svg.append(f'<text x="{s2_x+s2_w/2}" y="{s2_y+348}" font-size="7.5" fill="{CRIMSON}" text-anchor="middle">ZERO direct actuator register write authority</text>')

    # Arrow S1 -> S2
    svg.append(f'<line x1="{s1_x+s1_w}" y1="{s2_y+100}" x2="{s2_x}" y2="{s2_y+100}" stroke="{BLUE}" stroke-width="1.8" marker-end="url(#arr-blue)"/>')
    svg.append(f'<line x1="{s1_x+s1_w}" y1="{s2_y+240}" x2="{s2_x}" y2="{s2_y+240}" stroke="{PURPLE}" stroke-width="1.8" marker-end="url(#arr-purple)"/>')

    # =========================================================================
    # STAGE 3: ACTION TOKENIZATION PARADIGMS
    # =========================================================================
    s3_x = 605
    s3_y = 60
    s3_w = 330
    s3_h = 420
    svg.append(f'<rect x="{s3_x}" y="{s3_y}" width="{s3_w}" height="{s3_h}" rx="6" fill="{BG_LIGHT}" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{s3_x}" y="{s3_y}" width="{s3_w}" height="24" rx="6" fill="{TEAL}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s3_x+10}" y="{s3_y+16}" font-size="9.5" font-weight="700" fill="{TEAL}">3. ACTION TOKENIZATION &amp; EMISSION</text>')

    # Paradigm A: Discrete Bins
    svg.append(f'<rect x="{s3_x+12}" y="{s3_y+34}" width="{s3_w-24}" height="100" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+50}" font-size="9" font-weight="700" fill="{BLUE}">Paradigm A: Discrete Action Binning (RT-2 / OpenVLA)</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+66}" font-size="8" fill="{SLATE}">Quantize 7-DoF action space into K=256 uniform bins</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+80}" font-size="8" fill="{SLATE}">Delta coords: [dx, dy, dz, droll, dpitch, dyaw, grip]</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+94}" font-size="8" fill="{MUTED}">Emits 7 text vocabulary tokens sequentially</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+108}" font-size="7.5" fill="{MUTED}">Single-step delta proposal: a_t in R^7 @ 3–5 Hz</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+122}" font-size="7.5" font-weight="600" fill="{BLUE}">Artifact: Discrete Waypoint / Single-Step Goal</text>')

    # Paradigm B: Continuous Action Chunking & Diffusion
    svg.append(f'<rect x="{s3_x+12}" y="{s3_y+144}" width="{s3_w-24}" height="105" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+160}" font-size="9" font-weight="700" fill="{TEAL}">Paradigm B: Diffusion Action Chunking (Octo / ACT)</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+176}" font-size="8" fill="{SLATE}">Transformer features condition lightweight diffusion head</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+190}" font-size="8" fill="{SLATE}">Predicts continuous action chunk A_t:t+H (H = 16 steps)</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+204}" font-size="8" fill="{MUTED}">Temporal ensembling eliminates single-step Markov error</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+218}" font-size="7.5" fill="{MUTED}">Continuous trajectory horizon: 320 ms preview window</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+232}" font-size="7.5" font-weight="600" fill="{TEAL}">Artifact: C^2 Continuous Spline / Action Chunk</text>')

    # Downstream Nervous Enforcer Integration Card
    svg.append(f'<rect x="{s3_x+12}" y="{s3_y+258}" width="{s3_w-24}" height="105" rx="4" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+274}" font-size="9" font-weight="700" fill="{PETROL}">4. Downstream Nervous System Gating (MCU @ 1 kHz)</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+290}" font-size="8" fill="{SLATE}">RPMSG / PCIe DMA across isolation boundary</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+304}" font-size="8" fill="{SLATE}">Control Barrier Function (CBF-QP): h_dot(x,u) + gamma h(x) &gt;= 0</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+318}" font-size="8" fill="{SLATE}">Stopping distance invariant: d_stop(v) &lt;= d_clearance</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+334}" font-size="8" font-weight="700" fill="{PETROL}">Permitted Motor Commits u_t* Latch at 1000 Hz</text>')
    svg.append(f'<text x="{s3_x+20}" y="{s3_y+348}" font-size="7.5" fill="{MUTED}">Fallback: Safe Stop 1 (SS1) / Zero-torque if lease expires</text>')

    # Arrow S2 -> S3
    svg.append(f'<line x1="{s2_x+s2_w}" y1="{s3_y+85}" x2="{s3_x}" y2="{s3_y+85}" stroke="{AMBER}" stroke-width="1.8" marker-end="url(#arr-amber)"/>')
    svg.append(f'<line x1="{s2_x+s2_w}" y1="{s3_y+195}" x2="{s3_x}" y2="{s3_y+195}" stroke="{AMBER}" stroke-width="1.8" marker-end="url(#arr-amber)"/>')

    # Bottom Summary Ribbon
    svg.append(f'<rect x="24" y="495" width="{W-48}" height="45" rx="5" fill="{NAVY}" fill-opacity="0.08" stroke="{NAVY}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="514" font-size="9" font-weight="700" fill="{NAVY}" text-anchor="middle">THE PHYSICAL AI DILEMMA: HIGH CAPACITY (7B–55B) INFERENCE RUNS SLOW (3–5 Hz); PHYSICAL DYNAMICS REQUIRE FAST (1 kHz) SUPERVISION</text>')
    svg.append(f'<text x="{W/2}" y="528" font-size="8" fill="{SLATE}" text-anchor="middle">VLAs operate strictly as unauthenticated proposal engines across the slow cognitive tier, gated by bare-metal MCU barrier enforcers.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-brain/figures/fig03_vla_architecture.svg", "\n".join(svg))


def run_all():
    gen_fig03_two_speed_brain()
    gen_fig03_vla_architecture()

if __name__ == "__main__":
    run_all()
