"""
book/tools/figures/ch01.py
Figures for Chapter 1: Boundary (Introduction to Physical AI Systems)
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

def gen_fig01_causal_loop():
    """
    Figure 1.3: The Causal Closed Loop of Physical AI vs. Open-Loop Digital ML.
    Pristine textbook-grade diagram: clean routing, generous padding, zero line collisions.
    """
    W = 920
    H = 490
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="8" stroke="{BORDER}" stroke-width="1"/>')

    # Title & Subtitle
    svg.append(f'<text x="{W/2}" y="26" class="title">THE CAUSAL CLOSED LOOP OF PHYSICAL AI</text>')
    svg.append(f'<text x="{W/2}" y="42" class="subtitle">From Digital Policy Deliberation across the Causal Boundary to Physical Action and Endogenous Observation Shift</text>')

    # =========================================================================
    # TOP TIER: UNTRUSTED COMPUTATIONAL REALM (IDEMPOTENT DIGITAL SUBSTRATE)
    # =========================================================================
    t_y = 58
    t_h = 132
    svg.append(f'<rect x="24" y="{t_y}" width="{W-48}" height="{t_h}" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="24" y="{t_y}" width="{W-48}" height="22" rx="6" fill="{BLUE}" fill-opacity="0.10"/>')
    svg.append(f'<text x="36" y="{t_y+15}" font-size="9" font-weight="700" fill="{BLUE}">UNTRUSTED COMPUTATIONAL REALM · APPLICATION PROCESSOR (HOST LINUX MPU / ACCELERATOR)</text>')
    svg.append(f'<text x="{W-36}" y="{t_y+15}" font-size="8" font-weight="600" fill="{MUTED}" text-anchor="end">IDEMPOTENT · REVERSIBLE · SOFTWARE CHECKPOINTS</text>')

    # Box 1: Sensory Ingestion
    bx1_x = 42
    bx_w = 260
    bx_h = 86
    bx_y = t_y + 32
    svg.append(f'<rect x="{bx1_x}" y="{bx_y}" width="{bx_w}" height="{bx_h}" rx="5" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="1.2"/>')
    svg.append(f'<text x="{bx1_x+10}" y="{bx_y+17}" font-size="9.5" font-weight="700" fill="{BLUE}">1. Sensory Ingestion</text>')
    svg.append(f'<text x="{bx1_x+10}" y="{bx_y+33}" font-size="8.5" fill="{SLATE}">Multi-Camera / IMU / Encoders</text>')
    svg.append(f'<text x="{bx1_x+10}" y="{bx_y+49}" font-size="8.5" fill="{SLATE}">Observation Vector: o_t in R^d</text>')
    svg.append(f'<text x="{bx1_x+10}" y="{bx_y+65}" font-size="8" fill="{MUTED}">Spatial Tokenization &amp; Calibration</text>')

    # Arrow 1 -> 2
    arr1_x1 = bx1_x + bx_w
    arr1_x2 = bx1_x + bx_w + 35
    arr_y = bx_y + bx_h/2
    svg.append(f'<line x1="{arr1_x1}" y1="{arr_y}" x2="{arr1_x2}" y2="{arr_y}" stroke="{BLUE}" stroke-width="1.5" marker-end="url(#arr-blue)"/>')
    svg.append(f'<text x="{(arr1_x1+arr1_x2)/2}" y="{arr_y-8}" font-size="7.5" fill="{MUTED}" text-anchor="middle">tokens</text>')

    # Box 2: Neural Policy Deliberation
    bx2_x = arr1_x2
    svg.append(f'<rect x="{bx2_x}" y="{bx_y}" width="{bx_w}" height="{bx_h}" rx="5" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="1.2"/>')
    svg.append(f'<text x="{bx2_x+10}" y="{bx_y+17}" font-size="9.5" font-weight="700" fill="{BLUE}">2. Neural Policy Deliberation</text>')
    svg.append(f'<text x="{bx2_x+10}" y="{bx_y+33}" font-size="8.5" fill="{SLATE}">VLA / Diffusion Policy / ACT</text>')
    svg.append(f'<text x="{bx2_x+10}" y="{bx_y+49}" font-size="8.5" fill="{SLATE}">Stochastic Candidate: a_hat ~ pi_theta(o_t)</text>')
    svg.append(f'<text x="{bx2_x+10}" y="{bx_y+65}" font-size="8" fill="{MUTED}">Unprivileged Candidate Proposal</text>')

    # Arrow 2 -> 3
    arr2_x1 = bx2_x + bx_w
    arr2_x2 = bx2_x + bx_w + 35
    svg.append(f'<line x1="{arr2_x1}" y1="{arr_y}" x2="{arr2_x2}" y2="{arr_y}" stroke="{AMBER}" stroke-width="1.5" marker-end="url(#arr-amber)"/>')
    svg.append(f'<text x="{(arr2_x1+arr2_x2)/2}" y="{arr_y-8}" font-size="7.5" fill="{MUTED}" text-anchor="middle">proposal</text>')

    # Box 3: Real-Time Safety Filter
    bx3_x = arr2_x2
    svg.append(f'<rect x="{bx3_x}" y="{bx_y}" width="{bx_w}" height="{bx_h}" rx="5" fill="{BG_WHITE}" stroke="{AMBER}" stroke-width="1.4"/>')
    svg.append(f'<text x="{bx3_x+10}" y="{bx_y+17}" font-size="9.5" font-weight="700" fill="{AMBER}">3. Deterministic Safety Referee</text>')
    svg.append(f'<text x="{bx3_x+10}" y="{bx_y+33}" font-size="8.5" fill="{SLATE}">Control Barrier Function (CBF-QP)</text>')
    svg.append(f'<text x="{bx3_x+10}" y="{bx_y+49}" font-size="8.5" fill="{SLATE}">Forward Invariance: h_dot + gamma*h &gt;= 0</text>')
    svg.append(f'<text x="{bx3_x+10}" y="{bx_y+65}" font-size="8" font-weight="600" fill="{PETROL}">Permission / Fallback Safe Stop</text>')

    # =========================================================================
    # THE CAUSAL BOUNDARY BANNER (ACTUATOR REGISTER LATCH)
    # =========================================================================
    bnd_y = 205
    svg.append(f'<line x1="24" y1="{bnd_y+13}" x2="{W-24}" y2="{bnd_y+13}" stroke="{CRIMSON}" stroke-width="1.5" stroke-dasharray="6,4"/>')
    svg.append(f'<rect x="{W/2-230}" y="{bnd_y}" width="460" height="26" rx="5" fill="{CRIMSON}" filter="url(#shadow)"/>')
    svg.append(f'<text x="{W/2}" y="{bnd_y+17}" font-size="9.5" font-weight="800" fill="#FFFFFF" text-anchor="middle" letter-spacing="0.8px">THE CAUSAL BOUNDARY: MEMORY-MAPPED ACTUATOR REGISTER WRITE</text>')

    # Vertical Crossing Arrow
    svg.append(f'<line x1="{bx3_x+bx_w/2}" y1="{bx_y+bx_h}" x2="{bx3_x+bx_w/2}" y2="{bnd_y}" stroke="{AMBER}" stroke-width="1.8"/>')
    svg.append(f'<line x1="{bx3_x+bx_w/2}" y1="{bnd_y+26}" x2="{bx3_x+bx_w/2}" y2="{bnd_y+46}" stroke="{CRIMSON}" stroke-width="2" marker-end="url(#arr-crimson)"/>')
    svg.append(f'<text x="{bx3_x+bx_w/2+12}" y="{bnd_y+40}" font-size="8" font-weight="700" fill="{CRIMSON}">u_t (committed)</text>')

    # =========================================================================
    # BOTTOM TIER: PHYSICAL HARDWARE & ENVIRONMENT (THERMODYNAMIC WORK)
    # =========================================================================
    bot_y = 252
    bot_h = 132
    svg.append(f'<rect x="24" y="{bot_y}" width="{W-48}" height="{bot_h}" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="24" y="{bot_y}" width="{W-48}" height="22" rx="6" fill="{CRIMSON}" fill-opacity="0.10"/>')
    svg.append(f'<text x="36" y="{bot_y+15}" font-size="9" font-weight="700" fill="{CRIMSON}">PHYSICAL REALITY · DYNAMICS &amp; THERMODYNAMICS (ACTUATOR INVERTERS, MASS, ENVIRONMENT)</text>')
    svg.append(f'<text x="{W-36}" y="{bot_y+15}" font-size="8" font-weight="600" fill="{CRIMSON}" text-anchor="end">NON-REVERSIBLE · THERMODYNAMIC WORK · JOULE HEAT</text>')

    bot_w = 260
    bot_card_h = 86
    bot_card_y = bot_y + 32

    # Box 4: Power Inverter
    svg.append(f'<rect x="{bx1_x}" y="{bot_card_y}" width="{bot_w}" height="{bot_card_h}" rx="5" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<text x="{bx1_x+10}" y="{bot_card_y+17}" font-size="9.5" font-weight="700" fill="{CRIMSON}">4. Power Inverter &amp; Windings</text>')
    svg.append(f'<text x="{bx1_x+10}" y="{bot_card_y+33}" font-size="8.5" fill="{SLATE}">Gate Driver PWM Switching</text>')
    svg.append(f'<text x="{bx1_x+10}" y="{bot_card_y+49}" font-size="8.5" fill="{SLATE}">Joule Heat: Q = I^2 R Delta t</text>')
    svg.append(f'<text x="{bx1_x+10}" y="{bot_card_y+65}" font-size="8" fill="{MUTED}">Magnetic Flux &amp; Lorentz Force</text>')

    # Arrow 4 -> 5
    svg.append(f'<line x1="{bx1_x+bot_w}" y1="{bot_card_y+bot_card_h/2}" x2="{bx2_x}" y2="{bot_card_y+bot_card_h/2}" stroke="{CRIMSON}" stroke-width="1.5" marker-end="url(#arr-crimson)"/>')
    svg.append(f'<text x="{(bx1_x+bot_w+bx2_x)/2}" y="{bot_card_y+bot_card_h/2-8}" font-size="7.5" fill="{MUTED}" text-anchor="middle">torque tau</text>')

    # Box 5: Mechanical Dynamics
    svg.append(f'<rect x="{bx2_x}" y="{bot_card_y}" width="{bot_w}" height="{bot_card_h}" rx="5" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<text x="{bx2_x+10}" y="{bot_card_y+17}" font-size="9.5" font-weight="700" fill="{CRIMSON}">5. Mechanical Dynamics</text>')
    svg.append(f'<text x="{bx2_x+10}" y="{bot_card_y+33}" font-size="8.5" fill="{SLATE}">Torque tau = M(q)q_ddot + C(q,q_dot)</text>')
    svg.append(f'<text x="{bx2_x+10}" y="{bot_card_y+49}" font-size="8.5" fill="{SLATE}">Kinetic Momentum: p = mv</text>')
    svg.append(f'<text x="{bx2_x+10}" y="{bot_card_y+65}" font-size="8" fill="{MUTED}">Reflected Inertia: N^2 J_rotor</text>')

    # Arrow 5 -> 6
    svg.append(f'<line x1="{bx2_x+bot_w}" y1="{bot_card_y+bot_card_h/2}" x2="{bx3_x}" y2="{bot_card_y+bot_card_h/2}" stroke="{CRIMSON}" stroke-width="1.5" marker-end="url(#arr-crimson)"/>')
    svg.append(f'<text x="{(bx2_x+bot_w+bx3_x)/2}" y="{bot_card_y+bot_card_h/2-8}" font-size="7.5" fill="{MUTED}" text-anchor="middle">work F dx</text>')

    # Box 6: Physical Environment
    svg.append(f'<rect x="{bx3_x}" y="{bot_card_y}" width="{bot_w}" height="{bot_card_h}" rx="5" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1.2"/>')
    svg.append(f'<text x="{bx3_x+10}" y="{bot_card_y+17}" font-size="9.5" font-weight="700" fill="{CRIMSON}">6. Physical Environment</text>')
    svg.append(f'<text x="{bx3_x+10}" y="{bot_card_y+33}" font-size="8.5" fill="{SLATE}">Environment State: W_t -&gt; W_t+1</text>')
    svg.append(f'<text x="{bx3_x+10}" y="{bot_card_y+49}" font-size="8.5" fill="{SLATE}">Delay as Distance: Delta x = int v dt</text>')
    svg.append(f'<text x="{bx3_x+10}" y="{bot_card_y+65}" font-size="8" fill="{MUTED}">Unrecoverable State Transitions</text>')

    # =========================================================================
    # WIDE OUTER FEEDBACK LOOP (ENDOGENOUS SENSORY SHIFT)
    # =========================================================================
    loop_y = bot_y + bot_h + 28
    loop_x = 36

    # Line 1: Down from Box 6
    svg.append(f'<line x1="{bx3_x+bot_w/2}" y1="{bot_card_y+bot_card_h}" x2="{bx3_x+bot_w/2}" y2="{loop_y}" stroke="{PETROL}" stroke-width="2"/>')
    # Line 2: Across to the right of banner
    svg.append(f'<line x1="{bx3_x+bot_w/2}" y1="{loop_y}" x2="{W/2+250}" y2="{loop_y}" stroke="{PETROL}" stroke-width="2"/>')
    # Line 3: Across from left of banner to left margin
    svg.append(f'<line x1="{W/2-250}" y1="{loop_y}" x2="{loop_x}" y2="{loop_y}" stroke="{PETROL}" stroke-width="2"/>')
    # Line 4: Up the left margin
    svg.append(f'<line x1="{loop_x}" y1="{loop_y}" x2="{loop_x}" y2="{bx_y+bx_h/2}" stroke="{PETROL}" stroke-width="2"/>')
    # Line 5: Into Box 1
    svg.append(f'<line x1="{loop_x}" y1="{bx_y+bx_h/2}" x2="{bx1_x}" y2="{bx_y+bx_h/2}" stroke="{PETROL}" stroke-width="2" marker-end="url(#arr-petrol)"/>')

    # Centered Feedback Badge
    svg.append(f'<rect x="{W/2-245}" y="{loop_y-11}" width="490" height="22" rx="4" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{W/2}" y="{loop_y+4}" font-size="8.5" font-weight="700" fill="{PETROL}" text-anchor="middle">ENDOGENOUS SENSORY FEEDBACK: o_t+1 ~ P(O | s_t+1, W_t+1) [Actions reshape future observations]</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/01-boundary/figures/fig01_causal_loop.svg", "\n".join(svg))


def gen_fig01_scope_venn():
    """
    Figure 1.6: The Physical AI Scope Test & Dual-Brain Systems Convergence.
    Framed firmly from the Machine Learning Systems perspective:
    Circle 1: Machine Learning Systems (High-Capacity Stochastic Models, VLMs, MPU/NPU)
    Circle 2: Real-Time Embedded Systems (Deterministic Execution, Zero-Alloc SRAM, MCU)
    Circle 3: Physical Mechanics & Dynamics (Inertia, Momentum, Thermal Limits, Causal Loop)
    Center: Physical AI Systems & The Dual-Brain Propose-Permit Bridge (Arduino UNO Q)
    """
    W = 900
    H = 550
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="8" stroke="{BORDER}" stroke-width="1"/>')

    # Header
    svg.append(f'<text x="{W/2}" y="26" class="title">THE PHYSICAL AI SYSTEMS SCOPE: THREE-PILLAR CONVERGENCE</text>')
    svg.append(f'<text x="{W/2}" y="42" class="subtitle">An ML Systems Perspective: High-Capacity Learned Models · Real-Time Embedded Silicon · Physical Dynamics</text>')

    # 3 Circles: Centers and Radii
    R = 140
    cx1 = 360
    cy1 = 215

    cx2 = 540
    cy2 = 215

    cx3 = 450
    cy3 = 330

    # Draw 3 translucent circles
    svg.append(f'<circle cx="{cx1}" cy="{cy1}" r="{R}" fill="{BLUE}" fill-opacity="0.08" stroke="{BLUE}" stroke-width="1.6"/>')
    svg.append(f'<circle cx="{cx2}" cy="{cy2}" r="{R}" fill="{CRIMSON}" fill-opacity="0.08" stroke="{CRIMSON}" stroke-width="1.6"/>')
    svg.append(f'<circle cx="{cx3}" cy="{cy3}" r="{R}" fill="{PETROL}" fill-opacity="0.08" stroke="{PETROL}" stroke-width="1.6"/>')

    # =========================================================================
    # SET HEADERS (OUTER LABELS)
    # =========================================================================
    # Top Left Set: ML Systems
    svg.append(f'<text x="200" y="62" font-size="10.5" font-weight="700" fill="{BLUE}" text-anchor="middle">MACHINE LEARNING SYSTEMS</text>')
    svg.append(f'<text x="200" y="76" font-size="8.5" fill="{SLATE}" text-anchor="middle">High-capacity stochastic policies</text>')
    svg.append(f'<text x="200" y="88" font-size="8.5" fill="{MUTED}" text-anchor="middle">VLMs · Diffusion · MPU / NPU Proposers</text>')

    # Top Right Set: Real-Time Embedded Systems
    svg.append(f'<text x="700" y="62" font-size="10.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">REAL-TIME EMBEDDED SYSTEMS</text>')
    svg.append(f'<text x="700" y="76" font-size="8.5" fill="{SLATE}" text-anchor="middle">Deterministic execution &amp; WCET</text>')
    svg.append(f'<text x="700" y="88" font-size="8.5" fill="{MUTED}" text-anchor="middle">Zero-alloc SRAM · 1 kHz MCU Referees</text>')

    # Bottom Set: Physical Dynamics & Mechanics
    svg.append(f'<text x="450" y="496" font-size="10.5" font-weight="700" fill="{PETROL}" text-anchor="middle">PHYSICAL MECHANICS &amp; THERMODYNAMICS</text>')
    svg.append(f'<text x="450" y="511" font-size="8.5" fill="{SLATE}" text-anchor="middle">Inertia (p=mv) · Non-reversible energy (I^2 R) · Closed causal loop (s_t+1 ~ P(s|s_t,a_t))</text>')

    # =========================================================================
    # PURE SINGLE-SET OUTER REGIONS
    # =========================================================================
    # 1. Pure Digital ML (Top Left)
    svg.append(f'<text x="250" y="195" font-size="9" font-weight="700" fill="{BLUE}" text-anchor="middle">Pure Digital ML</text>')
    svg.append(f'<text x="250" y="209" font-size="8" fill="{MUTED}" text-anchor="middle">Cloud LLMs · Vision Classifiers</text>')
    svg.append(f'<text x="250" y="221" font-size="7.5" fill="{MUTED}" text-anchor="middle">(Behind glass; idempotent rollback)</text>')

    # 2. Hard Real-Time Discrete Systems (Top Right)
    svg.append(f'<text x="650" y="195" font-size="9" font-weight="700" fill="{CRIMSON}" text-anchor="middle">Discrete Embedded RTOS</text>')
    svg.append(f'<text x="650" y="209" font-size="8" fill="{MUTED}" text-anchor="middle">PLC Timers · Avionics Relays</text>')
    svg.append(f'<text x="650" y="221" font-size="7.5" fill="{MUTED}" text-anchor="middle">(No learned models; static logic)</text>')

    # 3. Passive Mechanics (Bottom)
    svg.append(f'<text x="450" y="405" font-size="9" font-weight="700" fill="{PETROL}" text-anchor="middle">Passive Mechanics</text>')
    svg.append(f'<text x="450" y="419" font-size="8" fill="{MUTED}" text-anchor="middle">Spring-Mass Dampers · Thermal Sinks</text>')

    # =========================================================================
    # TWO-SET INTERSECTION REGIONS (ADJACENT DISCIPLINES)
    # =========================================================================
    # Region A: Top Intersection (ML Systems + Embedded, No Physical Actuation)
    svg.append(f'<rect x="375" y="122" width="150" height="46" rx="4" fill="{BG_WHITE}" stroke="{PURPLE}" stroke-width="1" filter="url(#shadow)"/>')
    svg.append(f'<text x="450" y="137" font-size="8.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">Edge ML / TinyML</text>')
    svg.append(f'<text x="450" y="149" font-size="7.5" fill="{SLATE}" text-anchor="middle">Keyword Spotters · Visual Wakewords</text>')
    svg.append(f'<text x="450" y="159" font-size="7.5" fill="{MUTED}" text-anchor="middle">Advisory output; no physical loop</text>')

    # Region B: Left-Bottom Intersection (ML Systems + Physics, No Real-Time Authority)
    svg.append(f'<rect x="205" y="325" width="155" height="46" rx="4" fill="{BG_WHITE}" stroke="{AMBER}" stroke-width="1" filter="url(#shadow)"/>')
    svg.append(f'<text x="282" y="340" font-size="8.5" font-weight="700" fill="{AMBER}" text-anchor="middle">Physics-Informed ML</text>')
    svg.append(f'<text x="282" y="352" font-size="7.5" fill="{SLATE}" text-anchor="middle">Offline Surrogates · Simulators</text>')
    svg.append(f'<text x="282" y="362" font-size="7.5" fill="{MUTED}" text-anchor="middle">Synthetic telemetry; no live actuator</text>')

    # Region C: Right-Bottom Intersection (Embedded + Physics, No Learned Models)
    svg.append(f'<rect x="540" y="325" width="155" height="46" rx="4" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1.1" filter="url(#shadow)"/>')
    svg.append(f'<text x="617" y="340" font-size="8.5" font-weight="700" fill="{TEAL}" text-anchor="middle">Classical Robotics &amp; Control</text>')
    svg.append(f'<text x="617" y="352" font-size="7.5" fill="{SLATE}" text-anchor="middle">1 kHz PID · LQR · MPC Controllers</text>')
    svg.append(f'<text x="617" y="362" font-size="7.5" fill="{MUTED}" text-anchor="middle">Analytically specified transfer functions</text>')

    # =========================================================================
    # THREE-WAY INTERSECTION (PHYSICAL AI & DUAL-BRAIN SILICON CORE)
    # =========================================================================
    core_cx = 450
    core_cy = 255
    svg.append(f'<rect x="{core_cx-90}" y="{core_cy-35}" width="180" height="70" rx="6" fill="{NAVY}" stroke="{BORDER_DARK}" stroke-width="1.5" filter="url(#shadow)"/>')
    svg.append(f'<text x="{core_cx}" y="{core_cy-13}" font-size="11" font-weight="800" fill="#FFFFFF" text-anchor="middle" letter-spacing="1px">PHYSICAL AI SYSTEMS</text>')
    svg.append(f'<text x="{core_cx}" y="{core_cy+3}" font-size="8" font-weight="700" fill="#FDE047" text-anchor="middle">DUAL-BRAIN SILICON BRIDGE</text>')
    svg.append(f'<text x="{core_cx}" y="{core_cy+15}" font-size="7.5" fill="#E2E8F0" text-anchor="middle">Linux MPU (Propose) ⟷ MCU (Permit)</text>')
    svg.append(f'<text x="{core_cx}" y="{core_cy+27}" font-size="7" fill="#93C5FD" text-anchor="middle">Arduino UNO Q · Autonomous Machines</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/01-boundary/figures/fig01_scope_venn.svg", "\n".join(svg))


def run_all():
    gen_fig01_causal_loop()
    gen_fig01_scope_venn()

if __name__ == "__main__":
    run_all()
