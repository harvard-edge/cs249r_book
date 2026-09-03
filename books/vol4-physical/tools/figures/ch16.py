"""
book/tools/figures/ch16.py
Figures for Chapter 16: Release (Deciding Whether It Should Operate).
Pure-vector SVG generator using Harvard Crimson & ETH Zurich Semantic Palette.
"""

import os
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch16_cae_tree():
    """
    Figure 16.1: Claim-Argument-Evidence (CAE) Goal Structuring Notation (GSN)
    Safety Case Tree for the Tactile Manipulation Workstation.
    Illustrates top claim, sub-claims, warrants, empirical evidence,
    red-team defeater attack (oil mist slip failure), and compensatory
    conditional release branches.
    """
    W = 960
    H = 620
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    
    # Background card
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    
    # Title & Subtitle
    svg.append(f'<text x="{W/2}" y="28" class="title">CLAIM-ARGUMENT-EVIDENCE (CAE) GOAL STRUCTURING TREE FOR EMBODIED RELEASE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Autonomous Tactile Workstation: High-Level Safety Claim ⟷ Evidence Grounding ⟷ Defeater Analysis ⟷ Conditional Release</text>')

    # -------------------------------------------------------------
    # 1. TOP CLAIM C1 & CONTEXT X_ODD (y: 60 - 130)
    # -------------------------------------------------------------
    c1_x, c1_y, c1_w, c1_h = 40, 62, 540, 68
    svg.append(f'<rect x="{c1_x}" y="{c1_y}" width="{c1_w}" height="{c1_h}" rx="6" fill="{NAVY}" fill-opacity="0.06" stroke="{NAVY}" stroke-width="1.6"/>')
    svg.append(f'<rect x="{c1_x}" y="{c1_y}" width="{c1_w}" height="22" rx="6" fill="{NAVY}" fill-opacity="0.14"/>')
    svg.append(f'<text x="{c1_x+12}" y="{c1_y+15}" font-size="10.5" font-weight="700" fill="{NAVY}">TOP OPERATIONAL SAFETY CLAIM (C₁)</text>')
    svg.append(f'<text x="{c1_x+c1_w-12}" y="{c1_y+15}" font-size="9" font-weight="700" fill="{NAVY}" text-anchor="end">Falsifiable Physical Boundary</text>')
    svg.append(f'<text x="{c1_x+12}" y="{c1_y+38}" font-size="10" font-weight="600" fill="{INK}">"Manipulator never exerts normal force F_norm &gt; 50 N or shear force F_shear &gt; 20 N on unexpected obstacle</text>')
    svg.append(f'<text x="{c1_x+12}" y="{c1_y+53}" font-size="10" font-weight="600" fill="{INK}">or human operator across any operational state, with emergency stopping time t_stop ≤ 40 ms."</text>')

    # Context Box (ODD)
    ctx_x, ctx_y, ctx_w, ctx_h = 600, 62, 320, 68
    svg.append(f'<rect x="{ctx_x}" y="{ctx_y}" width="{ctx_w}" height="{ctx_h}" rx="6" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1.2" stroke-dasharray="4,3"/>')
    svg.append(f'<rect x="{ctx_x}" y="{ctx_y}" width="{ctx_w}" height="22" rx="6" fill="{MUTED}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{ctx_x+10}" y="{ctx_y+15}" font-size="10" font-weight="700" fill="{INK}">CONTEXT: OPERATIONAL DESIGN DOMAIN (X_ODD)</text>')
    svg.append(f'<text x="{ctx_x+10}" y="{ctx_y+36}" font-size="8.5" fill="{SLATE}">• Reach r ≤ 1.2 m · Max velocity v_max = 0.40 m/s</text>')
    svg.append(f'<text x="{ctx_x+10}" y="{ctx_y+49}" font-size="8.5" fill="{SLATE}">• Payload mass m ≤ 2.0 kg (nominal) · Dry friction μ ≥ 0.35</text>')
    svg.append(f'<text x="{ctx_x+10}" y="{ctx_y+62}" font-size="8.5" fill="{SLATE}">• Temp: 15°C–40°C · Light: 300–2000 lx · 1 co-present worker</text>')

    # Line connecting Context to Top Claim
    svg.append(f'<line x1="{ctx_x}" y1="{ctx_y+34}" x2="{c1_x+c1_w}" y2="{ctx_y+34}" stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="3,3"/>')

    # Main connector bar from C1 down to Sub-claims
    svg.append(f'<line x1="{c1_x+c1_w/2}" y1="{c1_y+c1_h}" x2="{c1_x+c1_w/2}" y2="150" stroke="{NAVY}" stroke-width="1.5"/>')
    svg.append(f'<line x1="165" y1="150" x2="795" y2="150" stroke="{NAVY}" stroke-width="1.5"/>')

    # -------------------------------------------------------------
    # 2. SUB-CLAIMS TIER (y: 165 - 240)
    # -------------------------------------------------------------
    sub_claims = [
        ("SUB-CLAIM C₁.₁: Tactile Reflex", "Primary Soft Tactile Slip Reflex",
         "Optical-tactile fingertips detect contact shear &amp; trip safety reflex within 12 ms; normal force arrested &lt; 14 N.",
         TEAL, 40, 165, 270, 72),
        ("SUB-CLAIM C₁.₂: Current Observer", "Secondary Torque Tripwire",
         "Joint servo drives observe motor current discrepancy Δτ &gt; 3.5 N·m to command friction brakes independently.",
         AMBER, 345, 165, 270, 72),
        ("SUB-CLAIM C₁.₃: Deterministic Gate", "Hard Real-Time Enforcer (Ch 12)",
         "1000 Hz safety loop isolates inverter gate signals within 1 ms; real-time bus latency bounded P99.99 ≤ 1.8 ms.",
         PETROL, 650, 165, 270, 72)
    ]

    for tag, title, desc, col, sx, sy, sw, sh in sub_claims:
        # Drop lines from horizontal rail
        svg.append(f'<line x1="{sx+sw/2}" y1="150" x2="{sx+sw/2}" y2="{sy}" stroke="{col}" stroke-width="1.4" marker-end="url(#arr-navy)"/>')
        
        svg.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.3" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="20" rx="6" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{sx+8}" y="{sy+14}" font-size="9.5" font-weight="700" fill="{col}">{tag}</text>')
        svg.append(f'<text x="{sx+8}" y="{sy+34}" font-size="10.5" font-weight="700" fill="{INK}">{title}</text>')
        
        words = desc.split()
        l1 = " ".join(words[:6])
        l2 = " ".join(words[6:13])
        l3 = " ".join(words[13:])
        svg.append(f'<text x="{sx+8}" y="{sy+48}" font-size="8.3" fill="{SLATE}">{l1}</text>')
        svg.append(f'<text x="{sx+8}" y="{sy+59}" font-size="8.3" fill="{SLATE}">{l2}</text>')
        if l3:
            svg.append(f'<text x="{sx+8}" y="{sy+70}" font-size="8.3" fill="{SLATE}">{l3}</text>')

    # -------------------------------------------------------------
    # 3. WARRANTS, DEFEATERS & COMPENSATORY BRANCHES (y: 255 - 380)
    # -------------------------------------------------------------
    
    # Left Column: Argument A1 -> Defeater D1 (Red Team Broken Link)
    a1_x, a1_y, a1_w, a1_h = 40, 255, 270, 60
    svg.append(f'<line x1="{a1_x+a1_w/2}" y1="237" x2="{a1_x+a1_w/2}" y2="{a1_y}" stroke="{TEAL}" stroke-width="1.2" marker-end="url(#arr-teal)"/>')
    svg.append(f'<rect x="{a1_x}" y="{a1_y}" width="{a1_w}" height="{a1_h}" rx="5" fill="{BLUE}" fill-opacity="0.05" stroke="{BLUE}" stroke-width="1"/>')
    svg.append(f'<text x="{a1_x+8}" y="{a1_y+14}" font-size="9" font-weight="700" fill="{BLUE}">ARGUMENT A₁ (Tactile Shear Warrant)</text>')
    svg.append(f'<text x="{a1_x+8}" y="{a1_y+28}" font-size="8.2" fill="{SLATE}">Elastomer marker displacement tracks surface</text>')
    svg.append(f'<text x="{a1_x+8}" y="{a1_y+40}" font-size="8.2" fill="{SLATE}">shear under Coulomb friction assumption (μ = 0.85).</text>')
    svg.append(f'<text x="{a1_x+8}" y="{a1_y+52}" font-size="8" font-weight="600" fill="{PETROL}">Assumes: Clean, dry contact interface</text>')

    # Defeater D1 Box (CRIMSON)
    d1_x, d1_y, d1_w, d1_h = 40, 328, 270, 72
    svg.append(f'<line x1="{a1_x+a1_w/2}" y1="{a1_y+a1_h}" x2="{d1_x+d1_w/2}" y2="{d1_y}" stroke="{CORAL}" stroke-width="1.4" stroke-dasharray="3,2" marker-end="url(#arr-coral)"/>')
    svg.append(f'<rect x="{d1_x}" y="{d1_y}" width="{d1_w}" height="{d1_h}" rx="5" fill="{CORAL}" fill-opacity="0.08" stroke="{CORAL}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{d1_x}" y="{d1_y}" width="{d1_w}" height="18" rx="5" fill="{CORAL}" fill-opacity="0.2"/>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+13}" font-size="8.8" font-weight="700" fill="{CORAL}">✕ DEFEATER D₁: OIL CONTAMINATION (BROKEN LINK)</text>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+30}" font-size="8.2" font-weight="600" fill="{INK}">Machining fluid drops friction: μ = 0.85 → 0.12.</text>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+43}" font-size="8.2" fill="{SLATE}">Workpiece slides freely across silicone; zero marker</text>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+56}" font-size="8.2" fill="{SLATE}">strain observed. Primary tactile reflex fails silently!</text>')
    svg.append(f'<text x="{d1_x+8}" y="{d1_y+67}" font-size="8" font-weight="700" fill="{CORAL}">Result: Unmonitored rigid collision trajectory</text>')

    # Middle Column: Argument A2 -> Unmitigated Defeater vs Compensated Condition
    a2_x, a2_y, a2_w, a2_h = 345, 255, 270, 60
    svg.append(f'<line x1="{a2_x+a2_w/2}" y1="237" x2="{a2_x+a2_w/2}" y2="{a2_y}" stroke="{AMBER}" stroke-width="1.2" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<rect x="{a2_x}" y="{a2_y}" width="{a2_w}" height="{a2_h}" rx="5" fill="{BLUE}" fill-opacity="0.05" stroke="{BLUE}" stroke-width="1"/>')
    svg.append(f'<text x="{a2_x+8}" y="{a2_y+14}" font-size="9" font-weight="700" fill="{BLUE}">ARGUMENT A₂ (Current Tripwire Warrant)</text>')
    svg.append(f'<text x="{a2_x+8}" y="{a2_y+28}" font-size="8.2" fill="{SLATE}">Motor torque observer trips at Δτ = 3.5 N·m</text>')
    svg.append(f'<text x="{a2_x+8}" y="{a2_y+40}" font-size="8.2" fill="{SLATE}">(F_trip = 35 N). Mechanical brakes halt moving arm.</text>')
    svg.append(f'<text x="{a2_x+8}" y="{a2_y+52}" font-size="8" font-weight="600" fill="{AMBER}">Unmitigated: Peak force reaches 88 N &gt; 50 N!</text>')

    # Conditional Branch Box (AMBER/TEAL)
    c_box_x, c_box_y, c_box_w, c_box_h = 345, 328, 270, 72
    svg.append(f'<line x1="{a2_x+a2_w/2}" y1="{a2_y+a2_h}" x2="{c_box_x+c_box_w/2}" y2="{c_box_y}" stroke="{AMBER}" stroke-width="1.4" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<rect x="{c_box_x}" y="{c_box_y}" width="{c_box_w}" height="{c_box_h}" rx="5" fill="{AMBER}" fill-opacity="0.08" stroke="{AMBER}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{c_box_x}" y="{c_box_y}" width="{c_box_w}" height="18" rx="5" fill="{AMBER}" fill-opacity="0.2"/>')
    svg.append(f'<text x="{c_box_x+8}" y="{c_box_y+13}" font-size="8.8" font-weight="700" fill="{AMBER}">⚠ COMPENSATORY CONDITIONAL BRANCH</text>')
    svg.append(f'<text x="{c_box_x+8}" y="{c_box_y+30}" font-size="8.2" font-weight="600" fill="{INK}">1. Derate payload mass: m ≤ 1.5 kg (m_eff = 4.0 kg)</text>')
    svg.append(f'<text x="{c_box_x+8}" y="{c_box_y+43}" font-size="8.2" font-weight="600" fill="{INK}">2. Add optical surface-dryness station at feeder</text>')
    svg.append(f'<text x="{c_box_x+8}" y="{c_box_y+56}" font-size="8.2" font-weight="600" fill="{INK}">3. Tighten torque tripwire threshold to F_trip = 18 N</text>')
    svg.append(f'<text x="{c_box_x+8}" y="{c_box_y+67}" font-size="8" font-weight="700" fill="{TEAL}">Bound Restored: F_peak = 38 N ≤ 50 N (∫F dt = 1.1 N·s)</text>')

    # Right Column: Argument A3 -> Hardware-Enforced Invariant Warrant
    a3_x, a3_y, a3_w, a3_h = 650, 255, 270, 60
    svg.append(f'<line x1="{a3_x+a3_w/2}" y1="237" x2="{a3_x+a3_w/2}" y2="{a3_y}" stroke="{PETROL}" stroke-width="1.2" marker-end="url(#arr-petrol)"/>')
    svg.append(f'<rect x="{a3_x}" y="{a3_y}" width="{a3_w}" height="{a3_h}" rx="5" fill="{BLUE}" fill-opacity="0.05" stroke="{BLUE}" stroke-width="1"/>')
    svg.append(f'<text x="{a3_x+8}" y="{a3_y+14}" font-size="9" font-weight="700" fill="{BLUE}">ARGUMENT A₃ (Architectural Warrant)</text>')
    svg.append(f'<text x="{a3_x+8}" y="{a3_y+28}" font-size="8.2" fill="{SLATE}">Dedicated MCU hardware interlock line cannot be</text>')
    svg.append(f'<text x="{a3_x+8}" y="{a3_y+40}" font-size="8.2" fill="{SLATE}">preempted, starved, or bypassed by MPU vision tasks.</text>')
    svg.append(f'<text x="{a3_x+8}" y="{a3_y+52}" font-size="8" font-weight="600" fill="{PETROL}">Independent permission path (Chapter 12)</text>')

    # Architectural Proof Box (PETROL)
    p3_x, p3_y, p3_w, p3_h = 650, 328, 270, 72
    svg.append(f'<line x1="{a3_x+a3_w/2}" y1="{a3_y+a3_h}" x2="{p3_x+p3_w/2}" y2="{p3_y}" stroke="{PETROL}" stroke-width="1.4" marker-end="url(#arr-petrol)"/>')
    svg.append(f'<rect x="{p3_x}" y="{p3_y}" width="{p3_w}" height="{p3_h}" rx="5" fill="{PETROL}" fill-opacity="0.08" stroke="{PETROL}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{p3_x}" y="{p3_y}" width="{p3_w}" height="18" rx="5" fill="{PETROL}" fill-opacity="0.2"/>')
    svg.append(f'<text x="{p3_x+8}" y="{p3_y+13}" font-size="8.8" font-weight="700" fill="{PETROL}">✓ VERIFIED ARCHITECTURAL INDEPENDENCE</text>')
    svg.append(f'<text x="{p3_x+8}" y="{p3_y+30}" font-size="8.2" fill="{SLATE}">• Inverter gate power cut latency: t_gate = 1.0 ms</text>')
    svg.append(f'<text x="{p3_x+8}" y="{p3_y+43}" font-size="8.2" fill="{SLATE}">• Holding brake engage latency: t_brake = 18.0 ms</text>')
    svg.append(f'<text x="{p3_x+8}" y="{p3_y+56}" font-size="8.2" fill="{SLATE}">• Watchdog hardware dead-man timeout: 50 ms</text>')
    svg.append(f'<text x="{p3_x+8}" y="{p3_y+67}" font-size="8" font-weight="700" fill="{TEAL}">Strict forward invariance guaranteed in hardware</text>')

    # -------------------------------------------------------------
    # 4. CONCRETE EVIDENCE NODES TIER (y: 415 - 510)
    # -------------------------------------------------------------
    ev_nodes = [
        ("EVIDENCE E₁", "Tactile Metrology", "• Calibrated pad stiffness k = 4.0 N/mm\n• P99.9 perception latency: 12 ms\n• 10⁵ HIL dropouts: 0 uncontained trips\n• Trace: LOG-TAC-2026-0814", TEAL, 40, 415, 210, 80),
        ("EVIDENCE E₂", "Impact Mechanics Proof", "• Dynamometer a_brake = 8.0 m/s²\n• Impulse integral: ∫F dt = 1.1 N·s\n• Max contact force: F_peak = 38 N\n• Trace: DYN-IMP-2026-0922", TEAL, 265, 415, 210, 80),
        ("EVIDENCE E₃", "Optical Dryness Check", "• Feeder strobe specular detector\n• Threshold: Fluid sheen &lt; 2% area\n• Automatic rejection of oily stock\n• Trace: OPT-DRY-2026-0301", AMBER, 490, 415, 210, 80),
        ("EVIDENCE E₄", "MCU Execution Traces", "• Logic analyzer trace: 1 kHz ± 5 µs\n• Current filter delay t_sense = 6 ms\n• SPI DMA bus load: 0 overruns\n• Trace: LOG-MCU-2026-1105", PETROL, 715, 415, 205, 80)
    ]

    for tag, title, items, col, ex, ey, ew, eh in ev_nodes:
        svg.append(f'<rect x="{ex}" y="{ey}" width="{ew}" height="{eh}" rx="5" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.1" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{ex}" y="{ey}" width="{ew}" height="18" rx="5" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{ex+8}" y="{ey+13}" font-size="8.8" font-weight="700" fill="{col}">{tag}</text>')
        svg.append(f'<text x="{ex+ew-8}" y="{ey+13}" font-size="8" font-weight="600" fill="{INK}" text-anchor="end">{title}</text>')
        
        for idx, line in enumerate(items.split("\n")):
            svg.append(f'<text x="{ex+8}" y="{ey+28+idx*12.5}" font-size="7.8" fill="{SLATE}">{line}</text>')

    # Connectors to Evidence
    svg.append(f'<line x1="{d1_x+100}" y1="{d1_y+d1_h}" x2="{40+105}" y2="415" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,2" marker-end="url(#arr-coral)"/>')
    svg.append(f'<line x1="{c_box_x+70}" y1="{c_box_y+c_box_h}" x2="{265+105}" y2="415" stroke="{AMBER}" stroke-width="1.2" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<line x1="{c_box_x+200}" y1="{c_box_y+c_box_h}" x2="{490+105}" y2="415" stroke="{AMBER}" stroke-width="1.2" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<line x1="{p3_x+p3_w/2}" y1="{p3_y+p3_h}" x2="{715+100}" y2="415" stroke="{PETROL}" stroke-width="1.2" marker-end="url(#arr-petrol)"/>')

    # -------------------------------------------------------------
    # 5. ADJUDICATION RESOLUTION BANNER (y: 512 - 595)
    # -------------------------------------------------------------
    res_y = 512
    svg.append(f'<rect x="40" y="{res_y}" width="{W-80}" height="90" rx="6" fill="{BG_LIGHT}" stroke="{NAVY}" stroke-width="1.3"/>')
    svg.append(f'<rect x="40" y="{res_y}" width="{W-80}" height="22" rx="6" fill="{NAVY}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{W/2}" y="{res_y+15}" font-size="10.5" font-weight="700" fill="{NAVY}" text-anchor="middle">ADJUDICATION RESOLUTION &amp; VERDICT CONTRACT</text>')

    # 3 Verdict outcomes comparison inside banner
    v_w = (W - 110) / 3
    
    # Verdict 1: Unconditional (Denied)
    v1_x = 55
    svg.append(f'<rect x="{v1_x}" y="{res_y+28}" width="{v_w}" height="54" rx="4" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="1"/>')
    svg.append(f'<text x="{v1_x+8}" y="{res_y+42}" font-size="9" font-weight="700" fill="{CORAL}">✕ UNCONDITIONAL OPERATE</text>')
    svg.append(f'<text x="{v1_x+8}" y="{res_y+55}" font-size="7.8" font-weight="700" fill="{INK}">STATUS: REFUSED / REJECTED</text>')
    svg.append(f'<text x="{v1_x+8}" y="{res_y+67}" font-size="7.5" fill="{SLATE}">Defeater D₁ invalidates unconstrained slip reflex.</text>')
    svg.append(f'<text x="{v1_x+8}" y="{res_y+77}" font-size="7.5" fill="{CORAL}">Peak contact force reaches 88 N (&gt; 50 N limit).</text>')

    # Verdict 2: Operate with Conditions (Approved)
    v2_x = 55 + v_w + 10
    svg.append(f'<rect x="{v2_x}" y="{res_y+28}" width="{v_w}" height="54" rx="4" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1.3"/>')
    svg.append(f'<text x="{v2_x+8}" y="{res_y+42}" font-size="9" font-weight="700" fill="{TEAL}">✓ OPERATE WITH CONDITIONS</text>')
    svg.append(f'<text x="{v2_x+8}" y="{res_y+55}" font-size="7.8" font-weight="700" fill="{TEAL}">STATUS: APPROVED &amp; SIGNED</text>')
    svg.append(f'<text x="{v2_x+8}" y="{res_y+67}" font-size="7.5" fill="{SLATE}">1. Payload ≤ 1.5 kg · 2. Feeder optical check</text>')
    svg.append(f'<text x="{v2_x+8}" y="{res_y+77}" font-size="7.5" fill="{TEAL}">3. F_trip = 18 N tripwire → F_peak = 38 N ≤ 50 N</text>')

    # Verdict 3: Refuse / Continuous Invalidation
    v3_x = 55 + 2 * (v_w + 10)
    svg.append(f'<rect x="{v3_x}" y="{res_y+28}" width="{v_w}" height="54" rx="4" fill="{BG_WHITE}" stroke="{AMBER}" stroke-width="1"/>')
    svg.append(f'<text x="{v3_x+8}" y="{res_y+42}" font-size="9" font-weight="700" fill="{AMBER}">⚠ STANDING CONDITIONS &amp; EXPIRY</text>')
    svg.append(f'<text x="{v3_x+8}" y="{res_y+55}" font-size="7.8" font-weight="700" fill="{INK}">LIFECYCLE CONTRACT</text>')
    svg.append(f'<text x="{v3_x+8}" y="{res_y+67}" font-size="7.5" fill="{SLATE}">• Sensor recalibration every 500 operating hours</text>')
    svg.append(f'<text x="{v3_x+8}" y="{res_y+77}" font-size="7.5" fill="{AMBER}">• Automatic invalidation upon oil mist detection</text>')

    svg.append('</svg>')
    out_svg = "book/chapters/16-release/figures/fig16_cae_tree.svg"
    save_svg_and_pdf(out_svg, "\n".join(svg))


def gen_ch16_release_record():
    """
    Figure 16.2: Signed Release Record Architecture, Standing Condition Contracts,
    and Continuous Invalidation Pipeline.
    Illustrates immutable provenance ledger, hardware Root of Trust, 1 kHz enforcers,
    and automatic invalidation triggers.
    """
    W = 960
    H = 550
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)

    # Background card
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')

    # Title & Subtitle
    svg.append(f'<text x="{W/2}" y="28" class="title">CRYPTOGRAPHIC RELEASE RECORD ARCHITECTURE &amp; RUNTIME ENFORCEMENT PIPELINE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Immutable Engineering Ledger ⟷ Hardware Root of Trust ⟷ Envelope Trichotomy ⟷ Automatic Invalidation Triggers</text>')

    col_w = 285
    col_gap = 20
    c1_x = 35
    c2_x = c1_x + col_w + col_gap
    c3_x = c2_x + col_w + col_gap
    top_y = 66
    col_h = 455

    # -------------------------------------------------------------
    # COLUMN 1: IMMUTABLE RELEASE RECORD LEDGER (Purple / Navy)
    # -------------------------------------------------------------
    svg.append(f'<rect x="{c1_x}" y="{top_y}" width="{col_w}" height="{col_h}" rx="8" fill="{BG_WHITE}" stroke="{PURPLE}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{c1_x}" y="{top_y}" width="{col_w}" height="28" rx="8" fill="{PURPLE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c1_x+col_w/2}" y="{top_y+19}" font-size="10" font-weight="700" fill="{PURPLE}" text-anchor="middle">1. IMMUTABLE RELEASE RECORD LEDGER</text>')

    fields = [
        ("Claim Identifier", "CLAIM-CH16-TACTILE-001", NAVY),
        ("GSN Tree Topology Hash", "SHA256: 8f3c4e...d94a", NAVY),
        ("Verified Evidence Pointer Array", "• LOG-TAC-2026-0814 (P99.9: 12 ms)\n• DYN-IMP-2026-0922 (∫F dt: 1.1 N·s)\n• LOG-MCU-2026-1105 (Jitter: 5 µs)", PETROL),
        ("Cryptographic Manifest Hashes", "• Policy Checkpoint: SHA256: e4b7...10c2\n• Safety FPGA Bitstream: SHA256: 3a91...fe78\n• FreeRTOS Kernel: SHA256: 7d18...99b0", BLUE),
        ("Standing Conditions Contract", "1. Max Payload Mass: m ≤ 1.5 kg\n2. Optical Feeder Dryness: &lt; 2% sheen\n3. Torque Observer Trip: F_trip = 18 N", AMBER),
        ("Signed Adjudicator Attestation", "Lead Systems Adjudicator Key:\nEd25519: sig_44b9_88fa... (Verified)\nExpiry: 2000 h / 10⁶ cycles", PURPLE)
    ]

    cur_y = top_y + 36
    for title, val, col in fields:
        svg.append(f'<rect x="{c1_x+8}" y="{cur_y}" width="{col_w-16}" height="56" rx="4" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="0.8"/>')
        svg.append(f'<rect x="{c1_x+8}" y="{cur_y}" width="3" height="56" rx="1" fill="{col}"/>')
        svg.append(f'<text x="{c1_x+16}" y="{cur_y+13}" font-size="8.5" font-weight="700" fill="{col}">{title}</text>')
        for idx, line in enumerate(val.split("\n")):
            svg.append(f'<text x="{c1_x+16}" y="{cur_y+26+idx*11}" font-size="7.5" font-family="monospace" fill="{SLATE}">{line}</text>')
        cur_y += 62

    svg.append(f'<text x="{c1_x+col_w/2}" y="{cur_y+14}" font-size="7.5" font-weight="600" fill="{MUTED}" text-anchor="middle">Stored in Machine Notebook Ledger (Section 4.10)</text>')

    # -------------------------------------------------------------
    # COLUMN 2: HARDWARE ROOT OF TRUST & 1 kHz ENFORCER (Petrol / Teal)
    # -------------------------------------------------------------
    svg.append(f'<rect x="{c2_x}" y="{top_y}" width="{col_w}" height="{col_h}" rx="8" fill="{BG_WHITE}" stroke="{PETROL}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{c2_x}" y="{top_y}" width="{col_w}" height="28" rx="8" fill="{PETROL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c2_x+col_w/2}" y="{top_y+19}" font-size="10" font-weight="700" fill="{PETROL}" text-anchor="middle">2. HARDWARE ROOT OF TRUST &amp; ENFORCER</text>')

    # Secure Boot & OTP eFuses Box
    b1_y = top_y + 36
    svg.append(f'<rect x="{c2_x+8}" y="{b1_y}" width="{col_w-16}" height="76" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{c2_x+14}" y="{b1_y+15}" font-size="9" font-weight="700" fill="{INK}">Hardware Secure Boot &amp; Manifest Verification</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b1_y+29}" font-size="7.8" fill="{SLATE}">• Public key burned into silicon OTP eFuses</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b1_y+41}" font-size="7.8" fill="{SLATE}">• Dual-Bank (A/B) flash with watchdog fallback</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b1_y+53}" font-size="7.8" fill="{SLATE}">• 1-bit hash mismatch halts inverter gate power</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b1_y+67}" font-size="8" font-weight="700" fill="{CORAL}">Refusal to energize without signed release record</text>')

    # Envelope Trichotomy Box
    b2_y = b1_y + 84
    svg.append(f'<rect x="{c2_x+8}" y="{b2_y}" width="{col_w-16}" height="135" rx="5" fill="{NAVY}" fill-opacity="0.04" stroke="{NAVY}" stroke-width="1"/>')
    svg.append(f'<text x="{c2_x+14}" y="{b2_y+15}" font-size="9" font-weight="700" fill="{NAVY}">Runtime Envelope Trichotomy (Section 16.5)</text>')

    # 3 states
    st_w = col_w - 32
    # Known-True
    svg.append(f'<rect x="{c2_x+16}" y="{b2_y+24}" width="{st_w}" height="30" rx="3" fill="{TEAL}" fill-opacity="0.1" stroke="{TEAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{c2_x+22}" y="{b2_y+36}" font-size="8" font-weight="700" fill="{TEAL}">1. KNOWN-TRUE (Inside Envelope)</text>')
    svg.append(f'<text x="{c2_x+22}" y="{b2_y+47}" font-size="7.2" fill="{SLATE}">Telemetry valid · Covariance bounded → Permit u_t = p_t</text>')

    # Known-False
    svg.append(f'<rect x="{c2_x+16}" y="{b2_y+58}" width="{st_w}" height="30" rx="3" fill="{CORAL}" fill-opacity="0.1" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{c2_x+22}" y="{b2_y+70}" font-size="8" font-weight="700" fill="{CORAL}">2. KNOWN-FALSE (Physical Boundary Breached)</text>')
    svg.append(f'<text x="{c2_x+22}" y="{b2_y+81}" font-size="7.2" fill="{SLATE}">F_norm &gt; 18 N or tracking error &gt; 50 mm → Trip brakes &lt; 1 ms</text>')

    # Unknown
    svg.append(f'<rect x="{c2_x+16}" y="{b2_y+92}" width="{st_w}" height="32" rx="3" fill="{CRIMSON}" fill-opacity="0.15" stroke="{CRIMSON}" stroke-width="1.1"/>')
    svg.append(f'<text x="{c2_x+22}" y="{b2_y+104}" font-size="8" font-weight="700" fill="{CRIMSON}">3. UNKNOWN (Epistemic Gap / Loss of Telemetry)</text>')
    svg.append(f'<text x="{c2_x+22}" y="{b2_y+117}" font-size="7.2" font-weight="600" fill="{INK}">CRC fail / blind lidar → MUST TREAT AS OUTSIDE → E-Stop</text>')

    # Standing Condition Detectors Box
    b3_y = b2_y + 143
    svg.append(f'<rect x="{c2_x+8}" y="{b3_y}" width="{col_w-16}" height="102" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+15}" font-size="9" font-weight="700" fill="{INK}">1 kHz Synchronous Condition Detectors</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+30}" font-size="7.8" fill="{SLATE}">• Continuously Monitored: Load cell payload mass check</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+42}" font-size="7.8" fill="{SLATE}">  (m ≤ 1.5 kg) before every arm acceleration cycle</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+56}" font-size="7.8" fill="{SLATE}">• Optical Dryness Strobe: Pre-grasp surface reflectance</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+68}" font-size="7.8" fill="{SLATE}">• Motor Thermistors: Winding temperature T ≤ 393 K</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+82}" font-size="7.8" fill="{SLATE}">• Periodically Re-established: Joint torque calibration (500 h)</text>')
    svg.append(f'<text x="{c2_x+14}" y="{b3_y+94}" font-size="7.8" fill="{SLATE}">• Friction Brake Inspection: Lining wear check (250k cycles)</text>')

    # -------------------------------------------------------------
    # COLUMN 3: AUTOMATIC INVALIDATION & RE-DECISION (Crimson / Coral)
    # -------------------------------------------------------------
    svg.append(f'<rect x="{c3_x}" y="{top_y}" width="{col_w}" height="{col_h}" rx="8" fill="{BG_WHITE}" stroke="{CRIMSON}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{c3_x}" y="{top_y}" width="{col_w}" height="28" rx="8" fill="{CRIMSON}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{c3_x+col_w/2}" y="{top_y+19}" font-size="10" font-weight="700" fill="{CRIMSON}" text-anchor="middle">3. INVALIDATION TRIGGERS &amp; RE-DECISION</text>')

    inv_categories = [
        ("Software Invalidation Triggers", "• Any policy neural network weight update\n• RTOS thread scheduling priority change\n• Lower-level sensor driver / CAN patch\n→ Automatically voids SHA-256 signature", CORAL),
        ("Hardware Invalidation Triggers", "• Unverified actuator or sensor part swap\n• Gearbox backlash &gt; 0.10° (15 ms phase lag)\n• Brake lining wear &lt; 2.0 mm thickness\n→ Hardware interlock inhibits power stage", CORAL),
        ("Operational Invalidation Triggers", "• Ambient temperature T &gt; 40°C\n• DC Bus voltage droop &lt; 21.6 V (10% drop)\n• Workpiece mass &gt; 1.5 kg\n→ Immediate safe stop &amp; operational latch", AMBER),
        ("Continuous Re-Decision Protocol", "• Non-volatile circular buffer logs all residuals\n• Periodic empirical audit every 500 op hours\n• Sensor noise drift &gt; 15% triggers derating\n• Re-adjudication required to renew authority", PETROL)
    ]

    cur_y3 = top_y + 36
    for title, desc, col in inv_categories:
        svg.append(f'<rect x="{c3_x+8}" y="{cur_y3}" width="{col_w-16}" height="76" rx="4" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="0.8"/>')
        svg.append(f'<rect x="{c3_x+8}" y="{cur_y3}" width="3" height="76" rx="1" fill="{col}"/>')
        svg.append(f'<text x="{c3_x+16}" y="{cur_y3+14}" font-size="8.5" font-weight="700" fill="{col}">{title}</text>')
        for idx, line in enumerate(desc.split("\n")):
            svg.append(f'<text x="{c3_x+16}" y="{cur_y3+28+idx*11}" font-size="7.5" fill="{SLATE}">{line}</text>')
        cur_y3 += 82

    # Bottom Invalidation Banner inside Col 3
    svg.append(f'<rect x="{c3_x+8}" y="{cur_y3}" width="{col_w-16}" height="42" rx="4" fill="{CRIMSON}" fill-opacity="0.12" stroke="{CRIMSON}" stroke-width="1"/>')
    svg.append(f'<text x="{c3_x+col_w/2}" y="{cur_y3+16}" font-size="8.5" font-weight="700" fill="{CRIMSON}" text-anchor="middle">REVOCATION OF OPERATING AUTHORITY</text>')
    svg.append(f'<text x="{c3_x+col_w/2}" y="{cur_y3+30}" font-size="7.5" fill="{INK}" text-anchor="middle">Inverter gates clamped · Mandatory re-audit</text>')

    # -------------------------------------------------------------
    # INTER-COLUMN CONNECTING ARROWS
    # -------------------------------------------------------------
    # From Ledger to Root of Trust
    svg.append(f'<line x1="{c1_x+col_w}" y1="{top_y+74}" x2="{c2_x}" y2="{top_y+74}" stroke="{PURPLE}" stroke-width="1.6" marker-end="url(#arr-purple)"/>')
    svg.append(f'<line x1="{c1_x+col_w}" y1="{top_y+340}" x2="{c2_x}" y2="{top_y+340}" stroke="{AMBER}" stroke-width="1.6" marker-end="url(#arr-bronze)"/>')

    # From Enforcer to Invalidation
    svg.append(f'<line x1="{c2_x+col_w}" y1="{top_y+180}" x2="{c3_x}" y2="{top_y+180}" stroke="{CORAL}" stroke-width="1.6" marker-end="url(#arr-coral)"/>')
    svg.append(f'<line x1="{c2_x+col_w}" y1="{top_y+340}" x2="{c3_x}" y2="{top_y+340}" stroke="{PETROL}" stroke-width="1.6" marker-end="url(#arr-petrol)"/>')

    svg.append('</svg>')
    out_svg = "book/chapters/16-release/figures/fig16_release_record.svg"
    save_svg_and_pdf(out_svg, "\n".join(svg))


def run_all():
    print("=== Generating Chapter 16 Figures ===")
    gen_ch16_cae_tree()
    gen_ch16_release_record()
    print("✓ Chapter 16 figures generated successfully.")

if __name__ == "__main__":
    run_all()
