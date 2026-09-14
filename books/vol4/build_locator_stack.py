#!/usr/bin/env python3
"""
Generate the four-tier "systems organ locator" stack figure that opens every
chapter of Physical AI.

Pure native SVG generator adhering to the Volume IV (Figure 3.7) Design System:
- Integrated Proposal–Permission Causal Boundary bridge
- Embedded frequency badges across all four tiers
- Dynamic active-tier and active-chapter "THIS CH" hero cards
- Strict orthogonal routing for feedback loops and command flows

Usage: python3 books/vol4/build_locator_stack.py [slug ...]
"""

import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

# slug -> (chapter number as printed, badge label, active tiers, own card id)
CHAPTERS = {
    "01_boundary":     (1,  "BOUNDARY",                (1,),   None),
    "02_body":         (2,  "THE BODY",                (1,),   None),
    "03_brain":        (3,  "THE BRAIN",               (3,),   None),
    "04_nervous":      (4,  "NERVOUS SYSTEM",          (2,),   None),
    "05_data":         (5,  "DATA & INGESTION",        (3,),   "b1"),
    "06_training":     (6,  "TRAINING & SIM-TO-REAL",  (3, 1), None),
    "07_evaluation":   (7,  "EVALUATION & METROLOGY",  (3, 1), None),
    "08_perception":   (8,  "SPATIAL PERCEPTION",      (3,),   "b2"),
    "09_memory":       (9,  "TEMPORAL MEMORY",         (3,),   "b3"),
    "10_intent":       (10, "INTENT & GOALS",          (3,),   "b4"),
    "11_planning":     (11, "TRAJECTORY PLANNING",     (3,),   "b5"),
    "12_enforcement":  (12, "RUNTIME ENFORCEMENT",     (2,),   "n1"),
    "13_placement":    (13, "SILICON PLACEMENT",       (2,),   "n2"),
    "14_intervention": (14, "SHARED AUTONOMY",         (4,),   "g1"),
    "15_verification": (15, "VERIFICATION",            (4,),   "g2"),
    "16_release":      (16, "SAFETY CASES",            (4,),   "g3"),
    "17_frontier":     (17, "FRONTIER LIMITS",         (4,),   "g4"),
}

# Subcard metadata definitions
# (id, title, line1, line2)
L4_CARDS = [
    ("g1", "Ch 14 Shared Autonomy", "Policy Blending · Takeover", "Arbitration &amp; Liveness"),
    ("g2", "Ch 15 Verification", "Sim → PIL → HIL → In-Situ", "Formal Invariants"),
    ("g3", "Ch 16 Safety Cases", "Goal Structuring · UL 4600", "Assurance Arguments"),
    ("g4", "Ch 17 Frontier Limits", "Observational Limits", "Open-World Grounding"),
]

L3_CARDS = [
    ("b1", "1. Ingestion", "Ch 05 · Data &amp; CSI-2", "Ring Buffers / SPSC"),
    ("b2", "2. Perception", "Ch 08 · Perception", "o_t → z_t · ViT Encoders"),
    ("b3", "3. Memory", "Ch 09 · Temporal Memory", "z_t → b_t · SE(3) Belief"),
    ("b4", "4. Intent", "Ch 10 · Intent &amp; Goals", "b_t → L_t · TTL Leases"),
    ("b5", "5. Planner", "Ch 11 · Trajectory Planning", "L_t → û · ACT / Diffusion"),
]

L2_CARDS = [
    ("n1", "1 kHz Reflex Enforcer", "Ch 12 · Runtime Enforcement", "CBF h(x) ≥ 0 · Safety Refusal"),
    ("n2", "Silicon Placement &amp; QoS", "Ch 13 · Silicon Placement", "Interconnect QoS · SRAM Seqlock"),
    ("n3", "Certified Actuator Latch", "Hardware Timing Latch", "Latches u* into PWM Registers"),
]

L1_CARDS = [
    ("p1", "Power Inverters &amp; Drivers", "PWM Switching → Phase Current", "H-Bridge Gate Timing"),
    ("p2", "Mechanics, Torque &amp; Momentum", "Lorentz Force → Torque τ", "Momentum p = mv"),
    ("p3", "Causal Boundary &amp; Dissipation", "Stopping Envelope d_stop ≤ d_clear", "Friction &amp; Thermal Dissipation"),
]

def render_subcard(cid, title, line1, line2, x, y, width, height, is_hero, is_tier_active, own_card):
    """Render a single modular subcard with crisp typography and exact padding."""
    cx = x + width / 2
    if is_hero:
        return f"""
      <!-- Hero Card {cid} -->
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="4" fill="#1F407A" stroke="#1F407A" stroke-width="2"/>
      <rect x="{x + 8}" y="{y + 5}" width="52" height="14" rx="2" fill="#93C5FD"/>
      <text x="{x + 34}" y="{y + 15.5}" font-size="8" font-weight="bold" fill="#0F172A" text-anchor="middle">THIS CH</text>
      <text x="{cx}" y="{y + 33}" font-size="11" font-weight="bold" fill="#FFFFFF" text-anchor="middle">{title}</text>
      <text x="{cx}" y="{y + 47}" font-size="8.5" font-weight="bold" fill="#93C5FD" text-anchor="middle">{line1}</text>
      <text x="{cx}" y="{y + 58.5}" font-size="8" fill="#E2E8F0" text-anchor="middle">{line2}</text>"""
    elif is_tier_active and own_card is None:
        return f"""
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="3" fill="#FFFFFF" stroke="#93C5FD" stroke-width="1.5"/>
      <text x="{cx}" y="{y + 22}" font-size="11" font-weight="bold" fill="#1F407A" text-anchor="middle">{title}</text>
      <text x="{cx}" y="{y + 39}" font-size="9" font-weight="600" fill="#334155" text-anchor="middle">{line1}</text>
      <text x="{cx}" y="{y + 53}" font-size="8" fill="#64748B" text-anchor="middle">{line2}</text>"""
    elif is_tier_active:
        return f"""
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="3" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
      <text x="{cx}" y="{y + 22}" font-size="11" font-weight="bold" fill="#475569" text-anchor="middle">{title}</text>
      <text x="{cx}" y="{y + 39}" font-size="9" fill="#64748B" text-anchor="middle">{line1}</text>
      <text x="{cx}" y="{y + 53}" font-size="8" fill="#94A3B8" text-anchor="middle">{line2}</text>"""
    else:
        return f"""
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="3" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1"/>
      <text x="{cx}" y="{y + 22}" font-size="11" font-weight="bold" fill="#475569" text-anchor="middle">{title}</text>
      <text x="{cx}" y="{y + 39}" font-size="9" fill="#64748B" text-anchor="middle">{line1}</text>
      <text x="{cx}" y="{y + 53}" font-size="8" fill="#64748B" text-anchor="middle">{line2}</text>"""

def build_svg(slug):
    num, badge, active_tiers, own_card = CHAPTERS[slug]
    
    # Active focus badge calculation
    badge_text = f"ACTIVE FOCUS: CHAPTER {num:02d} · {badge}".replace("&", "&amp;")
    badge_w = max(260, len(badge_text) * 7.0 + 36)
    badge_x = 950 - badge_w

    # Tier states
    is_l4_active = 4 in active_tiers
    is_l3_active = 3 in active_tiers
    is_l2_active = 2 in active_tiers
    is_l1_active = 1 in active_tiers

    # Dynamic subtitle texts based on chapter focus
    l4_sub = "Policy Supervision &amp; Safety Cases"
    if is_l4_active:
        l4_sub = "ACTIVE TIER (Ch 14–17 Governance)"

    l3_sub = "High-Capacity Foundation Deliberation (Ch 03)"
    if slug == "03_brain":
        l3_sub = "ACTIVE TIER · ENTIRE DELIBERATION PIPELINE"
    elif slug in ("06_training", "07_evaluation"):
        l3_sub = "ACTIVE TIER · SIMULATION &amp; EMBODIED LEARNING"
    elif is_l3_active:
        l3_sub = f"ACTIVE TIER · FOCAL STAGE: CH {num:02d}"

    l2_sub = "Zero-Allocation Deterministic Silicon · Watchdogs (Ch 04)"
    if slug == "04_nervous":
        l2_sub = "ACTIVE TIER · DETERMINISTIC SAFETY REFLEX"
    elif is_l2_active:
        l2_sub = f"ACTIVE TIER · FOCAL MECHANISM: CH {num:02d}"

    l1_sub = "Inertia, Transduction, Friction, Thermal Limits (Ch 01 · 02)"
    if slug in ("01_boundary", "02_body"):
        l1_sub = f"ACTIVE TIER · CONTINUOUS PHYSICAL DYNAMICS (CH {num:02d})"
    elif slug in ("06_training", "07_evaluation"):
        l1_sub = "ACTIVE TIER · PHYSICAL REALITY &amp; REALITY GAP"

    # ==================== BUILD LAYER 4 ====================
    l4_w, l4_h = 905, 102
    if is_l4_active:
        l4_card_bg = 'fill="#F0F4FA" stroke="#1F407A" stroke-width="2"'
        l4_hdr = f"""
    <rect x="0" y="0" width="{l4_w}" height="26" rx="4" fill="#1F407A"/>
    <rect x="0" y="22" width="{l4_w}" height="4" fill="#1F407A"/>
    <rect x="14" y="4" width="76" height="18" rx="3" fill="#FFFFFF" opacity="0.2"/>
    <text x="52" y="17" font-size="9.5" font-weight="bold" fill="#FFFFFF" text-anchor="middle">0.1–1 Hz</text>
    <text x="102" y="18" font-size="12" font-weight="bold" fill="#FFFFFF">LAYER 4 · SYSTEM GOVERNANCE &amp; ASSURANCE TIER</text>
    <text x="890" y="18" font-size="10.5" font-weight="bold" fill="#93C5FD" text-anchor="end">{l4_sub}</text>"""
    else:
        l4_card_bg = 'fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"'
        l4_hdr = f"""
    <rect x="14" y="8" width="76" height="18" rx="3" fill="#E2E8F0"/>
    <text x="52" y="21" font-size="9.5" font-weight="bold" fill="#475569" text-anchor="middle">0.1–1 Hz</text>
    <text x="102" y="21" font-size="12" font-weight="bold" fill="#475569">LAYER 4 · SYSTEM GOVERNANCE &amp; ASSURANCE TIER</text>
    <text x="890" y="21" font-size="10.5" fill="#64748B" text-anchor="end">{l4_sub}</text>"""

    # Layer 4 subcards
    l4_cards_svg = []
    card_w4 = 207
    gap4 = 16
    for i, (cid, title, l1, l2) in enumerate(L4_CARDS):
        x = i * (card_w4 + gap4)
        is_hero = (own_card == cid)
        c_svg = render_subcard(cid, title, l1, l2, x, 0, card_w4, 64, is_hero, is_l4_active, own_card)
        l4_cards_svg.append(c_svg)

    # ==================== BUILD LAYER 3 ====================
    l3_w, l3_h = 905, 120
    if is_l3_active:
        l3_card_bg = 'fill="#F0F4FA" stroke="#1F407A" stroke-width="2"'
        l3_hdr = f"""
    <rect x="0" y="0" width="{l3_w}" height="28" rx="4" fill="#1F407A"/>
    <rect x="0" y="24" width="{l3_w}" height="4" fill="#1F407A"/>
    <rect x="14" y="5" width="76" height="18" rx="3" fill="#FFFFFF" opacity="0.2"/>
    <text x="52" y="18" font-size="9.5" font-weight="bold" fill="#FFFFFF" text-anchor="middle">1–50 Hz</text>
    <text x="102" y="19" font-size="12" font-weight="bold" fill="#FFFFFF">LAYER 3 · THE BRAIN: COGNITIVE DELIBERATION TIER [Linux MPU / Edge NPU]</text>
    <text x="890" y="19" font-size="10.5" font-weight="bold" fill="#93C5FD" text-anchor="end">{l3_sub}</text>"""
    else:
        l3_card_bg = 'fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"'
        l3_hdr = f"""
    <rect x="14" y="8" width="76" height="18" rx="3" fill="#E2E8F0"/>
    <text x="52" y="21" font-size="9.5" font-weight="bold" fill="#475569" text-anchor="middle">1–50 Hz</text>
    <text x="102" y="21" font-size="12" font-weight="bold" fill="#475569">LAYER 3 · THE BRAIN: COGNITIVE DELIBERATION TIER [Linux MPU / Edge NPU]</text>
    <text x="890" y="21" font-size="10.5" fill="#64748B" text-anchor="end">{l3_sub}</text>"""

    # Layer 3 subcards & inter-card arrows
    l3_cards_svg = []
    card_w3 = 161
    gap3 = 18
    for i, (cid, title, l1, l2) in enumerate(L3_CARDS):
        x = i * (card_w3 + gap3)
        is_hero = (own_card == cid)
        c_svg = render_subcard(cid, title, l1, l2, x, 0, card_w3, 68, is_hero, is_l3_active, own_card)
        l3_cards_svg.append(c_svg)
        if i < 4:
            arr_x1 = x + card_w3
            arr_x2 = arr_x1 + gap3 - 3
            arr_col = "#1F407A" if (is_hero or (own_card is None and is_l3_active)) else "#CBD5E1"
            marker = "url(#arr-navy)" if (is_hero or (own_card is None and is_l3_active)) else "url(#arr-slate)"
            arr_w = "1.8" if (is_hero or (own_card is None and is_l3_active)) else "1.2"
            l3_cards_svg.append(f'<line x1="{arr_x1}" y1="34" x2="{arr_x2}" y2="34" stroke="{arr_col}" stroke-width="{arr_w}" marker-end="{marker}"/>')

    # ==================== BUILD LAYER 2 ====================
    l2_w, l2_h = 905, 114
    if is_l2_active:
        l2_card_bg = 'fill="#F0F4FA" stroke="#1F407A" stroke-width="2"'
        l2_hdr = f"""
    <rect x="0" y="0" width="{l2_w}" height="28" rx="4" fill="#1F407A"/>
    <rect x="0" y="24" width="{l2_w}" height="4" fill="#1F407A"/>
    <rect x="14" y="5" width="76" height="18" rx="3" fill="#FFFFFF" opacity="0.2"/>
    <text x="52" y="18" font-size="9.5" font-weight="bold" fill="#FFFFFF" text-anchor="middle">1000 Hz</text>
    <text x="102" y="19" font-size="12" font-weight="bold" fill="#FFFFFF">LAYER 2 · THE NERVOUS SYSTEM: REAL-TIME SAFETY &amp; TIMING TIER [Bare-Metal MCU]</text>
    <text x="890" y="19" font-size="10.5" font-weight="bold" fill="#93C5FD" text-anchor="end">{l2_sub}</text>"""
    else:
        l2_card_bg = 'fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"'
        l2_hdr = f"""
    <rect x="14" y="8" width="76" height="18" rx="3" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="0.8"/>
    <text x="52" y="21" font-size="9.5" font-weight="bold" fill="#A51C30" text-anchor="middle">1000 Hz</text>
    <text x="102" y="21" font-size="12" font-weight="bold" fill="#475569">LAYER 2 · THE NERVOUS SYSTEM: REAL-TIME SAFETY &amp; TIMING TIER [Bare-Metal MCU]</text>
    <text x="890" y="21" font-size="10.5" fill="#64748B" text-anchor="end">{l2_sub}</text>"""

    l2_cards_svg = []
    card_w2 = 279
    gap2 = 20
    for i, (cid, title, l1, l2) in enumerate(L2_CARDS):
        x = i * (card_w2 + gap2)
        is_hero = (own_card == cid)
        c_svg = render_subcard(cid, title, l1, l2, x, 0, card_w2, 64, is_hero, is_l2_active, own_card)
        l2_cards_svg.append(c_svg)
        if i < 2:
            arr_x1 = x + card_w2
            arr_x2 = arr_x1 + gap2 - 4
            arr_col = "#1F407A" if (is_hero or (own_card is None and is_l2_active)) else "#475569"
            marker = "url(#arr-navy)" if (is_hero or (own_card is None and is_l2_active)) else "url(#arr-slate)"
            arr_w = "1.8" if (is_hero or (own_card is None and is_l2_active)) else "1.2"
            l2_cards_svg.append(f'<line x1="{arr_x1}" y1="32" x2="{arr_x2}" y2="32" stroke="{arr_col}" stroke-width="{arr_w}" marker-end="{marker}"/>')

    # ==================== BUILD LAYER 1 ====================
    l1_w, l1_h = 905, 114
    if is_l1_active:
        l1_card_bg = 'fill="#F0F4FA" stroke="#1F407A" stroke-width="2"'
        l1_hdr = f"""
    <rect x="0" y="0" width="{l1_w}" height="28" rx="4" fill="#1F407A"/>
    <rect x="0" y="24" width="{l1_w}" height="4" fill="#1F407A"/>
    <rect x="14" y="5" width="76" height="18" rx="3" fill="#FFFFFF" opacity="0.2"/>
    <text x="52" y="18" font-size="9" font-weight="bold" fill="#FFFFFF" text-anchor="middle">Continuous</text>
    <text x="102" y="19" font-size="12" font-weight="bold" fill="#FFFFFF">LAYER 1 · THE PHYSICAL BODY &amp; CONTINUOUS PLANT [Dynamical Mechanics]</text>
    <text x="890" y="19" font-size="10.5" font-weight="bold" fill="#93C5FD" text-anchor="end">{l1_sub}</text>"""
    else:
        l1_card_bg = 'fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"'
        l1_hdr = f"""
    <rect x="14" y="8" width="76" height="18" rx="3" fill="#E2E8F0"/>
    <text x="52" y="21" font-size="9.5" font-weight="bold" fill="#2D3748" text-anchor="middle">Continuous</text>
    <text x="102" y="21" font-size="12" font-weight="bold" fill="#475569">LAYER 1 · THE PHYSICAL BODY &amp; CONTINUOUS PLANT [Dynamical Mechanics]</text>
    <text x="890" y="21" font-size="10.5" fill="#64748B" text-anchor="end">{l1_sub}</text>"""

    l1_cards_svg = []
    card_w1 = 279
    gap1 = 20
    for i, (cid, title, l1, l2) in enumerate(L1_CARDS):
        x = i * (card_w1 + gap1)
        is_hero = (own_card == cid)
        c_svg = render_subcard(cid, title, l1, l2, x, 0, card_w1, 64, is_hero, is_l1_active, own_card)
        l1_cards_svg.append(c_svg)
        if i < 2:
            arr_x1 = x + card_w1
            arr_x2 = arr_x1 + gap1 - 4
            arr_col = "#1F407A" if (is_hero or (own_card is None and is_l1_active)) else "#475569"
            marker = "url(#arr-navy)" if (is_hero or (own_card is None and is_l1_active)) else "url(#arr-slate)"
            arr_w = "1.8" if (is_hero or (own_card is None and is_l1_active)) else "1.2"
            l1_cards_svg.append(f'<line x1="{arr_x1}" y1="32" x2="{arr_x2}" y2="32" stroke="{arr_col}" stroke-width="{arr_w}" marker-end="{marker}"/>')

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="1000" height="600"
     font-family="Helvetica Neue, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arr-navy" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#1F407A" />
    </marker>
    <marker id="arr-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arr-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#A51C30" />
    </marker>
  </defs>

  <!-- Canvas Background -->
  <rect width="1000" height="600" fill="#FFFFFF" rx="4"/>

  <!-- Title & Active Chapter Badge -->
  <g transform="translate(45, 27)">
    <text x="0" y="0" font-size="15" font-weight="bold" fill="#1F407A" letter-spacing="0.03em">PHYSICAL AI SYSTEMS ARCHITECTURE STACK</text>
    <rect x="{badge_x - 45}" y="-15" width="{badge_w}" height="24" rx="12" fill="#1F407A"/>
    <circle cx="{badge_x - 45 + 14}" cy="-3" r="3.5" fill="#93C5FD"/>
    <text x="{badge_x - 45 + badge_w / 2 + 5}" y="-0.5" font-size="9.5" font-weight="bold" fill="#FFFFFF" text-anchor="middle" letter-spacing="0.04em">{badge_text}</text>
  </g>

  <!-- ==================== LAYER 4: GOVERNANCE ==================== -->
  <g transform="translate(45, 46)">
    <rect x="0" y="0" width="{l4_w}" height="{l4_h}" rx="4" {l4_card_bg}/>
    {l4_hdr}
    <g transform="translate(14, {34 if is_l4_active else 32})">
      {''.join(l4_cards_svg)}
    </g>
  </g>

  <!-- ==================== LAYER 3: BRAIN ==================== -->
  <g transform="translate(45, 158)">
    <rect x="0" y="0" width="{l3_w}" height="{l3_h}" rx="4" {l3_card_bg}/>
    {l3_hdr}
    <g transform="translate(14, 40)">
      {''.join(l3_cards_svg)}
    </g>
  </g>

  <!-- ==================== CAUSAL BOUNDARY BRIDGE ==================== -->
  <g transform="translate(45, 290)">
    <line x1="20" y1="13" x2="155" y2="13" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="6,4"/>
    <line x1="750" y1="13" x2="885" y2="13" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="6,4"/>
    <rect x="155" y="0" width="595" height="26" rx="13" fill="#FFFFFF" stroke="#A51C30" stroke-width="2"/>
    <text x="452.5" y="17" font-size="10.5" font-weight="bold" fill="#A51C30" text-anchor="middle" letter-spacing="0.04em">
      PROPOSAL–PERMISSION CAUSAL BOUNDARY · Untrusted Proposals û → Verified Commands u*
    </text>
  </g>

  <!-- ==================== LAYER 2: NERVOUS SYSTEM ==================== -->
  <g transform="translate(45, 328)">
    <rect x="0" y="0" width="{l2_w}" height="{l2_h}" rx="4" {l2_card_bg}/>
    {l2_hdr}
    <g transform="translate(14, {42 if is_l2_active else 40})">
      {''.join(l2_cards_svg)}
    </g>
  </g>

  <!-- ==================== LAYER 1: PHYSICAL BODY ==================== -->
  <g transform="translate(45, 454)">
    <rect x="0" y="0" width="{l1_w}" height="{l1_h}" rx="4" {l1_card_bg}/>
    {l1_hdr}
    <g transform="translate(14, {42 if is_l1_active else 40})">
      {''.join(l1_cards_svg)}
    </g>
  </g>

  <!-- Right Downward Proposal Arrow (Layer 3 -> Layer 2) -->
  <path fill="none" d="M 965 218 L 965 378" stroke="#1F407A" stroke-width="2" stroke-dasharray="4,2" marker-end="url(#arr-navy)"/>
  <text x="976" y="298" font-size="9.5" font-weight="bold" fill="#1F407A" transform="rotate(90, 976, 298)" text-anchor="middle">Proposal û</text>

  <!-- Right Downward Certified Command Arrow (Layer 2 -> Layer 1) -->
  <path fill="none" d="M 965 408 L 965 508" stroke="#A51C30" stroke-width="2.5" marker-end="url(#arr-red)"/>
  <text x="976" y="458" font-size="9.5" font-weight="bold" fill="#A51C30" transform="rotate(90, 976, 458)" text-anchor="middle">Certified u*</text>

  <!-- Left Upward Sensory Feedback Loop (Layer 1 -> Layer 3) -->
  <path fill="none" d="M 45 510 L 22 510 L 22 218 L 40 218" stroke="#475569" stroke-width="1.8" marker-end="url(#arr-slate)"/>
  <text x="13" y="364" font-size="9.5" font-weight="bold" fill="#475569" transform="rotate(-90, 13, 364)" text-anchor="middle">Endogenous Sensory Shift (o_t+1)</text>
</svg>"""

def build(slug):
    """Generate and write the SVG directly for a given chapter slug."""
    ch_dir = os.path.join(BASE, slug)
    images_svg = os.path.join(ch_dir, "images", "svg")
    os.makedirs(images_svg, exist_ok=True)
    
    out_svg = os.path.join(images_svg, "fig_locator.svg")
    svg_content = build_svg(slug)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"  ok   {slug} -> {out_svg}")
    return True

if __name__ == "__main__":
    targets = sys.argv[1:] or list(CHAPTERS)
    bad = [s for s in targets if s not in CHAPTERS]
    if bad:
        sys.exit(f"unknown chapter slug(s): {', '.join(bad)}")
    print(f"Regenerating {len(targets)} locator stack figures (pure SVG engine)")
    failed = [s for s in targets if not build(s)]
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")
    print("All locator stack figures regenerated successfully.")
