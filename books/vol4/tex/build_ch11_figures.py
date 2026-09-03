#!/usr/bin/env python3
"""
Generate publication-grade vector figures for Chapter 11: Planning
- fig11_action_chunk_seam_continuity: Action Chunk Seam Boundary & C^2 Jerk Continuity
- fig11_seam_timeline_fallback: Seam Latency Milestones & Fallback Deceleration Dynamics

Outputs both SVG and vector PDF directly into book/chapters/11-planning/figures/
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CH11_FIG_DIR = os.path.join(BOOK_DIR, "chapters", "11-planning", "figures")
os.makedirs(CH11_FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. FIG 11.1: ACTION CHUNK SEAM CONTINUITY & TRANSMISSION SHOCK
# -----------------------------------------------------------------------------
FIG11_SEAM_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 660" width="100%" height="100%">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .title { font-weight: 800; font-size: 16px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.5px; }
      .subtitle { font-size: 11.5px; fill: #475569; text-anchor: middle; }
      .card-title { font-weight: 700; font-size: 12px; }
      .axis-label { font-size: 10px; fill: #64748B; font-weight: 600; }
      .plot-title { font-weight: 700; font-size: 11px; fill: #1E293B; }
      .body-text { font-size: 10px; fill: #334155; line-height: 1.35; }
      .mono-text { font-family: "SF Mono", Menlo, Consolas, Monaco, "Liberation Mono", monospace; font-size: 9.5px; font-weight: 600; }
      .badge-text { font-weight: 700; font-size: 9.5px; text-anchor: middle; }
      .annot-text { font-size: 9px; font-weight: 600; }
    </style>

    <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A"/>
    </marker>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563EB"/>
    </marker>
    <marker id="arr-teal" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0D9488"/>
    </marker>
    <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
    </marker>
    <marker id="arr-amber" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#D97706"/>
    </marker>

    <filter id="card-shadow" x="-2%" y="-2%" width="104%" height="106%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.06"/>
    </filter>
  </defs>

  <!-- Outer Enclosure -->
  <rect x="8" y="8" width="964" height="644" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>

  <!-- Top Title Banner -->
  <rect x="20" y="20" width="940" height="52" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.2"/>
  <text x="490" y="42" class="title">ACTION CHUNK SEAM BOUNDARY: DERIVATIVE CONTINUITY VS. TRANSMISSION SHOCK</text>
  <text x="490" y="59" class="subtitle">Systems Dynamics of Unchecked C⁰ Concatenation vs. Jerk-Continuous C² Quintic Spline Trajectory Blending</text>

  <!-- TOP ROW: ARCHITECTURAL DATAFLOW AT REPLANNING HANDOFF -->
  <!-- Box 1: Action Chunk A -->
  <rect x="20" y="84" width="210" height="74" rx="6" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2" filter="url(#card-shadow)"/>
  <rect x="20" y="84" width="210" height="20" rx="6" fill="#2563EB"/>
  <text x="125" y="98" class="badge-text" fill="#FFFFFF">ACTIVE ACTION CHUNK A</text>
  <text x="30" y="120" class="mono-text" fill="#1E293B">Horizon: t ∈ [0, 50 ms] (20 Hz)</text>
  <text x="30" y="135" class="mono-text" fill="#1E293B">q_A(50) = 0.850 rad</text>
  <text x="30" y="150" class="mono-text" fill="#A51C30">q̇_A(50) = -0.45 rad/s (Decel)</text>

  <!-- Arrow from A to replan dispatch -->
  <line x1="230" y1="121" x2="268" y2="121" stroke="#2563EB" stroke-width="1.6" marker-end="url(#arr-blue)"/>

  <!-- Box 2: Latency & Replanning Engine -->
  <rect x="272" y="84" width="220" height="74" rx="6" fill="#FFFBEB" stroke="#D97706" stroke-width="1.2" filter="url(#card-shadow)"/>
  <rect x="272" y="84" width="220" height="20" rx="6" fill="#D97706"/>
  <text x="382" y="98" class="badge-text" fill="#FFFFFF">VLA POLICY INFERENCE (BRAIN)</text>
  <text x="282" y="120" class="mono-text" fill="#1E293B">Request Lead: T_req = 0 ms</text>
  <text x="282" y="135" class="mono-text" fill="#1E293B">Tail Latency: L ~ P99 = 48 ms</text>
  <text x="282" y="150" class="mono-text" fill="#D97706">Chunk B arrives at t = 48 ms</text>

  <!-- Split Arrow to Decision Paths -->
  <path d="M 492 110 L 528 98" fill="none" stroke="#A51C30" stroke-width="1.6" stroke-dasharray="3,2" marker-end="url(#arr-crimson)"/>
  <path d="M 492 132 L 528 144" fill="none" stroke="#0D9488" stroke-width="1.6" marker-end="url(#arr-teal)"/>

  <!-- Box 3A: Naive C0 Path -->
  <rect x="532" y="82" width="200" height="36" rx="5" fill="#FEF2F2" stroke="#A51C30" stroke-width="1.2"/>
  <text x="542" y="96" class="card-title" fill="#A51C30">CASE A: Naive C⁰ Concatenation</text>
  <text x="542" y="110" class="annot-text" fill="#A51C30">Pos matched; Δq̇ = +0.60 rad/s step!</text>

  <!-- Box 3B: C2 Spline Path -->
  <rect x="532" y="126" width="200" height="36" rx="5" fill="#ECFDF5" stroke="#0D9488" stroke-width="1.2"/>
  <text x="542" y="140" class="card-title" fill="#0D9488">CASE B: C² Jerk-Continuous Spline</text>
  <text x="542" y="154" class="annot-text" fill="#0D9488">Quintic bridge over T_blend = 20 ms</text>

  <!-- Connectors to Hardware -->
  <line x1="732" y1="100" x2="758" y2="110" stroke="#A51C30" stroke-width="1.4" stroke-dasharray="3,2"/>
  <line x1="732" y1="144" x2="758" y2="130" stroke="#0D9488" stroke-width="1.4"/>

  <!-- Box 4: Nervous System & Transmission -->
  <rect x="760" y="84" width="200" height="74" rx="6" fill="#F1F5F9" stroke="#475569" stroke-width="1.2" filter="url(#card-shadow)"/>
  <rect x="760" y="84" width="200" height="20" rx="6" fill="#475569"/>
  <text x="860" y="98" class="badge-text" fill="#FFFFFF">1 kHz SERVO LOOP &amp; PLANT</text>
  <text x="770" y="120" class="mono-text" fill="#1E293B">Servo Period: T_c = 1.0 ms</text>
  <text x="770" y="135" class="mono-text" fill="#1E293B">Reflected Inertia: J = 2.7 kg·m²</text>
  <text x="770" y="150" class="mono-text" fill="#1E293B">Pin Shear Limit: τ = 285 N·m</text>

  <!-- ========================================================================= -->
  <!-- COMPARATIVE WAVEFORM PANELS: LEFT (CASE A: C0) vs RIGHT (CASE B: C2) -->
  <!-- ========================================================================= -->

  <!-- LEFT COLUMN BACKGROUND (CASE A - RED ACCENT) -->
  <rect x="20" y="168" width="460" height="400" rx="6" fill="#FFFFFF" stroke="#FCA5A5" stroke-width="1.2"/>
  <rect x="20" y="168" width="460" height="24" rx="6" fill="#FEF2F2"/>
  <text x="30" y="184" class="plot-title" fill="#A51C30">CASE A: UNCHECKED C⁰ POSITION-ONLY CONCATENATION</text>
  <rect x="390" y="171" width="80" height="18" rx="3" fill="#A51C30"/>
  <text x="430" y="183" class="badge-text" fill="#FFFFFF">C⁰ FRACTURE</text>

  <!-- RIGHT COLUMN BACKGROUND (CASE B - TEAL ACCENT) -->
  <rect x="500" y="168" width="460" height="400" rx="6" fill="#FFFFFF" stroke="#6EE7B7" stroke-width="1.2"/>
  <rect x="500" y="168" width="460" height="24" rx="6" fill="#ECFDF5"/>
  <text x="510" y="184" class="plot-title" fill="#0D9488">CASE B: JERK-CONTINUOUS C² QUINTIC BLENDED SPLINE</text>
  <rect x="870" y="171" width="80" height="18" rx="3" fill="#0D9488"/>
  <text x="910" y="183" class="badge-text" fill="#FFFFFF">C² VERIFIED</text>

  <!-- ======================== ROW 1: POSITION q(t) ======================== -->
  <!-- Left Panel: Pos A -->
  <g transform="translate(0, 0)">
    <line x1="70" y1="272" x2="460" y2="272" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="70" y1="202" x2="70" y2="272" stroke="#CBD5E1" stroke-width="1"/>
    <text x="62" y="206" class="axis-label" text-anchor="end">0.86</text>
    <text x="62" y="240" class="axis-label" text-anchor="end">0.85</text>
    <text x="62" y="274" class="axis-label" text-anchor="end">0.84</text>
    <text x="25" y="242" class="axis-label" transform="rotate(-90 25 242)" text-anchor="middle">Position q [rad]</text>

    <!-- Seam line at t = 50 ms (x = 265) -->
    <line x1="265" y1="202" x2="265" y2="272" stroke="#A51C30" stroke-width="1" stroke-dasharray="3,3"/>
    
    <!-- Waveform: Chunk A decel, then Chunk B accel upward (sharp kink at seam) -->
    <path d="M 70 215 Q 167 232, 265 240 L 265 240 Q 362 230, 460 210" fill="none" stroke="#2563EB" stroke-width="2"/>
    <circle cx="265" cy="240" r="4" fill="#A51C30"/>
    <text x="272" y="234" class="annot-text" fill="#A51C30">Kink at Seam (Slope mismatch)</text>
    <text x="120" y="222" class="annot-text" fill="#2563EB">Chunk A</text>
    <text x="390" y="222" class="annot-text" fill="#2563EB">Chunk B</text>
  </g>

  <!-- Right Panel: Pos B -->
  <g transform="translate(0, 0)">
    <line x1="550" y1="272" x2="940" y2="272" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="550" y1="202" x2="550" y2="272" stroke="#CBD5E1" stroke-width="1"/>
    <text x="542" y="206" class="axis-label" text-anchor="end">0.86</text>
    <text x="542" y="240" class="axis-label" text-anchor="end">0.85</text>
    <text x="542" y="274" class="axis-label" text-anchor="end">0.84</text>
    <text x="505" y="242" class="axis-label" transform="rotate(-90 505 242)" text-anchor="middle">Position q [rad]</text>

    <!-- Blend Region at t in [40, 60] ms (x in [705, 785]) -->
    <rect x="705" y="202" width="80" height="70" fill="#ECFDF5" opacity="0.7"/>
    <line x1="745" y1="202" x2="745" y2="272" stroke="#0D9488" stroke-width="1" stroke-dasharray="3,3"/>

    <!-- Waveform: Quintic blend bridge connecting smoothly -->
    <path d="M 550 215 Q 630 230, 705 237 C 725 241, 765 235, 785 228 Q 862 216, 940 210" fill="none" stroke="#0D9488" stroke-width="2.2"/>
    <text x="745" y="214" class="annot-text" fill="#0D9488" text-anchor="middle">T_blend = 20 ms Bridge</text>
    <text x="600" y="222" class="annot-text" fill="#2563EB">Chunk A</text>
    <text x="870" y="222" class="annot-text" fill="#2563EB">Chunk B</text>
  </g>

  <!-- ======================== ROW 2: VELOCITY q̇(t) ======================== -->
  <!-- Left Panel: Vel A -->
  <g transform="translate(0, 0)">
    <line x1="70" y1="365" x2="460" y2="365" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="70" y1="295" x2="70" y2="365" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="70" y1="340" x2="460" y2="340" stroke="#E2E8F0" stroke-width="0.8" stroke-dasharray="2,2"/>
    <text x="62" y="305" class="axis-label" text-anchor="end">+0.30</text>
    <text x="62" y="322" class="axis-label" text-anchor="end">+0.15</text>
    <text x="62" y="343" class="axis-label" text-anchor="end">0.00</text>
    <text x="62" y="362" class="axis-label" text-anchor="end">-0.45</text>
    <text x="25" y="332" class="axis-label" transform="rotate(-90 25 332)" text-anchor="middle">Velocity q̇ [rad/s]</text>

    <line x1="265" y1="295" x2="265" y2="365" stroke="#A51C30" stroke-width="1" stroke-dasharray="3,3"/>

    <!-- Waveform: Chunk A at -0.45, jumps across 1 ms to +0.15 -->
    <path d="M 70 350 L 265 360" fill="none" stroke="#2563EB" stroke-width="2"/>
    <!-- STEP JUMP -->
    <line x1="265" y1="360" x2="267" y2="320" stroke="#A51C30" stroke-width="2.5"/>
    <path d="M 267 320 L 460 305" fill="none" stroke="#2563EB" stroke-width="2"/>

    <!-- Step jump annotation -->
    <rect x="275" y="325" width="175" height="30" rx="3" fill="#FEF2F2" stroke="#A51C30" stroke-width="0.8"/>
    <text x="280" y="337" class="mono-text" fill="#A51C30">Δq̇ = +0.60 rad/s step!</text>
    <text x="280" y="349" class="annot-text" fill="#A51C30">Across 1 ms servo period (Tc)</text>
  </g>

  <!-- Right Panel: Vel B -->
  <g transform="translate(0, 0)">
    <line x1="550" y1="365" x2="940" y2="365" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="550" y1="295" x2="550" y2="365" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="550" y1="340" x2="940" y2="340" stroke="#E2E8F0" stroke-width="0.8" stroke-dasharray="2,2"/>
    <text x="542" y="305" class="axis-label" text-anchor="end">+0.30</text>
    <text x="542" y="322" class="axis-label" text-anchor="end">+0.15</text>
    <text x="542" y="343" class="axis-label" text-anchor="end">0.00</text>
    <text x="542" y="362" class="axis-label" text-anchor="end">-0.45</text>
    <text x="505" y="332" class="axis-label" transform="rotate(-90 505 332)" text-anchor="middle">Velocity q̇ [rad/s]</text>

    <rect x="705" y="295" width="80" height="70" fill="#ECFDF5" opacity="0.7"/>
    <line x1="745" y1="295" x2="745" y2="365" stroke="#0D9488" stroke-width="1" stroke-dasharray="3,3"/>

    <!-- Waveform: Smooth transition curve -->
    <path d="M 550 350 L 705 360 C 730 360, 760 320, 785 320 L 940 305" fill="none" stroke="#0D9488" stroke-width="2.2"/>
    <text x="792" y="345" class="annot-text" fill="#0D9488">Smooth C¹ Velocity (No Step)</text>
  </g>

  <!-- ======================== ROW 3: ACCELERATION & TORQUE ======================== -->
  <!-- Left Panel: Acc & Torque A -->
  <g transform="translate(0, 0)">
    <line x1="70" y1="465" x2="460" y2="465" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="70" y1="390" x2="70" y2="465" stroke="#CBD5E1" stroke-width="1"/>
    <text x="62" y="398" class="axis-label" text-anchor="end">324</text>
    <text x="62" y="415" class="axis-label" text-anchor="end">285</text>
    <text x="62" y="438" class="axis-label" text-anchor="end">110</text>
    <text x="62" y="452" class="axis-label" text-anchor="end">45</text>
    <text x="62" y="468" class="axis-label" text-anchor="end">0</text>
    <text x="25" y="428" class="axis-label" transform="rotate(-90 25 428)" text-anchor="middle">Torque τ [N·m]</text>

    <!-- Threshold lines -->
    <line x1="70" y1="452" x2="460" y2="452" stroke="#059669" stroke-width="0.8" stroke-dasharray="3,2"/>
    <text x="375" y="450" class="annot-text" fill="#059669">Continuous Rating (45 N·m)</text>

    <line x1="70" y1="436" x2="460" y2="436" stroke="#D97706" stroke-width="0.8" stroke-dasharray="3,2"/>
    <text x="365" y="434" class="annot-text" fill="#D97706">Peak Ceiling (110 N·m)</text>

    <line x1="70" y1="412" x2="460" y2="412" stroke="#A51C30" stroke-width="1.2" stroke-dasharray="4,2"/>
    <text x="340" y="410" class="annot-text" fill="#A51C30" font-weight="700">Pin Shear Yield Limit (285 N·m)</text>

    <line x1="265" y1="390" x2="265" y2="465" stroke="#A51C30" stroke-width="1" stroke-dasharray="3,3"/>

    <!-- Waveform: Baseline ~25 N*m, then massive spike to 324 N*m at seam! -->
    <path d="M 70 458 L 263 458 L 265 394 L 268 458 L 460 458" fill="none" stroke="#A51C30" stroke-width="2.5"/>
    
    <!-- Fracture explosion callout -->
    <circle cx="265" cy="394" r="5" fill="#A51C30"/>
    <rect x="275" y="390" width="180" height="34" rx="3" fill="#A51C30"/>
    <text x="280" y="403" class="badge-text" fill="#FFFFFF" text-anchor="start">💥 SHEAR FRACTURE (324 N·m)</text>
    <text x="280" y="416" class="mono-text" fill="#FEE2E2" font-size="8.5px">q̈_cmd = 600 rad/s² (Exceeds 285 N·m)</text>
  </g>

  <!-- Right Panel: Acc & Torque B -->
  <g transform="translate(0, 0)">
    <line x1="550" y1="465" x2="940" y2="465" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="550" y1="390" x2="550" y2="465" stroke="#CBD5E1" stroke-width="1"/>
    <text x="542" y="398" class="axis-label" text-anchor="end">324</text>
    <text x="542" y="415" class="axis-label" text-anchor="end">285</text>
    <text x="542" y="438" class="axis-label" text-anchor="end">110</text>
    <text x="542" y="452" class="axis-label" text-anchor="end">45</text>
    <text x="542" y="468" class="axis-label" text-anchor="end">0</text>
    <text x="505" y="428" class="axis-label" transform="rotate(-90 505 428)" text-anchor="middle">Torque τ [N·m]</text>

    <!-- Threshold lines -->
    <line x1="550" y1="452" x2="940" y2="452" stroke="#059669" stroke-width="0.8" stroke-dasharray="3,2"/>
    <text x="855" y="450" class="annot-text" fill="#059669">Continuous Rating (45 N·m)</text>
    <line x1="550" y1="436" x2="940" y2="436" stroke="#D97706" stroke-width="0.8" stroke-dasharray="3,2"/>
    <line x1="550" y1="412" x2="940" y2="412" stroke="#A51C30" stroke-width="0.8" stroke-dasharray="4,2"/>

    <rect x="705" y="390" width="80" height="75" fill="#ECFDF5" opacity="0.7"/>
    <line x1="745" y1="390" x2="745" y2="465" stroke="#0D9488" stroke-width="1" stroke-dasharray="3,3"/>

    <!-- Waveform: Smooth bell-curve torque peaking at ~35 N*m (y = 455) -->
    <path d="M 550 458 L 705 458 C 725 458, 735 450, 745 450 C 755 450, 765 458, 785 458 L 940 458" fill="none" stroke="#0D9488" stroke-width="2.2"/>
    
    <rect x="755" y="428" width="180" height="24" rx="3" fill="#ECFDF5" stroke="#0D9488" stroke-width="0.8"/>
    <text x="760" y="440" class="annot-text" fill="#0D9488">Peak Torque τ_max = 38 N·m</text>
    <text x="760" y="449" class="annot-text" fill="#059669">Safe Margin: 15% under 45 N·m limit</text>
  </g>

  <!-- ======================== ROW 4: JERK & TRANSMISSION IMPACT ======================== -->
  <!-- Left Panel: Jerk A -->
  <g transform="translate(0, 0)">
    <line x1="70" y1="545" x2="460" y2="545" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="70" y1="485" x2="70" y2="545" stroke="#CBD5E1" stroke-width="1"/>
    <text x="62" y="490" class="axis-label" text-anchor="end">+∞</text>
    <text x="62" y="548" class="axis-label" text-anchor="end">0</text>
    <text x="25" y="515" class="axis-label" transform="rotate(-90 25 515)" text-anchor="middle">Jerk q⃛ [rad/s³]</text>

    <!-- Time Axis Ticks -->
    <text x="70" y="558" class="axis-label" text-anchor="middle">0 ms</text>
    <text x="167" y="558" class="axis-label" text-anchor="middle">25 ms</text>
    <text x="265" y="558" class="axis-label" text-anchor="middle" font-weight="700" fill="#A51C30">50 ms (Seam)</text>
    <text x="362" y="558" class="axis-label" text-anchor="middle">75 ms</text>
    <text x="460" y="558" class="axis-label" text-anchor="middle">100 ms</text>

    <!-- Dirac Delta Spike -->
    <path d="M 70 545 L 264 545 L 265 488 L 266 545 L 460 545" fill="none" stroke="#A51C30" stroke-width="2.5"/>
    <line x1="265" y1="488" x2="265" y2="480" stroke="#A51C30" stroke-width="2" marker-end="url(#arr-crimson)"/>
    <text x="272" y="498" class="annot-text" fill="#A51C30">Dirac Impulse δ(t) (Infinite Jerk)</text>
    <text x="272" y="510" class="body-text" fill="#64748B">• Bearing Brinelling · Cycloidal Tooth Impact</text>
  </g>

  <!-- Right Panel: Jerk B -->
  <g transform="translate(0, 0)">
    <line x1="550" y1="545" x2="940" y2="545" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="550" y1="485" x2="550" y2="545" stroke="#CBD5E1" stroke-width="1"/>
    <text x="542" y="490" class="axis-label" text-anchor="end">4.5k</text>
    <text x="542" y="515" class="axis-label" text-anchor="end">0</text>
    <text x="542" y="542" class="axis-label" text-anchor="end">-4.5k</text>
    <text x="505" y="515" class="axis-label" transform="rotate(-90 505 515)" text-anchor="middle">Jerk q⃛ [rad/s³]</text>

    <!-- Time Axis Ticks -->
    <text x="550" y="558" class="axis-label" text-anchor="middle">0 ms</text>
    <text x="647" y="558" class="axis-label" text-anchor="middle">25 ms</text>
    <text x="745" y="558" class="axis-label" text-anchor="middle" font-weight="700" fill="#0D9488">50 ms (Seam)</text>
    <text x="842" y="558" class="axis-label" text-anchor="middle">75 ms</text>
    <text x="940" y="558" class="axis-label" text-anchor="middle">100 ms</text>

    <!-- Waveform: Bounded continuous jerk S-curve -->
    <path d="M 550 515 L 705 515 C 715 492, 735 492, 745 515 C 755 538, 775 538, 785 515 L 940 515" fill="none" stroke="#0D9488" stroke-width="2.2"/>
    <text x="792" y="498" class="annot-text" fill="#0D9488">Continuous, Bounded Jerk (C² Spline)</text>
    <text x="792" y="510" class="body-text" fill="#64748B">• Zero Structural Resonance Excitation</text>
  </g>

  <!-- ========================================================================= -->
  <!-- BOTTOM FOOTER: SYSTEMS LESSON CALLOUT -->
  <!-- ========================================================================= -->
  <rect x="20" y="580" width="940" height="60" rx="6" fill="#1E293B"/>
  <text x="35" y="601" font-size="11px" font-weight="700" fill="#38BDF8">SYSTEMS IMPLICATION: CONCATENATION REQUIRES HIGHER-ORDER BOUNDARY VALIDATION</text>
  <text x="35" y="618" font-size="10px" fill="#E2E8F0">When an action chunk finishes, splicing a replacement that matches only position (C⁰) creates an instantaneous velocity step Δq̇.</text>
  <text x="35" y="631" font-size="10px" fill="#94A3B8">Across a 1.0 ms servo cycle (T_c), this step demands an acceleration impulse q̈ = Δq̇/T_c = 600 rad/s², transmitting 324 N·m through reflected inertia (J = 2.7 kg·m²) and shearing the drive pin.</text>
</svg>
"""

# -----------------------------------------------------------------------------
# 2. FIG 11.2: SEAM TIMELINE & FALLBACK DECELERATION DYNAMICS
# -----------------------------------------------------------------------------
FIG11_TIMELINE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 600" width="100%" height="100%">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .title { font-weight: 800; font-size: 16px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.5px; }
      .subtitle { font-size: 11.5px; fill: #475569; text-anchor: middle; }
      .swimlane-title { font-weight: 700; font-size: 12px; }
      .milestone-label { font-size: 10.5px; font-weight: 700; text-anchor: middle; }
      .time-label { font-family: "SF Mono", Menlo, Consolas, Monaco, monospace; font-size: 9.5px; font-weight: 700; text-anchor: middle; }
      .body-text { font-size: 10px; fill: #334155; line-height: 1.35; }
      .mono-text { font-family: "SF Mono", Menlo, Consolas, Monaco, monospace; font-size: 9.5px; font-weight: 600; }
      .badge-text { font-weight: 700; font-size: 9.5px; text-anchor: middle; }
      .annot-text { font-size: 9.5px; font-weight: 600; }
      .card-hdr { font-weight: 700; font-size: 10.5px; }
    </style>

    <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A"/>
    </marker>
    <marker id="arr-teal" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0D9488"/>
    </marker>
    <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
    </marker>
    <marker id="arr-amber" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#D97706"/>
    </marker>

    <filter id="card-shadow" x="-2%" y="-2%" width="104%" height="106%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.06"/>
    </filter>
  </defs>

  <!-- Outer Enclosure -->
  <rect x="8" y="8" width="964" height="584" rx="10" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>

  <!-- Top Title Banner -->
  <rect x="20" y="20" width="940" height="52" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.2"/>
  <text x="490" y="42" class="title">ACTION CHUNK SEAM TIMELINE &amp; FALLBACK DECELERATION DYNAMICS</text>
  <text x="490" y="59" class="subtitle">Temporal Milestones, Replacement Ingestion Windows, and Clearance-Bounded Braking Contracts</text>

  <!-- TOP TIMELINE AXIS BAR -->
  <rect x="20" y="82" width="940" height="52" rx="6" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>
  
  <!-- Axis line -->
  <line x1="180" y1="110" x2="920" y2="110" stroke="#475569" stroke-width="2" marker-end="url(#arr-navy)"/>
  
  <!-- Milestone 1: t_req = 0 ms (x = 180) -->
  <line x1="180" y1="92" x2="180" y2="128" stroke="#2563EB" stroke-width="2"/>
  <circle cx="180" cy="110" r="4.5" fill="#2563EB"/>
  <text x="180" y="102" class="milestone-label" fill="#2563EB">t_req</text>
  <text x="180" y="125" class="time-label" fill="#2563EB">0 ms</text>

  <!-- Milestone 2: P50 arrival = 180 ms (x = 402) -->
  <line x1="402" y1="95" x2="402" y2="125" stroke="#0D9488" stroke-width="1.5" stroke-dasharray="2,2"/>
  <circle cx="402" cy="110" r="3.5" fill="#0D9488"/>
  <text x="402" y="103" class="milestone-label" fill="#0D9488">t_arr (P50)</text>
  <text x="402" y="124" class="time-label" fill="#0D9488">180 ms</text>

  <!-- Milestone 3: t_blend = 300 ms (x = 550) -->
  <line x1="550" y1="92" x2="550" y2="128" stroke="#D97706" stroke-width="2"/>
  <circle cx="550" cy="110" r="4.5" fill="#D97706"/>
  <text x="550" y="102" class="milestone-label" fill="#D97706">t_blend</text>
  <text x="550" y="125" class="time-label" fill="#D97706">300 ms</text>

  <!-- Milestone 4: t_commit = 360 ms (x = 624) -->
  <line x1="624" y1="92" x2="624" y2="128" stroke="#A51C30" stroke-width="2.5"/>
  <circle cx="624" cy="110" r="5" fill="#A51C30"/>
  <text x="624" y="102" class="milestone-label" fill="#A51C30">t_commit</text>
  <text x="624" y="125" class="time-label" fill="#A51C30">360 ms</text>

  <!-- Milestone 5: t_exp = 400 ms (x = 673) -->
  <line x1="673" y1="92" x2="673" y2="128" stroke="#64748B" stroke-width="2"/>
  <circle cx="673" cy="110" r="4.5" fill="#64748B"/>
  <text x="673" y="102" class="milestone-label" fill="#64748B">t_exp</text>
  <text x="673" y="125" class="time-label" fill="#64748B">400 ms</text>

  <!-- Milestone 6: t_term = 560 ms (x = 870) -->
  <line x1="870" y1="92" x2="870" y2="128" stroke="#1E293B" stroke-width="2"/>
  <circle cx="870" cy="110" r="4.5" fill="#1E293B"/>
  <text x="870" y="102" class="milestone-label" fill="#1E293B">t_term</text>
  <text x="870" y="125" class="time-label" fill="#1E293B">560 ms</text>

  <text x="95" y="114" class="card-title" fill="#1F407A">TIME MILESTONES</text>

  <!-- ========================================================================= -->
  <!-- THREE REPLANNING REGIME SWIMLANES -->
  <!-- ========================================================================= -->

  <!-- SWIMLANE 1: ON-TIME REPLANNING (GREEN / TEAL) -->
  <rect x="20" y="142" width="940" height="78" rx="6" fill="#FFFFFF" stroke="#A7F3D0" stroke-width="1.2" filter="url(#card-shadow)"/>
  <rect x="20" y="142" width="150" height="78" rx="6" fill="#ECFDF5"/>
  <text x="95" y="165" class="swimlane-title" fill="#0D9488" text-anchor="middle">REGIME 1</text>
  <text x="95" y="180" class="badge-text" fill="#059669">ON-TIME ARRIVAL</text>
  <text x="95" y="196" class="mono-text" fill="#475569" text-anchor="middle">t_arr ≤ t_blend</text>

  <!-- Execution Bar: Active Chunk A execution up to t_blend -->
  <rect x="180" y="156" width="370" height="26" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1"/>
  <text x="365" y="173" class="badge-text" fill="#2563EB">Active Chunk A Execution (t ∈ [0, 300 ms])</text>
  
  <!-- Blend segment into Chunk B (t: 300 to 400 ms) -->
  <rect x="550" y="156" width="123" height="26" rx="4" fill="#CCFBF1" stroke="#0D9488" stroke-width="1.2"/>
  <text x="611" y="173" class="badge-text" fill="#0D9488">100 ms Blend</text>

  <!-- Chunk B Ongoing -->
  <rect x="673" y="156" width="247" height="26" rx="4" fill="#ECFDF5" stroke="#059669" stroke-width="1"/>
  <text x="796" y="173" class="badge-text" fill="#059669">Chunk B Continuous Execution →</text>

  <text x="180" y="206" class="body-text" fill="#334155">✓ Inference arrives at P50 = 180 ms. Full 100 ms blend window available. Smooth C² quintic interpolation preserves continuous motion.</text>

  <!-- SWIMLANE 2: LATE-BUT-BLENDABLE (AMBER / GOLD) -->
  <rect x="20" y="228" width="940" height="82" rx="6" fill="#FFFFFF" stroke="#FDE68A" stroke-width="1.2" filter="url(#card-shadow)"/>
  <rect x="20" y="228" width="150" height="82" rx="6" fill="#FFFBEB"/>
  <text x="95" y="251" class="swimlane-title" fill="#D97706" text-anchor="middle">REGIME 2</text>
  <text x="95" y="266" class="badge-text" fill="#B45309">LATE-BUT-BLENDABLE</text>
  <text x="95" y="282" class="mono-text" fill="#475569" text-anchor="middle">t_blend &lt; t_arr ≤ t_commit</text>

  <!-- Active Chunk A elongated -->
  <rect x="180" y="242" width="419" height="26" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1"/>
  <text x="389" y="259" class="badge-text" fill="#2563EB">Chunk A Latency Extension (t_arr = 340 ms)</text>

  <!-- Compressed Blend (t: 340 to 400 ms = 60 ms) -->
  <rect x="599" y="242" width="74" height="26" rx="4" fill="#FEF3C7" stroke="#D97706" stroke-width="1.2"/>
  <text x="636" y="259" class="badge-text" fill="#B45309">60 ms Blend</text>

  <rect x="673" y="242" width="247" height="26" rx="4" fill="#ECFDF5" stroke="#059669" stroke-width="1"/>
  <text x="796" y="259" class="badge-text" fill="#059669">Chunk B Continuous Execution →</text>

  <text x="180" y="294" class="body-text" fill="#334155">⚠️ Tail latency delay (P99 = 340 ms). Blend window compressed from 100 ms to 60 ms. Peak acceleration check required: a_peak ~ 6Δq/T_blend².</text>
  <text x="180" y="306" class="body-text" fill="#D97706">If a_peak ≤ a_max (6.0 m/s²), admit compressed blend; if a_peak &gt; a_max, abort to Fallback Suffix immediately.</text>

  <!-- SWIMLANE 3: FALLBACK DECELERATION & NEVER REPLACED (CRIMSON / RED) -->
  <rect x="20" y="318" width="940" height="96" rx="6" fill="#FFFFFF" stroke="#FECACA" stroke-width="1.2" filter="url(#card-shadow)"/>
  <rect x="20" y="318" width="150" height="96" rx="6" fill="#FEF2F2"/>
  <text x="95" y="343" class="swimlane-title" fill="#A51C30" text-anchor="middle">REGIME 3</text>
  <text x="95" y="358" class="badge-text" fill="#991B1B">FALLBACK SUFFIX</text>
  <text x="95" y="374" class="mono-text" fill="#475569" text-anchor="middle">t &gt; t_commit / Lost</text>

  <!-- Active Chunk up to t_commit (x: 180 to 624) -->
  <rect x="180" y="332" width="444" height="26" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1"/>
  <text x="402" y="349" class="badge-text" fill="#2563EB">Active Chunk Execution to Commitment (t_commit = 360 ms)</text>

  <!-- Fallback Deceleration braking phase (x: 624 to 870 = 200 ms) -->
  <rect x="624" y="332" width="246" height="26" rx="4" fill="#FEE2E2" stroke="#A51C30" stroke-width="1.5"/>
  <text x="747" y="349" class="badge-text" fill="#A51C30">PRECOMPUTED FALLBACK DECELERATION (200 ms Braking)</text>

  <!-- Terminal State (x: 870 to 920) -->
  <rect x="870" y="332" width="50" height="26" rx="4" fill="#334155" stroke="#0F172A" stroke-width="1"/>
  <text x="895" y="349" class="badge-text" fill="#FFFFFF">REST</text>

  <!-- Physical Distance Breakdown -->
  <text x="180" y="380" class="mono-text" fill="#1E293B">Distance Traveled: d_total = d_late (72 mm) + d_brake (120 mm) = 192 mm &lt; d_clear (300 mm)</text>
  <text x="180" y="394" class="body-text" fill="#334155">🛑 At t_commit = 360 ms, replanning closes. Controller branches into deterministic C² braking at a_max = 6.0 m/s², bringing mechanism</text>
  <text x="180" y="406" class="body-text" fill="#334155">to zero velocity at t_term = 560 ms with 108 mm remaining margin. Mechanical holding brakes engage; zero current draw.</text>

  <!-- ========================================================================= -->
  <!-- BOTTOM PANEL: WHY "HOLD" IS DANGEROUS VS. STRUCTURED FALLBACK SUFFIX -->
  <!-- ========================================================================= -->
  <rect x="20" y="422" width="940" height="162" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
  
  <rect x="20" y="422" width="940" height="26" rx="6" fill="#1E293B"/>
  <text x="490" y="439" font-size="11.5px" font-weight="700" fill="#F8FAFC" text-anchor="middle">PHYSICAL TAXONOMY: WHY DEFAULT "HOLD" BEHAVIORS FAIL AT SEAM EXPIRY</text>

  <!-- 4 Failure / Success Cards -->
  <!-- Card 1: Hold Torque -->
  <rect x="30" y="456" width="220" height="120" rx="5" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1"/>
  <text x="40" y="474" class="card-hdr" fill="#A51C30">1. Hold Commanded Torque</text>
  <text x="40" y="492" class="body-text" fill="#475569">• Maintains constant motor current</text>
  <text x="40" y="508" class="body-text" fill="#475569">• External contact / gravity changes</text>
  <text x="40" y="524" class="body-text" fill="#475569">• Mechanism accelerates uncontrolled</text>
  <text x="40" y="546" class="mono-text" fill="#A51C30">Result: Runaway joint acceleration</text>
  <text x="40" y="562" class="annot-text" fill="#991B1B">Thermal runaway in coils</text>

  <!-- Card 2: Hold Position -->
  <rect x="260" y="456" width="220" height="120" rx="5" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1"/>
  <text x="270" y="474" class="card-hdr" fill="#A51C30">2. Hold Position Setpoint</text>
  <text x="270" y="492" class="body-text" fill="#475569">• Commands v = 0 instantaneously</text>
  <text x="270" y="508" class="body-text" fill="#475569">• Step velocity jump at moving tool</text>
  <text x="270" y="524" class="body-text" fill="#475569">• Inverter overcurrent trip</text>
  <text x="270" y="546" class="mono-text" fill="#A51C30">Result: 324 N·m torque shock</text>
  <text x="270" y="562" class="annot-text" fill="#991B1B">Gear tooth brinelling &amp; trip</text>

  <!-- Card 3: Continue Velocity -->
  <rect x="490" y="456" width="220" height="120" rx="5" fill="#FEF2F2" stroke="#FCA5A5" stroke-width="1"/>
  <text x="500" y="474" class="card-hdr" fill="#A51C30">3. Continue Commanded Velocity</text>
  <text x="500" y="492" class="body-text" fill="#475569">• Avoids transient current spike</text>
  <text x="500" y="508" class="body-text" fill="#475569">• Expends clearance at 1.2 m/s</text>
  <text x="500" y="524" class="body-text" fill="#475569">• Traverses 300 mm gap in 250 ms</text>
  <text x="500" y="546" class="mono-text" fill="#A51C30">Result: Fixture hard impact</text>
  <text x="500" y="562" class="annot-text" fill="#991B1B">Tool breakage &amp; payload drop</text>

  <!-- Card 4: Verified Fallback Suffix -->
  <rect x="720" y="456" width="230" height="120" rx="5" fill="#ECFDF5" stroke="#6EE7B7" stroke-width="1.2"/>
  <text x="730" y="474" class="card-hdr" fill="#0D9488">4. Verified Fallback Suffix</text>
  <text x="730" y="492" class="body-text" fill="#475569">• Precomputed C² deceleration</text>
  <text x="730" y="508" class="body-text" fill="#475569">• Budgeted stopping dist: 192 mm</text>
  <text x="730" y="524" class="body-text" fill="#475569">• Resident in memory at t = 0 ms</text>
  <text x="730" y="546" class="mono-text" fill="#059669">Result: Safe stop at t_term</text>
  <text x="730" y="562" class="annot-text" fill="#059669">Mechanical brakes engage safely</text>
</svg>
"""

def generate_all():
    # Write SVG files
    svg1_path = os.path.join(CH11_FIG_DIR, "fig11_action_chunk_seam_continuity.svg")
    pdf1_path = os.path.join(CH11_FIG_DIR, "fig11_action_chunk_seam_continuity.pdf")
    with open(svg1_path, "w", encoding="utf-8") as f:
        f.write(FIG11_SEAM_SVG.strip())
    print(f"Generated {svg1_path}")

    svg2_path = os.path.join(CH11_FIG_DIR, "fig11_seam_timeline_fallback.svg")
    pdf2_path = os.path.join(CH11_FIG_DIR, "fig11_seam_timeline_fallback.pdf")
    with open(svg2_path, "w", encoding="utf-8") as f:
        f.write(FIG11_TIMELINE_SVG.strip())
    print(f"Generated {svg2_path}")

    # Convert to PDF via rsvg-convert
    for svg_p, pdf_p in [(svg1_path, pdf1_path), (svg2_path, pdf2_path)]:
        cmd = ["rsvg-convert", "-f", "pdf", "-o", pdf_p, svg_p]
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ Converted to PDF: {pdf_p}")
        except Exception as e:
            print(f"  ✗ Failed to convert {svg_p} to PDF: {e}")

if __name__ == "__main__":
    generate_all()
