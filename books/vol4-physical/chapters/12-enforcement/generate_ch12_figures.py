#!/usr/bin/env python3
"""
Generate publication-grade vector SVG figures for Chapter 12: Enforcement.
- fig12_cbf_safety_filter.svg: CBF Safe Set Invariance in State Space and Minimal-Intervention QP Projection in Control Space.
- fig12_fallback_ladder.svg: The Deterministic Four-Tier Fallback Escalation Ladder.
"""

import os
import subprocess

SVG_CBF_PATH = "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/12-enforcement/figures/fig12_cbf_safety_filter.svg"
PDF_CBF_PATH = "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/12-enforcement/figures/fig12_cbf_safety_filter.pdf"

SVG_LADDER_PATH = "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/12-enforcement/figures/fig12_fallback_ladder.svg"
PDF_LADDER_PATH = "/Users/VJ/GitHub/PhysicalAI-draft/book/chapters/12-enforcement/figures/fig12_fallback_ladder.pdf"

def generate_cbf_diagram():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 540" width="100%" height="100%">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .main-title { font-weight: 800; font-size: 15px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.6px; }
      .main-subtitle { font-size: 11px; fill: #64748B; text-anchor: middle; font-weight: 400; }
      .panel-title { font-weight: 700; font-size: 12px; fill: #1F407A; letter-spacing: 0.3px; }
      .axis-label { font-size: 10px; fill: #475569; font-weight: 600; }
      .tick-label { font-size: 8.5px; fill: #64748B; }
      .math-text { font-family: "STIX Two Text", "Times New Roman", Georgia, serif; font-style: italic; }
      .code-font { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 8.5px; }
      .badge-hdr { font-weight: 700; font-size: 9px; letter-spacing: 0.4px; }
      .bold-text { font-weight: 700; }
    </style>

    <!-- Drop Shadows -->
    <filter id="panelShadow" x="-2%" y="-2%" width="104%" height="106%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.05"/>
    </filter>

    <!-- Arrow Markers -->
    <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A"/>
    </marker>
    <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
    </marker>
    <marker id="arr-petrol" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#007A87"/>
    </marker>
    <marker id="arr-bronze" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#B87333"/>
    </marker>
    <marker id="arr-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B"/>
    </marker>

    <!-- Gradients -->
    <linearGradient id="safeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F0FDF4" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#ECFDF5" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="unsafeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FEF2F2" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#FFF1F2" stop-opacity="0.5"/>
    </linearGradient>
    <linearGradient id="insetGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#007A87" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#007A87" stop-opacity="0.04"/>
    </linearGradient>
    <linearGradient id="halfspaceGrad" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#EFF6FF" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#F0FDFA" stop-opacity="0.5"/>
    </linearGradient>
    <linearGradient id="forbiddenU" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FEF2F2" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#FFF1F2" stop-opacity="0.3"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="960" height="540" fill="#FFFFFF" rx="10" stroke="#CBD5E1" stroke-width="1"/>

  <!-- Master Header -->
  <text x="480" y="26" class="main-title">CONTROL BARRIER FUNCTIONS (CBF) &amp; MINIMAL INTERVENTION</text>
  <text x="480" y="42" class="main-subtitle">Forward Invariance in State Space and Orthogonal Quadratic Program Projection in Control Space</text>

  <!-- ========================================================================================= -->
  <!-- LEFT PANEL: (a) State Space Forward Invariance -->
  <!-- ========================================================================================= -->
  <g transform="translate(24, 56)">
    <rect width="444" height="462" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2" filter="url(#panelShadow)"/>
    <rect width="444" height="28" rx="8" fill="#1F407A" fill-opacity="0.08"/>
    <text x="14" y="19" class="panel-title">(a) State Space Invariance: Safe Set C &amp; Tangential Filtering</text>

    <!-- Phase Space Plot Area -->
    <!-- Safe Set Region C: h(p, v) >= 0 (Left of h=0 curve) -->
    <path d="M 50 50 L 256 50 Q 280 110 355 245 L 50 245 Z" fill="url(#safeGrad)" stroke="none"/>
    
    <!-- Tracking Margin Inset Region: between h=0 and h_safe=0 -->
    <path d="M 226 50 Q 250 110 325 245 L 355 245 Q 280 110 256 50 Z" fill="url(#insetGrad)" stroke="none"/>

    <!-- Unsafe Region: Right of h=0 curve -->
    <path d="M 256 50 L 420 50 L 420 245 L 355 245 Q 280 110 256 50 Z" fill="url(#unsafeGrad)" stroke="none"/>

    <!-- Grid / Tick guidelines -->
    <line x1="50" y1="175" x2="420" y2="175" stroke="#E2E8F0" stroke-width="0.8" stroke-dasharray="2 2"/>
    <line x1="50" y1="105" x2="420" y2="105" stroke="#E2E8F0" stroke-width="0.8" stroke-dasharray="2 2"/>
    <line x1="180" y1="50" x2="180" y2="245" stroke="#E2E8F0" stroke-width="0.8" stroke-dasharray="2 2"/>
    <line x1="320" y1="50" x2="320" y2="245" stroke="#E2E8F0" stroke-width="0.8" stroke-dasharray="2 2"/>

    <!-- Axes -->
    <line x1="50" y1="245" x2="425" y2="245" stroke="#475569" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <line x1="50" y1="245" x2="50" y2="42" stroke="#475569" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <text x="420" y="260" class="axis-label" text-anchor="end">Position / Distance <tspan class="math-text">p</tspan> [m]</text>
    <text x="54" y="48" class="axis-label" text-anchor="start">Velocity <tspan class="math-text">v</tspan> [m/s]</text>

    <!-- Region Labels & Badges -->
    <rect x="58" y="56" width="98" height="30" rx="4" fill="#FFFFFF" fill-opacity="0.95" stroke="#10B981" stroke-width="1"/>
    <text x="107" y="69" font-size="8.5" font-weight="700" fill="#065F46" text-anchor="middle">SAFE SET C</text>
    <text x="107" y="80" font-size="7.5" fill="#047857" text-anchor="middle" class="math-text">h(x) ≥ 0  (Admissible)</text>

    <rect x="352" y="56" width="76" height="30" rx="4" fill="#FFFFFF" fill-opacity="0.95" stroke="#A51C30" stroke-width="1"/>
    <text x="390" y="69" font-size="8.5" font-weight="700" fill="#A51C30" text-anchor="middle">UNSAFE SET</text>
    <text x="390" y="80" font-size="7.5" fill="#991B1B" text-anchor="middle" class="math-text">h(x) &lt; 0</text>

    <!-- Boundary Lines -->
    <!-- True Physical Boundary: h(x) = 0 -->
    <path d="M 256 50 Q 280 110 355 245" fill="none" stroke="#1F407A" stroke-width="2.5"/>
    
    <!-- Inset Safety Margin Boundary: h_safe(x) = h(x) - delta_margin = 0 -->
    <path d="M 226 50 Q 250 110 325 245" fill="none" stroke="#007A87" stroke-width="1.6" stroke-dasharray="4 3"/>

    <!-- Inset Margin Callout Badge placed safely on left -->
    <rect x="58" y="152" width="128" height="32" rx="4" fill="#FFFFFF" fill-opacity="0.95" stroke="#007A87" stroke-width="0.8"/>
    <text x="64" y="164" font-size="8" font-weight="700" fill="#007A87">Inset Margin δ_margin</text>
    <text x="64" y="176" font-size="7" fill="#475569">Tracking Buffer: ε_p + (v ε_v)/a_max</text>
    <line x1="186" y1="168" x2="250" y2="168" stroke="#007A87" stroke-width="1.2" marker-end="url(#arr-petrol)"/>

    <!-- Boundary Curve Labels (High up and clear) -->
    <text x="210" y="46" font-size="8" font-weight="700" fill="#007A87" text-anchor="end">h_safe(x) = 0</text>
    <text x="362" y="235" font-size="8.5" font-weight="700" fill="#1F407A">∂C: h(x) = 0</text>

    <!-- Operating Point on Inset Boundary x_0 = (p=1.5, v=1.5) -> (270, 135) -->
    <circle cx="270" cy="135" r="4" fill="#1F407A"/>
    <text x="252" y="125" font-size="8.5" font-weight="700" fill="#1F407A" text-anchor="end">State x(t)</text>

    <!-- Inward Gradient Vector ∇h(x) pointing up-left -->
    <line x1="270" y1="135" x2="215" y2="110" stroke="#1F407A" stroke-width="2" marker-end="url(#arr-navy)"/>
    <text x="208" y="106" font-size="8.5" font-weight="700" fill="#1F407A" text-anchor="end">∇h(x)</text>
    <text x="208" y="116" font-size="7" fill="#475569" text-anchor="end">Inward Normal</text>

    <!-- Tangent Line at x(t) -->
    <line x1="235" y1="50" x2="305" y2="220" stroke="#94A3B8" stroke-width="1" stroke-dasharray="2 2"/>
    <text x="245" y="65" font-size="7" fill="#64748B" transform="rotate(65, 245, 65)">Tangent Space T_C(x)</text>

    <!-- Trajectories -->
    <!-- 1. Nominal Policy: points into Unsafe Region -->
    <line x1="270" y1="135" x2="338" y2="106" stroke="#A51C30" stroke-width="2.2" marker-end="url(#arr-crimson)"/>
    <text x="345" y="105" font-size="8.5" font-weight="800" fill="#A51C30">f(x, u_nom)</text>
    <text x="345" y="115" font-size="7" fill="#A51C30">ḣ &lt; -γ h(x) (Violates Barrier)</text>

    <!-- Nominal Trajectory Trail -->
    <path d="M 120 225 Q 200 185 270 135 Q 310 100 370 70" fill="none" stroke="#A51C30" stroke-width="1.6" stroke-dasharray="3 3"/>

    <!-- 2. Filtered Action: Deflected Tangentially along Inset Boundary -->
    <line x1="270" y1="135" x2="303" y2="205" stroke="#007A87" stroke-width="2.5" marker-end="url(#arr-petrol)"/>
    <text x="316" y="195" font-size="8.5" font-weight="800" fill="#007A87">f(x, u*)</text>
    <text x="316" y="205" font-size="7" fill="#007A87">ḣ ≥ -γ h(x) (Safe Deflection)</text>

    <!-- Safe Filtered Trajectory Trail -->
    <path d="M 120 225 Q 200 185 270 135 Q 295 185 318 240" fill="none" stroke="#007A87" stroke-width="2.4"/>

    <!-- Bottom Mathematical Summary Card -->
    <rect x="12" y="278" width="420" height="172" rx="6" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>
    
    <rect x="20" y="287" width="190" height="18" rx="3" fill="#1F407A" fill-opacity="0.1"/>
    <text x="26" y="300" class="badge-hdr" fill="#1F407A">FORWARD INVARIANCE CERTIFICATE</text>
    
    <text x="20" y="320" font-size="9.5" class="code-font" fill="#1E293B">ḣ(x,u) = L_f h(x) + L_g h(x) u ≥ -γ h(x)</text>
    <text x="20" y="336" font-size="8.5" fill="#475569">• <tspan class="bold-text">Invariant Safe Set:</tspan> Zero-superlevel set <tspan class="math-text">C = {x ∈ X | h(x) ≥ 0}</tspan> is forward invariant.</text>
    <text x="20" y="350" font-size="8.5" fill="#475569">• <tspan class="bold-text">Nagumo's Theorem:</tspan> Control vector field <tspan class="math-text">f(x, u*)</tspan> remains inside tangent cone <tspan class="math-text">T_C(x)</tspan>.</text>
    <text x="20" y="364" font-size="8.5" fill="#475569">• <tspan class="bold-text">Tracking Inset Margin:</tspan> <tspan class="code-font">h_safe(x) = h(x) - δ_margin</tspan> buffers tracking error <tspan class="math-text">ε_track</tspan>.</text>
    <text x="20" y="378" font-size="8.5" fill="#475569">• <tspan class="bold-text">Kinematic Inset Formula:</tspan> <tspan class="code-font">δ_margin = ε_p + (v ε_v)/a_max</tspan> (accounts for loop delay).</text>
    <text x="20" y="398" font-size="8.5" font-weight="700" fill="#007A87">Result: Policy commands are filtered tangentially without task abortion.</text>
  </g>

  <!-- ========================================================================================= -->
  <!-- RIGHT PANEL: (b) Control Input Space Minimal Intervention -->
  <!-- ========================================================================================= -->
  <g transform="translate(492, 56)">
    <rect width="444" height="462" rx="8" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2" filter="url(#panelShadow)"/>
    <rect width="444" height="28" rx="8" fill="#007A87" fill-opacity="0.08"/>
    <text x="14" y="19" class="panel-title" fill="#007A87">(b) Control Space: Orthogonal QP Projection onto Admissible Half-Space</text>

    <!-- Isotropic Scaling: 1 unit = 48 px on both axes -->
    <!-- Origin (u1=0, u2=0) at (X=110, Y=235) -->
    <!-- Saturation Box: [-0.5, 4.5] x [-0.5, 3.5] -> X: 86..326, Y: 67..259 -->
    <rect x="86" y="67" width="240" height="192" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1.2" stroke-dasharray="4 3"/>
    <text x="320" y="79" font-size="7.5" fill="#64748B" text-anchor="end">Actuator Saturation Envelope U_act</text>

    <!-- Admissible Control Half-Space: u1 + 2u2 <= 5 -->
    <!-- Polygon of Admissible Region inside box: (86, 103) -> (326, 223) -> (326, 259) -> (86, 259) -->
    <polygon points="86,103 326,223 326,259 86,259" fill="url(#halfspaceGrad)" stroke="none"/>

    <!-- Forbidden Half-Space inside box: (86, 103) -> (326, 223) -> (326, 67) -> (86, 67) -->
    <polygon points="86,103 326,223 326,67 86,67" fill="url(#forbiddenU)" stroke="none"/>

    <!-- Active Barrier Hyperplane Line: a^T u = b (slope -0.5 in Cartesian -> slope +0.5 in SVG pixels) -->
    <line x1="70" y1="95" x2="350" y2="235" stroke="#A51C30" stroke-width="2.4"/>
    
    <!-- Axes: u1 (Torque 1) and u2 (Torque 2) -->
    <line x1="70" y1="235" x2="370" y2="235" stroke="#475569" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <line x1="110" y1="255" x2="110" y2="52" stroke="#475569" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <rect x="180" y="251" width="120" height="15" rx="2" fill="#FFFFFF"/>
    <text x="240" y="262" class="axis-label" text-anchor="middle">Joint Torque <tspan class="math-text">u₁</tspan> [N·m]</text>
    <text x="115" y="60" class="axis-label" text-anchor="start">Joint Torque <tspan class="math-text">u₂</tspan> [N·m]</text>

    <!-- Tick marks -->
    <text x="110" y="247" class="tick-label" text-anchor="middle">0</text>
    <text x="254" y="247" class="tick-label" text-anchor="middle">3.0</text>
    <text x="302" y="247" class="tick-label" text-anchor="middle">4.0</text>
    <text x="102" y="190" class="tick-label" text-anchor="end">1.0</text>
    <text x="102" y="94" class="tick-label" text-anchor="end">3.0</text>

    <!-- Safe Half-Space Badge (placed in lower left quadrant with solid bg) -->
    <rect x="125" y="195" width="118" height="24" rx="4" fill="#FFFFFF" fill-opacity="0.95" stroke="#0284C7" stroke-width="0.8"/>
    <text x="184" y="206" font-size="7.5" font-weight="700" fill="#0369A1" text-anchor="middle">ADMISSIBLE HALF-SPACE</text>
    <text x="184" y="215" font-size="7" fill="#0284C7" text-anchor="middle" class="code-font">a^T u ≤ b (U_safe)</text>

    <!-- Active Barrier Label on Line -->
    <rect x="52" y="78" width="105" height="16" rx="3" fill="#FFFFFF" fill-opacity="0.95" stroke="#A51C30" stroke-width="0.8"/>
    <text x="56" y="89" font-size="7.5" font-weight="700" fill="#A51C30">Active Barrier: a^T u = b</text>

    <!-- Nominal Proposal Point: u_nom = [4.0, 3.0]^T -> (302, 91) -->
    <circle cx="302" cy="91" r="4.5" fill="#A51C30"/>
    <rect x="312" y="76" width="118" height="36" rx="4" fill="#FFFFFF" fill-opacity="0.95" stroke="#A51C30" stroke-width="0.8"/>
    <text x="318" y="89" font-size="9" font-weight="800" fill="#A51C30">u_nom = [4.0, 3.0]^T</text>
    <text x="318" y="99" font-size="7.5" fill="#A51C30">Policy Proposal (Unsafe)</text>
    <text x="318" y="108" font-size="7" fill="#991B1B">a^T u_nom = 10.0 &gt; 5.0</text>

    <!-- Filtered Optimal Command u* = [3.0, 1.0]^T -> (254, 187) -->
    <circle cx="254" cy="187" r="4.5" fill="#1F407A"/>
    <rect x="145" y="152" width="102" height="34" rx="4" fill="#FFFFFF" fill-opacity="0.95" stroke="#1F407A" stroke-width="0.8"/>
    <text x="240" y="165" font-size="9" font-weight="800" fill="#1F407A" text-anchor="end">u* = [3.0, 1.0]^T</text>
    <text x="240" y="175" font-size="7.5" fill="#1F407A" text-anchor="end">Optimal QP Projection</text>
    <text x="240" y="183" font-size="7" fill="#007A87" text-anchor="end">a^T u* = 5.0 (Boundary)</text>

    <!-- Orthogonal Projection Vector: from u_nom (302, 91) to u* (254, 187) -->
    <line x1="302" y1="91" x2="257" y2="181" stroke="#1F407A" stroke-width="2.5" marker-end="url(#arr-navy)"/>
    
    <!-- Projection Label Card -->
    <rect x="290" y="132" width="128" height="24" rx="3" fill="#FFFFFF" fill-opacity="0.95" stroke="#1F407A" stroke-width="0.8"/>
    <text x="295" y="143" font-size="8" font-weight="700" fill="#1F407A">Δu = -λ a = [-1.0, -2.0]^T</text>
    <text x="295" y="152" font-size="7" fill="#475569">Minimal-Norm Projection</text>

    <!-- Right-angle symbol at u* (Isotropic 90-degree square) -->
    <!-- Vector along line: dx=14.3, dy=7.15; Vector along normal: dx=7.15, dy=-14.3 -->
    <path d="M 261.15 172.7 L 275.45 179.85 L 268.3 194.15" fill="none" stroke="#1F407A" stroke-width="1.2"/>

    <!-- Comparison: Uncoordinated Heuristic Clamping u_clip = [3.5, 2.0]^T -> (278, 139) -->
    <circle cx="278" cy="139" r="3.5" fill="#D97706"/>
    <line x1="302" y1="91" x2="278" y2="139" stroke="#D97706" stroke-width="1.2" stroke-dasharray="3 3"/>
    <rect x="145" y="112" width="138" height="28" rx="3" fill="#FFFBEB" stroke="#D97706" stroke-width="0.8"/>
    <text x="149" y="123" font-size="7.5" font-weight="700" fill="#B45309">u_clip = [3.5, 2.0]^T (Scalar Clamp)</text>
    <text x="149" y="133" font-size="7" fill="#92400E">a^T u_clip = 7.5 &gt; 5.0 (STILL UNSAFE!)</text>

    <!-- Bottom Mathematical Summary Card -->
    <rect x="12" y="278" width="420" height="172" rx="6" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1"/>

    <rect x="20" y="287" width="200" height="18" rx="3" fill="#007A87" fill-opacity="0.1"/>
    <text x="26" y="300" class="badge-hdr" fill="#007A87">QUADRATIC PROGRAM SAFETY FILTER</text>

    <text x="20" y="320" font-size="9.5" class="code-font" fill="#1E293B">u* = argmin_{u ∈ U} 1/2 ||u - u_nom||²  s.t.  a^T u ≤ b</text>
    <text x="20" y="336" font-size="8.5" fill="#475569">• <tspan class="bold-text">Closed-Form Analytical Solution:</tspan> <tspan class="code-font">u* = u_nom - (max(0, a^T u_nom - b) / ||a||²) a</tspan></text>
    <text x="20" y="350" font-size="8.5" fill="#475569">• <tspan class="bold-text">Zero Intervention Interior:</tspan> If <tspan class="math-text">a^T u_nom ≤ b</tspan>, then <tspan class="math-text">u* = u_nom</tspan> (unmodified policy intent).</text>
    <text x="20" y="364" font-size="8.5" fill="#475569">• <tspan class="bold-text">Coupled Multi-Axis Dynamics:</tspan> Preserves Cartesian wrench alignment vs scalar clipping.</text>
    <text x="20" y="378" font-size="8.5" fill="#475569">• <tspan class="bold-text">Policy Health Telemetry:</tspan> Rolling norm <tspan class="math-text">Σ ||u* - u_nom|| Δt</tspan> tracks upstream model drift.</text>
    <text x="20" y="398" font-size="8.5" font-weight="700" fill="#1F407A">Guarantee: Microsecond deterministic QP projection on embedded MCU SRAM.</text>
  </g>
</svg>
"""
    with open(SVG_CBF_PATH, "w") as f:
        f.write(svg)
    print(f"Generated: {SVG_CBF_PATH}")

    # Convert to PDF
    cmd = f"rsvg-convert -f pdf -o {PDF_CBF_PATH} {SVG_CBF_PATH}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Compiled PDF: {PDF_CBF_PATH}")

def generate_ladder_diagram():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 540" width="100%" height="100%">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .main-title { font-weight: 800; font-size: 15px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.6px; }
      .main-subtitle { font-size: 11px; fill: #64748B; text-anchor: middle; font-weight: 400; }
      .rung-num { font-weight: 800; font-size: 11px; fill: #FFFFFF; text-anchor: middle; }
      .rung-title { font-weight: 700; font-size: 12px; }
      .rung-cat { font-size: 9px; font-weight: 600; }
      .field-label { font-size: 8.5px; font-weight: 700; fill: #1E293B; }
      .field-val { font-size: 8.5px; fill: #475569; }
      .math-text { font-family: "STIX Two Text", "Times New Roman", Georgia, serif; font-style: italic; }
      .code-font { font-family: "SFMono-Regular", Consolas, Menlo, monospace; font-size: 8px; }
      .axis-label { font-size: 9px; font-weight: 700; }
    </style>

    <!-- Drop Shadows -->
    <filter id="cardShadow" x="-2%" y="-2%" width="104%" height="106%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="1.5" stdDeviation="2.5" flood-color="#0F172A" flood-opacity="0.06"/>
    </filter>

    <!-- Markers -->
    <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A"/>
    </marker>
    <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30"/>
    </marker>
    <marker id="arr-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B"/>
    </marker>

    <!-- Escalation Arrow Gradient -->
    <linearGradient id="escalateGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#007A87"/>
      <stop offset="35%" stop-color="#B87333"/>
      <stop offset="70%" stop-color="#D97706"/>
      <stop offset="100%" stop-color="#A51C30"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="960" height="540" fill="#FFFFFF" rx="10" stroke="#CBD5E1" stroke-width="1"/>

  <!-- Master Header -->
  <text x="480" y="27" class="main-title">THE DETERMINISTIC FOUR-TIER FALLBACK ESCALATION LADDER</text>
  <text x="480" y="44" class="main-subtitle">Graded Safety Authority: Escalating Physical Cost, Decreasing Software Dependency &amp; Increasing Isolation</text>

  <!-- ========================================================================================= -->
  <!-- 4 RUNG CARDS -->
  <!-- ========================================================================================= -->

  <!-- RUNG 1: Least-Squares QP Projection -->
  <g transform="translate(20, 60)">
    <rect width="216" height="375" rx="8" fill="#FFFFFF" stroke="#007A87" stroke-width="1.3" filter="url(#cardShadow)"/>
    <!-- Header Banner -->
    <rect width="216" height="36" rx="8" fill="#007A87" fill-opacity="0.12"/>
    <circle cx="22" cy="18" r="11" fill="#007A87"/>
    <text x="22" y="22" class="rung-num">1</text>
    <text x="40" y="16" class="rung-title" fill="#007A87">QP Safety Filter</text>
    <text x="40" y="29" class="rung-cat" fill="#047857">Least-Squares Projection</text>

    <!-- Content Fields -->
    <!-- Trigger -->
    <rect x="10" y="44" width="196" height="42" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="56" class="field-label">TRIGGER CONDITION</text>
    <text x="16" y="68" class="field-val">Policy proposal <tspan class="math-text">u_nom</tspan> violates CBF;</text>
    <text x="16" y="79" class="field-val">Feasible set non-empty (<tspan class="math-text">U_safe ≠ ∅</tspan>).</text>

    <!-- Execution Mechanism -->
    <rect x="10" y="92" width="196" height="44" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="104" class="field-label">MECHANISM &amp; ALGORITHM</text>
    <text x="16" y="116" class="field-val">Orthogonal QP projection in SRAM;</text>
    <text x="16" y="127" class="field-val">Calculates minimal correction <tspan class="math-text">Δu = -λa</tspan>.</text>

    <!-- Latency & Timing -->
    <rect x="10" y="142" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="154" class="field-label">REACTION LATENCY</text>
    <text x="16" y="166" class="field-val"><tspan font-weight="700" fill="#007A87">≤ 150 µs</tspan> (Active-set QP @ 1 kHz)</text>

    <!-- Physical Cost -->
    <rect x="10" y="186" width="196" height="48" rx="4" fill="#F0FDF4" stroke="#BBF7D0" stroke-width="0.8"/>
    <text x="16" y="198" class="field-label" fill="#166534">PHYSICAL &amp; SYSTEM COST</text>
    <text x="16" y="210" class="field-val" fill="#15803D">• 0 mechanical wear / 0 brake wear</text>
    <text x="16" y="221" class="field-val" fill="#15803D">• 0 mission downtime (Active Task)</text>

    <!-- Reversibility -->
    <rect x="10" y="240" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="252" class="field-label">REVERSIBILITY</text>
    <text x="16" y="264" class="field-val"><tspan font-weight="700" fill="#047857">100% Reversible</tspan> (Continuous)</text>

    <!-- Execution Substrate -->
    <rect x="10" y="284" width="196" height="42" rx="4" fill="#EFF6FF" stroke="#BFDBFE" stroke-width="0.8"/>
    <text x="16" y="296" class="field-label" fill="#1E40AF">EXECUTION SUBSTRATE</text>
    <text x="16" y="308" class="field-val" fill="#1E3A8A">NPU/MPU Software / MCU</text>
    <text x="16" y="319" class="field-val" fill="#1E3A8A">Pre-allocated static SRAM arrays</text>

    <!-- Standards & Guarantees -->
    <rect x="10" y="332" width="196" height="34" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="344" class="field-label">STANDARD / CLASSIFICATION</text>
    <text x="16" y="356" class="field-val">Control Barrier Certificate (CBF)</text>
  </g>

  <!-- RUNG 2: Active Position Hold -->
  <g transform="translate(254, 60)">
    <rect width="216" height="375" rx="8" fill="#FFFFFF" stroke="#B87333" stroke-width="1.3" filter="url(#cardShadow)"/>
    <!-- Header Banner -->
    <rect width="216" height="36" rx="8" fill="#B87333" fill-opacity="0.12"/>
    <circle cx="22" cy="18" r="11" fill="#B87333"/>
    <text x="22" y="22" class="rung-num">2</text>
    <text x="40" y="16" class="rung-title" fill="#B87333">Active Position Hold</text>
    <text x="40" y="29" class="rung-cat" fill="#9A3412">IEC 60204-1 Category 2 (SS2)</text>

    <!-- Content Fields -->
    <!-- Trigger -->
    <rect x="10" y="44" width="196" height="42" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="56" class="field-label">TRIGGER CONDITION</text>
    <text x="16" y="68" class="field-val">Empty safe set (<tspan class="math-text">U_safe = ∅</tspan>), solver fail,</text>
    <text x="16" y="79" class="field-val">or upstream policy lease timeout.</text>

    <!-- Execution Mechanism -->
    <rect x="10" y="92" width="196" height="44" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="104" class="field-label">MECHANISM &amp; ALGORITHM</text>
    <text x="16" y="116" class="field-val">PID/LQR arrests motion to <tspan class="math-text">x_hold</tspan>;</text>
    <text x="16" y="127" class="field-val">Continuous PWM active current control.</text>

    <!-- Latency & Timing -->
    <rect x="10" y="142" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="154" class="field-label">REACTION LATENCY</text>
    <text x="16" y="166" class="field-val"><tspan font-weight="700" fill="#B87333">10 – 50 ms</tspan> (Closed-loop holding)</text>

    <!-- Physical Cost -->
    <rect x="10" y="186" width="196" height="48" rx="4" fill="#FFFBEB" stroke="#FDE68A" stroke-width="0.8"/>
    <text x="16" y="198" class="field-label" fill="#92400E">PHYSICAL &amp; SYSTEM COST</text>
    <text x="16" y="210" class="field-val" fill="#B45309">• Continuous Joule heating (<tspan class="math-text">I²R</tspan>)</text>
    <text x="16" y="221" class="field-val" fill="#B45309">• 0 brake wear; mission paused</text>

    <!-- Reversibility -->
    <rect x="10" y="240" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="252" class="field-label">REVERSIBILITY</text>
    <text x="16" y="264" class="field-val"><tspan font-weight="700" fill="#9A3412">Reversible</tspan> on lease renewal</text>

    <!-- Execution Substrate -->
    <rect x="10" y="284" width="196" height="42" rx="4" fill="#FFF7ED" stroke="#FED7AA" stroke-width="0.8"/>
    <text x="16" y="296" class="field-label" fill="#9A3412">EXECUTION SUBSTRATE</text>
    <text x="16" y="308" class="field-val" fill="#9A3412">Real-Time Core / RTOS Loop</text>
    <text x="16" y="319" class="field-val" fill="#9A3412">Deterministic current regulation</text>

    <!-- Standards & Guarantees -->
    <rect x="10" y="332" width="196" height="34" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="344" class="field-label">STANDARD / CLASSIFICATION</text>
    <text x="16" y="356" class="field-val">IEC 60204-1 Category 2 (Safe Stop 2)</text>
  </g>

  <!-- RUNG 3: Controlled Dynamic Stop -->
  <g transform="translate(488, 60)">
    <rect width="216" height="375" rx="8" fill="#FFFFFF" stroke="#D97706" stroke-width="1.3" filter="url(#cardShadow)"/>
    <!-- Header Banner -->
    <rect width="216" height="36" rx="8" fill="#D97706" fill-opacity="0.12"/>
    <circle cx="22" cy="18" r="11" fill="#D97706"/>
    <text x="22" y="22" class="rung-num">3</text>
    <text x="40" y="16" class="rung-title" fill="#D97706">Dynamic Decel Stop</text>
    <text x="40" y="29" class="rung-cat" fill="#B45309">IEC 60204-1 Category 1 (SS1)</text>

    <!-- Content Fields -->
    <!-- Trigger -->
    <rect x="10" y="44" width="196" height="42" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="56" class="field-label">TRIGGER CONDITION</text>
    <text x="16" y="68" class="field-val">Tracking error breach (<tspan class="math-text">|ε| &gt; ε_max</tspan>),</text>
    <text x="16" y="79" class="field-val">obstacle approach, lease expiry.</text>

    <!-- Execution Mechanism -->
    <rect x="10" y="92" width="196" height="44" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="104" class="field-label">MECHANISM &amp; ALGORITHM</text>
    <text x="16" y="116" class="field-val">Certified max dynamic braking <tspan class="math-text">a_max</tspan>;</text>
    <text x="16" y="127" class="field-val">Brake chopper absorbs DC-bus surge.</text>

    <!-- Latency & Timing -->
    <rect x="10" y="142" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="154" class="field-label">REACTION LATENCY</text>
    <text x="16" y="166" class="field-val"><tspan font-weight="700" fill="#D97706">100 – 500 ms</tspan> (Stopping profile)</text>

    <!-- Physical Cost -->
    <rect x="10" y="186" width="196" height="48" rx="4" fill="#FEF3C7" stroke="#FDE68A" stroke-width="0.8"/>
    <text x="16" y="198" class="field-label" fill="#B45309">PHYSICAL &amp; SYSTEM COST</text>
    <text x="16" y="210" class="field-val" fill="#92400E">• Peak torque load on gear teeth</text>
    <text x="16" y="221" class="field-val" fill="#92400E">• Mission aborted; requires re-homing</text>

    <!-- Reversibility -->
    <rect x="10" y="240" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="252" class="field-label">REVERSIBILITY</text>
    <text x="16" y="264" class="field-val"><tspan font-weight="700" fill="#DC2626">Non-Reversible</tspan> (Task Abort)</text>

    <!-- Execution Substrate -->
    <rect x="10" y="284" width="196" height="42" rx="4" fill="#FEF2F2" stroke="#FECACA" stroke-width="0.8"/>
    <text x="16" y="296" class="field-label" fill="#B91C1C">EXECUTION SUBSTRATE</text>
    <text x="16" y="308" class="field-val" fill="#991B1B">Dedicated Safety MCU (Cortex-R)</text>
    <text x="16" y="319" class="field-val" fill="#991B1B">Independent sensor channel &amp; power</text>

    <!-- Standards & Guarantees -->
    <rect x="10" y="332" width="196" height="34" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="344" class="field-label">STANDARD / CLASSIFICATION</text>
    <text x="16" y="356" class="field-val">IEC 60204-1 Category 1 (Safe Stop 1)</text>
  </g>

  <!-- RUNG 4: Hard Power Cutoff & Mechanical Clamping -->
  <g transform="translate(722, 60)">
    <rect width="216" height="375" rx="8" fill="#FFFFFF" stroke="#A51C30" stroke-width="1.4" filter="url(#cardShadow)"/>
    <!-- Header Banner -->
    <rect width="216" height="36" rx="8" fill="#A51C30" fill-opacity="0.12"/>
    <circle cx="22" cy="18" r="11" fill="#A51C30"/>
    <text x="22" y="22" class="rung-num">4</text>
    <text x="40" y="16" class="rung-title" fill="#A51C30">Safe Torque Off (STO)</text>
    <text x="40" y="29" class="rung-cat" fill="#991B1B">IEC 60204-1 Category 0 (STO)</text>

    <!-- Content Fields -->
    <!-- Trigger -->
    <rect x="10" y="44" width="196" height="42" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="56" class="field-label">TRIGGER CONDITION</text>
    <text x="16" y="68" class="field-val">Watchdog timeout (>50 ms), shoot-through,</text>
    <text x="16" y="79" class="field-val">overcurrent comparator, E-stop switch.</text>

    <!-- Execution Mechanism -->
    <rect x="10" y="92" width="196" height="44" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="104" class="field-label">MECHANISM &amp; ALGORITHM</text>
    <text x="16" y="116" class="field-val">Optocouplers drop gate bias (&lt;5 µs);</text>
    <text x="16" y="127" class="field-val">Spring-applied friction brakes clamp.</text>

    <!-- Latency & Timing -->
    <rect x="10" y="142" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="154" class="field-label">REACTION LATENCY</text>
    <text x="16" y="166" class="field-val"><tspan font-weight="700" fill="#A51C30">&lt; 5 µs</tspan> (Silicon) / 15 ms (Pad seat)</text>

    <!-- Physical Cost -->
    <rect x="10" y="186" width="196" height="48" rx="4" fill="#FEF2F2" stroke="#FECACA" stroke-width="0.8"/>
    <text x="16" y="198" class="field-label" fill="#991B1B">PHYSICAL &amp; SYSTEM COST</text>
    <text x="16" y="210" class="field-val" fill="#B91C1C">• Friction pad ablation &amp; shock stress</text>
    <text x="16" y="221" class="field-val" fill="#B91C1C">• Potential gravity sag before pad seat</text>

    <!-- Reversibility -->
    <rect x="10" y="240" width="196" height="38" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="252" class="field-label">REVERSIBILITY</text>
    <text x="16" y="264" class="field-val"><tspan font-weight="700" fill="#991B1B">Manual Lockout Reset</tspan></text>

    <!-- Execution Substrate -->
    <rect x="10" y="284" width="196" height="42" rx="4" fill="#FEF2F2" stroke="#F87171" stroke-width="0.8"/>
    <text x="16" y="296" class="field-label" fill="#7F1D1D">EXECUTION SUBSTRATE</text>
    <text x="16" y="308" class="field-val" fill="#7F1D1D">Galvanic Hardware Safety Relay</text>
    <text x="16" y="319" class="field-val" fill="#7F1D1D">Hardware optocouplers &amp; contactors</text>

    <!-- Standards & Guarantees -->
    <rect x="10" y="332" width="196" height="34" rx="4" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="0.8"/>
    <text x="16" y="344" class="field-label">STANDARD / CLASSIFICATION</text>
    <text x="16" y="356" class="field-val">ISO 13849-1 PLe / ISO 26262 ASIL-D</text>
  </g>

  <!-- ========================================================================================= -->
  <!-- BOTTOM PROGRESSION BARS -->
  <!-- ========================================================================================= -->
  <g transform="translate(20, 448)">
    <rect width="918" height="34" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"/>
    <rect x="12" y="8" width="894" height="6" rx="3" fill="url(#escalateGrad)"/>
    
    <text x="20" y="26" class="axis-label" fill="#007A87">← Low Mechanical Cost / High Reversibility</text>
    <text x="460" y="26" class="axis-label" fill="#B87333" text-anchor="middle">ESCALATION OF DEFENSIVE INTERVENTION</text>
    <text x="900" y="26" class="axis-label" fill="#A51C30" text-anchor="end">High Hardware Cost / Non-Reversible →</text>
  </g>

  <g transform="translate(20, 490)">
    <rect width="918" height="34" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1"/>
    
    <text x="20" y="16" font-size="8.5" font-weight="700" fill="#1F407A">SUBSTRATE INDEPENDENCE:</text>
    <text x="175" y="16" font-size="8.5" fill="#475569">Shared MPU / RTOS Core</text>
    <line x1="300" y1="12" x2="330" y2="12" stroke="#94A3B8" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <text x="345" y="16" font-size="8.5" fill="#475569">Isolated Safety Co-Processor</text>
    <line x1="500" y1="12" x2="530" y2="12" stroke="#94A3B8" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <text x="545" y="16" font-size="8.5" fill="#475569">Dual-Channel Safety MCU</text>
    <line x1="685" y1="12" x2="715" y2="12" stroke="#94A3B8" stroke-width="1.2" marker-end="url(#arr-slate)"/>
    <text x="730" y="16" font-size="8.5" font-weight="700" fill="#A51C30">Galvanic Analog Hardware / E-Stop</text>

    <text x="20" y="28" font-size="7.5" fill="#64748B">Systems Architectural Rule: Lower rungs must execute on substrates physically isolated from upstream software failure modes.</text>
  </g>
</svg>
"""
    with open(SVG_LADDER_PATH, "w") as f:
        f.write(svg)
    print(f"Generated: {SVG_LADDER_PATH}")

    # Convert to PDF
    cmd = f"rsvg-convert -f pdf -o {PDF_LADDER_PATH} {SVG_LADDER_PATH}"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Compiled PDF: {PDF_LADDER_PATH}")

if __name__ == "__main__":
    generate_cbf_diagram()
    generate_ladder_diagram()
