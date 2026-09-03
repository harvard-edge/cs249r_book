#!/usr/bin/env python3
"""
Generate publication-grade SVG and PDF figures for Chapter 10 (Intent):
1. fig10_intent_lease_envelope (The Intent Lease Contract: Spatial Tolerance & Temporal Expiration)
2. fig10_lease_dynamics_tradeoff (Safe Lease Horizon vs. Scene Drift Dynamics & Ingestion Filter Pipeline)

Follows the Harvard Crimson & ETH Zurich design and color standards from figures.md.
"""
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CH10_FIG_DIR = os.path.join(BOOK_DIR, "chapters", "10-intent", "figures")

os.makedirs(CH10_FIG_DIR, exist_ok=True)


# =============================================================================
# FIGURE 1: INTENT LEASE TEMPORAL EXPIRATION & GEOMETRIC TOLERANCE ENVELOPE
# =============================================================================
FIG10_1_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 570" width="100%" height="auto" role="img" aria-label="The Intent Lease Contract: Spatial Tolerance and Temporal Expiration">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .title { font-weight: 800; font-size: 15px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.5px; }
      .subtitle { font-size: 11px; font-weight: 500; fill: #475569; text-anchor: middle; }
      .panel-hdr { font-weight: 700; font-size: 12.5px; fill: #1F407A; letter-spacing: 0.3px; }
      .card-title { font-weight: 700; font-size: 11.5px; }
      .body-text { font-size: 10px; fill: #334155; line-height: 1.35; }
      .bold-text { font-weight: 700; font-size: 10px; fill: #0F172A; }
      .small-text { font-size: 9px; fill: #64748B; }
      .code-text { font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 9.5px; font-weight: 600; }
      .badge-text { font-weight: 700; font-size: 9.5px; text-anchor: middle; }
    </style>

    <marker id="arr-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A" />
    </marker>
    <marker id="arr-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30" />
    </marker>
    <marker id="arr-teal" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#007A87" />
    </marker>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563EB" />
    </marker>
    <marker id="arr-amber" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#D97706" />
    </marker>
    <marker id="arr-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B" />
    </marker>

    <!-- Linear gradients for volumetric ellipsoid -->
    <radialGradient id="ellipsoid-grad" cx="45%" cy="40%" r="55%" fx="35%" fy="30%">
      <stop offset="0%" stop-color="#EFF6FF" stop-opacity="0.9" />
      <stop offset="60%" stop-color="#93C5FD" stop-opacity="0.5" />
      <stop offset="100%" stop-color="#2563EB" stop-opacity="0.25" />
    </radialGradient>
    <radialGradient id="drift-grad" cx="45%" cy="40%" r="55%">
      <stop offset="0%" stop-color="#FEF2F2" stop-opacity="0.9" />
      <stop offset="100%" stop-color="#FCA5A5" stop-opacity="0.3" />
    </radialGradient>
  </defs>

  <!-- Background Canvas -->
  <rect width="980" height="570" fill="#FFFFFF" rx="8" stroke="#CBD5E1" stroke-width="1.2"/>

  <!-- Top Title Banner -->
  <rect x="18" y="14" width="944" height="42" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.2"/>
  <text class="title" x="490" y="32">THE INTENT LEASE CONTRACT: SPATIAL TOLERANCE &amp; TEMPORAL EXPIRATION</text>
  <text class="subtitle" x="490" y="47">Bounding Cognitive Intent in Space, Time, and Wrench to Guarantee Safe Termination upon Silence or Failure</text>

  <!-- ========================================================================= -->
  <!-- LEFT PANEL: 3D SPATIAL TOLERANCE & WRENCH ENVELOPE                        -->
  <!-- ========================================================================= -->
  <g transform="translate(18, 66)">
    <!-- Container Card -->
    <rect width="460" height="488" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <rect x="12" y="10" width="175" height="20" rx="10" fill="#EFF6FF" stroke="#2563EB" stroke-width="1"/>
    <text class="badge-text" x="99" y="24" fill="#1D4ED8">SPATIAL CONTRACT · SE(3)</text>
    <text class="panel-hdr" x="195" y="25">PANEL A: 3D TOLERANCE ENVELOPE</text>

    <!-- 3D Geometric Scene Box -->
    <rect x="12" y="38" width="436" height="292" rx="6" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1"/>

    <!-- Worktable Surface (Isometric projection) -->
    <polygon points="40,285 240,285 410,230 210,230" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1"/>
    <line x1="40" y1="285" x2="40" y2="295" stroke="#94A3B8" stroke-width="1"/>
    <line x1="240" y1="285" x2="240" y2="295" stroke="#94A3B8" stroke-width="1"/>
    <polygon points="40,285 240,285 240,295 40,295" fill="#E2E8F0"/>
    <text class="small-text" x="70" y="278" fill="#94A3B8">Workcell Datum Plane</text>

    <!-- World Base Frame {W} -->
    <g transform="translate(50, 245)">
      <line x1="0" y1="0" x2="25" y2="10" stroke="#A51C30" stroke-width="1.5" marker-end="url(#arr-crimson)"/>
      <line x1="0" y1="0" x2="25" y2="-15" stroke="#059669" stroke-width="1.5" marker-end="url(#arr-teal)"/>
      <line x1="0" y1="0" x2="0" y2="-28" stroke="#2563EB" stroke-width="1.5" marker-end="url(#arr-blue)"/>
      <text class="code-text" x="-6" y="-32" fill="#1F407A">{W}</text>
      <text class="small-text" x="27" y="14" fill="#A51C30">X</text>
      <text class="small-text" x="27" y="-14" fill="#059669">Y</text>
      <text class="small-text" x="3" y="-22" fill="#2563EB">Z</text>
    </g>

    <!-- Target Workpiece at Nominal Pose p0 -->
    <g transform="translate(235, 175)">
      <!-- Target Workpiece Cylinder/Block -->
      <path d="M -16,35 L 16,35 L 16,50 L -16,50 Z" fill="#94A3B8" stroke="#475569" stroke-width="1"/>
      <ellipse cx="0" cy="35" rx="16" ry="6" fill="#CBD5E1" stroke="#475569" stroke-width="1"/>
      <text class="small-text" x="22" y="48" fill="#475569">Target Object (t = t₀)</text>

      <!-- 3D Oriented Covariance Ellipsoid (E) -->
      <ellipse cx="0" cy="0" rx="72" ry="38" fill="url(#ellipsoid-grad)" stroke="#2563EB" stroke-width="1.6" stroke-dasharray="none" transform="rotate(-15)"/>
      
      <!-- Conjugate Diameters / Principal Axes -->
      <line x1="-65" y1="17" x2="65" y2="-17" stroke="#1D4ED8" stroke-width="1.4" stroke-dasharray="3,2"/>
      <line x1="-10" y1="-35" x2="10" y2="35" stroke="#1D4ED8" stroke-width="1.4" stroke-dasharray="3,2"/>
      
      <!-- Nominal Setpoint Dot p0 -->
      <circle cx="0" cy="0" r="3.5" fill="#1F407A"/>
      <text class="code-text" x="6" y="-6" fill="#1F407A">p₀ ∈ ℝ³</text>
      
      <!-- Coordinate Frame at Target {T} -->
      <line x1="0" y1="0" x2="20" y2="8" stroke="#A51C30" stroke-width="1.4" marker-end="url(#arr-crimson)"/>
      <line x1="0" y1="0" x2="18" y2="-12" stroke="#059669" stroke-width="1.4" marker-end="url(#arr-teal)"/>
      <line x1="0" y1="0" x2="0" y2="-22" stroke="#2563EB" stroke-width="1.4" marker-end="url(#arr-blue)"/>
      <text class="code-text" x="18" y="-18" fill="#1D4ED8">{T_target}</text>

      <!-- Principal Semi-Axis Labels (from §10.3) -->
      <text class="small-text" x="38" y="-22" fill="#1D4ED8">a_z = 41.9 mm (depth)</text>
      <text class="small-text" x="52" y="10" fill="#1D4ED8">a_y = 16.8 mm</text>
      <text class="small-text" x="-85" y="32" fill="#1D4ED8">a_x = 11.2 mm</text>

      <!-- Mathematical Formulation of Ellipsoid -->
      <rect x="-105" y="-72" width="210" height="20" rx="4" fill="#FFFFFF" stroke="#2563EB" stroke-width="1"/>
      <text class="code-text" x="0" y="-58" text-anchor="middle" fill="#1E40AF">(p - p₀)ᵀ Σ⁻¹ (p - p₀) ≤ χ²₃,₀.₉₅</text>
    </g>

    <!-- Robot End-Effector / Gripper (Approaching) -->
    <g transform="translate(235, 62)">
      <!-- Wrist Tool Flange -->
      <rect x="-24" y="-12" width="48" height="12" rx="2" fill="#1F407A" stroke="#0F172A" stroke-width="1.2"/>
      <rect x="-16" y="0" width="32" height="10" fill="#475569" stroke="#0F172A" stroke-width="1"/>
      <!-- Parallel Gripper Jaws -->
      <path d="M -16,10 L -22,32 L -14,32 L -10,10 Z" fill="#334155" stroke="#0F172A" stroke-width="1"/>
      <path d="M 16,10 L 22,32 L 14,32 L 10,10 Z" fill="#334155" stroke="#0F172A" stroke-width="1"/>
      <line x1="-14" y1="30" x2="14" y2="30" stroke="#059669" stroke-width="1.2" stroke-dasharray="2,2"/>
      <text class="small-text" x="26" y="24" fill="#059669">Jaw Span (2·r_tol)</text>

      <!-- Approach Vector -->
      <line x1="0" y1="20" x2="0" y2="48" stroke="#D97706" stroke-width="2" marker-end="url(#arr-amber)"/>
      <text class="code-text" x="6" y="44" fill="#D97706">v_approach ≤ 0.25 m/s</text>
    </g>

    <!-- Physical Disturbance / Scene Drift -->
    <g transform="translate(365, 205)">
      <!-- Drift Vector -->
      <path d="M -125,5 Q -80,18 -30,5" fill="none" stroke="#DC2626" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arr-crimson)"/>
      <text class="small-text" x="-85" y="28" fill="#DC2626" font-weight="600">Scene Drift: v_drift · Δt</text>
      
      <!-- Displaced Object (Violating Tolerance) -->
      <ellipse cx="-15" cy="5" rx="14" ry="5" fill="url(#drift-grad)" stroke="#DC2626" stroke-width="1.2" stroke-dasharray="3,2"/>
      <text class="small-text" x="-28" y="20" fill="#DC2626">Object at t = t_exp</text>
      <rect x="-55" y="-30" width="125" height="18" rx="3" fill="#FFF1F2" stroke="#DC2626" stroke-width="1"/>
      <text class="code-text" x="7" y="-18" text-anchor="middle" fill="#991B1B">SPATIAL FAULT (Δx &gt; r_tol)</text>
    </g>

    <!-- Bottom Specification Cards: The 3 Bounds of Intent Contract -->
    <g transform="translate(12, 336)">
      <!-- Card 1: Spatial Bound -->
      <rect x="0" y="0" width="140" height="142" rx="5" fill="#FFFFFF" stroke="#2563EB" stroke-width="1.2"/>
      <rect x="0" y="0" width="140" height="22" rx="4" fill="#EFF6FF"/>
      <text class="card-title" x="70" y="15" text-anchor="middle" fill="#1D4ED8">1. Spatial Bound</text>
      <text class="body-text" x="8" y="36">
        <tspan x="8" dy="0" class="bold-text">• Target Pose:</tspan> <tspan class="code-text">T ∈ SE(3)</tspan>
        <tspan x="8" dy="16" class="bold-text">• Tolerance:</tspan> <tspan class="code-text">Σ ∈ ℝ⁶ˣ⁶</tspan>
        <tspan x="8" dy="16">• Semi-axes:</tspan>
        <tspan x="14" dy="14" class="code-text">a = [11.2, 16.8, 41.9] mm</tspan>
        <tspan x="8" dy="16" class="bold-text">• Volume:</tspan> <tspan class="code-text">V = 33.0 cm³</tspan>
        <tspan x="8" dy="16" class="small-text">Computable finish test</tspan>
      </text>

      <!-- Card 2: Temporal Lease -->
      <rect x="148" y="0" width="140" height="142" rx="5" fill="#FFFFFF" stroke="#007A87" stroke-width="1.2"/>
      <rect x="148" y="0" width="140" height="22" rx="4" fill="#F0FDFA"/>
      <text class="card-title" x="218" y="15" text-anchor="middle" fill="#007A87">2. Temporal Lease</text>
      <text class="body-text" x="156" y="36">
        <tspan x="156" dy="0" class="bold-text">• Validity:</tspan> <tspan class="code-text">τ ≤ 150 ms</tspan>
        <tspan x="156" dy="16" class="bold-text">• Expiry:</tspan> <tspan class="code-text">t_exp = t_src + τ</tspan>
        <tspan x="156" dy="16" class="bold-text">• Refresh:</tspan> <tspan class="code-text">T_period &lt; τ</tspan>
        <tspan x="156" dy="16" class="bold-text">• Invariant:</tspan>
        <tspan x="162" dy="14" class="small-text">Silence revokes motion</tspan>
        <tspan x="156" dy="16" class="code-text" fill="#007A87">Hardware WWDT Timer</tspan>
      </text>

      <!-- Card 3: Admissible Wrench -->
      <rect x="296" y="0" width="140" height="142" rx="5" fill="#FFFFFF" stroke="#A51C30" stroke-width="1.2"/>
      <rect x="296" y="0" width="140" height="22" rx="4" fill="#FEF2F2"/>
      <text class="card-title" x="366" y="15" text-anchor="middle" fill="#A51C30">3. Wrench Bound</text>
      <text class="body-text" x="304" y="36">
        <tspan x="304" dy="0" class="bold-text">• Force Cap:</tspan> <tspan class="code-text">F_max ≤ 12 N</tspan>
        <tspan x="304" dy="16" class="bold-text">• Torque Cap:</tspan> <tspan class="code-text">τ_max ≤ 2.5 Nm</tspan>
        <tspan x="304" dy="16" class="bold-text">• Power Cap:</tspan> <tspan class="code-text">P_max ≤ 25 W</tspan>
        <tspan x="304" dy="16" class="bold-text">• Terminal:</tspan>
        <tspan x="310" dy="14" class="code-text" fill="#A51C30">ACTIVE_HOLD</tspan>
        <tspan x="304" dy="16" class="small-text">Prevents runaway torque</tspan>
      </text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- RIGHT PANEL: MULTI-RATE TIMELINES & FAIL-SAFE TRANSITIONS                 -->
  <!-- ========================================================================= -->
  <g transform="translate(496, 66)">
    <!-- Container Card -->
    <rect width="466" height="488" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <rect x="12" y="10" width="195" height="20" rx="10" fill="#ECFDF5" stroke="#059669" stroke-width="1"/>
    <text class="badge-text" x="109" y="24" fill="#047857">TEMPORAL SAFETY · 1000 Hz INVARIANT</text>
    <text class="panel-hdr" x="218" y="25">PANEL B: MULTI-RATE TIMELINES</text>

    <!-- ===================================================================== -->
    <!-- TRACE 1: NOMINAL CONTINUOUS LEASE RENEWAL STREAM                      -->
    <!-- ===================================================================== -->
    <g transform="translate(12, 38)">
      <rect width="442" height="114" rx="5" fill="#FFFFFF" stroke="#10B981" stroke-width="1.2"/>
      
      <!-- Label Header -->
      <rect x="8" y="6" width="220" height="18" rx="3" fill="#ECFDF5"/>
      <text class="card-title" x="14" y="19" fill="#065F46">CASE 1: Nominal Lease Renewal Stream</text>
      <rect x="330" y="6" width="104" height="18" rx="3" fill="#D1FAE5"/>
      <text class="badge-text" x="382" y="19" fill="#047857">CONTINUOUS C² SPLICE</text>

      <!-- Time Axis Line -->
      <line x1="45" y1="64" x2="425" y2="64" stroke="#64748B" stroke-width="1.4" marker-end="url(#arr-slate)"/>
      <text class="small-text" x="430" y="67" fill="#64748B">Time (t)</text>

      <!-- Ticks -->
      <line x1="55" y1="60" x2="55" y2="68" stroke="#64748B" stroke-width="1.2"/>
      <text class="small-text" x="55" y="79" text-anchor="middle">t=0</text>
      <line x1="160" y1="60" x2="160" y2="68" stroke="#64748B" stroke-width="1.2"/>
      <text class="small-text" x="160" y="79" text-anchor="middle">100ms</text>
      <line x1="265" y1="60" x2="265" y2="68" stroke="#64748B" stroke-width="1.2"/>
      <text class="small-text" x="265" y="79" text-anchor="middle">200ms</text>
      <line x1="370" y1="60" x2="370" y2="68" stroke="#64748B" stroke-width="1.2"/>
      <text class="small-text" x="370" y="79" text-anchor="middle">300ms</text>

      <!-- Lease 1 Block -->
      <rect x="55" y="32" width="150" height="24" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2"/>
      <text class="code-text" x="130" y="48" text-anchor="middle" fill="#1E40AF">Lease #1 (TTL = 150 ms)</text>

      <!-- Lease 2 Renewal Block (Overlapping / Preemptive Refresh at t=100ms) -->
      <rect x="160" y="32" width="150" height="24" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2"/>
      <text class="code-text" x="235" y="48" text-anchor="middle" fill="#1E40AF">Lease #2 (TTL = 150 ms)</text>

      <!-- Lease 3 Renewal Block -->
      <rect x="265" y="32" width="150" height="24" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2"/>
      <text class="code-text" x="340" y="48" text-anchor="middle" fill="#1E40AF">Lease #3</text>

      <!-- Velocity Profile Trajectory (Continuous Quintic Spline) -->
      <path d="M 55,100 C 100,86 160,86 265,86 C 350,86 400,86 420,86" fill="none" stroke="#059669" stroke-width="2.2"/>
      <text class="small-text" x="65" y="104" fill="#047857" font-weight="600">Trajectory Velocity v(t) uninterrupted</text>
    </g>

    <!-- ===================================================================== -->
    <!-- TRACE 2: SILENT REASONER CRASH -> AUTONOMOUS FAIL-SAFE STOP           -->
    <!-- ===================================================================== -->
    <g transform="translate(12, 160)">
      <rect width="442" height="152" rx="5" fill="#FFFFFF" stroke="#007A87" stroke-width="1.2"/>
      
      <!-- Label Header -->
      <rect x="8" y="6" width="240" height="18" rx="3" fill="#F0FDFA"/>
      <text class="card-title" x="14" y="19" fill="#0F766E">CASE 2: Silent Crash &amp; Local Auto-Expiry</text>
      <rect x="306" y="6" width="128" height="18" rx="3" fill="#CCFBF1"/>
      <text class="badge-text" x="370" y="19" fill="#0F766E">FAIL-SAFE ON SILENCE</text>

      <!-- Time Axis Line -->
      <line x1="45" y1="62" x2="425" y2="62" stroke="#64748B" stroke-width="1.4" marker-end="url(#arr-slate)"/>
      <text class="small-text" x="430" y="65" fill="#64748B">Time (t)</text>

      <!-- Ticks -->
      <line x1="55" y1="58" x2="55" y2="66" stroke="#64748B" stroke-width="1.2"/>
      <text class="small-text" x="55" y="76" text-anchor="middle">t=0</text>
      <line x1="160" y1="58" x2="160" y2="66" stroke="#64748B" stroke-width="1.2"/>
      <text class="small-text" x="160" y="76" text-anchor="middle">100ms</text>
      <line x1="212" y1="58" x2="212" y2="66" stroke="#DC2626" stroke-width="1.6"/>
      <text class="small-text" x="212" y="76" text-anchor="middle" fill="#DC2626" font-weight="700">t_exp (150ms)</text>
      <line x1="330" y1="58" x2="330" y2="66" stroke="#059669" stroke-width="1.6"/>
      <text class="small-text" x="330" y="76" text-anchor="middle" fill="#059669" font-weight="700">t_stop (262ms)</text>

      <!-- Lease 1 Block -->
      <rect x="55" y="30" width="157" height="24" rx="4" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2"/>
      <text class="code-text" x="133" y="46" text-anchor="middle" fill="#1E40AF">Lease #1 (TTL = 150 ms)</text>

      <!-- Reasoner Crash Event -->
      <g transform="translate(140, 15)">
        <polygon points="0,0 8,14 -8,14" fill="#DC2626"/>
        <text class="badge-text" x="0" y="11" fill="#FFFFFF">!</text>
        <text class="small-text" x="12" y="12" fill="#DC2626" font-weight="700">VLM Host Hangs (t = 80 ms)</text>
      </g>

      <!-- No Renewal / Silence Zone -->
      <rect x="212" y="30" width="118" height="24" rx="4" fill="#FFF1F2" stroke="#DC2626" stroke-width="1.2" stroke-dasharray="3,2"/>
      <text class="small-text" x="271" y="45" text-anchor="middle" fill="#991B1B" font-weight="600">SILENCE (No refresh)</text>

      <!-- Hardware Expiry Trigger -->
      <line x1="212" y1="26" x2="212" y2="92" stroke="#DC2626" stroke-width="1.8" stroke-dasharray="2,2"/>
      <rect x="170" y="80" width="85" height="16" rx="3" fill="#DC2626"/>
      <text class="badge-text" x="212" y="92" fill="#FFFFFF">TTL EXPIRED</text>

      <!-- Controlled Deceleration Curve (1000 Hz Reflex) -->
      <path d="M 55,125 C 100,105 160,105 212,105 C 240,105 280,140 330,140 L 420,140" fill="none" stroke="#007A87" stroke-width="2.4"/>
      
      <!-- Stopping Metrics Annotation -->
      <g transform="translate(260, 100)">
        <rect x="0" y="0" width="170" height="42" rx="4" fill="#F0FDFA" stroke="#007A87" stroke-width="1"/>
        <text class="small-text" x="6" y="14" fill="#0F766E" font-weight="700">Deterministic Autonomous Decel:</text>
        <text class="code-text" x="6" y="26" fill="#0F766E">t_decel = v₀/a_max = 112 ms</text>
        <text class="code-text" x="6" y="38" fill="#0F766E">d_stop = 28 mm ≤ d_clearance</text>
      </g>
      <text class="small-text" x="340" y="148" fill="#059669" font-weight="700">✓ ACTIVE HOLD ENGAGED</text>
    </g>

    <!-- ===================================================================== -->
    <!-- TRACE 3: UNBOUNDED GOAL RUNAWAY ANTI-PATTERN                          -->
    <!-- ===================================================================== -->
    <g transform="translate(12, 320)">
      <rect width="442" height="114" rx="5" fill="#FFFFFF" stroke="#A51C30" stroke-width="1.2"/>
      
      <!-- Label Header -->
      <rect x="8" y="6" width="240" height="18" rx="3" fill="#FEF2F2"/>
      <text class="card-title" x="14" y="19" fill="#991B1B">CASE 3: Unbounded Goal Runaway (Anti-Pattern)</text>
      <rect x="310" y="6" width="124" height="18" rx="3" fill="#FEE2E2"/>
      <text class="badge-text" x="372" y="19" fill="#991B1B">OPEN-LOOP COLLISION</text>

      <!-- Time Axis Line -->
      <line x1="45" y1="52" x2="425" y2="52" stroke="#64748B" stroke-width="1.4" marker-end="url(#arr-slate)"/>
      <text class="small-text" x="430" y="55" fill="#64748B">Time (t)</text>

      <!-- Infinite horizon bar -->
      <rect x="55" y="26" width="280" height="20" rx="3" fill="#FEE2E2" stroke="#DC2626" stroke-width="1.2"/>
      <text class="code-text" x="195" y="40" text-anchor="middle" fill="#991B1B">Unbounded Command (TTL = ∞, No Expiration)</text>

      <!-- VLM Crash at t=80ms -->
      <g transform="translate(140, 12)">
        <polygon points="0,0 6,11 -6,11" fill="#DC2626"/>
        <text class="small-text" x="10" y="10" fill="#DC2626" font-weight="700">Crash (No abort message)</text>
      </g>

      <!-- Velocity Profile: Continues open loop until impact -->
      <path d="M 55,88 L 335,88" fill="none" stroke="#DC2626" stroke-width="2.4"/>
      
      <!-- Collision Event at t=320ms -->
      <g transform="translate(335, 70)">
        <polygon points="0,-12 12,-4 20,-16 16,0 26,10 12,12 8,24 -2,14 -16,18 -8,4 -20,-6 -6,-6" fill="#DC2626" stroke="#7F1D1D" stroke-width="1"/>
        <text class="badge-text" x="5" y="4" fill="#FFFFFF" font-size="8.5">IMPACT</text>
        <text class="code-text" x="32" y="-2" fill="#991B1B">F_peak = 1420 N</text>
        <text class="small-text" x="32" y="10" fill="#7F1D1D">Mechanical Pin Shear</text>
        <text class="small-text" x="32" y="22" fill="#7F1D1D">(OSHA Incident §10.6)</text>
      </g>
      <text class="small-text" x="65" y="102" fill="#DC2626" font-weight="600">Actuators drive unmonitored mass through shifted space</text>
    </g>

    <!-- Bottom Architectural Takeaway Banner -->
    <rect x="12" y="442" width="442" height="36" rx="4" fill="#1F407A"/>
    <text class="body-text" x="233" y="457" text-anchor="middle" fill="#FFFFFF" font-weight="600">
      SYSTEMS INVARIANT: A goal must die on its own.
    </text>
    <text class="small-text" x="233" y="470" text-anchor="middle" fill="#93C5FD">
      Silence is safe. Autonomous expiration eliminates reliance on a failed supervisor.
    </text>
  </g>
</svg>
"""


# =============================================================================
# FIGURE 2: LEASE DYNAMICS TRADEOFF & EARLY REJECTION INGESTION GATE
# =============================================================================
FIG10_2_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 540" width="100%" height="auto" role="img" aria-label="Intent Lease Dynamics Trade-off and Early Rejection Pipeline">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      .title { font-weight: 800; font-size: 15px; fill: #1F407A; text-anchor: middle; letter-spacing: 0.5px; }
      .subtitle { font-size: 11px; font-weight: 500; fill: #475569; text-anchor: middle; }
      .panel-hdr { font-weight: 700; font-size: 12.5px; fill: #1F407A; letter-spacing: 0.3px; }
      .card-title { font-weight: 700; font-size: 11.5px; }
      .body-text { font-size: 10px; fill: #334155; line-height: 1.35; }
      .bold-text { font-weight: 700; font-size: 10px; fill: #0F172A; }
      .small-text { font-size: 9px; fill: #64748B; }
      .code-text { font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 9.5px; font-weight: 600; }
      .badge-text { font-weight: 700; font-size: 9.5px; text-anchor: middle; }
    </style>

    <marker id="arr2-navy" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1F407A" />
    </marker>
    <marker id="arr2-crimson" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30" />
    </marker>
    <marker id="arr2-teal" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#007A87" />
    </marker>
    <marker id="arr2-blue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563EB" />
    </marker>
    <marker id="arr2-slate" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B" />
    </marker>
  </defs>

  <!-- Background Canvas -->
  <rect width="980" height="540" fill="#FFFFFF" rx="8" stroke="#CBD5E1" stroke-width="1.2"/>

  <!-- Top Title Banner -->
  <rect x="18" y="14" width="944" height="42" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.2"/>
  <text class="title" x="490" y="32">INTENT LEASE DYNAMICS &amp; EARLY ADMISSIBILITY FILTERING</text>
  <text class="subtitle" x="490" y="47">Physics-Derived TTL Horizon vs. O(1) Kinematic &amp; Dynamic Ingestion Gateways</text>

  <!-- ========================================================================= -->
  <!-- LEFT PANEL: MATHEMATICAL OPERATING CURVES (τ vs v_drift)                  -->
  <!-- ========================================================================= -->
  <g transform="translate(18, 66)">
    <rect width="460" height="458" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <rect x="12" y="10" width="180" height="20" rx="10" fill="#EFF6FF" stroke="#2563EB" stroke-width="1"/>
    <text class="badge-text" x="102" y="24" fill="#1D4ED8">SCENE PHYSICS · DYNAMICS</text>
    <text class="panel-hdr" x="200" y="25">PANEL A: LEASE DURATION vs. DRIFT</text>

    <!-- Graph Area -->
    <g transform="translate(10, 38)">
      <!-- Background grid and regions -->
      <rect x="50" y="20" width="375" height="270" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1"/>

      <!-- Safe Region Polygon (under Curve 2: Compliant Grasp) -->
      <path d="M 50,290 L 50,47 L 70,115 L 125,168 L 200,209 L 350,229 L 425,233 L 425,290 Z" fill="#ECFDF5" opacity="0.8"/>
      
      <!-- Hazard Region (above Curve 2) -->
      <path d="M 50,20 L 425,20 L 425,233 L 350,229 L 200,209 L 125,168 L 70,115 L 50,47 Z" fill="#FEF2F2" opacity="0.6"/>

      <text class="badge-text" x="180" y="260" fill="#047857">SAFE EXECUTION MANIFOLD (τ ≤ τ_max)</text>
      <text class="badge-text" x="270" y="65" fill="#B91C1C">STALE-GOAL COLLISION HAZARD ZONE</text>

      <!-- Grid lines -->
      <line x1="50" y1="290" x2="425" y2="290" stroke="#CBD5E1" stroke-width="1"/>
      <line x1="50" y1="222" x2="425" y2="222" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="50" y1="155" x2="425" y2="155" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="50" y1="87" x2="425" y2="87" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="50" y1="20" x2="425" y2="20" stroke="#F1F5F9" stroke-width="1"/>

      <line x1="50" y1="20" x2="50" y2="290" stroke="#CBD5E1" stroke-width="1"/>
      <line x1="125" y1="20" x2="125" y2="290" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="200" y1="20" x2="200" y2="290" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="275" y1="20" x2="275" y2="290" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="350" y1="20" x2="350" y2="290" stroke="#F1F5F9" stroke-width="1"/>
      <line x1="425" y1="20" x2="425" y2="290" stroke="#CBD5E1" stroke-width="1"/>

      <!-- Axes -->
      <line x1="50" y1="290" x2="430" y2="290" stroke="#0F172A" stroke-width="1.5" marker-end="url(#arr2-slate)"/>
      <line x1="50" y1="290" x2="50" y2="15" stroke="#0F172A" stroke-width="1.5" marker-end="url(#arr2-slate)"/>
      
      <!-- Axis Labels -->
      <text class="bold-text" x="237" y="318" text-anchor="middle">Scene Drift Velocity v_drift (mm/s)</text>
      <text class="bold-text" x="-155" y="18" text-anchor="middle" transform="rotate(-90)">Max Safe Lease Duration τ (ms)</text>

      <!-- Tick Values -->
      <text class="small-text" x="50" y="303" text-anchor="middle">0</text>
      <text class="small-text" x="125" y="303" text-anchor="middle">50</text>
      <text class="small-text" x="200" y="303" text-anchor="middle">100</text>
      <text class="small-text" x="275" y="303" text-anchor="middle">150</text>
      <text class="small-text" x="350" y="303" text-anchor="middle">200</text>
      <text class="small-text" x="425" y="303" text-anchor="middle">250</text>

      <text class="small-text" x="42" y="293" text-anchor="end">0</text>
      <text class="small-text" x="42" y="225" text-anchor="end">200</text>
      <text class="small-text" x="42" y="158" text-anchor="end">400</text>
      <text class="small-text" x="42" y="90" text-anchor="end">600</text>
      <text class="small-text" x="42" y="24" text-anchor="end">800</text>

      <!-- Hyperbolic Curves -->
      <!-- Curve 1: Tight Assembly (r_tol=5mm, sigma=2mm -> delta_r=3mm) -> Amber -->
      <path d="M 55,256 C 80,278 125,282 200,285 L 425,288" fill="none" stroke="#D97706" stroke-width="1.8" stroke-dasharray="4,2"/>
      <text class="code-text" x="210" y="280" fill="#B45309">r_tol = 5 mm (Precision Peg)</text>

      <!-- Curve 2: Compliant Grasp (r_tol=15mm, sigma=3mm -> delta_r=12mm) -> Cobalt (Highlighted) -->
      <path d="M 50,47 Q 80,140 125,168 Q 200,209 350,229 L 425,233" fill="none" stroke="#2563EB" stroke-width="2.6"/>
      <text class="code-text" x="230" y="195" fill="#1D4ED8" font-weight="700">r_tol = 15 mm (Compliant Grasp)</text>

      <!-- Curve 3: Loose Bin Retrieval (r_tol=30mm, sigma=3mm -> delta_r=27mm) -> Petrol -->
      <path d="M 60,20 Q 120,40 200,108 Q 275,155 425,195" fill="none" stroke="#007A87" stroke-width="1.8"/>
      <text class="code-text" x="315" y="165" fill="#007A87">r_tol = 30 mm (Loose Bin)</text>

      <!-- Operating Points from §10.4 -->
      <!-- Point A: Fast Conveyor (v=200 mm/s, tau=60 ms -> x=350, y=229) -->
      <circle cx="350" cy="229" r="4.5" fill="#DC2626"/>
      <line x1="350" y1="229" x2="350" y2="290" stroke="#DC2626" stroke-width="1" stroke-dasharray="2,2"/>
      <text class="small-text" x="355" y="222" fill="#DC2626" font-weight="700">Conveyor (v=200mm/s → τ≤60ms)</text>

      <!-- Point B: Slow Settling (v=20 mm/s, tau=600 ms -> x=80, y=87) -->
      <circle cx="80" cy="87" r="4.5" fill="#059669"/>
      <text class="small-text" x="90" y="85" fill="#047857" font-weight="700">Settling (v=20mm/s → τ≤600ms)</text>

      <!-- Fault Line: Coupling lease to VLM inference cadence (P99 = 650 ms) -->
      <line x1="50" y1="70" x2="425" y2="70" stroke="#DC2626" stroke-width="1.4" stroke-dasharray="4,3"/>
      <rect x="60" y="52" width="220" height="16" rx="3" fill="#FFF1F2" stroke="#DC2626" stroke-width="0.8"/>
      <text class="code-text" x="70" y="64" fill="#991B1B">P99 VLM Cadence = 650 ms (HAZARD)</text>
    </g>

    <!-- Formula Box at bottom of Panel A -->
    <g transform="translate(12, 368)">
      <rect width="436" height="78" rx="5" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <rect x="0" y="0" width="436" height="20" rx="4" fill="#F1F5F9"/>
      <text class="card-title" x="218" y="14" text-anchor="middle" fill="#1E293B">Mathematical Derivation: Maximum Safe TTL (§10.4)</text>
      
      <text class="code-text" x="20" y="38" fill="#1F407A" font-size="11">
        τ_max = (r_tol - σ_sensor) / v_drift
      </text>
      <text class="body-text" x="20" y="56">
        <tspan class="bold-text">Design Rule:</tspan> Lease validity is an invariant of <tspan class="bold-text" fill="#A51C30">scene physics</tspan>, not model compute speed.
        <tspan x="20" dy="14">If inference latency jitters (P99 &gt; τ_max), system must halt smoothly rather than stretch lease.</tspan>
      </text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- RIGHT PANEL: 4-STAGE INGESTION GATEWAY & DIAGNOSTIC FEEDBACK              -->
  <!-- ========================================================================= -->
  <g transform="translate(496, 66)">
    <rect width="466" height="458" rx="6" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.2"/>
    <rect x="12" y="10" width="210" height="20" rx="10" fill="#FEF2F2" stroke="#A51C30" stroke-width="1"/>
    <text class="badge-text" x="117" y="24" fill="#991B1B">INGESTION GATE · ARCHITECTURE</text>
    <text class="panel-hdr" x="232" y="25">PANEL B: EARLY REJECTION FILTER</text>

    <!-- Incoming Intent Proposal Box -->
    <g transform="translate(12, 38)">
      <rect width="442" height="38" rx="5" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2"/>
      <text class="badge-text" x="65" y="16" fill="#1D4ED8">SYSTEM 2 VLM / BRAIN</text>
      <text class="code-text" x="65" y="30" fill="#1E40AF">Intent Packet ⟨seq, t_src, T∈SE(3), Σ, τ, F_max⟩</text>
      
      <line x1="330" y1="19" x2="415" y2="19" stroke="#2563EB" stroke-width="1.5" marker-end="url(#arr2-blue)"/>
      <text class="small-text" x="372" y="14" text-anchor="middle" fill="#2563EB">168 Bytes</text>
    </g>

    <!-- The 4 Ingestion Filtering Stages -->
    <g transform="translate(12, 84)">
      <!-- Stage 1: Schema & Monotonicity -->
      <g transform="translate(0, 0)">
        <rect width="310" height="44" rx="4" fill="#FFFFFF" stroke="#007A87" stroke-width="1.2"/>
        <rect x="6" y="6" width="70" height="16" rx="3" fill="#F0FDFA"/>
        <text class="badge-text" x="41" y="18" fill="#007A87">&lt; 1 μs</text>
        <text class="card-title" x="85" y="18" fill="#0F766E">1. Monotonicity &amp; Hash Gate</text>
        <text class="small-text" x="85" y="34" fill="#475569">seq &gt; seq_curr ∧ parent_hash == ring_buffer[t_src]</text>

        <!-- Reject Arrow -->
        <line x1="310" y1="22" x2="350" y2="22" stroke="#DC2626" stroke-width="1.2" marker-end="url(#arr2-crimson)"/>
        <text class="small-text" x="356" y="25" fill="#DC2626" font-weight="600">Drop Stale/Replay</text>
      </g>

      <!-- Stage 2: Analytical Workspace & Occupancy Filter -->
      <g transform="translate(0, 52)">
        <rect width="310" height="44" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="1.2"/>
        <rect x="6" y="6" width="70" height="16" rx="3" fill="#F0F4FA"/>
        <text class="badge-text" x="41" y="18" fill="#1F407A">O(1) · 5 μs</text>
        <text class="card-title" x="85" y="18" fill="#1F407A">2. Kinematic Workspace &amp; Occupancy</text>
        <text class="small-text" x="85" y="34" fill="#475569">p_target ∈ 𝒲_reach ∧ Occupancy(p_target) == VALID</text>

        <!-- Reject Arrow -->
        <line x1="310" y1="22" x2="350" y2="22" stroke="#DC2626" stroke-width="1.2" marker-end="url(#arr2-crimson)"/>
        <text class="small-text" x="356" y="25" fill="#DC2626" font-weight="600">Out-of-Reach Reject</text>
      </g>

      <!-- Stage 3: Dynamic Time-to-Reach Filter -->
      <g transform="translate(0, 104)">
        <rect width="310" height="44" rx="4" fill="#FFFFFF" stroke="#D97706" stroke-width="1.2"/>
        <rect x="6" y="6" width="70" height="16" rx="3" fill="#FFFBEB"/>
        <text class="badge-text" x="41" y="18" fill="#D97706">&lt; 2 μs</text>
        <text class="card-title" x="85" y="18" fill="#B45309">3. Dynamic Reachability Bound</text>
        <text class="small-text" x="85" y="34" fill="#475569">t_min = 2√(d/a_max) ≤ τ_rem  (Under thermal limits)</text>

        <!-- Reject Arrow -->
        <line x1="310" y1="22" x2="350" y2="22" stroke="#DC2626" stroke-width="1.2" marker-end="url(#arr2-crimson)"/>
        <text class="small-text" x="356" y="25" fill="#DC2626" font-weight="600">Time Insufficient</text>
      </g>

      <!-- Stage 4: Semantic Ambiguity Gate -->
      <g transform="translate(0, 156)">
        <rect width="310" height="44" rx="4" fill="#FFFFFF" stroke="#7C3AED" stroke-width="1.2"/>
        <rect x="6" y="6" width="70" height="16" rx="3" fill="#F5F3FF"/>
        <text class="badge-text" x="41" y="18" fill="#7C3AED">&lt; 1 μs</text>
        <text class="card-title" x="85" y="18" fill="#6D28D9">4. Covariance &amp; Ambiguity Gate</text>
        <text class="small-text" x="85" y="34" fill="#475569">λ_max(Σ) ≤ σ²_max ∧ Softmax Entropy H(Y) ≤ H_thresh</text>

        <!-- Reject Arrow -->
        <line x1="310" y1="22" x2="350" y2="22" stroke="#DC2626" stroke-width="1.2" marker-end="url(#arr2-crimson)"/>
        <text class="small-text" x="356" y="25" fill="#DC2626" font-weight="600">Refusal on Distractor</text>
      </g>

      <!-- Connecting Downward Arrows between stages -->
      <line x1="155" y1="44" x2="155" y2="52" stroke="#1F407A" stroke-width="1.5" marker-end="url(#arr2-navy)"/>
      <line x1="155" y1="96" x2="155" y2="104" stroke="#1F407A" stroke-width="1.5" marker-end="url(#arr2-navy)"/>
      <line x1="155" y1="148" x2="155" y2="156" stroke="#1F407A" stroke-width="1.5" marker-end="url(#arr2-navy)"/>
    </g>

    <!-- Admission vs Rejection Dispatch -->
    <g transform="translate(12, 292)">
      <!-- Admitted Goal Card (Green) -->
      <rect x="0" y="0" width="215" height="65" rx="5" fill="#ECFDF5" stroke="#059669" stroke-width="1.4"/>
      <text class="card-title" x="12" y="18" fill="#065F46">✓ ADMITTED INTENT RECORD</text>
      <text class="small-text" x="12" y="34" fill="#047857">• Dispatched to 50 Hz Trajectory Optimizer</text>
      <text class="small-text" x="12" y="48" fill="#047857">• Enforced by 1000 Hz Real-Time MCU Shield</text>
      <text class="code-text" x="12" y="59" fill="#065F46">Total Ingestion Overhead: 42 μs (P99)</text>

      <!-- Diagnostic Feedback Loop (Red/Purple) -->
      <rect x="227" y="0" width="215" height="65" rx="5" fill="#FFF1F2" stroke="#DC2626" stroke-width="1.4"/>
      <text class="card-title" x="237" y="18" fill="#991B1B">✗ STRUCTURED REJECTION</text>
      <text class="small-text" x="237" y="34" fill="#7F1D1D">• Diagnostic Feedback to System 2 Reasoner</text>
      <text class="small-text" x="237" y="48" fill="#7F1D1D">• Returns Shortfall (t_min - τ) &amp; Margin</text>
      <text class="code-text" x="237" y="59" fill="#991B1B">Prevents Solver Queue Starvation</text>
    </g>

    <!-- Bottom Architectural Comparison Table -->
    <g transform="translate(12, 368)">
      <rect width="442" height="78" rx="5" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.2"/>
      <rect x="0" y="0" width="442" height="20" rx="4" fill="#F1F5F9"/>
      <text class="card-title" x="221" y="14" text-anchor="middle" fill="#1E293B">Computational Cost Asymmetry (§10.5)</text>

      <text class="body-text" x="12" y="36">
        <tspan class="bold-text" fill="#059669">Early Rejection Gate (Ingestion):</tspan> <tspan class="code-text">42 μs</tspan> scalar processor lookup. Zero queue bubble.
      </text>
      <text class="body-text" x="12" y="52">
        <tspan class="bold-text" fill="#DC2626">Late Rejection (Planner Interior):</tspan> <tspan class="code-text">15–80 ms</tspan> solver penalty search. Starves real-time pipeline.
      </text>
      <text class="small-text" x="12" y="68" fill="#64748B">
        Ingestion filtering guarantees downstream solver only executes kinematically solvable goals.
      </text>
    </g>
  </g>
</svg>
"""


def main():
    # 1. Write SVGs
    fig1_svg_path = os.path.join(CH10_FIG_DIR, "fig10_intent_lease_envelope.svg")
    fig1_pdf_path = os.path.join(CH10_FIG_DIR, "fig10_intent_lease_envelope.pdf")
    with open(fig1_svg_path, "w", encoding="utf-8") as f:
        f.write(FIG10_1_SVG.strip() + "\n")
    print(f"Wrote {fig1_svg_path}")

    fig2_svg_path = os.path.join(CH10_FIG_DIR, "fig10_lease_dynamics_tradeoff.svg")
    fig2_pdf_path = os.path.join(CH10_FIG_DIR, "fig10_lease_dynamics_tradeoff.pdf")
    with open(fig2_svg_path, "w", encoding="utf-8") as f:
        f.write(FIG10_2_SVG.strip() + "\n")
    print(f"Wrote {fig2_svg_path}")

    # 2. Compile to PDF using rsvg-convert
    for svg, pdf in [(fig1_svg_path, fig1_pdf_path), (fig2_svg_path, fig2_pdf_path)]:
        try:
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", pdf, svg], check=True)
            print(f"Compiled {pdf}")
        except Exception as e:
            print(f"Error compiling {pdf}: {e}")


if __name__ == "__main__":
    main()
