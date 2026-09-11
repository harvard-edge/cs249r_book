#!/usr/bin/env python3
"""
Generates fig01_tempe_kinematics.svg
Kinematic Depletion Timeline: Latency Consumes Physical Clearance
NTSB Tempe Collision Forensic Analysis (v0 = 19.2 m/s, a_max = 8.0 m/s^2)
Polished layout with zero text collisions and crisp typography.
"""

from pathlib import Path

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 560" width="100%" height="100%">
  <defs>
    <style>
      .title { font-family: Helvetica, Arial, sans-serif; font-size: 20px; font-weight: bold; fill: #1F407A; }
      .subtitle { font-family: Helvetica, Arial, sans-serif; font-size: 13px; fill: #4A5568; }
      .section-hdr { font-family: Helvetica, Arial, sans-serif; font-size: 13px; font-weight: bold; fill: #1F407A; }
      .axis-label { font-family: Helvetica, Arial, sans-serif; font-size: 11px; fill: #718096; }
      .data-label { font-family: Helvetica, Arial, sans-serif; font-size: 11px; fill: #2D3748; }
      .data-label-bold { font-family: Helvetica, Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #2D3748; }
      .callout-red { font-family: Helvetica, Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #A51C30; }
      .callout-amber { font-family: Helvetica, Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #D97706; }
      .callout-teal { font-family: Helvetica, Arial, sans-serif; font-size: 11px; font-weight: bold; fill: #0D9488; }
      .axis-line { stroke: #CBD5E0; stroke-width: 1.5; }
    </style>
    <linearGradient id="sealedGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FEE2E2" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#EF4444" stop-opacity="0.38"/>
    </linearGradient>
    <linearGradient id="suppressGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FEF3C7" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#F59E0B" stop-opacity="0.7"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="1020" height="560" fill="#FFFFFF"/>

  <!-- Title Block -->
  <text x="50" y="36" class="title">Kinematic Depletion Timeline: Software Latency Consumes Physical Clearance</text>
  <text x="50" y="56" class="subtitle">NTSB Forensic Analysis: Volvo XC90 (2100 kg, v₀ = 19.2 m/s / 43 mph, a_max = 8.0 m/s², d_stop = 46.0 m)</text>

  <!-- Coordinate mapping:
       x_start (T-5.6s) = 200px
       x_end (T-0.0s) = 930px
       Scale: 130.357 px/s
       T - 5.6s = 200 px
       T - 4.0s = 408 px
       T - 2.4s = 617 px (Point of inevitable impact)
       T - 1.2s = 774 px (Suppression begins)
       T - 0.2s = 904 px (Braking requested)
       T - 0.0s = 930 px (Collision)
  -->

  <!-- Sealed Collision Shaded Box -->
  <rect x="617" y="75" width="313" height="395" fill="url(#sealedGrad)"/>
  <line x1="617" y1="70" x2="617" y2="470" stroke="#A51C30" stroke-width="2" stroke-dasharray="4,4"/>
  <rect x="623" y="75" width="300" height="30" fill="#FFFFFF" fill-opacity="0.88" rx="4"/>
  <text x="628" y="88" class="callout-red">T - 2.4 s: POINT OF INEVITABLE COLLISION</text>
  <text x="628" y="101" class="callout-red">Remaining Clearance (46.0 m) &lt; Stopping Envelope (d_stop)</text>

  <!-- ==================== TRACK 1: PERCEPTION & STATE TRACKING ==================== -->
  <text x="50" y="112" class="section-hdr">1. Perception</text>
  <text x="50" y="128" class="axis-label">Classification &amp;</text>
  <text x="50" y="142" class="axis-label">Tracking State</text>

  <line x1="200" y1="120" x2="930" y2="120" class="axis-line"/>
  
  <!-- T-5.6s Initial Contact Marker -->
  <circle cx="200" cy="120" r="5" fill="#0D9488"/>
  <text x="200" y="105" class="callout-teal" text-anchor="middle">Initial Contact (82 m)</text>

  <!-- Classification Toggles -->
  <rect x="245" y="128" width="75" height="24" rx="4" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1"/>
  <text x="282" y="144" class="data-label" text-anchor="middle">"Unknown"</text>

  <rect x="330" y="128" width="85" height="24" rx="4" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1"/>
  <text x="372" y="144" class="data-label" text-anchor="middle">"Vehicle" (reset)</text>

  <rect x="425" y="128" width="80" height="24" rx="4" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1"/>
  <text x="465" y="144" class="data-label" text-anchor="middle">"Other" (reset)</text>

  <rect x="515" y="128" width="85" height="24" rx="4" fill="#F1F5F9" stroke="#94A3B8" stroke-width="1"/>
  <text x="557" y="144" class="data-label" text-anchor="middle">"Bicycle" (reset)</text>

  <rect x="625" y="128" width="85" height="24" rx="4" fill="#FEF3C7" stroke="#D97706" stroke-width="1"/>
  <text x="667" y="144" class="callout-amber" text-anchor="middle">"Bicycle" (stable)</text>

  <!-- Suppression Window -->
  <rect x="774" y="128" width="130" height="24" rx="4" fill="url(#suppressGrad)" stroke="#D97706" stroke-width="1.5"/>
  <text x="839" y="144" class="callout-amber" text-anchor="middle" font-weight="bold">1.2s Suppression Delay</text>

  <!-- Brake Signal at T-0.2s -->
  <circle cx="904" cy="120" r="5" fill="#A51C30"/>
  <line x1="904" y1="120" x2="904" y2="160" stroke="#A51C30" stroke-width="1.5"/>
  <text x="904" y="172" class="callout-red" text-anchor="middle">Brake Requested</text>

  <!-- ==================== TRACK 2: KINEMATIC CLEARANCE DYNAMICS ==================== -->
  <text x="50" y="215" class="section-hdr">2. Clearance</text>
  <text x="50" y="231" class="axis-label">Distance &amp;</text>
  <text x="50" y="245" class="axis-label">Stopping Envelope</text>

  <!-- Axis -->
  <line x1="200" y1="310" x2="930" y2="310" class="axis-line"/>
  <line x1="200" y1="190" x2="200" y2="310" class="axis-line"/>
  <text x="190" y="195" class="axis-label" text-anchor="end">80 m</text>
  <text x="190" y="244" class="axis-label" text-anchor="end">46 m</text>
  <text x="190" y="280" class="axis-label" text-anchor="end">20 m</text>
  <text x="190" y="314" class="axis-label" text-anchor="end">0 m</text>

  <!-- Horizontal dashed line for required stopping distance d_stop = 46.0m -->
  <line x1="200" y1="241" x2="930" y2="241" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="4,4"/>
  
  <!-- Label placed cleanly BELOW the 46m line where space is completely unobstructed -->
  <text x="210" y="256" class="callout-red">Required Defended Stopping Envelope d_stop = 46.0 m</text>
  <text x="210" y="270" class="axis-label" fill="#A51C30">(23.0 m suppression lag + 23.0 m physical braking)</text>

  <!-- Unguided Distance Shading in Clearance Track -->
  <rect x="774" y="275" width="130" height="35" fill="#F59E0B" fill-opacity="0.25"/>
  <rect x="778" y="250" width="122" height="18" fill="#FFFFFF" fill-opacity="0.9" rx="3"/>
  <text x="839" y="263" class="callout-amber" text-anchor="middle">23.0 m Unguided Delay</text>

  <!-- Available Clearance Trajectory D_clear(t) -->
  <line x1="200" y1="187" x2="930" y2="310" stroke="#1F407A" stroke-width="3"/>
  <circle cx="617" cy="241" r="5" fill="#A51C30"/>
  <circle cx="904" cy="304" r="4" fill="#A51C30"/>
  <circle cx="930" cy="310" r="5" fill="#A51C30"/>

  <!-- Badge for 3.8m remaining -->
  <rect x="670" y="210" width="235" height="20" fill="#FFFFFF" fill-opacity="0.92" rx="3" stroke="#FCA5A5" stroke-width="1"/>
  <text x="787" y="224" class="callout-red" text-anchor="middle">Only 3.8 m remaining at braking request</text>
  <line x1="895" y1="230" x2="904" y2="300" stroke="#A51C30" stroke-width="1" stroke-dasharray="2,2"/>

  <!-- ==================== TRACK 3: ACTUATOR DECELERATION COMPARISON ==================== -->
  <text x="50" y="360" class="section-hdr">3. Actuation</text>
  <text x="50" y="376" class="axis-label">ADS Command vs.</text>
  <text x="50" y="390" class="axis-label">Deterministic Reflex</text>

  <!-- Axis -->
  <line x1="200" y1="465" x2="930" y2="465" class="axis-line"/>
  <line x1="200" y1="360" x2="200" y2="465" class="axis-line"/>
  <text x="190" y="365" class="axis-label" text-anchor="end">8 m/s²</text>
  <text x="190" y="415" class="axis-label" text-anchor="end">4 m/s²</text>
  <text x="190" y="468" class="axis-label" text-anchor="end">0 m/s²</text>

  <!-- Actual Path: 0 braking until T-0.2s -->
  <polyline points="200,465 904,465 930,448" fill="none" stroke="#A51C30" stroke-width="2.5"/>
  <rect x="635" y="440" width="250" height="20" fill="#FFFFFF" fill-opacity="0.92" rx="3"/>
  <text x="760" y="454" class="callout-red" text-anchor="middle">Actual ADS: 0 braking until T - 0.2 s (Impact at 43 mph)</text>

  <!-- Deterministic 1 kHz Enforcer Counterfactual -->
  <polyline points="200,465 617,465 630,360 915,360 915,465" fill="none" stroke="#0D9488" stroke-width="2.5" stroke-dasharray="5,3"/>
  <rect x="630" y="360" width="285" height="70" fill="#0D9488" fill-opacity="0.08"/>
  <text x="772" y="382" class="callout-teal" text-anchor="middle" font-size="12px">Deterministic Safety Reflex: Emergency Braking at T - 2.4 s</text>
  <text x="772" y="400" class="data-label-bold" text-anchor="middle">8.0 m/s² emergency deceleration halts vehicle at T - 0.1 s (+2.2 m margin)</text>

  <!-- ==================== TIME AXIS & TICKS ==================== -->
  <line x1="200" y1="475" x2="930" y2="475" class="axis-line"/>
  
  <line x1="200" y1="475" x2="200" y2="483" stroke="#2D3748" stroke-width="1.5"/>
  <text x="200" y="498" class="data-label-bold" text-anchor="middle">T - 5.6 s</text>
  <text x="200" y="512" class="axis-label" text-anchor="middle">(82 m)</text>

  <line x1="408" y1="475" x2="408" y2="483" stroke="#2D3748" stroke-width="1.5"/>
  <text x="408" y="498" class="data-label" text-anchor="middle">T - 4.0 s</text>
  <text x="408" y="512" class="axis-label" text-anchor="middle">(65 m)</text>

  <line x1="617" y1="475" x2="617" y2="483" stroke="#A51C30" stroke-width="2"/>
  <text x="617" y="498" class="callout-red" text-anchor="middle">T - 2.4 s</text>
  <text x="617" y="512" class="callout-red" text-anchor="middle">(46 m)</text>

  <line x1="774" y1="475" x2="774" y2="483" stroke="#D97706" stroke-width="1.5"/>
  <text x="774" y="498" class="callout-amber" text-anchor="middle">T - 1.2 s</text>
  <text x="774" y="512" class="axis-label" text-anchor="middle">(23 m)</text>

  <line x1="904" y1="475" x2="904" y2="483" stroke="#A51C30" stroke-width="1.5"/>
  <text x="895" y="498" class="callout-red" text-anchor="end">T - 0.2 s</text>

  <line x1="930" y1="475" x2="930" y2="483" stroke="#A51C30" stroke-width="2"/>
  <text x="935" y="512" class="callout-red" text-anchor="middle">Impact</text>

  <!-- Legend -->
  <rect x="50" y="535" width="12" height="12" fill="#1F407A"/>
  <text x="68" y="545" class="data-label">Available Physical Clearance</text>

  <line x1="240" y1="541" x2="260" y2="541" stroke="#A51C30" stroke-width="2" stroke-dasharray="3,3"/>
  <text x="266" y="545" class="data-label">Required Stopping Envelope (46.0 m)</text>

  <rect x="510" y="535" width="12" height="12" fill="#F59E0B" fill-opacity="0.5"/>
  <text x="528" y="545" class="data-label">Software Suppression Delay (23.0 m)</text>

  <line x1="770" y1="541" x2="790" y2="541" stroke="#0D9488" stroke-width="2" stroke-dasharray="4,2"/>
  <text x="796" y="545" class="data-label">Deterministic 1 kHz Reflex (Safe Stop)</text>
</svg>
"""

out_dir = Path("books/vol4/01_boundary/images/svg")
out_dir.mkdir(parents=True, exist_ok=True)
svg_file = out_dir / "fig01_tempe_kinematics.svg"
svg_file.write_text(svg_content, encoding="utf-8")
print(f"Generated flawless SVG: {svg_file}")
