import os

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="100%" height="100%">
  <defs>
    <marker id="arrow-down" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2D3748"/>
    </marker>
    <marker id="arrow-down-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#A51C30"/>
    </marker>
  </defs>

  <style>
    .title { font-family: sans-serif; font-size: 20px; font-weight: bold; fill: #1F407A; }
    .rung-title { font-family: sans-serif; font-size: 18px; font-weight: bold; }
    .rung-text { font-family: sans-serif; font-size: 14px; }
    .rung-sub { font-family: monospace; font-size: 11px; }
    .axis-label { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #2D3748; }
  </style>

  <!-- Title -->
  <text x="500" y="40" text-anchor="middle" class="title">The Deterministic Four-Tier Fallback Escalation Ladder</text>

  <!-- Axes/Arrows -->
  <!-- Left downward arrow -->
  <line x1="130" y1="100" x2="130" y2="520" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow-down)"/>
  <text x="110" y="310" text-anchor="middle" transform="rotate(-90 110 310)" class="axis-label">Failure Severity / Loss of Trust</text>
  <text x="150" y="310" text-anchor="middle" transform="rotate(-90 150 310)" class="axis-label" fill="#A51C30">Decreasing Reaction Latency (&#x03BC;s)</text>

  <!-- Right downward arrow -->
  <line x1="870" y1="100" x2="870" y2="520" stroke="#2D3748" stroke-width="3" marker-end="url(#arrow-down)"/>
  <text x="890" y="310" text-anchor="middle" transform="rotate(90 890 310)" class="axis-label">Increasing Mechanical Wear &amp; Stress</text>
  <text x="850" y="310" text-anchor="middle" transform="rotate(90 850 310)" class="axis-label" fill="#A51C30">Loss of Mission Context</text>

  <!-- Rung 1 -->
  <g transform="translate(225, 80)">
    <rect x="0" y="0" width="550" height="90" rx="4" fill="#E2E8F0" stroke="#1F407A" stroke-width="2"/>
    <text x="20" y="30" class="rung-title" fill="#1F407A">Rung 1: Active QP Projection (SLS/SLF)</text>
    <text x="20" y="55" class="rung-text" fill="#2D3748">Sub-millisecond constraint projection. Keeps mission running.</text>
    <text x="20" y="75" class="rung-sub" fill="#2D3748">Substrate: Static MCU SRAM | Latency: &lt; 175 &#x03BC;s | Wear: Zero</text>
  </g>

  <!-- Rung 2 -->
  <g transform="translate(225, 190)">
    <rect x="0" y="0" width="550" height="90" rx="4" fill="#FFFFFF" stroke="#1F407A" stroke-width="2"/>
    <text x="20" y="30" class="rung-title" fill="#1F407A">Rung 2: Active Closed-Loop Hold (Category 2 / SS2)</text>
    <text x="20" y="55" class="rung-text" fill="#2D3748">Dissipates energy as Joule heating. Preserves context.</text>
    <text x="20" y="75" class="rung-sub" fill="#2D3748">Substrate: Real-time RTOS | Latency: 2-5 ms | Wear: Electrical</text>
  </g>

  <!-- Rung 3 -->
  <g transform="translate(225, 300)">
    <rect x="0" y="0" width="550" height="90" rx="4" fill="#FFFFFF" stroke="#A51C30" stroke-width="2"/>
    <text x="20" y="30" class="rung-title" fill="#A51C30">Rung 3: Controlled Dynamic Stop (Category 1 / SS1)</text>
    <text x="20" y="55" class="rung-text" fill="#2D3748">Absorbs regenerative DC-bus energy. Task abort &amp; re-homing.</text>
    <text x="20" y="75" class="rung-sub" fill="#A51C30">Substrate: Safety MCU Firmware | Latency: 10-20 ms | Wear: Regen load</text>
  </g>

  <!-- Rung 4 -->
  <g transform="translate(225, 410)">
    <rect x="0" y="0" width="550" height="90" rx="4" fill="#A51C30" stroke="#A51C30" stroke-width="2"/>
    <text x="20" y="30" class="rung-title" fill="#FFFFFF">Rung 4: Hardware Safe Torque Off (Category 0 / STO)</text>
    <text x="20" y="55" class="rung-text" fill="#E2E8F0">Galvanic isolation &amp; mechanical friction brakes. Manual reset.</text>
    <text x="20" y="75" class="rung-sub" fill="#E2E8F0">Substrate: Hardware/Optical | Latency: &lt; 5 &#x03BC;s | Wear: Lining ablation</text>
  </g>

  <!-- Connecting Arrows Between Rungs -->
  <line x1="500" y1="170" x2="500" y2="190" stroke="#2D3748" stroke-width="2" marker-end="url(#arrow-down)"/>
  <line x1="500" y1="280" x2="500" y2="300" stroke="#2D3748" stroke-width="2" marker-end="url(#arrow-down)"/>
  <line x1="500" y1="390" x2="500" y2="410" stroke="#A51C30" stroke-width="2" marker-end="url(#arrow-down-red)"/>
</svg>
"""

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svg'), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svg', 'fig12_fallback_ladder.svg'), 'w') as f:
    f.write(svg_content)

print("fig12_fallback_ladder.svg generated")
