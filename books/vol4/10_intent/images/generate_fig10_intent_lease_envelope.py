import sys

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600" width="1000" height="600" style="background-color: white; font-family: sans-serif;">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2D3748" />
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1F407A" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#A51C30" />
    </marker>
    <style>
      .title { font-size: 16px; font-weight: bold; fill: #2D3748; }
      .text { font-size: 14px; fill: #2D3748; }
      .text-small { font-size: 12px; fill: #2D3748; }
      .text-bold { font-size: 14px; font-weight: bold; fill: #2D3748; }
      .axis { stroke: #2D3748; stroke-width: 1.5; }
      .grid { stroke: #E2E8F0; stroke-width: 1; stroke-dasharray: 4 4; }
      .ellipse { fill: #E2E8F0; stroke: #1F407A; stroke-width: 2; fill-opacity: 0.5; stroke-dasharray: 5 5; }
      .ellipse-inner { fill: none; stroke: #1F407A; stroke-width: 1; }
      .vector-blue { stroke: #1F407A; stroke-width: 2.5; marker-end: url(#arrow-blue); }
      .vector-red { stroke: #A51C30; stroke-width: 2.5; marker-end: url(#arrow-red); }
      .vector-dark { stroke: #2D3748; stroke-width: 2.5; marker-end: url(#arrow); }
      .line-blue { stroke: #1F407A; stroke-width: 2.5; }
      .line-red { stroke: #A51C30; stroke-width: 2.5; }
      .line-dark { stroke: #2D3748; stroke-width: 2; stroke-dasharray: 4 4; }
      .line-green { stroke: #38A169; stroke-width: 2.5; }
    </style>
  </defs>

  <!-- Panel A: Spatial Tolerance Volume -->
  <text x="40" y="40" class="title">Panel A: Spatial Tolerance Volume</text>
  <rect x="40" y="60" width="400" height="500" rx="4" fill="none" stroke="#2D3748" stroke-width="1.5" />
  
  <g transform="translate(240, 310)">
    <!-- Ellipse representing covariance -->
    <ellipse cx="0" cy="0" rx="80" ry="140" class="ellipse" transform="rotate(15)" />
    <!-- Axes of the ellipse -->
    <line x1="-77.2" y1="-20.7" x2="77.2" y2="20.7" class="line-dark" />
    <line x1="-36.2" y1="135.2" x2="36.2" y2="-135.2" class="line-dark" />
    
    <!-- Target Point -->
    <circle cx="0" cy="0" r="5" fill="#1F407A" />
    <text x="20" y="5" class="text-bold" fill="#1F407A">p₀ (Nominal Target)</text>
    
    <!-- Semi-axes labels -->
    <text x="75" y="-20" class="text-small">a_x = 11.2 mm</text>
    <text x="85" y="45" class="text-small">a_y = 16.8 mm</text>
    <text x="-60" y="125" class="text-small">a_z = 41.9 mm</text>
    
    <!-- Approach velocity -->
    <line x1="-120" y1="-150" x2="-20" y2="-25" class="vector-blue" />
    <text x="-160" y="-160" class="text-bold" fill="#1F407A">v_approach ≤ 0.25 m/s</text>
    
    <!-- Wrench envelope -->
    <rect x="-35" y="-35" width="70" height="70" fill="none" stroke="#A51C30" stroke-width="2" stroke-dasharray="2 2" transform="rotate(15)" />
    <line x1="0" y1="0" x2="0" y2="-45" class="vector-red" transform="rotate(15) translate(0, 45)" />
    <text x="-140" y="60" class="text-bold" fill="#A51C30">F_max ≤ 12 N</text>
    <text x="-140" y="75" class="text-small" fill="#A51C30">(Admissible Wrench)</text>
    
    <!-- Spatial fault text -->
    <text x="-180" y="220" class="text-small">Scene drift past tolerance boundary</text>
    <text x="-180" y="235" class="text-small">registers a spatial fault.</text>
  </g>

  <!-- Panel B: Temporal Expiration Timelines -->
  <text x="480" y="40" class="title">Panel B: Temporal Expiration Timelines</text>
  <rect x="480" y="60" width="480" height="500" rx="4" fill="none" stroke="#2D3748" stroke-width="1.5" />
  
  <g transform="translate(500, 110)">
    <!-- Case 1 -->
    <text x="0" y="0" class="text-bold">Case 1: Nominal Continuous Lease Renewal</text>
    <!-- Axis -->
    <line x1="20" y1="40" x2="440" y2="40" class="axis" marker-end="url(#arrow)" />
    <text x="420" y="30" class="text-small">Time</text>
    <!-- Renewals -->
    <line x1="40" y1="35" x2="40" y2="45" class="axis" />
    <text x="35" y="25" class="text-small">0ms</text>
    <circle cx="40" cy="40" r="4" fill="#38A169" />
    
    <line x1="140" y1="35" x2="140" y2="45" class="axis" />
    <text x="125" y="25" class="text-small">100ms</text>
    <circle cx="140" cy="40" r="4" fill="#38A169" />
    
    <line x1="240" y1="35" x2="240" y2="45" class="axis" />
    <text x="225" y="25" class="text-small">200ms</text>
    <circle cx="240" cy="40" r="4" fill="#38A169" />
    
    <line x1="340" y1="35" x2="340" y2="45" class="axis" />
    <text x="325" y="25" class="text-small">300ms</text>
    <circle cx="340" cy="40" r="4" fill="#38A169" />
    
    <text x="40" y="65" class="text-small" fill="#38A169">C²-smooth quintic spline tracking</text>
    
    <!-- Velocity curve -->
    <path d="M 40 80 Q 240 60 400 80" fill="none" stroke="#38A169" stroke-width="2" />
  </g>

  <g transform="translate(500, 260)">
    <!-- Case 2 -->
    <text x="0" y="0" class="text-bold">Case 2: Silent Host Reasoner Hang (Safe)</text>
    <!-- Axis -->
    <line x1="20" y1="40" x2="440" y2="40" class="axis" marker-end="url(#arrow)" />
    <!-- Events -->
    <line x1="40" y1="35" x2="40" y2="45" class="axis" />
    <text x="35" y="25" class="text-small">0ms</text>
    <circle cx="40" cy="40" r="4" fill="#38A169" />
    <text x="10" y="65" class="text-small" fill="#38A169">Lease Issued</text>
    
    <line x1="120" y1="35" x2="120" y2="45" class="axis" />
    <text x="110" y="25" class="text-small">80ms</text>
    <circle cx="120" cy="40" r="4" fill="#A51C30" />
    <text x="90" y="65" class="text-small" fill="#A51C30">Host Hangs</text>
    <text x="90" y="100" class="text-small" fill="#A51C30">(No Abort Msg)</text>
    
    <line x1="190" y1="35" x2="190" y2="45" class="axis" />
    <text x="180" y="25" class="text-small">150ms</text>
    <circle cx="190" cy="40" r="4" fill="#E53E3E" />
    <text x="170" y="65" class="text-small" fill="#E53E3E">TTL Expires</text>
    
    <line x1="302" y1="35" x2="302" y2="45" class="axis" />
    <text x="290" y="25" class="text-small">262ms</text>
    
    <path d="M 40 80 L 190 80 Q 246 100 302 120" fill="none" stroke="#1F407A" stroke-width="2.5" />
    <line x1="302" y1="120" x2="400" y2="120" class="line-blue" />
    
    <text x="210" y="140" class="text-small" fill="#1F407A">Autonomous Deceleration</text>
    <text x="210" y="155" class="text-small" fill="#1F407A">t_decel = 112ms, d_stop = 28mm</text>
    <text x="310" y="95" class="text-small" fill="#1F407A">Active Hold</text>
  </g>

  <g transform="translate(500, 440)">
    <!-- Case 3 -->
    <text x="0" y="0" class="text-bold">Case 3: Unbounded Goal Anti-pattern (Hazard)</text>
    <!-- Axis -->
    <line x1="20" y1="40" x2="440" y2="40" class="axis" marker-end="url(#arrow)" />
    <!-- Events -->
    <line x1="40" y1="35" x2="40" y2="45" class="axis" />
    <text x="35" y="25" class="text-small">0ms</text>
    <circle cx="40" cy="40" r="4" fill="#38A169" />
    <text x="10" y="65" class="text-small" fill="#38A169">Goal Sent</text>
    
    <line x1="120" y1="35" x2="120" y2="45" class="axis" />
    <text x="110" y="25" class="text-small">80ms</text>
    <circle cx="120" cy="40" r="4" fill="#A51C30" />
    <text x="90" y="65" class="text-small" fill="#A51C30">Host Hangs</text>
    
    <line x1="320" y1="35" x2="320" y2="45" class="axis" />
    
    <path d="M 40 80 L 320 80 L 320 30" fill="none" stroke="#A51C30" stroke-width="2.5" />
    
    <text x="270" y="20" class="text-bold" fill="#A51C30">IMPACT!</text>
    <text x="200" y="100" class="text-small" fill="#A51C30">Open-loop continuation at 1.4 m/s</text>
    <text x="250" y="115" class="text-small" fill="#A51C30">F_peak = 1420 N</text>
  </g>

</svg>
"""

with open("/Users/VJ/GitHub/MLSysBook/books/vol4/10_intent/images/svg/fig10_intent_lease_envelope.svg", "w") as f:
    f.write(svg_content)
