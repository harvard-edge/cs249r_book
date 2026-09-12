import os

svg_template = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800" width="100%" height="100%">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1F407A" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#A51C30" />
    </marker>
  </defs>

  <style>
    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 28px; font-weight: bold; fill: #1F407A; }
    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 18px; fill: #2D3748; }
    
    .node-rect { fill: #E2E8F0; stroke: #1F407A; stroke-width: 2; }
    .node-rect-red { fill: #FFF0F0; stroke: #A51C30; stroke-width: 2; }
    .node-rect-green { fill: #F0FDF4; stroke: #15803D; stroke-width: 2; }
    
    .node-text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 20px; font-weight: bold; fill: #1F407A; text-anchor: middle; }
    .node-text-red { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 20px; font-weight: bold; fill: #A51C30; text-anchor: middle; }
    .node-text-green { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 20px; font-weight: bold; fill: #15803D; text-anchor: middle; }
    
    .node-desc { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; fill: #2D3748; text-anchor: middle; }
    
    .path-line { fill: none; stroke: #1F407A; stroke-width: 3; marker-end: url(#arrow); }
    .path-line-feedback { fill: none; stroke: #A51C30; stroke-width: 4; marker-end: url(#arrow-red); stroke-dasharray: 8,6; }
    
    .glass-text { font-family: monospace; font-size: 16px; fill: #2D3748; font-weight: bold; }
  </style>

  <rect width="100%" height="100%" fill="#ffffff" />

  <!-- (A) Open-Loop Digital Computation -->
  <g transform="translate(60, 60)">
    <text x="0" y="0" class="title">A) Open-Loop Digital Computation</text>
    <text x="0" y="30" class="subtitle">Exogenous data flows to prediction behind a glass barrier. Errors are idempotent.</text>
    
    <!-- Flat 2D Layout for Flow A -->
    <g transform="translate(100, 80)">
        <!-- Exogenous Data -->
        <g transform="translate(0, 0)">
          <rect x="0" y="0" width="180" height="120" class="node-rect" />
          <text x="90" y="55" class="node-text">Exogenous Data</text>
          <text x="90" y="80" class="node-desc">Benchmark dataset</text>
        </g>
        
        <!-- Arrow -->
        <path fill="none" d="M 180 60 L 320 60" class="path-line" />
        
        <!-- ML Model -->
        <g transform="translate(320, 0)">
          <rect x="0" y="0" width="180" height="120" class="node-rect" />
          <text x="90" y="55" class="node-text">ML Model</text>
          <text x="90" y="80" class="node-desc">Digital Inference</text>
        </g>
        
        <!-- Arrow -->
        <path fill="none" d="M 500 60 L 640 60" class="path-line" />
        
        <!-- Glass Barrier -->
        <line x1="570" y1="-20" x2="570" y2="140" stroke="#2D3748" stroke-width="3" stroke-dasharray="10,5" />
        <text x="570" y="-30" class="glass-text" text-anchor="middle">GLASS BARRIER</text>
        
        <!-- Prediction -->
        <g transform="translate(640, 0)">
          <rect x="0" y="0" width="180" height="120" class="node-rect" />
          <text x="90" y="55" class="node-text">Prediction</text>
          <text x="90" y="80" class="node-desc">No Physical Effect</text>
        </g>
    </g>
  </g>

  <!-- (B) Closed Causal Loop of Physical AI -->
  <g transform="translate(60, 400)">
    <text x="0" y="0" class="title">B) Closed Causal Loop of Physical AI</text>
    <text x="0" y="30" class="subtitle">Endogenous feedback: Model actions change physical environment, directly altering next sensor input.</text>
    
    <!-- Flat 2D Layout for Flow B -->
    <g transform="translate(10, 80)">
        
        <!-- Sensors -->
        <g transform="translate(0, 0)">
          <rect x="0" y="0" width="180" height="100" class="node-rect" />
          <text x="90" y="45" class="node-text">Sensors</text>
          <text x="90" y="70" class="node-desc">Endogenous Input</text>
        </g>
        
        <!-- Arrow to Deliberation -->
        <path fill="none" d="M 180 50 L 260 50" class="path-line" />
        
        <!-- Neural Deliberation -->
        <g transform="translate(260, 0)">
          <rect x="0" y="0" width="200" height="100" class="node-rect" />
          <text x="100" y="45" class="node-text">Neural Deliberation</text>
          <text x="100" y="70" class="node-desc">Action Proposal</text>
        </g>
        
        <!-- Arrow to Safety Gate -->
        <path fill="none" d="M 460 50 L 540 50" class="path-line" />
        
        <!-- Safety Gate -->
        <g transform="translate(540, 0)">
          <rect x="0" y="0" width="180" height="100" class="node-rect-red" />
          <text x="90" y="45" class="node-text-red">Safety Gate</text>
          <text x="90" y="70" class="node-desc">Deterministic Filter</text>
        </g>
        
        <!-- Arrow to Motors -->
        <path fill="none" d="M 720 50 L 800 50" class="path-line" />
        
        <!-- Motors/Actuators -->
        <g transform="translate(800, 0)">
          <rect x="0" y="0" width="180" height="100" class="node-rect" />
          <text x="90" y="45" class="node-text">Motors</text>
          <text x="90" y="70" class="node-desc">Hardware Actuation</text>
        </g>
        
        <!-- Arrow to Environment -->
        <path fill="none" d="M 890 100 L 890 170 L 760 170" class="path-line" />
        
        <!-- Physical Environment -->
        <g transform="translate(260, 120)">
          <rect x="0" y="0" width="500" height="100" class="node-rect-green" />
          <text x="250" y="45" class="node-text-green">Physical Environment</text>
          <text x="250" y="70" class="node-desc">State altered by physical actuation</text>
        </g>
        
        <!-- Feedback Arrow from Environment to Sensors -->
        <path fill="none" d="M 260 170 L 90 170 L 90 100" class="path-line-feedback" />
    </g>
  </g>
</svg>"""

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svg")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig01_causal_loop.svg")
with open(out_path, "w") as f:
    f.write(svg_template)

print(f"Generated {out_path}")
