import os

def generate_svg():
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 800" width="1000" height="800">
    <defs>
        <style>
            .title { font-family: sans-serif; font-size: 18px; font-weight: bold; fill: #1F407A; }
            .subtitle { font-family: sans-serif; font-size: 14px; font-weight: bold; fill: #2D3748; }
            .label { font-family: sans-serif; font-size: 12px; fill: #2D3748; }
            .label-bold { font-family: sans-serif; font-size: 12px; font-weight: bold; fill: #2D3748; }
            .milestone-line { stroke: #A0AEC0; stroke-width: 1.5; stroke-dasharray: 4,4; }
            .axis-line { stroke: #2D3748; stroke-width: 1.5; }
            .chunk-a { fill: #E2E8F0; stroke: #4A5568; stroke-width: 2; }
            .chunk-b { fill: #BEE3F8; stroke: #3182CE; stroke-width: 2; }
            .fallback-zone { fill: #FED7D7; stroke: #C53030; stroke-width: 2; }
            .blend-window { fill: #C6F6D5; opacity: 0.7; }
            .callout-box { fill: #FFFFFF; stroke: #CBD5E0; stroke-width: 1; rx: 4; ry: 4; }
            .callout-text { font-family: sans-serif; font-size: 11px; fill: #2D3748; }
            .card-title { font-family: sans-serif; font-size: 12px; font-weight: bold; fill: #1F407A; }
            .card-error { font-family: sans-serif; font-size: 12px; font-weight: bold; fill: #C53030; }
        </style>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#2D3748" />
        </marker>
    </defs>

    <rect width="1000" height="800" fill="#FFFFFF" />
    <text x="500" y="30" class="title" text-anchor="middle">Action Chunk Seam Timeline &amp; Fallback Deceleration Dynamics</text>
    
    <!-- Time Axis scale: 0 to 600 ms, map 0->100px, 600->900px, scale = (900-100)/600 = 1.33 px/ms -->
    <!-- X offsets:
         t_req = 0 -> 100
         P50 = 180 -> 340
         t_blend = 300 -> 500
         P99 = 340 -> 553.3
         t_commit = 360 -> 580
         t_exp = 400 -> 633.3
         T_chunk = 500 -> 766.6
         t_term = 560 -> 846.6
    -->
    
    <g transform="translate(0, 80)">
        <text x="50" y="0" class="subtitle">Timeline Milestones</text>
        <line x1="100" y1="20" x2="900" y2="20" class="axis-line" marker-end="url(#arrowhead)" />
        
        <!-- Milestones -->
        <!-- t_req -->
        <line x1="100" y1="15" x2="100" y2="500" class="milestone-line" />
        <text x="100" y="10" class="label-bold" text-anchor="middle">t_req=0</text>
        
        <!-- t_blend -->
        <line x1="500" y1="15" x2="500" y2="500" class="milestone-line" />
        <text x="500" y="10" class="label-bold" text-anchor="middle">t_blend=300</text>
        
        <!-- t_commit -->
        <line x1="580" y1="15" x2="580" y2="500" stroke="#C53030" stroke-width="2" stroke-dasharray="6,4" />
        <text x="580" y="10" class="label-bold" text-anchor="middle" fill="#C53030">t_commit=360</text>
        
        <!-- t_exp -->
        <line x1="633" y1="15" x2="633" y2="500" class="milestone-line" />
        <text x="633" y="10" class="label-bold" text-anchor="middle">t_exp=400</text>
        
        <!-- T_chunk -->
        <line x1="766" y1="15" x2="766" y2="500" class="milestone-line" />
        <text x="766" y="10" class="label-bold" text-anchor="middle">T_chunk=500</text>
        
        <!-- t_term -->
        <line x1="846" y1="15" x2="846" y2="500" class="milestone-line" />
        <text x="846" y="10" class="label-bold" text-anchor="middle">t_term=560</text>
    </g>

    <!-- Regime 1 -->
    <g transform="translate(0, 160)">
        <text x="50" y="0" class="subtitle">Regime 1: On-Time Arrival (P50 = 180 ms)</text>
        
        <rect x="100" y="10" width="533" height="40" class="chunk-a" />
        <text x="366" y="35" class="label">Active Chunk A</text>
        
        <rect x="340" y="25" width="400" height="40" class="chunk-b" />
        <text x="540" y="50" class="label">Incoming Chunk B (Arrived 180 ms)</text>
        
        <rect x="500" y="10" width="133" height="55" class="blend-window" />
        <text x="566" y="35" class="label" text-anchor="middle">Full 100ms Blend</text>
        <text x="566" y="50" class="callout-text" text-anchor="middle">Zero velocity penalty</text>
    </g>

    <!-- Regime 2 -->
    <g transform="translate(0, 270)">
        <text x="50" y="0" class="subtitle">Regime 2: Late-but-Blendable (P99 = 340 ms)</text>
        
        <rect x="100" y="10" width="533" height="40" class="chunk-a" />
        <text x="366" y="35" class="label">Active Chunk A</text>
        
        <rect x="553" y="25" width="280" height="40" class="chunk-b" />
        <text x="693" y="50" class="label">Incoming Chunk B (Arrived 340 ms)</text>
        
        <rect x="553" y="10" width="80" height="55" class="blend-window" />
        <text x="593" y="25" class="label" text-anchor="middle">60ms Blend</text>
        <text x="593" y="40" class="callout-text" text-anchor="middle">High peak</text>
        <text x="593" y="52" class="callout-text" text-anchor="middle">accel</text>
    </g>

    <!-- Regime 3 -->
    <g transform="translate(0, 380)">
        <text x="50" y="0" class="subtitle">Regime 3: Missing Replacement (Fallback at t_commit)</text>
        
        <rect x="100" y="10" width="480" height="40" class="chunk-a" />
        <text x="340" y="35" class="label">Active Chunk A</text>
        
        <!-- Fallback zone -->
        <rect x="580" y="10" width="266" height="40" class="fallback-zone" />
        <text x="713" y="35" class="label">Precomputed Deceleration</text>
        
        <!-- Distances -->
        <line x1="500" y1="65" x2="580" y2="65" stroke="#2D3748" stroke-width="2" marker-end="url(#arrowhead)" marker-start="url(#arrowhead)"/>
        <text x="540" y="80" class="callout-text" text-anchor="middle">d_late (72mm)</text>
        
        <line x1="580" y1="65" x2="846" y2="65" stroke="#C53030" stroke-width="2" marker-end="url(#arrowhead)" marker-start="url(#arrowhead)"/>
        <text x="713" y="80" class="callout-text" text-anchor="middle">d_brake (120mm) at a_max=6.0 m/s²</text>
        
        <!-- Clearance total -->
        <line x1="500" y1="100" x2="846" y2="100" stroke="#1F407A" stroke-width="2" stroke-dasharray="4,4" />
        <text x="673" y="115" class="label-bold" text-anchor="middle">d_total = 192mm &lt; d_clear (300mm) -> Safe Halt</text>
    </g>

    <!-- Bottom Cards: Failure Modes of Naive Hold Strategies -->
    <g transform="translate(50, 560)">
        <text x="0" y="0" class="subtitle">Failure Modes of Naive Hold Strategies vs Structured Fallback</text>
        
        <rect x="0" y="20" width="210" height="120" class="callout-box" />
        <text x="10" y="40" class="card-title">Holding Torque</text>
        <text x="10" y="60" class="callout-text">Constant actuator effort.</text>
        <text x="10" y="80" class="callout-text">Allows mechanism to</text>
        <text x="10" y="100" class="callout-text">accelerate uncontrollably</text>
        <text x="10" y="120" class="card-error">Result: Unstable drift</text>

        <rect x="230" y="20" width="210" height="120" class="callout-box" />
        <text x="240" y="40" class="card-title">Holding Position</text>
        <text x="240" y="60" class="callout-text">Instantaneous zero velocity.</text>
        <text x="240" y="80" class="callout-text">Step change in command</text>
        <text x="240" y="100" class="callout-text">draws peak current.</text>
        <text x="240" y="120" class="card-error">Result: Inverter trip &amp; shock</text>

        <rect x="460" y="20" width="210" height="120" class="callout-box" />
        <text x="470" y="40" class="card-title">Continuing Velocity</text>
        <text x="470" y="60" class="callout-text">Maintains current speed.</text>
        <text x="470" y="80" class="callout-text">Expends remaining clearance</text>
        <text x="470" y="100" class="callout-text">until impact.</text>
        <text x="470" y="120" class="card-error">Result: Workspace collision</text>

        <rect x="690" y="20" width="210" height="120" fill="#F0FFF4" stroke="#38A169" stroke-width="2" rx="4" ry="4" />
        <text x="700" y="40" class="card-title" fill="#276749">Structured Fallback</text>
        <text x="700" y="60" class="callout-text">Precomputed C² deceleration.</text>
        <text x="700" y="80" class="callout-text">Stops within clearance</text>
        <text x="700" y="100" class="callout-text">using valid torque bounds.</text>
        <text x="700" y="120" class="card-title" fill="#276749">Result: Safe controlled halt</text>
    </g>

</svg>
"""
    with open('/Users/VJ/GitHub/MLSysBook-vol4-physical/books/vol4/planning/images/svg/fig11_seam_timeline_fallback.svg', 'w') as f:
        f.write(svg_content)

if __name__ == "__main__":
    generate_svg()
