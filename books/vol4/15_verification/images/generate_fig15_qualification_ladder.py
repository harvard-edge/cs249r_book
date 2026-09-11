import os

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" style="background-color:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#E2E8F0" stroke-width="0.5"/>
        </pattern>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#2D3748" />
        </marker>
        
        <linearGradient id="gradSIL" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#E2E8F0" />
            <stop offset="100%" stop-color="#FFFFFF" />
        </linearGradient>
    </defs>

    <!-- Background Grid -->
    <rect width="1000" height="650" fill="url(#grid)" />

    <!-- Title -->
    <text x="50" y="40" font-size="20" font-weight="bold" fill="#1F407A">The Four-Stage Qualification Ladder &amp; Causal-Loop Matrix</text>
    <text x="50" y="60" font-size="12" fill="#2D3748">Trading throughput for physical fidelity across cyber-physical verification environments.</text>

    <!-- Axis Lines -->
    <!-- X-axis: Hardware Fidelity -->
    <path d="M 250 560 L 900 560" fill="none" stroke="#2D3748" stroke-width="2" marker-end="url(#arrow)"/>
    <text x="500" y="590" font-size="14" font-weight="bold" fill="#2D3748">Hardware &amp; Physical Fidelity (%)</text>
    <text x="250" y="580" font-size="12" fill="#2D3748">0%</text>
    <text x="900" y="580" font-size="12" fill="#2D3748">100%</text>
    
    <!-- Y-axis: Throughput -->
    <path d="M 230 540 L 230 100" fill="none" stroke="#2D3748" stroke-width="2" marker-end="url(#arrow)"/>
    <!-- We need to rotate the Y-axis label properly. Wait, SVG transform rotate takes x,y -->
    <text x="-380" y="180" font-size="14" font-weight="bold" fill="#2D3748" transform="rotate(-90)">Throughput (runs / day)</text>
    <text x="180" y="540" font-size="12" fill="#2D3748">10^0</text>
    <text x="180" y="420" font-size="12" fill="#2D3748">10^2</text>
    <text x="180" y="300" font-size="12" fill="#2D3748">10^4</text>
    <text x="180" y="180" font-size="12" fill="#2D3748">10^6</text>

    <!-- Blocks representing each stage -->
    
    <!-- SIL: High Throughput, Low Fidelity -->
    <g transform="translate(250, 140)">
        <rect width="180" height="70" fill="#E2E8F0" stroke="#2D3748" stroke-width="2"/>
        <text x="10" y="20" font-size="14" font-weight="bold" fill="#1F407A">1. SIL</text>
        <text x="10" y="38" font-size="11" fill="#2D3748">Software-in-the-Loop</text>
        <text x="10" y="55" font-size="10" fill="#2D3748">ODE Dynamics, Param Sweeps</text>
    </g>

    <!-- PIL: Med-High Throughput, Med-Low Fidelity -->
    <g transform="translate(400, 260)">
        <rect width="200" height="70" fill="#CBD5E0" stroke="#2D3748" stroke-width="2"/>
        <text x="10" y="20" font-size="14" font-weight="bold" fill="#1F407A">2. PIL</text>
        <text x="10" y="38" font-size="11" fill="#2D3748">Processor-in-the-Loop</text>
        <text x="10" y="55" font-size="10" fill="#2D3748">Target SoC, Interrupt Latency</text>
    </g>

    <!-- HIL: Med-Low Throughput, Med-High Fidelity -->
    <g transform="translate(560, 380)">
        <rect width="220" height="70" fill="#A0AEC0" stroke="#2D3748" stroke-width="2"/>
        <text x="10" y="20" font-size="14" font-weight="bold" fill="#1F407A">3. HIL</text>
        <text x="10" y="38" font-size="11" fill="#2D3748">Hardware-in-the-Loop</text>
        <text x="10" y="55" font-size="10" fill="#2D3748">Power Rail Droop, Motor Back-EMF</text>
    </g>

    <!-- In-Situ: Low Throughput, High Fidelity -->
    <g transform="translate(730, 490)">
        <rect width="220" height="70" fill="#A51C30" stroke="#2D3748" stroke-width="2"/>
        <text x="10" y="20" font-size="14" font-weight="bold" fill="#ffffff">4. In-Situ</text>
        <text x="10" y="38" font-size="11" fill="#ffffff">Physical Fault Injection</text>
        <text x="10" y="55" font-size="10" fill="#ffffff">Destructive Water Hammer, Jams</text>
    </g>
    
    <!-- Arrows connecting them -->
    <path d="M 340 210 L 410 260" fill="none" stroke="#2D3748" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#arrow)"/>
    <path d="M 500 330 L 570 380" fill="none" stroke="#2D3748" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#arrow)"/>
    <path d="M 670 450 L 740 490" fill="none" stroke="#2D3748" stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#arrow)"/>
    
    <!-- Causal-Loop Blind Spot Callouts -->
    <rect x="500" y="140" width="350" height="50" fill="#ffffff" stroke="#1F407A" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="510" y="160" font-size="11" font-weight="bold" fill="#1F407A">SIL Blind Spots:</text>
    <text x="510" y="175" font-size="10" fill="#2D3748">Abstracts clock drift, target silicon timing, &amp; DMA lockups.</text>

    <rect x="650" y="260" width="300" height="50" fill="#ffffff" stroke="#1F407A" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="660" y="280" font-size="11" font-weight="bold" fill="#1F407A">PIL Blind Spots:</text>
    <text x="660" y="295" font-size="10" fill="#2D3748">Ignores real bus contention &amp; electrical noise.</text>
    
    <rect x="250" y="380" width="280" height="50" fill="#ffffff" stroke="#1F407A" stroke-width="1" stroke-dasharray="4,4"/>
    <text x="260" y="400" font-size="11" font-weight="bold" fill="#1F407A">HIL Blind Spots:</text>
    <text x="260" y="415" font-size="10" fill="#2D3748">Synthesized plant cannot wear or break structurally.</text>

    <!-- Legend -->
    <rect x="250" y="605" width="700" height="30" fill="#F7FAFC" stroke="#E2E8F0"/>
    <text x="260" y="625" font-size="11" font-style="italic" fill="#2D3748">Note: Each stage establishes evidence only within its own causal loop, revealing entirely different fault classes.</text>

</svg>
"""

os.makedirs("books/vol4/15_verification/images/svg", exist_ok=True)
with open("books/vol4/15_verification/images/svg/fig15_qualification_ladder.svg", "w") as f:
    f.write(svg_content)
