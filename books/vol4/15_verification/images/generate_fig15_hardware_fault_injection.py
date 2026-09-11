import os

svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" style="background-color:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <defs>
        <!-- Gradients and Patterns -->
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#E2E8F0" stroke-width="0.5"/>
        </pattern>
        <pattern id="fineGrid" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#E2E8F0" stroke-width="0.2"/>
        </pattern>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#2D3748" />
        </marker>
        <marker id="arrowRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#A51C30" />
        </marker>
        <marker id="arrowBlue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#1F407A" />
        </marker>
    </defs>

    <!-- Background Grid -->
    <rect width="1000" height="650" fill="url(#fineGrid)" />
    <rect width="1000" height="650" fill="url(#grid)" />

    <!-- Title -->
    <text x="50" y="40" font-size="18" font-weight="bold" fill="#1F407A">Real-Time Hardware Fault Injection &amp; Deterministic Intervention Timeline</text>

    <!-- Timeline X-Axis Definition -->
    <!-- X ranges from 150 to 950, representing t=-5ms to t=50ms -->
    <!-- Map: x = 200 + (t * 14) -->
    <!-- t=0 -> 200, t=20 -> 480, t=24 -> 536, t=26 -> 564, t=42.8 -> 799.2, t=45 -> 830, t=50 -> 900 -->

    <g stroke="#2D3748" stroke-width="1.5">
        <line x1="200" y1="80" x2="200" y2="600" stroke-dasharray="4,4" opacity="0.5"/>
        <line x1="480" y1="80" x2="480" y2="600" stroke-dasharray="4,4" opacity="0.5"/>
        <line x1="536" y1="80" x2="536" y2="600" stroke-dasharray="4,4" stroke="#A51C30" opacity="0.5"/>
        <line x1="564" y1="80" x2="564" y2="600" stroke-dasharray="4,4" stroke="#1F407A" opacity="0.5"/>
        <line x1="799.2" y1="80" x2="799.2" y2="600" stroke-dasharray="4,4" opacity="0.5"/>
        <line x1="830" y1="80" x2="830" y2="600" stroke-dasharray="4,4" stroke="#A51C30" opacity="0.8"/>
    </g>

    <!-- Time Labels -->
    <g font-size="12" fill="#2D3748" font-family="monospace">
        <text x="185" y="70">t0=0.0</text>
        <text x="460" y="70">t1=20.0</text>
        <text x="520" y="55" fill="#A51C30">t2=24.0</text>
        <text x="548" y="70" fill="#1F407A">t3=26.0</text>
        <text x="780" y="70">t4=42.8</text>
        <text x="815" y="55" fill="#A51C30">t_limit=45.0</text>
    </g>

    <!-- TRACE 1: Sensor Output (ADC Register) -->
    <text x="40" y="115" font-size="12" font-weight="bold" fill="#2D3748">ADC Register</text>
    <text x="40" y="130" font-size="10" fill="#2D3748">Sensor Output</text>
    <path d="M 130 140 L 195 140 L 200 120 L 920 120" fill="none" stroke="#2D3748" stroke-width="2"/>
    <text x="140" y="135" font-size="10" fill="#2D3748">Dynamic</text>
    <text x="210" y="115" font-size="10" font-weight="bold" fill="#A51C30">Latched (Stall Injected) @ 4.2 kg/s</text>
    <circle cx="200" cy="120" r="4" fill="#A51C30"/>

    <!-- TRACE 2: Intent Lease -->
    <text x="40" y="185" font-size="12" font-weight="bold" fill="#2D3748">Intent Lease</text>
    <text x="40" y="200" font-size="10" fill="#2D3748">MPU Authority</text>
    <path d="M 130 180 L 480 180 L 480 210 L 920 210" fill="none" stroke="#1F407A" stroke-width="2"/>
    <text x="140" y="175" font-size="10" fill="#1F407A">Valid (High)</text>
    <text x="490" y="205" font-size="10" fill="#2D3748">Expired (Low)</text>
    <circle cx="480" cy="180" r="4" fill="#A51C30"/>

    <!-- TRACE 3: Diagnostic Monitor (NMI) -->
    <text x="40" y="255" font-size="12" font-weight="bold" fill="#2D3748">MCU NMI</text>
    <text x="40" y="270" font-size="10" fill="#2D3748">Stale Detection</text>
    <path d="M 130 280 L 536 280 L 536 250 L 920 250" fill="none" stroke="#A51C30" stroke-width="2"/>
    <text x="140" y="275" font-size="10" fill="#2D3748">Low</text>
    <text x="546" y="245" font-size="10" font-weight="bold" fill="#A51C30">NMI Asserted</text>
    
    <!-- Delta t_det -->
    <path d="M 480 230 L 536 230" fill="none" stroke="#2D3748" stroke-width="1.5" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
    <text x="485" y="225" font-size="10" font-style="italic" fill="#2D3748">Δt_det = 4.0ms</text>

    <!-- TRACE 4: Gate Driver Mux -->
    <text x="40" y="325" font-size="12" font-weight="bold" fill="#2D3748">HW Mux</text>
    <text x="40" y="340" font-size="10" fill="#2D3748">Gate Drivers</text>
    <path d="M 130 350 L 564 350 L 564 320 L 920 320" fill="none" stroke="#1F407A" stroke-width="2"/>
    <text x="140" y="345" font-size="10" fill="#2D3748">MPU Proposal</text>
    <text x="574" y="315" font-size="10" font-weight="bold" fill="#1F407A">Preempted (Safety Mode)</text>

    <!-- Delta t_takeover -->
    <path d="M 536 305 L 564 305" fill="none" stroke="#2D3748" stroke-width="1.5" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
    <text x="532" y="300" font-size="10" font-style="italic" fill="#2D3748">Δt_tk=2.0ms</text>

    <!-- TRACE 5: Motor Current -->
    <text x="40" y="395" font-size="12" font-weight="bold" fill="#2D3748">Motor Current</text>
    <text x="40" y="410" font-size="10" fill="#2D3748">I_phase (A)</text>
    <path d="M 130 400 L 250 395 L 350 405 L 564 400 L 564 450 L 700 450 L 799.2 410 L 920 410" fill="none" stroke="#2D3748" stroke-width="2"/>
    <text x="140" y="390" font-size="10" fill="#2D3748">Nominal Load</text>
    <text x="574" y="445" font-size="10" font-weight="bold" fill="#A51C30">Regen Braking (-12.0 A)</text>

    <!-- TRACE 6: Manifold Pressure -->
    <text x="40" y="495" font-size="12" font-weight="bold" fill="#2D3748">Manifold Pres.</text>
    <text x="40" y="510" font-size="10" fill="#2D3748">P(t) (bar)</text>
    <!-- Limits -->
    <line x1="130" y1="490" x2="920" y2="490" stroke="#A51C30" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="925" y="493" font-size="10" fill="#A51C30">16.5 bar (Containment)</text>
    
    <line x1="130" y1="470" x2="920" y2="470" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="6,4"/>
    <text x="925" y="473" font-size="10" font-weight="bold" fill="#A51C30">22.0 bar (Burst Limit)</text>

    <!-- Curve: from 10 bar (y=540) rising at 80bar/s. 1 bar = 4px (so 10 bar = 540, 16.5 = 514, 22=492... wait, let's adjust scaling.
    Let y=580 be 0 bar. Then 22 bar is y=470 -> 1 bar = 5px.
    0 bar = 580. 10 bar = 530. 16.5 bar = 497.5. 16.2 bar = 499. 22 bar = 470.
    At t=20 (480), P = 10 bar (530).
    Rises at 80 bar/s = 0.08 bar/ms.
    At t=26 (564), time elapsed since t=20 is 6ms. P increases by 6 * 0.08 = 0.48 bar.
    Wait, the surge starts at t=0 when it latches? "The test fixture injects a sensor stall... upstream manifold pressure P(t) surges at 80.0 bar/s"
    If it surges from t=0, at t=26 it's 26 * 0.08 = 2.08 bar increase.
    Let's say it starts at 14 bar (y=510). 
    At t=26 (564), it's 14 + 2.08 = 16.08 bar.
    At t=27.5 (585) it peaks at 16.2 bar (y=499).
    Then it drops to safe state at t=42.8 (799.2), say 5 bar (y=555).
    -->
    <path d="M 130 510 L 200 510 L 564 499.5 L 585 499 L 799.2 555 L 920 555" fill="none" stroke="#1F407A" stroke-width="2.5"/>
    
    <!-- Pressure Annotations -->
    <text x="350" y="525" font-size="10" fill="#1F407A">Surge @ 80.0 bar/s</text>
    <path d="M 330 505 L 500 500" fill="none" stroke="#1F407A" stroke-width="1.5" marker-end="url(#arrowBlue)"/>
    
    <circle cx="585" cy="499" r="4" fill="#A51C30"/>
    <text x="592" y="495" font-size="10" font-weight="bold" fill="#A51C30">Peak: 16.2 bar</text>

    <circle cx="799.2" cy="555" r="4" fill="#1F407A"/>
    <text x="805" y="570" font-size="10" font-weight="bold" fill="#1F407A">Safe State (Bounded)</text>
    
    <!-- Deadline Arrow -->
    <path d="M 799.2 580 L 830 580" fill="none" stroke="#2D3748" stroke-width="1.5" marker-start="url(#arrow)" marker-end="url(#arrow)"/>
    <text x="760" y="593" font-size="10" font-style="italic" fill="#2D3748">Safe margin</text>

    <!-- Legend -->
    <rect x="50" y="605" width="900" height="35" rx="4" fill="#F7FAFC" stroke="#E2E8F0"/>
    <text x="60" y="627" font-size="11" font-weight="bold" fill="#2D3748">LEGEND:</text>
    
    <line x1="120" y1="623" x2="150" y2="623" stroke="#A51C30" stroke-width="2"/>
    <text x="155" y="627" font-size="11" fill="#2D3748">Fault / Safety NMI</text>
    
    <line x1="280" y1="623" x2="310" y2="623" stroke="#1F407A" stroke-width="2"/>
    <text x="315" y="627" font-size="11" fill="#2D3748">Controller Authority / Pressure</text>
    
    <line x1="490" y1="623" x2="520" y2="623" stroke="#A51C30" stroke-width="1.5" stroke-dasharray="4,2"/>
    <text x="525" y="627" font-size="11" fill="#2D3748">Containment Limits</text>

</svg>
"""

os.makedirs("books/vol4/15_verification/images/svg", exist_ok=True)
with open("books/vol4/15_verification/images/svg/fig15_hardware_fault_injection.svg", "w") as f:
    f.write(svg_content)
