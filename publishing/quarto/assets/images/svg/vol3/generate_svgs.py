import os

def write_svg_1():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 450" width="100%" height="100%" style="font-family: 'Inter', sans-serif; background-color: #ffffff;">
    <!-- Definitions -->
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#4B5563" />
        </marker>
        <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#EF4444" />
        </marker>
        <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#10B981" />
        </marker>
        
        <style>
            .title { font-size: 16px; font-weight: 600; fill: #111827; }
            .subtitle { font-size: 14px; font-weight: 500; fill: #4B5563; }
            .box { fill: #F3F4F6; stroke: #D1D5DB; stroke-width: 2; rx: 8; ry: 8; }
            .box-red { fill: #FEE2E2; stroke: #FCA5A5; stroke-width: 2; rx: 8; ry: 8; }
            .box-green { fill: #D1FAE5; stroke: #6EE7B7; stroke-width: 2; rx: 8; ry: 8; }
            .box-blue { fill: #DBEAFE; stroke: #93C5FD; stroke-width: 2; rx: 8; ry: 8; }
            .box-purple { fill: #EDE9FE; stroke: #C4B5FD; stroke-width: 2; rx: 8; ry: 8; }
            .text-main { font-size: 13px; font-weight: 600; fill: #1F2937; text-anchor: middle; }
            .text-sub { font-size: 11px; fill: #4B5563; text-anchor: middle; }
            .path-line { fill: none; stroke: #4B5563; stroke-width: 2; }
            .path-line-red { fill: none; stroke: #EF4444; stroke-width: 2; }
            .path-line-green { fill: none; stroke: #10B981; stroke-width: 2; }
        </style>
    </defs>

    <!-- Section A -->
    <rect x="20" y="20" width="860" height="180" fill="#F9FAFB" stroke="#E5E7EB" stroke-width="1" rx="10" ry="10" />
    <text x="40" y="45" class="title">A. Naive Synchronous Runtime (Memory Wall Antipattern)</text>
    
    <g transform="translate(40, 70)">
        <rect x="0" y="0" width="160" height="100" class="box-blue" />
        <text x="80" y="30" class="text-main">Step t Infer</text>
        <text x="80" y="55" class="text-sub">(500 ms, Active)</text>
        <text x="80" y="80" class="text-sub">Alloc: +41.5 GB HBM</text>

        <path d="M 160 50 L 210 50" class="path-line" marker-end="url(#arrow)" />

        <rect x="220" y="0" width="180" height="100" class="box-red" />
        <text x="310" y="30" class="text-main">Blocking Human Gate</text>
        <text x="310" y="55" class="text-sub">(30 min - 24 hours, IDLE)</text>
        <text x="310" y="80" class="text-sub" font-weight="bold" fill="#DC2626">41.5 GB HBM Pinned</text>

        <path d="M 400 50 L 450 50" class="path-line-red" marker-end="url(#arrow-red)" />

        <rect x="460" y="0" width="200" height="100" class="box-red" />
        <text x="560" y="30" class="text-main" fill="#DC2626">Cascading Cluster Crisis</text>
        <text x="560" y="50" class="text-sub">• Zero GPU FLOP Utilization</text>
        <text x="560" y="70" class="text-sub">• Memory Starvation</text>
        <text x="560" y="90" class="text-sub">• Out-of-Memory (OOM) Aborts</text>

        <path d="M 660 50 L 710 50" class="path-line-red" marker-end="url(#arrow-red)" />

        <rect x="720" y="0" width="120" height="100" class="box" />
        <text x="780" y="45" class="text-main">Step t+1 Infer</text>
        <text x="780" y="70" class="text-sub">(Resume)</text>
    </g>

    <!-- Section B -->
    <rect x="20" y="230" width="860" height="200" fill="#F9FAFB" stroke="#E5E7EB" stroke-width="1" rx="10" ry="10" />
    <text x="40" y="255" class="title">B. Asynchronous Event-Driven Runtime (Hierarchical Memory Evacuation)</text>
    
    <g transform="translate(30, 280)">
        <rect x="0" y="0" width="140" height="120" class="box-blue" />
        <text x="70" y="35" class="text-main">Step t Infer</text>
        <text x="70" y="60" class="text-sub">(500 ms, Active)</text>
        <text x="70" y="85" class="text-sub">Alloc: +41.5 GB</text>

        <path d="M 140 60 L 170 60" class="path-line" marker-end="url(#arrow)" />

        <rect x="180" y="0" width="150" height="120" class="box-purple" />
        <text x="255" y="35" class="text-main">Trap: SYS_ESCALATE</text>
        <text x="255" y="60" class="text-sub">T_wait &gt;&gt; T_breakeven</text>
        <text x="255" y="85" class="text-sub">(332 ms Threshold)</text>

        <path d="M 330 60 L 360 60" class="path-line" marker-end="url(#arrow)" />

        <rect x="370" y="0" width="140" height="120" class="box-green" />
        <text x="440" y="35" class="text-main">DMA Stream</text>
        <text x="440" y="60" class="text-sub">to DRAM/NVMe</text>
        <text x="440" y="85" class="text-sub">Reclaim 100% HBM</text>
        <text x="440" y="105" class="text-sub" fill="#059669" font-weight="bold">(0 GB Resident)</text>

        <path d="M 510 60 L 540 60" class="path-line-green" marker-end="url(#arrow-green)" />

        <rect x="550" y="0" width="150" height="120" class="box" />
        <text x="625" y="35" class="text-main">Human Deliberation</text>
        <text x="625" y="60" class="text-sub">(Promise Pending)</text>
        <text x="625" y="85" class="text-sub">Cluster Serves</text>
        <text x="625" y="105" class="text-sub">Other Trajectories</text>

        <path d="M 700 60 L 730 60" class="path-line-green" marker-end="url(#arrow-green)" />

        <rect x="740" y="0" width="120" height="120" class="box-blue" />
        <text x="800" y="35" class="text-main">Hydrate &amp; Resume</text>
        <text x="800" y="60" class="text-sub">DMA Hydrate</text>
        <text x="800" y="85" class="text-sub">KV Cache</text>
        <text x="800" y="105" class="text-sub">(Zero Recompute)</text>
    </g>
</svg>"""
    with open('/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/assets/images/svg/vol3/09_latency_disparity.svg', 'w') as f:
        f.write(svg)

def write_svg_2():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 600" width="100%" height="100%" style="font-family: 'Inter', sans-serif; background-color: #ffffff;">
    <defs>
        <style>
            .title { font-size: 18px; font-weight: 600; fill: #111827; }
            .category-title { font-size: 14px; font-weight: 600; fill: #1F2937; }
            .box { fill: #F9FAFB; stroke: #D1D5DB; stroke-width: 2; rx: 8; ry: 8; }
            .item-box { fill: #FFFFFF; stroke: #E5E7EB; stroke-width: 1; rx: 4; ry: 4; }
            .item-text { font-size: 12px; fill: #374151; }
            .policy-box { fill: #F3F4F6; stroke: #9CA3AF; stroke-width: 1; stroke-dasharray: 4; rx: 4; ry: 4; }
            .policy-text { font-size: 11px; font-weight: 500; fill: #1F2937; }
        </style>
    </defs>

    <text x="450" y="30" class="title" text-anchor="middle">Agent Runtime Interrupt &amp; Trap Vector Space</text>
    <rect x="20" y="45" width="860" height="535" fill="none" stroke="#E5E7EB" stroke-width="2" rx="12" ry="12" />

    <!-- Top Left: Asynchronous External IRQs -->
    <g transform="translate(40, 65)">
        <rect x="0" y="0" width="400" height="240" class="box" fill="#F0FDF4" stroke="#86EFAC" />
        <text x="20" y="25" class="category-title">Asynchronous External Interrupts (Hardware IRQs)</text>
        
        <rect x="20" y="45" width="360" height="30" class="item-box" />
        <text x="35" y="65" class="item-text">Human Supervisory Approval / Rejection Webhook</text>

        <rect x="20" y="85" width="360" height="30" class="item-box" />
        <text x="35" y="105" class="item-text">Out-of-Band Signal (Slack, Dashboard, REST API)</text>

        <rect x="20" y="125" width="360" height="30" class="item-box" />
        <text x="35" y="145" class="item-text">Asynchronous Long-Running Tool Return Event</text>

        <rect x="20" y="175" width="360" height="45" class="policy-box" fill="#DCFCE7" />
        <text x="35" y="195" class="policy-text">Handling: Maskable, Drained at Token Step Boundary</text>
        <text x="35" y="210" class="policy-text">Action: Transition ACB to Wakeup / Rehydration</text>
    </g>

    <!-- Top Right: Synchronous Traps -->
    <g transform="translate(460, 65)">
        <rect x="0" y="0" width="400" height="240" class="box" fill="#EFF6FF" stroke="#93C5FD" />
        <text x="20" y="25" class="category-title">Synchronous Traps (AI System Calls)</text>
        
        <rect x="20" y="45" width="360" height="30" class="item-box" />
        <text x="35" y="65" class="item-text">SYS_ESCALATE (Reversibility Boundary Crossing)</text>

        <rect x="20" y="85" width="360" height="30" class="item-box" />
        <text x="35" y="105" class="item-text">SYS_YIELD (Asynchronous Tool Wait Delegation)</text>

        <rect x="20" y="125" width="360" height="30" class="item-box" />
        <text x="35" y="145" class="item-text">SYS_FORK (Subagent Tree Spawn &amp; Speculation)</text>

        <rect x="20" y="175" width="360" height="45" class="policy-box" fill="#DBEAFE" />
        <text x="35" y="195" class="policy-text">Handling: Programmed, Synchronous Execution Yield</text>
        <text x="35" y="210" class="policy-text">Action: Immediate Context Freeze &amp; Memory Dehydration</text>
    </g>

    <!-- Bottom Left: Synchronous Faults -->
    <g transform="translate(40, 320)">
        <rect x="0" y="0" width="400" height="240" class="box" fill="#FEF2F2" stroke="#FCA5A5" />
        <text x="20" y="25" class="category-title">Synchronous Faults &amp; Exceptions</text>
        
        <rect x="20" y="45" width="360" height="30" class="item-box" />
        <text x="35" y="65" class="item-text">Tool Return Schema Validation Failure (Malformed JSON)</text>

        <rect x="20" y="85" width="360" height="30" class="item-box" />
        <text x="35" y="105" class="item-text">Safety Invariant Violation / Jailbreak Detection</text>

        <rect x="20" y="125" width="360" height="30" class="item-box" />
        <text x="35" y="145" class="item-text">Cyclic Trajectory Livelock / Repetition Anomaly</text>

        <rect x="20" y="175" width="360" height="45" class="policy-box" fill="#FEE2E2" />
        <text x="35" y="195" class="policy-text">Handling: Synchronous, Software-Detected Exception</text>
        <text x="35" y="210" class="policy-text">Action: Deterministic Step Retry or Sagas Rollback</text>
    </g>

    <!-- Bottom Right: NMIs -->
    <g transform="translate(460, 320)">
        <rect x="0" y="0" width="400" height="240" class="box" fill="#FEF3C7" stroke="#FCD34D" />
        <text x="20" y="25" class="category-title">Non-Maskable Interrupts (Hardware NMIs)</text>
        
        <rect x="20" y="45" width="360" height="30" class="item-box" />
        <text x="35" y="65" class="item-text">Authoritative Administrator Kill Switch</text>

        <rect x="20" y="85" width="360" height="30" class="item-box" />
        <text x="35" y="105" class="item-text">Hard Token Quota / Budget Wall Breach</text>

        <rect x="20" y="125" width="360" height="30" class="item-box" />
        <text x="35" y="145" class="item-text">Critical Data Exfiltration / Taint Escape Alarm</text>

        <rect x="20" y="175" width="360" height="45" class="policy-box" fill="#FEF3C7" />
        <text x="35" y="195" class="policy-text">Handling: Unmaskable, Immediate Force Halting</text>
        <text x="35" y="210" class="policy-text">Action: Destroy MicroVM Sandbox &amp; Burn Capability Leases</text>
    </g>
</svg>"""
    with open('/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/assets/images/svg/vol3/09_interrupt_taxonomy.svg', 'w') as f:
        f.write(svg)

os.makedirs('/Users/VJ/GitHub/MLSysBook-gem-vol3/publishing/quarto/assets/images/svg/vol3/', exist_ok=True)
write_svg_1()
write_svg_2()
print("SVGs generated successfully.")
