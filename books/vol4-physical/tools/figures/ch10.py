"""
book/tools/figures/ch10.py
Figures for Chapter 10: Heterogeneous Compute Placement.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch10_placement():
    W = 900
    H = 440
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">4-TIER HETEROGENEOUS COMPUTE PLACEMENT TOPOLOGY</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Partitioning Latency-Critical Reflexes from High-Throughput Deliberation Across Power Envelopes</text>')

    tiers = [
        ("TIER 1 · CLOUD / SERVER", "Remote Cluster (> 1000 W)", "0.1–1 Hz · Latency: 100–500 ms",
         ["Foundation World Models (100B+)", "Fleet Map Synthesis & Semantic Search", "Non-Critical High-Level Mission Goals"],
         "Wireless 5G / Wi-Fi (Untrusted)", PURPLE),
        ("TIER 2 · EDGE WORKSTATION", "On-Premises Base (100–300 W)", "1–5 Hz · Latency: 20–80 ms",
         ["Spatial VLM Reasoning (PaliGemma)", "Local Workspace Digital Twin", "Dynamic Multi-Agent Traffic Deconfliction"],
         "Private Subnet / TSN Link", BLUE),
        ("TIER 3 · ONBOARD MPU / NPU", "Mobile Linux Host (15–60 W)", "20–60 Hz · Latency: 15–50 ms",
         ["Vision Tokenizers (DINOv2 / MobileNet)", "Action Chunking (ACT / Diffusion, H=16)", "Local SE(3) Frame Tree Updates"],
         "PCIe / Shared TCM SRAM", BRONZE),
        ("TIER 4 · REAL-TIME MCU", "Dedicated Enforcer (< 2 W)", "1000 Hz · Latency: < 1 ms (Jitter < 5 µs)",
         ["Active-Set CBF QP Filter (h(x) ≥ 0)", "Dynamic Stopping Envelope Watchdog", "Category 0/1 Hardware Fallback FSM"],
         "Direct Memory-Mapped PWM / Gate Drivers", PETROL)
    ]

    card_w = 200
    gap = 14
    start_x = (W - (4 * card_w + 3 * gap)) / 2

    for i, (tag, title, latency, items, link, col) in enumerate(tiers):
        x = start_x + i * (card_w + gap)
        y = 66
        h = 340

        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{h}" rx="8" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="24" rx="8" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+16}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+42}" font-size="11" font-weight="700" fill="{INK}" text-anchor="middle">{title}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+58}" font-size="8.5" font-weight="600" fill="{col}" text-anchor="middle">{latency}</text>')

        cy = y + 80
        for it in items:
            svg.append(f'<text x="{x+10}" y="{cy}" font-size="8.5" fill="{SLATE}">• {it}</text>')
            cy += 20

        svg.append(f'<rect x="{x+8}" y="{y+h-40}" width="{card_w-16}" height="28" rx="4" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="0.8"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+h-28}" font-size="7.5" font-weight="700" fill="{MUTED}" text-anchor="middle">DOWNSTREAM LINK</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+h-16}" font-size="8" font-weight="600" fill="{INK}" text-anchor="middle">{link}</text>')

        if i < 3:
            ax1 = x + card_w + 1
            ax2 = ax1 + gap - 2
            ay = y + h/2
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{col}" stroke-width="1.5" marker-end="url(#arr-blue)"/>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/10-placement/figures/fig09_heterogeneous_placement.svg", "\n".join(svg))

def gen_ch10_uma_bus():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">UMA SHARED MEMORY CONTENTION &amp; AXI QOS ARBITRATION</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Preventing Tail-Latency Spikes When Vision Ingestion Saturates the Memory Crossbar</text>')

    mx = 40
    mw = 220
    masters = [
        ("NPU / GPU (Vision Transformer)", "Batch DMA Ingestion · 4.2 GB/s", CORAL, "Low Priority (QoS=2)"),
        ("CPU (Linux World Model)", "SE(3) Frame Tree Updates · 800 MB/s", BLUE, "Medium Priority (QoS=8)"),
        ("MCU Safety Enforcer", "1 kHz QP Barrier Solver · 12 MB/s", TEAL, "Highest Priority (QoS=15)")
    ]
    for idx, (title, sub, col, qos) in enumerate(masters):
        by = 80 + idx * 95
        svg.append(f'<rect x="{mx}" y="{by}" width="{mw}" height="70" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<text x="{mx+12}" y="{by+20}" font-size="10.5" font-weight="700" fill="{col}">{title}</text>')
        svg.append(f'<text x="{mx+12}" y="{by+38}" font-size="8.5" fill="{SLATE}">{sub}</text>')
        svg.append(f'<rect x="{mx+12}" y="{by+46}" width="140" height="16" rx="3" fill="{col}" fill-opacity="0.1"/>')
        svg.append(f'<text x="{mx+18}" y="{by+58}" font-size="8" font-weight="700" fill="{col}">{qos}</text>')
        svg.append(f'<line x1="{mx+mw}" y1="{by+35}" x2="340" y2="210" stroke="{col}" stroke-width="1.4" marker-end="url(#arr-blue)"/>')

    cx = 340
    cy = 80
    cw = 200
    ch = 260
    svg.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="8" fill="{NAVY}" fill-opacity="0.06" stroke="{NAVY}" stroke-width="1.4"/>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+30}" font-size="11" font-weight="700" fill="{NAVY}" text-anchor="middle">AXI-4 INTERCONNECT</text>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+48}" font-size="9" fill="{MUTED}" text-anchor="middle">Crossbar Switch &amp; Arbiter</text>')

    svg.append(f'<rect x="{cx+15}" y="{cy+70}" width="{cw-30}" height="70" rx="4" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+90}" font-size="9" font-weight="700" fill="{CORAL}" text-anchor="middle">UNREGULATED BUS</text>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+108}" font-size="8" fill="{SLATE}" text-anchor="middle">NPU saturates DRAM channels</text>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+122}" font-size="8" font-weight="600" fill="{CORAL}" text-anchor="middle">MCU Enforcer Latency: 14.8 ms ✕</text>')

    svg.append(f'<rect x="{cx+15}" y="{cy+155}" width="{cw-30}" height="70" rx="4" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1"/>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+175}" font-size="9" font-weight="700" fill="{TEAL}" text-anchor="middle">AXI QOS ARBITRATION</text>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+193}" font-size="8" fill="{SLATE}" text-anchor="middle">MCU packets preempt bulk DMA</text>')
    svg.append(f'<text x="{cx+cw/2}" y="{cy+207}" font-size="8" font-weight="600" fill="{TEAL}" text-anchor="middle">MCU Enforcer Latency: 120 µs ✓</text>')

    rx = 620
    ry = 130
    rw = 220
    rh = 160
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="8" fill="{BG_WHITE}" stroke="{BORDER_DARK}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+26}" font-size="11" font-weight="700" fill="{INK}" text-anchor="middle">SHARED LPDDR5 MEMORY</text>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+42}" font-size="8.5" fill="{MUTED}" text-anchor="middle">Unified Memory Architecture (UMA)</text>')

    svg.append(f'<rect x="{rx+15}" y="{ry+55}" width="{rw-30}" height="32" rx="4" fill="{CORAL}" fill-opacity="0.1"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+75}" font-size="8.5" font-weight="600" fill="{CORAL}" text-anchor="middle">Bulk Framebuffers (RGB + Depth)</text>')

    svg.append(f'<rect x="{rx+15}" y="{ry+95}" width="{rw-30}" height="32" rx="4" fill="{TEAL}" fill-opacity="0.1"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+115}" font-size="8.5" font-weight="600" fill="{TEAL}" text-anchor="middle">Dedicated Locked SRAM Bank (TCM)</text>')

    svg.append(f'<line x1="{cx+cw}" y1="210" x2="{rx}" y2="210" stroke="{NAVY}" stroke-width="1.6" marker-end="url(#arr-navy)"/>')

    svg.append(f'<rect x="40" y="360" width="800" height="42" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="378" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">THE UMA MEMORY INVARIANT</text>')
    svg.append(f'<text x="{W/2}" y="392" font-size="8.5" fill="{SLATE}" text-anchor="middle">Real-time safety loops must never share un-arbitrated memory buses with high-bandwidth sensory ingestion pipelines without hardware QoS rate limiting.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/10-placement/figures/fig09_uma_bus_contention.svg", "\n".join(svg))

def gen_ch10_thermal_derating():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THERMODYNAMIC JUNCTION HEATING &amp; DVFS DOWNCLOCKING RIPPLE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Silicon Thermal RC Dynamics: T_j(t) = T_ambient + P · θ_JA · (1 - exp(-t / τ_th))</text>')

    # Timeline of Temperature Rise vs Clock Speed
    ax_x = 80
    ax_y = 260
    pw = 720
    ph = 180

    svg.append(f'<line x1="{ax_x}" y1="{ax_y}" x2="{ax_x+pw}" y2="{ax_y}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{ax_x+pw/2}" y="{ax_y+40}" font-size="10.5" font-weight="700" fill="{SLATE}" text-anchor="middle">Sustained High-Load Execution Time t (seconds) →</text>')

    # Temperature Curve (Red: rises from ambient to 85°C trip)
    temp_d = f"M {ax_x} {ax_y-25} Q {ax_x+250} {ax_y-80} {ax_x+pw-40} {ax_y-155}"
    svg.append(f'<path d="{temp_d}" fill="none" stroke="{CORAL}" stroke-width="2.5"/>')
    svg.append(f'<text x="{ax_x+pw-40}" y="{ax_y-163}" font-size="9" font-weight="700" fill="{CORAL}" text-anchor="end">Junction Temp T_j (85°C Max)</text>')

    # DVFS Clock Frequency Drop (Blue stepped curve: drops from 2.4 GHz to 800 MHz)
    clk_d = f"M {ax_x} {ax_y-145} L {ax_x+280} {ax_y-145} L {ax_x+280} {ax_y-45} L {ax_x+pw-40} {ax_y-45}"
    svg.append(f'<path d="{clk_d}" fill="none" stroke="{BLUE}" stroke-width="2.2" stroke-dasharray="6,3"/>')
    svg.append(f'<text x="{ax_x+pw-40}" y="{ax_y-53}" font-size="9" font-weight="700" fill="{BLUE}" text-anchor="end">DVFS Clock (2.4 GHz → 800 MHz)</text>')

    # Throttle Marker
    th_x = ax_x + 280
    svg.append(f'<line x1="{th_x}" y1="80" x2="{th_x}" y2="{ax_y}" stroke="{CORAL}" stroke-width="1.5" stroke-dasharray="3,3"/>')
    svg.append(f'<rect x="{th_x-60}" y="70" width="120" height="22" rx="4" fill="{CORAL}"/>')
    svg.append(f'<text x="{th_x}" y="85" font-size="8.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">THERMAL TRIP (85°C)</text>')

    # Consequence Box
    svg.append(f'<rect x="40" y="320" width="{W-80}" height="80" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="60" y="342" font-size="10" font-weight="700" fill="{NAVY}">THE THERMAL DERATING CASCADING FAILURE</text>')
    svg.append(f'<text x="60" y="360" font-size="8.5" fill="{SLATE}">1. NPU thermal throttle cuts frequency by 3× ⇒ Inference latency triples from 20 ms to 60 ms.</text>')
    svg.append(f'<text x="60" y="376" font-size="8.5" fill="{SLATE}">2. Stopping distance d_reaction triples from 10 cm to 30 cm ⇒ Robot breaches clearance buffer unless enforcer backs off speed.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/10-placement/figures/fig09_thermal_derating_ripple.svg", "\n".join(svg))

def run_all():
    gen_ch10_placement()
    gen_ch10_uma_bus()
    gen_ch10_thermal_derating()

if __name__ == "__main__":
    run_all()
