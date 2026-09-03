"""
book/tools/figures/ch15.py
Figures for Chapter 15: Whole-System Verification.
Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import math
from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch15_qualification_ladder():
    """
    Figure 15.1: The 4-Stage Qualification Ladder & Causal-Loop Verification Matrix.
    Shows the continuous trade-off between throughput (runs/day) and physical fidelity,
    along with the causal loop breakdown for each testing rung.
    """
    W = 960
    H = 540
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    
    # Title & Subtitle
    svg.append(f'<text x="{W/2}" y="28" class="title">THE 4-STAGE QUALIFICATION LADDER &amp; CAUSAL-LOOP VERIFICATION MATRIX</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Systematic Progression Across the Throughput–Fidelity Continuum: Software Simulation to In-Situ Physical Fault Injection</text>')

    # Dual Continuum Trade-off Banner
    bx = 30
    by = 58
    bw = W - 60
    bh = 38
    svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    
    # Throughput arrow (Left to Right: High to Low)
    svg.append(f'<text x="{bx+15}" y="{by+16}" font-size="9" font-weight="700" fill="{NAVY}">THROUGHPUT / SAMPLE VOLUME:</text>')
    svg.append(f'<text x="{bx+220}" y="{by+16}" font-size="8.5" font-weight="600" fill="{TEAL}">10⁶ runs/day (Massive Parallel)</text>')
    svg.append(f'<text x="{bx+430}" y="{by+16}" font-size="8.5" font-weight="600" fill="{BLUE}">10⁴ runs/day</text>')
    svg.append(f'<text x="{bx+590}" y="{by+16}" font-size="8.5" font-weight="600" fill="{BRONZE}">10² runs/day</text>')
    svg.append(f'<text x="{bx+740}" y="{by+16}" font-size="8.5" font-weight="700" fill="{CORAL}">10⁰–10¹ runs/day (Constrained)</text>')
    svg.append(f'<line x1="{bx+210}" y1="{by+20}" x2="{bx+bw-20}" y2="{by+20}" stroke="{NAVY}" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arr-coral)"/>')

    # Physical Fidelity arrow (Left to Right: Low to High)
    svg.append(f'<text x="{bx+15}" y="{by+32}" font-size="9" font-weight="700" fill="{CRIMSON}">HARDWARE &amp; PHYSICAL FIDELITY:</text>')
    svg.append(f'<text x="{bx+220}" y="{by+32}" font-size="8.5" font-weight="600" fill="{MUTED}">0% (Idealized Math / ODEs)</text>')
    svg.append(f'<text x="{bx+430}" y="{by+32}" font-size="8.5" font-weight="600" fill="{PURPLE}">40% (Target Silicon)</text>')
    svg.append(f'<text x="{bx+590}" y="{by+32}" font-size="8.5" font-weight="600" fill="{PETROL}">85% (Real Dyno Load)</text>')
    svg.append(f'<text x="{bx+740}" y="{by+32}" font-size="8.5" font-weight="700" fill="{CRIMSON}">100% (Full Physical Reality)</text>')

    # 4 Rung Cards
    card_w = 214
    card_gap = 14
    start_x = 30
    card_y = 106
    card_h = 414

    rungs = [
        {
            "stage": "STAGE 1",
            "title": "Software-in-the-Loop",
            "sub": "Multi-Physics Simulation (SIL)",
            "col": BLUE,
            "badge": "Statistical Parameter Sweeps",
            "substrate": "Cloud GPUs / MuJoCo / Isaac Sim",
            "causal": [
                ("Algorithm / Policy:", "Real Python / C++ binary", TEAL),
                ("Target Silicon / Clocks:", "Simulated / Host CPU time", MUTED),
                ("Bus Transport & Latency:", "Zero-latency idealized copy", MUTED),
                ("Actuation & Dynamics:", "Rigid-body ODE approximations", MUTED),
                ("Environmental Physics:", "Synthetic contact models", MUTED)
            ],
            "proves": "Broad combinatorial state-space coverage, policy loss convergence, nominal trajectory tracking.",
            "blind": "Silicon bus contention, clock skew, thermal derating, real fluid cavitation, unmodeled friction."
        },
        {
            "stage": "STAGE 2",
            "title": "Processor-in-the-Loop",
            "sub": "Target SoC / MCU Emulation (PIL)",
            "col": PURPLE,
            "badge": "Silicon Timing & Interrupts",
            "substrate": "Target ARM Cortex-A/M + FPGA Plant",
            "causal": [
                ("Algorithm / Policy:", "Real compiled firmware binary", TEAL),
                ("Target Silicon / Clocks:", "Real SoC oscillator & registers", TEAL),
                ("Bus Transport & Latency:", "Real AXI / SPI / CAN buffers", TEAL),
                ("Actuation & Dynamics:", "Virtual PWM capture (FPGA)", MUTED),
                ("Environmental Physics:", "Real-time 10 kHz math model", MUTED)
            ],
            "proves": "Zero dynamic malloc compliance, interrupt latency tails, RTOS context-switch deadlines, watchdog resets.",
            "blind": "Motor back-EMF, inverter power-rail droop, cable harness EMI, mechanical backlash."
        },
        {
            "stage": "STAGE 3",
            "title": "Hardware-in-the-Loop",
            "sub": "Actuator Dynamo Stand (HIL)",
            "col": PETROL,
            "badge": "Electrical & Thermal Load",
            "substrate": "Real Inverters, Motor & 4Q Dyno",
            "causal": [
                ("Algorithm / Policy:", "Real target firmware & weights", TEAL),
                ("Target Silicon / Clocks:", "Real multi-rate heterogeneous SoC", TEAL),
                ("Bus Transport & Latency:", "Real physical transceivers & harness", TEAL),
                ("Actuation & Dynamics:", "Real BLDC motor & dynamic brake", TEAL),
                ("Environmental Physics:", "Active dyno counter-torque (1 kHz)", AMBER)
            ],
            "proves": "Inverter thermal dissipation at 10 A, motor back-EMF, regenerative braking torque, power-rail droop.",
            "blind": "Hydrodynamic water hammer, fluid contamination, pipe vibration, acoustic standing waves."
        },
        {
            "stage": "STAGE 4",
            "title": "Physical Fault Rig",
            "sub": "In-Situ Fluid / Vehicle Stand",
            "col": CRIMSON,
            "badge": "Destructive Physical Limits",
            "substrate": "Full Assembled Machine in Blast Cell",
            "causal": [
                ("Algorithm / Policy:", "Production flashing on chassis", TEAL),
                ("Target Silicon / Clocks:", "Full vehicle harness & power rail", TEAL),
                ("Bus Transport & Latency:", "Full harness with ambient EMI", TEAL),
                ("Actuation & Dynamics:", "Real motorized valves & mechanics", TEAL),
                ("Environmental Physics:", "Real high-pressure fluid & wear", TEAL)
            ],
            "proves": "Hydraulic water hammer, elastomeric seal extrusion, mechanical jamming, pipe burst containment.",
            "blind": "Rare combinatorial parameter sweeps (destructive cost limits sample count to N ≤ 50)."
        }
    ]

    for i, r in enumerate(rungs):
        rx = start_x + i * (card_w + card_gap)
        
        # Outer Card
        svg.append(f'<rect x="{rx}" y="{card_y}" width="{card_w}" height="{card_h}" rx="7" fill="{BG_WHITE}" stroke="{r["col"]}" stroke-width="1.4" filter="url(#shadow)"/>')
        
        # Header banner
        svg.append(f'<rect x="{rx}" y="{card_y}" width="{card_w}" height="28" rx="7" fill="{r["col"]}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{rx+card_w/2}" y="{card_y+18}" font-size="10" font-weight="700" fill="{r["col"]}" text-anchor="middle">{r["stage"]} · {r["title"]}</text>')
        
        # Subtitle & Badge
        svg.append(f'<text x="{rx+card_w/2}" y="{card_y+42}" font-size="8.5" font-weight="600" fill="{INK}" text-anchor="middle">{r["sub"]}</text>')
        
        svg.append(f'<rect x="{rx+10}" y="{card_y+50}" width="{card_w-20}" height="18" rx="3" fill="{r["col"]}" fill-opacity="0.08" stroke="{r["col"]}" stroke-width="0.8"/>')
        svg.append(f'<text x="{rx+card_w/2}" y="{card_y+62}" font-size="8" font-weight="700" fill="{r["col"]}" text-anchor="middle">★ {r["badge"]}</text>')
        
        # Substrate
        svg.append(f'<text x="{rx+10}" y="{card_y+80}" font-size="8" font-weight="700" fill="{SLATE}">TEST SUBSTRATE:</text>')
        svg.append(f'<text x="{rx+10}" y="{card_y+92}" font-size="8" fill="{INK}">{r["substrate"]}</text>')
        
        # Causal loop box
        cy = card_y + 102
        ch = 142
        svg.append(f'<rect x="{rx+8}" y="{cy}" width="{card_w-16}" height="{ch}" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="0.9"/>')
        svg.append(f'<text x="{rx+14}" y="{cy+14}" font-size="8" font-weight="700" fill="{NAVY}">CAUSAL-LOOP FIDELITY:</text>')
        
        for idx, (elem_label, elem_stat, tag_col) in enumerate(r["causal"]):
            ey = cy + 28 + idx * 22
            svg.append(f'<circle cx="{rx+16}" cy="{ey-3}" r="3" fill="{tag_col}"/>')
            svg.append(f'<text x="{rx+24}" y="{ey-4}" font-size="7.5" font-weight="600" fill="{INK}">{elem_label}</text>')
            svg.append(f'<text x="{rx+24}" y="{ey+6}" font-size="7" fill="{SLATE}">{elem_stat}</text>')
            
        # What it establishes
        py = card_y + 252
        svg.append(f'<rect x="{rx+8}" y="{py}" width="{card_w-16}" height="76" rx="5" fill="{TEAL}" fill-opacity="0.05" stroke="{TEAL}" stroke-width="0.9"/>')
        svg.append(f'<text x="{rx+14}" y="{py+14}" font-size="8" font-weight="700" fill="{TEAL}">✓ WHAT IT ESTABLISHES:</text>')
        words_p = r["proves"].split()
        l1 = " ".join(words_p[:4])
        l2 = " ".join(words_p[4:8])
        l3 = " ".join(words_p[8:12])
        l4 = " ".join(words_p[12:])
        svg.append(f'<text x="{rx+14}" y="{py+28}" font-size="7.5" fill="{SLATE}">{l1}</text>')
        svg.append(f'<text x="{rx+14}" y="{py+40}" font-size="7.5" fill="{SLATE}">{l2}</text>')
        svg.append(f'<text x="{rx+14}" y="{py+52}" font-size="7.5" fill="{SLATE}">{l3}</text>')
        svg.append(f'<text x="{rx+14}" y="{py+64}" font-size="7.5" fill="{SLATE}">{l4}</text>')
        
        # Evidentiary blind spot / limitation
        by = card_y + 334
        svg.append(f'<rect x="{rx+8}" y="{by}" width="{card_w-16}" height="72" rx="5" fill="{CORAL}" fill-opacity="0.05" stroke="{CORAL}" stroke-width="0.9"/>')
        svg.append(f'<text x="{rx+14}" y="{by+14}" font-size="8" font-weight="700" fill="{CORAL}">✕ EVIDENTIARY BLIND SPOT:</text>')
        words_b = r["blind"].split()
        b1 = " ".join(words_b[:4])
        b2 = " ".join(words_b[4:8])
        b3 = " ".join(words_b[8:12])
        b4 = " ".join(words_b[12:])
        svg.append(f'<text x="{rx+14}" y="{by+28}" font-size="7.5" fill="{SLATE}">{b1}</text>')
        svg.append(f'<text x="{rx+14}" y="{by+40}" font-size="7.5" fill="{SLATE}">{b2}</text>')
        svg.append(f'<text x="{rx+14}" y="{by+52}" font-size="7.5" fill="{SLATE}">{b3}</text>')
        svg.append(f'<text x="{rx+14}" y="{by+64}" font-size="7.5" fill="{SLATE}">{b4}</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/15-verification/figures/fig15_qualification_ladder.svg", "\n".join(svg))


def gen_ch15_hardware_fault_injection():
    """
    Figure 15.2: Real-Time Hardware Fault Injection & Dynamic Enforcement Timeline.
    Multi-channel logic analyzer oscilloscope trace demonstrating the exact microsecond
    cascade from sensor freeze / expired intent to real-time MCU takeover and physical containment.
    """
    W = 940
    H = 490
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')

    # Title & Subtitle
    svg.append(f'<text x="{W/2}" y="28" class="title">REAL-TIME HARDWARE FAULT INJECTION &amp; INTERVENTION TIMELINE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Digital Logic Analyzer &amp; Pressure Sensor Trace: Sensor Stall &amp; Expired Intent → Hardware Gate Preemption → Containment Corridor</text>')

    # Oscilloscope Display Box
    ox = 35
    oy = 62
    ow = W - 70
    oh = 405
    svg.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="8" fill="#0B132B" stroke="{BORDER_DARK}" stroke-width="1.5"/>')

    # Background grid lines
    for gx in range(ox+40, ox+ow, 65):
        svg.append(f'<line x1="{gx}" y1="{oy+10}" x2="{gx}" y2="{oy+oh-24}" stroke="#1C2541" stroke-width="0.9"/>')
    for gy in range(oy+32, oy+oh-24, 38):
        svg.append(f'<line x1="{ox+10}" y1="{gy}" x2="{ox+ow-10}" y2="{gy}" stroke="#1C2541" stroke-width="0.9"/>')

    # Time offsets mapped to X coordinates (Time scale: 0 to 60 ms)
    t0_x = ox + 200  # t = 0 ms: Fault Injected (Sensor Register Freeze)
    t1_x = ox + 340  # t = 20 ms: Expiring Intent Lease Timeout
    t2_x = ox + 410  # t = 24 ms: Diagnostic Anomaly Detector Fires
    t3_x = ox + 470  # t = 26 ms: Hardware Gate Override & MPU Disconnect
    tsafe_x = ox + 680  # t = 42.8 ms: Physical Pressure Peak Arrested & Safe State
    tdead_x = ox + 730  # t = 45.0 ms: Max Vessel Physical Deadline Bound

    # Vertical Event Markers
    markers = [
        (t0_x, "t₀ = 0.0 ms", "INJECT STALE ADC", CORAL),
        (t1_x, "t₁ = 20.0 ms", "LEASE EXPIRES", AMBER),
        (t2_x, "t₂ = 24.0 ms", "DETECTOR FIRES", PURPLE),
        (t3_x, "t₃ = 26.0 ms", "HARDWARE OVERRIDE", PETROL),
        (tsafe_x, "t₄ = 42.8 ms", "SAFE CONTAINMENT", TEAL),
        (tdead_x, "t_limit = 45.0 ms", "VESSEL DEADLINE", CRIMSON)
    ]
    for mx, mtime, mlabel, mcol in markers:
        svg.append(f'<line x1="{mx}" y1="{oy+10}" x2="{mx}" y2="{oy+oh-26}" stroke="{mcol}" stroke-width="1.2" stroke-dasharray="3,3"/>')
        svg.append(f'<rect x="{mx-38}" y="{oy+12}" width="76" height="22" rx="3" fill="{mcol}"/>')
        svg.append(f'<text x="{mx}" y="{oy+22}" font-size="7" font-weight="700" fill="#FFFFFF" text-anchor="middle">{mtime}</text>')
        svg.append(f'<text x="{mx}" y="{oy+31}" font-size="6" font-weight="600" fill="#FFFFFF" text-anchor="middle">{mlabel}</text>')

    # Channel 1: Injected Sensor Disturbance (Optical Flow / ADC register)
    ch1_y = oy + 64
    svg.append(f'<text x="{ox+14}" y="{ch1_y}" font-size="8.5" font-weight="700" fill="{CORAL}">CH1: Sensor ADC Ingest</text>')
    svg.append(f'<text x="{ox+14}" y="{ch1_y+10}" font-size="7" fill="#8D99AE">4.2 kg/s Stale Latch</text>')
    p1 = [
        (ox+110, ch1_y+6), (ox+125, ch1_y+6), (ox+125, ch1_y-6), (ox+140, ch1_y-6),
        (ox+140, ch1_y+6), (ox+155, ch1_y+6), (ox+155, ch1_y-6), (ox+170, ch1_y-6),
        (ox+170, ch1_y+6), (ox+185, ch1_y+6), (ox+185, ch1_y-6), (t0_x, ch1_y-6),
        (t0_x, ch1_y), (ox+ow-15, ch1_y)
    ]
    for k in range(len(p1)-1):
        svg.append(f'<line x1="{p1[k][0]}" y1="{p1[k][1]}" x2="{p1[k+1][0]}" y2="{p1[k+1][1]}" stroke="{CORAL}" stroke-width="1.8"/>')
    svg.append(f'<text x="{t0_x+10}" y="{ch1_y-4}" font-size="7" font-weight="700" fill="{CORAL}">[FROZEN ADC REGISTER: 0.0 Hz update]</text>')

    # Channel 2: Intent Lease Validity Contract
    ch2_y = oy + 124
    svg.append(f'<text x="{ox+14}" y="{ch2_y}" font-size="8.5" font-weight="700" fill="{AMBER}">CH2: Expiring Intent Lease</text>')
    svg.append(f'<text x="{ox+14}" y="{ch2_y+10}" font-size="7" fill="#8D99AE">Cryptographic Lease t_exp</text>')
    svg.append(f'<line x1="{ox+110}" y1="{ch2_y-6}" x2="{t1_x}" y2="{ch2_y-6}" stroke="{AMBER}" stroke-width="2"/>')
    svg.append(f'<line x1="{t1_x}" y1="{ch2_y-6}" x2="{t1_x}" y2="{ch2_y+8}" stroke="{AMBER}" stroke-width="2"/>')
    svg.append(f'<line x1="{t1_x}" y1="{ch2_y+8}" x2="{ox+ow-15}" y2="{ch2_y+8}" stroke="{AMBER}" stroke-width="2"/>')
    svg.append(f'<text x="{ox+120}" y="{ch2_y-10}" font-size="7" font-weight="600" fill="{AMBER}">LEASE VALID (Host MPU Authority)</text>')
    svg.append(f'<text x="{t1_x+10}" y="{ch2_y+5}" font-size="7" font-weight="700" fill="{AMBER}">LEASE EXPIRED (Drop Authority)</text>')

    # Channel 3: MCU Safety Enforcer Anomaly Detector
    ch3_y = oy + 184
    svg.append(f'<text x="{ox+14}" y="{ch3_y}" font-size="8.5" font-weight="700" fill="{PURPLE}">CH3: Enforcer Fault Detector</text>')
    svg.append(f'<text x="{ox+14}" y="{ch3_y+10}" font-size="7" fill="#8D99AE">Stale Check &amp; NMI Assert</text>')
    svg.append(f'<line x1="{ox+110}" y1="{ch3_y+8}" x2="{t2_x}" y2="{ch3_y+8}" stroke="{PURPLE}" stroke-width="2"/>')
    svg.append(f'<line x1="{t2_x}" y1="{ch3_y+8}" x2="{t2_x}" y2="{ch3_y-6}" stroke="{PURPLE}" stroke-width="2"/>')
    svg.append(f'<line x1="{t2_x}" y1="{ch3_y-6}" x2="{ox+ow-15}" y2="{ch3_y-6}" stroke="{PURPLE}" stroke-width="2"/>')
    svg.append(f'<text x="{t2_x+10}" y="{ch3_y-10}" font-size="7" font-weight="700" fill="{PURPLE}">DIAGNOSTIC TRIP: Δt_det = 4.0 ms (NMI Raised)</text>')

    # Channel 4: Hardware Multiplexer & PWM Gate Driver
    ch4_y = oy + 244
    svg.append(f'<text x="{ox+14}" y="{ch4_y}" font-size="8.5" font-weight="700" fill="{PETROL}">CH4: Actuator Gate Driver</text>')
    svg.append(f'<text x="{ox+14}" y="{ch4_y+10}" font-size="7" fill="#8D99AE">Hardware MUX Preemption</text>')
    for px in range(ox+110, t3_x-10, 16):
        svg.append(f'<line x1="{px}" y1="{ch4_y+6}" x2="{px+8}" y2="{ch4_y+6}" stroke="{BLUE}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{px+8}" y1="{ch4_y+6}" x2="{px+8}" y2="{ch4_y-6}" stroke="{BLUE}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{px+8}" y1="{ch4_y-6}" x2="{px+16}" y2="{ch4_y-6}" stroke="{BLUE}" stroke-width="1.5"/>')
        svg.append(f'<line x1="{px+16}" y1="{ch4_y-6}" x2="{px+16}" y2="{ch4_y+6}" stroke="{BLUE}" stroke-width="1.5"/>')
    svg.append(f'<line x1="{t3_x}" y1="{ch4_y-6}" x2="{t3_x}" y2="{ch4_y+8}" stroke="{PETROL}" stroke-width="2"/>')
    svg.append(f'<line x1="{t3_x}" y1="{ch4_y+8}" x2="{ox+ow-15}" y2="{ch4_y+8}" stroke="{PETROL}" stroke-width="2"/>')
    svg.append(f'<text x="{t3_x+10}" y="{ch4_y-8}" font-size="7" font-weight="700" fill="{PETROL}">MUX DISCONNECT: Linux MPU Evicted (Δt_takeover = 2.0 ms)</text>')

    # Channel 5: Actuator Drive Current & Dynamic Braking (I_phase)
    ch5_y = oy + 304
    svg.append(f'<text x="{ox+14}" y="{ch5_y}" font-size="8.5" font-weight="700" fill="{NAVY}">CH5: Motor Phase Current</text>')
    svg.append(f'<text x="{ox+14}" y="{ch5_y+10}" font-size="7" fill="#8D99AE">Reverse Torque I_phase</text>')
    p5 = [
        (ox+110, ch5_y-4), (t3_x, ch5_y-4),
        (t3_x, ch5_y+10), (tsafe_x-40, ch5_y+10),
        (tsafe_x, ch5_y), (ox+ow-15, ch5_y)
    ]
    for k in range(len(p5)-1):
        svg.append(f'<line x1="{p5[k][0]}" y1="{p5[k][1]}" x2="{p5[k+1][0]}" y2="{p5[k+1][1]}" stroke="{NAVY}" stroke-width="1.8"/>')
    svg.append(f'<text x="{ox+120}" y="{ch5_y-8}" font-size="7" fill="{NAVY}">+8.0 A (Driving)</text>')
    svg.append(f'<text x="{t3_x+10}" y="{ch5_y+19}" font-size="7" font-weight="700" fill="{CORAL}">-12.0 A (Regenerative Dynamic Braking)</text>')
    svg.append(f'<text x="{tsafe_x+10}" y="{ch5_y-4}" font-size="7" fill="{TEAL}">0.0 A (Halted)</text>')

    # Channel 6: Physical Process State P(t) [Fluid Pressure & Containment Corridor]
    ch6_y = oy + 364
    svg.append(f'<text x="{ox+14}" y="{ch6_y}" font-size="8.5" font-weight="700" fill="{TEAL}">CH6: Manifold Pressure P(t)</text>')
    svg.append(f'<text x="{ox+14}" y="{ch6_y+10}" font-size="7" fill="#8D99AE">Fluid Transient vs Bound</text>')

    burst_y = ch6_y - 28
    contain_y = ch6_y - 18
    nom_y = ch6_y + 6

    svg.append(f'<line x1="{ox+110}" y1="{burst_y}" x2="{ox+ow-15}" y2="{burst_y}" stroke="{CORAL}" stroke-width="1" stroke-dasharray="2,2"/>')
    svg.append(f'<text x="{ox+ow-130}" y="{burst_y-3}" font-size="6.5" font-weight="700" fill="{CORAL}">PIPE BURST LIMIT: 22.0 bar</text>')

    svg.append(f'<line x1="{ox+110}" y1="{contain_y}" x2="{ox+ow-15}" y2="{contain_y}" stroke="{AMBER}" stroke-width="1" stroke-dasharray="4,2"/>')
    svg.append(f'<text x="{ox+ow-165}" y="{contain_y-3}" font-size="6.5" font-weight="700" fill="{AMBER}">MAX CONTAINMENT BOUND: 16.5 bar</text>')

    p6_curve = [
        (ox+110, nom_y), (t0_x, nom_y),
        (t0_x+50, nom_y-6), (t1_x, nom_y-12),
        (t2_x, nom_y-17), (t3_x, nom_y-20),
        (t3_x+40, contain_y+2),  # Peak: 16.2 bar
        (t3_x+100, nom_y-10),
        (tsafe_x, nom_y), (ox+ow-15, nom_y)
    ]
    d_path = f"M {p6_curve[0][0]} {p6_curve[0][1]}"
    for pt in p6_curve[1:]:
        d_path += f" L {pt[0]} {pt[1]}"
    svg.append(f'<path d="{d_path}" fill="none" stroke="{TEAL}" stroke-width="2.2"/>')
    svg.append(f'<circle cx="{t3_x+40}" cy="{contain_y+2}" r="3.5" fill="{AMBER}" stroke="#FFFFFF" stroke-width="1"/>')
    svg.append(f'<text x="{t3_x+48}" y="{contain_y+6}" font-size="7" font-weight="700" fill="{TEAL}">P_peak = 16.2 bar &lt; 16.5 bar ✓</text>')

    # Bottom summary measurement callout banner
    svg.append(f'<text x="{ox+ow/2}" y="{oy+oh-8}" font-size="8.5" font-weight="600" fill="#CBD5E1" text-anchor="middle">'
               f'Total Enforced Reaction: Δt_total = 42.8 ms &lt; 45.0 ms Vessel Limit  |  Physical Margin: +0.3 bar Clearance  |  VERDICT: PASS (Trace Verified)'
               f'</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/15-verification/figures/fig15_hardware_fault_injection.svg", "\n".join(svg))


def run_all():
    print("Generating Chapter 15 figures...")
    gen_ch15_qualification_ladder()
    gen_ch15_hardware_fault_injection()

if __name__ == "__main__":
    run_all()
