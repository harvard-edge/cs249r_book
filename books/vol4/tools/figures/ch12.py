"""
book/tools/figures/ch12.py
Figures for Chapter 12: Whole-System Qualification & Assurance.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch12_cae():
    W = 920
    H = 450
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="30" class="title">CLAIM-ARGUMENT-EVIDENCE (CAE) SAFETY CASE ARCHITECTURE</text>')
    svg.append(f'<text x="{W/2}" y="46" class="subtitle">Linking High-Level Safety Claims to Defensible Bench Metrology and Signed Release Verdicts</text>')

    cx = 40
    cy = 66
    cw = W - 80
    svg.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="46" rx="6" fill="{CRIMSON}" fill-opacity="0.06" stroke="{CRIMSON}" stroke-width="1.3"/>')
    svg.append(f'<text x="{W/2}" y="{cy+18}" font-size="11" font-weight="700" fill="{CRIMSON}" text-anchor="middle">TOP-LEVEL SAFETY CLAIM (C_top)</text>')
    svg.append(f'<text x="{W/2}" y="{cy+34}" font-size="10" font-weight="600" fill="{INK}" text-anchor="middle">"The Physical AI Agent operates acceptably safe within Operational Design Domain X_ODD."</text>')

    cols = [
        ("ARGUMENT A1", "Fault Containment", "All single-point hardware and software crashes are deterministically arrested by the MCU safety enforcer within 50 ms.",
         "EVIDENCE E1", "HIL Fault Log (REL-01)", "• 1000/1000 faults contained\n• 0 collisions recorded\n• Max stopping time: 42 ms\n• Watchdog trip: 50 ms"),
        ("ARGUMENT A2", "Latency Freshness", "The P99.9 sense-to-actuation latency Δt_wall remains strictly bounded so stopping distance d_stop ≤ d_gap.",
         "EVIDENCE E2", "Metrology CDF (REQ-01)", "• P50 = 22 ms\n• P99.9 = 72 ms &lt; 80 ms\n• Margin: +18 cm clearance\n• Zero seqlock overruns"),
        ("ARGUMENT A3", "Human Authority", "Human operators retain un-preemptible authority via bumpless joystick override and dedicated hardware E-stop circuits.",
         "EVIDENCE E3", "Override Logs (AUTH-01)", "• 50/50 seamless takeovers\n• Peak jerk &lt; 15 rad/s³\n• Zero gearbox shock damage\n• Tamper-evident hash log"),
        ("ARGUMENT A4", "ODD Enforcement", "Real-time interoceptive and exteroceptive health monitors detect out-of-ODD transitions and trigger safe deceleration.",
         "EVIDENCE E4", "ODD Monitor Logs", "• Dark/glare detection &lt; 20 ms\n• Friction drop fallback\n• Thermal derating at 85°C\n• Category 1 stop verified")
    ]

    col_w = 200
    col_gap = 12
    start_x = (W - (4 * col_w + 3 * col_gap)) / 2

    for i, (a_num, a_title, a_desc, e_num, e_title, e_desc) in enumerate(cols):
        x = start_x + i * (col_w + col_gap)

        ay = 132
        ah = 100
        svg.append(f'<rect x="{x}" y="{ay}" width="{col_w}" height="{ah}" rx="6" fill="{BLUE}" fill-opacity="0.04" stroke="{BLUE}" stroke-width="1.1"/>')
        svg.append(f'<text x="{x+10}" y="{ay+18}" font-size="10" font-weight="700" fill="{BLUE}">{a_num}</text>')
        svg.append(f'<text x="{x+10}" y="{ay+32}" font-size="10.5" font-weight="700" fill="{INK}">{a_title}</text>')

        words = a_desc.split()
        l1 = " ".join(words[:5])
        l2 = " ".join(words[5:11])
        l3 = " ".join(words[11:])
        svg.append(f'<text x="{x+10}" y="{ay+48}" font-size="8.5" fill="{SLATE}">{l1}</text>')
        svg.append(f'<text x="{x+10}" y="{ay+62}" font-size="8.5" fill="{SLATE}">{l2}</text>')
        svg.append(f'<text x="{x+10}" y="{ay+76}" font-size="8.5" fill="{SLATE}">{l3}</text>')

        svg.append(f'<line x1="{x+col_w/2}" y1="{cy+46}" x2="{x+col_w/2}" y2="{ay}" stroke="{CRIMSON}" stroke-width="1.2" marker-end="url(#arr-crimson)"/>')

        ey = 248
        eh = 110
        svg.append(f'<rect x="{x}" y="{ey}" width="{col_w}" height="{eh}" rx="6" fill="{TEAL}" fill-opacity="0.04" stroke="{TEAL}" stroke-width="1.1"/>')
        svg.append(f'<text x="{x+10}" y="{ey+18}" font-size="10" font-weight="700" fill="{TEAL}">{e_num}</text>')
        svg.append(f'<text x="{x+10}" y="{ey+32}" font-size="10.5" font-weight="700" fill="{INK}">{e_title}</text>')

        for idx, el in enumerate(e_desc.split("\n")):
            svg.append(f'<text x="{x+10}" y="{ey+50+idx*14}" font-size="8.5" fill="{SLATE}">{el}</text>')

        svg.append(f'<line x1="{x+col_w/2}" y1="{ay+ah}" x2="{x+col_w/2}" y2="{ey}" stroke="{BLUE}" stroke-width="1.2" marker-end="url(#arr-blue)"/>')
        svg.append(f'<line x1="{x+col_w/2}" y1="{ey+eh}" x2="{x+col_w/2}" y2="380" stroke="{TEAL}" stroke-width="1.2" marker-end="url(#arr-teal)"/>')

    vy = 380
    svg.append(f'<rect x="{cx}" y="{vy}" width="{cw}" height="48" rx="6" fill="{BG_LIGHT}" stroke="{NAVY}" stroke-width="1.2"/>')
    svg.append(f'<text x="{W/2}" y="{vy+18}" font-size="11" font-weight="700" fill="{NAVY}" text-anchor="middle">THE 3 ACCOUNTABLE RELEASE VERDICTS</text>')
    svg.append(f'<text x="{W/2}" y="{vy+34}" font-size="9.5" text-anchor="middle">'
               f'<tspan font-weight="700" fill="{TEAL}">✓ DEPLOY</tspan> (All thresholds met; unconstrained ODD)  |  '
               f'<tspan font-weight="700" fill="{AMBER}">⚠ CONDITION</tspan> (Restricted speed / human supervisor)  |  '
               f'<tspan font-weight="700" fill="{CORAL}">✕ REFUSE</tspan> (Gate failure; deployment blocked)'
               f'</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/12-assurance/figures/fig11_cae_safety_case.svg", "\n".join(svg))

def gen_ch12_ladder():
    W = 920
    H = 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="30" class="title">THE 4-RUNG QUALIFICATION LADDER FOR PHYSICAL AI</text>')
    svg.append(f'<text x="{W/2}" y="46" class="subtitle">Progressive Verification from Offline Log Replay to Active Shadow Fleet Deployment</text>')

    rungs = [
        ("RUNG 1 · BASELINE", "Historical Log Replay", "Offline Dataset Metrology", "Gate 0: Affordance Error &lt; 2 cm",
         [("Execution Substrate:", ["Static logged sensor streams", "Offline GPU training server", "Open-loop prediction loss"]),
          ("Verification Focus:", ["Model affordance accuracy", "Latent token prediction error", "Offline inference throughput"]),
          ("Blind Spot / Limit:", ["Zero closed-loop causality", "Blind to physical momentum", "Misses P99 latency tails"])],
         SLATE),
        ("RUNG 2 · CLOSED LOOP", "Physics Simulation", "Domain Randomization", "Gate 1: &gt; 98% Pass (10⁴ Seeds)",
         [("Execution Substrate:", ["MuJoCo / Isaac Sim physics", "Multi-GPU parallel rollouts", "Randomized mass, friction, light"]),
          ("Verification Focus:", ["Closed-loop policy stability", "Dynamic collision avoidance", "Generalization over 10⁴ seeds"]),
          ("Blind Spot / Limit:", ["Sim-to-real physics gap", "Idealized compute/bus timing", "Misses silicon fault modes"])],
         BLUE),
        ("RUNG 3 · TARGET SILICON", "Hardware-in-the-Loop", "Real Silicon + Plant Emulator", "Gate 2: 100% Fault Containment",
         [("Execution Substrate:", ["Production MPU + Real-Time MCU", "10 kHz FPGA plant emulator", "Real SPI, CAN, and MIPI buses"]),
          ("Verification Focus:", ["Seeded cross-layer faults", "MCU safety watchdog timing", "Memory DMA bus contention"]),
          ("Blind Spot / Limit:", ["Synthetic visual assets", "Emulated optical lighting", "Bounded human interaction"])],
         BRONZE),
        ("RUNG 4 · FLEET SHADOW", "Shadow Fleet Mode", "Passive Real-World Fleet", "Gate 3: 0 Divergence / 500 hrs",
         [("Execution Substrate:", ["Physical robot in active fleet", "Real-world warehouse ODD", "Inferences run in background"]),
          ("Verification Focus:", ["Uncontained divergence audit", "Long-tail corner case capture", "Zero actuator safety authority"]),
          ("Blind Spot / Limit:", ["Policy cannot actuate world", "Dependent on fleet coverage", "Requires data privacy filters"])],
         CRIMSON)
    ]

    card_w = 202
    gap = 14
    start_x = (W - (4 * card_w + 3 * gap)) / 2

    for i, (tag, title, sub, gate, sections, col) in enumerate(rungs):
        x = start_x + i * (card_w + gap)
        y = 68
        h = 320

        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{h}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="24" rx="6" fill="{col}" fill-opacity="0.1"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+16}" font-size="9" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+42}" font-size="12" font-weight="700" fill="{INK}" text-anchor="middle">{title}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+56}" font-size="9" fill="{MUTED}" text-anchor="middle">{sub}</text>')

        svg.append(f'<rect x="{x+10}" y="{y+66}" width="{card_w-20}" height="18" rx="3" fill="{col}" fill-opacity="0.08" stroke="{col}" stroke-width="0.8"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+78}" font-size="8" font-weight="700" fill="{col}" text-anchor="middle">{gate}</text>')

        cy = y + 96
        for sec_title, sec_bullets in sections:
            svg.append(f'<text x="{x+10}" y="{cy}" font-size="8.5" font-weight="700" fill="{INK}">{sec_title}</text>')
            cy += 12
            for b in sec_bullets:
                svg.append(f'<text x="{x+14}" y="{cy}" font-size="8" fill="{SLATE}">• {b}</text>')
                cy += 11
            cy += 4

        if i < 3:
            ax1 = x + card_w + 1
            ax2 = ax1 + gap - 2
            ay = y + h/2
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{col}" stroke-width="1.8" marker-end="url(#arr-blue)"/>')

    by = 402
    svg.append(f'<rect x="{start_x}" y="{by}" width="{4*card_w + 3*gap}" height="42" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="{by+18}" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">THE PROMOTION PRINCIPLE</text>')
    svg.append(f'<text x="{W/2}" y="{by+32}" font-size="9" fill="{SLATE}" text-anchor="middle">A high score on Rung 1 or 2 is a necessary prerequisite but never sufficient proof of release. Only Rungs 3 and 4 provide defensible evidence for release.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/12-assurance/figures/fig11_qualification_ladder.svg", "\n".join(svg))

def gen_ch12_faults():
    W = 880
    H = 440
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">CROSS-LAYER SEEDED FAULT INJECTION ARCHITECTURE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Stress-Testing Every Architectural Layer to Verify Deterministic MCU Safety Containment</text>')

    lx = 24
    lw = 240
    layers = [
        ("LAYER 1: SENSORY &amp; TRANSDUCTION", ["Dropped MIPI frames (&gt; 100 ms)", "Blinding glare / sudden lux drop", "PTP clock skew (&gt; 5 ms)"]),
        ("LAYER 2: COMPUTE, BUS &amp; MEMORY", ["Linux kernel panic / thread crash", "Synthetic DMA bus saturation", "malloc allocation stalls &amp; page faults"]),
        ("LAYER 3: MODEL &amp; ALGORITHMIC", ["Hallucinated 3D bounding boxes", "Out-of-workspace trajectories", "Discontinuous acceleration jumps"]),
        ("LAYER 4: ELECTRICAL &amp; PHYSICAL", ["Motor over-temp (&gt; 105°C)", "Battery voltage sag (&lt; 10.5 V)", "Encoder line break / phase loss"])
    ]

    cur_y = 64
    for title, items in layers:
        lh = 78
        svg.append(f'<rect x="{lx}" y="{cur_y}" width="{lw}" height="{lh}" rx="6" fill="{CORAL}" fill-opacity="0.06" stroke="{CORAL}" stroke-width="1"/>')
        svg.append(f'<text x="{lx+10}" y="{cur_y+16}" font-size="9.5" font-weight="700" fill="{CORAL}">{title}</text>')
        for idx, it in enumerate(items):
            svg.append(f'<text x="{lx+14}" y="{cur_y+32+idx*14}" font-size="8.5" fill="{SLATE}">• {it}</text>')
        cur_y += lh + 10

    cx = 290
    cw = 330

    svg.append(f'<rect x="{cx}" y="64" width="{cw}" height="100" rx="6" fill="{BG_WHITE}" stroke="{BRONZE}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{cx+12}" y="84" font-size="11" font-weight="700" fill="{BRONZE}">APPLICATION PROCESSOR (Linux MPU · Sys 2/1.5)</text>')
    svg.append(f'<text x="{cx+16}" y="102" font-size="8.5" fill="{SLATE}">• Runs Vision Transformers, Deliberation &amp; Action Chunking</text>')
    svg.append(f'<text x="{cx+16}" y="118" font-size="8.5" fill="{SLATE}">• Untrusted, stochastic, prone to crashes and latency tails</text>')
    svg.append(f'<text x="{cx+16}" y="134" font-size="8.5" fill="{SLATE}">• Emits candidate trajectory buffer p_t over shared IPC link</text>')

    svg.append(f'<line x1="{cx+cw/2}" y1="164" x2="{cx+cw/2}" y2="194" stroke="{BRONZE}" stroke-width="1.4" stroke-dasharray="4,2" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<rect x="{cx+cw/2-45}" y="172" width="90" height="15" rx="3" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="0.8"/>')
    svg.append(f'<text x="{cx+cw/2}" y="183" font-size="8" font-weight="600" fill="{BRONZE}" text-anchor="middle">Candidate p_t</text>')

    svg.append(f'<rect x="{cx}" y="194" width="{cw}" height="130" rx="6" fill="{TEAL}" fill-opacity="0.05" stroke="{TEAL}" stroke-width="1.4" filter="url(#shadow)"/>')
    svg.append(f'<text x="{cx+12}" y="214" font-size="11.5" font-weight="700" fill="{TEAL}">REAL-TIME SAFETY ENFORCER (MCU · Sys 1)</text>')
    svg.append(f'<text x="{cx+16}" y="234" font-size="9" font-weight="700" fill="{INK}">1. Control Barrier Function Invariant (h(x) ≥ 0):</text>')
    svg.append(f'<text x="{cx+24}" y="248" font-size="8.5" fill="{SLATE}">Projects candidate p_t onto forward invariant safe set C</text>')
    svg.append(f'<text x="{cx+16}" y="266" font-size="9" font-weight="700" fill="{INK}">2. Dynamic Stopping Clearance Check (d_gap &gt; d_stop):</text>')
    svg.append(f'<text x="{cx+24}" y="280" font-size="8.5" fill="{SLATE}">Vetoes proposals exceeding braking envelope</text>')
    svg.append(f'<text x="{cx+16}" y="298" font-size="9" font-weight="700" fill="{INK}">3. Zero-Software Hardware Watchdog Monitor:</text>')
    svg.append(f'<text x="{cx+24}" y="312" font-size="8.5" fill="{SLATE}">Trips Category 1 dynamic stop if MPU heartbeat &gt; 50 ms</text>')

    svg.append(f'<line x1="{cx+cw/2}" y1="324" x2="{cx+cw/2}" y2="350" stroke="{TEAL}" stroke-width="1.5" marker-end="url(#arr-teal)"/>')
    svg.append(f'<rect x="{cx+cw/2-60}" y="330" width="120" height="15" rx="3" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{cx+cw/2}" y="341" font-size="8" font-weight="600" fill="{TEAL}" text-anchor="middle">Permitted u_t / E-Stop</text>')

    svg.append(f'<rect x="{cx}" y="350" width="{cw}" height="56" rx="6" fill="{NAVY}" fill-opacity="0.05" stroke="{NAVY}" stroke-width="1"/>')
    svg.append(f'<text x="{cx+12}" y="370" font-size="11" font-weight="700" fill="{NAVY}">PHYSICAL PLANT &amp; ACTUATION (W_t → W_t+1)</text>')
    svg.append(f'<text x="{cx+16}" y="390" font-size="8.5" fill="{SLATE}">Closed-loop motor current loops · Mechanical brakes · Verified safe stop</text>')

    svg.append(f'<path d="M {lx+lw} 103 H {cx}" fill="none" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')
    svg.append(f'<path d="M {lx+lw} 191 H {cx-15} V 170 H {cx}" fill="none" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')
    svg.append(f'<path d="M {lx+lw} 279 H {cx}" fill="none" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')
    svg.append(f'<path d="M {lx+lw} 367 H {cx}" fill="none" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')

    rx = 646
    rw = 210
    svg.append(f'<rect x="{rx}" y="64" width="{rw}" height="342" rx="6" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1.2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{rx+12}" y="88" font-size="11.5" font-weight="700" fill="{TEAL}">✓ PASS CRITERION</text>')

    crit = [
        ("100% Containment Rate:", ["Zero collisions (d_gap &gt; 0)", "Bounded stopping time", "Controlled halt &lt; 50 ms"]),
        ("Zero Uncontained Escapes:", ["Any single uncontained fault", "trips a REFUSE verdict."]),
        ("Hardware Watchdog:", ["Dedicated NMI timer operates", "independently of Linux OS."])
    ]

    cur_y = 108
    for c_title, c_bullets in crit:
        svg.append(f'<text x="{rx+12}" y="{cur_y}" font-size="9" font-weight="700" fill="{INK}">{c_title}</text>')
        cur_y += 14
        for cb in c_bullets:
            svg.append(f'<text x="{rx+16}" y="{cur_y}" font-size="8.5" fill="{SLATE}">• {cb}</text>')
            cur_y += 13
        cur_y += 10

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/12-assurance/figures/fig11_cross_layer_faults.svg", "\n".join(svg))

def run_all():
    gen_ch12_cae()
    gen_ch12_ladder()
    gen_ch12_faults()

if __name__ == "__main__":
    run_all()
