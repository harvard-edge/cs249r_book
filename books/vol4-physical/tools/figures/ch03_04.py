"""
book/tools/figures/ch03_04.py
Figures for Chapter 3 (Cognitive Agency) and Chapter 4 (Multi-Rate Hierarchy).
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig03_agent_workflow():
    W = 920
    H = 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE FIVE COGNITIVE STAGES OF PHYSICAL AGENTS</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">From Continuous Physical Transduction to Real-Time Discrete Action Invariant Enforcement</text>')

    stages = [
        ("STAGE 1: TRANSDUCTION", "Physical Photons →", "Metric Tensors", "CMOS exposure · IMU sense\nMIPI zero-copy DMA views", NAVY),
        ("STAGE 2: WORLD MODELING", "Latent Belief &amp;", "SE(3) Frame Tree", "Dynamic transforms · Spatial map\nJEPA physics &amp; occlusion tracking", BLUE),
        ("STAGE 3: REASONING &amp; INTENT", "Goal Grounding &amp;", "Expiring Leases", "Edge VLM semantic spatial deliberation\nCoarse 3D target box + lease t_expire", BLUE),
        ("STAGE 4: ACTION PLANNING", "Generative Action", "Chunking (H=16)", "Diffusion / ACT trajectory synthesis\nC² continuous quintic spline ensembling", BRONZE),
        ("STAGE 5: REAL-TIME REFLEX", "Deterministic", "Barrier Enforcement", "1 kHz FreeRTOS QP solver: h(x) ≥ 0\nDynamic stopping check · Fallback FSM", PETROL)
    ]

    card_w = 168
    gap = 14
    start_x = (W - (5 * card_w + 4 * gap)) / 2

    for i, (tag, t1, t2, desc, col) in enumerate(stages):
        x = start_x + i * (card_w + gap)
        y = 70
        h = 320

        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{h}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="22" rx="6" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+15}" font-size="8" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+38}" font-size="9.5" font-weight="700" fill="{INK}" text-anchor="middle">{t1}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+51}" font-size="9.5" font-weight="700" fill="{INK}" text-anchor="middle">{t2}</text>')

        sub_y = y + 74
        for idx, l in enumerate(desc.split("\n")):
            svg.append(f'<text x="{x+8}" y="{sub_y+idx*16}" font-size="8" fill="{SLATE}">• {l}</text>')

        if i < 4:
            ax1 = x + card_w + 1
            ax2 = ax1 + gap - 2
            ay = y + h/2
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{col}" stroke-width="1.5" marker-end="url(#arr-blue)"/>')

    # Bottom Invariant Strip
    svg.append(f'<rect x="{start_x}" y="402" width="{5*card_w + 4*gap}" height="42" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="420" font-size="9.5" font-weight="700" fill="{NAVY}" text-anchor="middle">THE COGNITIVE PIPELINE PRINCIPLE</text>')
    svg.append(f'<text x="{W/2}" y="434" font-size="8.5" fill="{SLATE}" text-anchor="middle">High-level cognitive models propose intentions over long horizons; low-level reflex circuits guarantee physical safety over microsecond steps.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-cognition/figures/fig03_agent_workflow.svg", "\n".join(svg))

def gen_fig03_great_tension():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE FUNDAMENTAL SYSTEMS TENSION OF PHYSICAL AI</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">High-Capacity Learned Generative Models ⟷ Hard Real-Time Deterministic Safety Guarantees</text>')

    # Left: High-Capacity Learned Models
    lx = 30
    lw = 380
    svg.append(f'<rect x="{lx}" y="70" width="{lw}" height="320" rx="8" fill="{BLUE}" fill-opacity="0.04" stroke="{BLUE}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{lx}" y="70" width="{lw}" height="26" rx="8" fill="{BLUE}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{lx+lw/2}" y="88" font-size="10" font-weight="700" fill="{BLUE}" text-anchor="middle">HIGH-CAPACITY LEARNED MODELS (Cortex)</text>')
    svg.append(f'<text x="{lx+16}" y="120" font-size="11.5" font-weight="700" fill="{INK}">Transformers &amp; Diffusion Policies</text>')
    
    props_left = [
        ("Generalization:", "Open-world semantic reasoning, novel visual grounding"),
        ("Computational Substrate:", "Heterogeneous NPU / GPU with large DRAM footprint"),
        ("Execution Paradigm:", "Batched tensor inference, stochastic sampling"),
        ("Latency Profile:", "Variable 15–100 ms with long P99.9 tail spikes"),
        ("Failure Characteristics:", "Hallucinations, out-of-distribution drift, crashes")
    ]
    for idx, (p_t, p_d) in enumerate(props_left):
        by = 135 + idx * 46
        svg.append(f'<text x="{lx+16}" y="{by}" font-size="9" font-weight="700" fill="{BLUE}">• {p_t}</text>')
        svg.append(f'<text x="{lx+24}" y="{by+14}" font-size="8.5" fill="{SLATE}">{p_d}</text>')

    # Right: Hard Real-Time Control
    rx = 470
    rw = 380
    svg.append(f'<rect x="{rx}" y="70" width="{rw}" height="320" rx="8" fill="{PETROL}" fill-opacity="0.04" stroke="{PETROL}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{rx}" y="70" width="{rw}" height="26" rx="8" fill="{PETROL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{rx+rw/2}" y="88" font-size="10" font-weight="700" fill="{PETROL}" text-anchor="middle">HARD REAL-TIME DETERMINISTIC REFLEX (Enforcer)</text>')
    svg.append(f'<text x="{rx+16}" y="120" font-size="11.5" font-weight="700" fill="{INK}">Zero-Allocation Active-Set QP Barrier Solvers</text>')

    props_right = [
        ("Safety Guarantee:", "Strict forward invariance h(x) ≥ 0, zero collisions"),
        ("Computational Substrate:", "Real-Time MCU in static SRAM (0 dynamic malloc)"),
        ("Execution Paradigm:", "Deterministic 1000 Hz loop, bounded iterations"),
        ("Latency Profile:", "Bounded ≤ 175 µs with microsecond jitter (< 5 µs)"),
        ("Failure Characteristics:", "Fail-safe hardware arrest, Category 0/1 stops")
    ]
    for idx, (p_t, p_d) in enumerate(props_right):
        by = 135 + idx * 46
        svg.append(f'<text x="{rx+16}" y="{by}" font-size="9" font-weight="700" fill="{PETROL}">• {p_t}</text>')
        svg.append(f'<text x="{rx+24}" y="{by+14}" font-size="8.5" fill="{SLATE}">{p_d}</text>')

    # Central Handshake Badge
    cx = 440
    cy = 230
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="32" fill="{BG_WHITE}" stroke="{BRONZE}" stroke-width="2" filter="url(#shadow)"/>')
    svg.append(f'<text x="{cx}" y="{cy-4}" font-size="8" font-weight="700" fill="{BRONZE}" text-anchor="middle">PROPOSE</text>')
    svg.append(f'<text x="{cx}" y="{cy+8}" font-size="9" font-weight="700" fill="{PETROL}" text-anchor="middle">⟷</text>')
    svg.append(f'<text x="{cx}" y="{cy+18}" font-size="8" font-weight="700" fill="{PETROL}" text-anchor="middle">PERMIT</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-cognition/figures/fig03_great_tension.svg", "\n".join(svg))

def gen_fig03_modular_blueprint():
    W = 880
    H = 440
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">MODULAR BLUEPRINT OF PHYSICAL AGENT ARCHITECTURE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Layered Architectural Contract: Observations, World Models, Intent Leases, Spline Chunks, and Barrier Reflexes</text>')

    layers = [
        ("LAYER 1: SENSORY CONTRACT", "OBS-01: Zero-Copy Ring Buffers &amp; Hardware Timestamps", NAVY),
        ("LAYER 2: SPATIAL STATE CONTRACT", "STATE-01: Dynamic SE(3) Coordinate Frame Tree &amp; Proprioceptive Innovation", BLUE),
        ("LAYER 3: INTENT LEASE CONTRACT", "INTENT-01: 3D Workspace Grounding &amp; Monotonic Countdown Leases (t_expire)", BLUE),
        ("LAYER 4: TRAJECTORY CHUNK CONTRACT", "PLAN-01: H=16 Step Action Chunks &amp; C² Quintic Spline Continuity", BRONZE),
        ("LAYER 5: SAFETY ENFORCER CONTRACT", "ENF-01: 1 kHz Zero-Malloc Active-Set QP Barrier Filter (h(x) ≥ 0)", PETROL)
    ]

    cur_y = 66
    for l_tag, l_desc, col in layers:
        svg.append(f'<rect x="40" y="{cur_y}" width="{W-80}" height="56" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="40" y="{cur_y}" width="6" height="56" rx="3" fill="{col}"/>')
        svg.append(f'<text x="58" y="{cur_y+22}" font-size="10" font-weight="700" fill="{col}">{l_tag}</text>')
        svg.append(f'<text x="58" y="{cur_y+40}" font-size="9" font-weight="600" fill="{INK}">{l_desc}</text>')
        cur_y += 66

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-cognition/figures/fig03_modular_blueprint.svg", "\n".join(svg))

def gen_fig03_three_cadences():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE THREE CADENCES OF EMBODIED INTELLIGENCE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">System 2 Deliberation (1 Hz) ⟷ System 1.5 Planning (20 Hz) ⟷ System 1 Reflex (1000 Hz)</text>')

    cadences = [
        ("CADENCE 1: SYSTEM 2", "Deliberate Reasoning (1–2 Hz)", "Period: 500–1000 ms · Host MPU / VLM",
         ["Semantic task grounding", "Coarse 3D target bounding", "Expiring intent leases t_expire", "High latency tolerance"],
         BLUE),
        ("CADENCE 2: SYSTEM 1.5", "Trajectory Planning (20–50 Hz)", "Period: 20–50 ms · Host MPU / NPU",
         ["Diffusion policy / ACT chunks", "H=16 step horizon generation", "C² continuous spline fitting", "Temporal overlap ensembling"],
         BRONZE),
        ("CADENCE 3: SYSTEM 1", "Safety Reflex (1000 Hz)", "Period: 1.0 ms · Real-Time MCU",
         ["Active-Set CBF QP solver", "Control Barrier Invariant h(x) ≥ 0", "Dynamic stopping envelope check", "Deterministic FreeRTOS priority"],
         PETROL)
    ]

    cw = 250
    gap = 20
    start_x = (W - (3 * cw + 2 * gap)) / 2

    for i, (tag, title, rate, items, col) in enumerate(cadences):
        x = start_x + i * (cw + gap)
        y = 70
        h = 330

        svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{h}" rx="8" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.3" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="24" rx="8" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+cw/2}" y="{y+16}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{x+cw/2}" y="{y+42}" font-size="11" font-weight="700" fill="{INK}" text-anchor="middle">{title}</text>')
        svg.append(f'<text x="{x+cw/2}" y="{y+58}" font-size="8.5" font-weight="600" fill="{col}" text-anchor="middle">{rate}</text>')

        cy = y + 80
        for it in items:
            svg.append(f'<text x="{x+12}" y="{cy}" font-size="8.5" fill="{SLATE}">• {it}</text>')
            cy += 24

        if i < 2:
            ax1 = x + cw + 1
            ax2 = ax1 + gap - 2
            ay = y + h/2
            svg.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{col}" stroke-width="1.8" marker-end="url(#arr-blue)"/>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/03-cognition/figures/fig03_three_cadences.svg", "\n".join(svg))
    save_svg_and_pdf("book/chapters/04-hierarchy/figures/fig03_three_cadences.svg", "\n".join(svg))

def gen_fig03_codesign_matrix():
    # Reuse identical co-design matrix for Ch 3 and Ch 4
    from .ch01 import gen_fig01_codesign_matrix
    gen_fig01_codesign_matrix()
    # Also save to Ch 3 and Ch 4 locations
    import shutil
    shutil.copyfile("book/chapters/01-boundary/figures/fig01_codesign_matrix.svg", "book/chapters/03-cognition/figures/fig03_codesign_matrix.svg")
    shutil.copyfile("book/chapters/01-boundary/figures/fig01_codesign_matrix.pdf", "book/chapters/03-cognition/figures/fig03_codesign_matrix.pdf")
    shutil.copyfile("book/chapters/01-boundary/figures/fig01_codesign_matrix.svg", "book/chapters/04-hierarchy/figures/fig03_codesign_matrix.svg")
    shutil.copyfile("book/chapters/01-boundary/figures/fig01_codesign_matrix.pdf", "book/chapters/04-hierarchy/figures/fig03_codesign_matrix.pdf")

def run_all():
    gen_fig03_agent_workflow()
    gen_fig03_great_tension()
    gen_fig03_modular_blueprint()
    gen_fig03_three_cadences()
    gen_fig03_codesign_matrix()

if __name__ == "__main__":
    run_all()
