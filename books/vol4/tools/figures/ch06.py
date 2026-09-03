"""
book/tools/figures/ch06.py
Figures for Chapter 6: Temporal Memory & World Models.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig05_se3_frame_tree():
    W = 880
    H = 440
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">DYNAMIC SE(3) KINEMATIC COORDINATE FRAME TREE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Propagating Homogeneous Transformations T_A^B ∈ SE(3) with Proprioceptive Innovation Updates</text>')

    nodes = [
        ("World Frame {W}", "Global Inertial Anchor", NAVY, 440, 75),
        ("Robot Base {B}", "T_W^B (Odometry / VIO)", BLUE, 300, 170),
        ("Static Obstacle {O}", "T_W^O (Map Anchor)", SLATE, 580, 170),
        ("Camera Sensor {C}", "T_B^C (Extrinsic Calibration)", BRONZE, 180, 275),
        ("End-Effector {EE}", "T_B^EE (Forward Kinematics)", PETROL, 420, 275),
        ("Target Object {T}", "T_C^T (Visual Spatial Token)", TEAL, 180, 365),
        ("Grasp Affordance {G}", "T_T^G (Metric Contact Normal)", CRIMSON, 420, 365)
    ]

    nw = 160
    nh = 44
    for tag, sub, col, nx, ny in nodes:
        svg.append(f'<rect x="{nx-nw/2}" y="{ny-nh/2}" width="{nw}" height="{nh}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.3" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{nx-nw/2}" y="{ny-nh/2}" width="4" height="{nh}" rx="2" fill="{col}"/>')
        svg.append(f'<text x="{nx}" y="{ny-4}" font-size="9.5" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{nx}" y="{ny+12}" font-size="8" fill="{SLATE}" text-anchor="middle">{sub}</text>')

    # Edges
    edges = [
        (440, 97, 300, 148, "{W} → {B}", BLUE),
        (440, 97, 580, 148, "{W} → {O}", SLATE),
        (300, 192, 180, 253, "{B} → {C}", BRONZE),
        (300, 192, 420, 253, "{B} → {EE}", PETROL),
        (180, 297, 180, 343, "{C} → {T}", TEAL),
        (420, 297, 420, 343, "{EE} → {G}", CRIMSON)
    ]
    for x1, y1, x2, y2, lbl, col in edges:
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="1.4" marker-end="url(#arr-blue)"/>')

    # Cross kinematic transform: Grasp relative to End-Effector (below the boxes)
    svg.append(f'<path d="M 180 395 C 260 435, 340 435, 418 395" fill="none" stroke="{PURPLE}" stroke-width="1.8" stroke-dasharray="4,2" marker-end="url(#arr-purple)"/>')
    svg.append(f'<text x="300" y="426" font-size="8.5" font-weight="700" fill="{PURPLE}" text-anchor="middle">T_EE^G = (T_W^EE)⁻¹ · T_W^B · T_B^C · T_C^T · T_T^G</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/06-state/figures/fig05_se3_frame_tree.svg", "\n".join(svg))

def gen_fig05_world_model_paradigms():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THREE WORLD MODEL PARADIGMS FOR EMBODIED INTELLIGENCE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Pixel Video Rollouts vs Latent Space Dynamics (JEPA) vs Structured Geometric State Trees</text>')

    paradigms = [
        ("PARADIGM 1: GENERATIVE VIDEO", "Pixel-Level Diffusion Rollouts", "Predicts raw RGB video frames x̂_t+k\nHigh compute (> 500 ms per step)\nHallucinates physics &amp; phantom obstacles\nUnusable for 1 kHz safety loops", CORAL),
        ("PARADIGM 2: LATENT WORLD MODEL", "Joint Embedding Predictive Architecture", "Predicts abstract latent representations ẑ_t+k\nFocuses on predictable dynamics, ignores noise\nFast feature prediction (15–30 ms)\nRequires spatial decoder for geometric bounds", BLUE),
        ("PARADIGM 3: STRUCTURED GEOMETRY", "Dynamic SE(3) State Estimation", "Predicts metric poses, velocities &amp; covariances\nDeterministic Kalman / Factor Graph update\nMicrosecond execution (≤ 100 µs on MCU)\nDirectly feeds CBF safety enforcers h(x) ≥ 0", TEAL)
    ]

    card_w = 250
    gap = 20
    start_x = (W - (3 * card_w + 2 * gap)) / 2

    for i, (tag, title, desc, col) in enumerate(paradigms):
        x = start_x + i * (card_w + gap)
        y = 70
        h = 330

        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{h}" rx="8" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.3" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="24" rx="8" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+16}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle">{tag}</text>')
        svg.append(f'<text x="{x+card_w/2}" y="{y+42}" font-size="11" font-weight="700" fill="{INK}" text-anchor="middle">{title}</text>')

        cur_y = y + 70
        for l in desc.split("\n"):
            svg.append(f'<text x="{x+12}" y="{cur_y}" font-size="8.5" fill="{SLATE}">• {l}</text>')
            cur_y += 24

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/06-state/figures/fig05_world_model_paradigms.svg", "\n".join(svg))

def gen_fig05_uncertainty_expansion():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">SPATIAL UNCERTAINTY EXPANSION UNDER SENSOR OCCLUSION</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Open-Loop Covariance Growth: Σ_(t+k) = F^k Σ_t (F^k)ᵀ + ∑ F^i Q (F^i)ᵀ</text>')

    # Timeline of Covariance Ellipses
    ax_y = 260
    t_start = 80
    t_w = 720
    svg.append(f'<line x1="{t_start}" y1="{ax_y}" x2="{t_start+t_w}" y2="{ax_y}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{t_start+t_w/2}" y="{ax_y+40}" font-size="10.5" font-weight="700" fill="{SLATE}" text-anchor="middle">Occlusion Horizon k (Steps without Camera Frame) →</text>')

    steps = [
        ("k = 0 (Observed)", 12, 18, TEAL, t_start+80),
        ("k = 3 (100 ms)", 24, 32, BLUE, t_start+240),
        ("k = 8 (250 ms)", 42, 54, AMBER, t_start+440),
        ("k = 15 (500 ms)", 70, 85, CORAL, t_start+640)
    ]

    for lbl, rx_e, ry_e, col, cx in steps:
        svg.append(f'<ellipse cx="{cx}" cy="{ax_y-80}" rx="{rx_e}" ry="{ry_e}" fill="{col}" fill-opacity="0.15" stroke="{col}" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{ax_y-80}" r="3" fill="{col}"/>')
        svg.append(f'<text x="{cx}" y="{ax_y-80-ry_e-10}" font-size="8.5" font-weight="700" fill="{col}" text-anchor="middle">{lbl}</text>')
        svg.append(f'<text x="{cx}" y="{ax_y+18}" font-size="8" fill="{MUTED}" text-anchor="middle">σ = {rx_e/10.0:.1f} cm</text>')

    # Invariant Rule Card
    svg.append(f'<rect x="40" y="320" width="{W-80}" height="80" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="60" y="342" font-size="10" font-weight="700" fill="{NAVY}">THE COVARIANCE EXPANSION INVARIANT</text>')
    svg.append(f'<text x="60" y="360" font-size="8.5" fill="{SLATE}">1. A world model must never treat dead-reckoned predictions as deterministic ground truth.</text>')
    svg.append(f'<text x="60" y="376" font-size="8.5" fill="{SLATE}">2. When covariance Σ_t exceeds the clearance margin (3σ &gt; d_clearance), the CBF safety enforcer must trigger an immediate velocity back-off.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/06-state/figures/fig05_uncertainty_expansion.svg", "\n".join(svg))

def gen_fig05_pcb_assembly_collision():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">FORENSIC CASE STUDY: THE STALE FRAME TREE COLLISION</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Autopsy of an Industrial Manipulator Joint Jam Caused by Asynchronous Coordinate Frame Desynchronization</text>')

    steps = [
        ("1. CAMERA FRAME ARRIVAL (t = 0 ms)", "Vision pipeline estimates PCB socket target pose T_C^S\nProcessed in user-space Linux background thread", NAVY),
        ("2. WORKCELL VIBRATION (t = 35 ms)", "Mechanical fixture shifts 4.2 mm due to pneumatic ejector\nFrame tree is not updated because camera frame is still rendering", AMBER),
        ("3. BLIND ACTION DISPATCH (t = 60 ms)", "Action planner emits trajectory chunk targeting stale pose\nManipulator moves at 1.2 m/s with zero compliance", BRONZE),
        ("4. PHYSICAL JAM &amp; OVER-TORQUE (t = 78 ms)", "Pin collides with fixture edge; motor current spikes to 18 A\nGearbox teeth stripped; emergency E-stop tripped", CORAL)
    ]

    for idx, (t, d, col) in enumerate(steps):
        by = 70 + idx * 75
        svg.append(f'<rect x="40" y="{by}" width="{W-80}" height="64" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="40" y="{by}" width="6" height="64" rx="3" fill="{col}"/>')
        svg.append(f'<text x="60" y="{by+22}" font-size="10" font-weight="700" fill="{col}">{t}</text>')
        for l_idx, l in enumerate(d.split("\n")):
            svg.append(f'<text x="60" y="{by+38+l_idx*15}" font-size="8.5" fill="{SLATE}">• {l}</text>')

    # Bottom Takeaway
    svg.append(f'<rect x="40" y="380" width="{W-80}" height="36" rx="5" fill="{CORAL}" fill-opacity="0.08" stroke="{CORAL}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="402" font-size="9" font-weight="700" fill="{CORAL}" text-anchor="middle">FORENSIC ROOT CAUSE: Un-timestamped coordinate frame transforms without proprioceptive compliance.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/06-state/figures/fig05_pcb_assembly_collision.svg", "\n".join(svg))

def run_all():
    gen_fig05_se3_frame_tree()
    gen_fig05_world_model_paradigms()
    gen_fig05_uncertainty_expansion()
    gen_fig05_pcb_assembly_collision()

if __name__ == "__main__":
    run_all()
