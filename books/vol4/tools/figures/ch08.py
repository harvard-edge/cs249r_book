"""
book/tools/figures/ch08.py
Figures for Chapter 8: Action Generation & Trajectory Planning.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig07_three_speeds_latency():
    # Reuse from ch03_04 three cadences
    from .ch03_04 import gen_fig03_three_cadences
    gen_fig03_three_cadences()
    import shutil
    shutil.copyfile("book/chapters/03-cognition/figures/fig03_three_cadences.svg", "book/chapters/08-planning/figures/fig07_three_speeds_latency.svg")
    shutil.copyfile("book/chapters/03-cognition/figures/fig03_three_cadences.pdf", "book/chapters/08-planning/figures/fig07_three_speeds_latency.pdf")

def gen_fig07_action_chunking_ensembling():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">ACTION CHUNKING &amp; TEMPORAL ENSEMBLING</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Emitting Horizon H=8 Action Blocks with Exponential Overlap Weighting for Smooth Execution</text>')

    # Chunk 1 at t=0
    c1_y = 80
    svg.append(f'<text x="60" y="{c1_y+16}" font-size="10" font-weight="700" fill="{BLUE}">Chunk 1 (t = 0 ms):</text>')
    c1_tokens = ["a₀", "a₁", "a₂", "a₃", "a₄", "a₅", "a₆", "a₇"]
    for i, tok in enumerate(c1_tokens):
        tx = 200 + i * 52
        svg.append(f'<rect x="{tx}" y="{c1_y}" width="44" height="24" rx="4" fill="{BLUE}" fill-opacity="0.15" stroke="{BLUE}" stroke-width="1.2"/>')
        svg.append(f'<text x="{tx+22}" y="{c1_y+16}" font-size="9" font-weight="700" fill="{BLUE}" text-anchor="middle">{tok}</text>')

    # Chunk 2 at t=50ms (shifted by 4 steps)
    c2_y = 135
    svg.append(f'<text x="60" y="{c2_y+16}" font-size="10" font-weight="700" fill="{BRONZE}">Chunk 2 (t = 50 ms):</text>')
    c2_tokens = ["a₀'", "a₁'", "a₂'", "a₃'", "a₄'", "a₅'", "a₆'", "a₇'"]
    for i, tok in enumerate(c2_tokens):
        tx = 200 + (i + 3) * 52
        svg.append(f'<rect x="{tx}" y="{c2_y}" width="44" height="24" rx="4" fill="{BRONZE}" fill-opacity="0.15" stroke="{BRONZE}" stroke-width="1.2"/>')
        svg.append(f'<text x="{tx+22}" y="{c2_y+16}" font-size="9" font-weight="700" fill="{BRONZE}" text-anchor="middle">{tok}</text>')

    # Overlap Region Box
    ox = 200 + 3 * 52 - 4
    ow = 5 * 52 - 4
    svg.append(f'<rect x="{ox}" y="68" width="{ow}" height="102" rx="6" fill="{TEAL}" fill-opacity="0.06" stroke="{TEAL}" stroke-dasharray="4,2" stroke-width="1.3"/>')
    svg.append(f'<rect x="{ox+ow/2-90}" y="180" width="180" height="22" rx="4" fill="{TEAL}"/>')
    svg.append(f'<text x="{ox+ow/2}" y="194" font-size="8.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">TEMPORAL OVERLAP BLEND</text>')

    # Ensembled Output Trajectory
    ey = 224
    svg.append(f'<text x="60" y="{ey+18}" font-size="10" font-weight="700" fill="{PETROL}">Ensembled u_t:</text>')
    ens_tokens = ["u₀", "u₁", "u₂", "u₃", "u₄", "u₅", "u₆", "u₇", "u₈", "u₉", "u₁₀"]
    for i, tok in enumerate(ens_tokens):
        tx = 200 + i * 52
        col = TEAL if (3 <= i <= 7) else PETROL
        svg.append(f'<rect x="{tx}" y="{ey}" width="44" height="26" rx="4" fill="{col}" stroke="{col}" stroke-width="1.3" filter="url(#shadow)"/>')
        svg.append(f'<text x="{tx+22}" y="{ey+17}" font-size="9.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">{tok}</text>')

    # Math Card
    my = 276
    svg.append(f'<rect x="60" y="{my}" width="760" height="54" rx="6" fill="{BG_WHITE}" stroke="{BORDER}" stroke-width="1" filter="url(#shadow)"/>')
    svg.append(f'<text x="80" y="{my+22}" font-size="10" font-weight="700" fill="{NAVY}">MATHEMATICAL FORMULATION: u_t = ∑ w_i · a_t^(i) / ∑ w_i  where  w_i = exp(-m · (t - t_start))</text>')
    svg.append(f'<text x="80" y="{my+40}" font-size="8.5" fill="{SLATE}">Recent predictions carry higher exponential weights; older tail steps decay gracefully to eliminate joint torque jerks.</text>')

    # Bottom Invariant
    svg.append(f'<rect x="60" y="348" width="760" height="42" rx="5" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="366" font-size="9" font-weight="700" fill="{NAVY}" text-anchor="middle">THE ACTION CHUNKING PRINCIPLE</text>')
    svg.append(f'<text x="{W/2}" y="380" font-size="8.5" fill="{SLATE}" text-anchor="middle">Chunking amortizes high inference latency over future time steps; temporal ensembling eliminates inter-chunk torque discontinuities.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/08-planning/figures/fig07_action_chunking_ensembling.svg", "\n".join(svg))

def gen_fig07_c2_jerk_continuity():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">TRAJECTORY SMOOTHNESS: C⁰ STEP JUMP VS C² QUINTIC CONTINUITY</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Bounding Jerk j(t) = d³q/dt³ to Prevent Joint Mechanical Shock and Resonant Vibration</text>')

    # Left: C0 Discontinuous
    lx = 30
    lw = 380
    svg.append(f'<rect x="{lx}" y="70" width="{lw}" height="320" rx="8" fill="{CORAL}" fill-opacity="0.04" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{lx+lw/2}" y="92" font-size="11" font-weight="700" fill="{CORAL}" text-anchor="middle">✕ C⁰ ZERO-ORDER HOLD (Step Jumps)</text>')
    svg.append(f'<text x="{lx+lw/2}" y="108" font-size="9" fill="{MUTED}" text-anchor="middle">Direct execution of raw neural waypoints</text>')

    # Plots
    px = lx + 40
    py = 200
    svg.append(f'<polyline points="{px},220 {px+80},220 {px+80},160 {px+160},160 {px+160},180 {px+240},180 {px+240},130 {px+300},130" fill="none" stroke="{CORAL}" stroke-width="2.5"/>')
    svg.append(f'<text x="{px+80}" y="150" font-size="8" font-weight="700" fill="{CORAL}">Δv = step jump</text>')
    svg.append(f'<text x="{px+240}" y="120" font-size="8" font-weight="700" fill="{CORAL}">Jerk j → ∞</text>')

    svg.append(f'<text x="{lx+20}" y="270" font-size="8.5" fill="{SLATE}">• Infinite theoretical acceleration at boundaries</text>')
    svg.append(f'<text x="{lx+20}" y="286" font-size="8.5" fill="{SLATE}">• Gearbox tooth backlash impact &amp; wear</text>')
    svg.append(f'<text x="{lx+20}" y="302" font-size="8.5" fill="{SLATE}">• Excites high-frequency structural resonances</text>')
    svg.append(f'<text x="{lx+20}" y="318" font-size="8.5" font-weight="700" fill="{CORAL}">• Unacceptable in physical robots</text>')

    # Right: C2 Quintic Spline
    rx = 470
    rw = 380
    svg.append(f'<rect x="{rx}" y="70" width="{rw}" height="320" rx="8" fill="{TEAL}" fill-opacity="0.04" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{rx+rw/2}" y="92" font-size="11" font-weight="700" fill="{TEAL}" text-anchor="middle">✓ C² QUINTIC POLYNOMIAL SPLINE</text>')
    svg.append(f'<text x="{rx+rw/2}" y="108" font-size="9" fill="{MUTED}" text-anchor="middle">Continuous position, velocity, and acceleration</text>')

    rpx = rx + 40
    smooth_d = f"M {rpx} 220 C {rpx+40} 220, {rpx+60} 160, {rpx+100} 160 C {rpx+140} 160, {rpx+150} 180, {rpx+180} 180 C {rpx+210} 180, {rpx+230} 130, {rpx+300} 130"
    svg.append(f'<path d="{smooth_d}" fill="none" stroke="{TEAL}" stroke-width="2.5"/>')
    svg.append(f'<text x="{rpx+100}" y="145" font-size="8" font-weight="700" fill="{TEAL}">Smooth C² Curve</text>')
    svg.append(f'<text x="{rpx+230}" y="125" font-size="8" font-weight="700" fill="{TEAL}">Jerk j(t) ≤ j_max</text>')

    svg.append(f'<text x="{rx+20}" y="270" font-size="8.5" fill="{SLATE}">• Zero velocity or acceleration discontinuities</text>')
    svg.append(f'<text x="{rx+20}" y="286" font-size="8.5" fill="{SLATE}">• Smooth motor current rise within inverter slew limits</text>')
    svg.append(f'<text x="{rx+20}" y="302" font-size="8.5" fill="{SLATE}">• Clean torque tracking with minimum heat generation</text>')
    svg.append(f'<text x="{rx+20}" y="318" font-size="8.5" font-weight="700" fill="{TEAL}">• Standard for physical deployment</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/08-planning/figures/fig07_c2_jerk_continuity.svg", "\n".join(svg))

def gen_fig07_reanchoring_handshake():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">ASYNCHRONOUS STATE RE-ANCHORING HANDSHAKE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Planning Ahead: Projecting State x̂_(t + Δt_infer) During Model Inference to Eliminate Lag Shock</text>')

    # Timeline diagram
    ax_y = 220
    t_start = 80
    t_w = 720
    svg.append(f'<line x1="{t_start}" y1="{ax_y}" x2="{t_start+t_w}" y2="{ax_y}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{t_start+t_w/2}" y="{ax_y+40}" font-size="10.5" font-weight="700" fill="{SLATE}" text-anchor="middle">Wall Clock Time t →</text>')

    # t0: Trigger
    svg.append(f'<line x1="{t_start+80}" y1="80" x2="{t_start+80}" y2="{ax_y}" stroke="{BLUE}" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{t_start+80}" cy="{ax_y}" r="5" fill="{BLUE}"/>')
    svg.append(f'<text x="{t_start+80}" y="75" font-size="8.5" font-weight="700" fill="{BLUE}" text-anchor="middle">1. Sample x_0 (t₀)</text>')

    # Inference Window
    inf_w = 260
    svg.append(f'<rect x="{t_start+80}" y="{ax_y-70}" width="{inf_w}" height="55" rx="4" fill="{BRONZE}" fill-opacity="0.12" stroke="{BRONZE}" stroke-width="1.2"/>')
    svg.append(f'<text x="{t_start+80+inf_w/2}" y="{ax_y-45}" font-size="9" font-weight="700" fill="{BRONZE}" text-anchor="middle">NPU Inference Window (Δt_infer = 40 ms)</text>')
    svg.append(f'<text x="{t_start+80+inf_w/2}" y="{ax_y-30}" font-size="8" fill="{SLATE}" text-anchor="middle">Model plans chunk starting from projected x̂_(t₀ + 40ms)</text>')

    # t1: Arrival
    svg.append(f'<line x1="{t_start+80+inf_w}" y1="80" x2="{t_start+80+inf_w}" y2="{ax_y}" stroke="{TEAL}" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{t_start+80+inf_w}" cy="{ax_y}" r="5" fill="{TEAL}"/>')
    svg.append(f'<text x="{t_start+80+inf_w}" y="75" font-size="8.5" font-weight="700" fill="{TEAL}" text-anchor="middle">2. Chunk Dispatched (t₁)</text>')

    # Execution Window
    svg.append(f'<rect x="{t_start+80+inf_w}" y="{ax_y-70}" width="300" height="55" rx="4" fill="{TEAL}" fill-opacity="0.12" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{t_start+80+inf_w+150}" y="{ax_y-45}" font-size="9" font-weight="700" fill="{TEAL}" text-anchor="middle">Seamless Execution (H = 16 Steps)</text>')
    svg.append(f'<text x="{t_start+80+inf_w+150}" y="{ax_y-30}" font-size="8" fill="{SLATE}" text-anchor="middle">Zero state mismatch because chunk was anchored ahead!</text>')

    # Bottom Invariant Card
    svg.append(f'<rect x="40" y="300" width="{W-80}" height="95" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="60" y="322" font-size="10" font-weight="700" fill="{NAVY}">THE RE-ANCHORING INVARIANT</text>')
    svg.append(f'<text x="60" y="342" font-size="8.5" fill="{SLATE}">1. Naïve planners anchor at x(t_0); by the time inference completes at t_1, the robot has moved to x(t_1), causing a violent positional snap.</text>')
    svg.append(f'<text x="60" y="358" font-size="8.5" fill="{SLATE}">2. Defensive Physical AI architectures project the robot forward using the real-time MCU state estimator: x̂(t_0 + Δt_infer).</text>')
    svg.append(f'<text x="60" y="374" font-size="8.5" fill="{SLATE}">3. The chunk arrives exactly when the robot enters the projected state, ensuring C² continuity.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/08-planning/figures/fig07_reanchoring_handshake.svg", "\n".join(svg))

def run_all():
    gen_fig07_three_speeds_latency()
    gen_fig07_action_chunking_ensembling()
    gen_fig07_c2_jerk_continuity()
    gen_fig07_reanchoring_handshake()

if __name__ == "__main__":
    run_all()
