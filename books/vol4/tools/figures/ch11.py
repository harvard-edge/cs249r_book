"""
book/tools/figures/ch11.py
Figures for Chapter 11: Runtime Governance & Human Authority.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch11_authority_fsm():
    W = 880
    H = 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">RUNTIME AUTHORITY ARBITRATION STATE MACHINE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Deterministic Transitions between Autonomous Execution, Shared Authority, and Safe Fallback</text>')

    states = [
        ("STATE 1 · NOMINAL", "Fully Autonomous", "Policy has full control authority u_t = p_t\nh(x) ≥ 0 invariants certified by MCU", TEAL, 60, 90),
        ("STATE 2 · SHARED", "Shared Authority / Assist", "Torque blending u_t = (1-α)p_t + α u_human\nC² smoothstep transition S(α)", AMBER, 500, 90),
        ("STATE 3 · OVERRIDE", "Manual Teleoperation", "Human has 100% control authority\nPolicy actions logged in shadow mode", PURPLE, 500, 290),
        ("STATE 4 · SAFE FALLBACK", "Controlled Dynamic Stop", "Hardware watchdog / fault trigger\nCategory 1 deceleration to safe rest", CORAL, 60, 290)
    ]

    sw = 320
    sh = 110
    for tag, title, desc, col, x, y in states:
        svg.append(f'<rect x="{x}" y="{y}" width="{sw}" height="{sh}" rx="8" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.3" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{x}" y="{y}" width="{sw}" height="24" rx="8" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{x+14}" y="{y+16}" font-size="9" font-weight="700" fill="{col}">{tag}</text>')
        svg.append(f'<text x="{x+14}" y="{y+42}" font-size="11.5" font-weight="700" fill="{INK}">{title}</text>')
        for idx, line in enumerate(desc.split("\n")):
            svg.append(f'<text x="{x+14}" y="{y+62+idx*16}" font-size="9" fill="{SLATE}">• {line}</text>')

    svg.append(f'<line x1="380" y1="130" x2="500" y2="130" stroke="{AMBER}" stroke-width="1.4" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<text x="440" y="122" font-size="8" font-weight="600" fill="{AMBER}" text-anchor="middle">Operator Touch / Joy > 5%</text>')

    svg.append(f'<line x1="660" y1="200" x2="660" y2="290" stroke="{PURPLE}" stroke-width="1.4" marker-end="url(#arr-purple)"/>')
    svg.append(f'<text x="670" y="248" font-size="8" font-weight="600" fill="{PURPLE}">Full Displacement (α = 1.0)</text>')

    svg.append(f'<line x1="500" y1="345" x2="380" y2="345" stroke="{CORAL}" stroke-width="1.4" marker-end="url(#arr-coral)"/>')
    svg.append(f'<rect x="385" y="335" width="110" height="20" rx="3" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="440" y="348" font-size="7.5" font-weight="700" fill="{CORAL}" text-anchor="middle">E-Stop / Geofence</text>')

    svg.append(f'<line x1="210" y1="290" x2="210" y2="200" stroke="{TEAL}" stroke-width="1.4" marker-end="url(#arr-teal)"/>')
    svg.append(f'<text x="200" y="248" font-size="8" font-weight="600" fill="{TEAL}" text-anchor="end">Signed Release Handshake</text>')

    svg.append(f'<line x1="230" y1="200" x2="230" y2="290" stroke="{CORAL}" stroke-width="1.4" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')
    svg.append(f'<text x="240" y="248" font-size="8" font-weight="600" fill="{CORAL}">Fault / Watchdog (50 ms)</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/11-governance/figures/fig10_authority_state_machine.svg", "\n".join(svg))

def gen_ch11_bumpless_transfer():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">BUMPLESS TRANSFER: STEP DISCONTINUITY VS C² QUINTIC BLENDING</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Eliminating Mechanical Jerk Shock During Human Takeover via S(α) = 10α³ - 15α⁴ + 6α⁵</text>')

    lx = 30
    lw = 390
    svg.append(f'<rect x="{lx}" y="66" width="{lw}" height="330" rx="8" fill="{CORAL}" fill-opacity="0.04" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{lx+lw/2}" y="90" font-size="11.5" font-weight="700" fill="{CORAL}" text-anchor="middle">✕ STEP DISCONTINUITY (Naïve Hard Switch)</text>')
    svg.append(f'<text x="{lx+lw/2}" y="106" font-size="9" fill="{MUTED}" text-anchor="middle">Instantaneous authority handover u_t = u_human</text>')

    px = lx + 40
    py = 220
    svg.append(f'<line x1="{px}" y1="{py}" x2="{px+300}" y2="{py}" stroke="{SLATE}" stroke-width="1"/>')
    svg.append(f'<line x1="{px+150}" y1="{py-80}" x2="{px+150}" y2="{py+80}" stroke="{BORDER_DARK}" stroke-width="1" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{px+150}" y="{py+96}" font-size="8.5" fill="{MUTED}" text-anchor="middle">Takeover t₀</text>')

    svg.append(f'<line x1="{px}" y1="{py+30}" x2="{px+150}" y2="{py+30}" stroke="{BLUE}" stroke-width="2"/>')
    svg.append(f'<line x1="{px+150}" y1="{py+30}" x2="{px+150}" y2="{py-50}" stroke="{CORAL}" stroke-width="2.5"/>')
    svg.append(f'<line x1="{px+150}" y1="{py-50}" x2="{px+300}" y2="{py-50}" stroke="{PURPLE}" stroke-width="2"/>')
    svg.append(f'<text x="{px+160}" y="{py-10}" font-size="8.5" font-weight="700" fill="{CORAL}">⚡ JERK SPIKE (J → ∞)</text>')
    svg.append(f'<text x="{lx+lw/2}" y="350" font-size="9" fill="{SLATE}" text-anchor="middle">Gearbox tooth impact · Joint shock oscillation · Mechanical fatigue</text>')

    rx = 460
    rw = 390
    svg.append(f'<rect x="{rx}" y="66" width="{rw}" height="330" rx="8" fill="{TEAL}" fill-opacity="0.04" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{rx+rw/2}" y="90" font-size="11.5" font-weight="700" fill="{TEAL}" text-anchor="middle">✓ C² QUINTIC SMOOTHSTEP BLENDING</text>')
    svg.append(f'<text x="{rx+rw/2}" y="106" font-size="9" fill="{MUTED}" text-anchor="middle">u_t = (1 - S(α))·p_t + S(α)·u_human (Δt = 150 ms)</text>')

    rpx = rx + 40
    svg.append(f'<line x1="{rpx}" y1="{py}" x2="{rpx+300}" y2="{py}" stroke="{SLATE}" stroke-width="1"/>')
    svg.append(f'<line x1="{rpx+100}" y1="{py-80}" x2="{rpx+100}" y2="{py+80}" stroke="{BORDER_DARK}" stroke-width="1" stroke-dasharray="3,3"/>')
    svg.append(f'<line x1="{rpx+200}" y1="{py-80}" x2="{rpx+200}" y2="{py+80}" stroke="{BORDER_DARK}" stroke-width="1" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{rpx+100}" y="{py+96}" font-size="8.5" fill="{MUTED}" text-anchor="middle">t₀</text>')
    svg.append(f'<text x="{rpx+200}" y="{py+96}" font-size="8.5" fill="{MUTED}" text-anchor="middle">t₀ + 150 ms</text>')

    smooth_d = f"M {rpx} {py+30} L {rpx+100} {py+30} C {rpx+150} {py+30}, {rpx+150} {py-50}, {rpx+200} {py-50} L {rpx+300} {py-50}"
    svg.append(f'<path d="{smooth_d}" fill="none" stroke="{TEAL}" stroke-width="2.5"/>')
    svg.append(f'<rect x="{rpx+90}" y="{py-70}" width="140" height="20" rx="4" fill="{TEAL}" fill-opacity="0.1" stroke="{TEAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{rpx+160}" y="{py-56}" font-size="8.5" font-weight="700" fill="{TEAL}" text-anchor="middle">Bounded Jerk (j ≤ 15 rad/s³)</text>')
    svg.append(f'<text x="{rx+rw/2}" y="350" font-size="9" fill="{SLATE}" text-anchor="middle">Continuous acceleration · Zero contact bounce · Seamless operator feel</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/11-governance/figures/fig10_bumpless_transfer.svg", "\n".join(svg))

def gen_ch11_policy_flywheel():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE POLICY ENDOGENEITY TRAP &amp; ACTIVE DATA FLYWHEEL</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Preventing Distribution Collapse by Decoupling Autonomous Recovery from Intervention Demonstrations</text>')

    lx = 30
    lw = 390
    svg.append(f'<rect x="{lx}" y="66" width="{lw}" height="330" rx="8" fill="{CORAL}" fill-opacity="0.04" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{lx+lw/2}" y="90" font-size="11.5" font-weight="700" fill="{CORAL}" text-anchor="middle">✕ THE NAÏVE INTERVENTION TRAP</text>')
    svg.append(f'<text x="{lx+lw/2}" y="106" font-size="9" fill="{MUTED}" text-anchor="middle">Retraining directly on raw human takeovers</text>')

    steps_bad = [
        ("1. Policy drifts near ODD boundary", "Accumulating compounding error x_t ~ d_π"),
        ("2. Human takes over aggressively", "Discontinuous recovery action u_human"),
        ("3. Policy trained on takeover data", "Model learns that 'danger implies human assist'"),
        ("4. Autonomous recovery collapses", "Agent freezes or drives deeper into hazard")
    ]
    for idx, (t, d) in enumerate(steps_bad):
        by = 130 + idx * 56
        svg.append(f'<rect x="{lx+20}" y="{by}" width="{lw-40}" height="46" rx="5" fill="{BG_WHITE}" stroke="{CORAL}" stroke-opacity="0.3" stroke-width="1"/>')
        svg.append(f'<text x="{lx+30}" y="{by+18}" font-size="9.5" font-weight="700" fill="{CORAL}">{t}</text>')
        svg.append(f'<text x="{lx+30}" y="{by+34}" font-size="8.5" fill="{SLATE}">{d}</text>')
        if idx < 3:
            svg.append(f'<line x1="{lx+lw/2}" y1="{by+46}" x2="{lx+lw/2}" y2="{by+56}" stroke="{CORAL}" stroke-width="1.2" marker-end="url(#arr-coral)"/>')

    rx = 460
    rw = 390
    svg.append(f'<rect x="{rx}" y="66" width="{rw}" height="330" rx="8" fill="{TEAL}" fill-opacity="0.04" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{rx+rw/2}" y="90" font-size="11.5" font-weight="700" fill="{TEAL}" text-anchor="middle">✓ THE DEFENSIVE DATA FLYWHEEL</text>')
    svg.append(f'<text x="{rx+rw/2}" y="106" font-size="9" fill="{MUTED}" text-anchor="middle">DAgger with Truncated Demonstrations &amp; Counterfactuals</text>')

    steps_good = [
        ("1. State Visitation Distribution d_π", "Agent executes in world under policy π_θ"),
        ("2. Expert Labeling on Policy States", "Supervisor labels corrective action u*(s) for s ~ d_π"),
        ("3. Policy Lineage &amp; Hash Auditing", "Dataset indexed by commit hash &amp; seed"),
        ("4. Monotonic ODD Expansion", "Empirical safety margin improves across shadow fleet")
    ]
    for idx, (t, d) in enumerate(steps_good):
        by = 130 + idx * 56
        svg.append(f'<rect x="{rx+20}" y="{by}" width="{rw-40}" height="46" rx="5" fill="{BG_WHITE}" stroke="{TEAL}" stroke-opacity="0.3" stroke-width="1"/>')
        svg.append(f'<text x="{rx+30}" y="{by+18}" font-size="9.5" font-weight="700" fill="{TEAL}">{t}</text>')
        svg.append(f'<text x="{rx+30}" y="{by+34}" font-size="8.5" fill="{SLATE}">{d}</text>')
        if idx < 3:
            svg.append(f'<line x1="{rx+rw/2}" y1="{by+46}" x2="{rx+rw/2}" y2="{by+56}" stroke="{TEAL}" stroke-width="1.2" marker-end="url(#arr-teal)"/>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/11-governance/figures/fig10_policy_endogeneity_flywheel.svg", "\n".join(svg))

def run_all():
    gen_ch11_authority_fsm()
    gen_ch11_bumpless_transfer()
    gen_ch11_policy_flywheel()

if __name__ == "__main__":
    run_all()
