"""
book/tools/figures/ch14.py
Figures for Chapter 14: Intervention (Taking Authority Away From the Machine).
Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_ch14_handshake_fsm():
    """
    Figure 1: Four-Phase Authority Handshake Protocol & Discrete State Arbiter FSM.
    Visualizes the 4-phase sequence pipeline (Request -> Intercept/Auth -> Commit/Blend -> Confirm/Active),
    the 5 discrete authority states with guards, timeouts, and fallback edges, plus the 3-tier arbitration stack.
    """
    W = 920
    H = 525
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">THE FOUR-PHASE AUTHORITY HANDSHAKE PROTOCOL &amp; DISCRETE STATE ARBITER</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Deterministic 1 kHz Nervous-System Handshake: Autonomous Propose ⟶ Hardware Intercept ⟶ Operator Acknowledge ⟶ Bumpless Ramp ⟶ Active Authority</text>')

    # -------------------------------------------------------------
    # TOP SECTION: 4-PHASE PROTOCOL PIPELINE (Y: 60 to 182)
    # -------------------------------------------------------------
    top_y = 60
    top_h = 120
    svg.append(f'<rect x="25" y="{top_y}" width="{W-50}" height="{top_h}" rx="8" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1.1"/>')
    svg.append(f'<text x="40" y="{top_y+16}" font-size="9.5" font-weight="700" fill="{NAVY}" letter-spacing="0.4px">FOUR-PHASE AUTHORITY HANDSHAKE PIPELINE (1 kHz SYNCHRONOUS BUS)</text>')

    phases = [
        ("PHASE 1 · REQUEST", "Initiation Trigger", [
            "Operator torque |τ| &gt; 4.0 N·m",
            "or Policy ODD boundary departure",
            "u_prop flagged with lease intent"
        ], BLUE, 35),
        ("PHASE 2 · INTERCEPT &amp; AUTH", "Hardware Verification", [
            "1 kHz MCU intercepts token",
            "HMAC-SHA256 &amp; Nonce verified",
            "Freshness: Δt_transit ≤ 100 ms"
        ], PURPLE, 252),
        ("PHASE 3 · COMMIT &amp; BLEND", "Bumpless Ramp", [
            "Atomic swap at 1 ms tick",
            "C¹ blend: α(t) over τ_blend",
            "Integrators pre-biased to x(t)"
        ], BRONZE, 470),
        ("PHASE 4 · CONFIRM &amp; ACTIVE", "Active Authority", [
            "Haptic confirmation alert",
            "Human in command (α = 1.0)",
            "Active lease (T_lease ≤ 100 ms)"
        ], TEAL, 688)
    ]

    pw = 196
    ph = 84
    py_card = top_y + 24
    for idx, (p_tag, p_sub, p_items, col, px) in enumerate(phases):
        svg.append(f'<rect x="{px}" y="{py_card}" width="{pw}" height="{ph}" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="{px}" y="{py_card}" width="{pw}" height="19" rx="6" fill="{col}" fill-opacity="0.12"/>')
        svg.append(f'<text x="{px+10}" y="{py_card+13}" font-size="8.5" font-weight="700" fill="{col}">{p_tag}</text>')
        svg.append(f'<text x="{px+10}" y="{py_card+31}" font-size="9" font-weight="700" fill="{INK}">{p_sub}</text>')
        for l_idx, l_txt in enumerate(p_items):
            svg.append(f'<text x="{px+10}" y="{py_card+45+l_idx*12}" font-size="7.5" fill="{SLATE}">• {l_txt}</text>')

        # Phase connector arrow
        if idx < 3:
            arr_x = px + pw + 2
            arr_y = py_card + ph/2
            svg.append(f'<line x1="{arr_x}" y1="{arr_y}" x2="{arr_x+16}" y2="{arr_y}" stroke="{BORDER_DARK}" stroke-width="2" marker-end="url(#arr-slate)"/>')

    # -------------------------------------------------------------
    # BOTTOM LEFT: 5 DISCRETE STATES & TRANSITION FSM (X: 25 to 580, Y: 194 to 510)
    # -------------------------------------------------------------
    fsm_x = 25
    fsm_y = 194
    fsm_w = 555
    fsm_h = 316
    svg.append(f'<rect x="{fsm_x}" y="{fsm_y}" width="{fsm_w}" height="{fsm_h}" rx="8" fill="{BG_WHITE}" stroke="{NAVY}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{fsm_x}" y="{fsm_y}" width="{fsm_w}" height="24" rx="8" fill="{NAVY}" fill-opacity="0.10"/>')
    svg.append(f'<text x="{fsm_x+14}" y="{fsm_y+16}" font-size="9.5" font-weight="700" fill="{NAVY}">DISCRETE AUTHORITY STATE MACHINE (S_t ∈ &#123;Auto, Pending, Blend, Manual, Fallback&#125;)</text>')

    # State Cards:
    # Row 1 (y = fsm_y + 36 = 230): S_auto (left), S_pending (middle), S_blend (right)
    # Row 2 (y = fsm_y + 200 = 394): S_fallback (left), S_manual (right)
    
    # State S_auto
    s1_x, s1_y, s1_w, s1_h = fsm_x + 12, fsm_y + 36, 140, 88
    svg.append(f'<rect x="{s1_x}" y="{s1_y}" width="{s1_w}" height="{s1_h}" rx="6" fill="{BLUE}" fill-opacity="0.04" stroke="{BLUE}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{s1_x}" y="{s1_y}" width="{s1_w}" height="18" rx="6" fill="{BLUE}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s1_x+8}" y="{s1_y+13}" font-size="8.5" font-weight="700" fill="{BLUE}">S_auto · AUTONOMOUS</text>')
    svg.append(f'<text x="{s1_x+8}" y="{s1_y+30}" font-size="8" font-weight="600" fill="{INK}">Authority: Policy (α = 0.0)</text>')
    svg.append(f'<text x="{s1_x+8}" y="{s1_y+44}" font-size="7.5" fill="{SLATE}">• Action chunks p_t</text>')
    svg.append(f'<text x="{s1_x+8}" y="{s1_y+58}" font-size="7.5" fill="{SLATE}">• 1 kHz Enforcer bounds</text>')
    svg.append(f'<text x="{s1_x+8}" y="{s1_y+72}" font-size="7.5" fill="{SLATE}">• State tracking active</text>')

    # State S_pending
    s2_x, s2_y, s2_w, s2_h = fsm_x + 207, fsm_y + 36, 140, 88
    svg.append(f'<rect x="{s2_x}" y="{s2_y}" width="{s2_w}" height="{s2_h}" rx="6" fill="{AMBER}" fill-opacity="0.05" stroke="{AMBER}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{s2_x}" y="{s2_y}" width="{s2_w}" height="18" rx="6" fill="{AMBER}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s2_x+8}" y="{s2_y+13}" font-size="8.5" font-weight="700" fill="{AMBER}">S_pending · PENDING</text>')
    svg.append(f'<text x="{s2_x+8}" y="{s2_y+30}" font-size="8" font-weight="600" fill="{INK}">Authority: Intercept</text>')
    svg.append(f'<text x="{s2_x+8}" y="{s2_y+44}" font-size="7.5" fill="{SLATE}">• Timer T_timeout armed</text>')
    svg.append(f'<text x="{s2_x+8}" y="{s2_y+58}" font-size="7.5" fill="{SLATE}">• Token &amp; nonce check</text>')
    svg.append(f'<text x="{s2_x+8}" y="{s2_y+72}" font-size="7.5" fill="{CORAL}">• Timeout: 500 ms limit</text>')

    # State S_blend
    s3_x, s3_y, s3_w, s3_h = fsm_x + 402, fsm_y + 36, 140, 88
    svg.append(f'<rect x="{s3_x}" y="{s3_y}" width="{s3_w}" height="{s3_h}" rx="6" fill="{BRONZE}" fill-opacity="0.05" stroke="{BRONZE}" stroke-width="1.2"/>')
    svg.append(f'<rect x="{s3_x}" y="{s3_y}" width="{s3_w}" height="18" rx="6" fill="{BRONZE}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s3_x+8}" y="{s3_y+13}" font-size="8.5" font-weight="700" fill="{BRONZE}">S_blend · BUMPLESS</text>')
    svg.append(f'<text x="{s3_x+8}" y="{s3_y+30}" font-size="8" font-weight="600" fill="{INK}">Authority: Dynamic Blend</text>')
    svg.append(f'<text x="{s3_x+8}" y="{s3_y+44}" font-size="7.5" fill="{SLATE}">• C¹ cubic schedule α(t)</text>')
    svg.append(f'<text x="{s3_x+8}" y="{s3_y+58}" font-size="7.5" fill="{SLATE}">• Window τ_blend ≥ 1.5|Δu|/j</text>')
    svg.append(f'<text x="{s3_x+8}" y="{s3_y+72}" font-size="7.5" fill="{SLATE}">• Jerk bounded: j ≤ j_max</text>')

    # State S_fallback (Bottom Left)
    s5_x, s5_y, s5_w, s5_h = fsm_x + 12, fsm_y + 195, 235, 106
    svg.append(f'<rect x="{s5_x}" y="{s5_y}" width="{s5_w}" height="{s5_h}" rx="6" fill="{CORAL}" fill-opacity="0.05" stroke="{CORAL}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{s5_x}" y="{s5_y}" width="{s5_w}" height="18" rx="6" fill="{CORAL}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s5_x+8}" y="{s5_y+13}" font-size="8.5" font-weight="700" fill="{CORAL}">S_fallback · DEGRADED FALLBACK</text>')
    svg.append(f'<text x="{s5_x+8}" y="{s5_y+30}" font-size="8" font-weight="600" fill="{INK}">Category 1 Controlled Deceleration</text>')
    svg.append(f'<text x="{s5_x+8}" y="{s5_y+45}" font-size="7.5" fill="{SLATE}">• Dynamic braking: a_brake = μ·g (6.87 m/s²)</text>')
    svg.append(f'<text x="{s5_x+8}" y="{s5_y+59}" font-size="7.5" fill="{SLATE}">• Monotonic lock: No upward transitions</text>')
    svg.append(f'<text x="{s5_x+8}" y="{s5_y+73}" font-size="7.5" fill="{SLATE}">• Forensic freeze: 10s pre/post trigger window</text>')
    svg.append(f'<text x="{s5_x+8}" y="{s5_y+87}" font-size="7.5" fill="{TEAL}">• Exit: Zero velocity (v = 0) + Signed Re-init</text>')

    # State S_manual (Bottom Right)
    s4_x, s4_y, s4_w, s4_h = fsm_x + 307, fsm_y + 195, 235, 106
    svg.append(f'<rect x="{s4_x}" y="{s4_y}" width="{s4_w}" height="{s4_h}" rx="6" fill="{TEAL}" fill-opacity="0.05" stroke="{TEAL}" stroke-width="1.3"/>')
    svg.append(f'<rect x="{s4_x}" y="{s4_y}" width="{s4_w}" height="18" rx="6" fill="{TEAL}" fill-opacity="0.15"/>')
    svg.append(f'<text x="{s4_x+8}" y="{s4_y+13}" font-size="8.5" font-weight="700" fill="{TEAL}">S_manual · ACTIVE HUMAN MANUAL</text>')
    svg.append(f'<text x="{s4_x+8}" y="{s4_y+30}" font-size="8" font-weight="600" fill="{INK}">Authority: Human Operator (α = 1.0)</text>')
    svg.append(f'<text x="{s4_x+8}" y="{s4_y+45}" font-size="7.5" fill="{SLATE}">• Direct steering &amp; torque dispatch</text>')
    svg.append(f'<text x="{s4_x+8}" y="{s4_y+59}" font-size="7.5" fill="{SLATE}">• Deadman keep-alive (T_lease ≤ 100 ms)</text>')
    svg.append(f'<text x="{s4_x+8}" y="{s4_y+73}" font-size="7.5" fill="{PETROL}">• Subordinate to Tier-1 Safety Barrier h(x) ≥ 0</text>')
    svg.append(f'<text x="{s4_x+8}" y="{s4_y+87}" font-size="7.5" fill="{MUTED}">• Shadow logging of background policy</text>')

    # TRANSITION ARROWS & PILL BADGES
    # S_auto -> S_pending (x1: 152+25=177, x2: 232)
    svg.append(f'<line x1="{s1_x+s1_w}" y1="{s1_y+44}" x2="{s2_x}" y2="{s2_y+44}" stroke="{AMBER}" stroke-width="1.6" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<rect x="{(s1_x+s1_w+s2_x)/2-24}" y="{s1_y+24}" width="48" height="15" rx="3" fill="{BG_WHITE}" stroke="{AMBER}" stroke-width="0.8"/>')
    svg.append(f'<text x="{(s1_x+s1_w+s2_x)/2}" y="{s1_y+35}" font-size="6.8" font-weight="700" fill="{AMBER}" text-anchor="middle">Override</text>')

    # S_pending -> S_blend (x1: 347+25=372, x2: 427)
    svg.append(f'<line x1="{s2_x+s2_w}" y1="{s2_y+44}" x2="{s3_x}" y2="{s3_y+44}" stroke="{BRONZE}" stroke-width="1.6" marker-end="url(#arr-bronze)"/>')
    svg.append(f'<rect x="{(s2_x+s2_w+s3_x)/2-26}" y="{s2_y+24}" width="52" height="15" rx="3" fill="{BG_WHITE}" stroke="{BRONZE}" stroke-width="0.8"/>')
    svg.append(f'<text x="{(s2_x+s2_w+s3_x)/2}" y="{s2_y+35}" font-size="6.8" font-weight="700" fill="{BRONZE}" text-anchor="middle">Ack &amp; Auth</text>')

    # S_blend -> S_manual
    svg.append(f'<line x1="{s3_x+s3_w/2}" y1="{s3_y+s3_h}" x2="{s4_x+s4_w/2+30}" y2="{s4_y}" stroke="{TEAL}" stroke-width="1.6" marker-end="url(#arr-teal)"/>')
    svg.append(f'<rect x="{s3_x+s3_w/2-60}" y="{s4_y-25}" width="115" height="15" rx="3" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{s3_x+s3_w/2-2}" y="{s4_y-14}" font-size="6.8" font-weight="700" fill="{TEAL}" text-anchor="middle">Ramp Complete (α = 1.0)</text>')

    # S_pending -> S_fallback (Timeout drop)
    svg.append(f'<line x1="{s2_x+s2_w/2}" y1="{s2_y+s2_h}" x2="{s5_x+s5_w/2+20}" y2="{s5_y}" stroke="{CORAL}" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')
    svg.append(f'<rect x="{s2_x-30}" y="{s2_y+s2_h+18}" width="145" height="15" rx="3" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{s2_x+42}" y="{s2_y+s2_h+29}" font-size="6.8" font-weight="700" fill="{CORAL}" text-anchor="middle">Timeout t &gt; T_timeout (500 ms)</text>')

    # S_manual -> S_fallback (Lease Loss / Inattention)
    svg.append(f'<line x1="{s4_x}" y1="{s4_y+53}" x2="{s5_x+s5_w}" y2="{s5_y+53}" stroke="{CORAL}" stroke-width="1.6" stroke-dasharray="3,3" marker-end="url(#arr-coral)"/>')
    svg.append(f'<rect x="{(s4_x+s5_x+s5_w)/2-35}" y="{s4_y+36}" width="70" height="34" rx="3" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{(s4_x+s5_x+s5_w)/2}" y="{s4_y+49}" font-size="6.5" font-weight="700" fill="{CORAL}" text-anchor="middle">Lease Lost</text>')
    svg.append(f'<text x="{(s4_x+s5_x+s5_w)/2}" y="{s4_y+61}" font-size="6.2" fill="{MUTED}" text-anchor="middle">&gt;100 ms loss</text>')

    # S_fallback -> S_auto (Safe Re-init Reset)
    svg.append(f'<path d="M {s5_x+25} {s5_y} C {s5_x+25} {s1_y+s1_h+25}, {s1_x+25} {s1_y+s1_h+25}, {s1_x+25} {s1_y+s1_h}" fill="none" stroke="{BLUE}" stroke-width="1.4" marker-end="url(#arr-blue)"/>')
    svg.append(f'<rect x="{s1_x-8}" y="{s1_y+s1_h+15}" width="76" height="14" rx="3" fill="{BG_WHITE}" stroke="{BLUE}" stroke-width="0.8"/>')
    svg.append(f'<text x="{s1_x+30}" y="{s1_y+s1_h+25}" font-size="6.5" font-weight="700" fill="{BLUE}" text-anchor="middle">v = 0 &amp; Reset</text>')

    # -------------------------------------------------------------
    # BOTTOM RIGHT: 3-TIER ARBITRATION STACK & EVIDENCE SCHEMA (X: 590 to 895, Y: 194 to 510)
    # -------------------------------------------------------------
    arb_x = 590
    arb_y = 194
    arb_w = 305
    arb_h = 316
    svg.append(f'<rect x="{arb_x}" y="{arb_y}" width="{arb_w}" height="{arb_h}" rx="8" fill="{BG_WHITE}" stroke="{PURPLE}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{arb_x}" y="{arb_y}" width="{arb_w}" height="24" rx="8" fill="{PURPLE}" fill-opacity="0.10"/>')
    svg.append(f'<text x="{arb_x+14}" y="{arb_y+16}" font-size="9.5" font-weight="700" fill="{PURPLE}">3-TIER ARBITRATION &amp; EVIDENCE AUDIT</text>')

    # 3-Tier Preemption Stack
    tiers = [
        ("TIER 1 · ABSOLUTE VETO (MCU REFLEX)", "1000 Hz Hard Real-Time Safety Enforcer", [
            "CBF Invariant: h(x) ≥ 0 · Current/thermal limits",
            "Zero dynamic malloc · Rejects unsafe commands"
        ], PETROL),
        ("TIER 2 · SUPERVISORY OVERRIDE", "Authenticated Human Input (α = 1.0)", [
            "Bounded deadman lease (T_lease ≤ 100 ms)",
            "Overrides learned policy setpoints unconditionally"
        ], TEAL),
        ("TIER 3 · PROPOSAL STREAM (STOCHASTIC)", "20–50 Hz Host MPU Action Chunks", [
            "Candidate action buffer u_prop (Diffusion / ACT)",
            "Admitted only when S_t = S_auto &amp; Tier 1 clear"
        ], BLUE)
    ]

    ty = arb_y + 32
    for t_num, t_title, t_desc, col in tiers:
        svg.append(f'<rect x="{arb_x+10}" y="{ty}" width="{arb_w-20}" height="52" rx="5" fill="{col}" fill-opacity="0.05" stroke="{col}" stroke-width="1"/>')
        svg.append(f'<rect x="{arb_x+10}" y="{ty}" width="4" height="52" rx="2" fill="{col}"/>')
        svg.append(f'<text x="{arb_x+20}" y="{ty+14}" font-size="8" font-weight="700" fill="{col}">{t_num}</text>')
        svg.append(f'<text x="{arb_x+20}" y="{ty+27}" font-size="7.5" font-weight="600" fill="{INK}">{t_title}</text>')
        svg.append(f'<text x="{arb_x+20}" y="{ty+39}" font-size="7" fill="{SLATE}">{t_desc[0]}</text>')
        svg.append(f'<text x="{arb_x+20}" y="{ty+48}" font-size="7" fill="{SLATE}">{t_desc[1]}</text>')
        ty += 57

    # 64-byte Evidence Schema Box
    ey = ty + 2
    eh = 86
    svg.append(f'<rect x="{arb_x+10}" y="{ey}" width="{arb_w-20}" height="{eh}" rx="5" fill="{BG_LIGHT}" stroke="{BORDER_DARK}" stroke-width="1"/>')
    svg.append(f'<text x="{arb_x+20}" y="{ey+15}" font-size="8.5" font-weight="700" fill="{INK}">64-BYTE FORENSIC EVIDENCE TUPLE</text>')
    svg.append(f'<text x="{arb_x+20}" y="{ey+28}" font-size="7.2" font-family="monospace" fill="{NAVY}">⟨t_PTP, S_auth, x_state, u_prop, u_enf, u_act, ID_op⟩</text>')
    
    schema_points = [
        "• IEEE 1588 PTP hardware timestamps (&lt;100 ns uncertainty)",
        "• SHA-256 HMAC hash chaining into write-once ring buffer",
        "• Deterministic logging: &lt;4 µs latency (zero dynamic malloc)"
    ]
    for idx, sp in enumerate(schema_points):
        svg.append(f'<text x="{arb_x+20}" y="{ey+43+idx*13}" font-size="7.2" fill="{SLATE}">{sp}</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/14-intervention/figures/fig14_authority_handshake_fsm.svg", "\n".join(svg))


def gen_ch14_bumpless_transfer_dynamics():
    """
    Figure 2: Mechanical Dynamics of Control Handover: Step Discontinuity vs. C¹ Bumpless Blending.
    High-precision comparative waveforms across the handover window tau_blend:
    Panel A: Naive Step Handover (Jerk spike, drivetrain oscillation, tire traction breakaway).
    Panel B: C¹ Cubic Bumpless Blending (Bounded jerk, smooth torque ramp, preserved adhesion, pre-biased integrators).
    """
    W = 920
    H = 505
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">MECHANICAL DYNAMICS OF CONTROL HANDOVER: STEP DISCONTINUITY VS. C¹ BUMPLESS BLENDING</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Eliminating Mechanical Jerk Shock, Drive Slip, and Integrator Windup via Dynamic Blending τ_blend ≥ 1.5|Δu| / j_allowable</text>')

    # -------------------------------------------------------------
    # LEFT PANEL: UNBLENDED STEP HANDOVER (X: 25 to 455, Y: 60 to 490)
    # -------------------------------------------------------------
    lx = 25
    ly = 60
    lw = 425
    lh = 430
    svg.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="8" fill="{BG_WHITE}" stroke="{CORAL}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="24" rx="8" fill="{CORAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{lx+lw/2}" y="{ly+16}" font-size="9.5" font-weight="700" fill="{CORAL}" text-anchor="middle">✕ UNBLENDED STEP HANDOVER (Naïve Hard Switch, τ_blend = 0 ms)</text>')

    # Waveform Canvas A (Left)
    wx = lx + 55
    ww = lw - 75
    
    # 4 Traces: Authority Factor alpha(t), Motor Torque u(t), Chassis Jerk j(t), Traction Slip s(t)
    trace_h = 68
    trace_gap = 14
    start_ty = ly + 36

    # Time markers
    t0_x = wx + 80  # Takeover instant t0 = 200 ms
    
    # Grid lines & t0 dashed vertical
    for i in range(4):
        ty = start_ty + i * (trace_h + trace_gap)
        svg.append(f'<rect x="{wx}" y="{ty}" width="{ww}" height="{trace_h}" fill="#0F172A" rx="4" stroke="{BORDER_DARK}" stroke-width="0.8"/>')
        # Sub-grid lines
        for gx in range(wx + 40, wx + ww, 40):
            svg.append(f'<line x1="{gx}" y1="{ty}" x2="{gx}" y2="{ty+trace_h}" stroke="#1E293B" stroke-width="0.7"/>')
        svg.append(f'<line x1="{wx}" y1="{ty+trace_h/2}" x2="{wx+ww}" y2="{ty+trace_h/2}" stroke="#1E293B" stroke-width="0.7"/>')
        # t0 marker
        svg.append(f'<line x1="{t0_x}" y1="{ty}" x2="{t0_x}" y2="{ty+trace_h}" stroke="{CORAL}" stroke-width="1.2" stroke-dasharray="2,2"/>')

    # Trace 1: Blending Factor alpha(t) - Step Jump
    ty1 = start_ty
    svg.append(f'<text x="{lx+10}" y="{ty1+26}" font-size="8" font-weight="700" fill="{AMBER}">α(t)</text>')
    svg.append(f'<text x="{lx+10}" y="{ty1+38}" font-size="6.5" fill="{MUTED}">Weight</text>')
    y_zero1 = ty1 + trace_h - 10
    y_one1 = ty1 + 14
    step_alpha = f"M {wx+10} {y_zero1} L {t0_x} {y_zero1} L {t0_x} {y_one1} L {wx+ww-10} {y_one1}"
    svg.append(f'<path d="{step_alpha}" fill="none" stroke="{AMBER}" stroke-width="2"/>')
    svg.append(f'<rect x="{t0_x+10}" y="{y_one1+4}" width="150" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{t0_x+15}" y="{y_one1+14}" font-size="7" font-weight="700" fill="{AMBER}">Step: 0.0 ⟶ 1.0 (1 clock cycle)</text>')

    # Trace 2: Motor Torque u(t) - Discontinuous Drop + 25 Hz Ringing
    ty2 = start_ty + trace_h + trace_gap
    svg.append(f'<text x="{lx+10}" y="{ty2+26}" font-size="8" font-weight="700" fill="{BLUE}">u(t)</text>')
    svg.append(f'<text x="{lx+10}" y="{ty2+38}" font-size="6.5" fill="{MUTED}">Torque</text>')
    y_mid2 = ty2 + trace_h/2
    y_init2 = y_mid2 - 16 # Nominal cruising torque (0 Nm)
    y_target2 = y_mid2 + 18 # Full braking (-25 Nm)
    # Ringing waveform after step
    ring_pts = [
        f"M {wx+10} {y_init2} L {t0_x} {y_init2}",
        f"L {t0_x} {y_target2+12}", # overshoot
        f"Q {t0_x+15} {y_target2-14}, {t0_x+30} {y_target2+8}",
        f"Q {t0_x+45} {y_target2-6}, {t0_x+60} {y_target2+4}",
        f"Q {t0_x+75} {y_target2-2}, {t0_x+90} {y_target2}",
        f"L {wx+ww-10} {y_target2}"
    ]
    svg.append(f'<path d="{" ".join(ring_pts)}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>')
    svg.append(f'<rect x="{t0_x+10}" y="{ty2+6}" width="160" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{t0_x+15}" y="{ty2+16}" font-size="7" font-weight="700" fill="{CORAL}">Δu = 25 N·m Step + Severe Ringing</text>')

    # Trace 3: Chassis Jerk j(t) = d3x/dt3 - Infinite Spike
    ty3 = start_ty + 2 * (trace_h + trace_gap)
    svg.append(f'<text x="{lx+10}" y="{ty3+26}" font-size="8" font-weight="700" fill="{CORAL}">j(t)</text>')
    svg.append(f'<text x="{lx+10}" y="{ty3+38}" font-size="6.5" fill="{MUTED}">Jerk</text>')
    y_zero3 = ty3 + trace_h - 12
    y_spike3 = ty3 + 8
    jerk_pts = f"M {wx+10} {y_zero3} L {t0_x-2} {y_zero3} L {t0_x} {y_spike3} L {t0_x+4} {y_zero3+4} L {t0_x+8} {y_zero3-6} L {t0_x+14} {y_zero3} L {wx+ww-10} {y_zero3}"
    svg.append(f'<path d="{jerk_pts}" fill="none" stroke="{CORAL}" stroke-width="2"/>')
    svg.append(f'<rect x="{t0_x+12}" y="{y_spike3+2}" width="165" height="24" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{t0_x+16}" y="{y_spike3+12}" font-size="7" font-weight="700" fill="{CORAL}">⚡ j_max &gt; 6000 m/s³ ≫ j_allowable</text>')
    svg.append(f'<text x="{t0_x+16}" y="{y_spike3+22}" font-size="6.5" fill="#FCA5A5">Driveline shock &amp; gear pin fracture</text>')

    # Trace 4: Tire Slip Ratio s(t) & Integrator Transient
    ty4 = start_ty + 3 * (trace_h + trace_gap)
    svg.append(f'<text x="{lx+10}" y="{ty4+26}" font-size="8" font-weight="700" fill="{CRIMSON}">s(t)</text>')
    svg.append(f'<text x="{lx+10}" y="{ty4+38}" font-size="6.5" fill="{MUTED}">Tire Slip</text>')
    y_zero4 = ty4 + trace_h - 12
    y_crit4 = ty4 + trace_h - 32
    svg.append(f'<line x1="{wx}" y1="{y_crit4}" x2="{wx+ww}" y2="{y_crit4}" stroke="{CORAL}" stroke-width="0.8" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{wx+ww-8}" y="{y_crit4-3}" font-size="6.5" fill="{CORAL}" text-anchor="end">Slip Limit s_crit = 0.15</text>')
    slip_pts = f"M {wx+10} {y_zero4} L {t0_x} {y_zero4} Q {t0_x+20} {ty4+10}, {t0_x+50} {y_crit4+4} Q {t0_x+90} {y_crit4-4}, {wx+ww-10} {y_zero4+4}"
    svg.append(f'<path d="{slip_pts}" fill="none" stroke="{CRIMSON}" stroke-width="1.8"/>')
    svg.append(f'<rect x="{t0_x+12}" y="{ty4+8}" width="150" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{t0_x+16}" y="{ty4+18}" font-size="7" font-weight="700" fill="{CRIMSON}">Traction Loss: Wheels Skid / Lock</text>')

    # Bottom summary for Left Panel
    svg.append(f'<rect x="{lx+12}" y="{ly+lh-34}" width="{lw-24}" height="24" rx="4" fill="{CORAL}" fill-opacity="0.08" stroke="{CORAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{lx+lw/2}" y="{ly+lh-19}" font-size="8" font-weight="700" fill="{CORAL}" text-anchor="middle">FAILURE: Infinite Jerk · Mechanical Gearbox Shock · Stopping Path +3.67 m</text>')

    # -------------------------------------------------------------
    # RIGHT PANEL: C¹ CUBIC BUMPLESS BLENDING (X: 470 to 900, Y: 60 to 490)
    # -------------------------------------------------------------
    rx = 470
    ry = 60
    rw = 425
    rh = 430
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="8" fill="{BG_WHITE}" stroke="{TEAL}" stroke-width="1.3" filter="url(#shadow)"/>')
    svg.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="24" rx="8" fill="{TEAL}" fill-opacity="0.12"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+16}" font-size="9.5" font-weight="700" fill="{TEAL}" text-anchor="middle">✓ C¹ CUBIC BUMPLESS BLEND (Pre-Biased, τ_blend = 750 ms)</text>')

    rwx = rx + 55
    rww = rw - 75
    rt0_x = rwx + 80   # t0 = 200 ms
    rt1_x = rwx + 220  # t0 + tau_blend = 950 ms (blend window = 140 px)

    # Grid lines & blend window vertical bounds
    for i in range(4):
        ty = start_ty + i * (trace_h + trace_gap)
        svg.append(f'<rect x="{rwx}" y="{ty}" width="{rww}" height="{trace_h}" fill="#0F172A" rx="4" stroke="{BORDER_DARK}" stroke-width="0.8"/>')
        # Sub-grid lines
        for gx in range(rwx + 40, rwx + rww, 40):
            svg.append(f'<line x1="{gx}" y1="{ty}" x2="{gx}" y2="{ty+trace_h}" stroke="#1E293B" stroke-width="0.7"/>')
        svg.append(f'<line x1="{rwx}" y1="{ty+trace_h/2}" x2="{rwx+rww}" y2="{ty+trace_h/2}" stroke="#1E293B" stroke-width="0.7"/>')
        # Blend window shading
        svg.append(f'<rect x="{rt0_x}" y="{ty}" width="{rt1_x-rt0_x}" height="{trace_h}" fill="{TEAL}" fill-opacity="0.06"/>')
        svg.append(f'<line x1="{rt0_x}" y1="{ty}" x2="{rt0_x}" y2="{ty+trace_h}" stroke="{TEAL}" stroke-width="1.2" stroke-dasharray="2,2"/>')
        svg.append(f'<line x1="{rt1_x}" y1="{ty}" x2="{rt1_x}" y2="{ty+trace_h}" stroke="{TEAL}" stroke-width="1.2" stroke-dasharray="2,2"/>')

    # Trace 1: Blending Factor alpha(t) - C1 Cubic Smoothstep
    svg.append(f'<text x="{rx+10}" y="{ty1+26}" font-size="8" font-weight="700" fill="{TEAL}">α(t)</text>')
    svg.append(f'<text x="{rx+10}" y="{ty1+38}" font-size="6.5" fill="{MUTED}">Weight</text>')
    cubic_alpha = (
        f"M {rwx+10} {y_zero1} L {rt0_x} {y_zero1} "
        f"C {rt0_x+50} {y_zero1}, {rt1_x-50} {y_one1}, {rt1_x} {y_one1} "
        f"L {rwx+rww-10} {y_one1}"
    )
    svg.append(f'<path d="{cubic_alpha}" fill="none" stroke="{TEAL}" stroke-width="2.2"/>')
    svg.append(f'<rect x="{rwx+10}" y="{ty1+6}" width="165" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{rwx+15}" y="{ty1+16}" font-size="7" font-weight="700" fill="{TEAL}">C¹ Cubic: α(t) = 3(t/τ)² - 2(t/τ)³</text>')

    # Trace 2: Motor Torque u(t) - Smooth Blended S-curve
    svg.append(f'<text x="{rx+10}" y="{ty2+26}" font-size="8" font-weight="700" fill="{PETROL}">u(t)</text>')
    svg.append(f'<text x="{rx+10}" y="{ty2+38}" font-size="6.5" fill="{MUTED}">Torque</text>')
    smooth_u = (
        f"M {rwx+10} {y_init2} L {rt0_x} {y_init2} "
        f"C {rt0_x+50} {y_init2}, {rt1_x-50} {y_target2}, {rt1_x} {y_target2} "
        f"L {rwx+rww-10} {y_target2}"
    )
    svg.append(f'<path d="{smooth_u}" fill="none" stroke="{PETROL}" stroke-width="2.2"/>')
    svg.append(f'<rect x="{rwx+10}" y="{ty2+6}" width="180" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{rwx+15}" y="{ty2+16}" font-size="7" font-weight="700" fill="{PETROL}">Sigmoidal Torque Ramp (Zero Ringing)</text>')

    # Trace 3: Chassis Jerk j(t) - Smooth Parabolic Bell Curve
    svg.append(f'<text x="{rx+10}" y="{ty3+26}" font-size="8" font-weight="700" fill="{TEAL}">j(t)</text>')
    svg.append(f'<text x="{rx+10}" y="{ty3+38}" font-size="6.5" fill="{MUTED}">Jerk</text>')
    r_mid_x = (rt0_x + rt1_x) / 2
    y_bell_peak = ty3 + 24
    svg.append(f'<line x1="{rwx}" y1="{y_bell_peak-6}" x2="{rwx+rww}" y2="{y_bell_peak-6}" stroke="{AMBER}" stroke-width="0.8" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{rwx+rww-8}" y="{y_bell_peak-9}" font-size="6.5" fill="{AMBER}" text-anchor="end">Limit: j_allowable = 12 m/s³</text>')
    jerk_bell = (
        f"M {rwx+10} {y_zero3} L {rt0_x} {y_zero3} "
        f"Q {rt0_x+20} {y_zero3}, {r_mid_x} {y_bell_peak} "
        f"Q {rt1_x-20} {y_zero3}, {rt1_x} {y_zero3} "
        f"L {rwx+rww-10} {y_zero3}"
    )
    svg.append(f'<path d="{jerk_bell}" fill="none" stroke="{TEAL}" stroke-width="2.2"/>')
    svg.append(f'<rect x="{rwx+10}" y="{ty3+6}" width="165" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{rwx+15}" y="{ty3+16}" font-size="7" font-weight="700" fill="{TEAL}">j_max = 1.5|Δu|/τ_blend ≤ 12.0 m/s³ ✓</text>')

    # Trace 4: Tire Slip Ratio s(t) & Pre-Biased Adhesion
    svg.append(f'<text x="{rx+10}" y="{ty4+26}" font-size="8" font-weight="700" fill="{TEAL}">s(t)</text>')
    svg.append(f'<text x="{rx+10}" y="{ty4+38}" font-size="6.5" fill="{MUTED}">Tire Slip</text>')
    svg.append(f'<line x1="{rwx}" y1="{y_crit4}" x2="{rwx+rww}" y2="{y_crit4}" stroke="{CORAL}" stroke-width="0.8" stroke-dasharray="3,2"/>')
    svg.append(f'<text x="{rwx+rww-8}" y="{y_crit4-3}" font-size="6.5" fill="{CORAL}" text-anchor="end">Slip Limit s_crit = 0.15</text>')
    smooth_slip = (
        f"M {rwx+10} {y_zero4} L {rt0_x} {y_zero4} "
        f"Q {r_mid_x} {y_zero4-12}, {rt1_x} {y_zero4-8} "
        f"L {rwx+rww-10} {y_zero4-8}"
    )
    svg.append(f'<path d="{smooth_slip}" fill="none" stroke="{TEAL}" stroke-width="2.2"/>')
    svg.append(f'<rect x="{rwx+10}" y="{ty4+6}" width="175" height="14" rx="3" fill="#1E293B" fill-opacity="0.9"/>')
    svg.append(f'<text x="{rwx+15}" y="{ty4+16}" font-size="7" font-weight="700" fill="{TEAL}">Linear Adhesion (s ≤ 0.04 ≪ s_crit)</text>')

    # Bottom summary for Right Panel
    svg.append(f'<rect x="{rx+12}" y="{ry+rh-34}" width="{rw-24}" height="24" rx="4" fill="{TEAL}" fill-opacity="0.08" stroke="{TEAL}" stroke-width="0.8"/>')
    svg.append(f'<text x="{rx+rw/2}" y="{ry+rh-19}" font-size="8" font-weight="700" fill="{TEAL}" text-anchor="middle">SUCCESS: Zero Shock · Continuous Acceleration · Integrators Pre-Biased to x(t)</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/14-intervention/figures/fig14_bumpless_transfer_dynamics.svg", "\n".join(svg))


def run_all():
    gen_ch14_handshake_fsm()
    gen_ch14_bumpless_transfer_dynamics()

if __name__ == "__main__":
    run_all()
