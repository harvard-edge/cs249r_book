"""
book/tools/figures/ch07.py
Figures for Chapter 7: Intent & Semantic Reasoning.
"""

from .common import (
    NAVY, BLUE, PETROL, TEAL, BRONZE, AMBER, CRIMSON, CORAL, PURPLE,
    SLATE, MUTED, INK, BG_LIGHT, BG_WHITE, BORDER, BORDER_DARK,
    COMMON_STYLE, COMMON_DEFS, save_svg_and_pdf
)

def gen_fig06_vlm_grounding_pipeline():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">VISION-LANGUAGE INTENT GROUNDING PIPELINE</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Converting Natural Language Commands to 3D Affordance Bounding Primitives with Metric Leases</text>')

    stages = [
        ("1. NATURAL LANGUAGE PROMPT", 'User: "Pick up the red thermal vial from shelf B"', NAVY),
        ("2. MULTI-MODAL EMBEDDING", "Cross-attention over RGB image tokens + prompt tokens", BLUE),
        ("3. 3D SPATIAL GROUNDING", "Projects 2D attention peak into calibrated 3D world frame", BLUE),
        ("4. INTENT LEASE EMISSION", "Emits 3D Goal Primitive + Expiring Lease (t_expire = 2.0 s)", BRONZE),
        ("5. DOWNSTREAM EXECUTION", "Action chunk planner &amp; 1 kHz MCU safety filter", PETROL)
    ]

    cur_y = 70
    for tag, desc, col in stages:
        svg.append(f'<rect x="40" y="{cur_y}" width="{W-80}" height="54" rx="6" fill="{BG_WHITE}" stroke="{col}" stroke-width="1.2" filter="url(#shadow)"/>')
        svg.append(f'<rect x="40" y="{cur_y}" width="6" height="54" rx="3" fill="{col}"/>')
        svg.append(f'<text x="58" y="{cur_y+20}" font-size="9.5" font-weight="700" fill="{col}">{tag}</text>')
        svg.append(f'<text x="58" y="{cur_y+38}" font-size="9" font-weight="600" fill="{INK}">{desc}</text>')
        cur_y += 66

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/07-intent/figures/fig06_vlm_grounding_pipeline.svg", "\n".join(svg))

def gen_fig06_kinematic_reachability_filter():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">KINEMATIC REACHABILITY &amp; MANIPULABILITY FILTERING</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Filtering VLM Semantic Goals Against Yoshikawa Manipulability Index w = √(det(J(q) J(q)^T))</text>')

    # Left: Out-of-Reach / Singular Goal (Rejected)
    lx = 30
    lw = 380
    svg.append(f'<rect x="{lx}" y="70" width="{lw}" height="320" rx="8" fill="{CORAL}" fill-opacity="0.04" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{lx+lw/2}" y="92" font-size="11" font-weight="700" fill="{CORAL}" text-anchor="middle">✕ REJECTED: SINGULAR / OUT-OF-REACH</text>')
    svg.append(f'<text x="{lx+lw/2}" y="108" font-size="9" fill="{MUTED}" text-anchor="middle">Manipulability Index w &lt; 0.05 (Near Gimbal Lock)</text>')

    # Arm reaching at extreme limit
    svg.append(f'<line x1="{lx+70}" y1="240" x2="{lx+180}" y2="180" stroke="{SLATE}" stroke-width="6" stroke-linecap="round"/>')
    svg.append(f'<line x1="{lx+180}" y1="180" x2="{lx+285}" y2="140" stroke="{SLATE}" stroke-width="5" stroke-linecap="round"/>')
    svg.append(f'<circle cx="{lx+70}" cy="240" r="7" fill="{NAVY}"/>')
    svg.append(f'<circle cx="{lx+180}" cy="180" r="6" fill="{NAVY}"/>')
    svg.append(f'<circle cx="{lx+285}" cy="140" r="5" fill="{CORAL}"/>')

    svg.append(f'<ellipse cx="{lx+285}" cy="140" rx="35" ry="4" fill="{CORAL}" fill-opacity="0.3" stroke="{CORAL}" transform="rotate(-20 {lx+285} 140)"/>')
    svg.append(f'<text x="{lx+285}" y="120" font-size="8.5" font-weight="700" fill="{CORAL}" text-anchor="middle">Flattened Velocity Ellipsoid</text>')

    svg.append(f'<text x="{lx+20}" y="290" font-size="8.5" fill="{SLATE}">• Arm fully outstretched at joint limit</text>')
    svg.append(f'<text x="{lx+20}" y="306" font-size="8.5" fill="{SLATE}">• Inverse Kinematics Jacobian det(J) → 0</text>')
    svg.append(f'<text x="{lx+20}" y="322" font-size="8.5" fill="{SLATE}">• Joint velocities explode q̇ = J⁻¹ v → ∞</text>')
    svg.append(f'<text x="{lx+20}" y="338" font-size="8.5" font-weight="700" fill="{CORAL}">• Immediate Intent Rejection &amp; Back-Off</text>')

    # Right: High Manipulability Goal (Accepted)
    rx = 470
    rw = 380
    svg.append(f'<rect x="{rx}" y="70" width="{rw}" height="320" rx="8" fill="{TEAL}" fill-opacity="0.04" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{rx+rw/2}" y="92" font-size="11" font-weight="700" fill="{TEAL}" text-anchor="middle">✓ ACCEPTED: HIGH ISOTROPY GOAL</text>')
    svg.append(f'<text x="{rx+rw/2}" y="108" font-size="9" fill="{MUTED}" text-anchor="middle">Manipulability Index w &gt; 0.45 (Condition Number κ ≈ 1.2)</text>')

    # Arm in comfortable posture
    svg.append(f'<line x1="{rx+100}" y1="240" x2="{rx+190}" y2="170" stroke="{SLATE}" stroke-width="6" stroke-linecap="round"/>')
    svg.append(f'<line x1="{rx+190}" y1="170" x2="{rx+260}" y2="200" stroke="{SLATE}" stroke-width="5" stroke-linecap="round"/>')
    svg.append(f'<circle cx="{rx+100}" cy="240" r="7" fill="{NAVY}"/>')
    svg.append(f'<circle cx="{rx+190}" cy="170" r="6" fill="{NAVY}"/>')
    svg.append(f'<circle cx="{rx+260}" cy="200" r="5" fill="{TEAL}"/>')

    svg.append(f'<ellipse cx="{rx+260}" cy="200" rx="28" ry="24" fill="{TEAL}" fill-opacity="0.2" stroke="{TEAL}"/>')
    svg.append(f'<text x="{rx+260}" y="165" font-size="8.5" font-weight="700" fill="{TEAL}" text-anchor="middle">Isotropic Velocity Ellipsoid</text>')

    svg.append(f'<text x="{rx+20}" y="290" font-size="8.5" fill="{SLATE}">• Well within reachable dextrous workspace</text>')
    svg.append(f'<text x="{rx+20}" y="306" font-size="8.5" fill="{SLATE}">• Uniform control authority in all Cartesian directions</text>')
    svg.append(f'<text x="{rx+20}" y="322" font-size="8.5" fill="{SLATE}">• Bounded motor torques τ = J^T F_ext</text>')
    svg.append(f'<text x="{rx+20}" y="338" font-size="8.5" font-weight="700" fill="{TEAL}">• Intent Certified &amp; Dispatched to Planner</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/07-intent/figures/fig06_kinematic_reachability_filter.svg", "\n".join(svg))

def gen_fig06_expiring_intent_lease():
    W = 880
    H = 430
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%">']
    svg.append(COMMON_STYLE)
    svg.append(COMMON_DEFS)
    svg.append(f'<rect width="{W}" height="{H}" fill="{BG_WHITE}" rx="10" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="28" class="title">EXPIRING INTENT LEASE LIFECYCLE &amp; DEAD-MAN SWITCH</text>')
    svg.append(f'<text x="{W/2}" y="44" class="subtitle">Preventing Zombie Goal Execution: Every High-Level Intent Has a Finite Hardware Validity Window</text>')

    # Timeline Axis
    ax_y = 240
    t_start = 80
    t_w = 720
    svg.append(f'<line x1="{t_start}" y1="{ax_y}" x2="{t_start+t_w}" y2="{ax_y}" stroke="{SLATE}" stroke-width="1.2" marker-end="url(#arr-slate)"/>')
    svg.append(f'<text x="{t_start+t_w/2}" y="{ax_y+40}" font-size="10.5" font-weight="700" fill="{SLATE}" text-anchor="middle">Wall Clock Time t →</text>')

    # Lease Issuance (t_0)
    svg.append(f'<line x1="{t_start+60}" y1="80" x2="{t_start+60}" y2="{ax_y}" stroke="{BLUE}" stroke-width="2"/>')
    svg.append(f'<rect x="{t_start+5}" y="70" width="110" height="24" rx="4" fill="{BLUE}"/>')
    svg.append(f'<text x="{t_start+60}" y="86" font-size="8.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">Lease Issued (t₀)</text>')

    # Lease Active Zone
    svg.append(f'<rect x="{t_start+60}" y="{ax_y-60}" width="340" height="50" rx="4" fill="{TEAL}" fill-opacity="0.12" stroke="{TEAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{t_start+230}" y="{ax_y-30}" font-size="9.5" font-weight="700" fill="{TEAL}" text-anchor="middle">VALID EXECUTION WINDOW (t &lt; t_expire = t₀ + 1500 ms)</text>')

    # Expiration Marker (t_expire)
    exp_x = t_start + 400
    svg.append(f'<line x1="{exp_x}" y1="80" x2="{exp_x}" y2="{ax_y}" stroke="{CORAL}" stroke-width="2"/>')
    svg.append(f'<rect x="{exp_x-55}" y="70" width="110" height="24" rx="4" fill="{CORAL}"/>')
    svg.append(f'<text x="{exp_x}" y="86" font-size="8.5" font-weight="700" fill="#FFFFFF" text-anchor="middle">Lease Expired!</text>')

    # Dead Zone
    svg.append(f'<rect x="{exp_x}" y="{ax_y-60}" width="280" height="50" rx="4" fill="{CORAL}" fill-opacity="0.12" stroke="{CORAL}" stroke-width="1.2"/>')
    svg.append(f'<text x="{exp_x+140}" y="{ax_y-30}" font-size="9.5" font-weight="700" fill="{CORAL}" text-anchor="middle">INTENT REVOKED → Category 2 Hold</text>')

    # Invariant Card
    svg.append(f'<rect x="40" y="310" width="{W-80}" height="90" rx="6" fill="{BG_LIGHT}" stroke="{BORDER}" stroke-width="1"/>')
    svg.append(f'<text x="60" y="332" font-size="10" font-weight="700" fill="{NAVY}">THE INTENT LEASE CONTRACT</text>')
    svg.append(f'<text x="60" y="352" font-size="8.5" fill="{SLATE}">1. High-level reasoning models must never issue permanent goals; all intents carry an explicit monotonic timestamp deadline.</text>')
    svg.append(f'<text x="60" y="368" font-size="8.5" fill="{SLATE}">2. If the VLM crashes or hangs, the real-time MCU safety layer detects lease expiry and brings the robot to a controlled stop.</text>')
    svg.append(f'<text x="60" y="384" font-size="8.5" fill="{SLATE}">3. The agent never moves on stale intent without an active renewal heartbeat.</text>')

    svg.append('</svg>')
    save_svg_and_pdf("book/chapters/07-intent/figures/fig06_expiring_intent_lease.svg", "\n".join(svg))

def run_all():
    gen_fig06_vlm_grounding_pipeline()
    gen_fig06_kinematic_reachability_filter()
    gen_fig06_expiring_intent_lease()

if __name__ == "__main__":
    run_all()
