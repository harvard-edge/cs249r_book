import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-17: THE CONSTRAINTS NEVER LIE
#
# Volume II, Chapter 17 — Conclusion (Capstone)
# Core Invariant: Every invariant from both volumes reduces to one meta-principle:
#                 CONSTRAINTS DRIVE ARCHITECTURE.
#                 The interconnect wall forces parallelism strategies.
#                 The memory wall forces compression.
#                 Amdahl's Law bounds scaling.
#                 Little's Law bounds serving.
#                 Young-Daly bounds reliability.
#                 Chouldechova bounds fairness.
#                 None of these can be wished away — only navigated.
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Complete Systems Map (12-15 min)
#             Read ALL prior ledger entries (Vol1 ch1-16, Vol2 v2_01-v2_16).
#             Build a "Systems Intuition Report" showing prediction accuracy by domain.
#             Prediction: which category of invariants did you find most counterintuitive?
#             Radar chart across 8 dimensions.
#
#   Act II — Planet-Scale Architecture Challenge (20-25 min)
#             Scenario: Chief ML Architect for a planetary-scale AI system.
#             5B users, 1T parameter model, $10B budget, 2027 carbon-neutral,
#             99.99% availability, GDPR + CCAA DP requirements, 193 jurisdictions fairness.
#             Make 5 architectural decisions. 5 failure states.
#             Failure states: OOM (cluster too small), P99 SLO violation, DP ε > GDPR limit,
#             fairness gap > EU AI Act threshold, carbon over green grid capacity.
#
# Deployment Context: Full Fleet (all 4 tiers active)
#
# Design Ledger: saves chapter="v2_17"
#   Keys: context, cluster_gpus, parallelism_strategy, checkpoint_interval_min,
#         dp_epsilon, fairness_criterion, carbon_compliant, p99_slo_met,
#         total_system_cost_b, act1_prediction, act1_correct, act2_result,
#         act2_decision, constraint_hit, system_valid, invariants_connected
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─
@app.cell
def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from labs.core.state import DesignLedger
    from labs.core.style import COLORS, LAB_CSS, apply_plotly_theme

    ledger = DesignLedger()

    # ── Hardware constants — all tiers (sources documented inline) ────────────
    H100_BW_GBS       = 3350           # GB/s HBM3e; NVIDIA H100 SXM5 spec
    H100_TFLOPS_FP16  = 1979           # TFLOPS tensor-core FP16; NVIDIA spec
    H100_RAM_GB       = 80             # GB HBM3e; NVIDIA spec
    H100_TDP_W        = 700            # Watts TDP; NVIDIA spec
    H100_COST_USD     = 40_000         # $ purchase price; industry market rate 2024
    H100_CLOUD_HR     = 3.50           # $/GPU-hour cloud on-demand; AWS p4de rate
    H100_MTBF_HOURS   = 200            # hours per-GPU MTBF; @sec-fault-tolerance

    ORIN_BW_GBS       = 102            # GB/s; Jetson Orin NX 16GB spec
    ORIN_RAM_GB       = 16             # GB LPDDR5; Jetson Orin NX spec
    ORIN_TDP_W        = 25             # Watts TDP; Jetson Orin NX spec

    MOBILE_BW_GBS     = 68             # GB/s; Apple A17 class NPU spec
    MOBILE_RAM_GB     = 8              # GB; typical flagship smartphone
    MOBILE_TDP_W      = 5              # Watts sustained; mobile thermal envelope

    MCU_SRAM_KB       = 256            # KB; ARM Cortex-M7 on-chip SRAM ceiling
    MCU_BW_GBS        = 0.05           # GB/s; SRAM bandwidth on Cortex-M7

    IB_HDR200_BW_GBS  = 400            # GB/s InfiniBand HDR200; Mellanox spec
    NVLINK4_BW_GBS    = 900            # GB/s NVLink4 bidirectional; NVIDIA spec

    USERS_SCALE       = 5_000_000_000  # 5 billion users; planetary-scale target

    # ── Training physics constants ─────────────────────────────────────────────
    # 1T parameter model: FP16 weights + gradients + Adam optimizer = 20 bytes/param
    # Source: @sec-training-memory-anatomy
    BYTES_PER_PARAM_FULL = 20          # bytes; FP16 mixed-precision full training state
    BYTES_PER_PARAM_BF16 = 2           # bytes; BF16 inference weights only

    # ── Carbon constants ───────────────────────────────────────────────────────
    # EU average grid carbon intensity 2024; IEA grid data
    EU_GRID_CARBON_G_KWH  = 255        # g CO2/kWh; EU average 2024 (IEA)
    # Renewable PPA target (wind/solar) used by hyperscalers
    RENEW_CARBON_G_KWH    = 20         # g CO2/kWh; green PPA estimate
    # Carbon-neutral threshold as proxy: ≤ 50 g CO2/kWh effective average
    CARBON_THRESHOLD_G_KWH = 50        # g CO2/kWh; budget for "carbon-neutral by 2027"

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        H100_COST_USD, H100_CLOUD_HR, H100_MTBF_HOURS,
        ORIN_BW_GBS, ORIN_RAM_GB, ORIN_TDP_W,
        MOBILE_BW_GBS, MOBILE_RAM_GB, MOBILE_TDP_W,
        MCU_SRAM_KB, MCU_BW_GBS,
        IB_HDR200_BW_GBS, NVLINK4_BW_GBS,
        USERS_SCALE,
        BYTES_PER_PARAM_FULL, BYTES_PER_PARAM_BF16,
        EU_GRID_CARBON_G_KWH, RENEW_CARBON_G_KWH, CARBON_THRESHOLD_G_KWH,
    )


# ─── CELL 1: HEADER (hide_code=True) ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _c_cloud = COLORS["Cloud"]
    _c_green = COLORS["GreenLine"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #1a0a2e 100%);
                    padding: 40px 48px; border-radius: 16px; color: white;
                    box-shadow: 0 12px 48px rgba(0,0,0,0.45);
                    border: 1px solid rgba(99,102,241,0.2);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 17 · Capstone
            </div>
            <h1 style="margin: 0 0 12px 0; font-size: 2.6rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.05; letter-spacing: -0.03em;">
                The Constraints Never Lie
            </h1>
            <p style="margin: 0 0 24px 0; font-size: 1.08rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.7;">
                You have traversed 33 labs across two volumes. Every insight reduces
                to the same principle: <strong style="color:#a5b4fc;">constraints drive
                architecture.</strong> The memory wall, the interconnect wall, Amdahl's Law,
                Young-Daly, Little's Law, Chouldechova — none can be wished away.
                Your final task: architect a planet-scale AI system that must satisfy
                all of them simultaneously.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    Act I: Complete Systems Map &middot; Act II: Planet-Scale Architecture
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: All Vol 1 + Vol 2 chapters
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.3);">
                    5 Active Failure States
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                        margin-top: 8px;">
                <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: {_c_cloud}; font-weight: 700;">Cloud</span>
                    <div style="color: #94a3b8; margin-top: 2px;">H100 · 80 GB · 3.35 TB/s</div>
                </div>
                <div style="background: rgba(203,32,45,0.10); border: 1px solid rgba(203,32,45,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: #f87171; font-weight: 700;">Edge</span>
                    <div style="color: #94a3b8; margin-top: 2px;">Orin NX · 16 GB · 102 GB/s</div>
                </div>
                <div style="background: rgba(204,85,0,0.10); border: 1px solid rgba(204,85,0,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: #fb923c; font-weight: 700;">Mobile</span>
                    <div style="color: #94a3b8; margin-top: 2px;">NPU · 8 GB · 68 GB/s</div>
                </div>
                <div style="background: rgba(0,143,69,0.10); border: 1px solid rgba(0,143,69,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: #4ade80; font-weight: 700;">TinyML</span>
                    <div style="color: #94a3b8; margin-top: 2px;">Cortex-M7 · 256 KB · 0.05 GB/s</div>
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-conclusion-constraints-drive-architecture** — The meta-principle unifying
      all two-volume invariants; why physical laws cannot be abstracted away
    - **@sec-conclusion-vol1-synthesis** — Summary of the 8 invariant families from
      Volume I and how they compose at scale
    - **@sec-conclusion-vol2-synthesis** — Summary of distributed systems invariants;
      the emergent constraints that only appear at fleet scale
    - **@sec-conclusion-planet-scale** — Case study of hyperscaler architectural
      decisions viewed through the lens of competing constraints
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT I — THE COMPLETE SYSTEMS MAP
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT I: SECTION HEADER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Act I — The Complete Systems Map
    *Calibration · 12-15 minutes*
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["Cloud"]
    _bg    = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Chief Systems Architect
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "You have just completed 33 interactive labs spanning two volumes of ML systems
            content. Before you architect the planet-scale system in Act II, you need to
            understand your own intuition. Where did your mental models hold?
            Where did the physics surprise you? Your Design Ledger tells the story.
            Read it before you pick up the pen."
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT FRAMING ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The two-volume curriculum introduced eight families of physical invariants.
    Each family corresponds to a wall or ceiling that your architecture must navigate:

    | Domain | Core Invariant | Labs |
    |---|---|---|
    | **Memory** | Memory wall: bandwidth ≪ compute peak | V1: 05, 08, 10 |
    | **Compute** | Roofline / MFU ceiling | V1: 11, 12 |
    | **Serving** | Little's Law: N = λW; P99 ≠ average | V1: 13, V2: 10 |
    | **Networking** | AllReduce bandwidth wall; bisection BW | V2: 02, 03, 06 |
    | **Reliability** | Young-Daly: T\\* = sqrt(2C / λ) | V2: 07 |
    | **Scale** | Amdahl's ceiling; parallelism paradox | V2: 01, 05 |
    | **Ethics** | Chouldechova impossibility; DP ε-accuracy | V1: 15, 16; V2: 16 |
    | **Economics** | Jevons paradox; utilization vs. latency | V2: 08, 09, 15 |

    The radar chart below is your *Systems Intuition Report*. It reflects the eight
    domains where the curriculum tested your predictions. Before seeing the chart,
    commit to a hypothesis about your own blind spots.
    """)
    return


# ─── ACT I: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) Memory hierarchy effects — I underestimated how bandwidth-bound systems are": "A",
            "B) Communication overhead at scale — AllReduce and network costs surprised me most": "B",
            "C) Tail effects — P99 vs. average and cascade failures were the hardest to internalize": "C",
            "D) Fundamental impossibility theorems — Chouldechova, Amdahl ceilings felt unreachable": "D",
        },
        label="Which category of invariants did you find MOST counterintuitive across both volumes?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(act1_pred, mo):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act I Systems Map."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT I: LEDGER ARCHAEOLOGY ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Systems Intuition Report — Design Ledger Archaeology")
    return


@app.cell(hide_code=True)
def _(COLORS, go, ledger, mo, np, apply_plotly_theme):
    # ── Read all prior ledger entries ─────────────────────────────────────────
    _history = ledger._state.history if hasattr(ledger._state, "history") else []

    # ── Build a chapter→design map for all 33 labs ────────────────────────────
    _ledger_map = {}
    for _entry in _history:
        _ch = str(_entry.get("chapter", ""))
        _design = _entry.get("design", {})
        _ledger_map[_ch] = _design

    # ── Chapter membership per domain ─────────────────────────────────────────
    # Maps domain → list of chapter keys to look up in ledger
    _domain_chapters = {
        "Memory":      ["5", "8", "10"],
        "Compute":     ["11", "12"],
        "Serving":     ["13", "v2_10"],
        "Networking":  ["v2_02", "v2_03", "v2_06"],
        "Reliability": ["v2_07"],
        "Scale":       ["v2_01", "v2_05"],
        "Ethics":      ["15", "16", "v2_16"],
        "Economics":   ["v2_08", "v2_09", "v2_15"],
    }

    # ── Compute per-domain accuracy from ledger ────────────────────────────────
    # Each lab stores act1_correct: bool. Average across available labs in domain.
    _domain_accuracy = {}
    _domain_labs_done = {}
    _domain_constraints_hit = {}

    for _domain, _chapters in _domain_chapters.items():
        _correct_list = []
        _constraint_list = []
        for _ch in _chapters:
            if _ch in _ledger_map:
                _d = _ledger_map[_ch]
                if "act1_correct" in _d:
                    _correct_list.append(1.0 if _d["act1_correct"] else 0.0)
                if "constraint_hit" in _d:
                    _constraint_list.append(1.0 if _d["constraint_hit"] else 0.0)
        _domain_accuracy[_domain] = (
            sum(_correct_list) / len(_correct_list) if _correct_list else 0.5
        )
        _domain_labs_done[_domain] = len([c for c in _chapters if c in _ledger_map])
        _domain_constraints_hit[_domain] = (
            sum(_constraint_list) / len(_constraint_list) if _constraint_list else 0.0
        )

    # ── Summary statistics ────────────────────────────────────────────────────
    _total_labs = len(_history)
    _vol1_correct = [
        1.0 if _ledger_map.get(str(c), {}).get("act1_correct", False) else 0.0
        for c in range(1, 17) if str(c) in _ledger_map
    ]
    _vol2_correct = [
        1.0 if _ledger_map.get(f"v2_{c:02d}", {}).get("act1_correct", False) else 0.0
        for c in range(1, 17) if f"v2_{c:02d}" in _ledger_map
    ]
    _vol1_acc = sum(_vol1_correct) / max(len(_vol1_correct), 1) * 100
    _vol2_acc = sum(_vol2_correct) / max(len(_vol2_correct), 1) * 100
    _all_correct = _vol1_correct + _vol2_correct
    _overall_acc = sum(_all_correct) / max(len(_all_correct), 1) * 100

    # Weakest domain: lowest accuracy score
    _weakest_domain = min(_domain_accuracy, key=lambda d: _domain_accuracy[d])
    _strongest_domain = max(_domain_accuracy, key=lambda d: _domain_accuracy[d])

    # Most-triggered failure state domain
    _most_failures_domain = max(
        _domain_constraints_hit, key=lambda d: _domain_constraints_hit[d]
    )

    # ── Radar chart ───────────────────────────────────────────────────────────
    _domains = list(_domain_accuracy.keys())
    _scores = [_domain_accuracy[d] * 100 for d in _domains]
    _scores_closed = _scores + [_scores[0]]  # close the polygon
    _theta_closed  = _domains + [_domains[0]]

    _fig = go.Figure()

    # Ideal reference (100%)
    _fig.add_trace(go.Scatterpolar(
        r=[100] * (len(_domains) + 1),
        theta=_theta_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.06)",
        line=dict(color=COLORS["Cloud"], width=1, dash="dot"),
        name="Perfect (100%)",
    ))

    # Student scores
    _fig.add_trace(go.Scatterpolar(
        r=_scores_closed,
        theta=_theta_closed,
        fill="toself",
        fillcolor="rgba(0,143,69,0.12)",
        line=dict(color=COLORS["GreenLine"], width=2.5),
        name="Your accuracy",
        marker=dict(size=8, color=COLORS["GreenLine"]),
    ))

    _fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=["25%", "50%", "75%", "100%"],
                gridcolor=COLORS["Border"],
                tickfont=dict(size=9, color=COLORS["TextMuted"]),
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["TextSec"]),
                gridcolor=COLORS["Border"],
            ),
            bgcolor="rgba(248,250,252,0.6)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=440,
        margin=dict(t=40, b=60, l=50, r=50),
        title=dict(
            text="Systems Intuition Radar — Prediction Accuracy by Domain",
            font=dict(size=13, color=COLORS["Text"]),
            x=0.5,
        ),
    )
    apply_plotly_theme(_fig)

    # ── Summary metric cards ───────────────────────────────────────────────────
    _vol1_color  = COLORS["GreenLine"] if _vol1_acc >= 70 else (
        COLORS["OrangeLine"] if _vol1_acc >= 50 else COLORS["RedLine"]
    )
    _vol2_color  = COLORS["GreenLine"] if _vol2_acc >= 70 else (
        COLORS["OrangeLine"] if _vol2_acc >= 50 else COLORS["RedLine"]
    )
    _total_color = COLORS["GreenLine"] if _overall_acc >= 70 else (
        COLORS["OrangeLine"] if _overall_acc >= 50 else COLORS["RedLine"]
    )

    _summary_cards = mo.Html(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0;">
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Labs Completed
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['BlueLine']};">
                {_total_labs}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">of 33 total</div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Vol I Accuracy
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_vol1_color};">
                {_vol1_acc:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {len(_vol1_correct)} labs sampled
            </div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Vol II Accuracy
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_vol2_color};">
                {_vol2_acc:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {len(_vol2_correct)} labs sampled
            </div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Overall
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_total_color};">
                {_overall_acc:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">combined</div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 0;">
        <div class="lab-card" style="padding: 14px 16px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Weakest Domain
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['RedLine']};">
                {_weakest_domain}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {_domain_accuracy[_weakest_domain]*100:.0f}% accuracy
            </div>
        </div>
        <div class="lab-card" style="padding: 14px 16px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Strongest Domain
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['GreenLine']};">
                {_strongest_domain}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {_domain_accuracy[_strongest_domain]*100:.0f}% accuracy
            </div>
        </div>
        <div class="lab-card" style="padding: 14px 16px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Most Failure States
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['OrangeLine']};">
                {_most_failures_domain}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {_domain_constraints_hit[_most_failures_domain]*100:.0f}% of labs triggered
            </div>
        </div>
    </div>
    """)

    mo.vstack([_summary_cards, mo.ui.plotly(_fig)])
    return (
        _domain_accuracy,
        _domain_chapters,
        _domain_labs_done,
        _domain_constraints_hit,
        _weakest_domain,
        _strongest_domain,
        _most_failures_domain,
        _overall_acc,
        _vol1_acc,
        _vol2_acc,
        _total_labs,
        _ledger_map,
    )


# ─── ACT I: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    _domain_accuracy,
    _weakest_domain,
    act1_pred,
    mo,
):
    # Map prediction option to corresponding domain key
    _option_to_domain = {
        "A": "Memory",
        "B": "Networking",
        "C": "Serving",
        "D": "Ethics",
    }
    _predicted_domain = _option_to_domain.get(act1_pred.value, "Memory")
    _actual_weakest = _weakest_domain

    # Determine if prediction matched ledger-revealed weakness
    _matches = _predicted_domain == _actual_weakest

    # Generate feedback for all four options — each substantive, no single "correct"
    _feedback_map = {
        "A": (
            f"**Memory hierarchy effects.** "
            f"This is one of the most reliably counterintuitive invariants in all of computing. "
            f"The H100's peak compute (1,979 TFLOPS FP16) outpaces its memory bandwidth "
            f"(3,350 GB/s) by roughly 300 operations per byte. Most inference workloads "
            f"never reach that peak — they stall on bandwidth. "
            f"Your ledger shows your Memory domain accuracy at "
            f"{_domain_accuracy['Memory']*100:.0f}%. "
            f"If that surprised you, you're in good company: this is the wall that killed "
            f"CPU-only ML and drove the entire GPU ecosystem."
        ),
        "B": (
            f"**Communication overhead at scale.** "
            f"AllReduce over InfiniBand (400 GB/s) looks fast until your gradient tensor "
            f"is 280 GB (1T model × 2 bytes/param × 2× for BF16 accumulation). "
            f"At 16,384 GPUs with ring AllReduce, each of the 2(N-1)/N ≈ 2 passes "
            f"moves 280 GB across a fabric with shared bisection bandwidth. "
            f"Your Networking accuracy: {_domain_accuracy['Networking']*100:.0f}%. "
            f"The networking wall is the invariant that most surprises engineers who "
            f"trained single-node before moving to multi-node — the compute is ready; "
            f"the network is not."
        ),
        "C": (
            f"**Tail effects.** "
            f"P99 latency can be 10-100× the mean in a serving system under load. "
            f"Little's Law (N = λW) tells you the average, but it says nothing about "
            f"the tail. Cascade failures amplify: one slow node in a pipeline stage "
            f"causes timeout retries upstream, which increases load on that node, "
            f"which causes more timeouts. Your Serving accuracy: "
            f"{_domain_accuracy['Serving']*100:.0f}%. "
            f"The tail is the gap between your SLO contract and your monitoring dashboard."
        ),
        "D": (
            f"**Fundamental impossibility theorems.** "
            f"Chouldechova's theorem states that when base rates differ across groups, "
            f"no classifier can simultaneously equalize false positive rate, false negative "
            f"rate, and calibration. This is not an engineering challenge — it is a "
            f"mathematical constraint as immovable as Amdahl's Law. "
            f"Your Ethics accuracy: {_domain_accuracy['Ethics']*100:.0f}%. "
            f"The impossibility theorems are the domain where intuition fails most "
            f"catastrophically, because they look like they should be solvable with "
            f"more data or a better model. They are not."
        ),
    }

    _chosen_feedback = _feedback_map.get(act1_pred.value, _feedback_map["A"])

    _match_note = ""
    if _matches:
        _match_note = (
            f" Your ledger confirms this: **{_actual_weakest}** is your weakest domain "
            f"({_domain_accuracy[_actual_weakest]*100:.0f}% accuracy)."
        )
    else:
        _match_note = (
            f" Interestingly, your ledger shows your *actual* weakest domain is "
            f"**{_actual_weakest}** ({_domain_accuracy[_actual_weakest]*100:.0f}% accuracy) — "
            f"which means your intuition about your own intuition may also benefit from calibration."
        )

    _kind = "success" if _matches else "info"
    mo.callout(mo.md(_chosen_feedback + _match_note), kind=_kind)
    return


# ─── ACT I: MATH PEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — all eight invariant families": mo.md("""
        **Memory Wall (Roofline):**
        `Performance = min(peak_TFLOPS, BW_GBs × arithmetic_intensity)`
        — From @sec-hw-acceleration-roofline

        **Amdahl's Law (Scale Ceiling):**
        `Speedup(N) = 1 / ((1 - p) + p/N)` where p is the parallelizable fraction
        — Maximum speedup is bounded by serial fraction; source: @sec-distributed-training-amdahl

        **Young-Daly (Checkpoint Optimum):**
        `T* = sqrt(2 × C / λ)` where C = checkpoint cost, λ = cluster failure rate
        — Minimizes expected wasted time; source: @sec-fault-tolerance-young-daly

        **Little's Law (Serving Throughput):**
        `N = λ × W` where N = in-flight requests, λ = arrival rate, W = latency
        — Steady-state queueing identity; source: @sec-model-serving-littles-law

        **Differential Privacy (Accuracy-Privacy):**
        `ε ≥ Δf / σ` where Δf = sensitivity, σ = noise scale
        — Lower ε = stronger privacy; accuracy degrades as ε → 0; source: @sec-security-privacy-dp

        **Chouldechova Impossibility (Fairness):**
        When base rates differ: cannot simultaneously equalize FPR, FNR, and calibration
        — Source: @sec-responsible-ai-chouldechova

        **AllReduce Bandwidth (Ring):**
        `t_allreduce = 2 × (N-1)/N × M / BW` where M = gradient size, BW = fabric bandwidth
        — Source: @sec-collective-communication-ring-allreduce

        **Carbon-Aware Scheduling:**
        `CO2 = Energy_kWh × carbon_intensity_g_kWh`
        — Jevons Paradox: efficiency gains can be consumed by demand growth; source: @sec-sustainable-ai
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT II — PLANET-SCALE ARCHITECTURE CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT II: SECTION HEADER ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Act II — Planet-Scale Architecture Challenge
    *Design Challenge · 20-25 minutes*
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["OrangeLine"]
    _bg    = COLORS["OrangeL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 18px 24px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
            Incoming Message &middot; Board of Directors &middot; URGENT
        </div>
        <div style="font-style: italic; font-size: 1.02rem; color: #1e293b; line-height: 1.7;">
            "You have been appointed Chief ML Architect for a planetary-scale AI system.
            Requirements: serve <strong>5 billion users globally</strong> across cloud, edge,
            mobile, and TinyML tiers; train a <strong>1 trillion parameter foundation model</strong>
            with monthly updates; maintain <strong>P99 &lt; 500ms globally</strong> with
            <strong>99.99% availability</strong>; comply with GDPR (EU) and CCPA (California)
            differential privacy requirements; achieve <strong>carbon-neutral by 2027</strong>;
            ensure fair treatment across <strong>193 UN member countries</strong>.
            Your infrastructure budget is <strong>$10 billion</strong>.
            You have five architectural decisions to make. Every decision must satisfy a
            physical constraint. Some combinations are infeasible. Find one that is not."
        </div>
    </div>
    """)
    return


# ─── ACT II: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Training is the binding constraint — the 1T model OOMs on any realistic cluster": "A",
            "B) Serving is the binding constraint — P99 < 500ms at 5B users is physically unreachable": "B",
            "C) Privacy is the binding constraint — GDPR-grade DP destroys too much model accuracy": "C",
            "D) All constraints can be satisfied simultaneously with correct architectural choices": "D",
        },
        label="Before configuring the system: which constraint will be hardest to satisfy at 5B-user scale?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(act2_pred, mo):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the architecture instruments."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT II: DECISION 1 — TRAINING INFRASTRUCTURE ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Decision 1 — Training Infrastructure")
    return


@app.cell(hide_code=True)
def _(mo):
    d1_gpu_count = mo.ui.slider(
        start=1024, stop=65536, value=16384, step=1024,
        label="GPU cluster size (H100 count)",
        show_value=True,
    )
    d1_parallelism = mo.ui.dropdown(
        options={
            "Data Parallel only (DP)": "dp",
            "Tensor + Pipeline Parallel (TP+PP)": "tp_pp",
            "Full 3D Parallelism (DP+TP+PP)": "3d",
            "Expert Parallelism (MoE)": "moe",
        },
        value="Full 3D Parallelism (DP+TP+PP)",
        label="Parallelism strategy",
    )
    d1_mfu = mo.ui.slider(
        start=20, stop=60, value=40, step=5,
        label="Expected MFU % (Model FLOP Utilization)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([d1_gpu_count, d1_parallelism], justify="start", gap=4),
        d1_mfu,
    ])
    return (d1_gpu_count, d1_parallelism, d1_mfu)


@app.cell(hide_code=True)
def _(
    COLORS,
    BYTES_PER_PARAM_FULL,
    H100_CLOUD_HR,
    H100_RAM_GB,
    H100_TFLOPS_FP16,
    H100_TDP_W,
    NVLINK4_BW_GBS,
    IB_HDR200_BW_GBS,
    d1_gpu_count,
    d1_mfu,
    d1_parallelism,
    mo,
    math,
):
    _N = d1_gpu_count.value
    _mfu_frac = d1_mfu.value / 100.0
    _strategy = d1_parallelism.value

    # ── 1T model memory footprint (full training state) ────────────────────────
    # 1T params × 20 bytes/param = 20 TB total state
    # Source: @sec-training-memory-anatomy — weights(2) + grads(2) + Adam(8) + BF16(2) = 20 bytes
    _params = 1e12
    _total_state_bytes = _params * BYTES_PER_PARAM_FULL
    _total_state_tb = _total_state_bytes / 1e12

    # Memory required per GPU depends on sharding strategy
    _sharding_factor = {
        "dp":   1.0,    # no sharding → needs full model per replica
        "tp_pp": 32.0,  # TP=8 × PP=4 splits → 32× reduction
        "3d":   64.0,   # DP × TP × PP; assume DP=8, TP=8, PP=8 = 512 total
        "moe":  16.0,   # MoE: active params per expert shard
    }.get(_strategy, 32.0)

    _mem_per_gpu_tb = _total_state_tb / _sharding_factor
    _mem_per_gpu_gb = _mem_per_gpu_tb * 1000.0

    # OOM check: per-GPU required > H100_RAM_GB
    _oom = _mem_per_gpu_gb > H100_RAM_GB

    # ── Training throughput (tokens/day) ──────────────────────────────────────
    # Effective TFLOPS per GPU = peak × MFU
    # Tokens/step = 2 × params × seq_len (FLOPs per forward, assuming 2048 tokens)
    # Source: @sec-nn-computation-flop-counting — FLOPs ≈ 6 × P for a full train step
    _effective_tflops = H100_TFLOPS_FP16 * _mfu_frac  # per GPU
    _flops_per_token  = 6.0 * _params / 1e12           # TFLOPS needed for 1 token step
    _tokens_per_sec_per_gpu = _effective_tflops / _flops_per_token
    _tokens_per_day_total   = _tokens_per_sec_per_gpu * _N * 86400

    # Communication overhead: AllReduce gradient tensor
    # Gradient size = 2 bytes/param × 1T = 2 TB
    # Ring AllReduce time = 2(N-1)/N × gradient_size / BW
    _gradient_gb  = _params * 2 / 1e9   # BF16 gradients, GB
    _fabric_bw    = IB_HDR200_BW_GBS    # fallback: IB HDR200 400 GB/s
    _ring_time_s  = 2.0 * (_N - 1) / _N * _gradient_gb / _fabric_bw
    _step_compute_s = _flops_per_token * 2048 / _effective_tflops  # one step
    _comm_overhead_pct = (_ring_time_s / (_step_compute_s + _ring_time_s)) * 100.0

    # ── Training cost (cloud on-demand) ───────────────────────────────────────
    # Monthly update = 1 trillion tokens (GPT-3 class data budget × 3)
    # Source: @sec-vol2-introduction-training-scale
    _tokens_target = 1e12
    _days_to_train = _tokens_target / (_tokens_per_day_total + 1e-9)
    _gpu_hours     = _days_to_train * 24 * _N
    _train_cost_m  = _gpu_hours * H100_CLOUD_HR / 1e6   # millions $

    # ── Power (training cluster) ───────────────────────────────────────────────
    _cluster_power_mw = _N * H100_TDP_W / 1e6  # Megawatts

    # ── Color coding ──────────────────────────────────────────────────────────
    _mem_color  = COLORS["RedLine"] if _oom else COLORS["GreenLine"]
    _comm_color = (
        COLORS["GreenLine"] if _comm_overhead_pct < 15 else
        COLORS["OrangeLine"] if _comm_overhead_pct < 30 else
        COLORS["RedLine"]
    )
    _cost_color = COLORS["GreenLine"] if _train_cost_m < 500 else (
        COLORS["OrangeLine"] if _train_cost_m < 1000 else COLORS["RedLine"]
    )

    # ── Physics formula display ────────────────────────────────────────────────
    _formula = mo.Html(f"""
    <div class="lab-card" style="font-family: var(--font-mono); font-size: 0.88rem;
                                  margin: 8px 0; padding: 16px 20px;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            Training Physics — 1T Parameter Model
        </div>
        <div style="line-height: 2.1; color: {COLORS['Text']};">
            <div>Total state = 1T params &times; {BYTES_PER_PARAM_FULL} bytes/param
                = <strong>{_total_state_tb:.0f} TB</strong>
            </div>
            <div>Per-GPU memory = {_total_state_tb:.0f} TB / {_sharding_factor:.0f}
                (sharding) = <strong style="color:{_mem_color};">
                {_mem_per_gpu_gb:.1f} GB</strong>
                &nbsp;{'<span style="color:' + COLORS["RedLine"] + ';">&#x274C; OOM — exceeds 80 GB H100</span>'
                       if _oom else
                       '<span style="color:' + COLORS["GreenLine"] + ';">&#x2713; fits in H100 HBM</span>'}
            </div>
            <div>Effective throughput = {H100_TFLOPS_FP16} &times; {_mfu_frac:.2f} MFU
                &times; {_N:,} GPUs = <strong>{_tokens_per_day_total/1e9:.1f}B tokens/day</strong>
            </div>
            <div>Days to train 1T tokens = <strong>{_days_to_train:.1f} days</strong></div>
            <div>AllReduce overhead = <strong style="color:{_comm_color};">
                {_comm_overhead_pct:.1f}%</strong> of step time
            </div>
            <div>Training cost = <strong style="color:{_cost_color};">
                ${_train_cost_m:.0f}M</strong> (cloud on-demand per training run)
            </div>
            <div>Cluster power = <strong>{_cluster_power_mw:.1f} MW</strong></div>
        </div>
    </div>
    """)

    _oom_banner = None
    if _oom:
        _oom_banner = mo.callout(mo.md(
            f"**OOM — Infeasible.** With `{_strategy}` sharding, each GPU requires "
            f"**{_mem_per_gpu_gb:.1f} GB** but H100 HBM is only **{H100_RAM_GB} GB**. "
            f"The 1T model's full training state is {_total_state_tb:.0f} TB. "
            f"Increase sharding (try **3D Parallelism** with more GPUs) or the model "
            f"will not fit. This is not a software problem — it is a memory wall constraint."
        ), kind="danger")

    if _oom_banner:
        mo.vstack([_formula, _oom_banner])
    else:
        _formula
    return (
        _oom,
        _train_cost_m,
        _days_to_train,
        _tokens_per_day_total,
        _comm_overhead_pct,
        _cluster_power_mw,
        _N,
        _strategy,
    )


# ─── ACT II: DECISION 2 — INFERENCE SERVING ──────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Decision 2 — Inference Serving")
    return


@app.cell(hide_code=True)
def _(mo):
    d2_replicas = mo.ui.slider(
        start=100, stop=20000, value=5000, step=100,
        label="Cloud inference replica count (H100s)",
        show_value=True,
    )
    d2_quant = mo.ui.dropdown(
        options={
            "FP16 (full precision)":      "fp16",
            "INT8 (8-bit quantization)":  "int8",
            "INT4 (4-bit quantization)":  "int4",
            "1-bit (extreme compression)": "1bit",
        },
        value="INT8 (8-bit quantization)",
        label="Cloud tier quantization",
    )
    d2_edge_tier = mo.ui.dropdown(
        options={
            "None — cloud only":           "none",
            "Edge (Orin NX, INT4)":        "edge",
            "Edge + Mobile (INT4 + INT2)": "edge_mobile",
        },
        value="Edge + Mobile (INT4 + INT2)",
        label="Edge/mobile offload tier",
    )
    mo.hstack([d2_replicas, d2_quant, d2_edge_tier], justify="start", gap=4)
    return (d2_replicas, d2_quant, d2_edge_tier)


@app.cell(hide_code=True)
def _(
    COLORS,
    BYTES_PER_PARAM_BF16,
    H100_BW_GBS,
    H100_CLOUD_HR,
    H100_RAM_GB,
    USERS_SCALE,
    d2_edge_tier,
    d2_quant,
    d2_replicas,
    mo,
    math,
):
    _R = d2_replicas.value
    _quant = d2_quant.value
    _tier  = d2_edge_tier.value

    # ── Bytes per parameter for each quantization level ───────────────────────
    # Source: @sec-model-compression-quantization
    _bytes_per_param = {
        "fp16": 2.0,
        "int8": 1.0,
        "int4": 0.5,
        "1bit": 0.125,
    }.get(_quant, 1.0)

    _model_size_gb = 1e12 * _bytes_per_param / 1e9  # 1T params
    _fits_h100     = _model_size_gb <= H100_RAM_GB

    # For INT8 and below, assume sharded across N_shard H100s
    _n_shard = math.ceil(_model_size_gb / H100_RAM_GB)
    _n_shard = max(_n_shard, 1)

    # ── Per-replica throughput via arithmetic intensity (roofline) ─────────────
    # Decoding: 1 token per step, 2 × model_params FLOPs per token
    # Arithmetic intensity = 2P / (2P bytes) = 1 op/byte for batch=1
    # → memory-bandwidth bound → throughput = BW / bytes_per_param / params
    # Source: @sec-inference-roofline-decode
    _decode_toks_per_sec = H100_BW_GBS * 1e9 / (1e12 * _bytes_per_param)
    _decode_toks_per_sec_total = _decode_toks_per_sec * _R

    # ── Steady-state users via Little's Law ───────────────────────────────────
    # N = λ × W  →  λ_max = N_in_flight / W
    # Assume mean latency W = 100 tokens / decode_rate
    # Assume 100-token response, 1 concurrent request per H100 shard
    _tokens_per_response = 100
    _latency_s  = _tokens_per_response / (_decode_toks_per_sec + 1e-9)
    _rps_total  = _R / _n_shard / (_latency_s + 1e-9)   # requests per second

    # P99 estimate: Kingman's formula M/M/c → P99 ≈ avg_latency × log(100×(1-ρ))^-1
    # Simplified: assume P99 ≈ 3× avg for moderate utilization
    _p99_ms = _latency_s * 1000 * 3.0  # P99 in ms

    # SLO check: P99 < 500ms
    _slo_ok = _p99_ms < 500.0

    # ── Daily serving cost ────────────────────────────────────────────────────
    _serving_cost_day_m = _R * H100_CLOUD_HR * 24 / 1e6  # millions $/day
    _serving_cost_yr_b  = _serving_cost_day_m * 365 / 1000  # billion $/year

    # ── Daily concurrent user capacity at P99 SLO ─────────────────────────────
    # 5B users, assume peak 10% concurrent = 500M simultaneous
    _peak_concurrent = USERS_SCALE * 0.10
    _capacity_ok = _rps_total * _latency_s >= _peak_concurrent

    # ── Color coding ──────────────────────────────────────────────────────────
    _slo_color      = COLORS["GreenLine"] if _slo_ok else COLORS["RedLine"]
    _capacity_color = COLORS["GreenLine"] if _capacity_ok else COLORS["RedLine"]
    _cost_color     = COLORS["GreenLine"] if _serving_cost_yr_b < 5 else (
        COLORS["OrangeLine"] if _serving_cost_yr_b < 8 else COLORS["RedLine"]
    )

    _formula = mo.Html(f"""
    <div class="lab-card" style="font-family: var(--font-mono); font-size: 0.88rem;
                                  margin: 8px 0; padding: 16px 20px;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            Serving Physics — Little's Law + Roofline
        </div>
        <div style="line-height: 2.1; color: {COLORS['Text']};">
            <div>Model size ({_quant}) = 1T &times; {_bytes_per_param} bytes
                = <strong>{_model_size_gb:.0f} GB</strong>
                &nbsp;({'requires ' + str(_n_shard) + ' H100 shards/replica'})
            </div>
            <div>Decode rate = BW / bytes_per_param / params
                = {H100_BW_GBS}e9 / ({_bytes_per_param} &times; 1e12)
                = <strong>{_decode_toks_per_sec:.2f} tok/s/GPU</strong>
            </div>
            <div>Avg latency (100 tok) = 100 / {_decode_toks_per_sec:.2f}
                = <strong>{_latency_s*1000:.0f} ms</strong>
            </div>
            <div>P99 latency (est. 3&times; avg)
                = <strong style="color:{_slo_color};">{_p99_ms:.0f} ms</strong>
                &nbsp;{'&#x2713; &lt; 500ms SLO' if _slo_ok else '&#x274C; EXCEEDS 500ms SLO'}
            </div>
            <div>Total RPS = {_R:,} replicas / {_n_shard} shards / {_latency_s:.3f}s
                = <strong>{_rps_total:,.0f} req/s</strong>
            </div>
            <div>Concurrent users supported (N=&lambda;W)
                = <strong style="color:{_capacity_color};">
                {_rps_total * _latency_s:,.0f}</strong>
                &nbsp;({'&#x2713; &ge; 500M peak' if _capacity_ok else '&#x274C; below 500M peak'})
            </div>
            <div>Annual serving cost
                = <strong style="color:{_cost_color};">${_serving_cost_yr_b:.2f}B/yr</strong>
            </div>
        </div>
    </div>
    """)

    _banners = []
    if not _slo_ok:
        _banners.append(mo.callout(mo.md(
            f"**P99 SLO Violation.** Estimated P99 latency is **{_p99_ms:.0f} ms**, "
            f"exceeding the 500ms global SLO. "
            f"The {_quant} model decodes at only {_decode_toks_per_sec:.2f} tok/s/GPU "
            f"— bandwidth-bound, not compute-bound (arithmetic intensity = 1 op/byte). "
            f"Options: increase replicas, use a smaller distilled model per tier, "
            f"or shift load to edge/mobile (which reduces cloud P99 tail)."
        ), kind="danger"))

    if not _capacity_ok:
        _banners.append(mo.callout(mo.md(
            f"**Capacity Insufficient.** Your system handles "
            f"~{_rps_total * _latency_s:,.0f} concurrent users but "
            f"peak demand is {int(USERS_SCALE * 0.10):,} (10% of 5B). "
            f"Add replicas or offload a larger fraction of requests to edge/mobile tiers."
        ), kind="warn"))

    if _banners:
        mo.vstack([_formula] + _banners)
    else:
        _formula
    return (
        _slo_ok,
        _capacity_ok,
        _p99_ms,
        _rps_total,
        _serving_cost_yr_b,
        _model_size_gb,
        _n_shard,
        _quant,
        _tier,
    )


# ─── ACT II: DECISION 3 — FAULT TOLERANCE ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Decision 3 — Fault Tolerance")
    return


@app.cell(hide_code=True)
def _(mo):
    d3_ckpt_interval = mo.ui.slider(
        start=5, stop=120, value=30, step=5,
        label="Checkpoint interval (minutes)",
        show_value=True,
    )
    d3_replication = mo.ui.dropdown(
        options={
            "No replication (single copy)":  "1",
            "2× replication":                "2",
            "3× replication (standard)":     "3",
            "5× replication (high-value)":   "5",
        },
        value="3× replication (standard)",
        label="Checkpoint storage replication factor",
    )
    mo.hstack([d3_ckpt_interval, d3_replication], justify="start", gap=4)
    return (d3_ckpt_interval, d3_replication)


@app.cell(hide_code=True)
def _(
    COLORS,
    H100_MTBF_HOURS,
    _N,
    d3_ckpt_interval,
    d3_replication,
    mo,
    math,
):
    _T_min = d3_ckpt_interval.value
    _T_hr  = _T_min / 60.0
    _rep   = int(d3_replication.value)

    # ── Cluster-level failure rate ─────────────────────────────────────────────
    # lambda_cluster = N / MTBF_per_GPU  [independent failures]
    # Source: @sec-fault-tolerance-failure-modes
    _lambda = _N / H100_MTBF_HOURS      # failures/hour
    _cluster_mtbf_hr = 1.0 / _lambda
    _cluster_mtbf_min = _cluster_mtbf_hr * 60.0

    # ── Young-Daly optimal checkpoint interval ─────────────────────────────────
    # T* = sqrt(2 × C / lambda)  [source: @sec-fault-tolerance-young-daly]
    # Checkpoint cost C: 1T model at 2 bytes/param = 2 TB; Lustre 400 GB/s
    # C = 2000 GB / 400 GB/s = 5 seconds → 0.083 minutes
    _ckpt_size_gb   = 1e12 * 2 / 1e9     # 2 TB for BF16 weights
    _lustre_bw_gbs  = 400.0               # GB/s aggregate; @sec-fault-tolerance
    _C_s  = _ckpt_size_gb / _lustre_bw_gbs
    _C_hr = _C_s / 3600.0
    _C_min = _C_s / 60.0

    _T_opt_hr  = math.sqrt(2.0 * _C_hr / _lambda)
    _T_opt_min = _T_opt_hr * 60.0

    # ── Overhead and expected waste ────────────────────────────────────────────
    _overhead_pct = (_C_hr / _T_hr) * 100.0
    _waste_per_failure_min = _T_min / 2.0 + _C_min
    _expected_waste_rate   = _lambda * (_T_hr / 2.0 + _C_hr)  # fraction of time

    # Overhead ceiling: if checkpoint overhead > 20% → critical
    _overhead_ok = _overhead_pct < 20.0

    # ── 99.99% availability calculation ───────────────────────────────────────
    # Availability = 1 - downtime_fraction
    # Downtime per failure ≈ T/2 + C + restart_time (assume restart = 30 min)
    _restart_hr = 0.5  # 30 min restart
    _downtime_per_failure_hr = _T_hr / 2.0 + _C_hr + _restart_hr
    _downtime_fraction = _lambda * _downtime_per_failure_hr
    _availability_pct  = (1.0 - _downtime_fraction) * 100.0
    _avail_ok = _availability_pct >= 99.99

    # ── Color coding ──────────────────────────────────────────────────────────
    _T_ratio   = _T_min / max(_T_opt_min, 0.01)
    _int_color = (
        COLORS["GreenLine"] if 0.7 <= _T_ratio <= 1.4 else
        COLORS["OrangeLine"] if _T_ratio < 0.7 else
        COLORS["RedLine"]
    )
    _ovh_color = (
        COLORS["GreenLine"] if _overhead_pct < 10 else
        COLORS["OrangeLine"] if _overhead_pct < 20 else
        COLORS["RedLine"]
    )
    _avail_color = COLORS["GreenLine"] if _avail_ok else COLORS["RedLine"]

    _formula = mo.Html(f"""
    <div class="lab-card" style="font-family: var(--font-mono); font-size: 0.88rem;
                                  margin: 8px 0; padding: 16px 20px;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            Fault Tolerance Physics — Young-Daly + Availability
        </div>
        <div style="line-height: 2.1; color: {COLORS['Text']};">
            <div>Cluster &lambda; = {_N:,} GPUs / {H100_MTBF_HOURS}hr per-GPU MTBF
                = <strong>{_lambda:.2f} failures/hr</strong>
                &nbsp;(MTBF = <strong>{_cluster_mtbf_min:.1f} min</strong>)
            </div>
            <div>Checkpoint cost C = {_ckpt_size_gb:.0f} GB / {_lustre_bw_gbs:.0f} GB/s
                = <strong>{_C_min:.1f} min</strong>
            </div>
            <div>Young-Daly T* = sqrt(2 &times; {_C_hr:.4f}hr / {_lambda:.4f}/hr)
                = <strong style="color:{COLORS['GreenLine']};">{_T_opt_min:.1f} min</strong>
            </div>
            <div>Your T = {_T_min} min
                &nbsp;({_T_ratio:.1f}&times; {'too frequent' if _T_ratio < 0.7 else 'too infrequent' if _T_ratio > 1.4 else 'near-optimal'})
                &nbsp;<strong style="color:{_int_color};">
                {('near-optimal' if 0.7 <= _T_ratio <= 1.4 else 'suboptimal')}</strong>
            </div>
            <div>Checkpoint overhead = C/T = {_C_hr:.4f}/{_T_hr:.4f}
                = <strong style="color:{_ovh_color};">{_overhead_pct:.1f}%</strong>
                &nbsp;{'&#x274C; &gt;20% ceiling' if not _overhead_ok else '&#x2713; OK'}
            </div>
            <div>Expected waste rate = &lambda; &times; (T/2 + C)
                = <strong>{_expected_waste_rate*100:.1f}%</strong> of training time
            </div>
            <div>Availability = 1 - &lambda; &times; downtime
                = <strong style="color:{_avail_color};">{_availability_pct:.4f}%</strong>
                &nbsp;{'&#x2713; &ge; 99.99%' if _avail_ok else '&#x274C; &lt; 99.99% SLO'}
            </div>
        </div>
    </div>
    """)

    _banners = []
    if not _overhead_ok:
        _banners.append(mo.callout(mo.md(
            f"**Checkpoint Overhead Critical.** Your {_T_min}-minute interval "
            f"with {_C_min:.1f}-minute checkpoint cost yields "
            f"**{_overhead_pct:.1f}% overhead** — exceeding the 20% ceiling. "
            f"Young-Daly optimal is **{_T_opt_min:.1f} minutes**. "
            f"At cluster MTBF = {_cluster_mtbf_min:.0f} min ({_N:,} GPUs), "
            f"checkpointing more frequently than T* costs more than it saves."
        ), kind="danger"))

    if not _avail_ok:
        _banners.append(mo.callout(mo.md(
            f"**Availability Below 99.99%.** Current architecture delivers "
            f"**{_availability_pct:.4f}%** availability. "
            f"With cluster MTBF = {_cluster_mtbf_min:.0f} minutes and "
            f"restart overhead of 30 minutes per failure, "
            f"you cannot reach four-nines without checkpointing strategy optimization. "
            f"Consider async multi-level checkpointing to reduce restart cost."
        ), kind="warn"))

    if _banners:
        mo.vstack([_formula] + _banners)
    else:
        _formula
    return (
        _T_min,
        _T_opt_min,
        _overhead_ok,
        _avail_ok,
        _availability_pct,
        _overhead_pct,
        _C_min,
        _lambda,
        _cluster_mtbf_min,
        _ckpt_size_gb,
    )


# ─── ACT II: DECISION 4 — PRIVACY ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Decision 4 — Privacy")
    return


@app.cell(hide_code=True)
def _(mo):
    d4_epsilon = mo.ui.slider(
        start=0.1, stop=10.0, value=1.0, step=0.1,
        label="Differential privacy epsilon (ε) — lower = stronger privacy",
        show_value=True,
    )
    d4_strategy = mo.ui.dropdown(
        options={
            "Centralized training (all data to cloud)":     "central",
            "Federated learning (EU region)":               "federated_eu",
            "Federated (EU) + Central (non-EU)":            "hybrid",
            "Full federated (all regions)":                 "full_federated",
        },
        value="Federated (EU) + Central (non-EU)",
        label="Data residency strategy",
    )
    mo.hstack([d4_epsilon, d4_strategy], justify="start", gap=4)
    return (d4_epsilon, d4_strategy)


@app.cell(hide_code=True)
def _(
    COLORS,
    d4_epsilon,
    d4_strategy,
    mo,
    math,
):
    _eps    = d4_epsilon.value
    _strat  = d4_strategy.value

    # ── GDPR differential privacy compliance ──────────────────────────────────
    # GDPR Art. 25 + EDPB guidance: epsilon ≤ 1.0 required for strong DP guarantee
    # Source: @sec-security-privacy-dp-gdpr
    _GDPR_EPS_MAX = 1.0     # ε ≤ 1.0 for GDPR-grade DP; @sec-security-privacy-dp
    _CCPA_EPS_MAX = 3.0     # ε ≤ 3.0 for CCPA-grade; @sec-security-privacy-ccpa

    _gdpr_ok = _eps <= _GDPR_EPS_MAX
    _ccpa_ok = _eps <= _CCPA_EPS_MAX

    # ── Accuracy degradation model from DP noise ───────────────────────────────
    # Approximate relationship: accuracy_penalty ≈ k / epsilon (diminishing returns)
    # At ε=1.0: ~5% accuracy drop; at ε=0.1: ~15%; at ε=10: ~0.5%
    # Source: @sec-security-privacy-dp-accuracy-tradeoff
    _k_accuracy = 0.05  # empirical constant (5% penalty at ε=1)
    _accuracy_penalty_pct = min(_k_accuracy / _eps * 100, 25.0)

    # ── Federated learning communication overhead ─────────────────────────────
    # Federated: each round requires uploading model diff ≈ gradient size
    # 1T model gradient at BF16 = 2 TB per round
    # Mobile uplink ≈ 10 Mbps → 2 TB / 10 Mbps = 1.6M seconds → impractical
    # Source: @sec-edge-intelligence-federated-communication
    _is_federated = _strat in ("federated_eu", "hybrid", "full_federated")
    _gradient_gb  = 1e12 * 2 / 1e9  # 2000 GB
    _mobile_uplink_gbps = 0.010      # 10 Mbps typical; @sec-edge-intelligence
    _upload_time_s = _gradient_gb / _mobile_uplink_gbps
    _upload_time_days = _upload_time_s / 86400

    # Practical: federated sends only adapter diff (LoRA delta), not full gradient
    # LoRA rank=16, 1T model → ~0.001% of model = ~2 GB
    _lora_diff_gb = 2.0
    _lora_upload_s = _lora_diff_gb / _mobile_uplink_gbps
    _lora_upload_min = _lora_upload_s / 60.0

    # ── Color coding ──────────────────────────────────────────────────────────
    _gdpr_color = COLORS["GreenLine"] if _gdpr_ok else COLORS["RedLine"]
    _ccpa_color = COLORS["GreenLine"] if _ccpa_ok else COLORS["RedLine"]
    _acc_color  = (
        COLORS["GreenLine"] if _accuracy_penalty_pct < 5 else
        COLORS["OrangeLine"] if _accuracy_penalty_pct < 12 else
        COLORS["RedLine"]
    )

    _formula = mo.Html(f"""
    <div class="lab-card" style="font-family: var(--font-mono); font-size: 0.88rem;
                                  margin: 8px 0; padding: 16px 20px;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            Privacy Physics — Differential Privacy &epsilon;-&delta; Tradeoff
        </div>
        <div style="line-height: 2.1; color: {COLORS['Text']};">
            <div>DP noise scale: &sigma; &propto; &Delta;f / &epsilon;
                &nbsp;&mdash;&nbsp; smaller &epsilon; = more noise added
            </div>
            <div>GDPR (&epsilon; &le; {_GDPR_EPS_MAX}):
                &nbsp;<strong style="color:{_gdpr_color};">
                &epsilon; = {_eps:.1f} &nbsp;
                {'&#x2713; GDPR-compliant' if _gdpr_ok else '&#x274C; GDPR violation'}
                </strong>
            </div>
            <div>CCPA (&epsilon; &le; {_CCPA_EPS_MAX}):
                &nbsp;<strong style="color:{_ccpa_color};">
                {'&#x2713; CCPA-compliant' if _ccpa_ok else '&#x274C; CCPA violation'}
                </strong>
            </div>
            <div>Estimated accuracy penalty &asymp; k/&epsilon;
                = 0.05/{_eps:.1f}
                = <strong style="color:{_acc_color};">~{_accuracy_penalty_pct:.1f}%</strong>
                degradation
            </div>
            <div>Data strategy: <strong>{_strat}</strong>
                &nbsp;{'(federated requires LoRA adapter diffs)' if _is_federated else ''}
            </div>
            {f'<div>LoRA diff upload (10 Mbps): <strong>{_lora_upload_min:.1f} min/round</strong></div>' if _is_federated else ''}
        </div>
    </div>
    """)

    _banners = []
    if not _gdpr_ok:
        _banners.append(mo.callout(mo.md(
            f"**GDPR Violation.** Your epsilon = **{_eps:.1f}** exceeds the GDPR-grade "
            f"threshold of ε ≤ {_GDPR_EPS_MAX}. "
            f"EU data regulators interpret ε > 1 as providing insufficient anonymization "
            f"under the EDPB's differential privacy guidance. "
            f"Reduce ε or migrate EU users to a federated strategy where raw data "
            f"never leaves the device."
        ), kind="danger"))

    if not _ccpa_ok:
        _banners.append(mo.callout(mo.md(
            f"**CCPA Violation.** Epsilon = **{_eps:.1f}** also exceeds CCPA threshold "
            f"(ε ≤ {_CCPA_EPS_MAX}). California users' data is not adequately protected. "
            f"This may trigger regulatory enforcement under CPRA 2023 provisions."
        ), kind="danger"))

    if _banners:
        mo.vstack([_formula] + _banners)
    else:
        _formula
    return (
        _eps,
        _strat,
        _gdpr_ok,
        _ccpa_ok,
        _accuracy_penalty_pct,
        _is_federated,
    )


# ─── ACT II: DECISION 5 — FAIRNESS ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Decision 5 — Fairness")
    return


@app.cell(hide_code=True)
def _(mo):
    d5_fairness = mo.ui.dropdown(
        options={
            "Equalized odds (equal FPR + FNR across groups)":     "equalized_odds",
            "Demographic parity (equal positive rate)":            "dem_parity",
            "Calibration (equal predicted probabilities)":         "calibration",
            "Individual fairness (similar people treated similarly)": "individual",
        },
        value="Equalized odds (equal FPR + FNR across groups)",
        label="Fairness criterion",
    )
    d5_base_rate_gap = mo.ui.slider(
        start=0, stop=40, value=15, step=1,
        label="Base rate gap between highest and lowest country (%)",
        show_value=True,
    )
    d5_accuracy = mo.ui.slider(
        start=50, stop=99, value=85, step=1,
        label="Overall model accuracy (%)",
        show_value=True,
    )
    mo.vstack([
        d5_fairness,
        mo.hstack([d5_base_rate_gap, d5_accuracy], justify="start", gap=4),
    ])
    return (d5_fairness, d5_base_rate_gap, d5_accuracy)


@app.cell(hide_code=True)
def _(
    COLORS,
    d5_accuracy,
    d5_base_rate_gap,
    d5_fairness,
    mo,
    math,
):
    _criterion     = d5_fairness.value
    _base_rate_gap = d5_base_rate_gap.value / 100.0   # convert to fraction
    _acc           = d5_accuracy.value / 100.0

    # ── Chouldechova impossibility ─────────────────────────────────────────────
    # When base rates differ across groups, cannot simultaneously satisfy:
    # (1) equalized odds (equal FPR + FNR)
    # (2) calibration (PPV equal across groups)
    # Source: @sec-responsible-ai-chouldechova
    # Minimum gap if we enforce equalized odds given base rate difference:
    #   PPV_1 / PPV_2 >= (1 - base_rate_2) / (1 - base_rate_1) × base_rate_1/base_rate_2
    # Approximate: forced accuracy loss when equalizing FPR across groups
    # with base rate gap delta: loss ≈ delta × (1 - acc) / base_rate_midpoint
    _base_rate_low  = 0.20                          # lowest-prevalence group
    _base_rate_high = _base_rate_low + _base_rate_gap

    # Chouldechova: if we force equalized odds → calibration must be unequal
    # Calibration gap ≈ base_rate_gap / (avg_base_rate × (1 + base_rate_gap))
    _avg_base_rate = (_base_rate_low + _base_rate_high) / 2.0
    if _avg_base_rate > 0 and _avg_base_rate < 1:
        _calibration_gap = _base_rate_gap / (_avg_base_rate * (1.0 + _base_rate_gap))
    else:
        _calibration_gap = 0.0
    _calibration_gap_pct = _calibration_gap * 100.0

    # Equalized odds gap if we enforce calibration:
    # FPR gap ≈ base_rate_gap × (1 - acc) / avg_base_rate
    _fpr_gap_pct = _base_rate_gap * (1.0 - _acc) / max(_avg_base_rate, 0.01) * 100.0

    # EU AI Act Art. 10: equalized odds gap ≤ 10% for high-risk systems
    # Source: @sec-responsible-ai-eu-ai-act
    _EU_AIACT_MAX_GAP = 0.10  # 10% maximum FPR/FNR gap
    _eu_ok = _fpr_gap_pct / 100.0 <= _EU_AIACT_MAX_GAP

    # ── Accuracy penalty from fairness constraint ──────────────────────────────
    # Enforcing equalized odds costs accuracy proportional to base rate gap
    # Conservative estimate: 1-3% accuracy for every 5% base rate gap
    _fairness_acc_penalty_pct = _base_rate_gap * 100.0 * 0.20   # 0.2 pp per 1pp gap

    # ── Color coding ──────────────────────────────────────────────────────────
    _eu_color   = COLORS["GreenLine"] if _eu_ok else COLORS["RedLine"]
    _calib_color = (
        COLORS["GreenLine"] if _calibration_gap_pct < 10 else
        COLORS["OrangeLine"] if _calibration_gap_pct < 20 else
        COLORS["RedLine"]
    )

    # Is Chouldechova active? — when base rate gap > 0 and using equalized odds
    _chouldechova_active = _base_rate_gap > 0.01 and _criterion == "equalized_odds"

    _formula = mo.Html(f"""
    <div class="lab-card" style="font-family: var(--font-mono); font-size: 0.88rem;
                                  margin: 8px 0; padding: 16px 20px;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            Fairness Physics — Chouldechova Impossibility
        </div>
        <div style="line-height: 2.1; color: {COLORS['Text']};">
            <div>Base rate range: {_base_rate_low*100:.0f}% (lowest) &rarr;
                {_base_rate_high*100:.0f}% (highest);
                gap = <strong>{_base_rate_gap*100:.0f}%</strong>
                across 193 jurisdictions
            </div>
            <div>Criterion: <strong>{_criterion}</strong></div>
            <div>If equalized odds is enforced &rarr; calibration gap
                &asymp; <strong style="color:{_calib_color};">
                {_calibration_gap_pct:.1f}%</strong>
                (Chouldechova constraint, @sec-responsible-ai-chouldechova)
            </div>
            <div>If calibration is enforced &rarr; FPR/FNR gap
                &asymp; <strong style="color:{_eu_color};">
                {_fpr_gap_pct:.1f}%</strong>
                &nbsp;{'&#x2713; &le; 10% EU AI Act' if _eu_ok else '&#x274C; &gt;10% EU AI Act Art. 10'}
            </div>
            <div>Fairness constraint accuracy cost
                &asymp; <strong>~{_fairness_acc_penalty_pct:.1f}%</strong>
                degradation
            </div>
        </div>
    </div>
    """)

    _banners = []
    if not _eu_ok:
        _banners.append(mo.callout(mo.md(
            f"**EU AI Act Violation.** Enforcing calibration with a {_base_rate_gap*100:.0f}% "
            f"base rate gap across jurisdictions produces an FPR/FNR gap of "
            f"**{_fpr_gap_pct:.1f}%** — exceeding the 10% threshold under EU AI Act Article 10. "
            f"The Chouldechova impossibility theorem states this cannot be fixed by "
            f"better training data alone: the incompatibility is mathematical, not empirical. "
            f"Options: jurisdiction-specific models, or accept accuracy cost to enforce "
            f"equalized odds at the expense of calibration."
        ), kind="danger"))

    if _chouldechova_active and _base_rate_gap > 0.10:
        _banners.append(mo.callout(mo.md(
            f"**Chouldechova Theorem Active.** With a {_base_rate_gap*100:.0f}% base rate gap "
            f"and equalized odds enforcement, calibration error will be approximately "
            f"**{_calibration_gap_pct:.1f}%**. This is not a model quality problem — "
            f"it is a mathematical constraint from @sec-responsible-ai-chouldechova. "
            f"The only architectural solutions are: per-jurisdiction models, "
            f"rejection of the equalized odds criterion in high-gap jurisdictions, "
            f"or explicit transparency to regulators."
        ), kind="warn"))

    if _banners:
        mo.vstack([_formula] + _banners)
    else:
        _formula
    return (
        _criterion,
        _base_rate_gap,
        _eu_ok,
        _fpr_gap_pct,
        _calibration_gap_pct,
        _fairness_acc_penalty_pct,
        _chouldechova_active,
    )


# ─── ACT II: CARBON CONSTRAINT ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Carbon Constraint — 2027 Carbon-Neutral Commitment")
    return


@app.cell(hide_code=True)
def _(
    CARBON_THRESHOLD_G_KWH,
    COLORS,
    EU_GRID_CARBON_G_KWH,
    H100_TDP_W,
    RENEW_CARBON_G_KWH,
    _cluster_power_mw,
    _N,
    _serving_cost_yr_b,
    mo,
):
    # ── Total system power ─────────────────────────────────────────────────────
    # Training cluster + serving fleet + overhead (PUE 1.2×)
    # Source: @sec-sustainable-ai-data-center-pue
    _PUE = 1.2                          # Power Usage Effectiveness; industry average
    _serving_gpus_est = int(_serving_cost_yr_b * 1e9 / (3.50 * 8760))   # rough estimate

    _total_gpus   = int(_N) + _serving_gpus_est
    _raw_power_mw = _total_gpus * H100_TDP_W / 1e6   # MW
    _total_power_mw = _raw_power_mw * _PUE

    # ── Annual energy ─────────────────────────────────────────────────────────
    _annual_energy_gwh = _total_power_mw * 8760 / 1000  # GWh/year

    # ── Carbon emissions ──────────────────────────────────────────────────────
    # Assume 3 data center regions: EU, US-CA, US-East
    # Mix: 40% renewable PPA, 60% grid average
    _renew_fraction = 0.4
    _eff_carbon = (
        _renew_fraction * RENEW_CARBON_G_KWH +
        (1 - _renew_fraction) * EU_GRID_CARBON_G_KWH
    )
    _carbon_ok = _eff_carbon <= CARBON_THRESHOLD_G_KWH

    _annual_co2_kt = _annual_energy_gwh * _eff_carbon / 1e6 * 1e9 / 1e6  # kilotonnes CO2

    # Carbon-neutral path: 100% renewable PPA
    _eff_carbon_100renew = RENEW_CARBON_G_KWH
    _co2_100renew_kt     = _annual_energy_gwh * _eff_carbon_100renew / 1e6 * 1e9 / 1e6

    # ── Renewable PPA cost premium ─────────────────────────────────────────────
    # Renewable PPA ~$50/MWh vs grid ~$40/MWh → ~25% premium
    # Source: @sec-sustainable-ai-carbon-aware-scheduling
    _ppa_premium_usd_m = _annual_energy_gwh * 1000 * 10.0 / 1e6  # $10/MWh delta × GWh

    _carbon_color = COLORS["GreenLine"] if _carbon_ok else COLORS["RedLine"]
    _eff_color    = COLORS["GreenLine"] if _eff_carbon <= CARBON_THRESHOLD_G_KWH else (
        COLORS["OrangeLine"] if _eff_carbon <= 150 else COLORS["RedLine"]
    )

    _formula = mo.Html(f"""
    <div class="lab-card" style="font-family: var(--font-mono); font-size: 0.88rem;
                                  margin: 8px 0; padding: 16px 20px;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">
            Carbon Physics — Jevons Paradox + Grid Carbon Intensity
        </div>
        <div style="line-height: 2.1; color: {COLORS['Text']};">
            <div>Total GPUs (train + serve) &asymp; <strong>{_total_gpus:,}</strong></div>
            <div>Raw GPU power = {_total_gpus:,} &times; {H100_TDP_W}W
                = <strong>{_raw_power_mw:.0f} MW</strong>
            </div>
            <div>Total facility power (PUE {_PUE})
                = <strong>{_total_power_mw:.0f} MW</strong>
            </div>
            <div>Annual energy = {_total_power_mw:.0f}MW &times; 8,760 hr
                = <strong>{_annual_energy_gwh:.0f} GWh/yr</strong>
            </div>
            <div>Effective carbon intensity ({int(_renew_fraction*100)}% renewable PPA)
                = <strong style="color:{_eff_color};">
                {_eff_carbon:.0f} g CO&sub2;/kWh</strong>
                &nbsp;(threshold: {CARBON_THRESHOLD_G_KWH} g/kWh)
                &nbsp;{'&#x2713; carbon-neutral' if _carbon_ok else '&#x274C; above threshold'}
            </div>
            <div>Annual CO&sub2; emissions
                &asymp; <strong>{_annual_co2_kt:.0f} kt CO&sub2;</strong>
            </div>
            <div>100% renewable path: {_co2_100renew_kt:.0f} kt CO&sub2;
                | PPA premium: +${_ppa_premium_usd_m:.0f}M/yr
            </div>
        </div>
    </div>
    """)

    _banners = []
    if not _carbon_ok:
        _banners.append(mo.callout(mo.md(
            f"**Carbon Target Missed.** With {int(_renew_fraction*100)}% renewable PPA, "
            f"effective carbon intensity is **{_eff_carbon:.0f} g CO2/kWh**, "
            f"exceeding the carbon-neutral threshold of {CARBON_THRESHOLD_G_KWH} g/kWh. "
            f"Your {_total_power_mw:.0f} MW fleet emits ~{_annual_co2_kt:.0f} kt CO2/year. "
            f"To reach carbon-neutral by 2027, increase renewable PPA to ≥90% or "
            f"relocate training workloads to zero-carbon regions (Iceland, Norway, Quebec). "
            f"Note the Jevons Paradox (@sec-sustainable-ai-jevons): "
            f"efficiency improvements alone cannot reach this target if fleet size grows."
        ), kind="danger"))

    if _banners:
        mo.vstack([_formula] + _banners)
    else:
        _formula
    return (
        _carbon_ok,
        _total_power_mw,
        _annual_energy_gwh,
        _eff_carbon,
        _annual_co2_kt,
        _total_gpus,
    )


# ─── ACT II: SYSTEM FEASIBILITY VERDICT ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### System Feasibility Verdict")
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    _accuracy_penalty_pct,
    _annual_co2_kt,
    _avail_ok,
    _carbon_ok,
    _eu_ok,
    _fairness_acc_penalty_pct,
    _gdpr_ok,
    _oom,
    _overhead_ok,
    _p99_ms,
    _serving_cost_yr_b,
    _slo_ok,
    _total_gpus,
    _train_cost_m,
    go,
    apply_plotly_theme,
    mo,
):
    # ── Aggregate system validity ──────────────────────────────────────────────
    _constraints = {
        "Training: No OOM":          not _oom,
        "Checkpoint: Overhead OK":   _overhead_ok,
        "Serving: P99 < 500ms":      _slo_ok,
        "Reliability: 99.99% avail":  _avail_ok,
        "Privacy: GDPR ε ≤ 1":       _gdpr_ok,
        "Fairness: EU AI Act ≤ 10%": _eu_ok,
        "Carbon: Neutral by 2027":   _carbon_ok,
    }

    _total_pass   = sum(_constraints.values())
    _total_checks = len(_constraints)
    _system_valid = _total_pass == _total_checks

    # ── Total cost estimate ────────────────────────────────────────────────────
    _total_cost_b = (_train_cost_m * 12 / 1000) + _serving_cost_yr_b  # billion $/year

    # ── Constraint bar chart ───────────────────────────────────────────────────
    _labels = list(_constraints.keys())
    _pass   = [1 if v else 0 for v in _constraints.values()]
    _colors_bar = [
        COLORS["GreenLine"] if v else COLORS["RedLine"]
        for v in _constraints.values()
    ]

    _fig = go.Figure(go.Bar(
        x=_labels,
        y=_pass,
        marker_color=_colors_bar,
        text=["PASS" if v else "FAIL" for v in _constraints.values()],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["TextSec"]),
    ))
    _fig.update_layout(
        height=280,
        xaxis=dict(tickangle=-20, tickfont=dict(size=10, color=COLORS["TextSec"])),
        yaxis=dict(visible=False, range=[0, 1.4]),
        margin=dict(t=40, b=80, l=20, r=20),
        title=dict(
            text=f"System Constraint Audit — {_total_pass}/{_total_checks} Passed",
            font=dict(size=13, color=COLORS["Text"]),
            x=0.5,
        ),
    )
    apply_plotly_theme(_fig)

    # ── Summary card ──────────────────────────────────────────────────────────
    _verdict_color = COLORS["GreenLine"] if _system_valid else COLORS["RedLine"]
    _verdict_label = "FEASIBLE" if _system_valid else "INFEASIBLE"

    _summary = mo.Html(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
                margin: 16px 0;">
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                System Verdict
            </div>
            <div style="font-size: 1.8rem; font-weight: 900; color: {_verdict_color};">
                {_verdict_label}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {_total_pass}/{_total_checks} constraints
            </div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                P99 Latency
            </div>
            <div style="font-size: 1.8rem; font-weight: 800;
                        color: {COLORS['GreenLine'] if _slo_ok else COLORS['RedLine']};">
                {_p99_ms:.0f}ms
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">SLO: &lt; 500ms</div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Annual Cost
            </div>
            <div style="font-size: 1.8rem; font-weight: 800;
                        color: {COLORS['GreenLine'] if _total_cost_b < 10 else COLORS['RedLine']};">
                ${_total_cost_b:.1f}B
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                budget: $10B total
            </div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Annual CO2
            </div>
            <div style="font-size: 1.8rem; font-weight: 800;
                        color: {COLORS['GreenLine'] if _carbon_ok else COLORS['RedLine']};">
                {_annual_co2_kt:.0f}kt
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {'carbon-neutral' if _carbon_ok else 'above target'}
            </div>
        </div>
    </div>
    """)

    mo.vstack([_summary, mo.ui.plotly(_fig)])
    return (
        _constraints,
        _system_valid,
        _total_pass,
        _total_checks,
        _total_cost_b,
        _verdict_label,
    )


# ─── ACT II: PREDICTION REVEAL ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    _constraints,
    _system_valid,
    _total_pass,
    _total_checks,
    act2_pred,
    mo,
):
    _constraint_names_failing = [k for k, v in _constraints.items() if not v]

    _feedback_map = {
        "A": (
            "**Training constraint.** "
            "Memory is real: a 1T parameter model in full training state requires 20 TB. "
            "Without 3D parallelism, no H100 (80 GB) cluster of any size can hold it in "
            "per-GPU memory. The OOM constraint is structural, not solvable by adding GPUs "
            "without also changing the sharding strategy. However, with 3D parallelism "
            "and sufficient sharding, the training constraint *can* be satisfied — it is "
            "not the binding limit at reasonable cluster sizes."
        ),
        "B": (
            "**Serving constraint.** "
            "P99 < 500ms at 5B users is achievable — but only by combining cloud, edge, "
            "and mobile tiers. Serving all requests at cloud P99 would require enormous "
            "replica counts to keep utilization below the tail-latency cliff. "
            "Edge and mobile offloading (quantized models at INT4/INT2) are the architectural "
            "levers that make the P99 SLO achievable within budget. This is the design "
            "insight: serving is tractable *if* you use the full tier hierarchy."
        ),
        "C": (
            "**Privacy constraint.** "
            "GDPR-grade DP (ε ≤ 1) does impose accuracy degradation — approximately 5% at ε=1. "
            "This is significant but not fatal. The binding privacy constraint is not "
            "accuracy degradation but *data residency*: EU data cannot be used to train "
            "a centralized model without GDPR compliance, so federated learning with "
            "LoRA adapter updates is architecturally required regardless of ε. "
            "Privacy is a hard structural constraint, not just an accuracy tax."
        ),
        "D": (
            "**All constraints satisfiable.** "
            "You are correct in spirit: with the right architectural choices, all constraints "
            "can be satisfied simultaneously. But 'simultaneously' is doing a lot of work. "
            "Each satisfying configuration requires a specific combination: 3D parallelism "
            "for training, tier-aware serving for P99, Young-Daly optimal checkpointing, "
            "GDPR-compliant federated strategy for EU, per-jurisdiction fairness models, "
            "and ≥90% renewable PPA for carbon. The constraints are navigable — but they "
            "are not independent. Every architectural choice propagates to multiple constraints. "
            "That is the meta-principle."
        ),
    }

    _chosen = _feedback_map.get(act2_pred.value, _feedback_map["D"])

    if _system_valid:
        _status = mo.callout(mo.md(
            f"**Feasible architecture found.** {_total_pass}/{_total_checks} constraints pass. "
            + _chosen
        ), kind="success")
    else:
        _fail_list = ", ".join(_constraint_names_failing)
        _status = mo.callout(mo.md(
            f"**Architecture infeasible.** {_total_pass}/{_total_checks} constraints pass. "
            f"Failing: **{_fail_list}**. " + _chosen
        ), kind="warn")

    _status
    return


# ─── ACT II: MATH PEEK ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The five governing equations for Act II": mo.md("""
        **Young-Daly (Fault Tolerance):**
        `T* = sqrt(2 × C / λ)`
        where C = checkpoint write time, λ = cluster failure rate = N_GPUs / MTBF_per_GPU
        — Source: @sec-fault-tolerance-young-daly

        **Little's Law (Serving):**
        `N_concurrent = λ_arrival × W_latency`
        At saturation: `Throughput_max = N_replicas / (latency × n_shards)`
        P99 ≈ 3× mean for M/M/1 queues at moderate utilization
        — Source: @sec-model-serving-littles-law

        **Roofline (Inference Latency):**
        `Tokens/sec = min(TFLOPS × MFU, BW_GBs / bytes_per_param / params)`
        At batch=1 (autoregressive decode), arithmetic intensity = 1 op/byte → bandwidth-bound
        — Source: @sec-hw-acceleration-roofline

        **Differential Privacy:**
        `ε ≥ Δf / σ` where σ = noise scale, Δf = L2 sensitivity of query
        GDPR-grade: ε ≤ 1.0; CCPA-grade: ε ≤ 3.0
        Accuracy penalty ≈ k/ε (monotone: stronger privacy = more noise = lower accuracy)
        — Source: @sec-security-privacy-dp

        **Chouldechova Impossibility:**
        When base rates differ between groups A and B (p_A ≠ p_B), any classifier
        satisfying calibration AND equalized odds must have FPR_A ≠ FPR_B.
        No ML improvement can resolve this — it is an algebraic identity.
        — Source: @sec-responsible-ai-chouldechova
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN LEDGER SAVE + HUD
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(
    COLORS,
    _accuracy_penalty_pct,
    _avail_ok,
    _carbon_ok,
    _eu_ok,
    _gdpr_ok,
    _is_federated,
    _N,
    _oom,
    _overhead_ok,
    _slo_ok,
    _system_valid,
    _total_cost_b,
    _total_gpus,
    _total_pass,
    _total_checks,
    _T_min,
    _verdict_label,
    act1_pred,
    act2_pred,
    d1_parallelism,
    d4_epsilon,
    d5_fairness,
    ledger,
    mo,
):
    # ── Invariants applied in this lab ────────────────────────────────────────
    _invariants = [
        "Young-Daly Optimal Checkpoint",
        "Amdahl Scale Ceiling",
        "Roofline Bandwidth Bound",
        "Little's Law Serving",
        "Differential Privacy epsilon-delta",
        "Chouldechova Impossibility",
        "Jevons Carbon Paradox",
        "Memory Wall / OOM",
    ]

    # ── Save to Design Ledger ─────────────────────────────────────────────────
    ledger.save(
        chapter="v2_17",
        design={
            "context":                "full_fleet",
            "cluster_gpus":           int(_N),
            "parallelism_strategy":   d1_parallelism.value,
            "checkpoint_interval_min": float(_T_min),
            "dp_epsilon":             float(d4_epsilon.value),
            "fairness_criterion":     d5_fairness.value,
            "carbon_compliant":       bool(_carbon_ok),
            "p99_slo_met":            bool(_slo_ok),
            "total_system_cost_b":    float(_total_cost_b),
            "act1_prediction":        str(act1_pred.value),
            "act1_correct":           False,   # no single correct answer in Act I
            "act2_result":            "feasible" if _system_valid else "infeasible",
            "act2_decision":          d1_parallelism.value,
            "constraint_hit":         not _system_valid,
            "system_valid":           bool(_system_valid),
            "invariants_connected":   _invariants,
        }
    )

    # ── HUD footer ────────────────────────────────────────────────────────────
    _checks_list = [
        ("Training: No OOM",         not _oom),
        ("Checkpoint Overhead OK",   _overhead_ok),
        ("P99 < 500ms SLO",          _slo_ok),
        ("Availability 99.99%",      _avail_ok),
        ("GDPR ε Compliance",        _gdpr_ok),
        ("EU AI Act Fairness",       _eu_ok),
        ("Carbon Neutral 2027",      _carbon_ok),
    ]

    _badge_html = "".join([
        f"""<span style="background: {'rgba(0,143,69,0.15)' if ok else 'rgba(203,32,45,0.15)'};
                         color: {'#4ade80' if ok else '#f87171'};
                         border: 1px solid {'rgba(0,143,69,0.35)' if ok else 'rgba(203,32,45,0.35)'};
                         padding: 4px 10px; border-radius: 20px; font-size: 0.75rem;
                         font-weight: 600; margin: 3px;">
            {'&#x2713;' if ok else '&#x274C;'} {label}
        </span>"""
        for label, ok in _checks_list
    ])

    _hud = mo.Html(f"""
    <div style="background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 100%);
                border-radius: 12px; padding: 20px 28px; margin-top: 24px;
                border: 1px solid rgba(99,102,241,0.25);">
        <div style="display: flex; justify-content: space-between; align-items: center;
                    flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
            <div>
                <div style="font-size: 0.65rem; font-weight: 700; color: #475569;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;">
                    Design Ledger · Chapter v2_17 · Capstone
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #f1f5f9;">
                    Planet-Scale Architecture: {_verdict_label}
                    &nbsp;&nbsp;{_total_pass}/{_total_checks} constraints passed
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 2px;">
                    Annual cost estimate
                </div>
                <div style="font-size: 1.4rem; font-weight: 800;
                            color: {'#4ade80' if _total_cost_b < 10 else '#f87171'};">
                    ${_total_cost_b:.1f}B / yr
                </div>
            </div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 16px;">
            {_badge_html}
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.07); padding-top: 14px;
                    font-size: 0.78rem; color: #64748b; line-height: 1.8;">
            <strong style="color:#94a3b8;">Invariants applied:</strong>
            {" &middot; ".join(_invariants)}
        </div>
    </div>
    """)
    _hud
    return


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM SYNTHESIS — THE META-PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## The Meta-Principle

    Every invariant in this curriculum traces back to the same root cause:
    **physical laws create hard ceilings that no amount of engineering can dissolve.**

    You cannot wish away the memory wall — HBM bandwidth is determined by signal
    physics and pin count. You cannot wish away Amdahl's Law — coordination cost
    grows with cluster size regardless of how good your scheduler is. You cannot
    wish away Chouldechova's theorem — it follows from the definition of conditional
    probability. You cannot wish away Young-Daly — it follows from the calculus of
    minimization. You cannot wish away Little's Law — it follows from queueing theory
    steady-state.

    But you *can* navigate these constraints. That is the discipline of ML systems:
    not finding a way around the physics, but designing systems that respect it.

    The skilled ML architect does not ask: "How do I avoid the memory wall?"
    They ask: "Which memory-wall-respecting architecture best satisfies my
    throughput, latency, and cost requirements simultaneously?"

    That is the question this curriculum trained you to ask.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Key Takeaways

    1. **Constraints are not obstacles; they are the specification.**
       Every architectural choice in ML systems is ultimately a choice about
       which constraint to prioritize when they cannot all be satisfied simultaneously.
       The invariants give you the exact tradeoff surface. Read them.

    2. **The gap between your prediction and reality is your learning.**
       If your Systems Intuition Radar shows a weak domain, that is not a failure —
       it is a calibration report. The researchers who built the fastest training
       systems, the most efficient serving pipelines, and the fairest production
       models were the ones who had internalized which invariants bind when,
       and why. That intuition is now yours to develop.
    """)
    return


if __name__ == "__main__":
    app.run()
