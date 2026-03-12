import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-17: THE CONSTRAINTS NEVER LIE
#
# Volume II, Chapter 17 — Conclusion (Capstone)
#
# Core Invariant: Synthesis of ALL Vol1 + Vol2 invariants.
#   The physics does not change — the bottleneck moves with scale.
#   No single architectural decision satisfies all constraints independently.
#   The skilled ML architect does not escape the invariants; they navigate them.
#
# 2-Act Structure (35-40 minutes):
#   Act I  — Design Ledger Archaeology (12-15 min)
#     Read ALL prior ledger entries. Surface constraint frequency. Radar of
#     prediction accuracy. Commit to which invariant category you violated most.
#
#   Act II — The Final Architecture Challenge (20-25 min)
#     Scenario: Chief Architect for real-time medical image classification.
#     1,000 hospitals · 100k inferences/day each · ≥95% accuracy · P99 < 200ms
#     DP ε ≤ 1 (HIPAA) · >40% carbon reduction · 99.9% uptime
#     Adversarial robustness ≥ 50% (PGD) · Budget: 10,000 H100s
#     6 simultaneous constraint scorecards. All must be green to deploy.
#
# Deployment Context: Full Fleet (cloud + medical grade)
#
# Design Ledger: saves chapter="v2_17"
#   Keys match the capstone schema in the assignment spec.
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

    # ── Cloud fleet hardware constants ────────────────────────────────────────
    H100_BW_GBS       = 3350    # GB/s HBM3e; NVIDIA H100 SXM5 spec
    H100_TFLOPS_FP16  = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80      # GB HBM3e; NVIDIA spec
    H100_TDP_W        = 700     # Watts TDP; NVIDIA spec

    # ── Fleet scale constants ──────────────────────────────────────────────
    FLEET_SIZE_NODES  = 1000    # nodes in synthesis scenario; assignment spec
    GPUS_PER_NODE     = 8       # H100 SXM5 per node; NVIDIA DGX H100 config
    CHECKPOINT_COST_S = 120     # seconds per checkpoint (1 TB model, NVMe); spec
    MTBF_GPU_HOURS    = 2000    # mean time between GPU failures (hours); spec

    # ── NVLink / InfiniBand ────────────────────────────────────────────────
    NVLINK_BW_GBS     = 900     # GB/s NVLink4 bidirectional per GPU; NVIDIA spec
    IB_BW_GBPS        = 400     # Gb/s InfiniBand NDR per port; Mellanox NDR spec

    # ── Carbon constants ───────────────────────────────────────────────────
    COAL_CI_G_KWH     = 820     # g CO2/kWh coal-heavy grid; IEA 2024
    RENEW_CI_G_KWH    = 40      # g CO2/kWh renewable PPA; hyperscaler estimate
    # Mixed global fleet baseline: @sec-sustainable-ai
    BASELINE_CI_G_KWH = 386     # g CO2/kWh global fleet avg (mixed grid); spec

    # ── Medical classification scenario ───────────────────────────────────
    HOSPITAL_COUNT    = 1000    # hospitals in deployment scope; spec
    INF_PER_DAY       = 100_000 # inferences per hospital per day; spec
    P99_SLO_MS        = 200     # P99 latency SLO, milliseconds; spec
    ACCURACY_TARGET   = 0.95    # ≥95% accuracy; HIPAA-grade clinical requirement
    DP_EPS_LIMIT      = 1.0     # ε ≤ 1 for HIPAA differential privacy; spec
    ADV_ROBUSTNESS_TARGET = 0.50  # ≥50% accuracy under PGD attack; spec
    CARBON_REDUCTION_TARGET = 0.40  # >40% reduction vs baseline; spec
    UPTIME_TARGET     = 0.999   # 99.9% uptime; spec
    BUDGET_GPUS       = 10_000  # H100 GPU budget; spec

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        FLEET_SIZE_NODES, GPUS_PER_NODE, CHECKPOINT_COST_S, MTBF_GPU_HOURS,
        NVLINK_BW_GBS, IB_BW_GBPS,
        COAL_CI_G_KWH, RENEW_CI_G_KWH, BASELINE_CI_G_KWH,
        HOSPITAL_COUNT, INF_PER_DAY, P99_SLO_MS, ACCURACY_TARGET,
        DP_EPS_LIMIT, ADV_ROBUSTNESS_TARGET, CARBON_REDUCTION_TARGET,
        UPTIME_TARGET, BUDGET_GPUS,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

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
                Machine Learning Systems &middot; Volume II &middot; Lab 17 &middot; Capstone
            </div>
            <h1 style="margin: 0 0 12px 0; font-size: 2.6rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.05; letter-spacing: -0.03em;">
                The Constraints Never Lie
            </h1>
            <p style="margin: 0 0 24px 0; font-size: 1.08rem; color: #94a3b8;
                      max-width: 700px; line-height: 1.7;">
                You have traversed two volumes of ML systems. Every invariant you encountered
                reduces to one meta-principle: <strong style="color:#a5b4fc;">constraints
                drive architecture.</strong> The memory wall, Amdahl&#x2019;s Law, Young-Daly,
                Little&#x2019;s Law, Chouldechova, the DP &#x03B5;-&#x03B4; tradeoff,
                adversarial robustness &mdash; none can be wished away. Only navigated.
                Your final task: audit your own journey, then architect a production system
                that must satisfy all of them simultaneously.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    Act I: Design Ledger Archaeology &middot; Act II: Medical Fleet Architecture
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: All Vol 1 + Vol 2 chapters
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.3);">
                    6 Active Constraint Scorecards
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
                        margin-top: 8px;">
                <div style="background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: {_c_cloud}; font-weight: 700;">Memory</span>
                    <div style="color: #94a3b8; margin-top: 2px;">D&middot;A&middot;M Triad &middot; Roofline</div>
                </div>
                <div style="background: rgba(203,32,45,0.10); border: 1px solid rgba(203,32,45,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: #f87171; font-weight: 700;">Scale</span>
                    <div style="color: #94a3b8; margin-top: 2px;">Amdahl &middot; Parallelism Paradox</div>
                </div>
                <div style="background: rgba(204,85,0,0.10); border: 1px solid rgba(204,85,0,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: #fb923c; font-weight: 700;">Reliability</span>
                    <div style="color: #94a3b8; margin-top: 2px;">Young-Daly &middot; Little&#x2019;s Law</div>
                </div>
                <div style="background: rgba(0,143,69,0.10); border: 1px solid rgba(0,143,69,0.3);
                            border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;">
                    <span style="color: #4ade80; font-weight: 700;">Ethics</span>
                    <div style="color: #94a3b8; margin-top: 2px;">Chouldechova &middot; DP &#x03B5;-&#x03B4;</div>
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']}; background: #ffffff;
                border-radius: 0 10px 10px 0; padding: 20px 26px; margin: 12px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 14px;">
            Lab Briefing &nbsp;&middot;&nbsp; Capstone &nbsp;&middot;&nbsp; 35–40 minutes
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: {COLORS['TextSec']};
                            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                    Learning Objectives
                </div>
                <ol style="margin: 0; padding-left: 18px; color: {COLORS['Text']};
                           font-size: 0.88rem; line-height: 1.75;">
                    <li>Identify which constraint domain appeared most frequently in your
                        Design Ledger and explain why scale shifts the binding constraint
                        from compute to communication</li>
                    <li>Predict the orchestration multiplier required to achieve 100&times;
                        system efficiency given hardware&nbsp;=&nbsp;4&times; and
                        algorithm&nbsp;=&nbsp;2.5&times;</li>
                    <li>Design a fleet configuration satisfying all six simultaneous
                        constraints for 1,000 hospitals under HIPAA, carbon cap,
                        adversarial robustness, and 99.9% uptime requirements</li>
                </ol>
            </div>
            <div>
                <div style="font-size: 0.75rem; font-weight: 700; color: {COLORS['TextSec']};
                            text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
                    Prerequisites
                </div>
                <div style="color: {COLORS['TextSec']}; font-size: 0.85rem; line-height: 1.7;">
                    All Vol&nbsp;1 + Vol&nbsp;2 chapters. Key laws in scope:
                    Amdahl, Young-Daly, Chouldechova impossibility, DP &epsilon;-&delta;,
                    Jevons Paradox, Ring AllReduce bandwidth, Little&rsquo;s Law.
                </div>
                <div style="margin-top: 14px; padding: 10px 14px;
                            background: {COLORS['Surface2']}; border-radius: 6px;
                            border: 1px solid {COLORS['Border']};">
                    <div style="font-size: 0.75rem; font-weight: 700; color: {COLORS['BlueLine']};
                                margin-bottom: 4px;">Core Question</div>
                    <div style="font-size: 0.84rem; color: {COLORS['Text']}; line-height: 1.6;
                                font-style: italic;">
                        After optimizing memory, compute, networking, privacy, sustainability,
                        and fairness in isolation across 16 chapters, which constraint proves
                        hardest to satisfy simultaneously — and what does achieving all six
                        at once require that no individual optimization prepared you for?
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-conclusion-constraints-drive-architecture** — The meta-principle unifying
      all two-volume invariants; why physical laws cannot be abstracted away
    - **@sec-conclusion-vol1-synthesis** — Summary of the 8 invariant families from
      Volume I and how they compose at scale
    - **@sec-conclusion-vol2-synthesis** — Distributed systems invariants; the emergent
      constraints that only appear at fleet scale
    - **@sec-conclusion-planet-scale** — Case study of hyperscaler architectural
      decisions viewed through the lens of competing constraints
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT I: SECTION HEADER ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_color = COLORS["Cloud"]
    _act_why = (
        "At 1,000 GPUs, a 10% drop in network bandwidth costs more throughput than a "
        "10% drop in FLOPS — the bottleneck is not where you think it is."
    )
    mo.Html(f"""
    <div style="border-top: 2px solid {_act_color}; margin: 28px 0 20px 0; padding-top: 18px;">
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 10px;">
            <div style="width: 38px; height: 38px; border-radius: 50%;
                        background: {_act_color}; color: #fff;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 1.1rem; font-weight: 800; flex-shrink: 0;">I</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {COLORS['Text']};">
                    Act I &mdash; Design Ledger Archaeology
                </div>
                <div style="font-size: 0.78rem; color: {COLORS['TextMuted']}; margin-top: 2px;">
                    Calibration &middot; 12&ndash;15 minutes
                </div>
            </div>
        </div>
        <div style="font-size: 0.87rem; color: {COLORS['TextSec']}; line-height: 1.65;
                    padding-left: 52px; font-style: italic;">
            {_act_why}
        </div>
    </div>
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
            Incoming Message &middot; Chief Architect &middot; AI Infrastructure
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "You have spent two volumes designing ML systems. Before I promote you to
            Principal Engineer, I need you to audit the constraints YOU violated during
            training. Pull your Design Ledger and tell me: which constraints appeared most
            frequently, and what architectural pattern would have prevented the most failures?"
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT FRAMING ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The two-volume curriculum introduced invariant families spanning every layer of the ML
    systems stack. The table below maps each domain to the labs where it was tested:

    | Domain | Core Invariant | Vol 1 Labs | Vol 2 Labs |
    |---|---|---|---|
    | **Memory** | Memory Wall: bandwidth &ll; compute peak | 05, 08, 10 | — |
    | **Compute** | Roofline / MFU ceiling | 11, 12 | 09 |
    | **Serving** | Little&rsquo;s Law: N = &lambda;W; P99 &ne; avg | 13 | 10 |
    | **Scale** | Amdahl; Parallelism Paradox | — | 01, 05 |
    | **Networking** | AllReduce BW; Bisection BW | — | 02, 03, 06 |
    | **Reliability** | Young-Daly: T* = &radic;(2C/&lambda;) | — | 07 |
    | **Privacy & Ethics** | Chouldechova impossibility; DP &epsilon;-accuracy | 15, 16 | 13, 16 |
    | **Economics** | Jevons Paradox; utilization vs. queue latency | — | 08, 09, 15 |

    The bar chart below is your *constraint frequency report* — how often your design choices
    triggered a failure state in each domain. Before seeing it, commit to a hypothesis.
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
            "A) Memory bandwidth was the most common constraint across my labs": "A",
            "B) Parallelism communication overhead was the most common constraint": "B",
            "C) Power and thermal constraints dominated at fleet scale": "C",
            "D) The constraint varied — no single constraint dominates; it depends on scale": "D",
        },
        label="Which constraint category appeared most frequently in your Design Ledger?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(act1_pred, mo):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Design Ledger Archaeology."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT I: LEDGER ARCHAEOLOGY ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Design Ledger Archaeology")
    return


@app.cell(hide_code=True)
def _(COLORS, go, ledger, mo, np, apply_plotly_theme):
    # ── Read all prior ledger entries ─────────────────────────────────────────
    _history = ledger._state.history if hasattr(ledger._state, "history") else []

    # Build chapter → design map
    _ledger_map = {}
    for _entry in _history:
        _ch = str(_entry.get("chapter", ""))
        _design = _entry.get("design", {})
        _ledger_map[_ch] = _design

    # ── Domain → chapter membership ──────────────────────────────────────────
    # Maps domain → chapter keys (Vol1 are plain integers as strings, Vol2 are "v2_NN")
    _domain_chapters = {
        "Memory":      ["5", "8", "10"],
        "Compute":     ["11", "12", "v2_09"],
        "Serving":     ["13", "v2_10"],
        "Scale":       ["v2_01", "v2_05"],
        "Networking":  ["v2_02", "v2_03", "v2_06"],
        "Reliability": ["v2_07"],
        "Privacy/Ethics": ["15", "16", "v2_13", "v2_16"],
        "Economics":   ["v2_08", "v2_09", "v2_15"],
    }

    # ── Compute per-domain stats ───────────────────────────────────────────────
    _domain_constraint_hits = {}
    _domain_accuracy        = {}
    _domain_labs_done       = {}

    for _domain, _chapters in _domain_chapters.items():
        _hit_list     = []
        _correct_list = []
        for _ch in _chapters:
            if _ch in _ledger_map:
                _d = _ledger_map[_ch]
                if "constraint_hit" in _d:
                    _hit_list.append(1.0 if _d["constraint_hit"] else 0.0)
                if "act1_correct" in _d:
                    _correct_list.append(1.0 if _d["act1_correct"] else 0.0)
        _domain_constraint_hits[_domain] = (
            sum(_hit_list) / len(_hit_list) if _hit_list else 0.0
        )
        _domain_accuracy[_domain] = (
            sum(_correct_list) / len(_correct_list) if _correct_list else 0.5
        )
        _domain_labs_done[_domain] = len([c for c in _chapters if c in _ledger_map])

    # ── Summary statistics ────────────────────────────────────────────────────
    _total_labs  = len(_history)
    _total_hits  = sum(
        1 for e in _history if e.get("design", {}).get("constraint_hit", False)
    )

    _vol1_chs    = [str(c) for c in range(1, 17)]
    _vol2_chs    = [f"v2_{c:02d}" for c in range(1, 17)]

    _vol1_correct = [
        1.0 if _ledger_map.get(c, {}).get("act1_correct", False) else 0.0
        for c in _vol1_chs if c in _ledger_map
    ]
    _vol2_correct = [
        1.0 if _ledger_map.get(c, {}).get("act1_correct", False) else 0.0
        for c in _vol2_chs if c in _ledger_map
    ]
    _vol1_acc    = sum(_vol1_correct) / max(len(_vol1_correct), 1) * 100
    _vol2_acc    = sum(_vol2_correct) / max(len(_vol2_correct), 1) * 100
    _overall_acc = (sum(_vol1_correct) + sum(_vol2_correct)) / max(
        len(_vol1_correct) + len(_vol2_correct), 1
    ) * 100

    _weakest_domain = min(_domain_accuracy, key=lambda d: _domain_accuracy[d])
    _most_hit_domain = max(
        _domain_constraint_hits, key=lambda d: _domain_constraint_hits[d]
    )

    # Top 3 most-violated domains (by constraint_hit rate)
    _sorted_domains = sorted(
        _domain_constraint_hits.items(), key=lambda x: x[1], reverse=True
    )
    _top3 = _sorted_domains[:3]

    # ── Horizontal bar chart: constraint hit frequency ────────────────────────
    _domains_sorted = [d for d, _ in sorted(
        _domain_constraint_hits.items(), key=lambda x: x[1]
    )]
    _hits_sorted    = [_domain_constraint_hits[d] * 100 for d in _domains_sorted]
    _bar_colors     = [
        COLORS["RedLine"] if h >= 60 else (
            COLORS["OrangeLine"] if h >= 30 else COLORS["GreenLine"]
        )
        for h in _hits_sorted
    ]

    _fig_bar = go.Figure(go.Bar(
        x=_hits_sorted,
        y=_domains_sorted,
        orientation="h",
        marker_color=_bar_colors,
        text=[f"{h:.0f}%" for h in _hits_sorted],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["TextSec"]),
    ))
    _fig_bar.update_layout(
        height=320,
        xaxis=dict(
            title="Constraint hit rate (%)",
            range=[0, 115],
            gridcolor="#f1f5f9",
            tickfont=dict(size=10),
        ),
        yaxis=dict(tickfont=dict(size=11, color=COLORS["TextSec"])),
        margin=dict(t=50, b=40, l=130, r=60),
        title=dict(
            text="Constraint Hit Frequency by Domain (from your Design Ledger)",
            font=dict(size=13, color=COLORS["Text"]),
            x=0.5,
        ),
    )
    apply_plotly_theme(_fig_bar)

    # ── Radar chart: prediction accuracy by domain ────────────────────────────
    _radar_domains  = list(_domain_accuracy.keys())
    _radar_scores   = [_domain_accuracy[d] * 100 for d in _radar_domains]
    _radar_closed   = _radar_scores  + [_radar_scores[0]]
    _theta_closed   = _radar_domains + [_radar_domains[0]]

    _fig_radar = go.Figure()
    _fig_radar.add_trace(go.Scatterpolar(
        r=[100] * (len(_radar_domains) + 1),
        theta=_theta_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.06)",
        line=dict(color=COLORS["Cloud"], width=1, dash="dot"),
        name="Perfect (100%)",
    ))
    _fig_radar.add_trace(go.Scatterpolar(
        r=_radar_closed,
        theta=_theta_closed,
        fill="toself",
        fillcolor="rgba(0,143,69,0.12)",
        line=dict(color=COLORS["GreenLine"], width=2.5),
        name="Your prediction accuracy",
        marker=dict(size=8, color=COLORS["GreenLine"]),
    ))
    _fig_radar.update_layout(
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
                tickfont=dict(size=10, color=COLORS["TextSec"]),
                gridcolor=COLORS["Border"],
            ),
            bgcolor="rgba(248,250,252,0.6)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        height=400,
        margin=dict(t=40, b=60, l=40, r=40),
        title=dict(
            text="Systems Intuition Radar — Prediction Accuracy by Domain",
            font=dict(size=12, color=COLORS["Text"]),
            x=0.5,
        ),
    )
    apply_plotly_theme(_fig_radar)

    # ── Summary metric cards ───────────────────────────────────────────────────
    _v1_color = (
        COLORS["GreenLine"] if _vol1_acc >= 70 else
        COLORS["OrangeLine"] if _vol1_acc >= 50 else COLORS["RedLine"]
    )
    _v2_color = (
        COLORS["GreenLine"] if _vol2_acc >= 70 else
        COLORS["OrangeLine"] if _vol2_acc >= 50 else COLORS["RedLine"]
    )
    _ov_color = (
        COLORS["GreenLine"] if _overall_acc >= 70 else
        COLORS["OrangeLine"] if _overall_acc >= 50 else COLORS["RedLine"]
    )

    _summary = mo.Html(f"""
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
                Constraints Hit
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['OrangeLine']};">
                {_total_hits}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                of {_total_labs} labs
            </div>
        </div>
        <div class="lab-card" style="text-align: center; padding: 18px 12px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Vol I Accuracy
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {_v1_color};">
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
            <div style="font-size: 2.2rem; font-weight: 800; color: {_v2_color};">
                {_vol2_acc:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {len(_vol2_correct)} labs sampled
            </div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 0;">
        <div class="lab-card" style="padding: 14px 16px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Most-Violated Domain
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['RedLine']};">
                {_most_hit_domain}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {_domain_constraint_hits[_most_hit_domain]*100:.0f}% of labs triggered
            </div>
        </div>
        <div class="lab-card" style="padding: 14px 16px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Weakest Prediction Domain
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {COLORS['OrangeLine']};">
                {_weakest_domain}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">
                {_domain_accuracy[_weakest_domain]*100:.0f}% prediction accuracy
            </div>
        </div>
        <div class="lab-card" style="padding: 14px 16px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Overall Accuracy
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: {_ov_color};">
                {_overall_acc:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.8rem;">combined</div>
        </div>
    </div>
    """)

    mo.vstack([
        _summary,
        mo.hstack([
            mo.ui.plotly(_fig_bar),
            mo.ui.plotly(_fig_radar),
        ], justify="center", gap=2),
    ])
    return (
        _domain_accuracy,
        _domain_constraint_hits,
        _domain_labs_done,
        _most_hit_domain,
        _weakest_domain,
        _overall_acc,
        _vol1_acc,
        _vol2_acc,
        _total_labs,
        _total_hits,
        _ledger_map,
        _top3,
    )


# ─── ACT I: PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    _domain_accuracy,
    _domain_constraint_hits,
    _most_hit_domain,
    _top3,
    act1_pred,
    mo,
):
    # D is the correct answer — the bottleneck shifts with scale
    _correct = act1_pred.value == "D"

    _domain_top3_str = ", ".join(f"**{d}** ({r*100:.0f}%)" for d, r in _top3)

    _feedback_map = {
        "A": (
            "**Memory bandwidth** is among the most persistent constraints in single-node "
            "and inference workloads. The H100's arithmetic intensity ridge point "
            "(989 TFLOPS / 3,350 GB/s = ~295 FLOP/byte) means that most token-generation "
            "workloads are bandwidth-bound at batch=1. You were correct that memory "
            "dominates in many labs — but the ledger reveals it does not dominate "
            "*all* labs. At fleet scale, network fabric and checkpoint overhead become "
            "binding earlier. The constraint moves."
        ),
        "B": (
            "**Communication overhead** is genuinely severe at scale: ring AllReduce "
            "over InfiniBand (400 Gb/s) carries a gradient tensor that can exceed 2 TB "
            "for a 1T-parameter model. At 8,000+ GPUs, communication can consume "
            "20-40% of total training time. But your ledger likely shows this only "
            "became the dominant constraint in Vol 2 networking and distributed training "
            "labs. In Vol 1 — single-node workloads — it barely registers. "
            "The constraint moves with scale."
        ),
        "C": (
            "**Power and thermal constraints** bind at cluster level but are rarely the "
            "first failure mode in individual lab scenarios. A 10,000-GPU cluster "
            "draws 7 MW; carbon compliance is a real concern at fleet scale. "
            "But your ledger likely shows thermal constraints appear primarily in the "
            "sustainability labs, not across the full curriculum. The constraint moves "
            "with the deployment tier."
        ),
        "D": (
            "**Correct.** The constraint varies with scale, workload, and deployment tier. "
            "Your ledger confirms this: the top three hit domains are "
            f"{_domain_top3_str}. "
            "Each was most relevant in a specific context. Memory dominates in "
            "single-node inference. Communication dominates in multi-node training. "
            "Fairness constraints activate regardless of scale but are invisible until "
            "evaluated across populations. The meta-principle is not *which* constraint "
            "is hardest — it is that the bottleneck *moves*, and the architect who "
            "cannot see it move will be surprised by every system that scales."
        ),
    }

    _chosen = _feedback_map.get(act1_pred.value, _feedback_map["D"])

    _note = (
        f" Your ledger also shows **{_most_hit_domain}** as your highest-hit domain "
        f"({_domain_constraint_hits[_most_hit_domain]*100:.0f}% hit rate). "
        f"Your weakest prediction accuracy was in **{min(_domain_accuracy, key=lambda d: _domain_accuracy[d])}** "
        f"({_domain_accuracy[min(_domain_accuracy, key=lambda d: _domain_accuracy[d])]*100:.0f}%) — "
        f"which is where you had the most to learn."
    )

    mo.callout(
        mo.md(_chosen + _note),
        kind="success" if _correct else "info",
    )
    return


# ─── ACT I: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Reflection")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) More hardware always solves the constraint — scale cures all bottlenecks": "A",
            "B) Every system is defined by its most constrained resource — the laws don't change, but the bottleneck moves": "B",
            "C) Software optimization is always preferable to hardware scaling": "C",
            "D) The only invariant is that all constraints are temporary": "D",
        },
        label="What architectural principle unifies ALL the invariants you encountered?",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(act1_reflection, mo):
    if act1_reflection.value is None:
        mo.callout(
            mo.md("Select your reflection answer above to continue."),
            kind="warn",
        )
    elif act1_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** The bottleneck moves, but it never disappears. "
            "The Iron Law T = D/BW + O/R + L tells you three things that can "
            "limit latency. Roofline tells you two things that can limit compute. "
            "Amdahl tells you the ceiling on parallelism. Young-Daly tells you "
            "the optimal checkpoint interval. Chouldechova tells you the minimum "
            "fairness gap you must accept. None of these is 'temporary.' They are "
            "all expressions of the same underlying constraint: **physics drives architecture.**"
        ), kind="success")
    elif act1_reflection.value == "A":
        mo.callout(mo.md(
            "**Incorrect.** Adding hardware shifts the bottleneck but does not remove it. "
            "Amdahl's Law shows that the serial fraction of your workload caps speedup "
            "regardless of cluster size. The communication overhead of AllReduce *grows* "
            "with cluster size. The carbon footprint *grows* with hardware count. "
            "More hardware is a tool, not a solution."
        ), kind="warn")
    elif act1_reflection.value == "C":
        mo.callout(mo.md(
            "**Incorrect.** Software optimization is powerful — kernel fusion, "
            "continuous batching, and mixed-precision training all improve MFU "
            "substantially. But no software optimization escapes the Roofline "
            "ceiling or removes Amdahl's serial fraction. At some point, "
            "the physics imposes a hard limit that no optimizer can cross."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Incorrect.** Physical constraints are not temporary. "
            "The memory wall is determined by signal physics and HBM pin density — "
            "it has been 'temporary' for 30 years and remains. Chouldechova's "
            "theorem follows from conditional probability and will not be repealed "
            "by better hardware. Young-Daly follows from calculus. "
            "The constraints are permanent; only your architecture adapts."
        ), kind="warn")
    return


# ─── ACT I: MATHPEEK ACCORDION ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — all eight invariant families": mo.md("""
        **Iron Law (Latency):**
        `T = D/BW + O/R + L`
        — T = latency; D = data transferred; BW = bandwidth; O = operations; R = throughput; L = pipeline latency
        — Source: @sec-ml-systems-iron-law

        **Memory Anatomy (Training State):**
        `M_total = weights + gradients + optimizer_state + activations`
        — FP16 mixed precision: 2+2+8 = 12 bytes/param minimum; with activations varies by batch
        — Source: @sec-training-memory-anatomy

        **Roofline (Attainable Performance):**
        `Attainable_FLOPS = min(Peak_FLOPS, BW_GBs × Arithmetic_Intensity)`
        — Ridge point = Peak_FLOPS / BW; below ridge = bandwidth-bound
        — Source: @sec-hw-acceleration-roofline

        **Amdahl's Law (Scale Ceiling):**
        `Speedup(N) = 1 / (S + (1 - S)/N)`
        — S = serial fraction; maximum speedup = 1/S regardless of N
        — Source: @sec-distributed-training-amdahl

        **Little's Law (Serving Throughput):**
        `L = lambda × W`
        — L = in-flight requests; lambda = arrival rate; W = mean latency
        — Source: @sec-model-serving-littles-law

        **Young-Daly (Optimal Checkpoint Interval):**
        `T* = sqrt(2 × C / lambda)`
        — C = checkpoint write cost; lambda = cluster failure rate = N / MTBF_per_device
        — Source: @sec-fault-tolerance-young-daly

        **Jevons Paradox (Carbon):**
        `Delta_C = Energy × Intensity × (scale_up - efficiency_gain)`
        — Efficiency improvements can be consumed by demand growth; net carbon rises
        — Source: @sec-sustainable-ai-jevons

        **SLO Composition (Reliability):**
        `P(e2e_failure) = 1 - product_i(p_i)`
        — Approximate for independent services; cascade amplifies tail failures
        — Source: @sec-ops-scale-slo-composition
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT II: SECTION HEADER ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_color = COLORS["OrangeLine"]
    _act_why = (
        "Hardware gives 4x. Algorithms give 2.5x. Together that is only 10x of the "
        "100x target — the remaining 10x must come from orchestration, which no single "
        "chapter prepared you to deliver alone."
    )
    mo.Html(f"""
    <div style="border-top: 2px solid {_act_color}; margin: 28px 0 20px 0; padding-top: 18px;">
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 10px;">
            <div style="width: 38px; height: 38px; border-radius: 50%;
                        background: {_act_color}; color: #fff;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 1.1rem; font-weight: 800; flex-shrink: 0;">II</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {COLORS['Text']};">
                    Act II &mdash; The Final Architecture Challenge
                </div>
                <div style="font-size: 0.78rem; color: {COLORS['TextMuted']}; margin-top: 2px;">
                    Design Challenge &middot; 20&ndash;25 minutes
                </div>
            </div>
        </div>
        <div style="font-size: 0.87rem; color: {COLORS['TextSec']}; line-height: 1.65;
                    padding-left: 52px; font-style: italic;">
            {_act_why}
        </div>
    </div>
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
            Incoming Message &middot; Chief Architect &middot; Medical AI Division &middot; URGENT
        </div>
        <div style="font-style: italic; font-size: 1.02rem; color: #1e293b; line-height: 1.7;">
            "Design a production ML system for real-time medical image classification.
            Requirements: <strong>1,000 hospitals</strong>, <strong>100,000 inferences/day
            each</strong>, <strong>&ge;95% accuracy</strong>, <strong>P99 &lt; 200ms</strong>,
            <strong>DP &epsilon; &le; 1 (HIPAA)</strong>, <strong>&gt;40% carbon reduction
            vs. baseline</strong>, <strong>fault tolerance for 99.9% uptime</strong>,
            and <strong>adversarial robustness &ge;50%</strong> on PGD attacks.
            You have a budget of <strong>10,000 H100s</strong>.
            Every constraint must be satisfied simultaneously for deployment approval."
        </div>
    </div>
    """)
    return


# ─── ACT II: CONTEXT TOGGLE ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Global Fleet (mixed grid, 386 g CO\u2082/kWh)": "fleet",
            "Carbon-Optimized (renewable, 40 g CO\u2082/kWh)": "renewable",
        },
        value="Global Fleet (mixed grid, 386 g CO\u2082/kWh)",
        label="Deployment context:",
        inline=True,
    )
    context_toggle
    return (context_toggle,)


# ─── ACT II: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) DP \u03b5 \u2264 1 is the hardest constraint — it destroys too much accuracy for clinical use": "A",
            "B) No single architecture satisfies all constraints simultaneously — requires explicit tradeoff negotiation": "B",
            "C) The fleet size (10,000 H100s) is sufficient for all constraints at stated scale": "C",
            "D) Carbon reduction is the easiest constraint to satisfy independently of the others": "D",
        },
        label="Which statement best characterizes this architecture challenge?",
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


# ─── ACT II: ARCHITECTURE SYNTHESIZER — SLIDERS ───────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Final Architecture Synthesizer")
    return


@app.cell(hide_code=True)
def _(mo):
    model_size_b = mo.ui.slider(
        start=1, stop=70, value=7, step=1,
        label="Model size (B parameters)",
        show_value=True,
    )
    dp_epsilon = mo.ui.slider(
        start=0.1, stop=10.0, value=1.0, step=0.1,
        label="Differential privacy \u03b5 (lower = stronger privacy)",
        show_value=True,
    )
    adv_train_weight = mo.ui.slider(
        start=0.0, stop=1.0, value=0.3, step=0.05,
        label="Adversarial training weight (0 = clean only, 1 = adversarial only)",
        show_value=True,
    )
    parallelism_strategy = mo.ui.radio(
        options={
            "Data Parallel only (DP)": "dp",
            "Tensor + Data Parallel (TP+DP)": "tp_dp",
            "Full 3D Parallel (DP+TP+PP)": "3d",
        },
        value="Tensor + Data Parallel (TP+DP)",
        label="Parallelism strategy:",
        inline=True,
    )
    checkpoint_interval_min = mo.ui.slider(
        start=5, stop=120, value=30, step=5,
        label="Checkpoint interval (minutes)",
        show_value=True,
    )
    flexible_job_pct = mo.ui.slider(
        start=0, stop=50, value=20, step=5,
        label="Flexible / deferrable job percentage (% of workload shifted to low-carbon hours)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([model_size_b, dp_epsilon, adv_train_weight], justify="start", gap=4),
        parallelism_strategy,
        mo.hstack([checkpoint_interval_min, flexible_job_pct], justify="start", gap=4),
    ])
    return (
        model_size_b,
        dp_epsilon,
        adv_train_weight,
        parallelism_strategy,
        checkpoint_interval_min,
        flexible_job_pct,
    )


# ─── ACT II: CONSTRAINT COMPUTATION ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    ACCURACY_TARGET,
    ADV_ROBUSTNESS_TARGET,
    BASELINE_CI_G_KWH,
    BUDGET_GPUS,
    CARBON_REDUCTION_TARGET,
    CHECKPOINT_COST_S,
    COLORS,
    DP_EPS_LIMIT,
    GPUS_PER_NODE,
    H100_BW_GBS,
    H100_RAM_GB,
    H100_TDP_W,
    H100_TFLOPS_FP16,
    HOSPITAL_COUNT,
    INF_PER_DAY,
    MTBF_GPU_HOURS,
    P99_SLO_MS,
    RENEW_CI_G_KWH,
    UPTIME_TARGET,
    adv_train_weight,
    checkpoint_interval_min,
    context_toggle,
    dp_epsilon,
    flexible_job_pct,
    math,
    mo,
    model_size_b,
    parallelism_strategy,
):
    _ctx        = context_toggle.value
    _M_B        = model_size_b.value          # billions of parameters
    _M          = _M_B * 1e9                  # raw parameter count
    _eps        = dp_epsilon.value
    _adv_w      = adv_train_weight.value
    _strategy   = parallelism_strategy.value
    _T_min      = checkpoint_interval_min.value
    _flex_pct   = flexible_job_pct.value / 100.0
    _ci         = RENEW_CI_G_KWH if _ctx == "renewable" else BASELINE_CI_G_KWH

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ACCURACY
    # Base accuracy model: larger models are more accurate (diminishing returns).
    # Reference: scaling laws @sec-training-scaling-laws
    # Approximation: accuracy ≈ 0.82 + 0.13 × (1 - exp(-M_B / 20))
    # DP noise penalty: ~k / eps, where k = 0.05 (5% at eps=1)
    # Adversarial training penalty: clean accuracy drops with adv weight
    # Reference: @sec-robust-ai-adversarial-training-tradeoff
    # ─────────────────────────────────────────────────────────────────────────
    _base_acc      = 0.82 + 0.13 * (1.0 - math.exp(-_M_B / 20.0))
    _dp_acc_penalty = min(0.05 / _eps, 0.20)      # ≤ 20% cap
    _adv_acc_penalty = _adv_w * 0.08              # up to 8% clean accuracy cost
    _accuracy      = max(_base_acc - _dp_acc_penalty - _adv_acc_penalty, 0.0)
    _accuracy_met  = _accuracy >= ACCURACY_TARGET

    # ─────────────────────────────────────────────────────────────────────────
    # 2. P99 LATENCY (Little's Law + Roofline decode model)
    # Total daily requests = HOSPITAL_COUNT × INF_PER_DAY
    # Assume uniform distribution → arrival rate λ (req/s)
    # Decode: 1 token/step, arithmetic intensity = 1 op/byte → BW-bound
    # Latency per token ≈ bytes_per_token / BW_GBs
    # Model bytes (FP16 inference): M_params × 2 bytes
    # Sequence response: assume 50 tokens average
    # P99 ≈ 3× avg for M/M/1 at moderate utilization
    # Reference: @sec-model-serving-littles-law, @sec-inference-roofline-decode
    # ─────────────────────────────────────────────────────────────────────────
    _total_rps       = HOSPITAL_COUNT * INF_PER_DAY / 86400.0  # req/s
    _bytes_per_param = 2.0                                       # FP16
    _model_bytes_gb  = _M * _bytes_per_param / 1e9              # GB
    # Shards per replica: ceil(model_bytes / H100_RAM)
    _shards = max(math.ceil(_model_bytes_gb / H100_RAM_GB), 1)
    # Token decode rate per H100 (BW-bound)
    _tokens_per_sec_gpu = H100_BW_GBS * 1e9 / (_M * _bytes_per_param)
    _response_tokens    = 50                                     # tokens per response
    _avg_latency_s      = _response_tokens / max(_tokens_per_sec_gpu, 1e-6)
    _p99_latency_ms     = _avg_latency_s * 1000 * 3.0
    _latency_met        = _p99_latency_ms < P99_SLO_MS

    # Replicas needed to handle total_rps at avg_latency
    _replicas_needed = math.ceil(_total_rps * _avg_latency_s * _shards)
    _replicas_available = BUDGET_GPUS // _shards

    # ─────────────────────────────────────────────────────────────────────────
    # 3. DP COMPLIANCE (HIPAA)
    # eps ≤ 1.0 required for HIPAA-grade differential privacy
    # Reference: @sec-security-privacy-dp-hipaa
    # ─────────────────────────────────────────────────────────────────────────
    _dp_met = _eps <= DP_EPS_LIMIT

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ADVERSARIAL ROBUSTNESS
    # Adversarial robustness under PGD attack scales with adv_train_weight.
    # At adv_w = 0: robustness ≈ 5% (near-zero for undefended model)
    # At adv_w = 0.5: robustness ≈ 50%
    # At adv_w = 1.0: robustness ≈ 70%
    # Linear interpolation + saturation
    # Reference: @sec-robust-ai-pgd-training
    # ─────────────────────────────────────────────────────────────────────────
    _adv_robustness     = 0.05 + _adv_w * 0.65
    _adversarial_met    = _adv_robustness >= ADV_ROBUSTNESS_TARGET

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CARBON REDUCTION
    # Baseline: BASELINE_CI_G_KWH (386 g CO2/kWh)
    # Actual carbon intensity depends on context toggle + flexible scheduling
    # Carbon-aware scheduling shifts flex_pct of workload to low-CI hours
    # Effective CI = ci × (1 - flex_pct × 0.7)  [scheduling reduces CI by up to 70%]
    # Target: ≥ 40% reduction vs. BASELINE_CI_G_KWH
    # Reference: @sec-sustainable-ai-carbon-aware-scheduling
    # ─────────────────────────────────────────────────────────────────────────
    _eff_ci              = _ci * (1.0 - _flex_pct * 0.70)
    _carbon_reduction    = 1.0 - _eff_ci / BASELINE_CI_G_KWH
    _carbon_met          = _carbon_reduction >= CARBON_REDUCTION_TARGET

    # ─────────────────────────────────────────────────────────────────────────
    # 6. FAULT TOLERANCE (Young-Daly + availability)
    # Total GPUs in serving fleet
    # Cluster-level failure rate: lambda = N_gpus / MTBF_GPU_HOURS
    # Young-Daly: T* = sqrt(2 × C / lambda) where C = CHECKPOINT_COST_S / 3600
    # Availability = 1 - lambda × (T_min/2 + C + restart) / 1
    # Target: ≥ 99.9% uptime
    # Reference: @sec-fault-tolerance-young-daly
    # ─────────────────────────────────────────────────────────────────────────
    _N_gpus         = min(BUDGET_GPUS, max(_replicas_needed, 100))
    _lambda_hr      = _N_gpus / MTBF_GPU_HOURS   # failures/hour
    _C_hr           = CHECKPOINT_COST_S / 3600.0  # checkpoint cost in hours
    _T_hr           = _T_min / 60.0
    _T_opt_min      = math.sqrt(2.0 * _C_hr / max(_lambda_hr, 1e-9)) * 60.0
    _restart_hr     = 0.5                          # 30-minute restart
    _downtime_frac  = _lambda_hr * (_T_hr / 2.0 + _C_hr + _restart_hr)
    _uptime_pct     = max(1.0 - _downtime_frac, 0.0)
    _fault_tol_met  = _uptime_pct >= UPTIME_TARGET

    # ─────────────────────────────────────────────────────────────────────────
    # BUDGET CHECK: total GPUs needed vs. available
    # ─────────────────────────────────────────────────────────────────────────
    _budget_ok      = _replicas_needed <= BUDGET_GPUS

    # ─────────────────────────────────────────────────────────────────────────
    # OVERALL
    # ─────────────────────────────────────────────────────────────────────────
    _constraints_all_met = (
        _accuracy_met and _latency_met and _dp_met and
        _adversarial_met and _carbon_met and _fault_tol_met
    )
    _n_met = sum([
        _accuracy_met, _latency_met, _dp_met,
        _adversarial_met, _carbon_met, _fault_tol_met
    ])

    # ── Color helper ──────────────────────────────────────────────────────────
    def _sc(ok):
        return COLORS["GreenLine"] if ok else COLORS["RedLine"]

    def _tick(ok):
        return "&#x2713;" if ok else "&#x274C;"

    def _badge(ok):
        return "PASS" if ok else "FAIL"

    # ── 6-constraint scorecard ────────────────────────────────────────────────
    _scorecard = mo.Html(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0;">

        <div class="lab-card" style="border-top: 4px solid {_sc(_accuracy_met)}; text-align: center; padding: 18px 14px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.7rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                1. Accuracy
            </div>
            <div style="font-size: 2.0rem; font-weight: 900; color: {_sc(_accuracy_met)};">
                {_accuracy*100:.1f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-top: 4px;">
                target: &ge;{ACCURACY_TARGET*100:.0f}%
                &nbsp;<span style="font-weight:700;">{_tick(_accuracy_met)} {_badge(_accuracy_met)}</span>
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; margin-top: 6px; line-height:1.5;">
                base={_base_acc*100:.1f}% &minus; DP penalty={_dp_acc_penalty*100:.1f}%
                &minus; adv penalty={_adv_acc_penalty*100:.1f}%
            </div>
        </div>

        <div class="lab-card" style="border-top: 4px solid {_sc(_latency_met)}; text-align: center; padding: 18px 14px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.7rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                2. P99 Latency
            </div>
            <div style="font-size: 2.0rem; font-weight: 900; color: {_sc(_latency_met)};">
                {_p99_latency_ms:.0f}ms
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-top: 4px;">
                SLO: &lt;{P99_SLO_MS}ms
                &nbsp;<span style="font-weight:700;">{_tick(_latency_met)} {_badge(_latency_met)}</span>
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; margin-top: 6px; line-height:1.5;">
                {_M_B}B params &times; 2 bytes = {_model_bytes_gb:.0f} GB
                &nbsp;&rarr; {_shards} shard(s)/replica
            </div>
        </div>

        <div class="lab-card" style="border-top: 4px solid {_sc(_dp_met)}; text-align: center; padding: 18px 14px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.7rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                3. DP Compliance (HIPAA)
            </div>
            <div style="font-size: 2.0rem; font-weight: 900; color: {_sc(_dp_met)};">
                &epsilon; = {_eps:.1f}
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-top: 4px;">
                HIPAA limit: &epsilon; &le; {DP_EPS_LIMIT}
                &nbsp;<span style="font-weight:700;">{_tick(_dp_met)} {_badge(_dp_met)}</span>
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; margin-top: 6px; line-height:1.5;">
                accuracy penalty &asymp; {_dp_acc_penalty*100:.1f}%
            </div>
        </div>

        <div class="lab-card" style="border-top: 4px solid {_sc(_adversarial_met)}; text-align: center; padding: 18px 14px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.7rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                4. Adversarial Robustness
            </div>
            <div style="font-size: 2.0rem; font-weight: 900; color: {_sc(_adversarial_met)};">
                {_adv_robustness*100:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-top: 4px;">
                PGD target: &ge;{ADV_ROBUSTNESS_TARGET*100:.0f}%
                &nbsp;<span style="font-weight:700;">{_tick(_adversarial_met)} {_badge(_adversarial_met)}</span>
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; margin-top: 6px; line-height:1.5;">
                adv weight={_adv_w:.2f}
                &nbsp;&rarr; clean acc cost={_adv_acc_penalty*100:.1f}%
            </div>
        </div>

        <div class="lab-card" style="border-top: 4px solid {_sc(_carbon_met)}; text-align: center; padding: 18px 14px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.7rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                5. Carbon Reduction
            </div>
            <div style="font-size: 2.0rem; font-weight: 900; color: {_sc(_carbon_met)};">
                {_carbon_reduction*100:.0f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-top: 4px;">
                target: &gt;{CARBON_REDUCTION_TARGET*100:.0f}% vs. baseline
                &nbsp;<span style="font-weight:700;">{_tick(_carbon_met)} {_badge(_carbon_met)}</span>
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; margin-top: 6px; line-height:1.5;">
                eff CI = {_eff_ci:.0f} g/kWh
                &nbsp;(flex={_flex_pct*100:.0f}%)
            </div>
        </div>

        <div class="lab-card" style="border-top: 4px solid {_sc(_fault_tol_met)}; text-align: center; padding: 18px 14px;">
            <div style="color: {COLORS['TextMuted']}; font-size: 0.7rem; font-weight: 700;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                6. Fault Tolerance
            </div>
            <div style="font-size: 2.0rem; font-weight: 900; color: {_sc(_fault_tol_met)};">
                {_uptime_pct*100:.3f}%
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.78rem; margin-top: 4px;">
                uptime target: &ge;{UPTIME_TARGET*100:.1f}%
                &nbsp;<span style="font-weight:700;">{_tick(_fault_tol_met)} {_badge(_fault_tol_met)}</span>
            </div>
            <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; margin-top: 6px; line-height:1.5;">
                T*={_T_opt_min:.0f}min &nbsp;|&nbsp; your T={_T_min}min
            </div>
        </div>

    </div>
    """)
    _scorecard
    return (
        _accuracy,
        _accuracy_met,
        _latency_met,
        _dp_met,
        _adversarial_met,
        _carbon_met,
        _fault_tol_met,
        _constraints_all_met,
        _n_met,
        _budget_ok,
        _uptime_pct,
        _p99_latency_ms,
        _adv_robustness,
        _carbon_reduction,
        _eff_ci,
        _T_opt_min,
        _shards,
        _replicas_needed,
        _dp_acc_penalty,
        _adv_acc_penalty,
    )


# ─── ACT II: FAILURE STATES AND SUCCESS STATE ─────────────────────────────────
@app.cell(hide_code=True)
def _(
    ACCURACY_TARGET,
    ADV_ROBUSTNESS_TARGET,
    BUDGET_GPUS,
    CARBON_REDUCTION_TARGET,
    DP_EPS_LIMIT,
    P99_SLO_MS,
    UPTIME_TARGET,
    _accuracy,
    _accuracy_met,
    _adv_robustness,
    _adversarial_met,
    _budget_ok,
    _carbon_met,
    _carbon_reduction,
    _constraints_all_met,
    _dp_met,
    _eff_ci,
    _fault_tol_met,
    _latency_met,
    _n_met,
    _p99_latency_ms,
    _replicas_needed,
    _T_opt_min,
    _uptime_pct,
    checkpoint_interval_min,
    dp_epsilon,
    mo,
):
    _banners = []

    if not _accuracy_met:
        _banners.append(mo.callout(mo.md(
            f"**Accuracy below clinical threshold.** Current accuracy: "
            f"**{_accuracy*100:.1f}%** (required: {ACCURACY_TARGET*100:.0f}%). "
            f"DP noise (eps={dp_epsilon.value:.1f}) and adversarial training together "
            f"impose accuracy penalties that compound. "
            f"Increase model size OR reduce adversarial weight OR raise eps (if HIPAA allows). "
            f"Note: DP and adversarial training pull accuracy in the SAME downward direction "
            f"— both add noise/randomization that smooth decision boundaries."
        ), kind="danger"))

    if not _latency_met:
        _banners.append(mo.callout(mo.md(
            f"**P99 SLO violated.** Estimated P99 = **{_p99_latency_ms:.0f}ms** "
            f"(SLO: {P99_SLO_MS}ms). "
            f"The model's decode rate is bandwidth-bound (arithmetic intensity = 1 op/byte). "
            f"Reduce model size to lower per-token latency, or add more replicas. "
            f"You need {_replicas_needed:,} GPU-shards; budget is {BUDGET_GPUS:,}."
        ), kind="danger"))

    if not _dp_met:
        _banners.append(mo.callout(mo.md(
            f"**HIPAA DP violation.** epsilon = **{dp_epsilon.value:.1f}** exceeds "
            f"the HIPAA-grade limit of eps <= {DP_EPS_LIMIT}. "
            f"Medical image data under HIPAA requires strong differential privacy. "
            f"Reduce epsilon — at the cost of increased accuracy penalty."
        ), kind="danger"))

    if not _adversarial_met:
        _banners.append(mo.callout(mo.md(
            f"**Adversarial robustness insufficient.** Current PGD robustness: "
            f"**{_adv_robustness*100:.0f}%** (target: {ADV_ROBUSTNESS_TARGET*100:.0f}%). "
            f"Medical AI systems in adversarial environments require adversarial training. "
            f"Increase adversarial training weight — but note it reduces clean accuracy."
        ), kind="danger"))

    if not _carbon_met:
        _banners.append(mo.callout(mo.md(
            f"**Carbon reduction target missed.** Achieved: "
            f"**{_carbon_reduction*100:.0f}%** reduction "
            f"(effective CI: {_eff_ci:.0f} g CO2/kWh). "
            f"Target: {CARBON_REDUCTION_TARGET*100:.0f}% reduction vs. baseline. "
            f"Switch to carbon-optimized context OR increase flexible job percentage. "
            f"Jevons Paradox warning: efficiency gains alone may be insufficient "
            f"if fleet scale grows faster than carbon intensity falls."
        ), kind="danger"))

    if not _fault_tol_met:
        _banners.append(mo.callout(mo.md(
            f"**Uptime target missed.** Estimated uptime: "
            f"**{_uptime_pct*100:.3f}%** (target: {UPTIME_TARGET*100:.1f}%). "
            f"Young-Daly optimal checkpoint interval is **{_T_opt_min:.0f} min** "
            f"for this fleet size. Your interval: {checkpoint_interval_min.value} min. "
            f"Reduce checkpoint interval toward T* to minimize expected waste time."
        ), kind="danger"))

    if not _budget_ok:
        _banners.append(mo.callout(mo.md(
            f"**GPU budget exceeded.** Your configuration requires "
            f"**{_replicas_needed:,} GPU-shards** but the budget is {BUDGET_GPUS:,} H100s. "
            f"Reduce model size, increase quantization (which increases shards-per-replica "
            f"at lower memory), or accept lower replica count with higher latency."
        ), kind="warn"))

    if _constraints_all_met:
        mo.callout(mo.md(
            f"**ARCHITECTURE APPROVED: All {_n_met}/6 constraints satisfied. "
            f"System is deployable.** "
            f"Accuracy: {_accuracy*100:.1f}% | P99: {_p99_latency_ms:.0f}ms | "
            f"DP eps: {dp_epsilon.value:.1f} | Robustness: {_adv_robustness*100:.0f}% | "
            f"Carbon reduction: {_carbon_reduction*100:.0f}% | "
            f"Uptime: {_uptime_pct*100:.3f}%"
        ), kind="success")
    elif _banners:
        mo.vstack(_banners)
    else:
        mo.callout(mo.md(
            f"**{_n_met}/6 constraints met.** "
            f"Adjust the sliders above to satisfy all constraints simultaneously."
        ), kind="info")
    return


# ─── ACT II: PREDICTION REVEAL ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    _accuracy,
    _accuracy_met,
    _adversarial_met,
    _constraints_all_met,
    _dp_met,
    _latency_met,
    _n_met,
    act2_pred,
    mo,
):
    _failing = []
    if not _accuracy_met:   _failing.append("Accuracy")
    if not _latency_met:    _failing.append("P99 Latency")
    if not _dp_met:         _failing.append("DP Compliance")
    if not _adversarial_met: _failing.append("Adversarial Robustness")

    _feedback_map = {
        "A": (
            "**DP epsilon is genuinely difficult** — at eps=1, accuracy degrades by ~5%. "
            "For a baseline 95% target model, this leaves no margin for other accuracy "
            "costs. But this is only correct in a narrow sense. The deeper issue is that "
            "DP noise and adversarial training *both* degrade accuracy in the same direction: "
            "both smooth decision boundaries. These two constraints are "
            "**fundamentally incompatible**, not just difficult to balance simultaneously. "
            "DP adds noise to make the model's outputs less sensitive to any individual "
            "training sample. Adversarial training adds noise to make the model robust "
            "to input perturbations. Both mechanisms reduce model confidence — but for "
            "orthogonal reasons. This is the mathematical conflict at the heart of Act II."
        ),
        "B": (
            "**Correct.** No single architecture satisfies all six constraints without "
            "explicit tradeoff negotiation. The key conflicts are: "
            "(1) DP and adversarial robustness both reduce accuracy — they cannot both "
            "be maximized without a model large enough to absorb both penalties; "
            "(2) large models reduce P99 latency; (3) carbon reduction conflicts with "
            "fleet scale. The feasible region (all-green) requires navigating the "
            "intersection of these constraints — which is exactly what the Architecture "
            "Synthesizer reveals. This is the Chouldechova-generalized lesson: "
            "in multi-constraint systems, you choose which constraint to relax."
        ),
        "C": (
            "**Partially correct, but incomplete.** The 10,000 H100 budget *can* "
            "accommodate the serving load at smaller model sizes. But budget sufficiency "
            "does not equal constraint satisfaction. Even with 10,000 H100s, "
            "a 70B model at P99 < 200ms requires more shards-per-replica than available, "
            "and DP + adversarial training may push accuracy below 95%. "
            "Hardware budget is necessary but not sufficient."
        ),
        "D": (
            "**Incorrect.** Carbon reduction is NOT independent of other constraints. "
            "The Jevons Paradox directly links carbon to fleet scale: if you add GPUs "
            "to satisfy the latency SLO, you increase total power consumption, "
            "which makes the carbon target harder to hit. Carbon-aware scheduling "
            "reduces effective CI, but only if deferrable jobs exist to shift. "
            "Carbon is entangled with every other dimension through fleet size."
        ),
    }

    _chosen = _feedback_map.get(act2_pred.value, _feedback_map["B"])

    if _constraints_all_met:
        mo.callout(mo.md(
            f"**{_n_met}/6 constraints satisfied.** " + _chosen
        ), kind="success")
    else:
        _fail_str = ", ".join(_failing) if _failing else "multiple"
        mo.callout(mo.md(
            f"**{_n_met}/6 constraints satisfied.** "
            f"Currently failing: **{_fail_str}**. "
            + _chosen
        ), kind="warn")
    return


# ─── ACT II: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Reflection")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) P99 latency and model accuracy — larger models are slower": "A",
            "B) DP privacy and adversarial robustness — both require noise/randomization but in opposite directions for model confidence": "B",
            "C) Carbon reduction and fault tolerance — checkpointing uses more energy": "C",
            "D) Parallelism efficiency and checkpoint overhead — communication vs. recovery cost": "D",
        },
        label="Which two constraints are FUNDAMENTALLY incompatible (not just hard to balance simultaneously)?",
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(act2_reflection, mo):
    if act2_reflection.value is None:
        mo.callout(
            mo.md("Select your reflection answer above to continue."),
            kind="warn",
        )
    elif act2_reflection.value == "B":
        mo.callout(mo.md(
            "**Correct.** DP privacy and adversarial robustness are fundamentally "
            "incompatible in the following sense: "
            "**DP noise makes the model's outputs smoother and less sensitive** "
            "to individual inputs (including adversarial perturbations). "
            "**Adversarial training sharpens the model's decision boundaries** "
            "to resist those same perturbations. "
            "These two mechanisms push model confidence in opposite directions. "
            "DP adds isotropic Gaussian noise to gradients during training, which "
            "diffuses the loss landscape. Adversarial training concentrates the "
            "loss signal at adversarial examples, sharpening it. "
            "The result: achieving strong DP (low eps) while simultaneously achieving "
            "high adversarial robustness requires a model with enough capacity to "
            "maintain both — but both penalize clean accuracy. "
            "This is not an engineering challenge. It is an algebraic tension, "
            "analogous to Chouldechova's impossibility in the fairness domain."
        ), kind="success")
    elif act2_reflection.value == "A":
        mo.callout(mo.md(
            "**This is a tradeoff, not a fundamental incompatibility.** "
            "Larger models are slower — true. But you can add replicas, use "
            "quantization, or select a smaller model that still achieves 95% accuracy. "
            "Latency and accuracy can both be satisfied with the right design. "
            "There is no mathematical theorem preventing their simultaneous satisfaction. "
            "DP and adversarial robustness, by contrast, have mechanistic interference."
        ), kind="warn")
    elif act2_reflection.value == "C":
        mo.callout(mo.md(
            "**This is not fundamentally incompatible.** "
            "Carbon-aware scheduling and checkpoint frequency operate on different "
            "timescales and resource dimensions. You can checkpoint frequently "
            "without increasing power consumption (checkpoints are I/O-bound, "
            "not compute-bound). Fault tolerance and carbon are independently satisfiable "
            "with the right architectural choices. They are not mechanistically coupled."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**This is a tradeoff, not a fundamental incompatibility.** "
            "Communication overhead and checkpoint cost can be jointly minimized "
            "with asynchronous checkpointing and topology-aware AllReduce. "
            "They compete for network bandwidth but do not violate any theorem. "
            "The right system design reduces both independently."
        ), kind="warn")
    return


# ─── ACT II: MATHPEEK ACCORDION ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The fundamental conflict: DP noise vs. adversarial robustness": mo.md("""
        **Differential Privacy (DP Noise Direction):**
        During training, DP-SGD clips gradients to sensitivity S, then adds noise:
        `g_tilde = clip(g, S) + N(0, sigma^2 * S^2 * I)`
        Effect: gradients from all training examples (including adversarial ones)
        are *smoothed*. The learned decision boundary becomes flatter near training points.
        — Source: @sec-security-privacy-dp-sgd

        **Adversarial Training (Robustness Direction):**
        At each step, adversarial training maximizes the loss over a perturbation ball:
        `theta* = argmin_theta E[max_{delta: ||delta|| <= eps} L(x + delta, y; theta)]`
        Effect: the decision boundary is forced *sharp* at adversarial perturbations.
        The model must distinguish clean from perturbed inputs with high confidence.
        — Source: @sec-robust-ai-pgd

        **The Tension:**
        DP smooths → lower confidence near any input.
        Adversarial training sharpens → higher confidence near adversarial inputs.
        Both *penalize clean accuracy* for different reasons.
        At low DP epsilon (strong privacy), the noise scale sigma is large,
        and the gradients from adversarial examples are effectively washed out —
        the adversarial training signal is attenuated by DP noise.
        This is not fixable by adding more data or a larger model:
        it is a consequence of the conflicting objectives.

        **The Resolution:**
        The feasible region exists (all-green is achievable in this lab)
        but requires: (1) a model large enough that both accuracy penalties still
        leave you above 95%, (2) an epsilon in [0.5, 1.0] that satisfies HIPAA
        while not destroying the adversarial training signal, and (3) an
        adversarial weight calibrated to the DP noise level.
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# VOL1 + VOL2 SYNTHESIS TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Curriculum Journey Summary
    """)
    return


@app.cell(hide_code=True)
def _(COLORS, _ledger_map, mo):
    # Build a visual timeline of all 33 labs with constraint_hit indicators
    _vol1_entries = [
        (f"V1-{c:02d}", str(c)) for c in range(1, 17)
    ]
    _vol2_entries = [
        (f"V2-{c:02d}", f"v2_{c:02d}") for c in range(1, 18)
    ]
    _all_entries = _vol1_entries + _vol2_entries

    def _dot(label, key):
        _d      = _ledger_map.get(key, {})
        _done   = key in _ledger_map
        _hit    = _d.get("constraint_hit", False)
        if not _done:
            _bg    = "#e2e8f0"
            _color = "#94a3b8"
            _sym   = ""
        elif _hit:
            _bg    = COLORS["RedL"]
            _color = COLORS["RedLine"]
            _sym   = "!"
        else:
            _bg    = COLORS["GreenL"]
            _color = COLORS["GreenLine"]
            _sym   = "&#x2713;"
        return (
            f'<div style="background:{_bg}; color:{_color}; border:1px solid {_color}; '
            f'border-radius:6px; width:44px; height:38px; display:inline-flex; '
            f'flex-direction:column; align-items:center; justify-content:center; '
            f'font-size:0.6rem; font-weight:700; margin:3px; cursor:default;" '
            f'title="{label}">'
            f'<span style="font-size:0.55rem; color:{_color}; opacity:0.8;">{label}</span>'
            f'<span style="font-size:0.8rem;">{_sym}</span>'
            f'</div>'
        )

    _v1_dots = "".join(_dot(lbl, key) for lbl, key in _vol1_entries)
    _v2_dots = "".join(_dot(lbl, key) for lbl, key in _vol2_entries)

    _total_done = sum(1 for _, k in _all_entries if k in _ledger_map)
    _total_hit  = sum(
        1 for _, k in _all_entries
        if _ledger_map.get(k, {}).get("constraint_hit", False)
    )

    # Dominant context across all labs
    _contexts = [
        _ledger_map[k].get("context", "")
        for _, k in _all_entries if k in _ledger_map
    ]
    from collections import Counter as _Counter
    _ctx_count  = _Counter(_contexts)
    _dom_ctx    = _ctx_count.most_common(1)[0][0] if _ctx_count else "N/A"

    mo.Html(f"""
    <div class="lab-card" style="padding: 20px 24px;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px;">
            Lab Journey — All 33 Labs (Vol I + Vol II)
        </div>
        <div style="margin-bottom: 8px; font-size: 0.78rem; font-weight: 700;
                    color: {COLORS['TextSec']}; letter-spacing: 0.06em;">
            Volume I (Labs V1-01 through V1-16)
        </div>
        <div style="margin-bottom: 14px;">
            {_v1_dots}
        </div>
        <div style="margin-bottom: 8px; font-size: 0.78rem; font-weight: 700;
                    color: {COLORS['TextSec']}; letter-spacing: 0.06em;">
            Volume II (Labs V2-01 through V2-17)
        </div>
        <div style="margin-bottom: 18px;">
            {_v2_dots}
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; padding-top: 12px;
                    border-top: 1px solid {COLORS['Border']}; font-size: 0.82rem;">
            <div>
                <span style="color:{COLORS['TextMuted']};">Labs completed:</span>
                &nbsp;<strong style="color:{COLORS['BlueLine']};">{_total_done}/33</strong>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']};">Constraints triggered:</span>
                &nbsp;<strong style="color:{COLORS['OrangeLine']};">{_total_hit}</strong>
            </div>
            <div>
                <span style="color:{COLORS['TextMuted']};">Dominant context:</span>
                &nbsp;<strong style="color:{COLORS['Cloud']};">{_dom_ctx}</strong>
            </div>
            <div style="flex:1; text-align:right; color:{COLORS['TextMuted']}; font-style:italic;">
                Green = completed, no failure &nbsp;|&nbsp;
                Red = constraint triggered &nbsp;|&nbsp;
                Grey = not yet completed
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    *You have completed the ML Systems curriculum. The physics doesn't change — the constraints
    just shift with scale.*
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border-radius: 10px;
                    padding: 22px 26px; margin-bottom: 16px;
                    border: 1px solid {COLORS['Border']};">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 14px;">
                Key Takeaways
            </div>
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="width: 22px; height: 22px; border-radius: 50%;
                                background: {COLORS['BlueLine']}; color: #fff;
                                font-size: 0.7rem; font-weight: 800;
                                display: flex; align-items: center; justify-content: center;
                                flex-shrink: 0; margin-top: 2px;">1</div>
                    <div style="font-size: 0.88rem; color: {COLORS['Text']}; line-height: 1.65;">
                        <strong>Communication, not compute, dominates at scale.</strong>
                        At 1,000 GPUs, a 10% bandwidth degradation reduces cluster throughput
                        more than a 10% FLOPS reduction, because Ring AllReduce makes
                        synchronization the binding constraint. The bottleneck moves
                        as a function of fleet size — this is what Principle 6 means.
                    </div>
                </div>
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="width: 22px; height: 22px; border-radius: 50%;
                                background: {COLORS['OrangeLine']}; color: #fff;
                                font-size: 0.7rem; font-weight: 800;
                                display: flex; align-items: center; justify-content: center;
                                flex-shrink: 0; margin-top: 2px;">2</div>
                    <div style="font-size: 0.88rem; color: {COLORS['Text']}; line-height: 1.65;">
                        <strong>100x = 4x &times; 2.5x &times; 10x — and the 10x must come from orchestration.</strong>
                        Hardware and algorithms together contribute only 10x of the 100x
                        efficiency target. The next decade of ML systems engineering is
                        about compound AI: reasoning chains, tool use, and dynamic retrieval
                        that extract more capability from the same FLOPS.
                    </div>
                </div>
                <div style="display: flex; gap: 12px; align-items: flex-start;">
                    <div style="width: 22px; height: 22px; border-radius: 50%;
                                background: {COLORS['GreenLine']}; color: #fff;
                                font-size: 0.7rem; font-weight: 800;
                                display: flex; align-items: center; justify-content: center;
                                flex-shrink: 0; margin-top: 2px;">3</div>
                    <div style="font-size: 0.88rem; color: {COLORS['Text']}; line-height: 1.65;">
                        <strong>Constraints are coupled, not independent.</strong>
                        Every constraint you tightened in isolation (privacy &epsilon;,
                        carbon cap, adversarial robustness, fairness criterion) becomes
                        harder to satisfy when all six must hold simultaneously.
                        The skilled ML architect does not ask how to avoid the physics —
                        they ask which constraint to prioritize when they cannot all be
                        satisfied at once.
                    </div>
                </div>
            </div>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Production deployment.</strong> The curriculum ends here, but the
                    constraints do not. Every real system you build will navigate the same
                    six dimensions &mdash; memory, compute, networking, privacy, sustainability,
                    fairness &mdash; with a budget that forces explicit trade-offs.
                    The physics never lie; your job is to find the binding constraint first.
                </div>
            </div>

            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; Reference
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-conclusion-six-principles-distributed-ml-systems-746a
                    &mdash; Six principles and the communication-dominance invariant.<br/>
                    <strong>@sec-conclusion-path-forward-caa2</strong> &mdash; The 100x
                    decomposition and the Compound Capability Law.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. At 1,000 GPUs, a 10% bandwidth improvement yields more throughput gain than a 10% FLOPS improvement because Ring AllReduce makes synchronization the binding constraint. At what fleet size does the bottleneck shift, and what principle (Principle 6) explains why the bottleneck moves as a function of scale?
2. The 100x efficiency target decomposes as Hardware (4x) x Algorithm (2.5x) x Orchestration (10x). Hardware and algorithm gains face diminishing returns. Why must the 10x orchestration factor come from compound AI (reasoning chains, tool use, dynamic retrieval) rather than more hardware?
3. All six constraint families (memory, compute, networking, privacy, sustainability, fairness) interact as a coupled system. Tightening privacy epsilon makes sustainability harder (more compute for DP noise); tightening fairness makes accuracy harder. Describe one concrete tradeoff you explored across the labs and how a production system would resolve it.

**You're ready to move on if you can:**
- Identify the binding constraint at a given fleet scale and explain why it shifts as scale changes
- Decompose a system-level efficiency target into hardware, algorithm, and orchestration contributions
- Navigate tradeoffs between competing constraints when all six must be satisfied simultaneously
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN LEDGER SAVE + HUD FOOTER
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(
    COLORS,
    _accuracy,
    _accuracy_met,
    _adversarial_met,
    _adv_robustness,
    _carbon_met,
    _carbon_reduction,
    _constraints_all_met,
    _dp_met,
    _fault_tol_met,
    _latency_met,
    _n_met,
    _p99_latency_ms,
    _uptime_pct,
    act1_pred,
    act2_pred,
    adv_train_weight,
    checkpoint_interval_min,
    context_toggle,
    dp_epsilon,
    flexible_job_pct,
    ledger,
    model_size_b,
    mo,
    parallelism_strategy,
):
    # ── Save to Design Ledger ─────────────────────────────────────────────────
    ledger.save(
        chapter="v2_17",
        design={
            "context":                context_toggle.value,
            "model_size_b":           float(model_size_b.value),
            "dp_epsilon":             float(dp_epsilon.value),
            "adv_train_weight":       float(adv_train_weight.value),
            "parallelism_strategy":   parallelism_strategy.value,
            "checkpoint_interval_min": int(checkpoint_interval_min.value),
            "flexible_job_pct":       float(flexible_job_pct.value),
            "constraints_all_met":    bool(_constraints_all_met),
            "accuracy_met":           bool(_accuracy_met),
            "latency_met":            bool(_latency_met),
            "dp_met":                 bool(_dp_met),
            "adversarial_met":        bool(_adversarial_met),
            "carbon_met":             bool(_carbon_met),
            "fault_tolerance_met":    bool(_fault_tol_met),
            "act1_prediction":        str(act1_pred.value),
            "act1_correct":           act1_pred.value == "D",
            "act2_result":            "approved" if _constraints_all_met else "infeasible",
            "act2_decision":          parallelism_strategy.value,
            "constraint_hit":         not _constraints_all_met,
            "curriculum_complete":    True,
        }
    )

    # ── Build constraint status list ─────────────────────────────────────────
    _checks = [
        ("Accuracy >= 95%",    _accuracy_met),
        ("P99 < 200ms",        _latency_met),
        ("DP eps <= 1",        _dp_met),
        ("Robustness >= 50%",  _adversarial_met),
        ("Carbon -40%",        _carbon_met),
        ("Uptime 99.9%",       _fault_tol_met),
    ]

    _badge_html = "".join([
        f"""<span style="background: {'rgba(0,143,69,0.15)' if ok else 'rgba(203,32,45,0.15)'};
                         color: {'#4ade80' if ok else '#f87171'};
                         border: 1px solid {'rgba(0,143,69,0.35)' if ok else 'rgba(203,32,45,0.35)'};
                         padding: 4px 10px; border-radius: 20px; font-size: 0.75rem;
                         font-weight: 600; margin: 3px;">
            {'&#x2713;' if ok else '&#x274C;'} {label}
        </span>"""
        for label, ok in _checks
    ])

    _arch_status = "APPROVED" if _constraints_all_met else f"INFEASIBLE ({_n_met}/6)"
    _status_color = "#4ade80" if _constraints_all_met else "#f87171"

    _hud = mo.Html(f"""
    <div style="background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 100%);
                border-radius: 12px; padding: 20px 28px; margin-top: 24px;
                border: 1px solid rgba(99,102,241,0.25);">
        <div style="display: flex; justify-content: space-between; align-items: center;
                    flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
            <div>
                <div style="font-size: 0.65rem; font-weight: 700; color: #475569;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;">
                    LAB = V2-17 (CAPSTONE) &nbsp;&middot;&nbsp;
                    CONTEXT = {context_toggle.value.upper()} &nbsp;&middot;&nbsp;
                    CURRICULUM COMPLETE
                </div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #f1f5f9;">
                    Architecture Status:
                    <span style="color:{_status_color};">{_arch_status}</span>
                    &nbsp;&mdash;&nbsp; CONSTRAINTS MET: {_n_met}/6
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.72rem; color: #94a3b8; margin-bottom: 2px;">
                    Act 2 prediction
                </div>
                <div style="font-size: 1.0rem; font-weight: 700;
                            color: {'#4ade80' if act2_pred.value == 'B' else '#94a3b8'};">
                    Option {act2_pred.value}
                    {'&#x2713; Correct' if act2_pred.value == 'B' else ''}
                </div>
            </div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 14px;">
            {_badge_html}
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.07); padding-top: 14px;
                    font-size: 0.78rem; color: #64748b; line-height: 1.8;">
            <strong style="color:#94a3b8;">Model:</strong> {model_size_b.value}B params
            &nbsp;&middot;&nbsp;
            <strong style="color:#94a3b8;">DP &epsilon;:</strong> {dp_epsilon.value:.1f}
            &nbsp;&middot;&nbsp;
            <strong style="color:#94a3b8;">Adv weight:</strong> {adv_train_weight.value:.2f}
            &nbsp;&middot;&nbsp;
            <strong style="color:#94a3b8;">Ckpt interval:</strong> {checkpoint_interval_min.value} min
            &nbsp;&middot;&nbsp;
            <strong style="color:#94a3b8;">Flex jobs:</strong> {flexible_job_pct.value}%
            &nbsp;&middot;&nbsp;
            <strong style="color:#94a3b8;">Parallelism:</strong> {parallelism_strategy.value}
        </div>
        <div style="margin-top: 10px; font-size: 0.75rem; color: #4ade80; font-style: italic;">
            Curriculum complete. Vol I + Vol II Design Ledger saved. The physics doesn't change.
        </div>
    </div>
    """)
    _hud
    return


# ═══════════════════════════════════════════════════════════════════════════════
# THE META-PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## The Meta-Principle

    Every invariant in this curriculum traces back to the same root cause:
    **physical laws create hard ceilings that no amount of engineering can dissolve.**

    You cannot wish away the memory wall — HBM bandwidth is determined by signal
    physics and pin count. You cannot wish away Amdahl's Law — the serial fraction
    of your workload caps speedup regardless of cluster size. You cannot wish away
    Chouldechova's theorem — it follows from the definition of conditional probability
    when base rates differ. You cannot wish away Young-Daly — it follows from the
    calculus of minimization under a Poisson failure process. You cannot wish away
    Little's Law — it follows from queueing theory steady-state. You cannot make DP
    and adversarial robustness simultaneously costless — they are mechanistically
    opposed in the same loss landscape.

    But you *can* navigate these constraints. That is the discipline of ML systems:
    not finding a way around the physics, but designing systems that respect it.

    The skilled ML architect does not ask: "How do I avoid the memory wall?"
    They ask: "Which memory-wall-respecting architecture best satisfies my
    throughput, latency, cost, and safety requirements simultaneously?"

    That is the question this curriculum trained you to ask.
    """)
    return


if __name__ == "__main__":
    app.run()
