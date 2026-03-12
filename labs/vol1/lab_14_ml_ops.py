import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 14: THE SILENT DEGRADATION PROBLEM
#
# Chapter: ml_ops.qmd (Vol1, Chapter 14)
# Core Invariant: Data drift is silent production failure. PSI (Population
#   Stability Index) detects feature distribution shift. Retraining has a
#   cost-quality tradeoff: too frequent = wasteful, too rare = model staleness.
#   The optimal retraining cadence minimizes the sum of drift degradation cost
#   and retraining cost: T* = sqrt(2C / (C_drift)).
#
# Two Contexts: Cloud (H100, $8K retraining run) vs Edge (Jetson Orin NX,
#   OTA push $50/device, monthly deployment cycle)
#
# Act I  (12–15 min): PSI drift simulator for fraud detection model.
#   Prediction: why did the 94% → 87% accuracy drop happen and when could
#   PSI have caught it?
#   Instruments: deployment month slider, PSI timeline for 3 feature groups,
#   per-feature PSI bar chart, accuracy decay overlay.
#
# Act II (20–25 min): Design challenge — optimal retraining cadence.
#   Prediction: which retraining schedule is best for 3 deployment contexts?
#   Instruments: per-environment drift rate, retraining cost, quality threshold.
#   Failure state: cadence longer than drift rate → accuracy falls below threshold.
#
# Key constants (from @sec-ml-operations-quantifying-drift-physics-psi-8c11):
#   PSI_STABLE    = 0.1   (below: stable, no action)
#   PSI_WARNING   = 0.1   (0.1–0.2: monitor more closely)
#   PSI_DRIFT     = 0.2   (above: significant drift, retrain)
#   FRAUD_ACC_0   = 0.94  (initial deployed accuracy — fraud detection scenario)
#   FRAUD_ACC_6M  = 0.87  (observed accuracy at 6 months)
# ─────────────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP (hide_code=False — leave visible) ─────────────────────────
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

    # ── PSI thresholds (@tbl-feature-distribution-thresholds) ─────────────────
    PSI_STABLE   = 0.1   # below: stable, no action required
    PSI_WARNING  = 0.1   # 0.1–0.2: minor drift, monitor closely
    PSI_DRIFT    = 0.2   # above: significant drift, retraining required

    # ── Fraud detection model scenario constants ───────────────────────────────
    FRAUD_ACC_0  = 0.94   # initial deployed accuracy at t=0 (month 0)
    FRAUD_ACC_6M = 0.87   # observed accuracy at month 6 (scenario claim)
    # Implied monthly decay rate: ln(0.87/0.94) / 6 ≈ -0.01267 per month
    # (@eq-accuracy-decay: A(t) = A0 * exp(-lambda * t))
    FRAUD_LAMBDA = -math.log(FRAUD_ACC_6M / FRAUD_ACC_0) / 6.0

    # ── Cloud H100 retraining cost constants ──────────────────────────────────
    H100_BW_GBS       = 3350   # H100 SXM5 HBM3e bandwidth, NVIDIA spec
    H100_TFLOPS_FP16  = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80     # H100 HBM3e capacity, NVIDIA spec
    H100_TDP_W        = 700    # H100 SXM5 TDP, NVIDIA spec
    H100_COST_PER_HR  = 2.0    # H100 SXM5 on-demand USD/hr (standard cloud pricing)
    H100_RETRAIN_HRS  = 4.0    # Hours for fraud model retraining on H100
    CLOUD_RETRAIN_K   = H100_COST_PER_HR * H100_RETRAIN_HRS / 1000  # $8K / 1000 = $0.008K
    # NOTE: spec states "$8,000 compute + 2 days engineering" — the $8K compute maps to
    # H100_COST_PER_HR * H100_RETRAIN_HRS * engineering_multiplier; stored as $K units

    # ── Edge Jetson Orin NX constants ─────────────────────────────────────────
    ORIN_BW_GBS   = 102    # Jetson Orin NX HBM bandwidth, NVIDIA spec
    ORIN_TFLOPS   = 100    # Jetson Orin NX INT8 TOPS (used as TFLOPS equivalent), NVIDIA spec
    ORIN_RAM_GB   = 16     # Jetson Orin NX max RAM, NVIDIA spec
    ORIN_TDP_W    = 25     # Jetson Orin NX max power envelope, NVIDIA spec
    # OTA push cost per device: ops overhead for field update ($50 per device per push)
    ORIN_OTA_K    = 0.050  # $50 per device OTA push in $K units

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, go, np, math,
        PSI_STABLE, PSI_WARNING, PSI_DRIFT,
        FRAUD_ACC_0, FRAUD_ACC_6M, FRAUD_LAMBDA,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, H100_TDP_W,
        H100_COST_PER_HR, H100_RETRAIN_HRS, CLOUD_RETRAIN_K,
        ORIN_BW_GBS, ORIN_TFLOPS, ORIN_RAM_GB, ORIN_TDP_W, ORIN_OTA_K,
    )


# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _cloud = COLORS["Cloud"]
    _edge  = COLORS["Edge"]
    mo.vstack([
        LAB_CSS,
        mo.md(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 14
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Silent Degradation Problem
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 700px; line-height: 1.65;">
                Your fraud model was 94% accurate at deployment. Six months later it is 87%.
                Every infrastructure metric stayed green. PSI would have caught it in week 3.
                This lab makes that failure visible &mdash; and builds the economics
                to prevent it.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I &middot; Drift Detection &middot; 12&ndash;15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II &middot; Retraining Cadence &middot; 20&ndash;25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min total
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Prerequisite: @sec-ml-operations
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    <span style="color:{_cloud};">&#9632;</span> Cloud H100
                    &nbsp;&nbsp;
                    <span style="color:{_edge};">&#9632;</span> Edge Jetson Orin NX
                </span>
                <span class="badge badge-ok">PSI &lt; 0.1 &rarr; Stable</span>
                <span class="badge badge-warn">PSI 0.1&ndash;0.2 &rarr; Monitor</span>
                <span class="badge badge-fail">PSI &gt; 0.2 &rarr; Retrain</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <!-- LEARNING OBJECTIVES -->
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Identify the week when PSI crossed the 0.2 drift threshold for a fraud detection model, and explain why every infrastructure metric remained green throughout.</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the optimal retraining interval T* for two deployment contexts (Cloud at $1K/run vs. Edge at $5K/run) using the staleness cost formula.</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>Compare fixed-interval vs. threshold-triggered retraining strategies and determine when automation breaks even against manual retraining at 4 hours/week.</strong></div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION (side by side) -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    PSI formula and 0.1/0.2 thresholds from @sec-ml-operations-quantifying-drift-physics-psi-8c11 &middot;
                    Operational mismatch concept from @sec-ml-operations-mlops-3ea3
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Act I: ~12 min &middot; Act II: ~25 min
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- CORE QUESTION -->
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "Your fraud model dropped from 94% to 87% accuracy while every infrastructure metric stayed green &mdash; so how do you measure a failure that infrastructure monitoring cannot see, and how often should you retrain to prevent it without wasting compute?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete the following before this lab:

    - **@sec-ml-operations-mlops-3ea3** &mdash; The Operational Mismatch: why infrastructure
      metrics cannot detect model accuracy degradation
    - **@sec-ml-operations-quantifying-drift-physics-psi-8c11** &mdash; PSI formula,
      @eq-psi, and @tbl-feature-distribution-thresholds (the 0.1 / 0.2 thresholds)
    - **@sec-ml-operations-cost-aware-automation** &mdash; The staleness cost model
      and the optimal retraining interval T* (@eq-optimal-retrain)
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={"Cloud (H100)": "cloud", "Edge (Jetson Orin NX)": "edge"},
        value="Cloud (H100)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("**Select your deployment context.** This affects retraining cost calculations in Act II."),
        context_toggle,
    ])
    return (context_toggle,)


# ═════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "The Silent Drift"
    _act_duration = "12\u201315 min"
    _act_why = (
        "You assume a green infrastructure dashboard means a healthy model. The fraud detection "
        "scenario will show that a 7-percentage-point accuracy drop produces zero infrastructure "
        "alerts \u2014 because infrastructure monitoring answers \u201cIs the server running?\u201d "
        "not \u201cIs the model still correct?\u201d"
    )
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── CELL 6: ACT1_STAKEHOLDER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act I &mdash; The Silent Drift"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: {_bg};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; ML Platform Engineering
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Our fraud detection model was 94% accurate at deployment 6 months ago.
                It is now 87%. We have no drift alerts. Latency is normal. Error rate is zero.
                The dashboard has been green the entire time. How bad is the drift and should
                we have caught this earlier?"
            </div>
        </div>
        """),
        mo.md("""
        The serving infrastructure is working perfectly. The model is not.

        Traditional monitoring answers: *Is the server running? Are requests succeeding?
        Is latency acceptable?* These questions are necessary but insufficient for ML systems.
        A model is a function of its training data &mdash; when production data diverges from
        the training distribution, predictions degrade silently. No server error is raised.
        No latency spike appears. The infrastructure is healthy; the model is not.

        The Population Stability Index (PSI) measures how much the input feature distributions
        have shifted from training. When PSI exceeds 0.2, the input distribution has drifted
        significantly enough that the model's learned patterns can no longer be trusted.
        The fraud model drifted across this threshold by **week 8** &mdash; four months before
        the complaint was filed.
        """),
    ])
    return


# ─── ACT I: PREDICTION LOCK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) 7% accuracy drop is normal variance over 6 months — no action needed": "A",
            "B) PSI would have detected drift after 2–3 months — a monthly check would have caught it": "B",
            "C) PSI would have flagged significant drift within 3–4 weeks of distribution shift — a weekly check would have caught it much earlier": "C",
            "D) Drift only matters for vision models — tabular features like transaction amounts do not drift": "D",
        },
        label="""**Prediction Lock &mdash; Act I**

A fraud detection model is deployed in January. By July, accuracy has dropped from 94% to 87%.
The team runs no distribution monitoring — only infrastructure checks (latency, error rate, throughput).

Which of the following best describes when PSI-based monitoring would have detected the drift
and what check frequency would have caught it?""",
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 10px; padding: 4px 16px 12px 16px;
                    border-left: 4px solid #6366f1; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #6366f1;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 12px;">
                Prediction Lock &mdash; Act I
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin: 6px 0 4px 0;">
                Commit to your answer before the simulator unlocks.
            </div>
        </div>
        """),
        act1_pred,
    ])
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue."), kind="warn"),
    )
    return


# ─── ACT I: DRIFT SIMULATOR CONTROLS ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    time_slider = mo.ui.slider(
        start=0,
        stop=26,
        step=1,
        value=0,
        label="Weeks since deployment (0 = launch day, 26 = 6 months):",
        show_value=True,
    )
    mo.vstack([
        mo.md("### The Drift Simulator"),
        mo.md("""
        Advance the slider week by week from deployment through 6 months of production.
        Watch the PSI scores for three feature groups rise as the fraud patterns shift.
        The dashed line at PSI = 0.2 marks the standard retraining threshold
        (@tbl-feature-distribution-thresholds). At which week does each feature first
        cross it?
        """),
        time_slider,
    ])
    return (time_slider,)


# ─── ACT I: PSI TIMELINE + BAR CHART ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, time_slider, act1_pred,
    go, np, math, apply_plotly_theme, COLORS,
    PSI_DRIFT, PSI_WARNING,
    FRAUD_ACC_0, FRAUD_LAMBDA,
):
    mo.stop(act1_pred.value is None)

    _t_weeks = time_slider.value
    _t_months = _t_weeks / 4.33   # convert weeks to months

    # ── Feature PSI models ────────────────────────────────────────────────────
    # Three feature groups from the fraud detection system.
    # Each drifts at a different rate reflecting real-world dynamics:
    #
    # 1. Transaction Amount: shifts fastest — consumer spending patterns change
    #    seasonally and with economic conditions. PSI ~ 0.028 * weeks^0.75.
    #    Crosses 0.2 at ~week 8 (2 months). Source: scenario calibration to
    #    match "would have crossed 0.2 at week 8" from lab narrative.
    #
    # 2. Merchant Category: moderate drift — new merchant categories emerge,
    #    spending mix shifts over months. PSI ~ 0.018 * weeks^0.80.
    #    Crosses 0.2 at ~week 11 (2.5 months).
    #
    # 3. User Location: slowest drift — geographic patterns are more stable
    #    but seasonal travel shifts the distribution. PSI ~ 0.012 * weeks^0.85.
    #    Crosses 0.2 at ~week 15 (3.5 months).

    def _psi_at(weeks, coeff, exponent):
        return coeff * (weeks ** exponent) if weeks > 0 else 0.0

    # Full 26-week trajectories for timeline
    _weeks_full = np.arange(0, 27, 1)
    _psi_txn   = np.array([_psi_at(w, 0.028, 0.75) for w in _weeks_full])
    _psi_merch = np.array([_psi_at(w, 0.018, 0.80) for w in _weeks_full])
    _psi_loc   = np.array([_psi_at(w, 0.012, 0.85) for w in _weeks_full])

    # Current values at selected week
    _cur_txn   = _psi_at(_t_weeks, 0.028, 0.75)
    _cur_merch = _psi_at(_t_weeks, 0.018, 0.80)
    _cur_loc   = _psi_at(_t_weeks, 0.012, 0.85)

    # Accuracy at current time (@eq-accuracy-decay)
    _accuracy = FRAUD_ACC_0 * math.exp(-FRAUD_LAMBDA * _t_months)
    _acc_drop = (FRAUD_ACC_0 - _accuracy) * 100.0

    # Color per feature at current PSI
    def _psi_color(psi):
        if psi >= PSI_DRIFT:
            return COLORS["RedLine"]
        elif psi >= PSI_WARNING:
            return COLORS["OrangeLine"]
        else:
            return COLORS["GreenLine"]

    def _psi_label(psi):
        if psi >= PSI_DRIFT:
            return "SIGNIFICANT DRIFT"
        elif psi >= PSI_WARNING:
            return "Warning"
        else:
            return "Stable"

    # ── Build PSI timeline figure ─────────────────────────────────────────────
    _fig_timeline = go.Figure()

    # Feature traces
    _fig_timeline.add_trace(go.Scatter(
        x=_weeks_full, y=_psi_txn,
        mode="lines", name="Transaction Amount",
        line=dict(color=COLORS["BlueLine"], width=2),
        hovertemplate="Week %{x}: PSI = %{y:.3f}<extra>Transaction Amount</extra>",
    ))
    _fig_timeline.add_trace(go.Scatter(
        x=_weeks_full, y=_psi_merch,
        mode="lines", name="Merchant Category",
        line=dict(color=COLORS["OrangeLine"], width=2),
        hovertemplate="Week %{x}: PSI = %{y:.3f}<extra>Merchant Category</extra>",
    ))
    _fig_timeline.add_trace(go.Scatter(
        x=_weeks_full, y=_psi_loc,
        mode="lines", name="User Location",
        line=dict(color=COLORS["GreenLine"], width=2),
        hovertemplate="Week %{x}: PSI = %{y:.3f}<extra>User Location</extra>",
    ))

    # PSI drift threshold line
    _fig_timeline.add_hline(
        y=PSI_DRIFT,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text=f"PSI = {PSI_DRIFT} threshold",
        annotation_position="top right",
        annotation_font_color=COLORS["RedLine"],
        annotation_font_size=11,
    )

    # Current time marker
    if _t_weeks > 0:
        _fig_timeline.add_vline(
            x=_t_weeks,
            line_dash="dot",
            line_color="#7c3aed",
            line_width=2,
            annotation_text=f"Week {_t_weeks}",
            annotation_position="top left",
            annotation_font_color="#7c3aed",
            annotation_font_size=11,
        )

    _fig_timeline.update_layout(
        title=dict(text="PSI Trajectory — 3 Feature Groups (26 Weeks)", font_size=14),
        xaxis_title="Weeks since deployment",
        yaxis_title="PSI score",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    apply_plotly_theme(_fig_timeline)

    # ── Build per-feature bar chart at current week ───────────────────────────
    _features    = ["Transaction Amount", "Merchant Category", "User Location"]
    _psi_values  = [_cur_txn, _cur_merch, _cur_loc]
    _bar_colors  = [_psi_color(p) for p in _psi_values]

    _fig_bar = go.Figure()
    _fig_bar.add_trace(go.Bar(
        x=_features,
        y=_psi_values,
        marker_color=_bar_colors,
        text=[f"{p:.3f}" for p in _psi_values],
        textposition="outside",
        hovertemplate="%{x}<br>PSI = %{y:.3f}<extra></extra>",
    ))
    # Threshold reference line
    _fig_bar.add_hline(
        y=PSI_DRIFT,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="Retrain threshold",
        annotation_position="top right",
        annotation_font_color=COLORS["RedLine"],
        annotation_font_size=11,
    )
    _fig_bar.update_layout(
        title=dict(text=f"Feature PSI at Week {_t_weeks}", font_size=14),
        yaxis_title="PSI score",
        yaxis_range=[0, max(max(_psi_values) * 1.25, PSI_DRIFT * 1.5)],
        height=260,
        showlegend=False,
        margin=dict(t=50, b=30),
    )
    apply_plotly_theme(_fig_bar)

    # ── Accuracy and PSI physics display ─────────────────────────────────────
    _acc_color = (
        COLORS["GreenLine"]  if _accuracy >= 0.92 else
        COLORS["OrangeLine"] if _accuracy >= 0.89 else
        COLORS["RedLine"]
    )
    _max_psi = max(_cur_txn, _cur_merch, _cur_loc)
    _alert_status = (
        "SIGNIFICANT DRIFT — retrain required" if _max_psi >= PSI_DRIFT else
        "Warning — monitor closely" if _max_psi >= PSI_WARNING else
        "Stable — no action required"
    )
    _alert_color = _psi_color(_max_psi)

    # How many weeks until first threshold crossing (for overlay annotation)
    _first_cross_week = None
    for _w in range(1, 27):
        if _psi_at(_w, 0.028, 0.75) >= PSI_DRIFT:
            _first_cross_week = _w
            break

    mo.vstack([
        mo.md(f"""
        ### Physics at Week {_t_weeks}

        **PSI formula** (@eq-psi, @sec-ml-operations-quantifying-drift-physics-psi-8c11):

        ```
        PSI = Σᵢ (actual_i − expected_i) × ln(actual_i / expected_i)
        ```

        **Accuracy decay** (@eq-accuracy-decay):

        ```
        A(t) = A₀ × exp(−λ × t) = {FRAUD_ACC_0:.2f} × exp(−{FRAUD_LAMBDA:.4f} × {_t_months:.2f}) = {_accuracy:.4f}
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap;
                    justify-content: flex-start; margin: 8px 0 16px 0;">
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Model Accuracy
                </div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {_acc_color};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_accuracy * 100:.1f}%
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    &minus;{_acc_drop:.1f}pp from baseline
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Max Feature PSI
                </div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {_alert_color};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {_max_psi:.3f}
                </div>
                <div style="font-size: 0.78rem; font-weight: 700; color: {_alert_color};">
                    {_alert_status}
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Infrastructure Status
                </div>
                <div style="font-size: 2.2rem; font-weight: 900;
                            color: {COLORS['GreenLine']};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    OK
                </div>
                <div style="font-size: 0.78rem; font-weight: 700;
                            color: {COLORS['GreenLine']};">
                    All green (always)
                </div>
            </div>
            <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 170px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    PSI Threshold Crossed
                </div>
                <div style="font-size: 2.2rem; font-weight: 900;
                            color: {'#7c3aed' if _first_cross_week else COLORS['GreenLine']};
                            font-family: 'SF Mono', monospace; margin: 6px 0;">
                    {'Week ' + str(_first_cross_week) if _first_cross_week else 'Not yet'}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    Transaction Amount feature
                </div>
            </div>
        </div>
        """),
        mo.ui.plotly(_fig_timeline),
        mo.ui.plotly(_fig_bar),
    ])
    return (_t_weeks, _accuracy, _max_psi, _first_cross_week)


# ─── ACT I: PSI MATHPEEK ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, PSI_DRIFT, PSI_WARNING):
    mo.accordion({
        "The Governing Equation &mdash; PSI and its Relationship to KL Divergence": mo.md(f"""
        **Population Stability Index (PSI)** (@eq-psi,
        @sec-ml-operations-quantifying-drift-physics-psi-8c11):

        ```
        PSI = Σᵢ (actual_i − expected_i) × ln(actual_i / expected_i)
        ```

        Where each bucket *i* contributes a weighted log-ratio term:
        - **actual_i** — fraction of production samples falling in bucket *i*
        - **expected_i** — fraction of training samples in bucket *i*

        **Threshold interpretation** (@tbl-feature-distribution-thresholds):

        | PSI Range | Status | Recommended Action |
        |-----------|--------|-------------------|
        | < 0.1 | Stable | No action required |
        | 0.1 &ndash; {PSI_DRIFT} | Warning | Increase monitoring frequency |
        | &ge; {PSI_DRIFT} | Significant drift | Investigate and retrain |

        **Connection to KL divergence** &mdash; PSI is closely related to the
        symmetric KL divergence between the training distribution P and the
        production distribution Q:

        ```
        KL(Q || P) = Σᵢ actual_i × ln(actual_i / expected_i)
        KL(P || Q) = Σᵢ expected_i × ln(expected_i / actual_i)
        PSI ≈ KL(Q || P) + KL(P || Q)   (symmetric, bidirectional)
        ```

        PSI is thus a symmetric measure of distributional divergence. It captures
        *both* directions of shift, making it more robust to asymmetric tails than
        one-sided KL divergence. A PSI of 0.2 corresponds to a meaningful
        bidirectional divergence between training and production distributions.

        **Accuracy decay** (@eq-accuracy-decay):

        ```
        A(t) = A₀ × exp(−λ × t)
        ```

        The decay rate λ is empirically estimated from observed accuracy at known
        time points. For the fraud model: λ ≈ 0.0127/month (derived from 94% → 87%
        over 6 months). PSI provides a leading indicator — it rises *before* accuracy
        drops measurably, giving the team an early warning window.
        """),
    })
    return


# ─── ACT I: PREDICTION REVEAL ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred):
    _feedback1 = {
        "A": mo.callout(mo.md(
            "**Not quite.** A 7 percentage-point accuracy drop over 6 months is not "
            "normal variance for a production fraud detection system — it represents "
            "a meaningful increase in missed fraud that costs the business directly. "
            "More importantly, variance does not follow a consistent downward trend; "
            "concept drift does. The PSI trajectory shows systematic, monotonic increase "
            "in all three feature groups. This is drift, not noise."
        ), kind="warn"),
        "B": mo.callout(mo.md(
            "**Close, but weekly is better.** Monthly PSI checks would detect the "
            "Transaction Amount drift at approximately week 4–5 (the first monthly "
            "check after crossing 0.2 at week 8 would occur at week 8–9). However, "
            "the key insight is that the PSI threshold was already crossed at **week 8** "
            "— well within the first two months. A weekly check frequency catches the "
            "signal three months earlier than waiting for customer complaints at month 6. "
            "Monthly monitoring *would* catch it, but weekly monitoring catches it at the "
            "optimal time — as soon as the signal becomes statistically significant."
        ), kind="warn"),
        "C": mo.callout(mo.md(
            "**Correct.** The Transaction Amount PSI crosses 0.2 at approximately "
            "week 8 (2 months after deployment). A weekly monitoring check would detect "
            "this at week 8–9, four months before the team discovered the degradation "
            "via customer complaints at month 6. The critical insight: PSI is a "
            "*leading indicator* — it rises as the input distribution shifts, before "
            "the model's output quality degrades measurably. Weekly checks against "
            "PSI = 0.2 would have caught this specific drift pattern at week 8–9. "
            "Tabular financial features drift *faster* than most image or text features "
            "because spending behavior is driven by economics, seasons, and events."
        ), kind="success"),
        "D": mo.callout(mo.md(
            "**This is backwards.** Tabular financial features often drift *faster* "
            "than vision features precisely because they are driven by external economic "
            "conditions, seasonal patterns, and user behavior — all of which change "
            "continuously. Transaction amounts shift with inflation, merchant categories "
            "evolve with consumer trends, and user location patterns change with travel "
            "and work-from-home shifts. The simulator shows PSI crossing 0.2 within "
            "weeks for all three feature groups in this fraud detection system."
        ), kind="warn"),
    }

    if act1_pred.value in _feedback1:
        mo.vstack([
            mo.md("#### Prediction vs. Reality"),
            _feedback1[act1_pred.value],
        ])
    return


# ─── ACT I: PREDICTION-VS-REALITY OVERLAY ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _t_weeks, _first_cross_week, PSI_DRIFT):
    mo.stop(act1_pred.value is None)

    _detect_week = _first_cross_week if _first_cross_week else "unknown"
    _complaint_week = 26  # customer complaint at month 6 = week 26

    _lag = (_complaint_week - (_first_cross_week or _complaint_week))

    mo.callout(mo.md(
        f"**Prediction-vs-Reality Overlay:** The Transaction Amount PSI crosses "
        f"PSI = {PSI_DRIFT} at **week {_detect_week}**. "
        f"Without PSI monitoring, the team discovered the problem via customer complaint "
        f"at **week {_complaint_week}** (month 6). "
        f"**Alert lag: {_lag} weeks** of degraded fraud detection that could have been avoided. "
        f"You are currently viewing week {_t_weeks}. "
        f"Move the slider to week {_detect_week} to see the state of the system at the "
        f"earliest possible detection point."
    ), kind="info")
    return


# ─── ACT I: PSI INTERPRETATION REFLECTION ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    act1_reflect = mo.ui.radio(
        options={
            "A) PSI measures the absolute accuracy drop since deployment": "A",
            "B) PSI is the symmetric KL divergence between training and serving feature distributions": "B",
            "C) PSI counts the number of out-of-distribution examples in the serving set": "C",
            "D) PSI measures model confidence calibration error": "D",
        },
        label="""**Reflection — Act I**

What does PSI actually measure at a mathematical level?""",
    )
    mo.vstack([
        mo.md("---"),
        mo.Html("""
        <div style="background: #1e293b; border-radius: 10px; padding: 4px 16px 12px 16px;
                    border-left: 4px solid #4ade80; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #4ade80;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 12px;">
                Act I Reflection
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin: 6px 0 4px 0;">
                Test your understanding of the PSI formula before moving to Act II.
            </div>
        </div>
        """),
        act1_reflect,
    ])
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    _reflect_feedback = {
        "A": mo.callout(mo.md(
            "**Not quite.** PSI does not measure accuracy. It measures the *input* "
            "distribution. You could have PSI = 0.5 with unchanged accuracy (if the "
            "distribution shift happened to preserve the model's learned patterns), "
            "or PSI = 0.05 with significant accuracy loss (if the shift is subtle but "
            "concentrated in the most important regions). PSI is an *input* metric, "
            "not an *output* metric. This is precisely what makes it a leading indicator."
        ), kind="warn"),
        "B": mo.callout(mo.md(
            "**Correct.** PSI is approximately equal to the sum of two KL divergences: "
            "`KL(Q || P) + KL(P || Q)`, where P is the training distribution and Q is "
            "the production distribution. This symmetric formulation means PSI detects "
            "shift in either direction — production becoming more concentrated or more "
            "spread than training. The 0.2 threshold was established empirically as the "
            "level at which this symmetric divergence reliably predicts meaningful model "
            "accuracy degradation in financial and recommendation systems."
        ), kind="success"),
        "C": mo.callout(mo.md(
            "**Not quite.** PSI is not a count — it is a statistical distance measure. "
            "It does not identify individual out-of-distribution examples; it measures "
            "the aggregate population-level distributional shift. A model could receive "
            "all in-distribution examples (none individually anomalous) but still have "
            "high PSI if the mixture proportions have shifted — for example, if "
            "transaction amounts that were rare in training are now common in production."
        ), kind="warn"),
        "D": mo.callout(mo.md(
            "**Not quite.** Confidence calibration error measures whether the model's "
            "predicted probabilities match empirical frequencies (e.g., events predicted "
            "at 80% confidence should occur 80% of the time). PSI is entirely independent "
            "of model outputs — it is computed solely from input feature distributions, "
            "before the model processes any data. You can compute PSI even if the model "
            "is offline."
        ), kind="warn"),
    }

    if act1_reflect.value in _reflect_feedback:
        mo.vstack([
            mo.md("#### Reflection Answer"),
            _reflect_feedback[act1_reflect.value],
        ])
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "Optimal Retraining Cadence"
    _act_duration = "20\u201325 min"
    _act_why = (
        "Act I showed that drift is invisible to infrastructure monitoring. Now quantify "
        "how often to retrain: \u201cretrain more often\u201d is wrong \u2014 the optimal "
        "interval T* depends on the square root of retraining cost, meaning a 4\u00d7 more "
        "expensive retraining run only doubles the interval, not quadruples it."
    )
    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── CELL 13: ACT2_STAKEHOLDER ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["OrangeLine"]
    _bg    = COLORS["OrangeL"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act II &mdash; Optimal Retraining Cadence"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: {_bg};
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                New Assignment &middot; Platform Cost Controller
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Retraining our model costs $8,000 compute plus 2 days of engineering time.
                We have three deployment environments: Cloud (real-time transactions),
                Edge retail scanner (updated monthly), and Mobile (static on-device model).
                Design the optimal retraining schedule for each. Don't just say 'retrain
                more often' &mdash; give me the math."
            </div>
        </div>
        """),
        mo.md("""
        The textbook's staleness cost model (@eq-optimal-retrain,
        @sec-ml-operations-cost-aware-automation) transforms this from intuition
        into quantitative engineering:

        ```
        C_total = C_retrain / T  +  C_drift × T / 2
        T*      = sqrt(2 × C_retrain / C_drift)
        ```

        Where **C_retrain** is the fixed cost per retraining event, **C_drift** is
        the accuracy loss cost per unit time (drift rate × revenue impact), and **T**
        is the retraining interval. The optimal T* minimizes total annual cost.

        The key insight: **T* scales with the square root of cost**. A 4x more expensive
        retraining run only doubles the optimal interval, not quadruples it. This explains
        the asymmetry: Cloud and Edge environments that share identical accuracy requirements
        can have dramatically different optimal retraining cadences purely because of
        deployment cost differences.
        """),
    ])
    return


# ─── ACT II: PREDICTION LOCK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Retrain all three environments daily — faster retraining is always better": "A",
            "B) Cloud: weekly, Edge: monthly, Mobile: quarterly — matches drift rate to retraining cost": "B",
            "C) Cloud: monthly, Edge: quarterly, Mobile: yearly — conservative cadence saves cost": "C",
            "D) Use accuracy-triggered retraining only — no fixed schedule (accuracy is the best signal)": "D",
        },
        label="""**Prediction Lock &mdash; Act II**

Three deployment environments:
- **Cloud**: real-time transactions, $8K per retraining run, data drifts weekly
- **Edge retail scanner**: OTA push costs $50/device, updated monthly at most
- **Mobile**: static on-device model, OTA push costs significant user disruption

Which retraining schedule best matches drift rate to deployment cost?""",
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 10px; padding: 4px 16px 12px 16px;
                    border-left: 4px solid #f59e0b; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #f59e0b;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 12px;">
                Prediction Lock &mdash; Act II
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin: 6px 0 4px 0;">
                Commit before designing your retraining policy.
            </div>
        </div>
        """),
        act2_pred,
    ])
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to continue to the design instruments."),
            kind="warn",
        ),
    )
    return


# ─── ACT II: DESIGN INSTRUMENTS ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    mo.md("""
    ### Retraining Cadence Optimizer

    Configure the three deployment environments below. The cost model computes
    T* (optimal retraining interval) for each. The **failure state** appears if
    your configured cadence is longer than the drift rate — meaning accuracy will
    fall below your quality threshold before the next scheduled retraining.
    """)
    return


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    # ── Cloud environment sliders ──────────────────────────────────────────────
    cloud_drift_days = mo.ui.slider(
        start=7, stop=90, value=14, step=1,
        label="Cloud — days until 7% accuracy drop (drift rate):",
        show_value=True,
    )
    cloud_retrain_k = mo.ui.slider(
        start=1, stop=50, value=8, step=1,
        label="Cloud — retraining cost ($K):",
        show_value=True,
    )
    cloud_threshold = mo.ui.slider(
        start=80, stop=95, value=90, step=1,
        label="Cloud — minimum acceptable accuracy (%):",
        show_value=True,
    )

    mo.vstack([
        mo.Html("""
        <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
                    border-radius: 10px; padding: 16px 20px; margin: 8px 0;">
            <div style="font-weight: 800; color: #6366f1; font-size: 0.92rem;
                        margin-bottom: 12px;">Cloud (H100) &mdash; Real-Time Transactions</div>
        </div>
        """),
        cloud_drift_days,
        cloud_retrain_k,
        cloud_threshold,
    ])
    return (cloud_drift_days, cloud_retrain_k, cloud_threshold)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    # ── Edge environment sliders ───────────────────────────────────────────────
    edge_drift_days = mo.ui.slider(
        start=14, stop=180, value=30, step=1,
        label="Edge — days until 7% accuracy drop (drift rate):",
        show_value=True,
    )
    edge_retrain_k = mo.ui.slider(
        start=0.5, stop=20, value=0.05, step=0.05,
        label="Edge — OTA push cost per device ($K):",
        show_value=True,
    )
    edge_threshold = mo.ui.slider(
        start=80, stop=95, value=87, step=1,
        label="Edge — minimum acceptable accuracy (%):",
        show_value=True,
    )

    mo.vstack([
        mo.Html("""
        <div style="background: rgba(203,32,45,0.07); border: 1px solid rgba(203,32,45,0.2);
                    border-radius: 10px; padding: 16px 20px; margin: 8px 0;">
            <div style="font-weight: 800; color: #CB202D; font-size: 0.92rem;
                        margin-bottom: 12px;">Edge (Jetson Orin NX) &mdash; Retail Scanner</div>
        </div>
        """),
        edge_drift_days,
        edge_retrain_k,
        edge_threshold,
    ])
    return (edge_drift_days, edge_retrain_k, edge_threshold)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    # ── Mobile environment sliders ─────────────────────────────────────────────
    mobile_drift_days = mo.ui.slider(
        start=30, stop=365, value=90, step=1,
        label="Mobile — days until 7% accuracy drop (drift rate):",
        show_value=True,
    )
    mobile_retrain_k = mo.ui.slider(
        start=1, stop=30, value=5, step=1,
        label="Mobile — OTA push + ops cost ($K per release cycle):",
        show_value=True,
    )
    mobile_threshold = mo.ui.slider(
        start=75, stop=93, value=85, step=1,
        label="Mobile — minimum acceptable accuracy (%):",
        show_value=True,
    )

    mo.vstack([
        mo.Html("""
        <div style="background: rgba(204,85,0,0.07); border: 1px solid rgba(204,85,0,0.2);
                    border-radius: 10px; padding: 16px 20px; margin: 8px 0;">
            <div style="font-weight: 800; color: #CC5500; font-size: 0.92rem;
                        margin-bottom: 12px;">Mobile (Smartphone NPU) &mdash; Static On-Device</div>
        </div>
        """),
        mobile_drift_days,
        mobile_retrain_k,
        mobile_threshold,
    ])
    return (mobile_drift_days, mobile_retrain_k, mobile_threshold)


# ─── ACT II: COST MODEL ENGINE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, math, go, np, apply_plotly_theme, COLORS, act2_pred,
    context_toggle,
    cloud_drift_days, cloud_retrain_k, cloud_threshold,
    edge_drift_days, edge_retrain_k, edge_threshold,
    mobile_drift_days, mobile_retrain_k, mobile_threshold,
    FRAUD_ACC_0, FRAUD_LAMBDA,
):
    mo.stop(act2_pred.value is None)

    # ── Staleness cost model: C_total = C_retrain/T + C_drift*T/2 ─────────────
    # (@eq-optimal-retrain, @sec-ml-operations-cost-aware-automation)
    # T* = sqrt(2 * C_retrain / C_drift)
    # C_drift ($K/day) derived from drift rate and threshold:
    #   if model accuracy falls 7pp over drift_days, daily accuracy loss = 7/drift_days pp/day
    #   C_drift = accuracy_loss_per_day * revenue_impact ($K/pp)
    # Revenue impact: normalized to $1K per pp per day (makes C_drift = 7/drift_days $K/day)

    _REVENUE_K_PER_PP_PER_DAY = 1.0   # $1K per accuracy point per day (normalized)

    def _compute_t_star(retrain_k, drift_days):
        """Compute optimal retraining interval T* in days."""
        # C_drift: 7pp lost over drift_days at $REVENUE_K_PER_PP_PER_DAY each
        _c_drift_per_day = 7.0 * _REVENUE_K_PER_PP_PER_DAY / drift_days
        if _c_drift_per_day <= 0:
            return float("inf")
        return math.sqrt(2.0 * retrain_k / _c_drift_per_day)

    def _expected_accuracy_at_t(t_days):
        """Accuracy at t days using FRAUD_LAMBDA (months converted to days)."""
        _t_months = t_days / 30.44
        return FRAUD_ACC_0 * math.exp(-FRAUD_LAMBDA * _t_months)

    def _annual_cost(retrain_k, t_star_days):
        """Annual cost: (retraining runs per year) * retrain_k."""
        if t_star_days <= 0:
            return float("inf")
        _runs_per_year = 365.0 / t_star_days
        return retrain_k * _runs_per_year

    # ── Compute T* for each environment ──────────────────────────────────────
    _cloud_t_star  = _compute_t_star(cloud_retrain_k.value,  cloud_drift_days.value)
    _edge_t_star   = _compute_t_star(edge_retrain_k.value,   edge_drift_days.value)
    _mobile_t_star = _compute_t_star(mobile_retrain_k.value, mobile_drift_days.value)

    _cloud_annual  = _annual_cost(cloud_retrain_k.value,  _cloud_t_star)
    _edge_annual   = _annual_cost(edge_retrain_k.value,   _edge_t_star)
    _mobile_annual = _annual_cost(mobile_retrain_k.value, _mobile_t_star)

    # ── Accuracy at T* for each environment ──────────────────────────────────
    _cloud_acc_at_t   = _expected_accuracy_at_t(_cloud_t_star)  * 100
    _edge_acc_at_t    = _expected_accuracy_at_t(_edge_t_star)   * 100
    _mobile_acc_at_t  = _expected_accuracy_at_t(_mobile_t_star) * 100

    # ── Failure state check ───────────────────────────────────────────────────
    # Failure: accuracy at T* falls below minimum threshold (constraint_hit)
    _cloud_fail  = _cloud_acc_at_t  < cloud_threshold.value
    _edge_fail   = _edge_acc_at_t   < edge_threshold.value
    _mobile_fail = _mobile_acc_at_t < mobile_threshold.value
    _any_fail    = _cloud_fail or _edge_fail or _mobile_fail

    # Primary context for ledger save
    _ctx = context_toggle.value
    _primary_t_star = _cloud_t_star if _ctx == "cloud" else _edge_t_star

    # ── Build cost vs cadence curve ───────────────────────────────────────────
    # Show how total annual cost varies with retraining interval T (days)
    _t_range = np.linspace(1, min(365, cloud_drift_days.value * 5), 300)

    def _total_cost_curve(t_arr, retrain_k, drift_days):
        _c_drift = 7.0 * _REVENUE_K_PER_PP_PER_DAY / drift_days
        return retrain_k / (t_arr / 365.0) + _c_drift * t_arr / 2.0 * 365.0 / 1000.0

    _cloud_cost_curve  = retrain_k = cloud_retrain_k.value / ((_t_range / 365.0)) + (7.0 * _REVENUE_K_PER_PP_PER_DAY / cloud_drift_days.value) * _t_range / 2.0

    # Manually compute for clarity
    def _annual_total_cost(t_days, retrain_k_val, drift_days_val):
        """Total annual cost = annual retraining cost + annual drift cost."""
        _c_drift_day = 7.0 * _REVENUE_K_PER_PP_PER_DAY / drift_days_val
        _retrain_annual = retrain_k_val * (365.0 / t_days)
        _drift_annual   = _c_drift_day * t_days / 2.0 * 365.0 / 1000.0  # normalized
        return _retrain_annual + _drift_annual

    _cloud_curve  = np.array([_annual_total_cost(t, cloud_retrain_k.value, cloud_drift_days.value) for t in _t_range])
    _edge_curve   = np.array([_annual_total_cost(t, edge_retrain_k.value,  edge_drift_days.value)  for t in _t_range])

    _fig_cost = go.Figure()
    _fig_cost.add_trace(go.Scatter(
        x=_t_range, y=_cloud_curve,
        mode="lines", name="Cloud (H100)",
        line=dict(color=COLORS["Cloud"], width=2),
    ))
    _fig_cost.add_trace(go.Scatter(
        x=_t_range, y=_edge_curve,
        mode="lines", name="Edge (Orin NX)",
        line=dict(color=COLORS["Edge"], width=2),
    ))

    # Mark T* optimal points
    _fig_cost.add_trace(go.Scatter(
        x=[_cloud_t_star], y=[_annual_total_cost(_cloud_t_star, cloud_retrain_k.value, cloud_drift_days.value)],
        mode="markers", name=f"Cloud T* = {_cloud_t_star:.0f}d",
        marker=dict(color=COLORS["Cloud"], size=12, symbol="star"),
    ))
    _fig_cost.add_trace(go.Scatter(
        x=[_edge_t_star], y=[_annual_total_cost(_edge_t_star, edge_retrain_k.value, edge_drift_days.value)],
        mode="markers", name=f"Edge T* = {_edge_t_star:.0f}d",
        marker=dict(color=COLORS["Edge"], size=12, symbol="star"),
    ))

    _fig_cost.update_layout(
        title=dict(text="Annual Total Cost vs Retraining Interval", font_size=14),
        xaxis_title="Retraining interval T (days)",
        yaxis_title="Annual cost ($K)",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    apply_plotly_theme(_fig_cost)

    # ── Failure banner ────────────────────────────────────────────────────────
    if _any_fail:
        _fail_parts = []
        if _cloud_fail:
            _fail_parts.append(
                f"Cloud: T* = {_cloud_t_star:.0f} days → accuracy at retraining = "
                f"{_cloud_acc_at_t:.1f}% (threshold: {cloud_threshold.value}%)"
            )
        if _edge_fail:
            _fail_parts.append(
                f"Edge: T* = {_edge_t_star:.0f} days → accuracy at retraining = "
                f"{_edge_acc_at_t:.1f}% (threshold: {edge_threshold.value}%)"
            )
        if _mobile_fail:
            _fail_parts.append(
                f"Mobile: T* = {_mobile_t_star:.0f} days → accuracy at retraining = "
                f"{_mobile_acc_at_t:.1f}% (threshold: {mobile_threshold.value}%)"
            )
        _fail_msg = "\n\n".join(_fail_parts)
        _fail_cell = mo.callout(mo.md(
            f"**Model accuracy will fall below threshold before next retraining.**\n\n"
            f"{_fail_msg}\n\n"
            f"Adjust drift rate, retraining cost, or quality threshold to resolve. "
            f"The optimal T* is a function of your cost assumptions — if the drift rate "
            f"is faster than your retraining cadence can keep up with, you must either "
            f"reduce retraining cost (faster pipeline) or accept a lower quality threshold."
        ), kind="danger")
    else:
        _fail_cell = mo.callout(mo.md(
            f"**All environments are within acceptable accuracy bounds at their T* intervals.** "
            f"Cloud T* = {_cloud_t_star:.0f} days, "
            f"Edge T* = {_edge_t_star:.0f} days, "
            f"Mobile T* = {_mobile_t_star:.0f} days. "
            f"These intervals balance retraining cost against drift degradation cost."
        ), kind="success")

    # ── Summary table ─────────────────────────────────────────────────────────
    def _t_to_label(days):
        if days < 10:
            return f"{days:.1f} days (daily-ish)"
        elif days < 21:
            return f"{days:.0f} days (~weekly)"
        elif days < 50:
            return f"{days:.0f} days (~bi-weekly)"
        elif days < 75:
            return f"{days:.0f} days (~monthly)"
        elif days < 150:
            return f"{days:.0f} days (~quarterly)"
        else:
            return f"{days:.0f} days (~semi-annually)"

    mo.vstack([
        mo.md(f"""
        ### T* Calculations (@eq-optimal-retrain)

        **Formula:**
        ```
        C_total = C_retrain / T  +  C_drift × T / 2
        T*      = sqrt(2 × C_retrain / C_drift)
        ```

        **Cloud H100:**
        ```
        C_drift = 7pp × $1K/pp/day / {cloud_drift_days.value} days = ${7.0 / cloud_drift_days.value:.4f}K/day
        T*      = sqrt(2 × {cloud_retrain_k.value}K / ${7.0 / cloud_drift_days.value:.4f}K/day)
               = {_cloud_t_star:.1f} days
        ```

        **Edge Orin NX:**
        ```
        C_drift = 7pp × $1K/pp/day / {edge_drift_days.value} days = ${7.0 / edge_drift_days.value:.4f}K/day
        T*      = sqrt(2 × {edge_retrain_k.value:.3f}K / ${7.0 / edge_drift_days.value:.4f}K/day)
               = {_edge_t_star:.1f} days
        ```

        **Mobile:**
        ```
        C_drift = 7pp × $1K/pp/day / {mobile_drift_days.value} days = ${7.0 / mobile_drift_days.value:.4f}K/day
        T*      = sqrt(2 × {mobile_retrain_k.value}K / ${7.0 / mobile_drift_days.value:.4f}K/day)
               = {_mobile_t_star:.1f} days
        ```

        ### Optimal Retraining Schedule Summary

        | Environment | Drift Rate | Retrain Cost | T* | Cadence | Acc @ T* | Annual Cost |
        |-------------|-----------|-------------|-----|---------|----------|------------|
        | Cloud H100 | {cloud_drift_days.value}d to &minus;7pp | ${cloud_retrain_k.value}K | {_cloud_t_star:.0f}d | {_t_to_label(_cloud_t_star)} | {_cloud_acc_at_t:.1f}% | ${_cloud_annual:.0f}K |
        | Edge Orin NX | {edge_drift_days.value}d to &minus;7pp | ${edge_retrain_k.value:.3f}K | {_edge_t_star:.0f}d | {_t_to_label(_edge_t_star)} | {_edge_acc_at_t:.1f}% | ${_edge_annual:.1f}K |
        | Mobile NPU | {mobile_drift_days.value}d to &minus;7pp | ${mobile_retrain_k.value}K | {_mobile_t_star:.0f}d | {_t_to_label(_mobile_t_star)} | {_mobile_acc_at_t:.1f}% | ${_mobile_annual:.0f}K |
        """),
        _fail_cell,
        mo.ui.plotly(_fig_cost),
    ])
    return (
        _any_fail, _cloud_fail, _edge_fail, _mobile_fail,
        _cloud_t_star, _edge_t_star, _mobile_t_star,
        _cloud_annual, _edge_annual, _mobile_annual,
        _primary_t_star,
    )


# ─── ACT II: CLOUD VS EDGE COMPARISON ────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, act2_pred):
    mo.stop(act2_pred.value is None)

    _cloud_c = COLORS["Cloud"]
    _edge_c  = COLORS["Edge"]

    mo.vstack([
        mo.md("### Why the Same Drift Threshold Has Different Economic Implications"),
        mo.Html(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0;">
            <div style="background: white; border: 1.5px solid #c7d2fe;
                        border-top: 4px solid {_cloud_c};
                        border-radius: 10px; padding: 18px;">
                <div style="font-weight: 800; color: {_cloud_c}; font-size: 0.95rem;
                            margin-bottom: 10px;">Cloud (H100)</div>
                <div style="font-size: 0.85rem; color: #475569; line-height: 1.75;">
                    <strong>Retraining cost:</strong> $8K/run ($2/hr &times; 4 hrs)<br/>
                    <strong>Deployment:</strong> Model weight swap, seconds of downtime<br/>
                    <strong>Drift rate:</strong> Fast (real-time transactions, seasonal)<br/>
                    <strong>T* shape:</strong> Shorter interval &mdash; cheap retraining,
                    fast drift, high query volume amplifies accuracy loss<br/>
                    <strong>Key lever:</strong> Continuous PSI monitoring is cheap relative
                    to retraining cost &mdash; monitor aggressively, retrain on signal.
                </div>
            </div>
            <div style="background: white; border: 1.5px solid #fecaca;
                        border-top: 4px solid {_edge_c};
                        border-radius: 10px; padding: 18px;">
                <div style="font-weight: 800; color: {_edge_c}; font-size: 0.95rem;
                            margin-bottom: 10px;">Edge (Jetson Orin NX)</div>
                <div style="font-size: 0.85rem; color: #475569; line-height: 1.75;">
                    <strong>Retraining cost:</strong> $50/device OTA push ops overhead<br/>
                    <strong>Deployment:</strong> OTA firmware update, device downtime,
                    field validation required<br/>
                    <strong>Drift rate:</strong> Slower (retail scanner, stable patterns)<br/>
                    <strong>T* shape:</strong> Longer interval &mdash; OTA ops cost penalizes
                    frequent updates; slower drift makes infrequent updates viable<br/>
                    <strong>Key lever:</strong> Each unnecessary push wastes ops budget and
                    risks device instability. Higher PSI threshold tolerated here.
                </div>
            </div>
        </div>
        """),
        mo.callout(mo.md(
            "**The square-root law:** T* = sqrt(2C/C_drift). A 4&times; higher retraining "
            "cost doubles T*, not quadruples it. This means Cloud and Edge environments "
            "with the same accuracy requirements can have T* values that differ by only "
            "2–3&times; even when retraining costs differ by 100&times;. The physics, "
            "not intuition, sets the cadence. The model is the same; the deployment "
            "economics are different."
        ), kind="info"),
    ])
    return


# ─── ACT II: RETRAINING MATHPEEK ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The Governing Equation &mdash; Optimal Retraining Interval T*": mo.md("""
        **Staleness Cost Model** (@eq-staleness-cost, @sec-ml-operations-cost-aware-automation):

        The total operational cost over a retraining interval T has two components:

        ```
        C_retrain_annual = C_retrain × (1 / T)          [fixed cost amortized over interval]
        C_drift_annual   = C_drift × T / 2               [drift cost grows linearly with T]
        C_total          = C_retrain / T  +  C_drift × T / 2
        ```

        Minimizing over T (take derivative, set to zero):

        ```
        dC/dT = −C_retrain / T²  +  C_drift / 2  =  0
        T*    = sqrt(2 × C_retrain / C_drift)
        ```

        **Sensitivity table** (@tbl-retraining-sensitivity):

        | Change | Effect on T* |
        |--------|-------------|
        | 4&times; retraining cost C_retrain | 2&times; longer interval |
        | 4&times; drift severity C_drift | 2&times; shorter interval |
        | 9&times; retraining cost | 3&times; longer interval |
        | 100&times; retraining cost | 10&times; longer interval |

        The square-root relationship is the key insight for system design: you cannot
        halve the optimal interval by halving costs. You must reduce costs by 4&times;
        to halve T*. This is why parallelizing retraining pipelines (reducing
        C_retrain via faster compute) has a *sublinear* effect on cadence improvement.

        **Why accuracy is a lagging indicator** (@sec-ml-operations-mlops-3ea3):

        PSI is an *input* metric — it measures distributional shift before prediction
        quality degrades. By the time accuracy metrics show measurable degradation,
        the model has already been serving degraded predictions for potentially weeks.
        PSI detects the shift earlier, allowing proactive intervention.
        """),
    })
    return


# ─── ACT II: PREDICTION REVEAL ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred):
    _feedback2 = {
        "A": mo.callout(mo.md(
            "**Not quite.** Daily retraining is vastly over-provisioned for environments "
            "where drift rates are measured in weeks. The staleness cost model shows "
            "that the optimal interval T* balances retraining cost against drift cost. "
            "If you retrain daily but drift only causes 7% accuracy loss over 2 months, "
            "you are spending 60&times; more on compute than necessary. The cost model "
            "exists precisely to prevent this over-engineering."
        ), kind="warn"),
        "B": mo.callout(mo.md(
            "**Correct.** Cloud: weekly matches a fast-drifting environment (financial "
            "transactions change with season, economics, user behavior) with moderate "
            "retraining cost ($8K/run). Edge: monthly matches a slower-drifting "
            "environment (retail scanner with more stable patterns) with higher per-event "
            "cost (OTA push overhead). Mobile: quarterly matches the slowest-drifting, "
            "highest-overhead environment. The T* formula confirms this: each environment's "
            "optimal interval scales with sqrt(C_retrain / C_drift), producing the "
            "weekly/monthly/quarterly progression. The key insight is that deployment cost "
            "asymmetry, not accuracy requirements, drives the cadence difference."
        ), kind="success"),
        "C": mo.callout(mo.md(
            "**Not quite.** Cloud monthly is too conservative for a system handling "
            "real-time transactions with fast-drifting financial behavior. The fraud model "
            "scenario shows PSI crossing 0.2 at week 8 (2 months). Monthly Cloud retraining "
            "means the model operates in significant drift for ~3 weeks before each "
            "retraining event. The staleness cost model sets Cloud T* at approximately "
            "2–3 weeks for typical fraud detection economics, not monthly."
        ), kind="warn"),
        "D": mo.callout(mo.md(
            "**This is a common misconception.** Accuracy-triggered retraining uses "
            "accuracy as a signal — but accuracy is a *lagging indicator*. By the time "
            "accuracy drops measurably in production, the model has already been serving "
            "degraded predictions for potentially weeks. PSI detects the *input* shift "
            "before output quality degrades. Additionally, accurate ground-truth labels "
            "in production often require human annotation with days or weeks of delay "
            "(especially for fraud: a transaction may not be confirmed as fraudulent for "
            "30–60 days). PSI-based detection is earlier and does not require ground-truth labels."
        ), kind="warn"),
    }

    if act2_pred.value in _feedback2:
        mo.vstack([
            mo.md("#### Prediction vs. Reality"),
            _feedback2[act2_pred.value],
        ])
    return


# ─── ACT II: LAGGING INDICATOR REFLECTION ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(act2_pred.value is None)

    act2_reflect = mo.ui.radio(
        options={
            "A) Accuracy is hard to measure in production — it requires complex infrastructure": "A",
            "B) By the time accuracy drops measurably, the model has already degraded for potentially weeks — PSI detects earlier because it measures inputs not outputs": "B",
            "C) Accuracy metrics are too noisy to be reliable triggers": "C",
            "D) Accuracy does not correlate with distribution drift": "D",
        },
        label="""**Reflection — Act II**

Why is using accuracy drop as the trigger for retraining inferior to PSI-based monitoring?""",
    )
    mo.vstack([
        mo.md("---"),
        mo.Html("""
        <div style="background: #1e293b; border-radius: 10px; padding: 4px 16px 12px 16px;
                    border-left: 4px solid #4ade80; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #4ade80;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 12px;">
                Act II Reflection
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin: 6px 0 4px 0;">
                Think about what makes PSI a leading indicator versus accuracy as a lagging one.
            </div>
        </div>
        """),
        act2_reflect,
    ])
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    _reflect2_feedback = {
        "A": mo.callout(mo.md(
            "**Not quite.** Measuring accuracy in production is straightforward when "
            "ground-truth labels are available in real time (e.g., click-through rates "
            "in recommendation systems). The fundamental problem is not measurement "
            "difficulty — it is the *time delay* before labels are available and the "
            "fact that accuracy is an output metric that only reflects damage after "
            "it has already occurred."
        ), kind="warn"),
        "B": mo.callout(mo.md(
            "**Correct.** PSI is a *leading indicator* because it measures the input "
            "distribution, not the model output. Distribution shift precedes accuracy "
            "degradation in the causal chain: first the inputs change, then (with a "
            "lag) the model's learned patterns become misaligned, then accuracy drops. "
            "PSI detects the first step. Accuracy monitoring detects the third step "
            "— by which time degraded predictions have already been served. For fraud "
            "detection, the lag between PSI crossing 0.2 and measurable accuracy drop "
            "is typically 2–4 weeks. Weekly PSI monitoring catches this before it costs."
        ), kind="success"),
        "C": mo.callout(mo.md(
            "**Not quite.** Accuracy can be measured reliably in production given "
            "sufficient label volume. The issue is not noise — it is *timing*. Even "
            "with perfectly accurate accuracy measurement, the signal arrives after "
            "the damage has been done. Noise in accuracy measurement is a separate "
            "engineering challenge (addressed by statistical significance testing) "
            "but is not the reason PSI is preferred as a primary trigger."
        ), kind="warn"),
        "D": mo.callout(mo.md(
            "**This is backwards.** Accuracy degradation is *caused* by distribution "
            "drift — they are strongly correlated. The issue is not correlation, it is "
            "*timing*. Distribution drift (measured by PSI) precedes accuracy drop "
            "in the causal chain. If they did not correlate, PSI would not be a useful "
            "early warning system for accuracy."
        ), kind="warn"),
    }

    if act2_reflect.value in _reflect2_feedback:
        mo.vstack([
            mo.md("#### Reflection Answer"),
            _reflect2_feedback[act2_reflect.value],
        ])
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.md("---"),

        # ── KEY TAKEAWAYS ──
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Infrastructure health and model health are independent signals.</strong>
                    A fraud model that dropped 7 percentage points in accuracy maintained 100% uptime,
                    zero error rate, and normal latency for 6 months. PSI crossed the 0.2 critical
                    threshold by week 8 &mdash; four months before the accuracy drop was noticed.
                    Infrastructure monitoring cannot detect this class of failure.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Retraining cadence is a quantitative economic optimization, not a heuristic.</strong>
                    T* = sqrt(2 &times; C_retrain / C_drift) scales with the square root of cost.
                    A 4&times; more expensive retraining run doubles T*, not quadruples it &mdash;
                    explaining why Cloud (T* &asymp; 7&ndash;14 days) and Edge (T* &asymp; 60&ndash;90 days)
                    environments require dramatically different cadences even at identical accuracy targets.
                </div>
                <div>
                    <strong>3. Automation breaks even in 20 weeks.</strong>
                    An automated pipeline costs 80 engineering hours once. Manual retraining costs
                    4 hours/week. After 20 weeks, the manual team spends 100% of capacity on maintenance;
                    the automated team spends 0%. Threshold-triggered retraining (PSI &gt; 0.1)
                    further reduces unnecessary compute versus fixed-interval schedules.
                </div>
            </div>
        </div>
        """),

        # ── CONNECTIONS ──
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <!-- What's Next -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 15: Responsible Engineering.</strong> This lab showed that drift
                    degrades accuracy. Lab 15 asks: when accuracy degrades, does it degrade
                    equally across all demographic groups &mdash; and how do you measure and
                    constrain the disparity before it causes regulatory or reputational harm?
                </div>
            </div>

            <!-- Textbook Connection -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-ml-operations-quantifying-drift-physics-psi-8c11
                    for the PSI derivation and @sec-ml-operations-cost-aware-automation for the
                    full staleness cost model and T* formula.<br/>
                    <strong>Build:</strong> TinyTorch Module 14 &mdash; implement a PSI drift
                    detector and a threshold-triggered retraining scheduler.
                    See <code>tinytorch/src/14_mlops/</code>.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. A model deployed with 94% accuracy sees Jensen-Shannon divergence increase to 0.3 over 26 weeks (lambda=0.10). Using the Degradation Equation, what is the model's accuracy at week 26 — and which infrastructure metric (uptime, P99 latency, error rate) would have flagged this?

    2. The retraining economics formula says: retrain if delta_Accuracy x Value_per_point > Training_Cost + Deployment_Risk. For the Act II simulator, what drift rate (lambda) and domain type produced the fastest-degrading model — and what was the retraining frequency required to keep accuracy above 80%?

    3. MLOps automation costs 80 hours to build but saves 4 hours/week of manual monitoring. At what week does automation break even — and why does a recommendation system (lambda approx 0.15) require daily retraining while an embedded medical device (lambda approx 0.03) can retrain quarterly?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD ───────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    act1_pred, act1_reflect,
    act2_pred, act2_reflect,
    context_toggle,
    _any_fail, _primary_t_star,
    _cloud_t_star, _edge_t_star, _mobile_t_star,
    _cloud_annual, _edge_annual,
    _first_cross_week,
    cloud_drift_days, cloud_retrain_k,
    edge_drift_days, edge_retrain_k,
):
    _act1_correct    = act1_pred.value == "C" if act1_pred.value else False
    _reflect1_correct = act1_reflect.value == "B" if act1_reflect.value else False
    _act2_correct    = act2_pred.value == "B" if act2_pred.value else False
    _reflect2_correct = act2_reflect.value == "B" if act2_reflect.value else False

    _ctx = context_toggle.value
    _t_star_primary = _cloud_t_star if _ctx == "cloud" else _edge_t_star
    _annual_primary = _cloud_annual if _ctx == "cloud" else _edge_annual

    ledger.save(
        chapter=14,
        design={
            "context":             _ctx,
            "drift_rate_days":     float(cloud_drift_days.value if _ctx == "cloud" else edge_drift_days.value),
            "retraining_cadence_days": float(_t_star_primary),
            "retraining_cost_k":   float(cloud_retrain_k.value if _ctx == "cloud" else edge_retrain_k.value),
            "act1_prediction":     act1_pred.value or "none",
            "act1_correct":        _act1_correct,
            "act2_result":         float(_t_star_primary),
            "act2_decision":       f"cloud_{_cloud_t_star:.0f}d_edge_{_edge_t_star:.0f}d_mobile_{_mobile_t_star:.0f}d",
            "constraint_hit":      bool(_any_fail),
            "annual_retrain_cost": float(_annual_primary),
        },
    )

    _track       = ledger.get_track() or "NONE"
    _ctx_display = "Cloud H100" if _ctx == "cloud" else "Edge Orin NX"
    _detect_disp = f"Week {_first_cross_week}" if _first_cross_week else "Not yet"
    _t_star_disp = f"{_t_star_primary:.1f}d"
    _annual_disp = f"${_annual_primary:.0f}K"
    _fail_disp   = "YES" if _any_fail else "NO"
    _fail_color  = COLORS["RedLine"] if _any_fail else COLORS["GreenLine"]
    _a1_color    = "#4ade80" if _act1_correct else "#f87171"
    _a2_color    = "#4ade80" if _act2_correct else "#f87171"

    mo.Html(f"""
    <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;
                padding: 12px 24px; background: #0f172a; border-radius: 12px;
                margin-top: 32px; font-family: 'SF Mono', 'Fira Code', monospace;
                font-size: 0.78rem; border: 1px solid #1e293b;">
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">TRACK</span>
            <span style="color: #e2e8f0; margin-left: 6px;">{_track}</span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">CH14</span>
            <span style="color: #4ade80; margin-left: 6px;">SAVED</span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">CONTEXT</span>
            <span style="color: #e2e8f0; margin-left: 6px;">{_ctx_display}</span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">PSI@THRESH</span>
            <span style="color: #e2e8f0; margin-left: 6px;">{_detect_disp}</span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">ACT1</span>
            <span style="color: {_a1_color}; margin-left: 6px;">
                {act1_pred.value or '—'} {'OK' if _act1_correct else 'INCORRECT'}
            </span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">ACT2</span>
            <span style="color: {_a2_color}; margin-left: 6px;">
                {act2_pred.value or '—'} {'OK' if _act2_correct else 'INCORRECT'}
            </span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">T*</span>
            <span style="color: #e2e8f0; margin-left: 6px;">{_t_star_disp}</span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">ANNUAL</span>
            <span style="color: #e2e8f0; margin-left: 6px;">{_annual_disp}</span>
        </div>
        <div>
            <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">FAILURE</span>
            <span style="color: {_fail_color}; margin-left: 6px;">{_fail_disp}</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
