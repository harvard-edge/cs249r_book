import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-15: THE FAIRNESS BUDGET
#
# Volume II, Chapter 15 — Responsible AI
#
# Core Invariant: Fairness is a budget. The Impossibility Theorem proves you
#   cannot simultaneously satisfy DP, Equalized Odds, and Calibration when
#   base rates differ. The "fairness tax" scales with demographic heterogeneity.
#   Feedback loops amplify bias exponentially. Responsible AI infrastructure
#   has real system costs.
#
# 5 Parts (~60 minutes):
#   Part A — The Impossibility Wall (12 min)
#   Part B — The Fairness Tax (10 min)
#   Part C — The Feedback Loop (12 min)
#   Part D — The Responsible AI Overhead Budget (14 min)
#   Part E — The Fairness Audit Pipeline (12 min)
#
# Design Ledger: saves chapter="v2_15"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 0: SETUP ───────────────────────────────────────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import numpy as np

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install(
            "../../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog
    from mlsysim.hardware.registry import Hardware
    from mlsysim.models.registry import Models

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()

    # ── Hardware from registry (Cloud + Edge tiers) ─────────────────────────
    _cloud = Hardware.Cloud.H100
    _edge  = Hardware.Edge.JetsonOrinNX

    CLOUD_TFLOPS = _cloud.compute.peak_flops.m_as("TFLOPs/s")  # 989
    EDGE_TFLOPS  = _edge.compute.peak_flops.m_as("TFLOPs/s")   # 25

    # ── Fairness constants ──────────────────────────────────────────────────
    # Source: Responsible AI chapter
    BASELINE_ACCURACY   = 85.0    # % — well-calibrated classifier baseline
    DP_TAX_PER_GAP      = 26.0    # pp accuracy tax per 1.0 base rate gap (DP)
    EO_TAX_PER_GAP      = 17.0    # pp tax per 1.0 gap (Equalized Odds)
    EQOP_TAX_PER_GAP    = 10.0    # pp tax per 1.0 gap (Equal Opportunity)

    # Feedback loop amplification
    INITIAL_BIAS         = 0.05   # 5% initial disparity
    AMPLIFICATION_RATE   = 1.35   # multiplicative factor per retraining cycle

    # Overhead constants — Source: chapter responsible AI overhead table
    INFERENCE_LATENCY_MS = 30     # base inference latency
    MONITORING_BASIC_MS  = 10     # basic fairness monitoring overhead
    MONITORING_FULL_MS   = 20     # full monitoring overhead
    LIME_OVERHEAD_MS     = 25     # LIME explainability
    SHAP_OVERHEAD_MS     = 50     # SHAP explainability
    NETWORK_OVERHEAD_MS  = 5      # network + queuing
    SLA_MS               = 100    # latency SLA

    # Audit pipeline
    AUDIT_COMPUTE_BASE   = 1000   # GPU-hours per audit cycle (baseline)

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        CLOUD_TFLOPS, EDGE_TFLOPS,
        BASELINE_ACCURACY, DP_TAX_PER_GAP, EO_TAX_PER_GAP, EQOP_TAX_PER_GAP,
        INITIAL_BIAS, AMPLIFICATION_RATE,
        INFERENCE_LATENCY_MS, MONITORING_BASIC_MS, MONITORING_FULL_MS,
        LIME_OVERHEAD_MS, SHAP_OVERHEAD_MS, NETWORK_OVERHEAD_MS, SLA_MS,
        AUDIT_COMPUTE_BASE, DecisionLog,
    )


# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1a0e28 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 15
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Fairness Budget
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Impossibility &middot; Tax &middot; Feedback Loops &middot; Audit Infrastructure
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                A product manager asks: &ldquo;Can we add a fairness constraint?&rdquo;
                The answer is yes &mdash; but it costs accuracy, latency, and compute.
                And mathematics proves you cannot satisfy all fairness metrics simultaneously.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    5 Parts &middot; ~60 min
                </span>
                <span style="background: rgba(0,143,69,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.25);">
                    Chapter 15: Responsible AI
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-fail">Impossibility Theorem</span>
                <span class="badge badge-warn">7-10pp Fairness Tax</span>
                <span class="badge badge-info">Exponential Feedback Loops</span>
                <span class="badge badge-ok">Audit Pipeline Infrastructure</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Demonstrate the Impossibility Theorem</strong>: show that no single threshold simultaneously satisfies Demographic Parity, Equalized Odds, and Calibration when group base rates differ.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the fairness tax</strong>: measure the 7-10pp accuracy cost of enforcing Demographic Parity at a 30% base rate gap, and identify which fairness metric imposes the lowest tax.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a responsible AI infrastructure</strong> that meets accuracy &gt;80%, fairness disparity &lt;0.05, and latency &lt;100ms SLA simultaneously, discovering the hard trade-offs between monitoring, explainability, and performance.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Prerequisites</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Fairness metrics (DP, Equalized Odds, Calibration) from Responsible AI chapter &middot;
                    Impossibility Theorem (Kleinberg 2016) &middot; Feedback loops from Responsible AI chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Duration</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~60 min</strong><br/>A: 12 &middot; B: 10 &middot; C: 12 &middot; D: 14 &middot; E: 12
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">Core Question</div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Mathematics proves you cannot satisfy all fairness metrics. Feedback loops
                amplify bias exponentially. Monitoring and explainability consume the latency
                budget. Given these constraints, what does a responsible AI system actually cost?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Responsible AI chapter** -- Impossibility Theorem (Kleinberg 2016, Chouldechova 2017),
      fairness metric definitions, and the accuracy-fairness tradeoff.
    - **Responsible AI chapter** (feedback loops) -- Sociotechnical feedback invariant,
      exponential amplification of bias through retraining cycles.
    - **Responsible AI chapter** (overhead table) -- DP-SGD, SHAP, monitoring costs
      and their latency impact at fleet scale.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 4: Part A prediction + controls ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    partA_pred = mo.ui.radio(
        options={
            "A) Yes -- with careful threshold tuning": "yes_tune",
            "B) Yes -- with separate thresholds per group": "yes_separate",
            "C) No -- mathematically impossible when base rates differ": "no_math",
            "D) No -- but only because the model is poorly trained": "no_model",
        },
        label="Two groups with base rates 60% and 30%. Can you find a configuration satisfying BOTH Demographic Parity AND Equalized Odds?",
    )
    partA_threshold_slider = mo.ui.slider(start=0.05, stop=0.95, value=0.5, step=0.05, label="Classification threshold")
    partA_base_rate_a = mo.ui.slider(start=0.1, stop=0.9, value=0.6, step=0.05, label="Group A base rate")
    partA_base_rate_b = mo.ui.slider(start=0.1, stop=0.9, value=0.3, step=0.05, label="Group B base rate")
    return (partA_pred, partA_threshold_slider, partA_base_rate_a, partA_base_rate_b)


# ─── CELL 5: Part B prediction + controls ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partA_pred):
    mo.stop(partA_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_pred = mo.ui.number(
        start=60, stop=90, value=83, step=1,
        label="Baseline accuracy 85%. Groups: 60% vs 30% base rate. After enforcing Demographic Parity, accuracy = ?%",
    )
    partB_gap_slider = mo.ui.slider(start=0.0, stop=0.5, value=0.3, step=0.05, label="Base rate gap (|rate_A - rate_B|)")
    return (partB_pred, partB_gap_slider)


# ─── CELL 6: Part C prediction + controls ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partB_pred):
    mo.stop(partB_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partC_pred = mo.ui.radio(
        options={
            "A) ~8% -- modest growth": "8",
            "B) ~15% -- roughly doubles": "15",
            "C) ~40-60% -- exponential amplification": "50",
            "D) ~5% -- bias is stable": "5",
        },
        label="Initial data has 5% bias. Model retrains on its predictions monthly. After 10 cycles with no intervention, what is the group disparity?",
    )
    partC_data_audit = mo.ui.switch(label="Data Auditing", value=False)
    partC_fairness_constraint = mo.ui.switch(label="Fairness Constraint", value=False)
    partC_output_monitor = mo.ui.switch(label="Output Monitoring", value=False)
    partC_feedback_gov = mo.ui.switch(label="Feedback Governance", value=False)
    return (partC_pred, partC_data_audit, partC_fairness_constraint, partC_output_monitor, partC_feedback_gov)


# ─── CELL 7: Part D prediction + controls ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partC_pred):
    mo.stop(partC_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_pred = mo.ui.radio(
        options={
            "A) Yes -- 30+15+50 = 95ms, under budget": "yes_simple",
            "B) Yes -- but only with optimized monitoring": "yes_opt",
            "C) No -- 95ms is too close to 100ms with no margin": "no_margin",
            "D) No -- you must choose between monitoring and explainability": "no_choose",
        },
        label="Inference: 30ms. Monitoring: 15ms. SHAP: 50ms. SLA: 100ms. Can you fit all three?",
    )
    partD_metric = mo.ui.dropdown(
        options={"Demographic Parity": "dp", "Equalized Odds": "eo", "Equal Opportunity": "eqop"},
        value="Equal Opportunity", label="Fairness metric:",
    )
    partD_monitor = mo.ui.dropdown(
        options={"None (0ms)": "none", "Basic (10ms)": "basic", "Full (20ms)": "full"},
        value="Basic (10ms)", label="Monitoring level:",
    )
    partD_explain = mo.ui.dropdown(
        options={"None (0ms)": "none", "LIME (25ms)": "lime", "SHAP (50ms)": "shap"},
        value="None (0ms)", label="Explainability:",
    )
    return (partD_pred, partD_metric, partD_monitor, partD_explain)


# ─── CELL 8: Part E controls ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, partD_pred):
    mo.stop(partD_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partE_sample = mo.ui.dropdown(
        options={"1% of traffic": 0.01, "5% of traffic": 0.05, "10% of traffic": 0.10},
        value="5% of traffic", label="Sampling rate:",
    )
    partE_test = mo.ui.dropdown(
        options={"Chi-squared (fast, lower power)": "chi2", "Bootstrap (slow, higher power)": "bootstrap"},
        value="Chi-squared (fast, lower power)", label="Statistical test:",
    )
    partE_threshold = mo.ui.slider(start=0.01, stop=0.10, value=0.05, step=0.01, label="Disparity trigger threshold")
    partE_ab_days = mo.ui.slider(start=1, stop=7, value=3, step=1, label="A/B test duration (days)")
    return (partE_sample, partE_test, partE_threshold, partE_ab_days)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 9: TABS CELL ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np, math, apply_plotly_theme, COLORS, ledger,
    BASELINE_ACCURACY, DP_TAX_PER_GAP, EO_TAX_PER_GAP, EQOP_TAX_PER_GAP,
    INITIAL_BIAS, AMPLIFICATION_RATE,
    INFERENCE_LATENCY_MS, MONITORING_BASIC_MS, MONITORING_FULL_MS,
    LIME_OVERHEAD_MS, SHAP_OVERHEAD_MS, NETWORK_OVERHEAD_MS, SLA_MS,
    AUDIT_COMPUTE_BASE,
    partA_pred, partA_threshold_slider, partA_base_rate_a, partA_base_rate_b,
    partB_pred, partB_gap_slider,
    partC_pred, partC_data_audit, partC_fairness_constraint, partC_output_monitor, partC_feedback_gov,
    partD_pred, partD_metric, partD_monitor, partD_explain,
    partE_sample, partE_test, partE_threshold, partE_ab_days,
):
    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- The Impossibility Wall
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div id="part-a" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['RedLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">A</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part A &middot; 12 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Impossibility Wall</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                You believe fairness is a tuning problem: find the right threshold, and all metrics
                align. The Impossibility Theorem proves this is mathematically impossible when
                base rates differ between groups. You must choose which metric to satisfy &mdash;
                and accept that the others will be violated.
            </div>
        </div>
        """))

        items.append(mo.vstack([mo.md("### Your Prediction"), partA_pred]))

        if partA_pred.value is None:
            items.append(mo.callout(mo.md("**Select your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_threshold_slider, partA_base_rate_a, partA_base_rate_b], justify="start", gap=1))

        # ── Impossibility chart ──────────────────────────────────────────
        _threshold = partA_threshold_slider.value
        _br_a = partA_base_rate_a.value
        _br_b = partA_base_rate_b.value
        _gap = abs(_br_a - _br_b)

        # Simulate fairness metrics across threshold range
        _thresholds = np.linspace(0.05, 0.95, 50)

        # Simplified physics: DP gap, EO gap, and Calibration gap as functions of threshold
        # DP gap = |acceptance_rate_A - acceptance_rate_B|
        # When base rates differ, acceptance rates diverge at any single threshold
        _dp_gaps = np.array([abs(
            (1 - t) * _br_a + t * (1 - _br_a) * 0.1 - ((1 - t) * _br_b + t * (1 - _br_b) * 0.1)
        ) for t in _thresholds])
        _dp_gaps = _gap * (1 - 0.5 * np.abs(_thresholds - 0.5))  # peaks at middle thresholds

        # EO gap: TPR difference between groups
        _eo_gaps = np.array([abs(
            min(0.95, _br_a / max(0.01, t)) - min(0.95, _br_b / max(0.01, t))
        ) * 0.5 for t in _thresholds])
        _eo_gaps = np.clip(_eo_gaps, 0, 0.5)

        # Calibration gap: calibration diverges when base rates differ
        _cal_gaps = _gap * 0.3 * np.ones_like(_thresholds) * (1 + 0.5 * np.sin(np.pi * _thresholds))

        # Key insight: at no threshold do ALL three gaps go to zero simultaneously
        _min_total = float('inf')
        _best_threshold = 0.5
        for i, t in enumerate(_thresholds):
            _total = _dp_gaps[i] + _eo_gaps[i] + _cal_gaps[i]
            if _total < _min_total:
                _min_total = _total
                _best_threshold = t

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Scatter(
            x=_thresholds, y=_dp_gaps, name="DP Gap",
            line=dict(color=COLORS["RedLine"], width=2.5),
        ))
        fig_imp.add_trace(go.Scatter(
            x=_thresholds, y=_eo_gaps, name="Equalized Odds Gap",
            line=dict(color=COLORS["BlueLine"], width=2.5),
        ))
        fig_imp.add_trace(go.Scatter(
            x=_thresholds, y=_cal_gaps, name="Calibration Gap",
            line=dict(color=COLORS["OrangeLine"], width=2.5),
        ))

        # Current threshold marker
        _idx = int(_threshold * len(_thresholds) / 1.0)
        _idx = min(_idx, len(_thresholds) - 1)
        fig_imp.add_vline(x=_threshold, line_dash="dash", line_color=COLORS["TextMuted"])

        # Zero line (perfect fairness)
        fig_imp.add_hline(y=0, line_dash="dot", line_color=COLORS["GreenLine"],
                          annotation_text="Perfect fairness (all gaps = 0)", annotation_position="top right")

        fig_imp.update_layout(
            height=380,
            xaxis=dict(title="Classification Threshold"),
            yaxis=dict(title="Fairness Gap (lower = better)", range=[-0.05, 0.5]),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_imp)

        _equal_rates = _gap < 0.05
        _impossibility_msg = (
            "Base rates are equal -- all metrics CAN be simultaneously satisfied."
            if _equal_rates else
            f"Base rate gap = {_gap:.0%}. At every threshold, at least one metric is substantially violated. "
            "The three curves never all reach zero simultaneously."
        )

        items.append(mo.vstack([
            mo.md(f"### Threshold Sweep (Group A: {_br_a:.0%}, Group B: {_br_b:.0%}, Gap: {_gap:.0%})"),
            mo.as_html(fig_imp),
            mo.callout(mo.md(f"**{_impossibility_msg}**"),
                       kind="success" if _equal_rates else "danger"),
            mo.callout(mo.md(
                "**Try it:** Set both base rates equal (e.g., 0.50 and 0.50). The impossibility vanishes. "
                "This confirms the theorem: the constraint is the base rate difference, not the model."
            ), kind="info"),
        ]))

        # Reveal
        _correct = partA_pred.value == "no_math"
        _msg = ("Correct. The Impossibility Theorem (Kleinberg 2016, Chouldechova 2017) proves that "
                "DP, Equalized Odds, and Calibration cannot be simultaneously satisfied when base rates differ. "
                "This is not a training problem -- it is a mathematical constraint."
                if _correct else
                "The answer is (C): it is mathematically impossible. The Impossibility Theorem proves that "
                "with unequal base rates, no single threshold (or pair of thresholds) satisfies all three metrics.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The Fairness Tax
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div id="part-b" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">B</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part B &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Fairness Tax</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                You cannot satisfy all fairness metrics. You must choose one and pay a cost.
                How large is that cost? It scales with the base rate divergence between groups.
            </div>
        </div>
        """))

        items.append(mo.vstack([mo.md("### Your Prediction"), partB_pred]))

        if partB_pred.value is None:
            items.append(mo.callout(mo.md("**Enter your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        items.append(mo.vstack([partB_gap_slider]))

        # ── Fairness tax chart ───────────────────────────────────────────
        _gap = partB_gap_slider.value
        _gaps = np.linspace(0, 0.5, 50)

        # Tax = k * |base_rate_gap| (linear approximation from chapter examples)
        _dp_acc = BASELINE_ACCURACY - DP_TAX_PER_GAP * _gaps
        _eo_acc = BASELINE_ACCURACY - EO_TAX_PER_GAP * _gaps
        _eqop_acc = BASELINE_ACCURACY - EQOP_TAX_PER_GAP * _gaps

        fig_tax = go.Figure()
        fig_tax.add_trace(go.Scatter(x=_gaps, y=_dp_acc, name="Demographic Parity",
                                     line=dict(color=COLORS["RedLine"], width=2.5)))
        fig_tax.add_trace(go.Scatter(x=_gaps, y=_eo_acc, name="Equalized Odds",
                                     line=dict(color=COLORS["BlueLine"], width=2.5)))
        fig_tax.add_trace(go.Scatter(x=_gaps, y=_eqop_acc, name="Equal Opportunity",
                                     line=dict(color=COLORS["GreenLine"], width=2.5)))

        fig_tax.add_vline(x=_gap, line_dash="dash", line_color=COLORS["TextMuted"])

        fig_tax.update_layout(
            height=380,
            xaxis=dict(title="Base Rate Divergence"),
            yaxis=dict(title="Accuracy After Fairness Constraint (%)", range=[60, 90]),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_tax)

        _dp_at_gap = BASELINE_ACCURACY - DP_TAX_PER_GAP * _gap
        _eo_at_gap = BASELINE_ACCURACY - EO_TAX_PER_GAP * _gap
        _eqop_at_gap = BASELINE_ACCURACY - EQOP_TAX_PER_GAP * _gap

        items.append(mo.vstack([
            mo.md(f"### Accuracy vs Base Rate Gap (gap = {_gap:.0%})"),
            mo.as_html(fig_tax),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap;">
                <div style="padding: 14px 20px; border: 2px solid {COLORS['RedLine']}; border-radius: 10px;
                            text-align: center; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">DP Tax</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['RedLine']};">
                        {DP_TAX_PER_GAP * _gap:.1f}pp</div>
                    <div style="font-size: 0.8rem; color: {COLORS['TextSec']};">Acc: {_dp_at_gap:.1f}%</div>
                </div>
                <div style="padding: 14px 20px; border: 2px solid {COLORS['BlueLine']}; border-radius: 10px;
                            text-align: center; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">EO Tax</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['BlueLine']};">
                        {EO_TAX_PER_GAP * _gap:.1f}pp</div>
                    <div style="font-size: 0.8rem; color: {COLORS['TextSec']};">Acc: {_eo_at_gap:.1f}%</div>
                </div>
                <div style="padding: 14px 20px; border: 2px solid {COLORS['GreenLine']}; border-radius: 10px;
                            text-align: center; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">EqOp Tax</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['GreenLine']};">
                        {EQOP_TAX_PER_GAP * _gap:.1f}pp</div>
                    <div style="font-size: 0.8rem; color: {COLORS['TextSec']};">Acc: {_eqop_at_gap:.1f}%</div>
                </div>
            </div>
            """),
        ]))

        # Reveal
        _predicted = partB_pred.value
        _actual = _dp_at_gap
        _diff = abs(_predicted - _actual)
        _msg = (f"You predicted {_predicted:.0f}%. Actual under DP: {_actual:.1f}%. "
                + ("Excellent calibration." if _diff < 3 else
                   "Most students predict 82-84%, expecting a small fixed cost. "
                   "The tax is proportional to the base rate gap."))
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _diff < 3 else "warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- The Feedback Loop
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div id="part-c" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">C</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part C &middot; 12 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Feedback Loop</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                You chose a metric and accepted the tax. The model is deployed. But deployed
                models do not just predict &mdash; they shape the world that generates their
                future training data. A 5% initial bias amplifies to 40-60% disparity within
                10 retraining cycles.
            </div>
        </div>
        """))

        items.append(mo.vstack([mo.md("### Your Prediction"), partC_pred]))

        if partC_pred.value is None:
            items.append(mo.callout(mo.md("**Select your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        items.append(mo.vstack([
            mo.md("### Intervention Controls"),
            mo.md("*Toggle interventions to see their effect on the feedback loop trajectory:*"),
            mo.hstack([partC_data_audit, partC_fairness_constraint, partC_output_monitor, partC_feedback_gov],
                       justify="start", gap=1),
        ]))

        # ── Feedback loop chart ──────────────────────────────────────────
        _iterations = np.arange(0, 11)
        _n_interventions = sum([partC_data_audit.value, partC_fairness_constraint.value,
                                partC_output_monitor.value, partC_feedback_gov.value])

        # Pre-computed trajectories based on intervention combinations
        # Each intervention reduces the amplification rate
        _dampening = 1.0
        if partC_fairness_constraint.value:
            _dampening *= 0.7  # slows but doesn't stop
        if partC_data_audit.value:
            _dampening *= 0.8  # catches late
        if partC_output_monitor.value:
            _dampening *= 0.75
        if partC_feedback_gov.value:
            _dampening *= 0.6  # strongest individual effect

        _effective_rate = AMPLIFICATION_RATE * _dampening
        _no_intervention = np.array([INITIAL_BIAS * AMPLIFICATION_RATE ** i for i in _iterations])
        _with_intervention = np.array([INITIAL_BIAS * _effective_rate ** i for i in _iterations])

        # Cap at 100%
        _no_intervention = np.minimum(_no_intervention, 1.0)
        _with_intervention = np.minimum(_with_intervention, 1.0)

        _final_no_int = _no_intervention[-1] * 100
        _final_with_int = _with_intervention[-1] * 100

        fig_loop = go.Figure()
        fig_loop.add_trace(go.Scatter(
            x=_iterations, y=_no_intervention * 100, name="No Intervention",
            line=dict(color=COLORS["RedLine"], width=3), mode="lines+markers",
        ))
        fig_loop.add_trace(go.Scatter(
            x=_iterations, y=_with_intervention * 100, name=f"With {_n_interventions} Intervention(s)",
            line=dict(color=COLORS["GreenLine"] if _n_interventions >= 3 else COLORS["OrangeLine"], width=3),
            mode="lines+markers",
        ))

        fig_loop.add_hline(y=10, line_dash="dot", line_color=COLORS["OrangeLine"],
                           annotation_text="10% Disparity Threshold", annotation_position="top right")

        fig_loop.update_layout(
            height=380,
            xaxis=dict(title="Retraining Cycle", dtick=1),
            yaxis=dict(title="Group Disparity (%)", range=[0, min(100, max(_final_no_int * 1.2, 70))]),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_loop)

        _broken = _final_with_int < 15
        _status = ("LOOP BROKEN" if _broken else
                   "LOOP SLOWED" if _n_interventions > 0 else
                   "EXPONENTIAL AMPLIFICATION")
        _status_color = COLORS["GreenLine"] if _broken else COLORS["OrangeLine"] if _n_interventions > 0 else COLORS["RedLine"]

        items.append(mo.vstack([
            mo.md(f"### Feedback Loop ({_n_interventions} interventions active)"),
            mo.as_html(fig_loop),
            mo.Html(f"""
            <div style="display: flex; gap: 20px; justify-content: center; margin-top: 12px;">
                <div style="padding: 14px 20px; border: 2px solid {COLORS['RedLine']}; border-radius: 10px;
                            text-align: center; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">No Intervention</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['RedLine']};">
                        {_final_no_int:.0f}%</div>
                </div>
                <div style="padding: 14px 20px; border: 2px solid {_status_color}; border-radius: 10px;
                            text-align: center; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">With Interventions</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {_status_color};">
                        {_final_with_int:.0f}%</div>
                </div>
                <div style="padding: 14px 20px; border: 2px solid {_status_color}; border-radius: 10px;
                            text-align: center; min-width: 160px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Status</div>
                    <div style="font-size: 1.1rem; font-weight: 900; color: {_status_color};">
                        {_status}</div>
                </div>
            </div>
            """),
            mo.callout(mo.md(
                "**Key insight:** No single intervention breaks the loop. Fairness constraint alone slows "
                "but does not stop growth. Data auditing alone catches the problem late. Only the combination "
                "of 3+ interventions (fairness constraint + data auditing + output monitoring) breaks the "
                "exponential trajectory. Feedback loops require structural breaks, not point fixes."
            ), kind="info"),
        ]))

        # Reveal
        _correct = partC_pred.value == "50"
        _msg = ("Correct. A 5% initial bias amplifies to 40-60% disparity within 10 cycles. "
                "Students expect linear drift, not exponential amplification."
                if _correct else
                "The answer is (C): 40-60%. Bias amplification is exponential (1.35x per cycle), "
                "not linear. 0.05 * 1.35^10 = 0.05 * 20.1 = ~100%. In practice, saturating effects "
                "cap it at 40-60%.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- The Responsible AI Overhead Budget
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div id="part-d" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">D</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part D &middot; 14 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Responsible AI Overhead Budget</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Breaking the feedback loop costs compute: monitoring, auditing, re-evaluation.
                How much does responsible AI <em>actually cost</em> in system resources? At fleet
                scale (10B inferences/day), even 10ms overhead = 100M GPU-seconds/day.
            </div>
        </div>
        """))

        items.append(mo.vstack([mo.md("### Your Prediction"), partD_pred]))

        if partD_pred.value is None:
            items.append(mo.callout(mo.md("**Select your prediction to unlock.**"), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_metric, partD_monitor, partD_explain], justify="start", gap=1))

        # ── Overhead budget chart ────────────────────────────────────────
        # Latency calculation
        _monitor_ms = {"none": 0, "basic": MONITORING_BASIC_MS, "full": MONITORING_FULL_MS}[partD_monitor.value]
        _explain_ms = {"none": 0, "lime": LIME_OVERHEAD_MS, "shap": SHAP_OVERHEAD_MS}[partD_explain.value]
        _total_latency = INFERENCE_LATENCY_MS + _monitor_ms + _explain_ms + NETWORK_OVERHEAD_MS

        _sla_ok = _total_latency <= SLA_MS

        # Accuracy calculation (at gap=0.3 baseline)
        _gap = 0.3
        _tax_map = {"dp": DP_TAX_PER_GAP, "eo": EO_TAX_PER_GAP, "eqop": EQOP_TAX_PER_GAP}
        _tax = _tax_map[partD_metric.value] * _gap
        _accuracy = BASELINE_ACCURACY - _tax
        _acc_ok = _accuracy > 80

        # Disparity (monitoring reduces disparity detection time)
        _disparity = 0.08 if partD_monitor.value == "none" else 0.04 if partD_monitor.value == "basic" else 0.02
        _disp_ok = _disparity < 0.05

        # Latency waterfall
        _segments = ["Inference", "Monitoring", "Explainability", "Network"]
        _values = [INFERENCE_LATENCY_MS, _monitor_ms, _explain_ms, NETWORK_OVERHEAD_MS]
        _colors = [COLORS["BlueLine"], COLORS["GreenLine"], COLORS["OrangeLine"], COLORS["TextMuted"]]

        fig_waterfall = go.Figure()
        fig_waterfall.add_trace(go.Bar(
            x=_segments, y=_values, marker_color=_colors,
            text=[f"{v}ms" for v in _values], textposition="inside",
        ))
        fig_waterfall.add_hline(y=SLA_MS, line_dash="dash", line_color=COLORS["RedLine"], line_width=2,
                                annotation_text=f"SLA: {SLA_MS}ms", annotation_position="top right")
        fig_waterfall.update_layout(
            height=300, yaxis=dict(title="Latency (ms)", range=[0, max(120, _total_latency * 1.2)]),
        )
        apply_plotly_theme(fig_waterfall)

        _sla_banner = ""
        if not _sla_ok:
            _sla_banner = f"""<div style="background: {COLORS['RedLL']}; border: 2px solid {COLORS['RedLine']};
                             border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 12px;
                             font-weight: 700; color: {COLORS['RedLine']};">
                             SLA VIOLATED: {_total_latency}ms &gt; {SLA_MS}ms</div>"""

        _all_ok = _sla_ok and _acc_ok and _disp_ok

        items.append(mo.vstack([
            mo.Html(_sla_banner) if _sla_banner else mo.md(""),
            mo.as_html(fig_waterfall),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap;">
                <div style="padding: 12px 18px; border: 2px solid {'#008F45' if _acc_ok else COLORS['RedLine']};
                            border-radius: 10px; text-align: center; min-width: 140px;">
                    <div style="font-size: 0.65rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Accuracy</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {'#008F45' if _acc_ok else COLORS['RedLine']};">
                        {_accuracy:.1f}%</div>
                    <div style="font-size: 0.65rem; color: {COLORS['TextSec']};">target: &gt;80%</div>
                </div>
                <div style="padding: 12px 18px; border: 2px solid {'#008F45' if _disp_ok else COLORS['RedLine']};
                            border-radius: 10px; text-align: center; min-width: 140px;">
                    <div style="font-size: 0.65rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Disparity</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {'#008F45' if _disp_ok else COLORS['RedLine']};">
                        {_disparity:.2f}</div>
                    <div style="font-size: 0.65rem; color: {COLORS['TextSec']};">target: &lt;0.05</div>
                </div>
                <div style="padding: 12px 18px; border: 2px solid {'#008F45' if _sla_ok else COLORS['RedLine']};
                            border-radius: 10px; text-align: center; min-width: 140px;">
                    <div style="font-size: 0.65rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Latency</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {'#008F45' if _sla_ok else COLORS['RedLine']};">
                        {_total_latency}ms</div>
                    <div style="font-size: 0.65rem; color: {COLORS['TextSec']};">target: &lt;{SLA_MS}ms</div>
                </div>
            </div>
            """),
            mo.callout(mo.md(
                "**All constraints met.** Equal Opportunity + Basic Monitoring + On-demand LIME "
                "is the sweet spot: lowest fairness tax, adequate monitoring, SLA-compliant."
                if _all_ok else
                "**Adjust your configuration.** Full SHAP + Full Monitoring = SLA violation. "
                "Try Equal Opportunity (lower tax) + Basic Monitoring + LIME or None."
            ), kind="success" if _all_ok else "warn"),
        ]))

        # Reveal
        _correct = partD_pred.value == "no_choose"
        _msg = ("Correct. You must choose between full monitoring and SHAP explainability. "
                "30+15+50+5 = 100ms leaves zero margin for queuing variance. "
                "In production, you need at least 10-15% margin."
                if _correct else
                "The answer is (D). Simple addition (30+15+50=95ms) forgets network overhead (5ms) "
                "and queuing variance. At 100ms total, zero margin means any spike violates the SLA.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART E BUILDER -- The Fairness Audit Pipeline
    # ─────────────────────────────────────────────────────────────────────

    def build_part_e():
        items = []

        items.append(mo.Html(f"""
        <div id="part-e" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['TextMuted']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">E</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part E &middot; 12 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px;">The Fairness Audit Pipeline</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Fairness is not a one-time certification. It requires continuous monitoring and
                remediation. The audit pipeline has four stages, each with compute cost and
                detection latency. Your job: minimize the fairness debt window within budget.
            </div>
        </div>
        """))

        items.append(mo.vstack([
            mo.md("### Configure the Audit Pipeline"),
            mo.hstack([partE_sample, partE_test], justify="start", gap=1),
            mo.hstack([partE_threshold, partE_ab_days], justify="start", gap=1),
        ]))

        # ── Audit pipeline calculations ──────────────────────────────────
        _sample_rate = partE_sample.value
        _test_type = partE_test.value
        _disp_threshold = partE_threshold.value
        _ab_days = partE_ab_days.value

        # Detection latency: higher sampling = faster detection, lower threshold = slower detection
        _base_detect_hours = 24 / (_sample_rate * 100)  # inversely proportional to sampling
        _threshold_factor = 0.05 / _disp_threshold  # lower threshold = need more samples
        _test_factor = 1.0 if _test_type == "chi2" else 2.0  # bootstrap is slower
        _detection_hours = _base_detect_hours * _threshold_factor * _test_factor

        # Remediation time = detection + retraining (24h) + A/B test
        _retrain_hours = 24
        _ab_hours = _ab_days * 24
        _total_remediation_hours = _detection_hours + _retrain_hours + _ab_hours

        # Compute cost
        _compute_gpu_hours = AUDIT_COMPUTE_BASE * (_sample_rate / 0.05) * (1 if _test_type == "chi2" else 3)
        _ab_compute = _ab_days * 500  # GPU-hours for A/B testing
        _total_compute = _compute_gpu_hours + _ab_compute

        # Fairness debt window (hours where violation is active but undetected)
        _debt_window_hours = _detection_hours

        _debt_color = COLORS["GreenLine"] if _debt_window_hours < 12 else COLORS["OrangeLine"] if _debt_window_hours < 48 else COLORS["RedLine"]

        items.append(mo.vstack([
            mo.md("### Audit Pipeline Results"),
            mo.Html(f"""
            <div style="display: flex; gap: 20px; justify-content: center; margin-top: 8px; flex-wrap: wrap;">
                <div style="padding: 16px 20px; border: 2px solid {_debt_color}; border-radius: 12px;
                            text-align: center; min-width: 180px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Detection Latency</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {_debt_color};">
                        {_detection_hours:.1f}h</div>
                    <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">fairness debt window</div>
                </div>
                <div style="padding: 16px 20px; border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            text-align: center; min-width: 180px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Total Remediation</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['BlueLine']};">
                        {_total_remediation_hours:.0f}h</div>
                    <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">detect + retrain + verify</div>
                </div>
                <div style="padding: 16px 20px; border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            text-align: center; min-width: 180px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Compute Cost</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['OrangeLine']};">
                        {_total_compute:,.0f}</div>
                    <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">GPU-hours per audit</div>
                </div>
            </div>
            """),
            mo.callout(mo.md(
                f"**Trade-off:** Aggressive auditing (high sampling, low threshold) catches violations in "
                f"**{_detection_hours:.1f}h** but costs **{_total_compute:,.0f} GPU-hours**. Conservative auditing "
                f"is cheap but misses small persistent disparities. The fairness debt window of "
                f"**{_debt_window_hours:.1f}h** represents the time violations affect real users before detection."
            ), kind="info"),
        ]))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

    def build_synthesis():
        items = []

        items.append(mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Fairness is mathematically constrained, not just technically difficult.</strong>
                    The Impossibility Theorem proves that DP, Equalized Odds, and Calibration cannot
                    be simultaneously satisfied when base rates differ. You must choose which metric
                    to prioritize.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. The fairness tax scales with demographic heterogeneity.</strong>
                    At 30% base rate gap, DP costs ~8pp, EO costs ~5pp, Equal Opportunity costs ~3pp.
                    The choice of metric is itself a design decision with quantifiable cost.
                </div>
                <div>
                    <strong>3. Responsible AI is infrastructure, not a checkbox.</strong>
                    Feedback loops amplify bias exponentially (5% to 40-60% in 10 cycles).
                    Breaking them requires multiple simultaneous interventions. Monitoring and
                    explainability consume the latency budget. Every responsible AI decision
                    is a systems engineering decision.
                </div>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">What's Next</div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 16: The Fleet Synthesis (Capstone)</strong> -- Your fairness overhead,
                    carbon cap, and robustness budget all feed into the final fleet design. Every
                    principle interacts. No single-axis optimization works.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">Textbook Connection</div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> Responsible AI chapter for the Impossibility Theorem
                    derivation and feedback loop analysis.<br/>
                    <strong>Feeds into:</strong> V2-16 Capstone (fairness as fleet constraint).
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A -- The Impossibility Wall":            build_part_a(),
        "Part B -- The Fairness Tax":                  build_part_b(),
        "Part C -- The Feedback Loop":                 build_part_c(),
        "Part D -- The Responsible AI Overhead Budget": build_part_d(),
        "Part E -- The Fairness Audit Pipeline":       build_part_e(),
        "Synthesis":                                    build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 10: LEDGER HUD ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, ledger, COLORS, partD_metric):
    _metric_val = partD_metric.value if hasattr(partD_metric, 'value') else "eqop"
    ledger.save(chapter=15, design={
        "chapter": "v2_15",
        "fairness_metric": _metric_val,
        "fairness_overhead_ms": 15,
        "fairness_disparity_threshold": 0.05,
    })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-15: The Fairness Budget</span>
        <span class="hud-label">LEDGER</span>
        <span class="hud-active">Saved (ch15)</span>
        <span class="hud-label">NEXT</span>
        <span class="hud-value">V2-16: The Fleet Synthesis (Capstone)</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
