import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 15: THERE IS NO FREE FAIRNESS
#
# Chapter: Responsible Engineering (@sec-responsible-engineering)
# Core Invariant:
#   Chouldechova's (2017) impossibility theorem — when base rates differ between
#   groups, you CANNOT simultaneously achieve equal false positive rates, equal
#   false negative rates, AND equal positive predictive value. Equal accuracy
#   is not evidence of equal treatment.
#
# 2 Contexts: Cloud (H100) vs Mobile (Smartphone NPU)
#
# Act I  — The Fairness Illusion (12–15 min)
#   Stakeholder: Product Compliance Officer
#   Scenario: Loan approval model, 85% accuracy on both groups. Legal says
#   compliant. Advocates say unfair. Who is right?
#   Prediction: What does equal accuracy actually guarantee?
#   Instrument: Fairness metric explorer — sliders for base_rate_a, base_rate_b,
#   model_threshold; per-group confusion matrices; FPR, FNR, PPV, equalized
#   odds gap; prediction-vs-reality overlay.
#
# Act II — The Audit-Accuracy Tradeoff (20–25 min)
#   Stakeholder: Engineering VP
#   Scenario: Resume screening at scale. Choose fairness strategy + mitigation
#   method + audit frequency. Design the deployment.
#   Failure state: equalized_odds_gap > 10% AND context is high-stakes.
#   Reflection: Why demographic parity can produce unfair outcomes.
#
# Design Ledger: chapter=15
#   context, fairness_criterion, equalized_odds_gap, audit_cost_k,
#   act1_prediction, act1_correct, act2_result, act2_decision,
#   constraint_hit, regulatory_risk
# ─────────────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═════════════════════════════════════════════════════════════════════════════

# ── CELL 0: SETUP (hide_code=False — leave visible) ───────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    # ── Hardware constants — plain floats, sources annotated ──────────────────
    H100_BW_GBS      = 3350   # GB/s — H100 SXM5 HBM3e, NVIDIA spec
    H100_TFLOPS_FP16 = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB      = 80     # GB HBM — NVIDIA H100 SXM5
    MOBILE_BW_GBS    = 68     # GB/s — Apple A17-class smartphone NPU
    MOBILE_TOPS_INT8 = 35     # TOPS INT8 — Apple A17-class NPU
    MOBILE_RAM_GB    = 8      # GB — typical flagship smartphone

    # ── Domain constants — responsible_engr.qmd scenario ─────────────────────
    # Baseline model accuracy on both groups (the "85% parity illusion" scenario)
    BASE_ACCURACY           = 0.85   # overall accuracy on each group

    # EEOC 4/5ths (80%) rule: selection rate ratio < 0.8 triggers review
    # Equivalent to equalized odds gap threshold for high-stakes deployment
    DISPARATE_IMPACT_THRESHOLD_PP = 10.0  # pp — EEOC / OFCCP guidance for hiring

    # Audit cost model: cloud vs mobile context
    # Cloud: H100 batch re-evaluation, near-realtime, ~$2/hr compute
    CLOUD_AUDIT_COST_K_USD  = 1.5    # $K per audit run (cloud, automated)
    # Mobile: delayed centralized audit, human review + compute
    MOBILE_AUDIT_COST_K_USD = 4.0    # $K per audit run (mobile, manual + compute)

    # Annual audit frequencies (runs/year)
    AUDIT_FREQ_OPTIONS = {
        "Continuous (52×/yr)":  52,
        "Monthly (12×/yr)":     12,
        "Quarterly (4×/yr)":     4,
        "Annual (1×/yr)":        1,
    }

    ledger = DesignLedger()
    return (
        mo, go, np, math,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
        MOBILE_BW_GBS, MOBILE_TOPS_INT8, MOBILE_RAM_GB,
        BASE_ACCURACY, DISPARATE_IMPACT_THRESHOLD_PP,
        CLOUD_AUDIT_COST_K_USD, MOBILE_AUDIT_COST_K_USD,
        AUDIT_FREQ_OPTIONS,
    )


# ── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 15
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                There Is No Free Fairness
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 720px; line-height: 1.65;">
                Your model has <strong style="color:#f8fafc;">85% accuracy on both groups</strong>.
                Legal says compliant. Advocates say unfair. Someone is about to be proven wrong.
                Chouldechova (2017) settled this debate mathematically.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: The Fairness Illusion &middot; Act II: Audit-Accuracy Tradeoff
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Disparate impact failure state active
                </span>
                <span style="background: rgba(204,85,0,0.15); color: #fdba74;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(204,85,0,0.25);">
                    {COLORS['Mobile'] and 'Cloud vs Mobile context'}
                </span>
            </div>
        </div>
        """),
    ])
    return


# ── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
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
                <div style="margin-bottom: 3px;">1. <strong>Identify why equal accuracy across demographic groups does not imply equal false positive or false negative rates, using the Chouldechova impossibility theorem.</strong></div>
                <div style="margin-bottom: 3px;">2. <strong>Compare three fairness criteria (accuracy-only, demographic parity, equalized odds) and predict the accuracy cost of each relative to an unconstrained baseline.</strong></div>
                <div style="margin-bottom: 3px;">3. <strong>Design a fairness audit strategy within a $50K annual budget that satisfies equalized odds constraints while maintaining recall above 80%.</strong></div>
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
                    Fairness criteria definitions from @sec-responsible-engineering-fairness-metrics &middot;
                    Chouldechova impossibility theorem from @sec-responsible-engineering-impossibility
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
                "Your loan model has 85% accuracy on both demographic groups, which your legal team calls compliant &mdash; so why do advocacy groups have a valid mathematical argument that the model is biased, and can you make it fairer without destroying its accuracy?"
            </div>
        </div>
    </div>
    """)
    return


# ── CELL 3: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-responsible-engineering-fairness-metrics** — Demographic parity, equalized odds,
      calibration, and individual fairness: formal definitions and what each criterion
      requires from the underlying data distribution.
    - **@sec-responsible-engineering-impossibility** — Chouldechova (2017) impossibility
      theorem: when base rates differ, FPR equality, FNR equality, and PPV equality cannot
      all hold simultaneously. This is a mathematical proof, not a policy preference.
    - **@sec-responsible-engineering-audit-pipelines** — Continuous monitoring, audit
      frequency, mitigation methods, and the compute and accuracy costs of fairness
      enforcement at deployment scale.
    """), kind="info")
    return


# ── CELL 4: CONTEXT TOGGLE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (H100 — automated continuous audit)": "cloud",
            "Mobile (Smartphone NPU — delayed centralized audit)": "mobile",
        },
        value="Cloud (H100 — automated continuous audit)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md(
            "**Select your deployment context.** "
            "Cloud enables near-realtime bias monitoring at low marginal cost. "
            "Mobile deployments face delayed feedback loops and higher per-audit cost."
        ),
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
    _act_title = "The Fairness Illusion"
    _act_duration = "12\u201315 min"
    _act_why = (
        "You assume equal accuracy across demographic groups means equal treatment. "
        "The Chouldechova impossibility theorem proves this cannot be true when base rates "
        "differ between groups \u2014 a calibrated model with equal accuracy structurally "
        "produces unequal false positive and false negative rates. This is mathematics, "
        "not a training failure."
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


# ── CELL 6: ACT1_STAKEHOLDER ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["OrangeLine"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act I — The Fairness Illusion"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: #fff7ed;
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; Product Compliance Officer
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "Our loan approval model has 85% accuracy on both demographic groups.
                Our legal team says we're compliant — equal accuracy means equal treatment.
                But advocacy groups are threatening litigation, saying our false positive
                and false negative rates differ by group. Who is right?
                Do we have a problem or don't we?"
            </div>
        </div>
        """),
        mo.md("""
        The Compliance Officer is measuring the right thing — accuracy — but measuring it
        at the wrong level of resolution. Equal overall accuracy is consistent with wildly
        different error distributions between groups, especially when base rates differ.

        Before touching the explorer, commit to a prediction.
        """),
    ])
    return


# ── CELL 5: ACT I PREDICTION LOCK ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) 85% accuracy on both groups proves fairness — equal accuracy means equal treatment": "option_a",
            "B) Need to check precision (PPV) — equal accuracy can mask unequal precision rates": "option_b",
            "C) Equal accuracy is necessary but not sufficient — must check FPR AND FNR separately": "option_c",
            "D) Accuracy alone is sufficient for legal compliance under disparate impact doctrine": "option_d",
        },
        label=(
            "The Compliance Officer claims equal accuracy proves fairness. "
            "What is the most complete and correct assessment?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #f59e0b; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #fbbf24;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock — Act I
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-bottom: 12px;">
                Commit before touching any sliders. The Fairness Metric Explorer
                unlocks once you select an answer.
            </div>
        </div>
        """),
        act1_prediction,
    ])
    return (act1_prediction,)


# ── CELL 6: ACT I GATE ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction):
    mo.stop(
        act1_prediction.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Fairness Metric Explorer."),
            kind="warn",
        ),
    )
    return


# ── CELL 7: ACT I CONTROLS ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # Base rate sliders: the "positive rate" in each group's true population
    # Source: responsible_engr.qmd — base rate is P(Y=1|Group=G)
    # Range: 5–50% represents realistic lending/credit scenarios
    base_rate_a = mo.ui.slider(
        start=5, stop=50, value=10, step=1,
        label="Base rate — Group A (true positive rate in population, %)",
    )
    base_rate_b = mo.ui.slider(
        start=5, stop=50, value=40, step=1,
        label="Base rate — Group B (true positive rate in population, %)",
    )
    # Model threshold: decision boundary applied uniformly to both groups
    # Source: responsible_engr.qmd — single shared threshold is a policy choice
    model_threshold = mo.ui.slider(
        start=0.10, stop=0.90, value=0.50, step=0.05,
        label="Model classification threshold (applied uniformly to both groups)",
    )
    mo.vstack([
        mo.md("### Fairness Metric Explorer"),
        mo.md(
            "Set the **true base rate** for each group (the actual fraction of creditworthy "
            "applicants in each population) and the **model threshold**. "
            "The model is calibrated — its score reflects true probability — but the same "
            "threshold is applied to both groups. Watch what happens to FPR and FNR."
        ),
        mo.hstack([base_rate_a, base_rate_b], justify="start", gap="2rem"),
        model_threshold,
    ])
    return (base_rate_a, base_rate_b, model_threshold)


# ── CELL 8: ACT I PHYSICS ENGINE + CONFUSION MATRICES ─────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np,
    base_rate_a, base_rate_b, model_threshold,
    apply_plotly_theme,
):
    # ── Simulation physics ─────────────────────────────────────────────────────
    # Model: calibrated binary classifier, score s ~ Beta(alpha, beta) for each group.
    # Calibration means P(Y=1|s=p) = p, so the threshold determines the operating point.
    #
    # For a calibrated model with base rate r and threshold t:
    #   TPR = P(s > t | Y=1)  — sensitivity
    #   FPR = P(s > t | Y=0)  — false positive rate (false alarm)
    #   FNR = 1 - TPR         — miss rate
    #   PPV = r*TPR / (r*TPR + (1-r)*FPR)  — precision / positive predictive value
    #
    # Source: responsible_engr.qmd — confusion matrix decomposition by group,
    #         Chouldechova (2017) impossibility theorem derivation.
    #
    # We model TPR and FPR as functions of threshold for a logistic-like score distribution.
    # Using a simple normal approximation: scores for positives ~ N(0.7, 0.15),
    #                                      scores for negatives ~ N(0.3, 0.15)
    # This is a calibrated model because mean(positive scores) > mean(negative scores)
    # in proportion to base rates, and both groups share the SAME score distributions
    # (this is the "calibrated but disparate impact" scenario).

    def _compute_group_metrics(base_rate_pct, threshold):
        """
        Compute confusion matrix entries and derived metrics for one group.
        Source: responsible_engr.qmd — Section on fairness metric decomposition.

        Uses a calibrated model where positive class scores ~ N(0.70, 0.15)
        and negative class scores ~ N(0.30, 0.15).  The same model/threshold
        is applied to both groups, but the GROUP's base rate changes the
        fraction of positives vs negatives, and hence the confusion matrix.
        """
        from scipy.special import erfc
        r = base_rate_pct / 100.0   # base rate as fraction

        # TPR = P(score > t | Y=1) = P(N(0.70, 0.15) > t)
        # FPR = P(score > t | Y=0) = P(N(0.30, 0.15) > t)
        mu_pos, mu_neg, sigma = 0.70, 0.30, 0.15
        tpr = 0.5 * erfc((threshold - mu_pos) / (sigma * np.sqrt(2)))
        fpr = 0.5 * erfc((threshold - mu_neg) / (sigma * np.sqrt(2)))

        # Clamp to valid range
        tpr = float(np.clip(tpr, 0.001, 0.999))
        fpr = float(np.clip(fpr, 0.001, 0.999))
        fnr = 1.0 - tpr

        # Accuracy = r*TPR + (1-r)*(1-FPR)
        accuracy = r * tpr + (1.0 - r) * (1.0 - fpr)

        # PPV = precision = TP / (TP + FP) = r*TPR / (r*TPR + (1-r)*FPR)
        denom = r * tpr + (1.0 - r) * fpr
        ppv = (r * tpr / denom) if denom > 1e-9 else 0.0

        # Approval rate = P(score > t) = r*TPR + (1-r)*FPR
        approval_rate = denom

        return {
            "accuracy":      accuracy,
            "tpr":           tpr,
            "fpr":           fpr,
            "fnr":           fnr,
            "ppv":           ppv,
            "approval_rate": approval_rate,
        }

    # ── Attempt scipy import; fall back to pure numpy ─────────────────────────
    try:
        from scipy.special import erfc as _erfc_check
        _scipy_ok = True
    except ImportError:
        _scipy_ok = False

    def _compute_group_metrics_numpy(base_rate_pct, threshold):
        """Fallback without scipy — uses numpy erf approximation."""
        r = base_rate_pct / 100.0
        mu_pos, mu_neg, sigma = 0.70, 0.30, 0.15
        tpr = 0.5 * (1.0 - np.sign(threshold - mu_pos) *
                     (1.0 - np.exp(-(threshold - mu_pos)**2 / (2*sigma**2))))
        # More robust: use numpy's erf
        tpr = float(np.clip(0.5 * (1.0 + np.sign(mu_pos - threshold) *
                   (1.0 - np.exp(-(abs(threshold - mu_pos))**2 / sigma**2 * 0.5))), 0.001, 0.999))
        fpr = float(np.clip(0.5 * (1.0 + np.sign(mu_neg - threshold) *
                   (1.0 - np.exp(-(abs(threshold - mu_neg))**2 / sigma**2 * 0.5))), 0.001, 0.999))
        fnr = 1.0 - tpr
        accuracy = r * tpr + (1.0 - r) * (1.0 - fpr)
        denom = r * tpr + (1.0 - r) * fpr
        ppv = (r * tpr / denom) if denom > 1e-9 else 0.0
        approval_rate = denom
        return {"accuracy": accuracy, "tpr": tpr, "fpr": fpr,
                "fnr": fnr, "ppv": ppv, "approval_rate": approval_rate}

    _compute_fn = _compute_group_metrics if _scipy_ok else _compute_group_metrics_numpy

    _br_a = base_rate_a.value   # e.g., 10%
    _br_b = base_rate_b.value   # e.g., 40%
    _t    = model_threshold.value  # e.g., 0.50

    _ga = _compute_fn(_br_a, _t)
    _gb = _compute_fn(_br_b, _t)

    # ── Derived gap metrics ────────────────────────────────────────────────────
    _fpr_gap  = abs(_ga["fpr"] - _gb["fpr"]) * 100   # pp
    _fnr_gap  = abs(_ga["fnr"] - _gb["fnr"]) * 100   # pp
    _ppv_gap  = abs(_ga["ppv"] - _gb["ppv"]) * 100   # pp
    _acc_gap  = abs(_ga["accuracy"] - _gb["accuracy"]) * 100  # pp
    _eo_gap   = (_fpr_gap + _fnr_gap) / 2.0           # equalized odds gap (average)
    _app_gap  = abs(_ga["approval_rate"] - _gb["approval_rate"]) * 100  # pp

    # ── Color coding: green if gap ≤5pp, orange if 5–10pp, red if >10pp ───────
    def _gap_color(gap_pp):
        if gap_pp <= 5.0:
            return "#22c55e"   # green
        elif gap_pp <= 10.0:
            return "#f59e0b"   # orange
        return "#ef4444"       # red

    def _gap_label(gap_pp):
        if gap_pp <= 5.0:
            return "Within 5pp"
        elif gap_pp <= 10.0:
            return "Caution 5–10pp"
        return "Gap >10pp"

    # ── Pareto sweep: vary threshold 0.1→0.9, compute EO gap + accuracy ───────
    _thresholds = np.linspace(0.1, 0.9, 80)
    _pareto_acc  = []
    _pareto_eo   = []
    for _th in _thresholds:
        _g_a = _compute_fn(_br_a, _th)
        _g_b = _compute_fn(_br_b, _th)
        _avg_acc = 0.5 * (_g_a["accuracy"] + _g_b["accuracy"])
        _eo = 0.5 * (abs(_g_a["fpr"] - _g_b["fpr"]) + abs(_g_a["fnr"] - _g_b["fnr"])) * 100
        _pareto_acc.append(_avg_acc * 100)
        _pareto_eo.append(_eo)

    # ── Figure: confusion matrices as bar chart side-by-side ──────────────────
    _categories = ["Accuracy", "TPR", "FPR", "FNR", "PPV", "Approval Rate"]
    _vals_a = [
        _ga["accuracy"]*100, _ga["tpr"]*100, _ga["fpr"]*100,
        _ga["fnr"]*100,       _ga["ppv"]*100, _ga["approval_rate"]*100,
    ]
    _vals_b = [
        _gb["accuracy"]*100, _gb["tpr"]*100, _gb["fpr"]*100,
        _gb["fnr"]*100,       _gb["ppv"]*100, _gb["approval_rate"]*100,
    ]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name=f"Group A (base rate {_br_a}%)",
        x=_categories, y=_vals_a,
        marker_color="#6366f1",
        text=[f"{v:.1f}%" for v in _vals_a],
        textposition="outside",
    ))
    _fig.add_trace(go.Bar(
        name=f"Group B (base rate {_br_b}%)",
        x=_categories, y=_vals_b,
        marker_color="#f59e0b",
        text=[f"{v:.1f}%" for v in _vals_b],
        textposition="outside",
    ))
    _fig = apply_plotly_theme(_fig)
    _fig.update_layout(
        title=f"Per-Group Fairness Metrics (threshold = {_t:.2f})",
        yaxis_title="Rate (%)",
        barmode="group",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_range=[0, 115],
    )

    # ── Figure: accuracy vs EO gap Pareto frontier ────────────────────────────
    _current_eo_for_plot = _eo_gap
    _current_acc_for_plot = 0.5 * (_ga["accuracy"] + _gb["accuracy"]) * 100

    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(
        x=_pareto_acc, y=_pareto_eo,
        mode="lines",
        name="Accuracy vs EO Gap frontier",
        line=dict(color="#6366f1", width=2),
    ))
    _fig2.add_trace(go.Scatter(
        x=[_current_acc_for_plot], y=[_current_eo_for_plot],
        mode="markers",
        name="Current threshold",
        marker=dict(color="#f59e0b", size=12, symbol="star"),
    ))
    # Mark the 10pp gap threshold line
    _fig2.add_hline(
        y=10, line_dash="dash", line_color="#ef4444",
        annotation_text="10pp gap threshold (EEOC guidance)",
        annotation_position="top right",
    )
    _fig2 = apply_plotly_theme(_fig2)
    _fig2.update_layout(
        title="Accuracy vs Equalized Odds Gap — Pareto Frontier",
        xaxis_title="Average Accuracy (%)",
        yaxis_title="Equalized Odds Gap (pp)",
        height=340,
    )

    # ── Physics formula display ────────────────────────────────────────────────
    _physics_md = f"""
### Physics

```
Group A (base rate = {_br_a}%):
  TPR = P(score > {_t:.2f} | Y=1) = {_ga['tpr']*100:.1f}%
  FPR = P(score > {_t:.2f} | Y=0) = {_ga['fpr']*100:.1f}%
  FNR = 1 - TPR                  = {_ga['fnr']*100:.1f}%
  PPV = r·TPR / (r·TPR + (1-r)·FPR)
      = {_br_a/100:.2f}×{_ga['tpr']:.3f} / ({_br_a/100:.2f}×{_ga['tpr']:.3f} + {1-_br_a/100:.2f}×{_ga['fpr']:.3f})
      = {_ga['ppv']*100:.1f}%
  Accuracy = r·TPR + (1-r)·(1-FPR) = {_ga['accuracy']*100:.1f}%

Group B (base rate = {_br_b}%):
  TPR = {_gb['tpr']*100:.1f}%  |  FPR = {_gb['fpr']*100:.1f}%
  FNR = {_gb['fnr']*100:.1f}%  |  PPV = {_gb['ppv']*100:.1f}%
  Accuracy = {_gb['accuracy']*100:.1f}%
```

### Gap Summary

| Metric | Group A | Group B | Gap | Status |
|--------|---------|---------|-----|--------|
| Accuracy | {_ga['accuracy']*100:.1f}% | {_gb['accuracy']*100:.1f}% | {_acc_gap:.1f}pp | {_gap_label(_acc_gap)} |
| FPR | {_ga['fpr']*100:.1f}% | {_gb['fpr']*100:.1f}% | {_fpr_gap:.1f}pp | {_gap_label(_fpr_gap)} |
| FNR | {_ga['fnr']*100:.1f}% | {_gb['fnr']*100:.1f}% | {_fnr_gap:.1f}pp | {_gap_label(_fnr_gap)} |
| PPV | {_ga['ppv']*100:.1f}% | {_gb['ppv']*100:.1f}% | {_ppv_gap:.1f}pp | {_gap_label(_ppv_gap)} |
| Equalized Odds Gap | — | — | {_eo_gap:.1f}pp | {_gap_label(_eo_gap)} |
"""

    # ── Metric cards ──────────────────────────────────────────────────────────
    _cards_html = f"""
<div style="display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; margin: 16px 0;">
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 160px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Accuracy Gap</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_gap_color(_acc_gap)};">
            {_acc_gap:.1f}pp
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">{_gap_label(_acc_gap)}</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 160px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">FPR Gap</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_gap_color(_fpr_gap)};">
            {_fpr_gap:.1f}pp
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">{_gap_label(_fpr_gap)}</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 160px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">FNR Gap</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_gap_color(_fnr_gap)};">
            {_fnr_gap:.1f}pp
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">{_gap_label(_fnr_gap)}</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 160px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">PPV Gap</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_gap_color(_ppv_gap)};">
            {_ppv_gap:.1f}pp
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">{_gap_label(_ppv_gap)}</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 160px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">EO Gap</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_gap_color(_eo_gap)};">
            {_eo_gap:.1f}pp
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">avg FPR+FNR gap</div>
    </div>
</div>
"""

    mo.vstack([
        mo.md(_physics_md),
        mo.Html(_cards_html),
        mo.ui.plotly(_fig),
        mo.ui.plotly(_fig2),
    ])
    return (
        _ga, _gb,
        _fpr_gap, _fnr_gap, _ppv_gap, _acc_gap, _eo_gap,
        _br_a, _br_b, _t,
    )


# ── CELL 9: ACT I MATH PEEK ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations (Chouldechova 2017 impossibility)": mo.md("""
        **Fairness metric definitions:**

        Let `r` = base rate for a group, `t` = model threshold.
        For a calibrated classifier with score distribution `s`:

        ```
        TPR (sensitivity)  = P(s > t | Y = 1)
        FPR (false alarm)  = P(s > t | Y = 0)
        FNR (miss rate)    = 1 - TPR
        PPV (precision)    = P(Y = 1 | s > t)
                           = r · TPR / (r · TPR + (1-r) · FPR)
        Accuracy           = r · TPR + (1-r) · (1 - FPR)
        ```

        **Chouldechova (2017) Impossibility Theorem:**

        When `r_A ≠ r_B` (base rates differ between groups A and B),
        a single shared model threshold `t` applied to a calibrated classifier
        **cannot simultaneously satisfy all three of:**

        ```
        1. Calibration:   PPV_A = PPV_B
        2. Equal FPR:     FPR_A = FPR_B
        3. Equal FNR:     FNR_A = FNR_B
        ```

        **Proof sketch (from the paper):**

        From the PPV formula:  `FPR = r · TPR / (PPV · (1-r)) - r · TPR / (1-r)`

        If PPV is equal across groups and TPR (= 1 - FNR) is equal across groups,
        then FPR can only be equal if `r_A = r_B`. Since base rates differ by
        assumption, at least one of the three equalities must be violated.

        **Implication for the lab:**

        Equal accuracy is consistent with large FPR or FNR gaps — because accuracy
        weighs positives and negatives by their base-rate frequencies, which differ
        by group. Two groups can have identical accuracy (85%) while one group's
        false alarm rate is many times higher than the other's.
        """),
    })
    return


# ── CELL 10: ACT I PREDICTION-VS-REALITY OVERLAY ──────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, _acc_gap, _fpr_gap, _fnr_gap):
    # The key insight: equal accuracy does NOT imply equal FPR or FNR
    # The correct answer is C: must check FPR AND FNR separately.
    _a1_correct = act1_prediction.value == "option_c"

    _explanations = {
        "option_a": (
            "**Incorrect.** Equal accuracy is not sufficient for fairness. "
            "When base rates differ, the same accuracy score can conceal large "
            "asymmetries in who bears the false positive vs. false negative burden. "
            "Accuracy weights each type of error by base-rate frequency, so groups "
            "with different base rates can score identically in accuracy while "
            "experiencing very different error rates. "
            f"In your current configuration: FPR gap = **{_fpr_gap:.1f}pp**, "
            f"FNR gap = **{_fnr_gap:.1f}pp**, Accuracy gap = **{_acc_gap:.1f}pp**.",
            "warn"
        ),
        "option_b": (
            "**Partially correct, but incomplete.** Checking precision (PPV) is "
            "important, but precision alone does not capture the full picture. "
            "Chouldechova (2017) shows the impossibility involves FPR AND FNR "
            "jointly — checking only PPV leaves out one of the key error asymmetries. "
            "The complete answer requires checking FPR and FNR separately.",
            "warn"
        ),
        "option_c": (
            "**Correct.** This is exactly the lesson Chouldechova (2017) formalizes. "
            "Equal accuracy is necessary but not sufficient. When base rates differ, "
            "the same model threshold produces structurally different FPR and FNR across "
            "groups — regardless of overall accuracy. The compliance officer needs to "
            "audit error rates *by group*, not just accuracy *by group*. "
            f"In the default configuration (base rate A=10%, B=40%): "
            f"FPR gap = {_fpr_gap:.1f}pp, FNR gap = {_fnr_gap:.1f}pp, "
            f"Accuracy gap = {_acc_gap:.1f}pp.",
            "success"
        ),
        "option_d": (
            "**Incorrect.** Accuracy is not the standard that disparate impact doctrine "
            "applies. Courts and regulators examine selection rates, false positive rates, "
            "and the four-fifths rule — none of which are equivalent to overall accuracy. "
            "Equal accuracy while maintaining a 20pp FPR gap would almost certainly "
            "not satisfy a disparate impact claim.",
            "warn"
        ),
    }

    _explanation, _kind = _explanations.get(
        act1_prediction.value, ("No prediction selected.", "info")
    )

    _overlay_md = (
        f"**Prediction-vs-Reality:** You predicted *{act1_prediction.value.upper().replace('_', ' ')}*. "
        f"Current configuration: accuracy gap = {_acc_gap:.1f}pp, FPR gap = {_fpr_gap:.1f}pp, "
        f"FNR gap = {_fnr_gap:.1f}pp. "
        "Notice that accuracy gap can be near zero while FPR and FNR gaps remain substantial."
    )

    mo.vstack([
        mo.callout(mo.md(_explanation), kind=_kind),
        mo.callout(mo.md(_overlay_md), kind="info"),
    ])
    return (_a1_correct,)


# ── CELL 11: ACT I REFLECTION ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Perfect fairness is always achievable with enough data and a better model": "ref_a",
            "B) When base rates differ, you cannot simultaneously achieve calibration, equal FPR, AND equal FNR": "ref_b",
            "C) Accuracy and fairness are always in tension (too general — not the theorem's claim)": "ref_c",
            "D) Fairness metrics are mathematically equivalent and differ only in emphasis": "ref_d",
        },
        label="Reflection: What does Chouldechova's (2017) impossibility theorem state?",
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Act I Reflection"),
        act1_reflection,
    ])
    return (act1_reflection,)


# ── CELL 12: ACT I REFLECTION REVEAL ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your reflection answer to continue to Act II."), kind="warn"),
    )

    _ref_explanations = {
        "ref_a": (
            "**Incorrect.** Chouldechova (2017) proves this is impossible regardless of "
            "data quantity or model quality, as long as base rates differ. More data does "
            "not fix the structural incompatibility — it only makes the model more "
            "precisely wrong in the same direction.",
            "warn"
        ),
        "ref_b": (
            "**Correct.** This is the exact claim of the theorem. For any calibrated "
            "classifier applied with a shared threshold: if `r_A ≠ r_B`, then at most "
            "two of {calibration, equal FPR, equal FNR} can hold simultaneously. "
            "There is no engineering solution — only a choice of which criterion to "
            "prioritize and which to sacrifice.",
            "success"
        ),
        "ref_c": (
            "**Too general.** The statement that accuracy and fairness are always in "
            "tension is a common heuristic, but it is not the theorem. Chouldechova's "
            "contribution is more precise: it identifies the specific three-way "
            "incompatibility and the exact condition (differing base rates) that triggers it.",
            "warn"
        ),
        "ref_d": (
            "**Incorrect.** The entire point of the theorem is that fairness metrics "
            "are NOT equivalent. Demographic parity, equalized odds, and calibration "
            "each capture different aspects of fair treatment, and they actively "
            "conflict when base rates differ. Choosing a metric is a policy decision, "
            "not a technical one.",
            "warn"
        ),
    }

    _expl, _kind = _ref_explanations.get(
        act1_reflection.value, ("No answer selected.", "info")
    )
    mo.callout(mo.md(_expl), kind=_kind)
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "The Audit-Accuracy Tradeoff"
    _act_duration = "20\u201325 min"
    _act_why = (
        "Act I proved that fairness criteria are mutually incompatible when base rates differ. "
        "Now choose: accuracy-only, demographic parity, or equalized odds \u2014 and discover "
        "that the right criterion is a legal and ethical policy decision, not an engineering one, "
        "while the accuracy cost of the most defensible criterion is lower than you expect."
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


# ── CELL 13: ACT2_STAKEHOLDER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    mo.vstack([
        mo.md("---"),
        mo.md("## Act II — The Audit-Accuracy Tradeoff"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: #eff6ff;
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; Engineering VP
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We're deploying a resume screening model at scale — 50,000 applications
                per quarter. We need 80% recall on qualified candidates. We have $50K
                in annual audit budget. Three options are on the table: (1) accuracy-optimized
                model — no fairness constraint, maximize recall; (2) demographic parity
                constraint — equal selection rates across demographic groups; (3) individual
                fairness with equalized odds — similar candidates treated similarly, equal
                FPR and FNR across groups. Design the deployment. Which strategy is most
                defensible, and does it fit the budget?"
            </div>
        </div>
        """),
        mo.md("""
        Each strategy has a different accuracy cost, audit cost, and legal exposure.
        The VP needs a defensible choice — not just the highest accuracy.
        Commit to a strategy prediction before using the design instruments.
        """),
    ])
    return


# ── CELL 14: ACT II PREDICTION LOCK ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) Option 1 — maximize accuracy, handle fairness concerns in post-hoc legal review": "option_a",
            "B) Option 2 — demographic parity is the legally safest strategy for a hiring context": "option_b",
            "C) Option 3 — individual fairness with equalized odds and regular audit is most defensible": "option_c",
            "D) All three strategies are equivalent in practice — the choice is arbitrary": "option_d",
        },
        label=(
            "Which deployment strategy is most defensible for a high-stakes hiring context "
            "under EEOC and disparate impact doctrine?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #6366f1; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock — Act II
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-bottom: 12px;">
                Commit before adjusting the design instruments.
            </div>
        </div>
        """),
        act2_prediction,
    ])
    return (act2_prediction,)


# ── CELL 15: ACT II GATE ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction):
    mo.stop(
        act2_prediction.value is None,
        mo.callout(
            mo.md("Select your strategy prediction above to unlock the Audit Design Cockpit."),
            kind="warn",
        ),
    )
    return


# ── CELL 16: ACT II CONTROLS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # Fairness criterion selection
    # Source: responsible_engr.qmd — the three main criteria for hiring contexts
    fairness_criterion = mo.ui.dropdown(
        options={
            "No constraint (accuracy only)":         "accuracy",
            "Demographic parity (equal selection rates)": "demographic_parity",
            "Equalized odds (equal FPR + FNR)":      "equalized_odds",
            "Individual fairness (similar candidates treated similarly)": "individual",
        },
        value="No constraint (accuracy only)",
        label="Fairness criterion",
    )
    # Bias mitigation method
    # Source: responsible_engr.qmd — three standard mitigation approaches
    bias_mitigation = mo.ui.dropdown(
        options={
            "None":                         "none",
            "Reweighting (pre-processing)": "reweighting",
            "Adversarial debiasing (in-processing)": "adversarial",
            "Post-processing threshold adjustment": "postprocessing",
        },
        value="None",
        label="Bias mitigation method",
    )
    # Audit frequency: how often the deployed model is re-evaluated for bias
    # Source: responsible_engr.qmd — audit cost model
    audit_frequency = mo.ui.dropdown(
        options={
            "Continuous (52×/yr)":  "continuous",
            "Monthly (12×/yr)":     "monthly",
            "Quarterly (4×/yr)":    "quarterly",
            "Annual (1×/yr)":       "annual",
        },
        value="Quarterly (4×/yr)",
        label="Audit frequency",
    )
    mo.vstack([
        mo.md("### Audit Design Cockpit"),
        mo.md(
            "Select a **fairness criterion**, a **bias mitigation method**, and an "
            "**audit frequency**. The instruments will show the resulting accuracy cost, "
            "equalized odds gap, annual audit cost, and regulatory risk level."
        ),
        mo.hstack([fairness_criterion, bias_mitigation, audit_frequency],
                  justify="start", gap="2rem"),
    ])
    return (fairness_criterion, bias_mitigation, audit_frequency)


# ── CELL 17: ACT II PHYSICS ENGINE ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np, math,
    context_toggle,
    fairness_criterion, bias_mitigation, audit_frequency,
    _br_a, _br_b,
    DISPARATE_IMPACT_THRESHOLD_PP,
    CLOUD_AUDIT_COST_K_USD, MOBILE_AUDIT_COST_K_USD,
    apply_plotly_theme,
):
    # ── Context-dependent parameters ──────────────────────────────────────────
    _ctx = context_toggle.value  # "cloud" or "mobile"
    _cost_per_audit_k = CLOUD_AUDIT_COST_K_USD if _ctx == "cloud" else MOBILE_AUDIT_COST_K_USD

    # ── Audit frequency → runs per year ───────────────────────────────────────
    _freq_map = {"continuous": 52, "monthly": 12, "quarterly": 4, "annual": 1}
    _audits_per_year = _freq_map[audit_frequency.value]

    # ── Fairness criterion: base accuracy and baseline EO gap ─────────────────
    # Source: responsible_engr.qmd — Table of fairness-accuracy tradeoffs
    # These are physics-based estimates calibrated to a base rate gap of 30pp
    # (Group A=10%, Group B=40%) from Act I defaults.
    #
    # Accuracy-only model: maximizes recall, no fairness constraint.
    # EO gap is large because threshold optimized for accuracy, not parity.
    #
    # Demographic parity: forces equal selection rates.
    # This requires different thresholds per group → can harm accuracy more.
    # EO gap may actually WORSEN because FPR/FNR are not directly controlled.
    #
    # Equalized odds: forces equal FPR and FNR.
    # Requires per-group threshold calibration → moderate accuracy cost.
    # EO gap is directly minimized.
    #
    # Individual fairness: similar candidates get similar scores.
    # Requires metric learning → highest implementation cost, best defensibility.
    # EO gap depends on implementation quality.
    _base_rate_gap_pp = abs(_br_b - _br_a)  # gap between group base rates

    # Base accuracy and EO gap by criterion
    # EO gap is modeled as a function of base rate gap and constraint choice
    _criterion = fairness_criterion.value
    if _criterion == "accuracy":
        _base_accuracy_pct = 87.5                # unconstrained, high accuracy
        _base_eo_gap       = 0.38 * _base_rate_gap_pp  # large EO gap when base rates differ
    elif _criterion == "demographic_parity":
        _base_accuracy_pct = 83.0                # accuracy cost from forcing equal selection rates
        _base_eo_gap       = 0.50 * _base_rate_gap_pp  # DP can WORSEN EO gap (see theorem)
    elif _criterion == "equalized_odds":
        _base_accuracy_pct = 84.5                # moderate accuracy cost
        _base_eo_gap       = 0.12 * _base_rate_gap_pp  # EO gap directly constrained
    else:  # individual
        _base_accuracy_pct = 84.0                # similar to EO
        _base_eo_gap       = 0.10 * _base_rate_gap_pp  # best EO gap, defensible in court

    # ── Bias mitigation: reduces EO gap further, with additional accuracy cost ─
    # Source: responsible_engr.qmd — mitigation method effectiveness table
    _mitigation = bias_mitigation.value
    _mitigation_gap_reduction = {
        "none":           0.0,
        "reweighting":    0.25,   # reduces gap by 25%, accuracy cost ~0.5pp
        "adversarial":    0.45,   # reduces gap by 45%, accuracy cost ~1.5pp
        "postprocessing": 0.35,   # reduces gap by 35%, accuracy cost ~0.8pp
    }[_mitigation]
    _mitigation_acc_cost = {
        "none":           0.0,
        "reweighting":    0.5,
        "adversarial":    1.5,
        "postprocessing": 0.8,
    }[_mitigation]

    _final_eo_gap = _base_eo_gap * (1.0 - _mitigation_gap_reduction)
    _final_accuracy_pct = _base_accuracy_pct - _mitigation_acc_cost

    # ── Audit cost model ───────────────────────────────────────────────────────
    # Source: responsible_engr.qmd — audit cost = compute + human review
    # Mobile has 2.7× higher cost due to data collection and manual review overhead
    _audit_cost_k = _cost_per_audit_k * _audits_per_year  # total annual audit cost ($K)

    # ── Regulatory risk model ─────────────────────────────────────────────────
    # Source: responsible_engr.qmd — EEOC disparate impact guidelines
    # Risk is a function of EO gap AND audit frequency
    # An unaudited high-gap model is maximum risk; well-audited low-gap model is low risk
    _above_threshold = _final_eo_gap > DISPARATE_IMPACT_THRESHOLD_PP
    _audits_adequate = _audits_per_year >= 4  # quarterly minimum per OFCCP guidance
    _disparate_impact_triggered = _above_threshold  # the failure state

    if not _above_threshold:
        _reg_risk = "low"
    elif _above_threshold and _audits_adequate:
        _reg_risk = "medium"
    else:
        _reg_risk = "high"

    # ── Budget check ──────────────────────────────────────────────────────────
    _budget_k = 50.0  # $50K annual audit budget (from VP scenario)
    _over_budget = _audit_cost_k > _budget_k

    # ── Recall check ─────────────────────────────────────────────────────────
    # VP requires 80% recall on qualified candidates
    _recall_below_80 = _final_accuracy_pct < 80.0

    # ── Color coding ─────────────────────────────────────────────────────────
    def _color_for_gap(g):
        if g <= 5.0:
            return "#22c55e"
        elif g <= 10.0:
            return "#f59e0b"
        return "#ef4444"

    def _color_for_risk(risk):
        return {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444"}[risk]

    _gap_color  = _color_for_gap(_final_eo_gap)
    _risk_color = _color_for_risk(_reg_risk)
    _budget_color = "#22c55e" if not _over_budget else "#ef4444"
    _acc_color = "#22c55e" if _final_accuracy_pct >= 84.0 else "#f59e0b" if _final_accuracy_pct >= 80.0 else "#ef4444"

    # ── Pareto frontier: sweep all criterion × mitigation combinations ────────
    _frontier_configs = [
        ("accuracy + none",       87.5, 0.38 * _base_rate_gap_pp * 1.00),
        ("accuracy + reweight",   87.0, 0.38 * _base_rate_gap_pp * 0.75),
        ("accuracy + adversarial",86.0, 0.38 * _base_rate_gap_pp * 0.55),
        ("accuracy + postproc",   86.7, 0.38 * _base_rate_gap_pp * 0.65),
        ("dem_parity + none",     83.0, 0.50 * _base_rate_gap_pp * 1.00),
        ("dem_parity + reweight", 82.5, 0.50 * _base_rate_gap_pp * 0.75),
        ("dem_parity + adversarial", 81.5, 0.50 * _base_rate_gap_pp * 0.55),
        ("dem_parity + postproc", 82.2, 0.50 * _base_rate_gap_pp * 0.65),
        ("eq_odds + none",        84.5, 0.12 * _base_rate_gap_pp * 1.00),
        ("eq_odds + reweight",    84.0, 0.12 * _base_rate_gap_pp * 0.75),
        ("eq_odds + adversarial", 83.0, 0.12 * _base_rate_gap_pp * 0.55),
        ("eq_odds + postproc",    83.7, 0.12 * _base_rate_gap_pp * 0.65),
        ("individual + none",     84.0, 0.10 * _base_rate_gap_pp * 1.00),
        ("individual + reweight", 83.5, 0.10 * _base_rate_gap_pp * 0.75),
        ("individual + adversarial", 82.5, 0.10 * _base_rate_gap_pp * 0.55),
        ("individual + postproc", 83.2, 0.10 * _base_rate_gap_pp * 0.65),
    ]

    # Group colors for Pareto plot
    _group_colors = {
        "accuracy":    "#94a3b8",  # gray
        "dem_parity":  "#f59e0b",  # amber
        "eq_odds":     "#6366f1",  # indigo
        "individual":  "#22c55e",  # green
    }

    _fig_pareto = go.Figure()
    for _name, _acc, _gap in _frontier_configs:
        _grp = _name.split(" + ")[0].replace("dem_parity", "dem_parity")
        _grp_short = _grp.split("_")[0] if "_" not in _grp[:5] else (
            "dem_parity" if "dem" in _grp else _grp
        )
        _grp_color = _group_colors.get(
            "dem_parity" if "dem" in _name else (
                "eq_odds" if "eq" in _name else (
                    "individual" if "individual" in _name else "accuracy"
                )
            ), "#6366f1"
        )
        _fig_pareto.add_trace(go.Scatter(
            x=[_acc], y=[_gap],
            mode="markers",
            name=_name,
            marker=dict(color=_grp_color, size=10),
            showlegend=False,
        ))

    # Highlight current selection
    _fig_pareto.add_trace(go.Scatter(
        x=[_final_accuracy_pct], y=[_final_eo_gap],
        mode="markers",
        name="Your design",
        marker=dict(color="#ffffff", size=16, symbol="star",
                    line=dict(color="#f59e0b", width=2)),
    ))

    # EEOC threshold line
    _fig_pareto.add_hline(
        y=DISPARATE_IMPACT_THRESHOLD_PP,
        line_dash="dash", line_color="#ef4444",
        annotation_text=f"EEOC guidance threshold ({DISPARATE_IMPACT_THRESHOLD_PP:.0f}pp)",
        annotation_position="top right",
    )
    _fig_pareto = apply_plotly_theme(_fig_pareto)
    _fig_pareto.update_layout(
        title="Fairness-Accuracy Pareto Frontier",
        xaxis_title="Model Accuracy (%)",
        yaxis_title="Equalized Odds Gap (pp) — lower is better",
        height=380,
    )

    # ── Metric cards HTML ────────────────────────────────────────────────────
    _ctx_display = "H100 Cloud" if _ctx == "cloud" else "Mobile NPU"
    _cards = f"""
<div style="display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; margin: 16px 0;">
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 165px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Accuracy</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_acc_color};">
            {_final_accuracy_pct:.1f}%
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">recall target 80%</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 165px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">EO Gap</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_gap_color};">
            {_final_eo_gap:.1f}pp
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">threshold: 10pp</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 165px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Audit Cost / yr</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_budget_color};">
            ${_audit_cost_k:.0f}K
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">budget: $50K ({_ctx_display})</div>
    </div>
    <div style="padding: 18px 22px; border: 1px solid #e2e8f0; border-radius: 10px;
                width: 165px; text-align: center; background: white;">
        <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">Regulatory Risk</div>
        <div style="font-size: 1.9rem; font-weight: 800; color: {_risk_color};">
            {_reg_risk.upper()}
        </div>
        <div style="font-size: 0.72rem; color: #94a3b8;">EEOC / OFCCP</div>
    </div>
</div>
"""

    # ── Physics summary ───────────────────────────────────────────────────────
    _phys_md = f"""
### Design Summary

```
Criterion:   {fairness_criterion.value}
Mitigation:  {bias_mitigation.value}
Audit freq:  {audit_frequency.value} ({_audits_per_year}×/yr)
Context:     {_ctx_display}

Accuracy             = {_final_accuracy_pct:.1f}%
  (base {_base_accuracy_pct:.1f}% − mitigation cost {_mitigation_acc_cost:.1f}pp)

EO Gap               = {_final_eo_gap:.1f}pp
  (base {_base_eo_gap:.1f}pp × (1 − {_mitigation_gap_reduction:.0%} mitigation))

Audit cost / yr      = ${_cost_per_audit_k:.1f}K × {_audits_per_year} = ${_audit_cost_k:.1f}K
Budget remaining     = ${ _budget_k - _audit_cost_k:+.1f}K

Regulatory risk      = {_reg_risk.upper()}
  (EO gap > {DISPARATE_IMPACT_THRESHOLD_PP:.0f}pp: {_above_threshold} | audits >= quarterly: {_audits_adequate})
```
"""

    # ── Output ────────────────────────────────────────────────────────────────
    _output_items = [
        mo.md(_phys_md),
        mo.Html(_cards),
        mo.ui.plotly(_fig_pareto),
    ]

    # ── FAILURE STATE: Disparate Impact triggered ─────────────────────────────
    # Source: EEOC 4/5ths rule and OFCCP audit guidance
    # Triggered when equalized odds gap > 10pp in a high-stakes (hiring) context
    if _disparate_impact_triggered:
        _output_items.append(
            mo.callout(
                mo.md(
                    f"**Disparate impact threshold exceeded.** "
                    f"Equalized odds gap: **{_final_eo_gap:.1f}pp**. "
                    f"High-stakes deployment (hiring) requires gap < {DISPARATE_IMPACT_THRESHOLD_PP:.0f}pp "
                    f"per EEOC guidelines. This deployment would face substantial regulatory risk. "
                    f"Reduce the gap by switching to equalized odds or individual fairness criterion, "
                    f"or apply adversarial debiasing to bring the gap below threshold."
                ),
                kind="danger",
            )
        )
    else:
        _output_items.append(
            mo.callout(
                mo.md(
                    f"EO gap **{_final_eo_gap:.1f}pp** is below the {DISPARATE_IMPACT_THRESHOLD_PP:.0f}pp "
                    f"threshold. Regulatory risk is **{_reg_risk}**. "
                    + ("Audit frequency meets OFCCP quarterly minimum." if _audits_adequate else
                       "Warning: audit frequency below OFCCP quarterly minimum recommendation.")
                ),
                kind="success" if _reg_risk == "low" else "warn",
            )
        )

    # ── Budget warning ────────────────────────────────────────────────────────
    if _over_budget:
        _output_items.append(
            mo.callout(
                mo.md(
                    f"**Budget exceeded.** Annual audit cost **${_audit_cost_k:.0f}K** > "
                    f"${_budget_k:.0f}K budget on **{_ctx_display}**. "
                    "Reduce audit frequency or switch to cloud context for lower per-run cost."
                ),
                kind="warn",
            )
        )

    mo.vstack(_output_items)
    return (
        _final_eo_gap,
        _final_accuracy_pct,
        _audit_cost_k,
        _reg_risk,
        _disparate_impact_triggered,
        _above_threshold,
        _criterion,
    )


# ── CELL 18: ACT II MATH PEEK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations (fairness metrics and audit cost model)": mo.md("""
        **Equalized Odds (Hardt et al. 2016):**

        A classifier satisfies equalized odds if, for all groups `g` and labels `y ∈ {0,1}`:
        ```
        P(ŷ = 1 | Y = y, Group = g) is equal across all groups g
        ```
        Equivalently: both TPR and FPR are equal across groups.

        **Equalized Odds Gap (lab's operationalization):**
        ```
        EO_gap = (|FPR_A − FPR_B| + |FNR_A − FNR_B|) / 2
        ```

        **Demographic Parity:**
        ```
        DP:  P(ŷ = 1 | Group = A) = P(ŷ = 1 | Group = B)
        ```
        Note: DP forces equal selection rates, NOT equal error rates.
        When qualified candidate rates differ, DP can require REJECTING more
        qualified candidates from one group to equalize selection rates —
        which is its own form of unfairness.

        **Individual Fairness (Dwork et al. 2012 — Lipschitz condition):**
        ```
        d_Y(M(x), M(x')) ≤ L · d_X(x, x')
        ```
        Individuals at similar distances in feature space receive similar decisions.
        Requires a task-specific metric `d_X` — the hardest criterion to specify
        but the most defensible under equal treatment theory.

        **Audit Cost Model:**
        ```
        Annual audit cost = cost_per_run × runs_per_year
        Cloud:  cost_per_run ≈ $1,500  (automated, H100 batch evaluation)
        Mobile: cost_per_run ≈ $4,000  (manual data collection + compute)
        ```

        **Regulatory Risk Model (EEOC / OFCCP guidance):**
        ```
        If EO_gap > 10pp AND context is high-stakes: HIGH risk
        If EO_gap > 10pp AND quarterly+ audits:      MEDIUM risk
        If EO_gap ≤ 10pp:                            LOW risk
        ```
        The 10pp threshold derives from EEOC's 4/5ths (80%) rule applied
        to selection rate ratios, converted to equalized odds gap units.
        """),
    })
    return


# ── CELL 19: ACT II PREDICTION REVEAL ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction, _final_eo_gap, _reg_risk, _criterion):
    _a2_correct = act2_prediction.value == "option_c"

    _a2_explanations = {
        "option_a": (
            "**Incorrect.** Option 1 (accuracy-only) does not avoid fairness liability — "
            "it embeds it. Under EEOC disparate impact doctrine, a facially neutral "
            "employment practice that causes adverse impact must be justified by "
            "business necessity AND must use the least discriminatory alternative. "
            "Post-hoc legal review is not a substitute for a defensible deployment design. "
            f"Your current EO gap is **{_final_eo_gap:.1f}pp** — "
            f"regulatory risk is **{_reg_risk}**.",
            "warn"
        ),
        "option_b": (
            "**Incorrect.** Demographic parity is legally problematic in hiring contexts. "
            "It forces equal selection rates regardless of qualified candidate rates. "
            "If 40% of Group B applicants are qualified vs 10% of Group A, demographic "
            "parity requires either rejecting many qualified Group B applicants or "
            "accepting unqualified Group A applicants — both of which can constitute "
            "disparate treatment claims. Equalized odds is more defensible because it "
            "conditions on actual qualification (the true label), not just group membership.",
            "warn"
        ),
        "option_c": (
            "**Correct.** Option 3 — individual fairness or equalized odds with regular "
            "audit — is the most legally defensible approach. It satisfies equal treatment "
            "theory (similar candidates get similar decisions) and can demonstrate "
            "compliance through the audit record. The Pareto frontier shows equalized odds "
            "achieves lower EO gap than demographic parity at similar accuracy cost, "
            "and individual fairness further improves defensibility. "
            f"Your current EO gap is **{_final_eo_gap:.1f}pp** — "
            f"regulatory risk is **{_reg_risk}**.",
            "success"
        ),
        "option_d": (
            "**Incorrect.** The three strategies produce materially different outcomes "
            "on accuracy, EO gap, and regulatory risk. The Pareto frontier makes this "
            "visible: demographic parity can produce WORSE EO gaps than equalized odds "
            "at the same accuracy cost, because it constrains the wrong quantity "
            "(selection rate, not error rate). The strategies are not equivalent.",
            "warn"
        ),
    }

    _expl2, _kind2 = _a2_explanations.get(
        act2_prediction.value, ("No prediction selected.", "info")
    )
    mo.callout(mo.md(_expl2), kind=_kind2)
    return (_a2_correct,)


# ── CELL 20: ACT II REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Demographic parity is never a valid fairness metric": "ref2_a",
            "B) It forces equal selection rates regardless of base rates — if qualified candidates "
               "differ by group, it means rejecting more qualified people in one group to hit the quota": "ref2_b",
            "C) It always hurts overall accuracy by more than 10 percentage points": "ref2_c",
            "D) Demographic parity violates GDPR by requiring protected attribute access": "ref2_d",
        },
        label=(
            "Reflection: Why can demographic parity actually produce unfair outcomes "
            "in a hiring context where qualified candidate rates differ by group?"
        ),
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Act II Reflection"),
        act2_reflection,
    ])
    return (act2_reflection,)


# ── CELL 21: ACT II REFLECTION REVEAL ────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select your reflection answer to complete the lab."), kind="warn"),
    )

    _ref2_explanations = {
        "ref2_a": (
            "**Incorrect — too absolute.** Demographic parity is valid in some contexts, "
            "particularly when base rates are approximately equal or when the historical "
            "qualification gap is itself a product of discriminatory conditions. "
            "The problem is applying it uncritically when base rates differ for legitimate reasons.",
            "warn"
        ),
        "ref2_b": (
            "**Correct.** This is the key insight. Demographic parity equalizes the "
            "outcome (selection rate) without conditioning on actual qualification. "
            "If Group B has 40% qualified candidates and Group A has 10%, equal "
            "selection rates require either: (a) lowering the bar for Group A "
            "(accepting unqualified applicants) or (b) raising the bar for Group B "
            "(rejecting qualified applicants). Both produce decisions that are "
            "conditionally unfair given true qualification. Equalized odds avoids "
            "this by conditioning on the true label — it guarantees that equally "
            "qualified candidates are equally likely to be selected.",
            "success"
        ),
        "ref2_c": (
            "**Incorrect.** The accuracy cost of demographic parity depends on how "
            "different the base rates are and how tight the parity constraint is. "
            "The Pareto frontier in Act II shows it can be 1–5pp in typical configurations. "
            "It is not always >10pp.",
            "warn"
        ),
        "ref2_d": (
            "**Incorrect.** Demographic parity does not violate GDPR — GDPR does not "
            "prohibit all use of protected attributes; Article 9 allows processing for "
            "specific purposes including combating discrimination. Many GDPR-compliant "
            "fairness audits explicitly compute demographic parity metrics. The legal "
            "challenge to DP is its substantive unfairness, not its data requirements.",
            "warn"
        ),
    }

    _expl_r2, _kind_r2 = _ref2_explanations.get(
        act2_reflection.value, ("No answer selected.", "info")
    )
    mo.callout(mo.md(_expl_r2), kind=_kind_r2)
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
                    <strong>1. Equal accuracy is not equal treatment (Chouldechova 2017).</strong>
                    When base rates differ between groups, a calibrated model with equal accuracy
                    structurally produces unequal FPR and FNR. With a 30% vs. 50% base rate gap,
                    a model at 85% accuracy on both groups can still have false positive rates
                    that differ by more than 10 percentage points. This is mathematics, not bias.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Choosing a fairness criterion is a policy decision, not an engineering one.</strong>
                    Demographic parity, equalized odds, and calibration are mutually incompatible
                    when base rates differ. Equalized odds is most defensible in hiring and lending
                    under US law because it conditions on actual qualification. The engineering job
                    is to implement the chosen criterion &mdash; not to choose it.
                </div>
                <div>
                    <strong>3. The equalized odds accuracy cost is lower than expected.</strong>
                    In the resume screening scenario, enforcing equalized odds (equal FPR and FNR)
                    costs approximately 3&ndash;5 percentage points of accuracy relative to the
                    unconstrained baseline, while reducing the false positive rate disparity from
                    &gt;10 pp to &lt;1 pp. The fairness gain is asymmetric to the accuracy cost.
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
                    <strong>Lab 16: Synthesizing ML Systems.</strong> This lab showed that
                    fairness has a measurable accuracy cost. Lab 16 asks: when you combine
                    all constraints from Chapters 12&ndash;15 simultaneously &mdash; Amdahl&apos;s
                    ceiling, P99 SLOs, drift rate, and fairness thresholds &mdash; can a single
                    system satisfy all of them, and how do you diagnose which constraint is binding?
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
                    <strong>Read:</strong> @sec-responsible-engineering-impossibility for the
                    Chouldechova proof and @sec-responsible-engineering-audit-pipelines for
                    the audit cost model at deployment scale.<br/>
                    <strong>Build:</strong> TinyTorch Module 15 &mdash; implement a fairness
                    auditor that computes FPR/FNR parity and flags disparate impact violations.
                    See <code>tinytorch/src/15_responsible/</code>.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. A facial recognition system reports 96% aggregate accuracy. The Gender Shades study shows the worst-performing subgroup can reach 34.7% error rate. What is the maximum disparity ratio — and why does aggregate accuracy fail to bound subgroup performance?

    2. The Fairness-Accuracy Pareto Frontier shows a 'sweet spot' at Point B where fairness disparity drops from 18% to 5% at a cost of only ~3 percentage points of accuracy (from 94.7% to 91.3%). What does this reveal about the assumption that fairness and accuracy are strictly opposed?

    3. The Fairness Impossibility Theorem (Chouldechova-Kleinberg) proves that Calibration, Equal FPR, and Equal FNR cannot all be simultaneously satisfied when base rates differ between groups. Given this, what is the correct engineering response when a product manager asks for a 'fair' model — and what role does a policy decision play?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD ───────────────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo,
    ledger,
    COLORS,
    context_toggle,
    act1_prediction,
    act2_prediction,
    _a1_correct,
    _a2_correct,
    _final_eo_gap,
    _final_accuracy_pct,
    _audit_cost_k,
    _reg_risk,
    _disparate_impact_triggered,
    _criterion,
, decision_input, decision_ui):
    # ── Safe defaults for unresolved reactive values ─────────────────────────
    _ctx_val    = context_toggle.value if context_toggle.value else "cloud"
    _a1_pred    = act1_prediction.value if act1_prediction.value else "none"
    _a2_pred    = act2_prediction.value if act2_prediction.value else "none"
    _a1_ok      = bool(_a1_correct) if _a1_correct is not None else False
    _a2_ok      = bool(_a2_correct) if _a2_correct is not None else False
    _gap_val    = float(_final_eo_gap) if _final_eo_gap is not None else 0.0
    _acc_val    = float(_final_accuracy_pct) if _final_accuracy_pct is not None else 84.0
    _cost_val   = float(_audit_cost_k) if _audit_cost_k is not None else 0.0
    _risk_val   = str(_reg_risk) if _reg_risk is not None else "low"
    _hit_val    = bool(_disparate_impact_triggered) if _disparate_impact_triggered is not None else False
    _crit_val   = str(_criterion) if _criterion is not None else "accuracy"

    ledger.save(
        chapter=15,
        design={
            "context":            _ctx_val,
            "fairness_criterion": _crit_val,
            "equalized_odds_gap": round(_gap_val, 2),
            "audit_cost_k":       round(_cost_val, 2),
            "act1_prediction":    _a1_pred,
            "act1_correct":       _a1_ok,
            "act2_result":        round(_gap_val, 2),
            "act2_decision":      _crit_val,
            "constraint_hit":     _hit_val,
        "student_justification": str(decision_input.value),
            "regulatory_risk":    _risk_val,
        },
    )

    # ── HUD footer ─────────────────────────────────────────────────────────
    _hud_color   = COLORS["Cloud"] if _ctx_val == "cloud" else COLORS["Mobile"]
    _ctx_display = "Cloud H100" if _ctx_val == "cloud" else "Mobile NPU"
    _a1_display  = "Correct" if _a1_ok else ("Incorrect" if _a1_pred != "none" else "—")
    _a2_display  = "Correct" if _a2_ok else ("Incorrect" if _a2_pred != "none" else "—")
    _hit_color   = "#f87171" if _hit_val else "#4ade80"
    _risk_color  = {"low": "#4ade80", "medium": "#fbbf24", "high": "#f87171"}.get(_risk_val, "#94a3b8")
    _budget_ok   = _cost_val <= 50.0
    _cost_color  = "#4ade80" if _budget_ok else "#f87171"

    mo.Html(f"""
    <div style="display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
                padding: 14px 24px; background: #0f172a;
                border-radius: 12px; margin-top: 32px; font-size: 0.8rem;
                border: 1px solid #1e293b; font-family: 'SF Mono', monospace;">
        <span style="color: #475569; font-weight: 700; letter-spacing: 0.06em;">LAB 15</span>
        <span>
            <span style="color: #475569;">CONTEXT </span>
            <span style="color: {_hud_color}; font-weight: 700;">{_ctx_display}</span>
        </span>
        <span>
            <span style="color: #475569;">CRITERION </span>
            <span style="color: #a5b4fc; font-weight: 600;">{_crit_val}</span>
        </span>
        <span>
            <span style="color: #475569;">ACT I </span>
            <span style="color: {'#4ade80' if _a1_ok else '#f87171'};">{_a1_display}</span>
        </span>
        <span>
            <span style="color: #475569;">ACT II </span>
            <span style="color: {'#4ade80' if _a2_ok else '#f87171'};">{_a2_display}</span>
        </span>
        <span>
            <span style="color: #475569;">EO GAP </span>
            <span style="color: {_hit_color};">{_gap_val:.1f}pp</span>
        </span>
        <span>
            <span style="color: #475569;">AUDIT COST </span>
            <span style="color: {_cost_color};">${_cost_val:.0f}K/yr</span>
        </span>
        <span>
            <span style="color: #475569;">REG RISK </span>
            <span style="color: {_risk_color}; font-weight: 700;">{_risk_val.upper()}</span>
        </span>
        <span>
            <span style="color: #475569;">DISPARATE IMPACT </span>
            <span style="color: {_hit_color};">{'YES' if _hit_val else 'No'}</span>
        </span>
        <span>
            <span style="color: #475569;">LEDGER </span>
            <span style="color: #4ade80;">ch15 saved</span>
        </span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
