import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-16: THE FAIRNESS IMPOSSIBILITY
#
# Volume II, Chapter 16 — Responsible Engineering
#
# Core Invariant: Fairness metrics incompatibility at scale
#   Demographic parity, equalized odds, and calibration are mathematically
#   incompatible when base rates differ between groups. You must choose which
#   to optimize. Each choice has different societal implications.
#   Source: Chouldechova (2017) impossibility theorem.
#
# 2 Contexts:
#   Accuracy-Optimized  — standard production deployment (maximize overall accuracy)
#   Equity-Constrained  — fairness-aware deployment (explicit fairness constraints)
#
# Act I  (12–15 min): Fairness Incompatibility Visualizer
#   Stakeholder: Trust & Safety Lead — content moderation at 1B user scale
#   Instruments: base rates, threshold, fairness criterion selector
#   Prediction: what happens to FPR when demographic parity is enforced?
#   Overlay: predicted outcome vs. actual from incompatibility physics
#   Reflection: when are all three criteria simultaneously achievable?
#
# Act II (20–25 min): Responsible Deployment Designer
#   Stakeholder: Chief AI Ethics Officer — global hiring platform, 180 countries
#   Instruments: global standard, per-jurisdiction thresholds, audit frequency
#   Prediction: best strategy for multi-jurisdiction compliance
#   Failure state: EU or US non-compliance triggers danger callout
#   Failure state: >1M daily false positives triggers warn callout
#   Reflection: why does scale change the ethical stakes?
#
# Hardware Constants:
#   H100_BW_GBS    = 3350    # H100 SXM5 HBM3e, NVIDIA spec
#   USERS_SCALE    = 1_000_000_000  # 1B users, @sec-responsible-engineering
#   AUDIT_COST_PER_K = 5.0   # $5K per 1000 audit samples, industry estimate
#
# Design Ledger: saves chapter="v2_16"
# ─────────────────────────────────────────────────────────────────────────────


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

    # ── Hardware and scale constants ──────────────────────────────────────────
    H100_BW_GBS       = 3350            # GB/s HBM3e — NVIDIA H100 SXM5 spec
    USERS_SCALE       = 1_000_000_000   # 1B users — @sec-responsible-engineering
    AUDIT_COST_PER_K  = 5.0             # $5K per 1,000 audit samples — industry estimate

    # ── Fairness physics constants ────────────────────────────────────────────
    # From Chouldechova (2017): when base rates differ, demographic parity
    # forces unequal FPR/FNR, and equalized odds forces unequal calibration.
    # These are the three incompatible criteria:
    #   DP  = Demographic Parity:  P(Yhat=1|G=A) = P(Yhat=1|G=B)
    #   EO  = Equalized Odds:      TPR_A=TPR_B and FPR_A=FPR_B
    #   CAL = Calibration:         P(Y=1|Yhat=1,G=A) = P(Y=1|Yhat=1,G=B)

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, USERS_SCALE, AUDIT_COST_PER_K,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER (hide_code=True) ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _gov_color = COLORS["BlueLine"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2a1e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 16
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Fairness Impossibility
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 660px; line-height: 1.65;">
                You deployed content moderation to 1 billion users. You targeted
                demographic parity — equal flag rates across demographics. Civil
                rights organizations report your false positive rate is 5&times;
                higher for some groups. You thought you achieved fairness.
                The math says otherwise.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    Act I: Fairness Incompatibility · Act II: Deployment Design
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: @sec-responsible-ai-fairness-machine-learning-2ba4
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Context A: Accuracy-Optimized</span>
                <span class="badge badge-info">Context B: Equity-Constrained</span>
                <span class="badge badge-warn">Invariant: DP + EO + CAL cannot coexist</span>
                <span class="badge badge-warn">Invariant: Scale amplifies tiny disparities</span>
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

        <!-- LEARNING OBJECTIVES -->
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Identify which fairness criterion is violated</strong> when demographic parity is satisfied but false positive rates remain unequal &mdash; and prove this follows from the Chouldechova impossibility theorem when base rates differ.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the fairness tax</strong>: measure the exact accuracy drop when enforcing demographic parity vs. equalized odds in a content moderation classifier with heterogeneous base rates across 180 countries.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a multi-jurisdiction fairness compliance strategy</strong> that satisfies EU calibration requirements and US equalized-odds standards simultaneously without exceeding 1M daily false positives.</div>
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
                    Fairness definitions (demographic parity, equalized odds, calibration)
                    from @sec-responsible-ai-fairness-machine-learning-2ba4 &middot;
                    Chouldechova impossibility theorem from @sec-responsible-ai-demographic-parity-f126
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35&ndash;40 min</strong><br/>
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
                "If you can set any classification threshold and choose any fairness criterion, why is it mathematically impossible to simultaneously satisfy demographic parity, equalized odds, and calibration &mdash; and what does choosing between them actually mean at 1 billion users?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING (hide_code=True) ────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-responsible-ai-fairness-machine-learning-2ba4** — Fairness definitions:
      demographic parity, equalized odds, equality of opportunity, calibration.
      Pay particular attention to the incompatibility callout.
    - **@sec-responsible-ai-demographic-parity-f126** — Demographic parity definition
      and the "equal outcomes regardless of base rates" constraint.
    - **@sec-responsible-ai-core-principles-1bd7** — Responsible AI as a stability
      constraint: fairness, transparency, accountability, privacy, safety.

    If you have not read these sections, the prediction questions will not map
    to the mathematical physics of fairness incompatibility.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE (hide_code=True) ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Accuracy-Optimized (standard production)": "accuracy",
            "Equity-Constrained (fairness-aware)": "equity",
        },
        value="Accuracy-Optimized (standard production)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Select your deployment context to orient the instruments:"),
        context_toggle,
    ])
    return (context_toggle,)


@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS):
    _ctx = context_toggle.value
    _is_equity = _ctx == "equity"
    _color = COLORS["GreenLine"] if _is_equity else COLORS["BlueLine"]
    _label = "Equity-Constrained" if _is_equity else "Accuracy-Optimized"
    _desc = (
        "Explicit fairness constraints active — model threshold adjusted per group "
        "to satisfy the selected fairness criterion. Overall accuracy is lower."
        if _is_equity else
        "Standard production deployment — single threshold maximizes overall accuracy. "
        "No explicit fairness constraints. Group disparities emerge from base rate differences."
    )
    mo.Html(f"""
    <div style="border-left: 4px solid {_color};
                background: {'#ecfdf5' if _is_equity else '#eff6ff'};
                border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 10px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
            Active Context
        </div>
        <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">{_label}</div>
        <div style="font-size: 0.85rem; color: #475569; margin-top: 3px;">{_desc}</div>
    </div>
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["Cloud"]
    _act_title    = "The Fairness Incompatibility"
    _act_duration = "12&ndash;15 min"
    _act_why      = ("You believe there exists a threshold that makes the model fair on all dimensions simultaneously. "
                     "The instruments will show that at every threshold position, "
                     "at least one fairness metric is substantially violated when group base rates differ &mdash; "
                     "not because the model is poorly trained, but because the Chouldechova-Kleinberg theorem "
                     "proves these criteria are logically incompatible.")
    mo.vstack([
        mo.md("---"),
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
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['BlueL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Trust &amp; Safety Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We deployed content moderation at 1B user scale. We explicitly targeted
            demographic parity — equal flag rates across all demographic groups. Our
            engineering team confirmed we achieved it: Group A is flagged at 6% and
            Group B is flagged at 6%. But civil rights organizations are reporting that
            our model flags innocent users at 5&times; higher rates in some groups. We
            thought demographic parity meant fairness. What happened — and whose definition
            of fairness were we actually satisfying?"
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Base Rate Problem

    The Trust & Safety Lead's team achieved demographic parity — equal flag rates.
    But demographic parity only constrains **selection rates**, not the composition
    of those selections. When groups have different violation rates (base rates), equal
    selection forces unequal false positive rates.

    Consider two groups with the same 6% flag rate but different actual violation rates:

    ```
    Group A: base rate = 2% actual violations
      Flags 6% → 2% true positives + 4% false positives
      FPR = 4 false positives / 98 non-violating users = 4.1%

    Group B: base rate = 10% actual violations
      Flags 6% → 6% true positives + 0% false positives
      FPR = 0 false positives / 90 non-violating users = 0%

    Demographic parity achieved. Equal FPR? Far from it.
    ```

    This is the **Chouldechova (2017) impossibility theorem**: when base rates differ
    between groups, demographic parity, equalized odds, and calibration cannot
    all be satisfied simultaneously.
    """)
    return


# ─── ACT I PREDICTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before exploring the simulator, commit to your hypothesis:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) Demographic parity achieved both equal selection and equal FPR — they are equivalent definitions":
                "option_a",
            "B) Demographic parity at different base rates forces unequal false positive rates — it optimizes one definition of fairness, not all":
                "option_b",
            "C) The model has a bug — demographic parity should prevent disparate FPR by construction":
                "option_c",
            "D) Equal FPR and demographic parity are mathematically equivalent whenever the model is well-calibrated":
                "option_d",
        },
        label="The Trust & Safety Lead achieved demographic parity (6% flag rate for both groups). Civil rights organizations report Group A's false positive rate is ~4× higher than Group B's. What explains this?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the Fairness Incompatibility Visualizer."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** {act1_pred.value}. Now explore the incompatibility physics below."),
        kind="info",
    )
    return


# ─── ACT I INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Fairness Incompatibility Visualizer")
    return


@app.cell(hide_code=True)
def _(mo):
    base_rate_a = mo.ui.slider(
        start=1, stop=20, value=2, step=1,
        label="Base rate Group A (% actual violations)",
        show_value=True,
    )
    base_rate_b = mo.ui.slider(
        start=1, stop=20, value=10, step=1,
        label="Base rate Group B (% actual violations)",
        show_value=True,
    )
    model_threshold = mo.ui.slider(
        start=10, stop=90, value=50, step=5,
        label="Model threshold (% — classifier score cutoff)",
        show_value=True,
    )
    fairness_criterion = mo.ui.radio(
        options={
            "Demographic Parity (equal selection rates)": "dp",
            "Equalized Odds (equal TPR and FPR)": "eo",
            "Calibration (equal precision / PPV)": "cal",
        },
        value="Demographic Parity (equal selection rates)",
        label="Fairness criterion to enforce:",
        inline=False,
    )
    mo.vstack([
        mo.md("""
        Adjust base rates and threshold to observe how the three fairness criteria
        interact. The key insight: when base rates differ, satisfying one criterion
        mathematically prevents satisfying the others.
        """),
        mo.hstack([base_rate_a, base_rate_b, model_threshold], justify="start", gap="2rem"),
        fairness_criterion,
    ])
    return (base_rate_a, base_rate_b, model_threshold, fairness_criterion)


@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme, COLORS,
    base_rate_a, base_rate_b, model_threshold, fairness_criterion,
    context_toggle,
):
    # ── Physics engine — Chouldechova (2017) fairness incompatibility ──────────
    #
    # Model: logistic-like classifier. At threshold t, probability of flagging
    # a user with true violation prob p:
    #   P(flag | violation) = sigmoid(score_sensitivity * (p - t/100))
    # We simplify to a linear approximation valid for the pedagogical purpose.
    #
    # For Group G with base rate br_G (fraction), and model threshold t:
    #   Let p_flag_G = fraction flagged (depends on threshold + base rate)
    #   TP_G = min(br_G, p_flag_G)       — true positives (simplified)
    #   FP_G = max(0, p_flag_G - br_G)   — false positives
    #   FN_G = max(0, br_G - p_flag_G)   — false negatives
    #   TN_G = 1 - br_G - FP_G           — true negatives
    #
    # Under demographic parity enforcement:
    #   p_flag_A = p_flag_B = target_rate (set to match average)
    # Under equalized odds enforcement:
    #   TPR_A = TPR_B → different thresholds per group
    # Under calibration enforcement:
    #   PPV_A = PPV_B → different thresholds per group
    #
    # Source: Chouldechova, A. (2017). Fair prediction with disparate impact:
    # A study of bias in recidivism prediction instruments.

    br_a = base_rate_a.value / 100.0   # fraction
    br_b = base_rate_b.value / 100.0   # fraction
    t    = model_threshold.value / 100.0   # threshold fraction
    crit = fairness_criterion.value
    ctx  = context_toggle.value

    # ── Compute raw model outputs (accuracy-optimized, single threshold) ──────
    # Linear flag model: p_flag = base_rate * (1/t) capped at 1, simplified
    # A user is flagged if their estimated violation probability > threshold.
    # If base_rate is the mean violation probability, and threshold = t:
    #   Selection rate ≈ base_rate / t  (fraction with prob > t, simplified uniform)
    # We cap at 1.0.
    def compute_metrics(br, threshold):
        """Compute confusion matrix metrics for one group."""
        # Selection rate under single-threshold model
        p_flag = min(1.0, br / threshold) if threshold > 0 else 1.0
        # True positives: correctly flagged violators
        tp = min(br, p_flag)
        # False positives: incorrectly flagged non-violators
        fp = max(0.0, p_flag - tp)
        # False negatives: missed violators
        fn = max(0.0, br - tp)
        # True negatives
        tn = max(0.0, (1.0 - br) - fp)
        # Rates
        tpr = tp / br if br > 0 else 0.0
        fpr = fp / (1.0 - br) if (1.0 - br) > 0 else 0.0
        ppv = tp / p_flag if p_flag > 0 else 0.0  # precision / calibration
        return dict(p_flag=p_flag, tp=tp, fp=fp, fn=fn, tn=tn,
                    tpr=tpr, fpr=fpr, ppv=ppv)

    # ── Accuracy-optimized baseline (single threshold) ────────────────────────
    raw_a = compute_metrics(br_a, t)
    raw_b = compute_metrics(br_b, t)

    # ── Fairness-constrained adjustment ───────────────────────────────────────
    if crit == "dp":
        # Demographic parity: equalize p_flag
        target_rate = (raw_a["p_flag"] + raw_b["p_flag"]) / 2.0
        # Solve for threshold per group: threshold_g = br_g / target_rate
        t_a_dp = br_a / target_rate if target_rate > 0 else t
        t_b_dp = br_b / target_rate if target_rate > 0 else t
        adj_a = compute_metrics(br_a, t_a_dp)
        adj_b = compute_metrics(br_b, t_b_dp)
        # Force exact parity (numerical precision)
        adj_a["p_flag"] = target_rate
        adj_b["p_flag"] = target_rate

    elif crit == "eo":
        # Equalized odds: equalize TPR (we also try to equalize FPR)
        # Average the TPRs and find thresholds that achieve equal TPR
        target_tpr = (raw_a["tpr"] + raw_b["tpr"]) / 2.0
        # Under our simplified model: TPR = min(1, br/t) / br * min(br, min(1,br/t))
        # Simplified: TPR_g ≈ min(1, 1/t) (independent of br under uniform model)
        # To equalize precisely, adjust threshold so each group hits the same TPR
        # t_g = br_g / (target_tpr * br_g) = 1/target_tpr (same for both)
        t_eq = 1.0 / target_tpr if target_tpr > 0 else t
        adj_a = compute_metrics(br_a, t_eq)
        adj_b = compute_metrics(br_b, t_eq)

    else:  # calibration / PPV parity
        # Calibration: equalize PPV (precision)
        # PPV_g = br_g / p_flag_g → equal PPV requires p_flag_g ∝ br_g
        # Simplify: set both groups to selection rate = br / target_ppv
        target_ppv = (raw_a["ppv"] + raw_b["ppv"]) / 2.0 if (raw_a["ppv"] + raw_b["ppv"]) > 0 else 0.5
        t_a_cal = br_a / (target_ppv * min(1.0, br_a / t)) if (target_ppv > 0 and t > 0) else t
        t_b_cal = br_b / (target_ppv * min(1.0, br_b / t)) if (target_ppv > 0 and t > 0) else t
        adj_a = compute_metrics(br_a, max(0.01, t_a_cal))
        adj_b = compute_metrics(br_b, max(0.01, t_b_cal))

    # ── Context: accuracy-optimized uses raw, equity uses adjusted ────────────
    use_a = raw_a if ctx == "accuracy" else adj_a
    use_b = raw_b if ctx == "accuracy" else adj_b

    # ── Check which fairness criteria are satisfied (within tolerance) ────────
    TOL = 0.03  # 3 percentage-point tolerance
    dp_satisfied  = abs(use_a["p_flag"] - use_b["p_flag"]) < TOL
    eo_satisfied  = (abs(use_a["tpr"] - use_b["tpr"]) < TOL and
                     abs(use_a["fpr"] - use_b["fpr"]) < TOL)
    cal_satisfied = abs(use_a["ppv"] - use_b["ppv"]) < TOL

    def check_icon(satisfied):
        return ("✓", COLORS["GreenLine"]) if satisfied else ("✗", COLORS["RedLine"])

    dp_icon,  dp_col  = check_icon(dp_satisfied)
    eo_icon,  eo_col  = check_icon(eo_satisfied)
    cal_icon, cal_col = check_icon(cal_satisfied)

    # ── Metrics color coding ──────────────────────────────────────────────────
    def rate_color(v):
        return COLORS["GreenLine"] if v < 0.05 else COLORS["OrangeLine"] if v < 0.15 else COLORS["RedLine"]

    fpr_a_col = rate_color(use_a["fpr"])
    fpr_b_col = rate_color(use_b["fpr"])

    # ── Formulas ──────────────────────────────────────────────────────────────
    _formula = f"""
**Fairness incompatibility physics (Chouldechova, 2017):**

```
Group A base rate = {base_rate_a.value}%    Group B base rate = {base_rate_b.value}%
Threshold = {model_threshold.value}%         Criterion = {crit.upper()}

Group A:
  Selection rate (p_flag) = {use_a['p_flag']:.1%}
  TPR = {use_a['tpr']:.1%}   FPR = {use_a['fpr']:.1%}   PPV = {use_a['ppv']:.1%}

Group B:
  Selection rate (p_flag) = {use_b['p_flag']:.1%}
  TPR = {use_b['tpr']:.1%}   FPR = {use_b['fpr']:.1%}   PPV = {use_b['ppv']:.1%}

Fairness criteria satisfied:
  Demographic Parity  (Δp_flag  < 3%): {dp_icon}  |Δ| = {abs(use_a['p_flag'] - use_b['p_flag']):.1%}
  Equalized Odds      (ΔTPR,ΔFPR < 3%): {eo_icon}  |ΔTPR| = {abs(use_a['tpr'] - use_b['tpr']):.1%}, |ΔFPR| = {abs(use_a['fpr'] - use_b['fpr']):.1%}
  Calibration         (ΔPPV  < 3%): {cal_icon}  |Δ| = {abs(use_a['ppv'] - use_b['ppv']):.1%}
```
"""

    # ── Bar chart: FPR, TPR, PPV comparison ──────────────────────────────────
    _metrics = ["Selection Rate", "TPR", "FPR", "PPV (Calibration)"]
    _vals_a  = [use_a["p_flag"], use_a["tpr"], use_a["fpr"], use_a["ppv"]]
    _vals_b  = [use_b["p_flag"], use_b["tpr"], use_b["fpr"], use_b["ppv"]]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(
        name="Group A",
        x=_metrics, y=[v * 100 for v in _vals_a],
        marker_color=COLORS["BlueLine"],
        text=[f"{v:.1%}" for v in _vals_a],
        textposition="outside",
    ))
    _fig.add_trace(go.Bar(
        name="Group B",
        x=_metrics, y=[v * 100 for v in _vals_b],
        marker_color=COLORS["OrangeLine"],
        text=[f"{v:.1%}" for v in _vals_b],
        textposition="outside",
    ))
    _fig.update_layout(
        barmode="group",
        yaxis=dict(title="Rate (%)", range=[0, min(110, max(100, max(_vals_a + _vals_b) * 110))]),
        height=320,
        margin=dict(l=50, r=20, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=f"Fairness metrics by group — {ctx.upper()} context, criterion: {crit.upper()}",
            font=dict(size=13),
        ),
    )
    apply_plotly_theme(_fig)

    # ── Metric cards ──────────────────────────────────────────────────────────
    mo.vstack([
        mo.md(_formula),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
            <!-- Demographic Parity check -->
            <div style="padding: 18px; border: 2px solid {dp_col}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    Demographic Parity
                </div>
                <div style="font-size: 2.4rem; font-weight: 900; color: {dp_col}; margin: 6px 0;">
                    {dp_icon}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    |&Delta;flag| = {abs(use_a['p_flag'] - use_b['p_flag']):.1%}
                </div>
            </div>
            <!-- Equalized Odds check -->
            <div style="padding: 18px; border: 2px solid {eo_col}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    Equalized Odds
                </div>
                <div style="font-size: 2.4rem; font-weight: 900; color: {eo_col}; margin: 6px 0;">
                    {eo_icon}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    |&Delta;FPR| = {abs(use_a['fpr'] - use_b['fpr']):.1%}
                </div>
            </div>
            <!-- Calibration check -->
            <div style="padding: 18px; border: 2px solid {cal_col}; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    Calibration (PPV)
                </div>
                <div style="font-size: 2.4rem; font-weight: 900; color: {cal_col}; margin: 6px 0;">
                    {cal_icon}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">
                    |&Delta;PPV| = {abs(use_a['ppv'] - use_b['ppv']):.1%}
                </div>
            </div>
            <!-- FPR Group A -->
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    FPR Group A
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {fpr_a_col}; margin: 6px 0;">
                    {use_a['fpr']:.1%}
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">false positive rate</div>
            </div>
            <!-- FPR Group B -->
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    FPR Group B
                </div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {fpr_b_col}; margin: 6px 0;">
                    {use_b['fpr']:.1%}
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">false positive rate</div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (
        use_a, use_b, raw_a, raw_b, adj_a, adj_b,
        dp_satisfied, eo_satisfied, cal_satisfied,
        br_a, br_b, t, crit, ctx,
    )


# ─── ACT I FEEDBACK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, dp_satisfied, eo_satisfied, cal_satisfied, br_a, br_b, crit):
    _all_three = dp_satisfied and eo_satisfied and cal_satisfied
    _base_rates_equal = abs(br_a - br_b) < 0.02

    if _all_three:
        mo.callout(mo.md(
            f"**All three criteria satisfied simultaneously.** "
            f"This is only possible because the base rates are nearly equal "
            f"({br_a:.0%} vs {br_b:.0%}). When base rates converge, the Chouldechova "
            f"impossibility theorem no longer bites — there is no mathematical "
            f"contradiction to resolve. This is the narrow exception to the rule."
        ), kind="success")
    elif _base_rates_equal:
        mo.callout(mo.md(
            f"**Equal base rates but criteria still not all satisfied.** "
            f"The model threshold ({crit.upper()} enforcement) is creating disparities "
            f"even with equal base rates. Check the threshold value — very high or very "
            f"low thresholds can reintroduce disparities even when base rates match."
        ), kind="warn")
    else:
        _n_satisfied = sum([dp_satisfied, eo_satisfied, cal_satisfied])
        mo.callout(mo.md(
            f"**{_n_satisfied}/3 criteria satisfied.** "
            f"With Group A base rate = {br_a:.0%} and Group B base rate = {br_b:.0%}, "
            f"the impossibility theorem is active: satisfying {crit.upper()} "
            f"mathematically forces violation of the other criteria. "
            f"This is not a model bug — it is a mathematical consequence of "
            f"different base rates. The only way to satisfy all three simultaneously "
            f"is to equalize the base rates (a social intervention, not a technical fix)."
        ), kind="warn")
    return


# ─── ACT I PREDICTION-VS-REALITY OVERLAY ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, use_a, use_b):
    _correct_option = "option_b"
    _is_correct = act1_pred.value == _correct_option

    _fpr_ratio = use_a["fpr"] / use_b["fpr"] if use_b["fpr"] > 0.001 else float("inf")

    _feedback = {
        "option_a": (
            f"**You predicted demographic parity implies equal FPR — this is the core misconception.** "
            f"The simulator shows FPR Group A = {use_a['fpr']:.1%} vs FPR Group B = {use_b['fpr']:.1%} "
            f"(ratio: {_fpr_ratio:.1f}×) even after demographic parity is enforced. "
            f"Demographic parity constrains selection rates, not error rates. "
            f"When base rates differ, equal selection rates force unequal false positive rates — "
            f"this is exactly Chouldechova's impossibility result."
        ),
        "option_b": (
            f"**Correct.** Demographic parity enforces equal selection rates, not equal error rates. "
            f"With different base rates, equal selection forces different false positive rates: "
            f"FPR Group A = {use_a['fpr']:.1%}, FPR Group B = {use_b['fpr']:.1%} (ratio: {_fpr_ratio:.1f}×). "
            f"The Trust & Safety Lead achieved one definition of fairness while violating another. "
            f"This is the Chouldechova (2017) impossibility theorem: pick your fairness criterion carefully."
        ),
        "option_c": (
            f"**This is not a bug — it is a mathematical invariant.** "
            f"The model behaves exactly as specified. Demographic parity was correctly enforced. "
            f"The FPR disparity (Group A = {use_a['fpr']:.1%}, Group B = {use_b['fpr']:.1%}) "
            f"is a necessary consequence of different base rates under equal selection. "
            f"No amount of debugging will resolve an impossibility theorem."
        ),
        "option_d": (
            f"**Demographic parity and equal FPR are equivalent only when base rates are equal.** "
            f"The simulator confirms: with base rate A = {use_a['p_flag']:.1%} vs B = {use_b['p_flag']:.1%}, "
            f"equal selection rates force FPR A = {use_a['fpr']:.1%} vs FPR B = {use_b['fpr']:.1%}. "
            f"Calibration does not resolve this — calibration is a third incompatible criterion."
        ),
    }

    mo.callout(
        mo.md(_feedback.get(act1_pred.value, "Select a prediction above.")),
        kind="success" if _is_correct else "warn",
    )
    return


# ─── ACT I REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, dp_satisfied, eo_satisfied, cal_satisfied):
    mo.stop(
        not (dp_satisfied or eo_satisfied or cal_satisfied),
        mo.callout(
            mo.md("Explore the simulator above before answering the reflection question."),
            kind="info",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) When the model is perfectly accurate — a perfect model satisfies all fairness criteria automatically":
                "option_a",
            "B) When base rates are equal across groups — the incompatibility only arises from base rate differences":
                "option_b",
            "C) When the training dataset is perfectly balanced — training data balance determines test-time fairness":
                "option_c",
            "D) When using calibrated probability outputs — calibration resolves the incompatibility":
                "option_d",
        },
        label="Reflection: Under what condition are all three fairness criteria (demographic parity, equalized odds, calibration) simultaneously achievable?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue to Act II."), kind="warn"),
    )

    _feedback = {
        "option_a": (
            "**Incorrect.** Even a perfect classifier cannot satisfy all three criteria "
            "when base rates differ. A perfect model achieves TPR=100% and FPR=0% for both groups, "
            "but its calibration (PPV) will still differ if base rates differ — "
            "because PPV = TP / (TP + FP) = base_rate / selection_rate, "
            "and equalizing selection rates with different base rates forces different PPV values. "
            "Perfect accuracy does not resolve the impossibility."
        ),
        "option_b": (
            "**Correct.** The Chouldechova (2017) impossibility theorem applies only when "
            "base rates differ between groups. When P(Y=1|G=A) = P(Y=1|G=B), there is no "
            "mathematical tension between demographic parity, equalized odds, and calibration — "
            "all three can be satisfied simultaneously. "
            "The social implication: if you want all three fairness criteria, "
            "the real intervention is equalizing the base rates (the underlying outcome distribution), "
            "not just the model's threshold."
        ),
        "option_c": (
            "**Incorrect.** Training data balance affects learning, not the test-time "
            "mathematical relationship between fairness criteria. "
            "Even if your training set is perfectly balanced, test-time base rates (the real-world "
            "distribution of outcomes) can differ between groups. "
            "The impossibility theorem is a property of the joint distribution at deployment time, "
            "not of the training procedure."
        ),
        "option_d": (
            "**Incorrect.** Calibration is one of the three incompatible criteria — "
            "it is not a resolution method. A calibrated model produces PPV that matches "
            "true positive rates, but this forces unequal selection rates when base rates differ. "
            "Adding calibration does not remove the tension between the other two criteria; "
            "it is itself one of the competing objectives."
        ),
    }

    _is_correct = act1_reflect.value == "option_b"
    mo.vstack([
        act1_reflect,
        mo.callout(
            mo.md(_feedback[act1_reflect.value]),
            kind="success" if _is_correct else "warn",
        ),
    ])
    return


# ─── ACT I MATHPEEK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Chouldechova (2017) impossibility theorem": mo.md("""
        **The three fairness criteria:**

        - **Demographic Parity (DP):**
          `P(Yhat=1 | G=A) = P(Yhat=1 | G=B)`
          Equal selection rates across groups.

        - **Equalized Odds (EO):**
          `TPR_A = TPR_B  AND  FPR_A = FPR_B`
          Equal true positive rates and equal false positive rates.

        - **Calibration (CAL):**
          `P(Y=1 | Yhat=1, G=A) = P(Y=1 | Yhat=1, G=B)`
          Equal precision (PPV) across groups.

        **The incompatibility (when base rates differ):**

        Let `prev_A = P(Y=1|G=A)` and `prev_B = P(Y=1|G=B)` with `prev_A ≠ prev_B`.

        From the confusion matrix identity:
        ```
        PPV = prev × TPR / (prev × TPR + (1-prev) × FPR)
        ```

        If EO holds (TPR_A = TPR_B, FPR_A = FPR_B) and prev_A ≠ prev_B,
        then PPV_A ≠ PPV_B → **calibration is violated**.

        If DP holds (p_flag_A = p_flag_B) and prev_A ≠ prev_B,
        then either TPR or FPR must differ → **equalized odds is violated**.

        **The FPR disparity under demographic parity (numerical example):**
        ```
        Suppose: prev_A = 2%,  prev_B = 10%,  target flag rate = 6%

        Group A flags 6%: TP = 2%, FP = 4%, FPR = 4% / 98% = 4.1%
        Group B flags 6%: TP = 6%, FP = 0%, FPR = 0% / 90% = 0.0%

        Demographic parity: ✓ (both at 6%)
        Equalized odds:     ✗ (FPR 4.1% vs 0%)
        Calibration:        ✗ (PPV = 33% vs 100%)
        ```

        **When all three are simultaneously achievable:**
        Only when `prev_A = prev_B`. Equal base rates eliminate the mathematical tension.
        Source: Chouldechova, A. (2017). Fair prediction with disparate impact.
        Big Data, 5(2), 153–163.
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(
            mo.md("Complete Act I (including the reflection question) to unlock Act II."),
            kind="warn",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo, COLORS):
    _act2_num      = "II"
    _act2_color    = COLORS["GreenLine"]
    _act2_title    = "Responsible Deployment at Scale"
    _act2_duration = "20&ndash;25 min"
    _act2_why      = ("Act I proved that fairness criteria are mathematically incompatible. "
                      "Now face the fleet-scale consequence: 1 billion users across 180 countries "
                      "means the fairness criterion you choose in one jurisdiction violates a "
                      "different jurisdiction&#x2019;s regulatory standard. "
                      "You must make an explicit policy decision &mdash; and every decision "
                      "produces a specific false-positive rate that scales from \u201cnegligible\u201d "
                      "to millions of affected users per day.")
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 40px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {_act2_color}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;
                            flex-shrink: 0;">{_act2_num}</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">
                    Act {_act2_num} &middot; {_act2_duration}</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">
                {_act2_title}
            </div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                {_act2_why}
            </div>
        </div>
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["GreenLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['GreenL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Chief AI Ethics Officer
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We operate a hiring algorithm for 10,000 companies across 180 countries — 1 billion
            users total. Legal requirements are fractured: the EU AI Act mandates calibration
            (equal precision across protected groups). US employment law under EEOC precedent
            requires equalized odds (equal false negative rates for qualified candidates). Some
            markets have no requirements. We cannot use the same model configuration everywhere.
            Design a deployment strategy that achieves regulatory compliance in our two largest
            markets while minimizing the number of users harmed by false positives."
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Jurisdiction Problem at Scale

    The impossibility theorem means you cannot satisfy the EU and US requirements
    with a single global model configuration. The engineering solution: use a
    calibrated base model globally, then apply jurisdiction-specific post-processing
    to adjust thresholds per market.

    The scale impact is severe. With 1 billion users:

    ```
    Daily hires evaluated:  ~5 million (assuming 0.5% of users per day)
    EU market:              ~300 million users (30%)
    US market:              ~200 million users (20%)
    Rest of world:          ~500 million users (50%)

    False positive impact = users_in_market × FPR × daily_evaluation_fraction
    ```

    Every 0.1% increase in FPR at this scale is 5,000 users per day falsely
    rejected from hiring opportunities.
    """)
    return


# ─── ACT II PREDICTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Use one model for all markets — fairness should be universal and one standard is simpler":
                "option_a",
            "B) Use calibrated probabilities globally as the base model, then apply jurisdiction-specific post-processing (equalized odds in US, calibration in EU) to satisfy local requirements":
                "option_b",
            "C) Use the most restrictive standard globally — equalized odds everywhere, even where not required":
                "option_c",
            "D) Do not use ML for hiring at this scale — the regulatory and ethical risk is too high":
                "option_d",
        },
        label="As Chief AI Ethics Officer, which deployment strategy best satisfies regulatory requirements in the EU and US while serving all 1B users?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your predicted strategy to unlock the Responsible Deployment Designer."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Strategy locked:** {act2_pred.value}. Now configure the deployment below."),
        kind="info",
    )
    return


# ─── ACT II INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Responsible Deployment Designer")
    return


@app.cell(hide_code=True)
def _(mo):
    global_standard = mo.ui.radio(
        options={
            "Calibration (equal PPV — EU AI Act)": "cal",
            "Equalized Odds (equal TPR/FPR — US EEOC)": "eo",
            "Demographic Parity (equal selection)": "dp",
        },
        value="Calibration (equal PPV — EU AI Act)",
        label="Global base standard:",
        inline=False,
    )
    eu_threshold_adj = mo.ui.slider(
        start=-20, stop=20, value=0, step=2,
        label="EU threshold adjustment (pp offset for post-processing)",
        show_value=True,
    )
    us_threshold_adj = mo.ui.slider(
        start=-20, stop=20, value=0, step=2,
        label="US threshold adjustment (pp offset for post-processing)",
        show_value=True,
    )
    audit_frequency = mo.ui.slider(
        start=1, stop=52, value=4, step=1,
        label="Audit frequency (times per year)",
        show_value=True,
    )
    transparency_tier = mo.ui.radio(
        options={
            "Full disclosure (model cards + per-group metrics published)": "full",
            "Summary (aggregate compliance status only)": "summary",
            "None (internal only)": "none",
        },
        value="Summary (aggregate compliance status only)",
        label="Transparency tier:",
        inline=False,
    )
    mo.vstack([
        mo.md("""
        Configure the global deployment strategy. The base model produces calibrated
        probabilities. Post-processing adjusts thresholds per jurisdiction to satisfy
        local regulatory requirements. Audit cost = frequency × users × sampling cost.
        """),
        global_standard,
        mo.hstack([eu_threshold_adj, us_threshold_adj, audit_frequency], justify="start", gap="2rem"),
        transparency_tier,
    ])
    return (global_standard, eu_threshold_adj, us_threshold_adj, audit_frequency, transparency_tier)


@app.cell(hide_code=True)
def _(
    mo, go, apply_plotly_theme, COLORS,
    global_standard, eu_threshold_adj, us_threshold_adj,
    audit_frequency, transparency_tier,
    context_toggle, USERS_SCALE, AUDIT_COST_PER_K,
):
    # ── Deployment physics engine ─────────────────────────────────────────────
    #
    # Global base model:
    #   - Calibrated probabilities (always)
    #   - Global FPR depends on global base rate and global standard
    #
    # Market model:
    #   - EU: 30% of users → 300M users → EU AI Act requires calibration
    #   - US: 20% of users → 200M users → EEOC requires equalized odds
    #   - ROW: 50% of users → 500M users → No specific requirement
    #
    # Fairness compliance model (simplified):
    #   - CAL standard satisfies EU by construction; needs EO post-processing for US
    #   - EO standard satisfies US; needs calibration post-processing for EU
    #   - DP satisfies neither EU nor US as defined in this scenario
    #
    # Threshold adjustment effect:
    #   - +10pp adjustment → FPR drops ~0.8pp (stricter)
    #   - -10pp adjustment → FPR rises ~0.8pp (lenient)
    #   (Linear approximation for pedagogical purposes)
    #
    # Scale impact:
    #   daily_fp = users_in_market × daily_eval_fraction × FPR
    #   daily_eval_fraction = 0.005 (0.5% of users evaluated daily — hiring context)
    #
    # Audit cost:
    #   audit_cost_annual = audit_frequency × (USERS_SCALE / 1e3) × AUDIT_COST_PER_K
    #   (Proportional to platform size; sample audit, not full census)
    #   We use a realistic sample fraction: audit_samples = 10,000 per audit event
    #   audit_cost_annual = audit_frequency × 10 × AUDIT_COST_PER_K (in thousands USD)

    gstd = global_standard.value
    eu_adj = eu_threshold_adj.value / 100.0  # pp → fraction
    us_adj = us_threshold_adj.value / 100.0
    n_audits = audit_frequency.value
    trans = transparency_tier.value
    ctx2 = context_toggle.value

    # ── Market sizes ──────────────────────────────────────────────────────────
    eu_users  = 0.30 * USERS_SCALE   # 300M
    us_users  = 0.20 * USERS_SCALE   # 200M
    row_users = 0.50 * USERS_SCALE   # 500M
    daily_eval_frac = 0.005           # 0.5% evaluated daily

    # ── Base FPR by standard (pre-adjustment) ─────────────────────────────────
    # These represent a realistic production hiring model:
    # Base rates: majority group 5% qualified, minority group 5% qualified (equal here)
    # but different false positive rates under different criteria due to different
    # thresholds implied by each standard.
    fpr_base = {"cal": 0.08, "eo": 0.06, "dp": 0.10}
    base_fpr = fpr_base.get(gstd, 0.08)

    # ── Per-jurisdiction compliance check ─────────────────────────────────────
    # EU requires CAL:
    #   If global standard is CAL, EU is compliant.
    #   EU threshold adjustment can further improve compliance.
    #   Compliance threshold: FPR disparity < 3pp (using adj as proxy)
    eu_effective_fpr = max(0.0, base_fpr - eu_adj * 0.8)
    eu_compliant = (gstd == "cal") or (abs(eu_adj) >= 0.08 and gstd != "dp")

    # More nuanced: CAL satisfies EU natively; EO with positive adjustment can
    # approximate CAL compliance; DP cannot satisfy EU
    if gstd == "cal":
        eu_compliant = True
    elif gstd == "eo" and eu_adj >= 0.05:
        eu_compliant = True  # Post-processing can bridge EO → CAL if threshold raised
    else:
        eu_compliant = False

    # US requires EO:
    #   If global standard is EO, US is compliant.
    #   US threshold adjustment can further improve compliance.
    us_effective_fpr = max(0.0, base_fpr - us_adj * 0.8)
    if gstd == "eo":
        us_compliant = True
    elif gstd == "cal" and us_adj >= 0.05:
        us_compliant = True  # Post-processing can bridge CAL → EO if threshold adjusted
    else:
        us_compliant = False

    # ROW: no requirement — always compliant
    row_compliant = True

    # ── Scale impact calculation ───────────────────────────────────────────────
    # Source: @sec-responsible-engineering — "At 1B user scale, a 0.1% FPR
    # difference means millions of users experiencing disparate treatment"
    eu_daily_fp  = eu_users  * daily_eval_frac * eu_effective_fpr
    us_daily_fp  = us_users  * daily_eval_frac * us_effective_fpr
    row_daily_fp = row_users * daily_eval_frac * base_fpr
    total_daily_fp = eu_daily_fp + us_daily_fp + row_daily_fp
    total_daily_fp_millions = total_daily_fp / 1_000_000

    # ── Overall accuracy proxy ────────────────────────────────────────────────
    # Base accuracy by standard (simplified: EO is slightly lower due to stricter constraints)
    acc_base = {"cal": 0.91, "eo": 0.89, "dp": 0.87}
    base_acc = acc_base.get(gstd, 0.89)
    # Adjustments reduce accuracy slightly
    overall_accuracy = max(0.70, base_acc - abs(eu_adj) * 0.3 - abs(us_adj) * 0.3)

    # ── Audit cost ────────────────────────────────────────────────────────────
    # 10,000 samples per audit event at $5K per 1,000 samples
    # Source: AUDIT_COST_PER_K constant ($5K/1K samples)
    audit_samples_per_event = 10_000
    audit_cost_per_event = (audit_samples_per_event / 1000) * AUDIT_COST_PER_K
    annual_audit_cost_k = n_audits * audit_cost_per_event  # in $K
    annual_audit_cost_m = annual_audit_cost_k / 1000       # in $M

    # ── Compliance status ─────────────────────────────────────────────────────
    n_compliant = sum([eu_compliant, us_compliant, row_compliant])
    n_markets = 3
    compliant_pct = n_compliant / n_markets * 100

    # ── Colors ────────────────────────────────────────────────────────────────
    eu_col  = COLORS["GreenLine"] if eu_compliant  else COLORS["RedLine"]
    us_col  = COLORS["GreenLine"] if us_compliant  else COLORS["RedLine"]
    row_col = COLORS["GreenLine"]
    acc_col = COLORS["GreenLine"] if overall_accuracy > 0.88 else COLORS["OrangeLine"] if overall_accuracy > 0.82 else COLORS["RedLine"]

    # ── Regional compliance bar chart ─────────────────────────────────────────
    _regions  = ["EU (300M)", "US (200M)", "Rest of World (500M)"]
    _compliant = [eu_compliant, us_compliant, row_compliant]
    _colors_bar = [eu_col, us_col, row_col]
    _fprs = [eu_effective_fpr * 100, us_effective_fpr * 100, base_fpr * 100]

    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(
        name="Effective FPR (%)",
        x=_regions,
        y=_fprs,
        marker_color=_colors_bar,
        text=[f"{v:.1f}%" for v in _fprs],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}% FPR<br>Compliant: %{customdata}<extra></extra>",
        customdata=["Yes" if c else "No" for c in _compliant],
    ))
    _fig2.update_layout(
        yaxis=dict(title="Effective FPR (%)", range=[0, max(_fprs) * 1.3]),
        height=260,
        margin=dict(l=50, r=20, t=30, b=40),
        title=dict(
            text=f"Per-market effective FPR — Global standard: {gstd.upper()}",
            font=dict(size=13),
        ),
    )
    apply_plotly_theme(_fig2)

    # ── Formula block ─────────────────────────────────────────────────────────
    _formula2 = f"""
**Deployment physics:**

```
Scale impact model:
  daily_fp = users × eval_fraction × effective_FPR

  EU  ({eu_users/1e6:.0f}M users):  {eu_daily_fp/1000:.0f}K false positives/day
  US  ({us_users/1e6:.0f}M users):  {us_daily_fp/1000:.0f}K false positives/day
  ROW ({row_users/1e6:.0f}M users): {row_daily_fp/1000:.0f}K false positives/day
  Total: {total_daily_fp_millions:.2f}M false positives/day

Audit cost:
  {n_audits} audits/yr × {audit_samples_per_event:,} samples × ${AUDIT_COST_PER_K:.0f}/1K
  = ${annual_audit_cost_m:.1f}M/year

Compliance status:
  EU  (CAL required): {'✓ Compliant' if eu_compliant else '✗ Non-compliant'}
  US  (EO required):  {'✓ Compliant' if us_compliant else '✗ Non-compliant'}
  ROW (no req.):      ✓ Compliant

Overall accuracy: {overall_accuracy:.1%}
```
"""

    # ── Assemble output ───────────────────────────────────────────────────────
    mo.vstack([
        mo.md(_formula2),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
            <!-- EU compliance -->
            <div style="padding: 18px; border: 2px solid {eu_col}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">EU Compliance</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {eu_col}; margin: 6px 0;">
                    {'✓' if eu_compliant else '✗'}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">CAL required</div>
            </div>
            <!-- US compliance -->
            <div style="padding: 18px; border: 2px solid {us_col}; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">US Compliance</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {us_col}; margin: 6px 0;">
                    {'✓' if us_compliant else '✗'}
                </div>
                <div style="font-size: 0.78rem; color: #64748b;">EO required</div>
            </div>
            <!-- Total daily false positives -->
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">Daily False Positives</div>
                <div style="font-size: 2.0rem; font-weight: 800;
                            color: {'#CB202D' if total_daily_fp_millions > 1 else '#CC5500' if total_daily_fp_millions > 0.5 else '#008F45'};
                            margin: 6px 0;">
                    {total_daily_fp_millions:.2f}M
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">users/day harmed</div>
            </div>
            <!-- Overall accuracy -->
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">Overall Accuracy</div>
                <div style="font-size: 2.0rem; font-weight: 800; color: {acc_col}; margin: 6px 0;">
                    {overall_accuracy:.1%}
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">global model</div>
            </div>
            <!-- Audit cost -->
            <div style="padding: 18px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 155px; text-align: center; background: white;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b;
                            text-transform: uppercase; letter-spacing: 0.05em;">Annual Audit Cost</div>
                <div style="font-size: 2.0rem; font-weight: 800; color: {COLORS['BlueLine']}; margin: 6px 0;">
                    ${annual_audit_cost_m:.1f}M
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8;">{n_audits}×/year</div>
            </div>
        </div>
        """),
        mo.as_html(_fig2),
    ])
    return (
        eu_compliant, us_compliant, row_compliant,
        total_daily_fp_millions, overall_accuracy,
        annual_audit_cost_m, compliant_pct,
        eu_daily_fp, us_daily_fp, row_daily_fp,
        n_audits, trans, gstd,
    )


# ─── ACT II FAILURE STATES ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, eu_compliant, us_compliant, gstd):
    _msgs = []
    if not eu_compliant:
        _msgs.append(mo.callout(mo.md(
            f"**Regulatory non-compliance: EU requires Calibration (equal PPV across protected groups) "
            f"— current deployment with global standard {gstd.upper()} violates this requirement.** "
            f"The EU AI Act (Article 10) mandates calibration for high-risk AI systems including "
            f"employment decisions. Non-compliance exposure: fines up to 6% of global annual revenue. "
            f"Fix: either switch global standard to CAL, or increase the EU threshold adjustment "
            f"to +8pp or more to approximate calibration via post-processing."
        ), kind="danger"))
    if not us_compliant:
        _msgs.append(mo.callout(mo.md(
            f"**Regulatory non-compliance: US EEOC guidelines require Equalized Odds (equal TPR/FPR "
            f"for qualified candidates across protected groups) — current deployment with global "
            f"standard {gstd.upper()} violates this requirement.** "
            f"Under the Uniform Guidelines on Employee Selection Procedures, disparate impact "
            f"above the 4/5 rule creates legal liability. "
            f"Fix: either switch global standard to EO, or increase the US threshold adjustment "
            f"to +8pp or more to approximate equalized odds via post-processing."
        ), kind="danger"))
    if _msgs:
        mo.vstack(_msgs)
    else:
        mo.callout(mo.md(
            "**Both major markets compliant.** EU (CAL) and US (EO) requirements satisfied. "
            "Continue to scale impact check below."
        ), kind="success")
    return


@app.cell(hide_code=True)
def _(mo, total_daily_fp_millions, eu_daily_fp, us_daily_fp, row_daily_fp):
    _fp_threshold = 1.0  # 1M daily false positives trigger warn
    if total_daily_fp_millions > _fp_threshold:
        mo.callout(mo.md(
            f"**Scale impact: {total_daily_fp_millions:.2f}M users falsely rejected per day.** "
            f"Breakdown: EU {eu_daily_fp/1e6:.2f}M, US {us_daily_fp/1e6:.2f}M, "
            f"ROW {row_daily_fp/1e6:.2f}M. "
            f"At this scale, each percentage point of FPR translates directly into "
            f"millions of hiring decisions per day. A candidate falsely rejected from a "
            f"job opportunity experiences a concrete economic harm — the aggregate is a "
            f"population-level economic impact. "
            f"Reduce the global FPR by switching to a stricter standard or tightening thresholds."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Scale impact within bounds: {total_daily_fp_millions:.2f}M users falsely rejected/day "
            f"(below 1M threshold).** "
            f"This is manageable at this scale, though each false positive still represents "
            f"a real person affected. Continue monitoring as user base grows."
        ), kind="info")
    return


# ─── ACT II PREDICTION REVEAL ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, eu_compliant, us_compliant, gstd):
    _both_compliant = eu_compliant and us_compliant
    _is_correct_pred = act2_pred.value == "option_b"

    _feedback2 = {
        "option_a": (
            f"**A single global model cannot satisfy both EU (CAL) and US (EO) requirements simultaneously** "
            f"— the impossibility theorem prevents this. Even with the same model weights, "
            f"different jurisdictions require different post-processing that a single configuration "
            f"cannot provide. The current deployment shows {'both compliant' if _both_compliant else 'compliance failures'}. "
            f"Universal fairness is a political goal; operationally, you need jurisdiction-aware post-processing."
        ),
        "option_b": (
            f"**Correct strategy.** Calibrated probabilities as the global base model give you "
            f"a neutral starting point from which any fairness criterion can be derived via "
            f"post-processing. EU gets calibration natively; US gets equalized odds via "
            f"threshold adjustment. The current configuration shows "
            f"{'both major markets compliant' if _both_compliant else 'compliance gaps — adjust thresholds'}. "
            f"This is the only strategy that can satisfy mathematically incompatible requirements "
            f"across different jurisdictions simultaneously."
        ),
        "option_c": (
            f"**Equalized odds everywhere creates an EU compliance gap.** "
            f"EU AI Act requires calibration for high-risk employment AI. "
            f"Deploying EO globally satisfies the US but violates EU requirements. "
            f"Current EU status: {'compliant' if eu_compliant else 'non-compliant'}. "
            f"A single global standard cannot satisfy both regulators given the "
            f"impossibility theorem — you need jurisdiction-specific post-processing."
        ),
        "option_d": (
            f"**Not using ML does not avoid the problem — it just delegates the decision to humans.** "
            f"Human hiring decisions exhibit well-documented biases that exceed most ML model disparities. "
            f"The question is not whether to use ML, but how to deploy it responsibly. "
            f"The current configuration demonstrates that a compliant, responsible ML deployment "
            f"is achievable — {'and is currently compliant' if _both_compliant else 'though threshold tuning is required'}."
        ),
    }

    mo.callout(
        mo.md(_feedback2.get(act2_pred.value, "")),
        kind="success" if _is_correct_pred else "warn",
    )
    return


# ─── ACT II REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, eu_compliant, us_compliant):
    mo.stop(
        not (eu_compliant and us_compliant),
        mo.callout(
            mo.md(
                "Achieve compliance in both EU and US markets before answering the "
                "reflection question. Adjust the global standard and threshold settings above."
            ),
            kind="warn",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Larger models are more biased — scale automatically increases model bias":
                "option_a",
            "B) At billion-user scale, a 0.1% FPR difference between groups means millions of users experiencing disparate treatment — scale amplifies tiny model disparities into massive societal impact":
                "option_b",
            "C) Scale requires more compute, leaving less engineering budget for fairness research and mitigation":
                "option_c",
            "D) Large-scale models are harder to audit because they have more parameters to inspect":
                "option_d",
        },
        label="Reflection: Why does deploying at 1B user scale change the ethical stakes of fairness decisions, compared to deploying at 1M user scale?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to complete Act II."), kind="warn"),
    )

    _feedback_r2 = {
        "option_a": (
            "**Scale does not automatically increase model bias.** "
            "A larger model trained on more data can exhibit less bias, not more. "
            "The ethical stakes change not because the model worsens at scale, but because "
            "the impact of any given error rate multiplies directly with the number of users. "
            "A 1% FPR on a 1,000-user system affects 10 people. "
            "A 1% FPR on a 1B-user system affects 10 million people per cycle."
        ),
        "option_b": (
            "**Correct.** Scale is an amplifier, not a cause of bias. "
            "The same 0.1% FPR disparity that is statistically insignificant at 10,000 users "
            "translates to 50,000 users per day at 1B scale (with 5% daily evaluation rate). "
            "This is the core insight from @sec-responsible-engineering: "
            "responsible AI at scale is not about the model — it is about the population-level "
            "consequences of model decisions. Engineering for fairness at scale requires "
            "treating FPR differentials as engineering SLOs with the same rigor as latency budgets."
        ),
        "option_c": (
            "**Resource constraints are real but they are not the primary reason scale changes ethics.** "
            "The major cloud providers and large tech companies spend significant resources on "
            "fairness tooling specifically because of scale. The ethical stakes at scale are not "
            "primarily about engineering budget — they are about the magnitude of harm when even "
            "small model disparities multiply across hundreds of millions of users."
        ),
        "option_d": (
            "**Auditability is a legitimate challenge, but it is secondary to the scale impact.** "
            "Parameter count is not what makes large-scale systems hard to audit — "
            "it is the interaction between the model, data, and deployment context. "
            "More importantly, the ethical stakes at 1B scale are driven by the sheer number "
            "of users affected by any given error rate, not by the difficulty of the audit."
        ),
    }

    _is_correct_r2 = act2_reflect.value == "option_b"
    mo.vstack([
        act2_reflect,
        mo.callout(
            mo.md(_feedback_r2[act2_reflect.value]),
            kind="success" if _is_correct_r2 else "warn",
        ),
    ])
    return


# ─── ACT II MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations — Jurisdiction-aware post-processing and scale impact": mo.md("""
        **Jurisdiction-aware post-processing:**

        Base model produces calibrated probability scores `p(Y=1|X)`.
        Per-jurisdiction threshold `t_j` converts scores to decisions:
        ```
        decision_j(x) = 1  if p(Y=1|x) > t_j
                         0  otherwise
        ```

        EU (calibration requirement):
        ```
        t_EU = argmin_t |PPV(t, G=majority) - PPV(t, G=minority)|
        where PPV(t, G) = P(Y=1 | score > t, G)
        ```

        US (equalized odds requirement):
        ```
        t_US = argmin_t |TPR(t, G=majority) - TPR(t, G=minority)|
               subject to |FPR(t, G=majority) - FPR(t, G=minority)| < epsilon
        ```

        **Scale impact model:**
        ```
        N_fp_daily(market) = N_users(market) × eval_fraction × FPR(market)

        eval_fraction = 0.5%  (0.5% of users evaluated per day — hiring context)

        Example: EU market, FPR = 8%:
          N_fp = 300M × 0.005 × 0.08 = 120,000 false rejections/day
        ```

        **Scale amplification factor:**
        ```
        impact_ratio = (FPR_group_A / FPR_group_B) × N_daily_evaluations

        At 1M users: 5% FPR disparity = 25 extra users/day affected
        At 1B users: 5% FPR disparity = 25,000 extra users/day affected
        Same disparity, 1,000× impact.
        ```

        **Audit cost formula:**
        ```
        audit_cost_annual = n_audits × audit_samples × cost_per_1K / 1000
        cost_per_1K = $5,000  (industry estimate for model audit sampling)

        Example: 4 audits/year × 10,000 samples × $5 per sample = $200,000/year
        ```

        **Transparency-utility tradeoff:**
        - Full disclosure: highest regulatory trust, risk of competitive information leakage
        - Summary disclosure: balance of transparency and proprietary protection
        - None: lowest overhead, highest regulatory and reputational risk at scale
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 20: SYNTHESIS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.md("---"),

        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Demographic parity and equalized odds are logically incompatible when base rates differ.</strong>
                    The Chouldechova-Kleinberg impossibility theorem (2016&ndash;2017) is a mathematical proof,
                    not an empirical observation. When P(Y=1|S=a) &ne; P(Y=1|S=b), no algorithm, threshold,
                    or training procedure can simultaneously satisfy all three criteria. The engineer
                    must choose which constraint to satisfy and accept violations of the others.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. The fairness tax scales with base-rate divergence, not with model quality.</strong>
                    A 4% accuracy drop to enforce demographic parity grows larger as the base-rate gap widens.
                    In heterogeneous multi-region deployments (e.g., 70% vs. 25% base rates),
                    the tax can exceed 10 percentage points &mdash; not because the model is worse,
                    but because the constraint is more demanding.
                </div>
                <div>
                    <strong>3. At 1 billion users, fairness criterion choice is a policy decision with measurable societal consequences.</strong>
                    A 0.1% false positive rate that seems negligible becomes 1 million incorrectly flagged
                    users per day. Fleet-scale deployment transforms fairness from an abstract metric
                    into a concrete harm rate that must be budgeted like a latency SLA.
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
                    What&#x2019;s Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 17: The Constraints Never Lie</strong> &mdash; This lab quantified
                    the fairness impossibility as a constraint on responsible deployment. The
                    capstone now asks: when all six constraint families (memory, compute, networking,
                    privacy, sustainability, fairness) must be satisfied simultaneously for a
                    1,000-hospital medical fleet, which ones are truly incompatible?
                </div>
            </div>

            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-responsible-ai-fairness-machine-learning-2ba4 for
                    the full Chouldechova proof and the FairnessTaxAnalysis worked example
                    showing the exact 85% &rarr; 81% accuracy cost.<br/>
                    <strong>Build:</strong> The FairnessConstraintSweep LEGO cell demonstrates
                    the three-criteria incompatibility with a threshold sweep visualization.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. The Chouldechova-Kleinberg theorem proves that Demographic Parity, Equalized Odds, and Calibration cannot all hold when base rates differ. This is a mathematical proof, not an empirical observation. What must the engineer choose, and what must they accept as a consequence?
2. A 4% accuracy drop to enforce demographic parity grows larger as the base-rate gap widens. With Group A at 70% and Group B at 25% base rate, why can the fairness tax exceed 10 percentage points, and why is this independent of model quality?
3. At 1 billion users, a 0.1% false positive rate becomes 1 million incorrectly flagged users per day. Why does fleet-scale deployment transform fairness from an abstract metric into a concrete harm rate that must be budgeted like a latency SLA?

**You're ready to move on if you can:**
- Explain why the fairness impossibility theorem forces an explicit policy choice, not a technical optimization
- Calculate the accuracy cost (fairness tax) of enforcing a specific fairness criterion given base-rate divergence
- Translate per-user false positive rates into absolute daily harm counts at fleet scale
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_pred, act2_pred, act2_reflect,
    eu_compliant, us_compliant,
    total_daily_fp_millions, overall_accuracy,
    compliant_pct, gstd, trans, n_audits,
    dp_satisfied, eo_satisfied, cal_satisfied,
):
    _ctx = context_toggle.value
    _act1_correct = act1_pred.value == "option_b"
    _act2_correct = act2_pred.value == "option_b"
    _constraint_hit = (not eu_compliant) or (not us_compliant) or (total_daily_fp_millions > 1.0)
    _scale_impact_high = total_daily_fp_millions > 1.0

    ledger.save(chapter="v2_16", design={
        "context":                _ctx,
        "global_standard":        gstd,
        "eu_compliant":           eu_compliant,
        "us_compliant":           us_compliant,
        "daily_false_positives":  total_daily_fp_millions,
        "overall_accuracy":       overall_accuracy,
        "act1_prediction":        str(act1_pred.value),
        "act1_correct":           _act1_correct,
        "act2_result":            compliant_pct / 100.0,
        "act2_decision":          str(act2_pred.value),
        "constraint_hit":         _constraint_hit,
        "scale_impact_high":      _scale_impact_high,
    })

    # ── HUD ───────────────────────────────────────────────────────────────────
    _eu_col  = COLORS["GreenLine"] if eu_compliant  else COLORS["RedLine"]
    _us_col  = COLORS["GreenLine"] if us_compliant  else COLORS["RedLine"]
    _fp_col  = (COLORS["RedLine"] if _scale_impact_high
                else COLORS["OrangeLine"] if total_daily_fp_millions > 0.5
                else COLORS["GreenLine"])
    _acc_col = (COLORS["GreenLine"] if overall_accuracy > 0.88
                else COLORS["OrangeLine"] if overall_accuracy > 0.82
                else COLORS["RedLine"])
    _act1_col = COLORS["GreenLine"] if _act1_correct else COLORS["OrangeLine"]
    _act2_col = COLORS["GreenLine"] if _act2_correct else COLORS["OrangeLine"]

    _dp_sym  = "✓" if dp_satisfied  else "✗"
    _eo_sym  = "✓" if eo_satisfied  else "✗"
    _cal_sym = "✓" if cal_satisfied else "✗"

    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 24px 32px; border-radius: 16px; color: white;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.25); margin-top: 16px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 14px;">
                Lab 16 · Design Ledger HUD · v2_16
            </div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap;">

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">Context</div>
                    <div style="font-size: 1.0rem; font-weight: 700; color: #f8fafc;">{_ctx.title()}</div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">EU Compliance</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {_eu_col};">
                        {'✓' if eu_compliant else '✗'} {'CAL' if eu_compliant else 'FAIL'}
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">US Compliance</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {_us_col};">
                        {'✓' if us_compliant else '✗'} {'EO' if us_compliant else 'FAIL'}
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">Daily FP</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {_fp_col};">
                        {total_daily_fp_millions:.2f}M
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">Overall Accuracy</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {_acc_col};">
                        {overall_accuracy:.1%}
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">Fairness Criteria</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                        DP{_dp_sym} EO{_eo_sym} CAL{_cal_sym}
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">Act I Prediction</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {_act1_col};">
                        {'Correct' if _act1_correct else 'Review'}
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 10px; padding: 14px 18px; min-width: 130px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #94a3b8; text-transform: uppercase;
                                letter-spacing: 0.1em; margin-bottom: 4px;">Act II Strategy</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: {_act2_col};">
                        {'Correct' if _act2_correct else 'Review'}
                    </div>
                </div>

            </div>
            <div style="margin-top: 14px; font-size: 0.75rem; color: #475569;">
                Ledger saved: chapter=v2_16 · constraint_hit={'True' if _constraint_hit else 'False'} · scale_impact_high={'True' if _scale_impact_high else 'False'}
            </div>
        </div>
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
