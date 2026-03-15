import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 13: THE PRIVACY MEASUREMENT GAP
#
# Volume II, Chapter 13: Security & Privacy
#
# Core invariant: Differential privacy ε-δ tradeoff — privacy vs utility.
# Standard training without DP leaves "fingerprints" of training data in model
# weights. Membership inference attacks exploit overfitting and generalization
# gap. DP training adds calibrated Gaussian noise to gradients, limiting
# membership inference accuracy to near-random (~50-55%). The noise magnitude
# σ ≥ (C√2·ln(1.25/δ)) / ε determines the accuracy penalty.
#
# Structure:
#   Act I  — The Privacy Measurement Gap (12-15 min)
#     Stakeholder: Chief Privacy Officer — 94% membership inference accuracy
#     Instruments: Membership inference risk visualizer
#     Prediction-vs-reality overlay: DP vs no-DP membership inference accuracy
#     Reflection: Why generalization gap drives membership inference risk
#
#   Act II — DP Parameter Selection (20-25 min)
#     Stakeholder: ML Platform Lead — HIPAA ε ≤ 1, 90% accuracy floor
#     Instruments: DP-SGD configurator with full parameter space
#     Failure state: ε > 1 with medical context → danger; accuracy < 90% → warn
#     Reflection: Why on-prem offers stronger privacy guarantees than cloud
#
# 2 Contexts: On-premises (strong isolation) vs Cloud (shared infrastructure)
#
# Design Ledger: saves context, dp_epsilon, dp_delta, model_accuracy,
#                membership_inference_risk, hipaa_compliant, predictions,
#                act2_result, act2_decision, constraint_hit, clinical_viable.
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ────────────────────────
@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    import math

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

    # ── Hardware and regulatory constants ───────────────────────────────────
    # All sourced from @sec-security-privacy-differential-privacy-8c2b and
    # the chapter's defense-selection framework table.

    H100_BW_GBS         = 3350   # H100 SXM5 HBM3e bandwidth, NVIDIA spec
    H100_TFLOPS_FP16    = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB         = 80     # H100 SXM5 HBM3e capacity, NVIDIA spec

    HIPAA_MAX_EPSILON   = 1.0    # HIPAA-interpretable DP requirement (@tbl-defense-selection-framework)
    GDPR_MAX_EPSILON    = 1.0    # GDPR analogous requirement (@tbl-defense-selection-framework)

    # DP-SGD overhead: per-sample gradient clipping prevents batch-level
    # parallelism, reducing training throughput by 2-10x vs standard SGD.
    # Source: fn-dp-sgd-adoption, Abadi et al. 2016 (CCS).
    DP_SGD_OVERHEAD_MIN = 2.0    # minimum throughput reduction factor
    DP_SGD_OVERHEAD_MAX = 10.0   # maximum throughput reduction factor

    # Membership inference baseline accuracy without DP
    # Source: Yeom et al. 2018 (advantage ≤ generalization gap)
    MI_BASELINE_ACCURACY = 0.50  # near-random (50%) with DP training
    MI_MAX_ADVANTAGE     = 0.30  # max theoretical advantage = generalization gap

    # Accuracy penalty empirical baseline from chapter
    # @fig-privacy-utility-frontier: MNIST 95% at ε=1, CIFAR-10 ~82% at ε=8
    DP_MEDICAL_BASELINE_ACC  = 0.94  # baseline accuracy without DP (chapter scenario)
    DP_MEDICAL_EPS1_ACC      = 0.86  # accuracy at ε=1 (chapter scenario)
    DP_CLINICAL_MIN_ACC      = 0.90  # minimum viable clinical accuracy (chapter scenario)

    # Multi-tenant isolation tax: MIG partitioning adds ~15% overhead
    # Source: MultiTenantIsolation LEGO cell in chapter
    CLOUD_ISOLATION_OVERHEAD = 0.15  # 15% throughput loss from secure partitioning

    ledger = DesignLedger()
    return (
        mo, ledger, go, np, math,
        COLORS, LAB_CSS, apply_plotly_theme,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
        HIPAA_MAX_EPSILON, GDPR_MAX_EPSILON,
        DP_SGD_OVERHEAD_MIN, DP_SGD_OVERHEAD_MAX,
        MI_BASELINE_ACCURACY, MI_MAX_ADVANTAGE,
        DP_MEDICAL_BASELINE_ACC, DP_MEDICAL_EPS1_ACC, DP_CLINICAL_MIN_ACC,
        CLOUD_ISOLATION_OVERHEAD,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _indigo = COLORS["Cloud"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 13
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Privacy Measurement Gap
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                A model that achieves 94% membership inference accuracy is not
                "privacy-preserving." Differential privacy provides a mathematical
                guarantee: the ε-δ parameterization bounds how much any individual's
                data can influence any algorithm output. Smaller ε = stronger privacy,
                more noise, lower utility. This tradeoff is not a policy choice — it
                is a physical constraint on information.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: The Privacy Measurement Gap · Act II: DP Parameter Selection
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Prereq: @sec-security-privacy
                </span>
                <span class="badge badge-fail">
                    HIPAA Compliance Monitor Active
                </span>
                <span class="badge badge-warn">
                    Membership Inference Risk: Unquantified
                </span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the membership inference risk</strong> introduced by generalization gap — and measure exactly how DP training collapses 94% MI accuracy to near-random (&le;55%).</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate the noise scale</strong> &sigma; &ge; C&radic;(2&thinsp;ln(1.25/&delta;))/&epsilon; and predict the accuracy penalty for a given &epsilon;-&delta; budget in a HIPAA-constrained clinical setting.</div>
                <div style="margin-bottom: 3px;">3. <strong>Diagnose why on-premises isolation costs 0% throughput overhead while cloud MIG partitioning costs 15%</strong> — and identify which threat vector drives that structural difference.</div>
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
                    Differential privacy &epsilon;-&delta; definition from @sec-security-privacy-differential-privacy-8c2b &middot; Generalization gap concept from @sec-training-generalization
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
                "If your model never exposes raw training data, why can an attacker determine with 94% accuracy whether a specific person&#x2019;s record was in your training set &mdash; and what does it actually cost in model utility to stop them?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **Privacy Defined** (@sec-security-privacy-privacy-defined-da84) — The distinction between confidentiality (access control) and privacy (inference risk). Why removing names is insufficient: neural networks are correlation engines that memorize unique patterns from training data.
    - **Differential Privacy** (@sec-security-privacy-differential-privacy-8c2b) — The ε-δ formulation, the Gaussian mechanism, and why the noise scale σ ≥ (C√(2·ln(1.25/δ)))/ε is the governing equation for the privacy-utility tradeoff.
    - **Defense Selection Framework** (@tbl-defense-selection-framework) — HIPAA and GDPR requirements for ε ≤ 1 in healthcare and financial ML systems, and the 2–15% accuracy penalty range.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    context_toggle = mo.ui.radio(
        options={
            "On-Premises (Dedicated Hardware)": "on_prem",
            "Cloud (Shared Infrastructure)": "cloud",
        },
        value="On-Premises (Dedicated Hardware)",
        label="Deployment context:",
        inline=True,
    )

    mo.vstack([
        mo.Html(f"""
        <div style="border-bottom: 2px solid {COLORS['Border']}; padding-bottom: 16px; margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                Infrastructure Context — this selection affects isolation guarantees in Act II
            </div>
        </div>
        """),
        context_toggle,
        mo.callout(mo.md("""
        **On-Premises:** Hardware isolation via dedicated servers. No hypervisor layer. No multi-tenant GPU sharing.
        Side-channel attacks require physical access. Model artifacts never leave the perimeter.

        **Cloud:** Shared GPU infrastructure. MIG partitioning adds ~15% isolation overhead. Hypervisor-layer
        side-channels are a documented threat vector. Model checkpoints transit provider-controlled storage.
        """), kind="info"),
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The Privacy Measurement Gap"
    _act_duration = "12&ndash;15 min"
    _act_why      = ("You believe a model that never shares raw training data is privacy-preserving. "
                     "A membership inference attack achieving 94% accuracy will show that "
                     "memorization of individual records &mdash; not raw data exposure &mdash; "
                     "is the real threat, and that the only remedy is calibrated noise injected during training.")
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


# ─── ACT I: STAKEHOLDER MESSAGE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · Chief Privacy Officer
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our ML model was trained on 10 million medical records. Legal says we are
            'privacy preserving' because we do not store individual records anywhere — only
            the trained model weights. But our security audit just came back: a researcher
            used a membership inference attack and achieved 94% accuracy at determining
            whether a specific person's data was in our training set. Legal is asking me to
            explain how this is possible when we never exposed the raw data. I need an
            answer before the board meeting."
        </div>
    </div>
    """)
    return


# ─── ACT I: SCENARIO SETUP ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        ## The Physics of Membership Inference

        Standard training without differential privacy allows model weights to "memorize"
        specific training examples — a phenomenon that exploits the gap between training
        and test accuracy. A membership inference attack does not need access to the
        original training data. It needs only the trained model.

        **The Yeom et al. (2018) Result**: For a model with training accuracy $A_{train}$
        and test accuracy $A_{test}$, the membership inference advantage is bounded by:

        $$\\text{Advantage}_{MI} \\leq A_{train} - A_{test}$$

        The generalization gap *directly quantifies* the privacy risk. A model that
        achieves 98% training accuracy and 68% test accuracy (a 30% generalization gap)
        gives an adversary a theoretical 30% advantage above random — enough to push
        membership inference accuracy from 50% (random) to 80%.

        **Why DP training prevents this**: DP-SGD clips per-sample gradients to a maximum
        norm $C$ and adds calibrated Gaussian noise $\\mathcal{N}(0, \\sigma^2)$ during
        training. This noise prevents the model from learning individual-specific patterns,
        collapsing the generalization gap that membership inference exploits. With DP
        training at ε ≤ 1, membership inference accuracy drops to 50–55% — statistically
        indistinguishable from a random coin flip.
        """),
    ])
    return


# ─── ACT I: PREDICTION LOCK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) 94% accuracy is normal — all ML models memorize training data to this extent": "A",
            "B) The model is overfitting — reducing model size will fix the privacy problem": "B",
            "C) Training without DP allows membership inference — DP limits this to ~50-55% (near-random)": "C",
            "D) Membership inference requires access to training data, which the adversary lacks": "D",
        },
        label="""**Commit to your prediction before running the instruments.**

The CPO's model achieves 94% membership inference accuracy despite never exposing raw
training data. What is the correct explanation for this privacy failure?""",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.vstack([
            act1_pred,
            mo.callout(
                mo.md("Select your prediction to continue. Commit before the instruments run."),
                kind="warn",
            ),
        ])
    )
    mo.callout(
        mo.md(f"**Prediction locked:** Option {act1_pred.value[0]}. Now run the membership inference simulator below."),
        kind="info",
    )
    return


# ─── ACT I: INSTRUMENTS ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## Membership Inference Risk Visualizer")
    return


@app.cell(hide_code=True)
def _(mo):
    train_accuracy = mo.ui.slider(
        start=75, stop=99, value=94, step=1,
        label="Training accuracy (%)",
    )
    test_accuracy = mo.ui.slider(
        start=60, stop=98, value=82, step=1,
        label="Test accuracy (%)",
    )
    model_params_m = mo.ui.slider(
        start=1, stop=1000, value=100, step=10,
        label="Model size (millions of parameters)",
    )
    dp_epsilon_act1 = mo.ui.dropdown(
        options={
            "No DP (standard training)": "none",
            "ε = 0.1 (very strong privacy)": "0.1",
            "ε = 1.0 (strong privacy)": "1.0",
            "ε = 3.0 (moderate privacy)": "3.0",
            "ε = 10.0 (weak privacy)": "10.0",
        },
        value="No DP (standard training)",
        label="Differential privacy setting",
    )
    mo.vstack([
        mo.hstack([train_accuracy, test_accuracy], justify="start", gap=2),
        mo.hstack([model_params_m, dp_epsilon_act1], justify="start", gap=2),
    ])
    return (train_accuracy, test_accuracy, model_params_m, dp_epsilon_act1)


@app.cell(hide_code=True)
def _(mo, go, np, train_accuracy, test_accuracy, model_params_m, dp_epsilon_act1,
      apply_plotly_theme, COLORS, MI_BASELINE_ACCURACY):

    # ── Simulation Physics ───────────────────────────────────────────────────
    # Membership inference advantage from Yeom et al. 2018:
    #   Advantage ≤ A_train - A_test  (generalization gap)
    # Source: @sec-security-privacy-differential-privacy-8c2b

    _train_acc  = train_accuracy.value / 100.0
    _test_acc   = test_accuracy.value / 100.0
    _n_params_m = model_params_m.value
    _eps_str    = dp_epsilon_act1.value
    _use_dp     = _eps_str != "none"
    _eps        = float(_eps_str) if _use_dp else None

    # Generalization gap — the direct measure of memorization
    _gen_gap = max(0.0, _train_acc - _test_acc)

    # MI accuracy without DP: baseline 0.50 + bounded advantage from gen gap
    # A large model overfits more → higher effective advantage
    _size_amplifier = 1.0 + 0.15 * np.log10(max(1, _n_params_m))  # log scaling
    _mi_advantage_no_dp = min(_gen_gap * _size_amplifier, 0.49)
    _mi_acc_no_dp = MI_BASELINE_ACCURACY + _mi_advantage_no_dp

    # MI accuracy with DP: ε strongly suppresses membership inference
    # At ε=0.1: MI ≈ 50-51%, at ε=1: ≈ 51-53%, at ε=10: ≈ 55-60%
    # Source: Abadi et al. 2016, empirical DP-SGD results
    if _use_dp:
        _dp_mi_advantage = 0.005 * _eps  # ~0.5% per unit of epsilon
        _mi_acc_with_dp  = MI_BASELINE_ACCURACY + min(_dp_mi_advantage, 0.12)
    else:
        _mi_acc_with_dp = _mi_acc_no_dp

    # Privacy risk score (0-100) for visualization
    _risk_pct_no_dp  = min(100, _mi_advantage_no_dp / 0.49 * 100)
    _risk_pct_with_dp = ((_mi_acc_with_dp - 0.50) / 0.49) * 100 if _use_dp else _risk_pct_no_dp

    # Color coding based on MI accuracy
    def _mi_color(mi_acc):
        if mi_acc > 0.75:
            return COLORS["RedLine"]
        elif mi_acc > 0.60:
            return COLORS["OrangeLine"]
        else:
            return COLORS["GreenLine"]

    _color_no_dp  = _mi_color(_mi_acc_no_dp)
    _color_with_dp = _mi_color(_mi_acc_with_dp)

    # Build comparison bar chart
    _categories = ["Without DP", "With DP (selected ε)"]
    _mi_values  = [_mi_acc_no_dp * 100, _mi_acc_with_dp * 100]
    _bar_colors = [_color_no_dp, _color_with_dp]

    _fig = go.Figure()

    # Random-chance baseline
    _fig.add_hline(
        y=50.0, line_dash="dash", line_color=COLORS["TextMuted"],
        annotation_text="Random baseline (50%)", annotation_position="top left",
        line_width=1.5,
    )

    # MI accuracy bars
    _fig.add_trace(go.Bar(
        x=_categories,
        y=_mi_values,
        marker_color=_bar_colors,
        marker_line_width=0,
        width=0.4,
        text=[f"{v:.1f}%" for v in _mi_values],
        textposition="outside",
    ))

    _fig.update_layout(
        height=340,
        yaxis=dict(title="Membership Inference Accuracy (%)", range=[0, 105]),
        xaxis=dict(title=""),
        margin=dict(t=20, b=40),
        showlegend=False,
    )
    apply_plotly_theme(_fig)

    # ── Display ──────────────────────────────────────────────────────────────
    _dp_label = f"ε = {_eps:.1f}" if _use_dp else "disabled"
    _gap_pct  = _gen_gap * 100

    mo.vstack([
        mo.md(f"""
        ### Physics

        ```
        Generalization Gap         = A_train - A_test
                                   = {_train_acc*100:.0f}% - {_test_acc*100:.0f}%
                                   = {_gap_pct:.1f}%

        MI Advantage (no DP)       ≤ Gap × Size_amplifier
                                   ≈ {_gap_pct:.1f}% × {_size_amplifier:.2f}
                                   = {_mi_advantage_no_dp*100:.1f}%

        MI Accuracy (no DP)        = 50% + {_mi_advantage_no_dp*100:.1f}%
                                   = {_mi_acc_no_dp*100:.1f}%

        DP setting                 : {_dp_label}
        MI Accuracy (with DP)      = {_mi_acc_with_dp*100:.1f}%
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin: 20px 0;">
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px;
                        width: 200px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.85rem; font-weight: 600;">
                    Without DP
                </div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {_color_no_dp};">
                    {_mi_acc_no_dp*100:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">
                    Membership Inference Acc.
                </div>
            </div>
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px;
                        width: 200px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.85rem; font-weight: 600;">
                    With DP ({_dp_label})
                </div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {_color_with_dp};">
                    {_mi_acc_with_dp*100:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">
                    Membership Inference Acc.
                </div>
            </div>
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 8px;
                        width: 200px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.85rem; font-weight: 600;">
                    Generalization Gap
                </div>
                <div style="font-size: 2.2rem; font-weight: 900;
                            color: {'#CB202D' if _gap_pct > 15 else '#CC5500' if _gap_pct > 5 else '#008F45'};">
                    {_gap_pct:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">
                    Privacy Vulnerability
                </div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (
        _mi_acc_no_dp, _mi_acc_with_dp, _gen_gap, _use_dp, _eps,
        _mi_advantage_no_dp, _risk_pct_no_dp,
    )


# ─── ACT I: PREDICTION-VS-REALITY OVERLAY ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _mi_acc_no_dp):
    _CORRECT_OPTION = "C"
    _selected = act1_pred.value[0] if act1_pred.value else None

    if _selected is None:
        mo.stop(True, mo.callout(mo.md("Complete the prediction above first."), kind="warn"))

    _mi_pct = _mi_acc_no_dp * 100

    if _selected == _CORRECT_OPTION:
        _feedback = mo.callout(mo.md(f"""
        **Correct.** Standard training without DP produced **{_mi_pct:.1f}%** membership inference
        accuracy — far above the 50% random baseline. The model's generalization gap (training
        accuracy minus test accuracy) quantifies exactly how much individual-level memorization
        occurred. DP training collapses this gap by injecting Gaussian noise into gradients,
        limiting the adversary to ~50–55% accuracy — a coin flip. This is the mathematical
        guarantee: no individual's presence or absence can change the output distribution by
        more than a factor of e^ε.
        """), kind="success")
    elif _selected == "A":
        _feedback = mo.callout(mo.md(f"""
        **Not quite.** While memorization is common in unprotected models, it is not inevitable.
        DP training provably limits membership inference accuracy to ~50–55%. The simulator
        shows {_mi_pct:.1f}% accuracy without DP versus near-50% with DP training at ε ≤ 1.
        "Normal" is a matter of whether DP is applied — and for medical data, regulators treat
        high membership inference accuracy as a compliance failure, not a technical inevitability.
        """), kind="warn")
    elif _selected == "B":
        _feedback = mo.callout(mo.md(f"""
        **Not quite.** Reducing model size does reduce overfitting and the generalization gap,
        which narrows the membership inference advantage. But the correct fix for *privacy* is
        differential privacy, not model compression. A small model can still memorize training
        examples if trained without DP. The privacy guarantee requires adding calibrated noise
        during training — model size alone cannot provide a mathematical bound on information leakage.
        The simulator confirms: {_mi_pct:.1f}% MI accuracy without DP, regardless of model size.
        """), kind="warn")
    else:
        _feedback = mo.callout(mo.md(f"""
        **Not quite.** Membership inference attacks require only the trained model — not the
        original training data. The adversary queries the model with candidate records and
        observes whether the model's confidence is systematically higher for training members
        than non-members. This works because the model's weights "remember" the specific examples
        it trained on, producing measurably different outputs. The simulator shows {_mi_pct:.1f}%
        accuracy from model queries alone, with no training data access required.
        """), kind="warn")

    mo.vstack([
        mo.Html("""
        <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                    text-transform: uppercase; letter-spacing: 0.12em; margin: 16px 0 8px 0;">
            Prediction vs. Reality
        </div>
        """),
        _feedback,
    ])
    return


# ─── ACT I: MATH PEEK ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
        **Membership Inference Advantage Bound (Yeom et al. 2018)**

        For a model with training accuracy $A_{train}$ and test accuracy $A_{test}$:
        $$\\text{Advantage}_{MI} \\leq A_{train} - A_{test}$$

        The generalization gap is a *direct upper bound* on how much better than random
        an adversary can perform. A 30% gap → at most 80% MI accuracy.

        ---

        **Differential Privacy Definition (ε-δ formulation)**

        A randomized algorithm $\\mathcal{A}$ satisfies $(\\epsilon, \\delta)$-DP if for all
        adjacent datasets $D, D'$ differing in one record, and all output sets $S$:
        $$\\Pr[\\mathcal{A}(D) \\in S] \\leq e^{\\epsilon} \\cdot \\Pr[\\mathcal{A}(D') \\in S] + \\delta$$

        - **ε (epsilon)**: privacy budget — smaller = stronger guarantee
        - **δ (delta)**: probability the guarantee fails catastrophically; must be $< 1/n^2$

        ---

        **Gaussian Mechanism Noise Requirement (DP-SGD)**

        To achieve $(\\epsilon, \\delta)$-DP with gradient clipping norm $C$:
        $$\\sigma \\geq \\frac{C \\cdot \\sqrt{2 \\ln(1.25/\\delta)}}{\\epsilon}$$

        At ε = 1, δ = 10⁻⁵, C = 1.0: σ ≥ 4.8 — noise nearly 5× larger than the signal.
        This is what limits membership inference to near-random: individual gradients are
        overwhelmed by noise before they can influence the weights.
        """)
    })
    return


# ─── ACT I: REFLECTION ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflection = mo.ui.radio(
        options={
            "A) Larger models have more parameters to exploit as an attack surface": "A",
            "B) The generalization gap quantifies how much the model memorized training examples vs learned general patterns": "B",
            "C) Test accuracy is used to directly train the membership inference classifier": "C",
            "D) Larger models encrypt individual training records less efficiently": "D",
        },
        label="""**Reflection:** Why does a larger generalization gap increase membership inference risk?

(The Yeom bound establishes the inequality — this question asks for the *causal mechanism*.)""",
    )
    act1_reflection
    return (act1_reflection,)


@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )

    _sel = act1_reflection.value[0]
    if _sel == "B":
        mo.callout(mo.md("""
        **Correct.** The generalization gap measures how much better the model performs on
        training data than on held-out test data — and that difference is precisely the
        "fingerprint" that membership inference exploits. A model with a large gap has
        learned training-data-specific patterns (memorization) rather than general
        distributional patterns (generalization). The adversary exploits this: a model
        that memorized your record will respond to queries about you with measurably
        higher confidence than a model that only learned general patterns. DP training
        prevents memorization by forcing the model to learn only what survives the noise.
        """), kind="success")
    elif _sel == "A":
        mo.callout(mo.md("""
        **Not quite.** The number of parameters is not directly the attack surface —
        it is the generalization gap that is exploitable. A large model with strong
        DP training can have a negligible generalization gap and near-random MI accuracy.
        A small model trained without DP on a small dataset can exhibit a large gap and
        high MI accuracy. The causal chain is: more parameters → easier overfitting →
        larger generalization gap → higher MI risk. The gap, not the parameter count, is
        the direct vulnerability measure.
        """), kind="warn")
    elif _sel == "C":
        mo.callout(mo.md("""
        **Not quite.** Test accuracy measures generalization — it tells us how much the
        model has memorized training examples versus learned general patterns. The MI
        attack itself works by querying the target model with candidate records and
        observing confidence scores, not by using test accuracy as training signal.
        The gap (train - test) is the bound on advantage, not the training data for
        the attack.
        """), kind="warn")
    else:
        mo.callout(mo.md("""
        **Not quite.** Neural networks do not encrypt data — they compress statistical
        patterns from training data into weights. The vulnerability is not about encryption
        efficiency but about memorization: a model trained without DP learns to
        distinguish its training examples from unseen data, which is precisely what
        membership inference exploits. "Encryption" is not a meaningful frame here;
        the correct concept is the generalization gap.
        """), kind="warn")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _act2_num      = "II"
    _act2_color    = COLORS["OrangeLine"]
    _act2_title    = "DP Parameter Selection"
    _act2_duration = "20&ndash;25 min"
    _act2_why      = ("Act I proved that the privacy gap is real and measurable. "
                      "Now discover the engineering cost: every &epsilon; you choose "
                      "carries a noise scale &sigma; that erodes model accuracy, and "
                      "every multi-tenant deployment in the cloud adds a 15% throughput "
                      "tax that compounds with that noise. "
                      "Your constraint is HIPAA &epsilon;&thinsp;&le;&thinsp;1 and a 90% accuracy floor &mdash; both simultaneously.")
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
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["RedLine"]
    _bg    = COLORS["RedL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · ML Platform Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We need to deploy DP training on our medical diagnosis model. Legal has
            reviewed the HIPAA safe harbor framework and requires epsilon ≤ 1 for any model
            trained on protected health information. Our baseline model achieves 94% accuracy
            on our clinical evaluation set. When we ran DP-SGD at epsilon = 1 with default
            hyperparameters, accuracy dropped to 86%. The clinical team says 90% is the
            absolute minimum viable accuracy — below that, the model produces more harm
            than benefit. We have 1 million training records currently. Design the DP
            configuration that satisfies both the regulatory constraint and the clinical
            threshold simultaneously."
        </div>
    </div>
    """)
    return


# ─── ACT II: SCENARIO SETUP ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("""
        ## The DP-SGD Configuration Space

        The DP-SGD algorithm has three primary hyperparameters that jointly determine
        the privacy guarantee and accuracy penalty:

        **Clipping norm C**: Bounds the per-sample gradient norm before noise injection.
        Smaller C → smaller noise scale needed for same ε (better accuracy). But if C
        is too small, useful gradient signal is lost (truncation bias).

        **Noise multiplier σ_mult**: Controls noise scale as σ = σ_mult × C. Higher
        σ_mult → stronger privacy (smaller effective ε per step) but more accuracy loss.

        **Dataset size N**: More training examples dilute the noise per-parameter.
        The critical insight: **10× more data can compensate for DP noise** — this
        is why the standard approach is public pre-training + private fine-tuning on
        large datasets.

        The Gaussian mechanism requirement for (ε, δ)-DP:
        $$\\sigma \\geq \\frac{C \\cdot \\sqrt{2 \\ln(1.25/\\delta)}}{\\epsilon}$$

        Your design space: satisfy ε ≤ 1 (HIPAA) while achieving accuracy ≥ 90% (clinical).
        """),
    ])
    return


# ─── ACT II: PREDICTION LOCK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) You must choose — ε=1 and 86% accuracy are mathematically fixed by the algorithm": "A",
            "B) Use ε=1, increase training data 10× — more data compensates for DP noise": "B",
            "C) Use ε=2, which is close enough to ε=1 for practical HIPAA compliance": "C",
            "D) Switch to federated learning — it provides DP guarantees without accuracy loss": "D",
        },
        label="""**Commit to your prediction before configuring the instruments.**

At ε=1 with default DP-SGD settings, accuracy is 86% — below the 90% clinical threshold.
What is the primary mechanism for achieving ε=1 AND accuracy ≥ 90%?""",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.vstack([
            act2_pred,
            mo.callout(
                mo.md("Select your prediction to continue. Lock your answer before running the configurator."),
                kind="warn",
            ),
        ])
    )
    mo.callout(
        mo.md(f"**Prediction locked:** Option {act2_pred.value[0]}. Configure the DP-SGD parameters below."),
        kind="info",
    )
    return


# ─── ACT II: INSTRUMENTS ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("## DP-SGD Configurator")
    return


@app.cell(hide_code=True)
def _(mo):
    dp_epsilon = mo.ui.slider(
        start=0.1, stop=10.0, value=1.0, step=0.1,
        label="Privacy budget ε (epsilon)",
    )
    dp_delta_exp = mo.ui.slider(
        start=-8, stop=-4, value=-6, step=1,
        label="Failure probability δ exponent (10^x)",
    )
    clipping_norm = mo.ui.slider(
        start=0.1, stop=5.0, value=1.0, step=0.1,
        label="Gradient clipping norm C",
    )
    noise_multiplier = mo.ui.slider(
        start=0.1, stop=5.0, value=1.1, step=0.1,
        label="Noise multiplier (σ / C)",
    )
    training_steps_k = mo.ui.slider(
        start=1, stop=100, value=10, step=1,
        label="Training steps (thousands)",
    )
    dataset_size_k = mo.ui.slider(
        start=10, stop=10000, value=1000, step=10,
        label="Dataset size (thousands of records)",
    )
    context_medical = mo.ui.checkbox(
        value=True,
        label="Medical data (HIPAA applies: ε must be ≤ 1.0)",
    )
    mo.vstack([
        mo.hstack([dp_epsilon, dp_delta_exp], justify="start", gap=2),
        mo.hstack([clipping_norm, noise_multiplier], justify="start", gap=2),
        mo.hstack([training_steps_k, dataset_size_k], justify="start", gap=2),
        context_medical,
    ])
    return (
        dp_epsilon, dp_delta_exp, clipping_norm, noise_multiplier,
        training_steps_k, dataset_size_k, context_medical,
    )


@app.cell(hide_code=True)
def _(
    mo, go, np, math,
    dp_epsilon, dp_delta_exp, clipping_norm, noise_multiplier,
    training_steps_k, dataset_size_k, context_medical,
    apply_plotly_theme, COLORS,
    HIPAA_MAX_EPSILON, DP_MEDICAL_BASELINE_ACC, DP_MEDICAL_EPS1_ACC,
    DP_CLINICAL_MIN_ACC, CLOUD_ISOLATION_OVERHEAD, context_toggle,
):
    # ── Simulation Physics ───────────────────────────────────────────────────
    # DP-SGD Gaussian mechanism noise requirement:
    #   σ ≥ C × √(2 ln(1.25/δ)) / ε
    # Source: @sec-security-privacy-dp-mathematical-foundations-7e4a

    _eps        = dp_epsilon.value
    _delta      = 10 ** dp_delta_exp.value
    _C          = clipping_norm.value
    _mult       = noise_multiplier.value
    _steps_k    = training_steps_k.value
    _n_k        = dataset_size_k.value
    _is_medical = context_medical.value
    _is_cloud   = context_toggle.value == "cloud"

    # Required noise for (ε, δ)-DP guarantee
    _sigma_required = (_C * math.sqrt(2 * math.log(1.25 / _delta))) / _eps
    _sigma_actual   = _mult * _C

    # Check if actual noise satisfies the DP requirement
    _dp_satisfied = _sigma_actual >= _sigma_required

    # HIPAA compliance check
    _hipaa_violated = _is_medical and (_eps > HIPAA_MAX_EPSILON)

    # Accuracy model:
    # Baseline accuracy without DP = 94% (chapter scenario)
    # At ε=1, default hyperparams: 86% accuracy (chapter scenario)
    # Key insight: more data (larger N) reduces the per-parameter noise impact
    #   accuracy_penalty ∝ σ / √N  (DP noise scales inversely with √N)
    # Clipping norm tuning can recover 1-3% accuracy at the cost of slightly
    # looser effective ε per step.
    _n_ref     = 1000.0  # reference dataset size (1M records = 1000k)
    _data_factor = math.sqrt(_n_k / _n_ref)  # √N scaling from noise dilution

    # Noise ratio: how much noise relative to the signal
    _noise_ratio = _sigma_actual / _C

    # Base accuracy penalty at ε=1 with N=1M, default C
    # Interpolated from chapter: 94% baseline → 86% at ε=1 (default σ_mult≈1.1)
    _penalty_at_eps1_default = DP_MEDICAL_BASELINE_ACC - DP_MEDICAL_EPS1_ACC  # 0.08

    # Effective epsilon per step (using moments accountant approximation)
    # ε_step ∝ 1 / (σ_mult × √T), where T = steps
    _T = _steps_k * 1000
    _batch_frac = 1.0 / (_n_k * 1000.0 / 32.0)  # approx batch size 32
    # Simplified RDP → (ε, δ)-DP conversion (tight approximation)
    _eps_eff = _eps  # slider directly controls ε for clarity

    # Accuracy penalty scales with noise^2 / (N_data × steps) heuristic
    # More precisely: noise dominates when σ >> gradient signal; more data averages it out
    _noise_impact = (_noise_ratio ** 1.5) / (_data_factor * math.sqrt(_steps_k / 10.0))
    _accuracy_penalty = _penalty_at_eps1_default * _noise_impact * (1.0 / _eps) ** 0.5

    # Apply cloud isolation overhead if cloud context
    _cloud_penalty = CLOUD_ISOLATION_OVERHEAD * 0.02 if _is_cloud else 0.0

    _model_accuracy = min(
        DP_MEDICAL_BASELINE_ACC,
        max(0.50, DP_MEDICAL_BASELINE_ACC - _accuracy_penalty - _cloud_penalty)
    )
    _model_acc_pct = _model_accuracy * 100

    # Clinical viability
    _clinical_viable = _model_accuracy >= DP_CLINICAL_MIN_ACC

    # Membership inference risk with DP
    _mi_risk = 0.50 + max(0, 0.005 * _eps)

    # ── Color coding ─────────────────────────────────────────────────────────
    _acc_color  = COLORS["GreenLine"] if _model_accuracy >= DP_CLINICAL_MIN_ACC else \
                  COLORS["OrangeLine"] if _model_accuracy >= 0.87 else COLORS["RedLine"]
    _eps_color  = COLORS["GreenLine"] if _eps <= HIPAA_MAX_EPSILON else COLORS["RedLine"]
    _mi_color   = COLORS["GreenLine"] if _mi_risk <= 0.55 else \
                  COLORS["OrangeLine"] if _mi_risk <= 0.65 else COLORS["RedLine"]

    # ── Epsilon-accuracy Pareto curve ─────────────────────────────────────────
    _eps_range  = np.linspace(0.1, 10.0, 50)
    _acc_curve  = []
    for _e in _eps_range:
        _pen_e   = _penalty_at_eps1_default * (_noise_ratio ** 1.5) / (_data_factor * math.sqrt(_steps_k / 10.0)) * (1.0 / _e) ** 0.5
        _a_e     = min(DP_MEDICAL_BASELINE_ACC, max(0.5, DP_MEDICAL_BASELINE_ACC - _pen_e))
        _acc_curve.append(_a_e * 100)

    _fig2 = go.Figure()

    # ε-accuracy curve
    _fig2.add_trace(go.Scatter(
        x=_eps_range, y=_acc_curve,
        mode="lines", name="Accuracy vs ε",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))

    # Current operating point
    _fig2.add_trace(go.Scatter(
        x=[_eps], y=[_model_acc_pct],
        mode="markers", name="Current config",
        marker=dict(
            color=_acc_color, size=14, symbol="diamond",
            line=dict(color="white", width=2)
        ),
    ))

    # Clinical threshold line
    _fig2.add_hline(
        y=DP_CLINICAL_MIN_ACC * 100,
        line_dash="dash", line_color=COLORS["OrangeLine"], line_width=1.5,
        annotation_text=f"Clinical minimum ({DP_CLINICAL_MIN_ACC*100:.0f}%)",
        annotation_position="top right",
    )

    # HIPAA epsilon limit
    _fig2.add_vline(
        x=HIPAA_MAX_EPSILON,
        line_dash="dash", line_color=COLORS["RedLine"], line_width=1.5,
        annotation_text=f"HIPAA limit (ε={HIPAA_MAX_EPSILON})",
        annotation_position="top left",
    )

    # Feasible zone shading
    _fig2.add_shape(
        type="rect",
        x0=0.1, x1=HIPAA_MAX_EPSILON,
        y0=DP_CLINICAL_MIN_ACC * 100, y1=100,
        fillcolor="rgba(0,143,69,0.08)", line_width=0,
    )

    _fig2.update_layout(
        height=360,
        xaxis=dict(title="Privacy budget ε", range=[0, 10.5]),
        yaxis=dict(title="Model accuracy (%)", range=[70, 100]),
        margin=dict(t=20, b=40),
        showlegend=True,
        legend=dict(x=0.65, y=0.15),
    )
    apply_plotly_theme(_fig2)

    # ── Noise added per update display ────────────────────────────────────────
    _noise_per_update_str = f"σ = {_sigma_actual:.2f} (required: {_sigma_required:.2f})"

    # ── Output ───────────────────────────────────────────────────────────────
    mo.vstack([
        mo.md(f"""
        ### Physics

        ```
        DP-SGD Noise Requirement:  σ ≥ C × √(2 ln(1.25/δ)) / ε
                                   σ ≥ {_C:.1f} × √(2 × ln(1.25 / {_delta:.0e})) / {_eps:.1f}
                                   σ ≥ {_sigma_required:.3f}

        Actual noise:              σ = {_mult:.1f} × C = {_mult:.1f} × {_C:.1f} = {_sigma_actual:.3f}
        DP guarantee satisfied:    {'YES' if _dp_satisfied else 'NO — increase noise multiplier'}

        Dataset size:              N = {_n_k * 1000:,.0f} records
        Data scaling factor:       √(N / N_ref) = √({_n_k}/{_n_ref}) = {_data_factor:.3f}
        Estimated accuracy:        {_model_acc_pct:.1f}%
        Membership inference risk: {_mi_risk*100:.1f}%
        ```
        """),

        mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin: 16px 0;">
            <div style="padding: 18px; border: 1px solid #ddd; border-radius: 8px;
                        width: 160px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.8rem; font-weight: 600;">Privacy Budget</div>
                <div style="font-size: 2rem; font-weight: 900; color: {_eps_color};">ε={_eps:.1f}</div>
                <div style="color: #94a3b8; font-size: 0.72rem;">
                    {'HIPAA OK' if not _hipaa_violated else 'HIPAA FAIL'}
                </div>
            </div>
            <div style="padding: 18px; border: 1px solid #ddd; border-radius: 8px;
                        width: 160px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.8rem; font-weight: 600;">Model Accuracy</div>
                <div style="font-size: 2rem; font-weight: 900; color: {_acc_color};">
                    {_model_acc_pct:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem;">
                    {'Clinically viable' if _clinical_viable else 'Below 90% threshold'}
                </div>
            </div>
            <div style="padding: 18px; border: 1px solid #ddd; border-radius: 8px;
                        width: 160px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.8rem; font-weight: 600;">MI Risk</div>
                <div style="font-size: 2rem; font-weight: 900; color: {_mi_color};">
                    {_mi_risk*100:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem;">MI Accuracy</div>
            </div>
            <div style="padding: 18px; border: 1px solid #ddd; border-radius: 8px;
                        width: 160px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.8rem; font-weight: 600;">Noise per Step</div>
                <div style="font-size: 1.3rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    σ={_sigma_actual:.2f}
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem;">
                    Required: {_sigma_required:.2f}
                </div>
            </div>
            <div style="padding: 18px; border: 1px solid #ddd; border-radius: 8px;
                        width: 160px; text-align: center; background: white;">
                <div style="color: #666; font-size: 0.8rem; font-weight: 600;">
                    {'Cloud Overhead' if _is_cloud else 'On-Prem Overhead'}
                </div>
                <div style="font-size: 1.6rem; font-weight: 900;
                            color: {'#CB202D' if _is_cloud else '#008F45'};">
                    {'15%' if _is_cloud else '0%'}
                </div>
                <div style="color: #94a3b8; font-size: 0.72rem;">Isolation tax</div>
            </div>
        </div>
        """),

        mo.as_html(_fig2),
    ])
    return (
        _eps, _delta, _C, _mult, _n_k, _model_accuracy, _model_acc_pct,
        _clinical_viable, _hipaa_violated, _mi_risk, _dp_satisfied,
        _sigma_required, _sigma_actual, _is_medical, _is_cloud,
    )


# ─── ACT II: FAILURE STATES ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _hipaa_violated, _clinical_viable, _eps, _model_acc_pct, _dp_satisfied):
    _alerts = []

    if _hipaa_violated:
        _alerts.append(mo.callout(mo.md(
            f"**HIPAA compliance violated:** ε = {_eps:.1f} exceeds the regulatory limit of "
            f"ε = 1.0 for medical data. At this privacy budget, membership inference accuracy "
            f"climbs to ~{50 + _eps * 0.5:.0f}% — above the near-random threshold required "
            f"for regulatory protection. Reduce noise multiplier or increase dataset size to "
            f"achieve ε ≤ 1 while meeting the accuracy requirement."
        ), kind="danger"))

    if not _clinical_viable:
        _alerts.append(mo.callout(mo.md(
            f"**Clinical viability threshold not met:** Current accuracy {_model_acc_pct:.1f}% "
            f"is below the required 90% minimum. This configuration produces a model that "
            f"clinical evaluation deems more harmful than beneficial. Increase dataset size "
            f"(more data dilutes DP noise via √N scaling) or tune the clipping norm to recover "
            f"accuracy without weakening the privacy guarantee."
        ), kind="warn"))

    if not _dp_satisfied:
        _alerts.append(mo.callout(mo.md(
            f"**DP guarantee not satisfied:** The actual noise σ = {_sigma_actual:.3f} is below "
            f"the required σ ≥ {_sigma_required:.3f}. This configuration does NOT provide the "
            f"claimed (ε={_eps:.1f}, δ)-DP guarantee. Increase the noise multiplier to satisfy "
            f"the Gaussian mechanism requirement."
        ), kind="danger"))

    if not _alerts and _hipaa_violated is False and _clinical_viable and _dp_satisfied:
        _alerts.append(mo.callout(mo.md(
            f"**Configuration satisfies all constraints.** "
            f"ε = {_eps:.1f} ≤ 1.0 (HIPAA), accuracy = {_model_acc_pct:.1f}% ≥ 90% (clinical), "
            f"and the Gaussian mechanism requirement is satisfied. "
            f"Membership inference is limited to {50 + _eps * 0.5:.0f}% — near-random."
        ), kind="success"))

    mo.vstack(_alerts) if _alerts else mo.md("")
    return


# ─── ACT II: PREDICTION-VS-REALITY OVERLAY ───────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, _model_acc_pct, _clinical_viable):
    _sel2 = act2_pred.value[0] if act2_pred.value else None

    if _sel2 is None:
        mo.stop(True, mo.callout(mo.md("Complete Act II prediction above first."), kind="warn"))

    if _sel2 == "B":
        _f2 = mo.callout(mo.md(f"""
        **Correct.** The standard approach to recovering accuracy under DP is to increase
        training data. Because DP noise dilutes with dataset size via the √N scaling —
        more records means each individual contributes a smaller fraction of the gradient
        signal, so the noise-to-signal ratio falls. At 10 million records (10× the baseline
        1 million), the accuracy under ε = 1 can recover from 86% to above the 90% clinical
        threshold. The simulator confirms: with N ≥ 10,000k and ε = 1, accuracy reaches
        {_model_acc_pct:.1f}%. This is why the dominant production pattern is pre-training
        on large public datasets (no privacy cost) followed by private fine-tuning on a
        small sensitive dataset — the public pre-training provides strong feature representations
        that require less DP fine-tuning to reach target accuracy.
        """), kind="success")
    elif _sel2 == "A":
        _f2 = mo.callout(mo.md(f"""
        **Not quite.** ε = 1 and 86% accuracy are not fixed by the algorithm — they are
        fixed by one specific combination of hyperparameters and dataset size. The simulator
        shows that with more training data (higher N), the √N scaling of DP noise allows
        recovery to {_model_acc_pct:.1f}%. Similarly, tuning the clipping norm C and noise
        multiplier can shift the accuracy-privacy Pareto frontier. The constraints are real
        (ε ≤ 1 is non-negotiable for HIPAA; accuracy ≥ 90% is non-negotiable for clinical
        viability), but the parameter space to satisfy both simultaneously is non-empty.
        """), kind="warn")
    elif _sel2 == "C":
        _f2 = mo.callout(mo.md(f"""
        **Not quite.** ε = 2 is not "close enough" to ε = 1. The privacy guarantee degrades
        exponentially with ε: the maximum probability ratio between adjacent dataset outputs
        is e^ε, so e^2 ≈ 7.4 vs e^1 ≈ 2.7 — nearly 3× weaker. For HIPAA, the regulatory
        interpretation is a hard threshold, not a continuous scale. The correct path is to
        satisfy ε ≤ 1 by adjusting data size and hyperparameters, not by weakening the
        privacy budget. The simulator shows that ε = 2 in a medical context triggers the
        HIPAA compliance failure state.
        """), kind="warn")
    else:
        _f2 = mo.callout(mo.md(f"""
        **Not quite.** Federated learning is a valid privacy-enhancing architecture but
        it does not provide DP guarantees on its own — model updates in federated learning
        can still leak individual training data via gradient inversion attacks. To provide
        formal DP guarantees in federated settings, DP-SGD must still be applied (typically
        with secure aggregation). The accuracy cost does not disappear; it may be slightly
        lower because local models train on more IID data per participant. The correct
        primary mechanism for meeting the ε = 1 constraint while recovering accuracy is
        increasing the dataset size.
        """), kind="warn")

    mo.vstack([
        mo.Html("""
        <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                    text-transform: uppercase; letter-spacing: 0.12em; margin: 16px 0 8px 0;">
            Prediction vs. Reality
        </div>
        """),
        _f2,
    ])
    return


# ─── ACT II: MATH PEEK ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
        **DP-SGD Noise Requirement (Gaussian Mechanism)**

        To achieve $(\\epsilon, \\delta)$-DP with gradient clipping norm $C$:
        $$\\sigma \\geq \\frac{C \\cdot \\sqrt{2 \\ln(1.25/\\delta)}}{\\epsilon}$$

        At ε = 1, δ = 10⁻⁶, C = 1.0: σ ≥ 5.26 — noise more than 5× larger than the gradient.

        ---

        **Composition Theorem (Sequential Queries)**

        If mechanism $\\mathcal{M}$ is $(\\epsilon, \\delta)$-DP, and we apply it $k$ times,
        the composed mechanism is $(k\\epsilon, k\\delta)$-DP under naive composition.
        For tight bounds, use Rényi DP accounting:

        $$\\text{RDP-DP conversion: } (\\epsilon, \\delta) \\text{ where } \\delta = e^{-\\lambda(\\epsilon_R - \\epsilon)/(\\lambda - 1)}$$

        This is why the total privacy budget must be tracked across all training steps.

        ---

        **Data Scaling Intuition (Why More Data Helps)**

        DP adds noise of magnitude σ to the gradient sum of N samples.
        Per-sample gradient estimate noise: σ / N.
        As N increases, noise-to-signal ratio ∝ σ / (signal × √N).
        Doubling N reduces noise impact by √2 ≈ 1.41×.
        10× more data → ~3.16× lower effective noise impact.
        This is why public pre-training + private fine-tuning is the dominant production pattern.

        ---

        **Rényi Differential Privacy (Tight Composition)**

        $\\mathcal{M}$ is $(\\alpha, \\epsilon_R)$-RDP if:
        $$D_\\alpha(\\mathcal{M}(D) \\| \\mathcal{M}(D')) = \\frac{1}{\\alpha - 1} \\ln \\mathbb{E}\\left[\\left(\\frac{p(O|D)}{p(O|D')}\\right)^{\\alpha}\\right] \\leq \\epsilon_R$$

        RDP composes additively: $k$ applications of $(\\alpha, \\epsilon_R)$-RDP gives
        $(\\alpha, k\\epsilon_R)$-RDP. Converting to $(\\epsilon, \\delta)$-DP gives tighter
        bounds than naive composition, enabling more training steps for the same privacy budget.
        """)
    })
    return


# ─── ACT II: REFLECTION ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("---")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) Cloud providers cannot access encrypted model weights stored in their systems": "A",
            "B) On-prem provides hardware isolation — no hypervisor-level side-channel attacks or multi-tenant data leakage": "B",
            "C) On-prem hardware runs the same operations physically faster, reducing attack surface": "C",
            "D) Cloud providers are legally required to share data with government regulators": "D",
        },
        label="""**Reflection:** Why does on-premises deployment offer stronger privacy guarantees than cloud
for a model trained with DP?

(DP addresses the training-time privacy guarantee. This question asks about deployment-time threats.)""",
    )
    act2_reflection
    return (act2_reflection,)


@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(mo.md("Select your answer to see the explanation."), kind="warn"),
    )

    _sel_r2 = act2_reflection.value[0]
    if _sel_r2 == "B":
        mo.callout(mo.md("""
        **Correct.** On-premises deployment eliminates the hypervisor layer between the
        application and the hardware. In cloud environments, the hypervisor that enables
        multi-tenancy creates a documented attack surface: Spectre and Meltdown class
        vulnerabilities allow one tenant to read another's memory through speculative
        execution; timing side-channels can leak cryptographic keys across VM boundaries;
        and shared GPU scheduling creates memory residue risks if MIG partitioning is not
        enforced. On-prem hardware is dedicated — there is no adjacent tenant from whom
        to leak. This physical isolation is why HIPAA-regulated healthcare providers often
        require on-prem deployment for models processing PHI, even when DP provides
        training-time guarantees. The isolation tax of 0% (on-prem) vs 15% (cloud MIG)
        is the quantitative signature of this difference.
        """), kind="success")
    elif _sel_r2 == "A":
        mo.callout(mo.md("""
        **Not quite.** Key management is the critical issue, not encryption itself. Cloud
        providers can access data when they hold the encryption keys — and in most
        cloud deployments, the provider manages the hardware security modules (HSMs)
        that store those keys. Encryption at rest protects against external attackers
        but not against a provider with key access. On-prem deployment puts the HSMs
        under the organization's physical control, eliminating the key escrow problem.
        But the more direct advantage of on-prem is hardware isolation: no hypervisor,
        no multi-tenant side channels, no adjacent tenant risk.
        """), kind="warn")
    elif _sel_r2 == "C":
        mo.callout(mo.md("""
        **Not quite.** Physical speed is irrelevant to privacy guarantees. A faster GPU
        does not reduce attack surface — it may actually increase throughput for an
        adversary performing model extraction queries. The privacy advantage of on-prem
        is architectural isolation: dedicated hardware means no multi-tenant GPU sharing,
        no hypervisor layer, and no cross-tenant memory residue from the isolation
        overhead (the 15% MIG penalty exists precisely because isolation *is* preventing
        cross-tenant leakage, at a compute cost).
        """), kind="warn")
    else:
        mo.callout(mo.md("""
        **Not quite.** On-prem operators are equally subject to legal data access
        requests (subpoenas, national security letters) as cloud providers — the legal
        jurisdiction determines disclosure requirements, not the infrastructure model.
        Some organizations use on-prem specifically because they believe they have more
        control over legal process, but this is not the *technical* privacy advantage.
        The technical advantage is hardware isolation: no hypervisor, no multi-tenant
        side channels, no shared GPU memory residue. The isolation tax (0% on-prem
        vs 15% cloud) is the engineering signature of the difference.
        """), kind="warn")
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
                    <strong>1. The privacy gap is measurable, not assumed.</strong>
                    Standard training without DP allows membership inference accuracy bounded by the
                    generalization gap (Yeom bound: Advantage &le; A_train &minus; A_test).
                    A 94% MI accuracy signals deep memorization; DP training at &epsilon;&thinsp;&le;&thinsp;1
                    collapses that to 50&ndash;55% &mdash; statistically indistinguishable from random guessing.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. DP noise scales as &sigma; = C&radic;(2&thinsp;ln(1.25/&delta;))/&epsilon;.</strong>
                    The accuracy cost at fixed &epsilon; is proportional to &sigma;/&radic;N.
                    Increasing dataset size 10&times; recovers &radic;10 &asymp; 3.16&times; noise tolerance &mdash;
                    scale is the correct engineering response to the privacy-utility tradeoff,
                    not weakening &epsilon;.
                </div>
                <div>
                    <strong>3. Infrastructure isolation is a throughput tax, not a free feature.</strong>
                    Cloud MIG partitioning costs exactly 15% throughput (1,000 &rarr; 850 tokens/sec)
                    to prevent cross-tenant side-channel attacks; on-premises hardware isolation costs 0%.
                    The deployment context choice has a direct, quantifiable engineering consequence.
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
                    <strong>Lab 14: The Adversarial Wall</strong> &mdash; This lab quantified passive
                    inference attacks on a trained model. The next lab asks: what happens when the
                    attacker actively perturbs inputs to fool the model at inference time, and what
                    is the training cost of defending against them?
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
                    <strong>Read:</strong> @sec-security-privacy-differential-privacy-8c2b for the
                    full Gaussian mechanism derivation and DP-SGD algorithm.<br/>
                    <strong>Build:</strong> The MultiTenantIsolation LEGO cell in the chapter
                    demonstrates the 15% MIG throughput tax via explicit timing benchmarks.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. DP noise scales as sigma = C * sqrt(2 ln(1.25/delta)) / epsilon. At epsilon=1 with N=1,000 people, per-person error is $200. For N=100, the error rises to $2,000. What does this 1/sqrt(N) scaling imply about the minimum dataset size for DP to be practical?
2. MIG partitioning costs exactly 15% throughput (1,000 to 850 tokens/sec) to prevent cross-tenant side-channel attacks. On-premises hardware isolation costs 0%. What does this quantifiable difference reveal about how deployment context choices have direct engineering consequences?
3. Standard training allows membership inference accuracy bounded by the generalization gap (Yeom bound). DP training at epsilon <= 1 collapses MI accuracy to 50-55%. Why is increasing dataset size 10x (recovering sqrt(10) noise tolerance) the correct response rather than weakening epsilon?

**You're ready to move on if you can:**
- Calculate the per-record accuracy cost of differential privacy for a given epsilon, sensitivity, and dataset size
- Explain why infrastructure isolation is a throughput tax with quantifiable cost, not a free feature
- Determine when DP is feasible based on dataset size and the required accuracy tolerance
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD ──────────────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle, act1_pred, act2_pred,
    _eps, _delta, _model_accuracy, _mi_risk,
    _hipaa_violated, _clinical_viable,
    _is_medical, _is_cloud,
, decision_input, decision_ui):
    _context   = context_toggle.value
    _a1_val    = act1_pred.value
    _a2_val    = act2_pred.value

    _a1_correct = (_a1_val is not None and _a1_val.startswith("C"))
    _a2_decision = "increase_data" if (_a2_val and _a2_val.startswith("B")) else \
                   "reduce_epsilon" if (_a2_val and _a2_val.startswith("C")) else \
                   "federated" if (_a2_val and _a2_val.startswith("D")) else \
                   "accept_tradeoff"

    _constraint_hit = _hipaa_violated or (not _clinical_viable)

    ledger.save(chapter="v2_13", design={
        "context":                 _context,
        "dp_epsilon":              float(_eps),
        "dp_delta":                float(_delta),
        "model_accuracy":          float(_model_accuracy),
        "membership_inference_risk": float(_mi_risk),
        "hipaa_compliant":         not _hipaa_violated,
        "act1_prediction":         _a1_val or "none",
        "act1_correct":            bool(_a1_correct),
        "act2_result":             float(_model_accuracy),
        "act2_decision":           _a2_decision,
        "constraint_hit":          bool(_constraint_hit),
        "student_justification": str(decision_input.value),
        "clinical_viable":         bool(_clinical_viable),
    })

    # ── HUD Footer ────────────────────────────────────────────────────────────
    _c  = COLORS
    _badge_hipaa = (
        f'<span class="badge badge-ok">HIPAA: OK</span>'
        if not _hipaa_violated else
        f'<span class="badge badge-fail">HIPAA: VIOLATED</span>'
    )
    _badge_clinical = (
        f'<span class="badge badge-ok">Clinical: Viable</span>'
        if _clinical_viable else
        f'<span class="badge badge-fail">Clinical: Below Threshold</span>'
    )
    _badge_ctx = (
        f'<span class="badge badge-info">On-Premises</span>'
        if _context == "on_prem" else
        f'<span class="badge badge-warn">Cloud (+15% isolation tax)</span>'
    )

    mo.vstack([
        decision_ui,
        mo.Html(f"""
        <div class="lab-hud">
            <span>
                <span class="hud-label">CHAPTER</span>&nbsp;
                <span class="hud-value">v2_13 · Security &amp; Privacy</span>
            </span>
            <span>
                <span class="hud-label">ε</span>&nbsp;
                <span class="hud-value">{_eps:.1f}</span>
            </span>
            <span>
                <span class="hud-label">ACCURACY</span>&nbsp;
                <span class="hud-value">{_model_accuracy*100:.1f}%</span>
            </span>
            <span>
                <span class="hud-label">MI RISK</span>&nbsp;
                <span class="hud-value">{_mi_risk*100:.1f}%</span>
            </span>
            <span>
                <span class="hud-label">ACT I</span>&nbsp;
                <span class="{'hud-active' if _a1_correct else 'hud-none'}">
                    {'CORRECT' if _a1_correct else 'REVIEW'}
                </span>
            </span>
            <span>
                <span class="hud-label">CONSTRAINTS</span>&nbsp;
                <span class="{'hud-none' if _constraint_hit else 'hud-active'}">
                    {'VIOLATED' if _constraint_hit else 'SATISFIED'}
                </span>
            </span>
        </div>
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px;">
            {_badge_hipaa}
            {_badge_clinical}
            {_badge_ctx}
        </div>
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
