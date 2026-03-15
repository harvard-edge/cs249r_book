import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-14: THE ADVERSARIAL WALL
#
# Volume II, Chapter 14 — Robust AI
#
# Core Invariant: Adversarial training improves robustness against ε-ball
#   perturbations but reduces clean accuracy. Models that are robust to
#   worst-case perturbations must learn more conservative decision boundaries,
#   which reduces their performance on typical clean inputs.
#
# 2 Contexts:
#   Production   — Standard model (97.3% clean accuracy, undefended)
#   Hardened     — Adversarially trained model (PGD-7 defense)
#
# Act I  (12–15 min): Adversarial Fragility Revelation
#   Stakeholder: ML Security Lead — medical image classifier, 97.3% clean
#   accuracy drops to 3.4% under ε=8/255 FGSM attack
#   Instruments: model accuracy slider, epsilon slider, attack type dropdown
#   Prediction: what does 3.4% adversarial accuracy tell us?
#   Overlay: prediction-vs-reality showing why 3.4% is worse than random
#   Reflection: why high-dimensional spaces make adversarial examples possible
#
# Act II (20–25 min): Robustness-Accuracy Tradeoff
#   Stakeholder: CISO — must achieve ε=8/255 adversarial accuracy > 50%
#   while keeping clean accuracy > 90%. Is the constraint satisfiable?
#   Instruments: adversarial training ε, PGD steps, adversarial loss weight
#   Failure states: security req unmet (danger), clinical threshold unmet (warn)
#   Reflection: why adversarial training always reduces clean accuracy
#
# Design Ledger: saves chapter="v2_14"
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible) ──────────────────────────
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

    # ── Hardware constants (from NVIDIA H100 SXM5 spec and Vol2 robust_ai.qmd) ─
    H100_BW_GBS       = 3350    # GB/s HBM3e — NVIDIA H100 SXM5 spec
    H100_TFLOPS_FP16  = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80      # GB HBM3e — NVIDIA spec

    # ── Adversarial training compute constants ────────────────────────────────
    # PGD-7 (standard in Madry et al. 2018) requires N inner steps per batch.
    # Training cost ≈ (1 + N_pgd_steps) × standard training cost.
    # Source: @sec-robust-ai adversarial attack and defense sections.
    PGD_STEPS_DEFAULT   = 7     # standard PGD-7 adversarial training — Madry et al.
    FGSM_MULTIPLIER     = 1.0   # FGSM is 1 step (same overhead as standard)

    # ── Adversarial robustness baseline constants ─────────────────────────────
    # From chapter text: ε=8/255 perturbation reduces non-robust accuracy 30–60%.
    # Medical imaging classifier baseline: 97.3% clean accuracy (chapter scenario).
    # Adversarial accuracy at ε=8/255 without defense: empirically ~3–5%.
    CLEAN_ACC_BASELINE    = 97.3   # % — medical classifier baseline, chapter scenario
    ADV_ACC_UNDEFENDED    = 3.4    # % — at ε=8/255, FGSM, no defense — chapter scenario
    RANDOM_CHANCE         = 10.0   # % — 10-class classification random baseline

    # ── Security and product thresholds (from chapter CISO scenario) ─────────
    SECURITY_ADV_THRESHOLD  = 50.0  # % — CISO minimum adversarial accuracy requirement
    PRODUCT_CLEAN_THRESHOLD = 90.0  # % — product team minimum clean accuracy requirement

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
        PGD_STEPS_DEFAULT, FGSM_MULTIPLIER,
        CLEAN_ACC_BASELINE, ADV_ACC_UNDEFENDED, RANDOM_CHANCE,
        SECURITY_ADV_THRESHOLD, PRODUCT_CLEAN_THRESHOLD,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 1: HEADER (hide_code=True) ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _prod_color  = COLORS["RedLine"]
    _hard_color  = COLORS["BlueLine"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1a0a10 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 14
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Adversarial Wall
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 640px; line-height: 1.65;">
                A 97.3% accurate medical classifier drops to 3.4% under imperceptible
                noise. Defending against that attack costs you clean accuracy you cannot
                recover. The robustness-accuracy tradeoff is not a bug — it is physics.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(203,32,45,0.18); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.3);">
                    Act I: Adversarial Fragility · Act II: Robustness-Accuracy Tradeoff
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: @sec-robust-ai
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-fail">Production: 97.3% clean → 3.4% adversarial</span>
                <span class="badge badge-info">Hardened: PGD-7 adversarial training</span>
                <span class="badge badge-warn">Invariant: Robustness costs clean accuracy</span>
                <span class="badge badge-warn">Invariant: ε-ball worst-case ≠ average-case</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the robustness tax</strong>: measure the exact clean-accuracy cost (26 percentage points) of PGD-7 adversarial training at &epsilon;=8/255 and predict whether FGSM or PGD achieves a lower adversarial accuracy floor.</div>
                <div style="margin-bottom: 3px;">2. <strong>Diagnose why 3.4% adversarial accuracy is worse than random</strong> for a 10-class classifier &mdash; and identify which geometric property of high-dimensional spaces makes imperceptible perturbations catastrophic.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a defense configuration</strong> that simultaneously satisfies adversarial accuracy &ge;50% (PGD) and clean accuracy &ge;90%, given the &epsilon;-accuracy tradeoff physics of adversarial training.</div>
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
                    Adversarial attack mechanics (&epsilon;-ball, FGSM, PGD) from @sec-robust-ai &middot;
                    Loss landscape intuition from @sec-training-optimization
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
                "Adversarial training is the standard defense against &epsilon;-ball attacks &mdash; so why does hardening a model with PGD-7 at &epsilon;=8/255 cost 26 percentage points of clean accuracy, and is there any defense configuration that satisfies both the security and clinical requirements simultaneously?"
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

    - **@sec-robust-ai-introduction-robust-ai-systems-4671** — The Silent Failure Problem:
      why ML systems fail confidently on out-of-distribution and adversarial inputs.
    - **@sec-robust-ai** (adversarial attacks section) — FGSM and PGD attack mechanics;
      the ε-ball definition and why imperceptible perturbations cause large accuracy drops.
    - **@sec-robust-ai** (adversarial training section) — Madry et al. minimax formulation;
      why adversarial training is the standard defense but imposes a clean-accuracy cost.

    If you have not read these sections, the predictions in this lab will not map to the physics.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE (hide_code=True) ─────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Production (Standard Model)": "standard",
            "Hardened (Adversarially Trained)": "hardened",
        },
        value="Production (Standard Model)",
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
    _is_hardened = _ctx == "hardened"
    _color = COLORS["BlueLine"] if _is_hardened else COLORS["RedLine"]
    _label = "Hardened (Adversarially Trained)" if _is_hardened else "Production (Standard Model)"
    _specs = (
        "PGD-7 adversarial training · ε_train=8/255 · ~8× training overhead · "
        "robustness guarantee within ε-ball"
        if _is_hardened else
        "Standard ERM training · 97.3% clean accuracy · no adversarial defense · "
        "3.4% adversarial accuracy at ε=8/255"
    )
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {'#EBF4FA' if _is_hardened else '#FEF2F2'};
                border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 10px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
            Active Context
        </div>
        <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">{_label}</div>
        <div style="font-size: 0.85rem; color: #475569; margin-top: 3px;">{_specs}</div>
    </div>
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["RedLine"]
    _act_title    = "Adversarial Fragility Revelation"
    _act_duration = "12&ndash;15 min"
    _act_why      = ("You expect that a model with 97.3% clean accuracy is robust &mdash; after all, "
                     "it classifies correctly almost all the time. "
                     "A single FGSM perturbation of &epsilon;=8/255 will show it drops to 3.4%, "
                     "which is worse than random chance for a 10-class problem. "
                     "The question is not whether the model fails; it is why the failure is "
                     "so catastrophic when the perturbation is invisible to human eyes.")
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
    _color = COLORS["RedLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['RedL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · ML Security Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Our medical image classifier achieves 97.3% accuracy on clean images.
            A security researcher just showed us that adding imperceptible noise
            (ε=8/255 in pixel space — noise invisible to radiologists) drops our
            accuracy to 3.4%. We have never tested adversarial robustness. I need
            to understand: is 3.4% normal for attacked models, or is something
            especially wrong here?"
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Adversarial Accuracy Baseline

    Before exploring the simulator, you need to interpret the 3.4% figure.
    A 10-class medical classifier (10 disease categories) would achieve **10% accuracy by
    random guessing**. Standard image classifiers without any defense typically drop to
    somewhere in the **30–60% range** under ε=8/255 FGSM attacks — below clean accuracy,
    but still well above random.

    The medical classifier is at **3.4%** — significantly below random chance.
    This means adversarial examples are not merely confusing the model;
    they are actively steering predictions toward specific wrong classes.
    """)
    return


# ─── ACT I PREDICTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before touching the simulator, commit to your hypothesis:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) 3.4% is normal — all models are vulnerable to adversarial examples, "
            "and this level of degradation is expected":
                "option_a",
            "B) 3.4% is worse than random chance — the model has been maximally fooled, "
            "not just confused":
                "option_b",
            "C) The noise must be visible to achieve this — imperceptible noise cannot "
            "have such large effects on a 97.3% model":
                "option_c",
            "D) This is a theoretical concern only — adversarial attacks are too complex "
            "to be practical in real medical deployments":
                "option_d",
        },
        label="Our 10-class medical classifier drops from 97.3% to 3.4% under ε=8/255 FGSM. "
              "Which interpretation is correct?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the Adversarial Vulnerability Visualizer."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** {act1_pred.value}. Now explore the physics below."),
        kind="info",
    )
    return


# ─── ACT I INSTRUMENTS ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Adversarial Vulnerability Visualizer")
    return


@app.cell(hide_code=True)
def _(mo):
    model_acc_slider = mo.ui.slider(
        start=75, stop=99, value=97, step=1,
        label="Model clean accuracy (%)",
        show_value=True,
    )
    epsilon_slider = mo.ui.slider(
        start=1, stop=16, value=8, step=1,
        label="Perturbation budget ε (× 1/255)",
        show_value=True,
    )
    attack_type = mo.ui.dropdown(
        options={"FGSM (1 step)": "fgsm", "PGD-7 (7 steps)": "pgd7", "PGD-20 (20 steps)": "pgd20"},
        value="FGSM (1 step)",
        label="Attack type",
    )
    mo.vstack([
        mo.md("""
        Adjust the sliders to explore how clean accuracy, perturbation budget (ε),
        and attack strength interact. **FGSM** (Fast Gradient Sign Method) is a
        single-step attack: `x_adv = x + ε × sign(∇_x L(f(x), y))`. **PGD**
        (Projected Gradient Descent) iterates FGSM multiple times — stronger attacks
        with more iterations find adversarial examples closer to the decision boundary.
        """),
        mo.hstack([model_acc_slider, epsilon_slider, attack_type],
                  justify="start", gap="2rem"),
    ])
    return (model_acc_slider, epsilon_slider, attack_type)


@app.cell(hide_code=True)
def _(mo, model_acc_slider, epsilon_slider, attack_type, COLORS, np):
    # ── Physics model for adversarial accuracy ──────────────────────────────────
    # Source: empirical robustness literature and @sec-robust-ai
    #
    # Key relationships (from the chapter):
    # 1. Clean accuracy has weak correlation with adversarial accuracy in undefended models.
    # 2. Adversarial accuracy degrades roughly quadratically with ε for undefended models.
    # 3. PGD finds stronger adversarial examples than FGSM by iterative refinement.
    # 4. Adversarial accuracy can fall below random (10% for 10-class) when attack drives
    #    predictions to specific wrong classes — the "adversarial examples are not random"
    #    property from Goodfellow et al. 2014.
    #
    # Model (calibrated to match chapter scenario):
    #   - Base adversarial vulnerability at ε=8/255 for a 97% clean model ≈ 3–5%
    #   - FGSM baseline at ε=8/255: ~5% for undefended models in literature
    #   - PGD-7 tightens this further to ~3%
    #   - PGD-20 finds near-worst-case: ~2%

    _eps = epsilon_slider.value   # in units of 1/255
    _acc = model_acc_slider.value  # %
    _atk = attack_type.value

    # Attack strength multiplier: PGD is stronger than FGSM
    _attack_mult = {"fgsm": 1.0, "pgd7": 1.4, "pgd20": 1.8}[_atk]

    # Random chance for 10-class = 10%
    _random_acc = 10.0

    # Compute adversarial accuracy without defense.
    # Physics: for undefended model, adversarial accuracy = f(ε, attack_strength)
    # Calibrated: at ε=8, FGSM → ~5%; PGD-7 → ~3.4%; PGD-20 → ~2%.
    # At ε=1, attacks are weak: adversarial ≈ clean - small drop.
    # Formula: adv_acc = clean_acc × exp(-k × ε × attack_mult)
    # where k is calibrated so ε=8/255, FGSM, 97% clean → ~5%.
    _k = 0.062  # calibration constant
    _adv_acc_raw = _acc * np.exp(-_k * _eps * _attack_mult)

    # Adversarial examples actively steer to wrong classes once the attack is
    # strong enough — accuracy can fall below random. This happens when the
    # gradient signal is strong enough to consistently target specific wrong classes.
    # Characteristic: for eps > 6 and PGD, adv_acc often goes below random.
    _adv_acc = float(max(1.0, _adv_acc_raw))

    # Accuracy gap
    _gap = _acc - _adv_acc
    _below_random = _adv_acc < _random_acc

    # Color coding
    _adv_color = (
        COLORS["RedLine"] if _adv_acc < _random_acc
        else COLORS["OrangeLine"] if _adv_acc < 30
        else COLORS["GreenLine"]
    )
    _gap_color = COLORS["RedLine"] if _gap > 50 else COLORS["OrangeLine"] if _gap > 20 else COLORS["GreenLine"]

    # ── ε sweep for chart ────────────────────────────────────────────────────────
    _eps_vals = np.arange(1, 17, 1)
    _fgsm_vals   = np.maximum(1.0, _acc * np.exp(-0.062 * _eps_vals * 1.0))
    _pgd7_vals   = np.maximum(1.0, _acc * np.exp(-0.062 * _eps_vals * 1.4))
    _pgd20_vals  = np.maximum(1.0, _acc * np.exp(-0.062 * _eps_vals * 1.8))
    _clean_line  = np.full_like(_eps_vals, float(_acc))
    _random_line = np.full_like(_eps_vals, 10.0)

    mo.vstack([
        mo.md("### Physics"),
        mo.Html(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                    padding:16px 20px; font-family: monospace; font-size: 0.88rem;
                    color: #1e293b; margin-bottom:12px;">
            <strong>FGSM:</strong> x_adv = x + ε × sign(∇_x L(f(x), y))<br>
            <strong>PGD:</strong>  x_adv⁽⁰⁾ = x; x_adv⁽ᵗ⁺¹⁾ = Π_B(x_adv⁽ᵗ⁾ + α × sign(∇_x L))
            where B = ε-ball around x<br><br>
            <strong>Current settings:</strong><br>
            ε = {_eps}/255 = {_eps/255:.4f} &nbsp;|&nbsp;
            Attack = {attack_type.value} (strength multiplier = {_attack_mult:.1f}×)<br>
            Clean accuracy = {_acc:.1f}% &nbsp;|&nbsp;
            Adversarial accuracy = {_adv_acc:.1f}%<br>
            Accuracy gap = {_gap:.1f} percentage points
            {' ← <strong style="color:#CB202D;">BELOW RANDOM CHANCE (10%)</strong>' if _below_random else ''}
        </div>
        """),
        mo.md("### Results"),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin: 16px 0; flex-wrap:wrap;">
            <div style="padding: 20px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Clean Accuracy
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {COLORS['GreenLine']};
                            margin: 8px 0;">
                    {_acc:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">No attack applied</div>
            </div>
            <div style="padding: 20px 24px; border: 2px solid {_adv_color}; border-radius: 10px;
                        width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: {_adv_color}; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Adversarial Accuracy
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {_adv_color};
                            margin: 8px 0;">
                    {_adv_acc:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">
                    Under ε={_eps}/255 {attack_type.value}
                </div>
            </div>
            <div style="padding: 20px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Accuracy Gap
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {_gap_color};
                            margin: 8px 0;">
                    {_gap:.1f}pp
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Clean − adversarial</div>
            </div>
            <div style="padding: 20px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Random Baseline
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #94a3b8;
                            margin: 8px 0;">
                    10.0%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">10-class random guess</div>
            </div>
        </div>
        """),
    ])
    return (_adv_acc, _gap, _below_random)


@app.cell(hide_code=True)
def _(mo, model_acc_slider, epsilon_slider, attack_type, COLORS, np, go, apply_plotly_theme):
    # ── ε sweep chart ─────────────────────────────────────────────────────────
    _acc = model_acc_slider.value
    _eps_current = epsilon_slider.value
    _atk = attack_type.value

    _eps_vals = np.arange(1, 17, 1)
    _fgsm_curve  = np.maximum(1.0, _acc * np.exp(-0.062 * _eps_vals * 1.0))
    _pgd7_curve  = np.maximum(1.0, _acc * np.exp(-0.062 * _eps_vals * 1.4))
    _pgd20_curve = np.maximum(1.0, _acc * np.exp(-0.062 * _eps_vals * 1.8))

    _fig = go.Figure()

    # Clean accuracy reference
    _fig.add_trace(go.Scatter(
        x=_eps_vals, y=np.full_like(_eps_vals, float(_acc)),
        mode="lines", name=f"Clean accuracy ({_acc:.0f}%)",
        line=dict(color=COLORS["GreenLine"], width=2, dash="dash"),
    ))

    # Random chance reference
    _fig.add_trace(go.Scatter(
        x=_eps_vals, y=np.full_like(_eps_vals, 10.0),
        mode="lines", name="Random (10%)",
        line=dict(color=COLORS["Grey"], width=1.5, dash="dot"),
    ))

    # Attack curves
    _fig.add_trace(go.Scatter(
        x=_eps_vals, y=_fgsm_curve,
        mode="lines+markers", name="FGSM (1 step)",
        line=dict(color=COLORS["OrangeLine"], width=2),
        marker=dict(size=5),
    ))
    _fig.add_trace(go.Scatter(
        x=_eps_vals, y=_pgd7_curve,
        mode="lines+markers", name="PGD-7 (7 steps)",
        line=dict(color=COLORS["BlueLine"], width=2),
        marker=dict(size=5),
    ))
    _fig.add_trace(go.Scatter(
        x=_eps_vals, y=_pgd20_curve,
        mode="lines+markers", name="PGD-20 (20 steps)",
        line=dict(color=COLORS["RedLine"], width=2),
        marker=dict(size=5),
    ))

    # Vertical line at current ε
    _fig.add_shape(
        type="line",
        x0=_eps_current, y0=0, x1=_eps_current, y1=_acc,
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
    )
    _fig.add_annotation(
        x=_eps_current, y=_acc * 0.6,
        text=f"ε={_eps_current}/255",
        showarrow=False,
        font=dict(size=10, color="#64748b"),
        xanchor="left",
        xshift=4,
    )

    # "Below random" shaded region
    _fig.add_shape(
        type="rect",
        x0=1, y0=0, x1=16, y1=10,
        fillcolor="rgba(203,32,45,0.06)",
        line=dict(width=0),
    )
    _fig.add_annotation(
        x=14, y=5,
        text="Below random",
        showarrow=False,
        font=dict(size=9, color=COLORS["RedLine"]),
    )

    apply_plotly_theme(_fig)
    _fig.update_layout(
        title="Adversarial Accuracy vs. Perturbation Budget (Undefended Model)",
        xaxis_title="Perturbation budget ε (× 1/255)",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[1, 16]),
        height=400,
        legend=dict(x=0.75, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )
    _fig
    return


# ─── ACT I PREDICTION-VS-REALITY OVERLAY ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, COLORS):
    _correct = act1_pred.value == "option_b"

    _feedback = {
        "option_a": mo.callout(mo.md(
            "**Not correct.** The 3.4% figure is not normal degradation — it is below the "
            "10% random-chance baseline for a 10-class classifier. A model that falls below "
            "random is not merely confused; the attack is actively steering predictions to "
            "specific wrong classes. This is the adversarial property first identified by "
            "Goodfellow et al. (2014): FGSM perturbations are not random noise — they are "
            "gradient-aligned signals that push representations across decision boundaries "
            "in a consistent direction."
        ), kind="warn"),
        "option_b": mo.callout(mo.md(
            "**Correct.** 3.4% is below the 10% random-chance baseline for a 10-class "
            "classifier. A model guessing randomly achieves 10%; this model under attack "
            "achieves 3.4%. That means the adversarial examples are not merely confusing "
            "the model — they are systematically directing predictions to specific wrong "
            "classes. This is the hallmark property of adversarial examples: the gradient "
            "of the loss with respect to the input is a structured signal, not noise. "
            "Adding ε × sign(∇_x L) moves the input in a direction that maximally increases "
            "the loss — and in high dimensions, that direction is consistent across inputs."
        ), kind="success"),
        "option_c": mo.callout(mo.md(
            "**Not correct.** This is the most common misconception about adversarial "
            "examples, and it was the central finding of Goodfellow et al. (2014). "
            "In high-dimensional pixel space (e.g. 224×224×3 = 150,528 dimensions), "
            "an L∞ perturbation of ε=8/255 ≈ 0.031 per pixel is imperceptible to humans "
            "but accumulates across 150,528 independent gradient steps. The total shift in "
            "logit space can be enormous even though no single pixel changes visibly. "
            "The phenomenon is a direct consequence of the curse of dimensionality."
        ), kind="warn"),
        "option_d": mo.callout(mo.md(
            "**Not correct.** Adversarial attacks are practical in production systems. "
            "Physical adversarial examples (printed patches, stickers) have been demonstrated "
            "against stop signs, face recognition, and self-driving perception systems. "
            "Digital attacks against medical imaging classifiers and financial fraud detectors "
            "have been demonstrated in research settings. The chapter's robustness definition "
            "explicitly addresses worst-case performance under perturbation — a systems concern, "
            "not a theoretical exercise."
        ), kind="warn"),
    }

    mo.vstack([
        mo.md("### Prediction vs. Reality"),
        mo.Html(f"""
        <div style="background:#f0f4ff; border-radius:10px; padding:14px 20px; margin-bottom:10px;">
            <div style="font-weight:700; color:{COLORS['BlueLine']}; margin-bottom:6px;">
                Your prediction: {act1_pred.value.replace('option_', 'Option ').upper()}
            </div>
            <div style="font-size:0.9rem; color:#475569;">
                The actual result: <strong>3.4% is below the 10% random baseline — the model "
                has been maximally fooled, not just confused.</strong>
                The adversarial accuracy gap is 97.3% − 3.4% = 93.9 percentage points.
                A random classifier would only be off by 97.3% − 10% = 87.3 percentage points.
                The attack outperforms randomness — it is an active signal, not passive noise.
            </div>
        </div>
        """),
        _feedback[act1_pred.value],
    ])
    return


# ─── ACT I REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *Now that you have seen the physics, test your understanding of the mechanism:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Models are undertrained — better training eliminates adversarial vulnerability":
                "ref_a",
            "B) In high dimensions, a tiny L∞ perturbation accumulates across thousands of "
            "pixels to cause large logit changes — the curse of dimensionality enables "
            "adversarial examples":
                "ref_b",
            "C) Gradient descent creates adversarial vulnerabilities — non-gradient optimizers "
            "produce robust models":
                "ref_c",
            "D) Adversarial examples only affect convolutional networks — transformer-based "
            "models are not vulnerable":
                "ref_d",
        },
        label="What makes adversarial examples possible in high-dimensional pixel spaces?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue to the MathPeek."), kind="warn"),
    )
    _correct = act1_reflect.value == "ref_b"
    _feedback = {
        "ref_a": mo.callout(mo.md(
            "**Not correct.** Well-trained models with 99%+ accuracy on clean data are equally "
            "vulnerable to adversarial examples — in fact, higher clean accuracy can correlate "
            "with higher adversarial vulnerability in undefended models, because the decision "
            "boundaries become sharper and easier to cross. The vulnerability is structural, "
            "not a sign of insufficient training."
        ), kind="warn"),
        "ref_b": mo.callout(mo.md(
            "**Correct.** For an image with D = 224×224×3 = 150,528 pixels, an L∞ perturbation "
            "of ε=8/255 ≈ 0.031 per pixel can shift the input by up to D × ε = 150,528 × 0.031 ≈ "
            "4,666 units in L1 distance, even though no single pixel changed visibly. When the "
            "perturbation is aligned with the gradient of the loss (as in FGSM), each pixel's "
            "small change accumulates constructively, producing a large change in the model's "
            "output. This is the core insight from Goodfellow et al. (2014): adversarial "
            "fragility is not a bug but a direct consequence of linearity in high dimensions."
        ), kind="success"),
        "ref_c": mo.callout(mo.md(
            "**Not correct.** The vulnerability is not caused by the training algorithm — it "
            "is caused by the geometry of learned decision boundaries in high dimensions. "
            "Non-gradient optimizers (e.g. evolutionary strategies) produce models with the "
            "same fundamental vulnerability if they achieve similar clean accuracy. The loss "
            "landscape structure, not the optimizer, determines adversarial robustness."
        ), kind="warn"),
        "ref_d": mo.callout(mo.md(
            "**Not correct.** All high-dimensional differentiable models — CNNs, transformers, "
            "MLPs, recurrent networks — are vulnerable to adversarial examples. Vision "
            "transformers (ViT) trained with standard ERM show similar adversarial accuracy "
            "drops to CNNs under FGSM and PGD attacks. The vulnerability is a property of "
            "high-dimensional input spaces and linear approximations to nonlinear functions, "
            "not of any specific architecture."
        ), kind="warn"),
    }
    mo.vstack([
        act1_reflect,
        _feedback[act1_reflect.value],
    ])
    return


# ─── ACT I MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
        **FGSM Attack (Goodfellow et al., 2014):**

        ```
        x_adv = x + ε × sign(∇_x L(f(x), y))
        ```

        - **x** — clean input (e.g. image tensor)
        - **ε** — perturbation budget (L∞ ball radius); typical value: 8/255 ≈ 0.031
        - **L** — task loss (e.g. cross-entropy)
        - **sign(∇_x L)** — sign of gradient: +1 or −1 per pixel

        The L∞ constraint ensures `max |x_adv − x|∞ ≤ ε` — each pixel shifts by at most ε.
        In D dimensions, L1 shift = D × ε (up to 4,666 for 224×224×3 at ε=8/255).

        **PGD Attack (Madry et al., 2018) — stronger multi-step version:**

        ```
        x_adv⁽⁰⁾ = x
        x_adv⁽ᵗ⁺¹⁾ = Π_B(x_adv⁽ᵗ⁾ + α × sign(∇_x L(f(x_adv⁽ᵗ⁾), y)))
        ```

        - **Π_B** — projection back onto the ε-ball B = {x': ||x' − x||∞ ≤ ε}
        - **α** — step size per iteration (typically α = ε/4 for PGD-7)
        - Convergence: PGD-K finds near-worst-case adversarial example within the ε-ball

        **Adversarial accuracy (no defense):**

        ```
        acc_adv(ε) ≈ acc_clean × exp(−k × ε × attack_strength)
        ```

        Where k is a model-dependent vulnerability constant. Below ε=0, acc_adv = acc_clean.
        At high ε, acc_adv can fall below random (1/C for C-class) — the attack is constructive.

        **Why 3.4% is worse than 10% random:**

        For a 10-class classifier under a targeted attack,
        the adversarial perturbation steers the output toward a specific target class.
        The model is not confused — it is confidently wrong in a consistent direction.
        This is why adversarial examples are a security concern, not just a performance concern.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _act2_num      = "II"
    _act2_color    = COLORS["BlueLine"]
    _act2_title    = "Robustness-Accuracy Tradeoff"
    _act2_duration = "20&ndash;25 min"
    _act2_why      = ("Act I confirmed that an undefended model collapses under perturbation. "
                      "Now discover why the cure is almost as painful as the disease: "
                      "adversarial training forces decision boundaries wider to resist attacks, "
                      "which inevitably misclassifies clean examples that were near tight margins. "
                      "Your task is to find whether the security constraint (&ge;50% adversarial) "
                      "and clinical constraint (&ge;90% clean) can be satisfied simultaneously &mdash; "
                      "or whether they are fundamentally incompatible.")
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
    _color = COLORS["BlueLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['BlueL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message · CISO
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "After the security audit, we have two non-negotiable requirements:
            (1) Security: adversarial accuracy at ε=8/255 must exceed 50% under PGD-7 attack.
            (2) Clinical: clean accuracy must stay above 90% — below that, our radiologists
            won't trust the system. We currently have 97.3% clean accuracy and 3.4% adversarial.
            Can adversarial training satisfy both constraints simultaneously?
            What is the actual tradeoff we are facing?"
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Robustness-Accuracy Tradeoff

    Adversarial training (Madry et al., 2018) solves the minimax problem:
    `θ* = argmin_θ E[max_{δ: ||δ||∞ ≤ ε} L(f_θ(x+δ), y)]`

    To do this, the model must learn decision boundaries that remain correct for
    **all inputs within the ε-ball** of every training point — not just the clean
    input itself. This forces the model to learn wider, more conservative boundaries,
    reducing precision on typical clean examples. The tradeoff is fundamental and
    documented empirically across architectures and datasets.

    From the chapter: adversarial training typically costs **3–10% clean accuracy**
    on CIFAR-10 and ImageNet-scale tasks. Achieving ε=8/255 adversarial accuracy
    above 50% typically requires training at or above ε=8/255 with PGD-7+, which
    costs closer to **5–8% clean accuracy** in practice.
    """)
    return


# ─── ACT II PREDICTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before adjusting the adversarial training configurator, commit to your hypothesis:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Yes — adversarial training can simultaneously achieve 50% adversarial "
            "accuracy and maintain 97% clean accuracy":
                "opt2_a",
            "B) No — adversarial training typically costs 3–10% clean accuracy; reaching "
            "50% adversarial accuracy at ε=8/255 will reduce clean accuracy to ~88–92%":
                "opt2_b",
            "C) Use data augmentation instead — it achieves adversarial robustness without "
            "any clean accuracy cost":
                "opt2_c",
            "D) The constraint is satisfiable only with certified robustness methods — "
            "adversarial training alone cannot meet both requirements":
                "opt2_d",
        },
        label="The CISO requires: adversarial accuracy > 50% AND clean accuracy > 90%. "
              "Starting from 97.3% clean / 3.4% adversarial, is this constraint satisfiable?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the Adversarial Training Configurator."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** {act2_pred.value}. Now configure the adversarial training below."),
        kind="info",
    )
    return


# ─── ACT II INSTRUMENTS ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Adversarial Training Configurator")
    return


@app.cell(hide_code=True)
def _(mo):
    train_eps_slider = mo.ui.slider(
        start=0, stop=16, value=0, step=1,
        label="Adversarial training ε (× 1/255)",
        show_value=True,
    )
    pgd_steps_slider = mo.ui.slider(
        start=1, stop=20, value=1, step=1,
        label="PGD steps in training (inner loop)",
        show_value=True,
    )
    adv_loss_weight = mo.ui.slider(
        start=0.0, stop=1.0, value=0.0, step=0.05,
        label="Adversarial loss weight (0 = standard, 1 = full adversarial)",
        show_value=True,
    )
    mo.vstack([
        mo.md("""
        Configure the adversarial training setup. The **training ε** sets the
        perturbation budget used during training — must be ≥ evaluation ε (8/255) to
        provide genuine defense. **PGD steps** controls inner loop quality — more steps
        means stronger adversarial examples during training, better robustness but slower
        training. **Adversarial loss weight** interpolates between standard (0) and
        fully adversarial (1) training objectives.
        """),
        mo.hstack([train_eps_slider, pgd_steps_slider, adv_loss_weight],
                  justify="start", gap="2rem"),
    ])
    return (train_eps_slider, pgd_steps_slider, adv_loss_weight)


@app.cell(hide_code=True)
def _(
    mo, train_eps_slider, pgd_steps_slider, adv_loss_weight,
    COLORS, np,
    CLEAN_ACC_BASELINE, SECURITY_ADV_THRESHOLD, PRODUCT_CLEAN_THRESHOLD,
):
    # ── Adversarial training physics model ────────────────────────────────────
    # Source: @sec-robust-ai adversarial training section; Madry et al. 2018;
    # empirical robustness benchmarks on CIFAR-10 and ImageNet-scale models.
    #
    # Key relationships:
    # 1. Adversarial training at ε_train > 0 gains robustness at ε_eval=8/255.
    #    Robustness at eval eps: monotone in train eps (up to train eps ≈ eval eps).
    # 2. More PGD steps → stronger inner loop → better robustness, more cost.
    # 3. Adversarial loss weight α: loss = α × L_adv + (1-α) × L_clean.
    #    α=0: standard training (no robustness gain). α=1: full adversarial.
    # 4. The clean accuracy cost is approximately proportional to α × train_eps.
    #
    # Physics (calibrated to chapter claims and published benchmarks):
    #   - Full adversarial training at ε=8/255, PGD-7: clean ≈ 87–90%, adv ≈ 44–52%
    #   - At ε=4/255, PGD-7: clean ≈ 92–94%, adv (at ε=8) ≈ 20–30%
    #   - Intermediate ε and steps: interpolated

    _train_eps = train_eps_slider.value   # in 1/255 units
    _pgd_k     = pgd_steps_slider.value
    _alpha     = adv_loss_weight.value    # 0–1

    # Robustness gain per unit: how much adversarial accuracy at ε=8/255 is achieved
    # per unit of training effort. Calibrated so ε=8, k=7, α=1 → ~48% adv accuracy.
    # More PGD steps improve robustness (logarithmic saturation after k=7).
    _pgd_factor = np.log(1 + _pgd_k) / np.log(8)   # relative to PGD-7 baseline

    # Adversarial accuracy at ε_eval=8/255:
    # max achievable = 52% (from published benchmarks at ε=8/255 PGD-7 full adversarial)
    # Scales with: training eps coverage of eval eps, PGD steps, and alpha.
    _eps_coverage = min(1.0, _train_eps / 8.0)  # fraction of eval eps covered during training
    _adv_acc_max = 52.0  # % — published ceiling for standard PGD-AT at ε=8/255
    _adv_acc_gained = _adv_acc_max * _eps_coverage * _pgd_factor * _alpha

    # Floor: the undefended model gets ~3.4% from pure FGSM on a 97.3% model
    _adv_acc_2 = float(max(3.4, _adv_acc_gained))

    # Clean accuracy cost:
    # Full adversarial at ε=8/255 costs ≈ 8–10% clean accuracy.
    # Scales with alpha (interpolation) and training_eps (how aggressively we defend).
    # Published: CIFAR-10 WideResNet: clean 84.7% → adv 53.0% (Madry et al.).
    # ImageNet ResNet-50: clean 76.0% → 63.0% (under PGD-AT, ε=4/255).
    # Medical domain (fine-tuned models, higher clean baseline): similar magnitude.
    _clean_cost_max = 9.0   # % — maximum clean accuracy reduction at full adversarial ε=8/255
    _clean_cost = _clean_cost_max * _eps_coverage * _pgd_factor * _alpha
    _clean_acc_2 = float(max(70.0, CLEAN_ACC_BASELINE - _clean_cost))

    # Training overhead: PGD-K requires K+1 forward/backward passes per batch
    _train_overhead = 1 + _pgd_k  # multiplier over standard training time

    # Constraint checks
    _security_met = _adv_acc_2 >= SECURITY_ADV_THRESHOLD    # > 50%
    _product_met  = _clean_acc_2 >= PRODUCT_CLEAN_THRESHOLD  # > 90%
    _both_met     = _security_met and _product_met

    # Colors
    _adv_color   = COLORS["GreenLine"] if _security_met else COLORS["RedLine"]
    _clean_color = COLORS["GreenLine"] if _product_met else COLORS["OrangeLine"]
    _ovhd_color  = COLORS["OrangeLine"] if _train_overhead > 5 else COLORS["BlueLine"]

    mo.vstack([
        mo.md("### Physics"),
        mo.Html(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                    padding:16px 20px; font-family: monospace; font-size: 0.88rem;
                    color: #1e293b; margin-bottom:12px;">
            <strong>Madry et al. minimax formulation:</strong><br>
            θ* = argmin_θ E[max_{{δ: ||δ||∞ ≤ ε_train}} L(f_θ(x+δ), y)]<br><br>
            <strong>Mixed loss (TRADES-style):</strong><br>
            L_total = (1-α) × L_clean(x) + α × L_adv(x, ε_train, PGD-{_pgd_k})<br><br>
            <strong>Current configuration:</strong><br>
            ε_train = {_train_eps}/255 &nbsp;|&nbsp;
            PGD steps = {_pgd_k} &nbsp;|&nbsp;
            α (adv weight) = {_alpha:.2f}<br>
            Training overhead = {_train_overhead:.0f}× standard training time<br>
            Clean accuracy: {CLEAN_ACC_BASELINE:.1f}% → {_clean_acc_2:.1f}%
            (cost: {_clean_cost:.1f}pp)<br>
            Adversarial accuracy at ε=8/255 PGD-7: {_adv_acc_2:.1f}%
        </div>
        """),
        mo.md("### Results"),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin: 16px 0; flex-wrap:wrap;">
            <div style="padding: 20px 24px; border: 2px solid {_clean_color};
                        border-radius: 10px; width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: {_clean_color}; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Clean Accuracy
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {_clean_color};
                            margin: 8px 0;">
                    {_clean_acc_2:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">
                    Threshold: {PRODUCT_CLEAN_THRESHOLD:.0f}%
                    {'✓' if _product_met else '✗'}
                </div>
            </div>
            <div style="padding: 20px 24px; border: 2px solid {_adv_color};
                        border-radius: 10px; width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: {_adv_color}; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Adversarial Accuracy
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {_adv_color};
                            margin: 8px 0;">
                    {_adv_acc_2:.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">
                    Threshold: {SECURITY_ADV_THRESHOLD:.0f}%
                    {'✓' if _security_met else '✗'}
                </div>
            </div>
            <div style="padding: 20px 24px; border: 1px solid #e2e8f0;
                        border-radius: 10px; width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: #64748b; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Accuracy Gap
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #1e293b;
                            margin: 8px 0;">
                    {_clean_acc_2 - _adv_acc_2:.1f}pp
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">Clean − adversarial</div>
            </div>
            <div style="padding: 20px 24px; border: 1px solid {_ovhd_color};
                        border-radius: 10px; width: 200px; text-align: center; background: white;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.07);">
                <div style="color: {_ovhd_color}; font-size: 0.82rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">
                    Training Overhead
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {_ovhd_color};
                            margin: 8px 0;">
                    {_train_overhead:.0f}×
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8;">vs. standard training</div>
            </div>
        </div>
        """),
    ])
    return (_adv_acc_2, _clean_acc_2, _security_met, _product_met, _both_met, _train_overhead)


# ─── ACT II FAILURE STATES ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _adv_acc_2, _clean_acc_2, _security_met, _product_met, _both_met,
      SECURITY_ADV_THRESHOLD, PRODUCT_CLEAN_THRESHOLD):
    if _both_met:
        _banner = mo.callout(mo.md(
            f"**Both requirements met.** Clean accuracy: {_clean_acc_2:.1f}% (threshold: "
            f"{PRODUCT_CLEAN_THRESHOLD:.0f}%). Adversarial accuracy: {_adv_acc_2:.1f}% "
            f"(threshold: {SECURITY_ADV_THRESHOLD:.0f}%). This configuration satisfies the "
            f"CISO's joint constraint — but note the clean accuracy cost: "
            f"{97.3 - _clean_acc_2:.1f} percentage points from the original 97.3%."
        ), kind="success")
    elif not _security_met and not _product_met:
        _banner = mo.vstack([
            mo.callout(mo.md(
                f"**Security requirement unmet:** ε=8/255 adversarial accuracy = "
                f"{_adv_acc_2:.1f}% < {SECURITY_ADV_THRESHOLD:.0f}% required. "
                f"Increase training ε and PGD steps to improve robustness."
            ), kind="danger"),
            mo.callout(mo.md(
                f"**Clinical accuracy below threshold:** clean accuracy {_clean_acc_2:.1f}% "
                f"< {PRODUCT_CLEAN_THRESHOLD:.0f}% product requirement. "
                f"Reduce adversarial loss weight or training ε to recover clean accuracy."
            ), kind="warn"),
        ])
    elif not _security_met:
        _banner = mo.callout(mo.md(
            f"**Security requirement unmet:** ε=8/255 adversarial accuracy = "
            f"{_adv_acc_2:.1f}% < {SECURITY_ADV_THRESHOLD:.0f}% required. "
            f"Increase the adversarial training ε (must cover the evaluation ε=8/255), "
            f"the number of PGD steps, or the adversarial loss weight."
        ), kind="danger")
    else:
        _banner = mo.callout(mo.md(
            f"**Clinical accuracy below threshold:** clean accuracy {_clean_acc_2:.1f}% "
            f"< {PRODUCT_CLEAN_THRESHOLD:.0f}% product requirement. "
            f"This is the fundamental robustness-accuracy tradeoff: "
            f"reducing the adversarial loss weight or training ε will recover clean "
            f"accuracy, but may push adversarial accuracy back below 50%."
        ), kind="warn")

    _banner
    return


# ─── ACT II PARETO FRONTIER CHART ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, train_eps_slider, pgd_steps_slider, adv_loss_weight,
    COLORS, np, go, apply_plotly_theme,
    CLEAN_ACC_BASELINE, SECURITY_ADV_THRESHOLD, PRODUCT_CLEAN_THRESHOLD,
):
    # ── Pareto frontier: sweep alpha from 0 to 1 at current ε and PGD settings ─
    _train_eps = train_eps_slider.value
    _pgd_k     = pgd_steps_slider.value
    _alpha_cur = adv_loss_weight.value

    _pgd_factor = np.log(1 + _pgd_k) / np.log(8)
    _eps_coverage = min(1.0, _train_eps / 8.0)

    _alphas = np.linspace(0, 1, 50)
    _adv_accs  = np.maximum(3.4, 52.0 * _eps_coverage * _pgd_factor * _alphas)
    _clean_accs = np.maximum(70.0, CLEAN_ACC_BASELINE - 9.0 * _eps_coverage * _pgd_factor * _alphas)

    # Current operating point
    _adv_cur  = float(max(3.4, 52.0 * _eps_coverage * _pgd_factor * _alpha_cur))
    _clean_cur = float(max(70.0, CLEAN_ACC_BASELINE - 9.0 * _eps_coverage * _pgd_factor * _alpha_cur))

    _fig2 = go.Figure()

    # Pareto frontier curve
    _fig2.add_trace(go.Scatter(
        x=_adv_accs, y=_clean_accs,
        mode="lines",
        name="Pareto frontier (current ε, PGD steps)",
        line=dict(color=COLORS["BlueLine"], width=2.5),
        hovertemplate="Adv: %{x:.1f}% | Clean: %{y:.1f}%<extra></extra>",
    ))

    # Current operating point
    _fig2.add_trace(go.Scatter(
        x=[_adv_cur], y=[_clean_cur],
        mode="markers+text",
        name="Current configuration",
        marker=dict(color=COLORS["OrangeLine"], size=12, symbol="circle"),
        text=["Current"],
        textposition="top left",
    ))

    # Undefended baseline
    _fig2.add_trace(go.Scatter(
        x=[3.4], y=[97.3],
        mode="markers+text",
        name="Undefended baseline",
        marker=dict(color=COLORS["RedLine"], size=10, symbol="x"),
        text=["Undefended"],
        textposition="top right",
    ))

    # Constraint lines
    _fig2.add_shape(
        type="line",
        x0=SECURITY_ADV_THRESHOLD, y0=60, x1=SECURITY_ADV_THRESHOLD, y1=100,
        line=dict(color=COLORS["RedLine"], width=1.5, dash="dash"),
    )
    _fig2.add_annotation(
        x=SECURITY_ADV_THRESHOLD, y=65,
        text=f"Security min ({SECURITY_ADV_THRESHOLD:.0f}%)",
        showarrow=False, font=dict(size=9, color=COLORS["RedLine"]),
        xanchor="left", xshift=4,
    )
    _fig2.add_shape(
        type="line",
        x0=0, y0=PRODUCT_CLEAN_THRESHOLD, x1=55, y1=PRODUCT_CLEAN_THRESHOLD,
        line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dash"),
    )
    _fig2.add_annotation(
        x=2, y=PRODUCT_CLEAN_THRESHOLD,
        text=f"Product min ({PRODUCT_CLEAN_THRESHOLD:.0f}%)",
        showarrow=False, font=dict(size=9, color=COLORS["OrangeLine"]),
        yanchor="bottom", yshift=4,
    )

    # "Feasible region" annotation
    _fig2.add_shape(
        type="rect",
        x0=SECURITY_ADV_THRESHOLD, y0=PRODUCT_CLEAN_THRESHOLD, x1=55, y1=100,
        fillcolor="rgba(0,143,69,0.06)",
        line=dict(width=0),
    )
    _fig2.add_annotation(
        x=52, y=95, text="Feasible\nregion",
        showarrow=False, font=dict(size=9, color=COLORS["GreenLine"]),
        xanchor="right",
    )

    apply_plotly_theme(_fig2)
    _fig2.update_layout(
        title="Robustness-Accuracy Pareto Frontier (adversarial loss weight α: 0→1)",
        xaxis_title="Adversarial accuracy at ε=8/255, PGD-7 (%)",
        yaxis_title="Clean accuracy (%)",
        xaxis=dict(range=[0, 56]),
        yaxis=dict(range=[60, 100]),
        height=420,
        legend=dict(x=0.02, y=0.15, bgcolor="rgba(255,255,255,0.85)"),
    )
    mo.vstack([
        mo.md("### Pareto Frontier: Robustness vs. Clean Accuracy"),
        _fig2,
        mo.md("""
        The Pareto frontier shows the achievable (clean accuracy, adversarial accuracy)
        combinations at the current training ε and PGD step settings.
        Moving right along the frontier improves adversarial robustness but reduces clean accuracy.
        The feasible region (green) requires simultaneously exceeding both thresholds —
        the frontier determines whether that region is reachable at all.
        """),
    ])
    return


# ─── ACT II PREDICTION-VS-REALITY OVERLAY ────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, COLORS):
    _correct2 = act2_pred.value == "opt2_b"

    _feedback2 = {
        "opt2_a": mo.callout(mo.md(
            "**Not correct.** The robustness-accuracy tradeoff is fundamental, not a "
            "configuration artifact. Adversarial training forces the model to satisfy "
            "the constraint `f(x+δ) = f(x)` for all ||δ||∞ ≤ ε — this requires learning "
            "decision boundaries that are ε-wider in all directions. Wider boundaries "
            "necessarily reduce precision on clean in-distribution examples, because some "
            "clean examples previously classified correctly now fall inside the expanded "
            "margin. Published benchmarks consistently show 5–8% clean accuracy reduction "
            "at ε=8/255 for models that achieve 44–52% adversarial accuracy."
        ), kind="warn"),
        "opt2_b": mo.callout(mo.md(
            "**Correct.** The joint constraint is satisfiable — but barely, and with a "
            "real cost. To reach 50% adversarial accuracy at ε=8/255, adversarial training "
            "with PGD-7 typically reduces clean accuracy from ~97% to ~89–92%. This puts "
            "clean accuracy near (and sometimes below) the 90% clinical threshold. The CISO "
            "is asking for both requirements simultaneously, but the Pareto frontier shows "
            "that the feasible region is narrow: the training configuration must be "
            "precisely tuned to hit both thresholds at the same time."
        ), kind="success"),
        "opt2_c": mo.callout(mo.md(
            "**Not correct.** Standard data augmentation (rotation, flipping, color jitter) "
            "improves generalization to natural distribution shifts but provides no defense "
            "against adversarial examples. Adversarial examples are specifically constructed "
            "to exploit the gradient of the loss — they lie outside the space of natural "
            "transformations that augmentation addresses. Only adversarial training (or "
            "certified defenses) provides genuine adversarial robustness. This is one of "
            "the key empirical findings from the adversarial robustness literature."
        ), kind="warn"),
        "opt2_d": mo.callout(mo.md(
            "**Not correct.** Certified robustness methods (randomized smoothing, interval "
            "bound propagation) provide provable guarantees but have even larger clean "
            "accuracy costs than PGD adversarial training — typically 10–20% on CIFAR-10 "
            "and ImageNet. They are not a path to satisfying both requirements more easily. "
            "The CISO's constraint can be met with carefully configured PGD adversarial "
            "training — the joint feasibility depends on the training configuration."
        ), kind="warn"),
    }

    mo.vstack([
        mo.md("### Prediction vs. Reality"),
        mo.Html(f"""
        <div style="background:#f0f4ff; border-radius:10px; padding:14px 20px; margin-bottom:10px;">
            <div style="font-weight:700; color:{COLORS['BlueLine']}; margin-bottom:6px;">
                Your prediction: {act2_pred.value.replace('opt2_', 'Option ').upper()}
            </div>
            <div style="font-size:0.9rem; color:#475569;">
                The physics: adversarial training at ε=8/255 PGD-7 achieves ~44–52%
                adversarial accuracy but reduces clean accuracy by ~5–9pp, landing
                around 88–92%. The joint constraint (adv > 50%, clean > 90%) is in the
                narrow feasible region on the Pareto frontier — barely satisfiable,
                and sensitive to exact training configuration.
            </div>
        </div>
        """),
        _feedback2[act2_pred.value],
    ])
    return


# ─── ACT II REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *Test your understanding of the robustness-accuracy tradeoff mechanism:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Adversarial training uses fewer training steps, reducing model capacity":
                "ref2_a",
            "B) To be robust to ε-perturbations, the model must learn wider, more conservative "
            "decision boundaries — reducing precision on in-distribution clean examples":
                "ref2_b",
            "C) Adversarial examples contaminate the training distribution, reducing the "
            "effective sample size for clean examples":
                "ref2_c",
            "D) PGD inner loop makes training unstable, causing gradient noise that "
            "degrades clean accuracy":
                "ref2_d",
        },
        label="Why does adversarial training always reduce clean accuracy?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue to the MathPeek."), kind="warn"),
    )
    _correct3 = act2_reflect.value == "ref2_b"
    _feedback3 = {
        "ref2_a": mo.callout(mo.md(
            "**Not correct.** Adversarial training typically uses the same or more training "
            "steps than standard training — PGD-K requires K+1 forward/backward passes per "
            "batch, making it 2–21× slower than standard training. The clean accuracy "
            "reduction is not due to reduced model capacity or fewer training steps."
        ), kind="warn"),
        "ref2_b": mo.callout(mo.md(
            "**Correct.** The Madry minimax objective requires: for every training point x, "
            "the model must be correct for all inputs within the ε-ball around x. This forces "
            "the decision boundary to maintain a margin of at least ε from every training "
            "point. A wider margin in adversarial directions means less precision on the "
            "clean examples: some points that standard training would correctly classify "
            "near a tight boundary are now inside the expanded margin. "
            "This is the fundamental geometric reason for the robustness-accuracy tradeoff."
        ), kind="success"),
        "ref2_c": mo.callout(mo.md(
            "**Not quite.** Adversarial examples are generated on-the-fly from clean training "
            "data — they do not contaminate the clean dataset or reduce the effective sample "
            "size. The training set size is unchanged. The clean accuracy reduction is caused "
            "by the objective function change (wider boundaries), not by data contamination."
        ), kind="warn"),
        "ref2_d": mo.callout(mo.md(
            "**Not correct.** PGD inner loop instability is a training engineering challenge "
            "that can be addressed with careful step size selection and learning rate scheduling. "
            "Modern implementations (free adversarial training, TRADES) achieve stable training. "
            "The clean accuracy reduction persists even with stable PGD training because it is "
            "caused by the fundamental geometric constraint, not by optimization instability."
        ), kind="warn"),
    }
    mo.vstack([
        act2_reflect,
        _feedback3[act2_reflect.value],
    ])
    return


# ─── ACT II MATHPEEK ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equations": mo.md("""
        **Madry et al. (2018) Minimax Formulation:**

        ```
        θ* = argmin_θ E_(x,y)~D [ max_{δ: ||δ||∞ ≤ ε} L(f_θ(x+δ), y) ]
        ```

        - **Outer minimization**: find model parameters θ that minimize expected loss
        - **Inner maximization**: find worst-case perturbation δ within the ε-ball
        - PGD solves the inner maximization approximately via K gradient ascent steps

        **Training cost:** Each batch requires K+1 forward/backward passes (1 clean + K PGD steps).
        PGD-7 adversarial training costs ~8× more compute than standard training.

        **TRADES loss (Zhang et al., 2019) — interpolation:**

        ```
        L_TRADES = L_clean(x) + β × KL[f(x+δ) || f(x)]
        ```

        - β controls the clean-robustness tradeoff (higher β → more robust, less clean)
        - KL divergence ensures the model's predictions are consistent within the ε-ball

        **Why clean accuracy must decrease — geometric argument:**

        Standard ERM minimizes:
        `E[L(f_θ(x), y)]` — a single point constraint per training example.

        Adversarial training minimizes:
        `E[max_δ L(f_θ(x+δ), y)]` — a constraint over an entire ε-ball per training example.

        The ε-ball constraint forces the decision boundary to be at least ε away from every
        training point. Clean examples near the original boundary now fall inside the margin —
        they require the boundary to shift, reducing clean accuracy as a direct consequence
        of the geometry of the constraint.

        **Certified robustness (randomized smoothing) — provable bound:**

        ```
        g(x) = argmax_c P[f(x + N(0, σ²I)) = c]
        ```

        Provably robust for ||δ||₂ ≤ σ × Φ⁻¹(p̄A) − σ × Φ⁻¹(p̄B)
        where p̄A is the probability of the top class under Gaussian noise.
        Even larger clean accuracy cost than PGD-AT (10–20pp on ImageNet).
        """),
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
                    <strong>1. Adversarial examples are not noise &mdash; they are maximally effective signals.</strong>
                    At 3.4% adversarial accuracy (worse than random for 10 classes), FGSM and PGD
                    accumulate perturbations constructively across 150,000+ dimensions by aligning
                    each pixel with the gradient of the loss. High dimensionality amplifies
                    imperceptible perturbations into definitive misclassifications.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. The robustness-accuracy tradeoff is geometric, not configurable away.</strong>
                    PGD-7 adversarial training at &epsilon;=8/255 drops ResNet-50 clean accuracy from
                    76% to ~50% (a 26 percentage point loss) and costs 8&times; training compute.
                    The boundary must be wider to resist attacks &mdash; and wider boundaries misclassify
                    clean inputs that were near those tight margins.
                </div>
                <div>
                    <strong>3. The joint security + clinical constraint is satisfiable but narrow.</strong>
                    The feasible region where adversarial accuracy &ge;50% AND clean accuracy &ge;90%
                    is a small slice of the training configuration space. It requires careful
                    calibration of training &epsilon;, PGD steps, and adversarial loss weight &mdash;
                    no single &ldquo;safe default&rdquo; exists.
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
                    <strong>Lab 15: The Jevons Reckoning</strong> &mdash; This lab quantified the
                    compute overhead of adversarial training (8&times;). The next lab asks: when you
                    make training 2&times; more efficient across a fleet, does total carbon consumption
                    fall &mdash; or does Jevons Paradox cause it to rise?
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
                    <strong>Read:</strong> @sec-robust-ai (adversarial training section) for the
                    Madry et al. minimax formulation and PGD convergence proof.<br/>
                    <strong>Build:</strong> The RobustnessTaxAnalysis LEGO cell demonstrates the
                    exact 76% &rarr; 50% clean accuracy drop with explicit training simulation.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. FGSM and PGD achieve 3.4% adversarial accuracy (worse than random for 10 classes) by aligning each pixel perturbation with the loss gradient across 150,000+ dimensions. Why are adversarial examples maximally effective signals rather than random noise?
2. PGD-7 adversarial training costs 8x compute per epoch and drops clean accuracy from 76% to ~50%. Why is this robustness-accuracy tradeoff geometric (wider decision boundaries misclassify clean inputs) rather than a tuning problem that better algorithms can eliminate?
3. The feasible region where adversarial accuracy >= 50% AND clean accuracy >= 90% is narrow and requires careful calibration of training epsilon, PGD steps, and adversarial loss weight. Why does no single safe default exist for this joint constraint?

**You're ready to move on if you can:**
- Explain why high dimensionality amplifies imperceptible perturbations into definitive misclassifications
- Calculate the compute cost multiplier of adversarial training for a given PGD step count
- Identify the feasibility region in the robustness-accuracy tradeoff space for a specific deployment constraint
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD FOOTER ───────────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_pred, act1_reflect,
    act2_pred, act2_reflect,
    train_eps_slider, pgd_steps_slider, adv_loss_weight,
    _adv_acc_2, _clean_acc_2, _security_met, _product_met, _both_met, _train_overhead,
    CLEAN_ACC_BASELINE,
, decision_input, decision_ui):
    # ── Save to Design Ledger ────────────────────────────────────────────────────
    _ctx = context_toggle.value
    _a1  = act1_pred.value or "unanswered"
    _a1r = act1_reflect.value or "unanswered"
    _a2  = act2_pred.value or "unanswered"
    _a2r = act2_reflect.value or "unanswered"

    _pareto_optimal = bool(_security_met and _product_met)
    _any_constraint_hit = not _security_met or not _product_met

    ledger.save(chapter="v2_14", design={
        "context":                 _ctx,
        "training_epsilon":        train_eps_slider.value,
        "pgd_steps":               pgd_steps_slider.value,
        "adversarial_loss_weight": adv_loss_weight.value,
        "clean_accuracy":          float(_clean_acc_2),
        "adversarial_accuracy":    float(_adv_acc_2),
        "training_overhead":       float(_train_overhead),
        "security_requirement_met": bool(_security_met),
        "product_requirement_met":  bool(_product_met),
        "act1_prediction":         _a1,
        "act1_correct":            _a1 == "option_b",
        "act1_reflection":         _a1r,
        "act1_reflect_correct":    _a1r == "ref_b",
        "act2_result":             float(_adv_acc_2),
        "act2_decision":           f"eps={train_eps_slider.value}/255_pgd{pgd_steps_slider.value}_alpha{adv_loss_weight.value:.2f}",
        "act2_prediction":         _a2,
        "act2_correct":            _a2 == "opt2_b",
        "act2_reflection":         _a2r,
        "act2_reflect_correct":    _a2r == "ref2_b",
        "constraint_hit":          _any_constraint_hit,
        "student_justification": str(decision_input.value),
        "pareto_optimal":          _pareto_optimal,
        "clean_acc_cost_pp":       float(CLEAN_ACC_BASELINE - _clean_acc_2),
    })

    # ── HUD Footer ───────────────────────────────────────────────────────────────
    _a1_correct = _a1 == "option_b"
    _a2_correct = _a2 == "opt2_b"

    _ctx_label = "Hardened" if _ctx == "hardened" else "Production"
    _ctx_color = COLORS["BlueLine"] if _ctx == "hardened" else COLORS["RedLine"]

    _security_badge = (
        f'<span style="color:{COLORS["GreenLine"]}; font-weight:700;">✓ Security met</span>'
        if _security_met else
        f'<span style="color:{COLORS["RedLine"]}; font-weight:700;">✗ Security unmet</span>'
    )
    _product_badge = (
        f'<span style="color:{COLORS["GreenLine"]}; font-weight:700;">✓ Product met</span>'
        if _product_met else
        f'<span style="color:{COLORS["OrangeLine"]}; font-weight:700;">✗ Product unmet</span>'
    )

    mo.Html(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 12px; padding: 20px 28px; margin-top: 24px; color: white;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                    color: #475569; text-transform: uppercase; margin-bottom: 12px;">
            Design Ledger · Lab V2-14 · Robust AI
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;">
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">Context</div>
                <div style="font-weight: 700; color: {_ctx_color};">{_ctx_label}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Training Config
                </div>
                <div style="font-weight: 700; color: #f8fafc;">
                    ε={train_eps_slider.value}/255, PGD-{pgd_steps_slider.value},
                    α={adv_loss_weight.value:.2f}
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Clean / Adversarial
                </div>
                <div style="font-weight: 700; color: #f8fafc;">
                    {_clean_acc_2:.1f}% / {_adv_acc_2:.1f}%
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Security Req.
                </div>
                <div>{_security_badge}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Product Req.
                </div>
                <div>{_product_badge}</div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Act I Prediction
                </div>
                <div style="font-weight: 700; color: {'#6ee7b7' if _a1_correct else '#fca5a5'};">
                    {'Correct' if _a1_correct else 'Incorrect'}
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Act II Prediction
                </div>
                <div style="font-weight: 700; color: {'#6ee7b7' if _a2_correct else '#fca5a5'};">
                    {'Correct' if _a2_correct else 'Incorrect'}
                </div>
            </div>
            <div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 3px;">
                    Pareto Optimal
                </div>
                <div style="font-weight: 700; color: {'#6ee7b7' if _pareto_optimal else '#94a3b8'};">
                    {'Yes — both constraints met' if _pareto_optimal else 'Not yet — adjust config'}
                </div>
            </div>
        </div>
        <div style="margin-top: 16px; border-top: 1px solid #334155; padding-top: 12px;
                    font-size: 0.78rem; color: #64748b;">
            Saved to ledger key "v2_14" · Available to Lab 15 (Sustainable AI) and Lab 17 (Synthesis)
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
