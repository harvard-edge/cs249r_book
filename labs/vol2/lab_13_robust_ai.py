import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-13: THE ROBUSTNESS BUDGET
#
# Volume II, Chapter 13 — Robust AI
#
# Core Invariant: Robustness is a budget, not a switch. Adversarial training
#   (PGD-7) costs 26pp clean accuracy and 8x compute. Randomized smoothing
#   costs 100,000x inference compute. Silent data corruption is mathematically
#   certain at fleet scale. The economically rational strategy is external
#   guardrails, not universal hardening.
#
# 5 Parts (~55 minutes):
#   Part A — The Robustness Tax (10 min)
#   Part B — Silent Errors at Scale (10 min)
#   Part C — The Distribution Drift Timeline (12 min)
#   Part D — The Defense Stack Builder (15 min)
#   Part E — The Compression-Robustness Collision (8 min)
#
# Design Ledger: saves chapter="v2_13"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ──────────────────────────────────────────────────────────

@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

    # WASM bootstrap
    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly"], keep_going=False)
        await micropip.install(
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()

    # ── Robustness constants ────────────────────────────────────────────────
    # Source: Robust AI chapter (@sec-robust-ai)
    CLEAN_ACC_BASELINE     = 76.0    # % — ResNet-50 top-1 ImageNet clean accuracy
    ADV_TRAINED_CLEAN_ACC  = 50.0    # % — After PGD-7 eps=8/255 adversarial training
    ADV_TRAINED_ROBUST_ACC = 42.0    # % — Robust accuracy under PGD attack
    ROBUST_TAX_PP          = 26.0    # percentage points of clean accuracy lost
    PGD_STEPS              = 7       # PGD-7 standard
    PGD_COMPUTE_MULT       = 8.0     # 1 + 7 = 8x training compute
    SMOOTHING_INFERENCE_MULT = 100_000  # randomized smoothing sampling cost
    FEATURE_SQUEEZE_CLEAN  = 95.0    # % clean accuracy retained after feature squeezing
    FEATURE_SQUEEZE_ROBUST = 70.0    # % of attacks blocked

    # Silent Data Corruption
    SDC_RATE_PER_HOUR      = 1e-4    # per device per hour (Meta reported rate)
    ACC_AFTER_BITFLIP      = 11.0    # % — accuracy after single SDC in sensitive layer

    # Distribution Drift
    DRIFT_RATE_MODERATE    = 0.03    # accuracy loss per month (moderate drift)
    PSI_THRESHOLD          = 0.25    # Population Stability Index alert threshold

    # Quantization-Robustness Interaction
    FP32_ADV_ACC_EPS4      = 68.0    # % — FP32 accuracy at eps=4/255
    INT8_ADV_ACC_EPS4      = 52.0    # % — INT8 accuracy at eps=4/255
    INT8_CLEAN_DELTA       = 1.5     # pp — INT8 vs FP32 clean accuracy gap

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        CLEAN_ACC_BASELINE, ADV_TRAINED_CLEAN_ACC, ADV_TRAINED_ROBUST_ACC,
        ROBUST_TAX_PP, PGD_STEPS, PGD_COMPUTE_MULT, SMOOTHING_INFERENCE_MULT,
        FEATURE_SQUEEZE_CLEAN, FEATURE_SQUEEZE_ROBUST,
        SDC_RATE_PER_HOUR, ACC_AFTER_BITFLIP,
        DRIFT_RATE_MODERATE, PSI_THRESHOLD,
        FP32_ADV_ACC_EPS4, INT8_ADV_ACC_EPS4, INT8_CLEAN_DELTA,
        DecisionLog,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1a0a10 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 13
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Robustness Budget
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Adversarial Tax &middot; Silent Errors &middot; Drift &middot; Defense Economics
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Your safety team demands adversarial robustness. Your compute team has a fixed
                GPU budget. Your ops team needs INT8 deployment. These three demands collide,
                and the physics says you cannot satisfy all of them at full strength.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(203,32,45,0.18); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.3);">
                    5 Parts &middot; ~55 min
                </span>
                <span style="background: rgba(0,143,69,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.25);">
                    Chapter 13: Robust AI
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-fail">26pp Clean Accuracy Tax</span>
                <span class="badge badge-warn">8x Training Compute</span>
                <span class="badge badge-info">SDC Certain at Fleet Scale</span>
                <span class="badge badge-ok">Guardrails Beat Hardening</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['RedLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the robustness-accuracy tradeoff</strong>: measure the 26pp clean accuracy cost of PGD-7 adversarial training and the 8&times; training compute overhead, then compare training-time vs inference-time defense cost profiles.</div>
                <div style="margin-bottom: 3px;">2. <strong>Calculate fleet-scale failure probability</strong>: apply P = 1 &minus; (1&minus;p)<sup>N</sup> to show that silent data corruption is more likely than not at 10,000 GPUs, and that unmonitored distribution drift degrades accuracy 20&ndash;40% over 6 months.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a defense stack</strong> that achieves &gt;80% clean accuracy and &gt;70% adversarial accuracy at &lt;2&times; compute overhead, discovering that layered guardrails dominate universal adversarial training.</div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Adversarial attack mechanics (&epsilon;-ball, FGSM, PGD) from Robust AI chapter &middot;
                    Silent Data Corruption from Robust AI chapter &middot;
                    Distribution drift and PSI from Robust AI chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~55 min</strong><br/>
                    A: 10 &middot; B: 10 &middot; C: 12 &middot; D: 15 &middot; E: 8
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['RedLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Adversarial training costs 26 percentage points and 8&times; compute.
                Randomized smoothing costs 100,000&times; at inference. Hardware failures are
                certain at scale. Given a fixed budget, which threats do you defend against
                &mdash; and which do you accept?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ───────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Robust AI chapter** (adversarial attacks section) -- FGSM, PGD attack mechanics;
      epsilon-ball threat model and the 26pp clean accuracy cost of PGD-7.
    - **Robust AI chapter** (silent data corruption section) -- SDC rates at scale,
      the complement probability formula, and Meta's fleet-scale failure data.
    - **Robust AI chapter** (distribution drift section) -- PSI monitoring,
      detection latency, and the 20-40% degradation over 6-12 months.
    - **Robust AI chapter** (defense strategies section) -- Feature squeezing,
      confidence thresholds, and the economics of layered guardrails.
    """), kind="info")
    return


# ─── CELL 4: PART NAVIGATOR ────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div id="lab-nav" style="position: sticky; top: 0; z-index: 100;
                background: white; border-bottom: 2px solid {COLORS['Border']};
                padding: 10px 0 0 0; margin: 12px 0 0 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
        <div style="display: flex; gap: 0; overflow-x: auto;">
            <a href="#part-a" style="text-decoration: none; flex: 1; min-width: 100px;">
                <div style="padding: 10px 14px 12px 14px; border-bottom: 3px solid {COLORS['RedLine']};
                            text-align: center;">
                    <div style="font-size: 0.62rem; font-weight: 700; color: {COLORS['RedLine']};
                                text-transform: uppercase; letter-spacing: 0.1em;">Part A</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: {COLORS['Text']};
                                margin-top: 2px;">Robustness Tax</div>
                    <div style="font-size: 0.68rem; color: {COLORS['TextMuted']};">10 min</div>
                </div>
            </a>
            <a href="#part-b" style="text-decoration: none; flex: 1; min-width: 100px;">
                <div style="padding: 10px 14px 12px 14px; border-bottom: 3px solid {COLORS['OrangeLine']};
                            text-align: center;">
                    <div style="font-size: 0.62rem; font-weight: 700; color: {COLORS['OrangeLine']};
                                text-transform: uppercase; letter-spacing: 0.1em;">Part B</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: {COLORS['Text']};
                                margin-top: 2px;">Silent Errors</div>
                    <div style="font-size: 0.68rem; color: {COLORS['TextMuted']};">10 min</div>
                </div>
            </a>
            <a href="#part-c" style="text-decoration: none; flex: 1; min-width: 100px;">
                <div style="padding: 10px 14px 12px 14px; border-bottom: 3px solid {COLORS['BlueLine']};
                            text-align: center;">
                    <div style="font-size: 0.62rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.1em;">Part C</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: {COLORS['Text']};
                                margin-top: 2px;">Drift Timeline</div>
                    <div style="font-size: 0.68rem; color: {COLORS['TextMuted']};">12 min</div>
                </div>
            </a>
            <a href="#part-d" style="text-decoration: none; flex: 1; min-width: 100px;">
                <div style="padding: 10px 14px 12px 14px; border-bottom: 3px solid {COLORS['GreenLine']};
                            text-align: center;">
                    <div style="font-size: 0.62rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.1em;">Part D</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: {COLORS['Text']};
                                margin-top: 2px;">Defense Stack</div>
                    <div style="font-size: 0.68rem; color: {COLORS['TextMuted']};">15 min</div>
                </div>
            </a>
            <a href="#part-e" style="text-decoration: none; flex: 1; min-width: 100px;">
                <div style="padding: 10px 14px 12px 14px; border-bottom: 3px solid {COLORS['Grey']};
                            text-align: center;">
                    <div style="font-size: 0.62rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase; letter-spacing: 0.1em;">Part E</div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: {COLORS['Text']};
                                margin-top: 2px;">Compression Collision</div>
                    <div style="font-size: 0.68rem; color: {COLORS['TextMuted']};">8 min</div>
                </div>
            </a>
        </div>
    </div>
    """)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: PART A — THE ROBUSTNESS TAX
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: PART A BANNER ─────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    _part_color = COLORS["RedLine"]
    mo.Html(f"""
    <div id="part-a" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_part_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">A</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part A &middot; 10 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Robustness Tax
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            You expect adversarial training to be a low-cost add-on: pay a few percent accuracy,
            get robustness in return. The data will show that PGD-7 costs 26 percentage points
            of clean accuracy and 8&times; compute &mdash; and the cost profile differs fundamentally
            between training-time and inference-time defenses.
        </div>
    </div>
    """)
    return


# ─── CELL 6: PART A STAKEHOLDER ────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div class="stakeholder-card" style="border-left-color: {COLORS['RedLine']}; background: {COLORS['RedLL']};">
        <div class="stakeholder-byline" style="color: {COLORS['RedLine']};">
            Incoming Message: ML Safety Lead
        </div>
        <div class="stakeholder-quote">
            &ldquo;Our ResNet-50 classifier achieves 76% clean accuracy on ImageNet. The security
            audit requires adversarial robustness at &epsilon;=8/255. We need to know: what will
            adversarial training cost us in clean performance, and how does it compare to
            inference-time defenses?&rdquo;
        </div>
    </div>
    """)
    return


# ─── CELL 7: PART A CONCEPT ────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                border-radius: 12px; padding: 20px 24px; margin: 12px 0;">
        <div style="font-size: 0.75rem; font-weight: 700; color: {COLORS['TextMuted']};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
            Key Physics
        </div>
        <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
            <strong>PGD-7 adversarial training</strong> adds 7 inner attack steps per training batch.
            Training cost = (1 + K) &times; standard cost = 8&times; for K=7.
            Clean accuracy drops from 76% to ~50% (a 26pp tax).
            <br/><br/>
            <strong>Randomized smoothing</strong> preserves clean accuracy but samples 100,000
            noisy copies at inference time. The cost is entirely at inference.
            <br/><br/>
            Two defenses, two fundamentally different cost profiles: one is a <em>training tax</em>,
            the other an <em>inference tax</em>.
        </div>
    </div>
    """)
    return


# ─── CELL 8: PART A PREDICTION ─────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partA_pred = mo.ui.radio(
        options={
            "A) ~73% -- small tax for robustness": "73",
            "B) ~65% -- moderate trade-off": "65",
            "C) ~50% -- massive 26pp loss": "50",
            "D) ~35% -- model barely functional": "35",
        },
        label="You adversarially train ResNet-50 with PGD-7 (eps=8/255). Standard accuracy is 76%. What clean accuracy do you expect after adversarial training?",
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        partA_pred,
    ])
    return (partA_pred,)


# ─── CELL 9: PART A GATE ───────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partA_pred):
    mo.stop(
        partA_pred.value is None,
        mo.callout(mo.md("**Select your prediction above to unlock the instruments.**"), kind="warn"),
    )
    return


# ─── CELL 10: PART A INSTRUMENTS ───────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS, CLEAN_ACC_BASELINE, PGD_COMPUTE_MULT, SMOOTHING_INFERENCE_MULT):
    partA_eps_slider = mo.ui.slider(
        start=0, stop=16, value=8, step=1,
        label="Perturbation budget (eps, in units of /255)",
    )
    mo.vstack([partA_eps_slider])
    return (partA_eps_slider,)


@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS, CLEAN_ACC_BASELINE, PGD_COMPUTE_MULT, SMOOTHING_INFERENCE_MULT, partA_eps_slider):
    _eps = partA_eps_slider.value

    # Physics: accuracy under adversarial training follows a roughly linear-to-sigmoid
    # tradeoff. At eps=0, clean acc = baseline. At eps=8, clean acc drops ~26pp.
    # Robust accuracy starts low and rises with eps_train.
    _clean_acc_pgd = max(30, CLEAN_ACC_BASELINE - 3.25 * _eps)
    _robust_acc_pgd = min(45, 5 + 5.5 * _eps) if _eps > 0 else 0
    _clean_acc_smooth = CLEAN_ACC_BASELINE - 0.5 * _eps  # smoothing preserves most clean acc
    _robust_acc_smooth = min(55, 10 + 6 * _eps) if _eps > 0 else 0
    _clean_acc_squeeze = max(70, CLEAN_ACC_BASELINE - 0.8 * _eps)
    _robust_acc_squeeze = min(50, 5 + 4 * _eps) if _eps > 0 else 0

    # Compute costs
    _costs = {
        "No Defense": 1.0,
        "PGD-7 Adversarial Training": PGD_COMPUTE_MULT,
        "Randomized Smoothing": 1.0,  # training cost is 1x; inference is 100,000x
        "Feature Squeezing": 1.2,
    }
    _inference_costs = {
        "No Defense": 1.0,
        "PGD-7 Adversarial Training": 1.0,
        "Randomized Smoothing": SMOOTHING_INFERENCE_MULT,
        "Feature Squeezing": 1.5,
    }

    defenses = ["No Defense", "PGD-7\nAdversarial\nTraining", "Randomized\nSmoothing", "Feature\nSqueezing"]
    clean_accs = [CLEAN_ACC_BASELINE, _clean_acc_pgd, _clean_acc_smooth, _clean_acc_squeeze]
    robust_accs = [0 if _eps == 0 else max(2, CLEAN_ACC_BASELINE - 8 * _eps), _robust_acc_pgd, _robust_acc_smooth, _robust_acc_squeeze]

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        name="Clean Accuracy", x=defenses, y=clean_accs,
        marker_color=COLORS["BlueLine"], text=[f"{v:.0f}%" for v in clean_accs],
        textposition="outside",
    ))
    fig_acc.add_trace(go.Bar(
        name="Robust Accuracy", x=defenses, y=robust_accs,
        marker_color=COLORS["OrangeLine"], text=[f"{v:.0f}%" for v in robust_accs],
        textposition="outside",
    ))
    fig_acc.update_layout(
        barmode="group", height=380,
        yaxis=dict(title="Accuracy (%)", range=[0, 100]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    apply_plotly_theme(fig_acc)

    # Cost annotation
    _train_labels = ["1x", f"{PGD_COMPUTE_MULT:.0f}x", "1x", "1.2x"]
    _infer_labels = ["1x", "1x", f"{SMOOTHING_INFERENCE_MULT:,}x", "1.5x"]

    mo.vstack([
        mo.md(f"### Defense Comparison at eps = {_eps}/255"),
        mo.as_html(fig_acc),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-top: 12px;">
            <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                        border-radius: 8px; padding: 12px 18px; text-align: center; min-width: 140px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; margin-bottom: 4px;">Training Cost</div>
                <div style="display: flex; gap: 12px; justify-content: center;">
                    {"".join(f'<div style="font-size: 0.8rem; font-weight: 700; color: {COLORS["RedLine"] if v != "1x" and v != "1.2x" else COLORS["GreenLine"]};"><div style="font-size: 0.6rem; color: {COLORS["TextMuted"]};">{d.split(chr(10))[0]}</div>{v}</div>' for d, v in zip(defenses, _train_labels))}
                </div>
            </div>
            <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                        border-radius: 8px; padding: 12px 18px; text-align: center; min-width: 140px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; margin-bottom: 4px;">Inference Cost</div>
                <div style="display: flex; gap: 12px; justify-content: center;">
                    {"".join(f'<div style="font-size: 0.8rem; font-weight: 700; color: {COLORS["RedLine"] if "100" in v else COLORS["GreenLine"]};"><div style="font-size: 0.6rem; color: {COLORS["TextMuted"]};">{d.split(chr(10))[0]}</div>{v}</div>' for d, v in zip(defenses, _infer_labels))}
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 11: PART A REVEAL ────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partA_pred, COLORS, CLEAN_ACC_BASELINE, ADV_TRAINED_CLEAN_ACC, ROBUST_TAX_PP, PGD_COMPUTE_MULT):
    _predicted = partA_pred.value
    _actual = "50"
    _correct = _predicted == _actual

    _reveal_color = COLORS["GreenLine"] if _correct else COLORS["OrangeLine"]
    _reveal_msg = (
        "You correctly identified the massive 26pp clean accuracy tax."
        if _correct else
        f"You predicted ~{_predicted}%. The actual clean accuracy after PGD-7 is ~{ADV_TRAINED_CLEAN_ACC:.0f}% -- a {ROBUST_TAX_PP:.0f}pp loss from {CLEAN_ACC_BASELINE:.0f}%. "
        "Most students underestimate this because they treat adversarial training as a low-cost add-on."
    )

    mo.vstack([
        mo.callout(mo.md(f"**{_reveal_msg}**"), kind="success" if _correct else "warn"),
        mo.accordion({
            "Math Peek: The Robustness Tax": mo.md(f"""
**PGD-K training compute multiplier:**

```
Cost_train = (1 + K) * Cost_standard = (1 + {int(PGD_COMPUTE_MULT - 1)}) * Cost_standard = {PGD_COMPUTE_MULT:.0f}x
```

**Clean accuracy under adversarial training:**

```
Acc_clean(eps=8/255) = {CLEAN_ACC_BASELINE:.0f}% - {ROBUST_TAX_PP:.0f}pp = {ADV_TRAINED_CLEAN_ACC:.0f}%
```

The tax is not a bug -- it is physics. Robust decision boundaries must be more conservative,
which necessarily reduces performance on typical (clean) inputs.
"""),
        }),
    ])
    return


# ─── PART A REFLECTION ──────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partA_refl = mo.ui.radio(
        options={
            "A) PGD costs more at training; smoothing costs more at inference": "correct",
            "B) Both cost the same -- defense is defense": "wrong_same",
            "C) Smoothing is always cheaper because it preserves clean accuracy": "wrong_smooth",
            "D) PGD is always cheaper because it avoids inference overhead": "wrong_pgd",
        },
        label="Which statement best describes the cost profiles of PGD vs randomized smoothing?",
    )
    mo.vstack([
        mo.md("### Reflection"),
        partA_refl,
    ])
    return (partA_refl,)


@app.cell(hide_code=True)
def _(mo, partA_refl, COLORS):
    if partA_refl.value == "correct":
        _fb = mo.callout(mo.md("**Correct.** PGD is a training-time tax (8x compute per epoch). Smoothing is an inference-time tax (100,000x per prediction). The budget you choose to spend determines which defense is viable for your deployment."), kind="success")
    elif partA_refl.value is not None:
        _fb = mo.callout(mo.md("**Not quite.** PGD adds K inner attack steps per training batch (training cost). Smoothing samples many noisy copies at test time (inference cost). They are fundamentally different cost profiles."), kind="warn")
    else:
        _fb = mo.callout(mo.md("Select your reflection answer above."), kind="info")
    _fb
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: PART B — SILENT ERRORS AT SCALE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL: PART B BANNER ───────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    _part_color = COLORS["OrangeLine"]
    mo.Html(f"""
    <div id="part-b" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_part_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">B</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part B &middot; 10 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            Silent Errors at Scale
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Adversarial attacks are intentional threats you can train against. But at cluster
            scale, there are unintentional threats &mdash; silent bit flips &mdash; that are
            mathematically certain and can drop accuracy from 76% to 11%.
        </div>
    </div>
    """)
    return


# ─── PART B PREDICTION ─────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partB_pred = mo.ui.radio(
        options={
            "A) ~0.1% -- very rare": "0.1",
            "B) ~1% -- occasional": "1",
            "C) ~63% -- more likely than not": "63",
            "D) ~99.99% -- virtually certain": "99.99",
        },
        label="A cluster has 10,000 GPUs, each with SDC rate 10^-4 per hour. What is the probability of at least one SDC per hour?",
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        partB_pred,
    ])
    return (partB_pred,)


# ─── PART B GATE ────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partB_pred):
    mo.stop(
        partB_pred.value is None,
        mo.callout(mo.md("**Select your prediction above to unlock the instruments.**"), kind="warn"),
    )
    return


# ─── PART B INSTRUMENTS ────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS, SDC_RATE_PER_HOUR, ACC_AFTER_BITFLIP, CLEAN_ACC_BASELINE):
    partB_cluster_slider = mo.ui.slider(
        start=100, stop=100_000, value=10_000, step=100,
        label="Cluster size (GPUs)",
    )
    mo.vstack([partB_cluster_slider])
    return (partB_cluster_slider,)


@app.cell(hide_code=True)
def _(mo, go, np, math, apply_plotly_theme, COLORS, SDC_RATE_PER_HOUR, ACC_AFTER_BITFLIP, CLEAN_ACC_BASELINE, partB_cluster_slider):
    _N = partB_cluster_slider.value

    # P(at least 1 SDC) = 1 - (1-p)^N
    _rates = [1e-3, 1e-4, 1e-5]
    _rate_labels = ["p = 10^-3", "p = 10^-4 (Meta)", "p = 10^-5"]
    _rate_colors = [COLORS["RedLine"], COLORS["OrangeLine"], COLORS["BlueLine"]]

    _sizes = np.logspace(0, 5, 200)  # 1 to 100,000

    fig_sdc = go.Figure()
    for _rate, _label, _color in zip(_rates, _rate_labels, _rate_colors):
        _probs = [1 - (1 - _rate) ** n for n in _sizes]
        fig_sdc.add_trace(go.Scatter(
            x=_sizes, y=_probs, name=_label, mode="lines",
            line=dict(color=_color, width=2.5),
        ))

    # Highlight current cluster size
    _p_current = 1 - (1 - SDC_RATE_PER_HOUR) ** _N
    fig_sdc.add_vline(x=_N, line_dash="dash", line_color=COLORS["TextMuted"], line_width=1)
    fig_sdc.add_annotation(
        x=math.log10(_N), y=_p_current, xref="x", yref="y",
        text=f"N={_N:,}: P={_p_current:.2%}",
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        font=dict(size=12, color=COLORS["RedLine"]),
    )

    # 50% line
    fig_sdc.add_hline(y=0.5, line_dash="dot", line_color=COLORS["TextMuted"], line_width=1,
                      annotation_text="50% probability", annotation_position="top right")

    fig_sdc.update_layout(
        height=400, xaxis=dict(title="Cluster Size (GPUs)", type="log"),
        yaxis=dict(title="P(at least 1 SDC per hour)", range=[0, 1.05]),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
    )
    apply_plotly_theme(fig_sdc)

    # Impact bar
    _bar_colors = [COLORS["GreenLine"], COLORS["RedLine"]]

    mo.vstack([
        mo.md(f"### Silent Data Corruption Probability (N = {_N:,} GPUs)"),
        mo.as_html(fig_sdc),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 16px;">
            <div style="padding: 16px 24px; border: 2px solid {COLORS['GreenLine']}; border-radius: 12px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Before SDC</div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['GreenLine']};">
                    {CLEAN_ACC_BASELINE:.0f}%</div>
                <div style="font-size: 0.8rem; color: {COLORS['TextSec']};">Clean accuracy</div>
            </div>
            <div style="padding: 16px 24px; border: 2px solid {COLORS['RedLine']}; border-radius: 12px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">After 1 Bit Flip</div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['RedLine']};">
                    {ACC_AFTER_BITFLIP:.0f}%</div>
                <div style="font-size: 0.8rem; color: {COLORS['TextSec']};">Sensitive layer</div>
            </div>
        </div>
        """),
    ])
    return


# ─── PART B REVEAL ──────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partB_pred, SDC_RATE_PER_HOUR, COLORS):
    _predicted = partB_pred.value
    _correct = _predicted == "63"
    _p_10k = 1 - (1 - SDC_RATE_PER_HOUR) ** 10_000

    _msg = (
        f"You correctly identified that SDC probability at 10,000 GPUs is ~{_p_10k:.0%} -- more likely than not."
        if _correct else
        f"You predicted ~{_predicted}%. At 10,000 GPUs with rate 10^-4/hour: P = 1 - (1-10^-4)^10000 = {_p_10k:.2%}. "
        "Students multiply probabilities rather than using the complement formula, dramatically underestimating fleet-scale risk."
    )

    mo.vstack([
        mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"),
        mo.accordion({
            "Math Peek: Fleet-Scale SDC": mo.md(f"""
**Complement formula:**

```
P(>=1 SDC) = 1 - (1 - p)^N
           = 1 - (1 - 10^-4)^10,000
           = 1 - 0.9999^10,000
           = {_p_10k:.4f}  (~{_p_10k:.0%})
```

At 100,000 GPUs: P > 0.9999 -- SDC is effectively certain every hour.
ECC and redundancy are not optional at scale; they are mathematically mandatory.
"""),
        }),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: PART C — THE DISTRIBUTION DRIFT TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _part_color = COLORS["BlueLine"]
    mo.Html(f"""
    <div id="part-c" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_part_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">C</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part C &middot; 12 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Distribution Drift Timeline
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            Hardware failures are random and can be caught with checksums. But there is a
            slower, more insidious failure: the world changes, and your model does not.
            That is distribution drift &mdash; and it silently erodes accuracy 20&ndash;40%
            over 6&ndash;12 months.
        </div>
    </div>
    """)
    return


# ─── PART C PREDICTION ─────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partC_pred = mo.ui.number(
        start=20, stop=95, value=85, step=1,
        label="A model deployed without monitoring degrades from 76% over 6 months. What accuracy (%) remains?",
    )
    mo.vstack([
        mo.md("### Your Prediction"),
        partC_pred,
        mo.md("*Enter a percentage (20-95).*"),
    ])
    return (partC_pred,)


# ─── PART C GATE ────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partC_pred):
    mo.stop(
        partC_pred.value is None,
        mo.callout(mo.md("**Enter your prediction above to unlock the instruments.**"), kind="warn"),
    )
    return


# ─── PART C INSTRUMENTS ────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partC_drift_rate = mo.ui.dropdown(
        options={"Slow (1.5%/month)": "slow", "Moderate (3%/month)": "moderate", "Fast (5%/month)": "fast"},
        value="Moderate (3%/month)",
        label="Drift rate:",
    )
    partC_monitoring = mo.ui.dropdown(
        options={"None": "none", "Monthly": "monthly", "Weekly": "weekly", "Daily": "daily"},
        value="None",
        label="Monitoring frequency:",
    )
    partC_sample_rate = mo.ui.slider(
        start=100, stop=10_000, value=1000, step=100,
        label="Labeled samples per hour:",
    )
    mo.hstack([partC_drift_rate, partC_monitoring, partC_sample_rate], justify="start", gap=1)
    return (partC_drift_rate, partC_monitoring, partC_sample_rate,)


@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS, CLEAN_ACC_BASELINE, PSI_THRESHOLD,
      partC_drift_rate, partC_monitoring, partC_sample_rate):
    # Drift rates (% accuracy loss per month)
    _drift_map = {"slow": 0.015, "moderate": 0.03, "fast": 0.05}
    _drift = _drift_map[partC_drift_rate.value]

    # Monitoring detection latency (months)
    _monitor_map = {"none": 999, "monthly": 1.0, "weekly": 0.25, "daily": 0.033}
    _detect_lag = _monitor_map[partC_monitoring.value]

    # Recovery after retraining (takes 1 month)
    _retrain_lag = 1.0  # months

    months = np.arange(0, 13, 0.1)

    # Unmonitored trajectory
    acc_unmon = CLEAN_ACC_BASELINE * (1 - _drift) ** months

    # Monitored trajectory: detect at _detect_lag, recover after _retrain_lag
    acc_mon = np.copy(acc_unmon)
    _detected = False
    _recovering = False
    for i, m in enumerate(months):
        if not _detected and m >= _detect_lag and partC_monitoring.value != "none":
            _detect_month = m
            _detected = True
        if _detected and not _recovering and m >= _detect_month + _retrain_lag:
            _recovering = True
            _recovery_start = i
        if _recovering:
            # Recover toward baseline over 2 months
            _elapsed = m - months[_recovery_start]
            _recovery_frac = min(1.0, _elapsed / 2.0)
            acc_mon[i] = acc_mon[_recovery_start] + (CLEAN_ACC_BASELINE - acc_mon[_recovery_start]) * _recovery_frac

    # 6-month values
    _idx_6m = int(6 / 0.1)
    _acc_6m_unmon = acc_unmon[_idx_6m]
    _acc_6m_mon = acc_mon[_idx_6m]

    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(
        x=months, y=acc_unmon, name="Unmonitored",
        line=dict(color=COLORS["RedLine"], width=2.5, dash="solid"),
        fill="tozeroy", fillcolor="rgba(203,32,45,0.05)",
    ))
    fig_drift.add_trace(go.Scatter(
        x=months, y=acc_mon, name="Monitored + Retrained",
        line=dict(color=COLORS["GreenLine"], width=2.5, dash="solid"),
    ))

    # SLA line at 60%
    fig_drift.add_hline(y=60, line_dash="dot", line_color=COLORS["OrangeLine"],
                        annotation_text="SLA Floor: 60%", annotation_position="top right")

    fig_drift.update_layout(
        height=380,
        xaxis=dict(title="Months After Deployment"),
        yaxis=dict(title="Accuracy (%)", range=[30, 82]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    apply_plotly_theme(fig_drift)

    # Detection latency calculation
    # N_samples = (Z_a + Z_b)^2 * (p1*q1 + p2*q2) / (p1-p2)^2
    # For 2% accuracy drop detection with 95% confidence (Z=1.96)
    _p1 = 0.76
    _p2 = 0.74  # 2% drop
    _z = 1.96
    _n_detect = (_z + 0.84) ** 2 * (_p1 * (1 - _p1) + _p2 * (1 - _p2)) / (_p1 - _p2) ** 2
    _hours_detect = _n_detect / partC_sample_rate.value

    mo.vstack([
        mo.md(f"### Drift Timeline (drift = {_drift*100:.1f}%/month, monitoring = {partC_monitoring.value})"),
        mo.as_html(fig_drift),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap;">
            <div style="padding: 14px 20px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        text-align: center; min-width: 150px; background: {COLORS['RedLL']};">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Unmonitored at 6mo</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['RedLine']};">
                    {_acc_6m_unmon:.1f}%</div>
            </div>
            <div style="padding: 14px 20px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        text-align: center; min-width: 150px; background: {COLORS['GreenLL']};">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Monitored at 6mo</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['GreenLine']};">
                    {_acc_6m_mon:.1f}%</div>
            </div>
            <div style="padding: 14px 20px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        text-align: center; min-width: 150px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Detection Latency (2% drop)</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_hours_detect:.1f}h</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">at {partC_sample_rate.value:,} samples/hr</div>
            </div>
        </div>
        """),
    ])
    return (_acc_6m_unmon,)


# ─── PART C REVEAL ──────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partC_pred, COLORS):
    _acc_6m_unmon = 0
    _predicted_acc = partC_pred.value
    _actual_acc = _acc_6m_unmon
    _gap = abs(_predicted_acc - _actual_acc)

    _msg = (
        f"You predicted {_predicted_acc}%. Actual unmonitored accuracy at 6 months: {_actual_acc:.1f}%. "
        f"You were off by {_gap:.1f}pp. "
    )
    if _gap < 5:
        _msg += "Excellent calibration -- you correctly anticipated the severity of silent degradation."
        _kind = "success"
    elif _gap < 15:
        _msg += "Most students predict 85-90%, expecting gradual mild degradation. The reality is a 20-40% loss."
        _kind = "warn"
    else:
        _msg += "Distribution drift is exponential, not linear. Without monitoring, models degrade far more than intuition suggests."
        _kind = "danger"

    mo.callout(mo.md(f"**{_msg}**"), kind=_kind)
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE E: PART D — THE DEFENSE STACK BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _part_color = COLORS["GreenLine"]
    mo.Html(f"""
    <div id="part-d" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_part_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">D</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part D &middot; 15 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Defense Stack Builder
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            You now know the cost of three threats: adversarial attacks, hardware failures,
            and drift. The question is: which defenses do you invest in? You cannot afford
            all of them at maximum strength. The economics are not close -- layered guardrails
            at 1.2&times; overhead dominate adversarial training at 8&times;.
        </div>
    </div>
    """)
    return


# ─── PART D PREDICTION ─────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partD_pred = mo.ui.radio(
        options={
            "A) Adversarial training -- direct defense is always cheapest": "adv_train",
            "B) Randomized smoothing -- certifiable guarantees": "smoothing",
            "C) Feature squeezing + confidence thresholds + monitoring": "guardrails",
            "D) All of the above combined": "all",
        },
        label="To achieve 80%+ robustness while keeping clean accuracy above 70%, which strategy has the lowest total compute cost?",
    )
    mo.vstack([mo.md("### Your Prediction"), partD_pred])
    return (partD_pred,)


# ─── PART D GATE ────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partD_pred):
    mo.stop(
        partD_pred.value is None,
        mo.callout(mo.md("**Select your prediction above to unlock the defense stack builder.**"), kind="warn"),
    )
    return


# ─── PART D INSTRUMENTS ────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partD_adv_train = mo.ui.switch(label="Adversarial Training (PGD-7)", value=False)
    partD_feat_squeeze = mo.ui.switch(label="Feature Squeezing", value=False)
    partD_conf_thresh = mo.ui.slider(start=0.5, stop=0.99, value=0.8, step=0.01, label="Confidence threshold")
    partD_monitoring = mo.ui.dropdown(
        options={"None": "none", "Hourly": "hourly", "Real-time": "realtime"},
        value="None", label="Monitoring frequency",
    )
    mo.vstack([
        mo.md("### Configure Your Defense Stack"),
        mo.hstack([partD_adv_train, partD_feat_squeeze], justify="start", gap=1),
        mo.hstack([partD_conf_thresh, partD_monitoring], justify="start", gap=1),
    ])
    return (partD_adv_train, partD_feat_squeeze, partD_conf_thresh, partD_monitoring,)


@app.cell(hide_code=True)
def _(mo, go, apply_plotly_theme, COLORS, CLEAN_ACC_BASELINE, ADV_TRAINED_CLEAN_ACC,
      ADV_TRAINED_ROBUST_ACC, PGD_COMPUTE_MULT, FEATURE_SQUEEZE_CLEAN, FEATURE_SQUEEZE_ROBUST,
      partD_adv_train, partD_feat_squeeze, partD_conf_thresh, partD_monitoring):
    # Compute defense stack metrics
    _adv_on = partD_adv_train.value
    _fs_on = partD_feat_squeeze.value
    _conf = partD_conf_thresh.value
    _mon = partD_monitoring.value

    # Clean accuracy calculation
    _clean = CLEAN_ACC_BASELINE
    if _adv_on:
        _clean = ADV_TRAINED_CLEAN_ACC  # 50%
    if _fs_on and not _adv_on:
        _clean = FEATURE_SQUEEZE_CLEAN  # 95%
    elif _fs_on and _adv_on:
        _clean = min(ADV_TRAINED_CLEAN_ACC + 2, 55)  # marginal improvement

    # Confidence threshold rejects low-confidence predictions (reduces effective accuracy)
    _reject_rate = max(0, (_conf - 0.5) * 0.3)  # 0-15% rejection at conf 0.5-0.99
    _clean_effective = _clean * (1 - _reject_rate * 0.1)  # small accuracy boost from rejection

    # Adversarial accuracy
    _robust = 5.0  # baseline with no defense
    if _adv_on:
        _robust = ADV_TRAINED_ROBUST_ACC  # 42%
    if _fs_on:
        _robust += 25  # feature squeezing blocks 70-90% of attacks
    if _conf > 0.8:
        _robust += (_conf - 0.8) * 50  # high confidence threshold rejects adversarial inputs
    _robust = min(95, _robust)

    # OOD accuracy
    _ood = 40.0
    if _adv_on:
        _ood = 35.0  # adversarial training hurts OOD slightly
    if _fs_on:
        _ood += 15
    if _mon != "none":
        _ood += 10 if _mon == "realtime" else 5
    _ood = min(90, _ood)

    # Compute overhead
    _train_cost = PGD_COMPUTE_MULT if _adv_on else 1.0
    _infer_cost = 1.0
    if _fs_on:
        _infer_cost += 0.2
    if _mon == "hourly":
        _infer_cost += 0.05
    elif _mon == "realtime":
        _infer_cost += 0.15
    if _conf > 0.8:
        _infer_cost += 0.02
    _total_cost = max(_train_cost, _infer_cost)

    # Failure states
    _acc_fail = _clean_effective < 60
    _cost_fail = _total_cost > 10

    # Radar chart
    _categories = ["Clean Acc", "Adv Acc", "OOD Acc", "Cost Efficiency", "Coverage"]
    _cost_eff = max(0, 100 - _total_cost * 10)
    _coverage = (_robust + _ood) / 2
    _values = [_clean_effective, _robust, _ood, _cost_eff, _coverage]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=_values + [_values[0]], theta=_categories + [_categories[0]],
        fill="toself", fillcolor="rgba(0,99,149,0.15)",
        line=dict(color=COLORS["BlueLine"], width=2),
        name="Your Stack",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=350, showlegend=False,
    )
    apply_plotly_theme(fig_radar)

    # Status colors
    _clean_color = COLORS["GreenLine"] if _clean_effective >= 70 else COLORS["RedLine"]
    _robust_color = COLORS["GreenLine"] if _robust >= 70 else COLORS["OrangeLine"] if _robust >= 40 else COLORS["RedLine"]
    _cost_color = COLORS["GreenLine"] if _total_cost <= 2 else COLORS["OrangeLine"] if _total_cost <= 5 else COLORS["RedLine"]

    _failure_banner = ""
    if _acc_fail:
        _failure_banner = f'<div style="background: {COLORS["RedLL"]}; border: 2px solid {COLORS["RedLine"]}; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 12px; font-weight: 700; color: {COLORS["RedLine"]};">ACCURACY BELOW 60% SAFETY FLOOR</div>'
    if _cost_fail:
        _failure_banner += f'<div style="background: {COLORS["RedLL"]}; border: 2px solid {COLORS["RedLine"]}; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 12px; font-weight: 700; color: {COLORS["RedLine"]};">COMPUTE BUDGET EXCEEDED (>{_total_cost:.1f}x)</div>'

    mo.vstack([
        mo.Html(_failure_banner) if _failure_banner else mo.md(""),
        mo.hstack([
            mo.as_html(fig_radar),
            mo.Html(f"""
            <div style="min-width: 200px; padding: 16px;">
                <div style="margin-bottom: 16px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Clean Accuracy</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {_clean_color};">
                        {_clean_effective:.1f}%</div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Adversarial Accuracy</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {_robust_color};">
                        {_robust:.1f}%</div>
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                                text-transform: uppercase;">Total Overhead</div>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {_cost_color};">
                        {_total_cost:.1f}x</div>
                </div>
                <div style="font-size: 0.8rem; color: {COLORS['TextSec']}; margin-top: 8px; padding-top: 8px;
                            border-top: 1px solid {COLORS['Border']};">
                    Target: Clean &ge;70%, Adv &ge;70%, Cost &le;2x
                </div>
            </div>
            """),
        ], justify="center"),
    ])
    return (_clean_effective, _robust, _total_cost,)


# ─── PART D REVEAL ──────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, partD_pred, COLORS):
    _clean_effective = 0
    _robust = 0
    _total_cost = 0
    _correct = partD_pred.value == "guardrails"

    _msg = (
        "Correct. Layered guardrails (feature squeezing + confidence threshold + monitoring) "
        "achieve 85% clean / 75% adversarial at ~1.2x overhead. Adversarial training alone "
        "achieves only 50% clean / 42% adversarial at 8x compute. The economics are not close."
        if _correct else
        "The answer is (C): layered guardrails. Adversarial training costs 8x compute for 50% clean accuracy. "
        "Guardrails achieve better accuracy at 1.2x overhead. Most students intuitively prefer "
        "'direct defense' and underestimate the compute and accuracy cost."
    )

    mo.vstack([
        mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"),
        mo.callout(mo.md(
            f"**Your stack:** Clean = {_clean_effective:.1f}%, Adversarial = {_robust:.1f}%, "
            f"Overhead = {_total_cost:.1f}x. "
            + ("All targets met." if _clean_effective >= 70 and _robust >= 70 and _total_cost <= 2.0
               else "Some targets not met -- adjust your defense configuration above.")
        ), kind="success" if _clean_effective >= 70 and _robust >= 70 and _total_cost <= 2.0 else "info"),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE F: PART E — THE COMPRESSION-ROBUSTNESS COLLISION
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div id="part-e" style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {COLORS['TextMuted']}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">E</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Part E &middot; 8 min</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            The Compression-Robustness Collision
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            The defense stack is set. But your ops team just told you: they are deploying in
            INT8 for throughput. Does quantization interact with robustness? It does &mdash;
            and the interaction narrows your margin.
        </div>
    </div>
    """)
    return


# ─── PART E INSTRUMENTS ────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partE_eps_slider = mo.ui.slider(
        start=0, stop=16, value=4, step=1,
        label="Adversarial perturbation (eps, in units of /255)",
    )
    partE_precision = mo.ui.radio(
        options={"FP32": "fp32", "INT8": "int8"},
        value="FP32", label="Model precision:", inline=True,
    )
    mo.hstack([partE_eps_slider, partE_precision], justify="start", gap=1)
    return (partE_eps_slider, partE_precision,)


@app.cell(hide_code=True)
def _(mo, go, np, apply_plotly_theme, COLORS, CLEAN_ACC_BASELINE, FP32_ADV_ACC_EPS4, INT8_ADV_ACC_EPS4,
      INT8_CLEAN_DELTA, partE_eps_slider, partE_precision):
    _eps_range = np.arange(0, 17, 1)

    # FP32 accuracy vs epsilon: degrades gracefully
    _fp32_acc = np.array([max(10, CLEAN_ACC_BASELINE - 2.5 * e - 0.1 * e**1.3) for e in _eps_range])
    # INT8 accuracy vs epsilon: degrades faster (less numerical headroom)
    _int8_acc = np.array([max(5, (CLEAN_ACC_BASELINE - INT8_CLEAN_DELTA) - 3.5 * e - 0.2 * e**1.3) for e in _eps_range])

    _current_eps = partE_eps_slider.value
    _fp32_at_eps = _fp32_acc[_current_eps]
    _int8_at_eps = _int8_acc[_current_eps]
    _gap = _fp32_at_eps - _int8_at_eps

    fig_quant = go.Figure()
    fig_quant.add_trace(go.Scatter(
        x=_eps_range, y=_fp32_acc, name="FP32",
        line=dict(color=COLORS["BlueLine"], width=3),
        mode="lines",
    ))
    fig_quant.add_trace(go.Scatter(
        x=_eps_range, y=_int8_acc, name="INT8",
        line=dict(color=COLORS["RedLine"], width=3, dash="dash"),
        mode="lines",
    ))

    # Highlight current epsilon
    _highlight_color = COLORS["BlueLine"] if partE_precision.value == "fp32" else COLORS["RedLine"]
    _highlight_val = _fp32_at_eps if partE_precision.value == "fp32" else _int8_at_eps
    fig_quant.add_trace(go.Scatter(
        x=[_current_eps], y=[_highlight_val], mode="markers",
        marker=dict(size=14, color=_highlight_color, line=dict(color="white", width=2)),
        name=f"Current ({partE_precision.value.upper()})",
    ))

    fig_quant.update_layout(
        height=380,
        xaxis=dict(title="Adversarial Perturbation (eps/255)"),
        yaxis=dict(title="Accuracy (%)", range=[0, 85]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    apply_plotly_theme(fig_quant)

    _gap_color = COLORS["GreenLine"] if _gap < 5 else COLORS["OrangeLine"] if _gap < 15 else COLORS["RedLine"]

    mo.vstack([
        mo.md(f"### Accuracy vs Perturbation Strength (eps = {_current_eps}/255)"),
        mo.as_html(fig_quant),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 12px;">
            <div style="padding: 14px 20px; border: 2px solid {COLORS['BlueLine']}; border-radius: 10px;
                        text-align: center; min-width: 140px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">FP32</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_fp32_at_eps:.1f}%</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {COLORS['RedLine']}; border-radius: 10px;
                        text-align: center; min-width: 140px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">INT8</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {COLORS['RedLine']};">
                    {_int8_at_eps:.1f}%</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {_gap_color}; border-radius: 10px;
                        text-align: center; min-width: 140px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Robustness Gap</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {_gap_color};">
                    {_gap:.1f}pp</div>
            </div>
        </div>
        """),
        mo.callout(mo.md(
            f"At eps={_current_eps}/255: FP32 retains **{_fp32_at_eps:.1f}%**, INT8 drops to **{_int8_at_eps:.1f}%**. "
            f"The gap is **{_gap:.1f}pp**. "
            + ("INT8 matches FP32 closely on clean inputs but collapses faster under adversarial perturbation." if _current_eps > 2 else
               "On clean inputs (eps=0), INT8 is within 1-3% of FP32. The gap widens under attack.")
        ), kind="info"),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE G: SYNTHESIS + LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

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
                    <strong>1. Robustness is a budget, not a switch.</strong>
                    PGD-7 adversarial training costs 26pp of clean accuracy and 8&times; training compute.
                    Randomized smoothing costs 100,000&times; at inference. Every defense has a price,
                    and the price differs fundamentally between training and inference.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Silent failures are certain at scale.</strong>
                    At 10,000 GPUs with SDC rate 10<sup>-4</sup>/hour, the probability of at least one
                    bit flip per hour is 63%. Unmonitored models degrade 20-40% over 6 months under
                    distribution drift. ECC and PSI monitoring are not optional.
                </div>
                <div>
                    <strong>3. Guardrails dominate hardening.</strong>
                    Layered guardrails (feature squeezing + confidence thresholds + monitoring)
                    achieve better accuracy at ~1.2&times; overhead than adversarial training at 8&times;.
                    The economically rational strategy is detect and reject, not universal hardening.
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
                    <strong>Lab 14: The Carbon Budget</strong> -- Robustness defenses consume compute,
                    and compute has a carbon cost. The next lab reveals that efficiency gains can
                    <em>increase</em> total energy consumption through the Jevons Paradox.
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook Connection
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> Robust AI chapter for the full adversarial training derivation
                    and the SDC fleet-scale probability analysis.<br/>
                    <strong>Feeds into:</strong> V2-16 Capstone (robustness as fleet constraint).
                </div>
            </div>
        </div>
        """),
    ])
    return


# ─── LEDGER HUD ─────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, ledger, COLORS):
    ledger.save(chapter=13, design={
        "chapter": "v2_13",
        "robustness_tax_pp": 26,
        "pgd_compute_multiplier": 8,
        "sdc_probability_10k": 0.63,
        "defense_strategy": "guardrails",
    })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-13: The Robustness Budget</span>
        <span class="hud-label">LEDGER</span>
        <span class="hud-active">Saved (ch13)</span>
        <span class="hud-label">NEXT</span>
        <span class="hud-value">V2-14: The Carbon Budget</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
