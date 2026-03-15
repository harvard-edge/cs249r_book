import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 09: THE DATA SELECTION TRADEOFF
#
# Chapter: Data Selection (@sec-data-selection)
# Core Invariant:
#   Every data selection decision is a Pareto tradeoff — higher quality data
#   costs exponentially more to curate, but diminishing returns set in fast.
#   The optimal curriculum orders examples by difficulty, not randomly.
#
# 2 Contexts: Cloud (H100, 80 GB) vs Edge (Jetson Orin NX, 16 GB)
#
# Act I  — The Selection Cost Blindspot (12-15 min)
#   Prediction: Curriculum learning speedup vs random baseline
#   Instrument: Training efficiency curve by selection method
#   Reflection: Why easy-first builds stable gradients
#
# Act II — The Pareto Curve (FIRST INTRODUCTION in this curriculum) (20-25 min)
#   Prediction: Optimal annotation strategy under budget
#   Instrument: Pareto frontier — annotation cost vs validation accuracy
#   Failure state: Budget insufficient for minimum viable quality
#   Reflection: Active learning information gain per dollar
#
# Design Ledger: chapter=9, context, selection_strategy, annotation_budget_k,
#                act1_prediction, act1_correct, act2_result, act2_decision,
#                constraint_hit, pareto_optimal
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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

    # ── Hardware constants (source: NVIDIA spec sheets; Jetson Orin NX datasheet) ──
    H100_RAM_GB      = 80      # H100 SXM5 HBM3e memory capacity, GB
    H100_BW_GBS      = 3350    # H100 SXM5 HBM3e memory bandwidth, GB/s
    H100_TFLOPS_FP16 = 989     # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_TDP_W       = 700     # H100 TDP, Watts

    ORIN_RAM_GB      = 16      # Jetson Orin NX unified memory, GB
    ORIN_BW_GBS      = 102     # Jetson Orin NX memory bandwidth, GB/s
    ORIN_TFLOPS      = 100     # Jetson Orin NX INT8 equivalent TFLOPS
    ORIN_TDP_W       = 25      # Jetson Orin NX TDP, Watts

    # ── Curriculum learning constants (source: data_selection.qmd) ──
    # Curriculum speedup range: 2–3× over random baseline
    # Source: Bengio et al. 2009; confirmed in data_selection.qmd §Curriculum Learning
    CURRICULUM_MIN_SPEEDUP  = 2.0   # lower bound 2× speedup (data_selection.qmd)
    CURRICULUM_MAX_SPEEDUP  = 3.0   # upper bound 3× speedup (data_selection.qmd)
    HARD_FIRST_PENALTY      = 0.7   # hard-first is ~30% slower than random (data_selection.qmd)

    # ── Annotation cost constants (source: data_selection.qmd §Active Learning) ──
    # Annotation cost: $0.05/image for crowd-sourced (MTurk-class)
    # Active learning selects most informative 20% → Pareto optimal
    # Minimum viable annotation budget (edge deployment): $5,000
    ANNOTATION_COST_PER_IMAGE = 0.05   # USD per labeled image (data_selection.qmd)
    MIN_VIABLE_BUDGET_EDGE_K  = 5.0    # $5K minimum for edge-viable model quality
    MIN_VIABLE_BUDGET_CLOUD_K = 10.0   # $10K minimum for cloud-viable model quality

    ledger = DesignLedger()
    return (
        mo, go, np, math,
        ledger, COLORS, LAB_CSS, apply_plotly_theme,
        H100_RAM_GB, H100_BW_GBS, H100_TFLOPS_FP16, H100_TDP_W,
        ORIN_RAM_GB, ORIN_BW_GBS, ORIN_TFLOPS, ORIN_TDP_W,
        CURRICULUM_MIN_SPEEDUP, CURRICULUM_MAX_SPEEDUP, HARD_FIRST_PENALTY,
        ANNOTATION_COST_PER_IMAGE, MIN_VIABLE_BUDGET_EDGE_K, MIN_VIABLE_BUDGET_CLOUD_K,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 09
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Data Selection Tradeoff
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                Training on
                <strong style="color:#f8fafc;">more data</strong>
                is not the same as training on
                <strong style="color:#f8fafc;">better-ordered data</strong>.
                Curriculum learning delivers 2&ndash;3&times; speedups.
                Annotation budget is a Pareto frontier problem, not a linear one.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I: Selection Cost Blindspot &middot; Act II: Pareto Curve (first introduction)
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Budget failure state active
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
                <div style="margin-bottom: 3px;">1. <strong>Predict the curriculum learning speedup over random ordering</strong> — determine whether a 2&ndash;3&times; convergence speedup is plausible and identify which ordering strategy (easy-first, hard-first, random) is fastest.</div>
                <div style="margin-bottom: 3px;">2. <strong>Compare annotation strategies on the Pareto frontier</strong> — identify why active learning's uncertainty sampling achieves ~1.85&times; label efficiency over random annotation at the same budget.</div>
                <div style="margin-bottom: 3px;">3. <strong>Identify the budget threshold below which model quality is infeasible</strong> — find the annotation spending level where validation accuracy falls below the minimum viable threshold for each deployment context.</div>
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
                    Curriculum learning and convergence speed from @sec-data-selection-curriculum-learning &middot;
                    Pareto efficiency definition from @sec-data-selection-pareto-tradeoffs
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
                "If you can only spend $10,000 annotating training data, does randomly
                labeling examples cost more or less than selecting the most informative ones
                first &mdash; and is there a budget level below which no annotation strategy
                can produce a viable model?"
            </div>
        </div>
    </div>
    """)
    return


# ── CELL 3: READING ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-data-selection-curriculum-learning** — Curriculum Learning: why ordering
      examples from easy to hard accelerates convergence, and the competence-based
      pacing function &lambda;(t).
    - **@sec-data-selection-active-learning** — Active Learning: uncertainty sampling,
      diversity sampling, and expected information gain per annotation dollar.
    - **@sec-data-selection-pareto-tradeoffs** — The Pareto Frontier: why annotating
      everything is almost never optimal, and how to identify the efficient frontier.
    """), kind="info")
    return


# ── CELL 4: CONTEXT TOGGLE ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Cloud (H100, 80 GB HBM)": "cloud",
            "Edge (Jetson Orin NX, 16 GB)": "edge",
        },
        value="Cloud (H100, 80 GB HBM)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md(
            "**Select your deployment context.** "
            "This determines hardware constraints and minimum viable annotation budgets for both acts."
        ),
        context_toggle,
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 5: ACT1_BANNER ──────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "I"
    _act_color = COLORS["BlueLine"]
    _act_title = "The Selection Cost Blindspot"
    _act_duration = "12 min"
    _act_why = (
        "You expect that reordering training examples is a minor implementation detail. "
        "The instruments will show that presenting examples in curriculum order (easy to hard) "
        "achieves 2\u20133\u00d7 convergence speedup over random ordering "
        "\u2014 without changing the data, the model, or the hardware."
    )
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


# ── CELL 6: ACT1_STAKEHOLDER ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["BlueLine"]
    mo.vstack([
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: #eff6ff;
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; ML Research Lead, Vision Startup
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "We&rsquo;ve been training on our full 1M image dataset for 6 weeks and we&rsquo;re
                not converging. A colleague says curriculum learning could get us to the same
                accuracy in 2 weeks. That sounds like marketing. Is a 3&times; speedup from
                just reordering examples actually real?"
            </div>
        </div>
        """),
        mo.md("""
        The research lead is skeptical — reasonable, given how counterintuitive the claim sounds.
        But the speedup is grounded in learning theory: the order in which you present examples
        to a model is not neutral. Before running the simulator, commit to your prediction.
        """),
    ])
    return


# ── CELL 5: ACT I PREDICTION LOCK ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act1_prediction = mo.ui.radio(
        options={
            "A) Curriculum learning rarely helps — random order is usually fine": "option_a",
            "B) ~10% speedup at best — not worth the engineering effort": "option_b",
            "C) 2–3× speedup is achievable with a quality curriculum ordering": "option_c",
            "D) 10× speedup — filter to only the hardest examples first": "option_d",
        },
        label=(
            "Your startup has trained on 1M images in random order for 6 weeks without convergence. "
            "What speedup can curriculum learning (easy-to-hard ordering) realistically deliver?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #6366f1; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #a5b4fc;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock &mdash; Act I
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-bottom: 12px;">
                Commit to a prediction before touching any controls.
                The instruments unlock once you select an answer.
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
            mo.md("Select your prediction above to unlock the Training Efficiency Simulator."),
            kind="warn",
        ),
    )
    return


# ── CELL 7: ACT I CONTROLS ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # Dataset size slider: 10K to 10M images
    # Range source: data_selection.qmd — small vision datasets 10K, large 1M–10M
    act1_dataset_size = mo.ui.slider(
        start=10_000, stop=10_000_000, value=1_000_000, step=10_000,
        label="Dataset size (images)",
    )
    # Quality threshold: what fraction of data passes quality filter
    # Source: data_selection.qmd §Data Quality Filtering
    act1_quality_threshold = mo.ui.slider(
        start=0, stop=100, value=80, step=5,
        label="Data quality threshold (%)",
    )
    # Selection method dropdown
    act1_selection_method = mo.ui.dropdown(
        options={
            "Random (baseline)":                       "random",
            "Easy-first (start simple, stay simple)":  "easy_first",
            "Hard-first (start with hardest examples)": "hard_first",
            "Curriculum (easy to hard, paced)":        "curriculum",
        },
        value="Random (baseline)",
        label="Selection method",
    )
    mo.vstack([
        mo.md("### Training Efficiency Simulator — Controls"),
        mo.hstack([act1_dataset_size, act1_quality_threshold], justify="start", gap="2rem"),
        mo.hstack([act1_selection_method], justify="start", gap="2rem"),
    ])
    return (act1_dataset_size, act1_quality_threshold, act1_selection_method)


# ── CELL 8: ACT I PHYSICS ENGINE ──────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np, math,
    act1_dataset_size, act1_quality_threshold, act1_selection_method,
    COLORS, apply_plotly_theme,
    CURRICULUM_MIN_SPEEDUP, CURRICULUM_MAX_SPEEDUP, HARD_FIRST_PENALTY,
):
    # ── Physics model (source: data_selection.qmd §Training Efficiency) ──────
    # Baseline training steps to convergence scales sublinearly with dataset size.
    # Reference: data_selection.qmd — "training steps proportional to D^0.7 for vision tasks"
    dataset_n   = act1_dataset_size.value
    quality_pct = act1_quality_threshold.value / 100.0
    method      = act1_selection_method.value

    # Effective dataset size after quality filter
    effective_n = dataset_n * quality_pct

    # Baseline steps (random): sublinear scaling, approx D^0.7
    # Source: data_selection.qmd §Scaling Laws for Data
    _BASE_STEPS_REF = 50_000    # steps to convergence for 1M random images (reference)
    _REF_N          = 1_000_000
    _eff_n_safe     = max(effective_n, 1.0)  # guard against zero
    baseline_steps  = _BASE_STEPS_REF * ((_eff_n_safe / _REF_N) ** 0.7)

    # Method-specific speedup multiplier
    # Source: data_selection.qmd §Curriculum Learning
    #   - easy_first: minor speedup (removes some noise) but plateaus early
    #   - hard_first: slower convergence (unstable gradients early)
    #   - curriculum: 2–3× speedup (competence-based pacing)
    _QUALITY_BONUS = 1.0 + 0.3 * quality_pct  # higher quality data helps all methods

    if method == "random":
        speedup   = 1.0
        steps     = baseline_steps
        label_str = "Random (baseline)"
        color_bar = COLORS["BlueLine"]
    elif method == "easy_first":
        # Easy-first: 1.2–1.4× speedup, modest improvement, no curriculum pacing
        speedup   = 1.3 * _QUALITY_BONUS
        steps     = baseline_steps / speedup
        label_str = "Easy-first"
        color_bar = COLORS["OrangeLine"]
    elif method == "hard_first":
        # Hard-first: gradient instability early, 30% slower than random
        # Source: data_selection.qmd — "hard-first destabilizes early training"
        speedup   = HARD_FIRST_PENALTY
        steps     = baseline_steps / speedup
        label_str = "Hard-first"
        color_bar = COLORS["RedLine"]
    else:  # curriculum
        # Curriculum: 2–3× speedup from competence-based pacing
        # Using midpoint 2.5× for display; scales with quality
        speedup   = 2.5 * min(_QUALITY_BONUS, 1.4)  # cap at reasonable max
        steps     = baseline_steps / speedup
        label_str = "Curriculum (easy to hard)"
        color_bar = COLORS["GreenLine"]

    # Annotation pipeline cost: 1 annotation-hour per 500 images reviewed
    # Source: data_selection.qmd §Annotation Economics
    annotation_hours  = (dataset_n / 500) * quality_pct
    annotation_cost_k = (annotation_hours * 15) / 1000  # $15/hr crowd-source rate

    # Wall-clock training time (H100, ResNet-50 class)
    # Source: data_selection.qmd §Training Time Estimates — 12 steps/sec on H100
    STEPS_PER_SEC    = 12     # H100 steps/sec for ResNet-50 class (data_selection.qmd)
    wall_clock_hours = steps / (STEPS_PER_SEC * 3600)

    # ── Build comparison chart: all 4 methods side-by-side ───────────────────
    _methods      = ["Random", "Easy-first", "Hard-first", "Curriculum"]
    _speedups_all = [1.0, 1.3 * _QUALITY_BONUS, HARD_FIRST_PENALTY, 2.5 * min(_QUALITY_BONUS, 1.4)]
    _steps_list   = [baseline_steps / s for s in _speedups_all]
    _colors_list  = [COLORS["BlueLine"], COLORS["OrangeLine"], COLORS["RedLine"], COLORS["GreenLine"]]

    _method_idx = {"random": 0, "easy_first": 1, "hard_first": 2, "curriculum": 3}[method]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_methods,
        y=_steps_list,
        marker_color=_colors_list,
        marker_line=dict(
            width=[3 if i == _method_idx else 0 for i in range(4)],
            color=["#0f172a" if i == _method_idx else "transparent" for i in range(4)],
        ),
        text=[f"{s/1000:.0f}K steps<br>{sp:.1f}x speedup" for s, sp in zip(_steps_list, _speedups_all)],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title="Training Steps to Convergence by Selection Method",
        xaxis_title="Selection Method",
        yaxis_title="Steps to Convergence",
        height=380,
        showlegend=False,
    )
    apply_plotly_theme(fig)

    # ── Formula display ───────────────────────────────────────────────────────
    _speedup_color = "#4ade80" if speedup > 1.5 else "#fde68a" if speedup >= 1.0 else "#f87171"

    mo.vstack([
        mo.md("### Physics"),
        mo.Html(f"""
        <div style="background: #0f172a; border-radius: 10px; padding: 16px 20px;
                    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
                    color: #e2e8f0; line-height: 1.8; margin: 8px 0;">
            <span style="color:#94a3b8;">// Effective dataset after quality filter</span><br>
            N_eff = N &times; quality = {dataset_n:,} &times; {quality_pct:.2f}
                  = <strong style="color:#6ee7b7;">{effective_n:,.0f} images</strong><br><br>
            <span style="color:#94a3b8;">// Baseline steps (random order, sublinear scaling D^0.7)</span><br>
            steps_baseline = 50,000 &times; (N_eff / 1M)^0.7
                           = <strong style="color:#a5b4fc;">{baseline_steps:,.0f} steps</strong><br><br>
            <span style="color:#94a3b8;">// Method speedup factor: {label_str}</span><br>
            speedup = <strong style="color:{_speedup_color};">{speedup:.2f}&times;</strong><br><br>
            <span style="color:#94a3b8;">// Steps to convergence with selected method</span><br>
            steps_method = steps_baseline / speedup
                         = <strong style="color:#fde68a;">{steps:,.0f} steps</strong><br><br>
            <span style="color:#94a3b8;">// Wall-clock time (H100, 12 steps/sec)</span><br>
            wall_clock = steps / (12 steps/sec &times; 3600 sec/hr)
                       = <strong style="color:#fde68a;">{wall_clock_hours:.1f} hours</strong>
        </div>
        """),
        mo.md("### Results"),
        mo.plotly(fig),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px;">
            <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: #f8fafc;">
                <div style="color: #475569; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">Steps</div>
                <div style="font-size: 1.9rem; font-weight: 800;
                            color: {color_bar}; font-family: 'SF Mono', monospace;">
                    {steps/1000:.0f}K
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">to convergence</div>
            </div>
            <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: #f8fafc;">
                <div style="color: #475569; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">Speedup</div>
                <div style="font-size: 1.9rem; font-weight: 800;
                            color: {'#008F45' if speedup > 1.5 else '#CC5500' if speedup >= 1.0 else '#CB202D'};
                            font-family: 'SF Mono', monospace;">
                    {speedup:.2f}&times;
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">vs random</div>
            </div>
            <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: #f8fafc;">
                <div style="color: #475569; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">Wall-clock</div>
                <div style="font-size: 1.9rem; font-weight: 800;
                            color: {COLORS['BlueLine']}; font-family: 'SF Mono', monospace;">
                    {wall_clock_hours:.1f}h
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">H100 training time</div>
            </div>
            <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: #f8fafc;">
                <div style="color: #475569; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.06em;">Annotation Cost</div>
                <div style="font-size: 1.9rem; font-weight: 800;
                            color: {COLORS['OrangeLine']}; font-family: 'SF Mono', monospace;">
                    ${annotation_cost_k:.1f}K
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">curation pipeline</div>
            </div>
        </div>
        """),
    ])
    return (steps, speedup, baseline_steps, wall_clock_hours, annotation_cost_k, method)


# ── CELL 9: ACT I PREDICTION REVEAL ───────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_prediction, speedup, HARD_FIRST_PENALTY):
    _pred      = act1_prediction.value
    _is_correct = (_pred == "option_c")

    if _is_correct:
        mo.callout(mo.md(
            f"**Correct.** You predicted a 2&ndash;3&times; speedup. "
            f"With curriculum ordering at default settings, the simulator shows "
            f"**{speedup:.2f}&times;** — within the empirical range documented in "
            f"@sec-data-selection-curriculum-learning. "
            f"The key insight: easy examples first build numerically stable gradients "
            f"before hard examples are introduced."
        ), kind="success")
    elif _pred == "option_d":
        mo.callout(mo.md(
            f"**Not quite.** You predicted 10&times; from hard-first ordering. "
            f"The simulator shows hard-first is actually **{HARD_FIRST_PENALTY:.1f}&times; slower** "
            f"than random — the opposite of helpful. Hard examples early destabilize gradient "
            f"updates before the model has a stable loss landscape. "
            f"The correct answer is C: curriculum (easy to hard) gives **2&ndash;3&times; speedup**."
        ), kind="warn")
    elif _pred == "option_a":
        mo.callout(mo.md(
            f"**Not quite.** You predicted ordering does not matter. "
            f"The simulator shows curriculum ordering achieves a "
            f"**{speedup:.2f}&times; speedup** over random. "
            f"Ordering is not neutral — it determines the gradient signal quality at each "
            f"training step. See @sec-data-selection-curriculum-learning."
        ), kind="warn")
    elif _pred == "option_b":
        mo.callout(mo.md(
            f"**Not quite.** You predicted ~10% improvement. "
            f"The simulator shows curriculum ordering achieves **{speedup:.2f}&times;** — "
            f"substantially more. The mechanism is not subtle: easy examples produce "
            f"clean, consistent gradient directions early in training, compounding into "
            f"significantly faster convergence."
        ), kind="warn")
    else:
        mo.callout(mo.md("Select your prediction above to see the reveal."), kind="info")
    return (_is_correct,)


# ── CELL 10: ACT I REFLECTION ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _is_correct):
    mo.stop(
        _is_correct is None,
        mo.callout(
            mo.md("Complete the prediction above to unlock the reflection question."),
            kind="warn",
        ),
    )
    act1_reflection = mo.ui.radio(
        options={
            "A) It filters out bad data, reducing noise in the loss signal": "refl_a",
            "B) It reduces the effective dataset size, so training is faster": "refl_b",
            "C) Easy examples first build stable gradients before hard examples destabilize them": "refl_c",
            "D) It forces the model to memorize common patterns before rare ones": "refl_d",
        },
        label="Why does curriculum ordering (easy to hard) accelerate convergence?",
    )
    mo.vstack([
        mo.md("### Reflection — Why Does Curriculum Ordering Help?"),
        act1_reflection,
    ])
    return (act1_reflection,)


# ── CELL 11: ACT I REFLECTION FEEDBACK ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_reflection):
    mo.stop(
        act1_reflection.value is None,
        mo.callout(
            mo.md("Answer the reflection question to continue to the MathPeek."),
            kind="warn",
        ),
    )
    if act1_reflection.value == "refl_c":
        mo.callout(mo.md(
            "**Correct.** The gradient signal from easy examples is consistent and "
            "well-directed early in training. Hard examples with ambiguous or contradictory "
            "labels produce high-variance gradients that destabilize the loss landscape. "
            "The curriculum pacing function &lambda;(t) = &lambda;&#8320; &times; e^(&alpha;t) "
            "gradually increases task difficulty as the model's competence rises — "
            "matching example difficulty to current model capability. "
            "See @sec-data-selection-curriculum-learning for the full derivation."
        ), kind="success")
    elif act1_reflection.value == "refl_a":
        mo.callout(mo.md(
            "**Not quite.** Filtering is separate from ordering. A quality threshold "
            "removes low-quality examples regardless of order. Curriculum learning is "
            "about the *sequence* in which high-quality examples are presented, not which "
            "examples are included. The speedup comes from gradient signal quality, "
            "not data volume reduction."
        ), kind="warn")
    elif act1_reflection.value == "refl_b":
        mo.callout(mo.md(
            "**Not quite.** Curriculum learning operates on the full dataset — it does not "
            "reduce dataset size. In fact, it may require more data curation effort to "
            "score and order examples by difficulty. The speedup is in *training steps to "
            "convergence*, not in the amount of data seen."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Memorizing common patterns describes rote learning, "
            "not faster convergence. Curriculum ordering works because it provides "
            "numerically stable gradient updates at each phase of training — easy "
            "examples establish a reliable loss minimum before hard examples introduce "
            "the fine-grained discriminative signal."
        ), kind="warn")
    return


# ── CELL 12: ACT I MATHPEEK ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Curriculum Pacing Function": mo.md("""
        **Competence-Based Pacing** (from @sec-data-selection-curriculum-learning):

        The curriculum pacing function controls what fraction of the difficulty
        distribution is accessible at training step t:

        ```
        lambda(t) = lambda_0 * exp(alpha * t)
        ```

        - **&lambda;(t)** — maximum difficulty score accessible at step t
        - **&lambda;&#8320;** — initial difficulty threshold (e.g., 0.2 = bottom 20% by difficulty)
        - **&alpha;** — pacing rate; larger &alpha; = faster curriculum progression
        - **t** — normalized training step in [0, 1]

        **Difficulty scoring** per example x_i:

        ```
        difficulty(x_i) = 1 - P(correct | x_i, theta_0)
        ```

        Where P(correct | x_i, &theta;&#8320;) is the model's confidence on example x_i
        at initialization. Low confidence = high difficulty.

        **Training steps to convergence** with curriculum:

        ```
        steps_curriculum = steps_baseline / speedup
        speedup in [2.0x, 3.0x]  (empirical range from data_selection.qmd)
        ```

        **Key insight:** The speedup is not from seeing fewer examples —
        it is from seeing examples in the order that maximizes gradient signal
        stability at each phase of training.
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 12: ACT2_BANNER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num = "II"
    _act_color = COLORS["OrangeLine"]
    _act_title = "The Pareto Curve"
    _act_duration = "25 min"
    _act_why = (
        "Act I showed that ordering data delivers a free speedup. Now discover that "
        "selecting which data to annotate is a Pareto optimization problem: active learning "
        "achieves ~1.85\u00d7 label efficiency over random annotation at the same budget "
        "\u2014 but there is a budget threshold below which no strategy produces a viable model."
    )
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
                    Act {_act_num} &middot; {_act_duration} &middot; First introduction: Pareto Frontier</div>
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


# ── CELL 13: ACT2_STAKEHOLDER ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _color = COLORS["GreenLine"]
    mo.vstack([
        mo.callout(mo.md(
            "**New instrument introduced in this lab:** The Pareto Curve maps model quality "
            "against selection cost. The Pareto frontier is the set of strategies where "
            "no improvement in quality is achievable without increasing cost. "
            "Points below the frontier represent inefficient strategies — you are paying "
            "more than necessary for the quality you receive."
        ), kind="info"),
        mo.Html(f"""
        <div style="border-left: 4px solid {_color}; background: #f0fdf4;
                    border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
                Incoming Message &middot; CTO, Vision Startup
            </div>
            <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
                "The board approved a $50K annotation budget. We have 1M unlabeled images and
                need the highest possible model quality before our product demo in 8 weeks.
                Should we annotate everything we can afford, or is there a smarter strategy?
                Our data science team says active learning could be more efficient, but the
                CTO at our competitor just hired 50 labelers and is annotating everything."
            </div>
        </div>
        """),
        mo.md("""
        The CTO is asking the right question: how do you spend an annotation budget to
        maximize model quality? This is not a linear problem — the relationship between
        annotation cost and model quality is a curve, and the optimal strategy sits on
        the Pareto frontier.
        """),
    ])
    return


# ── CELL 14: ACT II PREDICTION LOCK ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_prediction = mo.ui.radio(
        options={
            "A) Annotate everything affordable — more data always wins": "pred2_a",
            "B) Randomly sample 10% — annotation cost grows linearly, so cap it early": "pred2_b",
            "C) Use active learning to select the most informative 20% — Pareto optimal": "pred2_c",
            "D) Use only synthetically generated data — zero annotation cost": "pred2_d",
        },
        label=(
            "Budget: $50K. Dataset: 1M unlabeled images at $0.05/image ($50K annotates all 1M). "
            "What strategy maximizes model quality?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background: #1e293b; border-radius: 12px; padding: 20px;
                    border-left: 4px solid #008F45; margin: 8px 0;">
            <div style="font-size: 0.72rem; font-weight: 700; color: #6ee7b7;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
                Prediction Lock &mdash; Act II
            </div>
            <div style="color: #e2e8f0; font-size: 0.88rem; margin-bottom: 12px;">
                Commit to a strategy before adjusting the budget slider.
                The Pareto Curve will reveal the frontier once you select.
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
            mo.md("Select your strategy prediction above to unlock the Pareto Curve simulator."),
            kind="warn",
        ),
    )
    return


# ── CELL 16: ACT II CONTROLS ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    # Annotation budget slider: $5K to $200K
    # Range source: data_selection.qmd — practical annotation budget range for vision
    act2_budget_k = mo.ui.slider(
        start=5, stop=200, value=50, step=5,
        label="Annotation budget ($K)",
    )
    # Selection strategy
    act2_strategy = mo.ui.dropdown(
        options={
            "Random sampling":                         "random",
            "Active learning — uncertainty sampling":  "active_uncertainty",
            "Active learning — diversity sampling":    "active_diversity",
            "Annotate maximum images (full budget)":   "full_budget",
        },
        value="Random sampling",
        label="Selection strategy",
    )
    # Model architecture affects label efficiency requirements
    act2_model_size = mo.ui.dropdown(
        options={
            "Small (ResNet-18, 11M params)":  "small",
            "Medium (ResNet-50, 25M params)": "medium",
            "Large (ViT-B/16, 86M params)":   "large",
        },
        value="Medium (ResNet-50, 25M params)",
        label="Model architecture",
    )
    mo.vstack([
        mo.md("### Pareto Curve Controls"),
        mo.hstack([act2_budget_k, act2_strategy], justify="start", gap="2rem"),
        mo.hstack([act2_model_size], justify="start", gap="2rem"),
    ])
    return (act2_budget_k, act2_strategy, act2_model_size)


# ── CELL 17: ACT II PARETO PHYSICS ENGINE ─────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, go, np, math,
    act2_budget_k, act2_strategy, act2_model_size, context_toggle,
    COLORS, apply_plotly_theme,
    ANNOTATION_COST_PER_IMAGE, MIN_VIABLE_BUDGET_EDGE_K, MIN_VIABLE_BUDGET_CLOUD_K,
):
    # ── Physics model (source: data_selection.qmd §Active Learning) ──────────
    # Validation accuracy as function of annotation cost and strategy.
    # Model quality follows a log-saturation curve:
    #   acc = acc_floor + gain * log(1 + cost / scale)
    # Source: data_selection.qmd §Scaling Laws for Labeled Data

    budget_k  = act2_budget_k.value
    strategy  = act2_strategy.value
    model_sz  = act2_model_size.value
    context   = context_toggle.value

    # Images affordable at $0.05/image
    affordable_images = (budget_k * 1000) / ANNOTATION_COST_PER_IMAGE

    # Model-specific saturation parameters
    # Source: data_selection.qmd — larger models require more data before saturating
    _MODEL_PARAMS = {
        "small":  {"acc_floor": 62.0, "gain": 14.0, "scale_k": 8.0},
        "medium": {"acc_floor": 65.0, "gain": 15.5, "scale_k": 12.0},
        "large":  {"acc_floor": 60.0, "gain": 18.0, "scale_k": 20.0},
    }
    _p = _MODEL_PARAMS[model_sz]

    # Active learning label efficiency multiplier
    # Source: data_selection.qmd §Active Learning —
    #   "uncertainty sampling selects examples where model entropy is highest,
    #    yielding ~1.85× label efficiency vs random at equal annotation cost"
    _STRATEGY_EFF = {
        "random":             1.00,   # baseline
        "active_uncertainty": 1.85,   # 1.85× label efficiency (data_selection.qmd)
        "active_diversity":   1.70,   # diversity sampling 1.7× (data_selection.qmd)
        "full_budget":        0.90,   # diminishing returns on marginal examples
    }
    label_eff = _STRATEGY_EFF[strategy]

    # Effective budget after label efficiency
    effective_budget_k = budget_k * label_eff

    # Accuracy: log-saturation model
    # acc = floor + gain * log(1 + eff_budget / scale)
    # Source: data_selection.qmd §Validation Accuracy Scaling
    def _acc_fn(bk, eff):
        return _p["acc_floor"] + _p["gain"] * math.log(1.0 + bk * eff / _p["scale_k"])

    acc_achieved = min(_acc_fn(budget_k, label_eff), 85.0)  # 85% ceiling for this task class

    # ── Build Pareto Curve: all 4 strategies across budget range ─────────────
    _budget_range = np.linspace(1, 200, 300)

    _curve_specs = [
        ("Random sampling",                     1.00,   COLORS["BlueLine"],    "dash"),
        ("Active — uncertainty",                1.85,   COLORS["GreenLine"],   "solid"),
        ("Active — diversity",                  1.70,   COLORS["OrangeLine"],  "dot"),
        ("Full budget (diminishing returns)",   0.90,   COLORS["TextMuted"],   "dashdot"),
    ]

    fig2 = go.Figure()

    _all_acc_arrays = []
    for _cname, _ceff, _ccol, _cdash in _curve_specs:
        _accs = [
            min(_p["acc_floor"] + _p["gain"] * math.log(1.0 + bk * _ceff / _p["scale_k"]), 85.0)
            for bk in _budget_range
        ]
        _all_acc_arrays.append(_accs)
        fig2.add_trace(go.Scatter(
            x=_budget_range,
            y=_accs,
            mode="lines",
            name=_cname,
            line=dict(color=_ccol, width=2.5, dash=_cdash),
        ))

    # Pareto frontier: upper envelope across all strategies
    _pareto_acc = np.array(_all_acc_arrays).max(axis=0)
    fig2.add_trace(go.Scatter(
        x=_budget_range,
        y=_pareto_acc,
        mode="lines",
        name="Pareto Frontier",
        line=dict(color="#0f172a", width=3.5, dash="solid"),
    ))

    # Mark the student's current operating point
    fig2.add_trace(go.Scatter(
        x=[budget_k],
        y=[acc_achieved],
        mode="markers",
        name="Your design",
        marker=dict(
            size=16,
            color=COLORS["RedLine"],
            symbol="star",
            line=dict(width=2, color="white"),
        ),
    ))

    # Minimum viable thresholds (source: data_selection.qmd)
    _min_acc    = 70.0 if context == "edge" else 75.0
    _min_budget = MIN_VIABLE_BUDGET_EDGE_K if context == "edge" else MIN_VIABLE_BUDGET_CLOUD_K

    fig2.add_hline(
        y=_min_acc,
        line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dot"),
        annotation_text=f"Min viable ({context}): {_min_acc:.0f}%",
        annotation_position="right",
    )
    fig2.add_vline(
        x=_min_budget,
        line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dot"),
        annotation_text=f"Min budget: ${_min_budget:.0f}K",
        annotation_position="top right",
    )

    fig2.update_layout(
        title=f"Pareto Curve: Annotation Cost vs Validation Accuracy ({model_sz} model)",
        xaxis_title="Annotation Cost ($K)",
        yaxis_title="Validation Accuracy (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_plotly_theme(fig2)
    fig2.update_yaxes(range=[55, 88])

    # ── Pareto optimality check ───────────────────────────────────────────────
    # Pareto optimal: student is within 1.5pp of the frontier at their budget
    _frontier_at_budget = min(
        _p["acc_floor"] + _p["gain"] * math.log(1.0 + budget_k * 1.85 / _p["scale_k"]),
        85.0
    )
    _pareto_gap       = _frontier_at_budget - acc_achieved
    is_pareto_optimal = _pareto_gap < 1.5

    # ── Formula display ────────────────────────────────────────────────────────
    mo.vstack([
        mo.md("### Physics"),
        mo.Html(f"""
        <div style="background: #0f172a; border-radius: 10px; padding: 16px 20px;
                    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
                    color: #e2e8f0; line-height: 1.8; margin: 8px 0;">
            <span style="color:#94a3b8;">// Images affordable at $0.05/image</span><br>
            images_affordable = (${budget_k}K &times; 1000) / $0.05
                              = <strong style="color:#6ee7b7;">{affordable_images:,.0f} images</strong><br><br>
            <span style="color:#94a3b8;">// Label efficiency: {strategy.replace('_', ' ')}</span><br>
            label_efficiency = <strong style="color:#fde68a;">{label_eff:.2f}&times;</strong>
              &nbsp;&nbsp;(expected info gain per dollar vs random)<br><br>
            <span style="color:#94a3b8;">// Effective budget after label efficiency</span><br>
            effective_budget = ${budget_k}K &times; {label_eff:.2f}
                             = <strong style="color:#a5b4fc;">${effective_budget_k:.1f}K</strong><br><br>
            <span style="color:#94a3b8;">// Validation accuracy (log-saturation model)</span><br>
            acc = floor + gain &times; log(1 + eff_budget / scale)<br>
                = {_p['acc_floor']:.1f} + {_p['gain']:.1f} &times; log(1 + {effective_budget_k:.1f} / {_p['scale_k']:.1f})<br>
                = <strong style="color:#4ade80;">{acc_achieved:.1f}%</strong><br><br>
            <span style="color:#94a3b8;">// Distance to Pareto frontier at this budget</span><br>
            frontier_at_${budget_k}K = <strong style="color:#fde68a;">{_frontier_at_budget:.1f}%</strong>
            &nbsp;&nbsp; gap = <strong style="color:{'#4ade80' if is_pareto_optimal else '#f87171'};">{_pareto_gap:.1f}pp</strong>
            &nbsp;&nbsp;{'[Pareto optimal]' if is_pareto_optimal else '[Suboptimal — switch to active learning]'}
        </div>
        """),
        mo.md("### Pareto Curve"),
        mo.plotly(fig2),
    ])
    return (acc_achieved, budget_k, strategy, is_pareto_optimal, _min_budget, _min_acc, label_eff, affordable_images, effective_budget_k)


# ── CELL 18: ACT II FAILURE STATE ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    mo, budget_k, acc_achieved, context_toggle,
    _min_budget, _min_acc,
):
    _context           = context_toggle.value
    _budget_sufficient = budget_k >= _min_budget
    _quality_sufficient = acc_achieved >= _min_acc

    if not _budget_sufficient:
        mo.callout(mo.md(
            f"**Budget insufficient for minimum viable model quality.** "
            f"Current budget: ${budget_k}K &mdash; "
            f"Minimum required for {_context} deployment: ${_min_budget:.0f}K. "
            f"At this budget, estimated validation accuracy is **{acc_achieved:.1f}%** "
            f"vs the minimum viable threshold of **{_min_acc:.0f}%**. "
            f"Increase the annotation budget above ${_min_budget:.0f}K, or switch to "
            f"active learning to reduce the minimum viable budget."
        ), kind="danger")
    elif not _quality_sufficient:
        mo.callout(mo.md(
            f"**Quality below deployment threshold.** "
            f"Achieved accuracy: **{acc_achieved:.1f}%** — "
            f"Minimum for {_context}: **{_min_acc:.0f}%**. "
            f"Switch to active learning (uncertainty or diversity sampling) "
            f"to improve label efficiency at the current budget."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Budget sufficient.** "
            f"Estimated validation accuracy: **{acc_achieved:.1f}%** "
            f"(minimum viable: {_min_acc:.0f}%) — "
            f"Deployment constraint satisfied for {_context} context."
        ), kind="success")
    return


# ── CELL 19: ACT II PARETO OPTIMALITY FEEDBACK ───────────────────────────────
@app.cell(hide_code=True)
def _(mo, is_pareto_optimal, strategy, acc_achieved, budget_k):
    if is_pareto_optimal and strategy in ("active_uncertainty", "active_diversity"):
        mo.callout(mo.md(
            f"**Pareto optimal.** "
            f"Your design ({strategy.replace('_', ' ')}, ${budget_k}K) sits on the "
            f"Pareto frontier — no strategy achieves higher accuracy at this cost. "
            f"Predicted accuracy: **{acc_achieved:.1f}%**. "
            f"Active learning achieves this by selecting examples with maximum expected "
            f"information gain per annotation dollar."
        ), kind="success")
    elif strategy == "full_budget":
        mo.callout(mo.md(
            f"**Suboptimal: diminishing returns.** "
            f"Annotating everything affordable uses the full ${budget_k}K but "
            f"achieves **{acc_achieved:.1f}%** — below the Pareto frontier. "
            f"The marginal examples added have low information content: "
            f"the model has already learned what they teach. "
            f"Active learning selects only the high-information subset, "
            f"achieving better accuracy at the same cost."
        ), kind="warn")
    elif strategy == "random":
        mo.callout(mo.md(
            f"**Below the Pareto frontier.** "
            f"Random sampling at ${budget_k}K achieves {acc_achieved:.1f}%. "
            f"The frontier (active learning) achieves higher accuracy at the same cost. "
            f"Random sampling wastes annotation budget on easy examples "
            f"the model would classify correctly anyway."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            f"**Exploring the frontier.** "
            f"Strategy: {strategy.replace('_', ' ')} — "
            f"Budget: ${budget_k}K — "
            f"Accuracy: {acc_achieved:.1f}%. "
            f"Adjust the budget and strategy to find the Pareto optimal operating point."
        ), kind="info")
    return


# ── CELL 20: ACT II PREDICTION REVEAL ────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_prediction, acc_achieved, budget_k):
    _pred2       = act2_prediction.value
    _is_correct2 = (_pred2 == "pred2_c")

    if _is_correct2:
        mo.callout(mo.md(
            f"**Correct.** Active learning with uncertainty sampling is Pareto optimal "
            f"at the $50K budget. The simulator confirms: at ${budget_k}K with active "
            f"learning, accuracy reaches **{acc_achieved:.1f}%** — higher than random "
            f"sampling or full-budget annotation at the same cost. "
            f"The Pareto insight: you are not trying to annotate the most images, "
            f"you are trying to buy the most information per dollar."
        ), kind="success")
    elif _pred2 == "pred2_a":
        mo.callout(mo.md(
            f"**Not quite.** Annotating everything affordable sounds intuitive, "
            f"but it sits below the Pareto frontier. At $50K with full-budget random "
            f"annotation, the accuracy is lower than active learning at the same budget. "
            f"More data is not always better — diminishing returns set in once the model "
            f"has seen enough easy examples. The correct answer is C."
        ), kind="warn")
    elif _pred2 == "pred2_b":
        mo.callout(mo.md(
            f"**Not quite.** Capping at 10% random sampling is conservative but "
            f"suboptimal. It leaves budget on the table and selects examples without "
            f"regard for their information content. Active learning achieves better "
            f"accuracy than 10% random at the same or lower cost. The correct answer is C."
        ), kind="warn")
    elif _pred2 == "pred2_d":
        mo.callout(mo.md(
            f"**Not quite.** Synthetic data at zero annotation cost sounds appealing, "
            f"but domain shift between synthetic and real distributions caps accuracy well "
            f"below what real-data annotation achieves. The correct answer is C: "
            f"active learning selects the most informative real examples "
            f"for maximum quality per annotation dollar."
        ), kind="warn")
    else:
        mo.callout(mo.md("Select your Act II prediction to see the reveal."), kind="info")
    return (_is_correct2,)


# ── CELL 21: ACT II REFLECTION ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    act2_reflection = mo.ui.radio(
        options={
            "A) It avoids duplicate examples in the training set": "refl2_a",
            "B) It selects examples where the model is most uncertain — highest expected information gain per annotation dollar": "refl2_b",
            "C) It automatically generates labels without human annotation": "refl2_c",
            "D) It trains the model faster per gradient step": "refl2_d",
        },
        label="What makes active learning more label-efficient than random sampling?",
    )
    mo.vstack([
        mo.md("### Reflection — Why Is Active Learning More Efficient?"),
        act2_reflection,
    ])
    return (act2_reflection,)


# ── CELL 22: ACT II REFLECTION FEEDBACK ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_reflection):
    mo.stop(
        act2_reflection.value is None,
        mo.callout(
            mo.md("Answer the reflection question to continue to the MathPeek."),
            kind="warn",
        ),
    )
    if act2_reflection.value == "refl2_b":
        mo.callout(mo.md(
            "**Correct.** Active learning queries the annotator only for examples "
            "where the model's current predictive entropy is highest. "
            "These examples sit near the model's decision boundary — "
            "each label shifts the boundary more than an easy example would. "
            "Formally: expected information gain H(y | x, &theta;) is maximized "
            "for examples where the model is most uncertain. "
            "This produces 1.7&ndash;1.85&times; label efficiency vs random sampling "
            "(from @sec-data-selection-active-learning)."
        ), kind="success")
    elif act2_reflection.value == "refl2_a":
        mo.callout(mo.md(
            "**Not quite.** Deduplication is a data cleaning step separate from "
            "active learning. Active learning does not filter duplicates — "
            "it selects from the full unlabeled pool based on model uncertainty. "
            "The efficiency gain comes from where on the decision boundary you "
            "spend annotation budget, not from reducing redundancy."
        ), kind="warn")
    elif act2_reflection.value == "refl2_c":
        mo.callout(mo.md(
            "**Not quite.** Active learning still requires human annotation — "
            "it selects which examples to send to human annotators, "
            "not how to label them automatically. "
            "Auto-labeling is a separate technique (pseudo-labeling, semi-supervised). "
            "Active learning's value is in selecting the most informative examples "
            "for the human to label."
        ), kind="warn")
    else:
        mo.callout(mo.md(
            "**Not quite.** Active learning does not change training speed per step. "
            "It changes which examples are labeled before training begins. "
            "The efficiency gain is in the data curation phase: "
            "fewer annotation dollars are spent on examples that provide little "
            "new information to the model."
        ), kind="warn")
    return


# ── CELL 23: ACT II MATHPEEK ─────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "The governing equation — Pareto Efficiency and Information Gain": mo.md("""
        **Pareto Efficiency Definition** (from @sec-data-selection-pareto-tradeoffs):

        A strategy S* is Pareto optimal if no other strategy S achieves
        strictly higher quality without strictly higher cost:

        ```
        S* is Pareto optimal iff:
            there is no S such that acc(S) >= acc(S*) and cost(S) <= cost(S*)
            with at least one strict inequality
        ```

        **Expected Information Gain** for active learning query selection
        (from @sec-data-selection-active-learning):

        ```
        EIG(x_i) = H(y | x_i, theta)  -- model predictive entropy

        H(y | x_i, theta) = - sum_c P(y=c | x_i, theta) * log P(y=c | x_i, theta)
        ```

        High entropy = high uncertainty = high expected value of annotation.

        **Label Efficiency** relative to random sampling:

        ```
        label_efficiency = acc_active(B) / acc_random(B)
        ```

        Where B is the annotation budget in dollars. Empirical values from
        @sec-data-selection-active-learning:
        - Uncertainty sampling: ~1.85x
        - Diversity sampling:   ~1.70x
        - Random:               1.00x (baseline)

        **Validation accuracy model** (log-saturation, capturing diminishing returns):

        ```
        acc(budget, eff) = floor + gain * log(1 + budget * eff / scale)
        ```

        Doubling the budget does not double the accuracy improvement —
        the log term captures the diminishing returns of additional annotation.

        **Key insight:** The Pareto frontier is traced by active learning.
        Any point below the frontier represents annotation dollars that
        bought less information than they could have.
        """)
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ── CELL 20: SYNTHESIS ────────────────────────────────────────────────────────
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
                    <strong>1. Curriculum learning delivers 2&ndash;3&times; convergence speedup with zero data change.</strong>
                    Presenting easy examples before hard ones produces stable gradient updates
                    at every training phase. The speedup is not from fewer examples &mdash; the
                    full dataset is used &mdash; but from the order that maximizes the gradient
                    signal quality at each stage.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Active learning achieves ~1.85&times; label efficiency over random annotation.</strong>
                    Uncertainty sampling selects the examples the model is least confident about,
                    concentrating annotation budget on the highest-information regions of the
                    decision boundary. At the same $50K budget, this produces a substantially
                    better model than exhaustive random labeling.
                </div>
                <div>
                    <strong>3. Below the minimum viable budget, no annotation strategy produces a deployable model.</strong>
                    The Pareto frontier has a floor: at insufficient annotation budgets, even
                    the most efficient strategy cannot reach the accuracy threshold needed for
                    production deployment. Budget is a hard constraint, not a soft optimization.
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
                    <strong>Lab 10: The Compression Frontier</strong> &mdash; This lab optimized
                    the training data. Lab 10 optimizes the model itself: quantization,
                    pruning, and the Pareto frontier between model size and accuracy that
                    determines whether your trained model fits on the deployment target.
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
                    <strong>Read:</strong> @sec-data-selection-curriculum-learning and
                    @sec-data-selection-pareto-tradeoffs for the full derivations.<br/>
                    <strong>Build:</strong> TinyTorch Module 09 &mdash; implement a difficulty
                    scorer and curriculum data sampler from scratch.
                </div>
            </div>

        </div>
        """),


        mo.accordion({
            "Self-Assessment: Can you answer these?": mo.md("""
    1. The scaling law L proportional to D^(-0.095) means doubling dataset size improves loss by approximately what factor — and why does this explain why a 10% coreset can match 100% dataset accuracy within 2%?

    2. A 10,000 H100 cluster can process 10T tokens in 3 months but only 5T quality tokens exist. At what ICR 'knee' (dataset fraction) does additional data yield near-zero learning per FLOP — and what is the correct strategy when you hit the Data Wall?

    3. A training run costs $100M. Data selection reduces the dataset by 50% while maintaining accuracy. What is the compute savings — and why does this demonstrate that data selection is the highest-leverage optimization in the D-A-M stack, not model architecture or hardware?

    *If you cannot answer all three from memory, revisit Act I and Act II.*
    """)
        }),
    ])
    return


# ── CELL 21: DESIGN LEDGER SAVE + HUD FOOTER ─────────────────────────────────
@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle, act1_prediction,
    act2_budget_k, act2_strategy,
    acc_achieved, is_pareto_optimal, budget_k, strategy,
    _is_correct,
    MIN_VIABLE_BUDGET_EDGE_K, MIN_VIABLE_BUDGET_CLOUD_K,
, decision_input, decision_ui):
    _context   = context_toggle.value
    _pred1     = act1_prediction.value or "none"
    _correct1  = bool(_is_correct) if _is_correct is not None else False
    _budget    = float(act2_budget_k.value)
    _strat     = act2_strategy.value or "none"
    _acc       = float(acc_achieved) if acc_achieved is not None else 0.0
    _pareto    = bool(is_pareto_optimal) if is_pareto_optimal is not None else False

    # Constraint hit: budget below minimum viable threshold for deployment context
    _min_k      = MIN_VIABLE_BUDGET_EDGE_K if _context == "edge" else MIN_VIABLE_BUDGET_CLOUD_K
    _constraint = _budget < _min_k

    ledger.save(
        chapter=9,
        design={
            "context":             _context,
            "selection_strategy":  _strat,
            "annotation_budget_k": _budget,
            "act1_prediction":     _pred1,
            "act1_correct":        _correct1,
            "act2_result":         _acc,
            "act2_decision":       _strat,
            "constraint_hit":      _constraint,
        "student_justification": str(decision_input.value),
            "pareto_optimal":      _pareto,
        },
    )

    _pred1_display = _pred1.replace("option_", "").upper() if _pred1 != "none" else "—"
    _pareto_badge  = "PARETO-OPTIMAL" if _pareto else "SUBOPTIMAL"
    _pareto_color  = COLORS["GreenLine"] if _pareto else COLORS["OrangeLine"]
    _ctx_color     = COLORS["Cloud"] if _context == "cloud" else COLORS["Edge"]

    mo.vstack([
        mo.md("---"),
        decision_ui,
        mo.Html(f"""
        <div style="display: flex; gap: 28px; align-items: center; flex-wrap: wrap;
                    padding: 14px 24px; background: #0f172a; border-radius: 12px;
                    margin-top: 16px; font-family: 'SF Mono', 'Fira Code', monospace;
                    font-size: 0.8rem; border: 1px solid #1e293b;">
            <div>
                <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
                    LAB
                </span>
                <span style="color: #e2e8f0; margin-left: 8px; font-weight: 700;">
                    09 / Data Selection
                </span>
            </div>
            <div>
                <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
                    CONTEXT
                </span>
                <span style="color: {_ctx_color}; margin-left: 8px; font-weight: 700;
                             text-transform: uppercase;">
                    {_context}
                </span>
            </div>
            <div>
                <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
                    ACT I
                </span>
                <span style="color: {'#4ade80' if _correct1 else '#f87171'};
                             margin-left: 8px; font-weight: 700;">
                    {'CORRECT' if _correct1 else 'INCORRECT'} (pred: {_pred1_display})
                </span>
            </div>
            <div>
                <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
                    BUDGET
                </span>
                <span style="color: {'#f87171' if _constraint else '#e2e8f0'};
                             margin-left: 8px; font-weight: 700;">
                    ${_budget:.0f}K {'(INFEASIBLE)' if _constraint else ''}
                </span>
            </div>
            <div>
                <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
                    ACC
                </span>
                <span style="color: #e2e8f0; margin-left: 8px; font-weight: 700;">
                    {_acc:.1f}%
                </span>
            </div>
            <div>
                <span style="color: #94a3b8; font-weight: 600; letter-spacing: 0.06em;">
                    FRONTIER
                </span>
                <span style="color: {_pareto_color}; margin-left: 8px; font-weight: 700;">
                    {_pareto_badge}
                </span>
            </div>
            <div style="margin-left: auto;">
                <span style="color: #94a3b8; font-size: 0.72rem;">
                    ch09 saved to ledger
                </span>
            </div>
        </div>
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
