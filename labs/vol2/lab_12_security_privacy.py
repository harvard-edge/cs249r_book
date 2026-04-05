import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-12: THE PRICE OF PRIVACY
#
# Volume II, Chapter: Security & Privacy (security_privacy.qmd)
#
# Four Parts (~56 minutes):
#   Part A — The Privacy Scaling Wall (10 min)
#             DP noise is constant; error per person scales as 1/N.
#             At N=100, per-person error is $2,000 (10x worse than N=1,000).
#
#   Part B — The Privacy-Accuracy Frontier (10 min)
#             CIFAR-10 struggles to reach 82% at epsilon=8.
#             The "knee" at epsilon 1-3 marks catastrophic quality loss.
#
#   Part C — The Defense Overhead Stack (12 min)
#             Every security measure extracts throughput. Full stack: 30-40% reduction.
#             Students must satisfy both privacy officer AND product manager.
#
#   Part D — The Privacy Budget Depletion (14 min, anchor)
#             Epsilon budget is finite and non-renewable. Basic composition
#             exhausts budget in hours, not months.
#
# Model extraction DROPPED from original plan. Arc restructured around privacy cost.
# Design Ledger: chapter="v2_12"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: SETUP + OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP ─────────────────────────────────────────────────────────────
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

    # ── Privacy constants ────────────────────────────────────────────────────
    # Source: @sec-security-privacy-differential-privacy-8c2b
    DEFAULT_SENSITIVITY = 200_000   # salary range sensitivity ($200K)
    DEFAULT_EPSILON = 1.0
    DEFAULT_DELTA = 1e-5

    # DP-SGD accuracy benchmarks (published results)
    # Source: @fig-privacy-utility-frontier
    MNIST_NO_DP = 99.0
    MNIST_EPS1 = 95.0
    MNIST_EPS01 = 88.0
    CIFAR_NO_DP = 93.0
    CIFAR_EPS8 = 82.0
    CIFAR_EPS1 = 65.0

    # Defense overhead constants
    # Source: @sec-security-privacy defense selection framework
    MIG_OVERHEAD = 0.15          # 15% throughput reduction
    DPSGD_OVERHEAD = 0.20        # 20% per-step overhead (avg)
    MONITORING_OVERHEAD_MS = 1.5  # ms per request
    OUTPUT_PERTURB_MS = 1.0       # ms per request
    RATE_LIMIT_OVERHEAD = 0.05    # 5% from rate limiting
    BASELINE_THROUGHPUT = 1000    # tok/s baseline
    SLO_THROUGHPUT = 800          # tok/s minimum SLO

    # Privacy budget composition
    # Source: @sec-security-privacy-privacy-budget-composition-edbe
    DEFAULT_BUDGET_EPSILON = 10.0
    DEFAULT_QUERY_EPSILON = 0.01
    DEFAULT_DAILY_QUERIES = 10_000

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math, DecisionLog,
        CLOUD_TFLOPS, EDGE_TFLOPS,
        DEFAULT_SENSITIVITY, DEFAULT_EPSILON, DEFAULT_DELTA,
        MNIST_NO_DP, MNIST_EPS1, MNIST_EPS01,
        CIFAR_NO_DP, CIFAR_EPS8, CIFAR_EPS1,
        MIG_OVERHEAD, DPSGD_OVERHEAD, MONITORING_OVERHEAD_MS,
        OUTPUT_PERTURB_MS, RATE_LIMIT_OVERHEAD,
        BASELINE_THROUGHPUT, SLO_THROUGHPUT,
        DEFAULT_BUDGET_EPSILON, DEFAULT_QUERY_EPSILON, DEFAULT_DAILY_QUERIES,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3); margin-bottom: 8px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 12
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.2rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Price of Privacy
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.1rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Noise Scaling &middot; Accuracy Frontier &middot; Defense Stack &middot; Budget Depletion
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 700px; line-height: 1.65;">
                Your privacy officer says "turn on differential privacy." Your product
                manager says "maintain 800 tokens/sec." These requirements are in direct
                tension, and the tension has a price that compounds across dataset size,
                task complexity, and query volume.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    4 Parts &middot; ~56 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter: Security &amp; Privacy
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px;">
                <span class="badge badge-fail">$2,000 error at N=100</span>
                <span class="badge badge-warn">30-40% throughput reduction</span>
                <span class="badge badge-info">Budget exhausted in hours</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
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
                <div style="margin-bottom: 3px;">1. <strong>Predict the privacy-dataset interaction</strong> &mdash;
                    discover that DP noise is constant but error per person scales as 1/N, making
                    privacy destructive for small datasets.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the defense overhead stack</strong> &mdash;
                    calculate that MIG + DP-SGD + monitoring + output perturbation reduces throughput
                    by 30-40%, and find a configuration that satisfies both privacy and performance.</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate privacy budget depletion</strong> &mdash;
                    discover that basic composition exhausts epsilon=10 in 2.4 hours at 10K queries/day,
                    and that privacy is a depletable resource, not a configuration setting.</div>
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
                    Differential privacy (epsilon-delta) from the Security and Privacy chapter &middot;
                    DP-SGD training mechanics &middot; Basic probability
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~56 min</strong><br/>
                    A: 10 &middot; B: 10 &middot; C: 12 &middot; D: 14 min
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Privacy has a price. How much does it cost in accuracy, throughput, and
                service lifetime &mdash; and why does the budget run out faster than anyone expects?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: READING ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete before this lab:

    - **Differential Privacy** &mdash; The epsilon-delta definition, Laplace mechanism,
      and sensitivity (the Security and Privacy chapter).
    - **DP-SGD** &mdash; Gradient clipping, noise addition, privacy accounting.
    - **Privacy Budget Composition** &mdash; Basic and advanced composition theorems.
    - **Defense Selection Framework** &mdash; MIG isolation, monitoring overhead.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── WIDGET CELL 1: Part A prediction ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    pA_pred = mo.ui.radio(
        options={
            "A: ~$200 -- same noise, same error": "A",
            "B: ~$500 -- modest increase": "B",
            "C: ~$2,000 -- 10x worse": "C",
            "D: ~$20,000 -- unusable": "D",
        },
        label=(
            "Salary analysis with DP (epsilon=1, sensitivity=$200K). "
            "At N=1,000 records, per-person error is $200. "
            "At N=100 records, what is the per-person error?"
        ),
    )
    return (pA_pred,)


# ─── WIDGET CELL 2: Part A controls + Part B prediction ──────────────────────
@app.cell(hide_code=True)
def _(mo, pA_pred):
    mo.stop(pA_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pA_epsilon = mo.ui.slider(start=0.1, stop=10.0, value=1.0, step=0.1, label="Epsilon")
    pA_N = mo.ui.slider(start=10, stop=10000, value=1000, step=10, label="Dataset size (N)")

    # -- Part B widgets --
    pB_pred = mo.ui.radio(
        options={
            "A: ~90% -- minimal loss": "A",
            "B: ~82% -- significant but usable": "B",
            "C: ~65% -- severely degraded": "C",
            "D: ~40% -- unusable": "D",
        },
        label=(
            "CIFAR-10 achieves 93% accuracy without DP. At epsilon=8 (weak privacy), "
            "what accuracy does DP-SGD achieve?"
        ),
    )
    return (pA_epsilon, pA_N, pB_pred)


# ─── WIDGET CELL 3: Part C prediction ────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pB_pred):
    mo.stop(pB_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pC_pred = mo.ui.radio(
        options={
            "A: ~900 tok/s -- overhead is small": "A",
            "B: ~850 tok/s -- MIG dominates": "B",
            "C: ~700-750 tok/s -- compound overhead": "C",
            "D: ~500 tok/s -- security halves throughput": "D",
        },
        label=(
            "Baseline: 1,000 tok/s. You add MIG isolation (-15%), monitoring (+1.5ms/req), "
            "output perturbation (+1.0ms/req), and rate limiting (-5%). Resulting throughput?"
        ),
    )
    return (pC_pred,)


# ─── WIDGET CELL 4: Part C controls + Part D prediction ─────────────────────
@app.cell(hide_code=True)
def _(mo, pC_pred):
    mo.stop(pC_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pC_mig = mo.ui.checkbox(label="MIG Isolation (-15%)", value=True)
    pC_monitor = mo.ui.checkbox(label="Monitoring (+1.5ms/req)", value=True)
    pC_output = mo.ui.checkbox(label="Output Perturbation (+1.0ms/req)", value=True)
    pC_rate = mo.ui.checkbox(label="Rate Limiting (-5%)", value=True)
    pC_dpsgd = mo.ui.checkbox(label="DP-SGD Training (-20% throughput)", value=False)

    # -- Part D widgets --
    pD_pred = mo.ui.number(
        start=0.01, stop=365, value=None, step=0.01,
        label=(
            "10,000 queries/day, epsilon=0.01 per query, total budget=10. "
            "Using basic composition, how many days until budget is exhausted? "
            "(Enter days, e.g., 0.1 for 2.4 hours.)"
        ),
    )
    return (pC_mig, pC_monitor, pC_output, pC_rate, pC_dpsgd, pD_pred)


# ─── WIDGET CELL 5: Part D controls ─────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, pD_pred):
    mo.stop(pD_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    pD_queries = mo.ui.slider(start=100, stop=100_000, value=10_000, step=100,
                              label="Daily query volume")
    pD_eps_q = mo.ui.slider(start=0.001, stop=0.1, value=0.01, step=0.001,
                            label="Epsilon per query")
    pD_budget = mo.ui.slider(start=1, stop=100, value=10, step=1,
                             label="Total epsilon budget")
    pD_comp = mo.ui.radio(
        options={"Basic Composition": "basic", "Advanced Composition": "advanced"},
        value="Basic Composition", label="Composition theorem", inline=True,
    )
    return (pD_queries, pD_eps_q, pD_budget, pD_comp)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(
    mo, go, np, math, COLORS, apply_plotly_theme,
    DEFAULT_SENSITIVITY, DEFAULT_DELTA,
    MNIST_NO_DP, MNIST_EPS1, MNIST_EPS01,
    CIFAR_NO_DP, CIFAR_EPS8, CIFAR_EPS1,
    MIG_OVERHEAD, DPSGD_OVERHEAD, MONITORING_OVERHEAD_MS,
    OUTPUT_PERTURB_MS, RATE_LIMIT_OVERHEAD,
    BASELINE_THROUGHPUT, SLO_THROUGHPUT,
    pA_pred, pA_epsilon, pA_N,
    pB_pred,
    pC_pred, pC_mig, pC_monitor, pC_output, pC_rate, pC_dpsgd,
    pD_pred, pD_queries, pD_eps_q, pD_budget, pD_comp,
):

    # ═════════════════════════════════════════════════════════════════════════
    # PART A: THE PRIVACY SCALING WALL
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; Chief Privacy Officer, Aether Health</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;We promised regulators that our patient data has differential privacy guarantees.
        But when I ask engineering what epsilon we are actually running at, they say 8.0 &mdash; which
        our auditor calls &lsquo;privacy theater.&rsquo; How much noise do we really need to add before
        the guarantee is meaningful, and what does that do to our model?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Dr. Anya Kowalski, Chief Privacy Officer &middot; Aether Health</div>
</div>
"""))

        items.append(mo.Html(f"""
        <div id="part-a" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['BlueLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">A</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part A &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">The Privacy Scaling Wall</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                DP noise magnitude is Sensitivity/epsilon, independent of dataset size. But the
                error per person scales as 1/N. At N=100, per-person error is $2,000 &mdash; ten
                times worse than at N=1,000. Privacy kills utility for small datasets.
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(pA_pred)
        if pA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the privacy scaling simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pA_epsilon, pA_N], gap="1.5rem"))

        _eps = pA_epsilon.value
        _N = pA_N.value

        # DP noise: Laplace(sensitivity / epsilon)
        _noise_magnitude = DEFAULT_SENSITIVITY / _eps
        _error_per_person = _noise_magnitude / _N

        # Curves for three epsilon values
        _n_range = np.logspace(1, 4, 100)  # 10 to 10,000
        _eps_vals = [0.1, 1.0, 10.0]
        _colors = [COLORS["RedLine"], COLORS["OrangeLine"], COLORS["GreenLine"]]

        _fig = go.Figure()
        for _e, _c in zip(_eps_vals, _colors):
            _noise = DEFAULT_SENSITIVITY / _e
            _errors = _noise / _n_range
            _fig.add_trace(go.Scatter(
                x=_n_range, y=_errors, mode="lines",
                name=f"epsilon={_e}", line=dict(color=_c, width=2),
            ))

        # Current point
        _fig.add_trace(go.Scatter(
            x=[_N], y=[_error_per_person], mode="markers",
            name=f"Current: eps={_eps}, N={_N}",
            marker=dict(size=14, color=COLORS["BlueLine"], line=dict(color="white", width=2)),
        ))

        # Utility zones
        _fig.add_hrect(y0=0, y1=100, fillcolor=COLORS["GreenLine"], opacity=0.08,
                       annotation_text="Useful", annotation_position="top left")
        _fig.add_hrect(y0=100, y1=1000, fillcolor=COLORS["OrangeLine"], opacity=0.08,
                       annotation_text="Marginal", annotation_position="top left")
        _fig.add_hrect(y0=1000, y1=100000, fillcolor=COLORS["RedLine"], opacity=0.08,
                       annotation_text="Destroyed", annotation_position="top left")

        _fig.update_layout(
            height=380,
            xaxis=dict(title="Dataset Size (N)", type="log"),
            yaxis=dict(title="Error per Person ($)", type="log"),
            legend=dict(orientation="h", y=-0.2, font_size=11),
            margin=dict(l=60, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        _err_color = COLORS["RedLine"] if _error_per_person > 1000 else COLORS["OrangeLine"] if _error_per_person > 100 else COLORS["GreenLine"]
        _utility = "DESTROYED" if _error_per_person > 1000 else "MARGINAL" if _error_per_person > 100 else "USEFUL"

        items.append(mo.as_html(_fig))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_err_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Error/Person</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_err_color};">${_error_per_person:,.0f}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_err_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Utility</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_err_color};">{_utility}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Noise</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">${_noise_magnitude:,.0f}</div>
            </div>
        </div>"""))

        items.append(mo.md(f"""
**Privacy Scaling Formula**

```
Noise magnitude = Sensitivity / epsilon = ${DEFAULT_SENSITIVITY:,} / {_eps} = ${_noise_magnitude:,.0f}
Error per person = Noise / N = ${_noise_magnitude:,.0f} / {_N:,} = ${_error_per_person:,.0f}

At N=1,000: ${_noise_magnitude:,.0f} / 1,000 = ${_noise_magnitude/1000:,.0f}
At N=100:   ${_noise_magnitude:,.0f} / 100   = ${_noise_magnitude/100:,.0f}  (10x worse)
```
*Source: @sec-security-privacy-differential-privacy-8c2b*
"""))

        # Prediction reveal
        if pA_pred.value == "C":
            _msg = "**Correct.** The noise is constant ($200K at epsilon=1). Dividing by N=100 instead of N=1,000 gives 10x worse error. DP is a technique for large datasets, not a universal privacy switch."
            _kind = "success"
        else:
            _msg = "**Per-person error scales as 1/N.** The noise magnitude ($200K at epsilon=1) does not change with dataset size. But when you divide that constant noise by N people, each person's error grows as N shrinks. At N=100, error is $2,000 per person -- 10x worse than N=1,000."
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.accordion({
            "Math Peek: Laplace Mechanism and Per-Person Error": mo.md("""
**Laplace mechanism noise:**
$$
\\text{noise} \\sim \\text{Lap}\\left(\\frac{\\Delta f}{\\varepsilon}\\right)
$$

**Where:**
- **$\\Delta f$**: Sensitivity (max change from one person's data, e.g., $200K salary range)
- **$\\varepsilon$**: Privacy parameter (lower = more private, more noise)

**Per-person error** when aggregating over $N$ records:
$$
\\text{Error}_{\\text{per-person}} = \\frac{\\Delta f / \\varepsilon}{N} = \\frac{\\Delta f}{\\varepsilon \\cdot N}
$$

At $\\varepsilon=1$, $\\Delta f=200K$: noise = $200K$.
- $N=1{,}000$: error = \\$200/person
- $N=100$: error = \\$2,000/person (10x worse)

**Key insight:** Noise is constant; per-person error scales as $1/N$.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B: THE PRIVACY-ACCURACY FRONTIER
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; ML Engineer, Aether Health</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;I implemented DP-SGD on our diagnostic model and accuracy dropped from 94% to 81%.
        The privacy team says epsilon must stay below 1.0, but the clinical team says anything
        below 90% accuracy is unsafe to deploy. Is there a clipping norm and noise multiplier
        combination that threads this needle, or is the privacy-accuracy tradeoff truly this brutal?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; James Oduya, ML Engineer &middot; Aether Health</div>
</div>
"""))

        items.append(mo.Html(f"""
        <div id="part-b" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['GreenLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">B</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part B &middot; 10 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">The Privacy-Accuracy Frontier</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                The "knee" at epsilon 1-3 marks the transition from practical to catastrophic
                quality loss. MNIST retains 95% at epsilon~1; CIFAR-10 struggles to reach 82% at epsilon=8.
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(pB_pred)
        if pB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the privacy-accuracy frontier."), kind="warn"))
            return mo.vstack(items)

        # Privacy-accuracy curves (parameterized from published benchmarks)
        _eps_range = np.logspace(-1, 2, 100)  # 0.1 to 100

        # MNIST: relatively resilient to DP noise
        _mnist_acc = MNIST_NO_DP - (MNIST_NO_DP - MNIST_EPS01) * np.exp(-_eps_range * 1.5)
        _mnist_acc = np.clip(_mnist_acc, 50, MNIST_NO_DP)

        # CIFAR-10: more sensitive to DP noise (weaker gradient signal)
        _cifar_acc = CIFAR_NO_DP - (CIFAR_NO_DP - 45) * np.exp(-_eps_range * 0.3)
        _cifar_acc = np.clip(_cifar_acc, 45, CIFAR_NO_DP)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_eps_range, y=_mnist_acc, mode="lines",
            name="MNIST (simple task)", line=dict(color=COLORS["GreenLine"], width=3),
        ))
        _fig.add_trace(go.Scatter(
            x=_eps_range, y=_cifar_acc, mode="lines",
            name="CIFAR-10 (complex task)", line=dict(color=COLORS["RedLine"], width=3),
        ))

        # Reference points
        _fig.add_trace(go.Scatter(
            x=[1, 8], y=[MNIST_EPS1, CIFAR_EPS8],
            mode="markers+text",
            text=[f"MNIST eps=1: {MNIST_EPS1}%", f"CIFAR eps=8: {CIFAR_EPS8}%"],
            textposition=["top center", "bottom center"],
            marker=dict(size=12, color=[COLORS["GreenLine"], COLORS["RedLine"]]),
            showlegend=False,
        ))

        # Knee region
        _fig.add_vrect(x0=1, x1=3, fillcolor="#94a3b8", opacity=0.1,
                       annotation_text="Knee (eps 1-3)", annotation_position="top left")

        _fig.update_layout(
            height=380,
            xaxis=dict(title="Epsilon (privacy budget)", type="log"),
            yaxis=dict(title="Accuracy (%)", range=[40, 100]),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=50, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        # DP-SGD noise-to-signal ratio at epsilon=1
        # sigma = C * sqrt(2*ln(1.25/delta)) / epsilon
        _C = 1.0  # clipping norm
        _sigma_eps1 = _C * math.sqrt(2 * math.log(1.25 / 1e-5)) / 1.0

        items.append(mo.as_html(_fig))
        items.append(mo.md(f"""
**DP-SGD Noise-to-Signal Ratio**

```
sigma = C * sqrt(2 * ln(1.25/delta)) / epsilon
      = 1.0 * sqrt(2 * ln(125000)) / 1.0
      = {_sigma_eps1:.1f}

At epsilon=1, the gradient is {_sigma_eps1:.1f}x more noise than signal.
MNIST survives because it is a simple task (strong gradient signal).
CIFAR-10 collapses because its gradient signal is weaker.
```
*Source: @sec-security-privacy-differential-privacy-8c2b, Abadi et al. 2016*
"""))

        # Prediction reveal
        if pB_pred.value == "B":
            _msg = f"**Correct.** CIFAR-10 at epsilon=8 achieves ~{CIFAR_EPS8}% -- significant loss from 93% but still usable. The knee at epsilon 1-3 is where accuracy drops steeply. Harder tasks lose accuracy faster because the gradient signal is weaker relative to DP noise."
            _kind = "success"
        else:
            _msg = f"**CIFAR-10 at epsilon=8 achieves ~{CIFAR_EPS8}%.** Students who predict ~90% expect 'epsilon=8 is weak privacy, so little cost.' But even weak privacy adds noise that degrades complex tasks. At epsilon=1, CIFAR-10 drops to ~65%. The knee region (epsilon 1-3) is the critical design boundary."
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.accordion({
            "Math Peek: DP-SGD Noise Multiplier": mo.md("""
**DP-SGD per-step noise:**
$$
\\tilde{g}_t = \\frac{1}{B}\\left(\\sum_{i \\in \\mathcal{B}} \\text{clip}(g_i, C) + \\mathcal{N}(0, \\sigma^2 C^2 \\mathbf{I})\\right)
$$

**Noise multiplier:**
$$
\\sigma = \\frac{c \\cdot \\Delta f}{\\varepsilon}
$$

**Where:**
- **$B$**: Batch size (sampling rate $q = B/N$)
- **$C$**: Gradient clipping norm (bounds per-sample sensitivity)
- **$\\sigma$**: Noise multiplier (scales with $1/\\varepsilon$)
- **$c$**: Constant depending on $\\delta$ and composition method

**Privacy-accuracy knee:** At $\\varepsilon \\approx 1$-$3$, DP noise overwhelms
the gradient signal for complex tasks (CIFAR-10), causing catastrophic accuracy loss.
Simple tasks (MNIST) tolerate lower $\\varepsilon$ because gradient signal is stronger.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C: THE DEFENSE OVERHEAD STACK
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; Security Architect, Aether Health</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;I am stacking input validation, output filtering, model encryption, and adversarial
        detection on our inference pipeline. Each layer adds latency. Our p99 just crossed 200ms
        and the product team is furious. How do I quantify the cumulative overhead of defense-in-depth
        so I can justify which layers to keep and which to cut?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Sofia Lindgren, Security Architect &middot; Aether Health</div>
</div>
"""))

        items.append(mo.Html(f"""
        <div id="part-c" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['OrangeLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">C</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part C &middot; 12 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">The Defense Overhead Stack</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                Every defense layer extracts measurable throughput. The full stack reduces
                throughput by 30-40%. Find the combination that satisfies both the privacy
                officer (epsilon &lt; 1.0) and the product manager (&gt;800 tokens/sec).
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(pC_pred)
        if pC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the defense stack builder."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pC_mig, pC_monitor, pC_output, pC_rate, pC_dpsgd], gap="1rem"))

        _throughput = float(BASELINE_THROUGHPUT)
        _layers = []

        if pC_mig.value:
            _reduction = _throughput * MIG_OVERHEAD
            _throughput -= _reduction
            _layers.append(("MIG Isolation", _reduction))

        if pC_dpsgd.value:
            _reduction = _throughput * DPSGD_OVERHEAD
            _throughput -= _reduction
            _layers.append(("DP-SGD", _reduction))

        if pC_monitor.value:
            # Convert ms overhead to throughput loss
            # At 1000 tok/s, each token takes 1ms. Adding 1.5ms monitoring per request
            # increases effective per-request time by ~15%
            _monitor_frac = MONITORING_OVERHEAD_MS / (1000 / BASELINE_THROUGHPUT + MONITORING_OVERHEAD_MS)
            _reduction = _throughput * _monitor_frac * 0.5  # amortized across batch
            _throughput -= _reduction
            _layers.append(("Monitoring", _reduction))

        if pC_output.value:
            _output_frac = OUTPUT_PERTURB_MS / (1000 / BASELINE_THROUGHPUT + OUTPUT_PERTURB_MS)
            _reduction = _throughput * _output_frac * 0.5
            _throughput -= _reduction
            _layers.append(("Output Perturbation", _reduction))

        if pC_rate.value:
            _reduction = _throughput * RATE_LIMIT_OVERHEAD
            _throughput -= _reduction
            _layers.append(("Rate Limiting", _reduction))

        # Waterfall chart
        _fig = go.Figure()
        _names = ["Baseline"] + [l[0] for l in _layers] + ["Final"]
        _values = [BASELINE_THROUGHPUT] + [-l[1] for l in _layers] + [0]
        _measures = ["absolute"] + ["relative"] * len(_layers) + ["total"]

        _fig.add_trace(go.Waterfall(
            x=_names, y=_values, measure=_measures,
            decreasing=dict(marker_color=COLORS["RedLine"]),
            increasing=dict(marker_color=COLORS["GreenLine"]),
            totals=dict(marker_color=COLORS["BlueLine"]),
            text=[f"{abs(v):.0f}" for v in _values],
            textposition="outside",
        ))
        _fig.add_hline(y=SLO_THROUGHPUT, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text=f"SLO: {SLO_THROUGHPUT} tok/s")
        _fig.update_layout(
            height=380,
            yaxis=dict(title="Throughput (tok/s)", range=[0, BASELINE_THROUGHPUT + 100]),
            margin=dict(l=60, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)

        _slo_met = _throughput >= SLO_THROUGHPUT
        _tp_color = COLORS["GreenLine"] if _slo_met else COLORS["RedLine"]
        _reduction_pct = (1 - _throughput / BASELINE_THROUGHPUT) * 100

        if _slo_met:
            _banner = mo.callout(mo.md(
                f"SLO met. {_throughput:.0f} tok/s exceeds {SLO_THROUGHPUT} tok/s minimum."
            ), kind="success")
        else:
            _banner = mo.callout(mo.md(
                f"**SLO VIOLATED.** {_throughput:.0f} tok/s is below the {SLO_THROUGHPUT} tok/s requirement. "
                "Disable some defenses or accept the trade-off."
            ), kind="danger")

        items.append(mo.as_html(_fig))
        items.append(_banner)
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_tp_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Final Throughput</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_tp_color};">{_throughput:.0f} tok/s</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Overhead</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_reduction_pct:.0f}%</div>
            </div>
        </div>"""))
        items.append(mo.md(f"""
**Defense Stack Compound Throughput**

```
T_secure = T_baseline x product(1 - overhead_i)
         = {BASELINE_THROUGHPUT} x {_throughput/BASELINE_THROUGHPUT:.3f}
         = {_throughput:.0f} tok/s  ({_reduction_pct:.0f}% reduction)
```
*Source: @sec-security-privacy, defense selection framework*
"""))

        # Prediction reveal
        if pC_pred.value == "C":
            _msg = "**Correct.** Compound overhead from MIG (15%) + monitoring + perturbation + rate limiting yields 25-35% total reduction, resulting in ~700-750 tok/s. Students who predict ~900 add overheads linearly rather than multiplicatively."
            _kind = "success"
        else:
            _msg = "**Compound overhead is 25-35%, yielding ~700-750 tok/s.** Each defense multiplies the remaining throughput, not the baseline. 0.85 x 0.93 x 0.95 x 0.95 = ~0.72, giving ~720 tok/s from a 1,000 tok/s baseline."
            _kind = "warn"
        items.append(mo.callout(mo.md(_msg), kind=_kind))

        items.append(mo.accordion({
            "Math Peek: Compound Defense Overhead": mo.md("""
**Multiplicative throughput reduction:**
$$
T_{\\text{final}} = T_{\\text{baseline}} \\times \\prod_{i=1}^{n} (1 - o_i)
$$

**Where:**
- **$T_{\\text{baseline}}$**: Baseline throughput (e.g., 1,000 tok/s)
- **$o_i$**: Overhead fraction of defense layer $i$

**Example stack:** MIG (15%) + DP-SGD (7%) + Monitoring (5%) + Rate Limit (5%):
$$
T = 1000 \\times 0.85 \\times 0.93 \\times 0.95 \\times 0.95 = 712 \\text{ tok/s}
$$

**Key insight:** Overheads multiply, not add. Linear estimation
($1 - 0.15 - 0.07 - 0.05 - 0.05 = 0.68$) over-predicts the loss.
Multiplicative: $0.85 \\times 0.93 \\times 0.95 \\times 0.95 = 0.712$ (less severe).
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D: THE PRIVACY BUDGET DEPLETION (ANCHOR)
    # ═════════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
<div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
            border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
    <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
        Incoming Message &middot; Compliance Lead, Aether Health</div>
    <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
        &ldquo;Our total privacy budget is epsilon = 10 for the year. We have already spent 4.2 on
        three model training runs and it is only March. At this burn rate, do we have enough budget
        left to retrain quarterly, or do we need to shut down experimentation until next fiscal year?&rdquo;</div>
    <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
        &mdash; Kenji Watanabe, Compliance Lead &middot; Aether Health</div>
</div>
"""))

        items.append(mo.Html(f"""
        <div id="part-d" style="margin: 32px 0 12px 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: {COLORS['RedLine']}; color: white; border-radius: 50%;
                            width: 32px; height: 32px; display: inline-flex; align-items: center;
                            justify-content: center; font-size: 0.9rem; font-weight: 800;">D</div>
                <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
                <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em;">Part D &middot; 14 min</div>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                        margin-top: 8px; line-height: 1.2;">The Privacy Budget Depletion</div>
            <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                        line-height: 1.55; max-width: 700px;">
                The epsilon budget is finite and non-renewable. Basic composition:
                epsilon_total = T x epsilon_per_query. After enough queries, the cumulative
                privacy loss exceeds any meaningful guarantee. Privacy is a depletable resource,
                not a configuration setting.
            </div>
        </div>
        """))

        items.append(mo.md("### Your Prediction"))
        items.append(pD_pred)
        if pD_pred.value is None:
            items.append(mo.callout(mo.md("Enter your prediction to unlock the budget depletion simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([pD_queries, pD_eps_q, pD_budget, pD_comp], gap="1rem"))

        _daily_q = pD_queries.value
        _eps_q = pD_eps_q.value
        _budget = pD_budget.value
        _comp = pD_comp.value

        # Basic composition: eps_total = T * eps_q
        # Advanced composition: eps_total = sqrt(2T * ln(1/delta)) * eps_q + T * eps_q * (exp(eps_q) - 1)
        # Simplified advanced: eps_total ~ sqrt(2T * ln(1/delta)) * eps_q for small eps_q

        if _comp == "basic":
            # T queries to exhaust budget: T = budget / eps_q
            _total_queries = _budget / _eps_q
            _days = _total_queries / _daily_q
            _comp_label = "Basic"
        else:
            # Advanced: budget = sqrt(2T * ln(1/delta)) * eps_q (dominant term)
            # T = (budget / eps_q)^2 / (2 * ln(1/delta))
            _ln_term = math.log(1 / DEFAULT_DELTA)
            _total_queries = (_budget / _eps_q)**2 / (2 * _ln_term)
            _days = _total_queries / _daily_q
            _comp_label = "Advanced"

        _hours = _days * 24

        # Timeline chart: cumulative epsilon over days
        _day_range = np.linspace(0, max(_days * 2, 30), 200)
        _queries_cum = _day_range * _daily_q

        if _comp == "basic":
            _eps_cum = _queries_cum * _eps_q
        else:
            _ln_term = math.log(1 / DEFAULT_DELTA)
            _eps_cum = np.sqrt(2 * _queries_cum * _ln_term) * _eps_q + _queries_cum * _eps_q * (math.exp(_eps_q) - 1)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_day_range, y=_eps_cum, mode="lines",
            name=f"Cumulative Epsilon ({_comp_label})",
            line=dict(color=COLORS["RedLine"], width=3),
        ))
        _fig.add_hline(y=_budget, line_dash="dash", line_color=COLORS["OrangeLine"],
                       annotation_text=f"Budget: epsilon={_budget}")
        if _days < max(_days * 2, 30):
            _fig.add_vline(x=_days, line_dash="dot", line_color=COLORS["RedLine"],
                           annotation_text=f"Exhausted: {_days:.2f} days ({_hours:.1f} hrs)")

        _fig.update_layout(
            height=380,
            xaxis=dict(title="Days"),
            yaxis=dict(title="Cumulative Epsilon"),
            margin=dict(l=60, r=20, t=30, b=40),
        )
        apply_plotly_theme(_fig)

        _exhausted = _days < 1
        _status_color = COLORS["RedLine"] if _exhausted else COLORS["OrangeLine"] if _days < 7 else COLORS["GreenLine"]
        _status = "EXHAUSTED IN HOURS" if _exhausted else "DAYS" if _days < 30 else "SUSTAINABLE"

        # Prediction comparison
        _predicted_days = pD_pred.value if pD_pred.value else 1
        _gap = abs(_days - _predicted_days) / max(_days, 0.001)

        if _gap < 0.5:
            _pred_msg = f"**Good estimate.** You predicted {_predicted_days:.2f} days. Actual ({_comp_label}): {_days:.2f} days."
            _pred_kind = "success"
        else:
            _pred_msg = (
                f"**You predicted {_predicted_days:.2f} days. Actual ({_comp_label}): {_days:.2f} days ({_hours:.1f} hours).** "
                f"With basic composition, 10,000 queries/day x $0.01/query = 100 epsilon/day. "
                f"Budget of 10 is exhausted in 0.1 days (2.4 hours)."
            )
            _pred_kind = "warn"

        # Service unavailable banner
        if _exhausted:
            _service_banner = mo.callout(mo.md(
                f"**SERVICE UNAVAILABLE: PRIVACY BUDGET EXHAUSTED.** "
                f"After {_hours:.1f} hours, all subsequent queries must be rejected. "
                f"Availability drops to 0% until budget is renewed."
            ), kind="danger")
        else:
            _service_banner = mo.callout(mo.md(
                f"Budget lasts {_days:.1f} days. Service remains available."
            ), kind="info")

        items.append(mo.as_html(_fig))
        items.append(_service_banner)
        items.append(mo.callout(mo.md(_pred_msg), kind=_pred_kind))
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_status_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Time to Exhaustion</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_status_color};">
                    {_hours:.1f} hrs</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {_status_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Status</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_status_color};">{_status}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white; border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Queries to Exhaust</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_total_queries:,.0f}</div>
            </div>
        </div>"""))
        items.append(mo.md(f"""
**Privacy Budget Composition ({_comp_label})**

```
{'Basic: eps_total = T x eps_q = ' + f'{_total_queries:,.0f} x {_eps_q} = {_total_queries * _eps_q:,.1f}' if _comp == 'basic' else 'Advanced: eps_total = sqrt(2T ln(1/delta)) x eps_q'}
Budget = {_budget}
Queries to exhaust = {_total_queries:,.0f}
Days = {_total_queries:,.0f} / {_daily_q:,} = {_days:.2f} days ({_hours:.1f} hours)
```
*Source: @sec-security-privacy-privacy-budget-composition-edbe*
"""))

        items.append(mo.accordion({
            "Math Peek: Privacy Budget Composition": mo.md("""
**Basic composition theorem:**
$$
\\varepsilon_{\\text{total}} = \\sum_{t=1}^{T} \\varepsilon_t = T \\cdot \\varepsilon_q
$$

**Advanced composition theorem** (tighter bound):
$$
\\varepsilon_{\\text{total}} = \\sqrt{2T \\ln(1/\\delta)} \\cdot \\varepsilon_q + T \\cdot \\varepsilon_q \\cdot (e^{\\varepsilon_q} - 1)
$$

**Where:**
- **$T$**: Total number of queries
- **$\\varepsilon_q$**: Privacy cost per query
- **$\\delta$**: Failure probability (typically $10^{-5}$)

**Budget exhaustion time:**
$$
t_{\\text{exhaust}} = \\frac{\\varepsilon_{\\text{budget}} / \\varepsilon_q}{\\text{queries/day}}
$$

Basic at $\\varepsilon_q=0.01$, budget=10, 10K queries/day: $10/0.01 = 1{,}000$ queries = 2.4 hours.

Advanced composition extends this to ~3.7 days via the $\\sqrt{T}$ scaling.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════

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
                    <strong>1. DP noise is constant; error per person scales as 1/N.</strong>
                    Privacy destroys utility for small datasets ($2,000 error at N=100 vs $200
                    at N=1,000). DP is a technique for large datasets, not a universal switch.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Every defense layer extracts measurable throughput.</strong>
                    The full defense stack (MIG + monitoring + perturbation + rate limiting)
                    reduces throughput by 25-35%. You cannot have maximum security AND maximum
                    throughput. The art is choosing which defenses to prioritize.
                </div>
                <div>
                    <strong>3. Privacy is a depletable resource, not a configuration setting.</strong>
                    Basic composition exhausts a budget of epsilon=10 in 2.4 hours at 10K queries/day.
                    Advanced composition extends this to ~3.7 days. Both eventually exhaust.
                    Systems must enforce explicit query budgets or face service unavailability.
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
                    <strong>Lab V2-13: The Robustness Budget</strong> &mdash; Privacy cost is one
                    constraint. Robustness is another. Adversarial training costs 26 percentage
                    points of clean accuracy AND 8x compute. How do you budget robustness?
                </div>
            </div>
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px; padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">Textbook &amp; TinyTorch</div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> the Security and Privacy chapter for full DP derivations and
                    the defense selection framework.<br/>
                    <strong>Build:</strong> TinyTorch DP-SGD module &mdash; implement gradient
                    clipping, noise addition, and privacy accounting.
                </div>
            </div>
        </div>
        """))
        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB COMPOSITION
    # ═════════════════════════════════════════════════════════════════════════

    tabs = mo.ui.tabs({
        "Part A -- The Privacy Scaling Wall": build_part_a(),
        "Part B -- The Privacy-Accuracy Frontier": build_part_b(),
        "Part C -- The Defense Overhead Stack": build_part_c(),
        "Part D -- The Privacy Budget Depletion": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: LEDGER HUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, pA_pred, pA_epsilon, pA_N, pB_pred, pC_pred, pD_pred, pD_comp, pD_budget):
    _scaling_pred = pA_pred.value if hasattr(pA_pred, 'value') else None
    _epsilon = pA_epsilon.value if hasattr(pA_epsilon, 'value') else 1.0
    _dataset_n = pA_N.value if hasattr(pA_N, 'value') else 1000
    _accuracy_pred = pB_pred.value if hasattr(pB_pred, 'value') else None
    _defense_pred = pC_pred.value if hasattr(pC_pred, 'value') else None
    _budget_pred = pD_pred.value if hasattr(pD_pred, 'value') else None
    _composition = pD_comp.value if hasattr(pD_comp, 'value') else "basic"
    _budget_eps = pD_budget.value if hasattr(pD_budget, 'value') else 10
    ledger.save(chapter=12, design={
        "partA_scaling_prediction": _scaling_pred,
        "partA_epsilon_choice": _epsilon,
        "partA_dataset_size": _dataset_n,
        "partB_accuracy_prediction": _accuracy_pred,
        "partC_defense_throughput_prediction": _defense_pred,
        "partD_budget_hours_prediction": _budget_pred,
        "partD_composition_method": _composition,
        "partD_budget_epsilon": _budget_eps,
    })

    mo.Html(f"""
    <div style="background: #0f172a; border-radius: 10px; padding: 18px 24px;
                margin-top: 32px; font-family: 'SF Mono', 'Fira Code', monospace;">
        <div style="color: #475569; font-size: 0.7rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 10px;">
            Design Ledger &middot; Lab V2-12 Saved
        </div>
        <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.8;">
            <span style="color: #64748b;">privacy_scaling:</span>
            <span style="color: {COLORS['RedLine']};">1/N per-person error</span><br/>
            <span style="color: #64748b;">defense_overhead:</span>
            <span style="color: {COLORS['OrangeLine']};">25-35% throughput</span><br/>
            <span style="color: #64748b;">budget_basic:</span>
            <span style="color: {COLORS['RedLine']};">2.4 hours</span><br/>
            <span style="color: #64748b;">budget_advanced:</span>
            <span style="color: {COLORS['GreenLine']};">~3.7 days</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
