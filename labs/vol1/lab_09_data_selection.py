import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


# ═════════════════════════════════════════════════════════════════════════════
# CELL 0: SETUP
# ═════════════════════════════════════════════════════════════════════════════

# ===========================================================================
# ZONE A: OPENING
# ===========================================================================

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
            "../../wheels/mlsysim-0.1.0-py3-none-any.whl", keep_going=False
        )
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    import plotly.graph_objects as go
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    import mlsysim

    # ── Hardware constants from registry ──────────────────────────────────
    A100_TFLOPS = mlsysim.Hardware.Cloud.A100.compute.peak_flops.m_as("TFLOPs/s")
    A100_BW     = mlsysim.Hardware.Cloud.A100.memory.bandwidth.m_as("GB/s")
    A100_RAM    = mlsysim.Hardware.Cloud.A100.memory.capacity.m_as("GB")

    JETSON_TFLOPS = mlsysim.Hardware.Edge.JetsonOrinNX.compute.peak_flops.m_as("TFLOPs/s")
    JETSON_BW     = mlsysim.Hardware.Edge.JetsonOrinNX.memory.bandwidth.m_as("GB/s")
    JETSON_RAM    = mlsysim.Hardware.Edge.JetsonOrinNX.memory.capacity.m_as("GB")

    # ── Model constants from registry ─────────────────────────────────────
    RESNET50_PARAMS = mlsysim.Models.ResNet50.parameters.m_as("count")
    RESNET50_FLOPS  = mlsysim.Models.ResNet50.inference_flops.m_as("flop")
    MOBILENET_PARAMS = mlsysim.Models.MobileNetV2.parameters.m_as("count")
    MOBILENET_FLOPS  = mlsysim.Models.MobileNetV2.inference_flops.m_as("flop")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        A100_BW, A100_RAM, A100_TFLOPS,
        COLORS, JETSON_BW, JETSON_RAM, JETSON_TFLOPS,
        LAB_CSS, MOBILENET_FLOPS, MOBILENET_PARAMS,
        RESNET50_FLOPS, RESNET50_PARAMS,
        apply_plotly_theme, go, ledger, math, mo, np,
    )


# ═════════════════════════════════════════════════════════════════════════════
# CELL 1: HEADER
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(LAB_CSS, mo):
    mo.vstack([
        LAB_CSS,
        mo.Html("""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 09
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Data Selection Paradox
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Coresets &middot; Selection Cost &middot; Preprocessing Tax &middot; Scaling Laws
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Less data trains faster &mdash; until the cost of choosing that data
                exceeds the savings, and meanwhile your GPU starves waiting for the
                CPU to finish preprocessing.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts &middot; ~52 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 9: Data Selection
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">ICR Frontier</span>
                <span class="badge badge-warn">Selection Inequality</span>
                <span class="badge badge-fail">Preprocessing Tax</span>
            </div>
        </div>
        """),
    ])
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 2: BRIEFING
# ═════════════════════════════════════════════════════════════════════════════
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
                <div style="margin-bottom: 3px;">1. <strong>Quantify the ICR frontier</strong> &mdash; determine that
                    50% of a large dataset can be discarded while retaining 99% of accuracy,
                    because the informativeness distribution follows a power law.</div>
                <div style="margin-bottom: 3px;">2. <strong>Evaluate the Selection Inequality</strong> &mdash;
                    T<sub>selection</sub> + T<sub>train</sub>(subset) &lt; T<sub>train</sub>(full)
                    breaks on edge hardware when scoring cost exceeds training savings.</div>
                <div style="margin-bottom: 3px;">3. <strong>Diagnose the preprocessing bottleneck</strong> &mdash;
                    identify when CPU augmentation starves the GPU, reducing utilization below 30%.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Data engineering pipeline from the Data Engineering chapter &middot;
                    Training fundamentals from the Training chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~52 min</strong><br/>
                    A: ~12 min &middot; B: ~12 min &middot; C: ~12 min &middot; D: ~10 min
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;If half your training data contributes almost nothing, why does
                throwing it away not always make training faster?&rdquo;
            </div>
        </div>
    </div>
    """)
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 3: READING
# ═════════════════════════════════════════════════════════════════════════════

# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **Chapter 9: Data Selection** -- ICR curve, coreset selection methods, and the
      Selection Inequality T_selection + T_train(subset) < T_train(full).
    - **Chapter 4: Data Engineering** -- data pipeline stages, CPU preprocessing,
      and data loading bottlenecks.
    - **Chapter 8: Training** -- training loop fundamentals and batch processing.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 4: TABS (Parts A-D + Synthesis)
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(
    A100_BW, A100_TFLOPS,
    COLORS, JETSON_BW, JETSON_TFLOPS,
    MOBILENET_FLOPS, MOBILENET_PARAMS,
    RESNET50_FLOPS, RESNET50_PARAMS,
    apply_plotly_theme, go, math, mo, np,
):
    # ── Part A widgets ────────────────────────────────────────────────────
    partA_pred = mo.ui.radio(
        options={
            "A) 10% (keep 900K) -- every image matters": "10",
            "B) 30% (keep 700K) -- some redundancy": "30",
            "C) 50% (keep 500K) -- massive redundancy": "50",
            "D) 80% (keep 200K) -- extreme redundancy": "80",
        },
        label="You have a 1M-image dataset. You want 99% of full-dataset accuracy. "
              "What fraction of the data can you discard?",
    )
    return (partA_pred,)

@app.cell(hide_code=True)
def _(mo):
    partA_frac = mo.ui.slider(
        start=5, stop=100, value=100, step=5,
        label="Dataset fraction (%)",
    )
    partA_redundancy = mo.ui.dropdown(
        options={"Low redundancy": "low", "Medium redundancy": "med", "High redundancy": "high"},
        value="Medium redundancy",
        label="Dataset redundancy level",
    )

    # ── Part B widgets ────────────────────────────────────────────────────
    partB_pred = mo.ui.radio(
        options={
            "A) Yes, significant time savings overall": "yes_large",
            "B) Yes, but barely worth it": "yes_barely",
            "C) No, the scoring overhead costs more than training savings": "no_loss",
            "D) It depends on the proxy model cost": "depends",
        },
        label="ResNet-50 scores 1M images on A100 (2.8 hrs). Full training: 8.0 hrs. "
              "Coreset (50%) training: 4.2 hrs. Is selection worth it?",
    )
    return (partB_pred,)

@app.cell(hide_code=True)
def _(mo):
    partB_coreset = mo.ui.slider(
        start=10, stop=90, value=50, step=5,
        label="Coreset fraction (%)",
    )
    partB_scorer = mo.ui.dropdown(
        options={
            "Full ResNet-50": "resnet",
            "Proxy MobileNetV2": "mobilenet",
            "Cached embeddings": "cached",
        },
        value="Full ResNet-50",
        label="Scoring model",
    )
    partB_hw = mo.ui.radio(
        options={"Cloud (A100)": "cloud", "Edge (Jetson Orin NX)": "edge"},
        value="Cloud (A100)",
        label="Deployment context:",
        inline=True,
    )

    # ── Part C widgets ────────────────────────────────────────────────────
    partC_pred = mo.ui.radio(
        options={
            "A) ~80% (GPU dominates, preprocessing is fast)": "80",
            "B) ~50% (roughly balanced)": "50",
            "C) ~25% (GPU waits for CPU most of the time)": "25",
            "D) ~10% (GPU is almost entirely idle)": "10",
        },
        label="RandAugment (5 transforms) + resize + normalize on 8 CPU workers, "
              "feeding an A100 GPU. GPU forward-backward: 12 ms/batch. "
              "What fraction of step time is GPU compute?",
    )
    return (partC_pred,)

@app.cell(hide_code=True)
def _(mo):
    partC_workers = mo.ui.slider(
        start=1, stop=16, value=8, step=1,
        label="CPU workers",
    )
    partC_augment = mo.ui.dropdown(
        options={
            "None (load only)": "none",
            "Basic (flip + crop)": "basic",
            "RandAugment-5": "ra5",
            "RandAugment-10 + MixUp": "ra10",
        },
        value="RandAugment-5",
        label="Augmentation complexity",
    )
    partC_model = mo.ui.dropdown(
        options={"MobileNetV2": "mobilenet", "ResNet-50": "resnet"},
        value="ResNet-50",
        label="Training model",
    )

    # ── Part D widgets ────────────────────────────────────────────────────
    partD_pred = mo.ui.radio(
        options={
            "A) 10B params on 200B tokens": "10b_200b",
            "B) 3B params on 660B tokens": "3b_660b",
            "C) 30B params on 66B tokens": "30b_66b",
            "D) All achieve roughly the same loss": "same",
        },
        label="Fixed budget: 10^21 FLOPs. Which achieves lower loss?",
    )
    return (partD_pred,)

@app.cell(hide_code=True)
def _(mo, partD_pred):
    mo.stop(partD_pred.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_params_b = mo.ui.slider(
        start=0.5, stop=50, value=10, step=0.5,
        label="Model size (B parameters)",
    )

    # ─────────────────────────────────────────────────────────────────────
    # PART A: ICR Frontier
    # ─────────────────────────────────────────────────────────────────────
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; ML Lead, WildWatch Conservation
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have 2 million trail camera images. Training on the full dataset takes
                14 hours on our A100 node. Our wildlife biologists say many images are near-duplicates
                (same empty trail, different timestamp). Can we train on less data without losing accuracy?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The ICR Frontier: Diminishing Returns

        The **Information-Compute Ratio (ICR)** measures how much each additional
        training sample contributes to model improvement. In natural image datasets,
        sample informativeness follows a **power law**: a small fraction of samples
        contribute most of the learning signal, while the long tail adds near-zero
        gradient information.

        ```
        ICR(fraction) = 1 / (O x D)    where O = overlap, D = dataset fraction
        Accuracy(f) = Acc_max * (1 - alpha * exp(-beta * f))
        ```

        The ICR curve flattens dramatically past the **knee point**: the dataset
        fraction where marginal accuracy gains approach zero.
        """))

        items.append(partA_pred)

        if partA_pred.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the ICR explorer."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partA_frac, partA_redundancy], justify="start"))

        # ICR curve physics
        _redundancy_params = {
            "low": (0.98, 0.02, 8.0),
            "med": (0.98, 0.02, 4.0),
            "high": (0.98, 0.02, 2.5),
        }
        _red = partA_redundancy.value
        _max_acc, _alpha, _beta = _redundancy_params[_red]

        _fracs = np.linspace(0.05, 1.0, 50)
        _accs = [_max_acc * (1.0 - _alpha * math.exp(-_beta * f)) for f in _fracs]
        _threshold = _max_acc * 0.99

        _knee_frac = 1.0
        for _f, _a in zip(_fracs, _accs):
            if _a >= _threshold:
                _knee_frac = _f
                break

        _cur_frac = partA_frac.value / 100.0
        _cur_acc = _max_acc * (1.0 - _alpha * math.exp(-_beta * _cur_frac))

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=[f * 100 for f in _fracs], y=[a * 100 for a in _accs],
            mode="lines", line=dict(color=COLORS["BlueLine"], width=3),
            name="Accuracy vs Dataset Fraction",
        ))
        _fig.add_hline(y=_threshold * 100, line_dash="dash",
                       line_color=COLORS["GreenLine"],
                       annotation_text="99% of max accuracy")
        _fig.add_vline(x=_knee_frac * 100, line_dash="dot",
                       line_color=COLORS["OrangeLine"],
                       annotation_text=f"Knee: {_knee_frac*100:.0f}%")
        _fig.add_trace(go.Scatter(
            x=[_cur_frac * 100], y=[_cur_acc * 100],
            mode="markers", marker=dict(size=14, color=COLORS["RedLine"],
                                        symbol="diamond"),
            name=f"Current: {_cur_frac*100:.0f}%",
        ))
        _fig.update_layout(
            height=360,
            xaxis=dict(title="Dataset Fraction (%)", range=[0, 105]),
            yaxis=dict(title="Accuracy (%)", range=[90, 100]),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _full_train_hrs = 8.0
        _subset_train_hrs = _full_train_hrs * _cur_frac
        _time_saved = _full_train_hrs - _subset_train_hrs
        _acc_pct = _cur_acc * 100
        _max_pct = _max_acc * 100

        _knee_color = COLORS["GreenLine"] if _cur_frac <= _knee_frac * 1.1 else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Dataset Fraction</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_cur_frac*100:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{int(_cur_frac*1e6):,} of 1M images</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {_knee_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Accuracy Retained</div>
                <div style="font-size:1.7rem; font-weight:800; color:{_knee_color};">
                    {_acc_pct/_max_pct*100:.1f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_acc_pct:.1f}% of {_max_pct:.1f}%</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Training Time Saved</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {_time_saved:.1f} hrs</div>
                <div style="font-size:0.72rem; color:#94a3b8;">
                    {_subset_train_hrs:.1f} vs {_full_train_hrs:.1f} hrs on A100</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**ICR Curve -- Live Calculation** (`redundancy = {_red}`)

```
Accuracy(f) = {_max_acc} x (1 - {_alpha} x exp(-{_beta} x f))
At f = {_cur_frac:.2f}: Accuracy = {_acc_pct:.2f}%
Knee point:  f = {_knee_frac:.2f} ({_knee_frac*100:.0f}% of data retains 99% accuracy)
Time saved:  {_full_train_hrs:.1f} - {_subset_train_hrs:.1f} = {_time_saved:.1f} hours on A100
```
*Source: Chapter 9, ICR decay curve and coreset selection*
        """))

        _pred_val = partA_pred.value
        if _pred_val == "50":
            _rev = ("**Correct.** The ICR curve shows that 50% of a typical image dataset "
                    f"contributes near-zero learning signal. The knee is at ~{_knee_frac*100:.0f}%, "
                    "meaning you can discard half the data and retain 99%+ accuracy. "
                    "Redundancy in natural image datasets follows a power law.")
            _kind = "success"
        elif _pred_val == "80":
            _rev = ("**Too aggressive.** Discarding 80% pushes past the knee into the "
                    "steep part of the ICR curve. Accuracy drops below the 99% threshold. "
                    f"The actual knee is at ~{_knee_frac*100:.0f}%.")
            _kind = "warn"
        else:
            _rev = (f"**Too conservative.** You predicted only {_pred_val}% could be discarded. "
                    f"The ICR curve knee is at ~{_knee_frac*100:.0f}% -- most datasets have "
                    "massive redundancy, especially natural image datasets with near-duplicates.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_rev), kind=_kind))
        items.append(mo.accordion({
            "Math Peek: Informativeness-Coverage-Redundancy (ICR) Curve": mo.md("""
**Formula:**
$$
\\text{Accuracy}(k) \\approx A_{\\max} \\left(1 - e^{-\\alpha \\cdot k / N}\\right)
$$

**Variables:**
- **$k$**: number of selected samples
- **$N$**: total dataset size
- **$A_{\\max}$**: full-dataset accuracy (asymptote)
- **$\\alpha$**: informativeness decay rate (higher = faster saturation, more redundancy)

The exponential saturation means the last 50% of data contributes almost nothing to accuracy -- the cost of labeling/storing it is pure waste.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B: Selection Inequality
    # ─────────────────────────────────────────────────────────────────────
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Systems Architect, WildWatch
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;You proved we can discard half the data. The obvious move: score every image
                by informativeness and keep the top 50%. But my GPU time is not free.
                Is the scoring investment worth the training savings?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Selection Inequality

        For coreset selection to be systems-efficient, this inequality **must** hold:

        ```
        T_selection + T_train(subset)  <  T_train(full)
        ```

        If scoring 1M images costs 2.8 hours and training on 50% costs 4.2 hours,
        the total is 7.0 hours vs. 8.0 hours for full training. The margin is thin.
        On edge hardware where scoring is 10x slower, the inequality **breaks**.
        """))

        items.append(partB_pred)

        if partB_pred.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the Selection Inequality analyzer."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_coreset, partB_scorer], justify="start"))
        items.append(partB_hw)

        _frac = partB_coreset.value / 100.0
        _scorer = partB_scorer.value
        _hw = partB_hw.value
        _n_images = 1_000_000

        if _hw == "cloud":
            _tflops, _bw = A100_TFLOPS, A100_BW
            _hw_label = "A100"
        else:
            _tflops, _bw = JETSON_TFLOPS, JETSON_BW
            _hw_label = "Jetson Orin NX"

        _eta = 0.5
        if _scorer == "resnet":
            _score_flops = RESNET50_FLOPS
            _scorer_label = "ResNet-50"
        elif _scorer == "mobilenet":
            _score_flops = MOBILENET_FLOPS
            _scorer_label = "MobileNetV2"
        else:
            _score_flops = MOBILENET_FLOPS * 0.1
            _scorer_label = "Cached embeddings"

        _t_score_s = _n_images * _score_flops / (_tflops * 1e12 * _eta)
        _t_score_hrs = _t_score_s / 3600.0

        _full_train_hrs = 8.0 if _hw == "cloud" else 80.0
        _subset_train_hrs = _full_train_hrs * _frac
        _total_selection = _t_score_hrs + _subset_train_hrs
        _savings = _full_train_hrs - _total_selection
        _inequality_holds = _total_selection < _full_train_hrs

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="Scoring Cost", x=["Selection Pipeline"], y=[_t_score_hrs],
            marker_color=COLORS["OrangeLine"], opacity=0.88,
        ))
        _fig.add_trace(go.Bar(
            name="Subset Training", x=["Selection Pipeline"], y=[_subset_train_hrs],
            marker_color=COLORS["BlueLine"], opacity=0.88,
        ))
        _fig.add_trace(go.Bar(
            name="Full Training", x=["Full Pipeline"], y=[_full_train_hrs],
            marker_color=COLORS["GreenLine"] if not _inequality_holds else COLORS["Grey"],
            opacity=0.88,
        ))
        _fig.update_layout(
            barmode="stack", height=340,
            yaxis=dict(title="Time (hours)"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        if not _inequality_holds:
            items.append(mo.callout(mo.md(
                f"**SELECTION INEQUALITY VIOLATED.** "
                f"Scoring ({_t_score_hrs:.1f} hrs) + subset training ({_subset_train_hrs:.1f} hrs) "
                f"= {_total_selection:.1f} hrs > full training ({_full_train_hrs:.1f} hrs). "
                "The scoring cost exceeds the training savings. Just train on everything."
            ), kind="danger"))

        _sav_color = COLORS["GreenLine"] if _inequality_holds else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Scoring Time</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_t_score_hrs:.1f} hrs</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_scorer_label} on {_hw_label}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Subset Training</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_subset_train_hrs:.1f} hrs</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_frac*100:.0f}% of full dataset</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {_sav_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Net Savings</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_sav_color};">
                    {_savings:+.1f} hrs</div>
                <div style="font-size:0.72rem; color:#94a3b8;">
                    {"Inequality holds" if _inequality_holds else "VIOLATED"}</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Selection Inequality -- Live Calculation**

```
T_score       = {_n_images:,} images x {_score_flops:.2e} FLOPs / ({_tflops:.1f} TFLOPS x {_eta})
              = {_t_score_hrs:.2f} hours ({_scorer_label} on {_hw_label})
T_train(sub)  = {_full_train_hrs:.1f} hrs x {_frac:.2f} = {_subset_train_hrs:.1f} hrs
T_train(full) = {_full_train_hrs:.1f} hrs
Pipeline:     {_t_score_hrs:.1f} + {_subset_train_hrs:.1f} = {_total_selection:.1f} hrs
Savings:      {_savings:+.1f} hrs  {"(HOLDS)" if _inequality_holds else "(VIOLATED)"}
```
*Source: Chapter 9, Selection Inequality*
        """))

        if partB_pred.value == "yes_barely":
            _rev = ("**Correct.** On Cloud A100 with ResNet-50 scoring, the margin is thin: "
                    "only ~1.0 hours saved (12.5% reduction). The scoring cost consumes most "
                    "of the benefit. Switch to Edge to see the inequality break entirely.")
            _kind = "success"
        else:
            _rev = ("**The margin is thinner than most expect.** On Cloud A100 the savings "
                    "are only ~12.5%. On Edge hardware, the inequality breaks entirely "
                    "because scoring is 10x slower. Try switching the deployment context.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_rev), kind=_kind))
        items.append(mo.accordion({
            "Math Peek: The Selection Inequality": mo.md("""
**Formula:**
$$
T_{\\text{select}} + T_{\\text{train}}(\\text{subset}) < T_{\\text{train}}(\\text{full})
$$

Rearranging, selection is worthwhile only when:
$$
T_{\\text{select}} < T_{\\text{train}}(\\text{full}) - T_{\\text{train}}(\\text{subset})
$$

**Variables:**
- **$T_{\\text{select}}$**: time to score and rank all $N$ samples (proxy model inference)
- **$T_{\\text{train}}(\\text{subset})$**: training time on the selected coreset ($k$ samples)
- **$T_{\\text{train}}(\\text{full})$**: training time on the full dataset ($N$ samples)

On edge hardware, $T_{\\text{select}}$ grows 10x due to lower throughput, breaking the inequality even when the coreset is small.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C: Preprocessing Tax
    # ─────────────────────────────────────────────────────────────────────
    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Infrastructure Engineer, WildWatch
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We added aggressive data augmentation to prevent overfitting on the coreset.
                RandAugment with 5 transforms per image. Training is now slower than before we
                selected data. GPU utilization shows 27%. What is going on?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Preprocessing Tax

        CPU-side data augmentation and preprocessing is often the **true training
        bottleneck**, not GPU compute. A pipeline that applies RandAugment on CPU
        can starve an A100 that finishes its forward-backward pass in 12 ms while
        preprocessing takes 45+ ms per batch.

        ```
        GPU utilization = T_gpu / max(T_gpu, T_preprocess)
        T_preprocess    = N_transforms x T_per_transform / N_workers
        ```

        Per-transform costs: resize (2 ms), flip (0.5 ms), RandAugment transform (8 ms each).
        """))

        items.append(partC_pred)

        if partC_pred.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the pipeline analyzer."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partC_workers, partC_augment, partC_model], justify="start"))

        _n_workers = partC_workers.value
        _aug = partC_augment.value
        _model = partC_model.value

        _augment_costs = {
            "none": 3.0,
            "basic": 6.0,
            "ra5": 45.0,
            "ra10": 95.0,
        }
        _aug_labels = {
            "none": "None", "basic": "Basic", "ra5": "RandAugment-5", "ra10": "RandAug-10+MixUp",
        }
        _t_preprocess_per_worker = _augment_costs[_aug]
        _t_preprocess = _t_preprocess_per_worker / _n_workers

        if _model == "resnet":
            _t_gpu = 12.0
            _model_label = "ResNet-50"
        else:
            _t_gpu = 4.0
            _model_label = "MobileNetV2"

        _t_step = max(_t_gpu, _t_preprocess)
        _gpu_util = _t_gpu / _t_step if _t_step > 0 else 0
        _gpu_idle = max(0, _t_preprocess - _t_gpu)
        _gpu_util_pct = _gpu_util * 100

        # Gantt-style timeline
        _n_steps = 6
        _fig = go.Figure()
        for _i in range(_n_steps):
            _start_cpu = _i * _t_step
            _start_gpu = _start_cpu + max(0, _t_preprocess - _t_gpu) if _t_preprocess > _t_gpu else _start_cpu
            _fig.add_trace(go.Bar(
                x=[_t_preprocess], y=[f"Step {_i+1}"],
                base=_start_cpu, orientation="h",
                marker_color=COLORS["OrangeLine"], opacity=0.7,
                name="CPU Preprocess" if _i == 0 else None,
                showlegend=(_i == 0),
            ))
            _gpu_base = _start_cpu + _t_preprocess if _t_preprocess > _t_gpu else _start_cpu
            _fig.add_trace(go.Bar(
                x=[_t_gpu], y=[f"Step {_i+1}"],
                base=_gpu_base, orientation="h",
                marker_color=COLORS["BlueLine"], opacity=0.7,
                name="GPU Compute" if _i == 0 else None,
                showlegend=(_i == 0),
            ))
        _fig.update_layout(
            barmode="overlay", height=280,
            xaxis=dict(title="Time (ms)"),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _util_color = (COLORS["GreenLine"] if _gpu_util_pct > 70
                       else COLORS["OrangeLine"] if _gpu_util_pct > 40
                       else COLORS["RedLine"])
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {_util_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">GPU Utilization</div>
                <div style="font-size:2rem; font-weight:800; color:{_util_color};">
                    {_gpu_util_pct:.0f}%</div>
                <div style="font-size:0.72rem; color:#94a3b8;">
                    {_t_gpu:.1f} ms compute / {_t_step:.1f} ms step</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">CPU Preprocess</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_t_preprocess:.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">
                    {_aug_labels[_aug]} / {_n_workers} workers</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">GPU Idle Time</div>
                <div style="font-size:2rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_gpu_idle:.1f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">per step</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Pipeline -- Live Calculation** (`{_aug_labels[_aug]}, {_n_workers} workers, {_model_label}`)

```
T_preprocess = {_t_preprocess_per_worker:.1f} ms / {_n_workers} workers = {_t_preprocess:.1f} ms
T_gpu        = {_t_gpu:.1f} ms (forward + backward on A100)
T_step       = max({_t_preprocess:.1f}, {_t_gpu:.1f}) = {_t_step:.1f} ms
GPU util     = {_t_gpu:.1f} / {_t_step:.1f} = {_gpu_util_pct:.1f}%
GPU idle     = max(0, {_t_preprocess:.1f} - {_t_gpu:.1f}) = {_gpu_idle:.1f} ms per step
```
*Source: Chapter 9, preprocessing pipeline costs; Chapter 4, data loading*
        """))

        if partC_pred.value == "25":
            _rev = ("**Correct.** With RandAugment-5 and 8 workers, preprocessing takes ~5.6 ms "
                    "while GPU compute is 12 ms -- in this configuration the GPU is actually the "
                    "bottleneck. However, increase augmentation to RA-10+MixUp or reduce workers "
                    "to 2-3 and watch GPU utilization plummet below 30%.")
            _kind = "success"
        else:
            _rev = ("**Explore the controls.** With aggressive augmentation and few workers, "
                    "GPU utilization drops below 30%. The expensive A100 sits idle waiting for "
                    "the cheap CPUs. This is the preprocessing tax.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_rev), kind=_kind))
        items.append(mo.accordion({
            "Math Peek: GPU Utilization Under Preprocessing Bottleneck": mo.md("""
**Formula:**
$$
\\text{GPU Utilization} = \\frac{T_{\\text{GPU}}}{T_{\\text{GPU}} + \\max(0,\\; T_{\\text{CPU}} - T_{\\text{GPU}})}
$$

When CPU preprocessing overlaps with GPU compute, the step time is $\\max(T_{\\text{CPU}}, T_{\\text{GPU}})$:
$$
T_{\\text{step}} = \\max\\!\\left(\\frac{T_{\\text{aug}} \\cdot B}{W},\\; T_{\\text{GPU}}\\right)
$$

**Variables:**
- **$T_{\\text{aug}}$**: per-sample augmentation time on one CPU core
- **$B$**: batch size
- **$W$**: number of CPU workers (data-loading parallelism)
- **$T_{\\text{GPU}}$**: forward + backward pass time on accelerator
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D: Chinchilla Scaling
    # ─────────────────────────────────────────────────────────────────────
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid #6366f1; background:#f0f4ff;
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CTO, LanguageAI Corp
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have a fixed compute budget of 10<sup>21</sup> FLOPs for our next
                training run. Our team is split: half want a 10B model on 200B tokens, the other
                half want a 3B model on 660B tokens. Which configuration wins?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Compute-Optimal Frontier

        The **Chinchilla scaling law** determines whether a training run is
        **data-starved** (more data would help) or **compute-starved** (bigger model
        would help). The optimal ratio requires ~20x more data tokens than parameters.

        ```
        L(N, D) = A / N^alpha + B / D^beta + E
        Chinchilla optimal: D_opt ~ 20 x N
        Compute: C ~ 6 x N x D  (FLOPs)
        ```

        Most teams over-allocate to model size and under-allocate to data.
        """))

        items.append(partD_pred)

        if partD_pred.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction above to unlock the scaling frontier."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partD_params_b)

        _N = partD_params_b.value
        _budget = 1e21
        _D_tokens = _budget / (6 * _N * 1e9)
        _D_B = _D_tokens / 1e9

        _N_opt = math.sqrt(_budget / 120) / 1e9
        _D_opt = 20 * _N_opt

        _A, _alpha_l = 406.4, 0.34
        _B, _beta_l = 410.7, 0.28
        _E_loss = 1.69

        def _loss(n_b, d_b):
            if n_b <= 0 or d_b <= 0:
                return 10.0
            return _A / (n_b ** _alpha_l) + _B / (d_b ** _beta_l) + _E_loss

        _cur_loss = _loss(_N, _D_B)
        _opt_loss = _loss(_N_opt, _D_opt)

        _fig = go.Figure()
        _ns = np.logspace(math.log10(0.5), math.log10(50), 80)
        _ds = [_budget / (6 * n * 1e9) / 1e9 for n in _ns]
        _fig.add_trace(go.Scatter(
            x=_ns.tolist(), y=_ds, mode="lines",
            line=dict(color=COLORS["Grey"], width=2, dash="dash"),
            name="IsoFLOP: 10^21",
        ))

        _chinch_ns = np.logspace(math.log10(0.5), math.log10(50), 50)
        _chinch_ds = [20 * n for n in _chinch_ns]
        _fig.add_trace(go.Scatter(
            x=_chinch_ns.tolist(), y=_chinch_ds,
            mode="lines", line=dict(color=COLORS["GreenLine"], width=2),
            name="Chinchilla Optimal (D = 20N)",
        ))
        _fig.add_trace(go.Scatter(
            x=[_N_opt], y=[_D_opt],
            mode="markers+text",
            marker=dict(size=14, color=COLORS["GreenLine"], symbol="star"),
            text=[f"Optimal: {_N_opt:.1f}B"], textposition="top right",
            name="Compute-Optimal",
        ))
        _fig.add_trace(go.Scatter(
            x=[_N], y=[_D_B],
            mode="markers+text",
            marker=dict(size=14, color=COLORS["RedLine"], symbol="diamond"),
            text=[f"Current: {_N:.1f}B"], textposition="bottom left",
            name="Your Configuration",
        ))
        _fig.update_layout(
            height=400,
            xaxis=dict(title="Model Size (B parameters)", type="log"),
            yaxis=dict(title="Dataset Size (B tokens)", type="log"),
            legend=dict(orientation="h", y=1.14, x=0),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _ratio = _D_B / (_N * 20) if _N > 0 else 0
        if _ratio < 0.8:
            _regime, _regime_color = "DATA-STARVED", COLORS["RedLine"]
            _regime_advice = "More data would help more than a bigger model."
        elif _ratio > 1.2:
            _regime, _regime_color = "COMPUTE-STARVED", COLORS["OrangeLine"]
            _regime_advice = "A bigger model would use this data more efficiently."
        else:
            _regime, _regime_color = "NEAR-OPTIMAL", COLORS["GreenLine"]
            _regime_advice = "Close to the Chinchilla-optimal balance."

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {_regime_color}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Regime</div>
                <div style="font-size:1.3rem; font-weight:800; color:{_regime_color};">{_regime}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_regime_advice}</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Your Loss</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_cur_loss:.2f}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_N:.1f}B params, {_D_B:.0f}B tokens</div>
            </div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Optimal Loss</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_opt_loss:.2f}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_N_opt:.1f}B params, {_D_opt:.0f}B tokens</div>
            </div>
        </div>
        """))

        items.append(mo.md(f"""
**Chinchilla Scaling -- Live Calculation**

```
Budget:    C = 10^21 FLOPs
Your cfg:  N = {_N:.1f}B, D = C/(6N) = {_D_B:.0f}B tokens
Optimal:   N* = sqrt(C/120) = {_N_opt:.1f}B, D* = 20N* = {_D_opt:.0f}B tokens
Your loss: L = {_cur_loss:.2f}   Optimal loss: L* = {_opt_loss:.2f}
Regime:    {_regime} (D/20N = {_ratio:.2f})
```
*Source: Chapter 9, Chinchilla scaling law (Hoffmann et al., 2022)*
        """))

        if partD_pred.value == "3b_660b":
            _rev = ("**Correct.** The 3B model on 660B tokens sits closest to the "
                    "Chinchilla-optimal point. The 10B model is data-starved and "
                    "the 30B model wastes FLOPs on a model too large for its data budget.")
            _kind = "success"
        else:
            _rev = ("**The Chinchilla-optimal point favors more data, not a bigger model.** "
                    f"The optimal split for 10^21 FLOPs is ~{_N_opt:.1f}B parameters on "
                    f"~{_D_opt:.0f}B tokens. Most teams over-allocate to model size.")
            _kind = "warn"
        items.append(mo.callout(mo.md(_rev), kind=_kind))
        items.append(mo.accordion({
            "Math Peek: Chinchilla Scaling Law": mo.md("""
**Formula:**
$$
L(N, D) = \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}} + L_{\\infty}
$$

For compute-optimal allocation under budget $C \\approx 6ND$:
$$
N^{*} = \\sqrt{\\frac{C}{120}}, \\qquad D^{*} = 20\\,N^{*}
$$

**Variables:**
- **$N$**: number of model parameters
- **$D$**: number of training tokens
- **$C$**: total compute budget (FLOPs)
- **$\\alpha, \\beta$**: scaling exponents ($\\approx 0.34, 0.28$)
- **$L_{\\infty}$**: irreducible loss

The optimal ratio is $D/N \\approx 20$ -- most teams over-allocate to model size and under-allocate to data.
""")
        }))
        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS
    # ─────────────────────────────────────────────────────────────────────
    def build_synthesis():
        return mo.vstack([
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
                        <strong>1. Half your data is dead weight.</strong>
                        The ICR curve shows that 50% of a typical image dataset contributes
                        near-zero learning signal. The informativeness distribution follows a
                        power law, not a uniform distribution.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Selection has a cost that can exceed the savings.</strong>
                        The Selection Inequality T_sel + T_train(subset) &lt; T_train(full) has
                        thin margins on cloud and breaks entirely on edge hardware.
                    </div>
                    <div>
                        <strong>3. The GPU is only as fast as its data pipeline.</strong>
                        CPU preprocessing with aggressive augmentation can starve an A100,
                        dropping GPU utilization below 30%.
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
                        <strong>Lab 10:</strong> The Compression Paradox -- you selected the right
                        data. Now discover that compressing the model itself is free at 4x but a
                        hardware trap at 90% sparsity.
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
                        <strong>Read:</strong> the Data Selection chapter for the full ICR derivation.<br/>
                        <strong>Build:</strong> TinyTorch Module 09 -- coreset selection.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: ICR Frontier": build_part_a(),
        "Part B: Selection Inequality": build_part_b(),
        "Part C: Preprocessing Tax": build_part_c(),
        "Part D: Scaling Frontier": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ═════════════════════════════════════════════════════════════════════════════
# CELL 5: LEDGER HUD
# ═════════════════════════════════════════════════════════════════════════════

# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_pred, partB_pred, partC_pred, partD_pred):
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=9, design={
            "lab": "data_selection",
            "completed": True,
            "icr_discard_fraction": partA_pred.value,
            "selection_inequality_verdict": partB_pred.value,
            "gpu_utilization_prediction": partC_pred.value,
            "chinchilla_optimal_config": partD_pred.value,
        })
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">09 &middot; Data Selection</span>
        <span style="flex:1;"></span>
        <span class="hud-label">CH</span>
        <span class="hud-value">9</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">COMPLETE</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
