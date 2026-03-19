import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
async def _():
    import marimo as mo
    import sys
    import math
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np

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
    import mlsysim

    H100_TDP_W = mlsysim.Hardware.Cloud.H100.tdp.m_as("W")

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        await ledger.load_async()
    return (
        COLORS, H100_TDP_W, LAB_CSS,
        apply_plotly_theme, go, ledger, math, mo, np,
    )


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
                Machine Learning Systems &middot; Volume I &middot; Lab 15
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                No Free Fairness
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Impossibility &middot; Accuracy Cost &middot; Explainability Tax &middot; Carbon
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                Fairness is mathematically impossible to achieve perfectly, fixing it
                costs accuracy, explaining it costs latency, and all of it costs carbon
                &mdash; and none of these costs are optional.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~48 min</span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 15: Responsible Engineering</span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Impossibility Theorem</span>
                <span class="badge badge-warn">Pareto Frontier</span>
                <span class="badge badge-fail">Carbon Budget</span>
            </div>
        </div>
        """),
    ])
    return


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
                Learning Objectives</div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Prove the impossibility</strong>
                    &mdash; equal accuracy across groups does not guarantee equal error rates
                    when base rates differ.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the accuracy cost of fairness</strong>
                    &mdash; the first 10pp gap reduction costs 3&ndash;5% accuracy (sweet spot);
                    the last 5pp costs 10%+ (cliff).</div>
                <div style="margin-bottom: 3px;">3. <strong>Calculate explanation overhead</strong>
                    &mdash; SHAP with 50 features adds 50x latency to each prediction.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    Classification metrics from @sec-model-training &middot;
                    Model serving SLOs from @sec-model-serving</div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~48 min</strong><br/>
                    A: ~12 &middot; B: ~12 &middot; C: ~10 &middot; D: ~8 min</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 12px -28px 0 -28px;
                    padding: 16px 28px 0 28px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question</div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;Your loan model has 92% accuracy for both demographic groups.
                Is it fair &mdash; and what does it cost to make it fairer?&rdquo;
            </div>
        </div>
    </div>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** &mdash; Complete before this lab:

    - **@sec-responsible-engineering** &mdash; Chouldechova impossibility theorem,
      fairness-accuracy Pareto frontier, explainability methods, and carbon accounting.
    """), kind="info")
    return


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAB CELL
# ═════════════════════════════════════════════════════════════════════════════
@app.cell(hide_code=True)
def _(COLORS, H100_TDP_W, apply_plotly_theme, go, math, mo, np):

    # ── Part A widgets ────────────────────────────────────────────────────────
    partA_pred = mo.ui.radio(
        options={
            "A) Yes -- equal accuracy guarantees equal error rates": "equal",
            "B) Nearly equal -- small differences from noise": "nearly",
            "C) No -- Group B FPR is 3x higher despite equal accuracy": "3x",
            "D) No -- Group A FPR is higher (more positives)": "a_higher",
        },
        label="Loan model: 92% accuracy for both groups. Group A base rate 30%, Group B 10%. FPR equal?",
    )
    return (partA_pred,)

@app.cell(hide_code=True)
def _(mo, partA_pred):
    partA_base_a = mo.ui.slider(start=5, stop=50, value=30, step=5,
                                 label="Group A base rate (%)")
    partA_base_b = mo.ui.slider(start=5, stop=50, value=10, step=5,
                                 label="Group B base rate (%)")
    partA_threshold = mo.ui.slider(start=0.10, stop=0.90, value=0.50, step=0.05,
                                    label="Classification threshold")

    # ── Part B widgets ────────────────────────────────────────────────────────
    partB_pred = mo.ui.radio(
        options={
            "A) ~1% (fairness is nearly free)": "1pct",
            "B) ~3-5% (sweet spot of the frontier)": "3-5pct",
            "C) ~10% (significant sacrifice)": "10pct",
            "D) ~15% (proportional to gap reduction)": "15pct",
        },
        label="Apply equalized odds to reduce FPR gap from 15% to 5%. Accuracy loss?",
    )
    return (partB_pred,)

@app.cell(hide_code=True)
def _(mo, partB_pred):
    partB_target_gap = mo.ui.slider(start=0, stop=20, value=5, step=1,
                                     label="Target FPR gap (pp)")
    partB_method = mo.ui.dropdown(
        options={"Threshold Adjustment": "threshold", "Reweighting": "reweight",
                 "Adversarial Debiasing": "adversarial"},
        value="Threshold Adjustment", label="Mitigation method")

    # ── Part C widgets ────────────────────────────────────────────────────────
    partC_pred = mo.ui.radio(
        options={
            "A) ~2x (modest overhead)": "2x",
            "B) ~10x (significant but manageable)": "10x",
            "C) ~50x (one forward pass per feature)": "50x",
            "D) ~2500x (all permutation subsets)": "2500x",
        },
        label="Loan model has 50 features. How much does SHAP add to inference latency?",
    )
    return (partC_pred,)

@app.cell(hide_code=True)
def _(mo, partC_pred):
    partC_features = mo.ui.slider(start=10, stop=200, value=50, step=10,
                                   label="Number of features")
    partC_method = mo.ui.dropdown(
        options={"SHAP (Kernel)": "shap", "LIME": "lime",
                 "Feature Importance": "fi", "None": "none"},
        value="SHAP (Kernel)", label="Explanation method")

    # ── Part D widgets ────────────────────────────────────────────────────────
    partD_pred = mo.ui.radio(
        options={
            "A) ~2x (modest increase)": "2x",
            "B) ~10x (significant but necessary)": "10x",
            "C) ~60x (52x retraining + 5x explanations)": "60x",
            "D) ~100x (even higher)": "100x",
        },
        label="Weekly retraining + SHAP for 10% of predictions. Carbon vs baseline?",
    )
    return (partD_pred,)

@app.cell(hide_code=True)
def _(mo, partD_pred):
    partD_retrain_freq = mo.ui.dropdown(
        options={"Weekly": 52, "Monthly": 12, "Quarterly": 4, "Once": 1},
        value="Weekly", label="Retraining frequency")
    partD_explain_pct = mo.ui.slider(start=0, stop=100, value=10, step=5,
                                      label="Explanation coverage (%)")
    partD_grid_mix = mo.ui.dropdown(
        options={"Clean (hydro, 20 gCO2/kWh)": 20, "Mixed (400 gCO2/kWh)": 400,
                 "Coal-heavy (800 gCO2/kWh)": 800},
        value="Mixed (400 gCO2/kWh)", label="Grid carbon intensity")

    # ═════════════════════════════════════════════════════════════════════════
    # PART A — The Fairness Illusion
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_a():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Chief Compliance Officer, EquiLend</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our loan approval model achieves 92% accuracy for both Group A and Group B.
                Can you confirm this means the model treats both groups equally?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The Chouldechova Impossibility Theorem

When **base rates differ** between groups, a calibrated classifier cannot simultaneously
satisfy equal FPR, equal FNR, and equal PPV. Equal accuracy does **not** mean equal
treatment. This is a mathematical impossibility, not an engineering limitation.

The key: accuracy = (TP + TN) / total. With different base rates, the composition
of TP and TN differs between groups even when the sum (accuracy) is identical.
        """))

        items.append(partA_pred)
        if partA_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the fairness simulator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_base_a, partA_base_b, partA_threshold], widths="equal"))

        _ba = partA_base_a.value / 100  # Group A base rate
        _bb = partA_base_b.value / 100  # Group B base rate
        _thresh = partA_threshold.value
        _n = 1000  # sample size per group

        # Simulate confusion matrices for each group
        # With a fixed threshold and Gaussian score distributions
        def _compute_metrics(base_rate, threshold, n=1000):
            positives = int(n * base_rate)
            negatives = n - positives
            # Model: P(score > threshold | positive) ~ base_rate-dependent
            # Higher base rate -> better separation
            tpr = min(0.98, 0.85 + (1 - threshold) * 0.3)
            fpr_raw = max(0.01, threshold * 0.2 + (1 - base_rate) * 0.08)
            tp = int(positives * tpr)
            fn = positives - tp
            fp = int(negatives * fpr_raw)
            tn = negatives - fp
            acc = (tp + tn) / n * 100
            fpr = fp / negatives * 100 if negatives > 0 else 0
            fnr = fn / positives * 100 if positives > 0 else 0
            ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            return {"acc": acc, "fpr": fpr, "fnr": fnr, "ppv": ppv,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn}

        _ma = _compute_metrics(_ba, _thresh)
        _mb = _compute_metrics(_bb, _thresh)
        _fpr_gap = abs(_ma["fpr"] - _mb["fpr"])

        # Grouped bar chart
        _metrics = ["Accuracy", "FPR", "FNR", "PPV"]
        _va = [_ma["acc"], _ma["fpr"], _ma["fnr"], _ma["ppv"]]
        _vb = [_mb["acc"], _mb["fpr"], _mb["fnr"], _mb["ppv"]]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name=f"Group A (base={_ba*100:.0f}%)",
                               x=_metrics, y=_va, marker_color=COLORS['BlueLine'], opacity=0.88))
        _fig.add_trace(go.Bar(name=f"Group B (base={_bb*100:.0f}%)",
                               x=_metrics, y=_vb, marker_color=COLORS['OrangeLine'], opacity=0.88))
        _fig.update_layout(barmode="group", height=360,
                           yaxis=dict(title="Percentage (%)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Gap cards
        _gap_col = COLORS['RedLine'] if _fpr_gap > 10 else (
            COLORS['OrangeLine'] if _fpr_gap > 5 else COLORS['GreenLine'])
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Accuracy Gap</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {abs(_ma['acc']-_mb['acc']):.1f} pp</div>
                <div style="font-size:0.72rem; color:#94a3b8;">Looks equal</div></div>
            <div style="padding:16px; border:2px solid {_gap_col}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_gap_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">FPR Gap</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_gap_col};">
                    {_fpr_gap:.1f} pp</div>
                <div style="font-size:0.72rem; color:#94a3b8;">NOT equal</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">FNR Gap</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {abs(_ma['fnr']-_mb['fnr']):.1f} pp</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">PPV Gap</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {abs(_ma['ppv']-_mb['ppv']):.1f} pp</div></div>
        </div>"""))

        if _fpr_gap > 5:
            items.append(mo.callout(mo.md(
                f"**Unequal treatment despite equal accuracy.** FPR gap = {_fpr_gap:.1f} pp. "
                "Group B (lower base rate) has higher false positive rate. "
                "This is the Chouldechova impossibility in action."), kind="danger"))

        # Live confusion matrix comparison
        items.append(mo.md(f"""
**Confusion Matrix Comparison &mdash; Live** (`base_A={_ba*100:.0f}%, base_B={_bb*100:.0f}%, threshold={_thresh:.2f}`)

```
Group A (base rate = {_ba*100:.0f}%):
  TP={_ma['tp']:>4}  FP={_ma['fp']:>4}  |  Acc={_ma['acc']:.1f}%  FPR={_ma['fpr']:.1f}%  FNR={_ma['fnr']:.1f}%  PPV={_ma['ppv']:.1f}%
  FN={_ma['fn']:>4}  TN={_ma['tn']:>4}  |

Group B (base rate = {_bb*100:.0f}%):
  TP={_mb['tp']:>4}  FP={_mb['fp']:>4}  |  Acc={_mb['acc']:.1f}%  FPR={_mb['fpr']:.1f}%  FNR={_mb['fnr']:.1f}%  PPV={_mb['ppv']:.1f}%
  FN={_mb['fn']:>4}  TN={_mb['tn']:>4}  |

Accuracy gap:  {abs(_ma['acc']-_mb['acc']):.1f} pp   (looks equal)
FPR gap:       {_fpr_gap:.1f} pp   (NOT equal)
FNR gap:       {abs(_ma['fnr']-_mb['fnr']):.1f} pp
PPV gap:       {abs(_ma['ppv']-_mb['ppv']):.1f} pp
```
*Source: Chouldechova impossibility from @sec-responsible-engineering-fairness*
        """))

        if partA_pred.value == "3x":
            items.append(mo.callout(mo.md(
                "**Correct.** Equal accuracy does not guarantee equal error rates. "
                "The impossibility theorem proves this mathematically when base rates differ."), kind="success"))
        elif partA_pred.value == "equal":
            items.append(mo.callout(mo.md(
                "**Equal accuracy does NOT mean equal fairness.** "
                f"Despite similar accuracy, FPR gap = {_fpr_gap:.1f} pp. "
                "This is mathematically inescapable."), kind="warn"))
        else:
            items.append(mo.callout(mo.md(
                f"**The FPR gap is {_fpr_gap:.1f} pp despite equal accuracy.** "
                "Different base rates force different error distributions."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART B — The Price of Fairness
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_b():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Escalation &middot; Chief Risk Officer, EquiLend</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Perfect fairness is impossible. How much accuracy must we sacrifice
                to reduce the disparity to an acceptable level? What is the sweet spot?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## The Fairness-Accuracy Pareto Frontier

The frontier has a specific shape: **concave near the sweet spot** (large fairness
gains for small accuracy cost) and **steep near strict equality** (small fairness
gains for large accuracy cost).

```
Accuracy_loss = k * (1 / max(target_gap, 1))^alpha
```

The first 10pp of gap reduction is cheap. The last 5pp is expensive.
        """))

        items.append(partB_pred)
        if partB_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the Pareto frontier."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_target_gap, partB_method], widths="equal"))

        _target = partB_target_gap.value
        _method = partB_method.value

        # Pareto frontier model
        _method_factors = {"threshold": 1.0, "reweight": 0.8, "adversarial": 0.6}
        _k = _method_factors[_method]
        _unconstrained_gap = 15.0  # pp

        # Accuracy loss as function of target gap
        _gaps = np.linspace(0, 15, 100)
        _losses = [_k * max(0, 3.0 * (15 - g) / 15 + 2.0 * max(0, 5 - g) / 5) for g in _gaps]

        _curr_loss = _k * max(0, 3.0 * (15 - _target) / 15 + 2.0 * max(0, 5 - _target) / 5)

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=list(_gaps), y=_losses, mode='lines',
                                   name='Pareto Frontier',
                                   line=dict(color=COLORS['BlueLine'], width=3)))
        _fig.add_trace(go.Scatter(x=[_target], y=[_curr_loss], mode='markers',
                                   name=f'Target gap={_target}pp',
                                   marker=dict(color=COLORS['RedLine'], size=14, symbol='diamond')))
        # Sweet spot annotation
        _fig.add_trace(go.Scatter(x=[5], y=[_k * (3.0 * 10/15 + 2.0 * 0/5)], mode='markers',
                                   name='Sweet Spot',
                                   marker=dict(color=COLORS['GreenLine'], size=12, symbol='star')))
        _fig.update_layout(height=360,
                           xaxis=dict(title="FPR Gap (pp) -- lower is fairer", autorange="reversed"),
                           yaxis=dict(title="Accuracy Cost (pp)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=50, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:2px solid {COLORS['OrangeLine']}; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Accuracy Cost</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_curr_loss:.1f} pp</div>
                <div style="font-size:0.72rem; color:#94a3b8;">gap: {_unconstrained_gap:.0f} to {_target} pp</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:150px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Method</div>
                <div style="font-size:1.2rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_method.replace('_', ' ').title()}</div>
                <div style="font-size:0.72rem; color:#94a3b8;">efficiency: {_k:.1f}x</div></div>
        </div>"""))

        items.append(mo.md(f"""
**Pareto Frontier &mdash; Live** (`target_gap={_target}pp, method={_method}, k={_k:.1f}`)

```
Unconstrained FPR gap = {_unconstrained_gap:.0f} pp
Target FPR gap        = {_target} pp
Gap reduction         = {_unconstrained_gap - _target:.0f} pp

Accuracy cost  = k * [3.0*(15-target)/15 + 2.0*max(0, 5-target)/5]
               = {_k:.1f} * [{3.0*(15-_target)/15:.2f} + {2.0*max(0,5-_target)/5:.2f}]
               = {_curr_loss:.1f} pp

Sweet spot (gap=5pp): ~{_k * (3.0*10/15):.1f} pp accuracy cost
Cliff (gap=0pp):      ~{_k * (3.0 + 2.0):.1f} pp accuracy cost
```
*Source: fairness-accuracy frontier from @sec-responsible-engineering-mitigation*
        """))

        if partB_pred.value == "3-5pct":
            items.append(mo.callout(mo.md(
                "**Correct.** The 15pp to 5pp reduction is in the sweet spot of the "
                "Pareto frontier. Pushing from 5pp to 0pp costs disproportionately more."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**The accuracy cost is ~{_curr_loss:.1f} pp.** The Pareto frontier "
                "is concave: first gains are cheap, strict equality is expensive."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART C — The Explainability Tax
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_c():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Regulatory Requirement &middot; Compliance Team, EquiLend</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Under ECOA, denied applicants must receive an explanation of which factors
                contributed to the decision. How much compute does this cost?&rdquo;</div>
        </div>"""))

        items.append(mo.md("""
## SHAP Explanations Require N Additional Forward Passes

Kernel SHAP computes marginal contributions by running the model with each feature
masked. For N features, this requires approximately **N forward passes**:

```
Explanation latency = N_features * base_inference_latency
```

Exact SHAP requires 2^N evaluations (infeasible for N > 30).
LIME uses ~100 samples (fixed). Feature importance is free (~1x).
        """))

        items.append(partC_pred)
        if partC_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the explainability calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_features, partC_method], widths="equal"))

        _nf = partC_features.value
        _method = partC_method.value
        _base_ms = 2.0  # base inference latency for a tabular model

        _multipliers = {"shap": _nf, "lime": 100, "fi": 1, "none": 0}
        _method_labels = {"shap": "SHAP (Kernel)", "lime": "LIME", "fi": "Feature Importance", "none": "None"}
        _mult = _multipliers[_method]
        _total_ms = _base_ms * (1 + _mult)
        _slo = 100  # ms
        _slo_viol = _total_ms > _slo

        # Comparison chart
        _methods = ["None", "Feature Imp.", "LIME", "SHAP"]
        _latencies = [_base_ms, _base_ms * 2, _base_ms * 101, _base_ms * (_nf + 1)]
        _colors = [COLORS['GreenLine'], COLORS['GreenLine'],
                   COLORS['OrangeLine'] if _base_ms * 101 < _slo else COLORS['RedLine'],
                   COLORS['RedLine'] if _base_ms * (_nf + 1) > _slo else COLORS['OrangeLine']]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_methods, y=_latencies, marker_color=_colors, opacity=0.88))
        _fig.add_hline(y=_slo, line_dash="dash", line_color=COLORS['GreenLine'],
                       annotation_text=f"SLO = {_slo} ms")
        _fig.update_layout(height=340, yaxis=dict(title="Total Latency (ms)",
                                                    type="log", gridcolor="#f1f5f9"),
                           margin=dict(l=50, r=20, t=40, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _tot_col = COLORS['RedLine'] if _slo_viol else COLORS['GreenLine']
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['BlueLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Base Inference</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['BlueLine']};">{_base_ms:.1f} ms</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Explanation Overhead</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['OrangeLine']};">{_mult}x</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{_method_labels[_method]}</div></div>
            <div style="padding:16px; border:2px solid {_tot_col}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {_tot_col}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Total Latency</div>
                <div style="font-size:1.5rem; font-weight:800; color:{_tot_col};">{_total_ms:.0f} ms</div>
                <div style="font-size:0.72rem; color:#94a3b8;">{"SLO VIOLATED" if _slo_viol else "Within SLO"}</div></div>
        </div>"""))

        if _slo_viol:
            items.append(mo.callout(mo.md(
                f"**SLO VIOLATED.** {_method_labels[_method]} with {_nf} features: "
                f"{_total_ms:.0f} ms > {_slo} ms SLO. Consider LIME or async explanations."), kind="danger"))

        # Method comparison table
        items.append(mo.md(f"""
**Explanation Method Comparison** (`{_nf} features, base inference = {_base_ms:.1f} ms`)

| Method | Forward Passes | Total Latency | Fidelity | Regulatory |
|--------|---------------|---------------|----------|------------|
| None | 0 | {_base_ms:.1f} ms | N/A | Non-compliant |
| Feature Importance | 1 | {_base_ms*2:.1f} ms | Low | Insufficient |
| LIME | ~100 | {_base_ms*101:.0f} ms | Medium | Acceptable |
| SHAP (Kernel) | ~{_nf} | {_base_ms*(_nf+1):.0f} ms | High | Gold standard |
| SHAP (Exact) | 2^{_nf} | Infeasible | Perfect | Infeasible |

*Source: explainability cost analysis from @sec-responsible-engineering-explanations*
        """))

        if partC_pred.value == "50x":
            items.append(mo.callout(mo.md(
                "**Correct.** SHAP with 50 features requires ~50 forward passes. "
                "Each pass has the same latency as the original prediction."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**SHAP overhead = {_nf}x.** Each feature requires a separate "
                "model evaluation to compute its marginal contribution."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # PART D — The Carbon Cost of Responsibility
    # ═════════════════════════════════════════════════════════════════════════
    def build_part_d():
        items = []
        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Board Inquiry &middot; Sustainability Committee, EquiLend</div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our responsible AI stack requires weekly retraining and SHAP explanations
                for all denied applications. What is the annual carbon footprint compared
                to our baseline model that was trained once?&rdquo;</div>
        </div>"""))

        items.append(partD_pred)
        if partD_pred.value is None:
            items.append(mo.callout(mo.md(
                "Select your prediction to unlock the carbon calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_retrain_freq, partD_explain_pct, partD_grid_mix], widths="equal"))

        _retrains_yr = partD_retrain_freq.value
        _explain_frac = partD_explain_pct.value / 100
        _carbon_intensity = partD_grid_mix.value  # gCO2/kWh

        # Training energy: 4 A100-hours per retrain ~= 4 * 0.4 kW * 1 hr = 1.6 kWh
        _train_kwh = 1.6  # kWh per retraining run
        _baseline_carbon = _train_kwh * _carbon_intensity / 1000  # kg CO2

        # Retraining carbon
        _retrain_carbon = _retrains_yr * _train_kwh * _carbon_intensity / 1000

        # Serving energy: 10M predictions/year, 2ms each
        _predictions_yr = 10_000_000
        _base_serve_kwh = _predictions_yr * 0.002 / 3600 * H100_TDP_W / 1000  # kWh
        _explain_serve_kwh = _predictions_yr * _explain_frac * 50 * 0.002 / 3600 * H100_TDP_W / 1000
        _serve_carbon = (_base_serve_kwh + _explain_serve_kwh) * _carbon_intensity / 1000

        _total_carbon = _retrain_carbon + _serve_carbon
        _baseline_total = _baseline_carbon + _base_serve_kwh * _carbon_intensity / 1000
        _ratio = _total_carbon / _baseline_total if _baseline_total > 0 else 1

        # Stacked bar chart
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Baseline (train-once)", x=["Baseline"],
                               y=[_baseline_total], marker_color=COLORS['GreenLine'], opacity=0.88))
        _fig.add_trace(go.Bar(name="Retraining Carbon", x=["Responsible AI"],
                               y=[_retrain_carbon], marker_color=COLORS['BlueLine'], opacity=0.88))
        _fig.add_trace(go.Bar(name="Serving + Explanations", x=["Responsible AI"],
                               y=[_serve_carbon], marker_color=COLORS['OrangeLine'], opacity=0.88))
        _fig.update_layout(barmode="stack", height=340,
                           yaxis=dict(title="Annual Carbon (kg CO2)", gridcolor="#f1f5f9"),
                           legend=dict(orientation="h", y=1.12, x=0),
                           margin=dict(l=60, r=20, t=60, b=40))
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['GreenLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Baseline Carbon</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['GreenLine']};">{_baseline_total:.1f} kg</div></div>
            <div style="padding:16px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['RedLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Responsible AI Carbon</div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['RedLine']};">{_total_carbon:.1f} kg</div></div>
            <div style="padding:16px; border:1px solid #e2e8f0; border-radius:10px;
                        min-width:130px; text-align:center; background:white;
                        border-top:3px solid {COLORS['OrangeLine']}; flex:1;">
                <div style="color:#94a3b8; font-size:0.78rem; font-weight:600;">Carbon Multiplier</div>
                <div style="font-size:1.7rem; font-weight:800; color:{COLORS['OrangeLine']};">{_ratio:.0f}x</div></div>
        </div>"""))

        items.append(mo.md(f"""
**Carbon Budget &mdash; Live** (`{_retrains_yr} retrains/yr, {_explain_frac*100:.0f}% coverage, {_carbon_intensity} gCO2/kWh`)

```
Baseline:
  Training carbon  = 1 * {_train_kwh:.1f} kWh * {_carbon_intensity} g/kWh = {_baseline_carbon:.2f} kg CO2
  Serving carbon   = {_base_serve_kwh:.1f} kWh * {_carbon_intensity} g/kWh = {_base_serve_kwh*_carbon_intensity/1000:.2f} kg CO2
  Total baseline   = {_baseline_total:.1f} kg CO2/year

Responsible AI:
  Retrain carbon   = {_retrains_yr} * {_train_kwh:.1f} kWh * {_carbon_intensity} g/kWh = {_retrain_carbon:.1f} kg CO2
  Serving+explain  = ({_base_serve_kwh:.1f} + {_explain_serve_kwh:.1f}) kWh * {_carbon_intensity} g/kWh = {_serve_carbon:.1f} kg CO2
  Total            = {_total_carbon:.1f} kg CO2/year
  Multiplier       = {_total_carbon:.1f} / {_baseline_total:.1f} = {_ratio:.0f}x
```
*Source: carbon accounting from @sec-responsible-engineering-sustainability*
        """))

        if partD_pred.value == "60x":
            items.append(mo.callout(mo.md(
                "**Correct.** Weekly retraining (52x) plus explanation overhead produces "
                f"~{_ratio:.0f}x carbon vs baseline. The cost is real, but so is the harm of unfairness."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**The carbon multiplier is ~{_ratio:.0f}x.** "
                "Responsible AI has a real environmental cost. The question is not whether "
                "to pay it, but how to budget it against the harms of an unfair system."), kind="warn"))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════════
    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                "**1. Equal accuracy does not mean equal treatment.**\n\n"
                "When base rates differ, you cannot simultaneously equalize FPR, FNR, "
                "and PPV. This is the Chouldechova impossibility theorem."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. The fairness-accuracy Pareto frontier has a sweet spot and a cliff.**\n\n"
                "The first 10pp gap reduction costs 3-5% accuracy. "
                "Strict equality costs 10%+. Find the knee."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. Responsible AI has quantifiable costs: accuracy, latency, and carbon.**\n\n"
                "SHAP adds Nx latency (N = features). Weekly retraining adds 52x carbon. "
                "These costs are not optional under regulation."
            ), kind="info"),
            mo.md("""
## Connections

**Textbook:** @sec-responsible-engineering &mdash; impossibility theorems,
fairness-accuracy trade-offs, explainability methods, carbon accounting.

**TinyTorch:** Module 15 &mdash; implement SHAP explanations from scratch.

**Next Lab:** Lab 16 is the capstone &mdash; every invariant from 15 labs
collapses into one deployment decision.
            """),
        ])

    tabs = mo.ui.tabs({
        "Part A \u2014 The Fairness Illusion":       build_part_a(),
        "Part B \u2014 The Price of Fairness":       build_part_b(),
        "Part C \u2014 The Explainability Tax":      build_part_c(),
        "Part D \u2014 The Carbon Ledger":           build_part_d(),
        "Synthesis":                                  build_synthesis(),
    })
    tabs
    return


@app.cell(hide_code=True)
def _(COLORS, ledger, mo):
    _track = ledger.get_track()
    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span><span class="hud-value">15 &mdash; Responsible Engineering</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'NONE' else 'hud-none'}">{_track}</span>
        <span class="hud-label">STATUS</span><span class="hud-active">ACTIVE</span>
    </div>""")
    return


if __name__ == "__main__":
    app.run()
