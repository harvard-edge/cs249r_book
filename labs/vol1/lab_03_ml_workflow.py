import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")



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
    from mlsysim import Engine, Models, Hardware

    H100_TFLOPS = Hardware.Cloud.H100.compute.peak_flops.m_as("TFLOPs/s")
    H100_RAM    = Hardware.Cloud.H100.memory.capacity.m_as("GB")
    ESP32_RAM_KB = Hardware.Tiny.ESP32_S3.memory.capacity.m_as("KiB")

    RESNET50_PARAMS = Models.ResNet50.parameters.m_as("count")
    RESNET50_SIZE_MB = RESNET50_PARAMS * 2 / (1024 * 1024)

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        COLORS, LAB_CSS, apply_plotly_theme,
        go, mo, np, math,
        Engine, Models, Hardware,
        H100_TFLOPS, H100_RAM,
        ESP32_RAM_KB,
        RESNET50_PARAMS, RESNET50_SIZE_MB,
        ledger,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 03
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Constraint Tax
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Orchestrating the ML Lifecycle
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                A DR screening team spends 5 months building a model, then discovers
                it cannot deploy. Constraints discovered late cost exponentially more
                than constraints discovered early.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~51 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    Chapter 3: ML Workflow
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Exponential Cost Curve 2^(N-1)</span>
                <span class="badge badge-warn">Iteration Velocity &gt; Starting Accuracy</span>
                <span class="badge badge-fail">200x OOM Discovered at Stage 5</span>
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
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the exponential cost</strong> of
                    discovering deployment constraints late: cost = 2^(N-1) where N is lifecycle stage.</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict when iteration velocity beats starting
                    accuracy</strong> using the logarithmic improvement model.</div>
                <div style="margin-bottom: 3px;">3. <strong>Identify hidden effort allocation</strong> &mdash;
                    data activities consume 60-80% of ML project effort, not model development.</div>
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
                    Engine.solve() from Lab 01-02 &middot;
                    Physical walls from Lab 02 &middot;
                    ML lifecycle stages from the ML Workflow chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~51 min</strong><br/>
                    Part A: ~12 min &middot; Part B: ~12 min<br/>
                    Part C: ~12 min &middot; Part D: ~9 min
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
                &ldquo;A team spent 5 months building a model that cannot deploy.
                Engine.solve() could have told them in 3 milliseconds. When should
                you check constraints, and what does it cost to check late?&rdquo;
            </div>
        </div>
    </div>
    """)
    return



# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** -- Complete the following before this lab:

    - **The ML Workflow chapter** -- The 6-stage ML lifecycle and the cost of late constraint discovery.
    - **The Iteration Velocity section (Ch. 3)** -- Iteration velocity vs starting accuracy trade-offs.
    - **The Effort Distribution section (Ch. 3)** -- Effort distribution in production ML projects.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(
    COLORS, H100_TFLOPS, H100_RAM, ESP32_RAM_KB,
    RESNET50_PARAMS, RESNET50_SIZE_MB,
    Engine, Models, Hardware,
    apply_plotly_theme, go, math, mo, np,
):
    # ── Part A widgets ───────────────────────────────────────────────────
    partA_prediction = mo.ui.radio(
        options={
            "A) During architecture selection (Stage 2, cost 2x)": "stage2",
            "B) During training (Stage 3, cost 4x)":               "stage3",
            "C) During evaluation (Stage 4, cost 8x)":             "stage4",
            "D) Doesn't matter -- cost is similar at all stages":   "same",
        },
        label="A team trained a 95%-accurate model (100 MB FP16) for deployment on "
              "an ESP32 (512 KB SRAM). When is the constraint cheapest to address?",
    )
    return (partA_prediction,)

@app.cell(hide_code=True)
def _(mo, partA_prediction):
    mo.stop(partA_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partA_stage = mo.ui.slider(
        start=1, stop=6, value=5, step=1,
        label="Discovery stage (1=Problem Definition, 6=Monitoring)",
    )

    # ── Part B widgets ───────────────────────────────────────────────────
    partB_prediction = mo.ui.radio(
        options={
            "A) Team A -- 5% head start is insurmountable": "team_a",
            "B) Team A -- but barely (within 1%)":           "team_a_barely",
            "C) Team B -- faster iteration wins":            "team_b",
            "D) They converge to the same accuracy":         "converge",
        },
        label="Team A: 95% start, 1-week cycles. Team B: 90% start, 1-hour cycles. "
              "After 26 weeks, which team has higher accuracy?",
    )
    return (partB_prediction,)

@app.cell(hide_code=True)
def _(mo, partB_prediction):
    mo.stop(partB_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partB_cycle_a = mo.ui.slider(
        start=1, stop=336, value=168, step=1,
        label="Team A cycle time (hours)",
    )
    partB_cycle_b = mo.ui.slider(
        start=1, stop=168, value=1, step=1,
        label="Team B cycle time (hours)",
    )

    # ── Part C widgets ───────────────────────────────────────────────────
    partC_prediction = mo.ui.radio(
        options={
            "A) 50-60% (it is the core task)":    "50",
            "B) 30-40% (significant but not dominant)": "30",
            "C) 10-20% (surprisingly small)":     "10",
            "D) <5% (negligible)":                "5",
        },
        label="What fraction of total engineering effort goes to model development "
              "(architecture, training, hyperparameters)?",
    )
    return (partC_prediction,)

@app.cell(hide_code=True)
def _(mo, partC_prediction):
    mo.stop(partC_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partC_sliders = {
        "data_collect": mo.ui.slider(start=0, stop=20, value=3, step=1,
                                      label="Data Collection (person-months)"),
        "data_label": mo.ui.slider(start=0, stop=20, value=2, step=1,
                                    label="Data Labeling/Validation"),
        "model_dev": mo.ui.slider(start=0, stop=20, value=4, step=1,
                                   label="Model Development"),
        "deploy": mo.ui.slider(start=0, stop=20, value=1, step=1,
                                label="Deployment/Infrastructure"),
        "monitor": mo.ui.slider(start=0, stop=20, value=0, step=1,
                                 label="Monitoring/Maintenance"),
    }

    # ── Part D widgets ───────────────────────────────────────────────────
    partD_prediction = mo.ui.radio(
        options={
            "A) 0 -- it was validated during pilot":    "zero",
            "B) 1 -- one round of fixes":                "one",
            "C) 2-3 -- a few adjustments":               "few",
            "D) 4-8 -- continuous iteration":             "many",
        },
        label="After deploying from 5 pilot clinics to 200 clinics, how many "
              "complete lifecycle iterations before the system stabilizes?",
    )
    return (partD_prediction,)

@app.cell(hide_code=True)
def _(mo, partD_prediction):
    mo.stop(partD_prediction.value is None, mo.md("**Make your prediction above to unlock this part.**"))

    partD_months = mo.ui.slider(
        start=0, stop=24, value=0, step=1,
        label="Months since production launch",
    )

    # ═════════════════════════════════════════════════════════════════════
    # PART A -- Constraint Propagation
    # ═════════════════════════════════════════════════════════════════════

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incident Report &middot; MedVision Health DR Screening Project
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;After 5 months of development, we achieved 95% accuracy on our DR
                screening model (ResNet-50 backbone, ~{RESNET50_SIZE_MB:.0f} MB FP16). Today we
                learned the rural clinic tablets are ESP32-based with {ESP32_RAM_KB:.0f} KB of SRAM.
                The model does not fit. We need to restart.&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Post-mortem report, Month 5
            </div>
        </div>
        """))

        items.append(mo.md(f"""
        ## The Exponential Cost of Late Constraint Discovery

        The cost of discovering a deployment constraint at lifecycle stage N grows as
        **2^(N-1)**. The DR screening case is concrete: a team spent 150 person-days
        building a model that Engine.solve() could have rejected in 3 milliseconds.

        ```
        Engine.solve(ResNet50, ESP32, batch_size=1, precision="fp16")
        -> INFEASIBLE: {RESNET50_SIZE_MB:.0f} MB model vs {ESP32_RAM_KB:.0f} KB SRAM (~{RESNET50_SIZE_MB*1024/ESP32_RAM_KB:.0f}x over)
        -> This diagnosis took 0.003 seconds.
        -> The team spent 150 days before discovering it.
        ```
        """))

        items.append(partA_prediction)

        if partA_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the cost curve."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partA_stage)

        _stage = partA_stage.value
        _stages = {
            1: "Problem Definition",
            2: "Data Engineering",
            3: "Model Development",
            4: "Evaluation",
            5: "Deployment",
            6: "Monitoring",
        }
        _stage_name = _stages[_stage]
        _cost_multiplier = 2 ** (_stage - 1)
        _base_cost_days = 5  # person-days for early discovery
        _actual_cost = _base_cost_days * _cost_multiplier

        _artifacts_to_rebuild = {
            1: ["Requirements document"],
            2: ["Requirements", "Data pipeline"],
            3: ["Requirements", "Data pipeline", "Model architecture", "Training runs"],
            4: ["Requirements", "Data pipeline", "Model", "Training", "Evaluation suite"],
            5: ["Requirements", "Data pipeline", "Model", "Training", "Evaluation", "Deployment config"],
            6: ["Everything + production rollback"],
        }
        _artifacts = _artifacts_to_rebuild[_stage]

        # Cost curve chart
        _stage_nums = [1, 2, 3, 4, 5, 6]
        _costs = [2 ** (s - 1) * _base_cost_days for s in _stage_nums]
        _colors_bar = [COLORS["GreenLine"] if s < _stage else
                       COLORS["RedLine"] if s == _stage else
                       COLORS["Grey"] for s in _stage_nums]

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[_stages[s] for s in _stage_nums], y=_costs,
            marker_color=_colors_bar, opacity=0.85,
            text=[f"{c:.0f} days" for c in _costs],
            textposition="outside",
        ))
        _fig.update_layout(
            height=320,
            yaxis=dict(title="Cost (person-days)", gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
            margin=dict(l=50, r=20, t=30, b=80),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### Exponential Cost Curve: 2^(N-1)"))
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:2px solid {COLORS['RedLine']}; border-radius:10px;
                        text-align:center; background:{COLORS['RedLL']}; flex:1;">
                <div style="color:{COLORS['RedLine']}; font-size:0.72rem; font-weight:700;">
                    Discovery Stage</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['RedLine']};">
                    Stage {_stage}: {_stage_name}</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['OrangeLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Cost Multiplier</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_cost_multiplier}x</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Person-Days</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_actual_cost:.0f}</div>
            </div>
        </div>
        """))

        # Artifacts list
        _artifacts_html = "".join([f"<li style='margin-bottom:4px;'>{a}</li>" for a in _artifacts])
        items.append(mo.Html(f"""
        <div style="background:{COLORS['Surface2']}; border:1px solid {COLORS['Border']};
                    border-radius:10px; padding:16px 20px; margin:8px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; margin-bottom:8px;">
                Artifacts that must be rebuilt at Stage {_stage}</div>
            <ul style="margin:0; padding-left:20px; font-size:0.88rem;
                       color:{COLORS['TextSec']}; line-height:1.65;">
                {_artifacts_html}
            </ul>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="background:{COLORS['BlueLL']}; border-left:4px solid {COLORS['BlueLine']};
                    border-radius:0 10px 10px 0; padding:14px 20px; margin:12px 0;">
            <div style="font-size:0.85rem; color:{COLORS['Text']}; line-height:1.6;">
                <strong>Engine.solve() check:</strong> This diagnosis took <strong>0.003 seconds</strong>.
                The team spent <strong>{_actual_cost:.0f} person-days</strong> before discovering it
                at Stage {_stage}. At Stage 1, it would have cost {_base_cost_days} person-days.
            </div>
        </div>
        """))

        _pred = partA_prediction.value
        if _pred == "stage2":
            items.append(mo.callout(mo.md(
                "**Correct.** Early discovery is exponentially cheaper. "
                "At Stage 2 (cost 2x), you rebuild only the requirements and data pipeline. "
                f"At Stage 5 (cost {2**4}x), you rebuild everything."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**The cost grows exponentially.** At Stage 1-2, fixing a constraint "
                "costs 5-10 person-days. At Stage 5, it costs 80+ person-days because "
                "every artifact built on top of the wrong assumption must be rebuilt."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Exponential Cost": mo.md("""
$$
\\text{Cost}(N) = C_0 \\cdot 2^{N-1}
$$

where $C_0$ is the base cost at Stage 1 and $N$ is the discovery stage.

| Stage | Name | Cost Multiplier | Person-Days |
|-------|------|-----------------|-------------|
| 1 | Problem Definition | 1x | 5 |
| 2 | Data Engineering | 2x | 10 |
| 3 | Model Development | 4x | 20 |
| 4 | Evaluation | 8x | 40 |
| 5 | Deployment | 16x | 80 |
| 6 | Monitoring | 32x | 160 |
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # PART B -- Iteration Velocity
    # ═════════════════════════════════════════════════════════════════════

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Team Decision &middot; MedVision Health
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We are choosing between two approaches. Team A proposes a large ensemble
                (95% starting accuracy, 1-week training cycles). Team B proposes a lightweight
                edge model (90% start, 1-hour cycles). We have 26 weeks. Which team wins?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Project Planning Meeting
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Iteration Velocity Dominates Starting Accuracy

        ```
        accuracy(t) = accuracy_0 + alpha * log(1 + experiments(t))
        ```

        More experiments explore more of the design space. A team that runs 100
        experiments in 26 weeks finds better configurations than a team running 26.
        """))

        items.append(partB_prediction)

        if partB_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the iteration race."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.hstack([partB_cycle_a, partB_cycle_b], justify="start", gap="2rem"))

        _cycle_a = partB_cycle_a.value  # hours
        _cycle_b = partB_cycle_b.value
        _weeks = 26
        _total_hours = _weeks * 168  # hours in 26 weeks
        _acc_a_start = 95.0
        _acc_b_start = 90.0
        _alpha = 2.0

        _exps_a = _total_hours / _cycle_a
        _exps_b = _total_hours / _cycle_b

        _acc_a_final = min(99.5, _acc_a_start + _alpha * math.log(1 + _exps_a))
        _acc_b_final = min(99.5, _acc_b_start + _alpha * math.log(1 + _exps_b))

        # Timeline
        _weeks_range = np.linspace(0, 26, 200)
        _acc_a_curve = [min(99.5, _acc_a_start + _alpha * math.log(1 + w * 168 / _cycle_a))
                        for w in _weeks_range]
        _acc_b_curve = [min(99.5, _acc_b_start + _alpha * math.log(1 + w * 168 / _cycle_b))
                        for w in _weeks_range]

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=_weeks_range.tolist(), y=_acc_a_curve, mode="lines",
            name=f"Team A ({_cycle_a}h cycle, {_exps_a:.0f} exps)",
            line=dict(color=COLORS["BlueLine"], width=2.5),
        ))
        _fig.add_trace(go.Scatter(
            x=_weeks_range.tolist(), y=_acc_b_curve, mode="lines",
            name=f"Team B ({_cycle_b}h cycle, {_exps_b:.0f} exps)",
            line=dict(color=COLORS["GreenLine"], width=2.5),
        ))
        _fig.update_layout(
            height=320,
            xaxis=dict(title="Weeks", gridcolor="#f1f5f9"),
            yaxis=dict(title="Accuracy (%)", range=[88, 100], gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### The Iteration Race"))
        items.append(mo.as_html(_fig))

        _winner = "Team A" if _acc_a_final > _acc_b_final else "Team B"
        _winner_color = COLORS["BlueLine"] if _winner == "Team A" else COLORS["GreenLine"]

        items.append(mo.Html(f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin:16px 0;">
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['BlueLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Team A Final</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['BlueLine']};">
                    {_acc_a_final:.1f}%</div>
                <div style="font-size:0.68rem; color:#94a3b8;">{_exps_a:.0f} experiments</div>
            </div>
            <div style="padding:14px; border:1px solid {COLORS['Border']}; border-radius:10px;
                        text-align:center; background:white; flex:1;
                        border-top:3px solid {COLORS['GreenLine']};">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Team B Final</div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    {_acc_b_final:.1f}%</div>
                <div style="font-size:0.68rem; color:#94a3b8;">{_exps_b:.0f} experiments</div>
            </div>
            <div style="padding:14px; border:2px solid {_winner_color}; border-radius:10px;
                        text-align:center; background:white; flex:1;">
                <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">Winner</div>
                <div style="font-size:1.4rem; font-weight:800; color:{_winner_color};">
                    {_winner}</div>
            </div>
        </div>
        """))

        _pred = partB_prediction.value
        if _pred == "team_b":
            items.append(mo.callout(mo.md(
                f"**Correct.** Team B runs {_exps_b:.0f} experiments vs Team A's {_exps_a:.0f}. "
                "Faster iteration explores more design space and finds better configurations."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                f"**{_winner} wins at week 26.** Team A: {_acc_a_final:.1f}% ({_exps_a:.0f} exps). "
                f"Team B: {_acc_b_final:.1f}% ({_exps_b:.0f} exps). "
                "Adjust the cycle time sliders to find the crossover point."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Logarithmic Improvement Model": mo.md("""
**Formula:**
$$
A(t) = A_0 + \\alpha \\cdot \\ln\\!\\left(1 + \\frac{t}{\\tau}\\right)
$$

**Variables:**
- **$A_0$**: starting accuracy (%)
- **$\\alpha$**: improvement rate per log-experiment (~2-3% typical)
- **$t$**: elapsed time (hours or weeks)
- **$\\tau$**: iteration cycle time (hours)
- **$t / \\tau$**: number of experiments run

The key insight: accuracy improves *logarithmically* with experiments, so the team
running 100 experiments (1-hour cycles) gains far more than the team running 26
experiments (1-week cycles), even starting 5% behind.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # PART C -- Where Does the Time Go?
    # ═════════════════════════════════════════════════════════════════════

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Resource Planning &middot; MedVision Health
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We have a 10-person team and 6 months. How should we allocate
                effort across the ML lifecycle? Most of us assumed model development
                would be the biggest block.&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Project kickoff meeting
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Data Activities Consume 60-80% of ML Project Effort

        Model development -- the phase that receives the most research attention --
        is typically only 10-20% of total effort. Data collection, cleaning, labeling,
        and validation dominate the budget.
        """))

        items.append(partC_prediction)

        if partC_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the effort allocator."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(mo.md("### Allocate Your Team's 10 Person-Months"))
        for _k, _s in partC_sliders.items():
            items.append(_s)

        _alloc = {k: s.value for k, s in partC_sliders.items()}
        _total = sum(_alloc.values())
        _model_pct = (_alloc["model_dev"] / _total * 100) if _total > 0 else 0
        _data_pct = ((_alloc["data_collect"] + _alloc["data_label"]) / _total * 100) if _total > 0 else 0

        # Industry reference (from Hidden Technical Debt, MLCommons)
        _industry = {"Data Collection": 25, "Data Labeling": 35, "Model Development": 15,
                     "Deployment": 15, "Monitoring": 10}
        _student = {
            "Data Collection": _alloc["data_collect"] / _total * 100 if _total > 0 else 0,
            "Data Labeling": _alloc["data_label"] / _total * 100 if _total > 0 else 0,
            "Model Development": _alloc["model_dev"] / _total * 100 if _total > 0 else 0,
            "Deployment": _alloc["deploy"] / _total * 100 if _total > 0 else 0,
            "Monitoring": _alloc["monitor"] / _total * 100 if _total > 0 else 0,
        }

        _phases = list(_industry.keys())
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            name="Your Allocation", x=_phases,
            y=[_student[p] for p in _phases],
            marker_color=COLORS["BlueLine"], opacity=0.85,
        ))
        _fig.add_trace(go.Bar(
            name="Industry Average", x=_phases,
            y=[_industry[p] for p in _phases],
            marker_color=COLORS["OrangeLine"], opacity=0.85,
        ))
        _fig.update_layout(
            barmode="group", height=320,
            yaxis=dict(title="% of Total Effort", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=80),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        # Consequences
        if _alloc["data_collect"] + _alloc["data_label"] < 3 and _total > 0:
            items.append(mo.callout(mo.md(
                "**Data starvation warning.** You allocated less than 30% to data activities. "
                "The team will run out of clean training data by month 2 -- "
                "three modelers will sit idle waiting for labeled images."
            ), kind="danger"))
        if _alloc["deploy"] == 0 and _total > 0:
            items.append(mo.callout(mo.md(
                "**Deployment crisis.** Zero allocation to deployment means the model "
                "achieves 95% accuracy in development, then fails the ESP32 feasibility "
                "check (see Part A). This is the DR clinic disaster repeating."
            ), kind="danger"))

        _pred = partC_prediction.value
        if _pred == "10":
            items.append(mo.callout(mo.md(
                "**Correct.** Model development is typically 10-20% of total effort. "
                "Data activities dominate at 60-80%."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Model development is only 10-20% of total effort.** "
                "Data collection + labeling + validation consume 60-80%. "
                "Compare your allocation to the industry average above."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Effort Distribution": mo.md("""
**Formula:**
$$
E_{\\text{total}} = E_{\\text{data}} + E_{\\text{model}} + E_{\\text{deploy}} + E_{\\text{monitor}}
$$

**Industry averages (from Google, Meta production ML studies):**

| Activity | Share of $E_{\\text{total}}$ |
|----------|---------------------------|
| Data collection + labeling | 25-35% |
| Data validation + cleaning | 20-30% |
| Model development | 10-20% |
| Deployment + infrastructure | 10-15% |
| Monitoring + maintenance | 10-15% |

The critical implication: under-investing in data activities creates idle engineers
downstream, while under-investing in deployment recreates the DR clinic disaster.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # PART D -- Feedback Loops
    # ═════════════════════════════════════════════════════════════════════

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Production Update &middot; MedVision Health (6 months post-launch)
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We scaled from 5 pilot clinics to 200 clinics. New camera equipment,
                new demographics, new failure modes. The pilot validation is not holding.
                How many iteration cycles should we budget?&rdquo;
            </div>
            <div style="font-size:0.78rem; color:#475569; margin-top:8px; font-weight:600;">
                &mdash; Operations team, Month 6
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Lifecycle Never Ends

        Unlike traditional software, ML systems require continuous feedback loops.
        Scaling from pilot to production reveals problems invisible at small scale.
        Each feedback event triggers re-entry into an earlier lifecycle stage.
        """))

        items.append(partD_prediction)

        if partD_prediction.value is None:
            items.append(mo.callout(
                mo.md("Select your prediction to unlock the feedback timeline."),
                kind="warn",
            ))
            return mo.vstack(items)

        items.append(partD_months)

        _month = partD_months.value

        # Feedback events
        _events = [
            (3, "Camera model changed at clinic #47", "accuracy drops 8%", "Data Engineering"),
            (6, "New demographic (elderly with cataracts)", "40% higher error rate", "Data Collection"),
            (9, "Regulatory audit requires balanced demographics", "model retraining mandated", "Problem Definition"),
            (12, "Seasonal lighting changes affect image quality", "5% accuracy drop", "Data Validation"),
            (15, "New clinic network has different image format", "pipeline failure", "Data Engineering"),
            (18, "Model drift detected across all clinics", "2% gradual degradation", "Model Development"),
            (21, "Competitor publishes better architecture", "accuracy gap identified", "Model Development"),
        ]

        _triggered = [e for e in _events if e[0] <= _month]
        _cycle_count = len(_triggered)

        # Timeline visualization
        _fig = go.Figure()

        # Base timeline
        _fig.add_trace(go.Scatter(
            x=[0, 24], y=[0, 0], mode="lines",
            line=dict(color=COLORS["Border"], width=2),
            showlegend=False,
        ))

        for _i, (_m, _desc, _impact, _stage) in enumerate(_events):
            _color = COLORS["RedLine"] if _m <= _month else COLORS["Grey"]
            _fig.add_trace(go.Scatter(
                x=[_m], y=[0], mode="markers+text",
                marker=dict(color=_color, size=14, symbol="diamond"),
                text=[f"M{_m}"], textposition="top center",
                name=f"M{_m}: {_desc[:30]}...",
                textfont=dict(size=9),
            ))

        _fig.add_vline(x=_month, line_dash="dash", line_color=COLORS["BlueLine"], line_width=1.5)

        _fig.update_layout(
            height=200,
            xaxis=dict(title="Months Since Launch", range=[-1, 25], gridcolor="#f1f5f9"),
            yaxis=dict(visible=False, range=[-1, 1]),
            legend=dict(orientation="v", x=1.05, y=1, font=dict(size=9)),
            margin=dict(l=30, r=200, t=20, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.md("### Feedback Event Timeline"))
        items.append(mo.as_html(_fig))

        # Events table
        if _triggered:
            _rows = ""
            for _m, _desc, _impact, _stage in _triggered:
                _rows += f"""
                <tr>
                    <td style="padding:8px; font-weight:700;">Month {_m}</td>
                    <td style="padding:8px;">{_desc}</td>
                    <td style="padding:8px; color:{COLORS['RedLine']};">{_impact}</td>
                    <td style="padding:8px; color:{COLORS['OrangeLine']}; font-weight:600;">
                        -> {_stage}</td>
                </tr>"""

            items.append(mo.Html(f"""
            <div style="overflow-x:auto; margin:12px 0;">
                <table style="width:100%; border-collapse:collapse; font-size:0.85rem;">
                    <thead>
                        <tr style="background:{COLORS['Surface2']}; border-bottom:2px solid {COLORS['Border']};">
                            <th style="padding:8px; text-align:left;">When</th>
                            <th style="padding:8px; text-align:left;">Event</th>
                            <th style="padding:8px; text-align:left;">Impact</th>
                            <th style="padding:8px; text-align:left;">Re-entry Stage</th>
                        </tr>
                    </thead>
                    <tbody>{_rows}</tbody>
                </table>
            </div>
            """))

        items.append(mo.Html(f"""
        <div style="padding:16px; border:2px solid {COLORS['OrangeLine']}; border-radius:10px;
                    text-align:center; background:white; margin:16px 0;">
            <div style="color:#94a3b8; font-size:0.72rem; font-weight:600;">
                Complete Lifecycle Iterations</div>
            <div style="font-size:2rem; font-weight:800; color:{COLORS['OrangeLine']};">
                {_cycle_count}</div>
            <div style="font-size:0.72rem; color:#94a3b8;">by month {_month}</div>
        </div>
        """))

        _pred = partD_prediction.value
        if _pred == "many":
            items.append(mo.callout(mo.md(
                f"**Correct.** By month 24, there are {len(_events)} feedback events, "
                "each triggering re-entry into an earlier stage. "
                "Production ML is a continuous loop, not a one-time pipeline."
            ), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Scaling reveals problems invisible at pilot scale.** "
                "Rare subgroups, equipment drift, regulatory changes, and distribution "
                "shift each trigger a complete lifecycle iteration. Budget 4-8 cycles."
            ), kind="warn"))

        items.append(mo.accordion({
            "Math Peek: Feedback Event Rate": mo.md("""
**Formula:**
$$
N_{\\text{events}}(t) \\approx \\lambda \\cdot t + \\beta \\cdot \\ln(S)
$$

**Variables:**
- **$N_{\\text{events}}$**: cumulative feedback events requiring lifecycle re-entry
- **$\\lambda$**: base event rate (~0.3 events/month from distribution shift, regulatory changes)
- **$t$**: months since production launch
- **$\\beta$**: scale sensitivity coefficient (~1.5)
- **$S$**: number of deployment sites (e.g., clinics)

Scaling from 5 to 200 sites increases $\\ln(S)$ from 1.6 to 5.3, exposing rare subgroups
and equipment variations invisible at pilot scale. Budget 4-8 complete lifecycle iterations
in the first 24 months of production deployment.
""")
        }))

        return mo.vstack(items)

    # ═════════════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════

    def build_synthesis():
        return mo.vstack([
            mo.Html(f"""
            <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                        border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                    Key Takeaways
                </div>
                <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                    <div style="margin-bottom: 10px;">
                        <strong>1. Constraint discovery cost grows as 2^(N-1).</strong>
                        A deployment constraint found at Stage 5 costs 16x more than at Stage 1.
                        Engine.solve() can check feasibility in 3 ms -- run it before writing
                        training code.
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>2. Iteration velocity beats starting accuracy.</strong>
                        1-hour cycles produce ~100 experiments in 26 weeks vs 26 for 1-week cycles.
                        The team that explores more of the design space wins.
                    </div>
                    <div>
                        <strong>3. Data activities dominate effort at 60-80%.</strong>
                        Model development is only 10-20%. Under-investing in data causes
                        team idle time; under-investing in deployment causes the DR disaster.
                    </div>
                </div>
            </div>
            """),

            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 8px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        What's Next
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        <strong>Lab 04: The Data Gravity Trap</strong> -- Data is the heaviest
                        object in your system. You will discover that GPUs starve when
                        storage is slow, and that moving 50 TB costs more than the compute.
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
                        <strong>Read:</strong> the ML Workflow chapter for the full lifecycle model,
                        iteration velocity analysis, and effort distribution data.
                        <br/><strong>Build:</strong> TinyTorch Module 03 -- implement an experiment tracker with iteration velocity metrics.
                    </div>
                </div>
            </div>
            """),
        ])

    # ── COMPOSE TABS ─────────────────────────────────────────────────────
    tabs = mo.ui.tabs({
        "Part A -- Constraint Propagation":     build_part_a(),
        "Part B -- Iteration Velocity":         build_part_b(),
        "Part C -- Where Does the Time Go?":    build_part_c(),
        "Part D -- Feedback Loops":             build_part_d(),
        "Synthesis":                             build_synthesis(),
    })
    tabs
    return



# ===========================================================================
# ZONE D: LEDGER HUD
# ===========================================================================

@app.cell(hide_code=True)
def _(COLORS, ledger, mo, partA_prediction, partB_prediction, partC_prediction, partD_prediction):
    _track = ledger._state.track or "not set"
    if partA_prediction.value is not None and partD_prediction.value is not None:
        ledger.save(chapter=3, design={
            "chapter": "v1_03",
            "constraint_discovery_prediction": partA_prediction.value,
            "iteration_velocity_prediction": partB_prediction.value,
            "effort_distribution_prediction": partC_prediction.value,
            "feedback_loops_prediction": partD_prediction.value,
            "constraint_discovery_correct": partA_prediction.value == "stage2",
            "iteration_velocity_correct": partB_prediction.value == "team_b",
            "effort_model_dev_correct": partC_prediction.value == "10",
            "feedback_loops_correct": partD_prediction.value == "many",
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">03 &middot; The Constraint Tax</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">TRACK</span>
        <span class="{'hud-active' if _track != 'not set' else 'hud-none'}">{_track}</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">CHAPTER&nbsp;3</span>
        <span class="hud-value">ML Workflow</span>
        <span style="color:{COLORS['Border']};">|</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">active</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
