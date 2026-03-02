import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 15: THE JEVONS RECKONING
#
# Chapter: Sustainable AI (@sec-sustainable-ai)
# Core Invariant: Jevons Paradox — efficiency improvements increase total
#                 consumption by enabling more usage:
#                 E_total = (E_per_unit / efficiency_gain) × demand × elasticity
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Efficiency Trap (12-15 min)
#             Stakeholder: Sustainability Director. V100→H100 upgrade was 5.7×
#             more efficient per FLOP. Energy consumption went UP 40%. Why?
#             Answer: Jevons Paradox — demand grew 8×, overwhelming efficiency.
#
#   Act II — Carbon-Aware Scheduling (20-25 min)
#             Stakeholder: Green Cloud Engineering Lead. Design a carbon-optimal
#             schedule for a 1,024-H100 training run across three grid regions.
#             Failure state: carbon footprint > 20 tonnes CO₂.
#
# Deployment Contexts:
#   Coal Region:      US-East (coal-heavy, 400 gCO₂/kWh)
#   Renewable Region: Iceland (geothermal, 10 gCO₂/kWh)
#
# Design Ledger: saves chapter="v2_15" with Jevons multiplier, region
#                allocation, carbon footprint, and carbon reduction %.
# ─────────────────────────────────────────────────────────────────────────────


# ─── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─
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

    ledger = DesignLedger()

    # ── Hardware constants (sources documented inline) ──────────────────────
    H100_TDP_W          = 700    # H100 SXM5 TDP; source: NVIDIA H100 SXM5 spec sheet
    H100_TFLOPS_FP16    = 1979   # H100 SXM5 FP16 Tensor Core; source: NVIDIA spec
    V100_TDP_W          = 250    # V100 SXM2 TDP; source: NVIDIA V100 spec sheet
    V100_TFLOPS_FP16    = 125    # V100 SXM2 FP16; source: NVIDIA spec sheet

    # ── Carbon intensity constants (sources documented inline) ──────────────
    CARBON_US_EAST_GCO2 = 400    # gCO₂/kWh US-East (coal-heavy); source: EPA eGRID 2022
    CARBON_US_WEST_GCO2 = 150    # gCO₂/kWh US-West (mixed grid); source: EPA eGRID 2022
    CARBON_ICELAND_GCO2 = 10     # gCO₂/kWh Iceland (geothermal); source: Statista 2023
    CARBON_PRICE_USD_T  = 50     # $/tonne CO₂; source: EU ETS average 2023 approximate

    # ── Training cluster constants ──────────────────────────────────────────
    TRAINING_GPUS       = 1024   # H100 cluster size for Act II scenario
    TRAINING_DAYS       = 7      # training duration (days) for Act II scenario

    return (
        COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        ledger,
        H100_TDP_W, H100_TFLOPS_FP16, V100_TDP_W, V100_TFLOPS_FP16,
        CARBON_US_EAST_GCO2, CARBON_US_WEST_GCO2, CARBON_ICELAND_GCO2,
        CARBON_PRICE_USD_T, TRAINING_GPUS, TRAINING_DAYS,
        mo,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, LAB_CSS, mo):
    _c_coal     = COLORS["RedLine"]
    _c_renew    = COLORS["GreenLine"]
    _c_surf0    = COLORS["Surface0"]
    _c_surf1    = COLORS["Surface1"]

    _header = mo.Html(f"""
    {LAB_CSS}
    <div style="background: linear-gradient(135deg, {_c_surf0} 0%, {_c_surf1} 100%);
                border-radius: 16px; padding: 32px 40px; margin-bottom: 8px;
                border: 1px solid #2d3748;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;
                    flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8;
                            text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 8px;">
                    Vol 2 &middot; Lab 15 &middot; Sustainable AI
                </div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #f1f5f9;
                            line-height: 1.15; margin-bottom: 10px;">
                    The Jevons Reckoning
                </div>
                <div style="font-size: 0.95rem; color: #94a3b8; max-width: 620px; line-height: 1.6;">
                    Your H100 upgrade was 5.7&times; more energy efficient per FLOP.
                    Your energy bill went up 40%. Jevons Paradox explains why efficiency
                    alone never reduces total consumption. Then: design a carbon-optimal
                    training schedule before the grid runs out of renewables.
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px; flex-shrink: 0;">
                <span class="badge badge-info">Jevons: E_total = (E/unit / gain) &times; demand</span>
                <span class="badge badge-info">Carbon = kWh &times; gCO&#8322;/kWh</span>
                <span class="badge badge-warn">35-40 minutes &middot; 2 Acts</span>
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
            <div style="background: rgba(203,32,45,0.12); border: 1px solid rgba(203,32,45,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_coal}; font-weight: 700;">Coal Region</span>
                <span style="color: #94a3b8;">
                    &nbsp;&mdash; US-East grid &middot; 400 gCO&#8322;/kWh &middot; coal-heavy
                </span>
            </div>
            <div style="background: rgba(0,143,69,0.12); border: 1px solid rgba(0,143,69,0.35);
                        border-radius: 8px; padding: 10px 16px; font-size: 0.82rem;">
                <span style="color: {_c_renew}; font-weight: 700;">Renewable Region</span>
                <span style="color: #94a3b8;">
                    &nbsp;&mdash; Iceland grid &middot; 10 gCO&#8322;/kWh &middot; geothermal
                </span>
            </div>
        </div>
    </div>
    """)
    _header
    return


# ─── CELL 2: RECOMMENDED READING ───────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-sustainable-ai-sustainable-ai-engineering-discipline-6d39** — The sustainability
      paradox: why 350,000&times; compute growth from 2012 to 2019 outpaced hardware efficiency
      gains, establishing the Jevons Paradox as the governing dynamic
    - **@sec-sustainable-ai-scale-environmental-impact-ac9a** — Carbon cost of training;
      lifecycle emissions (training 60&ndash;80%, inference 15&ndash;25%, manufacturing 5&ndash;15%)
    - **@sec-sustainable-ai** — Carbon intensity, geographic and temporal variation; how carbon
      intensity differences across grids enable 50&ndash;80% emission reductions via scheduling
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    context_toggle = mo.ui.radio(
        options={
            "Coal Region (US-East, 400 gCO\u2082/kWh)": "coal",
            "Renewable Region (Iceland, 10 gCO\u2082/kWh)": "renewable",
        },
        value="Coal Region (US-East, 400 gCO\u2082/kWh)",
        label="Deployment context for this session:",
        inline=True,
    )
    mo.hstack([
        mo.Html(f"""
        <div style="font-size:0.78rem; font-weight:700; color:{COLORS['TextMuted']};
                    text-transform:uppercase; letter-spacing:0.08em;
                    margin-right:8px; padding-top:2px;">
            Active Context:
        </div>
        """),
        context_toggle,
    ], justify="start", gap=0)
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ACT I — THE EFFICIENCY TRAP
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT I: SECTION HEADER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Act I — The Efficiency Trap
    *Calibration &middot; 12-15 minutes*
    """)
    return


# ─── ACT I: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["RedLine"]
    _bg    = COLORS["RedL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Sustainability Director
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "Six months ago we upgraded our entire inference fleet from V100s to H100s.
            The H100 is 5.7&times; more energy efficient per FLOP &mdash; that number came
            straight from NVIDIA. Our CFO built a forecast: energy bill drops 75%.
            I just got the Q3 numbers. Our energy consumption <strong>increased by 40%</strong>.
            The board is asking questions I don't have answers to.
            Can you explain what happened?"
        </div>
    </div>
    """)
    return


# ─── ACT I: CONCEPT FRAMING ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The CFO's forecast was not wrong about the hardware. The V100 delivers 0.5 TFLOPS per
    Watt; the H100 delivers 2.83 TFLOPS per Watt &mdash; a 5.7&times; efficiency improvement,
    exactly as advertised. The forecast failed because it held demand constant.

    This failure has a name: **Jevons Paradox**, formalized by economist William Stanley
    Jevons in 1865 when he observed that improvements in coal engine efficiency led to
    *increased* total coal consumption across England. The same dynamic governs AI systems:
    cheaper inference per request enables more API calls, more product features, and more
    downstream use cases. Demand grows faster than efficiency improves.

    The formal model from @sec-sustainable-ai:

    > **E_total = (E_per_unit / efficiency_gain) &times; demand(efficiency_gain^elasticity)**

    where **elasticity** measures how much demand grows per unit of efficiency improvement.
    When elasticity &gt; 1, the system is Jevons-dominated: total energy always increases
    with efficiency gains.

    Before running the numbers, commit to your prediction.
    """)
    return


# ─── ACT I: PREDICTION LOCK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) Hardware defect \u2014 H100s should not increase energy consumption; NVIDIA's efficiency numbers are wrong": "A",
            "B) The CFO was wrong \u2014 hardware efficiency improvements never reduce total energy consumption": "B",
            "C) Jevons Paradox: the 5.7\u00d7 efficiency gain lowered cost per query, demand grew faster than efficiency improved, net energy increased": "C",
            "D) The comparison is unfair \u2014 the H100s are doing more work, so higher energy consumption is expected and not a paradox": "D",
        },
        label="The H100 upgrade was 5.7\u00d7 more efficient per FLOP. Energy consumption rose 40%. What explains this?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(act1_pred, mo):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act I instruments."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT I: JEVONS PARADOX EXPLORER ────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Jevons Paradox Explorer")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_efficiency_slider = mo.ui.slider(
        start=1.0, stop=20.0, value=5.7, step=0.1,
        label="Efficiency improvement (TFLOPS/W ratio, new vs old)",
        show_value=True,
    )
    act1_elasticity_slider = mo.ui.slider(
        start=0.0, stop=5.0, value=1.4, step=0.1,
        label="Demand elasticity (demand growth exponent per unit efficiency gain)",
        show_value=True,
    )
    act1_utilization_slider = mo.ui.slider(
        start=10, stop=100, value=80, step=5,
        label="Initial GPU utilization (%)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([act1_efficiency_slider, act1_elasticity_slider], justify="start", gap=4),
        act1_utilization_slider,
    ])
    return (act1_efficiency_slider, act1_elasticity_slider, act1_utilization_slider)


@app.cell(hide_code=True)
def _(
    COLORS,
    H100_TDP_W,
    H100_TFLOPS_FP16,
    V100_TDP_W,
    V100_TFLOPS_FP16,
    act1_efficiency_slider,
    act1_elasticity_slider,
    act1_utilization_slider,
    apply_plotly_theme,
    go,
    np,
    mo,
):
    # ── Pull slider values ─────────────────────────────────────────────────
    _gain        = act1_efficiency_slider.value      # efficiency improvement ratio
    _elasticity  = act1_elasticity_slider.value      # demand elasticity exponent
    _util_pct    = act1_utilization_slider.value     # initial utilization %

    # ── Hardware physics ───────────────────────────────────────────────────
    # V100: 0.5 TFLOPS/W; H100: 2.83 TFLOPS/W
    # source: @sec-sustainable-ai and NVIDIA spec sheets
    _v100_flops_per_watt = V100_TFLOPS_FP16 / V100_TDP_W   # = 0.50 TFLOPS/W
    _h100_flops_per_watt = H100_TFLOPS_FP16 / H100_TDP_W   # = 2.83 TFLOPS/W
    _actual_hw_gain = _h100_flops_per_watt / _v100_flops_per_watt  # = 5.65×

    # ── Jevons Paradox model ───────────────────────────────────────────────
    # E_per_unit_new = E_per_unit_old / efficiency_gain
    # demand_new = demand_old × gain^elasticity
    # E_total_new = E_per_unit_new × demand_new
    #             = (1 / gain) × gain^elasticity
    #             = gain^(elasticity - 1)
    # Jevons multiplier = E_total_new / E_total_old = gain^(elasticity - 1)
    # source: Jevons Paradox model from @sec-sustainable-ai
    _jevons_multiplier = _gain ** (_elasticity - 1.0)

    # ── Naive savings (ignoring demand growth) ─────────────────────────────
    _naive_saving_pct = (1.0 - 1.0 / _gain) * 100.0  # % energy saved if demand fixed

    # ── Actual energy change ──────────────────────────────────────────────
    # Positive = increase, negative = decrease
    _energy_change_pct = (_jevons_multiplier - 1.0) * 100.0

    # ── Demand growth implied ─────────────────────────────────────────────
    _demand_multiplier = _gain ** _elasticity

    # ── Color coding ───────────────────────────────────────────────────────
    if _jevons_multiplier > 1.0:
        _result_color = COLORS["RedLine"]
        _result_label = f"Total energy INCREASED by {_energy_change_pct:.1f}%"
        _jevons_status = "Jevons-dominated"
    elif _jevons_multiplier > 0.8:
        _result_color = COLORS["OrangeLine"]
        _result_label = f"Total energy changed by {_energy_change_pct:+.1f}%"
        _jevons_status = "Marginal reduction"
    else:
        _result_color = COLORS["GreenLine"]
        _result_label = f"Total energy DECREASED by {abs(_energy_change_pct):.1f}%"
        _jevons_status = "Efficiency-dominated"

    # ── Scenario matching the stakeholder (5.7× gain, demand grew 8×) ─────
    # With gain=5.7, elasticity such that demand grows 8×:
    # 5.7^elasticity = 8 → elasticity = log(8)/log(5.7) ≈ 1.164
    # Jevons multiplier = 5.7^(1.164-1) = 5.7^0.164 ≈ 1.40 → 40% increase

    # ── Curve: Jevons multiplier vs efficiency gain ─────────────────────────
    _gain_range = np.linspace(1.0, 20.0, 300)
    _mult_high  = _gain_range ** (_elasticity - 1.0)          # current elasticity
    _mult_low   = _gain_range ** (0.5 - 1.0)                  # elasticity = 0.5 (elastic < 1)
    _mult_unit  = _gain_range ** (1.0 - 1.0)                  # elasticity = 1.0 (breakeven)
    _mult_high2 = _gain_range ** (2.0 - 1.0)                  # elasticity = 2.0 (strongly Jevons)

    _fig = go.Figure()

    # Reference lines
    _fig.add_hline(y=1.0, line_dash="dot", line_color="#94a3b8", opacity=0.5)
    _fig.add_annotation(x=18, y=1.05, text="No change", showarrow=False,
                        font=dict(size=10, color="#94a3b8"))

    # Elasticity curves
    _fig.add_trace(go.Scatter(
        x=_gain_range, y=_mult_low,
        mode="lines", name="Elasticity = 0.5 (energy always falls)",
        line=dict(color=COLORS["GreenLine"], width=1.5, dash="dot"),
    ))
    _fig.add_trace(go.Scatter(
        x=_gain_range, y=_mult_unit,
        mode="lines", name="Elasticity = 1.0 (breakeven)",
        line=dict(color=COLORS["OrangeLine"], width=1.5, dash="dash"),
    ))
    _fig.add_trace(go.Scatter(
        x=_gain_range, y=_mult_high2,
        mode="lines", name="Elasticity = 2.0 (strongly Jevons-dominated)",
        line=dict(color=COLORS["RedLine"], width=1.5, dash="dot"),
    ))

    # Active curve (current elasticity)
    _fig.add_trace(go.Scatter(
        x=_gain_range, y=_mult_high,
        mode="lines", name=f"Your elasticity = {_elasticity:.1f}",
        line=dict(color=COLORS["BlueLine"], width=2.5),
    ))

    # Current operating point
    _fig.add_trace(go.Scatter(
        x=[_gain], y=[_jevons_multiplier],
        mode="markers",
        name=f"Your scenario ({_gain:.1f}\u00d7 gain)",
        marker=dict(color=_result_color, size=14, symbol="diamond",
                    line=dict(color="white", width=2)),
    ))

    # H100 vs V100 scenario point
    _h100_mult = _actual_hw_gain ** (_elasticity - 1.0)
    _fig.add_trace(go.Scatter(
        x=[_actual_hw_gain], y=[_h100_mult],
        mode="markers",
        name=f"H100 vs V100 ({_actual_hw_gain:.1f}\u00d7 actual HW gain)",
        marker=dict(color=COLORS["OrangeLine"], size=10, symbol="circle",
                    line=dict(color="white", width=2)),
    ))

    _fig.update_layout(
        height=360,
        xaxis=dict(title="Efficiency improvement (TFLOPS/W ratio)", range=[1, 20]),
        yaxis=dict(title="Jevons multiplier (total energy change ratio)", range=[0, 6]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=10)),
        margin=dict(t=60, b=50, l=55, r=20),
    )
    apply_plotly_theme(_fig)

    # ── Physics formula display ────────────────────────────────────────────
    _formula_block = mo.Html(f"""
    <div class="lab-card" style="margin: 8px 0; font-family: 'SF Mono', monospace; font-size: 0.88rem;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
            Jevons Paradox Physics
        </div>
        <div style="line-height: 2.2; color: {COLORS['Text']};">
            <div>Jevons multiplier
                = gain<sup>(&epsilon; &minus; 1)</sup>
                = {_gain:.2f}<sup>({_elasticity:.2f} &minus; 1)</sup>
                = {_gain:.2f}<sup>{_elasticity - 1:.2f}</sup>
                = <strong style="color:{_result_color};">{_jevons_multiplier:.3f}</strong>
            </div>
            <div style="color:{COLORS['TextSec']}; font-size:0.83rem; margin-top:4px;">
                where gain = {_gain:.2f}&times; &nbsp;|&nbsp;
                &epsilon; (elasticity) = {_elasticity:.2f}
            </div>
        </div>
        <div style="margin-top:12px; padding-top:12px; border-top:1px solid {COLORS['Border']};
                    display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; font-size:0.85rem;">
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Naive Savings
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['GreenLine']};">
                    &minus;{_naive_saving_pct:.1f}%
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.78rem;">
                    if demand stayed flat
                </div>
            </div>
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Demand Growth
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:{COLORS['OrangeLine']};">
                    {_demand_multiplier:.1f}&times;
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.78rem;">
                    gain<sup>&epsilon;</sup> = {_gain:.1f}<sup>{_elasticity:.1f}</sup>
                </div>
            </div>
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Net Energy Change
                </div>
                <div style="font-size:1.4rem; font-weight:800; color:{_result_color};">
                    {_energy_change_pct:+.1f}%
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.78rem;">
                    {_jevons_status}
                </div>
            </div>
        </div>
    </div>
    """)

    mo.vstack([_formula_block, mo.ui.plotly(_fig)])
    return (
        _jevons_multiplier,
        _gain,
        _elasticity,
        _demand_multiplier,
        _naive_saving_pct,
        _energy_change_pct,
        _result_color,
        _jevons_status,
        _actual_hw_gain,
        _h100_flops_per_watt,
        _v100_flops_per_watt,
    )


# ─── ACT I: PREDICTION-VS-REALITY OVERLAY ──────────────────────────────────────
@app.cell(hide_code=True)
def _(
    COLORS,
    _actual_hw_gain,
    _elasticity,
    _energy_change_pct,
    _jevons_multiplier,
    _naive_saving_pct,
    act1_pred,
    mo,
):
    _correct = act1_pred.value == "C"

    # Stakeholder scenario: 5.7× gain, demand grew 8×
    # Jevons multiplier ≈ 1.40 → energy up 40%
    _scenario_gain = _actual_hw_gain
    _scenario_demand = 8.0
    _scenario_multiplier = _scenario_demand / _scenario_gain  # 8/5.65 ≈ 1.415

    if _correct:
        _reveal = mo.callout(mo.md(
            f"**Correct.** "
            f"This is Jevons Paradox in action. The H100 is {_actual_hw_gain:.1f}\u00d7 more "
            f"efficient per FLOP, which lowered the cost per inference request enough that "
            f"product teams shipped 8\u00d7 more API calls. The math: efficiency removes "
            f"{_naive_saving_pct:.1f}% of energy per unit, but demand grew {_scenario_demand:.0f}\u00d7, "
            f"so total energy = (1/{_scenario_gain:.1f}) \u00d7 {_scenario_demand:.0f} = "
            f"**{_scenario_multiplier:.2f}\u00d7 \u2014 a {(_scenario_multiplier-1)*100:.0f}% increase**, "
            f"exactly matching the sustainability director's observation. "
            f"The Jevons multiplier with your current elasticity = {_elasticity:.1f} is "
            f"{_jevons_multiplier:.2f}, producing a {_energy_change_pct:+.1f}% energy change."
        ), kind="success")
    elif act1_pred.value == "A":
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"NVIDIA's efficiency numbers are correct: H100 is {_actual_hw_gain:.1f}\u00d7 more "
            f"efficient per FLOP. The hardware performed exactly as specified. "
            f"The problem is not hardware defect but demand response: "
            f"cheaper inference per request enabled 8\u00d7 more inference calls. "
            f"Total energy = (efficiency savings)\u207b\u00b9 \u00d7 demand growth = "
            f"(1/{_actual_hw_gain:.1f}) \u00d7 8 = {8/_actual_hw_gain:.2f}\u00d7 &mdash; a "
            f"{(8/_actual_hw_gain-1)*100:.0f}% increase. This is Jevons Paradox."
        ), kind="warn")
    elif act1_pred.value == "B":
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"Efficiency improvements *can* reduce total energy when demand elasticity < 1 "
            f"(i.e., demand grows slower than efficiency improves). "
            f"The Jevons Paradox is not universal \u2014 it is a function of elasticity. "
            f"In this scenario, elasticity is approximately {_elasticity:.1f}: demand grew "
            f"8\u00d7 when efficiency improved {_actual_hw_gain:.1f}\u00d7. "
            f"For energy to fall, you would need demand growth < {_actual_hw_gain:.1f}\u00d7 "
            f"(elasticity < 1.0). The correct insight: efficiency alone is insufficient "
            f"when demand is highly elastic."
        ), kind="warn")
    else:  # D
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"This is exactly the reasoning Jevons identified as the trap. "
            f"'They are doing more work' is true, but it does not make the energy increase "
            f"acceptable or expected from an efficiency standpoint. "
            f"The CFO forecast was: same work at {_actual_hw_gain:.1f}\u00d7 lower energy = "
            f"{_naive_saving_pct:.1f}% savings. The actual outcome: {_scenario_demand:.0f}\u00d7 "
            f"more work at {_actual_hw_gain:.1f}\u00d7 lower energy per unit = "
            f"{(_scenario_demand/_actual_hw_gain-1)*100:.0f}% more total energy. "
            f"The paradox is precisely that doing more work is what efficiency enables. "
            f"Jevons Paradox names this dynamic so engineers can account for it."
        ), kind="warn")

    _reveal
    return


# ─── ACT I: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Reflection \u2014 Policy Implications")
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) Hardware efficiency improvements are counterproductive for sustainability \u2014 avoid them": "A",
            "B) Efficiency alone is insufficient \u2014 carbon reduction requires demand constraints or carbon pricing": "B",
            "C) Jevons Paradox only applies to consumer hardware, not data centers with managed workloads": "C",
            "D) Measuring energy per request (not total energy) resolves the paradox for sustainability purposes": "D",
        },
        label="What is the correct policy implication of Jevons Paradox for AI sustainability programs?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(act1_reflect, mo):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(
            mo.md("Select an answer to see the explanation."),
            kind="warn",
        ),
    )
    mo.md("")
    return


@app.cell(hide_code=True)
def _(act1_reflect, mo):
    if act1_reflect.value == "B":
        _r = mo.callout(mo.md(
            "**Correct.** "
            "Efficiency improvements are necessary but not sufficient for sustainability. "
            "When demand is elastic (grows with efficiency), total energy rises unless "
            "a second mechanism constrains demand: carbon pricing that makes each unit "
            "of energy consumption expensive, hard capacity caps on inference volume, "
            "or regulatory carbon budgets. The Sustainable AI engineering discipline "
            "from @sec-sustainable-ai formalizes this as requiring *both* efficiency "
            "optimization *and* demand management. Efficiency alone is a prerequisite, "
            "not a solution."
        ), kind="success")
    elif act1_reflect.value == "A":
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "Avoiding efficiency improvements is counterproductive: it raises cost per "
            "inference without reducing total consumption when demand is elastic. "
            "Efficiency improvements are still necessary \u2014 they reduce the energy floor. "
            "The issue is that efficiency alone is insufficient when elasticity > 1. "
            "The correct response is to pair efficiency gains with demand constraints "
            "or carbon pricing, not to abandon efficiency as a lever."
        ), kind="warn")
    elif act1_reflect.value == "C":
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "Jevons Paradox applies to data centers precisely because capacity is managed. "
            "When H100s replace V100s in a data center, the cost per inference falls, "
            "which product managers and platform teams respond to by shipping more "
            "AI-powered features. The paradox operates through business decisions, "
            "not just consumer behavior. The sustainability director's experience "
            "above is a direct data-center-scale example of Jevons Paradox in action."
        ), kind="warn")
    else:  # D
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "Energy per request is a valuable efficiency metric for individual optimization, "
            "but it does not resolve the systemic paradox. A data center that improves "
            "energy per request by 5.7\u00d7 and then processes 8\u00d7 more requests "
            "will report better energy-per-request numbers while consuming 40% more total "
            "energy. Sustainability programs that measure only efficiency ratios create "
            "exactly the incentive structure that produces Jevons-dominated outcomes. "
            "Total carbon footprint \u2014 not efficiency ratio \u2014 is the target metric."
        ), kind="warn")
    _r
    return


# ─── ACT I: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "\U0001f4d0 The Jevons Paradox Formal Model": mo.md("""
        **The original observation (Jevons, 1865):**
        Improvements in steam engine coal efficiency in Victorian England led to *increased*
        total coal consumption, not decreased. Cheaper energy per unit enabled broader
        adoption and higher-volume use cases that overwhelmed the per-unit savings.

        **Formal model (from @sec-sustainable-ai):**
        ```
        E_total = E_per_unit / efficiency_gain \u00d7 demand(efficiency_gain^\u03b5)
                = (1 / gain) \u00d7 gain^\u03b5
                = gain^(\u03b5 - 1)
        ```
        where **\u03b5** (epsilon) is the demand elasticity: how many times demand grows
        per unit of efficiency improvement.

        **The three regimes:**
        ```
        \u03b5 < 1 \u2192 gain^(\u03b5-1) < 1 \u2192 Total energy DECREASES (efficiency wins)
        \u03b5 = 1 \u2192 gain^0 = 1       \u2192 Total energy UNCHANGED (breakeven)
        \u03b5 > 1 \u2192 gain^(\u03b5-1) > 1 \u2192 Total energy INCREASES (Jevons-dominated)
        ```

        **Hardware efficiency (H100 vs V100):**
        ```
        V100: 125 TFLOPS / 250 W = 0.50 TFLOPS/W
        H100: 1979 TFLOPS / 700 W = 2.83 TFLOPS/W
        Actual gain = 2.83 / 0.50 = 5.65\u00d7
        ```

        **The stakeholder scenario:**
        ```
        gain = 5.65, demand grew 8\u00d7
        \u03b5 = log(8) / log(5.65) \u2248 1.17
        Jevons multiplier = 5.65^(1.17-1) = 5.65^0.17 \u2248 1.40 (+40% energy)
        ```

        **Carbon intensity formula:**
        ```
        Carbon (gCO\u2082) = Energy (kWh) \u00d7 Grid Intensity (gCO\u2082/kWh)
        ```
        Grid intensity is the *location and time* dependent variable that
        carbon-aware scheduling exploits. See Act II.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ACT II — CARBON-AWARE SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── ACT II: SECTION HEADER ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Act II \u2014 Carbon-Aware Scheduling
    *Design Challenge &middot; 20-25 minutes*
    """)
    return


# ─── ACT II: STAKEHOLDER MESSAGE ────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(COLORS, mo):
    _color = COLORS["GreenLine"]
    _bg    = COLORS["GreenL"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; Green Cloud Engineering Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have a 1,024-H100 training job running for 7 days. We can route compute
            across three regions: US-East (coal-heavy, 400 gCO&#8322;/kWh), US-West
            (mixed grid, 150 gCO&#8322;/kWh), and Iceland (geothermal, 10 gCO&#8322;/kWh).
            Off-peak hours (overnight, weekend) in each region drop carbon intensity by
            roughly 30% due to renewable surplus. The board has set a 20-tonne CO&#8322;
            carbon target for all training runs. Design the optimal carbon schedule."
        </div>
    </div>
    """)
    return


# ─── ACT II: SCENARIO FRAMING ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(TRAINING_DAYS, TRAINING_GPUS, H100_TDP_W, mo):
    # Compute baseline energy for the scenario display
    _total_kwh = TRAINING_GPUS * (H100_TDP_W / 1000) * TRAINING_DAYS * 24
    mo.md(f"""
    The training run consumes:

    > **{TRAINING_GPUS:,} GPUs &times; {H100_TDP_W} W &times; {TRAINING_DAYS} days &times; 24 hours
    = {_total_kwh:,.0f} kWh**

    At US-East baseline (400 gCO&#8322;/kWh):
    > **{_total_kwh:,.0f} kWh &times; 400 gCO&#8322;/kWh = {_total_kwh * 400 / 1e6:,.1f} tonnes CO&#8322;** &mdash;
    more than double the 20-tonne target.

    Carbon-aware scheduling moves compute to lower-carbon regions and off-peak windows.
    Before designing the schedule, commit to your prediction.
    """)
    return


# ─── ACT II: PREDICTION LOCK ────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Your Prediction")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Run in US-East \u2014 compute resources are most available and price is lowest": "A",
            "B) Run entirely in Iceland at any time \u2014 lowest carbon intensity always wins": "B",
            "C) Shift the majority of compute to Iceland and US-West, prioritize off-peak hours \u2014 carbon-temporal optimization": "C",
            "D) Carbon accounting doesn\u2019t matter for training runs, only for inference at scale": "D",
        },
        label="How do you design the carbon-optimal schedule for 1,024 H100s over 7 days?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(act2_pred, mo):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction above to unlock the Act II instruments."),
            kind="warn",
        ),
    )
    mo.md("")
    return


# ─── ACT II: CARBON SCHEDULER INSTRUMENTS ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Carbon-Aware Scheduler")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_us_east_pct = mo.ui.slider(
        start=0, stop=100, value=20, step=5,
        label="US-East allocation (%, coal-heavy: 400 gCO\u2082/kWh)",
        show_value=True,
    )
    act2_us_west_pct = mo.ui.slider(
        start=0, stop=100, value=30, step=5,
        label="US-West allocation (%, mixed: 150 gCO\u2082/kWh)",
        show_value=True,
    )
    act2_iceland_pct = mo.ui.slider(
        start=0, stop=100, value=50, step=5,
        label="Iceland allocation (%, geothermal: 10 gCO\u2082/kWh)",
        show_value=True,
    )
    act2_offpeak_pct = mo.ui.slider(
        start=0, stop=100, value=50, step=10,
        label="Off-peak scheduling (% of compute in off-peak window, -30% carbon intensity)",
        show_value=True,
    )
    act2_duration_slider = mo.ui.slider(
        start=3, stop=14, value=7, step=1,
        label="Training duration (days)",
        show_value=True,
    )
    mo.vstack([
        mo.hstack([act2_us_east_pct, act2_us_west_pct], justify="start", gap=4),
        mo.hstack([act2_iceland_pct, act2_offpeak_pct], justify="start", gap=4),
        act2_duration_slider,
    ])
    return (
        act2_us_east_pct,
        act2_us_west_pct,
        act2_iceland_pct,
        act2_offpeak_pct,
        act2_duration_slider,
    )


@app.cell(hide_code=True)
def _(
    CARBON_ICELAND_GCO2,
    CARBON_PRICE_USD_T,
    CARBON_US_EAST_GCO2,
    CARBON_US_WEST_GCO2,
    COLORS,
    H100_TDP_W,
    TRAINING_GPUS,
    act2_duration_slider,
    act2_iceland_pct,
    act2_offpeak_pct,
    act2_us_east_pct,
    act2_us_west_pct,
    apply_plotly_theme,
    go,
    mo,
):
    # ── Pull slider values ─────────────────────────────────────────────────
    _east_pct    = act2_us_east_pct.value        # % compute in US-East
    _west_pct    = act2_us_west_pct.value        # % compute in US-West
    _ice_pct     = act2_iceland_pct.value        # % compute in Iceland
    _offpeak_pct = act2_offpeak_pct.value        # % compute in off-peak window
    _days        = act2_duration_slider.value    # training days

    # ── Total allocation check ─────────────────────────────────────────────
    _total_pct = _east_pct + _west_pct + _ice_pct
    _allocation_valid = abs(_total_pct - 100) <= 1  # within 1% tolerance

    # ── Total energy (kWh) ─────────────────────────────────────────────────
    # E_total = GPUs × TDP (W) × days × 24 hours
    # source: hardware constants from @sec-sustainable-ai
    _total_kwh = TRAINING_GPUS * (H100_TDP_W / 1000) * _days * 24

    # ── Effective carbon intensity per region with off-peak discount ───────
    # Off-peak periods have 30% lower carbon intensity (renewable surplus)
    # source: temporal carbon-aware scheduling model from @sec-sustainable-ai
    _offpeak_discount = 0.30
    _offpeak_frac     = _offpeak_pct / 100.0
    _peak_frac        = 1.0 - _offpeak_frac

    # Weighted carbon intensity per region:
    # effective_ci = ci_peak × peak_frac + ci_peak × (1 - offpeak_discount) × offpeak_frac
    #              = ci_peak × (1 - offpeak_discount × offpeak_frac)
    _ci_east_eff = CARBON_US_EAST_GCO2 * (1.0 - _offpeak_discount * _offpeak_frac)
    _ci_west_eff = CARBON_US_WEST_GCO2 * (1.0 - _offpeak_discount * _offpeak_frac)
    _ci_ice_eff  = CARBON_ICELAND_GCO2 * (1.0 - _offpeak_discount * _offpeak_frac)

    # ── Carbon per region (kg CO₂) ─────────────────────────────────────────
    # Carbon_region = total_kWh × allocation_pct × effective_ci_g / 1000 (g to kg)
    _kwh_east = _total_kwh * (_east_pct / 100.0)
    _kwh_west = _total_kwh * (_west_pct / 100.0)
    _kwh_ice  = _total_kwh * (_ice_pct / 100.0)

    _co2_east_kg = _kwh_east * _ci_east_eff / 1000.0   # g to kg
    _co2_west_kg = _kwh_west * _ci_west_eff / 1000.0
    _co2_ice_kg  = _kwh_ice  * _ci_ice_eff  / 1000.0

    # ── Total carbon ───────────────────────────────────────────────────────
    _co2_total_kg    = _co2_east_kg + _co2_west_kg + _co2_ice_kg
    _co2_total_t     = _co2_total_kg / 1000.0           # kg to tonnes

    # ── Baseline (all US-East, no off-peak) ───────────────────────────────
    _baseline_kwh    = TRAINING_GPUS * (H100_TDP_W / 1000) * 7 * 24   # 7-day reference
    _baseline_co2_kg = _baseline_kwh * CARBON_US_EAST_GCO2 / 1000.0
    _baseline_co2_t  = _baseline_co2_kg / 1000.0

    # ── Carbon reduction % vs baseline ────────────────────────────────────
    _co2_reduction_pct = (1.0 - _co2_total_kg / _baseline_co2_kg) * 100.0 if _allocation_valid else 0.0

    # ── Carbon cost (USD) ─────────────────────────────────────────────────
    _carbon_cost_usd = _co2_total_t * CARBON_PRICE_USD_T

    # ── Equivalent metrics ─────────────────────────────────────────────────
    # 1 flight (NY-London round trip) ≈ 1 tonne CO₂
    # source: ICAO Carbon Emissions Calculator, used in @sec-sustainable-ai
    _flights_equiv = _co2_total_t
    # 1 car year ≈ 4.6 tonnes CO₂
    # source: EPA average US passenger vehicle, @sec-sustainable-ai
    _car_years_equiv = _co2_total_t / 4.6

    # ── Carbon target constraint ───────────────────────────────────────────
    _target_co2_t = 20.0    # board-set 20-tonne CO₂ target from stakeholder brief
    _target_met   = _co2_total_t <= _target_co2_t and _allocation_valid

    # ── Color coding ───────────────────────────────────────────────────────
    if not _allocation_valid:
        _co2_color = COLORS["OrangeLine"]
    elif _co2_total_t <= _target_co2_t:
        _co2_color = COLORS["GreenLine"]
    elif _co2_total_t <= _target_co2_t * 1.5:
        _co2_color = COLORS["OrangeLine"]
    else:
        _co2_color = COLORS["RedLine"]

    # ── Allocation warning ─────────────────────────────────────────────────
    if not _allocation_valid:
        _alloc_warn = mo.callout(mo.md(
            f"**Region allocation must sum to 100%.** "
            f"Currently: {_total_pct:.0f}%. "
            f"Adjust US-East ({_east_pct}%), US-West ({_west_pct}%), and "
            f"Iceland ({_ice_pct}%) sliders until they total 100%."
        ), kind="warn")
    else:
        _alloc_warn = mo.md("")

    # ── Failure state: carbon target exceeded ─────────────────────────────
    if _allocation_valid and _co2_total_t > _target_co2_t:
        _constraint_banner = mo.callout(mo.md(
            f"**Carbon target exceeded: {_co2_total_t:.1f} tonnes CO\u2082 > "
            f"{_target_co2_t:.0f}-tonne target.** "
            f"You are {_co2_total_t - _target_co2_t:.1f} tonnes over budget. "
            f"Shift more compute to Iceland (10 gCO\u2082/kWh) and increase "
            f"off-peak scheduling to reduce effective grid intensity. "
            f"Currently {_ice_pct}% in Iceland; try 60-70% to hit the target."
        ), kind="danger")
    elif _allocation_valid and _co2_total_t <= _target_co2_t:
        _constraint_banner = mo.callout(mo.md(
            f"**Carbon target met: {_co2_total_t:.1f} tonnes CO\u2082 &le; "
            f"{_target_co2_t:.0f}-tonne target.** "
            f"Carbon reduction vs all-US-East baseline: "
            f"**{_co2_reduction_pct:.1f}%**. "
            f"Carbon cost at ${CARBON_PRICE_USD_T}/tonne: "
            f"**${_carbon_cost_usd:,.0f}**."
        ), kind="success")
    else:
        _constraint_banner = mo.md("")

    # ── Physics formula display ────────────────────────────────────────────
    _formula_block = mo.Html(f"""
    <div class="lab-card" style="margin: 8px 0; font-family: 'SF Mono', monospace; font-size: 0.88rem;">
        <div style="color: {COLORS['TextMuted']}; font-size: 0.72rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">
            Carbon-Aware Scheduling Physics
        </div>
        <div style="line-height: 2.0; color: {COLORS['Text']}; font-size: 0.85rem;">
            <div>Total energy = {TRAINING_GPUS:,} GPUs &times; {H100_TDP_W} W &times; {_days} days &times; 24 hr
                = <strong>{_total_kwh:,.0f} kWh</strong>
            </div>
            <div>Effective CI_east = {CARBON_US_EAST_GCO2} &times; (1 &minus; 0.3 &times; {_offpeak_pct/100:.2f})
                = <strong>{_ci_east_eff:.0f} gCO&#8322;/kWh</strong>
            </div>
            <div>Carbon = ({_east_pct}% &times; {_ci_east_eff:.0f}) + ({_west_pct}% &times; {_ci_west_eff:.0f})
                + ({_ice_pct}% &times; {_ci_ice_eff:.1f}) = <strong style="color:{_co2_color};">{_co2_total_t:.2f} t CO&#8322;</strong>
            </div>
        </div>
        <div style="margin-top:12px; padding-top:12px; border-top:1px solid {COLORS['Border']};
                    display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:10px; font-size:0.82rem;">
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.70rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Total CO&#8322;
                </div>
                <div style="font-size:1.5rem; font-weight:800; color:{_co2_color};">
                    {_co2_total_t:.1f}t
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.75rem;">
                    target: {_target_co2_t:.0f}t
                </div>
            </div>
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.70rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    vs Baseline
                </div>
                <div style="font-size:1.5rem; font-weight:800;
                            color:{'#008F45' if _co2_reduction_pct > 0 else '#CB202D'};">
                    {_co2_reduction_pct:+.0f}%
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.75rem;">
                    baseline: {_baseline_co2_t:.1f}t
                </div>
            </div>
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.70rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Carbon Cost
                </div>
                <div style="font-size:1.5rem; font-weight:800; color:{COLORS['Text']};">
                    ${_carbon_cost_usd:,.0f}
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.75rem;">
                    @${CARBON_PRICE_USD_T}/tonne
                </div>
            </div>
            <div>
                <div style="color:{COLORS['TextMuted']}; font-size:0.70rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Equivalent
                </div>
                <div style="font-size:1.1rem; font-weight:800; color:{COLORS['Text']};">
                    {_flights_equiv:.1f} flights
                </div>
                <div style="color:{COLORS['TextSec']}; font-size:0.75rem;">
                    {_car_years_equiv:.2f} car-years
                </div>
            </div>
        </div>
    </div>
    """)

    # ── Stacked bar chart: carbon by region ───────────────────────────────
    _fig = go.Figure()

    _fig.add_trace(go.Bar(
        name=f"US-East ({_east_pct}%, {CARBON_US_EAST_GCO2} gCO\u2082/kWh)",
        x=["Carbon Breakdown"],
        y=[_co2_east_kg / 1000.0],
        marker_color=COLORS["RedLine"],
        width=0.5,
    ))
    _fig.add_trace(go.Bar(
        name=f"US-West ({_west_pct}%, {CARBON_US_WEST_GCO2} gCO\u2082/kWh)",
        x=["Carbon Breakdown"],
        y=[_co2_west_kg / 1000.0],
        marker_color=COLORS["OrangeLine"],
        width=0.5,
    ))
    _fig.add_trace(go.Bar(
        name=f"Iceland ({_ice_pct}%, {CARBON_ICELAND_GCO2} gCO\u2082/kWh)",
        x=["Carbon Breakdown"],
        y=[_co2_ice_kg / 1000.0],
        marker_color=COLORS["GreenLine"],
        width=0.5,
    ))

    # Target line
    _fig.add_hline(
        y=_target_co2_t,
        line_dash="dash",
        line_color=COLORS["BlueLine"],
        line_width=2,
        annotation_text=f"Target: {_target_co2_t:.0f}t CO\u2082",
        annotation_position="right",
        annotation_font=dict(color=COLORS["BlueLine"], size=11),
    )

    _fig.update_layout(
        barmode="stack",
        height=340,
        yaxis=dict(title="Carbon footprint (tonnes CO\u2082)"),
        xaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=10)),
        margin=dict(t=60, b=30, l=60, r=80),
    )
    apply_plotly_theme(_fig)

    mo.vstack([
        _alloc_warn,
        _constraint_banner,
        _formula_block,
        mo.ui.plotly(_fig),
    ])
    return (
        _co2_total_t,
        _co2_total_kg,
        _co2_east_kg,
        _co2_west_kg,
        _co2_ice_kg,
        _co2_reduction_pct,
        _total_kwh,
        _baseline_co2_t,
        _target_met,
        _total_pct,
        _allocation_valid,
        _flights_equiv,
        _car_years_equiv,
        _carbon_cost_usd,
    )


# ─── ACT II: PREDICTION REVEAL ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(
    CARBON_US_EAST_GCO2,
    TRAINING_GPUS,
    H100_TDP_W,
    _baseline_co2_t,
    _co2_total_t,
    _co2_reduction_pct,
    act2_pred,
    mo,
):
    _correct = act2_pred.value == "C"

    # Reference: optimal schedule (70% Iceland, 20% US-West, 10% US-East, 60% off-peak)
    # Carbon ≈ 7-9 tonnes CO₂, ~85% reduction vs baseline
    _baseline_kwh_ref = TRAINING_GPUS * (H100_TDP_W / 1000) * 7 * 24
    _baseline_co2_ref = _baseline_kwh_ref * CARBON_US_EAST_GCO2 / 1e6  # tonnes

    if _correct:
        _reveal = mo.callout(mo.md(
            f"**Correct.** "
            f"Carbon-temporal optimization combines *where* (low-carbon regions) with "
            f"*when* (renewable surplus periods). The all-US-East baseline is "
            f"{_baseline_co2_t:.1f} tonnes CO\u2082. By shifting the majority of compute "
            f"to Iceland (10 gCO\u2082/kWh) and using off-peak windows for renewable "
            f"surplus, the carbon footprint falls to the {_co2_total_t:.1f}-tonne range "
            f"&mdash; a **{abs(_co2_reduction_pct):.0f}% reduction** &mdash; with zero "
            f"performance cost. The scheduler makes a purely logistical decision: "
            f"the same compute runs, just in a different place and time."
        ), kind="success")
    elif act2_pred.value == "A":
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"US-East resource availability does not offset its carbon cost. "
            f"The all-US-East scenario produces {_baseline_co2_t:.1f} tonnes CO\u2082 "
            f"&mdash; more than {_baseline_co2_t / 20.0:.0f}\u00d7 the 20-tonne target. "
            f"Price per compute-hour and carbon footprint are separate optimization "
            f"objectives. Carbon-aware scheduling achieves both by exploiting the "
            f"40\u00d7 carbon intensity difference between Iceland and US-East."
        ), kind="warn")
    elif act2_pred.value == "B":
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"Iceland's 10 gCO\u2082/kWh is optimal on carbon intensity, but Icelandic "
            f"geothermal capacity is finite. A real scheduler must respect capacity "
            f"constraints: Iceland cannot absorb an arbitrary fraction of global AI "
            f"training demand. The correct approach is *carbon-temporal optimization*: "
            f"maximize Iceland allocation up to its capacity, then fill the remainder "
            f"with US-West during off-peak renewable-surplus windows. This achieves "
            f"85-90% of the pure-Iceland reduction while respecting physical capacity."
        ), kind="warn")
    else:  # D
        _reveal = mo.callout(mo.md(
            f"**Not quite.** "
            f"Training has massive and concentrated carbon impact. The baseline scenario "
            f"({TRAINING_GPUS:,} H100s, 7 days, US-East) produces {_baseline_co2_t:.1f} "
            f"tonnes CO\u2082 &mdash; equivalent to {_baseline_co2_t:.0f} transatlantic "
            f"flights. Inference at scale accumulates comparable carbon through volume, "
            f"but individual training runs have an outsized per-event footprint. "
            f"From @sec-sustainable-ai: training GPT-3 consumed 1,287 MWh and produced "
            f"approximately 552 tonnes CO\u2082 at average US grid intensity. "
            f"Carbon accounting applies to both training and inference."
        ), kind="warn")

    _reveal
    return


# ─── ACT II: REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Reflection \u2014 Why Carbon-Temporal Scheduling Works")
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) Renewable energy is always available in some region \u2014 scheduling just finds it": "A",
            "B) Solar and wind generation creates predictable daily surplus periods; carbon-aware workloads schedule during these windows without any performance cost": "B",
            "C) Carbon-temporal scheduling requires special hardware with renewable-sensing capability": "C",
            "D) Off-peak compute is always 50% cheaper, making the economics sufficient justification": "D",
        },
        label="Why is carbon-temporal scheduling (shifting workloads to periods of renewable surplus) increasingly effective for large training runs?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(act2_reflect, mo):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(
            mo.md("Select an answer to see the explanation."),
            kind="warn",
        ),
    )
    mo.md("")
    return


@app.cell(hide_code=True)
def _(act2_reflect, mo):
    if act2_reflect.value == "B":
        _r = mo.callout(mo.md(
            "**Correct.** "
            "Solar peaks midday; wind peaks overnight and in winter. These patterns are "
            "highly predictable from weather forecasting and historical grid data. "
            "A 7-day training job has significant schedule flexibility: it does not matter "
            "whether a given GPU runs between 2 AM and 4 AM versus 2 PM and 4 PM. "
            "Carbon-temporal scheduling exploits this *temporal flexibility* to shift "
            "compute into renewable surplus windows, achieving 20-30% carbon reductions "
            "through scheduling alone, with zero algorithm or hardware changes. "
            "This is the zero-cost optimization from @sec-sustainable-ai."
        ), kind="success")
    elif act2_reflect.value == "A":
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "This describes *location-based* carbon optimization (always running in "
            "Iceland), not *temporal* optimization. Temporal scheduling exploits the "
            "time-varying carbon intensity of a single grid, not the geographic "
            "variation across grids. A US-West region at 2 AM during high wind output "
            "may have effectively the same carbon intensity as Iceland at peak demand. "
            "Both spatial and temporal dimensions are levers; effective schedules use both."
        ), kind="warn")
    elif act2_reflect.value == "C":
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "Carbon-temporal scheduling requires only scheduling software and access to "
            "carbon intensity forecast APIs (e.g. Electricity Maps, WattTime). "
            "No special hardware is needed. The hardware runs identically; only "
            "the time and location of job submission changes. This is the key insight: "
            "it is a *software and operations* optimization with the economics and "
            "impact of a hardware change."
        ), kind="warn")
    else:  # D
        _r = mo.callout(mo.md(
            "**Not quite.** "
            "Off-peak pricing discounts vary by region and contract: some grids offer "
            "20% off-peak discounts, others offer 5%, and some industrial customers "
            "have flat-rate contracts. The economics are neither universal nor "
            "guaranteed to be 50%. The *primary* justification for carbon-temporal "
            "scheduling is the carbon intensity reduction (20-30% from temporal "
            "shifting alone), not the cost savings. The sustainability case stands "
            "independently of electricity pricing."
        ), kind="warn")
    _r
    return


# ─── ACT II: MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "\U0001f4d0 Carbon-Aware Scheduling Equations": mo.md("""
        **Carbon intensity formula:**
        ```
        Carbon (gCO\u2082) = Energy (kWh) \u00d7 Grid Intensity (gCO\u2082/kWh)
        ```

        **Training run energy:**
        ```
        E_total (kWh) = GPUs \u00d7 TDP_W / 1000 \u00d7 duration_hours
        ```
        Example: 1,024 H100s \u00d7 700 W \u00d7 168 hours = 120,422 kWh

        **Region-weighted carbon:**
        ```
        Carbon_total = \u03a3 (E_total \u00d7 pct_region \u00d7 CI_region)
        ```
        where CI_region (gCO\u2082/kWh):
        - US-East: 400 gCO\u2082/kWh (coal-heavy)
        - US-West: 150 gCO\u2082/kWh (mixed)
        - Iceland:  10 gCO\u2082/kWh (geothermal)

        **Temporal discount (off-peak renewable surplus):**
        ```
        CI_effective = CI_baseline \u00d7 (1 \u2212 0.30 \u00d7 offpeak_fraction)
        ```
        30% reduction during off-peak; source: @sec-sustainable-ai carbon intensity model

        **Social Cost of Carbon:**
        ```
        Financial Cost = Carbon_tonnes \u00d7 Carbon_Price ($/tonne)
        EU ETS approximate: $50/tonne CO\u2082 (2023)
        ```

        **Scope 1/2/3 emissions accounting:**
        - Scope 1: Direct emissions (diesel generators, on-site combustion)
        - Scope 2: Purchased electricity (grid carbon intensity \u00d7 kWh consumed)
        - Scope 3: Supply chain (chip manufacturing, hardware shipping, cooling water)

        Training run carbon is primarily **Scope 2** (grid electricity).
        Hardware manufacturing is **Scope 3** (embodied carbon): over 50% of
        edge device lifecycle carbon can come from manufacturing alone.

        **Carbon-temporal optimization algorithm:**
        ```
        for each time_window t:
            ci_t = forecast_grid_intensity(region, t)
            if ci_t < carbon_budget_remaining / kWh_remaining:
                schedule_compute(t)
        ```
        Requires: carbon intensity forecast API (e.g. Electricity Maps, WattTime)
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# LEDGER SAVE + HUD
# ═══════════════════════════════════════════════════════════════════════════════


@app.cell(hide_code=True)
def _(
    COLORS,
    _co2_reduction_pct,
    _co2_total_kg,
    _total_kwh,
    _jevons_multiplier,
    _target_met,
    _allocation_valid,
    act1_pred,
    act2_pred,
    act2_iceland_pct,
    act2_us_east_pct,
    act2_us_west_pct,
    context_toggle,
    ledger,
    mo,
):
    # ── Determine correctness ──────────────────────────────────────────────
    _act1_correct = act1_pred.value == "C"
    _act2_correct = act2_pred.value == "C"

    # ── Build region allocation dict ───────────────────────────────────────
    _region_alloc = {
        "us_east": act2_us_east_pct.value,
        "us_west": act2_us_west_pct.value,
        "iceland": act2_iceland_pct.value,
    }

    # ── Save to Design Ledger ──────────────────────────────────────────────
    ledger.save(
        chapter="v2_15",
        design={
            "context": context_toggle.value,
            "region_allocation": _region_alloc,
            "carbon_kg": round(_co2_total_kg, 1),
            "energy_kwh": round(_total_kwh, 1),
            "jevons_multiplier": round(_jevons_multiplier, 3),
            "carbon_target_met": bool(_target_met),
            "act1_prediction": str(act1_pred.value),
            "act1_correct": bool(_act1_correct),
            "act2_result": round(_co2_total_kg, 1),
            "act2_decision": str(act2_pred.value),
            "constraint_hit": bool(_allocation_valid and not _target_met),
            "carbon_reduction_pct": round(_co2_reduction_pct, 1),
        }
    )

    # ── HUD footer ─────────────────────────────────────────────────────────
    _act1_status  = "Correct" if _act1_correct  else "Incorrect"
    _act2_status  = "Correct" if _act2_correct  else "Incorrect"
    _target_label = "Met" if _target_met else "NOT Met"
    _target_hud   = "hud-active" if _target_met else "hud-none"

    mo.Html(f"""
    <div class="lab-hud">
        <div>
            <span class="hud-label">CONTEXT &nbsp;</span>
            <span class="hud-value">{context_toggle.value.upper()}</span>
        </div>
        <div>
            <span class="hud-label">ACT I &nbsp;</span>
            <span class="{'hud-active' if _act1_correct else 'hud-none'}">{_act1_status}</span>
        </div>
        <div>
            <span class="hud-label">ACT II &nbsp;</span>
            <span class="{'hud-active' if _act2_correct else 'hud-none'}">{_act2_status}</span>
        </div>
        <div>
            <span class="hud-label">JEVONS MULT &nbsp;</span>
            <span class="hud-value">{_jevons_multiplier:.3f}&times;</span>
        </div>
        <div>
            <span class="hud-label">CARBON (kg) &nbsp;</span>
            <span class="hud-value">{_co2_total_kg:,.0f}</span>
        </div>
        <div>
            <span class="hud-label">20t TARGET &nbsp;</span>
            <span class="{_target_hud}">{_target_label}</span>
        </div>
        <div>
            <span class="hud-label">REDUCTION &nbsp;</span>
            <span class="hud-value">{_co2_reduction_pct:+.1f}%</span>
        </div>
        <div>
            <span class="hud-label">CH &nbsp;</span>
            <span class="hud-value">v2_15</span>
        </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
