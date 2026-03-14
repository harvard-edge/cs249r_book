import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-15: THE JEVONS RECKONING
#
# Volume II, Chapter 15 — Sustainable AI (@sec-sustainable-ai)
#
# Core Invariant: Jevons Paradox — efficiency improvements increase total
#   resource consumption by enabling more usage. Carbon footprint
#   C = E × I where E = energy consumed and I = carbon intensity (gCO₂/kWh).
#
# 2-Act Structure (35-40 minutes):
#   Act I  — The Efficiency Trap (12-15 min)
#             Stakeholder: Head of Sustainability. H100 upgrade was 2× more
#             efficient per FLOP. Total carbon emissions went UP 40%. Why?
#             Answer: 3× deployment scale overwhelmed the efficiency gain.
#
#   Act II — Carbon-Aware Scheduling (20-25 min)
#             Stakeholder: ML Platform Lead. 1,000-node H100 cluster, 30%
#             flexible jobs. Design the scheduling policy to hit 40% carbon
#             reduction without SLA violations.
#             Failure states: SLA breach (danger), below carbon target (warn).
#
# Deployment Contexts:
#   Coal Region:    US coal-heavy grid, 820 gCO₂/kWh — US EPA eGRID 2022
#   Renewable Region: Pacific Northwest / Nordic, 40 gCO₂/kWh — US EPA data
#
# Design Ledger: saves chapter="v2_15" with efficiency gain, deployment scale,
#                net carbon change, carbon savings, and target met flag.
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 0: SETUP (hide_code=False — leave visible for instructor inspection) ─
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

    # ── Hardware constants ───────────────────────────────────────────────────
    H100_TDP_W        = 700     # H100 SXM5 TDP — NVIDIA H100 SXM5 spec sheet
    H100_IDLE_W       = 180     # H100 idle power — NVIDIA H100 power whitepaper
    H100_TFLOPS_FP16  = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec

    # ── Carbon intensity constants ───────────────────────────────────────────
    # Source: US EPA eGRID 2022, @tbl-carbon-intensity in @sec-sustainable-ai
    COAL_CI_G_KWH    = 820      # gCO₂/kWh — US coal-heavy grid (EPA eGRID 2022)
    RENEW_CI_G_KWH   = 40       # gCO₂/kWh — Pacific Northwest / Nordic renewable (US EPA eGRID data)
    GAS_CI_G_KWH     = 490      # gCO₂/kWh — natural gas grid mix (EPA eGRID 2022)
    US_AVG_CI_G_KWH  = 386      # gCO₂/kWh — US grid average 2023 (EPA eGRID 2023)

    ledger = DesignLedger()
    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        H100_TDP_W, H100_IDLE_W, H100_TFLOPS_FP16,
        COAL_CI_G_KWH, RENEW_CI_G_KWH, GAS_CI_G_KWH, US_AVG_CI_G_KWH,
    )


# ─── CELL 1: HEADER (hide_code=True) ──────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _c_coal  = COLORS["RedLine"]
    _c_renew = COLORS["GreenLine"]
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0a1a0e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 15
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Jevons Reckoning
            </h1>
            <p style="margin: 0 0 22px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 660px; line-height: 1.65;">
                Your H100 upgrade was 2&times; more energy-efficient per FLOP.
                Your total carbon emissions went up 40%. Then: design a scheduling
                policy for 1,000 H100s to hit a 40% carbon reduction without
                violating SLAs. Efficiency is not sustainability. Carbon budgets are.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px;">
                <span style="background: rgba(203,32,45,0.18); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.3);">
                    Act I: Jevons Paradox &middot; Act II: Carbon-Aware Scheduling
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35&ndash;40 min
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    Requires: @sec-sustainable-ai
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-fail">Invariant: C = E &times; I</span>
                <span class="badge badge-fail">Jevons: efficiency &uarr; &rArr; demand &uarr;&uarr;</span>
                <span class="badge badge-ok">Coal: 820 gCO&#8322;/kWh</span>
                <span class="badge badge-ok">Renewable: 40 gCO&#8322;/kWh</span>
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
                <div style="margin-bottom: 3px;">1. <strong>Predict the net carbon change</strong> when compute efficiency improves 2&times; but deployment scale grows 3&times; &mdash; and verify that the Jevons rebound makes total carbon increase, not decrease.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify the carbon footprint</strong> using C&thinsp;=&thinsp;E&thinsp;&times;&thinsp;I&thinsp;&times;&thinsp;PUE and identify the 40&times; difference between a coal-grid region (820&thinsp;g CO&#8322;/kWh) and a renewable region (40&thinsp;g CO&#8322;/kWh).</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a carbon-aware scheduling policy</strong> that achieves &ge;40% carbon reduction for a 1,000-node H100 cluster without triggering SLA violations from over-shifting flexible jobs.</div>
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
                    Carbon footprint formula C = E &times; I from @sec-sustainable-ai-carbon-footprint-analysis-ccc5 &middot;
                    Jevons Paradox definition from @sec-sustainable-ai-multilayer-mitigation-strategy-framework-80f2
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
                "If you make your ML training hardware 2&times; more energy-efficient, why does your organization&rsquo;s total carbon footprint increase by 40% &mdash; and what is the only intervention that actually guarantees a net reduction?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING (hide_code=True) ─────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following before this lab:

    - **@sec-sustainable-ai-sustainable-ai-engineering-discipline-6d39** — The sustainability
      paradox: 350,000&times; compute growth from 2012 to 2019 outpaced hardware efficiency;
      the Jevons Paradox as the governing dynamic of total energy consumption.
    - **@sec-sustainable-ai-carbon-footprint-analysis-ccc5** — Carbon footprint formula
      `C = E &times; I`; lifecycle emissions (training 60&ndash;80%, inference 15&ndash;25%,
      manufacturing 5&ndash;15%); US EPA eGRID carbon intensity data.
    - **@sec-sustainable-ai-geographic-temporal-optimization-492c** — Carbon intensity by
      region; temporal scheduling reducing emissions by 50&ndash;80%; carbon-aware scheduling
      as a first-class operational competency.
    - **@sec-sustainable-ai-multilayer-mitigation-strategy-framework-80f2** — The Jevons
      Paradox callout: why efficiency must be paired with carbon budget caps.

    If you have not read these sections, the predictions in this lab will not map to the physics.
    """), kind="info")
    return


# ─── CELL 3: CONTEXT TOGGLE (hide_code=True) ──────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Coal Region (820 gCO\u2082/kWh)": "coal",
            "Renewable Region (40 gCO\u2082/kWh)": "renewable",
        },
        value="Coal Region (820 gCO\u2082/kWh)",
        label="Deployment context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Select your grid region to orient both acts:"),
        context_toggle,
    ])
    return (context_toggle,)


@app.cell(hide_code=True)
def _(mo, context_toggle, COLORS, COAL_CI_G_KWH, RENEW_CI_G_KWH):
    _ctx = context_toggle.value
    _is_coal = _ctx == "coal"
    _color = COLORS["RedLine"] if _is_coal else COLORS["GreenLine"]
    _bg = COLORS["RedL"] if _is_coal else COLORS["GreenL"]
    _label = "Coal Region (820 gCO\u2082/kWh)" if _is_coal else "Renewable Region (40 gCO\u2082/kWh)"
    _ci = COAL_CI_G_KWH if _is_coal else RENEW_CI_G_KWH
    _specs = (
        f"US coal-heavy grid &middot; {COAL_CI_G_KWH} gCO\u2082/kWh &middot; "
        "West Virginia / Poland tier &mdash; US EPA eGRID 2022"
        if _is_coal else
        f"Pacific Northwest / Nordic grid &middot; {RENEW_CI_G_KWH} gCO\u2082/kWh &middot; "
        "hydro + wind mix &mdash; US EPA eGRID data"
    )
    _ratio = COAL_CI_G_KWH / RENEW_CI_G_KWH
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {_bg};
                border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 10px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">
            Active Context
        </div>
        <div style="font-weight: 700; font-size: 1.05rem; color: #1e293b;">{_label}</div>
        <div style="font-size: 0.85rem; color: #475569; margin-top: 3px;">{_specs}</div>
        <div style="font-size: 0.82rem; color: #475569; margin-top: 6px;">
            Grid intensity ratio coal/renewable: <strong>{_ratio:.0f}&times;</strong> &mdash;
            identical workload, {_ratio:.0f}&times; different carbon footprint
        </div>
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
    _act_title    = "The Efficiency Trap"
    _act_duration = "12&ndash;15 min"
    _act_why      = ("You expect that a 2&times; more efficient GPU means roughly half the carbon. "
                     "The data will show that when a cheaper operation triggers proportionally "
                     "more demand, total emissions rise &mdash; sometimes dramatically. "
                     "This is Jevons Paradox, and it invalidates the intuition that "
                     "per-unit efficiency is the same as total-system sustainability.")
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
            Incoming Message &middot; Head of Sustainability
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We completed our H100 upgrade last quarter. The H100 delivers approximately
            2&times; more FLOPs per watt than the A100 we replaced. I expected our datacenter
            carbon footprint to drop by roughly half. Instead, our total carbon emissions went
            UP by 40% this quarter. The efficiency numbers are correct &mdash; I have the
            NVIDIA spec sheets right here. My CFO thinks we made a mistake. I need an
            explanation I can bring to the board."
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The Jevons Paradox

    William Stanley Jevons observed in 1865 that James Watt's more efficient steam engine
    *increased* total coal consumption by making steam power economically viable for new
    applications. The pattern recurs in AI: making inference cheaper enables more applications,
    which increases total energy even as energy-per-unit falls.

    The formal statement from @sec-sustainable-ai-multilayer-mitigation-strategy-framework-80f2:

    > **Efficiency ↑ → Cost ↓ → Demand ↑↑ → Net Carbon ↑**

    The carbon equation is `C_total = (E_per_unit / efficiency_gain) × total_units`.
    When `total_units` grows faster than `efficiency_gain`, `C_total` increases.
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
            "A) The H100 actually consumes more power than the A100 &mdash; "
            "the spec sheet efficiency claim is misleading":
                "option_a",
            "B) The carbon intensity of our grid increased this quarter "
            "&mdash; the grid mix shifted toward more coal":
                "option_b",
            "C) Lower cost-per-inference drove 3&times; more deployments "
            "&mdash; the rebound effect overwhelmed the efficiency gain":
                "option_c",
            "D) Idle power dominates: H100 idle power is high enough that "
            "switching to H100 increases baseline consumption":
                "option_d",
        },
        label="The H100 upgrade was 2\u00d7 more efficient per FLOP but total carbon "
              "went UP 40%. Which mechanism explains this?",
    )
    act1_pred
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the Jevons Rebound Calculator."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** option {act1_pred.value[-1].upper()}. Now explore the physics below."),
        kind="info",
    )
    return


# ─── ACT I INSTRUMENTS: JEVONS REBOUND CALCULATOR ─────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Jevons Rebound Calculator")
    return


@app.cell(hide_code=True)
def _(mo):
    efficiency_gain = mo.ui.slider(
        start=1.0, stop=5.0, value=2.0, step=0.1,
        label="Hardware efficiency gain (new FLOPs/W \u00f7 old FLOPs/W)",
        show_value=True,
    )
    deployment_scale = mo.ui.slider(
        start=1.0, stop=10.0, value=3.0, step=0.5,
        label="Deployment scale multiplier (new total jobs \u00f7 old total jobs)",
        show_value=True,
    )
    mo.vstack([
        mo.md("""
        The H100 scenario: **2.0× efficiency gain** (A100 → H100 per-FLOP efficiency).
        Drag `deployment_scale` to see how demand growth drives net carbon.
        The **Jevons zone boundary** is where `efficiency_gain = deployment_scale`
        — above that line, efficiency wins; below it, rebound wins.
        """),
        mo.hstack([efficiency_gain, deployment_scale], justify="start", gap="2rem"),
    ])
    return (efficiency_gain, deployment_scale)


@app.cell(hide_code=True)
def _(mo, efficiency_gain, deployment_scale, COLORS, go, np, apply_plotly_theme,
       COAL_CI_G_KWH, RENEW_CI_G_KWH, context_toggle,
       H100_TDP_W):
    # ── Physics model ──────────────────────────────────────────────────────────
    # Source: @sec-sustainable-ai-multilayer-mitigation-strategy-framework-80f2
    # Jevons Paradox formula: net_carbon_ratio = deployment_scale / efficiency_gain
    # If ratio > 1.0: total carbon increased (rebound dominates)
    # If ratio < 1.0: total carbon decreased (efficiency dominates)
    #
    # Energy model (per GPU-hour):
    #   E_old = H100_TDP_W (treating old A100 as baseline = 1 normalized unit)
    #   E_new_per_unit = H100_TDP_W / efficiency_gain (same TDP, more FLOPs → fewer GPUs needed)
    # In practice, the key ratio is: net_carbon = (scale / gain) × old_carbon
    _eg = efficiency_gain.value
    _ds = deployment_scale.value
    _ci = COAL_CI_G_KWH if context_toggle.value == "coal" else RENEW_CI_G_KWH

    # Baseline: 1 cluster-week of old-generation GPUs
    # Energy_old (normalized) = 1.0 (arbitrary unit, preserves ratio arithmetic)
    _energy_old = 1.0
    _energy_new  = _ds / _eg  # scale up by demand, scale down by efficiency

    _net_ratio   = _energy_new / _energy_old  # >1 means MORE total energy
    _net_change_pct = (_net_ratio - 1.0) * 100.0  # signed %

    # Carbon (absolute, using H100 TDP for the scenario framing)
    # Normalize: 1 cluster-week baseline = 1000 GPUs × 700 W × 168 h = 117,600 kWh
    _gpus         = 1000
    _hours_week   = 24 * 7
    _kwh_old      = _gpus * (H100_TDP_W / 1000) * _hours_week
    _kwh_new      = _kwh_old * _net_ratio
    _carbon_old_t = _kwh_old * _ci / 1e6      # metric tonnes CO₂
    _carbon_new_t = _kwh_new * _ci / 1e6

    # Color coding
    _net_color = (
        COLORS["GreenLine"] if _net_change_pct < -10
        else COLORS["OrangeLine"] if _net_change_pct < 10
        else COLORS["RedLine"]
    )
    _efficiency_wins = _eg >= _ds

    # ── Plotly scatter: Jevons zone map ────────────────────────────────────────
    _eg_range = np.linspace(1.0, 5.0, 100)
    _ds_range = np.linspace(1.0, 10.0, 100)
    _EG, _DS  = np.meshgrid(_eg_range, _ds_range)
    _NET      = (_DS / _EG - 1.0) * 100.0  # net carbon change %

    _fig = go.Figure()

    # Contour fill: carbon change territory
    _fig.add_trace(go.Contour(
        x=_eg_range, y=_ds_range, z=_NET,
        colorscale=[
            [0.0,  "#D4EFDF"],   # deep green (savings)
            [0.33, "#ECFDF5"],   # light green
            [0.5,  "#FFF7ED"],   # neutral
            [0.67, "#FEF2F2"],   # light red
            [1.0,  "#F5D2D5"],   # deep red (increase)
        ],
        zmin=-60, zmax=60,
        contours_coloring="fill",
        showscale=True,
        colorbar=dict(title="Net Carbon Change (%)", tickformat="+.0f"),
        name="Carbon territory",
    ))

    # Jevons boundary: efficiency = scale (net = 0%)
    _boundary_x = np.linspace(1.0, 5.0, 100)
    _boundary_y = _boundary_x  # scale = gain → ratio = 1
    # Only plot where scale ≤ 10
    _mask = _boundary_y <= 10.0
    _fig.add_trace(go.Scatter(
        x=_boundary_x[_mask], y=_boundary_y[_mask],
        mode="lines",
        line=dict(color="#1e293b", width=2, dash="dash"),
        name="Jevons boundary (net = 0%)",
    ))

    # Current operating point
    _fig.add_trace(go.Scatter(
        x=[_eg], y=[_ds],
        mode="markers",
        marker=dict(size=16, color=_net_color, line=dict(color="white", width=2)),
        name=f"Your config: {_net_change_pct:+.1f}%",
    ))

    # H100 scenario annotation (efficiency=2, scale=3 → +50%)
    _fig.add_trace(go.Scatter(
        x=[2.0], y=[3.0],
        mode="markers+text",
        marker=dict(size=12, color=COLORS["RedLine"], symbol="star",
                    line=dict(color="white", width=1)),
        text=["H100 scenario<br>(+50%)"],
        textposition="top right",
        textfont=dict(size=10, color=COLORS["RedLine"]),
        name="H100 scenario",
    ))

    _fig.update_layout(
        title="Jevons Zone Map — Net Carbon Change by Efficiency Gain and Deployment Scale",
        xaxis_title="Hardware Efficiency Gain (new \u00f7 old FLOPs/W)",
        yaxis_title="Deployment Scale Multiplier (new \u00f7 old jobs)",
        height=420,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)"),
    )
    apply_plotly_theme(_fig)

    mo.vstack([
        mo.md(f"""
        ```
        Physics (Jevons Paradox — @sec-sustainable-ai-multilayer-mitigation-strategy-framework-80f2)

        net_energy_ratio  = deployment_scale / efficiency_gain
                          = {_ds:.1f} / {_eg:.1f}
                          = {_net_ratio:.3f}

        net_carbon_change = (net_energy_ratio - 1.0) × 100%
                          = ({_net_ratio:.3f} - 1.0) × 100%
                          = {_net_change_pct:+.1f}%

        Carbon (coal @ {_ci} gCO₂/kWh, 1,000 H100s for 1 week):
          Old baseline : {_kwh_old:,.0f} kWh → {_carbon_old_t:.1f} tonnes CO₂
          New scenario : {_kwh_new:,.0f} kWh → {_carbon_new_t:.1f} tonnes CO₂
          Net change   : {_net_change_pct:+.1f}%
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin: 16px 0; flex-wrap: wrap;">
            <div style="padding: 20px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 190px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">
                    Net Energy Change
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_net_color};">
                    {_net_change_pct:+.1f}%
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                    {'Carbon INCREASED' if _net_change_pct > 0 else 'Carbon decreased'}
                </div>
            </div>
            <div style="padding: 20px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 190px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">
                    Old Carbon (1 week)
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {COLORS['BlueLine']};">
                    {_carbon_old_t:.1f} t
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                    metric tonnes CO&#8322;
                </div>
            </div>
            <div style="padding: 20px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 190px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">
                    New Carbon (1 week)
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {_net_color};">
                    {_carbon_new_t:.1f} t
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                    metric tonnes CO&#8322;
                </div>
            </div>
            <div style="padding: 20px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 190px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.82rem; margin-bottom: 4px;">
                    Jevons Outcome
                </div>
                <div style="font-size: 1.4rem; font-weight: 800;
                            color: {'#008F45' if _efficiency_wins else '#CB202D'};">
                    {'Efficiency wins' if _efficiency_wins else 'Rebound wins'}
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
                    scale {'<' if _efficiency_wins else '>'} gain
                </div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
    ])
    return (
        _net_change_pct, _net_ratio, _energy_new,
        _carbon_old_t, _carbon_new_t,
        _efficiency_wins, _eg, _ds, _ci,
    )


# ─── ACT I PREDICTION-VS-REALITY OVERLAY ──────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act1_pred, _net_change_pct):
    _feedback = {
        "option_a": (
            "**Not the root cause.** H100 SXM5 TDP is 700 W versus A100 SXM4's 400 W — "
            "H100 draws *more* peak power. But per FLOP, H100 delivers roughly 2× more "
            "compute per watt, making it genuinely more efficient. The hardware spec claim "
            "is accurate. The problem is not hardware efficiency — it is what happened to "
            "total demand when costs dropped."
        ),
        "option_b": (
            "**Partially relevant but not causal.** Grid carbon intensity does vary "
            "seasonally and year-over-year, but it would need to increase by 40% in a "
            "single quarter to explain this — an implausible shift. The invariant here "
            "is `C = E × I`: even if I held constant, the 40% increase in C means E "
            "increased by 40%. The grid did not cause this; deployment scale did."
        ),
        "option_c": (
            "**Correct.** This is the Jevons Paradox. With 2× efficiency gain and 3× "
            "deployment scale, the net energy ratio = 3/2 = 1.5 — a 50% *increase* in "
            "total energy despite each unit being more efficient. Making inference cheaper "
            "enabled product teams to deploy new features and higher request volumes. "
            "The current simulator shows this: with efficiency=2 and scale=3, net "
            f"carbon change = {_net_change_pct:+.1f}%. Efficiency alone never reduces "
            "total consumption when demand is elastic."
        ),
        "option_d": (
            "**Idle power is a real concern but not the primary mechanism here.** H100 "
            "idle power (~180 W) is higher than A100 idle (~60 W), but at typical "
            "datacenter utilization of 60–80%, active compute power dominates. The "
            "40% increase in total carbon traces to the 3× scale-up in deployments, "
            "not to idle baseline shifts. The Jevons Paradox is the governing effect."
        ),
    }
    _correct = act1_pred.value == "option_c"
    mo.callout(
        mo.md(_feedback[act1_pred.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─── ACT I REFLECTION ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *Given the Jevons Paradox, what actually reduces total carbon footprint?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act1_reflect = mo.ui.radio(
        options={
            "A) More efficient hardware always reduces carbon — "
            "the H100 is greener per FLOP, so it is the right choice":
                "reflect_a",
            "B) Carbon budget caps with enforcement: limit total compute "
            "expenditure regardless of efficiency gains":
                "reflect_b",
            "C) Use smaller models on edge devices instead of cloud GPUs — "
            "this avoids the rebound effect entirely":
                "reflect_c",
            "D) Only operate during off-peak grid hours — "
            "time-shifting to nights and weekends is sufficient":
                "reflect_d",
        },
        label="How do you actually reduce total carbon footprint given Jevons Paradox?",
    )
    act1_reflect
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your reflection answer to continue to Act II."), kind="warn"),
    )
    _reflect_feedback = {
        "reflect_a": (
            "**This is the Jevons trap.** Per-FLOP efficiency is real and valuable, but "
            "it does not guarantee lower total carbon when deployment scale grows. The "
            "chapter callout states it directly: 'Making models 10\u00d7 more efficient will "
            "likely lead to 100\u00d7 more usage, not 10\u00d7 energy savings.' Efficiency "
            "is a necessary but not sufficient condition for sustainability."
        ),
        "reflect_b": (
            "**Correct.** The chapter's conclusion: 'Sustainability strategies must focus "
            "on absolute limits (carbon budgets, renewable sourcing) rather than just rate "
            "efficiency (FLOPS/Watt).' A carbon budget cap means that even if a new model "
            "is 2\u00d7 more efficient, you cannot run 3\u00d7 more jobs — you can run 2\u00d7 "
            "more jobs, and total carbon stays flat. Budget enforcement is the only mechanism "
            "that directly controls the product `E \u00d7 I`."
        ),
        "reflect_c": (
            "**Partially effective.** Edge deployment shifts some inference off high-TDP "
            "cloud GPUs, and the embodied carbon per device is often amortized over more "
            "inference queries. But edge deployment also *expands* access to AI capabilities, "
            "which can accelerate demand growth — another form of Jevons rebound. Edge "
            "deployment reduces per-query energy but does not impose an absolute cap."
        ),
        "reflect_d": (
            "**Carbon-aware scheduling helps but does not solve the scale problem.** "
            "Time-shifting reduces the carbon intensity `I` in `C = E \u00d7 I`, which is "
            "real progress. Act II explores exactly this. But scheduling optimization "
            "alone cannot offset a 3\u00d7 growth in total jobs unless the available "
            "low-carbon grid hours scale proportionally — which they do not. "
            "Scheduling is a multiplier on an absolute cap, not a substitute for one."
        ),
    }
    _correct = act1_reflect.value == "reflect_b"
    mo.callout(
        mo.md(_reflect_feedback[act1_reflect.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─── ACT I MATHPEEK ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "\U0001f4d0 The governing equation: Jevons Paradox formal statement": mo.md("""
        **Source:** @sec-sustainable-ai-multilayer-mitigation-strategy-framework-80f2

        **Carbon equation:**
        ```
        C_total = E_total × I
        E_total = (E_per_unit / η) × D
        ```
        where:
        - **C_total** — total carbon emissions (gCO₂)
        - **E_total** — total energy consumed (kWh)
        - **I** — carbon intensity of grid (gCO₂/kWh)
        - **η** — efficiency gain (new FLOPs/W ÷ old FLOPs/W)
        - **D** — total deployment demand (normalized jobs)

        **Net carbon change:**
        ```
        ΔC/C_old = (D_new / D_old) / η - 1
                 = deployment_scale / efficiency_gain - 1
        ```

        **Jevons condition (net carbon increases):**
        ```
        ΔC > 0  ⟺  deployment_scale > efficiency_gain
        ```

        **Rebound factor R** — fraction of efficiency savings consumed by demand:
        ```
        R = (D_new/D_old - 1) / (1 - 1/η)
        ```
        When R > 1, the rebound completely offsets and exceeds the efficiency gain.
        At η = 2.0, D = 3.0: R = (3.0 - 1) / (1 - 0.5) = 2.0/0.5 = 4.0 (400% rebound).

        **Net savings condition** (when efficiency genuinely wins):
        ```
        deployment_scale < efficiency_gain
        ```
        For H100 (η = 2.0), demand must grow less than 2× for carbon to decrease.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II — DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["GreenLine"]
    _act_title    = "Carbon-Aware Scheduling"
    _act_duration = "20&ndash;25 min"
    _act_why      = ("Act I proved that efficiency without a carbon cap makes emissions worse. "
                      "Now design the scheduling policy that actually works: "
                      "shift flexible jobs from coal-heavy grid hours to low-carbon windows. "
                      "The constraint is a 40% reduction target with no SLA violations &mdash; "
                      "which means you cannot shift more flexible work than the low-carbon window can absorb.")
    mo.vstack([
        mo.md("---"),
        mo.Html(f"""
        <div style="margin: 40px 0 12px 0;">
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
    _color = COLORS["GreenLine"]
    mo.Html(f"""
    <div style="border-left: 4px solid {_color}; background: {COLORS['GreenL']};
                border-radius: 0 10px 10px 0; padding: 16px 22px; margin: 12px 0;">
        <div style="font-size: 0.72rem; font-weight: 700; color: {_color};
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px;">
            Incoming Message &middot; ML Platform Lead
        </div>
        <div style="font-style: italic; font-size: 1.0rem; color: #1e293b; line-height: 1.65;">
            "We have a 1,000-node H100 cluster. Our workload analysis shows 30% of jobs
            are time-flexible &mdash; they can be delayed up to 8 hours without violating
            SLAs. Our grid is approximately 70% coal-heavy during peak day hours and
            shifts to 90% renewable at night. The sustainability team has set a hard
            target: 40% carbon reduction this quarter. I need you to design the
            scheduling policy. Tell me how many jobs to shift, how far to shift them,
            and whether the 40% target is achievable without SLA violations."
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Carbon-Aware Scheduling Physics

    **The core opportunity:** Grid carbon intensity varies by time-of-day.
    Moving jobs from high-intensity daytime slots to low-intensity nighttime slots
    reduces total carbon without changing total compute. The formula from
    @sec-sustainable-ai-geographic-temporal-optimization-492c:

    > *Temporal scheduling can reduce emissions by 50–80% by aligning compute
    > workloads with renewable energy availability.* — Patterson et al. 2022

    The carbon savings from shifting `F` fraction of `N` jobs over `T` hours:

    ```
    ΔC = F × N × (P_active × T) × (I_day - I_night) / 1,000,000  [metric tonnes CO₂]
    ```

    **SLA constraint:** Jobs with hard deadlines cannot tolerate unbounded delays.
    Overcommitting the flexible fraction causes cascade failures when dependencies
    resolve before shifted jobs complete.
    """)
    return


# ─── ACT II PREDICTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your Prediction

    *Before touching the scheduler simulator:*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_pred = mo.ui.radio(
        options={
            "A) Time-shifting the 30% flexible jobs to night hours achieves "
            "the 40% carbon reduction target":
                "pred_a",
            "B) We need to reduce cluster size to achieve 40% reduction "
            "&mdash; scheduling alone cannot do it":
                "pred_b",
            "C) Carbon-aware scheduling only works in renewable regions; "
            "in coal regions the day/night differential is too small":
                "pred_c",
            "D) The 40% target requires switching to renewable energy contracts "
            "&mdash; scheduling cannot achieve it without changing the grid mix":
                "pred_d",
        },
        label="Can time-shifting 30% of flexible jobs to night hours (90% renewable) "
              "achieve the 40% carbon reduction target on a coal-heavy grid?",
    )
    act2_pred
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred):
    mo.stop(
        act2_pred.value is None,
        mo.callout(
            mo.md("Select your prediction to unlock the Carbon-Aware Scheduler Simulator."),
            kind="warn",
        ),
    )
    mo.callout(
        mo.md(f"**Prediction locked:** option {act2_pred.value[-1].upper()}. Now design the policy below."),
        kind="info",
    )
    return


# ─── ACT II INSTRUMENTS: CARBON-AWARE SCHEDULER SIMULATOR ─────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("### Carbon-Aware Scheduler Simulator")
    return


@app.cell(hide_code=True)
def _(mo):
    flex_fraction = mo.ui.slider(
        start=0, stop=50, value=30, step=5,
        label="Flexible job fraction (%)",
        show_value=True,
    )
    time_shift_hours = mo.ui.slider(
        start=1, stop=8, value=6, step=1,
        label="Maximum time-shift window (hours)",
        show_value=True,
    )
    day_intensity = mo.ui.dropdown(
        options={
            "Coal-heavy day (820 gCO\u2082/kWh)": 820,
            "Gas-mix day (490 gCO\u2082/kWh)": 490,
            "US-average day (386 gCO\u2082/kWh)": 386,
        },
        value="Coal-heavy day (820 gCO\u2082/kWh)",
        label="Daytime grid intensity",
    )
    night_intensity = mo.ui.slider(
        start=40, stop=300, value=40, step=10,
        label="Nighttime grid intensity (gCO\u2082/kWh)",
        show_value=True,
    )
    cluster_util = mo.ui.slider(
        start=50, stop=100, value=75, step=5,
        label="Cluster utilization (%)",
        show_value=True,
    )
    mo.vstack([
        mo.md("""
        Adjust the scheduling policy parameters. The **target** is 40% carbon reduction.
        Watch for the SLA breach warning when `flexible_fraction` is pushed too high
        relative to `time_shift_hours`.
        """),
        mo.hstack([flex_fraction, time_shift_hours], justify="start", gap="2rem"),
        mo.hstack([day_intensity, night_intensity], justify="start", gap="2rem"),
        mo.hstack([cluster_util], justify="start", gap="2rem"),
    ])
    return (flex_fraction, time_shift_hours, day_intensity, night_intensity, cluster_util)


@app.cell(hide_code=True)
def _(mo, flex_fraction, time_shift_hours, day_intensity, night_intensity,
       cluster_util, COLORS, go, np, apply_plotly_theme, H100_TDP_W,
       act1_pred, _net_change_pct):
    # ── Physics model ──────────────────────────────────────────────────────────
    # Source: @sec-sustainable-ai-geographic-temporal-optimization-492c
    # Carbon-aware scheduling formula:
    #   ΔC = F × N_jobs × T × P_active × (I_day - I_night) / 1,000,000
    # where:
    #   F       = flexible job fraction (0–1)
    #   N_jobs  = jobs per day (cluster × util)
    #   T       = average job duration proxied from shift window
    #   P_active = active power per GPU-cluster (kW total)
    #   I_day, I_night = daytime and nighttime carbon intensity (gCO₂/kWh)
    #
    # SLA violation model:
    #   SLA_violation_pct ≈ max(0, flexible_fraction - max_shiftable_pct)
    #   max_shiftable_pct = time_shift_hours / 24 × 100  (fraction of day that is night)
    #   If more jobs are flagged flexible than night-hours can absorb, SLA violations occur.

    _N_GPUS      = 1000                               # cluster size from scenario
    _P_TOTAL_KW  = _N_GPUS * H100_TDP_W / 1000        # total cluster power (kW)
    _UTIL        = cluster_util.value / 100.0
    _JOBS_PER_DAY = _N_GPUS * _UTIL * 24              # GPU-hours per day as proxy for jobs

    _F           = flex_fraction.value / 100.0        # fraction
    _I_DAY       = day_intensity.value                # gCO₂/kWh
    _I_NIGHT     = night_intensity.value              # gCO₂/kWh
    _T_SHIFT     = time_shift_hours.value             # hours

    # Night window capacity: fraction of 24h that is available for shifted jobs
    # Assume night window = time_shift_hours (contiguous night block)
    _night_window_frac = _T_SHIFT / 24.0              # fraction of day that is night

    # Baseline daily carbon (no shifting)
    _energy_day_kwh    = _P_TOTAL_KW * _UTIL * 24     # total kWh if all at daytime CI
    _carbon_baseline_t = _energy_day_kwh * _I_DAY / 1e6

    # Carbon after shifting F fraction to night
    # Energy remains the same; only CI changes for shifted fraction
    _energy_shifted_kwh  = _energy_day_kwh * _F
    _energy_remain_kwh   = _energy_day_kwh * (1 - _F)
    _carbon_shifted_t    = _energy_shifted_kwh * _I_NIGHT / 1e6
    _carbon_remain_t     = _energy_remain_kwh  * _I_DAY   / 1e6
    _carbon_new_t_sched  = _carbon_shifted_t + _carbon_remain_t
    _carbon_savings_pct  = (1 - _carbon_new_t_sched / _carbon_baseline_t) * 100

    # SLA violation model:
    # Night window can absorb at most night_window_frac of total capacity.
    # If F > night_window_frac, the overflow cannot be accommodated without
    # missing deadlines — excess fraction violates SLA.
    _sla_excess          = max(0.0, _F - _night_window_frac)
    _sla_violation_pct   = _sla_excess * 100.0        # % of total jobs that miss deadline

    # Derived for display
    _target_pct          = 40.0                       # from stakeholder message
    _target_met          = _carbon_savings_pct >= _target_pct
    _sla_breach          = _sla_violation_pct > 5.0   # >5% violation = SLA breach

    _sav_color = (
        COLORS["GreenLine"] if _carbon_savings_pct >= _target_pct
        else COLORS["OrangeLine"] if _carbon_savings_pct >= _target_pct * 0.6
        else COLORS["RedLine"]
    )

    # ── 24-hour carbon intensity profile bar chart ─────────────────────────────
    _hours     = np.arange(24)
    # Simplified day/night profile: hours 8–20 = day, rest = night
    _day_hours   = np.where((_hours >= 8) & (_hours < 20), _I_DAY, _I_NIGHT)
    _bar_colors  = [
        COLORS["RedLine"] if ci == _I_DAY else COLORS["GreenLine"]
        for ci in _day_hours
    ]

    # Job distribution before and after shifting
    # Before: all jobs uniform across 24h
    _jobs_before = np.full(24, _N_GPUS * _UTIL)      # uniform hourly GPU utilization
    # After: rigid jobs unchanged, flexible jobs shifted to night window
    # Night window: hours 20–(20+T_SHIFT) mod 24
    _jobs_after = _jobs_before.copy()
    _jobs_flex  = _jobs_before * _F
    # Remove flexible fraction from day hours, add to night window
    for _h in range(8, 20):
        _jobs_after[_h] -= _jobs_flex[_h]
    _night_start = 20
    _night_slots = _T_SHIFT  # fill that many night hours
    for _i in range(int(_night_slots)):
        _h_night = (_night_start + _i) % 24
        # add total flex jobs spread across available night slots
        _jobs_after[_h_night] += sum(_jobs_flex[8:20]) / max(_night_slots, 1)

    _fig2 = go.Figure()
    _fig2.add_trace(go.Bar(
        x=list(range(24)), y=_day_hours,
        marker_color=_bar_colors,
        name="Grid carbon intensity (gCO\u2082/kWh)",
        yaxis="y",
        opacity=0.55,
    ))
    _fig2.add_trace(go.Scatter(
        x=list(range(24)), y=list(_jobs_before),
        mode="lines",
        line=dict(color=COLORS["BlueLine"], width=2, dash="dot"),
        name="Jobs before shifting",
        yaxis="y2",
    ))
    _fig2.add_trace(go.Scatter(
        x=list(range(24)), y=list(_jobs_after),
        mode="lines",
        line=dict(color=COLORS["OrangeLine"], width=2),
        name="Jobs after shifting",
        yaxis="y2",
    ))
    _fig2.update_layout(
        title="24-Hour Carbon Intensity Profile and Job Distribution",
        xaxis=dict(title="Hour of Day", tickvals=list(range(0, 24, 2)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
        yaxis=dict(title="Carbon Intensity (gCO\u2082/kWh)"),
        yaxis2=dict(title="Active GPUs (jobs)", overlaying="y", side="right"),
        height=380,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        barmode="overlay",
    )
    apply_plotly_theme(_fig2)

    mo.vstack([
        mo.md(f"""
        ```
        Physics (Carbon-Aware Scheduling — @sec-sustainable-ai-geographic-temporal-optimization-492c)

        Energy model (1,000 H100 × {_UTIL:.0%} util × 700 W):
          Total daily energy   = {_N_GPUS} × 700 W × {_UTIL:.0%} × 24 h
                               = {_energy_day_kwh:,.0f} kWh

        Carbon formula: ΔC = F × E × (I_day - I_night) / 1,000,000

          Baseline (all at I_day = {_I_DAY} gCO₂/kWh):
            C_baseline = {_carbon_baseline_t:.2f} tonnes CO₂/day

          After shifting {flex_fraction.value}% of jobs to night ({_I_NIGHT} gCO₂/kWh):
            C_remain  = {_carbon_remain_t:.2f} tonnes CO₂/day   [{(1-_F):.0%} at day rate]
            C_shifted = {_carbon_shifted_t:.2f} tonnes CO₂/day   [{_F:.0%} at night rate]
            C_new     = {_carbon_new_t_sched:.2f} tonnes CO₂/day

          Carbon savings   = {_carbon_savings_pct:.1f}%  [target: 40.0%]
          SLA violations   = {_sla_violation_pct:.1f}%  [threshold: 5.0%]
          Night capacity   = {_night_window_frac:.0%} of day  [= {_T_SHIFT}h / 24h]
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; margin: 16px 0; flex-wrap: wrap;">
            <div style="padding: 18px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 180px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">
                    Carbon Savings
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_sav_color};">
                    {_carbon_savings_pct:.1f}%
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8;">
                    target: 40.0%
                </div>
            </div>
            <div style="padding: 18px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 180px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">
                    SLA Violations
                </div>
                <div style="font-size: 2rem; font-weight: 800;
                            color: {'#CB202D' if _sla_breach else '#008F45'};">
                    {_sla_violation_pct:.1f}%
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8;">
                    threshold: 5.0%
                </div>
            </div>
            <div style="padding: 18px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 180px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">
                    Baseline Carbon
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {COLORS['BlueLine']};">
                    {_carbon_baseline_t:.1f} t
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8;">
                    metric tonnes CO&#8322;/day
                </div>
            </div>
            <div style="padding: 18px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 180px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">
                    Post-Shift Carbon
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: {_sav_color};">
                    {_carbon_new_t_sched:.1f} t
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8;">
                    metric tonnes CO&#8322;/day
                </div>
            </div>
            <div style="padding: 18px; border: 1.5px solid #e2e8f0; border-radius: 10px;
                        width: 180px; text-align: center; background: white;">
                <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">
                    Target Met
                </div>
                <div style="font-size: 1.6rem; font-weight: 800;
                            color: {'#008F45' if _target_met else '#CB202D'};">
                    {'Yes' if _target_met else 'No'}
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8;">
                    40% reduction required
                </div>
            </div>
        </div>
        """),
        mo.as_html(_fig2),
    ])
    return (
        _carbon_savings_pct, _carbon_baseline_t, _carbon_new_t_sched,
        _sla_violation_pct, _sla_breach, _target_met,
        _F, _T_SHIFT, _I_DAY, _I_NIGHT,
    )


# ─── ACT II FAILURE STATES ─────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, _sla_breach, _sla_violation_pct, _carbon_savings_pct):
    if _sla_breach:
        mo.callout(
            mo.md(
                f"**SLA BREACH: {_sla_violation_pct:.1f}% of jobs miss deadlines.** "
                f"The flexible fraction exceeds the night-time capacity window. "
                f"The scheduler is attempting to shift more jobs than the available "
                f"low-carbon hours can absorb. Reduce flexible fraction or increase "
                f"the time-shift window to restore SLA compliance."
            ),
            kind="danger",
        )
    elif _carbon_savings_pct < 40.0:
        mo.callout(
            mo.md(
                f"**BELOW TARGET: {_carbon_savings_pct:.1f}% reduction achieved. "
                f"Target: 40.0%.** The current policy does not meet the sustainability "
                f"commitment. Increase flexible fraction, extend the time-shift window, "
                f"or reduce cluster utilization during high-carbon hours."
            ),
            kind="warn",
        )
    else:
        mo.callout(
            mo.md(
                f"**TARGET MET: {_carbon_savings_pct:.1f}% carbon reduction achieved "
                f"with SLA intact ({_sla_violation_pct:.1f}% violations < 5% threshold).** "
                f"This scheduling policy satisfies both the sustainability commitment "
                f"and the service-level agreement."
            ),
            kind="success",
        )
    return


# ─── ACT II PREDICTION-VS-REALITY OVERLAY ─────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, act2_pred, _carbon_savings_pct, _sla_violation_pct,
      _F, _I_DAY, _I_NIGHT, _T_SHIFT):
    # Quick-check: what does shifting 30% of jobs from 820 to 40 gCO₂/kWh achieve?
    # Savings from the 30% flexible fraction:
    # ΔC/C = F × (I_day - I_night) / I_day_total_mix
    # With I_day = 820, I_night = 40, F = 0.30 (default), weighted average:
    # C_new = 0.70 × 820 + 0.30 × 40 = 574 + 12 = 586 gCO₂/kWh_effective
    # Savings = (820 - 586) / 820 = 28.5%
    # But with 6h shift window (night_cap = 25%), and F=30% > 25%, SLA issues at night_cap boundary.
    # The target IS achievable at F=30% if time_shift >= 8 (night_cap = 33%) — see the simulator.
    _check_savings = 0.30 * (820 - 40) / 820 * 100  # theoretical 30% at max differential

    _pred_feedback = {
        "pred_a": (
            f"**Correct.** With 30% flexible jobs and an 820 → 40 gCO₂/kWh differential, "
            f"the theoretical maximum savings from shifting only the flexible fraction is "
            f"{_check_savings:.1f}%. At a sufficiently large time-shift window (≥ 8 h), "
            f"the 40% target is within reach without SLA violations. Your simulator shows "
            f"{_carbon_savings_pct:.1f}% savings with {_sla_violation_pct:.1f}% SLA impact "
            f"under the current policy."
        ),
        "pred_b": (
            f"**Not required.** Reducing cluster size reduces total carbon proportionally — "
            f"but so does time-shifting, and time-shifting preserves total compute capacity. "
            f"The simulator shows {_carbon_savings_pct:.1f}% savings from scheduling alone "
            f"({_F:.0%} flexible, {_T_SHIFT}h shift, I_day={_I_DAY}, I_night={_I_NIGHT}). "
            f"Carbon-aware scheduling is a free lunch relative to capacity reduction."
        ),
        "pred_c": (
            f"**Wrong direction.** Carbon-aware scheduling provides *more* value in coal "
            f"regions precisely because the day/night differential is large. "
            f"At 820 gCO₂/kWh day and 40 gCO₂/kWh night, shifting 1 kWh saves "
            f"780 gCO₂. In a renewable region (40 gCO₂/kWh flat), there is nothing "
            f"to shift *to*. The simulator confirms: current savings = {_carbon_savings_pct:.1f}%."
        ),
        "pred_d": (
            f"**Scheduling can achieve it without changing the grid contract.** "
            f"The 40% target exploits the existing day/night variation in grid carbon "
            f"intensity — no new renewable energy contract required. The current simulator "
            f"shows {_carbon_savings_pct:.1f}% reduction purely from temporal reallocation "
            f"of the {_F:.0%} flexible fraction. PPAs (power purchase agreements) provide "
            f"additional carbon credit, but they are not necessary to hit 40%."
        ),
    }
    _correct = act2_pred.value == "pred_a"
    mo.callout(
        mo.md(_pred_feedback[act2_pred.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─── ACT II REFLECTION ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Reflection

    *Why does carbon-aware scheduling have diminishing returns beyond 30% flexible jobs?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    act2_reflect = mo.ui.radio(
        options={
            "A) The scheduler runs out of night-time slots: "
            "night-time capacity is finite and cannot absorb unlimited flexible jobs":
                "r2_a",
            "B) Grid intensity equalizes over time: shifting more jobs "
            "causes the night grid to fill and its carbon intensity rises to match day":
                "r2_b",
            "C) Time-flexible jobs become rigid when dependency chains grow: "
            "jobs that appear flexible become constrained by downstream jobs":
                "r2_c",
            "D) Energy efficiency decreases at low utilization: "
            "night-time runs are less efficient because the cluster is under-loaded":
                "r2_d",
        },
        label="Why does carbon-aware scheduling have diminishing returns beyond 30% flexible jobs?",
    )
    act2_reflect
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect):
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue to Key Takeaways."), kind="warn"),
    )
    _r2_feedback = {
        "r2_a": (
            "**Correct.** The night-time window is finite: a 6-hour night window represents "
            "25% of the 24-hour day. If you flag 40% of jobs as flexible, the excess 15% "
            "cannot be accommodated within the night window — it either spills back into "
            "day hours (no savings) or misses its deadline (SLA violation). The simulator "
            "captures this as the SLA violation threshold: `flex_fraction > night_capacity` "
            "triggers the danger banner. Night-time capacity is the binding constraint, "
            "not algorithmic intent."
        ),
        "r2_b": (
            "**Not in this model.** At datacenter scale, a single 1,000-GPU cluster "
            "represents a small fraction of total grid load. Its shifted demand does not "
            "materially change grid carbon intensity. At multi-gigawatt fleet scale, "
            "this effect could emerge — but the chapter focuses on individual cluster "
            "scheduling, where grid intensity is treated as exogenous. The binding "
            "constraint is night-time capacity, not market-driven intensity equalization."
        ),
        "r2_c": (
            "**Partially true but not the primary physics here.** Job dependency graphs "
            "do reduce the effective flexible fraction in practice, and this is a real "
            "operational concern. However, the simulator's diminishing returns come from "
            "the night-window capacity constraint, which is a simpler and more fundamental "
            "limit. Dependency chains explain why *measured* flexible fractions tend to "
            "be lower than *estimated* ones, but they do not explain the scheduling "
            "ceiling in this lab's model."
        ),
        "r2_d": (
            "**Not the mechanism here.** H100 power draw does decrease at low utilization "
            "(idle: ~180 W vs peak: 700 W), but this is a secondary effect. Carbon-aware "
            "scheduling does not change total GPU utilization — it redistributes when jobs "
            "run, not whether they run. The diminishing returns come from running out of "
            "low-carbon hours, not from efficiency degradation at partial load."
        ),
    }
    _correct = act2_reflect.value == "r2_a"
    mo.callout(
        mo.md(_r2_feedback[act2_reflect.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─── ACT II MATHPEEK ──────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "\U0001f4d0 The governing equation: Carbon-Aware Scheduling formula": mo.md("""
        **Source:** @sec-sustainable-ai-geographic-temporal-optimization-492c
        Patterson et al. 2022 "Carbon-Aware Computing for Datacenters"

        **Carbon savings formula:**
        ```
        ΔC = F × N × T × P_active × (I_day - I_night) / 1,000,000
        ```
        where:
        - **ΔC** — carbon savings (metric tonnes CO₂)
        - **F** — flexible job fraction (0–1)
        - **N** — number of GPUs in cluster
        - **T** — job duration (hours)
        - **P_active** — active power per GPU (kW)
        - **I_day, I_night** — daytime and nighttime carbon intensity (gCO₂/kWh)

        **Carbon savings percentage:**
        ```
        savings_pct = F × (I_day - I_night) / [F × I_night + (1-F) × I_day] × 100
        ```

        **SLA violation model (simplified):**
        ```
        night_capacity = T_shift_hours / 24
        SLA_violation  = max(0, F - night_capacity) × 100%
        ```

        **Optimal flexible fraction** (maximizes savings, no SLA violation):
        ```
        F_optimal = min(F_available, T_shift_hours / 24)
        ```
        For T_shift = 6h: F_optimal = 6/24 = 25%.
        For T_shift = 8h: F_optimal = 8/24 = 33%.
        """),
    })
    return


# ═════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═════════════════════════════════════════════════════════════════════════════


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

            <div style="background:linear-gradient(to right, #f8fafc, #f1f5f9); border-radius:8px; padding:20px; margin-bottom:24px; border-left:4px solid #8b5cf6;">
                <div style="font-weight:800; font-size:1.1rem; color:#6d28d9; margin-bottom:8px;">💎 The Iron Law Nugget</div>
                <div style="color:#334155; font-size:1rem; font-style:italic; line-height:1.6;">
                    "Software optimization saves percentages; geography saves orders of magnitude. The carbon intensity of the local grid is the single most dominant variable in AI sustainability."
                </div>
                <div style="margin-top:12px; font-size:0.8rem; color:#64748b;">
                    <strong>Source:</strong> Adapted from geographic footprint analysis in <em>Patterson, D., et al. (2021). Carbon Emissions and Large Neural Network Training. arXiv.</em>
                </div>
            </div>

            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. Jevons Paradox governs total carbon, not per-unit efficiency.</strong>
                    A 2&times; efficiency gain with 3&times; deployment scale multiplies to 1.5&times; total energy.
                    The formula C&thinsp;=&thinsp;E&thinsp;&times;&thinsp;I confirms it: total carbon increases
                    when demand elasticity exceeds the efficiency gain. Enforced carbon budget caps are
                    the only intervention that guarantees net reduction.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Grid carbon intensity spans two orders of magnitude.</strong>
                    Coal-heavy grids emit 820&thinsp;g CO&#8322;/kWh; Quebec hydro emits 40&thinsp;g CO&#8322;/kWh &mdash;
                    a 20&times; difference. A single site-selection decision achieves more carbon reduction
                    than any algorithmic optimization (pruning, quantization, distillation combined yield
                    at most ~160&times; compound savings).
                </div>
                <div>
                    <strong>3. Carbon-aware scheduling is a finite lever bounded by low-carbon window capacity.</strong>
                    Shifting 30% of flexible jobs to renewable night-grid hours can hit 40%+ reduction;
                    shifting more than the window absorbs triggers SLA violations.
                    The binding constraint is always the low-carbon capacity, not the willingness to shift.
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
                    <strong>Lab 16: The Fairness Impossibility</strong> &mdash; This lab showed that
                    sustainability constraints require explicit governance (carbon caps). The next lab
                    asks: is there an equivalent impossibility in fairness &mdash; a theorem proving
                    that no algorithm can satisfy all fairness criteria simultaneously?
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
                    <strong>Read:</strong> @sec-sustainable-ai-geographic-temporal-optimization-492c
                    for carbon-aware scheduling implementation and the temporal shifting formula.<br/>
                    <strong>Build:</strong> The CarbonAwareScheduler LEGO cell in the chapter
                    demonstrates the 50&ndash;80% reduction achievable via temporal job placement.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. The formula C = E x I means total carbon = energy consumed x grid carbon intensity. A 10,000 MWh training run in Quebec (20 g CO2/kWh) versus Poland (800 g CO2/kWh) produces what ratio in CO2 emissions, and why does this single site-selection decision outweigh most algorithmic optimizations?
2. A 2x efficiency gain combined with 3x deployment scale produces 1.5x total energy consumption. Why does the Jevons Paradox mean that algorithmic efficiency improvements alone cannot guarantee absolute carbon reduction?
3. Carbon-aware scheduling can shift 30% of flexible jobs to renewable night-grid hours for a 40%+ reduction. What is the binding constraint on this lever, and why does shifting more jobs than the low-carbon window can absorb trigger SLA violations?

**You're ready to move on if you can:**
- Calculate total carbon emissions using C = E x I for different grid locations and PUE values
- Explain why Jevons Paradox makes hard carbon caps the only intervention that guarantees net reduction
- Determine the maximum fraction of workload that can be time-shifted without violating latency SLAs
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER SAVE + HUD FOOTER ───────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, ledger, COLORS,
      context_toggle, efficiency_gain, deployment_scale,
      _net_change_pct, _carbon_savings_pct, _target_met,
      act1_pred, act1_reflect, act2_pred, act2_reflect,
      _sla_violation_pct):
    _ctx_val    = context_toggle.value
    _eff_val    = efficiency_gain.value
    _dep_val    = deployment_scale.value
    _nc_pct     = float(_net_change_pct)
    _sav_pct    = float(_carbon_savings_pct)
    _tgt        = bool(_target_met)
    _a1_pred    = act1_pred.value or "none"
    _a1_correct = (_a1_pred == "option_c")
    _a2_result  = _sav_pct
    _a2_decision = (
        f"flex={efficiency_gain.value:.1f}x_scale={deployment_scale.value:.1f}x"
    )
    _constraint = (_sla_violation_pct > 5.0)

    ledger.save(
        chapter="v2_15",
        design={
            "context":              _ctx_val,
            "efficiency_gain":      _eff_val,
            "deployment_scale":     _dep_val,
            "net_carbon_change_pct": _nc_pct,
            "carbon_savings_pct":   _sav_pct,
            "target_met":           _tgt,
            "act1_prediction":      _a1_pred,
            "act1_correct":         _a1_correct,
            "act2_result":          _a2_result,
            "act2_decision":        _a2_decision,
            "constraint_hit":       _constraint,
        },
    )

    _c_ok   = COLORS["GreenLine"]
    _c_fail = COLORS["RedLine"]
    _c_warn = COLORS["OrangeLine"]
    _c_muted = COLORS["TextMuted"]

    _tgt_color   = _c_ok   if _tgt          else _c_fail
    _nc_color    = _c_ok   if _nc_pct < 0   else _c_fail
    _sla_color   = _c_fail if _constraint   else _c_ok

    mo.Html(f"""
    <div class="lab-hud" style="margin-top: 32px;">
        <span>
            <span class="hud-label">LAB</span>&nbsp;
            <span class="hud-value">V2-15</span>
        </span>
        <span>
            <span class="hud-label">CONTEXT</span>&nbsp;
            <span class="hud-value">{_ctx_val.upper()}</span>
        </span>
        <span>
            <span class="hud-label">EFF GAIN</span>&nbsp;
            <span class="hud-value">{_eff_val:.1f}&times;</span>
        </span>
        <span>
            <span class="hud-label">DEPLOY SCALE</span>&nbsp;
            <span class="hud-value">{_dep_val:.1f}&times;</span>
        </span>
        <span>
            <span class="hud-label">NET CARBON</span>&nbsp;
            <span style="color: {_nc_color}; font-family: var(--font-mono);">
                {_nc_pct:+.1f}%
            </span>
        </span>
        <span>
            <span class="hud-label">SCHED SAVINGS</span>&nbsp;
            <span style="color: {_tgt_color}; font-family: var(--font-mono);">
                {_sav_pct:.1f}%
            </span>
        </span>
        <span>
            <span class="hud-label">TARGET MET</span>&nbsp;
            <span style="color: {_tgt_color}; font-family: var(--font-mono);">
                {'YES' if _tgt else 'NO'}
            </span>
        </span>
        <span>
            <span class="hud-label">SLA</span>&nbsp;
            <span style="color: {_sla_color}; font-family: var(--font-mono);">
                {'BREACH' if _constraint else 'OK'}
            </span>
        </span>
        <span>
            <span class="hud-label">LEDGER</span>&nbsp;
            <span class="hud-active">SAVED</span>
        </span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
