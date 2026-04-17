import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB V2-14: THE CARBON BUDGET
#
# Volume II, Chapter 14 — Sustainable AI
#
# Core Invariant: AI compute demand outpaces hardware efficiency by 195,000x.
#   Geography is a 40x carbon lever. Embodied carbon dominates on clean grids.
#   The Jevons Paradox means efficiency gains can INCREASE total consumption.
#   Only absolute caps guarantee net reduction.
#
# 5 Parts (~60 minutes):
#   Part A — The Energy Wall (10 min)
#   Part B — The Geography of Carbon (12 min)
#   Part C — The Lifecycle Carbon Shift (12 min)
#   Part D — The Jevons Trap (14 min)  *** THE highlight of Vol 2 ***
#   Part E — Carbon-Aware Fleet Design (12 min)
#
# Design Ledger: saves chapter="v2_14"
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
    from mlsysim.labs.components import DecisionLog
    from mlsysim.hardware.registry import Hardware
    from mlsysim.models.registry import Models

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()

    # ── Hardware from registry (Cloud + Edge tiers) ─────────────────────────
    _cloud = Hardware.Cloud.H100
    _edge  = Hardware.Edge.JetsonOrinNX

    # ── Sustainability constants ────────────────────────────────────────────
    # Grid carbon intensities (gCO2/kWh) — Source: IEA (2023), chapter data
    CI_QUEBEC    = 20     # Hydro-dominant
    CI_ICELAND   = 28     # Geothermal + hydro
    CI_FRANCE    = 56     # Nuclear-dominant
    CI_US_AVG    = 429    # Mixed grid
    CI_TEXAS     = 400    # Mixed (EPA eGRID South Central)
    CI_GERMANY   = 385    # Coal + wind transition
    CI_CHINA_AVG = 555    # Coal-heavy
    CI_POLAND    = 820    # Coal-dominant
    CI_INDIA     = 720    # Coal-heavy

    # PUE values — Source: chapter facility metrics section
    PUE_LIQUID   = 1.06   # Liquid-cooled hyperscale
    PUE_AIR      = 1.12   # Best air-cooled
    PUE_LEGACY   = 1.58   # Legacy facility

    # Hardware power from registry — Source: chapter lifecycle section
    H100_EMBODIED_KG = 175.0   # kg CO2eq per H100 (midpoint 150-200)
    H100_TDP_W       = _cloud.tdp.m_as("W")   # 700 W from registry
    EDGE_TDP_W       = _edge.tdp.m_as("W")    # 25 W — edge power for comparison

    # Growth rates — Source: chapter energy wall section
    DEMAND_DOUBLING_MONTHS = 3.4    # AI compute demand doubling time
    EFFICIENCY_DOUBLING_MONTHS = 24  # Hardware efficiency doubling time

    # Jevons Paradox — Source: chapter implementation solutions section
    # Elasticity estimates from observed API pricing vs volume data
    JEVONS_ELASTICITY_INELASTIC = 0.3
    JEVONS_ELASTICITY_UNIT      = 1.0
    JEVONS_ELASTICITY_ELASTIC   = 2.0

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme,
        go, np, math,
        CI_QUEBEC, CI_ICELAND, CI_FRANCE, CI_US_AVG, CI_TEXAS,
        CI_GERMANY, CI_CHINA_AVG, CI_POLAND, CI_INDIA,
        PUE_LIQUID, PUE_AIR, PUE_LEGACY,
        H100_EMBODIED_KG, H100_TDP_W, EDGE_TDP_W,
        DEMAND_DOUBLING_MONTHS, EFFICIENCY_DOUBLING_MONTHS,
        JEVONS_ELASTICITY_INELASTIC, JEVONS_ELASTICITY_UNIT, JEVONS_ELASTICITY_ELASTIC,
        DecisionLog,
    )


# ─── CELL 1: HEADER ────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    mo.vstack([
        LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0a1628 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab 14
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The Carbon Budget
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Energy Wall &middot; Geography &middot; Lifecycle &middot; Jevons Paradox
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #64748b;
                      max-width: 680px; line-height: 1.65;">
                An executive announces: &ldquo;We will make our AI 2&times; more efficient, cutting
                our carbon footprint in half.&rdquo; The math says otherwise. Efficiency gains can
                <em>increase</em> total consumption when demand is elastic.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(0,143,69,0.18); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(0,143,69,0.3);">
                    5 Parts &middot; ~60 min
                </span>
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    Chapter 14: Sustainable AI
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-fail">195,000x Energy Deficit</span>
                <span class="badge badge-warn">40x Geography Gap</span>
                <span class="badge badge-info">Embodied Carbon Dominates on Clean Grids</span>
                <span class="badge badge-ok">Jevons Paradox: Efficiency Backfires</span>
            </div>
        </div>
        """),
    ])
    return


# ─── CELL 2: BRIEFING ──────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['GreenLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Quantify the energy deficit</strong>: calculate the 195,000&times; gap between AI compute demand growth (~3.4-month doubling) and hardware efficiency growth (~24-month doubling) over 7 years.</div>
                <div style="margin-bottom: 3px;">2. <strong>Apply the Jevons Paradox equation</strong> to show that 2&times; efficiency with elasticity 2.0 produces a 100% <em>increase</em> in total energy, and identify the elasticity threshold where efficiency gains guarantee net reduction.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a carbon-aware fleet strategy</strong> combining geographic optimization (40&times; lever), lifecycle management (embodied carbon), temporal scheduling, and absolute carbon caps to achieve a 50% emission reduction target.</div>
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
                    Operational carbon equation (C = E &times; CI &times; PUE) from Sustainable AI chapter &middot;
                    PUE definition &middot; Jevons Paradox from Sustainable AI chapter
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>~60 min</strong><br/>
                    A: 10 &middot; B: 12 &middot; C: 12 &middot; D: 14 &middot; E: 12
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                &ldquo;If you double AI efficiency, does total energy consumption go up or down
                &mdash; and what is the only mechanism that guarantees it goes down?&rdquo;
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

    - **Sustainable AI chapter** (energy wall section) -- The 350,000x compute demand growth
      vs ~1.5x/year efficiency improvement, and the exponential energy deficit.
    - **Sustainable AI chapter** (carbon footprint calculation) -- Operational carbon equation
      C = E x CI x PUE, grid carbon intensity table, PUE range 1.06-1.58.
    - **Sustainable AI chapter** (lifecycle analysis) -- Embodied carbon (150-200 kg CO2eq per H100),
      the shift from operational to embodied dominance on clean grids.
    - **Sustainable AI chapter** (implementation solutions) -- Jevons Paradox, demand elasticity,
      carbon-aware scheduling, and absolute carbon caps.
    """), kind="info")
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: WIDGET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 4: PART A WIDGETS ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partA_pred = mo.ui.radio(
        options={
            "A) ~10x -- hardware almost keeps up": "10",
            "B) ~1,000x -- significant but manageable": "1000",
            "C) ~100,000x+ -- an exponential chasm": "100000",
            "D) Roughly even -- Moore's Law keeps up": "even",
        },
        label="AI compute demand doubles every ~3.4 months. Hardware efficiency doubles every ~24 months. Over 7 years (2012-2019), how large is the gap?",
    )
    partA_years_slider = mo.ui.slider(start=1, stop=10, value=7, step=1, label="Timeline (years)")
    return (partA_pred, partA_years_slider)


# ─── CELL 5: PART B WIDGETS ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partB_pred = mo.ui.radio(
        options={
            "A) ~2-3x -- not much variation": "2",
            "B) ~5-10x -- moderate difference": "5",
            "C) ~40x -- geography dominates": "40",
            "D) ~100x -- extreme variation": "100",
        },
        label="A 10,000 MWh training run. Quebec (hydro, 20 gCO2/kWh) vs Poland (coal, 820 gCO2/kWh). What is the carbon ratio?",
    )
    partB_energy_slider = mo.ui.slider(start=1000, stop=100000, value=10000, step=1000, label="Training energy (MWh)")
    partB_pue_slider = mo.ui.slider(start=1.0, stop=2.0, value=1.12, step=0.02, label="PUE")
    return (partB_energy_slider, partB_pred, partB_pue_slider)


# ─── CELL 6: PART C WIDGETS ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partC_pred = mo.ui.radio(
        options={
            "A) <5% -- hardware is a rounding error": "5",
            "B) ~10-15%": "15",
            "C) ~30-50% -- a major fraction": "40",
            "D) ~80%+ -- hardware dominates": "80",
        },
        label="In a datacenter powered by 100% renewable energy, what fraction of total lifecycle carbon comes from hardware manufacturing?",
    )
    partC_refresh_slider = mo.ui.slider(start=2, stop=5, value=3, step=1, label="Hardware refresh cycle (years)")
    partC_util_slider = mo.ui.slider(start=30, stop=90, value=60, step=5, label="GPU utilization (%)")
    partC_gpu_count = mo.ui.slider(start=100, stop=10000, value=1000, step=100, label="GPU count")
    return (partC_gpu_count, partC_pred, partC_refresh_slider, partC_util_slider)


# ─── CELL 7: PART D WIDGETS ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partD_pred = mo.ui.number(
        start=-80, stop=200, value=-25, step=5,
        label="You double inference efficiency (cost per query halves). Demand increases 3x (elastic market). What % change in total energy? (negative = decrease)",
    )
    partD_eff_slider = mo.ui.slider(start=1.0, stop=10.0, value=2.0, step=0.5, label="Efficiency improvement (x)")
    partD_elast_slider = mo.ui.slider(start=0.1, stop=3.0, value=2.0, step=0.1, label="Demand elasticity")
    partD_cap_toggle = mo.ui.switch(label="Carbon cap enabled", value=False)
    partD_cap_level = mo.ui.slider(start=0.5, stop=2.0, value=1.0, step=0.1, label="Cap level (fraction of baseline)")
    return (partD_cap_level, partD_cap_toggle, partD_eff_slider, partD_elast_slider, partD_pred)


# ─── CELL 8: PART E WIDGETS ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    partE_geo = mo.ui.dropdown(
        options={"Poland (820 g/kWh)": 820, "US Average (429 g/kWh)": 429,
                 "France (56 g/kWh)": 56, "Quebec (20 g/kWh)": 20},
        value="US Average (429 g/kWh)", label="Primary region:",
    )
    partE_temporal = mo.ui.slider(start=0, stop=60, value=0, step=5, label="Temporal shift (% of jobs to off-peak)")
    partE_eff_gain = mo.ui.slider(start=1.0, stop=4.0, value=1.0, step=0.5, label="Efficiency improvement (x)")
    partE_cap = mo.ui.slider(start=0.3, stop=1.5, value=1.0, step=0.1, label="Carbon cap (fraction of baseline)")
    return (partE_cap, partE_eff_gain, partE_geo, partE_temporal)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: SINGLE TABS CELL
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 9: TABS ────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo, go, np, math,
    COLORS, apply_plotly_theme, CI_QUEBEC, CI_ICELAND,
    CI_FRANCE, CI_US_AVG, CI_TEXAS, CI_GERMANY,
    CI_CHINA_AVG, CI_POLAND, CI_INDIA, PUE_LIQUID,
    PUE_AIR, PUE_LEGACY, H100_EMBODIED_KG, H100_TDP_W,
    DEMAND_DOUBLING_MONTHS, EFFICIENCY_DOUBLING_MONTHS, JEVONS_ELASTICITY_INELASTIC, JEVONS_ELASTICITY_UNIT,
    JEVONS_ELASTICITY_ELASTIC, partA_pred, partA_years_slider, partB_energy_slider,
    partB_pred, partB_pue_slider, partC_gpu_count, partC_pred,
    partC_refresh_slider, partC_util_slider, partD_cap_level, partD_cap_toggle,
    partD_eff_slider, partD_elast_slider, partD_pred, partE_cap,
    partE_eff_gain, partE_geo, partE_temporal,
):

    # ─────────────────────────────────────────────────────────────────────
    # PART A BUILDER -- The Energy Wall
    # ─────────────────────────────────────────────────────────────────────

    def build_part_a():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP of Infrastructure
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Our AI compute demand is doubling every few months, but our hardware
                roadmap shows efficiency doubling every two years. Over the next decade,
                how large will this gap actually get?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Energy Wall

        You expect hardware efficiency to keep pace with AI compute demand &mdash; Moore's
        Law has always delivered. But AI demand doubles every 3.4 months while efficiency
        doubles every 24 months. Over 7 years, the gap is not 10x. It is 195,000x.
        """))

        items.append(partA_pred)

        if partA_pred.value is None:
            items.append(mo.callout(
                mo.md("**Select your prediction above to unlock the instruments.**"),
                kind="warn",
            ))
            return mo.vstack(items)

        # Instruments
        items.append(partA_years_slider)

        _years = partA_years_slider.value
        _t = np.linspace(0, _years, 200)

        # demand(t) = 2^(t * 12 / doubling_months)
        _demand = 2 ** (_t * 12 / DEMAND_DOUBLING_MONTHS)
        _efficiency = 2 ** (_t * 12 / EFFICIENCY_DOUBLING_MONTHS)
        _gap = _demand / _efficiency

        _gap_final = _gap[-1]

        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=_t, y=_demand, name="AI Compute Demand",
            line=dict(color=COLORS["RedLine"], width=3),
            fill="tonexty" if False else None,
        ))
        fig_energy.add_trace(go.Scatter(
            x=_t, y=_efficiency, name="Hardware Efficiency",
            line=dict(color=COLORS["GreenLine"], width=3),
        ))

        # Shaded gap region
        fig_energy.add_trace(go.Scatter(
            x=np.concatenate([_t, _t[::-1]]),
            y=np.concatenate([_demand, _efficiency[::-1]]),
            fill="toself", fillcolor="rgba(203,32,45,0.08)",
            line=dict(width=0), name="Energy Deficit", showlegend=True,
        ))

        fig_energy.update_layout(
            height=400,
            xaxis=dict(title="Years (from 2012)"),
            yaxis=dict(title="Relative Scale", type="log"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_energy)

        # Annotation note about the caveat
        _caveat = ("Note: The demand curve represents frontier training run compute growth "
                   "(Amodei/Hernandez 2018), not total industry energy consumption. "
                   "Industry-wide growth is substantial but less extreme.")

        items.append(mo.md(f"### The Energy Deficit Over {_years} Years"))
        items.append(mo.as_html(fig_energy))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 12px;">
            <div style="padding: 16px 24px; border: 2px solid {COLORS['RedLine']}; border-radius: 12px;
                        text-align: center; min-width: 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Demand Growth</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['RedLine']};">
                    {_demand[-1]:,.0f}x</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">doubles every {DEMAND_DOUBLING_MONTHS:.1f} months</div>
            </div>
            <div style="padding: 16px 24px; border: 2px solid {COLORS['GreenLine']}; border-radius: 12px;
                        text-align: center; min-width: 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Efficiency Growth</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['GreenLine']};">
                    {_efficiency[-1]:,.0f}x</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">doubles every {EFFICIENCY_DOUBLING_MONTHS:.0f} months</div>
            </div>
            <div style="padding: 16px 24px; border: 2px solid {COLORS['OrangeLine']}; border-radius: 12px;
                        text-align: center; min-width: 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Energy Deficit</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['OrangeLine']};">
                    {_gap_final:,.0f}x</div>
            </div>
        </div>
        """))
        items.append(mo.callout(mo.md(f"*{_caveat}*"), kind="info"))

        # Prediction reveal
        _correct = partA_pred.value == "100000"
        _msg = ("You correctly identified the ~100,000x+ exponential chasm."
                if _correct else
                "The gap over 7 years is ~195,000x. Students intuitively expect hardware to 'keep up' "
                "because Moore's Law worked for decades. But AI demand grows 7x faster than efficiency.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        items.append(mo.accordion({
            "Math Peek: The Exponential Energy Deficit": mo.md("""
**Demand vs. efficiency growth:**
$$
\\text{Gap}(t) = \\frac{2^{t / T_{\\text{demand}}}}{2^{t / T_{\\text{eff}}}}
= 2^{t \\left(\\frac{1}{T_{\\text{demand}}} - \\frac{1}{T_{\\text{eff}}}\\right)}
$$

**Where:**
- **$T_{\\text{demand}}$**: Demand doubling time (~3.4 months for AI compute)
- **$T_{\\text{eff}}$**: Hardware efficiency doubling time (~24 months)
- **$t$**: Time in months

**At $t = 84$ months (7 years):**
$$
\\text{Gap} = 2^{84 \\times (1/3.4 - 1/24)} = 2^{84 \\times 0.253} = 2^{21.2} \\approx 195{,}000\\times
$$

Demand grows ~7x faster than efficiency. No amount of hardware improvement
closes this gap without demand-side intervention.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART B BUILDER -- The Geography of Carbon
    # ─────────────────────────────────────────────────────────────────────

    def build_part_b():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Head of Sustainability
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;The energy wall is real. You cannot outrun it with better chips. So
                <em>where</em> you compute and what powers it becomes the dominant variable.
                Grid carbon intensity varies 40&times; across regions.&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Geography of Carbon

        The energy wall is real. You cannot outrun it with better chips. So *where*
        you compute and what powers it becomes the dominant variable. Grid carbon intensity
        varies 40x across regions -- a single site selection decision rivals the
        entire algorithmic optimization toolkit.
        """))

        items.append(partB_pred)

        if partB_pred.value is None:
            items.append(mo.callout(
                mo.md("**Select your prediction to unlock.**"),
                kind="warn",
            ))
            return mo.vstack(items)

        # Controls
        items.append(mo.hstack([partB_energy_slider, partB_pue_slider], justify="start", gap=1))

        _energy = partB_energy_slider.value  # MWh
        _pue = partB_pue_slider.value

        _regions = ["Quebec\n(Hydro)", "Iceland\n(Geo)", "France\n(Nuclear)", "Germany\n(Mixed)",
                    "US Avg\n(Mixed)", "Texas\n(Mixed)", "India\n(Coal)", "Poland\n(Coal)"]
        _cis = [CI_QUEBEC, CI_ICELAND, CI_FRANCE, CI_GERMANY, CI_US_AVG, CI_TEXAS, CI_INDIA, CI_POLAND]

        # C_operational = E_total * CI_grid * PUE (convert g to tonnes: / 1e6)
        _carbons_t = [_energy * ci * _pue / 1e6 for ci in _cis]

        _min_c = min(_carbons_t)
        _max_c = max(_carbons_t)
        _ratio = _max_c / _min_c if _min_c > 0 else 0

        _bar_colors = [COLORS["GreenLine"] if c < _max_c * 0.15 else
                       COLORS["BlueLine"] if c < _max_c * 0.4 else
                       COLORS["OrangeLine"] if c < _max_c * 0.7 else
                       COLORS["RedLine"] for c in _carbons_t]

        fig_geo = go.Figure()
        fig_geo.add_trace(go.Bar(
            x=_regions, y=_carbons_t,
            marker_color=_bar_colors,
            text=[f"{c:,.0f}t" for c in _carbons_t],
            textposition="outside",
        ))
        fig_geo.update_layout(
            height=380,
            yaxis=dict(title="CO2 Emissions (tonnes)"),
            xaxis=dict(title="Region"),
        )
        apply_plotly_theme(fig_geo)

        items.append(mo.md(f"### Carbon Emissions by Region ({_energy:,} MWh, PUE = {_pue:.2f})"))
        items.append(mo.as_html(fig_geo))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 12px;">
            <div style="padding: 14px 20px; border: 2px solid {COLORS['GreenLine']}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Lowest (Quebec)</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['GreenLine']};">
                    {_carbons_t[0]:,.0f} t</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {COLORS['RedLine']}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Highest (Poland)</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['RedLine']};">
                    {_carbons_t[-1]:,.0f} t</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {COLORS['OrangeLine']}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Ratio</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['OrangeLine']};">
                    {_ratio:.0f}x</div>
            </div>
        </div>
        """))

        # Prediction reveal
        _correct = partB_pred.value == "40"
        _msg = ("Correct. The Quebec-to-Poland ratio is ~41x. A single site selection decision "
                "achieves ~25% of the total savings from the entire algorithmic toolkit."
                if _correct else
                "The ratio is ~41x. Students anchor on algorithmic speedup scales (2-5x) and do not "
                "realize that grid carbon varies by more than an order of magnitude.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        items.append(mo.accordion({
            "Math Peek: Operational Carbon Equation": mo.md("""
**Operational carbon:**
$$
C_{\\text{op}} = E \\times \\text{CI} \\times \\text{PUE}
$$

**Where:**
- **$E$**: Energy consumed (kWh) = Power (kW) $\\times$ Time (h)
- **$\\text{CI}$**: Carbon intensity of the grid (gCO$_2$/kWh)
- **$\\text{PUE}$**: Power Usage Effectiveness (total facility power / IT power)

**Geographic lever:**
$$
\\frac{C_{\\text{Poland}}}{C_{\\text{Quebec}}} = \\frac{820}{20} = 41\\times
$$

A single site selection decision provides a 41x carbon reduction --
larger than most algorithmic optimizations combined.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART C BUILDER -- The Lifecycle Carbon Shift
    # ─────────────────────────────────────────────────────────────────────

    def build_part_c():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; CFO, Green Compute Inc.
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;We moved to Quebec. Operational carbon is near zero. Problem solved?
                Not quite. When you reduce operational carbon, a different term dominates:
                the carbon cost of manufacturing the hardware itself.&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Lifecycle Carbon Shift

        In Quebec, operational carbon is low. Problem solved? Not quite. When you reduce
        operational carbon, a different term dominates: the carbon cost of manufacturing
        the hardware itself. An H100 has 150-200 kg CO2 embodied.
        """))

        items.append(partC_pred)

        if partC_pred.value is None:
            items.append(mo.callout(
                mo.md("**Select your prediction to unlock.**"),
                kind="warn",
            ))
            return mo.vstack(items)

        # Controls
        items.append(mo.hstack([partC_refresh_slider, partC_util_slider, partC_gpu_count], justify="start", gap=1))

        _refresh = partC_refresh_slider.value
        _util = partC_util_slider.value / 100
        _gpus = partC_gpu_count.value

        # Embodied carbon over lifecycle
        _embodied_total = _gpus * H100_EMBODIED_KG / 1000  # tonnes CO2
        _embodied_annual = _embodied_total / _refresh  # tonnes/year

        # Operational carbon per year
        # Energy = GPUs * TDP * utilization * hours/year / 1e6 (to MWh)
        _energy_mwh_year = _gpus * H100_TDP_W * _util * 8760 / 1e6

        # Two scenarios
        _ops_coal = _energy_mwh_year * CI_POLAND * PUE_LIQUID / 1e6  # tonnes/year
        _ops_hydro = _energy_mwh_year * CI_QUEBEC * PUE_LIQUID / 1e6  # tonnes/year

        _total_coal = _ops_coal + _embodied_annual
        _total_hydro = _ops_hydro + _embodied_annual

        _frac_embodied_coal = _embodied_annual / _total_coal * 100 if _total_coal > 0 else 0
        _frac_embodied_hydro = _embodied_annual / _total_hydro * 100 if _total_hydro > 0 else 0

        fig_life = go.Figure()
        fig_life.add_trace(go.Bar(
            name="Operational Carbon", x=["Coal Grid\n(Poland)", "Hydro Grid\n(Quebec)"],
            y=[_ops_coal, _ops_hydro],
            marker_color=COLORS["OrangeLine"],
            text=[f"{_ops_coal:,.0f}t", f"{_ops_hydro:,.0f}t"], textposition="inside",
        ))
        fig_life.add_trace(go.Bar(
            name="Embodied Carbon", x=["Coal Grid\n(Poland)", "Hydro Grid\n(Quebec)"],
            y=[_embodied_annual, _embodied_annual],
            marker_color=COLORS["BlueLine"],
            text=[f"{_embodied_annual:,.0f}t", f"{_embodied_annual:,.0f}t"], textposition="inside",
        ))
        fig_life.update_layout(
            barmode="stack", height=380,
            yaxis=dict(title="Annual Carbon (tonnes CO2)"),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_life)

        items.append(mo.md(f"### Lifecycle Carbon ({_gpus:,} GPUs, {_refresh}-year refresh, {_util*100:.0f}% util)"))
        items.append(mo.as_html(fig_life))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 12px;">
            <div style="padding: 14px 20px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        text-align: center; min-width: 180px; background: {COLORS['OrangeLL']};">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Coal Grid: Embodied Fraction</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['OrangeLine']};">
                    {_frac_embodied_coal:.0f}%</div>
            </div>
            <div style="padding: 14px 20px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        text-align: center; min-width: 180px; background: {COLORS['BlueLL']};">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Hydro Grid: Embodied Fraction</div>
                <div style="font-size: 1.6rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_frac_embodied_hydro:.0f}%</div>
            </div>
        </div>
        """))
        items.append(mo.callout(mo.md(
            f"On the hydro grid, embodied carbon represents **{_frac_embodied_hydro:.0f}%** of total lifecycle emissions. "
            "The most effective carbon intervention in a clean-grid datacenter is keeping hardware running longer at higher utilization."
        ), kind="info"))

        # Prediction reveal
        _correct = partC_pred.value == "40"
        _msg = ("Correct. On a clean grid, embodied carbon can represent 30-50%+ of total lifecycle emissions. "
                "'Green energy = zero carbon' is a myth -- you still pay the manufacturing cost."
                if _correct else
                "On a 100% renewable grid, embodied carbon represents 30-50%+ of total lifecycle emissions. "
                "Students assume 'green energy = zero carbon' and forget the physical carbon cost of fabricating silicon.")
        items.append(mo.callout(mo.md(f"**{_msg}**"), kind="success" if _correct else "warn"))

        items.append(mo.accordion({
            "Math Peek: Lifecycle Carbon (Embodied + Operational)": mo.md("""
**Total lifecycle carbon:**
$$
C_{\\text{total}} = C_{\\text{embodied}} + C_{\\text{operational}}
$$

**Embodied carbon per GPU-year:**
$$
C_{\\text{embodied}} = \\frac{C_{\\text{mfg}}}{L_{\\text{refresh}}} \\times N_{\\text{GPUs}}
$$

**Where:**
- **$C_{\\text{mfg}}$**: Manufacturing carbon per GPU (~175 kg CO$_2$eq for H100)
- **$L_{\\text{refresh}}$**: Hardware refresh cycle (years)
- **$N_{\\text{GPUs}}$**: Number of GPUs in the fleet

**On clean grids** (CI < 50 gCO$_2$/kWh), embodied carbon can exceed 30-50%
of total lifecycle emissions. "Green energy = zero carbon" is a myth.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART D BUILDER -- The Jevons Trap
    # ─────────────────────────────────────────────────────────────────────

    def build_part_d():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; Chief Sustainability Officer
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;You have optimized where you compute and how long you keep hardware.
                Your efficiency has doubled. But total energy consumption just went
                <em>up</em>. Welcome to the Jevons Paradox.&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## The Jevons Trap

        You have optimized where you compute (geography) and how long you keep hardware
        (lifecycle). Your efficiency has doubled. But total energy consumption just went
        *up*. Welcome to the Jevons Paradox -- the most counterintuitive
        result in sustainability economics, and arguably the most important insight
        in this entire two-volume curriculum.
        """))

        # Jevons explanation callout
        items.append(mo.Html(f"""
        <div style="background: linear-gradient(135deg, {COLORS['OrangeLL']} 0%, #fff7ed 100%);
                    border: 2px solid {COLORS['OrangeLine']}; border-radius: 12px;
                    padding: 24px 28px; margin: 12px 0;">
            <div style="font-size: 0.75rem; font-weight: 700; color: {COLORS['OrangeLine']};
                        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">
                The Jevons Paradox Applied to AI
            </div>
            <div style="font-size: 0.95rem; color: {COLORS['Text']}; line-height: 1.75;">
                <strong>William Stanley Jevons observed in 1865</strong> that James Watt's more efficient
                steam engine did not reduce coal consumption &mdash; it increased it, because efficiency
                made steam power cheaper, which expanded its use.<br/><br/>
                <strong>The AI version:</strong> Making inference more efficient reduces cost-per-query,
                which stimulates demand. If demand elasticity &gt; 1, total energy consumption
                <em>increases</em> despite per-unit efficiency gains.<br/><br/>
                <strong>The formula:</strong><br/>
                <code style="font-size: 1.1rem; background: white; padding: 6px 12px; border-radius: 6px;
                             border: 1px solid {COLORS['Border']}; display: inline-block; margin-top: 4px;">
                    E_total = (E_baseline / Efficiency) &times; V_baseline &times; Efficiency<sup>Elasticity</sup>
                </code>
            </div>
        </div>
        """))

        items.append(partD_pred)
        items.append(mo.md("*Enter a percentage: negative for decrease, positive for increase.*"))

        if partD_pred.value is None:
            items.append(mo.callout(
                mo.md("**Enter your prediction to unlock the Jevons simulator.**"),
                kind="warn",
            ))
            return mo.vstack(items)

        # Controls
        items.append(mo.hstack([partD_eff_slider, partD_elast_slider], justify="start", gap=1))
        items.append(mo.hstack([partD_cap_toggle, partD_cap_level], justify="start", gap=1))

        _eff = partD_eff_slider.value
        _elast = partD_elast_slider.value
        _cap_on = partD_cap_toggle.value
        _cap = partD_cap_level.value

        # E_total = (E_baseline / eff) * V_baseline * eff^elasticity
        # Normalize: E_baseline = 1, V_baseline = 1
        _eff_range = np.linspace(1, 10, 100)

        # Three demand elasticity curves
        _elasticities = [0.3, 1.0, _elast]
        _labels = ["Inelastic (0.3)", "Unit-elastic (1.0)", f"Current ({_elast:.1f})"]
        _colors = [COLORS["GreenLine"], COLORS["BlueLine"], COLORS["OrangeLine"]]

        fig_jevons = go.Figure()
        for _e, _lab, _col in zip(_elasticities, _labels, _colors):
            _e_total = (1.0 / _eff_range) * _eff_range ** _e
            if _cap_on:
                _e_total = np.minimum(_e_total, _cap)
            fig_jevons.add_trace(go.Scatter(
                x=_eff_range, y=_e_total, name=_lab,
                line=dict(color=_col, width=3 if _e == _elast else 2,
                          dash="solid" if _e == _elast else "dot"),
            ))

        # Baseline reference
        fig_jevons.add_hline(y=1.0, line_dash="dash", line_color=COLORS["TextMuted"],
                             annotation_text="Baseline energy", annotation_position="top right")

        if _cap_on:
            fig_jevons.add_hline(y=_cap, line_dash="solid", line_color=COLORS["RedLine"], line_width=2,
                                 annotation_text=f"Carbon cap ({_cap:.1f}x)", annotation_position="top left")

        # Current operating point
        _e_current = (1.0 / _eff) * _eff ** _elast
        if _cap_on:
            _e_current = min(_e_current, _cap)
        _pct_change = (_e_current - 1.0) * 100

        fig_jevons.add_trace(go.Scatter(
            x=[_eff], y=[_e_current], mode="markers",
            marker=dict(size=16, color=COLORS["RedLine"] if _e_current > 1 else COLORS["GreenLine"],
                        line=dict(color="white", width=2)),
            name="Your config",
        ))

        fig_jevons.update_layout(
            height=380,
            xaxis=dict(title="Efficiency Improvement (x)"),
            yaxis=dict(title="Total Energy (fraction of baseline)", range=[0, max(3, _e_current * 1.3)]),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        apply_plotly_theme(fig_jevons)

        # Failure state
        _rebound = _e_current > 1.5
        _banner = ""
        if _rebound and not _cap_on:
            _banner = f"""<div style="background: {COLORS['RedLL']}; border: 2px solid {COLORS['RedLine']};
                          border-radius: 8px; padding: 14px; text-align: center; margin-bottom: 12px;
                          font-weight: 700; color: {COLORS['RedLine']}; font-size: 1.1rem;">
                          JEVONS REBOUND: Total energy is {_pct_change:+.0f}% of baseline</div>"""

        _pct_color = COLORS["RedLine"] if _pct_change > 0 else COLORS["GreenLine"]
        _direction = "INCREASE" if _pct_change > 0 else "DECREASE"

        if _banner:
            items.append(mo.Html(_banner))
        items.append(mo.md(f"### Jevons Paradox (Efficiency = {_eff:.1f}x, Elasticity = {_elast:.1f})"))
        items.append(mo.as_html(fig_jevons))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 20px; justify-content: center; margin-top: 16px; flex-wrap: wrap;">
            <div style="padding: 16px 24px; border: 2px solid {_pct_color}; border-radius: 12px;
                        text-align: center; min-width: 200px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Net Energy Change</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {_pct_color};">
                    {_pct_change:+.0f}%</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: {_pct_color};">{_direction}</div>
            </div>
            <div style="padding: 16px 24px; border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        text-align: center; min-width: 200px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Per-Query Cost</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {COLORS['GreenLine']};">
                    {1/_eff:.1%}</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']};">of baseline</div>
            </div>
            <div style="padding: 16px 24px; border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        text-align: center; min-width: 200px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Induced Demand</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {COLORS['OrangeLine']};">
                    {_eff ** _elast:.1f}x</div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']};">volume increase</div>
            </div>
        </div>
        """))

        # Prediction reveal
        _predicted = partD_pred.value
        _actual = 50  # 2x efficiency, 3x demand = 0.5 * 3 = 1.5x = +50%

        _gap = abs(_predicted - _actual)
        if _gap < 10:
            _msg = f"Excellent. You predicted {_predicted:+.0f}%. The actual change is +50%. You grasped the counterintuitive result."
            _kind = "success"
        elif _predicted < 0:
            _msg = (f"You predicted {_predicted:+.0f}% (a decrease). The actual is +50% (an increase). "
                    "Most students predict savings because they compute 'half the energy per query' but "
                    "forget to multiply by the 3x demand response. E = 0.5 * 3 = 1.5x baseline = +50%.")
            _kind = "danger"
        else:
            _msg = f"You predicted {_predicted:+.0f}%. The actual is +50%. The Jevons formula: E = (1/2) * 1 * 2^2.0 = 0.5 * 4 = 2.0? No -- with demand 3x, E = 0.5 * 3 = 1.5x."
            _kind = "warn"

        items.append(mo.callout(mo.md(f"**{_msg}**"), kind=_kind))
        items.append(mo.accordion({
            "Math Peek: The Jevons Equation": mo.md("""
**The full Jevons formula for AI energy:**

```
E_total = (E_baseline / Efficiency) x V_baseline x Efficiency^Elasticity
```

At Efficiency = 2x, Elasticity = 2.0:

```
E_total = (1 / 2) x 1 x 2^2.0 = 0.5 x 4 = 2.0x baseline  (+100% increase!)
```

The efficiency halves per-query cost, but demand quadruples. Net result: total energy doubles.

**The only escape:** elasticity < 1.0 (inelastic demand) OR an absolute carbon cap.
When demand is elastic (empirically estimated at 1.5-3.0 for AI inference), efficiency
without caps guarantees increased consumption.

**Empirical evidence:** OpenAI API pricing dropped ~10x from GPT-3 to GPT-3.5-turbo
(2022-2023), while API call volume increased ~50-100x. Inference demand is firmly elastic.
"""),
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # PART E BUILDER -- Carbon-Aware Fleet Design
    # ─────────────────────────────────────────────────────────────────────

    def build_part_e():
        items = []

        items.append(mo.Html(f"""
        <div style="border-left:4px solid {COLORS['TextMuted']}; background:{COLORS['Surface2']};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{COLORS['TextMuted']};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message &middot; VP of Fleet Operations
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                &ldquo;Efficiency alone is not enough. You need a fleet-level strategy that combines
                geography, scheduling, and hard caps. Can you achieve a 50% emission reduction
                without exceeding a 48-hour project delay?&rdquo;
            </div>
        </div>
        """))

        items.append(mo.md("""
        ## Carbon-Aware Fleet Design

        Efficiency alone is not enough. You need a fleet-level strategy that combines
        geography, scheduling, and hard caps. Can you achieve a 50% emission reduction
        without exceeding a 48-hour project delay?
        """))

        # Controls
        items.append(mo.md("### Configure Your Carbon Strategy"))
        items.append(mo.hstack([partE_geo, partE_temporal], justify="start", gap=1))
        items.append(mo.hstack([partE_eff_gain, partE_cap], justify="start", gap=1))

        _ci = partE_geo.value
        _temporal_pct = partE_temporal.value / 100
        _eff = partE_eff_gain.value
        _cap = partE_cap.value

        # Baseline: US Average, no temporal shift, no efficiency, no cap
        _baseline_ci = CI_US_AVG
        _baseline_energy = 10000  # MWh (normalized training run)
        _baseline_carbon = _baseline_energy * _baseline_ci * PUE_AIR / 1e6  # tonnes

        # Geographic savings
        _geo_carbon = _baseline_energy * _ci * PUE_AIR / 1e6
        _geo_savings = 1 - (_geo_carbon / _baseline_carbon)

        # Temporal shift: off-peak hours have ~40% lower CI on average
        _temporal_savings = _temporal_pct * 0.4
        _after_temporal = _geo_carbon * (1 - _temporal_savings)

        # Efficiency: reduces energy but may trigger Jevons (assume elasticity 1.5 for fleet)
        _eff_energy = _baseline_energy / _eff
        _demand_response = _eff ** 1.5  # elastic demand
        _actual_energy = _eff_energy * _demand_response
        _after_eff = _actual_energy * _ci * PUE_AIR / 1e6 * (1 - _temporal_savings)

        # Cap
        _cap_carbon = _baseline_carbon * _cap
        _final_carbon = min(_after_eff, _cap_carbon)

        _total_reduction = 1 - (_final_carbon / _baseline_carbon)
        _target_met = _total_reduction >= 0.5

        # Delay estimate (geographic shift adds 0-24h, temporal shift adds 0-24h)
        _delay_hours = 0
        if _ci < 100:
            _delay_hours += 12  # cross-continent latency
        _delay_hours += _temporal_pct * 24  # waiting for off-peak
        _delay_ok = _delay_hours <= 48

        _reduction_color = COLORS["GreenLine"] if _target_met else COLORS["RedLine"]
        _delay_color = COLORS["GreenLine"] if _delay_ok else COLORS["RedLine"]

        _target_banner = ""
        if _target_met and _delay_ok:
            _target_banner = f"""<div style="background: {COLORS['GreenLL']}; border: 2px solid {COLORS['GreenLine']};
                                border-radius: 8px; padding: 14px; text-align: center; margin-bottom: 12px;
                                font-weight: 700; color: {COLORS['GreenLine']}; font-size: 1.0rem;">
                                TARGET MET: {_total_reduction:.0%} reduction within {_delay_hours:.0f}h delay</div>"""
        elif not _target_met:
            _target_banner = f"""<div style="background: {COLORS['RedLL']}; border: 2px solid {COLORS['RedLine']};
                                border-radius: 8px; padding: 14px; text-align: center; margin-bottom: 12px;
                                font-weight: 700; color: {COLORS['RedLine']}; font-size: 1.0rem;">
                                TARGET MISSED: Only {_total_reduction:.0%} reduction (need 50%)</div>"""

        if _target_banner:
            items.append(mo.Html(_target_banner))
        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; justify-content: center; margin-top: 12px; flex-wrap: wrap;">
            <div style="padding: 14px 20px; border: 2px solid {_reduction_color}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Carbon Reduction</div>
                <div style="font-size: 2rem; font-weight: 900; color: {_reduction_color};">
                    {_total_reduction:.0%}</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">target: 50%</div>
            </div>
            <div style="padding: 14px 20px; border: 2px solid {_delay_color}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Project Delay</div>
                <div style="font-size: 2rem; font-weight: 900; color: {_delay_color};">
                    {_delay_hours:.0f}h</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">max: 48h</div>
            </div>
            <div style="padding: 14px 20px; border: 1px solid {COLORS['Border']}; border-radius: 10px;
                        text-align: center; min-width: 160px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase;">Final Carbon</div>
                <div style="font-size: 2rem; font-weight: 900; color: {COLORS['BlueLine']};">
                    {_final_carbon:,.0f}t</div>
                <div style="font-size: 0.72rem; color: {COLORS['TextSec']};">baseline: {_baseline_carbon:,.0f}t</div>
            </div>
        </div>
        """))
        items.append(mo.callout(mo.md(
            "**Strategy insight:** Efficiency alone fails (Jevons, from Part D). Geographic shift alone "
            "may add unacceptable latency. Temporal scheduling alone misses the target if the grid has "
            "no clean windows. Only the combination of geographic shift + temporal scheduling + carbon cap "
            "reliably hits 50% reduction."
        ), kind="info"))

        items.append(mo.accordion({
            "Math Peek: Carbon-Aware Fleet Optimization": mo.md("""
**Fleet carbon with all levers:**
$$
C_{\\text{fleet}} = \\sum_{r \\in \\text{regions}} N_r \\times P_r \\times \\text{CI}_r \\times \\text{PUE}_r \\times t_r
+ C_{\\text{embodied}}
$$

**Carbon cap constraint:**
$$
C_{\\text{fleet}} \\leq \\alpha \\times C_{\\text{baseline}}, \\quad \\alpha \\in (0, 1]
$$

**Temporal scheduling savings:**
$$
C_{\\text{temporal}} = C_{\\text{baseline}} \\times \\left(1 - f_{\\text{shift}} \\times \\frac{\\text{CI}_{\\text{peak}} - \\text{CI}_{\\text{off}}}{\\text{CI}_{\\text{peak}}}\\right)
$$

- **$f_{\\text{shift}}$**: Fraction of jobs shifted to off-peak
- **$\\alpha$**: Carbon cap as fraction of baseline

Only the combination of geographic + temporal + absolute cap reliably
achieves 50%+ reduction targets.
""")
        }))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # SYNTHESIS BUILDER
    # ─────────────────────────────────────────────────────────────────────

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
                    <strong>1. The energy wall is real and growing.</strong>
                    AI compute demand has outpaced hardware efficiency by ~195,000&times; over 7 years.
                    You cannot outrun this deficit with better chips alone.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Geography is the single highest-leverage intervention.</strong>
                    Grid carbon intensity varies 40&times; (Quebec: 20 vs Poland: 820 gCO2/kWh).
                    On clean grids, embodied carbon (hardware manufacturing) dominates at 30-50%+.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>3. The Jevons Paradox means efficiency is necessary but not sufficient.</strong>
                    At demand elasticity &gt;1 (empirically 1.5-3.0 for AI inference), efficiency gains
                    stimulate demand that overwhelms the savings. Only absolute carbon caps guarantee
                    net reduction. Sustainable AI requires governance, not just better engineering.
                </div>
            </div>
        </div>
        """))

        items.append(mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 15: The Fairness Budget</strong> -- Carbon caps limit compute.
                    Fairness monitoring consumes latency. The next lab reveals that mathematical
                    impossibility governs fairness metrics, and responsible AI infrastructure
                    has real system costs.
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
                    <strong>Read:</strong> Sustainable AI chapter for the full Jevons derivation
                    and carbon-aware scheduling analysis.<br/>
                    <strong>Feeds into:</strong> V2-16 Capstone (carbon cap as fleet constraint).
                </div>
            </div>
        </div>
        """))

        return mo.vstack(items)

    # ─────────────────────────────────────────────────────────────────────
    # COMPOSE TABS
    # ─────────────────────────────────────────────────────────────────────

    tabs = mo.ui.tabs({
        "Part A -- The Energy Wall":            build_part_a(),
        "Part B -- The Geography of Carbon":    build_part_b(),
        "Part C -- The Lifecycle Carbon Shift":  build_part_c(),
        "Part D -- The Jevons Trap":             build_part_d(),
        "Part E -- Carbon-Aware Fleet Design":   build_part_e(),
        "Synthesis":                             build_synthesis(),
    })
    tabs
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════


# ─── CELL 10: LEDGER HUD ─────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, ledger, COLORS, partA_pred, partB_pred, partC_pred, partD_pred, partD_elast_slider, partE_geo, partE_cap):
    _energy_pred = partA_pred.value if hasattr(partA_pred, 'value') else None
    _geo_pred = partB_pred.value if hasattr(partB_pred, 'value') else None
    _lifecycle_pred = partC_pred.value if hasattr(partC_pred, 'value') else None
    _jevons_pred = partD_pred.value if hasattr(partD_pred, 'value') else None
    _elasticity = partD_elast_slider.value if hasattr(partD_elast_slider, 'value') else 2.0
    _geo_choice = partE_geo.value if hasattr(partE_geo, 'value') else "US average"
    _cap_choice = partE_cap.value if hasattr(partE_cap, 'value') else 1.0
    ledger.save(chapter=14, design={
        "partA_energy_deficit_prediction": _energy_pred,
        "partB_geography_prediction": _geo_pred,
        "partC_lifecycle_prediction": _lifecycle_pred,
        "partD_jevons_prediction_pct": _jevons_pred,
        "partD_elasticity_explored": _elasticity,
        "partE_geographic_choice": _geo_choice,
        "partE_carbon_cap": _cap_choice,
    })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">V2-14: The Carbon Budget</span>
        <span class="hud-label">LEDGER</span>
        <span class="hud-active">Saved (ch14)</span>
        <span class="hud-label">NEXT</span>
        <span class="hud-value">V2-15: The Fairness Budget</span>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
