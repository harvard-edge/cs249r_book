import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    from pathlib import Path
    import plotly.graph_objects as go

    # Robust path finding: find the repo root relative to this file
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from mlsysim.sim import Personas, ResourceSimulation
    from mlsysim import Applications, Fleet, viz
    from mlsysim.viz import dashboard
    return Applications, Fleet, Personas, ResourceSimulation, dashboard, mo


@app.cell
def _(mo):
    mo.md("""
    # 🧪 Lab 15: The Global Sustainability Challenge
    ### ML Systems Infrastructure & Modeling

    Welcome, Architect. This dashboard is your **Cockpit** for managing the global footprint of an ML fleet. Follow the phases below to navigate the **Energy Wall**.

    ---
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 🏁 Phase 1: Establish Your Persona
    Every system architect faces different "Physical Walls." A **Cloud Titan** cares about megawatts and TCO, while a **Tiny Pioneer** cares about microwatts and battery life.

    **Your Task:** Select a persona below and observe how the **Scale** and **Constraint Badges** in the header update.
    """)
    return


@app.cell
def _(mo):
    persona_selector = mo.ui.dropdown(
        options={
            "Cloud Titan": "cloud",
            "Edge Guardian": "edge",
            "Mobile Nomad": "mobile",
            "Tiny Pioneer": "tiny",
        },
        value="Cloud Titan",
        label="Assign Persona Role",
    )
    return (persona_selector,)


@app.cell
def _(mo):
    mo.md("""
    ### 🎛️ Phase 2: Navigating the Energy Wall
    A "Solar Flare" event has hit the Nevada grid, forcing you to rely on coal-heavy power. You must optimize your fleet's carbon footprint without breaking your budget or delaying the project.

    **The Trade-off:**
    1. **Spatial:** Move data to a green region (Quebec) but pay the **Data Gravity** tax (transfer time).
    2. **Infrastructure:** Switch to Liquid Cooling (lower PUE) but pay the **CapEx** tax (higher TCO).
    """)
    return


@app.cell
def _(mo):
    region_selector = mo.ui.dropdown(
        options={
            "Nevada (US Avg)": "US_Avg",
            "Quebec (Hydro)": "Quebec",
            "Poland (Coal)": "Poland",
        },
        value="Nevada (US Avg)",
        label="Target Deployment Region",
    )

    duration_slider = mo.ui.slider(
        start=1, stop=365, step=1, value=30, label="Simulation Duration (Days)"
    )

    cooling_radio = mo.ui.radio(
        options={"Standard Air (PUE 1.5)": "air", "Direct Liquid (PUE 1.1)": "liquid"},
        value="Standard Air (PUE 1.5)",
        label="Infrastructure Cooling",
    )
    return cooling_radio, duration_slider, region_selector


@app.cell
def _(
    Applications,
    Fleet,
    Personas,
    ResourceSimulation,
    cooling_radio,
    dashboard,
    duration_slider,
    mo,
    persona_selector,
    region_selector,
):
    # 1. Run Physics Engine
    _persona_key = persona_selector.value
    _persona = Personas.get(_persona_key)
    _scenario_map = {
        "cloud": Fleet.Frontier,
        "edge": Applications.AutoDrive,
        "mobile": Applications.Assistant,
        "tiny": Applications.Doorbell,
    }
    _scenario = _scenario_map.get(_persona_key, Fleet.Frontier)

    _sim = ResourceSimulation(_scenario, _persona)
    _ledger = _sim.evaluate(
        {
            "region": region_selector.value,
            "duration_days": duration_slider.value,
            "cooling": cooling_radio.value,
        }
    )

    # 2. Evaluate Hard Constraints
    _max_carbon_kg = 5_000_000 if _persona_key == "cloud" else 500_000
    _is_carbon_met = _ledger.sustainability.carbon_kg <= _max_carbon_kg
    _is_budget_met = _ledger.economics.tco <= 50_000_000

    _constraints = {
        f"Carbon < {_max_carbon_kg/1_000_000:.1f}k tonnes": _is_carbon_met,
        "TCO < $50M": _is_budget_met,
    }

    # 3. Build Zone 1 (Header)
    header = dashboard.command_header(
        title="Lab 15: Sustainable AI",
        subtitle="The Grid-Interactive Scheduler (Solar Flare Crisis)",
        persona_name=_persona.name,
        scale=f"{_persona.scale_factor:,.0f} {_persona.unit_of_scale}",
        constraints=_constraints,
    )

    # 4. Build Zone 3 (Telemetry)
    scorecard = dashboard.telemetry_scorecard(_ledger)

    # Generate Pareto Data (Cost vs Carbon) for the plot
    _pareto_x = [
        _ledger.economics.tco * 0.8,
        _ledger.economics.tco * 0.9,
        _ledger.economics.tco * 1.5,
    ]
    _pareto_y = [
        _ledger.sustainability.carbon_kg * 1.5,
        _ledger.sustainability.carbon_kg * 1.0,
        _ledger.sustainability.carbon_kg * 0.5,
    ]

    plot = dashboard.pareto_plot(
        x_val=_ledger.economics.tco,
        y_val=_ledger.sustainability.carbon_kg,
        x_label="Total Cost of Ownership ($)",
        y_label="Carbon Footprint (kg CO2e)",
        title="The Sustainability Pareto Frontier",
        pareto_x=_pareto_x,
        pareto_y=_pareto_y,
    )

    telemetry_ui = mo.vstack([scorecard, mo.as_html(plot)])
    return (telemetry_ui,)


@app.cell
def _(cooling_radio, duration_slider, mo, persona_selector, region_selector):
    levers_ui = mo.vstack(
        [persona_selector, region_selector, cooling_radio, duration_slider]
    )
    return (levers_ui,)


@app.cell
def _(mo):
    audit_log = mo.md(
        """
        **Mission Log:**
        * *08:00 UTC:* Solar flare impacts Nevada grid. Renewable mix drops to 10%.
        * *08:15 UTC:* System detects SLA at risk. Architect intervention required.
        """
    )

    justification_box = mo.ui.text_area(
        placeholder="Explain your trade-off. Did you prioritize TCO or Carbon? Why?",
        label="Architectural Justification (Required for Deployment)",
    )

    audit_ui = mo.vstack(
        [
            mo.md(
                "<div style='background: #fff5f5; padding: 10px; border-left: 4px solid #e53e3e; margin-bottom: 15px;'><b>CRISIS ACTIVE:</b> You must justify your architectural changes before the grid fails.</div>"
            ),
            audit_log,
            justification_box,
        ]
    )
    return (audit_ui,)


@app.cell
def _(audit_ui, levers_ui, mo, telemetry_ui):
    # Use the dashboard layout helper
    _cockpit = mo.vstack([
        mo.hstack([
            # ZONE 2: Engineering Levers (30% width)
            mo.vstack([
                mo.md("#### **Levers**"),
                mo.md("<div style='background: #f7fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;'>"),
                levers_ui,
                mo.md("</div>")
            ]),
            # ZONE 3: Telemetry Center (70% width)
            mo.vstack([
                mo.md("#### **Telemetry**"),
                telemetry_ui
            ])
        ], widths=[3, 7], gap=2),
        mo.md("---"),
        # ZONE 4: Audit Trail
        mo.md("### 📝 Phase 3: Architectural Justification"),
        audit_ui
    ], gap=1)
    return


if __name__ == "__main__":
    app.run()
