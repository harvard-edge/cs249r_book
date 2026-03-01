import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    from pathlib import Path

    # Robust path finding: find the repo root relative to this file
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from mlsysim.sim import Personas
    from mlsysim.viz import dashboard
    return Personas, dashboard, mo


@app.cell
def _():
    # ZONE 0: THE HOOK
    return


@app.cell
def _(mo):
    # ZONE 1: THE IDENTITY
    persona_selector = mo.ui.dropdown(
        options={
            "Cloud Titan": "cloud",
            "Edge Guardian": "edge",
            "Mobile Nomad": "mobile",
            "Tiny Pioneer": "tiny"
        },
        value="Cloud Titan",
        label="🚀 STEP 1: SELECT YOUR CAREER TRACK"
    )
    return (persona_selector,)


@app.cell
def _(Personas, persona_selector):
    # BRAIN: STORY & PHYSICS
    _key = persona_selector.value
    persona = Personas.get(_key)

    _stories = {
        "cloud": {
            "title": "The Exaflop Factory",
            "img": "🏭",
            "narrative": "You are the infrastructure lead for a global LLM. You manage 100,000 GPUs. Your enemies are <b>Electricity Bills</b> and <b>Grid Stability</b>. A 1% increase in MFU saves enough energy to power a small city.",
            "wall": "The Power Wall",
            "expect": "Vol 2: Fleet Orchestration"
        },
        "edge": {
            "title": "The Zero-Collision Loop",
            "img": "🏎️",
            "narrative": "You are the safety engineer for an Autonomous Vehicle fleet. Your model must detect a pedestrian and brake within 10ms. If the network jitters, the car stops. Safety is your only constraint.",
            "wall": "The Light Barrier",
            "expect": "Vol 2: Deterministic Latency"
        },
        "mobile": {
            "title": "The Palm-Sized Assistant",
            "img": "📱",
            "narrative": "Your vision app runs on 100 million smartphones. If you use too much NPU power, the phone gets hot and throttles. You must fit 'Intelligence' into a literal thermal pocket.",
            "wall": "The Thermal Wall",
            "expect": "Vol 1: Model Compression"
        },
        "tiny": {
            "title": "The Smart Doorbell",
            "img": "🔔",
            "narrative": "You are hacking on a $2 microcontroller with 256KB of memory. It runs on a battery for a year. You are fighting for every single byte and microwatt.",
            "wall": "The Memory Wall",
            "expect": "Vol 1: TinyML Physics"
        }
    }

    story = _stories.get(_key)
    _budgets = {"cloud": 95, "edge": 40, "mobile": 25, "tiny": 8}
    budget = _budgets.get(_key)
    return budget, persona, story


@app.cell
def _(mo, persona, story):
    # ZONE 2: THE BRIEFING
    story_card = mo.md(
        f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin: 20px 0;">
            <div style="background: #f7fafc; padding: 25px; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: 20px;">
                <span style="font-size: 4em;">{story['img']}</span>
                <div>
                    <h2 style="margin: 0; color: #2d3748; font-size: 1.8em;">{story['title']}</h2>
                    <p style="margin: 0; color: #3182ce; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Current Persona: {persona.name}</p>
                </div>
            </div>
            <div style="padding: 25px;">
                <p style="font-size: 1.2em; line-height: 1.6; color: #4a5568;">{story['narrative']}</p>
                <div style="margin-top: 20px; display: flex; gap: 10px;">
                    <span style="background: #fef2f2; color: #9b1c1c; padding: 6px 12px; border-radius: 6px; font-weight: bold;">⚠️ Primary Barrier: {story['wall']}</span>
                    <span style="background: #eff6ff; color: #1e40af; padding: 6px 12px; border-radius: 6px; font-weight: bold;">🎯 Track Focus: {story['expect']}</span>
                </div>
            </div>
        </div>
        """
    )
    return (story_card,)


@app.cell
def _(mo):
    complexity_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=80, 
        label="2. ADAPT THE ARCHITECTURE: Set Model Size (%)"
    )
    return (complexity_slider,)


@app.cell
def _(budget, complexity_slider, dashboard, persona, story):
    # CALCULATE SUCCESS
    is_safe = complexity_slider.value <= budget

    header = dashboard.command_header(
        title=f"Command Center: {persona.name}",
        subtitle=f"Mission: {story['title']} at scale {persona.scale_factor:,.0f} {persona.unit_of_scale}",
        persona_name=persona.role,
        scale=f"{persona.scale_factor:,.0f} {persona.unit_of_scale}",
        constraints={"Physics Compliance": is_safe, "Deployment Ready": is_safe}
    )
    return (is_safe,)


@app.cell
def _(
    complexity_slider,
    dashboard,
    is_safe,
    mo,
    persona,
    persona_selector,
    story_card,
):
    # TAB 1: BRIEFING
    briefing_content = mo.vstack([
        mo.md("### 🏁 Phase 1: Identity & Context"),
        persona_selector,
        story_card,
        dashboard.pro_note(
            "Architect Selection",
            "Choose your path. Each role below represents a different scale of the AI Triad."
        )
    ])

    # TAB 2: SIMULATION
    simulation_content = mo.vstack([
        mo.md("### 🎛️ Phase 2: The Balancing Act"),
        dashboard.pro_note(
            "The Crisis at Hand",
            f"Your current model complexity is at <b>{complexity_slider.value}%</b>. " +
            (f"The Physics Compliance badge is red! You are violating the laws of the <b>{persona.primary_constraint}</b>. Reduce complexity to recover." if not is_safe else "The system is balanced. You are ready for the final audit.")
        ),
        dashboard.layout_cockpit(
            dashboard.lever_panel(mo.vstack([
                mo.md("Adjust the lever below to resize your model."),
                complexity_slider
            ])),
            dashboard.telemetry_panel(mo.vstack([
                mo.md(f"#### instrument: {persona.name}"),
                mo.md(f"<b>Current Load:</b> {complexity_slider.value}%"),
                mo.md(f"<b>Constraint:</b> {persona.primary_constraint}"),
                mo.md("---"),
                mo.md(f"<div style='font-size: 1.5em; font-weight: 900; color: {'#38a169' if is_safe else '#e53e3e'}'>{'✅ SYSTEM NOMINAL' if is_safe else '⚠️ PHYSICAL OVERLOAD'}</div>"),
            ]), color="#3182ce" if is_safe else "#e53e3e"),
            mo.md("") # Audit trail moved to Tab 3
        )
    ])

    # TAB 3: DEPLOYMENT
    deployment_content = mo.vstack([
        mo.md("### 📝 Phase 3: Audit & Deployment"),
        dashboard.audit_panel(mo.vstack([
            mo.md("Before we deploy to the fleet, you must provide the engineering rationale for your chosen complexity level."),
            mo.ui.text_area(placeholder="Why is this setup optimal for your mission?", label="Justification"),
            mo.center(mo.ui.button(label="🚀 Deploy to Fleet", kind="success", disabled=not is_safe))
        ]))
    ])

    # ASSEMBLE TABS
    tabs = mo.ui.tabs({
        "1. Briefing": briefing_content,
        "2. Simulation": simulation_content,
        "3. Deployment": deployment_content
    })
    return


@app.cell
def _(view):
    view
    return


if __name__ == "__main__":
    app.run()
