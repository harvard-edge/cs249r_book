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
    from pathlib import Path

    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.1.2-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        _labs_dir = Path(__file__).resolve().parents[1]
        if str(_labs_dir) not in sys.path:
            sys.path.insert(0, str(_labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)

    import plotly.graph_objects as go
    import mlsysim
    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import (
        ACADEMIC_LAB_CSS,
        architecture_memo,
        build_lab_report,
        capstone_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        replay_ledger,
        report_export_panel,
        resolve_mlsysim_ref,
        sensitivity_audit,
        source_trace,
        track_context,
        track_selector,
    )

    ledger = DesignLedger()
    if getattr(ledger, "is_wasm", False):
        _ = await ledger.load_async()
    return (
        ACADEMIC_LAB_CSS,
        COLORS,
        LAB_CSS,
        apply_plotly_theme,
        architecture_memo,
        build_lab_report,
        capstone_track_profile,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        mlsysim,
        mo,
        replay_ledger,
        report_export_panel,
        resolve_mlsysim_ref,
        sensitivity_audit,
        source_trace,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_16_metadata = get_lab_metadata("vol1/lab_16_ml_conclusion.py")
    return (v1_16_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_16_track_picker = track_selector(default=_default_track)
    v1_16_track_picker
    return (v1_16_track_picker,)


@app.cell
def _(
    capstone_track_profile,
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    v1_16_track_picker,
):
    v1_16_track_id = v1_16_track_picker.value
    v1_16_profile = get_track_profile(v1_16_track_id)
    v1_16_variant = get_lab_track_variant("v1_16_architects_audit", v1_16_profile.track_id)
    v1_16_hardware = resolve_mlsysim_ref(v1_16_variant.hardware_ref)
    v1_16_model = resolve_mlsysim_ref(v1_16_variant.model_ref)
    v1_16_capstone = capstone_track_profile(
        v1_16_profile,
        v1_16_variant,
        v1_16_hardware,
        v1_16_model,
    )
    return (
        v1_16_capstone,
        v1_16_hardware,
        v1_16_model,
        v1_16_profile,
        v1_16_track_id,
        v1_16_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v1_16_capstone,
    v1_16_metadata,
    v1_16_profile,
    v1_16_variant,
):
    mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume I &middot; Lab 16 Capstone
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                The Architect's Audit
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Ledger Replay &middot; Architecture Map &middot; Sensitivity Audit &middot; Final Memo
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 780px; line-height: 1.65;">
                {v1_16_variant.workload_summary} The final artifact is not a
                summary; it is an audit of the architecture implied by earlier decisions.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Memo &middot; ~55 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_16_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_16_capstone.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Ledger Replay</span>
                <span class="badge badge-warn">Fragility Heatmap</span>
                <span class="badge badge-fail">Residual Risk</span>
            </div>
        </div>
        """),
        track_context(v1_16_profile),
        source_trace(
            {
                "lab_id": v1_16_metadata.lab_id,
                "track_id": v1_16_profile.track_id,
                "hardware_ref": v1_16_variant.hardware_ref,
                "model_ref": v1_16_variant.model_ref,
                "shared_helper": "mlsysbook_labs.capstone",
                "report_artifact": v1_16_capstone.report_artifact,
                "source_policy": v1_16_profile.source_policy,
            },
            summary="V1-16 resolves the capstone architecture audit through MLSysIM refs and mlsysbook_labs.capstone calculations.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_16_capstone):
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
                <div style="margin-bottom: 3px;">1. <strong>Replay the ledger:</strong>
                    distinguish student evidence from track presets when prior entries are missing.</div>
                <div style="margin-bottom: 3px;">2. <strong>Audit sensitivity:</strong>
                    perturb workload, model size, guardrails, and evidence confidence.</div>
                <div style="margin-bottom: 3px;">3. <strong>Write the memo:</strong>
                    revise one decision, name the top risk, and choose validation evidence.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 260px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Architecture Goal
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_16_capstone.architecture_goal}
                </div>
            </div>
            <div style="flex: 0 0 260px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Durable Principle
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_16_capstone.durable_principle}
                </div>
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - This is the Volume I capstone. Review your ledger
    entries from Labs 00-15 and the design memos you downloaded from recent labs.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    partA_pred = mo.ui.radio(
        options={
            "A) The capstone should only summarize prior answers": "summary",
            "B) Missing ledger entries can use presets but must be labeled": "presets",
            "C) Only the final lab matters for architecture": "final_only",
            "D) Ledger gaps should be ignored": "ignore",
        },
        label=f"{v1_16_capstone.label}: how should the architecture audit handle missing prior evidence?",
    )
    return (partA_pred,)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    partB_pred = mo.ui.radio(
        options={
            "A) A model choice": "model",
            "B) The accumulated result of prior constraints": "accumulated",
            "C) A hardware spec sheet": "hardware",
            "D) A deployment diagram without residual risk": "diagram",
        },
        label=f"What is the architecture map for {v1_16_capstone.label} actually auditing?",
    )
    return (partB_pred,)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    _defaults = v1_16_capstone.sensitivity_defaults
    partC_pred = mo.ui.radio(
        options={
            "A) Workload growth is always the most fragile assumption": "workload",
            "B) The most fragile axis depends on track-specific headroom": "track_specific",
            "C) Evidence confidence is not a systems constraint": "evidence_free",
            "D) Tightening guardrails always improves feasibility": "tighten",
        },
        label=f"Which sensitivity axis is most dangerous for {v1_16_capstone.label}?",
    )
    partC_workload = mo.ui.slider(
        start=0.5,
        stop=4.0,
        value=min(4.0, max(0.5, float(_defaults.get("workload_limit_multiplier", 2.0)))),
        step=0.1,
        label="Workload multiplier",
    )
    partC_model_growth = mo.ui.slider(
        start=0,
        stop=80,
        value=int(float(_defaults.get("model_growth_limit_pct", 30.0))),
        step=5,
        label="Model growth (%)",
    )
    return (partC_model_growth, partC_pred, partC_workload)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    _defaults = v1_16_capstone.sensitivity_defaults
    partC_guardrail = mo.ui.slider(
        start=0,
        stop=60,
        value=int(float(_defaults.get("guardrail_tightening_limit_pct", 25.0))),
        step=5,
        label="Guardrail tightening (%)",
    )
    partC_evidence = mo.ui.slider(
        start=0,
        stop=100,
        value=int(float(_defaults.get("evidence_confidence_floor_pct", 75.0))),
        step=5,
        label="Evidence confidence (%)",
    )

    partD_pred = mo.ui.radio(
        options={
            "A) Revise the riskiest decision and name validation evidence": "revise",
            "B) Keep every prior decision unchanged": "unchanged",
            "C) Remove residual risk from the memo": "remove_risk",
            "D) Optimize the easiest component rather than the binding one": "easy",
        },
        label="What should the final architecture memo do?",
    )
    return (partC_evidence, partC_guardrail, partD_pred)


@app.cell(hide_code=True)
def _(mo, v1_16_capstone):
    _revision_options = {item: item for item in v1_16_capstone.revision_options}
    _risk_options = {item: item for item in v1_16_capstone.top_risks}
    _mitigations = tuple(
        str(value)
        for key, value in v1_16_capstone.sensitivity_defaults.items()
        if key.endswith("_mitigation")
    )
    _mitigation_options = {item: item for item in _mitigations}
    partD_revision = mo.ui.dropdown(
        options=_revision_options,
        value=v1_16_capstone.revision_options[0],
        label="Decision to revise",
    )
    partD_top_risk = mo.ui.dropdown(
        options=_risk_options,
        value=v1_16_capstone.top_risks[0],
        label="Top residual risk",
    )
    partD_mitigation = mo.ui.dropdown(
        options=_mitigation_options,
        value=_mitigations[0],
        label="Mitigation evidence",
    )
    return (partD_mitigation, partD_revision, partD_top_risk)


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    architecture_memo,
    go,
    ledger,
    mo,
    partA_pred,
    partB_pred,
    partC_evidence,
    partC_guardrail,
    partC_model_growth,
    partC_pred,
    partC_workload,
    partD_mitigation,
    partD_pred,
    partD_revision,
    partD_top_risk,
    replay_ledger,
    sensitivity_audit,
    v1_16_capstone,
    v1_16_profile,
    v1_16_variant,
):
    def _metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.45rem; font-weight:800; color:{color};">{value}</div>
            <div style="font-size:0.72rem; color:#64748b;">{detail}</div>
        </div>
        """

    _ledger_result = replay_ledger(v1_16_capstone, ledger._state.history)

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Audit Brief &middot; {v1_16_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Use the student's ledger where it exists. When it does not,
                    label the track preset so the memo does not pretend to have evidence."
                </div>
            </div>
            """),
            mo.md("""
## Part A: Ledger Replay

**What you need to know.** A capstone architecture is the accumulated result of
prior constraints. Missing ledger evidence is allowed, but it must be visible.
The report should separate student evidence from track presets.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the ledger replay."), kind="warn"))
            return mo.vstack(items)

        _coverage_color = COLORS["GreenLine"] if _ledger_result.coverage_pct >= 70 else COLORS["OrangeLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Ledger Coverage", f"{_ledger_result.coverage_pct:.0f}%", f"{_ledger_result.entries_found}/{_ledger_result.entries_expected} entries", _coverage_color, True)}
            {_metric_card("Missing Chapters", f"{len(_ledger_result.missing_chapters)}", "presets used", COLORS["OrangeLine"])}
            {_metric_card("Hardware", v1_16_capstone.hardware_ref, v1_16_capstone.hardware_name, COLORS["BlueLine"])}
            {_metric_card("Model", v1_16_capstone.model_ref, v1_16_capstone.model_name, COLORS["RedLine"])}
        </div>
        """))

        _rows = "\n".join(
            f"| {decision.chapter} | {decision.label} | {decision.constraint} | {decision.source} | {decision.decision} |"
            for decision in _ledger_result.decisions
        )
        items.append(mo.md(f"""
**Ledger Replay Table**

| Lab | Evidence | Constraint | Source | Decision |
|---:|---|---|---|---|
{_rows}

*Source: `mlsysbook_labs.replay_ledger`; missing entries use typed V1-16 variants.*
        """))

        if _ledger_result.missing_chapters:
            items.append(mo.callout(mo.md(
                f"**Ledger gaps are labeled.** Missing chapters: {', '.join(str(ch) for ch in _ledger_result.missing_chapters)}. "
                "The exported memo will preserve that distinction."
            ), kind="warn"))

        if partA_pred.value == "presets":
            items.append(mo.callout(mo.md("**Correct.** Presets are acceptable only when they remain visible as presets."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Capstone evidence must be honest.** A missing ledger entry is not fatal, but hiding it weakens the memo."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Architecture Map &middot; {v1_16_capstone.label}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "This map is not a diagram for its own sake. It shows where
                    prior Volume I decisions now bind the architecture."
                </div>
            </div>
            """),
            mo.md(f"""
## Part B: Architecture Map

**What you need to know.** The map is driven by the selected track:
**{v1_16_capstone.architecture_goal}**
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the architecture map."), kind="warn"))
            return mo.vstack(items)

        _components = list(v1_16_capstone.architecture_components)
        _confidence_by_component = []
        for _idx, _component in enumerate(_components):
            _decision = _ledger_result.decisions[min(_idx, len(_ledger_result.decisions) - 1)]
            _confidence_by_component.append(_decision.confidence_pct)

        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[f"C{idx + 1}" for idx in range(len(_components))],
            y=_confidence_by_component,
            marker_color=[COLORS["GreenLine"] if val >= 90 else COLORS["OrangeLine"] for val in _confidence_by_component],
            opacity=0.9,
        ))
        _fig.update_layout(
            height=330,
            yaxis=dict(title="Evidence confidence (%)", gridcolor="#f1f5f9", range=[0, 105]),
            xaxis=dict(title="Architecture component"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _component_rows = "\n".join(
            f"| C{idx + 1} | {component} | {_confidence_by_component[idx]:.0f}% |"
            for idx, component in enumerate(_components)
        )
        items.append(mo.md(f"""
**Architecture Components**

| ID | Component | Evidence confidence |
|---|---|---:|
{_component_rows}
        """))

        if partB_pred.value == "accumulated":
            items.append(mo.callout(mo.md("**Correct.** Architecture is accumulated constraint management across the volume."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**The architecture map is not just model or hardware.** It is the chain of constraints and decisions accumulated across labs."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Stress Test &middot; Sensitivity Audit
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Perturb the assumptions. If the architecture only works for
                    one exact setting, it is not robust."
                </div>
            </div>
            """),
            mo.md("""
## Part C: Sensitivity Audit

**What you need to know.** Fragility is track-specific. Workload growth, model
growth, guardrail tightening, and evidence confidence stress different systems
in different ways.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the sensitivity audit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_workload, partC_model_growth, partC_guardrail, partC_evidence], widths="equal"))
        _audit = sensitivity_audit(
            v1_16_capstone,
            workload_multiplier=partC_workload.value,
            model_growth_pct=partC_model_growth.value,
            guardrail_tightening_pct=partC_guardrail.value,
            evidence_confidence_pct=partC_evidence.value,
        )
        _fig = go.Figure()
        _fig.add_trace(go.Bar(
            x=[axis.name for axis in _audit.axes],
            y=[axis.risk_pct for axis in _audit.axes],
            marker_color=[COLORS["GreenLine"] if axis.status == "PASS" else COLORS["RedLine"] for axis in _audit.axes],
            opacity=0.9,
        ))
        _fig.add_hline(y=100, line_dash="dash", line_color="#64748b", annotation_text="failure boundary")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Risk score (%)", gridcolor="#f1f5f9", range=[0, 110]),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _audit.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Most Fragile", _audit.most_fragile, "highest sensitivity score", COLORS["OrangeLine"], True)}
            {_metric_card("Fragility", f"{_audit.fragility_score_pct:.0f}%", "average risk score", COLORS["BlueLine"])}
            {_metric_card("Status", "PASS" if _audit.feasible else "FAIL", ", ".join(_audit.violations) or "no violations", _status_color, True)}
        </div>
        """))

        _axis_rows = "\n".join(
            f"| {axis.name} | {axis.value:.1f} | {axis.limit:.1f} | {axis.risk_pct:.0f}% | {axis.status} | {axis.mitigation} |"
            for axis in _audit.axes
        )
        items.append(mo.md(f"""
**Sensitivity Table**

| Axis | Value | Limit | Risk | Status | Mitigation |
|---|---:|---:|---:|---|---|
{_axis_rows}

*Source: `mlsysbook_labs.sensitivity_audit`.*
        """))

        if not _audit.feasible:
            items.append(mo.callout(mo.md(
                "**Sensitivity failure:** " + ", ".join(_audit.violations) + ". The memo must revise a decision or add validation evidence."
            ), kind="danger"))

        if partC_pred.value == "track_specific":
            items.append(mo.callout(mo.md("**Correct.** Fragility depends on the selected track and its headroom."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Fragility is not universal.** The same perturbation stresses iPhone, Oura Ring, RoboTaxi, and Cloud Fleet differently."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['GreenLine']}; background:{COLORS['GreenLL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['GreenLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Final Artifact &middot; Architecture Memo
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Revise one decision, name one top risk, and choose the evidence
                    that would make the design defensible."
                </div>
            </div>
            """),
            mo.md("""
## Part D: Architecture Memo

**What you need to know.** The capstone memo is a decision artifact. It should
not claim that every constraint is solved. It should state which constraint you
choose to live with and how you will validate it.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the final memo builder."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_revision, partD_top_risk, partD_mitigation], widths="equal"))
        _memo = architecture_memo(
            v1_16_capstone,
            revised_decision=partD_revision.value,
            top_risk=partD_top_risk.value,
            mitigation=partD_mitigation.value,
        )
        items.append(mo.callout(mo.md(f"**Memo summary:** {_memo.memo_summary}"), kind="info"))
        items.append(mo.md(f"""
**Validation Tests To Attach**

{chr(10).join(f"- {test}" for test in _memo.validation_tests)}

**Durable Principle**

{_memo.durable_principle}
        """))

        if partD_pred.value == "revise":
            items.append(mo.callout(mo.md("**Correct.** The memo must revise, risk-rank, and validate."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**A capstone memo is not a victory lap.** It is a defensible revision under residual risk."
            ), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                f"**1. The selected track controls the capstone.** {v1_16_capstone.label} has a different architecture memo than the other tracks."
            ), kind="info"),
            mo.callout(mo.md(
                "**2. Ledger gaps are evidence gaps.** Presets are useful for teaching, but the report must label them."
            ), kind="info"),
            mo.callout(mo.md(
                f"**3. Durable principle:** {v1_16_capstone.durable_principle}"
            ), kind="info"),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Ledger Replay": build_part_a(),
        "Part B: Architecture Map": build_part_b(),
        "Part C: Sensitivity Audit": build_part_c(),
        "Part D: Architecture Memo": build_part_d(),
        "Synthesis": build_synthesis(),
    })
    _tabs
    return


# ===========================================================================
# ZONE D: LEDGER HUD AND REPORT
# ===========================================================================


@app.cell(hide_code=True)
def _(
    ledger,
    mo,
    partA_pred,
    partB_pred,
    partC_pred,
    partD_pred,
    v1_16_capstone,
    v1_16_profile,
    v1_16_variant,
):
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=16, design={
            "chapter": "v1_16",
            "track_id": v1_16_profile.track_id,
            "scenario_id": v1_16_variant.scenario_id,
            "hardware_ref": v1_16_capstone.hardware_ref,
            "model_ref": v1_16_capstone.model_ref,
            "architecture_goal": v1_16_capstone.architecture_goal,
            "durable_principle": v1_16_capstone.durable_principle,
            "completed": True,
            "ledger_replay_prediction": partA_pred.value,
            "architecture_map_prediction": partB_pred.value,
            "sensitivity_prediction": partC_pred.value,
            "memo_prediction": partD_pred.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">16 &middot; Architect's Audit</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_16_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">ARTIFACT</span>
        <span class="hud-value">{v1_16_capstone.report_artifact}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    architecture_memo,
    build_lab_report,
    ledger,
    mo,
    partA_pred,
    partB_pred,
    partC_evidence,
    partC_guardrail,
    partC_model_growth,
    partC_pred,
    partC_workload,
    partD_mitigation,
    partD_pred,
    partD_revision,
    partD_top_risk,
    replay_ledger,
    report_export_panel,
    sensitivity_audit,
    v1_16_capstone,
    v1_16_metadata,
    v1_16_profile,
    v1_16_variant,
):
    _ledger_result = replay_ledger(v1_16_capstone, ledger._state.history)
    _audit = sensitivity_audit(
        v1_16_capstone,
        workload_multiplier=partC_workload.value,
        model_growth_pct=partC_model_growth.value,
        guardrail_tightening_pct=partC_guardrail.value,
        evidence_confidence_pct=partC_evidence.value,
    )
    _memo = architecture_memo(
        v1_16_capstone,
        revised_decision=partD_revision.value,
        top_risk=partD_top_risk.value,
        mitigation=partD_mitigation.value,
    )

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A ledger replay prediction")
    if partB_pred.value is None:
        _incomplete.append("Part B architecture map prediction")
    if partC_pred.value is None:
        _incomplete.append("Part C sensitivity prediction")
    if partD_pred.value is None:
        _incomplete.append("Part D architecture memo prediction")

    _report = build_lab_report(
        v1_16_metadata,
        track=v1_16_profile.label,
        scenario=v1_16_variant.workload_summary,
        learning_objectives=(
            "Replay prior Volume I ledger decisions and label missing evidence.",
            "Map the selected track's accumulated architecture constraints.",
            "Stress the architecture with sensitivity perturbations and revise one decision.",
        ),
        predictions={
            "ledger_replay": partA_pred.value,
            "architecture_map": partB_pred.value,
            "sensitivity_audit": partC_pred.value,
            "architecture_memo": partD_pred.value,
        },
        knob_settings={
            "workload_multiplier": partC_workload.value,
            "model_growth_pct": partC_model_growth.value,
            "guardrail_tightening_pct": partC_guardrail.value,
            "evidence_confidence_pct": partC_evidence.value,
            "revised_decision": partD_revision.value,
            "top_risk": partD_top_risk.value,
            "mitigation": partD_mitigation.value,
        },
        evidence_summary={
            "hardware_ref": v1_16_capstone.hardware_ref,
            "model_ref": v1_16_capstone.model_ref,
            "architecture_goal": v1_16_capstone.architecture_goal,
            "ledger_coverage_pct": round(_ledger_result.coverage_pct, 3),
            "missing_chapters": _ledger_result.missing_chapters,
            "most_fragile": _audit.most_fragile,
            "sensitivity_feasible": _audit.feasible,
            "sensitivity_violations": _audit.violations,
            "revised_decision": _memo.revised_decision,
            "top_risk": _memo.top_risk,
        },
        final_decision=_memo.memo_summary,
        big_takeaways=(
            "Architecture is accumulated constraint management.",
            "Missing ledger entries are evidence gaps and must be labeled.",
            "The capstone memo should revise one decision under a named residual risk.",
        ),
        reflections={
            "report_artifact": v1_16_capstone.report_artifact,
            "durable_principle": v1_16_capstone.durable_principle,
            "validation_tests": v1_16_capstone.validation_tests,
        },
        residual_risk=(
            f"{_memo.top_risk}. The mitigation is {_memo.mitigation}; evidence still needs "
            "track-specific validation before production use."
        ),
        source_trace={
            "track_id": v1_16_profile.track_id,
            "scenario_id": v1_16_variant.scenario_id,
            "hardware_ref": v1_16_variant.hardware_ref,
            "model_ref": v1_16_variant.model_ref,
            "shared_helper": "mlsysbook_labs.capstone",
            "source_policy": v1_16_profile.source_policy,
        },
        result_snapshot={
            "capstone_profile": v1_16_capstone,
            "ledger_replay": _ledger_result,
            "sensitivity_audit": _audit,
            "architecture_memo": _memo,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-16 architecture memo is generated locally from the selected track, "
                "ledger entries, MLSysIM refs, and shared `mlsysbook_labs.capstone` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
