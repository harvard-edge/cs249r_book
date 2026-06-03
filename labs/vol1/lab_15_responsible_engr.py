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
    import numpy as np

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
        build_lab_report,
        carbon_budget,
        explanation_overhead,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        metric_conflict,
        report_export_panel,
        resolve_mlsysim_ref,
        responsibility_budget,
        responsibility_track_profile,
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
        build_lab_report,
        carbon_budget,
        explanation_overhead,
        get_lab_metadata,
        get_lab_track_variant,
        get_track_profile,
        go,
        ledger,
        metric_conflict,
        mlsysim,
        mo,
        np,
        report_export_panel,
        resolve_mlsysim_ref,
        responsibility_budget,
        responsibility_track_profile,
        source_trace,
        track_context,
        track_selector,
    )


@app.cell
def _(get_lab_metadata):
    v1_15_metadata = get_lab_metadata("vol1/lab_15_responsible_engr.py")
    return (v1_15_metadata,)


@app.cell(hide_code=True)
def _(ledger, track_selector):
    _saved_track = ledger.get_track()
    _default_track = _saved_track if _saved_track and _saved_track != "NONE" else "iphone"
    v1_15_track_picker = track_selector(default=_default_track)
    v1_15_track_picker
    return (v1_15_track_picker,)


@app.cell
def _(
    get_lab_track_variant,
    get_track_profile,
    resolve_mlsysim_ref,
    responsibility_track_profile,
    v1_15_track_picker,
):
    v1_15_track_id = v1_15_track_picker.value
    v1_15_profile = get_track_profile(v1_15_track_id)
    v1_15_variant = get_lab_track_variant("v1_15_no_free_fairness", v1_15_profile.track_id)
    v1_15_hardware = resolve_mlsysim_ref(v1_15_variant.hardware_ref)
    v1_15_model = resolve_mlsysim_ref(v1_15_variant.model_ref)
    v1_15_resp = responsibility_track_profile(
        v1_15_profile,
        v1_15_variant,
        v1_15_hardware,
        v1_15_model,
    )
    return (
        v1_15_hardware,
        v1_15_model,
        v1_15_profile,
        v1_15_resp,
        v1_15_track_id,
        v1_15_variant,
    )


@app.cell(hide_code=True)
def _(
    ACADEMIC_LAB_CSS,
    LAB_CSS,
    mo,
    source_trace,
    track_context,
    v1_15_metadata,
    v1_15_profile,
    v1_15_resp,
    v1_15_variant,
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
                Machine Learning Systems &middot; Volume I &middot; Lab 15
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                No Free Fairness
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                Metric Conflict &middot; Responsibility Budget &middot; Explainability Tax &middot; Carbon
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 780px; line-height: 1.65;">
                {v1_15_variant.workload_summary} Responsible engineering is not an
                afterthought; it is a measurable systems constraint.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;">
                <span style="background: rgba(99,102,241,0.18); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.3);">
                    4 Parts + Synthesis &middot; ~50 min
                </span>
                <span style="background: rgba(203,32,45,0.15); color: #fca5a5;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(203,32,45,0.25);">
                    {v1_15_profile.label}
                </span>
                <span style="background: rgba(34,197,94,0.12); color: #86efac;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(34,197,94,0.20);">
                    {v1_15_resp.hardware_ref}
                </span>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">Harmed Party</span>
                <span class="badge badge-warn">Audit Signal</span>
                <span class="badge badge-fail">Residual Harm</span>
            </div>
        </div>
        """),
        track_context(v1_15_profile),
        source_trace(
            {
                "lab_id": v1_15_metadata.lab_id,
                "track_id": v1_15_profile.track_id,
                "hardware_ref": v1_15_variant.hardware_ref,
                "model_ref": v1_15_variant.model_ref,
                "shared_helper": "mlsysbook_labs.responsibility",
                "obligation": v1_15_resp.obligation,
                "audit_signal": v1_15_resp.audit_signal,
                "source_policy": v1_15_profile.source_policy,
            },
            summary="V1-15 resolves responsibility scenarios through MLSysIM refs and mlsysbook_labs.responsibility calculations.",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(COLORS, mo, v1_15_resp):
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
                <div style="margin-bottom: 3px;">1. <strong>Name the harmed party:</strong>
                    identify who pays when the selected track fails responsibly.</div>
                <div style="margin-bottom: 3px;">2. <strong>Quantify metric conflict:</strong>
                    show why one aggregate metric cannot satisfy every subgroup or context.</div>
                <div style="margin-bottom: 3px;">3. <strong>Budget responsibility:</strong>
                    connect privacy, explanations, robustness, monitoring, and carbon to systems overhead.</div>
            </div>
        </div>
        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>
        <div style="display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 240px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Obligation
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_15_resp.obligation}
                </div>
            </div>
            <div style="flex: 1; min-width: 240px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Audit Signal
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    {v1_15_resp.audit_signal}
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
                "What responsible design can protect {v1_15_resp.harmed_party}
                without pretending the overhead is free?"
            </div>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** - Complete before this lab:

    - **The Responsible Engineering chapter** - metric conflict, fairness trade-offs,
      explainability overhead, privacy constraints, robustness, and sustainability.
    """), kind="info")
    return


# ===========================================================================
# ZONE B: WIDGET DEFINITIONS
# ===========================================================================


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partA_pred = mo.ui.radio(
        options={
            "A) Aggregate quality is enough if it stays high": "aggregate",
            "B) The subgroup gap can violate the obligation even when aggregate quality looks good": "gap",
            "C) Lowering the threshold always fixes the harmed party": "threshold",
            "D) Audits are only needed after deployment incidents": "incident",
        },
        label=f"{v1_15_resp.label}: what is the first responsibility risk for {v1_15_resp.harmed_party}?",
    )
    return (partA_pred,)


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partA_base_a = mo.ui.slider(
        start=5,
        stop=60,
        value=30,
        step=5,
        label=f"{v1_15_resp.subgroup_a} base/context rate (%)",
    )
    partA_base_b = mo.ui.slider(
        start=5,
        stop=60,
        value=10,
        step=5,
        label=f"{v1_15_resp.subgroup_b} base/context rate (%)",
    )
    partA_threshold = mo.ui.slider(start=0.10, stop=0.90, value=0.50, step=0.05, label="Decision threshold")

    partB_pred = mo.ui.radio(
        options={
            "A) Add every control at maximum strength": "max",
            "B) Choose controls until the gap closes and budgets still pass": "balanced",
            "C) Explanations are free if they are only generated for some users": "free_explanations",
            "D) Monitoring changes governance but not system cost": "monitoring_free",
        },
        label=f"How should {v1_15_resp.stakeholder} budget responsibility controls?",
    )
    return (partA_base_a, partA_base_b, partA_threshold, partB_pred)


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partB_privacy = mo.ui.slider(start=0, stop=100, value=60, step=5, label="Privacy / consent strength")
    partB_explain = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_15_resp.explanation_coverage_pct),
        step=5,
        label="Explanation coverage (%)",
    )
    partB_robustness = mo.ui.slider(start=0, stop=100, value=70, step=5, label="Robustness / mitigation strength")
    partB_monitoring = mo.ui.slider(start=0, stop=100, value=60, step=5, label="Audit / monitoring strength")

    partC_pred = mo.ui.radio(
        options={
            "A) The explanation method only affects the report text": "text_only",
            "B) Explanations add latency proportional to the method and feature count": "latency",
            "C) SHAP is always safe for online serving": "always_safe",
            "D) Explanations remove the need for subgroup audits": "replace_audit",
        },
        label=f"What does explainability cost on {v1_15_resp.hardware_name}?",
    )
    return (partB_explain, partB_monitoring, partB_privacy, partB_robustness, partC_pred)


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partC_features = mo.ui.slider(
        start=5,
        stop=120,
        value=v1_15_resp.explanation_features,
        step=5,
        label="Explanation features / trace factors",
    )
    partC_method = mo.ui.dropdown(
        options={
            "None": "none",
            "Feature Importance": "feature_importance",
            "LIME": "lime",
            "Trace Replay": "trace_replay",
            "SHAP": "shap",
        },
        value={
            "none": "None",
            "feature_importance": "Feature Importance",
            "lime": "LIME",
            "trace_replay": "Trace Replay",
            "shap": "SHAP",
        }.get(v1_15_resp.explanation_method, "SHAP"),
        label="Explanation method",
    )
    partC_coverage = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_15_resp.explanation_coverage_pct),
        step=5,
        label="Online explanation coverage (%)",
    )

    partD_pred = mo.ui.radio(
        options={
            "A) Carbon is dominated by one training run": "one_train",
            "B) Retraining cadence and explanation coverage can dominate the footprint": "cadence",
            "C) Grid carbon intensity does not matter for the same model": "grid_irrelevant",
            "D) Carbon accounting is separate from responsible engineering": "separate",
        },
        label=f"What changes the carbon budget for {v1_15_resp.label}?",
    )
    return (partC_coverage, partC_features, partC_method, partD_pred)


@app.cell(hide_code=True)
def _(mo, v1_15_resp):
    partD_retrains = mo.ui.slider(
        start=1,
        stop=52,
        value=v1_15_resp.retrain_frequency_per_year,
        step=1,
        label="Retrains per year",
    )
    partD_explain = mo.ui.slider(
        start=0,
        stop=100,
        value=int(v1_15_resp.explanation_coverage_pct),
        step=5,
        label="Explanation coverage for carbon (%)",
    )
    partD_grid_ci = mo.ui.slider(
        start=20,
        stop=800,
        value=int(v1_15_resp.grid_ci_g_per_kwh),
        step=10,
        label="Grid carbon intensity (gCO2/kWh)",
    )
    return (partD_explain, partD_grid_ci, partD_retrains)


# ===========================================================================
# ZONE C: MAIN LAB
# ===========================================================================


@app.cell(hide_code=True)
def _(
    COLORS,
    apply_plotly_theme,
    carbon_budget,
    explanation_overhead,
    go,
    metric_conflict,
    mo,
    np,
    partA_base_a,
    partA_base_b,
    partA_pred,
    partA_threshold,
    partB_explain,
    partB_monitoring,
    partB_pred,
    partB_privacy,
    partB_robustness,
    partC_coverage,
    partC_features,
    partC_method,
    partC_pred,
    partD_explain,
    partD_grid_ci,
    partD_pred,
    partD_retrains,
    responsibility_budget,
    v1_15_profile,
    v1_15_resp,
    v1_15_variant,
):
    def _metric_card(label, value, detail, color, border=False):
        border_style = f"2px solid {color}" if border else "1px solid #e2e8f0"
        return f"""
        <div style="padding:16px; border:{border_style}; border-radius:10px;
                    min-width:150px; text-align:center; background:white;
                    border-top:3px solid {color}; flex:1;">
            <div style="color:#64748b; font-size:0.78rem; font-weight:700;">{label}</div>
            <div style="font-size:1.55rem; font-weight:800; color:{color};">{value}</div>
            <div style="font-size:0.72rem; color:#64748b;">{detail}</div>
        </div>
        """

    def build_part_a():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['BlueLine']}; background:{COLORS['BlueL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['BlueLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Responsibility Brief &middot; {v1_15_variant.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Aggregate quality is high. Can we ship, or does the responsibility
                    obligation require a subgroup audit first?"
                </div>
            </div>
            """),
            mo.md(f"""
## Part A: Metric Conflict

**What you need to know.** A high aggregate metric can hide a harmed subgroup or
context. For this track, the harmed party is **{v1_15_resp.harmed_party}** and
the obligation is **{v1_15_resp.obligation}**.

The simulator compares two subgroups or contexts with different base rates and
a shared threshold. The point is not to claim the formula is a production audit;
the point is to make the conflict visible before the design memo is written.
            """),
            partA_pred,
        ]
        if partA_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the subgroup metric audit."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partA_base_a, partA_base_b, partA_threshold], widths="equal"))
        _conflict = metric_conflict(
            v1_15_resp,
            base_rate_a_pct=partA_base_a.value,
            base_rate_b_pct=partA_base_b.value,
            threshold=partA_threshold.value,
        )

        _metrics = ["Accuracy", "FPR", "FNR", "PPV"]
        _a_values = [
            _conflict.accuracy_a_pct,
            _conflict.fpr_a_pct,
            _conflict.fnr_a_pct,
            _conflict.ppv_a_pct,
        ]
        _b_values = [
            _conflict.accuracy_b_pct,
            _conflict.fpr_b_pct,
            _conflict.fnr_b_pct,
            _conflict.ppv_b_pct,
        ]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(name=_conflict.subgroup_a, x=_metrics, y=_a_values, marker_color=COLORS["BlueLine"], opacity=0.88))
        _fig.add_trace(go.Bar(name=_conflict.subgroup_b, x=_metrics, y=_b_values, marker_color=COLORS["OrangeLine"], opacity=0.88))
        _fig.update_layout(
            barmode="group",
            height=360,
            yaxis=dict(title="Percentage (%)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=50, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _gap_color = COLORS["RedLine"] if _conflict.fpr_gap_pp > v1_15_resp.target_gap_pp else COLORS["GreenLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Accuracy Gap", f"{_conflict.accuracy_gap_pp:.1f} pp", "aggregate can look stable", COLORS["BlueLine"])}
            {_metric_card("FPR Gap", f"{_conflict.fpr_gap_pp:.1f} pp", f"target {v1_15_resp.target_gap_pp:.1f} pp", _gap_color, True)}
            {_metric_card("PPV Gap", f"{_conflict.ppv_gap_pp:.1f} pp", "different user impact", COLORS["OrangeLine"])}
            {_metric_card("Audit Signal", "required", v1_15_resp.audit_signal, COLORS["RedLine"])}
        </div>
        """))

        items.append(mo.md(f"""
**Metric Table - Live Values**

| Metric | {v1_15_resp.subgroup_a} | {v1_15_resp.subgroup_b} |
|---|---:|---:|
| Base/context rate | {_conflict.base_rate_a_pct:.1f}% | {_conflict.base_rate_b_pct:.1f}% |
| Accuracy | {_conflict.accuracy_a_pct:.1f}% | {_conflict.accuracy_b_pct:.1f}% |
| False positive rate | {_conflict.fpr_a_pct:.1f}% | {_conflict.fpr_b_pct:.1f}% |
| False negative rate | {_conflict.fnr_a_pct:.1f}% | {_conflict.fnr_b_pct:.1f}% |
| Positive predictive value | {_conflict.ppv_a_pct:.1f}% | {_conflict.ppv_b_pct:.1f}% |

*Source: `mlsysbook_labs.metric_conflict`, track `{v1_15_profile.track_id}`.*
        """))

        if _conflict.fpr_gap_pp > v1_15_resp.target_gap_pp:
            items.append(mo.callout(mo.md(
                f"**Gap above target.** {_conflict.conflict_summary} A responsible design needs mitigation and audit evidence."
            ), kind="danger"))
        else:
            items.append(mo.callout(mo.md(
                f"**Gap within target.** {_conflict.conflict_summary} The audit still belongs in the report."
            ), kind="success"))

        if partA_pred.value == "gap":
            items.append(mo.callout(mo.md("**Correct.** Responsibility starts with who can be harmed by an aggregate metric."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Aggregate quality is insufficient.** The report must name the harmed party, gap, audit signal, and residual harm."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_b():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Budget Review &middot; {v1_15_resp.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "Add privacy, explanations, robustness, and monitoring. Which
                    combination closes the gap without breaking the system?"
                </div>
            </div>
            """),
            mo.md(f"""
## Part B: Responsibility Budget

**What you need to know.** Responsibility controls are technical controls. They
change latency, energy, cost, quality, and governance delay. The track-specific
guardrail is **{v1_15_resp.guardrail_metric}**.
            """),
            partB_pred,
        ]
        if partB_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the responsibility budget."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partB_privacy, partB_explain, partB_robustness, partB_monitoring], widths="equal"))
        _budget = responsibility_budget(
            v1_15_resp,
            privacy_level=partB_privacy.value,
            explanation_coverage_pct=partB_explain.value,
            robustness_level=partB_robustness.value,
            monitoring_level=partB_monitoring.value,
        )
        _ratios = [
            _budget.latency_ms / _budget.latency_slo_ms,
            _budget.energy_factor / v1_15_resp.max_energy_factor,
            _budget.cost_factor / v1_15_resp.max_cost_factor,
            _budget.fairness_gap_pp / max(0.1, _budget.target_gap_pp),
        ]
        _names = ["Latency/SLO", "Energy/Budget", "Cost/Budget", "Gap/Target"]
        _colors = [COLORS["GreenLine"] if _r <= 1 else COLORS["RedLine"] for _r in _ratios]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=_names, y=_ratios, marker_color=_colors, opacity=0.9))
        _fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b", annotation_text="budget boundary")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Ratio to allowed budget", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _status_color = COLORS["GreenLine"] if _budget.feasible else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Latency", f"{_budget.latency_ms:.1f} ms", f"SLO {_budget.latency_slo_ms:.0f} ms", COLORS["BlueLine"])}
            {_metric_card("Energy Factor", f"{_budget.energy_factor:.2f}x", f"max {v1_15_resp.max_energy_factor:.1f}x", COLORS["OrangeLine"])}
            {_metric_card("Gap", f"{_budget.fairness_gap_pp:.1f} pp", f"target {_budget.target_gap_pp:.1f} pp", COLORS["RedLine"])}
            {_metric_card("Policy", "PASS" if _budget.feasible else "FAIL", ", ".join(_budget.violations) or "no violations", _status_color, True)}
        </div>
        """))

        items.append(mo.md(f"""
**Budget Table - Live Values**

| Quantity | Value |
|---|---:|
| Privacy strength | {_budget.privacy_level:.0f}% |
| Explanation coverage | {_budget.explanation_coverage_pct:.0f}% |
| Robustness strength | {_budget.robustness_level:.0f}% |
| Monitoring strength | {_budget.monitoring_level:.0f}% |
| Estimated quality | {_budget.estimated_quality_pct:.1f}% |
| Quality delta | {_budget.quality_delta_pp:+.1f} pp |
| Cost factor | {_budget.cost_factor:.2f}x |
| Governance delay | {_budget.governance_delay_days:.1f} days |

*Source: `mlsysbook_labs.responsibility_budget`.*
        """))

        if not _budget.feasible:
            items.append(mo.callout(mo.md(
                "**Budget violation:** " + ", ".join(_budget.violations) + ". Reduce coverage, change method, or increase the system budget."
            ), kind="danger"))

        if partB_pred.value == "balanced" and _budget.feasible:
            items.append(mo.callout(mo.md("**Correct.** A responsible system names a target and keeps the overhead budget visible."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Responsibility is not a maximum-controls checkbox.** The defensible point is the smallest control stack that satisfies the obligation and the system budgets."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_c():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['RedLine']}; background:{COLORS['RedL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['RedLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Explanation Requirement &middot; {v1_15_resp.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "We need a usable explanation. Can it run online, or does it
                    belong in an asynchronous audit/report path?"
                </div>
            </div>
            """),
            mo.md(f"""
## Part C: Explainability Tax

**What you need to know.** Explanations consume model evaluations, trace replay,
or feature passes. The same pedagogical idea becomes different across tracks:
mobile and wearable systems feel latency and battery, RoboTaxi feels p99 safety,
and cloud feels SLA and fleet cost.
            """),
            partC_pred,
        ]
        if partC_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the explanation calculator."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partC_method, partC_features, partC_coverage], widths="equal"))
        _explain = explanation_overhead(
            v1_15_resp,
            method=partC_method.value,
            features=partC_features.value,
            coverage_pct=partC_coverage.value,
        )

        _method_labels = {
            "none": "None",
            "feature_importance": "Feature Importance",
            "lime": "LIME",
            "trace_replay": "Trace Replay",
            "shap": "SHAP",
        }
        _methods = ["none", "feature_importance", "lime", "trace_replay", "shap"]
        _latencies = [
            explanation_overhead(v1_15_resp, method=_method, features=partC_features.value, coverage_pct=partC_coverage.value).total_latency_ms
            for _method in _methods
        ]
        _colors = [COLORS["GreenLine"] if _lat <= v1_15_resp.latency_slo_ms else COLORS["RedLine"] for _lat in _latencies]
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=[_method_labels[_m] for _m in _methods], y=_latencies, marker_color=_colors, opacity=0.9))
        _fig.add_hline(y=v1_15_resp.latency_slo_ms, line_dash="dash", line_color=COLORS["BlueLine"], annotation_text="latency SLO")
        _fig.update_layout(
            height=340,
            yaxis=dict(title="Total explanation path latency (ms)", gridcolor="#f1f5f9"),
            margin=dict(l=60, r=20, t=40, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        _slo_color = COLORS["GreenLine"] if _explain.slo_ok else COLORS["RedLine"]
        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Base Latency", f"{_explain.base_latency_ms:.1f} ms", v1_15_resp.hardware_name, COLORS["BlueLine"])}
            {_metric_card("Multiplier", f"{_explain.multiplier:.1f}x", _method_labels.get(_explain.method, _explain.method), COLORS["OrangeLine"])}
            {_metric_card("Total Latency", f"{_explain.total_latency_ms:.1f} ms", f"SLO {_explain.slo_ms:.0f} ms", _slo_color, True)}
            {_metric_card("p99 Added", f"{_explain.p99_added_ms:.1f} ms", f"{_explain.coverage_pct:.0f}% coverage", COLORS["RedLine"])}
        </div>
        """))

        items.append(mo.md(f"""
**Explanation Table - Live Values**

| Quantity | Value |
|---|---:|
| Method | {_method_labels.get(_explain.method, _explain.method)} |
| Features / trace factors | {_explain.features} |
| Base latency | {_explain.base_latency_ms:.1f} ms |
| Explanation latency | {_explain.explanation_latency_ms:.1f} ms |
| Total latency | {_explain.total_latency_ms:.1f} ms |
| p99 added at coverage | {_explain.p99_added_ms:.1f} ms |
| SLO status | {"PASS" if _explain.slo_ok else "FAIL"} |

*Source: `mlsysbook_labs.explanation_overhead`.*
        """))

        if not _explain.slo_ok:
            items.append(mo.callout(mo.md(
                f"**SLO violation.** {_method_labels.get(_explain.method, _explain.method)} adds enough work to exceed the track latency guardrail. Use async explanations, sampling, a lighter method, or a larger serving budget."
            ), kind="danger"))

        if partC_pred.value == "latency":
            items.append(mo.callout(mo.md("**Correct.** Explainability is a systems path with latency, cost, and coverage choices."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Explanation is not just report text.** The method and coverage determine whether the responsible design still satisfies the system guardrails."
            ), kind="warn"))
        return mo.vstack(items)

    def build_part_d():
        items = [
            mo.Html(f"""
            <div style="border-left:4px solid {COLORS['OrangeLine']}; background:{COLORS['OrangeL']};
                        border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
                <div style="font-size:0.72rem; font-weight:700; color:{COLORS['OrangeLine']};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    Sustainability Review &middot; {v1_15_resp.stakeholder}
                </div>
                <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                    "The responsible stack needs retraining and explanations. What
                    is the annual carbon footprint of that decision?"
                </div>
            </div>
            """),
            mo.md("""
## Part D: Carbon Ledger

**What you need to know.** Responsibility can require more retraining, more
audits, more explanation calls, and more governance work. Carbon is not a reason
to ignore harm, but it is part of the design budget and should be reported.
            """),
            partD_pred,
        ]
        if partD_pred.value is None:
            items.append(mo.callout(mo.md("Select your prediction to unlock the carbon ledger."), kind="warn"))
            return mo.vstack(items)

        items.append(mo.hstack([partD_retrains, partD_explain, partD_grid_ci], widths="equal"))
        _carbon = carbon_budget(
            v1_15_resp,
            retrain_frequency_per_year=partD_retrains.value,
            explanation_coverage_pct=partD_explain.value,
            grid_ci_g_per_kwh=partD_grid_ci.value,
        )

        _fig = go.Figure()
        _fig.add_trace(go.Bar(name="Baseline", x=["Baseline"], y=[_carbon.baseline_kgco2_per_year], marker_color=COLORS["GreenLine"], opacity=0.9))
        _fig.add_trace(go.Bar(name="Retraining", x=["Responsible stack"], y=[_carbon.retraining_kwh_per_year * _carbon.grid_ci_g_per_kwh / 1000.0], marker_color=COLORS["BlueLine"], opacity=0.9))
        _fig.add_trace(go.Bar(name="Serving", x=["Responsible stack"], y=[_carbon.base_serving_kwh_per_year * _carbon.grid_ci_g_per_kwh / 1000.0], marker_color=COLORS["OrangeLine"], opacity=0.9))
        _fig.add_trace(go.Bar(name="Explanations", x=["Responsible stack"], y=[_carbon.explanation_kwh_per_year * _carbon.grid_ci_g_per_kwh / 1000.0], marker_color=COLORS["RedLine"], opacity=0.9))
        _fig.update_layout(
            barmode="stack",
            height=340,
            yaxis=dict(title="Annual carbon (kg CO2)", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=60, b=40),
        )
        apply_plotly_theme(_fig)
        items.append(mo.as_html(_fig))

        items.append(mo.Html(f"""
        <div style="display:flex; gap:14px; flex-wrap:wrap; margin:16px 0;">
            {_metric_card("Baseline", f"{_carbon.baseline_kgco2_per_year:.1f} kg", "train once + serving", COLORS["GreenLine"])}
            {_metric_card("Responsible Stack", f"{_carbon.total_kgco2_per_year:.1f} kg", "retrain + serve + explain", COLORS["RedLine"], True)}
            {_metric_card("Multiplier", f"{_carbon.carbon_multiplier:.1f}x", f"{_carbon.retrain_frequency_per_year} retrains/year", COLORS["OrangeLine"])}
            {_metric_card("Grid", f"{_carbon.grid_ci_g_per_kwh:.0f}", "gCO2/kWh", COLORS["BlueLine"])}
        </div>
        """))

        items.append(mo.md(f"""
**Carbon Table - Live Values**

| Quantity | Value |
|---|---:|
| Training energy per run | {_carbon.train_energy_kwh:.1f} kWh |
| Retraining energy / year | {_carbon.retraining_kwh_per_year:.1f} kWh |
| Base serving energy / year | {_carbon.base_serving_kwh_per_year:.3f} kWh |
| Explanation energy / year | {_carbon.explanation_kwh_per_year:.3f} kWh |
| Total energy / year | {_carbon.total_kwh_per_year:.1f} kWh |
| Baseline carbon / year | {_carbon.baseline_kgco2_per_year:.1f} kg CO2 |
| Responsible carbon / year | {_carbon.total_kgco2_per_year:.1f} kg CO2 |

*Source: `mlsysbook_labs.carbon_budget`; hardware TDP from `{v1_15_resp.hardware_ref}`.*
        """))

        if partD_pred.value == "cadence":
            items.append(mo.callout(mo.md("**Correct.** Retraining cadence, explanation coverage, and grid intensity are all design knobs."), kind="success"))
        else:
            items.append(mo.callout(mo.md(
                "**Carbon is part of the responsible design budget.** It does not replace the harmed-party analysis, but it must be reported."
            ), kind="warn"))
        return mo.vstack(items)

    def build_synthesis():
        return mo.vstack([
            mo.md("## Key Takeaways"),
            mo.callout(mo.md(
                f"**1. Responsibility starts with a named harmed party.** For {v1_15_resp.label}, that is {v1_15_resp.harmed_party}."
            ), kind="info"),
            mo.callout(mo.md(
                f"**2. A responsible system has evidence.** The audit signal is {v1_15_resp.audit_signal}."
            ), kind="info"),
            mo.callout(mo.md(
                "**3. There is no free fairness.** Privacy, explanations, robustness, monitoring, and carbon all create measurable system overhead."
            ), kind="info"),
            mo.Html(f"""
            <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Responsible Decision Memo
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        Submit the track, harmed party, obligation, metric conflict,
                        control budget, explanation policy, carbon budget, audit
                        signal, and residual harm.
                    </div>
                </div>
                <div style="flex: 1; min-width: 280px; background: white;
                            border: 1px solid {COLORS['Border']}; border-radius: 12px;
                            padding: 20px 24px;">
                    <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                                text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                        Next Lab
                    </div>
                    <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                        Lab 16 combines the Volume I constraints into one deployment
                        decision. The responsibility policy becomes one of the binding
                        constraints rather than a separate checklist.
                    </div>
                </div>
            </div>
            """),
        ])

    _tabs = mo.ui.tabs({
        "Part A: Metric Conflict": build_part_a(),
        "Part B: Responsibility Budget": build_part_b(),
        "Part C: Explainability Tax": build_part_c(),
        "Part D: Carbon Ledger": build_part_d(),
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
    v1_15_profile,
    v1_15_resp,
    v1_15_variant,
):
    if partA_pred.value is not None and partB_pred.value is not None and partC_pred.value is not None and partD_pred.value is not None:
        ledger.save(chapter=15, design={
            "chapter": "v1_15",
            "track_id": v1_15_profile.track_id,
            "scenario_id": v1_15_variant.scenario_id,
            "hardware_ref": v1_15_resp.hardware_ref,
            "model_ref": v1_15_resp.model_ref,
            "harmed_party": v1_15_resp.harmed_party,
            "obligation": v1_15_resp.obligation,
            "audit_signal": v1_15_resp.audit_signal,
            "completed": True,
            "metric_conflict_prediction": partA_pred.value,
            "budget_prediction": partB_pred.value,
            "explainability_prediction": partC_pred.value,
            "carbon_prediction": partD_pred.value,
        })

    mo.Html(f"""
    <div class="lab-hud">
        <span class="hud-label">LAB</span>
        <span class="hud-value">15 &middot; Responsible Engineering</span>
        <span class="hud-label">TRACK</span>
        <span class="hud-value">{v1_15_profile.label}</span>
        <span style="flex:1;"></span>
        <span class="hud-label">OBLIGATION</span>
        <span class="hud-value">{v1_15_resp.obligation}</span>
        <span class="hud-label">STATUS</span>
        <span class="hud-active">ACTIVE</span>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(
    build_lab_report,
    carbon_budget,
    explanation_overhead,
    metric_conflict,
    mo,
    partA_base_a,
    partA_base_b,
    partA_pred,
    partA_threshold,
    partB_explain,
    partB_monitoring,
    partB_pred,
    partB_privacy,
    partB_robustness,
    partC_coverage,
    partC_features,
    partC_method,
    partC_pred,
    partD_explain,
    partD_grid_ci,
    partD_pred,
    partD_retrains,
    report_export_panel,
    responsibility_budget,
    v1_15_metadata,
    v1_15_profile,
    v1_15_resp,
    v1_15_variant,
):
    _conflict = metric_conflict(
        v1_15_resp,
        base_rate_a_pct=partA_base_a.value,
        base_rate_b_pct=partA_base_b.value,
        threshold=partA_threshold.value,
    )
    _budget = responsibility_budget(
        v1_15_resp,
        privacy_level=partB_privacy.value,
        explanation_coverage_pct=partB_explain.value,
        robustness_level=partB_robustness.value,
        monitoring_level=partB_monitoring.value,
    )
    _explain = explanation_overhead(
        v1_15_resp,
        method=partC_method.value,
        features=partC_features.value,
        coverage_pct=partC_coverage.value,
    )
    _carbon = carbon_budget(
        v1_15_resp,
        retrain_frequency_per_year=partD_retrains.value,
        explanation_coverage_pct=partD_explain.value,
        grid_ci_g_per_kwh=partD_grid_ci.value,
    )

    _incomplete = []
    if partA_pred.value is None:
        _incomplete.append("Part A metric conflict prediction")
    if partB_pred.value is None:
        _incomplete.append("Part B responsibility budget prediction")
    if partC_pred.value is None:
        _incomplete.append("Part C explainability prediction")
    if partD_pred.value is None:
        _incomplete.append("Part D carbon prediction")

    _report = build_lab_report(
        v1_15_metadata,
        track=v1_15_profile.label,
        scenario=v1_15_variant.workload_summary,
        learning_objectives=(
            "Name the harmed party, obligation, and audit signal for the selected track.",
            "Quantify subgroup or context metric conflict before accepting aggregate quality.",
            "Budget responsibility controls as latency, energy, cost, quality, governance, and carbon overhead.",
        ),
        predictions={
            "metric_conflict": partA_pred.value,
            "responsibility_budget": partB_pred.value,
            "explainability_tax": partC_pred.value,
            "carbon_ledger": partD_pred.value,
        },
        knob_settings={
            "base_rate_a_pct": partA_base_a.value,
            "base_rate_b_pct": partA_base_b.value,
            "threshold": partA_threshold.value,
            "privacy_level": partB_privacy.value,
            "explanation_coverage_budget_pct": partB_explain.value,
            "robustness_level": partB_robustness.value,
            "monitoring_level": partB_monitoring.value,
            "explanation_method": partC_method.value,
            "explanation_features": partC_features.value,
            "explanation_coverage_online_pct": partC_coverage.value,
            "retrains_per_year": partD_retrains.value,
            "explanation_coverage_carbon_pct": partD_explain.value,
            "grid_ci_g_per_kwh": partD_grid_ci.value,
        },
        evidence_summary={
            "harmed_party": v1_15_resp.harmed_party,
            "obligation": v1_15_resp.obligation,
            "audit_signal": v1_15_resp.audit_signal,
            "hardware_ref": v1_15_resp.hardware_ref,
            "model_ref": v1_15_resp.model_ref,
            "fpr_gap_pp": round(_conflict.fpr_gap_pp, 3),
            "target_gap_pp": v1_15_resp.target_gap_pp,
            "responsibility_budget_feasible": _budget.feasible,
            "budget_violations": _budget.violations,
            "explanation_total_latency_ms": round(_explain.total_latency_ms, 3),
            "explanation_slo_ok": _explain.slo_ok,
            "annual_carbon_kgco2": round(_carbon.total_kgco2_per_year, 3),
            "carbon_multiplier": round(_carbon.carbon_multiplier, 3),
        },
        final_decision=(
            f"Protect {v1_15_resp.harmed_party} by treating {v1_15_resp.obligation} "
            f"as a system constraint, auditing with {v1_15_resp.audit_signal}, "
            "and reporting residual harm explicitly."
        ),
        big_takeaways=(
            "A high aggregate metric can hide the harmed party.",
            "Responsibility controls have measurable latency, energy, cost, quality, governance, and carbon overhead.",
            "The report artifact is a decision memo, not a checklist of slogans.",
        ),
        reflections={
            "report_artifact": v1_15_resp.report_artifact,
            "validation_tests": v1_15_resp.validation_tests,
            "residual_harm_owner": v1_15_resp.residual_harm,
        },
        residual_risk=(
            f"{v1_15_resp.residual_harm} Teaching estimates must be validated with "
            "track-specific audits, representative cohorts, and deployment evidence."
        ),
        source_trace={
            "track_id": v1_15_profile.track_id,
            "scenario_id": v1_15_variant.scenario_id,
            "hardware_ref": v1_15_variant.hardware_ref,
            "model_ref": v1_15_variant.model_ref,
            "shared_helper": "mlsysbook_labs.responsibility",
            "source_policy": v1_15_profile.source_policy,
        },
        result_snapshot={
            "responsibility_profile": v1_15_resp,
            "metric_conflict": _conflict,
            "responsibility_budget": _budget,
            "explanation_overhead": _explain,
            "carbon_budget": _carbon,
        },
        incomplete_fields=tuple(_incomplete),
    )

    mo.vstack([
        mo.md("## Download Report"),
        mo.callout(
            mo.md(
                "This V1-15 report is generated locally from the selected track, MLSysIM hardware/model refs, "
                "and shared `mlsysbook_labs.responsibility` calculations."
            ),
            kind="info",
        ),
        report_export_panel(_report),
    ])
    return


if __name__ == "__main__":
    app.run()
