"""Reusable track-aware system-design helpers for remaining Volume II labs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import LabTrackVariant, TrackProfile


@dataclass(frozen=True)
class SystemDesignOption:
    option_id: str
    label: str
    emphasis: str
    capacity: float
    base_load: float
    base_latency_ms: float
    base_cost: float
    quality_pct: float
    guardrail_pct: float
    mitigation: str
    validation_requirement: str
    residual_risk: str


@dataclass(frozen=True)
class SystemDesignProfile:
    lab_id: str
    title: str
    track_id: str
    label: str
    hardware_ref: str
    hardware_name: str
    model_ref: str
    model_name: str
    stakeholder: str
    concept_label: str
    decision_story: str
    knob_label: str
    knob_unit: str
    default_knob: float
    knob_min: float
    knob_max: float
    knob_step: float
    capacity_budget: float
    latency_budget_ms: float
    cost_budget: float
    quality_floor_pct: float
    guardrail_floor_pct: float
    decision_options: tuple[SystemDesignOption, ...]
    validation_tests: tuple[str, ...]
    report_artifact: str
    primary_metric: str
    guardrail_metric: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class SystemOptionResult:
    option_id: str
    option_label: str
    knob_value: float
    load: float
    capacity: float
    stress_ratio: float
    latency_ms: float
    cost: float
    quality_pct: float
    guardrail_pct: float
    dominant_risk: str
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class SystemCurvePoint:
    knob_value: float
    stress_ratio: float
    latency_ms: float
    cost: float
    feasible: bool


@dataclass(frozen=True)
class SystemCurveResult:
    option_id: str
    option_label: str
    points: tuple[SystemCurvePoint, ...]
    first_failure: float | None


@dataclass(frozen=True)
class SystemDecisionResult:
    selected_id: str
    selected_label: str
    feasible: bool
    dominant_risk: str
    stress_ratio: float
    mitigation: str
    validation_requirement: str
    residual_risk: str
    rejected_alternatives: tuple[str, ...]
    memo_summary: str


@dataclass(frozen=True)
class SystemLedgerDecision:
    step_id: int
    chapter: str
    track_id: str
    scenario_id: str
    selected_option: str
    dominant_risk: str


@dataclass(frozen=True)
class SystemDesignContext:
    metadata: Any
    track: TrackProfile
    variant: LabTrackVariant
    hardware: Any
    model: Any
    profile: SystemDesignProfile


@dataclass(frozen=True)
class SystemDesignControls:
    prediction: Any
    knob: Any
    choice: Any
    reflection: Any


def _quantity_to_float(value: Any, unit: str, default: float) -> float:
    if value is None:
        return default
    if hasattr(value, "m_as"):
        try:
            return float(value.m_as(unit))
        except Exception:
            return default
    if hasattr(value, "to"):
        try:
            return float(value.to(unit).magnitude)
        except Exception:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    if value:
        return (str(value),)
    return ()


def _options(defaults: Mapping[str, Any]) -> tuple[SystemDesignOption, ...]:
    raw = defaults.get("decision_options", {})
    if not isinstance(raw, Mapping):
        raw = {}
    options = []
    for option_id, details_raw in raw.items():
        details = details_raw if isinstance(details_raw, Mapping) else {}
        options.append(
            SystemDesignOption(
                option_id=str(option_id),
                label=str(details.get("label", option_id)),
                emphasis=str(details.get("emphasis", "decision emphasis not specified")),
                capacity=float(details.get("capacity", defaults.get("capacity_budget", 1.0))),
                base_load=float(details.get("base_load", defaults.get("default_knob", 1.0))),
                base_latency_ms=float(details.get("base_latency_ms", defaults.get("latency_budget_ms", 100.0))),
                base_cost=float(details.get("base_cost", defaults.get("cost_budget", 100.0) * 0.5)),
                quality_pct=float(details.get("quality_pct", defaults.get("quality_floor_pct", 80.0))),
                guardrail_pct=float(details.get("guardrail_pct", defaults.get("guardrail_floor_pct", 80.0))),
                mitigation=str(details.get("mitigation", "mitigation not specified")),
                validation_requirement=str(details.get("validation_requirement", "validation not specified")),
                residual_risk=str(details.get("residual_risk", "residual risk not specified")),
            )
        )
    if options:
        return tuple(options)
    return (
        SystemDesignOption(
            "baseline",
            "Baseline option",
            "baseline",
            float(defaults.get("capacity_budget", 1.0)),
            float(defaults.get("default_knob", 1.0)),
            float(defaults.get("latency_budget_ms", 100.0)),
            float(defaults.get("cost_budget", 100.0) * 0.5),
            float(defaults.get("quality_floor_pct", 80.0)),
            float(defaults.get("guardrail_floor_pct", 80.0)),
            "baseline mitigation",
            "baseline validation",
            "baseline residual risk",
        ),
    )


def system_design_profile(
    profile: TrackProfile,
    variant: LabTrackVariant,
    hardware: Any,
    model: Any,
    *,
    title: str,
) -> SystemDesignProfile:
    """Build a generic track-aware design profile from typed variant defaults."""
    defaults = variant.defaults
    return SystemDesignProfile(
        lab_id=variant.lab_id,
        title=title,
        track_id=profile.track_id,
        label=profile.label,
        hardware_ref=variant.hardware_ref,
        hardware_name=getattr(hardware, "name", variant.hardware_ref),
        model_ref=variant.model_ref,
        model_name=getattr(model, "name", variant.model_ref),
        stakeholder=variant.stakeholder,
        concept_label=str(defaults.get("concept_label", title)),
        decision_story=str(defaults.get("decision_story", variant.workload_summary)),
        knob_label=str(defaults.get("knob_label", "scale")),
        knob_unit=str(defaults.get("knob_unit", "units")),
        default_knob=float(defaults.get("default_knob", 1.0)),
        knob_min=float(defaults.get("knob_min", 0.5)),
        knob_max=float(defaults.get("knob_max", 2.0)),
        knob_step=float(defaults.get("knob_step", 0.1)),
        capacity_budget=float(defaults.get("capacity_budget", 1.0)),
        latency_budget_ms=float(defaults.get("latency_budget_ms", 100.0)),
        cost_budget=float(defaults.get("cost_budget", 100.0)),
        quality_floor_pct=float(defaults.get("quality_floor_pct", 80.0)),
        guardrail_floor_pct=float(defaults.get("guardrail_floor_pct", 80.0)),
        decision_options=_options(defaults),
        validation_tests=_tuple_str(defaults.get("validation_tests")),
        report_artifact=str(variant.assumptions.get("report_artifact", f"{title} decision memo")),
        primary_metric=variant.primary_metric,
        guardrail_metric=variant.guardrail_metric,
        source_refs=tuple(ref for ref in (variant.hardware_ref, variant.model_ref, variant.system_ref) if ref),
    )


def system_design_context(*, lab_path: str, track_id: str) -> SystemDesignContext:
    """Resolve catalog metadata, track variant, and registry refs for a Volume II lab."""
    from .catalog import get_lab_metadata
    from .registry_refs import resolve_mlsysim_ref
    from .tracks import get_track_profile
    from .variants import get_lab_track_variant

    metadata = get_lab_metadata(lab_path)
    track = get_track_profile(track_id)
    variant = get_lab_track_variant(metadata.lab_id, track.track_id)
    hardware = resolve_mlsysim_ref(variant.hardware_ref)
    model = resolve_mlsysim_ref(variant.model_ref)
    profile = system_design_profile(track, variant, hardware, model, title=metadata.title)
    return SystemDesignContext(
        metadata=metadata,
        track=track,
        variant=variant,
        hardware=hardware,
        model=model,
        profile=profile,
    )


def system_design_controls(mo: Any, profile: SystemDesignProfile) -> SystemDesignControls:
    """Create UI controls without reading their values in the same Marimo cell."""
    prediction = mo.ui.radio(
        options={
            "Capacity stress binds first": "capacity stress",
            "Latency or freshness binds first": "latency",
            "Cost or utilization binds first": "cost",
            "Quality or guardrail binds first": "guardrail",
        },
        label=f"{profile.label}: which pressure do you expect to dominate?",
    )
    knob = mo.ui.slider(
        start=profile.knob_min,
        stop=profile.knob_max,
        value=profile.default_knob,
        step=profile.knob_step,
        label=f"{profile.knob_label} ({profile.knob_unit})",
    )
    options = {option.label: option.option_id for option in profile.decision_options}
    choice = mo.ui.dropdown(options=options, value=profile.decision_options[0].label, label="Decision option")
    reflection = mo.ui.text_area(
        label="Reflection",
        placeholder="Defend the selected option, rejected alternatives, and residual risk.",
        full_width=True,
    )
    return SystemDesignControls(
        prediction=prediction,
        knob=knob,
        choice=choice,
        reflection=reflection,
    )


def _option(profile: SystemDesignProfile, option_id: str) -> SystemDesignOption:
    return next((item for item in profile.decision_options if item.option_id == option_id), profile.decision_options[0])


def system_option_result(
    profile: SystemDesignProfile,
    *,
    option_id: str,
    knob_value: float | None = None,
) -> SystemOptionResult:
    """Evaluate one design option at a selected scale/load knob."""
    option = _option(profile, option_id)
    knob = profile.default_knob if knob_value is None else float(knob_value)
    knob = max(profile.knob_min, min(profile.knob_max, knob))
    scale_ratio = knob / max(profile.default_knob, 1e-9)
    load = option.base_load * scale_ratio
    stress = load / max(option.capacity, 1e-9)
    overload = max(0.0, stress - 1.0)
    latency = option.base_latency_ms * (1.0 + overload * 1.8)
    cost = option.base_cost * (0.5 + 0.5 * scale_ratio)
    quality = max(0.0, option.quality_pct - overload * 12.0)
    guardrail = max(0.0, option.guardrail_pct - overload * 18.0)
    ratios = {
        "capacity stress": stress,
        "latency": latency / max(profile.latency_budget_ms, 1e-9),
        "cost": cost / max(profile.cost_budget, 1e-9),
        "quality": profile.quality_floor_pct / max(quality, 1e-9),
        "guardrail": profile.guardrail_floor_pct / max(guardrail, 1e-9),
    }
    dominant = max(ratios, key=ratios.get)
    violations = []
    if stress > 1.0:
        violations.append(f"load/capacity stress {stress:.2f} > 1.00")
    if latency > profile.latency_budget_ms:
        violations.append(f"latency {latency:.2f} ms > {profile.latency_budget_ms:.2f} ms")
    if cost > profile.cost_budget:
        violations.append(f"cost {cost:.2f} > {profile.cost_budget:.2f}")
    if quality < profile.quality_floor_pct:
        violations.append(f"quality {quality:.1f}% < {profile.quality_floor_pct:.1f}%")
    if guardrail < profile.guardrail_floor_pct:
        violations.append(f"guardrail {guardrail:.1f}% < {profile.guardrail_floor_pct:.1f}%")
    return SystemOptionResult(
        option_id=option.option_id,
        option_label=option.label,
        knob_value=knob,
        load=load,
        capacity=option.capacity,
        stress_ratio=stress,
        latency_ms=latency,
        cost=cost,
        quality_pct=quality,
        guardrail_pct=guardrail,
        dominant_risk=dominant,
        feasible=not violations,
        violations=tuple(violations),
    )


def system_frontier(profile: SystemDesignProfile, *, knob_value: float | None = None) -> tuple[SystemOptionResult, ...]:
    """Evaluate every option at the selected scale/load knob."""
    return tuple(
        system_option_result(profile, option_id=option.option_id, knob_value=knob_value)
        for option in profile.decision_options
    )


def system_curve(
    profile: SystemDesignProfile,
    *,
    option_id: str,
    samples: int = 24,
) -> SystemCurveResult:
    """Sweep the lab knob for a selected option."""
    count = max(2, int(samples))
    span = profile.knob_max - profile.knob_min
    values = tuple(profile.knob_min + span * idx / (count - 1) for idx in range(count))
    points = tuple(
        SystemCurvePoint(
            knob_value=result.knob_value,
            stress_ratio=result.stress_ratio,
            latency_ms=result.latency_ms,
            cost=result.cost,
            feasible=result.feasible,
        )
        for result in (
            system_option_result(profile, option_id=option_id, knob_value=value)
            for value in values
        )
    )
    first_failure = next((point.knob_value for point in points if not point.feasible), None)
    return SystemCurveResult(
        option_id=option_id,
        option_label=_option(profile, option_id).label,
        points=points,
        first_failure=first_failure,
    )


def system_decision(
    profile: SystemDesignProfile,
    *,
    option_id: str,
    knob_value: float | None = None,
) -> SystemDecisionResult:
    """Return decision memo fields for a selected option."""
    option = _option(profile, option_id)
    selected = system_option_result(profile, option_id=option.option_id, knob_value=knob_value)
    rejected = tuple(
        f"{item.option_label}: {item.dominant_risk}; {'feasible' if item.feasible else 'not feasible'}"
        for item in system_frontier(profile, knob_value=knob_value)
        if item.option_id != option.option_id
    )
    summary = (
        f"Choose {option.label} for {profile.label}; dominant risk is "
        f"{selected.dominant_risk}, mitigation is {option.mitigation}."
    )
    return SystemDecisionResult(
        selected_id=option.option_id,
        selected_label=option.label,
        feasible=selected.feasible,
        dominant_risk=selected.dominant_risk,
        stress_ratio=selected.stress_ratio,
        mitigation=option.mitigation,
        validation_requirement=option.validation_requirement,
        residual_risk=option.residual_risk,
        rejected_alternatives=rejected,
        memo_summary=summary,
    )


def system_design_ledger_summary(
    ledger: Any,
    *,
    prefix: str = "v2_",
    upto: int | None = None,
) -> tuple[SystemLedgerDecision, ...]:
    """Collect prior design decisions for capstone-style synthesis panels."""
    state = getattr(ledger, "_state", None)
    history = getattr(state, "history", {}) if state is not None else {}
    if not isinstance(history, Mapping):
        return ()

    decisions: list[SystemLedgerDecision] = []
    for raw_step, raw_design in sorted(history.items(), key=lambda item: str(item[0])):
        try:
            step_id = int(raw_step)
        except (TypeError, ValueError):
            continue
        if upto is not None and step_id >= upto:
            continue
        if not isinstance(raw_design, Mapping):
            continue
        chapter = str(raw_design.get("chapter", ""))
        if not chapter.startswith(prefix):
            continue
        decisions.append(
            SystemLedgerDecision(
                step_id=step_id,
                chapter=chapter,
                track_id=str(raw_design.get("track_id", "")),
                scenario_id=str(raw_design.get("scenario_id", "")),
                selected_option=str(raw_design.get("selected_option", "")),
                dominant_risk=str(raw_design.get("dominant_risk", "")),
            )
        )
    return tuple(decisions)


def render_system_design_lab(
    *,
    mo: Any,
    go: Any,
    apply_plotly_theme: Any,
    colors: Mapping[str, str],
    chapter: int,
    ledger: Any,
    context: SystemDesignContext,
    controls: SystemDesignControls,
    track_picker: Any,
) -> Any:
    """Render a compact track-aware Volume II decision lab."""
    from .reports import build_lab_report, report_export_panel
    from .style import ACADEMIC_LAB_CSS
    from .ui import source_trace, track_arc_context, track_context
    from mlsysim.labs.style import LAB_CSS

    metadata = context.metadata
    track = context.track
    variant = context.variant
    profile = context.profile
    prediction = controls.prediction
    knob = controls.knob
    choice = controls.choice
    reflection = controls.reflection

    frontier = system_frontier(profile, knob_value=knob.value)
    curve = system_curve(profile, option_id=choice.value)
    selected = system_option_result(profile, option_id=choice.value, knob_value=knob.value)
    decision = system_decision(profile, option_id=choice.value, knob_value=knob.value)

    frontier_fig = go.Figure()
    frontier_fig.add_trace(go.Scatter(
        x=[row.cost for row in frontier],
        y=[row.stress_ratio for row in frontier],
        mode="markers+text",
        text=[row.option_label for row in frontier],
        textposition="top center",
        marker=dict(
            size=[14 + min(row.quality_pct, 100) / 10 for row in frontier],
            color=[colors["GreenLine"] if row.feasible else colors["RedLine"] for row in frontier],
            line=dict(color="#ffffff", width=1.5),
        ),
    ))
    frontier_fig.add_hline(y=1.0, line_dash="dash", line_color=colors["RedLine"], annotation_text="capacity limit")
    frontier_fig.add_vline(x=profile.cost_budget, line_dash="dash", line_color=colors["OrangeLine"], annotation_text="cost budget")
    frontier_fig.update_layout(
        height=340,
        xaxis=dict(title="Cost", gridcolor="#f1f5f9"),
        yaxis=dict(title="Stress ratio", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=60),
    )
    apply_plotly_theme(frontier_fig)

    curve_fig = go.Figure()
    curve_fig.add_trace(go.Scatter(
        x=[point.knob_value for point in curve.points],
        y=[point.stress_ratio for point in curve.points],
        mode="lines+markers",
        marker=dict(
            color=[colors["GreenLine"] if point.feasible else colors["RedLine"] for point in curve.points],
            size=7,
        ),
        line=dict(color=colors["BlueLine"], width=2.5),
        name=curve.option_label,
    ))
    curve_fig.add_hline(y=1.0, line_dash="dash", line_color=colors["RedLine"], annotation_text="first wall")
    curve_fig.update_layout(
        height=320,
        xaxis=dict(title=f"{profile.knob_label} ({profile.knob_unit})", gridcolor="#f1f5f9"),
        yaxis=dict(title="Stress ratio", gridcolor="#f1f5f9"),
        margin=dict(l=60, r=20, t=35, b=50),
    )
    apply_plotly_theme(curve_fig)

    frontier_rows = "".join(
        f"""
        <tr>
          <td>{row.option_label}</td>
          <td>{row.stress_ratio:.2f}</td>
          <td>{row.latency_ms:.2f} ms</td>
          <td>{row.cost:.2f}</td>
          <td>{row.quality_pct:.1f}%</td>
          <td>{row.guardrail_pct:.1f}%</td>
          <td>{'yes' if row.feasible else 'no - violation'}</td>
          <td>{row.dominant_risk}</td>
        </tr>
        """
        for row in frontier
    )
    rejected_items = "".join(f"<li>{item}</li>" for item in decision.rejected_alternatives)
    validation_items = "".join(f"<li>{item}</li>" for item in profile.validation_tests)

    if prediction.value is not None:
        ledger.save(chapter=chapter, design={
            "chapter": f"v2_{chapter:02d}",
            "track_id": track.track_id,
            "scenario_id": variant.scenario_id,
            "hardware_ref": profile.hardware_ref,
            "model_ref": profile.model_ref,
            "completed": True,
            "prediction": prediction.value,
            "selected_option": decision.selected_id,
            "dominant_risk": decision.dominant_risk,
            "stress_ratio": decision.stress_ratio,
        })

    prior_v2_decisions = system_design_ledger_summary(ledger, prefix="v2_", upto=chapter)
    incomplete = []
    if prediction.value is None:
        incomplete.append("Part A dominant-pressure prediction")
    if not str(reflection.value or "").strip():
        incomplete.append("Part C reflection")

    report = build_lab_report(
        metadata,
        track=track.label,
        scenario=variant.workload_summary,
        learning_objectives=(
            f"Connect {profile.concept_label} to the selected track's system constraints.",
            "Compare decision options against capacity, latency, cost, quality, and guardrail budgets.",
            "Choose a mitigation, state rejected alternatives, and name residual risk.",
        ),
        predictions={"dominant_pressure": prediction.value},
        knob_settings={"knob_value": knob.value, "selected_option": decision.selected_id},
        evidence_summary={
            "hardware_ref": profile.hardware_ref,
            "model_ref": profile.model_ref,
            "stress_ratio": selected.stress_ratio,
            "latency_ms": selected.latency_ms,
            "cost": selected.cost,
            "dominant_risk": decision.dominant_risk,
            "mitigation": decision.mitigation,
            "prior_v2_decisions": tuple(item.__dict__ for item in prior_v2_decisions),
        },
        final_decision=decision.memo_summary,
        big_takeaways=(
            "The same Volume II concept produces different system limits by track.",
            "A local optimum can violate the selected track's guardrail.",
            "A defensible decision includes mitigation, validation, and residual risk.",
        ),
        reflections={
            "student_reflection": reflection.value,
            "validation_requirement": decision.validation_requirement,
            "report_artifact": profile.report_artifact,
        },
        residual_risk=decision.residual_risk,
        source_trace={
            "track_id": track.track_id,
            "scenario_id": variant.scenario_id,
            "hardware_ref": variant.hardware_ref,
            "model_ref": variant.model_ref,
            "shared_helper": "mlsysbook_labs.system_design",
            "source_policy": track.source_policy,
        },
        result_snapshot={
            "profile": profile,
            "frontier": frontier,
            "curve": curve,
            "selected": selected,
            "decision": decision,
            "prior_v2_decisions": prior_v2_decisions,
        },
        incomplete_fields=tuple(incomplete),
    )

    if chapter == 17:
        if prior_v2_decisions:
            prior_rows = "".join(
                f"""
                <tr>
                  <td>{item.chapter}</td>
                  <td>{item.track_id}</td>
                  <td>{item.selected_option or 'not recorded'}</td>
                  <td>{item.dominant_risk or 'not recorded'}</td>
                </tr>
                """
                for item in prior_v2_decisions
            )
        else:
            prior_rows = """
                <tr>
                  <td colspan="4">No prior Volume II ledger entries found locally. The capstone can still be completed from the current evidence.</td>
                </tr>
            """
        capstone_panel = mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Prior Volume II Decisions</h2>
          <table class="mlsysbook-table">
            <thead><tr><th>Chapter</th><th>Track</th><th>Selected option</th><th>Dominant risk</th></tr></thead>
            <tbody>{prior_rows}</tbody>
          </table>
        </div>
        """)
    else:
        capstone_panel = mo.md("")

    return mo.vstack([
        LAB_CSS,
        ACADEMIC_LAB_CSS,
        mo.Html(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0c1a2e 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.35);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #94a3b8; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems &middot; Volume II &middot; Lab {chapter:02d}
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1;">
                {metadata.title}
            </h1>
            <p style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600;
                      color: #94a3b8; letter-spacing: 0.04em; font-family: 'SF Mono', monospace;">
                {profile.concept_label} &middot; Track Decision &middot; Residual Risk
            </p>
            <p style="margin: 0 0 22px 0; font-size: 1.0rem; color: #cbd5e1;
                      max-width: 860px; line-height: 1.65;">
                {variant.workload_summary}
            </p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span class="badge badge-info">{track.label}</span>
                <span class="badge badge-warn">{profile.knob_label}</span>
                <span class="badge badge-fail">{profile.report_artifact}</span>
            </div>
        </div>
        """),
        track_picker,
        track_context(track),
        track_arc_context(track, metadata.lab_id),
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Learning Objectives</h2>
          <ul class="mlsysbook-list">
            <li>Connect {profile.concept_label} to the selected track's capacity, latency, cost, and guardrail limits.</li>
            <li>Compare options with a table and frontier rather than relying on one aggregate metric.</li>
            <li>Recommend a mitigation and state validation evidence plus residual risk.</li>
          </ul>
          <div class="mlsysbook-callout"><strong>Track story:</strong> {profile.decision_story}</div>
        </div>
        """),
        mo.ui.tabs({
            "Part A · Frontier": mo.vstack([
                mo.Html('<div class="mlsysbook-panel mlsysbook-nugget"><div class="mlsysbook-part-title"><h2>Part A: Decision Frontier</h2></div></div>'),
                prediction,
                knob,
                mo.as_html(frontier_fig),
                mo.Html(f"""
                <div class="mlsysbook-panel">
                  <h2>Table Fallback</h2>
                  <table class="mlsysbook-table">
                    <thead>
                      <tr><th>Option</th><th>Stress</th><th>Latency</th><th>Cost</th><th>Quality</th><th>Guardrail</th><th>Feasible</th><th>Dominant risk</th></tr>
                    </thead>
                    <tbody>{frontier_rows}</tbody>
                  </table>
                </div>
                """),
            ]),
            "Part B · Scaling": mo.vstack([
                mo.Html(f"""
                <div class="mlsysbook-panel mlsysbook-nugget">
                  <div class="mlsysbook-part-title"><h2>Part B: Scaling Boundary</h2></div>
                  <div class="mlsysbook-callout"><strong>Failure boundary:</strong>
                    {curve.first_failure if curve.first_failure is not None else 'not reached'} {profile.knob_unit}</div>
                </div>
                """),
                mo.as_html(curve_fig),
            ]),
            "Part C · Decision": mo.vstack([
                mo.Html('<div class="mlsysbook-panel mlsysbook-nugget"><div class="mlsysbook-part-title"><h2>Part C: Decision Memo</h2></div></div>'),
                choice,
                mo.Html(f"""
                <div class="mlsysbook-panel">
                  <h2>Computed Evidence</h2>
                  <div class="mlsysbook-grid">
                    <div class="mlsysbook-field"><strong>Selected option</strong>{decision.selected_label}</div>
                    <div class="mlsysbook-field"><strong>Feasible</strong>{'yes' if decision.feasible else 'no - violation'}</div>
                    <div class="mlsysbook-field"><strong>Stress ratio</strong>{decision.stress_ratio:.2f}</div>
                    <div class="mlsysbook-field"><strong>Dominant risk</strong>{decision.dominant_risk}</div>
                    <div class="mlsysbook-field"><strong>Mitigation</strong>{decision.mitigation}</div>
                    <div class="mlsysbook-field"><strong>Validation</strong>{decision.validation_requirement}</div>
                  </div>
                  <div class="mlsysbook-callout"><strong>Memo decision:</strong> {decision.memo_summary}</div>
                </div>
                """),
                mo.Html(f"""
                <div class="mlsysbook-panel">
                  <h2>Rejected Alternatives</h2>
                  <ul class="mlsysbook-list">{rejected_items}</ul>
                  <h2>Validation Tests</h2>
                  <ul class="mlsysbook-list">{validation_items}</ul>
                </div>
                """),
                reflection,
            ]),
        }),
        capstone_panel,
        mo.Html(f"""
        <div class="mlsysbook-panel">
          <h2>Synthesis</h2>
          <div class="mlsysbook-grid">
            <div class="mlsysbook-field"><strong>Track</strong>{track.label}</div>
            <div class="mlsysbook-field"><strong>Selected option</strong>{decision.selected_label}</div>
            <div class="mlsysbook-field"><strong>Dominant risk</strong>{decision.dominant_risk}</div>
            <div class="mlsysbook-field"><strong>Residual risk</strong>{decision.residual_risk}</div>
          </div>
        </div>
        """),
        mo.Html("""
        <div class="mlsysbook-panel">
          <h2>Big Takeaways</h2>
          <ul class="mlsysbook-list">
            <li><strong>Track context changes the right answer.</strong> Mobile, tiny, edge, and cloud systems fail at different boundaries.</li>
            <li><strong>Feasibility is multi-dimensional.</strong> Capacity, latency, cost, quality, and guardrails must be checked together.</li>
            <li><strong>Reports need residual risk.</strong> A decision without validation and remaining risk is not complete.</li>
          </ul>
        </div>
        """),
        mo.md("## Download Report"),
        report_export_panel(report),
    ])


__all__ = [
    "SystemLedgerDecision",
    "SystemDesignContext",
    "SystemDesignControls",
    "SystemCurvePoint",
    "SystemCurveResult",
    "SystemDecisionResult",
    "SystemDesignOption",
    "SystemDesignProfile",
    "SystemOptionResult",
    "render_system_design_lab",
    "system_design_context",
    "system_design_controls",
    "system_design_ledger_summary",
    "system_curve",
    "system_decision",
    "system_design_profile",
    "system_frontier",
    "system_option_result",
]
