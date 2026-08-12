from __future__ import annotations

from mlsysbook_labs import (
    COMPLETION_STATES,
    big_takeaways,
    checkpoint_card,
    constraint_check,
    decision_flow,
    evidence_summary,
    lab_map,
    learning_objectives,
    part_header,
    scenario_slice,
    scenario_thread,
    track_selector,
    track_arc_context,
    source_trace,
    what_you_need_to_know,
)


def html_text(component) -> str:
    return component.text


def test_track_selector_accepts_canonical_and_alias_defaults():
    edge_selector = track_selector(default="edge")
    tiny_selector = track_selector(default="oura_ring")

    assert edge_selector.value == "robotaxi"
    assert tiny_selector.value == "oura_ring"


def test_lab_level_helpers_render_contract_headers():
    objectives = html_text(
        learning_objectives(
            (
                "Diagnose which constraint binds first.",
                "Defend a track-specific deployment decision.",
            )
        )
    )
    assert "Learning Objectives" in objectives
    assert "Diagnose which constraint binds first." in objectives

    thread = html_text(
        scenario_thread(
            "iPhone compression decision",
            "Keep the recipe tied to the selected workload.",
            callout="Battery is the release gate.",
        )
    )
    assert "Scenario Thread" in thread
    assert "Battery is the release gate." in thread

    flow = html_text(decision_flow("Decision path", ("Choose recipe", "Validate", "Report risk")))
    assert "Decision path" in flow
    assert "Choose recipe" in flow
    assert "Report risk" in flow

    arc = html_text(track_arc_context("iphone", "v1_10_compression_paradox"))
    assert "Where This Fits" in arc
    assert "iPhone Journey" in arc
    assert "Compression recipe" in arc
    assert "Source Trace" not in arc
    assert "MLSysIM" not in arc

    rendered_map = html_text(
        lab_map(
            (
                {"part_id": "A", "part": "A", "concept": "Bottleneck", "question": "What binds first?"},
                {"part_id": "B", "part": "B", "concept": "Frontier", "question": "What is feasible?"},
            ),
            completion={"A": "prediction_saved", "B": "decision_complete"},
        )
    )
    assert "Lab Map" in rendered_map
    assert "Part A - Bottleneck" in rendered_map
    assert "Prediction Saved" in rendered_map
    assert "Decision Complete" in rendered_map


def test_part_helpers_render_required_contract_headers():
    header = html_text(
        part_header(
            "A",
            "Compression Frontier",
            "How does bit width change the feasible set?",
            "Oura Ring treats flash as the first-order limit.",
        )
    )
    assert "Part A - Compression Frontier" in header
    assert "Systems question:" in header
    assert "Track frame:" in header

    brief = html_text(
        what_you_need_to_know(
            ("Quantization shrinks weights.", "Accuracy can drop nonlinearly."),
            equation="model_bytes = parameters * bits / 8",
            track_interpretation="Smaller models may fit flash but still miss latency.",
            watch_for="A feasible memory result can still fail the guardrail.",
        )
    )
    assert "What You Need To Know" in brief
    assert "Key relationship" in brief
    assert "Watch for" in brief

    scenario = html_text(
        scenario_slice(
            stakeholder_pressure="Product asks for local inference.",
            workload_slice="One-minute activity window.",
            active_constraint="flash",
            primary_metric="accuracy",
            guardrail_metric="latency",
        )
    )
    assert "Scenario Slice" in scenario
    assert "Stakeholder pressure" in scenario
    assert "Guardrail metric" in scenario


def test_source_constraint_checkpoint_and_takeaway_helpers():
    check = html_text(
        constraint_check(
            "SRAM",
            value=312,
            limit=256,
            unit="KB",
            status="fail",
            mitigation="Reduce activation footprint.",
        )
    )
    assert "Constraint Check" in check
    assert "Fails" in check
    assert "First mitigation:" in check

    trace = html_text(
        source_trace(
            {
                "api": "mlsysim.hardware.get_device",
                "hardware_ref": "Hardware.Tiny.OuraRing",
                "assumption": "<estimate-backed profile>",
            }
        )
    )
    assert "Source Trace" in trace
    assert "mlsysim.hardware.get_device" in trace
    assert "&lt;estimate-backed profile&gt;" in trace

    evidence = html_text(evidence_summary({"latency_ms": 18.2, "binding_constraint": "thermal"}))
    assert "Evidence Summary" in evidence
    assert "binding constraint" in evidence

    checkpoint = html_text(
        checkpoint_card(
            {
                "selected_option": "INT8",
                "primary_metric": "0.91 accuracy",
                "guardrail_metric": "18.2 ms",
                "residual_risk": "Dataset shift.",
            }
        )
    )
    assert "Checkpoint" in checkpoint
    assert "selected option" in checkpoint

    takeaways = html_text(big_takeaways(("Track constraints change the deployable answer.",)))
    assert "Big Takeaways" in takeaways
    assert "Track constraints change the deployable answer." in takeaways


def test_completion_states_match_contract_order():
    assert COMPLETION_STATES == (
        "not_started",
        "prediction_saved",
        "evidence_viewed",
        "checkpoint_saved",
        "decision_complete",
    )
