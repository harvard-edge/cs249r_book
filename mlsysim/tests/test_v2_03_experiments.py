import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_03_experiments import (
    alpha_beta_experiment,
    burst_spacing_params,
    build_traffic_matrix,
    congestion_experiment,
    default_overlap_label,
    default_payload_label,
    default_traffic_pattern_label,
    OVERLAP_WINDOW_OPTIONS,
    TRAFFIC_PATTERN_OPTIONS,
    deserialize_quantity,
    design_experiment,
    equal_budget_comparison,
    get_track_scenario,
    simulate_schedule,
    serialize_quantity,
    telemetry_experiment,
    topology_experiment,
    topology_options,
)


TRACKS = ("cloud_fleet", "iphone", "oura_ring", "robotaxi")
CANONICAL_TRACKS = ("cloud", "mobile", "tinyml", "edge")


@pytest.mark.parametrize("track_id", TRACKS)
def test_every_track_is_an_honest_fleet_scenario(track_id):
    scenario = get_track_scenario(track_id)
    assert scenario.participants >= 4
    assert any(word in scenario.fleet_unit for word in ("accelerators", "phones", "wearables", "vehicles"))
    assert scenario.default_link_id in scenario.links


@pytest.mark.parametrize("alias,canonical", tuple(zip(TRACKS, CANONICAL_TRACKS, strict=True)))
def test_track_aliases_resolve_to_stable_ledger_ids(alias, canonical):
    assert get_track_scenario(alias).track_id == canonical
    assert get_track_scenario(canonical) == get_track_scenario(alias)


def test_quantity_serialization_round_trips_units():
    original = Q_(3.5, "GB/s")
    restored = deserialize_quantity(serialize_quantity(original))
    assert restored == original


@pytest.mark.parametrize("track_id", TRACKS)
def test_alpha_beta_crosses_from_startup_to_serialization(track_id):
    scenario = get_track_scenario(track_id)
    link_id = scenario.default_link_id
    tiny = alpha_beta_experiment(track_id, link_id, Q_(1, "byte"))
    large = alpha_beta_experiment(track_id, link_id, Q_(1, "GB"))
    assert tiny["binding_term"] == "startup"
    assert large["binding_term"] == "serialization"
    assert large["total_ms"] > tiny["total_ms"]


def test_topology_changes_tail_for_identical_links_and_traffic():
    rows = {row["topology_id"]: row for row in topology_experiment("cloud_fleet")}
    assert rows["nonblocking"]["cut_capacity_mb_s"] > rows["oversubscribed"]["cut_capacity_mb_s"]
    assert rows["nonblocking"]["p95_ms"] < rows["oversubscribed"]["p95_ms"]
    assert rows["nonblocking"]["cost_usd"] > rows["oversubscribed"]["cost_usd"]


def test_traffic_matrix_is_causal_but_flow_names_are_not():
    scenario = get_track_scenario("cloud_fleet")
    link = scenario.links[scenario.default_link_id]
    topology = topology_options(16, link.bandwidth)["nonblocking"]
    incast = build_traffic_matrix(16, Q_(50, "MB"), "incast")
    neighbor = build_traffic_matrix(16, Q_(50, "MB"), "neighbor")
    renamed = tuple(
        type(flow)(f"renamed-{i}", flow.source, flow.destination, flow.payload, flow.ready_at)
        for i, flow in enumerate(incast)
    )
    incast_result = simulate_schedule(incast, topology, link)
    neighbor_result = simulate_schedule(neighbor, topology, link)
    renamed_result = simulate_schedule(renamed, topology, link)
    assert incast_result.p95_ms > neighbor_result.p95_ms
    assert renamed_result.p95_ms == pytest.approx(incast_result.p95_ms)


def test_isolation_reserves_capacity_and_has_an_adverse_consequence():
    shared = congestion_experiment("cloud_fleet", "oversubscribed", isolation_fraction=0)
    isolated = congestion_experiment("cloud_fleet", "oversubscribed", isolation_fraction=0.8)
    assert isolated["reserved_capacity_mb_s"] > 0
    assert isolated["capacity_left_for_background_mb_s"] < shared["capacity_left_for_background_mb_s"]
    assert isolated["foreground_completion_ms"] < shared["foreground_completion_ms"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_each_track_has_a_feasible_and_failed_link_case(track_id):
    scenario = get_track_scenario(track_id)
    link = scenario.links[scenario.default_link_id]
    feasible = alpha_beta_experiment(track_id, link.link_id, Q_(1, "byte"))
    guaranteed_failure_payload = link.bandwidth * scenario.communication_budget * 2
    failed = alpha_beta_experiment(track_id, link.link_id, guaranteed_failure_payload)
    assert feasible["within_budget"]
    assert not failed["within_budget"]


@pytest.mark.parametrize("fault", ("capacity", "startup", "traffic_hotspot"))
def test_counterfactual_telemetry_intervention_supports_actual_fault(fault):
    result = telemetry_experiment("cloud_fleet", "grouped", fault=fault)
    assert result["supports_diagnosis"]
    assert result["p95_improvement_ms"] > 0
    assert result["counter_to_watch"]


def test_overlap_is_bounded_by_dependency_ready_window():
    short = design_experiment("cloud_fleet", "grouped", traffic_pattern="all_to_all", overlap_window=Q_(5, "ms"))
    long = design_experiment("cloud_fleet", "grouped", traffic_pattern="all_to_all", overlap_window=Q_(50, "ms"))
    assert short["hidden_by_ready_compute_ms"] == pytest.approx(5)
    assert long["hidden_by_ready_compute_ms"] <= 50
    assert long["exposed_ms"] < short["exposed_ms"]


def test_equal_budget_comparison_counts_resources_and_reports_remainder():
    result = equal_budget_comparison(
        "cloud_fleet",
        "nonblocking",
        "grouped",
        traffic_pattern="all_to_all",
    )
    assert result["comparable"]
    assert result["first"]["cost_usd"] <= result["budget_usd"]
    assert result["second"]["cost_usd"] <= result["budget_usd"]
    assert result["first"]["unspent_usd"] >= 0
    assert result["second"]["unspent_usd"] >= 0


def test_equal_budget_capture_matches_bounded_overlap_arguments_and_is_finite_json():
    result = equal_budget_comparison(
        "cloud",
        "aligned",
        "grouped",
        traffic_pattern="incast",
        overlap_window=Q_(5, "ms"),
    )
    assert result["inputs"]["overlap_window"] == {"value": 5.0, "unit": "ms"}
    assert result["first"]["inputs"]["topology_id"] == "aligned"
    assert result["second"]["inputs"]["topology_id"] == "grouped"
    assert result["first"]["hidden_by_ready_compute_ms"] <= 5
    json.dumps(result, allow_nan=False)


def test_invalid_inputs_fail_explicitly():
    with pytest.raises(ValueError, match="unknown track_id"):
        get_track_scenario("single_mcu")
    with pytest.raises(ValueError, match="at least 4"):
        build_traffic_matrix(2, Q_(1, "MB"), "incast")
    with pytest.raises(ValueError, match="capacity_fraction"):
        scenario = get_track_scenario("cloud_fleet")
        topology = topology_options(4, scenario.links[scenario.default_link_id].bandwidth)["grouped"]
        flows = build_traffic_matrix(4, Q_(1, "MB"), "incast")
        simulate_schedule(flows, topology, scenario.links[scenario.default_link_id], capacity_fraction=0)


def test_topology_options_rejects_unknown_track():
    with pytest.raises(ValueError, match="unknown track_id"):
        topology_options(8, Q_(100, "GB/s"), track_id="unknown_track")


@pytest.mark.parametrize("track_id", TRACKS)
def test_topology_options_per_track_labels_role_cost(track_id):
    scenario = get_track_scenario(track_id)
    options = topology_options(
        scenario.participants, scenario.links[scenario.default_link_id].bandwidth, track_id=track_id
    )
    assert len(options) == 4
    for key in ("nonblocking", "aligned", "grouped", "oversubscribed"):
        top = options[key]
        assert top.topology_id == key
        assert len(top.label) > 0
        assert len(top.role) > 0
        assert "illustrative" in top.assumption.lower()
        assert float(top.cost.magnitude) > 0

    canonical = scenario.track_id
    if canonical == "tinyml":
        assert "gateway" in options["nonblocking"].label.lower()
        assert "wearable" in options["nonblocking"].role.lower()
        assert "mcu" in options["nonblocking"].assumption.lower() or "collective" in options["nonblocking"].assumption.lower()
        assert float(options["nonblocking"].cost.magnitude) < 20_000
    elif canonical == "mobile":
        assert "ingress" in options["nonblocking"].label.lower()
        assert "phone" in options["nonblocking"].role.lower()
        assert float(options["nonblocking"].cost.magnitude) < 100_000
    elif canonical == "edge":
        assert "vehicle" in options["nonblocking"].role.lower()
        assert "local" in options["aligned"].assumption.lower() or "safety" in options["aligned"].assumption.lower()
    elif canonical == "cloud":
        assert "fat-tree" in options["nonblocking"].label.lower()
        assert "accelerator" in options["nonblocking"].role.lower()
        assert float(options["nonblocking"].cost.magnitude) > 500_000


@pytest.mark.parametrize("track_id", TRACKS)
def test_per_track_scenario_control_defaults_and_options(track_id):
    scenario = get_track_scenario(track_id)
    assert scenario.traffic_pattern in TRAFFIC_PATTERN_OPTIONS.values()
    assert default_traffic_pattern_label(track_id) in TRAFFIC_PATTERN_OPTIONS
    assert default_overlap_label(track_id) in OVERLAP_WINDOW_OPTIONS

    canonical = scenario.track_id
    if canonical == "edge":
        assert default_overlap_label(track_id) == "12 ms"
    elif canonical == "mobile":
        assert default_overlap_label(track_id) == "20 ms"
    elif canonical == "tinyml":
        assert default_overlap_label(track_id) == "30 s"
        assert burst_spacing_params(track_id)["stop"] >= 1000
    elif canonical == "cloud":
        assert default_overlap_label(track_id) == "35 ms"
        assert burst_spacing_params(track_id)["stop"] <= 50
    assert default_payload_label(track_id)
