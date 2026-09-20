from __future__ import annotations

import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_02_experiments import (
    compare_systems,
    compare_transfer_tiers,
    deserialize_quantity,
    evaluate_facility,
    evaluate_memory_floor,
    evaluate_memory_scale,
    evaluate_roofline,
    evaluate_transfer_tier,
    get_facility_count_policy,
    get_track_config,
    get_transfer_tier_options,
    replay,
    serialize_quantity,
)


TRACKS = ("tinyml", "mobile", "edge", "cloud")


@pytest.mark.parametrize("track_id", TRACKS)
def test_all_track_endpoints_produce_physical_results(track_id):
    config = get_track_config(track_id)

    roofline = evaluate_roofline(track_id, Q_("100 flop/byte"))
    memory = evaluate_memory_floor(track_id)
    transfers = compare_transfer_tiers(track_id)
    facility = evaluate_facility(track_id)
    systems = compare_systems(track_id, arithmetic_intensity=Q_("1000 flop/byte"))

    assert roofline["track_id"] == track_id
    assert len(roofline["rows"]) == 2
    assert memory["device_key"] == config.memory_device
    assert memory["latency_floor_ms"] > 0
    expected_tier_count = 5 if track_id == "cloud" else 3
    assert len(transfers["rows"]) == expected_tier_count
    assert transfers["device_key"] == config.memory_device
    assert facility["it_power_kw"] > facility["accelerator_power_kw"]
    assert len(systems["rows"]) == 2
    for result in (roofline, memory, transfers, facility, systems):
        saved_inputs = json.loads(json.dumps(result["inputs"]))
        assert replay(result["model_key"], saved_inputs) == result


@pytest.mark.parametrize(
    ("track_id", "low_winner", "high_winner"),
    [
        ("tinyml", "workstation:MacBookM3Max", "workstation:DGX_Spark"),
        ("mobile", "workstation:MacBookM3Max", "workstation:DGX_Spark"),
        ("edge", "cloud:MI300X", "cloud:Gaudi3"),
        ("cloud", "cloud:MI300X", "cloud:Gaudi3"),
    ],
)
def test_roofline_ranking_reverses_with_arithmetic_intensity(
    track_id, low_winner, high_winner
):
    low = evaluate_roofline(track_id, Q_("10 flop/byte"))
    high = evaluate_roofline(track_id, Q_("1000 flop/byte"))

    assert low["winner_device_key"] == low_winner
    assert high["winner_device_key"] == high_winner


@pytest.mark.parametrize("track_id", TRACKS)
def test_capacity_failure_is_distinct_from_a_bandwidth_floor(track_id):
    config = get_track_config(track_id)
    valid = evaluate_memory_floor(track_id, working_set=Q_("1 GiB"))
    failed = evaluate_memory_floor(track_id, working_set=Q_("1 TiB"))

    assert valid["fits"]
    assert valid["latency_floor_ms"] is not None
    assert not failed["fits"]
    assert failed["latency_floor_ms"] is None
    assert failed["headroom_gib"] < 0
    assert valid["device_key"] == config.memory_device


def test_more_memory_traffic_increases_latency_without_changing_fit():
    baseline = evaluate_memory_floor("cloud", bytes_moved=Q_("100 GiB"))
    intervention = evaluate_memory_floor("cloud", bytes_moved=Q_("200 GiB"))

    assert intervention["fits"] == baseline["fits"]
    assert intervention["bandwidth_floor_ms"] == pytest.approx(
        2 * baseline["bandwidth_floor_ms"]
    )


def test_memory_scale_can_cross_capacity_and_replays_exactly():
    baseline = evaluate_memory_scale("cloud", scale=1.0)
    enlarged = evaluate_memory_scale("cloud", scale=1.5)

    assert baseline["fits"]
    assert not enlarged["fits"]
    assert enlarged["bandwidth_floor_ms"] == pytest.approx(
        1.5 * baseline["bandwidth_floor_ms"]
    )
    assert replay(enlarged["model_key"], enlarged["inputs"]) == enlarged


@pytest.mark.parametrize("track_id", TRACKS)
def test_transfer_tiers_move_the_identical_payload(track_id):
    comparison = compare_transfer_tiers(track_id, payload=Q_("8 GiB"))
    rows = comparison["rows"]

    assert comparison["payload_gib"] == pytest.approx(8)
    assert all(row["transfer_ms"] > 0 for row in rows)
    for i in range(len(rows) - 1):
        assert rows[i]["transfer_ms"] < rows[i + 1]["transfer_ms"]


def test_single_transfer_tier_is_exactly_replayable():
    config = get_track_config("mobile")
    result = evaluate_transfer_tier(
        "mobile", tier=config.network_tier_name, payload=Q_("12 GiB")
    )

    assert result["tier"] == config.network_tier_name
    assert result["transfer_ms"] > 0
    assert result["slowdown_vs_baseline"] > 1
    assert replay(result["model_key"], result["inputs"]) == result


@pytest.mark.parametrize("track_id", TRACKS)
def test_track_transfer_hierarchy_omits_unmodeled_links_and_uses_memory_device(track_id):
    config = get_track_config(track_id)
    comparison = compare_transfer_tiers(track_id)

    assert comparison["device_key"] == config.memory_device
    tier_names = [row["tier"] for row in comparison["rows"]]
    assert comparison["baseline_tier"] == "accelerator memory"
    assert "accelerator memory" in tier_names
    assert config.network_tier_name in tier_names
    assert "durable object store" in tier_names

    selectable = get_transfer_tier_options(track_id)
    assert "accelerator memory" not in selectable
    assert config.network_tier_name in selectable

    if track_id == "cloud":
        assert "within-node accelerator link" in tier_names
        assert "host-to-accelerator link" in tier_names
        assert not comparison["omitted_links"]
    else:
        assert "within-node accelerator link" not in tier_names
        assert "host-to-accelerator link" not in tier_names
        assert len(comparison["omitted_links"]) == 2
        assert comparison["device_key"] != "cloud:H100"


@pytest.mark.parametrize("track_id", TRACKS)
def test_facility_part_d_baseline_and_selectable_options(track_id):
    config = get_track_config(track_id)
    policy = get_facility_count_policy(track_id)

    assert policy["baseline_count"] == config.accelerator_count
    assert policy["min_count"] <= config.accelerator_count <= policy["max_count"]
    assert policy["default_count"] in range(policy["min_count"], policy["max_count"] + 1)
    baseline = evaluate_facility(track_id, accelerator_count=config.accelerator_count)
    assert baseline["accelerator_count"] == config.accelerator_count
def test_facility_reports_separate_electrical_and_cooling_boundaries():
    baseline = evaluate_facility("cloud")
    electrical_failure = evaluate_facility(
        "cloud", electrical_capacity=Q_("10 kW"), cooling_capacity=Q_("30 kW")
    )
    cooling_failure = evaluate_facility(
        "cloud", electrical_capacity=Q_("30 kW"), cooling_capacity=Q_("10 kW")
    )

    assert baseline["feasible"]
    assert not electrical_failure["electrical_ok"]
    assert electrical_failure["cooling_ok"]
    assert cooling_failure["electrical_ok"]
    assert not cooling_failure["cooling_ok"]


def test_pue_changes_electrical_draw_but_not_it_heat():
    efficient = evaluate_facility("cloud", pue=1.06)
    inefficient = evaluate_facility("cloud", pue=1.50)

    assert inefficient["it_power_kw"] == efficient["it_power_kw"]
    assert inefficient["facility_power_kw"] > efficient["facility_power_kw"]


def test_procurement_exposes_performance_cost_and_energy_tradeoffs():
    result = compare_systems(
        "cloud",
        arithmetic_intensity=Q_("1000 flop/byte"),
        duty_cycle=0.8,
        horizon=Q_("3 year"),
    )
    h100, b200 = result["rows"]

    assert h100["device_key"] == "cloud:H100"
    assert b200["device_key"] == "cloud:B200"
    assert b200["active_sustained_tflops"] > h100["active_sustained_tflops"]
    assert b200["ownership_cost_usd"] > h100["ownership_cost_usd"]
    assert b200["usd_per_useful_pflop_hour"] < h100["usd_per_useful_pflop_hour"]
    assert b200["average_it_power_kw"] > h100["average_it_power_kw"]
    assert not h100["memory_fits"]
    assert b200["memory_fits"]
    assert result["lowest_cost_per_work_feasible_device_key"] == "cloud:B200"


def test_cost_inputs_do_not_change_physical_throughput():
    low_price = compare_systems(
        "cloud",
        arithmetic_intensity=Q_("1000 flop/byte"),
        electricity_price=Q_("0.05 USD/kWh"),
    )
    high_price = compare_systems(
        "cloud",
        arithmetic_intensity=Q_("1000 flop/byte"),
        electricity_price=Q_("0.50 USD/kWh"),
    )

    assert [row["active_sustained_tflops"] for row in low_price["rows"]] == [
        row["active_sustained_tflops"] for row in high_price["rows"]
    ]
    assert all(
        expensive["ownership_cost_usd"] > cheap["ownership_cost_usd"]
        for cheap, expensive in zip(low_price["rows"], high_price["rows"])
    )


@pytest.mark.parametrize(
    ("function", "args", "kwargs", "exception"),
    [
        (get_track_config, ("unknown",), {}, ValueError),
        (evaluate_roofline, ("cloud", Q_("0 flop/byte")), {}, ValueError),
        (
            evaluate_memory_floor,
            ("cloud",),
            {"working_set": Q_("1 second")},
            ValueError,
        ),
        (
            compare_transfer_tiers,
            ("cloud",),
            {"device_key": "cloud:missing"},
            ValueError,
        ),
        (evaluate_facility, ("cloud",), {"accelerator_count": 0}, ValueError),
        (evaluate_facility, ("cloud",), {"pue": 0.9}, ValueError),
        (
            compare_systems,
            ("cloud",),
            {"arithmetic_intensity": Q_("10 flop/byte"), "duty_cycle": 0},
            ValueError,
        ),
    ],
)
def test_invalid_inputs_fail_explicitly(function, args, kwargs, exception):
    with pytest.raises(exception):
        function(*args, **kwargs)


def test_existing_notebook_track_aliases_resolve_to_fleet_semantics():
    assert get_track_config("oura_ring").track_id == "tinyml"
    assert get_track_config("iphone").track_id == "mobile"
    assert get_track_config("robotaxi").track_id == "edge"
    assert get_track_config("cloud_fleet").track_id == "cloud"
    assert "training" in get_track_config("oura_ring").infrastructure_role


def test_quantity_serialization_is_unit_preserving_and_replayable():
    payload = serialize_quantity(Q_("3.5 GiB"))
    restored = deserialize_quantity(payload)

    assert restored.to("MiB").magnitude == pytest.approx(3584)


def test_result_inputs_replay_the_exact_evaluator_call():
    original = evaluate_memory_floor(
        "edge",
        device_key="cloud:MI300X",
        working_set=Q_("96 GiB"),
        bytes_moved=Q_("321 GiB"),
        operations=Q_("77 TFLOP"),
    )
    saved = json.loads(json.dumps(original["inputs"]))
    replayed = replay(original["model_key"], saved)

    assert replayed == original


def test_unknown_replay_model_is_incompatible():
    with pytest.raises(ValueError, match="unknown model_key"):
        replay("inferred_from_keywords", {"track_id": "cloud"})


@pytest.mark.parametrize(
    "invalid_tier",
    ["", "memory", "accelerator", "unknown_tier"],
)
def test_evaluate_transfer_tier_rejects_malformed_tier(invalid_tier):
    with pytest.raises(ValueError, match="unknown transfer tier"):
        evaluate_transfer_tier("cloud", tier=invalid_tier)


@pytest.mark.parametrize(
    "invalid_baseline",
    ["", "memory", "accelerator", "unknown_baseline"],
)
def test_evaluate_transfer_tier_rejects_malformed_baseline(invalid_baseline):
    config = get_track_config("cloud")
    with pytest.raises(ValueError, match="unknown baseline tier"):
        evaluate_transfer_tier(
            "cloud", tier=config.network_tier_name, baseline_tier=invalid_baseline
        )


def test_evaluate_transfer_tier_nondefault_baseline_replay_exact():
    config = get_track_config("cloud")
    result = evaluate_transfer_tier(
        "cloud",
        tier=config.network_tier_name,
        baseline_tier="within-node accelerator link",
    )
    assert result["baseline_tier"] == "within-node accelerator link"
    saved = json.loads(json.dumps(result["inputs"]))
    assert replay(result["model_key"], saved) == result
