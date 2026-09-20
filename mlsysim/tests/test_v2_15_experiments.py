"""Causal tests for the sustainable-fleet experiment backend."""

from __future__ import annotations

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v2_15_experiments import (
    PlacementSite,
    WorkloadPlacement,
    calculate_energy,
    calculate_lifecycle,
    calculate_rebound,
    check_facility,
    compare_replacement,
    deserialize_experiment_value,
    evaluate_mitigation,
    evaluate_placements,
    facility_experiment_args,
    lifecycle_experiment_args,
    lower_lifecycle_option,
    mitigation_experiment_args,
    placement_experiment_args,
    rebound_experiment_args,
    serialize_evaluator_args,
    select_lowest_carbon_placement,
    track_profile,
    track_scenario,
)


def test_energy_decomposition_preserves_fixed_work_and_facility_boundary():
    result = calculate_energy(
        operations=Q_(2, "teraflop"),
        moved_bytes=Q_(4, "gigabyte"),
        energy_per_operation=Q_(3, "picojoule/flop"),
        energy_per_byte=Q_(30, "picojoule/byte"),
        other_it_energy=Q_(1, "joule"),
        pue=1.25,
    )

    assert result.operation_energy.to("joule").magnitude == pytest.approx(6)
    assert result.movement_energy.to("joule").magnitude == pytest.approx(0.12)
    assert result.it_energy.to("joule").magnitude == pytest.approx(7.12)
    assert result.facility_energy.to("joule").magnitude == pytest.approx(8.9)
    assert result.facility_overhead_energy.to("joule").magnitude == pytest.approx(1.78)


def test_reducing_operations_does_not_secretly_reduce_movement_energy():
    common = dict(
        moved_bytes=Q_(10, "gigabyte"),
        energy_per_operation=Q_(4, "picojoule/flop"),
        energy_per_byte=Q_(50, "picojoule/byte"),
        pue=1.1,
    )
    baseline = calculate_energy(operations=Q_(2, "teraflop"), **common)
    optimized = calculate_energy(operations=Q_(1, "teraflop"), **common)

    assert optimized.operation_energy < baseline.operation_energy
    assert optimized.movement_energy == baseline.movement_energy


def test_facility_power_and_cooling_fail_independently_of_carbon():
    power_limited = check_facility(
        it_power=Q_(900, "kilowatt"),
        pue=1.2,
        electrical_capacity=Q_(1, "megawatt"),
        cooling_capacity=Q_(950, "kilowatt"),
    )
    cooling_limited = check_facility(
        it_power=Q_(900, "kilowatt"),
        pue=1.05,
        electrical_capacity=Q_(1, "megawatt"),
        cooling_capacity=Q_(850, "kilowatt"),
    )

    assert not power_limited.electrical_feasible
    assert power_limited.cooling_feasible
    assert cooling_limited.electrical_feasible
    assert not cooling_limited.cooling_feasible
    assert not power_limited.feasible
    assert not cooling_limited.feasible


def test_lifecycle_inference_crossover_depends_on_horizon():
    common = dict(
        training_it_energy=Q_(1_000, "kilowatt_hour"),
        inference_it_energy_per_request=Q_(10, "joule/count"),
        requests_per_time=Q_(10, "count/second"),
        pue=1.1,
        carbon_intensity=Q_(0.4, "kilogram/kilowatt_hour"),
    )
    short = calculate_lifecycle(horizon=Q_(1, "day"), **common)
    long = calculate_lifecycle(horizon=Q_(2, "year"), **common)

    assert not short.inference_dominates_training
    assert long.inference_dominates_training
    assert long.total_emissions > short.total_emissions


def test_replacement_counts_only_new_manufacture_and_can_cross_over():
    common = dict(
        training_it_energy=Q_(0, "kilowatt_hour"),
        requests_per_time=Q_(100, "count/second"),
        current_inference_it_energy_per_request=Q_(20, "joule/count"),
        replacement_inference_it_energy_per_request=Q_(5, "joule/count"),
        pue=1.1,
        carbon_intensity=Q_(0.5, "kilogram/kilowatt_hour"),
        replacement_manufacturing_emissions=Q_(1_000, "kilogram"),
    )
    short = compare_replacement(horizon=Q_(30, "day"), **common)
    long = compare_replacement(horizon=Q_(3, "year"), **common)

    assert short.keep.additional_manufacturing_emissions.magnitude == 0
    assert short.replace.additional_manufacturing_emissions == Q_(1_000, "kilogram")
    assert not short.replacement_break_even
    assert long.replacement_break_even
    assert long.operational_savings > short.operational_savings


def test_lifecycle_preference_is_computed_by_mlsysim():
    keep_args, replace_args = lifecycle_experiment_args("cloud", horizon_years=0.5)
    keep = calculate_lifecycle(**keep_args)
    replace = calculate_lifecycle(**replace_args)

    assert lower_lifecycle_option(keep, replace) in {"keep", "replace"}
    with pytest.raises(TypeError, match="LifecycleImpact"):
        lower_lifecycle_option({}, replace)


def _placement_sites():
    return (
        PlacementSite(
            name="home",
            available_accelerators=8,
            execution_time=Q_(50, "minute"),
            pue=1.3,
            carbon_intensity=Q_(0.5, "kilogram/kilowatt_hour"),
            electrical_capacity=Q_(20, "kilowatt"),
            cooling_capacity=Q_(18, "kilowatt"),
            wue=Q_(1.5, "liter/kilowatt_hour"),
        ),
        PlacementSite(
            name="clean",
            available_accelerators=8,
            execution_time=Q_(70, "minute"),
            pue=1.1,
            carbon_intensity=Q_(0.05, "kilogram/kilowatt_hour"),
            electrical_capacity=Q_(20, "kilowatt"),
            cooling_capacity=Q_(18, "kilowatt"),
            wue=None,
        ),
    )


def test_cleanest_site_is_rejected_when_it_misses_deadline():
    workload = WorkloadPlacement(
        name="movable training job",
        it_energy=Q_(100, "kilowatt_hour"),
        it_power=Q_(10, "kilowatt"),
        accelerator_demand=8,
        deadline=Q_(60, "minute"),
        movable=True,
        home_site="home",
    )
    rows = evaluate_placements(workload, _placement_sites())

    assert rows[0].feasible
    assert rows[0].water_use == Q_(195, "liter")
    assert not rows[1].feasible
    assert "execution misses deadline" in rows[1].reasons
    assert select_lowest_carbon_placement(workload, _placement_sites()).site == "home"


def test_immutable_edge_inference_cannot_migrate_to_cleaner_site():
    workload = WorkloadPlacement(
        name="site camera inference",
        it_energy=Q_(100, "kilowatt_hour"),
        it_power=Q_(10, "kilowatt"),
        accelerator_demand=4,
        deadline=Q_(2, "hour"),
        movable=False,
        home_site="home",
    )
    rows = evaluate_placements(workload, _placement_sites())

    assert rows[0].feasible
    assert not rows[1].feasible
    assert "immutable workload cannot leave its home site" in rows[1].reasons


def test_mitigation_needs_energy_latency_and_quality_to_hold_together():
    common = dict(
        operations=Q_(1, "teraflop"),
        moved_bytes=Q_(10, "gigabyte"),
        energy_per_operation=Q_(4, "picojoule/flop"),
        energy_per_byte=Q_(40, "picojoule/byte"),
        pue=1.2,
        latency_deadline=Q_(100, "millisecond"),
        minimum_quality=0.9,
    )
    acceptable = evaluate_mitigation(**common, latency=Q_(90, "millisecond"), quality=0.92)
    too_slow = evaluate_mitigation(**common, latency=Q_(120, "millisecond"), quality=0.92)
    low_quality = evaluate_mitigation(**common, latency=Q_(90, "millisecond"), quality=0.85)

    assert acceptable.acceptable
    assert not too_slow.acceptable
    assert not low_quality.acceptable
    # Supplied behavior evidence is a constraint, not a causal energy input.
    assert acceptable.energy == low_quality.energy


@pytest.mark.parametrize("demand_multiplier,increases", [(1.9, False), (2.1, True)])
def test_rebound_crosses_at_explicit_demand_boundary(demand_multiplier, increases):
    result = calculate_rebound(
        horizon=Q_(1, "year"),
        baseline_requests_per_time=Q_(10, "count/second"),
        baseline_it_energy_per_request=Q_(20, "joule/count"),
        optimized_it_energy_per_request=Q_(10, "joule/count"),
        demand_multiplier=demand_multiplier,
        pue=1.2,
        carbon_intensity=Q_(0.4, "kilogram/kilowatt_hour"),
    )

    assert result.break_even_demand_multiplier == pytest.approx(2.0)
    assert result.rebound_increases_emissions is increases


@pytest.mark.parametrize(
    "track_id,movable",
    [("tinyml", False), ("mobile", False), ("edge", False), ("cloud", True)],
)
def test_all_track_endpoints_describe_real_fleets(track_id, movable):
    profile = track_profile(track_id)

    assert profile["fleet_size"] > 1
    assert "fleet" in profile["fleet_shape"] or "pools" in profile["fleet_shape"]
    assert profile["inference_movable"] is movable


def test_invalid_units_and_unknown_track_fail_loudly():
    with pytest.raises(TypeError, match="Pint Quantity"):
        calculate_energy(
            operations=1,
            moved_bytes=Q_(1, "byte"),
            energy_per_operation=Q_(1, "joule/flop"),
            energy_per_byte=Q_(1, "joule/byte"),
            pue=1.1,
        )
    with pytest.raises(ValueError, match="track_id"):
        track_profile("laptop")


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge", "cloud"])
def test_track_scenarios_supply_runnable_physical_evaluator_arguments(track_id):
    scenario = track_scenario(track_id)

    energy = calculate_energy(**scenario["energy_args"])
    facility = check_facility(**scenario["facility_args"])
    lifecycle = calculate_lifecycle(**scenario["lifecycle_args"])
    replacement = compare_replacement(**scenario["replacement_args"])
    rebound = calculate_rebound(**scenario["rebound_args"])
    placements = evaluate_placements(scenario["placement_workload"], scenario["placement_sites"])
    mitigation = evaluate_mitigation(**scenario["mitigation_args"])

    assert energy.facility_energy.magnitude > 0
    assert isinstance(facility.feasible, bool)
    assert lifecycle.horizon.to("year").magnitude == pytest.approx(2)
    assert replacement.additional_manufacturing_emissions.magnitude > 0
    assert rebound.demand_multiplier == 1.0
    assert any(row.feasible for row in placements)
    assert mitigation.acceptable


def test_evaluator_argument_serialization_round_trips_units_and_dataclasses():
    original = track_scenario("cloud")
    captured = serialize_evaluator_args(original)
    restored = deserialize_experiment_value(captured)

    assert captured["energy_args"]["operations"]["__type__"] == "quantity"
    assert captured["energy_args"]["operations"]["unit"] == "Pflop"
    assert restored["energy_args"]["operations"] == original["energy_args"]["operations"]
    assert restored["placement_workload"] == original["placement_workload"]
    # Rehydrated kwargs run directly, which is the capstone replay contract.
    assert calculate_energy(**restored["energy_args"]) == calculate_energy(**original["energy_args"])


def test_track_scenario_returns_an_independent_copy():
    first = track_scenario("tinyml")
    first["energy_args"]["pue"] = 9.0

    assert track_scenario("tinyml")["energy_args"]["pue"] == 1.15


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge", "cloud"])
def test_lab_control_helpers_return_exact_runnable_contrasts(track_id):
    facility_base, facility_changed = facility_experiment_args(track_id, load_scale=1.2)
    lifecycle_keep, lifecycle_replace = lifecycle_experiment_args(track_id, horizon_years=3.0)
    placement_base = placement_experiment_args(track_id, deadline_scale=1.0)
    placement_changed = placement_experiment_args(track_id, deadline_scale=1.5, clean_capacity_scale=0.5)
    mitigation_base, mitigation_changed = mitigation_experiment_args(track_id, intervention="movement")
    rebound_base, rebound_changed = rebound_experiment_args(track_id, demand_multiplier=2.0)

    assert check_facility(**facility_changed).it_power > check_facility(**facility_base).it_power
    assert calculate_lifecycle(**lifecycle_keep).horizon.to("year").magnitude == pytest.approx(3)
    assert calculate_lifecycle(**lifecycle_replace).additional_manufacturing_emissions.magnitude > 0
    assert placement_base != placement_changed
    assert len(evaluate_placements(**placement_changed)) == 2
    assert (
        evaluate_mitigation(**mitigation_changed).energy.movement_energy
        < evaluate_mitigation(**mitigation_base).energy.movement_energy
    )
    assert calculate_rebound(**rebound_base).demand_multiplier == 1.0
    assert calculate_rebound(**rebound_changed).demand_multiplier == 2.0


def test_unknown_mitigation_is_rejected():
    with pytest.raises(ValueError, match="intervention"):
        mitigation_experiment_args("cloud", intervention="offsets")


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge", "cloud"])
def test_track_profile_home_site_matches_workload_and_placement_sites(track_id):
    profile = track_profile(track_id)
    scenario = track_scenario(track_id)
    workload = scenario["placement_workload"]
    sites = scenario["placement_sites"]

    assert profile["home_site"] == workload.home_site
    assert any(site.name == profile["home_site"] for site in sites)


@pytest.mark.parametrize("track_id", ["tinyml", "mobile", "edge"])
def test_fixed_location_tracks_preserve_immobility_and_timing_causality(track_id):
    base_args = placement_experiment_args(track_id, deadline_scale=1.0, clean_capacity_scale=1.0)
    clean_scaled_args = placement_experiment_args(track_id, deadline_scale=1.0, clean_capacity_scale=0.0)
    base_results = evaluate_placements(**base_args)
    clean_scaled_results = evaluate_placements(**clean_scaled_args)

    clean_base = next(r for r in base_results if r.site == "clean region")
    clean_scaled = next(r for r in clean_scaled_results if r.site == "clean region")
    assert not clean_base.feasible
    assert not clean_scaled.feasible
    assert "immutable workload cannot leave its home site" in clean_base.reasons

    tight_deadline_args = placement_experiment_args(track_id, deadline_scale=0.25)
    tight_results = evaluate_placements(**tight_deadline_args)
    home_tight = next(r for r in tight_results if r.site == base_args["workload"].home_site)
    assert not home_tight.feasible
    assert "execution misses deadline" in home_tight.reasons
