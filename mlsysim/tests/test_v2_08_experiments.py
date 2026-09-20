import json

import pytest

from mlsysim.engine.v2_08_experiments import (
    TRACKS,
    dominant_resource_fairness,
    dominant_shares,
    fairness_comparison,
    gang_backfill_comparison,
    partial_allocation_deadlock,
    preemption_comparison,
    simulate_schedule,
    simulate_warm_capacity,
    topology_crossover,
    topology_wait_comparison,
    topology_wait_curve,
    track_fixture,
    warm_capacity_comparison,
)


def _small_nodes():
    return [
        {
            "id": "n0",
            "domain": "left",
            "capacity": {"accelerators": 2, "cpu_cores": 8, "memory_gb": 32, "network_gbps": 20},
        },
        {
            "id": "n1",
            "domain": "right",
            "capacity": {"accelerators": 2, "cpu_cores": 8, "memory_gb": 32, "network_gbps": 20},
        },
    ]


def _job(job_id, arrival, duration, accelerators, priority=1, same_domain=False, checkpoint=10):
    return {
        "id": job_id,
        "tenant": job_id.split("-")[0],
        "arrival_min": arrival,
        "duration_min": duration,
        "priority": priority,
        "same_domain": same_domain,
        "resources": {"accelerators": accelerators, "cpu_cores": accelerators * 2, "memory_gb": accelerators * 4},
        "checkpoint_interval_min": checkpoint,
    }


def test_every_track_is_an_honest_bounded_fleet_and_runs_all_jobs():
    long_durations = set()
    for track_id in TRACKS:
        fixture = track_fixture(track_id)
        result = simulate_schedule(fixture["nodes"], fixture["jobs"], "backfill", horizon_min=120)
        truncated = simulate_schedule(fixture["nodes"], fixture["jobs"], "backfill", horizon_min=3)

        assert len(fixture["nodes"]) == 4, track_id
        assert "pool" in fixture["fleet"], track_id
        assert fixture["urgent_work"] != fixture["background_work"], track_id
        assert result["completed_jobs"] == len(fixture["jobs"]), track_id
        assert truncated["completed_jobs"] < len(fixture["jobs"]), track_id
        assert truncated["utilization_pct"] > 0, track_id
        assert len(result["events"]) < 80, track_id
        long_durations.add(fixture["jobs"][0]["duration_min"])
    assert len(long_durations) == len(TRACKS)


def test_partial_rigid_allocations_deadlock_while_atomic_gangs_make_progress():
    partial = partial_allocation_deadlock(8, {"job-a": 6, "job-b": 6})

    nodes = [
        {
            "id": "n0",
            "domain": "d0",
            "capacity": {"accelerators": 8, "cpu_cores": 32, "memory_gb": 128, "network_gbps": 80},
        }
    ]
    jobs = [_job("job-a", 0, 5, 6), _job("job-b", 0, 5, 6)]
    gang = simulate_schedule(nodes, jobs, "fifo_gang", horizon_min=12)

    assert partial["deadlocked"]
    assert partial["idle_held_accelerators"] == 8
    assert partial["useful_accelerators"] == 0
    assert gang["completed_jobs"] == 2


def test_backfill_starts_short_job_while_fifo_blocks_behind_large_head():
    jobs = [
        _job("owner-running", 0, 20, 2),
        _job("large-head", 1, 4, 4),
        _job("short-backfill", 2, 3, 2),
    ]
    fifo = simulate_schedule(_small_nodes(), jobs, "fifo_gang", horizon_min=30)
    backfill = simulate_schedule(_small_nodes(), jobs, "backfill", horizon_min=30)

    assert fifo["jobs"]["short-backfill"]["start_min"] == 24
    assert backfill["jobs"]["short-backfill"]["start_min"] == 2
    assert any(event["kind"] == "backfill" and event["job_id"] == "short-backfill" for event in backfill["events"])


def test_topology_requirement_waits_despite_enough_aggregate_capacity():
    jobs = [
        _job("left-owner", 0, 8, 1),
        _job("right-owner", 0, 10, 1),
        _job("gang", 1, 2, 2, same_domain=True),
    ]
    result = simulate_schedule(_small_nodes(), jobs, "backfill", horizon_min=15)

    waits = [event for event in result["events"] if event["kind"] == "topology_wait" and event["job_id"] == "gang"]
    assert waits
    assert waits[0]["aggregate_free"] == 2
    assert result["jobs"]["gang"]["start_min"] == 8


def test_topology_crossover_uses_payload_bandwidth_and_wait_units():
    short_wait = topology_crossover(
        wait_min=0.5, payload_gb=10, iterations=100, local_gbps=900, remote_gbps=100, compute_min=10
    )
    long_wait = topology_crossover(
        wait_min=5, payload_gb=10, iterations=100, local_gbps=900, remote_gbps=100, compute_min=10
    )

    assert short_wait["decision"] == "wait_for_local"
    assert long_wait["decision"] == "start_remote"
    assert short_wait["crossover_wait_min"] == pytest.approx(1.185185)
    assert short_wait["remote_communication_min"] > short_wait["local_communication_min"]


def test_priority_preemption_reduces_urgent_wait_and_charges_lost_work():
    jobs = [
        _job("background-train", 0, 30, 4, priority=1, checkpoint=10),
        _job("urgent-repair", 7, 2, 2, priority=10, checkpoint=1),
    ]
    fifo = simulate_schedule(_small_nodes(), jobs, "fifo_gang", horizon_min=50)
    preemptive = simulate_schedule(_small_nodes(), jobs, "priority_preempt", horizon_min=50)

    assert fifo["jobs"]["urgent-repair"]["wait_min"] == 23
    assert preemptive["jobs"]["urgent-repair"]["wait_min"] == 0
    assert preemptive["jobs"]["background-train"]["lost_work_min"] == 7
    assert preemptive["lost_accelerator_minutes"] == 28
    assert preemptive["useful_utilization_pct"] < preemptive["utilization_pct"]
    assert preemptive["jobs"]["background-train"]["finish_min"] > fifo["jobs"]["background-train"]["finish_min"]
    assert any(event["kind"] == "preempt" for event in preemptive["events"])


def test_checkpoint_interval_changes_lost_work_but_not_urgent_start():
    slow_checkpoint = [_job("background", 0, 30, 4, checkpoint=10), _job("urgent", 7, 2, 2, priority=10)]
    fast_checkpoint = [_job("background", 0, 30, 4, checkpoint=2), _job("urgent", 7, 2, 2, priority=10)]

    slow = simulate_schedule(_small_nodes(), slow_checkpoint, "priority_preempt", horizon_min=50)
    fast = simulate_schedule(_small_nodes(), fast_checkpoint, "priority_preempt", horizon_min=50)

    assert slow["jobs"]["background"]["lost_work_min"] == 7
    assert fast["jobs"]["background"]["lost_work_min"] == 1
    assert slow["jobs"]["urgent"]["start_min"] == fast["jobs"]["urgent"]["start_min"] == 7


def test_drf_equalizes_dominant_shares_from_actual_bundles():
    capacity = {"accelerators": 8, "cpu_cores": 32, "memory_gb": 256, "network_gbps": 80}
    bundles = {
        "A": {"accelerators": 1, "cpu_cores": 4, "memory_gb": 32, "network_gbps": 10},
        "B": {"accelerators": 2, "cpu_cores": 2, "memory_gb": 16, "network_gbps": 20},
    }
    result = dominant_resource_fairness(capacity, bundles)

    assert result["bundle_counts"] == {"A": 4, "B": 2}
    assert result["tenants"]["A"]["dominant_share"] == 0.5
    assert result["tenants"]["B"]["dominant_share"] == 0.5
    assert result["dominant_share_gap"] == 0


def test_equal_accelerators_can_be_unequal_under_multiple_resources():
    capacity = {"accelerators": 8, "cpu_cores": 32, "memory_gb": 256, "network_gbps": 80}
    equal_gpu = dominant_shares(
        capacity,
        {
            "cpu-heavy": {"accelerators": 4, "cpu_cores": 28, "memory_gb": 64, "network_gbps": 20},
            "network-heavy": {"accelerators": 4, "cpu_cores": 4, "memory_gb": 64, "network_gbps": 40},
        },
    )

    assert equal_gpu["tenants"]["cpu-heavy"]["shares"]["accelerators"] == 0.5
    assert equal_gpu["tenants"]["network-heavy"]["shares"]["accelerators"] == 0.5
    assert equal_gpu["tenants"]["cpu-heavy"]["dominant_share"] == 0.875
    assert equal_gpu["dominant_share_gap"] == 0.375


def test_warm_reserve_reduces_ramp_shortfall_but_costs_more_idle_time():
    trace = [
        {"time_min": 0, "demand_per_min": 20},
        {"time_min": 2, "demand_per_min": 100},
        {"time_min": 10, "demand_per_min": 20},
    ]
    cold = simulate_warm_capacity(
        trace, 30, warm_reserve=0, loading_delay_min=2, warmup_delay_min=2, replica_cost_per_hour=6
    )
    warm = simulate_warm_capacity(
        trace, 30, warm_reserve=2, loading_delay_min=2, warmup_delay_min=2, replica_cost_per_hour=6
    )

    assert warm["missed_requests"] < cold["missed_requests"]
    assert warm["idle_replica_minutes"] > cold["idle_replica_minutes"]
    assert warm["cost_usd"] > cold["cost_usd"]
    assert warm["readiness_delay_min"] == cold["readiness_delay_min"] == 4


def test_noncausal_tenant_label_does_not_change_physical_schedule():
    jobs = [_job("team-a", 0, 4, 2), _job("team-b", 1, 4, 2)]
    renamed = [dict(job, tenant="renamed") for job in jobs]

    original = simulate_schedule(_small_nodes(), jobs, "fifo_gang", horizon_min=10)
    changed = simulate_schedule(_small_nodes(), renamed, "fifo_gang", horizon_min=10)

    assert original["events"] == changed["events"]
    assert original["utilization_pct"] == changed["utilization_pct"]


def test_results_record_exact_replay_inputs_and_are_json_serializable():
    nodes = _small_nodes()
    jobs = [_job("a", 0, 4, 2)]
    schedule = simulate_schedule(nodes, jobs, "backfill", horizon_min=9)
    crossover = topology_crossover(1, 4, 20, 40, 5, 3)
    capacity = {"accelerators": 4, "cpu_cores": 16, "memory_gb": 64, "network_gbps": 40}
    fairness = dominant_resource_fairness(
        capacity,
        {"a": {"accelerators": 1, "cpu_cores": 2, "memory_gb": 4, "network_gbps": 5}},
    )

    assert schedule["inputs"] == {"nodes": nodes, "jobs": jobs, "policy": "backfill", "horizon_min": 9.0}
    assert crossover["inputs"] == {
        "wait_min": 1,
        "payload_gb": 4,
        "iterations": 20,
        "local_gbps": 40,
        "remote_gbps": 5,
        "compute_min": 3,
    }
    assert fairness["inputs"]["capacity"] == capacity
    assert schedule["model_id"] == crossover["model_id"] == fairness["model_id"] == "v2_08_experiments"
    json.dumps({"schedule": schedule, "crossover": crossover, "fairness": fairness})


def test_chapter_comparisons_provide_causal_contrasts_for_every_track():
    for track_id in TRACKS:
        allocation = gang_backfill_comparison(track_id)
        topology = topology_wait_comparison(track_id, wait_min=10, payload_gb=10, iterations=100)
        preemption = preemption_comparison(track_id, urgent_arrival_min=7, checkpoint_interval_min=10)
        fairness = fairness_comparison(track_id)
        capacity = warm_capacity_comparison(track_id, warm_reserve=2, demand_scale=1)

        assert allocation["partial"]["deadlocked"], track_id
        assert (
            allocation["backfill"]["jobs"]["short"]["start_min"] < allocation["fifo"]["jobs"]["short"]["start_min"]
        ), track_id
        assert topology["baseline"]["inputs"] != topology["result"]["inputs"], track_id
        assert (
            preemption["result"]["jobs"]["urgent"]["wait_min"] < preemption["baseline"]["jobs"]["urgent"]["wait_min"]
        ), track_id
        assert preemption["result"]["jobs"]["background"]["lost_work_min"] > 0, track_id
        assert fairness["result"]["dominant_share_gap_pct"] < fairness["baseline"]["dominant_share_gap_pct"], track_id
        assert capacity["result"]["missed_requests"] < capacity["baseline"]["missed_requests"], track_id
        assert capacity["result"]["cost_usd"] > capacity["baseline"]["cost_usd"], track_id
        json.dumps(
            {
                "allocation": allocation,
                "topology": topology,
                "preemption": preemption,
                "fairness": fairness,
                "capacity": capacity,
            }
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: track_fixture("desktop"),
        lambda: simulate_schedule([], [_job("a", 0, 1, 1)]),
        lambda: simulate_schedule(_small_nodes(), [_job("a", 0, 1, 5)]),
        lambda: simulate_schedule(_small_nodes(), [_job("a", 0, 1, 1)], policy="magic"),
        lambda: partial_allocation_deadlock(8, {"only": 4}),
        lambda: topology_crossover(1, 1, 0, 10, 1),
        lambda: dominant_shares({"accelerators": 0}, {}),
        lambda: simulate_warm_capacity([], 10, 1, 1, 1, 1),
    ],
)
def test_invalid_scenarios_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()


def test_topology_wait_curve_spans_crossover_for_every_track():
    for track_id in TRACKS:
        curve = topology_wait_curve(track_id, payload_gb=10, iterations=100)
        decisions = [point["result"]["decision"] for point in curve]
        assert "wait_for_local" in decisions, f"{track_id} missing wait_for_local"
        assert "start_remote" in decisions, f"{track_id} missing start_remote"
        assert curve[0]["inputs"]["wait_min"] < curve[-1]["inputs"]["wait_min"]


def test_track_fixture_provides_calibrated_selector_defaults_and_ranges():
    for track_id, profile in TRACKS.items():
        fixture = track_fixture(track_id)
        assert "wait_slider" in fixture, track_id
        assert "arrival_slider" in fixture, track_id
        assert "checkpoint_slider" in fixture, track_id
        w = fixture["wait_slider"]
        assert w["min"] < w["value"] < w["max"], track_id
        c = fixture["checkpoint_slider"]
        assert c["value"] == profile["checkpoint_interval_min"], track_id
        assert c["min"] <= c["value"] <= c["max"], track_id
        a = fixture["arrival_slider"]
        assert a["min"] < a["value"] < a["max"], track_id
        assert a["max"] <= profile["long_duration_min"], track_id
