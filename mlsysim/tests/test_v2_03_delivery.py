"""Causal delivery/recovery checks for the finite network reference fixture."""
import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.labs.experiments.v2_03_experiments import reliable_delivery_experiment


def test_zero_drops_does_not_establish_reliable_completion():
    run = reliable_delivery_experiment(policy="pfc", queue_state="occupied_cycle")
    assert run["dropped_packets"] == 0
    assert not run["delivered"]
    assert run["completion_ms"] is None
    assert run["victim_wait_ms"] is None
    json.dumps(run, allow_nan=False)


def test_topological_cycle_requires_occupied_closed_wait_to_stall():
    chain = reliable_delivery_experiment(policy="pfc", queue_state="draining_chain")
    cycle = reliable_delivery_experiment(policy="pfc", queue_state="occupied_cycle")
    assert chain["delivered"] and chain["completion_ms"] > chain["base_transfer_ms"]
    assert not cycle["delivered"]
    assert chain["retransmit_bytes"] == 0


def test_watchdog_breaks_pause_but_requires_reliable_transport_recovery():
    run = reliable_delivery_experiment()
    assert run["watchdog_fired"] and run["delivered"]
    assert run["dropped_packets"] == 1
    assert run["retransmitted_packets"] > run["dropped_packets"]
    assert run["completion_ms"] == pytest.approx(
        run["base_transfer_ms"] + run["pause_ms"] + run["recovery_ms"])
    assert run["trace"][-1]["time_ms"] == run["completion_ms"]
    assert run["retransmit_bytes"] <= run["payload_bytes"]


def test_longer_detection_extends_go_back_n_window_until_payload_bound():
    short = reliable_delivery_experiment(detection_delay=Q_(5, "us"))
    longer = reliable_delivery_experiment(detection_delay=Q_(20, "us"))
    capped = reliable_delivery_experiment(detection_delay=Q_(1, "s"))
    assert longer["retransmit_bytes"] > short["retransmit_bytes"]
    assert longer["completion_ms"] > short["completion_ms"]
    assert capped["retransmit_bytes"] == capped["payload_bytes"]


def test_selective_recovery_does_not_require_lossless_ethernet():
    run = reliable_delivery_experiment(policy="selective")
    assert run["delivered"]
    assert run["pause_ms"] == 0
    assert run["dropped_packets"] == run["retransmitted_packets"] == 1
    assert run["retained_out_of_order_bytes"] > 0
    slower = reliable_delivery_experiment(policy="selective", detection_delay=Q_(100, "us"))
    assert slower["retransmit_bytes"] == run["retransmit_bytes"]
    assert slower["completion_ms"] > run["completion_ms"]


def test_watchdog_threshold_changes_pause_cost_without_erasing_recovery():
    short = reliable_delivery_experiment(watchdog_timeout=Q_(5, "ms"))
    long = reliable_delivery_experiment(watchdog_timeout=Q_(100, "ms"))
    assert long["completion_ms"] > short["completion_ms"]
    assert long["retransmit_bytes"] == short["retransmit_bytes"]
    assert reliable_delivery_experiment(queue_state="draining_chain")["dropped_packets"] == 0
    assert reliable_delivery_experiment(queue_state="draining_chain", watchdog_timeout=Q_(1, "ms"))["watchdog_fired"]


def test_priority_isolation_changes_victim_dependency_not_original_flow():
    shared = reliable_delivery_experiment(policy="pfc")
    isolated = reliable_delivery_experiment(policy="pfc", isolate_victim=True)
    assert isolated["victim_wait_ms"] == 0
    assert shared["victim_wait_ms"] is None
    assert not isolated["delivered"]  # Original occupied cycle persists.
    assert isolated["completion_ms"] == shared["completion_ms"]


@pytest.mark.parametrize("track", ("cloud", "mobile", "edge", "tinyml"))
def test_tracks_explicitly_share_reference_backend_not_device_rdma(track):
    run = reliable_delivery_experiment(track)
    assert run["inputs"]["track_id"] == track
    assert "reference" in reliable_delivery_experiment.__doc__
    assert "RoCE" in run["reference_backend"]
    json.dumps(run, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"policy": "magic"}, {"queue_state": "topology_only"},
    {"detection_delay": Q_(0, "us")}, {"watchdog_timeout": Q_(-1, "ms")},
    {"detection_delay": Q_(float("nan"), "us")},
])
def test_unsupported_policies_and_invalid_delays_are_explicit(kwargs):
    with pytest.raises(ValueError):
        reliable_delivery_experiment(**kwargs)
