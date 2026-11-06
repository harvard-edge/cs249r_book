import numpy as np
import pytest

from mlsysim.labs.experiments.same_bits_experiments import (
    ORDERS,
    TRACKS,
    addition_orders,
    batch_contrast,
    contract_snapshot,
    extra_reduction_adds,
    get_scenario,
    integer_agreement,
    matmul_ordered,
    order_snapshot,
    serving_contrast,
    splits_for_batch,
    ulp_distance,
)


def test_every_track_names_a_registry_device_and_offers_batch_one():
    for track_id in TRACKS:
        scenario = get_scenario(track_id)
        assert scenario.device_name
        assert 1 in scenario.batch_options
        assert scenario.default_batch in scenario.batch_options
        assert scenario.default_batch > 1
    with pytest.raises(KeyError):
        get_scenario("mainframe")


def test_ulp_distance_counts_representable_steps():
    one = np.array([1.0], np.float32)
    assert ulp_distance(one, one) == 0
    assert ulp_distance(one, np.nextafter(one, np.float32(2))) == 1
    # Steps across zero count both sides of the number line.
    tiny = np.array([np.finfo(np.float32).smallest_subnormal], np.float32)
    assert ulp_distance(tiny, -tiny) == 2


def test_reduction_order_changes_float32_sum():
    data = addition_orders()
    assert [row["order"] for row in data["rows"]] == list(ORDERS)
    assert data["distinct"] > 1
    assert all(row["error_vs_exact"] < 1e-3 for row in data["rows"])
    base, other = order_snapshot("left to right"), order_snapshot("reversed")
    assert base["inputs"] != other["inputs"]
    assert base["ulps_from_left_to_right"] == 0


def test_split_k_with_one_split_matches_sequential_accumulation():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 16)).astype(np.float32)
    w = rng.standard_normal((16, 3)).astype(np.float32)
    expected = np.zeros((2, 3), np.float32)
    for k in range(16):
        expected = (expected + a[:, k:k + 1] * w[k:k + 1, :]).astype(np.float32)
    assert np.array_equal(matmul_ordered(a, w, 1, np.float32), expected)


def test_batch_changes_bits_only_when_the_split_changes():
    for batch in (1, 2, 4, 16, 32):
        result = batch_contrast(batch)
        same_split = result["splits_alone"] == result["splits_batched"]
        assert (result["outputs_differing"] == 0) == same_split, batch
        assert (result["digest_alone"] == result["digest_batched"]) == same_split
    for batch in (4, 16, 32):
        assert batch_contrast(batch, policy="invariant")["outputs_differing"] == 0


def test_throughput_kernel_changes_greedy_text_and_both_contracts_remove_it():
    assert serving_contrast(1)["changed"] == 0
    assert serving_contrast(16)["changed"] > 0
    for batch in (4, 8, 16, 32):
        assert serving_contrast(batch, policy="invariant")["changed"] == 0
        assert serving_contrast(batch, mode="integer")["changed"] == 0


def test_serving_contrast_returns_independent_copies():
    first = serving_contrast(16)
    first["examples"].clear()
    assert serving_contrast(16)["examples"]


def test_fixed_split_costs_extra_partial_sums_at_large_batches():
    assert splits_for_batch(32, "throughput") == 1
    assert splits_for_batch(32, "invariant") == splits_for_batch(1, "invariant")
    assert extra_reduction_adds(32, "throughput") == 0
    assert extra_reduction_adds(32, "invariant") > 0
    snapshot = contract_snapshot(16, "float", "invariant")
    assert snapshot["changed"] == 0
    assert snapshot["extra_reduction_adds"] == extra_reduction_adds(16, "invariant")


def test_integer_contract_trades_agreement_for_determinism():
    quality = integer_agreement()
    assert 0.5 < quality["token_agreement"] < 1.0
    assert quality["identical_sequences"] < quality["users"]
