"""Core concept regressions: fixed work, residency, wide state, heldout repair."""
import json

import pytest

from mlsysim.core.units import Q_
from mlsysim.hardware import Hardware
from mlsysim.labs.experiments import v1_11_experiments as accel
from mlsysim.labs.experiments import v1_12_experiments as bench


def dataflow(stationary):
    return accel.analyze_stationary_dataflow(
        m=8, n=12, k=16, tile_m=4, tile_n=4, tile_k=4,
        precision="int8", stationary=stationary)


def test_stationarity_trades_operand_traffic_for_partial_sum_spills_at_fixed_work():
    output, weight, activation = [dataflow(mode) for mode in ("output", "weight", "activation")]
    assert output["operations"] == weight["operations"] == activation["operations"]
    assert weight["traffic"]["weight_read"] < output["traffic"]["weight_read"]
    assert activation["traffic"]["activation_read"] < output["traffic"]["activation_read"]
    assert output["traffic"]["partial_sum_write"] == Q_(0, "byte")
    assert weight["traffic"]["partial_sum_write"] > Q_(0, "byte")
    assert activation["traffic"]["partial_sum_read"] > Q_(0, "byte")
    for result in (output, weight, activation):
        for key, total in result["traffic"].items():
            assert sum((step["traffic"][key] for step in result["trace"]), Q_(0, "byte")) == total
        assert result["traffic"]["output_write"] == Q_(8 * 12, "byte")
        assert result["dram_bytes"] == sum(result["traffic"].values(), Q_(0, "byte"))
        assert result["movement_energy"] == (result["dram_bytes"] * Hardware.Tech.Memory.DRAM.energy_per_byte).to("mJ")


def test_single_reduction_tile_has_no_partial_sum_spill():
    result = accel.analyze_stationary_dataflow(m=4, n=4, k=4, tile_m=4,
        tile_n=4, tile_k=4, precision="fp16", stationary="weight")
    assert result["traffic"]["partial_sum_write"] == Q_(0, "byte")
    assert result["traffic"]["partial_sum_read"] == Q_(0, "byte")


def test_wide_accumulators_can_make_unchanged_operand_tiles_illegal():
    args = dict(hardware=Hardware.Cloud.H100, m=8, n=8, k=8,
                tile_m=8, tile_n=8, tile_k=8, precision="int8", fused=True,
                scratchpad_capacity=Q_(300, "byte"))
    narrow = accel.analyze_tile_mapping(**args, accumulator_precision="int8")
    wide = accel.analyze_tile_mapping(**args, accumulator_precision="fp32")
    assert narrow.operand_storage == wide.operand_storage
    assert wide.accumulator_storage > narrow.accumulator_storage
    assert narrow.fits and not wide.fits and wide.movement_time is None
    assert wide.operations == narrow.operations
    assert wide.scratchpad_required == wide.operand_storage + wide.accumulator_storage


def test_stationary_schedule_replays_from_json():
    result = dataflow("activation")
    packet = accel.make_replay_packet(track="tinyml", experiment="stationary_dataflow", inputs=result["inputs"])
    replayed = accel.replay_experiment(json.loads(json.dumps(packet)))
    assert accel.to_jsonable(replayed) == accel.to_jsonable(result)


@pytest.mark.parametrize("track", bench.TRACKS)
def test_calibration_changes_confidence_without_changing_classification_or_ranking(track):
    reference = bench.compare_calibration(track, confidence_mode="reference")
    overconfident = bench.compare_calibration(track, confidence_mode="overconfident")
    repaired = bench.compare_calibration(track, confidence_mode="repaired")
    for result in (reference, overconfident, repaired):
        assert result["classifications_unchanged"]
        assert result["reference"]["accuracy_pct"] == result["candidate"]["accuracy_pct"]
        assert result["reference"]["roc_auc"] == result["candidate"]["roc_auc"]
        assert sum(item["count"] for item in result["candidate"]["reliability_bins"]) == result["heldout_count"]
        assert result["missing_coverage"]
        fit_ids = {row["case_id"] for row in result["calibration_records"]}
        evaluation_ids = {row["case_id"] for row in result["records"]}
        assert not fit_ids & evaluation_ids
    assert reference["candidate"]["brier_score"] != overconfident["candidate"]["brier_score"]
    assert repaired["calibration_log_loss_after"] <= repaired["calibration_log_loss_before"]
    assert [row["case_id"] for row in repaired["records"]] == [row["case_id"] for row in reference["records"]]
    replayed = bench.replay("v1_12_experiments.compare_calibration", repaired["inputs"])
    assert bench.to_jsonable(replayed) == bench.to_jsonable(repaired)


def test_calibration_does_not_claim_both_classes_from_one_case_per_slice():
    result = bench.compare_calibration("tinyml", examples_per_slice=1)
    assert any(row["candidate"]["roc_auc"] is None for row in result["slices"].values())


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_stationary_dimensions_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        accel.analyze_stationary_dataflow(m=bad, n=4, k=4, tile_m=4, tile_n=4,
                                         tile_k=4, precision="fp16")


def test_full_traffic_counts_remain_complete_when_trace_is_excerpted():
    result = accel.analyze_stationary_dataflow(m=128, n=128, k=128,
        tile_m=4, tile_n=4, tile_k=4, precision="int8", stationary="weight")
    assert result["trace_truncated"] and len(result["trace"]) == 128
    assert result["trace_total_steps"] == (128 // 4) ** 3
    assert result["traffic"]["weight_read"] == Q_(128 * 128, "byte")
    assert result["traffic"]["output_write"] == Q_(128 * 128, "byte")
    excerpt_bytes = sum((sum(step["traffic"].values(), Q_(0, "byte"))
                         for step in result["trace"]), Q_(0, "byte"))
    assert result["dram_bytes"] > excerpt_bytes


def test_repair_fit_is_invariant_to_changed_heldout_labels(monkeypatch):
    original = bench.compare_calibration("tinyml", confidence_mode="repaired")
    rng_type = bench.random.Random
    heldout_seed = bench.TRACKS["tinyml"].seed + 120_000

    class ChangedHoldoutRandom:
        def __init__(self, seed):
            self.rng = rng_type(seed)
            self.changed = seed == heldout_seed

        def random(self):
            value = self.rng.random()
            return 1.0 - value if self.changed else value

    monkeypatch.setattr(bench.random, "Random", ChangedHoldoutRandom)
    altered = bench.compare_calibration("tinyml", confidence_mode="repaired")
    assert original["fitted_temperature"] == altered["fitted_temperature"]
    assert original["calibration_records"] == altered["calibration_records"]
    assert [row["label"] for row in original["records"]] != [row["label"] for row in altered["records"]]
