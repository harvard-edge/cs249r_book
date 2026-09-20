from dataclasses import replace

import pytest

from mlsysim.core.units import Q_
from mlsysim.engine.v1_12_experiments import (
    SCENARIO_ASSUMPTION,
    TRACKS,
    analyze_repeats,
    audit_protocols,
    compare_quality_slices,
    compare_scopes,
    compare_sustained,
    default_protocol,
    replay,
)


@pytest.mark.parametrize("track_id", TRACKS)
def test_every_track_produces_physical_scope_repeat_sustained_and_quality_evidence(track_id):
    scope = compare_scopes(track_id, sample_count=24)
    repeats = analyze_repeats(track_id, measured_iterations=12, repeat_count=3)
    sustained = compare_sustained(track_id, duration=Q_(10, "minute"), sample_count=40)
    quality = compare_quality_slices(track_id, examples_per_slice=12)

    assert scope["baseline_path"]["median"].check("[time]")
    assert repeats["run_to_run_std"].check("[time]")
    assert sustained["deadline"].check("[time]")
    assert sustained["candidate_energy_per_request"].check("[energy]")
    assert set(quality["slices"]) == {"ordinary", "difficult"}
    assert scope["assumption"] == repeats["assumption"] == SCENARIO_ASSUMPTION


@pytest.mark.parametrize("track_id", TRACKS)
def test_local_kernel_gain_is_diluted_at_the_whole_path_boundary(track_id):
    result = compare_scopes(track_id)

    assert result["kernel_speedup"] == pytest.approx(TRACKS[track_id].candidate_kernel_speedup)
    assert 1.0 < result["whole_path_speedup"] < result["kernel_speedup"]
    assert all(delta.m_as("ms") > 0 for delta in result["paired_delta"])


def test_scope_comparison_uses_common_paired_disturbances():
    result = compare_scopes("mobile", sample_count=16)
    expected_ratio = result["kernel_speedup"]
    baseline = result["baseline_kernel"]
    candidate = result["candidate_kernel"]

    assert baseline["median"].m_as("ms") / candidate["median"].m_as("ms") == pytest.approx(expected_ratio)
    assert result == compare_scopes("mobile", sample_count=16)


def test_more_kernel_speedup_changes_candidate_but_not_baseline():
    modest = compare_scopes("edge", kernel_speedup=1.5)
    strong = compare_scopes("edge", kernel_speedup=3.0)

    assert modest["baseline_path"] == strong["baseline_path"]
    assert strong["candidate_path"]["median"] < modest["candidate_path"]["median"]


def test_warmup_discard_and_drift_change_the_observed_sample():
    cold_included = analyze_repeats(
        "mobile", warmup_iterations=0, measured_iterations=20, repeat_count=4
    )
    warm_discarded = analyze_repeats(
        "mobile", warmup_iterations=10, measured_iterations=20, repeat_count=4
    )
    no_drift = analyze_repeats(
        "mobile", warmup_iterations=10, measured_iterations=20, repeat_count=4, drift_scale=0
    )
    high_drift = analyze_repeats(
        "mobile", warmup_iterations=10, measured_iterations=20, repeat_count=4, drift_scale=4
    )

    assert warm_discarded["median_of_run_medians"] < cold_included["median_of_run_medians"]
    assert high_drift["runs"][0]["p99"] > no_drift["runs"][0]["p99"]


def test_repeat_analysis_reports_run_variability_without_iid_interval():
    result = analyze_repeats("cloud", repeat_count=5)

    assert len(result["runs"]) == 5
    assert result["run_to_run_cv_pct"] > 0
    assert result["interval"] is None
    assert "No IID confidence interval" in result["interval_note"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_sustained_candidate_exposes_energy_tradeoff_where_present(track_id):
    result = compare_sustained(track_id, duration=Q_(10, "minute"))

    assert result["median_speedup"] > 1.0
    assert result["candidate_energy_per_request"].m_as("J") > 0
    assert 0 <= result["candidate_deadline_miss_pct"] <= 100


@pytest.mark.parametrize("track_id", ("tinyml", "edge", "cloud"))
def test_stress_can_reverse_deadline_miss_ranking(track_id):
    result = compare_sustained(track_id, duration=Q_(10, "minute"), stress_scale=4)

    assert result["candidate_deadline_misses"] > result["baseline_deadline_misses"]


def test_short_run_hides_part_of_sustained_candidate_drift():
    short = compare_sustained("cloud", duration=Q_(30, "second"))
    sustained = compare_sustained("cloud", duration=Q_(10, "minute"))

    assert sustained["median_speedup"] < short["median_speedup"]
    assert sustained["candidate"]["p99"] > short["candidate"]["p99"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_quality_gain_on_ordinary_slice_does_not_hide_difficult_slice_loss(track_id):
    result = compare_quality_slices(track_id)

    ordinary = result["slices"]["ordinary"]
    difficult = result["slices"]["difficult"]
    assert ordinary["candidate_accuracy_pct"] >= ordinary["baseline_accuracy_pct"]
    assert difficult["candidate_accuracy_pct"] < difficult["baseline_accuracy_pct"]
    assert not result["observed_overall_noninferior"]


def test_quality_floor_changes_eligibility_not_observed_behavior():
    permissive = compare_quality_slices("mobile", quality_floor_pct=50)
    strict = compare_quality_slices("mobile", quality_floor_pct=95)

    assert permissive["candidate_aggregate_eligible"]
    assert not strict["candidate_aggregate_eligible"]
    assert permissive["records"] == strict["records"]
    assert permissive["slices"] == strict["slices"]
    assert permissive["candidate_overall_accuracy_pct"] == strict["candidate_overall_accuracy_pct"]


@pytest.mark.parametrize("track_id", TRACKS)
def test_quality_preserving_mode_is_observed_noninferior_on_same_fixture(track_id):
    result = compare_quality_slices(track_id, candidate_mode="quality_preserving")

    assert result["observed_overall_noninferior"]
    assert result["baseline_overall_accuracy_pct"] == result["candidate_overall_accuracy_pct"]
    assert (
        result["slices"]["ordinary"]["candidate_accuracy_pct"]
        == result["slices"]["ordinary"]["baseline_accuracy_pct"]
    )


@pytest.mark.parametrize("track_id", TRACKS)
def test_baseline_quality_view_is_replayable_and_labeled(track_id):
    result = compare_quality_slices(track_id, candidate_mode="baseline")

    assert result["system_role"] == "reference baseline"
    assert result["observed_overall_noninferior"]
    assert compare_quality_slices(**result["inputs"])["records"] == result["records"]


def test_explicit_worst_slice_floor_prevents_aggregate_from_hiding_slice_failure():
    result = compare_quality_slices(
        "mobile", quality_floor_pct=60, worst_slice_floor_pct=60, candidate_mode="aggressive"
    )

    assert result["candidate_aggregate_eligible"]
    assert result["baseline_worst_slice_eligible"]
    assert not result["candidate_worst_slice_eligible"]


def test_absent_worst_slice_floor_does_not_invent_a_slice_verdict():
    result = compare_quality_slices("mobile", worst_slice_floor_pct=None)

    assert result["baseline_worst_slice_eligible"] is None
    assert result["candidate_worst_slice_eligible"] is None


def test_results_carry_complete_replay_inputs():
    scope = compare_scopes("mobile", sample_count=13, kernel_speedup=1.8, workload_scale=1.2, seed_offset=4)
    repeats = analyze_repeats(
        "mobile", warmup_iterations=3, measured_iterations=13, repeat_count=3,
        drift_scale=1.4, seed_offset=4,
    )
    sustained = compare_sustained(
        "mobile", duration=Q_(2, "minute"), sample_count=13, stress_scale=1.4, seed_offset=4,
    )
    quality = compare_quality_slices(
        "mobile", threshold=0.4, examples_per_slice=13, quality_floor_pct=70,
        worst_slice_floor_pct=60, candidate_mode="quality_preserving", seed_offset=4,
    )

    assert compare_scopes(**scope["inputs"])["paired_delta"] == scope["paired_delta"]
    assert analyze_repeats(**repeats["inputs"])["traces"] == repeats["traces"]
    assert compare_sustained(**sustained["inputs"])["paired_delta"] == sustained["paired_delta"]
    assert compare_quality_slices(**quality["inputs"])["records"] == quality["records"]


def test_public_replay_supports_every_saved_part_and_arm():
    scope = compare_scopes("edge", reported_scope="kernel")
    repeats = analyze_repeats("edge", warmup_iterations=4, measured_iterations=12, repeat_count=3)
    sustained = compare_sustained("edge", duration=Q_(2, "minute"), sample_count=20)
    quality = compare_quality_slices("edge", candidate_mode="aggressive", worst_slice_floor_pct=60)
    protocol = default_protocol("edge")
    audit = audit_protocols(protocol, replace(protocol, scope="isolated kernel"))

    assert replay("v1_12_experiments.compare_scopes", scope["inputs"])["reported_scope"] == "kernel"
    assert replay("analyze_repeats", repeats["inputs"])["traces"] == repeats["traces"]
    assert replay("compare_sustained", sustained["inputs"])["paired_delta"] == sustained["paired_delta"]
    assert replay("compare_quality_slices", quality["inputs"])["records"] == quality["records"]
    assert replay("audit_protocols", audit["inputs"])["mismatches"] == audit["mismatches"]


def test_public_replay_rejects_unknown_or_malformed_requests():
    with pytest.raises(ValueError, match="model_key"):
        replay("unknown", {})
    with pytest.raises(TypeError, match="dictionary"):
        replay("compare_scopes", [])
    with pytest.raises(ValueError, match="reference and candidate"):
        replay("audit_protocols", {})


@pytest.mark.parametrize("track_id", TRACKS)
def test_matching_protocol_is_included_and_mismatch_is_excluded_with_repair(track_id):
    reference = default_protocol(track_id)
    matching = audit_protocols(reference, reference)
    mismatched_spec = replace(reference, scope="isolated kernel", warmup_iterations=0)
    mismatched = audit_protocols(reference, mismatched_spec)

    assert matching["comparable"]
    assert matching["include_in_headline_comparison"]
    assert matching["mismatches"] == ()
    assert not mismatched["comparable"]
    assert not mismatched["include_in_headline_comparison"]
    assert {item["field"] for item in mismatched["mismatches"]} == {"scope", "warmup_iterations"}
    assert audit_protocols(reference, mismatched["repair"])["comparable"]


@pytest.mark.parametrize(
    "call",
    (
        lambda: compare_scopes("unknown"),
        lambda: compare_scopes("mobile", sample_count=0),
        lambda: compare_scopes("mobile", kernel_speedup=0.5),
        lambda: compare_scopes("mobile", reported_scope="unknown"),
        lambda: analyze_repeats("mobile", repeat_count=1),
        lambda: analyze_repeats("mobile", warmup_iterations=-1),
        lambda: compare_sustained("mobile", duration=Q_(2, "meter")),
        lambda: compare_sustained("mobile", stress_scale=0),
        lambda: compare_quality_slices("mobile", threshold=1.2),
        lambda: compare_quality_slices("mobile", quality_floor_pct=101),
        lambda: compare_quality_slices("mobile", worst_slice_floor_pct=101),
        lambda: compare_quality_slices("mobile", candidate_mode="unknown"),
    ),
)
def test_invalid_experiment_inputs_fail_explicitly(call):
    with pytest.raises(ValueError):
        call()
