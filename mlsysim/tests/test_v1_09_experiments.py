import json
import math

import pytest

from mlsysim.core.units import Q_, ureg
from mlsysim.engine.v1_09_experiments import (
    TRACK_IDS,
    TRACKS,
    acquisition_budget_policy,
    acquisition_option,
    acquisition_options,
    compare_policies,
    evaluate_policy,
    full_pool_baseline,
    learning_curve,
    population_shift,
    replay,
    scoring_time_policy,
    selection_amortization,
    serialize_value,
)


def test_all_track_learning_curves_show_supplied_saturation_contrast():
    for track_id in TRACK_IDS:
        redundant = learning_curve(track_id, "redundant")
        informative = learning_curve(track_id, "informative")

        assert redundant["points"][-1]["quality_pct"] < informative["points"][-1]["quality_pct"]
        assert redundant["points"][-1]["marginal_quality_pp"] <= 0.1
        assert informative["points"][-1]["marginal_quality_pp"] > redundant["points"][-1]["marginal_quality_pp"]
        assert "illustrative" in redundant["observation_label"].lower()


def test_all_track_policies_keep_equal_counts_but_change_duplicate_and_coverage_evidence():
    for track_id in TRACK_IDS:
        comparison = compare_policies(track_id, retained_count=8)
        policies = {item["policy_id"]: item for item in comparison["policies"]}

        assert {item["retained_count"] for item in comparison["policies"]} == {8}
        assert policies["uniform"]["duplicate_records"] == 1
        assert policies["deduplicate"]["duplicate_records"] == 0
        assert policies["coverage"]["cohort_counts"]["rare"] > policies["deduplicate"]["cohort_counts"]["rare"]
        assert policies["coverage"]["feasible"]
        assert not policies["uniform"]["feasible"]


def test_no_selection_baseline_uses_null_with_explicit_unavailable_status():
    baseline = full_pool_baseline("tinyml")

    assert baseline["retained_count"] == 12
    assert baseline["cohort_outcomes_pct"] is None
    assert baseline["weighted_quality_pct"] is None
    assert baseline["feasible"] is None
    assert baseline["quality_status"].startswith("unavailable")


def test_policy_quality_is_supplied_and_does_not_come_from_record_byte_size():
    tiny = evaluate_policy("tinyml", "coverage", 8)
    cloud = evaluate_policy("cloud", "coverage", 8)

    assert tiny["retained_bytes"].to("megabyte").magnitude != cloud["retained_bytes"].to("megabyte").magnitude
    assert tiny["cohort_outcomes_pct"] == {"common": 78, "rare": 69, "edge_case": 66}
    assert cloud["cohort_outcomes_pct"] == {"common": 82, "rare": 76, "edge_case": 72}


def test_amortization_sums_read_scoring_selection_and_repeated_training():
    result = selection_amortization("mobile", "coverage", retained_count=8, repeated_runs=3)

    expected_overhead = result["read_time"] + result["scoring_time"] + result["selection_time"]
    expected_selected = expected_overhead + 3 * result["subset_training_time_per_run"]
    expected_baseline = 3 * result["full_training_time_per_run"]
    assert result["selection_overhead"].to("second").magnitude == pytest.approx(expected_overhead.to("second").magnitude)
    assert result["selected_total"].to("second").magnitude == pytest.approx(expected_selected.to("second").magnitude)
    assert result["baseline_total"].to("second").magnitude == pytest.approx(expected_baseline.to("second").magnitude)


def test_amortization_paths_have_distinct_replayable_inputs():
    baseline = selection_amortization(
        "mobile", "coverage", 8, repeated_runs=3, comparison_path="full_pool"
    )
    selected = selection_amortization(
        "mobile", "coverage", 8, repeated_runs=3, comparison_path="selected_subset"
    )

    assert baseline["inputs"] != selected["inputs"]
    assert baseline["reported_total"] == baseline["baseline_total"]
    assert selected["reported_total"] == selected["selected_total"]


def test_scoring_cost_and_repeated_runs_cross_the_break_even_boundary():
    cheap_many = selection_amortization(
        "mobile", "coverage", retained_count=8,
        scoring_time_per_example=Q_(0, "microsecond"), repeated_runs=20,
    )
    expensive_once = selection_amortization(
        "mobile", "coverage", retained_count=8,
        scoring_time_per_example=Q_(2, "millisecond"), repeated_runs=1,
    )

    assert cheap_many["selection_pays"]
    assert not expensive_once["selection_pays"]
    assert expensive_once["selection_overhead"] > cheap_many["selection_overhead"]


def test_first_profitable_run_uses_strict_inequality_and_policy_gate_is_separate():
    valid = selection_amortization(
        "cloud", "coverage", retained_count=8,
        scoring_time_per_example=Q_(0, "microsecond"), repeated_runs=20,
    )
    invalid = selection_amortization(
        "cloud", "uniform", retained_count=8,
        scoring_time_per_example=Q_(0, "microsecond"), repeated_runs=20,
    )
    before = selection_amortization(
        "cloud", "coverage", retained_count=8,
        scoring_time_per_example=Q_(0, "microsecond"),
        repeated_runs=valid["first_profitable_runs"] - 1,
    ) if valid["first_profitable_runs"] > 1 else None

    assert valid["selection_pays"]
    assert valid["quality_and_coverage_accepted"]
    assert valid["decision_supported"]
    assert invalid["selection_pays"]
    assert not invalid["quality_and_coverage_accepted"]
    assert not invalid["decision_supported"]
    if before is not None:
        assert not before["selection_pays"]
    assert "not a time-to-common-quality" in valid["training_assumption"]
    assert "exclude training I/O" in valid["io_boundary"]


def test_equal_cost_boundary_is_not_reported_as_profitable():
    probe = selection_amortization(
        "mobile", "coverage", retained_count=8,
        scoring_time_per_example=Q_(0, "second"), repeated_runs=1,
    )
    saving_per_run = probe["full_training_time_per_run"] - probe["subset_training_time_per_run"]
    fixed_overhead = probe["selection_overhead"]
    boundary_runs = math.ceil((fixed_overhead / saving_per_run).to_base_units().magnitude) + 2
    required_scoring_total = boundary_runs * saving_per_run - fixed_overhead
    scoring_per_example = required_scoring_total / TRACKS["mobile"]["represented_examples"]

    equal = selection_amortization(
        "mobile", "coverage", retained_count=8,
        scoring_time_per_example=scoring_per_example, repeated_runs=boundary_runs,
    )

    assert equal["selected_total"].to("second").magnitude == pytest.approx(
        equal["baseline_total"].to("second").magnitude
    )
    assert not equal["selection_pays"]
    assert equal["first_profitable_runs"] == boundary_runs + 1


def test_retention_changes_training_cost_while_policy_name_is_noncausal_for_time():
    six = selection_amortization("edge", "coverage", retained_count=6, repeated_runs=4)
    ten = selection_amortization("edge", "coverage", retained_count=10, repeated_runs=4)
    same_time_other_policy = selection_amortization("edge", "uniform", retained_count=6, repeated_runs=4)

    assert six["subset_training_time_per_run"] < ten["subset_training_time_per_run"]
    assert six["selected_total"] < ten["selected_total"]
    assert six["selected_total"] == same_time_other_policy["selected_total"]


def test_acquisition_options_separate_creation_validation_turnaround_and_outcomes():
    for track_id in TRACK_IDS:
        result = acquisition_options(track_id, Q_(10_000, "USD"))
        labels, generated = result["options"]

        for option in result["options"]:
            assert option["total_cost"] == option["creation_cost"] + option["validation_cost"]
            assert option["turnaround"].check(ureg.hour)
            assert option["validated_examples"] > 0
        assert labels["created_examples"] < generated["created_examples"]
        assert labels["cohort_outcomes_pct"]["rare"] > generated["cohort_outcomes_pct"]["rare"]


@pytest.mark.parametrize(
    ("track_id", "middle_budget"),
    [
        ("tinyml", Q_(2_000, "USD")),
        ("mobile", Q_(4_500, "USD")),
        ("edge", Q_(12_000, "USD")),
        ("cloud", Q_(25_000, "USD")),
    ],
)
def test_every_track_has_an_affordable_generated_package_and_unaffordable_label_package(
    track_id, middle_budget
):
    result = acquisition_options(track_id, middle_budget)

    assert [option["affordable"] for option in result["options"]] == [False, True]


def test_budget_changes_affordability_but_not_supplied_quality_observations():
    low = acquisition_options("mobile", Q_(4_500, "USD"))
    high = acquisition_options("mobile", Q_(7_000, "USD"))

    assert [item["affordable"] for item in low["options"]] == [False, True]
    assert [item["affordable"] for item in high["options"]] == [True, True]
    assert [item["cohort_outcomes_pct"] for item in low["options"]] == [
        item["cohort_outcomes_pct"] for item in high["options"]
    ]


def test_single_acquisition_candidates_have_distinct_replayable_inputs():
    labels = acquisition_option("edge", "labels", Q_(20_000, "USD"))
    generated = acquisition_option("edge", "generated", Q_(20_000, "USD"))

    assert labels["inputs"] != generated["inputs"]
    assert labels["option_id"] == "labels"
    assert generated["option_id"] == "generated"


def test_no_purchase_uses_zero_cost_and_null_quality_with_explicit_status():
    none = acquisition_option("edge", "none", Q_(20_000, "USD"))

    assert none["total_cost"] == Q_(0, "USD")
    assert none["cohort_outcomes_pct"] is None
    assert none["weighted_quality_pct"] is None
    assert none["quality_status"].startswith("unavailable")


def test_population_shift_reweights_fixed_outcomes_and_can_invalidate_support():
    baseline = population_shift("edge", "coverage", 8, rare_share_pct=18)
    shifted = population_shift("edge", "coverage", 8, rare_share_pct=35)

    assert baseline["cohort_outcomes_pct"] == shifted["cohort_outcomes_pct"]
    assert baseline["changed_quality_pct"] == baseline["baseline_quality_pct"]
    assert shifted["changed_quality_pct"] < baseline["changed_quality_pct"]
    assert shifted["underrepresented_cohorts"] == ("rare",)
    assert baseline["still_supported"]
    assert not shifted["still_supported"]


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_every_track_population_endpoint_has_supported_baseline_and_failed_shift(track_id):
    baseline_rare_share = {
        "tinyml": 16,
        "mobile": 20,
        "edge": 18,
        "cloud": 22,
    }[track_id]

    baseline = population_shift(track_id, "coverage", 8, baseline_rare_share)
    shifted = population_shift(track_id, "coverage", 8, 35)

    assert baseline["still_supported"]
    assert not shifted["still_supported"]
    assert shifted["underrepresented_cohorts"] == ("rare",)


@pytest.mark.parametrize("track_id", TRACK_IDS)
def test_every_track_amortization_endpoint_has_opposite_sides(track_id):
    cheap_many = selection_amortization(
        track_id, "coverage", 6,
        scoring_time_per_example=Q_(0, "microsecond"), repeated_runs=100,
    )
    expensive_once = selection_amortization(
        track_id, "coverage", 10,
        scoring_time_per_example=Q_(20, "millisecond"), repeated_runs=1,
    )

    assert cheap_many["selection_pays"]
    assert not expensive_once["selection_pays"]


def test_scoring_time_policy_defines_selectable_defaults_and_valid_bounds():
    for track_id in TRACK_IDS:
        policy = scoring_time_policy(track_id)
        default_us = policy["default_scoring_time"].to("microsecond").magnitude
        step_us = policy["step_scoring_time"].to("microsecond").magnitude
        min_us = policy["min_scoring_time"].to("microsecond").magnitude
        max_us = policy["max_scoring_time"].to("microsecond").magnitude

        assert min_us <= default_us <= max_us
        assert math.isclose(default_us % step_us, 0.0, abs_tol=1e-9)
        assert default_us in (8, 18, 32, 55)


def test_acquisition_budget_policy_scales_bounds_and_preserves_meaningful_default_choice():
    for track_id in TRACK_IDS:
        policy = acquisition_budget_policy(track_id)
        options_default = acquisition_options(track_id, policy["default_budget"])["options"]
        options_min = acquisition_options(track_id, policy["min_budget"])["options"]
        options_max = acquisition_options(track_id, policy["max_budget"])["options"]

        assert [opt["affordable"] for opt in options_default] == [False, True]
        assert [opt["affordable"] for opt in options_min] == [False, False]
        assert [opt["affordable"] for opt in options_max] == [True, True]


def test_every_public_experiment_result_records_exact_replay_inputs():
    results = (
        learning_curve("tinyml", "informative"),
        evaluate_policy("mobile", "coverage", 8),
        compare_policies("edge", 10),
        full_pool_baseline("cloud"),
        selection_amortization(
            "cloud", "deduplicate", 6,
            scoring_time_per_example=Q_(125, "microsecond"), repeated_runs=4,
        ),
        acquisition_options("tinyml", Q_(4_000, "USD")),
        acquisition_option("edge", "labels", Q_(20_000, "USD")),
        population_shift("mobile", "coverage", 8, 32),
    )

    for result in results:
        assert result["inputs"]["track_id"] == result["track_id"]
    assert results[4]["inputs"]["policy_id"] == "deduplicate"
    assert results[4]["inputs"]["retained_count"] == 6
    assert results[4]["inputs"]["repeated_runs"] == 4
    assert results[4]["inputs"]["scoring_time_per_example"].to("microsecond").magnitude == 125
    assert results[5]["inputs"]["budget"] == Q_(4_000, "USD")
    assert results[7]["inputs"]["rare_share_pct"] == 32


def test_every_method_serializes_to_json_and_replays_exact_inputs():
    originals = (
        learning_curve("tinyml", "informative"),
        evaluate_policy("mobile", "coverage", 8),
        compare_policies("edge", 10),
        full_pool_baseline("cloud"),
        selection_amortization(
            "cloud", "deduplicate", 6,
            scoring_time_per_example=Q_(125, "microsecond"), repeated_runs=4,
        ),
        acquisition_options("tinyml", Q_(4_000, "USD")),
        acquisition_option("edge", "generated", Q_(20_000, "USD")),
        population_shift("mobile", "coverage", 8, 32),
    )

    for original in originals:
        frozen = json.loads(json.dumps(serialize_value(original)))
        replayed = replay(original["model_key"], frozen["inputs"])
        assert replayed == original


def test_quantity_snapshot_keeps_recoverable_magnitude_and_unit():
    original = selection_amortization(
        "mobile", "coverage", 8,
        scoring_time_per_example=Q_(125, "microsecond"), repeated_runs=4,
    )
    frozen = serialize_value(original)

    assert frozen["inputs"]["scoring_time_per_example"] == {
        "__type__": "quantity",
        "magnitude": pytest.approx(0.000125),
        "unit": "s",
    }


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: learning_curve("unknown", "redundant"), "track_id"),
        (lambda: learning_curve("tinyml", "invented"), "pool_kind"),
        (lambda: evaluate_policy("tinyml", "invented", 8), "policy_id"),
        (lambda: evaluate_policy("tinyml", "coverage", 7), "retained_count"),
        (lambda: scoring_time_policy("unknown"), "track_id"),
        (lambda: selection_amortization("tinyml", "coverage", repeated_runs=0), "repeated_runs"),
        (lambda: selection_amortization("tinyml", "coverage", scoring_time_per_example=Q_(1, "byte")), "time Quantity"),
        (lambda: acquisition_budget_policy("unknown"), "track_id"),
        (lambda: acquisition_options("tinyml", Q_(1, "second")), "currency Quantity"),
        (lambda: population_shift("tinyml", "coverage", 8, 95), "invalid common"),
    ],
)
def test_invalid_inputs_fail_with_explanations(call, message):
    with pytest.raises(ValueError, match=message):
        call()
