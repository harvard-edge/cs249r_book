"""Causal and provenance invariants of deterministic agent lab studies."""
from concurrent.futures import ThreadPoolExecutor
import pytest

from mlsysim.labs.experiments.v3_studies import choices, definition, projects, replay_guards, study


def test_permission_unknown_is_not_a_pass_and_open_policy_exposes_effect():
    records = [
        {"event": "known grant", "requirements": {"grant": True}, "effect": "visible"},
        {"event": "unknown grant", "requirements": {"grant": None}, "effect": "unverified effect"},
        {"event": "denied grant", "requirements": {"grant": False}, "effect": "forbidden effect"},
    ]
    opened = replay_guards(records, [])
    guarded = replay_guards(records, ["grant"])
    assert opened["admitted"] == len(records)
    assert opened["violations"] == 1
    assert guarded["admitted"] == 1
    assert guarded["violations"] == 0
    assert guarded["held_unknown"] == 1
    assert guarded["events"][1]["decision"] == "hold"
    assert "unverified effect" not in guarded["events"][1]["state"]


def test_extra_independent_guard_cannot_increase_dispatches():
    records = [
        {"event": "allowed", "requirements": {"scope": True, "version": True}, "effect": "write"},
        {"event": "stale", "requirements": {"scope": True, "version": False}, "effect": "overwrite"},
        {"event": "out of scope", "requirements": {"scope": False, "version": True}, "effect": "escape"},
    ]
    scope = replay_guards(records, ["scope"])
    both = replay_guards(records, ["scope", "version"])
    assert both["admitted"] < scope["admitted"]
    assert both["violations"] == 0 < scope["violations"]
    assert both["events"][0]["state"] == scope["events"][0]["state"]


def test_guard_rejects_invalid_evidence_instead_of_truthy_conversion():
    with pytest.raises(ValueError):
        replay_guards([{"event": "bad", "requirements": {"scope": "yes"}, "effect": "write"}], ["scope"])
    with pytest.raises(ValueError):
        replay_guards([{"event": "bad", "requirements": {"scope": True}, "effect": "write"}], ["nonexistent"])


def test_four_lighthouses_have_distinct_registry_identity():
    catalog = projects()
    assert set(catalog) == {"repair", "research", "remediation", "support"}
    assert len({p["registry"] for p in catalog.values()}) == len(catalog)


@pytest.mark.parametrize("chapter", range(1, 21))
@pytest.mark.parametrize("part", "ABCD")
def test_all_policy_choices_are_deterministic_isolated_and_have_evidence(chapter, part):
    options = choices(chapter, part)
    assert len(options) >= 2
    for choice in options:
        args = (chapter, part, choice)
        first = study(*args)
        assert first == study(*args)
        assert first["rows"][0]["policy"] == next(iter(options.values()))
        assert first["events"] and first["feedback"] and first["limits"]
        first["events"][0]["event"] = "mutated external result"
        assert study(*args)["events"][0]["event"] != "mutated external result"
    alternate = study(chapter, part, next(iter(options)), "research")
    assert alternate["source"]["registry"] != study(chapter, part, next(iter(options)))["source"]["registry"]


def test_concurrent_track_choices_do_not_mutate_shared_catalog():
    args = [(7, "A", choice, track) for track in projects() for choice in choices(7, "A")]
    with ThreadPoolExecutor() as executor:
        actual = list(executor.map(lambda a: study(*a), args))
    assert actual == [study(*a) for a in args]


def test_unsupported_id_is_an_error_not_fallback():
    with pytest.raises(ValueError): study(99, "A", "baseline")
    with pytest.raises(ValueError): study(7, "A", "nonexistent")
    with pytest.raises(ValueError): choices(7, "A", "cloud")


def test_missing_selected_guard_is_unknown_not_valid_block():
    result = replay_guards([
        {"event": "complete", "requirements": {"scope": True, "version": True}, "effect": "write"},
        {"event": "missing version", "requirements": {"scope": True}, "effect": "write"},
    ], ["scope", "version"])
    assert result["admitted"] == 1
    assert result["held_unknown"] == 1
    assert result["blocked_valid"] == 0


def test_hold_reconcile_and_retry_have_distinct_effect_and_progress_cost():
    hold = study(7, "B", "hold")["rows"][-1]
    reconcile = study(7, "B", "reconcile")["rows"][-1]
    retry = study(7, "B", "retry")["rows"][-1]
    assert hold["endpoint_effects"] == reconcile["endpoint_effects"]
    assert retry["endpoint_effects"] > reconcile["endpoint_effects"]
    assert reconcile["probes"] > hold["probes"]
    assert hold["obligation"] == "in doubt"
    assert reconcile["obligation"] == "established"


def test_stricter_grader_threshold_reduces_acceptance_and_can_lose_valid_work():
    permissive = study(13, "C", "permissive")["rows"][-1]
    strict = study(13, "C", "strict")["rows"][-1]
    assert strict["grader_acceptances"] < permissive["grader_acceptances"]
    assert strict["false_acceptances"] < permissive["false_acceptances"]
    assert strict["false_rejections"] > permissive["false_rejections"]
    assert permissive["false_acceptance_fraction"] != permissive["failure_share_among_accepted"]


def test_failed_attempts_raise_true_cost_without_creating_acceptances():
    from mlsysim.labs.experiments.v3_studies import cohort_accounting
    accepted = [{"cost_usd": "0.4", "accepted": True}]
    extended = accepted + [{"cost_usd": "0.3", "accepted": False}]
    before = cohort_accounting(accepted)
    after = cohort_accounting(extended)
    excluded = cohort_accounting(extended, "accepted_only")
    assert before["accepted_tasks"] == after["accepted_tasks"]
    assert float(after["actual_cost_per_accepted_usd"]) > float(before["actual_cost_per_accepted_usd"])
    assert after["actual_cost_per_accepted_usd"] == excluded["actual_cost_per_accepted_usd"]
    assert after["reported_spend_usd"] != excluded["reported_spend_usd"]
    zero = cohort_accounting([{"cost_usd": "0.3", "accepted": False}])
    assert zero["actual_cost_per_accepted_usd"].startswith("undefined")
    with pytest.raises(ValueError): cohort_accounting([{"cost_usd": -1, "accepted": False}])


def test_block_rounding_trades_table_entries_for_unused_slots_at_fixed_content():
    small = study(5, "A", "small")["rows"][-1]
    large = study(5, "A", "large")["rows"][-1]
    assert small["useful_tokens"] == large["useful_tokens"]
    assert small["allocated_blocks"] > large["allocated_blocks"]
    assert small["unused_token_slots"] < large["unused_token_slots"]
    assert small["resident_kv_bytes"] < large["resident_kv_bytes"]
    assert small["declared_pool_fits"] and not large["declared_pool_fits"]


def test_paused_kv_eviction_exchanges_residency_for_exact_prefix_recomputation():
    retained = study(5, "D", "retain")["rows"][-1]
    evicted = study(5, "D", "evict")["rows"][-1]
    assert retained["kv_bytes_while_paused"] > evicted["kv_bytes_while_paused"] == 0
    assert retained["resume_prefill_tokens"] == 0 < evicted["resume_prefill_tokens"]
    assert retained["kv_byte_seconds"] > evicted["kv_byte_seconds"]


def test_orientation_and_transfer_scope_precede_engineering_prediction():
    for part in "ABCD":
        spec = definition(0, part, "support")
        assert not spec["source_concepts"]
        assert spec["coverage"]["status"] == "orientation only"
    spec = definition(7, "A", "support")
    assert "bounded transfer" in spec["transfer_scope"]
    assert not spec["scene"].startswith("Transfer from the repair-primary reference")


def test_harness_closure_needs_current_record_and_settled_duty_not_prompt_eligibility():
    baseline = study(10, "B", "baseline")
    guarded = study(10, "B", "trial")
    assert baseline["rows"][-1]["false_requirement_dispatches"] > 0
    assert guarded["rows"][-1]["false_requirement_dispatches"] == 0
    events = guarded["events"]
    unsettled = next(e for e in events if "removing a late result" in e["event"])
    stale = next(e for e in events if "stale record" in e["event"])
    settled = next(e for e in events if "authentic late result" in e["event"])
    assert unsettled["decision"] == stale["decision"] == "hold"
    assert unsettled["observation_eligible"] is False
    assert settled["observation_eligible"] is False
    assert settled["decision"] == "admit"
    assert "not staged" in settled["state"]


def test_harness_favorable_review_cannot_restore_withdrawn_authority():
    guarded = study(10, "C", "trial")
    withdrawn = next(e for e in guarded["events"] if "task grant was withdrawn" in e["event"])
    assert withdrawn["decision"] == "hold"
    assert withdrawn["blocking_evidence"] == "current_admission"
    assert guarded["rows"][-1]["false_requirement_dispatches"] == 0


@pytest.mark.parametrize("chapter", range(21))
@pytest.mark.parametrize("part", "ABCD")
def test_every_investigation_has_specific_predictions_and_an_observable_intervention(chapter, part):
    spec = definition(chapter, part)
    predictions = spec["prediction_options"]
    assert len(predictions) >= 2
    assert len(predictions) == len(set(predictions))
    assert "The proposed change will improve the declared outcome" not in predictions
    for choice in list(choices(chapter, part))[1:]:
        result = study(chapter, part, choice)
        assert (result["events"] != result["baseline_events"] or
                result["rows"][0]["outcome"] != result["rows"][1]["outcome"])


def test_prefix_reuse_exposes_changed_work_without_inventing_price_or_latency():
    result = study(2, "D", "trial")
    start, trial = result["rows"]
    assert start["outcome"] == trial["outcome"] == "accepted"
    assert start["fresh_prefix_work"] != trial["fresh_prefix_work"]
    assert start["retained_prefix_reuse"] != trial["retained_prefix_reuse"]
    assert start["latency_and_bill"] == trial["latency_and_bill"]
    assert start["latency_and_bill"].startswith("Unknown")
    assert "fresh_prefix_work" in result["primary_fields"].values()


@pytest.mark.parametrize("chapter,part", [(2,"D"),(5,"A"),(5,"D"),(7,"B"),(13,"C"),(19,"A"),(19,"D")])
def test_primary_teaching_fields_exist_in_both_comparisons(chapter, part):
    for choice in choices(chapter, part):
        result = study(chapter, part, choice)
        assert result["primary_fields"]
        for field in result["primary_fields"].values():
            assert all(field in row for row in result["rows"])
