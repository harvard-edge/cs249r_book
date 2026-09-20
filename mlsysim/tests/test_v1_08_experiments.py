import json

import pytest

from mlsysim.engine.v1_08_experiments import (
    FIXTURE_NOTE,
    TRACKS,
    batch_experiment,
    batch_policy,
    bottleneck_experiment,
    checkpoint_experiment,
    compare_time_to_target,
    memory_outcome,
    optimizer_experiment,
    precision_experiment,
    replay,
    stage_timing,
    track_profile,
)


def test_all_tracks_use_relevant_upstream_scenarios_and_registry_memory_model():
    for track_id in TRACKS:
        profile = track_profile(track_id)
        outcome = memory_outcome(track_id)

        assert track_id in {"tinyml", "mobile", "edge", "cloud"}
        assert profile["task"]
        assert "upstream" in profile["task"].lower() or track_id == "cloud"
        assert outcome["model_name"] == "GPT-2 (1.5B)"
        assert outcome["total_memory_gb"] == pytest.approx(
            outcome["weights_gb"]
            + outcome["gradients_gb"]
            + outcome["optimizer_state_gb"]
            + outcome["activations_gb"]
            + outcome["communication_buffers_gb"],
            abs=5e-6,
        )
        assert outcome["fixture_note"] == FIXTURE_NOTE


def test_training_state_can_dominate_inference_weights_and_adamw_costs_more_than_sgd():
    for track_id in TRACKS:
        sgd = memory_outcome(track_id, optimizer="sgd")
        adamw = memory_outcome(track_id, optimizer="adamw")

        assert sgd["total_memory_gb"] > sgd["inference_weights_gb"]
        assert adamw["optimizer_state_gb"] > sgd["optimizer_state_gb"]
        assert adamw["total_memory_gb"] > sgd["total_memory_gb"]


def test_optimizer_experiment_uses_state_timing_and_its_own_supplied_trace():
    for track_id in TRACKS:
        sgd = optimizer_experiment(track_id, optimizer="sgd")
        adamw = optimizer_experiment(track_id, optimizer="adamw")

        assert sgd["memory"]["optimizer_state_gb"] < adamw["memory"]["optimizer_state_gb"]
        assert sgd["timing"]["optimizer_ms"] < adamw["timing"]["optimizer_ms"]
        assert sgd["updates_to_target"] > adamw["updates_to_target"]


def test_accumulation_repeats_microsteps_but_optimizer_runs_once_per_update():
    for track_id in TRACKS:
        one = stage_timing(track_id, physical_batch=8, accumulation_steps=1)
        four = stage_timing(track_id, physical_batch=8, accumulation_steps=4)

        assert four["effective_batch"] == 4 * one["effective_batch"]
        assert four["microstep_ms"] == one["microstep_ms"]
        assert four["optimizer_ms"] == one["optimizer_ms"]
        assert four["update_time_ms"] == pytest.approx(
            4 * one["microstep_ms"] + one["optimizer_ms"], abs=2e-6
        )
        assert four["update_time_ms"] < 4 * one["update_time_ms"]


def test_physical_and_effective_batch_are_not_interchangeable():
    for track_id in TRACKS:
        accumulated = batch_experiment(track_id, physical_batch=8, accumulation_steps=4)
        resident = batch_experiment(track_id, physical_batch=32, accumulation_steps=1)

        assert accumulated["timing"]["effective_batch"] == resident["timing"]["effective_batch"] == 32
        assert accumulated["updates_to_target"] == resident["updates_to_target"]
        assert accumulated["samples_presented"] == resident["samples_presented"]
        assert accumulated["memory"]["activations_gb"] < resident["memory"]["activations_gb"]
        assert accumulated["timing"]["update_time_ms"] != resident["timing"]["update_time_ms"]


def test_larger_effective_batch_can_reduce_updates_but_process_more_samples():
    for track_id in TRACKS:
        batch_32 = batch_experiment(track_id, physical_batch=8, accumulation_steps=4)
        batch_128 = batch_experiment(track_id, physical_batch=8, accumulation_steps=16)

        assert batch_128["updates_to_target"] < batch_32["updates_to_target"]
        assert batch_128["samples_presented"] > batch_32["samples_presented"]


def test_batch_with_higher_throughput_can_fail_to_reach_common_target():
    for track_id in TRACKS:
        baseline = batch_experiment(track_id, physical_batch=8, accumulation_steps=4)
        very_large = batch_experiment(track_id, physical_batch=8, accumulation_steps=64)

        assert baseline["reached_target"]
        assert not very_large["reached_target"]
        assert very_large["total_time_ms"] is None
        assert "target_quality" in very_large["violations"]

def test_batch_policy_derives_track_consistent_options_and_baseline():
    for track_id, settings in TRACKS.items():
        policy = batch_policy(track_id)
        assert policy["physical_batch"] == settings["physical_batch"]
        p = settings["physical_batch"]
        base_p, base_acc = policy["baseline"]
        assert base_p == p
        assert base_p * base_acc == 8
        assert policy["default_key"] == f"{p}x{32 // p}"
        assert "32x1" in policy["choices"]
        assert policy["choices"]["32x1"] == (32, 1)
        for choice_key, (phys, acc) in policy["choices"].items():
            outcome = batch_experiment(track_id, physical_batch=phys, accumulation_steps=acc)
            assert outcome["timing"]["physical_batch"] == phys
            assert outcome["timing"]["accumulation_steps"] == acc
            assert outcome["timing"]["effective_batch"] in {8, 32, 128, 512}


def test_precision_speed_memory_and_numerical_evidence_remain_separate():
    for track_id in TRACKS:
        fp32 = precision_experiment(track_id, precision="fp32")
        bf16 = precision_experiment(track_id, precision="bf16")
        fp8 = precision_experiment(track_id, precision="fp8")

        assert bf16["timing"]["update_time_ms"] < fp32["timing"]["update_time_ms"]
        assert bf16["memory"]["total_memory_gb"] < fp32["memory"]["total_memory_gb"]
        assert bf16["timing"]["optimizer_ms"] == fp32["timing"]["optimizer_ms"]
        assert bf16["reached_target"]
        assert not fp8["reached_target"]
        assert fp8["numerical_replay"]["finite_fraction"] < bf16["numerical_replay"]["finite_fraction"]
        assert fp8["observed_quality"] == 0.785


def test_checkpointing_saves_activations_by_adding_recomputation_work():
    for track_id in TRACKS:
        comparison = checkpoint_experiment(track_id, checkpointing="full")

        assert comparison["activation_memory_saved_gb"] > 0
        assert comparison["added_update_time_ms"] > 0
        assert comparison["result"]["memory"]["total_memory_gb"] < comparison["baseline"]["memory"]["total_memory_gb"]
        assert comparison["result"]["updates_to_target"] == comparison["baseline"]["updates_to_target"]


def test_checkpointing_can_make_a_resident_batch_feasible_on_every_track():
    for track_id in TRACKS:
        probe_batch = track_profile(track_id)["checkpoint_probe_batch"]
        no_checkpoint = memory_outcome(
            track_id, physical_batch=probe_batch, accumulation_steps=1, checkpointing="none"
        )
        full_checkpoint = memory_outcome(
            track_id, physical_batch=probe_batch, accumulation_steps=1, checkpointing="full"
        )

        assert not no_checkpoint["feasible"]
        assert full_checkpoint["feasible"]
        assert full_checkpoint["activations_gb"] < no_checkpoint["activations_gb"]


def test_interventions_change_only_their_physical_causes_and_expose_tradeoffs():
    for track_id in TRACKS:
        prefetch = bottleneck_experiment(track_id, intervention="prefetch")
        arithmetic = bottleneck_experiment(track_id, intervention="arithmetic")
        checkpoint = bottleneck_experiment(track_id, intervention="checkpoint")

        assert prefetch["result"]["timing"]["visible_input_ms"] < prefetch["baseline"]["timing"]["visible_input_ms"]
        assert prefetch["delta_memory_gb"] == 0
        assert arithmetic["result"]["timing"]["forward_ms"] < arithmetic["baseline"]["timing"]["forward_ms"]
        assert arithmetic["delta_memory_gb"] == 0
        assert checkpoint["delta_memory_gb"] < 0
        assert checkpoint["delta_update_time_ms"] > 0
        for comparison in (prefetch, arithmetic, checkpoint):
            assert comparison["result"]["trace"] == comparison["baseline"]["trace"]
            replay = comparison["result"]["inputs"]
            assert replay["track_id"] == track_id
            assert comparison["result"]["timing"]["inputs"]["input_policy"] == replay["input_policy"]
            assert comparison["result"]["timing"]["inputs"]["compute_policy"] == replay["compute_policy"]
            assert comparison["result"]["memory"]["inputs"]["optimizer"] == replay["optimizer"]


def test_optimizer_does_not_manufacture_a_batch_convergence_outcome():
    for track_id in TRACKS:
        sgd_memory = compare_time_to_target(
            track_id, evidence_family="effective_batch", evidence_key=32, optimizer="sgd"
        )
        adamw_memory = compare_time_to_target(
            track_id, evidence_family="effective_batch", evidence_key=32, optimizer="adamw"
        )

        assert sgd_memory["updates_to_target"] == adamw_memory["updates_to_target"]
        assert sgd_memory["observed_quality"] == adamw_memory["observed_quality"]
        assert sgd_memory["timing"]["forward_ms"] == adamw_memory["timing"]["forward_ms"]
        assert sgd_memory["timing"]["optimizer_ms"] < adamw_memory["timing"]["optimizer_ms"]
        assert sgd_memory["memory"]["optimizer_state_gb"] < adamw_memory["memory"]["optimizer_state_gb"]


def test_severe_input_starvation_can_break_schedule_and_cost_budgets():
    for track_id in TRACKS:
        starved = compare_time_to_target(
            track_id,
            evidence_family="effective_batch",
            evidence_key=32,
            input_policy="starved",
        )

        assert "schedule" in starved["violations"]
        assert "cost" in starved["violations"]


def test_unlisted_stage_configuration_is_not_extrapolated():
    with pytest.raises(ValueError, match="physical_batch"):
        stage_timing("edge", physical_batch=7)


def test_all_public_experiment_outcomes_are_strict_json_and_replay_ready():
    for track_id in TRACKS:
        outcomes = (
            optimizer_experiment(track_id, optimizer="adamw"),
            batch_experiment(track_id, physical_batch=8, accumulation_steps=4),
            precision_experiment(track_id, precision="bf16"),
            checkpoint_experiment(track_id, checkpointing="full"),
            bottleneck_experiment(track_id, intervention="prefetch"),
        )
        for outcome in outcomes:
            json.dumps(outcome, allow_nan=False)
        for outcome in outcomes[:3]:
            assert outcome["inputs"]["track_id"] == track_id
            assert outcome["timing"]["inputs"]["physical_batch"] == outcome["inputs"]["physical_batch"]
            assert outcome["memory"]["inputs"]["optimizer"] == outcome["inputs"]["optimizer"]


def test_actual_notebook_capture_outcomes_replay_after_json_roundtrip():
    keys_and_outcomes = []
    for track_id in TRACKS:
        optimizer_rows = [
            optimizer_experiment(track_id, optimizer=name)
            for name in ("sgd", "adam", "adamw")
        ]
        keys_and_outcomes.extend(
            ("v1_08_experiments.optimizer_experiment", outcome)
            for outcome in optimizer_rows
        )
        batch_rows = [
            batch_experiment(track_id, physical_batch=physical, accumulation_steps=accumulation)
            for physical, accumulation in batch_policy(track_id)["choices"].values()
        ]
        keys_and_outcomes.extend(
            ("v1_08_experiments.batch_experiment", outcome)
            for outcome in batch_rows
        )
        precision_rows = [
            precision_experiment(track_id, precision=name)
            for name in ("fp32", "bf16", "fp16", "fp8")
        ]
        keys_and_outcomes.extend(
            ("v1_08_experiments.precision_experiment", outcome)
            for outcome in precision_rows
        )
        probe = track_profile(track_id)["checkpoint_probe_batch"]
        memory_rows = [
            memory_outcome(
                track_id,
                physical_batch=probe,
                accumulation_steps=1,
                checkpointing=policy,
            )
            for policy in ("none", "selective", "full")
        ]
        keys_and_outcomes.extend(
            ("v1_08_experiments.memory_outcome", outcome)
            for outcome in memory_rows
        )
        for intervention in ("prefetch", "arithmetic", "checkpoint"):
            comparison = bottleneck_experiment(track_id, intervention=intervention)
            keys_and_outcomes.extend(
                (
                    ("v1_08_experiments.bottleneck_experiment", comparison["baseline"]),
                    ("v1_08_experiments.bottleneck_experiment", comparison["result"]),
                )
            )

    for model_key, outcome in keys_and_outcomes:
        captured = json.loads(json.dumps(outcome, allow_nan=False))
        replayed = json.loads(json.dumps(replay(model_key, captured["inputs"]), allow_nan=False))
        assert replayed == captured


def test_replay_rejects_unknown_methods_and_argument_injection():
    outcome = optimizer_experiment("edge", optimizer="adamw")
    with pytest.raises(ValueError, match="unknown"):
        replay("os.system", outcome["inputs"])
    with pytest.raises(ValueError, match="extra"):
        replay(
            "v1_08_experiments.optimizer_experiment",
            {**outcome["inputs"], "callable": "os.system"},
        )


@pytest.mark.parametrize(
    "call,args",
    [
        (track_profile, ("unknown",)),
        (batch_policy, ("unknown",)),
        (memory_outcome, ("edge",)),
        (stage_timing, ("edge",)),
        (optimizer_experiment, ("edge",)),
        (batch_experiment, ("edge",)),
        (precision_experiment, ("edge",)),
        (checkpoint_experiment, ("edge",)),
        (bottleneck_experiment, ("edge",)),
    ],
)
def test_invalid_inputs_fail_explicitly(call, args):
    kwargs = {
        memory_outcome: {"precision": "int2"},
        stage_timing: {"compute_policy": "magic"},
        optimizer_experiment: {"optimizer": "magic"},
        batch_experiment: {"physical_batch": 3, "accumulation_steps": 3},
        precision_experiment: {"precision": "int2"},
        checkpoint_experiment: {"checkpointing": "magic"},
        bottleneck_experiment: {"intervention": "buy_magic"},
    }.get(call, {})
    with pytest.raises(ValueError):
        call(*args, **kwargs)
