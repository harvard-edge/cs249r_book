from __future__ import annotations

import copy
import json

import pytest

from tools import import_reference_evidence as importer


SOURCE_SHA = "a" * 40
TOOL_SHA = "sha256:" + "b" * 64
FINGERPRINT = "c" * 64


def _host_power() -> dict:
    snapshot = {
        "schema": importer.HOST_POWER_STATE_SCHEMA,
        "platform": "Darwin",
        "captured_at": "2026-07-14T12:00:00+00:00",
        "provider": "macos-pmset-sysctl",
        "supported": True,
        "source": "external",
        "source_raw": "AC Power",
        "battery_percent": 100,
        "battery_status": "charged",
        "power_mode": 0,
        "low_power_mode": False,
        "last_sleep_epoch": 100,
        "last_wake_epoch": 101,
        "suspend_clock_offset_seconds": None,
        "query_errors": [],
    }
    return {
        "policy": dict(importer.POWER_STABILITY_POLICY),
        "promotion_conditions_required": True,
        "before": dict(snapshot),
        "after": dict(snapshot),
        "stable": True,
        "invalid_reasons": [],
    }


def _row(case: importer.EvidenceCase, position: int, primary: float) -> dict:
    gate_value = float(case.gate["target"])
    if case.gate["direction"] == "higher":
        gate_value += 0.01
    elif case.gate["direction"] == "lower":
        gate_value -= 0.001
    promotion = {
        "status": "passed",
        "promotion_eligible": True,
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
        "issues": [],
    }
    return {
        "execution_index": position,
        "requested_seed": case.canonical_seed,
        "report_recorded_seed": case.canonical_seed,
        "manifest_recorded_seed": case.canonical_seed,
        "seed_match": True,
        "status": "passed",
        "execution_ok": True,
        "evidence_valid": True,
        "timed_out": False,
        "manifest_verified": True,
        "data_mode": case.data_mode,
        "scenario": case.scenario,
        "manifest_scenario": case.scenario,
        "registry_scenario": case.scenario,
        "primary_metric_declared": case.measurement_protocol["primary_metric"],
        "primary_metric_key": case.measurement_protocol["primary_metric"],
        "primary_metric_value": primary,
        "reference_metric_role": "performance",
        "result_role": case.result_role,
        "quality_metric_declared": case.gate["metric"],
        "quality_metric_key": case.gate["metric_key"],
        "quality_value": gate_value,
        "functional_metric_declared": case.gate["metric"],
        "functional_metric_key": case.gate["metric_key"],
        "functional_metric_value": gate_value,
        "quality_target_met": True,
        "wall_seconds": primary + 1.0,
        "comparison_fingerprint_sha256": FINGERPRINT,
        "promotion_contract": promotion,
        "grade": {
            "passed": True,
            "target_met": True,
            "target": case.gate["target"],
        },
        "host_power": _host_power(),
        "invalid_reasons": [],
    }


def _summary(case: importer.EvidenceCase) -> dict:
    primary_values = [10.00, 10.05, 10.10, 10.15, 10.20]
    rows = [
        _row(case, position, value)
        for position, value in enumerate(primary_values, start=1)
    ]
    cooldown_seconds = float(
        case.measurement_protocol["outer_inter_execution_cooldown_seconds"]
    )
    outer_executions = [
        {
            "execution_index": position,
            "seed": case.canonical_seed,
            "fresh_process": True,
            "cooldown_before_seconds": 0.0 if position == 1 else cooldown_seconds,
        }
        for position in range(1, 6)
    ]
    for row, execution in zip(rows, outer_executions, strict=True):
        row["outer_process_execution"] = execution
        row["reproduce"] = {
            "reference_sweep": {
                "outer_process_execution": execution,
                "inter_execution_cooldown": {
                    "cli_option": "--inter-execution-cooldown-seconds",
                    "configured_seconds": cooldown_seconds,
                    "applied_before_this_execution_seconds": execution[
                        "cooldown_before_seconds"
                    ],
                    "applies": True,
                },
                "timing_scope": importer.OUTER_EXECUTION_TIMING_SCOPE,
            }
        }
    wall_values = [float(row["wall_seconds"]) for row in rows]
    quality_values = [float(row["quality_value"]) for row in rows]
    preconditioning_count = int(
        case.measurement_protocol.get("outer_preconditioning_runs", 0)
    )
    preconditioning_executions = [
        {
            "execution_index": position,
            "seed": case.canonical_seed,
            "fresh_process": True,
            "output_group": "preconditioning",
        }
        for position in range(1, preconditioning_count + 1)
    ]
    preconditioning_rows = []
    for execution in preconditioning_executions:
        row = _row(case, execution["execution_index"], 20.0)
        row["preconditioning_execution"] = execution
        row["reproduce"] = {
            "reference_sweep": {
                "preconditioning_execution": execution,
                "aggregate_inclusion": "excluded",
                "timing_scope": importer.PRECONDITIONING_TIMING_SCOPE,
            }
        }
        preconditioning_rows.append(row)
    score = case.result_role == "score-bearing"
    functional_gate = {
        "metric": case.gate["metric"],
        "metric_key": case.gate["metric_key"],
        "condition": "Every run must pass the canonical functional gate.",
        "target": case.gate["target"],
    }
    return {
        "schema": importer.SUMMARY_SCHEMA,
        "evidence_id": f"{case.workload.id}_max_20260712T120000.000000Z",
        "status": "valid",
        "evidence_tier": "promotion-candidate",
        "eligible_for_promotion": True,
        "eligible_for_public_baseline": False,
        "invalid_reasons": [],
        "workload": case.workload.id,
        "canonical_workload": case.workload.id,
        "profile": case.profile,
        "mode": case.mode,
        "phase": case.phase,
        "result_role": case.result_role,
        "seeds_requested": [case.canonical_seed] * 5,
        "power_stability_policy": dict(importer.POWER_STABILITY_POLICY),
        "inter_execution_stabilization": {
            "scope": "outer-process-executions",
            "applies": True,
            "applicability": (
                "all public-candidate score-bearing and performance-bearing workloads"
            ),
            "mode": "fixed-delay-between-fresh-processes",
            "execution_unit": "one fresh Python subprocess per repetition",
            "process_execution_count": 5,
            "configured_cooldown_seconds": cooldown_seconds,
            "maximum_cooldown_seconds": 300.0,
            "first_execution_has_no_cooldown": True,
            "timing_scope": importer.OUTER_EXECUTION_TIMING_SCOPE,
            "executions": outer_executions,
        },
        "preconditioning": {
            "scope": "outer-process-preconditioning",
            "applies": bool(preconditioning_count),
            "mode": (
                "complete-canonical-executions"
                if preconditioning_count
                else "not-applied"
            ),
            "execution_unit": (
                "one complete canonical workload in a fresh Python subprocess"
            ),
            "process_execution_count": preconditioning_count,
            "cooldown_between_executions_seconds": 0.0,
            "cooldown_before_first_measured_execution_seconds": 0.0,
            "timing_scope": importer.PRECONDITIONING_TIMING_SCOPE,
            "executions": preconditioning_executions,
            "runs": preconditioning_rows,
        },
        "comparison_fingerprint_sha256": FINGERPRINT,
        "primary_metric": {
            "name": case.measurement_protocol["primary_metric"],
            "role": "performance",
        },
        "quality_metric": case.gate["metric"] if score else None,
        "quality_target": case.gate["target"] if score else None,
        "quality_direction": case.gate["direction"] if score else None,
        "quality_gate": (
            {
                "metric": case.gate["metric"],
                "target": case.gate["target"],
                "direction": case.gate["direction"],
                "tolerance": float(case.workload.quality_tolerance or 0.0),
                "all_runs_must_pass": True,
            }
            if score
            else None
        ),
        "functional_gate": None if score else functional_gate,
        "aggregate": {
            "primary_metric": importer.aggregate(primary_values),
            "quality": importer.aggregate(quality_values) if score else None,
            "wall_seconds": importer.aggregate(wall_values),
        },
        "primary_metric_repeatability": importer._repeatability(primary_values, case),
        "acceptance": (
            {
                "passed": True,
                "all_runs_passed": True,
                "value": quality_values[2],
            }
            if score
            else {"passed": True, "value": 5}
        ),
        "runs": rows,
        "source": {
            "git_sha": SOURCE_SHA,
            "git_dirty": False,
            "git_status_sha256": importer.EMPTY_SHA256,
            "git_patch_sha256": importer.EMPTY_SHA256,
            "tool_sha256": TOOL_SHA,
        },
    }


def test_expected_case_closure_has_nine_workloads_and_twelve_cases():
    cases = importer.expected_cases()

    assert len(cases) == 12
    assert len({case.workload.id for case in cases.values()}) == 9
    assert {
        "code-generation",
        "function-calling",
        "recommendation",
        "image-generation",
        "reinforcement-learning",
    }.isdisjoint(case.workload.id for case in cases.values())
    assert cases["causal-language-modeling__max__training"].result_role == (
        "score-bearing"
    )
    for phase in ("full", "prefill", "decode"):
        assert (
            cases[f"causal-language-modeling__max__inference__{phase}"].result_role
            == "performance-bearing"
        )


def test_case_id_rejects_unsafe_identity_components():
    with pytest.raises(ValueError, match="unsafe evidence case id"):
        importer.case_id("../escape", "max", "training", None)


def test_score_summary_validation_recomputes_aggregates_and_roles(tmp_path):
    case = importer.expected_cases()["image-classification__max__inference"]
    payload = _summary(case)
    path = tmp_path / "evidence_summary.json"

    importer.validate_summary_structure(
        path,
        payload,
        case=case,
        source_git_sha=SOURCE_SHA,
        sweep_tool_sha256=TOOL_SHA,
    )

    tampered = copy.deepcopy(payload)
    tampered["aggregate"]["primary_metric"]["median"] = 999.0
    with pytest.raises(ValueError, match="primary aggregate"):
        importer.validate_summary_structure(
            path,
            tampered,
            case=case,
            source_git_sha=SOURCE_SHA,
            sweep_tool_sha256=TOOL_SHA,
        )

    tampered = copy.deepcopy(payload)
    tampered["inter_execution_stabilization"]["configured_cooldown_seconds"] = 0
    with pytest.raises(ValueError, match="configured_cooldown_seconds"):
        importer.validate_summary_structure(
            path,
            tampered,
            case=case,
            source_git_sha=SOURCE_SHA,
            sweep_tool_sha256=TOOL_SHA,
        )

    tampered = copy.deepcopy(payload)
    tampered["runs"][0]["host_power"]["after"]["source"] = "battery"
    with pytest.raises(ValueError, match="power source is not external|changed"):
        importer.validate_summary_structure(
            path,
            tampered,
            case=case,
            source_git_sha=SOURCE_SHA,
            sweep_tool_sha256=TOOL_SHA,
        )


def test_performance_summary_has_functional_gate_without_quality_score(tmp_path):
    case = importer.expected_cases()[
        "causal-language-modeling__max__inference__prefill"
    ]
    payload = _summary(case)
    path = tmp_path / "evidence_summary.json"

    importer.validate_summary_structure(
        path,
        payload,
        case=case,
        source_git_sha=SOURCE_SHA,
        sweep_tool_sha256=TOOL_SHA,
    )
    assert payload["quality_gate"] is None
    assert payload["functional_gate"]["metric"] == "prefill_tokens_per_sec"

    tampered = copy.deepcopy(payload)
    tampered["quality_metric"] = "invented_quality"
    with pytest.raises(ValueError, match="exposes score-quality fields"):
        importer.validate_summary_structure(
            path,
            tampered,
            case=case,
            source_git_sha=SOURCE_SHA,
            sweep_tool_sha256=TOOL_SHA,
        )


def test_build_index_preserves_case_identity_and_prefixed_digest(tmp_path):
    case = importer.expected_cases()["image-classification__max__inference"]
    payload = _summary(case)
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    source_lock = {
        "schema": "mlperf-edu-reference-source-lock/0.2",
        "file_count": 1,
        "contract_count": 1,
    }
    source_lock_bytes = b"{}\n"

    index = importer.build_index(
        {case.case_id: (tmp_path / "summary.json", payload, data)},
        source_git_sha=SOURCE_SHA,
        source_lock=source_lock,
        source_lock_bytes=source_lock_bytes,
    )

    assert index["schema"] == importer.INDEX_SCHEMA
    assert index["case_count"] == 1
    assert index["workload_count"] == 1
    entry = index["cases"][0]
    assert entry["case_id"] == case.case_id
    assert entry["mode"] == "inference"
    assert entry["phase"] is None
    assert entry["evidence_sha256"] == importer.sha256_bytes(data)


def test_causal_phase_lineage_closes_over_one_committed_training_run(tmp_path):
    cases = importer.expected_cases()
    training_case = cases["causal-language-modeling__max__training"]
    training = _summary(training_case)
    for position, run in enumerate(training["runs"], start=1):
        run["artifacts"] = [
            {
                "role": role,
                "sha256": f"sha256:{position:02x}{digit * 62}",
            }
            for role, digit in (
                ("checkpoint", "a"),
                ("report", "b"),
                ("provenance", "c"),
            )
        ]
    selected_run = training["runs"][2]
    source = {
        role: importer._artifact_digest(selected_run, role)
        for role in ("checkpoint", "report", "provenance")
    }
    training_bytes = (json.dumps(training, sort_keys=True) + "\n").encode()
    selected = {
        training_case.case_id: (
            tmp_path / "training" / "evidence_summary.json",
            training,
            training_bytes,
        )
    }
    package_digest = "sha256:" + "d" * 64
    for phase in ("full", "prefill", "decode"):
        case = cases[f"causal-language-modeling__max__inference__{phase}"]
        payload = _summary(case)
        payload["nanogpt_training_lineage"] = {
            "required": True,
            "status": "staged",
            "package_sha256": package_digest,
        }
        for run in payload["runs"]:
            run["artifacts"] = [
                {"role": "checkpoint", "sha256": source["checkpoint"]},
                {"role": "source_training_report", "sha256": source["report"]},
                {
                    "role": "source_training_provenance",
                    "sha256": source["provenance"],
                },
            ]
        data = (json.dumps(payload, sort_keys=True) + "\n").encode()
        selected[case.case_id] = (
            tmp_path / phase / "evidence_summary.json",
            payload,
            data,
        )

    bindings = importer.validate_lineage_closure(selected)

    assert set(bindings) == {
        f"causal-language-modeling__max__inference__{phase}"
        for phase in ("full", "prefill", "decode")
    }
    for binding in bindings.values():
        assert binding["source_training_execution_index"] == 3
        assert binding["source_training_checkpoint_sha256"] == source["checkpoint"]
        assert binding["source_training_package_sha256"] == package_digest


def test_json_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"valid","status":"invalid"}\n')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        importer.load_json_object(path, label="test")


def test_resolve_indexed_file_rejects_path_traversal(tmp_path):
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}\n")

    with pytest.raises(ValueError, match="unsafe"):
        importer.resolve_indexed_file(tmp_path, "../outside.json", label="artifact")
