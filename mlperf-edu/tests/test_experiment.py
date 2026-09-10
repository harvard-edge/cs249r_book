import json
import os
from pathlib import Path

import pytest

from mlperf import edu_cli
from mlperf.experiment import (
    EXPERIMENT_PLAN_SCHEMA,
    bind_instructor_reference,
    load_experiment_plan,
)


def write_plan(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(body)
    return path


def test_load_experiment_plan_normalizes_defaults_and_hashes_source(tmp_path):
    path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: batch-study
profile: pro
study:
  question: Does batch size change throughput?
  independent_variables: [batch size]
  controls: [model, dataset]
defaults:
  device: cpu
  repetitions: 1
  environment:
    MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_REPETITIONS: 10
output:
  open_report: false
runs:
  - name: batch-16
    workload: image-classification
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 16
""".lstrip(),
    )

    plan = load_experiment_plan(path)

    assert plan["id"] == "batch-study"
    assert plan["runs"][0]["device"] == "cpu"
    assert plan["runs"][0]["role"] == "condition"
    assert plan["runs"][0]["repetitions"] == 1
    assert plan["runs"][0]["environment"] == {
        "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "16",
        "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_REPETITIONS": "10",
    }
    assert plan["study"]["independent_variables"] == ["batch size"]
    assert plan["source_sha256"].startswith("sha256:")
    assert len(plan["source_sha256"]) == 71


def test_experiment_plan_does_not_open_reports_by_default(tmp_path):
    path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: closed-report
runs:
  - workload: image-classification
""".lstrip(),
    )

    plan = load_experiment_plan(path)

    assert plan["output"]["open_report"] is False


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("MLPERF_EDU_DEVICE: cpu", "dedicated plan field"),
        ("MLPERF_EDU_MAX_QUALITY_TARGET: 9.9", "cannot override a quality target"),
        ("MLPERF_EDU_API_TOKEN: unsafe", "sensitive data"),
        ("NOT_ALLOWED: value", "not an MLPERF_EDU"),
    ],
)
def test_load_experiment_plan_rejects_unsafe_environment(tmp_path, fragment, message):
    path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: unsafe-plan
runs:
  - workload: image-classification
    environment:
      {fragment}
""".lstrip(),
    )

    with pytest.raises(ValueError, match=message):
        load_experiment_plan(path)


def test_load_experiment_plan_rejects_duplicate_and_unsafe_run_names(tmp_path):
    duplicate = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: duplicate-plan
runs:
  - name: same
    role: baseline
    workload: image-classification
  - name: same
    workload: keyword-spotting
""".lstrip(),
    )
    with pytest.raises(ValueError, match="duplicate experiment run name"):
        load_experiment_plan(duplicate)

    unsafe = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: unsafe-name
runs:
  - name: ../escape
    workload: image-classification
""".lstrip(),
    )
    with pytest.raises(ValueError, match="lowercase letters"):
        load_experiment_plan(unsafe)

    multiple_baselines = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: multiple-baselines
runs:
  - name: first
    role: baseline
    workload: image-classification
  - name: second
    role: baseline
    workload: image-classification
""".lstrip(),
    )
    with pytest.raises(ValueError, match="at most one baseline"):
        load_experiment_plan(multiple_baselines)

    candidate_without_baseline = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: missing-baseline
runs:
  - name: candidate
    role: candidate
    workload: image-classification
""".lstrip(),
    )
    with pytest.raises(ValueError, match="require exactly one baseline"):
        load_experiment_plan(candidate_without_baseline)

    multi_workload_baselines = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: multi-workload-baselines
runs:
  - name: image-baseline
    role: baseline
    workload: image-classification
    mode: inference
  - name: image-candidate
    role: candidate
    workload: image-classification
    mode: inference
  - name: speech-baseline
    role: baseline
    workload: keyword-spotting
    mode: inference
  - name: speech-candidate
    role: candidate
    workload: keyword-spotting
    mode: inference
""".lstrip(),
    )
    plan = load_experiment_plan(multi_workload_baselines)
    assert [run["role"] for run in plan["runs"]] == [
        "baseline",
        "candidate",
        "baseline",
        "candidate",
    ]


def test_load_experiment_plan_rejects_unknown_fields(tmp_path):
    path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: typo-plan
repetitons: 2
runs:
  - workload: image-classification
""".lstrip(),
    )

    with pytest.raises(ValueError, match="unknown field.*repetitons"):
        load_experiment_plan(path)


def test_instructor_reference_accepts_only_declared_candidate_edits(tmp_path):
    reference_path = tmp_path / "reference.yaml"
    reference_path.write_text(
        f"""\
schema: {EXPERIMENT_PLAN_SCHEMA}
id: bound-plan
study:
  independent_variables:
    - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
edit_policy:
  allowed_candidate_environment:
    candidate:
      - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
runs:
  - name: baseline
    role: baseline
    workload: image-classification
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 16
  - name: candidate
    role: candidate
    workload: image-classification
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64
"""
    )
    student_path = tmp_path / "student.yaml"
    student_path.write_text(
        reference_path.read_text().replace(
            "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64",
            "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 128",
        )
    )

    reference = load_experiment_plan(reference_path)
    student = load_experiment_plan(student_path)
    binding = bind_instructor_reference(
        student, reference, reference_source=reference_path.name
    )

    assert binding["status"] == "passed"
    assert binding["reference_source_sha256"] == reference["source_sha256"]
    assert binding["submitted_source_sha256"] == student["source_sha256"]
    assert binding["accepted_changes"] == [
        {
            "run": "candidate",
            "setting": "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE",
            "reference_value": "64",
            "submitted_value": "128",
        }
    ]

    # Add an unauthorized explicit control while retaining the allowed edit.
    student_path.write_text(
        student_path.read_text().replace("runs:\n", "defaults:\n  device: cpu\nruns:\n")
    )
    unauthorized = load_experiment_plan(student_path)
    with pytest.raises(ValueError, match="outside allowed candidate settings"):
        bind_instructor_reference(
            unauthorized, reference, reference_source=reference_path.name
        )


def test_edit_policy_rejects_baseline_and_undeclared_settings(tmp_path):
    baseline_policy = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: unsafe-policy
study:
  independent_variables: [MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE]
edit_policy:
  allowed_candidate_environment:
    baseline:
      - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
runs:
  - name: baseline
    role: baseline
    workload: image-classification
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 16
""".lstrip(),
    )
    with pytest.raises(ValueError, match="only for candidate runs"):
        load_experiment_plan(baseline_policy)

    undeclared_policy = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: undeclared-policy
study:
  independent_variables: []
edit_policy:
  allowed_candidate_environment:
    candidate:
      - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
runs:
  - name: baseline
    role: baseline
    workload: image-classification
  - name: candidate
    role: candidate
    workload: image-classification
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64
""".lstrip(),
    )
    with pytest.raises(ValueError, match="not a declared independent variable"):
        load_experiment_plan(undeclared_policy)


def test_baseline_import_is_fail_closed_and_baseline_only(tmp_path):
    accepted = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: imported-baseline
runs:
  - name: baseline
    role: baseline
    workload: image-classification
    repetitions: 1
    baseline_import:
      manifest: baseline/source.provd.json
      sha256: sha256:{'0' * 64}
  - name: candidate
    role: candidate
    workload: image-classification
""".lstrip(),
    )
    plan = load_experiment_plan(accepted)
    assert plan["runs"][0]["baseline_import"]["manifest"] == (
        "baseline/source.provd.json"
    )

    candidate_import = write_plan(
        tmp_path,
        accepted.read_text().replace(
            "role: baseline\n", "role: candidate\n", 1
        ),
    )
    with pytest.raises(ValueError, match="requires role: baseline"):
        load_experiment_plan(candidate_import)

    unsafe_path = write_plan(
        tmp_path,
        accepted.read_text().replace(
            "baseline/source.provd.json", "../source.provd.json"
        ),
    )
    with pytest.raises(ValueError, match="within the plan directory"):
        load_experiment_plan(unsafe_path)


def test_import_plan_baseline_verifies_and_wraps_source_evidence(tmp_path):
    workloads = edu_cli.load_workloads(
        edu_cli.build_parser().parse_args(["list"])
    )
    workload = workloads["image-classification"]
    source_dir = tmp_path / "baseline"
    source_dir.mkdir()
    source_report_path = source_dir / "source_report.json"
    source_manifest_path = source_dir / "source.provd.json"
    source_report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "pro",
        "mode": "inference",
        "phase": None,
        "scenario": "offline",
        "status": "passed",
        "backend": "pytorch-cpu",
        "device_executed": "cpu",
        "config": {"batch_size": 16},
        "metrics": {"top1_accuracy": 0.87, "samples_per_second": 100.0},
        "quality": {
            "metric": "top1_accuracy",
            "target": 0.85,
            "direction": "higher",
            "tolerance": 0.0,
            "quality_required": True,
            "target_met": True,
        },
        "experiment_run": {
            "plan_id": "instructor-source",
            "name": "baseline",
            "role": "baseline",
        },
        "artifacts": {
            "report": str(source_report_path),
            "provenance": str(source_manifest_path),
        },
    }
    source_report_path.write_text(
        json.dumps(source_report, indent=2, sort_keys=True) + "\n"
    )
    source_manifest = edu_cli.build_provd(
        workload=workload.id,
        scenario="offline",
        division="open",
        hardware_fingerprint=edu_cli.detect_hardware(),
        report=source_report,
        report_path=source_report_path,
        dataset_name="fixture",
        dataset_files=[],
        repo_root=edu_cli.find_project_root(),
    )
    source_manifest_path.write_text(
        json.dumps(source_manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    source_report_before = source_report_path.read_bytes()
    source_manifest_before = source_manifest_path.read_bytes()
    manifest_digest, _ = edu_cli.sha256_file_for_report(source_manifest_path)
    plan_path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: candidate-only-training
study:
  independent_variables:
    - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
runs:
  - name: baseline
    role: baseline
    workload: image-classification
    mode: inference
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 16
    baseline_import:
      manifest: baseline/source.provd.json
      sha256: sha256:{manifest_digest}
  - name: candidate
    role: candidate
    workload: image-classification
    mode: inference
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64
""".lstrip(),
    )
    run = load_experiment_plan(plan_path)["runs"][0]

    imported = edu_cli.import_plan_baseline(
        run,
        workload,
        mode="inference",
        phase=None,
        plan_path=plan_path,
        output_dir=tmp_path / "imported",
    )

    assert imported["execution_origin"] == "provenance-bound-baseline-import"
    assert imported["baseline_import"]["source_manifest_sha256"] == (
        f"sha256:{manifest_digest}"
    )
    imported["experiment_run"] = {
        "plan_id": "candidate-only-training",
        "index": 1,
        "name": "baseline",
        "role": "baseline",
        "imported": True,
        "environment": run["environment"],
    }
    edu_cli.enrich_report_for_display(imported, workloads)
    edu_cli.export_workload_reports([imported], workloads)
    wrapper_manifest = Path(imported["artifacts"]["provenance"])
    assert edu_cli.verify_provd(
        wrapper_manifest, repo_root=edu_cli.find_project_root()
    ).all_ok
    aggregate = {
        "experiment_plan": load_experiment_plan(plan_path),
        "workloads": [imported],
    }
    assert "baseline · imported" in edu_cli.experiment_plan_section_html(aggregate)
    assert edu_cli.report_rows(imported)[0]["experiment_run_imported"] is True
    assert source_report_path.read_bytes() == source_report_before
    assert source_manifest_path.read_bytes() == source_manifest_before

    bad_run = json.loads(json.dumps(run))
    bad_run["baseline_import"]["sha256"] = f"sha256:{'f' * 64}"
    with pytest.raises(ValueError, match="manifest digest differs"):
        edu_cli.import_plan_baseline(
            bad_run,
            workload,
            mode="inference",
            phase=None,
            plan_path=plan_path,
            output_dir=tmp_path / "rejected",
        )


def test_run_plan_separates_runs_records_plan_and_restores_environment(
    tmp_path, monkeypatch
):
    plan_path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: execution-test
study:
  question: Does batch size change throughput?
  hypothesis: The larger batch will be faster.
  independent_variables:
    - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
  controls:
    - model
    - dataset
output:
  directory: ignored-by-cli
  open_report: false
edit_policy:
  allowed_candidate_environment:
    batch-64:
      - MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE
runs:
  - name: batch-16
    role: baseline
    workload: image-classification
    mode: inference
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 16
  - name: batch-64
    role: candidate
    workload: image-classification
    mode: inference
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64
""".lstrip(),
    )
    reference_path = tmp_path / "reference-plan.yaml"
    reference_path.write_text(
        plan_path.read_text().replace(
            "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64",
            "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 32",
        )
    )
    output_dir = tmp_path / "results"
    calls = []

    def fake_run(workload, profile, current_output, *, mode=None, phase=None):
        batch_size = os.environ["MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"]
        calls.append(
            {
                "workload": workload.id,
                "profile": profile,
                "output": current_output,
                "mode": mode,
                "phase": phase,
                "batch_size": batch_size,
                "repetitions": os.environ["MLPERF_EDU_PRO_REPETITIONS"],
                "quality_override": os.environ.get("MLPERF_EDU_MAX_QUALITY_TARGET"),
            }
        )
        current_output.mkdir(parents=True, exist_ok=True)
        report_path = current_output / "image-classification_pro_report.json"
        manifest_path = current_output / "image-classification_pro.provd.json"
        report = {
            "schema": "mlperf-edu-report/0.1",
            "id": workload.id,
            "workload": workload.id,
            "suite": workload.suite,
            "profile": profile,
            "status": "passed",
            "data_mode": "synthetic-layout-fixture",
            "backend": "pytorch-fixture",
            "device_executed": "mps",
            "config": {"batch_size": int(batch_size)},
            "dataset": {
                "name": "cifar10",
                "revision": "fixture-revision",
                "split": "mlperf-tiny-200-sample-accuracy-set",
            },
            "evaluator": {"name": "top1", "revision": "fixture-revision"},
            "model_source": {
                "repository": "https://example.test/model",
                "revision": "fixture-revision",
                "checkpoint_sha256": "sha256:fixture-checkpoint",
            },
            "metrics": {
                "top1_accuracy": 0.9,
                "samples_per_second": float(batch_size) * 10.0,
            },
            "quality": {
                "metric": "top1_accuracy",
                "metric_key": "top1_accuracy",
                "target": 0.85,
                "quality_required": True,
                "target_met": True,
            },
            "artifacts": {
                "report": str(report_path),
                "provenance": str(manifest_path),
            },
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest = edu_cli.build_provd(
            workload=workload.id,
            scenario="offline",
            division="open",
            hardware_fingerprint={"platform": "fixture"},
            report=report,
            report_path=report_path,
            repo_root=edu_cli.find_project_root(),
        )
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        return report

    monkeypatch.setattr(edu_cli, "run_workload", fake_run)
    monkeypatch.setenv("MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE", "original")
    monkeypatch.setenv("MLPERF_EDU_MAX_QUALITY_TARGET", "9.9")
    args = edu_cli.build_parser().parse_args(
        [
            "run",
            "--plan",
            str(plan_path),
            "--reference-plan",
            str(reference_path),
            "--output-dir",
            str(output_dir),
            "--no-open-report",
        ]
    )
    args.profile_explicit = False

    assert edu_cli.cmd_run(args) == 0

    assert [call["batch_size"] for call in calls] == ["16", "64"]
    assert [call["repetitions"] for call in calls] == ["1", "1"]
    assert [call["quality_override"] for call in calls] == [None, None]
    assert calls[0]["output"].name == "01-batch-16"
    assert calls[1]["output"].name == "02-batch-64"
    assert os.environ["MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"] == "original"
    assert os.environ["MLPERF_EDU_MAX_QUALITY_TARGET"] == "9.9"
    assert "MLPERF_EDU_PRO_REPETITIONS" not in os.environ

    aggregate_path = next(
        path
        for path in output_dir.glob("mlperf_edu_pro_*.json")
        if not path.name.endswith(".provd.json")
    )
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate["selection"] == {"kind": "plan", "name": "execution-test"}
    assert aggregate["experiment_plan"]["source_sha256"].startswith("sha256:")
    binding = aggregate["experiment_plan"]["instructor_binding"]
    assert binding["status"] == "passed"
    assert binding["accepted_changes"][0]["reference_value"] == "32"
    assert binding["accepted_changes"][0]["submitted_value"] == "64"
    assert aggregate["artifacts"]["instructor_reference_plan"] == str(
        reference_path
    )
    assert aggregate["workloads"][0]["experiment_run"]["name"] == "batch-16"
    assert aggregate["workloads"][0]["experiment_run"]["role"] == "baseline"
    assert aggregate["workloads"][1]["experiment_run"]["name"] == "batch-64"
    assert (
        "MLPERF_EDU_MAX_QUALITY_TARGET"
        not in (aggregate["run_fingerprint"]["software"]["performance_environment"])
    )
    aggregate_manifest = aggregate_path.with_name(aggregate_path.stem + ".provd.json")
    assert aggregate_manifest.is_file()
    assert edu_cli.verify_provd(
        aggregate_manifest, repo_root=edu_cli.find_project_root()
    ).all_ok
    html = aggregate_path.with_suffix(".html").read_text()
    assert "MLPerf EDU Research Report: execution-test" in html
    assert "Experiment Design" in html
    assert "Plan Provenance" in html
    assert "batch-16" in html
    assert "image-classification · batch-16 (baseline)" in html
    assert "Samples Per Second" in html
    assert "+300.00% vs baseline" in html
    assert "Target ≥ 85.00%" in html
    assert "2 of 2" in html
    assert "observed one-run delta" in html
    assert "not a public-result candidate" in html
    assert "Illustrative synthetic data" in html
    assert "Top-1 Accuracy" in html
    csv_text = aggregate_path.with_suffix(".csv").read_text()
    assert "experiment_run_name" in csv_text
    assert "experiment_run_role" in csv_text
    assert "batch-16" in csv_text
    assert "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE" in csv_text


def test_run_plan_rejects_ambiguous_cli_selection(tmp_path):
    plan_path = write_plan(
        tmp_path,
        f"""
schema: {EXPERIMENT_PLAN_SCHEMA}
id: conflict-test
runs:
  - workload: image-classification
""".lstrip(),
    )
    args = edu_cli.build_parser().parse_args(
        [
            "run",
            "--plan",
            str(plan_path),
            "--workload",
            "image-classification",
        ]
    )
    args.profile_explicit = False

    with pytest.raises(ValueError, match="replaces workload selection"):
        edu_cli.cmd_run(args)


def test_main_tracks_plan_cli_overrides_with_equals_syntax(monkeypatch):
    observed = {}

    def capture(args):
        observed.update(
            profile_explicit=args.profile_explicit,
            output_dir_explicit=args.output_dir_explicit,
            open_report_explicit=args.open_report_explicit,
        )
        return 0

    monkeypatch.setattr(edu_cli, "cmd_run", capture)

    assert (
        edu_cli.main(
            [
                "run",
                "--plan=plan.yaml",
                "--profile=pro",
                "--output-dir=submissions",
                "--open-report",
            ]
        )
        == 0
    )
    assert observed == {
        "profile_explicit": True,
        "output_dir_explicit": True,
        "open_report_explicit": True,
    }


def test_experiment_performance_blocks_quality_failed_condition():
    records = [
        {
            "name": "baseline",
            "role": "baseline",
            "workload": "image-classification",
            "mode": "inference",
            "phase": "",
            "metric": "samples_per_second",
            "value": 100.0,
            "quality_eligible": True,
            "dataset": {"name": "cifar10"},
            "evaluator": {"name": "top1"},
            "quality_contract": {"metric": "accuracy", "target": 0.85},
            "checkpoint": {"sha256": "same"},
            "backend": "pytorch-cpu",
            "device": "cpu",
            "config": {"batch_size": 16},
            "environment": {"MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "16"},
            "repetitions": 1,
        },
        {
            "name": "candidate",
            "role": "candidate",
            "workload": "image-classification",
            "mode": "inference",
            "phase": "",
            "metric": "samples_per_second",
            "value": 200.0,
            "quality_eligible": False,
            "dataset": {"name": "cifar10"},
            "evaluator": {"name": "top1"},
            "quality_contract": {"metric": "accuracy", "target": 0.85},
            "checkpoint": {"sha256": "same"},
            "backend": "pytorch-cpu",
            "device": "cpu",
            "config": {"batch_size": 32},
            "environment": {"MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "32"},
            "repetitions": 1,
        },
    ]

    html = edu_cli.experiment_performance_html(
        records,
        independent_variables={"MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"},
    )

    assert "comparison blocked" in html
    assert "must pass the same quality gate" in html
    assert "vs baseline" not in html


def test_experiment_next_action_does_not_recommend_a_blocked_delta(monkeypatch):
    monkeypatch.setattr(edu_cli, "_comparison_provenance_verified", lambda _item: True)
    report = {
        "experiment_plan": {
            "id": "blocked-comparison",
            "title": "Blocked comparison",
            "study": {
                "independent_variables": [
                    "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"
                ]
            },
            "runs": [
                {
                    "name": "baseline",
                    "role": "baseline",
                    "workload": "image-classification",
                    "mode": "inference",
                    "device": "cpu",
                    "repetitions": 1,
                    "environment": {
                        "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "16"
                    },
                },
                {
                    "name": "candidate",
                    "role": "candidate",
                    "workload": "image-classification",
                    "mode": "inference",
                    "device": "cpu",
                    "repetitions": 1,
                    "environment": {
                        "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "64"
                    },
                },
            ],
        },
        "workloads": [],
    }
    for index, batch_size in enumerate((16, 64), start=1):
        report["workloads"].append(
            {
                "workload": "image-classification",
                "profile": "pro",
                "mode": "inference",
                "status": "passed",
                "backend": "pytorch-cpu",
                "device_executed": "cpu",
                "dataset": {"name": "cifar10", "revision": "same"},
                "model_source": {"revision": "same"},
                "execution_lineage": {"checkpoint": {"revision": "same"}},
                "config": {"batch_size": batch_size},
                "metrics": {
                    "top1_accuracy": 0.87,
                    "samples_per_second": float(batch_size),
                },
                "quality": {
                    "metric": "top1_accuracy",
                    "metric_key": "top1_accuracy",
                    "target": 0.85,
                    "direction": "higher",
                    "quality_required": True,
                    "target_met": True,
                },
                "experiment_run": {"index": index},
            }
        )

    html = edu_cli.experiment_plan_section_html(report)

    assert "evaluator evidence is missing" in html
    assert "controlled performance comparison is blocked" in html
    assert "Use the observed delta" not in html
