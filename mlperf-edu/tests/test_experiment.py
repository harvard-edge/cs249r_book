import json
import os
from pathlib import Path

import pytest

from mlperf import edu_cli
from mlperf.experiment import EXPERIMENT_PLAN_SCHEMA, load_experiment_plan


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
def test_load_experiment_plan_rejects_unsafe_environment(
    tmp_path, fragment, message
):
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
                "quality_override": os.environ.get(
                    "MLPERF_EDU_MAX_QUALITY_TARGET"
                ),
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
    assert aggregate["workloads"][0]["experiment_run"]["name"] == "batch-16"
    assert aggregate["workloads"][0]["experiment_run"]["role"] == "baseline"
    assert aggregate["workloads"][1]["experiment_run"]["name"] == "batch-64"
    assert "MLPERF_EDU_MAX_QUALITY_TARGET" not in (
        aggregate["run_fingerprint"]["software"]["performance_environment"]
    )
    aggregate_manifest = aggregate_path.with_name(
        aggregate_path.stem + ".provd.json"
    )
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
            "environment": {
                "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "16"
            },
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
            "environment": {
                "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "32"
            },
            "repetitions": 1,
        },
    ]

    html = edu_cli.experiment_performance_html(
        records,
        independent_variables={
            "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"
        },
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
