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
    assert plan["runs"][0]["repetitions"] == 1
    assert plan["runs"][0]["environment"] == {
        "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE": "16",
        "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_REPETITIONS": "10",
    }
    assert plan["study"]["independent_variables"] == ["batch size"]
    assert plan["source_sha256"].startswith("sha256:")
    assert len(plan["source_sha256"]) == 71


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
output:
  directory: ignored-by-cli
  open_report: false
runs:
  - name: batch-16
    workload: image-classification
    mode: inference
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 16
  - name: batch-64
    workload: image-classification
    mode: inference
    environment:
      MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE: 64
""".lstrip(),
    )
    output_dir = tmp_path / "results"
    calls = []

    def fake_run(workload, profile, current_output, *, mode=None, phase=None):
        calls.append(
            {
                "workload": workload.id,
                "profile": profile,
                "output": current_output,
                "mode": mode,
                "phase": phase,
                "batch_size": os.environ[
                    "MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"
                ],
                "repetitions": os.environ["MLPERF_EDU_PRO_REPETITIONS"],
            }
        )
        return {
            "schema": "mlperf-edu-report/0.1",
            "id": workload.id,
            "workload": workload.id,
            "suite": workload.suite,
            "profile": profile,
            "status": "passed",
            "quality": {
                "metric": "accuracy",
                "value": 0.9,
                "target": 0.8,
                "quality_required": True,
                "target_met": True,
            },
        }

    monkeypatch.setattr(edu_cli, "run_workload", fake_run)
    monkeypatch.setenv("MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE", "original")
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
    assert calls[0]["output"].name == "01-batch-16"
    assert calls[1]["output"].name == "02-batch-64"
    assert os.environ["MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE"] == "original"
    assert "MLPERF_EDU_PRO_REPETITIONS" not in os.environ

    aggregate_path = next(output_dir.glob("mlperf_edu_pro_*.json"))
    aggregate = json.loads(aggregate_path.read_text())
    assert aggregate["selection"] == {"kind": "plan", "name": "execution-test"}
    assert aggregate["experiment_plan"]["source_sha256"].startswith("sha256:")
    assert aggregate["workloads"][0]["experiment_run"]["name"] == "batch-16"
    assert aggregate["workloads"][1]["experiment_run"]["name"] == "batch-64"
    html = aggregate_path.with_suffix(".html").read_text()
    assert "MLPerf EDU Research Report: execution-test" in html
    assert "Experiment Design" in html
    assert "Plan Provenance" in html
    assert "batch-16" in html
    csv_text = aggregate_path.with_suffix(".csv").read_text()
    assert "experiment_run_name" in csv_text
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
