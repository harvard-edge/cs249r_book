from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlperf.registry import load_registry
from tools import import_reference_evidence


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "reference_results" / "index.json"
SOURCE_SHA = "0ec4d3e1c415944227d0754d170edb0addc1d925"


def test_reference_index_binds_all_exact_imported_summary_bytes():
    index = json.loads(INDEX_PATH.read_text())

    assert index["schema"] == "mlperf-edu-reference-index/0.2"
    assert index["source_git_sha"] == SOURCE_SHA
    assert index["summary_count"] == 8
    assert len(index["summaries"]) == 8
    slm_entry = next(
        entry for entry in index["summaries"] if entry["workload"] == "slm-decode"
    )
    assert slm_entry["reference_metric_role"] == "performance"
    assert slm_entry["functional_gate"]["metric"] == "generated_tokens"
    assert "not a speed threshold" in slm_entry["legacy_summary_semantics"][
        "quality_target"
    ]

    selected = {}
    for entry in index["summaries"]:
        path = ROOT / entry["path"]
        packaged_path = ROOT / "src" / "mlperf_edu" / entry["path"]
        data = path.read_bytes()
        payload = json.loads(data)
        assert packaged_path.read_bytes() == data
        assert hashlib.sha256(data).hexdigest() == entry["evidence_sha256"]
        assert payload["evidence_id"] == entry["evidence_id"]
        assert payload["source"]["git_sha"] == SOURCE_SHA
        assert payload["source"]["git_dirty"] is False
        selected[entry["workload"]] = (path, payload, data)

    source_lock_path = ROOT / index["source_lock"]["path"]
    source_lock_bytes = source_lock_path.read_bytes()
    source_lock = json.loads(source_lock_bytes)
    assert hashlib.sha256(source_lock_bytes).hexdigest() == index["source_lock"][
        "sha256"
    ].removeprefix("sha256:")
    rebuilt = import_reference_evidence.build_index(
        selected,
        source_git_sha=SOURCE_SHA,
        source_lock=source_lock,
        source_lock_bytes=source_lock_bytes,
    )
    assert rebuilt == index
    assert (
        ROOT / "src" / "mlperf_edu" / "reference_results" / "index.json"
    ).read_bytes() == INDEX_PATH.read_bytes()


def test_importer_rejects_a_dirty_or_wrong_source_summary(tmp_path):
    path = tmp_path / "evidence_summary.json"
    payload = {
        "schema": "mlperf-edu-reference-evidence/0.3",
        "status": "valid",
        "evidence_tier": "public-candidate",
        "eligible_for_public_baseline": True,
        "public_status": "score-bearing",
        "workload": "example",
        "evidence_id": "example_max_attempt",
        "acceptance": {"passed": True},
        "invalid_reasons": [],
        "source": {"git_dirty": True, "git_sha": "0" * 40},
        "runs": [{"evidence_valid": True}],
    }

    with pytest.raises(ValueError, match="git_dirty.*source.git_sha"):
        import_reference_evidence.validate_summary(
            path,
            payload,
            workload_contract=load_registry()["nanogpt-train"],
            source_git_sha=SOURCE_SHA,
            sweep_tool_sha256=import_reference_evidence.source_sweep_tool_sha256(
                SOURCE_SHA
            ),
        )


def test_importer_rejects_source_destination_overlap():
    with pytest.raises(ValueError, match="may not contain one another"):
        import_reference_evidence.require_separate_source_and_destinations(
            ROOT / "reference_results"
        )

    with pytest.raises(ValueError, match="may not contain one another"):
        import_reference_evidence.require_separate_source_and_destinations(ROOT)


@pytest.mark.parametrize(
    "unsafe_path",
    ("../outside.json", "..\\outside.json", "bad\nname.json", "/absolute.json"),
)
def test_importer_rejects_nonportable_indexed_paths(tmp_path, unsafe_path):
    with pytest.raises(ValueError, match="path is missing, absolute, or escapes"):
        import_reference_evidence.resolve_indexed_file(
            tmp_path, unsafe_path, label="test artifact"
        )


def test_importer_rejects_symlinked_or_out_of_root_summaries(tmp_path):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    outside = tmp_path / "outside.json"
    data = b"{}\n"
    outside.write_bytes(data)
    linked = evidence_root / "evidence_summary.json"
    linked.symlink_to(outside)

    with pytest.raises(ValueError, match="summary may not be a symlink"):
        import_reference_evidence.verify_external_artifacts(
            evidence_root, linked, {}, data, cache={}
        )

    with pytest.raises(ValueError, match="escapes the evidence root"):
        import_reference_evidence.verify_external_artifacts(
            evidence_root, outside, {}, data, cache={}
        )


def test_importer_hashes_sidecar_and_every_retained_artifact(tmp_path):
    attempt = tmp_path / "attempt"
    artifact = attempt / "seed_0" / "report.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"report bytes\n")
    summary = attempt / "evidence_summary.json"
    data = b'{"summary": true}\n'
    summary.write_bytes(data)
    summary.with_suffix(".json.sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  evidence_summary.json\n"
    )
    payload = {
        "runs": [
            {
                "artifacts": [
                    {
                        "path": "seed_0/report.json",
                        "role": "report",
                        "n_bytes": artifact.stat().st_size,
                        "sha256": "sha256:"
                        + hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ]
            }
        ]
    }

    import_reference_evidence.verify_external_artifacts(
        tmp_path, summary, payload, data, cache={}
    )

    artifact.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="n_bytes|sha256"):
        import_reference_evidence.verify_external_artifacts(
            tmp_path, summary, payload, data, cache={}
        )


def test_importer_binds_summary_metric_to_raw_report(tmp_path, monkeypatch):
    report_path = tmp_path / "seed_0" / "example_max_report.json"
    manifest_path = tmp_path / "seed_0" / "example_max.provd.json"
    report_path.parent.mkdir()
    run = {
        "requested_seed": 0,
        "status": "passed",
        "backend": "pytorch-cpu",
        "chip": "Test Chip",
        "data_mode": "real",
        "quality_value": 0.75,
        "quality_metric_key": "accuracy",
        "quality_metric_declared": "accuracy",
        "functional_metric_declared": "accuracy",
        "quality_target_met": True,
        "report_recorded_seed": 0,
        "manifest_recorded_seed": 0,
        "fingerprint_backends": ["pytorch-cpu"],
        "hardware_backend": "CPU",
        "report_path": "seed_0/example_max_report.json",
        "manifest_path": "seed_0/example_max.provd.json",
        "grade": {
            "status": "passed",
            "passed": True,
            "target_met": True,
            "metric": "accuracy",
            "value": 0.75,
            "target": 0.7,
        },
        "artifacts": [
            {"role": "report", "path": "seed_0/example_max_report.json"},
            {"role": "provenance", "path": "seed_0/example_max.provd.json"},
        ],
    }
    report = {
        "workload": "example",
        "profile": "max",
        "status": "passed",
        "seed": 0,
        "backend": "pytorch-cpu",
        "data_mode": "real",
        "variant": None,
        "metrics": {"accuracy": 0.75},
        "quality": {"metric": "accuracy", "target": 0.7, "target_met": True},
        "review_contract": {
            "status": "passed",
            "review_eligible": True,
            "issues": [],
            "public_status": "score-bearing",
            "profile": "max",
            "data_mode": "real",
            "metric": "accuracy",
            "metric_key": "accuracy",
            "metric_value": 0.75,
            "functional_metric": "accuracy",
            "functional_metric_value": 0.75,
        },
        "run_fingerprint": {
            "execution": {
                "workload": "example",
                "profile": "max",
                "seed": 0,
                "status": "passed",
                "backends": ["pytorch-cpu"],
                "data_modes": ["real"],
            },
            "hardware": {"chip": "Test Chip", "backend": "CPU"},
        },
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    manifest = {
        "workload": "example",
        "leaves": {
            "rng": {"seed": 0},
            "measurement": {
                "report_path": str(report_path.resolve()),
                "report_file_sha256": import_reference_evidence.sha256_file(
                    report_path
                ),
                "n_bytes": report_path.stat().st_size,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    payload = {
        "workload": "example",
        "profile": "max",
        "variant": None,
        "public_status": "score-bearing",
        "quality_metric": "accuracy",
    }
    monkeypatch.setattr(
        import_reference_evidence,
        "verify_manifest_against_source",
        lambda *args, **kwargs: None,
    )
    contract = SimpleNamespace(raw={})

    import_reference_evidence.verify_run_semantics(
        tmp_path,
        payload,
        run,
        run_index=0,
        workload_contract=contract,
        source_project_root=tmp_path,
        manifest_cache={},
    )

    run["quality_value"] = 0.8
    with pytest.raises(ValueError, match="metric value"):
        import_reference_evidence.verify_run_semantics(
            tmp_path,
            payload,
            run,
            run_index=0,
            workload_contract=contract,
            source_project_root=tmp_path,
            manifest_cache={},
        )


def test_source_tool_digest_is_bound_to_exact_git_object(monkeypatch):
    source_bytes = b"exact historical sweep tool bytes\n"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout=str(ROOT.parent) + "\n")
        return SimpleNamespace(stdout=source_bytes)

    monkeypatch.setattr(import_reference_evidence.subprocess, "run", fake_run)

    assert import_reference_evidence.source_sweep_tool_sha256(SOURCE_SHA) == (
        "sha256:" + hashlib.sha256(source_bytes).hexdigest()
    )
    assert calls[1][2].startswith(f"{SOURCE_SHA}:")


def test_committed_summaries_bind_the_current_sweep_tool_bytes():
    assert import_reference_evidence.check_taxonomy.SWEEP_TOOL_SHA256 == (
        "sha256:b5409e20385add7cd7b7f6814bf902affa71fe4b43532477d2f82b1d05fd2bfd"
    )
