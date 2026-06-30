import json

from mlperf.manifest import PORTABLE_SIGNATURE_ALGO, build_provd, verify_provd


def test_build_provd_outside_git_repo_is_quiet(tmp_path, capfd):
    report_path = tmp_path / "report.json"
    report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "toy-workload",
        "status": "passed",
        "metrics": {"duration_seconds": 0.01},
    }
    report_path.write_text(json.dumps(report) + "\n")

    manifest = build_provd(
        workload="toy-workload",
        scenario="offline",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
    )

    captured = capfd.readouterr()
    assert captured.err == ""
    assert manifest.leaves["source_tree"]["git_sha"] is None
    assert manifest.leaves["source_tree"]["note"] == "no git repo or git unavailable"


def test_manifest_signature_is_portable_without_local_key(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    report_path = tmp_path / "report.json"
    report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "toy-workload",
        "status": "passed",
        "metrics": {"duration_seconds": 0.01},
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    manifest = build_provd(
        workload="toy-workload",
        scenario="offline",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
    )
    manifest_path = tmp_path / "toy.provd.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")

    assert manifest.signature["algo"] == PORTABLE_SIGNATURE_ALGO
    assert not (home / ".mlperf-edu" / "signing.key").exists()
    result = verify_provd(manifest_path, repo_root=tmp_path)
    assert result.all_ok, result.checks
