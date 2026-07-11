import hashlib
import json

from mlperf.manifest import (
    INTEGRITY_DIGEST_ALGO,
    LEGACY_PORTABLE_SIGNATURE_ALGO,
    LEGACY_PORTABLE_SIGNATURE_DOMAIN,
    build_provd,
    verify_provd,
)


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


def test_manifest_integrity_digest_is_portable_and_not_mislabeled_as_signature(
    tmp_path, monkeypatch
):
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
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )

    serialized = manifest.to_dict()
    assert serialized["integrity"]["algorithm"] == INTEGRITY_DIGEST_ALGO
    assert serialized["integrity"]["authenticated"] is False
    assert serialized["integrity"]["type"] == "unauthenticated_digest"
    assert "signature" not in serialized
    assert serialized["leaves"]["hardware"]["fingerprint"] == {"platform": "test"}
    assert not (home / ".mlperf-edu" / "signing.key").exists()
    result = verify_provd(manifest_path, repo_root=tmp_path)
    assert result.all_ok, result.checks
    assert any(
        name == "hardware.fingerprint_sha256" and ok for name, ok, _ in result.checks
    )


def test_manifest_verifies_relative_artifacts_after_relocation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    report_path = source / "report.json"
    weights_path = source / "weights.bin"
    dataset_path = source / "dataset.bin"
    roofline_path = source / "roofline.json"
    weights_path.write_bytes(b"weights")
    dataset_path.write_bytes(b"dataset")
    roofline_path.write_text("{}\n")
    report = {
        "schema": "mlperf-edu-report/0.1",
        "workload": "toy-workload",
        "status": "passed",
        "metrics": {"accuracy": 1.0},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = build_provd(
        workload="toy-workload",
        scenario="offline",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        weights_path=weights_path,
        dataset_name="toy-data",
        dataset_files=[dataset_path],
        roofline_sidecar_path=roofline_path,
        repo_root=tmp_path,
    ).to_dict()

    relocated = tmp_path / "relocated"
    manifest_dir = relocated / "manifest"
    manifest_dir.mkdir(parents=True)
    (relocated / "report").mkdir()
    (relocated / "weights").mkdir()
    (relocated / "dataset").mkdir()
    (relocated / "roofline").mkdir()
    (relocated / "report" / report_path.name).write_bytes(report_path.read_bytes())
    (relocated / "weights" / weights_path.name).write_bytes(weights_path.read_bytes())
    (relocated / "dataset" / dataset_path.name).write_bytes(dataset_path.read_bytes())
    (relocated / "roofline" / roofline_path.name).write_bytes(
        roofline_path.read_bytes()
    )

    leaves = manifest["leaves"]
    leaves["measurement"]["report_path"] = f"../report/{report_path.name}"
    leaves["weights"]["path"] = f"../weights/{weights_path.name}"
    leaves["dataset"]["files"][0]["path"] = f"../dataset/{dataset_path.name}"
    leaves["roofline_sidecar"]["path"] = f"../roofline/{roofline_path.name}"

    # Rebuilding a portable manifest is package functionality; this test focuses
    # on relative path resolution, so preserve the original integrity fields by
    # using the package helper exercised in the CLI test below.
    from mlperf.manifest import integrity_record, merkle_root

    dataset_digest = leaves["dataset"]["files"][0]["sha256"].removeprefix("sha256:")
    root = hashlib.sha256()
    root.update(f"{leaves['dataset']['files'][0]['path']}:{dataset_digest}\n".encode())
    leaves["dataset"]["merkle_root"] = "sha256:" + root.hexdigest()
    manifest["merkle_root"] = merkle_root(leaves)
    manifest["integrity"] = integrity_record(manifest["merkle_root"])
    manifest_path = manifest_dir / "toy.provd.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = verify_provd(manifest_path, repo_root=tmp_path)
    assert result.all_ok, result.checks

    (relocated / "weights" / weights_path.name).write_bytes(b"tampered")
    tampered = verify_provd(manifest_path, repo_root=tmp_path)
    assert not tampered.all_ok
    assert any(name == "weights.sha256" and not ok for name, ok, _ in tampered.checks)


def test_legacy_mislabeled_public_digest_remains_verifiable(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"workload": "legacy", "status": "passed", "metrics": {"loss": 1.0}}
    report_path.write_text(json.dumps(report) + "\n")
    manifest = build_provd(
        workload="legacy",
        scenario="offline",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
    ).to_dict()
    manifest["schema"] = "mlperf-edu-provd/1.0"
    manifest.pop("integrity")
    legacy_digest = hashlib.sha256(
        f"{LEGACY_PORTABLE_SIGNATURE_DOMAIN}:{manifest['merkle_root']}".encode()
    ).hexdigest()
    manifest["signature"] = {
        "algo": LEGACY_PORTABLE_SIGNATURE_ALGO,
        "key_id": "mlperf-edu-public-v1",
        "signature": legacy_digest,
    }
    manifest_path = tmp_path / "legacy.provd.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = verify_provd(manifest_path, repo_root=tmp_path)
    assert result.all_ok, result.checks
    assert any(name == "legacy_signature" and ok for name, ok, _ in result.checks)
