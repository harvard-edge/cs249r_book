import hashlib
import json
import subprocess

from mlperf.manifest import (
    INTEGRITY_DIGEST_ALGO,
    LEGACY_PORTABLE_SIGNATURE_ALGO,
    LEGACY_PORTABLE_SIGNATURE_DOMAIN,
    _git_leaf,
    build_provd,
    rng_leaf,
    verify_provd,
)


def test_rng_leaf_distinguishes_initial_and_manifest_capture_states():
    first = rng_leaf(7, b"state-after-work", None)
    second = rng_leaf(7, b"different-state-after-work", None)

    assert first["torch_initial_state_sha256"] == second["torch_initial_state_sha256"]
    assert first["torch_initial_state_derivation"].startswith("torch.Generator")
    assert first["torch_captured_state_sha256"] != second["torch_captured_state_sha256"]
    assert first["torch_captured_state_point"] == "manifest-construction"
    assert "numpy_state_sha256" not in first


def test_git_leaf_binds_staged_unstaged_and_untracked_content(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "benchmark@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Benchmark Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("committed\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    assert _git_leaf(tmp_path)["git_dirty"] is False

    untracked = tmp_path / "untracked.txt"
    untracked.write_text("first\n")
    first = _git_leaf(tmp_path)
    assert first["git_dirty"] is True
    assert first["patch_hash"].startswith("sha256:")

    untracked.write_text("second\n")
    second = _git_leaf(tmp_path)
    assert second["patch_hash"] != first["patch_hash"]

    subprocess.run(["git", "add", "untracked.txt"], cwd=tmp_path, check=True)
    staged = _git_leaf(tmp_path)
    assert staged["git_dirty"] is True
    assert staged["patch_hash"] != second["patch_hash"]

    tracked.write_text("unstaged\n")
    unstaged = _git_leaf(tmp_path)
    assert unstaged["patch_hash"] != staged["patch_hash"]


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


def test_manifest_recomputes_seed_derived_initial_rng_state(tmp_path):
    from mlperf.manifest import integrity_record, merkle_root

    report_path = tmp_path / "report.json"
    report = {"workload": "toy", "status": "passed", "metrics": {"loss": 1.0}}
    report_path.write_text(json.dumps(report) + "\n")
    manifest = build_provd(
        workload="toy",
        scenario="train",
        division="open",
        hardware_fingerprint={"platform": "test"},
        report=report,
        report_path=report_path,
        rng_seed=7,
        torch_state_bytes=b"captured-after-work",
        repo_root=tmp_path,
    ).to_dict()
    manifest["leaves"]["rng"]["torch_initial_state_sha256"] = "sha256:" + "0" * 64
    manifest["merkle_root"] = merkle_root(manifest["leaves"])
    manifest["integrity"] = integrity_record(manifest["merkle_root"])
    manifest_path = tmp_path / "toy.provd.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = verify_provd(manifest_path, repo_root=tmp_path)

    assert not result.all_ok
    assert any(
        name == "rng.torch_initial_state_sha256" and not ok
        for name, ok, _ in result.checks
    )


def test_dataset_merkle_root_is_independent_of_storage_path(tmp_path):
    from mlperf.manifest import dataset_leaf

    first_dir = tmp_path / "seed_0"
    second_dir = tmp_path / "seed_1"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "quality.json"
    second = second_dir / "quality.json"
    first.write_text('{"cases": []}\n')
    second.write_text(first.read_text())

    first_leaf = dataset_leaf("prompt-suite-local", [first])
    second_leaf = dataset_leaf("prompt-suite-local", [second])

    assert first_leaf["files"][0]["path"] != second_leaf["files"][0]["path"]
    assert first_leaf["files"][0]["logical_path"] == "quality.json"
    assert second_leaf["files"][0]["logical_path"] == "quality.json"
    assert first_leaf["merkle_root"] == second_leaf["merkle_root"]


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
