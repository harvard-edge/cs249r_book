import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tools import build_handoff_manifest as handoff


SOURCE_SHA = "a" * 40


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _included_file(role: str, path: str, payload: bytes) -> dict:
    return {
        "role": role,
        "path": path,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "n_bytes": len(payload),
    }


def _write_package(
    path: Path,
    *,
    workload: str,
    profile: str,
    variant: str | None,
    seed: int,
    report: dict,
) -> None:
    manifest_name = "manifest/run.provd.json"
    report_name = "report/run.json"
    manifest = {
        "workload": workload,
        "leaves": {
            "source_tree": {"git_sha": SOURCE_SHA},
            "rng": {"seed": seed},
            "measurement": {"report_path": "../report/run.json"},
        },
    }
    packaged_report = {
        **report,
        "workload": workload,
        "profile": profile,
        "variant": variant,
        "seed": seed,
        "status": "passed",
    }
    manifest_bytes = _json_bytes(manifest)
    report_bytes = _json_bytes(packaged_report)
    included = [
        _included_file("manifest", manifest_name, manifest_bytes),
        _included_file("report", report_name, report_bytes),
    ]
    checks = [{"check": "source_tree.git_sha", "ok": True}]
    index = {
        "schema": handoff.PACKAGE_SCHEMA,
        "workload": workload,
        "manifest": manifest_name,
        "source_manifest": manifest_name,
        "included_files": included,
        "source_verification": {"passed": True, "checks": checks},
        "verification": checks,
        "clean_extraction_verification": {
            "required": True,
            "status": "passed",
        },
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("package_index.json", _json_bytes(index))
        archive.writestr(manifest_name, manifest_bytes)
        archive.writestr(report_name, report_bytes)


@pytest.mark.parametrize(
    "unsafe",
    ["", "/absolute", "../escape", "a/../b", "a\\b", "C:/drive", "a//b"],
)
def test_safe_relative_path_rejects_nonportable_or_escaping_values(unsafe):
    with pytest.raises(handoff.HandoffError):
        handoff.safe_relative_path(unsafe, label="test path")


def test_json_loader_rejects_duplicate_keys():
    with pytest.raises(handoff.HandoffError, match="duplicate JSON key"):
        handoff.load_json_object_bytes(b'{"seed": 1, "seed": 2}', label="test")


def test_indexed_archive_verification_checks_digest_and_complete_coverage(tmp_path):
    payload = b"verified payload"
    package = tmp_path / "package.zip"
    index = {
        "schema": handoff.PACKAGE_SCHEMA,
        "included_files": [_included_file("report", "report.json", payload)],
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("package_index.json", _json_bytes(index))
        archive.writestr("report.json", payload)

    with zipfile.ZipFile(package) as archive:
        loaded, names = handoff._zip_index(archive, label="package")
        records = handoff._verify_indexed_members(
            archive, loaded, names, label="package"
        )
    assert records["report.json"]["sha256"] == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )

    index["included_files"][0]["sha256"] = "sha256:" + "0" * 64
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr("package_index.json", _json_bytes(index))
        archive.writestr("report.json", payload)
    with zipfile.ZipFile(tampered) as archive:
        loaded, names = handoff._zip_index(archive, label="tampered")
        with pytest.raises(handoff.HandoffError, match="digest mismatch"):
            handoff._verify_indexed_members(archive, loaded, names, label="tampered")

    index["included_files"][0]["sha256"] = (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(extra, "w") as archive:
        archive.writestr("package_index.json", _json_bytes(index))
        archive.writestr("report.json", payload)
        archive.writestr("unindexed.txt", b"not covered")
    with zipfile.ZipFile(extra) as archive:
        loaded, names = handoff._zip_index(archive, label="extra")
        with pytest.raises(handoff.HandoffError, match="coverage"):
            handoff._verify_indexed_members(archive, loaded, names, label="extra")


def _write_raw_attempt(
    evidence_root: Path,
    *,
    evidence_id: str,
    report: dict,
    dataset: dict | None = None,
) -> dict:
    attempt_root = evidence_root / evidence_id / "seed_0"
    attempt_root.mkdir(parents=True)
    report_path = attempt_root / "report.json"
    manifest_path = attempt_root / "run.provd.json"
    report_path.write_bytes(_json_bytes(report))
    manifest_path.write_bytes(_json_bytes({"leaves": {"dataset": dataset or {}}}))
    return {
        "requested_seed": 0,
        "status": "passed",
        "evidence_valid": True,
        "report_path": "seed_0/report.json",
        "manifest_path": "seed_0/run.provd.json",
        "artifacts": [],
    }


def test_manifest_counts_packages_and_policy_blocked_attempts_honestly(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "evidence"
    package_root = tmp_path / "packages"
    package_root.mkdir()
    report = {
        "metrics": {"score": 0.9},
        "quality": {"target_met": True},
        "review_contract": {"status": "passed"},
    }
    open_evidence_id = "open-workload_max_attempt"
    blocked_evidence_id = "restricted-workload_max_attempt"
    open_run = _write_raw_attempt(
        evidence_root, evidence_id=open_evidence_id, report=report
    )
    blocked_run = _write_raw_attempt(
        evidence_root,
        evidence_id=blocked_evidence_id,
        report=report,
        dataset={"name": "cifar10", "files": [{"path": "test.parquet"}]},
    )
    package_path = package_root / f"{open_evidence_id}-seed_0.zip"
    _write_package(
        package_path,
        workload="open-workload",
        profile="max",
        variant=None,
        seed=0,
        report=report,
    )
    selected = {
        "open-workload": (
            evidence_root / open_evidence_id / "evidence_summary.json",
            {
                "schema": "test-summary/0.1",
                "evidence_id": open_evidence_id,
                "workload": "open-workload",
                "profile": "max",
                "variant": None,
                "runs": [open_run],
            },
            b"open summary\n",
        ),
        "restricted-workload": (
            evidence_root / blocked_evidence_id / "evidence_summary.json",
            {
                "schema": "test-summary/0.1",
                "evidence_id": blocked_evidence_id,
                "workload": "restricted-workload",
                "profile": "max",
                "variant": None,
                "runs": [blocked_run],
            },
            b"restricted summary\n",
        ),
    }
    summaries = [
        {
            "workload": workload,
            "evidence_id": payload[1]["evidence_id"],
            "path": f"reference_results/{workload}/summary.json",
            "evidence_sha256": "0" * 64,
        }
        for workload, payload in selected.items()
    ]
    reference_set = handoff.VerifiedReferenceSet(
        index={
            "schema": handoff.INDEX_SCHEMA,
            "source_git_sha": SOURCE_SHA,
            "source_lock": {"path": "reference_results/source_lock.json"},
            "summaries": summaries,
        },
        index_bytes=b"index\n",
        source_lock={
            "schema": "test-lock/0.1",
            "file_count": 1,
            "contract_count": 1,
        },
        source_lock_bytes=b"source lock\n",
        selected=selected,
        lineage={"verification": "passed"},
    )
    expected_counts = {
        "reference_summaries": 2,
        "attempts": 2,
        "evidence_valid_attempts": 2,
        "portable_packages": 1,
        "policy_blocked_attempts": 1,
    }
    monkeypatch.setattr(
        handoff,
        "verify_package_archive",
        lambda _package_path, *, repo_root: [
            ("clean_extraction.provenance", repo_root == tmp_path, "test")
        ],
    )

    manifest = handoff.build_manifest(
        reference_set,
        evidence_root,
        package_root,
        promotion_git_sha="b" * 40,
        expected_counts=expected_counts,
        historical_source_root=tmp_path,
    )

    assert manifest["counts"] == expected_counts
    assert manifest["policy_blocked_by_workload"] == {"restricted-workload": 1}
    attempts = {item["workload"]: item for item in manifest["attempts"]}
    assert attempts["open-workload"]["portable_package"]["status"] == "packaged"
    assert attempts["open-workload"]["portable_package"]["sha256"] == (
        handoff.sha256_file(package_path)
    )
    assert (
        attempts["open-workload"]["portable_package"]["clean_extraction_verification"]
        == "passed"
    )
    assert (
        attempts["restricted-workload"]["portable_package"]["status"]
        == "policy-blocked"
    )
    assert (
        "cifar10 has release status"
        in (attempts["restricted-workload"]["portable_package"]["reason"])
    )
    rebuilt = handoff.build_manifest(
        reference_set,
        evidence_root,
        package_root,
        promotion_git_sha="b" * 40,
        expected_counts=expected_counts,
        historical_source_root=tmp_path,
    )
    assert handoff.canonical_bytes(manifest) == handoff.canonical_bytes(rebuilt)


def test_portable_package_rejects_self_asserted_but_invalid_clean_extraction(tmp_path):
    report = {
        "metrics": {"score": 0.9},
        "quality": {"target_met": True},
        "review_contract": {"status": "passed"},
    }
    package_path = tmp_path / "self-asserted.zip"
    _write_package(
        package_path,
        workload="open-workload",
        profile="max",
        variant=None,
        seed=0,
        report=report,
    )

    with pytest.raises(handoff.HandoffError, match="clean-extraction verification"):
        handoff.verify_portable_package(
            package_path,
            historical_source_root=tmp_path,
            source_git_sha=SOURCE_SHA,
            evidence_id="open-workload_max_attempt",
            payload={
                "workload": "open-workload",
                "profile": "max",
                "variant": None,
            },
            run={"requested_seed": 0, "status": "passed"},
            raw_report=report,
        )


def test_write_manifest_is_external_deterministic_and_checkable(tmp_path):
    output = tmp_path / "release" / "handoff_manifest.json"
    payload = {"schema": handoff.HANDOFF_SCHEMA, "counts": {"attempts": 40}}

    handoff.write_manifest(output, payload, check=False)
    expected = handoff.canonical_bytes(payload)
    assert output.read_bytes() == expected
    handoff.write_manifest(output, payload, check=True)

    output.write_text("stale\n")
    with pytest.raises(handoff.HandoffError, match="missing or stale"):
        handoff.write_manifest(output, payload, check=True)

    inside_checkout = handoff.ROOT / "handoff_manifest.json"
    with pytest.raises(handoff.HandoffError, match="outside the project checkout"):
        handoff.write_manifest(inside_checkout, payload, check=False)

    target = tmp_path / "target.json"
    target.write_text("target\n")
    symlink = tmp_path / "handoff-link.json"
    symlink.symlink_to(target)
    with pytest.raises(handoff.HandoffError, match="may not be a symlink"):
        handoff.write_manifest(symlink, payload, check=False)
