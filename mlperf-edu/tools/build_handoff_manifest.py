#!/usr/bin/env python3
"""Build a deterministic manifest for an external reference-evidence handoff.

The manifest contains only paths relative to the supplied evidence and package
roots. It verifies retained evidence against its historical source commit,
checks every byte indexed by each portable package, and records policy-blocked
attempts without pretending that a redistributable archive exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import stat
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mlperf.edu_cli import package_dataset_policy_issue, verify_package_archive  # noqa: E402
from tools import import_reference_evidence  # noqa: E402
from tools import reference_source_lock  # noqa: E402
from tools import run_reference_sweep  # noqa: E402

HANDOFF_SCHEMA = "mlperf-edu-handoff-manifest/0.1"
INDEX_SCHEMA = "mlperf-edu-reference-index/0.2"
PACKAGE_SCHEMA = "mlperf-edu-package/0.2"
MAX_PACKAGE_MEMBERS = 256
MAX_PACKAGE_BYTES = 2 * 1024**3
MAX_PACKAGE_INDEX_BYTES = 1 << 20
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OFFICIAL_COUNTS = {
    "reference_summaries": 8,
    "attempts": 40,
    "evidence_valid_attempts": 40,
    "portable_packages": 35,
    "policy_blocked_attempts": 5,
}


class HandoffError(ValueError):
    """Raised when handoff inputs do not form a closed, verified set."""


@dataclass(frozen=True)
class VerifiedReferenceSet:
    """Validated committed and retained evidence needed by the builder."""

    index: dict[str, Any]
    index_bytes: bytes
    source_lock: dict[str, Any]
    source_lock_bytes: bytes
    selected: dict[str, tuple[Path, dict[str, Any], bytes]]
    lineage: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest without loading a large file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_stream(handle: Any) -> str:
    """Return a prefixed SHA-256 digest for a binary stream."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    """Serialize a handoff manifest deterministically."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_object_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    """Load one UTF-8 JSON object while rejecting duplicate keys."""
    try:
        payload = json.loads(
            data.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HandoffError(f"{label} root must be an object")
    return payload


def load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise HandoffError(f"{label} is missing or is a symlink: {path}")
    data = path.read_bytes()
    return load_json_object_bytes(data, label=label), data


def safe_relative_path(value: object, *, label: str) -> str:
    """Validate a canonical POSIX path that cannot escape its logical root."""
    if not isinstance(value, str) or not value:
        raise HandoffError(f"{label} must be a non-empty string")
    if "\\" in value or "\x00" in value or any(ord(char) < 32 for char in value):
        raise HandoffError(f"{label} is not a safe POSIX path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise HandoffError(f"{label} must be a canonical relative path: {value!r}")
    if ":" in path.parts[0]:
        raise HandoffError(f"{label} contains a drive-like prefix: {value!r}")
    return value


def resolve_under(root: Path, relative: str, *, label: str) -> Path:
    relative = safe_relative_path(relative, label=label)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise HandoffError(f"{label} traverses a symlink: {relative}")
    resolved = cursor.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HandoffError(f"{label} escapes its root: {relative}") from exc
    if not resolved.is_file():
        raise HandoffError(f"{label} is missing: {relative}")
    return resolved


def archive_relative(owner: str, relative: object, *, label: str) -> str:
    """Resolve an owner-relative package pointer while staying in the archive."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise HandoffError(f"{label} is not a relative archive pointer")
    if PurePosixPath(relative).is_absolute() or "\x00" in relative:
        raise HandoffError(f"{label} is not a relative archive pointer")
    normalized = posixpath.normpath(posixpath.join(posixpath.dirname(owner), relative))
    return safe_relative_path(normalized, label=label)


def validate_index(index: dict[str, Any]) -> None:
    if index.get("schema") != INDEX_SCHEMA:
        raise HandoffError(
            f"unsupported reference index schema: {index.get('schema')!r}"
        )
    source_sha = index.get("source_git_sha")
    if not isinstance(source_sha, str) or not GIT_SHA_RE.fullmatch(source_sha):
        raise HandoffError("reference index source_git_sha must be a full Git SHA")
    summaries = index.get("summaries")
    if not isinstance(summaries, list) or not summaries:
        raise HandoffError("reference index summaries must be a non-empty list")
    if index.get("summary_count") != len(summaries):
        raise HandoffError("reference index summary_count does not match summaries")
    workloads: list[str] = []
    evidence_ids: list[str] = []
    paths: list[str] = []
    for position, entry in enumerate(summaries):
        if not isinstance(entry, dict):
            raise HandoffError(f"reference index summary {position} is not an object")
        workload = entry.get("workload")
        evidence_id = entry.get("evidence_id")
        if not isinstance(workload, str) or not workload:
            raise HandoffError(f"reference index summary {position} lacks workload")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise HandoffError(f"reference index summary {position} lacks evidence_id")
        safe_relative_path(evidence_id, label=f"summary {position} evidence_id")
        path = safe_relative_path(entry.get("path"), label=f"summary {position} path")
        digest = entry.get("evidence_sha256")
        if not isinstance(digest, str) or not RAW_SHA256_RE.fullmatch(digest):
            raise HandoffError(
                f"reference index summary {position} has invalid SHA-256"
            )
        workloads.append(workload)
        evidence_ids.append(evidence_id)
        paths.append(path)
    for label, values in (
        ("workload", workloads),
        ("evidence_id", evidence_ids),
        ("path", paths),
    ):
        if len(values) != len(set(values)):
            raise HandoffError(f"reference index contains duplicate {label} values")


def verify_source_lock(
    index: dict[str, Any], *, project_root: Path
) -> tuple[dict[str, Any], bytes, str]:
    record = index.get("source_lock")
    if not isinstance(record, dict):
        raise HandoffError("reference index lacks a source_lock record")
    relative = safe_relative_path(record.get("path"), label="source lock path")
    lock_path = resolve_under(project_root, relative, label="source lock path")
    lock_bytes = lock_path.read_bytes()
    actual_digest = reference_source_lock.sha256_bytes(lock_bytes)
    if record.get("sha256") != actual_digest:
        raise HandoffError("source lock digest does not match the reference index")
    try:
        payload = reference_source_lock.load_source_lock(
            lock_path,
            project_root=project_root,
            expected_source_git_sha=index["source_git_sha"],
        )
    except reference_source_lock.SourceLockError as exc:
        raise HandoffError(f"source lock verification failed: {exc}") from exc
    for field in ("schema", "file_count", "contract_count"):
        if record.get(field) != payload.get(field):
            raise HandoffError(
                f"source lock {field} does not match the reference index"
            )
    return payload, lock_bytes, relative


def _lineage_claims(
    selected: dict[str, tuple[Path, dict[str, Any], bytes]],
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for _path, payload, _data in selected.values():
        lineage = payload.get("nanogpt_training_lineage")
        if lineage is not None:
            if not isinstance(lineage, dict):
                raise HandoffError("NanoGPT lineage claim is not an object")
            claims.append(lineage)
    if not claims:
        raise HandoffError("reference set has no NanoGPT lineage claims")
    first = claims[0]
    if any(claim != first for claim in claims[1:]):
        raise HandoffError("NanoGPT reference summaries disagree on training lineage")
    return claims


def verify_lineage_archive(
    lineage_archive: Path,
    *,
    evidence_root: Path,
    selected: dict[str, tuple[Path, dict[str, Any], bytes]],
    historical_source_root: Path,
) -> dict[str, Any]:
    """Re-run clean-extraction verification for the shared NanoGPT lineage."""
    if lineage_archive.is_symlink() or not lineage_archive.is_file():
        raise HandoffError(
            f"lineage archive is missing or is a symlink: {lineage_archive}"
        )
    lineage_archive = lineage_archive.resolve()
    try:
        relative = lineage_archive.relative_to(evidence_root.resolve()).as_posix()
    except ValueError as exc:
        raise HandoffError(
            "lineage archive must be contained by the evidence root"
        ) from exc
    safe_relative_path(relative, label="lineage archive path")
    claims = _lineage_claims(selected)
    claim = claims[0]
    expected_digest = claim.get("package_sha256")
    if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
        raise HandoffError("NanoGPT lineage package digest is invalid")
    actual_digest = sha256_file(lineage_archive)
    if actual_digest != expected_digest:
        raise HandoffError("lineage archive digest does not match reference summaries")
    try:
        package_index = run_reference_sweep._preflight_lineage_archive(lineage_archive)
        checks = verify_package_archive(
            lineage_archive, repo_root=historical_source_root
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise HandoffError(f"lineage archive verification failed: {exc}") from exc
    failed = [name for name, ok, _detail in checks if not ok]
    if failed:
        raise HandoffError(f"lineage archive failed clean extraction: {failed}")
    expected_check_count = claim.get("verification_check_count")
    if expected_check_count != len(checks):
        raise HandoffError(
            "lineage verification count does not match reference summaries"
        )
    included = package_index.get("included_files") or []
    weights = [item for item in included if item.get("role") == "weights"]
    if len(weights) != 1:
        raise HandoffError("lineage archive must index exactly one weights artifact")
    manifest_name = safe_relative_path(
        package_index.get("manifest"), label="lineage manifest path"
    )
    with zipfile.ZipFile(lineage_archive) as archive:
        manifest = load_json_object_bytes(
            archive.read(manifest_name), label="lineage packaged manifest"
        )
    source_seed = ((manifest.get("leaves") or {}).get("rng") or {}).get("seed")
    if isinstance(source_seed, bool) or not isinstance(source_seed, int):
        raise HandoffError("lineage manifest does not record an integer seed")
    return {
        "path_root": "evidence_root",
        "path": relative,
        "sha256": actual_digest,
        "n_bytes": lineage_archive.stat().st_size,
        "schema": package_index.get("schema"),
        "workload": package_index.get("workload"),
        "source_seed": source_seed,
        "checkpoint_sha256": weights[0].get("sha256"),
        "indexed_file_count": len(included),
        "clean_extraction_verification_check_count": len(checks),
        "verification": "passed",
    }


def verify_reference_set(
    index_path: Path,
    evidence_root: Path,
    lineage_archive: Path,
    *,
    project_root: Path = ROOT,
) -> VerifiedReferenceSet:
    """Verify the committed index, source lock, raw evidence, and lineage."""
    canonical_index_path = project_root / "reference_results" / "index.json"
    if index_path.expanduser().resolve() != canonical_index_path.resolve():
        raise HandoffError(
            "handoff verification requires the committed canonical "
            "reference_results/index.json"
        )
    index, index_bytes = load_json_object(index_path, label="reference index")
    validate_index(index)
    source_lock, source_lock_bytes, _relative = verify_source_lock(
        index, project_root=project_root
    )
    evidence_root = evidence_root.resolve()
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        raise HandoffError(f"evidence root is missing or is a symlink: {evidence_root}")
    try:
        sweep_digest = import_reference_evidence.source_sweep_tool_sha256(
            index["source_git_sha"]
        )
        with import_reference_evidence.source_project_checkout(
            index["source_git_sha"]
        ) as historical_source_root:
            selected = import_reference_evidence.discover_summaries(
                evidence_root,
                source_git_sha=index["source_git_sha"],
                sweep_tool_sha256=sweep_digest,
                source_project_root=historical_source_root,
            )
            lineage = verify_lineage_archive(
                lineage_archive,
                evidence_root=evidence_root,
                selected=selected,
                historical_source_root=historical_source_root,
            )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise HandoffError(f"retained evidence verification failed: {exc}") from exc

    indexed = {entry["workload"]: entry for entry in index["summaries"]}
    if set(indexed) != set(selected):
        raise HandoffError("committed index and retained evidence workloads differ")
    for workload, entry in indexed.items():
        raw_path, payload, raw_bytes = selected[workload]
        if raw_path.parent.name != entry["evidence_id"]:
            raise HandoffError(
                f"{workload}: evidence directory does not match evidence_id"
            )
        committed_path = resolve_under(
            project_root, entry["path"], label=f"{workload} committed summary"
        )
        committed_bytes = committed_path.read_bytes()
        digest = hashlib.sha256(committed_bytes).hexdigest()
        if digest != entry["evidence_sha256"]:
            raise HandoffError(f"{workload}: committed summary digest mismatch")
        if committed_bytes != raw_bytes:
            raise HandoffError(f"{workload}: committed and retained summaries differ")
        if payload.get("evidence_id") != entry["evidence_id"]:
            raise HandoffError(f"{workload}: summary evidence_id mismatch")
        run_seeds = [run.get("requested_seed") for run in payload.get("runs") or []]
        if run_seeds != entry.get("seeds"):
            raise HandoffError(f"{workload}: index seeds differ from retained runs")
    return VerifiedReferenceSet(
        index=index,
        index_bytes=index_bytes,
        source_lock=source_lock,
        source_lock_bytes=source_lock_bytes,
        selected=selected,
        lineage=lineage,
    )


def _zip_index(
    archive: zipfile.ZipFile, *, label: str
) -> tuple[dict[str, Any], list[str]]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(infos) > MAX_PACKAGE_MEMBERS:
        raise HandoffError(
            f"{label} contains {len(infos)} members; limit is {MAX_PACKAGE_MEMBERS}"
        )
    if len(names) != len(set(names)):
        raise HandoffError(f"{label} contains duplicate archive members")
    total_size = 0
    for info in infos:
        if not run_reference_sweep._safe_archive_member_name(info.filename):
            raise HandoffError(f"{label} contains unsafe member {info.filename!r}")
        if info.flag_bits & 0x1:
            raise HandoffError(f"{label} contains encrypted member {info.filename!r}")
        file_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
        if info.is_dir() or file_type not in {0, stat.S_IFREG}:
            raise HandoffError(f"{label} contains non-regular member {info.filename!r}")
        total_size += info.file_size
    if total_size > MAX_PACKAGE_BYTES:
        raise HandoffError(
            f"{label} expands to {total_size} bytes; limit is {MAX_PACKAGE_BYTES}"
        )
    if "package_index.json" not in names:
        raise HandoffError(f"{label} lacks package_index.json")
    index_info = archive.getinfo("package_index.json")
    if index_info.file_size > MAX_PACKAGE_INDEX_BYTES:
        raise HandoffError(f"{label} package index is unexpectedly large")
    index = load_json_object_bytes(
        archive.read(index_info), label=f"{label} package index"
    )
    if index.get("schema") != PACKAGE_SCHEMA:
        raise HandoffError(f"{label} uses unsupported package schema")
    return index, names


def _verify_indexed_members(
    archive: zipfile.ZipFile,
    index: dict[str, Any],
    names: list[str],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    included = index.get("included_files")
    if not isinstance(included, list) or not included:
        raise HandoffError(f"{label} contains no indexed files")
    records: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(included):
        if not isinstance(item, dict):
            raise HandoffError(f"{label} index item {position} is not an object")
        relative = safe_relative_path(
            item.get("path"), label=f"{label} index item {position} path"
        )
        if relative in records:
            raise HandoffError(f"{label} indexes {relative} more than once")
        digest = item.get("sha256")
        n_bytes = item.get("n_bytes")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise HandoffError(f"{label} has invalid digest for {relative}")
        if isinstance(n_bytes, bool) or not isinstance(n_bytes, int) or n_bytes < 0:
            raise HandoffError(f"{label} has invalid size for {relative}")
        try:
            info = archive.getinfo(relative)
        except KeyError as exc:
            raise HandoffError(f"{label} is missing indexed member {relative}") from exc
        if info.file_size != n_bytes:
            raise HandoffError(f"{label} size mismatch for {relative}")
        with archive.open(info) as handle:
            if sha256_stream(handle) != digest:
                raise HandoffError(f"{label} digest mismatch for {relative}")
        records[relative] = item
    expected = {"package_index.json", *records}
    if set(names) != expected:
        raise HandoffError(f"{label} archive/index coverage differs")
    return records


def _passed_creation_verification(index: dict[str, Any], *, label: str) -> int:
    source = index.get("source_verification")
    checks = source.get("checks") if isinstance(source, dict) else None
    if (
        not isinstance(source, dict)
        or source.get("passed") is not True
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(check, dict) or check.get("ok") is not True
            for check in checks
        )
    ):
        raise HandoffError(f"{label} source verification is absent or failed")
    legacy_checks = index.get("verification")
    if legacy_checks != checks:
        raise HandoffError(f"{label} verification mirrors disagree")
    clean = index.get("clean_extraction_verification")
    if not isinstance(clean, dict) or clean != {"required": True, "status": "passed"}:
        raise HandoffError(f"{label} lacks passing clean-extraction metadata")
    return len(checks)


def verify_portable_package(
    package_path: Path,
    *,
    historical_source_root: Path,
    source_git_sha: str,
    evidence_id: str,
    payload: dict[str, Any],
    run: dict[str, Any],
    raw_report: dict[str, Any],
) -> dict[str, Any]:
    """Hash a portable archive and bind its semantic result to one raw attempt."""
    label = package_path.name
    if package_path.is_symlink() or not package_path.is_file():
        raise HandoffError(
            f"portable package is missing or is a symlink: {package_path}"
        )
    before = package_path.stat()
    try:
        with zipfile.ZipFile(package_path) as archive:
            package_index, names = _zip_index(archive, label=label)
            included = _verify_indexed_members(
                archive, package_index, names, label=label
            )
            check_count = _passed_creation_verification(package_index, label=label)
            manifest_name = safe_relative_path(
                package_index.get("manifest"), label=f"{label} manifest"
            )
            if manifest_name not in included:
                raise HandoffError(f"{label} manifest is not indexed")
            manifest = load_json_object_bytes(
                archive.read(manifest_name), label=f"{label} manifest"
            )
            report_name = archive_relative(
                manifest_name,
                ((manifest.get("leaves") or {}).get("measurement") or {}).get(
                    "report_path"
                ),
                label=f"{label} report",
            )
            if report_name not in included:
                raise HandoffError(f"{label} report is not indexed")
            report = load_json_object_bytes(
                archive.read(report_name), label=f"{label} report"
            )
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise HandoffError(f"cannot verify {label}: {exc}") from exc

    seed = run.get("requested_seed")
    source_tree = (manifest.get("leaves") or {}).get("source_tree") or {}
    manifest_seed = ((manifest.get("leaves") or {}).get("rng") or {}).get("seed")
    expected_identity = {
        "workload": payload.get("workload"),
        "profile": payload.get("profile"),
        "variant": payload.get("variant"),
        "seed": seed,
        "status": run.get("status"),
    }
    if package_index.get("workload") != payload.get("workload"):
        raise HandoffError(f"{label} workload differs from retained evidence")
    if manifest.get("workload") != payload.get("workload") or manifest_seed != seed:
        raise HandoffError(f"{label} manifest identity differs from retained evidence")
    if source_tree.get("git_sha") != source_git_sha:
        raise HandoffError(f"{label} source commit differs from the reference index")
    for field, expected in expected_identity.items():
        if report.get(field) != expected:
            raise HandoffError(f"{label} report {field} differs from raw evidence")
    for field in ("metrics", "quality", "review_contract"):
        if report.get(field) != raw_report.get(field):
            raise HandoffError(f"{label} report {field} differs from raw evidence")
    try:
        clean_checks = verify_package_archive(
            package_path, repo_root=historical_source_root
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise HandoffError(
            f"{label} clean-extraction verification could not run: {exc}"
        ) from exc
    failed_clean_checks = [name for name, ok, _detail in clean_checks if not ok]
    if failed_clean_checks:
        raise HandoffError(
            f"{label} failed clean-extraction verification: {failed_clean_checks}"
        )
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    package_digest = sha256_file(package_path)
    after = package_path.stat()
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if fingerprint_before != fingerprint_after:
        raise HandoffError(f"{label} changed during verification")
    return {
        "status": "packaged",
        "path_root": "portable_package_root",
        "path": package_path.name,
        "sha256": package_digest,
        "n_bytes": after.st_size,
        "schema": package_index.get("schema"),
        "indexed_file_count": len(package_index["included_files"]),
        "indexed_payload_bytes": sum(
            item["n_bytes"] for item in package_index["included_files"]
        ),
        "creation_source_verification_check_count": check_count,
        "indexed_payload_verification": "passed",
        "clean_extraction_verification": "passed",
        "clean_extraction_verification_check_count": len(clean_checks),
        "evidence_id": evidence_id,
    }


def _artifact_records(evidence_id: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, item in enumerate(run.get("artifacts") or []):
        if not isinstance(item, dict):
            raise HandoffError(f"{evidence_id}: artifact {position} is not an object")
        relative = safe_relative_path(
            item.get("path"), label=f"{evidence_id} artifact {position}"
        )
        records.append(
            {
                "path_root": "evidence_root",
                "path": f"{evidence_id}/{relative}",
                "role": item.get("role"),
                "sha256": item.get("sha256"),
                "n_bytes": item.get("n_bytes"),
            }
        )
    return sorted(records, key=lambda item: (str(item["path"]), str(item["role"])))


def _raw_attempt(
    evidence_root: Path,
    evidence_id: str,
    run: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_relative = safe_relative_path(
        run.get("report_path"), label=f"{evidence_id} report path"
    )
    manifest_relative = safe_relative_path(
        run.get("manifest_path"), label=f"{evidence_id} manifest path"
    )
    report_path = resolve_under(
        evidence_root / evidence_id, report_relative, label=f"{evidence_id} report"
    )
    manifest_path = resolve_under(
        evidence_root / evidence_id,
        manifest_relative,
        label=f"{evidence_id} manifest",
    )
    report, _ = load_json_object(report_path, label=f"{evidence_id} report")
    manifest, _ = load_json_object(manifest_path, label=f"{evidence_id} manifest")
    attempt_path = PurePosixPath(report_relative).parent.as_posix()
    return (
        {
            "path_root": "evidence_root",
            "path": f"{evidence_id}/{attempt_path}",
            "report_path": f"{evidence_id}/{report_relative}",
            "manifest_path": f"{evidence_id}/{manifest_relative}",
            "artifacts": _artifact_records(evidence_id, run),
        },
        {"report": report, "manifest": manifest},
    )


def build_attempts(
    reference_set: VerifiedReferenceSet,
    evidence_root: Path,
    package_root: Path,
    *,
    historical_source_root: Path,
) -> list[dict[str, Any]]:
    """Build the closed attempt ledger and reject extra or missing packages."""
    package_root = package_root.resolve()
    if not package_root.is_dir() or package_root.is_symlink():
        raise HandoffError(
            f"portable package root is missing or is a symlink: {package_root}"
        )
    actual_package_files: dict[str, Path] = {}
    for child in package_root.iterdir():
        if child.is_symlink() or not child.is_file() or child.suffix != ".zip":
            raise HandoffError(
                f"unexpected entry in portable package root: {child.name}"
            )
        if child.name in actual_package_files:
            raise HandoffError(f"duplicate portable package name: {child.name}")
        actual_package_files[child.name] = child

    evidence_root = evidence_root.resolve()
    attempts: list[dict[str, Any]] = []
    expected_package_names: set[str] = set()
    for workload, (_summary_path, payload, _data) in sorted(
        reference_set.selected.items()
    ):
        evidence_id = payload["evidence_id"]
        for run in sorted(payload["runs"], key=lambda item: item["requested_seed"]):
            seed = run["requested_seed"]
            raw_record, raw_payloads = _raw_attempt(evidence_root, evidence_id, run)
            policy_issue = package_dataset_policy_issue(raw_payloads["manifest"])
            package_name = f"{evidence_id}-seed_{seed}.zip"
            package_path = actual_package_files.get(package_name)
            if policy_issue:
                if package_path is not None:
                    raise HandoffError(
                        f"{package_name} exists even though dataset policy blocks packaging"
                    )
                dataset = (raw_payloads["manifest"].get("leaves") or {}).get(
                    "dataset"
                ) or {}
                package_record = {
                    "status": "policy-blocked",
                    "dataset": dataset.get("name"),
                    "reason": policy_issue,
                }
            else:
                expected_package_names.add(package_name)
                if package_path is None:
                    raise HandoffError(
                        f"required portable package is missing: {package_name}"
                    )
                package_record = verify_portable_package(
                    package_path,
                    historical_source_root=historical_source_root,
                    source_git_sha=reference_set.index["source_git_sha"],
                    evidence_id=evidence_id,
                    payload=payload,
                    run=run,
                    raw_report=raw_payloads["report"],
                )
            attempts.append(
                {
                    "attempt_id": f"{evidence_id}/seed_{seed}",
                    "evidence_id": evidence_id,
                    "workload": workload,
                    "profile": payload.get("profile"),
                    "variant": payload.get("variant"),
                    "seed": seed,
                    "status": run.get("status"),
                    "evidence_valid": run.get("evidence_valid"),
                    "raw": raw_record,
                    "portable_package": package_record,
                }
            )
    extras = sorted(set(actual_package_files).difference(expected_package_names))
    if extras:
        raise HandoffError(f"portable package root contains unbound archives: {extras}")
    return attempts


def _reference_summaries(
    reference_set: VerifiedReferenceSet,
) -> list[dict[str, Any]]:
    records = []
    for entry in reference_set.index["summaries"]:
        payload = reference_set.selected[entry["workload"]][1]
        records.append(
            {
                "workload": entry["workload"],
                "evidence_id": entry["evidence_id"],
                "schema": payload.get("schema"),
                "committed_path_root": "project_root",
                "committed_path": entry["path"],
                "retained_path_root": "evidence_root",
                "retained_path": f"{entry['evidence_id']}/evidence_summary.json",
                "sha256": "sha256:" + entry["evidence_sha256"],
                "n_bytes": len(reference_set.selected[entry["workload"]][2]),
                "run_count": len(payload.get("runs") or []),
            }
        )
    return sorted(records, key=lambda item: item["workload"])


def build_manifest(
    reference_set: VerifiedReferenceSet,
    evidence_root: Path,
    package_root: Path,
    *,
    promotion_git_sha: str | None,
    expected_counts: dict[str, int] | None = OFFICIAL_COUNTS,
    historical_source_root: Path | None = None,
) -> dict[str, Any]:
    if promotion_git_sha is not None and not GIT_SHA_RE.fullmatch(promotion_git_sha):
        raise HandoffError("promotion_git_sha must be a full 40-character Git SHA")
    if historical_source_root is None:
        with import_reference_evidence.source_project_checkout(
            reference_set.index["source_git_sha"]
        ) as source_root:
            attempts = build_attempts(
                reference_set,
                evidence_root,
                package_root,
                historical_source_root=source_root,
            )
    else:
        source_root = historical_source_root.resolve()
        if not source_root.is_dir() or source_root.is_symlink():
            raise HandoffError(
                f"historical source root is missing or is a symlink: {source_root}"
            )
        attempts = build_attempts(
            reference_set,
            evidence_root,
            package_root,
            historical_source_root=source_root,
        )
    package_statuses = Counter(
        attempt["portable_package"]["status"] for attempt in attempts
    )
    blocked_by_workload = Counter(
        attempt["workload"]
        for attempt in attempts
        if attempt["portable_package"]["status"] == "policy-blocked"
    )
    counts = {
        "reference_summaries": len(reference_set.index["summaries"]),
        "attempts": len(attempts),
        "evidence_valid_attempts": sum(
            attempt["evidence_valid"] is True for attempt in attempts
        ),
        "portable_packages": package_statuses["packaged"],
        "policy_blocked_attempts": package_statuses["policy-blocked"],
    }
    if expected_counts is not None and counts != expected_counts:
        raise HandoffError(
            f"handoff counts {counts!r} do not match {expected_counts!r}"
        )
    source_lock_record = reference_set.index["source_lock"]
    manifest = {
        "schema": HANDOFF_SCHEMA,
        "digest_policy": (
            "SHA-256 digests provide unauthenticated integrity checking, not "
            "producer authentication."
        ),
        "path_roots": {
            "project_root": "Paths are relative to the MLPerf EDU project root.",
            "evidence_root": "Paths are relative to --evidence-root.",
            "portable_package_root": ("Paths are relative to --portable-package-root."),
        },
        "source_git_sha": reference_set.index["source_git_sha"],
        "promotion_git_sha": promotion_git_sha,
        "reference_index": {
            "path_root": "project_root",
            "path": "reference_results/index.json",
            "schema": reference_set.index["schema"],
            "sha256": "sha256:" + hashlib.sha256(reference_set.index_bytes).hexdigest(),
            "n_bytes": len(reference_set.index_bytes),
        },
        "source_lock": {
            "path_root": "project_root",
            "path": source_lock_record["path"],
            "schema": reference_set.source_lock["schema"],
            "sha256": reference_source_lock.sha256_bytes(
                reference_set.source_lock_bytes
            ),
            "n_bytes": len(reference_set.source_lock_bytes),
            "file_count": reference_set.source_lock["file_count"],
            "contract_count": reference_set.source_lock["contract_count"],
            "verification": "passed",
        },
        "nanogpt_training_lineage": reference_set.lineage,
        "counts": counts,
        "policy_blocked_by_workload": dict(sorted(blocked_by_workload.items())),
        "reference_summaries": _reference_summaries(reference_set),
        "attempts": sorted(attempts, key=lambda item: (item["workload"], item["seed"])),
        "verification": {
            "committed_to_retained_summary_binding": "passed",
            "historical_raw_evidence": "passed",
            "source_lock": "passed",
            "portable_package_indexed_payloads": "passed",
            "portable_package_clean_extraction": "passed",
            "nanogpt_lineage_clean_extraction": "passed",
        },
    }
    return manifest


def write_manifest(output: Path, payload: dict[str, Any], *, check: bool) -> None:
    """Write or compare an external manifest without touching the checkout."""
    output = output.expanduser()
    resolved_parent = output.parent.resolve()
    resolved = resolved_parent / output.name
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise HandoffError("handoff output must be outside the project checkout")
    if output.is_symlink():
        raise HandoffError(f"handoff output may not be a symlink: {output}")
    expected = canonical_bytes(payload)
    if check:
        if not output.is_file() or output.read_bytes() != expected:
            raise HandoffError(f"handoff manifest is missing or stale: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--portable-package-root", type=Path, required=True)
    parser.add_argument("--lineage-archive", type=Path, required=True)
    parser.add_argument("--promotion-git-sha")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="Compare an existing output; do not write."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reference_set = verify_reference_set(
            ROOT / "reference_results" / "index.json",
            args.evidence_root,
            args.lineage_archive,
        )
        manifest = build_manifest(
            reference_set,
            args.evidence_root,
            args.portable_package_root,
            promotion_git_sha=args.promotion_git_sha,
        )
        write_manifest(args.output, manifest, check=args.check)
    except (HandoffError, OSError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "wrote"
    print(
        f"{action} {args.output}: {manifest['counts']['attempts']} attempts, "
        f"{manifest['counts']['portable_packages']} portable packages, "
        f"{manifest['counts']['policy_blocked_attempts']} policy-blocked attempts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
