from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from mlperf.registry import load_registry
from tools import reference_source_lock

ROOT = Path(__file__).resolve().parents[1]


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def one_file_lock(path: Path, *, source_git_sha: str = "1" * 40) -> dict:
    data = path.read_bytes()
    contract_path = path.parent / "contract.yaml"
    if not contract_path.exists():
        contract_path.write_text(
            "id: example\n"
            "runner:\n"
            "  min: example:run_min\n"
            "  max: example:run_max\n"
            "quality_target:\n"
            "  metric: accuracy\n"
            "  value: 0.8\n"
            "  direction: higher\n"
        )
    contract_data = reference_source_lock.measurement_contract_bytes(
        contract_path.read_bytes(), workload_id="example"
    )
    return {
        "schema": reference_source_lock.SOURCE_LOCK_SCHEMA,
        "source_git_sha": source_git_sha,
        "file_count": 1,
        "files": [
            {
                "path": path.name,
                "normalization": reference_source_lock.source_normalization(path.name),
                "sha256": reference_source_lock.sha256_bytes(data),
                "n_bytes": len(data),
            }
        ],
        "contract_count": 1,
        "contracts": [
            {
                "workload": "example",
                "path": contract_path.name,
                "normalization": reference_source_lock.contract_normalization(),
                "sha256": reference_source_lock.sha256_bytes(contract_data),
                "n_bytes": len(contract_data),
            }
        ],
    }


def test_build_source_lock_binds_exact_commit_and_current_measurement_surface():
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        pytest.skip("measurement-surface binding requires a committed checkout")
    source_git_sha = current_commit()
    lock = reference_source_lock.build_source_lock(source_git_sha)

    assert lock["schema"] == "mlperf-edu-reference-source-lock/0.2"
    assert lock["source_git_sha"] == source_git_sha
    assert lock["file_count"] == len(reference_source_lock.MEASUREMENT_SOURCE_PATHS)
    assert [entry["path"] for entry in lock["files"]] == sorted(
        reference_source_lock.MEASUREMENT_SOURCE_PATHS
    )
    assert lock["contract_count"] == len(reference_source_lock.PROMOTED_CONTRACT_PATHS)
    assert [entry["workload"] for entry in lock["contracts"]] == sorted(
        reference_source_lock.PROMOTED_CONTRACT_PATHS
    )
    selected = {entry["path"]: entry for entry in lock["files"]}
    assert selected["src/mlperf/edu_cli.py"]["normalization"] == {
        "kind": "python-remove-top-level-functions/0.1",
        "removed_top_level_functions": ["public_audit_warnings"],
    }
    assert selected["src/mlperf/registry.py"]["normalization"] == {
        "kind": "python-remove-top-level-functions/0.1",
        "removed_top_level_functions": ["public_contract_issues"],
    }
    reference_source_lock.verify_source_lock(
        lock, expected_source_git_sha=source_git_sha
    )


def test_promoted_reference_local_dependencies_are_locked():
    dependencies = set()
    roots = tuple(
        path
        for path in reference_source_lock.MEASUREMENT_SOURCE_PATHS
        if path.startswith("src/") and path.endswith(".py")
    )
    for path in roots:
        dependencies.update(reference_source_lock.local_python_dependencies(path))

    assert {
        "src/mlperf/reference/cloud/gpt2_infer.py",
        "src/mlperf/roofline.py",
    }.issubset(dependencies)
    assert dependencies.issubset(reference_source_lock.MEASUREMENT_SOURCE_PATHS)


def test_source_lock_contracts_exactly_cover_registered_workloads():
    workloads = load_registry(ROOT / "registry")
    promotion_scope = {
        workload_id
        for workload_id, workload in workloads.items()
        if workload.raw.get("promotion_scope", True)
    }

    assert set(reference_source_lock.PROMOTED_CONTRACT_PATHS) == promotion_scope
    for (
        workload_id,
        relative_path,
    ) in reference_source_lock.PROMOTED_CONTRACT_PATHS.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert workload_id == workloads[workload_id].id


def test_source_lock_protects_every_reference_implementation():
    reference_root = ROOT / "src" / "mlperf" / "reference"
    implementations = {
        path.relative_to(ROOT).as_posix()
        for path in reference_root.rglob("*.py")
        if path.name != "__init__.py"
    }

    assert implementations
    assert implementations.issubset(reference_source_lock.MEASUREMENT_SOURCE_PATHS)


def test_python_normalization_removes_only_declared_top_level_function():
    before = b"""\
def protected():
    return 1

def public_audit_warnings(workload):
    return [\"old publication warning\"]

VALUE = protected()
"""
    publication_change = before.replace(
        b'return ["old publication warning"]',
        b'return ["new publication warning"]',
    )
    measurement_change = before.replace(b"return 1", b"return 2")

    normalized = reference_source_lock.normalize_source_bytes(
        "src/mlperf/edu_cli.py", before
    )
    assert normalized == reference_source_lock.normalize_source_bytes(
        "src/mlperf/edu_cli.py", publication_change
    )
    assert normalized != reference_source_lock.normalize_source_bytes(
        "src/mlperf/edu_cli.py", measurement_change
    )


@pytest.mark.parametrize(
    "unsafe",
    [
        "",
        ".",
        "../runner.py",
        "runner/../other.py",
        "/absolute.py",
        "./runner.py",
        "runner//model.py",
        "runner/",
        "runner\\model.py",
        "C:/runner.py",
        "runner.py\x00suffix",
        "runner.py\nsuffix",
    ],
)
def test_relative_path_validation_rejects_unsafe_or_noncanonical_paths(unsafe):
    with pytest.raises(reference_source_lock.SourceLockError):
        reference_source_lock.validate_relative_path(unsafe)


def test_load_source_lock_round_trips_canonical_bytes_without_git_history(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"print('measurement')\n")
    payload = one_file_lock(source)
    lock_path = tmp_path / "source_lock.json"
    lock_path.write_bytes(reference_source_lock.canonical_json_bytes(payload))

    loaded = reference_source_lock.load_source_lock(
        lock_path,
        project_root=tmp_path,
        expected_source_git_sha="1" * 40,
        expected_paths=("runner.py",),
        expected_contracts={"example": "contract.yaml"},
    )

    assert loaded == payload
    assert reference_source_lock.canonical_json_bytes(loaded) == lock_path.read_bytes()


def test_verify_source_lock_rejects_changed_checkout_bytes(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"original\n")
    payload = one_file_lock(source)
    source.write_bytes(b"changed!\n")

    with pytest.raises(reference_source_lock.SourceLockError, match="sha256"):
        reference_source_lock.verify_source_lock(
            payload,
            project_root=tmp_path,
            expected_paths=("runner.py",),
            expected_contracts={"example": "contract.yaml"},
        )


def test_verify_source_lock_rejects_changed_protected_contract_field(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"runner\n")
    payload = one_file_lock(source)
    contract_path = tmp_path / "contract.yaml"

    reference_source_lock.verify_source_lock(
        payload,
        project_root=tmp_path,
        expected_paths=("runner.py",),
        expected_contracts={"example": "contract.yaml"},
    )
    contract_path.write_text(
        contract_path.read_text().replace("value: 0.8", "value: 0.9")
    )

    with pytest.raises(reference_source_lock.SourceLockError, match="contract sha256"):
        reference_source_lock.verify_source_lock(
            payload,
            project_root=tmp_path,
            expected_paths=("runner.py",),
            expected_contracts={"example": "contract.yaml"},
        )


def test_contract_projection_excludes_only_declared_review_metadata():
    original = b"""\
id: example
runner:
  max: example:run_max
quality_target:
  metric: accuracy
  value: 0.8
  variance_summary:
    median: 0.9
  reviewer_notes:
  - display only
verified_baseline:
  median: 0.9
verified_baselines:
  example__max__training:
    median: 0.9
calibration_observation:
  median: 0.7
"""
    metadata_change = original.replace(b"median: 0.9", b"median: 0.95").replace(
        b"display only", b"reworded"
    )
    protected_change = original.replace(b"value: 0.8", b"value: 0.85")

    projected = reference_source_lock.measurement_contract_bytes(
        original, workload_id="example"
    )
    assert projected == reference_source_lock.measurement_contract_bytes(
        metadata_change, workload_id="example"
    )
    assert projected != reference_source_lock.measurement_contract_bytes(
        protected_change, workload_id="example"
    )


def test_verify_source_lock_rejects_omitted_duplicate_and_unsorted_paths(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    first_entry = one_file_lock(first)["files"][0]
    second_entry = one_file_lock(second)["files"][0]

    omitted = one_file_lock(first)
    with pytest.raises(reference_source_lock.SourceLockError, match="missing"):
        reference_source_lock.verify_source_lock(
            omitted,
            project_root=tmp_path,
            expected_paths=("first.py", "second.py"),
        )

    duplicate = copy.deepcopy(omitted)
    duplicate["file_count"] = 2
    duplicate["files"].append(copy.deepcopy(first_entry))
    with pytest.raises(reference_source_lock.SourceLockError, match="duplicate"):
        reference_source_lock.verify_source_lock(
            duplicate,
            project_root=tmp_path,
            expected_paths=None,
            expected_contracts={"example": "contract.yaml"},
        )

    unsorted = copy.deepcopy(omitted)
    unsorted["file_count"] = 2
    unsorted["files"] = [second_entry, first_entry]
    with pytest.raises(reference_source_lock.SourceLockError, match="sorted"):
        reference_source_lock.verify_source_lock(
            unsorted,
            project_root=tmp_path,
            expected_paths=None,
            expected_contracts={"example": "contract.yaml"},
        )


def test_load_source_lock_rejects_duplicate_keys_and_noncanonical_json(tmp_path):
    source = tmp_path / "runner.py"
    source.write_bytes(b"runner\n")
    payload = one_file_lock(source)
    lock_path = tmp_path / "source_lock.json"

    lock_path.write_text('{"schema":"a","schema":"b"}\n')
    with pytest.raises(reference_source_lock.SourceLockError, match="duplicate JSON"):
        reference_source_lock.load_source_lock(
            lock_path,
            project_root=tmp_path,
            expected_paths=("runner.py",),
            expected_contracts={"example": "contract.yaml"},
        )

    lock_path.write_text(json.dumps(payload))
    with pytest.raises(reference_source_lock.SourceLockError, match="canonically"):
        reference_source_lock.load_source_lock(
            lock_path,
            project_root=tmp_path,
            expected_paths=("runner.py",),
            expected_contracts={"example": "contract.yaml"},
        )


def test_source_lock_rejects_symlinked_measurement_file(tmp_path):
    target = tmp_path / "target.py"
    target.write_bytes(b"target\n")
    link = tmp_path / "runner.py"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    payload = one_file_lock(target)
    payload["files"][0]["path"] = "runner.py"

    with pytest.raises(reference_source_lock.SourceLockError, match="symlink"):
        reference_source_lock.verify_source_lock(
            payload,
            project_root=tmp_path,
            expected_paths=("runner.py",),
            expected_contracts={"example": "contract.yaml"},
        )
