"""Build and verify the source lock for promoted reference evidence.

Reference summaries bind the sweep orchestrator, but benchmark meaning also
depends on the contracts, harness, runners, reference implementations, and
quality assets used by that orchestrator.  This module records that focused
measurement surface from an exact Git commit and verifies that a checkout
still contains the same bytes.  Verification of a committed lock does not
need Git history, which keeps it safe for shallow CI checkouts.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK_SCHEMA = "mlperf-edu-reference-source-lock/0.2"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RAW_NORMALIZATION = {"kind": "raw-bytes/0.1"}
PYTHON_FUNCTION_NORMALIZATION = "python-remove-top-level-functions/0.1"
CONTRACT_PROJECTION_NORMALIZATION = "native-registry-measurement-contract/0.1"
CONTRACT_EXCLUDED_TOP_LEVEL_FIELDS = (
    "calibration_observation",
    "verified_baseline",
    "verified_baselines",
)
CONTRACT_EXCLUDED_QUALITY_TARGET_FIELDS = (
    "reviewer_notes",
    "variance_summary",
)

# Keep this list focused on files that can change workload execution,
# measurement, grading, or provenance. Documentation, publication metadata,
# and generated evidence are deliberately excluded.
MEASUREMENT_SOURCE_PATHS = (
    "src/mlperf/assets.py",
    "src/mlperf/contracts.py",
    "src/mlperf/edu_cli.py",
    "src/mlperf/fingerprint.py",
    "src/mlperf/harness.py",
    "src/mlperf/manifest.py",
    "src/mlperf/power.py",
    "src/mlperf/reference/cloud/gpt2_infer.py",
    "src/mlperf/reference/cloud/nanogpt_decode.py",
    "src/mlperf/reference/cloud/nanogpt_prefill.py",
    "src/mlperf/reference/cloud/nanogpt_train.py",
    "src/mlperf/reference/timeseries/patchtst/__init__.py",
    "src/mlperf/reference/timeseries/patchtst/backbone.py",
    "src/mlperf/reference/timeseries/patchtst/layers.py",
    "src/mlperf/reference/timeseries/patchtst/revin.py",
    "src/mlperf/reference/tiny/mlperf_tiny_anomaly.py",
    "src/mlperf/reference/tiny/mlperf_tiny_kws.py",
    "src/mlperf/reference/tiny/mlperf_tiny_resnet.py",
    "src/mlperf/reference/tiny/mlperf_tiny_vww.py",
    "src/mlperf/registry.py",
    "src/mlperf/roofline.py",
    "src/mlperf/runners/common.py",
    "src/mlperf/runners/graph.py",
    "src/mlperf/runners/nanogpt.py",
    "src/mlperf/runners/retrieval.py",
    "src/mlperf/runners/text.py",
    "src/mlperf/runners/timeseries.py",
    "src/mlperf/runners/tiny.py",
    "src/mlperf/runners/vision.py",
    "tools/run_reference_sweep.py",
)

# These two product-path modules acquired publication-only checks after the
# evidence source commit.  Removing the complete named functions keeps those
# review-surface additions from masquerading as measurement drift while every
# other byte in each module remains protected.
SOURCE_NORMALIZATIONS: Mapping[str, tuple[str, ...]] = {
    "src/mlperf/edu_cli.py": ("public_audit_warnings",),
    "src/mlperf/registry.py": ("public_contract_issues",),
}

PROMOTED_CONTRACT_PATHS: Mapping[str, str] = {
    "anomaly-detection": "registry/suites/tiny/anomaly-detection.yaml",
    "causal-language-modeling": "registry/suites/language/causal-language-modeling.yaml",
    "graph-node-classification": "registry/suites/graph/graph-node-classification.yaml",
    "image-classification": "registry/suites/vision/image-classification.yaml",
    "information-retrieval": "registry/suites/language/information-retrieval.yaml",
    "keyword-spotting": "registry/suites/tiny/keyword-spotting.yaml",
    "text-classification": "registry/suites/language/text-classification.yaml",
    "time-series-forecasting": "registry/suites/timeseries/time-series-forecasting.yaml",
    "visual-wake-words": "registry/suites/tiny/visual-wake-words.yaml",
}


class SourceLockError(ValueError):
    """Raised when a source lock cannot be built or verified."""


def sha256_bytes(data: bytes) -> str:
    """Return a prefixed SHA-256 digest for *data*."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def validate_relative_path(value: object, *, label: str = "source path") -> str:
    """Return a canonical, safe POSIX path relative to the project root."""
    if not isinstance(value, str) or not value:
        raise SourceLockError(f"{label} must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise SourceLockError(f"{label} is not a safe POSIX relative path: {value!r}")
    if any(ord(character) < 32 for character in value):
        raise SourceLockError(f"{label} contains a control character")
    if re.match(r"^[A-Za-z]:", value):
        raise SourceLockError(f"{label} must not contain a drive prefix: {value!r}")

    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SourceLockError(f"{label} must remain relative to the project root")
    if relative.as_posix() != value or value in {".", ".."}:
        raise SourceLockError(f"{label} must use canonical POSIX spelling: {value!r}")
    return value


def normalize_paths(paths: Iterable[object]) -> tuple[str, ...]:
    """Validate, deduplicate, and sort measurement-surface paths."""
    normalized = [
        validate_relative_path(path, label="measurement source path") for path in paths
    ]
    if not normalized:
        raise SourceLockError("measurement source path list must not be empty")
    if len(set(normalized)) != len(normalized):
        raise SourceLockError("measurement source path list contains duplicates")
    return tuple(sorted(normalized))


def source_normalization(relative_path: str) -> dict[str, Any]:
    """Return the only permitted normalization for one protected source file."""
    functions = SOURCE_NORMALIZATIONS.get(relative_path)
    if functions is None:
        return dict(RAW_NORMALIZATION)
    return {
        "kind": PYTHON_FUNCTION_NORMALIZATION,
        "removed_top_level_functions": list(functions),
    }


def normalize_source_bytes(
    relative_path: str,
    data: bytes,
    *,
    normalization: Mapping[str, Any] | None = None,
) -> bytes:
    """Apply the declared, path-specific normalization to source bytes."""
    relative_path = validate_relative_path(relative_path)
    expected = source_normalization(relative_path)
    if normalization is not None and dict(normalization) != expected:
        raise SourceLockError(
            f"unsupported source normalization for {relative_path}: {normalization!r}"
        )
    if expected == RAW_NORMALIZATION:
        return data

    try:
        text = data.decode("utf-8")
        tree = ast.parse(text, filename=relative_path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SourceLockError(
            f"cannot parse protected Python source {relative_path}"
        ) from exc

    lines = data.splitlines(keepends=True)
    removals: list[tuple[int, int, str]] = []
    for function_name in expected["removed_top_level_functions"]:
        matches = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ]
        if len(matches) != 1:
            raise SourceLockError(
                f"{relative_path}: expected exactly one top-level function "
                f"{function_name!r}, found {len(matches)}"
            )
        node = matches[0]
        if node.end_lineno is None:
            raise SourceLockError(
                f"{relative_path}: cannot determine the end of {function_name!r}"
            )
        first_line = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        removals.append((first_line - 1, node.end_lineno, function_name))

    for (left_start, left_end, _), (right_start, _, _) in zip(
        sorted(removals), sorted(removals)[1:]
    ):
        if left_end > right_start:
            raise SourceLockError(
                f"{relative_path}: normalized top-level function ranges overlap"
            )

    for start, end, function_name in sorted(removals, reverse=True):
        marker = (
            f"# mlperf-edu source lock removed top-level function {function_name}\n"
        ).encode("utf-8")
        lines[start:end] = [marker]
    return b"".join(lines)


def contract_normalization() -> dict[str, Any]:
    """Return the explicit normalization for native registry contracts."""
    return {
        "kind": CONTRACT_PROJECTION_NORMALIZATION,
        "excluded_top_level_fields": list(CONTRACT_EXCLUDED_TOP_LEVEL_FIELDS),
        "excluded_quality_target_fields": list(CONTRACT_EXCLUDED_QUALITY_TARGET_FIELDS),
    }


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise SourceLockError(
                "registry contract contains an unhashable key"
            ) from exc
        if duplicate:
            raise SourceLockError(
                f"registry contract contains duplicate YAML key: {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def measurement_contract_bytes(
    data: bytes,
    *,
    workload_id: str,
    label: str = "registry contract",
) -> bytes:
    """Return canonical bytes for the protected portion of a native contract."""
    try:
        payload = yaml.load(data.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SourceLockError(f"cannot parse {label}") from exc
    if not isinstance(payload, dict):
        raise SourceLockError(f"{label} root must be an object")
    if payload.get("id") != workload_id:
        raise SourceLockError(
            f"{label} id={payload.get('id')!r}, expected {workload_id!r}"
        )

    projection = copy.deepcopy(payload)
    for field in CONTRACT_EXCLUDED_TOP_LEVEL_FIELDS:
        projection.pop(field, None)
    quality_target = projection.get("quality_target")
    if isinstance(quality_target, dict):
        for field in CONTRACT_EXCLUDED_QUALITY_TARGET_FIELDS:
            quality_target.pop(field, None)
    return canonical_json_bytes(projection)


def _project_file(project_root: Path, relative_path: str) -> Path:
    root = project_root.resolve()
    cursor = project_root
    for part in PurePosixPath(relative_path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SourceLockError(
                f"measurement source path may not traverse a symlink: {relative_path}"
            )

    resolved = cursor.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SourceLockError(
            f"measurement source path escapes the project root: {relative_path}"
        ) from exc
    if not resolved.is_file():
        raise SourceLockError(f"measurement source file is missing: {relative_path}")
    return resolved


def local_python_dependencies(
    relative_path: str, *, project_root: Path = PROJECT_ROOT
) -> set[str]:
    """Resolve direct local Python imports for one protected source file."""
    relative_path = validate_relative_path(relative_path)
    pure_path = PurePosixPath(relative_path)
    if len(pure_path.parts) < 3 or pure_path.parts[0] != "src":
        raise SourceLockError(
            f"local dependency discovery requires a source package path: {relative_path}"
        )
    module_parts = list(pure_path.with_suffix("").parts[1:])
    package_parts = module_parts[:-1]
    try:
        tree = ast.parse(
            _project_file(project_root, relative_path).read_bytes(),
            filename=relative_path,
        )
    except SyntaxError as exc:
        raise SourceLockError(
            f"cannot parse protected Python source {relative_path}"
        ) from exc

    def resolve_module(module_name: str) -> str | None:
        if not module_name:
            return None
        module_path = PurePosixPath("src", *module_name.split("."))
        candidates = (
            module_path.with_suffix(".py"),
            module_path / "__init__.py",
        )
        for candidate in candidates:
            text = candidate.as_posix()
            try:
                _project_file(project_root, text)
            except SourceLockError:
                continue
            return text
        return None

    dependencies: set[str] = set()
    for node in ast.walk(tree):
        module_names: list[str] = []
        if isinstance(node, ast.Import):
            module_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                retained = len(package_parts) - (node.level - 1)
                if retained < 0:
                    raise SourceLockError(
                        f"{relative_path}: relative import escapes its source package"
                    )
                prefix = package_parts[:retained]
                base = [*prefix, *(node.module or "").split(".")]
                module_name = ".".join(part for part in base if part)
            else:
                module_name = node.module or ""
            module_names.append(module_name)
            module_names.extend(
                ".".join(part for part in (module_name, alias.name) if part)
                for alias in node.names
            )
        for module_name in module_names:
            resolved = resolve_module(module_name)
            if resolved is not None and resolved != relative_path:
                dependencies.add(resolved)
    return dependencies


def _git_root(project_root: Path) -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceLockError(
            f"cannot locate Git repository for {project_root}"
        ) from exc
    git_root = Path(result.stdout.strip()).resolve()
    try:
        project_root.resolve().relative_to(git_root)
    except ValueError as exc:
        raise SourceLockError(
            f"project root is outside its reported Git repository: {project_root}"
        ) from exc
    return git_root


def _validate_source_commit(source_git_sha: object, *, git_root: Path) -> str:
    if not isinstance(source_git_sha, str) or not GIT_OBJECT_ID_RE.fullmatch(
        source_git_sha
    ):
        raise SourceLockError(
            "source_git_sha must be a complete lowercase Git object ID"
        )
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{source_git_sha}^{{commit}}"],
            cwd=git_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceLockError(
            f"source Git commit is unavailable: {source_git_sha}"
        ) from exc
    return source_git_sha


def _source_blob(
    source_git_sha: str,
    relative_path: str,
    *,
    git_root: Path,
    project_root: Path,
) -> bytes:
    project_prefix = project_root.resolve().relative_to(git_root)
    git_path = (PurePosixPath(project_prefix.as_posix()) / relative_path).as_posix()
    try:
        return subprocess.run(
            ["git", "show", f"{source_git_sha}:{git_path}"],
            cwd=git_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceLockError(
            f"cannot read {relative_path} from source commit {source_git_sha}"
        ) from exc


def build_source_lock(
    source_git_sha: str,
    *,
    project_root: Path = PROJECT_ROOT,
    paths: Iterable[object] = MEASUREMENT_SOURCE_PATHS,
    contracts: Mapping[str, str] = PROMOTED_CONTRACT_PATHS,
    verify_current: bool = True,
) -> dict[str, Any]:
    """Build a deterministic lock from an exact source commit.

    When ``verify_current`` is true, every source blob must be byte-identical
    to the corresponding file in the checkout.  This prevents an importer
    from silently promoting evidence against a changed measurement surface.
    """
    project_root = Path(project_root)
    git_root = _git_root(project_root)
    source_git_sha = _validate_source_commit(source_git_sha, git_root=git_root)
    normalized_paths = normalize_paths(paths)

    files: list[dict[str, Any]] = []
    for relative_path in normalized_paths:
        normalization = source_normalization(relative_path)
        source_data = normalize_source_bytes(
            relative_path,
            _source_blob(
                source_git_sha,
                relative_path,
                git_root=git_root,
                project_root=project_root,
            ),
            normalization=normalization,
        )
        if verify_current:
            current_data = normalize_source_bytes(
                relative_path,
                _project_file(project_root, relative_path).read_bytes(),
                normalization=normalization,
            )
            if current_data != source_data:
                raise SourceLockError(
                    f"current normalized checkout differs from source commit for "
                    f"{relative_path}: current {sha256_bytes(current_data)}, "
                    f"source {sha256_bytes(source_data)}"
                )
        files.append(
            {
                "path": relative_path,
                "normalization": normalization,
                "sha256": sha256_bytes(source_data),
                "n_bytes": len(source_data),
            }
        )

    contract_entries: list[dict[str, Any]] = []
    for workload_id, relative_path in sorted(contracts.items()):
        relative_path = validate_relative_path(
            relative_path, label=f"contract path for {workload_id}"
        )
        source_data = measurement_contract_bytes(
            _source_blob(
                source_git_sha,
                relative_path,
                git_root=git_root,
                project_root=project_root,
            ),
            workload_id=workload_id,
            label=f"source registry contract {relative_path}",
        )
        if verify_current:
            current_data = measurement_contract_bytes(
                _project_file(project_root, relative_path).read_bytes(),
                workload_id=workload_id,
                label=f"current registry contract {relative_path}",
            )
            if current_data != source_data:
                raise SourceLockError(
                    f"current measurement contract differs from source commit for "
                    f"{workload_id}: current {sha256_bytes(current_data)}, "
                    f"source {sha256_bytes(source_data)}"
                )
        contract_entries.append(
            {
                "workload": workload_id,
                "path": relative_path,
                "normalization": contract_normalization(),
                "sha256": sha256_bytes(source_data),
                "n_bytes": len(source_data),
            }
        )

    return {
        "schema": SOURCE_LOCK_SCHEMA,
        "source_git_sha": source_git_sha,
        "file_count": len(files),
        "files": files,
        "contract_count": len(contract_entries),
        "contracts": contract_entries,
    }


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a lock with stable ordering and a single trailing newline."""
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SourceLockError(f"source lock is not JSON-serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def verify_source_lock(
    payload: object,
    *,
    project_root: Path = PROJECT_ROOT,
    expected_source_git_sha: str | None = None,
    expected_paths: Iterable[object] | None = MEASUREMENT_SOURCE_PATHS,
    expected_contracts: Mapping[str, str] | None = PROMOTED_CONTRACT_PATHS,
    verify_current: bool = True,
) -> None:
    """Validate a parsed source lock and optionally verify checkout bytes."""
    if not isinstance(payload, dict):
        raise SourceLockError("source lock root must be an object")
    expected_top_level = {
        "schema",
        "source_git_sha",
        "file_count",
        "files",
        "contract_count",
        "contracts",
    }
    if set(payload) != expected_top_level:
        raise SourceLockError(
            "source lock fields must be exactly " + repr(sorted(expected_top_level))
        )
    if payload.get("schema") != SOURCE_LOCK_SCHEMA:
        raise SourceLockError(
            f"unsupported source lock schema: {payload.get('schema')!r}"
        )

    source_git_sha = payload.get("source_git_sha")
    if not isinstance(source_git_sha, str) or not GIT_OBJECT_ID_RE.fullmatch(
        source_git_sha
    ):
        raise SourceLockError(
            "source_git_sha must be a complete lowercase Git object ID"
        )
    if (
        expected_source_git_sha is not None
        and source_git_sha != expected_source_git_sha
    ):
        raise SourceLockError(
            f"source_git_sha={source_git_sha!r}, expected {expected_source_git_sha!r}"
        )

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise SourceLockError("source lock files must be a non-empty list")
    file_count = payload.get("file_count")
    if isinstance(file_count, bool) or file_count != len(files):
        raise SourceLockError(f"file_count={file_count!r}, expected {len(files)}")

    seen: list[str] = []
    for position, entry in enumerate(files):
        label = f"source lock files[{position}]"
        if not isinstance(entry, dict):
            raise SourceLockError(f"{label} must be an object")
        if set(entry) != {"path", "normalization", "sha256", "n_bytes"}:
            raise SourceLockError(
                f"{label} fields must be exactly "
                "['n_bytes', 'normalization', 'path', 'sha256']"
            )
        relative_path = validate_relative_path(entry.get("path"), label=f"{label}.path")
        seen.append(relative_path)

        normalization = entry.get("normalization")
        if normalization != source_normalization(relative_path):
            raise SourceLockError(
                f"{label}.normalization is not permitted for {relative_path}"
            )

        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise SourceLockError(f"{label}.sha256 is not a complete SHA-256 digest")
        n_bytes = entry.get("n_bytes")
        if isinstance(n_bytes, bool) or not isinstance(n_bytes, int) or n_bytes < 0:
            raise SourceLockError(f"{label}.n_bytes must be a nonnegative integer")

        if verify_current:
            current_data = normalize_source_bytes(
                relative_path,
                _project_file(project_root, relative_path).read_bytes(),
                normalization=normalization,
            )
            actual_digest = sha256_bytes(current_data)
            if len(current_data) != n_bytes:
                raise SourceLockError(
                    f"{relative_path}: n_bytes={n_bytes}, recomputed {len(current_data)}"
                )
            if actual_digest != digest:
                raise SourceLockError(
                    f"{relative_path}: sha256={digest}, recomputed {actual_digest}"
                )

    if len(set(seen)) != len(seen):
        raise SourceLockError("source lock contains duplicate paths")
    if seen != sorted(seen):
        raise SourceLockError("source lock files must be sorted by path")
    if expected_paths is not None:
        expected = list(normalize_paths(expected_paths))
        if seen != expected:
            missing = sorted(set(expected) - set(seen))
            extra = sorted(set(seen) - set(expected))
            raise SourceLockError(
                f"source lock measurement surface mismatch; missing={missing}, extra={extra}"
            )

    contracts = payload.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise SourceLockError("source lock contracts must be a non-empty list")
    contract_count = payload.get("contract_count")
    if isinstance(contract_count, bool) or contract_count != len(contracts):
        raise SourceLockError(
            f"contract_count={contract_count!r}, expected {len(contracts)}"
        )

    seen_contracts: dict[str, str] = {}
    previous_workload: str | None = None
    expected_contract_normalization = contract_normalization()
    for position, entry in enumerate(contracts):
        label = f"source lock contracts[{position}]"
        if not isinstance(entry, dict):
            raise SourceLockError(f"{label} must be an object")
        if set(entry) != {
            "workload",
            "path",
            "normalization",
            "sha256",
            "n_bytes",
        }:
            raise SourceLockError(
                f"{label} fields must be exactly "
                "['n_bytes', 'normalization', 'path', 'sha256', 'workload']"
            )
        workload_id = entry.get("workload")
        if not isinstance(workload_id, str) or not workload_id:
            raise SourceLockError(f"{label}.workload must be a non-empty string")
        if workload_id in seen_contracts:
            raise SourceLockError(
                f"source lock contains duplicate contract workload: {workload_id}"
            )
        if previous_workload is not None and workload_id < previous_workload:
            raise SourceLockError("source lock contracts must be sorted by workload")
        previous_workload = workload_id

        relative_path = validate_relative_path(entry.get("path"), label=f"{label}.path")
        if relative_path in seen_contracts.values():
            raise SourceLockError(
                f"source lock contains duplicate contract path: {relative_path}"
            )
        seen_contracts[workload_id] = relative_path
        if entry.get("normalization") != expected_contract_normalization:
            raise SourceLockError(f"{label}.normalization is not permitted")

        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise SourceLockError(f"{label}.sha256 is not a complete SHA-256 digest")
        n_bytes = entry.get("n_bytes")
        if isinstance(n_bytes, bool) or not isinstance(n_bytes, int) or n_bytes < 0:
            raise SourceLockError(f"{label}.n_bytes must be a nonnegative integer")

        if verify_current:
            current_data = measurement_contract_bytes(
                _project_file(project_root, relative_path).read_bytes(),
                workload_id=workload_id,
                label=f"current registry contract {relative_path}",
            )
            actual_digest = sha256_bytes(current_data)
            if len(current_data) != n_bytes:
                raise SourceLockError(
                    f"{workload_id}: contract n_bytes={n_bytes}, "
                    f"recomputed {len(current_data)}"
                )
            if actual_digest != digest:
                raise SourceLockError(
                    f"{workload_id}: contract sha256={digest}, "
                    f"recomputed {actual_digest}"
                )

    if expected_contracts is not None:
        expected_contract_map = {
            workload_id: validate_relative_path(
                path, label=f"expected contract path for {workload_id}"
            )
            for workload_id, path in expected_contracts.items()
        }
        if seen_contracts != expected_contract_map:
            missing = sorted(set(expected_contract_map) - set(seen_contracts))
            extra = sorted(set(seen_contracts) - set(expected_contract_map))
            wrong_paths = sorted(
                workload_id
                for workload_id in set(seen_contracts).intersection(
                    expected_contract_map
                )
                if seen_contracts[workload_id] != expected_contract_map[workload_id]
            )
            raise SourceLockError(
                "source lock contract surface mismatch; "
                f"missing={missing}, extra={extra}, wrong_paths={wrong_paths}"
            )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceLockError(f"source lock contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_source_lock(
    path: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    expected_source_git_sha: str | None = None,
    expected_paths: Iterable[object] | None = MEASUREMENT_SOURCE_PATHS,
    expected_contracts: Mapping[str, str] | None = PROMOTED_CONTRACT_PATHS,
    verify_current: bool = True,
    require_canonical: bool = True,
) -> dict[str, Any]:
    """Load and verify a source-lock JSON file without consulting Git history."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SourceLockError(f"source lock is missing or is a symlink: {path}")
    data = path.read_bytes()
    try:
        payload = json.loads(
            data.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceLockError(f"invalid source-lock JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceLockError("source lock root must be an object")
    if require_canonical and data != canonical_json_bytes(payload):
        raise SourceLockError(f"source lock is not canonically serialized: {path}")
    verify_source_lock(
        payload,
        project_root=project_root,
        expected_source_git_sha=expected_source_git_sha,
        expected_paths=expected_paths,
        expected_contracts=expected_contracts,
        verify_current=verify_current,
    )
    return payload
