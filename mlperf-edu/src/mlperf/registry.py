from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import PackageNotFoundError, files
from pathlib import Path
from typing import Any

import yaml

from .assets import has_asset_dossier


PROFILES = ("min", "max", "pro")

PUBLIC_STATUSES = (
    "score-bearing",
    "performance-bearing",
    "systems-only",
    "experimental",
)

PUBLIC_RELEASE_STATUSES = (
    "score-bearing",
    "performance-bearing",
    "systems-only",
)

ADAPTER_CONFORMANCE_STATUSES = (
    "exact",
    "quality-preserving-nonidentical",
)

QUALITY_TARGET_BASES = (
    "reference_runs",
    "literature",
    "mlcommons_derived",
    "pedagogical_baseline",
)

QUALITY_TARGET_KINDS = (
    "inherited_acceptance_gate",
    "published_reference_reproduction",
    "published_mean_with_tolerance",
    "policy_derived_gate",
)

REFERENCE_PROTOCOL_FIELDS = (
    "profile",
    "backend",
    "machine_class",
    "dataset_mode",
    "seeds",
    "aggregation",
    "artifact_policy",
    "rerun_policy",
)

EDU_SCENARIOS = (
    "single_stream",
    "offline",
    "server",
    "training",
    "inference",
)

PUBLIC_RESULT_SCENARIOS = (
    "single_stream",
    "offline",
    "server",
    "training",
)

PRODUCT_SUITES = (
    "language",
    "vision",
    "tiny",
    "graph",
    "timeseries",
    "recommendation",
    "reinforcement",
)

WORKLOAD_COLLECTIONS = (
    "starter",
    "standard",
    "research",
    "all",
)

DEFAULT_WORKLOAD_COLLECTION = "starter"

STARTER_WORKLOADS = {
    "causal-language-modeling",
    "image-classification",
    "graph-node-classification",
    "time-series-forecasting",
}

STARTER_WORKLOAD_ORDER = (
    "causal-language-modeling",
    "image-classification",
    "graph-node-classification",
    "time-series-forecasting",
)

STANDARD_WORKLOAD_ORDER = (
    *STARTER_WORKLOAD_ORDER,
    "keyword-spotting",
    "anomaly-detection",
    "visual-wake-words",
)

STANDARD_WORKLOADS = set(STANDARD_WORKLOAD_ORDER)

# The pro profile is the research envelope over the SAME workload identity, and
# it resolves to each workload's max runner. Every registered workload therefore
# has a pro path; excluding some left five workloads with no research profile at
# all, which contradicted the documented min/max/pro contract.
RESEARCH_WORKLOAD_ORDER = (
    "image-classification",
    "keyword-spotting",
    "anomaly-detection",
    "visual-wake-words",
    "causal-language-modeling",
    "text-classification",
    "information-retrieval",
    "graph-node-classification",
    "time-series-forecasting",
    "code-generation",
    "function-calling",
    "recommendation",
    "image-generation",
    "reinforcement-learning",
)

RESEARCH_WORKLOADS = set(RESEARCH_WORKLOAD_ORDER)


@dataclass(frozen=True)
class Workload:
    id: str
    suite: str
    source_suite: str
    maturity: str
    model: str
    dataset: str | None
    canonical_workload: str | None
    variant: str | None
    default_variant: bool
    scenario: str | None
    quality_metric: str | None
    quality_value: Any
    quality_direction: str | None
    quality_target_basis: str | None
    quality_target_kind: str | None
    quality_tolerance: Any
    quality_reference_runs: Any
    quality_acceptance_runs: Any
    quality_variance_summary: Any
    quality_reference_protocol: Any
    quality_reviewer_notes: tuple[str, ...]
    public_status: str
    public_rationale: str
    raw: dict[str, Any]

    @property
    def supports_profiles(self) -> tuple[str, ...]:
        return PROFILES


def find_project_root(start: Path | None = None) -> Path:
    """Find the mlperf-edu project root containing workloads.yaml."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve().parents[2])

    for base in candidates:
        for path in (base, *base.parents):
            if (path / "workloads.yaml").is_file():
                return path

    try:
        packaged_registry_path()
    except FileNotFoundError:
        pass
    else:
        return Path.cwd().resolve()

    raise FileNotFoundError("Could not find workloads.yaml from current directory")


def default_registry_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    try:
        project_root = find_project_root()
    except FileNotFoundError:
        return packaged_registry_path()
    native_registry_path = project_root / "registry"
    if (native_registry_path / "suites").is_dir():
        return native_registry_path
    registry_path = project_root / "workloads.yaml"
    if registry_path.is_file():
        return registry_path
    return packaged_registry_path()


def packaged_registry_path() -> Path:
    try:
        resource_path = resources.files("mlperf_edu").joinpath("workloads.yaml")
    except (ModuleNotFoundError, FileNotFoundError):
        resource_path = None
    if resource_path is not None and resource_path.is_file():
        return Path(str(resource_path))

    try:
        dist_files = files("mlperf_edu") or ()
    except PackageNotFoundError as exc:
        raise FileNotFoundError("Could not find packaged workloads.yaml") from exc

    for item in dist_files:
        if str(item).replace("\\", "/").endswith("workloads.yaml"):
            path = Path(str(item.locate()))
            if path.is_file():
                return path
    raise FileNotFoundError("Could not find packaged workloads.yaml")


def load_registry(
    path: str | Path | None = None, *, validate: bool = True
) -> dict[str, Workload]:
    """Load registry YAML and normalize it into flat Workload records.

    The current release path still accepts the legacy flat `workloads.yaml`.
    New registry experiments can also pass a directory with a native layout:

    suites/<suite>/<workload>.yaml
    suites/<suite>/<workload>/workload.yaml
    suites/<suite>/<workload>/variants/<variant>.yaml
    """
    registry_path = default_registry_path(path)
    if registry_path.is_dir():
        data = load_registry_directory(registry_path)
    else:
        with registry_path.open("r") as f:
            data = yaml.safe_load(f)

    return normalize_registry_data(
        data, registry_name=str(registry_path), validate=validate
    )


def normalize_registry_data(
    data: dict[str, Any],
    *,
    registry_name: str = "workloads.yaml",
    validate: bool = True,
) -> dict[str, Workload]:
    suites = data.get("suites")
    if not isinstance(suites, dict):
        raise ValueError(f"{registry_name} must contain a top-level 'suites' mapping")

    workloads: dict[str, Workload] = {}
    for source_suite, entries in suites.items():
        if not isinstance(entries, dict):
            raise ValueError(f"suite '{source_suite}' must be a mapping")

        for workload_id, raw in entries.items():
            if workload_id in workloads:
                raise ValueError(f"duplicate workload id: {workload_id}")
            if not isinstance(raw, dict):
                raise ValueError(f"workload '{workload_id}' must be a mapping")

            quality = raw.get("quality") or raw.get("quality_target") or {}
            public = raw.get("public") or {}
            suite = source_suite
            maturity = "base" if workload_id in STANDARD_WORKLOADS else "research"
            reviewer_notes = quality.get("reviewer_notes") or []
            if isinstance(reviewer_notes, str):
                reviewer_notes = [reviewer_notes]
            workloads[workload_id] = Workload(
                id=workload_id,
                suite=suite,
                source_suite=source_suite,
                maturity=maturity,
                model=str(raw.get("model", "unknown")),
                dataset=raw.get("dataset"),
                canonical_workload=raw.get("canonical_workload"),
                variant=raw.get("variant"),
                default_variant=bool(raw.get("default_variant", False)),
                scenario=raw.get("scenario"),
                quality_metric=quality.get("metric"),
                quality_value=quality.get("target", quality.get("value")),
                quality_direction=quality.get("direction"),
                quality_target_basis=quality.get("target_basis"),
                quality_target_kind=quality.get("target_kind"),
                quality_tolerance=quality.get("tolerance"),
                quality_reference_runs=quality.get("reference_runs"),
                quality_acceptance_runs=quality.get("acceptance_runs"),
                quality_variance_summary=quality.get("variance_summary"),
                quality_reference_protocol=quality.get("reference_protocol"),
                quality_reviewer_notes=tuple(str(note) for note in reviewer_notes),
                public_status=str(public.get("status", "")),
                public_rationale=str(public.get("rationale", "")),
                raw=raw,
            )

    if validate:
        validate_registry(workloads)
    return workloads


def load_registry_directory(registry_dir: Path) -> dict[str, Any]:
    """Load the native suite/workload/variant registry directory layout."""
    if (registry_dir / "workloads.yaml").is_file() and not (
        registry_dir / "suites"
    ).is_dir():
        with (registry_dir / "workloads.yaml").open("r") as f:
            return yaml.safe_load(f)

    suites_dir = registry_dir / "suites"
    if not suites_dir.is_dir():
        raise ValueError(
            f"registry directory '{registry_dir}' must contain suites/ or workloads.yaml"
        )

    suites: dict[str, dict[str, Any]] = {}
    for suite_dir in sorted(path for path in suites_dir.iterdir() if path.is_dir()):
        suite_entries: dict[str, Any] = {}
        for item in sorted(suite_dir.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_file() and item.suffix in {".yaml", ".yml"}:
                raw = read_yaml_mapping(item)
                workload_id = str(raw.get("id") or item.stem)
                suite_entries[workload_id] = strip_registry_identity(raw)
            elif item.is_dir():
                suite_entries.update(load_native_workload_directory(item))
        suites[suite_dir.name] = suite_entries
    return {"suites": suites}


def load_native_workload_directory(workload_dir: Path) -> dict[str, Any]:
    base_path = workload_dir / "workload.yaml"
    base = read_yaml_mapping(base_path) if base_path.is_file() else {}
    canonical_workload = str(base.get("id") or workload_dir.name)
    variants_dir = workload_dir / "variants"

    if not variants_dir.is_dir():
        raw = strip_registry_identity(base)
        return {str(base.get("id") or workload_dir.name): raw}

    entries: dict[str, Any] = {}
    shared = strip_registry_identity(base)
    variant_order = (
        [str(item) for item in base.get("variant_order", [])]
        if isinstance(base.get("variant_order"), list)
        else []
    )
    variant_paths = sorted(
        variants_dir.glob("*.y*ml"),
        key=lambda path: variant_sort_key(path, variant_order),
    )
    for variant_path in variant_paths:
        variant = read_yaml_mapping(variant_path)
        variant_name = str(variant.get("variant") or variant_path.stem)
        workload_id = str(variant.get("id") or f"{canonical_workload}-{variant_name}")
        raw = merge_registry_dicts(shared, strip_registry_identity(variant))
        raw.setdefault("canonical_workload", canonical_workload)
        raw.setdefault("variant", variant_name)
        entries[workload_id] = raw
    return entries


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"registry YAML file '{path}' must contain a mapping")
    return data


def variant_sort_key(path: Path, variant_order: list[str]) -> tuple[int, str]:
    try:
        return (variant_order.index(path.stem), path.stem)
    except ValueError:
        return (len(variant_order), path.stem)


def strip_registry_identity(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if key not in {"id", "variants", "variant_order"}
    }


def merge_registry_dicts(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_registry_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_registry(workloads: dict[str, Workload]) -> None:
    if not workloads:
        raise ValueError("registry contains no workloads")

    missing_starter = sorted(STARTER_WORKLOADS.difference(workloads))
    if missing_starter:
        raise ValueError(
            f"starter workload ids missing from workloads.yaml: {missing_starter}"
        )

    missing_standard = sorted(STANDARD_WORKLOADS.difference(workloads))
    if missing_standard:
        raise ValueError(
            f"standard workload ids missing from workloads.yaml: {missing_standard}"
        )

    missing_research = sorted(RESEARCH_WORKLOADS.difference(workloads))
    if missing_research:
        raise ValueError(
            f"research workload ids missing from workloads.yaml: {missing_research}"
        )

    invalid_suites = sorted(
        {w.suite for w in workloads.values() if w.suite not in PRODUCT_SUITES}
    )
    if invalid_suites:
        raise ValueError(f"invalid product suites: {invalid_suites}")

    invalid_scenarios = sorted(
        {
            str(w.scenario)
            for w in workloads.values()
            if not w.scenario or w.scenario not in EDU_SCENARIOS
        }
    )
    if invalid_scenarios:
        raise ValueError(f"invalid EDU scenarios: {invalid_scenarios}")

    invalid_public_statuses = sorted(
        {
            w.public_status
            for w in workloads.values()
            if w.public_status not in PUBLIC_STATUSES
        }
    )
    if invalid_public_statuses:
        raise ValueError(f"invalid public statuses: {invalid_public_statuses}")

    invalid_quality_acceptance = sorted(
        workload.id
        for workload in workloads.values()
        if workload.quality_acceptance_runs != 1
    )
    if invalid_quality_acceptance:
        raise ValueError(
            "quality_target.acceptance_runs must be 1 for the current quality "
            f"milestone: {invalid_quality_acceptance}"
        )

    invalid_quality_target_kinds = sorted(
        workload.id
        for workload in workloads.values()
        if workload.quality_target_kind not in QUALITY_TARGET_KINDS
    )
    if invalid_quality_target_kinds:
        raise ValueError(
            "quality_target.target_kind must distinguish inherited gates, "
            "published reproduction targets, published means with tolerance, "
            f"and policy-derived gates: {invalid_quality_target_kinds}"
        )

    canonical_metadata_issues: list[str] = []
    variants_by_canonical: dict[str, set[str]] = {}
    default_variants_by_canonical: dict[str, list[str]] = {}
    for workload in workloads.values():
        if workload.variant and not workload.canonical_workload:
            canonical_metadata_issues.append(
                f"{workload.id}:variant without canonical_workload"
            )
        if workload.default_variant and not workload.canonical_workload:
            canonical_metadata_issues.append(
                f"{workload.id}:default_variant without canonical_workload"
            )
        if workload.canonical_workload and not workload.variant:
            canonical_metadata_issues.append(
                f"{workload.id}:canonical_workload without variant"
            )
        if workload.canonical_workload:
            seen = variants_by_canonical.setdefault(workload.canonical_workload, set())
            if workload.variant in seen:
                canonical_metadata_issues.append(
                    f"{workload.id}:duplicate variant {workload.variant}"
                )
            seen.add(str(workload.variant))
            if workload.default_variant:
                default_variants_by_canonical.setdefault(
                    workload.canonical_workload, []
                ).append(workload.id)
    duplicate_defaults = {
        canonical: ids
        for canonical, ids in default_variants_by_canonical.items()
        if len(ids) > 1
    }
    if duplicate_defaults:
        canonical_metadata_issues.extend(
            f"{canonical}:multiple default variants {ids}"
            for canonical, ids in sorted(duplicate_defaults.items())
        )
    if canonical_metadata_issues:
        raise ValueError(
            f"invalid canonical workload metadata: {canonical_metadata_issues}"
        )

    missing_public_rationale = sorted(
        w.id for w in workloads.values() if not w.public_rationale.strip()
    )
    if missing_public_rationale:
        raise ValueError(
            f"public rationale missing for workloads: {missing_public_rationale}"
        )

    missing_runner_profiles: list[str] = []
    invalid_runner_specs: list[str] = []
    for workload in workloads.values():
        runner = workload.raw.get("runner") or {}
        for profile in ("min", "max"):
            spec = runner.get(profile)
            if not spec:
                missing_runner_profiles.append(f"{workload.id}:{profile}")
            elif ":" not in str(spec):
                invalid_runner_specs.append(f"{workload.id}:{profile}={spec}")
    if missing_runner_profiles:
        raise ValueError(
            f"runner profiles missing from workloads.yaml: {missing_runner_profiles}"
        )
    if invalid_runner_specs:
        raise ValueError(
            f"invalid runner specs in workloads.yaml: {invalid_runner_specs}"
        )

    max_execution_issues: list[str] = []
    for workload in workloads.values():
        execution = workload.raw.get("max_execution")
        if workload.public_status == "systems-only" and not isinstance(execution, dict):
            max_execution_issues.append(
                f"{workload.id}: systems-only max runner must declare max_execution"
            )
            continue
        if not isinstance(execution, dict):
            continue
        if workload.public_status != "systems-only":
            max_execution_issues.append(
                f"{workload.id}: max_execution boundary is allowed only for systems-only workloads"
            )
        if execution.get("scope") != "systems-only":
            max_execution_issues.append(
                f"{workload.id}: max_execution.scope must be systems-only"
            )
        if not str(execution.get("data_mode") or "").strip():
            max_execution_issues.append(
                f"{workload.id}: max_execution.data_mode is required"
            )
        if not isinstance(execution.get("quality_target_enforced"), bool):
            max_execution_issues.append(
                f"{workload.id}: max_execution.quality_target_enforced must be boolean"
            )
        for field in ("fetched_assets_used", "declared_dataset_used"):
            if not isinstance(execution.get(field), bool):
                max_execution_issues.append(
                    f"{workload.id}: max_execution.{field} must be boolean"
                )
        if not str(execution.get("note") or "").strip():
            max_execution_issues.append(
                f"{workload.id}: max_execution.note is required"
            )
    if max_execution_issues:
        raise ValueError(f"invalid max execution boundaries: {max_execution_issues}")

    adapter_conformance_issues: list[str] = []
    for workload in workloads.values():
        conformance = workload.raw.get("adapter_conformance")
        if conformance is None:
            continue
        if not isinstance(conformance, dict):
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance must be a mapping"
            )
            continue
        status = conformance.get("status")
        if status not in ADAPTER_CONFORMANCE_STATUSES:
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance.status must be one of "
                f"{list(ADAPTER_CONFORMANCE_STATUSES)}"
            )
        promotion_eligible = conformance.get("promotion_eligible")
        if not isinstance(promotion_eligible, bool):
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance.promotion_eligible must be boolean"
            )
        if status == "quality-preserving-nonidentical":
            if promotion_eligible is not False:
                adapter_conformance_issues.append(
                    f"{workload.id}: a nonidentical adapter cannot be promotion eligible"
                )
            if workload.public_status != "experimental":
                adapter_conformance_issues.append(
                    f"{workload.id}: a nonidentical adapter must remain experimental"
                )
        if not str(conformance.get("policy") or "").strip():
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance.policy is required"
            )
        if not str(conformance.get("resolution") or "").strip():
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance.resolution is required"
            )
        audit = conformance.get("audit")
        if not isinstance(audit, dict):
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance.audit must be a mapping"
            )
            continue
        for field in ("artifact", "sha256", "schema", "source_git_sha", "runtime"):
            if not str(audit.get(field) or "").strip():
                adapter_conformance_issues.append(
                    f"{workload.id}: adapter_conformance.audit.{field} is required"
                )
        resolvers = audit.get("resolvers")
        if (
            not isinstance(resolvers, list)
            or not resolvers
            or not all(isinstance(resolver, str) and resolver for resolver in resolvers)
        ):
            adapter_conformance_issues.append(
                f"{workload.id}: adapter_conformance.audit.resolvers must be a "
                "nonempty string list"
            )
    if adapter_conformance_issues:
        raise ValueError(
            f"invalid adapter conformance metadata: {adapter_conformance_issues}"
        )

    execution_role_issues: list[str] = []
    result_roles = {"score-bearing", "performance-bearing"}
    for workload in workloads.values():
        canonical = workload.raw.get("canonical_max_contract") or {}
        if canonical.get("result_role") not in result_roles:
            execution_role_issues.append(
                f"{workload.id}: canonical_max_contract.result_role must be one of "
                f"{sorted(result_roles)}"
            )
        if canonical.get("mode") not in set(
            workload.raw.get("implemented_modes") or []
        ):
            execution_role_issues.append(
                f"{workload.id}: canonical_max_contract.mode must name an "
                "implemented mode"
            )
        mode_contracts = workload.raw.get("mode_contracts") or {}
        inference = mode_contracts.get("inference") or {}
        phases = inference.get("phases") or {}
        for phase, contract in phases.items():
            if (
                not isinstance(contract, dict)
                or contract.get("result_role") not in result_roles
            ):
                execution_role_issues.append(
                    f"{workload.id}: inference phase {phase!r} must declare a valid "
                    "result_role"
                )
    if execution_role_issues:
        raise ValueError(f"invalid execution result roles: {execution_role_issues}")

    canonical_max_issues: list[str] = []
    for workload in workloads.values():
        if workload.public_status not in {"score-bearing", "performance-bearing"}:
            continue
        contract = workload.raw.get("canonical_max_contract")
        if not isinstance(contract, dict):
            canonical_max_issues.append(
                f"{workload.id}: public candidate must declare canonical_max_contract"
            )
            continue
        for field in ("model_id", "dataset", "data_mode"):
            if not str(contract.get(field) or "").strip():
                canonical_max_issues.append(
                    f"{workload.id}: canonical_max_contract.{field} is required"
                )
        if contract.get("dataset") != workload.dataset:
            canonical_max_issues.append(
                f"{workload.id}: canonical_max_contract.dataset must match registry dataset"
            )
        model_source = workload.raw.get("model_source")
        if isinstance(model_source, dict):
            if contract.get("model_revision") != model_source.get("revision"):
                canonical_max_issues.append(
                    f"{workload.id}: canonical max model revision must match model_source.revision"
                )
            declared_quality = workload.raw.get("quality_evaluation") or {}
            if declared_quality:
                canonical_evaluation = contract.get("quality_evaluation")
                if not isinstance(canonical_evaluation, dict):
                    canonical_max_issues.append(
                        f"{workload.id}: declared quality evaluation must be present "
                        "in canonical_max_contract"
                    )
                    continue
                expected_suite_sha256 = declared_quality.get("asset_sha256")
                if expected_suite_sha256:
                    expected_suite_sha256 = f"sha256:{expected_suite_sha256}"
                expected_quality_fields = {
                    "suite": declared_quality.get("suite"),
                    "fixture_version": declared_quality.get("fixture_version"),
                    "suite_sha256": expected_suite_sha256,
                    "cases": declared_quality.get("cases"),
                    "categories": declared_quality.get("categories"),
                    "aggregation": declared_quality.get("aggregation"),
                    "category_guard": declared_quality.get("category_guard"),
                }
                for field, expected in expected_quality_fields.items():
                    if (
                        expected is not None
                        and canonical_evaluation.get(field) != expected
                    ):
                        canonical_max_issues.append(
                            f"{workload.id}: canonical quality evaluation {field} "
                            "must match the declared quality fixture"
                        )
                expected_quality_gates = {
                    "overall_perplexity": {
                        "metric_key": "perplexity",
                        "target": declared_quality.get("maximum"),
                        "direction": "lower",
                    },
                    "worst_category_perplexity": {
                        "metric_key": "worst_category_perplexity",
                        "target": declared_quality.get("worst_category_maximum"),
                        "direction": "lower",
                    },
                }
                if canonical_evaluation.get("gates") != expected_quality_gates:
                    canonical_max_issues.append(
                        f"{workload.id}: canonical quality evaluation gates must "
                        "match the declared overall and worst-category limits"
                    )
        config = contract.get("config")
        if not isinstance(config, dict) or not config:
            canonical_max_issues.append(
                f"{workload.id}: canonical_max_contract.config must be a nonempty mapping"
            )
        quality = contract.get("quality")
        if not isinstance(quality, dict):
            canonical_max_issues.append(
                f"{workload.id}: canonical_max_contract.quality must be a mapping"
            )
        else:
            for field in ("metric", "metric_key", "target", "direction"):
                if field not in quality or quality[field] is None:
                    canonical_max_issues.append(
                        f"{workload.id}: canonical_max_contract.quality.{field} is required"
                    )
            expected_metric = (
                (workload.raw.get("functional_check") or {}).get("metric")
                if workload.public_status == "performance-bearing"
                else workload.quality_metric
            )
            if quality.get("metric") != expected_metric:
                canonical_max_issues.append(
                    f"{workload.id}: canonical max quality metric must match the public contract"
                )
            if workload.public_status == "score-bearing":
                if quality.get("target") != workload.quality_value:
                    canonical_max_issues.append(
                        f"{workload.id}: canonical max quality target must match quality_target.value"
                    )
                if quality.get("direction") != workload.quality_direction:
                    canonical_max_issues.append(
                        f"{workload.id}: canonical max quality direction must match quality_target.direction"
                    )
        quality_gates = contract.get("quality_gates")
        if quality_gates is not None:
            if not isinstance(quality_gates, dict) or not quality_gates:
                canonical_max_issues.append(
                    f"{workload.id}: canonical_max_contract.quality_gates must be a nonempty mapping"
                )
            else:
                for gate_name, gate in quality_gates.items():
                    if not isinstance(gate, dict):
                        canonical_max_issues.append(
                            f"{workload.id}: canonical quality gate {gate_name} must be a mapping"
                        )
                        continue
                    for field in ("metric_key", "target", "direction"):
                        if field not in gate or gate[field] is None:
                            canonical_max_issues.append(
                                f"{workload.id}: canonical quality gate {gate_name}.{field} is required"
                            )
        measurement = workload.raw.get("measurement_protocol")
        if not isinstance(measurement, dict) or not measurement:
            canonical_max_issues.append(
                f"{workload.id}: canonical public max requires measurement_protocol"
            )
        elif not str(measurement.get("primary_metric") or "").strip():
            canonical_max_issues.append(
                f"{workload.id}: measurement_protocol.primary_metric is required"
            )
        if isinstance(measurement, dict):
            if measurement.get("outer_reference_runs") != 5:
                canonical_max_issues.append(
                    f"{workload.id}: public measurement requires five outer reference runs"
                )
            preconditioning_runs = measurement.get("outer_preconditioning_runs")
            if (
                isinstance(preconditioning_runs, bool)
                or not isinstance(preconditioning_runs, int)
                or preconditioning_runs < 0
            ):
                canonical_max_issues.append(
                    f"{workload.id}: measurement_protocol.outer_preconditioning_runs "
                    "must be a nonnegative integer"
                )
            cooldown_seconds = measurement.get("outer_inter_execution_cooldown_seconds")
            if (
                isinstance(cooldown_seconds, bool)
                or not isinstance(cooldown_seconds, (int, float))
                or not math.isfinite(float(cooldown_seconds))
                or not 0 <= float(cooldown_seconds) <= 300
            ):
                canonical_max_issues.append(
                    f"{workload.id}: measurement_protocol."
                    "outer_inter_execution_cooldown_seconds must be finite and "
                    "between 0 and 300"
                )
        if workload.public_status == "performance-bearing":
            timing_counts = contract.get("timing_sample_counts")
            if not isinstance(timing_counts, dict) or not timing_counts:
                canonical_max_issues.append(
                    f"{workload.id}: canonical performance max requires timing_sample_counts"
                )
            elif any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in timing_counts.values()
            ):
                canonical_max_issues.append(
                    f"{workload.id}: canonical timing sample counts must be positive integers"
                )
        for mode_name, mode_contract in (
            workload.raw.get("mode_contracts") or {}
        ).items():
            if not isinstance(mode_contract, dict):
                continue
            for phase_name, phase_contract in (
                mode_contract.get("phases") or {}
            ).items():
                if not isinstance(phase_contract, dict):
                    continue
                phase_measurement = phase_contract.get("measurement_protocol")
                if not isinstance(phase_measurement, dict):
                    continue
                label = f"{workload.id}/{mode_name}/{phase_name}"
                if phase_measurement.get("outer_reference_runs") != 5:
                    canonical_max_issues.append(
                        f"{label}: public measurement requires five outer reference runs"
                    )
                phase_preconditioning = phase_measurement.get(
                    "outer_preconditioning_runs"
                )
                if (
                    isinstance(phase_preconditioning, bool)
                    or not isinstance(phase_preconditioning, int)
                    or phase_preconditioning < 0
                ):
                    canonical_max_issues.append(
                        f"{label}: outer_preconditioning_runs must be a nonnegative integer"
                    )
                phase_cooldown = phase_measurement.get(
                    "outer_inter_execution_cooldown_seconds"
                )
                if (
                    isinstance(phase_cooldown, bool)
                    or not isinstance(phase_cooldown, (int, float))
                    or not math.isfinite(float(phase_cooldown))
                    or not 0 <= float(phase_cooldown) <= 300
                ):
                    canonical_max_issues.append(
                        f"{label}: outer inter-execution cooldown must be finite "
                        "and between 0 and 300"
                    )
    if canonical_max_issues:
        raise ValueError(f"invalid canonical max contracts: {canonical_max_issues}")


def baseline_is_protocol_superseded(value: object) -> bool:
    """Return whether a baseline is retained only as protocol-superseded history."""
    return (
        isinstance(value, dict) and value.get("protocol_compatibility") == "superseded"
    )


def baseline_is_current_review_evidence(value: object) -> bool:
    """Return whether a baseline may support the workload's current contract."""
    return (
        isinstance(value, dict)
        and value.get("evidence_status") == "committed-reference-summary"
        and value.get("review_eligible") is True
        and not baseline_is_protocol_superseded(value)
        and value.get("replacement_required") is not True
    )


def baseline_lifecycle_issues(
    baseline: dict[str, Any], *, public_status: str
) -> list[str]:
    """Validate the fail-closed current-versus-historical evidence state."""
    issues: list[str] = []
    compatibility = baseline.get("protocol_compatibility")
    replacement_required = baseline.get("replacement_required")
    if compatibility not in (None, "current", "superseded"):
        issues.append(
            f"{public_status} verified_baseline has unsupported "
            f"protocol_compatibility {compatibility!r}"
        )

    if baseline_is_protocol_superseded(baseline):
        if baseline.get("review_eligible") is not False:
            issues.append(
                f"{public_status} protocol-superseded verified_baseline must set "
                "review_eligible to false"
            )
        if replacement_required is not True:
            issues.append(
                f"{public_status} protocol-superseded verified_baseline must set "
                "replacement_required to true"
            )
        if not str(baseline.get("superseded_reason") or "").strip():
            issues.append(
                f"{public_status} protocol-superseded verified_baseline must explain "
                "superseded_reason"
            )
        issues.append(
            f"{public_status} verified_baseline uses a superseded protocol and "
            "requires a replacement reference sweep"
        )
        return issues

    if replacement_required is True:
        issues.append(
            f"{public_status} verified_baseline may require replacement only when "
            "protocol_compatibility is superseded"
        )
    elif replacement_required not in (None, False):
        issues.append(
            f"{public_status} verified_baseline replacement_required must be boolean"
        )
    if baseline.get("review_eligible") is not True:
        issues.append(f"{public_status} verified_baseline must be review eligible")
    return issues


def public_contract_issues(workload: Workload) -> list[str]:
    """Return public-result contract blockers for one workload."""
    issues: list[str] = []
    if workload.public_status not in PUBLIC_STATUSES:
        issues.append(f"invalid public status '{workload.public_status}'")
    if workload.public_status not in PUBLIC_RELEASE_STATUSES:
        issues.append("experimental workloads are not public-release-ready")
    adapter_conformance = workload.raw.get("adapter_conformance") or {}
    if adapter_conformance.get("status") == "quality-preserving-nonidentical":
        issues.append(
            "adapter conformance is quality-preserving but nonidentical; public "
            "promotion is blocked"
        )
    if not workload.public_rationale.strip():
        issues.append("missing public rationale")
    if not workload.scenario:
        issues.append("missing scenario")
    elif (
        workload.public_status in {"score-bearing", "performance-bearing"}
        and workload.scenario not in PUBLIC_RESULT_SCENARIOS
    ):
        issues.append("public-result workload uses an unsupported scenario")
    elif (
        workload.public_status == "performance-bearing"
        and workload.scenario == "training"
    ):
        issues.append(
            "performance-bearing inference workloads cannot use the training scenario"
        )

    runner = workload.raw.get("runner") or {}
    for profile in ("min", "max"):
        if profile not in runner:
            issues.append(f"missing {profile} runner")

    if workload.public_status in {"score-bearing", "performance-bearing"}:
        if not isinstance(workload.raw.get("canonical_max_contract"), dict):
            issues.append(
                f"{workload.public_status} workload must declare canonical_max_contract"
            )
        if not isinstance(workload.raw.get("measurement_protocol"), dict):
            issues.append(
                f"{workload.public_status} workload must declare measurement_protocol"
            )
        baseline = workload.raw.get("verified_baseline")
        if not isinstance(baseline, dict):
            issues.append(
                f"{workload.public_status} workload must declare verified_baseline"
            )
        else:
            if baseline.get("evidence_status") != "committed-reference-summary":
                issues.append(
                    f"{workload.public_status} verified_baseline must cite a committed reference summary"
                )
            issues.extend(
                baseline_lifecycle_issues(
                    baseline, public_status=workload.public_status
                )
            )
        baselines = workload.raw.get("verified_baselines")
        phase_count = len(
            (
                ((workload.raw.get("mode_contracts") or {}).get("inference") or {}).get(
                    "phases", {}
                )
            )
        )
        if not isinstance(baselines, dict) or len(baselines) != 1 + phase_count:
            issues.append(
                f"{workload.public_status} workload must bind every canonical and "
                "phase execution under verified_baselines"
            )
        elif isinstance(baseline, dict):
            canonical_case = baseline.get("case_id")
            if (
                canonical_case not in baselines
                or baselines.get(canonical_case) != baseline
            ):
                issues.append(
                    "verified_baseline must match its canonical verified_baselines entry"
                )
            for identifier, case_baseline in baselines.items():
                if not isinstance(case_baseline, dict):
                    issues.append(f"verified_baselines.{identifier} must be a mapping")
                    continue
                issues.extend(
                    baseline_lifecycle_issues(
                        case_baseline,
                        public_status=str(
                            case_baseline.get("result_role") or "unknown"
                        ),
                    )
                )

    if workload.public_status == "score-bearing":
        if not workload.dataset:
            issues.append("score-bearing workload must declare a dataset")
        if not workload.raw.get("dataset_source"):
            issues.append("score-bearing workload must declare dataset_source")
        if workload.dataset and not has_asset_dossier(workload.dataset):
            issues.append("score-bearing dataset must have a structured asset dossier")
        if not workload.quality_metric:
            issues.append("score-bearing workload must declare quality_target.metric")
        if workload.quality_value is None:
            issues.append("score-bearing workload must declare quality_target.value")
        if workload.quality_direction not in {"higher", "lower"}:
            issues.append(
                "score-bearing workload must declare quality_target.direction as higher or lower"
            )
        if workload.quality_target_basis not in QUALITY_TARGET_BASES:
            issues.append(
                "score-bearing workload must declare quality_target.target_basis as one of "
                + ", ".join(QUALITY_TARGET_BASES)
            )
        if not workload.quality_reference_runs:
            issues.append(
                "score-bearing workload must declare quality_target.reference_runs"
            )
        elif workload.quality_target_basis == "reference_runs":
            try:
                if int(workload.quality_reference_runs) < 3:
                    issues.append(
                        "score-bearing reference-run targets require at least 3 reference runs"
                    )
            except (TypeError, ValueError):
                issues.append(
                    "score-bearing quality_target.reference_runs must be an integer"
                )
        variance = workload.quality_variance_summary
        if not isinstance(variance, dict) or not variance:
            issues.append(
                "score-bearing workload must declare quality_target.variance_summary"
            )
        else:
            for key in ("runs", "statistic", "acceptance_rule"):
                if not variance.get(key):
                    issues.append(f"quality_target.variance_summary must declare {key}")
        if workload.quality_target_basis == "reference_runs":
            protocol = workload.quality_reference_protocol
            if not isinstance(protocol, dict) or not protocol:
                issues.append(
                    "reference-run score-bearing workload must declare quality_target.reference_protocol"
                )
            else:
                for key in REFERENCE_PROTOCOL_FIELDS:
                    if not protocol.get(key):
                        issues.append(
                            f"quality_target.reference_protocol must declare {key}"
                        )
                seeds = protocol.get("seeds")
                if not isinstance(seeds, list) or len(seeds) != int(
                    workload.quality_reference_runs or 0
                ):
                    issues.append(
                        "quality_target.reference_protocol.seeds must list every reference run seed"
                    )
        if not workload.quality_reviewer_notes:
            issues.append(
                "score-bearing workload must declare quality_target.reviewer_notes"
            )
        baseline_record = workload.raw.get("verified_baseline")
        if not baseline_is_protocol_superseded(baseline_record):
            baseline = verified_baseline_value(workload)
            if baseline is None:
                issues.append(
                    "score-bearing verified_baseline must include the quality metric or a known alias"
                )
            elif (
                workload.quality_direction in {"higher", "lower"}
                and workload.quality_value is not None
            ):
                try:
                    if not quality_target_satisfied(
                        float(baseline),
                        float(workload.quality_value),
                        direction=workload.quality_direction,
                        tolerance=float(workload.quality_tolerance or 0),
                    ):
                        issues.append(
                            f"verified_baseline does not satisfy quality target ({baseline} vs {workload.quality_value}, {workload.quality_direction})"
                        )
                except (TypeError, ValueError):
                    issues.append(
                        "quality target and verified_baseline values must be numeric"
                    )

    if workload.public_status == "performance-bearing":
        if not any(
            workload.raw.get(key)
            for key in ("model_source", "shared_checkpoint", "dataset")
        ):
            issues.append(
                "performance-bearing workload must declare a model_source, shared_checkpoint, or dataset"
            )
        functional_check = workload.raw.get("functional_check")
        if not isinstance(functional_check, dict) or not functional_check:
            issues.append("performance-bearing workload must declare functional_check")
        else:
            for key in ("metric", "condition"):
                if not functional_check.get(key):
                    issues.append(f"functional_check must declare {key}")
        model_source = workload.raw.get("model_source")
        if isinstance(model_source, dict) and not model_source.get("license"):
            issues.append("performance-bearing model_source must declare license")
        if workload.dataset and not workload.raw.get("dataset_source"):
            issues.append(
                "performance-bearing workload with dataset must declare dataset_source"
            )
        if workload.dataset and not has_asset_dossier(workload.dataset):
            issues.append(
                "performance-bearing dataset must have a structured asset dossier"
            )
        if workload.raw.get("shared_checkpoint") and not workload.raw.get(
            "quality_dependency"
        ):
            issues.append(
                "performance-bearing shared_checkpoint workload must declare quality_dependency"
            )

    return issues


QUALITY_BASELINE_ALIASES = {
    "accuracy": (
        "accuracy",
        "val_accuracy",
        "test_accuracy",
        "top1_accuracy",
        "binary_accuracy",
    ),
    "top1_accuracy": ("top1_accuracy", "accuracy", "val_accuracy", "test_accuracy"),
    "binary_accuracy": ("binary_accuracy", "accuracy", "val_accuracy", "test_accuracy"),
    "test_accuracy": ("test_accuracy", "accuracy", "val_accuracy"),
    "val_accuracy": ("val_accuracy", "accuracy", "test_accuracy"),
    "cross_entropy_loss": ("cross_entropy_loss", "val_loss", "train_loss", "loss"),
    "mse_loss": ("mse_loss", "val_mse", "val_loss", "train_loss", "loss"),
    "val_mse": ("val_mse", "mse_loss", "val_loss"),
    "reconstruction_mse": (
        "reconstruction_mse",
        "final_reconstruction_mse",
        "train_loss",
        "val_loss",
    ),
}


def verified_baseline_value(workload: Workload) -> Any:
    baseline = workload.raw.get("verified_baseline")
    if not isinstance(baseline, dict) or not workload.quality_metric:
        return None
    for key in QUALITY_BASELINE_ALIASES.get(
        workload.quality_metric, (workload.quality_metric,)
    ):
        if key in baseline:
            return baseline[key]
    return baseline.get(workload.quality_metric)


def quality_target_satisfied(
    baseline: float,
    target: float,
    *,
    direction: str,
    tolerance: float = 0.0,
) -> bool:
    if direction == "higher":
        return baseline + tolerance >= target
    if direction == "lower":
        return baseline - tolerance <= target
    return False


def public_contract_report(workloads: dict[str, Workload]) -> dict[str, list[str]]:
    """Map workload id to public-result contract blockers."""
    return {
        workload.id: public_contract_issues(workload) for workload in workloads.values()
    }


def select_workloads(
    workloads: dict[str, Workload],
    *,
    suite: str | None = None,
    collection: str | None = None,
    workload_id: str | None = None,
    maturity: str | None = None,
    public_status: str | None = None,
) -> list[Workload]:
    if workload_id and (suite or collection):
        raise ValueError("choose either --workload or a suite/profile selection")
    if suite and suite not in PRODUCT_SUITES:
        raise ValueError(f"unknown suite '{suite}'")
    if collection and collection not in WORKLOAD_COLLECTIONS:
        raise ValueError(f"unknown workload collection '{collection}'")
    if public_status and public_status not in PUBLIC_STATUSES:
        raise ValueError(f"unknown public status '{public_status}'")

    selected = list(workloads.values())
    collection_order: dict[str, int] = {}
    if collection == "starter":
        collection_order = {
            workload_id: idx for idx, workload_id in enumerate(STARTER_WORKLOAD_ORDER)
        }
        selected = [workloads[workload_id] for workload_id in STARTER_WORKLOAD_ORDER]
    elif collection == "standard":
        collection_order = {
            workload_id: idx for idx, workload_id in enumerate(STANDARD_WORKLOAD_ORDER)
        }
        selected = [workloads[workload_id] for workload_id in STANDARD_WORKLOAD_ORDER]
    elif collection == "research":
        collection_order = {
            workload_id: idx for idx, workload_id in enumerate(RESEARCH_WORKLOAD_ORDER)
        }
        selected = [workloads[workload_id] for workload_id in RESEARCH_WORKLOAD_ORDER]
    elif collection == "all":
        selected = list(workloads.values())
    if suite:
        selected = [w for w in selected if w.suite == suite]
    if workload_id:
        if workload_id not in workloads:
            raise ValueError(f"unknown workload '{workload_id}'")
        selected = [workloads[workload_id]]
    if maturity:
        selected = [w for w in selected if w.maturity == maturity]
    if public_status:
        selected = [w for w in selected if w.public_status == public_status]

    if collection_order:
        return sorted(
            selected, key=lambda w: collection_order.get(w.id, len(collection_order))
        )

    order = {
        workload_id: idx for idx, workload_id in enumerate(STANDARD_WORKLOAD_ORDER)
    }
    return sorted(selected, key=lambda w: (w.suite, order.get(w.id, len(order)), w.id))
