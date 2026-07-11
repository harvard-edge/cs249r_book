from __future__ import annotations

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

QUALITY_TARGET_BASES = (
    "reference_runs",
    "literature",
    "mlcommons_derived",
    "pedagogical_baseline",
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
    "slm",
    "vision",
    "recommender",
    "tiny",
    "agent",
    "distributed",
    "graph",
    "timeseries",
    "rl",
)

WORKLOAD_COLLECTIONS = (
    "starter",
    "standard",
    "research",
    "all",
)

DEFAULT_WORKLOAD_COLLECTION = "starter"

STARTER_WORKLOADS = {
    "nanogpt-train",
    "nanogpt-prefill",
    "nanogpt-decode",
    "slm-decode",
    "resnet18-train",
    "micro-dlrm-train",
    "anomaly-ae-train",
    "nano-rag-agent",
    "micro-dlrm-distributed",
    "micro-gnn-train",
    "micro-lstm-train",
    "micro-rl-train",
}

STARTER_WORKLOAD_ORDER = (
    "nanogpt-train",
    "nanogpt-prefill",
    "nanogpt-decode",
    "slm-decode",
    "micro-dlrm-train",
    "resnet18-train",
    "anomaly-ae-train",
    "nano-rag-agent",
    "micro-dlrm-distributed",
    "micro-gnn-train",
    "micro-lstm-train",
    "micro-rl-train",
)

STANDARD_WORKLOAD_ORDER = (
    *STARTER_WORKLOAD_ORDER,
    "slm-quantized-decode",
    "mobilenetv2-train",
    "mobilenet-cifar100-composed-fp16",
    "micro-dlrm-dram-train",
    "dscnn-kws-train",
    "wake-vision-vww",
    "nano-codegen-agent",
    "nano-react-agent",
    "nano-toolcall-agent",
)

STANDARD_WORKLOADS = set(STANDARD_WORKLOAD_ORDER)

RESEARCH_WORKLOAD_ORDER = (
    "micro-bert-train",
    "micro-diffusion-train",
    "micro-gnn-train",
    "micro-lstm-train",
    "micro-rl-train",
    "nano-lora-finetune",
    "nano-moe-train",
    "slm-batched-decode",
    "slm-long-context-decode",
    "nanogpt-decode-fp16-b16",
    "nanogpt-decode-fp32-b16",
    "nanogpt-decode-spec",
)

RESEARCH_WORKLOADS = set(RESEARCH_WORKLOAD_ORDER)

SUITE_ALIASES = {
    "edge": "vision",
    "cloud": "language",
}


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
    quality_tolerance: Any
    quality_reference_runs: Any
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
            suite = infer_product_suite(workload_id, source_suite)
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
                quality_tolerance=quality.get("tolerance"),
                quality_reference_runs=quality.get("reference_runs"),
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


def infer_product_suite(workload_id: str, source_suite: str) -> str:
    if workload_id.startswith("slm-"):
        return "slm"
    if source_suite == "agent" or "agent" in workload_id:
        return "agent"
    if "distributed" in workload_id:
        return "distributed"
    if source_suite == "edge" or "mobilenet" in workload_id or "resnet" in workload_id:
        return "vision"
    if "diffusion" in workload_id:
        return "vision"
    if "dlrm" in workload_id:
        return "recommender"
    if source_suite == "tiny" or "kws" in workload_id or "anomaly" in workload_id:
        return "tiny"
    if "gnn" in workload_id:
        return "graph"
    if "lstm" in workload_id:
        return "timeseries"
    if "rl" in workload_id:
        return "rl"
    if (
        "gpt" in workload_id
        or "bert" in workload_id
        or "moe" in workload_id
        or "lora" in workload_id
        or source_suite == "cloud"
    ):
        return "language"
    return SUITE_ALIASES.get(
        source_suite, source_suite if source_suite in PRODUCT_SUITES else "language"
    )


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


def public_contract_issues(workload: Workload) -> list[str]:
    """Return public-result contract blockers for one workload."""
    issues: list[str] = []
    if workload.public_status not in PUBLIC_STATUSES:
        issues.append(f"invalid public status '{workload.public_status}'")
    if workload.public_status not in PUBLIC_RELEASE_STATUSES:
        issues.append("experimental workloads are not public-release-ready")
    if not workload.public_rationale.strip():
        issues.append("missing public rationale")
    if not workload.scenario:
        issues.append("missing scenario")
    elif (
        workload.public_status in {"score-bearing", "performance-bearing"}
        and workload.scenario not in PUBLIC_RESULT_SCENARIOS
    ):
        issues.append("public-result workload uses an unsupported scenario")
    elif workload.public_status == "score-bearing" and workload.scenario != "training":
        issues.append("current score-bearing workloads must use the training scenario")
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
            if baseline.get("review_eligible") is not True:
                issues.append(
                    f"{workload.public_status} verified_baseline must be review eligible"
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
