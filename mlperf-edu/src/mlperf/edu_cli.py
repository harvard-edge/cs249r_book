from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
import platform
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from html import escape
from importlib import import_module, resources
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Any

from rich.console import Console
from rich.table import Table

from .assets import (
    BFCL_ARCHIVE_URL,
    BFCL_DATA_FILES,
    CIFAR10_HF_REPO_ID,
    CIFAR10_HF_REVISION,
    EDM_CIFAR10_CHECKPOINT_URL,
    EDM_CIFAR10_FID_REFERENCE_URL,
    EEMBC_RUNNER_ARCHIVE_URL,
    HUMANEVAL_PLUS_URL,
    MLPERF_TINY_ANOMALY_ARCHIVE_URL,
    MLPERF_TINY_VWW_ARCHIVE_URL,
    TINY_SHAKESPEARE_URL,
    GLUE_SST2_URL,
    OGBN_ARXIV_URL,
    ETTM1_URL,
    NANOBEIR_REPO_ID,
    NANOBEIR_REVISION,
    asset_cache_root,
    asset_dossier,
    bfcl_non_live_ast_paths,
    cifar10_paths,
    edm_cifar10_paths,
    ensure_bfcl_non_live_ast,
    ensure_cifar10,
    ensure_edm_cifar10,
    ensure_humaneval_plus,
    ensure_mlperf_tiny_anomaly,
    ensure_mlperf_tiny_image,
    ensure_mlperf_tiny_kws,
    ensure_mlperf_tiny_vww,
    ensure_sst2,
    ensure_ogbn_arxiv,
    ensure_ettm1,
    ensure_nanobeir_reranking,
    ensure_tinyshakespeare,
    huggingface_model_dossier,
    has_asset_dossier,
    humaneval_plus_paths,
    mlperf_tiny_image_paths,
    mlperf_tiny_anomaly_paths,
    mlperf_tiny_kws_paths,
    mlperf_tiny_vww_paths,
    sst2_paths,
    ogbn_arxiv_paths,
    ettm1_paths,
    nanobeir_reranking_paths,
    sha256_file,
    tinyshakespeare_paths,
)
from .fingerprint import detect_hardware
from .contracts import aggregate_contract_issues, evaluate_report_contract
from .manifest import (
    build_provd,
    dataset_merkle_root,
    integrity_record,
    measurement_leaf,
    merkle_root,
    resolve_artifact_path,
    safe_logical_asset_path,
    verify_provd,
)
from .power import PowerMeter
from .registry import (
    DEFAULT_WORKLOAD_COLLECTION,
    PRODUCT_SUITES,
    PROFILES,
    PUBLIC_STATUSES,
    RESEARCH_WORKLOADS,
    STARTER_WORKLOADS,
    WORKLOAD_COLLECTIONS,
    Workload,
    default_registry_path,
    find_project_root,
    load_registry,
    public_contract_report,
    select_workloads,
)


console = Console(width=140)
DEFAULT_MLPERF_SUITE = "mlperf-edu"
VALIDATE_PRESETS = ("smoke", "coverage", "max", "pro", "release")
PROFILE_CHOICES = PROFILES
LEGACY_VALIDATE_LEVELS = {
    "quick": "smoke",
    "min": "coverage",
    "max": "max",
    "release": "release",
}
PROFILE_DESCRIPTIONS = {
    "min": "Minimum representative path for setup, CI, and quick instructional checks.",
    "max": "Full MLPerf EDU suite at comparable scale.",
    "pro": "Research envelope exposing controlled variants and optimization knobs.",
}
LEGACY_QUALITY_REQUIRED_FIELD = "gated"
PROMOTED_BASELINE_FIELDS = frozenset(
    {
        "evidence_id",
        "evidence_sha256",
        "promoted_aggregate",
        "source_git_sha",
        "source_verified_baseline",
        "variance_summary",
        "verified_baseline",
    }
)


def quality_required_value(quality: dict[str, Any], default: Any = False) -> Any:
    return quality.get(
        "quality_required", quality.get(LEGACY_QUALITY_REQUIRED_FIELD, default)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlperf",
        description=(
            "MLPerf EDU command harness. Defaults to the mlperf-edu suite. "
            "Common user path: init, list, fetch, audit, run, report. "
            "Instructor/maintainer path: audit, validate, grade."
        ),
        epilog=(
            "Common user commands:\n"
            "  doctor   check this machine\n"
            "  init     prepare caches and optionally smoke-test the setup\n"
            "  list     discover workloads\n"
            "  fetch    download or verify needed assets\n"
            "  audit    check source, license, dataset, model, and quality metadata\n"
            "  run      run a workload, suite, or default profile\n"
            "  report   open/export a result\n\n"
            "Instructor and maintainer commands:\n"
            "  validate run validation presets that execute workloads and grade artifacts\n"
            "  audit    check public-result metadata without running workloads\n"
            "  grade    grade a submissions directory\n"
            "  verify   verify one provenance manifest\n"
            "  package  bundle a verified submission"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to workloads.yaml. Defaults to the nearest project registry.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Check local environment and registry"
    )
    doctor_selection = doctor.add_mutually_exclusive_group()
    doctor_selection.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        help="Check one workload domain, such as language or vision",
    )
    doctor_selection.add_argument(
        "--workload",
        default=None,
        help="Check one workload id or canonical workload family",
    )
    doctor_selection.add_argument(
        "--collection",
        choices=WORKLOAD_COLLECTIONS,
        default=None,
        help="Check an explicit workload collection",
    )
    add_profile(doctor)
    doctor.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    doctor.add_argument("--format", choices=("summary", "json"), default="summary")
    doctor.set_defaults(func=cmd_doctor)

    init = subparsers.add_parser("init", help="Prepare local caches for a profile")
    add_profile(init)
    init_selection = init.add_mutually_exclusive_group()
    init_selection.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        help="Prepare one workload domain, such as language or vision",
    )
    init_selection.add_argument(
        "--workload",
        default=None,
        help="Prepare one workload id or canonical workload family",
    )
    init_selection.add_argument(
        "--collection",
        choices=WORKLOAD_COLLECTIONS,
        default=None,
        help="Prepare an explicit workload collection",
    )
    init.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    init.add_argument("--no-smoke", action="store_true")
    init.add_argument(
        "--output-dir",
        default="submissions/init_smoke",
        help="Directory for init smoke-validation artifacts.",
    )
    init.set_defaults(func=cmd_init)

    fetch = subparsers.add_parser("fetch", help="Fetch or verify assets for workloads")
    add_selection(fetch)
    add_profile(fetch)
    fetch.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    fetch.add_argument("--dry-run", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    run = subparsers.add_parser(
        "run", help="Run workloads by profile, suite, or workload id"
    )
    add_selection(run)
    add_profile(run)
    add_device(run)
    run.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    run.add_argument(
        "--output-dir",
        default="submissions",
        help="Directory for report artifacts.",
    )
    run.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated HTML report in the default browser.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected workloads without running them",
    )
    run.add_argument(
        "--mode",
        choices=("training", "inference"),
        default=None,
        help="Execution mode for a workload that defines training and inference under one identity",
    )
    run.add_argument(
        "--phase",
        choices=("full", "prefill", "decode"),
        default=None,
        help="Inference phase; valid only with an inference-capable single workload",
    )
    run.add_argument(
        "--power",
        action="store_true",
        help="Add estimated aggregate power and energy telemetry to the run report",
    )
    run.set_defaults(func=cmd_run)

    verify = subparsers.add_parser("verify", help="Verify a provenance manifest")
    verify.add_argument("manifest", type=str)
    verify.set_defaults(func=cmd_verify)

    report = subparsers.add_parser("report", help="Print a compact report summary")
    report.add_argument(
        "report",
        type=str,
        help="Path to a workload/aggregate report JSON, or a run directory",
    )
    report.add_argument(
        "--format", choices=("summary", "json", "csv", "html"), default="summary"
    )
    report.add_argument(
        "--output", type=str, default=None, help="Output path for json/csv/html formats"
    )
    report.add_argument(
        "--open", action="store_true", help="Open generated HTML in the default browser"
    )
    report.set_defaults(func=cmd_report)

    package = subparsers.add_parser("package", help="Package a verified submission")
    package.add_argument("manifest", type=str)
    package.add_argument(
        "--output", "-o", type=str, default=None, help="Output .zip path"
    )
    package.set_defaults(func=cmd_package)

    grade = subparsers.add_parser("grade", help="Grade a directory of submissions")
    grade.add_argument("submissions_dir", nargs="?", default="submissions")
    grade.add_argument(
        "--output", type=str, default=None, help="Write grading summary JSON"
    )
    grade.set_defaults(func=cmd_grade)

    validate = subparsers.add_parser(
        "validate",
        help="Run bundled validation presets",
        description="Run MLPerf EDU validation presets. This executes workloads and grades artifacts.",
    )
    add_validate_arguments(validate)
    validate.set_defaults(func=cmd_validate)

    audit = subparsers.add_parser(
        "audit",
        help="Audit the public result contract",
        description="Maintainer command: audit registry metadata and public-result labels. This does not run benchmarks.",
    )
    audit.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        help="Audit one workload domain, such as language or vision",
    )
    audit.add_argument(
        "--workload",
        default=None,
        help="Audit one workload id or canonical workload family",
    )
    audit.add_argument("--profile", choices=PROFILE_CHOICES, default=None)
    audit.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    audit.add_argument("--status", dest="public_status", choices=PUBLIC_STATUSES)
    audit.add_argument(
        "--policy",
        choices=("development", "public"),
        default="development",
        help="Use public to fail on unresolved endorsement warnings.",
    )
    audit.add_argument("--format", choices=("summary", "json"), default="summary")
    audit.set_defaults(func=cmd_audit)

    list_parser = subparsers.add_parser("list", help="List workloads")
    list_parser.add_argument(
        "subject",
        nargs="?",
        choices=("suites", "profiles", "workloads", "variants", "matrix"),
        default="workloads",
        help="Discovery subject to list.",
    )
    list_parser.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        help="Filter by workload domain, such as language or vision",
    )
    list_parser.add_argument("--profile", choices=PROFILE_CHOICES, default=None)
    list_parser.add_argument(
        "--workload", default=None, help="Filter by workload id or canonical workload"
    )
    list_parser.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    list_parser.add_argument(
        "--maturity",
        choices=("base", "research", "experimental"),
        help=argparse.SUPPRESS,
    )
    list_parser.add_argument("--public-status", choices=PUBLIC_STATUSES)
    list_parser.add_argument("--format", choices=("summary", "json"), default="summary")
    list_parser.set_defaults(func=cmd_list)

    show = subparsers.add_parser("show", help="Show one workload")
    show.add_argument("workload", type=str)
    show.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    show.set_defaults(func=cmd_show)

    info = subparsers.add_parser(
        "info", help="Show suite, profile, workload, model, dataset, or run details"
    )
    info_group = info.add_mutually_exclusive_group(required=True)
    info_group.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        help="Show one workload domain, such as language or vision",
    )
    info_group.add_argument("--profile", choices=PROFILE_CHOICES)
    info_group.add_argument(
        "--workload",
        default=None,
        help="Show one workload id or canonical workload family",
    )
    info_group.add_argument("--model", default=None)
    info_group.add_argument("--dataset", default=None)
    info_group.add_argument("--run", default=None)
    info.add_argument("--variant", default=None)
    info.set_defaults(func=cmd_info)

    cache = subparsers.add_parser(
        "cache", help="Inspect and verify local MLPerf EDU assets"
    )
    cache.add_argument("action", nargs="?", choices=("list", "verify"), default="list")
    cache_selection = cache.add_mutually_exclusive_group()
    cache_selection.add_argument(
        "--suite", choices=PRODUCT_SUITES, help="Inspect assets for one workload domain"
    )
    cache_selection.add_argument(
        "--workload",
        default=None,
        help="Inspect assets for one workload id or canonical workload family",
    )
    add_profile(cache)
    cache.add_argument(
        "--variant", default=None, help="Variant under a canonical workload"
    )
    cache.add_argument("--format", choices=("summary", "json"), default="summary")
    cache.set_defaults(func=cmd_cache)

    return parser


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "preset",
        metavar="preset",
        nargs="?",
        choices=VALIDATE_PRESETS,
        default=None,
        help="Validation preset: smoke=fast default/min, coverage=all workloads/min, max=all workloads/max, pro=research collection/pro, release=all workloads/min+max and research collection/pro.",
    )
    parser.add_argument(
        "--preset",
        dest="preset_option",
        choices=VALIDATE_PRESETS,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--level",
        dest="legacy_level",
        choices=tuple(LEGACY_VALIDATE_LEVELS),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        action="append",
        default=None,
        help="Restrict validation to one or more suites. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="submissions/validation",
        help="Root directory for validation artifacts.",
    )
    parser.add_argument(
        "--skip-doctor", action="store_true", help="Skip the doctor preflight"
    )
    parser.add_argument(
        "--skip-grade", action="store_true", help="Skip manifest grading after each run"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed validation item",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned validation items without running them",
    )
    parser.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated validation HTML summary",
    )
    add_device(parser)


def add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help=(
            "Execution device. auto selects CUDA, then MPS, then CPU. "
            "An explicit unavailable device fails before execution."
        ),
    )


def add_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="min",
        help=(
            "Run scale: min=quick representative path, max=full suite, "
            "pro=research variants and knobs. Defaults to min."
        ),
    )


def normalize_profile(profile: str) -> str:
    return profile


def add_selection(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--suite",
        choices=PRODUCT_SUITES,
        default=None,
        help="Select one workload domain, such as language or vision",
    )
    group.add_argument(
        "--workload",
        default=None,
        help="Select one workload id or canonical workload family",
    )
    group.add_argument(
        "--collection",
        choices=WORKLOAD_COLLECTIONS,
        default=None,
        help="Select an explicit workload collection",
    )


def load_workloads(args: argparse.Namespace) -> dict[str, Workload]:
    return load_registry(args.registry)


def default_collection_for(args: argparse.Namespace) -> str | None:
    if getattr(args, "collection", None):
        return args.collection
    if getattr(args, "suite", None) or getattr(args, "workload", None):
        return None
    return profile_collection(getattr(args, "profile", "min"))


def select_cli_workloads(
    workloads: dict[str, Workload], args: argparse.Namespace
) -> list[Workload]:
    workload = getattr(args, "workload", None)
    variant = getattr(args, "variant", None)
    if workload and not variant and workload not in workloads:
        selected_ids = resolve_workload_ids(workloads, workload)
        if not selected_ids:
            raise ValueError(f"unknown workload or canonical workload '{workload}'")
        selected = [workloads[workload_id] for workload_id in selected_ids]
        suite = getattr(args, "suite", None)
        if suite:
            selected = [item for item in selected if item.suite == suite]
        return selected

    resolved_workload = resolve_cli_workload_id(workloads, workload, variant)
    return select_workloads(
        workloads,
        suite=getattr(args, "suite", None),
        collection=default_collection_for(args),
        workload_id=resolved_workload,
    )


def selection_label(
    *,
    suite: str | None,
    workload: str | None = None,
    collection: str | None = None,
) -> str:
    if workload:
        return workload
    if suite:
        return suite
    if collection:
        return f"collection:{collection}"
    return "default"


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, str]] = []
    selected: list[Workload] = []
    workloads: dict[str, Workload] = {}
    profile = getattr(args, "profile", "min")

    def add_check(name: str, detail: str, status: str) -> None:
        checks.append({"name": name, "detail": detail, "status": status})

    add_check("mlperf suite", DEFAULT_MLPERF_SUITE, "ok")
    add_check("python", platform.python_version(), "ok")
    add_check("platform", platform.platform(), "ok")
    add_check("data cache", str(asset_cache_root()), "ok")
    add_check("model cache", str(default_model_cache_dir()), "ok")

    try:
        import torch

        backends = ["cpu"]
        if torch.cuda.is_available():
            backends.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            backends.append("mps")
        add_check("torch", f"{torch.__version__} ({', '.join(backends)})", "ok")
    except Exception as exc:
        add_check("torch", str(exc), "fail")

    for module_name in ("onnxruntime", "mlx", "llama_cpp"):
        status = "ok" if find_spec(module_name) else "optional"
        detail = "installed" if status == "ok" else "not installed"
        add_check(module_name, detail, status)

    try:
        workloads = load_workloads(args)
        registry_path = default_registry_path(args.registry)
        add_check("registry", f"{len(workloads)} workloads at {registry_path}", "ok")
    except Exception as exc:
        add_check("registry", str(exc), "fail")

    if workloads:
        try:
            selected = select_cli_workloads(workloads, args)
            selector = selection_label(
                suite=getattr(args, "suite", None),
                workload=getattr(args, "workload", None),
                collection=getattr(args, "collection", None),
            )
            if getattr(args, "variant", None):
                selector = f"{selector}:{args.variant}"
            add_check(
                "selection",
                f"{len(selected)} workload(s) for profile {profile} ({selector})",
                "ok",
            )
        except Exception as exc:
            add_check("selection", str(exc), "fail")

    try:
        hw = detect_hardware()
        add_check("hardware", f"{hw.get('chip')} / {hw.get('backend')}", "ok")
    except Exception as exc:
        add_check("hardware", str(exc), "warn")

    if getattr(args, "format", "summary") == "json":
        payload = {
            "schema": "mlperf-edu-doctor/0.1",
            "mlperf_suite": DEFAULT_MLPERF_SUITE,
            "profile": profile,
            "suite": getattr(args, "suite", None),
            "workload": getattr(args, "workload", None),
            "collection": getattr(args, "collection", None),
            "variant": getattr(args, "variant", None),
            "checks": checks,
            "selected_workloads": [workload_summary(workload) for workload in selected],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if any(check["status"] == "fail" for check in checks) else 0

    table = Table(title="MLPerf EDU Doctor")
    table.add_column("Check")
    table.add_column("Detail")
    table.add_column("Status")
    for check in checks:
        name = check["name"]
        detail = check["detail"]
        status = check["status"]
        style = (
            "green"
            if status == "ok"
            else "yellow"
            if status in {"optional", "warn"}
            else "red"
        )
        table.add_row(name, detail, f"[{style}]{status}[/{style}]")
    console.print(table)

    return 1 if any(check["status"] == "fail" for check in checks) else 0


def cmd_init(args: argparse.Namespace) -> int:
    console.print("[bold]Init preflight: doctor[/bold]")
    doctor_status = cmd_doctor(args)
    if doctor_status != 0:
        console.print("[red]init failed during doctor preflight[/red]")
        return doctor_status

    workloads = load_workloads(args)
    selected = select_cli_workloads(workloads, args)
    label = selection_label(
        suite=args.suite,
        workload=args.workload,
        collection=args.collection,
    )
    if args.variant:
        label = f"{label}:{args.variant}"
    console.print(f"[bold]Initialized profile[/bold] {args.profile} for {label}")
    console.print(f"Validated {len(selected)} workload definitions.")
    print_run_selection(
        args.profile,
        selected,
        suite=args.suite,
        workload=args.workload,
        collection=args.collection,
        variant=args.variant,
    )
    output_dir = Path(args.output_dir).resolve()
    print_init_locations(output_dir)

    if args.profile != "min":
        console.print(f"Preparing assets for {args.profile}.")
        for workload in selected:
            console.print(fetch_workload_asset(workload, dry_run=False))
    else:
        console.print(
            "No profile assets required beyond min-profile runner-local data."
        )

    if args.no_smoke:
        print_next_commands(args)
        return 0

    console.print("Running min-profile smoke validation.")
    workload_reports = [
        run_workload(workload, "min", output_dir) for workload in selected
    ]
    enrich_reports_for_display(workload_reports, workloads)
    export_workload_reports(workload_reports, workloads)
    _, report_path, exports = write_aggregate_report(
        profile="min",
        suite=args.suite,
        workload=args.workload,
        collection=args.collection,
        variant=args.variant,
        workload_reports=workload_reports,
        output_dir=output_dir,
        open_report=False,
    )
    status = print_run_summary("min", workload_reports, report_path, exports)
    print_next_commands(args)
    return status


def print_init_locations(output_dir: Path) -> None:
    table = Table(title="MLPerf EDU Local Paths")
    table.add_column("Purpose", no_wrap=True)
    table.add_column("Path", overflow="fold")
    table.add_row("data cache", str(asset_cache_root()))
    table.add_row("model cache", str(default_model_cache_dir()))
    table.add_row("reports", str(output_dir))
    console.print(table)


def default_model_cache_dir() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def print_next_commands(args: argparse.Namespace) -> None:
    selector = []
    if getattr(args, "suite", None):
        selector.extend(["--suite", args.suite])
    if getattr(args, "workload", None):
        selector.extend(["--workload", args.workload])
    if getattr(args, "collection", None):
        selector.extend(["--collection", args.collection])
    if getattr(args, "variant", None):
        selector.extend(["--variant", args.variant])
    selector_text = " ".join(selector)
    suffix = f" {selector_text}" if selector_text else ""
    console.print("[bold]Next commands[/bold]")
    print(f"  mlperf fetch --profile {args.profile}{suffix} --dry-run")
    print(
        f"  mlperf run --profile {args.profile}{suffix} --output-dir {Path(args.output_dir).resolve()}"
    )
    print(f"  mlperf report {Path(args.output_dir).resolve()} --format html --open")


def cmd_fetch(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    selected = select_cli_workloads(workloads, args)
    action = "Would fetch" if args.dry_run else "Fetched/validated"
    print_run_selection(
        args.profile,
        selected,
        suite=args.suite,
        workload=args.workload,
        collection=args.collection,
        variant=args.variant,
    )
    console.print(f"{action} {len(selected)} workload(s) for profile {args.profile}.")
    manual_setup_required = False
    for workload in selected:
        console.print(fetch_workload_asset(workload, dry_run=args.dry_run))
        if not args.dry_run and workload.dataset in {
            "criteo-terabyte",
            "minigo-self-play",
        }:
            manual_setup_required = True
    return 2 if manual_setup_required else 0


def fetch_workload_asset(workload: Workload, *, dry_run: bool) -> str:
    model_source = workload.raw.get("model_source") or {}
    if model_source.get("type") == "huggingface-pinned":
        repo_id = str(model_source["repo_id"])
        revision = str(model_source["revision"])
        dataset = workload.dataset or "no dataset declared"
        dossier = asset_dossier(
            workload.dataset, declared_source=workload.raw.get("dataset_source")
        )
        terms = (
            asset_terms_summary(dossier) if dossier else "no structured asset dossier"
        )
        if dry_run:
            dataset_detail = dataset
            if dataset == "humaneval-plus":
                dataset_detail = (
                    f"{humaneval_plus_paths()['dataset']} ({HUMANEVAL_PLUS_URL})"
                )
            elif dataset == "bfcl-v4-non-live-ast":
                dataset_detail = (
                    f"{bfcl_non_live_ast_paths()['data']} ({BFCL_ARCHIVE_URL})"
                )
            return (
                f"- {workload.id}: huggingface model -> {repo_id}@{revision}; "
                f"dataset={dataset_detail}; {terms}"
            )
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=model_source.get("allow_patterns"),
            local_files_only=os.environ.get("MLPERF_EDU_HF_LOCAL_ONLY", "0") == "1",
        )
        if dataset == "sst2":
            asset = ensure_sst2(download=True)
            return (
                f"- {workload.id}: model {repo_id}@{revision} at {snapshot}; "
                f"{dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
            )
        if dataset == "nanobeir-reranking":
            asset = ensure_nanobeir_reranking(download=True)
            return (
                f"- {workload.id}: model {repo_id}@{revision} at {snapshot}; "
                f"{dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
            )
        if dataset == "humaneval-plus":
            asset = ensure_humaneval_plus(download=True)
            return (
                f"- {workload.id}: model {repo_id}@{revision} at {snapshot}; "
                f"{dataset} at {asset.root} ({asset.sha256[:19]}, "
                f"{asset.n_bytes} bytes); {terms}"
            )
        if dataset == "bfcl-v4-non-live-ast":
            asset = ensure_bfcl_non_live_ast(download=True)
            return (
                f"- {workload.id}: model {repo_id}@{revision} at {snapshot}; "
                f"{dataset} at {asset.root} ({asset.sha256[:19]}, "
                f"{asset.n_bytes} bytes); {terms}"
            )
        return f"- {workload.id}: model {repo_id}@{revision} at {snapshot}; dataset={dataset}; {terms}"
    shared_checkpoint = workload.raw.get("shared_checkpoint")
    if shared_checkpoint:
        dependency = workload.raw.get("quality_dependency") or shared_checkpoint
        prompt_asset = ""
        if workload.dataset:
            dossier = asset_dossier(
                workload.dataset, declared_source=workload.raw.get("dataset_source")
            )
            terms = (
                asset_terms_summary(dossier)
                if dossier
                else "no structured asset dossier"
            )
            prompt_asset = f"; prompt_fixture={workload.dataset}; {terms}"
        return (
            f"- {workload.id}: shared checkpoint -> {shared_checkpoint}; "
            f"quality_dependency={dependency}; source=MLPerf EDU training workload"
            f"{prompt_asset}"
        )

    dataset = workload.dataset or "no dataset declared"
    dossier = asset_dossier(
        workload.dataset, declared_source=workload.raw.get("dataset_source")
    )
    terms = asset_terms_summary(dossier) if dossier else "no structured asset dossier"
    if dataset == "tinyshakespeare":
        if dry_run:
            paths = tinyshakespeare_paths()
            return f"- {workload.id}: {dataset} -> {paths['full']} ({TINY_SHAKESPEARE_URL}); {terms}"
        asset = ensure_tinyshakespeare(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "cifar10":
        if workload.id == "image-generation":
            paths = edm_cifar10_paths()
            if dry_run:
                return (
                    f"- {workload.id}: EDM checkpoint -> {paths['checkpoint']} "
                    f"({EDM_CIFAR10_CHECKPOINT_URL}); FID reference -> "
                    f"{paths['fid_reference']} ({EDM_CIFAR10_FID_REFERENCE_URL}); "
                    f"{terms}"
                )
            asset = ensure_edm_cifar10(download=True)
            return (
                f"- {workload.id}: EDM checkpoint and FID reference at "
                f"{asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); "
                f"{terms}"
            )
        if dry_run:
            paths = cifar10_paths()
            evaluation_paths = mlperf_tiny_image_paths()
            source = f"https://huggingface.co/datasets/{CIFAR10_HF_REPO_ID}/tree/{CIFAR10_HF_REVISION}"
            return (
                f"- {workload.id}: {dataset} -> {paths['root']} ({source}); "
                f"MLPerf Tiny model/index -> {evaluation_paths['root']}; {terms}"
            )
        asset = ensure_cifar10(download=True)
        evaluation = ensure_mlperf_tiny_image(download=True)
        return (
            f"- {workload.id}: {dataset} at {asset.root} "
            f"({asset.sha256[:19]}, {asset.n_bytes} bytes); MLPerf Tiny model/index "
            f"at {evaluation.root} ({evaluation.sha256[:19]}, "
            f"{evaluation.n_bytes} bytes); {terms}"
        )
    if dataset == "mlperf-tiny-kws-eval":
        if dry_run:
            paths = mlperf_tiny_kws_paths()
            return f"- {workload.id}: {dataset} -> {paths['dataset']} ({EEMBC_RUNNER_ARCHIVE_URL}); {terms}"
        asset = ensure_mlperf_tiny_kws(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "mlperf-tiny-anomaly-eval":
        if dry_run:
            paths = mlperf_tiny_anomaly_paths()
            return (
                f"- {workload.id}: {dataset} -> {paths['dataset']} "
                f"({MLPERF_TINY_ANOMALY_ARCHIVE_URL}; selective range fetch); "
                f"{terms}"
            )
        asset = ensure_mlperf_tiny_anomaly(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "mlperf-tiny-vww-eval":
        if dry_run:
            paths = mlperf_tiny_vww_paths()
            return (
                f"- {workload.id}: {dataset} -> {paths['dataset']} "
                f"({MLPERF_TINY_VWW_ARCHIVE_URL}); {terms}"
            )
        asset = ensure_mlperf_tiny_vww(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "sst2":
        if dry_run:
            paths = sst2_paths()
            return f"- {workload.id}: {dataset} -> {paths['dataset']} ({GLUE_SST2_URL}); {terms}"
        asset = ensure_sst2(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "ogbn-arxiv":
        if dry_run:
            paths = ogbn_arxiv_paths()
            return f"- {workload.id}: {dataset} -> {paths['dataset']} ({OGBN_ARXIV_URL}); {terms}"
        asset = ensure_ogbn_arxiv(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "ettm1":
        if dry_run:
            paths = ettm1_paths()
            return (
                f"- {workload.id}: {dataset} -> {paths['csv']} ({ETTM1_URL}); {terms}"
            )
        asset = ensure_ettm1(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "nanobeir-reranking":
        if dry_run:
            paths = nanobeir_reranking_paths()
            source = f"https://huggingface.co/datasets/{NANOBEIR_REPO_ID}/tree/{NANOBEIR_REVISION}"
            return f"- {workload.id}: {dataset} -> {paths['root']} ({source}); {terms}"
        asset = ensure_nanobeir_reranking(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "humaneval-plus":
        paths = humaneval_plus_paths()
        if dry_run:
            return f"- {workload.id}: {dataset} -> {paths['dataset']} ({HUMANEVAL_PLUS_URL}); {terms}"
        asset = ensure_humaneval_plus(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "bfcl-v4-non-live-ast":
        paths = bfcl_non_live_ast_paths()
        if dry_run:
            return f"- {workload.id}: {dataset} -> {paths['data']} ({BFCL_ARCHIVE_URL}); {terms}"
        asset = ensure_bfcl_non_live_ast(download=True)
        return f"- {workload.id}: {dataset} at {asset.root} ({asset.sha256[:19]}, {asset.n_bytes} bytes); {terms}"
    if dataset == "criteo-terabyte":
        return (
            f"- {workload.id}: MANUAL ACTION REQUIRED; accept the Criteo terms, "
            "prepare unshuffled day 23, and provide the official MLPerf "
            f"Inference v1.0.1 40M checkpoint; {terms}"
        )
    if dataset == "minigo-self-play":
        return (
            f"- {workload.id}: MANUAL ACTION REQUIRED; prepare the pinned MiniGo "
            "professional-move inputs and authoritative self-play environment; "
            f"{terms}"
        )
    return f"- {workload.id}: {dataset}; {terms}"


def asset_terms_summary(dossier: dict[str, Any]) -> str:
    license_value = dossier.get("license", "unknown")
    status = dossier.get("license_status", "unknown")
    use = dossier.get("public_result_use", "requires review")
    release = dossier.get("public_release_status", "needs-release-decision")
    return (
        f"license={license_value}; terms={status}; release={release}; public_use={use}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    selected = select_cli_workloads(workloads, args)
    if not selected:
        console.print("[red]No workloads selected.[/red]")
        return 1
    requested_mode = getattr(args, "mode", None)
    requested_phase = getattr(args, "phase", None)
    if (requested_mode or requested_phase) and len(selected) != 1:
        raise ValueError("--mode and --phase require selection of exactly one workload")
    execution_mode = None
    execution_phase = None
    if len(selected) == 1:
        execution_mode, execution_phase = resolve_execution_selection(
            selected[0], mode=requested_mode, phase=requested_phase
        )

    print_run_selection(
        args.profile,
        selected,
        suite=args.suite,
        workload=args.workload,
        collection=args.collection,
        variant=args.variant,
    )
    if getattr(args, "dry_run", False):
        console.print("[green]dry-run complete[/green]")
        return 0

    output_dir = Path(args.output_dir).resolve()
    power_meter = PowerMeter()
    if args.power:
        power_meter.start()
    workload_reports: list[dict[str, Any]] = []
    for workload in selected:
        report = run_workload(
            workload,
            args.profile,
            output_dir,
            mode=execution_mode,
            phase=execution_phase,
        )
        annotate_execution_device(report)

        # Finalize each report and its provenance before starting the next
        # workload. A dependent workload can bind the exact report and
        # manifest bytes produced by an earlier workload. Deferring all
        # exports would mutate the source report after inference recorded its
        # lineage digests.
        enrich_report_for_display(report, workloads)
        export_workload_reports([report], workloads)
        workload_reports.append(report)

    power_report = power_meter.stop_report() if args.power else None
    _, report_path, exports = write_aggregate_report(
        profile=args.profile,
        suite=args.suite,
        workload=args.workload,
        collection=args.collection,
        variant=args.variant,
        workload_reports=workload_reports,
        output_dir=output_dir,
        open_report=args.open_report,
        power=power_report,
    )
    return print_run_summary(args.profile, workload_reports, report_path, exports)


def print_run_selection(
    profile: str,
    selected: list[Workload],
    *,
    suite: str | None,
    workload: str | None,
    collection: str | None,
    variant: str | None,
) -> None:
    selector = selection_label(
        suite=suite,
        workload=workload,
        collection=collection,
    )
    if variant:
        selector = f"{selector}:{variant}"
    console.print(
        f"Selected {len(selected)} workload(s) for profile {profile} ({selector})."
    )
    if not selected:
        return

    if len(selected) <= 10:
        for item in selected:
            console.print(
                f"  - {item.id} | run as: {workload_run_selector(item)} | suite: {item.suite}"
            )
        return

    counts: dict[str, int] = {}
    for item in selected:
        counts[item.suite] = counts.get(item.suite, 0) + 1
    summary = ", ".join(
        f"{suite_name}={count}" for suite_name, count in sorted(counts.items())
    )
    console.print(f"  Suite coverage: {summary}")


def write_aggregate_report(
    *,
    profile: str,
    suite: str | None,
    workload: str | None,
    collection: str | None = None,
    variant: str | None = None,
    workload_reports: list[dict[str, Any]],
    output_dir: Path,
    open_report: bool,
    power: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"mlperf_edu_{profile}_{timestamp}.json"
    hardware = detect_hardware()
    report = {
        "schema": "mlperf-edu-report/0.1",
        "mlperf_suite": DEFAULT_MLPERF_SUITE,
        "profile": profile,
        "suite": suite,
        "workload": workload,
        "collection": collection,
        "variant": variant,
        "selection": {
            "kind": (
                "workload"
                if workload
                else "suite"
                if suite
                else "collection"
                if collection
                else "default"
            ),
            "name": f"{workload}:{variant}"
            if workload and variant
            else workload or suite or collection or "default",
        },
        "hardware": hardware,
        "workloads": workload_reports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if power:
        report["power"] = power
    attach_run_fingerprints(report, hardware=hardware)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    exports = write_report_exports(report, report_path, open_report=open_report)
    return report, report_path, exports


def export_workload_reports(
    workload_reports: list[dict[str, Any]], workloads: dict[str, Workload] | None = None
) -> None:
    """Create HTML/CSV siblings for per-workload JSON reports.

    Runners write the measurement JSON before building provenance manifests, so
    this function deliberately does not rewrite workload JSON. It only derives
    human/spreadsheet views and records their paths in the aggregate in-memory
    report.
    """
    for item in workload_reports:
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, dict):
            if item.get("status") in {"passed", "quality_failed"}:
                console.print(
                    f"[yellow]No workload artifacts declared for:[/yellow] {item.get('workload', item.get('id', 'unknown'))}"
                )
            continue
        report_value = artifacts.get("report")
        if not report_value:
            if item.get("status") in {"passed", "quality_failed"}:
                console.print(
                    f"[yellow]No workload report path declared for:[/yellow] {item.get('workload', item.get('id', 'unknown'))}"
                )
            continue
        report_path = Path(str(report_value))
        if not report_path.exists() or not report_path.is_file():
            console.print(
                f"[yellow]Workload report missing; skipped HTML/CSV export:[/yellow] {report_path}"
            )
            continue
        try:
            # Re-read the on-disk JSON so derived views match the exact report
            # bytes that the provenance manifest binds.
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError as exc:
            console.print(
                f"[yellow]Workload report is not valid JSON; skipped HTML/CSV export:[/yellow] {report_path} ({exc})"
            )
            continue
        try:
            for field in ("device_requested", "device_executed"):
                if item.get(field) is not None:
                    report[field] = item[field]
            enrich_report_for_display(report, workloads or {})
            attach_run_fingerprints(report)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            update_measurement_manifest(
                report, report_path, artifacts.get("provenance")
            )
            exports = write_report_exports(report, report_path, open_report=False)
        except Exception as exc:
            console.print(
                f"[yellow]Could not export workload HTML/CSV:[/yellow] {report_path} ({exc})"
            )
            continue
        artifacts["html"] = str(exports["html"])
        artifacts["csv"] = str(exports["csv"])


def enrich_reports_for_display(
    reports: list[dict[str, Any]], workloads: dict[str, Workload]
) -> None:
    for report in reports:
        enrich_report_for_display(report, workloads)


def enrich_report_for_display(
    report: dict[str, Any], workloads: dict[str, Workload]
) -> None:
    strip_promoted_baseline_metadata(report)
    if "workloads" in report:
        for item in report.get("workloads", []):
            if isinstance(item, dict):
                enrich_report_for_display(item, workloads)
        return

    workload_id = report.get("workload") or report.get("id")
    if not workload_id or workload_id not in workloads:
        return
    workload = workloads[str(workload_id)]

    quality = report.setdefault("quality", {})
    if isinstance(quality, dict):
        raw_quality = (
            workload.raw.get("quality") or workload.raw.get("quality_target") or {}
        )
        legacy_required = bool(
            raw_quality.get(
                LEGACY_QUALITY_REQUIRED_FIELD, workload.public_status == "score-bearing"
            )
        )
        quality["quality_required"] = bool(
            quality_required_value(quality, legacy_required)
        )
        quality.pop(LEGACY_QUALITY_REQUIRED_FIELD, None)
        if workload.quality_direction:
            quality.setdefault("direction", workload.quality_direction)
        if workload.quality_target_basis:
            quality.setdefault("target_basis", workload.quality_target_basis)
        if workload.quality_tolerance is not None:
            quality.setdefault("tolerance", workload.quality_tolerance)
        if workload.quality_reference_runs:
            quality.setdefault("reference_runs", workload.quality_reference_runs)
        if workload.quality_acceptance_runs:
            quality.setdefault("acceptance_runs", workload.quality_acceptance_runs)
        if workload.quality_reference_protocol:
            quality.setdefault(
                "reference_protocol", copy.deepcopy(workload.quality_reference_protocol)
            )
        functional_check = workload.raw.get("functional_check")
        if isinstance(functional_check, dict):
            quality.setdefault("functional_check", copy.deepcopy(functional_check))

    performance_reference_protocol = workload.raw.get("performance_reference_protocol")
    if isinstance(performance_reference_protocol, dict):
        report.setdefault(
            "performance_reference_protocol",
            copy.deepcopy(performance_reference_protocol),
        )

    report.setdefault(
        "public",
        {
            "status": workload.public_status,
            "rationale": workload.public_rationale,
        },
    )
    report.setdefault("model", workload.model)
    report.setdefault("scenario", workload.scenario)
    if workload.dataset:
        report.setdefault("dataset", workload.dataset)
        report.setdefault(
            "dataset_asset",
            asset_dossier(
                workload.dataset, declared_source=workload.raw.get("dataset_source")
            ),
        )
    if workload.raw.get("dataset_source"):
        report.setdefault("dataset_source", workload.raw.get("dataset_source"))
    if workload.raw.get("model_source"):
        report.setdefault("model_source", workload.raw.get("model_source"))
        model_source = workload.raw.get("model_source") or {}
        if (
            isinstance(model_source, dict)
            and model_source.get("type") == "huggingface-pinned"
        ):
            model_asset = report.setdefault("model_asset", {})
            if isinstance(model_asset, dict):
                for key, value in huggingface_model_dossier(
                    model_source,
                    model_name=workload.model,
                    model_id=str(model_source.get("repo_id") or workload.model),
                ).items():
                    model_asset.setdefault(key, value)
                model_info = report.get("model")
                if isinstance(model_info, dict) and model_info.get("revision"):
                    model_asset.setdefault("revision", model_info["revision"])
    if workload.raw.get("quality_dependency"):
        report.setdefault("quality_dependency", workload.raw.get("quality_dependency"))
    if workload.raw.get("shared_checkpoint"):
        report.setdefault("shared_checkpoint", workload.raw.get("shared_checkpoint"))
        checkpoint = report.setdefault("checkpoint_provenance", {})
        if isinstance(checkpoint, dict):
            for key, value in checkpoint_provenance_for(workload, workloads).items():
                checkpoint.setdefault(key, value)
    canonical = canonical_workload_for_id(workload)
    if canonical:
        report.setdefault("canonical_workload", canonical)
        report.setdefault("variant", workload_variant_name(workload))
        report.setdefault("run_selector", workload_run_selector(workload))

    report["review_contract"] = evaluate_report_contract(workload, report)
    strip_promoted_baseline_metadata(report)


def strip_promoted_baseline_metadata(value: Any) -> None:
    """Remove registry results that would make fresh evidence circular.

    A run owns its observed metrics, target decision, protocol snapshot, and
    content-addressed lineage. Promoted multi-run aggregates belong only in the
    registry and reference-evidence summaries; copying them into a new raw run
    would make that run appear to substantiate results produced before it.
    """
    if isinstance(value, dict):
        for key in list(value):
            if key in PROMOTED_BASELINE_FIELDS:
                value.pop(key, None)
                continue
            strip_promoted_baseline_metadata(value[key])
    elif isinstance(value, list):
        for item in value:
            strip_promoted_baseline_metadata(item)


def checkpoint_provenance_for(
    workload: Workload, workloads: dict[str, Workload]
) -> dict[str, Any]:
    shared_checkpoint = workload.raw.get("shared_checkpoint")
    if not shared_checkpoint:
        return {}
    source = workloads.get(str(shared_checkpoint))
    quality_dependency = workload.raw.get("quality_dependency") or shared_checkpoint
    provenance = {
        "artifact_role": "trained-checkpoint",
        "source_workload": str(shared_checkpoint),
        "quality_dependency": str(quality_dependency),
        "source_run_selector": workload_run_selector(source)
        if source
        else str(shared_checkpoint),
        "artifact_policy": "Preserve the source training report and .provd.json alongside checkpoint-backed inference results for public review.",
    }
    if source:
        provenance.update(
            {
                "source_public_status": source.public_status,
                "source_quality_metric": source.quality_metric,
                "source_quality_target": source.quality_value,
                "source_quality_direction": source.quality_direction,
                "source_target_basis": source.quality_target_basis,
                "source_reference_runs": source.quality_reference_runs,
            }
        )
    return {key: value for key, value in provenance.items() if value not in (None, "")}


def attach_run_fingerprints(
    report: dict[str, Any], *, hardware: dict[str, Any] | None = None
) -> None:
    """Attach stable machine-readable execution fingerprints to reports."""
    if "workloads" in report:
        aggregate_hardware = hardware or report.get("hardware") or detect_hardware()
        for item in report.get("workloads", []):
            if isinstance(item, dict):
                attach_run_fingerprints(item, hardware=aggregate_hardware)
        report["run_fingerprint"] = build_run_fingerprint(
            report, hardware=aggregate_hardware
        )
        return

    report["run_fingerprint"] = build_run_fingerprint(report, hardware=hardware)


def build_run_fingerprint(
    report: dict[str, Any], *, hardware: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest = load_report_manifest(report)
    manifest_hardware = None
    if isinstance(manifest, dict):
        manifest_hardware = ((manifest.get("leaves") or {}).get("hardware") or {}).get(
            "fingerprint"
        )
        if not isinstance(manifest_hardware, dict):
            manifest_hardware = None
    hw = hardware or report.get("hardware") or manifest_hardware or detect_hardware()
    fingerprint = {
        "schema": "mlperf-edu-run-fingerprint/0.1",
        "hardware": hardware_fingerprint_summary(hw),
        "software": software_fingerprint_summary(hw),
        "execution": execution_fingerprint_summary(report),
        "comparison_fingerprint_hash_algorithm": "sha256",
        "comparison_fingerprint_hash_scope": "canonical-run-comparison-record",
    }
    asset_hashes = asset_hashes_from_manifest(manifest)
    if asset_hashes:
        fingerprint["asset_hashes"] = asset_hashes
    fingerprint["comparison_fingerprint_sha256"] = run_comparison_fingerprint_sha256(
        fingerprint
    )
    return fingerprint


def run_comparison_fingerprint_sha256(fingerprint: dict[str, Any]) -> str:
    """Hash the canonical performance-comparison context."""
    payload = json.dumps(
        run_comparison_fingerprint_record(fingerprint),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_comparison_fingerprint_record(
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Return comparison inputs while excluding paths and result outcomes."""
    hardware = dict(fingerprint.get("hardware") or {})
    # The digest fields summarize the complete detected hardware record, including
    # process environment.  Keep the explicit hardware fields below, and let the
    # normalized software environment bind comparison-relevant runtime settings.
    hardware.pop("fingerprint_hash", None)
    hardware.pop("fingerprint_sha256", None)
    software = dict(fingerprint.get("software") or {})
    software.pop("python_executable", None)
    performance_environment = software.get("performance_environment")
    if isinstance(performance_environment, dict):
        performance_environment = dict(performance_environment)
        for key in ("MLPERF_EDU_MAX_SEED", "MLPERF_EDU_SEED"):
            performance_environment.pop(key, None)
        if performance_environment:
            software["performance_environment"] = performance_environment
        else:
            software.pop("performance_environment", None)
    execution = dict(fingerprint.get("execution") or {})
    execution.pop("status", None)
    # Seeds are intentional repeated-run variables. Keep them in the complete
    # run fingerprint and provenance, but exclude them from the digest that
    # decides whether separate executions share one comparison context.
    execution.pop("seed", None)
    record = {
        "schema": fingerprint.get("schema"),
        "hardware": hardware,
        "software": software,
        "execution": execution,
        "comparison_fingerprint_hash_algorithm": fingerprint.get(
            "comparison_fingerprint_hash_algorithm"
        ),
        "comparison_fingerprint_hash_scope": fingerprint.get(
            "comparison_fingerprint_hash_scope"
        ),
    }
    asset_hashes = comparison_asset_hashes(fingerprint)
    if asset_hashes:
        record["asset_hashes"] = asset_hashes
    return record


def comparison_asset_hashes(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """Return input asset hashes that define a repeatable comparison context."""
    asset_hashes = fingerprint.get("asset_hashes")
    if not isinstance(asset_hashes, dict):
        return {}
    normalized = strip_path_fields(copy.deepcopy(asset_hashes))
    execution = fingerprint.get("execution") or {}
    scenario = str(execution.get("scenario") or "").lower()
    if scenario in {"training", "train"} and isinstance(normalized, dict):
        normalized.pop("weights", None)
    return normalized if isinstance(normalized, dict) else {}


def strip_path_fields(value: Any) -> Any:
    """Remove host-local filesystem locations from a comparison record."""
    if isinstance(value, dict):
        return {
            key: strip_path_fields(item) for key, item in value.items() if key != "path"
        }
    if isinstance(value, list):
        return [strip_path_fields(item) for item in value]
    return value


def hardware_fingerprint_summary(hardware: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "machine_model",
        "chip",
        "cpu",
        "cpu_topology",
        "gpu",
        "accelerator",
        "memory_gb",
        "cache_sizes",
        "os",
        "os_version",
        "python_version",
        "pytorch_version",
        "backend",
        "availability_detected_backend",
        "available_backends",
        "fingerprint_schema",
        "fingerprint_hash_algorithm",
        "fingerprint_hash_scope",
        "fingerprint_hash",
        "fingerprint_sha256",
    )
    return {key: hardware.get(key) for key in keys if hardware.get(key) is not None}


def software_fingerprint_summary(
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "transformers": package_version("transformers"),
        "mlperf_edu": package_version("mlperf-edu"),
    }
    if hardware:
        audio_backend = hardware.get("audio_backend")
        if audio_backend is not None:
            summary["audio_backend"] = audio_backend
        torch_runtime = hardware.get("torch_runtime")
        if isinstance(torch_runtime, dict):
            summary["torch_runtime"] = torch_runtime
        performance_environment = hardware.get("performance_environment")
        if isinstance(performance_environment, dict):
            summary["performance_environment"] = performance_environment
    return summary


def package_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def execution_fingerprint_summary(report: dict[str, Any]) -> dict[str, Any]:
    workloads = (
        report.get("workloads") if isinstance(report.get("workloads"), list) else None
    )
    if workloads:
        backends = sorted(
            {
                str(item.get("backend"))
                for item in workloads
                if isinstance(item, dict) and item.get("backend")
            }
        )
        data_modes = sorted(
            {
                str(item.get("data_mode"))
                for item in workloads
                if isinstance(item, dict) and item.get("data_mode")
            }
        )
        devices = sorted(
            {
                str(item.get("device_requested") or item.get("device"))
                for item in workloads
                if isinstance(item, dict)
                and (item.get("device_requested") or item.get("device"))
            }
        )
        executed_devices = sorted(
            {
                str(item.get("device_executed"))
                for item in workloads
                if isinstance(item, dict) and item.get("device_executed")
            }
        )
        scenarios = sorted(
            {
                str(item.get("scenario"))
                for item in workloads
                if isinstance(item, dict) and item.get("scenario")
            }
        )
        precision_records = unique_fingerprint_records(
            report_precision_summary(item)
            for item in workloads
            if isinstance(item, dict)
        )
        compilation_records = unique_fingerprint_records(
            report_compilation_summary(item)
            for item in workloads
            if isinstance(item, dict)
        )
    else:
        backends = [str(report.get("backend"))] if report.get("backend") else []
        data_modes = [str(report.get("data_mode"))] if report.get("data_mode") else []
        selected_device = report.get("device_requested") or report.get("device")
        devices = [str(selected_device)] if selected_device else []
        executed_device = report.get("device_executed")
        executed_devices = [str(executed_device)] if executed_device else []
        scenarios = [str(report.get("scenario"))] if report.get("scenario") else []
        precision_records = unique_fingerprint_records(
            [report_precision_summary(report)]
        )
        compilation_records = unique_fingerprint_records(
            [report_compilation_summary(report)]
        )
    summary = {
        "profile": report.get("profile"),
        "suite": report.get("suite"),
        "workload": report.get("workload") or report.get("id"),
        "variant": report.get("variant"),
        "seed": report.get("seed"),
        "status": report.get("status"),
        "scenario": scenarios[0] if len(scenarios) == 1 else None,
        "scenarios": scenarios,
        # ``backends`` is retained for report compatibility. The explicit name
        # distinguishes execution selection from hardware availability.
        "backends": backends,
        "report_selected_backends": backends,
        "report_selected_devices": devices,
        "report_executed_devices": executed_devices,
        "data_modes": data_modes,
        "report_selected_precision": precision_records,
        "report_selected_compilation": compilation_records,
    }
    return {key: value for key, value in summary.items() if value not in (None, [], "")}


def report_precision_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return explicit report precision fields without inferring model dtype."""
    keys = (
        "dtype",
        "precision",
        "mixed_precision",
        "amp",
        "autocast",
        "quantization",
    )
    precision = {key: report[key] for key in keys if report.get(key) is not None}
    for container_name in ("configuration", "metrics"):
        container = report.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key not in precision and container.get(key) is not None:
                precision[key] = container[key]
    return precision


def report_compilation_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return compilation mode only when the workload report exposes it."""
    compilation: dict[str, Any] = {}
    nested = report.get("compilation")
    nested_keys = ("enabled", "mode", "backend", "fullgraph", "dynamic")
    if isinstance(nested, dict):
        compilation.update(
            {key: nested[key] for key in nested_keys if nested.get(key) is not None}
        )
    elif nested is not None:
        compilation["value"] = nested

    aliases = {
        "compiled": "enabled",
        "torch_compile": "torch_compile",
        "compile_mode": "mode",
        "compilation_mode": "mode",
        "compile_backend": "backend",
    }
    for source, destination in aliases.items():
        if destination not in compilation and report.get(source) is not None:
            compilation[destination] = report[source]
    configuration = report.get("configuration")
    if isinstance(configuration, dict):
        for source, destination in aliases.items():
            if destination not in compilation and configuration.get(source) is not None:
                compilation[destination] = configuration[source]
    return compilation


def unique_fingerprint_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate JSON records while preserving a canonical order."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record:
            continue
        canonical = json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        indexed[canonical] = record
    return [indexed[key] for key in sorted(indexed)]


def load_report_manifest(report: dict[str, Any]) -> dict[str, Any] | None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    manifest_value = artifacts.get("provenance")
    if not manifest_value:
        return None
    path = Path(str(manifest_value))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def asset_hashes_from_manifest(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    leaves = manifest.get("leaves") or {}
    hashes: dict[str, Any] = {}

    weights = leaves.get("weights") or {}
    if weights.get("sha256"):
        hashes["weights"] = {
            "path": weights.get("path"),
            "sha256": weights.get("sha256"),
            "n_bytes": weights.get("n_bytes"),
        }
    elif weights.get("files"):
        hashes["weights"] = {
            "format": weights.get("format"),
            "name": weights.get("name"),
            "revision": weights.get("revision"),
            "merkle_root": weights.get("merkle_root"),
            "file_count": weights.get("file_count"),
            "n_bytes": weights.get("n_bytes"),
            "files": [
                {
                    "logical_path": item.get("logical_path"),
                    "role": item.get("role"),
                    "sha256": item.get("sha256"),
                    "n_bytes": item.get("n_bytes"),
                }
                for item in weights.get("files") or []
                if isinstance(item, dict)
            ],
        }

    dataset = leaves.get("dataset") or {}
    dataset_hashes: dict[str, Any] = {}
    if dataset.get("merkle_root"):
        dataset_hashes["merkle_root"] = dataset.get("merkle_root")
    files = []
    for item in dataset.get("files") or []:
        if isinstance(item, dict) and item.get("sha256"):
            files.append(
                {
                    "path": item.get("path"),
                    "sha256": item.get("sha256"),
                    "n_bytes": item.get("n_bytes"),
                }
            )
    if files:
        dataset_hashes["files"] = files
    if dataset_hashes:
        if dataset.get("name"):
            dataset_hashes["name"] = dataset.get("name")
        hashes["dataset"] = dataset_hashes

    return hashes


def update_measurement_manifest(
    report: dict[str, Any], report_path: Path, manifest_value: Any
) -> None:
    if not manifest_value:
        return
    manifest_path = Path(str(manifest_value))
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return
    leaves = manifest.get("leaves")
    if not isinstance(leaves, dict):
        return
    leaves["measurement"] = measurement_leaf(report, report_path)
    manifest["merkle_root"] = merkle_root(leaves)
    manifest["integrity"] = integrity_record(manifest["merkle_root"])
    manifest.pop("signature", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def print_run_summary(
    profile: str,
    workload_reports: list[dict[str, Any]],
    report_path: Path,
    exports: dict[str, Path],
) -> int:
    quality_passed = sum(
        1
        for item in workload_reports
        if item.get("status") == "passed"
        and quality_required_value(item.get("quality") or {}, False) is True
        and (item.get("quality") or {}).get("target_met") is True
    )
    functional_passed = sum(
        1
        for item in workload_reports
        if item.get("status") == "passed"
        and quality_required_value(item.get("quality") or {}, False) is not True
    )
    definition_only = sum(
        1 for item in workload_reports if item.get("status") == "definition_valid"
    )
    unsupported = sum(
        1 for item in workload_reports if item.get("status") == "not_implemented"
    )
    quality_failed = sum(
        1
        for item in workload_reports
        if item.get("status") == "quality_failed"
        or (
            item.get("status") == "passed"
            and quality_required_value(item.get("quality") or {}, False) is True
            and (item.get("quality") or {}).get("target_met") is not True
        )
    )
    console.print(
        f"[green]{profile} run complete[/green]: "
        f"{quality_passed} quality-passed, {functional_passed} functional-passed, "
        f"{definition_only} definition-only, {unsupported} unsupported, "
        f"{quality_failed} quality-failed"
    )
    console.print(f"Report: {report_path}")
    console.print(f"HTML: {exports['html']}")
    console.print(f"CSV: {exports['csv']}")
    for item in workload_reports:
        artifacts = item.get("artifacts") or {}
        if artifacts.get("report"):
            console.print(f"Workload report: {artifacts['report']}")
        if artifacts.get("html"):
            console.print(f"Workload HTML: {artifacts['html']}")
        if artifacts.get("csv"):
            console.print(f"Workload CSV: {artifacts['csv']}")
        if artifacts.get("provenance"):
            console.print(f"Provenance: {artifacts['provenance']}")
    if quality_failed:
        return 1
    if unsupported:
        return 2
    return 0


def resolve_execution_selection(
    workload: Workload, *, mode: str | None, phase: str | None
) -> tuple[str | None, str | None]:
    implemented_modes = tuple(workload.raw.get("implemented_modes") or ())
    if not implemented_modes:
        if mode or phase:
            raise ValueError(
                f"workload {workload.id!r} does not expose selectable modes"
            )
        return None, None
    resolved_mode = mode or workload.raw.get("default_mode") or implemented_modes[0]
    if resolved_mode not in implemented_modes:
        raise ValueError(
            f"workload {workload.id!r} does not implement mode {resolved_mode!r}; "
            f"choose one of {', '.join(implemented_modes)}"
        )
    phases = tuple((workload.raw.get("phases") or {}).get(resolved_mode) or ())
    resolved_phase = (
        phase or workload.raw.get("default_phase")
        if resolved_mode == "inference"
        else phase
    )
    if resolved_phase and resolved_mode != "inference":
        raise ValueError("--phase is valid only for inference mode")
    if resolved_phase and resolved_phase not in phases:
        raise ValueError(
            f"workload {workload.id!r} does not implement phase {resolved_phase!r}; "
            f"choose one of {', '.join(phases)}"
        )
    return str(resolved_mode), str(resolved_phase) if resolved_phase else None


def run_workload(
    workload: Workload,
    profile: str,
    output_dir: Path,
    *,
    mode: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    validate_requested_torch_device()
    resolved_mode, resolved_phase = resolve_execution_selection(
        workload, mode=mode, phase=phase
    )
    runner = load_runner(workload, profile)
    if runner:
        parameters = inspect.signature(runner).parameters
        execution_kwargs: dict[str, str] = {}
        if "mode" in parameters:
            execution_kwargs["mode"] = resolved_mode
        if resolved_phase is not None and "phase" in parameters:
            execution_kwargs["phase"] = resolved_phase
        report = runner(workload, output_dir, **execution_kwargs)
        report["mode"] = resolved_mode
        report["phase"] = resolved_phase
        return report

    if profile == "pro" and load_runner(workload, "max"):
        report = run_pro_profile(
            workload, output_dir, mode=resolved_mode, phase=resolved_phase
        )
        report["mode"] = resolved_mode
        report["phase"] = resolved_phase
        return report

    if profile == "min":
        report = smoke_workload(workload)
        report["mode"] = resolved_mode
        report["phase"] = resolved_phase
        return report

    unsupported = smoke_workload(workload)
    unsupported["profile"] = profile
    unsupported["status"] = "not_implemented"
    unsupported["note"] = f"No {profile} runner is registered for this workload."
    unsupported["mode"] = resolved_mode
    unsupported["phase"] = resolved_phase
    return unsupported


def validate_requested_torch_device() -> None:
    """Reject unavailable or unsupported explicit PyTorch device requests."""
    requested = os.environ.get("MLPERF_EDU_DEVICE")
    if not requested:
        return
    normalized = requested.strip().lower()
    try:
        import torch
    except Exception as exc:
        raise ValueError(
            f"Requested device {requested!r}, but PyTorch could not be loaded: {exc}. "
            "Run 'mlperf doctor' to inspect the environment."
        ) from exc

    available = ["cpu"]
    if torch.cuda.is_available():
        available.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available.append("mps")

    device_family = normalized.split(":", 1)[0]
    if device_family not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Requested device {requested!r} is not supported by MLPerf EDU. "
            f"Available PyTorch devices: {', '.join(available)}. Set "
            "MLPERF_EDU_DEVICE to one of those values, or run 'mlperf doctor'."
        )
    if device_family not in available:
        raise ValueError(
            f"Requested device {requested!r} is unavailable in this PyTorch "
            f"environment. Available PyTorch devices: {', '.join(available)}. "
            "Choose an available device with MLPERF_EDU_DEVICE, or run "
            "'mlperf doctor' for details."
        )
    try:
        device = torch.device(normalized)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Requested device {requested!r} is invalid. Available PyTorch "
            f"devices: {', '.join(available)}."
        ) from exc
    if device_family == "cuda" and device.index is not None:
        device_count = torch.cuda.device_count()
        if device.index < 0 or device.index >= device_count:
            raise ValueError(
                f"Requested device {requested!r} is unavailable; this environment "
                f"reports {device_count} CUDA device(s). Choose a valid CUDA index, "
                "use MLPERF_EDU_DEVICE=cpu, or run 'mlperf doctor'."
            )


def annotate_execution_device(report: dict[str, Any]) -> None:
    """Record requested and executed devices without conflating availability."""
    if not report.get("device_requested"):
        report["device_requested"] = os.environ.get("MLPERF_EDU_DEVICE", "auto").lower()
    if report.get("device_executed"):
        return
    if report.get("device"):
        report["device_executed"] = str(report["device"]).lower()
        return
    backend = str(report.get("backend") or "").lower()
    if backend.startswith("pytorch-") and len(backend) > len("pytorch-"):
        report["device_executed"] = backend.removeprefix("pytorch-")


def run_pro_profile(
    workload: Workload,
    output_dir: Path,
    *,
    mode: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    max_runner = load_runner(workload, "max")
    if not max_runner:
        raise ValueError(f"No max runner is registered for pro fallback: {workload.id}")

    repetitions = int(os.environ.get("MLPERF_EDU_PRO_REPETITIONS", "1"))
    if repetitions < 1:
        raise ValueError("MLPERF_EDU_PRO_REPETITIONS must be >= 1")

    start = time.perf_counter()
    subreports: list[dict[str, Any]] = []
    for idx in range(repetitions):
        rep_dir = output_dir / ".pro_evidence" / workload.id / f"rep{idx + 1}"
        parameters = inspect.signature(max_runner).parameters
        execution_kwargs: dict[str, str] = {}
        if mode is not None and "mode" in parameters:
            execution_kwargs["mode"] = mode
        if phase is not None and "phase" in parameters:
            execution_kwargs["phase"] = phase
        with TemporaryNanogptCheckpoint(output_dir):
            subreport = max_runner(workload, rep_dir, **execution_kwargs)
        publish_shared_checkpoint(workload, subreport, output_dir)
        subreports.append(subreport)
    wall_time = time.perf_counter() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (output_dir / f"{workload.id}_pro_report.json").resolve()
    manifest_path = (output_dir / f"{workload.id}_pro.provd.json").resolve()
    execution_passed = all(item.get("status") == "passed" for item in subreports)
    subrun_qualities = [item.get("quality") or {} for item in subreports]
    quality_required = all(
        quality_required_value(quality, False) is True
        for quality in subrun_qualities
    )
    target_met = (
        all(quality.get("target_met") is True for quality in subrun_qualities)
        if quality_required
        else None
    )
    status = (
        "passed"
        if execution_passed and (not quality_required or target_met is True)
        else "quality_failed"
    )
    readiness_stage = "quality" if quality_required else "functional"
    metrics = aggregate_pro_metrics(subreports)
    metrics["repetitions"] = repetitions
    metrics["wall_time_seconds"] = float(wall_time)
    report = {
        "schema": "mlperf-edu-report/0.1",
        "id": workload.id,
        "workload": workload.id,
        "suite": workload.suite,
        "profile": "pro",
        "mode": mode,
        "phase": phase,
        "status": status,
        "backend": common_field(subreports, "backend"),
        "data_mode": common_field(subreports, "data_mode"),
        "pro_policy": {
            "mode": "max-repetition",
            "repetitions": repetitions,
            "note": "The pro profile runs the max runner once by default. Optional repetitions are reserved for later stability studies.",
        },
        "metrics": metrics,
        "quality": {
            "metric": workload.quality_metric,
            "target": workload.quality_value,
            "quality_required": quality_required,
            "target_met": target_met,
            "note": (
                "Every max subrun executed the authoritative quality contract and met its target."
                if quality_required and target_met
                else "The max subruns completed, but they did not execute an authoritative quality contract."
                if not quality_required
                else "At least one authoritative max subrun did not meet its quality target."
            ),
        },
        "readiness_stage": readiness_stage,
        "subruns": [subrun_evidence(item) for item in subreports],
        "artifacts": {
            "report": str(report_path),
            "provenance": str(manifest_path),
        },
    }
    if not quality_required:
        report["functional_readiness"] = {
            "schema": "mlperf-edu-functional-readiness/0.1",
            "stage": "functional",
            "end_to_end_execution": True,
            "authoritative_quality_contract_executed": False,
            "repeatability_verified": repetitions > 1,
            "promotion_eligible": False,
            "next_stage": "quality-conformance",
        }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    manifest = build_provd(
        workload=workload.id,
        scenario=workload.scenario or "pro",
        division="open",
        hardware_fingerprint=detect_hardware(),
        report=report,
        report_path=report_path,
        dataset_name=workload.dataset or "pro-aggregate",
        dataset_files=[],
        rng_seed=None,
        repo_root=find_project_root(),
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return report


class TemporaryNanogptCheckpoint:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.previous: str | None = None
        self.changed = False

    def __enter__(self):
        checkpoint = (
            self.output_dir / "causal-language-modeling_training_max_checkpoint.pt"
        )
        if checkpoint.exists() and "MLPERF_EDU_NANOGPT_CHECKPOINT" not in os.environ:
            self.previous = os.environ.get("MLPERF_EDU_NANOGPT_CHECKPOINT")
            os.environ["MLPERF_EDU_NANOGPT_CHECKPOINT"] = str(checkpoint)
            self.changed = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.changed:
            return
        if self.previous is None:
            os.environ.pop("MLPERF_EDU_NANOGPT_CHECKPOINT", None)
        else:
            os.environ["MLPERF_EDU_NANOGPT_CHECKPOINT"] = self.previous


def publish_shared_checkpoint(
    workload: Workload, report: dict[str, Any], output_dir: Path
) -> None:
    if workload.id != "causal-language-modeling" or report.get("mode") != "training":
        return
    artifacts = report.get("artifacts") or {}
    copies = {
        "checkpoint": "causal-language-modeling_training_max_checkpoint.pt",
        "report": "causal-language-modeling_training_max_report.json",
        "provenance": "causal-language-modeling_training_max.provd.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, filename in copies.items():
        source_value = artifacts.get(role)
        if not source_value:
            continue
        source = Path(str(source_value))
        if source.is_file():
            shutil.copy2(source, output_dir / filename)


def aggregate_pro_metrics(subreports: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for report in subreports:
        for key, value in (report.get("metrics") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values.setdefault(key, []).append(float(value))

    metrics: dict[str, Any] = {}
    for key, series in sorted(values.items()):
        metrics[f"{key}_mean"] = float(sum(series) / len(series))
        metrics[f"{key}_min"] = float(min(series))
        metrics[f"{key}_max"] = float(max(series))
    return metrics


def common_field(reports: list[dict[str, Any]], field: str) -> str:
    values = {str(report.get(field, "")) for report in reports if report.get(field)}
    if len(values) == 1:
        return next(iter(values))
    if values:
        return "mixed"
    return ""


def subrun_evidence(report: dict[str, Any]) -> dict[str, Any]:
    artifacts = report.get("artifacts") or {}
    evidence: dict[str, Any] = {
        "workload": report.get("workload", report.get("id")),
        "profile": report.get("profile"),
        "status": report.get("status"),
        "artifacts": artifacts,
    }
    quality = report.get("quality") or {}
    evidence["quality_required"] = quality_required_value(quality, False)
    evidence["target_met"] = quality.get("target_met")
    if report.get("functional_readiness"):
        evidence["functional_readiness"] = report["functional_readiness"]
    for name, path in artifacts.items():
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            digest, n_bytes = sha256_file_for_report(file_path)
            evidence[f"{name}_sha256"] = f"sha256:{digest}"
            evidence[f"{name}_n_bytes"] = n_bytes
    return evidence


def sha256_file_for_report(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n_bytes = 0
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
            n_bytes += len(data)
    return h.hexdigest(), n_bytes


def load_runner(workload: Workload, profile: str):
    runner = (workload.raw.get("runner") or {}).get(profile)
    if not runner:
        return None
    try:
        module_name, function_name = runner.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"invalid runner spec for {workload.id}: {runner}") from exc

    module = import_module(module_name)
    return getattr(module, function_name)


def smoke_workload(workload: Workload) -> dict[str, Any]:
    return {
        "id": workload.id,
        "suite": workload.suite,
        "maturity": workload.maturity,
        "model": workload.model,
        "dataset": workload.dataset,
        "quality_metric": workload.quality_metric,
        "quality_value": workload.quality_value,
        "status": "definition_valid",
    }


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    if not path.exists():
        console.print(f"[red]Manifest not found:[/red] {path}")
        return 1
    try:
        result = verify_provd(path, repo_root=find_project_root())
    except (OSError, ValueError) as exc:
        console.print(f"[red]Invalid provenance manifest:[/red] {exc}")
        return 1
    print_verification_checks(result)

    if result.all_ok:
        console.print("[green]verified[/green]")
        return 0
    console.print("[red]verification failed[/red]")
    return 1


def cmd_report(args: argparse.Namespace) -> int:
    path = resolve_report_input(Path(args.report))
    if not path.exists():
        console.print(f"[red]Report not found:[/red] {path}")
        return 1
    data = json.loads(path.read_text())
    try:
        enrich_report_for_display(data, load_workloads(args))
    except (FileNotFoundError, ValueError):
        pass
    attach_run_fingerprints(data)
    if args.format == "json":
        text = json.dumps(data, indent=2, sort_keys=True) + "\n"
        if args.output:
            output = Path(args.output).resolve()
            output.write_text(text)
            console.print(f"JSON: {output}")
        else:
            console.print(text)
        return 0
    if args.format == "csv":
        output = (
            Path(args.output).resolve() if args.output else path.with_suffix(".csv")
        )
        write_csv_report(data, output)
        console.print(f"CSV: {output}")
        return 0
    if args.format == "html":
        output = (
            Path(args.output).resolve() if args.output else path.with_suffix(".html")
        )
        write_html_report(data, output, source_path=path)
        console.print(f"HTML: {output}")
        if args.open:
            open_report_path(output)
        return 0

    console.print(f"schema: {data.get('schema', 'unknown')}")
    console.print(f"mlperf_suite: {data.get('mlperf_suite', DEFAULT_MLPERF_SUITE)}")
    console.print(f"profile: {data.get('profile', 'unknown')}")
    if data.get("power"):
        power = data["power"]
        console.print(f"power_average_watts: {power.get('average_watts')}")
        console.print(f"energy_joules: {power.get('energy_joules')}")
    if "workloads" in data:
        console.print(f"workloads: {len(data.get('workloads', []))}")
        status_counts: dict[str, int] = {}
        for item in data.get("workloads", []):
            status = str(item.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        for status, count in sorted(status_counts.items()):
            console.print(f"{status}: {count}")
        return 0

    console.print(f"workload: {data.get('workload', data.get('id', 'unknown'))}")
    console.print(f"status: {data.get('status', 'unknown')}")
    metrics = data.get("metrics") or {}
    quality = data.get("quality") or {}
    metric_name = quality_metric_name(quality)
    metric_key = metric_key_for_quality(metric_name, metrics)
    if metric_key:
        console.print(f"{metric_key}: {metrics[metric_key]}")
    for key in (
        "time_to_first_token_s",
        "inter_token_latency_s",
        "prefill_tokens_per_sec",
        "output_tokens_per_sec",
        "requests_per_sec",
        "batch_size",
        "context_tokens",
        "total_context_tokens",
    ):
        if key in metrics and key != metric_key:
            console.print(f"{key}: {metrics[key]}")
    if "target_met" in quality:
        console.print(f"target_met: {quality['target_met']}")
    if "quality_required" in quality:
        console.print(f"quality_required: {quality['quality_required']}")
    return 0


def resolve_report_input(path: Path) -> Path:
    if not path.is_dir():
        return path
    candidates = sorted(
        path.glob("mlperf_edu_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    validation_candidates = sorted(
        path.glob("mlperf_validate_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if validation_candidates:
        return validation_candidates[0]
    workload_candidates = sorted(
        path.glob("*_report.json"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    if workload_candidates:
        return workload_candidates[0]
    return path / "mlperf_edu_<timestamp>.json"


def write_report_exports(
    report: dict[str, Any], report_path: Path, *, open_report: bool = False
) -> dict[str, Path]:
    csv_path = report_path.with_suffix(".csv")
    html_path = report_path.with_suffix(".html")
    write_csv_report(report, csv_path)
    write_html_report(report, html_path, source_path=report_path)
    if open_report:
        open_report_path(html_path)
    return {"csv": csv_path, "html": html_path}


def open_report_path(path: Path) -> bool:
    try:
        return bool(webbrowser.open(path.as_uri()))
    except Exception as exc:
        console.print(f"[yellow]Could not open report automatically:[/yellow] {exc}")
        console.print(f"Open manually: {path}")
        return False


def write_csv_report(report: dict[str, Any], output: Path) -> None:
    rows = report_rows(report)
    rows.extend(aggregate_csv_rows(report))
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "canonical_workload",
        "variant",
        "run_selector",
        "suite",
        "profile",
        "status",
        "backend",
        "data_mode",
        "dataset",
        "dataset_license_status",
        "dataset_public_release_status",
        "dataset_public_use",
        "dataset_release_next_step",
        "model_source",
        "model_license",
        "model_rationale",
        "shared_checkpoint",
        "quality_dependency",
        "checkpoint_source_selector",
        "checkpoint_source_quality",
        "checkpoint_artifact_policy",
        "metric",
        "value",
        "target",
        "target_basis",
        "reference_runs",
        "acceptance_runs",
        "reference_statistic",
        "reference_protocol",
        "direction",
        "quality_required",
        "target_met",
        "functional_check",
        "duration_seconds",
        "throughput",
        "power_average_watts",
        "energy_joules",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def aggregate_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    power = report.get("power") or {}
    workloads = report.get("workloads") or []
    if not power or len(workloads) <= 1:
        return []
    return [
        {
            "workload": "__aggregate__",
            "suite": report.get("suite") or "aggregate",
            "profile": report.get("profile", ""),
            "status": "aggregate",
            "metric": "power",
            "power_average_watts": power.get("average_watts", ""),
            "energy_joules": power.get("energy_joules", ""),
        }
    ]


def write_html_report(
    report: dict[str, Any], output: Path, *, source_path: Path
) -> None:
    rows = report_rows(report)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    title = "MLPerf EDU Report"
    if len(rows) == 1:
        title = f"MLPerf EDU Report: {rows[0].get('workload', 'unknown')}"
    elif report.get("suite"):
        title = (
            f"MLPerf EDU Suite Report: {report.get('suite')} / {report.get('profile')}"
        )
    elif (report.get("selection") or {}).get("kind") == "default":
        title = f"MLPerf EDU Default Report: {report.get('profile')}"

    generated_at = report.get("generated_at") or datetime.now(timezone.utc).isoformat()
    status_cards = "\n".join(
        f"<div class='card'><div class='label'>{escape(status)}</div><div class='value'>{count}</div></div>"
        for status, count in sorted(status_counts.items())
    )
    power = report.get("power") or {}
    if power:
        status_cards += (
            f"\n<div class='card'><div class='label'>Average Watts</div><div class='value'>{escape(format_cell(power.get('average_watts')))}</div></div>"
            f"\n<div class='card'><div class='label'>Energy Joules</div><div class='value'>{escape(format_cell(power.get('energy_joules')))}</div></div>"
        )
    hardware_html = hardware_section_html(report)
    serving_html = serving_metrics_section_html(report)
    assets_html = assets_section_html(report)
    body_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.get('workload', '')))}</td>"
        f"<td>{escape(str(row.get('run_selector', '')))}</td>"
        f"<td>{escape(str(row.get('suite', '')))}</td>"
        f"<td>{escape(str(row.get('profile', '')))}</td>"
        f"<td><span class='badge {status_class(str(row.get('status', '')))}'>{escape(str(row.get('status', '')))}</span></td>"
        f"<td>{escape(str(row.get('metric', '')))}</td>"
        f"<td>{escape(format_cell(row.get('value')))}</td>"
        f"<td>{escape(format_cell(row.get('target')))}</td>"
        f"<td>{escape(format_cell(row.get('target_basis')))}</td>"
        f"<td>{escape(format_cell(row.get('reference_runs')))}</td>"
        f"<td>{escape(format_cell(row.get('acceptance_runs')))}</td>"
        f"<td>{escape(format_cell(row.get('reference_statistic')))}</td>"
        f"<td>{escape(format_cell(row.get('reference_protocol')))}</td>"
        f"<td>{escape(format_cell(row.get('quality_required')))}</td>"
        f"<td>{escape(format_cell(row.get('target_met')))}</td>"
        f"<td>{escape(format_cell(row.get('functional_check')))}</td>"
        f"<td>{escape(format_cell(row.get('duration_seconds')))}</td>"
        f"<td>{escape(format_cell(row.get('throughput')))}</td>"
        "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d0d5dd;
      --surface: #ffffff;
      --pass: #067647;
      --fail: #b42318;
      --warn: #b54708;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 48px; }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 13px; text-align: right; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 18px 0 24px; }}
    .section {{ margin: 0 0 24px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; background: #eef2f6; }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; font-weight: 600; }}
    .pass {{ color: var(--pass); background: #dcfae6; }}
    .fail {{ color: var(--fail); background: #fee4e2; }}
    .warn {{ color: var(--warn); background: #fef0c7; }}
    .note {{ color: var(--muted); margin-top: 18px; font-size: 12px; }}
    .table-scroll {{ overflow-x: auto; }}
    .table-scroll table {{ min-width: 980px; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>{escape(title)}</h1>
      <div class="note">Schema: {escape(str(report.get("schema", "unknown")))}</div>
    </div>
    <div class="meta">
      <div>{escape(str(generated_at))}</div>
      <div>{escape(str(source_path))}</div>
    </div>
  </header>
  <section class="grid">{status_cards}</section>
  {hardware_html}
  {serving_html}
  {assets_html}
  <section>
    <div class="table-scroll"><table>
      <thead>
        <tr><th>Workload</th><th>Run As</th><th>Suite</th><th>Profile</th><th>Status</th><th>Metric</th><th>Value</th><th>Target</th><th>Basis</th><th>Reference Runs</th><th>Acceptance Runs</th><th>Reference Statistic</th><th>Reference Protocol</th><th>Quality Required</th><th>Met</th><th>Check</th><th>Duration</th><th>Throughput</th></tr>
      </thead>
      <tbody>{body_rows}</tbody>
    </table></div>
  </section>
  <div class="note">Generated by mlperf report. Use the paired .provd.json manifests for integrity verification.</div>
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)


SERVING_METRIC_KEYS = (
    "batch_size",
    "configured_context_tokens",
    "context_tokens",
    "total_context_tokens",
    "generated_tokens",
    "total_generated_tokens",
    "time_to_first_token_s",
    "inter_token_latency_s",
    "prefill_tokens_per_sec",
    "output_tokens_per_sec",
    "requests_per_sec",
    "model_state_bytes",
)


def serving_metrics_section_html(report: dict[str, Any]) -> str:
    rows = []
    for item in report_items(report):
        metrics = item.get("metrics") or {}
        if not any(key in metrics for key in SERVING_METRIC_KEYS):
            continue
        base = report_row(
            item,
            default_profile=report.get("profile"),
            default_suite=report.get("suite"),
            default_power=None,
        )
        metric_cells = "".join(
            f"<td>{escape(format_cell(metrics.get(key)))}</td>"
            for key in SERVING_METRIC_KEYS
        )
        rows.append(
            "<tr>"
            f"<td>{escape(str(base.get('workload', '')))}</td>"
            f"<td>{escape(str(base.get('run_selector', '')))}</td>"
            f"{metric_cells}"
            "</tr>"
        )
    if not rows:
        return ""

    metric_headers = "".join(f"<th>{escape(key)}</th>" for key in SERVING_METRIC_KEYS)
    body = "\n".join(rows)
    return f"""
  <section class="section">
    <h2>Serving Metrics</h2>
    <div class="table-scroll"><table>
      <thead>
        <tr><th>Workload</th><th>Run As</th>{metric_headers}</tr>
      </thead>
      <tbody>{body}</tbody>
    </table></div>
    <div class="note">Generation-style workloads report latency, context, throughput, and model-state metrics. The paired JSON contains the full metric record.</div>
  </section>
"""


def hardware_section_html(report: dict[str, Any]) -> str:
    run_fingerprint = report.get("run_fingerprint") or {}
    hardware = report.get("hardware") or run_fingerprint.get("hardware") or {}
    rows = report_rows(report)
    backends = sorted(
        {str(row.get("backend", "")) for row in rows if row.get("backend")}
    )
    if not hardware and not backends:
        return ""

    details: list[tuple[str, Any]] = []
    for key in (
        "machine_model",
        "chip",
        "cpu",
        "gpu",
        "memory_gb",
        "os",
        "os_version",
        "python_version",
        "pytorch_version",
        "backend",
        "availability_detected_backend",
        "available_backends",
        "fingerprint_hash",
        "fingerprint_sha256",
        "machine_class",
        "platform",
        "processor",
        "python",
        "torch",
    ):
        if key in hardware:
            details.append((key, hardware.get(key)))
    if backends:
        details.append(("report_selected_backends", ", ".join(backends)))
    software = run_fingerprint.get("software") or {}
    for key in (
        "torchvision",
        "transformers",
        "mlperf_edu",
        "torch_runtime",
        "performance_environment",
    ):
        if software.get(key) and key not in hardware:
            details.append((key, software.get(key)))
    if not details and hardware:
        for key, value in sorted(hardware.items())[:8]:
            details.append((key, value))

    body = "\n".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(format_hardware_value(value))}</td></tr>"
        for key, value in details
    )
    return f"""
  <section class="section">
    <h2>Hardware and Backend</h2>
    <table>
      <tbody>{body}</tbody>
    </table>
  </section>
"""


def assets_section_html(report: dict[str, Any]) -> str:
    rows = report_rows(report)
    asset_rows = []
    for row in rows:
        if not any(
            row.get(key)
            for key in (
                "dataset",
                "dataset_license_status",
                "dataset_public_release_status",
                "model_source",
                "model_license",
                "model_rationale",
                "shared_checkpoint",
                "quality_dependency",
                "checkpoint_source_selector",
                "checkpoint_source_quality",
                "checkpoint_artifact_policy",
            )
        ):
            continue
        asset_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('workload', '')))}</td>"
            f"<td>{escape(format_cell(row.get('dataset')))}</td>"
            f"<td>{escape(format_cell(row.get('dataset_license_status')))}</td>"
            f"<td>{escape(format_cell(row.get('dataset_public_release_status')))}</td>"
            f"<td>{escape(format_cell(row.get('dataset_public_use')))}</td>"
            f"<td>{escape(format_cell(row.get('dataset_release_next_step')))}</td>"
            f"<td>{escape(format_cell(row.get('model_source')))}</td>"
            f"<td>{escape(format_cell(row.get('model_license')))}</td>"
            f"<td>{escape(format_cell(row.get('model_rationale')))}</td>"
            f"<td>{escape(format_cell(row.get('shared_checkpoint')))}</td>"
            f"<td>{escape(format_cell(row.get('quality_dependency')))}</td>"
            f"<td>{escape(format_cell(row.get('checkpoint_source_selector')))}</td>"
            f"<td>{escape(format_cell(row.get('checkpoint_source_quality')))}</td>"
            f"<td>{escape(format_cell(row.get('checkpoint_artifact_policy')))}</td>"
            "</tr>"
        )
    if not asset_rows:
        return ""
    body = "\n".join(asset_rows)
    return f"""
  <section class="section">
    <h2>Assets and Provenance</h2>
    <table>
      <thead>
        <tr><th>Workload</th><th>Dataset</th><th>Dataset Terms</th><th>Release Status</th><th>Public Use</th><th>Next Step</th><th>Model Source</th><th>Model License</th><th>Model Rationale</th><th>Checkpoint</th><th>Quality Dependency</th><th>Checkpoint Source</th><th>Source Quality</th><th>Checkpoint Policy</th></tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </section>
"""


def format_hardware_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return format_cell(value)


def report_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("workloads"), list):
        return report["workloads"]
    return [report]


def report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "workloads" in report:
        workload_items = report_items(report)
        aggregate_power = report.get("power") if len(workload_items) == 1 else None
        return [
            report_row(
                item,
                default_profile=report.get("profile"),
                default_suite=report.get("suite"),
                default_power=aggregate_power,
            )
            for item in workload_items
        ]
    return [
        report_row(
            report,
            default_profile=report.get("profile"),
            default_suite=report.get("suite"),
            default_power=report.get("power"),
        )
    ]


def report_row(
    item: dict[str, Any],
    *,
    default_profile: str | None,
    default_suite: str | None,
    default_power: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = item.get("metrics") or {}
    quality = item.get("quality") or {}
    checkpoint_provenance = (
        item.get("checkpoint_provenance")
        if isinstance(item.get("checkpoint_provenance"), dict)
        else {}
    )
    dataset_asset = (
        item.get("dataset_asset") if isinstance(item.get("dataset_asset"), dict) else {}
    )
    model_asset = (
        item.get("model_asset") if isinstance(item.get("model_asset"), dict) else {}
    )
    metric_name = quality_metric_name(quality)
    metric_key = metric_key_for_quality(metric_name, metrics)
    throughput_key = metric_key_for_throughput(metrics)
    dataset_value = dataset_name_for_row(item.get("dataset"))
    model_source = model_asset.get("source_url") or model_source_summary(
        item.get("model_source")
    )
    model_rationale = model_asset.get("selection_rationale", "")
    workload_id = str(item.get("workload", item.get("id", "")))
    canonical = (
        item.get("canonical_workload") or canonical_workload_for_id(workload_id) or ""
    )
    variant = item.get("variant") or (
        workload_variant_name(workload_id) if canonical else ""
    )
    run_selector = item.get("run_selector") or (
        f"{canonical} --variant {variant}" if canonical and variant else workload_id
    )
    quality_required = quality_required_value(quality, "")
    return {
        "workload": workload_id,
        "canonical_workload": canonical,
        "variant": variant,
        "run_selector": run_selector,
        "suite": item.get("suite", default_suite or ""),
        "profile": item.get("profile", default_profile or ""),
        "status": item.get("status", ""),
        "backend": item.get("backend", ""),
        "data_mode": item.get("data_mode", ""),
        "dataset": dataset_value,
        "dataset_license_status": dataset_asset.get("license_status", ""),
        "dataset_public_release_status": dataset_asset.get("public_release_status", ""),
        "dataset_public_use": dataset_asset.get("public_result_use", ""),
        "dataset_release_next_step": dataset_asset.get("release_next_step", ""),
        "model_source": model_source,
        "model_license": model_asset.get("license", ""),
        "model_rationale": model_rationale,
        "shared_checkpoint": item.get("shared_checkpoint", ""),
        "quality_dependency": item.get("quality_dependency", ""),
        "checkpoint_source_selector": checkpoint_provenance.get(
            "source_run_selector", ""
        ),
        "checkpoint_source_quality": checkpoint_source_quality_summary(
            checkpoint_provenance
        ),
        "checkpoint_artifact_policy": checkpoint_provenance.get("artifact_policy", ""),
        "metric": metric_key or metric_name or "",
        "value": metrics.get(metric_key) if metric_key else "",
        "target": quality.get("target", ""),
        "target_basis": quality.get("target_basis", ""),
        "reference_runs": quality.get("reference_runs", ""),
        "acceptance_runs": quality.get("acceptance_runs", ""),
        "reference_statistic": reference_statistic_summary(quality),
        "reference_protocol": reference_protocol_summary(
            quality.get("reference_protocol")
        ),
        "direction": quality.get("direction", ""),
        "quality_required": quality_required,
        "target_met": quality.get("target_met", ""),
        "functional_check": functional_check_summary(quality.get("functional_check")),
        "duration_seconds": metrics.get("duration_seconds", ""),
        "throughput": metrics.get(throughput_key) if throughput_key else "",
        "power_average_watts": (default_power or item.get("power") or {}).get(
            "average_watts", ""
        ),
        "energy_joules": (default_power or item.get("power") or {}).get(
            "energy_joules", ""
        ),
    }


def dataset_name_for_row(dataset: Any) -> str:
    if isinstance(dataset, dict):
        return str(dataset.get("name") or dataset.get("id") or "")
    if dataset is None:
        return ""
    return str(dataset)


def model_source_summary(model_source: Any) -> str:
    if isinstance(model_source, dict):
        return str(
            model_source.get("repo_id")
            or model_source.get("source_url")
            or model_source.get("type")
            or ""
        )
    return ""


def functional_check_summary(functional_check: Any) -> str:
    if not isinstance(functional_check, dict):
        return ""
    metric = functional_check.get("metric", "")
    condition = functional_check.get("condition", "")
    if metric and condition:
        return f"{metric}: {condition}"
    return str(metric or condition)


def reference_protocol_summary(protocol: Any) -> str:
    if not isinstance(protocol, dict):
        return ""
    parts = []
    for key in (
        "profile",
        "backend",
        "machine_class",
        "dataset_mode",
        "aggregation",
        "rerun_policy",
    ):
        value = protocol.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    seeds = protocol.get("seeds")
    if isinstance(seeds, list) and seeds:
        parts.append("seeds=" + ",".join(str(seed) for seed in seeds))
    return "; ".join(parts)


def reference_statistic_summary(quality: dict[str, Any]) -> str:
    """Return the planned aggregation statistic without copying past results."""
    explicit = quality.get("reference_statistic")
    if explicit not in (None, ""):
        return str(explicit)
    protocol = quality.get("reference_protocol")
    if not isinstance(protocol, dict):
        return ""
    aggregation = str(protocol.get("aggregation") or "").strip()
    if not aggregation:
        return ""
    statistic = aggregation.split(maxsplit=1)[0].lower()
    if statistic in {"mean", "median", "minimum", "maximum", "min", "max"}:
        return statistic
    return ""


def checkpoint_source_quality_summary(provenance: dict[str, Any]) -> str:
    metric = provenance.get("source_quality_metric")
    target = provenance.get("source_quality_target")
    direction = provenance.get("source_quality_direction")
    basis = provenance.get("source_target_basis")
    if not metric:
        return ""
    parts = [str(metric)]
    if direction:
        parts.append(str(direction))
    if target not in (None, ""):
        parts.append(str(target))
    if basis:
        parts.append(f"basis={basis}")
    return " ".join(parts)


def quality_metric_name(quality: dict[str, Any]) -> str | None:
    metric = quality.get("metric")
    if metric:
        return str(metric)
    functional_check = quality.get("functional_check")
    if isinstance(functional_check, dict) and functional_check.get("metric"):
        return str(functional_check["metric"])
    return None


def metric_key_for_throughput(metrics: dict[str, Any]) -> str | None:
    for key in (
        "tokens_per_second",
        "tokens_per_second_mean",
        "prefill_tokens_per_sec",
        "prefill_tokens_per_sec_mean",
        "output_tokens_per_sec",
        "output_tokens_per_sec_mean",
        "samples_per_second",
        "samples_per_second_mean",
    ):
        if key in metrics:
            return key
    return None


def status_class(status: str) -> str:
    if status == "passed":
        return "pass"
    if status in {"quality_failed", "failed"} or status.endswith("_failed"):
        return "fail"
    return "warn"


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_key_for_quality(
    metric_name: str | None, metrics: dict[str, Any]
) -> str | None:
    if metric_name and metric_name in metrics:
        return metric_name
    candidates_by_metric = {
        "accuracy": (
            "best_accuracy",
            "best_accuracy_mean",
            "final_accuracy",
            "final_accuracy_mean",
            "accuracy",
            "accuracy_mean",
        ),
        "top1_accuracy": (
            "best_accuracy",
            "best_accuracy_mean",
            "final_accuracy",
            "final_accuracy_mean",
            "accuracy",
            "accuracy_mean",
        ),
        "binary_accuracy": (
            "best_accuracy",
            "best_accuracy_mean",
            "final_accuracy",
            "final_accuracy_mean",
            "accuracy",
            "accuracy_mean",
        ),
        "test_accuracy": (
            "test_accuracy",
            "test_accuracy_mean",
            "accuracy",
            "accuracy_mean",
        ),
        "val_accuracy": (
            "val_accuracy",
            "val_accuracy_mean",
            "accuracy",
            "accuracy_mean",
        ),
        "mse_loss": ("mse_loss", "mse_loss_mean", "loss", "loss_mean"),
        "val_mse": ("val_mse", "val_mse_mean", "mse_loss", "mse_loss_mean"),
        "avg_episode_reward": (
            "avg_episode_reward",
            "avg_episode_reward_mean",
            "episode_reward",
        ),
        "reconstruction_mse": (
            "final_reconstruction_mse",
            "final_reconstruction_mse_mean",
            "reconstruction_mse",
            "reconstruction_mse_mean",
            "final_val_reconstruction_mse",
            "final_val_reconstruction_mse_mean",
        ),
        "cross_entropy_loss": (
            "final_val_loss",
            "final_val_loss_mean",
            "final_train_loss",
            "final_train_loss_mean",
            "loss",
            "loss_mean",
        ),
        "generated_tokens": ("generated_tokens", "generated_tokens_mean"),
        "retrieval_accuracy": (
            "queries_per_second",
            "retrieve_latency_ms",
            "total_latency_ms",
        ),
        "pass_at_1": ("iterations", "tokens_per_second", "total_latency_ms"),
        "trace_accuracy": ("steps", "total_reasoning_ms", "total_latency_ms"),
        "relative_loss_delta": ("relative_loss_delta", "relative_loss_delta_mean"),
    }
    for key in candidates_by_metric.get(metric_name or "", ()):
        if key in metrics:
            return key
    for fallback in (
        "final_val_loss",
        "final_val_loss_mean",
        "final_accuracy",
        "final_accuracy_mean",
        "final_reconstruction_mse",
        "final_reconstruction_mse_mean",
        "prefill_latency_s",
        "prefill_latency_s_mean",
        "prefill_tokens_per_sec",
        "prefill_tokens_per_sec_mean",
        "itl_median_s",
        "itl_median_s_mean",
        "output_tokens_per_sec",
        "output_tokens_per_sec_mean",
        "generated_tokens",
        "generated_tokens_mean",
        "relative_loss_delta",
        "relative_loss_delta_mean",
        "queries_per_second",
        "queries_per_second_mean",
        "valid_call_rate",
        "valid_call_rate_mean",
        "total_latency_ms",
        "total_latency_ms_mean",
        "tokens_per_second",
        "tokens_per_second_mean",
        "loss",
        "loss_mean",
        "mse_loss",
        "mse_loss_mean",
        "val_mse",
        "val_mse_mean",
        "test_accuracy",
        "test_accuracy_mean",
        "val_accuracy",
        "val_accuracy_mean",
        "avg_episode_reward",
        "avg_episode_reward_mean",
    ):
        if fallback in metrics:
            return fallback
    return None


def print_verification_checks(result: Any) -> None:
    table = Table(title=f"Verification: {result.workload}")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    dataset_file_checks = 0
    for name, ok, detail in result.checks:
        if ok and re.fullmatch(r"dataset\.files\[\d+\]\.(?:sha256|n_bytes)", name):
            dataset_file_checks += 1
            continue
        status = "[green]ok[/green]" if ok else "[red]fail[/red]"
        table.add_row(name, status, detail)
    if dataset_file_checks:
        table.add_row(
            "dataset.files",
            "[green]ok[/green]",
            f"{dataset_file_checks} per-file hash and size checks passed",
        )
    console.print(table)


def collect_package_files(
    manifest_path: Path, manifest: dict[str, Any]
) -> list[tuple[str, Path, str]]:
    """Collect every file needed to inspect and verify a submission archive."""
    files: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()
    archive_names: set[str] = set()

    def add(role: str, path: Path | None, archive_name: str) -> None:
        if not path or not path.exists() or not path.is_file():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        if archive_name in archive_names:
            raise ValueError(f"duplicate package archive path: {archive_name}")
        seen.add(resolved)
        archive_names.add(archive_name)
        files.append((role, resolved, archive_name))

    add("manifest", manifest_path, f"manifest/{manifest_path.name}")

    report_path = manifest_report_path(manifest, manifest_path)
    if report_path:
        add("report", report_path, f"report/{report_path.name}")
        add(
            "report_html",
            report_path.with_suffix(".html"),
            f"report/{report_path.with_suffix('.html').name}",
        )
        add(
            "report_csv",
            report_path.with_suffix(".csv"),
            f"report/{report_path.with_suffix('.csv').name}",
        )

    leaves = manifest.get("leaves") or {}
    weights = leaves.get("weights") or {}
    weights_path = weights.get("path")
    if weights_path:
        resolved = resolve_artifact_path(manifest_path, weights_path)
        add("weights", resolved, f"weights/{resolved.name}")
    for index, item in enumerate(weights.get("files") or []):
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError(f"invalid multi-file weights record at index {index}")
        logical_path = safe_logical_asset_path(str(item.get("logical_path") or ""))
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"model asset {logical_path!r} has no provenance role")
        safe_role = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in role
        )
        resolved = resolve_artifact_path(manifest_path, item["path"])
        add(
            f"weights:{safe_role}",
            resolved,
            f"weights/model-assets/{logical_path}",
        )

    roofline_path = (leaves.get("roofline_sidecar") or {}).get("path")
    if roofline_path:
        resolved = resolve_artifact_path(manifest_path, roofline_path)
        add("roofline_sidecar", resolved, f"roofline/{resolved.name}")

    dataset_files = (leaves.get("dataset") or {}).get("files") or []
    for index, item in enumerate(dataset_files):
        if not isinstance(item, dict) or not item.get("path"):
            continue
        resolved = resolve_artifact_path(manifest_path, item["path"])
        add("dataset", resolved, f"dataset/{index:03d}-{resolved.name}")

    # Reports can carry additional evidence that is not a first-class manifest
    # leaf, such as model metadata. Include it and cover it in package_index.json.
    if report_path and report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            report = {}
        artifacts = report.get("artifacts") or {}
        if isinstance(artifacts, dict):
            for key, value in sorted(artifacts.items()):
                if not isinstance(value, str) or not value:
                    continue
                source = Path(value)
                if not source.is_absolute():
                    source = report_path.parent / source
                safe_key = "".join(
                    char if char.isalnum() or char in "-_" else "_" for char in str(key)
                )
                add(
                    f"report_artifact:{key}",
                    source,
                    f"artifacts/{safe_key}/{source.name}",
                )

    return files


def manifest_report_path(
    manifest: dict[str, Any], manifest_path: Path | None = None
) -> Path | None:
    report_path = ((manifest.get("leaves") or {}).get("measurement") or {}).get(
        "report_path"
    )
    if not report_path:
        return None
    return (
        resolve_artifact_path(manifest_path, report_path)
        if manifest_path
        else Path(report_path)
    )


def grade_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    result = verify_provd(manifest_path, repo_root=find_project_root())

    report_path = manifest_report_path(manifest, manifest_path)
    report: dict[str, Any] = {}
    if report_path and report_path.exists():
        report = json.loads(report_path.read_text())

    metrics = report.get("metrics") or {}
    quality = report.get("quality") or {}
    metric_name = quality_metric_name(quality)
    metric_key = metric_key_for_quality(metric_name, metrics)
    status = report.get("status") or (
        "missing_report" if report_path else "missing_measurement"
    )
    quality_required = bool(quality_required_value(quality, False))
    target_met = quality.get("target_met", "")
    passed = bool(
        result.all_ok
        and status == "passed"
        and (not quality_required or target_met is True)
    )
    quality_ready = bool(
        result.all_ok
        and status == "passed"
        and quality_required
        and target_met is True
    )
    workload_id = report.get("workload") or manifest.get("workload", "unknown")
    canonical = (
        report.get("canonical_workload")
        or canonical_workload_for_id(str(workload_id))
        or ""
    )
    variant = report.get("variant") or (
        workload_variant_name(str(workload_id)) if canonical else ""
    )
    run_selector = report.get("run_selector") or (
        f"{canonical} --variant {variant}"
        if canonical and variant
        else str(workload_id)
    )
    return {
        "manifest": str(manifest_path),
        "report": str(report_path) if report_path else None,
        "workload": workload_id,
        "canonical_workload": canonical,
        "variant": variant,
        "run_selector": run_selector,
        "profile": report.get("profile", ""),
        "status": status,
        "verified": result.all_ok,
        "passed": passed,
        "quality_ready": quality_ready,
        "metric": metric_key or metric_name or "",
        "value": metrics.get(metric_key) if metric_key else "",
        "target": quality.get("target", ""),
        "quality_required": quality_required,
        "target_met": target_met,
        "warning_count": 0,
        "warnings": [],
    }


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _archive_relative(target: str, owner: str) -> str:
    return posixpath.relpath(target, start=posixpath.dirname(owner))


def _portable_report_paths(
    value: Any,
    *,
    source_to_archive: dict[Path, str],
    owner_archive: str,
) -> Any:
    """Replace machine-local absolute paths in a packaged report."""
    if isinstance(value, dict):
        return {
            key: _portable_report_paths(
                item,
                source_to_archive=source_to_archive,
                owner_archive=owner_archive,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _portable_report_paths(
                item,
                source_to_archive=source_to_archive,
                owner_archive=owner_archive,
            )
            for item in value
        ]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value

    source = Path(value).resolve()
    archive_name = source_to_archive.get(source)
    if archive_name:
        return _archive_relative(archive_name, owner_archive)

    descendant_archives: list[str] = []
    for candidate, candidate_archive in source_to_archive.items():
        try:
            candidate.relative_to(source)
        except ValueError:
            continue
        descendant_archives.append(candidate_archive)
    if descendant_archives:
        common_archive = posixpath.commonpath(descendant_archives)
        if common_archive not in {"", "."}:
            if posixpath.splitext(common_archive)[1]:
                common_archive = posixpath.dirname(common_archive)
            return _archive_relative(common_archive, owner_archive)

    # Interpreter/cache locations describe the source environment but are not
    # submission artifacts. Keep a recognizable label without leaking or
    # pretending that the original absolute path is portable.
    return f"local-environment:{source.name or 'root'}"


def _absolute_path_values(value: Any, prefix: str = "report") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_absolute_path_values(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_absolute_path_values(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and Path(value).is_absolute():
        found.append(prefix)
    return found


def _portable_report_exports(
    report: dict[str, Any],
    *,
    report_archive: str,
    package_files: list[tuple[str, Path, str]],
    source_to_archive: dict[Path, str],
) -> dict[str, bytes]:
    """Regenerate HTML/CSV views from the archive-specific JSON report."""
    role_to_archive = {role: archive for role, _, archive in package_files}
    overrides: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="mlperf-edu-package-report-") as tmp:
        temp_report = Path(tmp) / Path(report_archive).name
        temp_report.write_bytes(_json_bytes(report))
        exports = write_report_exports(report, temp_report, open_report=False)
        for role, export_key in (("report_html", "html"), ("report_csv", "csv")):
            archive_name = role_to_archive.get(role)
            if not archive_name:
                continue
            payload = exports[export_key].read_bytes()
            payload = payload.replace(
                str(temp_report).encode(),
                _archive_relative(report_archive, archive_name).encode(),
            )
            for source, target_archive in source_to_archive.items():
                payload = payload.replace(
                    str(source).encode(),
                    _archive_relative(target_archive, archive_name).encode(),
                )
            overrides[archive_name] = payload
    return overrides


def build_portable_package_files(
    manifest_path: Path,
    manifest: dict[str, Any],
    package_files: list[tuple[str, Path, str]],
) -> tuple[dict[str, bytes], str]:
    """Build archive-specific report and manifest bytes with relative paths."""
    source_to_archive = {
        source.resolve(): archive for _, source, archive in package_files
    }
    manifest_archive = source_to_archive[manifest_path.resolve()]
    report_path = manifest_report_path(manifest, manifest_path)
    if not report_path or not report_path.is_file():
        raise ValueError("manifest report is missing")
    report_archive = source_to_archive.get(report_path.resolve())
    if not report_archive:
        raise ValueError("manifest report was not collected for packaging")

    packaged_report = _portable_report_paths(
        json.loads(report_path.read_text()),
        source_to_archive=source_to_archive,
        owner_archive=report_archive,
    )

    report_bytes = _json_bytes(packaged_report)
    packaged_manifest = json.loads(json.dumps(manifest))
    leaves = packaged_manifest["leaves"]

    def rewrite_leaf_path(leaf: dict[str, Any]) -> None:
        raw_path = leaf.get("path")
        if not raw_path:
            return
        source = resolve_artifact_path(manifest_path, raw_path).resolve()
        archive_name = source_to_archive.get(source)
        if not archive_name:
            raise ValueError(f"manifest artifact was not collected: {raw_path}")
        leaf["path"] = _archive_relative(archive_name, manifest_archive)

    weights = leaves.get("weights") or {}
    rewrite_leaf_path(weights)
    for item in weights.get("files") or []:
        if isinstance(item, dict):
            rewrite_leaf_path(item)
    rewrite_leaf_path(leaves.get("roofline_sidecar") or {})
    dataset = leaves.get("dataset") or {}
    for item in dataset.get("files") or []:
        if isinstance(item, dict):
            rewrite_leaf_path(item)
    if dataset.get("files"):
        dataset["merkle_root"] = dataset_merkle_root(dataset["files"])

    leaves["measurement"] = measurement_leaf(
        packaged_report,
        _archive_relative(report_archive, manifest_archive),
        report_bytes=report_bytes,
    )
    packaged_manifest["merkle_root"] = merkle_root(leaves)
    packaged_manifest["integrity"] = integrity_record(packaged_manifest["merkle_root"])
    packaged_manifest.pop("signature", None)
    manifest_bytes = _json_bytes(packaged_manifest)
    archive_bytes = {
        report_archive: report_bytes,
        manifest_archive: manifest_bytes,
    }
    archive_bytes.update(
        _portable_report_exports(
            packaged_report,
            report_archive=report_archive,
            package_files=package_files,
            source_to_archive=source_to_archive,
        )
    )
    return archive_bytes, manifest_archive


def verify_package_archive(
    package_path: Path,
    *,
    repo_root: Path,
) -> list[tuple[str, bool, str]]:
    """Verify index coverage and provenance after extraction to a clean directory."""
    checks: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="mlperf-edu-package-verify-") as tmp:
        extraction_root = Path(tmp)
        with zipfile.ZipFile(package_path) as zf:
            names = set(zf.namelist())
            if "package_index.json" not in names:
                return [("package_index", False, "package_index.json is missing")]
            index = json.loads(zf.read("package_index.json"))
            included = index.get("included_files") or []
            indexed_names = [str(item.get("path", "")) for item in included]
            checks.append(
                (
                    "index.unique_paths",
                    len(indexed_names) == len(set(indexed_names)),
                    "every archive path must appear exactly once in the index",
                )
            )
            unindexed = sorted(names - {"package_index.json"} - set(indexed_names))
            checks.append(
                (
                    "index.complete_coverage",
                    not unindexed,
                    f"unindexed archive members: {unindexed}",
                )
            )
            for item in included:
                archive_name = str(item.get("path", ""))
                archive_path = Path(archive_name)
                safe = (
                    bool(archive_name)
                    and not archive_path.is_absolute()
                    and ".." not in archive_path.parts
                )
                checks.append(
                    (
                        f"index.path:{archive_name}",
                        safe,
                        "archive-relative path required",
                    )
                )
                if not safe or archive_name not in names:
                    checks.append(
                        (f"index.file:{archive_name}", False, "indexed file is missing")
                    )
                    continue
                payload = zf.read(archive_name)
                actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                checks.append(
                    (
                        f"index.sha256:{archive_name}",
                        actual_digest == item.get("sha256"),
                        f"claimed {item.get('sha256')}, recomputed {actual_digest}",
                    )
                )
                checks.append(
                    (
                        f"index.n_bytes:{archive_name}",
                        len(payload) == item.get("n_bytes"),
                        f"claimed {item.get('n_bytes')}, recomputed {len(payload)}",
                    )
                )
            zf.extractall(extraction_root)

        manifest_name = str(index.get("manifest") or index.get("source_manifest") or "")
        manifest_path = extraction_root / manifest_name
        if not manifest_name or not manifest_path.is_file():
            checks.append(
                ("clean_extraction.manifest", False, "packaged manifest is missing")
            )
            return checks

        packaged_manifest = json.loads(manifest_path.read_text())
        leaves = packaged_manifest.get("leaves") or {}
        path_values = [
            ((leaves.get("measurement") or {}).get("report_path")),
            ((leaves.get("weights") or {}).get("path")),
            *[
                item.get("path")
                for item in ((leaves.get("weights") or {}).get("files") or [])
                if isinstance(item, dict)
            ],
            ((leaves.get("roofline_sidecar") or {}).get("path")),
            *[
                item.get("path")
                for item in ((leaves.get("dataset") or {}).get("files") or [])
                if isinstance(item, dict)
            ],
        ]
        relative_paths = all(
            not Path(value).is_absolute() for value in path_values if value
        )
        checks.append(
            (
                "clean_extraction.manifest_paths",
                relative_paths,
                "all packaged manifest artifact paths must be relative",
            )
        )
        measurement = leaves.get("measurement") or {}
        report_path = resolve_artifact_path(
            manifest_path, measurement.get("report_path", "")
        )
        packaged_report = (
            json.loads(report_path.read_text()) if report_path.is_file() else {}
        )
        absolute_report_paths = _absolute_path_values(packaged_report)
        checks.append(
            (
                "clean_extraction.report_paths",
                not absolute_report_paths,
                f"absolute path values remain at: {absolute_report_paths}",
            )
        )
        report_artifacts = packaged_report.get("artifacts") or {}
        report_paths_relative = (
            all(
                not Path(value).is_absolute()
                for value in report_artifacts.values()
                if isinstance(value, str) and value
            )
            if isinstance(report_artifacts, dict)
            else False
        )
        checks.append(
            (
                "clean_extraction.report_artifact_paths",
                report_paths_relative,
                "all packaged report artifact paths must be relative",
            )
        )
        provenance = verify_provd(manifest_path, repo_root=repo_root)
        checks.extend(
            (f"clean_extraction.{name}", ok, detail)
            for name, ok, detail in provenance.checks
        )
    return checks


def cmd_package(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found:[/red] {manifest_path}")
        return 1

    result = verify_provd(manifest_path, repo_root=find_project_root())
    if not result.all_ok:
        console.print(f"[red]Cannot package unverified manifest:[/red] {manifest_path}")
        print_verification_checks(result)
        return 1

    manifest = json.loads(manifest_path.read_text())
    dataset_policy_issue = package_dataset_policy_issue(manifest)
    if dataset_policy_issue:
        console.print(
            f"[red]Cannot package restricted dataset bytes:[/red] {dataset_policy_issue}"
        )
        return 1
    package_path = (
        Path(args.output).resolve()
        if args.output
        else manifest_path.with_name(
            manifest_path.name.replace(".provd.json", ".mlperf-edu.zip")
        )
    )
    package_path.parent.mkdir(parents=True, exist_ok=True)

    package_files = collect_package_files(manifest_path, manifest)
    try:
        archive_bytes, packaged_manifest_name = build_portable_package_files(
            manifest_path,
            manifest,
            package_files,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        console.print(f"[red]Cannot build portable package:[/red] {exc}")
        return 1

    included_files = []
    for role, source_path, archive_name in package_files:
        payload = archive_bytes.get(archive_name)
        if payload is None:
            digest = "sha256:" + sha256_file(source_path)
            n_bytes = source_path.stat().st_size
        else:
            digest = "sha256:" + hashlib.sha256(payload).hexdigest()
            n_bytes = len(payload)
        included_files.append(
            {
                "role": role,
                "path": archive_name,
                "sha256": digest,
                "n_bytes": n_bytes,
            }
        )
    index = {
        "schema": "mlperf-edu-package/0.2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": manifest.get("workload", "unknown"),
        "manifest": packaged_manifest_name,
        "source_manifest": packaged_manifest_name,
        "path_policy": (
            "Package-index paths are relative to the archive root. Manifest and report artifact pointers "
            "are relative to their owning file. Other machine-local source locations are replaced with "
            "local-environment:<name> labels."
        ),
        "digest_policy": "SHA-256 digests provide unauthenticated integrity checking, not producer authentication.",
        "included_files": included_files,
        "source_verification": {
            "passed": True,
            "checks": [{"check": name, "ok": ok} for name, ok, _ in result.checks],
        },
        "verification": [{"check": name, "ok": ok} for name, ok, _ in result.checks],
        "clean_extraction_verification": {
            "required": True,
            "status": "passed",
        },
    }

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "package_index.json", json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        for _, source_path, archive_name in package_files:
            payload = archive_bytes.get(archive_name)
            if payload is None:
                zf.write(source_path, archive_name)
            else:
                zf.writestr(archive_name, payload)

    portable_checks = verify_package_archive(
        package_path, repo_root=find_project_root()
    )
    if not all(ok for _, ok, _ in portable_checks):
        package_path.unlink(missing_ok=True)
        console.print("[red]Clean-extraction package verification failed.[/red]")
        for name, ok, detail in portable_checks:
            if not ok:
                console.print(f"  [red]{name}[/red]: {detail}")
        return 1

    console.print(f"[green]packaged[/green]: {package_path}")
    return 0


def package_dataset_policy_issue(manifest: dict[str, Any]) -> str | None:
    """Return a fail-closed redistribution issue for known dataset bytes."""
    dataset = (manifest.get("leaves") or {}).get("dataset") or {}
    dataset_name = str(dataset.get("name") or "")
    files = [item for item in dataset.get("files") or [] if isinstance(item, dict)]
    if not dataset_name or not files or not has_asset_dossier(dataset_name):
        return None
    dossier = asset_dossier(dataset_name)
    release_status = str(dossier.get("public_release_status") or "")
    if release_status not in {"restricted-needs-approval", "needs-release-decision"}:
        return None
    policy = str(
        dossier.get("public_release_policy") or "Resolve the dataset release policy."
    )
    return f"{dataset_name} has release status {release_status}. {policy}"


def cmd_grade(args: argparse.Namespace) -> int:
    submissions_dir = Path(args.submissions_dir).resolve()
    if not submissions_dir.exists():
        console.print(f"[red]Submissions directory not found:[/red] {submissions_dir}")
        return 1

    manifests = sorted(
        path
        for path in submissions_dir.rglob("*.provd.json")
        if ".pro_evidence" not in path.parts and ".max_evidence" not in path.parts
    )
    if not manifests:
        console.print(
            f"[red]No .provd.json manifests found in:[/red] {submissions_dir}"
        )
        return 1

    rows = [grade_manifest(path) for path in manifests]
    table = Table(title=f"MLPerf EDU Grade: {submissions_dir}")
    table.add_column("Workload", no_wrap=True)
    table.add_column("Profile", no_wrap=True)
    table.add_column("Result", no_wrap=True)
    table.add_column("Verify", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Target")
    table.add_column("Quality Required", no_wrap=True)
    table.add_column("Met", no_wrap=True)
    table.add_column("Warnings", justify="right")
    for row in rows:
        style = "green" if row["passed"] else "red"
        table.add_row(
            str(row["workload"]),
            str(row["profile"]),
            f"[{style}]{row['status']}[/{style}]",
            "ok" if row["verified"] else "fail",
            str(row["metric"]),
            format_cell(row["value"]),
            format_cell(row["target"]),
            format_cell(row["quality_required"]),
            format_cell(row["target_met"]),
            str(row.get("warning_count", 0)),
        )
    console.print(table)

    passed = sum(1 for row in rows if row["passed"])
    failed = len(rows) - passed
    warning_count = sum(int(row.get("warning_count", 0)) for row in rows)
    summary = {
        "schema": "mlperf-edu-grade/0.1",
        "submissions_dir": str(submissions_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "failed": failed,
        "warning_count": warning_count,
        "results": rows,
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        console.print(f"Grade JSON: {output}")

    console.print(
        f"Grade summary: {passed} passed, {failed} failed, {warning_count} warning(s)"
    )
    return 0 if failed == 0 else 1


def cmd_validate(args: argparse.Namespace) -> int:
    validation_started_at = datetime.now(timezone.utc).isoformat()
    validation_start = time.perf_counter()
    try:
        preset = resolve_validation_preset(args)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    workloads = load_workloads(args)
    items = validation_plan(preset, workloads, suite_filter=args.suite)
    if not items:
        console.print("[red]No validation items selected.[/red]")
        return 1

    output_root = Path(args.output_dir).resolve()
    table = Table(title=f"MLPerf EDU Validation: {preset}")
    table.add_column("Validation")
    table.add_column("Selection")
    table.add_column("Profile")
    table.add_column("Output")
    for selector_kind, selector_name, profile in items:
        table.add_row(
            validation_id(selector_kind, selector_name, profile),
            validation_selection_label(selector_kind, selector_name),
            profile,
            str(
                validation_output_dir(
                    output_root, selector_kind, selector_name, profile
                )
            ),
        )
    console.print(table)

    if args.dry_run:
        console.print("[green]dry-run complete[/green]")
        return 0

    failures: list[tuple[str, str, str]] = []
    records: list[dict[str, Any]] = []
    preflight: dict[str, Any] = {"doctor_skipped": bool(args.skip_doctor)}
    if not args.skip_doctor:
        console.print("[bold]Validation preflight: doctor[/bold]")
        doctor_start = time.perf_counter()
        status = cmd_doctor(args)
        preflight["doctor_exit"] = status
        preflight["doctor_duration_seconds"] = float(time.perf_counter() - doctor_start)
        if status != 0:
            failures.append(("doctor", "-", f"exit {status}"))
            if not args.keep_going:
                return finish_validation(
                    preset,
                    output_root,
                    records,
                    failures,
                    preflight,
                    started_at=validation_started_at,
                    duration_seconds=time.perf_counter() - validation_start,
                    open_report=args.open_report,
                )

    for selector_kind, selector_name, profile in items:
        output_dir = validation_output_dir(
            output_root, selector_kind, selector_name, profile
        )
        item_id = validation_id(selector_kind, selector_name, profile)
        record: dict[str, Any] = {
            "validation": item_id,
            "selection_kind": selector_kind,
            "selection": validation_selection_label(selector_kind, selector_name),
            "suite": selector_name if selector_kind == "suite" else "",
            "profile": profile,
            "output_dir": str(output_dir),
            "status": "running",
        }
        records.append(record)
        console.print(
            f"[bold]Validation run:[/bold] {validation_selection_label(selector_kind, selector_name)} / {profile}"
        )
        record_start = time.perf_counter()
        run_args = argparse.Namespace(
            registry=args.registry,
            suite=selector_name if selector_kind == "suite" else None,
            workload=None,
            variant=None,
            profile=profile,
            output_dir=str(output_dir),
            open_report=False,
            power=False,
        )
        if selector_kind == "collection":
            run_args.collection = selector_name
        else:
            run_args.collection = None
        validation_seed = validation_seed_environment(profile)
        record["seed"] = validation_seed["seed"]
        record["seed_source"] = validation_seed["source"]
        console.print(
            "[bold]Validation seed:[/bold] "
            f"{validation_seed['seed']} ({validation_seed['source']})"
        )
        previous_max_seed = os.environ.get("MLPERF_EDU_MAX_SEED")
        if validation_seed["set_max_seed"]:
            os.environ["MLPERF_EDU_MAX_SEED"] = str(validation_seed["seed"])
        try:
            run_status = cmd_run(run_args)
        finally:
            if validation_seed["set_max_seed"]:
                if previous_max_seed is None:
                    os.environ.pop("MLPERF_EDU_MAX_SEED", None)
                else:
                    os.environ["MLPERF_EDU_MAX_SEED"] = previous_max_seed
        record["run_exit"] = run_status
        record.update(latest_aggregate_exports(output_dir, profile))
        contract_failures = validation_contract_failures(record.get("report"))
        record["review_contract_failures"] = contract_failures
        record["review_contract_failure_count"] = len(contract_failures)
        if run_status != 0:
            record["status"] = "run_failed"
            record["duration_seconds"] = float(time.perf_counter() - record_start)
            failures.append((item_id, profile, f"run exit {run_status}"))
            if not args.keep_going:
                return finish_validation(
                    preset,
                    output_root,
                    records,
                    failures,
                    preflight,
                    started_at=validation_started_at,
                    duration_seconds=time.perf_counter() - validation_start,
                    open_report=args.open_report,
                )
            continue

        if args.skip_grade:
            if contract_failures:
                record["status"] = "contract_failed"
                failures.append(
                    (
                        item_id,
                        profile,
                        f"{len(contract_failures)} public review contract failure(s)",
                    )
                )
            else:
                record["status"] = "passed"
            record["grade_skipped"] = True
            record["duration_seconds"] = float(time.perf_counter() - record_start)
            continue
        console.print(
            f"[bold]Validation grade:[/bold] {validation_selection_label(selector_kind, selector_name)} / {profile}"
        )
        grade_output = output_dir / "grade.json"
        grade_args = argparse.Namespace(
            registry=args.registry,
            submissions_dir=str(output_dir),
            output=str(grade_output),
        )
        grade_status = cmd_grade(grade_args)
        record["grade_exit"] = grade_status
        record["grade_json"] = str(grade_output)
        if grade_output.exists():
            grade_data = json.loads(grade_output.read_text())
            record["passed"] = int(grade_data.get("passed", 0))
            record["failed"] = int(grade_data.get("failed", 0))
            record["warning_count"] = int(grade_data.get("warning_count", 0))
        if grade_status != 0:
            record["status"] = "grade_failed"
            record["duration_seconds"] = float(time.perf_counter() - record_start)
            failures.append((item_id, profile, f"grade exit {grade_status}"))
            if not args.keep_going:
                return finish_validation(
                    preset,
                    output_root,
                    records,
                    failures,
                    preflight,
                    started_at=validation_started_at,
                    duration_seconds=time.perf_counter() - validation_start,
                    open_report=args.open_report,
                )
        elif contract_failures:
            record["status"] = "contract_failed"
            record["duration_seconds"] = float(time.perf_counter() - record_start)
            failures.append(
                (
                    item_id,
                    profile,
                    f"{len(contract_failures)} public review contract failure(s)",
                )
            )
        else:
            record["status"] = "passed"
            record["duration_seconds"] = float(time.perf_counter() - record_start)

    return finish_validation(
        preset,
        output_root,
        records,
        failures,
        preflight,
        started_at=validation_started_at,
        duration_seconds=time.perf_counter() - validation_start,
        open_report=args.open_report,
    )


def resolve_validation_preset(args: argparse.Namespace) -> str:
    candidates = []
    if getattr(args, "preset", None):
        candidates.append(args.preset)
    if getattr(args, "preset_option", None):
        candidates.append(args.preset_option)
    if getattr(args, "legacy_level", None):
        candidates.append(LEGACY_VALIDATE_LEVELS[args.legacy_level])
    if not candidates:
        return "smoke"
    selected = candidates[0]
    if any(candidate != selected for candidate in candidates[1:]):
        raise ValueError(f"conflicting validation presets: {', '.join(candidates)}")
    return selected


def validation_seed_environment(profile: str) -> dict[str, Any]:
    """Preserve explicit overrides or let each workload use its canonical seed."""
    for name in ("MLPERF_EDU_SEED", "MLPERF_EDU_MAX_SEED"):
        value = os.environ.get(name)
        if value is not None:
            return {
                "seed": int(value),
                "source": name,
                "set_max_seed": False,
            }
    return {
        "seed": None,
        "source": "per_workload_canonical_default",
        "set_max_seed": False,
    }


def validation_plan(
    preset: str,
    workloads: dict[str, Workload],
    *,
    suite_filter: list[str] | None,
) -> list[tuple[str, str, str]]:
    present_suites = tuple(
        suite
        for suite in PRODUCT_SUITES
        if any(workload.suite == suite for workload in workloads.values())
    )
    if suite_filter:
        allowed = set(suite_filter)
        selected_suites = tuple(suite for suite in present_suites if suite in allowed)
        if preset in {"smoke", "coverage"}:
            return [("suite", suite, "min") for suite in selected_suites]
        if preset in {"max", "pro"}:
            return [("suite", suite, preset) for suite in selected_suites]
        if preset == "release":
            return (
                [("suite", suite, "min") for suite in selected_suites]
                + [("suite", suite, "max") for suite in selected_suites]
                + [("suite", suite, "pro") for suite in selected_suites]
            )
        raise ValueError(f"unknown validation preset: {preset}")

    if preset == "smoke":
        return [("collection", DEFAULT_WORKLOAD_COLLECTION, "min")]
    if preset == "coverage":
        return [("collection", "all", "min")]
    if preset == "max":
        return [("collection", "all", "max")]
    if preset == "pro":
        return [("collection", "research", "pro")]
    if preset == "release":
        return [
            ("collection", "all", "min"),
            ("collection", "all", "max"),
            ("collection", "research", "pro"),
        ]
    raise ValueError(f"unknown validation preset: {preset}")


def validation_selection_label(selector_kind: str, selector_name: str) -> str:
    if selector_kind == "suite":
        return f"suite:{selector_name}"
    if selector_name == DEFAULT_WORKLOAD_COLLECTION:
        return "default"
    if selector_name == "all":
        return "all workloads"
    return f"{selector_name} workloads"


def suite_has_profile(
    workloads: dict[str, Workload], *, suite: str, profile: str
) -> bool:
    selected = [workload for workload in workloads.values() if workload.suite == suite]
    return bool(selected) and all(
        profile in (workload.raw.get("runner") or {}) for workload in selected
    )


def validation_output_dir(
    output_root: Path, selector_kind: str, selector_name: str, profile: str
) -> Path:
    return output_root / validation_id(selector_kind, selector_name, profile)


def validation_id(selector_kind: str, selector_name: str, profile: str) -> str:
    if selector_kind == "collection":
        if selector_name == DEFAULT_WORKLOAD_COLLECTION:
            return f"{profile}-default"
    return f"{profile}-{selector_name}"


def latest_aggregate_exports(output_dir: Path, profile: str) -> dict[str, str]:
    reports = sorted(output_dir.glob(f"mlperf_edu_{profile}_*.json"))
    if not reports:
        return {}
    report = reports[-1]
    return {
        "report": str(report),
        "report_html": str(report.with_suffix(".html")),
        "report_csv": str(report.with_suffix(".csv")),
    }


def validation_contract_failures(report_value: Any) -> list[dict[str, Any]]:
    if not report_value:
        return []
    try:
        report = json.loads(Path(str(report_value)).read_text())
    except (OSError, json.JSONDecodeError):
        return [
            {"workload": "aggregate", "issues": ["aggregate report could not be read"]}
        ]
    return aggregate_contract_issues(report)


def validation_workload_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        report_path = record.get("report")
        if not report_path:
            continue
        try:
            report = json.loads(Path(str(report_path)).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in report_rows(report):
            enriched = {
                "validation": record.get("validation", ""),
            }
            enriched.update(row)
            rows.append(enriched)
    return rows


def finish_validation(
    preset: str,
    output_root: Path,
    records: list[dict[str, Any]],
    failures: list[tuple[str, str, str]],
    preflight: dict[str, Any],
    *,
    started_at: str,
    duration_seconds: float,
    open_report: bool,
) -> int:
    write_validation_summary_report(
        preset=preset,
        output_root=output_root,
        records=records,
        failures=failures,
        preflight=preflight,
        started_at=started_at,
        duration_seconds=duration_seconds,
        open_report=open_report,
    )
    return validation_summary(failures)


def write_validation_summary_report(
    *,
    preset: str,
    output_root: Path,
    records: list[dict[str, Any]],
    failures: list[tuple[str, str, str]],
    preflight: dict[str, Any],
    started_at: str,
    duration_seconds: float,
    open_report: bool,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    status = "passed" if not failures else "failed"
    report_path = output_root / f"mlperf_validate_{preset}_{timestamp}.json"
    csv_path = report_path.with_suffix(".csv")
    html_path = report_path.with_suffix(".html")
    workload_csv_path = (
        output_root / f"mlperf_validate_workloads_{preset}_{timestamp}.csv"
    )
    workloads = validation_workload_rows(records)
    report = {
        "schema": "mlperf-edu-validation/0.1",
        "mlperf_suite": DEFAULT_MLPERF_SUITE,
        "preset": preset,
        "status": status,
        "started_at": started_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": float(duration_seconds),
        "preflight": preflight,
        "totals": {
            "validations": len(records),
            "passed_validations": sum(
                1 for record in records if record.get("status") == "passed"
            ),
            "failed_validations": sum(
                1
                for record in records
                if str(record.get("status", "")).endswith("_failed")
            ),
            "passed_manifests": sum(int(record.get("passed", 0)) for record in records),
            "failed_manifests": sum(int(record.get("failed", 0)) for record in records),
            "warning_count": sum(
                int(record.get("warning_count", 0)) for record in records
            ),
            "review_contract_failures": sum(
                int(record.get("review_contract_failure_count", 0))
                for record in records
            ),
            "failures": len(failures),
            "workloads": len(workloads),
            "duration_seconds": float(duration_seconds),
        },
        "validations": records,
        "workloads": workloads,
        "failures": [
            {"suite": suite, "profile": profile, "failure": failure}
            for suite, profile, failure in failures
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_validation_csv(report, csv_path)
    write_validation_workload_csv(report, workload_csv_path)
    write_validation_html(report, html_path, source_path=report_path)
    console.print(f"Validation JSON: {report_path}")
    console.print(f"Validation HTML: {html_path}")
    console.print(f"Validation CSV: {csv_path}")
    console.print(f"Validation Workloads CSV: {workload_csv_path}")
    if open_report:
        webbrowser.open(html_path.as_uri())
    return {"json": report_path, "csv": csv_path, "html": html_path}


def write_validation_csv(report: dict[str, Any], output: Path) -> None:
    fieldnames = [
        "validation",
        "suite",
        "profile",
        "status",
        "run_exit",
        "grade_exit",
        "duration_seconds",
        "passed",
        "failed",
        "warning_count",
        "output_dir",
        "report",
        "report_html",
        "report_csv",
        "grade_json",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in report.get("validations", []):
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def write_validation_workload_csv(report: dict[str, Any], output: Path) -> None:
    fieldnames = [
        "validation",
        "workload",
        "canonical_workload",
        "variant",
        "run_selector",
        "suite",
        "profile",
        "status",
        "backend",
        "data_mode",
        "dataset",
        "dataset_license_status",
        "dataset_public_release_status",
        "dataset_public_use",
        "dataset_release_next_step",
        "model_source",
        "model_license",
        "model_rationale",
        "shared_checkpoint",
        "quality_dependency",
        "checkpoint_source_selector",
        "checkpoint_source_quality",
        "checkpoint_artifact_policy",
        "metric",
        "value",
        "target",
        "target_basis",
        "reference_runs",
        "acceptance_runs",
        "reference_statistic",
        "reference_protocol",
        "quality_required",
        "target_met",
        "duration_seconds",
        "throughput",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.get("workloads", []):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def validation_artifact_links(record: dict[str, Any], *, base_dir: Path) -> str:
    links = []
    for label, key in (
        ("HTML", "report_html"),
        ("JSON", "report"),
        ("CSV", "report_csv"),
        ("Grade", "grade_json"),
    ):
        path = str(record.get(key, ""))
        if not path:
            continue
        href = relative_href(path, base_dir=base_dir)
        links.append(f"<a href='{escape(href)}'>{escape(label)}</a>")
    return " · ".join(links)


def relative_href(path: str, *, base_dir: Path) -> str:
    file_path = Path(path)
    try:
        return file_path.relative_to(base_dir).as_posix()
    except ValueError:
        return file_path.as_uri() if file_path.is_absolute() else file_path.as_posix()


def duration_sort_key(row: dict[str, Any]) -> float:
    try:
        return float(row.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def write_validation_html(
    report: dict[str, Any], output: Path, *, source_path: Path
) -> None:
    totals = report.get("totals") or {}
    base_dir = output.parent
    cards = "\n".join(
        f"<div class='card'><div class='label'>{escape(label)}</div><div class='value'>{escape(str(value))}</div></div>"
        for label, value in (
            ("Status", report.get("status", "unknown")),
            ("Validation Runs", totals.get("validations", 0)),
            ("Passed Manifests", totals.get("passed_manifests", 0)),
            ("Failed Manifests", totals.get("failed_manifests", 0)),
            ("Warnings", totals.get("warning_count", 0)),
            ("Workloads", totals.get("workloads", 0)),
            ("Duration", f"{float(report.get('duration_seconds', 0.0)):.1f}s"),
        )
    )
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(record.get('validation', '')))}</td>"
        f"<td>{escape(str(record.get('suite', '')))}</td>"
        f"<td>{escape(str(record.get('profile', '')))}</td>"
        f"<td><span class='badge {status_class(str(record.get('status', '')))}'>{escape(str(record.get('status', '')))}</span></td>"
        f"<td>{escape(format_cell(record.get('passed', '')))}</td>"
        f"<td>{escape(format_cell(record.get('failed', '')))}</td>"
        f"<td>{escape(format_cell(record.get('warning_count', '')))}</td>"
        f"<td>{escape(format_cell(record.get('duration_seconds', '')))}</td>"
        f"<td>{validation_artifact_links(record, base_dir=base_dir)}</td>"
        "</tr>"
        for record in report.get("validations", [])
    )
    workload_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row.get('validation', '')))}</td>"
        f"<td>{escape(str(row.get('workload', '')))}</td>"
        f"<td>{escape(str(row.get('suite', '')))}</td>"
        f"<td>{escape(str(row.get('profile', '')))}</td>"
        f"<td><span class='badge {status_class(str(row.get('status', '')))}'>{escape(str(row.get('status', '')))}</span></td>"
        f"<td>{escape(str(row.get('metric', '')))}</td>"
        f"<td>{escape(format_cell(row.get('value')))}</td>"
        f"<td>{escape(format_cell(row.get('target')))}</td>"
        f"<td>{escape(format_cell(row.get('quality_required')))}</td>"
        f"<td>{escape(format_cell(row.get('duration_seconds')))}</td>"
        f"<td>{escape(format_cell(row.get('throughput')))}</td>"
        "</tr>"
        for row in sorted(
            report.get("workloads", []), key=duration_sort_key, reverse=True
        )
    )
    workload_section = ""
    if workload_rows:
        workload_section = f"""
  <h2>Workload Breakdown</h2>
  <table>
    <thead><tr><th>Validation</th><th>Workload</th><th>Suite</th><th>Profile</th><th>Status</th><th>Metric</th><th>Value</th><th>Target</th><th>Quality Required</th><th>Duration</th><th>Throughput</th></tr></thead>
    <tbody>{workload_rows}</tbody>
  </table>
"""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MLPerf EDU Validation: {escape(str(report.get("preset", "unknown")))}</title>
  <style>
    :root {{ --bg:#f6f7f9; --ink:#1f2937; --muted:#667085; --line:#d0d5dd; --surface:#fff; --pass:#067647; --fail:#b42318; --warn:#b54708; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:32px 24px 48px; }}
    header {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:24px; }}
    h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:0; }}
    .meta,.note {{ color:var(--muted); font-size:12px; }}
    .meta {{ text-align:right; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:18px 0 24px; }}
    .card {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }}
    .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    .value {{ font-size:24px; font-weight:700; margin-top:4px; }}
    h2 {{ margin:28px 0 12px; font-size:18px; letter-spacing:0; }}
    table {{ width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; background:#eef2f6; }}
    tr:last-child td {{ border-bottom:0; }}
    .badge {{ display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; font-weight:600; }}
    .pass {{ color:var(--pass); background:#dcfae6; }}
    .fail {{ color:var(--fail); background:#fee4e2; }}
    .warn {{ color:var(--warn); background:#fef0c7; }}
    a {{ color:#175cd3; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>MLPerf EDU Validation: {escape(str(report.get("preset", "unknown")))}</h1>
      <div class="note">Schema: {escape(str(report.get("schema", "unknown")))} · Suite: {escape(str(report.get("mlperf_suite", "unknown")))}</div>
    </div>
    <div class="meta">
      <div>{escape(str(report.get("generated_at", "")))}</div>
      <div>{escape(str(source_path))}</div>
    </div>
  </header>
  <section class="grid">{cards}</section>
  <table>
    <thead><tr><th>Validation</th><th>Suite</th><th>Profile</th><th>Status</th><th>Passed</th><th>Failed</th><th>Warnings</th><th>Duration</th><th>Artifacts</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {workload_section}
  <div class="note" style="margin-top:18px">Generated by mlperf validate. Open per-suite reports for workload-level metrics.</div>
</main>
</body>
</html>
"""
    output.write_text(html)


def validation_summary(failures: list[tuple[str, str, str]]) -> int:
    if not failures:
        console.print("[green]Validation summary: all checks passed[/green]")
        return 0
    table = Table(title="MLPerf EDU Validation Failures")
    table.add_column("Validation")
    table.add_column("Profile")
    table.add_column("Failure")
    for suite, profile, failure in failures:
        table.add_row(suite, profile, failure)
    console.print(table)
    console.print(f"[red]Validation summary: {len(failures)} failure(s)[/red]")
    return 1


def select_discovery_workloads(
    workloads: dict[str, Workload],
    *,
    suite: str | None = None,
    profile: str | None = None,
    workload: str | None = None,
    variant: str | None = None,
    maturity: str | None = None,
    public_status: str | None = None,
) -> list[Workload]:
    if variant and not workload:
        raise ValueError("--variant requires --workload")

    if workload:
        if workload in workloads:
            selected_ids = (resolve_cli_workload_id(workloads, workload, variant),)
        else:
            ids = resolve_workload_ids(workloads, workload)
            if not ids:
                raise ValueError(f"unknown workload or canonical workload '{workload}'")
            if variant:
                selected_ids = (resolve_cli_workload_id(workloads, workload, variant),)
            else:
                selected_ids = ids
        selected = [
            workloads[workload_id]
            for workload_id in selected_ids
            if workload_id is not None
        ]
        if suite:
            selected = [item for item in selected if item.suite == suite]
        if maturity:
            selected = [item for item in selected if item.maturity == maturity]
        if public_status:
            selected = [
                item for item in selected if item.public_status == public_status
            ]
        return selected

    if profile and suite:
        return select_workloads(
            workloads,
            suite=suite,
            maturity=maturity,
            public_status=public_status,
        )

    collection = profile_collection(profile) if profile else None
    return select_workloads(
        workloads,
        suite=suite,
        collection=collection,
        maturity=maturity,
        public_status=public_status,
    )


def cmd_list(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    if args.subject == "suites":
        return list_suites(workloads, args.format)
    if args.subject == "profiles":
        return list_profiles(workloads, args.format)
    if args.subject == "variants":
        return list_variants(
            workloads,
            args.workload,
            args.format,
            suite=args.suite,
            maturity=args.maturity,
            public_status=args.public_status,
        )
    if args.subject == "matrix":
        return list_matrix(
            workloads,
            args.format,
            suite=args.suite,
            profile=args.profile,
            workload=args.workload,
            variant=getattr(args, "variant", None),
            maturity=args.maturity,
            public_status=args.public_status,
        )

    selected = select_discovery_workloads(
        workloads,
        suite=args.suite,
        profile=args.profile,
        workload=args.workload,
        variant=args.variant,
        maturity=args.maturity,
        public_status=args.public_status,
    )
    if args.format == "json":
        payload = {
            "schema": "mlperf-edu-list-workloads/0.1",
            "profile": args.profile,
            "suite": args.suite,
            "workload": args.workload,
            "variant": args.variant,
            "workloads": [workload_summary(workload) for workload in selected],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    table = Table(title="MLPerf EDU Workloads")
    table.add_column("Workload", no_wrap=True, min_width=44)
    table.add_column("Internal ID", no_wrap=True, min_width=22)
    table.add_column("Suite", no_wrap=True)
    table.add_column("Public", no_wrap=True, min_width=20)
    table.add_column("Quality", overflow="fold")
    for workload in selected:
        quality = "n/a"
        if workload.quality_metric:
            quality = f"{workload.quality_metric}={workload.quality_value}"
        elif isinstance(workload.raw.get("functional_check"), dict):
            quality = "functional: " + str(
                workload.raw["functional_check"].get("metric", "check")
            )
        table.add_row(
            workload_run_selector(workload),
            workload.id,
            workload.suite,
            workload.public_status,
            quality,
        )
    console.print(table)
    return 0


def list_matrix(
    workloads: dict[str, Workload],
    output_format: str,
    *,
    suite: str | None = None,
    profile: str | None = None,
    workload: str | None = None,
    variant: str | None = None,
    maturity: str | None = None,
    public_status: str | None = None,
) -> int:
    selected = select_discovery_workloads(
        workloads,
        suite=suite,
        profile=profile,
        workload=workload,
        variant=variant,
        maturity=maturity,
        public_status=public_status,
    )

    rows = [workload_matrix_row(workload) for workload in selected]
    if output_format == "json":
        payload = {
            "schema": "mlperf-edu-workload-matrix/0.1",
            "profile": profile,
            "suite": suite,
            "workload": workload,
            "variant": variant,
            "workloads": rows,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    table = Table(title="MLPerf EDU Workload Matrix")
    table.add_column("Workload", no_wrap=True, min_width=20)
    table.add_column("Run As", overflow="fold", min_width=34)
    table.add_column("Suite", no_wrap=True)
    table.add_column("Profiles", no_wrap=True)
    table.add_column("Role", no_wrap=True)
    table.add_column("Status", no_wrap=True, min_width=12)
    for row in rows:
        table.add_row(
            row["workload"],
            row["run_selector"],
            row["suite"],
            row["default_profiles"],
            row["role"],
            row["public_status"],
        )
    console.print(table)
    return 0


def workload_matrix_row(workload: Workload) -> dict[str, str]:
    canonical = canonical_workload_for_id(workload)
    return {
        "workload": workload.id,
        "canonical_workload": canonical or "",
        "variant": workload_variant_name(workload) if canonical else "",
        "run_selector": workload_run_selector(workload),
        "suite": workload.suite,
        "default_profiles": default_profiles_for_workload(workload),
        "role": workload_role(workload),
        "public_status": workload.public_status,
        "dataset": workload.dataset or "",
        "quality": workload_quality_summary(workload),
    }


def default_profiles_for_workload(workload: Workload) -> str:
    workload_id = workload.id
    profiles = []
    if workload_id in STARTER_WORKLOADS:
        profiles.append("min")
    if (workload.raw.get("runner") or {}).get("max"):
        profiles.append("max")
    if workload_id in RESEARCH_WORKLOADS:
        profiles.append("pro")
    elif (workload.raw.get("runner") or {}).get("max"):
        profiles.append("pro")
    return ", ".join(profiles) if profiles else "by-workload"


def workload_role(workload: Workload) -> str:
    if workload.scenario == "training":
        return "training"
    if workload.scenario in {"single_stream", "offline", "server"}:
        return "inference"
    return workload.scenario or "systems"


def workload_quality_summary(workload: Workload) -> str:
    if workload.quality_metric:
        parts = [str(workload.quality_metric)]
        if workload.quality_direction:
            parts.append(str(workload.quality_direction))
        if workload.quality_value is not None:
            parts.append(str(workload.quality_value))
        return " ".join(parts)
    if isinstance(workload.raw.get("functional_check"), dict):
        return "functional: " + str(
            workload.raw["functional_check"].get("metric", "check")
        )
    return ""


def list_suites(workloads: dict[str, Workload], output_format: str) -> int:
    rows = []
    for suite in PRODUCT_SUITES:
        suite_workloads = [
            workload for workload in workloads.values() if workload.suite == suite
        ]
        if not suite_workloads:
            continue
        rows.append(
            {
                "suite": suite,
                "workloads": len(suite_workloads),
                "score_bearing": sum(
                    1
                    for workload in suite_workloads
                    if workload.public_status == "score-bearing"
                ),
                "performance_bearing": sum(
                    1
                    for workload in suite_workloads
                    if workload.public_status == "performance-bearing"
                ),
                "systems_only": sum(
                    1
                    for workload in suite_workloads
                    if workload.public_status == "systems-only"
                ),
            }
        )
    if output_format == "json":
        print(
            json.dumps(
                {"schema": "mlperf-edu-list-suites/0.1", "suites": rows},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    table = Table(title="MLPerf EDU Suites")
    table.add_column("Suite", no_wrap=True)
    table.add_column("Workloads", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Performance", justify="right")
    table.add_column("Systems", justify="right")
    for row in rows:
        table.add_row(
            str(row["suite"]),
            str(row["workloads"]),
            str(row["score_bearing"]),
            str(row["performance_bearing"]),
            str(row["systems_only"]),
        )
    console.print(table)
    return 0


def list_profiles(workloads: dict[str, Workload], output_format: str) -> int:
    rows = []
    for profile in PROFILES:
        selected = select_workloads(workloads, collection=profile_collection(profile))
        rows.append(
            {
                "profile": profile,
                "workloads": len(selected),
                "description": PROFILE_DESCRIPTIONS[profile],
            }
        )
    if output_format == "json":
        print(
            json.dumps(
                {"schema": "mlperf-edu-list-profiles/0.1", "profiles": rows},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    table = Table(title="MLPerf EDU Profiles")
    table.add_column("Profile", no_wrap=True)
    table.add_column("Workloads", justify="right")
    table.add_column("Meaning")
    for row in rows:
        table.add_row(row["profile"], str(row["workloads"]), row["description"])
    console.print(table)
    return 0


def list_variants(
    workloads: dict[str, Workload],
    workload_id: str | None,
    output_format: str,
    *,
    suite: str | None = None,
    maturity: str | None = None,
    public_status: str | None = None,
) -> int:
    rows = variant_rows(
        workloads,
        workload_id,
        suite=suite,
        maturity=maturity,
        public_status=public_status,
    )
    if output_format == "json":
        print(
            json.dumps(
                {"schema": "mlperf-edu-list-variants/0.1", "variants": rows},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    table = Table(title="MLPerf EDU Variants")
    table.add_column("Canonical Workload", no_wrap=True)
    table.add_column("Variant", no_wrap=True)
    table.add_column("Registry ID", no_wrap=True)
    table.add_column("Suite", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Public", no_wrap=True)
    for row in rows:
        table.add_row(
            row["canonical_workload"],
            row["variant"],
            row["workload"],
            row["suite"],
            row["model"],
            row["public_status"],
        )
    console.print(table)
    return 0


def variant_rows(
    workloads: dict[str, Workload],
    workload_id: str | None = None,
    *,
    suite: str | None = None,
    maturity: str | None = None,
    public_status: str | None = None,
) -> list[dict[str, Any]]:
    canonical_ids: list[tuple[str, tuple[str, ...]]] = []
    if workload_id:
        canonical_ids = [(workload_id, resolve_workload_ids(workloads, workload_id))]
    else:
        canonical_ids = list(canonical_workload_groups(workloads).items())

    rows: list[dict[str, Any]] = []
    for canonical, ids in canonical_ids:
        for current_id in ids:
            if current_id not in workloads:
                continue
            workload = workloads[current_id]
            if suite and workload.suite != suite:
                continue
            if maturity and workload.maturity != maturity:
                continue
            if public_status and workload.public_status != public_status:
                continue
            rows.append(
                {
                    "canonical_workload": canonical,
                    "variant": workload_variant_name(workload),
                    "workload": current_id,
                    "suite": workload.suite,
                    "model": workload.model,
                    "public_status": workload.public_status,
                }
            )
    if workload_id and not rows:
        raise ValueError(f"unknown workload or canonical workload '{workload_id}'")
    return rows


def resolve_workload_ids(
    workloads: dict[str, Workload], workload_id: str
) -> tuple[str, ...]:
    if workload_id in workloads:
        return (workload_id,)
    groups = canonical_workload_groups(workloads)
    if workload_id in groups:
        return groups[workload_id]
    return ()


def canonical_workload_groups(
    workloads: dict[str, Workload],
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for workload in workloads.values():
        if workload.canonical_workload:
            groups.setdefault(workload.canonical_workload, []).append(workload.id)
    return {canonical: tuple(ids) for canonical, ids in groups.items()}


def canonical_default_variant(
    workloads: dict[str, Workload], canonical_workload: str
) -> str:
    ids = resolve_workload_ids(workloads, canonical_workload)
    for workload_id in ids:
        workload = workloads[workload_id]
        if workload.default_variant:
            return workload_variant_name(workload)
    for workload_id in ids:
        workload = workloads[workload_id]
        if workload_variant_name(workload) == "baseline":
            return "baseline"
    if ids:
        return workload_variant_name(workloads[ids[0]])
    return "baseline"


def registry_workload_for_id(
    workload: Workload | str,
    workloads: dict[str, Workload] | None = None,
) -> Workload | None:
    if isinstance(workload, Workload):
        return workload
    if workloads is not None:
        return workloads.get(workload)
    try:
        return load_registry().get(workload)
    except Exception:
        return None


def canonical_workload_for_id(
    workload: Workload | str,
    workloads: dict[str, Workload] | None = None,
) -> str | None:
    resolved = registry_workload_for_id(workload, workloads)
    return resolved.canonical_workload if resolved else None


def workload_variant_name(
    workload: Workload | str,
    workloads: dict[str, Workload] | None = None,
) -> str:
    resolved = registry_workload_for_id(workload, workloads)
    if resolved and resolved.variant:
        return resolved.variant
    return "baseline"


def workload_run_selector(workload: Workload) -> str:
    canonical = canonical_workload_for_id(workload)
    if not canonical:
        return workload.id
    return f"{canonical} --variant {workload_variant_name(workload)}"


def resolve_cli_workload_id(
    workloads: dict[str, Workload],
    workload_id: str | None,
    variant: str | None,
) -> str | None:
    if variant and not workload_id:
        raise ValueError("--variant requires --workload")
    if not workload_id:
        return None

    if workload_id in workloads:
        if variant and workload_variant_name(workloads[workload_id]) != variant:
            available = workload_variant_name(workloads[workload_id])
            raise ValueError(
                f"unknown variant '{variant}' for workload '{workload_id}'. Available: {available}"
            )
        return workload_id

    ids = resolve_workload_ids(workloads, workload_id)
    if not ids:
        raise ValueError(f"unknown workload or canonical workload '{workload_id}'")

    requested_variant = variant or canonical_default_variant(workloads, workload_id)
    for current_id in ids:
        if workload_variant_name(workloads[current_id]) == requested_variant:
            return current_id

    available = ", ".join(
        workload_variant_name(workloads[current_id]) for current_id in ids
    )
    raise ValueError(
        f"unknown variant '{requested_variant}' for workload '{workload_id}'. Available: {available}"
    )


def workload_summary(workload: Workload) -> dict[str, Any]:
    canonical = canonical_workload_for_id(workload)
    summary = {
        "workload": canonical or workload.id,
        "id": workload.id,
        "internal_id": workload.id,
        "run_selector": workload_run_selector(workload),
        "suite": workload.suite,
        "public_status": workload.public_status,
        "model": workload.model,
        "dataset": workload.dataset,
        "quality_metric": workload.quality_metric,
        "quality_value": workload.quality_value,
        "quality_target_basis": workload.quality_target_basis,
        "functional_check": functional_check_summary(
            workload.raw.get("functional_check")
        ),
    }
    if canonical:
        summary["canonical_workload"] = canonical
        summary["variant"] = workload_variant_name(workload)
    return summary


def cmd_show(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    if args.variant:
        ids = (resolve_cli_workload_id(workloads, args.workload, args.variant),)
    else:
        ids = resolve_workload_ids(workloads, args.workload)
        if not ids:
            console.print(f"[red]Unknown workload:[/red] {args.workload}")
            return 1
    if len(ids) > 1:
        console.print(f"[bold]Canonical workload:[/bold] {args.workload}")
        list_variants(workloads, args.workload, "summary")
        return 0
    workload = workloads[ids[0]]

    table = Table(title=f"Workload: {workload.id}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("suite", workload.suite)
    table.add_row("public_status", workload.public_status)
    table.add_row("public_rationale", workload.public_rationale)
    table.add_row("profiles", ", ".join(workload.supports_profiles))
    table.add_row("model", workload.model)
    table.add_row("dataset", workload.dataset or "")
    canonical = canonical_workload_for_id(workload)
    if canonical:
        table.add_row("canonical_workload", canonical)
        table.add_row("variant", workload_variant_name(workload))
        table.add_row("run_as", workload_run_selector(workload))
    table.add_row("scenario", workload.scenario or "")
    if workload.quality_metric:
        table.add_row("quality", f"{workload.quality_metric}={workload.quality_value}")
    console.print(table)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    if args.suite:
        return list_suites(
            {k: v for k, v in workloads.items() if v.suite == args.suite}, "summary"
        )
    if args.profile:
        console.print(f"[bold]Profile:[/bold] {args.profile}")
        console.print(PROFILE_DESCRIPTIONS[args.profile])
        selected = select_workloads(
            workloads, collection=profile_collection(args.profile)
        )
        print_run_selection(
            args.profile,
            selected,
            suite=None,
            workload=None,
            collection=None,
            variant=None,
        )
        console.print(f"List details: mlperf list --profile {args.profile}")
        return 0
    if args.workload:
        if args.variant:
            return cmd_show(
                argparse.Namespace(
                    registry=args.registry,
                    workload=args.workload,
                    variant=args.variant,
                )
            )
        return cmd_show(
            argparse.Namespace(
                registry=args.registry,
                workload=args.workload,
                variant=None,
            )
        )
    if args.model:
        matches = workloads_matching_model(workloads, args.model)
        return print_model_info(args.model, matches)
    if args.dataset:
        matches = [
            workload
            for workload in workloads.values()
            if workload.dataset == args.dataset
        ]
        return print_dataset_info(args.dataset, matches)
    if args.run:
        return cmd_report(
            argparse.Namespace(
                report=args.run, format="summary", output=None, open=False
            )
        )
    return 1


def profile_collection(profile: str) -> str:
    if profile == "min":
        return "starter"
    if profile == "max":
        return "all"
    return "research"


def print_info_matches(label: str, query: str, matches: list[Workload]) -> int:
    table = Table(title=f"{label}: {query}")
    table.add_column("Workload", no_wrap=True)
    table.add_column("Suite", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Dataset", overflow="fold")
    table.add_column("Public", no_wrap=True)
    for workload in matches:
        table.add_row(
            workload.id,
            workload.suite,
            workload.model,
            workload.dataset or "",
            workload.public_status,
        )
    console.print(table)
    return 0 if matches else 1


def print_dataset_info(dataset: str, matches: list[Workload]) -> int:
    dossier = asset_dossier(dataset)
    table = Table(title=f"Dataset: {dataset}")
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key in (
        "display_name",
        "source_url",
        "citation",
        "license",
        "license_status",
        "public_release_status",
        "public_result_use",
        "public_release_policy",
        "release_next_step",
        "license_evidence_url",
        "attribution",
        "terms_summary",
        "version",
        "expected_download_bytes",
        "expected_unpacked_bytes",
        "hash_policy",
    ):
        value = dossier.get(key)
        if value not in (None, ""):
            table.add_row(key, str(value))
    console.print(table)

    if matches:
        print_info_matches("Workloads using dataset", dataset, matches)
        return 0
    return 0 if dossier.get("license_status") != "unknown" else 1


def workloads_matching_model(
    workloads: dict[str, Workload], query: str
) -> list[Workload]:
    query_lower = query.lower()
    matches = []
    for workload in workloads.values():
        model_source = workload.raw.get("model_source") or {}
        values = [
            workload.model,
            str(model_source.get("repo_id", "")),
        ]
        if any(query_lower in value.lower() for value in values):
            matches.append(workload)
    return matches


def print_model_info(query: str, matches: list[Workload]) -> int:
    dossier = model_dossier_for_query(query, matches)
    if dossier:
        table = Table(title=f"Model: {query}")
        table.add_column("Field", no_wrap=True)
        table.add_column("Value", overflow="fold")
        for key in (
            "display_name",
            "id",
            "source_url",
            "provider",
            "license",
            "license_status",
            "selection_rationale",
            "size_rationale",
            "backend_rationale",
            "terms_summary",
        ):
            value = dossier.get(key)
            if value not in (None, ""):
                table.add_row(key, str(value))
        console.print(table)

    if matches:
        print_info_matches("Workloads using model", query, matches)
        return 0
    return 1


def model_dossier_for_query(query: str, matches: list[Workload]) -> dict[str, Any]:
    query_lower = query.lower()
    for workload in matches:
        model_source = workload.raw.get("model_source") or {}
        if (
            not isinstance(model_source, dict)
            or model_source.get("type") != "huggingface-pinned"
        ):
            continue
        model_id = str(model_source.get("repo_id") or workload.model)
        if (
            query_lower not in model_id.lower()
            and query_lower not in workload.model.lower()
        ):
            continue
        return huggingface_model_dossier(
            model_source, model_name=str(model_id), model_id=str(model_id)
        )
    return {}


def cmd_cache(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    selected = select_cli_workloads(workloads, args)
    rows = []
    for workload in selected:
        rows.extend(cache_asset_rows(workload))

    if args.format == "json":
        payload = {
            "schema": "mlperf-edu-cache/0.1",
            "action": args.action,
            "selection": cache_selection_summary(args, selected),
            "assets": rows,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        selection = cache_selection_summary(args, selected)
        console.print(
            f"Selected {selection['workloads']} workload(s) for {selection['label']}."
        )
        if not args.suite and not args.workload and args.profile == "min":
            console.print("Use --profile max to inspect assets for the full suite.")
        table = Table(title=f"MLPerf EDU Cache: {args.action}")
        table.add_column("Workload", no_wrap=True)
        table.add_column("Asset", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Path/Source", overflow="fold")
        table.add_column("Terms", overflow="fold")
        table.add_column("Digest", overflow="fold")
        for row in rows:
            style = (
                "green"
                if row["status"] in {"present", "embedded", "external"}
                else "red"
            )
            table.add_row(
                row["workload"],
                row["asset"],
                f"[{style}]{row['status']}[/{style}]",
                row["path"],
                cache_terms_cell(row),
                row.get("sha256", ""),
            )
        console.print(table)

    if args.action == "verify" and any(row["status"] == "missing" for row in rows):
        return 1
    return 0


def cache_terms_cell(row: dict[str, str]) -> str:
    license_status = row.get("license_status", "")
    release_status = row.get("public_release_status", "")
    if license_status and release_status:
        return f"{license_status}; release={release_status}"
    return license_status or release_status


def cache_selection_summary(
    args: argparse.Namespace, selected: list[Workload]
) -> dict[str, Any]:
    if args.workload:
        label = f"workload {args.workload}"
        if args.variant:
            label = f"{label}:{args.variant}"
    elif args.suite:
        label = f"suite {args.suite}"
    else:
        label = f"profile {args.profile}"
    return {
        "suite": args.suite or "",
        "profile": "" if args.suite or args.workload else args.profile,
        "workload": args.workload or "",
        "variant": args.variant or "",
        "label": label,
        "workloads": len(selected),
    }


def cache_asset_rows(workload: Workload) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    model_source = workload.raw.get("model_source") or {}
    if model_source.get("type") == "huggingface-pinned":
        model_id = str(model_source.get("repo_id") or workload.model)
        revision = str(model_source.get("revision") or "")
        dossier = huggingface_model_dossier(
            model_source, model_name=workload.model, model_id=model_id
        )
        rows.append(
            {
                "workload": workload.id,
                "asset": "model",
                "status": "external",
                "path": f"{model_id}@{revision}",
                "sha256": "pinned-file-manifest",
                "source": str(dossier.get("source_url", "")),
                "license": str(dossier.get("license", "")),
                "license_status": str(dossier.get("license_status", "")),
                "public_release_status": str(dossier.get("public_release_status", "")),
                "public_result_use": str(dossier.get("public_result_use", "")),
                "release_next_step": str(dossier.get("release_next_step", "")),
            }
        )

    dataset = workload.dataset
    dossier = asset_dossier(dataset, declared_source=workload.raw.get("dataset_source"))
    if not dataset:
        rows.append(
            {
                "workload": workload.id,
                "asset": "dataset",
                "status": "embedded",
                "path": "none declared",
                "sha256": "",
            }
        )
        return rows
    if dataset == "prompt-suite-local":
        rows.append(
            cache_dataset_row(
                workload,
                dataset,
                status="embedded",
                path=dataset,
                sha256="",
                dossier=dossier,
            )
        )
        return rows

    known_paths: dict[str, list[Path]] = {
        "tinyshakespeare": [
            tinyshakespeare_paths()["full"],
            tinyshakespeare_paths()["train"],
            tinyshakespeare_paths()["val"],
        ],
        "humaneval-plus": [
            humaneval_plus_paths()["archive"],
            humaneval_plus_paths()["dataset"],
        ],
        "bfcl-v4-non-live-ast": [
            bfcl_non_live_ast_paths()["data"] / relative
            for relative in BFCL_DATA_FILES
        ],
    }
    paths = (
        [
            edm_cifar10_paths()["checkpoint"],
            edm_cifar10_paths()["fid_reference"],
        ]
        if workload.id == "image-generation"
        else known_paths.get(dataset)
    )
    if not paths:
        rows.append(
            cache_dataset_row(
                workload,
                dataset,
                status="external",
                path=dataset,
                sha256="",
                dossier=dossier,
            )
        )
        return rows

    existing = [path for path in paths if path.exists()]
    status = "present" if len(existing) == len(paths) else "missing"
    digest = ""
    if status == "present" and len(existing) == 1 and existing[0].is_file():
        digest = f"sha256:{sha256_file(existing[0])}"
    rows.append(
        cache_dataset_row(
            workload,
            dataset,
            status=status,
            path=", ".join(str(path) for path in paths),
            sha256=digest,
            dossier=dossier,
        )
    )
    return rows


def cache_dataset_row(
    workload: Workload,
    dataset: str,
    *,
    status: str,
    path: str,
    sha256: str,
    dossier: dict[str, Any],
) -> dict[str, str]:
    return {
        "workload": workload.id,
        "asset": "dataset",
        "status": status,
        "path": path,
        "sha256": sha256,
        "source": str(dossier.get("source_url", "")),
        "license": str(dossier.get("license", "")),
        "license_status": str(dossier.get("license_status", "")),
        "public_release_status": str(dossier.get("public_release_status", "")),
        "public_result_use": str(dossier.get("public_result_use", "")),
        "release_next_step": str(dossier.get("release_next_step", "")),
    }


def load_draft_evidence_by_workload() -> dict[str, list[dict[str, Any]]]:
    """Load integrity-checked draft evidence for maintainer-facing audits."""
    index, evidence_root = draft_evidence_store()
    if index is None or evidence_root is None:
        return {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for entry in index.get("cases", []):
        if not isinstance(entry, dict):
            continue
        workload_id = str(entry.get("workload") or "")
        declared_path = str(entry.get("path") or "")
        if not workload_id or not declared_path:
            continue
        relative = PurePosixPath(declared_path)
        safe_path = (
            not relative.is_absolute()
            and len(relative.parts) == 2
            and relative.parts[0] == "provisional_results"
            and ".." not in relative.parts
        )
        case_path = evidence_root.joinpath(relative.name) if safe_path else None
        if case_path is None or not case_path.is_file():
            case = {
                **entry,
                "integrity_ok": False,
                "integrity_issue": "draft evidence file is missing or outside the project",
            }
        else:
            case_bytes = case_path.read_bytes()
            result = json.loads(case_bytes.decode("utf-8"))
            expected_sha256 = str(entry.get("sha256") or "").removeprefix("sha256:")
            observed_sha256 = hashlib.sha256(case_bytes).hexdigest()
            case = {
                **entry,
                "integrity_ok": bool(expected_sha256)
                and observed_sha256 == expected_sha256,
                "integrity_issue": ""
                if expected_sha256 and observed_sha256 == expected_sha256
                else "draft evidence hash does not match its index",
                "run_count": int((result.get("measurement") or {}).get("run_count", 0)),
                "review_eligible": bool(result.get("review_eligible", False)),
                "source_git_sha": result.get("source_git_sha"),
                "execution": result.get("execution") or {},
                "quality": draft_quality_summary(result),
                "repeatability": result.get("repeatability") or {},
            }
        evidence.setdefault(workload_id, []).append(case)
    return evidence


def draft_evidence_store() -> tuple[dict[str, Any] | None, Any | None]:
    """Return the draft index and its source-checkout or packaged resource root."""
    root = find_project_root()
    local_root = root / "provisional_results"
    local_index = local_root / "index.json"
    if local_index.is_file():
        return json.loads(local_index.read_text()), local_root
    try:
        packaged_root = resources.files("mlperf_edu").joinpath("provisional_results")
        packaged_index = packaged_root.joinpath("index.json")
    except (ModuleNotFoundError, FileNotFoundError):
        return None, None
    if not packaged_index.is_file():
        return None, None
    return json.loads(packaged_index.read_text(encoding="utf-8")), packaged_root


def draft_evidence_source_status() -> dict[str, Any]:
    """Describe whether draft evidence matches the active Git revision."""
    root = find_project_root()
    index, _evidence_root = draft_evidence_store()
    source_git_sha = index.get("source_git_sha") if index else None
    source_checkout = (root / "workloads.yaml").is_file() and (
        root / "provisional_results" / "index.json"
    ).is_file()
    if not source_checkout:
        return {
            "source_git_sha": source_git_sha,
            "current_git_sha": None,
            "current_git_dirty": None,
            "current_revision_match": None,
            "claim_scope": "unverified-installed-artifact",
        }
    try:
        current_git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        current_git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=root,
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        current_git_sha = None
        current_git_dirty = None
    revision_match = (
        source_git_sha == current_git_sha and current_git_dirty is False
        if source_git_sha and current_git_sha
        else None
    )
    return {
        "source_git_sha": source_git_sha,
        "current_git_sha": current_git_sha,
        "current_git_dirty": current_git_dirty,
        "current_revision_match": revision_match,
        "claim_scope": "current-source"
        if revision_match is True
        else "historical-draft"
        if revision_match is False
        else "unverified-installed-artifact",
    }


def draft_quality_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    quality = result.get("quality") or {}
    gate = quality.get("gate") or {}
    aggregate = quality.get("aggregate") or {}
    observed = aggregate.get("median")
    target = gate.get("target")
    direction = gate.get("direction")
    if not isinstance(observed, (int, float)) or not isinstance(target, (int, float)):
        return None
    tolerance = gate.get("tolerance", 0.0)
    if not isinstance(tolerance, (int, float)):
        tolerance = 0.0
    if direction == "higher":
        nominal_headroom = float(observed) - float(target)
    elif direction == "lower":
        nominal_headroom = float(target) - float(observed)
    else:
        return None
    denominator = abs(float(target))
    return {
        "metric": gate.get("metric") or quality.get("metric"),
        "observed_median": float(observed),
        "target": float(target),
        "direction": direction,
        "tolerance": float(tolerance),
        "nominal_headroom": nominal_headroom,
        "nominal_headroom_fraction": nominal_headroom / denominator
        if denominator
        else None,
        "gate_headroom": nominal_headroom + float(tolerance),
        "all_runs_pass": quality.get("all_runs_pass"),
    }


def summarize_draft_evidence(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "none"
    counts: dict[str, int] = {}
    for case in cases:
        evidence_class = str(case.get("evidence_class") or "unknown")
        counts[evidence_class] = counts.get(evidence_class, 0) + 1
    return ", ".join(
        f"{evidence_class}×{count}" for evidence_class, count in counts.items()
    )


def summarize_quality_headroom(cases: list[dict[str, Any]]) -> str:
    values = []
    for case in cases:
        quality = case.get("quality")
        if not isinstance(quality, dict):
            continue
        headroom = quality.get("nominal_headroom")
        if not isinstance(headroom, (int, float)):
            continue
        label = str(case.get("mode") or "")
        if case.get("phase"):
            label += f"/{case['phase']}"
        values.append(f"{label} {headroom:+.3g}")
    return "; ".join(values) if values else "functional"


def summarize_repeatability(cases: list[dict[str, Any]]) -> str:
    summaries = []
    for case in cases:
        repeatability = case.get("repeatability") or {}
        coefficient = repeatability.get("coefficient_of_variation")
        limit = repeatability.get("limit")
        if isinstance(coefficient, (int, float)) and isinstance(limit, (int, float)):
            state = "pass" if repeatability.get("passed") is True else "FAIL"
            summaries.append(f"{float(coefficient):.2%}/{float(limit):.0%} {state}")
        else:
            summaries.append("not established")
    return "; ".join(summaries) if summaries else "none"


def cmd_audit(args: argparse.Namespace) -> int:
    workloads = load_workloads(args)
    if args.variant and not args.workload:
        raise ValueError("--variant requires --workload")
    if args.workload and args.suite:
        raise ValueError("choose only one of --suite or --workload")
    if args.workload:
        if args.variant:
            selected_ids = (
                resolve_cli_workload_id(workloads, args.workload, args.variant),
            )
        else:
            selected_ids = resolve_workload_ids(workloads, args.workload)
            if not selected_ids:
                raise ValueError(
                    f"unknown workload or canonical workload '{args.workload}'"
                )
        selected = [
            workloads[workload_id]
            for workload_id in selected_ids
            if workload_id in workloads
        ]
        if args.public_status:
            selected = [
                workload
                for workload in selected
                if workload.public_status == args.public_status
            ]
    else:
        selected = select_workloads(
            workloads,
            suite=args.suite,
            collection=profile_collection(args.profile)
            if args.profile and not args.suite
            else None,
            public_status=args.public_status,
        )
    selected_ids = {workload.id for workload in selected}
    draft_evidence = load_draft_evidence_by_workload()
    draft_source = draft_evidence_source_status()
    issues_by_workload = {
        workload_id: issues
        for workload_id, issues in public_contract_report(workloads).items()
        if workload_id in selected_ids
    }
    warnings_by_workload = (
        {workload.id: public_audit_warnings(workload) for workload in selected}
        if args.policy == "public"
        else {workload.id: [] for workload in selected}
    )
    status_counts: dict[str, int] = {status: 0 for status in PUBLIC_STATUSES}
    for workload in selected:
        status_counts[workload.public_status] = (
            status_counts.get(workload.public_status, 0) + 1
        )
    failing = {
        workload_id: issues
        for workload_id, issues in issues_by_workload.items()
        if issues
    }
    warning_count = sum(len(warnings) for warnings in warnings_by_workload.values())
    warning_blocked = args.policy == "public" and warning_count > 0
    audit_passed = not failing and not warning_blocked
    issue_rows = [
        {
            "workload": workload.id,
            "run_selector": workload_run_selector(workload),
            "issue": issue,
        }
        for workload in selected
        for issue in issues_by_workload.get(workload.id, [])
    ]
    warning_rows = [
        {
            "workload": workload.id,
            "run_selector": workload_run_selector(workload),
            "warning": warning,
        }
        for workload in selected
        for warning in warnings_by_workload.get(workload.id, [])
    ]

    if args.format == "json":
        payload = {
            "schema": "mlperf-edu-public-contract-audit/0.2",
            "mlperf_suite": DEFAULT_MLPERF_SUITE,
            "status": "passed" if audit_passed else "failed",
            "policy": args.policy,
            "profile": args.profile,
            "suite": args.suite,
            "workload": args.workload,
            "variant": args.variant,
            "public_status": args.public_status,
            "counts": {
                status: count for status, count in status_counts.items() if count
            },
            "blocker_count": len(issue_rows),
            "warning_blocked": warning_blocked,
            "warning_count": warning_count,
            "draft_evidence_source": draft_source,
            "issues": issue_rows,
            "warnings": warning_rows,
            "workloads": [
                {
                    "id": workload.id,
                    "canonical_workload": canonical_workload_for_id(workload),
                    "variant": workload_variant_name(workload)
                    if canonical_workload_for_id(workload)
                    else None,
                    "run_selector": workload_run_selector(workload),
                    "suite": workload.suite,
                    "public_status": workload.public_status,
                    "scenario": workload.scenario,
                    "adapter_conformance": workload.raw.get("adapter_conformance"),
                    "draft_evidence": draft_evidence.get(workload.id, []),
                    "issues": issues_by_workload.get(workload.id, []),
                    "warnings": warnings_by_workload.get(workload.id, []),
                }
                for workload in selected
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0 if audit_passed else 1

    table = Table(title="MLPerf EDU Public Contract Audit")
    table.add_column("Workload", no_wrap=True)
    table.add_column("Run As", no_wrap=True, min_width=32)
    table.add_column("Suite", no_wrap=True)
    table.add_column("Public", no_wrap=True)
    table.add_column("Scenario", no_wrap=True)
    table.add_column("Draft evidence")
    table.add_column("Quality margin")
    table.add_column("Timing CV")
    table.add_column("Result", no_wrap=True)
    table.add_column("Issues")
    table.add_column("Warnings")
    for workload in selected:
        issues = issues_by_workload.get(workload.id, [])
        warnings = warnings_by_workload.get(workload.id, [])
        evidence = draft_evidence.get(workload.id, [])
        blocked = bool(issues) or (args.policy == "public" and bool(warnings))
        result = "[red]blocked[/red]" if blocked else "[green]ready[/green]"
        table.add_row(
            workload.id,
            workload_run_selector(workload),
            workload.suite,
            workload.public_status,
            workload.scenario or "",
            summarize_draft_evidence(evidence),
            summarize_quality_headroom(evidence),
            summarize_repeatability(evidence),
            result,
            "; ".join(issues),
            "; ".join(warnings),
        )
    console.print(table)
    console.print(
        "public contract audit: "
        + (
            "[green]passed[/green]"
            if audit_passed
            else f"[red]failed ({len(failing)} blocker workload(s), {warning_count if warning_blocked else 0} public warning(s))[/red]"
        )
    )
    console.print(
        "public status counts: "
        + ", ".join(
            f"{status}={count}" for status, count in status_counts.items() if count
        )
    )
    console.print(
        "draft evidence source: "
        f"{draft_source.get('source_git_sha') or 'unavailable'}; "
        f"current HEAD: {draft_source.get('current_git_sha') or 'unavailable'}; "
        f"scope: {draft_source['claim_scope']}"
    )
    console.print(f"public warnings: {warning_count}")
    return 0 if audit_passed else 1


def public_audit_warnings(workload: Workload) -> list[str]:
    warnings: list[str] = []
    if workload.public_status not in {"score-bearing", "performance-bearing"}:
        return warnings

    if workload.dataset:
        dossier = asset_dossier(
            workload.dataset, declared_source=workload.raw.get("dataset_source")
        )
        dataset_warning = dataset_public_release_warning(dossier)
        if dataset_warning:
            warnings.append(dataset_warning)

    baseline = workload.raw.get("verified_baseline") or {}
    if baseline.get("evidence_status") == "committed-reference-summary":
        availability = baseline.get("reference_package_availability")
        publication = baseline.get("external_publication_status")
        if availability != "published" or publication != "published":
            warnings.append(
                "external-publication blocker: registry declares local-handoff "
                "reference evidence, but no published package URL is recorded"
            )
        elif not baseline.get("external_publication_url"):
            warnings.append(
                "published reference evidence package lacks an external publication URL"
            )

    model_source = workload.raw.get("model_source")
    if (
        isinstance(model_source, dict)
        and model_source.get("type") == "huggingface-pinned"
    ):
        dossier = huggingface_model_dossier(
            model_source,
            model_name=workload.model,
            model_id=str(model_source.get("repo_id") or workload.model),
        )
        if dossier.get("license_status") == "requires-review":
            warnings.append(
                "model license status requires review before public endorsement"
            )

    return warnings


def dataset_public_release_warning(dossier: dict[str, Any]) -> str:
    release_status = str(dossier.get("public_release_status", "needs-release-decision"))
    next_step = str(
        dossier.get("release_next_step") or dossier.get("public_release_policy") or ""
    ).strip()
    if release_status in {
        "public-ok-bundled",
        "public-ok-with-attribution",
        "public-ok-fetch-only",
    }:
        return ""
    if release_status == "restricted-needs-approval":
        detail = f": {next_step}" if next_step else ""
        return f"dataset public release status is restricted-needs-approval{detail}"
    if release_status == "needs-release-decision":
        detail = f": {next_step}" if next_step else ""
        return f"dataset public release status is needs-release-decision{detail}"

    license_status = str(dossier.get("license_status", "unknown"))
    if license_status in {"requires-review", "unknown"}:
        return f"dataset license status is {license_status}; release policy must resolve before public endorsement"
    if license_status == "noncommercial-research-education":
        return "dataset terms are noncommercial/research/education; public-result policy needs explicit approval"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(normalized_argv)
    if hasattr(args, "profile"):
        args.profile = normalize_profile(args.profile)
    requested_device = getattr(args, "device", None)
    previous_device = os.environ.get("MLPERF_EDU_DEVICE")
    if requested_device == "auto":
        os.environ.pop("MLPERF_EDU_DEVICE", None)
    elif requested_device is not None:
        os.environ["MLPERF_EDU_DEVICE"] = requested_device
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    finally:
        if requested_device is not None:
            if previous_device is None:
                os.environ.pop("MLPERF_EDU_DEVICE", None)
            else:
                os.environ["MLPERF_EDU_DEVICE"] = previous_device


if __name__ == "__main__":
    sys.exit(main())
