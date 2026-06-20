"""Release orchestration for Binder.

`binder release` is the high-level release gate: it runs the checks that a
human release audit currently remembers by hand, classifies build/layout
outcomes, and writes one structured report for follow-up fixes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SCHEMA_VERSION = "binder-release/v1"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_OUTPUT_LIMIT = 5000
NETWORK_TIMEOUT_SECONDS = 900

STATUS_STYLES = {
    "passed": ("green", "PASS"),
    "passed_with_advisory": ("yellow", "PASS+ADVISORY"),
    "failed": ("red", "FAIL"),
    "planned": ("cyan", "PLAN"),
    "skipped": ("dim", "SKIP"),
}


@dataclass(frozen=True)
class ReleaseStage:
    id: str
    title: str
    category: str
    severity: str
    argv: tuple[str, ...]
    timeout_seconds: int | None = None
    network: bool = False
    build_stage: bool = False
    artifact_stage: bool = False
    allow_layout_advisory: bool = False


@dataclass
class StageResult:
    stage: ReleaseStage
    status: str
    exit_code: int | None
    elapsed_ms: int
    output_excerpt: str = ""
    skipped_reason: str = ""

    @property
    def blocking_failed(self) -> bool:
        return self.stage.severity == "required" and self.status == "failed"

    @property
    def advisory_failed(self) -> bool:
        return self.stage.severity == "advisory" and self.status == "failed"

    def to_dict(self, binder_label: str) -> dict[str, Any]:
        payload = {
            "id": self.stage.id,
            "title": self.stage.title,
            "category": self.stage.category,
            "severity": self.stage.severity,
            "status": self.status,
            "command": [binder_label, *self.stage.argv],
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.output_excerpt:
            payload["output_excerpt"] = self.output_excerpt
        if self.skipped_reason:
            payload["skipped_reason"] = self.skipped_reason
        return payload


def _stage(
    stage_id: str,
    title: str,
    category: str,
    severity: str,
    argv: Iterable[str],
    *,
    timeout_seconds: int | None = None,
    network: bool = False,
    build_stage: bool = False,
    artifact_stage: bool = False,
    allow_layout_advisory: bool = False,
) -> ReleaseStage:
    """Create a release stage with a stable ID and exact Binder argv."""
    return ReleaseStage(
        id=stage_id,
        title=title,
        category=category,
        severity=severity,
        argv=tuple(argv),
        timeout_seconds=timeout_seconds,
        network=network,
        build_stage=build_stage,
        artifact_stage=artifact_stage,
        allow_layout_advisory=allow_layout_advisory,
    )


SOURCE_STAGES: tuple[ReleaseStage, ...] = (
    _stage(
        "labels-vol1",
        "Volume I object labels",
        "source",
        "required",
        ("check", "labels", "--all-types", "--vol1", "--quiet"),
    ),
    _stage(
        "labels-vol2",
        "Volume II object labels",
        "source",
        "required",
        ("check", "labels", "--all-types", "--vol2", "--quiet"),
    ),
    _stage(
        "refs-inline",
        "Inline cross-reference hygiene",
        "source",
        "required",
        ("check", "refs", "--scope", "inline", "--quiet"),
    ),
    _stage(
        "refs-self-ref",
        "Self-referential prose audit",
        "source",
        "required",
        ("check", "refs", "--scope", "self-ref", "--quiet"),
    ),
    _stage(
        "sources-citations",
        "Source citation patterns",
        "source",
        "required",
        ("check", "sources", "--scope", "citations", "--quiet"),
    ),
    _stage(
        "prose-abbreviations",
        "Abbreviation first-use audit",
        "prose",
        "required",
        ("check", "prose", "--scope", "abbreviation-first-use", "--quiet"),
    ),
    _stage(
        "footnotes-cross-chapter",
        "Cross-chapter footnote audit",
        "objects",
        "required",
        ("check", "footnotes", "--scope", "cross-chapter", "--quiet"),
    ),
    _stage(
        "figures-flow",
        "Figure/object flow audit",
        "objects",
        "required",
        ("check", "figures", "--scope", "flow", "--quiet"),
    ),
    _stage(
        "lego-prose-units",
        "LEGO prose unit contract",
        "lego",
        "required",
        ("check", "code", "--scope", "lego-prose-units", "--quiet"),
    ),
    _stage(
        "lego-load-pint",
        "LEGO physical values load registry/Pint values",
        "lego",
        "required",
        ("check", "code", "--scope", "lego-load-pint", "--quiet"),
    ),
    _stage(
        "lego-equations",
        "LEGO equation/prose consistency",
        "lego",
        "required",
        ("check", "code", "--scope", "lego-equations", "--quiet"),
    ),
    _stage(
        "math-multiplier-style",
        "Multiplier prose style",
        "math",
        "required",
        ("check", "math", "--scope", "multiplier-style", "--quiet"),
    ),
    _stage(
        "math-suffix-semantics",
        "Typed suffix semantics",
        "math",
        "required",
        ("check", "math", "--scope", "suffix-semantics", "--quiet"),
    ),
    _stage(
        "content-tree",
        "Content tree schema",
        "infrastructure",
        "advisory",
        ("check", "content", "--scope", "tree", "--quiet"),
    ),
)

ARTIFACT_ADVISORY_STAGES: tuple[ReleaseStage, ...] = (
    _stage(
        "rendered-python-leak",
        "Rendered Python leak audit",
        "lego",
        "advisory",
        ("check", "code", "--scope", "rendered-python-leak", "--quiet"),
        artifact_stage=True,
    ),
)

NETWORK_ADVISORY_STAGES: tuple[ReleaseStage, ...] = (
    _stage(
        "bib-links",
        "Bibliography DOI/URL liveness",
        "network",
        "advisory",
        ("check", "bib", "--scope", "links", "--quiet"),
        network=True,
        timeout_seconds=NETWORK_TIMEOUT_SECONDS,
    ),
    _stage(
        "references-hallucinator",
        "Academic reference validation sample",
        "network",
        "advisory",
        (
            "check",
            "references",
            "--scope",
            "hallucinator",
            "--limit",
            "50",
            "--skip-verified",
            "--quiet",
        ),
        network=True,
        timeout_seconds=NETWORK_TIMEOUT_SECONDS,
    ),
)

PDF_VERIFY_SCOPES: tuple[tuple[str, str, str], ...] = (
    ("verify", "Verify {volume} PDF", "verify"),
    ("numbering", "Verify {volume} PDF numbering", "numbering"),
    ("table-spacing", "Verify {volume} table spacing", "table-spacing"),
)


class ReleaseCommand:
    """Run the Binder-native release workflow.

    Report contract:
        - stdout is pure JSON when --json is set.
        - every stage has a stable `id`, exact Binder `command`, `severity`,
          `status`, `exit_code`, and bounded `output_excerpt`.
        - required stages fail the command only when status == "failed";
          layout-planner rows after a successful PDF validation are classified
          as "passed_with_advisory".
    """

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        self.repo_root = self.config_manager.book_dir.parent.parent
        self.binder = self.repo_root / "book" / "binder"

    def run(self, args: list[str]) -> bool:
        parser = argparse.ArgumentParser(
            prog="binder release",
            description=(
                "Run the release gate: curated source checks, volume PDF builds, "
                "PDF verification, and optional network advisory checks."
            ),
        )
        volume_group = parser.add_mutually_exclusive_group()
        volume_group.add_argument(
            "--vol1", action="store_true", help="Run Volume I artifact stages only."
        )
        volume_group.add_argument(
            "--vol2", action="store_true", help="Run Volume II artifact stages only."
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the ordered workflow without running stages.",
        )
        parser.add_argument(
            "--json", action="store_true", help="Emit a structured JSON report to stdout."
        )
        parser.add_argument(
            "--output",
            help="Write JSON report to this path. Defaults under book/quarto/_build/release/.",
        )
        parser.add_argument(
            "--source-only",
            action="store_true",
            help="Run source checks only; skip builds and PDF artifact checks.",
        )
        parser.add_argument(
            "--skip-build",
            action="store_true",
            help="Skip PDF builds but still verify existing PDF artifacts.",
        )
        parser.add_argument(
            "--no-layout",
            action="store_true",
            help="Build PDFs without running the layout planner.",
        )
        parser.add_argument(
            "--include-network", action="store_true", help="Include network-bound advisory checks."
        )
        parser.add_argument(
            "--fail-on-advisory",
            action="store_true",
            help="Return nonzero for advisory-stage failures.",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=DEFAULT_TIMEOUT_SECONDS,
            help=f"Per-stage timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
        )
        parser.add_argument(
            "--output-limit",
            type=int,
            default=DEFAULT_OUTPUT_LIMIT,
            help="Maximum output excerpt chars per stage in the JSON report.",
        )

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            return ("-h" in args) or ("--help" in args)

        volumes = self._selected_volumes(ns)
        stages = self._build_stage_plan(volumes, include_layout=not ns.no_layout)
        stages = self._filter_stages(stages, ns)

        if ns.dry_run:
            results = [
                StageResult(stage=s, status="planned", exit_code=None, elapsed_ms=0) for s in stages
            ]
            payload = self._build_report(ns, volumes, results, status="planned")
            self._emit_report(payload, ns)
            return True

        results: list[StageResult] = []
        for index, stage in enumerate(stages, start=1):
            if not ns.json:
                console.print(
                    f"[cyan]release[/cyan] [{index}/{len(stages)}] "
                    f"{stage.id}: {stage.title}"
                )
            result = self._run_stage(stage, ns)
            results.append(result)
            if not ns.json:
                self._print_stage_result(result)

        required_failed = any(result.blocking_failed for result in results)
        advisory_failed = any(result.advisory_failed for result in results)
        failed = required_failed or (ns.fail_on_advisory and advisory_failed)
        status = "failed" if failed else "passed"
        payload = self._build_report(ns, volumes, results, status=status)
        self._emit_report(payload, ns)
        return not failed

    def _selected_volumes(self, ns: argparse.Namespace) -> list[str]:
        if ns.vol1:
            return ["vol1"]
        if ns.vol2:
            return ["vol2"]
        return ["vol1", "vol2"]

    def _build_stage_plan(self, volumes: list[str], *, include_layout: bool) -> list[ReleaseStage]:
        stages = list(SOURCE_STAGES)
        stages.extend(ARTIFACT_ADVISORY_STAGES)
        for volume in volumes:
            stages.append(self._build_pdf_stage(volume, include_layout=include_layout))
            stages.extend(self._pdf_verify_stages(volume))
        stages.append(self._final_check_stage(volumes))
        stages.extend(NETWORK_ADVISORY_STAGES)
        return stages

    def _build_pdf_stage(self, volume: str, *, include_layout: bool) -> ReleaseStage:
        argv = ("build", "pdf", f"--{volume}")
        if include_layout:
            argv = (*argv, "--layout")
        return _stage(
            f"build-{volume}-pdf",
            f"Build {self._volume_label(volume)} PDF",
            "artifact",
            "required",
            argv,
            build_stage=True,
            artifact_stage=True,
            allow_layout_advisory=include_layout,
        )

    def _pdf_verify_stages(self, volume: str) -> list[ReleaseStage]:
        volume_label = self._volume_label(volume)
        return [
            _stage(
                f"pdf-{volume}-{stage_id}",
                title.format(volume=volume_label),
                "artifact",
                "required",
                ("check", "pdf", "--scope", scope, f"--{volume}", "--quiet"),
                artifact_stage=True,
            )
            for scope, title, stage_id in PDF_VERIFY_SCOPES
        ]

    def _final_check_stage(self, volumes: list[str]) -> ReleaseStage:
        argv = ["check", "all", "--quiet"]
        if len(volumes) == 1:
            argv.insert(2, f"--{volumes[0]}")
        return _stage(
            "final-check-all",
            "Curated full check after artifacts exist",
            "final",
            "required",
            argv,
            artifact_stage=True,
        )

    def _filter_stages(
        self, stages: list[ReleaseStage], ns: argparse.Namespace
    ) -> list[ReleaseStage]:
        filtered = []
        for stage in stages:
            if stage.network and not ns.include_network:
                continue
            if ns.source_only and stage.artifact_stage:
                continue
            if ns.skip_build and stage.build_stage:
                continue
            filtered.append(stage)
        return filtered

    def _run_stage(self, stage: ReleaseStage, ns: argparse.Namespace) -> StageResult:
        started = time.time()
        timeout = stage.timeout_seconds or ns.timeout
        command = [sys.executable, str(self.binder), *stage.argv]
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1"})
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = completed.stdout or ""
            status = self._classify_status(stage, completed.returncode, output)
            return StageResult(
                stage=stage,
                status=status,
                exit_code=completed.returncode,
                elapsed_ms=int((time.time() - started) * 1000),
                output_excerpt=self._excerpt(output, ns.output_limit),
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            return StageResult(
                stage=stage,
                status="failed",
                exit_code=None,
                elapsed_ms=int((time.time() - started) * 1000),
                output_excerpt=self._excerpt(output, ns.output_limit),
                skipped_reason=f"timed out after {timeout}s",
            )

    def _classify_status(self, stage: ReleaseStage, exit_code: int, output: str) -> str:
        if exit_code == 0:
            return "passed"
        if stage.allow_layout_advisory and exit_code == 1:
            if (
                "PDF validation failed" not in output
                and "no unresolved refs or render errors in text" in output
                and "PASS: 0 Purpose overflow" in output
            ):
                return "passed_with_advisory"
        return "failed"

    def _print_stage_result(self, result: StageResult) -> None:
        style, label = STATUS_STYLES.get(result.status, ("white", result.status.upper()))
        console.print(
            f"  [{style}]{label}[/{style}] "
            f"exit={result.exit_code if result.exit_code is not None else '-'} "
            f"elapsed={result.elapsed_ms}ms"
        )

    def _build_report(
        self,
        ns: argparse.Namespace,
        volumes: list[str],
        results: list[StageResult],
        *,
        status: str,
    ) -> dict[str, Any]:
        binder_label = "book/binder"
        counts: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        head = self._git(["rev-parse", "--short=10", "HEAD"])
        branch = self._git(["branch", "--show-current"])
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "repo": {
                "root": str(self.repo_root),
                "branch": branch,
                "head": head,
            },
            "options": {
                "volumes": volumes,
                "source_only": ns.source_only,
                "skip_build": ns.skip_build,
                "layout": not ns.no_layout,
                "include_network": ns.include_network,
                "fail_on_advisory": ns.fail_on_advisory,
                "dry_run": ns.dry_run,
            },
            "summary": {
                "total": len(results),
                "counts": counts,
                "required_failures": sum(1 for result in results if result.blocking_failed),
                "advisory_failures": sum(1 for result in results if result.advisory_failed),
            },
            "stages": [result.to_dict(binder_label) for result in results],
        }

    def _emit_report(self, payload: dict[str, Any], ns: argparse.Namespace) -> None:
        report_path = None
        if ns.output:
            report_path = Path(ns.output)
            if not report_path.is_absolute():
                report_path = self.repo_root / report_path
        elif not ns.dry_run:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            report_path = (
                self.config_manager.book_dir
                / "_build"
                / "release"
                / f"release-{stamp}.json"
            )

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            payload["report_path"] = str(report_path)
            report_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        if ns.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return

        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Status", width=16)
        table.add_column("Count", justify="right")
        for key in ("passed", "passed_with_advisory", "failed", "planned", "skipped"):
            value = payload["summary"]["counts"].get(key, 0)
            if value:
                table.add_row(key, str(value))
        console.print(Panel(table, title="binder release summary", border_style="cyan"))
        if report_path is not None:
            console.print(f"[dim]JSON report: {report_path}[/dim]")
        if payload["status"] == "failed":
            console.print(
                "[red]Release gate failed. Inspect failed stage output in the JSON report.[/red]"
            )
        elif payload["status"] == "planned":
            console.print("[green]Release plan generated. No stages were run.[/green]")
        else:
            console.print("[green]Release gate passed.[/green]")

    def _git(self, args: Iterable[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return completed.stdout.strip() if completed.returncode == 0 else ""

    def _volume_label(self, volume: str) -> str:
        return "Volume I" if volume == "vol1" else "Volume II"

    def _excerpt(self, text: str, limit: int) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit] + "\n... [truncated]"
