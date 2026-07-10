#!/usr/bin/env python3
"""Generate the MLPerf EDU documentation site from the workload registry.

Single source of truth
----------------------
Every fact on the generated pages comes from exactly one of:

  * ``registry/suites/**``      — per-workload / per-variant benchmark metadata
  * ``registry/suites.yaml``    — suite-level titles and summaries
  * ``datasets.yaml``           — dataset catalog
  * the live ``mlperf`` CLI     — ``--help`` text for the command reference

Nothing on these pages is hand-written. To change a page, change the
registry (or the CLI help text) and regenerate:

    python3 tools/generate_docs.py            # rewrite generated pages
    python3 tools/generate_docs.py --check    # CI drift gate (no writes)

Generated outputs (the generator owns these paths entirely):

    site/benchmarks/**            one page per workload family + indexes
    site/reference/cli.qmd        CLI reference from --help output
    site/reference/datasets.qmd   dataset catalog + usage map
    site/_stats.qmd               include partial (landing-page stats)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from mlperf.registry import Workload, load_registry  # noqa: E402

GITHUB_BLOB = "https://github.com/harvard-edge/cs249r_book/blob/main/mlperf-edu"

GENERATED_NOTE = (
    "<!-- GENERATED FILE — do not edit by hand.\n"
    "     Source of truth: registry/ + datasets.yaml + the mlperf CLI.\n"
    "     Regenerate with: python3 tools/generate_docs.py -->\n"
)

CLI_COMMANDS = [
    "doctor",
    "init",
    "list",
    "show",
    "info",
    "fetch",
    "run",
    "verify",
    "report",
    "package",
    "audit",
    "validate",
    "grade",
    "cache",
]


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    """Escape a scalar for use inside a markdown pipe table cell."""
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def badge(status: str) -> str:
    slug = status.replace(" ", "-")
    return f'<span class="badge status-{slug}">{status}</span>'


def kv_table(rows: list[tuple[str, Any]]) -> str:
    """Render a two-column field table, skipping empty values."""
    kept = [(k, v) for k, v in rows if v not in (None, "", [], {})]
    if not kept:
        return ""
    out = ["| **Field** | **Value** |", "|:---|:---|"]
    for key, value in kept:
        out.append(f"| **{esc(key)}** | {esc(value)} |")
    return "\n".join(out) + "\n"


def bullet_block(title: str, items: list[Any]) -> str:
    if not items:
        return ""
    lines = [f"**{title}:**", ""]
    lines += [f"- {esc(item)}" for item in items]
    return "\n".join(lines) + "\n"


def workload_href(suite: str, family: str, depth: int) -> str:
    prefix = "../" * depth
    return f"{prefix}benchmarks/{suite}/{family}.qmd"


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------

def load_suite_meta() -> dict[str, dict[str, str]]:
    path = ROOT / "registry" / "suites.yaml"
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    suites = data.get("suites", {})
    if not isinstance(suites, dict):
        raise ValueError("registry/suites.yaml must define a 'suites' mapping")
    return suites


def index_source_paths() -> dict[str, str]:
    """Map workload id -> registry source path (repo-relative)."""
    paths: dict[str, str] = {}
    suites_dir = ROOT / "registry" / "suites"
    for suite_dir in sorted(p for p in suites_dir.iterdir() if p.is_dir()):
        for item in sorted(suite_dir.iterdir()):
            if item.is_file() and item.suffix in {".yaml", ".yml"}:
                raw = yaml.safe_load(item.read_text()) or {}
                workload_id = str(raw.get("id") or item.stem)
                paths[workload_id] = str(item.relative_to(ROOT))
            elif item.is_dir():
                base_path = item / "workload.yaml"
                base = yaml.safe_load(base_path.read_text()) if base_path.is_file() else {}
                canonical = str((base or {}).get("id") or item.name)
                variants_dir = item / "variants"
                if variants_dir.is_dir():
                    for variant_path in sorted(variants_dir.glob("*.y*ml")):
                        raw = yaml.safe_load(variant_path.read_text()) or {}
                        variant_name = str(raw.get("variant") or variant_path.stem)
                        workload_id = str(raw.get("id") or f"{canonical}-{variant_name}")
                        paths[workload_id] = str(variant_path.relative_to(ROOT))
                else:
                    paths[canonical] = str(base_path.relative_to(ROOT))
    return paths


def group_families(workloads: dict[str, Workload]) -> "OrderedDict[str, list[Workload]]":
    """Group workloads into families keyed by canonical workload id."""
    families: OrderedDict[str, list[Workload]] = OrderedDict()
    for workload in workloads.values():
        family = workload.canonical_workload or workload.id
        families.setdefault(family, []).append(workload)
    return families


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def section_at_a_glance(w: Workload) -> str:
    raw = w.raw
    rows = [
        ("Suite", w.suite),
        ("Model", w.model),
        ("Parameters", raw.get("params")),
        ("Dataset", w.dataset),
        ("Dataset source", raw.get("dataset_source")),
        ("Scenario", w.scenario),
        ("Maturity", w.maturity),
        ("Public status", w.public_status),
        ("Provenance", raw.get("provenance")),
    ]
    body = kv_table(rows)
    note = raw.get("params_note")
    if note:
        body += f"\n> **Parameter count note:** {esc(note)}\n"
    return f"## At a Glance\n\n{body}"


def section_how_to_run(w: Workload) -> str:
    lines = ["## How to Run", "", "```bash"]
    target = (
        f"--workload {w.canonical_workload} --variant {w.variant}"
        if w.variant
        else f"--workload {w.id}"
    )
    model_source = w.raw.get("model_source") or {}
    model_flag = ""
    if isinstance(model_source, dict) and model_source.get("default_alias"):
        model_flag = f" --model {model_source['default_alias']}"
    lines.append("# one-time asset preparation")
    lines.append(f"mlperf fetch {target} --profile max")
    lines.append("")
    lines.append("# benchmark run (writes JSON/HTML/CSV reports + .provd provenance)")
    lines.append(f"mlperf run {target} --profile max{model_flag} --open-report")
    lines.append("")
    lines.append("# quick smoke pass")
    lines.append(f"mlperf run {target} --profile min{model_flag}")
    lines.append("")
    lines.append("# research envelope")
    lines.append(f"mlperf run {target} --profile pro{model_flag}")
    lines.append("```")
    lines.append("")
    lines.append(
        "See the [running guide](../../guide/running.qmd) for profile semantics, "
        "report handling, and power measurement flags."
    )
    return "\n".join(lines) + "\n"


def section_quality_target(w: Workload) -> str:
    if not w.quality_metric:
        return ""
    rows = [
        ("Metric", w.quality_metric),
        ("Target", w.quality_value),
        ("Direction", w.quality_direction),
        ("Target basis", w.quality_target_basis),
        ("Tolerance", w.quality_tolerance),
        ("Reference runs", w.quality_reference_runs),
    ]
    body = kv_table(rows)

    variance = w.quality_variance_summary
    if isinstance(variance, dict) and variance:
        body += "\n**Variance summary:**\n\n" + kv_table(
            [(k.replace("_", " ").capitalize(), v) for k, v in variance.items()]
        )

    protocol = w.quality_reference_protocol
    if isinstance(protocol, dict) and protocol:
        rows = []
        for key, value in protocol.items():
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            rows.append((key.replace("_", " ").capitalize(), value))
        body += "\n**Reference protocol:**\n\n" + kv_table(rows)

    if w.quality_reviewer_notes:
        body += "\n" + bullet_block("Reviewer notes", list(w.quality_reviewer_notes))
    return f"## Quality Target\n\n{body}"


def section_verified_baseline(w: Workload) -> str:
    baseline = w.raw.get("verified_baseline")
    if not isinstance(baseline, dict) or not baseline:
        return ""
    note = baseline.get("baseline_note")
    rows = [
        (k.replace("_", " ").capitalize(), v)
        for k, v in baseline.items()
        if k != "baseline_note"
    ]
    body = kv_table(rows)
    if note:
        body += f"\n> {esc(note)}\n"
    return f"## Verified Baseline\n\n{body}"


def section_regime(w: Workload) -> str:
    regime = w.raw.get("regime")
    if not isinstance(regime, dict) or not regime:
        return ""
    parts = ["## Measured Systems Regime", ""]
    labels = {
        "working_set": "Working set",
        "arithmetic_intensity": "Arithmetic intensity",
        "dispatch": "Dispatch",
    }
    for key, label in labels.items():
        entry = regime.get(key)
        if not isinstance(entry, dict):
            continue
        rows = []
        for field, value in entry.items():
            rows.append((field.replace("_", " ").capitalize(), value))
        parts.append(f"**{label}:**\n")
        parts.append(kv_table(rows))
    return "\n".join(parts)


def section_model_source(w: Workload) -> str:
    source = w.raw.get("model_source")
    if not isinstance(source, dict) or not source:
        return ""
    rows = [
        ("Type", source.get("type")),
        ("Default model", source.get("default_model_id")),
        ("Default alias", source.get("default_alias")),
        ("License", source.get("license")),
    ]
    body = kv_table(rows)
    aliases = source.get("aliases")
    rationales = source.get("alias_rationales") or {}
    if isinstance(aliases, dict) and aliases:
        body += "\n**Model aliases (`--model <alias>`):**\n\n"
        body += "| **Alias** | **Model** | **Why it is offered** |\n|:---|:---|:---|\n"
        for alias, model_id in aliases.items():
            body += f"| `{esc(alias)}` | {esc(model_id)} | {esc(rationales.get(alias, ''))} |\n"
    for key in ("selection_rationale", "size_rationale", "backend_rationale"):
        if source.get(key):
            title = key.replace("_", " ").capitalize()
            body += f"\n**{title}:** {esc(source[key])}\n"
    return f"## Model Source\n\n{body}"


def section_runner(w: Workload, source_paths: dict[str, str]) -> str:
    runner = w.raw.get("runner")
    parts = ["## Implementation", ""]
    if isinstance(runner, dict) and runner:
        rows = [(profile, f"`{entry}`") for profile, entry in runner.items()]
        parts.append("| **Profile** | **Runner entry point** |\n|:---|:---|")
        for profile, entry in rows:
            parts.append(f"| {esc(profile)} | {entry} |")
        parts.append("")
    source = source_paths.get(w.id)
    if source:
        parts.append(f"Registry source: [`{source}`]({GITHUB_BLOB}/{source})")
        parts.append("")
    return "\n".join(parts)


def render_workload_body(w: Workload, source_paths: dict[str, str], heading_shift: bool) -> str:
    sections = [
        section_at_a_glance(w),
        section_how_to_run(w),
        section_quality_target(w),
        section_verified_baseline(w),
        section_regime(w),
        section_model_source(w),
        section_runner(w, source_paths),
    ]
    body = "\n".join(s for s in sections if s)
    if heading_shift:
        body = body.replace("\n## ", "\n### ").replace("## ", "### ", 1)
    return body


def public_line(w: Workload) -> str:
    return f"{badge(w.public_status)}\n\n> {esc(w.public_rationale)}\n" if w.public_rationale else badge(w.public_status) + "\n"


def family_page(family: str, members: list[Workload], source_paths: dict[str, str]) -> str:
    lead = members[0]
    suite = lead.suite
    lines = [
        "---",
        f'title: "{family}"',
        f'subtitle: "{suite} suite"',
        "---",
        "",
        GENERATED_NOTE,
    ]
    if len(members) == 1:
        w = members[0]
        lines.append(public_line(w))
        lines.append(render_workload_body(w, source_paths, heading_shift=False))
        return "\n".join(lines) + "\n"

    lines.append(
        f"This workload family exposes **{len(members)} measured variants** "
        f"of `{family}`. Every variant shares the family's model and dataset "
        "contract; each row below is independently runnable and reported."
    )
    lines.append("")
    lines.append("| **Variant** | **Workload ID** | **Scenario** | **Public status** |")
    lines.append("|:---|:---|:---|:---|")
    for member in members:
        anchor = f"#variant-{member.variant}"
        lines.append(
            f"| [{esc(member.variant)}]({anchor}) | `{esc(member.id)}` "
            f"| {esc(member.scenario or '')} | {esc(member.public_status)} |"
        )
    lines.append("")
    for member in members:
        lines.append(f"## Variant: {member.variant} {{#variant-{member.variant}}}")
        lines.append("")
        lines.append(public_line(member))
        lines.append(render_workload_body(member, source_paths, heading_shift=True))
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Index pages
# ---------------------------------------------------------------------------

def family_row(family: str, members: list[Workload], depth: int) -> str:
    lead = members[0]
    variants = f"{len(members)}" if len(members) > 1 else "—"
    statuses = sorted({m.public_status for m in members})
    href = workload_href(lead.suite, family, depth)
    return (
        f"| [`{family}`]({href}) | {esc(lead.suite)} | {esc(lead.model)} "
        f"| {esc(lead.raw.get('params') or '')} | {esc(lead.dataset or '')} "
        f"| {variants} | {esc(', '.join(statuses))} |"
    )


FAMILY_TABLE_HEADER = (
    "| **Workload** | **Suite** | **Model** | **Params** | **Dataset** "
    "| **Variants** | **Public status** |\n"
    "|:---|:---|:---|:---|:---|:---|:---|"
)


def benchmarks_index(
    families: "OrderedDict[str, list[Workload]]",
    suite_meta: dict[str, dict[str, str]],
    workloads: dict[str, Workload],
) -> str:
    by_suite: OrderedDict[str, list[str]] = OrderedDict()
    for suite in sorted(suite_meta):
        by_suite[suite] = []
    for family, members in families.items():
        by_suite.setdefault(members[0].suite, []).append(family)

    lines = [
        "---",
        'title: "Benchmark Registry"',
        "toc: true",
        "---",
        "",
        GENERATED_NOTE,
        f"MLPerf EDU currently registers **{len(workloads)} workloads** in "
        f"**{len(families)} families** across **{sum(1 for s in by_suite.values() if s)} suites**. "
        "Every page in this section is generated from the "
        f"[workload registry]({GITHUB_BLOB}/registry); the registry YAML is the "
        "single source of truth for models, datasets, quality targets, and "
        "public-result status.",
        "",
    ]
    for suite, suite_families in by_suite.items():
        if not suite_families:
            continue
        meta = suite_meta.get(suite, {})
        title = meta.get("title", suite)
        lines.append(f"## {title} (`{suite}`)")
        lines.append("")
        if meta.get("summary"):
            lines.append(esc(meta["summary"]))
            lines.append("")
        lines.append(FAMILY_TABLE_HEADER)
        for family in suite_families:
            lines.append(family_row(family, families[family], depth=1))
        lines.append("")
    return "\n".join(lines) + "\n"


def suite_index(
    suite: str,
    meta: dict[str, str],
    suite_families: list[tuple[str, list[Workload]]],
) -> str:
    lines = [
        "---",
        f'title: "{meta.get("title", suite)}"',
        f'subtitle: "`{suite}` suite"',
        "---",
        "",
        GENERATED_NOTE,
    ]
    if meta.get("summary"):
        lines += [esc(meta["summary"]), ""]
    if meta.get("focus"):
        lines += [f"**Systems focus:** {esc(meta['focus'])}", ""]
    lines.append(FAMILY_TABLE_HEADER)
    for family, members in suite_families:
        lines.append(family_row(family, members, depth=2))
    lines.append("")
    lines.append("```bash")
    lines.append(f"# run every {suite} workload in the max profile")
    lines.append(f"mlperf fetch --suite {suite} --profile max")
    lines.append(f"mlperf run --suite {suite} --profile max")
    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Datasets and CLI reference
# ---------------------------------------------------------------------------

def datasets_page(workloads: dict[str, Workload]) -> str:
    with (ROOT / "datasets.yaml").open("r") as handle:
        catalog = (yaml.safe_load(handle) or {}).get("datasets", {})

    usage: dict[str, list[Workload]] = {}
    for w in workloads.values():
        if w.dataset:
            usage.setdefault(w.dataset, []).append(w)

    lines = [
        "---",
        'title: "Dataset Catalog"',
        "---",
        "",
        GENERATED_NOTE,
        "Datasets are deliberately small: deterministic excerpts, synthetic "
        "generators, and torchvision downloads sized for laptops. The catalog "
        f"below is generated from [`datasets.yaml`]({GITHUB_BLOB}/datasets.yaml).",
        "",
        "| **Dataset** | **Description** | **Size (MB)** | **Source** | **Used by** |",
        "|:---|:---|:---|:---|:---|",
    ]
    for name in sorted(set(catalog) | set(usage)):
        entry = catalog.get(name, {})
        users = usage.get(name, [])
        links = ", ".join(
            f"[`{w.id}`](../benchmarks/{w.suite}/{w.canonical_workload or w.id}.qmd)"
            for w in sorted(users, key=lambda item: item.id)
        )
        lines.append(
            f"| `{esc(name)}` | {esc(entry.get('description', ''))} "
            f"| {esc(entry.get('estimated_size_mb', ''))} "
            f"| {esc(entry.get('uri', ''))} | {links or '—'} |"
        )
    lines += [
        "",
        "Fetch and inspect assets with:",
        "",
        "```bash",
        "mlperf fetch --profile max --dry-run   # show what would be downloaded",
        "mlperf info --dataset tinyshakespeare  # one dataset's dossier",
        "mlperf cache list                      # inspect the local cache",
        "```",
    ]
    return "\n".join(lines) + "\n"


def capture_help(args: list[str]) -> str:
    env = dict(os.environ, COLUMNS="80", PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "mlperf_edu", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=True,
    )
    return result.stdout.strip()


def cli_page() -> str:
    lines = [
        "---",
        'title: "CLI Reference"',
        "toc: true",
        "---",
        "",
        GENERATED_NOTE,
        "The `mlperf` command is the entire user surface. This reference is "
        "generated directly from the CLI's own `--help` output, so it cannot "
        "drift from the implementation.",
        "",
        "## mlperf",
        "",
        "```text",
        capture_help(["--help"]),
        "```",
        "",
    ]
    for command in CLI_COMMANDS:
        lines += [
            f"## mlperf {command}",
            "",
            "```text",
            capture_help([command, "--help"]),
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


def stats_partial(
    workloads: dict[str, Workload],
    families: "OrderedDict[str, list[Workload]]",
    suite_meta: dict[str, dict[str, str]],
) -> str:
    status_counts: dict[str, int] = {}
    for w in workloads.values():
        status_counts[w.public_status] = status_counts.get(w.public_status, 0) + 1
    suites_used = {members[0].suite for members in families.values()}
    status_text = ", ".join(
        f"{count} {status}" for status, count in sorted(status_counts.items())
    )
    lines = [
        GENERATED_NOTE,
        f"**{len(workloads)} workloads** · **{len(families)} families** · "
        f"**{len(suites_used)} suites** · {status_text}",
        "",
        "| **Suite** | **What it measures** | **Workloads** |",
        "|:---|:---|:---|",
    ]
    for suite in sorted(suites_used):
        meta = suite_meta.get(suite, {})
        count = sum(1 for members in families.values() if members[0].suite == suite)
        lines.append(
            f"| [{esc(meta.get('title', suite))}](benchmarks/{suite}/index.qmd) "
            f"| {esc(meta.get('summary', ''))} | {count} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def normalize(content: str) -> str:
    """Match the repo pre-commit contract: no trailing whitespace on any
    line, exactly one trailing newline. The generator must emit hook-clean
    text or every regeneration would fight the end-of-file/whitespace
    fixers."""
    lines = [line.rstrip() for line in content.split("\n")]
    return "\n".join(lines).rstrip("\n") + "\n"


def build_outputs() -> dict[Path, str]:
    workloads = load_registry(ROOT / "registry")
    suite_meta = load_suite_meta()
    source_paths = index_source_paths()
    families = group_families(workloads)

    unknown = {m[0].suite for m in families.values()} - set(suite_meta)
    if unknown:
        raise ValueError(
            f"suites missing from registry/suites.yaml: {sorted(unknown)}"
        )

    site = ROOT / "site"
    outputs: dict[Path, str] = {}
    outputs[site / "benchmarks" / "index.qmd"] = benchmarks_index(
        families, suite_meta, workloads
    )
    per_suite: OrderedDict[str, list[tuple[str, list[Workload]]]] = OrderedDict()
    for family, members in families.items():
        per_suite.setdefault(members[0].suite, []).append((family, members))
    for suite, suite_families in per_suite.items():
        outputs[site / "benchmarks" / suite / "index.qmd"] = suite_index(
            suite, suite_meta.get(suite, {}), suite_families
        )
        for family, members in suite_families:
            outputs[site / "benchmarks" / suite / f"{family}.qmd"] = family_page(
                family, members, source_paths
            )
    outputs[site / "reference" / "datasets.qmd"] = datasets_page(workloads)
    outputs[site / "reference" / "cli.qmd"] = cli_page()
    outputs[site / "_stats.qmd"] = stats_partial(
        workloads, families, suite_meta
    )
    return {path: normalize(content) for path, content in outputs.items()}


def managed_existing(site: Path) -> set[Path]:
    managed: set[Path] = set()
    for pattern in ("benchmarks/**/*.qmd", "_stats.qmd", "_generated/*.qmd"):
        managed.update(site.glob(pattern))
    for name in ("reference/cli.qmd", "reference/datasets.qmd"):
        candidate = site / name
        if candidate.exists():
            managed.add(candidate)
    return managed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify generated pages are current without writing.",
    )
    args = parser.parse_args()

    outputs = build_outputs()
    site = ROOT / "site"
    stale = managed_existing(site) - set(outputs)

    if args.check:
        problems: list[str] = []
        for path, content in outputs.items():
            if not path.exists():
                problems.append(f"missing: {path.relative_to(ROOT)}")
            elif path.read_text() != content:
                problems.append(f"out of date: {path.relative_to(ROOT)}")
        problems += [f"stale generated file: {p.relative_to(ROOT)}" for p in stale]
        if problems:
            print("generated docs are out of sync with the registry:")
            for problem in problems:
                print(f"  - {problem}")
            print("run: python3 tools/generate_docs.py")
            return 1
        print(f"generated docs are current ({len(outputs)} pages).")
        return 0

    for path in stale:
        path.unlink()
        print(f"removed stale {path.relative_to(ROOT)}")
    written = 0
    for path, content in sorted(outputs.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
            written += 1
    print(f"wrote {written} of {len(outputs)} generated pages under site/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
