#!/usr/bin/env python3
"""Build MLSysBook chapter packets for math-accuracy audit rounds.

The script uses the existing `book/binder` PDF build path. For each selected
unit it builds a standalone PDF, copies the generated PDF and keep-tex output
into a stable packet directory, and writes small-batch prompts for Codex agents.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared by repo
    raise SystemExit("PyYAML is required. Install project requirements first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[4]
BOOK_DIR = REPO_ROOT / "book" / "quarto"
BINDER = REPO_ROOT / "book" / "binder"
RUNS_DIR = REPO_ROOT / "review" / "math-audit" / "runs"

PDF_NAMES = {
    "vol1": "Machine-Learning-Systems-Vol1.pdf",
    "vol2": "Machine-Learning-Systems-Vol2.pdf",
}
TEX_NAMES = {
    "vol1": "Machine-Learning-Systems-Vol1.tex",
    "vol2": "Machine-Learning-Systems-Vol2.tex",
}
SKIP_STEMS = {"index", "references"}


@dataclass(frozen=True)
class BuildUnit:
    volume: str
    config_position: int
    stem: str
    rel_path: str
    source_path: Path
    kind: str
    source_section: str

    @property
    def packet_name(self) -> str:
        return f"{self.config_position:02d}-{self.stem}"

    @property
    def chapter_id(self) -> str:
        return f"{self.volume}/{self.stem}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def rel_entry_path(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        file_value = entry.get("file")
        return str(file_value) if file_value else None
    return None


def classify_unit(rel_path: str, source_section: str) -> str:
    stem = Path(rel_path).stem
    parts = Path(rel_path).parts
    if source_section == "appendices":
        return "glossary" if stem == "glossary" else "appendix"
    if "frontmatter" in parts:
        return "frontmatter"
    if "parts" in parts:
        return "part"
    if "backmatter" in parts:
        if stem.startswith("appendix_"):
            return "appendix"
        if stem == "glossary":
            return "glossary"
        return "backmatter"
    return "chapter"


def discover_units(volumes: Iterable[str]) -> list[BuildUnit]:
    units: list[BuildUnit] = []
    for volume in volumes:
        config_path = BOOK_DIR / "config" / f"_quarto-pdf-{volume}.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing PDF config: {config_path}")
        config = read_yaml(config_path)
        book_config = config.get("book", {})
        position = 0
        for section_name in ("chapters", "appendices"):
            for entry in book_config.get(section_name, []) or []:
                rel_path = rel_entry_path(entry)
                if not rel_path:
                    continue
                stem = Path(rel_path).stem
                if stem in SKIP_STEMS:
                    continue
                source_path = BOOK_DIR / rel_path
                if not source_path.exists():
                    raise FileNotFoundError(f"Configured source does not exist: {source_path}")
                position += 1
                units.append(
                    BuildUnit(
                        volume=volume,
                        config_position=position,
                        stem=stem,
                        rel_path=rel_path,
                        source_path=source_path,
                        kind=classify_unit(rel_path, section_name),
                        source_section=section_name,
                    )
                )
    return units


def include_unit(unit: BuildUnit, args: argparse.Namespace) -> bool:
    if args.include_all_units:
        return True
    if unit.kind == "frontmatter" and not args.include_frontmatter:
        return False
    if unit.kind == "part" and not args.include_parts:
        return False
    if unit.kind == "glossary" and not args.include_glossary:
        return False
    if unit.kind == "backmatter" and not args.include_backmatter:
        return False
    return True


def resolve_only_specs(specs: list[str], units: list[BuildUnit]) -> set[tuple[str, str]]:
    resolved: set[tuple[str, str]] = set()
    for raw_spec in specs:
        spec = raw_spec.strip()
        if not spec:
            continue
        if ":" in spec:
            volume, stem = spec.split(":", 1)
        elif "/" in spec:
            volume, stem = spec.split("/", 1)
        else:
            matches = [unit for unit in units if unit.stem == spec]
            if not matches:
                raise ValueError(f"No unit named {spec!r}")
            if len(matches) > 1:
                choices = ", ".join(unit.chapter_id for unit in matches)
                raise ValueError(f"Ambiguous unit {spec!r}; use one of: {choices}")
            volume, stem = matches[0].volume, matches[0].stem
        resolved.add((volume.strip(), stem.strip().removesuffix(".qmd")))
    return resolved


def selected_units(args: argparse.Namespace) -> list[BuildUnit]:
    volumes = parse_csv(args.volumes, ["vol1", "vol2"])
    units = [unit for unit in discover_units(volumes) if include_unit(unit, args)]
    if args.only:
        specs = resolve_only_specs(parse_csv(args.only, []), units)
        units = [unit for unit in units if (unit.volume, unit.stem) in specs]
    if args.limit is not None:
        units = units[: args.limit]
    return units


def output_dir_for(volume: str) -> Path:
    return BOOK_DIR / "_build" / f"pdf-{volume}"


def clean_previous_build(volume: str) -> None:
    output_dir = output_dir_for(volume)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tex_file = BOOK_DIR / TEX_NAMES[volume]
    if tex_file.exists():
        tex_file.unlink()


def run_command(cmd: list[str], cwd: Path, log_path: Path, verbose: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            if verbose:
                print(line, end="")
    return process.wait()


def build_unit(unit: BuildUnit, run_dir: Path, force: bool, verbose: bool) -> dict[str, Any]:
    packet_dir = run_dir / "packets" / unit.volume / unit.packet_name
    if packet_dir.exists() and force:
        shutil.rmtree(packet_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)

    log_path = packet_dir / "build.log"
    clean_previous_build(unit.volume)
    cmd = [str(BINDER), "build", "pdf", unit.stem, f"--{unit.volume}"]
    if verbose:
        cmd.append("-v")

    print(f"Building {unit.chapter_id} -> {packet_dir}", flush=True)
    return_code = run_command(cmd, REPO_ROOT, log_path, verbose=verbose)
    if return_code != 0:
        return {
            "chapter": unit.chapter_id,
            "kind": unit.kind,
            "packet_dir": str(packet_dir),
            "build_log": str(log_path),
            "status": "build_failed",
            "return_code": return_code,
        }

    return package_unit(unit, run_dir, packet_dir, log_path)


def build_volume(volume: str, run_dir: Path, force: bool, verbose: bool) -> dict[str, Any]:
    dest_dir = run_dir / "volumes" / volume
    if dest_dir.exists() and force:
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    log_path = dest_dir / "build.log"
    clean_previous_build(volume)
    cmd = [str(BINDER), "build", "pdf", f"--{volume}"]
    if verbose:
        cmd.append("-v")

    print(f"Building full {volume} -> {dest_dir}", flush=True)
    return_code = run_command(cmd, REPO_ROOT, log_path, verbose=verbose)
    if return_code != 0:
        return {
            "volume": volume,
            "dest_dir": str(dest_dir),
            "build_log": str(log_path),
            "status": "build_failed",
            "return_code": return_code,
        }

    pdf_src = output_dir_for(volume) / PDF_NAMES[volume]
    tex_src = BOOK_DIR / TEX_NAMES[volume]
    if not pdf_src.exists() or not tex_src.exists():
        missing = [str(path) for path in (pdf_src, tex_src) if not path.exists()]
        return {
            "volume": volume,
            "dest_dir": str(dest_dir),
            "build_log": str(log_path),
            "status": "missing_artifact",
            "missing": missing,
        }

    pdf_dest = dest_dir / f"{volume}-full.pdf"
    tex_dest = dest_dir / f"{volume}-full.tex"
    shutil.copy2(pdf_src, pdf_dest)
    shutil.copy2(tex_src, tex_dest)
    math_context = dest_dir / "math_context.md"
    extract_math_context(tex_dest, math_context)

    record = {
        "volume": volume,
        "dest_dir": str(dest_dir),
        "status": "built",
        "pdf": str(pdf_dest),
        "tex": str(tex_dest),
        "math_context": str(math_context),
        "build_log": str(log_path),
        "built_at": now_iso(),
    }
    write_yaml(dest_dir / "manifest.yml", record)
    return record


def package_unit(unit: BuildUnit, run_dir: Path, packet_dir: Path, log_path: Path) -> dict[str, Any]:
    pdf_src = output_dir_for(unit.volume) / PDF_NAMES[unit.volume]
    tex_src = BOOK_DIR / TEX_NAMES[unit.volume]
    missing = [str(path) for path in (pdf_src, tex_src, unit.source_path) if not path.exists()]
    if missing:
        return {
            "chapter": unit.chapter_id,
            "kind": unit.kind,
            "packet_dir": str(packet_dir),
            "build_log": str(log_path),
            "status": "missing_artifact",
            "missing": missing,
        }

    artifact_stem = f"{unit.volume}-{unit.stem}"
    pdf_dest = packet_dir / f"{artifact_stem}.pdf"
    tex_dest = packet_dir / f"{artifact_stem}.tex"
    qmd_dest = packet_dir / f"{artifact_stem}.qmd"
    shutil.copy2(pdf_src, pdf_dest)
    shutil.copy2(tex_src, tex_dest)
    shutil.copy2(unit.source_path, qmd_dest)

    for optional_name in ("FIGURE_LIST.txt", "index_figures.txt"):
        optional_src = output_dir_for(unit.volume) / optional_name
        if optional_src.exists():
            shutil.copy2(optional_src, packet_dir / optional_name)

    math_context = packet_dir / "math_context.md"
    extract_math_context(tex_dest, math_context)
    audit_output = run_dir / "audits" / f"{artifact_stem}.yml"
    packet_prompt = packet_dir / "agent_prompt.md"
    record = {
        "chapter": unit.chapter_id,
        "volume": unit.volume,
        "stem": unit.stem,
        "kind": unit.kind,
        "source_section": unit.source_section,
        "source_qmd": str(unit.source_path),
        "source_qmd_copy": str(qmd_dest),
        "rel_path": unit.rel_path,
        "packet_dir": str(packet_dir),
        "status": "built",
        "pdf": str(pdf_dest),
        "tex": str(tex_dest),
        "math_context": str(math_context),
        "audit_output": str(audit_output),
        "agent_prompt": str(packet_prompt),
        "build_log": str(log_path),
        "built_at": now_iso(),
    }
    write_yaml(packet_dir / "manifest.yml", record)
    packet_prompt.write_text(agent_prompt_for_records([record], 1), encoding="utf-8")
    return record


def extract_math_context(tex_path: Path, output_path: Path) -> None:
    lines = tex_path.read_text(encoding="utf-8", errors="replace").splitlines()
    env_start = re.compile(
        r"\\begin\{(equation\*?|align\*?|aligned|gather\*?|multline\*?|split|flalign\*?|eqnarray\*?)\}"
    )
    env_end = re.compile(
        r"\\end\{(equation\*?|align\*?|aligned|gather\*?|multline\*?|split|flalign\*?|eqnarray\*?)\}"
    )
    display_start_re = re.compile(r"(?<!\\)\\\[")
    display_end_re = re.compile(r"(?<!\\)\\\]")

    blocks: list[tuple[str, int, int, list[str]]] = []
    inline_lines: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        start_match = env_start.search(line)
        display_start = display_start_re.search(line) is not None
        if start_match or display_start:
            kind = start_match.group(1) if start_match else "display"
            start = i
            end = i
            while end < len(lines):
                if (start_match and env_end.search(lines[end])) or (
                    display_start and display_end_re.search(lines[end])
                ):
                    break
                end += 1
            end = min(end, len(lines) - 1)
            context_start = max(0, start - 2)
            context_end = min(len(lines), end + 3)
            blocks.append((kind, start + 1, end + 1, lines[context_start:context_end]))
            i = end + 1
            continue
        if "\\(" in line or "$" in line:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                inline_lines.append((i + 1, stripped[:500]))
        i += 1

    out: list[str] = [
        "# Math Candidate Context",
        "",
        "This helper is not exhaustive. Agents must still read the full TeX and QMD.",
        "",
        f"- source_tex: {tex_path}",
        f"- display_blocks: {len(blocks)}",
        f"- inline_candidate_lines: {len(inline_lines)}",
        "",
    ]
    for index, (kind, start_line, end_line, context) in enumerate(blocks, start=1):
        out.extend(
            [
                f"## Display Block {index}",
                "",
                f"- kind: {kind}",
                f"- tex_lines: {start_line}-{end_line}",
                "",
                "```tex",
                *context,
                "```",
                "",
            ]
        )
    if inline_lines:
        out.extend(["## Inline Math Candidate Lines", ""])
        for line_no, text in inline_lines:
            out.append(f"- line {line_no}: `{text}`")
        out.append("")
    output_path.write_text("\n".join(out), encoding="utf-8")


def agent_prompt_for_records(records: list[dict[str, Any]], batch_number: int) -> str:
    lines = [
        f"# MLSysBook Math Audit Batch {batch_number:03d}",
        "",
        "Use extra-high reasoning. The goal is mathematical accuracy of calculations.",
        "Read each TeX file, the copied source QMD, and the PDF when rendered context matters.",
        "The math_context.md file is only a locator; do not rely on it as complete coverage.",
        "",
        "For every numeric, algebraic, dimensional, probability, scaling, complexity,",
        "latency, throughput, memory, energy, carbon, or cost calculation:",
        "- recompute it independently from the stated assumptions;",
        "- verify unit conversions and orders of magnitude;",
        "- verify that surrounding prose matches the formula and result;",
        "- flag incorrect arithmetic, inconsistent assumptions, or prose/result mismatch;",
        "- do not rewrite source files in this audit pass.",
        "",
        "Write one YAML audit file per packet at the exact audit_output path below.",
        "Use status: clear with issues: [] when no math issues are found.",
        "At the top level, set chapter/source_tex/source_qmd/source_pdf to the",
        "exact values for that packet.",
        "",
        "YAML schema:",
        "```yaml",
        "chapter: vol1/introduction",
        "status: clear | issues_found",
        "reviewed_at: '2026-05-20T00:00:00Z'",
        "auditor: codex-agent",
        "source_tex: /abs/path/to/file.tex",
        "source_qmd: /abs/path/to/file.qmd",
        "source_pdf: /abs/path/to/file.pdf",
        "issues:",
        "  - id: vol1-introduction-MATH-001",
        "    severity: high | medium | low",
        "    tex_line: 123",
        "    qmd_line: 456",
        "    rendered_context: short description of where it appears",
        "    expression: formula or numeric claim being checked",
        "    prose_claim: prose tied to the calculation",
        "    observed_result: value or relationship in the book",
        "    expected_result: recomputed correct value or relationship",
        "    reasoning: concise derivation showing the correction",
        "    proposed_correction: exact correction to consider",
        "    confidence: high | medium | low",
        "```",
        "",
        "Packets:",
    ]
    for idx, record in enumerate(records, start=1):
        lines.extend(
            [
                f"{idx}. {record['chapter']}",
                f"   packet_dir: {record['packet_dir']}",
                f"   tex: {record['tex']}",
                f"   source_qmd: {record['source_qmd_copy']}",
                f"   pdf: {record['pdf']}",
                f"   math_context: {record['math_context']}",
                f"   audit_output: {record['audit_output']}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_agent_batches(run_dir: Path, packet_records: list[dict[str, Any]], batch_size: int) -> list[str]:
    batch_dir = run_dir / "agent_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[str] = []
    built_packets = [record for record in packet_records if record.get("status") == "built"]
    for start in range(0, len(built_packets), batch_size):
        batch = built_packets[start : start + batch_size]
        batch_number = start // batch_size + 1
        path = batch_dir / f"batch-{batch_number:03d}.md"
        path.write_text(agent_prompt_for_records(batch, batch_number), encoding="utf-8")
        batch_paths.append(str(path))
    return batch_paths


def cmd_list(args: argparse.Namespace) -> int:
    for unit in selected_units(args):
        print(f"{unit.chapter_id:36} {unit.kind:12} {unit.rel_path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    run_id = slugify(args.run_id or datetime.now().strftime("round-%Y%m%d-%H%M%S"))
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audits").mkdir(parents=True, exist_ok=True)

    units = selected_units(args)
    packet_records: list[dict[str, Any]] = []
    volume_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if args.mode in ("volumes", "both"):
        for volume in parse_csv(args.volumes, ["vol1", "vol2"]):
            record = build_volume(volume, run_dir, args.force, args.verbose)
            volume_records.append(record)
            if record.get("status") != "built":
                failures.append(record)
                if not args.continue_on_error:
                    break

    if args.mode in ("chapters", "both") and not (failures and not args.continue_on_error):
        for unit in units:
            record = build_unit(unit, run_dir, args.force, args.verbose)
            packet_records.append(record)
            if record.get("status") != "built":
                failures.append(record)
                if not args.continue_on_error:
                    break

    batch_paths = write_agent_batches(run_dir, packet_records, args.batch_size)
    run_manifest = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at": now_iso(),
        "mode": args.mode,
        "volumes": parse_csv(args.volumes, ["vol1", "vol2"]),
        "unit_count": len(units),
        "built_packet_count": len([r for r in packet_records if r.get("status") == "built"]),
        "failure_count": len(failures),
        "batch_size": args.batch_size,
        "agent_batches": batch_paths,
        "volumes_built": volume_records,
        "packets": packet_records,
        "failures": failures,
    }
    write_yaml(run_dir / "run_manifest.yml", run_manifest)
    print(f"Run directory: {run_dir}")
    print(f"Built packets: {run_manifest['built_packet_count']} / {len(packet_records)}")
    print(f"Agent batches: {len(batch_paths)}")
    if failures:
        print(f"Failures: {len(failures)}")
        return 1
    return 0


def cmd_prompts(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = read_yaml(run_dir / "run_manifest.yml")
    packet_records = manifest.get("packets", [])
    batch_paths = write_agent_batches(run_dir, packet_records, args.batch_size)
    manifest["batch_size"] = args.batch_size
    manifest["agent_batches"] = batch_paths
    write_yaml(run_dir / "run_manifest.yml", manifest)
    print(f"Wrote {len(batch_paths)} agent batch prompts under {run_dir / 'agent_batches'}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    audit_dir = run_dir / "audits"
    manifest_path = run_dir / "run_manifest.yml"
    audit_to_chapter: dict[str, str] = {}
    if manifest_path.exists():
        manifest = read_yaml(manifest_path)
        for packet in manifest.get("packets", []) or []:
            audit_output = packet.get("audit_output")
            chapter = packet.get("chapter")
            if audit_output and chapter:
                audit_to_chapter[str(Path(audit_output).resolve())] = chapter

    audit_files = sorted(audit_dir.glob("*.yml"))
    audits: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for audit_file in audit_files:
        data = read_yaml(audit_file)
        inferred_chapter = audit_to_chapter.get(str(audit_file.resolve()))
        if inferred_chapter and not data.get("chapter"):
            data["chapter"] = inferred_chapter
        data["_audit_file"] = str(audit_file)
        audits.append(data)
        for issue in data.get("issues", []) or []:
            issue_record = dict(issue)
            issue_record["chapter"] = data.get("chapter")
            issue_record["_audit_file"] = str(audit_file)
            issues.append(issue_record)

    summary = {
        "run_dir": str(run_dir),
        "generated_at": now_iso(),
        "audit_file_count": len(audits),
        "issue_count": len(issues),
        "clear_count": len([a for a in audits if a.get("status") == "clear"]),
        "issues_found_count": len([a for a in audits if a.get("status") == "issues_found"]),
        "issues": issues,
    }
    write_yaml(run_dir / "summary.yml", summary)
    print(f"Audit files: {summary['audit_file_count']}")
    print(f"Issues: {summary['issue_count']}")
    print(f"Summary: {run_dir / 'summary.yml'}")
    return 0


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--volumes", default="vol1,vol2", help="Comma-separated volumes.")
    parser.add_argument("--only", help="Comma-separated units such as vol1:introduction.")
    parser.add_argument("--limit", type=int, help="Limit selected units after filtering.")
    parser.add_argument("--include-frontmatter", action="store_true")
    parser.add_argument("--include-parts", action="store_true")
    parser.add_argument("--include-glossary", action="store_true")
    parser.add_argument("--include-backmatter", action="store_true")
    parser.add_argument("--include-all-units", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List selected build units.")
    add_selection_args(list_parser)
    list_parser.set_defaults(func=cmd_list)

    build_parser_ = subparsers.add_parser("build", help="Build packets and prompts.")
    add_selection_args(build_parser_)
    build_parser_.add_argument("--mode", choices=["chapters", "volumes", "both"], default="chapters")
    build_parser_.add_argument("--run-id", help="Run directory name under review/math-audit/runs.")
    build_parser_.add_argument("--batch-size", type=int, default=3)
    build_parser_.add_argument("--force", action="store_true", help="Replace existing packet dirs.")
    build_parser_.add_argument("--continue-on-error", action="store_true")
    build_parser_.add_argument("--verbose", action="store_true", help="Stream binder output.")
    build_parser_.set_defaults(func=cmd_build)

    prompts_parser = subparsers.add_parser("prompts", help="Regenerate agent batch prompts.")
    prompts_parser.add_argument("run_dir")
    prompts_parser.add_argument("--batch-size", type=int, default=3)
    prompts_parser.set_defaults(func=cmd_prompts)

    summary_parser = subparsers.add_parser("summarize", help="Summarize agent YAML audits.")
    summary_parser.add_argument("run_dir")
    summary_parser.set_defaults(func=cmd_summarize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
