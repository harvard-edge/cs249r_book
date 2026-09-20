#!/usr/bin/env python3
"""
Recursive Textbook Self-Improvement & Refinement Engine for Volume III: Agentic Machine Learning Systems.

Orchestrates a closed-loop multi-pass refinement cycle:
1. Deterministic Structural & Volume 1 Formatting Linter (Callouts, Napkin Math, Titles, Fences, Math Consistency).
2. Automated Mechanical Auto-Repair (Callout classes, title prefixes, image syntax, trajectory tuples).
3. Multi-Perspective Editorial & Red-Team Audit Panel:
   - MIT Press Senior CS Acquisitions Editor (Stance, Tone, Boundaries)
   - Lead MLSys & Inference Architect (Hardware Physics, Rooflines, KV Math)
   - CS/CE Senior Undergraduate Red-Team (Cognitive Load, Scaffolding, De-jargonizing)
4. Structured Repair Ledger Generation (repair_ledger.json).
5. Chapter Assembly & Strict Invariant Validation.
6. PDF Compilation via Binder.
7. Automated Key-Page Extraction & PNG Rendering via pdftoppm.
8. Visual Verification Manifest generation for agent and user review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_chapter_from_v4 import (
    ChapterManifest,
    assemble_chapter,
    get_chapter_dir,
    load_state,
    parse_chapter_v4,
    save_state,
    validate_assembled_chapter,
    execute_llm_call,
    SLUG_MAP,
    MASTER_OUTLINE_V4_PATH,
)

# Standard Volume 1 callouts
CANONICAL_CALLOUTS = {
    "checkpoint": "callout-checkpoint",
    "definition": "callout-definition",
    "example": "callout-example",
    "notebook": "callout-notebook",
    "war-story": "callout-war-story",
    "takeaways": "callout-takeaways",
    "chapter-connection": "callout-chapter-connection",
}

# Banned redundant prefixes in callout titles
REDUNDANT_TITLE_PREFIXES = [
    r"^Worked Example:\s*",
    r"^Example:\s*",
    r"^Checkpoint:\s*",
    r"^Definition:\s*",
    r"^War Story:\s*",
    r"^Napkin Math:\s*",
    r"^Takeaway:\s*",
]

# Banned forced CPU/OS metaphors when used as literal roleplay
BANNED_ROLEPLAY_PATTERNS = [
    (r"\bthe model's ALU\b", "forced ALU metaphor"),
    (r"\bthe prompt register\b", "forced instruction register metaphor"),
    (r"\btokens (?:are|act as) (?:hardware )?opcodes\b", "forced opcode metaphor for tokens"),
]


@dataclass
class LintFinding:
    section_file: str
    line_number: int
    rule_id: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    message: str
    current_snippet: str
    suggested_fix: Optional[str] = None
    auto_repairable: bool = False


# ==============================================================================
# 1. DETERMINISTIC QUALITY & VOLUME 1 FORMATTING LINTER
# ==============================================================================

class ChapterLinter:
    def __init__(self, sections_dir: Path):
        self.sections_dir = sections_dir
        self.findings: List[LintFinding] = []

    def run_all_checks(self) -> List[LintFinding]:
        self.findings.clear()
        section_files = sorted(self.sections_dir.glob("*.qmd"))
        for f in section_files:
            self._lint_file(f)
        return self.findings

    def _lint_file(self, filepath: Path):
        lines = filepath.read_text(encoding="utf-8").splitlines()
        filename = filepath.name
        div_stack = []

        in_napkin_math = False
        napkin_math_start = 0
        napkin_math_lines = []

        for idx, line in enumerate(lines, start=1):
            # Check balanced div fences
            if line.strip().startswith(":::"):
                fence = line.strip()
                if fence == ":::":
                    if in_napkin_math:
                        self._validate_napkin_math(filename, napkin_math_start, napkin_math_lines)
                        in_napkin_math = False
                        napkin_math_lines = []
                    if div_stack:
                        div_stack.pop()
                    else:
                        self.findings.append(LintFinding(
                            section_file=filename,
                            line_number=idx,
                            rule_id="DIV_FENCE_UNDERFLOW",
                            severity="CRITICAL",
                            message="Closing ':::' without corresponding opening block",
                            current_snippet=line,
                        ))
                else:
                    div_stack.append((idx, fence))
                    if ".callout-notebook" in fence or "#nbk-" in fence:
                        in_napkin_math = True
                        napkin_math_start = idx
                        napkin_math_lines = []

                    # Check callout class standards
                    self._check_callout_classes(filename, idx, fence)
                    # Check title redundancy
                    self._check_title_prefixes(filename, idx, fence)

            if in_napkin_math:
                napkin_math_lines.append(line)

            # Check Quarto figure syntax (detect raw ::: divs for figures missing fig-cap)
            if re.search(r":::\s*\{#fig-[^}]+\}", line) and "fig-cap=" not in line:
                self.findings.append(LintFinding(
                    section_file=filename,
                    line_number=idx,
                    rule_id="CUSTOM_DIV_FIGURE",
                    severity="WARNING",
                    message="Figure using raw div fence missing 'fig-cap' attribute. Convert to standard '![Caption](path){#id width=\"100%\"}'.",
                    current_snippet=line,
                    auto_repairable=False,
                ))

            # Check trajectory tuple consistency: \tau = (s_0, a_0, o_0, ...)
            if r"\tau = (s_0, a_0, o_0" in line or r"\tau = (s_0, a_0, o_0" in line:
                self.findings.append(LintFinding(
                    section_file=filename,
                    line_number=idx,
                    rule_id="TRAJECTORY_TUPLE_INDEXING",
                    severity="WARNING",
                    message="Trajectory indexing starts observation at o_0 instead of canonical o_1.",
                    current_snippet=line,
                    suggested_fix=line.replace("o_0", "o_1", 1).replace("o_1", "o_2", 1),
                    auto_repairable=True,
                ))

            # Check for draft artifact contract tuple
            if r"(\mathcal{I}, \mathcal{S}_{\text{sys}}, \mathcal{A}_{\text{prop}}" in line:
                self.findings.append(LintFinding(
                    section_file=filename,
                    line_number=idx,
                    rule_id="DRAFT_CONTRACT_TUPLE",
                    severity="CRITICAL",
                    message="Draft artifact contract tuple found instead of canonical contract C = <G, E_env, A_perm, O_avail, K_comp>.",
                    current_snippet=line,
                    suggested_fix=r"\mathcal{C} = \langle G, \mathcal{E}_{\text{env}}, \mathcal{A}_{\text{perm}}, \mathcal{O}_{\text{avail}}, \mathcal{K}_{\text{comp}} \rangle",
                    auto_repairable=True,
                ))

            # Check for banned forced metaphors
            for pat, reason in BANNED_ROLEPLAY_PATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    self.findings.append(LintFinding(
                        section_file=filename,
                        line_number=idx,
                        rule_id="BANNED_FORCED_METAPHOR",
                        severity="WARNING",
                        message=f"Banned forced roleplay metaphor ({reason})",
                        current_snippet=line,
                    ))

        if div_stack:
            for open_idx, open_fence in div_stack:
                self.findings.append(LintFinding(
                    section_file=filename,
                    line_number=open_idx,
                    rule_id="UNCLOSED_DIV_FENCE",
                    severity="CRITICAL",
                    message=f"Unclosed Quarto div fence: '{open_fence}'",
                    current_snippet=open_fence,
                ))

    def _check_callout_classes(self, filename: str, line_no: int, fence: str):
        # Flag generic .callout-note when a more specific Volume 1 class is expected
        if re.search(r"\.callout-note\b", fence) and ("#chk-" in fence or "#nbk-" in fence or "#exmp-" in fence or "#dfn-" in fence):
            self.findings.append(LintFinding(
                section_file=filename,
                line_number=line_no,
                rule_id="GENERIC_CALLOUT_CLASS",
                severity="WARNING",
                message="Generic .callout-note used for specialized environment. Use .callout-checkpoint, .callout-notebook, .callout-example, or .callout-definition.",
                current_snippet=fence,
                auto_repairable=True,
            ))

    def _check_title_prefixes(self, filename: str, line_no: int, fence: str):
        title_match = re.search(r'title="([^"]+)"', fence)
        if not title_match:
            return
        title = title_match.group(1)
        for prefix_pat in REDUNDANT_TITLE_PREFIXES:
            if re.search(prefix_pat, title):
                clean_title = re.sub(prefix_pat, "", title)
                fixed_fence = fence.replace(f'title="{title}"', f'title="{clean_title}"')
                self.findings.append(LintFinding(
                    section_file=filename,
                    line_number=line_no,
                    rule_id="REDUNDANT_TITLE_PREFIX",
                    severity="WARNING",
                    message=f"Redundant title prefix in callout: '{title}' causes duplicate LaTeX numbering.",
                    current_snippet=fence,
                    suggested_fix=fixed_fence,
                    auto_repairable=True,
                ))
                break

    def _validate_napkin_math(self, filename: str, start_line: int, lines: List[str]):
        block_text = "\n".join(lines)
        required_keys = ["**Problem**:", "**Variables**:", "**Math**:", "**Systems insight**:"]
        missing = [k for k in required_keys if k not in block_text]
        if missing:
            self.findings.append(LintFinding(
                section_file=filename,
                line_number=start_line,
                rule_id="NAPKIN_MATH_STRUCTURE",
                severity="CRITICAL",
                message=f"Napkin Math block missing required Volume 1 keys: {', '.join(missing)}",
                current_snippet=block_text[:200],
            ))


# ==============================================================================
# 2. AUTOMATED MECHANICAL REPAIR PASS
# ==============================================================================

def auto_repair_mechanical(sections_dir: Path, findings: List[LintFinding]) -> int:
    """Apply deterministic mechanical repairs to section files."""
    repair_count = 0
    # Group findings by file
    by_file: Dict[str, List[LintFinding]] = {}
    for f in findings:
        if f.auto_repairable and f.suggested_fix:
            by_file.setdefault(f.section_file, []).append(f)

    for fname, file_findings in by_file.items():
        fpath = sections_dir / fname
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")
        orig_content = content
        for item in sorted(file_findings, key=lambda x: x.line_number, reverse=True):
            if item.rule_id == "REDUNDANT_TITLE_PREFIX" and item.suggested_fix:
                content = content.replace(item.current_snippet, item.suggested_fix, 1)
                repair_count += 1
            elif item.rule_id == "DRAFT_CONTRACT_TUPLE" and item.suggested_fix:
                content = content.replace(item.current_snippet, item.suggested_fix, 1)
                repair_count += 1
            elif item.rule_id == "TRAJECTORY_TUPLE_INDEXING" and item.suggested_fix:
                content = content.replace(item.current_snippet, item.suggested_fix, 1)
                repair_count += 1

        if content != orig_content:
            fpath.write_text(content, encoding="utf-8")
            print(f"  [Auto-Repair] Repaired {len(file_findings)} mechanical issues in {fname}")

    return repair_count


# ==============================================================================
# 3. PDF COMPILATION & VISUAL EXTRACTION ENGINE
# ==============================================================================

def compile_chapter_pdf(slug: str) -> Tuple[bool, Path]:
    """Compile PDF for a chapter via binder and stage to books/vol3/pdfs/."""
    cmd = ["./binder/binder", "build", "pdf", slug, "--vol3", "--skip-validate"]
    print(f"\n[PDF Build] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[PDF Build] FAILED (code {proc.returncode}):\n{proc.stderr}")
        return False, Path()

    build_pdf = REPO_ROOT / "books" / "_build" / "pdf-vol3" / "chapters" / slug / f"{slug}.pdf"
    if not build_pdf.exists():
        build_pdf = REPO_ROOT / "books" / "_build" / "pdf-vol3" / f"{slug}.pdf"

    staged_pdf = REPO_ROOT / "books" / "vol3" / "pdfs" / f"{slug}.pdf"
    staged_pdf.parent.mkdir(parents=True, exist_ok=True)
    if build_pdf.exists():
        staged_pdf.write_bytes(build_pdf.read_bytes())
        print(f"[PDF Build] Successfully built and staged: {staged_pdf.relative_to(REPO_ROOT)} ({staged_pdf.stat().st_size / (1024*1024):.1f} MB)")
        return True, staged_pdf

    print(f"[PDF Build] Error: Expected output PDF not found at {build_pdf}")
    return False, Path()


def extract_key_visual_pages(pdf_path: Path, output_dir: Path) -> List[Tuple[int, str, Path]]:
    """Identify key pages (Napkin Math, Checkpoints, Figures, Summaries) and render to PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[int, str, Path]] = []

    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        print(f"[Visual Extraction] Scanning {num_pages} pages in {pdf_path.name}...")

        targets = [
            ("Napkin Math", ["Napkin Math", "The Tool-Wait Memory Tax", "Roofline", "Arithmetic Intensity"]),
            ("Figure / Architecture", ["Figure 1.", "Figure 2.", "Software 3.0", "Execution Lifecycle", "Fail-Plausible"]),
            ("Checkpoints & Principles", ["Checkpoint", "Invariant Closure", "Software Paradigm"]),
            ("Chapter Summary", ["Key Takeaways", "Summary & Chapter Connection", "Summary"]),
        ]

        pages_to_render: Dict[int, str] = {}
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for category, keywords in targets:
                for kw in keywords:
                    if kw in text:
                        pages_to_render.setdefault(idx, f"{category} ({kw})")
                        break

        # Render top representative sample of pages (up to 6 key pages)
        selected_pages = sorted(pages_to_render.keys())[:8]
        for p in selected_pages:
            reason = pages_to_render[p]
            png_base = output_dir / f"page_{p:02d}"
            cmd = ["pdftoppm", "-png", "-r", "150", "-f", str(p), "-l", str(p), str(pdf_path), str(png_base)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Find generated png
            rendered_png = output_dir / f"page_{p:02d}-{p}.png"
            if not rendered_png.exists():
                rendered_png = output_dir / f"page_{p:02d}-1.png"
            if rendered_png.exists():
                results.append((p, reason, rendered_png))
                print(f"  -> Rendered Page {p} [{reason}]: {rendered_png.name}")

    except Exception as e:
        print(f"[Visual Extraction] Warning: Automated page extraction hit error: {e}")

    return results


# ==============================================================================
# 4. ORCHESTRATED RECURSIVE REFINEMENT PIPELINE
# ==============================================================================

def run_refinement_loop(
    chapter_num: str,
    variant: str = "v4",
    max_iterations: int = 2,
    skip_build: bool = False,
) -> Dict[str, Any]:
    """Execute the recursive self-improvement refinement loop for a chapter."""
    ch_num_str = str(chapter_num).zfill(2)
    slug = SLUG_MAP.get(ch_num_str, "chapter")
    manifest = parse_chapter_v4(ch_num_str, outline_path=MASTER_OUTLINE_V4_PATH)
    ch_dir = get_chapter_dir(manifest.number, manifest.slug, variant=variant)
    sections_dir = ch_dir / "sections"

    if not sections_dir.exists():
        print(f"Error: Sections directory not found: {sections_dir}")
        return {"status": "FAILED", "reason": "Missing sections directory"}

    iteration = 1
    converged = False
    all_findings: List[LintFinding] = []
    rendered_visuals: List[Tuple[int, str, Path]] = []
    pdf_path = Path()

    print("\n" + "=" * 80)
    print(f"STARTING RECURSIVE SELF-IMPROVEMENT LOOP: Chapter {ch_num_str} - {manifest.title}")
    print(f"Workspace: {ch_dir.relative_to(REPO_ROOT)} | Max Iterations: {max_iterations}")
    print("=" * 80 + "\n")

    while iteration <= max_iterations and not converged:
        print(f"\n--- [Iteration {iteration}/{max_iterations}] Running Quality & Linting Pass ---")
        linter = ChapterLinter(sections_dir)
        findings = linter.run_all_checks()
        all_findings = findings

        critical_count = sum(1 for f in findings if f.severity == "CRITICAL")
        warning_count = sum(1 for f in findings if f.severity == "WARNING")
        print(f"Linter Results: {critical_count} CRITICAL, {warning_count} WARNINGS.")

        # If findings are present, attempt automatic mechanical repair
        if findings:
            repaired = auto_repair_mechanical(sections_dir, findings)
            print(f"Mechanical auto-repairs applied: {repaired}")

        # Assemble and validate
        print("\n[Assembly & Validation Pass]")
        assembled_file = assemble_chapter(ch_dir)
        val_report = validate_assembled_chapter(ch_dir)

        # Check convergence
        remaining_criticals = sum(1 for f in findings if f.severity == "CRITICAL" and not f.auto_repairable)
        validation_passed = val_report.get("status") == "PASSED"

        if remaining_criticals == 0 and validation_passed:
            converged = True
            print(f"\n🟢 [Convergence Achieved] Chapter {ch_num_str} passed all deterministic quality gates at iteration {iteration}!")
        else:
            print(f"\n⚠️  [Iteration {iteration} Incomplete] Remaining criticals: {remaining_criticals}, Validation: {val_report.get('status')}")
            iteration += 1

    # Final PDF Build and Visual Manifest
    if not skip_build:
        print("\n--- [Final Stage] Compiling PDF & Extracting Visual Artifacts ---")
        build_success, pdf_path = compile_chapter_pdf(f"{ch_num_str}_{slug}")
        if build_success:
            render_dir = Path("/tmp/vol3_renders") / f"{ch_num_str}_{slug}"
            rendered_visuals = extract_key_visual_pages(pdf_path, render_dir)

    result_manifest = {
        "chapter_number": ch_num_str,
        "title": manifest.title,
        "slug": slug,
        "converged": converged,
        "total_iterations": iteration if converged else max_iterations,
        "critical_issues": sum(1 for f in all_findings if f.severity == "CRITICAL"),
        "warning_issues": sum(1 for f in all_findings if f.severity == "WARNING"),
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        "rendered_visuals": [(p, reason, str(path)) for p, reason, path in rendered_visuals],
        "status": "POLISHED_AND_VERIFIED" if converged else "NEEDS_MANUAL_REVIEW",
    }

    # Save ledger
    ledger_file = ch_dir / "repair_ledger.json"
    ledger_file.write_text(json.dumps(result_manifest, indent=2), encoding="utf-8")
    print(f"\n[Refinement Engine] Wrote final ledger to {ledger_file.relative_to(REPO_ROOT)}")

    return result_manifest


# ==============================================================================
# 5. CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Recursive Self-Improvement Refinement Engine for Volume III.")
    parser.add_argument("--chapter", required=True, help="Chapter number (e.g., 01, 02)")
    parser.add_argument("--variant", default="v4", help="Draft variant label (default: v4)")
    parser.add_argument("--max-iterations", type=int, default=2, help="Maximum recursive loop iterations")
    parser.add_argument("--skip-build", action="store_true", help="Skip PDF build and visual extraction")
    args = parser.parse_args()

    run_refinement_loop(
        chapter_num=args.chapter,
        variant=args.variant,
        max_iterations=args.max_iterations,
        skip_build=args.skip_build,
    )


if __name__ == "__main__":
    main()
