#!/usr/bin/env python3
"""
scripts/streamline_vol3_sections.py

Parallel Section-by-Section Editorial Refinement Engine for Volume III.
Decomposes chapters into H2 sections, dispatches parallel agents to rewrite
each section into Volume 2's authoritative textbook style, verifies structural
invariants (cross-references, code fences, callouts), and reassembles the files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
VOL3_ROOT = REPO_ROOT / "books" / "vol3"

# ==============================================================================
# Editorial System Prompt & Directives
# ==============================================================================

EDITORIAL_SYSTEM_PROMPT = """You are a senior computer systems author and editorial architect for the MIT Press textbook 'Machine Learning Systems'.

Your task is to refine and streamline a single section of a textbook chapter on Agentic Machine Learning Systems.

### Editorial Tone & Voice Directives (Volume 2 Standard):
1. **Pedagogical Authority & Warmth**: Write like an experienced systems professor explaining real computer systems to graduate students and engineers. Be clear, engaging, grounded, and technically rigorous.
2. **Abolish Robotic 'LLM Dialect' Tells**:
   - Eliminate formulaic scaffolds:
     * Never write "Taking [X] seriously [changes/turns] the engineering question from [A] to [B]..."
     * Never write "The [noun] [verbs]; it does not [verb]" (e.g. "The picture orients; it does not explain").
     * Never write "A reader who arrives with [X] holds two pieces..."
     * Never write meta-announcements: "In this section, we examine...", "Having established X, we now turn to Y...".
   - Eliminate syntactic hyper-compression and stacked comma splices. Vary sentence lengths: combine punchy, clear topic sentences with well-structured compound explanations.
   - Replace passive, bloodless abstractions with active voice and concrete engineering causality.
3. **MIT Press Editorial & Typographic Standards**:
   - Use American English spelling (-ize, labeled, modeled).
   - Physical units must always be spaced: write "100 ms", "4 GB", "312 TFLOPS" (never "100ms" or "4GB").
   - Use "vs." with a period (never "versus" or bare "vs").
   - In running body prose, prefer English to Latin abbreviations: write "for example" instead of "e.g.", "that is" instead of "i.e.", "and so on" instead of "etc.".
   - **Standalone Book Rule**: Never reference "Volume I", "Volume II", "Volume III", or "this volume". Refer to "this book" or describe the concepts directly by name (e.g., "single-node training systems", "fleet-scale distribution", "agentic execution").

### Strict Invariant Preservation:
1. **Header & Structure**: Begin your output with the EXACT same section heading line (e.g. `## Section Title {#sec-vol3-...}`).
2. **Cross-References**: You MUST PRESERVE all Quarto cross-references (`@sec-...`, `@fig-...`, `@tbl-...`, `@eq-...`, `@lst-...`) and citations (`@author2024`). Do not rename, drop, or alter them.
3. **Code & Math**:
   - Preserve all code blocks (````python ... ````, ````yaml ... ````, etc.) and executable cells completely intact.
   - Preserve all LaTeX math environments (`$...$`, `$$...$$`) and mathematical notation intact.
4. **Callouts & Insets**: Preserve all callout definitions (`::: {#... .callout-...}` and their closing `:::`), figure divs, margin figures, and checklists.
5. **Footnotes & Indexing**: Preserve all footnote markers (`[^fn-...]`), footnote definitions, and index tags (`\\index{...}`).

### Output Format:
Return ONLY the updated section markdown. Do NOT wrap your output in triple backticks or conversational commentary. Start immediately with the `## ` heading.
"""


@dataclass
class SectionChunk:
    index: int
    title: str
    heading_line: str
    content: str
    is_frontmatter: bool = False
    invariants: Set[str] = field(default_factory=set)


# ==============================================================================
# Section Splitting & Invariant Extraction
# ==============================================================================

def extract_invariants(text: str) -> Set[str]:
    """Extract labels, cross-refs, and citations that must not disappear."""
    invariants = set()
    # Cross references and citations: @sec-..., @fig-..., @jimenez2024
    refs = re.findall(r"@[a-zA-Z0-9_\-]+", text)
    invariants.update(refs)
    # Footnote markers: [^fn-...]
    fn_refs = re.findall(r"\[\^fn-[a-zA-Z0-9_\-]+\]", text)
    invariants.update(fn_refs)
    # Callout IDs: {#dfn-... or {#fig-...
    callout_ids = re.findall(r"\{#([a-zA-Z0-9_\-]+)", text)
    invariants.update(callout_ids)
    return invariants


def split_qmd_into_sections(text: str) -> List[SectionChunk]:
    """Splits a QMD file into frontmatter/preamble and H2 sections."""
    lines = text.splitlines(keepends=True)
    sections: List[SectionChunk] = []

    # Find H2 headings: lines starting with '## '
    h2_indices = []
    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.strip().startswith("## Purpose"):
            h2_indices.append(i)

    if not h2_indices:
        # Single block
        return [SectionChunk(
            index=0,
            title="Full Document",
            heading_line="",
            content=text,
            is_frontmatter=False,
            invariants=extract_invariants(text),
        )]

    # Frontmatter / Purpose block (everything before the first content H2)
    first_h2 = h2_indices[0]
    front_text = "".join(lines[:first_h2])
    sections.append(SectionChunk(
        index=0,
        title="Frontmatter & Purpose",
        heading_line="",
        content=front_text,
        is_frontmatter=True,
        invariants=extract_invariants(front_text),
    ))

    # Content H2 sections
    for idx, start_i in enumerate(h2_indices):
        end_i = h2_indices[idx + 1] if idx + 1 < len(h2_indices) else len(lines)
        sec_text = "".join(lines[start_i:end_i])
        heading_line = lines[start_i].strip()
        title_m = re.search(r"##\s+([^\{]+)", heading_line)
        title = title_m.group(1).strip() if title_m else heading_line

        sections.append(SectionChunk(
            index=idx + 1,
            title=title,
            heading_line=heading_line,
            content=sec_text,
            is_frontmatter=False,
            invariants=extract_invariants(sec_text),
        ))

    return sections


# ==============================================================================
# Agent Execution Backend
# ==============================================================================

def execute_agent_call(
    prompt: str,
    backend: str = "agy",
    model: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Invokes headless agent CLI (agy or claude) to execute the section rewrite."""
    if backend == "mock":
        # Mock mode for testing without LLM calls
        return prompt.split("### Original Section Content:")[1].strip()

    if backend == "agy":
        cmd = [
            "agy",
            "-p", prompt,
            "--agent", "vol3-input-only-writer-v2",
            "--disable-slash-commands",
            "--dangerously-skip-permissions",
            "--print-timeout", f"{timeout}s",
        ]
        if model:
            cmd.extend(["--model", model])
    elif backend == "claude":
        cmd = [
            "claude",
            "-p", prompt,
            "--dangerously-skip-permissions",
        ]
        if model:
            cmd.extend(["--model", model])
    else:
        raise ValueError(f"Unknown backend: {backend}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout + 30,
        stdin=subprocess.DEVNULL,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Agent {backend} failed with code {proc.returncode}: {proc.stderr[:400]}")

    output = proc.stdout.strip()

    # Filter out JSON lines emitted by tool logs if any
    lines = [l for l in output.splitlines() if not l.startswith('{"toolAction":')]
    output = "\n".join(lines).strip()

    # Strip any accidental wrapping markdown backticks
    if output.startswith("```markdown"):
        output = output[len("```markdown"):].strip()
    elif output.startswith("```md"):
        output = output[len("```md"):].strip()
    elif output.startswith("```"):
        output = output[3:].strip()
    if output.endswith("```"):
        output = output[:-3].strip()

    # Find start of section (first line matching ^##\s+) to discard any greeting
    match = re.search(r"^##\s+", output, flags=re.MULTILINE)
    if match:
        output = output[match.start():].strip()

    return output


# ==============================================================================
# Validation & Invariant Checking
# ==============================================================================

def sanitize_standalone_prose(text: str) -> str:
    """Removes series-dependent volume references to satisfy standalone-prose rules."""
    text = re.sub(r"\bVolume\s+I\b", "single-device systems", text)
    text = re.sub(r"\bVolume\s+II\b", "distributed fleet systems", text)
    text = re.sub(r"\bVolume\s+III\b", "this book", text)
    text = re.sub(r"\bthis volume\b", "this book", text, flags=re.IGNORECASE)
    return text


def validate_section_rewrite(original: SectionChunk, rewritten: str) -> Tuple[bool, str, List[str]]:
    """Checks that the rewritten section preserved all essential invariants and sanitizes formatting."""
    issues = []

    # 1. Heading preserved
    if not rewritten.strip().startswith("## "):
        issues.append("Rewritten section does not start with an H2 ('## ') heading.")

    # 2. Block balance check (::: fences)
    open_count = len(re.findall(r"^:::\s*\{", rewritten, flags=re.MULTILINE))
    close_count = len(re.findall(r"^:::\s*$", rewritten, flags=re.MULTILINE))
    if open_count != close_count:
        issues.append(f"Mismatched ::: fences (opened: {open_count}, closed: {close_count}).")

    # 3. Code fence balance check (```)
    code_fences = len(re.findall(r"^```", rewritten, flags=re.MULTILINE))
    if code_fences % 2 != 0:
        issues.append(f"Odd number of code fences ({code_fences}).")

    # 4. Invariant presence check
    rewritten_invariants = extract_invariants(rewritten)
    missing = original.invariants - rewritten_invariants
    critical_missing = [m for m in missing if m.startswith("@sec-") or m.startswith("@fig-") or m.startswith("@eq-") or m.startswith("@tbl-") or m.startswith("@lst-")]
    if critical_missing:
        issues.append(f"Dropped critical cross-references: {critical_missing[:5]}")

    # 5. Check Python code labels
    orig_labels = set(re.findall(r"#\|\s*label:\s*([^\s\n]+)", original.content))
    new_labels = set(re.findall(r"#\|\s*label:\s*([^\s\n]+)", rewritten))
    missing_labels = orig_labels - new_labels
    if missing_labels:
        issues.append(f"Dropped Python cell labels: {missing_labels}")

    # Sanitize volume references
    sanitized = sanitize_standalone_prose(rewritten)

    # Clean trailing whitespace per line
    sanitized_lines = [line.rstrip() for line in sanitized.splitlines()]
    sanitized = "\n".join(sanitized_lines) + "\n"

    return (len(issues) == 0, sanitized, issues)


# ==============================================================================
# Section Refinement Pipeline
# ==============================================================================

def refine_single_section(
    section: SectionChunk,
    chapter_context: str,
    backend: str = "agy",
    model: Optional[str] = None,
    timeout: int = 300,
) -> Tuple[SectionChunk, str, bool, List[str]]:
    """Runs a single agent on a single section chunk."""
    # Frontmatter or special sections can optionally be preserved or refined
    if section.is_frontmatter:
        # Leave frontmatter untouched to avoid corrupting YAML or minitoc
        return section, section.content, True, []

    # Skip reference or exercises sections if they don't need prose rewriting
    if "## Exercises" in section.heading_line or "## References" in section.heading_line:
        return section, section.content, True, []

    prompt = f"""{EDITORIAL_SYSTEM_PROMPT}

### Chapter Context:
{chapter_context[:1000]}

### Section to Refine:
{section.heading_line}

### Original Section Content:
{section.content}
"""

    try:
        rewritten = execute_agent_call(prompt, backend=backend, model=model, timeout=timeout)
        valid, sanitized, issues = validate_section_rewrite(section, rewritten)
        if valid:
            # Append trailing newline for clean formatting
            if not sanitized.endswith("\n"):
                sanitized += "\n"
            return section, sanitized, True, []
        else:
            return section, section.content, False, issues
    except Exception as e:
        return section, section.content, False, [str(e)]


def process_chapter_file(
    file_path: Path,
    backend: str = "agy",
    model: Optional[str] = None,
    workers: int = 4,
    timeout: int = 300,
    dry_run: bool = False,
    target_section: Optional[int] = None,
) -> None:
    """Decomposes a chapter into sections and refines them in parallel."""
    ch_name = file_path.parent.name
    print(f"\n========================================================")
    print(f"📖 [{ch_name}] Processing: {file_path.relative_to(REPO_ROOT)}")
    print(f"========================================================")

    text = file_path.read_text(encoding="utf-8")
    sections = split_qmd_into_sections(text)

    # Extract brief chapter context from frontmatter
    chapter_context = sections[0].content if sections else ""

    content_sections = [s for s in sections if not s.is_frontmatter and "## Exercises" not in s.heading_line]
    if target_section is not None:
        content_sections = [s for s in content_sections if s.index == target_section]

    print(f"[{ch_name}] Found {len(sections)} chunks ({len(content_sections)} content sections to refine).")

    for s in content_sections:
        print(f"[{ch_name}]   • Section {s.index}: {s.title} ({len(s.content.splitlines())} lines)")

    if dry_run:
        print(f"[{ch_name}] [DRY-RUN] No edits applied.")
        return

    # Backup original file
    backup_path = file_path.with_suffix(".qmd.bak")
    shutil.copy2(file_path, backup_path)
    print(f"[{ch_name}] Created backup at {backup_path.name}")

    results: Dict[int, str] = {}
    errors: Dict[int, List[str]] = {}

    start_time = time.time()
    sections_to_process = content_sections
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                refine_single_section,
                sec,
                chapter_context,
                backend,
                model,
                timeout,
            ): sec
            for sec in sections_to_process
        }

        for future in concurrent.futures.as_completed(future_map):
            sec = future_map[future]
            try:
                orig_sec, refined_content, success, issues = future.result()
                results[orig_sec.index] = refined_content
                if success:
                    print(f"[{ch_name}]   ✅ Section {orig_sec.index} [{orig_sec.title}]: Refined successfully.")
                else:
                    print(f"[{ch_name}]   ⚠️ Section {orig_sec.index} [{orig_sec.title}]: Reverted to original due to validation: {issues}")
                    errors[orig_sec.index] = issues
            except Exception as e:
                print(f"[{ch_name}]   ❌ Section {sec.index} [{sec.title}]: Failed with error: {e}")
                results[sec.index] = sec.content
                errors[sec.index] = [str(e)]

    # Reassemble sections in original order
    ordered_content = []
    for i in range(len(sections)):
        ordered_content.append(results.get(i, sections[i].content))

    reassembled_text = "\n".join(chunk.strip("\n") for chunk in ordered_content) + "\n"

    # Quick sanity check on complete reassembled text
    open_c = len(re.findall(r"^:::\s*\{", reassembled_text, flags=re.MULTILINE))
    close_c = len(re.findall(r"^:::\s*$", reassembled_text, flags=re.MULTILINE))
    if open_c != close_c:
        print(f"[{ch_name}] 🚨 Final reassembly check failed (mismatched ::: {open_c} vs {close_c}). Retaining backup!")
        file_path.write_text(reassembled_text, encoding="utf-8")
        print(f"[{ch_name}] ⚠️ Reassembled file written, but backup is preserved at {backup_path.name}")
    else:
        file_path.write_text(reassembled_text, encoding="utf-8")
        print(f"[{ch_name}] ✨ Chapter successfully updated: {file_path.name} ({time.time() - start_time:.1f}s)")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel Section-by-Section Streamlining for Volume III")
    parser.add_argument("--chapter", "-c", type=str, help="Specific chapter number to process (e.g., '02' or '03')")
    parser.add_argument("--section", "-s", type=int, default=None, help="Optional specific section index to process")
    parser.add_argument("--file", "-f", type=Path, help="Specific .qmd file to process")
    parser.add_argument("--all", action="store_true", help="Process all chapter .qmd files in books/vol3")
    parser.add_argument("--chapter-workers", "-cw", type=int, default=1, help="Number of chapters to process concurrently (default: 1)")
    parser.add_argument("--backend", choices=["agy", "claude", "mock"], default="agy", help="Agent backend CLI to invoke")
    parser.add_argument("--model", type=str, default=None, help="Optional model identifier override")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of concurrent section agents per chapter (default: 4)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-section agent timeout in seconds (default: 300)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect section splits without dispatching agents")

    args = parser.parse_args()

    target_files: List[Path] = []

    if args.file:
        target_files.append(args.file.resolve())
    elif args.chapter:
        target_num = args.chapter.zfill(2)
        candidates = list(VOL3_ROOT.glob(f"{target_num}_*/*.qmd"))
        if not candidates:
            print(f"Error: No chapter found matching '{target_num}_*' in {VOL3_ROOT}")
            sys.exit(1)
        target_files.append(candidates[0])
    elif args.all:
        for ch_dir in sorted(VOL3_ROOT.glob("[0-9][0-9]_*")):
            qmds = list(ch_dir.glob("*.qmd"))
            if qmds:
                target_files.append(qmds[0])
    else:
        # Default to Chapter 02 if no args given
        print("No target specified. Use --chapter <num>, --file <path>, or --all.")
        print("Example: python scripts/streamline_vol3_sections.py --chapter 02 --dry-run")
        sys.exit(0)

    print(f"🚀 Starting Section-by-Section Streamline Engine")
    print(f"Backend: {args.backend} | Chapter Workers: {args.chapter_workers} | Section Workers/Ch: {args.workers} | Targets: {len(target_files)} files")

    if args.chapter_workers > 1 and len(target_files) > 1:
        print(f"Dispatching {len(target_files)} chapters across {args.chapter_workers} chapter threads...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.chapter_workers) as chapter_pool:
            futures = {
                chapter_pool.submit(
                    process_chapter_file,
                    file_path=f,
                    backend=args.backend,
                    model=args.model,
                    workers=args.workers,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    target_section=args.section,
                ): f for f in target_files if f.exists()
            }
            for fut in concurrent.futures.as_completed(futures):
                ch_file = futures[fut]
                try:
                    fut.result()
                    print(f"🏁 Finished chapter: {ch_file.name}")
                except Exception as e:
                    print(f"❌ Error processing chapter {ch_file.name}: {e}")
    else:
        for f in target_files:
            if not f.exists():
                print(f"File not found: {f}")
                continue
            process_chapter_file(
                file_path=f,
                backend=args.backend,
                model=args.model,
                workers=args.workers,
                timeout=args.timeout,
                dry_run=args.dry_run,
                target_section=args.section,
            )


if __name__ == "__main__":
    main()
