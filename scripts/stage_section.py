#!/usr/bin/env python3
"""
stage_section.py — Autonomous Section Staging & Prompt Generator for Volume III.

Prepares an authoritative, self-contained prompt to launch an interactive agent
session for drafting or revising any section in Volume III: The Stochastic Computer.

Key Features:
- Semantic extraction from MASTER_TEXTBOOK_OUTLINE_V2.md (no fragile line numbers).
- Injects Chapter North Star (The Core Question & Why It Matters) and Learning Objectives.
- Extracts previous section handoff context from the chapter .qmd file for seamless continuity.
- Enforces the Three-Filter Acceptance Test and Metaphor Boundary Directive.
- Targets in-place insertion/replacement between <!-- SECTION_START --> and <!-- SECTION_END --> tags.
- Copies the final prompt directly to your macOS clipboard (via pbcopy) and saves to scratch/.

Usage:
    # Stage Section 1.1 (copies prompt to clipboard)
    python3 scripts/stage_section.py 1.1

    # Stage Section 1.2 (extracts Section 1.1's trailing handoff automatically)
    python3 scripts/stage_section.py 1.2

    # View chapter drafting status (word counts, existing vs pending sections)
    python3 scripts/stage_section.py --status 1

    # Print prompt to stdout in addition to clipboard
    python3 scripts/stage_section.py 1.1 --print
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTLINE_PATH = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V2.md"
VOL3_DIR = REPO_ROOT / "books" / "vol3"
PROMPTS_DIR = REPO_ROOT / "scratch" / "prompts"

SLUG_MAP: Dict[str, str] = {
    "01": "01_introduction",
    "02": "02_processor",
    "03": "03_deliberation",
    "04": "04_working_sets",
    "05": "05_virtual_memory",
    "06": "06_episodic_memory",
    "07": "07_actuation",
    "08": "08_virtualization",
    "09": "09_checkpointing",
    "10": "10_interrupts",
    "11": "11_scheduling",
    "12": "12_data_flywheel",
    "13": "13_sft",
    "14": "14_rlvr",
    "15": "15_multi_agent",
    "16": "16_observability",
    "17": "17_tokenomics",
    "18": "18_conclusion",
}


def get_chapter_qmd_path(chap_num_str: str) -> Optional[Path]:
    chap_zfill = chap_num_str.zfill(2)
    slug = SLUG_MAP.get(chap_zfill)
    if not slug:
        return None
    qmd_name = f"{slug.split('_', 1)[1] if '_' in slug else slug}.qmd"
    # Special casing if dir is 01_introduction and file is 01_introduction.qmd
    cand1 = VOL3_DIR / slug / f"{slug}.qmd"
    if cand1.exists():
        return cand1
    cand2 = VOL3_DIR / slug / qmd_name
    if cand2.exists():
        return cand2
    # Fallback to any qmd in dir
    ch_dir = VOL3_DIR / slug
    if ch_dir.exists():
        qmds = list(ch_dir.glob("*.qmd"))
        if qmds:
            return qmds[0]
    return None


def extract_chapter_from_outline(chap_num_str: str) -> Dict[str, Any]:
    """Extract chapter purpose, objectives, and raw text from V2 outline."""
    if not OUTLINE_PATH.exists():
        raise FileNotFoundError(f"Outline not found at: {OUTLINE_PATH}")

    content = OUTLINE_PATH.read_text(encoding="utf-8")
    chap_zfill = chap_num_str.zfill(2)

    chap_pattern = (
        r"### Chapter "
        + re.escape(chap_zfill)
        + r":\s*([^\n]+)\n(.*?)(?=\n### Chapter |\n## Part |\Z)"
    )
    match = re.search(chap_pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Chapter {chap_num_str} not found in {OUTLINE_PATH}")

    title = match.group(1).strip()
    body = match.group(2)

    # Extract Purpose (The Core Question & Why It Matters)
    core_question = ""
    why_it_matters = ""
    purpose_match = re.search(
        r"#### Purpose \{\.unnumbered \.unlisted\}\n\n\*\*The Core Question:\*\*\s*\*(.+?)\*\n\n\*\*Why It Matters:\*\*\s*(.+?)(?=\n\n:::)",
        body,
        re.DOTALL,
    )
    if purpose_match:
        core_question = purpose_match.group(1).strip()
        why_it_matters = purpose_match.group(2).strip()

    # Extract Learning Objectives
    objectives = []
    obj_match = re.search(
        r"::: \{\.callout-learning-objectives\}\n\n(.*?)\n\n:::", body, re.DOTALL
    )
    if obj_match:
        obj_lines = obj_match.group(1).strip().splitlines()
        objectives = [
            line.lstrip("- ").strip() for line in obj_lines if line.strip().startswith("-")
        ]

    return {
        "chap_num": chap_zfill,
        "title": title,
        "core_question": core_question,
        "why_it_matters": why_it_matters,
        "objectives": objectives,
        "raw_body": body,
    }


def extract_section_from_outline(chap_body: str, sec_id: str) -> Dict[str, Any]:
    """Extract the exact section blueprint from the chapter outline text."""
    sec_escaped = re.escape(sec_id)
    sec_pattern = (
        r"#### Section "
        + sec_escaped
        + r":\s*([^\n]+)\n(.*?)(?=\n#### Section |\n#### Fallacies |\n## Fallacies |\n### Chapter |\n## Part |\Z)"
    )
    match = re.search(sec_pattern, chap_body, re.DOTALL)
    if not match:
        raise ValueError(f"Section {sec_id} not found in outline.")

    title = match.group(1).strip()
    sec_body = match.group(2).strip()

    # Extract components
    def get_field(field_name: str) -> str:
        f_pattern = r"- \*\*" + re.escape(field_name) + r":\*\*\s*([^\n]+)"
        m = re.search(f_pattern, sec_body)
        return m.group(1).strip() if m else ""

    heading_anchor = get_field("Heading & Anchor")
    structural_invariant = get_field("Structural Invariant")
    key_point = get_field("The Single Key Point")
    curricular_placement = get_field("Curricular Placement")
    causal_bridge = get_field("Causal Bridge to .*") or get_field("Causal Bridge to 1.2")

    return {
        "sec_id": sec_id,
        "title": title,
        "heading_anchor": heading_anchor,
        "structural_invariant": structural_invariant,
        "key_point": key_point,
        "curricular_placement": curricular_placement,
        "raw_body": sec_body,
    }


def inspect_manuscript(qmd_path: Path, chap_num_str: str) -> Dict[str, Any]:
    """Inspect chapter manuscript or modular draft sections directory for existing sections and word counts."""
    chap_zfill = chap_num_str.zfill(2)
    # Check for modular draft directory under drafts/vol3/chXX_.../sections/
    drafts_ch_dir = REPO_ROOT / "drafts" / "vol3" / f"ch{chap_zfill}_{qmd_path.parent.name.split('_', 1)[1] if '_' in qmd_path.parent.name else qmd_path.parent.name}" / "sections"
    if not drafts_ch_dir.exists():
        # Fallback to direct name match
        drafts_ch_dir = REPO_ROOT / "drafts" / "vol3" / f"ch{chap_zfill}_{qmd_path.parent.name}" / "sections"
        if not drafts_ch_dir.exists():
            # Check without ch prefix
            drafts_ch_dir = REPO_ROOT / "drafts" / "vol3" / qmd_path.parent.name / "sections"

    if drafts_ch_dir.exists():
        sec_files = sorted(drafts_ch_dir.glob("*.qmd"))
        sections = []
        for sf in sec_files:
            sf_name = sf.name
            # parse sec_num from filename e.g. 01_sec_1_1.qmd -> 1.1
            m = re.search(r"sec_(\d+)_(\d+)", sf_name)
            if m:
                s_num = f"{m.group(1)}.{m.group(2)}"
            elif "frontmatter" in sf_name:
                s_num = "Frontmatter"
            elif "fallacies" in sf_name:
                s_num = f"{int(chap_num_str)}.fallacies"
            elif "summary" in sf_name:
                s_num = f"{int(chap_num_str)}.summary"
            else:
                s_num = sf_name

            sf_content = sf.read_text(encoding="utf-8")
            words = len(sf_content.split())
            h_m = re.search(r"##\s+([^\n]+)", sf_content)
            h_title = h_m.group(1) if h_m else sf.stem
            h_clean = re.sub(r"\{#[^}]+\}", "", h_title).strip()
            sections.append({
                "sec_num": s_num,
                "anchor": "",
                "words": words,
                "heading": h_clean,
                "text": sf_content,
                "file_path": sf,
            })
        return {"exists": True, "sections": sections, "raw": "", "is_modular": True, "modular_dir": drafts_ch_dir}

    if not qmd_path.exists():
        return {"exists": False, "sections": [], "raw": "", "is_modular": False}

    content = qmd_path.read_text(encoding="utf-8")
    sections = []

    # Check for tagged sections <!-- SECTION_START: X.Y (...) -->
    tag_matches = list(
        re.finditer(
            r"<!-- SECTION_START:\s*([0-9]+\.[0-9]+)\s*(\([^)]+\))?\s*-->\n(.*?)(?=<!-- SECTION_END:\s*\1\s*-->)",
            content,
            re.DOTALL,
        )
    )

    if tag_matches:
        for m in tag_matches:
            sec_num = m.group(1)
            anchor = m.group(2) or ""
            sec_text = m.group(3).strip()
            words = len(sec_text.split())
            h_m = re.search(r"##\s+([^\n]+)", sec_text)
            heading = h_m.group(1) if h_m else "Untargeted"
            sections.append(
                {
                    "sec_num": sec_num,
                    "anchor": anchor,
                    "words": words,
                    "heading": heading,
                    "text": sec_text,
                }
            )
    else:
        # Fallback to ## headings
        chap_num = qmd_path.parent.name.split("_")[0].lstrip("0")
        h2_matches = list(re.finditer(r"^##\s+([^#\n]+)", content, re.MULTILINE))
        body_idx = 1
        for i, m in enumerate(h2_matches):
            raw_title = m.group(1).strip()
            if "{.unnumbered" in raw_title:
                continue
            h_clean = re.sub(r"\{#[^}]+\}", "", raw_title).strip()
            start_pos = m.start()
            end_pos = (
                h2_matches[i + 1].start()
                if i + 1 < len(h2_matches)
                else len(content)
            )
            sec_text = content[start_pos:end_pos].strip()
            words = len(sec_text.split())

            if "fallacies" in h_clean.lower():
                s_id = f"{chap_num}.fallacies"
            elif "summary" in h_clean.lower():
                s_id = f"{chap_num}.summary"
            else:
                s_id = f"{chap_num}.{body_idx}"
                body_idx += 1

            sections.append(
                {
                    "sec_num": s_id,
                    "anchor": "",
                    "words": words,
                    "heading": h_clean,
                    "text": sec_text,
                }
            )

    return {"exists": True, "sections": sections, "raw": content}


def get_preceding_handoff(
    manuscript_info: Dict[str, Any], target_sec_id: str
) -> Tuple[str, str]:
    """Extract the last ~200 words of the preceding section to bridge context."""
    sections = manuscript_info.get("sections", [])
    if not sections:
        return ("", "No sections drafted in this chapter file yet.")

    # Parse target section number
    try:
        chap_str, sec_str = target_sec_id.split(".")
        sec_int = int(sec_str)
    except Exception:
        return ("", "Non-numeric section.")

    if sec_int <= 1:
        return (
            "",
            "This is Section .1 (the chapter opening stage-setter). No preceding body section exists. Follow the frontmatter Purpose and Learning Objectives.",
        )

    prev_sec_id = f"{int(chap_str)}.{sec_int - 1}"
    prev_sec = next((s for s in sections if s["sec_num"] == prev_sec_id), None)

    if not prev_sec:
        return (
            "",
            f"Preceding Section {prev_sec_id} has not yet been drafted in the chapter file.",
        )

    # Extract last ~2 paragraphs or ~200 words
    paragraphs = [p.strip() for p in prev_sec["text"].split("\n\n") if p.strip()]
    if not paragraphs:
        return (prev_sec["heading"], "Preceding section is empty.")

    trailing_paragraphs = paragraphs[-2:] if len(paragraphs) >= 2 else paragraphs
    handoff_text = "\n\n".join(trailing_paragraphs)
    return (prev_sec["heading"], handoff_text)


def compose_session_prompt(
    chap_info: Dict[str, Any],
    sec_info: Dict[str, Any],
    manuscript_info: Dict[str, Any],
    qmd_path: Path,
) -> str:
    """Compose the turnkey prompt for the new agent session."""
    sec_id = sec_info["sec_id"]
    prev_heading, handoff_text = get_preceding_handoff(manuscript_info, sec_id)

    # Summarize existing chapter state
    existing_sections_summary = []
    for s in manuscript_info.get("sections", []):
        existing_sections_summary.append(
            f"- Section {s['sec_num']}: {s['heading']} ({s['words']} words)"
        )
    existing_summary_str = (
        "\n".join(existing_sections_summary)
        if existing_sections_summary
        else "None yet (this will be the first section drafted)."
    )

    objectives_str = "\n".join(f"- {obj}" for obj in chap_info["objectives"])

    prompt = f"""# TASK: Author Section {sec_id} of Volume III ("The Stochastic Computer")

You are a Senior Principal Machine Learning Systems Architect and Co-Author of *Machine Learning Systems: The Stochastic Computer* (Volume III).
Your mission is to draft or elevate **Section {sec_id}: {sec_info['title']}** in the chapter manuscript.

---

## 1. The Volume Big Picture & Curricular Compass
- **Volume III Architecture (7 Parts, 18 Chapters)**:
  - *Part I: The Stochastic Processor* (Ch 01: The Stochastic Computer, Ch 02: The Stochastic Processor Core, Ch 03: Inference-Time Deliberation)
  - *Part II: Context Memory & Storage* (Ch 04: Logical Working Sets, Ch 05: Virtual Context Memory & KV Management, Ch 06: Episodic Storage & Retrieval)
  - *Part III: Peripherals & Sandboxing* (Ch 07: Peripheral Actuation & Tool Interfaces, Ch 08: Sandboxing & Virtualization)
  - *Part IV: The Agent Operating System* (Ch 09: The Agent OS Control Plane, Ch 10: Trajectory State & Persistence, Ch 11: Fault Tolerance & Sagas)
  - *Part V: Continuous Learning & Adaptation* (Ch 12: The Trajectory Data Flywheel, Ch 13: Supervised Policy Adaptation, Ch 14: Reinforcement Learning with Verifiable Rewards)
  - *Part VI: Fleet Orchestration & Economics* (Ch 15: Multi-Agent Fleets, Ch 16: Evaluation & Observability, Ch 17: Fleet Sizing & Tokenomics)
  - *Part VII: Systems Synthesis* (Ch 18: System Synthesis & Release Assurance)
- **Target Audience & Pedagogy**: Graduate CS students and senior systems engineers with backgrounds in OS, distributed systems, and computer architecture. This textbook must match the pedagogical caliber of Hennessy & Patterson (*Computer Architecture: A Quantitative Approach*) and Saltzer & Kaashoek (*Principles of Computer System Design*).

---

## 2. Chapter North Star & Curricular Position
- **Chapter**: Chapter {chap_info['chap_num']}: {chap_info['title']}
- **Manuscript File**: [`{qmd_path.relative_to(REPO_ROOT)}`](file:///{qmd_path})
- **The Core Question**: *{chap_info['core_question']}*
- **Why It Matters**: {chap_info['why_it_matters']}

### Chapter Learning Objectives:
{objectives_str}

---

## 3. Prior Chapter State & Handoff Context
- **Sections Already Drafted in this Chapter**:
{existing_summary_str}

- **Preceding Section Narrative Handoff (Section {sec_id.split('.')[0]}.{int(sec_id.split('.')[1])-1 if '.' in sec_id and int(sec_id.split('.')[1])>1 else 'Frontmatter'})**:
\"\"\"
{handoff_text}
\"\"\"
*Ensure Section {sec_id} picks up this narrative baton smoothly, maintaining conceptual continuity without repeating the preceding exposition.*

---

## 4. Specification & Blueprint for Section {sec_id}
*(From `MASTER_TEXTBOOK_OUTLINE_V2.md`)*

{sec_info['raw_body']}

---

## 5. Systems Authoring Directives & Universal Invariants

Every paragraph must strictly adhere to the following rules:

1. 🎓 **Textbook Pedagogy & Engineering Rigor (Hennessy & Patterson Caliber)**:
   - **Do Not Write an Abstract Monologue or Whitepaper**: Concepts cannot merely be declared in prose; they must be taught through concrete mechanisms, physical numbers, and explicit structures.
   - **Canonical Callout Architecture**: Incorporate the registered series callouts wherever prescribed in the blueprint:
     - `::: {{#dfn-... .callout-definition title="..."}}`: Formal definitions of systems concepts.
     - `::: {{#pri-... .callout-principle title="..."}}`: Durable architectural design principles.
     - `::: {{#nbk-... .callout-notebook title="Napkin Math: ..."}}`: Quantitative back-of-the-envelope calculations with real hardware parameters (HBM bandwidth, dollar costs, latency budgets, Amdahl speedups).
     - `::: {{#exmp-... .callout-example title="..."}}`: Step-by-step concrete execution traces (e.g. JSON-RPC tool calls, exit codes, state transitions).
     - `::: {{#chk-... .callout-checkpoint title="..."}}`: Pause-and-reflect concept checks prompting students to analyze architectural trade-offs.
     - `::: {{#ws-... .callout-war-story title="War Story: ..."}}`: Real-world production incident retrospectives illustrating failure modes.

2. 🛑 **The Three-Filter Acceptance Test**:
   - **Filter 1 (Anti-NLP Gate):** No prompt engineering tricks ("let's think step by step"), conversational chatbot banter, persona roleplay, human dialogue flows, or subjective "LLM-as-a-judge" evaluation. Re-anchor in runtime state machines, typed RPC schemas, deterministic compilers, and POSIX exit codes.
   - **Filter 2 (Anti-Silicon Gate):** Do not dump raw GPU hardware specs prematurely without systems context, pretend attention heads are literal x86 ALUs/registers, or treat token IDs as machine opcodes. Re-anchor in the software systems layer (the Agent OS control plane, host runtime, memory allocation, and container/microVM sandboxing).
   - **Filter 3 (Systems Engineering Gate):** Define an explicit interface contract, an operational trade-off, a physical latency/memory cost, an error boundary, or an external mechanical verification protocol.

3. 💡 **The Metaphor Boundary Directive**:
   - *The Stochastic Computer* is a **software-level functional architecture**, not a physical silicon blueprint.
   - The foundation model functions as an unprivileged, non-deterministic execution core evaluated under **Zero Ambient Authority**.
   - Hardware analogies (ALU, registers, bus lines) must only be used as illustrative functional bridges. Token IDs are discrete integer indices, never executable machine opcodes.

4. 📐 **Structural & Stylistic Invariants**:
   - {"**NO SUBSECTIONS (NO ###).** Section .1 must be an unbroken narrative prose arc." if sec_id.endswith(".1") else "Use logical subsections (###) where appropriate to structure the systems mechanisms."}
   - **Zero Word-Count Restrictions:** Pacing follows subject mass and technical density. Be dense, rigorous, and complete; do not pad or prematurely truncate.
   - **American English:** Use American English spelling (`-ize` not `-ise`, `-or` not `-our`, `center`, `defense`, single-`l` past tenses like `labeled`, `modeled`).
   - **Mathematical & Quarto Hygiene:**
     - In prose, write `percent` instead of the raw `%` symbol (e.g. `95 percent`, not `95%`).
     - Escape currency symbols with backslashes (`\\$15.00`).
     - Use canonical units (`GB` not `GiB` in prose per style guide).
     - Cross-reference figures and tables properly (`@fig-...`, `@tbl-...`, `@dfn-...`, `@pri-...`, `@nbk-...`, `@chk-...`).

---

## 6. Execution Target & File Update Contract

1. Open and inspect [`{qmd_path.relative_to(REPO_ROOT)}`](file:///{qmd_path}).
2. Locate the section comment boundaries:
   ```markdown
   <!-- SECTION_START: {sec_id} ({sec_info['heading_anchor'].split('{')[-1].rstrip('}') if '{' in sec_info['heading_anchor'] else ''}) -->
   ... section content ...
   <!-- SECTION_END: {sec_id} -->
   ```
3. Use the file editing tool to write or replace the content strictly between these markers.
4. Run binder checks on the modified file to verify prose, structure, and math hygiene:
   `./binder/binder check all --path {qmd_path.relative_to(REPO_ROOT)}`
5. Report the completed word count, key systems invariants established, and the trailing handoff for Section {sec_id.split('.')[0]}.{int(sec_id.split('.')[1])+1 if '.' in sec_id else 'Next'}.
"""
    return prompt.strip()


def copy_to_clipboard(text: str) -> bool:
    """Copy text to macOS clipboard using pbcopy."""
    try:
        process = subprocess.Popen(
            ["pbcopy"], stdin=subprocess.PIPE, close_fds=True
        )
        process.communicate(input=text.encode("utf-8"))
        return process.returncode == 0
    except Exception:
        return False


def print_chapter_status(chap_num_str: str) -> None:
    """Display tabular status of a chapter's drafted sections."""
    chap_info = extract_chapter_from_outline(chap_num_str)
    qmd_path = get_chapter_qmd_path(chap_num_str)
    if not qmd_path:
        print(f"Error: Could not locate .qmd file for Chapter {chap_num_str}")
        return

    manuscript_info = inspect_manuscript(qmd_path, chap_num_str)

    print(
        f"\n================================================================================"
    )
    print(f"  CHAPTER {chap_info['chap_num']}: {chap_info['title']}")
    print(f"  File: {qmd_path.relative_to(REPO_ROOT)}")
    print(
        f"================================================================================"
    )
    print(f"  Core Question: {chap_info['core_question']}")
    print(f"  Why It Matters: {chap_info['why_it_matters'][:120]}...\n")

    # Parse all sections defined in outline
    sec_matches = list(
        re.finditer(
            r"#### Section ([0-9]+\.[0-9]+):\s*([^\n]+)", chap_info["raw_body"]
        )
    )
    existing_map = {
        s["sec_num"]: s for s in manuscript_info.get("sections", [])
    }

    total_words = sum(s["words"] for s in manuscript_info.get("sections", []))

    print(
        f"  {'Section':<10} | {'Status':<12} | {'Words':<8} | {'Heading'}"
    )
    print(f"  {'-'*10}-|-{'-'*12}-|-{'-'*8}-|-{'-'*40}")

    for m in sec_matches:
        sec_id = m.group(1)
        sec_title = m.group(2)
        if sec_id in existing_map:
            words = existing_map[sec_id]["words"]
            status = "🟢 Drafted"
        else:
            words = 0
            status = "⚪ Pending"
        print(f"  {sec_id:<10} | {status:<12} | {words:<8} | {sec_title[:45]}")

    print(f"  {'-'*10}-|-{'-'*12}-|-{'-'*8}-|-{'-'*40}")
    print(f"  Total Words in Manuscript: {total_words:,} words\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage a section and generate turnkey session prompt."
    )
    parser.add_argument(
        "section",
        nargs="?",
        help="Section identifier to stage (e.g. '1.1', '2.1', '18.4').",
    )
    parser.add_argument(
        "--status",
        metavar="CHAPTER",
        help="Display drafting status for a chapter (e.g. '--status 1').",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the full prompt to stdout.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy the generated prompt to the clipboard.",
    )

    args = parser.parse_args()

    if args.status:
        print_chapter_status(args.status)
        return 0

    if not args.section:
        parser.print_help()
        return 1

    sec_id = args.section.strip()
    if "." not in sec_id:
        print(f"Error: Invalid section identifier '{sec_id}'. Must be in format X.Y (e.g. '1.1').")
        return 1

    chap_num_str = sec_id.split(".")[0]
    qmd_path = get_chapter_qmd_path(chap_num_str)
    if not qmd_path:
        print(f"Error: Could not locate .qmd file for Chapter {chap_num_str}")
        return 1

    try:
        chap_info = extract_chapter_from_outline(chap_num_str)
        sec_info = extract_section_from_outline(chap_info["raw_body"], sec_id)
    except Exception as e:
        print(f"Error parsing outline: {e}")
        return 1

    manuscript_info = inspect_manuscript(qmd_path, chap_num_str)
    prompt = compose_session_prompt(chap_info, sec_info, manuscript_info, qmd_path)

    # Save prompt to scratch directory
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    sec_slug = sec_id.replace(".", "_")
    prompt_file = PROMPTS_DIR / f"stage_sec_{sec_slug}.md"
    prompt_file.write_text(prompt, encoding="utf-8")

    # Clipboard copy
    copied = False
    if not args.no_copy:
        copied = copy_to_clipboard(prompt)

    # Terminal summary
    print(
        f"\n================================================================================"
    )
    print(f"  🎯 SECTION STAGED: Section {sec_id} — {sec_info['title']}")
    print(
        f"================================================================================"
    )
    print(f"  • Target File: {qmd_path.relative_to(REPO_ROOT)}")
    print(f"  • Prompt Saved: {prompt_file.relative_to(REPO_ROOT)}")
    if copied:
        print(f"  • 📋 Prompt successfully copied to macOS clipboard!")
        print(f"  • Action: Open a new agent session and simply paste to begin drafting.")
    else:
        print(f"  • Clipboard copy skipped or unavailable.")

    # Show existing status
    sec_match = next(
        (s for s in manuscript_info.get("sections", []) if s["sec_num"] == sec_id),
        None,
    )
    if sec_match:
        print(
            f"  • Current Manuscript State: 🟢 Existing draft ({sec_match['words']} words) will be updated."
        )
    else:
        print(
            f"  • Current Manuscript State: ⚪ Fresh section to be inserted."
        )

    print(
        f"================================================================================\n"
    )

    if args.print:
        print(prompt)

    return 0


if __name__ == "__main__":
    sys.exit(main())
