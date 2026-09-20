#!/usr/bin/env python3
"""
Generate Chapter from V3 Master Outline (Multi-Call Systems Engineering Engine)

Authoritative chapter generation pipeline for Volume III: The Stochastic Computer.
Grounds the textbook in the systems design principles of Jerome H. Saltzer & M. Frans Kaashoek (MIT 6.033),
combining the quantitative hardware rigor of Hennessy & Patterson with the demystified software perspective
of Bryant & O'Hallaron (CS:APP).

Architecture:
- Sequential in-chapter multi-call execution (one model call per section).
- 4-Tier Bounded Context Staging Schema (Tier 1: S&K + Vol 1 Stance, Tier 2: Macro Blueprint & Chapter Roadmap,
  Tier 3: Upstream Interface & Active Section Spec, Tier 4: Forward Handoff).
- Section .1 Unbroken Narrative Invariant (Zero ### subsections, zero bullet lists in narrative, 4-beat stage-setting arc).
- Sections .2+ Analytical Mechanics (2–4 clean ### subheadings, code snippets, boxed worked examples, equations with physical units).
- Deterministic Step 0 Frontmatter generation from V3 manifest.
- Deterministic Pre-Merge Review Gates with automated targeted repair loop.
- Dynamic Active Symbol Registry and Terminal Bridge context accumulator.
- Clean-slate workspace under `drafts/vol3/chXX_<slug>/` and canonical staging to `books/vol3/XX_<slug>/XX_<slug>.qmd`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# Root paths
REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_OUTLINE_V3_PATH = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V3.md"
DRAFTS_ROOT = REPO_ROOT / "drafts" / "vol3"

SLUG_MAP: Dict[str, str] = {
    "01": "introduction",
    "02": "processor",
    "03": "deliberation",
    "04": "working_sets",
    "05": "virtual_memory",
    "06": "episodic_memory",
    "07": "actuation",
    "08": "virtualization",
    "09": "checkpointing",
    "10": "interrupts",
    "11": "scheduling",
    "12": "data_flywheel",
    "13": "sft",
    "14": "rlvr",
    "15": "multi_agent",
    "16": "observability",
    "17": "tokenomics",
    "18": "conclusion",
}

STACK_MAP: Dict[str, str] = {
    "01": r"\mlagentstack{17}{17}{17}{17}{16}{16}",
    "02": r"\mlagentstack{0}{0}{0}{0}{0}{100}",
    "03": r"\mlagentstack{0}{0}{0}{0}{0}{100}",
    "04": r"\mlagentstack{0}{0}{0}{0}{100}{0}",
    "05": r"\mlagentstack{0}{0}{0}{0}{100}{0}",
    "06": r"\mlagentstack{0}{0}{0}{0}{100}{0}",
    "07": r"\mlagentstack{0}{0}{0}{100}{0}{0}",
    "08": r"\mlagentstack{0}{0}{0}{100}{0}{0}",
    "09": r"\mlagentstack{0}{0}{100}{0}{0}{0}",
    "10": r"\mlagentstack{0}{0}{100}{0}{0}{0}",
    "11": r"\mlagentstack{0}{0}{100}{0}{0}{0}",
    "12": r"\mlagentstack{0}{100}{0}{0}{0}{0}",
    "13": r"\mlagentstack{0}{100}{0}{0}{0}{0}",
    "14": r"\mlagentstack{0}{100}{0}{0}{0}{0}",
    "15": r"\mlagentstack{100}{0}{0}{0}{0}{0}",
    "16": r"\mlagentstack{100}{0}{0}{0}{0}{0}",
    "17": r"\mlagentstack{100}{0}{0}{0}{0}{0}",
    "18": r"\mlagentstack{17}{17}{17}{17}{16}{16}",
}


# ==============================================================================
# 1. DATA STRUCTURES & MANIFEST
# ==============================================================================

@dataclass
class SectionSpec:
    section_num: str
    title: str
    heading_anchor: str
    budget_target: int
    budget_range: Tuple[int, int]
    structural_invariant: str
    key_point: str
    points: str
    visuals: str
    literature: str
    negative_scope: str = ""
    raw_spec: str = ""


@dataclass
class ChapterManifest:
    number: str
    title: str
    slug: str
    subtitle: str
    core_takeaway: str
    governing_question: str
    curricular_role: str
    purpose: str
    objectives: List[str]
    sections: List[SectionSpec]
    fallacies_raw: str
    summary_raw: str
    raw_outline_text: str = ""
    tier_1_guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChapterManifest:
        sections = [SectionSpec(**s) for s in data.get("sections", [])]
        data_copy = dict(data)
        data_copy["sections"] = sections
        data_copy.setdefault("subtitle", "")
        data_copy.setdefault("tier_1_guidance", "")
        return cls(**data_copy)


# ==============================================================================
# 2. V3 OUTLINE PARSER
# ==============================================================================

def parse_chapter_v3(chapter_num: str, outline_path: Path = MASTER_OUTLINE_V3_PATH) -> ChapterManifest:
    """Parse a single chapter's blueprint from MASTER_TEXTBOOK_OUTLINE_V3.md."""
    if not outline_path.exists():
        raise FileNotFoundError(f"V3 Master outline not found at: {outline_path}")

    content = outline_path.read_text(encoding="utf-8")
    target_num = chapter_num.zfill(2)

    # Extract frontmatter before blueprints split marker as tier_1_guidance
    split_marker = "# Detailed Chapter-by-Chapter Curricular Blueprints"
    frontmatter = ""
    if split_marker in content:
        frontmatter = content.split(split_marker)[0].strip()

    # Split by chapter header: "### Chapter XX:"
    parts = re.split(r"(?=### Chapter \d+:)", content)
    matched_part = None
    for p in parts[1:]:
        header_line = p.strip().splitlines()[0]
        m = re.search(r"### Chapter (\d+):\s*(.*)", header_line)
        if m and m.group(1).zfill(2) == target_num:
            matched_part = p
            break

    if not matched_part:
        raise ValueError(f"Chapter {chapter_num} not found in {outline_path}")

    lines = matched_part.strip().splitlines()
    header_m = re.search(r"### Chapter (\d+):\s*(.*)", lines[0])
    num = header_m.group(1).zfill(2) if header_m else target_num
    raw_title = header_m.group(2).strip() if header_m else f"Chapter {num}"
    # Clean title of any parenthetical suffix like "(Whole-Book Opening Bookend)"
    clean_title = re.sub(r"\s*\(.*?\)", "", raw_title).strip()
    slug = SLUG_MAP.get(num, f"chapter_{num}")

    subtitle_m = re.search(r"-\s+\*\*Subtitle:\*\*\s*(.+?)(?=\n-|\Z)", matched_part)
    subtitle = subtitle_m.group(1).strip() if subtitle_m else ""

    takeaway_m = re.search(r"-\s+\*\*Core Takeaway:\*\*\s*(.+?)(?=\n-|\Z)", matched_part, re.DOTALL)
    core_takeaway = takeaway_m.group(1).strip() if takeaway_m else ""

    question_m = re.search(r"-\s+\*\*Governing Systems Question:\*\*\s*(.+?)(?=\n-|\n####|\Z)", matched_part, re.DOTALL)
    governing_question = question_m.group(1).strip() if question_m else ""

    curricular_m = re.search(r"-\s+\*\*Curricular Role.*?:\*\*\s*(.+?)(?=\n-|\n####|\Z)", matched_part, re.DOTALL)
    curricular_role = curricular_m.group(1).strip() if curricular_m else ""

    purpose_m = re.search(r"#### Purpose\s*\{[^}]*\}\s*(?:\[[^\]]*\])?\s*\n+(.+?)(?=\n::: \{\.callout-learning-objectives\}|\Z)", matched_part, re.DOTALL)
    purpose = purpose_m.group(1).strip() if purpose_m else ""

    objs_m = re.search(r"::: \{\.callout-learning-objectives\}\s*\n+(.+?)(?=\n:::)", matched_part, re.DOTALL)
    objs: List[str] = []
    if objs_m:
        for line in objs_m.group(1).splitlines():
            line = line.strip()
            if line.startswith("- "):
                objs.append(line[2:].strip())

    # Parse sections
    sec_blocks = re.findall(
        r"(#### Section \d+\.\d+:.*?)(?=#### Section \d+\.\d+:|#### Fallacies and Pitfalls|\Z)",
        matched_part,
        re.DOTALL,
    )
    sections: List[SectionSpec] = []
    for b in sec_blocks:
        first_line = b.strip().splitlines()[0]
        sec_title_m = re.search(r"#### Section (\d+\.\d+):\s*(.*)", first_line)
        sec_num = sec_title_m.group(1) if sec_title_m else ""
        raw_sec_title = sec_title_m.group(2) if sec_title_m else ""
        # Clean title of bracket tags like [stage-setter], [core], [synthesis]
        clean_sec_title = re.sub(r"\s*\[.*?\]", "", raw_sec_title).strip()

        is_sec1 = sec_num.endswith(".1")
        if is_sec1:
            budget_tgt = 1400
            budget_rng = (1100, 1900)
            struct_inv = "Section .1 Unbroken Stage-Setter: ZERO ### subsections, ZERO bulleted lists in narrative. Flowing continuous exposition."
        else:
            struct_inv = "2–4 clean, scannable ### subsections. Concrete systems artifacts (structured Markdown comparison tables, failure traces, equations with physical units; NO sprawling Python dataclass boilerplate)."

        heading_anchor_m = re.search(r"-\s+\*\*Heading & Anchor:\*\*\s*`?([^`\n]+)`?", b)
        heading_anchor = heading_anchor_m.group(1).strip() if heading_anchor_m else f"## {clean_sec_title}"

        struct_m = re.search(r"-\s+\*\*Structural Invariant:\*\*\s*(.+?)(?=\n-|\Z)", b, re.DOTALL)
        if struct_m:
            struct_inv = struct_m.group(1).strip()

        key_point_m = re.search(r"-\s+\*\*The Single Key Point:\*\*\s*(.+?)(?=\n-|\Z)", b, re.DOTALL)
        key_point = key_point_m.group(1).strip() if key_point_m else ""

        points_m = re.search(
            r"-\s+\*\*(?:What to Cover|Pedagogical Arc|Points to explain).*?:\*\*\s*(.+?)(?=\n-\s+\*\*What NOT to Cover|\n-\s+\*\*Visuals|\n-\s+\*\*Seminal|\Z)",
            b,
            re.DOTALL,
        )
        points = points_m.group(1).strip() if points_m else ""

        neg_scope_m = re.search(
            r"-\s+\*\*What NOT to Cover.*?:\*\*\s*(.+?)(?=\n-\s+\*\*Visuals|\n-\s+\*\*Seminal|\Z)",
            b,
            re.DOTALL,
        )
        neg_scope = neg_scope_m.group(1).strip() if neg_scope_m else ""

        visuals_m = re.search(r"-\s+\*\*Visuals & Tables:\*\*\s*(.+?)(?=\n-\s+\*\*Seminal|\Z)", b, re.DOTALL)
        visuals = visuals_m.group(1).strip() if visuals_m else ""

        lit_m = re.search(r"-\s+\*\*Seminal Literature:\*\*\s*(.+?)(?=\n-|\Z)", b, re.DOTALL)
        lit = lit_m.group(1).strip() if lit_m else ""

        sections.append(
            SectionSpec(
                section_num=sec_num,
                title=clean_sec_title,
                heading_anchor=heading_anchor,
                budget_target=budget_tgt,
                budget_range=budget_rng,
                structural_invariant=struct_inv,
                key_point=key_point,
                points=points,
                visuals=visuals,
                literature=lit,
                negative_scope=neg_scope,
                raw_spec=b.strip(),
            )
        )

    fallacies_m = re.search(r"#### Fallacies and Pitfalls.*?\n+(.+?)(?=#### Summary|\Z)", matched_part, re.DOTALL)
    fallacies_raw = fallacies_m.group(1).strip() if fallacies_m else ""

    summary_m = re.search(r"#### Summary.*?\n+(.+?)(?=\n### Chapter|\n---\n## Part|\Z)", matched_part, re.DOTALL)
    summary_raw = summary_m.group(1).strip() if summary_m else ""

    return ChapterManifest(
        number=num,
        title=clean_title,
        slug=slug,
        subtitle=subtitle,
        core_takeaway=core_takeaway,
        governing_question=governing_question,
        curricular_role=curricular_role,
        purpose=purpose,
        objectives=objs,
        sections=sections,
        fallacies_raw=fallacies_raw,
        summary_raw=summary_raw,
        raw_outline_text=matched_part,
        tier_1_guidance=frontmatter,
    )


# ==============================================================================
# 3. 4-TIER BOUNDED CONTEXT PROMPT COMPOSER
# ==============================================================================

TIER_1_SYSTEMS_ENGINE = r"""You are an author of the premier senior-undergraduate and introductory graduate computer systems textbook:
'The Stochastic Computer: Agentic Machine Learning Systems' (Volume III).
Your voice, pedagogical clarity, and architectural rigor mirror Jerome H. Saltzer and M. Frans Kaashoek's
'Principles of Computer System Design' (MIT 6.033) and John L. Hennessy and David A. Patterson's
'Computer Architecture: A Quantitative Approach'.

================================================================================
TIER 1: THE SALTZER & KAASHOEK SYSTEMS STANCE & VOLUME 1 CONTINUOUS PROSE STANDARDS
================================================================================

1. AUDIENCE & PEDAGOGICAL ALTITUDE:
   - Target Reader: Senior CS/CE undergraduate or Master's student. Assume an average, solid student aiming to become a professional AI systems engineer.
   - Presumed Student Background: High proficiency in Python and C/C++; standard data structures (trees, hash maps, ring buffers); core Operating Systems (processes, virtual memory, page tables, syscalls, filesystems, concurrency, RPCs); introductory Machine Learning (tensors, matrix multiplication, softmax, loss functions, scaled dot-product attention).
   - Knowledge Gaps to Bridge: Do NOT assume the student has ever built or profiled an LLM serving runtime, knows GPU memory bus bottlenecks, or understands autonomous agent context staging and verification boundaries.
   - Pedagogical Mission: Teach how to build dependable, verifiable software systems around unprivileged, non-deterministic foundation models.

2. AUTHENTIC MACHINE LEARNING SYSTEMS LANGUAGE (NO CPU STRAIGHTJACKET):
   - Speak native, authentic systems engineering language: Large Language Model (LLM), Byte-Pair Encoding (BPE), tokens, embedding tables, autoregressive decode loops, logits, temperature scaling, softmax, KV cache, prefill phase (GEMM), decode phase (GEMV), PagedAttention, Radix trees, inference runtimes (vLLM, SGLang), and host agent runtimes (Claude Code, Devin).
   - Do NOT force artificial CPU roleplay: attention heads are NOT ALUs; prompts are NOT instruction registers; hallucinations are NOT x86 page faults; tokens are NOT opcodes. State the operational model cleanly, define its physical boundaries, and proceed directly to systems engineering.

3. VOLUME 1 CONTINUOUS PROSE STANDARDS:
   - Zero Bullet-Point Splatter in Expository Text: Write continuous, rhythmic, connected prose. Do NOT write lazy bulleted summaries in place of narrative analysis. Bullets are strictly reserved for callout checklists, structured taxonomies, or explicit multi-dimensional comparison tables.
   - Section .1 Unbroken Stage-Setter Invariant: Section .1 must have ZERO `###` subsections and NO bullet-point lists in the main expository narrative. It must be an unbroken, cohesive 4-beat stage-setting narrative:
     Beat 1: The Inflection Point (From Chat to Delegation).
     Beat 2: The Central Systems Paradox (The Unprivileged Predictor under Zero Ambient Authority $A=0$).
     Beat 3: The Open-Loop Failure Wall (Exponential error compounding and the epistemic gap).
     Beat 4: The Closed-Loop Trajectory (Runtime mediation, sandboxed execution, and Dijkstra's empirical verification principle).
     Conclude with an unbroken prose bridge directly posing the first mechanistic question for Section .2.
   - Sections .2 Onwards (The Analytical Systems Mechanics): Use 2–4 clean, scannable `###` subheadings. Unpack concrete data structures, typing schemas, failure traces, and Roofline mathematics.
   - Lead with the Physical Dilemma: Open every section with a concrete systems dilemma or observable failure mode, followed by a bold takeaway paragraph.
   - Boxed Worked Examples: Include Hennessy & Patterson-style boxed worked examples using `::: {.callout-note title="Worked Example: [Title]"}` with concrete hardware numbers and dimensional units.
   - Margin Figures and Notes: Use `::: {.column-margin}` for diagrams, small tables, sidebars, or definitions.
   - American English Spelling: Use -ize, -or, center, defense, meter, labeled, modeled throughout.
   - Zero AI Attribution Trailers: Never emit AI disclaimers, Co-Authored-By lines, or robotic meta-announcements.

4. THE SALTZER & KAASHOEK SYSTEMS FOUNDATIONS:
   - Modularity & Layering: The foundation model is an unprivileged coprocessor; the agent runtime is the supervisor.
   - Virtualization: Virtualizing deliberation (test-time search in Ch 03), virtualizing memory (logical context in Ch 04, PagedAttention physical frames in Ch 05), authoritative source truth vs derivative search indexes (Ch 06).
   - Protection & Least Privilege: Zero Ambient Authority ($A=0$). Candidate proposals are held in memory escrow; tools are mediated RPC peripherals (Ch 07); untrusted executions are isolated inside sandboxes (Ch 08).
   - Fault Tolerance, Atomicity & Sagas: Fail-plausible fault model. Write-Ahead Logging (WAL in Ch 10) and Saga compensating transactions (Ch 11).
   - The End-to-End Argument (Saltzer, Reed, & Clark 1984): Invariant closure is enforced not by model self-reports, but by external deterministic software checks (compilers, linters, sealed test suites in Ch 18).

5. STRICT SCOPE BOUNDARIES & NEGATIVE SCOPES:
   - Every section answers ONE clean architectural question.
   - Strictly obey all negative scope boundaries. Never leak concepts assigned to downstream chapters.

6. STRICT CODE & VISUAL USAGE POLICY (NO DATACLASS DUMPS, NO ASCII ART):
   - NO Sprawling Dataclass Dumps: NEVER emit multi-page Python class or @dataclass boilerplate dumps (e.g. declaring trivial fields like `tool_name: str`, `status: Enum`). Systems interfaces and contracts must be presented as structured Markdown comparison tables or formal mathematical tuples. Code listings must be sparse, empirical, and strictly under 15 lines (reserved for real regression diffs, compiler/pytest failure traces, or concrete Roofline calculations).
   - NO Monospaced ASCII Art: NEVER emit ASCII box diagrams, ASCII text flowcharts, or ASCII line plots inside code blocks. Visuals must be either official vector SVG links (`![](images/svg/...)`) or structured Markdown tables. Raw ASCII art degrades LaTeX and PDF typesetting quality.
   - NO Raw Mermaid Blocks: Do not emit raw mermaid code blocks in expository text. Use clean Markdown tables or refer to SVG figures.
"""


def compose_step_prompt(
    manifest: ChapterManifest,
    step_info: Dict[str, Any],
    upstream_terminal_bridge: str,
    active_symbols: List[str],
    next_step_contract: Optional[Dict[str, str]],
    cumulative_ledger: Optional[List[str]] = None,
) -> str:
    """Compose the complete 4-Tier Bounded Context Staging authoring prompt."""
    step_idx = step_info["step_index"]

    # Tier 2: Chapter Grounding Blueprint (The Big Picture)
    macro_map = (
        "MACRO ARCHITECTURE OF VOLUME III ('THE STOCHASTIC COMPUTER'):\n"
        "  - Part I: Inference Serving & Deliberation (Ch 01: Reference Architecture, Ch 02: Processor Core, Ch 03: Test-Time Deliberation)\n"
        "  - Part II: Context Memory & Serving State (Ch 04: Working Memory, Ch 05: KV-Cache Hierarchy, Ch 06: Persistent Storage)\n"
        "  - Part III: Tool Actuation & Sandboxing (Ch 07: Peripherals & RPC Tools, Ch 08: MicroVM Sandboxing & TCB)\n"
        "  - Part IV: The Agent Operating System (Ch 09: Control Plane/ACB, Ch 10: State WAL, Ch 11: Fault Tolerance & Sagas)\n"
        "  - Part V: The Policy Compiler (Ch 12: Data Flywheel, Ch 13: SFT Distillation, Ch 14: Verifiable RL / RLVR)\n"
        "  - Part VI: Distributed Fleets & Operations (Ch 15: Multi-Agent Concurrency, Ch 16: Observability & SWE-bench, Ch 17: Fleet Economics)\n"
        "  - Part VII: System Synthesis (Ch 18: End-to-End Verification across the Complete Stack)"
    )

    section_roadmap = "\n".join(
        f"  - Section {s.section_num}: {s.title} (Key Point: {s.key_point})"
        for s in manifest.sections
    )

    tier_2 = (
        "================================================================================\n"
        "TIER 2: CHAPTER GROUNDING BLUEPRINT (THE BIG PICTURE)\n"
        "================================================================================\n"
        f"{macro_map}\n\n"
        f"ACTIVE CHAPTER POSITION IN THE MACHINE ARCHITECTURE:\n"
        f"- Chapter {manifest.number}: {manifest.title}\n"
        f"- Chapter Anchor: #sec-vol3-{manifest.slug}\n"
        f"- Curricular Role: {manifest.curricular_role}\n"
        f"- Governing Systems Question: {manifest.governing_question}\n"
        f"- Core Takeaway: {manifest.core_takeaway}\n\n"
        f"Chapter Subsystem Section Roadmap (Separation of Concerns):\n{section_roadmap}\n"
    )

    # Tier 3: Upstream Stitching Interface
    tier_3_parts = [
        "================================================================================\n"
        "TIER 3: UPSTREAM STITCHING INTERFACE (BOUNDED CONTEXT)\n"
        "================================================================================\n"
    ]
    if step_idx == 0:
        tier_3_parts.append("This is the chapter opening scaffold. No preceding chapter section exists.\n")
    elif step_idx == 1:
        tier_3_parts.append(
            f"Preceding Context: Chapter Purpose and Learning Objectives have been staged.\n"
            f"Purpose Summary: {manifest.purpose}\n"
        )
    else:
        tier_3_parts.append(
            f"Terminal Bridge of Preceding Section (last ~200 words):\n"
            f"\"\"\"\n{upstream_terminal_bridge.strip()}\n\"\"\"\n\n"
            f"Active Symbol & Entity Registry (use established notation consistently):\n"
            + (", ".join(f"`{s}`" for s in active_symbols) if active_symbols else "None registered yet.")
            + "\n\n"
        )
        if cumulative_ledger:
            tier_3_parts.append(
                "CUMULATIVE ARCHITECTURAL INVARIANT LEDGER (Established Invariants Across Chapter):\n"
                + "\n".join(f"- {item}" for item in cumulative_ledger)
                + "\n\n"
            )
        tier_3_parts.append(
            "ANTI-RECAP OPENING DIRECTIVE:\n"
            "You are STRICTLY FORBIDDEN from opening with a retrospective summary ('In the previous section...', 'As discussed earlier...').\n"
            "Engage IMMEDIATELY with the technical mechanics or systems problem of this section in your very first sentence.\n"
        )
    tier_3 = "".join(tier_3_parts)

    # Tier 4: Forward Handoff Contract
    tier_4_parts = [
        "================================================================================\n"
        "TIER 4: FORWARD HANDOFF CONTRACT\n"
        "================================================================================\n"
    ]
    if next_step_contract:
        tier_4_parts.append(
            f"Next Section to Follow: {next_step_contract['title']} (`{next_step_contract['heading_anchor']}`)\n"
            f"Next Key Point: {next_step_contract['key_point']}\n"
            f"Directive: Conclude this section with an unbroken causal bridge that naturally sets up this upcoming technical topic.\n"
        )
    else:
        tier_4_parts.append("This is the final concluding section of the chapter. Provide definitive architectural synthesis.\n")
    tier_4 = "".join(tier_4_parts)

    # Task Instruction based on step type
    task_parts = [
        "================================================================================\n"
        "TASK SPECIFICATION & CONSTRAINTS\n"
        "================================================================================\n"
    ]

    if 1 <= step_idx <= len(manifest.sections):
        sec = manifest.sections[step_idx - 1]
        is_sec1 = sec.section_num.endswith(".1")

        negative_scope = ""
        if sec.negative_scope:
            negative_scope = f"\n**NEGATIVE SCOPE BOUNDARIES (STRICTLY FORBIDDEN IN THIS SECTION):**\n{sec.negative_scope.strip()}\n"

        if is_sec1:
            # Strip any contradictory subheading mentions from raw_spec for Section .1
            cleaned_spec = re.sub(r"-\s+\*\*Structural Invariant:\*\*.*?\n", "", sec.raw_spec)
            task_parts.append(
                f"### Task: Author Section {sec.section_num}: {sec.title}\n"
                f"Target Depth: Substantive and thorough exposition for senior CS/CE undergraduates. Let the text breathe naturally without arbitrary word caps.\n\n"
                f"**STRUCTURAL INVARIANT (SECTION .1 UNBROKEN STAGE-SETTER):**\n"
                f"Section .1 must have ZERO subsections (NO `###`) and NO bulleted lists in the main expository narrative. It is an unbroken, cohesive introduction to the chapter as a whole.\n"
                f"Use bold topic sentences to lead into thematic beats if helpful, but preserve a continuous, flowing narrative arc:\n"
                f"  - Beat 1: The Inflection Point / Systems Confrontation.\n"
                f"  - Beat 2: The Physical Boundary / Unprivileged Execution ($A=0$).\n"
                f"  - Beat 3: The Failure Wall / Epistemic Gap.\n"
                f"  - Beat 4: Closed-Loop Execution & External Verification Closure.\n"
                f"  - Conclude with an unbroken prose bridge directly posing the first mechanistic question for Section {manifest.sections[1].section_num if len(manifest.sections) > 1 else 'X.2'}.\n\n"
                f"**Heading & Anchor:** `{sec.heading_anchor}`\n"
                f"**Single Key Point:** {sec.key_point}\n\n"
                f"{negative_scope}\n"
                f"**Exact Section Specification from Master Outline V3:**\n{cleaned_spec}\n\n"
                "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
            )
        else:
            task_parts.append(
                f"### Task: Author Section {sec.section_num}: {sec.title}\n"
                f"Target Depth: Substantive and thorough exposition for senior CS/CE undergraduates. Let the text breathe naturally without arbitrary word caps.\n\n"
                f"**STRUCTURAL INVARIANT:**\n"
                f"Must contain 2–4 clean `###` subsections. Unpack rigorous analytical mechanics, concrete data structures, and failure traces.\n\n"
                f"**Heading & Anchor:** `{sec.heading_anchor}`\n"
                f"**Single Key Point:** {sec.key_point}\n\n"
                f"{negative_scope}\n"
                f"**Exact Section Specification from Master Outline V3:**\n{sec.raw_spec}\n\n"
                "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
            )
    elif step_idx == len(manifest.sections) + 1:
        # Fallacies and Pitfalls
        task_parts.append(
            f"### Task: Author Chapter {manifest.number} Fallacies and Pitfalls\n"
            f"Target Depth: Rigorous and thorough analysis of the Fallacies and Pitfalls established in Outline V3.\n\n"
            f"**Heading & Anchor:** `## Fallacies and Pitfalls {{#sec-vol3-{manifest.slug}-fallacies}}`\n\n"
            f"**Exact Specification from Master Outline V3:**\n{manifest.fallacies_raw}\n\n"
            f"**REQUIRED STRUCTURE:**\n"
            f"Format every entry using standard Quarto callout div syntax:\n\n"
            f"::: {{.fallacy-pitfall}}\n"
            f"**Fallacy:** *[Exact fallacy statement in italics]*\n\n"
            f"[Rigorous systems analysis explaining the failure mechanism and the architectural defense.]\n"
            f":::\n\n"
            f"::: {{.fallacy-pitfall}}\n"
            f"**Pitfall:** *[Exact pitfall statement in italics]*\n\n"
            f"[Rigorous systems analysis explaining the subtle failure mode and the architectural mitigation.]\n"
            f":::\n\n"
            "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
        )
    else:
        # Summary and Takeaways (Volume 1 Canonical Pattern)
        task_parts.append(
            f"### Task: Author Chapter {manifest.number} Summary, Takeaways, and Chapter Connection\n"
            f"Target Depth: Comprehensive chapter synthesis and takeaways.\n\n"
            f"**Heading & Anchor:** `## Summary {{#sec-vol3-{manifest.slug}-summary}}`\n\n"
            f"**Exact Specification from Master Outline V3:**\n{manifest.summary_raw}\n\n"
            f"**REQUIRED STRUCTURE (VOLUME 1 CANONICAL PATTERN):**\n"
            f"1. Return to the governing question hook from the opening Purpose section, answering it decisively with the chapter's conceptual findings.\n"
            f"2. `::: {{.callout-takeaways title=\"Core Systems Principles of [Chapter Title]\"}}` containing 4–6 bold-lead takeaways summarizing the durable engineering laws established in this chapter.\n"
            f"3. Post-takeaway synthesis paragraph connecting the takeaways into a cohesive architectural principle.\n"
            f"4. `::: {{.callout-chapter-connection title=\"[Chapter Connection Title]\"}}` containing the conceptual bridge to the next chapter.\n\n"
            "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
        )

    task_instruction = "".join(task_parts)

    tier_1 = manifest.tier_1_guidance if manifest.tier_1_guidance else TIER_1_SYSTEMS_ENGINE
    return f"{tier_1}\n\n{tier_2}\n\n{tier_3}\n\n{tier_4}\n\n{task_instruction}"


# ==============================================================================
# 4. DETERMINISTIC PRE-MERGE REVIEW GATES
# ==============================================================================

@dataclass
class GateResult:
    passed: bool
    gate_name: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def run_review_gates(content: str, step_info: Dict[str, Any], is_sec1: bool = False) -> List[GateResult]:
    """Execute deterministic pre-merge review gates on generated text."""
    results: List[GateResult] = []
    lines = content.splitlines()
    words = len(content.split())

    # Gate 1: gate_minimum_length
    min_words = 150 if step_info.get("step_index") == 0 else 400
    if words < min_words:
        results.append(GateResult(
            passed=False,
            gate_name="gate_minimum_length",
            message=f"FAILED: Output contains only {words} words (minimum required: {min_words}).",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_minimum_length",
            message=f"PASSED: Length check ({words} words).",
        ))

    # Gate 2: gate_heading_anchor
    expected_anchor = step_info.get("heading_anchor", "")
    if expected_anchor and not any(expected_anchor in line for line in lines[:5]):
        # Check if anchor tag exists anywhere in first 5 lines
        anchor_tag_m = re.search(r"\{#(sec-[^}]+)\}", expected_anchor)
        if anchor_tag_m and not any(anchor_tag_m.group(1) in line for line in lines[:5]):
            results.append(GateResult(
                passed=False,
                gate_name="gate_heading_anchor",
                message=f"FAILED: Expected heading anchor '{expected_anchor}' not found in first 5 lines.",
            ))
        else:
            results.append(GateResult(
                passed=True,
                gate_name="gate_heading_anchor",
                message="PASSED: Heading and anchor found.",
            ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_heading_anchor",
            message="PASSED: Heading and anchor check.",
        ))

    # Gate 3: gate_section1_unbroken_narrative (Section .1 Invariant)
    if is_sec1:
        subheadings = [l for l in lines if l.strip().startswith("### ")]
        bullet_lines = [l for l in lines if re.match(r"^\s*[-*]\s+", l) and "::: {" not in l and not l.strip().startswith("- **Heading")]
        # In Section .1, we strictly forbid ### subsections and narrative bulleted lists
        if subheadings:
            results.append(GateResult(
                passed=False,
                gate_name="gate_section1_unbroken_narrative",
                message=f"FAILED: Section .1 contains {len(subheadings)} '###' subheadings. Must be an unbroken narrative stage-setter.",
                details={"subheadings": subheadings[:3]},
            ))
        elif len(bullet_lines) > 6:
            results.append(GateResult(
                passed=False,
                gate_name="gate_section1_unbroken_narrative",
                message=f"FAILED: Section .1 contains {len(bullet_lines)} bulleted lines. Expository narrative must be continuous unbroken prose.",
            ))
        else:
            results.append(GateResult(
                passed=True,
                gate_name="gate_section1_unbroken_narrative",
                message="PASSED: Section .1 satisfies unbroken narrative invariant (0 '###', continuous prose).",
            ))

    # Gate 4: gate_subsections (Section .2+ Structural Invariant)
    if not is_sec1 and step_info.get("step_index", 0) not in [0, 98, 99]:
        subheadings = [l for l in lines if l.strip().startswith("### ")]
        if len(subheadings) < 2:
            results.append(GateResult(
                passed=False,
                gate_name="gate_subsections",
                message=f"WARNING: Section contains only {len(subheadings)} '###' subsections (expected 2–4).",
            ))
        else:
            results.append(GateResult(
                passed=True,
                gate_name="gate_subsections",
                message=f"PASSED: Found {len(subheadings)} clean '###' subsections.",
            ))

    # Gate 5: gate_anti_anthropomorphism
    anthro_patterns = [
        (r"\bthe model thinks\b", "'the model thinks'"),
        (r"\bthe model decides\b", "'the model decides'"),
        (r"\bthe agent realizes\b", "'the agent realizes'"),
        (r"\bthe model gets confused\b", "'the model gets confused'"),
        (r"\blet's think step by step\b", "'let's think step by step'"),
    ]
    anthro_found = []
    for pat, desc in anthro_patterns:
        if re.search(pat, content, re.IGNORECASE):
            anthro_found.append(desc)
    if anthro_found:
        results.append(GateResult(
            passed=False,
            gate_name="gate_anti_anthropomorphism",
            message=f"FAILED: Anthropomorphic phrasing detected: {anthro_found}.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_anti_anthropomorphism",
            message="PASSED: No anthropomorphic phrasing found.",
        ))

    # Gate 6: gate_no_micro_instructions (Silicon Metaphor Purity)
    token_silicon_m = re.search(r"\b(tokens? (are|as|is) (micro-instructions?|opcodes?))\b", content, re.IGNORECASE)
    if token_silicon_m:
        results.append(GateResult(
            passed=False,
            gate_name="gate_no_micro_instructions",
            message=f"FAILED: Found banned token-silicon metaphor: '{token_silicon_m.group(1)}'. Tokens are data symbols and gather addresses, not opcodes.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_no_micro_instructions",
            message="PASSED: No false token-silicon metaphors found.",
        ))

    # Gate 7: gate_status_envelope_drift (Status Enum Locking)
    drifted_patterns = [
        (r"\b(`?Incomplete`?)\b", "Drifted status name: 'Incomplete' (must be TRUNCATED)"),
        (r"\b(`?Refusal`?)\b", "Drifted status name: 'Refusal' (must be REFUSED)"),
        (r"\b(`?Failed`?)\b", "Drifted status name: 'Failed' (must be TRANSPORT_FAILURE)"),
    ]
    status_drifts = []
    if any(k in content.lower() for k in ["status envelope", "invocation envelope", "invocationstatus"]):
        for pat, desc in drifted_patterns:
            m = re.findall(pat, content)
            if m:
                status_drifts.append(f"{desc}: {m[:3]}")
    if status_drifts:
        results.append(GateResult(
            passed=False,
            gate_name="gate_status_envelope_drift",
            message=f"FAILED: Found drifted status envelope names: {status_drifts}. Use strictly COMPLETED, TRUNCATED, REFUSED, TRANSPORT_FAILURE.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_status_envelope_drift",
            message="PASSED: Status envelope uses locked enumeration symbols.",
        ))

    # Gate 8: gate_negative_scope (Prevent Downstream Architectural Leaks)
    chapter_num = str(step_info.get("chapter_number", "")).zfill(2)
    leaks_found = []
    chapter_forbidden_rules = {
        "01": [
            (r"\b(PagedAttention|virtual memory block tables?|swapping to host DRAM)\b", "KV-Cache Hierarchy leak (belongs to Chapter 05)"),
            (r"\b(MCTS|Monte Carlo Tree Search|Process Reward Models?|PRMs?)\b", "Deliberation search leak (belongs to Chapter 03)"),
        ],
        "02": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
            (r"\b(PagedAttention|virtual memory block tables?|swapping to host DRAM)\b", "KV-Cache Hierarchy leak (belongs to Chapter 05)"),
            (r"\b(MCTS|Monte Carlo Tree Search|Process Reward Models?|PRMs?)\b", "Deliberation search leak (belongs to Chapter 03)"),
        ],
    }

    rules = chapter_forbidden_rules.get(chapter_num, [])
    for pat, reason in rules:
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            leaks_found.append(f"{reason}: {matches[:3]}")

    sec_num = step_info.get("section_num", "")
    if sec_num == "2.3" and re.search(r"\b(Roofline|arithmetic intensity\s*=\s*2/P)\b", content, re.IGNORECASE):
        leaks_found.append("Roofline derivation leak in Sec 2.3 (belongs to Section 2.4)")

    if leaks_found:
        results.append(GateResult(
            passed=False,
            gate_name="gate_negative_scope",
            message=f"FAILED: Found premature scope leaks / negative scope violations: {leaks_found}.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_negative_scope",
            message="PASSED: No premature downstream scope leaks detected.",
        ))

    # Gate 9: gate_quarto_syntax
    fence_count = content.count(":::")
    even_fences = (fence_count % 2 == 0)
    unbalanced_math = (content.count("$$") % 2 != 0)
    if not even_fences or unbalanced_math:
        results.append(GateResult(
            passed=False,
            gate_name="gate_quarto_syntax",
            message=f"FAILED: Syntax imbalance detected (::: count = {fence_count}, $$ count = {content.count('$$')}).",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_quarto_syntax",
            message="PASSED: Quarto syntax balanced.",
        ))

    # Gate 10: gate_anti_recap
    first_non_header = ""
    for l in lines:
        s = l.strip()
        if s and not s.startswith("#") and not s.startswith(":::") and not s.startswith("<!--"):
            first_non_header = s
            break
    recap_m = re.match(r"^(in the previous section|as we saw in|in section \d+\.\d+|recall from section)", first_non_header, re.IGNORECASE)
    if recap_m:
        results.append(GateResult(
            passed=False,
            gate_name="gate_anti_recap",
            message=f"FAILED: Section opens with retrospective recap opener: '{recap_m.group(0)}'. Engage immediately with technical mechanics.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_anti_recap",
            message="PASSED: Direct opening engagement.",
        ))

    return results


# ==============================================================================
# 5. LLM EXECUTION & REPAIR ENGINE
# ==============================================================================

def execute_llm_call(
    prompt: str,
    backend: str = "agy",
    model: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Execute an LLM call via the specified backend CLI."""
    if backend == "mock":
        heading = re.search(r"##\s+([^\n]+)", prompt)
        heading_title = heading.group(1) if heading else "Generated Section"
        return (
            f"## {heading_title}\n\n"
            "This is mock systems prose verifying the multi-call pipeline and review gates. "
            "The model proposal must be escrowed in host memory before environmental execution.\n\n"
            "### Mechanics and Systems Invariants\n\n"
            "The foundation model is an unprivileged coprocessor evaluating conditional distributions under zero ambient authority. "
            "Invariant closure is enforced by external deterministic test suites with explicit exit codes.\n\n"
            "$$T_{\\text{call}} = T_{\\text{queue}} + T_{\\text{prefill}} + \\sum_{t=1}^K T_{\\text{decode}, t}$$\n\n"
            "This establishes the systems contract."
        )

    if backend == "agy":
        cmd = ["agy", "-p", prompt, "--disable-slash-commands", "--dangerously-skip-permissions", "--print-timeout", f"{timeout}s"]
        if model:
            cmd.extend(["--model", model])
    elif backend == "claude":
        cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
        if model:
            cmd.extend(["--model", model])
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{backend} failed (code {proc.returncode}): {proc.stderr}")

    out = proc.stdout.strip()
    # Strip CLI notices and thinking tags
    out = re.sub(r"^Ignoring \d+ permissions\.allow entries[^\n]*\n?", "", out).strip()
    out = re.sub(r"Separately, the claude\.ai [^\n]* connector[^\n]*\n?", "", out).strip()
    out = re.sub(r"<thinking>.*?</thinking>", "", out, flags=re.DOTALL).strip()
    # Strip markdown wrappers if present
    if out.startswith("```markdown") and out.endswith("```"):
        out = out[len("```markdown"): -3].strip()
    elif out.startswith("```qmd") and out.endswith("```"):
        out = out[len("```qmd"): -3].strip()
    elif out.startswith("```") and out.endswith("```"):
        out = out[3:-3].strip()

    return out


def generate_with_review_and_repair(
    prompt: str,
    step_info: Dict[str, Any],
    is_sec1: bool = False,
    backend: str = "agy",
    model: Optional[str] = None,
    max_repairs: int = 2,
) -> Tuple[str, List[GateResult]]:
    """Generate section content, run review gates, and execute targeted repairs if critical gates fail."""
    content = execute_llm_call(prompt, backend=backend, model=model)
    gate_results = run_review_gates(content, step_info, is_sec1=is_sec1)

    critical_failures = [
        g for g in gate_results
        if not g.passed and g.gate_name in [
            "gate_minimum_length",
            "gate_section1_unbroken_narrative",
            "gate_anti_anthropomorphism",
            "gate_no_micro_instructions",
            "gate_status_envelope_drift",
            "gate_negative_scope",
            "gate_quarto_syntax",
            "gate_anti_recap",
        ]
    ]

    repair_count = 0
    while critical_failures and repair_count < max_repairs and backend != "mock":
        repair_count += 1
        print(f"    [Gate Check] Critical failure detected ({len(critical_failures)} issues). Attempting automated repair {repair_count}/{max_repairs}...")
        error_msgs = "\n".join(f"- {g.gate_name}: {g.message}" for g in critical_failures)

        repair_prompt = (
            f"{TIER_1_SYSTEMS_ENGINE}\n\n"
            "================================================================================\n"
            "MANDATORY REVISION / REPAIR TASK\n"
            "================================================================================\n"
            "Your previous draft failed our strict textbook review gates with the following errors:\n"
            f"{error_msgs}\n\n"
            "CRITICAL INSTRUCTIONS FOR REPAIR:\n"
            "1. If 'gate_section1_unbroken_narrative' failed: Section .1 must have ZERO `###` subsections and ZERO bulleted lists in the narrative! Write continuous, flowing academic prose.\n"
            "2. If 'gate_anti_anthropomorphism' failed: Remove phrasing like 'model thinks', 'model decides', or 'agent gets confused'.\n"
            "3. If 'gate_status_envelope_drift' failed: Use strictly COMPLETED, TRUNCATED, REFUSED, TRANSPORT_FAILURE.\n"
            "4. If 'gate_no_micro_instructions' failed: Eliminate any phrasing calling tokens 'micro-instructions' or 'opcodes'.\n"
            "5. If 'gate_anti_recap' failed: Ensure the very first sentence engages directly with the technical mechanics.\n"
            "6. If 'gate_quarto_syntax' failed: Ensure all ::: fences and $$ math tags are perfectly balanced.\n"
            "7. If 'gate_negative_scope' failed: Remove any leaked downstream concepts.\n\n"
            "Here is the draft that must be revised:\n"
            f"\"\"\"\n{content}\n\"\"\"\n\n"
            "Output ONLY the corrected Quarto markdown text. No backtick code fences wrapping the entire response."
        )

        content = execute_llm_call(repair_prompt, backend=backend, model=model)
        gate_results = run_review_gates(content, step_info, is_sec1=is_sec1)
        critical_failures = [
            g for g in gate_results
            if not g.passed and g.gate_name in [
                "gate_minimum_length",
                "gate_section1_unbroken_narrative",
                "gate_anti_anthropomorphism",
                "gate_no_micro_instructions",
                "gate_status_envelope_drift",
                "gate_negative_scope",
                "gate_quarto_syntax",
                "gate_anti_recap",
            ]
        ]

    return content, gate_results


# ==============================================================================
# 6. CONTEXT EXTRACTION & ACCUMULATION
# ==============================================================================

def extract_section_context(content: str) -> Dict[str, Any]:
    """Extract terminal bridge (~200 words) and active notation symbols from generated text."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and not p.strip().startswith("#")]

    terminal_bridge = ""
    if paragraphs:
        if len(paragraphs) >= 2:
            last_two = paragraphs[-2] + "\n\n" + paragraphs[-1]
            words = last_two.split()
            terminal_bridge = " ".join(words[-220:]) if len(words) > 220 else last_two
        else:
            words = paragraphs[-1].split()
            terminal_bridge = " ".join(words[-200:]) if len(words) > 200 else paragraphs[-1]

    inline_symbols = re.findall(r"\$([a-zA-Z_\\{}^0-9]+)\$", content)
    unique_symbols = list(dict.fromkeys(s for s in inline_symbols if len(s) <= 25))[:8]
    anchors = re.findall(r"\{#(sec-[^}]+|fig-[^}]+|tbl-[^}]+)\}", content)

    return {
        "terminal_bridge": terminal_bridge,
        "active_symbols": unique_symbols,
        "anchors": anchors,
    }


# ==============================================================================
# 7. DETERMINISTIC STEP 0 FRONTMATTER GENERATOR
# ==============================================================================

def generate_frontmatter_step0(manifest: ChapterManifest) -> str:
    """Generate Step 0 frontmatter deterministically from the V3 outline manifest."""
    num = manifest.number
    slug = manifest.slug
    title = manifest.title

    anchor = f"#sec-vol3-intro" if num == "01" else f"#sec-vol3-{slug}"
    alias_line = "\n[]{#sec-vol3-introduction}\n" if num == "01" else ""
    stack_macro = STACK_MAP.get(num, r"\mlagentstack{0}{0}{0}{0}{0}{100}")

    # Format the Purpose paragraph
    # Strip any markdown header or bullet from purpose if present
    clean_purpose = manifest.purpose.strip()
    # Remove "**The Core Question:** ..." if present since we print it separately
    clean_purpose = re.sub(r"\*\*The Core Question:\*\*\s*\*.*?\*\s*", "", clean_purpose).strip()
    clean_purpose = re.sub(r"\*\*Why It Matters:\*\*\s*", "", clean_purpose).strip()
    clean_purpose = clean_purpose.strip("*").strip()

    objs_formatted = "\n".join(f"- {o}" for o in manifest.objectives)

    frontmatter_text = f"""# {title} {{{anchor}}}
{alias_line}
::: {{layout-narrow}}
::: {{.column-margin}}

\\chapterminitoc

:::

::: {{.content-visible when-format="pdf"}}
\\noindent
![](images/png/cover_{slug}_blueprint_labeled_print.png){{fig-alt="Blueprint for {title}."}}
:::

::: {{.content-visible unless-format="pdf"}}
![](images/webp/cover_{slug}_blueprint_labeled.webp){{fig-alt="Blueprint for {title}."}}
:::

:::

## Purpose {{.unnumbered .unlisted}}

\\begin{{marginfigure}}
{stack_macro}
\\end{{marginfigure}}

_{manifest.governing_question}_

{clean_purpose}

::: {{.content-visible when-format="pdf"}}

\\newpage

:::

::: {{.callout-learning-objectives}}

{objs_formatted}

:::
"""
    return frontmatter_text.strip() + "\n"


# ==============================================================================
# 8. WORKSPACE & STEP MANAGEMENT
# ==============================================================================

def get_chapter_dir(chapter_num: str, slug: Optional[str] = None, variant: Optional[str] = None) -> Path:
    ch_num = chapter_num.zfill(2)
    s = slug or SLUG_MAP.get(ch_num, f"chapter_{ch_num}")
    if variant:
        return DRAFTS_ROOT / f"ch{ch_num}_{s}_{variant}"
    return DRAFTS_ROOT / f"ch{ch_num}_{s}"


def init_chapter_workspace(manifest: ChapterManifest, ch_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Initialize isolated draft workspace directory for a chapter."""
    if ch_dir is None:
        ch_dir = get_chapter_dir(manifest.number, manifest.slug)
    ch_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = ch_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    manifest_file = ch_dir / "manifest.json"
    if not manifest_file.exists() or force:
        manifest_file.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    state_file = ch_dir / "state.json"
    if not state_file.exists() or force:
        steps = []
        # Step 0: Frontmatter (Purpose & Objectives)
        steps.append({
            "step_index": 0,
            "id": "frontmatter",
            "title": "Frontmatter: Purpose & Learning Objectives",
            "file": "00_frontmatter.qmd",
            "budget_target": 350,
            "budget_range": [300, 450],
            "status": "pending",
            "word_count": 0,
            "completed_at": None,
        })
        # Step 1..N: Body Sections
        for idx, sec in enumerate(manifest.sections, start=1):
            file_name = f"{idx:02d}_sec_{sec.section_num.replace('.', '_')}.qmd"
            steps.append({
                "step_index": idx,
                "id": f"sec_{sec.section_num}",
                "section_num": sec.section_num,
                "title": f"Section {sec.section_num}: {sec.title}",
                "file": file_name,
                "budget_target": sec.budget_target,
                "budget_range": list(sec.budget_range),
                "heading_anchor": sec.heading_anchor,
                "status": "pending",
                "word_count": 0,
                "completed_at": None,
            })
        # Step N+1: Fallacies & Pitfalls
        steps.append({
            "step_index": len(manifest.sections) + 1,
            "id": "fallacies_pitfalls",
            "title": "Fallacies and Pitfalls",
            "file": "98_fallacies_pitfalls.qmd",
            "budget_target": 900,
            "budget_range": [700, 1100],
            "status": "pending",
            "word_count": 0,
            "completed_at": None,
        })
        # Step N+2: Summary & Connection
        steps.append({
            "step_index": len(manifest.sections) + 2,
            "id": "summary_connection",
            "title": "Summary & Chapter Connection",
            "file": "99_summary_connection.qmd",
            "budget_target": 600,
            "budget_range": [450, 750],
            "status": "pending",
            "word_count": 0,
            "completed_at": None,
        })

        state = {
            "chapter_number": manifest.number,
            "title": manifest.title,
            "slug": manifest.slug,
            "status": "pending",
            "current_step_index": 0,
            "total_steps": len(steps),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "steps": steps,
            "active_symbols": ["$M$", "$K$", "$S_{\\max}$", "$T_{\\max}$", "$A=0$"],
            "terminal_bridge": "",
            "cumulative_ledger": [],
        }
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    acc_file = ch_dir / "context_accumulator.md"
    if not acc_file.exists() or force:
        initial_acc = (
            f"# Context Accumulator: Chapter {manifest.number} - {manifest.title}\n\n"
            f"**Governing Systems Question:** {manifest.governing_question}\n\n"
            f"**Core Takeaway:** {manifest.core_takeaway}\n\n"
            "## Running Narrative & Symbols\n"
        )
        acc_file.write_text(initial_acc, encoding="utf-8")

    return ch_dir


def load_state(ch_dir: Path) -> Dict[str, Any]:
    state = json.loads((ch_dir / "state.json").read_text(encoding="utf-8"))
    state.setdefault("cumulative_ledger", [])
    sections_dir = ch_dir / "sections"
    updated = False
    for step in state.get("steps", []):
        target_file = sections_dir / step["file"]
        if target_file.exists():
            content = target_file.read_text(encoding="utf-8").strip()
            words = len(content.split())
            if words > 150 and step.get("status") != "completed" and "This is mock systems prose" not in content:
                step["status"] = "completed"
                step["word_count"] = words
                step["completed_at"] = datetime.now().isoformat()
                updated = True
            elif words < 150 and step.get("status") == "completed":
                step["status"] = "pending"
                step["word_count"] = 0
                step["completed_at"] = None
                updated = True
        else:
            if step.get("status") == "completed":
                step["status"] = "pending"
                step["word_count"] = 0
                step["completed_at"] = None
                updated = True
    if updated:
        save_state(ch_dir, state)
    return state


def save_state(ch_dir: Path, state: Dict[str, Any]) -> None:
    state["last_updated"] = datetime.now().isoformat()
    (ch_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


# ==============================================================================
# 9. EXECUTION PIPELINE (STEP, SEQUENTIAL, ASSEMBLY)
# ==============================================================================

def execute_step(
    ch_dir: Path,
    step_idx: int,
    backend: str = "agy",
    model: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Execute a single section step with review gates and context accumulation."""
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))
    state = load_state(ch_dir)
    sections_dir = ch_dir / "sections"

    if step_idx < 0 or step_idx >= len(state["steps"]):
        print(f"Error: Step index {step_idx} out of range (0..{len(state['steps'])-1})")
        return False

    step_info = state["steps"][step_idx]
    step_info["chapter_number"] = manifest.number
    if 1 <= step_idx <= len(manifest.sections):
        sec = manifest.sections[step_idx - 1]
        step_info["section_num"] = sec.section_num
        step_info["negative_scope"] = sec.negative_scope
    target_file = sections_dir / step_info["file"]
    is_sec1 = (step_idx == 1)

    if step_info["status"] == "completed" and target_file.exists() and not force:
        print(f"  [Step {step_idx}] Already completed: {step_info['title']} ({target_file.name})")
        return True

    # Step 0 is generated deterministically from manifest
    if step_idx == 0:
        print(f"  [Step 0] Generating deterministic Frontmatter...")
        content = generate_frontmatter_step0(manifest)
        target_file.write_text(content, encoding="utf-8")
        words = len(content.split())
        step_info["word_count"] = words
        step_info["status"] = "completed"
        step_info["completed_at"] = datetime.now().isoformat()
        save_state(ch_dir, state)
        print(f"  [Step 0] COMPLETED: {words} words -> {target_file.name}")
        return True

    # Identify forward handoff contract
    next_step_contract = None
    if step_idx < len(manifest.sections):
        next_sec = manifest.sections[step_idx]
        next_step_contract = {
            "title": f"Section {next_sec.section_num}: {next_sec.title}",
            "heading_anchor": next_sec.heading_anchor,
            "key_point": next_sec.key_point,
        }
    elif step_idx == len(manifest.sections):
        next_step_contract = {
            "title": "Fallacies and Pitfalls",
            "heading_anchor": f"## Fallacies and Pitfalls {{#sec-vol3-{manifest.slug}-fallacies}}",
            "key_point": "Refuting common systems misconceptions regarding foundation models in production.",
        }
    elif step_idx == len(manifest.sections) + 1:
        next_step_contract = {
            "title": "Summary & Chapter Connection",
            "heading_anchor": f"## Summary {{#sec-vol3-{manifest.slug}-summary}}",
            "key_point": "Authoritative architectural synthesis and forward handoff to the next chapter.",
        }

    upstream_bridge = state.get("terminal_bridge", "")
    active_symbols = state.get("active_symbols", [])
    cumulative_ledger = state.get("cumulative_ledger", [])

    prompt = compose_step_prompt(
        manifest=manifest,
        step_info=step_info,
        upstream_terminal_bridge=upstream_bridge,
        active_symbols=active_symbols,
        next_step_contract=next_step_contract,
        cumulative_ledger=cumulative_ledger,
    )

    if dry_run:
        prompt_file = ch_dir / f"prompt_step_{step_idx:02d}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        print(f"  [DRY RUN Step {step_idx}] Wrote prompt to {prompt_file.name}")
        return True

    print(f"  [Step {step_idx}] Generating: {step_info['title']} (target: {step_info['budget_target']}w)...")
    try:
        content, gate_results = generate_with_review_and_repair(
            prompt=prompt,
            step_info=step_info,
            is_sec1=is_sec1,
            backend=backend,
            model=model,
        )
    except Exception as e:
        print(f"  [Step {step_idx}] ERROR during generation: {e}")
        step_info["status"] = "failed"
        step_info["error"] = str(e)
        save_state(ch_dir, state)
        return False

    # Save content
    target_file.write_text(content, encoding="utf-8")
    words = len(content.split())
    step_info["word_count"] = words
    step_info["status"] = "completed"
    step_info["completed_at"] = datetime.now().isoformat()
    step_info["gate_summary"] = {g.gate_name: g.passed for g in gate_results}
    if "error" in step_info:
        del step_info["error"]

    # Extract context and update state
    ctx = extract_section_context(content)
    if ctx["terminal_bridge"]:
        state["terminal_bridge"] = ctx["terminal_bridge"]
    for sym in ctx["active_symbols"]:
        if sym not in state["active_symbols"]:
            state["active_symbols"].append(sym)

    # Record architectural invariant into cumulative ledger
    if 1 <= step_idx <= len(manifest.sections):
        sec_obj = manifest.sections[step_idx - 1]
        inv_summary = f"Section {sec_obj.section_num} ({sec_obj.title}): {sec_obj.key_point}"
    else:
        inv_summary = f"{step_info['title']}"
    if "cumulative_ledger" not in state:
        state["cumulative_ledger"] = []
    if inv_summary not in state["cumulative_ledger"]:
        state["cumulative_ledger"].append(inv_summary)

    state["current_step_index"] = step_idx + 1
    if all(s["status"] == "completed" for s in state["steps"]):
        state["status"] = "completed"
    save_state(ch_dir, state)

    # Append to context accumulator
    acc_entry = (
        f"\n### Completed Step {step_idx}: {step_info['title']}\n"
        f"- **File:** `{step_info['file']}` | **Word Count:** {words:,} words\n"
        f"- **Active Symbols Added:** " + (", ".join(f"`{s}`" for s in ctx["active_symbols"]) if ctx["active_symbols"] else "None") + "\n"
        f"- **Terminal Bridge Handed Off:**\n  _{ctx['terminal_bridge'][:180]}..._\n"
    )
    with (ch_dir / "context_accumulator.md").open("a", encoding="utf-8") as f:
        f.write(acc_entry)

    # Print review gate status
    all_passed = all(g.passed for g in gate_results)
    gate_status_str = "All Gates Passed" if all_passed else "Warnings Present"
    print(f"  [Step {step_idx}] COMPLETED: {words:,} words -> {target_file.name} [{gate_status_str}]")
    for g in gate_results:
        if not g.passed:
            print(f"      ⚠️  {g.gate_name}: {g.message}")

    return True


def run_full_chapter(
    chapter_num: str,
    outline_path: Path = MASTER_OUTLINE_V3_PATH,
    variant: Optional[str] = None,
    backend: str = "agy",
    model: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Execute all steps sequentially from Step 0 to Step N+2, then assemble."""
    manifest = parse_chapter_v3(chapter_num, outline_path=outline_path)
    ch_dir = get_chapter_dir(manifest.number, manifest.slug, variant=variant)
    init_chapter_workspace(manifest, ch_dir=ch_dir, force=force)
    state = load_state(ch_dir)

    print("\n" + "=" * 80)
    print(f"LAUNCHING V3 MULTI-CALL CHAPTER DRAFTING: Chapter {manifest.number} - {manifest.title}")
    print(f"Workspace: {ch_dir.relative_to(REPO_ROOT)}")
    print(f"Total Steps: {len(state['steps'])} | Backend: {backend} | Model: {model or 'default'}")
    print("=" * 80 + "\n")

    for step in state["steps"]:
        idx = step["step_index"]
        if step["status"] == "completed" and not force:
            continue
        success = execute_step(
            ch_dir=ch_dir,
            step_idx=idx,
            backend=backend,
            model=model,
            force=force,
            dry_run=dry_run,
        )
        if not success:
            print(f"\nExecution halted at step {idx} due to error.")
            return False

    if not dry_run:
        assemble_chapter(ch_dir)
        validate_assembled_chapter(ch_dir)

    print(f"\nMulti-call generation finished successfully for Chapter {chapter_num}!\n")
    return True


def assemble_chapter(ch_dir: Path) -> Path:
    """Assemble all section files in sequence into assembled_draft.qmd and stage to books/vol3."""
    state = load_state(ch_dir)
    sections_dir = ch_dir / "sections"
    assembled_file = ch_dir / "assembled_draft.qmd"

    blocks: List[str] = []
    for step in state["steps"]:
        sec_file = sections_dir / step["file"]
        if sec_file.exists():
            blocks.append(sec_file.read_text(encoding="utf-8").strip())
        else:
            blocks.append(f"<!-- MISSING STEP: {step['title']} ({step['file']}) -->")

    full_text = "\n\n".join(blocks) + "\n"
    # Normalize image paths to be relative to the chapter directory
    full_text = re.sub(r"books/vol3/[^/]+/images/", "images/", full_text)
    assembled_file.write_text(full_text, encoding="utf-8")
    words = len(full_text.split())
    print(f"\n[Assembly] Compiled {len(blocks)} parts into {assembled_file.name} ({words:,} total words)")

    # Stage to canonical book directory: books/vol3/<ch_num>_<slug>/<ch_num>_<slug>.qmd
    ch_num = str(state.get("chapter_number", "01")).zfill(2)
    slug = state.get("slug", "introduction")
    target_ch_dir = REPO_ROOT / "books" / "vol3" / f"{ch_num}_{slug}"
    target_ch_dir.mkdir(parents=True, exist_ok=True)
    target_qmd = target_ch_dir / f"{ch_num}_{slug}.qmd"
    target_qmd.write_text(full_text, encoding="utf-8")
    print(f"[Assembly] Staged assembled draft to {target_qmd.relative_to(REPO_ROOT)}")

    return assembled_file


def validate_assembled_chapter(ch_dir: Path) -> Dict[str, Any]:
    """Audit the assembled chapter draft against Quarto, style, and budget invariants."""
    assembled_file = ch_dir / "assembled_draft.qmd"
    if not assembled_file.exists():
        assemble_chapter(ch_dir)

    text = assembled_file.read_text(encoding="utf-8")
    words = len(text.split())
    issues: List[str] = []

    # Check for Section .1 subsections in assembled text
    sec1_match = re.search(r"## [^\n]+sec-vol3-[^\n]+-(?:operational-incident|role)\}(.*?)(?=## |\Z)", text, re.DOTALL)
    if sec1_match:
        sec1_text = sec1_match.group(1)
        subheadings = re.findall(r"^###\s+.*", sec1_text, re.MULTILINE)
        if subheadings:
            issues.append(f"CRITICAL: Section .1 contains {len(subheadings)} '###' subheadings in assembled draft! Section .1 must be an unbroken narrative stage-setter.")

    # Check balanced div fences
    div_count = text.count(":::")
    if div_count % 2 != 0:
        issues.append(f"Unbalanced Quarto div fences (::: count = {div_count})")

    # Check contractions
    contractions = re.findall(r"\b(can't|won't|don't|doesn't|isn't|aren't|haven't|hasn't|it's)\b", text, re.IGNORECASE)
    if contractions:
        issues.append(f"Found {len(contractions)} contractions (American formal systems prose requires non-contracted forms).")

    report = {
        "chapter_dir": str(ch_dir),
        "total_words": words,
        "div_fences": div_count,
        "contractions_count": len(contractions),
        "issues": issues,
        "status": "PASSED" if not issues else "WARNINGS",
    }
    (ch_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Validation] {report['status']}: {words:,} total words, {len(issues)} issues reported.")
    return report


# ==============================================================================
# 10. CLI ENTRY POINT
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-Call Chapter Generation Engine from Master Outline V3."
    )
    parser.add_argument("--chapter", type=str, required=True, help="Chapter number to generate (e.g. 01, 02).")
    parser.add_argument("--step", type=int, default=None, help="Execute only a single step index (0..N+2).")
    parser.add_argument("--run-all", action="store_true", help="Execute all steps sequentially to completion.")
    parser.add_argument("--backend", choices=["agy", "claude", "mock"], default="agy", help="LLM backend.")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash-high", help="Model override (default: gemini-3.8-flash-high).")
    parser.add_argument("--outline", type=str, default=str(MASTER_OUTLINE_V3_PATH), help="Path to master outline file.")
    parser.add_argument("--variant", type=str, default=None, help="Variant label (e.g. v3, v4).")
    parser.add_argument("--assemble", action="store_true", help="Assemble existing section drafts into assembled_draft.qmd.")
    parser.add_argument("--validate", action="store_true", help="Audit existing assembled draft.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing steps.")
    parser.add_argument("--dry-run", action="store_true", help="Write step prompts to disk without invoking LLM.")
    parser.add_argument("--status", action="store_true", help="Show status of the chapter draft workspace.")

    args = parser.parse_args()
    num = args.chapter.zfill(2)
    outline_path = Path(args.outline).resolve()

    try:
        manifest = parse_chapter_v3(num, outline_path=outline_path)
    except Exception as e:
        print(f"Error parsing chapter {num} from {outline_path}: {e}")
        return 1

    ch_dir = get_chapter_dir(num, manifest.slug, variant=args.variant)

    if args.status:
        if not ch_dir.exists():
            print(f"Chapter {num} ({manifest.slug}) workspace uninitialized at {ch_dir}.")
            return 0
        state = load_state(ch_dir)
        print("\n" + "=" * 70)
        print(f"Chapter {num}: {state['title']} Status")
        print("=" * 70)
        print(f"Status: {state['status'].upper()} | Current Step: {state['current_step_index']}/{state['total_steps']}")
        print("-" * 70)
        for s in state["steps"]:
            status_icon = "✅" if s["status"] == "completed" else ("⏳" if s["status"] == "in_progress" else "⚪")
            print(f" {status_icon} Step {s['step_index']:02d}: {s['title']:<45} ({s['word_count']}w)")
        print("=" * 70 + "\n")
        return 0

    if args.assemble:
        if not ch_dir.exists():
            print(f"Workspace {ch_dir} does not exist. Run generation first.")
            return 1
        assemble_chapter(ch_dir)
        validate_assembled_chapter(ch_dir)
        return 0

    if args.validate:
        if not ch_dir.exists():
            print(f"Workspace {ch_dir} does not exist.")
            return 1
        validate_assembled_chapter(ch_dir)
        return 0

    if args.step is not None:
        init_chapter_workspace(manifest, ch_dir=ch_dir, force=args.force)
        success = execute_step(
            ch_dir=ch_dir,
            step_idx=args.step,
            backend=args.backend,
            model=args.model,
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0 if success else 1

    if args.run_all:
        success = run_full_chapter(
            chapter_num=num,
            outline_path=outline_path,
            variant=args.variant,
            backend=args.backend,
            model=args.model,
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0 if success else 1

    # Default: show help and status
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
