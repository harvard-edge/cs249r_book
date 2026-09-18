#!/usr/bin/env python3
"""
Generate Chapter from V2 Master Outline (Multi-Call Systems Engineering Engine)

Authoritative chapter generation pipeline for Volume III: The Stochastic Computer.
Operates squarely in the systems engineering middle ground (not NLP linguistics, not CPU silicon roleplay).

Architecture:
- Sequential in-chapter multi-call execution (one model call per section).
- 4-Tier Bounded Context Staging Schema (Tier 1: Systems Engine, Tier 2: Chapter Blueprint, Tier 3: Upstream Interface, Tier 4: Forward Handoff).
- Enforced Calibrated Word Budgets per section type.
- Section .1 Unbroken Narrative Invariant (Zero ### subsections, 4-beat stage-setting arc).
- 6 Deterministic Pre-Merge Review Gates with automated repair loop.
- Dynamic Active Symbol Registry and Terminal Bridge context accumulator.
- Clean-slate isolation under `books/vol3/drafts/chXX_<slug>/`.
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
MASTER_OUTLINE_V2_PATH = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V2.md"
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
    hook: str
    points: str
    visuals: str
    literature: str
    causal_bridge: str
    negative_scope: str = ""


@dataclass
class ChapterManifest:
    number: str
    title: str
    slug: str
    word_budget_target: int
    canonical_scenario: str
    core_takeaway: str
    governing_question: str
    curricular_role: str
    purpose: str
    objectives: List[str]
    sections: List[SectionSpec]
    fallacies_raw: str
    summary_raw: str
    curricular_compass: str = ""
    raw_outline_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChapterManifest:
        sections = [SectionSpec(**s) for s in data.get("sections", [])]
        data_copy = dict(data)
        data_copy["sections"] = sections
        return cls(**data_copy)


# ==============================================================================
# 2. V2 OUTLINE PARSER
# ==============================================================================

def parse_chapter_v2(chapter_num: str, outline_path: Path = MASTER_OUTLINE_V2_PATH) -> ChapterManifest:
    """Parse a single chapter's blueprint from MASTER_TEXTBOOK_OUTLINE_V2.md."""
    if not outline_path.exists():
        raise FileNotFoundError(f"V2 Master outline not found at: {outline_path}")

    content = outline_path.read_text(encoding="utf-8")
    target_num = chapter_num.zfill(2)

    # Split by chapter header
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

    # Parse metadata
    lines = matched_part.strip().splitlines()
    header_m = re.search(r"### Chapter (\d+):\s*(.*)", lines[0])
    num = header_m.group(1).zfill(2) if header_m else target_num
    title = header_m.group(2).strip() if header_m else f"Chapter {num}"
    slug = SLUG_MAP.get(num, f"chapter_{num}")

    budget_m = re.search(r"-\s+\*\*Word Budget Target:\*\*\s*([\d,]+)", matched_part)
    budget_target = int(budget_m.group(1).replace(",", "")) if budget_m else 12000

    scenario_m = re.search(r"-\s+\*\*Canonical Systems Scenario:\*\*\s*(.+?)(?=\n-|\Z)", matched_part, re.DOTALL)
    canonical_scenario = scenario_m.group(1).strip() if scenario_m else ""
    if not canonical_scenario:
        # Fallback to Section 8 of the Master Outline
        sec8_m = re.search(r"\*\s+\*\*Chapter\s+" + re.escape(num) + r":\*\*\s*(.+?)(?=\n\*|\n###|\Z)", content)
        if sec8_m:
            canonical_scenario = sec8_m.group(1).strip()

    takeaway_m = re.search(r"-\s+\*\*Core Takeaway:\*\*\s*(.+?)(?=\n-|\Z)", matched_part, re.DOTALL)
    core_takeaway = takeaway_m.group(1).strip() if takeaway_m else ""

    question_m = re.search(r"-\s+\*\*Governing Systems Question:\*\*\s*(.+?)(?=\n-|\n####|\Z)", matched_part, re.DOTALL)
    governing_question = question_m.group(1).strip() if question_m else ""

    curricular_m = re.search(r"-\s+\*\*Curricular Role.*?:\*\*\s*(.+?)(?=\n-|\n####|\Z)", matched_part, re.DOTALL)
    curricular_role = curricular_m.group(1).strip() if curricular_m else ""

    compass_m = re.search(r"#### The Curricular Compass.*?\n```(.*?)```", matched_part, re.DOTALL)
    curricular_compass = compass_m.group(1).strip() if compass_m else ""

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
        # Clean title of any bracketed budget: e.g. "Title [Budget: 1,000 words ...]"
        clean_sec_title = re.sub(r"\s*\[Budget:.*?\]", "", raw_sec_title).strip()

        # Structure defaults
        is_sec1 = sec_num.endswith(".1")
        if is_sec1:
            budget_tgt = 1200
            budget_rng = (900, 1600)
            struct_inv = "Use 2–3 clean, scannable ### subheadings to structure the conceptual contrast, the systems boundary, and the failure modes. Conclude with an unbroken prose bridge directly posing the first mechanistic question for Section .2."
        else:
            budget_tgt = 1500
            budget_rng = (1100, 2000)
            struct_inv = "2–3 ### subsections. Analytical mechanics, physical equations, concrete metrics, ending with explicit Causal Bridge."

        heading_anchor_m = re.search(r"-\s+\*\*Heading & Anchor:\*\*\s*`?([^`\n]+)`?", b)
        heading_anchor = heading_anchor_m.group(1).strip() if heading_anchor_m else f"## {clean_sec_title}"

        struct_m = re.search(r"-\s+\*\*Structural Invariant:\*\*\s*(.+?)(?=\n-|\Z)", b, re.DOTALL)
        if struct_m:
            struct_inv = struct_m.group(1).strip()

        key_point_m = re.search(r"-\s+\*\*The Single Key Point:\*\*\s*(.+?)(?=\n-|\Z)", b, re.DOTALL)
        key_point = key_point_m.group(1).strip() if key_point_m else ""

        hook_m = re.search(r"-\s+\*\*Concrete Systems Hook:\*\*\s*(.+?)(?=\n-\s+\*\*(?:Points|What to Cover)|\Z)", b, re.DOTALL)
        hook = hook_m.group(1).strip() if hook_m else ""

        points_m = re.search(
            r"-\s+\*\*(?:What to Cover|Points to explain).*?:\*\*\s*(.+?)(?=\n-\s+\*\*What NOT to Cover|\n-\s+\*\*Visuals|\n-\s+\*\*Seminal|\n-\s+\*\*Causal|\Z)",
            b,
            re.DOTALL,
        )
        points = points_m.group(1).strip() if points_m else ""

        neg_scope_m = re.search(
            r"-\s+\*\*What NOT to Cover.*?:\*\*\s*(.+?)(?=\n-\s+\*\*Visuals|\n-\s+\*\*Seminal|\n-\s+\*\*Causal|\Z)",
            b,
            re.DOTALL,
        )
        neg_scope = neg_scope_m.group(1).strip() if neg_scope_m else ""

        visuals_m = re.search(r"-\s+\*\*Visuals & Tables:\*\*\s*(.+?)(?=\n- \*\*Seminal|\n- \*\*Causal|\Z)", b, re.DOTALL)
        visuals = visuals_m.group(1).strip() if visuals_m else ""

        lit_m = re.search(r"-\s+\*\*Seminal Literature:\*\*\s*(.+?)(?=\n- \*\*Causal|\Z)", b, re.DOTALL)
        lit = lit_m.group(1).strip() if lit_m else ""

        bridge_m = re.search(r"-\s+\*\*Causal Bridge.*:\*\*\s*(.+?)(?=\n-|\Z)", b, re.DOTALL)
        bridge = bridge_m.group(1).strip() if bridge_m else ""

        sections.append(
            SectionSpec(
                section_num=sec_num,
                title=clean_sec_title,
                heading_anchor=heading_anchor,
                budget_target=budget_tgt,
                budget_range=budget_rng,
                structural_invariant=struct_inv,
                key_point=key_point,
                hook=hook,
                points=points,
                visuals=visuals,
                literature=lit,
                causal_bridge=bridge,
                negative_scope=neg_scope,
            )
        )

    fallacies_m = re.search(r"#### Fallacies and Pitfalls.*?\n+(.+?)(?=#### Summary|\Z)", matched_part, re.DOTALL)
    fallacies_raw = fallacies_m.group(1).strip() if fallacies_m else ""

    summary_m = re.search(r"#### Summary.*?\n+(.+?)(?=\n### Chapter|\Z)", matched_part, re.DOTALL)
    summary_raw = summary_m.group(1).strip() if summary_m else ""

    return ChapterManifest(
        number=num,
        title=title,
        slug=slug,
        word_budget_target=budget_target,
        canonical_scenario=canonical_scenario,
        core_takeaway=core_takeaway,
        governing_question=governing_question,
        curricular_role=curricular_role,
        purpose=purpose,
        objectives=objs,
        sections=sections,
        fallacies_raw=fallacies_raw,
        summary_raw=summary_raw,
        curricular_compass=curricular_compass,
        raw_outline_text=matched_part,
    )


# ==============================================================================
# 3. 4-TIER BOUNDED CONTEXT PROMPT COMPOSER
# ==============================================================================

TIER_1_SYSTEMS_ENGINE = r"""You are an author of the premier senior-undergraduate and introductory graduate computer systems textbook:
'The Stochastic Computer: Agentic Machine Learning Systems' (Volume III).
Your voice, pedagogical clarity, and architectural rigor mirror Saltzer & Kaashoek's 'Principles of Computer System Design'
and Hennessy & Patterson's 'Computer Architecture: A Quantitative Approach'.

================================================================================
TIER 1: THE SYSTEMS ENGINEERING STANCE & UNIVERSAL ARCHITECTURAL INVARIANTS
================================================================================

1. AUDIENCE & PEDAGOGICAL STANCE:
   - Target Reader: Senior CS/CE undergraduate or Master's student. Assume an average, solid student aiming to become a professional AI systems engineer.
   - Presumed Student Background: Familiar with Python, standard data structures (trees, hash maps, queues), basic Operating Systems (processes, virtual memory, syscalls, filesystems, concurrency, client-server RPCs), and introductory Deep Learning (tensors, matrix multiplication, softmax, loss functions, transformers).
   - Knowledge Gaps to Bridge: Do NOT assume the student is a superstar genius or already knows low-level GPU microarchitecture, proprietary cluster interconnect fabrics, or esoteric chip physics. Every advanced systems concept MUST be built up from familiar software systems intuition first.
   - Pedagogical Mission: Train the reader to think like a systems architect who builds robust, dependable software runtimes around non-deterministic foundation models.
   - Speak AUTHENTIC MACHINE LEARNING SYSTEMS LANGUAGE: Large Language Model (LLM), tokens, Byte-Pair Encoding (BPE), embedding tables, autoregressive decode loop, logits, softmax, KV cache, prefill, decode, inference engines (vLLM, TensorRT-LLM), and host agent runtimes.
   - Do NOT force a literal silicon straightjacket: do not pretend an attention head is an x86 ALU, that tokens are "machine opcodes", or that prompt text is an "instruction register".

2. STANDALONE STAGE-SETTER (.1) VS. REAL MECHANICS (.2 ONWARDS):
   - Section .1 (The Standalone Stage-Setter & Chapter Introduction):
     * Serves as the conceptual foundation and stage-setter for the entire chapter.
     * Accessible and engaging: frames the governing systems dilemma, contrasts classical deterministic systems with unprivileged stochastic generation, establishes the 3-tier boundary (Host Runtime, Inference Service, Neural Core), and introduces fail-stop vs. fail-plausible execution.
     * Uses 2–3 clean, scannable `###` subheadings to provide clear cognitive road signs.
     * Concludes with an unbroken prose bridge directly posing the first mechanistic systems question for Section .2.
   - Section .2 Onwards (The Real Technical Mechanics):
     * This is where the concrete engineering and mathematical mechanics live!
     * Each subsequent section dives deep into ONE specific subsystem, interface, or cost model.
     * Follows the 4-step scaffolding ladder: Dilemma -> Systems Intuition -> Concrete Code/Artifact -> Grounded Math.

3. HIGHER-LEVEL ARCHITECTURAL INTUITION (THE SEVEN SUBSYSTEM LENSES):
   Use high-level, intuitive computer systems analogies that students immediately grasp:
   - Part I (Chapters 2–3 — Processing Element & Deliberation): Treat the LLM as an unprivileged coprocessor / processing element evaluating probability distributions under zero ambient authority. Deliberation (search, Best-of-N, MCTS) is test-time compute allocation and speculative branch exploration.
   - Part II (Chapters 4–6 — Working Sets, Physical Memory & Storage): Treat the context window and KV cache as a multi-tier memory hierarchy: prompt context is the logical working set; GPU KV cache is physical page-table managed device memory (PagedAttention); external stores (vector DBs, files, Git) are persistent secondary storage with cache invalidation and freshness challenges.
   - Part III (Chapters 7–8 — Tool Actuation & Sandboxing): Treat tools as peripheral devices and mediated system calls (syscalls). The unprivileged model proposes an RPC payload; the host runtime acts as reference monitor, validating parameters and dispatching execution into isolated sandboxes (containers, microVMs, seccomp-bpf, namespaces).
   - Part IV (Chapters 9–11 — The Agent Operating System): Treat the host runtime as an operating system kernel managing long-horizon processes via Agent Control Blocks (ACBs), priority scheduling, Write-Ahead Logging (WAL) state persistence, and distributed saga compensation.
   - Part V (Chapters 12–14 — The Policy Compiler): Treat adaptation (SFT & RLVR) as an offline policy compiler specializing model weights from verified traces using test suites as reward oracles.
   - Part VI (Chapters 15–17 — Distributed Fleets & Operations): Treat multi-agent systems as distributed concurrent processes with communication costs and contention; distributed tracing as telemetry; and capacity economics as cost per accepted deliverable.
   - Part VII (Chapter 18 — System Synthesis): Synthesize all subsystems into an end-to-end verifiable computer.

3. THE 4-STEP PEDAGOGICAL SCAFFOLDING LADDER:
   Every section should flow naturally through this progression:
   1. The Governing Systems Dilemma / Observable Failure: Open with a concrete systems problem an engineer or student can picture.
   2. The Systems / Architecture Intuition (The Rosetta Stone): Translate the ML mechanism into concepts known from classical computer systems.
   3. The Concrete Artifact: Anchor in a typed Python @dataclass, C struct, AST trace, or clear comparison table BEFORE formulas.
   4. Grounded Systems Math & Physical Provenance: Walk through equations step-by-step with explicit units. Adhere to The Law of Constant Provenance: never drop an unexplained constant without physical explanation in text.

4. FREEDOM, FLEXIBILITY, AND PEDAGOGICAL BREATHING ROOM:
   - NO RIGID WORD CAPS: Let the text breathe naturally without arbitrary word ceilings or floors. Substantive engineering depth takes precedence over word targets.
   - STRUCTURAL HIERARCHY: Every section (including Section .1) should use 2–3 clean, scannable `###` subheadings to provide cognitive road signs.
   - Rich visual, tabular, and worked example callouts (`::: {.callout-note title="Worked Example..."}`).

5. DUAL-TOPOLOGY EXECUTION TIERS:
   Maintain a strict boundary between execution tiers:
   - **Tier 1 (Host Agent OS):** User-space CPU runtime. Owns task orchestration, compiles grammars (DFAs/PDAs), stages prompt context, manages tool actuation, evaluates deterministic test suites in isolated sandboxes, and verifies exit codes.
   - **Tier 2 (Inference Service Daemon):** GPU-side serving runtime (e.g., vLLM, TensorRT-LLM). Manages PagedAttention KV cache page tables, schedules batched forward passes, executes fused CUDA kernels, and performs decode-time logit masking in device memory so vocabulary vectors ($100\text{k+}$ floats) never transit PCIe.
   - **Tier 3 (Neural Core):** Parameter tensors $\Theta$. Executes tensor contractions on accelerator Tensor Cores under zero ambient authority.
   - *Never blur the boundary:* The Host OS compiles the grammar $\mathcal{G}$, but the Inference Engine executes logit masking on GPU device memory.

6. NORMALIZED STATUS ENVELOPE & ENUM LOCKING:
   All processor invocations return a normalized 4-outcome status envelope. Always use these exact uppercase enumeration names:
   - `COMPLETED`: Normal termination via stop token or delimiter within token budget.
   - `TRUNCATED`: Severed at step ceiling $K_{\max}$ without emission of stop token. Payload must be quarantined.
   - `REFUSED`: Rejected by internal alignment classifier.
   - `TRANSPORT_FAILURE`: Communication or socket error (timeout, connection reset, OOM crash).
   ❌ BANNED DRIFT: Never use informal synonyms (`Incomplete`, `Failed`, `Refusal`). Always use `COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`.

7. PHYSICAL HARDWARE BASELINE RIGOR:
   - Base all quantitative examples on physical reality. An unquantized 70B FP16 model ($140\text{ GB}$) exceeds a single $80\text{ GB}$ H100 GPU and requires either Tensor Parallelism across 2 GPUs ($TP=2$ over NVLink, aggregate bandwidth $6{,}700\text{ GB/s}$), FP8 quantization ($70\text{ GB}$ on a single H100 at $3{,}350\text{ GB/s} \to 20.9\text{ ms}$ decode floor), or next-gen hardware (B200 with $192\text{ GB}$ at $8{,}000\text{ GB/s}$).
   - When deriving decode operational arithmetic intensity, include the active KV cache traffic $\text{Mem}_{\text{KV}}(M+t)$ shuttled from HBM alongside weights $\Theta$:
     $$I_{\text{decode}}(t) = \frac{2|\Theta|}{P|\Theta| + \text{Mem}_{\text{KV}}(M+t)}$$

8. SECURITY & SPECULATIVE EXECUTION RIGOR:
   - State mutation follows **Speculative Execution with Sandboxed Verification and Rollback**: candidate sequences are staged in escrow, tested in disposable sandboxes, and committed only upon zero exit codes.
   - Defense against **Oracle Poisoning**: All test fixtures, linters, and verification assertions must be mounted as strictly read-only (`ro`), kept in isolated out-of-tree directories, and executed with sanitized `PYTHONPATH` so untrusted code proposals cannot mutate their own verifiers.

9. STRICT SCOPE OWNERSHIP & ARCHITECTURAL MODULARITY:
   - **Every section answers exactly ONE clean, distinct architectural question** in the subsystem's journey without anticipating or cannibalizing adjacent sections.
   - **Strict Subsystem Separation of Concerns:**
     * Dedicated Hardware Cost Home: Roofline models, GEMM vs GEMV arithmetic intensity, memory bus shuttles, and hardware balance tables belong exclusively in dedicated cost/capacity sections (never in data representation, execution loop, or protocol sections).
     * Protocol & Interface Boundary: Formal request contracts, typed parameters, and status envelopes belong strictly in invocation interface sections.
     * Verification & Epistemic Boundary: Speculative execution, sandbox isolation, and defense against Oracle Poisoning belong strictly in verification/sandboxing sections.
     * Memory Hierarchy: Virtual memory paging (PagedAttention), fragmentation, and prefix caching belong in context memory sections.
   - **Adhere Strictly to Assigned Negative Scopes:** Follow all negative constraints in the section specification. If a concept belongs to an adjacent section or chapter, bridge to it cleanly without pre-empting its mathematical derivations or mechanisms.

10. LINGUISTIC AND PEDAGOGICAL INTEGRITY:
   - **American English Spelling:** Use -ize, -or, center, defense, meter, labeled, modeled throughout.
   - ❌ **NEVER use anthropomorphic phrasing:** "the model thinks", "the model decides", "the agent realizes its mistake", "the model gets confused".
     -> *Write:* "the neural core evaluates", "the runtime detects an invariant violation", "attentional errors cascade across historical context".
   - ❌ **NEVER use conversational or prompt tropes:** "When chatting with the user", "Prompt tricks like 'let's think step by step'", "Having agents debate in a meeting".
     -> *Write:* "In the client-runtime interaction", "Allocating test-time compute to expand candidate exploration", "Executing concurrent stochastic worker processes across distributed queues".
   - ❌ **NEVER claim stochastic self-verification:** "The model checks its own answer to ensure correctness."
     -> *Write:* "Stochastic self-evaluation cannot close invariants ($P < 1.0$); invariant closure requires external deterministic execution (compilers, test runners, exit codes)."
   - ❌ **NEVER call tokens 'micro-instructions' or 'opcodes':** Tokens are discrete integer data symbols and embedding gather addresses; the neural core has no opcode decoder or register file.

11. VOLUME 1 FIVE-TYPE FOOTNOTE TAXONOMY WITH INDEXED TERMS:
   When introducing foundational systems concepts, specialized hardware terminology, or historical context, provide rigorous footnotes formatted with indexed terms:
   - Footnote format: `[^fn-label]` in text, followed by:
     `[^fn-label]: **Term** (Etymology/Category): Definition and systems context. \index{Term}\index{Category!Subterm}`
   - Categories:
     1. Clarification (term definition, scope boundary)
     2. Historical / Context (etymology, early systems origins, seminal paper context)
     3. Caveat / Edge Case (subtle hardware or kernel behavior, exception cases)
     4. Cross-Reference (connections across chapters or volumes)
     5. Physical / Mathematical Detail (minor derivation step, unit conversion)
   - Every footnote MUST include at least one `\index{...}` command on its core technical term for the book index.
"""


def get_part_traps(chapter_num: str, outline_path: Path = MASTER_OUTLINE_V2_PATH) -> str:
    """Extract the Part-level Non-Systems Traps applicable to the given chapter from the outline."""
    if not outline_path.exists():
        return ""
    try:
        content = outline_path.read_text(encoding="utf-8")
        num = int(chapter_num)
        part_ranges = [
            ((2, 3), "Part I:"),
            ((4, 6), "Part II:"),
            ((7, 8), "Part III:"),
            ((9, 11), "Part IV:"),
            ((12, 14), "Part V:"),
            ((15, 17), "Part VI:"),
            ((18, 18), "Part VII:"),
        ]
        part_prefix = None
        for (start_c, end_c), prefix in part_ranges:
            if start_c <= num <= end_c:
                part_prefix = prefix
                break
        if not part_prefix:
            return ""

        part_match = re.search(r"###\s+" + re.escape(part_prefix) + r".*?(?=###\s+Part|\n##\s+|\Z)", content, re.DOTALL)
        if not part_match:
            return ""

        part_text = part_match.group(0)
        traps_match = re.search(r"-\s+\*\*❌ Non-Systems Traps.*?\*\*:\s*\n(.+?)(?=\n###|\n##|\Z)", part_text, re.DOTALL)
        if traps_match:
            return traps_match.group(1).strip()
    except Exception:
        pass
    return ""


# Modular registry of section-specific negative scope and focus directives
SECTION_NEGATIVE_SCOPES: Dict[str, str] = {
    "2.3": (
        "\n**NEGATIVE SCOPE INVARIANT (STRICTLY FORBIDDEN IN THIS SECTION):**\n"
        "- Do NOT derive the Hardware Roofline Model or arithmetic intensity ($I = 2/P$).\n"
        "- Do NOT include hardware balance comparison tables (H100 vs B200 vs M4 Max).\n"
        "- Do NOT derive matrix-matrix GEMM vs matrix-vector GEMV arithmetic intensity.\n"
        "-> REASON: Those hardware Roofline derivations and balance tables belong EXCLUSIVELY in Section 2.7 ('The Cost of an Invocation').\n"
        "- Focus this section STRICTLY on the sequential autoregressive control loop, causal serialization, token-by-token emission, KV cache state mutation, and temperature/sampling on the probability simplex.\n"
        "- Do NOT open with raw hardware acronym dumps (e.g. do not open with 'Staged in accelerator High Bandwidth Memory (HBM)...'). Lead with the sequential execution model and causal dependency chain.\n"
    ),
    "2.4": (
        "\n**KEY SYSTEMS REQUIREMENTS FOR THIS SECTION:**\n"
        "- Decouple likelihood from operational validity and execution safety.\n"
        "- Treat the foundation model as an unprivileged Byzantine execution unit proposing candidate strings under zero ambient authority.\n"
        "- Two-Phase Speculative Execution pattern: proposals are escrowed in an isolated buffer, validated externally, and applied only upon passing mechanical checks.\n"
        "- Defense against Oracle Poisoning: test suites and linters must be mounted read-only (`ro,noexec`) in isolated ephemeral sandboxes so candidate code cannot modify its own verifier.\n"
    ),
    "2.5": (
        "\n**CRITICAL NEGATIVE SCOPE & SECTION MISSION (DO NOT VIOLATE):**\n"
        "- Your SOLE systems mission in this section is the **Invocation Contract & Status Envelope**.\n"
        "- STRICTLY FORBIDDEN: Do NOT derive Key-Value cache memory formulas ($2LHd$). Do NOT do GPU HBM byte math. That was already covered in 2.2.\n"
        "- STRICTLY FORBIDDEN: Do NOT derive the Roofline model, GEMM vs GEMV arithmetic intensity, or memory bus shuttling formulas. That belongs exclusively in Section 2.7.\n"
        "- Focus 100% on the systems engineering interface:\n"
        "  1. The typed RPC request specification (context, ceilings $K_{\\max}, T_{\\max}$, schemas, stop delimiters).\n"
        "  2. The Normalized 4-Outcome Status Envelope (`COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`).\n"
        "  3. The systems principle that HTTP 200 OK != task completion.\n"
        "  4. Truncation Quarantining (why partial diffs or unclosed JSON must be quarantined and discarded to prevent corrupting downstream tools/compilers).\n"
    ),
    "2.6": (
        "\n**CRITICAL NEGATIVE SCOPE & SECTION MISSION (DO NOT VIOLATE):**\n"
        "- Your SOLE systems mission in this section is **Constraining the Output Surface via Decode-Time Logit Masking**.\n"
        "- STRICTLY FORBIDDEN: Do NOT derive the Roofline model, GEMM vs GEMV arithmetic intensity, memory bus shuttling, or hardware tables. That belongs exclusively in Section 2.7.\n"
        "- STRICTLY FORBIDDEN: Do NOT derive KV cache memory formulas ($2LHd$). That was covered in 2.2.\n"
        "- Focus 100% on the systems engineering sweet spot:\n"
        "  1. How the host runtime compiles formal schemas (JSON Schema, regex, context-free grammars) into Finite State Machines (DFAs for flat/regex schemas, Pushdown Automata for nested CFGs).\n"
        "  2. The Logit Masking Mechanism: at step $t$, the FSM identifies the subset of valid continuation tokens $\\mathcal{V}_{\\text{valid}} \\subset \\mathcal{V}_0$ and sets illegal logits $z_i = -\\infty$ before categorical sampling.\n"
        "  3. The Syntactic Divide: structural validity != semantic correctness. A grammar mask guarantees matching brackets, quotes, and valid types; it provides zero guarantee that a file path exists, that a function name is valid, or that an edit is safe.\n"
        "  4. The Hazard of Schema Forcing: if the schema does not include uncertainty or error variants, logit masking forces the core to hallucinate plausible values to satisfy the grammar.\n"
        "  5. Visual / Diagram: Include an ASCII architectural schematic and reference @fig-grammar-constrained-decoding (`grammar_constrained_decoding_fsm.svg`).\n"
    ),
    "2.7": (
        "\n**KEY SYSTEMS REQUIREMENTS FOR THIS SECTION (THE DEDICATED HARDWARE COST HOME):**\n"
        "- Derive the Roofline Model for prefill ($I_{\\text{prefill}} \\approx 2M/P$) and decode ($I_{\\text{decode}} \\approx 2/P$).\n"
        "- Contrast parallel compute-bound GEMM with serialized memory-bandwidth-bound GEMV.\n"
        "- The Memory Shuttle Problem: shuttling model weights $\\Theta$ across the memory bus for every single token.\n"
        "- The single-agent $B=1$ serialization wall: why agent loops cannot amortize memory transfers with multi-tenant batching.\n"
        "- Reference Hardware Balance Table (@tbl-hardware-balance): NVIDIA H100 SXM5, NVIDIA B200, Apple M4 Max.\n"
        "- Prefix caching (Prompt caching): Radix tree of KV-cache blocks, cache hits, and Amdahl's Law speedups.\n"
        "- End-to-end latency decomposition: $T_{\\text{call}} = T_{\\text{queue}} + T_{\\text{transport}} + T_{\\text{prefill}} + \\sum_t T_{\\text{decode}, t} + T_{\\text{validate}}$.\n"
    ),
    "2.8": (
        "\n**KEY SYSTEMS REQUIREMENTS FOR THIS SECTION:**\n"
        "- Controlled systems benchmarking of invocation interfaces under equal resource budgets.\n"
        "- Compare Free-form Text vs Strict Grammar-Constrained JSON vs Decomposed Two-Stage Probes.\n"
        "- Metrics: structural validity rate, token consumption ($M, K$), TTFT, total latency, truncation rate, and verified task acceptance rate.\n"
        "- Interface Comparison Table (@tbl-vol3-interface-evaluation).\n"
    ),
}


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
        "  - Part I: The Stochastic Processor & Deliberation (Ch 02: Processor Core [Current Foundation], Ch 03: Test-Time Deliberation)\n"
        "  - Part II: Context Memory & Storage Hierarchy (Ch 04: Working Memory, Ch 05: Attention State/PagedAttention, Ch 06: Persistent Storage)\n"
        "  - Part III: Tool Actuation, I/O Peripherals & Sandboxing (Ch 07: Peripherals/Syscalls, Ch 08: MicroVM Sandboxing)\n"
        "  - Part IV: The Agent Operating System (Ch 09: Control Plane/ACB, Ch 10: State WAL, Ch 11: Sagas/Transactions)\n"
        "  - Part V: The Policy Compiler (Ch 12: Data Flywheel, Ch 13: Distillation, Ch 14: RLVR)\n"
        "  - Part VI: Distributed Fleets & Operations (Ch 15: Concurrency/Graphs, Ch 16: Telemetry/Tracing, Ch 17: Capacity Economics)\n"
        "  - Part VII: System Synthesis (Ch 18: Capstone End-to-End Upgrade)"
    )
    section_roadmap = "\n".join(
        f"  - Section {s.section_num}: {s.title} (Key Point: {s.key_point})"
        for s in manifest.sections
    )
    compass_block = ""
    if manifest.curricular_compass:
        compass_block = (
            "CURRICULAR COMPASS (WHERE WE ARE IN THE 18 CHAPTERS):\n"
            f"```\n{manifest.curricular_compass}\n```\n\n"
        )
    tier_2 = (
        "================================================================================\n"
        "TIER 2: CHAPTER GROUNDING BLUEPRINT (THE BIG PICTURE)\n"
        "================================================================================\n"
        f"{macro_map}\n\n"
        f"{compass_block}"
        f"ACTIVE CHAPTER POSITION IN THE MACHINE ARCHITECTURE:\n"
        f"- Chapter {manifest.number}: {manifest.title}\n"
        f"- Chapter Anchor: #sec-vol3-{manifest.slug}\n"
        f"- Curricular Role: {manifest.curricular_role}\n"
        f"- Governing Systems Question: {manifest.governing_question}\n"
        f"- Core Takeaway: {manifest.core_takeaway}\n"
        f"- Canonical Systems Scenario: {manifest.canonical_scenario}\n\n"
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

    if step_idx == 0:
        # Frontmatter
        task_parts.append(
            f"Draft the Chapter {manifest.number} Opening Frontmatter:\n"
            f"Target Budget: 250 words (Range: 200–300 words).\n\n"
            f"MANDATORY STRUCTURAL INVARIANT (THE SINGLE-PARAGRAPH PURPOSE LAW):\n"
            f"- Under `## Purpose {{.unnumbered .unlisted}}`, include `\\begin{{marginfigure}}\n\\mlagentstack{{0}}{{0}}{{0}}{{0}}{{0}}{{100}}\n\\end{{marginfigure}}`.\n"
            f"- Follow with the italicized governing question `_{manifest.governing_question}_`.\n"
            f"- Follow immediately with EXACTLY ONE single, cohesive, authoritative paragraph (150–220 words) expanding on: {manifest.purpose}.\n"
            f"- ZERO internal paragraph breaks. It must be an unbroken single paragraph defining the systems confrontation, the architectural role of the subsystem, and the governing invariant/trade-off.\n"
            f"- Follow with PDF conditional page break and `::: {{.callout-learning-objectives}}`.\n\n"
            f"Required Output Format:\n"
            f"# {manifest.title} {{#sec-vol3-{manifest.slug}}}\n\n"
            f"::: {{layout-narrow}}\n"
            f"::: {{.column-margin}}\n\n"
            f"\\chapterminitoc\n\n"
            f":::\n\n"
            f"::: {{.content-visible when-format=\"pdf\"}}\n"
            f"\\noindent\n"
            f"![](images/png/cover_{manifest.slug}_blueprint_labeled_print.png){{fig-alt=\"Blueprint for {manifest.title}.\"}}\n"
            f":::\n\n"
            f"::: {{.content-visible unless-format=\"pdf\"}}\n"
            f"![](images/webp/cover_{manifest.slug}_blueprint_labeled.webp){{fig-alt=\"Blueprint for {manifest.title}.\"}}\n"
            f":::\n\n"
            f":::\n\n"
            f"## Purpose {{.unnumbered .unlisted}}\n\n"
            f"\\begin{{marginfigure}}\n"
            f"\\mlagentstack{{0}}{{0}}{{0}}{{0}}{{0}}{{100}}\n"
            f"\\end{{marginfigure}}\n\n"
            f"_{manifest.governing_question}_\n\n"
            f"[Single unbroken 150-220 word purpose paragraph]\n\n"
            f"::: {{.content-visible when-format=\"pdf\"}}\n\n"
            f"\\newpage\n\n"
            f":::\n\n"
            f"::: {{.callout-learning-objectives}}\n\n"
            + "\n".join(f"- {o}" for o in manifest.objectives)
            + "\n\n:::\n\n"
            "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
        )
    elif 1 <= step_idx <= len(manifest.sections):
        sec = manifest.sections[step_idx - 1]
        is_sec1 = sec.section_num.endswith(".1")
        sec_specific_scope = sec.negative_scope or SECTION_NEGATIVE_SCOPES.get(sec.section_num, "")
        part_traps = get_part_traps(manifest.number)
        negative_parts = []
        if sec_specific_scope:
            negative_parts.append(f"**NEGATIVE SCOPE BOUNDARIES (STRICTLY FORBIDDEN IN THIS SECTION):**\n{sec_specific_scope.strip()}")
        if part_traps:
            negative_parts.append(f"**PART-LEVEL NON-SYSTEMS TRAPS (STRICTLY FORBIDDEN):**\n{part_traps}")
        negative_scope = ("\n\n" + "\n\n".join(negative_parts) + "\n") if negative_parts else ""

        if is_sec1:
            task_parts.append(
                f"### Task: Author Section {sec.section_num}: {sec.title}\n"
                f"Target Depth: Substantive and thorough exposition for senior CS/CE undergraduates. Let the text breathe naturally without arbitrary word caps.\n\n"
                f"**STRUCTURAL INVARIANT (SECTION .1 ROAD SIGNS):**\n"
                f"Use 2–3 clean, scannable `###` subheadings to structure the conceptual contrast, the systems boundary, and the failure modes.\n"
                f"  - Architectural Stage-Setting: Contrast deterministic classical systems with the unprivileged generative model operating under zero ambient authority.\n"
                f"  - Situate within The Stochastic Computer: define the functional boundary between the host agent runtime and the underlying execution engine.\n"
                f"  - Systems Confrontation: The candidate proposal disconnect, fail-plausible execution, resource ceilings ($K_{{\\max}}, T_{{\\max}}$), and external invariant closure.\n"
                f"  - Conclude with an unbroken prose bridge directly posing the first mechanistic question for Section {manifest.sections[1].section_num if len(manifest.sections) > 1 else 'X.2'}.\n\n"
                f"**Heading & Anchor:** `{sec.heading_anchor}`\n"
                f"**Single Key Point:** {sec.key_point}\n\n"
                f"{negative_scope}\n"
                f"**Points to explain (from Master Outline):**\n{sec.points}\n\n"
                f"**Visuals to reference:**\n{sec.visuals or 'None specified'}\n\n"
                f"**Causal Bridge to conclude with:**\n{sec.causal_bridge}\n\n"
                "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
            )
        else:
            task_parts.append(
                f"### Task: Author Section {sec.section_num}: {sec.title}\n"
                f"Target Depth: Substantive and thorough exposition for senior CS/CE undergraduates. Let the text breathe naturally without arbitrary word caps.\n\n"
                f"**STRUCTURAL INVARIANT:**\n"
                f"Must contain 2–3 clean `###` subsections. Unpack rigorous analytical mechanics, concrete data structures, and failure traces.\n\n"
                f"**Heading & Anchor:** `{sec.heading_anchor}`\n"
                f"**Single Key Point:** {sec.key_point}\n"
                f"**Concrete Systems Hook:**\n{sec.hook}\n\n"
                f"{negative_scope}\n"
                f"**Paragraph-by-Paragraph Technical Mechanics to Develop:**\n{sec.points}\n\n"
                f"**Visuals & Tables to Reference:**\n{sec.visuals or 'None specified'}\n\n"
                f"**Causal Bridge (Must conclude with):**\n{sec.causal_bridge}\n\n"
                "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
            )
    elif step_idx == len(manifest.sections) + 1:
        # Fallacies and Pitfalls
        task_parts.append(
            f"### Task: Author Chapter {manifest.number} Fallacies and Pitfalls\n"
            f"Target Depth: Rigorous and thorough analysis of 2 Fallacies and 2 Pitfalls.\n\n"
            f"**Heading & Anchor:** `## Fallacies and Pitfalls {{#sec-vol3-{manifest.slug}-fallacies}}`\n\n"
            f"**Outline Specification:**\n{manifest.fallacies_raw}\n\n"
            f"**REQUIRED STRUCTURE:**\n"
            f"Provide exactly 2 Fallacies and 2 Pitfalls. For each entry, format using standard Quarto div syntax with 3 distinct parts:\n\n"
            f"::: {{.fallacy-pitfall}}\n"
            f"**Fallacy:** *[Exact fallacy statement in italics]*\n\n"
            f"**Mechanism of Failure:** [Rigorous systems explanation citing physical trade-offs, memory bus bottlenecks, or failure modes.]\n\n"
            f"**Architectural Defense:** [Concrete systems remediation, verification gates, or runtime isolation patterns.]\n"
            f":::\n\n"
            f"::: {{.fallacy-pitfall}}\n"
            f"**Pitfall:** *[Exact pitfall statement in italics]*\n\n"
            f"**Mechanism of Failure:** [Rigorous systems explanation.]\n\n"
            f"**Architectural Defense:** [Concrete systems remediation.]\n"
            f":::\n\n"
            "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
        )
    else:
        # Summary and Takeaways (Volume 1 Canonical Pattern)
        task_parts.append(
            f"### Task: Author Chapter {manifest.number} Summary, Takeaways, and Chapter Connection\n"
            f"Target Depth: Comprehensive chapter synthesis and takeaways.\n\n"
            f"**Heading & Anchor:** `## Summary {{#sec-vol3-{manifest.slug}-summary}}`\n\n"
            f"**Outline Specification:**\n{manifest.summary_raw}\n\n"
            f"**REQUIRED STRUCTURE (VOLUME 1 CANONICAL PATTERN):**\n"
            f"1. Return to the governing question hook from the opening Purpose section, answering it decisively with the chapter's conceptual findings.\n"
            f"2. `::: {{.callout-takeaways title=\"Key Takeaways\"}}` containing 3–5 bold-lead takeaways summarizing the durable engineering laws established in this chapter.\n"
            f"3. Post-takeaway synthesis paragraph connecting the takeaways into a cohesive architectural principle.\n"
            f"4. `::: {{.callout-chapter-connection title=\"What's Next: Inference-Time Deliberation\"}}` containing the conceptual bridge to test-time search and deliberation without using clumsy meta-phrases like 'In Chapter 3'.\n\n"
            "Output ONLY the Quarto markdown text. No backtick code fences wrapping the entire response."
        )

    task_instruction = "".join(task_parts)

    # Full Chapter Road Map to orient the student and generator
    chapter_map_parts = [
        "================================================================================\n",
        f"FULL CHAPTER ROAD MAP: CHAPTER {manifest.number} ({manifest.title})\n",
        "================================================================================\n",
        f"Chapter Governing Question: {manifest.governing_question}\n",
        f"Chapter Core Takeaway: {manifest.core_takeaway}\n\n",
        "Progressive Section Path:\n",
    ]
    for s_idx, s in enumerate(manifest.sections, start=1):
        if s_idx == step_idx:
            chapter_map_parts.append(f"  👉 [CURRENT STEP {s_idx}] Section {s.section_num}: {s.title} ({s.key_point})\n")
        elif s_idx < step_idx:
            chapter_map_parts.append(f"  ✅ [COMPLETED STEP {s_idx}] Section {s.section_num}: {s.title}\n")
        else:
            chapter_map_parts.append(f"  ⏳ [UPCOMING STEP {s_idx}] Section {s.section_num}: {s.title}\n")
    chapter_map_parts.append("================================================================================\n\n")
    chapter_map = "".join(chapter_map_parts)

    outline_block = ""
    if manifest.raw_outline_text:
        outline_block = (
            "================================================================================\n"
            f"AUTHORITATIVE MASTER BLUEPRINT FOR CHAPTER {manifest.number}: {manifest.title}\n"
            "(SOURCE OF TRUTH: books/vol3/MASTER_TEXTBOOK_OUTLINE_V2.md)\n"
            "================================================================================\n"
            f"{manifest.raw_outline_text}\n\n"
            "================================================================================\n"
            "AUTHOR META-DIRECTIVE (READ AND ABSORB BEFORE AUTHORING):\n"
            "================================================================================\n"
            f"You have been provided with the complete, unabridged Master Blueprint for Chapter {manifest.number} above.\n"
            "Now that you understand the entire curricular arc, pedagogical progression, and architectural constraints:\n"
            "1. Read the corresponding section specification in the outline that is directly relevant to the section you are developing below.\n"
            "2. Ground the narrative in authentic systems engineering realities (formal interface contracts, typed parameter specifications, POSIX exit codes, tokenizer AST splits, physical memory bus bandwidth and arithmetic intensity, fail-plausible semantic corruption) rather than synthetic scenario scripts.\n"
            "3. Enforce all boilerplate directives: MIT Press tone, active voice, 5-type footnote taxonomy with indexed terms (`\\index{...}`), no anthropomorphism, zero ambient authority, and external invariant closure.\n"
            "4. Strictly obey all negative scope boundaries—do NOT cannibalize topics assigned to subsequent sections or chapters.\n"
            "================================================================================\n\n"
        )

    return f"{TIER_1_SYSTEMS_ENGINE}\n\n{chapter_map}{outline_block}{tier_2}\n\n{tier_3}\n\n{tier_4}\n\n{task_instruction}"


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
    """Execute all 6 deterministic pre-merge review gates on generated text."""
    results: List[GateResult] = []
    lines = content.splitlines()
    words = len(content.split())

    # Gate 0.1: gate_minimum_length (Undergeneration Guard)
    min_words = 150 if step_info.get("step_index") == 0 else 400
    if words < min_words:
        results.append(GateResult(
            passed=False,
            gate_name="gate_minimum_length",
            message=f"FAILED: Output severely truncated or empty ({words} words vs minimum {min_words} words).",
            details={"words": words, "min_words": min_words},
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_minimum_length",
            message=f"PASSED: Word count {words} satisfies minimum length {min_words}.",
        ))

    # Gate 0: gate_purpose_single_paragraph (For Step 0 Frontmatter)
    if step_info.get("step_index") == 0:
        purpose_m = re.search(r"## Purpose [^\n]+\n+(.*?)(?=::: \{\.callout-learning-objectives\}|\Z)", content, re.DOTALL)
        if purpose_m:
            p_text = purpose_m.group(1).strip()
            p_paras = [
                p.strip() for p in p_text.split("\n\n")
                if p.strip()
                and not (p.strip().startswith("_") and p.strip().endswith("_"))
                and not p.strip().startswith(("\\begin", "\\end", "\\newpage", ":::", "!["))
            ]
            if len(p_paras) > 1:
                results.append(GateResult(
                    passed=False,
                    gate_name="gate_purpose_single_paragraph",
                    message=f"FAILED: Purpose contains {len(p_paras)} prose paragraphs. The Single-Paragraph Purpose Law requires exactly ONE cohesive paragraph.",
                    details={"paragraphs": len(p_paras)},
                ))
            else:
                results.append(GateResult(
                    passed=True,
                    gate_name="gate_purpose_single_paragraph",
                    message="PASSED: Purpose is exactly one single paragraph.",
                ))
        else:
            results.append(GateResult(passed=True, gate_name="gate_purpose_single_paragraph", message="Purpose block not matched."))

    # Gate 1: gate_subsections (Structural Scaffolding)
    # Ensure clean section structure with 2–4 ### subsections to organize the narrative
    subsections = [line for line in lines if re.match(r"^###\s+", line)]
    if len(subsections) < 2 and words >= 800:
        results.append(GateResult(
            passed=True,
            gate_name="gate_subsections",
            message=f"NOTICE: Found {len(subsections)} '###' subsections. Recommended: 2–4 clean, scannable '###' subsections for structured readability.",
            details={"subsections": subsections},
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_subsections",
            message=f"PASSED: Found {len(subsections)} clean '###' subsections.",
        ))

    # Gate 2: gate_anti_anthropomorphism (Linguistic Purity)
    banned_patterns = [
        (r"\b(model|agent|processor|core)\s+(thinks|believes|decides|realizes|understands|knows|remembers|wants|intends|feels|gets confused)\b", "Banned anthropomorphic cognition verb"),
        (r"\b(let's think|step by step|chatting with|talks to the user|having a meeting)\b", "Banned conversational/prompt trope"),
        (r"\b(model can verify its own|self-verification guarantees)\b", "Banned claim of stochastic self-verification"),
    ]
    anthro_violations = []
    for pat, desc in banned_patterns:
        m = re.findall(pat, content, re.IGNORECASE)
        if m:
            anthro_violations.append(f"{desc}: {m[:3]}")

    if anthro_violations:
        results.append(GateResult(
            passed=False,
            gate_name="gate_anti_anthropomorphism",
            message=f"FAILED: Found {len(anthro_violations)} forbidden anthropomorphic phrasing instances.",
            details={"violations": anthro_violations},
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_anti_anthropomorphism",
            message="PASSED: Zero forbidden anthropomorphisms or conversational tropes found.",
        ))

    # Gate 2.5: gate_no_micro_instructions (Silicon Metaphor Purity)
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

    # Gate 2.7: gate_status_envelope_drift (Status Enum Locking)
    # Check for informal drifted names when referring to invocation status outcomes
    drifted_patterns = [
        (r"\b(`?Incomplete`?)\b", "Drifted status name: 'Incomplete' (must be TRUNCATED)"),
        (r"\b(`?Refusal`?)\b", "Drifted status name: 'Refusal' (must be REFUSED)"),
        (r"\b(`?Failed`?)\b", "Drifted status name: 'Failed' (must be TRANSPORT_FAILURE)"),
        (r"(?:status\s+|envelope\s+|S\s*=\s*|S\s*\\neq\s*|S\s*\\in\s*)`?\\?texttt\{Completed\}", "Titlecase status: 'Completed' (must be uppercase COMPLETED)"),
        (r"(?:status\s+|envelope\s+|S\s*=\s*|S\s*\\neq\s*|S\s*\\in\s*)`Completed`", "Titlecase status: '`Completed`' (must be uppercase `COMPLETED`)"),
        (r"(?:status\s+|envelope\s+|S\s*=\s*|S\s*\\neq\s*|S\s*\\in\s*)`?\\?texttt\{Truncated\}", "Titlecase status: 'Truncated' (must be uppercase TRUNCATED)"),
        (r"(?:status\s+|envelope\s+|S\s*=\s*|S\s*\\neq\s*|S\s*\\in\s*)`Truncated`", "Titlecase status: '`Truncated`' (must be uppercase `TRUNCATED`)"),
        (r"(?:status\s+|envelope\s+|S\s*=\s*|S\s*\\neq\s*|S\s*\\in\s*)`?\\?texttt\{Refused\}", "Titlecase status: 'Refused' (must be uppercase REFUSED)"),
        (r"(?:status\s+|envelope\s+|S\s*=\s*|S\s*\\neq\s*|S\s*\\in\s*)`Refused`", "Titlecase status: '`Refused`' (must be uppercase `REFUSED`)"),
        (r"\b`?TransportFailure`?\b", "PascalCase status: 'TransportFailure' (must be uppercase TRANSPORT_FAILURE)"),
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
            details={"drifts": status_drifts},
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_status_envelope_drift",
            message="PASSED: Status envelope uses locked enumeration symbols.",
        ))

    # Gate 2.8: gate_negative_scope (Prevent Downstream/Premature Architectural Leaks)
    chapter_num = str(step_info.get("chapter_number", "")).zfill(2)
    leaks_found = []

    chapter_forbidden_rules = {
        "02": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
            (r"\b(PagedAttention|virtual memory block tables?|swapping to host DRAM)\b", "KV-Cache Hierarchy leak (belongs to Chapter 05)"),
            (r"\b(MCTS|Monte Carlo Tree Search|Process Reward Models?|PRMs?)\b", "Deliberation search leak (belongs to Chapter 03)"),
        ],
        "03": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
            (r"\b(PagedAttention|virtual memory block tables?)\b", "KV-Cache Hierarchy leak (belongs to Chapter 05)"),
        ],
        "04": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
            (r"\b(PagedAttention|virtual memory block tables?)\b", "KV-Cache Hierarchy leak (belongs to Chapter 05)"),
        ],
        "05": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
        ],
        "06": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
        ],
        "07": [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (belongs to Chapter 08)"),
        ],
    }

    rules = chapter_forbidden_rules.get(chapter_num, [])
    for pat, reason in rules:
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            leaks_found.append(f"{reason}: {matches[:3]}")

    sec_num = step_info.get("section_num", "")
    if sec_num == "2.3" and re.search(r"\b(Roofline|arithmetic intensity\s*=\s*2/P)\b", content, re.IGNORECASE):
        leaks_found.append("Roofline derivation leak in Sec 2.3 (belongs to Section 2.7)")
    if sec_num == "2.5" and re.search(r"\b(2\s*L\s*H\s*d|Mem_\{?KV\}?)\b", content):
        leaks_found.append("KV memory formula leak in Sec 2.5 (covered in Section 2.2)")

    if leaks_found:
        results.append(GateResult(
            passed=False,
            gate_name="gate_negative_scope",
            message=f"FAILED: Found premature scope leaks / negative scope violations: {leaks_found}.",
            details={"leaks": leaks_found},
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_negative_scope",
            message="PASSED: No premature downstream scope leaks detected.",
        ))

    # Gate 3: gate_external_closure (Verification Boundary)
    # Check that correctness and invariants are attributed to external software
    has_external_closure = bool(re.search(
        r"\b(external|deterministic|compiler|linter|test runner|test suite|exit code|AST|type checker|sandbox|hypervisor|ACL)\b",
        content,
        re.IGNORECASE,
    ))
    if not has_external_closure and not is_sec1 and step_info["step_index"] not in [0, 99]:
        results.append(GateResult(
            passed=False,
            gate_name="gate_external_closure",
            message="WARNING: Section lacks explicit mention of external deterministic verification mechanisms.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_external_closure",
            message="PASSED: Mentions external deterministic verification.",
        ))

    # Gate 4: gate_systems_metrics (Physical Grounding Density)
    metric_matches = re.findall(
        r"\b(FLOP|TFLOP|HBM|DRAM|NVMe|bandwidth|TB/s|GB/s|MB/s|bytes/token|GEMM|GEMV|Roofline|latency|p50|p99|ms|\$M\$|\$K\$|\$T_|\$B=1\$|int32|BPE|KV cache|RPC|gRPC|IPC|socket|exit code|WEXITSTATUS|timeout|deadline|AST|schema|status envelope|quarantine|DFA|PDA|FSM|logit mask)\b",
        content,
    )
    if len(metric_matches) < 2 and step_info["step_index"] > 0:
        results.append(GateResult(
            passed=False,
            gate_name="gate_systems_metrics",
            message=f"WARNING: Low physical metrics density (only {len(metric_matches)} metrics matched).",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_systems_metrics",
            message=f"PASSED: High systems metric density ({len(metric_matches)} systems metrics matched).",
            details={"count": len(metric_matches)},
        ))

    # Gate 5: gate_quarto_crossref (Syntax Integrity)
    fence_count = content.count(":::")
    even_fences = (fence_count % 2 == 0)
    unbalanced_math = (content.count("$$") % 2 != 0)
    if not even_fences or unbalanced_math:
        results.append(GateResult(
            passed=False,
            gate_name="gate_quarto_crossref",
            message=f"FAILED: Syntax imbalance (::: count = {fence_count}, unbalanced $$ = {unbalanced_math}).",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_quarto_crossref",
            message="PASSED: Quarto div fences and math delimiters are balanced.",
        ))

    # Gate 6: gate_anti_recap (No Retrospective Openers)
    first_100_words = " ".join(content.split()[:100])
    recap_m = re.search(r"\b(in the previous section|as we discussed|in section \d+\.\d+|as seen previously|as covered earlier)\b", first_100_words, re.IGNORECASE)
    if recap_m:
        results.append(GateResult(
            passed=False,
            gate_name="gate_anti_recap",
            message=f"FAILED: Opening recap detected in first 100 words: '{recap_m.group(1)}'.",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_anti_recap",
            message="PASSED: Opening engages directly without retrospective recap.",
        ))

    # Gate 6.5: gate_no_opening_hardware_dump (Pedagogical Stance)
    # Ensure opening sentence does not lead with raw hardware silicon acronyms (HBM, PCIe, NVLink, SRAM)
    non_header_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    first_sentence = non_header_lines[0] if non_header_lines else ""
    first_80_chars = first_sentence[:80]
    hw_opening_m = re.search(r"\b(HBM|High Bandwidth Memory|PCIe|NVLink|SRAM)\b", first_80_chars, re.IGNORECASE)
    if hw_opening_m and step_info.get("step_index", 0) > 0:
        results.append(GateResult(
            passed=False,
            gate_name="gate_no_opening_hardware_dump",
            message=f"FAILED: Opening sentence leads with hardware silicon dump: '{hw_opening_m.group(1)}'. Lead with the conceptual systems architecture first!",
            details={"match": hw_opening_m.group(1), "first_sentence": first_sentence[:120]},
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_no_opening_hardware_dump",
            message="PASSED: Opening does not lead with premature hardware silicon dump.",
        ))

    # Gate 7: gate_word_budget (Budget Adherence)
    budget_rng = step_info.get("budget_range", (700, 1600))
    if words < budget_rng[0] * 0.75:
        results.append(GateResult(
            passed=False,
            gate_name="gate_word_budget",
            message=f"WARNING: Section under budget ({words} words vs target range {budget_rng[0]}–{budget_rng[1]}).",
        ))
    elif words > budget_rng[1] * 1.3:
        results.append(GateResult(
            passed=False,
            gate_name="gate_word_budget",
            message=f"WARNING: Section over budget ({words} words vs target range {budget_rng[0]}–{budget_rng[1]}).",
        ))
    else:
        results.append(GateResult(
            passed=True,
            gate_name="gate_word_budget",
            message=f"PASSED: Word count {words} within acceptable envelope ({budget_rng[0]}–{budget_rng[1]}).",
        ))

    return results


# ==============================================================================
# 5. LLM BACKEND EXECUTION & AUTO-REPAIR LOOP
# ==============================================================================

def execute_llm_call(
    prompt: str,
    backend: str = "agy",
    model: Optional[str] = None,
    timeout: int = 600,
) -> str:
    """Execute raw LLM generation call."""
    if backend == "mock":
        h_m = re.search(r"##\s+([^\n]+)", prompt)
        heading = h_m.group(1) if h_m else "Mock Section"
        # Determine if Section .1
        if "Section 2.1" in prompt or "01_sec_" in prompt or "NO SUBSECTIONS (NO ###)" in prompt:
            return (
                f"## {heading}\n\n"
                "In classical computing, microprocessors process deterministic opcodes mapping directly to register transitions. "
                "The Stochastic Computer operates under a fundamentally different execution model: the foundation model functions as an unprivileged stochastic processor core. "
                "Conditioned on a staged sequence of integer token inputs, the core evaluates tensor operations to propose candidate continuations under zero ambient authority.\n\n"
                "To anchor this architectural boundary, consider the configuration parser regression where fractional timeouts truncate milliseconds. "
                "An agent issues a model invocation proposing `edit_file(path='parser.py')`. Nothing in the file system mutates upon generation; "
                "the proposal is merely an unprivileged string in an output memory buffer until the host runtime verifies and executes the change within an isolated sandbox.\n\n"
                "The execution boundary spans three tiers: the host agent runtime enforcing budgets $K_{\\max}$ and deadlines $T_{\\max}$, "
                "the inference engine managing accelerator HBM and KV caches, and the learned neural core computing vocabulary logits on accelerator silicon. "
                "Within the H-S-A-C coordinate space, an atomic invocation occupies the baseline coordinate ($H=1, S=\\text{staged}, A=0, C=\\text{external}$). "
                "Before the runtime can interpret proposals, it must understand the discrete token currency in which the processor communicates."
            )
        return (
            f"## {heading}\n\n"
            "This is mock systems prose verifying the multi-call pipeline and review gates. "
            "The model proposal must be escrowed before environmental execution.\n\n"
            "### Mechanics and Hardware Invariants\n\n"
            "The memory shuttle transfers parameter weights $\\Theta$ across the $3.35\\text{ TB/s}$ HBM bus, "
            "collapsing arithmetic intensity to $I_{\\text{decode}} \\approx 1\\text{ FLOP/byte}$. "
            "Invariant closure is enforced by external deterministic test suites with explicit exit codes.\n\n"
            "$$T_{\\text{call}} = T_{\\text{queue}} + T_{\\text{prefill}} + \\sum_{t=1}^K T_{\\text{decode}, t}$$\n\n"
            "This sets up the transition to the next subsystem."
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
    # Strip CLI warnings and connector notices if present
    out = re.sub(r"^Ignoring \d+ permissions\.allow entries[^\n]*\n?", "", out).strip()
    out = re.sub(r"Separately, the claude\.ai [^\n]* connector[^\n]*\n?", "", out).strip()
    # Strip <thinking>...</thinking> if present
    out = re.sub(r"<thinking>.*?</thinking>", "", out, flags=re.DOTALL).strip()
    # Strip markdown code wrappers if model emitted ```markdown ... ```
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
            "gate_purpose_single_paragraph",
            "gate_anti_anthropomorphism",
            "gate_no_micro_instructions",
            "gate_status_envelope_drift",
            "gate_negative_scope",
            "gate_no_opening_hardware_dump",
            "gate_quarto_crossref",
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
            "1. Maintain clean section structure with 2–3 scannable `###` subheadings.\n"
            "2. If 'gate_anti_anthropomorphism' failed: Remove any phrasing like 'model thinks', 'model decides', 'let's think step by step', or 'model verifies its own answer'.\n"
            "3. If 'gate_status_envelope_drift' failed: Lock status envelope enumeration names strictly to `COMPLETED`, `TRUNCATED`, `REFUSED`, `TRANSPORT_FAILURE`. Remove drifted names like `Incomplete`, `Refusal`, `Failed`.\n"
            "4. If 'gate_no_opening_hardware_dump' failed: Rephrase the opening sentence. NEVER lead with hardware acronyms (HBM, PCIe, NVLink, SRAM). Open with the conceptual systems architecture, computational interface, or control loop.\n"
            "5. If 'gate_no_micro_instructions' failed: Eliminate any phrasing calling tokens 'micro-instructions' or 'opcodes'. Tokens are discrete data symbols and gather addresses, not executable opcodes.\n"
            "6. If 'gate_anti_recap' failed: Ensure the very first sentence engages directly with the technical mechanics without any recap opener ('In the previous section...').\n"
            "7. If 'gate_quarto_crossref' failed: Ensure all ::: fences and $$ math tags are perfectly balanced.\n\n"
            "Here is the draft that must be revised:\n"
            f"\"\"\"\n{content}\n\"\"\"\n\n"
            "Output ONLY the corrected Quarto markdown text. No backtick code fences wrapping the entire response."
        )

        content = execute_llm_call(repair_prompt, backend=backend, model=model)
        gate_results = run_review_gates(content, step_info, is_sec1=is_sec1)
        critical_failures = [
            g for g in gate_results
            if not g.passed and g.gate_name in [
                "gate_anti_anthropomorphism",
                "gate_no_micro_instructions",
                "gate_status_envelope_drift",
                "gate_no_opening_hardware_dump",
                "gate_quarto_crossref",
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

    # Terminal bridge: last 1-2 paragraphs (~150-250 words)
    terminal_bridge = ""
    if paragraphs:
        if len(paragraphs) >= 2:
            last_two = paragraphs[-2] + "\n\n" + paragraphs[-1]
            words = last_two.split()
            terminal_bridge = " ".join(words[-220:]) if len(words) > 220 else last_two
        else:
            words = paragraphs[-1].split()
            terminal_bridge = " ".join(words[-200:]) if len(words) > 200 else paragraphs[-1]

    # Math symbols
    inline_symbols = re.findall(r"\$([a-zA-Z_\\{}^0-9]+)\$", content)
    unique_symbols = list(dict.fromkeys(s for s in inline_symbols if len(s) <= 25))[:8]

    # Anchors and labels
    anchors = re.findall(r"\{#(sec-[^}]+|fig-[^}]+|tbl-[^}]+)\}", content)

    return {
        "terminal_bridge": terminal_bridge,
        "active_symbols": unique_symbols,
        "anchors": anchors,
    }


# ==============================================================================
# 7. WORKSPACE & STEP MANAGEMENT
# ==============================================================================

def get_chapter_dir(chapter_num: str, slug: Optional[str] = None) -> Path:
    ch_num = chapter_num.zfill(2)
    s = slug or SLUG_MAP.get(ch_num, f"chapter_{ch_num}")
    return DRAFTS_ROOT / f"ch{ch_num}_{s}"


def init_chapter_workspace(manifest: ChapterManifest, force: bool = False) -> Path:
    """Initialize isolated draft workspace directory for a chapter."""
    ch_dir = get_chapter_dir(manifest.number, manifest.slug)
    ch_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = ch_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    manifest_file = ch_dir / "manifest.json"
    if not manifest_file.exists():
        manifest_file.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    state_file = ch_dir / "state.json"
    if not state_file.exists():
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
            "budget_target": 800,
            "budget_range": [700, 950],
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
            "budget_target": 500,
            "budget_range": [450, 600],
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
            "active_symbols": ["$M$", "$K$", "$S_{\\max}$", "$T_{\\max}$"],
            "terminal_bridge": "",
            "cumulative_ledger": [],
        }
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    acc_file = ch_dir / "context_accumulator.md"
    if not acc_file.exists():
        initial_acc = (
            f"# Context Accumulator: Chapter {manifest.number} - {manifest.title}\n\n"
            f"**Governing Systems Question:** {manifest.governing_question}\n\n"
            f"**Core Takeaway:** {manifest.core_takeaway}\n\n"
            f"**Canonical Systems Scenario:** {manifest.canonical_scenario}\n\n"
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
# 8. EXECUTION PIPELINE (STEP, SEQUENTIAL, ASSEMBLY)
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
    backend: str = "agy",
    model: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """Execute all steps sequentially from Step 0 to Step N+2, then assemble."""
    manifest = parse_chapter_v2(chapter_num)
    ch_dir = init_chapter_workspace(manifest, force=force)
    state = load_state(ch_dir)

    print("\n" + "=" * 80)
    print(f"LAUNCHING MULTI-CALL CHAPTER DRAFTING: Chapter {manifest.number} - {manifest.title}")
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
    """Assemble all section files in sequence into assembled_draft.qmd."""
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
    assembled_file.write_text(full_text, encoding="utf-8")
    words = len(full_text.split())
    print(f"\n[Assembly] Compiled {len(blocks)} parts into {assembled_file.name} ({words:,} total words)")

    # Stage to canonical book directory: books/vol3/<ch_num>_<slug>/<ch_num>_<slug>.qmd
    ch_num = str(state.get("chapter_number", "02")).zfill(2)
    slug = state.get("slug", "processor")
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
    # Extract Section .1 text block
    sec1_match = re.search(r"## [^\n]+sec-vol3-[^\n]+-role\}(.*?)(?=## |\Z)", text, re.DOTALL)
    if sec1_match:
        sec1_text = sec1_match.group(1)
        subheadings = re.findall(r"^###\s+.*", sec1_text, re.MULTILINE)
        if len(subheadings) < 2 and len(sec1_text.split()) >= 800:
            issues.append(f"NOTICE: Section .1 contains {len(subheadings)} '###' subheadings (recommended: 2–3 clean subheadings).")

    # Check balanced div fences
    div_count = text.count(":::")
    if div_count % 2 != 0:
        issues.append(f"Unbalanced Quarto div fences (::: count = {div_count})")

    # Check contractions
    contractions = re.findall(r"\b(can't|won't|don't|doesn't|isn't|aren't|haven't|hasn't|it's)\b", text, re.IGNORECASE)
    if contractions:
        issues.append(f"Found {len(contractions)} contractions (American formal systems prose requires non-contracted forms).")

    # Check for forbidden negative scope leaks
    ch_match = re.search(r"ch(\d+)_", ch_dir.name)
    ch_num = ch_match.group(1).zfill(2) if ch_match else ""
    if ch_num in ["02", "03", "04", "05", "06", "07"]:
        forbidden_patterns = [
            (r"\b(microVMs?|Firecracker|cgroups?|seccomp|OverlayFS)\b", "Sandboxing/Virtualization leak (Chapter 08)")
        ]
        if ch_num in ["02", "03", "04"]:
            forbidden_patterns.append((r"\b(PagedAttention|virtual memory block tables?)\b", "KV-Cache Hierarchy leak (Chapter 05)"))
        if ch_num == "02":
            forbidden_patterns.append((r"\b(MCTS|Monte Carlo Tree Search|Process Reward Models?|PRMs?)\b", "Deliberation search leak (Chapter 03)"))

        for pat, desc in forbidden_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                issues.append(f"CRITICAL: Found forbidden {desc} in assembled draft: {matches[:5]}")

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
# 9. CLI ENTRY POINT
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-Call Chapter Generation Engine from Master Outline V2."
    )
    parser.add_argument("--chapter", type=str, required=True, help="Chapter number to generate (e.g. 02, 03).")
    parser.add_argument("--step", type=int, default=None, help="Execute only a single step index (0..N+2).")
    parser.add_argument("--run-all", action="store_true", help="Execute all steps sequentially to completion.")
    parser.add_argument("--backend", choices=["agy", "claude", "mock"], default="agy", help="LLM backend.")
    parser.add_argument("--model", type=str, default="gemini-3.8-flash-high", help="Model override (default: gemini-3.8-flash-high).")
    parser.add_argument("--assemble", action="store_true", help="Assemble existing section drafts into assembled_draft.qmd.")
    parser.add_argument("--validate", action="store_true", help="Audit existing assembled draft.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing steps.")
    parser.add_argument("--dry-run", action="store_true", help="Write step prompts to disk without invoking LLM.")
    parser.add_argument("--status", action="store_true", help="Show status of the chapter draft workspace.")

    args = parser.parse_args()
    num = args.chapter.zfill(2)

    try:
        manifest = parse_chapter_v2(num)
    except Exception as e:
        print(f"Error parsing chapter {num}: {e}")
        return 1

    ch_dir = get_chapter_dir(num, manifest.slug)

    if args.status:
        if not ch_dir.exists():
            print(f"Chapter {num} ({manifest.slug}) workspace uninitialized.")
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
        init_chapter_workspace(manifest, force=args.force)
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
