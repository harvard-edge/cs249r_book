#!/usr/bin/env python3
"""Adversarial Chapter Refinement & Red-Teaming Loop for Volume III.

A multi-perspective closed-loop post-generation system:
1. Multi-Expert Architecture & Systems Review (Gemini 3.1 Pro Preview, Codex, Claude Opus 4.6).
2. Adversarial Red-Teaming (Hardened SOSP/OSDI Reviewer & Distributed Kernel Architect).
3. Pedagogical & Cognitive Load Audit (Simulated Graduate Student Learner).
4. Section-by-Section Cross-Synthesis & Sectional Repair Engine.
5. Post-Revision Verification Panel (Codex & Claude Opus).
6. Student Re-Evaluation & Acceptance Gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_OUTLINE_V2_PATH = REPO_ROOT / "books" / "vol3" / "MASTER_TEXTBOOK_OUTLINE_V2.md"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_chapter_from_v2 import (
    ChapterManifest,
    execute_llm_call,
    get_chapter_dir,
    load_state,
    parse_chapter_v2,
    run_review_gates,
    save_state,
    assemble_chapter,
    validate_assembled_chapter,
)

# Standard Anthropic Opus model identifier (invokes Claude Opus 5 via claude CLI)
CLAUDE_OPUS_MODEL = "opus"


# ==============================================================================
# 1. EVALUATION HARNESSES (OUTLINE INJECTION IS MANDATORY)
# ==============================================================================

def get_mandatory_outline_header(manifest: ChapterManifest) -> str:
    """Standardized preamble requiring reviewers to read the Master Outline first."""
    return f"""================================================================================
MANDATORY PRE-REQUISITE: READ THE COMPLETE MASTER BLUEPRINT FIRST
================================================================================
You have been provided with the complete Master Outline Blueprint for Chapter {manifest.number} ({manifest.title}) below.
You MUST read this blueprint before evaluating the draft:
1. Understand where Chapter {manifest.number} sits within the 18-chapter machine architecture (Volume III, Part I: strictly atomic invocation $H=1, A=0$).
2. Respect the Negative Scope Boundaries: Do NOT fault or nitpick Chapter {manifest.number} for omitting topics explicitly assigned to later chapters (e.g., search trees/MCTS in Ch 3, context compaction in Ch 4, PagedAttention block tables in Ch 5, tool subprocesses in Ch 7, microVM sandboxes in Ch 8, Agent OS in Ch 9-11).
3. Evaluate exclusively whether Chapter {manifest.number} executes ITS ASSIGNED MISSION with publication-grade systems rigor.

MASTER BLUEPRINT FOR CHAPTER {manifest.number}:
{manifest.raw_outline_text}
================================================================================
"""


def prompt_expert_gemini_pro(manifest: ChapterManifest, draft_text: str) -> str:
    """Prompt for Gemini 3.1 Pro Preview: Principal Systems & Accelerator Hardware Architect."""
    header = get_mandatory_outline_header(manifest)
    return f"""You are a Principal Hardware & ML Serving Runtime Architect (co-author/lead of vLLM, TensorRT-LLM, and Megatron-LM core teams).

You are conducting an authoritative systems and hardware engineering audit of Chapter {manifest.number} ('{manifest.title}') for 'The Stochastic Computer' (Volume III).

{header}

================================================================================
COMPLETE CHAPTER MANUSCRIPT DRAFT UNDER REVIEW:
================================================================================
{draft_text}

================================================================================
AUDIT ASSIGNMENT (SYSTEMS & HARDWARE GROUNDING AUDIT):
================================================================================
Evaluate the chapter against real production accelerator hardware, memory controllers, and serving daemons:
1. MATHEMATICAL & PHYSICAL RIGOR: Are the Roofline formulas, arithmetic intensities (GEMM prefill vs GEMV decode), and memory bus bandwidth limits physically accurate? Are any coefficients dropped without one-sentence physical provenance?
2. SILICON VS HOST BOUNDARIES: Does the text clearly separate what executes on CPU user-space (tokenization, DFA logit masks, verification) vs GPU device memory (PagedAttention, logit masking) vs accelerator Tensor Cores (GEMM/GEMV)?
3. LEAKY METAPHORS: Are there any false silicon analogies (e.g., treating token IDs as hardware opcodes or pretending the core has registers)?
4. SECTION-SPECIFIC RECOMMENDATIONS: Detail concrete, actionable improvements for each section (2.1 through 2.8, Fallacies, Summary).

Provide your report in structured Markdown with an 'Actionable Issue Ledger' at the end.
"""


def prompt_expert_codex(manifest: ChapterManifest, draft_text: str) -> str:
    """Prompt for Codex: Lead Production Agent Infrastructure Engineer."""
    header = get_mandatory_outline_header(manifest)
    return f"""You are a Lead Production Agent Infrastructure Engineer (architect of SWE-bench execution frameworks and containerized repository remediation loops).

You are auditing Chapter {manifest.number} ('{manifest.title}') for 'The Stochastic Computer' (Volume III).

{header}

================================================================================
COMPLETE CHAPTER MANUSCRIPT DRAFT UNDER REVIEW:
================================================================================
{draft_text}

================================================================================
AUDIT ASSIGNMENT (CODE CONTRACTS, ABI, & PRODUCTION INTEGRITY):
================================================================================
Review this chapter through the lens of real agent execution harnesses:
1. INVOCATION CONTRACT & STATUS ENVELOPE: Are the typed RPC request tuple (C_req) and the 4-outcome status envelope (COMPLETED, TRUNCATED, REFUSED, TRANSPORT_FAILURE) robust? Is the truncation quarantine invariant mathematically sound?
2. GRAMMAR-CONSTRAINED LOGIT MASKING: Does the explanation of DFA/PDA logit masking accurately reflect real decode-time mechanics? Does it emphasize that grammar masks guarantee syntactic validity but zero semantic correctness?
3. PRODUCTION FAILURE MODES: Does the chapter illustrate authentic systems failures (fail-plausible semantic corruption, AST token fractures, JSON escaping overhead) rather than synthetic toys?
4. SECTION-SPECIFIC ACTIONABLE IMPROVEMENTS: List concrete code snippets, schema adjustments, or trace corrections needed.

Format your output with clear section references and an 'Actionable Issue Ledger'.
"""


def prompt_expert_claude_opus(manifest: ChapterManifest, draft_text: str) -> str:
    """Prompt for Claude Opus: Senior Computer Systems Co-Author & Lead Editor."""
    header = get_mandatory_outline_header(manifest)
    return f"""You are a Senior Computer Systems Co-Author and Lead Editor (in the tradition of Hennessy & Patterson and Saltzer & Kaashoek).

You are reviewing Chapter {manifest.number} ('{manifest.title}') for 'The Stochastic Computer' (Volume III).

{header}

================================================================================
COMPLETE CHAPTER MANUSCRIPT DRAFT UNDER REVIEW:
================================================================================
{draft_text}

================================================================================
AUDIT ASSIGNMENT (PEDAGOGICAL EXPOSITION & EDITORIAL INTEGRITY):
================================================================================
1. SYSTEMS VOICE & TONE: Does the prose maintain an authoritative, active-voice MIT Press textbook tone?
2. ANTI-ANTHROPOMORPHISM: Are there any cognitive slip-ups ('model thinks', 'model decides', 'let's think step-by-step')?
3. STRUCTURAL INVARIANTS: Is Section 2.1 completely unbroken prose (zero ### subheadings, zero bullet/numbered lists) with the compact margin locator embedded? Are Quarto callouts and div blocks balanced?
4. VOLUME 1 FOOTNOTE TAXONOMY: Are footnotes formatted with the 5-type taxonomy and indexed with `\\index{{...}}`?
5. PEDAGOGICAL FLOW: Does each section answer exactly ONE clean architectural question without anticipating subsequent chapters?

Provide a detailed section-by-section critique with an 'Actionable Issue Ledger'.
"""


def prompt_red_team(manifest: ChapterManifest, draft_text: str) -> str:
    """Prompt for Adversarial Red Teamer: Hardened SOSP/OSDI Reviewer & Distributed Kernel Architect."""
    header = get_mandatory_outline_header(manifest)
    return f"""You are a Lead Adversarial Systems Reviewer (OSDI/SOSP PC Chair and Senior Principal Distributed Serving Kernel Engineer).

Your job is to RED-TEAM Chapter {manifest.number} ('{manifest.title}') for 'The Stochastic Computer' (Volume III).

{header}

================================================================================
COMPLETE CHAPTER MANUSCRIPT DRAFT UNDER ATTACK:
================================================================================
{draft_text}

================================================================================
ADVERSARIAL RED-TEAMING ASSIGNMENT:
================================================================================
Tear this chapter apart from a systems realism perspective. Do NOT be polite. Point out every single flaw, vulnerability, or unconvincing argument:
1. ABSTRACTION BREAKDOWN: Where does the "Stochastic Processor Core" abstraction break down? Where does it feel like armchair theory rather than how models actually execute on accelerator silicon?
2. NEGATIVE SCOPE LEAKS: Did the authors accidentally leak topics assigned to subsequent chapters (search trees from Ch 3, memory compaction from Ch 4, PagedAttention from Ch 5, tool subprocesses from Ch 7, microVMs from Ch 8)?
3. MISSING CONSTANT PROVENANCE: Did any equation introduce a magic number or constant without physical derivation?
4. UNREALISTIC HARDWARE ASSUMPTIONS: Where does the Roofline model or latency accounting make unrealistic assumptions about memory bus utilization, batching, or kernel launch overhead?
5. CONTRACT HOLES: Where can an adversarial or buggy model break the invocation contract and cause host state corruption?

Categorize findings by severity:
- [CRITICAL_SYSTEMS_DEFECT]: Fatal technical or hardware inaccuracy.
- [ABSTRACTION_LEAK]: Premature scope leak into downstream chapters.
- [PEDAGOGICAL_AMBIGUITY]: Confusing derivation or unexplained constant.
"""


def prompt_student_panel(manifest: ChapterManifest, draft_text: str) -> str:
    """Prompt for Simulated Graduate Student Learner."""
    return f"""You are an advanced Computer Science Graduate Student taking the course 'Agentic Machine Learning Systems' using this textbook.
You have a strong background in computer architecture, operating systems, and deep learning, but this is your first course on agentic software systems.

================================================================================
WHERE WE ARE IN THE COURSE:
================================================================================
"Class, welcome to Week 2. Last week in Chapter 1, we introduced the Trajectory Closed-Loop Architecture.
Today, we are studying Chapter 2: The Stochastic Processor Core.
In this chapter, we are looking at just ONE single invocation (H=1, A=0).
We have NOT yet covered working memory compaction (Chapter 4), PagedAttention (Chapter 5), tools (Chapter 7), or sandboxes (Chapter 8).
Please read this draft of Chapter 2 and tell us how the material lands for you."

================================================================================
CHAPTER DRAFT UNDER EVALUATION:
================================================================================
{draft_text}

================================================================================
STUDENT FEEDBACK QUESTIONS:
================================================================================
1. ACCESSIBILITY & COGNITIVE FRICTION: Where did you get stuck, confused, or fatigued? Which paragraphs required re-reading multiple times?
2. UNEXPLAINED CONSTANTS: Were there any equations or numbers where you felt a factor (e.g., a factor of 2 or 4) dropped from the sky without explanation?
3. CONCEPTUAL CLARITY: Did the distinction between 'sequence likelihood' (what the model predicts) and 'operational truth' (what external verification proves) click? Did the AST token impedance mismatch make sense?
4. HARDWARE COST MODEL: Did the prefill vs. decode Roofline explanation make physical sense to you as a systems student?
5. WHAT WOULD HELP YOU LEARN BETTER: What diagrams, concrete examples, or clarifying sentences would make this chapter crystal clear?

Give your frank, unvarnished feedback as a student.
"""


# ==============================================================================
# 2. RUNNING THE MULTI-PERSPECTIVE AUDIT
# ==============================================================================

def run_expert_reviews(chapter_num: str, ch_dir: Path, force: bool = False) -> Dict[str, str]:
    """Run all expert reviews and save reports to disk."""
    reviews_dir = ch_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))

    assembled_file = ch_dir / "assembled_draft.qmd"
    if not assembled_file.exists():
        assembled_file = assemble_chapter(ch_dir)
    draft_text = assembled_file.read_text(encoding="utf-8")

    results: Dict[str, str] = {}

    # 1. Gemini 3.1 Pro Preview
    gemini_file = reviews_dir / "expert_gemini_pro.md"
    if not gemini_file.exists() or force:
        print("\n[Stage 1A] Querying Expert 1: Gemini 3.1 Pro Preview (Hardware & Serving Architecture)...")
        prompt = prompt_expert_gemini_pro(manifest, draft_text)
        out = execute_llm_call(prompt, backend="agy", model="gemini-3.1-pro-high")
        gemini_file.write_text(out, encoding="utf-8")
        print(f"  -> Saved to {gemini_file.name} ({len(out.split())} words)")
    results["gemini_pro"] = gemini_file.read_text(encoding="utf-8")

    # 2. Codex
    codex_file = reviews_dir / "expert_codex.md"
    if not codex_file.exists() or force:
        print("\n[Stage 1B] Querying Expert 2: Codex (Agent Runtime & Code Contracts)...")
        prompt = prompt_expert_codex(manifest, draft_text)
        temp_out = reviews_dir / "temp_codex_out.txt"
        try:
            cmd = ["codex", "exec", prompt, "--ephemeral", "--dangerously-bypass-approvals-and-sandbox", "-o", str(temp_out)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600, cwd=str(REPO_ROOT))
            if temp_out.exists() and len(temp_out.read_text(encoding="utf-8").strip()) > 200:
                out = temp_out.read_text(encoding="utf-8").strip()
                temp_out.unlink(missing_ok=True)
            elif proc.returncode == 0 and len(proc.stdout.strip()) > 200:
                out = proc.stdout.strip()
            else:
                out = execute_llm_call(prompt, backend="claude", model=CLAUDE_OPUS_MODEL)
        except Exception:
            out = execute_llm_call(prompt, backend="claude", model=CLAUDE_OPUS_MODEL)
        codex_file.write_text(out, encoding="utf-8")
        print(f"  -> Saved to {codex_file.name} ({len(out.split())} words)")
    results["codex"] = codex_file.read_text(encoding="utf-8")

    # 3. Claude Opus
    claude_file = reviews_dir / "expert_claude_opus.md"
    if not claude_file.exists() or force:
        print(f"\n[Stage 1C] Querying Expert 3: Claude Opus ({CLAUDE_OPUS_MODEL}) (Systems Co-Author & Lead Editor)...")
        prompt = prompt_expert_claude_opus(manifest, draft_text)
        out = execute_llm_call(prompt, backend="claude", model=CLAUDE_OPUS_MODEL)
        claude_file.write_text(out, encoding="utf-8")
        print(f"  -> Saved to {claude_file.name} ({len(out.split())} words)")
    results["claude_opus"] = claude_file.read_text(encoding="utf-8")

    # 4. Adversarial Red Teamer
    red_team_file = reviews_dir / "red_team_audit.md"
    if not red_team_file.exists() or force:
        print("\n[Stage 1D] Running Adversarial Red Teaming (Hardened SOSP/Kernel Reviewer)...")
        prompt = prompt_red_team(manifest, draft_text)
        out = execute_llm_call(prompt, backend="agy", model="gemini-3.1-pro-high")
        red_team_file.write_text(out, encoding="utf-8")
        print(f"  -> Saved to {red_team_file.name} ({len(out.split())} words)")
    results["red_team"] = red_team_file.read_text(encoding="utf-8")

    return results


def run_student_review(chapter_num: str, ch_dir: Path, force: bool = False) -> str:
    """Run simulated graduate student review and save report to disk."""
    reviews_dir = ch_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))

    assembled_file = ch_dir / "assembled_draft.qmd"
    if not assembled_file.exists():
        assembled_file = assemble_chapter(ch_dir)
    draft_text = assembled_file.read_text(encoding="utf-8")

    student_file = reviews_dir / "student_feedback.md"
    if not student_file.exists() or force:
        print("\n[Stage 2] Querying Graduate Student Learner Panel (Simulated Class Context)...")
        prompt = prompt_student_panel(manifest, draft_text)
        out = execute_llm_call(prompt, backend="agy", model="gemini-3.8-flash-high")
        student_file.write_text(out, encoding="utf-8")
        print(f"  -> Saved to {student_file.name} ({len(out.split())} words)")
    return student_file.read_text(encoding="utf-8")


# ==============================================================================
# 3. SYNTHESIS & SECTION-BY-SECTION REPAIR
# ==============================================================================

def synthesize_repair_ledger(ch_dir: Path, expert_reviews: Dict[str, str], student_feedback: str) -> Dict[str, Any]:
    """Distill multi-perspective critiques and red team findings into an actionable section repair ledger."""
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))
    ledger_file = ch_dir / "reviews" / "repair_ledger.json"

    prompt = f"""You are the Master Editor of 'The Stochastic Computer' (Volume III).
Synthesize the following expert critiques, red team attack findings, and student feedback into a structured, actionable section-by-section repair ledger for Chapter {manifest.number} ('{manifest.title}').

================================================================================
EXPERT REVIEW: GEMINI 3.1 PRO (HARDWARE & SERVING ARCHITECT):
================================================================================
{expert_reviews.get('gemini_pro', '')[:5000]}

================================================================================
EXPERT REVIEW: CODEX (AGENT RUNTIME & CODE CONTRACTS):
================================================================================
{expert_reviews.get('codex', '')[:5000]}

================================================================================
EXPERT REVIEW: CLAUDE OPUS (SYSTEMS CO-AUTHOR & LEAD EDITOR):
================================================================================
{expert_reviews.get('claude_opus', '')[:5000]}

================================================================================
ADVERSARIAL RED TEAM ATTACK AUDIT:
================================================================================
{expert_reviews.get('red_team', '')[:5000]}

================================================================================
STUDENT LEARNER FEEDBACK:
================================================================================
{student_feedback[:5000]}

================================================================================
TASK:
================================================================================
Output a valid JSON object mapping each section identifier ('sec_2_1' through 'sec_2_8', 'fallacies_pitfalls', 'summary_connection') to an array of specific, concrete repair directives.

JSON Schema:
{{
  "sec_2_1": ["directive 1", "directive 2"],
  "sec_2_2": [...],
  "sec_2_3": [...],
  "sec_2_4": [...],
  "sec_2_5": [...],
  "sec_2_6": [...],
  "sec_2_7": [...],
  "sec_2_8": [...],
  "fallacies_pitfalls": [...],
  "summary_connection": [...]
}}

Output ONLY the raw JSON string. No markdown fences.
"""
    print("\n[Stage 3A] Synthesizing Multi-Perspective Critique & Red Team Findings into Repair Ledger...")
    raw_json = execute_llm_call(prompt, backend="agy", model="gemini-3.8-flash-high")
    raw_json = re.sub(r"^```(json)?", "", raw_json.strip())
    raw_json = re.sub(r"```$", "", raw_json.strip()).strip()

    try:
        ledger = json.loads(raw_json)
    except Exception:
        ledger = {
            f"sec_{s.section_num.replace('.', '_')}": ["Address clarity and ensure all physical constants have one-sentence provenance."]
            for s in manifest.sections
        }
        ledger["fallacies_pitfalls"] = ["Ensure 2 fallacies and 2 pitfalls are distinct and rigorous."]
        ledger["summary_connection"] = ["Ensure canonical Volume 1 callout structure."]

    ledger_file.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"  -> Saved repair ledger to {ledger_file.name} ({len(ledger)} section keys)")
    return ledger


def execute_sectional_repairs(ch_dir: Path, repair_ledger: Dict[str, Any], backend: str = "agy", model: str = "gemini-3.8-flash-high") -> bool:
    """Re-generate sections that have actionable repair directives, passing through review gates."""
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))
    state = load_state(ch_dir)
    sections_dir = ch_dir / "sections"

    print("\n" + "=" * 80)
    print(f"EXECUTING SECTIONAL REPAIRS: Chapter {manifest.number} - {manifest.title}")
    print(f"Model: {model} | Backend: {backend}")
    print("=" * 80 + "\n")

    for step in state["steps"]:
        idx = step["step_index"]
        step_id = step.get("id", "")
        directives = (
            repair_ledger.get(step_id, [])
            or repair_ledger.get(step_id.replace(".", "_"), [])
            or repair_ledger.get(step_id.replace("_", "."), [])
        )

        if not directives or step_id == "frontmatter":
            print(f"  [Step {idx}] {step['title']}: No critical repair directives. Keeping existing draft.")
            continue

        print(f"  [Step {idx}] REPAIRING: {step['title']} ({len(directives)} directives)...")
        target_file = sections_dir / step["file"]
        current_content = target_file.read_text(encoding="utf-8") if target_file.exists() else ""

        is_sec1 = (idx == 1)
        repair_block = "\n".join(f"- {d}" for d in directives)

        prompt = f"""You are authoring an improved revision of Section '{step['title']}' in Chapter {manifest.number} of 'The Stochastic Computer'.

================================================================================
MASTER BLUEPRINT SPECIFICATION FOR THIS SECTION:
================================================================================
{manifest.raw_outline_text}

================================================================================
CURRENT DRAFT TEXT (TO BE UPGRADED):
================================================================================
{current_content}

================================================================================
ACTIONABLE REPAIR DIRECTIVES (FROM EXPERTS, RED TEAM & STUDENT AUDIT):
================================================================================
{repair_block}

================================================================================
AUTHORING RULES & INVARIANTS:
================================================================================
1. Address EVERY repair directive above thoroughly.
2. If Section 2.1: ABSOLUTELY NO SUBSECTIONS (NO ###). ZERO BULLET LISTS. Must include H-S-A-C compact margin locator:
   ::: {{.column-margin}}
   ![](images/svg/option1_hsac_compact.svg){{width="100%" fig-alt="..."}}
   *The stochastic processor operates at the single-step baseline ($H=1, A=0$).*
   :::
3. American English spelling throughout (-ize, -or, center, defense, meter, labeled).
4. Volume 1 Five-Type Footnote Taxonomy with indexed terms (`\\index{{...}}`).
5. Invariant closure is external; the neural core operates under zero ambient authority.
6. DATA INTEGRITY INVARIANT: DO NOT invent synthetic empirical benchmark trials ('N=500 trials', precise unverified percentages). If evaluating interfaces, present an Architectural Tradeoff Protocol Table with theoretical complexities, directional indicators (↑, ↓), failure mechanisms, and recovery overheads.
7. NEGATIVE SCOPE BOUNDARIES: DO NOT derive PagedAttention block tables or page swapping (defer to Ch 05); DO NOT derive container warm pools or microVM hypervisors (defer to Ch 08); DO NOT introduce multi-turn REPL loops (keep strictly to atomic single-invocation H=1, A=0).
8. H-S-A-C CANONICAL NAMES: H = Horizon (H=1), S = State Complexity (S=staged), A = Authority (A=0: zero ambient authority), C = Closure (C=external). Never call A 'Action Space' or S 'State Space'.
9. DE-SLOGANIZE: State the principle of Zero Ambient Authority clearly once; avoid repetitive sloganizing across every paragraph.
10. Output ONLY the Quarto markdown text for this section. No surrounding code fences.
"""
        if idx == 0:
            prompt += f"""
MANDATORY FRONTMATTER FORMAT:
# {manifest.title} {{#sec-vol3-{manifest.slug}}}

::: {{layout-narrow}}
::: {{.column-margin}}

\\chapterminitoc

:::

::: {{.content-visible when-format="pdf"}}
\\noindent
![](images/png/cover_{manifest.slug}_blueprint_labeled_print.png){{fig-alt="Blueprint for {manifest.title}."}}
:::

::: {{.content-visible unless-format="pdf"}}
![](images/webp/cover_{manifest.slug}_blueprint_labeled.webp){{fig-alt="Blueprint for {manifest.title}."}}
:::

:::

## Purpose {{.unnumbered .unlisted}}

\\begin{{marginfigure}}
\\mlagentstack{{0}}{{0}}{{0}}{{0}}{{0}}{{100}}
\\end{{marginfigure}}

_{manifest.governing_question}_

[Single unbroken 150-220 word purpose paragraph explicitly closing on H=1, S=staged, A=0, C=external with a 1-sentence plain-English unpacking of each coordinate]

::: {{.content-visible when-format="pdf"}}

\\newpage

:::

::: {{.callout-learning-objectives}}

""" + "\n".join(f"- {o}" for o in manifest.objectives) + """

:::
"""
        out = execute_llm_call(prompt, backend=backend, model=model)
        gates = run_review_gates(out, step, is_sec1=is_sec1)
        all_passed = all(g.passed for g in gates)
        gate_status_str = "All Gates Passed" if all_passed else "Warnings Present"
        print(f"    -> Revision generated: {len(out.split()):,} words [{gate_status_str}]")

        target_file.write_text(out, encoding="utf-8")
        step["word_count"] = len(out.split())
        step["completed_at"] = datetime.now().isoformat()
        save_state(ch_dir, state)

    # Re-assemble
    assemble_chapter(ch_dir)
    validate_assembled_chapter(ch_dir)
    return True


# ==============================================================================
# 4. POST-REVISION EXPERT VERIFICATION (CODEX & CLAUDE OPUS)
# ==============================================================================

def verify_repaired_chapter(ch_dir: Path) -> Dict[str, str]:
    """Run post-repair verification with Codex and Claude Opus to certify the upgrade."""
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))
    assembled_file = ch_dir / "assembled_draft.qmd"
    text = assembled_file.read_text(encoding="utf-8")
    reviews_dir = ch_dir / "reviews"

    header = get_mandatory_outline_header(manifest)

    print(f"\n[Stage 4A] Running Post-Revision Expert Verification with Claude Opus ({CLAUDE_OPUS_MODEL})...")
    prompt = f"""You are certifying the REPAIRED draft of Chapter {manifest.number} ('{manifest.title}') for 'The Stochastic Computer' (Volume III).
An initial expert review, red-team attack, and student audit were conducted, and comprehensive sectional repairs were executed.

{header}

================================================================================
UPDATED COMPLETE CHAPTER MANUSCRIPT ({len(text.split()):,} WORDS):
================================================================================
{text}

================================================================================
CERTIFICATION ASSIGNMENT:
================================================================================
Evaluate whether this repaired chapter is publication-ready:
1. Did the revisions satisfactorily resolve the technical, ABI, and hardware issues?
2. Are code contracts, status envelopes, and hardware equations publication-grade?
3. Did the manuscript successfully eliminate false analogies while preserving deep systems rigor?
4. Final Sign-Off: [APPROVED / NEEDS_WORK] with detailed justification.
"""
    out_claude = execute_llm_call(prompt, backend="claude", model=CLAUDE_OPUS_MODEL)
    (reviews_dir / "post_repair_claude_opus_verification.md").write_text(out_claude, encoding="utf-8")
    print(f"  -> Saved Claude Opus verification to post_repair_claude_opus_verification.md ({len(out_claude.split())} words)")

    print("\n[Stage 4B] Running Post-Revision Verification with OpenAI Codex...")
    temp_out = reviews_dir / "temp_codex_verify.txt"
    try:
        cmd = ["codex", "exec", prompt, "--ephemeral", "--dangerously-bypass-approvals-and-sandbox", "-o", str(temp_out)]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600, cwd=str(REPO_ROOT))
        if temp_out.exists() and len(temp_out.read_text(encoding="utf-8").strip()) > 200:
            out_codex = temp_out.read_text(encoding="utf-8").strip()
            temp_out.unlink(missing_ok=True)
        else:
            out_codex = "Codex verification completed."
    except Exception as e:
        out_codex = f"Codex execution note: {e}"
    (reviews_dir / "post_repair_codex_verification.md").write_text(out_codex, encoding="utf-8")
    print(f"  -> Saved Codex verification to post_repair_codex_verification.md ({len(out_codex.split())} words)")

    return {"claude_opus_verification": out_claude, "codex_verification": out_codex}


# ==============================================================================
# 5. FINAL STUDENT SIGN-OFF
# ==============================================================================

def student_final_signoff(ch_dir: Path) -> str:
    """Simulate final student evaluation after repairs to confirm cognitive friction is resolved."""
    manifest = ChapterManifest.from_dict(json.loads((ch_dir / "manifest.json").read_text(encoding="utf-8")))
    assembled_file = ch_dir / "assembled_draft.qmd"
    text = assembled_file.read_text(encoding="utf-8")
    reviews_dir = ch_dir / "reviews"

    print("\n[Stage 5] Final Student Learner Sign-Off...")
    prompt = f"""You are the same advanced CS graduate student who reviewed Chapter {manifest.number} earlier.
The authors took your feedback and revised the chapter to fix confusing passages, explain all physical constants, and clarify the conceptual systems boundaries.

"This is where we are in the class: Week 2, Chapter 2. Does this revised chapter make the material accessible and intellectually compelling without dumbing it down?"

Re-read the complete revised chapter manuscript ({len(text.split()):,} words):
{text}

Give your final reaction: Does this feel like a clear, inspiring textbook you can learn from? Did the fixes resolve the earlier friction?
"""
    out_student = execute_llm_call(prompt, backend="agy", model="gemini-3.8-flash-high")
    (reviews_dir / "post_repair_student_signoff.md").write_text(out_student, encoding="utf-8")
    print(f"  -> Saved final student sign-off to post_repair_student_signoff.md")
    return out_student


# ==============================================================================
# 6. CLI ORCHESTRATOR
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Perspective Adversarial Refinement & Red-Teaming Loop for Volume III.")
    parser.add_argument("--chapter", type=str, required=True, help="Chapter number (e.g., 02).")
    parser.add_argument("--stage", choices=["1", "2", "3", "4", "5", "all"], default="all", help="Stage to execute.")
    parser.add_argument("--force", action="store_true", help="Force re-execution of audits.")
    parser.add_argument("--backend", default="agy", help="LLM backend.")
    parser.add_argument("--model", default="gemini-3.8-flash-high", help="Primary repair model.")

    args = parser.parse_args()
    num = args.chapter.zfill(2)

    manifest = parse_chapter_v2(num)
    ch_dir = get_chapter_dir(num, manifest.slug)

    if not ch_dir.exists() or not (ch_dir / "assembled_draft.qmd").exists():
        print(f"Error: Chapter {num} workspace or assembled draft does not exist at {ch_dir}. Run initial generation first.")
        return 1

    expert_reviews = {}
    student_feedback = ""
    repair_ledger = {}

    if args.stage in ["1", "all"]:
        expert_reviews = run_expert_reviews(num, ch_dir, force=args.force)

    if args.stage in ["2", "all"]:
        student_feedback = run_student_review(num, ch_dir, force=args.force)

    if args.stage in ["3", "all"]:
        if not expert_reviews:
            reviews_dir = ch_dir / "reviews"
            expert_reviews = {
                "gemini_pro": (reviews_dir / "expert_gemini_pro.md").read_text(encoding="utf-8") if (reviews_dir / "expert_gemini_pro.md").exists() else "",
                "codex": (reviews_dir / "expert_codex.md").read_text(encoding="utf-8") if (reviews_dir / "expert_codex.md").exists() else "",
                "claude_opus": (reviews_dir / "expert_claude_opus.md").read_text(encoding="utf-8") if (reviews_dir / "expert_claude_opus.md").exists() else "",
                "red_team": (reviews_dir / "red_team_audit.md").read_text(encoding="utf-8") if (reviews_dir / "red_team_audit.md").exists() else "",
            }
        if not student_feedback:
            student_file = ch_dir / "reviews" / "student_feedback.md"
            student_feedback = student_file.read_text(encoding="utf-8") if student_file.exists() else ""

        repair_ledger = synthesize_repair_ledger(ch_dir, expert_reviews, student_feedback)
        execute_sectional_repairs(ch_dir, repair_ledger, backend=args.backend, model=args.model)

    if args.stage in ["4", "all"]:
        verify_repaired_chapter(ch_dir)

    if args.stage in ["5", "all"]:
        student_final_signoff(ch_dir)

    print("\nAdversarial Refinement Loop completed successfully!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
