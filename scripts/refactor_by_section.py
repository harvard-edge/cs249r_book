#!/usr/bin/env python3
"""
scripts/refactor_by_section.py

Progressive Disclosure Section-by-Section Orchestrator for Volume IV: Physical AI Systems.
Prepares fine-grained, line-accurate section prompts and pipelines for ML Systems students
transitioning into physical AI systems.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
VOL4_ROOT = REPO_ROOT / "books" / "vol4"

# ==============================================================================
# Master Curricular Graph: Progressive Knowledge Across Volume IV
# ==============================================================================

CHAPTER_METADATA: Dict[int, Dict[str, str]] = {
    1: {
        "dir": "01_boundary",
        "file": "01_boundary.qmd",
        "title": "The Causal Boundary",
        "role": "Introduces the cyber-physical causal boundary, proposal vs. permission, the kinematic roofline, and the 4 bedrock laws.",
        "established": (
            "The cyber-physical causal boundary; software undo vs. physical irreversibility; "
            "Proposal is not permission (unprivileged Brain vs. deterministic Nervous System); "
            "The Kinematic Roofline ($v_{\\max} \\le \\min(\\dots)$); The 4 Bedrock Laws."
        ),
    },
    2: {
        "dir": "02_body",
        "file": "02_body.qmd",
        "title": "The Physical Body",
        "role": "Establishes the Five Physical Budgets and hardware constraints governing robot actuators and mechanics.",
        "established": (
            "The Five Physical Budgets: Freshness ($\\tau$), Kinetic Momentum ($E_k, p$), Actuator Bandwidth ($f_a$), "
            "Thermal Duty Cycle ($T_j$), Electrical Power Distribution Network ($V_{\\text{rail}}$); "
            "Actuator saturation, torque-speed curves, reflected rotor inertia ($N^2 J$), and PDN inductive collapse ($L dI/dt$)."
        ),
    },
    3: {
        "dir": "03_brain",
        "file": "03_brain.qmd",
        "title": "The Cognitive Brain",
        "role": "Explores the edge compute and memory wall bottlenecks for foundation models and action chunking.",
        "established": (
            "Batch-size 1 memory wall ($f \\le B/M$); Roofline analysis for edge NPUs/GPUs; "
            "Action chunking ($H$) amortizing weight transfer over physical wall-clock time; "
            "Diffusion refinement heads ($K$ denoising steps) on compact policies vs. frozen VLA backbones."
        ),
    },
    4: {
        "dir": "04_nervous",
        "file": "04_nervous.qmd",
        "title": "The Nervous System",
        "role": "Defines the multi-rate real-time communication hierarchy bridging high-latency AI with deterministic motors.",
        "established": (
            "Multi-rate timing hierarchy (20 kHz FOC current loop, 1 kHz safety enforcer, 50 Hz chunk policy, 5 Hz intent VLA); "
            "CAN-FD, EtherCAT, and PCIe determinism; Lock-free ring buffers and seqlocks; Stale KV-cache drops under deadline misses."
        ),
    },
    5: {
        "dir": "05_data",
        "file": "05_data.qmd",
        "title": "Physical Data",
        "role": "Analyzes endogenous data generation, phase lag, and dataset collection pipelines.",
        "established": (
            "Endogenous covariate shift ($O(T^2 \\epsilon)$ compounding error); Phase lag and timestamp alignment (optical exposure midpoint); "
            "Multi-frequency telemetry logging; Dataset yield economics and human teleoperation intervention bounds."
        ),
    },
    6: {
        "dir": "06_training",
        "file": "06_training.qmd",
        "title": "Policy Training",
        "role": "Examines behavioral cloning, ACT, diffusion policies, and sim-to-real contact dynamics.",
        "established": (
            "Multimodal action distribution collapse under MSE regression; Diffusion policy score-matching; "
            "Sim-to-real transfer gaps; Contact dynamics non-smoothness; Domain randomization."
        ),
    },
    7: {
        "dir": "07_evaluation",
        "file": "07_evaluation.qmd",
        "title": "Closed-Loop Evaluation",
        "role": "Replaces static benchmark metrics with closed-loop statistical sequential testing.",
        "established": (
            "Why 99.9% offline validation accuracy fails in closed loop; Sequential Probability Ratio Test (SPRT); "
            "Clopper-Pearson exact confidence intervals; Rule of Three for zero-event Poisson trials ($T \\ge 3.0/\\lambda$)."
        ),
    },
    8: {
        "dir": "08_perception",
        "file": "08_perception.qmd",
        "title": "Sensor Perception",
        "role": "Establishes typed sensor observation contracts and latency-to-distance conversion.",
        "established": (
            "Latency-to-distance conversion ($d = v \\cdot \\tau_{\\text{delay}}$); Rolling vs. global shutter distortion; "
            "Camera, lidar, radar, and event camera physical contracts; Sensor uncertainty covariances."
        ),
    },
    9: {
        "dir": "09_memory",
        "file": "09_memory.qmd",
        "title": "Spatial Memory",
        "role": "Treats spatial representations as 3D-indexed external KV-caches with physical decay.",
        "established": (
            "Spatial memory as an externalized, bandwidth-budgeted KV-cache; Truncated Signed Distance Fields (TSDF); "
            "Epistemic uncertainty decay over time; 3D Gaussian Splatting and occupancy voxel grids."
        ),
    },
    10: {
        "dir": "10_intent",
        "file": "10_intent.qmd",
        "title": "Grounded Intent",
        "role": "Bridges semantic foundation model reasoning with downstream geometric affordances and leases.",
        "established": (
            "Grounded affordance volumes; Expiring time-bounded capability leases; Semantic hallucination sandboxing; "
            "Typed proposal handoffs from high-level reasoning to low-level controllers."
        ),
    },
    11: {
        "dir": "11_planning",
        "file": "11_planning.qmd",
        "title": "Trajectory Planning",
        "role": "Solves action chunk seam continuity ($C^2$), stopping suffixes, and tail latency handling.",
        "established": (
            "$C^2$ quintic spline reconstruction filters across discrete chunk seams; Torque shock mitigation; "
            "Compulsory stopping suffixes in every emitted chunk; 1.5x deceleration margin tax."
        ),
    },
    12: {
        "dir": "12_enforcement",
        "file": "12_enforcement.qmd",
        "title": "Safety Enforcement",
        "role": "Constructs the 1 kHz deterministic eBPF-style actuator packet firewall and fallback ladder.",
        "established": (
            "Control Barrier Functions (CBFs) as runtime inductive loop invariants; Minimal intervention QP projection; "
            "The Fallback Ladder (TRACK, SLOW, STOP, HARDWARE CUTOFF); Hardware privilege ring separation."
        ),
    },
    13: {
        "dir": "13_placement",
        "file": "13_placement.qmd",
        "title": "Silicon Partitioning",
        "role": "Maps heterogeneous workloads onto SoCs under DRAM crossbar contention and thermal limits.",
        "established": (
            "Heterogeneous SoC layout (Safety MCU, CPU cores, GPU/NPU accelerators, unified DRAM); "
            "DRAM crossbar memory interference and bandwidth starvation; Inductive voltage droop ($L dI/dt$) and thermal throttling."
        ),
    },
    14: {
        "dir": "14_intervention",
        "file": "14_intervention.qmd",
        "title": "Supervisory Intervention",
        "role": "Examines human-in-the-loop shared autonomy, bumpless handoffs, and teleoperation physics.",
        "established": (
            "Remote teleoperation latency limits; Single-writer fieldbus mutex; Bumpless torque transfer; "
            "Causal masking for imitation learning datasets (DAgger intervention relabeling)."
        ),
    },
    15: {
        "dir": "15_verification",
        "file": "15_verification.qmd",
        "title": "Adversarial Verification",
        "role": "Applies Signal Temporal Logic (STL) and Hardware-in-the-Loop stress testing.",
        "established": (
            "Digital twin delusion; Quantitative STL robustness as adversarial loss; SOTIF vs. Functional Safety; "
            "Hardware-in-the-Loop (HIL) testbed architectures; Extreme Value Theory (EVT) tail risk."
        ),
    },
    16: {
        "dir": "16_release",
        "file": "16_release.qmd",
        "title": "Deployment Release",
        "role": "Structures safety cases, decaying ODD micro-leases, and auditable release sign-offs.",
        "established": (
            "Claim-Argument-Evidence (CAE) assertion trees; Decaying ODD micro-leases as capability tokens; "
            "Cryptographically signed release manifests; Mandatory extrinsic recalibration intervals."
        ),
    },
    17: {
        "dir": "17_frontier",
        "file": "17_frontier.qmd",
        "title": "The Epistemic Frontier",
        "role": "Synthesizes the 4 bedrock laws, observational indistinguishability, and the 4 grand open challenges.",
        "established": (
            "Capstone synthesis: Physical AI as an architectural discipline; The Poisson Exposure Wall ($3 \\times 10^9$ h); "
            "The Grid Wall ($N^d$) in formal verification; Observational indistinguishability; The 4 Grand Open MLSys Challenges."
        ),
    },
}

# ==============================================================================
# Data Structures
# ==============================================================================

@dataclasses.dataclass
class Section:
    index: int
    title: str
    slug: str
    start_line: int
    end_line: int
    raw_lines: List[str]

    @property
    def content(self) -> str:
        return "\n".join(self.raw_lines)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def summary(self) -> str:
        # Return first 3 non-empty, non-header lines as quick summary
        lines = [
            line.strip()
            for line in self.raw_lines
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith(":::")
        ]
        sample = " ".join(lines[:3])
        return sample[:200] + "..." if len(sample) > 200 else sample


@dataclasses.dataclass
class Chapter:
    num: int
    title: str
    path: Path
    sections: List[Section]

    @property
    def total_words(self) -> int:
        return sum(s.word_count for s in self.sections)


# ==============================================================================
# Parser
# ==============================================================================

SECTION_HEADER_PAT = re.compile(r"^##\s+(.+?)(?:\s+\{#([a-z0-9-_]+)\}|\s*\{[^}]*\}|\s*)$")

def parse_chapter(chap_num: int) -> Chapter:
    meta = CHAPTER_METADATA.get(chap_num)
    if not meta:
        raise ValueError(f"Unknown chapter number: {chap_num}")

    chap_dir = VOL4_ROOT / meta["dir"]
    chap_file = chap_dir / meta["file"]
    if not chap_file.exists():
        raise FileNotFoundError(f"File not found: {chap_file}")

    lines = chap_file.read_text(encoding="utf-8").splitlines()

    sections: List[Section] = []
    current_title = "Preamble"
    current_slug = f"sec-chap{chap_num}-preamble"
    current_start = 1
    current_lines: List[str] = []
    sec_idx = 0

    for idx, line in enumerate(lines, start=1):
        m = SECTION_HEADER_PAT.match(line)
        if m:
            # End previous section if it has content
            if current_lines:
                sections.append(
                    Section(
                        index=sec_idx,
                        title=current_title,
                        slug=current_slug,
                        start_line=current_start,
                        end_line=idx - 1,
                        raw_lines=current_lines,
                    )
                )
                sec_idx += 1
            current_title = m.group(1).strip()
            current_slug = m.group(2) if m.group(2) else f"sec-chap{chap_num}-s{sec_idx}"
            current_start = idx
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            Section(
                index=sec_idx,
                title=current_title,
                slug=current_slug,
                start_line=current_start,
                end_line=len(lines),
                raw_lines=current_lines,
            )
        )

    return Chapter(
        num=chap_num,
        title=meta["title"],
        path=chap_file,
        sections=sections,
    )


# ==============================================================================
# Progressive Context & Prompt Builder
# ==============================================================================

def build_progressive_context(chap: Chapter, target_sec_idx: int) -> Dict[str, str]:
    # 1. Prior chapters summary
    prior_chaps: List[str] = []
    for c in range(1, chap.num):
        cm = CHAPTER_METADATA[c]
        prior_chaps.append(f"- **Ch {c:02d} ({cm['title']})**: {cm['established']}")
    prior_summary = "\n".join(prior_chaps) if prior_chaps else "*(This is Chapter 1: foundational opening)*"

    # 2. Preceding sections in this chapter
    preceding_secs: List[str] = []
    for s in chap.sections:
        if s.index < target_sec_idx and s.index > 0:
            preceding_secs.append(f"- **Section {s.index} ({s.title})**: {s.summary()}")
    preceding_summary = "\n".join(preceding_secs) if preceding_secs else "*(First section of this chapter)*"

    # 3. Subsequent sections in this chapter (so the agent knows what NOT to jump ahead to)
    subsequent_secs: List[str] = []
    for s in chap.sections:
        if s.index > target_sec_idx:
            subsequent_secs.append(f"- Section {s.index}: {s.title}")
    subsequent_summary = "\n".join(subsequent_secs) if subsequent_secs else "*(Final section of this chapter)*"

    return {
        "prior_chapters": prior_summary,
        "preceding_sections": preceding_summary,
        "subsequent_sections": subsequent_summary,
    }


def generate_section_prompt(chap: Chapter, sec: Section) -> str:
    ctx = build_progressive_context(chap, sec.index)
    meta = CHAPTER_METADATA[chap.num]

    prompt = f"""# ML Systems Pedagogical Refactoring Task
**Volume**: Vol IV — Physical AI: Machine Learning Systems That Sense and Act
**Target**: Chapter {chap.num:02d} ({chap.title}), Section {sec.index}: "{sec.title}"
**Anchor**: `{{#{sec.slug}}}`
**Line Range in File**: Lines {sec.start_line}–{sec.end_line} (Word count: ~{sec.word_count} words)

---

### 1. AUDIENCE PERSONA: ML Systems Student Transitioning to Physical AI
The reader is a graduate or advanced undergraduate student in Computer Science / Electrical Engineering 
who already masters standard Machine Learning Systems (e.g., CS 288 / MLSys at Harvard, Stanford, MIT, Berkeley):
- **What they know deeply**: Tensors, GPU/NPU architectures, memory hierarchies (SRAM, HBM, LPDDR5), 
  the Roofline model, cache lines, Linux OS, concurrency, eBPF, PyTorch/JAX, distributed systems, 
  transformers, diffusion models, loss functions, compilation, latency vs. throughput trade-offs.
- **What they are encountering for the first time**: Physical mass, momentum, inertia, motor torque saturation, 
  thermal dissipation ceilings, non-smooth contact transitions, and sensor latency converting into unrecoverable stopping distance.
- **THE LITMUS TEST**: Physical and mechanical concepts must ONLY appear to answer ML systems architectural 
  decisions. If this section reads like a classical mechanical engineering or robotics textbook 
  (Denavit-Hartenberg parameters, Lie algebra brackets, Christoffel symbols, Bode/Nyquist plots, gear tooth metallurgy), 
  IT FAILS.
- **NO GIMMICKY SOFTWARE METAPHORS**: Do NOT pepper paragraphs with cheap analogies 
  (e.g., avoid "Unlike a database rollback...", "Unlike SIGKILL...", "Just like Redis..."). 
  State the physical reality directly and rigorously from first principles (mass, heat, voltage, delay, reflected inertia), 
  then formulate the exact systems consequence for ML architectures.
- **PRESERVE MATHEMATICAL RIGOR**: Do NOT dilute the math into pop-science conversational fluff. 
  Keep and clearly motivate formal equations whenever they decide an ML systems design trade-off 
  (e.g., minimum chunk horizons, control loop rate limits, stopping distance bounds, QP feasibility margins).
- **VOICE & TONE**: Maintain the textbook's signature authoritative, sober, empirical, declarative voice. 
  Avoid conversational filler, hype words, or cliché transitions.

---

### 2. PROGRESSIVE DISCLOSURE & PREVIOUS KNOWLEDGE
The student reads linearly and has ALREADY mastered the preceding curriculum:

#### A. Concepts Established in Earlier Chapters:
{ctx["prior_chapters"]}

#### B. Concepts Established Earlier in THIS Chapter ({chap.title}):
{ctx["preceding_sections"]}

#### C. Topics Reserved for Subsequent Sections (DO NOT PREEMPT OR JUMP AHEAD):
{ctx["subsequent_sections"]}

#### ⚠️ ANTI-REPETITION MANDATE:
- **DO NOT** re-explain concepts, acronyms, or formulas that were already established in earlier chapters 
  (e.g., do not re-derive why proposal is not permission, the 5 budgets, the batch-1 memory wall, or SPRT).
- **DO** build directly on top of them using precise cross-references (e.g., `@sec-body-five-budgets`, `@eq-brain-memory-wall`).
- Focus 100% of your pedagogical energy on the **new architectural mechanism** introduced in THIS section.

---

### 3. SECTION QUALITY GATES & INVARIANTS
1. **American English**: Strictly use American English (`-ize`, `center`, `behavior`, `canceled`).
2. **Preserve Quarto/Python Blocks**: Do NOT delete or modify any ````{{python}}```` calculation blocks or `#| label:` tags.
3. **Preserve Cross-Reference Anchors**: Keep section labels (`{{#{sec.slug}}}`), equation labels (`{{#eq-...}}`), and figure labels intact.
4. **Footnotes (3-Job Rule)**: 
   - Footnote slugs MUST follow the 3-token taxonomy: `[^fn-word1-word2-word3]`.
   - Footnotes must NEVER be placed inside Pandoc/Quarto callout div blocks (`::: {{.callout-...}}`).
   - Maximum 3 sentences per footnote definition.
5. **No AI Attribution**: Never add auto-footers, emojis, or co-author tags.

---

### 4. ORIGINAL SECTION CONTENT TO REFACTOR (Lines {sec.start_line}–{sec.end_line}):

```markdown
{sec.content}
```

---

### YOUR DELIVERABLE:
Provide the completely refactored, production-ready markdown replacement for this section.
Ensure it satisfies every constraint above, achieves total pedagogical clarity for an MLSys student, 
and seamlessly fits between the preceding and subsequent sections.
"""
    return prompt


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Progressive Disclosure Section-by-Section Orchestrator for Volume IV."
    )
    parser.add_argument("--chapter", "-c", type=int, help="Chapter number (1-17)")
    parser.add_argument("--section", "-s", type=int, help="Section index within chapter")
    parser.add_argument("--list-chapters", action="store_true", help="List all 17 chapters and word counts")
    parser.add_argument("--list-sections", action="store_true", help="List all sections for the chosen chapter")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated prompt for the section")
    parser.add_argument("--export-prompt", type=Path, help="Export prompt to a file")
    parser.add_argument("--test", action="store_true", help="Run self-test on Chapter 2, Section 2")

    args = parser.parse_args()

    if args.list_chapters:
        print("\n=== Volume IV: Chapters Overview ===")
        total_all = 0
        for num in sorted(CHAPTER_METADATA.keys()):
            chap = parse_chapter(num)
            print(f"Ch {num:02d}: {chap.title:<36} ({len(chap.sections):2d} sections, {chap.total_words:6d} words)")
            total_all += chap.total_words
        print(f"Total Volume IV Words: {total_all:,} words\n")
        return

    if args.test:
        chap = parse_chapter(2)
        # Select Section 2: "The Five Physical Budgets"
        target_sec = chap.sections[2] if len(chap.sections) > 2 else chap.sections[1]
        print(f"\n[SELF-TEST] Testing Chapter {chap.num} ({chap.title}) -> Section {target_sec.index} ({target_sec.title})")
        prompt = generate_section_prompt(chap, target_sec)
        print(f"[SELF-TEST] Generated prompt length: {len(prompt):,} characters")
        print("--- PROMPT PREVIEW (First 60 lines) ---")
        for line in prompt.splitlines()[:60]:
            print(line)
        print("--- [END PREVIEW] ---\n")
        return

    if not args.chapter:
        parser.print_help()
        return

    chap = parse_chapter(args.chapter)

    if args.list_sections:
        print(f"\n=== Sections for Chapter {chap.num:02d}: {chap.title} ({chap.total_words:,} words) ===")
        for s in chap.sections:
            print(f"  [{s.index:2d}] {s.title:<45} (Lines {s.start_line:5d}–{s.end_line:5d}, {s.word_count:5d} words) {{#{s.slug}}}")
        print()
        return

    if args.section is not None:
        target_sec = None
        for s in chap.sections:
            if s.index == args.section:
                target_sec = s
                break
        if not target_sec:
            print(f"Error: Section index {args.section} not found in Chapter {chap.num}. Use --list-sections.")
            sys.exit(1)

        prompt = generate_section_prompt(chap, target_sec)

        if args.export_prompt:
            args.export_prompt.write_text(prompt, encoding="utf-8")
            print(f"Exported prompt to {args.export_prompt} ({len(prompt):,} bytes)")
        elif args.show_prompt:
            print(prompt)
        else:
            print(f"Chapter {chap.num}, Section {target_sec.index}: '{target_sec.title}' ready.")
            print(f"Use --show-prompt to display or --export-prompt <path> to save.")


if __name__ == "__main__":
    main()
