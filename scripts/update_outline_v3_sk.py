#!/usr/bin/env python3
"""
Updates MASTER_TEXTBOOK_OUTLINE_V3.md:
1. Enshrines the Saltzer & Kaashoek (MIT 6.033) Systems Principles charter as the primary North Star.
2. Documents the Self-Contained Curricular Guarantee (Senior Undergrad / Introductory Grad; no Vol 1/2 prerequisite).
3. Enshrines the 5 Essential Systems Pillars for chapter specifications.
4. Strips nuance guidance bloat:
   - Replaces the 18 bulky 20-line Curricular Compass blocks with concise 4-line Systems Coordinates blocks.
   - Converts all '*Possible focus (XYZ):*' meta-prompts into declarative '*XYZ:*' headings.
   - Removes all 139 micromanaged 'Causal Bridge to X.Y' lines.
"""

import re
import os

OUTLINE_PATH = "books/vol3/MASTER_TEXTBOOK_OUTLINE_V3.md"

def main():
    with open(OUTLINE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Original line count: {len(text.splitlines())}")

    # 1. Update Frontmatter
    # Replace Section 3 (Intellectual Posture) with the S&K + H&P + CS:APP Triad & Curricular Guarantee
    old_posture_start = "### 3. Intellectual Posture: The Hennessy & Patterson + CS:APP Standard"
    old_posture_end = "## Core Authoring Directives: The Authentic ML Systems Stance"

    sk_posture = """### 3. Intellectual Posture: The Saltzer & Kaashoek + Hennessy & Patterson + CS:APP Standard

The textbook's tone, pedagogy, and technical depth are anchored in the grand tradition of three landmark computer systems classics:

1. **Primary Systems Anchor — Jerome H. Saltzer & M. Frans Kaashoek (*Principles of Computer System Design: An Introduction*, MIT 6.033):**
   - **Modularity & Layering:** Coping with complexity by dividing the system into distinct client-server modules. The foundation model is an unprivileged coprocessor; the agent runtime is the supervisor.
   - **Virtualization:** Virtualizing compute (test-time deliberation search in Ch 03), virtualizing memory (logical working context in Ch 04, PagedAttention physical HBM frames in Ch 05), and naming/resolution (authoritative truth vs derivative indexes in Ch 06).
   - **Protection & Least Privilege:** Complete mediation of tool calls (Ch 07) and containment inside sandboxed Trusted Computing Bases (Ch 08).
   - **Fault Tolerance, Atomicity & Sagas:** Building reliable systems from unreliable, fail-plausible components. Write-Ahead Logging (WAL) in Ch 10 and Saga compensating transactions ($C_i$) in Ch 11.
   - **The End-to-End Argument (Saltzer, Reed, & Clark 1984):** Lower layers enforce bounded mechanical invariants (The Invariant Closure Principle), but application task correctness can only be certified end-to-end at the top layer via deterministic execution evidence (compilers, linters, sealed test suites in Ch 18).

2. **Quantitative Architecture Anchor — John L. Hennessy & David A. Patterson (*Computer Architecture: A Quantitative Approach*):**
   - **Quantitative Rigor:** Never claim a subsystem is "fast" or "efficient" without numbers. Derive memory footprints, bandwidth saturation floors, and arithmetic intensities with explicit dimensional units.
   - **The Roofline Model:** Evaluate every computational phase against accelerator compute and memory bandwidth ceilings.
   - **Boxed Worked Examples:** Every major chapter includes step-by-step boxed worked examples (`::: {.callout-note title="Worked Example: [Title]"}`) with concrete hardware numbers (e.g., sizing an H100 parameter shuttle floor or a 128k context memory footprint).
   - **Fallacies and Pitfalls:** Every chapter concludes with an unvarnished analysis of widespread industry misconceptions and subtle systems traps.

3. **Programmer's Systems Interface Anchor — Randal E. Bryant & David R. O'Hallaron (*Computer Systems: A Programmer's Perspective* - CS:APP):**
   - **The Programmer's Systems Interface:** Demystify the system from the programmer's perspective. No magic, no hand-waving, and no black boxes.
   - **Concrete ABI Contracts:** Anchor every boundary in concrete typed schemas, C structs, or Python `@dataclass` definitions before presenting abstract equations.

### 4. Self-Contained Curricular Guarantee (No Vol 1 or Vol 2 Prerequisite)

- **Volume III is 100% self-contained.** It does NOT require or assume prior completion of Volume I (*Neural Training & Backpropagation*) or Volume II (*Matrix Kernels & Megatron Distributed Sharding*).
- *Volume I & II* operate at the "chip and training engine" level (matrix multiplication kernels, CUDA tiling, gradient backpropagation, tensor sharding).
- *Volume III* is the **Systems Design level**: the neural model is taken as an unprivileged, non-deterministic coprocessor with well-characterized physical limits ($A=0$, HBM memory bandwidth floor, fail-plausible error modes), and students build the dependable, verifiable software system around it.

### 5. The Saltzer & Kaashoek Systems Framework for Agentic AI

Every part and chapter maps directly to foundational systems design principles:

| Saltzer & Kaashoek Principle | Volume III Subsystem & Chapter Mapping |
| :--- | :--- |
| **1. The Unprivileged Processor & Virtualized Deliberation** (S&K Ch 1, 2) | **Part I: Inference Serving & Deliberation** (Ch 01 Reference Arch, Ch 02 Processor Core, Ch 03 Deliberation Search) |
| **2. Virtualization of Memory & Naming** (S&K Ch 3, 4) | **Part II: Context Memory & Storage** (Ch 04 Working Sets, Ch 05 PagedAttention Frames, Ch 06 Authoritative External Memory) |
| **3. Modularity, Protection & Least Privilege** (S&K Ch 2, 5) | **Part III: Tool Actuation & I/O Peripherals** (Ch 07 RPC Tool Drivers, Ch 08 Virtualization & Sandboxing TCB) |
| **4. Control, Fault Tolerance & Atomicity** (S&K Ch 8, 9) | **Part IV: The Agent Operating System** (Ch 09 Supervisor ACB, Ch 10 Write-Ahead Logging, Ch 11 Saga Transactions) |
| **5. The Policy Compiler** (S&K Ch 2) | **Part V: The Policy Compiler** (Ch 12 Telemetry Mining, Ch 13 SFT Distillation, Ch 14 RLVR Software Oracles) |
| **6. Distributed Coordination & Systems Economics** (S&K Ch 6, 7) | **Part VI: Distributed Fleets & Operations** (Ch 15 Multi-Agent Concurrency, Ch 16 Tracing & Evaluation, Ch 17 Fleet Economics) |
| **7. The End-to-End Argument** (Saltzer, Reed, & Clark 1984) | **Part VII: System Synthesis** (Ch 18 End-to-End Verification across the Complete Stack) |

### 6. The 5 Essential Systems Pillars for Chapter Specifications

To ensure authoring agents and human contributors produce consistent, high-altitude systems text, every chapter blueprint is specified through five essential pillars:
1. **The Governing Systems Dilemma:** The concrete physical or operational failure mode the subsystem addresses.
2. **The Abstraction & Interface Contract:** Explicit input contracts, output contracts, and authority boundaries ($A=0$).
3. **What to Cover:** Core systems mechanisms, data structures, algorithms, Roofline physics, and quantitative trade-offs.
4. **What NOT to Cover (Strict Negative Scope):** Ironclad jurisdictional boundaries preventing concept bleed into other chapters.
5. **The End-to-End Verification Check:** Deterministic software tests (compilers, linters, exit codes) that close invariants without trusting model self-reports.

---

"""

    if old_posture_start in text and old_posture_end in text:
        start_idx = text.find(old_posture_start)
        end_idx = text.find(old_posture_end)
        text = text[:start_idx] + sk_posture + text[end_idx:]
        print("Updated frontmatter posture successfully.")
    else:
        print("WARNING: Could not find exact frontmatter posture markers.")

    # 2. Clean 'Possible focus'
    text = re.sub(r'\*Possible focus \(([^)]+)\):\*', r'*\1:*', text)
    print("Cleaned '*Possible focus (...):*' meta-prompts.")

    # 3. Remove 'Causal Bridge to...'
    text = re.sub(r'^[ \t]*- \*\*Causal Bridge to[^\n]*\n?', '', text, flags=re.MULTILINE)
    print("Removed micromanaged 'Causal Bridge to X.Y' lines.")

    # 4. Streamline 18 Curricular Compass blocks
    # Pattern to match the Curricular Compass block
    compass_pattern = re.compile(
        r'#### The Curricular Compass \(Where We Are in the 18 Chapters\)\s*\n\s*```[\s\S]*?Subsystem Under Construction:\s*([^\n]+)\s*\n\s*- Computational Scope:\s*([^\n]+)\s*\n\s*- Subsystems Active:\s*([^\n]+)[\s\S]*?```',
        re.MULTILINE
    )

    def compass_repl(m):
        subsystem = m.group(1).strip()
        scope = m.group(2).strip()
        active = m.group(3).strip()
        return f"""#### Systems Coordinates

- **Subsystem Under Construction:** {subsystem}
- **Computational Scope:** {scope}
- **Subsystems Active:** {active}
- **Jurisdictional Boundary:** Strict subsystem ownership; do not implement or bleed into downstream chapters."""

    text, count = compass_pattern.subn(compass_repl, text)
    print(f"Replaced {count} Curricular Compass blocks with streamlined Systems Coordinates.")

    # Write out updated text
    with open(OUTLINE_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Updated line count: {len(text.splitlines())}")
    print("Done!")

if __name__ == "__main__":
    main()
