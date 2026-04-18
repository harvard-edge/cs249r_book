# MLSys·im CLI Design Guidelines

This document establishes the unbreakable design rules for the `mlsysim` CLI.

If this project succeeds, it will not just be a textbook companion, but the industry standard for AI infrastructure planning (akin to Terraform or kubectl). To achieve this, the CLI must be built for **automation, extensibility, and AI agents**.

All code merged into `mlsysim/cli/` must strictly adhere to the following five rules.

---

### Rule 1: Schema is Law (Pre-Computation Validation)

The core physics engine (`mlsysim.core.solver`) must never receive bad data.
*   **The Design:** The CLI acts as an impenetrable shield. Every input (CLI flags, YAML files, JSON strings) must be parsed and validated by a strict Pydantic schema (e.g., `EvalNodeSchema`) *before* any core mathematical logic is invoked.
*   **The Rationale:** If a user specifies an H100 GPU but sets the bandwidth to 100 GB/s (which is physically wrong for that chip), the schema validator rejects it immediately. This guarantees that the analytical engine only ever processes mathematically and physically valid states.

### Rule 2: Strict I/O Purity (The `stdout` vs `stderr` Rule)

A modern CLI must serve two masters equally well: the human at the keyboard, and the machine/agent in the pipeline.
*   **The Design:** Standard Output (`stdout`) is *exclusively* for the final payload. Standard Error (`stderr`) is for logs, warnings, errors, and progress bars.
*   **The Rationale:** If a user runs `mlsysim --output json eval Llama3_8B H100 > result.json`, they must end up with a perfectly valid JSON file. If a progress spinner (`[⠋] Calculating...`) leaks into `stdout`, it corrupts the JSON and breaks CI/CD pipelines.

### Rule 3: Semantic Exit Codes (Agent-Ready State)

In the agentic era, scripts shouldn't have to parse text to know what went wrong.
*   **The Design:** The CLI uses a rigid taxonomy of POSIX exit codes, defined in `exceptions.py`.
    *   `Exit 0`: **Success / Feasible.** The system runs and meets all SLAs.
    *   `Exit 1`: **Bad Input.** Syntax Error, Typo, Validation Failure.
    *   `Exit 2`: **Physics Violation (Infeasible).** The model OOMs, or the pipeline is completely starved. (A hardware limitation).
    *   `Exit 3`: **SLA/Constraint Violation.** The model fits, but P99 latency > 50ms, or TCO > Budget. (A business limitation).
*   **The Rationale:** In a CI/CD pipeline, `Exit 2` tells the developer "change your architecture," while `Exit 3` tells them "ask for more budget."

### Rule 4: The 3-Tier Command Mapping

The CLI UX must accurately reflect the architectural rigor of the underlying engine.
*   **The Design:** The CLI commands must explicitly map to the three tiers of the MLSysim engine. We do not mash everything into one command.
    *   `mlsysim eval ...` strictly calls `BaseModel` components (Physics Engine). It can take direct flags or a full `mlsys.yaml` specification.
    *   `mlsysim solve ...` strictly calls `BaseSolver` components (Math Engine).
    *   `mlsysim optimize ...` strictly calls `BaseOptimizer` components (Engineering Engine).
*   **The Rationale:** The CLI help text reinforces the textbook's pedagogical goals: teaching students the difference between evaluating a state, algebraically inverting an equation, and searching a design space.

### Rule 5: Presentation is a Translation Layer, Not Logic

The CLI should not do any math. It only formats the math.
*   **The Design:** The core `mlsysim.core.solver` modules return strictly typed Pydantic objects. The CLI's only job (via `renderers.py`) is to translate that object into a `rich.Table` (for humans) or a `JSON` string (for machines/agents).
*   **The Rationale:** We can completely rewrite the terminal UI in the future (e.g., adding Textual dashboards or WebAssembly interfaces) without touching a single equation.

---

## Future Vision: Infrastructure as Code

The ultimate goal of this CLI is to support **Infrastructure as Code (IaC)**.

The `eval` command handles both quick terminal checks and full infrastructure evaluation. When you pass `mlsysim eval my_cluster.yaml`, it acts like a compiler for infrastructure, taking a declarative YAML specification of *Demand*, *Supply*, and *Ops Context*, and verifying it against all 22 system constraints simultaneously.
