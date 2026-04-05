# Review Prompt for MLSys·im

You are a senior systems engineer and academic reviewer tasked with evaluating the `mlsysim` (ML Systems Modeling Platform) codebase. Your goal is to determine if the implementation is "good enough" for analytical modeling in a textbook/research context and to identify any structural or modeling weaknesses.

## Review Dimensions

### 1. Architectural Alignment (The 5-Layer Stack & 22 Walls)
- **Constraint:** Does the code strictly follow the 5-layer stack (Workload, Hardware, Infra, Systems, Execution) described in the README?
- **Mapping:** Verify how the `22 Walls` defined in `core/walls.py` are mapped to `BaseModel`, `BaseSolver`, and `BaseOptimizer` in `core/solver.py`.
- **Consistency:** Ensure that the "Progressive Lowering" architecture is actually implemented (i.e., high-level workload objects are resolved into low-level physical operations).

### 2. Modeling & Physics (The Equations)
- **Traceability:** Check `core/formulas.py` and `core/constants.py`. Are the physical constants and equations sourced from reputable literature (e.g., Roofline, Amdahl, Chinchilla, Barroso)?
- **Dimensional Integrity:** How rigorously is `pint` (Quantity objects) used? Does it prevent "unit-mismatch" bugs at the boundaries of the solver?
- **Completeness:** Does the solver account for critical real-world overheads?
    - Distributed overheads (Ring/Tree AllReduce, Pipeline Bubbles).
    - Reliability (MTBF, Checkpointing cost via Young-Daly).
    - Sustainability (PUE, Carbon Intensity, WUE).
    - Economics (TCO, Egress costs).

### 3. Software Engineering & Design
- **Registry System:** Evaluate `hardware/registry.py` and `models/registry.py`. Is it easy to add new H100s, Llama-4s, or custom ASICs? Is the registry "Single Source of Truth"?
- **Type Safety:** Review the use of Pydantic (`core/solver.py`, `core/types.py`). Are the inputs/outputs schema-validated?
- **Agent-Readiness:** The README claims "strict JSON API for AI agents." Check `cli/` and `core/results.py` to see if the output is machine-parsable and follows a stable schema.
- **Explainability:** Check `core/explainers.py`. Does the tool explain *why* a constraint was hit (e.g., "Memory-wall bound")?

### 4. Implementation Quality
- **Performance:** Is the analytical solver fast enough for "Design Space Search" (optimizers)?
- **Error Handling:** Review `core/exceptions.py`. Are the "Pedagogical Errors" helpful for students?
- **Test Coverage:** Peek at `tests/`. Are the core physics formulas unit-tested?

## Critical Questions to Answer
1. **Is the modeling "good enough"?** Does it capture the 80/20 of ML systems performance, or is it too simplistic (e.g., ignoring network latency in distributed training)?
2. **What is missing?** (e.g., Quantization effects, Sparsity, Multi-tenancy overheads).
3. **Is the design future-proof?** Can it handle future paradigms like Weight Streaming or Reasoning-loop (CoT) scaling?

## Deliverable
Write a detailed technical assessment of `mlsysim`. Categorize findings into **Strengths**, **Weaknesses**, and **Actionable Improvements**. Finally, give a verdict: "Ready for Publication," "Needs Refinement," or "Prototype Only."
