# mlsysim Solver & Optimization Plan

## Problem Statement
The current solvers (`ParallelismOptimizer`, `BatchingOptimizer`, etc.) rely on naive, pure-Python search algorithms (nested `for` loops, binary search). As the simulation space grows (e.g., modeling 3D parallelism splits across heterogeneous networks), these custom search loops will suffer from combinatorial explosion and become unmaintainable.

## Architectural Philosophy
**Operations Research (OR) "Black Box" Decoupling**
`mlsysim` should not *find* the optimal solution. It should only *describe* the physical limits ($Y = f(X)$). The logic of traversing the search space should be delegated to industry-standard mathematical programming engines.

**Strictly Analytical (No Simulation)**
Unlike cycle-accurate simulators or stochastic Monte Carlo engines, `mlsysim` remains a purely analytical framework. It computes provable Roofline bounds and deterministic physical limits. It is a mathematical oracle, not a time-stepping simulator.

## 1. Solver Selection Strategy

We will use a "Right Tool for the Job" routing strategy based on the mathematical shape of the constraints:

| Problem Domain | Mathematical Class | Characteristics | Chosen Engine | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **System Tuning** (e.g., finding the precise batch size to maximize throughput before hitting a 50ms latency wall) | Continuous Optimization | Variables are continuous (floats). Finding local/global minima on a smooth curve. | `scipy.optimize` | Best-in-class gradient descent algorithms (L-BFGS-B, SLSQP) to quickly traverse Roofline equations. |
| **Architecture Search** (e.g., finding the exact TP/PP/DP integer split for 1024 GPUs) | Constraint Satisfaction / Integer Programming | Variables are strictly discrete integers (you can't have 1.5 pipeline stages). Search space is massive but highly constrained. | Google `ortools.sat` (CP-SAT) | The industry gold standard for Constraint Programming. It uses advanced bounding heuristics to instantly eliminate invalid configurations without evaluating them. |
| **Fleet Routing & Economics** (e.g., routing workloads across datacenters to minimize carbon given varying electricity prices) | Linear Programming (LP) | Optimization of a linear objective function subject to linear equality and inequality constraints. | Google `ortools.linear_solver` (GLOP) | Built specifically for large-scale capacity planning and flow problems. |
| **Queueing Boundaries** (e.g. M/M/c walls) | Exhaustive Grid Search | Highly non-linear discontinuities where gradients fail. | `scipy.optimize.brute` | Safely evaluates small grids without hanging on mathematical infinities. |

## 2. Abstraction Interface (`OptimizerProtocol`)

To ensure the CLI and Python API remain clean, users will never import SciPy or OR-Tools directly. They will interact with a unified interface:

```python
class OptimizerProtocol(Protocol):
    def compile(self) -> Any:
        """Translates mlsysim physics equations into the specific solver's native format."""
        ...

    def solve(self, **kwargs) -> OptimizationResult:
        """Executes the external C++ solver and returns a standardized result."""
        ...
```

## 3. Pedagogical Context for Assumptions

"Magic numbers" (like `overlap_efficiency=0.85`) currently hidden deep in solver methods are antithetical to an academic textbook tool.

We will introduce a `SystemAssumption` wrapper class that strictly defines *why* a constraint or efficiency exists.

## 4. Execution Roadmap

1. **Phase 1 (Foundation):** Establish the `OptimizerProtocol` and the `SystemAssumption` data structures. (COMPLETED)
2. **Phase 2 (Continuous & Exhaustive):** Implement `ScipyBackend` and `ExhaustiveBackend`. Refactor `BatchingOptimizer`. (COMPLETED)
3. **Phase 3 (Discrete):** Implement `ORToolsBackend` for linear/integer programming. (COMPLETED)
4. **Phase 4 (Trace Mode - Future):** Implement "Show Your Work" trace outputs in the optimization results so students can read the mathematical proof of *why* a configuration was chosen, preserving the strictly analytical nature of the tool.
