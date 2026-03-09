"""Pipeline composer for chaining mlsysim solvers.

Layer C of the composition architecture: a transparent Pipeline that
chains solvers, validates compatibility, and shows students the full
Demand → Supply → Consequence data flow.

The Pipeline is NOT a black box — it is a pedagogical tool that makes
the solver DAG visible. Students use `explain()` to see what flows
between each stage and `run()` to execute the chain.

Example
-------
>>> from mlsysim.core.pipeline import Pipeline
>>> from mlsysim.core.solver import ScalingSolver, DistributedSolver, EconomicsSolver
>>> pipe = Pipeline([ScalingSolver(), DistributedSolver(), EconomicsSolver()])
>>> pipe.explain()  # Shows the DAG and identifies gaps
>>> result = pipe.run(compute_budget=Q_("1e21 FLOP"), fleet=cluster)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .solver import BaseSolver


class CompositionError(Exception):
    """Raised when two solvers cannot be connected."""
    pass


class Pipeline:
    """A transparent chain of solvers for macro-level analysis.

    Parameters
    ----------
    solvers : list[BaseSolver]
        Ordered list of solvers to execute.
    name : str, optional
        Human-readable pipeline name.
    """

    def __init__(self, solvers: List[BaseSolver], name: str = "Pipeline"):
        if not solvers:
            raise ValueError("Pipeline requires at least one solver.")
        self.solvers = solvers
        self.name = name

    def explain(self) -> str:
        """Show the solver DAG: what each stage requires, produces, and where gaps exist.

        Returns a human-readable string describing the pipeline flow.
        This is the pedagogical entry point — students call this to
        understand how solvers compose.
        """
        lines = [f"═══ {self.name} ═══", ""]

        from .walls import walls_for_solver, Domain

        # Collect what has been produced so far
        produced_concepts = set()

        for i, solver in enumerate(self.solvers):
            cls = solver.__class__
            # Look up walls from the taxonomy (single source of truth)
            solver_walls = walls_for_solver(cls.__name__)
            tag = ", ".join(f"Wall {w.number}: {w.name}" for w in solver_walls) or "?"
            produces_name = cls.produces.__name__ if cls.produces else "Any"

            lines.append(f"  Stage {i+1}: {cls.__name__}")
            lines.append(f"    walls:    {tag}")
            lines.append(f"    requires: {cls.requires}")
            lines.append(f"    produces: {produces_name}")

            # Check for unmet requirements
            missing = set(cls.requires) - produced_concepts
            if missing and i > 0:
                lines.append(f"    ⚠ needs {missing} — must be provided at run()")

            # Update produced set
            produced_concepts.update(cls.requires)
            if cls.produces:
                produced_concepts.add(produces_name)

            if i < len(self.solvers) - 1:
                lines.append(f"    {'│':>4}")

        lines.append("")
        lines.append(f"  Flow: {' → '.join(s.__class__.__name__ for s in self.solvers)}")

        # Domain coverage summary
        domain_order = list(Domain)
        covered = set()
        for solver in self.solvers:
            for w in walls_for_solver(solver.__class__.__name__):
                covered.add(w.domain)
        covered_names = [d.value for d in domain_order if d in covered]
        lines.append(f"  Domains covered: {', '.join(covered_names)}")

        lines.append("")
        return "\n".join(lines)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline, passing results between stages.

        Parameters
        ----------
        **kwargs
            All inputs needed by the pipeline. Each solver's `solve()` is
            called with whatever kwargs match its signature. Results from
            earlier stages are accumulated and available to later stages.

        Returns
        -------
        dict
            Merged results from all stages, keyed by solver name.
        """
        accumulated = dict(kwargs)
        stage_results = {}

        for solver in self.solvers:
            cls = solver.__class__
            # Call solve with whatever kwargs match the signature
            import inspect
            sig = inspect.signature(solver.solve)
            valid_kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name in accumulated:
                    valid_kwargs[param_name] = accumulated[param_name]
                elif param.default is inspect.Parameter.empty:
                    # Required parameter not provided — let it fail naturally
                    pass

            result = solver.solve(**valid_kwargs)

            # Store result under solver name
            stage_results[cls.__name__] = result

            # Make result fields available to subsequent stages
            if hasattr(result, "model_fields"):
                for field_name in result.model_fields:
                    accumulated[field_name] = getattr(result, field_name)

        return stage_results

    def __repr__(self) -> str:
        solvers_str = " → ".join(s.__class__.__name__ for s in self.solvers)
        return f"Pipeline({solvers_str})"

    def __len__(self) -> int:
        return len(self.solvers)
