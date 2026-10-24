"""Pipeline composer for chaining mlsysim analytical models and solvers.

Layer C of the composition architecture: a transparent Pipeline that
chains resolvers (models and solvers), validates compatibility, and 
shows students the full Demand → Supply → Consequence data flow.

The Pipeline is NOT a black box — it is an analytical tool that makes
the resolver DAG visible. Students use `explain()` to see what flows
between each stage and `run()` to execute the chain.

Example
-------
>>> from mlsysim.engine.pipeline import Pipeline
>>> from mlsysim.solvers import ScalingModel, DistributedModel, EconomicsModel
>>> pipe = Pipeline([ScalingModel(), DistributedModel(), EconomicsModel()])
>>> pipe.explain()  # Shows the DAG and identifies gaps
>>> result = pipe.run(compute_budget=Q_("1e21 FLOP"), fleet=cluster)
"""

from __future__ import annotations

from typing import Any, Dict, List
from .solvers import BaseResolver


class CompositionError(Exception):
    """Raised when two resolvers cannot be connected."""
    pass


class Pipeline:
    """A transparent chain of models and solvers for macro-level analysis.

    Parameters
    ----------
    resolvers : list[BaseResolver]
        Ordered list of models and solvers to execute.
    name : str, optional
        Human-readable pipeline name.
    """

    def __init__(self, resolvers: List[BaseResolver], name: str = "Pipeline"):
        if not resolvers:
            raise ValueError("Pipeline requires at least one resolver.")
        self.resolvers = resolvers
        self.name = name
        self.callbacks = []

    def register_callback(self, callback: Any):
        """Register a callback middleware to intercept pipeline execution."""
        self.callbacks.append(callback)

    def explain(self) -> str:
        """Show the resolver DAG: what each stage requires, produces, and where gaps exist.

        Returns a human-readable string describing the pipeline flow.
        This is the analytical entry point — engineers call this to
        understand how models and solvers compose.
        """
        lines = [f"═══ {self.name} ═══", ""]

        from .walls import walls_for_resolver, Domain

        # Collect what has been produced so far
        produced_concepts = set()

        for i, resolver in enumerate(self.resolvers):
            cls = resolver.__class__
            # Look up walls from the taxonomy (single source of truth)
            resolver_walls = walls_for_resolver(cls.__name__)
            tag = ", ".join(f"Wall {w.number}: {w.name}" for w in resolver_walls) or "?"
            produces_name = cls.produces.__name__ if cls.produces else "Any"

            lines.append(f"  Stage {i+1}: {cls.__name__}")
            lines.append(f"    walls:    {tag}")
            lines.append(f"    requires: {cls.requires}")
            lines.append(f"    produces: {produces_name}")

            # Check for unmet requirements
            missing = set(cls.requires) - produced_concepts
            if missing and i > 0:
                lines.append(f"    ⚠ needs {missing} — must be provided at run()")

            # Update produced set with both the result class name, its fields,
            # and canonical pipeline aliases (e.g. ScalingResult -> workload/model).
            if cls.produces:
                produced_concepts.add(produces_name)
                if hasattr(cls.produces, "model_fields"):
                    produced_concepts.update(cls.produces.model_fields.keys())
                    if "optimal_parameters" in cls.produces.model_fields:
                        produced_concepts.update({"workload", "model"})
                    if "step_latency_total" in cls.produces.model_fields:
                        produced_concepts.update({"training_time", "base_training_time", "duration_days"})

            if i < len(self.resolvers) - 1:
                lines.append(f"    {'│':>4}")

        lines.append("")
        lines.append(f"  Flow: {' → '.join(s.__class__.__name__ for s in self.resolvers)}")

        # Domain coverage summary
        domain_order = list(Domain)
        covered = set()
        for resolver in self.resolvers:
            for w in walls_for_resolver(resolver.__class__.__name__):
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
            All inputs needed by the pipeline. Each resolver's `solve()` is
            called with whatever kwargs match its signature. Results from
            earlier stages are accumulated and available to later stages.

        Returns
        -------
        dict
            Merged results from all stages, keyed by resolver name.

        Notes
        -----
        Stage wiring is **by parameter name**: each resolver receives only the
        kwargs matching its ``solve()`` signature, and after each stage the
        result model's fields are flattened into the shared namespace so a
        later stage can consume them. Common concept aliases (``workload`` ↔
        ``model``, ``ScalingResult.optimal_parameters`` → ``Workload``, and
        default ``duration_days``) are bridged automatically.
        """
        import inspect
        from ..core.units import Q_
        from ..models.types import Workload

        accumulated = dict(kwargs)
        if "workload" in accumulated and "model" not in accumulated:
            accumulated["model"] = accumulated["workload"]
        elif "model" in accumulated and "workload" not in accumulated:
            accumulated["workload"] = accumulated["model"]

        stage_results = {}

        for resolver in self.resolvers:
            cls = resolver.__class__
            sig = inspect.signature(resolver.solve)
            valid_kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param_name in accumulated:
                    valid_kwargs[param_name] = accumulated[param_name]
                elif param_name == "duration_days" and param.default is inspect.Parameter.empty:
                    valid_kwargs["duration_days"] = accumulated.get("duration_days", 30.0)
                elif param.default is inspect.Parameter.empty:
                    # Required parameter not provided — let it fail naturally
                    pass

            result = resolver.solve(**valid_kwargs)

            # Store result under class name
            stage_name = cls.__name__
            stage_results[stage_name] = result

            # Invoke callbacks
            for callback in self.callbacks:
                if hasattr(callback, "on_stage_end"):
                    callback.on_stage_end(stage_name=stage_name, result=result)

            # Make result fields available to subsequent stages
            result_type = type(result)
            if hasattr(result_type, "model_fields"):
                for field_name in result_type.model_fields:
                    accumulated[field_name] = getattr(result, field_name)
                if "optimal_parameters" in result_type.model_fields and "model" not in accumulated:
                    synth = Workload(
                        name="ScalingOptimalWorkload",
                        params=getattr(result, "optimal_parameters"),
                        inference_flops=getattr(result, "optimal_parameters").to("param").magnitude * 2 * Q_("1 flop"),
                    )
                    accumulated["model"] = synth
                    accumulated["workload"] = synth

        return stage_results

    def __repr__(self) -> str:
        resolvers_str = " → ".join(s.__class__.__name__ for s in self.resolvers)
        return f"Pipeline({resolvers_str})"

    def __len__(self) -> int:
        return len(self.resolvers)
