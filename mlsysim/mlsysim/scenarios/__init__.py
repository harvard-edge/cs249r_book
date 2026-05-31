from .registry import ReferenceStats

# ``mlsysim.scenarios.Scenarios`` is retained only for subpackage-level
# compatibility. The top-level ``mlsysim.Scenarios`` is the executable scenario
# registry from ``mlsysim.engine.scenarios``.
Scenarios = ReferenceStats

__all__ = ["ReferenceStats", "Scenarios"]
