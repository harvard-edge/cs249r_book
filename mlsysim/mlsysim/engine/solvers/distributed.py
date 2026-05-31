"""Distributed-training solver aliases.

These names currently re-export implementations from ``engine.solver``. Keeping
the aliases separate lets future refactors move code by domain without changing
student-facing imports.
"""

from ..solver import DistributedModel, NetworkRooflineModel, ParallelismOptimizer

__all__ = ["DistributedModel", "NetworkRooflineModel", "ParallelismOptimizer"]
