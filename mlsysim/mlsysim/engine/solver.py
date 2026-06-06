"""Public import surface for MLSysIM solvers.

The implementations live in ``mlsysim.engine.solvers`` by domain. This module
keeps solver imports stable while the implementation remains split into small,
domain-oriented files.

The export list is derived mechanically from ``mlsysim.engine.solvers.__all__``
(the canonical list) so the three public surfaces — ``mlsysim.engine.solvers``,
``mlsysim.engine.solver``, and ``mlsysim.solvers`` — can never drift apart.
"""

from .solvers import *  # noqa: F401,F403 — re-export the canonical solver set
from .solvers import __all__ as _solver_all

__all__ = list(_solver_all)
