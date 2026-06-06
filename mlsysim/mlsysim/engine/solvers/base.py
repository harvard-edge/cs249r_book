from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from ..results import SolverResult

class BaseResolver(ABC):
    """Base class for all mlsysim analytical components (Models, Solvers, Optimizers).

    Each resolver declares its input requirements and output type.
    Taxonomic classification lives in ``core/walls.py``.
    """
    requires: tuple = ()
    produces: Optional[Type[SolverResult]] = None

    @abstractmethod
    def solve(self, *args, **kwargs) -> Any:
        """
        Executes the analytical solver.

        Subclasses must implement this method to provide the mathematical
        resolution of the constraints they govern.
        """
        pass

    @classmethod
    def schema(cls) -> dict:
        """Return a machine-readable summary of this resolver's interface.

        Used by the CLI (``mlsysim solvers``) and notebooks to introspect a
        resolver without instantiating it.

        Returns
        -------
        dict
            Keys: ``resolver`` (class name), ``type`` ('model' / 'solver' /
            'optimizer'), ``walls`` (the taxonomy walls this resolver
            addresses, from ``engine/walls.py``), ``requires`` (declared
            input names), and ``produces`` (result class name, or 'Any').
        """
        from ..walls import walls_for_resolver
        wall_info = [
            {"number": w.number, "name": w.name, "domain": w.domain.value}
            for w in walls_for_resolver(cls.__name__)
        ]
        return {
            "resolver": cls.__name__,
            "type": cls.resolver_type(),
            "walls": wall_info,
            "requires": cls.requires,
            "produces": cls.produces.__name__ if cls.produces else "Any",
        }

    @classmethod
    def resolver_type(cls) -> str:
        if issubclass(cls, ForwardModel): return "model"
        if issubclass(cls, BaseSolver): return "solver"
        if issubclass(cls, BaseOptimizer): return "optimizer"
        return "unknown"

    # ── Fallacies and Pitfalls (Patterson & Hennessy tradition) ──
    _fallacies: Dict[str, str] = {}

    @classmethod
    def fallacies(cls) -> Dict[str, str]:
        """Return common fallacies and pitfalls for this solver.

        Following the Hennessy & Patterson tradition, each solver declares
        the misconceptions students most commonly hold about its domain.
        Call ``solver.fallacies()`` in notebooks for pedagogical discussion.
        """
        return cls._fallacies

class ForwardModel(BaseResolver):
    """Forward-evaluating mechanistic engine (Y = f(X))."""
    pass

class BaseSolver(BaseResolver):
    """Inverse-design or diagnostic engine (X = f^-1(Y) or grad f)."""
    pass

class BaseOptimizer(BaseResolver):
    """Design-space search engine (max f(X) s.t. g(X) < c)."""
    pass

__all__ = ["BaseResolver", "ForwardModel", "BaseSolver", "BaseOptimizer"]
