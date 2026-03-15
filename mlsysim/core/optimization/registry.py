from typing import Type
from .protocol import OptimizerProtocol
from .scipy_backend import ScipyBackend
from .ortools_backend import ORToolsDiscreteBackend
from .exhaustive_backend import ExhaustiveBackend

class OptimizationRegistry:
    """
    Routes the optimization request to the correct mathematical backend.
    """
    _backends = {
        "continuous": ScipyBackend,
        "discrete": ORToolsDiscreteBackend,
        "exhaustive": ExhaustiveBackend,
    }

    @classmethod
    def get_backend(cls, mathematical_class: str) -> OptimizerProtocol:
        """
        Retrieves an uninitialized backend based on the mathematical shape of the problem.
        
        Args:
            mathematical_class: 'continuous' (for SciPy) or 'discrete' (for OR-Tools)
        """
        if mathematical_class not in cls._backends:
            raise ValueError(f"Unknown optimization class: {mathematical_class}. Valid options: {list(cls._backends.keys())}")
        
        backend_class = cls._backends[mathematical_class]
        return backend_class()
