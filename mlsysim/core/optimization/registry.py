from typing import Type
from .protocol import OptimizerProtocol


def _load_scipy_backend():
    try:
        from .scipy_backend import ScipyBackend
        return ScipyBackend
    except ImportError:
        raise ImportError(
            "SciPy is required for continuous optimization. "
            "Install it with: pip install mlsysim[opt]"
        )


def _load_ortools_backend():
    try:
        from .ortools_backend import ORToolsDiscreteBackend
        return ORToolsDiscreteBackend
    except ImportError:
        raise ImportError(
            "OR-Tools is required for discrete optimization. "
            "Install it with: pip install mlsysim[opt]"
        )


def _load_exhaustive_backend():
    from .exhaustive_backend import ExhaustiveBackend
    return ExhaustiveBackend


class OptimizationRegistry:
    """
    Routes the optimization request to the correct mathematical backend.
    
    Backends are loaded lazily so the core package doesn't require scipy/ortools
    at import time. They are only needed when an optimizer is actually invoked.
    """
    _backend_loaders = {
        "continuous": _load_scipy_backend,
        "discrete": _load_ortools_backend,
        "exhaustive": _load_exhaustive_backend,
    }

    @classmethod
    def get_backend(cls, mathematical_class: str) -> OptimizerProtocol:
        """
        Retrieves an uninitialized backend based on the mathematical shape of the problem.
        
        Args:
            mathematical_class: 'continuous' (for SciPy), 'discrete' (for OR-Tools),
                                or 'exhaustive' (for SciPy brute-force grid search)
        """
        if mathematical_class not in cls._backend_loaders:
            raise ValueError(f"Unknown optimization class: {mathematical_class}. Valid options: {list(cls._backend_loaders.keys())}")
        
        backend_class = cls._backend_loaders[mathematical_class]()
        return backend_class()
