from .protocol import OptimizerProtocol, OptimizationResult

__all__ = [
    "OptimizerProtocol",
    "OptimizationResult",
    "OptimizationRegistry",
]


def __getattr__(name):
    if name == "OptimizationRegistry":
        from .registry import OptimizationRegistry
        return OptimizationRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
