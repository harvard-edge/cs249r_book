"""MLOps assumption registries (monitoring thresholds, drift detection)."""

from ..core.registry import Registry
from .monitoring import Monitoring
from .runtime import MemoryProtection, RuntimeOverheads
from .training import TrainingRunOverheads


class Ops(Registry):
    """Registry namespace for Ops."""
    Monitoring = Monitoring
    RuntimeOverheads = RuntimeOverheads
    MemoryProtection = MemoryProtection
    TrainingRunOverheads = TrainingRunOverheads
