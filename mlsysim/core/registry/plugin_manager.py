import sys
import logging
from typing import Dict, Any, Type, TypeVar
import importlib.metadata

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Registry:
    """
    A generic plugin registry that dynamically discovers and loads classes or constants
    from Python entry_points, allowing third parties to inject custom hardware, models, or constants.
    """
    def __init__(self, entry_point_group: str):
        self._group = entry_point_group
        self._items: Dict[str, Any] = {}
        self._loaded = False

    def _load_plugins(self):
        if self._loaded:
            return
        
        # Load from Python Entry Points (e.g., pip install mlsysim-nvidia-specs)
        try:
            # Python 3.10+ compatible
            eps = importlib.metadata.entry_points(group=self._group)
            for ep in eps:
                try:
                    plugin = ep.load()
                    self.register(ep.name, plugin)
                    logger.debug(f"Loaded plugin '{ep.name}' into '{self._group}' registry.")
                except Exception as e:
                    logger.error(f"Failed to load plugin '{ep.name}' for group '{self._group}': {e}")
        except Exception as e:
            logger.debug(f"No entry points found for {self._group}: {e}")
            
        self._loaded = True

    def register(self, name: str, item: Any):
        """Manually register an item."""
        self._items[name] = item

    def get(self, name: str) -> Any:
        """Retrieve an item by name."""
        self._load_plugins()
        if name not in self._items:
            raise KeyError(f"Item '{name}' not found in registry '{self._group}'. Available: {list(self._items.keys())}")
        return self._items[name]

    def all(self) -> Dict[str, Any]:
        """Return all registered items."""
        self._load_plugins()
        return self._items.copy()

# Global Registries
hardware_registry = Registry("mlsysim.hardware")
model_registry = Registry("mlsysim.models")
constants_registry = Registry("mlsysim.constants")
solver_registry = Registry("mlsysim.solvers")
