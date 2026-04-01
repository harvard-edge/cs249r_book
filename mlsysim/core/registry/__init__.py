from typing import List, Any, Optional

from .plugin_manager import hardware_registry, model_registry, constants_registry
from .plugin_manager import Registry as PluginRegistry


class Registry:
    """
    Base class for registries that provides a coherent way to list and sort items.
    Used by Hardware, Models, and other static registries.
    """

    @classmethod
    def list(cls, sort_by: Optional[str] = None, reverse: bool = False) -> List[Any]:
        """
        Returns a list of all registry items, optionally sorted.
        """
        items = []
        for attr_name in dir(cls):
            if attr_name.startswith("_") or attr_name == "list":
                continue

            attr = getattr(cls, attr_name)

            # Skip sub-classes (hierarchical registries) unless they are items themselves
            if isinstance(attr, type) and issubclass(attr, Registry):
                continue

            # Skip callable methods
            if callable(attr):
                continue

            items.append(attr)

        if sort_by:
            def get_deep_attr(obj, path):
                parts = path.split('.')
                val = obj
                for p in parts:
                    val = getattr(val, p, 0)
                return val

            items.sort(key=lambda x: get_deep_attr(x, sort_by), reverse=reverse)

        return items


__all__ = [
    "hardware_registry", "model_registry", "constants_registry",
    "PluginRegistry", "Registry",
]
