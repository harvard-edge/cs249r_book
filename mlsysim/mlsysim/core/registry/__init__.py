from typing import ClassVar, List, Any, Optional

from .plugin_manager import hardware_registry, model_registry, constants_registry
from .plugin_manager import Registry as PluginRegistry


class Registry:
    """
    Base class for registries that provides a coherent way to list and sort items.
    Used by Hardware, Models, and other static registries.
    """

    __registry_list_exclude__: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def list(cls, sort_by: Optional[str] = None, reverse: bool = False) -> List[Any]:
        """
        Returns a list of all registry items, optionally sorted.
        Recurses into nested Registry subclasses (e.g. Hardware.Cloud.*).
        """
        items: List[Any] = []
        for attr_name in dir(cls):
            if attr_name.startswith("_") or attr_name == "list" or attr_name in cls.__registry_list_exclude__:
                continue

            attr = getattr(cls, attr_name)

            if isinstance(attr, type) and issubclass(attr, Registry):
                items.extend(attr.list())
                continue

            if callable(attr):
                continue

            items.append(attr)

        # Deduplicate alias attributes pointing at the same entry, by object
        # identity. (No aliases exist today — hard-delete policy — but the
        # guard keeps list() correct if one is ever reintroduced.)
        seen_ids = set()
        unique_items = []
        for item in items:
            if id(item) not in seen_ids:
                seen_ids.add(id(item))
                unique_items.append(item)
        items = unique_items

        if sort_by:
            def normalize_sort_value(val):
                if val is None:
                    return 0
                if hasattr(val, "to_base_units"):
                    try:
                        return val.to_base_units().magnitude
                    except Exception:
                        pass
                return getattr(val, "magnitude", val)

            def get_deep_attr(obj, path):
                parts = path.split('.')
                val = obj
                for p in parts:
                    val = getattr(val, p, 0)
                return normalize_sort_value(val)

            items.sort(key=lambda x: get_deep_attr(x, sort_by), reverse=reverse)

        return items


__all__ = [
    "hardware_registry", "model_registry", "constants_registry",
    "PluginRegistry", "Registry",
]
