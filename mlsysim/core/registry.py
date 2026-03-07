# registry.py
# Base Registry Pattern for MLSys registries

from typing import List, Any, Type, Optional, Callable

class Registry:
    """
    Base class for registries that provides a coherent way to list and sort items.
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
            # Handle deep sorting (e.g. compute.peak_flops)
            def get_deep_attr(obj, path):
                parts = path.split('.')
                val = obj
                for p in parts:
                    val = getattr(val, p, 0)
                return val

            items.sort(key=lambda x: get_deep_attr(x, sort_by), reverse=reverse)
            
        return items
