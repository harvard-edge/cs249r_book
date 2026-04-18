import logging
from typing import Type, Dict
from .solver import BaseResolver

logger = logging.getLogger(__name__)

class ResolverFactory:
    """
    Factory for creating and retrieving Solvers/Models.
    
    This acts as the entry point for the Pluggable Solvers interface.
    It automatically discovers all built-in subclasses of BaseResolver,
    and dynamically loads third-party plugins from the `mlsysim.solvers` entry point group.
    """
    _registry: Dict[str, Type[BaseResolver]] = {}
    _loaded = False

    @classmethod
    def _load_all(cls):
        if cls._loaded:
            return
            
        # 1. Register Built-ins (via __subclasses__)
        from . import solver
        def register_subclasses(base_class):
            for subclass in base_class.__subclasses__():
                if not subclass.__name__.startswith("Base"):
                    cls._registry[subclass.__name__] = subclass
                register_subclasses(subclass)
                
        register_subclasses(solver.BaseResolver)
        
        # 2. Register Third-Party Plugins
        from .registry.plugin_manager import solver_registry
        plugins = solver_registry.all()
        for name, plugin_class in plugins.items():
            if issubclass(plugin_class, solver.BaseResolver):
                cls._registry[name] = plugin_class
            else:
                logger.warning(f"Plugin '{name}' is not a subclass of BaseResolver and will be ignored.")
                
        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> Type[BaseResolver]:
        cls._load_all()
        if name not in cls._registry:
            raise KeyError(f"Resolver '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> Dict[str, Type[BaseResolver]]:
        cls._load_all()
        return cls._registry.copy()
