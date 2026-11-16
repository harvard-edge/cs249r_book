from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Dict, Any
from ..models.registry import Models
from ..hardware.registry import Hardware

class SimulationConfig(BaseModel):
    """
    Standard schema for an ML systems analytical modeling run.
    Can be loaded from YAML, JSON, or Python Dicts.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identifiers (can be names from registry or full objects)
    model: str = Field(description="Name of the model (e.g., 'GPT3', 'ResNet50')")
    hardware: str = Field(description="Name of the accelerator (e.g., 'A100', 'H100')")
    
    # Execution Parameters
    batch_size: int = 1
    precision: str = "fp16"
    efficiency: float = 0.5
    
    # Scale Parameters
    fleet_size: int = 1
    fabric: str = "100GbE"
    
    # Environment
    region: str = "US_Avg"
    duration_days: float = 30.0

    @model_validator(mode='after')
    def validate_physical_feasibility(self) -> 'SimulationConfig':
        """Apply a weight-only capacity check when root registry names resolve.

        Parameters
        ----------

        Returns
        -------
        SimulationConfig
            The same configuration on success or when either registry name is unresolved. Raises
            ValueError only when resolved weight storage exceeds resolved accelerator capacity.

        Notes
        -----
        Equation: When single-device (``fleet_size == 1``) and both ``model`` and
        ``hardware`` resolve from the ``Models`` and ``Hardware`` registries,
        require ``model.size_in_bytes(precision) <= hardware.memory.capacity``.

        Assumptions: Registry identifiers match either top-level attributes, dotted
        sub-registry paths, or leaf entry names under ``Models`` and ``Hardware``
        (excluding ``Hardware.Tech``). Multi-device configurations defer memory
        sharding checks to distributed solvers.

        Excludes: Batch activations, KV cache, optimizer state, and distributed
        sharding; passing this check does not guarantee runtime memory fit.

        References
        ----------
        Derivation: Weight bytes at the requested precision provide a necessary
        lower bound on single-accelerator memory capacity.

        Examples
        --------
        >>> config = SimulationConfig(model="unknown_model", hardware="unknown_device")
        >>> config.validate_physical_feasibility() is config
        True
        """
        if self.fleet_size > 1:
            return self

        def _resolve(root: Any, name: str, skip_attrs: tuple[str, ...] = ()) -> Any:
            cur = root
            for part in name.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    break
            if cur is not None:
                return cur
            for attr in dir(root):
                if attr.startswith("_") or attr in skip_attrs:
                    continue
                sub = getattr(root, attr, None)
                found = getattr(sub, name, None)
                if found is not None:
                    return found
            return None

        m_obj = _resolve(Models, self.model)
        h_obj = _resolve(Hardware, self.hardware, skip_attrs=("Tech",))
        if not m_obj or not h_obj or not hasattr(m_obj, "size_in_bytes") or not hasattr(h_obj, "memory"):
            return self

        weight_size = m_obj.size_in_bytes(self.precision)
        if weight_size > h_obj.memory.capacity:
            raise ValueError(
                f"Configuration Infeasible: {self.model} weights ({weight_size.to('GB')}) "
                f"exceed {self.hardware} capacity ({h_obj.memory.capacity.to('GB')})."
            )

        return self

def load_config(data: Dict[str, Any]) -> SimulationConfig:
    """Helper to parse a dictionary into a validated analytical configuration."""
    return SimulationConfig.model_validate(data)
