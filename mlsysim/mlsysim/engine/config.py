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
        If getattr(Models, model) and getattr(Hardware, hardware) resolve, require
        model.size_in_bytes() <= hardware.memory.capacity; otherwise return self unchanged.

        Names are direct attributes of the registry roots. The workload default precision is
        used by size_in_bytes, independently of this configuration's precision field.

        The method does not provide nested registry-path resolution, batch activations, KV cache, optimizer state,
        distributed sharding, and full input validation. An unchanged result is not proof the
        configuration fits.

        Examples
        --------
        >>> config = SimulationConfig(model="unknown_model", hardware="unknown_device")
        >>> config.validate_physical_feasibility() is config
        True
        """
        # 1. Resolve registry items
        m_obj = getattr(Models, self.model, None)
        h_obj = getattr(Hardware, self.hardware, None)
        
        if not m_obj or not h_obj:
            return self # Let the solver handle missing objects with better errors
            
        # 2. Check basic OOM (Weights only)
        weight_size = m_obj.size_in_bytes()
        if weight_size > h_obj.memory.capacity:
             raise ValueError(
                 f"Configuration Infeasible: {self.model} weights ({weight_size.to('GB')}) "
                 f"exceed {self.hardware} capacity ({h_obj.memory.capacity.to('GB')})."
             )
             
        return self

def load_config(data: Dict[str, Any]) -> SimulationConfig:
    """Helper to parse a dictionary into a validated analytical configuration."""
    return SimulationConfig.model_validate(data)
