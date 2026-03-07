from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, Union, Dict, Any
from ..models.registry import Models
from ..hardware.registry import Hardware
from ..infra.registry import Infra
from ..systems.registry import Fabrics
from .exceptions import OOMError

class SimulationConfig(BaseModel):
    """
    Standard schema for an ML Systems Simulation.
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
        """
        Runs a pre-simulation check to ensure the configuration isn't 
        physically impossible (e.g., OOM on start).
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
    """Helper to parse a dictionary into a validated simulation configuration."""
    return SimulationConfig.model_validate(data)
