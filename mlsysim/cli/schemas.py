from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, Any, List

# ... (rest of the file logic to remain mostly same, I need to be precise, let's look at it)
from mlsysim.models.registry import Models
from mlsysim.hardware.registry import Hardware
from mlsysim.core.registry import Registry

def _find_flat(reg: Any, name: str) -> Any:
    """Helper to do a deep search for a name in the registry tree."""
    for k, v in vars(reg).items():
        if k == name and not isinstance(v, type):
            return v
        if isinstance(v, type) and issubclass(v, Registry):
            res = _find_flat(v, name)
            if res: 
                return res
    return None

class EvalNodeSchema(BaseModel):
    """Schema for evaluating a single node (Gate 1: Schema Validation)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(..., description="The name of the workload (e.g. Llama3_8B)")
    hardware_name: str = Field(..., description="The name of the hardware (e.g. H100)")
    batch_size: int = Field(default=1, gt=0)
    precision: str = Field(default="fp16")
    efficiency: float = Field(default=0.5, gt=0.0, le=1.0)
    
    # Hidden resolved objects populated during validation
    model_obj: Optional[Any] = None
    hardware_obj: Optional[Any] = None

    @model_validator(mode='after')
    def resolve_and_validate(self) -> 'EvalNodeSchema':
        """Gate 2: Registry Resolution."""
        # 1. Resolve Model
        m_obj = _find_flat(Models, self.model_name)
        if not m_obj:
            raise ValueError(f"Model '{self.model_name}' not found in the MLSysZoo. (Check case and spelling)")
            
        # 2. Resolve Hardware
        h_obj = _find_flat(Hardware, self.hardware_name)
        if not h_obj:
            raise ValueError(f"Hardware '{self.hardware_name}' not found in the MLSysZoo. (Check case and spelling)")

        # 3. Store the resolved objects for the solver to use
        self.model_obj = m_obj
        self.hardware_obj = h_obj
            
        return self


class WorkloadConfig(BaseModel):
    name: str
    batch_size: int = 1
    seq_len: int = 2048

class HardwareConfig(BaseModel):
    name: str
    nodes: int = 1
    precision: str = "fp16"
    efficiency: float = 0.5

class OpsConfig(BaseModel):
    region: str = "US_Avg"
    duration_days: float = 30.0

class AssertionConfig(BaseModel):
    metric: str
    max: Optional[float] = None
    min: Optional[float] = None

class ConstraintsConfig(BaseModel):
    asserts: List[AssertionConfig] = Field(default_factory=list, alias="assert")

class MlsysPlanSchema(BaseModel):
    version: str
    name: str
    workload: WorkloadConfig
    hardware: HardwareConfig
    ops: Optional[OpsConfig] = None
    constraints: Optional[ConstraintsConfig] = None
    
    model_obj: Optional[Any] = None
    hardware_obj: Optional[Any] = None
    fleet_obj: Optional[Any] = None
    
    @model_validator(mode='after')
    def resolve_and_validate(self) -> 'MlsysPlanSchema':
        # 1. Resolve Model
        m_obj = _find_flat(Models, self.workload.name)
        if not m_obj:
            raise ValueError(f"Model '{self.workload.name}' not found in the MLSysZoo. (Check case and spelling)")
            
        # 2. Resolve Hardware
        h_obj = _find_flat(Hardware, self.hardware.name)
        if not h_obj:
            raise ValueError(f"Hardware '{self.hardware.name}' not found in the MLSysZoo. (Check case and spelling)")

        self.model_obj = m_obj
        self.hardware_obj = h_obj
        
        # 3. Resolve Fleet if nodes > 1
        if self.hardware.nodes > 1:
            from mlsysim.systems.types import Fleet, Node, NetworkFabric
            from mlsysim.core.constants import Q_
            
            # Simple automatic fleet construction for now
            # In a full implementation, fabric would be configurable in the YAML
            self.fleet_obj = Fleet(
                name=f"{self.hardware.name} Fleet",
                node=Node(
                    name=f"{self.hardware.name} Node",
                    accelerator=self.hardware_obj,
                    accelerators_per_node=8,
                    intra_node_bw=Q_("900 GB/s")
                ),
                count=max(self.hardware.nodes // 8, 1),
                fabric=NetworkFabric(name="Default IB", bandwidth=Q_("50 GB/s"))
            )
            
        return self


