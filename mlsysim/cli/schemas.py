from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Optional, Any, List
import yaml
from pathlib import Path

from mlsysim.models.registry import Models
from mlsysim.hardware.registry import Hardware
from mlsysim.core.registry import Registry
from mlsysim.hardware.types import HardwareNode
from mlsysim.models.types import TransformerWorkload, CNNWorkload, SSMWorkload, DiffusionWorkload, Workload

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

def _resolve_hardware(name_or_path: str) -> HardwareNode:
    """Resolves a hardware node from the Zoo or a local YAML file."""
    if name_or_path.endswith((".yaml", ".yml")):
        p = Path(name_or_path)
        if not p.exists():
            raise ValueError(f"Hardware YAML not found: {p}")
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        return HardwareNode(**data)
        
    h_obj = _find_flat(Hardware, name_or_path)
    if not h_obj:
        raise ValueError(f"Hardware '{name_or_path}' not found in the MLSysZoo. (Check case and spelling)")
    return h_obj

def _resolve_model(name_or_path: str) -> Workload:
    """Resolves a model from the Zoo or a local YAML file."""
    if name_or_path.endswith((".yaml", ".yml")):
        p = Path(name_or_path)
        if not p.exists():
            raise ValueError(f"Model YAML not found: {p}")
        with open(p, "r") as f:
            data = yaml.safe_load(f)
            
        arch = data.get("architecture", "").lower()
        if "transformer" in arch:
            return TransformerWorkload(**data)
        elif "cnn" in arch:
            return CNNWorkload(**data)
        elif "ssm" in arch:
            return SSMWorkload(**data)
        elif "diffusion" in arch:
            return DiffusionWorkload(**data)
        else:
            return Workload(**data)
            
    m_obj = _find_flat(Models, name_or_path)
    if not m_obj:
        raise ValueError(f"Model '{name_or_path}' not found in the MLSysZoo. (Check case and spelling)")
    return m_obj

class EvalNodeSchema(BaseModel):
    """Schema for evaluating a single node (Gate 1: Schema Validation)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model_name: str = Field(..., description="The name of the workload (e.g. Llama3_8B) or path to YAML")
    hardware_name: str = Field(..., description="The name of the hardware (e.g. H100) or path to YAML")
    batch_size: int = Field(default=1, gt=0)
    precision: str = Field(default="fp16")
    efficiency: float = Field(default=0.5, gt=0.0, le=1.0)
    
    # Hidden resolved objects populated during validation
    model_obj: Optional[Any] = None
    hardware_obj: Optional[Any] = None

    @model_validator(mode='after')
    def resolve_and_validate(self) -> 'EvalNodeSchema':
        """Gate 2: Registry Resolution."""
        self.model_obj = _resolve_model(self.model_name)
        self.hardware_obj = _resolve_hardware(self.hardware_name)
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
        self.model_obj = _resolve_model(self.workload.name)
        self.hardware_obj = _resolve_hardware(self.hardware.name)
        
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


