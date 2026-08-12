from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from typing import Optional, Any, List
import yaml
from pathlib import Path

from mlsysim.core.units import Q_, ureg
from mlsysim.core.types import Quantity, require_unit_family
from mlsysim.core.units import normalize_precision
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
    """Resolves a model from the Zoo, a local YAML file, or the Hugging Face Hub (hf://)."""
    if name_or_path.startswith("hf://"):
        from mlsysim.models.importer import import_hf_model
        model_id = name_or_path[5:]  # Strip hf://
        return import_hf_model(model_id)

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
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    
    model_name: str = Field(..., description="The name of the workload (e.g. Llama3_8B) or path to YAML")
    hardware_name: str = Field(..., description="The name of the hardware (e.g. H100) or path to YAML")
    batch_size: int = Field(default=1, gt=0)
    precision: str = Field(default="fp16")
    efficiency: float = Field(default=0.5, gt=0.0, le=1.0)
    
    # Resolved objects are runtime-only and must not appear in public schemas.
    _model_obj: Any = PrivateAttr(default=None)
    _hardware_obj: Any = PrivateAttr(default=None)

    @property
    def model_obj(self) -> Any:
        return self._model_obj

    @property
    def hardware_obj(self) -> Any:
        return self._hardware_obj

    @field_validator("precision", mode="after")
    @classmethod
    def _validate_precision(cls, v):
        return normalize_precision(v)

    @model_validator(mode='after')
    def resolve_and_validate(self) -> 'EvalNodeSchema':
        """Gate 2: Registry Resolution."""
        self._model_obj = _resolve_model(self.model_name)
        self._hardware_obj = _resolve_hardware(self.hardware_name)
        return self


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    batch_size: int = Field(default=1, gt=0)
    seq_len: int = Field(default=2048, gt=0)

class HardwareConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    accelerators: Optional[int] = Field(default=None, gt=0)
    node_count: Optional[int] = Field(default=None, gt=0)
    accelerators_per_node: Optional[int] = Field(default=None, gt=0)
    intra_node_bw: Optional[Quantity] = None
    fabric_bandwidth: Quantity = Field(default_factory=lambda: Q_("50 GB/s"))
    precision: str = "fp16"
    efficiency: float = Field(default=0.5, gt=0.0, le=1.0)

    @field_validator("precision", mode="after")
    @classmethod
    def _validate_precision(cls, v):
        return normalize_precision(v)

    @field_validator("intra_node_bw", "fabric_bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth_fields(cls, v, info):
        return require_unit_family(v, ureg.bit / ureg.second, info.field_name, "data")

    @property
    def total_accelerators(self) -> int:
        if self.node_count is not None and self.accelerators_per_node is not None:
            return self.node_count * self.accelerators_per_node
        if self.accelerators is not None:
            return self.accelerators
        return 1

    @model_validator(mode="after")
    def _validate_topology_consistency(self) -> "HardwareConfig":
        if self.node_count is not None and self.accelerators_per_node is None:
            raise ValueError("hardware.node_count requires hardware.accelerators_per_node")

        derived_total = None
        if self.node_count is not None and self.accelerators_per_node is not None:
            derived_total = self.node_count * self.accelerators_per_node

        explicit_total = self.accelerators
        if derived_total is not None and explicit_total is not None and derived_total != explicit_total:
            raise ValueError("hardware.node_count * hardware.accelerators_per_node must equal hardware.accelerators")
        return self

class OpsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str = "US_Avg"
    duration_days: float = Field(default=30.0, gt=0.0)

class AssertionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    max: Optional[float] = None
    min: Optional[float] = None

class ConstraintsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asserts: List[AssertionConfig] = Field(default_factory=list, alias="assert")

class MlsysPlanSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    version: str
    name: str
    workload: WorkloadConfig
    hardware: HardwareConfig
    ops: Optional[OpsConfig] = None
    constraints: Optional[ConstraintsConfig] = None
    
    # Resolved objects are runtime-only and must not appear in public schemas.
    _model_obj: Any = PrivateAttr(default=None)
    _hardware_obj: Any = PrivateAttr(default=None)
    _fleet_obj: Any = PrivateAttr(default=None)

    @property
    def model_obj(self) -> Any:
        return self._model_obj

    @property
    def hardware_obj(self) -> Any:
        return self._hardware_obj

    @property
    def fleet_obj(self) -> Any:
        return self._fleet_obj
    
    @model_validator(mode='after')
    def resolve_and_validate(self) -> 'MlsysPlanSchema':
        self._model_obj = _resolve_model(self.workload.name)
        self._hardware_obj = _resolve_hardware(self.hardware.name)
        
        # 3. Resolve Fleet if total accelerators > 1.
        total_accelerators = self.hardware.total_accelerators
        if total_accelerators > 1:
            from mlsysim.systems.types import Fleet, Node, NetworkFabric

            # Step 1: prefer an explicit plan topology, then hardware aggregate
            # metadata such as GB200 NVL72, then single-accelerator nodes.
            accelerators_per_node = (
                self.hardware.accelerators_per_node
                or self._hardware_obj.accelerator_count
                or 1
            )
            # Step 2: refuse to floor partial nodes; that would silently change
            # the number of accelerators represented by the plan.
            if total_accelerators % accelerators_per_node != 0:
                raise ValueError(
                    "hardware total accelerators must be divisible by hardware.accelerators_per_node "
                    f"(got {total_accelerators} and {accelerators_per_node})"
                )
            # Step 3: use the most specific bandwidth source available.
            intra_node_bw = (
                self.hardware.intra_node_bw
                or (self._hardware_obj.nvlink.bandwidth if self._hardware_obj.nvlink else None)
                or (self._hardware_obj.interconnect.bandwidth if self._hardware_obj.interconnect else None)
                or self.hardware.fabric_bandwidth
            )
            # Step 4: compile the declarative plan into the Fleet object used by
            # distributed solvers.
            self._fleet_obj = Fleet(
                name=f"{self.hardware.name} Fleet",
                node=Node(
                    name=f"{self.hardware.name} Node",
                    accelerator=self._hardware_obj,
                    accelerators_per_node=accelerators_per_node,
                    intra_node_bw=intra_node_bw,
                ),
                count=total_accelerators // accelerators_per_node,
                fabric=NetworkFabric(name="Default Fabric", bandwidth=self.hardware.fabric_bandwidth),
            )
            
        return self
