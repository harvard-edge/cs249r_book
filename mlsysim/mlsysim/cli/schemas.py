from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from typing import Optional, Any, List
from pathlib import Path

from mlsysim.core.units import Q_, ureg, normalize_precision
from mlsysim.core.types import Quantity, require_unit_family
from mlsysim.core.loader import _load_mapping, _instantiate, load_yaml_entry
from mlsysim.models.registry import Models
from mlsysim.hardware.registry import Hardware
from mlsysim.infrastructure.registry import Infrastructure
from mlsysim.core.registry import Registry
from mlsysim.hardware.types import HardwareNode
from mlsysim.models.types import (
    CNNWorkload,
    DiffusionWorkload,
    EmbodiedWorkload,
    SparseTransformerWorkload,
    SSMWorkload,
    TransformerWorkload,
    Workload,
)

_CLI_WORKLOAD_TYPE_MAP = {
    "Workload": Workload,
    "TransformerWorkload": TransformerWorkload,
    "SparseTransformerWorkload": SparseTransformerWorkload,
    "CNNWorkload": CNNWorkload,
    "SSMWorkload": SSMWorkload,
    "DiffusionWorkload": DiffusionWorkload,
    "EmbodiedWorkload": EmbodiedWorkload,
}


def _find_flat(reg: Any, name: str) -> Any:
    """Deep-search for a leaf entry in a registry tree, respecting __registry_list_exclude__."""
    excluded = set(getattr(reg, "__registry_list_exclude__", ()))
    for k, v in vars(reg).items():
        if k in excluded or k.startswith("_"):
            continue
        if k == name and not isinstance(v, type):
            return v
        if isinstance(v, type) and issubclass(v, Registry):
            res = _find_flat(v, name)
            if res is not None:
                return res
    return None


def _decompose_bundled_hardware(hw: HardwareNode) -> HardwareNode:
    """Scale a multi-accelerator rack/appliance HardwareNode down to per-accelerator supply."""
    bundle_count = hw.accelerator_count or 1
    if bundle_count <= 1:
        return hw
    compute_updates: dict[str, Any] = {
        "peak_flops": hw.compute.peak_flops / bundle_count,
        "precision_flops": {k: v / bundle_count for k, v in hw.compute.precision_flops.items()},
    }
    mem_updates: dict[str, Any] = {
        "capacity": hw.memory.capacity / bundle_count,
        "bandwidth": hw.memory.bandwidth / bundle_count,
    }
    if hw.memory.sram_capacity is not None:
        mem_updates["sram_capacity"] = hw.memory.sram_capacity / bundle_count
    hw_updates: dict[str, Any] = {
        "compute": hw.compute.model_copy(update=compute_updates),
        "memory": hw.memory.model_copy(update=mem_updates),
        "accelerator_count": 1,
    }
    if hw.nvlink is not None:
        hw_updates["nvlink"] = hw.nvlink.model_copy(
            update={"bandwidth": hw.nvlink.bandwidth / bundle_count}
        )
    if hw.interconnect is not None:
        hw_updates["interconnect"] = hw.interconnect.model_copy(
            update={"bandwidth": hw.interconnect.bandwidth / bundle_count}
        )
    if hw.tdp is not None:
        hw_updates["tdp"] = hw.tdp / bundle_count
    if hw.unit_cost is not None:
        hw_updates["unit_cost"] = hw.unit_cost / bundle_count
    if hw.embodied_carbon_kg is not None:
        hw_updates["embodied_carbon_kg"] = hw.embodied_carbon_kg / bundle_count
    return hw.model_copy(update=hw_updates)


def _resolve_hardware(name_or_path: str) -> HardwareNode:
    """Resolves a hardware node from the Zoo or a local YAML file."""
    if name_or_path.endswith((".yaml", ".yml")):
        p = Path(name_or_path)
        if not p.exists():
            raise ValueError(f"Hardware YAML not found: {p}")
        return load_yaml_entry(p, model_cls=HardwareNode, tech_root=Hardware.Tech)

    h_obj = _find_flat(Hardware, name_or_path)
    if not isinstance(h_obj, HardwareNode):
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
        data = _load_mapping(p)
        data.pop("__key__", None)
        if "__type__" in data:
            return _instantiate(data, model_cls=None, type_map=_CLI_WORKLOAD_TYPE_MAP, tech_root=Hardware.Tech)

        arch = str(data.get("architecture", "")).lower()
        if "moe" in arch or "sparse" in arch or "num_experts" in data:
            cls = SparseTransformerWorkload
        elif "transformer" in arch:
            cls = TransformerWorkload
        elif "cnn" in arch:
            cls = CNNWorkload
        elif "ssm" in arch or "mamba" in arch:
            cls = SSMWorkload
        elif "diffusion" in arch:
            cls = DiffusionWorkload
        elif "embodied" in arch or "vla" in arch:
            cls = EmbodiedWorkload
        else:
            cls = Workload
        return _instantiate(data, model_cls=cls, type_map=_CLI_WORKLOAD_TYPE_MAP, tech_root=Hardware.Tech)

    m_obj = _find_flat(Models, name_or_path)
    if not isinstance(m_obj, Workload):
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
    _total_accelerators: int = PrivateAttr(default=1)

    @property
    def model_obj(self) -> Any:
        return self._model_obj

    @property
    def hardware_obj(self) -> Any:
        return self._hardware_obj

    @property
    def fleet_obj(self) -> Any:
        return self._fleet_obj

    @property
    def total_accelerators(self) -> int:
        return self._total_accelerators

    @model_validator(mode='after')
    def resolve_and_validate(self) -> 'MlsysPlanSchema':
        self._model_obj = _resolve_model(self.workload.name)
        raw_hw = _resolve_hardware(self.hardware.name)
        bundle_count = raw_hw.accelerator_count or 1

        if self.hardware.node_count is not None and self.hardware.accelerators_per_node is not None:
            total_accelerators = self.hardware.node_count * self.hardware.accelerators_per_node
        elif self.hardware.accelerators is not None:
            total_accelerators = self.hardware.accelerators
        else:
            total_accelerators = bundle_count
        self._total_accelerators = total_accelerators

        per_accel_hw = _decompose_bundled_hardware(raw_hw) if bundle_count > 1 else raw_hw
        self._hardware_obj = per_accel_hw if total_accelerators > 1 else raw_hw

        grid_obj = None
        if self.ops is not None:
            grid_obj = getattr(Infrastructure.Grids, self.ops.region, None) or _find_flat(
                Infrastructure.Grids, self.ops.region
            )
            if grid_obj is None:
                valid = ", ".join(
                    sorted(
                        k
                        for k, v in vars(Infrastructure.Grids).items()
                        if not k.startswith("_") and not isinstance(v, type)
                    )
                )
                raise ValueError(f"Unknown ops.region '{self.ops.region}'. Valid regions: {valid}")

        # Build Fleet whenever distributed (total_accelerators > 1) or ops economics are requested.
        if total_accelerators > 1 or self.ops is not None:
            from mlsysim.systems.types import Fleet, Node, NetworkFabric

            accelerators_per_node = (
                self.hardware.accelerators_per_node
                or (bundle_count if total_accelerators >= bundle_count else None)
                or 1
            )
            if total_accelerators % accelerators_per_node != 0:
                raise ValueError(
                    "hardware total accelerators must be divisible by hardware.accelerators_per_node "
                    f"(got {total_accelerators} and {accelerators_per_node})"
                )
            intra_node_bw = (
                self.hardware.intra_node_bw
                or (per_accel_hw.nvlink.bandwidth_per_direction if per_accel_hw.nvlink else None)
                or (per_accel_hw.interconnect.bandwidth_per_direction if per_accel_hw.interconnect else None)
                or self.hardware.fabric_bandwidth
            )
            self._fleet_obj = Fleet(
                name=f"{self.hardware.name} Fleet",
                node=Node(
                    name=f"{self.hardware.name} Node",
                    accelerator=per_accel_hw,
                    accelerators_per_node=accelerators_per_node,
                    intra_node_bw=intra_node_bw,
                ),
                count=total_accelerators // accelerators_per_node,
                fabric=NetworkFabric(name="Default Fabric", bandwidth=self.hardware.fabric_bandwidth),
                region=grid_obj,
            )

        return self
