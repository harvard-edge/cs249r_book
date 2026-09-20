"""Agent system profile and architecture types."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality, require_unit_family
from ..models.types import Workload
from ..systems.types import Node


class AgentArchitecture(BaseModel):
    """Vetted agent system profile (SWE-bench coding, deliberation tree search, multi-agent fleet, streaming)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    paradigm: str  # ReAct, TreeSearch, MultiAgent, InteractiveStreaming, Flywheel
    context_window: Quantity
    working_memory: Optional[Quantity] = None
    sandbox_startup_latency: Optional[Quantity] = None
    sandbox_snapshot_restore_latency: Optional[Quantity] = None
    sandbox_memory_footprint: Optional[Quantity] = None
    max_trajectory_steps: Optional[int] = None
    reference_model: Optional[Workload] = None
    serving_node: Optional[Node] = None
    deliberation_branches: Optional[int] = None
    verifier_cost_ratio: Optional[float] = None
    coordination_overhead_beta: Optional[float] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("sandbox_startup_latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "sandbox_startup_latency")

    @field_validator("sandbox_snapshot_restore_latency", mode="after")
    @classmethod
    def _validate_restore_latency(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "sandbox_snapshot_restore_latency")

    @field_validator("sandbox_memory_footprint", mode="after")
    @classmethod
    def _validate_sandbox_memory(cls, v):
        if v is None:
            return v
        return require_unit_family(v, ureg.byte, "sandbox_memory_footprint", "data")
