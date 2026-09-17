"""Agent system profile and architecture types."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality
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
