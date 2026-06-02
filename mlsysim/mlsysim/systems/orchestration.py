"""Fleet orchestration scenario parameters (queueing, utilization)."""

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Metadata


class Orchestration(BaseModel):
    """Shared cluster scheduling assumptions for scenario calculations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    target_cluster_utilization: float = 0.80
    queue_discipline: str = "FIFO"
    average_researcher_job_days: float = 2.0
    metadata: Metadata = Field(default_factory=Metadata)
