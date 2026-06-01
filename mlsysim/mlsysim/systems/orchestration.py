"""Fleet orchestration scenario parameters (queueing, utilization)."""

from pydantic import BaseModel, ConfigDict


class Orchestration(BaseModel):
    """Shared cluster scheduling assumptions for scenario calculations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_cluster_utilization: float = 0.80
    queue_discipline: str = "FIFO"
    average_researcher_job_days: float = 2.0
