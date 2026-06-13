from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.units import ureg
from ..core.types import Metadata, Quantity, require_dimensionality, require_unit_family


class PlatformEnvelope(BaseModel):
    """Abstract deployment envelope (RAM, storage, latency budget)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    ram: Quantity
    storage: Quantity
    typical_latency_budget: Quantity
    latency_range_ms: str | None = None
    ram_range: str | None = None
    storage_range: str | None = None
    tdp_range_w: str | None = None
    compute_threshold: Quantity | None = None
    bandwidth_threshold: Quantity | None = None
    power_threshold: Quantity | None = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("ram", "storage", mode="after")
    @classmethod
    def _validate_capacity_fields(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")

    @field_validator("typical_latency_budget", mode="after")
    @classmethod
    def _validate_latency_budget(cls, v):
        return require_dimensionality(v, ureg.second, "typical_latency_budget")

    @field_validator("compute_threshold", mode="after")
    @classmethod
    def _validate_compute_threshold(cls, v):
        return require_unit_family(
            v,
            ureg.flop / ureg.second,
            "compute_threshold",
            "operation",
        )

    @field_validator("bandwidth_threshold", mode="after")
    @classmethod
    def _validate_bandwidth_threshold(cls, v):
        return require_unit_family(
            v,
            ureg.byte / ureg.second,
            "bandwidth_threshold",
            "data",
        )

    @field_validator("power_threshold", mode="after")
    @classmethod
    def _validate_power_threshold(cls, v):
        return require_dimensionality(v, ureg.watt, "power_threshold")
