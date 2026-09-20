"""Sensor subsystem types for physical AI and embodied robotics platforms."""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality, require_unit_family


class Sensor(BaseModel):
    """Base sensor type for cyber-physical perception systems."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    nominal_latency: Optional[Quantity] = None
    interface: Optional[str] = None
    nominal_power: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("nominal_latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "nominal_latency")

    @field_validator("nominal_power", mode="after")
    @classmethod
    def _validate_power(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.watt, "nominal_power")


class CameraSensor(Sensor):
    """Optical image sensor (CMOS rolling or global shutter, stereo depth)."""

    resolution_width: int
    resolution_height: int
    frame_rate: Quantity
    shutter_type: Literal["rolling", "global"]
    exposure_time: Optional[Quantity] = None
    readout_time: Optional[Quantity] = None

    @field_validator("frame_rate", mode="after")
    @classmethod
    def _validate_frame_rate(cls, v):
        return require_dimensionality(v, 1 / ureg.second, "frame_rate")

    @field_validator("exposure_time", "readout_time", mode="after")
    @classmethod
    def _validate_sensor_times(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, info.field_name)


class LiDARSensor(Sensor):
    """Time-of-flight LiDAR sensor (mechanical spinning or solid-state)."""

    channels: int
    scan_rate: Quantity
    max_range: Optional[Quantity] = None
    points_per_second: Optional[Quantity] = None

    @field_validator("scan_rate", mode="after")
    @classmethod
    def _validate_scan_rate(cls, v):
        return require_dimensionality(v, 1 / ureg.second, "scan_rate")

    @field_validator("max_range", mode="after")
    @classmethod
    def _validate_range(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter, "max_range")

    @field_validator("points_per_second", mode="after")
    @classmethod
    def _validate_pps(cls, v):
        if v is None:
            return v
        return require_unit_family(v, 1 / ureg.second, "points_per_second", "frequency")


class IMUSensor(Sensor):
    """Inertial measurement unit (accelerometer + gyroscope)."""

    dofs: int = 6
    sample_rate: Quantity

    @field_validator("sample_rate", mode="after")
    @classmethod
    def _validate_sample_rate(cls, v):
        return require_dimensionality(v, 1 / ureg.second, "sample_rate")
