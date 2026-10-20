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
    shutter_type: Literal["rolling", "global"]  # the color/image stream the resolution describes
    # Stereo depth cameras carry separate depth imagers whose shutter can differ
    # from the color sensor's; None for plain image sensors.
    depth_shutter_type: Optional[Literal["rolling", "global"]] = None
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
    # Datasheet ranging and timing specs (all upper bounds or 1-sigma typicals).
    range_precision_min: Optional[Quantity] = None   # 1-sigma range noise, best case
    range_precision_max: Optional[Quantity] = None   # 1-sigma range noise, worst case
    range_accuracy: Optional[Quantity] = None        # mean range bias, Lambertian targets
    time_sync_accuracy: Optional[Quantity] = None    # PTP / gPTP / PPS error bound
    internal_clock_drift: Optional[Quantity] = None  # free-running clock drift (ppm)
    laser_wavelength: Optional[Quantity] = None      # emitter wavelength (nm)
    # Added 2026-10-20: bias on retroreflective targets, which datasheets state
    # separately, and the vertical field of view.
    range_accuracy_retroreflective: Optional[Quantity] = None  # mean range bias, retroreflectors
    vertical_fov: Optional[Quantity] = None          # full vertical field of view

    @field_validator("laser_wavelength", mode="after")
    @classmethod
    def _validate_wavelength(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter, "laser_wavelength")

    @field_validator("range_precision_min", "range_precision_max", "range_accuracy",
                     "range_accuracy_retroreflective", mode="after")
    @classmethod
    def _validate_range_specs(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter, info.field_name)

    @field_validator("vertical_fov", mode="after")
    @classmethod
    def _validate_fov(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.radian, "vertical_fov")

    @field_validator("time_sync_accuracy", mode="after")
    @classmethod
    def _validate_sync(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "time_sync_accuracy")

    @field_validator("internal_clock_drift", mode="after")
    @classmethod
    def _validate_drift(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.dimensionless, "internal_clock_drift")

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
    # Datasheet error terms (typical values). Noise densities integrate as
    # sqrt(T); offsets integrate as T (rate) or T^2 (acceleration to position).
    gyro_noise_density: Optional[Quantity] = None      # deg/s/sqrt(Hz)
    gyro_zero_rate_offset: Optional[Quantity] = None   # deg/s
    accel_noise_density: Optional[Quantity] = None     # g/sqrt(Hz) (micro-g/sqrt(Hz) on sheets)
    accel_zero_g_offset: Optional[Quantity] = None     # g (mg on sheets)

    @field_validator("sample_rate", mode="after")
    @classmethod
    def _validate_sample_rate(cls, v):
        return require_dimensionality(v, 1 / ureg.second, "sample_rate")

    @field_validator("gyro_noise_density", mode="after")
    @classmethod
    def _validate_gyro_noise(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.degree / ureg.second / ureg.hertz**0.5, "gyro_noise_density")

    @field_validator("gyro_zero_rate_offset", mode="after")
    @classmethod
    def _validate_gyro_offset(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.degree / ureg.second, "gyro_zero_rate_offset")

    @field_validator("accel_noise_density", mode="after")
    @classmethod
    def _validate_accel_noise(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter / ureg.second**2 / ureg.hertz**0.5, "accel_noise_density")

    @field_validator("accel_zero_g_offset", mode="after")
    @classmethod
    def _validate_accel_offset(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.meter / ureg.second**2, "accel_zero_g_offset")


class SafetyLaserScanner(Sensor):
    """Certified planar safety laser scanner (an AOPDDR type-tested under IEC 61496-3).

    Its safety output reports only that something occupies a configured
    protective field; it does not classify what it detects. The response time
    is the scanner's own contribution to the certified stopping chain.
    """

    protective_field_range: Quantity   # maximum protective-field radius for the stated object size
    response_time: Quantity            # ON-to-OFF response at the stated scan count and mode
    scan_angle: Quantity
    iec61496_type: str                 # e.g. "Type 3"
    performance_level: str             # ISO 13849-1, e.g. "PL d, Category 3"
    configuration: str                 # mode and object size the stored values describe
    # Added 2026-10-20: the slower standard-mode response at the same scan count,
    # for scanners whose datasheet states two modes (None where it states one).
    response_time_standard_mode: Optional[Quantity] = None

    @field_validator("protective_field_range", mode="after")
    @classmethod
    def _validate_field_range(cls, v):
        return require_dimensionality(v, ureg.meter, "protective_field_range")

    @field_validator("response_time", mode="after")
    @classmethod
    def _validate_response(cls, v):
        return require_dimensionality(v, ureg.second, "response_time")

    @field_validator("scan_angle", mode="after")
    @classmethod
    def _validate_scan_angle(cls, v):
        return require_dimensionality(v, ureg.degree, "scan_angle")

    @field_validator("response_time_standard_mode", mode="after")
    @classmethod
    def _validate_response_standard(cls, v):
        return v if v is None else require_dimensionality(v, ureg.second, "response_time_standard_mode")
