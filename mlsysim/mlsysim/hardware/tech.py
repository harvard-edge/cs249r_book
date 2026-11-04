"""Hardware technology-class reference facts (exposed as ``Hardware.Tech``).

Technology-CLASS properties — access latency, per-operation/per-byte energy, and generic
component bandwidth — that are ~constant across parts of a generation. This is the
counterpart to the per-instance specs on the Hardware.Cloud/Edge/Mobile/Tiny nodes
(capacity, bandwidth, TDP, price), which genuinely vary part-to-part.
See the project MLSysIM rules -> Canonical organization (the instance-vs-tech-class split).

The tier *data* lives as YAML under ``hardware/data/tech/<category>.yaml`` (loaded +
validated against the tier schemas below); the tier *types*, the ``Tech`` namespace, and
the instance→tech-class fallback resolver are code. Energy figures are Horowitz (2014),
45 nm — process-stamped in the names and provenance so a future process node can coexist
rather than overwrite the teaching anchor (invariant #3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality, require_unit_family
from ..core.registry import Registry
from ..core.loader import load_collection


class MemoryTier(BaseModel):
    """Memory technology tier: access latency, access energy, and generic bandwidth."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    latency: Optional[Quantity] = None            # access latency
    # Measured access latency in core clock cycles, when the source reports
    # cycles (GPU microbenchmarks do). ``latency`` then holds the derived
    # wall-clock value at the source's stated clock.
    latency_cycles: Optional[float] = None
    energy_per_access: Optional[Quantity] = None  # pJ per access
    energy_per_byte: Optional[Quantity] = None    # pJ per byte
    bandwidth: Optional[Quantity] = None          # representative interface bandwidth
    # DRAM core timings of a speed grade (command clock, activate-to-column,
    # precharge). A row conflict costs t_rp + t_rcd before column service.
    t_ck: Optional[Quantity] = None
    t_rcd: Optional[Quantity] = None
    t_rp: Optional[Quantity] = None               # per-bank precharge
    # Analog minima the standard states in nanoseconds; the clocked timings
    # above are these rounded up to whole command-clock cycles.
    t_rcd_min: Optional[Quantity] = None
    t_rp_min: Optional[Quantity] = None           # per-bank precharge minimum
    t_rpab_min: Optional[Quantity] = None         # all-bank precharge minimum
    channel_width: Optional[Quantity] = None      # data width of one channel (bits)
    # Added 2026-10-20: data width of the whole device interface, which the
    # bandwidth comparison between memory technologies turns on (one HBM stack,
    # one DDR DIMM channel).
    interface_width: Optional[Quantity] = None
    # Added 2026-10-20: per-pin data rate of a speed grade (one bit per
    # transfer, so DDR5-4800 is 4.8 Gb/s per pin). Peak bandwidth is
    # interface_width x pin_data_rate (physics.retrieval.calc_interface_bandwidth).
    pin_data_rate: Optional[Quantity] = None
    # Added 2026-10-20: a representative range across a technology class, when
    # the class has no single value (on-chip SRAM per accelerator chip).
    capacity_low: Optional[Quantity] = None
    capacity_high: Optional[Quantity] = None
    bandwidth_low: Optional[Quantity] = None
    bandwidth_high: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("latency", "t_ck", "t_rcd", "t_rp", "t_rcd_min", "t_rp_min", "t_rpab_min",
                     mode="after")
    @classmethod
    def _validate_latency(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("channel_width", "interface_width", mode="after")
    @classmethod
    def _validate_channel_width(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.bit, "interface width")

    @field_validator("capacity_low", "capacity_high", mode="after")
    @classmethod
    def _validate_capacity(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")

    @field_validator("energy_per_access", mode="after")
    @classmethod
    def _validate_energy_per_access(cls, v):
        return require_dimensionality(v, ureg.joule, "energy_per_access")

    @field_validator("energy_per_byte", mode="after")
    @classmethod
    def _validate_energy_per_byte(cls, v):
        return require_unit_family(v, ureg.joule / ureg.byte, "energy_per_byte", "data")

    @field_validator("bandwidth", "pin_data_rate", "bandwidth_low", "bandwidth_high", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v, info):
        return require_unit_family(v, ureg.byte / ureg.second, info.field_name, "data")


class StorageTier(BaseModel):
    """Generic storage/memory bandwidth tier (sequential bandwidth + optional latency)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.byte / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")


class InterconnectTier(BaseModel):
    """Interconnect technology generation: link access latency (technology-class).

    Per-instance link *bandwidth* lives on each node's IOInterconnect (it varies
    part-to-part); the access *latency* floor is a property of the interconnect
    generation, ~constant across parts, so it lives here."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    latency: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")


class ReticleField(BaseModel):
    """Lithography exposure-field limit (technology-class): the largest die a
    stepper/scanner can pattern in one exposure without stitching."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    width: Quantity
    height: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("width", "height", mode="after")
    @classmethod
    def _validate_length(cls, v, info):
        return require_dimensionality(v, ureg.meter, info.field_name)

    @property
    def area(self) -> Quantity:
        """Calculate the rectangular exposure-field area.

        Returns
        -------
        Quantity
            Exposure-field area in square millimeters.

        Notes
        -----
        Equation: area = width * height. The instance's ``width`` and ``height``
        fields are length quantities and should both be positive.
        Assumptions: a rectangular, single-exposure field with the stated dimensions.
        Excludes: wafer-edge loss, scribe lanes, yield, stitching, and usable die area.

        References
        ----------
        Derivation: the area of a rectangle is the product of its orthogonal side
        lengths; unit conversion expresses the result in square millimeters.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> field = ReticleField(name='Example field', width=Q_('2 mm'), height=Q_('3 mm'))
        >>> field.area.m_as('mm^2')
        6
        """
        return (self.width * self.height).to(ureg.millimeter ** 2)


class OpEnergy(BaseModel):
    """Per-operation energy for an arithmetic op (technology/process-class)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    energy: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("energy", mode="after")
    @classmethod
    def _validate_energy(cls, v):
        return require_unit_family(v, ureg.joule / ureg.flop, "energy", "operation")


class EffectiveComputeEnergy(BaseModel):
    """Architecture-class effective energy per FLOP."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    energy_per_flop: Quantity
    metadata: Metadata = Field(default_factory=Metadata)


class MovementEnergy(BaseModel):
    """Technology-class effective data-movement energy per byte."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    energy_per_byte: Quantity
    metadata: Metadata = Field(default_factory=Metadata)


class BatteryCell(BaseModel):
    """Standard primary-cell format (technology-class): nominal charge capacity and
    nominal voltage of a cell designation such as CR2032.

    Added 2026-10-30. A standard designation fixes chemistry and size, so capacity
    and voltage are ~constant across parts of that format; the cited datasheet pins
    one manufacturer's rating. ``energy`` is derived (capacity x voltage)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    capacity: Quantity           # nominal charge capacity (mAh)
    voltage: Quantity            # nominal voltage (V)
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("capacity", mode="after")
    @classmethod
    def _validate_capacity(cls, v):
        return require_dimensionality(v, ureg.ampere * ureg.hour, "capacity")

    @field_validator("voltage", mode="after")
    @classmethod
    def _validate_voltage(cls, v):
        return require_dimensionality(v, ureg.volt, "voltage")

    @property
    def energy(self) -> Quantity:
        """Calculate nominal stored energy from charge capacity and voltage.

        Returns
        -------
        Quantity
            Nominal stored energy in milliwatt-hours.

        Notes
        -----
        Equation: energy = capacity * voltage. The instance's ``capacity`` is a
        charge quantity (such as mAh) and ``voltage`` is an electric potential;
        both should be positive.
        Assumptions: the nominal voltage represents the voltage across the full
        rated discharge, so the charge integral is approximated by Q * V_nominal.
        Excludes: voltage variation, load-dependent capacity, temperature, aging,
        self-discharge, and conversion losses. This is not usable load energy.

        References
        ----------
        Derivation: electrical work is the integral of voltage over transferred
        charge. Holding voltage at its nominal value gives energy = charge * voltage.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> cell = BatteryCell(name='Example cell', capacity=Q_('100 mAh'), voltage=Q_('3 V'))
        >>> round(cell.energy.m_as('mWh'), 6)
        300.0
        """
        return (self.capacity * self.voltage).to(ureg.mWh)


_DATA = Path(__file__).parent / "data" / "tech"

Memory = load_collection(
    _DATA / "memory.yaml", MemoryTier, name="Memory",
    doc="On-chip memory hierarchy tiers (access latency + 45 nm access energy).",
)
Storage = load_collection(
    _DATA / "storage.yaml", StorageTier, name="Storage",
    doc="Generic storage / off-chip memory bandwidth tiers.",
)
Op = load_collection(
    _DATA / "op.yaml", OpEnergy, name="Op",
    doc="Per-operation arithmetic energy (Horowitz 2014, 45 nm).",
)
EffectiveCompute = load_collection(
    _DATA / "effective_compute.yaml", EffectiveComputeEnergy, name="EffectiveCompute",
    doc="Architecture-class effective energy per FLOP.",
)
Movement = load_collection(
    _DATA / "movement.yaml", MovementEnergy, name="Movement",
    doc="Technology-class effective data-movement energy per byte.",
)
Interconnect = load_collection(
    _DATA / "interconnect.yaml", InterconnectTier, name="Interconnect",
    doc="Interconnect technology generations (link access-latency floor).",
)

Lithography = load_collection(
    _DATA / "lithography.yaml", ReticleField, name="Lithography",
    doc="Lithography technology limits (exposure-field / reticle size).",
)

Battery = load_collection(
    _DATA / "battery.yaml", BatteryCell, name="Battery",
    doc="Standard primary-cell formats (nominal capacity + nominal voltage).",
)


class Tech(Registry):
    """Technology-class reference facts: latency / energy / generic bandwidth."""

    Memory = Memory
    Storage = Storage
    Op = Op
    EffectiveCompute = EffectiveCompute
    Movement = Movement
    Interconnect = Interconnect
    Lithography = Lithography
    Battery = Battery


def resolve_latency(instance_latency, tech_default):
    """Select an instance latency when present, otherwise its technology default.

    Parameters
    ----------
    instance_latency : Quantity or None
        Optional instance-specific time quantity, normally nonnegative. A supplied
        zero is an explicit value, not a missing value.
    tech_default : Quantity or None
        Technology-class fallback time quantity, normally nonnegative. None is
        returned if neither source supplies a value.

    Returns
    -------
    Quantity or None
        Selected input object with its original units; no conversion is performed.

    Notes
    -----
    Instance-specific values take precedence over technology-class defaults.
    None is the absence sentinel, so an explicitly supplied zero is preserved.
    Inputs must already be validated by their owning schema; this helper does
    not check dimensions or sign, average values, or normalize units.

    Examples
    --------
    >>> from mlsysim.core.units import Q_
    >>> resolve_latency(None, Q_('10 ns')).m_as('ns')
    10
    >>> resolve_latency(Q_('0 ns'), Q_('10 ns')).m_as('ns')
    0
    """
    return instance_latency if instance_latency is not None else tech_default
