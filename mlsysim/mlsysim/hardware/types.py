from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Dict, Literal
from ..core.units import Q_, ureg
from ..core.types import Quantity, Metadata, require_dimensionality, require_unit_family

class ComputeCore(BaseModel):
    """
    Represents the processing units of a hardware accelerator.
    
    Contains the theoretical peak throughput capabilities of the silicon,
    differentiated by numerical precision.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    peak_flops: Quantity
    precision_flops: Dict[str, Quantity] = Field(default_factory=dict)
    # Streaming-multiprocessor count. Present for GPUs; absent (None) for device
    # classes that have no SM concept (MCUs, TPUs). Pairs with the per-SM memory
    # fields below to derive chip-level totals (e.g. register file across the die).
    sm_count: Optional[int] = None
    # Core clock frequency. Lets cycle arithmetic (e.g. an MCU task's cycle
    # count) come from the registry instead of from a FLOP-per-cycle assumption
    # on peak_flops. Populated where a device's clock is stated; absent (None)
    # for entries whose source gives only aggregate throughput.
    clock_rate: Optional[Quantity] = None
    # Count of programmable compute cores for architectures that publish one
    # (e.g. a wafer-scale engine's processing elements). Absent (None) where the
    # vendor does not state a core count or where sm_count is the unit.
    core_count: Optional[int] = None
    # Per-SM execution units of a GPU (added 2026-10-15): FP32 (CUDA) cores and
    # Tensor Cores per streaming multiprocessor. Absent (None) for device classes
    # without SMs. Chip totals are derived (fp32_core_count, tensor_core_count),
    # so a total can never disagree with sm_count.
    fp32_cores_per_sm: Optional[int] = None
    tensor_cores_per_sm: Optional[int] = None
    # Matrix unit built as a square systolic array (added 2026-10-20): the
    # array is systolic_array_dim x systolic_array_dim multiply-accumulate
    # cells. None for devices whose matrix unit is not a published square array.
    systolic_array_dim: Optional[int] = None
    # Per-SM thread limits of a GPU (added 2026-10-20): resident threads per SM,
    # registers per thread, and threads per warp. Together with
    # register_file_per_sm they set how many threads a kernel's register use
    # leaves resident (occupancy). None for device classes without SMs.
    max_threads_per_sm: Optional[int] = None
    max_registers_per_thread: Optional[int] = None
    warp_size: Optional[int] = None

    @property
    def systolic_macs_per_cycle(self) -> Optional[int]:
        """Count the peak MACs in one square systolic array.

        Returns
        -------
        int or None
            Multiply-accumulate operations per clock cycle, or None when the array dimension is unknown.

        Notes
        -----
        Equation: MACs_per_cycle = systolic_array_dim ** 2.

        Assumptions: each cell performs one MAC each cycle after the array is filled; one square array is described.

        Excludes: pipeline fill/drain, operand starvation, utilization, multiple arrays, clock frequency, and conversion from MACs to FLOPs.

        References
        ----------
        Derivation: a square array with dimension d contains ``d*d`` independent MAC cells. Peak cell count is not sustained throughput.

        Examples
        --------
        >>> from mlsysim.hardware.types import ComputeCore
        >>> from mlsysim.core.units import Q_
        >>> core = ComputeCore(peak_flops=Q_('1 TFLOP/s'), systolic_array_dim=8)
        >>> core.systolic_macs_per_cycle == core.systolic_array_dim ** 2
        True
        """
        if self.systolic_array_dim is None:
            return None
        return self.systolic_array_dim * self.systolic_array_dim

    @property
    def fp32_core_count(self) -> Optional[int]:
        """Count the chip-wide FP32 execution units.

        Returns
        -------
        int or None
            FP32 core count, or None if either per-SM count or SM count is absent.

        Notes
        -----
        Equation: N_FP32 = sm_count * fp32_cores_per_sm.

        Assumptions: all listed SMs have the same declared FP32-unit count.

        Excludes: operating frequency, issue width, instructions per cycle, disabled units, and achieved utilization.

        References
        ----------
        Derivation: sum an equal per-SM unit count over all registered SMs; this inventory multiplication is not a FLOP/s estimate.

        Examples
        --------
        >>> from mlsysim.hardware.types import ComputeCore
        >>> from mlsysim.core.units import Q_
        >>> core = ComputeCore(peak_flops=Q_('1 TFLOP/s'), sm_count=2, fp32_cores_per_sm=8)
        >>> core.fp32_core_count == core.sm_count * core.fp32_cores_per_sm
        True
        """
        if self.sm_count is None or self.fp32_cores_per_sm is None:
            return None
        return self.sm_count * self.fp32_cores_per_sm

    @property
    def tensor_core_count(self) -> Optional[int]:
        """Count the chip-wide matrix execution units.

        Returns
        -------
        int or None
            Tensor Core count, or None when either constituent count is absent.

        Notes
        -----
        Equation: N_tensor = sm_count * tensor_cores_per_sm.

        Assumptions: all registered SMs contain the same number of Tensor Cores.

        Excludes: matrix shape, precision support, sparsity modes, clock frequency, and utilization; cores from different generations need not have equal throughput.

        References
        ----------
        Derivation: multiply equal per-SM inventories by the SM count. The result counts hardware units rather than their arithmetic work.

        Examples
        --------
        >>> from mlsysim.hardware.types import ComputeCore
        >>> from mlsysim.core.units import Q_
        >>> core = ComputeCore(peak_flops=Q_('1 TFLOP/s'), sm_count=2, tensor_cores_per_sm=4)
        >>> core.tensor_core_count == core.sm_count * core.tensor_cores_per_sm
        True
        """
        if self.sm_count is None or self.tensor_cores_per_sm is None:
            return None
        return self.sm_count * self.tensor_cores_per_sm

    @field_validator("peak_flops", mode="after")
    @classmethod
    def _validate_peak_flops(cls, v):
        return require_unit_family(v, ureg.flop / ureg.second, "peak_flops", "operation")

    @field_validator("precision_flops", mode="after")
    @classmethod
    def _validate_precision_flops(cls, v):
        return {
            key: require_unit_family(val, ureg.flop / ureg.second, f"precision_flops[{key!r}]", "operation")
            for key, val in v.items()
        }

    @field_validator("clock_rate", mode="after")
    @classmethod
    def _validate_clock_rate(cls, v):
        q = require_dimensionality(v, 1 / ureg.second, "clock_rate")
        if q is None:
            return None
        # FLOP/s shares the 1/time dimension; require an explicit hertz unit.
        if not any("hertz" in name for name in q.units._units):
            raise ValueError(f"clock_rate must use hertz units; got {q.units}")
        return q

class MemoryHierarchy(BaseModel):
    """
    Represents the memory subsystem of a hardware accelerator.
    
    Captures capacity and bandwidth across different levels of the memory
    hierarchy, from primary HBM/DRAM down to on-chip SRAM or Flash storage.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    capacity: Quantity           # Primary memory (HBM for GPUs, SRAM for MCUs)
    bandwidth: Quantity          # Primary memory bandwidth
    # On-chip SRAM (optional — for modeling FlashAttention tiling on GPUs,
    # or the fast scratchpad on microcontrollers)
    sram_capacity: Optional[Quantity] = None
    sram_bandwidth: Optional[Quantity] = None
    # Flash storage (optional — for TinyML devices where weights live in flash)
    flash_capacity: Optional[Quantity] = None
    flash_bandwidth: Optional[Quantity] = None
    l2_cache: Optional[Quantity] = None
    # Per-SM on-chip memory. Present for GPUs; absent (None) for device classes
    # without SMs. Multiply by ComputeCore.sm_count to derive chip-level totals.
    register_file_per_sm: Optional[Quantity] = None
    shared_memory_per_sm: Optional[Quantity] = None
    # DRAM interface: total data-bus width and per-pin data rate (bit/s per pin,
    # one bit per transfer, so 6.4 Gb/s is 6400 MT/s). bandwidth = width x rate / 8.
    bus_width: Optional[Quantity] = None
    pin_data_rate: Optional[Quantity] = None
    # Number of active HBM stacks on the package (HBM devices only). Per-stack
    # bandwidth/capacity = bandwidth / hbm_stacks, capacity / hbm_stacks.
    hbm_stacks: Optional[int] = None

    @field_validator("capacity", "sram_capacity", "flash_capacity", "l2_cache", "register_file_per_sm", "shared_memory_per_sm", mode="after")
    @classmethod
    def _validate_capacity_fields(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")

    @field_validator("bus_width", mode="after")
    @classmethod
    def _validate_bus_width(cls, v):
        if v is None:
            return v
        return require_unit_family(v, ureg.bit, "bus_width", "data")

    @field_validator("bandwidth", "sram_bandwidth", "flash_bandwidth", "pin_data_rate", mode="after")
    @classmethod
    def _validate_bandwidth_fields(cls, v, info):
        return require_unit_family(v, ureg.byte / ureg.second, info.field_name, "data")

class StorageHierarchy(BaseModel):
    """
    Represents the persistent storage subsystem connected to the hardware.
    
    Crucial for analyzing checkpointing overheads and the Data Wall (data
    ingestion rates for training).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    capacity: Quantity
    bandwidth: Quantity
    latency: Optional[Quantity] = None

    @field_validator("capacity", mode="after")
    @classmethod
    def _validate_capacity(cls, v):
        return require_unit_family(v, ureg.byte, "capacity", "data")

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.byte / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")

class IOInterconnect(BaseModel):
    """
    Represents a point-to-point interconnect link.

    Used to model PCIe links (host to device) or NVLink/ICI connections
    (device to device within a node).

    Direction convention (2026-06-10 audit). ``bandwidth`` stores the figure
    exactly as the vendor datasheet quotes it, and ``direction`` declares
    which convention that figure uses:

    - ``per_direction`` (default): the one-way deliverable rate. PCIe entries
      (64 GB/s Gen5 x16) and Ethernet/RoCE entries use this.
    - ``bidirectional_total``: send + receive summed. NVIDIA quotes NVLink
      this way (H100 "900 GB/s" = 450 GB/s each direction); Google quotes
      TPU ICI the same way.

    Formulas that divide bytes by a link rate (transfer time, ring-collective
    beta terms) need the ONE-WAY rate: always consume
    ``bandwidth_per_direction``, never raw ``bandwidth``. Feeding the
    bidirectional total into a beta term renders every intra-node collective
    ~2x optimistic — the exact bug this field exists to prevent.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str # e.g., "PCIe Gen4 x16"
    bandwidth: Quantity
    latency: Optional[Quantity] = None
    direction: Literal["per_direction", "bidirectional_total"] = "per_direction"

    @property
    def bandwidth_per_direction(self) -> Quantity:
        """Normalize the advertised link rate to one direction.

        Returns
        -------
        Quantity
            One-way data rate in the same compatible units as bandwidth.

        Notes
        -----
        Equation: BW_oneway = bandwidth / 2 for bidirectional_total, otherwise bandwidth.

        Assumptions: a bidirectional total sums equal transmit and receive capacities. The direction field correctly describes the vendor specification.

        Excludes: protocol overhead, contention, payload efficiency, and asymmetric transmit/receive rates.

        References
        ----------
        Derivation: a symmetric full-duplex total is the sum of two equal directional rates, so dividing by two recovers the transfer-time denominator.

        Examples
        --------
        >>> from mlsysim.hardware.types import IOInterconnect
        >>> from mlsysim.core.units import Q_
        >>> link = IOInterconnect(name='Duplex', bandwidth=Q_('100 GB/s'), direction='bidirectional_total')
        >>> 2 * link.bandwidth_per_direction == link.bandwidth
        True
        """
        if self.direction == "bidirectional_total":
            return self.bandwidth / 2
        return self.bandwidth

    @field_validator("bandwidth", mode="after")
    @classmethod
    def _validate_bandwidth(cls, v):
        return require_unit_family(v, ureg.bit / ureg.second, "bandwidth", "data")

    @field_validator("latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        return require_dimensionality(v, ureg.second, "latency")

class MigProfile(BaseModel):
    """A single NVIDIA Multi-Instance GPU partition profile (e.g. ``1g.10gb``).

    MIG carves one physical GPU into hardware-isolated instances, each with a fixed
    slice of HBM and a fixed number of streaming multiprocessors. The profiles are a
    frozen per-product spec (NVIDIA MIG User Guide), so they live on the device node.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str                     # NVIDIA profile name, e.g. "1g.10gb"
    gpu_memory: Quantity          # dedicated HBM per instance
    sm_count: int                 # streaming multiprocessors per instance
    typical_workload: str = ""

    @field_validator("gpu_memory", mode="after")
    @classmethod
    def _validate_gpu_memory(cls, v):
        return require_unit_family(v, ureg.byte, "gpu_memory", "data")


class HardwareNode(BaseModel):
    """
    Layer B (Hardware Supply): Represents a complete hardware accelerator.
    
    This is the primary object that solvers evaluate against. It encapsulates
    the compute, memory, IO, and power constraints of a single physical device 
    (e.g., an NVIDIA H100 GPU or a Jetson Orin Nano).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    release_year: int
    compute: ComputeCore
    memory: MemoryHierarchy
    storage: Optional[StorageHierarchy] = None
    interconnect: Optional[IOInterconnect] = None  # Host I/O (e.g., PCIe)
    nvlink: Optional[IOInterconnect] = None        # Intra-node GPU-GPU (NVLink/ICI)
    tdp: Optional[Quantity] = None
    tdp_min: Optional[Quantity] = None
    tdp_max: Optional[Quantity] = None
    die_area: Optional[Quantity] = None
    # Transistors on the die, as the vendor states it (added 2026-10-20).
    transistor_count: Optional[float] = None
    # Package-internal link joining the compute dies of a multi-die (chiplet)
    # accelerator, stored as the vendor quotes it. None for monolithic dies.
    # Vendors do not state a per-direction convention for this figure, so it
    # is not an IOInterconnect; do not feed it into beta terms as a one-way rate.
    die_to_die_bandwidth: Optional[Quantity] = None
    battery_capacity: Optional[Quantity] = None
    unit_cost: Optional[Quantity] = None
    unit_cost_max: Optional[Quantity] = None
    accelerator_count: Optional[int] = None
    embodied_carbon_kg: Optional[float] = Field(
        default=None,
        description="Embodied CO2e in kg from manufacturing, packaging, and transport (Gupta et al. 2022)."
    )
    dispatch_tax: Quantity = Field(default_factory=lambda: Q_("0.01 ms"))
    # Multi-Instance GPU partition profiles, when the device supports MIG (A100+).
    # None for devices without MIG. Frozen per-product spec (NVIDIA MIG User Guide).
    mig_profiles: Optional[tuple[MigProfile, ...]] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("tdp", "tdp_min", "tdp_max", mode="after")
    @classmethod
    def _validate_power_fields(cls, v, info):
        return require_dimensionality(v, ureg.watt, info.field_name)

    @field_validator("die_area", mode="after")
    @classmethod
    def _validate_die_area(cls, v):
        return require_dimensionality(v, ureg.meter**2, "die_area")

    @field_validator("die_to_die_bandwidth", mode="after")
    @classmethod
    def _validate_die_to_die_bandwidth(cls, v):
        return require_unit_family(v, ureg.byte / ureg.second, "die_to_die_bandwidth", "data")

    @field_validator("battery_capacity", mode="after")
    @classmethod
    def _validate_battery_capacity(cls, v):
        return require_dimensionality(v, ureg.joule, "battery_capacity")

    @field_validator("unit_cost", "unit_cost_max", mode="after")
    @classmethod
    def _validate_cost_fields(cls, v, info):
        return require_unit_family(v, ureg.dollar, info.field_name, "currency")

    @field_validator("dispatch_tax", mode="after")
    @classmethod
    def _validate_dispatch_tax(cls, v):
        return require_dimensionality(v, ureg.second, "dispatch_tax")

    @property
    def embodied_carbon(self) -> Optional[Quantity]:
        """Attach mass units to the stored manufacturing-carbon estimate.

        Returns
        -------
        Quantity or None
            Kilograms of CO2-equivalent emissions, or None when no estimate is supplied.

        Notes
        -----
        Equation: carbon = embodied_carbon_kg * kg.

        Assumptions: the field already represents the intended manufacturing, packaging, and transport boundary. The registry source determines that boundary.

        Excludes: operational energy emissions, service-life amortization, and a new life-cycle assessment.

        References
        ----------
        Derivation: this property adds the declared kilogram unit to a stored scalar; it does not derive or validate the underlying emissions estimate.

        Examples
        --------
        >>> from mlsysim import Hardware
        >>> hw = Hardware.Cloud.H100.model_copy(update={'embodied_carbon_kg': 100.0})
        >>> hw.embodied_carbon.m_as('kg') == hw.embodied_carbon_kg
        True
        """
        if self.embodied_carbon_kg is None:
            return None
        return Q_(self.embodied_carbon_kg, "kg")

    def ridge_point(self, precision: Optional[str] = None) -> Quantity:
        """Calculate the peak Roofline arithmetic-intensity crossover.

        Parameters
        ----------
        precision : str, optional
            Precision key for compute.precision_flops, matched case-insensitively.
            None selects peak_flops; an unknown key warns and falls back to it.

        Returns
        -------
        Quantity
            Peak compute throughput divided by primary-memory bandwidth, in
            FLOP/byte. This is not derated by a solver's efficiency input.

        Notes
        -----
        Equation: I_ridge = F_peak / BW_primary. At this intensity, the ideal
        compute and memory-time bounds coincide.

        Assumptions: positive primary-memory bandwidth and the declared peak
        compute rate; peak compute and bandwidth can be compared for the same
        workload precision and memory traffic boundary.

        Excludes: dispatch overhead, SRAM/flash/offload paths, achieved bandwidth,
        and compute efficiency. A solver's effective ridge may differ.

        References
        ----------
        Williams et al. (2009), "Roofline: An Insightful Visual Performance Model
        for Multicore Architectures."
        https://doi.org/10.1145/1498765.1498785 (bibliography key: williams2009).

        Examples
        --------
        >>> from mlsysim import Hardware
        >>> hw = Hardware.Cloud.H100
        >>> hw.ridge_point() == (hw.compute.peak_flops / hw.memory.bandwidth).to('flop/byte')
        True
        """
        flops = self.compute.peak_flops
        if precision:
            if precision.lower() in self.compute.precision_flops:
                flops = self.compute.precision_flops[precision.lower()]
            else:
                import warnings
                warnings.warn(
                    f"Precision '{precision}' not found in hardware '{self.name}'; "
                    f"using default peak_flops.",
                    stacklevel=2
                )
        return (flops / self.memory.bandwidth).to('flop/byte')

    def __repr__(self):
        return f"HardwareNode({self.name}, {self.release_year})"
