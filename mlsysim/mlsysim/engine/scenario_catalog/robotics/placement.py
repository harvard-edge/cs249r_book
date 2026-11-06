"""Named robotics scenarios for placement.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....hardware.registry import Hardware as _Hardware
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ..robotics.realtime import WarehousePermissionTickResponse


# --- Workload placement: the permission tick against what it shares (added 2026-10-20) ---
# The warehouse machine's 1 kHz permission tick (site WarehouseAisle), its
# unloaded sub-costs, and the rejected shared-die placements that price
# memory queueing, miss-handling stalls, rail droop, and thermal throttling.
# Inputs are illustrative teaching assumptions unless a source is named.
WarehouseEnforcerUnloadedTick = _st.CycleBudgetScenario(
    name="Unloaded run of the warehouse machine's 1 kHz permission check",
    description="Sense, check, and stage sub-costs of the 135 microsecond unloaded tick on the discrete safety MCU.",
    site=_Embodied.Scenario.WarehouseAisle,
    stages=[  # scenario assumption: illustrative sub-costs
        ("sense", Q_(40.0, "us")),  # encoders, IMU over SPI, evidence epoch
        ("constraint check", Q_(80.0, "us")),  # kinematic and dynamic envelope
        ("drive-frame staging", Q_(15.0, "us")),
    ],
)


WarehouseSharedDieLoadedRead = _st.SharedDramReadScenario(
    name="The permission tick's snapshot read behind proposer backlogs on the application SoC",
    description="A FIFO LPDDR5 controller serves whole backlogs (a 256 KiB weight tile, a camera "
    "transfer, a telemetry record) before the tick's 4 KiB snapshot; addresses "
    "interleave across every channel.",
    site=_Embodied.Scenario.WarehouseAisle,
    memory_tech=_Hardware.Tech.Memory.LPDDR5_6400,
    sense=WarehouseEnforcerUnloadedTick.stage("sense"),
    compute_after_sense=(
        WarehouseEnforcerUnloadedTick.stage("constraint check")
        + WarehouseEnforcerUnloadedTick.stage("drive-frame staging")
    ),
    tile_bytes=256 * 1024,  # scenario assumption: one weight tile
    camera_bytes=64 * 1024,  # scenario assumption
    telemetry_bytes=32 * 1024,  # scenario assumption
    snapshot_bytes=4 * 1024,  # scenario assumption: encoder, IMU, evidence-epoch snapshot
    transaction_bytes=4 * 1024,  # AXI4 4 KiB burst boundary
    bank_switches=3,  # scenario assumption: same-bank row conflicts per backlog
)


WarehouseSharedDieLoadedReadOneChannel = WarehouseSharedDieLoadedRead.variant(
    name="The same snapshot read with backlog and read confined to one LPDDR5 channel",
    description="Contrast case: the address mapping puts both on the snapshot's channel.",
    one_channel=True,
)


WarehouseTickMshrStallShare = _st.StallShareScenario(
    name="Miss-stall share of the permission tick that a 21x MSHR slowdown allows",
    description="Measured shared-cache contention with partitioning (Valsan et al. 2016) as the "
    "slowdown of the tick's stalled share, against the declared WCET.",
    site=_Embodied.Scenario.WarehouseAisle,
    slowdown=21.0,  # Valsan, P. K., Yun, H., and Farshchi, F. (2016), Taming Non-blocking...
)


WarehouseTickMshrStallShareAtDeadline = WarehouseTickMshrStallShare.variant(
    name="The same 21x slowdown against the deadline less blocking and interference",
    description="Bound = execution ceiling of the full response budget.",
    bound=(
        _Embodied.Scenario.WarehouseAisle.enforcer_deadline
        - _Embodied.Scenario.WarehouseAisle.enforcer_blocking
        - _Embodied.Scenario.WarehouseAisle.enforcer_interference
    ).to("us"),
)


WarehouseTickWritebackStallShare = WarehouseTickMshrStallShare.variant(
    name="Miss-stall share of the permission tick that a 300x writeback-buffer slowdown allows",
    description="A round floor under the writeback-buffer attack slowdowns Bechtel and Yun (2019) "
    "measured (up to 346x).",
    slowdown=300.0,  # scenario assumption: round floor below the measured maximum
)


WarehouseTickTenfoldStallShare = WarehouseTickMshrStallShare.variant(
    name="Miss-stall share of the permission tick that a 10x slowdown allows",
    description="A self-check estimate between the measured slowdowns.",
    slowdown=10.0,  # scenario assumption: illustrative slowdown
)


PackageRailDroop = _st.RailDroopScenario(
    name="A 6 A accelerator current step on a shared 0.85 V core rail",
    description="L dI/dt plus I R through an illustrative package, and the gate-delay stretch of "
    "the alpha-power model (Sakurai and Newton 1990) at the drooped rail.",
    rail_voltage=Q_(0.850, "V"),  # scenario assumption
    inductance=Q_(30.0, "pH"),  # scenario assumption: package loop inductance
    resistance=Q_(8.0, "milliohm"),  # scenario assumption
    delta_i=Q_(6.0, "A"),  # scenario assumption: accelerator current step
    rise_time=Q_(2.0, "ns"),  # scenario assumption
    threshold_voltage=Q_(0.420, "V"),  # scenario assumption
    alpha=1.3,  # scenario assumption: velocity-saturation index
)


WarehouseSharedDieThermalThrottle = _st.ThermalThrottleScenario(
    name="The permission tick on a shared-die real-time cluster whose governor halves its clock",
    description="Rejected placement: the tick sized to its WCET at 1.2 GHz; continuous proposal "
    "inference heats the junction past a 95 degC trip, and the clock stays at 600 MHz "
    "until the junction clears 85 degC.",
    site=_Embodied.Scenario.WarehouseAisle,
    nominal_clock=Q_(1.2, "GHz"),  # scenario assumption
    throttled_clock=Q_(600.0, "MHz"),  # scenario assumption: the governor halves the clock
    time_constant=Q_(8.0, "s"),  # scenario assumption
    ambient=Q_(45.0, "degC"),  # scenario assumption: electronics bay
    theta=Q_(2.0, "K/W"),  # scenario assumption: junction to ambient
    quiet_power=Q_(4.0, "W"),  # scenario assumption: enforcer, idle, memory
    loaded_power=Q_(26.0, "W"),  # scenario assumption: with continuous inference
    trip=Q_(95.0, "degC"),  # scenario assumption
    clear=Q_(85.0, "degC"),  # scenario assumption
)


WarehouseLoadedCycleCheckpoint = WarehousePermissionTickResponse.variant(
    name="A 300 microsecond loaded cycle of the permission check on a shared die",
    description="Self-check: past the declared WCET, still inside the deadline.",
    wcet=Q_(300.0, "us"),  # scenario assumption: one measured loaded cycle
)


WarehouseRejectedPlacementTrace = _st.TickTraceScenario(
    name="Illustrative ten-million-tick trace of the permission check on the shared die",
    description="Cold with an idle proposer, warm under contention; percentiles are illustrative, not a measurement.",
    site=_Embodied.Scenario.WarehouseAisle,
    ticks=10_000_000,  # scenario assumption: trace length
    cold={"p50": Q_(95.0, "us"), "p99": Q_(128.0, "us"), "max": Q_(152.0, "us")},  # scenario assumption
    warm={
        "p50": Q_(210.0, "us"),
        "p99": Q_(330.0, "us"),
        "p99.9": Q_(395.0, "us"),
        "max": Q_(468.0, "us"),
    },  # scenario assumption
)
