"""Named sustainability scenarios for foundations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Clusters as _Clusters
from ....infrastructure.registry import Infrastructure as _Infrastructure
from ... import scenario_types as _st


H100BarrierLoadStep = _st.PowerTransientScenario(
    name="Barrier-aligned H100 load step",
    description="Each H100 steps from communication-phase power to its TDP when a "
    "collective barrier releases it; one board loop, then 1,024 GPUs together.",
    hardware=_Hardware.Cloud.H100,
    low_power=Q_(400, "W"),  # scenario assumption: communication-phase GPU power
    rail_voltage=Q_(0.8, "V"),  # scenario assumption: accelerator core-rail voltage
    loop_inductance=Q_(50, "pH"),  # scenario assumption: VRM-to-package loop on one board
    rise_times={
        "slow": Q_(1, "microsecond"),  # edge once package capacitors act
        "fast": Q_(100, "nanosecond"),  # the same step with no package decoupling
    },
    n_devices=_Clusters.Training_1K.total_accelerators,
    ramp_window=Q_(100, "microsecond"),  # scenario assumption: cluster transition window
)


AiRackAirCooling = _st.CoolingScenario(
    name="High-density AI rack cooled by air",
    description="The high-density AI rack's heat load carried by air at a 15 K rise.",
    heat_load=Q_(_Infrastructure.Racks.AI_High.power_kw, "kW"),
    temperature_rise=Q_(15, "K"),  # scenario assumption: inlet 20 C, outlet 35 C
    coolant="air",
)


Nvl72WaterCooling = _st.CoolingScenario(
    name="GB200 NVL72 rack cooled by water",
    description="The GB200 NVL72 rack's heat carried by water at a 10 K coolant rise.",
    heat_load=_Hardware.Cloud.GB200_NVL72.tdp,
    temperature_rise=Q_(10, "K"),  # scenario assumption: coolant rise across the rack
    coolant="water",
)


H100AirCooling = _st.CoolingScenario(
    name="One H100 cooled by air",
    description="One H100's TDP carried by air at a 15 K rise.",
    heat_load=_Hardware.Cloud.H100.tdp,
    temperature_rise=Q_(15, "K"),
    coolant="air",
)


CduOutageTraining10K = _st.OutageScenario(
    name="Coolant distribution unit outage under a 10,000-GPU job",
    description="One rack's CDU pump fails for a four-hour repair; a synchronous job "
    "on the 10,000-GPU cluster idles with it.",
    failed_devices=256,  # scenario assumption: GPUs behind the failed CDU
    fleet=_Clusters.Training_10K,
    repair_time=Q_(4, "hour"),  # scenario assumption: pump repair window
    device_hour_rate=_Infrastructure.Pricing.Fleet.GpuHourRef.rate,
    rollback_times={"low": Q_(30, "minute"), "high": Q_(60, "minute")},
)
