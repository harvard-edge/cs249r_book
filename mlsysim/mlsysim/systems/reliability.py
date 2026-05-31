"""Component MTTF and recovery assumptions (fleet reliability appendix)."""

from pydantic import BaseModel, ConfigDict

from ..core.provenance import Sourced, sourced, fleet_mttf_hours
from ..core.registry import Registry
from ..core import provenance_catalog as pc


class ReliabilityComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    mttf_hours: Sourced | float
    failure_mode: str = ""


class RecoveryProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    heartbeat_timeout_s: Sourced | float
    reschedule_time_s: Sourced | float
    checkpoint_write_bw_gbs: Sourced | float


class Reliability(Registry):
    Gpu = ReliabilityComponent(
        name="GPU",
        mttf_hours=fleet_mttf_hours(50_000, component="GPU", failure_mode="die defect, thermal fatigue"),
    )
    Nic = ReliabilityComponent(
        name="NIC",
        mttf_hours=fleet_mttf_hours(150_000, component="NIC", failure_mode="transceiver degradation"),
    )
    Psu = ReliabilityComponent(
        name="PSU",
        mttf_hours=fleet_mttf_hours(100_000, component="PSU", failure_mode="capacitor aging"),
    )
    PcieSwitch = ReliabilityComponent(
        name="PCIe switch",
        mttf_hours=fleet_mttf_hours(200_000, component="PCIe switch", failure_mode="solder joint, ESD"),
    )
    Cable = ReliabilityComponent(
        name="optical cable / transceiver",
        mttf_hours=fleet_mttf_hours(50_000, component="optical cable / transceiver", failure_mode="fiber bend, connector wear"),
    )
    TorSwitch = ReliabilityComponent(
        name="top-of-rack switch",
        mttf_hours=fleet_mttf_hours(300_000, component="top-of-rack switch", failure_mode="ASIC, fan bearing"),
    )
    Hbm = ReliabilityComponent(
        name="HBM",
        mttf_hours=fleet_mttf_hours(200_000, component="HBM", failure_mode="bit-flip accumulation, TSV"),
    )
    SdcRatePerGpuHr = sourced(
        1e-6,
        pc.MEMORY_SOFT_ERROR_RATE,
        name="Silent data corruption rate per GPU-hour",
        description="Order-of-magnitude SDC rate used for fleet reliability examples.",
    )
    Recovery = RecoveryProfile(
        heartbeat_timeout_s=sourced(30, pc.RECOVERY_TIME_ASSUMPTIONS, name="Heartbeat timeout", description="Failure detection latency before reschedule."),
        reschedule_time_s=sourced(60, pc.RECOVERY_TIME_ASSUMPTIONS, name="Reschedule time", description="Time to allocate a replacement node after failure detection."),
        checkpoint_write_bw_gbs=sourced(100, pc.RECOVERY_TIME_ASSUMPTIONS, name="Checkpoint write bandwidth", description="Aggregate checkpoint write bandwidth to storage (GB/s)."),
    )
