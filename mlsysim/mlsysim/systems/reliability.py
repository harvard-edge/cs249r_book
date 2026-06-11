"""Component MTTF and recovery assumptions (fleet reliability appendix)."""

from pydantic import BaseModel, ConfigDict, field_validator

from ..core.provenance import Sourced, sourced, fleet_mttf_hours
from ..core.registry import Registry
from ..core import provenance_catalog as pc


class ReliabilityComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    mttf_hours: Sourced | float
    failure_mode: str = ""
    # Unprotected soft-error rate in FIT per megabit (FIT = failures per 1e9 device-hours).
    # Optional: only memory components (HBM/DRAM) carry an intrinsic soft-error budget.
    soft_error_fit_per_mbit: Sourced | float | None = None

    @field_validator("soft_error_fit_per_mbit", mode="after")
    @classmethod
    def _validate_soft_error_fit(cls, v):
        # Guard against drift: the published DRAM/HBM soft-error rate spans
        # 200-5000 FIT/Mbit (Tezzaron; soft-error literature). Anything outside
        # that band is almost certainly a units or transcription error.
        if v is not None and not (200.0 <= float(v) <= 5000.0):
            raise ValueError(
                f"soft_error_fit_per_mbit={v} is outside the published "
                "200-5000 FIT/Mbit DRAM soft-error range"
            )
        return v


class RecoveryProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    heartbeat_timeout_s: Sourced | float
    reschedule_time_s: Sourced | float
    detection_time_s: Sourced | float
    restart_time_s: Sourced | float
    warmup_time_s: Sourced | float
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
        soft_error_fit_per_mbit=sourced(
            250,
            pc.HBM_SOFT_ERROR_FIT_PER_MBIT,
            name="HBM unprotected soft-error rate (FIT/Mbit)",
            description="Low-end teaching figure for the per-megabit soft-error budget of unprotected HBM.",
        ),
    )
    DgxNodeComposite = ReliabilityComponent(
        name="DGX node composite",
        mttf_hours=fleet_mttf_hours(
            1_000,
            component="DGX node composite",
            failure_mode="GPU, host, power, cooling, or network component failure",
        ),
    )
    NodeRecoveryLowMin = sourced(
        10,
        pc.RECOVERY_TIME_ASSUMPTIONS,
        name="Node recovery low estimate (minutes)",
        description="Lower-bound automated node-drain and restart time.",
    )
    NodeRecoveryHighMin = sourced(
        30,
        pc.RECOVERY_TIME_ASSUMPTIONS,
        name="Node recovery high estimate (minutes)",
        description="Upper-bound automated node-drain and restart time.",
    )
    SdcRatePerGpuHr = 1e-6
    Recovery = RecoveryProfile(
        # Two DISTINCT detection quantities (clarified 2026-06-06; they are not
        # interchangeable): heartbeat_timeout_s is the missed-heartbeat interval
        # alone (the lower bound on detection); detection_time_s is the
        # end-to-end budget — timeout plus failure confirmation and propagation
        # to the controller — before any recovery action begins.
        heartbeat_timeout_s=sourced(
            30,
            pc.RECOVERY_TIME_ASSUMPTIONS,
            name="Heartbeat timeout",
            description=(
                "Missed-heartbeat interval after which a coordinator declares a "
                "worker failed; the lower bound on failure detection."
            ),
        ),
        reschedule_time_s=sourced(
            60,
            pc.RECOVERY_TIME_ASSUMPTIONS,
            name="Reschedule time",
            description="Time to allocate a replacement node after failure detection.",
        ),
        detection_time_s=sourced(
            60,
            pc.RECOVERY_TIME_ASSUMPTIONS,
            name="End-to-end failure-detection time",
            description=(
                "Heartbeat timeout plus failure confirmation and propagation to "
                "the controller, before recovery actions begin."
            ),
        ),
        restart_time_s=sourced(
            180,
            pc.RECOVERY_TIME_ASSUMPTIONS,
            name="Recovery-budget restart time",
            description="Reference queue, launch, import, and NCCL setup time used in the recovery-budget example.",
        ),
        warmup_time_s=sourced(
            120,
            pc.RECOVERY_TIME_ASSUMPTIONS,
            name="Recovery-budget warmup time",
            description="Reference JIT, buffer, and connection warmup time used in the recovery-budget example.",
        ),
        checkpoint_write_bw_gbs=sourced(
            100,
            pc.RECOVERY_TIME_ASSUMPTIONS,
            name="Checkpoint write bandwidth",
            description="Aggregate checkpoint write bandwidth to storage (GB/s).",
        ),
    )
