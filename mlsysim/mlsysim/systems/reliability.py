"""Component MTTF and recovery assumptions (fleet reliability appendix)."""

from pydantic import BaseModel, ConfigDict, field_validator

from ..core.provenance import Sourced, sourced, fleet_mttf_hours
from ..core.registry import Registry
from ..core import provenance_catalog as pc
from ..core.units import GB, second


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

    def cold_restart_time(self, checkpoint_bytes, *, include_detection: bool = True):
        """Calculate a serial cold-restart budget from a resumable checkpoint.

        Parameters
        ----------
        checkpoint_bytes : Quantity
            Nonnegative checkpoint size, with data-size units such as bytes or GB.
        include_detection : bool, optional
            Include the profile's failure-detection delay when True (default).
            Use False for a scheduler-initiated stop or planned relaunch.

        Returns
        -------
        Quantity
            Modeled recovery duration in seconds, excluding lost work before failure.

        Notes
        -----
        Equation: T = T_restart + checkpoint_bytes / BW_checkpoint + T_warmup,
        plus T_detection when include_detection is True. The instance fields
        ``restart_time_s``, ``warmup_time_s``, and ``detection_time_s`` are nonnegative
        seconds; ``checkpoint_write_bw_gbs`` is positive decimal GB/s. This model uses
        the same aggregate per-job bandwidth for checkpoint writes and restore reads.
        Assumptions: stages execute serially at the profile's fixed times and
        bandwidth. Detection is the end-to-end budget, not just the heartbeat timeout.
        Excludes: work redone since the last checkpoint, failures during recovery,
        contention changes, and overlap. ``heartbeat_timeout_s`` and
        ``reschedule_time_s`` are not additional terms in this method. A failure
        charges lost work to checkpoint waste; a forced preemption adds it separately.

        References
        ----------
        Derivation: add the durations of the serial detection, restart, checkpoint
        load, and warmup stages. Load duration is bytes divided by aggregate bandwidth.
        This shared budget keeps recovery and preemption accounting on the same inputs.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> profile = RecoveryProfile(heartbeat_timeout_s=1, reschedule_time_s=2,
        ...     detection_time_s=3, restart_time_s=4, warmup_time_s=5,
        ...     checkpoint_write_bw_gbs=2)
        >>> profile.cold_restart_time(Q_('4 GB')).m_as('s')
        14.0
        >>> profile.cold_restart_time(Q_('4 GB'), include_detection=False).m_as('s')
        11.0
        """
        load = (checkpoint_bytes / (float(self.checkpoint_write_bw_gbs) * GB / second)).to(second)
        total = float(self.restart_time_s) * second + load + float(self.warmup_time_s) * second
        if include_detection:
            total = total + float(self.detection_time_s) * second
        return total.to(second)


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
            description=(
                "Aggregate checkpoint write (and restore read) bandwidth available to one job (GB/s): "
                "the per-job share of the 1 TB/s reference parallel file system "
                "(Systems.Storage.PfsOneTbPerSecond) when ten concurrent jobs share it."
            ),
        ),
    )
