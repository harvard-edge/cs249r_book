"""Operational training-run profiles and overhead assumptions."""

from ..core.provenance import sourced, sourced_qty
from ..core.units import ureg
from ..core.registry import Registry
from ..core import provenance_catalog as pc


class TrainingRunOverheads(Registry):
    """Reusable goodput-loss fractions for large distributed training runs."""

    PipelineBubble = sourced(
        0.05,
        pc.OVERHEAD_BUDGETS,
        name="Pipeline bubble overhead",
        description="Pipeline-parallel bubble overhead fraction for a well-tuned training run.",
    )
    Checkpoint = sourced(
        0.03,
        pc.OVERHEAD_BUDGETS,
        name="Checkpoint overhead",
        description="Asynchronous checkpointing overhead fraction.",
    )
    FailureRecovery = sourced(
        0.10,
        pc.OVERHEAD_BUDGETS,
        name="Failure recovery overhead",
        description="Failure and restart overhead fraction at 10k+ GPU scale.",
    )
    Maintenance = sourced(
        0.05,
        pc.OVERHEAD_BUDGETS,
        name="Maintenance overhead",
        description="Rolling upgrade and maintenance-window overhead fraction.",
    )
    # Added 2026-09-28. Not part of the four-term OVERHEAD_BUDGETS set above: a
    # shared-fleet scheduling/preemption loss for the 100K-GPU case study. With
    # the four fractions above it sums to a 0.60 goodput ratio (1 - 0.40).
    SchedulerPreemption = sourced(
        0.17,
        pc.SCHEDULER_PREEMPTION_CASE_STUDY,
        name="Scheduler and preemption overhead",
        description="Goodput-loss fraction from scheduling delays and preemption on a shared fleet.",
    )


class Preemption(Registry):
    """Cloud preemption-notice windows (added 2026-09-28)."""

    AwsSpotNotice = sourced_qty(
        120 * ureg.second,
        pc.CLOUD_PREEMPTION_NOTICE,
        name="AWS Spot interruption notice",
        description="Warning time AWS gives before reclaiming a Spot instance (two minutes).",
    )
    GcpPreemptibleNotice = sourced_qty(
        30 * ureg.second,
        pc.CLOUD_PREEMPTION_NOTICE,
        name="Google Cloud preemptible/Spot VM notice",
        description="Warning time Google Cloud gives before preempting a preemptible or Spot VM.",
    )
