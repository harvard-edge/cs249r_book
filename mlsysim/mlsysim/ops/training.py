"""Operational training-run profiles and overhead assumptions."""

from ..core.provenance import sourced
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
