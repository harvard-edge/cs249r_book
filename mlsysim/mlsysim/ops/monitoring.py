"""MLOps monitoring thresholds (drift, stability)."""

from ..core.provenance import sourced
from ..core.registry import Registry
from ..core import provenance_catalog as pc


class Monitoring(Registry):
    MemoryBitErrorRatePerBit = sourced(
        1e-17,
        pc.MEMORY_SOFT_ERROR_RATE,
        name="Memory soft-error rate per bit",
        description="Order-of-magnitude bit-error rate used for reliability and monitoring examples.",
    )
    KsTestCoefficient = sourced(
        1.36,
        pc.MLOPS_DRIFT_THRESHOLDS,
        name="KS-test coefficient",
        description="Asymptotic two-sample Kolmogorov-Smirnov 5% threshold coefficient.",
    )
    PsiWarnThreshold = sourced(
        0.10,
        pc.MLOPS_DRIFT_THRESHOLDS,
        name="PSI warning threshold",
        description="Population Stability Index threshold for drift warning.",
    )
    PsiReviewThreshold = sourced(
        0.20,
        pc.MLOPS_DRIFT_THRESHOLDS,
        name="PSI review threshold",
        description="Population Stability Index threshold for manual review.",
    )
    PsiCriticalThreshold = sourced(
        0.25,
        pc.MLOPS_DRIFT_THRESHOLDS,
        name="PSI critical threshold",
        description="Population Stability Index threshold for critical drift.",
    )
