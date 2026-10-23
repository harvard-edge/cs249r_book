"""Agent-security solvers: injection exposure and defense evaluation (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic on
stated rates and episode counts: how often a task meets attacker text given
how much untrusted content it reads, the same exposure counted in turns, and
attack success against benign acceptance for two policies run on equal
cohorts. None models an adaptive attacker or correlated payloads.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..results import (
    InjectionExposureResult,
    PolicyComparisonResult,
)
from .base import ForwardModel
from ...physics.agents import calc_cost_per_accepted, calc_injection_exposure


class InjectionExposureModel(ForwardModel):
    """
    Per-task exposure to injected payloads across workloads that read different volumes.

        Pr(exp) = 1 - (1 - q)^n,   linear estimate q n,
        behind a screen that misses a fraction m:  1 - (1 - q m)^n

    Reads are independent; a payload is counted once it is read, whatever it asks.
    """
    requires = ()
    produces = InjectionExposureResult

    def solve(self, payload_fraction: float, reads: Dict[str, int], miss_rate: float,
              cohort: int) -> InjectionExposureResult:
        q, qm = payload_fraction, payload_fraction * miss_rate
        screened = {k: calc_injection_exposure(qm, n) for k, n in reads.items()}
        return InjectionExposureResult(
            exposure={k: calc_injection_exposure(q, n) for k, n in reads.items()},
            linear={k: q * n for k, n in reads.items()},
            screened_rate=qm,
            screened=screened,
            screened_per_cohort={k: e * cohort for k, e in screened.items()},
        )




class PolicyComparisonModel(ForwardModel):
    """
    Two policies on equal, separate attacked and benign cohorts.

        lower = confirmed / attacked,  upper = (confirmed + unresolved) / attacked
        reduction = 1 - candidate / baseline   (observed, both upper, least favorable)
        p_acc = accepted benign / benign,  C_eff = C_attempt / p_acc

    Unresolved episodes count as harm in the upper rate; the least favorable
    reduction sets the candidate's upper rate against the baseline's lower.
    At equal attempt cost, C_eff rises by p_acc(baseline) / p_acc(candidate) - 1.
    """
    requires = ()
    produces = PolicyComparisonResult

    def solve(self, attacked: int, benign: int, baseline: Dict[str, int], candidate: Dict[str, int],
              attempt_cost: Optional[float] = None) -> PolicyComparisonResult:
        def rates(p):
            return (p["confirmed"] / attacked, (p["confirmed"] + p["unresolved"]) / attacked,
                    p["accepted_benign"] / benign)

        b_lo, b_hi, b_acc = rates(baseline)
        c_lo, c_hi, c_acc = rates(candidate)
        priced = attempt_cost is not None
        return PolicyComparisonResult(
            baseline_lower=b_lo, candidate_lower=c_lo,
            baseline_upper=b_hi, candidate_upper=c_hi,
            reduction_observed=1 - c_lo / b_lo,
            reduction_upper=1 - c_hi / b_hi,
            reduction_least_favorable=1 - c_hi / b_lo,
            baseline_acceptance=b_acc, candidate_acceptance=c_acc,
            lost_benign=baseline["accepted_benign"] - candidate["accepted_benign"],
            extra_false_blocks=candidate["false_block"] - baseline["false_block"],
            cost_rise=b_acc / c_acc - 1,
            baseline_cost_per_accepted=calc_cost_per_accepted(attempt_cost, b_acc) if priced else None,
            candidate_cost_per_accepted=calc_cost_per_accepted(attempt_cost, c_acc) if priced else None,
        )
