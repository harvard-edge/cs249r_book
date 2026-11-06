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

    def solve(
        self, payload_fraction: float, reads: Dict[str, int], miss_rate: float, cohort: int
    ) -> InjectionExposureResult:
        """Estimate independent payload exposure across read volumes.

        Parameters
        ----------
        payload_fraction : float
            Probability a read contains an injected payload, in [0, 1].
        reads : Dict[str, int]
            Workload names mapped to nonnegative counts of untrusted reads per task.
        miss_rate : float
            Fraction of payloads missed by a screen, in [0, 1].
        cohort : int
            Nonnegative task count used to report expected exposed tasks.

        Returns
        -------
        InjectionExposureResult
            Dimensionless exposure probabilities, a linear approximation,
            screened probabilities, and expected exposed-task counts.

        Notes
        -----
        Equation: For payload fraction q and n reads, exposure is ``1-(1-q)**n``.
        Screening replaces q by ``q*miss_rate``. The small-exposure approximation is ``n*q``.

        Assumptions: Reads are independent and have constant payload prevalence.
        The screen has a fixed miss probability; a read counts as exposure whether
        or not the payload affects the agent.

        Excludes: Adaptive attackers, correlated documents, execution of injected
        instructions, and any security guarantee. The linear approximation can exceed one.

        References
        ----------
        Derivation: Exposure is the complement of no payload in any independent
        read. Multiplying by cohort size gives an expected count;
        physics.agents.calc_injection_exposure computes the probability.

        Examples
        --------
        >>> from mlsysim.solvers import InjectionExposureModel
        >>> result = InjectionExposureModel().solve(0.01, {"small": 1, "large": 10}, 0.5, 100)
        >>> result.screened["large"] < result.exposure["large"]
        True
        """
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

    def solve(
        self,
        attacked: int,
        benign: int,
        baseline: Dict[str, int],
        candidate: Dict[str, int],
        attempt_cost: Optional[float] = None,
    ) -> PolicyComparisonResult:
        """Compare observed attack bounds and benign acceptance for two policies.

        Parameters
        ----------
        attacked : int
            Positive attacked-cohort size, identical for the two policies.
        benign : int
            Positive benign-cohort size, identical for the two policies.
        baseline : Dict[str, int]
            Nonnegative counts keyed confirmed, unresolved, accepted_benign, and false_block; counts must respect their cohort sizes.
        candidate : Dict[str, int]
            Same count schema and cohort denominators as baseline.
        attempt_cost : float, optional
            Nonnegative common cost per attempt in a consistent currency unit; None omits priced outputs.

        Returns
        -------
        PolicyComparisonResult
            Dimensionless harm bounds, reductions, acceptance and cost rise;
            count differences; optional costs in the supplied cost unit.

        Notes
        -----
        Equation: Lower harm is confirmed/attacked and upper harm is
        (confirmed+unresolved)/attacked. Least-favorable reduction compares candidate
        upper to baseline lower. Effective cost is attempt_cost/benign_acceptance.

        Assumptions: Cohorts have the stated common sizes. Baseline harm
        denominators and candidate benign acceptance are positive for ratios.
        Unresolved outcomes count as harm only in the upper bound.

        Excludes: Sampling uncertainty, causal attribution, differing cohort
        difficulty, adaptive attackers, and a security certification.

        References
        ----------
        Derivation: Divide observed outcome counts by their cohort denominators;
        setting unresolved outcomes to none or all harmful gives bounds. Equal
        attempt costs leave the inverse acceptance ratio as the cost multiplier.

        Examples
        --------
        >>> from mlsysim.solvers import PolicyComparisonModel
        >>> base = {"confirmed": 4, "unresolved": 1, "accepted_benign": 9, "false_block": 1}
        >>> candidate = {"confirmed": 2, "unresolved": 1, "accepted_benign": 8, "false_block": 2}
        >>> result = PolicyComparisonModel().solve(10, 10, base, candidate)
        >>> result.candidate_lower <= result.candidate_upper
        True
        """

        def rates(p):
            return (
                p["confirmed"] / attacked,
                (p["confirmed"] + p["unresolved"]) / attacked,
                p["accepted_benign"] / benign,
            )

        b_lo, b_hi, b_acc = rates(baseline)
        c_lo, c_hi, c_acc = rates(candidate)
        priced = attempt_cost is not None
        return PolicyComparisonResult(
            baseline_lower=b_lo,
            candidate_lower=c_lo,
            baseline_upper=b_hi,
            candidate_upper=c_hi,
            reduction_observed=1 - c_lo / b_lo,
            reduction_upper=1 - c_hi / b_hi,
            reduction_least_favorable=1 - c_hi / b_lo,
            baseline_acceptance=b_acc,
            candidate_acceptance=c_acc,
            lost_benign=baseline["accepted_benign"] - candidate["accepted_benign"],
            extra_false_blocks=candidate["false_block"] - baseline["false_block"],
            cost_rise=b_acc / c_acc - 1,
            baseline_cost_per_accepted=calc_cost_per_accepted(attempt_cost, b_acc) if priced else None,
            candidate_cost_per_accepted=calc_cost_per_accepted(attempt_cost, c_acc) if priced else None,
        )
