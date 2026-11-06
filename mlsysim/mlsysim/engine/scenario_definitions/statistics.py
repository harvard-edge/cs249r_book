"""Sampling, uncertainty, evidence, and evaluation decisions.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
import math as _math
from typing import Any, Dict, List, Optional
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from .base import BaseScenario
from .reliability import AccuracyDecayScenario


class ConfidenceRegimesScenario(BaseScenario):
    """Stipulated proposer confidence against task success in several operating regimes.

    Notes
    -----
    Equation: confidence gap = supplied confidence - supplied success for the named
    regime.

    Assumptions: Each tuple contains a regime name and compatible probability-scale
    values.

    Excludes: Estimating calibration from samples or interpreting confidence as a safety
    guarantee.

    References
    ----------
    Derivation: confidence gap = supplied confidence -
    supplied success for the named regime. The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ProposerConfidenceRegimes
    >>> round(float(case.gap('In distribution')), 3)
    0.02"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    regimes: List[Any]  # (regime, confidence, success)

    def gap(self, regime: str) -> float:
        """Supplied confidence minus task-success fraction for the named regime; raises
        KeyError if absent.

        Equation: Follow the gap calculation below; branch conditions and rounding are
        part of its definition."""
        for name, conf, success in self.regimes:
            if name == regime:
                return conf - success
        raise KeyError(regime)

    def row(self, regime: str):
        """The stored (regime, confidence, success) tuple; this accessor performs no
        calibration.

        Equation: Follow the row calculation below; branch conditions and rounding are
        part of its definition."""
        for r in self.regimes:
            if r[0] == regime:
                return r
        raise KeyError(regime)


class FalseAlarmBudgetScenario(BaseScenario):
    """One detection window every ``window``, ``listening_per_day`` of each day, over ``horizon``.

    ``budget`` is the expected false alarms allowed per horizon. Each window with
    no keyword is a chance to fire, so a per-window rejection rate multiplies into
    a count; ``rejection_levels`` lists rates to compare against the budget.

    Notes
    -----
    Equation: expected alarms = listening windows * false-positive probability; allowable
    probability = budget / windows.

    Assumptions: Listening windows tile the active listening duration. The calculation
    controls expected counts, not a probability of no alarms.

    Excludes: Temporal dependence in alarm episodes, deduplication, and uncertainty in the
    supplied rejection rate.

    References
    ----------
    Derivation: expected alarms = listening windows * false-
    positive probability; allowable probability = budget / windows. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KeywordSpotterFalseWakes
    >>> round(float(case.listening_time.to('h').magnitude), 3)
    720.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: Quantity
    listening_per_day: Quantity
    horizon: Quantity
    budget: float
    rejection_levels: List[float] = []

    @property
    def listening_time(self) -> Quantity:
        """Total active listening hours over the declared horizon, scaling the daily
        listening schedule.

        Equation: ``return (horizon.to('day').magnitude * listening_per_day).to('hour')``."""
        return (self.horizon.to("day").magnitude * self.listening_per_day).to("hour")

    @property
    def windows(self) -> float:
        """Number of detection-window opportunities over active listening time; fractional
        counts are retained.

        Equation: ``return (listening_time / window).to('').magnitude``."""
        return (self.listening_time / self.window).to("").magnitude

    @property
    def target_false_positive_rate(self) -> float:
        """False-positive probability per window that exhausts the expected-count budget.

        Equation: ``return budget / windows``."""
        return self.budget / self.windows

    @property
    def rejection_needed(self) -> float:
        """Per-window true-negative fraction needed to meet the false-alarm budget.

        Equation: ``return 1 - target_false_positive_rate``."""
        return 1 - self.target_false_positive_rate

    def false_alarms(self, rejection: float) -> float:
        """Expected count over the horizon from windows times one minus the supplied
        rejection fraction.

        Equation: ``return windows * (1 - rejection)``."""
        return self.windows * (1 - rejection)

    def false_alarms_at_rate(self, rate: Quantity) -> float:
        """Expected count over active listening time at a supplied inverse-time false-alarm
        rate.

        Equation: ``return (rate * listening_time).to('').magnitude``."""
        return (rate * self.listening_time).to("").magnitude


class ZeroEventExposureScenario(BaseScenario):
    """Exposure needed, with zero events observed, to bound a rate of ``budget`` per ``horizon``
    at one-sided ``confidence``, and the size of recording that exposure at ``sample_rate``.

    Notes
    -----
    Equation: zero-event bound = -log(1-confidence); required exposure = bound / target
    event rate.

    Assumptions: Events follow the stationary Poisson model used by the exposure helper
    and confidence lies between zero and one.

    Excludes: Nonstationary hazards, correlated events, and a guarantee of zero failures
    after testing.

    References
    ----------
    Derivation: zero-event bound = -log(1-confidence);
    required exposure = bound / target event rate. The individual properties state units
    and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KeywordSpotterFalseWakeEvidence
    >>> round(float(case.target_rate.to('1/h').magnitude), 3)
    0.001"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    budget: float
    horizon: Quantity
    confidence: float
    sample_rate: Quantity
    sample_width: Quantity

    @property
    def target_rate(self) -> Quantity:
        """Allowed event rate in inverse hours, budget divided by horizon.

        Equation: ``return (budget / horizon).to('1/hour')``."""
        return (self.budget / self.horizon).to("1/hour")

    @property
    def events_bound(self) -> float:
        """One-sided Poisson expected-count bound after zero events: negative log of one
        minus confidence.

        Equation: ``return -math.log(1 - confidence)``."""
        import math

        return -math.log(1 - self.confidence)

    @property
    def exposure_needed(self) -> Quantity:
        """Required observation hours from
        ``physics.robotics.assurance.calc_empirical_testing_exposure``.

        Equation: ``return calc_empirical_testing_exposure(target_rate,
        confidence).to('hour')``."""
        from ...physics import calc_empirical_testing_exposure

        return calc_empirical_testing_exposure(self.target_rate, self.confidence).to("hour")

    @property
    def horizons_needed(self) -> float:
        """Exposure duration divided by one budget horizon, a dimensionless observation
        multiple.

        Equation: ``return (exposure_needed / horizon).to('').magnitude``."""
        return (self.exposure_needed / self.horizon).to("").magnitude

    @property
    def recording_size(self) -> Quantity:
        """Raw recorded bytes during the required exposure, at the supplied sample rate and
        width.

        Equation: ``return (exposure_needed * sample_rate * sample_width).to('byte')``."""
        return (self.exposure_needed * self.sample_rate * self.sample_width).to("byte")


class AnnotatorAgreementScenario(BaseScenario):
    """Two independent annotators with the same label mix ``class_shares``: chance agreement,
    and Cohen's kappa at an ``observed_agreement``.

    Notes
    -----
    Equation: chance agreement = sum(class share squared); kappa = (agreement - chance) /
    (1 - chance).

    Assumptions: Both annotators have the same class marginal distribution; observed
    agreement is supplied unless the independent-raters case is selected.

    Excludes: Estimating reliability intervals, annotator-specific marginals, and task
    correctness.

    References
    ----------
    Derivation: chance agreement = sum(class share squared);
    kappa = (agreement - chance) / (1 - chance). The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.BinaryLabelChanceAgreement
    >>> round(float(case.chance_agreement), 3)
    0.82"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    class_shares: List[float]
    observed_agreement: Optional[float] = None
    # Optional (added 2026-10-20): raters who label independently of the item,
    # so their observed agreement is the chance agreement and kappa is zero.
    independent_raters: bool = False

    @property
    def chance_agreement(self) -> float:
        """Chance-agreement fraction from ``physics.statistics.calc_chance_agreement`` and
        equal class marginals.

        Equation: ``return calc_chance_agreement(class_shares)``."""
        from ...physics import calc_chance_agreement

        return calc_chance_agreement(self.class_shares)

    @property
    def agreement(self) -> float:
        """Supplied observed agreement, or chance agreement when the independent-raters flag
        is set.

        Equation: ``return chance_agreement if independent_raters else
        observed_agreement``."""
        return self.chance_agreement if self.independent_raters else self.observed_agreement

    @property
    def kappa(self) -> float:
        """Chance-corrected agreement from ``physics.statistics.calc_cohens_kappa``;
        dimensionless.

        Equation: ``return calc_cohens_kappa(agreement, chance_agreement)``."""
        from ...physics import calc_cohens_kappa

        return calc_cohens_kappa(self.agreement, self.chance_agreement)


class AgreementDriftScenario(BaseScenario):
    """A labeling campaign's kappa at a baseline and again after ``elapsed``.

    Notes
    -----
    Equation: kappa drop = baseline kappa - later kappa.

    Assumptions: The two kappa values concern comparable tasks and label definitions.

    Excludes: A fitted temporal decay rate or attribution of the change to a particular
    cause.

    References
    ----------
    Derivation: kappa drop = baseline kappa - later kappa.
    The individual properties state units and intermediate accounting; this class
    introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WakeWordLabelQualityDrift
    >>> round(float(case.drop), 3)
    0.13"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_kappa: float
    later_kappa: float
    elapsed: Quantity

    @property
    def drop(self) -> float:
        """Baseline minus later Cohen's kappa, a signed dimensionless change; elapsed time
        does not scale it.

        Equation: ``return baseline_kappa - later_kappa``."""
        return self.baseline_kappa - self.later_kappa


class KsDriftScenario(BaseScenario):
    """A two-sample K-S drift test of ``n_reference`` training against ``n_current`` recent values of one feature.

    Notes
    -----
    Equation: critical statistic = coefficient * sqrt((n_reference + n_current) /
    (n_reference * n_current)).

    Assumptions: The supplied coefficient matches the intended test level and the standard
    two-sample K-S assumptions; policy gate is distinct from significance.

    Excludes: A p-value, repeated-testing correction, and proof that distribution drift
    reduces model quality.

    References
    ----------
    Derivation: critical statistic = coefficient *
    sqrt((n_reference + n_current) / (n_reference * n_current)). The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.LoudnessDriftKs1000
    >>> round(float(case.scale), 3)
    0.045"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    coefficient: float  # Ops.Monitoring.KsTestCoefficient at the 0.05 level
    n_reference: int
    n_current: int
    observed_statistic: Optional[float] = None
    # Optional (added 2026-10-20): the significance level the coefficient is
    # for, when it is not the registry's 0.05 (calc_ks_coefficient(level)).
    significance: Optional[float] = None
    # Optional (added 2026-10-20): the K-S statistic above which a release
    # policy acts, compared with the critical value at this sample size.
    alert_gate: Optional[float] = None

    @property
    def gate_over_critical(self) -> float:
        """Policy alert gate divided by the statistical critical value, a dimensionless
        multiple.

        Equation: ``return alert_gate / critical_value``."""
        return self.alert_gate / self.critical_value

    @property
    def scale(self) -> float:
        """Two-sample size factor in the K-S threshold, excluding the confidence coefficient.

        Equation: ``return critical_value / coefficient``."""
        return self.critical_value / self.coefficient

    @property
    def critical_value(self) -> float:
        """Dimensionless K-S threshold from ``physics.statistics.calc_ks_critical_value``.

        Equation: ``return calc_ks_critical_value(coefficient, n_reference, n_current)``."""
        from ...physics import calc_ks_critical_value

        return calc_ks_critical_value(self.coefficient, self.n_reference, self.n_current)

    @property
    def flags(self) -> bool:
        """Whether the supplied observed K-S statistic exceeds the critical value; equality
        does not flag.

        Equation: ``return observed_statistic > critical_value``."""
        return self.observed_statistic > self.critical_value

    def flags_statistic(self, statistic: float) -> bool:
        """Whether a candidate dimensionless K-S statistic exceeds this scenario's critical
        value.

        Equation: ``return statistic > critical_value``."""
        return statistic > self.critical_value


class BinShareShiftScenario(BaseScenario):
    """A feature binned at the training quantiles: training shares against recent shares, scored by PSI.

    Notes
    -----
    Equation: PSI = sum((current share - baseline share) * log(current share / baseline
    share)).

    Assumptions: The lists refer to the same bins. Per-bin term access requires positive
    shares; the aggregate helper defines its own zero handling.

    Excludes: Causal drift diagnosis, uncertainty intervals, and a universal PSI alert
    threshold.

    References
    ----------
    Derivation: pSI = sum((current share - baseline share) *
    log(current share / baseline share)). The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FraudCountryPsi
    >>> round(float(case.psi), 3)
    0.065"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_shares: List[float]
    current_shares: List[float]
    bins: Optional[List[str]] = None  # bin names, aligned with the shares (added 2026-10-20)

    def term(self, name: str) -> float:
        """Dimensionless PSI contribution for the named bin, selected from ``terms``.

        Equation: ``return terms[bins.index(name)]``."""
        return self.terms[self.bins.index(name)]

    def shares(self, name: str):
        """Pair of baseline and current fractions for the named bin.

        Equation: ``i = bins.index(name); return (baseline_shares[i],
        current_shares[i])``."""
        i = self.bins.index(name)
        return self.baseline_shares[i], self.current_shares[i]

    @property
    def terms(self) -> List[float]:
        """Per-bin PSI contributions, each current-minus-baseline share times log of their
        ratio.

        Equation: ``return [(c - b) * math.log(c / b) for b, c in zip(baseline_shares,
        current_shares)]``."""
        import math

        return [(c - b) * math.log(c / b) for b, c in zip(self.baseline_shares, self.current_shares)]

    @property
    def psi(self) -> float:
        """Total dimensionless PSI from
        ``physics.statistics.calc_population_stability_index``.

        Equation: ``return calc_population_stability_index(baseline_shares,
        current_shares)``."""
        from ...physics import calc_population_stability_index

        return calc_population_stability_index(self.baseline_shares, self.current_shares)

    def ratio(self, i: int) -> float:
        """Current share divided by baseline share for indexed bin ``i``.

        Equation: ``return current_shares[i] / baseline_shares[i]``."""
        return self.current_shares[i] / self.baseline_shares[i]


class TwoProportionTestScenario(BaseScenario):
    """A two-arm test sized to detect an absolute ``lift`` on a ``baseline_rate`` at the
    normal quantiles ``z_alpha`` and ``z_beta``, on ``daily_requests`` split between arms.

    ``family_size`` uncorrected tests at ``significance`` give the familywise error.

    Notes
    -----
    Equation: The normal planning approximation uses ``2*p*(1-p)*(z_alpha+z_beta)**2/lift**2`` per arm.

    Assumptions: Independent Bernoulli samples, equal arms, a nonzero lift, and declared normal quantiles.

    Excludes: Sequential peeking, correlated samples, and multiplicity correction; familywise_error assumes independent tests.

    References
    ----------
    Derivation: Delegates sample planning to
    mlsysim.physics.statistics.calc_two_proportion_sample_size and detectable lift to
    calc_two_proportion_detectable_lift. Their equal-arm normal approximation combines
    two Bernoulli variances with significance/power quantiles. The familywise error uses
    the complement of all independent tests avoiding a false rejection.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.CtrLiftExperiment
    >>> s.total == 2*s.per_variant
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_rate: float
    lift: float
    z_alpha: float
    z_beta: float
    daily_requests: Optional[Quantity] = None
    significance: float = 0.05
    family_size: int = 20
    # Optional (added 2026-10-20): the confidence and power the quantiles come
    # from, the lift as a share of the baseline, and the users per arm that
    # traffic actually allows (for the detectable lift).
    confidence: Optional[float] = None
    power: Optional[float] = None
    relative_lift: Optional[float] = None
    available_per_variant: Optional[float] = None

    @property
    def z_sum_squared(self) -> float:
        """Squared sum of the supplied normal quantiles for significance and power.

        Equation: ``(z_alpha + z_beta) ** 2``."""
        return (self.z_alpha + self.z_beta) ** 2

    @property
    def two_variance(self) -> float:
        """Sum of two Bernoulli variances at the common planning baseline.

        Equation: ``2 * baseline_rate * (1 - baseline_rate)``."""
        return 2 * self.baseline_rate * (1 - self.baseline_rate)

    @property
    def lift_squared(self) -> float:
        """Squared absolute probability difference used for sample planning.

        Equation: ``lift ** 2``."""
        return self.lift**2

    @property
    def per_variant(self) -> float:
        """Continuous sample count per arm from the two-proportion normal approximation.

        Equation: ``calc_two_proportion_sample_size(baseline_rate, lift, z_alpha, z_beta)``."""
        from ...physics.statistics import calc_two_proportion_sample_size

        return calc_two_proportion_sample_size(self.baseline_rate, self.lift, self.z_alpha, self.z_beta)

    @property
    def total(self) -> float:
        """Combined continuous sample count across the two equally sized arms.

        Equation: ``2 * per_variant``."""
        return 2 * self.per_variant

    @property
    def target_rate(self) -> float:
        """Alternative response probability formed by adding absolute lift to baseline.

        Equation: ``baseline_rate + lift``."""
        return self.baseline_rate + self.lift

    def detectable_lift(self, n_per_variant=None) -> float:
        """Smallest absolute lift ``n_per_variant`` users per arm can detect (default:
        ``available_per_variant``). Added 2026-10-20.

        Equation: ``n = available_per_variant if n_per_variant is None else n_per_variant;
        calc_two_proportion_detectable_lift(baseline_rate, n, z_alpha, z_beta)``."""
        from ...physics.statistics import calc_two_proportion_detectable_lift

        n = self.available_per_variant if n_per_variant is None else n_per_variant
        return calc_two_proportion_detectable_lift(self.baseline_rate, n, self.z_alpha, self.z_beta)

    @property
    def collection_time(self) -> Quantity:
        """Hours to collect the combined sample target at the supplied daily request rate.

        Equation: ``(total / daily_requests).to('hour')``."""
        return (self.total / self.daily_requests).to("hour")

    @property
    def familywise_error(self) -> float:
        """Probability of at least one false rejection under independent tests with the supplied significance.

        Equation: ``calc_independent_success(significance, family_size)``."""
        from ...physics.agents import calc_independent_success

        return calc_independent_success(self.significance, self.family_size)


class MonitorFleetAlertScenario(BaseScenario):
    """``models`` x ``metrics_per_model`` monitors checked every ``check_interval`` at a per-check
    ``false_positive_rate`` on a stable process; ``dedup_share`` of crossings are merged away.

    Notes
    -----
    Equation: Expected crossings are ``models*metrics*checks*false_positive_rate``; independent-event complement gives any-alert probability.

    Assumptions: Per-check error probability is stable. Set dedup_share explicitly
    before requesting remaining_after_dedup; it is an assumed reduction fraction.

    Excludes: Temporal dependence for the any-alert calculation, incident precision, and learned deduplication behavior.

    References
    ----------
    Derivation: Linearity of expectation gives
    ``monitor_count*checks_per_monitor*per_check_error`` for expected crossings. A
    separately supplied deduplication fraction reduces that count. The any-alert
    probability additionally assumes independent checks and uses
    mlsysim.physics.agents.calc_independent_success for the complement of no crossings.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ThreeSigmaModelFleetAlerts
    >>> s = s.variant(dedup_share=0.5)
    >>> s.remaining_after_dedup <= s.daily_false_alerts
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    models: int
    metrics_per_model: int
    check_interval: Quantity
    false_positive_rate: float
    dedup_share: Optional[float] = None

    @property
    def monitors(self) -> int:
        """Count of monitored model-metric pairs.

        Equation: ``models * metrics_per_model``."""
        return self.models * self.metrics_per_model

    @property
    def checks_per_day(self) -> float:
        """Dimensionless number of checks per monitor in one day.

        Equation: ``(Q_(1, 'day') / check_interval).to('').magnitude``."""
        return (Q_(1, "day") / self.check_interval).to("").magnitude

    @property
    def daily_false_alerts(self) -> float:
        """Expected daily false crossings before deduplication.

        Equation: ``monitors * checks_per_day * false_positive_rate``."""
        return self.monitors * self.checks_per_day * self.false_positive_rate

    @property
    def remaining_after_dedup(self) -> float:
        """Expected daily false alerts after applying the assumed merged-away fraction.

        Equation: ``daily_false_alerts * (1 - dedup_share)``."""
        return self.daily_false_alerts * (1 - self.dedup_share)

    def any_false_alert(self, checks: int) -> float:
        """Chance that at least one of ``checks`` independent checks crosses falsely.

        Equation: ``calc_independent_success(false_positive_rate, checks)``."""
        from ...physics.agents import calc_independent_success

        return calc_independent_success(self.false_positive_rate, checks)


class LaplaceMeanReleaseScenario(BaseScenario):
    """The mean of a public roster of ``group_size`` values in ``[0, value_range]`` released
    under ``epsilon``-DP by one Laplace draw on the sum (replace-one adjacency).

    Notes
    -----
    Equation: Laplace scale equals sensitivity divided by epsilon; dividing that scale by
    roster size gives mean error.

    Assumptions: A public fixed-size roster, bounded values, replace-one adjacency, and
    positive epsilon.

    Excludes: Membership privacy of the roster and composition across repeated releases.

    References
    ----------
    Derivation: Laplace scale equals sensitivity divided by epsilon; dividing that scale by
    roster size gives mean error. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SalaryMeanLaplaceRelease
    >>> round(float(scenario.sensitivity.magnitude), 4)
    200000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    value_range: Quantity
    epsilon: float
    group_size: int
    small_group_size: Optional[int] = None

    @property
    def sensitivity(self) -> Quantity:
        """Replace-one sensitivity of the sum, in the same units as the permitted value range.

        The most one person can move the sum.

        Equation: ``value_range``."""
        return self.value_range

    @property
    def noise_scale(self) -> Quantity:
        """Laplace scale in value units from ``calc_laplace_scale(sensitivity, epsilon)``.

        Equation: ``calc_laplace_scale(sensitivity, epsilon)``."""
        from ...physics.privacy import calc_laplace_scale

        return calc_laplace_scale(self.sensitivity, self.epsilon)

    def mean_error(self, group_size=None) -> Quantity:
        """Expected absolute error of the released mean in value units; ``group_size`` overrides
        the configured positive roster size.

        Expected absolute error of the released mean (the Laplace scale over the group).

        Equation: ``n = group_size if group_size is None else group_size; result = noise_scale /
        n``."""
        n = self.group_size if group_size is None else group_size
        return self.noise_scale / n

    @property
    def small_group_error(self) -> Quantity:
        """Expected mean error for the optional small group; requires its configured size.

        Equation: ``mean_error(small_group_size)``."""
        return self.mean_error(self.small_group_size)


class MembershipAuditScenario(BaseScenario):
    """A membership-inference audit on the ROC plane against the ceiling a pure
    ``epsilon``-DP release allows; the audit curve ``FPR + a FPR (1 - FPR)`` is
    illustrative.

    Notes
    -----
    Equation: The DP ceiling delegates to the privacy helper; the audit curve adds a
    quadratic uplift to FPR.

    Assumptions: False-positive rates lie in [0,1] and the audit curve is constructed by the
    supplied shape parameter.

    Excludes: Empirical attack calibration and approximate-DP delta accounting.

    References
    ----------
    Derivation: The DP ceiling delegates to the privacy helper; the audit curve adds a
    quadratic uplift to FPR. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.MembershipAuditAtEpsilonOne
    >>> round(scenario.ceiling(0.1), 4)
    0.2718"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    epsilon: float
    audit_shape: float

    def ceiling(self, fpr) -> float:
        """Pure-epsilon-DP TPR ceiling at the dimensionless ``fpr``, from
        ``calc_membership_tpr_ceiling``.

        Equation: ``calc_membership_tpr_ceiling(fpr, epsilon)``."""
        from ...physics.privacy import calc_membership_tpr_ceiling

        return calc_membership_tpr_ceiling(fpr, self.epsilon)

    def audit(self, fpr) -> float:
        """Illustrative audit TPR at ``fpr``: a quadratic uplift controlled by ``audit_shape``; not
        a measured attack curve.

        Equation: ``fpr + audit_shape * fpr * (1 - fpr)``."""
        return fpr + self.audit_shape * fpr * (1 - fpr)


class MeanShiftTestScenario(BaseScenario):
    """A feature mean that moves from ``reference_mean`` to ``observed_mean`` over
    ``n_samples`` requests.

    The test assumes a known reference mean, independent requests, and the
    stated spread ``feature_sd``: z is the move in standard errors of the mean.

    Notes
    -----
    Equation: The standardized mean shift is the mean difference divided by the standard
    error.

    Assumptions: Independent samples, known reference mean, and supplied positive feature
    standard deviation.

    Excludes: Uncertainty in the reference mean and multiple-testing correction.

    References
    ----------
    Derivation: The standardized mean shift is the mean difference divided by the standard
    error. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FeatureMeanShiftTest
    >>> round(float(scenario.mean_shift), 4)
    0.05"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    reference_mean: float
    observed_mean: float
    feature_sd: float
    n_samples: int
    significance_bound: Optional[float] = None  # the bound prose compares the p-value with

    @property
    def mean_shift(self) -> float:
        """Observed-minus-reference feature mean, in the feature's numerical units.

        Equation: ``observed_mean - reference_mean``."""
        return self.observed_mean - self.reference_mean

    @property
    def standard_error(self) -> float:
        """Mean's standard error in feature units, using independent samples and the supplied
        feature spread.

        Equation: ``calc_standard_error(feature_sd, n_samples)``."""
        from ...physics import calc_standard_error

        return calc_standard_error(self.feature_sd, self.n_samples)

    @property
    def z_score(self) -> float:
        """Dimensionless mean shift divided by its standard error, treating the reference mean as
        known.

        Equation: ``mean_shift / standard_error``."""
        return self.mean_shift / self.standard_error

    @property
    def p_value(self) -> float:
        """Two-sided standard-normal tail probability at the absolute observed z-score.

        Two-sided normal-tail probability of a move at least this large from noise alone.

        Equation: ``2 * (1 - NormalDist().cdf(abs(z_score)))``."""
        from statistics import NormalDist

        return 2 * (1 - NormalDist().cdf(abs(self.z_score)))


class GaussianDriftScenario(BaseScenario):
    """A continuous feature fitted as a normal at reference and now: KL divergence of the
    reference from the current fit.

    Notes
    -----
    Equation: The statistics helpers decompose and evaluate KL(reference || current); the
    trigger compares that value with a threshold.

    Assumptions: Both feature distributions are Gaussian fits with positive standard
    deviations.

    Excludes: Proof of concept drift, non-Gaussian tails, and calibration of the supplied
    trigger.

    References
    ----------
    Derivation: The statistics helpers decompose and evaluate KL(reference || current); the
    trigger compares that value with a threshold. Individual property contracts identify the
    exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FraudAmountDrift
    >>> round(float(scenario.kl_terms[0]), 4)
    0.1671"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    reference_mean: float
    reference_sd: float
    current_mean: float
    current_sd: float
    kl_trigger: Optional[float] = None  # the service's investigate threshold on KL

    @property
    def triggers(self) -> bool:
        """Whether the reference-to-current Gaussian KL divergence reaches the supplied trigger.

        Equation: ``kl >= kl_trigger``."""
        return self.kl >= self.kl_trigger

    @property
    def kl_terms(self):
        """Log term, squared-error numerator, and variance denominator from
        ``calc_gaussian_kl_terms``.

        (log term, ratio numerator, ratio denominator) of the closed form.

        Equation: ``calc_gaussian_kl_terms(reference_mean, reference_sd, current_mean,
        current_sd)``."""
        from ...physics import calc_gaussian_kl_terms

        return calc_gaussian_kl_terms(self.reference_mean, self.reference_sd, self.current_mean, self.current_sd)

    @property
    def kl_log_term(self) -> float:
        """Dimensionless log-scale-ratio contribution to the Gaussian KL decomposition.

        Equation: ``kl_terms[0]``."""
        return self.kl_terms[0]

    @property
    def kl_ratio_term(self) -> float:
        """Dimensionless variance-and-mean-shift contribution before subtracting one half in the KL
        formula.

        Equation: ``_, num, den = kl_terms; result = num / den``."""
        _, num, den = self.kl_terms
        return num / den

    @property
    def kl(self) -> float:
        """Reference-to-current Gaussian KL divergence in nats, from ``calc_gaussian_kl``.

        Equation: ``calc_gaussian_kl(reference_mean, reference_sd, current_mean, current_sd)``."""
        from ...physics import calc_gaussian_kl

        return calc_gaussian_kl(self.reference_mean, self.reference_sd, self.current_mean, self.current_sd)


class CertifiedRadiusScenario(BaseScenario):
    """Binary randomized smoothing at noise ``noise_sd``, against an l-infinity budget on one
    input.

    The l-infinity budget is ``linf_levels`` steps of an image whose pixels take
    ``pixel_levels`` values; its box corner has Euclidean length
    ``budget * sqrt(coordinates)``, the length the certificate must reach to
    cover the whole box.

    Notes
    -----
    Equation: The smoothing helper gives an L2 radius; the L-infinity corner has norm
    sqrt(coordinates) times its coordinate budget.

    Assumptions: The supplied top-class probability is a valid smoothing probability bound
    under Gaussian noise.

    Excludes: Estimating that probability from samples or certifying a different noise
    distribution.

    References
    ----------
    Derivation: The smoothing helper gives an L2 radius; the L-infinity corner has norm
    sqrt(coordinates) times its coordinate budget. Individual property contracts identify
    the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SmoothingRadiusCifar10
    >>> round(float(scenario.radius), 4)
    1.5451"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    noise_sd: float
    top_class_probability: float
    linf_levels: int
    pixel_levels: int  # pixel values an 8-bit channel takes
    coordinates: int  # width * height * channels

    @property
    def radius(self) -> float:
        """Certified L2 radius in normalized input units under the supplied smoothed top-class
        probability.

        Equation: ``calc_smoothing_radius(noise_sd, top_class_probability)``."""
        from ...physics import calc_smoothing_radius

        return calc_smoothing_radius(self.noise_sd, self.top_class_probability)

    @property
    def quantile(self) -> float:
        """Dimensionless Gaussian quantile implicit in the smoothing radius divided by noise
        standard deviation.

        ``Phi^-1(p_A)``.

        Equation: ``radius / noise_sd``."""
        return self.radius / self.noise_sd

    @property
    def linf_budget(self) -> float:
        """Per-coordinate normalized perturbation budget from integer intensity levels.

        Equation: ``linf_levels / (pixel_levels - 1)``."""
        return self.linf_levels / (self.pixel_levels - 1)

    @property
    def box_corner_l2(self) -> float:
        """L2 norm of an L-infinity box corner over the declared number of coordinates.

        Equation: ``calc_linf_corner_l2(linf_budget, coordinates)``."""
        from ...physics import calc_linf_corner_l2

        return calc_linf_corner_l2(self.linf_budget, self.coordinates)

    @property
    def covers_box(self) -> bool:
        """Whether the smoothing radius contains the entire L-infinity perturbation box.

        Equation: ``radius >= box_corner_l2``."""
        return self.radius >= self.box_corner_l2


class FloatBitFlipScenario(BaseScenario):
    """One stored binary32 value with bit ``flipped_bit`` flipped (bit 30 is the top exponent
    bit).

    Notes
    -----
    Equation: Decode the original and XOR-flipped binary32 words, then compare their
    numerical scale.

    Assumptions: The stored representation is IEEE binary32; the ratio summary requires
    positive finite values.

    Excludes: Fault likelihood, ECC behavior, and propagation through a network.

    References
    ----------
    Derivation: Decode the original and XOR-flipped binary32 words, then compare their
    numerical scale. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Fp32WeightExponentFlip
    >>> round(float(scenario.stored), 4)
    0.042"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    value: float
    flipped_bit: int
    exponent_bits: int  # core.units.FLOAT_FORMAT_BITS["fp32"]
    fraction_bits: int

    @property
    def _pair(self):
        from ...physics import calc_float32_bit_flip

        return calc_float32_bit_flip(self.value, self.flipped_bit)

    @property
    def stored(self) -> float:
        """Original value rounded to IEEE binary32 before the selected bit is flipped.

        Equation: ``_pair[0]``."""
        return self._pair[0]

    @property
    def corrupted(self) -> float:
        """Value decoded from the binary32 word after the selected bit flip.

        Equation: ``_pair[1]``."""
        return self._pair[1]

    @property
    def scale_log2(self) -> int:
        """Rounded base-two logarithm of corrupted/original value; requires a positive finite
        ratio.

        The flip multiplies the value by 2 to this power.

        Equation: ``round(math.log2(corrupted / stored))``."""
        import math

        return round(math.log2(self.corrupted / self.stored))

    @property
    def word_bits(self) -> int:
        """Total sign, exponent, and fraction bit count of the configured representation.

        Equation: ``1 + exponent_bits + fraction_bits``."""
        return 1 + self.exponent_bits + self.fraction_bits


class ClosedLoopCampaignScenario(BaseScenario):
    """``clean_runs`` failure-free closed-loop trials of a frozen policy in one cell of
    ``table`` on
    ``site``, each ``run`` long with an automated ``reset``, bounded at one-sided
    ``confidence``.

    What a success target ``1 - target_rate`` costs in runs, cell time, and operator time;
    the
    exact bound when ``recorded_failures`` runs fail; and the contrasts the bound is taught
    with:
    a policy that fails one run in ``weak_policy_one_in``, ``fallacy_runs`` clean runs
    against the
    target, ``looks`` independent looks at the significance level, and a Bonferroni split of
    the
    significance level over ``joint_cells`` cells. A run that jams costs ``jam_reset``
    instead of
    ``reset``.

    Notes
    -----
    Equation: Binomial confidence helpers convert failure-free counts to bounds and invert
    them; cycle accounting adds run and reset time.

    Assumptions: Independent identically distributed trials of a frozen policy in a fixed
    evaluation cell; planned confidence is one-sided.

    Excludes: Distribution shift, adaptive test reuse, dependence between trials, and safety
    certification.

    References
    ----------
    Derivation: Binomial confidence helpers convert failure-free counts to bounds and invert
    them; cycle accounting adds run and reset time. Individual property contracts identify
    the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchEvaluationCampaign
    >>> round(float(scenario.alpha), 4)
    0.05"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (latch limits and approach speed)
    table: Any  # ScenarioTableCoverageScenario (the cells the campaign samples)
    standoff: Quantity
    run: Quantity
    reset: Quantity
    jam_reset: Quantity
    operators: int
    shift: Quantity
    clean_runs: int
    confidence: float
    target_rate: float  # failure rate the campaign argues toward
    estimate_target_rate: float  # failure rate of the trial-budget estimate
    weak_policy_one_in: int
    recorded_failures: int
    joint_cells: int
    looks: int
    omissions: int  # unclosed departures in the policy manifest's omission log
    fallacy_runs: int

    @property
    def alpha(self) -> float:
        """One-sided confidence tail probability, equal to one minus confidence.

        Equation: ``1.0 - confidence``."""
        return 1.0 - self.confidence

    @property
    def ln_inv_alpha(self) -> float:
        """Dimensionless negative logarithm of the confidence tail probability.

        The numerator of every zero-failure count, about 3 at 95 percent.

        Equation: ``-_math.log(alpha)``."""
        return -_math.log(self.alpha)

    def lower_bound_at(self, trials: int, confidence: Optional[float] = None) -> float:
        """Exact zero-failure success-probability lower bound for ``trials`` at the specified or
        configured confidence.

        Equation: ``calc_clopper_pearson_zero_failure_bound(trials, confidence if confidence is
        None else confidence)``."""
        from ...physics.robotics.assurance import calc_clopper_pearson_zero_failure_bound

        return calc_clopper_pearson_zero_failure_bound(trials, self.confidence if confidence is None else confidence)

    def runs_for(self, target: float, confidence: Optional[float] = None) -> int:
        """Independent failure-free trial count required for target success probability ``target``
        at the selected confidence.

        Equation: ``calc_zero_failure_sample_size(target, confidence if confidence is None else
        confidence)``."""
        from ...physics.robotics.assurance import calc_zero_failure_sample_size

        return calc_zero_failure_sample_size(target, self.confidence if confidence is None else confidence)

    @property
    def lower_bound(self) -> float:
        """Exact success-probability lower bound after the configured failure-free run count.

        Equation: ``lower_bound_at(clean_runs)``."""
        return self.lower_bound_at(self.clean_runs)

    @property
    def failure_upper(self) -> float:
        """Complement of the success lower bound, expressed as a failure probability.

        Equation: ``1.0 - lower_bound``."""
        return 1.0 - self.lower_bound

    @property
    def rule_of_three(self) -> float:
        """Approximate failure-probability ceiling after failure-free runs, using
        ``calc_rule_of_three_bound``.

        Equation: ``calc_rule_of_three_bound(clean_runs)``."""
        from ...physics.robotics.assurance import calc_rule_of_three_bound

        return calc_rule_of_three_bound(self.clean_runs)

    @property
    def target(self) -> float:
        """Success-probability target complementary to the specified target failure rate.

        Equation: ``1.0 - target_rate``."""
        return 1.0 - self.target_rate

    @property
    def nine_targets(self) -> List[float]:
        """Success probabilities with successive decimal nines up to the configured target rate's
        order.

        Success targets with one, two, ... nines, up to the campaign target (0.9, 0.99, 0.999).

        Equation: ``nines = round(-_math.log10(target_rate)); result = [1.0 - 10.0 ** (-k) for k
        in range(1, nines + 1)]``."""
        nines = round(-_math.log10(self.target_rate))
        return [1.0 - 10.0**-k for k in range(1, nines + 1)]

    @property
    def target_runs(self) -> int:
        """Failure-free trials required to demonstrate the configured success target.

        Equation: ``runs_for(target)``."""
        return self.runs_for(self.target)

    @property
    def joint_runs(self) -> int:
        """Failure-free trials per cell with Bonferroni allocation of the confidence tail across
        ``joint_cells``.

        Runs per cell when the significance level is split evenly over ``joint_cells``.

        Equation: ``runs_for(target, 1.0 - alpha / joint_cells)``."""
        return self.runs_for(self.target, 1.0 - self.alpha / self.joint_cells)

    @property
    def estimate_target(self) -> float:
        """Success probability complementary to the separate estimation target failure rate.

        Equation: ``1.0 - estimate_target_rate``."""
        return 1.0 - self.estimate_target_rate

    @property
    def estimate_runs(self) -> int:
        """Failure-free trial count for the separate estimation target under the configured
        confidence.

        Equation: ``runs_for(estimate_target)``."""
        return self.runs_for(self.estimate_target)

    @property
    def estimate_rule_of_three_runs(self) -> float:
        """Approximate trial requirement for the estimation target from the rule of three.

        Equation: ``calc_rule_of_three_trials(estimate_target)``."""
        from ...physics.robotics.assurance import calc_rule_of_three_trials

        return calc_rule_of_three_trials(self.estimate_target)

    def clean_probability(self, success: float) -> float:
        """Probability that all configured runs succeed at per-run success probability ``success``,
        assuming independence.

        Chance that a policy of per-run success ``success`` logs the campaign clean.

        Equation: ``success ** clean_runs``."""
        return success**self.clean_runs

    @property
    def weak_policy_success(self) -> float:
        """Per-run success probability for a policy failing once per ``weak_policy_one_in`` trials
        on average.

        Equation: ``1.0 - 1.0 / weak_policy_one_in``."""
        return 1.0 - 1.0 / self.weak_policy_one_in

    @property
    def weak_policy_clean_probability(self) -> float:
        """Probability the weaker policy passes the full campaign without a failure.

        Equation: ``clean_probability(weak_policy_success)``."""
        return self.clean_probability(self.weak_policy_success)

    @property
    def lower_bound_with_failures(self) -> float:
        """Exact success lower bound after subtracting recorded failures from the configured total
        runs.

        Equation: ``calc_clopper_pearson_lower_bound(clean_runs - recorded_failures, clean_runs,
        confidence)``."""
        from ...physics.robotics.assurance import calc_clopper_pearson_lower_bound

        return calc_clopper_pearson_lower_bound(
            self.clean_runs - self.recorded_failures, self.clean_runs, self.confidence
        )

    @property
    def observed_rate_with_failures(self) -> float:
        """Observed success fraction after recorded failures, without a confidence adjustment.

        Equation: ``(clean_runs - recorded_failures) / clean_runs``."""
        return (self.clean_runs - self.recorded_failures) / self.clean_runs

    @property
    def fallacy_lower_bound(self) -> float:
        """Success lower bound from the smaller illustrative ``fallacy_runs`` campaign.

        Equation: ``lower_bound_at(fallacy_runs)``."""
        return self.lower_bound_at(self.fallacy_runs)

    @property
    def any_false_positive(self) -> float:
        """Probability of at least one false positive across independent looks at the configured
        tail probability.

        At least one false positive across ``looks`` independent looks at ``alpha``.

        Equation: ``calc_independent_success(alpha, looks)``."""
        from ...physics.agents import calc_independent_success

        return calc_independent_success(self.alpha, self.looks)

    @property
    def cycle(self) -> Quantity:
        """Seconds per trial including the normal reset time.

        Equation: ``(run + reset).to('s')``."""
        return (self.run + self.reset).to("s")

    def campaign_time(self, trials: Optional[int] = None) -> Quantity:
        """Total seconds for ``trials`` independent serial cycles, defaulting to the configured
        campaign count.

        Equation: ``((clean_runs if trials is None else trials) * cycle).to('s')``."""
        return ((self.clean_runs if trials is None else trials) * self.cycle).to("s")

    def operator_time(self, trials: int) -> Quantity:
        """Operator-hours for the serial campaign with the declared concurrent staffing count.

        Equation: ``(operators * campaign_time(trials)).to('hour')``."""
        return (self.operators * self.campaign_time(trials)).to("hour")

    def shifts(self, trials: int) -> float:
        """Dimensionless staffed shift equivalents required for the campaign's operator-hours.

        Equation: ``(operator_time(trials) / shift).to('').magnitude``."""
        return (self.operator_time(trials) / self.shift).to("").magnitude

    @property
    def jam_reset_share(self) -> float:
        """Fraction of a jammed trial cycle spent resetting rather than running.

        Equation: ``(jam_reset / (run + jam_reset)).to('').magnitude``."""
        return (self.jam_reset / (self.run + self.jam_reset)).to("").magnitude

    @property
    def jam_cycle(self) -> Quantity:
        """Seconds per trial when the longer jam-reset time replaces the normal reset.

        Equation: ``(run + jam_reset).to('s')``."""
        return (self.run + self.jam_reset).to("s")


class SequentialTestScenario(BaseScenario):
    """Wald's SPRT between an unacceptable failure rate ``p_unacceptable`` and an acceptable
    one
    ``p_acceptable`` at error rates ``alpha`` and ``beta``; the score after
    ``trace_clean_runs``
    clean runs and one failure; and the fixed zero-failure test for success ``1 -
    p_unacceptable``
    at the same ``alpha``.

    Notes
    -----
    Equation: Success and failure log-likelihood increments accumulate until a Wald boundary
    is crossed.

    Assumptions: Independent Bernoulli outcomes under the two fixed hypotheses and the
    specified SPRT error budgets.

    Excludes: Changing hypotheses, dependent outcomes, and a guaranteed stopping-time
    distribution.

    References
    ----------
    Derivation: Success and failure log-likelihood increments accumulate until a Wald
    boundary is crossed. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchSequentialTest
    >>> round(float(scenario.success_increment), 4)
    0.0953"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    p_unacceptable: float
    p_acceptable: float
    alpha: float
    beta: float
    trace_clean_runs: int

    @property
    def success_increment(self) -> float:
        """Log-likelihood increment after a successful trial, from ``calc_sprt_increments``.

        Equation: ``calc_sprt_increments(p_unacceptable, p_acceptable)[0]``."""
        from ...physics.statistics import calc_sprt_increments

        return calc_sprt_increments(self.p_unacceptable, self.p_acceptable)[0]

    @property
    def failure_increment(self) -> float:
        """Log-likelihood increment after a failed trial under the same two Bernoulli hypotheses.

        Equation: ``calc_sprt_increments(p_unacceptable, p_acceptable)[1]``."""
        from ...physics.statistics import calc_sprt_increments

        return calc_sprt_increments(self.p_unacceptable, self.p_acceptable)[1]

    @property
    def accept(self) -> float:
        """Upper log-likelihood boundary for accepting the lower-failure hypothesis.

        Equation: ``calc_sprt_boundaries(alpha, beta)[0]``."""
        from ...physics.statistics import calc_sprt_boundaries

        return calc_sprt_boundaries(self.alpha, self.beta)[0]

    @property
    def reject(self) -> float:
        """Lower log-likelihood boundary for rejecting the lower-failure hypothesis.

        Equation: ``calc_sprt_boundaries(alpha, beta)[1]``."""
        from ...physics.statistics import calc_sprt_boundaries

        return calc_sprt_boundaries(self.alpha, self.beta)[1]

    @property
    def runs_to_accept(self) -> int:
        """Ceiling number of consecutive successful trials needed to reach the acceptance boundary
        from zero.

        Equation: ``_math.ceil(accept / success_increment)``."""
        return _math.ceil(self.accept / self.success_increment)

    @property
    def false_accept_bound(self) -> float:
        """Nominal exponential false-accept bound implied by the upper boundary.

        Ville's inequality: ``e^-A = alpha / (1 - beta)``.

        Equation: ``_math.exp(-accept)``."""
        return _math.exp(-self.accept)

    @property
    def false_reject_bound(self) -> float:
        """Nominal exponential false-reject bound implied by the lower boundary.

        Equation: ``_math.exp(reject)``."""
        return _math.exp(self.reject)

    @property
    def trace_score(self) -> float:
        """Log-likelihood score of the illustrative trace: declared successful trials plus one
        failure.

        Equation: ``trace_clean_runs * success_increment + failure_increment``."""
        return self.trace_clean_runs * self.success_increment + self.failure_increment

    @property
    def fixed_runs(self) -> int:
        """Zero-failure fixed-sample trial count for the unacceptable-rate ceiling at confidence
        one minus alpha.

        Equation: ``calc_zero_failure_sample_size(1.0 - p_unacceptable, 1.0 - alpha)``."""
        from ...physics.robotics.assurance import calc_zero_failure_sample_size

        return calc_zero_failure_sample_size(1.0 - self.p_unacceptable, 1.0 - self.alpha)

    @property
    def fixed_rejects_acceptable(self) -> float:
        """Probability the fixed zero-failure rule sees at least one failure under the acceptable
        rate.

        Chance that a policy at the acceptable rate fails the fixed zero-failure test.

        Equation: ``1.0 - (1.0 - p_acceptable) ** fixed_runs``."""
        return 1.0 - (1.0 - self.p_acceptable) ** self.fixed_runs


class ArmComparisonScenario(BaseScenario):
    """Two policies run ``arm_runs`` times each, with ``successes_a`` and ``successes_b``
    successes; Fisher's exact test, and the runs per arm that separate ``p_high`` from
    ``p_low``
    two-sided at ``alpha`` with ``power``.

    Notes
    -----
    Equation: Observed tables use Fisher's exact test; planning uses the arcsine effect size
    and normal-quantile approximation.

    Assumptions: Independent equal-sized arms and fixed planned rates for sample-size
    calculations.

    Excludes: Paired-outcome analysis, sequential peeking, and a causal conclusion from a
    p-value alone.

    References
    ----------
    Derivation: Observed tables use Fisher's exact test; planning uses the arcsine effect
    size and normal-quantile approximation. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchCheckpointComparison
    >>> round(float(scenario.fisher_p), 4)
    0.106"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    arm_runs: int
    successes_a: int
    successes_b: int
    alpha: float
    power: float
    p_high: float
    p_low: float

    @property
    def fisher_p(self) -> float:
        """Two-sided Fisher exact p-value for the two equal-size policy arms' success/failure
        table.

        Equation: ``n = arm_runs; result = calc_fisher_exact_two_sided(successes_a, n -
        successes_a, successes_b, n - successes_b)``."""
        from ...physics.statistics import calc_fisher_exact_two_sided

        n = self.arm_runs
        return calc_fisher_exact_two_sided(
            self.successes_a, n - self.successes_a, self.successes_b, n - self.successes_b
        )

    @property
    def total_runs(self) -> int:
        """Total trial count across the two equal-size arms.

        Equation: ``2 * arm_runs``."""
        return 2 * self.arm_runs

    @property
    def failures(self) -> int:
        """Combined observed failure count across both arms.

        Equation: ``total_runs - successes_a - successes_b``."""
        return self.total_runs - self.successes_a - self.successes_b

    @property
    def all_in_one_arm(self) -> float:
        """Conditional probability that all observed failures fall in one designated arm under
        random allocation.

        Chance that every failure lands in one named arm under the null.

        Equation: ``_math.comb(arm_runs, failures) / _math.comb(2 * arm_runs, failures)``."""
        return _math.comb(self.arm_runs, self.failures) / _math.comb(2 * self.arm_runs, self.failures)

    @property
    def z_alpha(self) -> float:
        """Standard-normal quantile for the two-sided significance level.

        Equation: ``NormalDist().inv_cdf(1 - alpha / 2)``."""
        from statistics import NormalDist

        return NormalDist().inv_cdf(1 - self.alpha / 2)

    @property
    def z_power(self) -> float:
        """Standard-normal quantile for the target detection power.

        Equation: ``NormalDist().inv_cdf(power)``."""
        from statistics import NormalDist

        return NormalDist().inv_cdf(self.power)

    @property
    def cohens_h(self) -> float:
        """Absolute arcsine-transformed effect between the planned high and low success rates.

        Equation: ``calc_cohens_h(p_high, p_low)``."""
        from ...physics.statistics import calc_cohens_h

        return calc_cohens_h(self.p_high, self.p_low)

    def runs_per_arm_for(self, effect_h: float) -> int:
        """Approximate equal-arm sample size for ``effect_h`` under the stated significance and
        power.

        Equation: ``calc_arcsine_two_arm_sample_size(effect_h, alpha, power)``."""
        from ...physics.statistics import calc_arcsine_two_arm_sample_size

        return calc_arcsine_two_arm_sample_size(effect_h, self.alpha, self.power)

    @property
    def runs_per_arm(self) -> int:
        """Approximate sample size per arm for the configured planned effect.

        Equation: ``runs_per_arm_for(cohens_h)``."""
        return self.runs_per_arm_for(self.cohens_h)


class HazardExposureScenario(BaseScenario):
    """Failure-free exposure that bounds each of ``rates`` (constant hazards per hour) at
    one-sided
    ``confidence``, against what ``rigs`` rigs or one rig over ``rig_time`` accumulate.

    Notes
    -----
    Equation: The zero-event Poisson relation ``exp(-rate*time)``=1-confidence gives exposure by
    inversion.

    Assumptions: A constant hazard, independent exposure, and no observed failures; rig
    exposure pools linearly.

    Excludes: Changing operating conditions, reset downtime beyond the supplied duty
    schedule, and field-safety certification.

    References
    ----------
    Derivation: The zero-event Poisson relation ``exp(-rate*time)``=1-confidence gives exposure
    by inversion. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.HazardExposureWall
    >>> round(float(scenario.ln_inv_alpha), 4)
    2.9957"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rates: List[Quantity]
    confidence: float
    rigs: int
    rig_time: Optional[Quantity] = None  # optional (2026-10-20): needed only for rig_rate
    # Optional (added 2026-10-20): operating hours each rig runs per calendar day,
    # so a fleet's failure-free exposure converts to the days a stage must run.
    hours_per_day: Optional[Quantity] = None

    @property
    def ln_inv_alpha(self) -> float:
        """Dimensionless negative logarithm of the one-sided confidence tail.

        Equation: ``-_math.log(1.0 - confidence)``."""
        return -_math.log(1.0 - self.confidence)

    def exposure(self, rate: Quantity) -> Quantity:
        """Failure-free operating hours needed to bound the supplied constant hazard ``rate``.

        Equation: ``calc_empirical_testing_exposure(rate, confidence=confidence).to('hour')``."""
        from ...physics.robotics.assurance import calc_empirical_testing_exposure

        return calc_empirical_testing_exposure(rate, confidence=self.confidence).to("hour")

    def fleet_time(self, rate: Quantity) -> Quantity:
        """Hours per rig when the required exposure is shared evenly across the independent rigs.

        Equation: ``(exposure(rate) / rigs).to('hour')``."""
        return (self.exposure(rate) / self.rigs).to("hour")

    @property
    def rig_rate(self) -> Quantity:
        """Smallest constant hazard in inverse hours that one rig's failure-free ``rig_time`` can
        bound.

        The smallest constant rate one rig running for ``rig_time`` can bound.

        Equation: ``(ln_inv_alpha / rig_time).to('1/hour')``."""
        return (self.ln_inv_alpha / self.rig_time).to("1/hour")

    def shortfall(self, rate: Quantity) -> float:
        """Dimensionless ratio of the one-rig demonstrated rate to the target ``rate``.

        Equation: ``(rig_rate / rate).to('').magnitude``."""
        return (self.rig_rate / rate).to("").magnitude

    def calendar_days(self, rate: Quantity) -> Quantity:
        """Calendar days needed at the supplied per-rig operating hours per day.

        Days the ``rigs`` need, at ``hours_per_day`` each, to accumulate the exposure (added
        2026-10-20).

        Equation: ``(fleet_time(rate) / hours_per_day).to('day')``."""
        return (self.fleet_time(rate) / self.hours_per_day).to("day")


class MissionExposureScenario(BaseScenario):
    """Failure-free missions of ``mission`` at ``machine``'s drive limit, with ``reset``
    between
    them, that bound ``hazard_rate`` at one-sided ``confidence``; and the exposure a relaxed
    target of one failure per ``relaxed_mtbf`` needs.

    Notes
    -----
    Equation: Failure-free exposure converts to mission count; adding resets converts active
    exposure to campaign time.

    Assumptions: Constant hazard, repeated equal missions, and travel at the registered
    drive limit.

    Excludes: Hazard variation, acceleration transients, and wear-dependent failure rates.

    References
    ----------
    Derivation: Failure-free exposure converts to mission count; adding resets converts
    active exposure to campaign time. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseMissionExposure
    >>> round(float(scenario.mission_distance.magnitude), 4)
    540.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # MobileManipulator (drive limit, wheel diameter)
    hazard_rate: Quantity
    mission: Quantity
    reset: Quantity
    confidence: float
    relaxed_mtbf: Quantity

    @property
    def mission_distance(self) -> Quantity:
        """Meters traveled at the machine's maximum drive speed during one mission.

        Equation: ``(machine.base.max_velocity * mission).to('m')``."""
        return (self.machine.base.max_velocity * self.mission).to("m")

    @property
    def cycle(self) -> Quantity:
        """Seconds per mission including the reset interval.

        Equation: ``(mission + reset).to('s')``."""
        return (self.mission + self.reset).to("s")

    @property
    def continuous_exposure(self) -> Quantity:
        """Failure-free active hours needed for the constant-hazard target, excluding reset time.

        Equation: ``calc_empirical_testing_exposure(hazard_rate,
        confidence=confidence).to('hour')``."""
        from ...physics.robotics.assurance import calc_empirical_testing_exposure

        return calc_empirical_testing_exposure(self.hazard_rate, confidence=self.confidence).to("hour")

    @property
    def mission_failure_probability(self) -> float:
        """Rare-event approximation hazard times mission duration, a dimensionless probability
        estimate.

        Equation: ``(hazard_rate * mission).to('').magnitude``."""
        return (self.hazard_rate * self.mission).to("").magnitude

    @property
    def missions_needed(self) -> float:
        """Failure-free mission count required by ``calc_zero_failure_mission_count``.

        Equation: ``calc_zero_failure_mission_count(hazard_rate, mission, confidence)``."""
        from ...physics.robotics.assurance import calc_zero_failure_mission_count

        return calc_zero_failure_mission_count(self.hazard_rate, self.mission, self.confidence)

    @property
    def calendar_time(self) -> Quantity:
        """Campaign hours including reset time between the modeled missions.

        Equation: ``(missions_needed * cycle).to('hour')``."""
        return (self.missions_needed * self.cycle).to("hour")

    @property
    def wheel_revolutions(self) -> float:
        """Total driven wheel revolutions at the stated diameter over the required mission
        distance.

        Equation: ``(missions_needed * mission_distance / (_math.pi *
        machine.drive_unit.wheel_diameter)).to('').magnitude``."""
        return (
            (self.missions_needed * self.mission_distance / (_math.pi * self.machine.drive_unit.wheel_diameter))
            .to("")
            .magnitude
        )

    @property
    def relaxed_exposure(self) -> Quantity:
        """Failure-free operating hours for the less demanding reciprocal-MTBF target.

        Equation: ``calc_empirical_testing_exposure(1 / relaxed_mtbf,
        confidence=confidence).to('hour')``."""
        from ...physics.robotics.assurance import calc_empirical_testing_exposure

        return calc_empirical_testing_exposure(1 / self.relaxed_mtbf, confidence=self.confidence).to("hour")


class InterventionCampaignScenario(BaseScenario):
    """``interventions`` interventions in ``missions`` supervised runs of ``mission``: burden
    per
    hour and per kilometer, and an exact two-sided ``1 - alpha`` interval on the rate.

    Notes
    -----
    Equation: Count divided by exposure gives the rate; exact Poisson count limits are
    divided by the same exposure.

    Assumptions: Event counts follow a homogeneous Poisson model over the supplied active
    exposure.

    Excludes: Dependence among interventions, severity differences, and zero-count MTBI
    division.

    References
    ----------
    Derivation: Count divided by exposure gives the rate; exact Poisson count limits are
    divided by the same exposure. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.WarehouseSupervisedMissions
    >>> round(float(scenario.active_time.magnitude), 4)
    10.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mission: MissionExposureScenario
    missions: int
    interventions: int
    alpha: float

    @property
    def active_time(self) -> Quantity:
        """Total mission hours excluding resets for the supervised campaign.

        Equation: ``(missions * mission.mission).to('hour')``."""
        return (self.missions * self.mission.mission).to("hour")

    @property
    def distance(self) -> Quantity:
        """Total kilometers traveled at the mission's modeled speed.

        Equation: ``(missions * mission.mission_distance).to('km')``."""
        return (self.missions * self.mission.mission_distance).to("km")

    @property
    def mtbi(self) -> Quantity:
        """Mean active minutes between interventions; requires a positive intervention count.

        Equation: ``(active_time / interventions).to('minute')``."""
        return (self.active_time / self.interventions).to("minute")

    @property
    def density(self) -> Quantity:
        """Observed intervention count per kilometer traveled.

        Equation: ``(interventions / distance).to('1/km')``."""
        return (self.interventions / self.distance).to("1/km")

    @property
    def rate(self) -> Quantity:
        """Observed intervention count per active operating hour.

        Equation: ``(interventions / active_time).to('1/hour')``."""
        return (self.interventions / self.active_time).to("1/hour")

    @property
    def count_interval(self):
        """Two-sided exact Poisson interval for the campaign event count, from
        ``calc_poisson_exact_interval``.

        Equation: ``calc_poisson_exact_interval(interventions, alpha)``."""
        from ...physics.statistics import calc_poisson_exact_interval

        return calc_poisson_exact_interval(self.interventions, self.alpha)

    @property
    def rate_interval(self):
        """Lower and upper intervention rates per hour obtained by dividing count limits by active
        exposure.

        Equation: ``lo, hi = count_interval; result = ((lo / active_time).to('1/hour'), (hi /
        active_time).to('1/hour'))``."""
        lo, hi = self.count_interval
        return (lo / self.active_time).to("1/hour"), (hi / self.active_time).to("1/hour")


class RoadExposureScenario(BaseScenario):
    """Miles-to-demonstrate requirements per event class (zero failures, precision,
    improvement),
    in millions of miles at human benchmark ``rates`` per 100 million miles, against one
    dated
    ``fleet_miles`` total (millions) at one-sided ``confidence``.

    Notes
    -----
    Equation: Zero-failure exposure inverts the constant-rate zero-event probability; the
    other requirements are supplied reference values.

    Assumptions: Supplied event rates and exposure-requirement tables share the stated
    million-mile conventions.

    Excludes: Re-estimation of precision/improvement tables, fleet-domain comparability, and
    an autonomous-driving safety claim.

    References
    ----------
    Derivation: Zero-failure exposure inverts the constant-rate zero-event probability; the
    other requirements are supplied reference values. Individual property contracts identify
    the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RoadMilesToDemonstrate
    >>> round(float(scenario.cleared[0]), 4)
    3.9"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rates: Dict[str, float]
    zero_failure: Dict[str, float]
    precision: Dict[str, float]
    improvement: Dict[str, float]
    fleet_miles: float
    confidence: float

    def recomputed_zero_failure(self, event: str) -> float:
        """Required failure-free exposure in millions of miles from -log(1-confidence) divided by
        the event rate.

        ``-ln(1 - C) / rate``, in millions of miles: RAND's row (1) re-derived.

        Equation: ``-_math.log(1.0 - confidence) / (rates[event] / 100.0)``."""
        return -_math.log(1.0 - self.confidence) / (self.rates[event] / 100.0)

    def requirements(self, event: str):
        """Supplied zero-failure, precision, and improvement exposure requirements for ``event``,
        each in millions of miles.

        RAND's three rows for one event class: zero failures, precision, improvement (millions).

        Equation: ``tuple((float(t[event]) for t in (zero_failure, precision, improvement)))``."""
        return tuple(float(t[event]) for t in (self.zero_failure, self.precision, self.improvement))

    def shortfall(self, event: str) -> float:
        """Dimensionless zero-failure exposure requirement divided by the reported fleet exposure
        for ``event``.

        Equation: ``zero_failure[event] / fleet_miles``."""
        return self.zero_failure[event] / self.fleet_miles

    def precision_over_fleet(self, event: str) -> float:
        """Dimensionless precision exposure requirement divided by fleet exposure for ``event``.

        Equation: ``precision[event] / fleet_miles``."""
        return self.precision[event] / self.fleet_miles

    @property
    def cleared(self) -> List[float]:
        """Supplied exposure requirements in millions of miles that are below the fleet's reported
        total.

        Requirements, of every event class and question, the fleet total exceeds.

        Equation: Select supplied requirements r for which r<fleet_miles; returned values retain
        million-mile units."""
        rows = list(self.zero_failure.values()) + list(self.precision.values()) + list(self.improvement.values())
        return [m for m in rows if m < self.fleet_miles]


class CrashRateDenominatorScenario(BaseScenario):
    """A reported before-and-after crash rate and the same comparison on the cohort whose
    exposure
    is complete (rates per million miles).

    Notes
    -----
    Equation: Each fractional change is after divided by before minus one; the denominator
    choice remains explicit.

    Assumptions: Before/after rates are comparable within each supplied denominator
    definition.

    Excludes: Causal treatment effects and correction for missing exposure beyond the
    supplied cohort rates.

    References
    ----------
    Derivation: Each fractional change is after divided by before minus one; the denominator
    choice remains explicit. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AutosteerCrashRateDenominator
    >>> round(float(scenario.reported_change), 4)
    -0.3846"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    reported_before: float
    reported_after: float
    cohort_before: float
    cohort_after: float
    exact_exposure_share: float  # fraction of vehicles with exact exposure

    @property
    def reported_change(self) -> float:
        """Fractional after/before change in the reported crash rate.

        Equation: ``reported_after / reported_before - 1.0``."""
        return self.reported_after / self.reported_before - 1.0

    @property
    def cohort_change(self) -> float:
        """Fractional after/before change in the complete-exposure cohort's crash rate.

        Equation: ``cohort_after / cohort_before - 1.0``."""
        return self.cohort_after / self.cohort_before - 1.0


class GroupConfusionScenario(BaseScenario):
    """Binary decisions for two or more groups as confusion counts ``{group: {tp, fp, fn,
    tn}}``.

    A positive decision is an approval (or a flag); the positive label is the
    outcome the decision should select. ``gap`` compares the first two groups,
    first minus second.

    Notes
    -----
    Equation: Each metric is the stated confusion-count numerator divided by its
    corresponding population denominator.

    Assumptions: Supplied confusion counts share the same outcome definition and every
    requested rate has a nonzero denominator.

    Excludes: Sampling uncertainty, causal fairness claims, and metrics for unspecified
    groups.

    References
    ----------
    Derivation: Each metric is the stated confusion-count numerator divided by its
    corresponding population denominator. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CreditServiceTwoGroups
    >>> round(float(scenario.pooled_positive_rate), 4)
    0.525"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    groups: Dict[str, Dict[str, float]]

    def _c(self, group: str) -> Dict[str, float]:
        return self.groups[group]

    @property
    def first(self) -> str:
        """First group label in insertion order, used as the signed-gap reference group.

        Equation: ``list(groups)[0]``."""
        return list(self.groups)[0]

    @property
    def second(self) -> str:
        """Second group label in insertion order, used as the signed-gap comparison group.

        Equation: ``list(groups)[1]``."""
        return list(self.groups)[1]

    def size(self, group: str) -> float:
        """Total decisions in the named group, summing all confusion-table cells.

        Equation: ``c = _c(group); result = c['tp'] + c['fp'] + c['fn'] + c['tn']``."""
        c = self._c(group)
        return c["tp"] + c["fp"] + c["fn"] + c["tn"]

    def positives(self, group: str) -> float:
        """Predicted-positive count in the group, summing true and false positives.

        Positive decisions (approved or flagged).

        Equation: ``c = _c(group); result = c['tp'] + c['fp']``."""
        c = self._c(group)
        return c["tp"] + c["fp"]

    def positive_rate(self, group: str) -> float:
        """Predicted-positive fraction among all decisions in the group.

        Share of the group given a positive decision (the demographic-parity quantity).

        Equation: ``positives(group) / size(group)``."""
        return self.positives(group) / self.size(group)

    def tpr(self, group: str) -> float:
        """True-positive fraction among actual positives in the group.

        Equation: ``c = _c(group); result = c['tp'] / (c['tp'] + c['fn'])``."""
        c = self._c(group)
        return c["tp"] / (c["tp"] + c["fn"])

    def fpr(self, group: str) -> float:
        """False-positive fraction among actual negatives in the group.

        Equation: ``c = _c(group); result = c['fp'] / (c['fp'] + c['tn'])``."""
        c = self._c(group)
        return c["fp"] / (c["fp"] + c["tn"])

    def ppv(self, group: str) -> float:
        """True-positive fraction among predicted positives in the group.

        Equation: ``c = _c(group); result = c['tp'] / (c['tp'] + c['fp'])``."""
        c = self._c(group)
        return c["tp"] / (c["tp"] + c["fp"])

    def base_rate(self, group: str) -> float:
        """Actual-positive fraction among all decisions in the group.

        Equation: ``c = _c(group); result = (c['tp'] + c['fn']) / size(group)``."""
        c = self._c(group)
        return (c["tp"] + c["fn"]) / self.size(group)

    def fnr(self, group: str) -> float:
        """False-negative fraction among actual positives in the group.

        Share of the positive-label members given a negative decision (added 2026-10-20).

        Equation: ``c = _c(group); result = c['fn'] / (c['tp'] + c['fn'])``."""
        c = self._c(group)
        return c["fn"] / (c["tp"] + c["fn"])

    @property
    def pooled_positive_rate(self) -> float:
        """Predicted-positive fraction pooled by decision count across all groups.

        Share of all members, every group pooled, given a positive decision (added 2026-10-20).

        Equation: ``sum((positives(g) for g in groups)) / sum((size(g) for g in groups))``."""
        return sum(self.positives(g) for g in self.groups) / sum(self.size(g) for g in self.groups)

    def accuracy(self, group: str) -> float:
        """Correct-decision fraction in the named group.

        Share of the group decided correctly (added 2026-10-20).

        Equation: ``c = _c(group); result = (c['tp'] + c['tn']) / size(group)``."""
        c = self._c(group)
        return (c["tp"] + c["tn"]) / self.size(group)

    @property
    def overall_accuracy(self) -> float:
        """Correct-decision fraction pooled by decision count across all groups.

        Share of all members of all groups decided correctly (added 2026-10-20).

        Equation: ``right = sum((c['tp'] + c['tn'] for c in groups.values())); result = right /
        sum((size(g) for g in groups))``."""
        right = sum(c["tp"] + c["tn"] for c in self.groups.values())
        return right / sum(self.size(g) for g in self.groups)

    def share(self, group: str) -> float:
        """Named group's fraction of the combined decision population.

        The group's share of all members (added 2026-10-20).

        Equation: ``size(group) / sum((size(g) for g in groups))``."""
        return self.size(group) / sum(self.size(g) for g in self.groups)

    def gap(self, metric: str) -> float:
        """Signed first-minus-second group difference for the named rate ``metric``.

        ``metric`` (positive_rate, tpr, fpr, ppv, base_rate) of the first group minus the
        second's.

        Equation: metric(first group)-metric(second group); metric names select the documented
        rate methods."""
        f = getattr(self, metric)
        return f(self.first) - f(self.second)

    @property
    def disparate_impact(self) -> float:
        """Smaller divided by larger group positive rate, a dimensionless selection-rate ratio.

        Lower positive rate over higher positive rate across the first two groups.

        Equation: ``a, b = (positive_rate(first), positive_rate(second)); result = min(a, b) /
        max(a, b)``."""
        a, b = self.positive_rate(self.first), self.positive_rate(self.second)
        return min(a, b) / max(a, b)

    @property
    def average_odds_difference(self) -> float:
        """Mean of the absolute group TPR and FPR gaps; this is an unsigned disparity summary.

        Mean of the absolute TPR and FPR gaps.

        Equation: ``0.5 * (abs(gap('tpr')) + abs(gap('fpr')))``."""
        return 0.5 * (abs(self.gap("tpr")) + abs(self.gap("fpr")))


class ProportionGapTestScenario(BaseScenario):
    """An observed gap between two rates, ``rate_a`` and ``rate_b``, on ``n_per_group``
    decisions each.

    Tested with the pooled two-proportion z statistic against the two-sided
    normal critical value at ``significance``; the interval is the unpooled
    normal-approximation interval at the same level.

    Notes
    -----
    Equation: Statistics helpers compute pooled z tests, unpooled gap intervals, and the
    inverse equal-size requirement.

    Assumptions: Independent groups of equal size and sufficient counts for a normal
    approximation.

    Excludes: Paired observations, multiplicity corrections, and small-sample exact
    inference.

    References
    ----------
    Derivation: Statistics helpers compute pooled z tests, unpooled gap intervals, and the
    inverse equal-size requirement. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LoanApprovalGapTest
    >>> round(float(scenario.gap), 4)
    0.3"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rate_a: float
    rate_b: float
    n_per_group: int
    significance: float = 0.05
    small_n_per_group: Optional[int] = None  # a second, smaller sample for comparison

    @property
    def gap(self) -> float:
        """Signed group-A minus group-B probability difference.

        Equation: ``rate_a - rate_b``."""
        return self.rate_a - self.rate_b

    @property
    def pooled_rate(self) -> float:
        """Mean of the two group rates under equal group sizes.

        Equation: ``(rate_a + rate_b) / 2``."""
        return (self.rate_a + self.rate_b) / 2

    @property
    def z_critical(self) -> float:
        """Standard-normal critical value for the supplied two-sided significance level.

        Equation: ``NormalDist().inv_cdf(1 - significance / 2)``."""
        from statistics import NormalDist

        return NormalDist().inv_cdf(1 - self.significance / 2)

    def z_at(self, n: float) -> float:
        """Pooled two-proportion z statistic using ``n`` independent decisions per group.

        Equation: ``calc_two_proportion_z(rate_a, rate_b, n, n)``."""
        from ...physics import calc_two_proportion_z

        return calc_two_proportion_z(self.rate_a, self.rate_b, n, n)

    def interval_at(self, n: float):
        """Normal-approximation probability-gap interval at ``n`` independent decisions per group.

        Equation: ``calc_proportion_gap_interval(rate_a, rate_b, n, n, z_critical)``."""
        from ...physics import calc_proportion_gap_interval

        return calc_proportion_gap_interval(self.rate_a, self.rate_b, n, n, self.z_critical)

    @property
    def z(self) -> float:
        """Pooled two-proportion z statistic at the configured equal group size.

        Equation: ``z_at(n_per_group)``."""
        return self.z_at(self.n_per_group)

    @property
    def interval(self):
        """Normal-approximation probability-gap interval at the configured group size.

        Equation: ``interval_at(n_per_group)``."""
        return self.interval_at(self.n_per_group)

    @property
    def n_at_critical(self) -> float:
        """Equal per-group count at which the observed gap reaches the critical z value.

        Per-group sample size at which this gap first reaches the critical value.

        Equation: ``calc_equal_n_for_z(rate_a, rate_b, z_critical)``."""
        from ...physics import calc_equal_n_for_z

        return calc_equal_n_for_z(self.rate_a, self.rate_b, self.z_critical)


class ConstraintAccuracyCostScenario(BaseScenario):
    """Accuracy before and after a fairness constraint (percent), with the policy's stated
    rates.

    ``context`` holds the scenario's descriptive rates in percent (default
    rates, approval rates, thresholds) so prose and slides quote one source.

    Notes
    -----
    Equation: Subtract constrained accuracy from baseline and normalize by baseline for
    relative cost.

    Assumptions: Input accuracies use the same scale; percent inputs produce
    percentage-point differences.

    Excludes: A causal price of fairness or a prediction outside the supplied comparison.

    References
    ----------
    Derivation: Subtract constrained accuracy from baseline and normalize by baseline for
    relative cost. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CreditParityAccuracyCost
    >>> round(float(scenario.accuracy_cost), 4)
    4.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_accuracy: float  # percent
    constrained_accuracy: float  # percent
    context: Dict[str, float] = {}

    @property
    def accuracy_cost(self) -> float:
        """Baseline minus constrained accuracy, in percentage points when input accuracies are
        percentages.

        Points of accuracy given up.

        Equation: ``baseline_accuracy - constrained_accuracy``."""
        return self.baseline_accuracy - self.constrained_accuracy

    @property
    def relative_cost(self) -> float:
        """Fraction of baseline accuracy lost under the supplied fairness constraint.

        Accuracy given up as a fraction of the baseline.

        Equation: ``accuracy_cost / baseline_accuracy``."""
        return self.accuracy_cost / self.baseline_accuracy


class SampledDecisionMonitorScenario(BaseScenario):
    """Per-group metric monitoring of a decision service at ``request_rate``.

    ``sample_share`` of decisions are logged at ``record_size`` each; ``groups``
    subgroups x ``metrics_per_group`` metrics are each tested at
    ``false_alarm_rate`` per evaluation interval on a stable process.

    Notes
    -----
    Equation: Daily rate integration determines records and bytes; summed per-metric tail
    budgets give expected false alerts.

    Assumptions: Constant request rate, fixed sampling share, and fixed record size.

    Excludes: Bursty logging, storage replication, and a proof that all metric tests are
    calibrated.

    References
    ----------
    Derivation: Daily rate integration determines records and bytes; summed per-metric tail
    budgets give expected false alerts. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FairnessMonitor10K
    >>> round(float(scenario.decisions_per_day), 4)
    864000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_rate: Quantity
    sample_share: float
    record_size: Quantity
    groups: int
    metrics_per_group: int
    false_alarm_rate: float

    @property
    def decisions_per_day(self) -> float:
        """Decision count in one day at the constant request rate.

        Equation: ``(request_rate * Q_(1, 'day')).to('').magnitude``."""
        return (self.request_rate * Q_(1, "day")).to("").magnitude

    @property
    def records_per_day(self) -> float:
        """Daily logged record count after applying the supplied sample fraction.

        Equation: ``decisions_per_day * sample_share``."""
        return self.decisions_per_day * self.sample_share

    @property
    def log_per_day(self) -> Quantity:
        """Decimal GB of daily logs at the declared record size.

        Equation: ``(records_per_day * record_size).to('GB')``."""
        return (self.records_per_day * self.record_size).to("GB")

    @property
    def metrics(self) -> int:
        """Total monitored metric count across groups.

        Equation: ``groups * metrics_per_group``."""
        return self.groups * self.metrics_per_group

    @property
    def false_alerts_per_interval(self) -> float:
        """Expected false-alert count per interval without multiplicity correction.

        Expected false alerts per interval if nothing is wrong (no independence needed).

        Equation: ``metrics * false_alarm_rate``."""
        return self.metrics * self.false_alarm_rate

    @property
    def bonferroni_level(self) -> float:
        """Per-metric false-alarm probability after dividing the familywise budget by metric count.

        Equation: ``false_alarm_rate / metrics``."""
        return self.false_alarm_rate / self.metrics

    @property
    def false_alerts_bonferroni(self) -> float:
        """Expected false-alert count summed over metrics at the corrected per-metric level.

        Equation: ``metrics * bonferroni_level``."""
        return self.metrics * self.bonferroni_level


class InlineOverheadScenario(BaseScenario):
    """A safeguard added inline to a request path of ``base_latency``, costing
    ``overhead_low`` to ``overhead_high`` (fractions of the base) per request.

    Under the proportional assumption, holding throughput needs capacity in the
    ratio of the new latency to the old.

    Notes
    -----
    Equation: Multiply baseline latency by each overhead fraction, then add it back to form
    the latency stretch.

    Assumptions: Inline overhead scales baseline latency by the supplied low/high fractions.

    Excludes: Queueing, overlap, batching, and an independent provisioning calculation.

    References
    ----------
    Derivation: Multiply baseline latency by each overhead fraction, then add it back to
    form the latency stretch. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.InlineShapRetrofit
    >>> round(float(scenario.added_low.magnitude), 4)
    50.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base_latency: Quantity
    overhead_low: float
    overhead_high: float
    request_rate: Optional[Quantity] = None

    @property
    def added_low(self) -> Quantity:
        """Milliseconds of added inline work at the lower assumed overhead fraction.

        Equation: ``(base_latency * overhead_low).to('ms')``."""
        return (self.base_latency * self.overhead_low).to("ms")

    @property
    def added_high(self) -> Quantity:
        """Milliseconds of added inline work at the upper assumed overhead fraction.

        Equation: ``(base_latency * overhead_high).to('ms')``."""
        return (self.base_latency * self.overhead_high).to("ms")

    @property
    def latency_low(self) -> Quantity:
        """Total milliseconds after adding the lower inline overhead.

        Equation: ``(base_latency + added_low).to('ms')``."""
        return (self.base_latency + self.added_low).to("ms")

    @property
    def latency_high(self) -> Quantity:
        """Total milliseconds after adding the upper inline overhead.

        Equation: ``(base_latency + added_high).to('ms')``."""
        return (self.base_latency + self.added_high).to("ms")

    @property
    def capacity_low(self) -> float:
        """Dimensionless lower latency stretch, interpreted as capacity demand only under unchanged
        throughput assumptions.

        Equation: ``(latency_low / base_latency).to('').magnitude``."""
        return (self.latency_low / self.base_latency).to("").magnitude

    @property
    def capacity_high(self) -> float:
        """Dimensionless upper latency stretch, interpreted as capacity demand only under unchanged
        throughput assumptions.

        Equation: ``(latency_high / base_latency).to('').magnitude``."""
        return (self.latency_high / self.base_latency).to("").magnitude


class ExplanationLogScenario(BaseScenario):
    """Logging an explanation with every prediction: ``predictions_per_day`` for
    ``days_per_month``,
    each a ``map_size_low`` to ``map_size_high`` attribution map (or tabular attributions of
    ``attribution_bytes_low`` to ``attribution_bytes_high`` per feature).

    Notes
    -----
    Equation: Prediction count times days gives monthly count; multiplying by map bytes
    gives stored volume.

    Assumptions: Fixed daily prediction count, month length, and explanation sizes.

    Excludes: Compression, replication, retention beyond one month, and
    explanation-generation compute.

    References
    ----------
    Derivation: Prediction count times days gives monthly count; multiplying by map bytes
    gives stored volume. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ImageExplanationLog
    >>> round(float(scenario.predictions_per_month), 4)
    30000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    predictions_per_day: float
    days_per_month: float
    map_size_low: Quantity
    map_size_high: Quantity
    attribution_bytes_low: Quantity
    attribution_bytes_high: Quantity

    @property
    def predictions_per_month(self) -> float:
        """Prediction count during the supplied number of billing days.

        Equation: ``predictions_per_day * days_per_month``."""
        return self.predictions_per_day * self.days_per_month

    @property
    def monthly_low(self) -> Quantity:
        """Decimal TB of monthly explanations at the lower per-explanation size.

        Equation: ``(predictions_per_month * map_size_low).to('TB')``."""
        return (self.predictions_per_month * self.map_size_low).to("TB")

    @property
    def monthly_high(self) -> Quantity:
        """Decimal TB of monthly explanations at the upper per-explanation size.

        Equation: ``(predictions_per_month * map_size_high).to('TB')``."""
        return (self.predictions_per_month * self.map_size_high).to("TB")


class ShardedUnlearningScenario(BaseScenario):
    """Exact unlearning by sharded retraining (SISA) against a full retrain of ``workload``.

    The full retrain runs on ``n_accelerators`` of ``accelerator`` at ``mfu`` and
    costs ``full_cost``. Planning rounds it to whole days; one request retrains
    one of ``shards`` equal shards, at a proportional share of time and cost.
    ``requests_per_day`` arrive uniformly at random over the shards.

    Notes
    -----
    Equation: Uniform occupancy gives the expected touched-shard count; multiply by
    one-shard retraining cost.

    Assumptions: Equal-sized shards, linear retraining cost, uniform independent
    deletion-to-shard assignment, and fixed MFU.

    Excludes: SISA slice mechanics, imbalance, optimizer startup, and exact unlearning
    verification.

    References
    ----------
    Derivation: Uniform occupancy gives the expected touched-shard count; multiply by
    one-shard retraining cost. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Gpt3ShardedUnlearning
    >>> round(float(scenario.full_retrain_time.magnitude), 4)
    35.5477"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    accelerator: HardwareNode
    n_accelerators: int
    mfu: float
    full_cost: Quantity
    shards: int
    requests_per_day: int
    accuracy_drop_low: Optional[float] = None  # fraction
    accuracy_drop_high: Optional[float] = None  # fraction

    @property
    def full_retrain_time(self) -> Quantity:
        """Days to perform the stated training FLOPs at aggregate peak FLOP/s times the declared
        MFU.

        Equation: ``rate = n_accelerators * accelerator.compute.peak_flops * mfu; result =
        (workload.training_ops / rate).to('day')``."""
        rate = self.n_accelerators * self.accelerator.compute.peak_flops * self.mfu
        return (self.workload.training_ops / rate).to("day")

    @property
    def planning_days(self) -> int:
        """Full-retrain time rounded to integer days for the scenario's planning convention.

        The full retrain rounded to whole days, the figure planning quotes.

        Equation: ``round(full_retrain_time.to('day').magnitude)``."""
        return round(self.full_retrain_time.to("day").magnitude)

    @property
    def shard_share(self) -> float:
        """Fraction of equal-sized training work assigned to one shard.

        Equation: ``1 / shards``."""
        return 1 / self.shards

    @property
    def shard_cost(self) -> Quantity:
        """USD for one shard retrain under linear equal-shard cost scaling.

        Equation: ``(full_cost * shard_share).to('USD')``."""
        return (self.full_cost * self.shard_share).to("USD")

    @property
    def shard_time(self) -> Quantity:
        """Hours per shard retrain using rounded planning days rather than exact full-retrain time.

        Equation: ``(Q_(planning_days, 'day') * shard_share).to('hour')``."""
        return (Q_(self.planning_days, "day") * self.shard_share).to("hour")

    @property
    def p_shard_untouched(self) -> float:
        """Probability one shard receives no deletion requests when requests independently choose
        shards uniformly.

        Chance a given shard receives none of a day's requests.

        Equation: ``(1 - shard_share) ** requests_per_day``."""
        return (1 - self.shard_share) ** self.requests_per_day

    @property
    def shards_touched_per_day(self) -> float:
        """Expected daily count of distinct shards hit by deletion requests.

        Equation: ``shards * (1 - p_shard_untouched)``."""
        return self.shards * (1 - self.p_shard_untouched)

    @property
    def daily_batch_cost(self) -> Quantity:
        """Expected USD to retrain each touched shard once in a daily deletion batch.

        Cost if every touched shard retrains once a day.

        Equation: ``(shards_touched_per_day * shard_cost).to('USD')``."""
        return (self.shards_touched_per_day * self.shard_cost).to("USD")


class PreferenceLabelBudgetScenario(BaseScenario):
    """Human preference labels for a reward model: ``comparisons_low`` to ``comparisons_high``
    at ``price_low`` to ``price_high`` per comparison.

    Notes
    -----
    Equation: Multiply comparisons by price for each endpoint and compare the resulting
    budgets.

    Assumptions: The stated comparison-count and price endpoints define two planning cases.

    Excludes: Annotator-quality variation, overhead, and a learning-performance guarantee.

    References
    ----------
    Derivation: Multiply comparisons by price for each endpoint and compare the resulting
    budgets. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RewardModelLabelBudget
    >>> round(float(scenario.budget_low.magnitude), 4)
    25000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    comparisons_low: float
    comparisons_high: float
    price_low: Quantity
    price_high: Quantity

    @property
    def budget_low(self) -> Quantity:
        """USD for the lower comparison-count and per-comparison price assumptions.

        Equation: ``(comparisons_low * price_low).to('USD')``."""
        return (self.comparisons_low * self.price_low).to("USD")

    @property
    def budget_high(self) -> Quantity:
        """USD for the upper comparison-count and per-comparison price assumptions.

        Equation: ``(comparisons_high * price_high).to('USD')``."""
        return (self.comparisons_high * self.price_high).to("USD")

    @property
    def spread(self) -> float:
        """Dimensionless upper-to-lower labeling-budget ratio.

        Equation: ``(budget_high / budget_low).to('').magnitude``."""
        return (self.budget_high / self.budget_low).to("").magnitude


class AutomationBiasScenario(BaseScenario):
    """A human reviews an AI's output; the human accepts ``miss_acceptance`` of the AI's
    misses.

    The best case assumes the reviewer catches every other miss and never
    overturns a correct detection, so team sensitivity is at most one minus the
    accepted misses.

    Notes
    -----
    Equation: Remaining misses equal AI miss rate times human acceptance of those misses;
    sensitivity is the complement.

    Assumptions: Every rejected AI miss is corrected and the reviewer introduces no
    additional misses.

    Excludes: Correlated human errors, false positives, and an empirically validated team
    model.

    References
    ----------
    Derivation: Remaining misses equal AI miss rate times human acceptance of those misses;
    sensitivity is the complement. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RadiologyAutomationBias
    >>> round(float(scenario.ai_miss_rate), 4)
    0.05"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    human_sensitivity: float
    ai_sensitivity: float
    miss_acceptance: float
    acceptance_low: float
    acceptance_high: float

    @property
    def ai_miss_rate(self) -> float:
        """Probability an actual positive is missed by the AI, complementary to its sensitivity.

        Equation: ``1 - ai_sensitivity``."""
        return 1 - self.ai_sensitivity

    def accepted_misses(self, acceptance: Optional[float] = None) -> float:
        """Fraction of actual positives remaining missed when the human accepts the specified
        fraction of AI misses.

        Equation: ``a = miss_acceptance if acceptance is None else acceptance; result =
        ai_miss_rate * a``."""
        a = self.miss_acceptance if acceptance is None else acceptance
        return self.ai_miss_rate * a

    @property
    def team_sensitivity_bound(self) -> float:
        """Optimistic team sensitivity when every unaccepted AI miss is corrected and no new misses
        are introduced.

        Equation: ``1 - accepted_misses()``."""
        return 1 - self.accepted_misses()

    @property
    def gain_over_ai(self) -> float:
        """Absolute sensitivity gain of that optimistic team bound over the AI alone.

        Equation: ``team_sensitivity_bound - ai_sensitivity``."""
        return self.team_sensitivity_bound - self.ai_sensitivity


class BinomialIntervalsScenario(BaseScenario):
    """Several success counts out of one number of independent tasks.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.BinomialIntervalsModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.BinomialIntervalsModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.BinomialIntervalsModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FiftyTaskIntervals
    >>> round(float(scenario.result.z_critical), 4)
    1.96"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    trials: int
    successes: List[int]
    confidence: float = 0.95

    @property
    def result(self):
        """Return the modeled BinomialIntervals result from
        ``mlsysim.solvers.BinomialIntervalsModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import BinomialIntervalsModel

        return BinomialIntervalsModel().solve(self.trials, self.successes, self.confidence)


class ProportionPlanningScenario(BaseScenario):
    """Planning targets for one acceptance rate and for two independent arms.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.ProportionPlanningModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ProportionPlanningModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ProportionPlanningModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AcceptanceRatePlanning
    >>> round(float(scenario.result.z_critical), 4)
    1.96"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rate: float  # planning acceptance probability (0.5 is the worst case)
    planning_tasks: int  # task count whose half-width is quoted
    target_half_width: float
    differences: List[float]  # detectable differences between two arms
    significance: float  # two-sided alpha the detection quantile stands for
    power: float  # power the second quantile stands for
    z_alpha: float  # rounded quantile used in the planning formula
    z_beta: float
    confidence: float = 0.95

    @property
    def result(self):
        """Return the modeled ProportionPlanning result from
        ``mlsysim.solvers.ProportionPlanningModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import ProportionPlanningModel

        return ProportionPlanningModel().solve(
            self.rate,
            self.planning_tasks,
            self.target_half_width,
            self.differences,
            self.z_alpha,
            self.z_beta,
            self.confidence,
        )


class PairedComparisonScenario(BaseScenario):
    """One candidate and one baseline trial on each sampled task, outcomes kept in pairs.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.PairedComparisonModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.PairedComparisonModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.PairedComparisonModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairCandidatePairedComparison
    >>> round(float(scenario.result.pairs), 4)
    150.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    both: int  # tasks both configurations accepted
    candidate_only: int
    baseline_only: int
    neither: int
    tolerance: float  # noninferiority tolerance, as a fraction
    confidence: float = 0.95

    @property
    def result(self):
        """Return the modeled PairedComparison result from
        ``mlsysim.solvers.PairedComparisonModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import PairedComparisonModel

        return PairedComparisonModel().solve(
            self.both, self.candidate_only, self.baseline_only, self.neither, self.tolerance, self.confidence
        )


class TaskMixConsistencyScenario(BaseScenario):
    """Equally weighted task types and a number of runs that must all pass.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.TaskMixConsistencyModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.TaskMixConsistencyModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.TaskMixConsistencyModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TwoTaskTypeConsistency
    >>> round(float(scenario.result.mean_rate), 4)
    0.7"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    task_pass_rates: List[float]
    repetitions: int

    @property
    def result(self):
        """Return the modeled TaskMixConsistency result from
        ``mlsysim.solvers.TaskMixConsistencyModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import TaskMixConsistencyModel

        return TaskMixConsistencyModel().solve(self.task_pass_rates, self.repetitions)


class CappedSelectionCostScenario(BaseScenario):
    """Candidates for one fixed task, an exact checker, and a cap on candidates.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.CappedSelectionCostModel.solve`` with the stored fields and any
    derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.CappedSelectionCostModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.CappedSelectionCostModel.solve`` with the stored fields and any
    derived properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CappedCandidateSelection
    >>> round(float(scenario.result.availability), 4)
    0.784"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pass_rate: float
    cap: int
    generation_cost: Quantity  # USD per candidate
    check_cost: Quantity  # USD per candidate

    @property
    def result(self):
        """Return the modeled CappedSelectionCost result from
        ``mlsysim.solvers.CappedSelectionCostModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import CappedSelectionCostModel

        return CappedSelectionCostModel().solve(self.pass_rate, self.cap, self.generation_cost, self.check_cost)


class EfficiencyClaimGapScenario(BaseScenario):
    """A datasheet operating point against a sustained one: operations per watt at each, and their ratio.

    The gap is the product of the throughput fall and the power rise, so a
    reader who compares only the throughputs underestimates it.

    Notes
    -----
    Equation: The calculation members below define ``claimed_efficiency``, ``sustained_efficiency``, ``gap``, ``rate_fall``, ``power_rise``.

    Assumptions: Each supplied rate is paired with the power at that operating point.

    Excludes: A prediction of sustained efficiency from datasheet peaks.

    References
    ----------
    Derivation: A datasheet operating point against a sustained one: operations per watt at each, and their ratio. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DatasheetVersusSustainedEdgeChip
    >>> value = case.gap
    >>> print(f"{value:.3g}")
    13.3"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    claimed_rate: Quantity
    claimed_power: Quantity
    sustained_rate: Quantity
    sustained_power: Quantity

    @property
    def claimed_efficiency(self) -> Quantity:
        """Claimed operation rate per watt, expressed as TOPS/W.

        Equation: ``(claimed_rate / claimed_power).to('TOPS/W')``."""
        return (self.claimed_rate / self.claimed_power).to("TOPS/W")

    @property
    def sustained_efficiency(self) -> Quantity:
        """Sustained operation rate per watt, expressed as TOPS/W.

        Equation: ``(sustained_rate / sustained_power).to('TOPS/W')``."""
        return (self.sustained_rate / self.sustained_power).to("TOPS/W")

    @property
    def gap(self) -> float:
        """Dimensionless claimed-to-sustained efficiency ratio.

        Equation: ``(claimed_efficiency / sustained_efficiency).to('').magnitude``."""
        return (self.claimed_efficiency / self.sustained_efficiency).to("").magnitude

    @property
    def rate_fall(self) -> float:
        """Claimed rate divided by sustained rate, a dimensionless throughput gap.

        Equation: ``(claimed_rate / sustained_rate).to('').magnitude``."""
        return (self.claimed_rate / self.sustained_rate).to("").magnitude

    @property
    def power_rise(self) -> float:
        """Sustained power divided by claimed power, a dimensionless power multiplier.

        Equation: ``(sustained_power / claimed_power).to('').magnitude``."""
        return (self.sustained_power / self.claimed_power).to("").magnitude


class HostAcceleratorBoardScenario(BaseScenario):
    """One board, a host CPU and an attached accelerator: model-only and end-to-end latency, active energy.

    The end-to-end times add host preprocessing (and, on the accelerator path,
    the copy across the host link) to the model time; active energy is the
    component's active power times its model-only latency, so it leaves out
    preprocessing and idle power.

    Notes
    -----
    Equation: The calculation members below define ``frame_budget``, ``int8_speedup``, ``isolated_speedup``, ``end_to_end_speedup``, ``power_ratio``, ``host_energy``, ``accelerator_energy``, ``energy_ratio``.

    Assumptions: Active energy is power times model-only latency; end-to-end times are supplied separately.

    Excludes: Idle energy, host/accelerator overlap, and energy of the surrounding response path.

    References
    ----------
    Derivation: One board, a host CPU and an attached accelerator: model-only and end-to-end latency, active energy. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MobileNetV2CameraTrapBoard
    >>> value = case.end_to_end_speedup
    >>> print(f"{value:.3g}")
    3"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    host_fp32_latency: Quantity
    host_int8_latency: Quantity
    accelerator_latency: Quantity
    host_end_to_end: Quantity
    accelerator_end_to_end: Quantity
    host_power: Quantity
    accelerator_power: Quantity
    frame_rate: Optional[Quantity] = None  # frames per second the camera delivers

    @property
    def frame_budget(self) -> Quantity:
        """Per-frame latency allowance in ms, reciprocal of frame rate.

        Time per frame at ``frame_rate``: the latency each inference must meet.

        Equation: ``(1 / frame_rate).to('ms')``."""
        return (1 / self.frame_rate).to("ms")

    @property
    def int8_speedup(self) -> float:
        """Dimensionless host FP32-to-INT8 latency ratio.

        Equation: ``(host_fp32_latency / host_int8_latency).to('').magnitude``."""
        return (self.host_fp32_latency / self.host_int8_latency).to("").magnitude

    @property
    def isolated_speedup(self) -> float:
        """Dimensionless host INT8-to-accelerator model-only latency ratio.

        Equation: ``(host_int8_latency / accelerator_latency).to('').magnitude``."""
        return (self.host_int8_latency / self.accelerator_latency).to("").magnitude

    @property
    def end_to_end_speedup(self) -> float:
        """Dimensionless host-to-accelerator full-path latency ratio.

        Equation: ``(host_end_to_end / accelerator_end_to_end).to('').magnitude``."""
        return (self.host_end_to_end / self.accelerator_end_to_end).to("").magnitude

    @property
    def power_ratio(self) -> float:
        """Accelerator active power divided by host active power, dimensionless.

        Equation: ``(accelerator_power / host_power).to('').magnitude``."""
        return (self.accelerator_power / self.host_power).to("").magnitude

    @property
    def host_energy(self) -> Quantity:
        """Host active energy per inference in mJ, power times INT8 model latency.

        Equation: ``(host_power * host_int8_latency).to('mJ')``."""
        return (self.host_power * self.host_int8_latency).to("mJ")

    @property
    def accelerator_energy(self) -> Quantity:
        """Accelerator active energy per inference in mJ, power times model latency.

        Equation: ``(accelerator_power * accelerator_latency).to('mJ')``."""
        return (self.accelerator_power * self.accelerator_latency).to("mJ")

    @property
    def energy_ratio(self) -> float:
        """Host-to-accelerator modeled active energy ratio, dimensionless.

        Equation: ``(host_energy / accelerator_energy).to('').magnitude``."""
        return (self.host_energy / self.accelerator_energy).to("").magnitude


class MeanVersusTailScenario(BaseScenario):
    """A benchmark's mean latency against a production tail stated as a range.

    Notes
    -----
    Equation: The calculation members below define ``gap_low``, ``gap_high``.

    Assumptions: Mean and tail endpoints are supplied latency quantities with positive mean.

    Excludes: Inferring a tail distribution or equating benchmark and production workloads.

    References
    ----------
    Derivation: A benchmark's mean latency against a production tail stated as a range. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ModelOnlyMeanVersusProductionP99
    >>> value = case.gap_low
    >>> print(f"{value:.3g}")
    10"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    mean: Quantity
    tail_low: Quantity
    tail_high: Quantity

    @property
    def gap_low(self) -> float:
        """Lower production-tail latency divided by benchmark mean, dimensionless.

        Equation: ``(tail_low / mean).to('').magnitude``."""
        return (self.tail_low / self.mean).to("").magnitude

    @property
    def gap_high(self) -> float:
        """Upper production-tail latency divided by benchmark mean, dimensionless.

        Equation: ``(tail_high / mean).to('').magnitude``."""
        return (self.tail_high / self.mean).to("").magnitude


class StrongScalingPairScenario(BaseScenario):
    """The same fixed job on ``base_devices`` and on ``scaled_devices``: speedup, ideal time, and efficiency.

    Strong-scaling efficiency is base time over (device ratio times scaled time).

    Notes
    -----
    Equation: The calculation members below define ``device_ratio``, ``speedup``, ``ideal_time``, ``efficiency``.

    Assumptions: Both durations describe the same fixed job; ideal scaling divides time by device-count growth.

    Excludes: Weak scaling, changed accuracy targets, and attribution of overhead to specific mechanisms.

    References
    ----------
    Derivation: The same fixed job on ``base_devices`` and on ``scaled_devices``: speedup, ideal time, and efficiency. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ImageClassifierEightGpuScaling
    >>> value = case.efficiency
    >>> print(f"{value:.3g}")
    0.75"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base_devices: int
    base_time: Quantity
    scaled_devices: int
    scaled_time: Quantity

    @property
    def device_ratio(self) -> float:
        """Scaled device count divided by baseline count, dimensionless.

        Equation: ``scaled_devices / base_devices``."""
        return self.scaled_devices / self.base_devices

    @property
    def speedup(self) -> float:
        """Baseline elapsed time divided by scaled elapsed time for the fixed job.

        Equation: ``(base_time / scaled_time).to('').magnitude``."""
        return (self.base_time / self.scaled_time).to("").magnitude

    @property
    def ideal_time(self) -> Quantity:
        """Ideal scaled duration in baseline-time units, assuming linear speedup.

        Equation: ``(base_time / device_ratio).to(base_time.units)``."""
        return (self.base_time / self.device_ratio).to(self.base_time.units)

    @property
    def efficiency(self) -> float:
        """Observed speedup divided by device-count growth, dimensionless.

        Equation: ``speedup / device_ratio``."""
        return self.speedup / self.device_ratio


def _z_two_sided(confidence: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + confidence / 2)


class ProportionSampleSizeScenario(BaseScenario):
    """Samples that pin a pass fraction ``rate`` to within ``half_width`` at ``confidence``: ``z^2 p (1 - p) / m^2``.

    ``published_minimum`` is a suite's rounded-up count for the same target.

    Notes
    -----
    Equation: The calculation members below define ``z``, ``samples``, ``samples_needed``.

    Assumptions: Independent Bernoulli observations support the normal-approximation precision calculation.

    Excludes: Exact coverage for rare events and uncertainty in the planning rate.

    References
    ----------
    Derivation: Samples that pin a pass fraction ``rate`` to within ``half_width`` at ``confidence``: ``z^2 p (1 - p) / m^2``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AccuracyWithinOnePointSampleSize
    >>> value = case.samples_needed
    >>> print(value)
    1825"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rate: float
    half_width: float
    confidence: float
    published_minimum: Optional[int] = None

    @property
    def z(self) -> float:
        """Two-sided standard-normal critical value for the requested confidence.

        Equation: ``_z_two_sided(confidence)``."""
        return _z_two_sided(self.confidence)

    @property
    def samples(self) -> float:
        """Unrounded normal-approximation sample requirement, z squared times Bernoulli variance over squared margin.

        Unrounded sample count.

        Equation: ``z ** 2 * rate * (1 - rate) / half_width ** 2``."""
        return self.z**2 * self.rate * (1 - self.rate) / self.half_width**2

    @property
    def samples_needed(self) -> int:
        """Integer sample requirement from ``calc_samples_for_half_width``, rounded upward.

        Calculation: ``mlsysim.physics.statistics.calc_samples_for_half_width``."""
        from ...physics.statistics import calc_samples_for_half_width

        return calc_samples_for_half_width(self.rate, self.half_width, self.z)


class SameTestSetAccuracyComparisonScenario(BaseScenario):
    """Two models scored on the same ``n_samples`` test items: independent-set interval and McNemar's limit.

    The scores fix the net gap ``b - c`` (items only the baseline gets right
    minus items only the candidate gets right) but not the disagreements
    ``b + c``; McNemar's statistic (no continuity correction) exceeds the
    chi-square critical value ``z^2`` only when ``b + c <= floor(gap^2 / z^2)``.

    Notes
    -----
    Equation: The calculation members below define ``z``, ``baseline_errors``, ``candidate_errors``, ``baseline_error_sd``, ``accuracy_difference``, ``independent_interval``, ``net_gap``, ``chi2_critical``, ``max_discordant``, ``max_discordant_share``.

    Assumptions: Scores use the same item count; the displayed independent interval is a counterfactual comparison, while the McNemar threshold constrains discordant pairs.

    Excludes: Recovering actual discordant counts from aggregate accuracies alone.

    References
    ----------
    Derivation: Two models scored on the same ``n_samples`` test items: independent-set interval and McNemar's limit. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.NinetyFiveAgainstNinetyFourOnSameImages
    >>> value = case.max_discordant
    >>> print(value)
    26"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_samples: int
    baseline_accuracy: float
    candidate_accuracy: float
    confidence: float

    @property
    def z(self) -> float:
        """Two-sided standard-normal critical value for the requested confidence.

        Equation: ``_z_two_sided(confidence)``."""
        return _z_two_sided(self.confidence)

    @property
    def baseline_errors(self) -> float:
        """Expected baseline error count from sample count times one minus accuracy.

        Equation: ``n_samples * (1 - baseline_accuracy)``."""
        return self.n_samples * (1 - self.baseline_accuracy)

    @property
    def candidate_errors(self) -> float:
        """Expected candidate error count from sample count times one minus accuracy.

        Equation: ``n_samples * (1 - candidate_accuracy)``."""
        return self.n_samples * (1 - self.candidate_accuracy)

    @property
    def baseline_error_sd(self) -> float:
        """Binomial standard deviation of baseline error count, assuming independent items.

        Binomial standard deviation of the baseline's error count.

        Equation: ``_math.sqrt(n_samples * p * (1 - p))``."""
        p = self.baseline_accuracy
        return _math.sqrt(self.n_samples * p * (1 - p))

    @property
    def accuracy_difference(self) -> float:
        """Baseline-minus-candidate accuracy difference as a fraction, not percentage points.

        Equation: ``baseline_accuracy - candidate_accuracy``."""
        return self.baseline_accuracy - self.candidate_accuracy

    @property
    def independent_interval(self):
        """Hypothetical independent-test-set interval via ``calc_proportion_gap_interval``; not the paired-data interval.

        (low, high) of the accuracy difference had the scores come from independent test sets.

        Calculation: ``mlsysim.physics.statistics.calc_proportion_gap_interval``."""
        from ...physics.statistics import calc_proportion_gap_interval

        return calc_proportion_gap_interval(
            self.baseline_accuracy, self.candidate_accuracy, self.n_samples, self.n_samples, self.z
        )

    @property
    def net_gap(self) -> int:
        """Rounded net number of items correct only for baseline minus only for candidate.

        b - c: items the baseline alone gets right, net.

        Equation: ``round(n_samples * accuracy_difference)``."""
        return round(self.n_samples * self.accuracy_difference)

    @property
    def chi2_critical(self) -> float:
        """One-degree-of-freedom chi-square threshold formed by squaring the normal critical value.

        Equation: ``z ** 2``."""
        return self.z**2

    @property
    def max_discordant(self) -> int:
        """Largest discordant count consistent with the threshold via ``calc_mcnemar_max_discordant``.

        Calculation: ``mlsysim.physics.statistics.calc_mcnemar_max_discordant``."""
        from ...physics.statistics import calc_mcnemar_max_discordant

        return calc_mcnemar_max_discordant(self.net_gap, self.chi2_critical)

    @property
    def max_discordant_share(self) -> float:
        """Maximum discordant count divided by test-set count, dimensionless.

        Equation: ``max_discordant / n_samples``."""
        return self.max_discordant / self.n_samples


class ProxyMetricLatencyScenario(BaseScenario):
    """A search-width change that buys an offline metric gain with latency, against a serving budget.

    Notes
    -----
    Equation: The calculation members below define ``metric_gain``, ``width_ratio``, ``slowdown``, ``budget_ratio``.

    Assumptions: Metric values, widths, and latencies are supplied for the same comparison.

    Excludes: Treating offline score improvement as improved online task outcomes.

    References
    ----------
    Derivation: A search-width change that buys an offline metric gain with latency, against a serving budget. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WiderBeamSearchTranslation
    >>> value = case.budget_ratio
    >>> print(f"{value:.3g}")
    2"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    original_metric: float
    optimized_metric: float
    original_width: int
    optimized_width: int
    original_latency: Quantity
    optimized_latency: Quantity
    latency_budget: Quantity

    @property
    def metric_gain(self) -> float:
        """Optimized-minus-original offline score in the supplied score units.

        Equation: ``optimized_metric - original_metric``."""
        return self.optimized_metric - self.original_metric

    @property
    def width_ratio(self) -> float:
        """Optimized search width divided by original width, dimensionless.

        Equation: ``optimized_width / original_width``."""
        return self.optimized_width / self.original_width

    @property
    def slowdown(self) -> float:
        """Optimized latency divided by original latency, dimensionless.

        Equation: ``(optimized_latency / original_latency).to('').magnitude``."""
        return (self.optimized_latency / self.original_latency).to("").magnitude

    @property
    def budget_ratio(self) -> float:
        """Optimized latency divided by serving budget; above one exceeds the budget.

        Optimized latency over the budget (above 1: the budget is broken).

        Equation: ``(optimized_latency / latency_budget).to('').magnitude``."""
        return (self.optimized_latency / self.latency_budget).to("").magnitude


class QuantizedAcceptanceGatesScenario(BaseScenario):
    """A float model and its quantized copy against acceptance gates set before compression.

    Accuracies are percent; ECE is a fraction. Edge cases are ``edge_share``
    of inputs, so a ``test_images`` test set holds ``edge_share * test_images``
    of them, and the float edge-case accuracy is known only to the binomial
    half-width at ``confidence`` on that count.

    Notes
    -----
    Equation: The calculation members below define ``top1_loss``, ``top5_loss``, ``edge_drop``, ``ece_ratio``, ``edge_images``, ``edge_half_width``.

    Assumptions: Accuracy inputs use percentage points; subgroup precision uses an independent-binomial normal approximation.

    Excludes: Multiple-testing correction, paired disagreement, and certification from point estimates.

    References
    ----------
    Derivation: A float model and its quantized copy against acceptance gates set before compression. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MobileNetV2Int8CameraTrapGates
    >>> value = case.edge_half_width
    >>> print(f"{value:.3g}")
    12.9"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    float_top1: float
    int_top1: float
    float_top5: float
    int_top5: float
    float_ece: float
    int_ece: float
    float_edge: float
    int_edge: float
    accuracy_loss_limit: float  # percentage points
    edge_share: float
    edge_occlusion: float  # fraction of the object hidden
    edge_illuminance: Quantity
    edge_rotation_deg: float
    confidence_threshold: float  # the deployment acts above this confidence
    test_images: int
    confidence: float

    @property
    def top1_loss(self) -> float:
        """Float-minus-quantized top-1 score in the supplied percentage-point convention.

        Equation: ``float_top1 - int_top1``."""
        return self.float_top1 - self.int_top1

    @property
    def top5_loss(self) -> float:
        """Float-minus-quantized top-5 score in percentage points.

        Equation: ``float_top5 - int_top5``."""
        return self.float_top5 - self.int_top5

    @property
    def edge_drop(self) -> float:
        """Float-minus-quantized edge-subgroup score in percentage points.

        Equation: ``float_edge - int_edge``."""
        return self.float_edge - self.int_edge

    @property
    def ece_ratio(self) -> float:
        """Quantized calibration error divided by float calibration error, dimensionless.

        Equation: ``int_ece / float_ece``."""
        return self.int_ece / self.float_ece

    @property
    def edge_images(self) -> int:
        """Rounded count of edge-subgroup test images from subgroup share times test count.

        Equation: ``round(edge_share * test_images)``."""
        return round(self.edge_share * self.test_images)

    @property
    def edge_half_width(self) -> float:
        """Normal-approximation float edge-accuracy half-width in percentage points, via ``calc_proportion_half_width``.

        Binomial half-width of the float edge-case accuracy on ``edge_images``, in points.

        Calculation: ``mlsysim.physics.statistics.calc_proportion_half_width``."""
        from ...physics.statistics import calc_proportion_half_width

        return 100 * calc_proportion_half_width(self.float_edge / 100, self.edge_images, _z_two_sided(self.confidence))


class ArgumentConfidenceScenario(BaseScenario):
    """A stipulated safety argument: ``premises`` required premises at ``premise_confidence`` each,
    and a claim carried by ``legs`` independent legs at ``leg_confidence`` each.

    Required premises combine as a product (``calc_conjunctive_confidence``);
    independent sufficient legs fail together with the product of their
    failure probabilities (``calc_multilegged_confidence``).

    Notes
    -----
    Equation: The calculation members below define ``conjunction``, ``multilegged``.

    Assumptions: Premise confidence and sufficient-leg confidence are stipulated probabilities with the specified independence structure.

    Excludes: Empirical validation of those probabilities and a formal safety guarantee.

    References
    ----------
    Derivation: A stipulated safety argument: ``premises`` required premises at ``premise_confidence`` each, and a claim carried by ``legs`` independent legs at ``leg_confidence`` each. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SafetyArgumentConfidence
    >>> value = case.conjunction
    >>> print(f"{value:.3g}")
    0.904"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    premise_confidence: float
    premises: int
    leg_confidence: float
    legs: int = 2

    @property
    def conjunction(self) -> float:
        """Dimensionless stipulated joint confidence via ``calc_conjunctive_confidence`` under independence.

        Calculation: ``mlsysim.physics.calc_conjunctive_confidence``."""
        from ...physics import calc_conjunctive_confidence

        return calc_conjunctive_confidence([self.premise_confidence] * self.premises)

    @property
    def multilegged(self) -> float:
        """Dimensionless stipulated confidence from sufficient independent legs via ``calc_multilegged_confidence``.

        Calculation: ``mlsysim.physics.calc_multilegged_confidence``."""
        from ...physics import calc_multilegged_confidence

        return calc_multilegged_confidence([self.leg_confidence] * self.legs)


class SubgroupEvaluationBudgetScenario(BaseScenario):
    """The labeled data a per-group evaluation needs: one group's rate within ``half_width`` at
    ``confidence`` (worst case ``rate`` 0.5), the collection random sampling needs to reach a
    ``target_per_group`` from a group of ``group_share``, and the per-group count that pins a gap
    between two equal-size groups to ``gap_half_width``.

    Notes
    -----
    Equation: Normal-error half-widths determine sample counts; random collection divides
    required subgroup count by its population share.

    Assumptions: Samples are independent Bernoulli observations; equal-size groups and the
    declared rate underlie gap planning.

    Excludes: A guaranteed subgroup yield from random sampling, exact small-sample coverage,
    and cluster-sampling design effects.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.fairness.calc_gap_samples_per_group``,
    ``mlsysim.physics.statistics.calc_proportion_gap_se``,
    ``mlsysim.physics.statistics.calc_samples_for_half_width``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.OnePercentGroupEvaluationBudget
    >>> value = scenario.z
    >>> round(float(value), 3)
    1.96
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rate: float = 0.5
    half_width: float
    confidence: float
    group_share: float
    target_per_group: int
    gap_half_width: Optional[float] = None

    @property
    def z(self) -> float:
        """Two-sided standard-normal quantile for the requested confidence, dimensionless.

        Equation: ``NormalDist().inv_cdf(0.5 + confidence / 2)``.
        """
        from statistics import NormalDist

        return NormalDist().inv_cdf(0.5 + self.confidence / 2)

    @property
    def variance(self) -> float:
        """Bernoulli outcome variance ``rate * (1 - rate)``, dimensionless.

        Equation: ``rate * (1 - rate)``.
        """
        return self.rate * (1 - self.rate)

    @property
    def samples_needed(self) -> int:
        """Normal-approximation count needed for the declared one-group proportion half-width.

        Equation: ``calc_samples_for_half_width(rate, half_width, z)``.
        """
        from ...physics.statistics import calc_samples_for_half_width

        return calc_samples_for_half_width(self.rate, self.half_width, self.z)

    @property
    def random_collection(self) -> float:
        """Expected total random samples needed to collect the target subgroup count; not a
        high-probability bound.

        Equation: ``target_per_group / group_share``.
        """
        return self.target_per_group / self.group_share

    @property
    def random_multiple(self) -> float:
        """Expected random collection count per targeted subgroup example, reciprocal of subgroup
        share.

        Equation: ``random_collection / target_per_group``.
        """
        return self.random_collection / self.target_per_group

    def gap_samples_per_group(self, half_width: Optional[float] = None) -> int:
        """Per-group sample count for a two-group gap half-width using the shared planning helper.

        Equation: ``calc_gap_samples_per_group(rate, half_width or gap_half_width, z)``.
        """
        from ...physics.fairness import calc_gap_samples_per_group

        return calc_gap_samples_per_group(self.rate, half_width or self.gap_half_width, self.z)

    def gap_half_width_at(self, n_per_group: int) -> float:
        """Normal-approximation gap half-width at ``n_per_group`` independent observations in each
        group.

        Equation: ``z * calc_proportion_gap_se(rate, rate, n_per_group, n_per_group)``.
        """
        from ...physics.statistics import calc_proportion_gap_se

        return self.z * calc_proportion_gap_se(self.rate, self.rate, n_per_group, n_per_group)


class EqualOpportunityPriceScenario(BaseScenario):
    """Utility per applicant of raising ``group``'s TPR to ``reference_group``'s by moving its
    threshold, when that move raises its FPR by ``fpr_increase``: a repaid loan earns
    ``repay_margin``, a default loses ``default_loss`` (both USD).

    Notes
    -----
    Equation: Utility is ``TPR*base_rate*repay_margin-FPR*(1-base_rate)*default_loss``;
    threshold changes add their corresponding expected-value differences.

    Assumptions: The TPR target and added FPR are supplied jointly, with fixed base rates
    and monetary outcomes.

    Excludes: Estimating an ROC frontier, downstream behavioral changes, and a policy
    judgment about the acceptability of the tradeoff.

    References
    ----------
    Derivation: Utility is ``TPR*base_rate*repay_margin-FPR*(1-base_rate)*default_loss``;
    threshold changes add their corresponding expected-value differences. This is declared
    scenario accounting rather than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CreditServiceEqualOpportunityPrice
    >>> value = scenario.base_rate
    >>> round(float(value), 3)
    0.5
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    confusion: GroupConfusionScenario
    group: str
    reference_group: str
    repay_margin: Quantity
    default_loss: Quantity
    fpr_increase: float

    @property
    def base_rate(self) -> float:
        """Positive-label fraction for the selected group from the confusion-matrix scenario.

        Equation: ``confusion.base_rate(group)``.
        """
        return self.confusion.base_rate(self.group)

    @property
    def tpr_before(self) -> float:
        """Selected group's current true-positive rate, dimensionless.

        Equation: ``confusion.tpr(group)``.
        """
        return self.confusion.tpr(self.group)

    @property
    def tpr_after(self) -> float:
        """Reference group's true-positive rate used as the selected group's target.

        Equation: ``confusion.tpr(reference_group)``.
        """
        return self.confusion.tpr(self.reference_group)

    @property
    def fpr_before(self) -> float:
        """Selected group's current false-positive rate, dimensionless.

        Equation: ``confusion.fpr(group)``.
        """
        return self.confusion.fpr(self.group)

    @property
    def tpr_gap(self) -> float:
        """Target minus current true-positive rate, in fractional rate units.

        Equation: ``tpr_after - tpr_before``.
        """
        return self.tpr_after - self.tpr_before

    @property
    def baseline_utility(self) -> Quantity:
        """Expected USD per applicant from repaid true positives less losses on false positives.

        Equation: ``tpr_before * p * repay_margin - fpr_before * (1 - p) * default_loss``.
        """
        p = self.base_rate
        return self.tpr_before * p * self.repay_margin - self.fpr_before * (1 - p) * self.default_loss

    @property
    def added_repaid_value(self) -> Quantity:
        """Additional USD per applicant from the TPR increase at the same base rate.

        Equation: ``tpr_gap * base_rate * repay_margin``.
        """
        return self.tpr_gap * self.base_rate * self.repay_margin

    @property
    def added_default_cost(self) -> Quantity:
        """Additional USD per applicant from the supplied FPR increase among negative-label
        applicants.

        Equation: ``fpr_increase * (1 - base_rate) * default_loss``.
        """
        return self.fpr_increase * (1 - self.base_rate) * self.default_loss

    @property
    def net_change(self) -> Quantity:
        """Additional repaid-loan value minus additional default loss, in USD per applicant.

        Equation: ``added_repaid_value - added_default_cost``.
        """
        return self.added_repaid_value - self.added_default_cost

    @property
    def utility_after(self) -> Quantity:
        """Baseline utility plus modeled change, in USD per applicant.

        Equation: ``baseline_utility + net_change``.
        """
        return self.baseline_utility + self.net_change

    @property
    def loss_fraction(self) -> float:
        """Negative utility change divided by nonzero baseline utility, dimensionless.

        Equation: ``(-net_change / baseline_utility).to('').magnitude``.
        """
        return (-self.net_change / self.baseline_utility).to("").magnitude

    @property
    def group_share(self) -> float:
        """Selected group's applicant share from the confusion-matrix scenario.

        Equation: ``confusion.share(group)``.
        """
        return self.confusion.share(self.group)

    @property
    def population_average_loss(self) -> Quantity:
        """Selected-group utility loss weighted by its population share, in USD per population
        applicant.

        Equation: ``-net_change * group_share``.
        """
        return -self.net_change * self.group_share


class BiasFeedbackScenario(BaseScenario):
    """A gap passed at launch, refit from the next audit, iterated over retraining cycles:
    ``gap(k) = launch_gap * factor^k`` against a release ``gate`` (all in percentage points).

    Notes
    -----
    Equation: Disparity compounds as ``launch_gap*factor**cycles``; factor is fitted from the
    next audit and the gate-crossing index is rounded upward.

    Assumptions: One constant multiplicative factor represents each retraining cycle; gap
    and gate use matching percentage-point units.

    Excludes: A causal feedback model, saturation, sampling uncertainty, and assurance that
    the fitted factor persists.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.fairness.calc_cycles_to_gate``,
    ``mlsysim.physics.fairness.calc_feedback_disparity``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CreditServiceBiasFeedback
    >>> value = scenario.factor
    >>> round(float(value), 3)
    1.1
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    launch_gap: float
    next_audit_gap: float
    gate: float
    sampled_factor: float  # the refit factor once censored outcomes are sampled
    unmitigated_gap: float
    horizon_cycles: int

    @property
    def factor(self) -> float:
        """Multiplicative per-cycle disparity factor fitted from the first two audits.

        Equation: ``next_audit_gap / launch_gap``.
        """
        return self.next_audit_gap / self.launch_gap

    @property
    def crossing(self) -> float:
        """Continuous cycle index at which geometric disparity reaches the gate.

        Equation: ``calc_cycles_to_gate(launch_gap, factor, gate)``.
        """
        from ...physics.fairness import calc_cycles_to_gate

        return calc_cycles_to_gate(self.launch_gap, self.factor, self.gate)

    @property
    def breach_cycle(self) -> int:
        """Ceiling of the continuous gate-crossing index, including equality with the gate.

        Equation: ``_math.ceil(crossing)``.
        """
        return _math.ceil(self.crossing)

    def gap_at(self, cycles: int, factor: Optional[float] = None) -> float:
        """Disparity in percentage points after ``cycles`` under the chosen multiplicative factor.

        Equation: ``calc_feedback_disparity(launch_gap, factor or factor, cycles)``.
        """
        from ...physics.fairness import calc_feedback_disparity

        return calc_feedback_disparity(self.launch_gap, factor or self.factor, cycles)

    @property
    def gap_at_breach(self) -> float:
        """Fitted disparity at the integer gate-crossing cycle, in percentage points.

        Equation: ``gap_at(breach_cycle)``.
        """
        return self.gap_at(self.breach_cycle)

    @property
    def unmitigated_at_horizon(self) -> float:
        """Unmitigated launch gap compounded through the declared horizon, in percentage points.

        Equation: ``calc_feedback_disparity(unmitigated_gap, factor, horizon_cycles)``.
        """
        from ...physics.fairness import calc_feedback_disparity

        return calc_feedback_disparity(self.unmitigated_gap, self.factor, self.horizon_cycles)

    @property
    def sampled_gap_at_breach(self) -> float:
        """Gap at the original crossing cycle under the alternative sampled factor.

        Equation: ``gap_at(breach_cycle, sampled_factor)``.
        """
        return self.gap_at(self.breach_cycle, self.sampled_factor)


class ExplorationExposureScenario(BaseScenario):
    """A ranking service showing ``slots_per_user`` items a day to ``daily_users`` from a catalog of
    ``catalog_items``, with ``explore_share`` of slots served by randomized exploration.

    Notes
    -----
    Equation: Daily impressions are ``users*slots_per_user``, and expected exploration slots
    multiply by explore_share.

    Assumptions: Exposure accounting treats slots as opportunities, with a fixed exploration
    share.

    Excludes: Unique item coverage, repeat exposures, exploration quality, and
    individualized exposure distributions.

    References
    ----------
    Derivation: Daily impressions are ``users*slots_per_user``, and expected exploration slots
    multiply by explore_share. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RecommenderExplorationExposure
    >>> value = scenario.impressions
    >>> round(float(value), 3)
    200000000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    daily_users: int
    slots_per_user: int
    catalog_items: int
    explore_share: float

    @property
    def impressions(self) -> int:
        """Total daily recommendation slots, users times slots per user.

        Equation: ``daily_users * slots_per_user``.
        """
        return self.daily_users * self.slots_per_user

    @property
    def catalog_coverage(self) -> float:
        """One user's daily slot count divided by catalog size; no deduplication of repeated items.

        Equation: ``slots_per_user / catalog_items``.
        """
        return self.slots_per_user / self.catalog_items

    @property
    def explore_impressions(self) -> float:
        """Expected daily slots assigned to exploration under the declared fraction.

        Equation: ``impressions * explore_share``.
        """
        return self.impressions * self.explore_share


class AccuracyFrontierScenario(BaseScenario):
    """A hypothetical accuracy-against-disparity frontier, ``floor + span (1 - exp(-rate x))``, with
    named candidate disparities on it.

    Notes
    -----
    Equation: The explicitly constructed curve is ``floor+span*(1-exp(-rate*disparity))``; named
    points index supplied disparities.

    Assumptions: The curve is an illustrative functional form, not a fit to measured
    accuracy or disparity.

    Excludes: Evidence that a deployed classifier can realize these points or that the curve
    is a universal fairness frontier.

    References
    ----------
    Derivation: The explicitly constructed curve is ``floor+span*(1-exp(-rate*disparity))``;
    named points index supplied disparities. This is declared scenario accounting rather
    than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FairnessAccuracyFrontierSketch
    >>> value = scenario.point(next(iter(scenario.candidates)))
    >>> type(value).__name__
    'tuple'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floor: float
    span: float
    rate: float
    candidates: Dict[str, float]

    def accuracy(self, disparity):
        """Constructed accuracy for scalar or array disparity using the declared exponential
        frontier.

        Equation: ``floor + span * (1 - np.exp(-rate * np.asarray(disparity)))``.
        """
        import numpy as np

        return self.floor + self.span * (1 - np.exp(-self.rate * np.asarray(disparity)))

    def point(self, name: str):
        """Named candidate's disparity and constructed dimensionless accuracy.

        Equation: ``(x, float(accuracy(x)))``.
        """
        x = self.candidates[name]
        return x, float(self.accuracy(x))

    def disparity_ratio(self, worse: str, better: str) -> float:
        """Named worse candidate's disparity divided by the nonzero better candidate's disparity.

        Equation: ``candidates[worse] / candidates[better]``.
        """
        return self.candidates[worse] / self.candidates[better]


class DriftConfirmationScenario(BaseScenario):
    """Labeled outcomes that confirm a drop of ``drop`` from ``baseline_accuracy``.

    ``evidence_budget`` is the policy's count, set above the one-sample binomial
    floor at ``confidence`` (two-sided) and ``power``; ``label_rates`` maps a
    label to the rate at which labeled outcomes arrive. ``compared_drops`` are
    further drop sizes to size against the same floor.

    Notes
    -----
    Equation: One-proportion sample planning uses normal confidence/power quantiles;
    collection time is the chosen evidence count divided by label rate.

    Assumptions: Labels arrive at fixed named rates and outcome observations satisfy the
    independent binomial planning approximation.

    Excludes: Sequential testing corrections, labeling delay distributions, and proof the
    policy evidence budget meets the computed floor.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.statistics.calc_one_proportion_sample_size``. Their contracts supply
    the underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LabeledOutcomeDriftConfirmation
    >>> value = scenario.z_alpha
    >>> round(float(value), 3)
    1.96
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline_accuracy: float
    drop: float
    confidence: float
    power: float
    evidence_budget: int
    label_rates: Dict[str, Quantity]
    compared_drops: Optional[List[float]] = None

    @property
    def z_alpha(self) -> float:
        """Two-sided normal quantile for the declared confidence, dimensionless.

        Equation: ``NormalDist().inv_cdf(1 - (1 - confidence) / 2)``.
        """
        from statistics import NormalDist

        return NormalDist().inv_cdf(1 - (1 - self.confidence) / 2)

    @property
    def z_beta(self) -> float:
        """Normal quantile for the declared detection power, dimensionless.

        Equation: ``NormalDist().inv_cdf(power)``.
        """
        from statistics import NormalDist

        return NormalDist().inv_cdf(self.power)

    def outcomes_needed(self, drop: Optional[float] = None) -> float:
        """One-proportion planning count for the supplied or default accuracy drop.

        Equation: ``calc_one_proportion_sample_size(baseline_accuracy, baseline_accuracy - d,
        z_alpha, z_beta)``.
        """
        from ...physics.statistics import calc_one_proportion_sample_size

        d = self.drop if drop is None else drop
        return calc_one_proportion_sample_size(
            self.baseline_accuracy, self.baseline_accuracy - d, self.z_alpha, self.z_beta
        )

    def collection_time(self, rate: str) -> Quantity:
        """SI duration to collect the policy evidence budget at a named label arrival rate.

        Equation: ``(evidence_budget / label_rates[rate]).to_base_units()``.
        """
        return (self.evidence_budget / self.label_rates[rate]).to_base_units()

    def collection_time_for(self, samples: float, rate: str) -> Quantity:
        """SI duration to collect ``samples`` at a named positive label arrival rate.

        Equation: ``(samples / label_rates[rate]).to_base_units()``.
        """
        return (samples / self.label_rates[rate]).to_base_units()


class DetectionDelayLossScenario(BaseScenario):
    """A fault that cuts ``quality_drop`` of ``annual_value`` until it is detected,
    found after ``slow_detection`` or ``fast_detection``, ``incidents_per_year`` times a year.

    Notes
    -----
    Equation: Loss is annual value times quality drop times the fraction of a year spent
    undetected; incident frequency scales savings.

    Assumptions: Value accrues uniformly and each incident has the same constant quality
    loss.

    Excludes: Compounding, overlapping incidents, recovery costs, and uncertainty in the
    declared annual value.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.economics.calc_detection_delay_loss``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RecommenderDetectionDelayLoss
    >>> value = scenario.slow_loss
    >>> str(value.units)
    'USD'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    annual_value: Quantity  # one year's value
    quality_drop: float
    slow_detection: Quantity
    fast_detection: Quantity
    incidents_per_year: float

    def loss(self, delay: Quantity) -> Quantity:
        """USD lost while the fault persists for ``delay`` under a constant value-loss rate.

        Equation: ``calc_detection_delay_loss(annual_value, quality_drop, delay).to('USD')``.
        """
        from ...physics.economics import calc_detection_delay_loss

        return calc_detection_delay_loss(self.annual_value, self.quality_drop, delay).to("USD")

    @property
    def slow_loss(self) -> Quantity:
        """Per-incident USD loss under the slower detection duration.

        Equation: ``loss(slow_detection)``.
        """
        return self.loss(self.slow_detection)

    @property
    def fast_loss(self) -> Quantity:
        """Per-incident USD loss under the faster detection duration.

        Equation: ``loss(fast_detection)``.
        """
        return self.loss(self.fast_detection)

    @property
    def savings_per_incident(self) -> Quantity:
        """Slow minus fast loss, in USD per incident.

        Equation: ``slow_loss - fast_loss``.
        """
        return self.slow_loss - self.fast_loss

    @property
    def annual_savings(self) -> Quantity:
        """Expected USD saved in a year at the supplied incident frequency.

        Equation: ``(savings_per_incident * incidents_per_year).to('USD')``.
        """
        return (self.savings_per_incident * self.incidents_per_year).to("USD")

    @property
    def full_year_loss(self) -> Quantity:
        """USD lost if the quality drop persisted for an entire year.

        Equation: ``(annual_value * quality_drop).to('USD')``.
        """
        return (self.annual_value * self.quality_drop).to("USD")

    @property
    def slow_over_fast(self) -> float:
        """Ratio of slow-detection to nonzero fast-detection loss, dimensionless.

        Equation: ``(slow_loss / fast_loss).to('').magnitude``.
        """
        return (self.slow_loss / self.fast_loss).to("").magnitude


class RetrainingIntervalScenario(BaseScenario):
    """Retraining a model whose accuracy decays as ``drift`` fits it, serving
    ``query_rate`` queries worth ``value_per_query`` per unit accuracy, at
    ``run_cost`` a run. ``footprint`` and ``accelerator_price`` price the
    compute share of a run; ``compared_intervals`` and ``scheduled_interval``
    are cadences to compare with the optimum.

    Notes
    -----
    Equation: Average cost combines run_cost/interval with ``staleness_slope*interval/2``; the
    optimum balances these terms.

    Assumptions: Exponential accuracy decay is linearized locally, each retrain resets
    quality, and run cost is fixed.

    Excludes: Nonlinear long-interval drift, retraining duration, concurrent serving
    interference, and adaptation effectiveness uncertainty.

    References
    ----------
    Derivation: the scenario composes
    ``mlsysim.physics.economics.calc_optimal_retraining_interval``,
    ``mlsysim.physics.economics.calc_retraining_cost_rate``,
    ``mlsysim.physics.economics.calc_staleness_slope``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RankingRetrainingInterval
    >>> value = scenario.staleness_slope
    >>> str(value.units)
    'USD/d²'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    drift: AccuracyDecayScenario
    query_rate: Quantity  # queries per day
    value_per_query: Quantity  # USD per query for a unit change in accuracy fraction
    run_cost: Quantity
    footprint: Optional[Any] = None  # ReleaseFootprintScenario (retraining accelerators and time)
    accelerator_price: Optional[Quantity] = None
    compared_intervals: Optional[List[Quantity]] = None
    scheduled_interval: Optional[Quantity] = None

    @property
    def staleness_slope(self) -> Quantity:
        """Linearized growth in loss rate, in USD/day², from queries, value, initial accuracy, and
        fitted decay.

        Equation: ``calc_staleness_slope(query_rate, value_per_query, drift.initial_accuracy,
        drift.decay_rate).to('USD/day**2')``.
        """
        from ...physics.economics import calc_staleness_slope

        return calc_staleness_slope(
            self.query_rate, self.value_per_query, self.drift.initial_accuracy, self.drift.decay_rate
        ).to("USD/day**2")

    @property
    def optimal_interval(self) -> Quantity:
        """Days minimizing fixed retraining cost per interval plus linearized staleness cost.

        Equation: ``calc_optimal_retraining_interval(run_cost, staleness_slope).to('day')``.
        """
        from ...physics.economics import calc_optimal_retraining_interval

        return calc_optimal_retraining_interval(self.run_cost, self.staleness_slope).to("day")

    def cost_rate(self, interval: Quantity) -> Quantity:
        """USD/day for retraining at ``interval`` under the fixed-cost and linear-staleness
        approximation.

        Equation: ``calc_retraining_cost_rate(interval, run_cost,
        staleness_slope).to('USD/day')``.
        """
        from ...physics.economics import calc_retraining_cost_rate

        return calc_retraining_cost_rate(interval, self.run_cost, self.staleness_slope).to("USD/day")

    @property
    def compute_cost(self) -> Quantity:
        """USD compute portion of one retraining run, device-hours times accelerator-hour price.

        Equation: ``(footprint.retrain_accelerator_hours * accelerator_price).to('USD')``.
        """
        return (self.footprint.retrain_accelerator_hours * self.accelerator_price).to("USD")

    @property
    def compute_share(self) -> float:
        """Compute cost divided by full run cost, dimensionless.

        Equation: ``(compute_cost / run_cost).to('').magnitude``.
        """
        return (self.compute_cost / self.run_cost).to("").magnitude


class SegmentAccuracyScenario(BaseScenario):
    """Accuracy measured separately on each traffic segment; ``traffic_shares``
    and ``accuracies`` are keyed by segment name.

    Notes
    -----
    Equation: Overall accuracy sums traffic_share[k]*accuracy[k]; the worst segment
    minimizes the declared accuracy mapping.

    Assumptions: The two mappings use matching keys and traffic shares sum to one when
    interpreting the weighted sum as population accuracy.

    Excludes: Automatic share normalization, uncertainty intervals, and changes in segment
    traffic mix.

    References
    ----------
    Derivation: Overall accuracy sums traffic_share[k]*accuracy[k]; the worst segment
    minimizes the declared accuracy mapping. This is declared scenario accounting rather
    than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RecommendationAccuracyByDevice
    >>> value = scenario.total_share
    >>> round(float(value), 3)
    1.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    traffic_shares: Dict[str, float]
    accuracies: Dict[str, float]

    @property
    def total_share(self) -> float:
        """Sum of declared traffic shares; this property reports rather than normalizes them.

        Equation: ``sum(traffic_shares.values())``.
        """
        return sum(self.traffic_shares.values())

    @property
    def overall_accuracy(self) -> float:
        """Sum of traffic share times accuracy by matching segment key, dimensionless.

        Equation: ``sum((traffic_shares[k] * accuracies[k] for k in traffic_shares))``.
        """
        return sum(self.traffic_shares[k] * self.accuracies[k] for k in self.traffic_shares)

    @property
    def worst_segment(self) -> str:
        """Segment key with the smallest declared accuracy, preserving mapping tie order.

        Equation: ``min(accuracies, key=accuracies.get)``.
        """
        return min(self.accuracies, key=self.accuracies.get)
