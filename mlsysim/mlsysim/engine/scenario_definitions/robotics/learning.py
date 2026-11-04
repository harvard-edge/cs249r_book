"""Physical-machine scenarios for learning."""

from __future__ import annotations
import math as _math
from typing import Any, Optional
from ....core.types import Quantity
from ....core.units import Q_
from ..base import BaseScenario


class DiscountHorizonScenario(BaseScenario):
    """One ``rollout`` of decisions ``step`` apart under discount ``discount``: how far the return
    reaches, in decisions and in seconds, and the discount that keeps ``retained_weight`` of the
    final reward at the first decision.

    Notes
    -----
    Equation: Terminal weight=``discount**decisions``; effective horizon=1/(1-discount); time horizon=``steps*step``.

    Assumptions: A fixed decision period and stationary geometric discount in its valid range.

    Excludes: Policy quality or a causal memory horizon implied by the discount alone.

    References
    ----------
    Implementation references: ``mlsysim.physics.robotics.learning.calc_discount_effective_horizon``, ``mlsysim.physics.robotics.learning.calc_discount_for_retained_weight``, ``mlsysim.physics.robotics.learning.calc_discounted_weight``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchOpeningDiscount
    >>> scenario.decisions
    320
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    step: Quantity
    rollout: Quantity
    discount: float
    retained_weight: float = 0.5

    @property
    def decisions(self) -> int:
        """Number of decision intervals in a rollout, rounding rollout duration divided by step duration.

        Equation: int(round((self.rollout / self.step).to('').magnitude))
        """
        return int(round((self.rollout / self.step).to("").magnitude))

    @property
    def terminal_weight(self) -> float:
        """Discount weight of the final decision via ``calc_discounted_weight``; equals discount raised to the decision count.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.learning import calc_discounted_weight

        return calc_discounted_weight(self.discount, self.decisions)

    @property
    def horizon_steps(self) -> float:
        """Geometric effective horizon in decisions from ``calc_discount_effective_horizon``; assumes a stationary discount.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.learning import calc_discount_effective_horizon

        return calc_discount_effective_horizon(self.discount)

    @property
    def horizon_time(self) -> Quantity:
        """Effective discounted horizon in seconds: horizon steps times decision period.

        Equation: (self.horizon_steps * self.step).to('second')
        """
        return (self.horizon_steps * self.step).to("second")

    @property
    def retaining_discount(self) -> float:
        """Discount required to retain the declared terminal weight over the decision horizon, via ``calc_discount_for_retained_weight``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.learning import calc_discount_for_retained_weight

        return calc_discount_for_retained_weight(self.decisions, self.retained_weight)


class IterativeDecoderBudgetScenario(BaseScenario):
    """``machine``'s one-pass chunk decoder against an iterative head that encodes once and runs
    ``denoising_steps`` dependent evaluations in ``denoising_latency``, and a ``flow_steps`` head at
    the same cost per evaluation, on the setpoint step and the renewal period with
    ``renewal_jitter`` held free.

    Notes
    -----
    Equation: Per-evaluation time=(denoising_latency-encode)/denoising_steps; latency=``encode+evaluations*per_evaluation``.

    Assumptions: Equal sequential evaluation cost shared by compared decoder heads and a supplied jitter allowance.

    Excludes: Convergence/quality differences, overlap, and measured worst-case execution guarantees.

    References
    ----------
    Implementation references: ``mlsysim.physics.robotics.learning.calc_iterative_decoder_latency``, ``mlsysim.physics.robotics.learning.calc_max_decoder_evaluations``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchDecoderBudget
    >>> scenario.evaluations_within_renewal
    14
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # EmbodiedMobileManipulator (chunk step, period, p99 latency)
    denoising_steps: int
    flow_steps: int
    encode: Quantity
    denoising_latency: Quantity
    renewal_jitter: Quantity

    @property
    def setpoint_step(self) -> Quantity:
        """Machine setpoint period in ms, copied from ``machine.chunk_step``.

        Equation: self.machine.chunk_step.to('ms')
        """
        return self.machine.chunk_step.to("ms")

    @property
    def renewal_period(self) -> Quantity:
        """Machine chunk-renewal period in ms, copied from ``machine.chunk_period``.

        Equation: self.machine.chunk_period.to('ms')
        """
        return self.machine.chunk_period.to("ms")

    @property
    def one_pass_latency(self) -> Quantity:
        """Declared one-pass chunk inference latency in ms; no timing experiment is performed here.

        Equation: self.machine.chunk_inference_latency.to('ms')
        """
        return self.machine.chunk_inference_latency.to("ms")

    @property
    def per_evaluation(self) -> Quantity:
        """Average dependent decoder-evaluation time in ms after subtracting the one-time encode cost from denoising latency.

        Equation: ((self.denoising_latency - self.encode) / self.denoising_steps).to('ms')
        """
        return ((self.denoising_latency - self.encode) / self.denoising_steps).to("ms")

    @property
    def flow_latency(self) -> Quantity:
        """Encoding plus ``flow_steps`` equal-cost evaluations via ``calc_iterative_decoder_latency``; assumes the inferred evaluation cost transfers to this head.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.learning import calc_iterative_decoder_latency

        return calc_iterative_decoder_latency(self.encode, self.per_evaluation, self.flow_steps)

    @property
    def one_pass_slack(self) -> Quantity:
        """Renewal period minus one-pass latency in ms; negative values indicate an overrun.

        Equation: (self.renewal_period - self.one_pass_latency).to('ms')
        """
        return (self.renewal_period - self.one_pass_latency).to("ms")

    @property
    def denoising_slack(self) -> Quantity:
        """Renewal period minus denoising latency in ms, without jitter reserve.

        Equation: (self.renewal_period - self.denoising_latency).to('ms')
        """
        return (self.renewal_period - self.denoising_latency).to("ms")

    @property
    def one_pass_end_under_jitter(self) -> Quantity:
        """One-pass completion time in ms after adding the declared jitter allowance.

        Equation: (self.one_pass_latency + self.renewal_jitter).to('ms')
        """
        return (self.one_pass_latency + self.renewal_jitter).to("ms")

    @property
    def denoising_overrun_under_jitter(self) -> Quantity:
        """Denoising completion minus renewal deadline in ms after adding jitter; negative values retain slack.

        Equation: (self.denoising_latency + self.renewal_jitter - self.renewal_period).to('ms')
        """
        return (self.denoising_latency + self.renewal_jitter - self.renewal_period).to("ms")

    @property
    def renewal_less_jitter(self) -> Quantity:
        """Usable renewal time in ms after reserving jitter.

        Equation: (self.renewal_period - self.renewal_jitter).to('ms')
        """
        return (self.renewal_period - self.renewal_jitter).to("ms")

    def evaluation_room(self, budget: Quantity) -> float:
        """Unrounded count of equal-cost evaluations fitting a supplied time budget after encoding; may be negative when encoding alone exceeds budget.

        Equation: ((budget - self.encode) / self.per_evaluation).to('').magnitude
        """
        return ((budget - self.encode) / self.per_evaluation).to("").magnitude

    def max_evaluations(self, budget: Quantity) -> int:
        """Whole evaluations fitting a time budget via ``calc_max_decoder_evaluations``; inherits the helper's infeasible-budget handling.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.learning import calc_max_decoder_evaluations

        return calc_max_decoder_evaluations(budget, self.encode, self.per_evaluation)

    def latency_at(self, evaluations: int) -> Quantity:
        """Time for encoding plus a supplied integer evaluation count via ``calc_iterative_decoder_latency``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.learning import calc_iterative_decoder_latency

        return calc_iterative_decoder_latency(self.encode, self.per_evaluation, evaluations)

    @property
    def evaluations_within_step(self) -> int:
        """Whole decoder evaluations admitted by the setpoint period, using ``max_evaluations``.

        Equation: self.max_evaluations(self.setpoint_step)
        """
        return self.max_evaluations(self.setpoint_step)

    @property
    def evaluations_within_renewal(self) -> int:
        """Whole evaluations admitted by renewal time after reserving jitter, using ``max_evaluations``.

        Equation: self.max_evaluations(self.renewal_less_jitter)
        """
        return self.max_evaluations(self.renewal_less_jitter)


class ParallelSimRunScenario(BaseScenario):
    """A GPU-parallel on-policy run: ``robots`` environments each stepping ``steps_per_update`` times
    per update for ``updates`` updates within ``training_time``, at ``policy_rate``.

    Notes
    -----
    Equation: Transitions=``robots*steps_per_update*updates``; throughput=transitions/training_time.

    Assumptions: Uniform update sizes and a throughput representative of the requested extrapolation.

    Excludes: Reset overhead changes, simulator fidelity, and future hardware speedups.

    References
    ----------
    Derivation: Each update contains environment count times steps per environment. Multiplying by update count gives transitions; dividing by wall time gives aggregate throughput, while dividing by policy frequency gives aggregate simulated experience.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LeggedGymQuadrupedRun
    >>> scenario.batch
    98304
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    robots: int
    steps_per_update: int
    updates: int
    training_time: Quantity  # an upper bound ("under 20 minutes")
    policy_rate: Quantity

    @property
    def batch(self) -> int:
        """Policy transitions per update: environment count times steps per environment.

        Equation: self.robots * self.steps_per_update
        """
        return self.robots * self.steps_per_update

    @property
    def samples(self) -> int:
        """Total policy transitions: batch transitions times update count.

        Equation: self.batch * self.updates
        """
        return self.batch * self.updates

    @property
    def throughput(self) -> Quantity:
        """Policy transitions per second; if ``training_time`` is an upper bound, this rate is a lower bound.

        Equation: (self.samples / self.training_time).to('1/s')
        """
        return (self.samples / self.training_time).to("1/s")

    @property
    def simulated_experience(self) -> Quantity:
        """Aggregate environment experience in days: total transitions divided by policy frequency; not wall-clock duration.

        Equation: (self.samples / self.policy_rate).to('day')
        """
        return (self.samples / self.policy_rate).to("day")

    def time_for(self, steps: float) -> Quantity:
        """Estimated wall-clock minutes for a supplied number of policy transitions at this run's aggregate throughput.

        Equation: (steps / self.throughput).to('minute')
        """
        return (steps / self.throughput).to("minute")


class TrialCampaignScenario(BaseScenario):
    """``trials`` hardware trials of ``rollout`` each, decisions ``decision_step`` apart, with automated
    and manual resets, on one machine and on a farm; what a demonstration-seeded ``seeded_budget``
    allows; and the same trials in ``simulator`` at its published throughput.

    Notes
    -----
    Equation: Campaign time=``trials*(rollout+reset)/rigs``; simulator time=required_transitions/simulator_throughput.

    Assumptions: Identical trials, resets after each trial, and ideal work division across rigs.

    Excludes: Rig failures, reset variability, and simulation-to-hardware validity.

    References
    ----------
    Implementation references: ``mlsysim.physics.robotics.learning.calc_trial_campaign_wall_time``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchTrialCampaign
    >>> scenario.seeded_trials
    750
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    trials: int
    rollout: Quantity
    decision_step: Quantity
    automated_reset: Quantity
    manual_reset: Quantity
    farm_rigs: int
    reversals_per_trial: int
    seeded_budget: Quantity
    simulator: Optional[ParallelSimRunScenario] = None

    def _wall(self, reset, rigs=1) -> Quantity:
        from ....physics.robotics.learning import calc_trial_campaign_wall_time

        return calc_trial_campaign_wall_time(self.trials, self.rollout, reset, rigs)

    @property
    def execution_time(self) -> Quantity:
        """Serial rollout time with zero reset overhead, via ``calc_trial_campaign_wall_time``.

        Equation: self._wall(Q_(0, 's'))
        """
        return self._wall(Q_(0, "s"))

    @property
    def automated_time(self) -> Quantity:
        """Campaign time including the supplied automated reset duration after each trial, via the campaign helper.

        Equation: self._wall(self.automated_reset)
        """
        return self._wall(self.automated_reset)

    @property
    def manual_time(self) -> Quantity:
        """Campaign time including manual reset duration after each trial, on one rig.

        Equation: self._wall(self.manual_reset)
        """
        return self._wall(self.manual_reset)

    @property
    def farm_manual_time(self) -> Quantity:
        """Campaign days with the same trials ideally divided across ``farm_rigs``; excludes scheduling imbalance.

        Equation: self._wall(self.manual_reset, self.farm_rigs).to('day')
        """
        return self._wall(self.manual_reset, self.farm_rigs).to("day")

    @property
    def manual_reset_share(self) -> float:
        """Fraction of a rollout/reset cycle spent resetting manually.

        Equation: (self.manual_reset / (self.rollout + self.manual_reset)).to('').magnitude
        """
        return (self.manual_reset / (self.rollout + self.manual_reset)).to("").magnitude

    @property
    def reversals(self) -> int:
        """Mechanical reversal count over all requested trials.

        Equation: self.trials * self.reversals_per_trial
        """
        return self.trials * self.reversals_per_trial

    @property
    def seeded_trials(self) -> int:
        """Whole automated-reset trials fitting the demonstration-seeded budget, rounded down.

        Equation: int(_math.floor((self.seeded_budget / (self.rollout + self.automated_reset)).to('').magnitude))
        """
        return int(_math.floor((self.seeded_budget / (self.rollout + self.automated_reset)).to("").magnitude))

    @property
    def seeded_reversals(self) -> int:
        """Reversal count in those budget-limited seeded trials.

        Equation: self.seeded_trials * self.reversals_per_trial
        """
        return self.seeded_trials * self.reversals_per_trial

    @property
    def trial_ratio(self) -> float:
        """Requested trial count divided by seeded trials; undefined if no seeded trial fits.

        Equation: self.trials / self.seeded_trials
        """
        return self.trials / self.seeded_trials

    @property
    def decisions_per_trial(self) -> int:
        """Policy decision count per rollout, rounding duration divided by decision period.

        Equation: int(round((self.rollout / self.decision_step).to('').magnitude))
        """
        return int(round((self.rollout / self.decision_step).to("").magnitude))

    @property
    def simulated_steps(self) -> int:
        """Policy transitions needed to simulate all requested trials.

        Equation: self.trials * self.decisions_per_trial
        """
        return self.trials * self.decisions_per_trial

    @property
    def simulated_time(self) -> Quantity:
        """Wall-clock time for those transitions at the supplied simulator's reported throughput, via ``simulator.time_for``.

        Equation: self.simulator.time_for(self.simulated_steps)
        """
        return self.simulator.time_for(self.simulated_steps)

    @property
    def simulation_speedup(self) -> float:
        """Automated hardware campaign time divided by modeled simulator time; compares the supplied rates and excludes simulation transfer fidelity.

        Equation: (self.automated_time / self.simulated_time).to('').magnitude
        """
        return (self.automated_time / self.simulated_time).to("").magnitude


class ContactDeadlineScenario(BaseScenario):
    """A constant-speed drive into ``site``'s latch plate: when the force ramp k v t reaches the
    tripwire and the limit, when braking can begin, and the time left, at both approaches.

    Notes
    -----
    Equation: Linear contact ramp F(t)=``stiffness*speed*t``; force-limit deadline=``force_limit/(stiffness*speed)``.

    Assumptions: Constant approach speed until brake onset, linear contact stiffness, and fixed response delay.

    Excludes: Post-onset braking dynamics, nonlinear impact, and certified safety margins.

    References
    ----------
    Implementation references: ``mlsysim.physics.robotics.actuation.calc_contact_force``, ``mlsysim.physics.robotics.actuation.calc_contact_yield_deadline``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LatchContactDeadlines
    >>> round((scenario.time_left_guarded).to('ms').magnitude, 6)
    5.083333
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedScenario (latch stiffness, limit, tripwire, contact response)

    def deadline(self, speed: Quantity) -> Quantity:
        """Time in ms for a constant-speed linear spring contact to reach the force limit, via ``calc_contact_yield_deadline``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.actuation import calc_contact_yield_deadline

        s = self.site
        return calc_contact_yield_deadline(s.f_latch_limit, s.k_latch, speed).to("ms")

    def tripwire_time(self, speed: Quantity) -> Quantity:
        """Time in ms to reach the tripwire force under the ramp ``stiffness * speed * time``.

        Equation: (self.site.f_latch_tripwire / (self.site.k_latch * speed)).to('ms')
        """
        return (self.site.f_latch_tripwire / (self.site.k_latch * speed)).to("ms")

    def brake_onset(self, speed: Quantity) -> Quantity:
        """Brake-command onset in ms: tripwire crossing plus the site's fixed contact-response delay.

        Equation: (self.tripwire_time(speed) + self.site.t_contact_response).to('ms')
        """
        return (self.tripwire_time(speed) + self.site.t_contact_response).to("ms")

    def time_left(self, speed: Quantity) -> Quantity:
        """Milliseconds from brake onset to the force-limit crossing; negative values mean the assumed response is already late.

        Equation: (self.deadline(speed) - self.brake_onset(speed)).to('ms')
        """
        return (self.deadline(speed) - self.brake_onset(speed)).to("ms")

    @property
    def guarded_speed(self) -> Quantity:
        """Guarded approach velocity supplied by the site's ``v_latch_approach``.

        Equation: self.site.v_latch_approach
        """
        return self.site.v_latch_approach

    @property
    def fast_speed(self) -> Quantity:
        """Faster comparison velocity supplied by the site's ``v_latch_high``.

        Equation: self.site.v_latch_high
        """
        return self.site.v_latch_high

    @property
    def deadline_guarded(self) -> Quantity:
        """Force-limit deadline in ms at guarded speed, via ``deadline``.

        Equation: self.deadline(self.guarded_speed)
        """
        return self.deadline(self.guarded_speed)

    @property
    def deadline_fast(self) -> Quantity:
        """Force-limit deadline in ms at faster speed, via ``deadline``.

        Equation: self.deadline(self.fast_speed)
        """
        return self.deadline(self.fast_speed)

    @property
    def tripwire_guarded(self) -> Quantity:
        """Tripwire crossing time in ms at guarded speed, via ``tripwire_time``.

        Equation: self.tripwire_time(self.guarded_speed)
        """
        return self.tripwire_time(self.guarded_speed)

    @property
    def tripwire_fast(self) -> Quantity:
        """Tripwire crossing time in ms at faster speed, via ``tripwire_time``.

        Equation: self.tripwire_time(self.fast_speed)
        """
        return self.tripwire_time(self.fast_speed)

    @property
    def brake_onset_guarded(self) -> Quantity:
        """Brake onset time in ms at guarded speed, via ``brake_onset``.

        Equation: self.brake_onset(self.guarded_speed)
        """
        return self.brake_onset(self.guarded_speed)

    @property
    def brake_onset_fast(self) -> Quantity:
        """Brake onset time in ms at faster speed, via ``brake_onset``.

        Equation: self.brake_onset(self.fast_speed)
        """
        return self.brake_onset(self.fast_speed)

    @property
    def time_left_guarded(self) -> Quantity:
        """Remaining response margin in ms at guarded speed, via ``time_left``.

        Equation: self.time_left(self.guarded_speed)
        """
        return self.time_left(self.guarded_speed)

    @property
    def time_left_fast(self) -> Quantity:
        """Remaining response margin in ms at faster speed, via ``time_left``.

        Equation: self.time_left(self.fast_speed)
        """
        return self.time_left(self.fast_speed)

    @property
    def fast_margin_ratio(self) -> float:
        """Dimensionless force-limit deadline divided by brake-onset time at faster speed; a timing comparison, not a safety certificate.

        Equation: (self.deadline_fast / self.brake_onset_fast).to('').magnitude
        """
        return (self.deadline_fast / self.brake_onset_fast).to("").magnitude

    # Optional (added 2026-10-20): the force when the brake command issues, and
    # a counterfactual stall of ``stall`` at the guarded approach with no local
    # trip, priced by the plate's spring model k x.
    stall: Optional[Quantity] = None

    def force_rate(self, speed: Quantity) -> Quantity:
        """Linear contact-force rise in N/ms: site stiffness times constant approach speed.

        Equation: (self.site.k_latch * speed).to('N/ms')
        """
        return (self.site.k_latch * speed).to("N/ms")

    def force_at_brake_onset(self, speed: Quantity) -> Quantity:
        """Force in N when braking begins, multiplying constant force rate by the modeled onset time.

        Equation: (self.force_rate(speed) * self.brake_onset(speed)).to('N')
        """
        return (self.force_rate(speed) * self.brake_onset(speed)).to("N")

    @property
    def force_at_brake_onset_guarded(self) -> Quantity:
        """Brake-onset force in N at the guarded speed, via ``force_at_brake_onset``.

        Equation: self.force_at_brake_onset(self.guarded_speed)
        """
        return self.force_at_brake_onset(self.guarded_speed)

    @property
    def compression_left_guarded(self) -> Quantity:
        """Remaining linear-spring compression in mm before the force limit at guarded brake onset; negative values indicate excess force already.

        Equation: ((self.site.f_latch_limit - self.force_at_brake_onset_guarded) / self.site.k_latch).to('mm')
        """
        return ((self.site.f_latch_limit - self.force_at_brake_onset_guarded) / self.site.k_latch).to("mm")

    @property
    def stall_travel(self) -> Quantity:
        """Unbraked constant-speed travel in mm during the optional stall duration at guarded speed.

        Equation: (self.guarded_speed * self.stall).to('mm')
        """
        return (self.guarded_speed * self.stall).to("mm")

    @property
    def stall_force(self) -> Quantity:
        """Linear-spring contact force in N after that stall travel, via ``calc_contact_force``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ....physics.robotics.actuation import calc_contact_force

        return calc_contact_force(self.site.k_latch, self.stall_travel).to("N")

    @property
    def stall_over_limit(self) -> float:
        """Dimensionless stall-force-to-limit ratio; values above one exceed the declared limit under the model.

        Equation: (self.stall_force / self.site.f_latch_limit).to('').magnitude
        """
        return (self.stall_force / self.site.f_latch_limit).to("").magnitude


class SoftContactExploitScenario(BaseScenario):
    """``machine``'s arm coasting into ``site``'s latch plate at the fast approach: the peak force a
    penalty spring that lets the gripper sink ``penalty_penetration`` reports, the peak the real
    plate imposes on a rigid arm, and the arm stiffness at which the series chain reaches the limit.
    The moving mass is the arm's effective contact mass without its payload capacity (the gripper
    holds no object at the door).

    Notes
    -----
    Equation: Impact energy ``m*v**2/2`` becomes spring energy ``k*x**2/2``; series compliance sets the effective k.

    Assumptions: Linear undamped springs, empty gripper, and the declared effective contact mass.

    Excludes: Damping, controller intervention, nonlinear contact, and any safety certification.

    References
    ----------
    Derivation: Conservation of kinetic energy against a linear spring gives F_peak =
    ``v*sqrt(m*k)``. Springs in series add their compliances. The implementation composes
    mlsysim.physics.robotics.actuation.calc_transient_impact_force with
    mlsysim.physics.robotics.learning.calc_series_stiffness and
    calc_complementary_series_stiffness.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.LatchSoftSpringExploit
    >>> s.penalty_timescale.to("second").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # EmbodiedMobileManipulator (arm contact mass, payload capacity)
    site: Any  # EmbodiedScenario (latch stiffness, limit, approach speeds)
    penalty_penetration: Quantity

    @property
    def moving_mass(self) -> Quantity:
        """Arm contact mass in kg after subtracting payload capacity for an empty gripper.

        Equation: ``(machine.arm_effective_contact_mass -
        machine.arm.payload_capacity).to('kg')``."""
        return (self.machine.arm_effective_contact_mass - self.machine.arm.payload_capacity).to("kg")

    @property
    def penalty_stiffness(self) -> Quantity:
        """Effective penalty-spring stiffness from stopping the moving mass over the assumed penetration.

        Equation: ``calc_penalty_stiffness_from_penetration(moving_mass, site.v_latch_high,
        penalty_penetration)``."""
        from ....physics.robotics.learning import calc_penalty_stiffness_from_penetration

        return calc_penalty_stiffness_from_penetration(
            self.moving_mass, self.site.v_latch_high, self.penalty_penetration
        )

    def peak_force(self, stiffness: Quantity, speed: Optional[Quantity] = None) -> Quantity:
        """Undamped impact-force peak in N at the supplied stiffness and speed; an omitted speed uses the fast approach.

        Equation: Delegates F_peak = speed * sqrt(moving_mass * stiffness) to
        mlsysim.physics.robotics.actuation.calc_transient_impact_force."""
        from ....physics.robotics.actuation import calc_transient_impact_force

        v = self.site.v_latch_high if speed is None else speed
        return calc_transient_impact_force(v, self.moving_mass, stiffness).to("N")

    def chain_peak_force(self, arm_stiffness: Quantity) -> Quantity:
        """Fast-approach peak with the arm in series with the plate.

        Equation: ``peak_force(calc_series_stiffness(site.k_latch, arm_stiffness))``."""
        from ....physics.robotics.learning import calc_series_stiffness

        return self.peak_force(calc_series_stiffness(self.site.k_latch, arm_stiffness))

    @property
    def simulated_peak(self) -> Quantity:
        """Impact-force peak in N predicted by the penetration-derived penalty spring.

        Equation: ``peak_force(penalty_stiffness)``."""
        return self.peak_force(self.penalty_stiffness)

    @property
    def rigid_peak(self) -> Quantity:
        """Impact-force peak in N with the latch plate as the only compliance.

        Equation: ``peak_force(site.k_latch)``."""
        return self.peak_force(self.site.k_latch)

    @property
    def guarded_rigid_peak(self) -> Quantity:
        """Impact-force peak in N at the guarded approach speed against the plate.

        Equation: ``peak_force(site.k_latch, site.v_latch_approach)``."""
        return self.peak_force(self.site.k_latch, self.site.v_latch_approach)

    @property
    def break_even_series_stiffness(self) -> Quantity:
        """Combined series stiffness in N/m whose impact peak reaches the latch-force limit.

        Equation: ``calc_impact_stiffness_for_force(site.f_latch_limit, site.v_latch_high,
        moving_mass)``."""
        from ....physics.robotics.learning import calc_impact_stiffness_for_force

        return calc_impact_stiffness_for_force(self.site.f_latch_limit, self.site.v_latch_high, self.moving_mass)

    @property
    def break_even_arm_stiffness(self) -> Quantity:
        """Arm stiffness in N/m that, in series with the plate, reaches the force-limit stiffness.

        Equation: ``calc_complementary_series_stiffness(break_even_series_stiffness,
        site.k_latch)``."""
        from ....physics.robotics.learning import calc_complementary_series_stiffness

        return calc_complementary_series_stiffness(self.break_even_series_stiffness, self.site.k_latch)

    @property
    def plate_over_penalty(self) -> float:
        """Dimensionless plate-to-penalty stiffness ratio, exposing artificially soft simulated contact.

        Equation: ``(site.k_latch / penalty_stiffness).to('').magnitude``."""
        return (self.site.k_latch / self.penalty_stiffness).to("").magnitude

    @property
    def rigid_over_limit(self) -> float:
        """Dimensionless rigid-arm force peak divided by the permitted latch force.

        Equation: ``(rigid_peak / site.f_latch_limit).to('').magnitude``."""
        return (self.rigid_peak / self.site.f_latch_limit).to("").magnitude

    @property
    def penalty_timescale(self) -> Quantity:
        """Contact timescale sqrt(m/k) in seconds for the penalty spring.

        Equation: ``calc_contact_timescale(moving_mass, penalty_stiffness)``."""
        from ....physics.robotics.learning import calc_contact_timescale

        return calc_contact_timescale(self.moving_mass, self.penalty_stiffness)

    @property
    def plate_timescale(self) -> Quantity:
        """Contact timescale sqrt(m/k) in seconds for the latch plate.

        Equation: ``calc_contact_timescale(moving_mass, site.k_latch)``."""
        from ....physics.robotics.learning import calc_contact_timescale

        return calc_contact_timescale(self.moving_mass, self.site.k_latch)

    @property
    def timescale_ratio(self) -> float:
        """Dimensionless penalty-contact duration relative to plate-contact duration.

        Equation: ``(penalty_timescale / plate_timescale).to('').magnitude``."""
        return (self.penalty_timescale / self.plate_timescale).to("").magnitude


class ForceRiseGapScenario(BaseScenario):
    """An illustrative linear drive regulating ``target_force`` against a surface, in simulation and on
    hardware, with one stated discrepancy in each reality-gap channel: dynamics (force-rise rate and
    drive delay), timing (p99 actuation delay against the simulator's), sensing (range bias and
    dropout), and appearance (pixel difference hiding a contact-patch shift).

    Notes
    -----
    Equation: Target time is force/rise_rate; missed travel is speed times the extra actuation delay.

    Assumptions: Constant force-rise rates and approach speed over the compared delay windows.

    Excludes: Closed-loop dynamics, statistical fitting of the supplied p99, and coupled reality-gap channels.

    References
    ----------
    Derivation: A constant force slope reaches a specified force after force/slope. A
    constant approach speed traverses ``speed*extra_delay`` during the omitted delay. The
    supplied simulated and physical rates define the comparison; the class does not
    estimate them.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.LinearDriveRealityGap
    >>> s.physical_time_to_target.to("ms").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    approach_speed: Quantity
    target_force: Quantity
    simulated_rise_rate: Quantity
    physical_rise_rate: Quantity
    p99_delay: Quantity
    simulated_delay: Quantity
    simulator_step: Quantity  # integration period, not pipeline latency
    drive_delay: Quantity
    overtravel: Quantity
    range_bias: Quantity
    dropout_frames: int
    pixel_difference: float  # fraction of the image
    patch_shift: Quantity
    chamfer_tolerance: Quantity

    @property
    def simulated_time_to_target(self) -> Quantity:
        """Time in ms to reach the target under the assumed simulated linear force-rise rate.

        Equation: ``(target_force / simulated_rise_rate).to('ms')``."""
        return (self.target_force / self.simulated_rise_rate).to("ms")

    @property
    def physical_time_to_target(self) -> Quantity:
        """Time in ms to reach the target under the supplied hardware force-rise rate.

        Equation: ``(target_force / physical_rise_rate).to('ms')``."""
        return (self.target_force / self.physical_rise_rate).to("ms")

    @property
    def unmodeled_travel(self) -> Quantity:
        """Travel before correction that the simulator's delay does not predict.

        Equation: ``(approach_speed * (p99_delay - simulated_delay)).to('mm')``."""
        return (self.approach_speed * (self.p99_delay - self.simulated_delay)).to("mm")


class OnRobotFineTuneScenario(BaseScenario):
    """``interaction_steps`` setpoints ``step`` apart collected on the machine to absorb a
    ``plate_offset``, of which ``exceed_fraction`` cross the contact tripwire.

    Notes
    -----
    Equation: Rate is 1/step, active time is ``count*step``, and expected exceedances are rounded ``count*fraction``.

    Assumptions: Every interaction consumes one equal-length interval; exceed_fraction is supplied.

    Excludes: Training efficacy, convergence, and force-limit guarantees.

    References
    ----------
    Derivation: Equally spaced setpoints have frequency 1/step and consume ``count*step`` of
    active time. Multiplying the count by the supplied exceedance fraction gives an
    expected count, rounded to an integer for the displayed sample budget.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.LatchResidualFineTune
    >>> s.active_time.to("second").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    step: Quantity
    interaction_steps: int
    exceed_fraction: float
    plate_offset: Quantity

    @property
    def setpoint_rate(self) -> Quantity:
        """Setpoint frequency in Hz, the reciprocal of the integration interval.

        Equation: ``(1 / step).to('Hz')``."""
        return (1 / self.step).to("Hz")

    @property
    def active_time(self) -> Quantity:
        """Interaction duration in seconds for the configured setpoint count.

        Equation: ``(interaction_steps * step).to('second')``."""
        return (self.interaction_steps * self.step).to("second")

    @property
    def exceed_samples(self) -> int:
        """Tripwire-crossing sample count, rounded from the assumed crossing fraction.

        Equation: ``int(round(interaction_steps * exceed_fraction))``."""
        return int(round(self.interaction_steps * self.exceed_fraction))
