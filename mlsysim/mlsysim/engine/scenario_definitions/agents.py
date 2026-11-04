"""Model calls, context, tools, and coordinated agent execution.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
import math as _math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from .base import BaseScenario
from .serving import DecodeScenario


class HorizonCompoundingScenario(BaseScenario):
    """Independent, unchecked steps each right with ``step_success``; success at each horizon.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.HorizonCompoundingModel``.

    Assumptions: The horizons count unchecked steps with the success assumptions in
    HorizonCompoundingModel; None removes an optional horizon.

    Excludes: Task-dependent correlations or recovery behavior are absent unless represented
    by the chosen solver.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.HorizonCompoundingModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.UncheckedRepairHorizon
    >>> type(case.horizons).__name__
    'tuple'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    step_success: float
    short_horizon: int
    long_horizon: int  # the horizon the target must hold at
    target_success: float
    extended_horizon: Optional[int] = None  # a further horizon to price (for example, twice the long one)

    @property
    def horizons(self):
        """Configured unchecked-step horizons with absent optional horizons removed. Equation:
        retain non-None values from short, long, and extended horizons. Units: steps; ordering
        follows the inputs.
        """
        return tuple(h for h in (self.short_horizon, self.long_horizon, self.extended_horizon) if h is not None)

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.HorizonCompoundingModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import HorizonCompoundingModel

        return HorizonCompoundingModel().solve(self.step_success, self.horizons, self.target_success, self.long_horizon)


class AcceptedTaskCostScenario(BaseScenario):
    """An attempt of model tokens plus other cost, retried until accepted; optional levers on
    it.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AcceptedTaskCostModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.AcceptedTaskCostModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.AcceptedTaskCostModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AcceptedTaskCostModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairAttemptCostLevers
    >>> type(case.result).__name__
    'AcceptedTaskCostResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model_cost: Quantity  # USD per attempt
    other_cost: Quantity  # USD per attempt: sandbox, tests, review
    acceptance: float
    token_price_cut: Optional[float] = None  # fraction of the token price removed
    improved_acceptance: Optional[float] = None
    other_cost_cut: Optional[float] = None  # fraction of the other cost removed
    acceptance_gain: Optional[float] = None  # added to acceptance
    independent_attempts: Optional[int] = None
    # Optional (added 2026-10-20): more model calls per attempt, as a fraction
    # added to the model cost (1.0 doubles it), and the acceptance that keeps
    # cost per accepted task unchanged at the costlier attempt.
    model_cost_growth: Optional[float] = None

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AcceptedTaskCostModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import AcceptedTaskCostModel

        return AcceptedTaskCostModel().solve(
            self.model_cost,
            self.other_cost,
            self.acceptance,
            self.token_price_cut,
            self.improved_acceptance,
            self.other_cost_cut,
            self.acceptance_gain,
            self.independent_attempts,
            self.model_cost_growth,
        )


class AppendOnlyContextScenario(BaseScenario):
    """A trajectory that appends each turn to its prompt and resends the whole history.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AppendOnlyContextModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.AppendOnlyContextModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.AppendOnlyContextModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AppendOnlyContextModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AppendOnlyRepairContext
    >>> type(case.result).__name__
    'AppendOnlyContextResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    initial_tokens: Quantity
    tokens_per_turn: Quantity
    turns: int
    context_window: Quantity

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AppendOnlyContextModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import AppendOnlyContextModel

        return AppendOnlyContextModel().solve(
            self.initial_tokens, self.tokens_per_turn, self.turns, self.context_window
        )


class ContextAdmissionScenario(BaseScenario):
    """A serialized request (text plus an image charge) against a window with a reserved reply.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ContextAdmissionModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ContextAdmissionModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ContextAdmissionModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ContextAdmissionModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MultimodalContextAdmission
    >>> type(case.result).__name__
    'ContextAdmissionResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: int
    output_reserve: int
    text_tokens: int
    image_tokens: int = 0

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ContextAdmissionModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import ContextAdmissionModel

        return ContextAdmissionModel().solve(self.window, self.output_reserve, self.text_tokens, self.image_tokens)


class SamplingScenario(BaseScenario):
    """One fixed logit vector at a base and a sharpened temperature, then top-p at the base.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.SamplingModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.SamplingModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.SamplingModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.SamplingModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TemperatureNucleusExample
    >>> type(case.result).__name__
    'SamplingResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    logits: List[float]
    base_temperature: float
    sharp_temperature: float
    top_p: float

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.SamplingModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import SamplingModel

        return SamplingModel().solve(self.logits, self.base_temperature, self.sharp_temperature, self.top_p)


class CheckerPosteriorScenario(BaseScenario):
    """A fallible check over candidates, ``defect_rate`` of which violate the property.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.CheckerPosteriorModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.CheckerPosteriorModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.CheckerPosteriorModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.CheckerPosteriorModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PatchCheckerPosterior
    >>> type(case.result).__name__
    'CheckerPosteriorResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    defect_rate: float
    false_accept_rate: float  # P(PASS | defective)
    valid_accept_rate: float  # P(PASS | valid)

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.CheckerPosteriorModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import CheckerPosteriorModel

        return CheckerPosteriorModel().solve(self.defect_rate, self.false_accept_rate, self.valid_accept_rate)


class ModelCallScenario(BaseScenario):
    """One call of ``prompt_tokens`` (+ ``instruction_tokens``) in and ``output_tokens`` out at
    ``floors``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ModelCallModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ModelCallModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ModelCallModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ModelCallModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CodingAgentFirstTurnCall
    >>> type(case.result).__name__
    'ModelCallResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    prompt_tokens: int
    output_tokens: int
    instruction_tokens: int = 0  # format instructions added to the prompt
    shared_prefix_tokens: int = 0  # prompt prefix served from cache
    deadline: Optional[Quantity] = None
    stage_success: Optional[Dict[str, float]] = None  # stages a reply must clear to be accepted

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ModelCallModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import ModelCallModel

        f = self.floors.result
        return ModelCallModel().solve(
            f.compute_time,
            f.memory_time,
            self.prompt_tokens + self.instruction_tokens,
            self.output_tokens,
            self.shared_prefix_tokens,
            self.deadline,
            self.stage_success,
        )


class StreamCancelScenario(BaseScenario):
    """A streamed reply with ``output_allowance`` tokens, rejected and canceled at
    ``reject_step``.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Cancellation prevents all remaining allowed decode steps immediately, each
    priced at the fixed memory-time floor.

    Excludes: Compute-bound decode, KV growth, cancellation latency, and already-generated
    buffered tokens are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CanceledStreamDecodeSavings
    >>> round(case.skipped_steps, 4)
    4000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    output_allowance: int
    reject_step: int

    @property
    def skipped_steps(self) -> int:
        """Decode steps avoided by cancellation at the stated rejection step. Equation:
        ``output_allowance - reject_step``. Units: tokens/steps; cancellation is assumed
        immediate.
        """
        return self.output_allowance - self.reject_step

    @property
    def decode_time_avoided(self) -> Quantity:
        """Weight-read-floor time attributed to canceled decode steps. Equation: ``skipped_steps *
        floors.result.memory_time``. Units: seconds; other decode work and cancellation overhead
        are excluded.
        """
        return (self.skipped_steps * self.floors.result.memory_time).to("second")

    @property
    def allowance_decode_time(self) -> Quantity:
        """Weight-read-floor time for the full output allowance. Equation: ``output_allowance *
        floors.result.memory_time``. Units: seconds; context growth and serving overhead are
        excluded.

        Decode time of the whole allowance (added 2026-10-20): ``K_max t_dec``, the longest
        one admitted call postpones the next turn boundary when control waits for its reply.
        """
        return (self.output_allowance * self.floors.result.memory_time).to("second")


class TrajectoryResendScenario(BaseScenario):
    """A trajectory of ``turns`` calls that each resend the whole history
    (``TrajectoryResendModel``).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TrajectoryResendModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.TrajectoryResendModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TrajectoryResendModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TrajectoryResendModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CodingAgentResend30Turns
    >>> type(case.result).__name__
    'TrajectoryResendResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    first_prompt: int
    appended: int  # model output plus tool observation added per turn
    output_per_turn: int
    turns: int

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TrajectoryResendModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import TrajectoryResendModel

        f = self.floors.result
        return TrajectoryResendModel().solve(
            f.compute_time, f.memory_time, self.first_prompt, self.appended, self.output_per_turn, self.turns
        )


class ProbeVsAnswerScenario(BaseScenario):
    """A generated explanation of ``answer_tokens`` against a prepared test of ``probe_time``.

    Notes
    -----
    Equation: result = ProbeVsAnswerModel.solve(...) with the declared floors,
    answer_tokens, probe_time inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ProbeVsAnswerModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond ProbeVsAnswerModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ProbeVsAnswerModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ProbeVsGeneratedExplanation
    >>> round(float(case.result.token_time.to('s').magnitude), 3)
    0.021"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    answer_tokens: int
    probe_time: Quantity
    output_price: Quantity

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ProbeVsAnswerModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ProbeVsAnswerModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import ProbeVsAnswerModel

        return ProbeVsAnswerModel().solve(
            self.floors.result.memory_time, self.answer_tokens, self.probe_time, self.output_price
        )


class AllocationAxesScenario(BaseScenario):
    """One output ``ceiling`` spent as one chain, ``candidates`` streams, or ``rounds`` of feedback.

    Notes
    -----
    Equation: result = AllocationAxesModel.solve(...) with the declared floors, prompt,
    ceiling inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.AllocationAxesModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond AllocationAxesModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.AllocationAxesModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.OutputCeilingThreeAllocations
    >>> round(float(case.result.token_time.to('s').magnitude), 3)
    0.021"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    prompt: int
    ceiling: int
    candidates: int  # breadth: concurrent streams, each billed its own prompt
    candidate_tokens: int  # generated tokens per breadth or feedback candidate
    rounds: int  # feedback rounds
    report_tokens: int  # check report appended after each round
    check_time: Quantity  # per feedback check
    input_price: Quantity
    output_price: Quantity
    context_growth: Optional[int] = None  # a grown prompt, to price breadth's premium again

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.AllocationAxesModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.AllocationAxesModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import AllocationAxesModel

        return AllocationAxesModel().solve(
            self.floors.result.memory_time,
            self.prompt,
            self.ceiling,
            self.candidates,
            self.candidate_tokens,
            self.rounds,
            self.report_tokens,
            self.check_time,
            self.input_price,
            self.output_price,
            self.context_growth,
        )


class BestOfNPoolScenario(BaseScenario):
    """Pools of independent candidates at ``pass_rate``, and a mixed workload with the same average.

    Notes
    -----
    Equation: result = BestOfNPoolModel.solve(...) with the declared pass_rate,
    pool_sizes, easy_pass_rate inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.BestOfNPoolModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond BestOfNPoolModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.BestOfNPoolModel.solve``. The solver contract records the equations
    and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IndependentCandidatePools
    >>> round(float(case.result.last_gain), 3)
    0.025"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pass_rate: float
    pool_sizes: List[int]  # ascending; the last is the pool compared with retry
    easy_pass_rate: float  # half the mixed workload
    hard_pass_rate: float  # the other half

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.BestOfNPoolModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.BestOfNPoolModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import BestOfNPoolModel

        return BestOfNPoolModel().solve(self.pass_rate, self.pool_sizes, self.easy_pass_rate, self.hard_pass_rate)


class VerifiedPoolScenario(BaseScenario):
    """A pool of ``pool_size`` candidates, first passing one returned, under a fallible check.

    Notes
    -----
    Equation: result = VerifiedPoolModel.solve(...) with the declared pool_size,
    valid_rate, false_reject_rate inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.VerifiedPoolModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond VerifiedPoolModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.VerifiedPoolModel.solve``. The solver contract records the equations
    and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SparseValidPoolFirstPass
    >>> round(float(case.result.valid_pass), 3)
    0.045"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool_size: int
    valid_rate: float  # v
    false_reject_rate: float  # eps_FR, P(FAIL | valid)
    false_accept_rate: float  # eps_FA, P(PASS | invalid)
    tokens_per_candidate: Quantity
    output_price: Quantity
    incident_loss: Quantity  # USD per defective selection

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.VerifiedPoolModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.VerifiedPoolModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import VerifiedPoolModel

        return VerifiedPoolModel().solve(
            self.pool_size,
            self.valid_rate,
            self.false_reject_rate,
            self.false_accept_rate,
            self.tokens_per_candidate,
            self.output_price,
            self.incident_loss,
        )


class ArgmaxNoiseScenario(BaseScenario):
    """Argmax over equally good candidates with Gaussian score noise; a rare loophole in the pool.

    Notes
    -----
    Equation: result = ArgmaxNoiseModel.solve(...) with the declared pool_sizes,
    small_pool, large_pool inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ArgmaxNoiseModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond ArgmaxNoiseModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ArgmaxNoiseModel.solve``. The solver contract records the equations
    and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ArgmaxScoreNoise
    >>> round(float(case.result.exploit_presence), 3)
    0.274"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool_sizes: List[int]
    small_pool: int
    large_pool: int
    exploit_rate: float  # fraction of single samples that exploit a check gap

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ArgmaxNoiseModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ArgmaxNoiseModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import ArgmaxNoiseModel

        return ArgmaxNoiseModel().solve(self.pool_sizes, self.large_pool, self.exploit_rate)


class ScopedRevisionScenario(BaseScenario):
    """A candidate plan of equal nodes, ``replaced_nodes`` of which a refuted premise invalidates.

    Notes
    -----
    Equation: result = ScopedRevisionModel.solve(...) with the declared total_nodes,
    replaced_nodes, tokens_per_node inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ScopedRevisionModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond ScopedRevisionModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ScopedRevisionModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ScopedPlanRevision
    >>> round(float(case.result.full_output), 3)
    1800.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    total_nodes: int
    replaced_nodes: int
    tokens_per_node: int

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ScopedRevisionModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ScopedRevisionModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import ScopedRevisionModel

        return ScopedRevisionModel().solve(self.total_nodes, self.replaced_nodes, self.tokens_per_node)


class PrunedSearchTreeScenario(BaseScenario):
    """A search tree of ``branching`` children per node whose step check rejects ``rejection_rate``.

    Notes
    -----
    Equation: result = PrunedSearchTreeModel.solve(...) with the declared branching,
    rejection_rate, depth inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.PrunedSearchTreeModel.solve``; accessing result evaluates that
    solver again.

    Excludes: Any extra mechanism beyond PrunedSearchTreeModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.PrunedSearchTreeModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.StepCheckedSearchTree
    >>> round(float(case.result.effective_branching), 3)
    1.6"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    branching: int
    rejection_rate: float
    depth: int
    wide_branching: Optional[int] = None  # an unpruned tree to size by its leaves
    wide_depth: Optional[int] = None

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.PrunedSearchTreeModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.PrunedSearchTreeModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ..solvers import PrunedSearchTreeModel

        return PrunedSearchTreeModel().solve(
            self.branching, self.rejection_rate, self.depth, self.wide_branching, self.wide_depth
        )


class SearchAccountsScenario(BaseScenario):
    """Flat best-of-N against beam search over the same task, kept as separate accounts.

    Notes
    -----
    Equation: result = SearchAccountsModel.solve(...) with the declared floors, segments,
    segment_tokens inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.SearchAccountsModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond SearchAccountsModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.SearchAccountsModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FlatVsBeamSearch
    >>> round(float(case.result.token_time.to('s').magnitude), 3)
    0.021"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    segments: int  # candidate length in equal segments
    segment_tokens: int
    prompt_tokens: int
    flat_candidates: int
    beam_width: int
    extensions: int  # extensions per retained prefix
    generation_slots: int  # concurrent streams at an unchanged rate
    test_slots: int  # concurrent final tests
    score_time: Quantity  # per full-prefix score
    test_time: Quantity  # per final test
    input_price: Quantity
    output_price: Quantity
    scorer_price: Quantity  # per scorer input token

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.SearchAccountsModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.SearchAccountsModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import SearchAccountsModel

        return SearchAccountsModel().solve(
            self.floors.result.memory_time,
            self.segments,
            self.segment_tokens,
            self.prompt_tokens,
            self.flat_candidates,
            self.beam_width,
            self.extensions,
            self.generation_slots,
            self.test_slots,
            self.score_time,
            self.test_time,
            self.input_price,
            self.output_price,
            self.scorer_price,
        )


class DeadlinePolicyScenario(BaseScenario):
    """One candidate, ``breadth`` in parallel, or feedback, for one task under one ``deadline``.

    Notes
    -----
    Equation: result = DeadlinePolicyModel.solve(...) with the declared floors, prompt,
    output inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.DeadlinePolicyModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond DeadlinePolicyModel; this record supplies
    configuration and does not collect performance measurements.

    The optional input/output/check ceilings and concurrency fields record
    scenario context. This result wrapper does not forward them to the solver
    and therefore does not enforce those additional limits.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.DeadlinePolicyModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DeadlineBoundRepairPolicies
    >>> round(float(case.result.token_time.to('s').magnitude), 3)
    0.021"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    floors: DecodeScenario
    prompt: int
    output: int  # generated tokens per candidate
    diagnostic: int  # failure-report tokens fed to a revision
    breadth: int
    max_rounds: int  # requested feedback rounds, before the deadline
    deadline: Quantity
    check_time: Quantity
    initial_pass: float
    revision_pass: float  # conditional, after a failed first candidate
    input_price: Quantity
    output_price: Quantity
    input_ceiling: Optional[int] = None  # per-decision ceilings the policies must respect
    output_ceiling: Optional[int] = None
    check_ceiling: Optional[int] = None
    concurrency: Optional[int] = None  # generator and check capacity

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.DeadlinePolicyModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.DeadlinePolicyModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import DeadlinePolicyModel

        return DeadlinePolicyModel().solve(
            self.floors.result.memory_time,
            self.prompt,
            self.output,
            self.diagnostic,
            self.breadth,
            self.max_rounds,
            self.deadline,
            self.check_time,
            self.initial_pass,
            self.revision_pass,
            self.input_price,
            self.output_price,
        )


class ContextBudgetScenario(BaseScenario):
    """A request budgeted by role, appending each tool result against selecting a fixed share.

    Notes
    -----
    Equation: result = ContextBudgetModel.solve(...) with the declared window,
    output_reserve, system_tokens inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ContextBudgetModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond ContextBudgetModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ContextBudgetModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairContextBudget
    >>> round(float(case.result.input_limit), 3)
    120000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: int
    output_reserve: int
    system_tokens: int
    tool_tokens: int
    state_tokens: int
    observation_per_turn: int  # tool result appended per turn
    selected_observation: int  # observations a selecting policy keeps on every call
    turns: int
    digest_tokens: int  # a compacted per-turn result

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ContextBudgetModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ContextBudgetModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import ContextBudgetModel

        return ContextBudgetModel().solve(
            self.window,
            self.output_reserve,
            self.system_tokens,
            self.tool_tokens,
            self.state_tokens,
            self.observation_per_turn,
            self.selected_observation,
            self.turns,
            self.digest_tokens,
        )


class AttentionDilutionScenario(BaseScenario):
    """One attention head with fixed logits: a target among background keys, optionally look-alikes.

    Notes
    -----
    Equation: result = AttentionDilutionModel.solve(...) with the declared target_logit,
    background_logit, short_keys inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.AttentionDilutionModel.solve``; accessing result evaluates that
    solver again.

    Excludes: Any extra mechanism beyond AttentionDilutionModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.AttentionDilutionModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ToyHeadAttentionDilution
    >>> round(float(case.result.short_weight), 3)
    0.968"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    target_logit: float
    background_logit: float
    short_keys: int  # total keys, target included
    long_keys: int
    unrelated_keys: Optional[int] = None  # competitors at the background logit
    lookalike_keys: Optional[int] = None  # competitors that share the evidence's identifiers
    lookalike_logit: Optional[float] = None

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.AttentionDilutionModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.AttentionDilutionModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ..solvers import AttentionDilutionModel

        return AttentionDilutionModel().solve(
            self.target_logit,
            self.background_logit,
            self.short_keys,
            self.long_keys,
            self.unrelated_keys,
            self.lookalike_keys,
            self.lookalike_logit,
        )


class PrefixLayoutBillScenario(BaseScenario):
    """A trajectory of root, trunk, and leaf requests, billed volatile-first and zoned.

    Notes
    -----
    Equation: result = PrefixLayoutBillModel.solve(...) with the declared turns,
    root_tokens, trunk_tokens inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.PrefixLayoutBillModel.solve``; accessing result evaluates that
    solver again.

    Excludes: Any extra mechanism beyond PrefixLayoutBillModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.PrefixLayoutBillModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairPrefixLayoutBill
    >>> round(float(case.result.request_tokens), 3)
    47000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    turns: int
    root_tokens: int
    trunk_tokens: int
    leaf_tokens: int
    trunk_rewrites: int  # calls after the first that rewrite the trunk
    input_price: Quantity  # USD per uncached input token
    cached_fraction: float  # reused tokens billed at this fraction of the price

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.PrefixLayoutBillModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.PrefixLayoutBillModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ..solvers import PrefixLayoutBillModel

        return PrefixLayoutBillModel().solve(
            self.turns,
            self.root_tokens,
            self.trunk_tokens,
            self.leaf_tokens,
            self.trunk_rewrites,
            self.input_price,
            self.cached_fraction,
        )


class CompactionCadenceScenario(BaseScenario):
    """Compaction between a trigger and a target watermark over a turn budget.

    Notes
    -----
    Equation: result = CompactionCadenceModel.solve(...) with the declared window,
    output_reserve, target inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.CompactionCadenceModel.solve``; accessing result evaluates that
    solver again.

    Excludes: Any extra mechanism beyond CompactionCadenceModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.CompactionCadenceModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairCompactionWatermarks
    >>> round(float(case.result.input_limit), 3)
    120000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: int
    output_reserve: int
    target: int  # request length a pass resets to
    trigger: int  # request length that fires a pass
    tokens_per_turn: int
    turns: int
    alternate_target: Optional[int] = None  # a narrower gap to compare

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.CompactionCadenceModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.CompactionCadenceModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ..solvers import CompactionCadenceModel

        return CompactionCadenceModel().solve(
            self.window,
            self.output_reserve,
            self.target,
            self.trigger,
            self.tokens_per_turn,
            self.turns,
            self.alternate_target,
        )


class ContextInvalidationScenario(BaseScenario):
    """A superseded excerpt right after the reused prefix: remove, replace, or append a notice.

    Notes
    -----
    Equation: result = ContextInvalidationModel.solve(...) with the declared
    request_tokens, prefix_tokens, superseded_tokens inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ContextInvalidationModel.solve``; accessing result evaluates that
    solver again.

    Excludes: Any extra mechanism beyond ContextInvalidationModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ContextInvalidationModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SupersededExcerptInvalidation
    >>> round(float(case.result.removed_total), 3)
    46150.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_tokens: int
    prefix_tokens: int  # reused prefix before the excerpt
    superseded_tokens: int
    marker_tokens: int  # in-place replacement marker
    notice_tokens: int  # appended notice, with its framing

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ContextInvalidationModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ContextInvalidationModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ..solvers import ContextInvalidationModel

        return ContextInvalidationModel().solve(
            self.request_tokens, self.prefix_tokens, self.superseded_tokens, self.marker_tokens, self.notice_tokens
        )


class ProtectedFieldCheckScenario(BaseScenario):
    """Constraints lost independently across compaction rounds, with and without a host check.

    Notes
    -----
    Equation: result = ProtectedFieldCheckModel.solve(...) with the declared constraints,
    loss_per_round, rounds inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ProtectedFieldCheckModel.solve``; accessing result evaluates that
    solver again.

    Excludes: Any extra mechanism beyond ProtectedFieldCheckModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ProtectedFieldCheckModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ProtectedFieldCheckPayback
    >>> round(float(case.result.exposures), 3)
    36.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    constraints: int
    loss_per_round: float
    rounds: int
    acceptance_if_intact: float
    attempt_cost: Quantity  # USD
    detection: float  # share of losses the check detects and repairs
    check_cost_per_round: Quantity  # USD
    survival_floor: float  # survival level whose first crossing is reported
    target_survival: Optional[float] = None  # survival to hold without a check

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ProtectedFieldCheckModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ProtectedFieldCheckModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ..solvers import ProtectedFieldCheckModel

        return ProtectedFieldCheckModel().solve(
            self.constraints,
            self.loss_per_round,
            self.rounds,
            self.acceptance_if_intact,
            self.attempt_cost,
            self.detection,
            self.check_cost_per_round,
            self.survival_floor,
            self.target_survival,
        )


class RetrievalPayloadScenario(BaseScenario):
    """Whole serialized records staged in what a call's window leaves for retrieval.

    Notes
    -----
    Equation: Result = mlsysim.solvers.RetrievalPayloadModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by RetrievalPayloadModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by RetrievalPayloadModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.RetrievalPayloadModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RetrievalRecordsInAllowance
    >>> scenario.result.allowance
    6144
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: int
    output_reserve: int
    other_input: int  # system prompt, tool schemas, history
    envelope: int  # response envelope, paid once
    body: int  # body tokens per record
    metadata: int  # provenance tokens per record
    metadata_scale: Optional[int] = None  # a metadata multiple to compare

    @property
    def result(self):
        """Return ``mlsysim.solvers.RetrievalPayloadModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.RetrievalPayloadModel.solve (scenario fields).
        """
        from ..solvers import RetrievalPayloadModel

        return RetrievalPayloadModel().solve(
            self.window,
            self.output_reserve,
            self.other_input,
            self.envelope,
            self.body,
            self.metadata,
            self.metadata_scale,
        )


class Bm25PairScenario(BaseScenario):
    """One query term scored by BM25 in a short and a long document of one corpus.

    Notes
    -----
    Equation: Result = mlsysim.solvers.Bm25PairModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by Bm25PairModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by Bm25PairModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.Bm25PairModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Bm25ShortAgainstLongFile
    >>> round(scenario.result.idf, 6)
    9.808187
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    documents: int
    matching_documents: int
    average_length: float  # analyzer terms
    short_length: float
    short_frequency: float
    long_length: float
    long_frequency: float
    k1: float
    b: float

    @property
    def result(self):
        """Return ``mlsysim.solvers.Bm25PairModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.Bm25PairModel.solve (scenario fields).
        """
        from ..solvers import Bm25PairModel

        return Bm25PairModel().solve(
            self.documents,
            self.matching_documents,
            self.average_length,
            self.short_length,
            self.short_frequency,
            self.long_length,
            self.long_frequency,
            self.k1,
            self.b,
        )


class ExactScanScenario(BaseScenario):
    """An exact vector scan on one host: bytes, intensity, and the bandwidth floor.

    Notes
    -----
    Equation: Result = mlsysim.solvers.ExactScanFloorModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by ExactScanFloorModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by ExactScanFloorModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ExactScanFloorModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ExactScanTenMillionVectors
    >>> round((scenario.result.footprint).to('B').magnitude, 6)
    15360000000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    host: HardwareNode
    vectors: int
    dimension: int
    bytes_per_value: Quantity
    searches: Optional[int] = None  # searches in one trajectory
    filtered_vectors: Optional[int] = None  # eligible set after a scope filter
    budget: Optional[Quantity] = None  # per-search latency budget

    @property
    def result(self):
        """Return ``mlsysim.solvers.ExactScanFloorModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ExactScanFloorModel.solve (scenario fields).
        """
        from ..solvers import ExactScanFloorModel

        return ExactScanFloorModel().solve(
            self.host,
            self.vectors,
            self.dimension,
            self.bytes_per_value,
            self.searches,
            self.filtered_vectors,
            self.budget,
        )


class AnnFootprintScenario(BaseScenario):
    """Resident bytes of a graph index and of product-quantization codes over one corpus.

    Notes
    -----
    Equation: Result = mlsysim.solvers.AnnFootprintModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by AnnFootprintModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by AnnFootprintModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.AnnFootprintModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.AnnIndexesTenMillionVectors
    >>> round((scenario.result.raw).to('B').magnitude, 6)
    15360000000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    vectors: int
    dimension: int
    bytes_per_value: Quantity
    links_per_node: int
    link_bytes: Quantity
    subvectors: int
    centroids: int
    code_bytes: Quantity
    table_entry_bytes: Quantity

    @property
    def result(self):
        """Return ``mlsysim.solvers.AnnFootprintModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.AnnFootprintModel.solve (scenario fields).
        """
        from ..solvers import AnnFootprintModel

        return AnnFootprintModel().solve(
            self.vectors,
            self.dimension,
            self.bytes_per_value,
            self.links_per_node,
            self.link_bytes,
            self.subvectors,
            self.centroids,
            self.code_bytes,
            self.table_entry_bytes,
        )


class GraphNeighborhoodScenario(BaseScenario):
    """A typed traversal's exact-tree neighborhood against a call's retrieval allowance.

    Notes
    -----
    Equation: Result = mlsysim.solvers.GraphNeighborhoodModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by GraphNeighborhoodModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by GraphNeighborhoodModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.GraphNeighborhoodModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TypedTraversalDepthThree
    >>> scenario.result.allowance
    6144
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: int
    output_reserve: int
    other_input: int
    children: int
    depth: int  # the seed is depth zero
    node_tokens: int  # serialized tokens per node, provenance included
    envelope_tokens: int  # edges and envelope, paid once

    @property
    def result(self):
        """Return ``mlsysim.solvers.GraphNeighborhoodModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.GraphNeighborhoodModel.solve (scenario fields).
        """
        from ..solvers import GraphNeighborhoodModel

        return GraphNeighborhoodModel().solve(
            self.window,
            self.output_reserve,
            self.other_input,
            self.children,
            self.depth,
            self.node_tokens,
            self.envelope_tokens,
        )


class IndexMaintenanceScenario(BaseScenario):
    """One edit's synchronous re-indexing against exclusion of the stale entries.

    Notes
    -----
    Equation: Result = mlsysim.solvers.IndexMaintenanceModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by IndexMaintenanceModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by IndexMaintenanceModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.IndexMaintenanceModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairEditIndexMaintenance
    >>> scenario.result.chunks
    16
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    files: int
    chunks_per_file: int
    parse_time: Quantity  # per file
    postings_time: Quantity  # per chunk
    local_batch_time: Quantity  # one local embedding batch
    batch: int  # chunks per embedding batch
    remote_request_time: Quantity  # one remote embedding request
    exclusion_write_time: Quantity  # per exclusion entry
    candidates: int  # candidates checked per query
    exclusion_check_time: Quantity  # per candidate

    @property
    def result(self):
        """Return ``mlsysim.solvers.IndexMaintenanceModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.IndexMaintenanceModel.solve (scenario fields).
        """
        from ..solvers import IndexMaintenanceModel

        return IndexMaintenanceModel().solve(
            self.files,
            self.chunks_per_file,
            self.parse_time,
            self.postings_time,
            self.local_batch_time,
            self.batch,
            self.remote_request_time,
            self.exclusion_write_time,
            self.candidates,
            self.exclusion_check_time,
        )


class StorageAmplificationScenario(BaseScenario):
    """Auxiliary bytes and re-index time of a lexical-plus-graph and a lexical-plus-dense store.

    Notes
    -----
    Equation: Result = mlsysim.solvers.StorageAmplificationModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by StorageAmplificationModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by StorageAmplificationModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.StorageAmplificationModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LexicalGraphAgainstDenseStore
    >>> round((scenario.result.raw).to('B').magnitude, 6)
    40000000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    files: int
    tokens_per_file: int
    bytes_per_file: Quantity
    postings_per_token: Quantity  # bytes of postings per token
    edges_per_file: int
    edge_bytes: Quantity
    passage_tokens: int  # tokens per dense chunk
    dimension: int
    bytes_per_value: Quantity
    links_per_node: int
    link_bytes: Quantity
    edited_files: int
    lexical_rate: float  # files per second
    parse_rate: float  # files per second
    embed_rate: float  # chunks per second

    @property
    def result(self):
        """Return ``mlsysim.solvers.StorageAmplificationModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.StorageAmplificationModel.solve (scenario fields).
        """
        from ..solvers import StorageAmplificationModel

        return StorageAmplificationModel().solve(
            self.files,
            self.tokens_per_file,
            self.bytes_per_file,
            self.postings_per_token,
            self.edges_per_file,
            self.edge_bytes,
            self.passage_tokens,
            self.dimension,
            self.bytes_per_value,
            self.links_per_node,
            self.link_bytes,
            self.edited_files,
            self.lexical_rate,
            self.parse_rate,
            self.embed_rate,
        )


class BlindRetryScenario(BaseScenario):
    """Mutating calls retried once after any timeout, with independent request and reply loss.

    Notes
    -----
    Equation: Calls=``workers*calls_per_worker``; duplicate and timeout expectations come from BlindRetryModel.

    Assumptions: Independent request/reply losses and one blind retry; cost properties treat the workload as daily.

    Excludes: Correlated loss, idempotency safeguards, and distributions of duplicate harm.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.BlindRetryModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.BlindRetryTwoPercentLoss
    >>> scenario.calls
    200000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_loss: float  # probability a request is lost
    reply_loss: float  # probability a reply is lost
    workers: int
    calls_per_worker: int  # mutating calls per worker per day
    # Optional (added 2026-10-20): what a duplicated change and a read-only
    # status lookup cost, to price blind retry against settlement by status.
    duplicate_cost: Optional[Quantity] = None  # USD per duplicated change
    status_lookup_cost: Optional[Quantity] = None  # USD per status query

    @property
    def calls(self) -> int:
        """Total mutating call count over the scenario's accounting period: workers times calls per worker.

        Equation: self.workers * self.calls_per_worker
        """
        return self.workers * self.calls_per_worker

    @property
    def result(self):
        """Return ``mlsysim.solvers.BlindRetryModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.BlindRetryModel.solve (scenario fields).
        """
        from ..solvers import BlindRetryModel

        return BlindRetryModel().solve(self.request_loss, self.reply_loss, self.calls)

    @property
    def lost_request_share(self) -> float:
        """Fraction of timeouts attributable to request loss under the solver's independent request/reply-loss model; undefined when no timeout is possible.

        Equation: self.request_loss / self.result.timeout_probability
        """
        return self.request_loss / self.result.timeout_probability

    @property
    def unapplied_closures(self) -> float:
        """Expected calls whose first request never arrived: call count times request-loss probability.

        Equation: self.calls * self.request_loss
        """
        return self.calls * self.request_loss

    @property
    def duplicate_cost_per_day(self) -> Quantity:
        """Expected duplicate-side-effect cost in USD for the declared daily workload; uses ``result.duplicates`` and supplied cost per duplicate.

        Equation: (self.result.duplicates * self.duplicate_cost).to('USD')
        """
        return (self.result.duplicates * self.duplicate_cost).to("USD")

    @property
    def lookup_cost_per_day(self) -> Quantity:
        """Status-lookup cost in USD when every modeled timeout triggers one lookup.

        Equation: (self.result.timeouts * self.status_lookup_cost).to('USD')
        """
        return (self.result.timeouts * self.status_lookup_cost).to("USD")


class ObservationCarryScenario(BaseScenario):
    """A raw tool output sized in windows, and its shaped view carried by later calls.

    Notes
    -----
    Equation: Result = mlsysim.solvers.ObservationCarryModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by ObservationCarryModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by ObservationCarryModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ObservationCarryModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TwentyFourMegabyteLogObservation
    >>> round((scenario.result.raw_tokens).to('token').magnitude, 6)
    8000000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    raw_bytes: Quantity
    bytes_per_token: Quantity  # assumed serialization ratio, byte/token
    window: int  # context window, tokens
    shaped_tokens: int  # staged observation, tokens
    carry_calls: int  # later calls that carry it
    input_price: Quantity  # USD per uncached input token
    cached_fraction: float  # cached input billed at this share of the price

    @property
    def result(self):
        """Return ``mlsysim.solvers.ObservationCarryModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ObservationCarryModel.solve (scenario fields).
        """
        from ..solvers import ObservationCarryModel

        return ObservationCarryModel().solve(
            self.raw_bytes,
            self.bytes_per_token,
            self.window,
            self.shaped_tokens,
            self.carry_calls,
            self.input_price,
            self.cached_fraction,
        )


class ModelPollingScenario(BaseScenario):
    """A model that checks a pending job at a fixed interval until it completes.

    Notes
    -----
    Equation: Result = mlsysim.solvers.ModelPollingModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by ModelPollingModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by ModelPollingModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ModelPollingModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ModelPollingEveryFiveSeconds
    >>> scenario.result.polls
    59
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    job_duration: Quantity
    poll_interval: Quantity
    context_tokens: int  # selected input per status call
    output_tokens: int  # generated tokens per status call
    cache_hit: float  # share of input billed as cached
    input_price: Quantity  # USD per uncached input token
    cached_fraction: float  # cached input billed at this share of the price
    output_price: Quantity  # USD per output token

    @property
    def result(self):
        """Return ``mlsysim.solvers.ModelPollingModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ModelPollingModel.solve (scenario fields).
        """
        from ..solvers import ModelPollingModel

        return ModelPollingModel().solve(
            self.job_duration,
            self.poll_interval,
            self.context_tokens,
            self.output_tokens,
            self.cache_hit,
            self.input_price,
            self.cached_fraction,
            self.output_price,
        )


class ToolCatalogSelectionScenario(BaseScenario):
    """A registered tool catalog staged whole or as a selected subset with overhead.

    Notes
    -----
    Equation: Result = mlsysim.solvers.ToolCatalogSelectionModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by ToolCatalogSelectionModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by ToolCatalogSelectionModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ToolCatalogSelectionModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FiftyToolCatalogSixSelected
    >>> scenario.result.full_tokens
    15000
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    registered: int  # tools registered
    selected: int  # tools staged by selection
    descriptor_tokens: int  # per descriptor, final serialization
    selection_overhead: int  # selection interface and metadata, tokens

    @property
    def result(self):
        """Return ``mlsysim.solvers.ToolCatalogSelectionModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ToolCatalogSelectionModel.solve (scenario fields).
        """
        from ..solvers import ToolCatalogSelectionModel

        return ToolCatalogSelectionModel().solve(
            self.registered, self.selected, self.descriptor_tokens, self.selection_overhead
        )


class ToolkitGranularityScenario(BaseScenario):
    """Structured tools against one shell tool, judged by cost per accepted task.

    Notes
    -----
    Equation: Result = mlsysim.solvers.ToolkitGranularityModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by ToolkitGranularityModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by ToolkitGranularityModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ToolkitGranularityModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ShellAgainstTwelveStructuredTools
    >>> scenario.result.structured_descriptor_tokens
    4200
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tools: int  # structured tools
    tool_tokens: int  # per structured descriptor
    shell_tokens: int  # the one shell descriptor
    calls: int  # model calls per trajectory
    input_price: Quantity  # USD per uncached input token
    structured_attempt: Quantity  # USD per structured attempt
    structured_acceptance: float
    structured_change: float  # probability of an unconfined change
    shell_acceptance: float
    shell_change: float
    change_cost: Quantity  # USD to detect and repair one unconfined change

    @property
    def result(self):
        """Return ``mlsysim.solvers.ToolkitGranularityModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ToolkitGranularityModel.solve (scenario fields).
        """
        from ..solvers import ToolkitGranularityModel

        return ToolkitGranularityModel().solve(
            self.tools,
            self.tool_tokens,
            self.shell_tokens,
            self.calls,
            self.input_price,
            self.structured_attempt,
            self.structured_acceptance,
            self.structured_change,
            self.shell_acceptance,
            self.shell_change,
            self.change_cost,
        )


class SyscallTaxScenario(BaseScenario):
    """``calls`` ordinary system calls priced by the terms each isolation envelope pays.

    Notes
    -----
    Equation: Result = mlsysim.solvers.SyscallTaxModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by SyscallTaxModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by SyscallTaxModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.SyscallTaxModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SyscallTaxPerMillionCalls
    >>> round((scenario.result.container_call).to('μs').magnitude, 6)
    0.15
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    calls: int
    trap: Quantity  # trap into a kernel
    filter: Quantity  # in-kernel filter (shared-kernel container)
    emulate: Quantity  # user-space kernel emulation
    switch: Quantity  # switch to the emulator

    @property
    def result(self):
        """Return ``mlsysim.solvers.SyscallTaxModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.SyscallTaxModel.solve (scenario fields).
        """
        from ..solvers import SyscallTaxModel

        return SyscallTaxModel().solve(self.calls, self.trap, self.filter, self.emulate, self.switch)


class EnvelopeAmortizationScenario(BaseScenario):
    """``calls`` sequential calls of ``work_per_call``, with envelope setup paid per call or once.

    Notes
    -----
    Equation: Result = mlsysim.solvers.EnvelopeAmortizationModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by EnvelopeAmortizationModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by EnvelopeAmortizationModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.EnvelopeAmortizationModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ParserCallsEnvelopeSetup
    >>> round((scenario.result.useful_work).to('ms').magnitude, 6)
    300
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    calls: int
    setup: Quantity  # envelope preparation through readiness for useful work
    work_per_call: Quantity
    overhead_cap: float  # largest acceptable setup share of a fresh call

    @property
    def result(self):
        """Return ``mlsysim.solvers.EnvelopeAmortizationModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.EnvelopeAmortizationModel.solve (scenario fields).
        """
        from ..solvers import EnvelopeAmortizationModel

        return EnvelopeAmortizationModel().solve(self.calls, self.setup, self.work_per_call, self.overhead_cap)


class WorkspaceProvisioningScenario(BaseScenario):
    """``workspaces`` on one pinned base, provisioned by copy or copy-on-write, then a first write.

    Notes
    -----
    Equation: Result = mlsysim.solvers.WorkspaceProvisioningModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by WorkspaceProvisioningModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by WorkspaceProvisioningModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.WorkspaceProvisioningModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CopyOnWriteRepairWorkspaces
    >>> round((scenario.result.full_copy).to('GB').magnitude, 6)
    76.8
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workspaces: int
    base_image: Quantity  # repository, toolchain, and dependencies
    written: Quantity  # edits and build outputs per workspace
    bandwidth: Quantity  # one drive's sequential rate
    fixture: Quantity  # a large shared file each task modifies
    block: Quantity  # block-granular copying unit
    modified: Quantity  # bytes each task changes in the fixture

    @property
    def result(self):
        """Return ``mlsysim.solvers.WorkspaceProvisioningModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.WorkspaceProvisioningModel.solve (scenario fields).
        """
        from ..solvers import WorkspaceProvisioningModel

        return WorkspaceProvisioningModel().solve(
            self.workspaces, self.base_image, self.written, self.bandwidth, self.fixture, self.block, self.modified
        )


class SandboxPoolReserveScenario(BaseScenario):
    """A warm sandbox pool under steady arrivals: active leases, a standby reserve, and memory.

    Notes
    -----
    Equation: Result = mlsysim.solvers.SandboxPoolReserveModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by SandboxPoolReserveModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by SandboxPoolReserveModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.SandboxPoolReserveModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CodingSandboxPoolTwoPerSecond
    >>> round(scenario.result.active, 6)
    600.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    arrival_rate: Quantity
    hold: Quantity  # lease held for the whole trajectory
    restore: Quantity  # one clean instance from its snapshot
    guest_memory: Quantity  # full private guest image
    dirty: Quantity  # private pages written per active lease
    standby: Quantity  # private state per standby instance
    target_miss: float  # request-level miss target
    initial_reserve: int = 1  # candidate evaluated before the minimum
    slow_factor: float = 10  # a cold boot this many times slower than restore

    @property
    def result(self):
        """Return ``mlsysim.solvers.SandboxPoolReserveModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.SandboxPoolReserveModel.solve (scenario fields).
        """
        from ..solvers import SandboxPoolReserveModel

        return SandboxPoolReserveModel().solve(
            self.arrival_rate,
            self.hold,
            self.restore,
            self.guest_memory,
            self.dirty,
            self.standby,
            self.target_miss,
            self.initial_reserve,
            self.slow_factor,
        )


class ApprovalEscrowScenario(BaseScenario):
    """Approval escrow for a long-running workload (held resources) and an urgent one (time).

    Notes
    -----
    Equation: The ``result`` property composes ``mlsysim.solvers.ApprovalEscrowModel.solve``
    with the stored fields and any derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.ApprovalEscrowModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.ApprovalEscrowModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairRemediationApprovalEscrow
    >>> round(float(scenario.result.held), 4)
    12.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    repair_rate: Quantity  # review requests per unit time
    repair_wait: Quantity  # mean time to a review decision
    lease: Quantity  # workspace each held trajectory keeps
    remediation_rate: Quantity
    diagnosis: Quantity  # before the proposal
    remediation_wait: Quantity  # mean time to an on-call decision
    apply_verify: Quantity  # after approval

    @property
    def result(self):
        """Return the modeled ApprovalEscrow result from
        ``mlsysim.solvers.ApprovalEscrowModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import ApprovalEscrowModel

        return ApprovalEscrowModel().solve(
            self.repair_rate,
            self.repair_wait,
            self.lease,
            self.remediation_rate,
            self.diagnosis,
            self.remediation_wait,
            self.apply_verify,
        )


class DeficitRoundRobinScenario(BaseScenario):
    """Backlogged request classes sharing model calls under weighted deficit round-robin.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.DeficitRoundRobinModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.DeficitRoundRobinModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.DeficitRoundRobinModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.InteractiveMaintenanceDrr
    >>> round(float(scenario.result.credit_per_round['interactive']), 4)
    24000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    quantum: float  # charge units per unit weight per round
    classes: Dict[str, Dict[str, int]]  # name -> weight, input, output (allowance)
    simulated_rounds: int = 10_000

    @property
    def result(self):
        """Return the modeled DeficitRoundRobin result from
        ``mlsysim.solvers.DeficitRoundRobinModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import DeficitRoundRobinModel

        return DeficitRoundRobinModel().solve(self.quantum, self.classes, self.simulated_rounds)


class BoundedLiabilityAdmissionScenario(BaseScenario):
    """The next model call of a trajectory, tested against its dollar allowance.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.BoundedLiabilityAdmissionModel.solve`` with the stored fields and any
    derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.BoundedLiabilityAdmissionModel.solve``. The named scenario supplies a
    fixed configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.BoundedLiabilityAdmissionModel.solve`` with the stored fields and any
    derived properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairNextCallAdmission
    >>> round(float(scenario.result.input_liability), 4)
    0.15"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    allowance: Quantity  # USD
    settled: Quantity  # actual spend so far
    reserved: Quantity  # outstanding reservations of unsettled work
    finalization_reserve: Quantity  # protected for checks, report, and cleanup
    input_tokens: int
    output_allowance: int
    input_price: Quantity  # USD per token (undiscounted)
    output_price: Quantity
    probe: Quantity  # an unsettled amount for expressing sensitivity

    @property
    def result(self):
        """Return the modeled BoundedLiabilityAdmission result from
        ``mlsysim.solvers.BoundedLiabilityAdmissionModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import BoundedLiabilityAdmissionModel

        return BoundedLiabilityAdmissionModel().solve(
            self.allowance,
            self.settled,
            self.reserved,
            self.finalization_reserve,
            self.input_tokens,
            self.output_allowance,
            self.input_price,
            self.output_price,
            self.probe,
        )


class TrajectoryBudgetMeterScenario(BaseScenario):
    """An append-only trajectory metered against turns, tokens, time, and dollars.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.TrajectoryBudgetMeterModel.solve`` with the stored fields and any
    derived properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.TrajectoryBudgetMeterModel.solve``. The named scenario supplies a
    fixed configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.TrajectoryBudgetMeterModel.solve`` with the stored fields and any
    derived properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairTrajectoryBudgetMeter
    >>> round(float(scenario.result.input_tokens), 4)
    892500.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    trajectory: TrajectoryResendScenario
    input_price: Quantity
    cached_fraction: float  # cached input billed at this share of the input price
    output_price: Quantity
    turn_allowance: int
    token_allowance: int
    time_allowance: Quantity
    dollar_allowance: Quantity
    elapsed: Quantity  # wall clock at the readout

    @property
    def result(self):
        """Return the modeled TrajectoryBudgetMeter result from
        ``mlsysim.solvers.TrajectoryBudgetMeterModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import TrajectoryBudgetMeterModel

        t = self.trajectory.result
        allowances = {
            "turns": self.turn_allowance,
            "tokens": self.token_allowance,
            "time": float(self.time_allowance.to("second").magnitude),
            "dollars": float(self.dollar_allowance.to("USD").magnitude),
        }
        return TrajectoryBudgetMeterModel().solve(
            t.turns,
            t.resent_tokens,
            t.reused_prefix_tokens,
            t.output_tokens,
            self.input_price,
            self.input_price * self.cached_fraction,
            self.output_price,
            allowances,
            float(self.elapsed.to("second").magnitude),
        )


class InjectionExposureScenario(BaseScenario):
    """Workloads reading ``reads`` untrusted items each, a fraction ``payload_fraction``
    carrying
    a payload, with and without a screen that misses ``miss_rate`` of payloads.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.InjectionExposureModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.InjectionExposureModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.InjectionExposureModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.InjectionExposureByReadVolume
    >>> round(float(scenario.result.exposure['repair']), 4)
    0.1821"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    payload_fraction: float
    reads: Dict[str, int]  # untrusted items read per task, by workload
    miss_rate: float  # fraction of payloads the screen misses
    cohort: int  # tasks in the per-cohort count

    @property
    def result(self):
        """Return the modeled InjectionExposure result from
        ``mlsysim.solvers.InjectionExposureModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import InjectionExposureModel

        return InjectionExposureModel().solve(self.payload_fraction, self.reads, self.miss_rate, self.cohort)


class PolicyComparisonScenario(BaseScenario):
    """Two policies, each run on ``attacked`` attacked and ``benign`` separate benign episodes.

    Each policy's counts: ``confirmed`` and ``unresolved`` harm and ``accepted_attacked``
    among
    attacked episodes; ``accepted_benign`` and ``false_block`` among benign ones.

    Notes
    -----
    Equation: The ``result`` property composes
    ``mlsysim.solvers.PolicyComparisonModel.solve`` with the stored fields and any derived
    properties described below.

    Assumptions: Inputs use the units and validity ranges of
    ``mlsysim.solvers.PolicyComparisonModel.solve``. The named scenario supplies a fixed
    configuration rather than observations of a new run.

    Excludes: Additional runtime effects or statistical guarantees beyond the cited solver
    contracts; no measurement is performed by this wrapper.

    References
    ----------
    Derivation: The ``result`` property composes
    ``mlsysim.solvers.PolicyComparisonModel.solve`` with the stored fields and any derived
    properties described below. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FixtureReleasePolicyComparison
    >>> round(float(scenario.result.baseline_lower), 4)
    0.2"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    attacked: int
    benign: int
    baseline: Dict[str, int]
    candidate: Dict[str, int]
    attempt_cost: Optional[float] = None  # USD per attempt, to price cost per accepted task

    @property
    def result(self):
        """Return the modeled PolicyComparison result from
        ``mlsysim.solvers.PolicyComparisonModel.solve``.

        Equation: the scenario supplies its named fields to that solver. Returned
        quantity units, validity conditions, and omissions follow its calculation
        contract; this property adds no empirical calibration."""
        from ..solvers import PolicyComparisonModel

        return PolicyComparisonModel().solve(
            self.attacked, self.benign, self.baseline, self.candidate, self.attempt_cost
        )


class SpanCriticalPathScenario(BaseScenario):
    """One trial's observed spans on a shared clock, the spans each waits for, and one intervention.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Observed spans share a clock and dependencies describe the modeled intervention schedule.

    Excludes: Causal discovery, contention induced by rescheduling, and missing spans.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``SpanCriticalPathModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairTrialSpanTimeline
    >>> value = case.result
    >>> print(f"{value.critical_path:.3f~P}")
    11.200 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    intervals: Dict[str, Any]  # span -> (start, end) Quantity pair after segment start
    depends_on: Dict[str, List[str]]
    intervention: Dict[str, Any] = {}  # span -> new duration (Quantity); others keep theirs

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.SpanCriticalPathModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = SpanCriticalPathModel().solve(...)`` using the fields shown below."""
        from ..solvers import SpanCriticalPathModel

        return SpanCriticalPathModel().solve(self.intervals, self.depends_on, self.intervention)

    def duration(self, span: str) -> Quantity:
        """Duration of the named observed span in seconds, end minus start.

        Equation:
            start, end = self.intervals[span]
            return (end - start).to('s')"""
        start, end = self.intervals[span]
        return (end - start).to("s")


class HeadSamplingMissScenario(BaseScenario):
    """Head sampling keeps each trial with ``retention_probability``, decided before its outcome;
    a failure mode that occurs ``occurrences`` times leaves no detailed trace with ``miss``.

    Notes
    -----
    Equation: ``miss`` is the worked quantity; each property below states its equation and units.

    Assumptions: Each failure occurrence is independently retained with the same head-sampling probability.

    Excludes: Outcome-dependent sampling and correlated retention decisions.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_independent_success``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RareFailureUnderHeadSampling
    >>> value = case.miss
    >>> round(value, 3)
    0.59"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    retention_probability: float
    occurrences: int

    @property
    def miss(self) -> float:
        """Probability that independent head sampling retains none of the failure occurrences.

        ``(1 - p)^n``: no occurrence retained, the complement of ``calc_independent_success``.

        Equation:
            return 1 - calc_independent_success(self.retention_probability, self.occurrences)"""
        from ...physics.agents import calc_independent_success

        return 1 - calc_independent_success(self.retention_probability, self.occurrences)


class TraceRetentionScenario(BaseScenario):
    """One cohort period of trials in disjoint verdict classes, each with a ``trace_size`` body,
    retained over ``window`` by keeping everything, head sampling, or verdict-aware selection.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: Outcome cohorts, trace sizes, and retention probabilities are supplied for a fixed period.

    Excludes: Compression variation, correlation between sampling decisions, and downstream storage overhead.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``TraceRetentionModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.VerdictAwareTraceRetention
    >>> value = case.result
    >>> print(f"{value.tail_volume:.3f~P}")
    23.184 GB"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    failed: int  # failed a criterion or violated a policy
    unresolved: int  # pending or unknown verdict
    ordinary: int  # ordinary accepted
    cohort_period: Quantity  # the period the counts cover
    trace_size: Quantity  # one permitted trace body
    window: Quantity
    head_probability: float
    control_probability: float  # control sample of ordinary accepted trials
    control_step: float = 0.01  # one percentage point of control sampling

    @property
    def trials(self) -> int:
        """Total trial count across failed, unresolved, and ordinary outcomes.

        Equation:
            return self.failed + self.unresolved + self.ordinary"""
        return self.failed + self.unresolved + self.ordinary

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.TraceRetentionModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = TraceRetentionModel().solve(...)`` using the fields shown below."""
        from ..solvers import TraceRetentionModel

        return TraceRetentionModel().solve(
            self.failed,
            self.unresolved,
            self.ordinary,
            self.cohort_period,
            self.trace_size,
            self.window,
            self.head_probability,
            self.control_probability,
            self.control_step,
        )


class ControlledBranchScenario(BaseScenario):
    """Fresh continuations from one saved prefix under several branches, counted by outcome;
    the Wilson bound a clean branch leaves and the clean runs a ``target_bound`` needs.

    Notes
    -----
    Equation: ``clean_runs_for_target`` is the worked quantity; each property below states its equation and units.

    Assumptions: Each branch has a nonempty binomial outcome sample and the confidence target is supplied.

    Excludes: Dependence between trials, multiple-comparison corrections, and evidence for unobserved branches.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_wilson_clean_runs``, ``calc_wilson_interval``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.InterfaceRepairBranches
    >>> value = case.clean_runs_for_target
    >>> value
    73"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    outcomes: Dict[str, Dict[str, int]]  # branch -> outcome class -> runs
    accepted_class: str = "accepted"
    target_bound: float  # failure-rate upper limit to demonstrate
    confidence: float = 0.95  # two-sided

    @property
    def z(self) -> float:
        """Standard-normal quantile for the supplied two-sided confidence level.

        Equation:
            return NormalDist().inv_cdf(1 - (1 - self.confidence) / 2)"""
        from statistics import NormalDist

        return NormalDist().inv_cdf(1 - (1 - self.confidence) / 2)

    def runs(self, branch: str) -> int:
        """Total observed runs in the named branch across all outcome classes.

        Equation:
            return sum(self.outcomes[branch].values())"""
        return sum(self.outcomes[branch].values())

    def accepted(self, branch: str) -> int:
        """Accepted-run count for the named branch, zero when the accepted class is absent.

        Equation:
            return self.outcomes[branch].get(self.accepted_class, 0)"""
        return self.outcomes[branch].get(self.accepted_class, 0)

    def acceptance_interval(self, branch: str):
        """Wilson interval for the named branch's acceptance probability.

        Wilson interval on the branch's acceptance rate.

        Equation:
            return calc_wilson_interval(self.accepted(branch), self.runs(branch), self.z)"""
        from ...physics.statistics import calc_wilson_interval

        return calc_wilson_interval(self.accepted(branch), self.runs(branch), self.z)

    def failure_upper(self, branch: str) -> float:
        """Wilson upper bound for the named branch's non-acceptance probability.

        Wilson upper limit on the branch's non-acceptance rate.

        Equation:
            n = self.runs(branch)
            return calc_wilson_interval(n - self.accepted(branch), n, self.z)[1]"""
        from ...physics.statistics import calc_wilson_interval

        n = self.runs(branch)
        return calc_wilson_interval(n - self.accepted(branch), n, self.z)[1]

    def clean_failure_upper(self, runs: int) -> float:
        """Wilson failure-probability upper bound after the supplied number of failure-free runs.

        Wilson upper limit on the failure rate after ``runs`` clean runs.

        Equation:
            return calc_wilson_interval(0, runs, self.z)[1]"""
        from ...physics.statistics import calc_wilson_interval

        return calc_wilson_interval(0, runs, self.z)[1]

    @property
    def clean_runs_for_target(self) -> int:
        """Failure-free run count needed for the target Wilson upper bound.

        Equation:
            return calc_wilson_clean_runs(self.target_bound, self.z)"""
        from ...physics.statistics import calc_wilson_clean_runs

        return calc_wilson_clean_runs(self.target_bound, self.z)


class ToolDriftScenario(BaseScenario):
    """Baseline and candidate distributions of tool calls over the same tools, compared by
    the Jensen-Shannon divergence (natural logarithms).

    Notes
    -----
    Equation: ``divergence`` is the worked quantity; each property below states its equation and units.

    Assumptions: Both probability mappings use the same tool keys; natural logarithms measure divergence in nats.

    Excludes: A significance test or handling candidate-only keys absent from the baseline mapping.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_jensen_shannon_divergence``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TestLintToolDrift
    >>> value = case.divergence
    >>> round(value, 3)
    0.021"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline: Dict[str, float]
    candidate: Dict[str, float]

    @property
    def divergence(self) -> float:
        """Jensen-Shannon divergence in nats over the baseline tool keys.

        Equation:
            keys = list(self.baseline)
            return calc_jensen_shannon_divergence([self.baseline[k] for k in keys], [self.candidate[k] for k in keys])"""
        from ...physics.statistics import calc_jensen_shannon_divergence

        keys = list(self.baseline)
        return calc_jensen_shannon_divergence([self.baseline[k] for k in keys], [self.candidate[k] for k in keys])

    @property
    def maximum(self) -> float:
        """Maximum binary-mixture Jensen-Shannon divergence, ln(2) nats.

        ``ln 2``, reached by two distributions with no tool in common.

        Equation:
            return _math.log(2)"""
        return _math.log(2)

    @property
    def share_of_maximum(self) -> float:
        """Dimensionless observed divergence divided by its theoretical maximum.

        Equation:
            return self.divergence / self.maximum"""
        return self.divergence / self.maximum


class SequentialCanaryScenario(BaseScenario):
    """A canary monitor: Wald's SPRT between acceptance ``p_base`` and a specified degraded
    ``p_degraded`` at targets ``alpha`` (false hold) and ``beta`` (missed degradation), on
    ``fraction`` of eligible tasks arriving at ``arrival_rate``, labels ``label_delay`` late.

    Notes
    -----
    Equation: ``result`` evaluates the configured solver; its contract owns the governing equations.

    Assumptions: The shared SPRT model compares fixed baseline/degraded probabilities at the supplied sample rate.

    Excludes: Dependence, drifting alternatives, and a proof that target error rates hold for operational data.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``SequentialCanaryModel``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AcceptanceCanaryMonitor
    >>> value = case.result
    >>> print(f"{value.decision_time:.3f~P}")
    15.579 h"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    p_base: float
    p_degraded: float
    alpha: float
    beta: float
    arrival_rate: Quantity  # eligible tasks per hour
    fraction: float
    label_delay: Quantity

    @property
    def result(self):
        """Evaluate the configured inputs with ``mlsysim.solvers.SequentialCanaryModel.solve``.

        The returned typed result retains the solver's documented units, assumptions,
        and feasibility semantics; this property adds no measurement or safety claim.
        Equation: ``result = SequentialCanaryModel().solve(...)`` using the fields shown below."""
        from ..solvers import SequentialCanaryModel

        return SequentialCanaryModel().solve(
            self.p_base, self.p_degraded, self.alpha, self.beta, self.arrival_rate, self.fraction, self.label_delay
        )


class PartitionScreenScenario(BaseScenario):
    """A fixed partition screened over worker counts: (1 - p) + p / N + c (N - 1) of the serial time.

    The best count is the integer that minimizes elapsed time under the ceiling
    ``max_workers``; no count beats 1 / (1 - p), and the continuous optimum sits
    at sqrt(p / c).

    Notes
    -----
    Equation: The calculation members below define ``elapsed``, ``best_workers``, ``best_time``, ``best_speedup``, ``speedup_cap``, ``continuous_best``.

    Assumptions: The parallel share divides evenly across workers and each extra worker adds the same coordination fraction.

    Excludes: Overhead changing with topology, data skew, or a nonlinear worker count.

    References
    ----------
    Derivation: A fixed partition screened over worker counts: (1 - p) + p / N + c (N - 1) of the serial time. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairPartitionScreen
    >>> value = case.best_workers
    >>> print(value)
    6"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    parallel_fraction: float
    overhead_per_extra_worker: float
    max_workers: int

    @property
    def elapsed(self) -> Dict[int, float]:
        """Normalized elapsed time by integer worker count via ``calc_partition_elapsed``.

        Elapsed time over the serial reference, for each count from one to the ceiling.

        Calculation: ``mlsysim.physics.coordination.calc_partition_elapsed``."""
        from ...physics.coordination import calc_partition_elapsed

        return {
            n: calc_partition_elapsed(n, self.parallel_fraction, self.overhead_per_extra_worker)
            for n in range(1, self.max_workers + 1)
        }

    @property
    def best_workers(self) -> int:
        """Worker count minimizing normalized elapsed time within the configured search ceiling.

        Equation: ``min(t, key=t.get)``."""
        t = self.elapsed
        return min(t, key=t.get)

    @property
    def best_time(self) -> float:
        """Smallest normalized elapsed time in the enumerated search.

        Equation: ``elapsed[best_workers]``."""
        return self.elapsed[self.best_workers]

    @property
    def best_speedup(self) -> float:
        """Dimensionless serial-to-best-parallel speedup, reciprocal normalized elapsed time.

        Equation: ``1 / best_time``."""
        return 1 / self.best_time

    @property
    def speedup_cap(self) -> float:
        """Amdahl ceiling with coordination removed, one over the serial fraction.

        Amdahl's ceiling with free coordination: 1 / (1 - p).

        Equation: ``1 / (1 - parallel_fraction)``."""
        return 1 / (1 - self.parallel_fraction)

    @property
    def continuous_best(self) -> float:
        """Unconstrained continuous optimum sqrt(parallel fraction / per-worker overhead); not an integer search result.

        sqrt(p / c), where the derivative of the elapsed time vanishes.

        Equation: ``_math.sqrt(parallel_fraction / overhead_per_extra_worker)``."""
        return _math.sqrt(self.parallel_fraction / self.overhead_per_extra_worker)


class ScalabilityLawScenario(BaseScenario):
    """Relative capacity N / (1 + sigma (N - 1) + kappa N (N - 1)) over integer worker counts.

    Notes
    -----
    Equation: The calculation members below define ``capacity``, ``best_workers``, ``peak_capacity``, ``continuous_peak``, ``retrograde_threshold``, ``first_slower``.

    Assumptions: Contention and coherency coefficients define the stated USL capacity curve over positive worker counts.

    Excludes: Fitting coefficients to measurements or extrapolation guarantees outside the selected range.

    References
    ----------
    Derivation: Relative capacity N / (1 + sigma (N - 1) + kappa N (N - 1)) over integer worker counts. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairPartitionScalability
    >>> value = case.best_workers
    >>> print(value)
    6"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    contention: float  # sigma
    coherency: float  # kappa
    search_limit: int

    @property
    def capacity(self) -> Dict[int, float]:
        """Relative capacity by integer worker count via ``calc_universal_scalability_law``.

        Calculation: ``mlsysim.physics.performance.calc_universal_scalability_law``."""
        from ...physics.performance import calc_universal_scalability_law

        return {
            n: calc_universal_scalability_law(n, self.contention, self.coherency)
            for n in range(1, self.search_limit + 1)
        }

    @property
    def best_workers(self) -> int:
        """Integer count maximizing relative capacity within the search limit.

        Equation: ``max(c, key=c.get)``."""
        c = self.capacity
        return max(c, key=c.get)

    @property
    def peak_capacity(self) -> float:
        """Largest dimensionless relative capacity among enumerated worker counts.

        Equation: ``capacity[best_workers]``."""
        return self.capacity[self.best_workers]

    @property
    def continuous_peak(self) -> float:
        """Continuous optimum via ``calc_usl_optimal_concurrency``, distinct from integer search.

        N* = sqrt((1 - sigma) / kappa).

        Calculation: ``mlsysim.physics.performance.calc_usl_optimal_concurrency``."""
        from ...physics.performance import calc_usl_optimal_concurrency

        return calc_usl_optimal_concurrency(self.contention, self.coherency)

    @property
    def retrograde_threshold(self) -> float:
        """Worker threshold above which USL capacity falls below the one-worker baseline.

        (1 - sigma) / kappa: past it, capacity falls below one worker's.

        Equation: ``(1 - contention) / coherency``."""
        return (1 - self.contention) / self.coherency

    @property
    def first_slower(self) -> Optional[int]:
        """First enumerated count above one with capacity below baseline, or None if absent.

        The smallest count above one whose capacity is below one worker's.

        Equation: ``next((n for n, c in capacity.items() if n > 1 and c < 1), None)``."""
        return next((n for n, c in self.capacity.items() if n > 1 and c < 1), None)


class DelegationTokenBreakEvenScenario(BaseScenario):
    """Delegation at a token multiple m per attempt: cost per accepted task scales by m p_single / p_multi.

    With cost proportional to tokens, delegation breaks even only if it multiplies
    acceptance by m, which is possible only while single-agent acceptance is at
    most 1 / m.

    Notes
    -----
    Equation: The calculation members below define ``token_multiple``, ``max_single_acceptance``, ``acceptance_multiplier``, ``cost_ratio_if_gain_is_acceptance``, ``required_multi_acceptance``.

    Assumptions: Cost is proportional to tokens; the score-gain interpretation as acceptance improvement is explicitly hypothetical.

    Excludes: Equating benchmark score and acceptance probability without independent evidence.

    References
    ----------
    Derivation: Delegation at a token multiple m per attempt: cost per accepted task scales by m p_single / p_multi. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResearchSystemTokenBreakEven
    >>> value = case.cost_ratio_if_gain_is_acceptance
    >>> print(f"{value:.3g}")
    1.97"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    agent_tokens_vs_chat: float  # single agent's tokens over a chat answer's
    multi_agent_tokens_vs_chat: float  # delegated system's tokens over a chat answer's
    reported_gain: float  # a reported relative score gain, read as acceptance
    single_acceptance: Optional[float] = None  # a stated single-agent acceptance to test

    @property
    def token_multiple(self) -> float:
        """Delegated-to-single-agent token volume ratio, dimensionless.

        Equation: ``multi_agent_tokens_vs_chat / agent_tokens_vs_chat``."""
        return self.multi_agent_tokens_vs_chat / self.agent_tokens_vs_chat

    @property
    def max_single_acceptance(self) -> float:
        """Largest single-agent success probability permitting delegation break-even, reciprocal token multiple.

        Largest single-agent acceptance at which break-even is possible: 1 / m.

        Equation: ``1 / token_multiple``."""
        return 1 / self.token_multiple

    @property
    def acceptance_multiplier(self) -> float:
        """Hypothetical acceptance multiplier formed as one plus the reported relative gain.

        Equation: ``1 + reported_gain``."""
        return 1 + self.reported_gain

    @property
    def cost_ratio_if_gain_is_acceptance(self) -> float:
        """Delegated-to-single cost per accepted task if the reported score gain is an acceptance gain.

        Delegated over single cost per accepted task if the gain multiplied acceptance.

        Equation: ``token_multiple / acceptance_multiplier``."""
        return self.token_multiple / self.acceptance_multiplier

    @property
    def required_multi_acceptance(self) -> Optional[float]:
        """Delegated acceptance needed for break-even, token multiple times single acceptance; None if unspecified.

        Delegated acceptance that breaks even against ``single_acceptance``: m p_single.

        Equation: ``None if single_acceptance is None else token_multiple * single_acceptance``."""
        return None if self.single_acceptance is None else self.token_multiple * self.single_acceptance


class DependencyScheduleScenario(BaseScenario):
    """A dependency graph on interchangeable slots: aggregate work, the critical path, and both bounds.

    The schedule starts each node when its last predecessor finishes. No schedule
    on ``slots`` slots ends before ``max(W / slots, L)``.

    Notes
    -----
    Equation: The calculation members below define ``schedule``, ``start``, ``finish``, ``work``, ``makespan``, ``work_bound``, ``speedup``, ``peak_slots``, ``free_float``.

    Assumptions: Dependencies form a nonempty acyclic graph and tasks execute without preemption; earliest starts ignore slot scarcity.

    Excludes: Claiming the critical-path schedule is feasible on the configured finite slot count.

    References
    ----------
    Derivation: A dependency graph on interchangeable slots: aggregate work, the critical path, and both bounds. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.InterfaceMigrationSchedule
    >>> value = case.makespan
    >>> print(f"{value:.3g~}")
    174 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    durations: Dict[int, Quantity]
    predecessors: Dict[int, List[int]]
    slots: int

    @property
    def schedule(self):
        """Earliest starts and finishes from ``calc_earliest_schedule`` on the dependency DAG, before imposing slot limits.

        Calculation: ``mlsysim.physics.coordination.calc_earliest_schedule``."""
        from ...physics.coordination import calc_earliest_schedule

        return calc_earliest_schedule(self.durations, self.predecessors)

    @property
    def start(self) -> Dict[int, Quantity]:
        """Mapping from node identifier to earliest-start time quantity, the schedule's first component.

        Equation: ``schedule[0]``."""
        return self.schedule[0]

    @property
    def finish(self) -> Dict[int, Quantity]:
        """Mapping from node identifier to earliest-finish time quantity, the schedule's second component.

        Equation: ``schedule[1]``."""
        return self.schedule[1]

    @property
    def work(self) -> Quantity:
        """Total serial work as a duration quantity, summing every node duration.

        Equation: ``sum(durations.values(), 0 * next(iter(durations.values())))``."""
        return sum(self.durations.values(), 0 * next(iter(self.durations.values())))

    @property
    def makespan(self) -> Quantity:
        """Dependency critical-path duration, the latest earliest finish; not a slot-constrained schedule.

        The critical path: the latest earliest finish.

        Equation: ``max(finish.values())``."""
        return max(self.finish.values())

    @property
    def work_bound(self) -> Quantity:
        """Aggregate-work duration divided by slot count, a capacity lower bound.

        Equation: ``work / slots``."""
        return self.work / self.slots

    @property
    def speedup(self) -> float:
        """Dimensionless serial-work to dependency-critical-path ratio.

        Serial time over the makespan.

        Equation: ``(work / makespan).to('').magnitude``."""
        return (self.work / self.makespan).to("").magnitude

    @property
    def peak_slots(self) -> int:
        """Maximum concurrently active nodes in the earliest-start schedule; endpoints are half open.

        Most nodes running at once in the earliest schedule.

        Equation: ``max((sum((1 for m in durations if s[m] <= s[n] < f[m])) for n in durations))``."""
        s, f = self.start, self.finish
        return max(sum(1 for m in self.durations if s[m] <= s[n] < f[m]) for n in self.durations)

    @property
    def free_float(self) -> Dict[int, Quantity]:
        """Per-node duration slack before delaying its earliest successor; sink nodes omitted.

        For each node with successors, how long it can slip before delaying the earliest one.

        Equation: ``out``."""
        s, f = self.start, self.finish
        out = {}
        for n in self.durations:
            succ = [m for m, preds in self.predecessors.items() if n in preds]
            if succ:
                out[n] = min(s[m] for m in succ) - f[n]
        return out


class ReferencePassingScenario(BaseScenario):
    """Children handed a pasted repository slice, or a reference and the files they then read.

    Notes
    -----
    Equation: The calculation members below define ``pasted_tokens``, ``reference_total_tokens``, ``full_read_tokens``, ``pasted_cost``, ``reference_cost``, ``reduction``, ``read_fraction``, ``break_even_read_tokens``.

    Assumptions: Each worker receives the same envelope and reads the stated token count at a constant input price.

    Excludes: Cache discounts, output tokens, and differences in task quality.

    References
    ----------
    Derivation: Children handed a pasted repository slice, or a reference and the files they then read. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SixChildReferencePassing
    >>> value = case.reduction
    >>> print(f"{value:.3g}")
    12.2"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workers: int
    slice_tokens: int  # pasted into each child
    reference_tokens: int  # the envelope's reference instead
    read_tokens: int  # files each child then reads
    input_price: Quantity  # USD per million input tokens

    @property
    def pasted_tokens(self) -> int:
        """Total input tokens when every worker receives the entire pasted slice.

        Equation: ``workers * slice_tokens``."""
        return self.workers * self.slice_tokens

    @property
    def reference_total_tokens(self) -> int:
        """Total input tokens when each worker receives a reference and reads the stated files.

        Equation: ``workers * (reference_tokens + read_tokens)``."""
        return self.workers * (self.reference_tokens + self.read_tokens)

    @property
    def full_read_tokens(self) -> int:
        """Tokens if each referenced worker subsequently reads the entire original slice.

        A reference after which each child reads the whole slice.

        Equation: ``workers * (reference_tokens + slice_tokens)``."""
        return self.workers * (self.reference_tokens + self.slice_tokens)

    @property
    def pasted_cost(self) -> Quantity:
        """USD input cost of duplicating the pasted slice across workers.

        Equation: ``(Q_(pasted_tokens, 'token') * input_price).to('dollar')``."""
        return (Q_(self.pasted_tokens, "token") * self.input_price).to("dollar")

    @property
    def reference_cost(self) -> Quantity:
        """USD input cost of reference envelopes plus the stated file reads.

        Equation: ``(Q_(reference_total_tokens, 'token') * input_price).to('dollar')``."""
        return (Q_(self.reference_total_tokens, "token") * self.input_price).to("dollar")

    @property
    def reduction(self) -> float:
        """Pasted-to-reference input-token ratio, dimensionless.

        Equation: ``pasted_tokens / reference_total_tokens``."""
        return self.pasted_tokens / self.reference_total_tokens

    @property
    def read_fraction(self) -> float:
        """Read tokens divided by full slice tokens, dimensionless.

        Equation: ``read_tokens / slice_tokens``."""
        return self.read_tokens / self.slice_tokens

    @property
    def break_even_read_tokens(self) -> int:
        """Per-worker read-token allowance where reference overhead exactly cancels savings.

        Reads per child at which the reference saves nothing.

        Equation: ``slice_tokens - reference_tokens``."""
        return self.slice_tokens - self.reference_tokens


class SharedWriteScenario(BaseScenario):
    """Workers each writing ``writes_per_worker`` of ``files`` files, uniformly and independently.

    Notes
    -----
    Equation: The calculation members below define ``disjoint``, ``disjoint_approx``, ``collision``, ``even_odds_fleet``.

    Assumptions: Each worker chooses a uniform write set independently, with no repeated file within a set.

    Excludes: Hot-file concentration, synchronized edit behavior, and semantic merge conflicts between distinct files.

    References
    ----------
    Derivation: Workers each writing ``writes_per_worker`` of ``files`` files, uniformly and independently. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ThousandFileSharedWrites
    >>> value = case.disjoint[case.small_fleet]
    >>> print(f"{value:.3g}")
    0.904"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    files: int
    writes_per_worker: int
    max_fleet: int
    small_fleet: int  # fleet sizes the comparison quotes
    large_fleet: int

    @property
    def disjoint(self) -> Dict[int, float]:
        """Exact no-overlap probability by fleet count via ``calc_disjoint_writes``.

        Exact probability that no two write sets overlap, for each fleet size.

        Calculation: ``mlsysim.physics.coordination.calc_disjoint_writes``."""
        from ...physics.coordination import calc_disjoint_writes

        return {n: calc_disjoint_writes(n, self.files, self.writes_per_worker) for n in range(1, self.max_fleet + 1)}

    @property
    def disjoint_approx(self) -> Dict[int, float]:
        """Approximate no-overlap probability by fleet count via ``calc_disjoint_writes_approx``.

        Calculation: ``mlsysim.physics.coordination.calc_disjoint_writes_approx``."""
        from ...physics.coordination import calc_disjoint_writes_approx

        return {
            n: calc_disjoint_writes_approx(n, self.files, self.writes_per_worker) for n in range(1, self.max_fleet + 1)
        }

    @property
    def collision(self) -> Dict[int, float]:
        """Probability of at least one shared write by fleet count, complement of no-overlap probability.

        Probability of at least one shared write.

        Equation: ``{n: 1 - p for n, p in disjoint.items()}``."""
        return {n: 1 - p for n, p in self.disjoint.items()}

    @property
    def even_odds_fleet(self) -> Optional[int]:
        """First enumerated fleet count with collision probability above one half, or None.

        The smallest fleet for which a shared write is more likely than not.

        Equation: ``next((n for n, p in disjoint.items() if p < 0.5), None)``."""
        return next((n for n, p in self.disjoint.items() if p < 0.5), None)


class CorrelatedReviewersScenario(BaseScenario):
    """Reviewers who each falsely approve with the same marginal rate, independent or correlated.

    The correlated model draws one propensity per candidate from a beta law with
    the stated mean and pairwise correlation; reviewers err independently given
    it (a beta-binomial). The beta mass above one half floors the majority
    error of any panel size.

    Notes
    -----
    Equation: The calculation members below define ``shape``, ``independent_majority``, ``correlated_majority``, ``independent_unanimous``, ``correlated_unanimous``, ``unanimous_ratio``, ``floor``, ``panels``.

    Assumptions: Conditional reviewer errors are independent given a shared beta-distributed propensity; panel sizes are odd.

    Excludes: Evidence that this correlation model describes a particular reviewer population.

    References
    ----------
    Derivation: Reviewers who each falsely approve with the same marginal rate, independent or correlated. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FiveCorrelatedReviewers
    >>> value = case.unanimous_ratio
    >>> print(f"{value:.3g}")
    122"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    reviewers: int  # an odd panel
    false_approval: float  # marginal false-approval rate
    correlation: float  # pairwise correlation of the beta mixture
    large_ensemble: int  # a larger odd panel to compare

    @property
    def shape(self):
        """Beta-mixture shape parameters from ``calc_beta_mixture_shape`` for the stated marginal error and correlation.

        Calculation: ``mlsysim.physics.coordination.calc_beta_mixture_shape``."""
        from ...physics.coordination import calc_beta_mixture_shape

        return calc_beta_mixture_shape(self.false_approval, self.correlation)

    def independent_majority(self, panel: int) -> float:
        """False-approval majority probability for the supplied panel count via ``calc_binomial_tail``.

        Calculation: ``mlsysim.physics.coordination.calc_binomial_tail``."""
        from ...physics.coordination import calc_binomial_tail

        return calc_binomial_tail(panel, self.false_approval, panel // 2 + 1)

    def correlated_majority(self, panel: int) -> float:
        """False-approval majority probability from the upper half of the beta-binomial mass.

        Calculation: ``mlsysim.physics.coordination.calc_beta_binomial_pmf``."""
        from ...physics.coordination import calc_beta_binomial_pmf

        return sum(calc_beta_binomial_pmf(panel, self.false_approval, self.correlation)[panel // 2 + 1 :])

    @property
    def independent_unanimous(self) -> float:
        """Unanimous false-approval probability with independent reviewers, marginal error raised to panel size.

        Equation: ``false_approval ** reviewers``."""
        return self.false_approval**self.reviewers

    @property
    def correlated_unanimous(self) -> float:
        """Unanimous false-approval probability from the last beta-binomial mass.

        Calculation: ``mlsysim.physics.coordination.calc_beta_binomial_pmf``."""
        from ...physics.coordination import calc_beta_binomial_pmf

        return calc_beta_binomial_pmf(self.reviewers, self.false_approval, self.correlation)[-1]

    @property
    def unanimous_ratio(self) -> float:
        """Correlated-to-independent unanimous-error probability ratio, dimensionless.

        Equation: ``correlated_unanimous / independent_unanimous``."""
        return self.correlated_unanimous / self.independent_unanimous

    @property
    def floor(self) -> float:
        """Beta propensity mass above one half via ``calc_beta_tail``; asymptotic majority-error limit.

        P(propensity > 1/2): the limit of correlated majority error as the panel grows.

        Calculation: ``mlsysim.physics.coordination.calc_beta_tail``."""
        from ...physics.coordination import calc_beta_tail

        return calc_beta_tail(self.false_approval, self.correlation, 0.5)

    @property
    def panels(self) -> List[int]:
        """Odd panel sizes from one through the configured large-panel ceiling.

        Odd panel sizes from one to the large ensemble.

        Equation: ``list(range(1, large_ensemble + 1, 2))``."""
        return list(range(1, self.large_ensemble + 1, 2))


class FanoutDeadlineScenario(BaseScenario):
    """A fan-out whose workers each finish by the deadline with one probability, independently.

    An all-results barrier needs every worker (horizon compounding across workers);
    a quorum contract accepts any ``required`` of ``workers``.

    Notes
    -----
    Equation: All-results probability is ``p**workers``; quorum probability is the binomial tail
    at required successes.

    Assumptions: Workers have identical independent on-time probabilities supplied
    externally.

    Excludes: Correlated stalls and an inferred latency distribution; deadline labels the
    supplied probability.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.agents.calc_first_horizon_below``,
    ``mlsysim.physics.agents.calc_unchecked_success``,
    ``mlsysim.physics.coordination.calc_binomial_tail``. Their contracts supply the
    underlying equations; property equations show the remaining unit conversions and
    accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.EightWorkerDeadlineFanout
    >>> value = scenario.all_on_time
    >>> round(float(value), 3)
    0.43
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workers: int
    required: int
    on_time_probability: float
    deadline: Quantity
    max_fleet: int  # fleet sizes to tabulate

    @property
    def all_on_time(self) -> float:
        """Probability that every worker meets the deadline, using independent worker success
        compounding.

        Equation: ``calc_unchecked_success(on_time_probability, workers)``.
        """
        from ...physics.agents import calc_unchecked_success

        return calc_unchecked_success(self.on_time_probability, self.workers)

    @property
    def quorum_on_time(self) -> float:
        """Probability that at least ``required`` workers finish on time, using the binomial upper
        tail.

        Equation: ``calc_binomial_tail(workers, on_time_probability, required)``.
        """
        from ...physics.coordination import calc_binomial_tail

        return calc_binomial_tail(self.workers, self.on_time_probability, self.required)

    @property
    def even_odds_workers(self) -> int:
        """Smallest worker count whose all-results success probability is below one half.

        Equation: ``calc_first_horizon_below(on_time_probability, 0.5)``.
        """
        from ...physics.agents import calc_first_horizon_below

        return calc_first_horizon_below(self.on_time_probability, 0.5)

    @property
    def all_on_time_by_fleet(self) -> Dict[int, float]:
        """Map each fleet size to its dimensionless all-results success probability.

        Equation: ``{n: calc_unchecked_success(on_time_probability, n) for n in range(1,
        max_fleet + 1)}``.
        """
        from ...physics.agents import calc_unchecked_success

        return {n: calc_unchecked_success(self.on_time_probability, n) for n in range(1, self.max_fleet + 1)}


class CreditWindowScenario(BaseScenario):
    """A producer feeding a slower model consumer, unbounded or behind a credit window.

    The consumer serves one item per ``tokens_per_item`` at its decode rate. With
    no bound the queue grows at the rate difference; a saturated window of W
    credits holds each item about W / mu by Little's law.

    Notes
    -----
    Equation: Service rate is decode_rate/tokens_per_item; fluid backlog is
    ``(producer_rate-service_rate)*horizon``; saturated window residence is
    credits/service_rate.

    Assumptions: Rates and per-item work remain constant, the initial backlog is zero, and
    the growth calculation describes overload.

    Excludes: Bursts, integer event scheduling, and a nonnegative queue clamp; underload
    produces a negative fluid backlog.

    References
    ----------
    Derivation: Service rate is decode_rate/tokens_per_item; fluid backlog is
    ``(producer_rate-service_rate)*horizon``; saturated window residence is
    credits/service_rate. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.DiffReviewCreditWindow
    >>> value = scenario.service_rate
    >>> str(value.units)
    '1/s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    producer_rate: Quantity  # items per second
    decode_rate: Quantity  # consumer tokens per second
    tokens_per_item: Quantity
    horizon: Quantity
    credits: int
    residence_target: Quantity  # a mean residence to size a window for

    @property
    def service_rate(self) -> Quantity:
        """Consumer completions per second, dividing decode tokens per second by tokens needed per
        item.

        Equation: ``(decode_rate / tokens_per_item).to('1/s')``.
        """
        return (self.decode_rate / self.tokens_per_item).to("1/s")

    @property
    def growth_rate(self) -> Quantity:
        """Net queue growth in items per second; negative values indicate producer demand below
        service capacity.

        Equation: ``(producer_rate - service_rate).to('1/s')``.
        """
        return (self.producer_rate - self.service_rate).to("1/s")

    @property
    def backlog(self) -> float:
        """Fluid backlog count accumulated from an initially empty queue over ``horizon``; not
        clamped at zero.

        Equation: ``(growth_rate * horizon).to('').magnitude``.
        """
        return (self.growth_rate * self.horizon).to("").magnitude

    @property
    def backlog_wait(self) -> Quantity:
        """Seconds needed to drain the fluid backlog at the fixed service rate.

        Equation: ``(backlog / service_rate).to('s')``.
        """
        return (self.backlog / self.service_rate).to("s")

    @property
    def window_residence(self) -> Quantity:
        """Saturated mean item residence in seconds from credit count divided by service rate.

        Equation: ``(credits / service_rate).to('s')``.
        """
        return (self.credits / self.service_rate).to("s")

    @property
    def window_for_target(self) -> int:
        """Largest integer credit count within the requested mean residence time.

        Equation: ``_math.floor((residence_target * service_rate).to('').magnitude + 1e-09)``.
        """
        return _math.floor((self.residence_target * self.service_rate).to("").magnitude + 1e-9)


class CohortArmRecord(BaseModel):
    """One policy's outcomes and settled usage over a cohort of attempted tasks.

    Notes
    -----
    Equation: This record stores settled usage and outcome counts; it performs no numerical
    transformation.

    Assumptions: Counts and metered usage refer to the same attempted cohort; discarded
    usage is already included in total usage.

    Excludes: Independent validation of outcome labels, metering accuracy, and duplicate
    usage records.

    References
    ----------
    Derivation: This record stores settled usage and outcome counts; it performs no
    numerical transformation. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Q_
    >>> record = CohortArmRecord(accepted=1, on_time=1, median_accepted_time=Q_(1, "s"),
    ...     input_tokens=Q_(1, "token"), output_tokens=Q_(1, "token"), service_time=Q_(1, "s"),
    ...     discard_input_tokens=Q_(0, "token"), discard_output_tokens=Q_(0, "token"),
    ...     discard_service_time=Q_(0, "s"))
    >>> record.accepted
    1
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    accepted: int  # accepted by the final cutoff
    on_time: int  # accepted by the deadline
    median_accepted_time: Quantity  # among accepted tasks
    input_tokens: Quantity
    output_tokens: Quantity
    service_time: Quantity  # metered tool, assessment, and cleanup time
    discard_input_tokens: Quantity  # failed attempts and canceled branches, included above
    discard_output_tokens: Quantity
    discard_service_time: Quantity


class DelegationCohortScenario(BaseScenario):
    """A single controller and a delegated design on one matched cohort, priced alike.

    Spend charges every initiated attempt at the same prices; cost per accepted
    task divides it by the accepted count. A policy passes the screen if it meets
    both acceptance floors within the budget.

    Notes
    -----
    Equation: Spend sums token counts times their prices and service duration times its
    price; accepted-task cost divides spend by accepted count.

    Assumptions: Both policies share the same cohort, prices, deadlines, and acceptance
    contract; ratio denominators are nonzero.

    Excludes: Prediction of acceptance rates or latency; supplied outcomes are inputs rather
    than results generated by the model.

    References
    ----------
    Derivation: Spend sums token counts times their prices and service duration times its
    price; accepted-task cost divides spend by accepted count. This is declared scenario
    accounting rather than a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.MatchedDelegationCohort
    >>> value = scenario.extra_spend
    >>> str(value.units)
    'dollar'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    attempted: int
    accepted_floor: int
    on_time_floor: int
    deadline: Quantity
    cutoff: Quantity
    budget: Quantity
    input_price: Quantity  # USD per million tokens
    output_price: Quantity
    service_price: Quantity  # USD per second
    single: CohortArmRecord
    delegated: CohortArmRecord

    def spend(self, arm: CohortArmRecord) -> Quantity:
        """Total USD charged to ``arm`` for input tokens, output tokens, and settled service time.

        Equation: ``(arm.input_tokens * input_price + arm.output_tokens * output_price +
        arm.service_time * service_price).to('dollar')``.
        """
        return (
            arm.input_tokens * self.input_price
            + arm.output_tokens * self.output_price
            + arm.service_time * self.service_price
        ).to("dollar")

    def discarded(self, arm: CohortArmRecord) -> Quantity:
        """USD spent on the arm's failed or canceled work already included in its total usage.

        Equation: ``(arm.discard_input_tokens * input_price + arm.discard_output_tokens *
        output_price + arm.discard_service_time * service_price).to('dollar')``.
        """
        return (
            arm.discard_input_tokens * self.input_price
            + arm.discard_output_tokens * self.output_price
            + arm.discard_service_time * self.service_price
        ).to("dollar")

    def cost_per_accepted(self, arm: CohortArmRecord) -> Quantity:
        """Total spend divided by positive accepted count, in USD per accepted task.

        Equation: ``spend(arm) / arm.accepted``.
        """
        return self.spend(arm) / arm.accepted

    def passes(self, arm: CohortArmRecord) -> bool:
        """Whether acceptance count, on-time count, and total spend jointly meet the declared
        screen.

        Equation: ``arm.accepted >= accepted_floor and arm.on_time >= on_time_floor and
        (spend(arm) <= budget)``.
        """
        return (
            arm.accepted >= self.accepted_floor and arm.on_time >= self.on_time_floor and self.spend(arm) <= self.budget
        )

    @property
    def extra_spend(self) -> Quantity:
        """Delegated minus single-controller spend in USD, allowing a negative saving.

        Equation: ``spend(delegated) - spend(single)``.
        """
        return self.spend(self.delegated) - self.spend(self.single)

    @property
    def extra_on_time(self) -> int:
        """Additional on-time accepted tasks under delegation, allowing a negative difference.

        Equation: ``delegated.on_time - single.on_time``.
        """
        return self.delegated.on_time - self.single.on_time

    @property
    def marginal_on_time_cost(self) -> Quantity:
        """Additional USD per additional on-time acceptance; requires nonzero additional count.

        Equation: ``extra_spend / extra_on_time``.
        """
        return self.extra_spend / self.extra_on_time

    @property
    def cost_ratio(self) -> float:
        """Delegated cost per acceptance divided by single-controller cost, dimensionless.

        Equation: ``(cost_per_accepted(delegated) /
        cost_per_accepted(single)).to('').magnitude``.
        """
        return (self.cost_per_accepted(self.delegated) / self.cost_per_accepted(self.single)).to("").magnitude

    @property
    def discard_share(self) -> float:
        """Fraction of delegated spend attributable to failed or canceled work.

        Equation: ``(discarded(delegated) / spend(delegated)).to('').magnitude``.
        """
        return (self.discarded(self.delegated) / self.spend(self.delegated)).to("").magnitude


class ResidualViolationScenario(BaseScenario):
    """A residual per-step rate of violating proposals over ``horizon`` steps, for a day's trajectories.

    Notes
    -----
    Equation: Trajectory violation probability is ``1-(1-step_violation)**horizon``; multiplying
    by daily trajectories gives an expected count.

    Assumptions: Per-step violation events are independent with a fixed rate.

    Excludes: Correlated violations, event severity, and distinction between proposed and
    executed harmful actions.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.agents.calc_unchecked_success``.
    Their contracts supply the underlying equations; property equations show the remaining
    unit conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ResidualViolationsFiftyTurns
    >>> value = scenario.trajectory_violation
    >>> round(float(value), 3)
    0.049
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    step_violation: float  # per-step rate of violating proposals that training leaves
    horizon: int  # steps per trajectory
    trajectories_per_day: int

    @property
    def trajectory_violation(self) -> float:
        """Chance of at least one violating proposal under independent per-step events.

        Equation: ``1 - calc_unchecked_success(1 - step_violation, horizon)``.
        """
        from ...physics.agents import calc_unchecked_success

        return 1 - calc_unchecked_success(1 - self.step_violation, self.horizon)

    @property
    def violating_per_day(self) -> float:
        """Expected daily trajectories containing a violation, not a count of violation events.

        Equation: ``trajectory_violation * trajectories_per_day``.
        """
        return self.trajectory_violation * self.trajectories_per_day


class ShortenedWindowScenario(BaseScenario):
    """An observation window cut to ``window_share`` of its adopted length; samples scale with it.

    Notes
    -----
    Equation: Half-width changes by 1/sqrt(window_share) when independent sample count is
    proportional to window length.

    Assumptions: The rate of independent samples and outcome variance stay constant;
    window_share is positive.

    Excludes: Changes in bias, autocorrelation, or nonstationarity when the observation
    window is shortened.

    References
    ----------
    Derivation: Half-width changes by 1/sqrt(window_share) when independent sample count is
    proportional to window length. This is declared scenario accounting rather than a fitted
    empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.HalvedHealthWindow
    >>> value = scenario.interval_widening
    >>> round(float(value), 3)
    1.414
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window_share: float

    @property
    def interval_widening(self) -> float:
        """Normal-error half-width multiplier when sample count scales by positive
        ``window_share``.

        Equation: ``1 / _math.sqrt(window_share)``.
        """
        return 1 / _math.sqrt(self.window_share)


class CounterfactualSearchScenario(BaseScenario):
    """Replay paths that blame for a failure must search: a whole horizon, or one checked window.

    Notes
    -----
    Equation: Leaf paths number ``branching**depth``; serial replay time is checked-window path
    count times time per path.

    Assumptions: Every step has the same branching factor and each full path is replayed
    separately.

    Excludes: Pruning, shared prefixes, parallel replay, and proof that a replay identifies
    causal responsibility.

    References
    ----------
    Derivation: Leaf paths number ``branching**depth``; serial replay time is checked-window
    path count times time per path. This is declared scenario accounting rather than a
    fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CheckedWindowBlameSearch
    >>> value = scenario.full_paths
    >>> round(float(value), 3)
    1.2676506002282294e+30
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    branching: int  # candidate actions per step
    horizon: int  # steps in the task
    window: int  # steps between intermediate checks
    replay_time: Quantity  # time per replayed check

    @property
    def full_paths(self) -> int:
        """Leaf-path count for fixed branching at every step of the full horizon.

        Equation: ``branching ** horizon``.
        """
        return self.branching**self.horizon

    @property
    def full_exponent(self) -> float:
        """Base-ten logarithm of full leaf-path count, dimensionless.

        Equation: ``_math.log10(full_paths)``.
        """
        return _math.log10(self.full_paths)

    @property
    def window_paths(self) -> int:
        """Leaf-path count within the shorter checked replay window.

        Equation: ``branching ** window``.
        """
        return self.branching**self.window

    @property
    def window_search_time(self) -> Quantity:
        """Serial minutes to replay every checked-window path at constant per-path duration.

        Equation: ``(window_paths * replay_time).to('minute')``.
        """
        return (self.window_paths * self.replay_time).to("minute")


class SerialTailExposureScenario(BaseScenario):
    """Serial calls that each exceed their own tail latency independently.

    Notes
    -----
    Equation: The result property delegates to the named solver documented below; it
    converts declared scenario units before passing the inputs.

    Assumptions: Each serial call independently exceeds its own declared tail threshold.

    Excludes: A distribution for the total serial latency and correlated stragglers.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.agents.calc_independent_success``.
    Their contracts supply the underlying equations; property equations show the remaining
    unit conversions and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ThirtyCallTailExposure
    >>> value = scenario.any_tail
    >>> round(float(value), 3)
    0.26
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    calls: int
    tail_probability: float  # 0.01 for a 99th percentile

    @property
    def any_tail(self) -> float:
        """Probability that at least one independent serial call exceeds its own tail threshold.

        Equation: ``calc_independent_success(tail_probability, calls)``.
        """
        from ...physics.agents import calc_independent_success

        return calc_independent_success(self.tail_probability, self.calls)
