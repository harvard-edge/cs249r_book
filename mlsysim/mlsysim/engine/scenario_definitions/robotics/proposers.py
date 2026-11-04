"""Physical-machine scenarios for proposers."""

from __future__ import annotations
from typing import Any, Dict, Optional
from ....core.types import Quantity
from ..base import BaseScenario


class VlaEvaluationScenario(BaseScenario):
    """One fresh evaluation of ``machine``'s intent model on its application processor.

    Each input left as None takes the machine's deployed value: the weight
    precision, the input and action token counts, and the decoding mode. The
    sustained-bandwidth fraction and compute utilization are the machine's.

    Notes
    -----
    Equation: evaluation = VlaEvaluationModel.solve(model, processor, selected precision
    and token counts, supplied efficiencies).

    Assumptions: An omitted override takes its value from the machine configuration;
    compute and bandwidth efficiencies remain declared inputs.

    Excludes: Scheduling and control guarantees beyond the underlying VLA model contract.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.VlaEvaluationModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IntentModelEvaluationFp16
    >>> round(float(case.bytes_per_param.to('B/param').magnitude), 3)
    2.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # EmbodiedMobileManipulator
    weight_bytes_per_param: Optional[Quantity] = None  # default: the deployed precision
    input_tokens: Optional[int] = None  # default: machine.intent_input_tokens
    action_tokens: Optional[int] = None  # default: the model's action dimension
    parallel_decode: Optional[bool] = None  # default: machine.intent_parallel_decode

    @property
    def bytes_per_param(self) -> Quantity:
        """Stored weight bytes per parameter, taking the machine's deployed format unless
        overridden.

        Equation: ``return machine.intent_weight_bytes_per_param if weight_bytes_per_param
        is None else weight_bytes_per_param``."""
        return (
            self.machine.intent_weight_bytes_per_param
            if self.weight_bytes_per_param is None
            else self.weight_bytes_per_param
        )

    @property
    def tokens_in(self) -> int:
        """Input-token count, taking the deployed machine setting unless explicitly
        overridden.

        Equation: ``return machine.intent_input_tokens if input_tokens is None else
        input_tokens``."""
        return self.machine.intent_input_tokens if self.input_tokens is None else self.input_tokens

    @property
    def tokens_out(self) -> int:
        """Output-token count, taking the model's action dimension unless explicitly
        overridden.

        Equation: ``return machine.intent_model.action_dim if action_tokens is None else
        action_tokens``."""
        return self.machine.intent_model.action_dim if self.action_tokens is None else self.action_tokens

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.VlaEvaluationModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.VlaEvaluationModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ...solvers import VlaEvaluationModel

        rm = self.machine
        parallel = rm.intent_parallel_decode if self.parallel_decode is None else self.parallel_decode
        return VlaEvaluationModel().solve(
            rm.intent_model,
            rm.brain_soc,
            self.bytes_per_param,
            self.tokens_in,
            self.tokens_out,
            rm.dram_efficiency,
            rm.intent_compute_utilization,
            parallel_decode=parallel,
        )


class RateTargetScenario(BaseScenario):
    """``evaluation`` held against a ``target_rate``: one fresh evaluation per period, not pipelined.

    Notes
    -----
    Equation: period = 1 / target rate; slack = period - evaluation time; token budget
    scales the prefill remainder.

    Assumptions: One fresh evaluation must complete per period without pipelining; prefill
    time scales linearly for the token-budget approximation.

    Excludes: Overlap across evaluations, nonlinear prefill scaling, and clamping a
    negative token budget to zero.

    References
    ----------
    Derivation: period = 1 / target rate; slack = period -
    evaluation time; token budget scales the prefill remainder. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IntentModelFiveHertz
    >>> round(float(case.period.to('ms').magnitude), 3)
    200.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    evaluation: VlaEvaluationScenario
    target_rate: Quantity

    @property
    def period(self) -> Quantity:
        """Required interval between fresh evaluations in milliseconds, inverse of target
        rate.

        Equation: ``return (1 / target_rate).to('ms')``."""
        return (1 / self.target_rate).to("ms")

    @property
    def fits(self) -> bool:
        """Whether modeled total evaluation time fits one non-pipelined period; a Boolean
        feasibility check.

        Equation: ``return evaluation.result.t_total <= period``."""
        return self.evaluation.result.t_total <= self.period

    @property
    def slack(self) -> Quantity:
        """Period minus modeled evaluation time in milliseconds; negative values indicate
        overrun.

        Equation: ``return (period - evaluation.result.t_total).to('ms')``."""
        return (self.period - self.evaluation.result.t_total).to("ms")

    @property
    def slack_fraction(self) -> float:
        """Remaining period as a dimensionless share, which may be negative.

        Equation: ``return (slack / period).to('').magnitude``."""
        return (self.slack / self.period).to("").magnitude

    @property
    def prefill_budget(self) -> Quantity:
        """Milliseconds available to prefill after reserving the modeled decode duration.

        Equation: ``return (period - evaluation.result.t_decode).to('ms')``."""
        return (self.period - self.evaluation.result.t_decode).to("ms")

    @property
    def token_fraction(self) -> float:
        """Admissible input fraction under a linear-prefill approximation; it is not clamped
        to zero or one.

        Equation: ``return (prefill_budget /
        evaluation.result.t_prefill).to('').magnitude``."""
        return (self.prefill_budget / self.evaluation.result.t_prefill).to("").magnitude

    @property
    def tokens_within(self) -> float:
        """Input-token budget under the linear-prefill approximation; returned as a
        fractional count.

        Equation: ``return token_fraction * evaluation.result.prefill_tokens``."""
        return self.token_fraction * self.evaluation.result.prefill_tokens


class VlaTokenBudgetScenario(BaseScenario):
    """Tokens a VLA backbone reads from ``views`` square camera frames, and their memory per token.

    Notes
    -----
    Equation: visual tokens = views * floor(frame pixels / patch pixels)^2; KV bytes per
    token = 2 * layers * hidden width * element bytes.

    Assumptions: Views are square, one token represents each complete patch, and keys and
    values use full hidden width.

    Excludes: Image encoder intermediates, grouped-query or compressed caches, padding
    patches, and runtime workspace.

    References
    ----------
    Derivation: visual tokens = views * floor(frame pixels /
    patch pixels)^2; KV bytes per token = 2 * layers * hidden width * element bytes. The
    individual properties state units and intermediate accounting; this class introduces
    no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ThreeViewVlaTokens
    >>> round(float(case.patches_per_view), 3)
    256.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model: Any  # EmbodiedWorkload (layers, hidden width)
    views: int
    frame_px: int
    patch_px: int
    prompt_tokens: int
    state_tokens: int
    context_tokens: int  # the rounded context length S the prose uses
    element_bytes: Quantity  # bytes per stored activation, key, or value

    @property
    def patches_per_view(self) -> int:
        """Patch-token count per square view, discarding pixels outside complete patches.

        Equation: ``return (frame_px // patch_px) ** 2``."""
        return (self.frame_px // self.patch_px) ** 2

    @property
    def patches_per_side(self) -> int:
        """Complete patches along one image side, using integer division.

        Equation: ``return frame_px // patch_px``."""
        return self.frame_px // self.patch_px

    @property
    def visual_tokens(self) -> int:
        """Visual-token count across all views, one token per complete square patch.

        Equation: ``return views * patches_per_view``."""
        return self.views * self.patches_per_view

    @property
    def total_tokens(self) -> int:
        """Combined visual, prompt, and state token count before context rounding.

        Equation: ``return visual_tokens + prompt_tokens + state_tokens``."""
        return self.visual_tokens + self.prompt_tokens + self.state_tokens

    @property
    def single_view_tokens(self) -> int:
        """One view's patch tokens plus prompt tokens; state tokens are not included in this
        subtotal.

        Equation: ``return patches_per_view + prompt_tokens``."""
        return self.patches_per_view + self.prompt_tokens

    @property
    def embedding_memory(self) -> Quantity:
        """Embedding-state memory in MB for the stated context and model hidden width.

        Equation: ``return (context_tokens * model.hidden_dim * element_bytes).to('MB')``."""
        return (self.context_tokens * self.model.hidden_dim * self.element_bytes).to("MB")

    @property
    def kv_per_token(self) -> Quantity:
        """Full-width key and value storage per token in bytes across every layer.

        Equation: ``return (2 * model.layers * model.hidden_dim *
        element_bytes).to('byte')``."""
        return (2 * self.model.layers * self.model.hidden_dim * self.element_bytes).to("byte")


class KvHistoryScenario(BaseScenario):
    """Key-value cache for ``history_window`` of retained context at ``history_rate`` beside ``machine``'s weights.

    Notes
    -----
    Equation: retained KV bytes = rounded(history rate * window) * per-step KV bytes;
    combined = weights + retained KV.

    Assumptions: Frames retain the full declared context independently; history rate and
    window use compatible units.

    Excludes: Cache sharing between frames, eviction policy, activations, allocator
    reserve, and memory-access latency.

    References
    ----------
    Derivation: retained KV bytes = rounded(history rate *
    window) * per-step KV bytes; combined = weights + retained KV. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ThreeViewKvHistory
    >>> round(float(case.frames), 3)
    50.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tokens: VlaTokenBudgetScenario
    machine: Any  # EmbodiedMobileManipulator (module memory, intent model)
    history_window: Quantity
    history_rate: Quantity
    weight_bytes_per_param: Quantity  # the stored weights the history sits beside

    @property
    def frames(self) -> int:
        """Retained frame count, rounding history duration times sampling rate.

        Equation: ``return round((history_window * history_rate).to('').magnitude)``."""
        return round((self.history_window * self.history_rate).to("").magnitude)

    @property
    def kv_per_frame(self) -> Quantity:
        """Visual key-value memory per multi-view frame in MB.

        Equation: ``return (tokens.kv_per_token * tokens.visual_tokens).to('MB')``."""
        return (self.tokens.kv_per_token * self.tokens.visual_tokens).to("MB")

    @property
    def kv_per_step(self) -> Quantity:
        """Key-value memory for the full declared context of one step in MB.

        Equation: ``return (tokens.kv_per_token * tokens.context_tokens).to('MB')``."""
        return (self.tokens.kv_per_token * self.tokens.context_tokens).to("MB")

    @property
    def visual_history(self) -> Quantity:
        """Retained visual-only key-value history in GB.

        Equation: ``return (kv_per_frame * frames).to('GB')``."""
        return (self.kv_per_frame * self.frames).to("GB")

    @property
    def history(self) -> Quantity:
        """Retained full-context key-value history in GB.

        Equation: ``return (kv_per_step * frames).to('GB')``."""
        return (self.kv_per_step * self.frames).to("GB")

    @property
    def weights(self) -> Quantity:
        """Intent-model weight memory in GB at the stipulated storage precision.

        Equation: ``return (machine.intent_model.parameters *
        weight_bytes_per_param).to('GB')``."""
        return (self.machine.intent_model.parameters * self.weight_bytes_per_param).to("GB")

    @property
    def combined(self) -> Quantity:
        """Weight plus retained full-context key-value memory in GB; runtime workspace is
        excluded.

        Equation: ``return (weights + history).to('GB')``."""
        return (self.weights + self.history).to("GB")

    @property
    def module_memory(self) -> Quantity:
        """Application-processor memory capacity in GB from the hardware specification.

        Equation: ``return machine.brain_soc.memory.capacity.to('GB')``."""
        return self.machine.brain_soc.memory.capacity.to("GB")

    @property
    def combined_share(self) -> float:
        """Fraction of application-processor memory occupied by weights and retained history.

        Equation: ``return (combined / module_memory).to('').magnitude``."""
        return (self.combined / self.module_memory).to("").magnitude


class DecoderFamiliesScenario(BaseScenario):
    """One fresh ``machine.chunk_horizon``-step chunk from ``backbone``, per decoder family.

    Notes
    -----
    Equation: result = DecoderFamiliesModel.solve(...) with the declared backbone,
    fast_tokens, expert_params inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.DecoderFamiliesModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond DecoderFamiliesModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.DecoderFamiliesModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IntentBackboneDecoderFamilies
    >>> round(float(case.result.binned_tokens), 3)
    112.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    backbone: VlaEvaluationScenario
    fast_tokens: int
    expert_params: Quantity
    expert_steps: int
    expert_bytes_per_param: Quantity
    kv_bytes_per_value: Quantity

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.DecoderFamiliesModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.DecoderFamiliesModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ...solvers import DecoderFamiliesModel

        b = self.backbone
        rm = b.machine
        return DecoderFamiliesModel().solve(
            rm.intent_model,
            rm.brain_soc,
            b.bytes_per_param,
            b.tokens_in,
            rm.dram_efficiency,
            rm.intent_compute_utilization,
            rm.chunk_horizon,
            rm.intent_model.action_dim,
            self.fast_tokens,
            self.expert_params,
            self.expert_steps,
            self.expert_bytes_per_param,
            self.kv_bytes_per_value,
            rm.chunk_period,
        )


class ProposerTrafficScenario(BaseScenario):
    """``machine``'s intent model and chunk policy sharing its memory bus (weight traffic only).

    Notes
    -----
    Equation: result = ProposerTrafficModel.solve(...) with the declared machine,
    fp16_bytes_per_param inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.ProposerTrafficModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond ProposerTrafficModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.ProposerTrafficModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WarehouseProposerTraffic
    >>> round(float(case.result.sustained_bandwidth.to('GB/s').magnitude), 3)
    142.8"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    machine: Any  # EmbodiedMobileManipulator
    fp16_bytes_per_param: Quantity  # the 16-bit format the sweep and supply comparisons price

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.ProposerTrafficModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.ProposerTrafficModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ...solvers import ProposerTrafficModel

        return ProposerTrafficModel().solve(self.machine, self.fp16_bytes_per_param)


class EvidenceHorizonScenario(BaseScenario):
    """``site``'s conveyor target tracked by ``machine``'s chunk policy: evidence horizon against renewal.

    Notes
    -----
    Equation: horizon comparison = EvidenceHorizonModel.solve(site error budget, target
    motion, slip, and chunk schedule).

    Assumptions: Slip acceleration and renewal rate use site and machine defaults unless
    overridden.

    Excludes: A measured grasp-success probability or a certified stopping/control
    guarantee.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.EvidenceHorizonModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ConveyorMugEvidenceHorizon
    >>> round(float(case.a_slip.to('m/s²').magnitude), 3)
    0.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    site: Any  # EmbodiedSiteScenario (conveyor, tolerance, initial error, slip)
    machine: Any  # EmbodiedMobileManipulator (chunk schedule)
    slip_acceleration: Optional[Quantity] = None  # default: the site's residual slip
    chunk_rate: Optional[Quantity] = None  # default: the machine's renewal rate

    @property
    def a_slip(self) -> Quantity:
        """Residual target-slip acceleration from the site, unless explicitly overridden.

        Equation: ``return site.a_conveyor_slip if slip_acceleration is None else
        slip_acceleration``."""
        return self.site.a_conveyor_slip if self.slip_acceleration is None else self.slip_acceleration

    @property
    def renewal_rate(self) -> Quantity:
        """Chunk renewal frequency from the machine, unless explicitly overridden.

        Equation: ``return machine.chunk_rate if chunk_rate is None else chunk_rate``."""
        return self.machine.chunk_rate if self.chunk_rate is None else self.chunk_rate

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.EvidenceHorizonModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.EvidenceHorizonModel.solve``; returned quantities and
        feasibility follow that contract."""
        from ...solvers import EvidenceHorizonModel

        rm = self.machine
        return EvidenceHorizonModel().solve(
            self.site.grasp_tolerance,
            self.site.grasp_initial_error,
            self.site.v_conveyor,
            self.a_slip,
            self.renewal_rate,
            rm.chunk_step,
            rm.chunk_horizon,
            rm.chunk_inference_latency,
        )


class WeightOnlyQuantizationScenario(BaseScenario):
    """One backbone priced at several weight formats against measurements of the same formats.

    ``formats`` maps a label to its evaluation scenario; ``measured`` and
    ``scores`` map the same labels to a measured latency and a task score.
    ``group_size``, ``scale_bits``, and ``weight_bits`` price the per-group
    scale overhead of the deployed format.

    Notes
    -----
    Equation: relative gap = (modeled - measured latency) / measured latency; scale
    overhead = scale bits / (group size * weight bits).

    Assumptions: Every label names the same workload and hardware configuration across
    modeled and measured mappings; group size and bit widths are positive.

    Excludes: A universal quantization speedup, statistical uncertainty in measurements,
    and automatic quality validation.

    References
    ----------
    Derivation: relative gap = (modeled - measured latency)
    / measured latency; scale overhead = scale bits / (group size * weight bits). The
    individual properties state units and intermediate accounting; this class introduces
    no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.OpenVlaOrinQuantization
    >>> round(float(case.measured_speedup), 3)
    2.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    formats: Dict[str, VlaEvaluationScenario]
    measured: Dict[str, Quantity]
    scores: Dict[str, float]
    deployed: str  # label of the deployed format
    reference: str  # label of the 16-bit reference format
    group_size: int
    scale_bits: int
    weight_bits: int

    def computed(self, label: str) -> Quantity:
        """Modeled total evaluation latency for the named format, with time units.

        Equation: ``return formats[label].result.t_total``."""
        return self.formats[label].result.t_total

    def gap(self, label: str) -> float:
        """Signed fractional latency error relative to the supplied measurement for the same
        label.

        Equation: ``return ((computed(label) - measured[label]) /
        measured[label]).to('').magnitude``."""
        return ((self.computed(label) - self.measured[label]) / self.measured[label]).to("").magnitude

    @property
    def measured_speedup(self) -> float:
        """Reference measured latency divided by deployed measured latency; inputs are
        source-reported data.

        Equation: ``return (measured[reference] / measured[deployed]).to('').magnitude``."""
        return (self.measured[self.reference] / self.measured[self.deployed]).to("").magnitude

    @property
    def score_drop(self) -> float:
        """Reference task score minus deployed task score, in the supplied score's units.

        Equation: ``return scores[reference] - scores[deployed]``."""
        return self.scores[self.reference] - self.scores[self.deployed]

    @property
    def scale_overhead(self) -> float:
        """Scale-storage bits per weight divided by payload bits per weight, a fractional
        overhead.

        Equation: ``return scale_bits / group_size / weight_bits``."""
        return self.scale_bits / self.group_size / self.weight_bits
