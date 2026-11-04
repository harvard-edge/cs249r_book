"""Training work, state, updates, and parallel execution.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
import math as _math
from typing import Dict, List, Optional
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...systems.types import Fleet
from ...datasets.types import DatasetProfile
from ...physics.layers import calc_separable_cost_fraction as _separable_fraction
from .base import BaseScenario
from .optimization import SelfContainedReplicaScenario


class TrainingRunScenario(BaseScenario):
    """Training ``workload`` on ``fleet``: peak time to delivered wall-clock time.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TrainingRunModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.TrainingRunModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TrainingRunModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TrainingRunModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3PretrainingFrontier8K
    >>> type(case.result).__name__
    'TrainingRunResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    fleet: Fleet
    precision: str = "fp8"
    local_efficiency: float
    scaling_efficiency: float
    overheads: Dict[str, float]  # applied in order, each to the subtotal before it

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TrainingRunModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import TrainingRunModel

        return TrainingRunModel().solve(
            self.workload, self.fleet, self.local_efficiency, self.scaling_efficiency, self.overheads, self.precision
        )


class TrainingStateScenario(BaseScenario):
    """Persistent mixed-precision Adam state of ``workload`` and its shards.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TrainingStateModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.TrainingStateModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TrainingStateModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TrainingStateModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3TrainingStateTp8
    >>> type(case.result).__name__
    'TrainingStateResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tp_size: int
    dp_size: int
    activations: Quantity  # stated total for the whole model

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TrainingStateModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import TrainingStateModel

        return TrainingStateModel().solve(self.workload, self.tp_size, self.dp_size, self.activations)


class SustainedThroughputScenario(BaseScenario):
    """A run budgeted from peak throughput, then run at its sustained share of peak.

    Device-hours scale as ``1 / sustained_fraction``: a plan made from the
    datasheet rate needs that many times the hours when the workload sustains
    only part of peak.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: The work budget is fixed and the positive sustained fraction scales
    delivered arithmetic capacity uniformly.

    Excludes: Changes to algorithmic work and time-varying utilization are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PeakPlannedTrainingBudget
    >>> round(case.required_device_hours, 4)
    2500.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    planned_device_hours: float
    sustained_fraction: float

    @property
    def required_device_hours(self) -> float:
        """Device-time needed when only a fraction of peak is delivered. Equation:
        ``planned_device_hours / sustained_fraction``. Units: device-hours; the fraction must be
        positive.
        """
        return self.planned_device_hours / self.sustained_fraction

    @property
    def stretch(self) -> float:
        """Run-time multiplier relative to the peak-rate plan. Equation: ``1 /
        sustained_fraction``. Units: dimensionless; total work is fixed.
        """
        return 1 / self.sustained_fraction


class ScalingOverheadScenario(BaseScenario):
    """A planned speedup when communication and coordination take a share of the scaled step.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: The overhead fraction describes the scaled step, so the useful fraction
    multiplies the ideal speedup.

    Excludes: An overhead fraction measured against baseline time cannot be substituted
    unchanged.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.CoordinationOverheadExample
    >>> round(case.realized_speedup, 4)
    1.8
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    planned_speedup: float
    overhead_fraction: float

    @property
    def realized_speedup(self) -> float:
        """Planned speedup reduced by the scaled step's overhead share. Equation: ``planned_speedup
        * (1 - overhead_fraction)``. Units: dimensionless; the overhead fraction belongs to the
        new step, not the baseline.
        """
        return self.planned_speedup * (1 - self.overhead_fraction)


class PowerLawScenario(BaseScenario):
    """A power-law term ``A R^-alpha`` and the resource multiple that halves it.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.scaling.calc_power_law_factor``,
    ``mlsysim.physics.scaling.calc_halving_multiple``,
    ``mlsysim.physics.scaling.calc_loss_reduction_compute_multiple``.

    Assumptions: The reducible term obeys a fixed positive power-law exponent; optional
    resource and reduction inputs must be supplied for their properties.

    Excludes: Irreducible loss floors and extrapolation failure are not modeled.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.scaling.calc_power_law_factor``,
    ``mlsysim.physics.scaling.calc_halving_multiple``,
    ``mlsysim.physics.scaling.calc_loss_reduction_compute_multiple``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.KaplanTenPercentLowerLoss
    >>> round(case.halving_multiple, 4)
    1048576.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    exponent: float
    # Optional (added 2026-10-20): a fractional reduction of the term; the
    # resource multiple that buys it is ``reduction_multiple``.
    loss_reduction: Optional[float] = None
    # Optional (added 2026-10-20): a resource multiple; the term falls to
    # ``multiple_factor`` of its value, a fractional reduction ``multiple_reduction``.
    resource_multiple: Optional[float] = None

    @property
    def multiple_factor(self) -> float:
        """Remaining power-law term after increasing the resource. Equation:
        ``mlsysim.physics.scaling.calc_power_law_factor(resource_multiple, exponent)``. Units:
        dimensionless; the exponent and power-law regime are fixed.
        """
        from ...physics.scaling import calc_power_law_factor

        return calc_power_law_factor(self.resource_multiple, self.exponent)

    @property
    def multiple_reduction(self) -> float:
        """Fraction removed from the original power-law term. Equation: ``1 - multiple_factor``.
        Units: dimensionless; any irreducible additive floor is excluded.
        """
        return 1 - self.multiple_factor

    @property
    def halving_multiple(self) -> float:
        """Resource multiplier needed to halve the reducible power-law term. Equation:
        ``mlsysim.physics.scaling.calc_halving_multiple(exponent)``. Units: dimensionless;
        requires a positive exponent.
        """
        from ...physics.scaling import calc_halving_multiple

        return calc_halving_multiple(self.exponent)

    @property
    def reduction_multiple(self) -> float:
        """Resource multiplier for the requested reduction in the power-law term. Equation:
        ``mlsysim.physics.scaling.calc_loss_reduction_compute_multiple(exponent,
        loss_reduction)``. Units: dimensionless; the reduction must remain within the helper's
        valid range.
        """
        from ...physics.scaling import calc_loss_reduction_compute_multiple

        return calc_loss_reduction_compute_multiple(self.exponent, self.loss_reduction)


class PublishedRunScenario(BaseScenario):
    """A published training run: its total compute spread over its accelerators and days.

    Optional (added 2026-10-20): a run reported as ``accelerator_hours`` (a
    model card's GPU hours) at ``device_power`` per accelerator. Its compute is
    the workload's ``training_ops`` or, when unset, 6PD from its training tokens.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.transformer.calc_transformer_training_flops``.

    Assumptions: Published workload fields and externally supplied accelerator-time describe
    the same training run; the fallback work estimate uses parameter-token accounting.

    Excludes: Reconstructed energy is not meter data, and missing reference run fields are
    not inferred.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.transformer.calc_transformer_training_flops``. The methods
    below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3PublishedRun
    >>> str(case.training_operations)
    '3.14×10²³ flop'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    accelerator: HardwareNode  # the part the run used, for its peak rate
    accelerator_hours: Optional[Quantity] = None
    device_power: Optional[Quantity] = None

    @property
    def training_operations(self) -> Quantity:
        """Recorded training work, falling back to parameter-token accounting when absent.
        Equation: stored ``training_ops`` or
        ``mlsysim.physics.transformer.calc_transformer_training_flops(parameters,
        training_tokens)``. Units: FLOPs; the fallback is modeled, not measured.
        """
        w = self.workload
        if w.training_ops is not None:
            return w.training_ops
        from ...physics.transformer import calc_transformer_training_flops

        return calc_transformer_training_flops(w.parameters, w.training_tokens).to("flop")

    @property
    def achieved_rate(self) -> Quantity:
        """Training work amortized over accelerator-time. Equation: work divided by supplied
        accelerator-hours, otherwise recorded work divided by reference accelerator count and
        run duration. Units: TFLOP/s per accelerator; the fallback requires recorded training
        work.
        """
        w = self.workload
        if self.accelerator_hours is not None:
            return (self.training_operations / self.accelerator_hours).to("TFLOP/s")
        n = w.training_accelerators_ref.to("count").magnitude
        return (w.training_ops / (n * w.training_days_ref)).to("TFLOP/s")

    @property
    def energy(self) -> Quantity:
        """Energy implied by supplied accelerator-time and constant device power. Equation:
        ``accelerator_hours * device_power``. Units: GWh; facility overhead and varying power
        are excluded.
        """
        return (self.accelerator_hours * self.device_power).to("GWh")

    @property
    def peak_fraction(self) -> float:
        """Achieved per-accelerator arithmetic rate relative to catalog peak. Equation:
        ``achieved_rate / accelerator.compute.peak_flops``. Units: dimensionless; the peak
        precision must match the interpretation of training work.
        """
        return (self.achieved_rate / self.accelerator.compute.peak_flops).to("").magnitude


class ParallelLayoutScenario(BaseScenario):
    """``workload`` trained on a TP x PP x DP layout: step time, efficiency, and state per
    device.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ParallelLayoutModel``.

    Assumptions: Tensor, pipeline, and data parallelism form a rectangular product layout,
    and the solver's precision and overlap inputs apply uniformly.

    Excludes: The scenario does not independently validate physical placement or execution
    schedules.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ParallelLayoutModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3DataParallel1K
    >>> round(case.n_devices, 4)
    1024
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int
    microbatches: int = 1
    batch_tokens: float  # tokens per optimizer update
    flops_per_param_token: float
    accelerator: HardwareNode
    precision: str = "fp8"  # the compute precision whose peak sets the window
    local_efficiency: float  # share of peak before communication
    link_bandwidth: Quantity  # per-device data-parallel link
    overlap_fraction: float  # share of the AllReduce that may hide under compute
    gradient_precision: str = "fp16"
    window_resolution: Optional[Quantity] = None  # carry the compute window at this precision
    device_memory: Optional[Quantity] = None  # for the fewest devices that hold the state

    @property
    def n_devices(self) -> int:
        """Accelerator count consumed by the parallel layout. Equation: ``tp_size * pp_size *
        dp_size``. Units: devices; the three parallel dimensions form a rectangular layout.
        """
        return self.tp_size * self.pp_size * self.dp_size

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ParallelLayoutModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import ParallelLayoutModel

        return ParallelLayoutModel().solve(
            self.workload,
            self.tp_size,
            self.pp_size,
            self.dp_size,
            self.microbatches,
            self.batch_tokens,
            self.flops_per_param_token,
            self.accelerator.compute.precision_flops[self.precision],
            self.local_efficiency,
            self.link_bandwidth,
            self.overlap_fraction,
            self.gradient_precision,
            self.window_resolution,
            self.device_memory,
        )


class ActivationMemoryScenario(BaseScenario):
    """Activation memory of ``workload`` at a microbatch, under each recomputation strategy.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.memory.KORTHIKANTI_PER_TOKEN_BYTES``,
    ``mlsysim.physics.memory.KORTHIKANTI_SCORE_BYTES``,
    ``mlsysim.physics.memory.calc_checkpoint_memory_units``,
    ``mlsysim.physics.memory.calc_activation_memory``,
    ``mlsysim.physics.transformer.calc_transformer_training_flops``,
    ``mlsysim.physics.transformer.calc_transformer_decode_flops``.

    Assumptions: Transformer layers share geometry. Aggregate Korthikanti-style coefficients
    and explicit tensor line items are separate approximations; segment sizes use integer
    division.

    Excludes: Allocator reserves, runtime workspaces, unequal layer shapes, and leftover
    segment layers are absent from the simple segment estimate.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.memory.KORTHIKANTI_PER_TOKEN_BYTES``,
    ``mlsysim.physics.memory.KORTHIKANTI_SCORE_BYTES``,
    ``mlsysim.physics.memory.calc_checkpoint_memory_units``,
    ``mlsysim.physics.memory.calc_activation_memory``,
    ``mlsysim.physics.transformer.calc_transformer_training_flops``,
    ``mlsysim.physics.transformer.calc_transformer_decode_flops``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3ActivationsMicrobatch4
    >>> round(case.sequence, 4)
    2048
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    micro_batch: int
    seq_len: Optional[int] = None  # defaults to the model's context length
    # Optional (added 2026-10-20): bytes per stored activation (2 for 16-bit),
    # and a stated normalization buffer per layer for the per-layer line items.
    precision_bytes: float = 2
    normalization_per_layer: Optional[Quantity] = None
    ffn_multiple: int = 4
    # Optional (added 2026-10-20): checkpoint the layers in this many equal
    # segments, keeping each segment's input and rebuilding one segment at a
    # time in the backward pass.
    checkpoint_segments: Optional[int] = None

    @property
    def sequence(self) -> int:
        """Context length used by the activation model. Equation: explicit ``seq_len`` when
        present, otherwise ``workload.context_length``. Units: tokens per sequence.
        """
        return self.seq_len if self.seq_len is not None else self.workload.context_length

    # Korthikanti et al. Table 2 per layer, sbh(34 + 5as/h) (added 2026-10-20).
    @property
    def per_token_bytes_coefficient(self) -> float:
        """Linear activation coefficient scaled to the requested element width. Equation:
        ``KORTHIKANTI_PER_TOKEN_BYTES * precision_bytes / 2``. Units: bytes per hidden element;
        the reference coefficient is normalized to two-byte storage.
        """
        from ...physics.memory import KORTHIKANTI_PER_TOKEN_BYTES

        return KORTHIKANTI_PER_TOKEN_BYTES * self.precision_bytes / 2

    @property
    def score_bytes_coefficient(self) -> float:
        """Attention-score coefficient expressed relative to the linear hidden-state term.
        Equation: ``KORTHIKANTI_SCORE_BYTES * heads * sequence / hidden_dim * precision_bytes /
        2``. Units: bytes per hidden element under the same activation accounting convention.
        """
        from ...physics.memory import KORTHIKANTI_SCORE_BYTES

        w = self.workload
        return KORTHIKANTI_SCORE_BYTES * w.heads * self.sequence / w.hidden_dim * self.precision_bytes / 2

    @property
    def score_share(self) -> float:
        """Attention-score fraction of the unrecomputed activation estimate. Equation: score
        coefficient divided by the sum of linear and score coefficients. Units: dimensionless;
        all shared shape factors cancel.
        """
        c = self.score_bytes_coefficient
        return c / (self.per_token_bytes_coefficient + c)

    @property
    def per_sequence_activations(self) -> Quantity:
        """Selective-recomputation activation footprint per sequence. Equation:
        ``per_token_activations / micro_batch``. Units: GB; the property name refers to sequence
        allocation, not a single token.

        Per-token activations of one sequence (scores not stored): the slope against batch size.
        """
        return (self.per_token_activations / self.micro_batch).to("GB")

    # Segment checkpointing (added 2026-10-20).
    @property
    def segment_layers(self) -> int:
        """Integer number of layers within each checkpoint segment. Equation: ``workload.layers //
        checkpoint_segments``. Units: layers; leftover layers are not represented by this floor-
        division estimate.
        """
        return self.workload.layers // self.checkpoint_segments

    @property
    def segment_input(self) -> Quantity:
        """One segment boundary's retained hidden-state tensor. Equation: batch times sequence
        times hidden width times element bytes via ``_elements``. Units: MB; other boundary
        state is excluded.
        """
        return Q_(self._elements(self.workload.hidden_dim) * self.precision_bytes, "byte").to("MB")

    @property
    def segment_inputs(self) -> Quantity:
        """Retained inputs across all checkpoint segments. Equation: ``checkpoint_segments *
        segment_input``. Units: GB; the same tensor shape is assumed at each boundary.
        """
        return (self.checkpoint_segments * self.segment_input).to("GB")

    @property
    def live_segment(self) -> Quantity:
        """Linear activation footprint of the currently live segment. Equation: full linear
        footprint divided by model layers times segment layers. Units: GB; segments are treated
        as equal-sized.

        One segment's per-token activations, rebuilt in the backward pass (scores not stored).
        """
        return (self.per_token_activations / self.workload.layers * self.segment_layers).to("GB")

    @property
    def checkpointed_activations(self) -> Quantity:
        """Boundary storage plus live-segment activations. Equation: ``segment_inputs +
        live_segment``. Units: GB; runtime workspaces and uneven segment effects are excluded.

        Segment inputs plus one live rebuilt segment (its input counted twice: conservative).
        """
        return (self.segment_inputs + self.live_segment).to("GB")

    def checkpoint_memory_units(self, checkpoints: int) -> float:
        """Normalized checkpoint storage at the requested checkpoint count. Equation:
        ``mlsysim.physics.memory.calc_checkpoint_memory_units(workload.layers, checkpoints)``.
        Units: layer-equivalent activation storage.

        Activation memory with ``checkpoints`` evenly spaced checkpoints, in layers: k + L/k.
        """
        from ...physics.memory import calc_checkpoint_memory_units

        return calc_checkpoint_memory_units(self.workload.layers, checkpoints)

    def activations(self, strategy: str) -> Quantity:
        """Activation footprint for the named recomputation strategy. Equation:
        ``mlsysim.physics.memory.calc_activation_memory`` with scenario layers, sequence,
        microbatch, width, heads, and element bytes. Units: bytes; the helper owns the
        approximation and valid strategy names.
        """
        from ...physics.memory import calc_activation_memory

        w = self.workload
        return calc_activation_memory(
            w.layers,
            self.sequence,
            self.micro_batch,
            w.hidden_dim,
            n_heads=w.heads,
            precision_bytes=self.precision_bytes,
            strategy=strategy,
        )

    @property
    def per_token_activations(self) -> Quantity:
        """Linear-in-sequence activation footprint with attention-score recomputation. Equation:
        ``activations('selective')``. Units: GB; despite the name this is the whole microbatch,
        not per-token storage.

        Activations that grow in proportion to sequence length (everything but the attention
        scores).
        """
        return self.activations("selective").to("GB")

    @property
    def score_activations(self) -> Quantity:
        """Stored attention-score contribution removed by selective recomputation. Equation:
        ``activations('none') - activations('selective')``. Units: GB; both branches use
        identical workload geometry.
        """
        return (self.activations("none") - self.activations("selective")).to("GB")

    # Per-layer line items (added 2026-10-20): the tensors a reader can name.
    def _elements(self, width) -> float:
        return self.micro_batch * self.sequence * width

    @property
    def projection_activations_per_layer(self) -> Quantity:
        """Illustrative saved state for four hidden-width projection tensors. Equation: ``4 *
        _elements(hidden_dim) * precision_bytes``. Units: MB per layer; this line-item view is
        distinct from the aggregate coefficient model.
        """
        return Q_(4 * self._elements(self.workload.hidden_dim) * self.precision_bytes, "byte").to("MB")

    @property
    def ffn_activations_per_layer(self) -> Quantity:
        """Illustrative expanded feed-forward activation tensor. Equation: ``_elements(ffn_multiple
        * hidden_dim) * precision_bytes``. Units: MB per layer; the supplied expansion factor
        sets the intermediate width.
        """
        return Q_(self._elements(self.ffn_multiple * self.workload.hidden_dim) * self.precision_bytes, "byte").to("MB")

    @property
    def score_activations_per_layer(self) -> Quantity:
        """Average stored attention-score footprint per layer. Equation: ``score_activations /
        workload.layers``. Units: MB; layers share the same geometry.
        """
        return (self.score_activations / self.workload.layers).to("MB")

    @property
    def line_items_per_layer(self) -> Quantity:
        """Sum of the explicit projection, feed-forward, score, and optional normalization tensors.
        Equation: add these line items, using zero for unspecified normalization. Units: MB per
        layer; this is not forced to equal the separate aggregate coefficient estimate.
        """
        norm = self.normalization_per_layer if self.normalization_per_layer is not None else Q_(0, "MB")
        return (
            self.projection_activations_per_layer
            + self.ffn_activations_per_layer
            + self.score_activations_per_layer
            + norm
        ).to("MB")

    @property
    def line_items_total(self) -> Quantity:
        """Explicit line-item footprint across the model. Equation: ``line_items_per_layer *
        workload.layers``. Units: GB; identical layer shapes are assumed.
        """
        return (self.line_items_per_layer * self.workload.layers).to("GB")

    @property
    def forward_equivalents(self) -> float:
        """Training arithmetic expressed in forward-pass work units. Equation:
        ``calc_transformer_training_flops(P, 1) / calc_transformer_decode_flops(P, 1)``. Units:
        dimensionless; parameter-dominated accounting omits context attention.
        """
        from ...physics.transformer import calc_transformer_training_flops, calc_transformer_decode_flops

        p = self.workload.parameters
        return (calc_transformer_training_flops(p, 1) / calc_transformer_decode_flops(p, 1)).to("").magnitude

    @property
    def recompute_forward_equivalents(self) -> float:
        """Full-recomputation training arithmetic in forward-pass work units. Equation:
        ``forward_equivalents * (1 + recompute_overhead)``. Units: dimensionless; the extra work
        is a recomputed forward pass.
        """
        return self.forward_equivalents * (1 + self.recompute_overhead)

    @property
    def recompute_overhead(self) -> float:
        """Relative arithmetic increase from full activation recomputation. Equation: checkpointed
        training FLOPs divided by baseline training FLOPs, minus one. Units: dimensionless; this
        is a work ratio, not measured runtime overhead.
        """
        from ...physics.transformer import calc_transformer_training_flops

        full = calc_transformer_training_flops(self.workload.parameters, 1, activation_checkpointing=True)
        base = calc_transformer_training_flops(self.workload.parameters, 1)
        return (full / base).to("").magnitude - 1


class ZeroShardingScenario(BaseScenario):
    """``workload``'s training state per device at each ZeRO stage over ``n_devices``.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ZeroShardingModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ZeroShardingModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ZeroShardingModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ZeroShardingModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_7B_ZeroStages64
    >>> type(case.result).__name__
    'ZeroShardingResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    n_devices: int
    activations: Optional[ActivationMemoryScenario] = None
    activation_strategy: str = "full"
    link_bandwidth: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ZeroShardingModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import ZeroShardingModel

        act = None if self.activations is None else self.activations.activations(self.activation_strategy)
        return ZeroShardingModel().solve(self.workload, self.n_devices, act, self.link_bandwidth)


class DataParallelScaleOutScenario(BaseScenario):
    """A data-parallel job of ``workload`` on ``nodes`` nodes, from a one-device profile.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.DataParallelScaleOutModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.DataParallelScaleOutModel``; its documented validity regime, units,
    and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.DataParallelScaleOutModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.DataParallelScaleOutModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt2DataParallelOneNode
    >>> type(case.result).__name__
    'DataParallelScaleOutResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    precision: str = "fp16"
    step_time: Quantity  # one device, one local step
    training_time: Quantity  # one device, whole run
    per_device_batch: int
    devices_per_node: int
    nodes: int = 1
    intra_bandwidth: Quantity
    inter_bandwidth: Optional[Quantity] = None
    accumulation_steps: int = 1
    node_hour_rate: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.DataParallelScaleOutModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import DataParallelScaleOutModel

        return DataParallelScaleOutModel().solve(
            self.workload,
            self.precision,
            self.step_time,
            self.training_time,
            self.per_device_batch,
            self.devices_per_node,
            self.nodes,
            self.intra_bandwidth,
            self.inter_bandwidth,
            self.accumulation_steps,
            self.node_hour_rate,
        )


class PipelineScheduleScenario(BaseScenario):
    """A synchronous pipeline of ``stages`` stages fed ``microbatches`` microbatches per step.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.performance.calc_pipeline_bubble``.

    Assumptions: All pipeline stages take equal time and use the noninterleaved synchronous
    fill/drain abstraction.

    Excludes: Communication, stage imbalance, and interleaved virtual stages are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.performance.calc_pipeline_bubble``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.GPipeFourStageWalkthrough
    >>> round(case.bubble, 4)
    0.4286
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    stages: int
    microbatches: int

    @property
    def bubble(self) -> float:
        """Idle fraction of the balanced synchronous pipeline schedule. Equation:
        ``mlsysim.physics.performance.calc_pipeline_bubble(stages, microbatches)``. Units:
        dimensionless; imbalance and communication are excluded.
        """
        from ...physics.performance import calc_pipeline_bubble

        return calc_pipeline_bubble(self.stages, self.microbatches)

    @property
    def slots(self) -> int:
        """Total forward-plus-backward scheduling slots. Equation: ``2 * (microbatches + stages -
        1)``. Units: equal-duration slots; stages are balanced.
        """
        return 2 * (self.microbatches + self.stages - 1)

    @property
    def idle_slots(self) -> int:
        """Fill/drain slots without useful work at a pipeline stage. Equation: ``2 * (stages -
        1)``. Units: slots; this is the noninterleaved schedule.
        """
        return 2 * (self.stages - 1)

    @property
    def fill_slots(self) -> int:
        """Initial pipeline fill depth. Equation: ``stages - 1``. Units: slots; each stage consumes
        one slot.
        """
        return self.stages - 1

    @property
    def phase_slots(self) -> int:
        """Slots in one forward or backward phase. Equation: ``microbatches + stages - 1``. Units:
        slots under the balanced-stage abstraction.
        """
        return self.microbatches + self.stages - 1

    def microbatches_for_bubble(self, target: float) -> float:
        """Continuous microbatch count meeting the target bubble fraction. Equation: ``(stages - 1)
        * (1 - target) / target``. Units: microbatches; callers must round up for an integer
        schedule and use a positive target.
        """
        return (self.stages - 1) * (1 - target) / target


class TensorParallelLayerScenario(BaseScenario):
    """One layer of ``workload`` split ``tp_size`` ways: compute against the exchange on each
    link.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.TensorParallelLayerModel``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.solvers.TensorParallelLayerModel``; its documented validity regime, units, and
    modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.TensorParallelLayerModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.TensorParallelLayerModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3TensorParallelLayerTp8
    >>> type(case.result).__name__
    'TensorParallelLayerResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tp_size: int
    tokens: int
    accelerator: HardwareNode
    precision: str = "fp16_dense"
    local_efficiency: float
    links: Dict[str, Quantity]
    collectives_per_layer: int = 2

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.TensorParallelLayerModel.solve``.
        Equation: pass the stated fields to that implementation; its contract defines the
        calculation and feasibility tests. Units: each returned field follows the referenced
        result schema; no independent equation is evaluated by this wrapper.
        """
        from ..solvers import TensorParallelLayerModel

        return TensorParallelLayerModel().solve(
            self.workload,
            self.tp_size,
            self.tokens,
            self.accelerator.compute.precision_flops[self.precision],
            self.local_efficiency,
            self.links,
            self.collectives_per_layer,
        )


class AlignmentMemoryScenario(BaseScenario):
    """PPO and DPO alignment of ``policy`` with ``critic``-sized reward and value models.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AlignmentMemoryModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.AlignmentMemoryModel``;
    its documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.AlignmentMemoryModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AlignmentMemoryModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_70B_PpoAlignment
    >>> type(case.result).__name__
    'AlignmentMemoryResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    policy: Workload
    critic: Workload
    tp_size: int
    pp_size: int
    dp_size: int = 1
    rollout_sequences: int
    rollout_tokens: int
    device_memory: Optional[Quantity] = None

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AlignmentMemoryModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import AlignmentMemoryModel

        return AlignmentMemoryModel().solve(
            self.policy,
            self.critic,
            self.tp_size,
            self.pp_size,
            self.rollout_sequences,
            self.rollout_tokens,
            self.device_memory,
        )


class ReferenceTrainingStepScenario(BaseScenario):
    """One optimizer step of a reference pretraining job, fixed once for every chapter that quotes it.

    The layout is ``replica`` (tensor parallelism across one node, the shallowest
    pipeline that fits unsharded training state) repeated across ``fleet`` for
    data parallelism. One step is one optimizer update over the global batch,
    split into microbatches of ``sequences_per_microbatch`` sequences, so the
    data-parallel gradient reduction happens once per step. The global batch is
    the cited token batch divided by the context length, rounded down to a
    multiple of ``batch_divisor`` so every data-parallel degree used in the
    book divides it. Step time is modeled by ``DistributedModel``, not measured.

    Notes
    -----
    Equation: global sequences = rounded token batch / context; data degree = fleet
    devices / replica devices; step time = DistributedModel.solve(...).

    Assumptions: The declared replica layout and global batch admit whole microbatches.
    The local efficiency and communication-overlap choice are inputs.

    Excludes: A measured step-time claim, convergence prediction, and incomplete replica
    or microbatch allocations.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.DistributedModel.solve``. The solver contract records the equations
    and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DenseLLMPretrainingStep
    >>> round(float(case.tensor_parallel), 3)
    8.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    replica: SelfContainedReplicaScenario
    fleet: Fleet
    batch_tokens: float
    batch_divisor: int
    sequences_per_microbatch: int
    precision: str
    local_efficiency: float
    overlap_comm: bool

    @property
    def tensor_parallel(self) -> int:
        """Tensor-parallel degree inherited from the self-contained replica layout.

        Equation: ``return replica.tensor_parallel``."""
        return self.replica.tensor_parallel

    @property
    def pipeline_depth(self) -> int:
        """Pipeline stage count inherited from the self-contained replica layout.

        Equation: ``return replica.pipeline_depth``."""
        return self.replica.pipeline_depth

    @property
    def data_parallel(self) -> int:
        """Whole replicas that fit in the fleet; incomplete replicas are discarded.

        Equation: ``return fleet.total_accelerators // replica.replica_gpus``."""
        return self.fleet.total_accelerators // self.replica.replica_gpus

    @property
    def context_length(self) -> int:
        """Tokens per sequence taken from the replica's workload specification.

        Equation: ``return replica.workload.context_length``."""
        return self.replica.workload.context_length

    @property
    def global_batch_sequences(self) -> int:
        """Global sequence count rounded down to a multiple of ``batch_divisor``.

        Equation: ``return int(batch_tokens / context_length) // batch_divisor *
        batch_divisor``."""
        return int(self.batch_tokens / self.context_length) // self.batch_divisor * self.batch_divisor

    @property
    def global_batch_tokens(self) -> int:
        """Actual token count per optimizer step after sequence-count rounding.

        Equation: ``return global_batch_sequences * context_length``."""
        return self.global_batch_sequences * self.context_length

    @property
    def microbatches_per_replica(self) -> int:
        """Whole microbatches in each replica's sequence allocation.

        Equation: ``return global_batch_sequences // (data_parallel *
        sequences_per_microbatch)``."""
        return self.global_batch_sequences // (self.data_parallel * self.sequences_per_microbatch)

    @property
    def state_per_gpu(self) -> Quantity:
        """Unsharded training-state memory per accelerator, excluding activations and
        workspace.

        Equation: ``return replica.workload.parameters.magnitude * replica.bytes_per_param
        / replica.replica_gpus``."""
        return self.replica.workload.parameters.magnitude * self.replica.bytes_per_param / self.replica.replica_gpus

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.DistributedModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.DistributedModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import DistributedModel

        return DistributedModel().solve(
            self.replica.workload,
            self.fleet,
            batch_size=self.global_batch_sequences,
            precision=self.precision,
            efficiency=self.local_efficiency,
            tp_size=self.tensor_parallel,
            pp_size=self.pipeline_depth,
            microbatch_count=self.microbatches_per_replica,
            overlap_comm=self.overlap_comm,
            seq_len=self.context_length,
        )

    @property
    def step_time(self) -> Quantity:
        """Modeled optimizer-step latency from ``DistributedModel.solve``, carrying time
        units.

        Equation: ``return result.step_latency_total``."""
        return self.result.step_latency_total


class UpdateWorkingSetScenario(BaseScenario):
    """A full on-device update with Adam: its working set and the DRAM bytes of one step.

    Gradients equal the weights; the optimizer keeps ``optimizer_moments``
    weight-sized arrays; saved activations are ``activation_factor`` times the
    weight bytes (``activation_factor_high`` bounds the quoted range).

    Notes
    -----
    Equation: Working set=weights+gradients+optimizer_state+saved_activations; traffic delegates to calc_update_step_traffic.

    Assumptions: Gradients use weight width; optimizer moments are weight-sized arrays; activation factors are declared inputs.

    Excludes: Master-weight precision differences, allocator reserve, and runtime workspace.

    References
    ----------
    Implementation references: ``mlsysim.physics.calc_update_step_traffic``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PhoneFullUpdateWorkingSet
    >>> round(scenario.amplification, 6)
    9.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    parameters: float
    bytes_per_param: Quantity
    optimizer_moments: int
    activation_factor: float
    activation_factor_high: float
    bandwidth: Optional[Quantity] = None  # device DRAM bandwidth, for the traffic floor

    @property
    def weights(self) -> Quantity:
        """Parameter storage in decimal MB: parameter count times the declared bytes per parameter.

        Equation: (self.parameters * self.bytes_per_param).to('MB')
        """
        return (self.parameters * self.bytes_per_param).to("MB")

    @property
    def gradients(self) -> Quantity:
        """Gradient storage in MB, assumed equal to weight storage in this scenario.

        Equation: self.weights
        """
        return self.weights

    @property
    def optimizer_state(self) -> Quantity:
        """Optimizer-state MB: ``optimizer_moments`` weight-sized arrays; storage widths are not inferred from a runtime optimizer.

        Equation: (self.optimizer_moments * self.weights).to('MB')
        """
        return (self.optimizer_moments * self.weights).to("MB")

    @property
    def activations(self) -> Quantity:
        """Saved activation MB under the supplied activation-to-weight multiplier.

        Equation: (self.activation_factor * self.weights).to('MB')
        """
        return (self.activation_factor * self.weights).to("MB")

    @property
    def persistent(self) -> Quantity:
        """Activation-free MB: weights plus gradients plus optimizer state.

        Equation: (self.weights + self.gradients + self.optimizer_state).to('MB')
        """
        return (self.weights + self.gradients + self.optimizer_state).to("MB")

    @property
    def total(self) -> Quantity:
        """Working-set MB: persistent state plus the assumed saved activations.

        Equation: (self.persistent + self.activations).to('MB')
        """
        return (self.persistent + self.activations).to("MB")

    @property
    def amplification(self) -> float:
        """Total working set divided by weight bytes, a dimensionless multiplier.

        Equation: (self.total / self.weights).to('').magnitude
        """
        return (self.total / self.weights).to("").magnitude

    @property
    def floor(self) -> float:
        """Persistent-state-to-weight ratio before any saved activations are added.

        Equation: (self.persistent / self.weights).to('').magnitude
        """
        return (self.persistent / self.weights).to("").magnitude

    @property
    def ceiling(self) -> float:
        """Upper working-set-to-weight ratio using ``activation_factor_high`` rather than the nominal activation multiplier.

        Equation: self.floor + self.activation_factor_high
        """
        return self.floor + self.activation_factor_high

    @property
    def step_traffic(self) -> Quantity:
        """DRAM bytes per update in MB from ``calc_update_step_traffic``; uses its declared read/write accounting, not a measured trace.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics import calc_update_step_traffic

        traffic, _ = calc_update_step_traffic(self.weights, self.gradients, self.optimizer_state, self.activations)
        return traffic.to("MB")

    @property
    def weight_reads(self) -> Quantity:
        """Weight-read portion of the update traffic in MB, returned by ``calc_update_step_traffic``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics import calc_update_step_traffic

        _, reads = calc_update_step_traffic(self.weights, self.gradients, self.optimizer_state, self.activations)
        return reads.to("MB")

    @property
    def traffic_ratio(self) -> float:
        """Update-step traffic divided by weight storage; dimensionless bytes-moved amplification.

        Equation: (self.step_traffic / self.weights).to('').magnitude
        """
        return (self.step_traffic / self.weights).to("").magnitude

    @property
    def weight_read_share(self) -> float:
        """Fraction of update traffic attributed to the helper's weight-read component.

        Equation: (self.weight_reads / self.step_traffic).to('').magnitude
        """
        return (self.weight_reads / self.step_traffic).to("").magnitude

    @property
    def step_floor(self) -> Quantity:
        """Traffic-only latency floor in ms: ``step_traffic / bandwidth``; requires bandwidth and excludes compute/overlap effects.

        Equation: (self.step_traffic / self.bandwidth).to('millisecond')
        """
        return (self.step_traffic / self.bandwidth).to("millisecond")


class LowRankUpdateScenario(BaseScenario):
    """A rank-``rank`` update to one ``out_dim`` by ``in_dim`` weight matrix.

    Notes
    -----
    Equation: Dense parameters=``in_dim*out_dim``; rank-r factors contain ``rank*(in_dim+out_dim)`` parameters.

    Assumptions: Bias-free matrix and positive rank/dimensions; byte properties require bytes_per_param.

    Excludes: Accuracy effects, optimizer state, and the frozen backbone when interpreting update_bytes.

    References
    ----------
    Implementation references: ``mlsysim.physics.calc_low_rank_update_params``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.TransformerLayerRank4Update
    >>> scenario.update_params
    6144
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    in_dim: int
    out_dim: int
    rank: int

    @property
    def full_params(self) -> int:
        """Dense matrix parameter count: output width times input width, excluding bias.

        Equation: self.in_dim * self.out_dim
        """
        return self.in_dim * self.out_dim

    @property
    def update_params(self) -> int:
        """Parameter count in two rank-r factors via ``calc_low_rank_update_params``: ``rank * (in_dim + out_dim)``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics import calc_low_rank_update_params

        return calc_low_rank_update_params(self.in_dim, self.out_dim, self.rank)

    @property
    def reduction(self) -> float:
        """Fraction of trainable dense parameters removed by the low-rank update; can be negative for excessive rank.

        Equation: 1 - self.update_params / self.full_params
        """
        return 1 - self.update_params / self.full_params

    # Optional (added 2026-10-20): the stored width of each value, so the same
    # rank-r product prices a factorized layer (two thin factors replacing the
    # dense matrix, r (d_in + d_out) values) as well as a LoRA update.
    bytes_per_param: Optional[Quantity] = None

    @property
    def compression(self) -> float:
        """Dense-to-factor parameter-count ratio; applies to replacing the matrix, not to storing LoRA alongside its backbone.

        Equation: self.full_params / self.update_params
        """
        return self.full_params / self.update_params

    @property
    def full_bytes(self) -> Quantity:
        """Dense matrix storage in MB at the optional declared bytes-per-parameter width.

        Equation: (self.full_params * self.bytes_per_param).to('MB')
        """
        return (self.full_params * self.bytes_per_param).to("MB")

    @property
    def update_bytes(self) -> Quantity:
        """Two-factor storage in MB at that same width; excludes the frozen backbone and optimizer state.

        Equation: (self.update_params * self.bytes_per_param).to('MB')
        """
        return (self.update_params * self.bytes_per_param).to("MB")


class ResidualAdapterScenario(BaseScenario):
    """Residual bottleneck adapters of ``width`` to ``bottleneck``, one at each of ``sites``.

    Notes
    -----
    Equation: Adapter parameters follow calc_residual_adapter_params; total scales by insertion-site count.

    Assumptions: Identical adapter shape at every site and uniform parameter storage width.

    Excludes: Base-model weights, optimizer state, and adapter quality.

    References
    ----------
    Implementation references: ``mlsysim.physics.calc_residual_adapter_params``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CompactEncoderResidualAdapters
    >>> scenario.params_per_adapter
    25360
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    width: int
    bottleneck: int
    sites: int
    bytes_per_param: Quantity

    @property
    def params_per_adapter(self) -> int:
        """One bottleneck adapter's parameter count from ``calc_residual_adapter_params``, including its defined bias terms.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics import calc_residual_adapter_params

        return calc_residual_adapter_params(self.width, self.bottleneck)

    @property
    def params_total(self) -> int:
        """Adapter parameter count across identical insertion sites.

        Equation: self.params_per_adapter * self.sites
        """
        return self.params_per_adapter * self.sites

    @property
    def adapter_bytes(self) -> Quantity:
        """Storage of one adapter in decimal KB at the supplied parameter width.

        Equation: (self.params_per_adapter * self.bytes_per_param).to('KB')
        """
        return (self.params_per_adapter * self.bytes_per_param).to("KB")

    @property
    def total_bytes(self) -> Quantity:
        """Storage across all adapter sites in decimal MB; excludes the base network and training state.

        Equation: (self.params_total * self.bytes_per_param).to('MB')
        """
        return (self.params_total * self.bytes_per_param).to("MB")


class PersonalizationStorageScenario(BaseScenario):
    """``contexts`` personalized models stored as full copies or as one backbone plus adapters.

    Notes
    -----
    Equation: Independent copies=``contexts*backbone_bytes``; shared storage=``backbone_bytes+contexts*adapter_bytes``.

    Assumptions: One backbone can be reused across all contexts and parameters share the declared storage width.

    Excludes: Activation caches, optimizer state, and context-switching overhead.

    References
    ----------
    Implementation references: ``mlsysim.physics.memory_from_params``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CameraModelTenContextPersonalization
    >>> round(scenario.total_savings, 6)
    9.52381
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model_params: Quantity  # in param
    adapter_params: Quantity  # in param, per context
    bytes_per_param: Quantity  # bytes per param
    contexts: int

    @property
    def model_size(self) -> Quantity:
        """Backbone weight storage via ``memory_from_params``; bytes follow the supplied parameter-width quantity.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics import memory_from_params

        return memory_from_params(self.model_params, self.bytes_per_param)

    @property
    def adapter_size(self) -> Quantity:
        """One context's adapter storage via the same parameter-to-memory conversion.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics import memory_from_params

        return memory_from_params(self.adapter_params, self.bytes_per_param)

    @property
    def total_copies(self) -> Quantity:
        """Storage for one complete backbone copy per context.

        Equation: self.contexts * self.model_size
        """
        return self.contexts * self.model_size

    @property
    def total_shared(self) -> Quantity:
        """Storage for one shared backbone plus one adapter per context.

        Equation: self.model_size + self.contexts * self.adapter_size
        """
        return self.model_size + self.contexts * self.adapter_size

    @property
    def total_savings(self) -> float:
        """Dimensionless ratio of independent copies to shared-backbone storage; a multiplicative saving, not a fractional reduction.

        Equation: (self.total_copies / self.total_shared).to_base_units().magnitude
        """
        return (self.total_copies / self.total_shared).to_base_units().magnitude

    @property
    def per_context_savings(self) -> float:
        """Backbone-to-adapter size ratio for one context; excludes the amortized shared backbone.

        Equation: (self.model_size / self.adapter_size).to_base_units().magnitude
        """
        return (self.model_size / self.adapter_size).to_base_units().magnitude


class FederatedUplinkScenario(BaseScenario):
    """A week of raw captures against compressed model updates sent at named round cadences.

    Notes
    -----
    Equation: Update bytes=``update_params*update_bits_per_param/bits_per_byte``; weekly traffic=``rounds*update_bytes``.

    Assumptions: Each round sends a full quantized update of the same size.

    Excludes: Protocol overhead, retries, secure aggregation, and convergence effects of cadence.

    References
    ----------
    Derivation: Raw weekly volume counts items times item bytes. An update counts parameters times bits per parameter, converted to bytes. Multiplying update size by cadence gives weekly traffic; equality of the two volumes gives the break-even round count.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PhoneImagesVersusQuantizedUpdates
    >>> round(scenario.break_even_rounds, 6)
    80.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    items_per_week: int
    item_size: Quantity
    update_params: float
    update_bits_per_param: int
    rounds_per_week: Dict[str, int]  # cadence name -> rounds a week

    @property
    def raw_volume(self) -> Quantity:
        """Weekly raw capture bytes: item count times bytes per item.

        Equation: self.items_per_week * self.item_size
        """
        return self.items_per_week * self.item_size

    @property
    def update_size(self) -> Quantity:
        """One quantized model update in decimal MB: parameter count times bits per parameter converted to bytes; excludes protocol overhead.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...core.units import BITS_PER_BYTE

        return (self.update_params * self.update_bits_per_param / BITS_PER_BYTE * Q_(1, "byte")).to("MB")

    @property
    def reduction(self) -> float:
        """Raw weekly bytes divided by bytes in one update; also the unrounded break-even number of updates.

        Equation: (self.raw_volume / self.update_size).to_base_units().magnitude
        """
        return (self.raw_volume / self.update_size).to_base_units().magnitude

    @property
    def break_even_rounds(self) -> float:
        """Unrounded updates per week at which update traffic equals raw capture traffic; alias of ``reduction``.

        Equation: self.reduction
        """
        return self.reduction

    def weekly_upload(self, cadence: str) -> Quantity:
        """Weekly MB at the named round cadence: rounds per week times update size.

        Equation: (self.rounds_per_week[cadence] * self.update_size).to('MB')
        """
        return (self.rounds_per_week[cadence] * self.update_size).to("MB")


class DenseLayerScenario(BaseScenario):
    """One dense layer: ``in_features`` inputs to ``out_features`` outputs, ``bytes_per_weight`` each.

    Notes
    -----
    Equation: A matrix with n inputs and m outputs has ``n*m`` weights; a batch costs ``2*B*n*m`` FLOPs.

    Assumptions: Dense multiplication, one weight read shared by the batch, and one read/write of activations in the traffic variant.

    Excludes: Cache hierarchy, bias/activation FLOPs, and kernel scheduling; target-intensity inversions require a positive denominator.

    References
    ----------
    Derivation: Every output dot product reads n weights, so m outputs require ``n*m``
    weights and MACs per example. Counting one multiply and one add per MAC gives
    ``2*B*n*m`` FLOPs. The traffic variant separately counts input reads, shared weight
    reads, and output writes.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.MnistMlpFirstLayer
    >>> s.variant(batch_size=2*s.batch_size).flops == 2*s.flops
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    in_features: int
    out_features: int
    bytes_per_weight: Quantity
    batch_size: int = 1

    @property
    def weights(self) -> int:
        """Matrix weight count: one connection for every input-output feature pair.

        Equation: ``in_features * out_features``."""
        return self.in_features * self.out_features

    @property
    def biases(self) -> int:
        """Bias count: one scalar for each output feature.

        Equation: ``out_features``."""
        return self.out_features

    @property
    def parameters(self) -> int:
        """Weights plus biases (what a framework's parameter count reports).

        Equation: ``weights + biases``."""
        return self.weights + self.biases

    @property
    def macs_per_example(self) -> int:
        """Multiply-accumulate count for one input vector, excluding bias and activation work.

        Equation: ``weights``."""
        return self.weights

    @property
    def weight_bytes(self) -> Quantity:
        """Stored weight-matrix footprint in bytes; biases are excluded.

        Equation: ``(weights * bytes_per_weight).to('byte')``."""
        return (self.weights * self.bytes_per_weight).to("byte")

    @property
    def intensity(self) -> float:
        """FLOPs per weight byte at ``batch_size``: each weight read serves the whole batch.

        Equation: ``flops = 2 * batch_size * weights; flops /
        weight_bytes.to('byte').magnitude``."""
        flops = 2 * self.batch_size * self.weights
        return flops / self.weight_bytes.to("byte").magnitude

    def batch_for_intensity(self, target: float) -> float:
        """Batch size at which the weight-only intensity reaches ``target`` FLOP/byte.

        Equation: ``target / variant(batch_size=1).intensity``."""
        return target / self.variant(batch_size=1).intensity

    # One-pass traffic model (added 2026-10-20): the layer reads its input and
    # weights once and writes its output once. ``activation_bytes_per_value``
    # sizes inputs and outputs; None uses ``bytes_per_weight``.
    activation_bytes_per_value: Optional[Quantity] = None

    @property
    def _act_bytes(self) -> float:
        b = self.activation_bytes_per_value or self.bytes_per_weight
        return b.to("byte").magnitude

    @property
    def flops(self) -> Quantity:
        """2 x batch x in x out: one multiply and one add per weight per example.

        Equation: ``Q_(2 * batch_size * weights, 'flop')``."""
        return Q_(2 * self.batch_size * self.weights, "flop")

    @property
    def input_bytes(self) -> Quantity:
        """Batch input traffic in bytes under one read per activation value.

        Equation: ``Q_(batch_size * in_features * _act_bytes, 'byte')``."""
        return Q_(self.batch_size * self.in_features * self._act_bytes, "byte")

    @property
    def output_bytes(self) -> Quantity:
        """Batch output traffic in bytes under one write per activation value.

        Equation: ``Q_(batch_size * out_features * _act_bytes, 'byte')``."""
        return Q_(self.batch_size * self.out_features * self._act_bytes, "byte")

    @property
    def traffic_bytes(self) -> Quantity:
        """Input read, weights read, and output written once each.

        Equation: ``input_bytes + weight_bytes + output_bytes``."""
        return self.input_bytes + self.weight_bytes + self.output_bytes

    @property
    def traffic_intensity(self) -> float:
        """FLOP per byte of the one-pass traffic.

        Equation: ``(flops / traffic_bytes).to('flop/byte').magnitude``."""
        return (self.flops / self.traffic_bytes).to("flop/byte").magnitude

    def batch_for_traffic_intensity(self, target: float) -> float:
        """Batch at which ``traffic_intensity`` reaches ``target`` FLOP/byte.

        Solves 2 B n m = target (B n a + n m w + B m a) for B, with n, m the
        widths and a, w the activation and weight bytes per value.

        Equation: ``n, m = (in_features, out_features); a, w = (_act_bytes,
        bytes_per_weight.to('byte').magnitude); target * n * m * w / (2 * n * m - target * a
        * (n + m))``."""
        n, m = self.in_features, self.out_features
        a, w = self._act_bytes, self.bytes_per_weight.to("byte").magnitude
        return target * n * m * w / (2 * n * m - target * a * (n + m))


class ConvLayerScenario(BaseScenario):
    """One ``kernel``-by-``kernel`` convolution, stride one, over a ``height``-by-``width``-by-``in_channels``
    input to ``out_channels`` feature maps; values of ``bytes_per_value`` each.

    ``same_padding`` keeps the output the size of the input; without it the
    window stays inside the input ("valid"), as an unpadded unfold does.

    Notes
    -----
    Equation: Each output position reuses ``k*k*Cin*Cout`` weights; batch FLOPs are twice ``positions*weights*batch``.

    Assumptions: Stride one, same or valid padding, and one-pass traffic at the declared precisions.

    Excludes: Dilation, groups other than the separate depthwise illustration, workspace, and measured kernel traffic.

    References
    ----------
    Derivation: A convolution filter contains ``k*k*Cin`` weights for each output channel.
    Applying that filter at each output location repeats its MACs without adding new
    parameters. The separate depthwise-plus-pointwise construction uses one spatial
    filter per input channel followed by channel mixing.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.MnistFirstConvLayer
    >>> s.variant(batch_size=2*s.batch_size).flops == 2*s.flops
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    height: int
    width: int
    kernel: int
    in_channels: int
    out_channels: int
    bytes_per_value: Quantity
    same_padding: bool = True

    @property
    def weights(self) -> int:
        """Filter weight count across spatial kernel, input channels, and output channels.

        Equation: ``kernel * kernel * in_channels * out_channels``."""
        return self.kernel * self.kernel * self.in_channels * self.out_channels

    @property
    def output_height(self) -> int:
        """Output rows for stride-one same or valid padding.

        Equation: ``height if same_padding else height - kernel + 1``."""
        return self.height if self.same_padding else self.height - self.kernel + 1

    @property
    def output_width(self) -> int:
        """Output columns for stride-one same or valid padding.

        Equation: ``width if same_padding else width - kernel + 1``."""
        return self.width if self.same_padding else self.width - self.kernel + 1

    @property
    def positions(self) -> int:
        """Output positions per feature map, each a reuse of every filter weight.

        Equation: ``output_height * output_width``."""
        return self.output_height * self.output_width

    @property
    def macs(self) -> int:
        """Multiply-accumulate count for one example, reusing each filter at every output position.

        Equation: ``positions * weights``."""
        return self.positions * self.weights

    @property
    def outputs(self) -> int:
        """Output activation values for one example.

        Equation: ``positions * out_channels``."""
        return self.positions * self.out_channels

    # Batch and one-pass traffic model (added 2026-10-20): ``batch_size``
    # examples share one read of the weights; inputs are read and outputs
    # written once. ``weight_bytes_per_value`` sizes the weights when they use a
    # different format from the activations (INT8 weights, FP16 activations).
    batch_size: int = 1
    weight_bytes_per_value: Optional[Quantity] = None

    @property
    def weight_bytes(self) -> Quantity:
        """Filter storage in bytes, using a separate weight width when supplied.

        Equation: ``(weights * (weight_bytes_per_value or bytes_per_value)).to('byte')``."""
        return (self.weights * (self.weight_bytes_per_value or self.bytes_per_value)).to("byte")

    @property
    def activation_bytes(self) -> Quantity:
        """Output activation footprint in bytes for one example.

        Equation: ``(outputs * bytes_per_value).to('byte')``."""
        return (self.outputs * self.bytes_per_value).to("byte")

    @property
    def flops(self) -> Quantity:
        """2 x batch x MACs: one multiply and one add per weight per position.

        Equation: ``Q_(2 * batch_size * macs, 'flop')``."""
        return Q_(2 * self.batch_size * self.macs, "flop")

    @property
    def batch_input_bytes(self) -> Quantity:
        """Input activation bytes read once for the whole batch.

        Equation: ``(batch_size * input_values * bytes_per_value).to('byte')``."""
        return (self.batch_size * self.input_values * self.bytes_per_value).to("byte")

    @property
    def batch_output_bytes(self) -> Quantity:
        """Output activation bytes written once for the whole batch.

        Equation: ``(batch_size * outputs * bytes_per_value).to('byte')``."""
        return (self.batch_size * self.outputs * self.bytes_per_value).to("byte")

    @property
    def traffic_bytes(self) -> Quantity:
        """Batch input read, weights read, and batch output written once each.

        Equation: ``batch_input_bytes + weight_bytes + batch_output_bytes``."""
        return self.batch_input_bytes + self.weight_bytes + self.batch_output_bytes

    @property
    def traffic_intensity(self) -> float:
        """FLOP per byte of the one-pass traffic.

        Equation: ``(flops / traffic_bytes).to('flop/byte').magnitude``."""
        return (self.flops / self.traffic_bytes).to("flop/byte").magnitude

    @property
    def activation_to_weight_ratio(self) -> float:
        """Dimensionless one-example output footprint divided by filter storage.

        Equation: ``(activation_bytes / weight_bytes).to('').magnitude``."""
        return (self.activation_bytes / self.weight_bytes).to("").magnitude

    @property
    def input_values(self) -> int:
        """Input activation count across spatial positions and channels.

        Equation: ``height * width * in_channels``."""
        return self.height * self.width * self.in_channels

    @property
    def dense_weights(self) -> int:
        """Weights of a dense layer from the whole input to ``out_channels`` outputs.

        Equation: ``input_values * out_channels``."""
        return self.input_values * self.out_channels

    @property
    def sharing_ratio(self) -> float:
        """Dense weights over convolution weights for the same input and feature count.

        Equation: ``dense_weights / weights``."""
        return self.dense_weights / self.weights

    def dense_weight_growth(self, height: int, width: int) -> float:
        """Factor by which the dense layer's weights grow when the input becomes ``height`` by ``width``.

        Equation: ``height * width / (height * width)``."""
        return (height * width) / (self.height * self.width)

    @property
    def unfolded_entries(self) -> int:
        """Entries of the im2col patch matrix: one row of ``kernel**2 * in_channels`` per position.

        Equation: ``positions * kernel * kernel * in_channels``."""
        return self.positions * self.kernel * self.kernel * self.in_channels

    @property
    def max_copies(self) -> int:
        """Most rows of the patch matrix one input value can appear in.

        Equation: ``kernel * kernel``."""
        return self.kernel * self.kernel

    @property
    def depthwise_weights(self) -> int:
        """One ``kernel``-by-``kernel`` filter per input channel (multiply-adds per position).

        Equation: ``kernel * kernel * in_channels``."""
        return self.kernel * self.kernel * self.in_channels

    @property
    def pointwise_weights(self) -> int:
        """A 1x1 convolution mixing ``in_channels`` into ``out_channels`` (multiply-adds per position).

        Equation: ``in_channels * out_channels``."""
        return self.in_channels * self.out_channels

    @property
    def separable_weights(self) -> int:
        """Combined depthwise and pointwise weight count.

        Equation: ``depthwise_weights + pointwise_weights``."""
        return self.depthwise_weights + self.pointwise_weights

    @property
    def separable_ratio(self) -> float:
        """Standard over depthwise separable weights (and multiply-adds per position).

        Equation: ``weights / separable_weights``."""
        return self.weights / self.separable_weights

    @property
    def separable_fraction(self) -> float:
        """Depthwise-separable parameter fraction relative to a standard convolution.

        Equation: ``_separable_fraction(kernel, out_channels)``."""
        return _separable_fraction(self.kernel, self.out_channels)

    @property
    def separable_limit(self) -> int:
        """The ratio's limit with many output channels: the filter area.

        Equation: ``kernel * kernel``."""
        return self.kernel * self.kernel


class LayerStackScenario(BaseScenario):
    """Weight-only count of a network built from convolution and dense layers, input first (biases omitted).

    A multilayer perceptron has no convolution layers; a small CNN ends in a
    dense classifier head. (The full forward and training costs of a dense
    network belong to the neural-computation solvers; this counts weights.)

    Notes
    -----
    Equation: Total weights are the sum of convolution and dense matrix counts; the dense share divides its subtotal by the total.

    Assumptions: A nonempty dense-layer sequence with compatible widths and positive total weight count.

    Excludes: Biases, activation state, and training work; use neural solvers for those costs.

    References
    ----------
    Derivation: Convolution filters and dense matrices are disjoint parameter sets, so
    their counts add. Dividing the dense subtotal by the combined count expresses its
    share. Biases are omitted consistently from both numerator and denominator.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.MnistMlpLayerStack
    >>> s.weights == s.conv_weights + s.dense_weights
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    conv_layers: List[ConvLayerScenario] = []
    dense_layers: List[DenseLayerScenario]

    @property
    def widths(self) -> List[int]:
        """Dense widths, input first.

        Equation: ``[dense_layers[0].in_features] + [d.out_features for d in
        dense_layers]``."""
        return [self.dense_layers[0].in_features] + [d.out_features for d in self.dense_layers]

    @property
    def conv_weights(self) -> int:
        """Sum of convolution filter counts; dense weights and biases are excluded.

        Equation: ``sum((c.weights for c in conv_layers))``."""
        return sum(c.weights for c in self.conv_layers)

    @property
    def dense_weights(self) -> int:
        """Sum of dense matrix counts; convolution filters and biases are excluded.

        Equation: ``sum((d.weights for d in dense_layers))``."""
        return sum(d.weights for d in self.dense_layers)

    @property
    def weights(self) -> int:
        """Total convolution and dense weight count, with biases omitted.

        Equation: ``conv_weights + dense_weights``."""
        return self.conv_weights + self.dense_weights

    @property
    def macs_per_example_dense(self) -> int:
        """Multiply-adds of the dense layers per example (one per weight).

        Equation: ``dense_weights``."""
        return self.dense_weights

    @property
    def dense_share(self) -> float:
        """Share of the weights in the dense layers (a CNN's classifier head).

        Equation: ``dense_weights / weights``."""
        return self.dense_weights / self.weights


class ElementwiseIntensityScenario(BaseScenario):
    """An unfused elementwise operation: ``flops_per_element`` FLOPs while ``operands_moved`` values
    of ``bytes_per_value`` each are read or written.

    Notes
    -----
    Equation: Traffic per element is ``moved_values*bytes_per_value``; intensity divides per-element FLOPs by that traffic.

    Assumptions: Every declared operand is transferred once and transfer width is positive.

    Excludes: Fusion, caching, and extra address or indexing traffic.

    References
    ----------
    Derivation: One element moves operands_moved values of the declared byte width.
    Arithmetic intensity is defined as its useful FLOP count divided by those
    transferred bytes; no hardware efficiency estimate enters this ratio.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ResidualAddFp32
    >>> s.bytes_per_element.to("byte").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    flops_per_element: int
    operands_moved: int
    bytes_per_value: Quantity

    @property
    def bytes_per_element(self) -> Quantity:
        """Traffic in bytes per output element, counting each declared read or write.

        Equation: ``(operands_moved * bytes_per_value).to('byte')``."""
        return (self.operands_moved * self.bytes_per_value).to("byte")

    @property
    def intensity(self) -> float:
        """FLOPs per byte moved.

        Equation: ``flops_per_element / bytes_per_element.to('byte').magnitude``."""
        return self.flops_per_element / self.bytes_per_element.to("byte").magnitude


class FrameRateHeadroomScenario(BaseScenario):
    """``workload`` at ``frame_rate`` on an accelerator of ``peak_rate`` that sustains a
    ``utilization_low`` to ``utilization_high`` share of peak; ``alternative_flops`` is a heavier model per frame.

    Notes
    -----
    Equation: Stream demand is ``frame_rate*work_per_frame``; sustained supply is ``peak_rate*utilization``.

    Assumptions: Constant frame complexity and declared utilization, with no frame dropping.

    Excludes: Memory limits, input preprocessing, deadline jitter, and measured sustained efficiency.

    References
    ----------
    Derivation: A stream of equally costly frames requires frames/second times
    FLOPs/frame. Multiplying peak FLOP/s by a declared utilization gives the assumed
    sustained supply; dividing supply by demand gives dimensionless headroom.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ResNet50ThirtyFpsStream
    >>> s.required_rate.to("GFLOP/s").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    frame_rate: Quantity
    peak_rate: Quantity
    utilization_low: float
    utilization_high: float
    alternative_flops: Quantity

    @property
    def flops_per_frame(self) -> Quantity:
        """Registered model inference work per frame, expressed in GFLOP.

        Equation: ``workload.inference_flops.to('GFLOP')``."""
        return self.workload.inference_flops.to("GFLOP")

    @property
    def required_rate(self) -> Quantity:
        """Compute demand in GFLOP/s at the configured frame frequency.

        Equation: ``(frame_rate * flops_per_frame).to('GFLOP/s')``."""
        return (self.frame_rate * self.flops_per_frame).to("GFLOP/s")

    @property
    def sustained_low(self) -> Quantity:
        """Available TFLOP/s at the lower assumed utilization.

        Equation: ``(peak_rate * utilization_low).to('TFLOP/s')``."""
        return (self.peak_rate * self.utilization_low).to("TFLOP/s")

    @property
    def sustained_high(self) -> Quantity:
        """Available TFLOP/s at the upper assumed utilization.

        Equation: ``(peak_rate * utilization_high).to('TFLOP/s')``."""
        return (self.peak_rate * self.utilization_high).to("TFLOP/s")

    @property
    def headroom(self) -> float:
        """Sustained rate at the low utilization over the rate the stream needs.

        Equation: ``(sustained_low / required_rate).to('').magnitude``."""
        return (self.sustained_low / self.required_rate).to("").magnitude

    @property
    def alternative_required_rate(self) -> Quantity:
        """Compute demand in TFLOP/s for the alternative model at the same frame rate.

        Equation: ``(frame_rate * alternative_flops).to('TFLOP/s')``."""
        return (self.frame_rate * self.alternative_flops).to("TFLOP/s")

    @property
    def alternative_headroom(self) -> float:
        """Dimensionless lower sustained rate divided by alternative-model demand.

        Equation: ``(sustained_low / alternative_required_rate).to('').magnitude``."""
        return (self.sustained_low / self.alternative_required_rate).to("").magnitude

    @property
    def peak_frame_rate(self) -> Quantity:
        """Frames a second if the accelerator ran at peak (the planning mistake).

        Equation: ``(peak_rate / flops_per_frame).to('1/s')``."""
        return (self.peak_rate / self.flops_per_frame).to("1/s")


class RecurrentStepScenario(BaseScenario):
    """One step of a recurrent layer: ``input_dim`` inputs, a ``hidden_dim`` state, and ``gates`` affine
    maps per step (one for a plain RNN, four for an LSTM); ``steps`` steps in a sequence.

    Notes
    -----
    Equation: Each gate uses ``H*I`` input MACs and ``H*H`` recurrent MACs; all gates and steps share their matrices.

    Assumptions: Dense gates and a retained hidden vector of H values at the declared width.

    Excludes: Biases, nonlinear gate work, LSTM cell-state extras, and recomputation policies.

    References
    ----------
    Derivation: Each recurrent gate applies a matrix to the input and another to the
    preceding hidden state, giving ``H*I`` + ``H*H`` MACs. Gates multiply that work; time steps
    reuse the same weights but retain separate states when full history is stored.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ElmanRnnStep
    >>> s.macs_per_step == s.input_macs + s.recurrent_macs
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    input_dim: int
    hidden_dim: int
    bytes_per_value: Quantity
    gates: int = 1
    steps: int = 1

    @property
    def gate_rows(self) -> int:
        """Rows of the fused input and recurrent weight matrices.

        Equation: ``gates * hidden_dim``."""
        return self.gates * self.hidden_dim

    @property
    def recurrent_macs(self) -> int:
        """Recurrent-state multiply-accumulate count across all gates for one step.

        Equation: ``gate_rows * hidden_dim``."""
        return self.gate_rows * self.hidden_dim

    @property
    def input_macs(self) -> int:
        """Input-projection multiply-accumulate count across all gates for one step.

        Equation: ``gate_rows * input_dim``."""
        return self.gate_rows * self.input_dim

    @property
    def macs_per_step(self) -> int:
        """Combined input and recurrent multiply-accumulate count for one time step.

        Equation: ``recurrent_macs + input_macs``."""
        return self.recurrent_macs + self.input_macs

    @property
    def weights(self) -> int:
        """Input and recurrent weights (biases omitted); one multiply-add each per step.

        Equation: ``macs_per_step``."""
        return self.macs_per_step

    @property
    def state_bytes(self) -> Quantity:
        """One state vector: what inference keeps at any sequence length.

        Equation: ``(hidden_dim * bytes_per_value).to('byte')``."""
        return (self.hidden_dim * self.bytes_per_value).to("byte")

    @property
    def training_states(self) -> int:
        """States backpropagation through time keeps: one per step.

        Equation: ``steps``."""
        return self.steps


class KernelShareScenario(BaseScenario):
    """One kernel class's multiply-adds against a whole model's, per example.

    Notes
    -----
    Equation: Kernel share is kernel_macs/total_macs for the same example.

    Assumptions: Both counts use the same MAC convention and the total is nonzero.

    Excludes: Latency attribution and memory traffic; an operation share is not a runtime share.

    References
    ----------
    Derivation: The kernel MAC count is a subset of total model MACs for the same
    example. Their ratio is an accounting fraction; converting that fraction into
    elapsed-time attribution would additionally require execution and memory
    information.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ResNet50ConvKernelShare
    >>> 0 <= s.share <= 1
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    kernel_macs: Quantity  # multiply-adds of the kernel class, per example
    total_macs: Quantity  # multiply-adds of the whole model, per example

    @property
    def share(self) -> float:
        """Dimensionless fraction of model multiply-accumulates attributed to this kernel class.

        Equation: ``(kernel_macs / total_macs).to('').magnitude``."""
        return (self.kernel_macs / self.total_macs).to("").magnitude


class DenseNetworkScenario(BaseScenario):
    """A dense network of ``layer_widths`` at ``batch_size``: parameters, FLOPs, traffic, memory.

    ``bytes_per_value`` is the storage width of every weight, gradient, and
    activation; ``optimizer_bytes_per_parameter`` the optimizer's state per
    parameter (two FP32 values for Adam). With a ``dataset``, the result also
    gives the steps in one epoch over its training split.

    Notes
    -----
    Equation: DenseNetworkModel combines adjacent-layer matrix work and training/activation storage.

    Assumptions: The layer-width sequence, batch, and precision apply to all dense layers; optional epoch size comes from dataset.training_examples.

    Excludes: Attention, convolution, and architecture-specific kernel execution.

    References
    ----------
    Derivation: Delegates to mlsysim.engine.solvers.neural.DenseNetworkModel.solve. That
    contract composes adjacent-layer matrix counts with shared activation and
    training-state functions in mlsysim.physics.neural. This wrapper supplies widths,
    storage choices, batch size, and the optional dataset epoch count.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.MnistDenseNetwork
    >>> s.result.macs_per_example > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    layer_widths: List[int]
    batch_size: int = 1
    bytes_per_value: Quantity
    optimizer_bytes_per_parameter: Quantity
    dataset: Optional[DatasetProfile] = None

    @property
    def result(self):
        """Evaluate DenseNetworkModel with these widths, storage choices, batch, and optional training-example count.

        Equation: Delegates to mlsysim.engine.solvers.neural.DenseNetworkModel.solve, using
        the dataset training-example count when available."""
        from ..solvers import DenseNetworkModel

        examples = None
        if self.dataset is not None and self.dataset.training_examples is not None:
            examples = int(self.dataset.training_examples.to("count").magnitude)
        return DenseNetworkModel().solve(
            self.layer_widths, self.batch_size, self.bytes_per_value, self.optimizer_bytes_per_parameter, examples
        )

    @property
    def architecture(self) -> str:
        """The widths joined by arrows, ``784 → 128 → 64 → 10``.

        Equation: ``' → '.join((str(w) for w in layer_widths))``."""
        return " → ".join(str(w) for w in self.layer_widths)


class DenseScalingSweepScenario(BaseScenario):
    """Dense networks between a fixed input and output, swept in hidden width and in depth.

    Width sweep: ``hidden_layers`` hidden layers, each of every width in
    ``widths``. Depth sweep: 1 to ``max_depth`` hidden layers of
    ``fixed_width``. Each doubling factor compares the last setting with the
    one half its size.

    Notes
    -----
    Equation: Width and depth sweeps call calc_dense_macs; saved state calls calc_dense_saved_activations.

    Assumptions: The width list contains its final width halved; max_depth is at least two for doubling comparisons.

    Excludes: Runtime, hardware saturation, and accuracy changes with model size.

    References
    ----------
    Derivation: Calls mlsysim.physics.neural.calc_dense_macs for each constructed width
    sequence and calc_dense_saved_activations for the corresponding training-state
    count. Ratios compare already evaluated settings; they do not assume exact quadratic
    or linear scaling for the fixed input/output layers.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.DenseWidthDepthSweep
    >>> len(s.width_sweep_macs) == len(s.widths)
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    input_dim: int
    output_dim: int
    hidden_layers: int
    widths: List[int]
    fixed_width: int
    max_depth: int

    def net(self, hidden_layers: int, width: int) -> List[int]:
        """Build input-hidden-output layer widths for the requested hidden depth and width.

        Equation: ``[input_dim] + [width] * hidden_layers + [output_dim]``."""
        return [self.input_dim] + [width] * hidden_layers + [self.output_dim]

    @property
    def depths(self) -> List[int]:
        """Enumerate hidden-layer counts from one through the configured maximum.

        Equation: ``list(range(1, max_depth + 1))``."""
        return list(range(1, self.max_depth + 1))

    @property
    def width_sweep_macs(self) -> List[int]:
        """Per-example multiply-accumulate counts for each hidden width at fixed depth.

        Equation: ``[calc_dense_macs(net(hidden_layers, w)) for w in widths]``."""
        from ...physics.neural import calc_dense_macs

        return [calc_dense_macs(self.net(self.hidden_layers, w)) for w in self.widths]

    @property
    def depth_sweep_macs(self) -> List[int]:
        """Per-example multiply-accumulate counts for each hidden depth at fixed width.

        Equation: ``[calc_dense_macs(net(d, fixed_width)) for d in depths]``."""
        from ...physics.neural import calc_dense_macs

        return [calc_dense_macs(self.net(d, self.fixed_width)) for d in self.depths]

    @property
    def depth_sweep_saved_activations(self) -> List[int]:
        """Values saved per example: the input and every layer output.

        Equation: ``[calc_dense_saved_activations(net(d, fixed_width)) for d in depths]``."""
        from ...physics.neural import calc_dense_saved_activations

        return [calc_dense_saved_activations(self.net(d, self.fixed_width)) for d in self.depths]

    @property
    def width_doubled_from(self) -> int:
        """Hidden width used as the denominator of the final width-doubling comparison.

        Equation: ``widths[-1] // 2``."""
        return self.widths[-1] // 2

    @property
    def depth_doubled_from(self) -> int:
        """Hidden depth used as the denominator of the final depth-doubling comparison.

        Equation: ``max_depth // 2``."""
        return self.max_depth // 2

    @property
    def width_double_factor(self) -> float:
        """Dimensionless MAC increase between the final width and its half-width entry.

        Equation: ``m = width_sweep_macs; m[-1] / m[widths.index(width_doubled_from)]``."""
        m = self.width_sweep_macs
        return m[-1] / m[self.widths.index(self.width_doubled_from)]

    @property
    def depth_double_factor(self) -> float:
        """Dimensionless MAC increase between the final depth and its half-depth entry.

        Equation: ``m = depth_sweep_macs; m[-1] / m[depth_doubled_from - 1]``."""
        m = self.depth_sweep_macs
        return m[-1] / m[self.depth_doubled_from - 1]

    @property
    def saved_activations_double_factor(self) -> float:
        """Dimensionless saved-activation increase between final and half-depth entries.

        Equation: ``a = depth_sweep_saved_activations; a[-1] / a[depth_doubled_from - 1]``."""
        a = self.depth_sweep_saved_activations
        return a[-1] / a[self.depth_doubled_from - 1]


class EpochOverheadScenario(BaseScenario):
    """One training epoch of ``network`` on a host CPU and an accelerator: work against fixed overhead.

    Every launched operation pays ``dispatch`` (the framework's Python-side
    cost); on the accelerator it also pays ``launch``. ``flop_energy`` and
    ``dram_energy_per_byte`` price arithmetic against data movement.

    Notes
    -----
    Equation: EpochOverheadModel combines batched work, batch-one work, steps per epoch, and per-operation dispatch/launch costs.

    Assumptions: The dataset supplies an epoch size and per-operation overhead/energy inputs are held fixed.

    Excludes: Measured overlap, dynamic power, and training quality.

    References
    ----------
    Derivation: Delegates to mlsysim.engine.solvers.neural.EpochOverheadModel.solve,
    supplying both the configured-batch and batch-one DenseNetworkModel results. The two
    profiles let the solver distinguish repeated per-step overhead from work amortized
    by batching.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.MnistEpochCpuVersusH100
    >>> s.network.result.steps_per_epoch > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    network: DenseNetworkScenario
    host: HardwareNode
    accelerator: HardwareNode
    precision: str = "fp32"
    dispatch: Quantity
    launch: Quantity
    dram_energy_per_byte: Quantity
    flop_energy: Quantity

    @property
    def result(self):
        """Evaluate EpochOverheadModel using batched and single-example network costs and explicit host/launch overheads.

        Equation: Delegates to mlsysim.engine.solvers.neural.EpochOverheadModel.solve with
        batch and batch-one network profiles."""
        from ..solvers import EpochOverheadModel

        net = self.network.result
        return EpochOverheadModel().solve(
            net,
            self.network.variant(batch_size=1).result,
            self.host,
            self.accelerator,
            self.precision,
            self.dispatch,
            self.launch,
            net.steps_per_epoch,
            self.dram_energy_per_byte,
            self.flop_energy,
        )


class ParadigmCostScenario(BaseScenario):
    """One input classified three ways: hand-written rules, engineered features, and a learned network.

    Rules: ``rule_comparisons`` comparisons over the raw pixels. Features: a
    ``feature_grid`` by ``feature_grid`` grid of cells, ``feature_bins``
    orientation bins each, ``feature_operations`` operations in
    ``feature_memory`` of working state. Network: ``network``'s MACs.

    Notes
    -----
    Equation: Rule image storage is ``pixel_count*pixel_width``; a feature grid has ``grid**2*bins`` entries.

    Assumptions: The three supplied work budgets refer to the same input but may count different operation kinds.

    Excludes: Accuracy equivalence and conversion of work-count ratios into latency ratios.

    References
    ----------
    Derivation: The rule path stores one value per image pixel. The feature path places
    a fixed number of orientation bins in every grid cell. The neural path obtains MACs
    from DenseNetworkModel. Their count ratios compare declared work categories rather
    than pretending comparisons and MACs have identical costs.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.DigitRecognitionParadigms
    >>> s.feature_count == s.feature_grid**2 * s.feature_bins
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    image_pixels: int
    bytes_per_pixel: Quantity
    rule_comparisons: int
    feature_grid: int
    feature_bins: int
    feature_operations: int
    feature_memory: Quantity
    network: DenseNetworkScenario

    @property
    def rule_memory(self) -> Quantity:
        """Raw image storage in bytes assumed accessible to the handwritten-rule classifier.

        Equation: ``(image_pixels * bytes_per_pixel).to('byte')``."""
        return (self.image_pixels * self.bytes_per_pixel).to("byte")

    @property
    def feature_count(self) -> int:
        """Engineered feature count: orientation bins for each cell in the square grid.

        Equation: ``feature_grid ** 2 * feature_bins``."""
        return self.feature_grid**2 * self.feature_bins

    @property
    def feature_to_rule_ratio(self) -> float:
        """Dimensionless engineered-feature operations divided by rule comparisons; these are different operation categories.

        Equation: ``feature_operations / rule_comparisons``."""
        return self.feature_operations / self.rule_comparisons

    @property
    def network_to_rule_ratio(self) -> float:
        """Dimensionless network MACs divided by rule comparisons; this is a work-count contrast, not a latency ratio.

        Equation: ``network.result.macs_per_example / rule_comparisons``."""
        return self.network.result.macs_per_example / self.rule_comparisons


class ActivationLogicCostScenario(BaseScenario):
    """Logic cost of two activation datapaths, in transistor equivalents (an illustrative budget).

    Notes
    -----
    Equation: The illustrative implementation ratio divides costly_transistors by cheap_transistors.

    Assumptions: Both budgets count transistor equivalents at a comparable implementation boundary.

    Excludes: Process scaling, area routing overhead, power, and measured synthesized circuits.

    References
    ----------
    Derivation: Dividing two explicitly supplied transistor-equivalent budgets compares
    the assumed activation implementations. The ratio is defined by those budgets; it is
    not a universal sigmoid-to-ReLU cost law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ReluVersusSigmoidLogic
    >>> s.ratio > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    cheap_label: str
    cheap_transistors: int
    costly_label: str
    costly_transistors: int

    @property
    def ratio(self) -> float:
        """Dimensionless costly-to-cheap transistor budget ratio under the supplied illustrative implementations.

        Equation: ``costly_transistors / cheap_transistors``."""
        return self.costly_transistors / self.cheap_transistors


class GradientChainScenario(BaseScenario):
    """The activation-slope factor a gradient keeps after ``depth`` layers, at each slope's peak.

    Notes
    -----
    Equation: The activation-only gradient factor is ``max_sigmoid_slope**depth``.

    Assumptions: Every layer is evaluated at the sigmoid derivative maximum; weight Jacobians are treated separately.

    Excludes: Weight-matrix amplification, residual paths, and actual training-gradient distributions.

    References
    ----------
    Derivation: The chain rule multiplies activation derivatives across layers. The
    maximum sigmoid derivative is supplied by mlsysim.physics.neural.calc_sigmoid_slope
    at zero; raising it to depth isolates the activation-only factor without assigning
    values to the weight Jacobians.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.TwentySigmoidLayers
    >>> s.variant(depth=s.depth+1).surviving_fraction <= s.surviving_fraction
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    depth: int
    activation: str = "sigmoid"

    @property
    def peak_slope(self) -> float:
        """Maximum sigmoid derivative, evaluated at zero; other activation names are rejected.

        Equation: ``calc_sigmoid_slope(0.0)``."""
        from ...physics.neural import calc_sigmoid_slope

        if self.activation != "sigmoid":
            raise ValueError(f"unsupported activation {self.activation!r}")
        return calc_sigmoid_slope(0.0)

    @property
    def surviving_fraction(self) -> float:
        """Dimensionless product of the peak sigmoid slopes over the configured depth.

        Equation: ``peak_slope ** depth``."""
        return self.peak_slope**self.depth


class CrossEntropyScenario(BaseScenario):
    """Cross-entropy losses for named predicted probabilities of the correct class, among ``classes``.

    ``misread`` gives the probabilities of a confident mistake, keyed
    ``"true"`` (the correct class) and ``"wrong"`` (the class predicted);
    the gradient on each score is its probability minus its target.

    Notes
    -----
    Equation: Correct-class loss is -ln(p); softmax-logit gradients are predicted probability minus one-hot target.

    Assumptions: Valid nonzero correct-class probabilities and one categorical target per example.

    Excludes: Calibration quality, label smoothing, and class weighting.

    References
    ----------
    Derivation: Uses mlsysim.physics.neural.calc_cross_entropy for -ln(p) of the correct
    class. Differentiating softmax cross-entropy with respect to a logit gives p-y, so
    the true-class target is one and a wrong-class target is zero.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.DigitCrossEntropyExamples
    >>> s.uniform_loss > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    classes: int
    probabilities: Dict[str, float]
    misread: Dict[str, float]

    def loss(self, key: str) -> float:
        """Natural-log cross-entropy in nats for the named correct-class probability.

        Equation: ``calc_cross_entropy(probabilities[key])``."""
        from ...physics.neural import calc_cross_entropy

        return calc_cross_entropy(self.probabilities[key])

    @property
    def uniform_loss(self) -> float:
        """Natural-log loss in nats when probability is shared equally among classes.

        Equation: ``calc_cross_entropy(1 / classes)``."""
        from ...physics.neural import calc_cross_entropy

        return calc_cross_entropy(1 / self.classes)

    @property
    def misread_loss(self) -> float:
        """Natural-log loss in nats at the supplied confident-error probability for the true class.

        Equation: ``calc_cross_entropy(misread['true'])``."""
        from ...physics.neural import calc_cross_entropy

        return calc_cross_entropy(self.misread["true"])

    @property
    def misread_gradient_true(self) -> float:
        """Dimensionless true-class softmax-logit gradient, probability minus its one-hot target.

        Equation: ``misread['true'] - 1.0``."""
        return self.misread["true"] - 1.0

    @property
    def misread_gradient_wrong(self) -> float:
        """Dimensionless wrong-class softmax-logit gradient, probability minus zero target.

        Equation: ``misread['wrong'] - 0.0``."""
        return self.misread["wrong"] - 0.0


class TrainingStateFitScenario(BaseScenario):
    """Weights, gradients, and optimizer state of ``workload`` at one precision, against ``hardware``'s memory.

    ``bytes_per_value`` stores each weight and gradient; the optimizer keeps
    ``optimizer_bytes_per_parameter`` (two FP32 values for Adam, one for SGD
    with momentum). Activations are not included.

    Notes
    -----
    Equation: Parameter state sums weights, gradients, optimizer state, and optional master weights; adapter mode keeps base weights frozen.

    Assumptions: Bytes per value/state and the activation budget are supplied; activation storage scales linearly with batch.

    Excludes: Runtime workspaces, allocator reserve, optimizer-specific extra temporaries, and capacity safety margins.

    References
    ----------
    Derivation: Uses the neural training-state accounting for full-model weights,
    gradients, and optimizer state, then adds an explicitly configured master copy. In
    adapter mode the base weights remain resident while trained adapter parameters
    receive their own state. The activation budget is supplied separately, so
    parameter-state fit and total training fit are different questions.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Gpt2Fp32AdamOnH100
    >>> s.weights.to("byte").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: Optional[HardwareNode] = None
    bytes_per_value: Quantity
    optimizer_bytes_per_parameter: Quantity
    # Optional (added 2026-10-20): the activations training saves for the
    # backward pass at ``batch_size``, a stated estimate; saved activations
    # grow in proportion to the batch.
    batch_size: Optional[int] = None
    activations: Optional[Quantity] = None
    # Optional (added 2026-10-20): mixed precision keeps a higher-precision
    # master copy of every trained weight (4 B, FP32) beside the 16-bit
    # weights and gradients; None for a single-precision recipe.
    master_bytes_per_parameter: Optional[Quantity] = None
    # Optional (added 2026-10-20): trainable adapter parameters beside a
    # frozen base (low-rank adapters). The base keeps only its weights; each
    # adapter parameter carries weights, gradients, master copy, and
    # optimizer state.
    adapter_parameters: Optional[int] = None
    # Optional (added 2026-10-20): low-rank adapters stated by shape instead of
    # a count: rank r on ``adapted_matrices`` d_out-by-d_in matrices in each of
    # the workload's layers, r (d_in + d_out) parameters per matrix. Used when
    # ``adapter_parameters`` is None.
    adapter_rank: Optional[int] = None
    adapted_matrices: Optional[int] = None
    adapter_in_dim: Optional[int] = None
    adapter_out_dim: Optional[int] = None

    @property
    def adapter_count(self) -> Optional[int]:
        """Trainable adapter parameters, stated or from the adapter shape.

        Equation: Use adapter_parameters if supplied; otherwise layers * adapted_matrices *
        calc_low_rank_update_params(in_dim, out_dim, rank). No adapter specification returns
        None."""
        if self.adapter_parameters is not None:
            return self.adapter_parameters
        if self.adapter_rank is None:
            return None
        from ...physics.memory import calc_low_rank_update_params

        return (
            self.workload.layers
            * self.adapted_matrices
            * calc_low_rank_update_params(self.adapter_in_dim, self.adapter_out_dim, self.adapter_rank)
        )

    def activations_at(self, batch_size: int) -> Quantity:
        """Saved activations at another batch size, scaled linearly from ``activations``.

        Equation: ``(activations * batch_size / batch_size).to('byte')``."""
        return (self.activations * batch_size / self.batch_size).to("byte")

    @property
    def training_total(self) -> Quantity:
        """Weights, gradients, optimizer state, and saved activations.

        Equation: ``(total + activations).to('byte')``."""
        return (self.total + self.activations).to("byte")

    @property
    def training_copies(self) -> float:
        """Training total in units of the weights.

        Equation: ``(training_total / weights).to('').magnitude``."""
        return (self.training_total / self.weights).to("").magnitude

    def _state(self):
        from ...physics.neural import calc_training_state

        p = self.workload.parameters.to("count").magnitude
        return [q.to("byte") for q in calc_training_state(p, self.bytes_per_value, self.optimizer_bytes_per_parameter)]

    @property
    def weights(self) -> Quantity:
        """Base-model weight storage in bytes from the shared neural training-state accounting.

        Equation: ``_state()[0]``."""
        return self._state()[0]

    @property
    def gradients(self) -> Quantity:
        """Full-model gradient storage in bytes from the shared neural training-state accounting.

        Equation: ``_state()[1]``."""
        return self._state()[1]

    @property
    def optimizer_state(self) -> Quantity:
        """Full-model optimizer storage in bytes from the shared neural training-state accounting.

        Equation: ``_state()[2]``."""
        return self._state()[2]

    @property
    def optimizer_values_per_parameter(self) -> float:
        """Optimizer byte budget per parameter divided by the configured weight-value width.

        Equation: ``(optimizer_bytes_per_parameter / bytes_per_value).to('').magnitude``."""
        return (self.optimizer_bytes_per_parameter / self.bytes_per_value).to("").magnitude

    @property
    def master_weights(self) -> Quantity:
        """Optional full-model master-weight storage in bytes; zero when no master copy is configured.

        Equation: ``(workload.parameters.to('count').magnitude *
        master_bytes_per_parameter).to('byte')``."""
        if self.master_bytes_per_parameter is None:
            return Q_(0, "byte")
        return (self.workload.parameters.to("count").magnitude * self.master_bytes_per_parameter).to("byte")

    @property
    def bytes_per_trained_parameter(self) -> Quantity:
        """Weights, gradients, master copy, and optimizer state for one trained parameter.

        Equation: ``master = Q_(0, 'byte') if master_bytes_per_parameter is None else
        master_bytes_per_parameter; (2 * bytes_per_value + master +
        optimizer_bytes_per_parameter).to('byte')``."""
        master = Q_(0, "byte") if self.master_bytes_per_parameter is None else self.master_bytes_per_parameter
        return (2 * self.bytes_per_value + master + self.optimizer_bytes_per_parameter).to("byte")

    @property
    def adapter_state(self) -> Quantity:
        """Everything the adapters hold: weights, gradients, master copy, optimizer state.

        Equation: ``(adapter_count * bytes_per_trained_parameter).to('byte')``."""
        return (self.adapter_count * self.bytes_per_trained_parameter).to("byte")

    # Adapter state by array (added 2026-10-20): one resident copy of each.
    @property
    def adapter_weights(self) -> Quantity:
        """Adapter weight storage in bytes at the configured value width.

        Equation: ``(adapter_count * bytes_per_value).to('byte')``."""
        return (self.adapter_count * self.bytes_per_value).to("byte")

    @property
    def adapter_gradients(self) -> Quantity:
        """Adapter gradient storage in bytes at the configured value width.

        Equation: ``(adapter_count * bytes_per_value).to('byte')``."""
        return (self.adapter_count * self.bytes_per_value).to("byte")

    @property
    def adapter_master(self) -> Quantity:
        """Adapter master-copy storage in bytes, or zero without a master format.

        Equation: ``master = Q_(0, 'byte') if master_bytes_per_parameter is None else
        master_bytes_per_parameter; (adapter_count * master).to('byte')``."""
        master = Q_(0, "byte") if self.master_bytes_per_parameter is None else self.master_bytes_per_parameter
        return (self.adapter_count * master).to("byte")

    @property
    def adapter_optimizer_state(self) -> Quantity:
        """Adapter optimizer-state storage in bytes.

        Equation: ``(adapter_count * optimizer_bytes_per_parameter).to('byte')``."""
        return (self.adapter_count * self.optimizer_bytes_per_parameter).to("byte")

    @property
    def adapter_fraction(self) -> float:
        """Dimensionless adapter parameter count relative to all base-model parameters.

        Equation: ``adapter_count / workload.parameters.to('count').magnitude``."""
        return self.adapter_count / self.workload.parameters.to("count").magnitude

    # Adapter run against full fine-tuning (added 2026-10-20), state before activations.
    @property
    def full_state(self) -> Quantity:
        """State if every parameter trained: the same recipe with no adapters.

        Equation: ``variant(adapter_parameters=None, adapter_rank=None).total``."""
        return self.variant(adapter_parameters=None, adapter_rank=None).total

    @property
    def adapter_saved(self) -> Quantity:
        """Parameter-state bytes avoided by the configured adapter setup relative to full tuning.

        Equation: ``(full_state - total).to('byte')``."""
        return (self.full_state - self.total).to("byte")

    @property
    def adapter_reduction(self) -> float:
        """Full-state bytes over adapter-run bytes.

        Equation: ``(full_state / total).to('').magnitude``."""
        return (self.full_state / self.total).to("").magnitude

    @property
    def adapter_kept_share(self) -> float:
        """Dimensionless fraction of full-tuning parameter state retained by this setup.

        Equation: ``(total / full_state).to('').magnitude``."""
        return (self.total / self.full_state).to("").magnitude

    @property
    def frozen_share(self) -> float:
        """The frozen base's share of the adapter run's state.

        Equation: ``(weights / total).to('').magnitude``."""
        return (self.weights / self.total).to("").magnitude

    @property
    def adapter_reduction_ceiling(self) -> float:
        """The reduction as adapters shrink to nothing: bytes per trained over bytes per frozen parameter.

        Equation: ``(bytes_per_trained_parameter / bytes_per_value).to('').magnitude``."""
        return (self.bytes_per_trained_parameter / self.bytes_per_value).to("").magnitude

    @property
    def total(self) -> Quantity:
        """Parameter-state bytes: frozen base weights plus adapter state, or full training state and master weights.

        Equation: Adapter mode uses base_weight_bytes + adapter_state_bytes; full tuning
        sums weights, gradients, optimizer state, and master weights."""
        if self.adapter_count is not None:
            return (self.weights + self.adapter_state).to("byte")
        return sum(self._state()[1:], self._state()[0]) + self.master_weights

    @property
    def copies(self) -> float:
        """Total state in units of the weights.

        Equation: ``(total / weights).to('').magnitude``."""
        return (self.total / self.weights).to("").magnitude

    @property
    def fits(self) -> bool:
        """Whether parameter state alone fits the selected device; false if no hardware is supplied.

        Equation: ``hardware is not None and total <= hardware.memory.capacity``."""
        return self.hardware is not None and self.total <= self.hardware.memory.capacity

    @property
    def capacity_share(self) -> float:
        """State before activations as a share of ``hardware``'s memory.

        Equation: ``(total / hardware.memory.capacity).to('').magnitude``."""
        return (self.total / self.hardware.memory.capacity).to("").magnitude

    @property
    def training_capacity_multiple(self) -> float:
        """State and saved activations in units of ``hardware``'s memory.

        Equation: ``(training_total / hardware.memory.capacity).to('').magnitude``."""
        return (self.training_total / self.hardware.memory.capacity).to("").magnitude

    @property
    def resident_batch_limit(self) -> float:
        """Sequences whose activations fit beside the state in ``hardware``'s memory (added 2026-10-20).

        Equation: ``per_sequence = activations / batch_size; ((hardware.memory.capacity -
        total) / per_sequence).to('').magnitude``."""
        per_sequence = self.activations / self.batch_size
        return ((self.hardware.memory.capacity - self.total) / per_sequence).to("").magnitude

    @property
    def max_resident_batch(self) -> int:
        """Integer floor of the continuous resident-batch limit, without clamping infeasible values.

        Equation: ``_math.floor(resident_batch_limit)``."""
        return _math.floor(self.resident_batch_limit)

    @property
    def checkpoint(self) -> Quantity:
        """A resumable checkpoint: weights, master copy, and optimizer state; gradients are recomputed.

        Equation: ``(weights + master_weights + optimizer_state).to('byte')``."""
        return (self.weights + self.master_weights + self.optimizer_state).to("byte")

    def checkpoint_write_time(self, bandwidth: Quantity) -> Quantity:
        """Seconds to write ``checkpoint`` at ``bandwidth``.

        Equation: ``(checkpoint / bandwidth).to('second')``."""
        return (self.checkpoint / bandwidth).to("second")

    @property
    def update_bytes_per_parameter(self) -> Quantity:
        """Bytes one optimizer update moves per parameter: read the gradient, the updated
        weights, and the optimizer state; write the state and the weights back.

        Equation: ``p = workload.parameters.to('count').magnitude; updated = (master_weights
        if master_bytes_per_parameter is not None else weights) / p; (gradients / p + 2 *
        updated + 2 * optimizer_state / p).to('byte')``."""
        p = self.workload.parameters.to("count").magnitude
        updated = (self.master_weights if self.master_bytes_per_parameter is not None else self.weights) / p
        return (self.gradients / p + 2 * updated + 2 * self.optimizer_state / p).to("byte")

    @property
    def update_traffic(self) -> Quantity:
        """Full-model optimizer-update traffic in GB under the declared read/write accounting.

        Equation: ``(update_bytes_per_parameter *
        workload.parameters.to('count').magnitude).to('GB')``."""
        return (self.update_bytes_per_parameter * self.workload.parameters.to("count").magnitude).to("GB")

    @property
    def update_time(self) -> Quantity:
        """The optimizer update at ``hardware``'s memory bandwidth (elementwise, bandwidth-bound).

        Equation: ``(update_traffic / hardware.memory.bandwidth).to('ms')``."""
        return (self.update_traffic / self.hardware.memory.bandwidth).to("ms")


class DpSgdRunScenario(BaseScenario):
    """Inputs a DP-SGD privacy accountant needs for one Poisson-sampled run on ``dataset_size``
    examples; this type does not estimate epsilon.

    Notes
    -----
    Equation: Batch-to-dataset ratio specifies sampling; epochs times dataset size divided
    by batch size specifies steps.

    Assumptions: The caller supplies DP-SGD training choices and an intended privacy target.

    Excludes: Computation or certification of epsilon, gradient clipping behavior, and
    remainder-batch handling beyond integer truncation.

    References
    ----------
    Derivation: Batch-to-dataset ratio specifies sampling; epochs times dataset size divided
    by batch size specifies steps. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Cifar10DpSgdRun
    >>> round(float(scenario.sampling_rate), 4)
    0.08"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dataset_size: int
    batch_size: int
    clip_norm: float
    noise_multiplier: float
    epochs: int
    target_epsilon: float
    delta: float

    @property
    def sampling_rate(self) -> float:
        """Poisson sample probability per step, approximated by batch size divided by dataset size.

        Equation: ``batch_size / dataset_size``."""
        return self.batch_size / self.dataset_size

    @property
    def steps(self) -> int:
        """Integer training-step count for the declared epochs, truncating a fractional final
        batch.

        Equation: ``int(epochs * dataset_size / batch_size)``."""
        return int(self.epochs * self.dataset_size / self.batch_size)


class AdversarialTrainingPassesScenario(BaseScenario):
    """Work per batch of adversarial training, in standard forward-backward passes (equal cost,
    no reuse).

    Notes
    -----
    Equation: Attack and update pass counts add, then divide by standard-pass count for
    relative work.

    Assumptions: All declared forward-backward passes have equal cost and no reusable work.

    Excludes: Actual kernel runtime, cache reuse, and attack effectiveness.

    References
    ----------
    Derivation: Attack and update pass counts add, then divide by standard-pass count for
    relative work. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.PgdSevenTraining
    >>> round(float(scenario.passes), 4)
    8.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    attack_passes: int  # passes spent finding the perturbation
    update_passes: int  # passes spent on the update (2 for a doubled clean-plus-adversarial batch)
    standard_passes: int = 1  # one pass per batch for standard training

    @property
    def passes(self) -> int:
        """Total attack plus update passes per batch under equal-cost pass accounting.

        Equation: ``attack_passes + update_passes``."""
        return self.attack_passes + self.update_passes

    @property
    def slowdown(self) -> float:
        """Dimensionless adversarial pass count divided by standard training pass count.

        Work multiple over standard training.

        Equation: ``passes / standard_passes``."""
        return self.passes / self.standard_passes


class DpTrainingPlanScenario(BaseScenario):
    """An illustrative differential-privacy planning scenario for one training run of
    ``base_cost``.

    Each privacy budget is an assumed accountant result; the stronger budget is
    assumed to need ``cost_multiplier`` times the training, with the stated
    accuracy drops and overheads (percent) as planning assumptions.

    Notes
    -----
    Equation: Apply the declared multiplier to baseline cost; it is not derived from the
    target privacy budget.

    Assumptions: The privacy cost multiplier is a supplied illustrative planning input.

    Excludes: A DP accountant, epsilon certification, and measured training slowdown.

    References
    ----------
    Derivation: Apply the declared multiplier to baseline cost; it is not derived from the
    target privacy budget. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Gpt3PrivacyPlan
    >>> round(float(scenario.cost_strong.magnitude), 4)
    13800000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    epsilon_strong: float
    epsilon_moderate: float
    epsilon_weak: float
    base_cost: Quantity
    cost_multiplier: float
    accuracy_drop_strong: float  # percent
    accuracy_drop_moderate: float  # percent
    clipped_share: float  # percent of per-example gradients clipped
    overhead_moderate: float  # percent

    @property
    def cost_strong(self) -> Quantity:
        """USD training cost under the supplied privacy cost multiplier; no privacy accountant is
        evaluated.

        Equation: ``(base_cost * cost_multiplier).to('USD')``."""
        return (self.base_cost * self.cost_multiplier).to("USD")


class AttentionTilingScenario(BaseScenario):
    """One head's attention at ``seq_len`` tokens, whole or in square tiles, and what each kernel stores.

    Standard attention stores the scores and the probabilities (``seq_len``
    squared each) beside Q, K, V, and the output; FlashAttention stores a
    running maximum and sum per row instead.

    Notes
    -----
    Equation: ``score_matrix`` is the worked quantity; each property below states its equation and units.

    Assumptions: Dense square scores use fixed precision and tiles divide the sequence length exactly.

    Excludes: Kernel workspace, inter-head buffer reuse, and measured FlashAttention memory or speed.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.AttentionScores4096TokensFp32
    >>> value = case.score_matrix
    >>> print(f"{value:.3f~P}")
    67.109 MB"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    seq_len: int
    head_dim: int
    tile: int
    heads: int
    bytes_per_value: Quantity
    io_tensors: int = 4  # Q, K, V, and the output
    stored_square_matrices: int = 2  # scores and probabilities
    row_statistics: int = 2  # running maximum and sum

    @property
    def score_matrix(self) -> Quantity:
        """Storage in MB for one head's square attention-score matrix.

        Equation:
            return (self.seq_len ** 2 * self.bytes_per_value).to('MB')"""
        return (self.seq_len**2 * self.bytes_per_value).to("MB")

    @property
    def all_heads_scores(self) -> Quantity:
        """Storage in GB for the score matrices of all heads.

        Equation:
            return (self.heads * self.score_matrix).to('GB')"""
        return (self.heads * self.score_matrix).to("GB")

    @property
    def tile_bytes(self) -> Quantity:
        """Storage in KB for one square score tile.

        Equation:
            return (self.tile ** 2 * self.bytes_per_value).to('KB')"""
        return (self.tile**2 * self.bytes_per_value).to("KB")

    @property
    def tiles_per_side(self) -> int:
        """Whole tiles along one sequence axis; integer division drops any partial tile.

        Equation:
            return self.seq_len // self.tile"""
        return self.seq_len // self.tile

    @property
    def tile_count(self) -> int:
        """Whole square-tile count, tiles per side squared.

        Equation:
            return self.tiles_per_side ** 2"""
        return self.tiles_per_side**2

    @property
    def qkv_bytes(self) -> Quantity:
        """Q, K, and V storage in MB for one head.

        Q, K, and V: three tensors of seq_len x head_dim.

        Equation:
            return (3 * self.seq_len * self.head_dim * self.bytes_per_value).to('MB')"""
        return (3 * self.seq_len * self.head_dim * self.bytes_per_value).to("MB")

    @property
    def io_bytes(self) -> Quantity:
        """Configured input/output tensor storage in MB for one head.

        Equation:
            return (self.io_tensors * self.seq_len * self.head_dim * self.bytes_per_value).to('MB')"""
        return (self.io_tensors * self.seq_len * self.head_dim * self.bytes_per_value).to("MB")

    @property
    def stored_standard(self) -> Quantity:
        """Standard-attention storage in MB including materialized square matrices.

        Equation:
            return (self.io_bytes + self.stored_square_matrices * self.score_matrix).to('MB')"""
        return (self.io_bytes + self.stored_square_matrices * self.score_matrix).to("MB")

    @property
    def stored_flash(self) -> Quantity:
        """Tiled-attention storage in MB including per-row statistics instead of square matrices.

        Equation:
            return (self.io_bytes + self.row_statistics * self.seq_len * self.bytes_per_value).to('MB')"""
        return (self.io_bytes + self.row_statistics * self.seq_len * self.bytes_per_value).to("MB")

    @property
    def stored_ratio(self) -> float:
        """Dimensionless standard-to-tiled storage ratio.

        Equation:
            return (self.stored_standard / self.stored_flash).to('').magnitude"""
        return (self.stored_standard / self.stored_flash).to("").magnitude


class TokenizedBatchFeedScenario(BaseScenario):
    """A host that tokenizes each batch and copies it to the accelerator, against the device's step.

    Without prefetching the host and the device run back to back, so the
    device is busy ``compute_time`` out of every ``prep_time + compute_time``.

    Notes
    -----
    Equation: ``tokenization_time`` is the worked quantity; each property below states its equation and units.

    Assumptions: Preparation and compute are serial; multiple workers scale tokenization ideally in the separate parallel estimate.

    Excludes: Overlap, tokenizer imbalance, and parallel workers in busy_fraction, which uses single-core prep_time.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt2XlSingleCoreTokenizerFeed
    >>> value = case.tokenization_time
    >>> print(f"{value:.3f~P}")
    0.655 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    compute_time: Quantity
    tokens_per_batch: int
    token_rate: Quantity  # tokens per second on one host core
    bytes_per_token: Quantity  # token ids as stored for the copy
    link_bandwidth: Quantity
    workers: int = 1

    @property
    def tokenization_time(self) -> Quantity:
        """Single-core tokenization duration in seconds for one batch.

        Equation:
            return (self.tokens_per_batch / self.token_rate).to('s')"""
        return (self.tokens_per_batch / self.token_rate).to("s")

    @property
    def parallel_tokenization_time(self) -> Quantity:
        """Ideal parallel tokenization duration in seconds, dividing by worker count.

        Equation:
            return (self.tokenization_time / self.workers).to('s')"""
        return (self.tokenization_time / self.workers).to("s")

    @property
    def batch_bytes(self) -> Quantity:
        """Token-ID storage in bytes for one batch transfer.

        Equation:
            return (self.tokens_per_batch * self.bytes_per_token).to('byte')"""
        return (self.tokens_per_batch * self.bytes_per_token).to("byte")

    @property
    def transfer_time(self) -> Quantity:
        """Host-device transfer duration in seconds at the supplied bandwidth.

        Equation:
            return (self.batch_bytes / self.link_bandwidth).to('s')"""
        return (self.batch_bytes / self.link_bandwidth).to("s")

    @property
    def prep_time(self) -> Quantity:
        """Sequential single-core preparation duration in seconds, tokenization plus transfer.

        Single-core preparation: tokenization plus the copy.

        Equation:
            return (self.tokenization_time + self.transfer_time).to('s')"""
        return (self.tokenization_time + self.transfer_time).to("s")

    @property
    def busy_fraction(self) -> float:
        """Dimensionless accelerator busy share with serial single-core preparation and compute.

        Share of time the device computes when preparation and compute run in sequence.

        Equation:
            return (self.compute_time / (self.prep_time + self.compute_time)).to('').magnitude"""
        return (self.compute_time / (self.prep_time + self.compute_time)).to("").magnitude


class TargetMaskScenario(BaseScenario):
    """One serialized training example by segment, and the responses kept as context only.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.TargetMaskModel.solve``.

    Assumptions: Inputs are routed to ``TargetMaskModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to TargetMaskModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RecoveryRecordTargetMask
    >>> value = case.result.targets_without_rationale
    >>> print(value)
    800"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    prompt: int  # includes the initial boundary position
    task: int
    rationale: int  # optional reviewed rationale
    response: int  # policy responses with their stop tokens
    observation: int
    unselected_response: int  # responses retained as context, not targets

    @property
    def result(self):
        """Evaluate the stated target mask case through ``TargetMaskModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.TargetMaskModel.solve``."""
        from ..solvers import TargetMaskModel

        return TargetMaskModel().solve(
            self.prompt, self.task, self.rationale, self.response, self.observation, self.unselected_response
        )


class PerInvocationSerializationScenario(BaseScenario):
    """A ``turns``-turn trajectory stored as one example per model call, each turn adding ``tokens_per_turn``.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.PerInvocationSerializationModel.solve``.

    Assumptions: Inputs are routed to ``PerInvocationSerializationModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to PerInvocationSerializationModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairTrajectoryPerCallExamples
    >>> value = case.result.expansion
    >>> print(f"{value:.3g}")
    15.5"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    turns: int
    tokens_per_turn: int

    @property
    def result(self):
        """Evaluate the stated per invocation serialization case through ``PerInvocationSerializationModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.PerInvocationSerializationModel.solve``."""
        from ..solvers import PerInvocationSerializationModel

        return PerInvocationSerializationModel().solve(self.turns, self.tokens_per_turn)


class LossNormalizationScenario(BaseScenario):
    """A long trajectory split into ``focus_pieces`` records, batched with ``other_examples`` short ones.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.LossNormalizationModel.solve``.

    Assumptions: Inputs are routed to ``LossNormalizationModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to LossNormalizationModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairTraceWithLookupWeighting
    >>> value = case.result.focus_token_share
    >>> print(f"{value:.3g}")
    0.97"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    focus_targets: int
    other_examples: int
    other_targets: int
    focus_pieces: int = 1

    @property
    def result(self):
        """Evaluate the stated loss normalization case through ``LossNormalizationModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.LossNormalizationModel.solve``."""
        from ..solvers import LossNormalizationModel

        return LossNormalizationModel().solve(
            self.focus_targets, self.other_examples, self.other_targets, self.focus_pieces
        )


class SequencePackingScenario(BaseScenario):
    """Stored example lengths against ``capacity``-token buffers.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.SequencePackingModel.solve``.

    Assumptions: Inputs are routed to ``SequencePackingModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to SequencePackingModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.EightExamplePackingBatch
    >>> value = case.result.packed_occupancy
    >>> print(f"{value:.3g}")
    0.766"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    lengths: List[int]
    capacity: int

    @property
    def result(self):
        """Evaluate the stated sequence packing case through ``SequencePackingModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.SequencePackingModel.solve``."""
        from ..solvers import SequencePackingModel

        return SequencePackingModel().solve(self.lengths, self.capacity)


class PreferencePairScenario(BaseScenario):
    """Log-probability sums over a preferred and a dispreferred record's selected tokens.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.PreferencePairModel.solve``.

    Assumptions: Inputs are routed to ``PreferencePairModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to PreferencePairModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairPreferencePair
    >>> value = case.result.loss
    >>> print(f"{value:.3g}")
    0.437"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    policy_preferred: float
    reference_preferred: float
    policy_dispreferred: float
    reference_dispreferred: float
    beta: float

    @property
    def result(self):
        """Evaluate the stated preference pair case through ``PreferencePairModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.PreferencePairModel.solve``."""
        from ..solvers import PreferencePairModel

        return PreferencePairModel().solve(
            self.policy_preferred,
            self.reference_preferred,
            self.policy_dispreferred,
            self.reference_dispreferred,
            self.beta,
        )


class RewardCompositionScenario(BaseScenario):
    """One update's rollouts scored by a verifier that accepts some failures.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RewardCompositionModel.solve``.

    Assumptions: Inputs are routed to ``RewardCompositionModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RewardCompositionModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.VerifierFalseAcceptsAtTenPercentPass
    >>> value = case.result.false_share
    >>> print(f"{value:.3g}")
    0.184"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tasks: int
    group: int  # rollouts per task
    pass_rate: float  # true pass probability of one rollout
    false_accept_rate: float  # P(accept | failure)
    valid_accept_rate: float  # P(accept | pass)

    @property
    def result(self):
        """Evaluate the stated reward composition case through ``RewardCompositionModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RewardCompositionModel.solve``."""
        from ..solvers import RewardCompositionModel

        return RewardCompositionModel().solve(
            self.tasks, self.group, self.pass_rate, self.false_accept_rate, self.valid_accept_rate
        )


class CenteredRewardScenario(BaseScenario):
    """Binary terminal rewards centered on a fixed baseline.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.CenteredRewardModel.solve``.

    Assumptions: Inputs are routed to ``CenteredRewardModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to CenteredRewardModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.BinaryRewardFixedBaseline
    >>> value = case.result.passing_score
    >>> print(f"{value:.3g}")
    0.75"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline: float
    passing_reward: float
    failing_reward: float

    @property
    def result(self):
        """Evaluate the stated centered reward case through ``CenteredRewardModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.CenteredRewardModel.solve``."""
        from ..solvers import CenteredRewardModel

        return CenteredRewardModel().solve(self.baseline, self.passing_reward, self.failing_reward)


class ForkValueScenario(BaseScenario):
    """Binary continuations forked before and after one turn, with their pass counts.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.ForkValueModel.solve``.

    Assumptions: Inputs are routed to ``ForkValueModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to ForkValueModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DebuggerTurnForkValue
    >>> value = case.result.difference
    >>> print(f"{value:.3g}")
    0.55"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    continuations: int  # per fork
    passes_before: int
    passes_after: int

    @property
    def result(self):
        """Evaluate the stated fork value case through ``ForkValueModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.ForkValueModel.solve``."""
        from ..solvers import ForkValueModel

        return ForkValueModel().solve(self.continuations, self.passes_before, self.passes_after)


class ContinuationForkCostScenario(BaseScenario):
    """A trajectory whose value is sampled by forking continuations at chosen turns.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.ContinuationForkCostModel.solve``.

    Assumptions: Inputs are routed to ``ContinuationForkCostModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to ContinuationForkCostModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ThirtyTurnForksAtEveryBoundary
    >>> value = case.result.multiple
    >>> print(f"{value:.3g}")
    117"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    turns: int
    tokens_per_turn: int
    branches: int  # continuations per fork
    boundaries: Optional[List[int]] = None  # completed turns before each fork; None forks at every one

    @property
    def result(self):
        """Evaluate the stated continuation fork cost case through ``ContinuationForkCostModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.ContinuationForkCostModel.solve``."""
        from ..solvers import ContinuationForkCostModel

        return ContinuationForkCostModel().solve(self.turns, self.tokens_per_turn, self.branches, self.boundaries)


class GroupCoefficientsScenario(BaseScenario):
    """The rewards of one group of rollouts and the coefficient convention applied to them.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.GroupCoefficientsModel.solve``.

    Assumptions: Inputs are routed to ``GroupCoefficientsModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to GroupCoefficientsModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairGroupOnePassOfThree
    >>> value = case.result.std
    >>> print(f"{value:.3g}")
    0.471"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rewards: List[float]
    stabilizer: float  # epsilon added to the dispersion
    bessel: bool = False  # divide the variance by G - 1 instead of G

    @property
    def result(self):
        """Evaluate the stated group coefficients case through ``GroupCoefficientsModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.GroupCoefficientsModel.solve``."""
        from ..solvers import GroupCoefficientsModel

        return GroupCoefficientsModel().solve(self.rewards, self.stabilizer, self.bessel)


class LengthNormalizationScenario(BaseScenario):
    """Two rollouts with one group coefficient and different lengths, averaged per rollout.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.LengthNormalizationModel.solve``.

    Assumptions: Inputs are routed to ``LengthNormalizationModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to LengthNormalizationModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ShortAndLongFailureLengths
    >>> value = case.result.per_token_ratio
    >>> print(f"{value:.3g}")
    32"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    short_tokens: int
    long_tokens: int

    @property
    def result(self):
        """Evaluate the stated length normalization case through ``LengthNormalizationModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.LengthNormalizationModel.solve``."""
        from ..solvers import LengthNormalizationModel

        return LengthNormalizationModel().solve(self.short_tokens, self.long_tokens)


class RolloutStepBudgetScenario(BaseScenario):
    """One update's tasks, group size, rollout length, and the policy's pass rate on the mix.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RolloutStepBudgetModel.solve``.

    Assumptions: Inputs are routed to ``RolloutStepBudgetModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RolloutStepBudgetModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.HardTaskRolloutStep
    >>> value = case.result.step_tokens
    >>> print(value)
    62914560"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tasks: int
    group: int
    turns: int
    tokens_per_turn: int
    pass_rate: float

    @property
    def result(self):
        """Evaluate the stated rollout step budget case through ``RolloutStepBudgetModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RolloutStepBudgetModel.solve``."""
        from ..solvers import RolloutStepBudgetModel

        return RolloutStepBudgetModel().solve(self.tasks, self.group, self.turns, self.tokens_per_turn, self.pass_rate)


class RepeatedVerificationScenario(BaseScenario):
    """A timing-sensitive check repeated on one candidate, with an all-pass rule.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RepeatedVerificationModel.solve``.

    Assumptions: Inputs are routed to ``RepeatedVerificationModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RepeatedVerificationModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.FlakyRaceTestAllPassRule
    >>> value = case.result.group_false_pass
    >>> print(f"{value:.3g}")
    0.972"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    incorrect_pass: float  # P(one trial passes | incorrect candidate)
    correct_pass: float  # P(one trial passes | correct candidate)
    trials: int
    group: int  # candidates graded once each in one group

    @property
    def result(self):
        """Evaluate the stated repeated verification case through ``RepeatedVerificationModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RepeatedVerificationModel.solve``."""
        from ..solvers import RepeatedVerificationModel

        return RepeatedVerificationModel().solve(self.incorrect_pass, self.correct_pass, self.trials, self.group)


class RatioClipScenario(BaseScenario):
    """One token's old probability under a symmetric ratio clip.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RatioClipModel.solve``.

    Assumptions: Inputs are routed to ``RatioClipModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RatioClipModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DominantTokenRatioClip
    >>> value = case.result.ceiling
    >>> print(f"{value:.3g}")
    1.08"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    epsilon: float
    probability: float

    @property
    def result(self):
        """Evaluate the stated ratio clip case through ``RatioClipModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RatioClipModel.solve``."""
        from ..solvers import RatioClipModel

        return RatioClipModel().solve(self.epsilon, self.probability)


class RolloutSlotCycleScenario(BaseScenario):
    """Sandbox slots running one episode at a time, with a cold and a snapshot reset.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.RolloutSlotCycleModel.solve``.

    Assumptions: Inputs are routed to ``RolloutSlotCycleModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to RolloutSlotCycleModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairRolloutSlotResets
    >>> value = case.result.capacity_ratio
    >>> print(f"{value:.3g}")
    1.77"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    slots: int
    decisions: int  # model calls per episode
    model_time: Quantity  # per decision
    tool_time: Quantity  # per decision
    assessment_time: Quantity  # per episode
    cold_reset: Quantity
    snapshot_reset: Quantity

    @property
    def result(self):
        """Evaluate the stated rollout slot cycle case through ``RolloutSlotCycleModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.RolloutSlotCycleModel.solve``."""
        from ..solvers import RolloutSlotCycleModel

        return RolloutSlotCycleModel().solve(
            self.slots,
            self.decisions,
            self.model_time,
            self.tool_time,
            self.assessment_time,
            self.cold_reset,
            self.snapshot_reset,
        )


class GroupBarrierScenario(BaseScenario):
    """The rollout latencies of one synchronous group.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.GroupBarrierModel.solve``.

    Assumptions: Inputs are routed to ``GroupBarrierModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to GroupBarrierModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RolloutGroupOfFourBarrier
    >>> value = case.result.idle_share
    >>> print(f"{value:.3g}")
    0.653"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    latencies: List[Quantity]

    @property
    def result(self):
        """Evaluate the stated group barrier case through ``GroupBarrierModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.GroupBarrierModel.solve``."""
        from ..solvers import GroupBarrierModel

        return GroupBarrierModel().solve(self.latencies)


class VerifierPoolScenario(BaseScenario):
    """Completed trajectories arriving in batches, each verified by repeated trials.

    Notes
    -----
    Equation: Composition of ``mlsysim.solvers.VerifierPoolModel.solve``.

    Assumptions: Inputs are routed to ``VerifierPoolModel.solve``; their documented regimes and units apply to this configuration.

    Excludes: Additional mechanisms beyond the linked solver contracts; the scenario does not collect measurements.

    References
    ----------
    Derivation: This scenario binds named inputs to VerifierPoolModel. The linked solver contracts own equations, sources, and result-unit definitions.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RepairVerifierPool
    >>> value = case.result.slots
    >>> print(value)
    960"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    trajectories: int  # per interval
    interval: Quantity
    trials: int  # per trajectory; all execute in this sizing case
    trial_time: Quantity
    target_utilization: float

    @property
    def result(self):
        """Evaluate the stated verifier pool case through ``VerifierPoolModel.solve``; returns its typed result and quantity units.

        Calculation: ``mlsysim.solvers.VerifierPoolModel.solve``."""
        from ..solvers import VerifierPoolModel

        return VerifierPoolModel().solve(
            self.trajectories, self.interval, self.trials, self.trial_time, self.target_utilization
        )


class PerExampleGradientScenario(BaseScenario):
    """Naive DP-SGD keeps one gradient per example: ``bytes_per_param`` per parameter of each
    named model, ``batch_size`` copies per step.

    Notes
    -----
    Equation: Naive gradient storage is ``parameters*bytes_per_param`` per example and
    batch_size times that for a full batch.

    Assumptions: Each example retains a separate dense full-parameter gradient at the
    selected precision.

    Excludes: Ghost clipping, microbatching, optimizer state, activations, and a
    differential-privacy guarantee.

    References
    ----------
    Derivation: Naive gradient storage is ``parameters*bytes_per_param`` per example and
    batch_size times that for a full batch. This is declared scenario accounting rather than
    a fitted empirical law.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.DpSgdPerExampleGradients
    >>> value = scenario.per_batch(next(iter(scenario.models)))
    >>> str(value.units)
    'B'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    models: Dict[str, Workload]
    batch_size: int
    bytes_per_param: Quantity

    def per_example(self, name: str) -> Quantity:
        """Gradient bytes for named model parameters at the selected gradient precision.

        Equation: ``models[name].size_in_bytes(bytes_per_param).to('byte')``.
        """
        return self.models[name].size_in_bytes(self.bytes_per_param).to("byte")

    def per_batch(self, name: str) -> Quantity:
        """Naive per-example gradient bytes across the full batch; no streaming or clipping
        optimization.

        Equation: ``(per_example(name) * batch_size).to('byte')``.
        """
        return (self.per_example(name) * self.batch_size).to("byte")
