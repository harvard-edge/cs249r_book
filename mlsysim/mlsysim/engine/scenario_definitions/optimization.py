"""Execution bottlenecks, hardware utilization, and resource placement.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...systems.types import Fleet
from ...datasets.types import DatasetProfile
from .base import BaseScenario, _usd_per_hour
from .reliability import CheckpointIntervalScenario


class AcceleratorComparison(BaseScenario):
    """``candidate`` against ``baseline``: peak, power, and peak per watt.

    Peak rates compare at one convention (dense FP16 unless ``precision`` says
    otherwise); a ratio across precisions mixes a format change into the
    hardware change.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: The same arithmetic workload runs at comparable utilization and selected
    precision; rated TDP supplies the power comparison.

    Excludes: Memory limits, communication, and measured workload power are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.V100ToB200
    >>> round(case.peak_ratio, 4)
    18.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    baseline: HardwareNode
    candidate: HardwareNode
    precision: Optional[str] = None
    baseline_device_hours: Optional[float] = None  # a job's size measured on the baseline

    def _peak(self, hw: HardwareNode) -> Quantity:
        if self.precision is None:
            return hw.compute.peak_flops
        return hw.compute.precision_flops[self.precision]

    @property
    def peak_ratio(self) -> float:
        """Candidate peak arithmetic rate divided by baseline peak rate at the selected precision.
        Equation: ``_peak(candidate) / _peak(baseline)``. Units: dimensionless; achieved
        efficiency is not modeled.
        """
        return (self._peak(self.candidate) / self._peak(self.baseline)).to("").magnitude

    @property
    def tdp_ratio(self) -> float:
        """Candidate rated power relative to the baseline. Equation: ``candidate.tdp /
        baseline.tdp``. Units: dimensionless; TDP is not measured workload power.
        """
        return (self.candidate.tdp / self.baseline.tdp).to("").magnitude

    @property
    def efficiency_ratio(self) -> float:
        """Relative peak arithmetic work per rated watt. Equation: ``peak_ratio / tdp_ratio``.
        Units: dimensionless; neither memory limits nor utilization are included.
        """
        return self.peak_ratio / self.tdp_ratio

    def equivalent_hours(self, baseline_hours: float) -> float:
        """Candidate device-hours for the same ideal arithmetic work. Equation: ``baseline_hours /
        peak_ratio``. Units: the input's hour unit; assumes equal utilization at the compared
        peaks.

        Candidate device-hours that match ``baseline_hours`` at equal sustained share of peak.
        """
        return baseline_hours / self.peak_ratio

    @property
    def candidate_device_hours(self) -> float:
        """Candidate device-hours corresponding to the scenario's baseline budget. Equation:
        ``equivalent_hours(baseline_device_hours)``. Units: device-hours under ideal peak-rate
        scaling.

        Candidate device-hours for ``baseline_device_hours`` at equal sustained share of peak.
        """
        return self.equivalent_hours(self.baseline_device_hours)


class InputSpaceScenario(BaseScenario):
    """Every possible input of ``dataset``'s image shape: ``levels`` values per channel
    element.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Every image channel element independently takes one of the stated discrete
    levels.

    Excludes: The counted input space ignores natural-image structure and does not measure
    dataset representativeness.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ImageNetInputSpace
    >>> round(case.elements, 4)
    150528
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dataset: DatasetProfile
    levels: int  # values one 8-bit channel element can take

    @property
    def elements(self) -> int:
        """Independent channel values in one image. Equation: image width times height times
        channel count. Units: scalar elements; image geometry comes from the dataset entry.
        """
        d = self.dataset
        return d.image_width * d.image_height * d.image_channels

    @property
    def log10_inputs(self) -> float:
        """Base-ten logarithm of the discrete image input space. Equation: ``elements *
        log10(levels)``. Units: dimensionless; all element assignments are counted
        independently.
        """
        import math

        return self.elements * math.log10(self.levels)

    @property
    def digits(self) -> int:
        """Decimal digits required to write the input-space cardinality. Equation:
        ``floor(log10_inputs) + 1``. Units: digits; this avoids materializing the enormous
        integer.
        """
        import math

        return math.floor(self.log10_inputs) + 1

    @property
    def test_examples(self) -> int:
        """Registered validation-set size as an integer. Equation: convert
        ``dataset.validation_examples`` to count and then integer. Units: examples; this is
        sourced dataset metadata, not input-space coverage.
        """
        return int(self.dataset.validation_examples.to("count").magnitude)


class ClusterTrainingTimeScenario(BaseScenario):
    """Training ``workload`` on ``n_devices`` x ``hardware`` at one end-to-end utilization
    (compute-bound).

    Optional (added 2026-10-20): ``tokens_per_step`` turns the run into steps
    (a step costs 6P FLOPs per token); ``hourly_price`` is the rental price of
    one device for an hour. Energy is every device at its TDP for the whole run.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.transformer.calc_transformer_training_flops``.

    Assumptions: The registered training work is compute bound at a fixed aggregate
    delivered rate. The supplied efficiency must already absorb any intended end-to-end
    losses.

    Excludes: Memory capacity and explicit communication scheduling are not checked; TDP
    energy is modeled, not measured.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.transformer.calc_transformer_training_flops``. The methods
    below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3TrainingTime1KA100
    >>> str(case.peak_rate)
    '312.0 TFLOPs/s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    n_devices: int
    efficiency: float
    tokens_per_step: Optional[int] = None
    hourly_price: Optional[Quantity] = None
    # Optional (added 2026-10-20): run on the peak of another number format
    # (a key of hardware.compute.precision_flops, "fp32" or "tf32"); None uses
    # the dense 16-bit peak, hardware.compute.peak_flops.
    peak_precision: Optional[str] = None

    @property
    def peak_rate(self) -> Quantity:
        """Per-device peak arithmetic rate at the selected precision. Equation: catalog default
        when no override is supplied, otherwise ``precision_flops[peak_precision]``. Units:
        FLOP/s; neither utilization nor device count is included yet.
        """
        if self.peak_precision is None:
            return self.hardware.compute.peak_flops
        return self.hardware.compute.precision_flops[self.peak_precision]

    def _time(self, ops) -> Quantity:
        from ...physics.performance import dTime

        return dTime(
            total_ops=ops,
            num_devices=self.n_devices,
            peak_flops_per_device=self.peak_rate,
            efficiency_eta=self.efficiency,
        )

    @property
    def training_time(self) -> Quantity:
        """Compute-bound duration for the registered total training work. Equation:
        ``_time(workload.training_ops)`` using work divided by aggregate delivered rate. Units:
        days; communication must already be reflected in efficiency.
        """
        return self._time(self.workload.training_ops).to("day")

    @property
    def delivered_rate(self) -> Quantity:
        """Aggregate arithmetic capacity after the supplied efficiency. Equation: ``n_devices *
        peak_rate * efficiency``. Units: TFLOP/s; efficiency is assumed rather than learned from
        measurements here.
        """
        return (self.n_devices * self.peak_rate * self.efficiency).to("TFLOP/s")

    @property
    def training_tokens(self) -> Quantity:
        """Parameter-dominated token count implied by total training work. Equation: total FLOPs
        divided by ``calc_transformer_training_flops(parameters, 1)``. Units: count; this
        inverts the helper's accounting approximation.
        """
        from ...physics.transformer import calc_transformer_training_flops

        per_token = calc_transformer_training_flops(self.workload.parameters, 1)
        return Q_((self.workload.training_ops / per_token).to("").magnitude, "count")

    @property
    def step_operations(self) -> Quantity:
        """Parameter-dominated work for the stated tokens per step. Equation:
        ``mlsysim.physics.transformer.calc_transformer_training_flops(parameters,
        tokens_per_step)``. Units: TFLOPs; attention-over-context is excluded by the helper.
        """
        from ...physics.transformer import calc_transformer_training_flops

        return calc_transformer_training_flops(self.workload.parameters, self.tokens_per_step).to("TFLOP")

    @property
    def step_time(self) -> Quantity:
        """Compute-bound duration of one training step. Equation: ``_time(step_operations)``.
        Units: seconds; the same delivered rate applies throughout the run.
        """
        return self._time(self.step_operations).to("second")

    @property
    def steps(self) -> float:
        """Number of equal-work steps in the training budget. Equation: ``workload.training_ops /
        step_operations``. Units: steps returned as a scalar; a partial final step is not
        rounded.
        """
        return (self.workload.training_ops / self.step_operations).to("").magnitude

    @property
    def energy(self) -> Quantity:
        """Constant-TDP accelerator energy over the modeled training duration. Equation:
        ``n_devices * hardware.tdp * training_time``. Units: kWh; host/facility overhead and
        varying power are excluded.
        """
        return (self.n_devices * self.hardware.tdp * self.training_time).to("kWh")

    @property
    def step_energy(self) -> Quantity:
        """Constant-TDP accelerator energy for one modeled step. Equation: ``n_devices *
        hardware.tdp * step_time``. Units: joules; this is an estimate, not meter data.
        """
        return (self.n_devices * self.hardware.tdp * self.step_time).to("joule")

    @property
    def device_hours(self) -> Quantity:
        """Aggregate accelerator-time allocated to training. Equation: ``n_devices *
        training_time``. Units: device-hours; every device is counted throughout the run.
        """
        return (self.n_devices * self.training_time).to("hour")

    @property
    def rental_cost(self) -> Quantity:
        """Rental charge for modeled aggregate accelerator-time. Equation: ``device_hours *
        hourly_price``. Units: USD; storage and network charges are excluded.
        """
        return (self.device_hours * self.hourly_price).to("USD")


class IronLawScenario(BaseScenario):
    """The iron law's three terms for one task, and an upgrade to the memory or compute term.

    Give the memory term as ``memory_time`` or as ``data_volume`` over ``bandwidth``,
    and the compute term as ``compute_time`` or as ``operations`` over ``effective_rate``
    (peak times utilization).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.IronLawModel``.

    Assumptions: Explicit stage durations take precedence over work/rate reconstruction; the
    solver combines the three supplied timing terms under its stated overlap model.

    Excludes: A stated effective rate is not inferred from kernels, and unlisted runtime
    stages are absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.IronLawModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IronLawMemoryBoundExample
    >>> str(case.memory_time)
    '20.0 ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    overhead: Quantity
    memory_time_stated: Optional[Quantity] = None
    data_volume: Optional[Quantity] = None
    bandwidth: Optional[Quantity] = None
    compute_time_stated: Optional[Quantity] = None
    operations: Optional[Quantity] = None
    effective_rate: Optional[Quantity] = None
    memory_speedup: float = 1.0
    compute_speedup: float = 1.0

    @property
    def memory_time(self) -> Quantity:
        """Explicit memory duration, or bytes divided by bandwidth when absent. Equation:
        ``memory_time_stated`` else ``data_volume / bandwidth``. Units: the overhead's time
        unit; bandwidth is treated as constant.
        """
        if self.memory_time_stated is not None:
            return self.memory_time_stated
        return (self.data_volume / self.bandwidth).to(self.overhead.units)

    @property
    def compute_time(self) -> Quantity:
        """Explicit compute duration, or work divided by effective rate when absent. Equation:
        ``compute_time_stated`` else ``operations / effective_rate``. Units: the overhead's time
        unit; effective rate is supplied.
        """
        if self.compute_time_stated is not None:
            return self.compute_time_stated
        return (self.operations / self.effective_rate).to(self.overhead.units)

    @property
    def upgraded_bandwidth(self) -> Quantity:
        """Bandwidth after the hypothetical memory-side upgrade. Equation: ``bandwidth *
        memory_speedup``. Units: bandwidth; other resources remain fixed.
        """
        return self.bandwidth * self.memory_speedup

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.IronLawModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import IronLawModel

        return IronLawModel().solve(
            self.memory_time, self.compute_time, self.overhead, self.memory_speedup, self.compute_speedup
        )


class OffloadDeadlineScenario(BaseScenario):
    """One input sent over an uplink to a remote model, against a per-frame deadline (lower
    bound).

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Transmission, remote network time, and remote compute add in sequence; one
    frame period is the deadline.

    Excludes: Compression compute, queueing not already in the inputs, and return-payload
    serialization are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Drone4KObstacleOffload
    >>> str(case.transmit_time)
    '1990.656 ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    payload: Quantity
    uplink_bandwidth: Quantity
    network_latency: Quantity
    remote_compute_time: Quantity
    frame_rate: Quantity
    compression: float = 1.0  # payload bytes divided by this before sending

    @property
    def transmit_time(self) -> Quantity:
        """Serialization of the compressed input on the uplink. Equation: ``payload / compression /
        uplink_bandwidth``. Units: milliseconds; compression is a size ratio and codec time is
        excluded.
        """
        return (self.payload / self.compression / self.uplink_bandwidth).to("millisecond")

    @property
    def remote_response(self) -> Quantity:
        """Remote network-plus-compute duration after input transmission. Equation:
        ``network_latency + remote_compute_time``. Units: milliseconds; queueing is absent
        unless already included in the inputs.
        """
        return (self.network_latency + self.remote_compute_time).to("millisecond")

    @property
    def remote_lower_bound(self) -> Quantity:
        """Serial lower bound on offloaded response time. Equation: ``transmit_time +
        remote_response``. Units: milliseconds; codec, additional queueing, and output transfer
        are excluded.
        """
        return (self.transmit_time + self.remote_response).to("millisecond")

    @property
    def deadline(self) -> Quantity:
        """Per-frame time budget implied by the frame rate. Equation: ``1 / frame_rate``. Units:
        milliseconds; frames are treated as uniformly spaced.
        """
        return (1 / self.frame_rate).to("millisecond")

    @property
    def deadline_ratio(self) -> float:
        """Modeled remote lower bound relative to one frame period. Equation: ``remote_lower_bound
        / deadline``. Units: dimensionless; values above one already exceed the budget.
        """
        return (self.remote_lower_bound / self.deadline).to("").magnitude


class SingleNodeInferenceScenario(BaseScenario):
    """One inference of ``workload`` on ``hardware`` (roofline profile from SingleNodeModel).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.SingleNodeModel``.

    Assumptions: The generic SingleNodeModel controls work, traffic, and feasibility;
    optional precision and efficiency override its defaults.

    Excludes: The generic activation-traffic heuristic is not a KV-aware serving trace.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.SingleNodeModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50A100Batch1
    >>> type(case.result).__name__
    'PerformanceProfile'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    batch_size: int = 1
    precision: Optional[str] = None  # None: the solver's default
    efficiency: Optional[float] = None

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.SingleNodeModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import SingleNodeModel

        kw = {}
        if self.precision is not None:
            kw["precision"] = self.precision
        if self.efficiency is not None:
            kw["efficiency"] = self.efficiency
        return SingleNodeModel().solve(self.workload, self.hardware, batch_size=self.batch_size, **kw)

    @property
    def traffic(self) -> Quantity:
        """Memory bytes implied by the generic roofline profile. Equation: ``result.latency_memory
        * result.peak_bw_actual``. Units: bytes; this inherits the engine's activation-traffic
        heuristic rather than a KV trace.
        """
        p = self.result
        return (p.latency_memory * p.peak_bw_actual).to("byte")

    @property
    def intensity(self) -> Quantity:
        """Inference work per modeled byte moved for the batch. Equation:
        ``workload.inference_flops * batch_size / traffic``. Units: FLOPs/byte; requires a
        workload with stated forward work.
        """
        return (self.workload.inference_flops * self.batch_size / self.traffic).to("flop/byte")

    @property
    def ridge(self) -> Quantity:
        """Peak compute-to-bandwidth ratio of the evaluated device profile. Equation:
        ``result.peak_flops_actual / result.peak_bw_actual``. Units: FLOPs/byte; this is the
        peak ridge, not an efficiency-adjusted ridge.
        """
        p = self.result
        return (p.peak_flops_actual / p.peak_bw_actual).to("flop/byte")


class AmdahlScenario(BaseScenario):
    """A sequential request path whose ``stage`` is made ``speedup`` times faster.

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.AmdahlModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.AmdahlModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.AmdahlModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.AmdahlModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ModelSpeedupRequestPath
    >>> type(case.result).__name__
    'AmdahlResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    stages: Dict[str, Quantity]
    stage: str
    speedup: float

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.AmdahlModel.solve``. Equation: pass the
        stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import AmdahlModel

        return AmdahlModel().solve(self.stages, self.stage, self.speedup)


class RooflinePointScenario(BaseScenario):
    """A workload of ``intensity`` on ``hardware``'s roofline: attainable rate and share of
    peak.

    A measured kernel may be placed from its counter readings instead
    (``achieved_rate`` and ``achieved_bandwidth``, added 2026-10-20): its
    intensity is their ratio and ``bandwidth_utilization`` its share of peak
    memory bandwidth.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Attainable arithmetic throughput is bounded by catalog compute peak and
    line-rate bandwidth times intensity.

    Excludes: Dispatch overhead, reduced utilization, and cache behavior are absent from
    this peak bound.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.LayerNormMeasuredH100
    >>> str(case.point_intensity)
    '1.7857142857142858 flop/B'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    hardware: HardwareNode
    intensity: Optional[Quantity] = None
    precision: Optional[str] = None
    achieved_rate: Optional[Quantity] = None
    achieved_bandwidth: Optional[Quantity] = None

    @property
    def point_intensity(self) -> Quantity:
        """Stated arithmetic intensity or the ratio inferred from achieved rates. Equation:
        ``intensity`` else ``achieved_rate / achieved_bandwidth``. Units: FLOPs/byte; inferred
        rates must describe the same workload interval.

        The stated intensity, or achieved rate over achieved bandwidth for a measured kernel.
        """
        if self.intensity is not None:
            return self.intensity
        return (self.achieved_rate / self.achieved_bandwidth).to("flop/byte")

    @property
    def bandwidth_utilization(self) -> float:
        """Achieved memory rate as a fraction of catalog peak. Equation: ``achieved_bandwidth /
        hardware.memory.bandwidth``. Units: dimensionless; the achieved rate is an input.
        """
        return (self.achieved_bandwidth / self.hardware.memory.bandwidth).to("").magnitude

    @property
    def peak(self) -> Quantity:
        """Peak compute rate at the selected arithmetic precision. Equation: catalog default or
        ``precision_flops[precision]``. Units: FLOP/s; achieved efficiency is not applied.
        """
        if self.precision is None:
            return self.hardware.compute.peak_flops
        return self.hardware.compute.precision_flops[self.precision]

    @property
    def ridge(self) -> Quantity:
        """Peak Roofline crossover intensity. Equation: ``peak / hardware.memory.bandwidth``.
        Units: FLOPs/byte; bandwidth is the catalog maximum.
        """
        return (self.peak / self.hardware.memory.bandwidth).to("flop/byte")

    @property
    def attainable(self) -> Quantity:
        """Roofline throughput upper bound at the point intensity. Equation: ``min(peak,
        hardware.memory.bandwidth * point_intensity)``. Units: FLOP/s; overheads and reduced
        utilization can lower achieved throughput.
        """
        return min(
            self.peak.to("flop/second"), (self.hardware.memory.bandwidth * self.point_intensity).to("flop/second")
        )

    @property
    def peak_fraction(self) -> float:
        """Roofline upper bound relative to peak compute rate. Equation: ``attainable / peak``.
        Units: dimensionless; this is not measured MFU.
        """
        return (self.attainable / self.peak).to("").magnitude


class AcceleratorPodScenario(BaseScenario):
    """A pod envelope of ``pod.chips`` copies of ``accelerator``: aggregate peak and memory.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: The pod envelope contains identical chips; capacities and peaks add
    arithmetically.

    Excludes: Aggregate local memory is not a unified address space, and aggregate peak does
    not imply achieved scaling.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TPUv4Pod4096
    >>> str(case.peak_flops)
    '1126400.0 TFLOPs/s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pod: object  # systems PodEnvelope
    accelerator: HardwareNode

    @property
    def peak_flops(self) -> Quantity:
        """Sum of per-chip arithmetic peaks in the pod envelope. Equation: ``pod.chips *
        accelerator.compute.peak_flops``. Units: FLOP/s; perfect parallel utilization is not
        implied.
        """
        return self.pod.chips * self.accelerator.compute.peak_flops

    @property
    def chip_memory_total(self) -> Quantity:
        """Sum of local chip memory capacities. Equation: ``pod.chips *
        accelerator.memory.capacity``. Units: bytes; this does not make the memory a single
        shared address space.
        """
        return self.pod.chips * self.accelerator.memory.capacity


class GrowthGapScenario(BaseScenario):
    """Compute doubling every ``compute_doubling_period`` against bandwidth growing
    ``bandwidth_growth`` a year.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: The supplied doubling period and annual bandwidth growth remain constant
    over the horizon.

    Excludes: Technology forecasts, saturation, and uncertainty are not estimated.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IllustrativeMemoryWall
    >>> round(case.compute_annual_factor, 4)
    1.5874
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    compute_doubling_period: Quantity
    bandwidth_growth: float  # fractional growth per year
    horizon: Quantity

    @property
    def compute_annual_factor(self) -> float:
        """Yearly compute-growth multiplier implied by a doubling interval. Equation: ``2 **
        (one_year / compute_doubling_period)``. Units: dimensionless; the doubling trend is
        assumed stationary.
        """
        return 2 ** (Q_(1, "year") / self.compute_doubling_period).to("").magnitude

    @property
    def bandwidth_annual_factor(self) -> float:
        """Yearly bandwidth multiplier from a fractional growth rate. Equation: ``1 +
        bandwidth_growth``. Units: dimensionless; growth is compounded annually.
        """
        return 1 + self.bandwidth_growth

    @property
    def gap(self) -> float:
        """Growth in the compute-to-bandwidth ratio over the horizon. Equation: annual compute
        factor divided by annual bandwidth factor, raised to horizon years. Units:
        dimensionless; this extrapolates the supplied trends.
        """
        years = self.horizon.to("year").magnitude
        return (self.compute_annual_factor / self.bandwidth_annual_factor) ** years


class WakeWordFleetScenario(BaseScenario):
    """Wake-word detection for a device fleet: stream everything to the cloud, or detect on
    device.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Devices have identical daily query activity; reference datacenters provide
    full accelerator-time capacity; peak load follows the supplied waking-window model.

    Excludes: Admission queues, spare capacity, geographic placement, and wake-word accuracy
    are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.VoiceAssistantBillionDevices
    >>> str(case.stream_rate)
    '32000.0 B/s'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    devices: float
    audio_sample_rate: Quantity
    audio_sample_bits: Quantity
    queries_per_day: float
    query_audio: Quantity  # audio uploaded per post-wake query
    accelerator_time_per_query: Quantity
    accelerators_per_datacenter: int
    waking_hours: Quantity  # queries cluster in these hours of the day
    peak_multiplier: float  # peak over waking-hours average
    cloud_detection_cost: Quantity  # per device per year
    local_power_low: Optional[Quantity] = None  # on-device detector power range
    local_power_high: Optional[Quantity] = None
    local_detection_cost: Optional[Quantity] = None  # per device per year

    @property
    def stream_rate(self) -> Quantity:
        """Uncompressed audio byte rate per continuously streaming device. Equation:
        ``audio_sample_rate * audio_sample_bits``. Units: bytes/s after bit conversion; framing
        and compression are excluded.
        """
        return (self.audio_sample_rate * self.audio_sample_bits).to("byte/second")

    @property
    def fleet_stream_rate(self) -> Quantity:
        """Aggregate always-streaming audio rate. Equation: ``devices * stream_rate``. Units:
        bytes/s; all devices stream concurrently.
        """
        return (self.devices * self.stream_rate).to("byte/second")

    @property
    def upload_reduction(self) -> float:
        """Always-streamed duration relative to retained query-audio duration per day. Equation:
        one day divided by ``queries_per_day * query_audio``. Units: dimensionless; query clips
        do not overlap or add metadata.
        """
        return (Q_(1, "day") / (self.queries_per_day * self.query_audio)).to("").magnitude

    @property
    def fleet_query_rate(self) -> Quantity:
        """Daily average audio upload rate after local wake-word filtering. Equation:
        ``fleet_stream_rate / upload_reduction``. Units: bytes/s; burst timing is represented
        separately.
        """
        return (self.fleet_stream_rate / self.upload_reduction).to("byte/second")

    @property
    def cloud_detection_cost_per_year(self) -> Quantity:
        """Annual cloud detection price across the device fleet. Equation: ``devices *
        cloud_detection_cost``. Units: USD/year; per-device cost is supplied.
        """
        return (self.devices * self.cloud_detection_cost).to("USD/year")

    @property
    def queries_per_fleet_day(self) -> float:
        """Fleet-wide daily query count. Equation: ``devices * queries_per_day``. Units: queries
        per stated day; identical device usage is assumed.
        """
        return self.devices * self.queries_per_day

    @property
    def accelerator_hours_per_day(self) -> float:
        """Accelerator-time demanded by the fleet's daily queries. Equation: daily query count
        times accelerator time per query. Units: scalar accelerator-hours per stated day;
        concurrent batching effects must be reflected in the input.
        """
        return (self.queries_per_fleet_day * self.accelerator_time_per_query).to("hour").magnitude

    @property
    def datacenter_hours_per_day(self) -> float:
        """Daily accelerator-time capacity of one reference datacenter. Equation:
        ``accelerators_per_datacenter * one_day``. Units: scalar accelerator-hours; full
        availability is assumed.
        """
        return (self.accelerators_per_datacenter * Q_(1, "day")).to("hour").magnitude

    @property
    def datacenters_average(self) -> float:
        """Reference datacenter equivalents required by daily average query work. Equation:
        ``accelerator_hours_per_day / datacenter_hours_per_day``. Units: dimensionless; this is
        continuous capacity, not a rounded site count.
        """
        return self.accelerator_hours_per_day / self.datacenter_hours_per_day

    @property
    def peak_ratio(self) -> float:
        """Peak-to-daily-average load multiplier under the stated waking window. Equation:
        ``peak_multiplier * one_day / waking_hours``. Units: dimensionless; the waking-window
        concentration is assumed.

        Peak over the all-day average: the waking-hours concentration times the peak multiplier.
        """
        return self.peak_multiplier * (Q_(1, "day") / self.waking_hours).to("").magnitude

    @property
    def datacenters_peak(self) -> float:
        """Reference datacenter equivalents required at the modeled peak. Equation:
        ``datacenters_average * peak_ratio``. Units: dimensionless; spare capacity and
        geographic placement are excluded.
        """
        return self.datacenters_average * self.peak_ratio


class PipelineSpeedupScenario(BaseScenario):
    """Serial pipeline stages; one stage is made ``speedup`` times faster (Amdahl's law).

    Notes
    -----
    Equation: Compose ``mlsysim.physics.performance.calc_amdahls_speedup``.

    Assumptions: The baseline stages are serial and only one stage changes duration;
    optional overlap shares are budget ratios rather than a scheduler simulation.

    Excludes: Parallel critical paths and dependencies beyond the named serial stages are
    absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.performance.calc_amdahls_speedup``. The methods below
    identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PhoneCameraPipeline
    >>> str(case.total)
    '200 ms'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    stages: Dict[str, Quantity]
    accelerated: str
    speedup: float = 1.0  # default (added 2026-10-20): the path as it is

    @property
    def total(self) -> Quantity:
        """Serial end-to-end duration of all named stages. Equation: sum of stage times. Units:
        milliseconds; stages do not overlap in this baseline.
        """
        return sum(self.stages.values(), Q_(0, "millisecond")).to("millisecond")

    @property
    def fraction(self) -> float:
        """Baseline latency fraction belonging to the accelerated stage. Equation:
        ``stages[accelerated] / total``. Units: dimensionless; the stage must be present and
        total duration positive.
        """
        return (self.stages[self.accelerated] / self.total).to("").magnitude

    @property
    def accelerated_stage(self) -> Quantity:
        """New duration of the selected stage. Equation: ``stages[accelerated] / speedup``. Units:
        milliseconds; speedup changes only this stage.
        """
        return (self.stages[self.accelerated] / self.speedup).to("millisecond")

    @property
    def accelerated_total(self) -> Quantity:
        """Serial request duration after the single-stage upgrade. Equation: replace the original
        stage time in ``total`` with ``accelerated_stage``. Units: milliseconds; all other times
        are fixed.
        """
        return (self.total - self.stages[self.accelerated] + self.accelerated_stage).to("millisecond")

    @property
    def overall_speedup(self) -> float:
        """End-to-end Amdahl speedup for the single-stage upgrade. Equation:
        ``mlsysim.physics.performance.calc_amdahls_speedup(fraction, speedup)``. Units:
        dimensionless; stages remain sequential.
        """
        from ...physics.performance import calc_amdahls_speedup

        return calc_amdahls_speedup(self.fraction, self.speedup)

    @property
    def limit(self) -> float:
        """Amdahl ceiling when the selected stage becomes instantaneous. Equation:
        ``calc_amdahls_speedup(fraction, infinity)``. Units: dimensionless; unchanged stages
        bound the total gain.
        """
        from ...physics.performance import calc_amdahls_speedup

        return calc_amdahls_speedup(self.fraction, float("inf"))

    # Stage groups and overlap (added 2026-10-20).
    def subtotal(self, *names) -> Quantity:
        """Serial duration of the selected named stages. Equation: sum the stage times for
        ``names``. Units: milliseconds; repeated names would count the stage repeatedly.
        """
        return sum((self.stages[n] for n in names), Q_(0, "millisecond")).to("millisecond")

    def share(self, *names) -> float:
        """Selected stages' share of baseline serial duration. Equation: ``subtotal(*names) / total``.
        Units: dimensionless; the names should describe the intended nonoverlapping
        subset.
        """
        return (self.subtotal(*names) / self.total).to("").magnitude

    def overlapped_share(self, stage: str, *feeders) -> float:
        """Fraction of feeder time that can cover the named stage, capped at one. Equation:
        ``min(1, stages[stage] / subtotal(*feeders))``. Units: dimensionless; this is a timing
        budget, not a scheduled overlap measurement.

        Busy share of ``stage`` when the ``feeders`` run concurrently on other hardware.

        With the stages pipelined, one request leaves per max(feeders, stage),
        so ``stage`` is busy stage / max(feeders, stage) of the time.
        """
        return min(1.0, (self.stages[stage] / self.subtotal(*feeders)).to("").magnitude)


class DominantShareScenario(BaseScenario):
    """Dominant-resource fairness by progressive filling of indivisible ``bundles`` on
    ``capacity``.

    Resources are accelerators, CPU cores, memory (GB), and network (Gb/s).
    Each tenant's dominant share is its largest share of any one resource;
    progressive filling grants the next bundle to the tenant with the lowest
    dominant share until no bundle fits.

    Notes
    -----
    Equation: Compose
    ``mlsysim.labs.experiments.v2_08_experiments.dominant_resource_fairness``.

    Assumptions: The stated fields are inputs to
    ``mlsysim.labs.experiments.v2_08_experiments.dominant_resource_fairness``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.labs.experiments.v2_08_experiments.dominant_resource_fairness``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.labs.experiments.v2_08_experiments.dominant_resource_fairness``.
    The methods below identify any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DrfTwoTenantExample
    >>> type(case.result).__name__
    'dict'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    capacity: Dict[str, float]
    bundles: Dict[str, Dict[str, float]]

    @property
    def result(self) -> Dict[str, Any]:
        """Evaluate this scenario through
        ``mlsysim.labs.experiments.v2_08_experiments.dominant_resource_fairness``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ...labs.experiments.v2_08_experiments import dominant_resource_fairness

        return dominant_resource_fairness(self.capacity, self.bundles)


class NodePackingScenario(BaseScenario):
    """Jobs of ``job_gpus`` GPUs packed on ``nodes`` nodes of ``gpus_per_node``.

    A job must sit in one node, so each node runs ``gpus_per_node //
    job_gpus`` jobs and strands the remainder. ``free_slots`` (optional)
    records a fragmented snapshot: free GPUs scattered so that no node is
    whole.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: Equal-sized indivisible jobs must fit wholly within identically sized
    nodes.

    Excludes: Cross-node placement, heterogeneous jobs, and fragmentation in other resource
    dimensions are absent.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SixGpuJobsOnDgxH100Nodes
    >>> round(case.total_gpus, 4)
    512
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    nodes: int
    gpus_per_node: int
    job_gpus: int
    free_slots: Optional[int] = None

    @property
    def total_gpus(self) -> int:
        """Accelerators physically available across the nodes. Equation: ``nodes * gpus_per_node``.
        Units: GPUs; nodes are identically populated.
        """
        return self.nodes * self.gpus_per_node

    @property
    def stranded_per_node(self) -> int:
        """Accelerators left after filling a node with indivisible jobs. Equation: node capacity
        minus whole-job count times GPUs per job. Units: GPUs; jobs cannot span nodes.
        """
        return self.gpus_per_node - (self.gpus_per_node // self.job_gpus) * self.job_gpus

    @property
    def usable_share(self) -> float:
        """Fraction of each node usable under whole-job packing. Equation: ``1 - stranded_per_node
        / gpus_per_node``. Units: dimensionless; identical job sizes are assumed.
        """
        return 1 - self.stranded_per_node / self.gpus_per_node

    @property
    def free_share(self) -> float:
        """Currently free slots as a fraction of fleet accelerator capacity. Equation: ``free_slots
        / total_gpus``. Units: dimensionless; availability alone does not imply a job-sized
        contiguous placement.
        """
        return self.free_slots / self.total_gpus


class PreemptionTaxScenario(BaseScenario):
    """Forced preemptions of an ``n_devices`` job whose checkpoint is ``checkpoint_bytes``.

    Notes
    -----
    Equation: result = PreemptionTaxModel.solve(...) with the declared n_devices,
    checkpoint_bytes, lost_work inputs.

    Assumptions: The inputs instantiate the assumptions documented by
    ``mlsysim.solvers.PreemptionTaxModel.solve``; accessing result evaluates that solver
    again.

    Excludes: Any extra mechanism beyond PreemptionTaxModel; this record supplies
    configuration and does not collect performance measurements.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.PreemptionTaxModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ForcedPreemption175BFineTune64
    >>> round(float(case.result.lost_work.to('min').magnitude), 3)
    15.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    n_devices: int
    checkpoint_bytes: Quantity
    lost_work: Quantity
    preemption_rate: Quantity  # forced preemptions per unit time
    recovery: Any  # RecoveryProfile (Systems.Reliability.Recovery)
    device_hour_rate: Quantity

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.PreemptionTaxModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.PreemptionTaxModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import PreemptionTaxModel

        return PreemptionTaxModel().solve(
            self.n_devices,
            self.checkpoint_bytes,
            self.lost_work,
            self.preemption_rate,
            self.recovery,
            self.device_hour_rate,
        )


class ElasticResizeScenario(BaseScenario):
    """Three ways to run one job over ``window``: start small, wait for the full size, or step up.

    Start small runs ``small_gpus`` for the whole window. Wait queues
    ``queue_wait`` for ``full_gpus``. Step up starts on ``small_gpus`` and,
    when the full allocation would free, pauses ``rescale_overhead`` and grows
    to ``step_up_gpus``, leaving the rest to queued jobs. Throughputs are
    epochs per hour at each size. GPU-hours count the step-up size through
    its resize pause.

    Notes
    -----
    Equation: epochs = throughput * productive hours for each phase; efficiency = epochs /
    reserved device-hours.

    Assumptions: Throughputs are fixed epochs per hour and enough window remains for each
    chosen phase. The larger reservation includes the resize pause.

    Excludes: Convergence differences, variable scaling efficiency, and a scheduler that
    predicts future queue availability.

    References
    ----------
    Derivation: epochs = throughput * productive hours for
    each phase; efficiency = epochs / reserved device-hours. The individual properties
    state units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ElasticResizeDecision
    >>> round(float(case.window_h), 3)
    24.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: Quantity
    queue_wait: Quantity
    rescale_overhead: Quantity
    small_gpus: int
    step_up_gpus: int
    full_gpus: int
    small_throughput: float
    step_up_throughput: float
    full_throughput: float

    def _h(self, q) -> float:
        return q.to("hour").magnitude

    @property
    def window_h(self) -> float:
        """Scheduling decision window in scalar hours.

        Equation: ``return _h(window)``."""
        return self._h(self.window)

    @property
    def wait_h(self) -> float:
        """Queue delay before the larger allocation becomes available, in scalar hours.

        Equation: ``return _h(queue_wait)``."""
        return self._h(self.queue_wait)

    @property
    def rescale_h(self) -> float:
        """Resize pause expressed in scalar hours.

        Equation: ``return rescale_overhead.to('minute').magnitude / 60``."""
        return self.rescale_overhead.to("minute").magnitude / 60

    @property
    def start_small_epochs(self) -> float:
        """Epochs completed by the small allocation at its fixed throughput throughout the
        window.

        Equation: ``return window_h * small_throughput``."""
        return self.window_h * self.small_throughput

    @property
    def wait_run_h(self) -> float:
        """Hours remaining after waiting for the full allocation.

        Equation: ``return window_h - wait_h``."""
        return self.window_h - self.wait_h

    @property
    def wait_epochs(self) -> float:
        """Epochs completed at full throughput after the queue delay.

        Equation: ``return wait_run_h * full_throughput``."""
        return self.wait_run_h * self.full_throughput

    @property
    def step_up_phase1_epochs(self) -> float:
        """Epochs completed on the small allocation before resizing begins.

        Equation: ``return wait_h * small_throughput``."""
        return self.wait_h * self.small_throughput

    @property
    def step_up_phase2_h(self) -> float:
        """Productive hours remaining after both queue delay and resize pause.

        Equation: ``return window_h - wait_h - rescale_h``."""
        return self.window_h - self.wait_h - self.rescale_h

    @property
    def step_up_phase2_epochs(self) -> float:
        """Epochs completed at the larger allocation's throughput after resizing.

        Equation: ``return step_up_phase2_h * step_up_throughput``."""
        return self.step_up_phase2_h * self.step_up_throughput

    @property
    def step_up_epochs(self) -> float:
        """Total completed epochs from the two productive phases.

        Equation: ``return step_up_phase1_epochs + step_up_phase2_epochs``."""
        return self.step_up_phase1_epochs + self.step_up_phase2_epochs

    @property
    def step_up_trails_wait(self) -> float:
        """Fractional epoch shortfall of step-up relative to waiting; negative means step-up
        wins.

        Equation: ``return (wait_epochs - step_up_epochs) / wait_epochs``."""
        return (self.wait_epochs - self.step_up_epochs) / self.wait_epochs

    @property
    def step_up_over_small(self) -> float:
        """Completed-epoch ratio of step-up to keeping the small allocation.

        Equation: ``return step_up_epochs / start_small_epochs``."""
        return self.step_up_epochs / self.start_small_epochs

    @property
    def rescale_break_even_h(self) -> float:
        """Resize pause in hours at which step-up completes as many epochs as staying small.

        Equation: ``return window_h - wait_h - (start_small_epochs -
        step_up_phase1_epochs) / step_up_throughput``."""
        return (
            self.window_h
            - self.wait_h
            - (self.start_small_epochs - self.step_up_phase1_epochs) / self.step_up_throughput
        )

    @property
    def step_up_efficiency(self) -> float:
        """Throughput scaling relative to ideal linear growth from the small allocation, as a
        ratio.

        Equation: ``return step_up_throughput / (step_up_gpus / small_gpus *
        small_throughput)``."""
        return self.step_up_throughput / (self.step_up_gpus / self.small_gpus * self.small_throughput)

    @property
    def full_efficiency(self) -> float:
        """Full-allocation throughput divided by linear scaling of the small allocation's
        throughput.

        Equation: ``return full_throughput / (full_gpus / small_gpus *
        small_throughput)``."""
        return self.full_throughput / (self.full_gpus / self.small_gpus * self.small_throughput)

    @property
    def gpu_hours(self) -> Dict[str, float]:
        """Device-hours by policy; step-up reserves the larger allocation during its resize
        pause.

        Equation: ``return {'start_small': small_gpus * window_h, 'wait': full_gpus *
        wait_run_h, 'step_up': small_gpus * wait_h + step_up_gpus * wait_run_h}``."""
        return {
            "start_small": self.small_gpus * self.window_h,
            "wait": self.full_gpus * self.wait_run_h,
            "step_up": self.small_gpus * self.wait_h + self.step_up_gpus * self.wait_run_h,
        }

    @property
    def epochs_per_gpu_hour(self) -> Dict[str, float]:
        """Completed epochs per reserved device-hour for each policy, including idle
        reservations.

        Equation: ``epochs = {'start_small': start_small_epochs, 'wait': wait_epochs,
        'step_up': step_up_epochs}; return {k: epochs[k] / h for k, h in
        gpu_hours.items()}``."""
        epochs = {"start_small": self.start_small_epochs, "wait": self.wait_epochs, "step_up": self.step_up_epochs}
        return {k: epochs[k] / h for k, h in self.gpu_hours.items()}


class ServingConsolidationScenario(BaseScenario):
    """Small and large serving models on whole devices, exclusive against shared in pairs.

    Each small model occupies ``small_footprint`` (weights, KV cache, and
    activations) of a ``device_memory`` budget; ``tenants`` small models share
    one device when load tests allow. Each large model takes
    ``gpus_per_large`` devices for tensor parallelism either way.

    Notes
    -----
    Equation: shared devices = floor(small models / tenants) + large-model devices;
    savings = exclusive minus shared devices.

    Assumptions: Small-model count is divisible by tenants for complete accounting; tenant
    throughput and memory isolation are supplied assumptions.

    Excludes: Partial tenant groups, interference, latency-tail guarantees, and workspace
    in the large-model weight-only floor.

    References
    ----------
    Derivation: shared devices = floor(small models /
    tenants) + large-model devices; savings = exclusive minus shared devices. The
    individual properties state units and intermediate accounting; this class introduces
    no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ServingConsolidationA100
    >>> round(float(case.exclusive_unused), 3)
    0.675"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    device_memory: Quantity  # branded decimal budget per device
    partition_memory: Optional[Quantity] = None  # a hardware partition (MIG profile) for comparison
    small_model: Workload
    small_footprint: Quantity
    tenants: int
    n_small: int
    large_model: Workload
    n_large: int
    gpus_per_large: int
    bytes_per_param: Quantity
    device_hour_rate: Quantity

    def _ratio(self, a, b) -> float:
        return (a / b).to("").magnitude

    @property
    def exclusive_unused(self) -> float:
        """Unused fraction of a device reserved for one small model.

        Equation: ``return _ratio(device_memory - small_footprint, device_memory)``."""
        return self._ratio(self.device_memory - self.small_footprint, self.device_memory)

    @property
    def shared_footprint(self) -> Quantity:
        """Combined memory quantity for ``tenants`` copies of the small model footprint.

        Equation: ``return tenants * small_footprint``."""
        return self.tenants * self.small_footprint

    @property
    def shared_unused(self) -> float:
        """Unused fraction after packing the declared tenants; negative indicates a memory
        overrun.

        Equation: ``return _ratio(device_memory - shared_footprint, device_memory)``."""
        return self._ratio(self.device_memory - self.shared_footprint, self.device_memory)

    @property
    def partition_unused(self) -> float:
        """Unused fraction of a hardware partition containing one small model; requires
        ``partition_memory``.

        Equation: ``return _ratio(partition_memory - small_footprint, partition_memory)``."""
        return self._ratio(self.partition_memory - self.small_footprint, self.partition_memory)

    @property
    def large_weights(self) -> Quantity:
        """Large-model weight memory, parameter count times stored bytes per parameter.

        Equation: ``return large_model.parameters * bytes_per_param``."""
        return self.large_model.parameters * self.bytes_per_param

    @property
    def min_devices_large(self) -> int:
        """Whole-device weight-residency floor, rounding weight bytes divided by device
        capacity upward.

        Equation: ``return math.ceil(_ratio(large_weights, device_memory))``."""
        import math

        return math.ceil(self._ratio(self.large_weights, self.device_memory))

    @property
    def large_gpus(self) -> int:
        """Devices reserved for all large models at their declared tensor-parallel degree.

        Equation: ``return n_large * gpus_per_large``."""
        return self.n_large * self.gpus_per_large

    @property
    def exclusive_gpus(self) -> int:
        """Total device count when every small model owns a device and large models retain
        their allocation.

        Equation: ``return n_small + large_gpus``."""
        return self.n_small + self.large_gpus

    @property
    def shared_small_gpus(self) -> int:
        """Device count for small models grouped into complete tenant sets; partial sets are
        truncated.

        Equation: ``return n_small // tenants``."""
        return self.n_small // self.tenants

    @property
    def shared_gpus(self) -> int:
        """Total device count with shared small models and unchanged large-model allocation.

        Equation: ``return shared_small_gpus + large_gpus``."""
        return self.shared_small_gpus + self.large_gpus

    @property
    def saved_gpus(self) -> int:
        """Devices released by sharing, relative to exclusive allocation.

        Equation: ``return exclusive_gpus - shared_gpus``."""
        return self.exclusive_gpus - self.shared_gpus

    @property
    def saved_share(self) -> float:
        """Fraction of the exclusive device allocation released by sharing.

        Equation: ``return saved_gpus / exclusive_gpus``."""
        return self.saved_gpus / self.exclusive_gpus

    @property
    def exclusive_utilization(self) -> float:
        """Memory occupancy fraction for one small model, not a compute-utilization estimate.

        Equation: ``return _ratio(small_footprint, device_memory)``."""
        return self._ratio(self.small_footprint, self.device_memory)

    @property
    def shared_utilization(self) -> float:
        """Memory occupancy fraction for the declared tenant group, not a throughput
        measurement.

        Equation: ``return _ratio(shared_footprint, device_memory)``."""
        return self._ratio(self.shared_footprint, self.device_memory)

    @property
    def annual_value(self) -> float:
        """Annual USD value of released devices at the scenario's fixed hourly rate.

        Equation: ``return saved_gpus * _usd_per_hour(device_hour_rate) *
        HOURS_PER_YEAR``."""
        from ...core.units import HOURS_PER_YEAR

        return self.saved_gpus * _usd_per_hour(self.device_hour_rate) * HOURS_PER_YEAR


class DiurnalLendingScenario(BaseScenario):
    """A fleet split between a serving reservation and a training base, static against lent.

    The reservation is the forecast serving peak plus ``surge_buffer``.
    Lending gives training whatever the reservation holds above average
    demand plus the buffer; the GPUs left after the reservation and the
    training base are maintenance headroom.

    Notes
    -----
    Equation: lent devices = serving reservation - average demand - surge buffer; dynamic
    active = static active + lent.

    Assumptions: Demand and reservation counts describe one common fleet and the average
    lent capacity can be reclaimed.

    Excludes: A time-series demand simulation, reclaim delay, and preemption losses.

    References
    ----------
    Derivation: lent devices = serving reservation - average
    demand - surge buffer; dynamic active = static active + lent. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DiurnalServingLending1K
    >>> round(float(case.peak_demand), 3)
    283.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fleet_gpus: int
    serving_reservation: int
    serving_average: int
    surge_buffer: int
    training_base: int
    device_hour_rate: Quantity

    @property
    def peak_demand(self) -> int:
        """Serving device demand before adding the reservation's surge buffer.

        Equation: ``return serving_reservation - surge_buffer``."""
        return self.serving_reservation - self.surge_buffer

    @property
    def serving_reclaimable(self) -> int:
        """Reserved devices above average serving demand, before preserving the buffer.

        Equation: ``return serving_reservation - serving_average``."""
        return self.serving_reservation - self.serving_average

    @property
    def lent(self) -> int:
        """Average devices lent to training after keeping the surge buffer available.

        Equation: ``return serving_reclaimable - surge_buffer``."""
        return self.serving_reclaimable - self.surge_buffer

    @property
    def headroom(self) -> int:
        """Devices outside both the serving reservation and the training base allocation.

        Equation: ``return fleet_gpus - serving_reservation - training_base``."""
        return self.fleet_gpus - self.serving_reservation - self.training_base

    @property
    def static_active(self) -> int:
        """Active device-equivalents under the fixed allocation: average serving plus base
        training.

        Equation: ``return serving_average + training_base``."""
        return self.serving_average + self.training_base

    @property
    def dynamic_active(self) -> int:
        """Active device-equivalents after adding the lent capacity to training.

        Equation: ``return static_active + lent``."""
        return self.static_active + self.lent

    @property
    def static_utilization(self) -> float:
        """Active fraction of the fleet with no lending.

        Equation: ``return static_active / fleet_gpus``."""
        return self.static_active / self.fleet_gpus

    @property
    def dynamic_utilization(self) -> float:
        """Active fraction of the fleet under the assumed lending policy.

        Equation: ``return dynamic_active / fleet_gpus``."""
        return self.dynamic_active / self.fleet_gpus

    @property
    def annual_value(self) -> float:
        """Annual USD value of average lent device capacity at the fixed hourly price.

        Equation: ``return lent * _usd_per_hour(device_hour_rate) * HOURS_PER_YEAR``."""
        from ...core.units import HOURS_PER_YEAR

        return self.lent * _usd_per_hour(self.device_hour_rate) * HOURS_PER_YEAR


class QuotaLendingScenario(BaseScenario):
    """A team guaranteed ``guarantee`` GPUs that uses ``used_share`` of them; the rest can be lent.

    Lent GPUs stay reclaimable: borrowers run them as lower-priority,
    preemptible work and return them when the owner's demand comes back.

    Notes
    -----
    Equation: used devices = round(guarantee * used share); lendable = guarantee - used.

    Assumptions: Usage share describes the guaranteed pool and rounds with Python nearest-
    integer semantics.

    Excludes: Borrower demand, fairness policy, and reclaim latency.

    References
    ----------
    Derivation: used devices = round(guarantee * used
    share); lendable = guarantee - used. The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.TeamQuotaLending500
    >>> round(float(case.used), 3)
    300.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    guarantee: int
    used_share: float
    borrower_guarantee: Optional[int] = None

    @property
    def used(self) -> int:
        """Used device count, rounding guarantee times the declared usage share to the
        nearest integer.

        Equation: ``return round(guarantee * used_share)``."""
        return round(self.guarantee * self.used_share)

    @property
    def lendable(self) -> int:
        """Guaranteed devices remaining after rounded usage; borrower demand is not modeled.

        Equation: ``return guarantee - used``."""
        return self.guarantee - self.used


class SelfContainedReplicaScenario(BaseScenario):
    """The smallest model replica of ``workload`` whose training state fits without sharding.

    Tensor parallelism spans one node of ``fleet``; the pipeline depth is the
    smallest power of two at which each accelerator's share of the
    ``bytes_per_param`` training state fits its memory. An elastic job that
    keeps this layout fixed grows or shrinks one whole replica at a time.

    Notes
    -----
    Equation: pipeline depth is the first power of two satisfying state / (tensor degree *
    pipeline depth) <= device memory.

    Assumptions: Training-state bytes per parameter are supplied; every shard receives an
    equal share and tensor parallelism spans one node.

    Excludes: Activations, runtime workspace, communication buffers, and layout-
    performance ranking.

    References
    ----------
    Derivation: pipeline depth is the first power of two
    satisfying state / (tensor degree * pipeline depth) <= device memory. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3SelfContainedReplicaDgxH100
    >>> round(float(case.tensor_parallel), 3)
    8.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    fleet: Fleet
    bytes_per_param: Quantity

    @property
    def tensor_parallel(self) -> int:
        """Tensor-parallel degree equal to the accelerator count in one node.

        Equation: ``return fleet.node.accelerators_per_node``."""
        return self.fleet.node.accelerators_per_node

    @property
    def pipeline_depth(self) -> int:
        """Smallest power-of-two pipeline depth whose equal training-state shards fit device
        memory.

        Equation: Follow the pipeline_depth calculation below; branch conditions and
        rounding are part of its definition."""
        hbm = self.fleet.node.accelerator.memory.capacity
        state = self.workload.parameters.magnitude * self.bytes_per_param
        pp = 1
        while (state / (self.tensor_parallel * pp)).to("GiB") > hbm:
            pp *= 2
        return pp

    @property
    def replica_gpus(self) -> int:
        """Devices per complete replica: tensor-parallel degree times pipeline depth.

        Equation: ``return tensor_parallel * pipeline_depth``."""
        return self.tensor_parallel * self.pipeline_depth


class ScreeningRequirementsScenario(BaseScenario):
    """A screener's release floors and device budget, with measured slices checked against the floor.

    ``device`` supplies the power envelope; ``slice_sensitivity`` maps a test slice
    (all photos, one patient subgroup) to its measured sensitivity; a release passes
    only if every slice meets the floor. ``field_operating_point`` is an optional
    (specificity, sensitivity) pair for one threshold on field data.

    Notes
    -----
    Equation: a slice passes when its supplied sensitivity is at least sensitivity_floor.

    Assumptions: Slice sensitivities are externally supplied evaluation values; power
    limits come from the device record.

    Excludes: A clinical validation study, uncertainty intervals, and enforcement of
    specificity or memory through passes().

    References
    ----------
    Derivation: a slice passes when its supplied sensitivity
    is at least sensitivity_floor. The individual properties state units and intermediate
    accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RetinopathyScreenerRequirements
    >>> round(float(case.power_low.to('W').magnitude), 3)
    7.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    sensitivity_floor: float
    specificity_floor: float
    memory_budget: Quantity
    device: HardwareNode
    slice_sensitivity: Dict[str, float] = {}
    field_operating_point: Optional[Tuple[float, float]] = None

    @property
    def power_low(self) -> Quantity:
        """Lower device power-envelope quantity from the hardware specification.

        Equation: ``return device.tdp_min``."""
        return self.device.tdp_min

    @property
    def power_high(self) -> Quantity:
        """Upper device power-envelope quantity from the hardware specification.

        Equation: ``return device.tdp_max``."""
        return self.device.tdp_max

    def passes(self, slice_name: str) -> bool:
        """Whether the named slice's supplied sensitivity meets the release floor; returns a
        Boolean.

        Equation: ``return slice_sensitivity[slice_name] >= sensitivity_floor``."""
        return self.slice_sensitivity[slice_name] >= self.sensitivity_floor

    @property
    def failing_slices(self) -> List[str]:
        """Slice names below the supplied sensitivity floor; no new evaluation data are
        collected.

        Equation: ``return [k for k in slice_sensitivity if not passes(k)]``."""
        return [k for k in self.slice_sensitivity if not self.passes(k)]


class LateConstraintScenario(BaseScenario):
    """A model built without its deployment limit, the day the limit surfaces, and the repair it forces.

    ``validation_sensitivity`` maps a day to the sensitivity reached that day.
    ``repair_tasks`` lists the redone artifacts in order; each waits for the one
    before, so the repair takes their sum when staff and compute are free.

    Notes
    -----
    Equation: repair duration is the sum of ordered repair tasks; memory overrun = model
    bytes / budget bytes.

    Assumptions: Tasks execute sequentially with immediately available staff and compute.
    Dates share one day origin.

    Excludes: Parallel repairs, resource queues, and a prediction of when engineers
    discover a requirement.

    References
    ----------
    Derivation: repair duration is the sum of ordered repair
    tasks; memory overrun = model bytes / budget bytes. The individual properties state
    units and intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RetinopathyScreenerLateMemoryLimit
    >>> round(float(case.overrun_ratio), 3)
    8.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model_memory: Quantity
    memory_budget: Quantity
    device_memory_low: Quantity
    device_memory_high: Quantity
    request_day: int
    validation_sensitivity: Dict[int, float]
    handoff_day: int
    discovery_day: int
    check_day: int
    discard_day: int
    repair_tasks: Dict[str, Quantity]

    @property
    def overrun_ratio(self) -> float:
        """Model memory divided by deployment budget, a dimensionless residency ratio.

        Equation: ``return (model_memory / memory_budget).to('').magnitude``."""
        return (self.model_memory / self.memory_budget).to("").magnitude

    @property
    def work_discarded(self) -> Quantity:
        """Elapsed days from the request to discarding the incompatible work.

        Equation: ``return Q_(discard_day - request_day, 'day')``."""
        return Q_(self.discard_day - self.request_day, "day")

    @property
    def repair_elapsed(self) -> Quantity:
        """Sequential repair duration in days, adding every listed task without overlap.

        Equation: Follow the repair_elapsed calculation below; branch conditions and
        rounding are part of its definition."""
        total = Q_(0, "day")
        for duration in self.repair_tasks.values():
            total = total + duration
        return total.to("day")


class IterationBudgetScenario(BaseScenario):
    """Two teams in one development window: a slow cycle against a fast one, with assumed gains per run.

    Accuracies are in percent and gains in points per run. The slow team learns
    from every run; the fast team can run far more than it can learn from, so
    ``fast_effective_runs`` caps the runs that count. Both stop at ``ceiling``.

    Notes
    -----
    Equation: final accuracy = min(start + useful runs * gain per run, ceiling).

    Assumptions: Accuracy values are percentages and gains are percentage points. Run
    gains and effective fast runs are illustrative inputs, not learned response curves.

    Excludes: A statistical learning-curve model and a causal claim that faster
    experiments improve accuracy by this amount.

    References
    ----------
    Derivation: final accuracy = min(start + useful runs *
    gain per run, ceiling). The individual properties state units and intermediate
    accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SlowVersusFastIteration
    >>> round(float(case.slow_runs), 3)
    26.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: Quantity
    slow_cycle: Quantity
    slow_start: float
    slow_gain: float
    fast_cycle: Quantity
    fast_start: float
    fast_gain: float
    fast_effective_runs: int
    ceiling: float

    @property
    def slow_runs(self) -> float:
        """Run-equivalents available in the slow team's development window; fractional runs
        are retained.

        Equation: ``return (window / slow_cycle).to('').magnitude``."""
        return (self.window / self.slow_cycle).to("").magnitude

    @property
    def slow_final(self) -> float:
        """Illustrative final accuracy in percent: starting accuracy plus per-run gain,
        capped at the ceiling.

        Equation: ``return min(slow_start + slow_runs * slow_gain, ceiling)``."""
        return min(self.slow_start + self.slow_runs * self.slow_gain, self.ceiling)

    @property
    def fast_possible_runs(self) -> float:
        """Run-equivalents permitted by the fast cycle, before limiting useful learning
        opportunities.

        Equation: ``return (window / fast_cycle).to('').magnitude``."""
        return (self.window / self.fast_cycle).to("").magnitude

    @property
    def fast_uncapped(self) -> float:
        """Illustrative accuracy in percent from the stipulated number of useful fast runs.

        Equation: ``return fast_start + fast_effective_runs * fast_gain``."""
        return self.fast_start + self.fast_effective_runs * self.fast_gain

    @property
    def fast_final(self) -> float:
        """Fast team's illustrative accuracy in percent after applying the shared ceiling.

        Equation: ``return min(fast_uncapped, ceiling)``."""
        return min(self.fast_uncapped, self.ceiling)

    @property
    def ideas_ratio(self) -> float:
        """Useful fast experiments per slow run-equivalent, a dimensionless opportunity
        ratio.

        Equation: ``return fast_effective_runs / slow_runs``."""
        return self.fast_effective_runs / self.slow_runs

    @property
    def final_gap(self) -> float:
        """Absolute difference in final illustrative accuracy, in percentage points.

        Equation: ``return abs(fast_final - slow_final)``."""
        return abs(self.fast_final - self.slow_final)


class DailyUploadScenario(BaseScenario):
    """A site's daily captures sent over its uplink inside a working window, against compact summaries.

    Notes
    -----
    Equation: raw bytes = cases * files per case * file bytes; transfer time = bytes /
    bandwidth.

    Assumptions: The uplink sustains the supplied bandwidth and summaries contain one
    payload per case.

    Excludes: Protocol overhead, retransmissions, compression work, and quality
    differences between raw files and summaries.

    References
    ----------
    Derivation: raw bytes = cases * files per case * file
    bytes; transfer time = bytes / bandwidth. The individual properties state units and
    intermediate accounting; this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RuralClinicRetinalUpload
    >>> round(float(case.daily_volume.to('MB').magnitude), 3)
    2000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    cases_per_day: int
    files_per_case: int
    file_size: Quantity
    uplink: Quantity
    window: Quantity
    summary_size: Quantity  # one summary per case

    @property
    def daily_volume(self) -> Quantity:
        """Raw daily data quantity: cases times files per case times bytes per file.

        Equation: ``return cases_per_day * files_per_case * file_size``."""
        return self.cases_per_day * self.files_per_case * self.file_size

    @property
    def upload_time(self) -> Quantity:
        """Ideal raw-data transfer time from ``physics.quantities.transfer_time``; no
        protocol overhead.

        Equation: ``return transfer_time(daily_volume, uplink)``."""
        from ...physics.quantities import transfer_time

        return transfer_time(self.daily_volume, self.uplink)

    @property
    def window_share(self) -> float:
        """Fraction of the working window consumed by the ideal daily upload.

        Equation: ``return (upload_time / window).to('').magnitude``."""
        return (self.upload_time / self.window).to("").magnitude

    @property
    def file_upload_time(self) -> Quantity:
        """Ideal transfer time of one file through ``physics.quantities.transfer_time``.

        Equation: ``return transfer_time(file_size, uplink)``."""
        from ...physics.quantities import transfer_time

        return transfer_time(self.file_size, self.uplink)

    @property
    def summary_volume(self) -> Quantity:
        """Daily summary data quantity, with one summary per case.

        Equation: ``return cases_per_day * summary_size``."""
        return self.cases_per_day * self.summary_size

    @property
    def reduction(self) -> float:
        """Raw bytes divided by summary bytes, a dimensionless data-volume reduction factor.

        Equation: ``return (daily_volume / summary_volume).to('').magnitude``."""
        return (self.daily_volume / self.summary_volume).to("").magnitude


class HostDispatchScenario(BaseScenario):
    """The host's work to issue one eager step against ``device_time`` of accelerator work.

    The step has ``blocks`` blocks of ``ops_per_block`` Python-level
    operations, each paying ``dispatch``, and ``launches_per_block`` kernels
    per block (default: one per operation), each paying ``launch``. The model
    is additive with no host-device overlap: the overhead ratio is host time
    over device time, and above 1 the host takes more than half of the step.

    Notes
    -----
    Equation: Host time is ``operation_count*dispatch`` + ``launch_count*launch``; the additive step includes device_time.

    Assumptions: Constant overhead per event and serial host/device timing.

    Excludes: Asynchronous overlap, graph capture, and measured interpreter or driver behavior.

    References
    ----------
    Derivation: Serial event accounting adds dispatch cost once per Python operation and
    launch cost once per kernel launch. Adding device work then gives the assumed serial
    step; removing the entire host term defines a ceiling rather than a predicted
    optimization gain.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Gpt2XlEagerDecodeDispatch
    >>> s.host_time == s.dispatch_time+s.launch_time
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    ops_per_block: int
    blocks: int = 1
    launches_per_block: Optional[int] = None
    dispatch: Quantity
    launch: Quantity
    device_time: Quantity

    @property
    def operations(self) -> int:
        """Count of Python-level operations across all blocks.

        Equation: ``blocks * ops_per_block``."""
        return self.blocks * self.ops_per_block

    @property
    def launches(self) -> int:
        """Kernel launch count across blocks; defaults to one launch per operation.

        Equation: ``per = ops_per_block if launches_per_block is None else
        launches_per_block; blocks * per``."""
        per = self.ops_per_block if self.launches_per_block is None else self.launches_per_block
        return self.blocks * per

    @property
    def per_op_cost(self) -> Quantity:
        """Dispatch plus launch, for an operation that launches one kernel.

        Equation: ``(dispatch + launch).to('microsecond')``."""
        return (self.dispatch + self.launch).to("microsecond")

    @property
    def dispatch_time(self) -> Quantity:
        """Total Python-side dispatch time in microseconds.

        Equation: ``(operations * dispatch).to('microsecond')``."""
        return (self.operations * self.dispatch).to("microsecond")

    @property
    def launch_time(self) -> Quantity:
        """Total kernel-launch overhead in microseconds.

        Equation: ``(launches * launch).to('microsecond')``."""
        return (self.launches * self.launch).to("microsecond")

    @property
    def host_time(self) -> Quantity:
        """Serial dispatch-plus-launch time in microseconds.

        Equation: ``(dispatch_time + launch_time).to('microsecond')``."""
        return (self.dispatch_time + self.launch_time).to("microsecond")

    @property
    def overhead_ratio(self) -> float:
        """Host time over device time (the dispatch overhead ratio).

        Equation: ``(host_time / device_time).to('').magnitude``."""
        return (self.host_time / self.device_time).to("").magnitude

    @property
    def launch_ratio(self) -> float:
        """Kernel launches alone over device time.

        Equation: ``(launch_time / device_time).to('').magnitude``."""
        return (self.launch_time / self.device_time).to("").magnitude

    @property
    def host_share(self) -> float:
        """Host time as a share of the serial step, host plus device.

        Equation: ``(host_time / (host_time + device_time)).to('').magnitude``."""
        return (self.host_time / (self.host_time + self.device_time)).to("").magnitude

    @property
    def speedup_limit(self) -> float:
        """Serial step over device time: the most removing all host work could gain.

        Equation: ``((host_time + device_time) / device_time).to('').magnitude``."""
        return ((self.host_time + self.device_time) / self.device_time).to("").magnitude


class KernelFusionScenario(BaseScenario):
    """A chain of ``chain_length`` element-wise operations, run as one kernel each or fused into one.

    Unfused, every operation reads its input from device memory and writes its
    output back (two trips each) and pays ``dispatch`` plus ``launch``; fused,
    the chain reads once, keeps intermediates on chip, writes once, and pays
    one launch. For N elements of b bytes the traffic is 2kNb against 2Nb.

    Notes
    -----
    Equation: Unfused traffic makes ``2*k`` whole-array trips; fused traffic makes two while launch count falls from k to one.

    Assumptions: All intermediates fit on chip and the complete chain can be fused.

    Excludes: Register pressure, extra arithmetic, occupancy changes, and end-to-end speedup prediction.

    References
    ----------
    Derivation: Separate kernels write each intermediate to memory and read it for the
    next operation, giving two whole-array transfers per operation. A fused chain
    retains intermediates on chip and transfers only its external input/output, while
    paying one dispatch/launch pair.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.ElementwisePairFusion
    >>> s.traffic_ratio == s.chain_length
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    chain_length: int
    dispatch: Quantity
    launch: Quantity

    @property
    def unfused_trips(self) -> int:
        """Number of whole-array read/write trips when each operation is a separate kernel.

        Equation: ``2 * chain_length``."""
        return 2 * self.chain_length

    @property
    def fused_trips(self) -> int:
        """Whole-array read/write trips for one fused kernel retaining intermediates on chip.

        Equation: ``2``."""
        return 2

    @property
    def unfused_launches(self) -> int:
        """Kernel launch count before fusion, one per elementwise operation.

        Equation: ``chain_length``."""
        return self.chain_length

    @property
    def fused_launches(self) -> int:
        """Kernel launch count after fusion into one kernel.

        Equation: ``1``."""
        return 1

    @property
    def traffic_ratio(self) -> float:
        """Dimensionless unfused-to-fused whole-array traffic ratio.

        Equation: ``unfused_trips / fused_trips``."""
        return self.unfused_trips / self.fused_trips

    @property
    def unfused_overhead(self) -> Quantity:
        """Total dispatch-plus-launch overhead in microseconds before fusion.

        Equation: ``(unfused_launches * (dispatch + launch)).to('microsecond')``."""
        return (self.unfused_launches * (self.dispatch + self.launch)).to("microsecond")

    @property
    def fused_overhead(self) -> Quantity:
        """Dispatch-plus-launch overhead in microseconds for the fused kernel.

        Equation: ``(fused_launches * (dispatch + launch)).to('microsecond')``."""
        return (self.fused_launches * (self.dispatch + self.launch)).to("microsecond")

    @property
    def overhead_speedup(self) -> float:
        """Dimensionless launch-overhead ratio; it is not an end-to-end execution speedup.

        Equation: ``(unfused_overhead / fused_overhead).to('').magnitude``."""
        return (self.unfused_overhead / self.fused_overhead).to("").magnitude


class CompileBreakevenScenario(BaseScenario):
    """Eager against compiled execution: the work that repays ``compile_time``.

    Compiling pays when the time saved per item, summed over items, exceeds
    the compile time. An experiment of ``experiment_items`` items that changes
    the code ``recompiles`` times pays a compile for every change.

    Notes
    -----
    Equation: Compile cost divided by per-item time savings gives the breakeven item count.

    Assumptions: Steady item rates and a positive compiled speed advantage; recompilation is
    charged once per declared code change.

    Excludes: Warmup, cache effects, and rate changes during the experiment.

    References
    ----------
    Derivation: Compile cost divided by per-item time savings gives the breakeven item
    count. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ResNet50CompileBreakeven
    >>> round(float(scenario.eager_item_time.magnitude), 4)
    0.6897"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    eager_rate: Quantity  # items per second
    compiled_rate: Quantity  # items per second
    compile_time: Quantity
    experiment_items: Optional[int] = None
    recompiles: Optional[int] = None

    @property
    def eager_item_time(self) -> Quantity:
        """Time per eager item in milliseconds, the inverse of the eager item rate.

        Equation: ``(1 / eager_rate).to('ms')``."""
        return (1 / self.eager_rate).to("ms")

    @property
    def compiled_item_time(self) -> Quantity:
        """Time per compiled item in milliseconds, the inverse of the compiled rate.

        Equation: ``(1 / compiled_rate).to('ms')``."""
        return (1 / self.compiled_rate).to("ms")

    @property
    def saving_per_item(self) -> Quantity:
        """Milliseconds saved per item by compilation; a nonpositive value cannot repay
        compilation.

        Equation: ``(eager_item_time - compiled_item_time).to('ms')``."""
        return (self.eager_item_time - self.compiled_item_time).to("ms")

    @property
    def breakeven_items(self) -> float:
        """Item count at which accumulated per-item savings equal the compile time; requires
        positive savings.

        Equation: ``(compile_time / saving_per_item).to('').magnitude``."""
        return (self.compile_time / self.saving_per_item).to("").magnitude

    @property
    def speedup(self) -> float:
        """Dimensionless compiled-to-eager throughput ratio for identical items.

        Equation: ``(compiled_rate / eager_rate).to('').magnitude``."""
        return (self.compiled_rate / self.eager_rate).to("").magnitude

    @property
    def experiment_eager(self) -> Quantity:
        """Seconds for all experiment items at the eager rate, excluding compilation.

        Equation: ``(experiment_items / eager_rate).to('second')``."""
        return (self.experiment_items / self.eager_rate).to("second")

    @property
    def experiment_compiled_running(self) -> Quantity:
        """Seconds executing all experiment items after compilation, excluding compile costs.

        Equation: ``(experiment_items / compiled_rate).to('second')``."""
        return (self.experiment_items / self.compiled_rate).to("second")

    @property
    def experiment_compiled_total(self) -> Quantity:
        """Total seconds including one compile charge per declared recompilation.

        Equation: ``(recompiles * compile_time + experiment_compiled_running).to('second')``."""
        return (self.recompiles * self.compile_time + self.experiment_compiled_running).to("second")

    @property
    def experiment_slowdown(self) -> float:
        """Dimensionless total compiled-experiment time divided by eager-experiment time.

        Equation: ``(experiment_compiled_total / experiment_eager).to('').magnitude``."""
        return (self.experiment_compiled_total / self.experiment_eager).to("").magnitude


class JacobianScenario(BaseScenario):
    """The full Jacobian of one layer from ``input_width`` inputs to ``output_width`` outputs.

    Reverse-mode differentiation never builds it; it chains vector-Jacobian products.

    Notes
    -----
    Equation: Output width times input width counts entries; entry width converts the count
    to bytes.

    Assumptions: A dense rectangular Jacobian with one stored scalar per input-output pair.

    Excludes: Sparse structure and vector-Jacobian product implementations.

    References
    ----------
    Derivation: Output width times input width counts entries; entry width converts the
    count to bytes. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Dense100kLayerJacobianFp32
    >>> round(float(scenario.elements), 4)
    10000000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    input_width: int
    output_width: int
    bytes_per_value: Quantity

    @property
    def elements(self) -> int:
        """Scalar entry count of the dense output-by-input Jacobian.

        Equation: ``input_width * output_width``."""
        return self.input_width * self.output_width

    @property
    def size(self) -> Quantity:
        """Bytes required to materialize every Jacobian entry at the declared value width.

        Equation: ``(elements * bytes_per_value).to('byte')``."""
        return (self.elements * self.bytes_per_value).to("byte")


class HostInputPipelineScenario(BaseScenario):
    """A data loader feeding one accelerator over its host link.

    ``workers`` loader processes each prepare ``prefetch_factor`` batches of
    ``batch_size`` samples. A sample is ``raw_sample`` bytes as decoded pixels
    and ``model_sample`` bytes as the tensor copied to the device; the
    accelerator consumes ``consume_rate`` samples a second. Copy times are
    bandwidth-only lower bounds; ``step_times`` are compute times a serial
    copy is compared with.

    Notes
    -----
    Equation: Batch size times tensor bytes gives payload; payload divided by bandwidth
    gives copy time.

    Assumptions: Constant consumption rate and ideal bandwidth for complete model-tensor
    copies.

    Excludes: Worker scheduling, contention, and copy/compute overlap.

    References
    ----------
    Derivation: Batch size times tensor bytes gives payload; payload divided by bandwidth
    gives copy time. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ImageNetBatch64HostFeed
    >>> round(float(scenario.batches_in_flight), 4)
    8.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    batch_size: int
    workers: int
    prefetch_factor: int
    raw_sample: Quantity
    model_sample: Quantity
    consume_rate: Quantity  # samples per second
    host_link: Quantity  # bandwidth
    device_memory: Quantity  # bandwidth
    step_times: Dict[str, Quantity] = {}

    @property
    def batches_in_flight(self) -> int:
        """Prefetched batch count, with an independent prefetch allowance for each worker.

        Equation: ``workers * prefetch_factor``."""
        return self.workers * self.prefetch_factor

    @property
    def raw_feed(self) -> Quantity:
        """Decoded input bandwidth in decimal MB/s at the accelerator's sample-consumption rate.

        Equation: ``(consume_rate * raw_sample).to('MB/second')``."""
        return (self.consume_rate * self.raw_sample).to("MB/second")

    @property
    def batch_bytes(self) -> Quantity:
        """Bytes copied per batch, using the model tensor size rather than decoded raw pixels.

        Equation: ``(batch_size * model_sample).to('byte')``."""
        return (self.batch_size * self.model_sample).to("byte")

    @property
    def copy_time(self) -> Quantity:
        """Host-link transfer floor in milliseconds, from
        ``mlsysim.physics.quantities.transfer_time``.

        Equation: ``transfer_time(batch_bytes, host_link).to('ms')``."""
        from ...physics import transfer_time

        return transfer_time(self.batch_bytes, self.host_link).to("ms")

    @property
    def device_copy_time(self) -> Quantity:
        """Device-memory transfer floor in microseconds for the same batch payload.

        Equation: ``transfer_time(batch_bytes, device_memory).to('microsecond')``."""
        from ...physics import transfer_time

        return transfer_time(self.batch_bytes, self.device_memory).to("microsecond")

    def copy_share(self, step: str) -> float:
        """Dimensionless serial host-copy time divided by the named ``step`` compute time; it does
        not model overlap.

        A serial copy as a share of one named step's compute time.

        Equation: ``(copy_time / step_times[step]).to('').magnitude``."""
        return (self.copy_time / self.step_times[step]).to("").magnitude


_MLP_FORWARD_OPS = ("matmul1", "relu", "matmul2", "loss")


_MLP_BACKWARD_OPS = ("grad_w2", "grad_h", "grad_w1", "relu_backward", "loss_backward")


class MlpTrainingStepScenario(BaseScenario):
    """One eager training step of a one-hidden-layer MLP, operator by operator.

    Forward: ``x @ W1``, ReLU, ``h @ W2``, and a combined cross-entropy loss
    (``loss_flops_per_logit`` operations per logit). Backward: the weight
    gradients of both layers and the hidden-activation gradient (raw inputs
    need no gradient), the ReLU derivative, and the loss derivative, which
    reads saved probabilities and integer labels of ``label_bytes`` each.
    Bias adds are folded into their matrix multiplies. Every value is
    ``bytes_per_value`` wide. Times are ideal lower bounds on ``hardware``;
    the host issues one Python entry per forward operator plus one for
    ``backward()``, and launches one kernel per operator.

    Notes
    -----
    Equation: Each dense multiply counts two FLOPs per MAC; operator tables separately sum
    work and tensor movement.

    Assumptions: A dense two-layer network, separate eager operators, and the explicit
    operation-count conventions above.

    Excludes: Cross-operator fusion, cache reuse, optimizer FLOPs, and measured utilization.

    References
    ----------
    Derivation: Each dense multiply counts two FLOPs per MAC; operator tables separately sum
    work and tensor movement. Individual property contracts identify the exact
    accounting expression or reusable calculation. Registry entries supply the
    scenario inputs; this class does not turn them into measured outcomes.

    Examples
    --------
    Evaluate a named scenario using its declared inputs.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.MnistTwoLayerTrainingStepOnH100
    >>> round(float(scenario.op_flops['matmul1'].magnitude), 4)
    6422528.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    batch_size: int
    input_width: int
    hidden_width: int
    output_width: int
    bytes_per_value: Quantity
    label_bytes: Quantity
    loss_flops_per_logit: int
    optimizer_values_moved: int  # values read and written per parameter by one optimizer step
    hardware: HardwareNode
    precision: str = "fp32"
    forward_ops: int
    backward_ops: int
    dispatch: Quantity
    launch: Quantity

    def _values(self, n: int) -> Quantity:
        return (n * self.bytes_per_value).to("byte")

    @property
    def op_flops(self) -> Dict[str, Quantity]:
        """Per-operator FLOP counts: dense matrix multiplication costs twice its MAC count; ReLU
        and loss use the declared scalar-operation convention.

        Equation: For matrix shapes (m,k) and (k,n), FLOPs=``2*m*k*n``; elementwise counts use the
        batch and feature dimensions stated in the operator table."""
        b, i, h, o = self.batch_size, self.input_width, self.hidden_width, self.output_width
        f = {
            "matmul1": 2 * b * i * h,
            "relu": b * h,
            "matmul2": 2 * b * h * o,
            "loss": self.loss_flops_per_logit * b * o,
            "grad_w2": 2 * h * b * o,
            "grad_h": 2 * b * o * h,
            "grad_w1": 2 * i * b * h,
            "relu_backward": b * h,
            "loss_backward": 2 * b * o,
        }
        return {k: Q_(v, "flop") for k, v in f.items()}

    @property
    def op_bytes(self) -> Dict[str, Quantity]:
        """Per-operator bytes read and written, including saved activations and integer labels;
        tensors are counted separately for each eager operator.

        Equation: For each operator, bytes=``value_width*(input`` values+output values); loss
        backward additionally reads ``batch_size*label_bytes``."""
        b, i, h, o = self.batch_size, self.input_width, self.hidden_width, self.output_width
        v = self._values
        return {
            "matmul1": v(b * i + i * h + b * h),
            "relu": v(2 * b * h),
            "matmul2": v(b * h + h * o + b * o),
            "loss": v(2 * b * o),
            "grad_w2": v(b * h + b * o + h * o),
            "grad_h": v(b * o + h * o + b * h),
            "grad_w1": v(b * i + b * h + i * h),
            "relu_backward": v(3 * b * h),
            "loss_backward": (v(2 * b * o) + b * self.label_bytes).to("byte"),
        }

    def _sum(self, table, keys) -> Quantity:
        return sum((table[k] for k in keys[1:]), table[keys[0]])

    @property
    def forward_flops(self) -> Quantity:
        """FLOPs summed over the declared forward operators in ``op_flops``.

        Equation: ``_sum(op_flops, _MLP_FORWARD_OPS)``."""
        return self._sum(self.op_flops, _MLP_FORWARD_OPS)

    @property
    def backward_flops(self) -> Quantity:
        """FLOPs summed over the declared backward operators in ``op_flops``.

        Equation: ``_sum(op_flops, _MLP_BACKWARD_OPS)``."""
        return self._sum(self.op_flops, _MLP_BACKWARD_OPS)

    @property
    def forward_bytes(self) -> Quantity:
        """Bytes summed over the forward operators in ``op_bytes`` without cross-operator cache
        reuse.

        Equation: ``_sum(op_bytes, _MLP_FORWARD_OPS)``."""
        return self._sum(self.op_bytes, _MLP_FORWARD_OPS)

    @property
    def backward_bytes(self) -> Quantity:
        """Bytes summed over the backward operators in ``op_bytes`` without fusion.

        Equation: ``_sum(op_bytes, _MLP_BACKWARD_OPS)``."""
        return self._sum(self.op_bytes, _MLP_BACKWARD_OPS)

    @property
    def total_flops(self) -> Quantity:
        """Forward plus backward FLOPs per batch; optimizer update work is not included.

        Equation: ``forward_flops + backward_flops``."""
        return self.forward_flops + self.backward_flops

    @property
    def total_bytes(self) -> Quantity:
        """Forward plus backward bytes per batch; optimizer traffic is accounted separately.

        Equation: ``forward_bytes + backward_bytes``."""
        return self.forward_bytes + self.backward_bytes

    @property
    def dense_training_ratio(self) -> float:
        """Dimensionless total training-matmul work divided by forward-matmul work; raw inputs do
        not require gradients.

        Matrix-multiply work of the whole step over the forward matrix multiplies.

        Equation: (forward matmul FLOPs+weight-gradient FLOPs+hidden-gradient FLOPs)/forward
        matmul FLOPs."""
        f = self.op_flops
        fwd = f["matmul1"] + f["matmul2"]
        return ((fwd + f["grad_w2"] + f["grad_h"] + f["grad_w1"]) / fwd).to("").magnitude

    @property
    def parameters(self) -> int:
        """Trainable scalar count including both dense weight matrices and their bias vectors.

        Equation: ``i, h, o = (input_width, hidden_width, output_width); result = i * h + h + h
        * o + o``."""
        i, h, o = self.input_width, self.hidden_width, self.output_width
        return i * h + h + h * o + o

    @property
    def optimizer_step_bytes(self) -> Quantity:
        """Bytes moved by one optimizer update, using the supplied values-moved count per
        parameter.

        Equation: ``(parameters * optimizer_values_moved * bytes_per_value).to('byte')``."""
        return (self.parameters * self.optimizer_values_moved * self.bytes_per_value).to("byte")

    @property
    def peak_rate(self) -> Quantity:
        """Hardware peak FLOP/s for the selected precision, without an efficiency derating.

        Equation: ``hardware.compute.precision_flops[precision]``."""
        return self.hardware.compute.precision_flops[self.precision]

    @property
    def compute_time(self) -> Quantity:
        """Ideal microseconds for forward and backward work at the selected peak FLOP/s.

        Equation: ``(total_flops / peak_rate).to('microsecond')``."""
        return (self.total_flops / self.peak_rate).to("microsecond")

    @property
    def memory_time(self) -> Quantity:
        """Ideal microseconds for eager tensor traffic at peak memory bandwidth.

        Equation: ``(total_bytes / hardware.memory.bandwidth).to('microsecond')``."""
        return (self.total_bytes / self.hardware.memory.bandwidth).to("microsecond")

    @property
    def host(self) -> HostDispatchScenario:
        """Host-dispatch scenario combining forward Python entries and forward/backward kernel
        launches; device time is the sum of the compute and memory floors.

        The host's dispatch and launch work against the step's ideal device time.

        Equation: Python entries=forward_ops+1; kernel launches=forward_ops+backward_ops; ideal
        device time=compute_time+memory_time."""
        return HostDispatchScenario(
            name=f"Host work of: {self.name}",
            description="Python entries and kernel launches of the step.",
            ops_per_block=self.forward_ops + 1,
            launches_per_block=self.forward_ops + self.backward_ops,
            dispatch=self.dispatch,
            launch=self.launch,
            device_time=self.compute_time + self.memory_time,
        )


class ClusterExpansionScenario(BaseScenario):
    """A synchronous job grown from ``n_devices`` to ``factor`` times as many on the same checkpoint tier.

    ``checkpoint`` supplies the write path; at each size its MTBF is replaced by the
    GPU-only interval ``component_mtbf / N`` and the overhead is the checkpoint-and-rework
    waste at the Young-Daly optimum. For fixed work, run time is proportional to step
    time over the retained fraction (one minus that overhead), so the larger job shortens
    the run only when its step speedup exceeds ``breakeven_speedup``.

    Notes
    -----
    Equation: ``breakeven_speedup`` is the worked quantity; each property below states its equation and units.

    Assumptions: GPU failures are independent and fixed work experiences the checkpoint solver's steady-state waste.

    Excludes: Non-GPU failures, changing failure distributions, and prediction of the supplied step speedup.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_mtbf_cluster``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt3ClusterDoubling1K
    >>> value = case.breakeven_speedup
    >>> round(value, 3)
    1.007"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    checkpoint: CheckpointIntervalScenario
    component_mtbf: Quantity
    n_devices: int
    factor: float

    @property
    def expanded_devices(self) -> int:
        """Expanded accelerator count rounded to the nearest integer.

        Equation:
            return round(self.n_devices * self.factor)"""
        return round(self.n_devices * self.factor)

    def at(self, n_devices: int) -> CheckpointIntervalScenario:
        """Checkpoint scenario with GPU-only independent-failure MTBF at the supplied device count.

        The checkpoint scenario against the GPU-only interval of ``n_devices``.

        Equation:
            return self.checkpoint.variant(mtbf=calc_mtbf_cluster(self.component_mtbf, n_devices).to('hour'))"""
        from ...physics.reliability import calc_mtbf_cluster

        return self.checkpoint.variant(mtbf=calc_mtbf_cluster(self.component_mtbf, n_devices).to("hour"))

    @property
    def mtbf(self) -> Quantity:
        """Original fleet mean time between failures from the adjusted checkpoint scenario.

        Equation:
            return self.at(self.n_devices).mtbf"""
        return self.at(self.n_devices).mtbf

    @property
    def expanded_mtbf(self) -> Quantity:
        """Expanded fleet mean time between failures from the adjusted checkpoint scenario.

        Equation:
            return self.at(self.expanded_devices).mtbf"""
        return self.at(self.expanded_devices).mtbf

    @property
    def interval(self) -> Quantity:
        """Original fleet optimal checkpoint interval from the checkpoint solver.

        Equation:
            return self.at(self.n_devices).result.interval"""
        return self.at(self.n_devices).result.interval

    @property
    def expanded_interval(self) -> Quantity:
        """Expanded fleet optimal checkpoint interval from the checkpoint solver.

        Equation:
            return self.at(self.expanded_devices).result.interval"""
        return self.at(self.expanded_devices).result.interval

    @property
    def overhead(self) -> float:
        """Original fleet checkpoint-and-rework waste fraction.

        Equation:
            return self.at(self.n_devices).result.waste"""
        return self.at(self.n_devices).result.waste

    @property
    def expanded_overhead(self) -> float:
        """Expanded fleet checkpoint-and-rework waste fraction.

        Equation:
            return self.at(self.expanded_devices).result.waste"""
        return self.at(self.expanded_devices).result.waste

    @property
    def overhead_growth(self) -> float:
        """Dimensionless expanded-to-original waste ratio.

        Equation:
            return self.expanded_overhead / self.overhead"""
        return self.expanded_overhead / self.overhead

    @property
    def retained(self) -> float:
        """Original fleet productive-time fraction, one minus waste.

        Equation:
            return 1 - self.overhead"""
        return 1 - self.overhead

    @property
    def expanded_retained(self) -> float:
        """Expanded fleet productive-time fraction, one minus waste.

        Equation:
            return 1 - self.expanded_overhead"""
        return 1 - self.expanded_overhead

    @property
    def breakeven_speedup(self) -> float:
        """Step speedup needed for equal fixed-work completion time after reliability overhead.

        Step speedup at which the larger job finishes the fixed work in the same time.

        Equation:
            return self.retained / self.expanded_retained"""
        return self.retained / self.expanded_retained

    @property
    def superlinear_threshold(self) -> float:
        """Step speedup needed for equal accelerator-hours after device-count and reliability changes.

        Step speedup above which the larger job also uses fewer accelerator-hours.

        Equation:
            return self.factor * self.breakeven_speedup"""
        return self.factor * self.breakeven_speedup

    def run_time_change(self, step_speedup: float) -> float:
        """Fractional wall-time change for the supplied positive step speedup.

        Fractional change in wall-clock run time at a measured step speedup.

        Equation:
            return self.breakeven_speedup / step_speedup - 1"""
        return self.breakeven_speedup / step_speedup - 1

    def accelerator_hours_change(self, step_speedup: float) -> float:
        """Fractional accelerator-hour change for the supplied positive step speedup.

        Fractional change in accelerator-hours (devices times run time) at a step speedup.

        Equation:
            return self.superlinear_threshold / step_speedup - 1"""
        return self.superlinear_threshold / step_speedup - 1


class ExpertWeightTrafficScenario(BaseScenario):
    """Weight bytes of the mixture-of-experts ``workload``, resident and read per token, against a
    dense model of ``dense_parameters`` that reads every weight each step.

    Notes
    -----
    Equation: ``resident_bytes`` is the worked quantity; each property below states its equation and units.

    Assumptions: Resident storage uses total MoE parameters while per-token expert traffic uses active parameters.

    Excludes: Routing skew, expert-cache behavior, activation traffic, and communication.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DeepSeekV3WeightTraffic
    >>> value = case.resident_bytes
    >>> print(f"{value:.3f~P}")
    1342.000 GB"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    dense_parameters: Quantity
    bytes_per_param: Quantity

    @property
    def resident_bytes(self) -> Quantity:
        """Resident MoE weight storage in GB using total parameter count.

        Equation:
            return memory_from_params(self.workload.parameters, self.bytes_per_param).to('GB')"""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("GB")

    @property
    def active_bytes(self) -> Quantity:
        """Active MoE weight traffic in GB using active parameter count.

        Equation:
            return memory_from_params(self.workload.active_parameters, self.bytes_per_param).to('GB')"""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.active_parameters, self.bytes_per_param).to("GB")

    @property
    def dense_bytes(self) -> Quantity:
        """Dense comparison-model weight traffic in GB.

        Equation:
            return memory_from_params(self.dense_parameters, self.bytes_per_param).to('GB')"""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.dense_parameters, self.bytes_per_param).to("GB")

    @property
    def resident_to_active(self) -> float:
        """Dimensionless resident-to-active MoE weight ratio.

        Equation:
            return (self.resident_bytes / self.active_bytes).to('').magnitude"""
        return (self.resident_bytes / self.active_bytes).to("").magnitude

    @property
    def dense_to_active(self) -> float:
        """Dimensionless dense-to-active MoE weight ratio.

        Equation:
            return (self.dense_bytes / self.active_bytes).to('').magnitude"""
        return (self.dense_bytes / self.active_bytes).to("").magnitude


class WeightReadScenario(BaseScenario):
    """Every weight of ``workload`` read once from ``hardware``'s memory at ``bytes_per_weight``.

    The weight-streaming floor of one batch-one pass: per frame for a vision
    model, per token for a decoder. Unlike ``DecodeScenario`` it assumes no FLOP
    count, so it serves convolutional networks as well as decoders.
    ``energy_per_byte`` (optional) prices the reads.

    Notes
    -----
    Equation: ``read_time`` is the worked quantity; each property below states its equation and units.

    Assumptions: Weights and optional full KV state stream once at the registered peak memory bandwidth.

    Excludes: Compute, runtime overhead, cache reuse, interdevice communication, and serving-latency validation.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_7BTokenOnFlagshipPhone
    >>> value = case.read_time
    >>> print(f"{value:.3f~P}")
    273.438 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    bytes_per_weight: Quantity
    energy_per_byte: Optional[Quantity] = None

    @property
    def weight_bytes(self) -> Quantity:
        """Workload weight storage at the supplied precision, in byte units.

        Equation:
            return self.workload.size_in_bytes(self.bytes_per_weight)"""
        return self.workload.size_in_bytes(self.bytes_per_weight)

    @property
    def read_time(self) -> Quantity:
        """Weight-only streaming lower bound in milliseconds.

        Equation:
            return (self.weight_bytes / self.hardware.memory.bandwidth).to('millisecond')"""
        return (self.weight_bytes / self.hardware.memory.bandwidth).to("millisecond")

    @property
    def read_energy(self) -> Quantity:
        """Energy in joules for one weight read; requires energy_per_byte.

        Equation:
            return (self.weight_bytes * self.energy_per_byte).to('joule')"""
        return (self.weight_bytes * self.energy_per_byte).to("joule")

    @property
    def read_power(self) -> Quantity:
        """Average read power in watts under the stated energy-per-byte assumption.

        Average power of the reads alone, sustained over one pass.

        Equation:
            return (self.read_energy / self.read_time).to('watt')"""
        return (self.read_energy / self.read_time).to("watt")

    # Optional (added 2026-10-20): a decoder's KV cache read with the weights
    # every token, and weights unpacked to a wider format in a separate pass.
    context_tokens: Optional[int] = None
    kv_bytes_per_value: Optional[Quantity] = None
    unpacked_bytes_per_weight: Optional[Quantity] = None

    @property
    def kv_cache_bytes(self) -> Quantity:
        """Batch-one KV storage in bytes; requires transformer cache geometry and context settings.

        Keys and values of ``context_tokens`` tokens at batch one.

        Equation:
            return self.workload.get_kv_cache_size(seq_len=self.context_tokens, batch_size=1, precision=self.kv_bytes_per_value).to('byte')"""
        return self.workload.get_kv_cache_size(
            seq_len=self.context_tokens, batch_size=1, precision=self.kv_bytes_per_value
        ).to("byte")

    @property
    def token_bytes(self) -> Quantity:
        """Bytes read per token, all weights plus the complete resident KV cache.

        Bytes read per generated token: every weight and the whole KV cache.

        Equation:
            return (self.weight_bytes + self.kv_cache_bytes).to('byte')"""
        return (self.weight_bytes + self.kv_cache_bytes).to("byte")

    @property
    def token_time(self) -> Quantity:
        """Weight-plus-KV streaming lower bound in milliseconds per token.

        Equation:
            return (self.token_bytes / self.hardware.memory.bandwidth).to('millisecond')"""
        return (self.token_bytes / self.hardware.memory.bandwidth).to("millisecond")

    @property
    def token_rate(self) -> Quantity:
        """Reciprocal streaming time in tokens per second for one request.

        Equation:
            return (1 / self.token_time).to('1/second')"""
        return (1 / self.token_time).to("1/second")

    @property
    def separate_unpack_bytes(self) -> Quantity:
        """Bytes moved by separate unpacking, including packed read, unpacked write/read, and KV read.

        Bytes per token when a separate kernel unpacks the weights first: read the packed
        weights, write them unpacked, read them back for the matrix multiply, read the cache.

        Equation:
            unpacked = self.workload.size_in_bytes(self.unpacked_bytes_per_weight)
            return (self.weight_bytes + 2 * unpacked + self.kv_cache_bytes).to('byte')"""
        unpacked = self.workload.size_in_bytes(self.unpacked_bytes_per_weight)
        return (self.weight_bytes + 2 * unpacked + self.kv_cache_bytes).to("byte")

    @property
    def separate_unpack_time(self) -> Quantity:
        """Streaming lower bound in milliseconds including the separate unpack pass.

        Equation:
            return (self.separate_unpack_bytes / self.hardware.memory.bandwidth).to('millisecond')"""
        return (self.separate_unpack_bytes / self.hardware.memory.bandwidth).to("millisecond")


class FrameCompressionScenario(BaseScenario):
    """A per-frame model against a camera frame rate: the full model and a channel-pruned
    variant, run in floating point on the CPU and in integers on the accelerator.

    ``cpu_rate`` and ``accelerator_rate`` are the full model's frame rates. The
    pruned model's frame time scales with its share of the arithmetic, because
    channel pruning leaves dense tensors that run the same kernels on fewer
    channels (an upper bound on the saving; measurement decides).

    Notes
    -----
    Equation: ``frame_budget`` is the worked quantity; each property below states its equation and units.

    Assumptions: Pruning preserves dense kernels and frame time scales ideally with forward FLOP fraction.

    Excludes: Predicting accuracy, sparse execution, launch overhead, and empirical latency scaling.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.MobileNetV2BlurPipelinePhone
    >>> value = case.frame_budget
    >>> print(f"{value:.3f~P}")
    33.333 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    full: Workload
    pruned: Workload
    frame_rate: Quantity
    cpu_rate: Quantity
    accelerator_rate: Quantity
    float_bytes: Quantity
    int_bytes: Quantity

    @property
    def frame_budget(self) -> Quantity:
        """Available frame duration in milliseconds at the camera cadence.

        Equation:
            return (1 / self.frame_rate).to('millisecond')"""
        return (1 / self.frame_rate).to("millisecond")

    @property
    def cpu_frame(self) -> Quantity:
        """Full-model CPU frame duration in milliseconds.

        Equation:
            return (1 / self.cpu_rate).to('millisecond')"""
        return (1 / self.cpu_rate).to("millisecond")

    @property
    def accelerator_frame(self) -> Quantity:
        """Full-model accelerator frame duration in milliseconds.

        Equation:
            return (1 / self.accelerator_rate).to('millisecond')"""
        return (1 / self.accelerator_rate).to("millisecond")

    @property
    def flop_fraction(self) -> float:
        """Dimensionless pruned-to-full forward-FLOP ratio.

        Equation:
            return (self.pruned.inference_flops / self.full.inference_flops).to('').magnitude"""
        return (self.pruned.inference_flops / self.full.inference_flops).to("").magnitude

    @property
    def param_fraction(self) -> float:
        """Dimensionless pruned-to-full parameter-count ratio.

        Equation:
            return (self.pruned.parameters / self.full.parameters).to('').magnitude"""
        return (self.pruned.parameters / self.full.parameters).to("").magnitude

    @property
    def full_bytes(self) -> Quantity:
        """Full-model floating-point weight storage in MB.

        Equation:
            return self.full.size_in_bytes(self.float_bytes).to('MB')"""
        return self.full.size_in_bytes(self.float_bytes).to("MB")

    @property
    def pruned_bytes(self) -> Quantity:
        """Pruned-model floating-point weight storage in MB.

        Equation:
            return self.pruned.size_in_bytes(self.float_bytes).to('MB')"""
        return self.pruned.size_in_bytes(self.float_bytes).to("MB")

    @property
    def pruned_int_bytes(self) -> Quantity:
        """Pruned-model integer weight storage in MB.

        Equation:
            return self.pruned.size_in_bytes(self.int_bytes).to('MB')"""
        return self.pruned.size_in_bytes(self.int_bytes).to("MB")

    @property
    def pruned_cpu_frame(self) -> Quantity:
        """Ideal pruned CPU frame duration in milliseconds, scaled only by FLOP fraction.

        Equation:
            return (self.cpu_frame * self.flop_fraction).to('millisecond')"""
        return (self.cpu_frame * self.flop_fraction).to("millisecond")

    @property
    def pruned_accelerator_frame(self) -> Quantity:
        """Ideal pruned accelerator frame duration in milliseconds, scaled only by FLOP fraction.

        Equation:
            return (self.accelerator_frame * self.flop_fraction).to('millisecond')"""
        return (self.accelerator_frame * self.flop_fraction).to("millisecond")


class ConvBnReluFusionScenario(BaseScenario):
    """A convolution, batch norm, and ReLU on one feature map, run as three kernels or one.

    Unfused, each kernel reads its input from device memory and writes its
    output back (six transfers), and the convolution and norm also read their
    parameters; fused (the norm folded into the convolution's weights and bias,
    the ReLU applied before the write), one read and one write remain. A lower
    bound on time: traffic over ``hardware``'s memory bandwidth, compute ignored.

    Notes
    -----
    Equation: ``traffic_cut`` is the worked quantity; each property below states its equation and units.

    Assumptions: Batch normalization folds into dense convolution parameters and ReLU fuses before the output write.

    Excludes: Compute work, cache hits, workspace, and measured kernel-fusion speedup.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50Conv256FusionH100
    >>> value = case.traffic_cut
    >>> round(value, 3)
    0.448"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    channels: int
    height: int
    width: int
    kernel_size: int
    bytes_per_value: Quantity
    norm_params_per_channel: int
    hardware: HardwareNode

    @property
    def feature_map(self) -> Quantity:
        """One feature-map footprint in KB.

        Equation:
            return (self.channels * self.height * self.width * self.bytes_per_value).to('KB')"""
        return (self.channels * self.height * self.width * self.bytes_per_value).to("KB")

    @property
    def conv_weights(self) -> Quantity:
        """Dense same-width convolution weight footprint in MB.

        Equation:
            return (self.channels * self.channels * self.kernel_size ** 2 * self.bytes_per_value).to('MB')"""
        return (self.channels * self.channels * self.kernel_size**2 * self.bytes_per_value).to("MB")

    @property
    def norm_params(self) -> Quantity:
        """Batch-normalization parameter storage in KB.

        Equation:
            return (self.channels * self.norm_params_per_channel * self.bytes_per_value).to('KB')"""
        return (self.channels * self.norm_params_per_channel * self.bytes_per_value).to("KB")

    @property
    def conv_traffic(self) -> Quantity:
        """Convolution traffic in MB, one feature-map read/write plus weights.

        Equation:
            return (2 * self.feature_map + self.conv_weights).to('MB')"""
        return (2 * self.feature_map + self.conv_weights).to("MB")

    @property
    def norm_traffic(self) -> Quantity:
        """Normalization traffic in MB, one feature-map read/write plus norm parameters.

        Equation:
            return (2 * self.feature_map + self.norm_params).to('MB')"""
        return (2 * self.feature_map + self.norm_params).to("MB")

    @property
    def relu_traffic(self) -> Quantity:
        """ReLU traffic in MB, one feature-map read/write.

        Equation:
            return (2 * self.feature_map).to('MB')"""
        return (2 * self.feature_map).to("MB")

    @property
    def unfused_trips(self) -> int:
        """Activation-transfer count for separate convolution, normalization, and ReLU kernels.

        Equation:
            return 6"""
        return 6

    @property
    def fused_trips(self) -> int:
        """Activation-transfer count after folding normalization and fusing ReLU.

        Equation:
            return 2"""
        return 2

    @property
    def unfused_traffic(self) -> Quantity:
        """Total unfused traffic in MB across the three kernels.

        Equation:
            return (self.conv_traffic + self.norm_traffic + self.relu_traffic).to('MB')"""
        return (self.conv_traffic + self.norm_traffic + self.relu_traffic).to("MB")

    @property
    def fused_traffic(self) -> Quantity:
        """Fused traffic in MB after folding norm parameters into convolution.

        Equation:
            return self.conv_traffic"""
        return self.conv_traffic

    @property
    def eliminated_activation_traffic(self) -> Quantity:
        """Activation traffic eliminated by fusion, four feature-map transfers in MB.

        Activation transfers fusion removes (four feature-map trips).

        Equation:
            return ((self.unfused_trips - self.fused_trips) * self.feature_map).to('MB')"""
        return ((self.unfused_trips - self.fused_trips) * self.feature_map).to("MB")

    @property
    def traffic_cut(self) -> float:
        """Dimensionless fractional traffic reduction from fusion.

        Equation:
            return 1 - (self.fused_traffic / self.unfused_traffic).to('').magnitude"""
        return 1 - (self.fused_traffic / self.unfused_traffic).to("").magnitude

    @property
    def unfused_time(self) -> Quantity:
        """Unfused memory-bandwidth time floor in microseconds.

        Equation:
            return (self.unfused_traffic / self.hardware.memory.bandwidth).to('microsecond')"""
        return (self.unfused_traffic / self.hardware.memory.bandwidth).to("microsecond")

    @property
    def fused_time(self) -> Quantity:
        """Fused memory-bandwidth time floor in microseconds.

        Equation:
            return (self.fused_traffic / self.hardware.memory.bandwidth).to('microsecond')"""
        return (self.fused_traffic / self.hardware.memory.bandwidth).to("microsecond")

    @property
    def time_ratio(self) -> float:
        """Dimensionless unfused-to-fused memory-time ratio.

        Equation:
            return (self.unfused_time / self.fused_time).to('').magnitude"""
        return (self.unfused_time / self.fused_time).to("").magnitude


class SparseFlashFitScenario(BaseScenario):
    """``workload`` pruned until its surviving values total ``surviving_bytes``, stored with one
    ``index_bytes`` position index per value, against a ``flash_budget``.

    Notes
    -----
    Equation: ``fits`` is the worked quantity; each property below states its equation and units.

    Assumptions: Each nonzero stores one value and one index; surviving_bytes counts only surviving values.

    Excludes: Sparse row pointers, alignment, runtime workspaces, and execution-speed prediction.

    References
    ----------
    Derivation: The scenario binds declared inputs to ``calc_sparse_bytes_per_nonzero``. Those implementations document their derivations and sources.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.WearableMobileNetV2SparseFlash
    >>> value = case.fits
    >>> value
    False"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    value_bytes: Quantity
    index_bytes: Quantity
    surviving_bytes: Quantity
    flash_budget: Quantity

    @property
    def dense_bytes(self) -> Quantity:
        """Original dense checkpoint size in MB.

        Equation:
            return self.workload.size_in_bytes(self.value_bytes).to('MB')"""
        return self.workload.size_in_bytes(self.value_bytes).to("MB")

    @property
    def pruned_share(self) -> float:
        """Fraction of dense value storage removed before accounting for sparse indices.

        Equation:
            return 1 - (self.surviving_bytes / self.dense_bytes).to('').magnitude"""
        return 1 - (self.surviving_bytes / self.dense_bytes).to("").magnitude

    @property
    def bytes_per_nonzero(self) -> Quantity:
        """Stored bytes per surviving nonzero, value plus index width.

        Equation:
            return calc_sparse_bytes_per_nonzero(self.value_bytes, self.index_bytes)"""
        from ...physics.memory import calc_sparse_bytes_per_nonzero

        return calc_sparse_bytes_per_nonzero(self.value_bytes, self.index_bytes)

    @property
    def stored_bytes(self) -> Quantity:
        """Sparse checkpoint size in MB including one index per surviving value.

        Equation:
            return (self.surviving_bytes * (self.bytes_per_nonzero / self.value_bytes)).to('MB')"""
        return (self.surviving_bytes * (self.bytes_per_nonzero / self.value_bytes)).to("MB")

    @property
    def fits(self) -> bool:
        """Whether indexed sparse storage fits within the declared flash budget.

        Equation:
            return self.stored_bytes <= self.flash_budget"""
        return self.stored_bytes <= self.flash_budget


class SpeedupCompositionScenario(BaseScenario):
    """Independent paper speedups multiplied together, against the ``measured`` speedup.

    Notes
    -----
    Equation: ``paper`` is the worked quantity; each property below states its equation and units.

    Assumptions: The product assumes independent component speedups over compatible baselines.

    Excludes: Establishing independence or predicting end-to-end measured performance.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PrunedInt8PaperVsMeasured
    >>> value = case.paper
    >>> round(value, 3)
    8.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    factors: Dict[str, float]
    measured: float

    @property
    def paper(self) -> float:
        """Product of supplied dimensionless speedup factors under independence.

        Equation:
            out = 1.0
            for f in self.factors.values():
                out *= f
            return out"""
        out = 1.0
        for f in self.factors.values():
            out *= f
        return out

    @property
    def realized_share(self) -> float:
        """Dimensionless observed speedup divided by the product of component factors.

        Equation:
            return self.measured / self.paper"""
        return self.measured / self.paper


class WorkLatencyGapScenario(BaseScenario):
    """A cut in work (parameters or FLOPs) against the latency cut it delivered.

    Notes
    -----
    Equation: ``delivered_share`` is the worked quantity; each property below states its equation and units.

    Assumptions: Both supplied reductions refer to the same workload and baseline, with nonzero work reduction.

    Excludes: Causal attribution or a universal relation between FLOPs and latency.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.IrregularSparsityLatencyTrace
    >>> value = case.delivered_share
    >>> round(value, 3)
    0.3"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    work_reduction: float
    latency_reduction: float

    @property
    def delivered_share(self) -> float:
        """Dimensionless latency-cut share of the supplied work reduction.

        Share of the work cut that reached latency.

        Equation:
            return self.latency_reduction / self.work_reduction"""
        return self.latency_reduction / self.work_reduction


class StagedCompressionScenario(BaseScenario):
    """A teacher pruned and distilled into a compact student, then quantized, with the
    task accuracy after each stage and the loss when quantization comes first.

    Notes
    -----
    Equation: ``full_bytes`` is the worked quantity; each property below states its equation and units.

    Assumptions: Student size, precisions, and stage accuracies are supplied inputs.

    Excludes: Predicting accuracy, runtime speedup, or using intermediate accuracy fields in final_loss.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.BertBaseMobileCompression
    >>> value = case.full_bytes
    >>> print(f"{value:.3f~P}")
    440.000 MB"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    teacher: Workload
    student_params: Quantity
    float_bytes: Quantity
    int_bytes: Quantity
    baseline_accuracy: float
    pruned_accuracy: float
    distilled_accuracy: float
    final_accuracy: float
    mobile_speedup: float
    quantize_first_loss: float

    @property
    def full_bytes(self) -> Quantity:
        """Teacher floating-point checkpoint size in MB.

        Equation:
            return self.teacher.size_in_bytes(self.float_bytes).to('MB')"""
        return self.teacher.size_in_bytes(self.float_bytes).to("MB")

    @property
    def student_float_bytes(self) -> Quantity:
        """Student floating-point checkpoint size in MB.

        Equation:
            return (self.student_params.to('param').magnitude * self.float_bytes).to('MB')"""
        return (self.student_params.to("param").magnitude * self.float_bytes).to("MB")

    @property
    def compressed_bytes(self) -> Quantity:
        """Quantized student checkpoint size in MB.

        Equation:
            return (self.student_params.to('param').magnitude * self.int_bytes).to('MB')"""
        return (self.student_params.to("param").magnitude * self.int_bytes).to("MB")

    @property
    def parameter_reduction(self) -> float:
        """Fraction of teacher parameters removed in the student.

        Equation:
            return 1 - (self.student_params / self.teacher.parameters).to('').magnitude"""
        return 1 - (self.student_params / self.teacher.parameters).to("").magnitude

    @property
    def compression_ratio(self) -> float:
        """Dimensionless teacher-to-compressed-student storage ratio.

        Equation:
            return (self.full_bytes / self.compressed_bytes).to('').magnitude"""
        return (self.full_bytes / self.compressed_bytes).to("").magnitude

    @property
    def final_loss(self) -> float:
        """Supplied baseline-minus-final accuracy difference, preserving the input accuracy scale.

        Equation:
            return self.baseline_accuracy - self.final_accuracy"""
        return self.baseline_accuracy - self.final_accuracy


class ElementwiseOpScenario(BaseScenario):
    """An element-wise or normalization kernel over ``elements`` values, run unfused.

    Each element is read ``reads`` times and written ``writes`` times from
    device memory; ``parameter_values`` (a LayerNorm's scale and shift) are
    read once. ``flops_per_element`` counts the arithmetic on each element.

    Notes
    -----
    Equation: ``intensity`` is the worked quantity; each property below states its equation and units.

    Assumptions: An unfused kernel performs fixed per-element work and reads parameters once.

    Excludes: Cache reuse, kernel fusion, vectorization, and measured execution latency.

    References
    ----------
    Derivation: The equations below follow the stated counts, geometric relations, or accounting definitions; scenario inputs are supplied assumptions or observations, not fitted predictions.

    Examples
    --------
    Execute the registered worked inputs; this is a modeled calculation.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Gpt2XlLayerNormPrefill
    >>> value = case.intensity
    >>> round(value, 3)
    2.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    elements: int
    flops_per_element: float
    bytes_per_value: Quantity
    parameter_values: int = 0
    reads: int = 1
    writes: int = 1

    @property
    def flops(self) -> Quantity:
        """Arithmetic work in FLOPs, element count times FLOPs per element.

        Equation:
            return Q_(self.elements * self.flops_per_element, 'flop')"""
        return Q_(self.elements * self.flops_per_element, "flop")

    @property
    def input_bytes(self) -> Quantity:
        """Input traffic in bytes, element footprint times read count.

        Equation:
            return (self.reads * self.elements * self.bytes_per_value).to('byte')"""
        return (self.reads * self.elements * self.bytes_per_value).to("byte")

    @property
    def output_bytes(self) -> Quantity:
        """Output traffic in bytes, element footprint times write count.

        Equation:
            return (self.writes * self.elements * self.bytes_per_value).to('byte')"""
        return (self.writes * self.elements * self.bytes_per_value).to("byte")

    @property
    def parameter_bytes(self) -> Quantity:
        """Parameter traffic in bytes, stored parameter values read once.

        Equation:
            return (self.parameter_values * self.bytes_per_value).to('byte')"""
        return (self.parameter_values * self.bytes_per_value).to("byte")

    @property
    def traffic_bytes(self) -> Quantity:
        """Total unfused traffic in bytes across inputs, outputs, and parameters.

        Equation:
            return self.input_bytes + self.output_bytes + self.parameter_bytes"""
        return self.input_bytes + self.output_bytes + self.parameter_bytes

    @property
    def intensity(self) -> float:
        """Arithmetic intensity in FLOPs per byte of modeled unfused traffic.

        FLOP per byte of the unfused traffic.

        Equation:
            return (self.flops / self.traffic_bytes).to('flop/byte').magnitude"""
        return (self.flops / self.traffic_bytes).to("flop/byte").magnitude


class AcceleratedFractionScenario(BaseScenario):
    """Amdahl's law with a path gain set by two rates: ``fraction`` of the
    baseline time runs ``accelerated_rate / baseline_rate`` times faster.

    The rates may be operation rates (a compute-bound path) or bandwidths (a
    bandwidth-bound path); only their ratio enters.

    Notes
    -----
    Equation: The calculation members below define ``gain``, ``speedup``, ``ceiling``.

    Assumptions: The accelerated fraction is baseline elapsed-time share; the two rates measure compatible work.

    Excludes: Changes to the unaccelerated path, contention, and overlap.

    References
    ----------
    Derivation: Amdahl's law with a path gain set by two rates: ``fraction`` of the baseline time runs ``accelerated_rate / baseline_rate`` times faster. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50InferenceH100OverServerCpu
    >>> value = case.speedup
    >>> print(f"{value:.3g}")
    18.6"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fraction: float
    accelerated_rate: Quantity
    baseline_rate: Quantity

    @property
    def gain(self) -> float:
        """Dimensionless local speedup from accelerated rate divided by baseline rate.

        Equation: ``(accelerated_rate / baseline_rate).to('').magnitude``."""
        return (self.accelerated_rate / self.baseline_rate).to("").magnitude

    @property
    def speedup(self) -> float:
        """Whole-program speedup from the Amdahl contract in ``calc_amdahls_speedup``.

        Calculation: ``mlsysim.physics.calc_amdahls_speedup``."""
        from ...physics import calc_amdahls_speedup

        return calc_amdahls_speedup(self.fraction, self.gain)

    @property
    def ceiling(self) -> float:
        """Dimensionless Amdahl ceiling with the accelerated portion taking no time.

        The speedup with an infinitely fast path: 1 / (1 - fraction).

        Calculation: ``mlsysim.physics.calc_amdahls_speedup``."""
        from ...physics import calc_amdahls_speedup

        return calc_amdahls_speedup(self.fraction, float("inf"))


class SystolicReuseEnergyScenario(BaseScenario):
    """Energy per MAC with every operand from off-chip memory, against a systolic array.

    Streaming: ``naive_accesses`` off-chip accesses per MAC (load both operands
    and the running sum, write the sum back). Systolic: ``edge_loads`` operands
    enter at the array's edge and serve ``array_dim`` MACs; partial sums stay
    in the array.

    Notes
    -----
    Equation: The calculation members below define ``naive_energy``, ``systolic_accesses``, ``systolic_energy``, ``ratio``.

    Assumptions: Each edge operand is reused across array_dim MACs; partial sums remain local; access energies are fixed.

    Excludes: Utilization loss, leakage, interconnect energy, and edge padding.

    References
    ----------
    Derivation: Energy per MAC with every operand from off-chip memory, against a systolic array. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SystolicArrayReuseEnergy
    >>> value = case.ratio
    >>> print(f"{value:.3g}")
    233"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    access_energy: Quantity
    mac_energy: Quantity
    naive_accesses: int
    edge_loads: int
    array_dim: int

    @property
    def naive_energy(self) -> Quantity:
        """Energy per MAC in pJ when every stated off-chip access is paid.

        Equation: ``(naive_accesses * access_energy + mac_energy).to('pJ')``."""
        return (self.naive_accesses * self.access_energy + self.mac_energy).to("pJ")

    @property
    def systolic_accesses(self) -> float:
        """Off-chip operand accesses per MAC after reuse across the array dimension.

        Off-chip accesses per MAC in the array.

        Equation: ``edge_loads / array_dim``."""
        return self.edge_loads / self.array_dim

    @property
    def systolic_energy(self) -> Quantity:
        """Energy per MAC in pJ after amortizing edge loads across the array.

        Equation: ``(systolic_accesses * access_energy + mac_energy).to('pJ')``."""
        return (self.systolic_accesses * self.access_energy + self.mac_energy).to("pJ")

    @property
    def ratio(self) -> float:
        """Dimensionless naive-to-systolic energy ratio; larger values favor reuse.

        Equation: ``(naive_energy / systolic_energy).to('').magnitude``."""
        return (self.naive_energy / self.systolic_energy).to("").magnitude


class ParameterBandwidthGrowthScenario(BaseScenario):
    """Dense-model parameter growth against per-device memory-bandwidth growth.

    Each series is indexed to its baseline entry; the growth is the largest
    index. ``sparse_models`` are reported but kept out of the growth figure,
    because their total parameter count does not set per-token traffic.

    Notes
    -----
    Equation: The calculation members below define ``bandwidth_index``, ``dense_index``, ``sparse_index``, ``bandwidth_growth``, ``parameter_growth``.

    Assumptions: The supplied baseline is comparable to every member; dense total parameters proxy a weight read.

    Excludes: Sparse active-parameter traffic and changes in numerical precision.

    References
    ----------
    Derivation: Dense-model parameter growth against per-device memory-bandwidth growth. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DenseParametersVsBandwidthSince2014
    >>> value = case.parameter_growth
    >>> print(f"{value:.3g}")
    3.9e+03"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    accelerators: List[HardwareNode]
    dense_models: List[Workload]
    sparse_models: List[Workload] = []
    baseline_accelerator: HardwareNode
    baseline_model: Workload

    @property
    def bandwidth_index(self) -> List[float]:
        """Each accelerator's bandwidth divided by the baseline bandwidth, dimensionless.

        Equation: ``[(h.memory.bandwidth / b0).to('').magnitude for h in accelerators]``."""
        b0 = self.baseline_accelerator.memory.bandwidth
        return [(h.memory.bandwidth / b0).to("").magnitude for h in self.accelerators]

    @property
    def dense_index(self) -> List[float]:
        """Each dense model's parameter count divided by the baseline count, dimensionless.

        Equation: ``[(m.parameters / p0).to('').magnitude for m in dense_models]``."""
        p0 = self.baseline_model.parameters
        return [(m.parameters / p0).to("").magnitude for m in self.dense_models]

    @property
    def sparse_index(self) -> List[float]:
        """Total sparse-model parameters relative to baseline; not active parameters per token.

        Equation: ``[(m.parameters / p0).to('').magnitude for m in sparse_models]``."""
        p0 = self.baseline_model.parameters
        return [(m.parameters / p0).to("").magnitude for m in self.sparse_models]

    @property
    def bandwidth_growth(self) -> float:
        """Largest dimensionless bandwidth index in the selected accelerator series.

        Equation: ``max(bandwidth_index)``."""
        return max(self.bandwidth_index)

    @property
    def parameter_growth(self) -> float:
        """Largest dense-model parameter index; sparse models are excluded.

        Equation: ``max(dense_index)``."""
        return max(self.dense_index)


class AcceleratorGenerationsScenario(BaseScenario):
    """Peak compute against memory bandwidth across ``accelerators``, first to last.

    The ridge point of each is its peak over its bandwidth; the gap growth is
    how far the ridge rose from the first to the last.

    Notes
    -----
    Equation: The calculation members below define ``ridges``, ``compute_growth``, ``bandwidth_growth``, ``gap_growth``.

    Assumptions: Input order defines the first and last generations; compute and bandwidth are peak specifications.

    Excludes: Sustained utilization, kernel dispatch, capacity, and workload-specific traffic.

    References
    ----------
    Derivation: Peak compute against memory bandwidth across ``accelerators``, first to last. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.NvidiaGenerationsV100ToB200
    >>> value = case.gap_growth
    >>> print(f"{value:.3g}")
    2.02"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    accelerators: List[HardwareNode]

    @property
    def ridges(self) -> List[float]:
        """Peak-compute-to-bandwidth ratios in FLOP/byte, one per accelerator.

        Equation: ``[h.ridge_point().to('flop/byte').magnitude for h in accelerators]``."""
        return [h.ridge_point().to("flop/byte").magnitude for h in self.accelerators]

    @property
    def compute_growth(self) -> float:
        """Dimensionless last-to-first peak compute growth for the ordered series.

        Equation: ``(b.compute.peak_flops / a.compute.peak_flops).to('').magnitude``."""
        a, b = self.accelerators[0], self.accelerators[-1]
        return (b.compute.peak_flops / a.compute.peak_flops).to("").magnitude

    @property
    def bandwidth_growth(self) -> float:
        """Dimensionless last-to-first memory bandwidth growth.

        Equation: ``(b.memory.bandwidth / a.memory.bandwidth).to('').magnitude``."""
        a, b = self.accelerators[0], self.accelerators[-1]
        return (b.memory.bandwidth / a.memory.bandwidth).to("").magnitude

    @property
    def gap_growth(self) -> float:
        """Compute growth divided by bandwidth growth, equal to ridge-point growth.

        Equation: ``compute_growth / bandwidth_growth``."""
        return self.compute_growth / self.bandwidth_growth


class PeakRatioRangeScenario(BaseScenario):
    """One peak rate against a baseline stated as a range, ``baseline_low`` to ``baseline_high``.

    Notes
    -----
    Equation: The calculation members below define ``ratio_low``, ``ratio_high``.

    Assumptions: Both baseline endpoints are positive rates measuring the same work as rate.

    Excludes: Uncertainty distributions within the range and sustained efficiency.

    References
    ----------
    Derivation: One peak rate against a baseline stated as a range, ``baseline_low`` to ``baseline_high``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.H100OverReferenceCpu
    >>> value = case.ratio_low
    >>> print(f"{value:.3g}")
    494"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rate: Quantity
    baseline_low: Quantity
    baseline_high: Quantity

    @property
    def ratio_low(self) -> float:
        """Lower dimensionless rate ratio, using the highest baseline rate.

        Against the high baseline.

        Equation: ``(rate / baseline_high).to('').magnitude``."""
        return (self.rate / self.baseline_high).to("").magnitude

    @property
    def ratio_high(self) -> float:
        """Upper dimensionless rate ratio, using the lowest baseline rate.

        Against the low baseline.

        Equation: ``(rate / baseline_low).to('').magnitude``."""
        return (self.rate / self.baseline_low).to("").magnitude


class MatmulTileScenario(BaseScenario):
    """C = A B with A ``m`` x ``k`` and B ``k`` x ``n``, computed one ``tile_m`` x ``tile_n``
    output tile at a time, stepping through the reduction ``tile_k`` elements per step.

    Operands are ``operand_bytes`` each and the accumulator ``accumulator_bytes``. The
    on-chip store holds one A tile and one B tile (two of each when double buffered)
    and the accumulator tile. Traffic counts element moves between memory and the
    on-chip store: A is loaded once per output-tile column, B once per output-tile
    row, and C written once. The single-cell loop orders (no tile) count one held
    value per cell: the other operands stream once per MAC, and a partial sum
    that is not held is read and written once per MAC.

    Notes
    -----
    Equation: The calculation members below define ``macs``, ``flops``, ``tiles_m``, ``tiles_n``, ``tiles_k``, ``output_tiles``, ``tile_products``, ``tile_product_flops``, ``tile_product_bytes``, ``tile_intensity``, ``unblocked_intensity``, ``compulsory_intensity``, ``buffer_copies``, ``a_tile_bytes``, ``b_tile_bytes``, ``accumulator_tile_bytes``, ``operand_tile_bytes``, ``live_bytes``, ``fits``, ``a_bytes``, ``a_elements``, ``b_elements``, ``c_elements``, ``a_moves``, ``b_moves``, ``tiled_moves``, ``global_loads_per_mac``, ``partial_sum_moves``, ``a_stationary_moves``, ``output_stationary_moves``, ``b_stationary_moves``, ``row_b_reads``.

    Assumptions: Dense multiplication uses two FLOPs per MAC and fixed-width operands/accumulators; full-tile work counts padded edges.

    Excludes: Bank conflicts, cache effects, scheduling overhead, and precision-dependent arithmetic throughput.

    References
    ----------
    Derivation: C = A B with A ``m`` x ``k`` and B ``k`` x ``n``, computed one ``tile_m`` x ``tile_n`` output tile at a time, stepping through the reduction ``tile_k`` elements per step. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Matmul4096Fp16Tile128
    >>> value = case.tile_intensity
    >>> print(f"{value:.3g}")
    64"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    operand_bytes: Quantity
    accumulator_bytes: Quantity
    double_buffered: bool = False

    @property
    def macs(self) -> int:
        """Multiply-accumulate count for dense M-by-K times K-by-N multiplication.

        Equation: ``m * n * k``."""
        return self.m * self.n * self.k

    @property
    def flops(self) -> Quantity:
        """Dense multiplication work in FLOPs, two operations per MAC.

        Equation: ``Q_(2 * macs, 'flop')``."""
        return Q_(2 * self.macs, "flop")

    @property
    def tiles_m(self) -> int:
        """Output tile-row count, rounding M up to a multiple of tile height.

        Equation: ``-(-m // tile_m)``."""
        return -(-self.m // self.tile_m)

    @property
    def tiles_n(self) -> int:
        """Output tile-column count, rounding N up to a multiple of tile width.

        Equation: ``-(-n // tile_n)``."""
        return -(-self.n // self.tile_n)

    @property
    def tiles_k(self) -> int:
        """Reduction tile count, rounding K up to a multiple of reduction width.

        Equation: ``-(-k // tile_k)``."""
        return -(-self.k // self.tile_k)

    @property
    def output_tiles(self) -> int:
        """Number of output tiles, tile rows times tile columns.

        Equation: ``tiles_m * tiles_n``."""
        return self.tiles_m * self.tiles_n

    @property
    def tile_products(self) -> int:
        """Tile products required, one per output tile and reduction slice.

        A-tile-by-B-tile products: every output tile sums ``tiles_k`` of them.

        Equation: ``output_tiles * tiles_k``."""
        return self.output_tiles * self.tiles_k

    @property
    def tile_product_flops(self) -> Quantity:
        """FLOPs for one full tile product; edge padding is counted as full work.

        Equation: ``Q_(2 * tile_m * tile_n * tile_k, 'flop')``."""
        return Q_(2 * self.tile_m * self.tile_n * self.tile_k, "flop")

    @property
    def tile_product_bytes(self) -> Quantity:
        """Operand bytes loaded for one full A-tile/B-tile product.

        One A tile and one B tile loaded.

        Equation: ``((tile_m * tile_k + tile_k * tile_n) * operand_bytes).to('byte')``."""
        return ((self.tile_m * self.tile_k + self.tile_k * self.tile_n) * self.operand_bytes).to("byte")

    @property
    def tile_intensity(self) -> float:
        """FLOP per operand byte loaded for a full tile, excluding output traffic.

        FLOP per operand byte loaded: (2 / b) T_M T_N / (T_M + T_N), independent of T_K.

        Equation: ``(tile_product_flops / tile_product_bytes).to('flop/byte').magnitude``."""
        return (self.tile_product_flops / self.tile_product_bytes).to("flop/byte").magnitude

    @property
    def unblocked_intensity(self) -> float:
        """FLOP/byte when every MAC loads both operands without reuse.

        No reuse: every MAC (2 FLOP) loads its two operands.

        Equation: ``(Q_(2, 'flop') / (2 * operand_bytes)).to('flop/byte').magnitude``."""
        return (Q_(2, "flop") / (2 * self.operand_bytes)).to("flop/byte").magnitude

    @property
    def compulsory_intensity(self) -> float:
        """FLOP/byte if each A, B, and C element moves exactly once at operand precision.

        Each of A, B, and C moved exactly once, all at ``operand_bytes``.

        Equation: ``(flops / moved).to('flop/byte').magnitude``."""
        moved = (self.m * self.k + self.k * self.n + self.m * self.n) * self.operand_bytes
        return (self.flops / moved).to("flop/byte").magnitude

    @property
    def buffer_copies(self) -> int:
        """Operand buffer-copy count: two for double buffering, otherwise one.

        Equation: ``2 if double_buffered else 1``."""
        return 2 if self.double_buffered else 1

    @property
    def a_tile_bytes(self) -> Quantity:
        """Live A-tile operand buffers in bytes, including double buffering if enabled.

        Equation: ``(buffer_copies * tile_m * tile_k * operand_bytes).to('byte')``."""
        return (self.buffer_copies * self.tile_m * self.tile_k * self.operand_bytes).to("byte")

    @property
    def b_tile_bytes(self) -> Quantity:
        """Live B-tile operand buffers in bytes, including double buffering if enabled.

        Equation: ``(buffer_copies * tile_k * tile_n * operand_bytes).to('byte')``."""
        return (self.buffer_copies * self.tile_k * self.tile_n * self.operand_bytes).to("byte")

    @property
    def accumulator_tile_bytes(self) -> Quantity:
        """Output-tile accumulator storage in bytes at accumulator precision.

        Equation: ``(tile_m * tile_n * accumulator_bytes).to('byte')``."""
        return (self.tile_m * self.tile_n * self.accumulator_bytes).to("byte")

    @property
    def operand_tile_bytes(self) -> Quantity:
        """Combined A and B operand-buffer bytes, excluding accumulators.

        The A and B buffers alone (what a GPU stages in shared memory).

        Equation: ``a_tile_bytes + b_tile_bytes``."""
        return self.a_tile_bytes + self.b_tile_bytes

    @property
    def live_bytes(self) -> Quantity:
        """Combined operand and accumulator bytes simultaneously resident.

        Every buffer live at once: 2 T_M T_K + 2 T_K T_N + 4 T_M T_N bytes for FP16 operands, FP32 sums.

        Equation: ``operand_tile_bytes + accumulator_tile_bytes``."""
        return self.operand_tile_bytes + self.accumulator_tile_bytes

    def fits(self, capacity: Quantity) -> bool:
        """Whether live bytes fit the supplied capacity quantity; equality is feasible.

        Equation: ``bool(live_bytes <= capacity)``."""
        return bool(self.live_bytes <= capacity)

    @property
    def a_bytes(self) -> Quantity:
        """Full A-matrix storage in bytes at operand precision.

        Equation: ``(m * k * operand_bytes).to('byte')``."""
        return (self.m * self.k * self.operand_bytes).to("byte")

    @property
    def a_elements(self) -> int:
        """Number of elements in the M-by-K input matrix A.

        Equation: ``m * k``."""
        return self.m * self.k

    @property
    def b_elements(self) -> int:
        """Number of elements in the K-by-N input matrix B.

        Equation: ``k * n``."""
        return self.k * self.n

    @property
    def c_elements(self) -> int:
        """Number of elements in the M-by-N output matrix C.

        Equation: ``m * n``."""
        return self.m * self.n

    @property
    def a_moves(self) -> int:
        """A-element reads when A is reloaded for every output tile column.

        Equation: ``a_elements * tiles_n``."""
        return self.a_elements * self.tiles_n

    @property
    def b_moves(self) -> int:
        """B-element reads when B is reloaded for every output tile row.

        Equation: ``b_elements * tiles_m``."""
        return self.b_elements * self.tiles_m

    @property
    def tiled_moves(self) -> int:
        """Total element transfers: tiled A and B reads plus one output write.

        Element moves with the tiling: A per output-tile column, B per output-tile row, C once.

        Equation: ``a_moves + b_moves + c_elements``."""
        return self.a_moves + self.b_moves + self.c_elements

    @property
    def global_loads_per_mac(self) -> float:
        """Operand loads per MAC for a full tile, reciprocal tile dimensions summed.

        Operand loads from memory per MAC inside a full tile: 1/T_N + 1/T_M.

        Equation: ``1 / tile_n + 1 / tile_m``."""
        return 1 / self.tile_n + 1 / self.tile_m

    @property
    def partial_sum_moves(self) -> int:
        """Partial-sum transfers if each MAC reads and writes its sum.

        A partial sum read and written once per MAC.

        Equation: ``2 * macs``."""
        return 2 * self.macs

    @property
    def a_stationary_moves(self) -> int:
        """Element transfers when each cell retains A while B and partial sums stream.

        One cell holds an A value: A loaded once; B and the partial sums stream.

        Equation: ``a_elements + macs + partial_sum_moves``."""
        return self.a_elements + self.macs + self.partial_sum_moves

    @property
    def output_stationary_moves(self) -> int:
        """Element transfers when output sums remain local and both operands stream.

        One cell holds the running sum: A and B stream, C written once.

        Equation: ``2 * macs + c_elements``."""
        return 2 * self.macs + self.c_elements

    @property
    def b_stationary_moves(self) -> int:
        """Element transfers when each cell retains B while A and partial sums stream.

        One cell holds a B value: B loaded once; A and the partial sums stream.

        Equation: ``b_elements + macs + partial_sum_moves``."""
        return self.b_elements + self.macs + self.partial_sum_moves

    def row_b_reads(self, cells: int) -> int:
        """B-element reads after reuse along ``cells`` cells; integer division requires complete sharing groups.

        B reads when each B value passes along a row of ``cells`` cells holding A values.

        Equation: ``macs // cells``."""
        return self.macs // cells


class MemoryRoundTripScenario(BaseScenario):
    """A tensor of ``tensor_bytes`` crossing to and from memory ``trips`` times at
    ``energy_per_byte`` (a layer boundary writes it and the next layer reads it back).

    Notes
    -----
    Equation: The calculation members below define ``energy``.

    Assumptions: trips counts charged transfers of tensor_bytes, with constant energy per transferred byte.

    Excludes: Compute energy, standby power, and transfer setup overhead.

    References
    ----------
    Derivation: A tensor of ``tensor_bytes`` crossing to and from memory ``trips`` times at ``energy_per_byte`` (a layer boundary writes it and the next layer reads it back). Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50Stage2MapInt8ReadOnce
    >>> value = case.energy
    >>> print(f"{value.m_as('microjoule'):.3g} microjoule")
    32.1 microjoule"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    tensor_bytes: Quantity
    energy_per_byte: Quantity
    trips: int = 2

    @property
    def energy(self) -> Quantity:
        """Transfer energy in microjoules, stated byte transfers times energy per byte.

        Equation: ``(trips * tensor_bytes * energy_per_byte).to('microjoule')``."""
        return (self.trips * self.tensor_bytes * self.energy_per_byte).to("microjoule")


class QuantizationGridScenario(BaseScenario):
    """Values stored on a ``bits``-bit integer grid over a clipping range.

    Affine (``symmetric=False``): ``s = (beta - alpha) / (2^b - 1)`` and a
    zero-point ``z = qmin + round(-alpha / s)``; codes run 0 to 2^b - 1, or
    -2^(b-1) to 2^(b-1) - 1 when ``signed``. Symmetric: ``s = max(|alpha|,
    |beta|) / (2^(b-1) - 1)``, ``z = 0``, codes +-(2^(b-1) - 1). ``alpha`` and
    ``beta`` default to the values' minimum and maximum (min-max calibration).
    Rounding is to nearest, ties to even.

    Notes
    -----
    Equation: The calculation members below define ``lo``, ``hi``, ``steps``, ``qmin``, ``qmax``, ``scale``, ``zero_point_exact``, ``zero_point``, ``codes``, ``readback``, ``errors``, ``max_rounding_error``.

    Assumptions: All input values share a finite clipping range and one integer grid; the scale is positive.

    Excludes: Learned per-channel scales and accuracy effects beyond reconstruction error.

    References
    ----------
    Derivation: Values stored on a ``bits``-bit integer grid over a clipping range. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Uint8RangeMinusOneToThree
    >>> value = case.max_rounding_error
    >>> print(f"{value:.3g}")
    0.00784"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    values: List[float] = []
    bits: int
    alpha: Optional[float] = None
    beta: Optional[float] = None
    symmetric: bool = False
    signed: bool = False

    @property
    def lo(self) -> float:
        """Lower clipping endpoint, explicit alpha or the minimum input value.

        Equation: ``min(values) if alpha is None else alpha``."""
        return min(self.values) if self.alpha is None else self.alpha

    @property
    def hi(self) -> float:
        """Upper clipping endpoint, explicit beta or the maximum input value.

        Equation: ``max(values) if beta is None else beta``."""
        return max(self.values) if self.beta is None else self.beta

    @property
    def steps(self) -> int:
        """Number of affine intervals, one fewer than the number of integer codes.

        Steps across the range of an affine grid, ``2^b - 1``.

        Calculation: ``mlsysim.physics.numerics.calc_int_levels``."""
        from ...physics.numerics import calc_int_levels

        return calc_int_levels(self.bits) - 1

    @property
    def qmin(self) -> int:
        """Minimum integer code under signed, unsigned, or symmetric mode.

        Calculation: ``mlsysim.physics.numerics.calc_int_max``."""
        from ...physics.numerics import calc_int_max

        if self.symmetric:
            return -calc_int_max(self.bits)
        return -(calc_int_max(self.bits) + 1) if self.signed else 0

    @property
    def qmax(self) -> int:
        """Maximum integer code under signed, unsigned, or symmetric mode.

        Calculation: ``mlsysim.physics.numerics.calc_int_max``."""
        from ...physics.numerics import calc_int_max

        return calc_int_max(self.bits) if (self.symmetric or self.signed) else self.steps

    @property
    def scale(self) -> float:
        """Value-space step per integer code, via affine or symmetric scale helpers.

        Calculation: ``mlsysim.physics.numerics.calc_affine_scale``, ``mlsysim.physics.numerics.calc_symmetric_scale``."""
        from ...physics.numerics import calc_affine_scale, calc_symmetric_scale

        if self.symmetric:
            return calc_symmetric_scale(max(abs(self.lo), abs(self.hi)), self.bits)
        return calc_affine_scale(self.lo, self.hi, self.bits)

    @property
    def zero_point_exact(self) -> float:
        """Unrounded zero-point displacement, or zero for a symmetric grid.

        ``-alpha / s``, before rounding (0 for a symmetric grid).

        Calculation: ``mlsysim.physics.numerics.calc_zero_point``."""
        from ...physics.numerics import calc_zero_point

        return 0.0 if self.symmetric else calc_zero_point(self.lo, self.scale)[0]

    @property
    def zero_point(self) -> int:
        """Integer code offset used to represent zero, via ``calc_zero_point``.

        Calculation: ``mlsysim.physics.numerics.calc_zero_point``."""
        from ...physics.numerics import calc_zero_point

        return 0 if self.symmetric else calc_zero_point(self.lo, self.scale, self.qmin)[1]

    @property
    def codes(self) -> List[int]:
        """Quantized integer codes, clipped to the chosen code range by ``calc_quantize``.

        Calculation: ``mlsysim.physics.numerics.calc_quantize``."""
        from ...physics.numerics import calc_quantize

        return [calc_quantize(x, self.scale, self.zero_point, self.qmin, self.qmax) for x in self.values]

    @property
    def readback(self) -> List[float]:
        """Reconstructed input-unit values from codes via ``calc_dequantize``.

        Calculation: ``mlsysim.physics.numerics.calc_dequantize``."""
        from ...physics.numerics import calc_dequantize

        return [calc_dequantize(q, self.scale, self.zero_point) for q in self.codes]

    @property
    def errors(self) -> List[float]:
        """Absolute reconstruction error per input, in the same units as the input values.

        Absolute error of each value after the round trip.

        Equation: ``[abs(r - x) for r, x in zip(readback, values)]``."""
        return [abs(r - x) for r, x in zip(self.readback, self.values)]

    @property
    def max_rounding_error(self) -> float:
        """Half a grid step in input units; excludes clipping error outside the supported range.

        Largest error inside the range: half a step.

        Equation: ``scale / 2``."""
        return self.scale / 2


class OperationEnergyComparisonScenario(BaseScenario):
    """A wide and a narrow format's energy per add, multiply, and multiply-accumulate, and one
    memory read against the narrow multiply-accumulate. The narrow MAC accumulates in a wider
    integer, so its add is ``accumulate_add``.

    Notes
    -----
    Equation: The calculation members below define ``wide_mac``, ``narrow_mac``, ``add_ratio``, ``mul_ratio``, ``mac_ratio``, ``read_over_narrow_mac``.

    Assumptions: Per-operation energies refer to compatible operations; one MAC includes the stated multiply and accumulation add.

    Excludes: Data reuse, clock gating, and whole-device energy.

    References
    ----------
    Derivation: A wide and a narrow format's energy per add, multiply, and multiply-accumulate, and one memory read against the narrow multiply-accumulate. The narrow MAC accumulates in a wider integer, so its add is ``accumulate_add``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Int8VsFp32OperationEnergy45nm
    >>> value = case.mac_ratio
    >>> print(f"{value:.3g}")
    15.3"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    wide_add: Quantity
    wide_mul: Quantity
    narrow_add: Quantity
    narrow_mul: Quantity
    accumulate_add: Quantity
    memory_read: Quantity

    @property
    def wide_mac(self) -> Quantity:
        """Wide-format multiply plus add energy per arithmetic operation pair.

        Equation: ``wide_mul + wide_add``."""
        return self.wide_mul + self.wide_add

    @property
    def narrow_mac(self) -> Quantity:
        """Narrow multiply plus the stated accumulation-add energy.

        Equation: ``narrow_mul + accumulate_add``."""
        return self.narrow_mul + self.accumulate_add

    @property
    def add_ratio(self) -> float:
        """Dimensionless wide-to-narrow add-energy ratio.

        Equation: ``(wide_add / narrow_add).to('').magnitude``."""
        return (self.wide_add / self.narrow_add).to("").magnitude

    @property
    def mul_ratio(self) -> float:
        """Dimensionless wide-to-narrow multiply-energy ratio.

        Equation: ``(wide_mul / narrow_mul).to('').magnitude``."""
        return (self.wide_mul / self.narrow_mul).to("").magnitude

    @property
    def mac_ratio(self) -> float:
        """Dimensionless wide-to-narrow multiply-accumulate energy ratio.

        Equation: ``(wide_mac / narrow_mac).to('').magnitude``."""
        return (self.wide_mac / self.narrow_mac).to("").magnitude

    @property
    def read_over_narrow_mac(self) -> float:
        """Number of narrow MAC energy budgets equivalent to one memory read under the stated per-operation convention.

        The memory read in narrow multiply-accumulates (per-operation energies, one op each).

        Equation: ``memory_read.to('pJ').magnitude / narrow_mac.to('pJ/flop').magnitude``."""
        return self.memory_read.to("pJ").magnitude / self.narrow_mac.to("pJ/flop").magnitude


class QuantizedDeploymentProfileScenario(BaseScenario):
    """A float model and its INT8 version on one accelerator, judged against acceptance limits.

    Accuracy and confidence-calibration error are stated inputs (percent); latency and
    throughput come from ``profile`` (an engine profile with ``tensorrt(precision, batch)``
    and ``large_batch_throughput(precision)``); energy per image assumes full board power.

    Notes
    -----
    Equation: The calculation members below define ``top1_drop``, ``float_latency``, ``int_latency``, ``latency_speedup``, ``float_throughput``, ``int_throughput``, ``throughput_gain``, ``float_size``, ``int_size``, ``size_reduction``, ``float_energy``, ``int_energy``, ``energy_improvement``, ``passes_top1``, ``passes_calibration``.

    Assumptions: Latency/throughput are supplied by the selected profile; weight storage scales with bytes per parameter and energy uses TDP.

    Excludes: A fresh hardware measurement, KV/workspace memory, and measured per-request power.

    References
    ----------
    Derivation: A float model and its INT8 version on one accelerator, judged against acceptance limits. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50Int8ProfileH100
    >>> value = case.latency_speedup
    >>> print(f"{value:.3g}")
    1.25"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    profile: Any
    float_bytes: Quantity
    int_bytes: Quantity
    float_top1_pct: float
    int_top1_pct: float
    float_calibration_error_pct: float
    int_calibration_error_pct: float
    top1_limit_pct: float
    calibration_error_limit_pct: float

    @property
    def top1_drop(self) -> float:
        """Float-minus-INT8 top-1 accuracy in percentage points.

        Equation: ``float_top1_pct - int_top1_pct``."""
        return self.float_top1_pct - self.int_top1_pct

    @property
    def float_latency(self) -> Quantity:
        """Profiled FP32 TensorRT batch-one latency; uses the profile's latency quantity.

        Equation: ``profile.tensorrt('fp32', batch=1)``."""
        return self.profile.tensorrt("fp32", batch=1)

    @property
    def int_latency(self) -> Quantity:
        """Profiled INT8 TensorRT batch-one latency; uses the profile's latency quantity.

        Equation: ``profile.tensorrt('int8', batch=1)``."""
        return self.profile.tensorrt("int8", batch=1)

    @property
    def latency_speedup(self) -> float:
        """Dimensionless profiled FP32-to-INT8 batch-one latency ratio.

        Equation: ``(float_latency / int_latency).to('').magnitude``."""
        return (self.float_latency / self.int_latency).to("").magnitude

    @property
    def float_throughput(self) -> float:
        """Profile's FP32 large-batch throughput in requests per second.

        Equation: ``profile.large_batch_throughput('fp32')``."""
        return self.profile.large_batch_throughput("fp32")

    @property
    def int_throughput(self) -> float:
        """Profile's INT8 large-batch throughput in requests per second.

        Equation: ``profile.large_batch_throughput('int8')``."""
        return self.profile.large_batch_throughput("int8")

    @property
    def throughput_gain(self) -> float:
        """Dimensionless INT8-to-FP32 large-batch throughput ratio.

        Equation: ``int_throughput / float_throughput``."""
        return self.int_throughput / self.float_throughput

    @property
    def float_size(self) -> Quantity:
        """Float weight footprint from ``Workload.size_in_bytes`` at the stated bytes per parameter.

        Equation: ``workload.size_in_bytes(float_bytes)``."""
        return self.workload.size_in_bytes(self.float_bytes)

    @property
    def int_size(self) -> Quantity:
        """Quantized weight footprint from ``Workload.size_in_bytes`` at the stated bytes per parameter.

        Equation: ``workload.size_in_bytes(int_bytes)``."""
        return self.workload.size_in_bytes(self.int_bytes)

    @property
    def size_reduction(self) -> float:
        """Dimensionless float-to-quantized weight-memory ratio.

        Equation: ``(float_size / int_size).to('').magnitude``."""
        return (self.float_size / self.int_size).to("").magnitude

    @property
    def float_energy(self) -> Quantity:
        """TDP-divided-by-throughput estimate in mJ per FP32 request, not measured energy.

        Equation: ``(hardware.tdp / (float_throughput / Q_(1, 'second'))).to('mJ')``."""
        return (self.hardware.tdp / (self.float_throughput / Q_(1, "second"))).to("mJ")

    @property
    def int_energy(self) -> Quantity:
        """TDP-divided-by-throughput estimate in mJ per INT8 request, not measured energy.

        Equation: ``(hardware.tdp / (int_throughput / Q_(1, 'second'))).to('mJ')``."""
        return (self.hardware.tdp / (self.int_throughput / Q_(1, "second"))).to("mJ")

    @property
    def energy_improvement(self) -> float:
        """Dimensionless FP32-to-INT8 estimated energy ratio.

        Equation: ``(float_energy / int_energy).to('').magnitude``."""
        return (self.float_energy / self.int_energy).to("").magnitude

    @property
    def passes_top1(self) -> bool:
        """Whether INT8 top-1 accuracy meets the preset percentage threshold.

        Equation: ``int_top1_pct >= top1_limit_pct``."""
        return self.int_top1_pct >= self.top1_limit_pct

    @property
    def passes_calibration(self) -> bool:
        """Whether INT8 calibration error stays below the preset percentage limit.

        Equation: ``int_calibration_error_pct <= calibration_error_limit_pct``."""
        return self.int_calibration_error_pct <= self.calibration_error_limit_pct
