"""Inference work, request scheduling, and token-cache capacity.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
import math as _math
from typing import Any, Dict, List, Optional
from ...core.types import Quantity
from ...core.units import Q_
from ...hardware.types import HardwareNode
from ...models.types import Workload
from ...datasets.types import DatasetProfile
from .base import BaseScenario


class DecodeScenario(BaseScenario):
    """One decode step of ``workload`` on ``hardware`` (weight-streaming bound).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.DecodeBoundModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.DecodeBoundModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.DecodeBoundModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.DecodeBoundModel``. The methods below identify any
    additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama3_70B_DecodeH100
    >>> type(case.result).__name__
    'DecodeBoundResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    precision: str = "fp16"
    batch_size: int = 1
    # Optional (added 2026-10-20): the precision whose peak rate prices the
    # compute side, when it differs from the weight precision (8-bit weights
    # on an FP8 tensor-core peak). None uses hardware.compute.peak_flops.
    peak_precision: Optional[str] = None
    # Optional (added 2026-10-20): tensor-parallel width; each device streams
    # and multiplies only its 1/tp_size shard of the weights.
    tp_size: int = 1

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.DecodeBoundModel.solve``. Equation:
        pass the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import DecodeBoundModel

        peak = self.hardware.compute.precision_flops[self.peak_precision] if self.peak_precision else None
        return DecodeBoundModel().solve(
            self.workload, self.hardware, self.precision, self.batch_size, peak_flops=peak, tp_size=self.tp_size
        )


class ServingScenario(BaseScenario):
    """Serving ``workload`` on ``hardware`` at a stated context and batch (``ServingModel``).

    Notes
    -----
    Equation: Compose ``mlsysim.solvers.ServingModel``.

    Assumptions: The stated fields are inputs to ``mlsysim.solvers.ServingModel``; its
    documented validity regime, units, and modeling choices apply unchanged.

    Excludes: This scenario adds no measurement, empirical calibration, or mechanisms beyond
    ``mlsysim.solvers.ServingModel``.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.solvers.ServingModel``. The methods below identify any additional
    arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama3_70B_ServingH100
    >>> type(case.result).__name__
    'ServingResult'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    seq_len: int
    batch_size: int = 1

    @property
    def result(self):
        """Evaluate this scenario through ``mlsysim.solvers.ServingModel.solve``. Equation: pass
        the stated fields to that implementation; its contract defines the calculation and
        feasibility tests. Units: each returned field follows the referenced result schema; no
        independent equation is evaluated by this wrapper.
        """
        from ..solvers import ServingModel

        return ServingModel().solve(self.workload, self.hardware, seq_len=self.seq_len, batch_size=self.batch_size)


class QueueWaitScenario(BaseScenario):
    """A fleet modeled as one M/G/1 server at offered load ``utilization``.

    The mean queue wait, as a multiple of the mean job, is a load factor
    ``rho/(1-rho)`` times a variability factor ``(1 + C_s^2)/2``. The model
    ignores gang packing and parallel admission; it isolates the effect of
    load and of spread in job durations.

    Notes
    -----
    Equation: Compose ``mlsysim.physics.serving.calc_mg1_wait_ratio``.

    Assumptions: A stable FCFS M/G/1 queue has Poisson arrivals and a supplied service-time
    coefficient of variation.

    Excludes: Multiple-server scheduling, non-Poisson arrivals, and transient overload are
    absent.

    References
    ----------
    Derivation: this scenario binds its input fields to the referenced implementation
    contracts: ``mlsysim.physics.serving.calc_mg1_wait_ratio``. The methods below identify
    any additional arithmetic.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Mg1MixedJobQueueAt80
    >>> round(case.load_factor, 4)
    4.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    utilization: float
    service_cv: float  # coefficient of variation of job duration

    @property
    def load_factor(self) -> float:
        """Utilization contribution to the M/G/1 waiting-time ratio. Equation: ``utilization / (1 -
        utilization)``. Units: dimensionless; requires utilization below one for a stable queue.
        """
        return self.utilization / (1 - self.utilization)

    @property
    def variability_factor(self) -> float:
        """Service-time variability contribution to M/G/1 mean waiting. Equation: ``(1 + service_cv
        ** 2) / 2``. Units: dimensionless; arrivals are Poisson in this model.
        """
        return (1 + self.service_cv**2) / 2

    @property
    def wait_ratio(self) -> float:
        """Mean queue wait divided by mean service duration. Equation:
        ``mlsysim.physics.serving.calc_mg1_wait_ratio(utilization, service_cv)``. Units:
        dimensionless; FCFS single-server assumptions apply.
        """
        from ...physics.serving import calc_mg1_wait_ratio

        return calc_mg1_wait_ratio(self.utilization, self.service_cv)


class SchedulingSnapshotScenario(BaseScenario):
    """One snapshot of a shared pool's in-service capacity, read at the scheduling boundary.

    Capacity assigned to jobs includes gangs still waiting for members; only
    complete allocations can run a synchronous step, so scheduling goodput is
    the assigned share less the incomplete-gang share. Capacity in complete
    allocations waiting for input crosses this boundary and is a runtime loss.

    Notes
    -----
    Equation: scheduling goodput = assigned share - incomplete-gang share; executing share
    subtracts input waits.

    Assumptions: Shares use offered capacity as the same denominator and incomplete gangs
    cannot perform a synchronous step.

    Excludes: A time-integrated scheduler trace, queueing predictions, and removal of
    runtime loss by a scheduling-only change.

    References
    ----------
    Derivation: scheduling goodput = assigned share -
    incomplete-gang share; executing share subtracts input waits. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.SharedPoolSchedulingSnapshot
    >>> round(float(case.scheduling_goodput), 3)
    0.65"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    assigned_share: float
    incomplete_gang_share: float
    input_wait_share: float

    @property
    def scheduling_goodput(self) -> float:
        """Fraction of offered devices in complete allocations, assigned share minus
        incomplete gangs.

        Equation: ``return assigned_share - incomplete_gang_share``."""
        return self.assigned_share - self.incomplete_gang_share

    @property
    def scheduling_loss(self) -> float:
        """Fraction of offered capacity outside complete allocations.

        Equation: ``return 1 - scheduling_goodput``."""
        return 1 - self.scheduling_goodput

    @property
    def unassigned_share(self) -> float:
        """Fraction of offered devices assigned to no job.

        Equation: ``return 1 - assigned_share``."""
        return 1 - self.assigned_share

    @property
    def executing_share(self) -> float:
        """Fraction of offered capacity executing after removing input waits from complete
        allocations.

        Equation: ``return scheduling_goodput - input_wait_share``."""
        return self.scheduling_goodput - self.input_wait_share

    @property
    def input_wait_of_held(self) -> float:
        """Input-wait fraction normalized by complete-allocation held capacity.

        Equation: ``return input_wait_share / scheduling_goodput``."""
        return self.input_wait_share / self.scheduling_goodput

    @property
    def partial_hold_ceiling(self) -> float:
        """Scheduling-goodput ceiling if incomplete gangs become complete allocations without
        other changes.

        Equation: ``return scheduling_goodput + incomplete_gang_share``."""
        return self.scheduling_goodput + self.incomplete_gang_share


class RuntimeGoodputScenario(BaseScenario):
    """Runtime goodput of a job on ``failures.n_devices`` devices over a held ``window``.

    ``failures`` is the job's series failure model (SeriesFailureScenario);
    the checkpoint is ``workload`` at ``bytes_per_param``; ``recovery`` is the
    four-term budget profile (Systems.Reliability.Recovery).

    Notes
    -----
    Equation: runtime result = RuntimeGoodputModel.solve(fleet MTBF, checkpoint bytes,
    recovery profile, waits, and preemption inputs).

    Assumptions: Checkpoint bytes use the supplied per-parameter convention. Device-count
    sweeps hold all other inputs fixed. Crossover search needs a positive step and a
    reachable crossing.

    Excludes: Searching arbitrary layouts, convergence changes, and a termination
    guarantee when no crossover exists.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.RuntimeGoodputModel.solve``. The solver contract records the
    equations and their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_70BRuntimeGoodput128
    >>> round(float(case.checkpoint_bytes.to('B').magnitude), 3)
    980000000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    failures: Any  # SeriesFailureScenario
    workload: Workload
    bytes_per_param: Quantity
    recovery: Any
    input_wait: float  # share of held time
    window: Quantity  # held time per launch
    lost_work: Quantity  # work since the last save, per forced preemption
    preemption_rate: Quantity  # forced preemptions per unit time

    @property
    def checkpoint_bytes(self) -> Quantity:
        """Checkpoint bytes from ``physics.quantities.memory_from_params`` under the declared
        storage convention.

        Equation: ``return memory_from_params(workload.parameters,
        bytes_per_param).to('byte')``."""
        from ...physics.quantities import memory_from_params

        return memory_from_params(self.workload.parameters, self.bytes_per_param).to("byte")

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.RuntimeGoodputModel.solve`` using this
        scenario's declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to
        ``mlsysim.solvers.RuntimeGoodputModel.solve``; returned quantities and feasibility
        follow that contract."""
        from ..solvers import RuntimeGoodputModel

        return RuntimeGoodputModel().solve(
            self.failures.result.mtbf,
            self.checkpoint_bytes,
            self.recovery,
            self.input_wait,
            self.window,
            self.lost_work,
            self.preemption_rate,
        )

    def at(self, n_devices: int) -> "RuntimeGoodputScenario":
        """Copy the scenario for a new device count while preserving all other job and
        recovery inputs.

        Equation: ``return variant(failures=failures.variant(n_devices=n_devices))``."""
        return self.variant(failures=self.failures.variant(n_devices=n_devices))

    def sweep(self, sizes: List[int]) -> Dict[int, Any]:
        """Map each requested device count to its ``RuntimeGoodputModel.solve`` result.

        Equation: ``return {n: at(n).result for n in sizes}``."""
        return {n: self.at(n).result for n in sizes}

    def crossover(self, step: int) -> int:
        """First device count, in positive ``step`` increments, where failure-side waste
        exceeds input waits.

        Equation: Follow the crossover calculation below; branch conditions and rounding
        are part of its definition."""
        n = self.failures.n_devices
        while self.at(n).result.failure_side <= self.input_wait:
            n += step
        return n


class GoodputScenario(BaseScenario):
    """ML productivity goodput of a job on a pool: scheduling, runtime, and program goodput.

    ``ceilings`` gives, per factor ("S", "R", "P"), the value its owner's lever
    can reach; ``device_hour_rate`` is the price of one offered device-hour.

    Notes
    -----
    Equation: useful fraction = scheduling * runtime * program; trade ratio divides the
    two products.

    Assumptions: The factors describe successive boundaries with compatible denominators,
    as defined by GoodputModel.

    Excludes: A universal performance metric independent of workload or a new measurement
    of the factors.

    References
    ----------
    Derivation: This configuration forwards its stored inputs to
    ``mlsysim.solvers.GoodputModel.solve``. The solver contract records the equations and
    their provenance.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_70BSharedPoolGoodput128
    >>> round(float(case.result.mpg), 3)
    0.282"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    scheduling: float
    runtime: float
    program: float
    ceilings: Dict[str, float] = {}
    device_hour_rate: Optional[Quantity] = None

    @property
    def result(self):
        """Modeled result from ``mlsysim.solvers.GoodputModel.solve`` using this scenario's
        declared inputs; see that method for units, limits, and feasibility.

        Equation: Delegates the declared fields to ``mlsysim.solvers.GoodputModel.solve``;
        returned quantities and feasibility follow that contract."""
        from ..solvers import GoodputModel

        return GoodputModel().solve(self.scheduling, self.runtime, self.program, self.ceilings, self.device_hour_rate)

    def trade_ratio(self, other: "GoodputScenario") -> float:
        """Other scenario's modeled useful-work fraction divided by this scenario's fraction.

        Equation: ``return other.result.mpg / result.mpg``."""
        return other.result.mpg / self.result.mpg


class FleetMfuScenario(BaseScenario):
    """Program goodput (MFU) of a training step on one node and on a fleet of ``fleet_devices``.

    Useful work per step is the model's training arithmetic for the tokens in
    the step (6 x parameters x tokens); MFU divides it by every device's peak
    over the measured step time. Their ratio is the scaling efficiency measured
    from the node.

    Notes
    -----
    Equation: MFU = useful training FLOPs / (device count * peak FLOP/s * step time);
    scaling ratio = fleet MFU / node MFU.

    Assumptions: Supplied step times are comparable and training work follows the
    parameter-dominated helper without full recomputation.

    Excludes: Predicting step latency, identifying causes of scaling loss, and correcting
    measurement uncertainty.

    References
    ----------
    Derivation: mFU = useful training FLOPs / (device count
    * peak FLOP/s * step time); scaling ratio = fleet MFU / node MFU. The individual
    properties state units and intermediate accounting; this class introduces no fitted
    coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Llama2_70BFleetMfu128
    >>> round(float(case.node_useful_work.to('flop').magnitude), 3)
    925680000000000.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    node_devices: int
    fleet_devices: int
    node_tokens: int  # tokens per node step
    fleet_tokens: int  # tokens per fleet step
    node_step_time: Quantity
    fleet_step_time: Quantity

    def _useful(self, tokens: int) -> Quantity:
        from ...physics.transformer import calc_transformer_training_flops

        return calc_transformer_training_flops(self.workload.parameters, tokens)

    @property
    def node_useful_work(self) -> Quantity:
        """Useful training FLOPs in one node step from the parameter-token training-work
        helper.

        Equation: ``return _useful(node_tokens)``."""
        return self._useful(self.node_tokens)

    @property
    def fleet_useful_work(self) -> Quantity:
        """Useful training FLOPs in one fleet step from the parameter-token training-work
        helper.

        Equation: ``return _useful(fleet_tokens)``."""
        return self._useful(self.fleet_tokens)

    @property
    def fleet_peak_work(self) -> Quantity:
        """Fleet peak FLOPs over the supplied fleet step duration, the MFU denominator.

        Equation: ``return fleet_devices * hardware.compute.peak_flops *
        fleet_step_time``."""
        return self.fleet_devices * self.hardware.compute.peak_flops * self.fleet_step_time

    @property
    def node_mfu(self) -> float:
        """Useful node-step FLOPs divided by all node devices' peak work over the observed
        step time.

        Equation: ``peak = node_devices * hardware.compute.peak_flops * node_step_time;
        return (node_useful_work / peak).to('').magnitude``."""
        peak = self.node_devices * self.hardware.compute.peak_flops * self.node_step_time
        return (self.node_useful_work / peak).to("").magnitude

    @property
    def fleet_mfu(self) -> float:
        """Useful fleet-step FLOPs divided by the fleet peak-work envelope.

        Equation: ``return (fleet_useful_work / fleet_peak_work).to('').magnitude``."""
        return (self.fleet_useful_work / self.fleet_peak_work).to("").magnitude

    @property
    def scaling_efficiency(self) -> float:
        """Fleet MFU divided by node MFU, a dimensionless per-device scaling ratio.

        Equation: ``return fleet_mfu / node_mfu``."""
        return self.fleet_mfu / self.node_mfu

    @property
    def extra_step_time(self) -> Quantity:
        """Fleet step duration minus node step duration, in milliseconds.

        Equation: ``return (fleet_step_time - node_step_time).to('ms')``."""
        return (self.fleet_step_time - self.node_step_time).to("ms")

    @property
    def extra_share(self) -> float:
        """Extra step duration divided by fleet step duration, a signed fractional
        difference.

        Equation: ``return (extra_step_time / fleet_step_time).to('').magnitude``."""
        return (self.extra_step_time / self.fleet_step_time).to("").magnitude


class CompilationPaybackScenario(BaseScenario):
    """A compiled decode step and the executions that repay its build.

    The eager step splits into ``shares`` (matrix multiplies, element-wise
    kernels, launch overhead); compilation removes ``removed[k]`` of each
    share. A build is repaid after ``build / saved_per_token`` tokens; shared
    by ``replicas`` it is repaid in a fraction of that time, and rebuilt by
    each it costs ``replicas x gpus_per_replica x build``.

    Notes
    -----
    Equation: remaining step = eager step * sum(component share * retained fraction);
    repayment tokens = build time / time saved per token.

    Assumptions: Component fractions describe the same eager workload, and build repayment
    requires a positive saving per token.

    Excludes: Compiler-specific prediction, overlap between removed components, and build-
    cache invalidation.

    References
    ----------
    Derivation: remaining step = eager step * sum(component
    share * retained fraction); repayment tokens = build time / time saved per token. The
    individual properties state units and intermediate accounting; this class introduces
    no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.Decode13BCompilationPayback
    >>> round(float(case.step_fraction), 3)
    0.635"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    token_rate: Quantity  # eager tokens per second
    step_time: Quantity  # eager time per token, as the example carries it
    shares: Dict[str, float]
    removed: Dict[str, float]
    builds: Dict[str, Quantity]
    replicas: int
    gpus_per_replica: int = 1
    replica_life: Optional[Quantity] = None  # held time of a short-lived replica

    @property
    def step_fraction(self) -> float:
        """Fraction of eager-step time remaining after removing each declared component's
        saving share.

        Equation: ``return sum((v * (1 - removed.get(k, 0)) for k, v in
        shares.items()))``."""
        return sum(v * (1 - self.removed.get(k, 0)) for k, v in self.shares.items())

    @property
    def compiled_step(self) -> Quantity:
        """Compiled token-step duration in milliseconds under the component-removal
        approximation.

        Equation: ``return (step_time * step_fraction).to('ms')``."""
        return (self.step_time * self.step_fraction).to("ms")

    @property
    def saved_per_token(self) -> Quantity:
        """Milliseconds saved per token relative to the eager step.

        Equation: ``return (step_time - compiled_step).to('ms')``."""
        return (self.step_time - self.compiled_step).to("ms")

    @property
    def compiled_rate(self) -> Quantity:
        """Tokens per second implied by the reciprocal compiled step time.

        Equation: ``return (1 / compiled_step).to('1/second')``."""
        return (1 / self.compiled_step).to("1/second")

    @property
    def speedup(self) -> float:
        """Compiled token rate divided by the supplied eager rate; these inputs must describe
        the same workload.

        Equation: ``return (compiled_rate / token_rate).to('').magnitude``."""
        return (self.compiled_rate / self.token_rate).to("").magnitude

    def tokens_to_repay(self, build: str) -> float:
        """Token count whose cumulative saved time equals the selected build duration.

        Equation: ``return (builds[build] / saved_per_token).to('').magnitude``."""
        return (self.builds[build] / self.saved_per_token).to("").magnitude

    def time_to_repay(self, build: str) -> Quantity:
        """Minutes of compiled generation needed to produce the repayment token count.

        Equation: ``return (tokens_to_repay(build) / compiled_rate).to('minute')``."""
        return (self.tokens_to_repay(build) / self.compiled_rate).to("minute")

    def shared_repay(self, build: str) -> Quantity:
        """Per-replica repayment seconds when the pool shares a single build cost.

        Equation: ``return (time_to_repay(build) / replicas).to('second')``."""
        return (self.time_to_repay(build) / self.replicas).to("second")

    def rollout_cost(self, build: str) -> Quantity:
        """Total build device-hours when each replica builds independently on its allocated
        devices.

        Equation: ``return (builds[build] * gpus_per_replica * replicas).to('hour')``."""
        return (self.builds[build] * self.gpus_per_replica * self.replicas).to("hour")

    def build_share(self, build: str) -> float:
        """Selected build duration divided by replica lifetime, a dimensionless startup
        fraction.

        Equation: ``return (builds[build] / replica_life).to('').magnitude``."""
        return (self.builds[build] / self.replica_life).to("").magnitude


class TraceVolumeScenario(BaseScenario):
    """A profiler trace of ``window`` on every one of ``devices`` devices.

    Notes
    -----
    Equation: aggregate trace bytes = device count * per-device trace bytes.

    Assumptions: Each device emits the supplied volume for the same observation window.

    Excludes: Trace compression, sampling changes, collection overhead, and retention
    beyond the stated window.

    References
    ----------
    Derivation: aggregate trace bytes = device count * per-
    device trace bytes. The individual properties state units and intermediate accounting;
    this class introduces no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.NsightTraceVolume1K
    >>> round(float(case.total.to('GB').magnitude), 3)
    512.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    per_device: Quantity  # trace bytes per device for one window
    devices: int
    window: Quantity

    @property
    def total(self) -> Quantity:
        """Aggregate trace storage in GB, per-device volume times device count.

        Equation: ``return (per_device * devices).to('GB')``."""
        return (self.per_device * self.devices).to("GB")


class ServingTaxScenario(BaseScenario):
    """A request's latency target split between model execution and everything outside it.

    Notes
    -----
    Equation: outside-model budget = latency target - model time; network time =
    sequential exchange count * exchange latency.

    Assumptions: Network exchanges are sequential and low/high exchange latencies are
    supplied bounds.

    Excludes: Queueing tails, overlap, serialization work, and enforcement of an end-to-
    end SLA.

    References
    ----------
    Derivation: outside-model budget = latency target -
    model time; network time = sequential exchange count * exchange latency. The
    individual properties state units and intermediate accounting; this class introduces
    no fitted coefficient.

    Examples
    --------
    The named scenario supplies illustrative inputs; the output is calculated.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.InteractiveRequestServingTax
    >>> round(float(case.outside_model.to('ms').magnitude), 3)
    30.0"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    latency_target: Quantity
    model_time: Quantity  # model execution on an idle replica
    exchanges: int  # sequential network exchanges on the request path
    exchange_low: Quantity
    exchange_high: Quantity

    @property
    def outside_model(self) -> Quantity:
        """Latency budget remaining outside model execution in milliseconds; negative means
        no feasible remainder.

        Equation: ``return (latency_target - model_time).to('ms')``."""
        return (self.latency_target - self.model_time).to("ms")

    @property
    def exchanges_low(self) -> Quantity:
        """Milliseconds consumed by all sequential exchanges at the low per-exchange latency.

        Equation: ``return (exchanges * exchange_low).to('ms')``."""
        return (self.exchanges * self.exchange_low).to("ms")

    @property
    def exchanges_high(self) -> Quantity:
        """Milliseconds consumed by all sequential exchanges at the high per-exchange
        latency.

        Equation: ``return (exchanges * exchange_high).to('ms')``."""
        return (self.exchanges * self.exchange_high).to("ms")


class BatchingKneeScenario(BaseScenario):
    """Batch service time S(B) = t_0 + B t_req: where throughput turns over.

    Notes
    -----
    Equation: S(B)=``fixed_time+B*per_request_time``; throughput=B/S(B).

    Assumptions: Positive affine service costs and back-to-back full batches.

    Excludes: Queueing, batch formation, and batch-dependent efficiency changes.

    References
    ----------
    Implementation references: ``mlsysim.physics.serving.calc_batch_knee``, ``mlsysim.physics.serving.calc_batch_throughput_share``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.VisionBatchingKnee
    >>> round(scenario.knee, 6)
    2.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    fixed_time: Quantity  # paid once per batch
    per_request_time: Quantity  # added per request
    batch_sizes: List[int] = []  # batch sizes at which the service is reported (added 2026-10-20)

    @property
    def knee(self) -> float:
        """Batch size where fixed and request-dependent service costs are equal; dimensionless ``fixed_time / per_request_time`` via ``calc_batch_knee``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics.serving import calc_batch_knee

        return calc_batch_knee(self.fixed_time, self.per_request_time)

    def throughput_share(self, batch_size) -> float:
        """Fraction of asymptotic throughput at positive ``batch_size``; ``B / (B + knee)`` via ``calc_batch_throughput_share``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics.serving import calc_batch_throughput_share

        return calc_batch_throughput_share(batch_size, self.fixed_time, self.per_request_time)

    # The affine service model as a function of batch size (added 2026-10-20).
    def service_time(self, batch_size) -> Quantity:
        """Time for one batch in ms: fixed overhead plus ``batch_size * per_request_time``.

        Equation: (self.fixed_time + batch_size * self.per_request_time).to('ms')
        """
        return (self.fixed_time + batch_size * self.per_request_time).to("ms")

    def throughput(self, batch_size) -> Quantity:
        """Saturated request rate in 1/s: ``batch_size / service_time(batch_size)``; excludes queueing and batch formation.

        Equation: (batch_size / self.service_time(batch_size)).to('1/second')
        """
        return (batch_size / self.service_time(batch_size)).to("1/second")

    def time_per_request(self, batch_size) -> Quantity:
        """Amortized service cost in ms per request: batch service time divided by positive batch size.

        Equation: (self.service_time(batch_size) / batch_size).to('ms')
        """
        return (self.service_time(batch_size) / batch_size).to("ms")

    def fixed_share(self, batch_size) -> float:
        """Dimensionless fixed-overhead share: ``fixed_time / service_time(batch_size)``.

        Equation: (self.fixed_time / self.service_time(batch_size)).to('').magnitude
        """
        return (self.fixed_time / self.service_time(batch_size)).to("").magnitude


class LittlesLawScenario(BaseScenario):
    """Requests in flight at ``arrival_rate``, and replicas sized from a batch service time.

    Notes
    -----
    Equation: N=``lambda*T``; replica capacity=B/S; provisioned replicas=``ceil(lambda/(u*capacity))``.

    Assumptions: Stationary mean rates and identical replica service times; replica properties require all optional sizing fields.

    Excludes: Burst distributions and tail-latency guarantees.

    References
    ----------
    Implementation references: ``mlsysim.physics.serving.calc_littles_law_occupancy``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.VisionServingLittlesLaw
    >>> round(scenario.in_flight, 6)
    100.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    arrival_rate: Quantity
    time_in_system: Quantity
    batch_size: Optional[int] = None
    batch_service_time: Optional[Quantity] = None
    utilization_target: Optional[float] = None

    @property
    def in_flight(self) -> float:
        """Mean outstanding request count from Little's law, ``arrival_rate * time_in_system``; delegates to ``calc_littles_law_occupancy``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics.serving import calc_littles_law_occupancy

        return calc_littles_law_occupancy(self.arrival_rate, self.time_in_system).to("").magnitude

    @property
    def replica_throughput(self) -> Quantity:
        """Saturated capacity of one replica in 1/s: ``batch_size / batch_service_time``; requires both optional fields.

        Equation: (self.batch_size / self.batch_service_time).to('1/second')
        """
        return (self.batch_size / self.batch_service_time).to("1/second")

    @property
    def throughput_floor(self) -> int:
        """Integer replica count at full utilization, rounding ``arrival_rate / replica_throughput`` upward.

        Equation: _math.ceil((self.arrival_rate / self.replica_throughput).to('').magnitude)
        """
        return _math.ceil((self.arrival_rate / self.replica_throughput).to("").magnitude)

    @property
    def replicas_needed(self) -> int:
        """Integer replica count allowing the declared utilization headroom; ceiling of demand divided by target-utilization capacity.

        Equation: _math.ceil((self.arrival_rate / (self.utilization_target * self.replica_throughput)).to('').magnitude)
        """
        return _math.ceil((self.arrival_rate / (self.utilization_target * self.replica_throughput)).to("").magnitude)

    @property
    def executing_positions(self) -> int:
        """Total batch slots in the provisioned replicas: ``replicas_needed * batch_size``; a count, not mean occupancy.

        Equation: self.replicas_needed * self.batch_size
        """
        return self.replicas_needed * self.batch_size

    @property
    def mean_executing(self) -> float:
        """Mean busy batch positions under the fixed service-time approximation: ``arrival_rate * batch_service_time``, dimensionless.

        Equation: (self.arrival_rate * self.batch_service_time).to('').magnitude
        """
        return (self.arrival_rate * self.batch_service_time).to("").magnitude

    @property
    def service_utilization(self) -> float:
        """Offered load divided by provisioned saturated capacity; dimensionless and distinct from the input utilization target.

        Equation: (self.arrival_rate / (self.replicas_needed * self.replica_throughput)).to('').magnitude
        """
        return (self.arrival_rate / (self.replicas_needed * self.replica_throughput)).to("").magnitude


class RequestZonesScenario(BaseScenario):
    """One serialized request laid out in zones, the leading ones an invariant shared prefix.

    Notes
    -----
    Equation: Total tokens=sum(zones); prefix tokens=sum(first shared_zones entries).

    Assumptions: Dictionary insertion order is serialization order and the leading prefix is identical across requests.

    Excludes: Tokenizer execution and checking that prefix contents actually match.

    References
    ----------
    Derivation: A serialized sequence is a concatenation of zones. Summing all zone lengths gives the request length; summing the leading subset gives the shareable prefix. The prefix fraction divides those two counts.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairRequestZones
    >>> scenario.shared_tokens
    14336
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    zones: Dict[str, int]  # zone name -> tokens, in serialization order
    shared_zones: int  # leading zones that form the invariant prefix

    @property
    def total_tokens(self) -> int:
        """Serialized request token count, summing every zone in insertion order.

        Equation: sum(self.zones.values())
        """
        return sum(self.zones.values())

    @property
    def shared_tokens(self) -> int:
        """Token count in the first ``shared_zones`` entries; only that contiguous leading prefix is shareable.

        Equation: sum(list(self.zones.values())[:self.shared_zones])
        """
        return sum(list(self.zones.values())[: self.shared_zones])

    @property
    def shared_share(self) -> float:
        """Dimensionless fraction of request tokens in the invariant prefix: ``shared_tokens / total_tokens``; needs a nonempty request.

        Equation: self.shared_tokens / self.total_tokens
        """
        return self.shared_tokens / self.total_tokens


class KvAdmissionScenario(BaseScenario):
    """A replica's KV budget against one context length (``KvAdmissionModel``).

    Notes
    -----
    Equation: Result = mlsysim.solvers.KvAdmissionModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by KvAdmissionModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by KvAdmissionModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.KvAdmissionModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama3_70BLongContextAdmissionDgxH100
    >>> scenario.result.head_dim
    128
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    accelerators: int
    hbm_per_accelerator: Quantity
    weight_bytes_per_param: Quantity
    reserve: Quantity
    context_tokens: int
    kv_bytes_per_elem: Quantity
    # Optional (added 2026-10-20): a concurrent request count for ``context_fit``.
    requests: Optional[int] = None

    @property
    def result(self):
        """Return ``mlsysim.solvers.KvAdmissionModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.KvAdmissionModel.solve (scenario fields).
        """
        from ..solvers import KvAdmissionModel

        return KvAdmissionModel().solve(
            self.workload,
            self.accelerators,
            self.hbm_per_accelerator,
            self.weight_bytes_per_param,
            self.reserve,
            self.context_tokens,
            self.kv_bytes_per_elem,
        )

    def context_fit(self, requests: Optional[int] = None) -> float:
        """Unrounded per-request context capacity in tokens: remaining KV bytes divided by positive request count and bytes per token. Uses the admission model's ``kv_budget`` and ``kv_per_token``; supply ``requests`` or the scenario default.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        r = self.result
        n = self.requests if requests is None else requests
        return (r.kv_budget / (n * r.kv_per_token)).to("").magnitude


class PrefixSharingScenario(BaseScenario):
    """KV cache for users who share one prompt prefix, stored per user or once.

    Notes
    -----
    Equation: Independent storage=``users*(prefix+unique)*KV_bytes_per_token``; shared storage=``(prefix+users*unique)*KV_bytes_per_token``.

    Assumptions: Uniform KV width and token storage, one physically shared prefix, positive user count.

    Excludes: Allocator fragmentation, copy-on-write suffix blocks, and prefix cache misses.

    References
    ----------
    Implementation references: ``mlsysim.physics.memory.calc_kv_cache_size``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ChatbotSharedSystemPrompt
    >>> round(scenario.capacity_ratio, 6)
    4.98008
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    prefix_tokens: int
    users: int
    unique_tokens_per_user: int
    kv_width: int  # key-value heads x head width cached per layer
    bytes_per_elem: Quantity

    @property
    def tokens_per_user(self) -> int:
        """Context tokens for one user: shared prefix plus that user's unique suffix.

        Equation: self.prefix_tokens + self.unique_tokens_per_user
        """
        return self.prefix_tokens + self.unique_tokens_per_user

    @property
    def tokens_unshared(self) -> int:
        """Total stored token positions without sharing: ``users * tokens_per_user``.

        Equation: self.tokens_per_user * self.users
        """
        return self.tokens_per_user * self.users

    @property
    def tokens_shared(self) -> int:
        """Total stored token positions with one prefix copy and one unique suffix per user.

        Equation: self.prefix_tokens + self.unique_tokens_per_user * self.users
        """
        return self.prefix_tokens + self.unique_tokens_per_user * self.users

    def _kv(self, tokens: int) -> Quantity:
        from ...physics.memory import calc_kv_cache_size

        return calc_kv_cache_size(self.workload.layers, 1, self.kv_width, tokens, 1, self.bytes_per_elem)

    @property
    def kv_unshared(self) -> Quantity:
        """KV storage in bytes for all independent contexts; ``_kv`` delegates to ``calc_kv_cache_size`` with the declared cache width.

        Equation: self._kv(self.tokens_unshared)
        """
        return self._kv(self.tokens_unshared)

    @property
    def kv_shared(self) -> Quantity:
        """KV storage in bytes for one prefix and all suffixes; uses the same ``calc_kv_cache_size`` accounting as ``kv_unshared``.

        Equation: self._kv(self.tokens_shared)
        """
        return self._kv(self.tokens_shared)

    @property
    def reduction(self) -> float:
        """Fraction of KV storage removed by prefix sharing: ``1 - kv_shared / kv_unshared``.

        Equation: 1 - (self.kv_shared / self.kv_unshared).to('').magnitude
        """
        return 1 - (self.kv_shared / self.kv_unshared).to("").magnitude

    @property
    def capacity_ratio(self) -> float:
        """Dimensionless storage ratio ``kv_unshared / kv_shared``; a capacity gain under identical requests, not measured throughput.

        Equation: (self.kv_unshared / self.kv_shared).to('').magnitude
        """
        return (self.kv_unshared / self.kv_shared).to("").magnitude

    @property
    def prefill_tokens_skipped(self) -> int:
        """Prefix tokens avoided after the first user has populated the cache: ``(users - 1) * prefix_tokens``.

        Equation: (self.users - 1) * self.prefix_tokens
        """
        return (self.users - 1) * self.prefix_tokens


class PrecisionDividendScenario(BaseScenario):
    """Maximum batch before and after lower-precision weights and KV (``PrecisionDividendModel``).

    Notes
    -----
    Equation: Result = mlsysim.solvers.PrecisionDividendModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by PrecisionDividendModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by PrecisionDividendModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.PrecisionDividendModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama2_70BPrecisionDividendGqaTp8
    >>> round((scenario.result.weights_per_device).to('B').magnitude, 6)
    17500000000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tp_size: int
    hbm_per_accelerator: Quantity
    context_tokens: int
    kv_heads: int
    weight_bytes: Quantity
    weight_bytes_low: Quantity
    kv_bytes: Quantity
    kv_bytes_low: Quantity

    @property
    def result(self):
        """Return ``mlsysim.solvers.PrecisionDividendModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.PrecisionDividendModel.solve (scenario fields).
        """
        from ..solvers import PrecisionDividendModel

        return PrecisionDividendModel().solve(
            self.workload,
            self.tp_size,
            self.hbm_per_accelerator,
            self.context_tokens,
            self.kv_heads,
            self.weight_bytes,
            self.weight_bytes_low,
            self.kv_bytes,
            self.kv_bytes_low,
        )


class KvAttentionDesignScenario(BaseScenario):
    """Bytes per token across registry models, with one MHA, one GQA, and one MLA reference.

    Notes
    -----
    Equation: Result = mlsysim.solvers.KvAttentionDesignModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by KvAttentionDesignModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by KvAttentionDesignModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.KvAttentionDesignModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.KvBytesPerTokenByAttentionDesign
    >>> round(scenario.result.mha_over_mla, 6)
    67.147541
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    models: List[Workload]  # display order
    mha: Workload
    gqa: Workload
    mla: Workload
    precision: str = "fp16"
    session_tokens: int  # one long session, for the GQA and MLA references

    @property
    def result(self):
        """Return ``mlsysim.solvers.KvAttentionDesignModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.KvAttentionDesignModel.solve (scenario fields).
        """
        from ..solvers import KvAttentionDesignModel

        return KvAttentionDesignModel().solve(
            self.models, self.mha, self.gqa, self.mla, self.precision, self.session_tokens
        )


class PagedAllocationScenario(BaseScenario):
    """Requests under a length cap in ``pool``'s KV pool: full-cap reservation against paged blocks.

    Notes
    -----
    Equation: Result = mlsysim.solvers.PagedAllocationModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by PagedAllocationModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by PagedAllocationModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.PagedAllocationModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.CappedRequestsPagedAllocation
    >>> round((scenario.result.reservation).to('B').magnitude, 6)
    10737418240
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: KvAdmissionScenario
    cap: int  # admission cap, tokens
    mean_final: int  # mean final length, tokens
    small_block: int  # tokens per block
    large_block: int

    @property
    def result(self):
        """Return ``mlsysim.solvers.PagedAllocationModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.PagedAllocationModel.solve (scenario fields).
        """
        from ..solvers import PagedAllocationModel

        p = self.pool.result
        return PagedAllocationModel().solve(
            p.kv_per_token, p.kv_budget, self.cap, self.mean_final, self.small_block, self.large_block
        )


class KvForkCopyScenario(BaseScenario):
    """Candidates forked from one prefix, each copying the prefix's KV state without block sharing.

    Notes
    -----
    Equation: Copied bytes=``branches*prefix_tokens*pool.result.kv_per_token``.

    Assumptions: Every candidate receives a separate full prefix copy.

    Excludes: Copy-on-write blocks, transfer time, and generated suffix storage.

    References
    ----------
    Derivation: Each independent branch stores prefix_tokens positions at the admission model's bytes per position. Multiplying by the number of branches counts the copied state without any sharing.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ForkedCandidatesPrefixCopy
    >>> round((scenario.copied).to('B').magnitude, 6)
    10737418240
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: KvAdmissionScenario
    prefix_tokens: int
    branches: int

    @property
    def copied(self) -> Quantity:
        """KV bytes copied when each branch receives the complete prefix: ``branches * prefix_tokens * pool.result.kv_per_token``; excludes block sharing.

        Equation: self.branches * self.prefix_tokens * self.pool.result.kv_per_token
        """
        return self.branches * self.prefix_tokens * self.pool.result.kv_per_token


class PrefixRebuildScenario(BaseScenario):
    """A cached prefix on ``node``: the prefill a miss rebuilds against one short suffix pass on a hit.

    Notes
    -----
    Equation: Result = mlsysim.solvers.PrefixRebuildModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by PrefixRebuildModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by PrefixRebuildModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.PrefixRebuildModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.InvariantPrefixRebuildDgxH100
    >>> round(scenario.result.n_params, 6)
    70600000000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    node: Any
    precision: str = "fp16"
    prefix_tokens: int
    suffix_tokens: int
    utilization: float  # sustained share of the node's peak rate

    @property
    def result(self):
        """Return ``mlsysim.solvers.PrefixRebuildModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.PrefixRebuildModel.solve (scenario fields).
        """
        from ..solvers import PrefixRebuildModel

        return PrefixRebuildModel().solve(
            self.workload, self.node, self.precision, self.prefix_tokens, self.suffix_tokens, self.utilization
        )


class ChunkedPrefillScenario(BaseScenario):
    """A long prompt advanced ``chunk_tokens`` per iteration beside decode.

    Notes
    -----
    Equation: Iterations=ceil(prompt_tokens/chunk_tokens); the optional result composes PrefixRebuildModel with ChunkedPrefillModel.

    Assumptions: Positive chunk size; prefill and decodes must be supplied to request the solver result.

    Excludes: A runtime scheduler or a measured gap distribution.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.ChunkedPrefillModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.LongPromptChunkedPrefill
    >>> scenario.iterations
    20
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    # Same name and first fields as the v2-10 inference branch's type (prompt_tokens,
    # chunk_tokens, iterations); the decode-gap inputs below are optional additions.
    prompt_tokens: int
    chunk_tokens: int
    prefill: Optional[PrefixRebuildScenario] = None  # model, node, and sustained rate
    decodes: Optional[int] = None  # ongoing decode requests

    @property
    def iterations(self) -> int:
        """Integer iterations required to consume the prompt: ceiling of ``prompt_tokens / chunk_tokens``; chunk size must be positive.

        Equation: _math.ceil(self.prompt_tokens / self.chunk_tokens)
        """
        return _math.ceil(self.prompt_tokens / self.chunk_tokens)

    @property
    def result(self):
        """Return ``mlsysim.solvers.ChunkedPrefillModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.ChunkedPrefillModel.solve (scenario fields).
        """
        from ..solvers import ChunkedPrefillModel

        p = self.prefill.result
        return ChunkedPrefillModel().solve(
            self.prefill.workload.parameters,
            p.sustained_rate,
            p.weight_read_time,
            self.decodes,
            self.prompt_tokens,
            self.chunk_tokens,
        )


class TensorParallelSpeedupScenario(BaseScenario):
    """Stated per-layer compute split ``tp_size`` ways, plus the AllReduces each layer adds.

    Notes
    -----
    Equation: T_parallel_layer=``sum(T_serial_phase)/tp_size+collectives_per_layer*T_allreduce``.

    Assumptions: Ideal equal compute partition and identical layers; all collective costs serialize with compute.

    Excludes: Communication overlap, load imbalance, and capacity checks.

    References
    ----------
    Derivation: The ideal partition assumption divides each serial compute phase by the tensor-parallel degree. Collectives add serial delays, so their count times per-collective time is added to the compute sum.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama2_70BPrefillTensorParallel8
    >>> round(scenario.speedup, 6)
    6.896552
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    serial_layer_times: Dict[str, Quantity]  # compute per layer on one device, by phase
    tp_size: int
    allreduce_time: Quantity  # per collective
    collectives_per_layer: int = 2
    layers: int

    def sharded(self, phase: str) -> Quantity:
        """Compute time in ms for a named phase after ideal division by positive ``tp_size``; excludes that phase's communication.

        Equation: (self.serial_layer_times[phase] / self.tp_size).to('ms')
        """
        return (self.serial_layer_times[phase] / self.tp_size).to("ms")

    @property
    def serial_layer_time(self) -> Quantity:
        """Total compute time in ms across the declared serial phases of one layer.

        Equation: sum(self.serial_layer_times.values(), Q_(0, 'ms')).to('ms')
        """
        return sum(self.serial_layer_times.values(), Q_(0, "ms")).to("ms")

    @property
    def sharded_compute_time(self) -> Quantity:
        """Ideal layer compute time in ms: serial phase sum divided by ``tp_size``.

        Equation: (self.serial_layer_time / self.tp_size).to('ms')
        """
        return (self.serial_layer_time / self.tp_size).to("ms")

    @property
    def parallel_layer_time(self) -> Quantity:
        """Time in ms per parallel layer: ideally sharded compute plus every declared AllReduce, with no overlap.

        Equation: (sum((self.sharded(p) for p in self.serial_layer_times), Q_(0, 'ms')) + self.collectives_per_layer * self.allreduce_time).to('ms')
        """
        return (
            sum((self.sharded(p) for p in self.serial_layer_times), Q_(0, "ms"))
            + self.collectives_per_layer * self.allreduce_time
        ).to("ms")

    @property
    def speedup(self) -> float:
        """Dimensionless layer speedup: ``serial_layer_time / parallel_layer_time``; communication can make it less than one.

        Equation: (self.serial_layer_time / self.parallel_layer_time).to('').magnitude
        """
        return (self.serial_layer_time / self.parallel_layer_time).to("").magnitude

    @property
    def serial_total(self) -> Quantity:
        """Compute time in ms across ``layers`` identical serial layers.

        Equation: (self.serial_layer_time * self.layers).to('ms')
        """
        return (self.serial_layer_time * self.layers).to("ms")

    @property
    def parallel_total(self) -> Quantity:
        """Total modeled time in ms across ``layers`` identical compute-plus-collective layers.

        Equation: (self.parallel_layer_time * self.layers).to('ms')
        """
        return (self.parallel_layer_time * self.layers).to("ms")


class DecodeBudgetScenario(BaseScenario):
    """A constructed decode-step budget, cut interval by interval, against a TPOT target.

    Notes
    -----
    Equation: Baseline intervals=``baseline_step*shares``; precision and fusion replace declared portions; weight floor=weight_shard/HBM_bandwidth.

    Assumptions: Constructed ordered intervals: GEMM, attention, elementwise, AllReduce, launch; supplied cuts are scenario assumptions.

    Excludes: Predicting the cuts from kernels, allocator/workspace costs, and calibrated execution time.

    References
    ----------
    Derivation: This is a constructed interval budget: supplied fractional cuts remove portions of named baseline intervals, while supplied absolute costs replace other intervals. The independent weight-read floor follows bytes divided by peak bandwidth; it does not validate those cuts.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama3_70B_ReplicaDecodeBudget
    >>> round((scenario.floor).to('ms').magnitude, 6)
    5.223881
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tp_size: int
    accelerator: HardwareNode
    weight_bytes_per_param: Quantity
    tpot_target: Quantity
    baseline_step: Quantity
    intervals: List[str]  # GEMM, attention, element-wise, AllReduce, launch
    baseline_shares: List[float]
    gemm_cut: float  # after precision: share of GEMM time removed
    attention_cut: float  # after fusion: share of attention time removed
    fused_elementwise: Quantity
    replay_overhead: Quantity
    collective_launch: Quantity
    allreduces_per_layer: int = 2

    @property
    def baseline(self) -> List[float]:
        """Constructed per-interval times in ms, multiplying ``baseline_step`` by each declared share; these are supplied budget allocations, not measurements.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        total = self.baseline_step.to("ms").magnitude
        return [total * s for s in self.baseline_shares]

    @property
    def after_precision(self) -> List[float]:
        """Per-interval times in ms after removing ``gemm_cut`` of the first baseline interval; other intervals remain unchanged.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        r = self.baseline.copy()
        r[0] *= 1 - self.gemm_cut
        return r

    @property
    def after_fusion(self) -> List[float]:
        """Per-interval times in ms after the precision change, attention reduction, and replacement elementwise/replay costs; follows the fixed interval ordering.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        r = self.after_precision.copy()
        r[1] *= 1 - self.attention_cut
        r[2] = self.fused_elementwise.to("ms").magnitude
        r[4] = self.replay_overhead.to("ms").magnitude
        return r

    @property
    def totals(self) -> List[float]:
        """Total step times in ms for baseline, precision, and fusion budgets, respectively, by summing their intervals.

        Equation: [sum(self.baseline), sum(self.after_precision), sum(self.after_fusion)]
        """
        return [sum(self.baseline), sum(self.after_precision), sum(self.after_fusion)]

    @property
    def weight_shard(self) -> Quantity:
        """Per-accelerator weight storage in decimal GB: checkpoint size divided evenly by ``tp_size``.

        Equation: (self.workload.size_in_bytes(self.weight_bytes_per_param) / self.tp_size).to('GB')
        """
        return (self.workload.size_in_bytes(self.weight_bytes_per_param) / self.tp_size).to("GB")

    @property
    def floor(self) -> Quantity:
        """Weight-streaming lower bound in ms: one weight shard divided by peak HBM bandwidth; excludes KV reads, compute, and collectives.

        Equation: (self.weight_shard / self.accelerator.memory.bandwidth).to('ms')
        """
        return (self.weight_shard / self.accelerator.memory.bandwidth).to("ms")

    @property
    def allreduces_per_step(self) -> int:
        """Number of AllReduce calls per token step: ``allreduces_per_layer * workload.layers``.

        Equation: self.allreduces_per_layer * self.workload.layers
        """
        return self.allreduces_per_layer * self.workload.layers

    @property
    def allreduce_per_collective(self) -> Quantity:
        """Average constructed collective time in microseconds: baseline collective interval divided by calls per step.

        Equation: Q_(self.baseline[3] / self.allreduces_per_step, 'ms').to('microsecond')
        """
        return Q_(self.baseline[3] / self.allreduces_per_step, "ms").to("microsecond")

    @property
    def collective_estimate(self) -> Quantity:
        """Launch-plus-transfer estimate in microseconds for two FP16 hidden-width vectors on NVLink; uses peak one-direction bandwidth and excludes topology contention.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...core.units import BYTES_FP16

        payload = 2 * self.workload.hidden_dim * BYTES_FP16
        return (self.collective_launch + payload / self.accelerator.nvlink.bandwidth_per_direction).to("microsecond")


class SpeculativeRoundScenario(BaseScenario):
    """A draft proposes ``draft_length`` tokens; the target verifies them in one step.

    Notes
    -----
    Equation: Time per token=(draft_cost+verify_time)/expected_tokens; expected_tokens is the geometric acceptance sum including a bonus token.

    Assumptions: Constant independent acceptance probability and nonoverlapping draft/verify phases; provide total draft time or per-step draft time.

    Excludes: Changing context lengths, model residency, and measuring acceptance quality.

    References
    ----------
    Implementation references: ``mlsysim.physics.serving.calc_speculative_expected_tokens``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama3_70B_ReplicaSpeculation
    >>> round(scenario.expected_tokens, 6)
    3.52182
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    acceptance: float
    draft_length: int
    draft_time: Optional[Quantity] = None  # all draft steps of one round
    verify_time: Quantity
    # Optional (added 2026-10-20): one draft step; with no ``draft_time`` the
    # round's draft cost is ``draft_length`` such steps.
    draft_step: Optional[Quantity] = None

    @property
    def draft_cost(self) -> Quantity:
        """Total draft time in ms; use explicit ``draft_time`` when supplied, otherwise ``draft_length * draft_step``.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        if self.draft_time is not None:
            return self.draft_time.to("ms")
        return (self.draft_length * self.draft_step).to("ms")

    @property
    def expected_tokens(self) -> float:
        """Expected emitted token count per round, including the bonus token, via ``calc_speculative_expected_tokens`` and its constant-acceptance assumption.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        from ...physics.serving import calc_speculative_expected_tokens

        return calc_speculative_expected_tokens(self.acceptance, self.draft_length)

    @property
    def round_time(self) -> Quantity:
        """Serial draft-plus-verification time in ms; assumes the two phases do not overlap.

        Equation: (self.draft_cost + self.verify_time).to('ms')
        """
        return (self.draft_cost + self.verify_time).to("ms")

    @property
    def per_token_time(self) -> Quantity:
        """Amortized ms per emitted token: ``round_time / expected_tokens``.

        Equation: (self.round_time / self.expected_tokens).to('ms')
        """
        return (self.round_time / self.expected_tokens).to("ms")

    @property
    def tpot_gain(self) -> float:
        """Dimensionless ratio of the declared verification time to amortized speculative token time; depends on interpreting that verification input as the baseline step cost.

        Equation: (self.verify_time / self.per_token_time).to('').magnitude
        """
        return (self.verify_time / self.per_token_time).to("").magnitude


class WeightedRoutingScenario(BaseScenario):
    """Traffic split over server pools in proportion to each pool's per-server capacity.

    Notes
    -----
    Equation: Fleet capacity=``sum(servers*per_server_capacity)``; per-server route weight=capacity/fleet_capacity.

    Assumptions: Offered requests split proportionally to constant per-server capacities.

    Excludes: Queueing, routing overhead, and heterogeneous request costs.

    References
    ----------
    Derivation: A pool contributes server count times per-server capacity. Dividing one server's capacity by the fleet sum assigns a per-server routing probability; offered rate times that probability gives its mean load.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.HeterogeneousH100A100Routing
    >>> round(scenario.total_capacity, 6)
    22000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    servers: Dict[str, int]
    capacity: Dict[str, float]  # requests per second one server sustains
    offered: float  # requests per second offered to the fleet

    @property
    def total_capacity(self) -> float:
        """Fleet request capacity in requests/s, summing each pool's server count times per-server capacity.

        Equation: sum((self.servers[k] * self.capacity[k] for k in self.servers))
        """
        return sum(self.servers[k] * self.capacity[k] for k in self.servers)

    def weight(self, pool: str) -> float:
        """Dimensionless traffic fraction for one server in ``pool``: its capacity divided by fleet capacity; multiply by pool size for a pool-wide share.

        Equation: self.capacity[pool] / self.total_capacity
        """
        return self.capacity[pool] / self.total_capacity

    def utilization(self, pool: str) -> float:
        """Dimensionless per-server load under capacity-weighted routing: offered rate times per-server weight divided by that server's capacity.

        Equation: self.offered * self.weight(pool) / self.capacity[pool]
        """
        return self.offered * self.weight(pool) / self.capacity[pool]

    @property
    def equal_split_load(self) -> float:
        """Requests/s sent to each server under a uniform split, independent of its capacity.

        Equation: self.offered / sum(self.servers.values())
        """
        return self.offered / sum(self.servers.values())

    def equal_split_utilization(self, pool: str) -> float:
        """Dimensionless utilization of a server in ``pool`` under equal traffic per server.

        Equation: self.equal_split_load / self.capacity[pool]
        """
        return self.equal_split_load / self.capacity[pool]


class TailEventRateScenario(BaseScenario):
    """How often an event of probability ``event_probability`` per request occurs at ``request_rate``.

    Notes
    -----
    Equation: Expected events per second=``request_rate*event_probability``.

    Assumptions: A constant mean request rate and a probability applicable to each request.

    Excludes: Event clustering, confidence intervals, and severity of an event.

    References
    ----------
    Derivation: Linearity of expectation gives the mean number of qualifying requests as request count times the per-request event probability. Applying the same product to the request rate gives events per second without assuming independent event timing.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RecommenderTailEvents
    >>> round(scenario.events_per_second, 6)
    1000.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    request_rate: Quantity
    event_probability: float

    @property
    def events_per_second(self) -> float:
        """Expected event frequency in 1/s: request arrival rate times event probability; this is a mean, not a tail-latency bound.

        Equation: (self.request_rate * self.event_probability).to('1/second').magnitude
        """
        return (self.request_rate * self.event_probability).to("1/second").magnitude


class ColdStartScenario(BaseScenario):
    """A new replica: provision, fetch the weight artifact, load and warm up.

    Notes
    -----
    Equation: Start time=provisioning_time+artifact_bytes/fetch_bandwidth+load_warmup_time.

    Assumptions: Sequential startup phases and constant available transfer bandwidth; peak replica counts are supplied.

    Excludes: Overlap, network contention, and forecasting the predictive fleet size.

    References
    ----------
    Derivation: Nonoverlapping startup phases add in elapsed time. The fetch phase is artifact bytes divided by available bandwidth. Replica saving is an arithmetic comparison of two supplied peak counts, not a predictive scaling model.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama70BReplicaColdStart
    >>> round((scenario.cold_start_time).to('s').magnitude, 6)
    400.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    weight_bytes_per_param: Quantity
    fetch_bandwidth: Quantity
    provisioning_time: Quantity
    load_warmup_time: Quantity
    reactive_peak_replicas: int  # peak fleet when scaling only reacts
    predictive_peak_replicas: int  # peak fleet when replicas start ahead of the ramp
    prewarm_lead: Quantity

    @property
    def artifact(self) -> Quantity:
        """Model weight artifact size in decimal GB via ``workload.size_in_bytes`` at the declared storage width.

        Equation: self.workload.size_in_bytes(self.weight_bytes_per_param).to('GB')
        """
        return self.workload.size_in_bytes(self.weight_bytes_per_param).to("GB")

    @property
    def fetch_time(self) -> Quantity:
        """Ideal transfer time in seconds: artifact bytes divided by available fetch bandwidth.

        Equation: (self.artifact / self.fetch_bandwidth).to('second')
        """
        return (self.artifact / self.fetch_bandwidth).to("second")

    @property
    def cold_start_time(self) -> Quantity:
        """Serial start delay in seconds: provisioning plus artifact fetch plus load/warmup; no overlap is modeled.

        Equation: (self.provisioning_time + self.fetch_time + self.load_warmup_time).to('second')
        """
        return (self.provisioning_time + self.fetch_time + self.load_warmup_time).to("second")

    @property
    def peak_saving(self) -> float:
        """Fraction of reactive peak replicas avoided by the supplied predictive peak count; the counts are inputs, not forecasts made here.

        Equation: 1 - self.predictive_peak_replicas / self.reactive_peak_replicas
        """
        return 1 - self.predictive_peak_replicas / self.reactive_peak_replicas


class NoisyNeighborScenario(BaseScenario):
    """Tenants sharing a KV pool and a replica, one of them bursting, with and without quotas.

    Notes
    -----
    Equation: Other tenant share=(pool-swollen_share)/(tenants-1); sequences=truncate(other_share/baseline_bytes_per_sequence).

    Assumptions: Equal allocation among nonbursting tenants and a fixed per-sequence KV footprint.

    Excludes: Computing p99 from arrival distributions; the scenario carries p99 values as inputs.

    References
    ----------
    Derivation: After one tenant takes its supplied allocation, the remaining pool is divided equally among the other tenants. Dividing each remaining allocation by baseline bytes per sequence gives its whole-sequence capacity.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.SharedPoolNoisyNeighbor
    >>> round(scenario.batch_reduction, 6)
    0.625
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    kv_pool: Quantity
    pool_tenants: int
    swollen_share: Quantity  # long-context tenant's allocation without limits
    base_sequences: int  # sequences per equal share
    base_rate: float  # requests per second per tenant
    burst_rate: float
    slo: Quantity  # p99 target
    unprotected_p99_low: Quantity
    unprotected_p99_high: Quantity
    quota_rate: float
    protected_p99: Quantity

    @property
    def base_share(self) -> Quantity:
        """Equal-share KV allocation per tenant, preserving the pool's byte units.

        Equation: self.kv_pool / self.pool_tenants
        """
        return self.kv_pool / self.pool_tenants

    @property
    def other_share(self) -> Quantity:
        """KV bytes per remaining tenant after the bursting tenant takes ``swollen_share``; requires more than one tenant.

        Equation: (self.kv_pool - self.swollen_share) / (self.pool_tenants - 1)
        """
        return (self.kv_pool - self.swollen_share) / (self.pool_tenants - 1)

    @property
    def other_sequences(self) -> int:
        """Whole sequences each remaining tenant can retain, truncating its allocation divided by the baseline bytes per sequence.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        per_seq = self.base_share / self.base_sequences
        return int((self.other_share / per_seq).to("").magnitude)

    @property
    def batch_reduction(self) -> float:
        """Fractional loss of sequence capacity relative to ``base_sequences``.

        Equation: 1 - self.other_sequences / self.base_sequences
        """
        return 1 - self.other_sequences / self.base_sequences

    @property
    def burst_multiple(self) -> float:
        """Dimensionless offered-load multiplier ``burst_rate / base_rate``; base rate must be positive.

        Equation: self.burst_rate / self.base_rate
        """
        return self.burst_rate / self.base_rate


class ReasoningLatencyScenario(BaseScenario):
    """Serial reasoning tokens before the answer, at a fixed time per token.

    Notes
    -----
    Equation: Reasoning duration=``reasoning_tokens*per_token_time``.

    Assumptions: Serial decoding at a constant supplied token time.

    Excludes: Prefill, final answer tokens, and any prediction of reasoning quality.

    References
    ----------
    Derivation: A serial sequence of identical-duration token steps takes token count times duration per token. The expansion value uses one token as its reference unit.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.ReasoningTokenLatency
    >>> round((scenario.reasoning_time).to('s').magnitude, 6)
    12.8
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    per_token_time: Quantity
    reasoning_tokens: int

    @property
    def reasoning_time(self) -> Quantity:
        """Serial reasoning latency in seconds: fixed time per token times token count; excludes prefill and answer generation.

        Equation: (self.per_token_time * self.reasoning_tokens).to('second')
        """
        return (self.per_token_time * self.reasoning_tokens).to("second")

    @property
    def expansion(self) -> int:
        """Reasoning token count, interpreted as expansion relative to a one-token reference; dimensionless.

        Equation: self.reasoning_tokens
        """
        return self.reasoning_tokens


class FixedBatchScheduleScenario(BaseScenario):
    """Requests in one batch of fixed membership, one token per iteration.

    Notes
    -----
    Equation: Slot work=``batch_width*max(output_lengths)``; continuous scheduling assigns each waiting request to the earliest free slot.

    Assumptions: One token per iteration, fixed initial slot count, and arrival-ordered waiting requests; input lists are aligned and nonempty.

    Excludes: Prefill costs and variable token times; the request-level comparison starts the entire waiting wave together without a second-wave width check.

    References
    ----------
    Implementation references: ``mlsysim.physics.serving.calc_padding_waste_fraction``.
    These helper contracts state the source assumptions; the scenario binds
    their inputs and derives the displayed ratios and budgets.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.FourRequestDecodeBatch
    >>> round(scenario.idle_share, 6)
    0.375
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    output_lengths: List[int]
    prompt_lengths: Optional[List[int]] = None
    iteration_time: Optional[Quantity] = None
    # Optional (added 2026-10-20): requests waiting from the start, admitted in
    # order, for the request-level against continuous schedules below.
    waiting_lengths: Optional[List[int]] = None

    @property
    def total_tokens(self) -> List[int]:
        """Prompt-plus-output token count for each paired request; ``zip`` requires aligned input lists to retain every request.

        Equation: [p + o for p, o in zip(self.prompt_lengths, self.output_lengths)]
        """
        return [p + o for p, o in zip(self.prompt_lengths, self.output_lengths)]

    @property
    def max_output(self) -> int:
        """Longest initial output in tokens, which sets the fixed batch's completion iteration.

        Equation: max(self.output_lengths)
        """
        return max(self.output_lengths)

    @property
    def mean_output(self) -> float:
        """Mean initial output length in tokens; requires a nonempty batch.

        Equation: sum(self.output_lengths) / len(self.output_lengths)
        """
        return sum(self.output_lengths) / len(self.output_lengths)

    @property
    def slot_iterations(self) -> int:
        """Available decode slot-iterations until the longest initial request ends: batch width times longest output.

        Equation: len(self.output_lengths) * self.max_output
        """
        return len(self.output_lengths) * self.max_output

    @property
    def useful_iterations(self) -> int:
        """Useful initial token iterations, summing actual output lengths.

        Equation: sum(self.output_lengths)
        """
        return sum(self.output_lengths)

    @property
    def idle_share(self) -> float:
        """Fraction of fixed-batch slot-iterations spent after a request completes; delegates to ``calc_padding_waste_fraction``.

        Equation: The calculation follows the stated schedule construction rule; the method body shows each intermediate term.
        """
        from ...physics.serving import calc_padding_waste_fraction

        return calc_padding_waste_fraction(self.output_lengths)

    @property
    def requests_advanced(self) -> int:
        """Initial request count plus one replacement opportunity for each slot ending before the longest initial output; not a complete queue simulation.

        Equation: len(self.output_lengths) + sum((1 for x in self.output_lengths if x < self.max_output))
        """
        return len(self.output_lengths) + sum(1 for x in self.output_lengths if x < self.max_output)

    @property
    def all_lengths(self) -> List[int]:
        """Initial and waiting output lengths concatenated in arrival order, in token iterations.

        Equation: list(self.output_lengths) + list(self.waiting_lengths or [])
        """
        return list(self.output_lengths) + list(self.waiting_lengths or [])

    @property
    def all_useful_iterations(self) -> int:
        """Useful token iterations across both initial and waiting requests.

        Equation: sum(self.all_lengths)
        """
        return sum(self.all_lengths)

    def request_level_spans(self) -> List[tuple]:
        """Start/end iteration pairs with all waiting requests starting after the initial batch; this illustrative trace does not enforce a second-wave slot limit.

        Equation: The calculation follows the stated schedule construction rule; the method body shows each intermediate term.
        """
        first = [(0, n) for n in self.output_lengths]
        return first + [(self.max_output, self.max_output + n) for n in (self.waiting_lengths or [])]

    def continuous_spans(self) -> List[tuple]:
        """Start/end iteration pairs from greedy slot reuse: each waiting request enters the next free initial-batch slot.

        Equation: The calculation follows the stated schedule construction rule; the method body shows each intermediate term.
        """
        import heapq

        free = [(n, i) for i, n in enumerate(self.output_lengths)]
        heapq.heapify(free)
        spans = [(0, n) for n in self.output_lengths]
        for n in self.waiting_lengths or []:
            t, slot = heapq.heappop(free)
            spans.append((t, t + n))
            heapq.heappush(free, (t + n, slot))
        return spans

    @property
    def request_level_iterations(self) -> int:
        """Completion iteration of the illustrated request-level schedule, taking the latest span endpoint.

        Equation: max((end for _, end in self.request_level_spans()))
        """
        return max(end for _, end in self.request_level_spans())

    @property
    def continuous_iterations(self) -> int:
        """Completion iteration of the greedy continuous-batching schedule, taking its latest endpoint.

        Equation: max((end for _, end in self.continuous_spans()))
        """
        return max(end for _, end in self.continuous_spans())

    @property
    def request_level_utilization(self) -> float:
        """Useful token work divided by initial batch width times the illustrated request-level horizon; the trace's second-wave assumption can exceed that capacity.

        Equation: self.all_useful_iterations / (len(self.output_lengths) * self.request_level_iterations)
        """
        return self.all_useful_iterations / (len(self.output_lengths) * self.request_level_iterations)

    @property
    def continuous_utilization(self) -> float:
        """Fraction of available slot-iterations occupied by useful work under greedy slot reuse.

        Equation: self.all_useful_iterations / (len(self.output_lengths) * self.continuous_iterations)
        """
        return self.all_useful_iterations / (len(self.output_lengths) * self.continuous_iterations)


class PausedKvStateScenario(BaseScenario):
    """A request paused on a tool: keep, move over the host links, or drop and recompute its KV state.

    Notes
    -----
    Equation: Host-link bandwidth=``node.accelerators_per_node*accelerator.interconnect.bandwidth``; pause choices delegate to PausedKvStateModel.

    Assumptions: Independent host links and the cache geometry inherited from pool/prefill scenarios.

    Excludes: PCIe contention beyond the supplied link fraction and measured reload behavior.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.PausedKvStateModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairToolWaitPausedState
    >>> round((scenario.link).to('GB/s').magnitude, 6)
    512.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: KvAdmissionScenario
    prefill: PrefixRebuildScenario  # node, links, and sustained prefill rate
    context_tokens: int
    block_tokens: int
    wait: Quantity
    link_fraction: float  # share of rated link bandwidth under contention
    returns: int  # paused requests whose waits end together

    @property
    def link(self) -> Quantity:
        """Aggregate host-link bandwidth: accelerators per node times each accelerator's interconnect bandwidth; assumes independent links and excludes contention.

        Equation: The calculation follows the stated accounting rule; the method body shows each intermediate term.
        """
        node = self.prefill.node
        return node.accelerators_per_node * node.accelerator.interconnect.bandwidth

    @property
    def result(self):
        """Return ``mlsysim.solvers.PausedKvStateModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.PausedKvStateModel.solve (scenario fields).
        """
        from ..solvers import PausedKvStateModel

        return PausedKvStateModel().solve(
            self.pool.result.kv_per_token,
            self.prefill.workload.parameters,
            self.prefill.result.sustained_rate,
            self.link,
            self.context_tokens,
            self.block_tokens,
            self.wait,
            self.link_fraction,
            self.returns,
        )


class KvBlockSharingScenario(BaseScenario):
    """Requests with an output allowance in whole blocks, with and without a shared prefix, as a turn rate.

    Notes
    -----
    Equation: Result = mlsysim.solvers.KvBlockSharingModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by KvBlockSharingModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by KvBlockSharingModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.KvBlockSharingModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairBlockSharingResidency
    >>> round((scenario.result.block_bytes).to('B').magnitude, 6)
    5242880
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    pool: KvAdmissionScenario
    headroom: float  # fraction of the pool withheld
    context: int  # input tokens per request
    output: int  # output allowance per request
    common: int  # shareable prefix tokens
    block_tokens: int
    generation: Quantity  # residence while generating
    wait: Quantity  # tool wait with the state kept

    @property
    def result(self):
        """Return ``mlsysim.solvers.KvBlockSharingModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.KvBlockSharingModel.solve (scenario fields).
        """
        from ..solvers import KvBlockSharingModel

        p = self.pool.result
        return KvBlockSharingModel().solve(
            p.kv_per_token,
            p.kv_budget,
            self.headroom,
            self.context,
            self.output,
            self.common,
            self.block_tokens,
            self.generation,
            self.wait,
        )


class KvResidencyArchetypeScenario(BaseScenario):
    """One agent workload's sharing gain, residency tax, and reload against recompute.

    Notes
    -----
    Equation: Result = mlsysim.solvers.KvResidencyArchetypeModel.solve with the named scenario fields; no extra fitting or postprocessing is applied.

    Assumptions: The supplied inputs use the units and validity regime documented by KvResidencyArchetypeModel.solve; this class binds one reproducible configuration.

    Excludes: Any behavior excluded by KvResidencyArchetypeModel.solve, and measurement or calibration of the supplied inputs.

    References
    ----------
    Implementation reference: ``mlsysim.solvers.KvResidencyArchetypeModel.solve`` documents
    the composed equation and its sources.

    Examples
    --------
    The named registry scenario supplies the stated inputs; this is a modeled
    calculation, not a new measurement.

    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.RepairAgentResidency
    >>> round(scenario.result.shared_fraction, 6)
    0.1875
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    paused: PausedKvStateScenario  # pool, link, and sustained prefill rate
    context_tokens: int
    shared_prefix: int
    generation: Quantity
    wait: Quantity

    @property
    def result(self):
        """Return ``mlsysim.solvers.KvResidencyArchetypeModel.solve`` for this scenario; output fields retain the units and feasibility semantics specified by that solver.

        Equation: mlsysim.solvers.KvResidencyArchetypeModel.solve (scenario fields).
        """
        from ..solvers import KvResidencyArchetypeModel

        p = self.paused
        pool = p.pool.result
        return KvResidencyArchetypeModel().solve(
            pool.kv_per_token,
            pool.kv_budget,
            p.prefill.workload.parameters,
            p.prefill.result.sustained_rate,
            p.link,
            self.context_tokens,
            self.shared_prefix,
            self.generation,
            self.wait,
        )


def _labeled_capacity(hardware: HardwareNode) -> Quantity:
    """The capacity a vendor label prints: binary gigabytes read as "GB" (an
    "80 GB" A100 holds 80 GiB). Shares of a labeled device divide decimal bytes
    by this figure, as the chapters do."""
    return Q_(hardware.memory.capacity.to("GiB").magnitude, "GB")


class PackedQkvProjectionScenario(BaseScenario):
    """``tokens`` rows through ``workload``'s packed query-key-value projection, one d x 3d matrix.

    Notes
    -----
    Equation: Packed width is ``projections*d``; projection MACs are ``tokens*d*packed_width``.

    Assumptions: Equal-width dense projections, usually query/key/value, with no bias counting.

    Excludes: Grouped-query asymmetry, attention products, and output projection.

    References
    ----------
    Derivation: Concatenating equally wide projections produces a ``d-by-(projections*d)``
    matrix. Every token row performs one MAC for every matrix entry, giving
    ``tokens*d*packed_width``. Grouped-query projections require a different geometry.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Gpt2SmallSixTokenQkvProjection
    >>> s.macs == s.tokens*s.width*s.packed_width
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tokens: int
    projections: int = 3  # query, key, value

    @property
    def width(self) -> int:
        """Model hidden width used as the projection input dimension.

        Equation: ``workload.hidden_dim``."""
        return self.workload.hidden_dim

    @property
    def packed_width(self) -> int:
        """Combined output width of the configured query/key/value projections.

        Equation: ``projections * width``."""
        return self.projections * self.width

    @property
    def macs(self) -> int:
        """Multiply-accumulate count for all token rows through the packed projection.

        Equation: ``tokens * width * packed_width``."""
        return self.tokens * self.width * self.packed_width


class TransformerBlockScenario(BaseScenario):
    """``workload``'s decoder blocks at a context of ``context_tokens``: the classic block's weights
    (4 d^2 attention projections, 2 x ``ffn_multiple`` d^2 MLP), its dense and attention multiply-adds,
    and, with ``hardware``, the attention arithmetic of one prompt at peak.

    Notes
    -----
    Equation: Shared transformer helpers count dense block parameters and full attention products; two FLOPs are counted per MAC.

    Assumptions: Classic dense blocks at the declared FFN multiplier; exact_parameters specifically describes the GPT-2 bias-enabled layout.

    Excludes: Gated FFNs, grouped-query geometry, kernel fusion, and vocabulary-output work from prompt_flops.

    References
    ----------
    Derivation: Uses mlsysim.physics.transformer.calc_transformer_block_weights,
    calc_attention_score_macs, and calc_attention_crossover for the classic block
    geometry. Matrix MAC counts become FLOPs by counting a multiply and add separately.
    The exact_parameters property adds GPT-2-specific biases, normalization parameters,
    and position embeddings rather than treating the coarse block rule as exact.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Gpt2XlBlockFullWindow
    >>> s.block_macs == s.dense_macs+s.attention_macs
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    context_tokens: int
    ffn_multiple: int = 4  # feed-forward width over model width
    attention_projections: int = 4  # query, key, value, output
    hardware: Optional[HardwareNode] = None
    # Optional (added 2026-10-20): sequences of ``context_tokens`` in one training step.
    batch_size: int = 1

    def _w(self):
        from ...physics.transformer import calc_transformer_block_weights

        return calc_transformer_block_weights(self.width, self.ffn_multiple, self.attention_projections)

    @property
    def width(self) -> int:
        """Model hidden width shared by the classic decoder-block formulas.

        Equation: ``workload.hidden_dim``."""
        return self.workload.hidden_dim

    @property
    def attention_weights_per_block(self) -> int:
        """Attention projection parameter count returned by calc_transformer_block_weights.

        Equation: ``_w()[0]``."""
        return self._w()[0]

    @property
    def mlp_weights_per_block(self) -> int:
        """Feed-forward projection parameter count returned by calc_transformer_block_weights.

        Equation: ``_w()[1]``."""
        return self._w()[1]

    @property
    def weights_per_block(self) -> int:
        """Combined attention and feed-forward matrix parameter count for one decoder block.

        Equation: ``_w()[2]``."""
        return self._w()[2]

    @property
    def block_coefficient(self) -> int:
        """Weights per block in units of d^2 (12 for the classic block).

        Equation: ``weights_per_block // width ** 2``."""
        return self.weights_per_block // self.width**2

    @property
    def mlp_share(self) -> float:
        """Dimensionless feed-forward matrix share of one block's counted parameters.

        Equation: ``mlp_weights_per_block / weights_per_block``."""
        return self.mlp_weights_per_block / self.weights_per_block

    @property
    def block_weights(self) -> int:
        """All blocks' weights.

        Equation: ``weights_per_block * workload.layers``."""
        return self.weights_per_block * self.workload.layers

    @property
    def attention_weights(self) -> int:
        """Attention projection parameter count across all decoder blocks.

        Equation: ``attention_weights_per_block * workload.layers``."""
        return self.attention_weights_per_block * self.workload.layers

    @property
    def mlp_weights(self) -> int:
        """Feed-forward projection parameter count across all decoder blocks.

        Equation: ``mlp_weights_per_block * workload.layers``."""
        return self.mlp_weights_per_block * self.workload.layers

    @property
    def token_table(self) -> int:
        """Token-embedding weights: one row of width d per vocabulary entry.

        Equation: ``workload.vocab_size * width``."""
        return self.workload.vocab_size * self.width

    @property
    def block_rule_parameters(self) -> int:
        """Block matrix parameters plus one token embedding table; biases and position table omitted.

        Equation: ``block_weights + token_table``."""
        return self.block_weights + self.token_table

    @property
    def exact_parameters(self) -> int:
        """GPT-2 layout counted in full: projection and MLP biases, two LayerNorms per block,
        the final LayerNorm, and a position table of ``context_length`` rows (nanoGPT, bias=True).

        Equation: ``d, f = (width, ffn_multiple); per_block = d * 3 * d + 3 * d + (d * d +
        d) + (d * f * d + f * d) + (f * d * d + d) + 2 * (2 * d); workload.layers *
        per_block + workload.vocab_size * d + workload.context_length * d + 2 * d``."""
        d, f = self.width, self.ffn_multiple
        per_block = (d * 3 * d + 3 * d) + (d * d + d) + (d * f * d + f * d) + (f * d * d + d) + 2 * (2 * d)
        return (
            self.workload.layers * per_block + self.workload.vocab_size * d + self.workload.context_length * d + 2 * d
        )

    @property
    def flops_per_token(self) -> Quantity:
        """Two FLOPs (a multiply and an add) per block weight per token.

        Equation: ``Q_(2 * block_weights, 'flop')``."""
        return Q_(2 * self.block_weights, "flop")

    @property
    def output_layer_flops(self) -> Quantity:
        """The output projection onto the vocabulary, per token.

        Equation: ``Q_(2 * token_table, 'flop')``."""
        return Q_(2 * self.token_table, "flop")

    @property
    def dense_macs(self) -> int:
        """One block's projection and MLP multiply-adds over the context.

        Equation: ``weights_per_block * context_tokens``."""
        return self.weights_per_block * self.context_tokens

    @property
    def attention_macs(self) -> int:
        """One block's two attention products over the context.

        Equation: ``calc_attention_score_macs(context_tokens, width)``."""
        from ...physics.transformer import calc_attention_score_macs

        return calc_attention_score_macs(self.context_tokens, self.width)

    @property
    def block_macs(self) -> int:
        """Dense projection plus attention multiply-accumulates for one block over the prompt.

        Equation: ``dense_macs + attention_macs``."""
        return self.dense_macs + self.attention_macs

    @property
    def attention_share(self) -> float:
        """Dimensionless attention-product fraction of one block's prompt MACs.

        Equation: ``attention_macs / block_macs``."""
        return self.attention_macs / self.block_macs

    @property
    def dense_to_attention(self) -> float:
        """Dimensionless projection-and-MLP MAC count divided by attention-product MACs.

        Equation: ``dense_macs / attention_macs``."""
        return self.dense_macs / self.attention_macs

    @property
    def attention_to_dense(self) -> float:
        """Dimensionless attention-product MAC count divided by projection-and-MLP MACs.

        Equation: ``attention_macs / dense_macs``."""
        return self.attention_macs / self.dense_macs

    @property
    def crossover(self) -> float:
        """Context length at which the attention products equal the dense products (6 d).

        Equation: ``calc_attention_crossover(width, ffn_multiple, attention_projections)``."""
        from ...physics.transformer import calc_attention_crossover

        return calc_attention_crossover(self.width, self.ffn_multiple, self.attention_projections)

    @property
    def crossover_multiple(self) -> float:
        """The crossover in units of the width (six for the classic block).

        Equation: ``crossover / width``."""
        return self.crossover / self.width

    @property
    def prompt_dense_flops(self) -> Quantity:
        """Dense block work in FLOPs over all prompt tokens and layers.

        Equation: ``(flops_per_token * context_tokens).to('flop')``."""
        return (self.flops_per_token * self.context_tokens).to("flop")

    @property
    def prompt_attention_flops(self) -> Quantity:
        """Every block's attention products over the prompt, two FLOPs per multiply-add.

        Equation: ``Q_(2 * attention_macs * workload.layers, 'flop')``."""
        return Q_(2 * self.attention_macs * self.workload.layers, "flop")

    @property
    def prompt_flops(self) -> Quantity:
        """Prompt FLOPs combining dense block work and attention products, excluding vocabulary output work.

        Equation: ``prompt_dense_flops + prompt_attention_flops``."""
        return self.prompt_dense_flops + self.prompt_attention_flops

    @property
    def attention_time_at_peak(self) -> Quantity:
        """Ideal attention-only time in minutes at the selected device's peak compute rate.

        Equation: ``(prompt_attention_flops / hardware.compute.peak_flops).to('minute')``."""
        return (self.prompt_attention_flops / self.hardware.compute.peak_flops).to("minute")

    @property
    def training_to_forward(self) -> float:
        """A training step in forward passes: 6P over 2P per token.

        Equation: ``p = workload.parameters; (calc_transformer_training_flops(p, 1) /
        calc_transformer_decode_flops(p, 1)).to('').magnitude``."""
        from ...physics.transformer import calc_transformer_training_flops, calc_transformer_decode_flops

        p = self.workload.parameters
        return (calc_transformer_training_flops(p, 1) / calc_transformer_decode_flops(p, 1)).to("").magnitude

    @property
    def layer_forward_flops(self) -> Quantity:
        """One block's forward FLOPs for the whole batch: dense and attention products, two FLOPs each.

        Equation: ``Q_(2 * block_macs * batch_size, 'flop').to('GFLOP')``."""
        return Q_(2 * self.block_macs * self.batch_size, "flop").to("GFLOP")

    @property
    def training_step_flops(self) -> Quantity:
        """One training step counted layer by layer, attention products included.

        Equation: ``(layer_forward_flops * workload.layers *
        training_to_forward).to('TFLOP')``."""
        return (self.layer_forward_flops * self.workload.layers * self.training_to_forward).to("TFLOP")

    @property
    def six_p_step_flops(self) -> Quantity:
        """The same step by the 6P-per-token rule, which omits the attention products.

        Equation: ``calc_transformer_training_flops(workload.parameters, batch_size *
        context_tokens).to('TFLOP')``."""
        from ...physics.transformer import calc_transformer_training_flops

        return calc_transformer_training_flops(self.workload.parameters, self.batch_size * self.context_tokens).to(
            "TFLOP"
        )

    @property
    def attention_step_flops(self) -> Quantity:
        """The attention products' share of a training step, all layers.

        Equation: ``Q_(2 * attention_macs * batch_size * workload.layers *
        training_to_forward, 'flop').to('TFLOP')``."""
        return Q_(
            2 * self.attention_macs * self.batch_size * self.workload.layers * self.training_to_forward, "flop"
        ).to("TFLOP")


class MaterializedScoresScenario(BaseScenario):
    """Attention scores stored in full: ``seq_len``^2 per head, ``heads`` per layer, ``layers`` layers.

    Notes
    -----
    Equation: A full score tensor has ``seq_len**2*heads`` values per layer and multiplies by layers for total storage.

    Assumptions: Every head materializes every query-key score at the declared width.

    Excludes: FlashAttention-style streaming, sparsity, gradients, and other activation storage.

    References
    ----------
    Derivation: Every query token has a score for every key token, forming seq_len
    squared entries per head. Multiplication by heads, layers, and bytes_per_score
    counts the fully materialized storage without assuming an attention implementation
    streams or discards those entries.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.TwelveHeadScoresAt100kTokens
    >>> s.total_bytes == s.layer_bytes*s.layers
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    seq_len: int
    heads: int
    layers: int
    bytes_per_score: Quantity
    hardware: Optional[HardwareNode] = None

    @property
    def scores_per_layer(self) -> int:
        """Stored attention-score count for one layer, retaining the complete square matrix for each head.

        Equation: ``seq_len ** 2 * heads``."""
        return self.seq_len**2 * self.heads

    @property
    def layer_bytes(self) -> Quantity:
        """One layer's full attention-score storage in decimal GB.

        Equation: ``(scores_per_layer * bytes_per_score).to('GB')``."""
        return (self.scores_per_layer * self.bytes_per_score).to("GB")

    @property
    def total_bytes(self) -> Quantity:
        """All layers' full attention-score storage in decimal GB.

        Equation: ``(layer_bytes * layers).to('GB')``."""
        return (self.layer_bytes * self.layers).to("GB")

    @property
    def layer_device_multiple(self) -> float:
        """One layer's scores in units of ``hardware``'s labeled capacity.

        Equation: ``(layer_bytes / _labeled_capacity(hardware)).to('').magnitude``."""
        return (self.layer_bytes / _labeled_capacity(self.hardware)).to("").magnitude


class ContextGrowthScenario(BaseScenario):
    """A context window grown from ``base_tokens`` to ``tokens``: a kept cache grows with the ratio,
    attention's score work and stored scores with its square.

    Notes
    -----
    Equation: Linear cache growth is tokens/base_tokens; full score growth is its square.

    Assumptions: Model geometry and storage precision remain fixed; base_tokens is positive.

    Excludes: Windowed/sparse attention, cache eviction, and measured latency scaling.

    References
    ----------
    Derivation: Retained KV state adds a fixed amount per token, so its ratio follows
    the context-length ratio. A full query-key matrix has two context axes, so its
    work/storage ratio is the square of that growth factor.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.BertWindowToFourK
    >>> s.score_ratio == s.ratio**2
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    base_tokens: int
    tokens: int

    @property
    def ratio(self) -> float:
        """Dimensionless context-length growth factor; also the linear cache-growth factor.

        Equation: ``tokens / base_tokens``."""
        return self.tokens / self.base_tokens

    @property
    def score_ratio(self) -> float:
        """Dimensionless full attention-score growth factor, the square of context growth.

        Equation: ``ratio ** 2``."""
        return self.ratio**2


class KvFootprintScenario(BaseScenario):
    """``workload``'s KV cache for ``tokens`` kept tokens; with ``hardware`` and ``kv_share``, how many
    such requests the cache's share of the labeled device memory holds. ``kv_width_factor`` scales the
    key-value width (one half: half the key-value heads).

    Notes
    -----
    Equation: The workload cache-size contract gives bytes per token; token count and KV-width factor scale the footprint.

    Assumptions: One request per cache estimate, fixed precision, and an explicit fractional reservation when using capacity methods.

    Excludes: Runtime reserve, batch scheduler limits, and automatic reconciliation of the reservation with weight residency.

    References
    ----------
    Derivation: The workload method get_kv_cache_size supplies the cache geometry and
    storage convention. The wrapper scales its one-token result by retained tokens and
    an explicit width multiplier; dividing the reserved device-memory budget by this
    footprint gives continuous token or request capacity.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Gpt2XlKvFullWindow
    >>> s.total.to("GB").magnitude > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    tokens: int
    bytes_per_value: Quantity
    hardware: Optional[HardwareNode] = None
    kv_share: Optional[float] = None
    kv_width_factor: float = 1.0

    @property
    def per_token(self) -> Quantity:
        """Per-request cache bytes per kept token, with the explicit KV-width multiplier applied.

        Equation: ``(workload.get_kv_cache_size(1, 1, bytes_per_value) *
        kv_width_factor).to('byte')``."""
        return (self.workload.get_kv_cache_size(1, 1, self.bytes_per_value) * self.kv_width_factor).to("byte")

    @property
    def total(self) -> Quantity:
        """Per-request KV cache in decimal GB for the configured retained token count.

        Equation: ``(per_token * tokens).to('GB')``."""
        return (self.per_token * self.tokens).to("GB")

    @property
    def weights(self) -> Quantity:
        """Model weight footprint in decimal GB at the chosen storage width.

        Equation: ``workload.size_in_bytes(bytes_per_value).to('GB')``."""
        return self.workload.size_in_bytes(self.bytes_per_value).to("GB")

    @property
    def kv_budget(self) -> Quantity:
        """Decimal GB reserved for KV cache as the configured share of labeled device capacity.

        Equation: ``(_labeled_capacity(hardware) * kv_share).to('GB')``."""
        return (_labeled_capacity(self.hardware) * self.kv_share).to("GB")

    @property
    def tokens_in_budget(self) -> float:
        """Continuous cache-token capacity in the reserved budget.

        Equation: ``(kv_budget / per_token).to('').magnitude``."""
        return (self.kv_budget / self.per_token).to("").magnitude

    @property
    def requests(self) -> float:
        """Continuous number of full-context requests fitting the reserved KV budget.

        Equation: ``(kv_budget / total).to('').magnitude``."""
        return (self.kv_budget / self.total).to("").magnitude


class ContextMemoryCrossingScenario(BaseScenario):
    """Three memories of ``workload`` as context grows on ``hardware``: fixed weights, a KV cache linear
    in tokens, and one layer's stored scores (all heads) quadratic in tokens.

    Notes
    -----
    Equation: Weights are fixed, full-request KV is linear in context, and one layer of scores grows quadratically.

    Assumptions: All query heads materialize scores and the workload supplies its KV geometry.

    Excludes: Other layers of scores in score_crossing, workspaces, and a combined residency feasibility test.

    References
    ----------
    Derivation: Equate the constant weight footprint with linear KV storage to find
    their crossing. Equate device capacity with the quadratic score footprint for a
    score-only capacity crossing. Equating linear KV with quadratic scores and canceling
    one positive context factor gives the KV-score crossing.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.Gpt2XlContextMemoryOnH100
    >>> s.kv_crossing > 0
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    bytes_per_value: Quantity

    @property
    def weights(self) -> Quantity:
        """Model weight footprint in decimal GB at the chosen storage width.

        Equation: ``workload.size_in_bytes(bytes_per_value).to('GB')``."""
        return self.workload.size_in_bytes(self.bytes_per_value).to("GB")

    @property
    def kv_per_token(self) -> Quantity:
        """Per-request KV-cache bytes added by one retained token.

        Equation: ``workload.get_kv_cache_size(1, 1, bytes_per_value).to('byte')``."""
        return self.workload.get_kv_cache_size(1, 1, self.bytes_per_value).to("byte")

    @property
    def score_bytes_per_pair(self) -> Quantity:
        """One layer's stored scores for one token pair, all heads.

        Equation: ``(workload.heads * bytes_per_value).to('byte')``."""
        return (self.workload.heads * self.bytes_per_value).to("byte")

    @property
    def kv_crossing(self) -> float:
        """Tokens at which the KV cache equals the weights.

        Equation: ``(weights / kv_per_token).to('').magnitude``."""
        return (self.weights / self.kv_per_token).to("").magnitude

    @property
    def score_crossing(self) -> float:
        """Tokens at which one layer's stored scores fill the device.

        Equation: ``_math.sqrt((hardware.memory.capacity /
        score_bytes_per_pair).to('').magnitude)``."""
        return _math.sqrt((self.hardware.memory.capacity / self.score_bytes_per_pair).to("").magnitude)

    @property
    def kv_score_crossing(self) -> float:
        """Tokens at which the KV cache and one layer's scores are equal.

        Equation: ``(kv_per_token / score_bytes_per_pair).to('').magnitude``."""
        return (self.kv_per_token / self.score_bytes_per_pair).to("").magnitude


class EmbeddingTableScenario(BaseScenario):
    """``tables`` embedding tables of ``rows`` rows, each ``width`` values wide, against ``hardware``.

    Notes
    -----
    Equation: Row bytes are ``width*value_width``; total storage multiplies by rows and table count.

    Assumptions: Fully allocated equal-shaped tables and labeled memory capacity.

    Excludes: Index metadata, replicas, allocator reserve, and cached or compressed tables.

    References
    ----------
    Derivation: A dense embedding table contains ``rows*width`` values. Multiplying by
    bytes_per_value and table count gives allocated storage. Comparing that storage with
    labeled capacity tests only table residency, not the rest of the serving process.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.HundredMillionItemTableOnA100
    >>> s.total_bytes == s.table_bytes*s.tables
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    rows: int
    width: int
    bytes_per_value: Quantity
    hardware: HardwareNode
    tables: int = 1

    @property
    def row_bytes(self) -> Quantity:
        """Bytes in one embedding vector.

        Equation: ``(width * bytes_per_value).to('byte')``."""
        return (self.width * self.bytes_per_value).to("byte")

    @property
    def table_bytes(self) -> Quantity:
        """Decimal GB in one fully allocated embedding table.

        Equation: ``(rows * row_bytes).to('GB')``."""
        return (self.rows * self.row_bytes).to("GB")

    @property
    def total_bytes(self) -> Quantity:
        """Decimal GB across all identically shaped tables.

        Equation: ``(table_bytes * tables).to('GB')``."""
        return (self.table_bytes * self.tables).to("GB")

    @property
    def labeled_share(self) -> float:
        """All tables over the device's labeled capacity.

        Equation: ``(total_bytes / _labeled_capacity(hardware)).to('').magnitude``."""
        return (self.total_bytes / _labeled_capacity(self.hardware)).to("").magnitude

    @property
    def fits(self) -> bool:
        """Whether table storage alone fits labeled device capacity; runtime reserve is excluded.

        Equation: ``labeled_share <= 1``."""
        return self.labeled_share <= 1


class EmbeddingGatherScenario(BaseScenario):
    """One recommender request on ``hardware``: one row of ``width`` values per categorical feature of
    ``dataset`` is gathered, then the rows and the bottom MLP's output interact by pairwise dot products.

    Notes
    -----
    Equation: One embedding row is gathered per categorical feature; ``vectors*(vectors-1)/2`` dot products model interactions.

    Assumptions: One lookup per feature, equal embedding widths, and one bottom-MLP vector.

    Excludes: Caching, random-access efficiency, MLP execution, and overlap between host and device transfers.

    References
    ----------
    Derivation: The declared categorical-feature count determines gathered rows. Adding
    the bottom-MLP output gives the interacting vector count; choosing every unordered
    pair gives ``n*(n-1)/2`` dot products. Dividing a bandwidth by gathered bytes/request
    yields an ideal request-rate ceiling.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> s = Scenarios.CriteoShapedRequestOnH100
    >>> s.pairs == s.vectors*(s.vectors-1)//2
    True"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    dataset: DatasetProfile
    width: int
    bytes_per_value: Quantity
    hardware: HardwareNode

    @property
    def features(self) -> int:
        """Categorical-feature count that determines rows gathered per request.

        Equation: ``dataset.categorical_features``."""
        return self.dataset.categorical_features

    @property
    def row_bytes(self) -> Quantity:
        """Bytes fetched for one embedding row.

        Equation: ``(width * bytes_per_value).to('byte')``."""
        return (self.width * self.bytes_per_value).to("byte")

    @property
    def request_bytes(self) -> Quantity:
        """Bytes gathered across all categorical features in one request.

        Equation: ``(features * row_bytes).to('byte')``."""
        return (self.features * self.row_bytes).to("byte")

    @property
    def vectors(self) -> int:
        """The gathered rows and the bottom MLP's output.

        Equation: ``features + 1``."""
        return self.features + 1

    @property
    def pairs(self) -> int:
        """Unordered pair count among gathered embeddings and the bottom-MLP vector.

        Equation: ``vectors * (vectors - 1) // 2``."""
        return self.vectors * (self.vectors - 1) // 2

    @property
    def interaction_macs(self) -> int:
        """Multiply-accumulate count for all pairwise dot products.

        Equation: ``pairs * width``."""
        return self.pairs * self.width

    @property
    def intensity(self) -> Quantity:
        """Interaction FLOPs (two per multiply-add) per gathered byte.

        Equation: ``(Q_(2 * interaction_macs, 'flop') / request_bytes).to('flop/byte')``."""
        return (Q_(2 * self.interaction_macs, "flop") / self.request_bytes).to("flop/byte")

    @property
    def device_rate(self) -> Quantity:
        """Requests a second if every row streamed at device-memory bandwidth (an upper bound).

        Equation: ``(hardware.memory.bandwidth / request_bytes).to('1/s')``."""
        return (self.hardware.memory.bandwidth / self.request_bytes).to("1/s")

    @property
    def host_rate(self) -> Quantity:
        """Requests a second if every row crossed the host link (an upper bound).

        Equation: ``(hardware.interconnect.bandwidth / request_bytes).to('1/s')``."""
        return (self.hardware.interconnect.bandwidth / self.request_bytes).to("1/s")

    @property
    def bandwidth_ratio(self) -> float:
        """Dimensionless ideal device-memory request rate divided by ideal host-link rate.

        Equation: ``(device_rate / host_rate).to('').magnitude``."""
        return (self.device_rate / self.host_rate).to("").magnitude


class PrecomputeVersusOnDemandScenario(BaseScenario):
    """``items`` inputs answered one at a time on demand, or precomputed at a large-batch ``offline_rate``.

    Notes
    -----
    Equation: The calculation members below define ``on_demand_total``, ``offline_total``, ``ratio``.

    Assumptions: Every item is processed once; on-demand work is serial while offline throughput already includes batching.

    Excludes: Storage, refresh, and unused precomputed answers.

    References
    ----------
    Derivation: ``items`` inputs answered one at a time on demand, or precomputed at a large-batch ``offline_rate``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.PhotoLibraryPrecomputeH100
    >>> value = case.ratio
    >>> print(f"{value:.3g}")
    316"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    items: int
    on_demand_time: Quantity  # one request at batch one
    offline_rate: Quantity  # requests per second at large batch
    latency_budget: Quantity  # the on-demand service's deadline

    @property
    def on_demand_total(self) -> Quantity:
        """Seconds to answer every item serially at the on-demand per-item time.

        Equation: ``(items * on_demand_time).to('second')``."""
        return (self.items * self.on_demand_time).to("second")

    @property
    def offline_total(self) -> Quantity:
        """Seconds to precompute every item at the stated offline throughput.

        Equation: ``(items / offline_rate).to('second')``."""
        return (self.items / self.offline_rate).to("second")

    @property
    def ratio(self) -> float:
        """Dimensionless serial on-demand time divided by offline batch time.

        Equation: ``(on_demand_total / offline_total).to('').magnitude``."""
        return (self.on_demand_total / self.offline_total).to("").magnitude


class MM1CapacityPlanScenario(BaseScenario):
    """Accelerators for ``peak_rate`` when each is an M/M/1 server whose tail must meet ``tail_budget``.

    The tail is ``tail_factor`` times the mean time in system S/(1 - rho), so
    the highest safe utilization is 1 - k S / T. The plan divides the peak by
    that utilization, rounds up to whole accelerators, then adds ``headroom``.
    Optional: ``batched_rate`` re-plans with batched service at the same
    utilization; ``accelerator_price`` prices a plan per million requests.

    Notes
    -----
    Equation: The calculation members below define ``service_rate``, ``time_in_system``, ``tail_time``, ``exact_tail_factor``, ``max_utilization``, ``required_rate``, ``accelerators_exact``, ``accelerators``, ``accelerators_planned``, ``utilization_after_failure``, ``saturated_accelerators_exact``, ``batched_accelerators``, ``batched_accelerators_planned``, ``cost_per_million``.

    Assumptions: Each replica has exponential service and Poisson arrivals; load balances evenly; the configured tail factor approximates the target quantile.

    Excludes: Correlated bursts, heterogeneous replicas, batching-induced distribution changes, and startup delays.

    References
    ----------
    Derivation: Accelerators for ``peak_rate`` when each is an M/M/1 server whose tail must meet ``tail_budget``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50FleetPlanH100
    >>> value = case.accelerators_planned
    >>> print(value)
    36"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    service_time: Quantity
    tail_factor: float  # tail as a multiple of the mean
    tail_quantile: float = 0.99  # the tail the factor stands for
    tail_budget: Quantity  # queue-plus-service tail target
    peak_rate: Quantity
    headroom: float  # multiplier on the rounded-up count
    utilization_points: List[float] = []  # loads at which to report time in system
    example_utilization: Optional[float] = None
    batched_rate: Optional[Quantity] = None
    accelerator_price: Optional[Quantity] = None  # USD per accelerator-hour

    @property
    def service_rate(self) -> Quantity:
        """Per-accelerator service capacity in requests per second, reciprocal service time.

        Equation: ``(1 / service_time).to('1/second')``."""
        return (1 / self.service_time).to("1/second")

    def time_in_system(self, utilization) -> Quantity:
        """Mean request latency in ms via ``calc_mm1_time_in_system`` at the supplied utilization.

        Calculation: ``mlsysim.physics.serving.calc_mm1_time_in_system``."""
        from ...physics.serving import calc_mm1_time_in_system

        return calc_mm1_time_in_system(self.service_time, utilization).to("ms")

    def tail_time(self, utilization) -> Quantity:
        """Tail latency estimate in ms, configured tail factor times M/M/1 mean latency.

        Equation: ``(tail_factor * time_in_system(utilization)).to('ms')``."""
        return (self.tail_factor * self.time_in_system(utilization)).to("ms")

    @property
    def exact_tail_factor(self) -> float:
        """Exponential quantile multiplier via ``calc_exponential_quantile_factor``.

        The factor ``tail_factor`` rounds: ln(1/(1 - q)) for an exponential time in system.

        Calculation: ``mlsysim.physics.serving.calc_exponential_quantile_factor``."""
        from ...physics.serving import calc_exponential_quantile_factor

        return calc_exponential_quantile_factor(self.tail_quantile)

    @property
    def max_utilization(self) -> float:
        """Largest utilization meeting the tail target via ``calc_mm1_max_utilization``; negative means even idle service misses it.

        Calculation: ``mlsysim.physics.serving.calc_mm1_max_utilization``."""
        from ...physics.serving import calc_mm1_max_utilization

        return calc_mm1_max_utilization(self.service_time, self.tail_budget, self.tail_factor)

    @property
    def required_rate(self) -> Quantity:
        """Aggregate service rate in requests per second after utilization headroom.

        Equation: ``(peak_rate / max_utilization).to('1/second')``."""
        return (self.peak_rate / self.max_utilization).to("1/second")

    @property
    def accelerators_exact(self) -> float:
        """Fractional accelerator count before rounding, required rate divided by device capacity.

        Equation: ``(required_rate / service_rate).to('').magnitude``."""
        return (self.required_rate / self.service_rate).to("").magnitude

    @property
    def accelerators(self) -> int:
        """Integer accelerator count rounded up from the capacity requirement.

        Equation: ``_math.ceil(accelerators_exact)``."""
        return _math.ceil(self.accelerators_exact)

    @property
    def accelerators_planned(self) -> int:
        """Integer planned fleet after applying the multiplicative headroom factor and rounding up.

        Equation: ``_math.ceil(accelerators * headroom)``."""
        return _math.ceil(self.accelerators * self.headroom)

    @property
    def utilization_after_failure(self) -> float:
        """Dimensionless utilization after losing one planned accelerator; assumes equal load balancing.

        Utilization of the planned fleet with one accelerator lost.

        Equation: ``(peak_rate / (service_rate * (accelerators_planned - 1))).to('').magnitude``."""
        return (self.peak_rate / (self.service_rate * (self.accelerators_planned - 1))).to("").magnitude

    @property
    def saturated_accelerators_exact(self) -> float:
        """Fractional fleet size at full utilization, ignoring queueing headroom.

        Accelerators if each ran 100 percent busy (no queueing headroom).

        Equation: ``(peak_rate / service_rate).to('').magnitude``."""
        return (self.peak_rate / self.service_rate).to("").magnitude

    @property
    def batched_accelerators(self) -> int:
        """Integer fleet count using batched throughput and the same utilization ceiling.

        Equation: ``_math.ceil((peak_rate / (max_utilization * batched_rate)).to('').magnitude)``."""
        return _math.ceil((self.peak_rate / (self.max_utilization * self.batched_rate)).to("").magnitude)

    @property
    def batched_accelerators_planned(self) -> int:
        """Batched fleet count with multiplicative headroom and upward rounding.

        Equation: ``_math.ceil(batched_accelerators * headroom)``."""
        return _math.ceil(self.batched_accelerators * self.headroom)

    def cost_per_million(self, accelerators) -> Quantity:
        """USD per million requests at peak load for the supplied accelerator count.

        Fleet cost per million requests served at ``peak_rate``.

        Equation: ``(accelerators * accelerator_price / peak_rate * 1000000.0).to('dollar')``."""
        return (accelerators * self.accelerator_price / self.peak_rate * 1e6).to("dollar")


class MM1ServiceSpeedupScenario(BaseScenario):
    """The same traffic on an M/M/1 server before and after its service time falls.

    Notes
    -----
    Equation: The calculation members below define ``arrival_rate``, ``fast_utilization``, ``slow_time``, ``fast_time``, ``slow_wait``, ``fast_wait``, ``service_gain``, ``time_gain``, ``wait_gain``.

    Assumptions: Arrival rate stays fixed while service time changes; stable M/M/1 queues require utilization below one.

    Excludes: Burstiness, heavy-tailed service, and finite-queue rejection.

    References
    ----------
    Derivation: The same traffic on an M/M/1 server before and after its service time falls. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50FasterModelAt80PctH100
    >>> value = case.time_gain
    >>> print(f"{value:.3g}")
    5.87"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    slow_service: Quantity
    fast_service: Quantity
    utilization: float  # load with the slow service

    @property
    def arrival_rate(self) -> Quantity:
        """Fixed offered request rate per second inferred from slow-service utilization.

        Equation: ``(utilization / slow_service).to('1/second')``."""
        return (self.utilization / self.slow_service).to("1/second")

    @property
    def fast_utilization(self) -> float:
        """Dimensionless fast-server utilization at the same offered request rate.

        Equation: ``utilization * (fast_service / slow_service).to('').magnitude``."""
        return self.utilization * (self.fast_service / self.slow_service).to("").magnitude

    @property
    def slow_time(self) -> Quantity:
        """Slow-server mean time in system in ms via ``calc_mm1_time_in_system``.

        Calculation: ``mlsysim.physics.serving.calc_mm1_time_in_system``."""
        from ...physics.serving import calc_mm1_time_in_system

        return calc_mm1_time_in_system(self.slow_service, self.utilization).to("ms")

    @property
    def fast_time(self) -> Quantity:
        """Fast-server mean time in system in ms at its reduced utilization.

        Calculation: ``mlsysim.physics.serving.calc_mm1_time_in_system``."""
        from ...physics.serving import calc_mm1_time_in_system

        return calc_mm1_time_in_system(self.fast_service, self.fast_utilization).to("ms")

    @property
    def slow_wait(self) -> Quantity:
        """Slow-server queueing wait in ms, mean time in system minus service time.

        Equation: ``(slow_time - slow_service).to('ms')``."""
        return (self.slow_time - self.slow_service).to("ms")

    @property
    def fast_wait(self) -> Quantity:
        """Fast-server queueing wait in ms, mean time in system minus service time.

        Equation: ``(fast_time - fast_service).to('ms')``."""
        return (self.fast_time - self.fast_service).to("ms")

    @property
    def service_gain(self) -> float:
        """Dimensionless slow-to-fast service-time ratio.

        Equation: ``(slow_service / fast_service).to('').magnitude``."""
        return (self.slow_service / self.fast_service).to("").magnitude

    @property
    def time_gain(self) -> float:
        """Dimensionless slow-to-fast total-latency ratio, including queueing.

        Equation: ``(slow_time / fast_time).to('').magnitude``."""
        return (self.slow_time / self.fast_time).to("").magnitude

    @property
    def wait_gain(self) -> float:
        """Dimensionless slow-to-fast queueing-wait ratio.

        Equation: ``(slow_wait / fast_wait).to('').magnitude``."""
        return (self.slow_wait / self.fast_wait).to("").magnitude


class BatchWindowScenario(BaseScenario):
    """Requests arriving as a Poisson stream into a batching window of length ``window``.

    The window collects lambda W arrivals on average; a request waits half the
    window on average; the batch a request finds itself in is size-biased,
    1 + lambda W on average, and its ``tail_quantile`` is 1 plus that
    quantile of the Poisson count.

    Notes
    -----
    Equation: The calculation members below define ``mean_arrivals``, ``arrival_spread``, ``empty_share``, ``count_at_least``, ``mean_wait``, ``budget_share``, ``seen_batch_mean``, ``seen_batch_tail``.

    Assumptions: Arrivals follow a homogeneous Poisson process and positions in a fixed window are uniform.

    Excludes: Worker backlog, service time, and arrival correlations.

    References
    ----------
    Derivation: Requests arriving as a Poisson stream into a batching window of length ``window``. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.BatchWindow10ms
    >>> value = case.mean_wait
    >>> print(f"{value:.3g~}")
    5 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    window: Quantity
    arrival_rates: List[Quantity] = []
    latency_budget: Optional[Quantity] = None
    tail_quantile: float = 0.99

    def mean_arrivals(self, arrival_rate) -> float:
        """Expected arrivals in the window, arrival rate times duration, dimensionless count.

        Equation: ``(arrival_rate * window).to('').magnitude``."""
        return (arrival_rate * self.window).to("").magnitude

    def arrival_spread(self, arrival_rate) -> float:
        """Poisson arrival-count standard deviation, square root of the mean count.

        Standard deviation of the count, sqrt(lambda W).

        Equation: ``_math.sqrt(mean_arrivals(arrival_rate))``."""
        return _math.sqrt(self.mean_arrivals(arrival_rate))

    def empty_share(self, arrival_rate) -> float:
        """Probability of no arrivals, exponential of minus the mean count.

        Equation: ``_math.exp(-mean_arrivals(arrival_rate))``."""
        return _math.exp(-self.mean_arrivals(arrival_rate))

    def count_at_least(self, arrival_rate, at_least: int) -> float:
        """Poisson probability of at least the supplied count, via ``calc_poisson_tail``.

        Calculation: ``mlsysim.physics.statistics.calc_poisson_tail``."""
        from ...physics.statistics import calc_poisson_tail

        return calc_poisson_tail(self.mean_arrivals(arrival_rate), at_least)

    @property
    def mean_wait(self) -> Quantity:
        """Mean window-formation wait in ms under uniform arrival position, half the window.

        Equation: ``(window / 2).to('ms')``."""
        return (self.window / 2).to("ms")

    @property
    def budget_share(self) -> float:
        """Dimensionless share of latency budget consumed by mean formation wait.

        Equation: ``(mean_wait / latency_budget).to('').magnitude``."""
        return (self.mean_wait / self.latency_budget).to("").magnitude

    def seen_batch_mean(self, arrival_rate) -> float:
        """Expected tagged-request batch size, the request itself plus expected additional arrivals.

        Equation: ``1 + mean_arrivals(arrival_rate)``."""
        return 1 + self.mean_arrivals(arrival_rate)

    def seen_batch_tail(self, arrival_rate) -> int:
        """Tagged-request batch count threshold via ``calc_poisson_reserve`` under the stated tail quantile.

        1 plus the ``tail_quantile`` of the Poisson count: the smallest s with P(N >= s) below 1 - q.

        Calculation: ``mlsysim.physics.statistics.calc_poisson_reserve``."""
        from ...physics.statistics import calc_poisson_reserve

        return calc_poisson_reserve(self.mean_arrivals(arrival_rate), 1 - self.tail_quantile)


class BatchDispatchScenario(BaseScenario):
    """A dual-trigger batch: it may leave at ``timeout`` after its oldest arrival, then waits for a worker.

    Notes
    -----
    Equation: The calculation members below define ``formation_end``, ``dispatch``, ``backlog_wait``, ``batch_time``, ``oldest_latency``, ``slack``, ``timer_only_latency``.

    Assumptions: Arrivals are chronologically ordered; the timer expires after the oldest arrival and dispatch waits for the worker.

    Excludes: An independently simulated queue, size-trigger timing, and inter-request scheduling.

    References
    ----------
    Derivation: A dual-trigger batch: it may leave at ``timeout`` after its oldest arrival, then waits for a worker. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.DualTriggerBatchFourRequests
    >>> value = case.oldest_latency
    >>> print(f"{value:.3g~}")
    25.6 ms"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    arrivals: List[Quantity]  # offsets from the oldest request
    timeout: Quantity
    worker_free: Quantity  # when the worker finishes its current batch
    response_path: Quantity  # time after the batch finishes
    deadline: Quantity
    service: BatchingKneeScenario

    @property
    def formation_end(self) -> Quantity:
        """Timer-trigger time in ms, oldest arrival plus timeout.

        Equation: ``(arrivals[0] + timeout).to('ms')``."""
        return (self.arrivals[0] + self.timeout).to("ms")

    @property
    def dispatch(self) -> Quantity:
        """Actual dispatch time in ms, later of timer trigger and worker availability.

        Equation: ``max(formation_end, worker_free.to('ms'))``."""
        return max(self.formation_end, self.worker_free.to("ms"))

    @property
    def backlog_wait(self) -> Quantity:
        """Worker-availability delay after the timer trigger, in ms.

        Equation: ``(dispatch - formation_end).to('ms')``."""
        return (self.dispatch - self.formation_end).to("ms")

    @property
    def batch_time(self) -> Quantity:
        """Service duration for the observed arrival-count batch from the service profile.

        Equation: ``service.service_time(len(arrivals))``."""
        return self.service.service_time(len(self.arrivals))

    @property
    def oldest_latency(self) -> Quantity:
        """Oldest request's response latency in ms including formation, backlog, service, and response path.

        Equation: ``(dispatch + batch_time + response_path - arrivals[0]).to('ms')``."""
        return (self.dispatch + self.batch_time + self.response_path - self.arrivals[0]).to("ms")

    @property
    def slack(self) -> Quantity:
        """Deadline minus oldest-request latency in ms; negative means missed deadline.

        Equation: ``(deadline - oldest_latency).to('ms')``."""
        return (self.deadline - self.oldest_latency).to("ms")

    @property
    def timer_only_latency(self) -> Quantity:
        """Timeout plus batch service in ms, deliberately excluding worker backlog and response path.

        The answer that ignores the busy worker and the response path.

        Equation: ``(timeout + batch_time).to('ms')``."""
        return (self.timeout + self.batch_time).to("ms")


class BatchCapChoiceScenario(BaseScenario):
    """Two batch caps at one arrival rate, batches waiting until full: throughput against formation wait.

    Notes
    -----
    Equation: The calculation members below define ``formation_wait``, ``throughput_gain``, ``capacity_over_load``.

    Assumptions: Batches wait until full under the supplied arrival rate; service throughput comes from the same profile.

    Excludes: Timeout-triggered partial batches and worker backlog.

    References
    ----------
    Derivation: Two batch caps at one arrival rate, batches waiting until full: throughput against formation wait. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.BatchCap16Or32At500Qps
    >>> value = case.throughput_gain
    >>> print(f"{value:.3g}")
    0.769"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    service: BatchingKneeScenario
    arrival_rate: Quantity
    small_cap: int
    large_cap: int

    def formation_wait(self, cap) -> Quantity:
        """Mean full-batch formation wait in ms via ``calc_batch_formation_wait`` for the supplied cap.

        Calculation: ``mlsysim.physics.serving.calc_batch_formation_wait``."""
        from ...physics.serving import calc_batch_formation_wait

        return calc_batch_formation_wait(cap, self.arrival_rate).to("ms")

    @property
    def throughput_gain(self) -> float:
        """Fractional throughput increase from small to large batch cap.

        Equation: ``(service.throughput(large_cap) / service.throughput(small_cap)).to('').magnitude - 1``."""
        return (self.service.throughput(self.large_cap) / self.service.throughput(self.small_cap)).to("").magnitude - 1

    @property
    def capacity_over_load(self) -> float:
        """Small-cap saturated throughput divided by arrival rate, dimensionless.

        Equation: ``(service.throughput(small_cap) / arrival_rate).to('').magnitude``."""
        return (self.service.throughput(self.small_cap) / self.arrival_rate).to("").magnitude


class ReplicaStartupScenario(BaseScenario):
    """A serving replica's startup phases; ``warm_path`` and ``cold_path`` name the phases each start pays.

    Notes
    -----
    Equation: The calculation members below define ``warm_total``, ``cold_total``.

    Assumptions: Selected startup phases execute sequentially and each named phase is charged once per listed occurrence.

    Excludes: Parallel phase overlap and contention-dependent startup durations.

    References
    ----------
    Derivation: A serving replica's startup phases; ``warm_path`` and ``cold_path`` name the phases each start pays. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50ReplicaStartup
    >>> value = case.warm_total
    >>> print(f"{value:.3g~}")
    1.5 s"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    phases: Dict[str, Quantity]
    ranges: Dict[str, List[Quantity]] = {}  # stated low and high for phases given as a range
    warm_path: List[str]
    cold_path: List[str]

    @property
    def warm_total(self) -> Quantity:
        """Sum of named warm-path phase durations in seconds, with sequential execution assumed.

        Equation: ``sum((phases[p] for p in warm_path), Q_(0, 'second')).to('second')``."""
        return sum((self.phases[p] for p in self.warm_path), Q_(0, "second")).to("second")

    @property
    def cold_total(self) -> Quantity:
        """Sum of named cold-path phase durations in seconds, with sequential execution assumed.

        Equation: ``sum((phases[p] for p in cold_path), Q_(0, 'second')).to('second')``."""
        return sum((self.phases[p] for p in self.cold_path), Q_(0, "second")).to("second")


class TrafficSurgeScenario(BaseScenario):
    """A service sized for ``normal_rate`` that receives ``surge_rate``; latency in each regime.

    Notes
    -----
    Equation: The calculation members below define ``surge_factor``.

    Assumptions: normal_rate is positive and both rates count comparable requests.

    Excludes: A latency prediction from the surge ratio alone.

    References
    ----------
    Derivation: A service sized for ``normal_rate`` that receives ``surge_rate``; latency in each regime. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.RecommenderTrafficSurge
    >>> value = case.surge_factor
    >>> print(f"{value:.3g}")
    10"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    normal_rate: Quantity
    surge_rate: Quantity
    normal_latency: Quantity
    surge_latency: Quantity  # stated latency once the queue has grown

    @property
    def surge_factor(self) -> float:
        """Surge arrival rate divided by normal arrival rate, dimensionless.

        Equation: ``(surge_rate / normal_rate).to('').magnitude``."""
        return (self.surge_rate / self.normal_rate).to("").magnitude


class BatchEnergyPerImageScenario(BaseScenario):
    """Bytes, arithmetic intensity, and energy per image when a batch shares one weight read.

    Each image moves the weights divided by the batch plus its own
    activations ``activation_passes`` times (written once, read once), at
    ``byte_energy``; its arithmetic costs ``flop_energy`` per FLOP.

    Notes
    -----
    Equation: The calculation members below define ``weights``, ``weight_per_image``, ``bytes_per_image``, ``intensity``, ``movement_energy``, ``compute_energy``, ``energy``, ``movement_share``.

    Assumptions: One weight read is shared by the batch; each image pays the stated activation passes and arithmetic work.

    Excludes: Cache retention across batches, leakage, and utilization-dependent device power.

    References
    ----------
    Derivation: Bytes, arithmetic intensity, and energy per image when a batch shares one weight read. Each member records the corresponding algebra or named physics function.

    Examples
    --------
    Evaluate the named configuration and display a modeled result.

    >>> from mlsysim import Scenarios
    >>> case = Scenarios.ResNet50BatchEnergyPerImage
    >>> value = case.energy(1)
    >>> print(f"{value:.3g~}")
    39.2 mJ"""

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    weight_bytes_per_param: Quantity
    activation_bytes: Quantity  # one image's activation footprint
    activation_passes: int
    byte_energy: Quantity
    flop_energy: Quantity

    @property
    def weights(self) -> Quantity:
        """Weight footprint quantity from the workload and configured parameter precision.

        Equation: ``workload.size_in_bytes(weight_bytes_per_param)``."""
        return self.workload.size_in_bytes(self.weight_bytes_per_param)

    def weight_per_image(self, batch_size) -> Quantity:
        """Weight bytes charged per image when a batch shares one full weight read.

        Equation: ``weights / batch_size``."""
        return self.weights / batch_size

    def bytes_per_image(self, batch_size) -> Quantity:
        """Per-image traffic: amortized weight read plus stated activation passes.

        Equation: ``weight_per_image(batch_size) + activation_passes * activation_bytes``."""
        return self.weight_per_image(batch_size) + self.activation_passes * self.activation_bytes

    def intensity(self, batch_size) -> Quantity:
        """Per-image FLOP/byte under the weight-sharing and activation-traffic assumptions.

        Equation: ``(workload.inference_flops / bytes_per_image(batch_size)).to('flop/byte')``."""
        return (self.workload.inference_flops / self.bytes_per_image(batch_size)).to("flop/byte")

    def movement_energy(self, batch_size) -> Quantity:
        """Estimated data-movement energy per image in mJ from byte traffic times byte energy.

        Equation: ``(bytes_per_image(batch_size) * byte_energy).to('mJ')``."""
        return (self.bytes_per_image(batch_size) * self.byte_energy).to("mJ")

    @property
    def compute_energy(self) -> Quantity:
        """Estimated arithmetic energy per image in mJ from workload FLOPs times FLOP energy.

        Equation: ``(workload.inference_flops * flop_energy).to('mJ')``."""
        return (self.workload.inference_flops * self.flop_energy).to("mJ")

    def energy(self, batch_size) -> Quantity:
        """Total modeled per-image energy in mJ, movement plus arithmetic.

        Equation: ``movement_energy(batch_size) + compute_energy``."""
        return self.movement_energy(batch_size) + self.compute_energy

    def movement_share(self, batch_size) -> float:
        """Fraction of modeled per-image energy due to data movement.

        Equation: ``(movement_energy(batch_size) / energy(batch_size)).to('').magnitude``."""
        return (self.movement_energy(batch_size) / self.energy(batch_size)).to("").magnitude


class KvDecodeBatchScenario(BaseScenario):
    """Batched decode on one accelerator when every sequence also reads its own KV cache.

    Each step reads the weights once and each sequence's cache of
    ``context_tokens``: bytes ``W + B m_seq``, time ``(W + B m_seq) / B_mem``
    (a bandwidth floor), aggregate rate ``B / step``. The rate approaches
    ``B_mem / m_seq`` as caches dominate. A TPOT target caps the batch at
    ``floor((t B_mem - W) / m_seq)``; memory caps it at
    ``floor((M - W) / m_seq)``. Prefill of ``prompt_tokens`` runs at
    ``prefill_efficiency`` of the ``peak_precision`` peak.

    Notes
    -----
    Equation: Each memory step reads ``W+B*KV_sequence`` bytes; rate is B/step_time; capacity
    and latency ceilings solve those inequalities for B.

    Assumptions: Every sequence reads a fixed full-length cache, weights are read once per
    batch, and memory transfers use peak bandwidth.

    Excludes: Decode compute, collectives, launch overhead, workspaces, and evolving
    sequence lengths; prefill uses only the registered per-token forward count.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.transfer_time``. Their contracts
    supply the underlying equations; property equations show the remaining unit conversions
    and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Gpt2XlChatDecodeBatchesH100
    >>> value = scenario.weights
    >>> str(value.units)
    'GB'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    weight_bytes_per_param: Quantity
    kv_bytes_per_value: Quantity
    context_tokens: int
    batch_sizes: List[int]
    prompt_tokens: int
    prefill_efficiency: float
    peak_precision: str = "fp16_dense"

    @property
    def weights(self) -> Quantity:
        """Resident model weights in decimal GB at the selected bytes per parameter.

        Equation: ``workload.size_in_bytes(weight_bytes_per_param).to('GB')``.
        """
        return self.workload.size_in_bytes(self.weight_bytes_per_param).to("GB")

    @property
    def kv_per_token(self) -> Quantity:
        """Single-request KV-cache bytes per context token from the workload's cache geometry.

        Equation: ``workload.get_kv_cache_size(1, 1, kv_bytes_per_value).to('byte')``.
        """
        return self.workload.get_kv_cache_size(1, 1, self.kv_bytes_per_value).to("byte")

    @property
    def kv_per_sequence(self) -> Quantity:
        """Single-request cache in decimal MB at the fixed full context length.

        Equation: ``(kv_per_token * context_tokens).to('MB')``.
        """
        return (self.kv_per_token * self.context_tokens).to("MB")

    @property
    def bandwidth(self) -> Quantity:
        """Registered device memory bandwidth, without a sustained-efficiency adjustment.

        Equation: ``hardware.memory.bandwidth``.
        """
        return self.hardware.memory.bandwidth

    def step_bytes(self, batch_size: int) -> Quantity:
        """Decimal GB read per decode step: weights once plus a full cache per active sequence.

        Equation: ``(weights + batch_size * kv_per_sequence).to('GB')``.
        """
        return (self.weights + batch_size * self.kv_per_sequence).to("GB")

    def step_time(self, batch_size: int) -> Quantity:
        """Memory-transfer floor in milliseconds via ``transfer_time``; compute and launch costs
        are excluded.

        Equation: ``transfer_time(step_bytes(batch_size), bandwidth).to('ms')``.
        """
        from ...physics import transfer_time

        return transfer_time(self.step_bytes(batch_size), self.bandwidth).to("ms")

    def token_rate(self, batch_size: int) -> Quantity:
        """Aggregate tokens per second from one output token per sequence per modeled memory step.

        Equation: ``(batch_size / step_time(batch_size)).to('1/second')``.
        """
        return (batch_size / self.step_time(batch_size)).to("1/second")

    def cache_share(self, batch_size: int) -> float:
        """Fraction of modeled step bytes attributable to KV reads rather than weights.

        Equation: ``(batch_size * kv_per_sequence / step_bytes(batch_size)).to('').magnitude``.
        """
        return (batch_size * self.kv_per_sequence / self.step_bytes(batch_size)).to("").magnitude

    @property
    def rate_ceiling(self) -> Quantity:
        """Aggregate tokens per second approached when cache traffic dominates the memory step.

        Equation: ``(bandwidth / kv_per_sequence).to('1/second')``.
        """
        return (self.bandwidth / self.kv_per_sequence).to("1/second")

    def max_batch_within(self, tpot: Quantity) -> int:
        """Integer batch ceiling from the TPOT memory budget; negative values indicate no feasible
        positive batch.

        Equation: ``_math.floor(((tpot * bandwidth - weights) /
        kv_per_sequence).to('').magnitude)``.
        """
        return _math.floor(((tpot * self.bandwidth - self.weights) / self.kv_per_sequence).to("").magnitude)

    @property
    def max_batch_in_memory(self) -> int:
        """Integer weight-plus-cache capacity ceiling; negative values indicate weights already
        exceed capacity.

        Equation: ``_math.floor((free / kv_per_sequence).to('').magnitude)``.
        """
        free = self.hardware.memory.capacity - self.weights
        return _math.floor((free / self.kv_per_sequence).to("").magnitude)

    @property
    def peak_rate(self) -> Quantity:
        """Registered FLOP/s at the requested precision key, with no efficiency adjustment.

        Equation: ``hardware.compute.precision_flops[peak_precision]``.
        """
        return self.hardware.compute.precision_flops[self.peak_precision]

    @property
    def ridge(self) -> Quantity:
        """Peak compute divided by memory bandwidth in FLOP/byte.

        Equation: ``(peak_rate / bandwidth).to('flop/byte')``.
        """
        return (self.peak_rate / self.bandwidth).to("flop/byte")

    @property
    def prefill_ops(self) -> Quantity:
        """Prompt FLOPs from the workload's per-token forward count, without context-dependent
        attention work.

        Equation: ``(workload.inference_flops * prompt_tokens).to('flop')``.
        """
        return (self.workload.inference_flops * self.prompt_tokens).to("flop")

    @property
    def prefill_time(self) -> Quantity:
        """Compute-only prompt milliseconds at the declared fraction of registered peak FLOP/s.

        Equation: ``(prefill_ops / (peak_rate * prefill_efficiency)).to('ms')``.
        """
        return (self.prefill_ops / (self.peak_rate * self.prefill_efficiency)).to("ms")


class SingleAcceleratorLlmServiceScenario(BaseScenario):
    """An LLM chat service on one accelerator: latency, KV residency, request rate, and token cost.

    ``ServingModel`` gives TTFT and the weight footprint. TPOT combines the
    weight and request-cache transfer time with the solver's ITL residual
    beyond its memory floor; that residual includes compute dominance as well
    as framework overhead. The KV reserve over cache bytes per token sets
    resident requests. Request rate is the smaller of prefill admission and
    residency turnover (Little's law at the batch-one request time). The
    request-time calculation uses full-context TTFT; the separate ``ttft``
    property uses prompt-only TTFT. Token billing counts prompt and output tokens.

    Notes
    -----
    Equation: ServingModel supplies prompt/full-context timing; request rate is
    min(prefill_rate/prompt_tokens, resident_requests/request_time).

    Assumptions: Full-context cache size remains fixed throughout modeled decode; cache
    reserve and external admission rate are supplied assumptions.

    Excludes: Queueing and dynamic batching. The ITL-minus-memory-floor residual may include
    compute cost, and request_time uses full-context TTFT rather than the prompt-only ttft
    property.

    References
    ----------
    Derivation: the scenario composes ``mlsysim.physics.serving.calc_littles_law_rate``,
    ``mlsysim.physics.transfer_time``, ``mlsysim.solvers.ServingModel``. Their contracts
    supply the underlying equations; property equations show the remaining unit conversions
    and accounting.

    Examples
    --------
    >>> from mlsysim import Scenarios
    >>> scenario = Scenarios.Llama3_8BInt4ChatServiceH100
    >>> value = scenario.total_tokens
    >>> round(float(value), 3)
    1256.0
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    workload: Workload
    hardware: HardwareNode
    precision: str  # weight precision for ServingModel
    kv_bytes_per_value: Quantity
    prompt_tokens: int
    decode_tokens: int
    prefill_efficiency: float
    kv_reserve: Quantity
    prefill_rate: Quantity  # prompt tokens per second
    price: Quantity  # USD per accelerator-hour
    ttft_target: Optional[Quantity] = None
    tpot_target: Optional[Quantity] = None

    @property
    def total_tokens(self) -> int:
        """Prompt plus generated token count used to size the full resident request cache.

        Equation: ``prompt_tokens + decode_tokens``.
        """
        return self.prompt_tokens + self.decode_tokens

    def _serve(self, seq_len: int):
        from ..solvers import ServingModel

        return ServingModel().solve(
            self.workload,
            self.hardware,
            seq_len=seq_len,
            batch_size=1,
            precision=self.precision,
            efficiency=self.prefill_efficiency,
        )

    @property
    def ttft(self) -> Quantity:
        """Prompt-only model-side first-token milliseconds from ``ServingModel.solve``.

        Equation: ``_serve(prompt_tokens).ttft.to('ms')``.
        """
        return self._serve(self.prompt_tokens).ttft.to("ms")

    @property
    def weights(self) -> Quantity:
        """Resident weight bytes reported by the serving solver at the full request context.

        Equation: ``_serve(total_tokens).model_weights_size``.
        """
        return self._serve(self.total_tokens).model_weights_size

    @property
    def bandwidth(self) -> Quantity:
        """Registered memory bandwidth, treated as peak rather than measured sustained throughput.

        Equation: ``hardware.memory.bandwidth``.
        """
        return self.hardware.memory.bandwidth

    @property
    def kv_per_token(self) -> Quantity:
        """Single-request KV bytes for one context token at the chosen cache precision.

        Equation: ``workload.get_kv_cache_size(seq_len=1, batch_size=1,
        precision=kv_bytes_per_value)``.
        """
        return self.workload.get_kv_cache_size(seq_len=1, batch_size=1, precision=self.kv_bytes_per_value)

    @property
    def kv_per_request(self) -> Quantity:
        """Single-request KV bytes at prompt plus generated length.

        Equation: ``workload.get_kv_cache_size(seq_len=total_tokens, batch_size=1,
        precision=kv_bytes_per_value)``.
        """
        return self.workload.get_kv_cache_size(
            seq_len=self.total_tokens, batch_size=1, precision=self.kv_bytes_per_value
        )

    @property
    def weight_floor(self) -> Quantity:
        """Seconds to read the weights once at registered bandwidth via ``transfer_time``.

        Equation: ``transfer_time(weights, bandwidth)``.
        """
        from ...physics import transfer_time

        return transfer_time(self.weights, self.bandwidth)

    @property
    def dispatch_cost(self) -> Quantity:
        """Serving ITL minus its memory floor; this residual includes compute dominance, not solely
        dispatch overhead.

        Equation: ``res.itl - transfer_time(res.total_memory_required, bandwidth)``.
        """
        from ...physics import transfer_time

        res = self._serve(self.total_tokens)
        return res.itl - transfer_time(res.total_memory_required, self.bandwidth)

    @property
    def tpot(self) -> Quantity:
        """One-request weight-plus-cache transfer time plus the serving residual, in time units.

        Equation: ``transfer_time(weights + kv_per_request, bandwidth) + dispatch_cost``.
        """
        from ...physics import transfer_time

        return transfer_time(self.weights + self.kv_per_request, self.bandwidth) + self.dispatch_cost

    @property
    def decode_time(self) -> Quantity:
        """Generated token count times fixed-context TPOT, in seconds.

        Equation: ``(decode_tokens * tpot).to('second')``.
        """
        return (self.decode_tokens * self.tpot).to("second")

    @property
    def kv_capacity_tokens(self) -> int:
        """Rounded cache-token slots within the declared KV reserve, not total device memory.

        Equation: ``int(round((kv_reserve / kv_per_token).to('').magnitude))``.
        """
        return int(round((self.kv_reserve / self.kv_per_token).to("").magnitude))

    @property
    def resident_requests(self) -> int:
        """Whole full-context requests fitting the rounded KV-token budget.

        Equation: ``int(kv_capacity_tokens / total_tokens)``.
        """
        return int(self.kv_capacity_tokens / self.total_tokens)

    @property
    def request_time(self) -> Quantity:
        """Seconds using full-context serving TTFT plus generated-token decode time; differs from
        prompt-only ``ttft``.

        Equation: ``(_serve(total_tokens).ttft + decode_time).to('second')``.
        """
        return (self._serve(self.total_tokens).ttft + self.decode_time).to("second")

    @property
    def prefill_request_rate(self) -> Quantity:
        """Requests per second supported by the supplied prompt-token admission rate.

        Equation: ``(prefill_rate / prompt_tokens).to('1/second')``.
        """
        return (self.prefill_rate / self.prompt_tokens).to("1/second")

    @property
    def residency_request_rate(self) -> Quantity:
        """Requests per second from resident population over request time via Little's law.

        Equation: ``calc_littles_law_rate(resident_requests, request_time).to('1/second')``.
        """
        from ...physics.serving import calc_littles_law_rate

        return calc_littles_law_rate(self.resident_requests, self.request_time).to("1/second")

    @property
    def request_rate(self) -> Quantity:
        """Smaller of prompt admission and residency turnover limits, in requests per second.

        Equation: ``min(prefill_request_rate, residency_request_rate)``.
        """
        return min(self.prefill_request_rate, self.residency_request_rate)

    @property
    def tokens_per_hour(self) -> float:
        """Prompt and output tokens served in an hour at the modeled request rate.

        Equation: ``(request_rate * Q_(1, 'hour')).to('').magnitude * total_tokens``.
        """
        return (self.request_rate * Q_(1, "hour")).to("").magnitude * self.total_tokens

    @property
    def cost_per_million_tokens(self) -> Quantity:
        """Hourly device cost divided by hourly token volume, expressed as USD per million total
        tokens.

        Equation: ``(price * Q_(1, 'hour') / (tokens_per_hour / 1000000.0)).to('dollar')``.
        """
        return (self.price * Q_(1, "hour") / (self.tokens_per_hour / 1e6)).to("dollar")

    @property
    def remaining_memory(self) -> Quantity:
        """Decimal GB left after weights alone; excludes KV reserve and runtime workspace.

        Equation: ``(hardware.memory.capacity - weights).to('GB')``.
        """
        return (self.hardware.memory.capacity - self.weights).to("GB")
