"""Common scenario metadata and reusable input conventions.

Public compatibility imports remain in ``mlsysim.engine.scenario_types``.
"""

from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class BaseScenario(BaseModel):
    """Base for calculation scenarios: a name, a description, and stated inputs.

    Notes
    -----
    Equation: Use the dimensional products, ratios, or selection rules documented on the
    calculated properties below.

    Assumptions: A variant shares unchanged nested objects and does not revalidate
    replacement fields.

    Excludes: A variant is not a deep clone or a validation boundary.

    References
    ----------
    Derivation: the calculated properties below state their defining arithmetic and units.
    They are scenario accounting or input-selection rules, not independently measured
    relationships.

    Examples
    --------
    >>> case = BaseScenario(name="original", description="stated inputs")
    >>> case.variant(name="comparison").name
    'comparison'
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenario_types"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str

    def variant(self, **changes) -> "BaseScenario":
        """Return a shallow scenario copy with the supplied field replacements. Equation: Pydantic
        ``model_copy(update=changes)``; replacements are not revalidated. Units: unchanged
        fields retain their original quantities.
        """
        return self.model_copy(update=changes)


def _usd_per_hour(rate) -> float:
    """A GPU-hour price as a plain number of dollars per hour."""
    return rate.to("USD/hour").magnitude


from pydantic import BaseModel, ConfigDict
from typing import Optional, Union
from ...core.units import Q_
from ...core.types import Quantity
from ...models.types import Workload
from ...hardware.types import HardwareNode
from ...systems.types import Fleet
from ...core.exceptions import OOMError, SLAViolation
from ..evaluation import SystemEvaluation, EvaluationLevel


class Scenario(BaseModel):
    """
    Concrete executable case: a workload on a system with constraints.

    ``application`` names the product/domain context, while the scenario itself
    is what the simulator can evaluate. Model facts stay in ``Models.*``;
    hardware and fleet facts stay in ``Hardware.*`` / ``Systems.*``.
    """

    # Retain the public class identity used by Pydantic schemas and pickle.
    __module__ = "mlsysim.engine.scenarios"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    application: Optional[str] = None
    workload: Workload
    system: Union[Fleet, HardwareNode]

    # Constraints (SLAs)
    sla_latency: Optional[Quantity] = None
    target_accuracy: Optional[float] = None
    power_budget: Optional[Quantity] = None

    @property
    def is_distributed(self) -> bool:
        """Identify whether the scenario supplies a fleet object.

        Returns
        -------
        bool
            True for a Fleet system, false for a HardwareNode.

        Notes
        -----
        Equation: is_distributed = isinstance(system, Fleet).

        Assumptions: system satisfies the Scenario union type. A one-node Fleet still selects the distributed branch.

        Excludes: inferring parallelism from accelerator count or deciding whether distributed execution improves performance.

        References
        ----------
        Derivation: type-based dispatch selects which solver receives the scenario; it does not calculate a scaling benefit.

        Examples
        --------
        >>> from mlsysim import Scenario, Models, Hardware
        >>> scenario = Scenario(name='Example', description='Single device', workload=Models.Vision.ResNet50, system=Hardware.Cloud.H100)
        >>> scenario.is_distributed
        False
        """
        return isinstance(self.system, Fleet)

    def evaluate(self, batch_size: int = 1, precision: str = "fp16") -> SystemEvaluation:
        """Compose feasibility, performance, and annual accounting summaries.

        Parameters
        ----------
        batch_size : int, default: 1
            Positive batch count forwarded to the selected performance solver.
            Its work unit follows that solver: samples for generic single-node
            inference and the distributed solver's default training convention.
        precision : str, default: 'fp16'
            Registered weight format used for weight size and performance.

        Returns
        -------
        SystemEvaluation
            Three levels containing status, explanation, and metrics: weight/data
            feasibility, latency/SLA performance, and annual carbon/TCO estimates.
            Results include quantities with units; carbon_kg and tco_usd are scalars
            in kilograms and USD. A FAIL level does not stop later calculations.

        Notes
        -----
        Equation: feasible = (weight_bytes <= one_accelerator_capacity) and
        not data_stalled. Performance passes when no latency SLA is supplied or
        modeled_latency <= sla_latency. Macro metrics come from EconomicsModel
        with duration_days=365 and its other default assumptions.

        Assumptions: a Fleet uses DistributedModel; a HardwareNode uses
        SingleNodeModel. The initial memory check is weights on one accelerator,
        even for a Fleet. The single-node accounting branch constructs a one-node
        fleet with an Ethernet_10G fabric. Declared data_rate is checked through
        DataModel. Macro status is always PASS because it has no threshold here.

        Excludes: validation of target_accuracy or power_budget (stored fields
        not evaluated by this method), KV-aware serving capacity, allocator
        reserve, an explicit training-memory feasibility test at the first level,
        and matching macro utilization to the performance result. Solver defaults
        must be reviewed before treating the combined scorecard as a deployment.

        References
        ----------
        Derivation: this orchestration method composes existing solver results
        and applies explicit comparisons to weight capacity, data supply, and
        latency. It does not introduce a new performance law; annual economics
        is a separate default scenario rather than measured operating cost.

        Examples
        --------
        >>> from mlsysim import Scenario, Models, Hardware
        >>> scenario = Scenario(name='Example', description='Single device', workload=Models.Vision.ResNet50, system=Hardware.Cloud.H100)
        >>> result = scenario.evaluate()
        >>> result.feasibility.metrics['weight_size'] == scenario.workload.size_in_bytes('fp16')
        True
        >>> result.macro.status
        'PASS'
        """
        from ..solvers import DistributedModel, EconomicsModel

        # 1. Resolve Hardware
        hardware = self.system.node.accelerator if self.is_distributed else self.system

        # --- LEVEL 1: FEASIBILITY ---
        from ..solvers import DataModel

        weights = self.workload.size_in_bytes(precision=precision)
        mem_feasible = weights <= hardware.memory.capacity

        # Data Pipeline Check
        data_status = "PASS"
        data_summary = ""
        if self.workload.data_rate:
            ds = DataModel().solve(self.workload.data_rate, hardware)
            if ds.is_stalled:
                data_status = "FAIL"
                data_summary = f" | Data Wall: Pipeline Stalled ({ds.utilization:.1f}x capacity)"
            else:
                data_summary = f" | Data Pipeline: OK ({ds.utilization * 100:.1f}%)"

        feasible = mem_feasible and (data_status == "PASS")
        f_status = "PASS" if feasible else "FAIL"

        # Dynamic unit scaling for summary
        unit = "MB" if weights < Q_("1 GB") else "GB"
        mem_summary = (
            f"Model fits in memory ({weights.to(unit):.1f~P} / {hardware.memory.capacity.to(unit):.1f~P})"
            if mem_feasible
            else f"OOM: Requires {weights.to(unit):.1f~P} but only has {hardware.memory.capacity.to(unit):.1f~P}"
        )

        l1 = EvaluationLevel(
            level_name="Feasibility",
            status=f_status,
            summary=mem_summary + data_summary,
            metrics={
                "weight_size": weights,
                "capacity": hardware.memory.capacity,
                "data_stalled": data_status == "FAIL",
            },
        )

        # --- LEVEL 2: PERFORMANCE ---
        if self.is_distributed:
            solver = DistributedModel()
            perf = solver.solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.step_latency_total
            throughput = perf.effective_throughput
            perf_metrics = {
                "latency": actual_latency,
                "throughput": throughput,
                "scaling_eff": perf.scaling_efficiency,
                "mfu": perf.node_profile.mfu * perf.scaling_efficiency,
                "node_mfu": perf.node_profile.mfu,
                "sla_latency": self.sla_latency,
            }
        else:
            from ..solvers import SingleNodeModel

            perf = SingleNodeModel().solve(self.workload, self.system, batch_size=batch_size, precision=precision)
            actual_latency = perf.latency
            throughput = perf.throughput
            perf_metrics = {
                "latency": actual_latency,
                "throughput": throughput,
                "bottleneck": perf.bottleneck,
                "sla_latency": self.sla_latency,
            }

        p_status = "PASS"
        if self.sla_latency and actual_latency > self.sla_latency:
            p_status = "FAIL"

        p_summary = f"Latency: {actual_latency:.2f~P} (Target: {self.sla_latency or 'N/A'})"
        l2 = EvaluationLevel(level_name="Performance", status=p_status, summary=p_summary, metrics=perf_metrics)

        # --- LEVEL 3: MACRO ---
        # Scale to 1 year operation for macro view
        if self.is_distributed:
            sim_fleet = self.system
        else:
            from ...systems.types import Node, Fleet
            from ...systems.registry import Fabrics

            dummy_node = Node(name="Standard", accelerator=hardware, accelerators_per_node=1, intra_node_bw="50 GB/s")
            sim_fleet = Fleet(name="SimFleet", node=dummy_node, count=1, fabric=Fabrics.Ethernet_10G)

        # EconomicsModel internally delegates to SustainabilityModel and merges
        # its results, so we only need one call (avoids computing energy twice).
        econ = EconomicsModel().solve(sim_fleet, duration_days=365)

        m_summary = f"Annual Carbon: {econ.carbon_footprint_kg:.1f} kg | TCO: ${econ.tco_usd:,.0f}"
        l3 = EvaluationLevel(
            level_name="Macro",
            status="PASS",
            summary=m_summary,
            metrics={"carbon_kg": econ.carbon_footprint_kg, "tco_usd": econ.tco_usd},
        )

        return SystemEvaluation(scenario_name=self.name, feasibility=l1, performance=l2, macro=l3)

    def validate(self, batch_size: int = 1, precision: str = "fp16") -> bool:
        """Raise when the scenario's implemented feasibility or latency check fails.

        Parameters
        ----------
        batch_size : int, default: 1
            Positive batch count forwarded unchanged to evaluate.
        precision : str, default: 'fp16'
            Registered numeric format forwarded unchanged to evaluate.

        Returns
        -------
        bool
            True when the feasibility and performance levels do not report FAIL.

        Raises
        ------
        OOMError
            The first evaluation level fails, including a data-pipeline failure
            as well as insufficient single-accelerator weight capacity.
        SLAViolation
            The modeled latency exceeds the supplied SLA after feasibility passes.

        Notes
        -----
        Equation: validate returns True iff evaluate's feasibility and
        performance statuses are not FAIL; feasibility failures are raised first.

        Assumptions: evaluate defines the entire check scope. This method reruns
        that evaluation, including annual accounting, rather than using a cache.

        Excludes: checks for target_accuracy, power_budget, macro thresholds,
        KV residency, physical deployment safety, and validity of every modeling
        assumption. A passing result covers only the implemented comparisons.

        References
        ----------
        Derivation: translate the two evaluation statuses into exceptions in
        feasibility-first order. No new numerical calculation is performed beyond
        the delegated scenario evaluation.

        Examples
        --------
        >>> from mlsysim import Scenario, Models, Hardware
        >>> scenario = Scenario(name='Example', description='Single device', workload=Models.Vision.ResNet50, system=Hardware.Cloud.H100)
        >>> scenario.validate()
        True
        """
        eval_res = self.evaluate(batch_size=batch_size, precision=precision)

        # Level 1: Feasibility
        if eval_res.feasibility.status == "FAIL":
            raise OOMError(f"Feasibility Failure: {eval_res.feasibility.summary}")

        # Level 2: Performance
        if eval_res.performance.status == "FAIL":
            raise SLAViolation(f"Performance Failure: {eval_res.performance.summary}")

        return True
