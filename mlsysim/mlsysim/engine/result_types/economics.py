"""Typed result models for economics calculations."""

from __future__ import annotations

from typing import Dict, Optional, List, Tuple
from ...core.types import Quantity
from .base import SolverResult

__all__ = [
    "EconomicsResult",
    "ResponsibleEngineeringResult",
    "CohortAttemptCostResult",
    "TokenPriceComparisonResult",
    "PrefixCacheBillResult",
    "CompactionPaybackResult",
    "RetryPolicyResult",
    "SerialEpisodeResult",
    "SpeculativeEpisodeResult",
    "ModelCascadeResult",
    "TrajectoryAdmissionResult",
    "KingmanQueueResult",
    "DedicatedBreakEvenResult",
    "ConservedBudgetResult",
    "ArchitectureChoiceResult",
    "AdaptationPaybackResult",
    "WorkloadLeverRow",
    "WorkloadLeverResult",
]


class EconomicsResult(SolverResult):
    """Result from EconomicsModel: TCO breakdown.

    Embeds sustainability metrics since EconomicsModel delegates to
    SustainabilityModel internally.
    """

    __module__ = "mlsysim.engine.results"

    capex_usd: float
    opex_energy_usd: float
    opex_maintenance_usd: float
    total_opex_usd: float
    tco_usd: float
    # Embedded from SustainabilityResult
    it_energy_kwh: Quantity
    total_energy_kwh: Quantity
    carbon_footprint_kg: float
    water_usage_liters: float
    pue: float
    region_name: str

    @property
    def capex(self) -> Quantity:
        """Return amortized capital expenditure with units.

        Returns
        -------
        Quantity
            Amortized capital expenditure as a Pint quantity, conventionally in USD.

        Notes
        -----
        Equation: capex = Q_(capex_usd, "USD").

        Assumptions: The stored scalar capex_usd is expressed in USD and retains the
        accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying account, converting currencies or reporting
        periods, and checking consistency with other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit USD to the stored scalar
        capex_usd; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.capex == Q_(r.capex_usd, "USD")
        True
        """
        from ...core.units import Q_

        return Q_(self.capex_usd, "USD")

    @property
    def opex_energy(self) -> Quantity:
        """Return electricity operating expenditure with units.

        Returns
        -------
        Quantity
            Electricity operating expenditure as a Pint quantity, conventionally in USD.

        Notes
        -----
        Equation: opex_energy = Q_(opex_energy_usd, "USD").

        Assumptions: The stored scalar opex_energy_usd is expressed in USD and retains the
        accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying account, converting currencies or reporting
        periods, and checking consistency with other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit USD to the stored scalar
        opex_energy_usd; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.opex_energy == Q_(r.opex_energy_usd, "USD")
        True
        """
        from ...core.units import Q_

        return Q_(self.opex_energy_usd, "USD")

    @property
    def opex_maintenance(self) -> Quantity:
        """Return maintenance operating expenditure with units.

        Returns
        -------
        Quantity
            Maintenance operating expenditure as a Pint quantity, conventionally in USD.

        Notes
        -----
        Equation: opex_maintenance = Q_(opex_maintenance_usd, "USD").

        Assumptions: The stored scalar opex_maintenance_usd is expressed in USD and retains
        the accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying account, converting currencies or reporting
        periods, and checking consistency with other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit USD to the stored scalar
        opex_maintenance_usd; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.opex_maintenance == Q_(r.opex_maintenance_usd, "USD")
        True
        """
        from ...core.units import Q_

        return Q_(self.opex_maintenance_usd, "USD")

    @property
    def total_opex(self) -> Quantity:
        """Return total operating expenditure with units.

        Returns
        -------
        Quantity
            Total operating expenditure as a Pint quantity, conventionally in USD.

        Notes
        -----
        Equation: total_opex = Q_(total_opex_usd, "USD").

        Assumptions: The stored scalar total_opex_usd is expressed in USD and retains the
        accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying account, converting currencies or reporting
        periods, and checking consistency with other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit USD to the stored scalar
        total_opex_usd; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.total_opex == Q_(r.total_opex_usd, "USD")
        True
        """
        from ...core.units import Q_

        return Q_(self.total_opex_usd, "USD")

    @property
    def tco(self) -> Quantity:
        """Return total cost of ownership with units.

        Returns
        -------
        Quantity
            Total cost of ownership as a Pint quantity, conventionally in USD.

        Notes
        -----
        Equation: tco = Q_(tco_usd, "USD").

        Assumptions: The stored scalar tco_usd is expressed in USD and retains the
        accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying account, converting currencies or reporting
        periods, and checking consistency with other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit USD to the stored scalar
        tco_usd; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.tco == Q_(r.tco_usd, "USD")
        True
        """
        from ...core.units import Q_

        return Q_(self.tco_usd, "USD")

    @property
    def it_energy(self) -> Quantity:
        """Return it energy before facility overhead with units.

        Returns
        -------
        Quantity
            IT energy before facility overhead as a Pint quantity, conventionally in kWh.

        Notes
        -----
        Equation: it_energy = it_energy_kwh, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied it_energy_kwh as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing it_energy_kwh field
        directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.it_energy is r.it_energy_kwh
        True
        """
        return self.it_energy_kwh

    @property
    def total_energy(self) -> Quantity:
        """Return facility energy after pue overhead with units.

        Returns
        -------
        Quantity
            Facility energy after PUE overhead as a Pint quantity, conventionally in kWh.

        Notes
        -----
        Equation: total_energy = total_energy_kwh, with no arithmetic or unit conversion.

        Assumptions: The producing solver supplied total_energy_kwh as a time, energy, or
        memory quantity appropriate to this result.

        Excludes: Re-evaluation of the model and any new feasibility or unit-dimension
        check. The original quantity and its unit are returned unchanged.

        References
        ----------
        Derivation: This compatibility property returns the existing total_energy_kwh field
        directly; its value is defined by that field rather than a separate formula.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.total_energy is r.total_energy_kwh
        True
        """
        return self.total_energy_kwh

    @property
    def carbon_footprint(self) -> Quantity:
        """Return stored carbon footprint with units.

        Returns
        -------
        Quantity
            Stored carbon footprint as a Pint quantity, conventionally in kg CO2-equivalent. The
            attached mass unit does not distinguish greenhouse gases.

        Notes
        -----
        Equation: carbon_footprint = Q_(carbon_footprint_kg, "kg").

        Assumptions: The stored scalar carbon_footprint_kg is expressed in kg CO2-equivalent
        and retains the accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying footprint or checking consistency with the
        other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit kg to the stored scalar
        carbon_footprint_kg; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.carbon_footprint == Q_(r.carbon_footprint_kg, "kg")
        True
        """
        from ...core.units import Q_

        return Q_(self.carbon_footprint_kg, "kg")

    @property
    def water_usage(self) -> Quantity:
        """Return stored cooling-water use with units.

        Returns
        -------
        Quantity
            Stored cooling-water use as a Pint quantity, conventionally in liters.

        Notes
        -----
        Equation: water_usage = Q_(water_usage_liters, "liter").

        Assumptions: The stored scalar water_usage_liters is expressed in liters and retains
        the accounting boundary chosen by the producing solver.

        Excludes: Recomputing the underlying footprint or checking consistency with the
        other result fields.

        References
        ----------
        Derivation: This accessor attaches the documented unit liter to the stored scalar
        water_usage_liters; it performs no additional physical calculation.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> from mlsysim.engine.results import EconomicsResult
        >>> r = EconomicsResult(capex_usd=10, opex_energy_usd=2,
        ...     opex_maintenance_usd=3, total_opex_usd=5, tco_usd=15,
        ...     it_energy_kwh=Q_(1, "kWh"), total_energy_kwh=Q_(2, "kWh"),
        ...     carbon_footprint_kg=3, water_usage_liters=4, pue=2, region_name="example")
        >>> r.water_usage == Q_(r.water_usage_liters, "liter")
        True
        """
        from ...core.units import Q_

        return Q_(self.water_usage_liters, "liter")


class ResponsibleEngineeringResult(SolverResult):
    """Result from ResponsibleEngineeringModel: ethics tax quantification."""

    __module__ = "mlsysim.engine.results"

    dp_slowdown_factor: float
    effective_training_time: Quantity
    additional_data_requirement: float
    epsilon: float
    delta: float
    min_subgroup_prevalence: float
    privacy_cost_ratio: float
    fairness_data_ratio: float


class CohortAttemptCostResult(SolverResult):
    """Result from CohortAttemptCostModel: one configuration's cohort, one attempt per task (dollars)."""

    __module__ = "mlsysim.engine.results"

    model_cost: float  # per attempt
    environment_cost: float  # per attempt
    attempt_cost: float
    cohort_cost: float
    rejected: int
    cost_per_accepted: float
    rejected_per_accepted: float
    cohort_cost_with_triage: float
    cost_per_accepted_with_triage: float


class TokenPriceComparisonResult(SolverResult):
    """Result from comparing two cohort configurations under one acceptance contract."""

    __module__ = "mlsysim.engine.results"

    baseline: CohortAttemptCostResult
    alternative: CohortAttemptCostResult
    cost_ratio: float  # alternative over baseline, per accepted task
    cost_ratio_with_triage: float
    model_cut: float  # fraction of the token charge removed per attempt
    attempt_cut: float  # fraction of the all-in attempt cost removed
    acceptance_cut: float  # fraction of accepted tasks lost: the attempt cut that would tie


class PrefixCacheBillResult(SolverResult):
    """Result from PrefixCacheBillModel: an append-only trajectory billed with and without prefix reuse."""

    __module__ = "mlsysim.engine.results"

    sent_tokens: int  # logical input over the trajectory
    fresh_tokens: int  # billed at the input rate under ideal reuse
    input_uncached: Quantity  # USD
    input_cached: Quantity
    output_cost: Quantity
    total_uncached: Quantity
    total_cached: Quantity
    cache_ratio: float  # total_uncached / total_cached
    cached_reread_cost: Quantity  # cached re-reads of earlier history
    reread_share: float  # their share of the cached bill
    miss_cost: Quantity  # added when miss_fraction of the eligible prefix misses
    miss_share: float  # miss_cost / total_cached
    per_turn_uncached: List[float]  # dollars per call
    per_turn_cached: List[float]


class CompactionPaybackResult(SolverResult):
    """Result from CompactionPaybackModel: whether one compaction repays its summary and cache miss."""

    __module__ = "mlsysim.engine.results"

    context_at_compaction: int  # tokens before the compacting call
    remaining_calls: int  # calls from the compacting call on
    summary_cost: Quantity  # USD: the compacting call kept plus the summary's output
    upfront: Quantity  # incremental cost of the first compacted call
    per_call_saving: Quantity  # on every later call
    breakeven_calls: int  # first call count at which compaction has paid
    keep_cost: Quantity  # input bill of the remaining calls without compaction
    compact_cost: Quantity  # with compaction, before the summary
    compact_total: Quantity
    gap_before: Quantity  # still owed one call before the crossing
    saving_at: Quantity  # ahead at the crossing
    net_by_calls: Dict[int, float]  # compact minus keep, dollars, by calls remaining


class RetryPolicyResult(SolverResult):
    """Result from RetryPolicyModel: sequential early stop, a paid batch, and one stronger attempt."""

    __module__ = "mlsysim.engine.results"

    coverage: float  # 1 - (1 - p)^k
    expected_attempts: float  # sum of (1 - p)^j, j < k
    sequential_cost: Quantity  # USD per submitted task
    batch_cost: Quantity
    sequential_per_accepted: Quantity
    batch_per_accepted: Quantity
    strong_per_accepted: Quantity
    batch_premium: float  # k / expected attempts


class SerialEpisodeResult(SolverResult):
    """Result from SerialEpisodeModel: one serial episode and what a faster model buys (Amdahl)."""

    __module__ = "mlsysim.engine.results"

    model_time: Quantity
    tests_time: Quantity
    lifecycle_time: Quantity
    runtime_time: Quantity
    other_time: Quantity
    total_time: Quantity
    model_fraction: float
    changed_time: Quantity  # with the model intervals sped up
    saving: Quantity
    speedup: float
    limiting_speedup: float  # instant model calls: 1 / (1 - f)
    tests_half_saving: Quantity  # alternative: halve the required tests


class SpeculativeEpisodeResult(SolverResult):
    """Result from SpeculativeEpisodeModel: a decode-stream speedup composed with a serial episode."""

    __module__ = "mlsysim.engine.results"

    break_even_accepted: float  # gamma r
    speedup_high: float
    speedup_low: float
    episode_time_high: Quantity
    episode_time_low: Quantity
    episode_speedup_high: float
    episode_speedup_bound: float  # 1 / (1 - f)


class ModelCascadeResult(SolverResult):
    """Result from ModelCascadeModel: a tiered cascade under one gate, against the top tier alone (dollars)."""

    __module__ = "mlsysim.engine.results"

    reached: List[int]
    conditional_acceptance: List[float]
    reach_shares: List[float]
    accepted: int
    unresolved: int
    spend: float
    per_initiated: float
    per_accepted: float
    direct_spend: float
    direct_per_accepted: float
    mean_time: float  # seconds
    all_tiers_time: float
    direct_time: float
    direct_ratio: float
    lower_tier_toll: float  # paid by every task reaching the top tier
    saving_per_accepted: float
    # Optional: a harm charged per false accept at the first tier.
    false_accept_breakeven: Optional[float] = None  # first-tier false-accept rate that erases the saving
    harm_cost: Optional[float] = None  # at the stated false-accept rate
    per_accepted_with_harm: Optional[float] = None


class TrajectoryAdmissionResult(SolverResult):
    """Result from TrajectoryAdmissionModel: admitted trajectories against a token quota (mean demand)."""

    __module__ = "mlsysim.engine.results"

    concurrent: float  # Little's law, trajectories present
    call_rate: Quantity  # model calls per unit time
    tokens_per_trajectory: int
    demand: Quantity  # quota-counted tokens per unit time
    quota_share: float
    arrival_ceiling: Quantity  # admissions the quota allows
    acceptance_margin: float  # relative acceptance drop that fills the quota at fixed goodput
    quota_share_after_drop: Optional[float] = None


class KingmanQueueResult(SolverResult):
    """Result from KingmanQueueModel: mean wait at two utilizations and its holding price."""

    __module__ = "mlsysim.engine.results"

    variability: float  # (c_a^2 + c_s^2) / 2
    wait_low: Quantity
    wait_high: Quantity
    holding_low: float  # dollars per call
    holding_high: float
    wait_ratio: float
    throughput_gain: float  # rho_high / rho_low - 1
    unit_wait_utilization: float  # utilization at which the wait equals one service time


class DedicatedBreakEvenResult(SolverResult):
    """Result from DedicatedBreakEvenModel: renting tokens against owning capacity (dollars)."""

    __module__ = "mlsysim.engine.results"

    weighted_input_price: float  # per million input tokens
    api_price: float  # per million output tokens, input included
    reference_hour_tokens: float
    delivered_hour_tokens: float
    api_hour: float
    dedicated_hour: float
    crossing: float  # load fraction at which the costs tie
    saving_hour: float
    api_price_no_cache: float
    crossing_no_cache: float


class ConservedBudgetResult(SolverResult):
    """Result from ConservedBudgetModel: a root budget across two children, a stop, and a late bill.

    Integer thousandths of a dollar, so conservation is exact.
    """

    __module__ = "mlsysim.engine.results"

    units_per_dollar: int
    budget: int
    planning: int
    allocations: Tuple[int, int]
    settled_children: Tuple[int, int]
    pending: int
    late_charge: int
    free_after_planning: int
    reserved_delegated: int
    free_after_delegation: int
    refund_first: int
    settled_after_first: int
    reserved_after_first: int
    free_after_first: int
    child_free_at_stop: int  # what the stopped child may release
    child_apparent_free: int  # what a release at local stop would return
    child_free_settled: int  # released at settlement
    settled_final: int
    reserved_final: int
    free_final: int
    states: List[Tuple[int, int, int]]  # (settled, reserved, free) after each step


class ArchitectureChoiceResult(SolverResult):
    """Result from ArchitectureChoiceModel: three designs priced with and without human fallback."""

    __module__ = "mlsysim.engine.results"

    loop_acceptance: float
    loop_turns: float
    reach: List[float]  # share of tickets reaching each loop turn
    automated: Dict[str, Quantity]  # automated spend per ticket
    automated_per_accepted: Dict[str, Quantity]
    combined: Dict[str, Quantity]  # automated spend plus fallback, per ticket
    resolved: Dict[str, float]  # combined resolution
    per_resolved: Dict[str, Quantity]
    month_spend: Quantity  # for the loop
    month_resolved: float
    point_value: Quantity  # fallback avoided per ticket by one point of acceptance


class AdaptationPaybackResult(SolverResult):
    """Result from AdaptationPaybackModel: tasks until a one-time adaptation repays itself."""

    __module__ = "mlsysim.engine.results"

    base_step: Quantity
    base_task: Quantity
    adapted_step: Quantity
    adapted_task: Quantity
    investment: Quantity
    saving_per_task: Quantity
    breakeven_tasks: float
    breakeven_whole: int
    payback_months: float
    payback_months_low: float
    input_saving_share: float
    evaluation_share: float
    training_share: float


class WorkloadLeverRow(SolverResult):
    """One workload in WorkloadLeverResult (dollars)."""

    __module__ = "mlsysim.engine.results"

    model: float
    environment: float
    assessment: float
    human: float
    acceptance: float
    attempt: float
    cost_per_accepted: float
    model_share: float
    largest_term: str
    halve_tokens: float  # fractional saving in cost per accepted task
    halve_largest: float
    acceptance_gain: float
    token_threshold: float  # model share below which the acceptance gain beats halving tokens


class WorkloadLeverResult(SolverResult):
    """Result from WorkloadLeverModel: cost per accepted task by workload and three levers on it."""

    __module__ = "mlsysim.engine.results"

    rows: Dict[str, WorkloadLeverRow]
