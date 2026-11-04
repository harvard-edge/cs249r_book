"""Named statistics scenarios for fault injection.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st
from ....physics.drivetrain import calc_traction_deceleration_limit as _traction_limit
from ..robotics.realtime import WarehouseChunkLease
from ..robotics.stopping_enforcement import (
    WarehouseFinishedStopAtAisleSpeed,
    WarehouseFinishedStopInspectedFloor,
    WarehouseFinishedStopWithDispatch,
)


# --- Fault injection at the warehouse rack end: oracles, supply, and campaign statistics (added 2026-10-20) ---
# The renewal-referenced oracle scores a proposer stall from the last valid
# lease renewal: onset after the lease path and brake onset (the chunk
# lease's admission-to-brake-onset), the resident S-curve stop, and the
# tracking inset; it carries no evidence age and no localization or
# protective allowance, because a stalled proposer has no detected target.
WarehouseRenewalReferencedStop = _st.StoppingBudgetScenario(
    name="Warehouse base's renewal-referenced stopping oracle at the candidate aisle speed",
    description="Onset bound after the last valid renewal, the resident S-curve stop, and the tracking "
    "inset: the travel bound of a proposer-stall trial.",
    speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
    delay=WarehouseChunkLease.result.admission_to_brake_onset,
    deceleration=_Embodied.Scenario.WarehouseAisle.a_brake,
    clear_distance=_Embodied.Scenario.WarehouseAisle.d_clear,
    jerk=_Embodied.Scenario.WarehouseAisle.j_brake,
    tracking_bound=_Embodied.Scenario.WarehouseAisle.eps_track,
)


WarehouseRenewalReferencedStopRestricted = WarehouseRenewalReferencedStop.variant(
    name="Renewal-referenced stopping oracle at the restricted point on the inspected floor",
    description="1.0 m/s, below the inspected floor's ceiling, with deceleration capped by its friction "
    "(optimistic all-weight screen); the jerk bound belongs to the load and is unchanged.",
    speed=Q_(1.0, "m/s"),  # scenario assumption: candidate restricted speed on inspected floors
    deceleration=_traction_limit(_Embodied.Scenario.WarehouseAisle.mu_inspected_floor),
)


WarehouseFinishedStopRestricted = WarehouseFinishedStopInspectedFloor.variant(
    name="Capture-referenced finished budget at the restricted point on the inspected floor",
    description="The design subtotal from capture at the restricted speed and the inspected floor's deceleration.",
    speed=WarehouseRenewalReferencedStopRestricted.speed,
)


WarehouseFlagReferencedStop = WarehouseRenewalReferencedStop.variant(
    name="Stop at the candidate aisle speed scored from an undervoltage or feed-loss flag",
    description="One permission tick, one bus cycle, and the brake onset from the flag, the resident "
    "S-curve stop, and the tracking inset, when the drives keep their authority.",
    delay=(
        _Embodied.Scenario.WarehouseAisle.t_tick
        + _Embodied.Scenario.WarehouseAisle.t_bus
        + _Embodied.Scenario.WarehouseAisle.t_brake_onset
    ).to("ms"),
)


WarehouseRenewalStopLateOnsetTrial = _st.OnsetTrialScenario(
    name="A proposer-stall trace whose onset is 30 ms late at the candidate aisle speed",
    description="Signed margins against the renewal-referenced oracle: the onset predicate fails while "
    "the travel predicate passes.",
    oracle=WarehouseRenewalReferencedStop,
    onset=(WarehouseChunkLease.result.admission_to_brake_onset + Q_(30.0, "ms")).to(
        "ms"
    ),  # scenario assumption: illustrative late onset (a memory stall)
    lease=_Embodied.Scenario.WarehouseAisle.t_lease,
    probe_after_expiry=Q_(5.0, "ms"),  # scenario assumption: chosen replay of a refused renewal
)


WarehouseRenewalStopNinetyMsOnsetTrial = WarehouseRenewalStopLateOnsetTrial.variant(
    name="A proposer-stall trace with deceleration onset measured at 90 ms",
    description="A checkpoint given: the onset margin and the travel the late onset adds.",
    onset=Q_(90.0, "ms"),  # scenario assumption: checkpoint given, a measured onset
)


WarehousePermissionRailHoldUp = _st.RailHoldUpScenario(
    name="The warehouse machine's permission-rail hold-up during the longest inspected-floor stop",
    description="Feed lost while the base stops from the inspected floor's optimistic ceiling: flag to "
    "onset, the S-curve stop, and spring-brake engagement against the rail's hold-up.",
    site=_Embodied.Scenario.WarehouseAisle,
    stop=WarehouseFinishedStopInspectedFloor,
)


WarehouseQueueDelaySpareExhaustion = _st.ExponentialRareEventScenario(
    name="An unbudgeted queue delay that spends the rack-end spare at the candidate aisle speed",
    description="Exponential delay per stop; failure past the spare's delay at the aisle speed; plain "
    "Monte Carlo, zero-failure testing, and a cross-entropy importance-sampling proposal.",
    budget=WarehouseFinishedStopAtAisleSpeed,
    mean=Q_(5.0, "ms"),  # scenario assumption: illustrative mean queue delay per stop
    elite_share=0.1,  # scenario assumption: chosen cross-entropy elite fraction
    relative_error=0.1,  # scenario assumption: chosen target relative standard error
    confidence=0.95,  # scenario assumption: chosen
)


WarehouseQueueDelayChargedSpareExhaustion = WarehouseQueueDelaySpareExhaustion.variant(
    name="The same queue delay against the spare left after the dispatch-to-admission allocation",
    description="The failure threshold after the dispatch-to-admission allocation is charged.",
    budget=WarehouseFinishedStopWithDispatch,
)


WarehouseDiagnosticCoverageHundredFit = _st.DiagnosticCoverageScenario(
    name="Diagnostic coverage and zero-miss injections that leave 100 FIT unmitigated",
    description="Four component strata of the running machine's raw fault rate, a residual-rate target, "
    "and 95 percent confidence.",
    strata={  # scenario assumption: illustrative component fault rates, FIT
        "sensors": 1500.0,
        "compute": 1000.0,
        "transceivers and power": 2500.0,
        "gate drivers": 4800.0,
    },
    target=100.0,  # scenario assumption: chosen unmitigated-rate target, FIT
    confidence=0.95,  # scenario assumption: chosen
)


WarehouseDiagnosticCoverageTenFit = WarehouseDiagnosticCoverageHundredFit.variant(
    name="Diagnostic coverage and zero-miss injections that leave 10 FIT unmitigated",
    description="A tenfold tighter residual-rate target on the same strata.",
    target=10.0,  # scenario assumption: chosen tighter target, FIT
)
