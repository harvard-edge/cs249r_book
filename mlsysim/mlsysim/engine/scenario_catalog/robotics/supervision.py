"""Named robotics scenarios for supervision.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st


# --- Supervisory intervention on the warehouse base (added 2026-10-20) ---
# Site record Embodied.Scenario.WarehouseAisle: the finished stopping budget
# (S-curve jerk bound, tracking inset), the sourced takeover terms, the
# dispatch-to-admission allocation, the permission tick, and the chunk lease.
WarehouseInLoopTakeoverAtAisleSpeed = _st.TakeoverBudgetScenario(
    name="An in-loop takeover requested at the warehouse aisle speed",
    description="Slow-bound Model Human Processor response plus video and the four-phase handshake, "
    "waited out before braking at the candidate aisle speed; the handover timer is clipped "
    "to the fallback deadline.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.Scenario.WarehouseAisle.v_aisle,
    takeover=_Embodied.Scenario.WarehouseAisle.t_takeover_in_loop,
    configured_timeout=Q_(500.0, "ms"),  # scenario assumption: configured protocol ceiling on the handover timer
    manual_lease=Q_(100.0, "ms"),  # scenario assumption: the operator's loss-of-engagement lease
)


WarehouseOutOfLoopTakeoverAtAisleSpeed = WarehouseInLoopTakeoverAtAisleSpeed.variant(
    name="An out-of-loop takeover requested at the warehouse aisle speed",
    description="The 90th-percentile condition-mean take-over time plus the same console transport.",
    takeover=_Embodied.Scenario.WarehouseAisle.t_takeover_out_of_loop,
)


WarehouseInLoopTakeoverAtRequestSpeed = WarehouseInLoopTakeoverAtAisleSpeed.variant(
    name="An in-loop takeover requested after slowing to the in-loop request speed",
    description="The dispatch-charged in-loop ceiling rounded down to the speed resolution.",
    speed=_Embodied.Scenario.WarehouseAisle.v_takeover_in_loop,
)


WarehouseInLoopTakeoverAtRetiredSlowdown = WarehouseInLoopTakeoverAtAisleSpeed.variant(
    name="An in-loop takeover requested at 0.8 m/s, the partial ceiling rounded down",
    description="A slow-down that fits the partial budget and misses once dispatch to admission is charged.",
    speed=Q_(0.8, "m/s"),  # scenario assumption: the partial in-loop ceiling rounded down to 0.05 m/s
)


WarehouseInLoopTakeoverAtFastSlowdown = WarehouseInLoopTakeoverAtAisleSpeed.variant(
    name="An in-loop takeover requested at 1.2 m/s",
    description="A slow-down that would fit only a much faster human response.",
    speed=Q_(1.2, "m/s"),  # scenario assumption: illustrative slow-down
)


WarehouseOutOfLoopTakeoverAtCrawl = WarehouseOutOfLoopTakeoverAtAisleSpeed.variant(
    name="An out-of-loop takeover requested at the crawl speed",
    description="The chosen crawl under the out-of-loop ceiling.",
    speed=_Embodied.Scenario.WarehouseAisle.v_takeover_out_of_loop,
)


WarehouseOutOfLoopTakeoverAtCrawlP99 = WarehouseOutOfLoopTakeoverAtCrawl.variant(
    name="A 99th-percentile out-of-loop takeover at the crawl speed",
    description="The 99th-percentile condition-mean take-over time plus the console transport.",
    takeover=(
        _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p99
        + _Embodied.Scenario.WarehouseAisle.t_takeover_transport
    ).to("ms"),
)


WarehouseFixedWindowAtAisleSpeed = WarehouseInLoopTakeoverAtAisleSpeed.variant(
    name="The configured 500 ms handover window waited out at the aisle speed",
    description="One fixed acknowledgment window judged against the finished budget.",
    takeover=WarehouseInLoopTakeoverAtAisleSpeed.configured_timeout,
)


WarehouseFixedWindowAtCrawl = WarehouseFixedWindowAtAisleSpeed.variant(
    name="The configured 500 ms handover window waited out at the crawl speed",
    description="The same window at the out-of-loop crawl.",
    speed=_Embodied.Scenario.WarehouseAisle.v_takeover_out_of_loop,
)


WarehouseTakeoverCeilingLadder = _st.TakeoverCeilingLadderScenario(
    name="Speed ceilings of the warehouse base against sourced human responses",
    description="Model Human Processor simple reaction (nominal, slow bound), perception-brake times "
    "(expected, unexpected, surprise), and percentiles of condition-mean take-over time, "
    "each plus video and the four-phase handshake; partial budget.",
    base=WarehouseInLoopTakeoverAtAisleSpeed,
    transport=_Embodied.Scenario.WarehouseAisle.t_takeover_transport,
    responses={
        "mhp": _Embodied.Scenario.WarehouseAisle.t_human_simple_reaction,
        "mhp_slow": _Embodied.Scenario.WarehouseAisle.t_human_simple_reaction_slow,
        "expected": _Embodied.Scenario.WarehouseAisle.t_brake_response_expected,
        "unexpected": _Embodied.Scenario.WarehouseAisle.t_brake_response_unexpected,
        "surprise": _Embodied.Scenario.WarehouseAisle.t_brake_response_surprise,
        "p50": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p50,
        "mean": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_mean,
        "p90": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p90,
        "p99": _Embodied.Scenario.WarehouseAisle.t_takeover_driving_p99,
    },
)


WarehouseTakeoverBrakingBlend = _st.BrakingBlendScenario(
    name="A bumpless transfer into braking at the in-loop request speed, and an arm-joint blend",
    description="Smoothstep from cruise to the credible deceleration under the resident stop's jerk "
    "bound, against a step and the resident S-curve stop; an arm-joint torque change under "
    "a torque-rate ceiling.",
    site=_Embodied.Scenario.WarehouseAisle,
    speed=_Embodied.Scenario.WarehouseAisle.v_takeover_in_loop,
    torque_change=Q_(15.0, "N*m"),  # scenario assumption: an arm-joint command change (illustrative)
    torque_rate_limit=Q_(750.0, "N*m/s"),  # scenario assumption: torque-command rate ceiling (chosen)
)


WarehouseRemoteCommandAuthentication = _st.CommandAuthenticationScenario(
    name="Remote takeover commands authenticated on the warehouse machine's safety MCU",
    description="Published Cortex-M4 cycle counts (Ed25519 verification, AES-128 block) at the permission "
    "MCU's clock, against the aisle-speed spare and fallback deadline.",
    budget=WarehouseInLoopTakeoverAtAisleSpeed,
    clock=_Embodied.MobileManipulator.WarehouseMobileManipulator.permission_mcu.compute.clock_rate,
    verify_cycles=720000.0,  # Owens, D., et al. (2024), Efficient and Side-Channel Resistant Ed25...
    mac_block_cycles=661.7,  # Schwabe, P., and Stoffelen, K. (2016), All the AES You Need on Cort...
    mac_blocks=3,  # scenario assumption: a 48-byte frame (counter, timestamp, role, mask, command)
    freshness_window=Q_(100.0, "ms"),  # scenario assumption: freshness window (illustrative)
)
