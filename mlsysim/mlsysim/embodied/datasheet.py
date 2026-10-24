"""Machine data sheet: every input the Volume IV calculations take from the running machine.

The book's running machine is the warehouse mobile manipulator
(``Embodied.MobileManipulator.WarehouseMobileManipulator``) at its aisle site
(``Embodied.Scenario.WarehouseAisle``). This module aggregates the registry
values those chapters use into one ordered sheet, each row carrying its
evidence category in the vocabulary of the book's Reader Guide (Measured,
Derived, Datasheet, Illustrative, Chosen), the MLSysIM field that holds it, and
a note. A small subset forms the machine card that Chapter 1 prints.

This is an aggregation helper: it adds no value of its own. Every number is
read from the registry, or computed from registry values by an existing
MLSysIM function, when the sheet is built, so a printed sheet cannot drift from
the registry. Categories for sourced leaves follow their provenance kind
(datasheet -> Datasheet, literature -> Measured); categories for the
registry's unsourced design values restate the registry comments.
``tests/test_machine_datasheet.py`` checks both, and checks that every scalar
field of the machine and its site scenario appears in the sheet.

Source: category vocabulary from the Volume IV Reader Guide
(``books/vol4/frontmatter/prerequisites.qmd``, #sec-reader-numbers); values
from ``embodied/registry.py``, ``actuators/registry.py``, ``sensors`` and
``hardware`` registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..core.units import ureg

MEASURED = "Measured"
DERIVED = "Derived"
DATASHEET = "Datasheet"
ILLUSTRATIVE = "Illustrative"
CHOSEN = "Chosen"
CATEGORIES = (MEASURED, DERIVED, DATASHEET, ILLUSTRATIVE, CHOSEN)

#: Provenance kind -> Reader Guide category, for a leaf value that carries its own
#: provenance (a sourced quantity): a published literature value is a measurement
#: (of people, drivers, or a reference board, never of this machine).
PROVENANCE_CATEGORY = {"datasheet": DATASHEET, "literature": MEASURED}

#: Provenance kind -> category for a field of a sourced component (an arm, a
#: motor, a camera, a model): the maker's published specification.
OWNER_PROVENANCE_CATEGORY = {"datasheet": DATASHEET, "literature": DATASHEET}

NEEDS_SOURCE = "needs source"

# Subsystem groups, in the order the sheet prints them.
BODY = "Body: base, arm, and drivetrain"
POWER = "Body: drive bus and permission rail"
BRAIN = "Brain: proposers and application processor"
NERVOUS = "Nervous system: timing and leases"
SENSING = "Sensing and observation age"
STOP = "Stopping budget at the rack end"
CONTACT = "Contact, conveyor, and coworker"
HUMAN = "Human takeover"
SUBSYSTEMS = (BODY, POWER, BRAIN, NERVOUS, SENSING, STOP, CONTACT, HUMAN)


@dataclass(frozen=True)
class DataSheetSpec:
    """One row definition: where the value lives and how the sheet shows it."""

    key: str
    subsystem: str
    label: str                  # reader-facing name; may carry a LaTeX symbol
    field: str                  # MLSysIM path: RM.* (machine) or AISLE.* (site scenario)
    category: str
    unit: Optional[str] = None  # display unit; None for counts and dimensionless values
    precision: Optional[int] = None
    note: str = ""
    compute: Optional[Callable[[Any, Any], Any]] = None  # derived rows: f(machine, scenario)


@dataclass(frozen=True)
class DataSheetRow:
    """A resolved row: the spec plus its value and display string."""

    key: str
    subsystem: str
    label: str
    field: str
    category: str
    value: Any
    value_str: str          # MarkdownStr
    category_str: str       # MarkdownStr of ``category``
    note: str


def _j_brake(rm, aisle):
    from ..physics.robotics import calc_c2_stop_suffix  # avoid import cycle

    return calc_c2_stop_suffix(rm.base.max_velocity, aisle.a_brake)["peak_jerk"]


def _mu_required(rm, aisle):
    from ..physics.drivetrain import calc_required_friction  # avoid import cycle

    return calc_required_friction(aisle.a_brake)


def _mu_required_rated_share(rm, aisle):
    from ..physics.drivetrain import calc_required_friction  # avoid import cycle

    return calc_required_friction(aisle.a_brake, rm.drive_wheel_load_share_max)


_S = DataSheetSpec

#: The full sheet, grouped by subsystem.
SPECS: tuple[DataSheetSpec, ...] = (
    # ── Body: base, arm, and drivetrain ─────────────────────────────────────
    _S("m_base", BODY, "Base mass", "RM.base.mass", ILLUSTRATIVE, "kg", 0,
       "Class-profile value, gross"),
    _S("m_arm", BODY, "Arm mass", "RM.arm.mass", DATASHEET, "kg", 0, "Panda-class arm"),
    _S("m_payload", BODY, "Tote-rack payload capacity", "RM.onboard_payload_capacity", CHOSEN, "kg", 0),
    _S("m_loaded", BODY, "Loaded mass $m$", "RM.loaded_mass", DERIVED, "kg", 0,
       "Upper bound: base, arm, full tote rack"),
    _S("footprint", BODY, "Footprint width", "RM.footprint_width", ILLUSTRATIVE, "m", 2),
    _S("v_drive", BODY, "Drive limit $v_{\\max}$", "RM.base.max_velocity", ILLUSTRATIVE, "m/s", 1,
       "Class-profile value"),
    _S("a_brake", BODY, "Credible deceleration $a_{\\text{brake}}$", "AISLE.a_brake", ILLUSTRATIVE,
       "m/s**2", 1, "Loaded, dry floor; checked against traction and drive torque"),
    _S("mu_req", BODY, "Friction $a_{\\text{brake}}$ needs, all weight on driven wheels",
       "calc_required_friction(AISLE.a_brake)", DERIVED, None, 3, compute=_mu_required),
    _S("load_share_max", BODY, "Largest driven-wheel load share $\\varphi$",
       "RM.drive_wheel_load_share_max", DERIVED, None, 3, "Hub motors' load rating over loaded mass"),
    _S("mu_req_rated", BODY, "Friction $a_{\\text{brake}}$ needs at that share",
       "calc_required_friction(AISLE.a_brake, RM.drive_wheel_load_share_max)", DERIVED, None, 3,
       compute=_mu_required_rated_share),
    _S("load_share", BODY, "Measured driven-wheel load share", "RM.drive_wheel_load_share", MEASURED,
       None, None, "Needs source: corner-scale weighing of the loaded machine"),
    _S("cg_height", BODY, "Loaded center-of-gravity height $h_{\\text{cg}}$", "RM.cg_height", MEASURED,
       "m", None, "Needs source: tilt-table or corner-scale measurement; tip-over unchecked"),
    _S("tip_arm", BODY, "Center of gravity to leading support, arm stowed", "RM.tip_arm_stowed",
       MEASURED, "m", None, "Needs source: measured with the center-of-gravity height"),
    _S("j_brake", BODY, "Stop jerk bound $j_{\\max}$", "calc_c2_stop_suffix(v_max, a_brake)['peak_jerk']",
       DERIVED, "m/s**3", 2, "Stipulated: the reference polynomial stop's peak jerk", compute=_j_brake),
    _S("wheel_radius", BODY, "Drive wheel radius", "RM.drive_unit.wheel_radius", DATASHEET, "mm", 0,
       "Hub motor"),
    _S("wheel_count", BODY, "Driven wheels", "RM.drive_wheel_count", CHOSEN, None, None),
    _S("k_t_wheel", BODY, "Hub-motor torque constant $K_t$", "RM.drive_unit.torque_constant", DERIVED,
       "N*m/A", 1, "Datasheet peak torque over peak current"),
    _S("i_drive", BODY, "Drive current limit", "RM.drive_current_limit", CHOSEN, "A", 0,
       "Set at the motor's datasheet peak"),
    _S("i_rated", BODY, "Hub-motor rated current", "RM.drive_unit.rated_current", DATASHEET, "A", 1),
    _S("load_pair", BODY, "Hub-motor load rating per pair", "RM.drive_unit.load_rating_per_pair",
       DATASHEET, "kg", 0),
    _S("rotor_inertia", BODY, "Hub-motor rotor inertia", "RM.drive_unit.rotor_inertia", DATASHEET,
       "kg*m**2", None, "Needs source: not published; motor mass bounds $J/r^2$"),
    _S("arm_torque", BODY, "Arm repeatable peak joint torque", "RM.arm.max_torque", DATASHEET, "N*m", 0),
    _S("tcp_limit", BODY, "Arm free-space TCP speed limit", "RM.tcp_speed_limit", CHOSEN, "m/s", 1),
    _S("servo_axes", BODY, "Servo axes on the fieldbus", "RM.servo_axes", CHOSEN, None, None,
       "Seven arm joints and two drive wheels"),
    # ── Body: drive bus and permission rail ─────────────────────────────────
    _S("v_bus_full", POWER, "Drive bus, full charge", "RM.drive_bus_full_charge", DATASHEET, "V", 0,
       "48 V-class pack"),
    _S("v_bus_nom", POWER, "Drive bus, nominal", "RM.drive_bus_nominal", DATASHEET, "V", 1),
    _S("v_bus_low", POWER, "Drive bus, depleted", "RM.drive_bus_battery_low", DATASHEET, "V", 0),
    _S("r_bus", POWER, "Drive-bus source resistance $R_{\\text{bus}}$", "RM.drive_bus_resistance",
       ILLUSTRATIVE, "milliohm", 0, "Needs a measured cell, BMS, fuse, and harness value"),
    _S("v_uvlo", POWER, "Converter input lockout", "RM.drive_bus_uvlo", DATASHEET, "V", 0,
       "48-to-24 V converter feeding the application processor"),
    _S("v_conv_max", POWER, "Converter continuous input maximum", "RM.drive_bus_converter_max",
       DATASHEET, "V", 1),
    _S("eff_conv", POWER, "Converter efficiency", "RM.drive_bus_converter_efficiency", DATASHEET, None, 2),
    _S("v_clamp", POWER, "Brake-chopper clamp", "RM.drive_bus_clamp", CHOSEN, "V", 0),
    _S("v_switch", POWER, "Inverter switch rating", "RM.drive_bus_switch_rating", CHOSEN, "V", 0),
    _S("i_transient", POWER, "Coincident inference and drive current step", "AISLE.i_rail_transient",
       ILLUSTRATIVE, "A", 0),
    _S("holdup", POWER, "Permission-rail hold-up", "RM.permission_rail_holdup", CHOSEN, "s", 0),
    _S("p_perm", POWER, "Permission-rail load", "RM.permission_rail_load", ILLUSTRATIVE, "W", 0),
    _S("v_store_low", POWER, "Permission store, lowest charge", "RM.control_rail_battery_low",
       ILLUSTRATIVE, "V", 1),
    _S("v_store_dropout", POWER, "Permission-load regulator dropout", "RM.control_rail_dropout",
       ILLUSTRATIVE, "V", 0),
    _S("brake_min", POWER, "Spring-brake engage time, fast end", "RM.spring_brake_engage_min",
       ILLUSTRATIVE, "ms", 0),
    _S("brake_max", POWER, "Spring-brake engage time, slow end", "RM.spring_brake_engage_max",
       ILLUSTRATIVE, "ms", 0),
    # ── Brain: proposers and application processor ──────────────────────────
    _S("bw_soc", BRAIN, "Application-processor memory bandwidth", "RM.brain_soc.memory.bandwidth",
       DATASHEET, "GB/s", 0),
    _S("dram_eff", BRAIN, "Sustained fraction of that bandwidth", "RM.dram_efficiency", ILLUSTRATIVE,
       None, 2),
    _S("intent_params", BRAIN, "Intent-model parameters", "RM.intent_model.parameters", DATASHEET,
       "param", None, "OpenVLA-7B"),
    _S("intent_bytes", BRAIN, "Intent-model stored weight precision", "RM.intent_weight_bytes_per_param",
       CHOSEN, "byte/param", 1, "INT4 weight-only quantization"),
    _S("intent_tokens", BRAIN, "Intent-model input tokens", "RM.intent_input_tokens", ILLUSTRATIVE,
       None, None),
    _S("intent_util", BRAIN, "Intent-model prefill utilization", "RM.intent_compute_utilization",
       ILLUSTRATIVE, None, 2),
    _S("t_intent", BRAIN, "Intent-model latency", "RM.intent_inference_latency", DERIVED, "ms", 0,
       "Prefill plus decode"),
    _S("t_intent_ref", BRAIN, "Published intent-model latency, same board", "RM.intent_reference_latency",
       MEASURED, "ms", 0, "Published measurement, not this machine"),
    _S("f_intent", BRAIN, "Intent-model rate", "RM.intent_rate", CHOSEN, "Hz", 0,
       "Validated against the computed latency"),
    _S("chunk_params", BRAIN, "Chunk-policy parameters", "RM.chunk_model.parameters", DATASHEET,
       "param", None, "ACT"),
    _S("t_chunk", BRAIN, "Chunk-policy inference, P99", "RM.chunk_inference_latency", ILLUSTRATIVE, "ms", 0),
    _S("f_chunk", BRAIN, "Chunk-policy renewal rate", "RM.chunk_rate", CHOSEN, "Hz", 0),
    _S("t_period", BRAIN, "Chunk renewal (replan) period", "RM.chunk_period", DERIVED, "ms", 0,
       "Whole number of setpoint steps"),
    _S("t_step", BRAIN, "Setpoint step $\\Delta t$", "RM.chunk_step", CHOSEN, "ms", 0),
    _S("horizon", BRAIN, "Chunk horizon $H$", "RM.chunk_horizon", CHOSEN, None, None),
    # ── Nervous system: timing and leases ───────────────────────────────────
    _S("f_perm", NERVOUS, "Permission-loop rate", "RM.permission_rate", CHOSEN, "kHz", 0),
    _S("t_tick", NERVOUS, "Permission-loop period $T_{\\text{tick}}$", "AISLE.t_tick", DERIVED, "ms", 0),
    _S("f_current", NERVOUS, "Drive current-loop rate", "RM.current_loop_rate", CHOSEN, "kHz", 0),
    _S("t_bus", NERVOUS, "Fieldbus cycle $T_{\\text{bus}}$", "AISLE.t_bus", ILLUSTRATIVE, "ms", 0),
    _S("t_lease", NERVOUS, "Chunk lease $T_{\\text{lease}}$", "AISLE.t_lease", CHOSEN, "ms", 0),
    _S("t_act", NERVOUS, "Brake onset $T_{\\text{act}}$", "AISLE.t_brake_onset", ILLUSTRATIVE, "ms", 0),
    _S("f_heartbeat", NERVOUS, "Heartbeat rate", "AISLE.heartbeat_rate", CHOSEN, "Hz", 0),
    _S("t_heartbeat", NERVOUS, "Heartbeat timeout", "AISLE.heartbeat_timeout", CHOSEN, "ms", 0),
    _S("t_watchdog", NERVOUS, "Safety-MCU self-watchdog", "AISLE.mcu_self_watchdog", CHOSEN, "ms", 0),
    _S("t_enf_unloaded", NERVOUS, "Permission check, unloaded maximum", "AISLE.enforcer_unloaded",
       ILLUSTRATIVE, "us", 0),
    _S("t_enf_wcet", NERVOUS, "Permission check, declared WCET", "AISLE.enforcer_wcet", CHOSEN, "us", 0),
    _S("t_enf_deadline", NERVOUS, "Permission check deadline", "AISLE.enforcer_deadline", CHOSEN, "us", 0),
    _S("t_enf_blocking", NERVOUS, "Permission check, blocking allowance", "AISLE.enforcer_blocking", ILLUSTRATIVE, "us", 0),
    _S("t_enf_interference", NERVOUS, "Permission check, interference allowance", "AISLE.enforcer_interference", ILLUSTRATIVE, "us", 0),
    _S("chunk_bytes", NERVOUS, "Serialized chunk size", "AISLE.chunk_wire_bytes", ILLUSTRATIVE, "byte", 0,
       "Waypoints, header, and evidence epoch"),
    _S("f_link", NERVOUS, "Board-link clock", "AISLE.board_link_clock", ILLUSTRATIVE, "MHz", 0,
       "SPI, one bit per clock"),
    _S("t_link", NERVOUS, "Board-link serialization", "AISLE.t_board_link", DERIVED, "ms", 2,
       "Chunk bits over the link clock"),
    _S("t_deser", NERVOUS, "Chunk deserialization", "AISLE.t_chunk_deserialize", ILLUSTRATIVE, "ms", 0),
    _S("t_validate", NERVOUS, "Whole-plan validation", "AISLE.t_plan_validate", ILLUSTRATIVE, "ms", 0),
    _S("t_activate", NERVOUS, "Plan activation", "AISLE.t_plan_activate", ILLUSTRATIVE, "ms", 0),
    _S("t_da", NERVOUS, "Dispatch-to-admission allocation $\\Delta_{\\text{da}}$",
       "AISLE.t_dispatch_to_admission", DERIVED, "ms", 1,
       "Inference, link, deserialize, validate, activate, one tick"),
    # ── Sensing and observation age ─────────────────────────────────────────
    _S("t_exposure", SENSING, "Navigation-camera exposure", "RM.nav_camera.exposure_time", CHOSEN,
       "ms", 0, "Configured setting, not a sensor property"),
    _S("t_readout", SENSING, "Navigation-camera readout", "RM.nav_camera.readout_time", DATASHEET, "ms", 1),
    _S("t_transport", SENSING, "Sensor transport", "AISLE.t_transport", ILLUSTRATIVE, "ms", 1),
    _S("t_dma", SENSING, "DMA transfer", "AISLE.t_dma", ILLUSTRATIVE, "ms", 1),
    _S("t_isp", SENSING, "Image signal processing", "AISLE.t_isp", ILLUSTRATIVE, "ms", 1),
    _S("t_backbone", SENSING, "Perception backbone", "AISLE.t_backbone", ILLUSTRATIVE, "ms", 0),
    _S("t_ipc", SENSING, "Hand-off to the proposer", "AISLE.t_ipc", ILLUSTRATIVE, "ms", 1),
    _S("t_age", SENSING, "Observation age at dispatch $t_{\\text{age}}$", "AISLE.perception_age",
       DERIVED, "ms", 1, "Exposure midpoint to dispatch"),
    _S("yaw_rate_turn", SENSING, "Yaw rate turning into the aisle $\\YawRate$", "AISLE.yaw_rate_turn", ILLUSTRATIVE,
       "rad/s", 1, "Sets rolling-shutter and sweep skew"),
    _S("eps_sync", SENSING, "Clock-conversion bound $\\ClockConversionBound$", "AISLE.clock_conversion_bound",
       ILLUSTRATIVE, "ms", 1, "Capture clock to the safety MCU's clock"),
    # ── Stopping budget at the rack end ─────────────────────────────────────
    _S("tau_delay", STOP, "Total pre-brake delay $\\tau_{\\text{delay}}$", "AISLE.tau_delay", DERIVED,
       "ms", 1, "$t_{\\text{age}} + T_{\\text{lease}} + T_{\\text{tick}} + T_{\\text{bus}} + T_{\\text{act}}$"),
    _S("d_clear", STOP, "Rack-end clear distance $D_{\\text{clear}}$", "AISLE.d_clear", ILLUSTRATIVE, "m", 2),
    _S("delta_loc", STOP, "Localization bound $\\delta_{\\text{loc}}$", "AISLE.delta_loc", ILLUSTRATIVE,
       "mm", 0),
    _S("delta_margin", STOP, "Protective clearance $\\delta_{\\text{margin}}$", "AISLE.delta_margin",
       CHOSEN, "mm", 0),
    _S("eps_track", STOP, "Tracking-error bound $\\epsilon_{\\text{track}}$", "AISLE.eps_track",
       ILLUSTRATIVE, "mm", 0),
    _S("v_aisle", STOP, "Aisle speed $v_{\\text{aisle}}$", "AISLE.v_aisle", CHOSEN, "m/s", 1,
       "Below the derived dry-floor ceiling"),
    _S("aisle_width", STOP, "Aisle width", "AISLE.aisle_width", ILLUSTRATIVE, "m", 2),
    _S("aisle_length", STOP, "Aisle length", "AISLE.aisle_length", ILLUSTRATIVE, "m", 0),
    _S("side_clearance", STOP, "Side clearance", "AISLE.side_clearance", ILLUSTRATIVE, "m", 2),
    _S("mu_dry", STOP, "Floor friction, dry", "AISLE.mu_dry", ILLUSTRATIVE, None, 2),
    _S("mu_insp", STOP, "Floor friction, inspected floor", "AISLE.mu_inspected_floor", ILLUSTRATIVE, None, 2),
    _S("mu_oil", STOP, "Floor friction, oil film", "AISLE.mu_oil_film", ILLUSTRATIVE, None, 2),
    _S("v_human", STOP, "Human approach speed $v_h$", "AISLE.v_human_approach", DATASHEET, "m/s", 1,
       "ISO 13855 value"),
    _S("t_scanner", STOP, "Safety scanner response time", "RM.safety_scanner.response_time",
       DATASHEET, "ms", 0, "Keyence SZ-V32, high-speed mode, 2 scans; Type 3, PL d"),
    _S("d_scanner", STOP, "Safety scanner protective-field range", "RM.safety_scanner.protective_field_range",
       DATASHEET, "m", 1, "Same mode, 70/150 mm objects"),
    # ── Contact, conveyor, and coworker ─────────────────────────────────────
    _S("k_latch", CONTACT, "Latch plate stiffness", "AISLE.k_latch", ILLUSTRATIVE, "N/mm", 0),
    _S("v_latch", CONTACT, "Latch approach speed", "AISLE.v_latch_approach", ILLUSTRATIVE, "m/s", 2),
    _S("v_latch_high", CONTACT, "Fast latch approach speed", "AISLE.v_latch_high", ILLUSTRATIVE, "m/s", 2),
    _S("f_trip", CONTACT, "Latch force tripwire", "AISLE.f_latch_tripwire", ILLUSTRATIVE, "N", 0),
    _S("t_contact", CONTACT, "Contact response time", "AISLE.t_contact_response", ILLUSTRATIVE, "ms", 0),
    _S("f_latch", CONTACT, "Latch force limit", "AISLE.f_latch_limit", ILLUSTRATIVE, "N", 0),
    _S("v_conveyor", CONTACT, "Conveyor speed", "AISLE.v_conveyor", ILLUSTRATIVE, "m/s", 2),
    _S("a_slip", CONTACT, "Mug slip acceleration on the conveyor", "AISLE.a_conveyor_slip", ILLUSTRATIVE,
       "m/s**2", 1),
    _S("r_tol", CONTACT, "Grasp tolerance", "AISLE.grasp_tolerance", ILLUSTRATIVE, "mm", 0),
    _S("e_0", CONTACT, "Initial grasp-estimate error", "AISLE.grasp_initial_error", ILLUSTRATIVE, "mm", 0),
    _S("k_human", CONTACT, "Coworker hand spring constant $k_H$", "AISLE.k_contact_human", DATASHEET, "N/mm", 0,
       "ISO/TS 15066 hand body model"),
    _S("m_human", CONTACT, "Coworker hand effective mass $m_H$", "AISLE.m_contact_human", DATASHEET, "kg", 1,
       "ISO/TS 15066 hand body model"),
    _S("f_contact", CONTACT, "Coworker contact force criterion", "AISLE.f_contact_criterion", CHOSEN,
       "N", 0),
    _S("v_handover", CONTACT, "Handover TCP speed", "AISLE.v_handover", CHOSEN, "m/s", 2),
    # ── Human takeover ──────────────────────────────────────────────────────
    _S("t_perc", HUMAN, "Perceptual cycle, nominal", "AISLE.t_human_perceptual", MEASURED, "ms", 0,
       "Model Human Processor"),
    _S("t_perc_slow", HUMAN, "Perceptual cycle, slow bound", "AISLE.t_human_perceptual_slow", MEASURED,
       "ms", 0),
    _S("t_cog", HUMAN, "Cognitive cycle, nominal", "AISLE.t_human_cognitive", MEASURED, "ms", 0),
    _S("t_cog_slow", HUMAN, "Cognitive cycle, slow bound", "AISLE.t_human_cognitive_slow", MEASURED, "ms", 0),
    _S("t_motor", HUMAN, "Motor cycle, nominal", "AISLE.t_human_motor", MEASURED, "ms", 0),
    _S("t_motor_slow", HUMAN, "Motor cycle, slow bound", "AISLE.t_human_motor_slow", MEASURED, "ms", 0),
    _S("t_brake_exp", HUMAN, "Perception-brake time, expected signal", "AISLE.t_brake_response_expected",
       MEASURED, "s", 2, "Drivers"),
    _S("t_brake_unexp", HUMAN, "Perception-brake time, unexpected signal",
       "AISLE.t_brake_response_unexpected", MEASURED, "s", 2, "Drivers"),
    _S("t_brake_surp", HUMAN, "Perception-brake time, surprise", "AISLE.t_brake_response_surprise",
       MEASURED, "s", 1, "Drivers"),
    _S("t_to_p50", HUMAN, "Take-over time, median condition", "AISLE.t_takeover_driving_p50", MEASURED,
       "s", 2, "Percentiles of condition means, not of individuals"),
    _S("t_to_mean", HUMAN, "Take-over time, mean", "AISLE.t_takeover_driving_mean", MEASURED, "s", 2),
    _S("t_to_p90", HUMAN, "Take-over time, 90th percentile condition", "AISLE.t_takeover_driving_p90",
       MEASURED, "s", 2),
    _S("t_to_p99", HUMAN, "Take-over time, 99th percentile condition", "AISLE.t_takeover_driving_p99",
       MEASURED, "s", 1),
    _S("t_video", HUMAN, "Console video, glass to glass", "AISLE.t_video_glass_to_glass", MEASURED, "ms", 0,
       "Wired best case; the site's link needs a measured value"),
    _S("t_hs_req", HUMAN, "Handshake: request", "AISLE.t_handshake_request", CHOSEN, "ms", 0),
    _S("t_hs_ack", HUMAN, "Handshake: acknowledge", "AISLE.t_handshake_acknowledge", CHOSEN, "ms", 0),
    _S("t_hs_conf", HUMAN, "Handshake: confirm", "AISLE.t_handshake_confirm", CHOSEN, "ms", 0),
    _S("t_handshake", HUMAN, "Four-phase handshake", "AISLE.t_handshake", DERIVED, "ms", 0,
       "Three phases plus the commit tick"),
    _S("t_takeover_in", HUMAN, "In-loop takeover $T_{\\Sigma,1}$", "AISLE.t_takeover_in_loop", DERIVED,
       "ms", 0, "Slow-bound reaction plus video and handshake"),
    _S("t_takeover_out", HUMAN, "Out-of-loop takeover $T_{\\Sigma,2}$", "AISLE.t_takeover_out_of_loop",
       DERIVED, "s", 2, "90th-percentile condition plus video and handshake"),
    _S("v_takeover_res", HUMAN, "Slow-down speed resolution", "AISLE.v_takeover_resolution", CHOSEN,
       "m/s", 2),
    _S("v_takeover_in_ceil", HUMAN, "In-loop slow-down ceiling with $\\Delta_{\\text{da}}$",
       "AISLE.v_takeover_in_loop_ceiling", DERIVED, "m/s", 3,
       "S-curve ceiling at $\\tau_{\\text{delay}} + T_{\\Sigma,1} + \\Delta_{\\text{da}}$"),
    _S("v_takeover_in", HUMAN, "Speed before an in-loop request", "AISLE.v_takeover_in_loop", DERIVED,
       "m/s", 2, "Ceiling rounded down to the resolution"),
    _S("v_takeover_out", HUMAN, "Crawl before an out-of-loop request", "AISLE.v_takeover_out_of_loop",
       CHOSEN, "m/s", 2),
    _S("t_pre", HUMAN, "Pre-trigger logging window", "AISLE.pre_trigger_window", DERIVED, "s", 0,
       "Out-of-loop takeover rounded up"),
)

#: Keys of the Chapter 1 machine card, in the order the card prints them.
CARD_KEYS: tuple[str, ...] = (
    "m_loaded", "v_drive", "v_aisle", "a_brake", "j_brake", "d_clear", "tau_delay",
    "t_tick", "t_lease", "t_step", "t_period", "t_intent", "f_intent",
    "t_takeover_in", "t_takeover_out",
)


def _default_scenario():
    from .registry import Embodied  # avoid import cycle

    return Embodied.Scenario.WarehouseAisle


def resolve_field(path: str, machine, scenario):
    """Follow an ``RM.``/``AISLE.`` attribute path to its registry leaf."""
    root, *attrs = path.split(".")
    obj = {"RM": machine, "AISLE": scenario}[root]
    for name in attrs:
        obj = getattr(obj, name)
    return obj


_UNIT_LABELS = {"N*m": "N·m", "N*m/A": "N·m/A", "kg*m**2": "kg·m²"}


def _round_half_up(x: float, places: int) -> float:
    """Round as a reader would (4.465 -> 4.47), from the shortest decimal repr."""
    from decimal import ROUND_HALF_UP, Decimal

    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def format_value(value, unit: Optional[str], precision: Optional[int]) -> str:
    """Display string for one sheet value; ``needs source`` when the registry holds None.

    Returns a ``MarkdownStr`` so a Quarto inline reference prints the text itself.
    """
    from ..fmt import MarkdownStr, fmt, fmt_int, fmt_params, fmt_qty  # avoid import cycle

    def _prec(magnitude):
        # An integer-like value prints without decimals (the fmt precision guard
        # rejects spurious trailing zeros such as 1.00 V).
        if precision is not None and float(magnitude).is_integer():
            return 0
        return precision

    if value is None:
        return MarkdownStr(NEEDS_SOURCE)
    if isinstance(value, bool):
        raise TypeError("boolean sheet values are not supported")
    if isinstance(value, ureg.Quantity):
        if unit is None:
            raise ValueError(f"a Quantity needs a display unit: {value!r}")
        if unit == "param":
            return MarkdownStr(fmt_params(value))
        display_unit = ureg.Unit(unit)
        magnitude = value.to(display_unit).magnitude
        prec = precision
        if prec is not None:
            # Round to the displayed precision first; fmt refuses to hide a
            # fraction silently, and a data sheet shows its value at a stated
            # precision (the registry keeps the full value).
            magnitude = _round_half_up(magnitude, prec)
            prec = _prec(magnitude)
        return MarkdownStr(
            fmt_qty(
                magnitude * display_unit,
                display_unit,
                precision=prec,
                commas=True,
                unit_label=_UNIT_LABELS.get(unit),
            )
        )
    if isinstance(value, int):
        return MarkdownStr(fmt_int(value))
    return MarkdownStr(fmt(float(value), precision=_prec(value)))


def machine_data_sheet(scenario=None) -> list[DataSheetRow]:
    """Resolve every sheet row against the registry (the warehouse aisle by default)."""
    from ..fmt import MarkdownStr  # avoid import cycle

    aisle = scenario if scenario is not None else _default_scenario()
    machine = aisle.machine
    rows = []
    for spec in SPECS:
        if spec.compute is not None:
            value = spec.compute(machine, aisle)
        else:
            value = resolve_field(spec.field, machine, aisle)
        rows.append(
            DataSheetRow(
                key=spec.key,
                subsystem=spec.subsystem,
                label=spec.label,
                field=spec.field,
                category=spec.category,
                value=value,
                value_str=format_value(value, spec.unit, spec.precision),
                category_str=MarkdownStr(spec.category),
                note=spec.note,
            )
        )
    return rows


def machine_card(scenario=None) -> list[DataSheetRow]:
    """The Chapter 1 machine card: the sheet rows students use most, in card order."""
    by_key = {row.key: row for row in machine_data_sheet(scenario)}
    return [by_key[k] for k in CARD_KEYS]


def data_sheet_lookup(scenario=None) -> dict[str, DataSheetRow]:
    """The sheet keyed by row key, for LEGO cells that print it."""
    return {row.key: row for row in machine_data_sheet(scenario)}
