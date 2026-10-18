"""Stopping envelopes, S-curve trajectories, and velocity bounds for Physical AI."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_nonnegative, validate_positive
from .actuation import calc_tripwire_contact_force_accumulation


def calc_sensor_to_actuator_latency(
    sensor_latency,
    inference_latency,
    arbitration_latency,
    actuator_latency,
):
    """
    Calculate the total end-to-end perception-to-actuation loop latency.

    Source: modeling assumption. The stages are summed as a serial pipeline,
    not taken from a published model; sampling phase, jitter, and queueing
    between stages are ignored.

    Parameters
    ----------
    sensor_latency : Quantity
        Sensor capture and transmission delay (e.g. camera frame exposure + USB/CSI bus).
    inference_latency : Quantity
        Policy inference duration on neural accelerator (e.g. VLA, Diffusion Policy).
    arbitration_latency : Quantity
        Deterministic nervous system / Control Barrier Function arbitration delay.
    actuator_latency : Quantity
        Motor controller response time and CAN/EtherCAT bus transit delay.

    Returns
    -------
    Quantity
        Total closed-loop latency in milliseconds or seconds.
    """
    total = sensor_latency + inference_latency + arbitration_latency + actuator_latency
    return total.to(ureg.millisecond)


def calc_safe_stopping_distance(
    velocity,
    total_latency,
    max_deceleration,
    loc_margin=None,
    physical_margin=None,
):
    """
    Calculate dynamic stopping distance under non-zero control latency.

    Equation:
        d_stop = v * tau_latency + (v^2) / (2 * a_max) + delta_loc + delta_margin

    Source: reaction distance v * tau plus constant-deceleration braking
    distance v^2 / (2 * a), from v^2 = v0^2 + 2 * a * (x - x0) (OpenStax,
    University Physics Volume 1, Sec. 3.4, Eq. 3.14; Example 3.10 adds
    reaction-time travel to braking distance the same way). The additive
    localization and clearance margins are a modeling assumption.

    Parameters
    ----------
    velocity : Quantity
        Current operational velocity (e.g. m/s).
    total_latency : Quantity
        End-to-end perception-to-brake latency (e.g. ms or s).
    max_deceleration : Quantity
        Maximum braking deceleration available (e.g. m/s^2).
    loc_margin : Quantity, optional
        Localization uncertainty bound (e.g. cm or m).
    physical_margin : Quantity, optional
        Physical obstacle clearance safety margin (e.g. cm or m).

    Returns
    -------
    Quantity
        Total stopping distance in meters.
    """
    validate_nonnegative(velocity, "velocity")
    validate_positive(total_latency, "total_latency")
    validate_positive(max_deceleration, "max_deceleration")

    reaction_distance = velocity * total_latency
    braking_distance = (velocity ** 2) / (2 * max_deceleration)
    total_distance = reaction_distance + braking_distance

    if loc_margin is not None:
        validate_nonnegative(loc_margin, "loc_margin")
        total_distance = total_distance + loc_margin
    if physical_margin is not None:
        validate_nonnegative(physical_margin, "physical_margin")
        total_distance = total_distance + physical_margin

    return total_distance.to(ureg.meter)


def calc_max_permitted_velocity(
    d_clear,
    total_latency,
    max_deceleration,
    loc_margin=None,
    physical_margin=None,
):
    """
    Calculate maximum safe forward velocity admitted by physical stopping clearance.

    Solves the quadratic inequality:
        v * tau + v^2 / (2 * a) <= D_clear - delta_loc - delta_margin
    for the maximum non-negative velocity root:
        v_max = -a * tau + sqrt((a * tau)^2 + 2 * a * D_eff)
    where D_eff = D_clear - delta_loc - delta_margin. If D_eff <= 0, v_max = 0.

    Source: derived here as the positive root of the
    calc_safe_stopping_distance quadratic, so it rests on the same kinematics
    (OpenStax, University Physics Volume 1, Sec. 3.4, Eq. 3.14) and the same
    additive-margin modeling assumption.

    Parameters
    ----------
    d_clear : Quantity
        Detected clearance distance to obstacle along travel vector (e.g. m).
    total_latency : Quantity
        End-to-end sense-to-actuation latency tau_delay (e.g. s or ms).
    max_deceleration : Quantity
        Maximum braking deceleration capacity a_brake (e.g. m/s^2).
    loc_margin : Quantity, optional
        Localization uncertainty bound (e.g. cm or m).
    physical_margin : Quantity, optional
        Physical clearance safety buffer (e.g. cm or m).

    Returns
    -------
    Quantity
        Maximum safe forward velocity in meters/second.
    """
    validate_nonnegative(d_clear, "d_clear")
    validate_positive(total_latency, "total_latency")
    validate_positive(max_deceleration, "max_deceleration")

    d_eff = d_clear
    if loc_margin is not None:
        validate_nonnegative(loc_margin, "loc_margin")
        d_eff = d_eff - loc_margin
    if physical_margin is not None:
        validate_nonnegative(physical_margin, "physical_margin")
        d_eff = d_eff - physical_margin

    if d_eff.to(ureg.meter).magnitude <= 0.0:
        return (0.0 * (ureg.meter / ureg.second)).to(ureg.meter / ureg.second)

    a_tau = max_deceleration * total_latency
    discriminant = (a_tau ** 2) + 2.0 * max_deceleration * d_eff
    v_max = -a_tau + discriminant ** 0.5

    v_mps = max(0.0, v_max.to(ureg.meter / ureg.second).magnitude)
    return (v_mps * (ureg.meter / ureg.second)).to(ureg.meter / ureg.second)


def calc_c2_stop_suffix(entry_velocity, peak_deceleration):
    """
    Calculate the duration and distance of a C2-continuous stop from cruise.

    A stop that begins and ends at zero acceleration, entered from constant
    velocity v_c, follows the minimum-order polynomial

        q(u) = q_c + v_c * T * (u - u^3 + u^4 / 2),  u = t / T in [0, 1],

    whose peak deceleration is 1.5 * v_c / T and whose distance is v_c * T / 2.
    Setting the peak equal to the credible deceleration a gives

        T = 1.5 * v / a,    d = 0.75 * v^2 / a,

    which is 1.5 times the constant-deceleration distance v^2 / (2 * a).
    The factor belongs to this polynomial, not to smooth stopping in general:
    its jerk q''' = -(6 v_c / T^2)(1 - 2u) peaks in magnitude at the ends,

        j_peak = 6 * v / T^2 = (8 / 3) * a^2 / v,

    and a jerk-limited S-curve with the same a and j_peak stops shorter
    (calc_scurve_stop_distance).

    Source: derived here. The polynomial satisfies q' = v_c, q'' = 0 at u = 0
    and q' = 0, q'' = 0 at u = 1; peak |q''| occurs at u = 1/2 (minimum of
    q''(u) = v_c / T * (-6u + 6u^2)); peak |q'''| occurs at u = 0 and u = 1.

    Parameters
    ----------
    entry_velocity : Quantity
        Cruise velocity at the start of the stop (e.g. m/s).
    peak_deceleration : Quantity
        Credible peak deceleration the stop may reach (e.g. m/s^2).

    Returns
    -------
    dict
        duration (ms), distance (m), and peak_jerk (m/s^3).
    """
    validate_positive(entry_velocity, "entry_velocity")
    validate_positive(peak_deceleration, "peak_deceleration")

    v = entry_velocity.to(ureg.meter / ureg.second)
    a = peak_deceleration.to(ureg.meter / ureg.second**2)
    duration = 1.5 * v / a
    return {
        "duration": duration.to(ureg.millisecond),
        "distance": (0.75 * v**2 / a).to(ureg.meter),
        "peak_jerk": (6.0 * v / duration**2).to(ureg.meter / ureg.second**3),
    }


def _scurve_stop_si(v, a, j):
    """Duration (s) and distance (m) of a jerk-limited stop, SI magnitudes."""
    if v <= 0.0:
        return 0.0, 0.0
    if v >= a * a / j:
        # Trapezoidal acceleration: ramp to a in a/j, hold, ramp back to zero.
        t = v / a + a / j
    else:
        # Triangular acceleration: the peak sqrt(v j) stays below a.
        t = 2.0 * math.sqrt(v / j)
    # The velocity profile is point-symmetric about its midpoint, so d = v T / 2.
    return t, 0.5 * v * t


def _scurve_si_inputs(entry_velocity, max_deceleration, max_jerk):
    validate_nonnegative(entry_velocity, "entry_velocity")
    validate_positive(max_deceleration, "max_deceleration")
    validate_positive(max_jerk, "max_jerk")
    return (
        entry_velocity.to(ureg.meter / ureg.second).magnitude,
        max_deceleration.to(ureg.meter / ureg.second**2).magnitude,
        max_jerk.to(ureg.meter / ureg.second**3).magnitude,
    )


def calc_scurve_stop_distance(entry_velocity, max_deceleration, max_jerk):
    """
    Distance of a jerk-limited S-curve stop entered from zero-acceleration cruise.

    The stop drives acceleration from zero to -a at the jerk limit j, holds
    -a, and returns acceleration to zero at the jerk limit as velocity reaches
    zero (trapezoidal acceleration, the "double S" velocity profile). The
    profile is C2: position, velocity, and acceleration are continuous and
    jerk is piecewise constant.

        if v >= a^2 / j:   d = (v / 2) * (v / a + a / j)
                             = v^2 / (2 a) + v * a / (2 j)
        else:              d = v * sqrt(v / j)     (a is never reached)

    In the first case the stop equals constant-deceleration braking at a plus
    cruise over half of one jerk ramp, a / (2 j). Under the bounds |a| <= a_max
    and |j| <= j_max this is the time-optimal stop, and for a stop that does not
    reverse it is also the shortest.

    Source: Biagiotti, L. and Melchiorri, C., Trajectory Planning for Automatic
    Machines and Robots, Springer (2008; DOI 10.1007/978-3-540-85629-0),
    ch. 3, trajectory with double S velocity profile, specialized here to a
    stop from v to rest with zero boundary accelerations. Formula re-derived and
    checked against a discretized minimum-distance linear program.

    Parameters
    ----------
    entry_velocity : Quantity
        Cruise velocity at stop onset (e.g. m/s); zero gives zero distance.
    max_deceleration : Quantity
        Deceleration bound a_max (e.g. m/s^2).
    max_jerk : Quantity
        Jerk bound j_max (e.g. m/s^3).

    Returns
    -------
    Quantity
        Stopping distance in meters.
    """
    v, a, j = _scurve_si_inputs(entry_velocity, max_deceleration, max_jerk)
    return (_scurve_stop_si(v, a, j)[1] * ureg.meter).to(ureg.meter)


def calc_scurve_stop_time(entry_velocity, max_deceleration, max_jerk):
    """
    Duration of the jerk-limited S-curve stop of calc_scurve_stop_distance.

        if v >= a^2 / j:   T = v / a + a / j
        else:              T = 2 * sqrt(v / j)

    Source: Biagiotti and Melchiorri (2008), ch. 3, double S velocity profile;
    see calc_scurve_stop_distance.

    Returns
    -------
    Quantity
        Stopping time in milliseconds.
    """
    v, a, j = _scurve_si_inputs(entry_velocity, max_deceleration, max_jerk)
    return (_scurve_stop_si(v, a, j)[0] * ureg.second).to(ureg.millisecond)


def calc_scurve_max_permitted_velocity(
    d_clear,
    total_latency,
    max_deceleration,
    max_jerk,
    loc_margin=None,
    physical_margin=None,
    approach_velocity=None,
):
    """
    Highest cruise speed whose delayed S-curve stop fits the clearance.

    Solves for the largest v with

        v * tau + d_S(v) + v_h * (tau + T_S(v)) <= D_clear - delta_loc - delta_margin,

    where d_S and T_S are the S-curve stop distance and time
    (calc_scurve_stop_distance, calc_scurve_stop_time) and v_h is an optional
    obstacle approach speed (zero for a static obstacle). The left side is
    continuous and strictly increasing in v, so bisection finds the root.
    Because d_S is not a single quadratic in v, no one closed form covers both
    stop regimes; for v_h = 0 in the trapezoidal regime the result equals
    calc_max_permitted_velocity with delay tau + a / (2 j).

    Source: derived here from calc_scurve_stop_distance (Biagiotti and
    Melchiorri 2008, ch. 3) and the additive delay-plus-braking budget of
    calc_safe_stopping_distance.

    Parameters
    ----------
    d_clear : Quantity
        Clear distance to the obstacle (e.g. m).
    total_latency : Quantity
        Pre-brake delay tau (e.g. ms).
    max_deceleration, max_jerk : Quantity
        Stop limits a_max (m/s^2) and j_max (m/s^3).
    loc_margin, physical_margin : Quantity, optional
        Localization bound and protective clearance subtracted from d_clear.
    approach_velocity : Quantity, optional
        Speed at which the obstacle closes during delay and stop (e.g. m/s).

    Returns
    -------
    Quantity
        Speed ceiling in m/s (zero if the margins consume the clearance).
    """
    validate_nonnegative(d_clear, "d_clear")
    validate_nonnegative(total_latency, "total_latency")
    a = max_deceleration.to(ureg.meter / ureg.second**2).magnitude
    j = max_jerk.to(ureg.meter / ureg.second**3).magnitude
    validate_positive(max_deceleration, "max_deceleration")
    validate_positive(max_jerk, "max_jerk")

    d_eff = d_clear.to(ureg.meter).magnitude
    for margin, name in ((loc_margin, "loc_margin"), (physical_margin, "physical_margin")):
        if margin is not None:
            validate_nonnegative(margin, name)
            d_eff -= margin.to(ureg.meter).magnitude
    tau = total_latency.to(ureg.second).magnitude
    v_h = 0.0
    if approach_velocity is not None:
        validate_nonnegative(approach_velocity, "approach_velocity")
        v_h = approach_velocity.to(ureg.meter / ureg.second).magnitude

    def excess(v):
        t_s, d_s = _scurve_stop_si(v, a, j)
        return v * tau + d_s + v_h * (tau + t_s) - d_eff

    zero = 0.0 * (ureg.meter / ureg.second)
    if excess(0.0) >= 0.0:
        return zero
    # d_S(v) >= v^2 / (2 a), so excess(sqrt(2 a d_eff)) >= 0: a valid upper bracket.
    lo, hi = 0.0, math.sqrt(2.0 * a * d_eff)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * max(1.0, hi):
            break
    return (lo * (ureg.meter / ureg.second)).to(ureg.meter / ureg.second)


def calc_scurve_rest_to_rest_time(distance, max_acceleration, max_jerk):
    """
    Calculate the duration of a jerk-limited rest-to-rest move with no cruise phase.

    A symmetric double-S profile that accelerates, then immediately decelerates.
    With t_j = a/j and x the time from the start of the jerk ramp to the end of
    the constant-acceleration plateau minus t_j (peak speed a*x):

        plateau reached (x >= t_j):  D = a * x * (x + t_j),  T = 2 (x + t_j)
        no plateau (D < 2 a^3 / j^2): D = 2 j tau^3,           T = 4 tau

    Returns a dict with ``duration``, ``peak_velocity``, and ``reaches_max_acceleration``.
    Works for linear or angular moves (distance in m or rad, rates to match).

    Source: Biagiotti and Melchiorri, Trajectory Planning for Automatic Machines
    and Robots (Springer, 2008), ch. 3, double-S velocity profile.
    """
    validate_positive(distance, "distance")
    validate_positive(max_acceleration, "max_acceleration")
    validate_positive(max_jerk, "max_jerk")

    t_unit = ureg.second
    d_unit = distance.units
    d = distance.magnitude
    a = max_acceleration.to(d_unit / t_unit**2).magnitude
    j = max_jerk.to(d_unit / t_unit**3).magnitude

    t_j = a / j
    x = (-t_j + math.sqrt(t_j**2 + 4 * d / a)) / 2
    if x >= t_j:
        duration = 2 * (x + t_j)
        v_peak = a * x
        plateau = True
    else:
        tau = (d / (2 * j)) ** (1.0 / 3.0)
        duration = 4 * tau
        v_peak = j * tau**2
        plateau = False
    return {
        "duration": duration * t_unit,
        "peak_velocity": v_peak * (d_unit / t_unit),
        "reaches_max_acceleration": plateau,
    }


def calc_c2_effective_deceleration(peak_deceleration):
    """
    Calculate the equivalent constant deceleration for a C2-continuous stop.

    A C2-continuous stop suffix entered from cruise velocity v with peak
    deceleration a_peak requires stopping distance d_C2 = 0.75 * v^2 / a_peak
    (derived in calc_c2_stop_suffix). The equivalent constant deceleration
    a_eff that achieves the exact same stopping distance (v^2 / (2 * a_eff) = d_C2) is:

        a_eff = a_peak / 1.5 = (2 / 3) * a_peak

    Parameters
    ----------
    peak_deceleration : Quantity
        Peak credible deceleration of the C2 stop profile (e.g. m/s^2).

    Returns
    -------
    Quantity
        Equivalent constant deceleration in m/s^2.
    """
    validate_positive(peak_deceleration, "peak_deceleration")
    a = peak_deceleration.to(ureg.meter / ureg.second**2)
    return (a / 1.5).to(ureg.meter / ureg.second**2)


def calc_latch_contact_stopping_profile(
    approach_velocity=None,
    tripwire_force=None,
    latch_stiffness=None,
    response_time=None,
    force_limit=None,
    *,
    scenario=None,
):
    """
    Calculate contact force rise, brake command force, and required arm deceleration upon latch contact.

    When a robot manipulator contacts a compliant latch or strike plate:
    1. Force accumulation rate: df/dt = k_latch * v_approach
    2. Force at brake command onset (tripwire force + accumulation during response time):
       f_cmd = f_tripwire + k_latch * v_approach * t_response
    3. Remaining allowable deflection before exceeding contact limit f_limit:
       x_remaining = (f_limit - f_cmd) / k_latch
    4. Required constant arm deceleration to halt within x_remaining:
       a_arm_needed = v_approach^2 / (2 * x_remaining)
    5. Duration of the arm stopping phase:
       t_arm_stop = v_approach / a_arm_needed

    Parameters
    ----------
    approach_velocity : Quantity, optional
        Arm approach speed (e.g. m/s). If omitted and scenario is given, defaults to scenario.v_latch_approach.
    tripwire_force : Quantity, optional
        Tripwire threshold force (e.g. N). Defaults to scenario.f_latch_tripwire if scenario is given.
    latch_stiffness : Quantity, optional
        Spring stiffness of the latch mechanism (e.g. N/m). Defaults to scenario.k_latch.
    response_time : Quantity, optional
        Contact-loop response latency from tripwire detection to brake onset (e.g. ms).
        Defaults to scenario.t_contact_response.
    force_limit : Quantity, optional
        Maximum permissible contact force limit (e.g. N). Defaults to scenario.f_latch_limit.
    scenario : EmbodiedSiteScenario, optional
        Site scenario supplying default latch physical parameters.

    Returns
    -------
    dict
        - "f_at_brake_onset": Force at brake command onset (Quantity in N).
        - "x_remaining": Remaining allowable displacement (Quantity in mm).
        - "a_arm_needed": Required arm deceleration to honor force limit (Quantity in m/s^2).
        - "t_arm_stop": Arm stopping duration (Quantity in ms).
        - "df_dt": Force rate of rise during contact (Quantity in N/s).
    """
    if scenario is not None:
        if approach_velocity is None:
            approach_velocity = scenario.v_latch_approach
        if tripwire_force is None:
            tripwire_force = scenario.f_latch_tripwire
        if latch_stiffness is None:
            latch_stiffness = scenario.k_latch
        if response_time is None:
            response_time = scenario.t_contact_response
        if force_limit is None:
            force_limit = scenario.f_latch_limit

    if approach_velocity is None or tripwire_force is None or latch_stiffness is None or response_time is None or force_limit is None:
        raise ValueError("All parameters must be provided directly or inferred from scenario.")

    validate_positive(approach_velocity, "approach_velocity")
    validate_positive(tripwire_force, "tripwire_force")
    validate_positive(latch_stiffness, "latch_stiffness")
    validate_positive(response_time, "response_time")
    validate_positive(force_limit, "force_limit")

    v = approach_velocity.to(ureg.meter / ureg.second)
    k = latch_stiffness.to(ureg.newton / ureg.meter)
    f_trip = tripwire_force.to(ureg.newton)
    t_resp = response_time.to(ureg.second)
    f_lim = force_limit.to(ureg.newton)

    df_dt = (k * v).to(ureg.newton / ureg.second)
    f_cmd = calc_tripwire_contact_force_accumulation(f_trip, k, v, t_resp).to(ureg.newton)
    if f_cmd >= f_lim:
        raise ValueError(f"Brake onset force {f_cmd} already exceeds or meets limit {f_lim}")

    x_rem = ((f_lim - f_cmd) / k).to(ureg.millimeter)
    x_rem_m = x_rem.to(ureg.meter)
    a_need = (v**2 / (2 * x_rem_m)).to(ureg.meter / ureg.second**2)
    t_stop = (v / a_need).to(ureg.millisecond)

    return {
        "f_at_brake_onset": f_cmd,
        "x_remaining": x_rem,
        "a_arm_needed": a_need,
        "t_arm_stop": t_stop,
        "df_dt": df_dt,
    }


def calc_scenario_stopping_budget(
    scenario,
    *,
    velocity=None,
    deceleration=None,
    c2_profile=True,
):
    """
    Calculate stopping distance budget, spare clearance, and speed ceiling for a scenario.

    Consolidates scenario-level stopping physics from an EmbodiedSiteScenario
    (e.g., WarehouseAisle):
    1. Pre-braking delay tau_delay is taken from scenario.tau_delay.
    2. Operating velocity defaults to scenario.v_aisle.
    3. Peak deceleration defaults to scenario.a_brake.
    4. When c2_profile=True, calculates effective constant deceleration
       a_eff = calc_c2_effective_deceleration(deceleration); otherwise a_eff = deceleration.
    5. Localization and tracking overhead is overhead_loc = scenario.delta_loc + scenario.eps_track.
    6. Physical margin is phys_margin = scenario.delta_margin.
    7. Total stopping distance budget:
       budget = calc_safe_stopping_distance(v, tau_delay, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin)
    8. Residual spare clearance:
       spare = scenario.d_clear - budget
    9. Maximum permitted velocity ceiling:
       v_ceiling = calc_max_permitted_velocity(scenario.d_clear, tau_delay, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin)

    Parameters
    ----------
    scenario : EmbodiedSiteScenario
        Site scenario instance containing d_clear, delta_loc, eps_track, delta_margin,
        tau_delay, and default speeds.
    velocity : Quantity, optional
        Operating velocity to evaluate (default: scenario.v_aisle).
    deceleration : Quantity, optional
        Loaded deceleration (default: scenario.a_brake).
    c2_profile : bool, optional
        Whether the stop follows a C2-continuous profile (default: True).

    Returns
    -------
    dict
        - "budget": Total safe stopping distance budget (Quantity in mm).
        - "spare": Residual clearance margin d_clear - budget (Quantity in mm).
        - "v_ceiling": Maximum safe operating velocity (Quantity in m/s).
        - "a_eff": Effective deceleration used (Quantity in m/s^2).
        - "tau_delay": Pre-braking delay (Quantity in ms).
        - "overhead_loc": Localization and tracking uncertainty (Quantity in m).
    """
    v = velocity if velocity is not None else scenario.v_aisle
    a_peak = deceleration if deceleration is not None else scenario.a_brake

    validate_positive(v, "velocity")
    validate_positive(a_peak, "deceleration")

    a_eff = calc_c2_effective_deceleration(a_peak) if c2_profile else a_peak
    tau = scenario.tau_delay
    overhead_loc = (scenario.delta_loc + scenario.eps_track).to(ureg.meter)
    phys_margin = scenario.delta_margin.to(ureg.meter)

    budget = calc_safe_stopping_distance(
        v, tau, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin
    ).to(ureg.millimeter)

    spare = (scenario.d_clear - budget).to(ureg.millimeter)

    v_ceil = calc_max_permitted_velocity(
        scenario.d_clear, tau, a_eff, loc_margin=overhead_loc, physical_margin=phys_margin
    ).to(ureg.meter / ureg.second)

    return {
        "budget": budget,
        "spare": spare,
        "v_ceiling": v_ceil,
        "a_eff": a_eff,
        "tau_delay": tau,
        "overhead_loc": overhead_loc,
    }
