"""State estimation and belief expiry: scalar fusion, inertial coasting, and error-bound growth (added 2026-10-20)."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_nonnegative, validate_positive


def calc_scalar_kalman_update(prior_variance, measurement_variance):
    """
    Gain and posterior variance of one scalar Kalman update.

    Equation:
        g = P / (P + R) ;  P+ = (1 - g) P

    The gain is how far the estimate moves toward the reading, between 0 and
    1; the posterior is tighter than either input. Both variances must share
    a unit.

    Source: Kalman (1960), Trans. ASME J. Basic Eng. 82(1); Thrun, Burgard,
    and Fox, Probabilistic Robotics (2005), Table 3.1, in one dimension.

    Returns
    -------
    tuple
        (gain as a float, posterior variance in the unit of ``prior_variance``)
    """
    validate_positive(prior_variance, "prior_variance")
    validate_positive(measurement_variance, "measurement_variance")
    gain = float((prior_variance / (prior_variance + measurement_variance)).to("").magnitude)
    return gain, ((1 - gain) * prior_variance).to(prior_variance.units)


def calc_inertial_coast_errors(gyro_noise_density, gyro_bias, accel_noise_density, accel_bias, duration):
    """
    Heading and position error of an IMU integrated open loop for ``duration``.

    Equations:
        heading from white rate noise   = N_g sqrt(T)
        heading from a constant offset  = b_g T
        position from white accel noise = N_a sqrt(T^3 / 3)
        position from a constant offset = b_a T^2 / 2

    Noise grows with the square root of time (a random walk, integrated once
    more for position); an offset grows with time, or its square for
    position, so offsets limit how long a robot can coast between fixes.

    Source: Woodman, "An introduction to inertial navigation," University of
    Cambridge Computer Laboratory Tech. Rep. UCAM-CL-TR-696 (2007), Secs. 4.2
    and 4.3.

    Returns
    -------
    dict
        heading_noise, heading_bias (angles), position_noise, position_bias (lengths).
    """
    validate_positive(duration, "duration")
    t = duration.to(ureg.second)
    root_t = t.magnitude ** 0.5 * ureg.second ** 0.5
    return {
        "heading_noise": (gyro_noise_density * root_t).to(ureg.degree),
        "heading_bias": (gyro_bias * t).to(ureg.degree),
        "position_noise": (accel_noise_density * (t.magnitude ** 3 / 3) ** 0.5 * ureg.second ** 1.5).to(ureg.millimeter),
        "position_bias": (0.5 * accel_bias * t ** 2).to(ureg.millimeter),
    }


def calc_belief_error_bound(initial_error, growth_rate, elapsed, growth_acceleration=None):
    """
    Error bound of a belief after ``elapsed`` without new evidence.

    Equation:
        E(dt) = E_0 + v dt + a dt^2 / 2

    ``v`` bounds the unobserved velocity and ``a`` the disturbance
    acceleration; a growth law bounded by a speed alone has ``a = 0``.

    Source: kinematics of a bounded disturbance (displacement under a
    velocity bound plus a constant-acceleration bound).
    """
    validate_nonnegative(elapsed, "elapsed")
    e = initial_error + growth_rate * elapsed
    if growth_acceleration is not None:
        e = e + growth_acceleration * elapsed ** 2 / 2
    return e.to(ureg.millimeter)


def calc_quadratic_invalidation_horizon(initial_error, growth_rate, growth_acceleration, error_limit):
    """
    Invalidation horizon of a belief whose error bound grows with velocity and acceleration terms.

    Equation:
        a dt^2 / 2 + v dt + (E_0 - E_max) = 0 ;  T_inv = the positive root

    With ``a = 0`` this is ``calc_linear_invalidation_horizon``. A zero result
    means the initial error already spends the limit.

    Source: first-passage time of the bound ``calc_belief_error_bound``
    (definition of the invalidation horizon, T_inv = inf{dt >= 0 : E(dt) >= E_max}).
    """
    from .perception import calc_linear_invalidation_horizon
    validate_positive(growth_rate, "growth_rate")
    a = growth_acceleration.to(ureg.meter / ureg.second ** 2).magnitude
    if a == 0:
        return calc_linear_invalidation_horizon(initial_error, growth_rate, error_limit)
    validate_positive(growth_acceleration, "growth_acceleration")
    v = growth_rate.to(ureg.meter / ureg.second).magnitude
    c = (initial_error - error_limit).to(ureg.meter).magnitude
    if c >= 0:
        return 0.0 * ureg.millisecond
    half_a = a / 2
    root = (-v + math.sqrt(v * v - 4 * half_a * c)) / (2 * half_a)
    return (root * ureg.second).to(ureg.millisecond)
