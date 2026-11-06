"""State estimation and belief expiry: scalar fusion, inertial coasting, and error-bound growth (added 2026-10-20)."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_nonnegative, validate_positive


def calc_scalar_kalman_update(prior_variance, measurement_variance):
    """Compute scalar Kalman gain and posterior variance.

    Parameters
    ----------
    prior_variance : Quantity
        Positive prior-error variance in squared state units.
    measurement_variance : Quantity
        Positive measurement-error variance with the same dimension as the prior.

    Returns
    -------
    tuple
        Dimensionless float gain and posterior variance in the prior variance unit.

    Notes
    -----
    Equation: gain = P / (P + R); posterior = (1 - gain) * P.

    Assumptions: A scalar direct observation with independent prior and measurement
    errors.

    Excludes: State propagation, cross-covariance, bias, and covariance consistency
    checks.

    Equation:
        g = P / (P + R) ;  P+ = (1 - g) P

    The gain is how far the estimate moves toward the reading, between 0 and
    1; the posterior is tighter than either input. Both variances must share
    a unit.

    Source: Kalman (1960), Trans. ASME J. Basic Eng. 82(1); Thrun, Burgard,
    and Fox, Probabilistic Robotics (2005), Table 3.1, in one dimension.

    References
    ----------
    Derivation: Minimizing the variance of a weighted combination of independent scalar
    estimates yields gain P/(P+R); the existing Kalman/Thrun references give the estimator
    context.

    Examples
    --------
    >>> from mlsysim.physics.robotics.estimation import calc_scalar_kalman_update
    >>> from mlsysim.core.units import ureg
    >>> gain, posterior = calc_scalar_kalman_update(1 * ureg.meter**2, 1 * ureg.meter**2)
    >>> assert gain == 0.5
    >>> assert posterior == 0.5 * ureg.meter**2
    """
    validate_positive(prior_variance, "prior_variance")
    validate_positive(measurement_variance, "measurement_variance")
    gain = float((prior_variance / (prior_variance + measurement_variance)).to("").magnitude)
    return gain, ((1 - gain) * prior_variance).to(prior_variance.units)


def calc_inertial_coast_errors(gyro_noise_density, gyro_bias, accel_noise_density, accel_bias, duration):
    """Estimate open-loop inertial error components over a duration.

    Parameters
    ----------
    gyro_noise_density : Quantity
        Angular-rate white-noise density in angle / sqrt(time).
    gyro_bias : Quantity
        Constant signed gyroscope bias in angle / time.
    accel_noise_density : Quantity
        Acceleration white-noise density in length / time**1.5.
    accel_bias : Quantity
        Constant signed acceleration bias in length / time**2.
    duration : Quantity
        Positive integration duration in time units.

    Returns
    -------
    dict
        Heading_noise and heading_bias in degrees; position_noise and position_bias in
        millimeters.

    Notes
    -----
    Equation: heading_noise = ``N_g*sqrt(T)``; heading_bias = ``b_g*T``; position_noise =
    ``N_a*sqrt(T**3/3)``; position_bias = ``b_a*T**2/2``.

    Assumptions: White independent rate/acceleration noise and constant biases over the
    interval.

    Excludes: Gravity projection from attitude errors, bias drift, cross-axis coupling,
    and sensor fusion.

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

    References
    ----------
    Derivation: Integrate constant biases and white-noise covariance over time; Woodman
    (2007), UCAM-CL-TR-696, Sections 4.2 and 4.3, supplies the existing inertial-
    navigation context.

    Examples
    --------
    >>> from mlsysim.physics.robotics.estimation import calc_inertial_coast_errors
    >>> from mlsysim.core.units import ureg
    >>> errors = calc_inertial_coast_errors(0 * ureg.radian / ureg.second**0.5, 0 * ureg.radian / ureg.second, 0 * ureg.meter / ureg.second**1.5, 1 * ureg.meter / ureg.second**2, 2 * ureg.second)
    >>> assert errors["position_bias"] == 2 * ureg.meter
    """
    validate_positive(duration, "duration")
    t = duration.to(ureg.second)
    root_t = t.magnitude**0.5 * ureg.second**0.5
    return {
        "heading_noise": (gyro_noise_density * root_t).to(ureg.degree),
        "heading_bias": (gyro_bias * t).to(ureg.degree),
        "position_noise": (accel_noise_density * (t.magnitude**3 / 3) ** 0.5 * ureg.second**1.5).to(ureg.millimeter),
        "position_bias": (0.5 * accel_bias * t**2).to(ureg.millimeter),
    }


def calc_belief_error_bound(initial_error, growth_rate, elapsed, growth_acceleration=None):
    """Propagate a scalar error allowance without new evidence.

    Parameters
    ----------
    initial_error : Quantity
        Initial nonnegative error allowance in length units.
    growth_rate : Quantity
        Nonnegative error-growth speed in length / time.
    elapsed : Quantity
        Nonnegative elapsed time since the last evidence update.
    growth_acceleration : Quantity, optional
        Nonnegative error-growth acceleration in length / time**2; None omits the
        quadratic term.

    Returns
    -------
    Quantity
        Propagated error allowance in millimeters.

    Notes
    -----
    Equation: E(t) = initial_error + ``growth_rate*t`` + ``growth_acceleration*t**2/2``.

    Assumptions: Inputs bound additive scalar disturbances over the interval.

    Excludes: Measurement updates, correlated error cancellation, and proof that supplied
    allowances are bounds.

    Equation:
        E(dt) = E_0 + v dt + a dt^2 / 2

    ``v`` bounds the unobserved velocity and ``a`` the disturbance
    acceleration; a growth law bounded by a speed alone has ``a = 0``.

    Source: kinematics of a bounded disturbance (displacement under a
    velocity bound plus a constant-acceleration bound).

    References
    ----------
    Derivation: Integrate a bounded velocity plus a constant acceleration allowance and
    add initial error.

    Examples
    --------
    >>> from mlsysim.physics.robotics.estimation import calc_belief_error_bound
    >>> from mlsysim.core.units import ureg
    >>> bound = calc_belief_error_bound(1 * ureg.millimeter, 1 * ureg.millimeter / ureg.second, 2 * ureg.second)
    >>> assert bound == 3 * ureg.millimeter
    """
    validate_nonnegative(elapsed, "elapsed")
    e = initial_error + growth_rate * elapsed
    if growth_acceleration is not None:
        e = e + growth_acceleration * elapsed**2 / 2
    return e.to(ureg.millimeter)


def calc_quadratic_invalidation_horizon(initial_error, growth_rate, growth_acceleration, error_limit):
    """Find when a growing error allowance reaches its limit.

    Parameters
    ----------
    initial_error : Quantity
        Initial nonnegative error allowance in length units.
    growth_rate : Quantity
        Positive error-growth speed in length / time.
    growth_acceleration : Quantity
        Nonnegative error-growth acceleration in length / time**2.
    error_limit : Quantity
        Allowed error in length units; a spent allowance yields zero time.

    Returns
    -------
    Quantity
        First-passage time in milliseconds, floored at zero.

    Notes
    -----
    Equation: Solve ``a*t**2/2`` + ``v*t`` + initial_error - error_limit = 0 for its nonnegative
    root.

    Assumptions: Monotone scalar error growth; zero acceleration uses the linear horizon.

    Excludes: Stochastic confidence interpretation and validation of the disturbance
    envelope.

    Equation:
        a dt^2 / 2 + v dt + (E_0 - E_max) = 0 ;  T_inv = the positive root

    With ``a = 0`` this is ``calc_linear_invalidation_horizon``. A zero result
    means the initial error already spends the limit.

    Source: first-passage time of the bound ``calc_belief_error_bound``
    (definition of the invalidation horizon, T_inv = inf{dt >= 0 : E(dt) >= E_max}).

    References
    ----------
    Derivation: Set calc_belief_error_bound equal to the limit and solve the resulting
    quadratic; the linear branch is its zero-acceleration limit.

    Examples
    --------
    >>> from mlsysim.physics.robotics.estimation import calc_quadratic_invalidation_horizon
    >>> from mlsysim.core.units import ureg
    >>> horizon = calc_quadratic_invalidation_horizon(0 * ureg.meter, 1 * ureg.meter / ureg.second, 0 * ureg.meter / ureg.second**2, 1 * ureg.meter)
    >>> assert horizon == 1 * ureg.second
    """
    from .perception import calc_linear_invalidation_horizon

    validate_positive(growth_rate, "growth_rate")
    a = growth_acceleration.to(ureg.meter / ureg.second**2).magnitude
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
