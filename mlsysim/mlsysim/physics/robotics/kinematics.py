"""Manipulator kinematics: the Jacobian's conditioning and damped inverse kinematics (added 2026-10-20)."""

import math

from ...core.units import ureg


def calc_two_link_singular_values(upper_arm, forearm, elbow_angle):
    """
    Singular values of the planar two-link arm's Jacobian at elbow angle q2.

    The Jacobian maps joint rates (q1dot, q2dot) to tool velocity (x, y):

        J = [[-l1 sin q1 - l2 sin(q1+q2), -l2 sin(q1+q2)],
             [ l1 cos q1 + l2 cos(q1+q2),  l2 cos(q1+q2)]],

    and its singular values depend on q2 only (q1 rotates the whole picture),
    so q1 = 0 here. det J = l1 l2 sin q2, so the smaller singular value falls
    to zero as the elbow straightens: the singular pose. A tool speed v in the
    weakest direction needs a joint-rate vector of norm v / sigma_min.

    Source: Lynch, K. M. and Park, F. C., Modern Robotics (2017), ch. 5.

    Returns
    -------
    tuple of Quantity
        (sigma_max, sigma_min) in meters per radian of joint motion.
    """
    l1 = upper_arm.to(ureg.meter).magnitude
    l2 = forearm.to(ureg.meter).magnitude
    q2 = elbow_angle.to(ureg.radian).magnitude
    j11, j12 = -l2 * math.sin(q2), -l2 * math.sin(q2)
    j21, j22 = l1 + l2 * math.cos(q2), l2 * math.cos(q2)
    # Singular values of a 2x2 from the eigenvalues of J^T J.
    a = j11 * j11 + j21 * j21
    b = j11 * j12 + j21 * j22
    d = j12 * j12 + j22 * j22
    half_trace, det = (a + d) / 2, a * d - b * b
    disc = math.sqrt(max(half_trace * half_trace - det, 0.0))
    s_max = math.sqrt(half_trace + disc)
    s_min = math.sqrt(max(half_trace - disc, 0.0))
    return s_max * ureg.meter, s_min * ureg.meter


def calc_dls_joint_rate(tool_speed, singular_value, damping):
    """
    Joint rate damped least squares commands for ``tool_speed`` along one singular direction.

    Damped least squares minimizes ||J qdot - v||^2 + lambda^2 ||qdot||^2;
    along a singular direction with value sigma its gain is

        qdot / v = sigma / (sigma^2 + lambda^2)  <=  1 / (2 lambda),

    with the bound reached at sigma = lambda, and it realizes the fraction
    sigma^2 / (sigma^2 + lambda^2) of the requested tool motion. With
    ``damping`` zero the gain is the exact inverse, 1 / sigma.

    Source: Wampler, C. W. (1986), Manipulator inverse kinematic solutions
    based on vector formulations and damped least-squares methods, IEEE
    Trans. SMC 16(1); Nakamura, Y. and Hanafusa, H. (1986), ASME J. Dyn. Syst.

    Returns
    -------
    tuple
        (joint rate in rad/s, realized fraction of the tool motion).
    """
    s = singular_value.to(ureg.meter).magnitude
    lam = damping.to(ureg.meter).magnitude
    v = tool_speed.to(ureg.meter / ureg.second).magnitude
    gain = s / (s * s + lam * lam)
    return v * gain * ureg.radian / ureg.second, s * s / (s * s + lam * lam)
