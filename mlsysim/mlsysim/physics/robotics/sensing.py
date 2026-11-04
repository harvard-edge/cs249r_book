"""Sensing error scales: stereo triangulation, lever arms, scan sectors, fog, and expansion (added 2026-10-20).

Each function is one textbook relation with units attached. The solvers in
``engine/solvers/perception.py`` compose them for one stated machine and site.
"""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_nonnegative, validate_positive


def calc_stereo_depth_sigma(depth, focal_length_px: float, baseline, disparity_sigma_px: float):
    """Estimate stereo depth uncertainty from disparity noise.

    Parameters
    ----------
    depth : Quantity
        Positive target range in length units.
    focal_length_px : float
        Positive rectified focal length in pixels.
    baseline : Quantity
        Positive separation between camera centers in length units.
    disparity_sigma_px : float
        Positive disparity standard deviation in pixels.

    Returns
    -------
    Quantity
        Depth standard deviation in millimeters.

    Notes
    -----
    Equation: sigma_z = ``z**2`` * sigma_d / (f * b).

    Assumptions: Rectified stereo, fixed calibration, and small disparity perturbations.

    Excludes: Calibration bias, correlated image errors, occlusions, and a safety bound.

    Equation:
        z = f b / d  =>  sigma_z = z^2 sigma_d / (f b)

    First-order propagation of disparity noise through triangulation; the
    error grows with the square of range.

    Source: Hartley and Zisserman, Multiple View Geometry, 2nd ed. (2003),
    Sec. 12.6; Gallup et al., CVPR 2008, eq. 2.

    References
    ----------
    Derivation: Differentiate z = ``f*b/d`` and multiply the magnitude of its derivative by
    disparity standard deviation; the existing Hartley-Zisserman attribution describes the
    stereo geometry.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_stereo_depth_sigma
    >>> from mlsysim.core.units import ureg
    >>> near = calc_stereo_depth_sigma(1 * ureg.meter, 100, 0.1 * ureg.meter, 1)
    >>> far = calc_stereo_depth_sigma(2 * ureg.meter, 100, 0.1 * ureg.meter, 1)
    >>> assert far == near * 4
    """
    validate_positive(depth, "depth")
    validate_positive(baseline, "baseline")
    if focal_length_px <= 0 or disparity_sigma_px <= 0:
        raise ValueError("focal_length_px and disparity_sigma_px must be positive")
    return (depth**2 * disparity_sigma_px / (focal_length_px * baseline)).to(ureg.millimeter)


def calc_stereo_range_for_sigma(depth_sigma, focal_length_px: float, baseline, disparity_sigma_px: float):
    """Invert the stereo depth-uncertainty relation.

    Parameters
    ----------
    depth_sigma : Quantity
        Positive allowed one-sigma depth error in length units.
    focal_length_px : float
        Positive rectified focal length in pixels.
    baseline : Quantity
        Positive camera baseline in length units.
    disparity_sigma_px : float
        Positive disparity standard deviation in pixels.

    Returns
    -------
    Quantity
        Range in meters where the modeled uncertainty reaches the allowance.

    Notes
    -----
    Equation: z = sqrt(depth_sigma * f * b / sigma_d).

    Assumptions: The same rectified, small-noise model as calc_stereo_depth_sigma.

    Excludes: A deterministic clearance guarantee, calibration bias, and occlusion.

    Equation:
        z* = sqrt(C f b / sigma_d)

    The inverse of ``calc_stereo_depth_sigma``: the range wall beyond which
    one sigma alone spends a clearance C.

    References
    ----------
    Derivation: Rearrange sigma_z = ``z**2*sigma_d/(f*b)`` and select the positive square
    root.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_stereo_range_for_sigma
    >>> from mlsysim.core.units import ureg
    >>> limit = calc_stereo_range_for_sigma(0.1 * ureg.meter, 100, 0.1 * ureg.meter, 1)
    >>> assert limit.to("meter").magnitude == 1
    """
    validate_positive(depth_sigma, "depth_sigma")
    validate_positive(baseline, "baseline")
    if focal_length_px <= 0 or disparity_sigma_px <= 0:
        raise ValueError("focal_length_px and disparity_sigma_px must be positive")
    return ((depth_sigma * focal_length_px * baseline / disparity_sigma_px) ** 0.5).to(ureg.meter)


def calc_stereo_baseline_bias(depth, baseline, baseline_change):
    """Propagate a baseline calibration change into depth bias.

    Parameters
    ----------
    depth : Quantity
        Positive target range in length units.
    baseline : Quantity
        Positive calibrated stereo baseline in length units.
    baseline_change : Quantity
        Nonnegative magnitude of the baseline change in length units.

    Returns
    -------
    Quantity
        Depth-bias magnitude in millimeters.

    Notes
    -----
    Equation: delta_z = z * delta_b / b.

    Assumptions: Disparity and focal length remain fixed as the baseline changes.

    Excludes: Other intrinsic/extrinsic calibration changes and random disparity noise.

    Equation:
        |dz| = z |db| / b

    z = f b / d is linear in b at fixed disparity, so a relative baseline
    error becomes the same relative depth error.

    References
    ----------
    Derivation: The stereo equation z = ``f*b/d`` is linear in baseline at fixed f and d.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_stereo_baseline_bias
    >>> from mlsysim.core.units import ureg
    >>> bias = calc_stereo_baseline_bias(1 * ureg.meter, 0.1 * ureg.meter, 1 * ureg.millimeter)
    >>> assert bias == (1 * ureg.meter) * (1 * ureg.millimeter) / (0.1 * ureg.meter)
    """
    validate_positive(depth, "depth")
    validate_positive(baseline, "baseline")
    validate_nonnegative(baseline_change, "baseline_change")
    return (depth * baseline_change / baseline).to(ureg.millimeter)


def calc_lever_arm_error(distance, angle):
    """Estimate signed lateral error from a small angular error.

    Parameters
    ----------
    distance : Quantity
        Nonnegative lever-arm length, in length units.
    angle : Quantity
        Signed angular error convertible to radians; use only small angles.

    Returns
    -------
    Quantity
        Signed lateral displacement in millimeters.

    Notes
    -----
    Equation: delta_y = distance * angle_radians.

    Assumptions: Small-angle linearization about the aligned direction.

    Excludes: Higher-order trigonometric terms and translational calibration error.

    Equation:
        Delta_y = d theta   (small angle; d sin(theta) and d tan(theta) differ
                             from it by under 0.01 percent at one degree)

    An attitude, mount, or extrinsic angle error grows with range: the same
    angle that moves a near point millimeters moves a far one decimeters.

    References
    ----------
    Derivation: Linearize sin(theta) around zero, then multiply the angular displacement
    by the lever arm.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_lever_arm_error
    >>> from mlsysim.core.units import ureg
    >>> error = calc_lever_arm_error(1 * ureg.meter, 0.01 * ureg.radian)
    >>> assert error == 0.01 * ureg.meter
    """
    validate_nonnegative(distance, "distance")
    theta = angle.to(ureg.radian).magnitude
    return (distance * theta).to(ureg.millimeter)


def calc_scan_sector_angle(width, distance):
    """Compute the full scan angle needed to span a width.

    Parameters
    ----------
    width : Quantity
        Positive transverse width in length units.
    distance : Quantity
        Positive perpendicular distance to the center of that width.

    Returns
    -------
    Quantity
        Full scan angle in degrees.

    Notes
    -----
    Equation: phi = 2 * atan(width / (2 * distance)).

    Assumptions: A centered planar width perpendicular to the viewing direction.

    Excludes: Lens distortion, occlusion, sampling density, and off-axis coverage.

    Equation:
        phi = 2 atan((w / 2) / d)

    References
    ----------
    Derivation: The half-width and distance form a right triangle; double its arctangent
    half-angle.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_scan_sector_angle
    >>> from mlsysim.core.units import ureg
    >>> angle = calc_scan_sector_angle(2 * ureg.meter, 1 * ureg.meter)
    >>> assert abs(angle.to("degree").magnitude - 90) < 1e-10
    """
    validate_positive(width, "width")
    validate_positive(distance, "distance")
    half = (width / 2 / distance).to(ureg.dimensionless).magnitude
    return (2 * math.atan(half) * ureg.radian).to(ureg.degree)


def calc_contrast_visibility_range(extinction, contrast_threshold: float):
    """Invert exponential contrast attenuation for a black target.

    Parameters
    ----------
    extinction : Quantity
        Positive homogeneous extinction coefficient in inverse length.
    contrast_threshold : float
        Detectable contrast fraction in (0, 1).

    Returns
    -------
    Quantity
        Modeled visibility range in meters.

    Notes
    -----
    Equation: range = log(1 / contrast_threshold) / extinction.

    Assumptions: Unit initial contrast and homogeneous extinction along the sightline.

    Excludes: Sensor-specific detection probability, lighting variation, and safety
    certification.

    Equation:
        C(d) = C_0 exp(-beta d)  =>  d = ln(1 / epsilon) / beta

    Koschmieder's law for a black target against the horizon sky; with
    epsilon = 0.05 the range is the meteorological optical range.

    Source: Koschmieder (1924); WMO Guide to Instruments and Methods of
    Observation (WMO-No. 8), Vol. I, Ch. 9 (5 percent contrast threshold).

    References
    ----------
    Derivation: Solve C(d) = ``exp(-beta*d)`` for d at C = contrast_threshold; the existing
    Koschmieder/WMO attribution motivates this attenuation model.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_contrast_visibility_range
    >>> from mlsysim.core.units import ureg
    >>> a = calc_contrast_visibility_range(1 / ureg.meter, 0.1)
    >>> b = calc_contrast_visibility_range(2 / ureg.meter, 0.1)
    >>> assert a == 2 * b
    """
    validate_positive(extinction, "extinction")
    if not 0.0 < contrast_threshold < 1.0:
        raise ValueError("contrast_threshold must be in (0, 1)")
    return (math.log(1 / contrast_threshold) / extinction).to(ureg.meter)


def calc_thermal_expansion(coefficient, length, temperature_rise):
    """Compute a member's signed linear thermal expansion.

    Parameters
    ----------
    coefficient : Quantity
        Linear expansion coefficient in inverse temperature, assumed constant.
    length : Quantity
        Positive reference length in length units.
    temperature_rise : Quantity
        Signed temperature difference, using kelvin or a delta-temperature unit.

    Returns
    -------
    Quantity
        Signed length change in millimeters.

    Notes
    -----
    Equation: delta_L = coefficient * length * temperature_rise.

    Assumptions: Uniform temperature change and a constant linear expansion coefficient.

    Excludes: Thermal gradients, bending, constraints, and temperature-dependent
    coefficients.

    Equation:
        dL = alpha L dT

    References
    ----------
    Derivation: The linear expansion coefficient is the fractional length change per unit
    temperature difference.

    Examples
    --------
    >>> from mlsysim.physics.robotics.sensing import calc_thermal_expansion
    >>> from mlsysim.core.units import ureg
    >>> change = calc_thermal_expansion(1e-5 / ureg.kelvin, 1 * ureg.meter, 10 * ureg.kelvin)
    >>> assert change.to("millimeter").magnitude > 0
    """
    validate_positive(length, "length")
    return (coefficient * length * temperature_rise).to(ureg.millimeter)
