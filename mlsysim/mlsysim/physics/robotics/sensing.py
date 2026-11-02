"""Sensing error scales: stereo triangulation, lever arms, scan sectors, fog, and expansion (added 2026-10-20).

Each function is one textbook relation with units attached. The solvers in
``engine/solvers/perception.py`` compose them for one stated machine and site.
"""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_nonnegative, validate_positive


def calc_stereo_depth_sigma(depth, focal_length_px: float, baseline, disparity_sigma_px: float):
    """
    One-sigma depth error of a rectified stereo pair at range ``depth``.

    Equation:
        z = f b / d  =>  sigma_z = z^2 sigma_d / (f b)

    First-order propagation of disparity noise through triangulation; the
    error grows with the square of range.

    Source: Hartley and Zisserman, Multiple View Geometry, 2nd ed. (2003),
    Sec. 12.6; Gallup et al., CVPR 2008, eq. 2.
    """
    validate_positive(depth, "depth")
    validate_positive(baseline, "baseline")
    if focal_length_px <= 0 or disparity_sigma_px <= 0:
        raise ValueError("focal_length_px and disparity_sigma_px must be positive")
    return (depth ** 2 * disparity_sigma_px / (focal_length_px * baseline)).to(ureg.millimeter)


def calc_stereo_range_for_sigma(depth_sigma, focal_length_px: float, baseline, disparity_sigma_px: float):
    """
    Range at which a stereo pair's one-sigma depth error reaches ``depth_sigma``.

    Equation:
        z* = sqrt(C f b / sigma_d)

    The inverse of ``calc_stereo_depth_sigma``: the range wall beyond which
    one sigma alone spends a clearance C.
    """
    validate_positive(depth_sigma, "depth_sigma")
    validate_positive(baseline, "baseline")
    if focal_length_px <= 0 or disparity_sigma_px <= 0:
        raise ValueError("focal_length_px and disparity_sigma_px must be positive")
    return ((depth_sigma * focal_length_px * baseline / disparity_sigma_px) ** 0.5).to(ureg.meter)


def calc_stereo_baseline_bias(depth, baseline, baseline_change):
    """
    Depth bias from an unmodeled change in a stereo baseline.

    Equation:
        |dz| = z |db| / b

    z = f b / d is linear in b at fixed disparity, so a relative baseline
    error becomes the same relative depth error.
    """
    validate_positive(depth, "depth")
    validate_positive(baseline, "baseline")
    validate_nonnegative(baseline_change, "baseline_change")
    return (depth * baseline_change / baseline).to(ureg.millimeter)


def calc_lever_arm_error(distance, angle):
    """
    Lateral displacement of a point at ``distance`` seen through an angular error.

    Equation:
        Delta_y = d theta   (small angle; d sin(theta) and d tan(theta) differ
                             from it by under 0.01 percent at one degree)

    An attitude, mount, or extrinsic angle error grows with range: the same
    angle that moves a near point millimeters moves a far one decimeters.
    """
    validate_nonnegative(distance, "distance")
    theta = angle.to(ureg.radian).magnitude
    return (distance * theta).to(ureg.millimeter)


def calc_scan_sector_angle(width, distance):
    """
    Full angle a scanning sensor sweeps to cover a width centered ``distance`` ahead.

    Equation:
        phi = 2 atan((w / 2) / d)
    """
    validate_positive(width, "width")
    validate_positive(distance, "distance")
    half = (width / 2 / distance).to(ureg.dimensionless).magnitude
    return (2 * math.atan(half) * ureg.radian).to(ureg.degree)


def calc_contrast_visibility_range(extinction, contrast_threshold: float):
    """
    Range at which a black target's apparent contrast falls to ``contrast_threshold``.

    Equation:
        C(d) = C_0 exp(-beta d)  =>  d = ln(1 / epsilon) / beta

    Koschmieder's law for a black target against the horizon sky; with
    epsilon = 0.05 the range is the meteorological optical range.

    Source: Koschmieder (1924); WMO Guide to Instruments and Methods of
    Observation (WMO-No. 8), Vol. I, Ch. 9 (5 percent contrast threshold).
    """
    validate_positive(extinction, "extinction")
    if not 0.0 < contrast_threshold < 1.0:
        raise ValueError("contrast_threshold must be in (0, 1)")
    return (math.log(1 / contrast_threshold) / extinction).to(ureg.meter)


def calc_thermal_expansion(coefficient, length, temperature_rise):
    """
    Length change of a uniformly heated member.

    Equation:
        dL = alpha L dT
    """
    validate_positive(length, "length")
    return (coefficient * length * temperature_rise).to(ureg.millimeter)
