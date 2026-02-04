"""
roofline.py
Roofline-related helpers: arithmetic intensity, ridge point, bounds.
"""

from .constants import ureg


def arithmetic_intensity(flops, bytes_moved):
    """Return arithmetic intensity (flops/byte) as a Pint Quantity."""
    return flops / bytes_moved


def ridge_point(peak_flops, mem_bw):
    """Ridge point in flops/byte."""
    return peak_flops / mem_bw


def roofline_bound(intensity, peak_flops, mem_bw):
    """
    Return achievable performance given intensity, peak, and bandwidth.
    """
    bw_bound = intensity * mem_bw
    return min(peak_flops, bw_bound)
