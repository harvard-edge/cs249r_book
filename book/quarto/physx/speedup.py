"""
speedup.py
Speedup and efficiency helpers.
"""

from .formulas import calc_amdahls_speedup


def amdahl_speedup(p, s):
    return calc_amdahls_speedup(p, s)


def efficiency(speedup, processors):
    return speedup / processors
