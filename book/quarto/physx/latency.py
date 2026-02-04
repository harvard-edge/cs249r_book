"""
latency.py
Latency helpers for pipelined systems.
"""

from .constants import ureg


def pipeline_latency(stages, overhead=0):
    """
    stages: iterable of durations (Pint or numeric, assumed seconds)
    overhead: optional overhead time
    """
    if not stages:
        return 0 * ureg.second
    max_stage = max(_ensure_seconds(s) for s in stages)
    return max_stage + _ensure_seconds(overhead)


def _ensure_seconds(val):
    if isinstance(val, (int, float)):
        return val * ureg.second
    return val.to(ureg.second)
