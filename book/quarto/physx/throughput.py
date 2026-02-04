"""
throughput.py
Throughput helpers (QPS, tokens/sec).
"""

from .constants import ureg


def qps(total_requests, total_seconds):
    if isinstance(total_seconds, (int, float)):
        total_seconds = total_seconds * ureg.second
    return total_requests / total_seconds


def tokens_per_sec(total_tokens, total_seconds):
    if isinstance(total_seconds, (int, float)):
        total_seconds = total_seconds * ureg.second
    return total_tokens / total_seconds
