"""Serving-at-scale solvers: request traces, measured frontiers, KV admission, routing.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.serving`` (Little's law, the batching knee, padding waste,
speculative rounds) and ``physics.memory`` (KV-cache size); these models
apply them to a published trace, a benchmark table, or one stated replica
(added 2026-10-20).
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Dict, Optional, Sequence

import numpy as np

from ..results import (
    KvAdmissionResult,
    PrecisionDividendResult,
)
from ...core.types import Quantity
from ...core.units import Q_, ureg
from ...physics.memory import calc_kv_cache_size
from ...physics.serving import calc_padding_waste_fraction
from .base import ForwardModel

_TRACE_TIME = "%Y-%m-%d %H:%M:%S.%f"
_BIN_TIME = "%Y-%m-%d %H:%M"
_WEEKDAYS = 5                       # Monday to Friday: datetime.weekday() below 5










class KvAdmissionModel(ForwardModel):
    """
    How many requests of one context length a replica's KV budget admits.

    The budget is the replica's HBM minus its weights and a runtime reserve;
    each request holds ``2 L H_kv d_head b`` bytes per token for its whole
    context (``calc_kv_cache_size``). Paging changes none of these bytes, only
    the waste around them.
    """
    requires = ("workload",)
    produces = KvAdmissionResult

    def solve(self, model, accelerators: int, hbm_per_accelerator: Quantity,
              weight_bytes_per_param: Quantity, reserve: Quantity, context_tokens: int,
              kv_bytes_per_elem: Quantity) -> KvAdmissionResult:
        head_dim = model.hidden_dim // model.heads
        kv_heads = model.kv_heads or model.heads
        per_token = calc_kv_cache_size(model.layers, kv_heads, head_dim, 1, 1, kv_bytes_per_elem)
        per_request = calc_kv_cache_size(model.layers, kv_heads, head_dim, context_tokens, 1, kv_bytes_per_elem)
        weights = model.size_in_bytes(weight_bytes_per_param)
        total = accelerators * hbm_per_accelerator
        budget = total - weights - reserve
        return KvAdmissionResult(
            head_dim=head_dim, weights=weights.to("GB"), hbm_total=total, kv_budget=budget.to("GB"),
            kv_per_token=per_token, kv_per_request=per_request.to("GB"),
            max_requests=int((budget / per_request).to("").magnitude),
        )


class PrecisionDividendModel(ForwardModel):
    """
    The maximum batch before and after lower-precision weights and KV cache.

    Weights and each request's KV cache split evenly across ``tp_size``
    devices. Lower-precision weights raise each device's KV budget once;
    lower-precision KV shrinks every request's share. The dividend is the
    ratio of the two maximum batches.
    """
    requires = ("workload",)
    produces = PrecisionDividendResult

    def solve(self, model, tp_size: int, hbm_per_accelerator: Quantity, context_tokens: int,
              kv_heads: int, weight_bytes: Quantity, weight_bytes_low: Quantity,
              kv_bytes: Quantity, kv_bytes_low: Quantity) -> PrecisionDividendResult:
        head_dim = model.hidden_dim // model.heads
        w = model.size_in_bytes(weight_bytes) / tp_size
        w_low = model.size_in_bytes(weight_bytes_low) / tp_size
        kv = calc_kv_cache_size(model.layers, kv_heads, head_dim, context_tokens, 1, kv_bytes)
        kv_low = calc_kv_cache_size(model.layers, kv_heads, head_dim, context_tokens, 1, kv_bytes_low)
        budget, budget_low = hbm_per_accelerator - w, hbm_per_accelerator - w_low
        b = int((budget / (kv / tp_size)).to("").magnitude)
        b_low = int((budget_low / (kv_low / tp_size)).to("").magnitude)
        return PrecisionDividendResult(
            weights_per_device=w, weights_per_device_low=w_low,
            kv_budget_per_device=budget, kv_budget_per_device_low=budget_low,
            kv_per_request=kv, kv_per_request_low=kv_low,
            kv_per_request_per_device=kv / tp_size, kv_per_request_per_device_low=kv_low / tp_size,
            max_batch=b, max_batch_low=b_low, dividend=b_low / b,
            budget_gain=(budget_low / budget).to("").magnitude,
        )


