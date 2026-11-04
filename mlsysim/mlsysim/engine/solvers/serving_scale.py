"""Serving-at-scale solvers: request traces, measured frontiers, KV admission, routing.

Domain implementations behind ``mlsysim.solvers`` (the public import path,
derived from ``engine.solvers.__init__``). The equations live in
``physics.serving`` (Little's law, the batching knee, padding waste,
speculative rounds) and ``physics.memory`` (KV-cache size); these models
apply them to a published trace, a benchmark table, or one stated replica
(added 2026-10-20).
"""

from __future__ import annotations



from ..results import (
    KvAdmissionResult,
    PrecisionDividendResult,
)
from ...core.types import Quantity
from ...physics.memory import calc_kv_cache_size
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
        """Divide a replica's available KV memory by one request's KV footprint.

        Parameters
        ----------
        model : TransformerWorkload
            Transformer dimensions and parameter count; hidden width must be
            divisible by attention heads for the integer head width used here.
        accelerators : int
            Positive number of devices whose HBM is pooled into the replica.
        hbm_per_accelerator : Quantity
            Positive memory capacity per accelerator, convertible to bytes.
        weight_bytes_per_param : Quantity
            Positive bytes stored per weight element, expressed as a byte quantity.
        reserve : Quantity
            Nonnegative total replica memory reserved for other runtime state.
        context_tokens : int
            Positive resident context tokens per request.
        kv_bytes_per_elem : Quantity
            Positive bytes stored per key or value element.

        Returns
        -------
        KvAdmissionResult
            Weights and KV budget in GB, per-token/per-request cache storage, and
            integer request capacity. Negative budgets are not clamped; they indicate
            an infeasible allocation rather than a valid admission count.

        See Also
        --------
        mlsysim.physics.memory.calc_kv_cache_size : Key/value tensor footprint.

        Notes
        -----
        Equation: KV_request = ``2*layers*kv_heads*head_dim*context_tokens*element_bytes``;
        budget = ``accelerators*HBM`` - weights - reserve;
        max_requests = int(budget/KV_request).

        Assumptions: uniform request lengths and a perfectly pooled replica budget.
        For a nonnegative budget, integer truncation implements floor division.
        Missing kv_heads falls back to the query-head count.

        Excludes: per-device imbalance, communication, block fragmentation, and
        workspace beyond the explicitly supplied reserve.

        References
        ----------
        Derivation: count both key and value tensors for every layer and resident
        token, subtract static allocation, and divide remaining bytes by request bytes.

        Examples
        --------
        >>> from mlsysim import Models, Q_
        >>> result = KvAdmissionModel().solve(Models.Language.Llama3_8B, 1,
        ...     Q_("80 GB"), Q_("2 byte"), Q_("4 GB"), 1024, Q_("2 byte"))
        >>> result.max_requests > 0
        True
        """
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
        """Compare KV admission limits before and after reducing storage precision.

        Parameters
        ----------
        model : TransformerWorkload
            Transformer dimensions and weight count; hidden width must divide by heads.
        tp_size : int
            Positive tensor-parallel degree for even weight and KV sharding.
        hbm_per_accelerator : Quantity
            Positive device capacity, convertible to bytes.
        context_tokens : int
            Positive resident tokens per request.
        kv_heads : int
            Positive key/value-head count in the modeled attention layout.
        weight_bytes : Quantity
            Positive baseline bytes per weight element.
        weight_bytes_low : Quantity
            Positive alternative bytes per weight element.
        kv_bytes : Quantity
            Positive baseline bytes per key/value element.
        kv_bytes_low : Quantity
            Positive alternative bytes per key/value element.

        Returns
        -------
        PrecisionDividendResult
            Weight and KV byte budgets per device, whole-request capacities, and
            dimensionless capacity dividend. Baseline capacity must be positive
            because the dividend divides by it; no feasibility flag is returned.

        See Also
        --------
        mlsysim.physics.memory.calc_kv_cache_size : KV tensor footprint.

        Notes
        -----
        Equation: budget = HBM - weights/tp_size;
        batch = int(budget / (KV_request/tp_size)); dividend = batch_low/batch.
        Both branches use the same context and head count, varying only element sizes.

        Assumptions: even sharding and identical devices; usable HBM is devoted only
        to weights and KV. Integer truncation is a floor only for nonnegative budgets.

        Excludes: quantization accuracy, dequantization cost, workspace, allocator
        reserve, and communication. A memory dividend is not a throughput prediction.

        References
        ----------
        Derivation: divide per-device free bytes by the per-device request footprint
        for each representation, then compare the integer capacities.

        Examples
        --------
        >>> from mlsysim import Models, Q_
        >>> result = PrecisionDividendModel().solve(Models.Language.Llama3_8B, 1,
        ...     Q_("80 GB"), 1024, 8, Q_("2 byte"), Q_("1 byte"), Q_("2 byte"), Q_("1 byte"))
        >>> result.max_batch_low >= result.max_batch
        True
        """
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


