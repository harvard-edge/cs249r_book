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
    BallsIntoBinsResult,
    KvAdmissionResult,
    PrecisionDividendResult,
    RequestTraceResult,
    ServingFrontierResult,
    TrafficProfileResult,
)
from ...core.types import Quantity
from ...core.units import Q_, ureg
from ...physics.memory import calc_kv_cache_size
from ...physics.serving import calc_padding_waste_fraction
from .base import ForwardModel

_TRACE_TIME = "%Y-%m-%d %H:%M:%S.%f"
_BIN_TIME = "%Y-%m-%d %H:%M"
_WEEKDAYS = 5                       # Monday to Friday: datetime.weekday() below 5


def _parse(stamp: str, fmt: str) -> datetime:
    # The trace carries seven fractional digits; strptime reads six.
    return datetime.strptime(stamp[:26] if fmt == _TRACE_TIME else stamp, fmt)


class RequestTraceModel(ForwardModel):
    """
    Arrival rate, token lengths, and fixed-batch slot waste of a request trace.

    The idle share groups consecutive requests into batches of ``batch_size``
    that keep every slot until the longest answer ends (``1 - mean/max`` per
    batch, averaged); a trailing partial batch is dropped. Per-second counts
    bin arrivals by whole seconds since the first arrival, empty seconds
    included.
    """
    requires = ()
    produces = RequestTraceResult

    def solve(self, timestamps: Sequence[str], prompt_tokens: Sequence[float],
              output_tokens: Sequence[float], batch_size: int = 8) -> RequestTraceResult:
        t = [_parse(s, _TRACE_TIME) for s in timestamps]
        out = np.array(output_tokens, dtype=float)
        n = len(out) // batch_size
        idle = [calc_padding_waste_fraction(b) for b in out[: n * batch_size].reshape(n, batch_size)]
        secs = np.array([(x - t[0]).total_seconds() for x in t]).astype(int)
        per = np.bincount(secs)
        duration = Q_((t[-1] - t[0]).total_seconds(), "second")
        return RequestTraceResult(
            requests=len(t),
            duration=duration,
            arrival_rate=(len(t) / duration).to("1/second"),
            median_prompt=statistics.median(prompt_tokens),
            median_output=statistics.median(output_tokens),
            mean_prompt=float(np.mean(prompt_tokens)),
            mean_output=float(out.mean()),
            max_output=float(out.max()),
            batch_size=batch_size,
            fixed_batch_idle_share=float(np.mean(idle)),
            per_second_mean=float(per.mean()),
            per_second_peak=int(per.max()),
        )


class TrafficProfileModel(ForwardModel):
    """
    Peak, trough, and the weekday day-night swing of a binned request series.

    Each bin's rate is its count over ``bin_seconds``. For every weekday the
    model takes the day's lowest and highest bin; the swing is the mean of
    their ratio.
    """
    requires = ()
    produces = TrafficProfileResult

    def solve(self, bin_starts: Sequence[str], counts: Sequence[float],
              bin_seconds: Quantity) -> TrafficProfileResult:
        width = bin_seconds.to("second").magnitude
        rates = np.array(counts, dtype=float) / width
        days: Dict[str, list] = {}
        for stamp, r in zip(bin_starts, rates):
            if _parse(stamp, _BIN_TIME).weekday() < _WEEKDAYS:
                days.setdefault(stamp[:10], []).append(r)
        troughs = [min(d) for d in days.values()]
        peaks = [max(d) for d in days.values()]
        per_s = ureg("1/second")
        return TrafficProfileResult(
            peak_rate=float(rates.max()) * per_s, min_rate=float(rates.min()) * per_s,
            weekday_trough_low=min(troughs) * per_s, weekday_trough_high=max(troughs) * per_s,
            weekday_peak_low=min(peaks) * per_s, weekday_peak_high=max(peaks) * per_s,
            weekday_peak_to_trough=float(np.mean([p / t for p, t in zip(peaks, troughs)])),
        )


class ServingFrontierModel(ForwardModel):
    """
    The measured throughput-interactivity frontier of serving configurations.

    Each row is one measured configuration with an interactivity (output
    tokens per second each user receives, 1 / TPOT) and a per-accelerator
    throughput. The frontier keeps the configurations no other configuration
    on the same accelerator beats on both rates; the best throughput at a
    target is the highest throughput among rows that meet it.
    """
    requires = ()
    produces = ServingFrontierResult

    @staticmethod
    def frontier(rows: Sequence[dict], x: str, y: str):
        pts = sorted(((r[x], r[y]) for r in rows), key=lambda p: -p[0])
        out, best = [], -1.0
        for a, b in pts:
            if b > best:
                out.append((a, b))
                best = b
        return sorted(out)

    @staticmethod
    def best_at(rows: Sequence[dict], x: str, y: str, minimum: float) -> float:
        return max(r[y] for r in rows if r[x] >= minimum)

    def solve(self, rows: Sequence[dict], x: str, y: str, minimum: float) -> ServingFrontierResult:
        f = self.frontier(rows, x, y)
        peak = max(f, key=lambda p: p[1])
        return ServingFrontierResult(frontier=f, peak_interactivity=peak[0], peak_throughput=peak[1],
                                     best_at=self.best_at(rows, x, y, minimum))


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


class BallsIntoBinsModel(ForwardModel):
    """
    Busiest-bin load when ``n`` equal jobs go into ``n`` empty bins, no departures.

    With one choice each job picks a bin at random; with two it probes two
    random bins and joins the less loaded (Mitzenmacher 2001). Each bin count
    is averaged over ``trials`` runs. One seeded generator draws every run in
    order (all one-choice runs, then all two-choice runs), so a seed fixes the
    whole table.
    """
    requires = ()
    produces = BallsIntoBinsResult

    def solve(self, bins: Sequence[int], choices: Sequence[int], trials: Dict[int, int],
              seed: int) -> BallsIntoBinsResult:
        rng = np.random.default_rng(seed)

        def busiest(n: int, d: int) -> int:
            if d == 1:
                return int(np.bincount(rng.integers(0, n, n), minlength=n).max())
            load = np.zeros(n, dtype=int)
            a, b = rng.integers(0, n, n), rng.integers(0, n, n)
            for i in range(n):
                j = a[i] if load[a[i]] <= load[b[i]] else b[i]
                load[j] += 1
            return int(load.max())

        table = {d: [float(np.mean([busiest(n, d) for _ in range(trials[n])])) for n in bins]
                 for d in choices}
        return BallsIntoBinsResult(bins=list(bins), max_load=table)
