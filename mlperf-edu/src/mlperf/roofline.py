"""
MLPerf EDU: Roofline-coordinate emitter.

Wraps a workload's hot loop in a context manager that measures wall time,
then divides caller-supplied analytic FLOP and byte counts to produce
(arithmetic intensity, achieved FLOPS, achieved bandwidth, dispatch
utilization). Writes a JSON sidecar consumed by the iter-5 provenance
chain (manifest.py) and the iter-4 taxonomy linter (check_taxonomy.py).

Per Dean's iter-5 spec: emitter is the instrument that produces the
telemetry the provenance chain hashes. Splitting them would mean the
manifest hashes nothing real and the emitter produces numbers no one
verifies.
"""
from __future__ import annotations

import json
import time
import os
import datetime
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


SCHEMA_VERSION = "mlperf-edu-roofline/1.0"

# M1 reference platform peaks. These are first-pass heuristics; iter-5.5
# adds bench/peak_flops_mps.py and bench/peak_bw_mps.py to measure them
# per-machine. For the iter-5 smoke test these defaults give a usable
# ridge point.
DEFAULT_PEAK_FLOPS = 2.6e12   # M1 8-core GPU fp32, rough
DEFAULT_PEAK_BW_GBPS = 68.25  # M1 unified memory peak

# Threshold rules (must match check_taxonomy.py).
_RIDGE_LOW_MULTIPLIER = 0.5   # bandwidth_bound if intensity < 0.5*ridge
_RIDGE_HIGH_MULTIPLIER = 2.0  # compute_bound if intensity > 2*ridge
_DISPATCH_BOUND_UTIL = 0.25   # dispatch_bound if utilization < 0.25
_DEVICE_SAT_UTIL = 0.50       # device_saturated if utilization > 0.50


@dataclass
class RooflineMeasurement:
    workload: str
    sidecar_path: Path
    achieved_flops: float
    achieved_bw_gbps: float
    intensity: float
    dispatch_utilization: float
    axis_arithmetic_intensity: str
    axis_dispatch: str

    def summary(self) -> str:
        return (f"{self.workload}: intensity={self.intensity:.2f} FLOP/byte, "
                f"util={self.dispatch_utilization:.3f}, "
                f"AI={self.axis_arithmetic_intensity}, "
                f"dispatch={self.axis_dispatch}")


def _classify_axes(intensity: float, utilization: float, ridge: float) -> tuple[str, str]:
    if intensity > _RIDGE_HIGH_MULTIPLIER * ridge:
        ai = "compute_bound"
    elif intensity < _RIDGE_LOW_MULTIPLIER * ridge:
        ai = "bandwidth_bound"
    else:
        ai = "unmeasured"  # in grey band; honest.
    if utilization >= _DEVICE_SAT_UTIL:
        di = "device_saturated"
    elif utilization < _DISPATCH_BOUND_UTIL:
        di = "dispatch_bound"
    else:
        di = "unmeasured"
    return ai, di


def _hardware_fingerprint_short() -> str:
    """Short hash of the platform for sidecar naming."""
    try:
        from .hardware import profile_hardware
        fp = profile_hardware()
    except Exception:
        fp = {"system": "unknown"}
    payload = json.dumps(fp, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _sync():
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


@contextmanager
def measure_roofline(workload_name: str,
                      analytic_flops: Callable[[], float] | float,
                      analytic_bytes: Callable[[], float] | float,
                      n_iter: int = 1,
                      output_dir: str | Path = "roofline",
                      peak_flops: float = DEFAULT_PEAK_FLOPS,
                      peak_bw_gbps: float = DEFAULT_PEAK_BW_GBPS,
                      machine_class: str = "apple-m1-16gb",
                      ) -> Iterator[dict]:
    """Wrap a hot loop and emit a roofline sidecar on exit.

    Usage:
        with measure_roofline("nanogpt-prefill",
                               analytic_flops=lambda: 2 * n_params * ctx,
                               analytic_bytes=lambda: n_params * 4 + activation_bytes,
                               n_iter=200) as ctx_dict:
            for _ in range(n_iter):
                model(x)

    The yielded dict is mutable: callers can stuff additional context
    (e.g. {"context_length": 1792}) and it will be persisted into the
    sidecar's "extra" field.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra: dict = {}
    _sync()
    t0 = time.perf_counter()
    try:
        yield extra
    finally:
        _sync()
        wall_time = time.perf_counter() - t0
        flops = analytic_flops() if callable(analytic_flops) else float(analytic_flops)
        byts = analytic_bytes() if callable(analytic_bytes) else float(analytic_bytes)
        # Per-iter metrics if n_iter > 1; otherwise aggregate.
        per_iter_time = wall_time / max(n_iter, 1)
        per_iter_flops = flops / max(n_iter, 1) if n_iter > 1 else flops
        per_iter_bytes = byts / max(n_iter, 1) if n_iter > 1 else byts
        achieved_flops = per_iter_flops / per_iter_time if per_iter_time > 0 else 0.0
        achieved_bw = (per_iter_bytes / per_iter_time / 1e9) if per_iter_time > 0 else 0.0
        intensity = (per_iter_flops / per_iter_bytes) if per_iter_bytes > 0 else float("inf")
        util = max(achieved_flops / peak_flops if peak_flops > 0 else 0.0,
                   achieved_bw / peak_bw_gbps if peak_bw_gbps > 0 else 0.0)
        ridge = peak_flops / (peak_bw_gbps * 1e9) if peak_bw_gbps > 0 else 0.0
        axis_ai, axis_disp = _classify_axes(intensity, util, ridge)

        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        hw_short = _hardware_fingerprint_short()
        sidecar_path = out_dir / f"{workload_name}_{ts}_{hw_short}.json"
        sidecar = {
            "schema": SCHEMA_VERSION,
            "workload": workload_name,
            "utc": ts,
            "platform": {
                "hardware_fingerprint_short": hw_short,
                "machine_class": machine_class,
                "peak_FLOPS": peak_flops,
                "peak_BW_GBps": peak_bw_gbps,
                "ridge_FLOPS_per_byte": ridge,
            },
            "measurement": {
                "wall_time_s": wall_time,
                "n_iter": n_iter,
                "analytic_flops_total": flops,
                "analytic_bytes_total": byts,
                "achieved_FLOPS": achieved_flops,
                "achieved_BW_GBps": achieved_bw,
                "intensity_FLOPS_per_byte": intensity,
                "dispatch_utilization": util,
            },
            "regime_inference": {
                "axis_arithmetic_intensity": axis_ai,
                "axis_dispatch": axis_disp,
                "rule": (f"intensity {intensity:.2f} vs "
                         f"[low {_RIDGE_LOW_MULTIPLIER*ridge:.2f}, "
                         f"high {_RIDGE_HIGH_MULTIPLIER*ridge:.2f}]; "
                         f"util {util:.3f} vs "
                         f"[dispatch {_DISPATCH_BOUND_UTIL}, sat {_DEVICE_SAT_UTIL}]"),
            },
            "extra": extra,
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))
        extra["_sidecar_path"] = str(sidecar_path)
        # Surface for downstream consumers (loadgen.py reads this env var).
        os.environ["MLPERF_EDU_LAST_SIDECAR"] = str(sidecar_path)


def latest_sidecar(workload: str, output_dir: str | Path = "roofline") -> Path | None:
    """Return the most recent sidecar for a given workload (by mtime), or None."""
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return None
    matches = sorted(out_dir.glob(f"{workload}_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None
