from __future__ import annotations

import os
import sys
import time
from copy import deepcopy


def configured_seed(default: int = 42) -> int:
    """Return the benchmark seed from the shared, documented environment contract."""
    for name in ("MLPERF_EDU_SEED", "MLPERF_EDU_MAX_SEED"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return int(default)


def select_torch_device():
    """Select the requested device with one consistent auto-selection policy."""
    import torch

    requested = os.environ.get("MLPERF_EDU_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device) -> None:
    """Synchronize supported accelerators at a measurement boundary."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _format_duration(seconds: float) -> str:
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"


class TrainingProgress:
    """Report progress during a long training loop.

    A run that prints nothing for half an hour is indistinguishable from one
    that has hung, and the person waiting cannot tell whether to keep waiting
    or to stop and reconfigure. That matters most in a classroom, where the run
    is someone's first contact with the suite.

    Reporting is throttled by wall-clock rather than by step count so that a
    500-epoch graph run and a 20-epoch recommendation run both stay readable,
    and it writes to stderr so it never contaminates anything a caller parses
    from stdout. Set ``MLPERF_EDU_QUIET=1`` to silence it.

    Progress output is not measurement. It is deliberately excluded from the
    timed region by callers and never enters a report.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        unit: str = "step",
        min_interval_seconds: float = 15.0,
        stream=None,
    ) -> None:
        self.label = label
        self.total = int(total)
        self.unit = unit
        self.min_interval = float(min_interval_seconds)
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = os.environ.get("MLPERF_EDU_QUIET", "") not in {"1", "true", "yes"}
        self._start = time.perf_counter()
        self._last_emit = 0.0
        self._emitted = False

    def update(self, step: int, **metrics: float) -> None:
        """Record completion of ``step`` (1-based) and emit if due.

        The first and last steps always print, so even a short run shows that
        work started and finished.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        elapsed = now - self._start
        is_edge = step <= 1 or step >= self.total
        if not is_edge and (now - self._last_emit) < self.min_interval:
            return
        self._last_emit = now
        self._emitted = True

        parts = [f"{self.label}", f"{self.unit} {step}/{self.total}"]
        for name, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{name} {value:.4g}")
            else:
                parts.append(f"{name} {value}")
        parts.append(f"elapsed {_format_duration(elapsed)}")
        if 0 < step < self.total:
            remaining = elapsed / step * (self.total - step)
            parts.append(f"eta {_format_duration(remaining)}")
        print("  ".join(parts), file=self.stream, flush=True)

    def close(self, note: str = "") -> None:
        if not self.enabled or not self._emitted:
            return
        elapsed = _format_duration(time.perf_counter() - self._start)
        suffix = f"  {note}" if note else ""
        print(f"{self.label}  done in {elapsed}{suffix}", file=self.stream, flush=True)


def training_measurement_protocol(workload) -> dict:
    """Copy the registry-owned canonical training timing boundary."""
    protocol = workload.raw.get("measurement_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            f"{workload.id} does not declare a training measurement protocol"
        )
    return deepcopy(protocol)
