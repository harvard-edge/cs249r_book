"""Explicit estimated-energy helper for local educational runs."""

from __future__ import annotations

import platform
import subprocess
import time
from typing import Any


class PowerMeter:
    """Record elapsed time and a clearly labeled nominal-power estimate."""

    def __init__(self, nominal_watts: float | None = None):
        self.start_time: float | None = None
        self.nominal_watts = nominal_watts or default_nominal_watts()

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> float:
        return self.stop_report()["energy_joules"]

    def stop_report(self) -> dict[str, Any]:
        if self.start_time is None:
            return {
                "source": "estimated_nominal",
                "average_watts": 0.0,
                "duration_seconds": 0.0,
                "energy_joules": 0.0,
            }
        duration = time.time() - self.start_time
        return {
            "source": "estimated_nominal",
            "average_watts": round(float(self.nominal_watts), 3),
            "duration_seconds": round(float(duration), 6),
            "energy_joules": round(float(duration * self.nominal_watts), 6),
        }


def default_nominal_watts() -> float:
    """Return a labeled estimate, using a live NVIDIA reading when available."""
    if platform.system() == "Darwin":
        return 15.2
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(result.decode("utf-8").strip().splitlines()[0])
    except Exception:
        return 65.0
