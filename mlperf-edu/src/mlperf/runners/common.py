from __future__ import annotations

import os
from copy import deepcopy


def configured_seed(default: int = 42) -> int:
    """Return the benchmark seed from the shared, documented environment contract."""
    for name in ("MLPERF_EDU_SEED", "MLPERF_EDU_MAX_SEED"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return int(default)


def synchronize_device(device) -> None:
    """Synchronize supported accelerators at a measurement boundary."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def training_measurement_protocol(workload) -> dict:
    """Copy the registry-owned canonical training timing boundary."""
    protocol = workload.raw.get("measurement_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(
            f"{workload.id} does not declare a training measurement protocol"
        )
    return deepcopy(protocol)
