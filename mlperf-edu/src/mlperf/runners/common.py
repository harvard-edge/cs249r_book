from __future__ import annotations

import os


def configured_seed(default: int = 42) -> int:
    """Return the benchmark seed from the shared, documented environment contract."""
    for name in ("MLPERF_EDU_SEED", "MLPERF_EDU_MAX_SEED", "MLPERF_EDU_SLM_SEED"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return int(default)
