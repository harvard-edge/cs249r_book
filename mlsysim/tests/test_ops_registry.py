from __future__ import annotations

import pytest

from mlsysim import Ops


def test_runtime_overhead_latency_profiles():
    assert Ops.RuntimeOverheads.PythonDispatch.to("microsecond").magnitude == pytest.approx(10)
    assert Ops.RuntimeOverheads.KernelLaunch.to("microsecond").magnitude == pytest.approx(5)
    assert Ops.RuntimeOverheads.TinyMemoryAccess.to("microsecond").magnitude == pytest.approx(1)
