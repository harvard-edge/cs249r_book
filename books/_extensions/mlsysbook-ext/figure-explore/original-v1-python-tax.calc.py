#!/usr/bin/env python3
"""Original PythonTaxTimeline scenario constants. PYTHONPATH=mlsysim /tmp/venv/bin/python this.py"""
import json
from mlsysim import Ops,microsecond
n_ops=5
python_dispatch=Ops.RuntimeOverheads.PythonDispatch.to(microsecond).magnitude
kernel_time=1 # chapter-declared illustrative small-kernel microseconds
launch=Ops.RuntimeOverheads.KernelLaunch.to(microsecond).magnitude
fused_efficiency=.8 # original PythonTaxTimeline scenario, not a general fusion law
print(json.dumps(dict(n_ops=n_ops,dispatch_us=python_dispatch,kernel_us=kernel_time,launch_us=launch,fused_compute_us=n_ops*kernel_time*fused_efficiency)))
