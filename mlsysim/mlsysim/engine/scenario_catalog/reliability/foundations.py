"""Named reliability scenarios for foundations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....systems.registry import Clusters as _Clusters
from ....systems.reliability import Reliability as _Reliability
from ... import scenario_types as _st


GpuFailuresTraining10K = _st.FailureIntervalScenario(
    name="GPU failures across 10,000 H100s",
    description="Independent GPU failures at the per-GPU MTTF over a two-week run.",
    component_mtbf=Q_(_Reliability.Gpu.mttf_hours, "hour"),
    n_components=_Clusters.Training_10K.total_accelerators,
    run_duration=Q_(14, "day"),  # scenario assumption: two-week training run
)


NodeFailuresTraining1K = _st.FailureIntervalScenario(
    name="Node failures across the 128-node reference cluster",
    description="Composite DGX node failures (GPUs, hosts, power, links) over a two-week run.",
    component_mtbf=Q_(_Reliability.DgxNodeComposite.mttf_hours, "hour"),
    n_components=_Clusters.Training_1K.count,
    run_duration=Q_(14, "day"),
)
