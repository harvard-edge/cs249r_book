"""Named training scenarios for data parallel.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Fabrics as _Fabrics, NetworkEnergy as _NetworkEnergy
from ... import scenario_types as _st


# --- Data-parallel training at fleet scale (added 2026-10-20) -----------
Gpt3GradientSync = _st.GradientSyncScenario(
    name="175B data-parallel gradient sync over InfiniBand HDR and 100G Ethernet",
    description="One worker computes for a stated 1.2 s, then ring-AllReduces 16-bit "
    "gradients with no overlap; energy charges each bit once on InfiniBand.",
    workload=_Models.Language.GPT3,
    precision="fp16",
    links={"InfiniBand": _Fabrics.InfiniBand_HDR.bandwidth, "Ethernet": _Fabrics.Ethernet_100G.bandwidth},
    compute_time=Q_(1.2, "second"),  # scenario assumption: illustrative compute per step, not a measured GPT-3 step
    energy_per_bit=_NetworkEnergy.InfiniBandEnergyPerBit,
    steps=1_000_000,  # scenario assumption: steps in the run
)


Gpt3GradientSyncFasterAccelerator = Gpt3GradientSync.variant(
    name="The same step on accelerators with twice the compute rate",
    description="Only the compute term halves; the exchange is unchanged.",
    compute_speedup=2.0,  # scenario assumption: one accelerator generation
)


Gpt3DataParallelIntensity = _st.DataParallelIntensityScenario(
    name="Communication intensity of a 175B data-parallel worker",
    description="Four full-length GPT-3 sequences per worker per step; balance of one A100 on one 200 Gb/s HDR link.",
    workload=_Models.Language.GPT3,
    sequences_per_worker=4,  # scenario assumption: local batch of four sequences
    link_bandwidth=_Fabrics.InfiniBand_HDR.bandwidth,
    accelerator=_Hardware.Cloud.A100,
)


NodeAvailability1K = _st.AvailabilityScenario(
    name="1,000 nodes at 99.9 percent",
    description="Illustrative per-node availability, independent failures.",
    availability=0.999,
    n_components=1000,
)


NodeAvailability10K = NodeAvailability1K.variant(name="10,000 nodes at 99.9 percent", n_components=10_000)


FourNinesAvailability10K = NodeAvailability1K.variant(
    name="10,000 nodes at 99.99 percent", availability=0.9999, n_components=10_000
)


CoordinationOverheadExample = _st.ScalingOverheadScenario(
    name="A planned 3x speedup with 40 percent overhead",
    description="Communication and coordination take 40 percent of the scaled step.",
    planned_speedup=3.0,
    overhead_fraction=0.40,
)


Gpt3PublishedRun = _st.PublishedRunScenario(
    name="GPT-3 175B as trained",
    description="GPT-3's training compute over its reference V100 count and days (Patterson et al. 2021).",
    workload=_Models.Language.GPT3,
    accelerator=_Hardware.Cloud.V100,
)
