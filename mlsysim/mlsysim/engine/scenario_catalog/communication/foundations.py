"""Named communication scenarios for foundations.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....models.registry import Models as _Models
from ....hardware.registry import Hardware as _Hardware
from ....systems.registry import Fabrics as _Fabrics, Nodes as _Nodes
from ... import scenario_types as _st


BufferTransfer10GB = _st.TransferScenario(
    name="10 GB buffer across the H100 bandwidth hierarchy",
    description="One synchronized 10 GB buffer moved one hop on each tier: local HBM, "
    "NVLink inside a node (one direction), InfiniBand NDR between nodes, "
    "PCIe Gen5 to the host. Bandwidth-only bounds.",
    payload=Q_(10, "GB"),  # scenario assumption: synchronized buffer size
    links={
        "HBM": _Hardware.Cloud.H100.memory.bandwidth,
        "NVLink": _Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
        "InfiniBand": _Fabrics.InfiniBand_NDR.bandwidth,
        "PCIe": _Hardware.Cloud.H100.interconnect.bandwidth,
    },
)


Gpt3TensorParallelStepDgxH100 = _st.TensorParallelScenario(
    name="175B tensor-parallel step inside one DGX H100",
    description="GPT-3 175B split eight ways inside one DGX H100 node; one activation "
    "AllReduce per layer per pass over NVLink; 200 ms of compute is a stated input.",
    workload=_Models.Language.GPT3,
    tp_size=_Nodes.DGX_H100.accelerators_per_node,
    microbatch=4,  # scenario assumption: sequences per microbatch
    link_bandwidth=_Hardware.Cloud.H100.nvlink.bandwidth_per_direction,
    compute_time=Q_(200, "ms"),  # scenario assumption: compute per step, not derived
)
