"""
Example 03: Heterogeneous Cluster Modeling
Demonstrates how to model a cluster with mixed hardware types (e.g., compute nodes and storage nodes).
"""
from mlsysim.systems.types import Fleet, Node, NodeGroup, NetworkFabric
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.core.solver import DistributedModel, EconomicsModel
from mlsysim.core.constants import Q_

# 1. Define the Workload
model = Models.Language.Llama3_8B

# 2. Define the Heterogeneous Fleet
compute_group = NodeGroup(
    name="Compute Tier",
    node=Node(
        name="H100 Node",
        accelerator=Hardware.H100,
        accelerators_per_node=8,
        intra_node_bw=Q_("900 GB/s")
    ),
    count=16,
    role="compute"
)

storage_group = NodeGroup(
    name="Storage Tier",
    node=Node(
        name="Storage Node",
        accelerator=Hardware.StorageServer,
        accelerators_per_node=0,
        intra_node_bw=Q_("100 GB/s")
    ),
    count=4,
    role="storage"
)

fleet = Fleet(
    name="Mixed AI Cluster",
    node_groups=[compute_group, storage_group],
    fabric=NetworkFabric(name="400G Fabric", bandwidth=Q_("400 GB/s"))
)

# 3. Evaluate Performance (Only looks at compute tier)
dist_solver = DistributedModel()
perf_result = dist_solver.solve(
    model=model,
    fleet=fleet,
    batch_size=1024,
    dp_size=16,
    tp_size=8,
    pp_size=1
)

print(f"--- Performance (Compute Tier Only) ---")
print(f"Step Latency: {perf_result.step_latency_total:.2f}")
print(f"Scaling Efficiency: {perf_result.scaling_efficiency * 100:.1f}%")

# 4. Evaluate Economics (Looks at entire fleet)
econ_solver = EconomicsModel()
econ_result = econ_solver.solve(fleet=fleet, duration_days=30)

print(f"\n--- Economics (Entire Fleet) ---")
print(f"Total CapEx: ${econ_result.capex_usd:,.2f}")
print(f"Total OpEx (30 days): ${econ_result.total_opex_usd:,.2f}")
print(f"Total TCO: ${econ_result.total_tco_usd:,.2f}")
