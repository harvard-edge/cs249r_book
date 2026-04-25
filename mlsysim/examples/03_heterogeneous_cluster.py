"""
Example 03: Cluster Modeling
Demonstrates how to model a multi-node GPU cluster and evaluate
distributed training performance + economics.
"""
from mlsysim.systems.types import Fleet, Node, NetworkFabric
from mlsysim.hardware.registry import Hardware
from mlsysim.models.registry import Models
from mlsysim.core.solver import DistributedModel, EconomicsModel
from mlsysim.core.constants import Q_

# 1. Define the Workload
model = Models.Language.Llama3_8B

# 2. Define the Cluster
node = Node(
    name="H100 Node",
    accelerator=Hardware.H100,
    accelerators_per_node=8,
    intra_node_bw=Q_("900 GB/s")  # NVLink 4.0
)

fleet = Fleet(
    name="Training Cluster",
    node=node,
    count=16,  # 16 nodes = 128 GPUs
    fabric=NetworkFabric(name="InfiniBand NDR", bandwidth=Q_("400 Gbit/s"))
)

# 3. Evaluate Distributed Training Performance
dist_solver = DistributedModel()
perf = dist_solver.solve(
    model=model,
    fleet=fleet,
    batch_size=1024,
    tp_size=8,       # Tensor parallelism within each node
    pp_size=1,
    efficiency=0.45
)

print("--- Distributed Training Performance ---")
print(f"Step Latency:       {perf.step_latency_total:.2f}")
print(f"Scaling Efficiency: {perf.scaling_efficiency * 100:.1f}%")
print(f"DP Communication:   {perf.dp_communication_latency:.2f}")
print(f"TP Communication:   {perf.tp_communication_latency:.2f}")

# 4. Evaluate Economics (30-day training run)
econ_solver = EconomicsModel()
econ = econ_solver.solve(fleet=fleet, duration_days=30)

print(f"\n--- Economics (30-day run) ---")
print(f"CapEx (amortized): ${econ.capex_usd:,.0f}")
print(f"OpEx (energy):     ${econ.opex_energy_usd:,.0f}")
print(f"Total TCO:         ${econ.tco_usd:,.0f}")
print(f"Carbon Footprint:  {econ.carbon_footprint_kg:.0f} kg CO₂")

# Expected output (mlsysim v0.1.1):
# --- Distributed Training Performance ---
# Step Latency:       2004.60 millisecond
# Scaling Efficiency: 92.7%
# DP Communication:   0.01 second
# TP Communication:   0.13 second
#
# --- Economics (30-day run) ---
# CapEx (amortized): $105,205
# OpEx (energy):     $8,670
# Total TCO:         $129,657
# Carbon Footprint:  30997 kg CO2
