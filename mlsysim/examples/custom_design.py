"""
Example: Custom System Design
=============================
This script demonstrates how to build a hypothetical system from scratch
without using the vetted registries. This is how researchers can use
mlsysim to model unreleased or generic hardware.
"""

import mlsysim
from mlsysim.hardware.types import HardwareNode, ComputeCore, MemoryHierarchy
from mlsysim.models.types import CNNWorkload
from mlsysim.core.scenarios import Scenario

def main():
    print("--- Designing a Hypothetical 'Generic Drone' ---")
    
    # 1. Manually define hardware (Supply)
    drone_chip = HardwareNode(
        name="Hypothetical Drone NPU",
        release_year=2026,
        compute=ComputeCore(peak_flops="10 TFLOPs/s"),
        memory=MemoryHierarchy(capacity="2 GB", bandwidth="50 GB/s"),
        tdp="10 W",
        dispatch_tax="0.5 ms"
    )
    
    # 2. Manually define workload (Demand)
    my_model = CNNWorkload(
        name="Custom Vision Model",
        architecture="CNN",
        parameters="50 Mparam",
        inference_flops="10 Gflop"
    )
    
    # 3. Bundle into a Scenario
    my_scenario = Scenario(
        name="Generic Drone Vision",
        description="A custom vision task on unreleased drone hardware.",
        workload=my_model,
        system=drone_chip,
        sla_latency="30 ms"
    )
    
    # 4. Evaluate the custom design
    print(f"Evaluating {my_scenario.name}...")
    report = my_scenario.evaluate()
    print(report.scorecard())

if __name__ == "__main__":
    main()

# Expected output (mlsysim v0.1.0):
# --- Designing a Hypothetical 'Generic Drone' ---
# Evaluating Generic Drone Vision...
# +============================================================+
# | MLSys-im SYSTEM EVALUATION
# | Scenario: Generic Drone Vision
# +============================================================+
# | Level 1: Feasibility [PASS]
# |   Model fits in memory (100.0 MB / 2000.0 MB)
# +------------------------------------------------------------+
# | Level 2: Performance [PASS]
# |   Latency: 2.71 millisecond (Target: 30 ms)
# +------------------------------------------------------------+
# | Level 3: Macro/Economics [PASS]
# |   Annual Carbon: 42.1 kg | TCO: $11,512
# +============================================================+
