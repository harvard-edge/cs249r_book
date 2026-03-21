#!/usr/bin/env python3
"""
Example 1: The Roofline Model
-----------------------------
This script demonstrates the absolute core of the MLSys·im framework:
evaluating a workload against a specific piece of hardware to find the bottleneck.
"""
import mlsysim

def main():
    print("Evaluating Llama-3 8B on an NVIDIA A100 vs H100...\n")

    # 1. Define the Workload
    model = mlsysim.Models.Language.Llama3_8B

    # 2. Define the Hardware targets
    a100 = mlsysim.Hardware.Cloud.A100
    h100 = mlsysim.Hardware.Cloud.H100

    # 3. Evaluate the A100
    print("--- A100 Performance ---")
    prof_a100 = mlsysim.Engine.solve(
        model=model,
        hardware=a100,
        batch_size=1,
        precision="fp16"
    )
    print(prof_a100.summary())

    print("\n--- H100 Performance ---")
    # 4. Evaluate the H100
    prof_h100 = mlsysim.Engine.solve(
        model=model,
        hardware=h100,
        batch_size=1,
        precision="fp16"
    )
    print(prof_h100.summary())
    
    print("\nConclusion: Notice that despite having 3.2x more FLOPs, the H100 only yields ")
    print("a ~1.7x speedup for this batch size 1 inference, because the workload is Memory Bound!")

if __name__ == "__main__":
    main()
