"""
Example 04: The Data Wall
-------------------------
This script demonstrates the "Data Wall" concept from Volume 2, Lab 4.
It shows how faster GPUs can actually result in lower utilization if the 
storage bandwidth cannot keep up with the compute demand.
"""
import mlsysim
from mlsysim.core.constants import Q_

def main():
    print("Evaluating ResNet-50 Training Data Pipeline...\n")

    # 1. Define the Workload
    model = mlsysim.Models.Vision.ResNet50
    batch_size = 256
    
    # Calculate data demand per step
    # ResNet50 input: 224x224x3 FP16 (2 bytes)
    bytes_per_sample = 224 * 224 * 3 * 2
    batch_bytes = bytes_per_sample * batch_size * Q_("1 byte")
    
    # 2. Define the Hardware Targets
    v100 = mlsysim.Hardware.Cloud.V100
    a100 = mlsysim.Hardware.Cloud.A100
    h100 = mlsysim.Hardware.Cloud.H100
    
    # 3. Define the Storage (A standard NVMe SSD)
    storage_bw = Q_("3.0 GB/s")
    
    print(f"Storage Bandwidth: {storage_bw}")
    print(f"Batch Data Size: {batch_bytes.to('MB'):.2f}\n")
    
    print(f"{'GPU':<10} | {'Compute Time':<15} | {'I/O Time':<15} | {'GPU Utilization':<15}")
    print("-" * 65)
    
    for hw in [v100, a100, h100]:
        # Calculate Compute Time
        prof = mlsysim.Engine.solve(
            model=model,
            hardware=hw,
            batch_size=batch_size,
            precision="fp16",
            is_training=True
        )
        t_compute = prof.latency
        
        # Calculate I/O Time
        t_io = (batch_bytes / storage_bw).to("ms")
        
        # Calculate Utilization (assuming no perfect overlap for simplicity, or just taking the ratio)
        # If I/O takes longer than compute, GPU is idle waiting for data.
        # Utilization = T_compute / (T_compute + T_io)  (if purely sequential)
        # Or if pipelined: T_compute / max(T_compute, T_io)
        utilization = t_compute / max(t_compute, t_io)
        
        print(f"{hw.name:<10} | {t_compute.m_as('ms'):.1f} ms        | {t_io.m_as('ms'):.1f} ms        | {utilization.m_as('dimensionless') * 100:.1f}%")

    print("\nConclusion: As GPUs get faster (V100 -> H100), the compute time drops.")
    print("But because storage bandwidth is fixed, the GPU spends more time waiting for data,")
    print("causing utilization to plummet. This is the Data Wall.")

if __name__ == "__main__":
    main()

# Expected output (mlsysim v0.1.0):
# Evaluating ResNet-50 Training Data Pipeline...
#
# Storage Bandwidth: 3.0 GB/s
# Batch Data Size: 77.07 MB
#
# GPU        | Compute Time    | I/O Time        | GPU Utilization
# -----------------------------------------------------------------
# NVIDIA V100 | 51.9 ms        | 25.7 ms        | 100.0%
# NVIDIA A100 | 41.9 ms        | 25.7 ms        | 100.0%
# NVIDIA H100 | 7.9 ms        | 25.7 ms        | 30.7%
#
# Conclusion: As GPUs get faster (V100 -> H100), the compute time drops.
# But because storage bandwidth is fixed, the GPU spends more time waiting for data,
# causing utilization to plummet. This is the Data Wall.
