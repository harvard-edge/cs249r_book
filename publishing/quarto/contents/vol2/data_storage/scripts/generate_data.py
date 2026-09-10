import pandas as pd
import os

def generate_storage_compute_gap_data():
    """
    Generates a CSV comparing GPU compute (TFLOPS), Memory Bandwidth (HBM, GB/s),
    and Local Storage Bandwidth (NVMe, GB/s) over time.
    
    Data Sources / Logic:
    - Compute TFLOPS tracks typical FP32 performance for earlier architectures (Fermi to Maxwell)
      and transitions to FP16 / Tensor Core performance (without sparsity) for Pascal onwards,
      as this is the most relevant metric for ML workloads.
    - Memory Bandwidth tracks the peak memory bandwidth of flagship data center GPUs
      (e.g., M2090, K20X, M40, P100, V100, A100-40GB, H100-SXM, B200).
    - Storage Bandwidth represents the approximate single-drive sequential read limits of the 
      dominant storage interface for that era:
        - 2010-2012: SATA 3Gbps to 6Gbps / early SSDs (0.3 - 0.6 GB/s)
        - 2014: Early PCIe 2.0/3.0 SSDs (1.0 GB/s)
        - 2016-2018: PCIe 3.0 x4 NVMe SSDs (4.0 GB/s)
        - 2020: PCIe 4.0 x4 NVMe SSDs (8.0 GB/s)
        - 2022-2024: PCIe 5.0 x4 NVMe SSDs (14.0 GB/s)
    """
    data = [
        {
            "Year": 2010,
            "GPU_Architecture": "Fermi",
            "Compute_TFLOPS": 1.03,        # e.g., M2090 FP32
            "Memory_Bandwidth_GBps": 144,  # e.g., M2090
            "Storage_Bandwidth_GBps": 0.3  # SATA 3Gbps / early SSDs
        },
        {
            "Year": 2012,
            "GPU_Architecture": "Kepler",
            "Compute_TFLOPS": 3.95,        # e.g., K20X FP32
            "Memory_Bandwidth_GBps": 250,  # e.g., K20X
            "Storage_Bandwidth_GBps": 0.6  # SATA 6Gbps SSDs
        },
        {
            "Year": 2014,
            "GPU_Architecture": "Maxwell",
            "Compute_TFLOPS": 7.0,         # e.g., M40 FP32
            "Memory_Bandwidth_GBps": 336,  # e.g., M40
            "Storage_Bandwidth_GBps": 1.0  # Early PCIe SSDs
        },
        {
            "Year": 2016,
            "GPU_Architecture": "Pascal",
            "Compute_TFLOPS": 21.2,        # e.g., P100 FP16
            "Memory_Bandwidth_GBps": 732,  # e.g., P100 (HBM2)
            "Storage_Bandwidth_GBps": 4.0  # PCIe 3.0 x4 NVMe
        },
        {
            "Year": 2018,
            "GPU_Architecture": "Volta",
            "Compute_TFLOPS": 125.0,       # e.g., V100 Tensor Core
            "Memory_Bandwidth_GBps": 900,  # e.g., V100
            "Storage_Bandwidth_GBps": 4.0  # PCIe 3.0 x4 NVMe
        },
        {
            "Year": 2020,
            "GPU_Architecture": "Ampere",
            "Compute_TFLOPS": 312.0,       # e.g., A100 Tensor Core (FP16/BF16)
            "Memory_Bandwidth_GBps": 1555, # e.g., A100 40GB
            "Storage_Bandwidth_GBps": 8.0  # PCIe 4.0 x4 NVMe
        },
        {
            "Year": 2022,
            "GPU_Architecture": "Hopper",
            "Compute_TFLOPS": 989.0,       # e.g., H100 Tensor Core (FP16/BF16, no sparsity)
            "Memory_Bandwidth_GBps": 3350, # e.g., H100 SXM
            "Storage_Bandwidth_GBps": 14.0 # PCIe 5.0 x4 NVMe
        },
        {
            "Year": 2024,
            "GPU_Architecture": "Blackwell",
            "Compute_TFLOPS": 1979.0,      # e.g., B200 Tensor Core (FP16/BF16, no sparsity)
            "Memory_Bandwidth_GBps": 8000, # e.g., B200 (HBM3e)
            "Storage_Bandwidth_GBps": 14.0 # PCIe 5.0 x4 NVMe
        }
    ]

    df = pd.DataFrame(data)

    # Determine paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chapter_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(chapter_dir, "data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(data_dir, "storage_compute_gap.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {csv_path}")

if __name__ == "__main__":
    generate_storage_compute_gap_data()
