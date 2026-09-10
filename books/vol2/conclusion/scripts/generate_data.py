import pandas as pd
import os

# Data sourced from NVIDIA datasheets and publicly available specifications
# The data tracks the historical progression of peak compute and memory bandwidth
# for NVIDIA data center GPUs from 2010 to 2024, highlighting the "memory wall" divergence.
# Data points are based on typical FP32/Tensor core (where applicable) TFLOPS 
# and memory bandwidth (GB/s) from official NVIDIA product specifications.

def generate_gpu_memory_wall_data():
    """
    Recreates the gpu_memory_wall.csv file with historical NVIDIA GPU specs.
    """
    
    # Define raw data
    data = [
        {"Year": 2010, "Architecture": "Fermi",     "GPU": "M2090", "Compute_TFLOPS": 1.3,    "Memory_Bandwidth_GBps": 177},
        {"Year": 2012, "Architecture": "Kepler",    "GPU": "K20X",  "Compute_TFLOPS": 3.9,    "Memory_Bandwidth_GBps": 250},
        {"Year": 2014, "Architecture": "Maxwell",   "GPU": "M40",   "Compute_TFLOPS": 7.0,    "Memory_Bandwidth_GBps": 288},
        {"Year": 2016, "Architecture": "Pascal",    "GPU": "P100",  "Compute_TFLOPS": 21.2,   "Memory_Bandwidth_GBps": 732},
        {"Year": 2017, "Architecture": "Volta",     "GPU": "V100",  "Compute_TFLOPS": 125.0,  "Memory_Bandwidth_GBps": 900},
        {"Year": 2020, "Architecture": "Ampere",    "GPU": "A100",  "Compute_TFLOPS": 312.0,  "Memory_Bandwidth_GBps": 2039},
        {"Year": 2022, "Architecture": "Hopper",    "GPU": "H100",  "Compute_TFLOPS": 989.0,  "Memory_Bandwidth_GBps": 3350},
        {"Year": 2024, "Architecture": "Blackwell", "GPU": "B200",  "Compute_TFLOPS": 4500.0, "Memory_Bandwidth_GBps": 8000},
    ]

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Determine output path relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data")
    
    # Ensure data directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "gpu_memory_wall.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_gpu_memory_wall_data()
