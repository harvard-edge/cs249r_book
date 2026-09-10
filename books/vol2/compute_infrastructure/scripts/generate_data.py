import pandas as pd
import os

def generate_gpu_scaling_trends():
    """
    Generates the GPU scaling trends dataset.
    
    Data sources:
    - V100 (2017): NVIDIA Tesla V100 GPU Architecture Whitepaper
      Compute: 125 TFLOPs (Tensor Core), Memory Bandwidth: 900 GB/s, Power: 300W
    - A100 (2020): NVIDIA A100 Tensor Core GPU Architecture Whitepaper
      Compute: 312 TFLOPs (FP16 Tensor Core), Memory Bandwidth: 2039 GB/s, Power: 400W
    - H100 (2022): NVIDIA H100 Tensor Core GPU Architecture Whitepaper
      Compute: 989 TFLOPs (FP16 Tensor Core, dense), Memory Bandwidth: 3350 GB/s, Power: 700W
    - B200 (2024): NVIDIA Blackwell Architecture Technical Brief
      Compute: 2250 TFLOPs (FP16 Tensor Core, dense), Memory Bandwidth: 8000 GB/s, Power: 1000W
    """
    data = [
        {"GPU": "V100", "Year": 2017, "Compute_TFLOPs": 125, "Memory_Bandwidth_GBps": 900, "Power_W": 300},
        {"GPU": "A100", "Year": 2020, "Compute_TFLOPs": 312, "Memory_Bandwidth_GBps": 2039, "Power_W": 400},
        {"GPU": "H100", "Year": 2022, "Compute_TFLOPs": 989, "Memory_Bandwidth_GBps": 3350, "Power_W": 700},
        {"GPU": "B200", "Year": 2024, "Compute_TFLOPs": 2250, "Memory_Bandwidth_GBps": 8000, "Power_W": 1000}
    ]
    
    df = pd.DataFrame(data)
    
    # Ensure the target directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'gpu_scaling_trends.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_gpu_scaling_trends()
