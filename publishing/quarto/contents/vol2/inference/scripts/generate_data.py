import pandas as pd
import os

def generate_gpu_memory_wall_data(output_dir):
    """
    Generates the gpu_memory_wall.csv dataset containing historical data
    on NVIDIA Data Center GPUs, specifically tracking Peak TFLOPS and 
    Memory Bandwidth over the years.
    
    Data sources: NVIDIA official datasheets and product specifications 
    for K80, P100, V100, A100, and H100 architectures.
    - K80 (2014): Peak single-precision / FP32, combined for dual-GPU.
    - P100 (2016): Peak FP16.
    - V100 (2017): Peak FP16 Tensor Core.
    - A100 (2020): Peak FP16 Tensor Core.
    - H100 (2022): Peak FP16/BF16 Tensor Core without sparsity.
    """
    data = [
        {"Year": 2014, "GPU": "K80", "Peak_TFLOPS": 8.7, "Memory_Bandwidth_GBps": 480},
        {"Year": 2016, "GPU": "P100", "Peak_TFLOPS": 21.2, "Memory_Bandwidth_GBps": 732},
        {"Year": 2017, "GPU": "V100", "Peak_TFLOPS": 125.0, "Memory_Bandwidth_GBps": 900},
        {"Year": 2020, "GPU": "A100", "Peak_TFLOPS": 312.0, "Memory_Bandwidth_GBps": 1555},
        {"Year": 2022, "GPU": "H100", "Peak_TFLOPS": 989.0, "Memory_Bandwidth_GBps": 3350},
    ]

    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "gpu_memory_wall.csv")
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Output directory is `../data` relative to the script
    output_dir = os.path.join(script_dir, "..", "data")
    
    generate_gpu_memory_wall_data(output_dir)
