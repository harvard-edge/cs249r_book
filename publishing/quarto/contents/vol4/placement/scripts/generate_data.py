"""
generate_data.py

This script regenerates the gpu_memory_wall.csv file.
The data contains historical GPU compute and memory bandwidth specifications.
Data sourced from NVIDIA datasheets and technical whitepapers for data center GPUs.
- P100 (2016): FP16 compute (21.2 TFLOPS), Memory Bandwidth (732 GB/s)
- V100 (2017): FP16 Tensor Core (125 TFLOPS), Memory Bandwidth (900 GB/s)
- A100 (2020): FP16 Tensor Core (312 TFLOPS), Memory Bandwidth (1555 GB/s)
- H100 (2022): FP16 Tensor Core (989 TFLOPS dense), Memory Bandwidth (3350 GB/s)
- B200 (2024): FP16 Tensor Core (2250 TFLOPS dense), Memory Bandwidth (8000 GB/s)

This data is used to demonstrate the "Memory Wall" phenomenon in ML systems, 
showing how compute capabilities have historically outpaced memory bandwidth improvements.
"""

import pandas as pd
import os

def main():
    # Define the raw data sources and numbers
    # Numbers sourced from NVIDIA datasheets for P100, V100, A100, H100, and B200
    gpu_data = [
        {
            "Year": 2016,
            "Accelerator": "P100",
            "Compute_TFLOPS": 21.2,
            "Memory_Bandwidth_GBs": 732
        },
        {
            "Year": 2017,
            "Accelerator": "V100",
            "Compute_TFLOPS": 125,
            "Memory_Bandwidth_GBs": 900
        },
        {
            "Year": 2020,
            "Accelerator": "A100",
            "Compute_TFLOPS": 312,
            "Memory_Bandwidth_GBs": 1555
        },
        {
            "Year": 2022,
            "Accelerator": "H100",
            "Compute_TFLOPS": 989,
            "Memory_Bandwidth_GBs": 3350
        },
        {
            "Year": 2024,
            "Accelerator": "B200",
            "Compute_TFLOPS": 2250,
            "Memory_Bandwidth_GBs": 8000
        }
    ]

    # Create a Pandas DataFrame
    df = pd.DataFrame(gpu_data)

    # Convert to standard python types to avoid decimal points on whole numbers in CSV (e.g., 125.0 -> 125)
    df['Compute_TFLOPS'] = df['Compute_TFLOPS'].apply(lambda x: f"{x:g}")

    # Ensure output directory exists
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'gpu_memory_wall.csv')
    
    # Save the dataframe to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully generated and saved data to {output_path}")

if __name__ == "__main__":
    main()
