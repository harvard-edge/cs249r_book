#!/usr/bin/env python3
import pandas as pd
import os

def main():
    # Define the raw data sources and numbers for the memory wall trend.
    # Data is sourced from official NVIDIA datasheets and technical briefs for each architecture.
    data = [
        {
            "year": 2017,
            "gpu": "V100",
            "peak_tflops": 125,   # FP16 Tensor Core TFLOPS (NVIDIA V100 datasheet)
            "bandwidth_tbps": 0.9 # HBM2 Bandwidth in TB/s (NVIDIA V100 datasheet)
        },
        {
            "year": 2020,
            "gpu": "A100",
            "peak_tflops": 312,   # FP16 Tensor Core TFLOPS, Dense (NVIDIA A100 datasheet)
            "bandwidth_tbps": 1.55# HBM2 Bandwidth in TB/s (NVIDIA A100 datasheet)
        },
        {
            "year": 2022,
            "gpu": "H100",
            "peak_tflops": 989,   # FP16 Tensor Core TFLOPS, Dense (NVIDIA H100 datasheet)
            "bandwidth_tbps": 3.35# HBM3 Bandwidth in TB/s (NVIDIA H100 SXM datasheet)
        },
        {
            "year": 2024,
            "gpu": "B200",
            "peak_tflops": 4500,  # FP16 Tensor Core TFLOPS, Dense (NVIDIA Blackwell architecture brief)
            "bandwidth_tbps": 8.0 # HBM3e Bandwidth in TB/s (NVIDIA Blackwell architecture brief)
        }
    ]

    df = pd.DataFrame(data)

    # Determine the directory this script is in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the data directory (one level up, then into data/)
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'memory_wall_trend.csv')
    
    # Save to CSV, matching the original format precisely
    df.to_csv(output_path, index=False)
    print(f"Successfully generated data at {output_path}")

if __name__ == "__main__":
    main()
