"""
generate_data.py

This script reverse-engineers the AI training compute trends dataset.
Data points (Model, Date, Compute in FLOPs, and Era) are curated from 
historical publications, technical reports, and Epoch AI's Parameter/Compute databases.

Sources:
- Pre-2020 models (AlexNet, VGG-19, ResNet-152, Transformer, BERT, GPT-2, Megatron-LM, T5): 
  Sourced from their respective landmark papers and synthesized in the Green AI (Schwartz et al. 2020) and Epoch AI databases.
- GPT-3: OpenAI GPT-3 paper (Brown et al., 2020).
- Gopher, Chinchilla: DeepMind publications (Rae et al., 2021; Hoffmann et al., 2022).
- PaLM: Google Research (Chowdhery et al., 2022).
- LLaMA, Llama-3: Meta AI technical reports.
- GPT-4: OpenAI Technical Report (2023), estimated compute.
"""

import pandas as pd
import os

def generate_ai_compute_trend():
    # Raw data definition
    data = [
        {"Model": "AlexNet", "Year": "2012-09-01", "Compute_FLOPs": 4.7e17, "Era": "Pre-Deep Learning Era"},
        {"Model": "VGG-19", "Year": "2014-09-01", "Compute_FLOPs": 2.0e18, "Era": "Deep Learning Era"},
        {"Model": "ResNet-152", "Year": "2015-12-01", "Compute_FLOPs": 1.1e19, "Era": "Deep Learning Era"},
        {"Model": "Transformer", "Year": "2017-06-01", "Compute_FLOPs": 2.8e19, "Era": "Deep Learning Era"},
        {"Model": "BERT-Large", "Year": "2018-10-01", "Compute_FLOPs": 3.2e20, "Era": "Large Scale Era"},
        {"Model": "GPT-2", "Year": "2019-02-01", "Compute_FLOPs": 1.5e21, "Era": "Large Scale Era"},
        {"Model": "Megatron-LM", "Year": "2019-09-01", "Compute_FLOPs": 3.2e21, "Era": "Large Scale Era"},
        {"Model": "T5", "Year": "2019-10-01", "Compute_FLOPs": 3.3e22, "Era": "Large Scale Era"},
        {"Model": "GPT-3", "Year": "2020-05-01", "Compute_FLOPs": 3.14e23, "Era": "Large Scale Era"},
        {"Model": "Gopher", "Year": "2021-12-01", "Compute_FLOPs": 5.0e23, "Era": "Large Scale Era"},
        {"Model": "Chinchilla", "Year": "2022-03-01", "Compute_FLOPs": 4.3e23, "Era": "Large Scale Era"},
        {"Model": "PaLM", "Year": "2022-04-01", "Compute_FLOPs": 2.5e24, "Era": "Large Scale Era"},
        {"Model": "LLaMA", "Year": "2023-02-01", "Compute_FLOPs": 1.0e24, "Era": "Large Scale Era"},
        {"Model": "GPT-4", "Year": "2023-03-01", "Compute_FLOPs": 2.1e25, "Era": "Large Scale Era"},
        {"Model": "Llama-3 (70B)", "Year": "2024-04-01", "Compute_FLOPs": 3.8e25, "Era": "Large Scale Era"},
    ]

    df = pd.DataFrame(data)

    # Resolve output path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'data', 'ai_compute_trend.csv')
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_ai_compute_trend()
