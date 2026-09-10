import pandas as pd
import os

def generate_brain_frequency_data():
    """
    Generates the CSV file for the Cadence Gap figure (brain_frequency_vs_parameters.csv).
    Data sources:
    - Classical PID/MPC: General industry estimates for real-time control frequencies and minimal parameter footprint.
    - MobileNet-V3: Mobile vision models typically run at ~60 Hz, parameters around 5.4M.
    - RT-1: 35M parameters, runs around 10 Hz.
    - Octo: 93M parameters, runs around 10 Hz.
    - ViT-L: 300M parameters, standard vision transformer inference, ~30 Hz on modern GPUs.
    - OpenVLA: 7B parameters, reported ~4-5 Hz depending on hardware.
    - RT-2 (5B/55B): 5B runs at ~5 Hz, 55B runs at ~1-2 Hz.
    - PaLM-E: 562B parameters, extremely large, typical inference latency multi-second (~0.2 Hz).
    - GPT-4V: Proxy parameter count ~1.7T, typical cloud API round trip ~2s (~0.5 Hz).
    """
    
    data = [
        {"Model": "Classical PID", "Parameters": 1e1, "Frequency_Hz": 1000, "Type": "Classical", "Category": "Control"},
        {"Model": "Model Predictive Control", "Parameters": 1e3, "Frequency_Hz": 200, "Type": "Classical", "Category": "Control"},
        {"Model": "MobileNet-V3", "Parameters": 5.4e6, "Frequency_Hz": 60, "Type": "Learned", "Category": "Vision"},
        {"Model": "RT-1", "Parameters": 35e6, "Frequency_Hz": 10, "Type": "Learned", "Category": "VLA"},
        {"Model": "Octo", "Parameters": 93e6, "Frequency_Hz": 10, "Type": "Learned", "Category": "VLA"},
        {"Model": "ViT-L", "Parameters": 300e6, "Frequency_Hz": 30, "Type": "Learned", "Category": "Vision"},
        {"Model": "OpenVLA", "Parameters": 7e9, "Frequency_Hz": 4, "Type": "Learned", "Category": "VLA"},
        {"Model": "RT-2 (5B)", "Parameters": 5e9, "Frequency_Hz": 5, "Type": "Learned", "Category": "VLA"},
        {"Model": "RT-2 (55B)", "Parameters": 55e9, "Frequency_Hz": 1, "Type": "Learned", "Category": "VLA"},
        {"Model": "PaLM-E", "Parameters": 562e9, "Frequency_Hz": 0.2, "Type": "Learned", "Category": "VLM"},
        {"Model": "GPT-4V (Cloud)", "Parameters": 1.7e12, "Frequency_Hz": 0.5, "Type": "Learned", "Category": "VLM"},
    ]
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(data_dir, "brain_frequency_vs_parameters.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {csv_path}")

if __name__ == "__main__":
    generate_brain_frequency_data()
