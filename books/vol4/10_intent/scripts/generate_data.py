import pandas as pd
import os

def generate_frequency_gap_data():
    """
    Generates historical trend data for operating frequencies of Low-Level Control vs High-Level Reasoning.
    
    Sources & Methodology:
    - Low-Level Control: Proxy data based on standard industry milestones. Early microcontrollers 
      supported ~500Hz, while DSPs and EtherCAT standardized the 1kHz (1000Hz) real-time control 
      loop widely used in robotics today.
    - Classical Computer Vision (1990-2006): Proxy data for historical algorithms (Blob Tracking, 
      Viola-Jones, HOG+SVM) running on contemporary CPUs.
    - Deep Learning (2012-2018): Based on seminal paper benchmarks (e.g., AlexNet 60M params on GTX 580 at ~30Hz, 
      YOLOv1 26M params at 45Hz for real-time detection).
    - Foundation Models (2020-2024): Based on robotics foundation model literature (e.g., RT-1 at 3Hz, 
      RT-2 55B at ~1Hz, OpenVLA 7B quantized at ~5Hz).
    """

    data = [
        # Low-Level Control (Control Tier)
        {"Year": 1990, "System_Tier": "Low-Level Control", "Technology": "PID Joint Control", "Frequency_Hz": 500, "Model_Size_Params": 0, "Category": "Control"},
        {"Year": 1995, "System_Tier": "Low-Level Control", "Technology": "DSP-based Control", "Frequency_Hz": 1000, "Model_Size_Params": 0, "Category": "Control"},
        {"Year": 2000, "System_Tier": "Low-Level Control", "Technology": "Industrial Impedance Control", "Frequency_Hz": 1000, "Model_Size_Params": 0, "Category": "Control"},
        {"Year": 2010, "System_Tier": "Low-Level Control", "Technology": "Real-time EtherCAT / Torque", "Frequency_Hz": 1000, "Model_Size_Params": 0, "Category": "Control"},
        {"Year": 2015, "System_Tier": "Low-Level Control", "Technology": "Whole-Body MPC", "Frequency_Hz": 1000, "Model_Size_Params": 0, "Category": "Control"},
        {"Year": 2020, "System_Tier": "Low-Level Control", "Technology": "High-Frequency Reflexes", "Frequency_Hz": 1000, "Model_Size_Params": 0, "Category": "Control"},
        {"Year": 2024, "System_Tier": "Low-Level Control", "Technology": "Fast-Loop MCU", "Frequency_Hz": 1000, "Model_Size_Params": 0, "Category": "Control"},

        # High-Level Reasoning (Classical)
        {"Year": 1990, "System_Tier": "High-Level Reasoning", "Technology": "Blob Tracking", "Frequency_Hz": 10, "Model_Size_Params": 0, "Category": "Classical"},
        {"Year": 1995, "System_Tier": "High-Level Reasoning", "Technology": "Color Histograms", "Frequency_Hz": 15, "Model_Size_Params": 0, "Category": "Classical"},
        {"Year": 2001, "System_Tier": "High-Level Reasoning", "Technology": "Viola-Jones", "Frequency_Hz": 15, "Model_Size_Params": 0, "Category": "Classical"},
        {"Year": 2006, "System_Tier": "High-Level Reasoning", "Technology": "HOG + SVM", "Frequency_Hz": 20, "Model_Size_Params": 0, "Category": "Classical"},

        # High-Level Reasoning (Deep Learning)
        {"Year": 2012, "System_Tier": "High-Level Reasoning", "Technology": "AlexNet (GPU)", "Frequency_Hz": 30, "Model_Size_Params": 60000000, "Category": "Deep Learning"},
        {"Year": 2015, "System_Tier": "High-Level Reasoning", "Technology": "YOLOv1", "Frequency_Hz": 45, "Model_Size_Params": 26000000, "Category": "Deep Learning"},
        {"Year": 2018, "System_Tier": "High-Level Reasoning", "Technology": "Mask R-CNN", "Frequency_Hz": 10, "Model_Size_Params": 44000000, "Category": "Deep Learning"},

        # High-Level Reasoning (Foundation Models)
        {"Year": 2020, "System_Tier": "High-Level Reasoning", "Technology": "DETR (Transformers)", "Frequency_Hz": 5, "Model_Size_Params": 41000000, "Category": "Foundation Models"},
        {"Year": 2022, "System_Tier": "High-Level Reasoning", "Technology": "RT-1", "Frequency_Hz": 3, "Model_Size_Params": 35000000, "Category": "Foundation Models"},
        {"Year": 2023, "System_Tier": "High-Level Reasoning", "Technology": "RT-2 (55B)", "Frequency_Hz": 1, "Model_Size_Params": 55000000000, "Category": "Foundation Models"},
        {"Year": 2024, "System_Tier": "High-Level Reasoning", "Technology": "OpenVLA (7B quantized)", "Frequency_Hz": 5, "Model_Size_Params": 7000000000, "Category": "Foundation Models"},
    ]

    df = pd.DataFrame(data)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct path to the data directory (one level up)
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'frequency_gap_trends.csv')
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_frequency_gap_data()
