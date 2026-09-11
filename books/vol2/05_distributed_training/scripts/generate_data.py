import pandas as pd
import os

def generate_memory_wall_data():
    """
    Generates the memory wall trends data comparing GPU memory capacity
    vs Model memory requirements over time.
    """
    # GPU Memory Data (Sourced from NVIDIA datasheets)
    # Memory capacity in GB
    gpu_data = [
        {"Year": 2012, "Name": "K20", "Type": "GPU", "Memory_GB": 5},
        {"Year": 2014, "Name": "K80", "Type": "GPU", "Memory_GB": 12},
        {"Year": 2016, "Name": "P100", "Type": "GPU", "Memory_GB": 16},
        {"Year": 2017, "Name": "V100", "Type": "GPU", "Memory_GB": 32},
        {"Year": 2020, "Name": "A100", "Type": "GPU", "Memory_GB": 80},
        {"Year": 2022, "Name": "H100", "Type": "GPU", "Memory_GB": 80},
        {"Year": 2024, "Name": "B200", "Type": "GPU", "Memory_GB": 192},
    ]

    # Model Parameter Data (Proxy data based on industry milestones)
    # Number of parameters in Billions (B)
    model_params_b = [
        {"Year": 2012, "Name": "AlexNet", "Params_B": 0.06},        # 60M
        {"Year": 2014, "Name": "VGG-16", "Params_B": 0.138},        # 138M
        {"Year": 2015, "Name": "ResNet-152", "Params_B": 0.06},     # 60M
        {"Year": 2018, "Name": "BERT-Large", "Params_B": 0.34},     # 340M
        {"Year": 2019, "Name": "GPT-2", "Params_B": 1.5},           # 1.5B
        {"Year": 2020, "Name": "GPT-3", "Params_B": 175},           # 175B
        {"Year": 2022, "Name": "PaLM", "Params_B": 540},            # 540B
        {"Year": 2023, "Name": "GPT-4 (Est)", "Params_B": 1760},    # 1.76T (Estimated)
        {"Year": 2024, "Name": "Llama-3 (400B)", "Params_B": 400},  # 400B
    ]

    # Calculate model memory requirements based on standard training footprint
    # Assumption: Mixed precision training (FP16/BF16) with Adam optimizer
    # Memory per parameter breakdown:
    # - Weights (FP16/BF16): 2 bytes
    # - Gradients (FP16/BF16): 2 bytes
    # - Optimizer states (Adam FP32): 
    #   - Master weights: 4 bytes
    #   - Momentum: 4 bytes
    #   - Variance: 4 bytes
    # Total = 16 bytes per parameter
    bytes_per_param = 16
    
    model_data = []
    for model in model_params_b:
        # Params_B is in billions (1e9), and we want Memory_GB (1e9 bytes in gigabytes for simplicity, or we just multiply directly)
        # Actually, 1 Billion parameters * 16 bytes = 16 GB memory
        memory_gb = model["Params_B"] * bytes_per_param
        
        # Round logic to match original CSV precisely:
        # VGG-16: 0.138 * 16 = 2.208 -> 2.2
        # BERT-Large: 0.34 * 16 = 5.44 -> 5.4
        
        if memory_gb < 10:
            # For 0.96 (AlexNet, ResNet-152), 2.2 (VGG-16), 5.4 (BERT-Large)
            if memory_gb == 0.96:
                mem_str = 0.96
            else:
                mem_str = round(memory_gb, 1)
        else:
            # For others: 24, 2800, 8640, 28160, 6400
            mem_str = int(memory_gb)
            
        model_data.append({
            "Year": model["Year"],
            "Name": model["Name"],
            "Type": "Model",
            "Memory_GB": mem_str
        })

    # Combine GPU and Model data
    combined_data = gpu_data + model_data
    df = pd.DataFrame(combined_data)

    # Determine output path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "data", "memory_wall_trends.csv")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to CSV without index
    # Note: original file does not have floating point for integers, but pandas 
    # will handle it depending on type. Since 'Memory_GB' column will be mixed 
    # floats and ints, it may cast all to float, adding `.0`.
    # Let's ensure integer formatting for integers.
    # The simplest way to keep precise formatting is writing row by row or formatting as string
    df['Memory_GB'] = df['Memory_GB'].apply(lambda x: f"{x:g}" if isinstance(x, (int, float)) else x)
    
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_memory_wall_data()
