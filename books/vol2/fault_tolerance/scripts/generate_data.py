import os
import pandas as pd

def generate_cluster_size_trend():
    """
    Generates historical cluster size data and calculates expected system MTBF.
    
    Data sources & Assumptions:
    - Base accelerator MTBF is assumed to be 50,000 hours (approx 5.7 years),
      which is a common proxy for enterprise GPU/TPU hardware reliability in datacenters.
    - System MTBF is calculated as (Base MTBF) / (Number of Accelerators),
      assuming independent failures and no fault tolerance mechanisms.
      
    Model cluster sizes are based on historical estimates, papers, and technical reports:
    - AlexNet (2012): 2x GTX 580 (from original paper)
    - VGG-16 (2014): 4x Titan Black
    - ResNet-50 (2015): 8x K20
    - BERT-Large (2018): 64x TPUv2 (Google BERT paper)
    - GPT-2 (2019): 256x V100
    - GPT-3 (2020): 1024x V100 (proxy for scale)
    - PaLM (2022): 6144x TPUv4 (Google PaLM paper)
    - LLaMA (2023): 2048x A100 (Meta LLaMA 1 paper)
    - GPT-4 (est) (2023): ~25,000x A100 (industry estimates)
    - Llama 3 (2024): 24,576x H100 (Meta Llama 3 blog)
    - Grok-2 (est) (2024): 100,000x H100 (xAI announcements)
    """
    
    data = [
        {"year": 2012, "model": "AlexNet", "accelerators": 2, "accelerator_type": "GTX 580"},
        {"year": 2014, "model": "VGG-16", "accelerators": 4, "accelerator_type": "Titan Black"},
        {"year": 2015, "model": "ResNet-50", "accelerators": 8, "accelerator_type": "K20"},
        {"year": 2018, "model": "BERT-Large", "accelerators": 64, "accelerator_type": "TPUv2"},
        {"year": 2019, "model": "GPT-2", "accelerators": 256, "accelerator_type": "V100"},
        {"year": 2020, "model": "GPT-3", "accelerators": 1024, "accelerator_type": "V100"},
        {"year": 2022, "model": "PaLM", "accelerators": 6144, "accelerator_type": "TPUv4"},
        {"year": 2023, "model": "LLaMA", "accelerators": 2048, "accelerator_type": "A100"},
        {"year": 2023, "model": "GPT-4 (est)", "accelerators": 25000, "accelerator_type": "A100"},
        {"year": 2024, "model": "Llama 3", "accelerators": 24576, "accelerator_type": "H100"},
        {"year": 2024, "model": "Grok-2 (est)", "accelerators": 100000, "accelerator_type": "H100"},
    ]
    
    df = pd.DataFrame(data)
    
    # Base MTBF per accelerator (hours)
    BASE_MTBF = 50000
    
    def format_mtbf_str(accel_count):
        val = BASE_MTBF / accel_count
        if val >= 40:
            return f"{int(round(val, 0))}"
        else:
            return f"{val:.1f}"
            
    df['system_mtbf_hours'] = df['accelerators'].apply(format_mtbf_str)
    
    # Ensure data directory exists relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'cluster_size_trend.csv')
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_cluster_size_trend()
