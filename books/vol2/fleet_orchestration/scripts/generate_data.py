import os
import pandas as pd

def main():
    """
    Generates the historical trend data for cluster scaling in ML training.
    
    Data sources and proxies:
    - AlexNet (2012): 2x GTX 580 GPUs (Krizhevsky et al.)
    - Seq2Seq (2014): 8x GPUs (Sutskever et al.)
    - ResNet (2015): 8x GPUs (He et al.)
    - Transformer (2017): 8x P100 GPUs (Vaswani et al.)
    - BERT (2018): 64x TPUv2 chips (Devlin et al.)
    - GPT-3 (2020): ~10,000 V100 GPUs (Brown et al. / OpenAI / Microsoft cluster)
    - PaLM (2022): 6144 TPUv4 chips (Chowdhery et al.)
    - Llama-2 (2023): ~24,000 A100 GPUs (Meta's Research SuperCluster)
    - Llama-3 (2024): 24,000 H100 GPUs (Meta's H100 cluster announcement)
    - Grok-3 (2024): 100,000 H100 GPUs (xAI's Colossus cluster)
    """
    
    data = [
        {"Year": 2012, "Model": "AlexNet", "Accelerators": 2},
        {"Year": 2014, "Model": "Seq2Seq", "Accelerators": 8},
        {"Year": 2015, "Model": "ResNet", "Accelerators": 8},
        {"Year": 2017, "Model": "Transformer", "Accelerators": 8},
        {"Year": 2018, "Model": "BERT", "Accelerators": 64},
        {"Year": 2020, "Model": "GPT-3", "Accelerators": 10000},
        {"Year": 2022, "Model": "PaLM", "Accelerators": 6144},
        {"Year": 2023, "Model": "Llama-2", "Accelerators": 24000},
        {"Year": 2024, "Model": "Llama-3", "Accelerators": 24000},
        {"Year": 2024, "Model": "Grok-3", "Accelerators": 100000}
    ]

    df = pd.DataFrame(data)

    # Resolve the path to the data directory (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Write the CSV file
    csv_path = os.path.join(data_dir, "cluster_scaling.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {csv_path}")

if __name__ == "__main__":
    main()
