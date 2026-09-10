import pandas as pd
import os

def generate_training_compute_data():
    """
    Generates the training compute data for landmark ML models.
    Data is primarily sourced from published papers and compute-trend studies 
    such as Amodei et al. (2018) and Sevilla et al. (2022) (Epoch AI).
    """
    # Raw data sources and proxy data based on industry milestones
    data = [
        # Vision Models (ImageNet era)
        {"Model": "AlexNet", "Year": 2012, "FLOPs": 4.7e17, "Domain": "Vision"},
        {"Model": "VGG-19", "Year": 2014, "FLOPs": 1.9e18, "Domain": "Vision"},
        {"Model": "ResNet-50", "Year": 2015, "FLOPs": 3.8e18, "Domain": "Vision"},
        
        # Language Models (Transformer era and beyond)
        {"Model": "Transformer", "Year": 2017, "FLOPs": 2.8e19, "Domain": "Language"},
        {"Model": "BERT-Large", "Year": 2018, "FLOPs": 1.9e20, "Domain": "Language"},
        {"Model": "Megatron-LM", "Year": 2019, "FLOPs": 1.1e21, "Domain": "Language"},
        {"Model": "GPT-3", "Year": 2020, "FLOPs": 3.1e23, "Domain": "Language"},
        {"Model": "PaLM", "Year": 2022, "FLOPs": 2.5e24, "Domain": "Language"},
        
        # GPT-4 estimate based on illustrative proxy data (no official compute disclosed)
        {"Model": "GPT-4 (Est.)", "Year": 2023, "FLOPs": 2.1e25, "Domain": "Language"},
        
        # Llama-3 (405B) compute based on Meta's technical report
        {"Model": "Llama-3 (405B)", "Year": 2024, "FLOPs": 3.8e25, "Domain": "Language"},
    ]

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'training_compute.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_training_compute_data()
