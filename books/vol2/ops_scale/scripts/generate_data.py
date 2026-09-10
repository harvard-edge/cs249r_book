import pandas as pd
import os

def main():
    # Historical training compute data for state-of-the-art ML models.
    # Data is commonly sourced from papers on compute trends, such as:
    # "Compute Trends Across Three Eras of Machine Learning" (Sevilla et al., 2022)
    # and updated with recent models based on their technical reports (e.g., Llama 2, Llama 3, GPT-4).
    #
    # Note on Year formatting: Decimal years are used (e.g., 2012.8 corresponds to roughly late October 2012).
    # FLOPs represent the estimated total floating point operations used during training.
    
    data = [
        {"Year": 2012.8, "Model": "AlexNet", "FLOPs": "4.7e14"},       # Krizhevsky et al., 2012
        {"Year": 2014.7, "Model": "VGG-16", "FLOPs": "1.2e16"},        # Simonyan & Zisserman, 2014
        {"Year": 2017.5, "Model": "Transformer", "FLOPs": "8.9e16"},   # Vaswani et al., 2017 (Attention Is All You Need)
        {"Year": 2018.5, "Model": "GPT-1", "FLOPs": "1.8e18"},         # Radford et al., 2018
        {"Year": 2018.9, "Model": "BERT-Large", "FLOPs": "2.0e19"},    # Devlin et al., 2018
        {"Year": 2019.8, "Model": "GPT-2", "FLOPs": "6.0e19"},         # Radford et al., 2019
        {"Year": 2020.4, "Model": "GPT-3", "FLOPs": "3.14e23"},        # Brown et al., 2020
        {"Year": 2022.3, "Model": "PaLM", "FLOPs": "2.5e24"},          # Chowdhery et al., 2022
        {"Year": 2023.2, "Model": "GPT-4", "FLOPs": "2.1e25"},         # OpenAI, 2023 (estimated based on leaked/speculated specs)
        {"Year": 2023.6, "Model": "Llama-2-70B", "FLOPs": "3.0e24"},   # Touvron et al., 2023 (Llama 2 tech report)
        {"Year": 2024.3, "Model": "Llama-3-400B", "FLOPs": "3.8e25"},  # Meta, 2024 (Llama 3 tech report estimates)
    ]
    
    df = pd.DataFrame(data)
    
    # Ensure the target directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(target_dir, exist_ok=True)
    
    # Save the dataframe to CSV
    output_path = os.path.join(target_dir, 'training_compute_flops.csv')
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    main()
