import pandas as pd
import os

def generate_context_length_data():
    """
    Generates historical context length scaling data for foundation models.
    Data is sourced from historical model release announcements and technical papers
    (e.g., OpenAI, Anthropic, Google DeepMind milestones).
    """
    data = [
        {"Model": "GPT-1", "Release_Date": "2018-06-11", "Context_Length": 512},
        {"Model": "GPT-2", "Release_Date": "2019-02-14", "Context_Length": 1024},
        {"Model": "GPT-3", "Release_Date": "2020-05-28", "Context_Length": 2048},
        {"Model": "Gopher", "Release_Date": "2021-12-08", "Context_Length": 2048},
        {"Model": "Chinchilla", "Release_Date": "2022-03-29", "Context_Length": 2048},
        {"Model": "PaLM", "Release_Date": "2022-04-04", "Context_Length": 2048},
        {"Model": "GPT-3.5 (Turbo)", "Release_Date": "2022-11-30", "Context_Length": 4096},
        {"Model": "GPT-4", "Release_Date": "2023-03-14", "Context_Length": 32768},
        {"Model": "Claude 2", "Release_Date": "2023-07-11", "Context_Length": 100000},
        {"Model": "GPT-4 Turbo", "Release_Date": "2023-11-06", "Context_Length": 128000},
        {"Model": "Claude 2.1", "Release_Date": "2023-11-21", "Context_Length": 200000},
        {"Model": "Gemini 1.5 Pro", "Release_Date": "2024-02-15", "Context_Length": 1048576},
        {"Model": "Llama 3", "Release_Date": "2024-04-18", "Context_Length": 8192},
        {"Model": "Llama 3.1", "Release_Date": "2024-07-23", "Context_Length": 131072},
        {"Model": "Gemini 1.5 Pro (2M)", "Release_Date": "2024-05-14", "Context_Length": 2097152},
    ]

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Save to CSV
    df.to_csv("../data/context_length_scaling.csv", index=False)
    print("Successfully generated ../data/context_length_scaling.csv")

if __name__ == "__main__":
    generate_context_length_data()
