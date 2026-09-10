"""
Generates the context length scaling historical data.

Data sources:
- GPT-3: OpenAI paper "Language Models are Few-Shot Learners" (May 2020)
- GPT-3.5: OpenAI ChatGPT release blog post (Nov 2022)
- GPT-4 (8k) & GPT-4 (32k): OpenAI GPT-4 announcement (March 2023)
- Claude 1 & Claude 2: Anthropic API / model announcements (May/July 2023)
- GPT-4 Turbo: OpenAI DevDay announcement (Nov 2023)
- Claude 2.1: Anthropic blog post (Nov 2023)
- Gemini 1.5 Pro: Google DeepMind announcement (Feb 2024)
- Claude 3 Opus: Anthropic Claude 3 family announcement (March 2024)
- Gemini 1.5 Pro v2: Google I/O announcement (May 2024)
- Llama 3.1: Meta AI announcement (July 2024)

These numbers represent the maximum officially supported context length (in tokens) at release.
"""

import os
import pandas as pd

def generate_context_length_data():
    data = [
        {"Model": "GPT-3", "ReleaseDate": "2020-05-28", "ContextLength": 2048},
        {"Model": "GPT-3.5", "ReleaseDate": "2022-11-30", "ContextLength": 4096},
        {"Model": "GPT-4 (8k)", "ReleaseDate": "2023-03-14", "ContextLength": 8192},
        {"Model": "GPT-4 (32k)", "ReleaseDate": "2023-03-14", "ContextLength": 32768},
        {"Model": "Claude 1", "ReleaseDate": "2023-05-11", "ContextLength": 100000},
        {"Model": "Claude 2", "ReleaseDate": "2023-07-11", "ContextLength": 100000},
        {"Model": "GPT-4 Turbo", "ReleaseDate": "2023-11-06", "ContextLength": 128000},
        {"Model": "Claude 2.1", "ReleaseDate": "2023-11-21", "ContextLength": 200000},
        {"Model": "Gemini 1.5 Pro", "ReleaseDate": "2024-02-15", "ContextLength": 1000000},
        {"Model": "Claude 3 Opus", "ReleaseDate": "2024-03-04", "ContextLength": 200000},
        {"Model": "Gemini 1.5 Pro v2", "ReleaseDate": "2024-05-14", "ContextLength": 2000000},
        {"Model": "Llama 3.1", "ReleaseDate": "2024-07-23", "ContextLength": 128000}
    ]

    df = pd.DataFrame(data)
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve path to the data directory (one level up, then data)
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, "context_length_scaling.csv")
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_context_length_data()
