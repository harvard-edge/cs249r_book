import pandas as pd
import os

def generate_context_length_scaling():
    # Data sourced from historical model announcements and technical reports
    # Contains frontier models from 2020 to 2024 and their maximum context window length
    data = [
        {"model": "GPT-3", "release_date": "2020-05-28", "context_length": 2048, "org": "OpenAI"},
        {"model": "Jurassic-1", "release_date": "2021-08-11", "context_length": 2048, "org": "AI21"},
        {"model": "Gopher", "release_date": "2021-12-08", "context_length": 2048, "org": "DeepMind"},
        {"model": "Chinchilla", "release_date": "2022-03-29", "context_length": 2048, "org": "DeepMind"},
        {"model": "PaLM", "release_date": "2022-04-04", "context_length": 2048, "org": "Google"},
        {"model": "GPT-3.5", "release_date": "2022-11-30", "context_length": 4096, "org": "OpenAI"},
        {"model": "LLaMA", "release_date": "2023-02-24", "context_length": 2048, "org": "Meta"},
        {"model": "GPT-4", "release_date": "2023-03-14", "context_length": 32768, "org": "OpenAI"},
        {"model": "Claude 1", "release_date": "2023-05-11", "context_length": 100000, "org": "Anthropic"},
        {"model": "Llama 2", "release_date": "2023-07-18", "context_length": 4096, "org": "Meta"},
        {"model": "Claude 2.1", "release_date": "2023-11-21", "context_length": 200000, "org": "Anthropic"},
        {"model": "GPT-4 Turbo", "release_date": "2023-11-06", "context_length": 128000, "org": "OpenAI"},
        {"model": "Mistral 7B", "release_date": "2023-09-27", "context_length": 8192, "org": "Mistral"},
        {"model": "Mixtral 8x7B", "release_date": "2023-12-11", "context_length": 32768, "org": "Mistral"},
        {"model": "Gemini 1.5 Pro", "release_date": "2024-02-15", "context_length": 1048576, "org": "Google"},
        {"model": "Claude 3 Opus", "release_date": "2024-03-04", "context_length": 200000, "org": "Anthropic"},
        {"model": "Llama 3", "release_date": "2024-04-18", "context_length": 8192, "org": "Meta"},
        {"model": "Gemini 1.5 Pro 2M", "release_date": "2024-05-14", "context_length": 2097152, "org": "Google"},
        {"model": "Llama 3.1", "release_date": "2024-07-23", "context_length": 128000, "org": "Meta"},
        {"model": "Qwen 2", "release_date": "2024-06-06", "context_length": 128000, "org": "Alibaba"},
        {"model": "DeepSeek-V2", "release_date": "2024-05-06", "context_length": 128000, "org": "DeepSeek"},
    ]

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Save to CSV
    output_path = "../data/context_length_scaling.csv"
    df.to_csv(output_path, index=False)
    print(f"Data successfully generated and saved to {output_path}")

if __name__ == "__main__":
    generate_context_length_scaling()
