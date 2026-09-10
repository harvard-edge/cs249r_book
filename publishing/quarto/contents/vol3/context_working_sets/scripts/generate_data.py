import pandas as pd
import os

def generate_context_length_scaling_data():
    """
    Generates the context_length_scaling.csv dataset.
    
    Data sources:
    The numbers represent the maximum context window size (in tokens) at the time of 
    release for various prominent foundation models. Dates correspond to the initial 
    announcement or publication date of the respective model.
    - Transformer, BERT (Google): Original research papers
    - GPT series (OpenAI): Official OpenAI blog posts and technical reports
    - Jurassic-1 (AI21): AI21 Labs technical paper
    - Gopher (DeepMind): DeepMind Gopher paper
    - Llama series (Meta): Meta AI blog posts and papers
    - Claude series (Anthropic): Anthropic release notes
    - Mixtral 8x7B (Mistral): Mistral AI announcements
    - Gemini series (Google): Google DeepMind technical reports
    """
    
    data = [
        {"Model": "Transformer", "Date": "2017-06-01", "ContextLength": 512, "Organization": "Google"},
        {"Model": "GPT-1", "Date": "2018-06-01", "ContextLength": 512, "Organization": "OpenAI"},
        {"Model": "BERT", "Date": "2018-10-11", "ContextLength": 512, "Organization": "Google"},
        {"Model": "GPT-2", "Date": "2019-02-14", "ContextLength": 1024, "Organization": "OpenAI"},
        {"Model": "GPT-3", "Date": "2020-05-28", "ContextLength": 2048, "Organization": "OpenAI"},
        {"Model": "Jurassic-1", "Date": "2021-09-01", "ContextLength": 2048, "Organization": "AI21"},
        {"Model": "Gopher", "Date": "2021-12-08", "ContextLength": 2048, "Organization": "DeepMind"},
        {"Model": "GPT-3.5", "Date": "2022-11-28", "ContextLength": 4096, "Organization": "OpenAI"},
        {"Model": "LLaMA", "Date": "2023-02-24", "ContextLength": 2048, "Organization": "Meta"},
        {"Model": "GPT-4", "Date": "2023-03-14", "ContextLength": 8192, "Organization": "OpenAI"},
        {"Model": "GPT-4-32k", "Date": "2023-03-14", "ContextLength": 32768, "Organization": "OpenAI"},
        {"Model": "Claude 1 (100k)", "Date": "2023-05-11", "ContextLength": 100000, "Organization": "Anthropic"},
        {"Model": "Llama 2", "Date": "2023-07-18", "ContextLength": 4096, "Organization": "Meta"},
        {"Model": "Claude 2.1", "Date": "2023-11-21", "ContextLength": 200000, "Organization": "Anthropic"},
        {"Model": "Mixtral 8x7B", "Date": "2023-12-11", "ContextLength": 32768, "Organization": "Mistral"},
        {"Model": "Gemini 1.5 Pro", "Date": "2024-02-15", "ContextLength": 1000000, "Organization": "Google"},
        {"Model": "Llama 3", "Date": "2024-04-18", "ContextLength": 8192, "Organization": "Meta"},
        {"Model": "GPT-4o", "Date": "2024-05-13", "ContextLength": 128000, "Organization": "OpenAI"},
        {"Model": "Gemini 1.5 Pro (2M)", "Date": "2024-05-14", "ContextLength": 2000000, "Organization": "Google"},
        {"Model": "Llama 3.1", "Date": "2024-07-23", "ContextLength": 131072, "Organization": "Meta"},
        {"Model": "Claude 3.5 Sonnet", "Date": "2024-06-21", "ContextLength": 200000, "Organization": "Anthropic"},
    ]
    
    df = pd.DataFrame(data)
    
    # Ensure the data directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Save to CSV
    df.to_csv("../data/context_length_scaling.csv", index=False)
    print("Successfully generated ../data/context_length_scaling.csv")

if __name__ == "__main__":
    generate_context_length_scaling_data()
