"""
Example 05: Hugging Face Integration
------------------------------------
This script demonstrates how to dynamically import a model architecture
directly from the Hugging Face Hub, without needing to download the weights
or install heavy dependencies like `transformers` or `torch`.
"""
import mlsysim
from mlsysim.models.importer import import_hf_model

def main():
    print("Importing Mistral-7B directly from Hugging Face Hub...\n")

    # 1. Fetch the model configuration from HF
    # This reads the config.json and calculates the exact parameter count
    # based on the hidden dimensions, layers, and vocabulary size.
    try:
        mistral = import_hf_model("mistralai/Mistral-7B-v0.1")
        
        print(f"Model Name: {mistral.name}")
        print(f"Architecture: {mistral.architecture}")
        print(f"Parameters: {mistral.parameters:~P}")
        print(f"Layers: {mistral.layers}")
        print(f"Hidden Dim: {mistral.hidden_dim}")
        print(f"Attention Heads: {mistral.heads}")
        print(f"KV Heads: {mistral.kv_heads} (Grouped Query Attention)")
        
        print("\n--- Evaluating Mistral-7B on a single H100 ---")
        
        # 2. Evaluate the imported model just like a built-in model
        prof = mlsysim.Engine.solve(
            model=mistral,
            hardware=mlsysim.Hardware.Cloud.H100,
            batch_size=1,
            precision="fp16"
        )
        
        print(f"Latency (Batch 1): {prof.latency:.2f}")
        print(f"Memory Footprint: {prof.memory_footprint.to('GB'):.2f}")
        print(f"Bottleneck: {prof.bottleneck}")
        
    except Exception as e:
        print(f"Error fetching model: {e}")
        print("Note: If you are trying to fetch a gated model (like Llama-3),")
        print("you must set the HF_TOKEN environment variable.")

if __name__ == "__main__":
    main()
