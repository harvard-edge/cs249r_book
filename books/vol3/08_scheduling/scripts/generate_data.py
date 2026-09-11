import pandas as pd
import os

def generate_breakeven_data():
    """
    Generates breakeven thresholds for KV-cache swapping over PCIe.
    
    Data sources & logic:
    - Context Lengths: Standard powers-of-two (8192 to 131072).
    - Models:
        - 8B model uses ~131 KB per token for KV cache (based on typical Llama 3 8B dims).
        - 70B model uses ~327 KB per token for KV cache (based on typical Llama 3 70B dims).
    - PCIe Bandwidths (Unidirectional transfer, though DMA may be full duplex, we use the raw GB/s for the calculation):
        - Gen3 x16: 16 GB/s
        - Gen4 x16: 32 GB/s
        - Gen5 x16: 64 GB/s
        
    Formula:
    KV Size (GB) = Context Length * Token Size (KB) / 1,000,000
    Breakeven Time (ms) = (KV Size (GB) * 2 / Bandwidth (GB/s)) * 1000
    (Multiplier of 2 accounts for a round-trip swap: GPU -> RAM, then RAM -> GPU)
    """
    
    contexts = [8192, 16384, 32768, 65536, 131072]
    
    models = [
        {"name": "8B (131 KB/token)", "size_kb_per_token": 131},
        {"name": "70B (327 KB/token)", "size_kb_per_token": 327}
    ]
    
    pcies = [
        {"name": "PCIe Gen3 x16 (16 GB/s)", "bw_gb_s": 16},
        {"name": "PCIe Gen4 x16 (32 GB/s)", "bw_gb_s": 32},
        {"name": "PCIe Gen5 x16 (64 GB/s)", "bw_gb_s": 64}
    ]
    
    rows = []
    
    for ctx in contexts:
        for model in models:
            for pcie in pcies:
                # Calculate KV size in GB. Assuming 1,000,000 KB per GB for these numbers.
                kv_size_gb = (ctx * model["size_kb_per_token"]) / 1000000.0
                
                # Round trip DMA transfer latency
                breakeven_ms = (kv_size_gb * 2 / pcie["bw_gb_s"]) * 1000.0
                
                rows.append({
                    "Context Length": ctx,
                    "Model": model["name"],
                    "PCIe Gen": pcie["name"],
                    "KV Size (GB)": kv_size_gb,
                    "Breakeven Time (ms)": breakeven_ms
                })
                
    df = pd.DataFrame(rows)
    
    # Save to data directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'breakeven_thresholds.csv')
    df.to_csv(output_file, index=False)
    print(f"Data successfully generated and saved to {output_file}")

if __name__ == "__main__":
    generate_breakeven_data()
