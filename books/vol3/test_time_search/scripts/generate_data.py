import os
import pandas as pd
import numpy as np

# Data sources and logic explanation:
# The data is based on hypothetical but realistic numbers for a 70B parameter model
# using 16-bit FP16 precision across 80 layers.
# - Shared prompt prefix length: 8,000 tokens
# - Divergent generation suffix length per branch: 1,000 tokens
# - KV-cache cost: 2.5 GB for the 8,000-token prefix, which equates to 0.3125 MB per token.
#
# Naive independent allocation:
# Each branch stores the full 8,000 + 1,000 = 9,000 tokens.
# 9,000 tokens * 0.3125 MB/token = 2,812.5 MB = 2.8125 GB per branch.
#
# Radix prefix sharing (Copy-on-Write):
# The 8,000 token prefix is stored exactly once, costing 2.5 GB.
# Each branch only allocates its 1,000 divergent tokens, costing 1,000 * 0.3125 MB = 0.3125 GB per branch.

def generate_kv_cache_scaling_data():
    branches = np.arange(1, 129)
    
    # Costs
    cost_per_token_mb = 0.3125
    prefix_tokens = 8000
    suffix_tokens = 1000
    
    # 2.5 GB base cost for the prefix
    prefix_cost_gb = (prefix_tokens * cost_per_token_mb) / 1024  # Wait, 2500 MB = 2.5 GB? The text uses 1000 MB = 1 GB math?
    # Text says: "2.5 GB for 8000 tokens so per-token KV cost is 2.5 GB / 8000 = 0.3125 MB"
    # Wait, 2.5 / 8000 = 0.0003125 GB = 0.3125 MB. This assumes 1 GB = 1000 MB.
    # We will stick to the exact math from the existing script to match the CSV exactly.
    
    naive_memory_gb = branches * 2.8125
    radix_memory_gb = 2.5 + branches * 0.3125

    df = pd.DataFrame({
        'branches': branches,
        'naive_memory_gb': naive_memory_gb,
        'radix_memory_gb': radix_memory_gb
    })

    # Ensure the target directory exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(target_dir, exist_ok=True)
    
    target_file = os.path.join(target_dir, 'kv_cache_scaling.csv')
    df.to_csv(target_file, index=False)
    print(f"Successfully generated {target_file}")

if __name__ == '__main__':
    generate_kv_cache_scaling_data()
