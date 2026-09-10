"""
This script generates the synthetic data for Multi-Agent Amdahl's Law
as discussed in the 'multi_agent_coordination.qmd' chapter.

Data is based on an analytical model (the "Coordination Tax") derived in the text.
Parameters:
- s = 0.12 (fraction of task that is strictly sequential)
- alpha = 0.003 (quadratic coordination coefficient, representing pairwise communication and consensus)
- beta = 0.02 (linear coordination coefficient, representing context re-encoding and prompt overhead)

Equations:
Classical Amdahl's Law: S_classical(M) = 1 / (s + (1 - s) / M)
Multi-Agent Amdahl's Law: S_multi_agent(M) = 1 / (s + (1 - s) / M + alpha * M^2 + beta * M)
"""

import pandas as pd
import os

def generate_amdahls_law_data():
    s = 0.12
    alpha = 0.003
    beta = 0.02

    data = []
    for M in range(1, 21):
        classical_speedup = 1 / (s + (1 - s) / M)
        multi_agent_speedup = 1 / (s + (1 - s) / M + alpha * (M**2) + beta * M)
        data.append({
            "Agents": M,
            "Classical_Speedup": classical_speedup,
            "Multi_Agent_Speedup": multi_agent_speedup
        })

    df = pd.DataFrame(data)
    
    # Ensure the target directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(script_dir, "../data/amdahls_law_multi_agent.csv")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    df.to_csv(target_path, index=False)
    print(f"Successfully generated data and saved to {target_path}")

if __name__ == "__main__":
    generate_amdahls_law_data()
