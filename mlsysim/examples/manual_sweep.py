"""
Tutorial: The Manual Sweep Pattern
==================================
This tutorial teaches students how to "think like a systems engineer" 
by manually sweeping a parameter (Batch Size) to find the "Cliff."
"""

import mlsysim
import pandas as pd # Optional, but common for students

def main():
    print("Scenario: Autonomous Vehicle Perception on Jetson Orin NX")
    scenario = mlsysim.Applications.AutoDrive
    
    results = []
    
    # 1. Manually sweep batch sizes from 1 to 128
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print(f"{'Batch':<10} | {'Status':<10} | {'Latency':<15} | {'Bottleneck':<15}")
    print("-" * 60)
    
    for b in batch_sizes:
        # Evaluate each point
        evaluation = scenario.evaluate(batch_size=b)
        
        # Flatten results for our table
        row = evaluation.to_dict()
        
        # Print a quick summary row
        status = row['p_status'] if row['f_status'] == "PASS" else "OOM"
        latency = f"{row['p_latency']:.2f}" if row['f_status'] == "PASS" else "---"
        bottleneck = row.get('p_bottleneck', "---")
        
        print(f"{b:<10} | {status:<10} | {latency:<15} | {bottleneck:<15}")
        
        # Collect for later deep analysis
        results.append(row)

    # 2. Convert the list of dicts to a DataFrame for analysis
    df = pd.DataFrame(results)
    print("\nDataFrame Summary (First 5 rows):")
    print(df[['scenario', 'f_status', 'p_latency']].head())

if __name__ == "__main__":
    main()

# Expected output (mlsysim v0.1.1):
# Scenario: Autonomous Vehicle Perception on Jetson Orin NX
# Batch      | Status     | Latency         | Bottleneck
# ------------------------------------------------------------
# 1          | PASS       | 1.25 millisecond | Memory
# 2          | PASS       | 1.36 millisecond | Compute
# 4          | PASS       | 2.01 millisecond | Compute
# 8          | PASS       | 3.32 millisecond | Compute
# 16         | PASS       | 5.95 millisecond | Compute
# 32         | FAIL       | 11.20 millisecond | Compute
# 64         | FAIL       | 21.69 millisecond | Compute
# 128        | FAIL       | 42.68 millisecond | Compute
#
# DataFrame Summary (First 5 rows):
#              scenario f_status              p_latency
# 0  Autonomous Vehicle     PASS   1.252156862745098 ms
# 1  Autonomous Vehicle     PASS               1.356 ms
# 2  Autonomous Vehicle     PASS               2.012 ms
# 3  Autonomous Vehicle     PASS  3.3240000000000003 ms
# 4  Autonomous Vehicle     PASS               5.948 ms
