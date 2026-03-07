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
