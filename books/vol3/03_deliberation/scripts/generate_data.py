import pandas as pd
import os

def calculate_speedup(workers, serial_fraction):
    """
    Calculate effective speedup using Amdahl's Law.
    Speedup = 1 / (s + (1 - s) / N)
    where s is the serial fraction and N is the number of workers.
    """
    return 1 / (serial_fraction + (1 - serial_fraction) / workers)

def main():
    # Scenarios for Amdahl's law applied to agent topologies.
    # Numbers come from theoretical analysis of agent coordination overhead:
    # - Baseline Scenario (s=0.143): derived from 20s serial (8s planner + 12s aggregator) 
    #   and 120s parallelizable subtasks. (20 / 140 = ~0.143)
    # - Low Overhead (s=0.05): Idealized scenario with highly parallelizable tasks
    # - High Overhead (s=0.33): Scenario where planning/aggregation takes a significant portion of the time
    scenarios = [
        {"serial_fraction": 0.05, "label": "Low Overhead (s=0.05)"},
        {"serial_fraction": 0.143, "label": "Baseline Scenario (s=0.143)"},
        {"serial_fraction": 0.33, "label": "High Overhead (s=0.33)"}
    ]

    max_workers = 32
    records = []

    for workers in range(1, max_workers + 1):
        for scenario in scenarios:
            s = scenario["serial_fraction"]
            speedup = calculate_speedup(workers, s)
            records.append({
                "Workers": workers,
                "SerialFraction": s,
                "Scenario": scenario["label"],
                "Speedup": speedup
            })

    df = pd.DataFrame(records)
    
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    
    # Save to CSV
    csv_path = "../data/agent_amdahls_law.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data successfully generated and saved to {csv_path}")

if __name__ == "__main__":
    main()
