"""
Example 06: Multi-Objective Optimization (Pareto Fronts)
--------------------------------------------------------
This script demonstrates how to use the BatchingOptimizer to find the
Pareto front of trade-offs between Throughput (maximizing) and Latency (minimizing).
This is a core concept in ML systems engineering.
"""
import mlsysim

def main():
    print("Finding the Pareto Front for Llama-3 8B Inference on A100...\n")

    # 1. Define the Workload and Hardware
    model = mlsysim.Models.Language.Llama3_8B
    hardware = mlsysim.Hardware.Cloud.A100

    # 2. Define the Optimizer
    optimizer = mlsysim.BatchingOptimizer()

    # 3. Solve for the optimal batch sizes
    # We set a strict SLA of 50ms. The optimizer will find the batch size
    # that maximizes throughput without violating this latency constraint.
    result = optimizer.solve(
        model=model,
        hardware=hardware,
        seq_len=128,
        arrival_rate_qps=10.0,
        sla_latency_ms=20000.0, # 20 seconds
        precision="fp16"
    )

    print(f"Optimal Batch Size (to meet 20s SLA): {result.best_batch_size}")
    print(f"Max Throughput at SLA: {result.max_throughput:.1f} seq/s")
    print(f"Latency at optimal batch: {result.p99_latency:.1f}")
    
    print("\n--- The Pareto Front ---")
    print(f"{'Batch Size':<12} | {'Latency (ms)':<15} | {'Throughput (seq/s)':<20}")
    print("-" * 55)
    
    # 4. Display the Pareto Front
    for point in result.pareto_front:
        b = point['batch_size']
        lat = point['p99_latency'].m_as('ms')
        tpt = point['throughput']
        
        # Highlight the optimal point
        marker = " <--- OPTIMAL" if b == result.best_batch_size else ""
        print(f"{b:<12} | {lat:<15.1f} | {tpt:<20.1f}{marker}")

if __name__ == "__main__":
    main()
