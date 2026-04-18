"""
Example 06: Multi-Objective Optimization (Pareto Fronts)
--------------------------------------------------------
This script demonstrates how to construct a Pareto front of trade-offs
between Throughput (maximize) and P99 Latency (minimize) by sweeping
batch sizes through ``ServingModel`` and ``TailLatencyModel``.

It then uses ``BatchingOptimizer`` to confirm the largest batch size
that still satisfies a P99 SLA, marked on the front below.

This is a core pattern in ML systems engineering: you rarely have a
single "best" design — you have a frontier of feasible operating points,
and the SLA chooses one of them.
"""
import mlsysim
from mlsysim.core.solver import ServingModel, TailLatencyModel


def main():
    print("Building Pareto Front: Llama-3 8B Inference on A100\n")

    model = mlsysim.Models.Language.Llama3_8B
    hardware = mlsysim.Hardware.Cloud.A100

    seq_len = 128
    arrival_rate_qps = 10.0
    sla_latency_ms = 20_000.0  # 20 second budget
    precision = "fp16"

    serving = ServingModel()
    tail = TailLatencyModel()

    # 1. Sweep batch sizes to construct the Pareto front
    points = []
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        srv = serving.solve(
            model, hardware,
            seq_len=seq_len, batch_size=b,
            precision=precision,
        )
        if not srv.feasible:
            continue

        service_latency = srv.ttft + (srv.itl * seq_len)
        tl = tail.solve(
            arrival_rate_qps=arrival_rate_qps / b,
            service_latency_ms=service_latency.m_as("ms"),
            num_replicas=1,
        )
        if not tl.is_stable:
            continue

        # Saturation throughput at this batch size: how many sequences/sec
        # one replica could serve back-to-back (1 / per-batch service time).
        sat_throughput = b * 1000.0 / service_latency.m_as("ms")

        points.append({
            "batch_size": b,
            "p99_latency_ms": tl.p99_latency.m_as("ms"),
            "throughput_seq_per_s": sat_throughput,
            "feasible_under_sla": tl.p99_latency.m_as("ms") <= sla_latency_ms,
        })

    # 2. Confirm the optimum via BatchingOptimizer
    optimizer = mlsysim.BatchingOptimizer()
    result = optimizer.solve(
        model=model,
        hardware=hardware,
        seq_len=seq_len,
        arrival_rate_qps=arrival_rate_qps,
        sla_latency_ms=sla_latency_ms,
        precision=precision,
    )

    print(f"Optimal Batch Size (≤ {sla_latency_ms/1000:.0f}s SLA): {result.best_batch_size}")
    print(f"Throughput at optimum:                 {result.max_throughput:.1f} seq/s")
    print(f"P99 latency at optimum:                {result.p99_latency:~.1f}\n")

    print("--- Pareto Front (sweep) ---")
    print(f"{'Batch':<6} | {'P99 latency (ms)':<18} | {'Throughput (seq/s)':<20} | SLA")
    print("-" * 70)
    for p in points:
        marker = " <-- OPTIMAL" if p["batch_size"] == result.best_batch_size else ""
        sla_flag = "OK" if p["feasible_under_sla"] else "violates"
        print(
            f"{p['batch_size']:<6} | {p['p99_latency_ms']:<18.1f} | "
            f"{p['throughput_seq_per_s']:<20.1f} | {sla_flag}{marker}"
        )


if __name__ == "__main__":
    main()
