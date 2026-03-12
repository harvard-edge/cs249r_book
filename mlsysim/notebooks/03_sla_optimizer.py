import marimo

__generated_with = "0.19.6"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    import mlsysim
    import pandas as pd
    import matplotlib.pyplot as plt
    return mlsysim, mo, pd, plt

@app.cell
def __(mo):
    mo.md(
        """
        # The Batching Paradox (SLA Search)
        
        This notebook demonstrates the **BatchingOptimizer** (Tier 3: Engineering Engine).
        
        In production LLM serving, there is a fundamental tension:
        * The **Provider** wants to maximize *Throughput* (by increasing batch size) to make money.
        * The **User** wants to minimize *Tail Latency* (wait time) to have a good experience.
        
        If you batch too aggressively, the queueing delay explodes non-linearly (the "Wait Wall").
        
        Use the slider below to set your strict **P99 Latency Service Level Agreement (SLA)**. The Optimizer will search the continuous batching space and the M/M/c queueing models to find the absolute maximum throughput you can sustain without violating that SLA.
        """
    )
    return

@app.cell
def __(mo):
    sla_slider = mo.ui.slider(
        start=50, stop=5000, step=50, value=1000, 
        label="Strict P99 SLA Limit (ms)"
    )
    
    qps_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=10, 
        label="Arrival Rate (Queries Per Sec)"
    )
    
    mo.md(
        f"""
        ### 1. Set the Production Constraints
        {sla_slider}
        
        {qps_slider}
        """
    )
    return qps_slider, sla_slider

@app.cell
def __(mlsysim, mo, pd, plt, qps_slider, sla_slider):
    # 1. Setup the Model and Hardware
    model = mlsysim.Models.Language.Llama3_8B
    hardware = mlsysim.Hardware.Cloud.H100
    seq_len = 2048
    
    # 2. Run the Optimizer
    optimizer = mlsysim.BatchingOptimizer()
    
    result = optimizer.solve(
        model=model,
        hardware=hardware,
        seq_len=seq_len,
        sla_latency_ms=sla_slider.value,
        arrival_rate_qps=qps_slider.value,
        num_replicas=4, # 4-GPU serving fleet
        max_search_batch=128
    )
    
    if result.is_feasible:
        scorecard = mo.md(
            f"""
            ### 2. Optimizer Scorecard
            
            **Winning Configuration:**
            * Optimal Max Batch Size: `{result.best_batch_size}`
            
            **Performance Results:**
            * Maximum Safe Throughput: `{result.max_throughput:,.0f} tokens/sec`
            * Expected P99 Latency: `{result.p99_latency.to("ms"):~.0f}` (SLA: {sla_slider.value} ms)
            * Probability of SLA Violation: `{result.slo_violation_probability:.2%}`
            
            *The optimizer searched {result.total_searched} batch sizes before finding the limit.*
            """
        )
    else:
        scorecard = mo.md(
            f"""
            ### 🚨 SLA Unattainable
            Even at batch size 1, the system cannot process `{qps_slider.value} QPS` while keeping the P99 latency under `{sla_slider.value} ms`. You must add more replicas or relax the SLA.
            """
        )
    
    scorecard
    return hardware, model, optimizer, result, scorecard, seq_len

if __name__ == "__main__":
    app.run()
