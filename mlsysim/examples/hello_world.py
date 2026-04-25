"""
Hello World: Your First mlsysim Analysis
=========================================
The simplest possible mlsysim workflow: load a model and hardware,
run the Roofline engine, and see where the bottleneck is.

    python3 examples/hello_world.py
"""

import mlsysim

# 1. Pick a model and hardware from the built-in registries
model = mlsysim.Models.Language.Llama3_8B
hardware = mlsysim.Hardware.Cloud.H100

# 2. Run the Roofline engine
profile = mlsysim.Engine.solve(model, hardware, batch_size=1)

# 3. See the results
print(f"Model:      {model.name}")
print(f"Hardware:   {hardware.name}")
print(f"Bottleneck: {profile.bottleneck}")
print(f"Latency:    {profile.latency:~P}")
print(f"Throughput: {profile.throughput:~P}")
print(f"MFU:        {profile.mfu:.3f}")
print(f"Feasible:   {profile.feasible}")

# 4. Try a different configuration — just change the inputs
print("\n--- Batch size 32 ---")
profile_batched = mlsysim.Engine.solve(model, hardware, batch_size=32)
print(f"Bottleneck: {profile_batched.bottleneck}")
print(f"Latency:    {profile_batched.latency:~P}")
print(f"Throughput: {profile_batched.throughput:~P}")
print(f"MFU:        {profile_batched.mfu:.3f}")

# Expected output (mlsysim v0.1.1):
# Model:      Llama-3.1-8B
# Hardware:   NVIDIA H100
# Bottleneck: Memory
# Latency:    5.603432835820896 ms
# Throughput: 178.4620302053645 1/s
# MFU:        0.003
# Feasible:   True
#
# --- Batch size 32 ---
# Bottleneck: Memory
# Latency:    20.46492537313433 ms
# Throughput: 1563.6509499325382 1/s
# MFU:        0.025
