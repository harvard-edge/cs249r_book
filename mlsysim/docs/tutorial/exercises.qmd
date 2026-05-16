# MLSys·im Tutorial: Hands-On Exercises

Eight exercises aligned with the eight tutorial parts.
Each exercise is self-contained and takes 5-10 minutes.

---

## Exercise 1 — The Roofline Transition (Part 1: Single-Node Performance)

**Learning Objective:** Identify the batch size where a CNN workload transitions
from memory-bound to compute-bound on a datacenter GPU.

### Setup

```python
import mlsysim
from mlsysim import Engine, Hardware, Models

model = Models.ResNet50
hw    = Hardware.A100
```

### Task

Sweep `batch_size` from 1 to 512 (powers of 2). For each, call
`Engine.solve()` and record the `bottleneck` field along with the
compute and memory latency components. Find the smallest batch size
where the compute term overtakes the memory term.

```python
for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    p = Engine.solve(model, hw, batch_size=bs, precision="fp16", efficiency=0.5)
    print(f"bs={bs:>4d}  bottleneck={p.bottleneck:<10s}  "
          f"T_compute={p.latency_compute.to('ms'):~P.3f}  "
          f"T_memory={p.latency_memory.to('ms'):~P.3f}  "
          f"throughput={p.throughput:~P.1f}")
```

### Question

At what batch size does ResNet-50's compute time overtake its memory time
on the A100? Compare `latency_compute` vs. `latency_memory` to find the
exact crossover point.

### Hint

Watch for when `latency_compute > latency_memory`. The crossover is the
Roofline ridge point. Note that the reported `bottleneck` field depends
on the overall arithmetic intensity calculation, so inspect both latency
components directly.

<details>
<summary><strong>Expected Answer</strong></summary>

ResNet-50 on the A100 at FP16 with efficiency=0.5 is already compute-bound
at very small batch sizes because of its high FLOP count (~8 GFLOPs per
image) relative to its small weight footprint (~51 MB at FP16). The compute
term scales linearly with batch size while the memory term grows more slowly
(weights loaded once, only activation traffic scales). For CNN workloads
like ResNet-50, the transition may already be at batch_size=1, unlike
Transformer decode which is memory-bound at batch_size=1. Compare with
`Models.Llama3_8B` to see a memory-bound regime.

</details>

### Discussion

Why does this transition point matter for production deployment? Compare
ResNet-50 (compute-bound even at batch_size=1) with Llama-3-8B (memory-bound
at batch_size=1). What hardware characteristic should you optimize for
in each case -- peak FLOPS or memory bandwidth?

---

## Exercise 2 — LLM Serving Capacity (Part 2: Serving and Inference)

**Learning Objective:** Determine how many concurrent LLM requests fit in
GPU memory, accounting for both model weights and KV-cache.

### Setup

```python
from mlsysim import ServingModel, Hardware, Models

serving = ServingModel()
model   = Models.Llama3_8B
hw      = Hardware.H100
```

### Task

Sweep `batch_size` from 1 to 128 at `seq_len=4096`. Find the maximum
batch size where the serving result is still `feasible`.

```python
for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
    r = serving.solve(model, hw, seq_len=4096, batch_size=bs, precision="fp16")
    print(f"bs={bs:>4d}  feasible={r.feasible}  "
          f"mem={r.total_memory_required:~P.1f}  "
          f"kv_cache={r.kv_cache_size:~P.1f}  "
          f"TTFT={r.ttft:~P.1f}  ITL={r.itl:~P.2f}")
```

### Question

How many concurrent Llama-3-8B requests can a single H100 (80 GB) serve
at 4K context in FP16? What is the dominant memory consumer at that
maximum batch size -- weights or KV-cache?

### Hint

H100 has 80 GB HBM3. Llama-3-8B at FP16 is ~16 GB of weights.
Each request's KV-cache at 4096 tokens depends on the model's
layer count, head count, and head dimension.

<details>
<summary><strong>Expected Answer</strong></summary>

The maximum concurrent batch size is approximately **128 requests** on
the H100 (80 GiB = ~85.9 GB). At FP16, Llama-3-8B weights consume ~16 GB.
Each request's KV-cache at 4096 tokens is ~0.5 GB, so after reserving
~16 GB for weights, the remaining ~70 GB fits roughly 128 KV-cache slots.
At maximum capacity, **KV-cache dominates** memory usage (~69 GB KV vs.
~16 GB weights). At bs=160, total memory exceeds capacity and becomes
infeasible.

</details>

### Discussion

What happens if you switch from FP16 to INT8 quantization? The weights
halve, but the KV-cache also shrinks. Which effect matters more for
concurrent serving capacity?

---

## Exercise 3 — Quantization Trade-offs (Part 3: Compression and Efficiency)

**Learning Objective:** Quantify the memory savings and inference speedup
from INT4 quantization, and understand the accuracy trade-off.

### Setup

```python
from mlsysim import CompressionModel, Engine, Hardware, Models

compress = CompressionModel()
model    = Models.Llama3_8B
hw       = Hardware.H100
```

### Task

Compare the FP16 baseline with INT4 quantization. Run the compression
model and then run `Engine.solve()` at both precisions.

```python
# Compression analysis
result = compress.solve(model, hw, method="quantization", target_bitwidth=4)
print(f"Compression ratio:   {result.compression_ratio:.1f}x")
print(f"Original size:       {result.original_size_gb:~P.2f}")
print(f"Compressed size:     {result.compressed_size_gb:~P.2f}")
print(f"Memory savings:      {result.memory_savings_pct:.1f}%")
print(f"Inference speedup:   {result.inference_speedup:.2f}x")
print(f"Accuracy delta:      {result.estimated_accuracy_delta:.2f}%")

# Roofline comparison
fp16 = Engine.solve(model, hw, batch_size=1, precision="fp16", efficiency=0.5)
int4 = Engine.solve(model, hw, batch_size=1, precision="int4", efficiency=0.5)
print(f"\nFP16 latency: {fp16.latency:~P.2f}  bottleneck: {fp16.bottleneck}")
print(f"INT4 latency: {int4.latency:~P.2f}  bottleneck: {int4.bottleneck}")
```

### Question

What is the memory savings from FP32 baseline to INT4 for Llama-3-8B?
What is the estimated accuracy degradation? Is the speedup closer to
8x or 4x, and why?

### Hint

The `CompressionModel` measures compression ratio from the **FP32 baseline**
(32-bit), so INT4 gives an 8x ratio on paper (32/4). But inference speedup
depends on whether the workload is compute-bound or memory-bound. At
batch_size=1, LLM inference is memory-bound, so the speedup tracks with
the reduction in bytes moved, not FLOPS saved.

<details>
<summary><strong>Expected Answer</strong></summary>

- **Memory savings:** ~87.5% (8x compression from FP32 to INT4 baseline),
  reducing the model from ~32 GB (FP32) to ~4 GB (INT4).
- **Accuracy delta:** Approximately 2-5% degradation (conservative estimate
  from the Gholami et al. survey).
- **Inference speedup:** At batch_size=1, the workload is memory-bound, so
  the speedup is roughly **proportional to the bytes reduction**. Compared to
  FP16 inference (the practical baseline), the speedup from INT4 is ~4x in
  memory traffic. The exact speedup depends on whether the hardware has
  native INT4 execution units (B200 does, H100 does not).

</details>

### Discussion

When would you choose INT8 over INT4? Consider a deployment where you
need <1% accuracy loss but also need to fit the model on a single GPU.
What is the optimal compression point?

---

## Exercise 4 — Parallelism Strategy Search (Part 4: Distributed Training)

**Learning Objective:** Find the optimal 3D parallelism configuration
(TP x PP x DP) for training a 70B model on a 64-GPU cluster.

### Setup

```python
from mlsysim import ParallelismOptimizer, Models, Systems

optimizer = ParallelismOptimizer()
model     = Models.Llama3_70B
fleet     = Systems.Clusters.Research_256  # 256 H100s
```

### Task

Since we want 64 GPUs, build a custom fleet. Then run the optimizer.

```python
from mlsysim.systems.types import Fleet
from mlsysim.systems.registry import Nodes, Fabrics

fleet_64 = Fleet(
    name="64-GPU H100 Cluster",
    node=Nodes.DGX_H100,
    count=8,  # 8 nodes x 8 GPUs = 64
    fabric=Fabrics.InfiniBand_NDR
)

result = optimizer.solve(
    model, fleet_64,
    batch_size=512,
    precision="fp16",
    efficiency=0.4,
    overlap_comm=True
)

print(f"Best config:  {result.best_config}")
print(f"Best MFU:     {result.best_mfu:.3f}")
print(f"Best step time: {result.best_step_time:~P.1f}")
print(f"Configs explored: {result.total_searched}")

# Print top candidates
for c in result.top_candidates[:10]:
    cfg = c['config']
    print(f"  TP={cfg['tp']:>2d}  PP={cfg['pp']:>2d}  DP={cfg['dp']:>2d}  "
          f"MFU={c['mfu']:.3f}")
```

### Question

What is the optimal TP x PP x DP split for Llama-3-70B on 64 H100s?
Why does the optimizer prefer TP=8 (one full node) for tensor parallelism?

### Hint

TP communication happens over NVLink (900 GB/s within a DGX H100 node).
PP communication is point-to-point (small volume). DP communication is
AllReduce over InfiniBand (across nodes). The optimizer balances memory
fit (TP+PP must shard the 70B model enough to fit) against communication
overhead.

<details>
<summary><strong>Expected Answer</strong></summary>

The optimizer typically finds **TP=8, PP=2, DP=4** or a nearby configuration.

- **TP=8** keeps tensor parallelism within a single DGX node (8 GPUs connected
  by NVLink at 900 GB/s), minimizing communication latency for the 2 AllReduce
  operations per layer.
- **PP=2** splits the 80 layers across 2 pipeline stages, reducing per-GPU
  memory to fit in 80 GB HBM.
- **DP=4** provides data parallelism across 4 groups, allowing a reasonable
  local batch size of 128 per DP rank.

The MFU is typically 0.30-0.45, reflecting the pipeline bubble and
communication overheads.

</details>

### Discussion

What happens if you double the cluster to 128 GPUs? Does MFU go up or
down? Which parallelism dimension should absorb the extra GPUs?

---

## Exercise 5 — Carbon Geography (Part 5: Sustainability)

**Learning Objective:** Quantify how datacenter location affects the
carbon footprint of a long training run.

### Setup

```python
from mlsysim import SustainabilityModel, Infra, Systems

sustain = SustainabilityModel()
fleet   = Systems.Clusters.Research_256  # 256 H100s
```

### Task

Compare a 30-day training run in Virginia (US Average grid) versus
Quebec (hydroelectric).

```python
# Virginia (US Average)
r_va = sustain.solve(fleet, duration_days=30, datacenter=Infra.US_Avg, mfu=0.4)
print(f"=== Virginia (US Average Grid) ===")
print(f"Energy:  {r_va.total_energy_kwh:,.0f} kWh")
print(f"Carbon:  {r_va.carbon_footprint_kg:,.0f} kg CO2")
print(f"Water:   {r_va.water_usage_liters:,.0f} liters")

# Quebec (Hydro)
r_qc = sustain.solve(fleet, duration_days=30, datacenter=Infra.Quebec, mfu=0.4)
print(f"\n=== Quebec (Hydroelectric) ===")
print(f"Energy:  {r_qc.total_energy_kwh:,.0f} kWh")
print(f"Carbon:  {r_qc.carbon_footprint_kg:,.0f} kg CO2")
print(f"Water:   {r_qc.water_usage_liters:,.0f} liters")

print(f"\n=== Comparison ===")
print(f"Carbon reduction:  {(1 - r_qc.carbon_footprint_kg / r_va.carbon_footprint_kg) * 100:.1f}%")
print(f"Water reduction:   {(1 - r_qc.water_usage_liters / r_va.water_usage_liters) * 100:.1f}%")
```

### Question

How much carbon (in kg CO2) does moving from Virginia to Quebec save for
a 30-day training run on 256 H100s? What fraction of the total energy
is consumed by cooling and power delivery (not compute)?

### Hint

Quebec's carbon intensity is ~1.2 gCO2/kWh (hydro) vs. US average
~390 gCO2/kWh (mixed grid). The PUE overhead is the ratio of total
facility energy to IT energy -- a PUE of 1.1 means 10% overhead.

<details>
<summary><strong>Expected Answer</strong></summary>

- **Energy:** Both regions consume similar total energy (~180,000-220,000 kWh
  depending on PUE), since the GPUs draw the same power regardless of location.
- **Carbon:** Virginia produces roughly **80,000-90,000 kg CO2** while Quebec
  produces approximately **200-300 kg CO2** -- a **~99% reduction**.
- **PUE overhead:** Quebec's liquid-cooled facility (PUE ~1.05) wastes ~5% on
  infrastructure vs. US average air-cooled (PUE ~1.1-1.2) wasting 10-20%.

The carbon savings come entirely from the grid's energy source, not from
using less power. This is why datacenter location is the single highest-leverage
sustainability decision.

</details>

### Discussion

If you also factor in cost (electricity price per kWh), does Quebec remain
the optimal choice? Use `EconomicsModel` to find out. What about the
network latency penalty if your team is in California?

---

## Exercise 6 — Pareto-Optimal Serving (Part 6: Economics and Fleet Design)

**Learning Objective:** Given a fixed budget, find the serving configuration
that maximizes throughput while meeting latency SLAs.

### Setup

```python
from mlsysim import (
    EconomicsModel, ServingModel, Hardware, Models,
    Systems, Infra
)
from mlsysim.systems.types import Fleet, Node, NetworkFabric
from mlsysim.systems.registry import Nodes, Fabrics

serving = ServingModel()
econ    = EconomicsModel()
model   = Models.Llama3_8B
```

### Task

Compare three hardware options for serving Llama-3-8B at 4K context,
each scaled to fit within a $1M annual budget.

```python
configs = [
    ("A100 cluster", Hardware.A100, Nodes.DGX_A100, 20),   # ~20 nodes
    ("H100 cluster", Hardware.H100, Nodes.DGX_H100, 8),    # ~8 nodes
    ("B200 cluster", Hardware.B200, Nodes.DGX_B200, 4),     # ~4 nodes
]

for name, hw, node_template, n_nodes in configs:
    fleet = Fleet(
        name=name, node=node_template, count=n_nodes,
        fabric=Fabrics.InfiniBand_NDR, region=Infra.US_Avg
    )
    # Economics: 365-day TCO
    tco = econ.solve(fleet, duration_days=365, mfu=0.3)

    # Serving: per-GPU capacity
    r = serving.solve(model, hw, seq_len=4096, batch_size=32, precision="fp16")

    total_gpus = fleet.total_accelerators
    print(f"\n=== {name} ({total_gpus} GPUs) ===")
    print(f"  TCO:        ${tco.tco_usd:,.0f}")
    print(f"  TTFT:       {r.ttft:~P.1f}")
    print(f"  ITL:        {r.itl:~P.2f}")
    print(f"  Feasible:   {r.feasible}")
    print(f"  Per-GPU mem: {r.total_memory_required:~P.1f}")
```

### Question

With a $1M annual budget, which hardware generation provides the best
cost-per-request for Llama-3-8B serving? Is it always the newest GPU?

### Hint

Newer GPUs have higher unit cost but also higher bandwidth and FLOPS.
The Pareto-optimal choice depends on whether serving is memory-bound
(bandwidth matters) or compute-bound (FLOPS matter), and how many
GPUs you can afford.

<details>
<summary><strong>Expected Answer</strong></summary>

The answer depends on the specific unit costs in the registry, but the
general finding is:

- **A100:** Cheapest per-unit, so you get the most GPUs, but each has lower
  bandwidth (2 TB/s vs. 3.35 TB/s for H100). Good for throughput-oriented
  workloads where you can batch aggressively.
- **H100:** Best balance of cost and performance for LLM serving. Higher
  bandwidth directly reduces ITL (inter-token latency) in the memory-bound
  decode phase.
- **B200:** Highest per-unit cost but offers FP8/INT4 support and highest
  bandwidth (~8 TB/s). Most cost-effective only if you can use lower precision.

The Pareto frontier typically shows **H100 as the sweet spot** for FP16
serving, with B200 winning if INT4/FP8 quantization is acceptable.

</details>

### Discussion

How does the analysis change if you add a latency SLA (e.g., ITL < 20ms)?
Does the Pareto-optimal choice shift when you constrain latency instead
of just minimizing cost?

---

## Exercise 7 — TinyML SLA Feasibility (Part 7: Edge and TinyML)

**Learning Objective:** Determine whether a keyword-spotting CNN can meet
a real-time SLA on a microcontroller.

### Setup

```python
from mlsysim import Engine, Models, ureg
from mlsysim.hardware.types import HardwareNode, ComputeCore, MemoryHierarchy

model = Models.Tiny.DS_CNN

# Construct the nRF52840 (Cortex-M4F @ 64 MHz) -- MLPerf Tiny reference platform
hw = HardwareNode(
    name="Nordic nRF52840 (Cortex-M4F)",
    release_year=2018,
    compute=ComputeCore(
        peak_flops=0.000064 * ureg.TFLOPs / ureg.s,
        precision_flops={"int8": 0.000128 * ureg.TFLOPs / ureg.s},
    ),
    memory=MemoryHierarchy(
        capacity=1 * ureg.MB,
        bandwidth=0.064 * ureg.GB / ureg.s,
        sram_capacity=256 * ureg.KiB,
        sram_bandwidth=0.256 * ureg.GB / ureg.s,
        flash_capacity=1 * ureg.MB,
        flash_bandwidth=0.064 * ureg.GB / ureg.s,
    ),
    tdp=0.015 * ureg.W,
    dispatch_tax=0.5 * ureg.ms,
)
```

### Task

Check if DS-CNN keyword spotting meets a 30ms latency SLA on the
nRF52840. Then explore what happens at different precisions and
efficiency levels.

```python
# Baseline: INT8 (native on Cortex-M4F)
p_int8 = Engine.solve(model, hw, batch_size=1, precision="int8", efficiency=0.3)
print(f"=== DS-CNN on nRF52840 (INT8) ===")
print(f"Latency:    {p_int8.latency:~P.2f}")
print(f"Bottleneck: {p_int8.bottleneck}")
print(f"Memory:     {p_int8.memory_footprint:~P.2f}")
print(f"Feasible:   {p_int8.feasible}")
print(f"Energy:     {p_int8.energy:~P.4f}")
print(f"Meets 30ms: {p_int8.latency.to('ms').magnitude < 30}")

# Compare: FP16 (no hardware support -- uses FP32 path)
p_fp16 = Engine.solve(model, hw, batch_size=1, precision="fp16", efficiency=0.1)
print(f"\n=== DS-CNN on nRF52840 (FP16 emulated) ===")
print(f"Latency:    {p_fp16.latency:~P.2f}")
print(f"Meets 30ms: {p_fp16.latency.to('ms').magnitude < 30}")

# Sweep efficiency to find the minimum required
for eff in [0.1, 0.2, 0.3, 0.4, 0.5]:
    p = Engine.solve(model, hw, batch_size=1, precision="int8", efficiency=eff)
    print(f"eff={eff:.1f}  latency={p.latency:~P.2f}  meets_30ms={p.latency.to('ms').magnitude < 30}")
```

### Question

Can DS-CNN meet a 30ms SLA on the nRF52840? If not, what is the binding
constraint, and what hardware or algorithmic change would close the gap?

### Hint

The nRF52840 has ~128 MOPS at INT8 and ~64 MFLOPS at FP32. DS-CNN
has ~6M FLOPs. Do the back-of-envelope math: 6M / (128M * efficiency).
Is this compute-bound or memory-bound?

<details>
<summary><strong>Expected Answer</strong></summary>

- **INT8 at efficiency=0.3:** DS-CNN inference takes approximately
  **500ms** on the nRF52840 -- far exceeding the 30ms SLA. The workload is
  **compute-bound**: 6M FLOPs / (128 MOPS * 0.3) ~ 156ms of raw compute,
  plus framework overhead (dispatch tax, layer tax) pushes it past 500ms.
- **FP16 emulated:** Even slower (~1500ms+) because the Cortex-M4F has no
  native FP16 support and falls back to the FP32 path at 64 MFLOPS.
- **Energy:** At 15 mW TDP over ~500ms, a single inference consumes roughly
  **8 millijoules** -- acceptable for battery operation, but the latency
  is the problem, not energy.
- **The gap:** To meet 30ms, you would need either a faster MCU (e.g.,
  Cortex-M7 at 480 MHz with DSP extensions) or a smaller model (fewer FLOPs).
  Even at efficiency=1.0, the raw compute time is 6M/128M = 47ms -- still
  above 30ms. The nRF52840 simply cannot meet this SLA for DS-CNN.

</details>

### Discussion

This result surprises many students: a "tiny" 26K-parameter model still
cannot meet a 30ms SLA on a microcontroller. What does this teach about
the relationship between model size and latency? Try the ESP32-S3
(`Hardware.Tiny.ESP32_S3`) which has ~20x the compute throughput.
Does it meet 30ms? What if you needed to run MobileNetV2 -- try
`Engine.solve(Models.Vision.MobileNetV2, hw, ...)` and observe
the `feasible` field for the memory wall.

---

## Exercise 8 — Capstone: Fleet Design Under Constraints (Parts 1-7)

**Learning Objective:** Design a complete serving fleet for Llama-3-70B
at 1000 QPS, within a $5M annual budget, deployed across two regions.

### Setup

```python
from mlsysim import (
    ServingModel, EconomicsModel, SustainabilityModel, CompressionModel,
    Hardware, Models, Infra
)
from mlsysim.systems.types import Fleet
from mlsysim.systems.registry import Nodes, Fabrics

model   = Models.Llama3_70B
serving = ServingModel()
econ    = EconomicsModel()
sustain = SustainabilityModel()
compress = CompressionModel()
```

### Task

Design a fleet that meets ALL constraints simultaneously:
- **Throughput:** 1000 QPS (queries per second) total across two regions
- **Latency:** ITL < 50ms per token
- **Budget:** < $5M annual TCO
- **Carbon:** < 500 tonnes CO2/year
- **Regions:** US East + Quebec (for redundancy)

**Step 1:** Determine per-GPU serving capacity.

```python
# How many QPS per H100 at FP16?
r = serving.solve(model, Hardware.H100, seq_len=2048, batch_size=1, precision="fp16")
print(f"Single H100: TTFT={r.ttft:~P.1f}  ITL={r.itl:~P.1f}  feasible={r.feasible}")

# Try with INT4 compression
c = compress.solve(model, Hardware.H100, method="quantization", target_bitwidth=4)
print(f"INT4 compression: {c.compression_ratio:.1f}x  accuracy_delta={c.accuracy_delta:.1f}%")
```

**Step 2:** Calculate fleet size needed for 1000 QPS.

```python
# Estimate: each H100 can decode ~X tokens/sec for this model
# tokens_per_sec_per_gpu = 1000 / itl_in_seconds
itl_sec = r.itl.to("s").magnitude
tokens_per_sec = 1.0 / itl_sec if itl_sec > 0 else 0
print(f"Tokens/sec per GPU: {tokens_per_sec:.1f}")
print(f"GPUs needed for 1000 QPS: ~{1000 / max(tokens_per_sec, 1):.0f}")
```

**Step 3:** Check budget and carbon for each region split.

```python
for us_pct in [0.3, 0.5, 0.7]:
    qc_pct = 1.0 - us_pct
    # (Adapt node count based on your QPS calculation above)
    n_total = 100  # placeholder -- replace with your calculation
    n_us = int(n_total * us_pct)
    n_qc = n_total - n_us

    fleet_us = Fleet(name="US East", node=Nodes.DGX_H100, count=max(1,n_us),
                     fabric=Fabrics.InfiniBand_NDR, region=Infra.US_Avg)
    fleet_qc = Fleet(name="Quebec", node=Nodes.DGX_H100, count=max(1,n_qc),
                     fabric=Fabrics.InfiniBand_NDR, region=Infra.Quebec)

    tco_us = econ.solve(fleet_us, duration_days=365, mfu=0.3)
    tco_qc = econ.solve(fleet_qc, duration_days=365, mfu=0.3)
    co2_us = sustain.solve(fleet_us, duration_days=365, mfu=0.3)
    co2_qc = sustain.solve(fleet_qc, duration_days=365, mfu=0.3)

    total_tco = tco_us.tco_usd + tco_qc.tco_usd
    total_co2 = (co2_us.carbon_footprint_kg + co2_qc.carbon_footprint_kg) / 1000  # tonnes
    print(f"Split {us_pct:.0%} US / {qc_pct:.0%} QC:  "
          f"TCO=${total_tco:,.0f}  CO2={total_co2:.0f}t")
```

### Question

What is your recommended fleet configuration? How many total GPUs,
what regional split, and does INT4 quantization change your answer?

### Hint

Llama-3-70B at FP16 requires ~140 GB -- it does not fit on a single
H100 (80 GB). You must either use tensor parallelism (2+ GPUs per
inference instance) or quantize to INT4 (~35 GB, fits on one H100).
This fundamentally changes the fleet size calculation.

<details>
<summary><strong>Expected Answer</strong></summary>

This is an open-ended design exercise. A strong answer includes:

1. **Precision choice:** INT4 quantization (35 GB) fits on a single H100,
   halving the GPU requirement vs. FP16 (which needs TP=2 minimum).
   The 2-5% accuracy trade-off is usually acceptable for serving.

2. **Fleet size:** With INT4 and typical ITL of ~30-50ms per token, each
   H100 can handle roughly 20-30 QPS. For 1000 QPS total, you need
   approximately 35-50 GPUs (5-7 DGX H100 nodes).

3. **Regional split:** Placing 70% of capacity in Quebec dramatically
   reduces carbon (hydro grid at ~1.2 gCO2/kWh vs. US average at
   ~390 gCO2/kWh) while keeping 30% in US East for latency-sensitive
   users. Total CO2 stays well under 500 tonnes/year.

4. **Budget check:** 5-7 DGX H100 nodes at ~$200-300K each, plus
   electricity and networking, fits within the $5M annual budget.

The key insight is that **compression is not just an optimization -- it is
an architectural decision** that changes the fleet design by 2x.

</details>

### Discussion

What are the failure modes of this design? What happens when a full
DGX node goes down in the Quebec region? How does the ReliabilityModel
inform your redundancy strategy? Should you over-provision by N+1 or N+2?
