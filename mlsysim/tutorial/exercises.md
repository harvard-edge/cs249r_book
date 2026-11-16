# MLSys·im Tutorial: Hands-On Exercises

Eight exercises spanning seven core tutorial parts plus an integrated capstone.
Each exercise is self-contained and takes 5-10 minutes.

---

## Exercise 1: The Roofline Transition (Part 1: Single-Node Performance)

**Learning Objective:** Identify the batch size where a CNN workload transitions
from memory-bound to compute-bound on a datacenter GPU.

### Setup

```python
import mlsysim
from mlsysim import Engine, Hardware, Models

model = Models.Vision.ResNet50
hw    = Hardware.Cloud.A100
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
Roofline ridge point. The reported `bottleneck` field depends
on the overall arithmetic intensity calculation, so inspect both latency
components directly.

<details>
<summary><strong>Expected Answer</strong></summary>

ResNet-50 on the A100 at FP16 with `efficiency=0.5` is **already compute-bound at `batch_size=1`** (`T_compute = 0.053 ms` vs. `T_memory = 0.048 ms`, `14,800.8` samples/s) because of its high FLOP count (~`8.2` GFLOPs per image) relative to its small weight footprint (~`51.1` MB at FP16). As `batch_size` grows to `512`, `T_compute` (`26.913 ms`) outpaces `T_memory` (`11.544 ms`) and throughput saturates near `19,013.8` samples/s. Compare with `Models.Language.Llama3_8B` on the same A100, which stays **memory-bound** from `batch_size=1` (`T_compute = 0.103 ms` vs. `T_memory = 7.877 ms`) up through `batch_size=64` and only crosses into the compute-bound regime at `batch_size=128` (`13.177 ms` vs. `7.942 ms`).

</details>

### Discussion

Why does this transition point matter for production deployment? Compare
ResNet-50 (compute-bound even at batch_size=1) with Llama-3-8B (memory-bound
at batch_size=1). What hardware characteristic should you optimize for
in each case: peak FLOP/s or memory bandwidth?

---

## Exercise 2: LLM Serving Capacity (Part 2: Serving and Inference)

**Learning Objective:** Determine how many concurrent LLM requests fit in
GPU memory, accounting for both model weights and KV-cache.

### Setup

```python
from mlsysim import Hardware, Models
from mlsysim.solvers import ServingModel

serving = ServingModel()
model   = Models.Language.Llama3_8B
hw      = Hardware.Cloud.H100
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

How many concurrent Llama-3-8B requests can a single H100 (80 GiB / 85.9 GB) serve
at 4K context in FP16? What is the dominant memory consumer at that
maximum batch size: weights or KV-cache?

### Hint

H100 has 80 GiB (`85.9 GB`) HBM3. Llama-3-8B at FP16 is ~16 GB of weights.
Each request's KV-cache at 4096 tokens depends on the model's
layer count, head count, and head dimension.

<details>
<summary><strong>Expected Answer</strong></summary>

The maximum concurrent power-of-two batch size in the sweep is **128 requests** on the H100 (`80 GiB` = `85.90 GB` HBM3). At FP16, Llama-3-8B weights consume `14.96 GiB` (`16.06 GB`), and grouped-query attention (`H_kv = 8`) keeps each 4096-token request's KV-cache to `0.50 GiB` (`0.54 GB`). At `bs=128`, total memory reaches `78.96 GiB` (`84.78 GB`) — **KV-cache dominates** (`64.00 GiB` KV vs. `14.96 GiB` weights). Increasing concurrency past `bs=130` (e.g., `bs=136` at `82.96 GiB`) exceeds `80 GiB` HBM capacity and flips `feasible=False`.

</details>

### Discussion

What happens if you switch from FP16 to INT8 quantization? The weights
halve, but the KV-cache also shrinks. Which effect matters more for
concurrent serving capacity?

---

## Exercise 3: Quantization Trade-offs (Part 3: Compression and Efficiency)

**Learning Objective:** Quantify the memory savings and inference speedup
from INT4 quantization, and understand the accuracy trade-off.

### Setup

```python
from mlsysim import Engine, Hardware, Models
from mlsysim.solvers import CompressionModel

compress = CompressionModel()
model    = Models.Language.Llama3_8B
hw       = Hardware.Cloud.H100
```

### Task

Compare the FP16 baseline with INT4 quantization. Run the compression
model and then run `Engine.solve()` at both precisions.

```python
# Compression analysis (measured against FP16 weights with group_size=128 FP16 scales)
result = compress.solve(
    model, hw, method="quantization", target_bitwidth=4,
    baseline_precision="fp16", group_size=128, accuracy_delta=-0.015,
)
print(f"Compression ratio:   {result.compression_ratio:.2f}x")
print(f"Original size:       {result.original_size_gb:~P.2f}")
print(f"Compressed size:     {result.compressed_size_gb:~P.2f}")
print(f"Memory savings:      {result.memory_savings_pct:.1f}%")
print(f"Inference speedup:   {result.inference_speedup:.2f}x")
print(f"Accuracy delta:      {result.estimated_accuracy_delta:+.1%}")

# Roofline comparison
fp16 = Engine.solve(model, hw, batch_size=1, precision="fp16", efficiency=0.5)
int4 = Engine.solve(model, hw, batch_size=1, precision="int4", efficiency=0.5)
print(f"\nFP16 latency: {fp16.latency:~P.2f}  bottleneck: {fp16.bottleneck}")
print(f"INT4 latency: {int4.latency:~P.2f}  bottleneck: {int4.bottleneck}")
```

### Question

What is the memory savings from the FP16 baseline to grouped INT4 (`group_size=128`) for Llama-3-8B?
Why is `accuracy_delta` passed explicitly rather than guessed from bitwidth alone? Does the compression ratio
become an equivalent speedup here, and why?

### Hint

`CompressionModel` measures compression ratio against `baseline_precision` (which defaults to `"fp32"`). With `baseline_precision="fp16"` and `group_size=128` (`16`-bit scale per `128` weights = `4.125` effective bits/weight), INT4 achieves `16 / 4.125 = 3.88x` (`4.00x` if `group_size=None`). Because task degradation depends on model family, calibration set, and benchmark rather than bitwidth alone, `CompressionModel` echoes a caller-supplied `accuracy_delta` (`0.0` when omitted). At `batch_size=1`, LLM decode is memory-bound, so speedup tracks bytes moved rather than native INT4 Tensor Core FLOP/s.

<details>
<summary><strong>Expected Answer</strong></summary>

- **Memory savings:** **74.2%** (`3.88x` compression with `group_size=128` FP16 block scales, reducing weights from `16.06 GB` to `4.14 GB`; `75.0%` / `4.00x` without group-scale metadata).
- **Accuracy delta:** `CompressionModel` intentionally refuses to invent a synthetic accuracy curve; here we pass an assumed `-1.5%` illustrative quality delta (`accuracy_delta=-0.015`, within the range surveyed by Gholami et al. for 4-bit weight quantization).
- **Inference speedup:** At `batch_size=1`, Llama-3-8B is **memory-bound**, so latency drops from `4.80 ms` (FP16) to `1.21 ms` (INT4, ~`3.98x` raw weight-read reduction on `Engine.solve`, or `3.88x` with group scales), even though H100 has no native INT4 Tensor Core path and dequantizes weights in registers before FP16 math.

</details>

### Discussion

When would you choose INT8 over INT4? Consider a deployment where you
need <1% accuracy loss but also need to fit the model on a single GPU.
What is the optimal compression point?

---

## Exercise 4: Parallelism Strategy Search (Part 4: Distributed Training)

**Learning Objective:** Find the optimal 3D parallelism configuration
(TP x PP x DP) for training a 70B model on a 64-GPU cluster.

### Setup

```python
from mlsysim import Models, Systems
from mlsysim.solvers import ParallelismOptimizer

optimizer = ParallelismOptimizer()
model     = Models.Language.Llama3_70B
fleet     = Systems.Clusters.Research_256  # 256 H100s
```

### Task

This exercise uses 64 GPUs, so build a custom fleet. Then run the optimizer.

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

What is the optimal TP x PP x DP split for Llama-3-70B on 64 H100s at `batch_size=512`?
Why are `TP * PP < 16` splits eliminated outright, and why does `ParallelismOptimizer` keep `TP <= 8` within a single DGX node?

### Hint

Mixed-precision Adam requires `16 bytes/param` (`2 B` FP16 weights + `2 B` FP16 gradients + `12 B` FP32 master weights and moments), or `~1.03 TiB` (`1,130 GB`) of static training state for a `70.6B` model before activations! Without ZeRO sharding, `TP * PP` must be at least `16` (`>= 32` once activations at `seq_len=2048` are counted) just to fit in `80 GiB` HBM. Meanwhile, `TP` stays within a single DGX node (`TP <= 8` over `900 GB/s` NVLink), while `ParallelismOptimizer` schedules one-sample pipeline microbatches (`M = batch_size // DP`) to shrink the pipeline bubble.

<details>
<summary><strong>Expected Answer</strong></summary>

Out of all factorizations of 64 GPUs, **only 8 configurations fit static Adam state + activations in 80 GiB HBM** (`TP=8, PP=2, DP=4` OOMs once activations are added without activation recomputation). Among the 8 survivors, `ParallelismOptimizer` selects **`TP=2, PP=16, DP=2`** (`MFU = 0.370`, `step_time = 18,966.6 ms`), closely followed by `TP=2, PP=32, DP=1` (`0.368`) and `TP=4, PP=4, DP=4` (`0.367`):

- **Memory feasibility (`TP * PP >= 16`–`32`):** Sharding `~1.03 TiB` (`1,130 GB`) of Adam state + activations across model-parallel ranks eliminates pure DP and shallow sharding immediately.
- **Intra-node `TP <= 8`:** Tensor parallelism requires 4 AllReduces per layer per step and never crosses slow inter-node InfiniBand links; smaller intra-node `TP` (`2` or `4`) trims NVLink collective latency when `256` one-sample microbatches (`batch_size=512, DP=2`) keep the pipeline bubble small (`(16 - 1) / (256 + 16 - 1) = 5.5%`). At smaller global batch sizes (fewer microbatches, larger bubble) or with `activation_recomputation=True`, shallower pipelines with higher intra-node `TP` move to the front.

</details>

### Discussion

What happens if you double the cluster to 128 GPUs? Does MFU go up or
down? Which parallelism dimension should absorb the extra GPUs?

---

## Exercise 5: Carbon Geography (Part 5: Sustainability)

**Learning Objective:** Quantify how datacenter location affects the
carbon footprint of a long training run.

### Setup

```python
from mlsysim import Infrastructure, Systems
from mlsysim.solvers import SustainabilityModel

sustain = SustainabilityModel()
fleet   = Systems.Clusters.Research_256  # 256 H100s
```

### Task

Compare a 30-day training run in Virginia (US Average grid) versus
Quebec (hydroelectric).

```python
# Virginia (US Average)
r_va = sustain.solve(fleet, duration_days=30, datacenter=Infrastructure.Grids.US_Avg, mfu=0.4)
print(f"=== Virginia (US Average Grid) ===")
print(f"Energy:  {r_va.total_energy_kwh:~P,.0f}")
print(f"Carbon:  {r_va.carbon_footprint_kg:,.0f} kg CO2")
print(f"Water:   {r_va.water_usage_liters:,.0f} liters")

# Quebec (Hydro)
r_qc = sustain.solve(fleet, duration_days=30, datacenter=Infrastructure.Grids.Quebec, mfu=0.4)
print(f"\n=== Quebec (Hydroelectric) ===")
print(f"Energy:  {r_qc.total_energy_kwh:~P,.0f}")
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

Quebec's carbon intensity in the registry is `20 gCO2/kWh` (hydro, cold-climate `PUE = 1.06`, `WUE = 0.0 L/kWh`) vs. US Average `429 gCO2/kWh` (mixed grid, `PUE = 1.12`, `WUE = 1.8 L/kWh`). `SustainabilityModel` scales chip power proportionally (`30%` idle TDP + `70% * MFU` dynamic TDP), applies `PUE` to facility electricity and carbon, and applies on-site `WUE` to pre-PUE IT energy (`74,834 kWh`).

<details>
<summary><strong>Expected Answer</strong></summary>

- **Energy:** At `mfu=0.4`, each H100 draws `406 W` (`0.30 + 0.70 * 0.40 = 58%` of `700 W` TDP), giving `74,834 kWh` of IT energy over 30 days. Facility energy is **`83,814 kWh`** on the US Average grid (`PUE = 1.12`, so `10.7%` of total facility energy goes to cooling and power delivery) vs. **`79,324 kWh`** in Quebec (`PUE = 1.06`, `5.7%` infrastructure overhead).
- **Carbon:** Virginia (`US_Avg`) produces **`35,956 kg CO2`** vs. **`1,586 kg CO2`** in Quebec — saving **`34,370 kg CO2`** (**`95.6%` reduction**).
- **Water:** US Average evaporative cooling consumes **`134,701 liters`** (`74,834 kWh_IT * 1.8 L/kWh`) vs. **`0 liters`** in Quebec's cold-climate closed-loop cooling (**`100.0%` reduction**).

Nearly all of the carbon savings come from the grid's generation mix (`20` vs. `429 gCO2/kWh`), making datacenter location the highest-leverage sustainability decision.

</details>

### Discussion

If you also factor in cost (electricity price per kWh), does Quebec remain
the optimal choice? Use `EconomicsModel` to find out. What about the
network latency penalty if your team is in California?

---

## Exercise 6: Pareto-Optimal Serving (Part 6: Economics and Fleet Design)

**Learning Objective:** Given a fixed budget, find the serving configuration
that maximizes throughput while meeting latency SLAs.

### Setup

```python
from mlsysim import Hardware, Infrastructure, Models, Systems
from mlsysim.solvers import EconomicsModel, ServingModel
from mlsysim.systems.types import Fleet, Node, NetworkFabric
from mlsysim.systems.registry import Nodes, Fabrics

serving = ServingModel()
econ    = EconomicsModel()
model   = Models.Language.Llama3_8B
```

### Task

Compare three hardware options for serving Llama-3-8B at 4K context,
each scaled to fit within a $1M annual budget.

```python
configs = [
    ("A100 cluster", Hardware.Cloud.A100, Nodes.DGX_A100, 20),   # ~20 nodes
    ("H100 cluster", Hardware.Cloud.H100, Nodes.DGX_H100, 8),    # ~8 nodes
    ("B200 cluster", Hardware.Cloud.B200, Nodes.DGX_B200, 4),     # ~4 nodes
]

for name, hw, node_template, n_nodes in configs:
    fleet = Fleet(
        name=name, node=node_template, count=n_nodes,
        fabric=Fabrics.InfiniBand_NDR, region=Infrastructure.Grids.US_Avg
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

Newer GPUs have higher unit cost but also higher bandwidth and FLOP/s.
The Pareto-optimal choice depends on whether serving is memory-bound
(bandwidth matters) or compute-bound (FLOP/s matter), and how many
GPUs the budget buys.

<details>
<summary><strong>Expected Answer</strong></summary>

Under the 3-year amortized accelerator CapEx + OpEx model (`mfu=0.3` on `US_Avg`):

- **A100 (`20` DGX nodes = `160` GPUs, TCO `$958,429`/yr):** Highest raw aggregate decode capacity within `$1M` (`~314K` tok/s across 160 GPUs at `bs=32`), but slowest per-request latency (`TTFT = 14.4 s`, `ITL = 16.30 ms`) because each GPU has `2.04 TB/s` bandwidth and `312 TFLOP/s` FP16.
- **H100 (`8` DGX nodes = `64` GPUs, TCO `$640,233`/yr):** Cuts `TTFT` by `3.2x` (`4.54 s`) and `ITL` to `9.92 ms` (`~206K` tok/s fleet-wide, `~$0.10` per million decode tokens, virtually tied with A100 on cost-per-token while meeting tighter SLAs).
- **B200 (`4` DGX nodes = `32` GPUs, TCO `$509,881`/yr):** Highest `8.0 TB/s` HBM3e bandwidth drives `ITL` down to `4.15 ms` (`TTFT = 2.00 s`, `~246K` tok/s across just 32 GPUs), yielding the **lowest cost per million tokens (`~$0.066/M`)** and scaling further within the `$1M` budget (up to `7` DGX B200 nodes = `$892K`/yr), especially when native FP8/FP4 execution is enabled.

</details>

### Discussion

How does the analysis change if you add a latency SLA (e.g., ITL < 20ms)?
Does the Pareto-optimal choice shift when you constrain latency instead
of only minimizing cost?

---

## Exercise 7: TinyML SLA Feasibility (Part 7: Edge and TinyML)

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
        precision_flops={"int8": 0.000128 * ureg.TOPS},
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

The nRF52840 has `128 MOPS` at INT8 and `64 MFLOP/s` at FP32. `Models.Tiny.DS_CNN`
has `200K` parameters and `20M` FLOPs per inference. Do the back-of-envelope math: `20M / (128M * efficiency)`.
Is this compute-bound or memory-bound?

<details>
<summary><strong>Expected Answer</strong></summary>

- **INT8 at `efficiency=0.3`:** DS-CNN inference takes **`521.33 ms`** on the nRF52840 (`520.83 ms` compute + `0.50 ms` dispatch tax), far exceeding the `30 ms` SLA. The workload is **compute-bound**: `20M FLOPs / (128 MOPS * 0.3) = 520.83 ms` vs. `0.78 ms` memory time (`200 KB` INT8 weights fit inside the nRF52840's `256 KiB` on-chip SRAM at `0.256 GB/s`, avoiding the `0.064 GB/s` Flash ceiling).
- **FP16 emulated (`efficiency=0.1`):** Slows to **`3,125.50 ms`** (`~3.13 s`) because the Cortex-M4F lacks native FP16 Tensor/SIMD units and falls back to the `64 MFLOP/s` FP32 path.
- **Energy:** Under the utilization-scaled power model (`15 mW` TDP), one `521 ms` INT8 inference consumes **`3.99 mJ`**, comfortable for battery operation — latency is the binding constraint, not energy.
- **The gap:** Even at `efficiency=1.0`, raw INT8 compute takes `20M / 128M + 0.5 ms = 156.75 ms`, over `5x` slower than `30 ms`. Upgrading to `Hardware.Tiny.ESP32_S3` (`0.0024 TOPS` = `2.4 GOPS` vector INT8) brings latency down to **`28.78 ms`** at `efficiency=0.3`, meeting the `30 ms` real-time SLA.

</details>

### Discussion

This result surprises many students: a "tiny" `200K`-parameter model still
cannot meet a `30 ms` SLA on a `64 MHz` microcontroller. What does this teach about
the relationship between parameter count, FLOPs (`20M` FLOPs = `10M` conv MACs), and latency? Try the ESP32-S3
(`Hardware.Tiny.ESP32_S3`), which achieves `28.78 ms` at `efficiency=0.3` and meets the `30 ms` target.
What if you needed to run MobileNetV2? Try
`Engine.solve(Models.Vision.MobileNetV2, hw, ...)` and observe
the `feasible` field for the `1 MB` memory wall.

---

## Exercise 8: Capstone: Fleet Design Under Constraints (Parts 1-7)

**Learning Objective:** Design a complete serving fleet for Llama-3-70B
at 1000 QPS, within a $5M annual budget, deployed across two regions.

### Setup

```python
from mlsysim import Hardware, Infrastructure, Models
from mlsysim.solvers import (
    CompressionModel,
    EconomicsModel,
    ServingModel,
    SustainabilityModel,
)
from mlsysim.systems.types import Fleet
from mlsysim.systems.registry import Nodes, Fabrics

model    = Models.Language.Llama3_70B
serving  = ServingModel()
econ     = EconomicsModel()
sustain  = SustainabilityModel()
compress = CompressionModel()
```

### Task

Design a fleet that meets ALL constraints simultaneously:
- **Throughput:** 1000 QPS (queries per second) total across two regions
- **Latency:** ITL < 50ms per token
- **Budget:** < $5M annual TCO
- **Carbon:** < 500 tonnes CO2/year
- **Regions:** US East + Quebec (for redundancy)

**Step 1:** Determine per-GPU serving capacity (for interactive chat with 256-token prompts and 128-token responses).

```python
# Does Llama-3-70B fit on a single H100 at FP16 vs. INT4?
r_fp16 = serving.solve(model, Hardware.Cloud.H100, seq_len=256, batch_size=1, precision="fp16")
print(f"Single H100 (FP16): TTFT={r_fp16.ttft:~P.1f}  ITL={r_fp16.itl:~P.1f}  feasible={r_fp16.feasible}")

# Quantize to INT4 (group_size=128, assumed -1.5% quality delta) and check batched serving
c = compress.solve(
    model, Hardware.Cloud.H100, method="quantization", target_bitwidth=4,
    baseline_precision="fp16", group_size=128, accuracy_delta=-0.015,
)
print(f"INT4 compression: {c.compression_ratio:.2f}x ({c.compressed_size_gb:~P.1f})  accuracy_delta={c.estimated_accuracy_delta:+.1%}")

r = serving.solve(model, Hardware.Cloud.H100, seq_len=256, batch_size=32, precision="int4")
print(f"Single H100 (INT4, bs=32): TTFT={r.ttft:~P.1f}  ITL={r.itl:~P.1f}  feasible={r.feasible}")
```

**Step 2:** Calculate fleet size needed for 1000 QPS (accounting for both prefill `TTFT` and 128 decode steps per `bs=32` batch).

```python
out_tokens = 128
batch_time_s = r.ttft.to("s").magnitude + out_tokens * r.itl.to("s").magnitude
qps_per_gpu = 32.0 / batch_time_s if batch_time_s > 0 else 0.0
gpus_needed = 1000.0 / max(qps_per_gpu, 1e-6)
print(f"Batch service time (prefill + {out_tokens} decode steps): {batch_time_s:.2f} s  (~{qps_per_gpu:.2f} QPS/GPU)")
print(f"Minimum active GPUs for 1000 QPS: ~{gpus_needed:.0f} ({gpus_needed / 8:.1f} DGX H100 nodes)")
```

**Step 3:** Check budget and carbon for each region split.

```python
for us_pct in [0.3, 0.5, 0.7]:
    qc_pct = 1.0 - us_pct
    # 16 DGX H100 nodes = 128 GPUs (~116 min active GPUs + queueing/N+1 headroom)
    n_total = 16
    n_us = round(n_total * us_pct)
    n_qc = n_total - n_us

    fleet_us = Fleet(name="US East", node=Nodes.DGX_H100, count=max(1, n_us),
                     fabric=Fabrics.InfiniBand_NDR, region=Infrastructure.Grids.US_Avg)
    fleet_qc = Fleet(name="Quebec", node=Nodes.DGX_H100, count=max(1, n_qc),
                     fabric=Fabrics.InfiniBand_NDR, region=Infrastructure.Grids.Quebec)

    tco_us = econ.solve(fleet_us, duration_days=365, mfu=0.3)
    tco_qc = econ.solve(fleet_qc, duration_days=365, mfu=0.3)
    co2_us = sustain.solve(fleet_us, duration_days=365, mfu=0.3)
    co2_qc = sustain.solve(fleet_qc, duration_days=365, mfu=0.3)

    total_tco = tco_us.tco_usd + tco_qc.tco_usd
    total_co2 = (co2_us.carbon_footprint_kg + co2_qc.carbon_footprint_kg) / 1000  # tonnes
    print(f"Split {us_pct:.0%} US ({n_us}) / {qc_pct:.0%} QC ({n_qc}):  "
          f"TCO=${total_tco:,.0f}  CO2={total_co2:.1f}t")
```

### Question

What is your recommended fleet configuration? How many total GPUs,
what regional split, and does INT4 quantization change your answer?

### Hint

Llama-3-70B at FP16 requires `141.3 GB` (`> 85.9 GB` on one 80 GiB H100), so it does not fit on a single
GPU without tensor parallelism (`2+` GPUs per replica) or INT4 quantization (`~36 GB`, fits on one H100).
Remember to include **both** prompt prefill (`TTFT`) and `128`-token decode (`128 * ITL`) when turning per-batch latency into QPS/GPU.

<details>
<summary><strong>Expected Answer</strong></summary>

This is an open-ended design exercise. A strong answer includes:

1. **Precision choice:** FP16 requires `141.3 GB` (`feasible=False` on one `80 GiB` / `85.9 GB` H100, forcing `TP >= 2`). Grouped INT4 in `CompressionModel(..., group_size=128)` yields `3.88x` compression (`36.4 GB` weights including FP16 block scales), while `ServingModel(..., precision="int4")` models ideal 4-bit weights plus KV-cache (`35.3 GB + 0.67 GB = 36.0 GB` at `seq_len=256, bs=32`), fitting comfortably on a single H100 with `TTFT = 2,344.7 ms` (`2.34 s`) and `ITL = 10.7 ms` (`< 50 ms` SLA).

2. **Fleet size (prefill + decode):** At `seq_len=256, bs=32`, one batch takes `2.345 s` prefill + `128 * 0.01074 s = 1.374 s` decode = **`3.72 s` total** (`32 / 3.72 s = 8.60 QPS/GPU`; ignoring prefill would optimistically overpredict `23.3 QPS/GPU`). Serving `1,000 QPS` therefore requires at least **`~116` active GPUs** (`14.5` 8-GPU nodes). Provisioning **`16`–`18` DGX H100 nodes (`128`–`144` GPUs)** provides queueing headroom and `N+1` node redundancy.

3. **Regional split:** Placing `70%` of capacity in Quebec (`11` of `16` nodes on `20 gCO2/kWh` hydro) and `30%` in US East (`5` of `16` nodes on `US_Avg`) emits **`65.9 tonnes CO2/year`** (`67.0 tonnes/year` at `18` nodes) — over `7x` below the `500 tonnes/year` ceiling while maintaining dual-region availability.

4. **Budget check:** `16` DGX H100 nodes (`128` GPUs) carry an accelerator + energy + maintenance TCO of **`$1.28M/year`** (`$1.44M/year` at `18` nodes), or **`$2.10M/year`** (`$2.36M/year` at `18` nodes) with `infrastructure_multiplier=1.67` for full-system chassis/networking overhead — well under the `$5M` annual budget.

**Compression is an architectural decision, not only an optimization**, because fitting on a single GPU eliminates cross-GPU tensor-parallel synchronization and cuts fleet size in half.

</details>

### Discussion

What are the failure modes of this design? What happens when a full
DGX node goes down in the Quebec region? How does the ReliabilityModel
inform your redundancy strategy? Should you over-provision by N+1 or N+2?

---

## Textbook & Curriculum Cross-Links

To pair each exercise above with its corresponding *Machine Learning Systems* textbook chapter, interactive Marimo lab, and lecture slide deck, see the [Cross-Volume Physics & Chapter Map](https://mlsysbook.ai/mlsysim/curriculum-map.html).
