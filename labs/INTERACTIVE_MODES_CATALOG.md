# MLSysBook Interactive Modes & Tooling Catalog

> **Version:** 1.0.0
> **Target Environment:** Marimo Reactive Notebooks (`marimo >= 0.11.0`) & MLSysBook Web/WASM Runtime
> **Core Framework:** Python 3.11+, Plotly Graph Objects (`plotly.graph_objects`), `mlsysim`, `mlsysbook_labs`

---

## Table of Contents

1. **Executive Overview & Pedagogical Philosophy**
2. **Category 1: Input & Control Devices**
   - 1.1 Continuous Value Sliders
   - 1.2 Discrete Step & Exponential Sliders
   - 1.3 Dual-Thumb Range Sliders
   - 1.4 Radio Button Groups & Mutually Exclusive Pickers
   - 1.5 Dropdown Selectors
   - 1.6 Multi-Select Pills & Checkbox Groups
   - 1.7 Toggle Switches & Boolean Flags
   - 1.8 Action & Event Trigger Buttons
   - 1.9 Form Submitters & Batch Checkpoint Enclosures
   - 1.10 Numeric Precision Inputs
   - 1.11 Freeform Engineering Decision Logs
3. **Category 2: Analytical Visualizations & Trade-Off Surfaces**
   - 2.1 Dynamic Pareto Frontiers
   - 2.2 Williams Roofline Model Charts
   - 2.3 Waterfall Latency & Energy Decomposition Bars
   - 2.4 Radar / Spider Multi-Axis System Trade-Off Charts
   - 2.5 2D Feasibility & Contingency Heatmaps
   - 2.6 Cumulative Distribution Function (CDF) Tail-Latency Curves
   - 2.7 Interactive Telemetry DataFrames & Margin Tables
   - 2.8 Hardware Topology & Pipeline Execution Flow Diagrams
4. **Category 3: Pedagogical & Feedback Mechanisms**
   - 3.1 Prediction Locks with `mo.stop`
   - 3.2 Productive Failure Redouts & Visceral Alert Callouts
   - 3.3 Calculation Notes & MathPeek Physics Accordions
   - 3.4 Multi-Track Perspective Switchers
   - 3.5 Design Ledger State Persistence & Audit Logging
5. **Cross-Cutting Interaction Matrix**

---

## 1. Executive Overview & Pedagogical Philosophy

The MLSysBook interactive curriculum rejects passive notebook reading. Systems engineering cannot be mastered through static code cells or linear prose because **systems are defined by coupling, constraints, and non-linear multi-dimensional trade-off spaces**.

When an engineer scales batch size, latency does not simply drop; memory spills, cache lines thrash, and queueing delays spike at the 99th percentile. When precision is quantized from FP16 to INT4, memory bandwidth bottlenecks ease, but kernel dequantization overheads arise and catastrophic numerical divergence looms.

To teach this intuition, our Marimo labs adhere to four strict design tenets:
1. **Predict Before Observing (Gated Cognitive Friction):** Students must record a hypothesis *before* simulation instrumentation renders. Free-wheeling widget twiddling without prior mental commitment breeds false fluency.
2. **Visceral Productive Failure:** Hitting a hardware thermal ceiling, an Out-Of-Memory (OOM) panic, or an SLA deadline violation must trigger clear, unignorable feedback (shaking redouts, cliff graphs) that directly exposes the binding physical constraint.
3. **Multi-Perspective Grounding (The 4 Archetypes):** Every concept is evaluated through four distinct industry archetypes:
   - **Cloud Supercomputing:** Scale-out H100/H200 clusters, SLA cliffs, multi-tenant throughput, energy megawatts.
   - **Edge & Embodied AI:** NVIDIA Drive Orin / Jetson, sub-10ms hard real-time safety, rare-hazard perception.
   - **Mobile SoC:** Apple Silicon / Qualcomm Snapdragon NPU, unified memory, thermal throttling envelopes.
   - **TinyML & Microcontrollers:** ARM Cortex-M55 / ESP32-S3, strict <512 KB SRAM limits, micro-watt energy harvesting.
4. **Traceable Architectural Accounting:** Every design choice updates an immutable **Design Ledger**, teaching students to defend decisions with empirical telemetry and Iron Law accounting.

---

## 2. Category 1: Input & Control Devices

### 1.1 Continuous Value Sliders

#### Formal Name
`Continuous Value Slider`

#### Marimo / Plotly Construct
`marimo.ui.slider(start, stop, step, value, label, debounce=False)`

#### When to Use
Use when sweeping through a smooth physical continuum where marginal incremental changes reveal continuous trade-off dynamics (e.g., clock frequency throttling, continuous memory allocation percentage, data drift intensity, operational duty cycle).

#### Parameters & Configuration
- `start` (float): Lower bound of the physical sweep.
- `stop` (float): Upper bound of the physical sweep.
- `step` (float): Fine-grained increment (e.g., `0.1` or `1.0`).
- `value` (float): Default baseline setting.
- `label` (str): Descriptive label including physical units (e.g., `"Clock Frequency (GHz)"`, `"Data Drift Pressure (%)"`).
- `debounce` (bool): Set to `True` if triggering expensive multi-variable Monte Carlo simulations, or `False` for instant analytical formula updates.

#### Concrete MLSys Example
```python
import marimo as mo

# Continuous drift pressure knob affecting the verification gap and accuracy degradation
partA_drift = mo.ui.slider(
    start=0.0,
    stop=100.0,
    step=1.0,
    value=35.0,
    label="Production Drift Rate λ (%/month)",
)
```

---

### 1.2 Discrete Step & Exponential Sliders

#### Formal Name
`Discrete Step & Power-of-Two Slider`

#### Marimo / Plotly Construct
`marimo.ui.slider(steps=[...], value=..., label=...)`

#### When to Use
Use for hardware and architectural parameters that only physically exist in discrete powers of two, standard memory alignments, SIMD vector widths, or quantized bit-widths (e.g., batch size $B \in \{1, 2, 4, 8, 16, 32, 64, 128\}$, cache line widths, quantization precision levels).

#### Parameters & Configuration
- `steps` (list[Union[int, float]]): The exact valid physical discrete values.
- `value` (Union[int, float]): Initial default chosen from `steps`.
- `label` (str): Label indicating discrete systems parameter.

#### Concrete MLSys Example
```python
import marimo as mo

# Micro-batch size sweep: changes GEMM operational intensity and memory access patterns
batch_size_knob = mo.ui.slider(
    steps=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    value=16,
    label="Inference Micro-Batch Size (B)",
)
```

---

### 1.3 Dual-Thumb Range Sliders

#### Formal Name
`Dual-Thumb Range Slider`

#### Marimo / Plotly Construct
`marimo.ui.range_slider(start, stop, step, value=[low, high], label=...)`

#### When to Use
Use when defining operating windows, confidence intervals, filtering regions, percentile spans (e.g., $p50$ to $p99$ tail boundaries), or bounded resource partitioning between competing pipeline stages.

#### Parameters & Configuration
- `start` (float): Minimum global scale.
- `stop` (float): Maximum global scale.
- `step` (float): Granularity of adjustment.
- `value` (tuple/list): Initial `[lower_bound, upper_bound]`.
- `label` (str): Domain label.

#### Concrete MLSys Example
```python
import marimo as mo

# Dynamic speculative decoding window: minimum draft length vs maximum verification ceiling
speculative_window = mo.ui.range_slider(
    start=1,
    stop=16,
    step=1,
    value=[2, 6],
    label="Draft Token Speculative Window [K_min, K_max]",
)
```

---

### 1.4 Radio Button Groups & Mutually Exclusive Pickers

#### Formal Name
`Radio Button Group`

#### Marimo / Plotly Construct
`marimo.ui.radio(options={label: value}, value=..., label=...)`

#### When to Use
Use for discrete, mutually exclusive architectural choices or rigorous multiple-choice cognitive prediction gates where only one state can physically hold (e.g., numeric precision format, dominant bottleneck hypothesis, memory hierarchy placement tier).

#### Parameters & Configuration
- `options` (dict[str, Any] or list[str]): Dictionary mapping formatted human-readable labels to programmatic tokens.
- `value` (Any): Default selected option (or `None` for unforced initial commitment).
- `label` (str): The engineering prompt or hypothesis question.

#### Concrete MLSys Example
```python
import marimo as mo

# Memory allocation tier selection for KV cache offloading
kv_storage_tier = mo.ui.radio(
    options={
        "High-Bandwidth Memory (HBM3e) — 3.35 TB/s, $35/GB": "hbm",
        "Host DDR5 Memory (PCIe 5.0 x16) — 128 GB/s, $4/GB": "pcie_ddr5",
        "Local NVMe CXL SSD — 14 GB/s, $0.50/GB": "nvme_cxl",
    },
    value="hbm",
    label="KV Cache Physical Placement Tier",
)
```

---

### 1.5 Dropdown Selectors

#### Formal Name
`Dropdown Menu Picker`

#### Marimo / Plotly Construct
`marimo.ui.dropdown(options={label: value}, value=..., label=...)`

#### When to Use
Use when selecting from a catalog of distinct hardware accelerators, target models, benchmark datasets, or deployment tracks where screen real-estate must be preserved and choices exceed 3–4 items.

#### Parameters & Configuration
- `options` (dict[str, Any]): Label-to-value map.
- `value` (Any): Initial selection.
- `label` (str): Component label.

#### Concrete MLSys Example
```python
import marimo as mo

hardware_picker = mo.ui.dropdown(
    options={
        "Cloud: NVIDIA H100 SXM5 (80GB, 3350 GB/s, 989 TFLOPS FP16)": "h100",
        "Edge: NVIDIA Jetson AGX Orin (64GB, 204.8 GB/s, 170 TOPS INT8)": "orin",
        "Mobile: Apple M4 Neural Engine (16-core NPU, 38 TOPS, 120 GB/s)": "apple_m4",
        "TinyML: ARM Cortex-M55 (512KB SRAM, 150 MHz, 1.2 GOPS)": "cortex_m55",
    },
    value="h100",
    label="Deployment Hardware Target",
)
```

---

### 1.6 Multi-Select Pills & Checkbox Groups

#### Formal Name
`Multi-Select Token Pills / Multi-Checkbox Group`

#### Marimo / Plotly Construct
`marimo.ui.multiselect(options=[...], value=[...], label=...)`

#### When to Use
Use when composing non-exclusive optimization stacks, enabling concurrent compiler passes (e.g., FlashAttention + Weight Quantization + Operator Fusion), or selecting multi-metric telemetry overlays.

#### Parameters & Configuration
- `options` (list[str] or dict[str, Any]): Available optimization passes.
- `value` (list[str]): Currently activated passes.
- `label` (str): Header prompt.

#### Concrete MLSys Example
```python
import marimo as mo

compiler_passes = mo.ui.multiselect(
    options=[
        "Kernel Fusion (Vertical + Horizontal)",
        "FlashAttention-2 Tiling",
        "FP8 KV-Cache Compression",
        "Activation Checkpointing",
        "Zero-Bubble Pipeline Scheduling",
    ],
    value=["Kernel Fusion (Vertical + Horizontal)", "FlashAttention-2 Tiling"],
    label="Active Kernel & Memory Optimization Passes",
)
```

---

### 1.7 Toggle Switches & Boolean Flags

#### Formal Name
`Toggle Switch / Boolean Checkbox`

#### Marimo / Plotly Construct
`marimo.ui.switch(value=False, label=...)` or `marimo.ui.checkbox(value=False, label=...)`

#### When to Use
Use for binary operational modes, enabling/disabling safety guardrails, turning on fallback caches, toggling asynchronous double-buffering, or enabling trace profiling.

#### Parameters & Configuration
- `value` (bool): Initial state (`True`/`False`).
- `label` (str): Name of the feature flag.

#### Concrete MLSys Example
```python
import marimo as mo

double_buffering_toggle = mo.ui.switch(
    value=True,
    label="Enable Async Double-Buffering (Overlap DtoH Copy with Tensor Core Compute)",
)
```

---

### 1.8 Action & Event Trigger Buttons

#### Formal Name
`Action / Event Trigger Button`

#### Marimo / Plotly Construct
`marimo.ui.button(value=0, label=..., kind="primary"|"warn"|"neutral")`

#### When to Use
Use to trigger discrete non-continuous actions: stepping a clock cycle in a pipeline simulator, injecting an adversarial noise burst, running a cache warm-up sequence, or clearing a faulted buffer.

#### Parameters & Configuration
- `value` (int): Increments on every user click.
- `label` (str): Action text (e.g., `"Inject Burst Traffic Spike (10x Load)"`).
- `kind` (str): Styling semantic (`"primary"`, `"warn"`, `"danger"`).

#### Concrete MLSys Example
```python
import marimo as mo

inject_fault_button = mo.ui.button(
    label="⚡ Inject Sensor Glitch & Temperature Drift Event",
    kind="warn",
)
```

---

### 1.9 Form Submitters & Batch Checkpoint Enclosures

#### Formal Name
`Form Submitter / Gated Action Box`

#### Marimo / Plotly Construct
`marimo.ui.form(element=..., submit_button_label=...)` or `mlsysbook_labs.action_box(element, title=..., body=...)`

#### When to Use
Use when several inter-dependent knobs must be adjusted simultaneously without re-evaluating the entire expensive computational graph on every micro-slider adjustment, or when creating a formal **Checkpoint Gate** where the student legally signs off on a configuration.

#### Parameters & Configuration
- `element` (marimo UI element or layout): Child widget(s).
- `submit_button_label` (str): Confirmation button text.

#### Concrete MLSys Example
```python
import marimo as mo
from mlsysbook_labs import action_box

partB_decision = action_box(
    mo.ui.radio(
        options={"Data": "Data", "Algorithm": "Algorithm", "Machine": "Machine"},
        label="Select the binding axis to address:",
    ),
    title="Part B Checkpoint — Commit First Engineering Fix",
    body="Review your evidence table. Committing will lock your diagnosis into the lab audit ledger.",
    name="binding_axis_commit",
)
```

---

### 1.10 Numeric Precision Inputs

#### Formal Name
`Numeric Precision Input`

#### Marimo / Plotly Construct
`marimo.ui.number(start=..., stop=..., step=..., value=..., label=...)`

#### When to Use
Use when exact integer or floating point parameters are needed that are clumsy on a slider (e.g., entering exact memory budget in MB, exact target P99 latency threshold in milliseconds, or parameter count in billions).

#### Parameters & Configuration
- `start`, `stop`, `step`: Numeric constraints.
- `value`: Initial default.
- `label`: Name with physical unit.

#### Concrete MLSys Example
```python
import marimo as mo

sla_budget_input = mo.ui.number(
    start=1.0,
    stop=500.0,
    step=0.5,
    value=25.0,
    label="P99 SLA Latency Budget (ms)",
)
```

---

### 1.11 Freeform Engineering Decision Logs

#### Formal Name
`Engineering Decision Log`

#### Marimo / Plotly Construct
`mo.ui.text_area(label=..., placeholder=..., full_width=True)` or `mlsysim.labs.components.DecisionLog()`

#### When to Use
Use in the final Synthesis section of every lab. Students must explain *why* their configuration succeeded using Iron Law terms, citing specific numbers from their instruments and acknowledging remaining trade-offs.

#### Parameters & Configuration
- `placeholder` (str): Guiding prompt enforcing quantitative justification.
- `full_width` (bool): `True` for ergonomic typing.

#### Concrete MLSys Example
```python
import marimo as mo

decision_log = mo.ui.text_area(
    label="Architectural Decision Log (Required for Ledger Signoff):",
    placeholder="I chose FP8 quantization with a batch size of 32 because the workload shifted from memory-bound (AI = 42 FLOP/B) to compute-bound on the H100 SXM5, dropping latency by 2.4x while maintaining 99.2% accuracy...",
    full_width=True,
)
```

---

## 3. Category 2: Analytical Visualizations & Trade-Off Surfaces

### 2.1 Dynamic Pareto Frontiers

#### Formal Name
`Dynamic Multi-Objective Pareto Frontier`

#### Plotly Construct
`go.Scatter(x=..., y=..., mode='markers+lines')` with convex-hull or non-dominated sorting logic.

#### When to Use
Use to demonstrate that in real-world ML systems, **there are no unilateral optimizations—only trade-offs along an optimal boundary**. Any point below/behind the frontier is sub-optimal; any point beyond it is physically impossible under current technology constraints.

#### Key Parameters & Visual Encodings
- **X-Axis:** Cost Metric (Latency in ms, Energy in Joules, Dollar Cost $/1M tokens, or SRAM footprint in KB).
- **Y-Axis:** Quality Metric (Top-1 Accuracy %, BLEU/ROUGE score, Safety Margin %).
- **Marker Color/Size:** Secondary trade-off (e.g., Model Parameter Scale or Peak Thermal Dissipation).
- **Frontier Line:** Step-wise or convex hull connecting non-dominated operating configurations.

#### Concrete MLSys Example
```python
import plotly.graph_objects as go
import numpy as np

def build_pareto_frontier(models, active_point):
    # Sort models by latency
    sorted_models = sorted(models, key=lambda m: m['latency_ms'])
    pareto_x, pareto_y = [], []
    max_acc = -1.0
    for m in sorted_models:
        if m['accuracy_pct'] > max_acc:
            pareto_x.append(m['latency_ms'])
            pareto_y.append(m['accuracy_pct'])
            max_acc = m['accuracy_pct']

    fig = go.Figure()
    # Non-dominated frontier
    fig.add_trace(go.Scatter(
        x=pareto_x, y=pareto_y,
        mode='lines',
        name='Pareto Frontier',
        line=dict(color='#A51C30', width=2, dash='dash')
    ))
    # All candidate architectural designs
    fig.add_trace(go.Scatter(
        x=[m['latency_ms'] for m in models],
        y=[m['accuracy_pct'] for m in models],
        mode='markers+text',
        text=[m['name'] for m in models],
        textposition='top right',
        marker=dict(size=10, color='#1F407A', opacity=0.7),
        name='Architectural Candidates'
    ))
    # Currently evaluated student design
    fig.add_trace(go.Scatter(
        x=[active_point['latency_ms']],
        y=[active_point['accuracy_pct']],
        mode='markers',
        marker=dict(size=16, color='#247A4D', symbol='star', line=dict(width=2, color='white')),
        name='Your Active Design'
    ))
    fig.update_layout(
        title="Accuracy vs. Latency Pareto Surface",
        xaxis=dict(title="P99 Inference Latency (ms) [Lower is Better]"),
        yaxis=dict(title="Downstream Task Accuracy (%) [Higher is Better]"),
        template="plotly_white",
        height=380,
    )
    return fig
```

---

### 2.2 Williams Roofline Model Charts

#### Formal Name
`Williams Operational Roofline Model`

#### Plotly Construct
`go.Scatter(..., x=..., y=..., mode='lines')` on dual logarithmic axes ($\log_{10} - \log_{10}$) with dynamic design point diamonds.

#### When to Use
The foundational visualization for hardware-software co-design (Labs 02, 05, 10, 11). Quantifies whether a workload is **Memory-Bandwidth Bound** or **Peak-Compute Bound** based on its Arithmetic Intensity ($I = \text{FLOPs} / \text{Byte}$).

#### Key Parameters & Visual Encodings
- **X-Axis:** Arithmetic Intensity ($\text{FLOPs/Byte}$, log scale from $10^{-1}$ to $10^4$).
- **Y-Axis:** Attainable Performance ($\text{TFLOPs/s}$ or $\text{GFLOPs/s}$, log scale).
- **Ridge Point ($I^* = R_{\text{peak}} / \text{BW}$):** The transition point between memory-bound sloped line ($P = I \times \text{BW}$) and horizontal ceiling ($P = R_{\text{peak}}$).
- **Design Point:** A diamond marker showing the workload's current intensity and attainable throughput.

#### Concrete MLSys Example
```python
import plotly.graph_objects as go
import numpy as np

def build_roofline_chart(peak_tflops: float, bandwidth_gbs: float, workload_ai: float):
    # Ridge point
    ridge_ai = (peak_tflops * 1e3) / bandwidth_gbs

    x = np.logspace(-1, 4, 200)
    # Roofline equation: P_attainable = min(peak, AI * BW)
    y_roof = np.minimum(peak_tflops * 1e3, x * bandwidth_gbs) / 1e3 # in TFLOPs/s

    # Attainable performance for active workload
    workload_perf = min(peak_tflops * 1e3, workload_ai * bandwidth_gbs) / 1e3
    regime = "Compute-Bound" if workload_ai >= ridge_ai else "Memory-Bound"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_roof,
        mode='lines',
        name='Hardware Ceiling',
        line=dict(color='#64748B', width=3),
        fill='tozeroy',
        fillcolor='rgba(241, 245, 249, 0.5)'
    ))
    fig.add_trace(go.Scatter(
        x=[workload_ai], y=[workload_perf],
        mode='markers+text',
        marker=dict(size=14, color='#A51C30' if regime == "Memory-Bound" else '#1F407A', symbol='diamond'),
        text=[f"Design Point ({regime})"],
        textposition="top center",
        name='Workload'
    ))
    fig.add_vline(x=ridge_ai, line_dash="dot", line_color="#94A3B8", annotation_text=f"Ridge ({ridge_ai:.1f} FLOP/B)")
    fig.update_layout(
        xaxis=dict(type='log', title="Arithmetic Intensity (FLOPs/Byte)"),
        yaxis=dict(type='log', title="Attainable Performance (TFLOPs/s)"),
        title=f"Roofline Analysis: {regime} (Ridge = {ridge_ai:.1f} FLOP/B)",
        height=350,
        template="plotly_white",
    )
    return fig
```

---

### 2.3 Waterfall Latency & Energy Decomposition Bars

#### Formal Name
`Waterfall Latency & Energy Decomposition Stack`

#### Plotly Construct
`go.Bar` with overlay or stacked modes, or `go.Waterfall`.

#### When to Use
Essential for teaching **The Iron Law of ML Systems** ($T = T_{\text{mem}} + T_{\text{comp}} + L_{\text{lat}}$) and **Amdahl's Law**. Shows why accelerating compute by $10\times$ yields only negligible speedup if memory movement or un-fused kernel launches dominate runtime.

#### Key Parameters & Visual Encodings
- **Stages:** Pre-processing/Tokenization, Memory Transfer ($D_{\text{vol}} / \text{BW}$), Tensor Core Compute ($O / (R \cdot \eta)$), Kernel Launch & Pipeline Overhead ($L$).
- **Ghost Bar:** Gray dashed outline of previous un-optimized baseline to immediately reveal deltas.

#### Concrete MLSys Example
```python
import plotly.graph_objects as go

def build_latency_waterfall(comp_ms, mem_ms, ovh_ms, prev_breakdown=None):
    stages = ['Compute (GEMM)', 'Memory Traffic (HBM/SRAM)', 'Kernel Launch Overhead']
    current_vals = [comp_ms, mem_ms, ovh_ms]

    fig = go.Figure()
    if prev_breakdown:
        fig.add_trace(go.Bar(
            x=stages, y=prev_breakdown,
            name='Baseline (Pre-Opt)',
            marker=dict(color='rgba(148, 163, 184, 0.3)', line=dict(color='#94A3B8', width=1.5, dash='dash')),
            width=0.5
        ))
    fig.add_trace(go.Bar(
        x=stages, y=current_vals,
        name='Optimized System',
        marker=dict(color=['#1F407A', '#A51C30', '#D97706']),
        width=0.35
    ))
    fig.update_layout(
        title="Iron Law Latency Decomposition Breakdown",
        yaxis=dict(title="Execution Time (ms)"),
        barmode='overlay',
        template="plotly_white",
        height=320,
    )
    return fig
```

---

### 2.4 Radar / Spider Multi-Axis System Trade-Off Charts

#### Formal Name
`Radar / Spider Multi-Axis System Trade-Off Chart`

#### Plotly Construct
`go.Scatterpolar(r=..., theta=..., fill='toself')`

#### When to Use
Use when evaluating holistically across conflicting system properties: Compute Efficiency, Memory Footprint Margin, Latency Margin, Downstream Accuracy, Energy Efficiency, and Engineering Dollar Cost.

#### Key Parameters & Visual Encodings
- **Radial Coordinates:** Normalized scores ($0\%$ to $100\%$, where $100\%$ represents meeting or exceeding the budget).
- **Threshold Ring:** A dashed circle at $70\%$ or $100\%$ representing the mandatory production feasibility boundary.

#### Concrete MLSys Example
```python
import plotly.graph_objects as go

def build_dam_radar(data_score: float, algo_score: float, machine_score: float, threshold: float = 70.0):
    categories = ['Data Freshness/Coverage', 'Algorithmic Capacity', 'Machine Throughput/HW Envelope']
    # Close the loop
    r_vals = [data_score, algo_score, machine_score, data_score]
    theta_vals = categories + [categories[0]]
    threshold_vals = [threshold, threshold, threshold, threshold]

    fig = go.Figure()
    # Required threshold perimeter
    fig.add_trace(go.Scatterpolar(
        r=threshold_vals, theta=theta_vals,
        mode='lines',
        name='Feasibility Floor',
        line=dict(color='#DC2626', dash='dash', width=2),
    ))
    # Active system profile
    fig.add_trace(go.Scatterpolar(
        r=r_vals, theta=theta_vals,
        fill='toself',
        name='System Readiness',
        fillcolor='rgba(31, 64, 122, 0.25)',
        line=dict(color='#1F407A', width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="D·A·M Triad System Diagnostic Radar",
        height=350,
        template="plotly_white",
    )
    return fig
```

---

### 2.5 2D Feasibility & Contingency Heatmaps

#### Formal Name
`2D Design-Space Sweep Heatmap`

#### Plotly Construct
`go.Heatmap(z=..., x=..., y=..., colorscale=...)`

#### When to Use
Use when exploring the joint interaction of two critical knobs (e.g., Batch Size vs. Sequence Length, or Quantization Bitwidth vs. KV Cache Size). Immediately highlights the **Feasibility Cliff** (OOM faults or SLA misses).

#### Key Parameters & Visual Encodings
- **X-Axis:** First control variable.
- **Y-Axis:** Second control variable.
- **Z-Axis (Color):** Latency (ms), Throughput (tok/s), or Memory (GB). Cells violating constraints are highlighted with a distinct hatching or red palette.

#### Concrete MLSys Example
```python
import plotly.graph_objects as go
import numpy as np

def build_feasibility_heatmap(batch_sizes, precisions, memory_limit_gb=80.0):
    # Mock computation of memory allocation
    z_mem = []
    text_labels = []
    for p in precisions:
        row = []
        label_row = []
        b_bytes = 2 if p == "FP16" else (1 if p == "INT8" else 0.5)
        for b in batch_sizes:
            mem = 14.0 + (b * 2048 * 4096 * 32 * 2 * b_bytes) / 1e9 # weights + KV cache
            row.append(mem)
            label_row.append(f"{mem:.1f} GB<br>({'OOM' if mem > memory_limit_gb else 'OK'})")
        z_mem.append(row)
        text_labels.append(label_row)

    fig = go.Figure(data=go.Heatmap(
        z=z_mem,
        x=[f"B={b}" for b in batch_sizes],
        y=precisions,
        text=text_labels,
        texttemplate="%{text}",
        colorscale=[[0, '#10B981'], [0.75, '#F59E0B'], [1.0, '#EF4444']],
        colorbar=dict(title="VRAM (GB)"),
    ))
    fig.update_layout(
        title=f"VRAM Footprint Surface (Hard Limit: {memory_limit_gb} GB)",
        xaxis_title="Batch Size",
        yaxis_title="Numeric Precision",
        height=340,
        template="plotly_white",
    )
    return fig
```

---

### 2.6 Cumulative Distribution Function (CDF) Tail-Latency Curves

#### Formal Name
`Empirical CDF & Tail-Latency Percentile Curve`

#### Plotly Construct
`go.Scatter(x=..., y=..., mode='lines')` plotting sorted query execution latencies against percentiles ($0\%$ to $99.9\%$).

#### When to Use
Vital for Serving and MLOps labs (Labs 12, 13, 14). Convinces students that **mean latency ($\mu$) is an engineering lie**—in real-world distributed inference, multi-tenant contention, queueing bursts, and garbage collection produce severe $p99$ and $p99.9$ tail spikes that breach service SLAs.

#### Key Parameters & Visual Encodings
- **X-Axis:** Latency in milliseconds.
- **Y-Axis:** Percentile of requests ($0.50$ to $0.999$, often logit or non-linear scale).
- **SLA Wall:** Vertical dashed line indicating the maximum allowable client deadline.

#### Concrete MLSys Example
```python
import plotly.graph_objects as go
import numpy as np

def build_tail_cdf(latencies_ms, sla_budget_ms=25.0):
    sorted_lat = np.sort(latencies_ms)
    p = 100.0 * np.arange(len(sorted_lat)) / (len(sorted_lat) - 1)

    p50 = np.percentile(sorted_lat, 50)
    p95 = np.percentile(sorted_lat, 95)
    p99 = np.percentile(sorted_lat, 99)
    sla_violation_pct = np.mean(sorted_lat > sla_budget_ms) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_lat, y=p, mode='lines', name='Empirical CDF', line=dict(color='#1F407A', width=2.5)))
    fig.add_vline(x=sla_budget_ms, line_dash="dash", line_color="#DC2626", annotation_text=f"SLA Ceiling ({sla_budget_ms}ms)")
    fig.add_trace(go.Scatter(
        x=[p50, p95, p99], y=[50, 95, 99],
        mode='markers+text',
        text=[f"p50: {p50:.1f}ms", f"p95: {p95:.1f}ms", f"p99: {p99:.1f}ms"],
        textposition="top left",
        marker=dict(size=9, color=['#10B981', '#F59E0B', '#EF4444']),
        name='Percentiles'
    ))
    fig.update_layout(
        title=f"Latency Distribution (Tail Violation Rate: {sla_violation_pct:.2f}%)",
        xaxis_title="Inference Latency (ms)",
        yaxis_title="CDF Percentile (%)",
        height=340,
        template="plotly_white",
    )
    return fig
```

---

### 2.7 Interactive Telemetry DataFrames & Margin Tables

#### Formal Name
`Interactive Telemetry Margin Table`

#### Marimo Construct
`marimo.ui.table(data=..., selection=...)` or custom HTML margin components via `mlsysbook_labs.ui.checkpoint_card`.

#### When to Use
Use to compare system metrics across configurations or deployment tracks side-by-side, explicitly computing the **Safety Margin** ($\text{Margin} = \text{Achieved} - \text{Threshold}$).

#### Concrete MLSys Example
```python
import marimo as mo

def build_margin_table(data_margin, algo_margin, machine_margin):
    rows = [
        {"Axis": "Data", "Score": "62%", "Threshold": "70%", "Margin": f"{data_margin:+.1f} pp", "Status": "FAIL ❌" if data_margin < 0 else "PASS ✅"},
        {"Axis": "Algorithm", "Score": "78%", "Threshold": "70%", "Margin": f"{algo_margin:+.1f} pp", "Status": "PASS ✅"},
        {"Axis": "Machine", "Score": "85%", "Threshold": "70%", "Margin": f"{machine_margin:+.1f} pp", "Status": "PASS ✅"},
    ]
    return mo.ui.table(rows, label="D·A·M Triad Axis Margins")
```

---

### 2.8 Hardware Topology & Pipeline Execution Flow Diagrams

#### Formal Name
`Hardware Topology & Pipeline Flow Diagram`

#### Plotly / HTML-SVG / Mermaid Construct
`mo.mermaid(...)` or custom SVG layout rendering interconnects, memory hierarchies, and pipeline bubbles.

#### When to Use
Use when teaching tensor parallelism (TP), pipeline parallelism (PP), ring-allreduce network saturation, or cache hierarchy data movement (Registers $\to$ L1 $\to$ L2 $\to$ HBM).

#### Concrete MLSys Example
```python
import marimo as mo

def render_pp_pipeline(num_stages=4, num_microbatches=8):
    # Generates 1F1B (One-Forward-One-Backward) schedule diagram
    mermaid_code = """
    gantt
        title 1F1B Pipeline Parallel Schedule
        dateFormat X
        axisFormat %s
        section Stage 0 (GPU 0)
        F0 :active, 0, 1
        F1 :active, 1, 2
        F2 :active, 2, 3
        F3 :active, 3, 4
        B0 :crit, 4, 5
        F4 :active, 5, 6
        B1 :crit, 6, 7
        section Stage 1 (GPU 1)
        F0 :active, 1, 2
        F1 :active, 2, 3
        B0 :crit, 5, 6
    """
    return mo.mermaid(mermaid_code)
```

---

## 4. Category 3: Pedagogical & Feedback Mechanisms

### 4.1 Prediction Locks with `mo.stop`

#### Formal Name
`Structured Prediction Gate with Dataflow Halt`

#### Marimo Construct
`mlsysim.labs.components.PredictionLock` or `mlsysbook_labs.gated_hypothesis_card` paired with `mo.stop(prediction.value is None, ...)`

#### When to Use
Mandatory entry gate for all lab Parts A, B, C, and D. Prevents any instruments, charts, or sliders from rendering until the student formally registers an engineering hypothesis.

#### Parameters & Configuration
- `radio_element` (mo.ui.radio): The hypothesis options.
- `mo.stop(condition, message)`: Marimo primitive that completely halts execution of downstream cells in the reactive graph.

#### Concrete MLSys Example
```python
import marimo as mo
from mlsysbook_labs import gated_hypothesis_card

# Cell 1: Define Prediction Gate
partA_pred = mo.ui.radio(
    options={
        "A) Memory bandwidth saturation (D_vol / BW)": "mem_bw",
        "B) Arithmetic functional unit saturation (O / R_peak)": "compute_peak",
        "C) Host PCIe transfer bottleneck": "pcie_bottleneck",
    },
    label="Which physical constraint will bind first when micro-batch size scales from 1 to 64?",
)
hypothesis_card = gated_hypothesis_card(partA_pred, title="Task 1: Hypothesis Gate")

# Cell 2: Enforce the Gate
mo.stop(
    partA_pred.value is None,
    mo.callout(mo.md("⚠️ **Simulation Locked**: Select your hypothesis above to initialize the telemetry instruments."), kind="warn")
)
```

---

### 4.2 Productive Failure Redouts & Visceral Alert Callouts

#### Formal Name
`Productive Failure Visceral Banner`

#### Marimo Construct
`mlsysim.labs.components.FailureBanner(condition, message, animation_class="shake-hard")` or `mo.callout(..., kind="danger")`

#### When to Use
Use whenever a student's knob adjustment crosses a hard physical boundary:
- SRAM Out-Of-Memory ($M_{\text{alloc}} > M_{\text{SRAM}}$)
- Thermal Junction Trip ($T_j > 105^\circ\text{C}$)
- Hard SLA Deadline Miss ($p99 > L_{\text{budget}}$)
- Verification Gap Accuracy Collapse ($A(t) < A_{\text{floor}}$)

#### Parameters & Configuration
- `condition` (bool): Evaluates whether a constraint is breached.
- `message` (str): Plain-spoken engineering failure explanation.
- `animation_class` (str): CSS animation triggering a visual pulse or shake.

#### Concrete MLSys Example
```python
from mlsysim.labs.components import FailureBanner

# Rendered reactively inside telemetry stack
failure_widget = FailureBanner(
    condition=vram_allocated_gb > 80.0,
    message=f"CUDA OUT OF MEMORY: Attempted to allocate {vram_allocated_gb:.2f} GB on NVIDIA H100 SXM5 (Capacity: 80.0 GB). Process terminated.",
    animation_class="shake-hard"
)
```

---

### 4.3 Calculation Notes & MathPeek Physics Accordions

#### Formal Name
`MathPeek Physics & Invariant Accordion`

#### Marimo Construct
`mlsysbook_labs.legacy_components.MathPeek(formula, variables)` or `mo.accordion({"Math Peek": ...})`

#### When to Use
Placed beneath every telemetry instrument to link empirical widget readings directly to textbook equations. Ensures students connect visual observations to first-principles mathematics without cluttering the primary UI.

#### Parameters & Configuration
- `formula` (str): LaTeX mathematical formulation.
- `variables` (dict[str, str]): Exact mapping of mathematical symbols to physical units and current simulated values.

#### Concrete MLSys Example
```python
from mlsysim.labs.components import MathPeek

math_peek = MathPeek(
    formula=r"T_{\text{step}} = \frac{D_{\text{vol}}}{\text{BW}_{\text{eff}}} + \frac{O_{\text{FLOPs}}}{R_{\text{peak}} \cdot \eta_{\text{hw}}} + L_{\text{lat}}",
    variables={
        "D_vol": f"{bytes_moved_gb:.2f} GB (Weights + KV Cache Activations)",
        "BW_eff": "3,350 GB/s (HBM3e Sustained Interconnect)",
        "O_FLOPs": f"{total_flops / 1e12:.2f} TFLOPs",
        "R_peak * eta": "989 TFLOPs/s * 0.62 MFU = 613.18 Effective TFLOPs/s",
        "L_lat": "12.5 µs (CUDA Kernel Launch & Synchronization Overhead)",
    }
)
```

---

### 4.4 Multi-Track Perspective Switchers

#### Formal Name
`Universal Track Selector & Cross-Tier Hardware Lens`

#### Marimo Construct
`mlsysbook_labs.ui.track_selector()` or `v1_01_track_picker` dropdown wired into `get_track_profile()` and `resolve_mlsysim_ref()`.

#### When to Use
Positioned at the top of every lab. Dynamically swaps the entire laboratory context:
- Stakeholder persona & quote
- Underlying hardware target (H100 vs. Orin vs. Apple M4 vs. ESP32-S3)
- Physical thresholds & failure modes
- Primary objective vs. secondary guardrail

#### Concrete MLSys Example
```python
import marimo as mo
from mlsysbook_labs import get_track_profile, track_context

track_picker = mo.ui.dropdown(
    options={
        "☁️ Cloud Supercomputing Track (H100 & Cluster Ingestion vs SLA Walls)": "cloud_fleet",
        "🤖 Edge & Embodied Track (Jetson Orin & Safety Latency vs Perception)": "robotaxi",
        "📱 Mobile Track (Apple Silicon & Neural Engine vs Thermal Envelope)": "iphone",
        "⚡ TinyML Track (ESP32-S3 & Bio-Sensing vs 512KB SRAM Wall)": "oura_ring",
    },
    value="☁️ Cloud Supercomputing Track (H100 & Cluster Ingestion vs SLA Walls)",
    label="Select Course / Industry Track",
)
profile = get_track_profile(track_picker.value)
banner = track_context(profile)
```

---

### 4.5 Design Ledger State Persistence & Audit Logging

#### Formal Name
`Design Ledger & WASM Local-Storage State Store`

#### Construct
`mlsysim.labs.state.DesignLedger`

#### When to Use
Used across all lab checkpoints and synthesis cells. Records student decisions, hypotheses, and telemetry numbers into persistent browser storage (`localStorage` in WASM or JSON file in native Python), allowing cumulative progress tracking across the 16 labs and generating graded lab reports.

#### Methods & API
- `ledger.save_decision(lab_id, step, decision_dict)`: Commits a decision.
- `ledger.get_decision(lab_id, step)`: Retrieves recorded entry.
- `ledger.export_markdown_report()`: Generates full traceable engineering audit report.

#### Concrete MLSys Example
```python
from mlsysim.labs.state import DesignLedger

ledger = DesignLedger()

# Committing final design decision
ledger.save_decision(
    lab_id="lab_01",
    step="synthesis",
    decision={
        "track_id": "cloud_fleet",
        "binding_axis_diagnosed": "Data",
        "first_intervention": "Automated Freshness Canary Pipeline",
        "rejected_alternative": "Purchasing 4x more H100 GPU nodes",
        "iron_law_justification": "Training accuracy degradation was driven by covariate shift (lambda = 0.42), which additional compute FLOPs cannot resolve.",
    }
)
```

---

## 5. Cross-Cutting Interaction Matrix

The following matrix guides lab designers on matching pedagogical goals to specific interactive devices:

| Cognitive Objective | Recommended Input Device | Recommended Analytical Visualizer | Feedback / Invariant Device |
|:---|:---|:---|:---|
| **Expose Latency Bottleneck** | Discrete Batch Slider (`1.2`) | Roofline Chart (`2.2`) & Latency Waterfall (`2.3`) | MathPeek (`3.3`) on Iron Law |
| **Quantify Verification Limits**| Drift & Timeline Sliders (`1.1`) | Degradation Curve vs Floor (`2.1`) | Productive Failure Banner (`3.2`) |
| **Balance Multi-Axis Constraints**| Triple-Budget Sliders (`1.1`) | Radar Chart (`2.4`) & Margin Table (`2.7`) | D·A·M Diagnosis Callout (`3.2`) |
| **Explore Pipeline Scalability** | Pipeline Stage Number Input (`1.10`)| Bubble Gantt Schedule (`2.8`) | Amdahl's Law Accordion (`3.3`) |
| **Tune Tail Latency & SLA** | Request Arrival Rate Slider (`1.1`) | Empirical CDF Percentile Curve (`2.6`) | SLA Redout Violation (`3.2`) |
| **Evaluate Memory Cliff** | Sequence Length Slider (`1.1`) | 2D Feasibility Heatmap (`2.5`) | CUDA OOM Panic Banner (`3.2`) |
| **Commit Synthesis Decision** | Form Submitter / Action Box (`1.9`)| Pareto Optimal Scatter (`2.1`) | Design Ledger Export (`3.5`) |
