# Visual Architecture Debugging

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <b>📱 Mobile</b> · <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

*Can you spot the bottleneck in a mobile system diagram?*

Mobile system architecture diagrams with hidden bottlenecks.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/04_visual_debugging.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Launch Blocker</b> · <code>serving</code> <code>ux</code></summary>

### Synchronous Download Blocks the Critical Path

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef safe fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5
    classDef process fill:#f3f4f6,stroke:#4b5563,stroke-width:2px,color:#1f2937

    T1["Launch: App UI Loaded"]:::safe
    T2["Step 1: Download Model<br/>(Wait: 60s)"]:::danger
    T3["Step 2: Validate Checksum<br/>(Wait: 2s)"]:::process
    T4["Step 3: Compile for NPU<br/>(Wait: 8s)"]:::process
    T5["READY: User can Translate"]:::safe

    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5

    Note["🚨 CRITICAL FRICTION<br/>Total blocked time: 70 seconds<br/>Status: 95% User Abandonment"]:::danger
    T2 --- Note
```

- **Interviewer:** "A user downloads your new LLM-powered translation app. Upon first launch, the app shows a spinner for 70 seconds while it prepares the model. Based on the launch timeline, why are you seeing a 95% abandonment rate?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Show a progress bar instead of a spinner." A progress bar is better UX, but 70 seconds is too long regardless of visual feedback.

  **Realistic Solution:** The architecture has a **Synchronous Dependency on the Critical Path**. The app forces users to wait for sequential download, validation, and compilation before any functionality is available. The fix is a **Progressive Launch Architecture**: ship a tiny fallback model (~3 MB) in the app bundle for instant value, while downloading and compiling the high-quality model in the background.

  > **Napkin Math:** User attention span for a new app launch is ~2-5 seconds. A 70-second block is 14-35x longer than the tolerance threshold of a typical mobile user.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Operator Gap</b> · <code>frameworks</code> <code>latency</code></summary>

### NPU-CPU Ping-Pong Creates Pipeline Bubbles

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph LR
    classDef npu fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef cpu fill:#f3f4f6,stroke:#4b5563,stroke-width:2px,color:#1f2937
    classDef bounce fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    NPU1["NPU Segment 1<br/>(8 Conv Blocks)"]:::npu
    DMA1["DMA Bounce<br/>(Unsupported GELU)"]:::bounce
    NPU2["NPU Segment 2<br/>(4 Conv Blocks)"]:::npu
    DMA2["DMA Bounce<br/>(Dynamic Shape)"]:::bounce
    NPU3["NPU Segment 3<br/>(4 Conv Blocks)"]:::npu

    NPU1 -->|1.2ms| DMA1
    DMA1 -->|1.2ms| NPU2
    NPU2 -->|1.2ms| DMA2
    DMA2 -->|1.2ms| NPU3

    Note["🚨 HIDDEN OVERHEAD<br/>4.8ms spent in DMA<br/>32% of total latency"]:::bounce
```

- **Interviewer:** "You are deploying a model to a mobile NPU. The math should only take 10ms, but the actual inference latency is 15ms. You find two unsupported operators in your graph. Based on the DMA flow, why are these two ops causing a 50% latency penalty?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The NPU is slow — use a bigger model." The NPU compute is fine. The overhead is in the data movement.

  **Realistic Solution:** You are hitting **Graph Partitioning Overhead**. Each unsupported operator forces the graph to 'ping-pong' data from the NPU to the CPU and back via DMA. This incurs a latency tax for the transfer and leaves the NPU idle while the CPU processes the single op. The fix is to eliminate partition boundaries by replacing unsupported ops with approximations (e.g., Sigmoid approximation for GELU) or static-shape equivalents.

  > **Napkin Math:** Each NPU-CPU-NPU round-trip costs ~2.4ms in DMA overhead. With 2 such bounces, you spend 4.8ms just moving data—roughly 32% of your 15ms total budget—without doing any significant math.

  📖 **Deep Dive:** [ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Jetsam Guillotine</b> · <code>memory</code> <code>reliability</code></summary>

### ML Model + Camera ISP Compete for the Same RAM Pool

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef app fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef sys fill:#f3f4f6,stroke:#4b5563,stroke-width:2px,color:#1f2937
    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    subgraph "Physical RAM (4 GB Total)"
        direction TB
        iOS["iOS + System Services<br/>(1.5 GB)"]:::sys
        ISP["Camera ISP Service<br/>(1.2 GB)"]:::sys
        App["Your App<br/>(822 MB)"]:::app
        OOM["OUT OF MEMORY<br/>(Demand: 4.7 GB)"]:::danger
    end

    App -- "Allocation" --> OOM
    ISP -- "Allocation" --> OOM
    OOM -- "SIGKILL (Jetsam)" --> App

    Note["🚨 SILENT KILLER<br/>ISP memory is invisible to your app<br/>Total demand exceeds 4GB limit"]:::danger
```

- **Interviewer:** "Your AR app crashes with a 'SIGKILL' (Jetsam kill) when users take a high-resolution photo, even though your app is only using 822 MB of its 2 GB memory limit. Based on the RAM diagram, what is the 'invisible' resource consumer causing the OS to kill your process?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is too big — quantize it." Quantization helps, but the real issue is that the camera ISP's memory allocation is invisible to your app and uncontrollable.

  **Realistic Solution:** You are hitting a **Shared Resource RAM Collision**. The camera ISP service runs in a separate process but shares the same physical RAM pool. During photo capture, the ISP HDR pipeline can consume >1.2 GB. When combined with the OS and your app, total demand exceeds 4 GB, and the OS kills the app to protect the camera. The fix is to use **mmap** for weights (allowing OS eviction), reduce camera resolution during inference, or sequentialize the capture and ML phases.

  > **Napkin Math:** Total Memory = 1.5 GB (OS) + 1.2 GB (ISP) + 0.82 GB (App) = 3.52 GB. This is dangerously close to the 4 GB limit. Any background task (like a notification or a message) will push the system over the edge.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Burst Benchmarking Illusion</b> · <code>benchmarking</code> <code>power-thermal</code></summary>

### Benchmarking Peak Performance, Not Sustained Performance

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef peak fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef throttle fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5
    classDef process fill:#f3f4f6,stroke:#4b5563,stroke-width:2px,color:#1f2937

    T1["0-30s: PEAK STATE<br/>(45 TOPS / 30 FPS)"]:::peak
    T2["30-60s: HEATING<br/>(SoC Junction @ 80°C)"]:::process
    T3["60s+: THERMAL THROTTLE<br/>(20 TOPS / 12 FPS)"]:::throttle

    T1 --> T2
    T2 --> T3

    Note["🚨 BENCHMARKING ERROR<br/>Model designed for Peak (30 FPS)<br/>Field reality: Sustained (12 FPS)"]:::throttle
    T3 --- Note
```

- **Interviewer:** "Your mobile model runs at 30 FPS during internal demos, but users complain that it slows down to 12 FPS after a minute of use. Based on the performance timeline, what physical protection mechanism is engaging inside the smartphone?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The phone is defective" or "Add a cooling fan." You can't attach a fan to a user's phone. The phone is working exactly as designed — DVFS (Dynamic Voltage and Frequency Scaling) protects the SoC from thermal damage.

  **Realistic Solution:** You are hitting the **Sustained Thermal Envelope**. Mobile SoCs support high TOPS only for short "bursts." For continuous inference, the device throttles to its sustained thermal design power (sTDP), often 40-60% of peak. The fix is to target sustained performance (e.g., design for 20 TOPS, not 45 TOPS peak), implement adaptive frame rates based on `ThermalState`, or use the NPU instead of the GPU to halve heat generation.

  > **Napkin Math:** Sustained TOPS is typically ~50% of Peak. If your model requires 100% of peak TOPS to hit 30 FPS, you are guaranteed to drop to 15 FPS once the thermal envelope is saturated (~60 seconds).

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Backbone Bloat</b> · <code>architecture</code> <code>memory</code></summary>

### Three Copies of the Same Backbone

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef weight fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef bottleneck fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    subgraph "VRAM (Current: 171 MB)"
        direction LR
        subgraph "Team A"
            B1["Backbone 1<br/>(12 MB)"]:::bottleneck
            H1["Face Head"]
        end
        subgraph "Team B"
            B2["Backbone 2<br/>(12 MB)"]:::bottleneck
            H2["Mesh Head"]
        end
        subgraph "Team C"
            B3["Backbone 3<br/>(12 MB)"]:::bottleneck
            H3["Seg Head"]
        end
    end

    Note["🚨 REDUNDANCY ERROR<br/>Identical weights loaded 3x<br/>Shared activations computed 3x"]:::bottleneck
```

- **Interviewer:** "Your mobile app has three different computer vision features (Face Detection, Face Mesh, and Segmentation). All three use the same MobileNetV3 backbone. Based on the VRAM diagram, what is the 'efficiency gap' in your model loading strategy?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run the models in parallel on different cores." The phone has one NPU — parallel execution doesn't help. And you'd still waste 171 MB of memory.

  **Realistic Solution:** You have **Backbone Redundancy**. The app loads three identical copies of the backbone weights and computes the same features three times per frame. The fix is a **Multi-Task Architecture with Shared Backbone**: load the backbone once (12 MB), run it once (6ms), and branch into three lightweight task heads. This reduces memory by 64% and latency by 40%.

  > **Napkin Math:** Current: 3 backbones × 12 MB = 36 MB weights. Shared: 1 backbone × 12 MB = 12 MB. Latency: 3 × 6ms = 18ms (sequential backbone runs) vs 1 × 6ms = 6ms.

  📖 **Deep Dive:** [Network Architectures](https://harvard-edge.github.io/cs249r_book_dev/contents/network_architectures/network_architectures.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Frankenstein Model</b> · <code>ml-ops</code> <code>reliability</code></summary>

### No Atomicity Guarantee for Model Updates

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph LR
    classDef safe fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef partial fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    subgraph "Active Model File"
        L1["Layer 1 (v2.0 NEW)"]:::safe
        L2["Layer 2 (v2.0 NEW)"]:::safe
        LX["CRASH / POWER LOST"]:::danger
        L3["Layer 3 (v1.0 OLD)"]:::partial
        L4["Layer 4 (v1.0 OLD)"]:::partial
    end

    Note["🚨 CORRUPT STATE<br/>Frankenstein Model: Mixed Weights<br/>Output: Numerical Garbage"]:::danger
```

- **Interviewer:** "During a background model update on a user's phone, the device suddenly reboots. Upon next launch, the app doesn't crash, but the ML model produces nonsensical 'garbage' outputs. Based on the file-write diagram, what state is the model file in?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Check battery level before training." Battery checks don't prevent interruptions like OS kills or reboots.

  **Realistic Solution:** You have a **Frankenstein Model** caused by non-atomic updates. The pipeline wrote the updated model directly to the active file. If interrupted, the file contains a mix of new and old layers, which is numerically invalid. The fix is **Atomic Promotion (Shadow Copy)**: download/train the model in a temporary directory, and only after validation is complete, use an atomic `rename()` call to replace the active model file.

  > **Napkin Math:** If 60% of layers are v2.0 and 40% are v1.0, the feature map distribution from Layer 2 will not match the weights expected by Layer 3, leading to catastrophic error propagation.

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The CPU Wake-Lock Tax</b> · <code>sustainable-ai</code> <code>power-thermal</code></summary>

### The CPU Wake-Lock Tax

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef sleep fill:#f3f4f6,stroke:#94a3b8,stroke-width:1px,color:#64748b
    classDef active fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    subgraph "Application Processor (AP)"
        CPU["Cortex-A Core<br/>(AWAKE)"]:::active
        RAM["DDR RAM<br/>(POWERED)"]:::active
        Bus["Memory Bus<br/>(ACTIVE)"]:::active
    end

    subgraph "Always-On Domain (AOD)"
        DSP["DSP / micro-NPU<br/>(IDLE)"]:::sleep
    end

    Mic["Microphone"] --> CPU
    Note["🚨 POWER TAX<br/>AP held awake via Wake-Lock<br/>Basal power: 80mW (vs 1mW sleep)"]:::danger
```

- **Interviewer:** "You deploy a simple wake-word detection model. Even though the model uses very little CPU, users report that the app is the top battery consumer on their device. Based on the power domain diagram, where is the energy going?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model needs to be quantized to INT8 so it uses less power." The model's compute cost is negligible; the power is being burned by the hardware state required to execute it.

  **Realistic Solution:** You are suffering from **Application Processor Wake-Lock Tax**. To run the model on the main CPU, the OS must hold a wake-lock, powering up the high-power CPU rails and DDR RAM. This prevents the phone from entering Deep Sleep. The fix is to push always-on models down to the **Always-On Domain (AOD) / DSP**, which reads the microphone directly into tiny SRAM and only wakes the main AP if the wake-word is detected.

  > **Napkin Math:** Basal AP power is ~80 mW. DSP power is ~1 mW. By failing to delegate to the AOD, you are using 80x more power than necessary just to keep the lights on for a simple task.

  📖 **Deep Dive:** [Sustainable AI](https://harvard-edge.github.io/cs249r_book_dev/contents/sustainable_ai/sustainable_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silicon Shared Oven</b> · <code>sustainable-ai</code> <code>power-thermal</code></summary>

### The Shared Silicon Thermal Envelope

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef cooling fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef heat fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    subgraph "Single Silicon Die (SoC)"
        direction LR
        GPU["GPU<br/>(Heavy 3D Game)"]:::danger
        CPU["CPU<br/>(Background)"]:::heat
        NPU["NPU<br/>(Translation)"]:::danger
    end

    Skin["Phone Surface<br/>(Limit: 45°C)"]

    GPU -- "Massive Heat" --> Skin
    CPU -- "Heat" --> Skin
    Skin -- "🚨 THERMAL EVENT" --> SoC_Control["DVFS Controller"]
    SoC_Control -- "GLOBAL THROTTLE" --> GPU
    SoC_Control -- "GLOBAL THROTTLE" --> NPU

    Note["🚨 SILENT THROTTLE<br/>NPU latency triples because<br/>GPU generates the heat"]:::danger
```

- **Interviewer:** "Your real-time translation app works perfectly until the user starts a 3D game in picture-in-picture mode. Suddenly, your NPU translation latency triples. Based on the SoC diagram, why is the game affecting the NPU speed?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The game is stealing NPU cycles." The game uses the GPU, not the NPU. They are separate physical cores.

  **Realistic Solution:** You are hitting **Global Thermal Throttling**. In a mobile SoC, all cores share the same silicon die and thermal envelope. When the GPU generates massive heat from a game, the system's DVFS controller downclocks *the entire SoC*—including the NPU—to stay within safe skin-temperature limits. The fix is to use hyper-optimized, low-power models that can still meet deadlines even when the SoC is forced into its lowest frequency state.

  > **Napkin Math:** A phone can only dissipate ~3-5 Watts sustained. If a 3D game pulls 4 Watts, the NPU is left with <1 Watt of thermal margin, triggering immediate frequency capping.

  📖 **Deep Dive:** [Sustainable AI](https://harvard-edge.github.io/cs249r_book_dev/contents/sustainable_ai/sustainable_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Radio Energy Wall</b> · <code>sustainable-ai</code> <code>power-thermal</code></summary>

### The Cellular Radio Wake Dominates Power, Not the Model

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef model fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef cpu fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef radio fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b,stroke-dasharray: 5 5

    subgraph "Power Budget (mW)"
        direction TB
        M["ML Model Inference<br/>(3 mW)"]:::model
        C["CPU Always-On<br/>(80 mW)"]:::cpu
        R["Cellular Radio Wake<br/>(800 mW)"]:::radio
    end

    Note["🚨 ENERGY MISMATCH<br/>Radio uses 266x more power than ML<br/>Waking every 5m destroys battery"]:::radio
```

- **Interviewer:** "You optimize your activity-tracking model down to 3 mW. However, the app still drains the battery by 10% per hour because it uploads labels to your server every 5 minutes. Based on the power breakdown, where is your optimization effort being wasted?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Retrain the model to be even smaller." The model's 3 mW is already negligible compared to the system overhead.

  **Realistic Solution:** You are hitting the **Radio Wake Overhead**. The cellular radio consumes ~800 mW every time it wakes from idle. Waking it every 5 minutes for a tiny payload prevents the radio from staying in its low-power state. The fix is **Batching and Delegation**: buffer results locally and upload once per hour, and use the OS's low-power activity co-processor (like Apple's M-series) instead of a custom always-on CPU loop.

  > **Napkin Math:** Radio: 800 mW × 10s (wake + tail) × 12 times/hr = 26.7 mWh. Model: 3 mW × 1 hr = 3 mWh. The radio wake is ~9x more expensive than the actual ML work.

  📖 **Deep Dive:** [Sustainable AI](https://harvard-edge.github.io/cs249r_book_dev/contents/sustainable_ai/sustainable_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The UMA Bandwidth Wall</b> · <code>hardware</code> <code>memory</code></summary>

### UMA Bandwidth Contention

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'monospace', 'primaryColor': '#ffffff', 'edgeLabelBackground':'#ffffff'}}}%%
graph TD
    classDef device fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef memory fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#9a3412
    classDef bottleneck fill:#fee2e2,stroke:#dc2626,stroke-width:4px,color:#991b1b,stroke-dasharray: 5 5

    Disp["Display Controller<br/>(120Hz Refresh)"]:::device
    GPU["GPU<br/>(Render Framebuffer)"]:::device
    NPU["NPU<br/>(Style Transfer Model)"]:::device

    subgraph "Unified Memory Architecture (UMA)"
        Bus[["Shared Memory Bus<br/>(ARB: Display Priority)"]]:::bottleneck
    end

    RAM[("System RAM (LPDDR5)")]:::memory

    Disp -->|High BW / High Priority| Bus
    GPU -->|High BW| Bus
    NPU -->|Wait / Starved| Bus
    Bus --> RAM

    Note["🚨 BUS SATURATION<br/>120Hz 4K display consumes 40% BW<br/>NPU bandwidth slashed -> Stutters"]:::bottleneck
```

- **Interviewer:** "Your AR app runs a real-time style transfer model. When the user upgrades to a Pro phone with a 120Hz 'ProMotion' display, the NPU inference suddenly starts stuttering and missing deadlines. Based on the UMA diagram, why is a faster screen slowing down your AI?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The phone is thermal throttling." While heat is a factor, the immediate drop is due to memory bus physics.

  **Realistic Solution:** You are suffering from **Memory Bus Contention**. In a Unified Memory Architecture (UMA), the Display Controller, GPU, and NPU all share the same LPDDR5 bus. A 120Hz display must refresh the framebuffer twice as often as a 60Hz one, consuming a massive slice of the total bandwidth. The memory controller prioritizes the display to prevent tearing, starving the bandwidth-hungry NPU. The fix is to drop the refresh rate to 60Hz during heavy AR features or reduce NPU precision to INT8 to halve its bandwidth needs.

  > **Napkin Math:** A 4K 120Hz display requires moving ~4.5 GB/s just for frame refresh. On a phone with 50 GB/s peak bandwidth, the display system (plus GPU rendering) can easily consume 40-50% of the practical bus capacity.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>
