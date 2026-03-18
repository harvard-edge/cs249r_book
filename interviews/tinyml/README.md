# 🔬 TinyML Track — Microcontroller & Ultra-Low-Power AI

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> ·
  <a href="../edge/README.md">🤖 Edge</a> ·
  <a href="../mobile/README.md">📱 Mobile</a> ·
  <b>🔬 TinyML</b>
</div>

---

The TinyML track covers ML systems running on microcontrollers — where the entire model, runtime, and inference engine must fit in kilobytes of SRAM, execute in microseconds, and run on milliwatts of power.

### The Constraint Regime

<table>
  <thead>
    <tr>
      <th width="25%">Dimension</th>
      <th width="75%">TinyML Reality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Compute</b></td>
      <td>MFLOPS (Cortex-M4/M7, RISC-V, ESP32) — often no FPU</td>
    </tr>
    <tr>
      <td><b>Memory</b></td>
      <td>256 KB–2 MB SRAM, 1–16 MB Flash</td>
    </tr>
    <tr>
      <td><b>Interconnect</b></td>
      <td>SPI, I2C, UART — kilobytes/second</td>
    </tr>
    <tr>
      <td><b>Power budget</b></td>
      <td>1–100 mW (often battery or energy harvesting)</td>
    </tr>
    <tr>
      <td><b>Primary bottleneck</b></td>
      <td>SRAM capacity — the model must fit entirely on-chip</td>
    </tr>
    <tr>
      <td><b>Failure mode</b></td>
      <td>Model doesn't fit, misses hard real-time deadline, exceeds power budget</td>
    </tr>
  </tbody>
</table>

### What Makes TinyML Different from Everything Else

In the cloud, you optimize for throughput. On mobile, you optimize for battery. In TinyML, you optimize for **existence** — can the model even fit? There is no operating system, no virtual memory, no dynamic allocation. The entire inference pipeline — weights, activations, scratch buffers, and application code — must coexist in a flat memory space measured in kilobytes. Every byte is a design decision.

### Topics That Need Questions

These are the areas where TinyML-specific interview questions would be most valuable. Each maps to real interview scenarios at companies like Arduino, Edge Impulse, Qualcomm (for always-on sensing), or embedded AI teams at larger companies.

<table>
  <thead>
    <tr>
      <th width="22%">Topic</th>
      <th width="28%">What TinyML interviews test</th>
      <th width="50%">Example scenario</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Memory layout</b></td>
      <td>SRAM partitioning, activation reuse, operator scheduling for peak RAM</td>
      <td>"Your model needs 300 KB peak RAM but you only have 256 KB SRAM. How do you fit it without changing the model?"</td>
    </tr>
    <tr>
      <td><b>Quantization</b></td>
      <td>INT8, INT4, binary/ternary, fixed-point arithmetic, post-training vs QAT</td>
      <td>"Your keyword spotting model loses 8% accuracy going from INT8 to INT4. Is that acceptable? How do you recover it?"</td>
    </tr>
    <tr>
      <td><b>Integer-only inference</b></td>
      <td>No floating point — all math in fixed-point, requantization between layers</td>
      <td>"Explain how a quantized Conv2D executes on a Cortex-M4 with no FPU."</td>
    </tr>
    <tr>
      <td><b>Model architecture</b></td>
      <td>MobileNet, MCUNet, depth-wise separable convolutions, NAS for MCUs</td>
      <td>"Why does MobileNetV2 use inverted residuals, and why does that matter on a microcontroller?"</td>
    </tr>
    <tr>
      <td><b>Power & energy</b></td>
      <td>Active vs sleep power, duty cycling, energy harvesting budgets</td>
      <td>"Your sensor wakes up every 10 seconds, runs inference, and sleeps. What's the average power draw?"</td>
    </tr>
    <tr>
      <td><b>Compiler & runtime</b></td>
      <td>TFLite Micro, TVM, CMSIS-NN, ahead-of-time compilation, no dynamic allocation</td>
      <td>"Why can't TFLite Micro use malloc? What does it use instead?"</td>
    </tr>
    <tr>
      <td><b>Sensor pipelines</b></td>
      <td>Audio (keyword spotting), accelerometer (gesture), image (person detection)</td>
      <td>"Your microphone samples at 16 kHz. How do you extract Mel spectrograms in real-time on a Cortex-M4?"</td>
    </tr>
  </tbody>
</table>

### The Rounds

<table>
  <thead>
    <tr>
      <th width="45%">Round</th>
      <th width="40%">Focus</th>
      <th width="15%">Questions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href="01_micro_architectures.md">1. TinyML Systems — Inference at the Edge of Physics</a></b></td>
      <td>Memory layout, quantization, integer inference, power, CMSIS-NN, sensor pipelines</td>
      <td>27</td>
    </tr>
    <tr>
      <td><b><a href="02_compute_and_memory.md">2. Constraints & Trade-offs</a></b></td>
      <td>Compute analysis, memory systems, numerical representation, architecture costs, latency, power</td>
      <td>62</td>
    </tr>
    <tr>
      <td><b><a href="03_data_and_deployment.md">3. Operations & Deployment</a></b></td>
      <td>FOTA updates, SRAM overflow, bootloader design, watchdog timers, security, fleet management, power profiling</td>
      <td>25</td>
    </tr>
    <tr>
      <td><b><a href="04_visual_debugging.md">4. Visual Architecture Debugging</a></b></td>
      <td>Mermaid diagram challenges: spot the flaw in TinyML system designs</td>
      <td>11</td>
    </tr>
    <tr>
      <td><b><a href="05_advanced_systems.md">5. Advanced TinyML Systems</a></b></td>
      <td>NAS for MCUs, energy harvesting, multi-sensor fusion, compiler design, federated learning, always-on detection</td>
      <td>46</td>
    </tr>
  </tbody>
</table>

### Contributing

We need more TinyML questions — especially from engineers at Edge Impulse, Arduino, and embedded AI teams. See the [question format](../README.md#question-format) and submit a PR.
