# 📱 Mobile Track — On-Device AI for Smartphones

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> ·
  <a href="../edge/README.md">🤖 Edge</a> ·
  <b>📱 Mobile</b> ·
  <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Mobile track covers ML systems deployed on smartphones and tablets — the most constrained high-performance computing environment in the world. Every phone is a shared-resource system where your ML model competes with the camera, the display, the cellular modem, and the user's other apps for memory, compute, and battery.

### The Constraint Regime

<table>
  <thead>
    <tr>
      <th width="25%">Dimension</th>
      <th width="75%">Mobile Reality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Compute</b></td>
      <td>TOPS (Apple ANE, Qualcomm Hexagon, Google Tensor TPU), shared with GPU and CPU</td>
    </tr>
    <tr>
      <td><b>Memory</b></td>
      <td>6–12 GB DRAM, shared with OS, apps, camera ISP</td>
    </tr>
    <tr>
      <td><b>Interconnect</b></td>
      <td>On-chip NoC (NPU↔CPU↔GPU), UFS flash</td>
    </tr>
    <tr>
      <td><b>Power budget</b></td>
      <td>3–5W total device power (ML gets a fraction)</td>
    </tr>
    <tr>
      <td><b>Primary bottleneck</b></td>
      <td>Battery life and shared resource contention</td>
    </tr>
    <tr>
      <td><b>Failure mode</b></td>
      <td>App killed by OS for memory pressure, thermal throttling, jank</td>
    </tr>
  </tbody>
</table>

### What Makes Mobile Different from Edge

Edge devices are dedicated to ML — the entire system exists to run inference. Mobile devices are general-purpose computers where ML is one of dozens of competing workloads. Your model must coexist with the camera app, the browser, and the OS itself. The user can switch away at any moment, and the OS will kill your process to reclaim memory. Battery life is the ultimate constraint — no user will tolerate an app that drains their phone.

### The Rounds

<table>
  <thead>
    <tr>
      <th width="35%">Round</th>
      <th width="45%">Focus</th>
      <th width="20%">Questions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href="01_systems_and_soc.md">1. Mobile Systems & On-Device Physics</a></b></td>
      <td>NPU delegation, thermal throttling, memory pressure, app lifecycle</td>
      <td>26</td>
    </tr>
    <tr>
      <td><b><a href="02_compute_and_memory.md">2. Constraints & Trade-offs</a></b></td>
      <td>Compute analysis, memory, quantization, architecture, latency, power, NPU delegation, LLMs</td>
      <td>61</td>
    </tr>
    <tr>
      <td><b><a href="03_data_and_deployment.md">3. Operations & Deployment</a></b></td>
      <td>Model optimization, app store delivery, monitoring, privacy, platform design</td>
      <td>53</td>
    </tr>
    <tr>
      <td><b><a href="04_visual_debugging.md">4. Visual Architecture Debugging</a></b></td>
      <td>Spot the bottleneck in mobile ML system diagrams</td>
      <td>11</td>
    </tr>
    <tr>
      <td><b><a href="05_advanced_systems.md">5. Advanced Mobile Systems</a></b></td>
      <td>On-device LLMs, cross-platform deployment, federated learning, NAS, personalization</td>
      <td>26</td>
    </tr>
    <tr>
      <td><b>Total</b></td>
      <td></td>
      <td><b>177</b></td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at Apple (Core ML, ANE), Google (TFLite, Pixel), Qualcomm (Hexagon, QNN), Samsung (Exynos NPU), Meta (on-device AI), and any company shipping ML features in mobile apps. Also valuable for mobile developers adding ML capabilities to existing apps.

### Contributing

We need more mobile questions — especially from engineers at Apple, Google, and Qualcomm. See the [question format](../README.md#question-format) and submit a PR.
