# 🤖 Edge Track — Autonomous Systems & Industrial AI

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> ·
  <b>🤖 Edge</b> ·
  <a href="../mobile/README.md">📱 Mobile</a> ·
  <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Edge track covers ML systems deployed on dedicated hardware at the point of action — autonomous vehicles, robotics platforms, CCTV and surveillance systems, industrial inspection, and medical devices.

### The Constraint Regime

<table>
  <thead>
    <tr>
      <th width="25%">Dimension</th>
      <th width="75%">Edge Reality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Compute</b></td>
      <td>TOPS (Jetson Orin, Hailo-8, Intel Movidius, Google Coral)</td>
    </tr>
    <tr>
      <td><b>Memory</b></td>
      <td>8–32 GB DRAM, shared with sensor pipelines</td>
    </tr>
    <tr>
      <td><b>Interconnect</b></td>
      <td>PCIe, MIPI CSI (camera), CAN bus (automotive)</td>
    </tr>
    <tr>
      <td><b>Power budget</b></td>
      <td>15–75W per module</td>
    </tr>
    <tr>
      <td><b>Primary bottleneck</b></td>
      <td>Thermal envelope and real-time deadlines</td>
    </tr>
    <tr>
      <td><b>Failure mode</b></td>
      <td>Missing a hard real-time deadline, thermal throttling under sustained load</td>
    </tr>
  </tbody>
</table>

### What Makes Edge Different from Cloud

In the cloud, you can always add more GPUs. At the edge, the hardware is fixed and the environment is hostile. An autonomous vehicle running object detection at 30 FPS cannot drop frames when the sun angle changes. A robotic arm running pose estimation cannot pause for garbage collection. The physics of edge is the physics of **hard constraints under uncertainty**.

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
      <td><b><a href="01_systems_and_real_time.md">1. Edge Systems & Real-Time Physics</a></b></td>
      <td>Roofline, real-time deadlines, thermal management, sensor fusion, OTA</td>
      <td>40</td>
    </tr>
    <tr>
      <td><b><a href="02_compute_and_memory.md">2. Constraints & Trade-offs</a></b></td>
      <td>Compute analysis, memory budgets, quantization, architecture, latency, power, edge LLMs</td>
      <td>63</td>
    </tr>
    <tr>
      <td><b><a href="03_data_and_deployment.md">3. Operations & Deployment</a></b></td>
      <td>Model optimization, fleet management, monitoring, security, economics</td>
      <td>47</td>
    </tr>
    <tr>
      <td><b><a href="04_visual_debugging.md">4. Visual Architecture Debugging</a></b></td>
      <td>Spot the bottleneck in edge system diagrams</td>
      <td>11</td>
    </tr>
    <tr>
      <td><b><a href="05_heterogeneous_and_advanced.md">5. Advanced Edge Systems</a></b></td>
      <td>Safety certification, multi-sensor fusion, privacy, long-term reliability, edge-cloud hybrid, heterogeneous accelerators, thermal, fleet ops</td>
      <td>46</td>
    </tr>
    <tr>
      <td><b>Total</b></td>
      <td></td>
      <td><b>207</b></td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at autonomous vehicle companies (Tesla, Waymo, Cruise), robotics firms (Boston Dynamics, Agility), industrial AI startups, and edge computing platforms (NVIDIA, Qualcomm, Hailo). Also valuable for anyone deploying ML to devices that must operate reliably in the physical world.

### Contributing

We need more edge questions — especially from engineers at Tesla, Waymo, Boston Dynamics, and industrial AI companies. See the [question format](../README.md#question-format) and submit a PR.
