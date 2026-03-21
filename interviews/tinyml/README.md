# 🔬 TinyML Track — Microcontroller & Ultra-Low-Power AI

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <a href="../mobile/README.md">📱 Mobile</a> · <b>🔬 TinyML</b>
</div>

---

The TinyML track covers ML systems that run on microcontrollers and ultra-low-power devices — always-on sensing, energy harvesting, hard real-time inference in kilobytes of SRAM.

### The Constraint Regime

<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">TinyML Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>MFLOPS (Cortex-M, RISC-V, custom accelerators)</td></tr>
    <tr><td><b>Memory</b></td><td>256 KB – 2 MB SRAM, 1–16 MB flash</td></tr>
    <tr><td><b>Interconnect</b></td><td>SPI, I2C, UART, GPIO</td></tr>
    <tr><td><b>Power budget</b></td><td>Microwatts to milliwatts</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>SRAM capacity and hard real-time deadlines</td></tr>
    <tr><td><b>Failure mode</b></td><td>Model doesn't fit in SRAM, missed interrupt deadlines, energy budget exceeded</td></tr>
  </tbody>
</table>


### The Learning Journey

Each file represents a **system scope** — the system you're reasoning about. Within each file, questions are organized by competency topic and mastery level.

<table>
  <thead>
    <tr>
      <th width="5%">#</th>
      <th width="30%">Scope</th>
      <th width="35%">What you're studying</th>
      <th width="10%">L3</th>
      <th width="10%">L4</th>
      <th width="10%">L5</th>
      <th width="10%">L6+</th>
      <th width="10%">Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>01</b></td>
      <td><b><a href="01_microcontroller.md">The Microcontroller</a></b></td>
      <td><i>What fits in 256 KB of SRAM?</i></td>
      <td>27</td>
      <td>62</td>
      <td>40</td>
      <td>16</td>
      <td><b>145</b></td>
    </tr>
    <tr>
      <td><b>02</b></td>
      <td><b><a href="02_sensing_pipeline.md">The Sensing Pipeline</a></b></td>
      <td><i>From sensor input to inference output</i></td>
      <td>7</td>
      <td>26</td>
      <td>17</td>
      <td>6</td>
      <td><b>56</b></td>
    </tr>
    <tr>
      <td><b>03</b></td>
      <td><b><a href="03_deployed_device.md">The Deployed Device</a></b></td>
      <td><i>How you update firmware and keep it alive for years</i></td>
      <td>6</td>
      <td>9</td>
      <td>9</td>
      <td>2</td>
      <td><b>26</b></td>
    </tr>
    <tr>
      <td><b>04</b></td>
      <td><b><a href="04_visual_debugging.md">Visual Architecture Debugging</a></b></td>
      <td><i>Can you spot the bottleneck in a TinyML system diagram?</i></td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>—</td>
      <td><b>11</b></td>
    </tr>
    <tr>
      <td></td><td><b>Total</b></td><td></td><td></td><td></td><td></td><td></td><td><b>238</b></td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at sensor companies, IoT platforms, wearable tech firms, and embedded AI startups deploying ML to devices that run on batteries or harvested energy.
