# 🤖 Edge Track — Autonomous Systems & Industrial AI

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <b>🤖 Edge</b> · <a href="../mobile/README.md">📱 Mobile</a> · <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Edge track covers ML systems deployed on dedicated hardware at the point of action — autonomous vehicles, robotics platforms, CCTV and surveillance systems, industrial inspection, and medical devices.

### The Constraint Regime

<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">Edge Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>TOPS (Jetson Orin, Hailo-8, Intel Movidius, Google Coral)</td></tr>
    <tr><td><b>Memory</b></td><td>8–32 GB DRAM, shared with sensor pipelines</td></tr>
    <tr><td><b>Interconnect</b></td><td>PCIe, MIPI CSI (camera), CAN bus (automotive)</td></tr>
    <tr><td><b>Power budget</b></td><td>15–75W per module</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>Thermal envelope and real-time deadlines</td></tr>
    <tr><td><b>Failure mode</b></td><td>Missing a hard real-time deadline, thermal throttling under sustained load</td></tr>
  </tbody>
</table>


### The Learning Journey

Each file represents a **system scope** — the system you're reasoning about. Within each file, questions are organized by competency topic and mastery level.

<table>
  <thead>
    <tr>
      <th width="5%">#</th>
      <th width="25%">Scope</th>
      <th width="25%">What you're studying</th>
      <th width="7%">L1</th>
      <th width="7%">L2</th>
      <th width="7%">L3</th>
      <th width="7%">L4</th>
      <th width="7%">L5</th>
      <th width="7%">L6+</th>
      <th width="8%">Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>01</b></td>
      <td><b><a href="01_hardware_platform.md">The Hardware Platform</a></b></td>
      <td><i>What silicon are you working with and what are its limits?</i></td>
      <td>130</td>
      <td>163</td>
      <td>135</td>
      <td>50</td>
      <td>61</td>
      <td>60</td>
      <td><b>599</b></td>
    </tr>
    <tr>
      <td><b>02</b></td>
      <td><b><a href="02_realtime_pipeline.md">The Real-Time Pipeline</a></b></td>
      <td><i>How you meet deadlines with sensor data</i></td>
      <td>7</td>
      <td>2</td>
      <td>18</td>
      <td>33</td>
      <td>28</td>
      <td>12</td>
      <td><b>100</b></td>
    </tr>
    <tr>
      <td><b>03</b></td>
      <td><b><a href="03_deployed_system.md">The Deployed System</a></b></td>
      <td><i>How you get it into the field and keep it running</i></td>
      <td>7</td>
      <td>29</td>
      <td>12</td>
      <td>24</td>
      <td>17</td>
      <td>12</td>
      <td><b>101</b></td>
    </tr>
    <tr>
      <td><b>04</b></td>
      <td><b><a href="04_visual_debugging.md">Visual Architecture Debugging</a></b></td>
      <td><i>Can you spot the bottleneck in an edge system diagram?</i></td>
      <td>—</td>
      <td>—</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>—</td>
      <td><b>11</b></td>
    </tr>
    <tr>
      <td></td><td><b>Total</b></td><td></td><td><b>144</b></td><td><b>194</b></td><td><b>166</b></td><td><b>114</b></td><td><b>109</b></td><td><b>84</b></td><td><b>811</b></td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at autonomous vehicle companies (Tesla, Waymo, Cruise), robotics firms (Boston Dynamics, Agility), industrial AI startups, and edge computing platforms (NVIDIA, Qualcomm, Hailo).
