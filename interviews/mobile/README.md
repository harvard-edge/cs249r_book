# 📱 Mobile Track — On-Device AI for Smartphones

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <b>📱 Mobile</b> · <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Mobile track covers ML systems that run on smartphones and tablets — on-device inference, NPU delegation, app store constraints, and battery-aware optimization.

### The Constraint Regime

<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">Mobile Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>TOPS (Snapdragon, Apple Neural Engine, MediaTek APU, Samsung NPU)</td></tr>
    <tr><td><b>Memory</b></td><td>6–12 GB shared with OS and apps, no dedicated VRAM</td></tr>
    <tr><td><b>Interconnect</b></td><td>On-SoC fabric, shared memory bus</td></tr>
    <tr><td><b>Power budget</b></td><td>3–5W total device power</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>Battery life and shared resources</td></tr>
    <tr><td><b>Failure mode</b></td><td>Thermal throttling, app eviction, silent accuracy loss</td></tr>
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
      <td><b><a href="01_device_hardware.md">The Device & SoC</a></b></td>
      <td><i>What hardware are you working with?</i></td>
      <td>99</td>
      <td>162</td>
      <td>189</td>
      <td>59</td>
      <td>56</td>
      <td>42</td>
      <td><b>605</b></td>
    </tr>
    <tr>
      <td><b>02</b></td>
      <td><b><a href="02_app_experience.md">The App Experience</a></b></td>
      <td><i>How you make inference feel instant</i></td>
      <td>4</td>
      <td>5</td>
      <td>18</td>
      <td>31</td>
      <td>16</td>
      <td>10</td>
      <td><b>82</b></td>
    </tr>
    <tr>
      <td><b>03</b></td>
      <td><b><a href="03_ship_and_update.md">Ship & Update</a></b></td>
      <td><i>How you ship models to a billion phones and keep them current</i></td>
      <td>3</td>
      <td>7</td>
      <td>12</td>
      <td>18</td>
      <td>8</td>
      <td>6</td>
      <td><b>54</b></td>
    </tr>
    <tr>
      <td><b>04</b></td>
      <td><b><a href="04_visual_debugging.md">Visual Architecture Debugging</a></b></td>
      <td><i>Can you spot the bottleneck in a mobile system diagram?</i></td>
      <td>—</td>
      <td>—</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>—</td>
      <td><b>10</b></td>
    </tr>
    <tr>
      <td></td><td><b>Total</b></td><td></td><td><b>106</b></td><td><b>174</b></td><td><b>220</b></td><td><b>115</b></td><td><b>82</b></td><td><b>58</b></td><td><b>751</b></td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at Apple, Google, Samsung, Qualcomm, and mobile-first AI companies building on-device ML features.
