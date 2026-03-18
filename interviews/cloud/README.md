# ☁️ Cloud Track — Data Center & Distributed Systems

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <b>☁️ Cloud</b> ·
  <a href="../edge/README.md">🤖 Edge</a> ·
  <a href="../mobile/README.md">📱 Mobile</a> ·
  <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Cloud track covers ML systems that run in data centers — from a single H100 to 10,000-GPU training clusters to production serving fleets handling millions of requests per second.

### The Constraint Regime

<table>
  <thead>
    <tr>
      <th width="25%">Dimension</th>
      <th width="75%">Cloud Reality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Compute</b></td>
      <td>PFLOPS (H100, TPU, B200)</td>
    </tr>
    <tr>
      <td><b>Memory</b></td>
      <td>80 GB HBM per chip, terabytes across a cluster</td>
    </tr>
    <tr>
      <td><b>Interconnect</b></td>
      <td>NVLink (900 GB/s intra-node), InfiniBand (400 Gbps inter-node)</td>
    </tr>
    <tr>
      <td><b>Power budget</b></td>
      <td>700W–1000W per chip, megawatts per cluster</td>
    </tr>
    <tr>
      <td><b>Primary bottleneck</b></td>
      <td>Memory bandwidth (single node), network (multi-node)</td>
    </tr>
    <tr>
      <td><b>Failure mode</b></td>
      <td>Silent data corruption at scale, straggler nodes, MTBF collapse</td>
    </tr>
  </tbody>
</table>

### The Rounds

<table>
  <thead>
    <tr>
      <th width="30%">Round</th>
      <th width="45%">Focus</th>
      <th width="25%">Questions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><a href="01_compute_and_memory.md">1. Silicon Physics</a></b></td>
      <td>What happens inside a single server</td>
      <td>61</td>
    </tr>
    <tr>
      <td><b><a href="02_network_and_distributed.md">2. Distributed Infra</a></b></td>
      <td>What happens when you exceed one node</td>
      <td>38</td>
    </tr>
    <tr>
      <td><b><a href="03_inference_and_serving.md">3. Production Serving</a></b></td>
      <td>Surviving real user traffic</td>
      <td>40</td>
    </tr>
    <tr>
      <td><b><a href="04_data_and_mlops.md">4. Ops & Economics</a></b></td>
      <td>Keeping systems healthy over time</td>
      <td>49</td>
    </tr>
    <tr>
      <td><b><a href="05_visual_debugging.md">5. Visual Debugging</a></b></td>
      <td>Spotting bottlenecks in diagrams</td>
      <td>10</td>
    </tr>
    <tr>
      <td><b><a href="06_advanced_systems.md">6. Advanced Systems</a></b></td>
      <td>Compute analysis, power, architecture cost, security & fairness</td>
      <td>55</td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at frontier labs and cloud infrastructure companies — Meta, Google, OpenAI, Anthropic, NVIDIA, Amazon, Microsoft, and similar organizations building or operating large-scale ML systems.
