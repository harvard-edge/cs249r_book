# ☁️ Cloud Track — Data Center & Distributed Systems

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <b>☁️ Cloud</b> · <a href="../edge/README.md">🤖 Edge</a> · <a href="../mobile/README.md">📱 Mobile</a> · <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Cloud track covers ML systems that run in data centers — from a single H100 to 10,000-GPU training clusters to production serving fleets handling millions of requests per second.

### The Constraint Regime

<table>
  <thead><tr><th width="25%">Dimension</th><th width="75%">Cloud Reality</th></tr></thead>
  <tbody>
    <tr><td><b>Compute</b></td><td>PFLOPS (H100, TPU, B200)</td></tr>
    <tr><td><b>Memory</b></td><td>80 GB HBM per chip, terabytes across a cluster</td></tr>
    <tr><td><b>Interconnect</b></td><td>NVLink (900 GB/s intra-node), InfiniBand (400 Gbps inter-node)</td></tr>
    <tr><td><b>Power budget</b></td><td>700W–1000W per chip, megawatts per cluster</td></tr>
    <tr><td><b>Primary bottleneck</b></td><td>Memory bandwidth (single node), network (multi-node)</td></tr>
    <tr><td><b>Failure mode</b></td><td>Silent data corruption at scale, straggler nodes, MTBF collapse</td></tr>
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
      <td><b><a href="01_single_machine.md">The Single Machine</a></b></td>
      <td><i>What happens inside one server</i></td>
      <td>29</td>
      <td>43</td>
      <td>36</td>
      <td>22</td>
      <td><b>130</b></td>
    </tr>
    <tr>
      <td><b>02</b></td>
      <td><b><a href="02_distributed_systems.md">The Distributed System</a></b></td>
      <td><i>What happens when you exceed one node</i></td>
      <td>8</td>
      <td>21</td>
      <td>24</td>
      <td>18</td>
      <td><b>71</b></td>
    </tr>
    <tr>
      <td><b>03</b></td>
      <td><b><a href="03_serving_stack.md">The Serving Stack</a></b></td>
      <td><i>How you serve models to real users</i></td>
      <td>8</td>
      <td>21</td>
      <td>15</td>
      <td>4</td>
      <td><b>48</b></td>
    </tr>
    <tr>
      <td><b>04</b></td>
      <td><b><a href="04_production_ops.md">The Production System</a></b></td>
      <td><i>How you keep it running and healthy</i></td>
      <td>7</td>
      <td>13</td>
      <td>8</td>
      <td>9</td>
      <td><b>37</b></td>
    </tr>
    <tr>
      <td><b>05</b></td>
      <td><b><a href="05_visual_debugging.md">Visual Architecture Debugging</a></b></td>
      <td><i>Can you spot the bottleneck in a diagram?</i></td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td><b>10</b></td>
    </tr>
    <tr>
      <td></td><td><b>Total</b></td><td></td><td></td><td></td><td></td><td></td><td><b>296</b></td>
    </tr>
  </tbody>
</table>

### Who This Track Is For

Engineers interviewing at frontier labs and cloud infrastructure companies — Meta, Google, OpenAI, Anthropic, NVIDIA, Amazon, Microsoft, and similar organizations building or operating large-scale ML systems.
