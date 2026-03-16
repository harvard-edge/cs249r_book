# The Architect's Rubric — Self-Evaluation Guide 📋

<div align="center">
  <a href="README.md">🏠 Home</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="cloud/01_Single_Node_Physics.md">🧱 Round 1</a> ·
  <a href="cloud/02_Distributed_Infrastructure.md">🚀 Round 2</a> ·
  <a href="cloud/03_Production_Serving.md">⚡ Round 3</a> ·
  <a href="cloud/04_Operations_and_Economics.md">💼 Round 4</a> ·
  <a href="cloud/05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---

How do Principal Engineers grade your system design interviews? They don't look for a single "correct" architecture — they evaluate you across multiple axes of system maturity.

Use this rubric to self-evaluate your proposed designs. For each axis, check where your answer lands. The two columns map to the [Mastery Levels](README.md#mastery-levels): **Mid-Level (L4)** tests whether you can *apply* known concepts to diagnose a system, while **Staff-Level (L6+)** tests whether you can *derive* solutions from first principles under novel constraints.

> **[➕ Propose a new Rubric Axis](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/00_The_Architects_Rubric.md)** (Edit in Browser)

---

## The Evaluation Matrix

<table>
  <thead>
    <tr>
      <th width="18%">Engineering Axis</th>
      <th width="41%">Mid-Level Signal (L4)</th>
      <th width="41%">Staff-Level Signal (L6+)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Bottleneck Reasoning</b><br><sub>Can you find the wall?</sub></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Identifies compute vs memory bound</li>
          <li><input type="checkbox" disabled> Knows Roofline model exists</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Calculates Arithmetic Intensity and places workload on Roofline</li>
          <li><input type="checkbox" disabled checked> Predicts how bottleneck shifts under quantization, batching, or hardware change</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Parallelism Strategy</b><br><sub>How do you split the work?</sub></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Knows Data Parallelism replicates the model</li>
          <li><input type="checkbox" disabled> Understands gradient synchronization</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Dimensions a 3D parallelism cube (TP × PP × DP) from physical constraints</li>
          <li><input type="checkbox" disabled checked> Maps TP to NVLink domain, PP across racks, DP to remainder</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Latency Reasoning</b><br><sub>Where does the time go?</sub></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Measures end-to-end request latency</li>
          <li><input type="checkbox" disabled> Understands batching improves throughput</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Splits TTFT vs TPOT and knows which is compute-bound vs bandwidth-bound</li>
          <li><input type="checkbox" disabled checked> Models queueing behavior (Erlang-C / Little's Law) under load</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Data Gravity</b><br><sub>Where does the data live?</sub></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Understands storage I/O limits</li>
          <li><input type="checkbox" disabled> Uses batching for throughput</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Calculates exact network transfer times for model weights and KV-cache</li>
          <li><input type="checkbox" disabled checked> Designs zero-copy inference pipelines and co-locates compute with data</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Failure Modes</b><br><sub>What breaks at scale?</sub></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Handles node crashes with retries</li>
          <li><input type="checkbox" disabled> Implements basic checkpointing</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Mitigates silent data corruption (SDC) and straggler nodes</li>
          <li><input type="checkbox" disabled checked> Calculates optimal checkpoint interval using Young-Daly equation</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Cost & TCO</b><br><sub>What does it actually cost?</sub></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Chooses cheaper instances</li>
          <li><input type="checkbox" disabled> Understands spot-instance risks</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Formulates retraining staleness math (cost of drift vs cost of retraining)</li>
          <li><input type="checkbox" disabled checked> Factors PUE, cooling, and OpEx into fleet TCO over hardware lifecycle</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

---

## How to Use This Rubric

1. **After answering any question** from Rounds 1–5, come back here and check which signals your answer hit.
2. **For mock interviews**, have a partner grade you on each axis. A staff-level answer should hit at least 4 of the 6 axes at the staff level.
3. **For system design practice**, sketch an architecture on a whiteboard, then audit it against every row. The gaps reveal what you need to study next.

---

## Scoring Guide

| Score | Interpretation |
|---|---|
| **0–2 staff signals** | Solid mid-level. Focus on quantitative reasoning — put numbers on everything. |
| **3–4 staff signals** | Strong senior. You reason about trade-offs well. Push on failure modes and cost. |
| **5–6 staff signals** | Staff-ready. You think in constraints, not features. |

<br>
<p align="center">
  <img src="https://img.shields.io/badge/Status-Staff_Ready-success?style=for-the-badge" alt="Staff Ready Badge">
</p>
