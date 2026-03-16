# Round 6: The Architect's Rubric 📋

<div align="center">
  <a href="README.md">🏠 Hub Home</a> |
  <a href="00_The_Architects_Rubric.md">📋 The Rubric</a> |
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> |
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> |
  <a href="03_Production_Serving.md">⚡ Round 3</a> |
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> |
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---


How do Principal Engineers grade your system design interviews? They don't look for a single "correct" architecture; they evaluate you across multiple axes of system maturity.

Use this interactive rubric to self-evaluate your proposed designs.

> **[➕ Propose a new Rubric Axis](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/06_System_Design_Rubric.md)** (Edit in Browser)

---

## The Evaluation Matrix

Hover over the tooltip icons `(?)` or click the checkmarks to see the criteria used by top-tier interviewers.

<table>
  <thead>
    <tr>
      <th width="20%">Engineering Axis</th>
      <th width="40%">Mid-Level Signal (L4)</th>
      <th width="40%">Staff-Level Signal (L6+)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Data Gravity <kbd>?</kbd></b></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Understands storage I/O limits</li>
          <li><input type="checkbox" disabled> Uses batching for throughput</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Calculates exact network transfer times</li>
          <li><input type="checkbox" disabled checked> Designs zero-copy inference pipelines</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Failure Modes <kbd>?</kbd></b></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Handles node crashes</li>
          <li><input type="checkbox" disabled> Implements retry logic</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Mitigates silent data corruption (SDC)</li>
          <li><input type="checkbox" disabled checked> Calculates MTBF using Young-Daly</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td><b>Cost & TCO <kbd>?</kbd></b></td>
      <td>
        <ul>
          <li><input type="checkbox" disabled> Chooses cheaper instances</li>
          <li><input type="checkbox" disabled> Understands spot-instance risks</li>
        </ul>
      </td>
      <td>
        <ul>
          <li><input type="checkbox" disabled checked> Formulates exact Retraining Staleness math</li>
          <li><input type="checkbox" disabled checked> Factors PUE and cooling into fleet TCO</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

---

## 🎨 Keyboard Shortcuts & Annotations

When writing out your system design on a virtual whiteboard, use standard notation to signal intent quickly:

*   <kbd>Shift</kbd> + <kbd>C</kbd> : Denote a Compute Bottleneck
*   <kbd>Shift</kbd> + <kbd>M</kbd> : Denote a Memory Bottleneck
*   <kbd>Shift</kbd> + <kbd>N</kbd> : Denote a Network Bottleneck

<br>
<p align="center">
  <img src="https://img.shields.io/badge/Status-Staff_Ready-success?style=for-the-badge" alt="Staff Ready Badge">
</p>
