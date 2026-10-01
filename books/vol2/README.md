# Volume II: Scaling Machine Learning Systems

*Distributed systems and production infrastructure for ML.*

[![Read Online](https://img.shields.io/badge/Read_Online-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/vol2/)

---

> [!CAUTION]
> **This volume is under active development.** I am writing and revising chapters continuously. Diagrams, figures, and cross-references are being created and updated throughout. What you see here is a work in progress, not a finished product. I share it openly because I believe in transparent development. Expect the content to evolve significantly before the Summer 2027 release.

## About This Volume

Volume II moves from introductory single-machine ML systems to fleets of machines connected by high-speed networks. It covers the mathematical and algorithmic demand for scale, how to build the physical infrastructure that meets it, how to serve models to billions of users, and how to do all of this safely and responsibly.

The assumed foundation is local model execution, memory hierarchy, training and inference, and accelerator basics. The advanced material explains how those constraints change when many nodes coordinate through network fabrics, message passing, and fault tolerance across racks and datacenters. Security, privacy, environmental limits, and governance enter as design requirements throughout; the final part develops their enforcement mechanisms.

## What You Will Learn

<table width="100%">
    <thead>
    <tr>
      <th width="25%">Part</th>
      <th width="20%">Focus</th>
      <th width="55%">What You Will Learn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>I. The Fleet</b></td>
      <td>Physical infrastructure</td>
      <td>How to design compute clusters, high-speed network fabrics, and scalable data storage systems that form the foundation of distributed ML</td>
    </tr>
    <tr>
      <td><b>II. Distributed ML</b></td>
      <td>The logic of distributed coordination</td>
      <td>How to partition workloads with 3D parallelism, coordinate collective communication primitives, survive inevitable hardware failures, and orchestrate shared clusters</td>
    </tr>
    <tr>
      <td><b>III. Deployment at Scale</b></td>
      <td>Serving and operations</td>
      <td>How to diagnose bottlenecks and improve runtime performance, serve models under tail-latency SLOs, deploy to edge devices, and manage production MLOps lifecycles</td>
    </tr>
    <tr>
      <td><b>IV. The Responsible Fleet</b></td>
      <td>Governance and boundaries</td>
      <td>How to secure ML systems against adversaries, evaluate and improve empirical robustness under drift, operate within planetary carbon limits, and govern models responsibly</td>
    </tr>
  </tbody>
</table>

## Chapter Map

<table width="100%">
    <thead>
    <tr>
      <th width="5%">#</th>
      <th width="22%">Chapter</th>
      <th width="22%">Directory</th>
      <th width="51%">Core Question</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><b>Introduction to Scale</b></td>
      <td><code>01_introduction/</code></td>
      <td>Why do single-machine systems break down at production scale?</td>
    </tr>
    <tr>
      <td>2</td>
      <td><b>Compute Infrastructure</b></td>
      <td><code>02_compute_infrastructure/</code></td>
      <td>How do you build and provision GPU clusters and thermal envelopes?</td>
    </tr>
    <tr>
      <td>3</td>
      <td><b>Network Fabrics</b></td>
      <td><code>03_network_fabrics/</code></td>
      <td>How does data move between machines across high-speed fabrics?</td>
    </tr>
    <tr>
      <td>4</td>
      <td><b>Data Storage</b></td>
      <td><code>04_data_storage/</code></td>
      <td>How do you feed training pipelines and persist checkpoints without starving accelerators?</td>
    </tr>
    <tr>
      <td>5</td>
      <td><b>Distributed Training</b></td>
      <td><code>05_distributed_training/</code></td>
      <td>How do you partition models across thousands of accelerators?</td>
    </tr>
    <tr>
      <td>6</td>
      <td><b>Collective Communication</b></td>
      <td><code>06_collective_communication/</code></td>
      <td>How do machines coordinate and exchange state during distributed training?</td>
    </tr>
    <tr>
      <td>7</td>
      <td><b>Fault Tolerance</b></td>
      <td><code>07_fault_tolerance/</code></td>
      <td>Why is hardware failure routine, and how do you recover without losing progress?</td>
    </tr>
    <tr>
      <td>8</td>
      <td><b>Fleet Orchestration</b></td>
      <td><code>08_fleet_orchestration/</code></td>
      <td>How do you schedule, pack, and isolate multi-tenant workloads across shared clusters?</td>
    </tr>
    <tr>
      <td>9</td>
      <td><b>Performance Engineering</b></td>
      <td><code>09_performance_engineering/</code></td>
      <td>How do you close the gap between hardware peak and serving throughput?</td>
    </tr>
    <tr>
      <td>10</td>
      <td><b>Inference at Scale</b></td>
      <td><code>10_inference/</code></td>
      <td>How do you serve models to millions of concurrent requests under latency SLOs?</td>
    </tr>
    <tr>
      <td>11</td>
      <td><b>Edge Intelligence</b></td>
      <td><code>11_edge_intelligence/</code></td>
      <td>How do you adapt and coordinate models on resource-constrained edge devices?</td>
    </tr>
    <tr>
      <td>12</td>
      <td><b>Ops at Scale</b></td>
      <td><code>12_ops_scale/</code></td>
      <td>How do you monitor, version, and operate fleets of production models?</td>
    </tr>
    <tr>
      <td>13</td>
      <td><b>Security & Privacy</b></td>
      <td><code>13_security_privacy/</code></td>
      <td>How do you defend ML systems against adversarial attacks and preserve privacy?</td>
    </tr>
    <tr>
      <td>14</td>
      <td><b>Robust AI</b></td>
      <td><code>14_robust_ai/</code></td>
      <td>How do you make ML systems reliable under distribution drift and silent failures?</td>
    </tr>
    <tr>
      <td>15</td>
      <td><b>Sustainable AI</b></td>
      <td><code>15_sustainable_ai/</code></td>
      <td>How do you measure and reduce the energy and carbon footprint of the fleet?</td>
    </tr>
    <tr>
      <td>16</td>
      <td><b>Responsible AI</b></td>
      <td><code>16_responsible_ai/</code></td>
      <td>How do you govern ML systems fairly, transparently, and accountably?</td>
    </tr>
    <tr>
      <td>17</td>
      <td><b>Conclusion</b></td>
      <td><code>17_conclusion/</code></td>
      <td>Why must the unit of machine learning systems design become the fleet?</td>
    </tr>
  </tbody>
</table>

## Prerequisites

Volume II is a standalone book. It assumes an introductory ML systems course or equivalent experience with single-machine execution, memory hierarchy, training and inference pipelines, and hardware acceleration. No other book is required. Appendix A recaps the single-machine performance model (the iron law, arithmetic intensity, and the roofline), and the book introduces its own C³ framework for reasoning about fleets.

## Working in the Open

I develop this volume in the open because I believe it produces a better textbook. Every commit is visible, every editorial decision is traceable. If something looks rough, that is because you are watching the book being written.

If you notice an error, have a suggestion, or want to propose a topic, please [open an issue](https://github.com/harvard-edge/cs249r_book/issues) or [start a discussion](https://github.com/harvard-edge/cs249r_book/discussions). Feedback on structure, missing topics, examples, and clarity is especially valuable at this stage. The book is better for every reader who engages with it.

## Links

<table width="100%">
    <thead>
    <tr>
      <th width="25%">Resource</th>
      <th width="75%">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Read online</b></td>
      <td><a href="https://mlsysbook.ai/vol2/">mlsysbook.ai/vol2</a></td>
    </tr>
    <tr>
      <td><b>Volume I</b></td>
      <td><a href="../vol1/">books/vol1/</a></td>
    </tr>
    <tr>
      <td><b>Main README</b></td>
      <td><a href="../../README.md">Repository root</a></td>
    </tr>
    <tr>
      <td><b>Full textbook</b></td>
      <td><a href="https://mlsysbook.ai/book/">mlsysbook.ai/book</a></td>
    </tr>
    <tr>
      <td><b>Discussions</b></td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/discussions">GitHub Discussions</a></td>
    </tr>
  </tbody>
</table>
