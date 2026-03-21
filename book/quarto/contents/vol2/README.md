# Volume II: Machine Learning Systems at Scale

*Distributed systems and production infrastructure for ML.*

[![Read Online](https://img.shields.io/badge/Read_Online-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/vol2/)

---

> [!CAUTION]
> **This volume is under active development.** I am writing and revising chapters continuously. Diagrams, figures, and cross-references are being created and updated throughout. What you see here is a work in progress, not a finished product. I share it openly because I believe in transparent development. Expect the content to evolve significantly before the Summer 2026 release.

## About This Volume

Volume II picks up where Volume I ends, moving from a single machine to fleets of machines connected by high-speed networks. It covers the mathematical and algorithmic demand for scale, how to build the physical infrastructure that meets it, how to serve models to billions of users, and how to do all of this safely and responsibly.

Where Volume I teaches you to optimize a single node (one to eight accelerators, shared memory, PCIe/NVLink within one box), Volume II teaches you to orchestrate many nodes (hundreds to thousands of accelerators, InfiniBand/Ethernet fabric, message passing, fault tolerance across racks and datacenters).

## What You Will Learn

<table>
  <thead>
    <tr>
      <th width="25%">Part</th>
      <th width="20%">Focus</th>
      <th width="55%">What You Will Learn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>I. Foundations of Scale</b></td>
      <td>The logic of distributed systems</td>
      <td>Why single machines cannot keep up, how parallelism strategies work, collective communication primitives, and what happens when hardware fails</td>
    </tr>
    <tr>
      <td><b>II. Building the Fleet</b></td>
      <td>Physical infrastructure</td>
      <td>How to design compute clusters, network fabrics, storage systems, and orchestration layers for ML workloads</td>
    </tr>
    <tr>
      <td><b>III. Deployment at Scale</b></td>
      <td>Serving at global scale</td>
      <td>How to serve models to billions of users, engineer for performance, deploy at the edge, and operate ML infrastructure</td>
    </tr>
    <tr>
      <td><b>IV. Production Concerns</b></td>
      <td>Safety and governance</td>
      <td>How to secure ML systems, make them robust, sustainable, and responsible at production scale</td>
    </tr>
  </tbody>
</table>

## Chapter Map

<table>
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
      <td><code>introduction/</code></td>
      <td>Why does ML demand distributed systems?</td>
    </tr>
    <tr>
      <td>2</td>
      <td><b>Distributed Training</b></td>
      <td><code>distributed_training/</code></td>
      <td>How do you split training across many machines?</td>
    </tr>
    <tr>
      <td>3</td>
      <td><b>Collective Communication</b></td>
      <td><code>collective_communication/</code></td>
      <td>How do machines coordinate during distributed training?</td>
    </tr>
    <tr>
      <td>4</td>
      <td><b>Fault Tolerance</b></td>
      <td><code>fault_tolerance/</code></td>
      <td>What happens when hardware fails mid-training?</td>
    </tr>
    <tr>
      <td>5</td>
      <td><b>Compute Infrastructure</b></td>
      <td><code>compute_infrastructure/</code></td>
      <td>How do you build and provision GPU clusters?</td>
    </tr>
    <tr>
      <td>6</td>
      <td><b>Network Fabrics</b></td>
      <td><code>network_fabrics/</code></td>
      <td>How does data move between machines at scale?</td>
    </tr>
    <tr>
      <td>7</td>
      <td><b>Data Storage</b></td>
      <td><code>data_storage/</code></td>
      <td>How do you store and access training data for distributed workloads?</td>
    </tr>
    <tr>
      <td>8</td>
      <td><b>Fleet Orchestration</b></td>
      <td><code>fleet_orchestration/</code></td>
      <td>How do you schedule and manage thousands of accelerators?</td>
    </tr>
    <tr>
      <td>9</td>
      <td><b>Inference at Scale</b></td>
      <td><code>inference/</code></td>
      <td>How do you serve models to billions of requests?</td>
    </tr>
    <tr>
      <td>10</td>
      <td><b>Performance Engineering</b></td>
      <td><code>performance_engineering/</code></td>
      <td>How do you find and fix bottlenecks in distributed systems?</td>
    </tr>
    <tr>
      <td>11</td>
      <td><b>Edge Intelligence</b></td>
      <td><code>edge_intelligence/</code></td>
      <td>How do you deploy ML on devices at the network edge?</td>
    </tr>
    <tr>
      <td>12</td>
      <td><b>Ops at Scale</b></td>
      <td><code>ops_scale/</code></td>
      <td>How do you monitor and operate ML infrastructure?</td>
    </tr>
    <tr>
      <td>13</td>
      <td><b>Security and Privacy</b></td>
      <td><code>security_privacy/</code></td>
      <td>How do you protect ML systems from attacks and preserve privacy?</td>
    </tr>
    <tr>
      <td>14</td>
      <td><b>Robust AI</b></td>
      <td><code>robust_ai/</code></td>
      <td>How do you make ML systems reliable and verifiable?</td>
    </tr>
    <tr>
      <td>15</td>
      <td><b>Sustainable AI</b></td>
      <td><code>sustainable_ai/</code></td>
      <td>How do you reduce the environmental cost of ML at scale?</td>
    </tr>
    <tr>
      <td>16</td>
      <td><b>Responsible AI</b></td>
      <td><code>responsible_ai/</code></td>
      <td>How do you govern ML systems fairly and accountably?</td>
    </tr>
  </tbody>
</table>

## Prerequisites

Volume II assumes you have read Volume I or have equivalent knowledge of single-machine ML systems: the Iron Law of ML Systems, the D.A.M Taxonomy, training and inference pipelines, model compression, and hardware acceleration fundamentals.

## Working in the Open

I develop this volume in the open because I believe it produces a better textbook. Every commit is visible, every editorial decision is traceable. If something looks rough, that is because you are watching the book being written.

If you notice an error, have a suggestion, or want to propose a topic, please [open an issue](https://github.com/harvard-edge/cs249r_book/issues) or [start a discussion](https://github.com/harvard-edge/cs249r_book/discussions). Feedback on structure, missing topics, examples, and clarity is especially valuable at this stage. The book is better for every reader who engages with it.

## Links

<table>
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
      <td><a href="../vol1/">book/quarto/contents/vol1/</a></td>
    </tr>
    <tr>
      <td><b>Main README</b></td>
      <td><a href="../../../../README.md">Repository root</a></td>
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
