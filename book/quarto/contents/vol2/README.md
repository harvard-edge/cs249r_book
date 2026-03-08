# Volume II: Machine Learning Systems at Scale

*Distributed systems and production infrastructure for ML.*

[![Read Online](https://img.shields.io/badge/Read_Online-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/vol2/)

---

> [!CAUTION]
> **This volume is under active development.** I am writing and revising chapters continuously. Diagrams, figures, and cross-references are being created and updated throughout. What you see here is a work in progress, not a finished product. I share it openly because I believe in transparent development. Expect the content to evolve significantly before the Summer 2026 release.

## About This Volume

Volume II picks up where Volume I ends, moving from a single machine to fleets of machines connected by high speed networks. It covers the mathematical and algorithmic demand for scale, how to build the physical infrastructure that meets it, how to serve models to billions of users, and how to do all of this safely and responsibly.

Where Volume I teaches you to optimize a single node (one to eight accelerators, shared memory, PCIe/NVLink within one box), Volume II teaches you to orchestrate many nodes (hundreds to thousands of accelerators, InfiniBand/Ethernet fabric, message passing, fault tolerance across racks and datacenters).

## What You Will Learn

| Part | Focus | What You Will Learn |
|------|-------|---------------------|
| **I. Foundations of Scale** | The logic of distributed systems | Why single machines cannot keep up, how parallelism strategies work, collective communication primitives, and what happens when hardware fails |
| **II. Building the Fleet** | Physical infrastructure | How to design compute clusters, network fabrics, storage systems, and orchestration layers for ML workloads |
| **III. Deployment at Scale** | Serving at global scale | How to serve models to billions of users, engineer for performance, deploy at the edge, and operate ML infrastructure |
| **IV. Production Concerns** | Safety and governance | How to secure ML systems, make them robust, sustainable, and responsible at production scale |

## Chapter Map

| # | Chapter | Directory | Core Question |
|---|---------|-----------|---------------|
| 1 | Introduction to Scale | `introduction/` | Why does ML demand distributed systems? |
| 2 | Distributed Training | `distributed_training/` | How do you split training across many machines? |
| 3 | Collective Communication | `collective_communication/` | How do machines coordinate during distributed training? |
| 4 | Fault Tolerance | `fault_tolerance/` | What happens when hardware fails mid-training? |
| 5 | Compute Infrastructure | `compute_infrastructure/` | How do you build and provision GPU clusters? |
| 6 | Network Fabrics | `network_fabrics/` | How does data move between machines at scale? |
| 7 | Data Storage | `data_storage/` | How do you store and access training data for distributed workloads? |
| 8 | Fleet Orchestration | `fleet_orchestration/` | How do you schedule and manage thousands of accelerators? |
| 9 | Inference at Scale | `inference/` | How do you serve models to billions of requests? |
| 10 | Performance Engineering | `performance_engineering/` | How do you find and fix bottlenecks in distributed systems? |
| 11 | Edge Intelligence | `edge_intelligence/` | How do you deploy ML on devices at the network edge? |
| 12 | Ops at Scale | `ops_scale/` | How do you monitor and operate ML infrastructure? |
| 13 | Security and Privacy | `security_privacy/` | How do you protect ML systems from attacks and preserve privacy? |
| 14 | Robust AI | `robust_ai/` | How do you make ML systems reliable and verifiable? |
| 15 | Sustainable AI | `sustainable_ai/` | How do you reduce the environmental cost of ML at scale? |
| 16 | Responsible AI | `responsible_ai/` | How do you govern ML systems fairly and accountably? |

## Prerequisites

Volume II assumes you have read Volume I or have equivalent knowledge of single-machine ML systems: the Iron Law of ML Systems, the D.A.M Taxonomy, training and inference pipelines, model compression, and hardware acceleration fundamentals.

## Working in the Open

I develop this volume in the open because I believe it produces a better textbook. Every commit is visible, every editorial decision is traceable. If something looks rough, that is because you are watching the book being written.

If you notice an error, have a suggestion, or want to contribute, open an issue or start a discussion. The book is better for every reader who engages with it.

## Links

| Resource | Link |
|----------|------|
| Read online | [mlsysbook.ai/vol2](https://mlsysbook.ai/vol2/) |
| Volume I | [book/quarto/contents/vol1/](../vol1/) |
| Main README | [Repository root](../../../../README.md) |
| Full textbook | [mlsysbook.ai/book](https://mlsysbook.ai/book/) |
| Discussions | [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) |
