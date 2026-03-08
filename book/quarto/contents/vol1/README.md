# Volume I: Introduction to Machine Learning Systems

*Foundations for single-machine ML systems.*

[![Read Online](https://img.shields.io/badge/Read_Online-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/vol1/)

---

## About This Volume

Volume I teaches how to build, optimize, and deploy machine learning systems on a single machine with one to eight accelerators. It covers the full stack from data engineering through model serving, grounding every concept in the physical constraints of real hardware: memory hierarchies, compute throughput, and power budgets.

This is the foundational volume. It establishes the quantitative frameworks (the Iron Law of ML Systems, the D.A.M Taxonomy) that Volume II builds on when scaling to fleets of machines.

## Status

Volume I content is **complete and undergoing final editorial polish.** It is ready for classroom use. Chapters are being reviewed for prose quality, figure consistency, and cross-reference accuracy, but the technical content is stable.

## What You Will Learn

| Part | Focus | What You Will Learn |
|------|-------|---------------------|
| **I. Foundations** | Core concepts | How ML systems differ from traditional software, the Iron Law framework, data engineering at scale |
| **II. Development** | Building blocks | Neural computation, model architectures, framework internals, training systems |
| **III. Optimization** | Making it fast | Data selection, model compression, hardware acceleration, benchmarking methodology |
| **IV. Deployment** | Making it work | Model serving, MLOps, responsible engineering |

## Chapter Map

| # | Chapter | Directory | Core Question |
|---|---------|-----------|---------------|
| 1 | Introduction | `introduction/` | What are ML systems and why do they need engineering? |
| 2 | ML Systems | `ml_systems/` | How do deployment constraints shape system design? |
| 3 | ML Workflow | `ml_workflow/` | What does the end-to-end ML pipeline look like? |
| 4 | Data Engineering | `data_engineering/` | How do you build data pipelines that feed ML? |
| 5 | Neural Computation | `nn_computation/` | How does math become silicon operations? |
| 6 | Architectures | `nn_architectures/` | How do CNNs, RNNs, and Transformers differ as systems? |
| 7 | Frameworks | `frameworks/` | How do PyTorch and TensorFlow actually work inside? |
| 8 | Training | `training/` | How do you train models efficiently on real hardware? |
| 9 | Data Selection | `data_selection/` | How do you choose the right data for training? |
| 10 | Model Compression | `optimizations/` | How do you shrink models without losing accuracy? |
| 11 | HW Acceleration | `hw_acceleration/` | How do GPUs, TPUs, and accelerators execute neural networks? |
| 12 | Benchmarking | `benchmarking/` | How do you measure and compare ML system performance? |
| 13 | Model Serving | `model_serving/` | How do you deploy models to serve real users? |
| 14 | MLOps | `ml_ops/` | How do you keep ML systems running in production? |
| 15 | Responsible Engineering | `responsible_engr/` | How do you build ML systems that are fair, safe, and trustworthy? |

## Prerequisites

Volume I assumes a CS/EE undergraduate background: operating systems, computer architecture, data structures and algorithms, linear algebra, calculus, and basic probability.

## How to Read

The chapters are designed to be read in order. Each chapter builds on the frameworks and vocabulary established in prior chapters. The Iron Law introduced in the opening chapter is used throughout the entire volume to reason about performance bottlenecks.

If you are an instructor adopting this for a course, the natural division is:
- **Parts I and II** (Chapters 1 through 8) for a foundations course
- **Parts III and IV** (Chapters 9 through 15) for an optimization and deployment course

## Feedback

If you spot an error, find an explanation that could be clearer, or have a suggestion, please [open an issue](https://github.com/harvard-edge/cs249r_book/issues) or [start a discussion](https://github.com/harvard-edge/cs249r_book/discussions). Even small corrections make the book better for every reader.

## Links

| Resource | Link |
|----------|------|
| Read online | [mlsysbook.ai/vol1](https://mlsysbook.ai/vol1/) |
| Volume II | [book/quarto/contents/vol2/](../vol2/) |
| Main README | [Repository root](../../../../README.md) |
| Full textbook | [mlsysbook.ai/book](https://mlsysbook.ai/book/) |
| Discussions | [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) |
