# Machine Learning Systems: Two-Volume Structure

**Status**: Implemented
**Target Publisher**: MIT Press
**Audience**: Undergraduate and graduate CS/ECE students, academic courses

---

## Overview

This textbook is organized into two volumes following the Hennessy & Patterson pedagogical model:

- **Volume I: Build, Optimize, Operate** - Foundational knowledge for single-machine ML systems
- **Volume II: Scale, Distribute, Govern** - Advanced distributed systems at production scale

Each volume stands alone as a complete learning experience while together forming a comprehensive treatment of the field.

---

## Volume I: Build, Optimize, Operate

### Goal
A reader completes Volume I and can competently build, optimize, and deploy ML systems on a single machine with awareness of responsible practices.

### Target Audience
- Upper-level undergraduates
- Early graduate students
- Practitioners transitioning into ML systems

### Course Mapping
- Single semester "Introduction to ML Systems" course
- Foundation for more advanced distributed systems or MLOps courses

### Structure (15 chapters)

#### Part I: Foundations
Establish the conceptual framework for understanding ML as a systems discipline.

| Ch | Title | Purpose |
|----|-------|---------|
| 1 | Introduction | Why ML systems thinking matters |
| 2 | ML Systems | Survey of the field, deployment paradigms |
| 3 | Deep Learning Primer | Mathematical and conceptual foundations |
| 4 | DNN Architectures | CNNs, RNNs, Transformers, architectural choices |

#### Part II: Development
Practical skills for constructing ML systems from data to trained model.

| Ch | Title | Purpose |
|----|-------|---------|
| 5 | Workflow | End-to-end ML development process |
| 6 | Data Engineering | Pipelines, preprocessing, data quality |
| 7 | Frameworks | PyTorch, TensorFlow, JAX ecosystem |
| 8 | Training | Training loops, optimization, debugging |

#### Part III: Optimization
Techniques for making ML systems efficient and fast.

| Ch | Title | Purpose |
|----|-------|---------|
| 9 | Efficient AI | Why efficiency matters, scaling laws, metrics |
| 10 | Optimizations | Quantization, pruning, distillation |
| 11 | Hardware Acceleration | GPUs, TPUs, custom accelerators |
| 12 | Benchmarking | Measuring performance, MLPerf |

#### Part IV: Deployment
Getting models into production responsibly.

| Ch | Title | Purpose |
|----|-------|---------|
| 13 | ML Operations | Deployment, monitoring, CI/CD for ML |
| 14 | Responsible Engineering | Ethics, safety, and professional practice |
| 15 | Conclusion | Synthesis and bridge to Volume II |

---

## Volume II: Scale, Distribute, Govern

### Goal
A reader completes Volume II understanding how to build and operate ML systems at scale, with production resilience and responsible practices.

### Target Audience
- Graduate students
- Industry practitioners
- Researchers building large-scale systems

### Prerequisites
- Volume I or equivalent knowledge
- Basic distributed systems concepts helpful

### Course Mapping
- Graduate seminar on large-scale ML systems
- Advanced MLOps course
- Research group reading material

### Structure (16 chapters)

#### Part I: Foundations of Scale
Infrastructure and concepts for scaling beyond single machines.

| Ch | Title | Purpose |
|----|-------|---------|
| 1 | Introduction | Motivation, challenges of scale |
| 2 | Infrastructure | Clusters, cloud, resource management |
| 3 | Storage Systems | Data lakes, distributed storage, checkpointing |
| 4 | Communication | AllReduce, parameter servers, network topology |

#### Part II: Distributed Systems
Training and inference across multiple machines.

| Ch | Title | Purpose |
|----|-------|---------|
| 5 | Distributed Training | Parallelism strategies, multi-chip hardware, scaling infrastructure |
| 6 | Fault Tolerance | Checkpointing, recovery, handling failures |
| 7 | Inference at Scale | Serving systems, batching, latency optimization |
| 8 | Edge Intelligence | Federated learning, fleet coordination, on-device adaptation |

#### Part III: Production Challenges
Real-world complexities of operating ML systems.

| Ch | Title | Purpose |
|----|-------|---------|
| 9 | Privacy & Security | Differential privacy, secure computation, attacks |
| 10 | Robust AI | Adversarial robustness, distribution shift |
| 11 | ML Ops at Scale | Advanced MLOps, platform engineering |

#### Part IV: Responsible Deployment
Building ML systems that benefit society.

| Ch | Title | Purpose |
|----|-------|---------|
| 12 | Responsible AI | Fairness, accountability, transparency |
| 13 | Sustainable AI | Environmental impact, efficient computing |
| 14 | AI for Good | Applications for societal benefit |
| 15 | Frontiers | Emerging trends, open problems |
| 16 | Conclusion | Synthesis, future of the field |

---

## Key Design Decisions

### Why This Split?

1. **Pedagogical Progression**: Volume I covers what every ML practitioner needs. Volume II covers what scale/production engineers need.

2. **Course Adoptability**: Volume I maps to a single semester intro course. Volume II maps to an advanced graduate seminar.

3. **Standalone Completeness**: A reader of only Volume I gets responsible engineering awareness through Chapter 14.

4. **Industry Alignment**: Volume I produces capable junior engineers. Volume II produces senior/staff-level systems thinkers.

### The Hennessy & Patterson Test

When deciding where content belongs, ask: **What is the SCOPE of the system being discussed?**

| Aspect | Volume I | Volume II |
|--------|----------|-----------|
| **Scope** | Single-machine systems (1-8 GPUs) | Multi-machine distributed systems |
| **Math & Theory** | Full rigor, derivations | Full rigor, derivations |
| **Performance Metrics** | Single-system analysis | Scaling/efficiency analysis |
| **Code Examples** | Single-node implementations | Multi-node implementations |

---

## Summary Statistics

| Metric | Volume I | Volume II |
|--------|----------|-----------|
| Chapters | 15 | 16 |
| Parts | 4 | 4 |
| Focus | Single system | Distributed systems |
| Prerequisite | None | Volume I |

---

*Document Version: January 2025*
*Reflects current implementation in `_quarto-html.yml`*
