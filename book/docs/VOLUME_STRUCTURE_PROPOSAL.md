# Machine Learning Systems: Two-Volume Structure

**Proposal for MIT Press**
*Draft: December 2024*

---

## Executive Summary

The *Machine Learning Systems* textbook will be published as two complementary volumes of 14 chapters each:

| Volume | Title | Focus | Chapters |
|--------|-------|-------|----------|
| **Volume 1** | Introduction to Machine Learning Systems | Complete ML lifecycle, single-system focus | 14 (all existing) |
| **Volume 2** | Advanced Machine Learning Systems | Principles of scale, distribution, and production | 14 (6 existing, 8 new) |

**Guiding Philosophy:**
- **Volume 1**: Everything you need to build ML systems on a single machine, ending on a positive note with societal impact
- **Volume 2**: Timeless principles for operating ML systems at scale, grounded in physics and mathematics rather than current technologies

---

## Volume 1: Introduction to Machine Learning Systems

*The complete ML lifecycle: understand it, build it, optimize it, deploy it, use it for good.*

| Part | Chapter | Description |
|------|---------|-------------|
| **Part I: Systems Foundations** | | *What are ML systems?* |
| | 1. Introduction | Motivation and scope |
| | 2. ML Systems | System-level view of machine learning |
| | 3. Deep Learning Primer | Neural network fundamentals |
| | 4. DNN Architectures | Modern architecture patterns |
| **Part II: Design Principles** | | *How do you build ML systems?* |
| | 5. Workflow | End-to-end ML pipeline design |
| | 6. Data Engineering | Data collection, processing, validation |
| | 7. Frameworks | PyTorch, TensorFlow, JAX ecosystem |
| | 8. Training | Training loops, hyperparameters, convergence |
| **Part III: Performance Engineering** | | *How do you make ML systems fast?* |
| | 9. Efficient AI | Efficiency principles and metrics |
| | 10. Optimizations | Quantization, pruning, distillation |
| | 11. Hardware Acceleration | GPUs, TPUs, custom accelerators |
| | 12. Benchmarking | Measurement, MLPerf, evaluation methodology |
| **Part IV: Practice & Impact** | | *How do you deploy and use ML systems responsibly?* |
| | 13. ML Operations | Deployment, monitoring, CI/CD for ML |
| | 14. AI for Good | Positive societal applications |

**Total: 14 chapters across 4 parts (all existing content)**

*Early awareness:* include a short Sustainable AI note in Benchmarking or ML Operations to flag energy and carbon impacts without adding another chapter.

### Volume 1 Narrative Arc

The book progresses from understanding → building → optimizing → deploying → impact:

1. **Foundations** establish what ML systems are and why they matter
2. **Design** teaches how to construct complete pipelines
3. **Performance** shows how to make systems efficient
4. **Practice & Impact** completes the lifecycle and ends on an inspirational note

Ending on "AI for Good" leaves students with a positive vision of what they can build.

---

## Volume 2: Advanced Machine Learning Systems

*Timeless principles for building and operating ML systems at scale.*

| Part | Chapter | Status | Description |
|------|---------|--------|-------------|
| **Part I: Data Movement & Memory** | | | *Moving data is the bottleneck* |
| | 1. Memory Hierarchies for ML | 🆕 NEW | GPU memory, HBM, activation checkpointing |
| | 2. Storage Systems for ML | 🆕 NEW | Distributed storage, checkpointing, feature stores |
| | 3. Communication & Collective Operations | 🆕 NEW | AllReduce, gradient compression, network topology |
| **Part II: Parallelism & Coordination** | | | *Decomposing computation across machines* |
| | 4. Distributed Training | 🆕 NEW | Data/model/pipeline/tensor parallelism |
| | 5. Fault Tolerance & Recovery | 🆕 NEW | Checkpointing, elastic training, failure handling |
| | 6. Inference Systems | 🆕 NEW | Batching, serving architectures, autoscaling |
| **Part III: Constrained Environments** | | | *Doing more with less* |
| | 7. On-device Learning | Existing | Training and adaptation on edge devices |
| | 8. Edge Deployment | 🆕 NEW | Compilation, runtime optimization, real-time |
| **Part IV: Adversarial Environments** | | | *Systems under attack and uncertainty* |
| | 9. Privacy in ML Systems | Existing | Differential privacy, federated learning, secure aggregation |
| | 10. Security in ML Systems | 🆕 NEW | Supply chain, API security, multi-tenant isolation |
| | 11. Robust AI | Existing | Adversarial robustness, distribution shift, monitoring |
| **Part V: Stewardship** | | | *Building systems that serve humanity* |
| | 12. Responsible AI | Existing | Fairness, accountability, transparency at scale |
| | 13. Sustainable AI | Existing | Energy efficiency, carbon footprint, environmental impact |
| | 14. Frontiers & Future Directions | Existing | Emerging paradigms, open problems, conclusion |

**Total: 14 chapters across 5 parts (6 existing, 8 new)**

---

## New Content for Volume 2

### Part I: Data Movement & Memory
*The physics of data movement is the fundamental constraint in modern ML.*

| Chapter | Key Topics | Timeless Principle |
|---------|------------|-------------------|
| **Memory Hierarchies for ML** | GPU memory management, HBM architecture, caching strategies, activation checkpointing, memory-efficient attention | Memory bandwidth limits compute utilization |
| **Storage Systems for ML** | Distributed file systems, checkpoint I/O, feature stores, data lakes, prefetching, I/O scheduling | Storage throughput gates training speed |
| **Communication & Collective Operations** | AllReduce algorithms, ring/tree topologies, gradient compression, RDMA fundamentals, network topology design | Communication overhead limits scaling |

### Part II: Parallelism & Coordination
*The mathematics of decomposing work across machines.*

| Chapter | Key Topics | Timeless Principle |
|---------|------------|-------------------|
| **Distributed Training** | Data parallelism, model parallelism (tensor, pipeline, expert), hybrid strategies, synchronization, load balancing | Parallelism has fundamental trade-offs |
| **Fault Tolerance & Recovery** | Checkpoint strategies, async checkpointing, elastic training, failure detection, graceful degradation | Large systems fail; recovery must be designed in |
| **Inference Systems** | Batching strategies, continuous batching, KV cache management, model serving patterns, autoscaling, SLO management | Serving has different constraints than training |

### Part III: Constrained Environments
*Operating under resource limitations.*

| Chapter | Key Topics | Timeless Principle |
|---------|------------|-------------------|
| **Edge Deployment** | Model compilation, runtime optimization, heterogeneous hardware, real-time constraints, power management | Constraints force creativity |

### Part IV: Adversarial Environments
*Systems facing attacks, privacy requirements, and uncertainty.*

| Chapter | Key Topics | Timeless Principle |
|---------|------------|-------------------|
| **Security in ML Systems** | Model provenance, supply chain security, API protection, multi-tenant isolation, access control | Production systems face adversaries |

---

## Design Principles

### Why This Structure Works

**Volume 1 (Single System)**
- Teaches the complete lifecycle
- Everything can be learned and practiced on one machine
- Ends positively with societal impact

**Volume 2 (Distributed Systems)**
- Builds on Volume 1 foundations
- Addresses what changes at scale
- Organized around timeless constraints, not current technologies

### What Makes Volume 2 Timeless

Each part addresses constraints rooted in physics, mathematics, or human nature:

| Part | Eternal Constraint | Foundation |
|------|-------------------|------------|
| Data Movement & Memory | Moving data costs more than compute | Physics: speed of light, memory bandwidth |
| Parallelism & Coordination | Work must be decomposed and synchronized | Mathematics of parallel computation |
| Constrained Environments | Resources are always finite | Economics and physics |
| Adversarial Environments | Attackers and uncertainty exist | Human nature, statistics |
| Stewardship | Technology must serve humanity | Ethics, sustainability |

Chapters use current examples (LLMs, transformers, specific hardware) but frame them as instances of these enduring principles.

---

## Content Migration Summary

| Chapter | Volume 1 | Volume 2 | Rationale |
|---------|----------|----------|-----------|
| Introduction through Benchmarking | ✓ | | Core technical content |
| ML Operations | ✓ | | Completes the lifecycle |
| AI for Good | ✓ | | Positive conclusion |
| On-device Learning | | ✓ | Edge/constrained is advanced |
| Privacy & Security | | ✓ | Production security is advanced |
| Robust AI | | ✓ | Production robustness is advanced |
| Responsible AI | | ✓ | Scale changes the challenges |
| Sustainable AI | | ✓ | Datacenter scale is advanced |
| Frontiers | | ✓ | Conclusion for advanced volume |

---

## Audience

| Volume | Primary Audience | Use Cases |
|--------|-----------------|-----------|
| Volume 1 | All ML practitioners, undergraduates, bootcamp students | First course in ML systems, self-study |
| Volume 2 | Infrastructure engineers, graduate students, researchers | Advanced course, reference for practitioners at scale |

---

## Collaboration Model

Volume 2's new chapters are candidates for collaborative authorship:

| Topic Area | Ideal Collaborator Profile |
|------------|---------------------------|
| Memory & Storage | Datacenter architects, MLPerf Storage contributors |
| Networking & Communication | Distributed systems researchers, framework developers |
| Distributed Training | PyTorch/JAX distributed teams, hyperscaler engineers |
| Fault Tolerance | Site reliability engineers, systems researchers |
| Inference Systems | ML serving infrastructure engineers |
| Edge Deployment | Embedded ML practitioners, compiler engineers |
| Security | ML security researchers, production security engineers |

---

## Summary Statistics

| Metric | Volume 1 | Volume 2 |
|--------|----------|----------|
| Chapters | 14 | 14 |
| Parts | 4 | 5 |
| Existing content | 14 | 6 |
| New content | 0 | 8 |
| Focus | Single system | Distributed systems |
| Prerequisite | None | Volume 1 |

---

*Document Version: December 2024*
*For discussion with MIT Press and potential collaborators*
