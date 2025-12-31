# Machine Learning Systems: Two-Volume Structure

**Status**: Approved (Round 2 Review Complete)
**Target Publisher**: MIT Press
**Audience**: Undergraduate and graduate CS/ECE students, academic courses

---

## Overview

This textbook is being split into two volumes to serve different learning objectives:

- **Volume I: Introduction to ML Systems** - Foundational knowledge for building, optimizing, and deploying ML systems
- **Volume II: Advanced ML Systems** - Scale, distributed systems, production hardening, and responsible deployment

Each volume should stand alone as a complete learning experience while together forming a comprehensive treatment of the field.

---

## Volume I: Introduction to ML Systems

### Goal
A reader completes Volume I and can competently build, optimize, and deploy ML systems with awareness of responsible practices.

### Target Audience
- Upper-level undergraduates
- Early graduate students
- Practitioners transitioning into ML systems

### Course Mapping
- Single semester "Introduction to ML Systems" course
- Foundation for more advanced distributed systems or MLOps courses

### Structure (14 chapters)

#### Part I: Foundations
Establish the conceptual framework for understanding ML as a systems discipline.

| Ch | Title | Purpose |
|----|-------|---------|
| 1 | Introduction to Machine Learning Systems | Why ML systems thinking matters |
| 2 | The ML Systems Landscape | Survey of the field, key components |
| 3 | Deep Learning Foundations | Mathematical and conceptual foundations |
| 4 | Modern Neural Architectures | CNNs, RNNs, Transformers, architectural choices |

#### Part II: Development
Practical skills for constructing ML systems from data to trained model.

| Ch | Title | Purpose |
|----|-------|---------|
| 5 | ML Development Workflow | End-to-end process, experimentation |
| 6 | Data Engineering for ML | Pipelines, preprocessing, data quality |
| 7 | ML Frameworks and Tools | PyTorch, TensorFlow, ecosystem |
| 8 | Training Systems | Training loops, distributed basics, debugging |

#### Part III: Optimization
Techniques for making ML systems efficient and fast.

| Ch | Title | Purpose |
|----|-------|---------|
| 9 | Efficiency in AI Systems | Why efficiency matters, metrics |
| 10 | Model Optimization Techniques | Quantization, pruning, distillation |
| 11 | Hardware Acceleration | GPUs, TPUs, custom accelerators |
| 12 | Benchmarking and Evaluation | Measuring performance, MLPerf |

#### Part IV: Operations
Getting models into production responsibly.

| Ch | Title | Purpose |
|----|-------|---------|
| 13 | ML Operations Fundamentals | Deployment, monitoring, CI/CD for ML |
| 14 | Responsible Systems Preview | Brief intro to robustness, security, fairness, sustainability (preview of Vol II topics) |

#### Volume I Conclusion
Synthesis, what was learned, bridge to Volume II.

---

## Volume II: Advanced ML Systems

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
| 1 | From Single Systems to Planetary Scale | Motivation, challenges of scale |
| 2 | Infrastructure for Large-Scale ML | Clusters, cloud, resource management |
| 3 | Storage Systems for ML | Data lakes, distributed storage, checkpointing |
| 4 | Communication and Collective Operations | AllReduce, parameter servers, network topology |

#### Part II: Distributed Systems
Training and inference across multiple machines.

| Ch | Title | Purpose |
|----|-------|---------|
| 5 | Distributed Training Systems | Data parallel, model parallel, pipeline parallel |
| 6 | Fault Tolerance and Resilience | Checkpointing, recovery, handling failures |
| 7 | Inference at Scale | Serving systems, batching, latency optimization |
| 8 | Edge Intelligence Systems | Deploying ML at the edge, constraints |

#### Part III: Production Challenges
Real-world complexities of operating ML systems.

| Ch | Title | Purpose |
|----|-------|---------|
| 9 | On-Device Learning | Training on edge devices, federated learning |
| 10 | Privacy-Preserving ML Systems | Differential privacy, secure computation |
| 11 | Robust and Reliable AI | Adversarial robustness, distribution shift |
| 12 | ML Operations at Scale | Advanced MLOps, platform engineering |

#### Part IV: Responsible Deployment
Building ML systems that benefit society.

| Ch | Title | Purpose |
|----|-------|---------|
| 13 | Responsible AI Systems | Fairness, accountability, transparency |
| 14 | Sustainable AI | Environmental impact, efficient computing |
| 15 | AI for Good | Applications for societal benefit |
| 16 | Frontiers and Future Directions | Emerging trends, open problems |

#### Volume II Conclusion
Synthesis, future of the field, call to action.

---

## Key Design Decisions

### Why This Split?

1. **Pedagogical Progression**: Vol I covers what every ML practitioner needs. Vol II covers what scale/production engineers need.

2. **Course Adoptability**: Vol I maps to a single semester intro course. Vol II maps to an advanced graduate seminar.

3. **Standalone Completeness**: A reader of only Vol I still gets responsible systems awareness through Chapter 14.

4. **Industry Alignment**: Vol I produces capable junior engineers. Vol II produces senior/staff-level systems thinkers.

### Chapter 14 in Volume I: Responsible Systems Preview

This chapter is intentionally brief (preview, not deep dive) covering:
- Robustness basics (models can fail)
- Security basics (models can be attacked)
- Fairness basics (models can discriminate)
- Sustainability basics (training has environmental cost)

Each topic points to the relevant Volume II chapter for deep treatment. This ensures Vol I readers are aware of these concerns without duplicating Vol II content.

### What Moves Between Volumes?

From original single-book structure:
- On-Device Learning → Vol II (requires scale context)
- Privacy/Security → Vol II (production concern)
- Robust AI → Vol II (advanced topic)
- Responsible AI → Vol II (deep treatment)
- Sustainable AI → Vol II (deep treatment)
- AI for Good → Vol II (capstone application)
- Frontiers → Vol II (forward-looking capstone)

---

## Questions for Reviewers

1. Does Volume I stand alone as a complete, responsible introduction to ML systems?

2. Is the progression within each volume logical for students?

3. Would you adopt Volume I for an introductory ML systems course?

4. Is Chapter 14 (Responsible Systems Preview) sufficient, or should Vol I include more depth on any topic?

5. Are any chapters misplaced between volumes?

6. Is 14 chapters (Vol I) and 16 chapters (Vol II) appropriate sizing?

7. What's missing from either volume?

---

## Revision History

- **v0.1** (2024-12-31): Initial draft for review
- **v0.2** (2024-12-31): Updated Part names based on reviewer feedback
  - Volume I: Single-word Part names (Foundations, Development, Optimization, Operations)
  - Volume II: Two-word Part names (unchanged, already clear)
- **v1.0** (2024-12-31): Structure approved after Round 2 review
  - All reviewers (Patterson, Stoica, Reddi, Dean) approve chapter structure
  - Part naming convention approved
  - Ready for website implementation
