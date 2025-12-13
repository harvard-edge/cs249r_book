# Machine Learning Systems: Comprehensive Volume Split Surgical Plan

**Document Version**: December 2024
**Purpose**: Detailed section-by-section surgery roadmap for splitting ML Systems textbook into two volumes
**Timeline**: 2-month execution phase

---

## Executive Summary

This surgical plan provides precise instructions for splitting each of the 22 chapters between Volume 1 (Introduction to ML Systems) and Volume 2 (Advanced ML Systems). Every section and subsection has been analyzed and assigned a specific action.

### Decision Legend
- **KEEP_V1**: Retain in Volume 1 as-is
- **MOVE_V2**: Move entirely to Volume 2
- **SPLIT**: Divide content between volumes (specify what goes where)
- **MODIFY**: Rewrite/restructure for target volume
- **REMOVE**: Delete as redundant or out of scope
- **BRIDGE**: Add summary/recap content for cross-volume references

### Volume Targets
- **Volume 1**: 14 chapters, ~800 pages (focus: single-system ML)
- **Volume 2**: 14 chapters, ~800 pages (focus: distributed systems and advanced topics)

---

## PART I: FOUNDATIONAL CHAPTERS (1-4)
*These establish core concepts needed by both volumes*

## CHAPTER 1: Introduction (Current: ~90 pages)
**V1 Target: 60 pages | V2 Target: 0 pages | Remove: 30 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Essential framing for the textbook

### ## The Engineering Revolution in Artificial Intelligence (4 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Sets context for entire book series

### ## From Artificial Intelligence Vision to Machine Learning Practice (6 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Historical context valuable for all readers

### ## Defining ML Systems (20 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Core definitional content

### ## How ML Systems Differ from Traditional Software (3 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Fundamental distinction needed early

### ## The Bitter Lesson: Why Systems Engineering Matters (6 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Central thesis of the book

### ## Historical Evolution of AI Paradigms (30 pages)
**DECISION**: MODIFY
**RATIONALE**: Too long for intro chapter, compress to 15 pages
**ACTION**: Condense each era from 5 pages to 2-3 pages

#### ### Symbolic AI Era (5 pages)
**DECISION**: MODIFY - Compress to 2 pages

#### ### Expert Systems Era (4 pages)
**DECISION**: MODIFY - Compress to 2 pages

#### ### Statistical Learning Era (8 pages)
**DECISION**: MODIFY - Compress to 3 pages

#### ### Shallow Learning Era (6 pages)
**DECISION**: MODIFY - Compress to 3 pages

#### ### Deep Learning Era (7 pages)
**DECISION**: KEEP_V1 - Keep at current length

### ## Understanding ML System Lifecycle and Deployment (8 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Essential lifecycle overview

#### ### The ML Development Lifecycle (3 pages)
**DECISION**: KEEP_V1

#### ### The Deployment Spectrum (2 pages)
**DECISION**: KEEP_V1

#### ### How Deployment Shapes the Lifecycle (3 pages)
**DECISION**: KEEP_V1

### ## Case Studies in Real-World ML Systems (6 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Concrete examples ground abstract concepts

#### ### Case Study: Autonomous Vehicles (4 pages)
**DECISION**: KEEP_V1

#### ### Contrasting Deployment Scenarios (2 pages)
**DECISION**: KEEP_V1

### ## Core Engineering Challenges in ML Systems (8 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic challenges (4 pages)
**V2_CONTENT**: Scale-related challenges move to V2 intro

#### ### Data Challenges (2 pages)
**DECISION**: KEEP_V1

#### ### Model Challenges (2 pages)
**DECISION**: KEEP_V1

#### ### System Challenges (2 pages)
**DECISION**: SPLIT - Basic in V1, distributed in V2

#### ### Ethical Considerations (1 page)
**DECISION**: KEEP_V1 - Brief mention, detail in V2

#### ### Understanding Challenge Interconnections (1 page)
**DECISION**: KEEP_V1

### ## Defining AI Engineering (4 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Core professional identity content

### ## Organizing ML Systems Engineering: The Five-Pillar Framework (8 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Structural framework for book

#### ### The Five Engineering Disciplines (3 pages)
**DECISION**: KEEP_V1

#### ### Connecting Components, Lifecycle, and Disciplines (2 pages)
**DECISION**: KEEP_V1

#### ### Future Directions in ML Systems Engineering (1 page)
**DECISION**: REMOVE - Save for V2

#### ### The Nature of Systems Knowledge (1 page)
**DECISION**: KEEP_V1

#### ### How to Use This Textbook (1 page)
**DECISION**: MODIFY - Update for two-volume structure

---

## CHAPTER 2: ML Systems (Current: ~70 pages)
**V1 Target: 70 pages | V2 Target: 0 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Deployment Paradigm Framework (3 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Essential taxonomy for understanding ML systems

### ## The Deployment Spectrum (20 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Core conceptual framework

#### ### Deployment Paradigm Foundations (15 pages)
**DECISION**: KEEP_V1

### ## Cloud ML: Maximizing Computational Power (10 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic cloud concepts (5 pages)
**V2_CONTENT**: Large-scale distributed training details → V2 Ch2

#### ### Cloud Infrastructure and Scale (3 pages)
**DECISION**: KEEP_V1 - Basic concepts only

#### ### Cloud ML Trade-offs and Constraints (2 pages)
**DECISION**: KEEP_V1

#### ### Large-Scale Training and Inference (5 pages)
**DECISION**: MOVE_V2 - Goes to "Distributed Training" chapter
**V2_DESTINATION**: Volume 2, Chapter 4: Distributed Training

### ## Edge ML: Reducing Latency and Privacy Risk (10 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Single-device edge important for V1

#### ### Distributed Processing Architecture (3 pages)
**DECISION**: MOVE_V2 - Multi-device edge is advanced
**V2_DESTINATION**: Volume 2, Chapter 8: Edge Deployment

#### ### Edge ML Benefits and Deployment Challenges (4 pages)
**DECISION**: KEEP_V1

#### ### Real-Time Industrial and IoT Systems (3 pages)
**DECISION**: KEEP_V1

### ## Mobile ML: Personal and Offline Intelligence (10 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Single-device mobile essential for V1

#### ### Battery and Thermal Constraints (3 pages)
**DECISION**: KEEP_V1

#### ### Mobile ML Benefits and Resource Constraints (4 pages)
**DECISION**: KEEP_V1

#### ### Personal Assistant and Media Processing (3 pages)
**DECISION**: KEEP_V1

### ## Tiny ML: Ubiquitous Sensing at Scale (10 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Important for complete deployment spectrum

#### ### Extreme Resource Constraints (3 pages)
**DECISION**: KEEP_V1

#### ### TinyML Advantages and Operational Trade-offs (3 pages)
**DECISION**: KEEP_V1

#### ### Environmental and Health Monitoring (4 pages)
**DECISION**: KEEP_V1

### ## Hybrid Architectures: Combining Paradigms (8 pages)
**DECISION**: MOVE_V2
**RATIONALE**: Multi-tier systems are advanced topic
**V2_DESTINATION**: Volume 2, Chapter 6: Inference Systems

#### ### Multi-Tier Integration Patterns (4 pages)
**DECISION**: MOVE_V2

#### ### Production System Case Studies (4 pages)
**DECISION**: MOVE_V2

### ## Shared Principles Across Deployment Paradigms (5 pages)
**DECISION**: KEEP_V1

### ## Comparative Analysis and Selection Framework (8 pages)
**DECISION**: KEEP_V1

### ## Decision Framework for Deployment Selection (8 pages)
**DECISION**: KEEP_V1

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 3: Deep Learning Primer (Current: ~100 pages)
**V1 Target: 100 pages | V2 Target: 0 pages**

*ALL SECTIONS: KEEP_V1*
**RATIONALE**: Entire chapter is foundational knowledge needed before any advanced topics

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Deep Learning Systems Engineering Foundation (5 pages)
**DECISION**: KEEP_V1

### ## Evolution of ML Paradigms (20 pages)
**DECISION**: KEEP_V1

### ## From Biology to Silicon (15 pages)
**DECISION**: KEEP_V1

### ## Neural Network Fundamentals (30 pages)
**DECISION**: KEEP_V1

### ## Learning Process (20 pages)
**DECISION**: KEEP_V1

### ## Inference Pipeline (8 pages)
**DECISION**: KEEP_V1

### ## Case Study: USPS Digit Recognition (5 pages)
**DECISION**: KEEP_V1

### ## Deep Learning and the AI Triangle (2 pages)
**DECISION**: KEEP_V1

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 4: DNN Architectures (Current: ~100 pages)
**V1 Target: 100 pages | V2 Target: 0 pages**

*ALL SECTIONS: KEEP_V1*
**RATIONALE**: Core architectures needed for understanding all subsequent content

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Architectural Principles and Engineering Trade-offs (3 pages)
**DECISION**: KEEP_V1

### ## Multi-Layer Perceptrons: Dense Pattern Processing (20 pages)
**DECISION**: KEEP_V1

### ## CNNs: Spatial Pattern Processing (25 pages)
**DECISION**: KEEP_V1

### ## RNNs: Sequential Pattern Processing (15 pages)
**DECISION**: KEEP_V1

### ## Attention Mechanisms: Dynamic Pattern Processing (30 pages)
**DECISION**: KEEP_V1

### ## Architectural Building Blocks (10 pages)
**DECISION**: KEEP_V1

### ## System-Level Building Blocks (15 pages)
**DECISION**: KEEP_V1

### ## Architecture Selection Framework (10 pages)
**DECISION**: KEEP_V1

### ## Unified Framework: Inductive Biases (3 pages)
**DECISION**: KEEP_V1

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## PART II: DESIGN PRINCIPLES (5-8)
*Building ML systems end-to-end*

## CHAPTER 5: Workflow (Current: ~40 pages)
**V1 Target: 40 pages | V2 Target: 0 pages**

*ALL SECTIONS: KEEP_V1*
**RATIONALE**: Workflow fundamentals apply to all scales

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Systematic Framework for ML Development (2 pages)
**DECISION**: KEEP_V1

### ## Understanding the ML Lifecycle (5 pages)
**DECISION**: KEEP_V1

### ## ML vs Traditional Software Development (3 pages)
**DECISION**: KEEP_V1

### ## Six Core Lifecycle Stages (20 pages total)
**DECISION**: KEEP_V1 (all subsections)

### ## Integrating Systems Thinking Principles (5 pages)
**DECISION**: KEEP_V1

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 6: Data Engineering (Current: ~120 pages)
**V1 Target: 80 pages | V2 Target: 40 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Data Engineering as a Systems Discipline (3 pages)
**DECISION**: KEEP_V1

### ## Four Pillars Framework (20 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Core framework applies at all scales

### ## Data Cascades and Systematic Foundations (10 pages)
**DECISION**: KEEP_V1

### ## Data Pipeline Architecture (15 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-machine pipelines (10 pages)
**V2_CONTENT**: Distributed pipelines (5 pages) → V2 Ch2

#### ### Quality Through Validation and Monitoring (4 pages)
**DECISION**: KEEP_V1

#### ### Reliability Through Graceful Degradation (3 pages)
**DECISION**: KEEP_V1

#### ### Scalability Patterns (4 pages)
**DECISION**: MOVE_V2 → Storage Systems chapter
**V2_DESTINATION**: Volume 2, Chapter 2: Storage Systems for ML

#### ### Governance Through Observability (4 pages)
**DECISION**: SPLIT - Basic in V1, distributed in V2

### ## Strategic Data Acquisition (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-source acquisition (15 pages)
**V2_CONTENT**: Multi-source federation (5 pages) → V2 Ch2

### ## Data Ingestion (15 pages)
**DECISION**: SPLIT

#### ### Batch vs. Streaming Ingestion Patterns (8 pages)
**DECISION**: SPLIT - Batch in V1, streaming to V2
**V2_DESTINATION**: Volume 2, Chapter 2: Storage Systems

#### ### ETL and ELT Comparison (4 pages)
**DECISION**: KEEP_V1

#### ### Multi-Source Integration Strategies (3 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 2: Storage Systems

### ## Systematic Data Processing (15 pages)
**DECISION**: SPLIT

#### ### Ensuring Training-Serving Consistency (3 pages)
**DECISION**: KEEP_V1

#### ### Building Idempotent Data Transformations (3 pages)
**DECISION**: KEEP_V1

#### ### Scaling Through Distributed Processing (3 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 2: Storage Systems

#### ### Tracking Data Transformation Lineage (3 pages)
**DECISION**: KEEP_V1

#### ### End-to-End Processing Pipeline Design (3 pages)
**DECISION**: KEEP_V1

### ## Data Labeling (15 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Core capability needed at all scales

### ## Strategic Storage Architecture (10 pages)
**DECISION**: SPLIT

#### ### ML Storage Systems Architecture Options (3 pages)
**DECISION**: KEEP_V1 - Basic options

#### ### ML Storage Requirements and Performance (3 pages)
**DECISION**: SPLIT - Basic in V1, distributed in V2

#### ### Storage Across the ML Lifecycle (2 pages)
**DECISION**: KEEP_V1

#### ### Feature Stores: Bridging Training and Serving (2 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 2: Storage Systems

### ## Data Governance (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic governance (10 pages)
**V2_CONTENT**: Enterprise governance (10 pages) → V2 Ch9

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 7: Frameworks (Current: ~150 pages)
**V1 Target: 100 pages | V2 Target: 50 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Framework Abstraction and Necessity (3 pages)
**DECISION**: KEEP_V1

### ## Historical Development Trajectory (10 pages)
**DECISION**: KEEP_V1

### ## Fundamental Concepts (80 pages)
**DECISION**: SPLIT

#### ### Computational Graphs (20 pages)
**DECISION**: KEEP_V1

#### ### Automatic Differentiation (30 pages)
**DECISION**: KEEP_V1

#### ### Data Structures (20 pages)
**DECISION**: KEEP_V1

#### ### Programming and Execution Models (30 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-device execution (20 pages)
**V2_CONTENT**: Distributed execution (10 pages) → V2 Ch4

#### ### Core Operations (20 pages)
**DECISION**: KEEP_V1

### ## Framework Architecture (5 pages)
**DECISION**: KEEP_V1

### ## Framework Ecosystem (5 pages)
**DECISION**: KEEP_V1

### ## System Integration (8 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Local integration (4 pages)
**V2_CONTENT**: Distributed integration (4 pages) → V2 Ch4

### ## Major Framework Platform Analysis (30 pages)
**DECISION**: SPLIT

#### ### TensorFlow Ecosystem (10 pages)
**DECISION**: SPLIT - Core in V1, distributed in V2

#### ### PyTorch (10 pages)
**DECISION**: SPLIT - Core in V1, distributed in V2

#### ### JAX (5 pages)
**DECISION**: KEEP_V1

#### ### Framework Design Philosophy (5 pages)
**DECISION**: KEEP_V1

### ## Deployment Environment-Specific Frameworks (20 pages)
**DECISION**: SPLIT

#### ### Distributed Computing Platform Optimization (5 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 4: Distributed Training

#### ### Local Processing and Low-Latency Optimization (5 pages)
**DECISION**: KEEP_V1

#### ### Resource-Constrained Device Optimization (5 pages)
**DECISION**: KEEP_V1

#### ### Microcontroller and Embedded System Implementation (5 pages)
**DECISION**: KEEP_V1

### ## Systematic Framework Selection Methodology (10 pages)
**DECISION**: KEEP_V1

### ## Systematic Framework Performance Assessment (8 pages)
**DECISION**: KEEP_V1

### ## Common Framework Selection Misconceptions (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 8: Training (Current: ~160 pages)
**V1 Target: 100 pages | V2 Target: 60 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Training Systems Evolution and Architecture (5 pages)
**DECISION**: KEEP_V1

### ## Training Systems (10 pages)
**DECISION**: KEEP_V1

### ## Mathematical Foundations (30 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Core math needed for all training

### ## Pipeline Architecture (25 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Single-machine pipeline fundamentals

### ## Pipeline Optimizations (40 pages)
**DECISION**: SPLIT

#### ### Systematic Optimization Framework (3 pages)
**DECISION**: KEEP_V1

#### ### Production Optimization Decision Framework (2 pages)
**DECISION**: KEEP_V1

#### ### Data Prefetching and Pipeline Overlapping (10 pages)
**DECISION**: KEEP_V1

#### ### Mixed-Precision Training (8 pages)
**DECISION**: KEEP_V1

#### ### Gradient Accumulation and Checkpointing (10 pages)
**DECISION**: KEEP_V1

#### ### Optimization Technique Comparison (3 pages)
**DECISION**: KEEP_V1

#### ### Multi-Machine Scaling Fundamentals (4 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 4: Distributed Training

### ## Distributed Systems (60 pages)
**DECISION**: MOVE_V2
**RATIONALE**: Entire section about multi-machine training
**V2_DESTINATION**: Volume 2, Chapter 4: Distributed Training

#### ### Distributed Training Efficiency Metrics
**DECISION**: MOVE_V2

#### ### Data Parallelism
**DECISION**: MOVE_V2

#### ### Model Parallelism
**DECISION**: MOVE_V2

#### ### Hybrid Parallelism
**DECISION**: MOVE_V2

#### ### Parallelism Strategy Comparison
**DECISION**: MOVE_V2

#### ### Framework Integration
**DECISION**: MOVE_V2

### ## Performance Optimization (5 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-machine optimization (3 pages)
**V2_CONTENT**: Distributed optimization (2 pages) → V2 Ch4

### ## Hardware Acceleration (8 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-accelerator (5 pages)
**V2_CONTENT**: Multi-accelerator (3 pages) → V2 Ch4

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## PART III: PERFORMANCE ENGINEERING (9-12)
*Making ML systems efficient*

## CHAPTER 9: Efficient AI (Current: ~60 pages)
**V1 Target: 60 pages | V2 Target: 0 pages**

*ALL SECTIONS: KEEP_V1*
**RATIONALE**: Efficiency principles apply at all scales

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## The Efficiency Imperative (2 pages)
**DECISION**: KEEP_V1

### ## Defining System Efficiency (3 pages)
**DECISION**: KEEP_V1

### ## AI Scaling Laws (20 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Fundamental principles needed for understanding efficiency

### ## The Efficiency Framework (25 pages)
**DECISION**: KEEP_V1

### ## Real-World Efficiency Strategies (3 pages)
**DECISION**: KEEP_V1

### ## Efficiency Trade-offs and Challenges (10 pages)
**DECISION**: KEEP_V1

### ## Engineering Principles for Efficient AI (3 pages)
**DECISION**: KEEP_V1

### ## Societal and Ethical Implications (5 pages)
**DECISION**: KEEP_V1

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 10: Optimizations (Current: ~200 pages)
**V1 Target: 120 pages | V2 Target: 80 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Model Optimization Fundamentals (3 pages)
**DECISION**: KEEP_V1

### ## Optimization Framework (5 pages)
**DECISION**: KEEP_V1

### ## Deployment Context (5 pages)
**DECISION**: KEEP_V1

### ## Framework Application and Navigation (5 pages)
**DECISION**: KEEP_V1

### ## Optimization Dimensions (5 pages)
**DECISION**: KEEP_V1

### ## Structural Model Optimization Methods (80 pages)
**DECISION**: SPLIT

#### ### Pruning (40 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic pruning techniques (25 pages)
**V2_CONTENT**: Advanced/structured pruning (15 pages) → V2 Ch4

#### ### Knowledge Distillation (15 pages)
**DECISION**: KEEP_V1

#### ### Structured Approximations (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic approximations (10 pages)
**V2_CONTENT**: Advanced approximations (10 pages) → V2 Ch4

#### ### Neural Architecture Search (5 pages)
**DECISION**: MOVE_V2
**RATIONALE**: NAS requires significant compute
**V2_DESTINATION**: Volume 2, Chapter 4: Distributed Training

### ## Quantization and Precision Optimization (50 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic quantization (30 pages)
**V2_CONTENT**: Extreme quantization (20 pages) → V2 Ch8

### ## Architectural Efficiency Techniques (30 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic techniques (20 pages)
**V2_CONTENT**: Advanced techniques (10 pages) → V2 Ch8

### ## Implementation Strategy and Evaluation (5 pages)
**DECISION**: KEEP_V1

### ## AutoML and Automated Optimization Strategies (5 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 4: Distributed Training

### ## Implementation Tools and Software Frameworks (10 pages)
**DECISION**: KEEP_V1

### ## Technique Comparison (3 pages)
**DECISION**: KEEP_V1

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 11: Hardware Acceleration (Current: ~140 pages)
**V1 Target: 90 pages | V2 Target: 50 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## AI Hardware Acceleration Fundamentals (3 pages)
**DECISION**: KEEP_V1

### ## Evolution of Hardware Specialization (15 pages)
**DECISION**: KEEP_V1

### ## AI Compute Primitives (20 pages)
**DECISION**: KEEP_V1

### ## AI Memory Systems (15 pages)
**DECISION**: KEEP_V1

### ## Hardware Mapping Fundamentals for Neural Networks (10 pages)
**DECISION**: KEEP_V1

### ## Dataflow Optimization Strategies (20 pages)
**DECISION**: KEEP_V1

### ## Compiler Support (10 pages)
**DECISION**: KEEP_V1

### ## Runtime Support (5 pages)
**DECISION**: KEEP_V1

### ## Multi-Chip AI Acceleration (20 pages)
**DECISION**: MOVE_V2
**RATIONALE**: Multi-chip is distributed computing
**V2_DESTINATION**: Volume 2, Chapter 1: Memory Hierarchies

#### ### Chiplet-Based Architectures
**DECISION**: MOVE_V2

#### ### Multi-GPU Systems
**DECISION**: MOVE_V2

#### ### TPU Pods
**DECISION**: MOVE_V2

#### ### Wafer-Scale AI
**DECISION**: MOVE_V2

### ## Heterogeneous SoC AI Acceleration (8 pages)
**DECISION**: KEEP_V1
**RATIONALE**: Single-chip heterogeneity important for edge

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 12: Benchmarking (Current: ~120 pages)
**V1 Target: 80 pages | V2 Target: 40 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Machine Learning Benchmarking Framework (3 pages)
**DECISION**: KEEP_V1

### ## Historical Context (5 pages)
**DECISION**: KEEP_V1

### ## Machine Learning Benchmarks (15 pages)
**DECISION**: KEEP_V1

### ## Benchmarking Granularity (10 pages)
**DECISION**: KEEP_V1

### ## Benchmark Components (20 pages)
**DECISION**: KEEP_V1

### ## Training vs. Inference Evaluation (3 pages)
**DECISION**: KEEP_V1

### ## Training Benchmarks (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-system benchmarks (10 pages)
**V2_CONTENT**: Distributed benchmarks (10 pages) → V2 Ch4

### ## Inference Benchmarks (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-system benchmarks (10 pages)
**V2_CONTENT**: Distributed benchmarks (10 pages) → V2 Ch6

### ## Power Measurement Techniques (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Device-level measurement (10 pages)
**V2_CONTENT**: Datacenter measurement (10 pages) → V2 Ch13

### ## Benchmarking Limitations and Best Practices (20 pages)
**DECISION**: KEEP_V1

### ## Model and Data Benchmarking (15 pages)
**DECISION**: KEEP_V1

### ## Production Environment Evaluation (5 pages)
**DECISION**: MOVE_V2
**V2_DESTINATION**: Volume 2, Chapter 6: Inference Systems

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## PART IV: PRACTICE & IMPACT (13-14 for V1)

## CHAPTER 13: ML Operations (Current: ~80 pages)
**V1 Target: 50 pages | V2 Target: 30 pages**

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Introduction to Machine Learning Operations (3 pages)
**DECISION**: KEEP_V1

### ## Historical Context (5 pages)
**DECISION**: KEEP_V1

### ## Technical Debt and System Complexity (20 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Basic technical debt (10 pages)
**V2_CONTENT**: Distributed system debt (10 pages) → V2 Ch5

### ## Development Infrastructure and Automation (30 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-system CI/CD (20 pages)
**V2_CONTENT**: Distributed CI/CD (10 pages) → V2 Ch5

### ## Production Operations (30 pages)
**DECISION**: SPLIT

#### ### Model Deployment and Serving (10 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-model serving (5 pages)
**V2_CONTENT**: Multi-model orchestration (5 pages) → V2 Ch6

#### ### Resource Management and Performance Monitoring (8 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-system monitoring (4 pages)
**V2_CONTENT**: Distributed monitoring (4 pages) → V2 Ch5

#### ### Model Governance and Team Coordination (8 pages)
**DECISION**: KEEP_V1

#### ### Managing Hidden Technical Debt (4 pages)
**DECISION**: KEEP_V1

### ## Case Studies (15 pages)
**DECISION**: SPLIT
**V1_CONTENT**: Single-system deployments (8 pages)
**V2_CONTENT**: Large-scale deployments (7 pages) → V2 Ch5

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTER 14: AI for Good (Current: ~50 pages)
**V1 Target: 50 pages | V2 Target: 0 pages**

*ALL SECTIONS: KEEP_V1*
**RATIONALE**: Positive conclusion for Volume 1, inspiring students

### ## Purpose (2 pages)
**DECISION**: KEEP_V1

### ## Trustworthy AI Under Extreme Constraints (3 pages)
**DECISION**: KEEP_V1

### ## Societal Challenges and AI Opportunities (3 pages)
**DECISION**: KEEP_V1

### ## Real-World Deployment Paradigms (8 pages)
**DECISION**: KEEP_V1

### ## Sustainable Development Goals Framework (5 pages)
**DECISION**: KEEP_V1

### ## Resource Constraints and Engineering Challenges (10 pages)
**DECISION**: KEEP_V1

### ## Design Pattern Framework (5 pages)
**DECISION**: KEEP_V1

### ## Design Patterns Implementation (25 pages)
**DECISION**: KEEP_V1

### ## Theoretical Foundations for Constrained Learning (5 pages)
**DECISION**: KEEP_V1

### ## Common Deployment Failures and Sociotechnical Pitfalls (5 pages)
**DECISION**: KEEP_V1

### ## Summary (2 pages)
**DECISION**: KEEP_V1

---

## CHAPTERS MOVING TO VOLUME 2 (15-21)

## CHAPTER 15: On-Device Learning → V2 Chapter 7
**V1 Target: 0 pages | V2 Target: 80 pages**

*ALL SECTIONS: MOVE_V2*
**RATIONALE**: Advanced topic requiring distributed coordination

### ## Purpose (2 pages)
**DECISION**: MOVE_V2

### ## Distributed Learning Paradigm Shift (4 pages)
**DECISION**: MOVE_V2

### ## Motivations and Benefits (20 pages)
**DECISION**: MOVE_V2

### ## Design Constraints (20 pages)
**DECISION**: MOVE_V2

### ## Model Adaptation (20 pages)
**DECISION**: MOVE_V2

### ## Data Efficiency (10 pages)
**DECISION**: MOVE_V2

### ## Federated Learning (25 pages)
**DECISION**: MOVE_V2

### ## Production Integration (10 pages)
**DECISION**: MOVE_V2

### ## Systems Integration for Production Deployment (5 pages)
**DECISION**: MOVE_V2

### ## Persistent Technical and Operational Challenges (15 pages)
**DECISION**: MOVE_V2

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: MOVE_V2

### ## Summary (2 pages)
**DECISION**: MOVE_V2

---

## CHAPTER 16: Privacy & Security → V2 Chapter 9-10
**V1 Target: 0 pages | V2 Target: 100 pages**

*Note: Split into two chapters in V2*

### Privacy Content → V2 Chapter 9: Privacy in ML Systems (50 pages)

### ## Purpose (2 pages)
**DECISION**: MOVE_V2

### ## Foundational Concepts and Definitions (10 pages)
**DECISION**: MOVE_V2

### ## Privacy-Preserving Data Techniques (30 pages)
**DECISION**: MOVE_V2

### ## Federated Learning Privacy (included from Ch15)
**DECISION**: MOVE_V2

### Security Content → V2 Chapter 10: Security in ML Systems (50 pages)

### ## Learning from Security Breaches (20 pages)
**DECISION**: MOVE_V2

### ## Model-Specific Attack Vectors (20 pages)
**DECISION**: MOVE_V2

### ## Hardware-Level Security Vulnerabilities (15 pages)
**DECISION**: MOVE_V2

### ## Comprehensive Defense Architectures (30 pages)
**DECISION**: MOVE_V2

### ## Practical Implementation Roadmap (8 pages)
**DECISION**: MOVE_V2

---

## CHAPTER 17: Robust AI → V2 Chapter 11
**V1 Target: 0 pages | V2 Target: 100 pages**

*ALL SECTIONS: MOVE_V2*
**RATIONALE**: Production robustness at scale

### ## Purpose (2 pages)
**DECISION**: MOVE_V2

### ## Introduction to Robust AI Systems (5 pages)
**DECISION**: MOVE_V2

### ## Real-World Robustness Failures (15 pages)
**DECISION**: MOVE_V2

### ## A Unified Framework for Robust AI (10 pages)
**DECISION**: MOVE_V2

### ## Hardware Faults (35 pages)
**DECISION**: MOVE_V2

### ## Intentional Input Manipulation (10 pages)
**DECISION**: MOVE_V2

### ## Environmental Shifts (5 pages)
**DECISION**: MOVE_V2

### ## Input-Level Attacks and Model Robustness (35 pages)
**DECISION**: MOVE_V2

### ## Software Faults (20 pages)
**DECISION**: MOVE_V2

### ## Fault Injection Tools and Frameworks (10 pages)
**DECISION**: MOVE_V2

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: MOVE_V2

### ## Summary (2 pages)
**DECISION**: MOVE_V2

---

## CHAPTER 18: Responsible AI → V2 Chapter 12
**V1 Target: 0 pages | V2 Target: 80 pages**

*ALL SECTIONS: MOVE_V2*
**RATIONALE**: Scale changes responsibility challenges

### ## Purpose (2 pages)
**DECISION**: MOVE_V2

### ## Introduction to Responsible AI (5 pages)
**DECISION**: MOVE_V2

### ## Core Principles (5 pages)
**DECISION**: MOVE_V2

### ## Integrating Principles Across the ML Lifecycle (20 pages)
**DECISION**: MOVE_V2

### ## Responsible AI Across Deployment Environments (15 pages)
**DECISION**: MOVE_V2

### ## Technical Foundations (30 pages)
**DECISION**: MOVE_V2

### ## Sociotechnical Dynamics (10 pages)
**DECISION**: MOVE_V2

### ## Implementation Challenges (15 pages)
**DECISION**: MOVE_V2

### ## AI Safety and Value Alignment (8 pages)
**DECISION**: MOVE_V2

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: MOVE_V2

### ## Summary (2 pages)
**DECISION**: MOVE_V2

---

## CHAPTER 19: Sustainable AI → V2 Chapter 13
**V1 Target: 0 pages | V2 Target: 80 pages**

*ALL SECTIONS: MOVE_V2*
**RATIONALE**: Datacenter-scale sustainability

### ## Purpose (2 pages)
**DECISION**: MOVE_V2

### ## Sustainable AI as an Engineering Discipline (3 pages)
**DECISION**: MOVE_V2

### ## The Sustainability Crisis in AI (3 pages)
**DECISION**: MOVE_V2

### ## Part I: Environmental Impact and Ethical Foundations (8 pages)
**DECISION**: MOVE_V2

### ## Part II: Measurement and Assessment (40 pages)
**DECISION**: MOVE_V2

### ## Hardware Lifecycle Environmental Assessment (10 pages)
**DECISION**: MOVE_V2

### ## Part III: Implementation and Solutions (15 pages)
**DECISION**: MOVE_V2

### ## Embedded AI and E-Waste (10 pages)
**DECISION**: MOVE_V2

### ## Policy and Regulation (8 pages)
**DECISION**: MOVE_V2

### ## Public Engagement (8 pages)
**DECISION**: MOVE_V2

### ## Future Challenges (5 pages)
**DECISION**: MOVE_V2

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: MOVE_V2

### ## Summary (2 pages)
**DECISION**: MOVE_V2

---

## CHAPTER 20: Frontiers → V2 Chapter 14
**V1 Target: 0 pages | V2 Target: 80 pages**

*ALL SECTIONS: MOVE_V2*
**RATIONALE**: Advanced future directions

### ## Purpose (2 pages)
**DECISION**: MOVE_V2

### ## From Specialized AI to General Intelligence (3 pages)
**DECISION**: MOVE_V2

### ## Defining AGI: Intelligence as a Systems Problem (8 pages)
**DECISION**: MOVE_V2

### ## The Compound AI Systems Framework (3 pages)
**DECISION**: MOVE_V2

### ## Building Blocks for Compound Intelligence (20 pages)
**DECISION**: MOVE_V2

### ## Alternative Architectures for AGI (8 pages)
**DECISION**: MOVE_V2

### ## Training Methodologies for Compound Systems (20 pages)
**DECISION**: MOVE_V2

### ## Production Deployment of Compound AI Systems (15 pages)
**DECISION**: MOVE_V2

### ## Remaining Technical Barriers (10 pages)
**DECISION**: MOVE_V2

### ## Emergent Intelligence Through Multi-Agent Coordination (5 pages)
**DECISION**: MOVE_V2

### ## Engineering Pathways to AGI (5 pages)
**DECISION**: MOVE_V2

### ## Implications for ML Systems Engineers (5 pages)
**DECISION**: MOVE_V2

### ## Core Design Principles for AGI Systems (2 pages)
**DECISION**: MOVE_V2

### ## Fallacies and Pitfalls (2 pages)
**DECISION**: MOVE_V2

### ## Summary (2 pages)
**DECISION**: MOVE_V2

---

## CHAPTER 21: AGI Systems (REMOVE - Content merged into Frontiers)
**V1 Target: 0 pages | V2 Target: 0 pages**

**DECISION**: REMOVE
**RATIONALE**: Content consolidated into expanded Frontiers chapter

---

## CHAPTER 22: Conclusion
**V1 Target: 10 pages | V2 Target: 10 pages**

### Volume 1 Conclusion (NEW - 10 pages)
**DECISION**: CREATE NEW
**CONTENT**:
- Synthesize single-system ML engineering
- Bridge to Volume 2 concepts
- Inspire continued learning
- Celebrate accomplishments

### Volume 2 Conclusion (MODIFY from existing - 10 pages)
**DECISION**: MODIFY
**CONTENT**:
- Synthesize distributed systems principles
- Future of ML systems at scale
- Call to action for responsible development

---

## NEW VOLUME 2 CHAPTERS NEEDED

### V2 Chapter 1: Memory Hierarchies for ML (NEW - 60 pages)
**SOURCE**: Extract from distributed sections of Ch11 + new content
**TOPICS**:
- GPU memory management
- HBM architecture
- Activation checkpointing
- Multi-chip memory systems

### V2 Chapter 2: Storage Systems for ML (NEW - 60 pages)
**SOURCE**: Extract from Ch6 distributed sections + new content
**TOPICS**:
- Distributed file systems
- Checkpoint I/O
- Feature stores
- Data lakes

### V2 Chapter 3: Communication & Collective Operations (NEW - 60 pages)
**SOURCE**: Extract from Ch8 distributed sections + new content
**TOPICS**:
- AllReduce algorithms
- Network topology
- Gradient compression
- RDMA

### V2 Chapter 4: Distributed Training (NEW - 80 pages)
**SOURCE**: Consolidate from Ch8 + Ch10 distributed sections
**TOPICS**:
- Data/model/pipeline parallelism
- Synchronization strategies
- Load balancing

### V2 Chapter 5: Fault Tolerance & Recovery (NEW - 60 pages)
**SOURCE**: Extract from Ch13 + new content
**TOPICS**:
- Checkpointing strategies
- Elastic training
- Failure handling

### V2 Chapter 6: Inference Systems (NEW - 60 pages)
**SOURCE**: Extract from Ch2 hybrid + Ch13 serving + new content
**TOPICS**:
- Batching strategies
- Model serving patterns
- Autoscaling

### V2 Chapter 8: Edge Deployment (NEW - 60 pages)
**SOURCE**: Extract from Ch2 edge sections + new content
**TOPICS**:
- Model compilation
- Runtime optimization
- Real-time constraints

---

## EXECUTION TIMELINE

### Month 1: Content Extraction and Migration
**Week 1-2**: Extract and migrate V2 content from existing chapters
- Pull distributed systems content from Ch6, Ch7, Ch8
- Extract multi-chip content from Ch11
- Move advanced chapters (15-20) to V2

**Week 3-4**: Create bridging content
- Write V1→V2 transitions
- Add recaps to V2 chapters
- Update cross-references

### Month 2: New Chapter Development
**Week 5-6**: Draft new V2 chapters 1-3
- Memory Hierarchies
- Storage Systems
- Communication & Collectives

**Week 7-8**: Draft new V2 chapters 4-6, 8
- Distributed Training
- Fault Tolerance
- Inference Systems
- Edge Deployment

---

## CRITICAL DEPENDENCIES TO ADDRESS

### Cross-Volume References
1. **V2 depends on V1 concepts**: Add 2-page recaps at start of V2 chapters
2. **V1 mentions advanced topics**: Add "See Volume 2" callout boxes
3. **Shared examples**: Maintain consistency in running examples

### Content Gaps to Fill
1. **V1 needs**: Brief sustainability mention in Ch12 or Ch13
2. **V2 needs**: Stronger introduction chapter setting distributed context
3. **Both need**: Updated prefaces explaining two-volume structure

### Risk Mitigation
1. **Page count imbalance**: Monitor during extraction phase
2. **Dependency cycles**: Review after initial split
3. **Missing topics**: Keep running list during surgery

---

## SUCCESS METRICS

### Volume 1 Success Criteria
- Complete single-system ML lifecycle coverage
- No hard dependencies on V2 content
- Positive, inspiring conclusion
- 750-850 pages total

### Volume 2 Success Criteria
- Complete distributed systems coverage
- Clear value beyond V1
- Timeless principles focus
- 750-850 pages total

### Overall Success
- Each volume adoptable independently
- Together form comprehensive curriculum
- Minimal content duplication
- Clear progression path

---

## APPENDIX: QUICK REFERENCE

### Chapters Staying in V1 (with modifications)
1. Introduction (compressed)
2. ML Systems (remove distributed)
3. DL Primer (complete)
4. DNN Architectures (complete)
5. Workflow (complete)
6. Data Engineering (remove distributed)
7. Frameworks (remove distributed)
8. Training (remove distributed)
9. Efficient AI (complete)
10. Optimizations (remove advanced)
11. Hardware Acceleration (remove multi-chip)
12. Benchmarking (remove distributed)
13. ML Operations (basic only)
14. AI for Good (complete)

### Chapters Moving to V2
- On-Device Learning → V2 Ch7
- Privacy & Security → V2 Ch9-10 (split)
- Robust AI → V2 Ch11
- Responsible AI → V2 Ch12
- Sustainable AI → V2 Ch13
- Frontiers → V2 Ch14

### New V2 Chapters
1. Memory Hierarchies for ML
2. Storage Systems for ML
3. Communication & Collective Operations
4. Distributed Training
5. Fault Tolerance & Recovery
6. Inference Systems
8. Edge Deployment

---

*End of Surgical Plan Document*
