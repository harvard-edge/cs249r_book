# Volume 2: Machine Learning Systems at Scale - Seminal Bibliography

This document tracks the foundational research papers, hardware architectures, and industry standards that anchor Volume 2. Organised for a "textbook-scale" deep dive (targeting 700-800 citations).

---

## Part I: Foundations of Scale (Distributed Logic)

### Distributed Training Paradigms
*   **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** (Huang et al., 2019)
*   **PipeDream: Generalized Pipeline Parallelism for DNN Training** (Narayanan et al., 2019)
*   **Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism** (Shoeybi et al., 2019)
*   **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2020)
*   **Efficient Large-Scale Language Model Training on GPU Clusters** (Narayanan et al., 2021) - *The 3D Parallelism Blueprint.*
*   **DistBelief: Large Scale Distributed Deep Networks** (Dean et al., 2012) - *Parameter Server origins.*

### Collective Communication & Algorithms
*   **Bandwidth Optimal All-reduce Algorithms** (Patarasuk & Yuan, 2009) - *Ring AllReduce proof.*
*   **Synthesizing Optimal Collective Algorithms (SCCL)** (Cai et al., 2020)
*   **Rethinking ML Collective Communication as a Multi-Commodity Flow Problem** (Liu et al., 2024)
*   **NCCL: Accelerated Multi-GPU Collective Communications** (NVIDIA, 2017-2025)

### Fault Tolerance & Resilience
*   **Oobleck: Resilient Distributed Training using Pipeline Templates** (Jang et al., 2023)
*   **Varuna: Scalable, Low-cost Training of Massive Models** (Athlur et al., 2022) - *Spot Instance resilience.*
*   **Bamboo: Making Preemptible Instances Resilient for Affordable Training** (Thorpe et al., 2023)

---

## Part II: Building the Machine Learning Fleet (Physical Layer)

### Compute Infrastructure (Silicon & Systems)
*   **In-Datacenter Performance Analysis of a TPU** (Jouppi et al., 2017) - *TPU v1.*
*   **The Design Process for Google’s Training Chips: TPUv2 and TPUv3** (Norrie et al., 2021)
*   **TPU v4: An Optically Reconfigurable Supercomputer** (Jouppi et al., 2023) - *SparseCores & Optical Switching.*
*   **Dissecting the NVIDIA Hopper Architecture** (Luo et al., 2025) - *H100/H200 analysis.*
*   **Microbenchmarking NVIDIA’s Blackwell Architecture** (Jarmusch et al., 2025) - *B200 analysis.*
*   **Cerebras Wafer-Scale Integration: The Cerebras Story** (Lauterbach, 2021)

### Network Fabrics (Topologies & Protocols)
*   **A Scalable, Commodity Data Center Network Architecture (Fat-Tree)** (Al-Fares et al., 2008)
*   **Technology-Driven, Highly-Scalable Dragonfly Topology** (Kim et al., 2008)
*   **Jellyfish: Networking Data Centers Randomly** (Singla et al., 2012)
*   **Congestion Control for Large-Scale RDMA Deployments (DCQCN)** (Zhu et al., 2015)
*   **HPCC: High Precision Congestion Control** (Li et al., 2019)
*   **Swift: Delay is Simple and Effective for Congestion Control** (Kumar et al., 2020) - *Google's Swift protocol.*

### Memory & Interconnect Standards
*   **HBM3: Enabling Memory Resilience at Scale** (Standardization Papers)
*   **Compute Express Link (CXL): A Comprehensive Survey** (Lian et al., 2024)
*   **Next-Gen Interconnection Systems with CXL** (2024)

---

## Part III: Deployment & Optimization (The Serving Layer)

### Inference & Serving
*   **Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** (Kwon et al., 2023)
*   **Orca: A Distributed Serving System for [Transformer] Models** (Yu et al., 2022) - *Continuous Batching.*
*   **FlexFlow: A Distributed Deep Learning Framework** (Jia et al., 2019)

### Performance Engineering (Quantization & Compression)
*   **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** (Dettmers et al., 2022)
*   **SmoothQuant: Accurate and Efficient Post-Training Quantization** (Xiao et al., 2023)
*   **FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022)

---

## Part IV: The Vanguard (The Future of Scale)

### Optical & Photonic Systems
*   **Leveraging Optical Chip-to-chip Connectivity** (Ayar Labs, 2023)
*   **Photonic AI Acceleration: A New Kind of Computer** (Lightmatter, 2025)
*   **Panel-Scale Reconfigurable Photonic Interconnects** (Hsueh et al., 2025)

---

## Core Textbooks (Strategic Guides)
*   **Computer Architecture: A Quantitative Approach** (Hennessy & Patterson) - *The architectural Bible.*
*   **Designing Machine Learning Systems** (Chip Huyen, 2022)
*   **Designing Data-Intensive Applications** (Martin Kleppmann, 2017)
*   **Distributed Systems: Principles and Paradigms** (Tanenbaum & van Steen)
