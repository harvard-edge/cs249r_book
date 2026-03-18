# The 4-Week ML Systems Study Guide 🗓️

<div align="center">
  <a href="README.md">🏠 Playbook Home</a> ·
  <a href="cloud/README.md">☁️ Cloud</a> ·
  <a href="edge/README.md">🤖 Edge</a> ·
  <a href="mobile/README.md">📱 Mobile</a> ·
  <a href="tinyml/README.md">🔬 TinyML</a>
</div>

---

With over 800 questions in this playbook, it is easy to get overwhelmed. This 4-week study guide curates a structured path through the core material, taking you from single-node physics up to fleet-scale architectural design.

If you have 30 minutes a day, follow this syllabus.

## Week 1: The Physics of Compute & Memory
**Goal:** Stop guessing about performance and start calculating it. This week builds your "Mechanical Sympathy."

*   **Day 1: The Roofline Model**
    *   *Cloud:* [The Profiling Crisis](cloud/01_compute_and_memory.md)
    *   *Edge:* [The NPU Roofline](edge/02_compute_and_memory.md)
    *   *Homework:* Memorize the Peak FLOPS and Bandwidth for the A100 and H100 from the [Numbers](NUMBERS.md) page.
*   **Day 2: Precision & Representation**
    *   *Cloud:* [The Underflow Crisis](cloud/01_compute_and_memory.md) (FP16 vs BF16)
    *   *Mobile:* [The Accuracy vs Power Trap](mobile/02_compute_and_memory.md) (INT8 vs FP16)
*   **Day 3: Memory Systems & The KV-Cache**
    *   *Cloud:* [The Sequence Length Trap](cloud/01_compute_and_memory.md)
    *   *Cloud:* [The Fragmentation Crisis](cloud/03_inference_and_serving.md) (PagedAttention)
*   **Day 4: Hardware Architecture**
    *   *Cloud:* [The Sparsity Fallacy](cloud/01_compute_and_memory.md)
    *   *TinyML:* [Co-Designing a TinyML Accelerator](tinyml/05_advanced_systems.md)
*   **Day 5: Pipeline Bottlenecks**
    *   *Cloud:* [The Data Pipeline Stall](cloud/01_compute_and_memory.md)
    *   *Edge:* [The ISP/NPU Handoff](edge/01_systems_and_real_time.md)

---

## Week 2: Scaling Up (Distributed Systems)
**Goal:** Understand how to split a 70B parameter model across 1,000 GPUs without the network starving the compute.

*   **Day 1: The Memory Wall of Training**
    *   *Cloud:* [The OOM Error](cloud/02_network_and_distributed.md) (Optimizer State)
    *   *Cloud:* [The Gradient Checkpointing Boundary](cloud/06_advanced_systems.md)
*   **Day 2: Network Topologies**
    *   *Cloud:* [The Cross-Rack Stall](cloud/02_network_and_distributed.md) (InfiniBand vs Ethernet)
    *   *Cloud:* [The Oversubscription Choke](cloud/02_network_and_distributed.md)
*   **Day 3: Collective Communication**
    *   *Cloud:* [The Ring vs Tree Dilemma](cloud/02_network_and_distributed.md)
*   **Day 4: 3D Parallelism (The Master Class)**
    *   *Cloud:* [The Pipeline Bubble](cloud/02_network_and_distributed.md)
    *   *Cloud:* [Dimensioning the 3D Cube](cloud/02_network_and_distributed.md)
*   **Day 5: Fault Tolerance at Scale**
    *   *Cloud:* [The MTBF Crisis](cloud/02_network_and_distributed.md) (Young-Daly Equation)
    *   *Cloud:* [The Straggler Problem](cloud/02_network_and_distributed.md)

---

## Week 3: Serving at Scale & Edge Deployment
**Goal:** Move from training to inference. Understand queuing theory, batching, and physical deployment constraints.

*   **Day 1: LLM Serving Metrics**
    *   *Cloud:* [The Serving Inversion](cloud/03_inference_and_serving.md) (TTFT vs TPOT)
    *   *Cloud:* [The Black Friday Collapse](cloud/03_inference_and_serving.md) (Queuing Theory)
*   **Day 2: Advanced Batching**
    *   *Cloud:* [The Serverless Freeze](cloud/03_inference_and_serving.md)
    *   *Cloud:* [The Batching Dilemma](cloud/03_inference_and_serving.md) (Continuous Batching)
*   **Day 3: Disaggregated Serving**
    *   *Cloud:* [The Disaggregated Serving](cloud/03_inference_and_serving.md) (Prefill vs Decode splitting)
    *   *Cloud:* [The Decoding Bottleneck](cloud/03_inference_and_serving.md) (Speculative Decoding)
*   **Day 4: Mobile & App Store Constraints**
    *   *Mobile:* [The App Size Limit](mobile/03_data_and_deployment.md)
    *   *Mobile:* [The App Store Model Size Rejection](mobile/03_data_and_deployment.md)
*   **Day 5: Fleet Updates**
    *   *Mobile:* [The Cross-Version Compatibility Maze](mobile/03_data_and_deployment.md)
    *   *TinyML:* [The FOTA Update Risk](tinyml/03_data_and_deployment.md)

---

## Week 4: Ops, Power, and the "Real World"
**Goal:** Answer the questions that separate standard engineers from technical leads. Security, thermal physics, and silent failures.

*   **Day 1: Thermal & Power Budgets**
    *   *Cloud:* [The Rack Power Budget](cloud/06_advanced_systems.md)
    *   *Cloud:* [The Thermal Throttling Mystery](cloud/06_advanced_systems.md)
    *   *Mobile:* [The Thermal Throttling Cycle](mobile/02_compute_and_memory.md)
*   **Day 2: Monitoring & Silent Failures**
    *   *Cloud:* [The GPU Utilization Paradox](cloud/04_data_and_mlops.md)
    *   *Mobile:* [The Silent Accuracy Degradation](mobile/03_data_and_deployment.md)
*   **Day 3: On-Device Architectures**
    *   *TinyML:* [Sub-Milliwatt Always-On Wake Word Detection](tinyml/05_advanced_systems.md)
    *   *Mobile:* [The 50-Feature Mobile ML Platform](mobile/03_data_and_deployment.md)
*   **Day 4: Security, Privacy & Silent Failures**
    *   *Cloud:* [The Tokenizer Mismatch](cloud/06_advanced_systems.md)
    *   *Mobile:* [The Federated Keyboard](mobile/03_data_and_deployment.md) (DP-FedAvg)
*   **Day 5: Visual Architecture Debugging**
    *   Take the Ultimate Test: Try to solve the architectures in [Cloud 5. Visual Debugging](cloud/05_visual_debugging.md) without looking at the answers.
