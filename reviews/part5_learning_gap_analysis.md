# Gap Analysis Report: Part V (Learning from Trajectories)

## Overview
This report evaluates **Part V: Learning from Trajectories** (Chapters 13, 14, and 15) of the MLSysBook Volume III against the state-of-the-art in agentic machine learning systems as of mid-2025/2026.

## Covered Topics
Based on the provided drafts, the current chapters robustly cover:
*   **Ch13 (Trajectory Curation):** Capability gap diagnosis, task fixture design, stratified task matrices, data sources (human, telemetry, synthetic), self-improvement loops (expert iteration), distributed collection pipeline architecture (backpressure, microVMs, hypervisor hygiene), trajectory provenance tracking, cryptographic lineage, repository-family split partitioning, dataset quality metrics.
*   **Ch14 (Fine-Tuning):** Scope of fine-tuning (procedure vs. context/enforcement), trajectory serialization (parallel calls, causal monotonicity), observation loss masking, sequence packing with block-diagonal attention, exposure bias and on-policy data (behavioral cloning vs DAgger), distillation, PEFT/adapters, evaluation.
*   **Ch15 (RLVR):** Reinforcement learning from verifiable rewards, reward hacking / specification gaming, reward composition (outcome vs. format terms), credit assignment across turns (Outcome vs Process vs Monte Carlo sub-trees), Group Relative Policy Optimization (GRPO), separating rollout from training (asynchronous freshless gates), release gating enclaves.

## Strengths
*   **Systems-First Perspective:** Exceptional coverage of the physical infrastructure required to harvest and train agents. The detailed treatment of hypervisor hygiene, CoW overlayFS reset latencies, and backpressure governance in disaggregated rollout clusters is rare and highly valuable.
*   **Agent-Specific Adaptations:** Excellent adaptation of standard ML techniques to the agent domain, particularly **observation loss masking** and **block-diagonal causal masking for packed trajectories**.
*   **Rigorous RLVR Focus:** The deep dive into GRPO and Monte Carlo credit assignment, recognizing that RL for agents is bound by environment execution time rather than just GPU FLOPs, is perfectly aligned with current state-of-the-art practices.
*   **Data Hygiene:** Treating trajectory datasets as cryptographic artifacts with repository-level split hygiene addresses major pitfalls in real-world ML engineering.

## Missing Topics

### 1. Offline Preference Optimization (DPO, KTO, IPO)
*   **Gap:** The text focuses heavily on Supervised Fine-Tuning (Ch14) and online RLVR via GRPO (Ch15). It completely misses offline preference optimization techniques such as Direct Preference Optimization (DPO), Kahneman-Tversky Optimization (KTO), or Identity Preference Optimization (IPO).
*   **Why it matters:** Online RLVR is compute-intensive and requires a live sandbox. For many teams, offline preference tuning on historical trajectory pairs (chosen vs. rejected) is the standard middle-ground between SFT and RLVR.
*   **Priority:** **Critical**

### 2. Reward Model Training and Robustness
*   **Gap:** Chapter 15 rightly emphasizes verifiable (deterministic) rewards and correctly identifies that Learned Process Reward Models (PRMs) are highly gameable. However, it ignores *how* to train a PRM or a generalized Reward Model (RM) for domains where deterministic verifiers do not exist (e.g., customer support, UX navigation).
*   **Why it matters:** Not all agentic tasks have a compiler or test suite. Techniques for making PRMs more robust against reward hacking (e.g., RLHF with ensembles, synthetic critique generation) are industry standard.
*   **Priority:** **Important**

### 3. Curriculum Learning for Agents
*   **Gap:** The text discusses a "stratified task matrix" but doesn't explain how to schedule these tasks during training.
*   **Why it matters:** If an RL policy is exposed to $H=50$ horizon tasks immediately, its pass rate will be 0, yielding no gradient. A progressive curriculum—training on $H=1$ tasks, then $H=5$, then $H=50$—is often necessary to bootstrap RLVR.
*   **Priority:** **Important**

### 4. Constitutional AI and Self-Alignment
*   **Gap:** Methods where models critique and revise their own trajectories based on a set of principles (Constitutional AI) to generate synthetic preference pairs or safe fine-tuning data.
*   **Why it matters:** Essential for aligning tone, safety, and adherence to corporate guidelines without requiring expensive human-in-the-loop annotations.
*   **Priority:** **Nice-to-Have**

### 5. Scaling Laws for Agent Adaptation
*   **Gap:** No quantitative framework for how agent capabilities scale with trajectory count, compute budget, or context length during post-training.
*   **Why it matters:** Systems engineers need to know whether to spend their budget on 10k high-quality trajectories or 1M synthetically generated ones.
*   **Priority:** **Nice-to-Have**

## Under-Developed Areas

### 1. Catastrophic Forgetting and Continual Learning
*   **Current State:** Mentioned briefly in Ch13 as a risk of "Premature Fine-Tuning".
*   **Needs Depth:** The text lacks concrete ML mitigations for catastrophic forgetting. How do teams mix foundational data with agent-specific trajectories during SFT? Do they use KL-divergence penalties or elastic weight consolidation to preserve general reasoning?

### 2. Multi-Task Generalization and Negative Transfer
*   **Current State:** Implicitly addressed via dataset splits.
*   **Needs Depth:** When an agent is trained simultaneously on bash scripting, database migrations, and web browsing, gradients can conflict (negative transfer). Strategies for multi-task balancing in agent SFT/RL are under-explored.

### 3. Safety-Aware Training
*   **Current State:** Ch15 notes that soft penalties for dangerous actions don't work, relying instead on sandbox containment.
*   **Needs Depth:** While true from a systems perspective, safety alignment *during* training (e.g., using explicit hard negatives of harmful tool usage in DPO, or unlearning algorithms) is still a necessary defense-in-depth layer.

### 4. RLHF vs RLVR Comparison
*   **Current State:** RLHF is mentioned in a definition box in Ch15.
*   **Needs Depth:** A dedicated comparison section highlighting when to use RLHF (subjective, human-centric tasks) versus RLVR (objective, verifiable tasks) would strengthen the introduction to Chapter 15.

## Suggested Additions
To fill these gaps, consider adding the following concrete sections/subsections:

*   **Ch14 Additions:**
    *   *14.x Direct Preference Optimization for Trajectories*: Covering DPO, formatting chosen/rejected pairs, and offline alignment.
    *   *14.y Mitigating Catastrophic Forgetting*: Discussing data mixing ratios, KL-penalties, and continual learning buffers.
*   **Ch15 Additions:**
    *   *15.1.x RLHF vs RLVR*: A formal comparison of the two paradigms.
    *   *15.x Curriculum Design and Task Escalation*: How to dynamically adjust task difficulty to maintain a non-zero advantage signal.
    *   *15.y Training and Robustifying Process Reward Models*: For domains lacking a deterministic compiler/test-suite.
