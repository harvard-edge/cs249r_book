# System Design: The "LeetCode for ML Systems"

If we are building the "LeetCode for ML Systems," the design fundamentally shifts from a **reading platform** (flashcards) to a **validation platform** (simulated execution).

LeetCode works because you write code, click "Submit", and get an objective `PASS` or `FAIL` based on execution time and edge cases. You cannot do that with system design using simple text boxes. We must evaluate **architecture**, not algorithms.

Here is the blueprint for **"IronLaw"** (or whatever we name it)—the interactive ML Systems interview platform.

---

## 1. The Core Loop (The "Submit" Button for Systems)

Instead of writing a Python function to reverse a string, the user configures a system to meet an SLA (Service Level Agreement).

### The Challenge Prompt:
> **Task:** Serve a 70B parameter LLM to 5,000 concurrent users.
> **Constraints:**
> - Time-To-First-Token (TTFT) < 200ms
> - Time-Per-Output-Token (TPOT) < 50ms
> - Budget: < $50,000 / month

### The User Interface:
The user doesn't type text. They use a **Visual Architecture Builder** (or a clean YAML config editor for power users). They drag and drop:
* **Hardware Nodes:** (e.g., Select 8x H100s vs. 16x A100s)
* **Network Topology:** (e.g., NVLink within node, 400Gbps InfiniBand between nodes)
* **Serving Framework Options:** (e.g., Enable PagedAttention? Enable Continuous Batching? Enable Speculative Decoding?)
* **Model Optimizations:** (e.g., Apply INT8 Quantization? Use Grouped Query Attention?)

### The Validation Engine (The Secret Sauce):
When the user clicks **"Submit Architecture"**, we do not use an LLM to guess if they are right.
We pass their JSON configuration directly into our **`mlsysim` engine**.

The engine calculates the deterministic physics:
1. Calculates the KV-Cache memory footprint.
2. Calculates the arithmetic intensity and Roofline boundaries.
3. Simulates the queueing theory (Little's Law) under the 5,000 concurrent user load.

### The Result Screen:
```text
❌ FAILED: SLA Violation
Your architecture costs $32,000/month (PASS), but your TPOT is 120ms (FAIL).
Diagnostic: You are memory-bandwidth bound. Your 8x A100s (2.0 TB/s each) cannot stream the 70B INT8 weights fast enough at batch size 32.

Hint: You have spare compute. Have you considered trading compute for memory bandwidth using Speculative Decoding?
```

---

## 2. The Gamification Engine

To build the addictive loop of LeetCode, we need status and progression.

### "Iron Ranks" (Elo for Systems Engineers)
* Users start at **L3 (Junior)**.
* They are given basic challenges: *"Fit a 13B model on a single 24GB GPU."*
* As they pass challenges (verified by `mlsysim`), their Elo rating increases.
* When they hit **L5 (Senior)**, the challenges introduce distributed failures: *"Your cluster has a 1% per-GPU failure rate. Design a checkpointing interval that maximizes total training throughput."*

### The "Global Leaderboard" (Cost vs. Latency)
On LeetCode, you want your algorithm to be in the "Top 1% of execution time."
On our platform, architecture is a multi-objective optimization problem.
When a user passes a challenge, they see a **Pareto Frontier plot**.
* *"Your solution passed! But it costs $40,000/month. View user `sys_arch_99`'s solution which achieved the exact same latency for only $18,000/month."*

---

## 3. The LLM Mentor (The "Discussion" Tab)

LeetCode's most valuable feature is the Discussion tab where people explain the optimal solution. We can automate the perfect mentor.

If a user is stuck, they click **"Ask the Architect"**.
* We pass their current failing architecture JSON into an LLM, along with the specific **textbook chapter** (RAG from `book/quarto/`).
* The LLM responds: *"I see you selected Pipeline Parallelism across 4 nodes. But look at your InfiniBand utilization—it's at 2%. Meanwhile, your Pipeline Bubble is 45%. You need more microbatches. Read Chapter 4 on the 1F1B schedule."*

---

## 4. The Technical Architecture (How we build it)

We already have 80% of the pieces.

1. **Frontend (Next.js / React Flow):** A node-based UI where users connect GPUs, specify networks, and set batch sizes. (Or a Monaco editor for YAML).
2. **Backend Engine (FastAPI + `mlsysim`):** The engine takes the JSON representation of the user's architecture, runs the exact physics formulas you already wrote in the simulator, and returns the SLA metrics (Latency, Memory, Cost).
3. **Content DB (The 240 Scenarios):** We translate your markdown interview flashcards into objective, programmatic constraints.
   * *Old Flashcard:* "Why does a 128k context window OOM an 8B model?"
   * *New Challenge:* "Configure a serving stack that successfully processes 128k context windows for an 8B model on a 40GB GPU." (The user *must* discover and toggle PagedAttention/RingAttention to pass).

---

## 5. The Business Model (Why it makes money)

* **B2C (The LeetCode Model):** Free tier with basic single-node challenges. Premium tier ($30/month) unlocks distributed training challenges, advanced network topologies, and the LLM Mentor.
* **B2B Hiring (The HackerRank Model):** Companies stop asking candidates to reverse linked lists. They send candidates a link to our platform: *"Here is our actual production workload. Use the architecture builder to optimize it. You have 45 minutes."* The company gets a deterministic score of the candidate's mechanical sympathy.
* **B2B Infrastructure (The CAD Tool):** Startups use the platform not for interviews, but to design their actual systems before buying $1M in cloud compute.

## Summary
To build the "LeetCode for ML Systems," we must stop grading with text and start grading with physics. The core loop is: **Given a workload and an SLA, design an architecture that satisfies `mlsysim`.**
