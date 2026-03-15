# AI Systems Interview Hub 🧠

A community-driven collection of high-stakes systems design questions for AI infrastructure and ML systems engineering roles.

> [!IMPORTANT]
> **The "Systems-First" Philosophy:** This is *not* a pure machine learning study guide. We do not ask how to tune hyperparameters or derive backpropagation. We focus on the **intersection of ML and Systems**. Companies like Meta, Google, and OpenAI are aggressively hiring engineers who understand the *physics* of AI computation—how to keep 10,000 GPUs fed, how to manage P99 latency during traffic spikes, and how to survive the memory wall. 

---

## 📊 The Mastery Levels

Every question in this hub is tagged with a mastery level, reflecting the "Funnel of Mastery" used in real systems engineering interviews.

*   **🟢 Level 1: The Screen (Junior/Mid)**
    *   *Goal:* Can you define the core systems concepts and hardware constraints? (e.g., "What is the memory bandwidth of an A100?")
*   **🟡 Level 2: The Architect (Senior)**
    *   *Goal:* Can you reason about trade-offs and bottleneck physics? (e.g., "Why does increasing batch size reduce dispatch overhead?")
*   **🔴 Level 3: The Lead (Staff/Principal)**
    *   *Goal:* Can you perform "Whiteboard Physics" on the fly? (e.g., "Size the KV-cache for a 70B model with 128k context and prove it fits on 8xH100s.")

---

## 🥊 The 4 Interview Rounds (Question Categories)

Real-world interviews for AI Systems roles are structured around domains of operational responsibility. Our categories mirror the exact rounds you will face at frontier labs.

1.  **[Round 1: Single-Node Systems & Silicon Physics](01_Single_Node_Physics.md)** 
    *   *The Focus:* VRAM accounting, Arithmetic Intensity, Roofline models, Memory Bandwidth vs. Compute bounds, and Data Ingestion bottlenecks.
2.  **[Round 2: Distributed AI Infrastructure](02_Distributed_Infrastructure.md)**
    *   *The Focus:* 3D Parallelism, Bisection bandwidth, NVLink vs. InfiniBand, AllReduce algorithms, and Fault Tolerance.
3.  **[Round 3: Production ML Systems Design](03_Production_Serving.md)**
    *   *The Focus:* Queueing theory (Erlang-C), Tail Latency, KV-Cache memory fragmentation (PagedAttention), and Continuous Batching.
4.  **[Round 4: ML Operations & Economics](04_Operations_and_Economics.md)**
    *   *The Focus:* Total Cost of Ownership (TCO), The Degradation Equation (Data Drift), Silent Failures, and Technical Debt in ML.

---

## 🗳️ Trending Community Scenarios

We use GitHub Issues as a public "Drafting Room" for new questions. 

<!-- TRENDING-QUESTIONS-START -->
*No trending questions yet. [Submit a question](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml) to start the list.*
<!-- TRENDING-QUESTIONS-END -->

---

## 🤝 Contributing

We welcome contributions of questions and scenarios from recent AI systems interviews.

1.  **Submit an Issue:** Use the **[💼 New Interview Question](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml)** template.
2.  **The 10-Upvote Rule:** Once a community question reaches **10 upvotes (👍)**, it is added to the official category files and you are added to the **Hall of Fame**.

---

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧠 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
