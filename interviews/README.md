# AI Systems Interview Hub 🧠

A community-driven collection of high-stakes systems design questions for AI infrastructure and ML systems engineering roles.

---

## ⚡ The Motivation: Why Systems Engineering?

The traditional software engineering interview has fundamentally changed. 

Using LLMs to generate algorithms or boilerplate code is the new normal. But when it comes to *engineering* AI systems at scale, you are hard-wired by the physics of the data center. **You cannot prompt your way out of a silicon bottleneck.**

An LLM can write a flawless PyTorch training loop, but it cannot intuitively calculate why a 175B parameter model will hit a communication wall across InfiniBand, or why serving a 128k context window will fragment your KV-cache and explode your P99 latency. These are physical constraints, not code syntax.

**The frontier of AI hiring is at the intersection of Machine Learning and Systems Physics.** Companies like Meta, Google, and OpenAI are aggressively seeking engineers who possess *Mechanical Sympathy*—the ability to look past the framework abstractions and understand the hard reality of how to keep 10,000 GPUs fed and 1 million users served. 

This hub is your playbook for that new frontier.

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

## 💌 A Note on the Mission

I created this hub to help people learn the true physical realities of building AI systems. My hope is that as you interview and encounter new challenges, you will bring those scenarios back here and contribute them to the community. Together, we can build and train the next generation of AI engineers. 

Wishing you all the best in your interviews and your engineering journey!

— *Vijay Janapa Reddi*

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
