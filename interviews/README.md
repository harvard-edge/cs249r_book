# AI Systems Interview Hub 🧠

A community-driven collection of high-stakes systems design questions for AI infrastructure and model deployment.

> [!IMPORTANT]
> **Work in Progress:** This hub is currently being seeded with scenarios from recent interviews at frontier labs (OpenAI, Anthropic, Meta, Google). [Contribute your experience here](#-contributing).

---

## 🏗️ Question Categories

Follow the progression from single-accelerator foundations to global production fleets.

**Phase 1: Single-Machine Foundations**
1.  **[Foundations & Hardware](01_Foundations.md)**: VRAM, Peak Performance, and Memory Bandwidth.
2.  **[Models & Architectures](02_Architectures.md)**: KV-Cache, Transformers, and CNN efficiency.
3.  **[Data & Pipelines](03_Data_Pipelines.md)**: Ingestion, Transformation, and GPU starvation.

**Phase 2: Distributed Systems & Scale**
4.  **[Distributed ML & Scaling](04_Distributed_ML.md)**: AllReduce, Parallelism, and Fault Tolerance.
5.  **[Serving & Operations](05_Serving_Ops.md)**: Tail Latency, TCO, and Production Reliability.

---

## 🖊️ The Whiteboard Challenge

Interviewers at top labs focus on **Visual Logic and Precise Calculation**. These challenges test your ability to do napkin math on the fly.

<details>
<summary><b>WHITEBOARD: Draw the Roofline for a new "Mystery Chip"</b></summary>

### The Prompt
"I have a new accelerator with 500 TFLOPS of compute and 500 GB/s of HBM bandwidth. Draw the Roofline model and plot where a ResNet-50 (Intensity: 50) and a Llama-3 (Intensity: 0.5) would fall."

### The Realistic Solution
1.  **Calculate the Ridge Point ($I^*$):** $I^* = \frac{500 \text{ TFLOPS}}{500 \text{ GB/s}} = 1.0 \text{ FLOPs/Byte}$.
2.  **Plot ResNet-50:** Since $50 > 1.0$, the model is **Compute Bound**. It sits on the flat part of the roof.
3.  **Plot Llama-3:** Since $0.5 < 1.0$, the model is **Memory Bound**. It sits on the slanted "memory" part of the roof.
**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b>WHITEBOARD: KV-Cache Scaling Logic</b></summary>

### The Prompt
"We are increasing the context window of our model from 8k to 128k tokens. What happens to our serving capacity? Give me the exact memory scaling logic."

### The Realistic Solution
KV-cache grows linearly with sequence length ($S$) and batch size ($B$). At 128k tokens, the KV-cache for a 70B model can exceed 100GB per request. You cannot use static allocation; you must use PagedAttention or you will waste 40%+ of your HBM to fragmentation.
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

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
<!-- ALL-CONTRIBUTORS-LIST:END -->
