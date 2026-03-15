# AI Systems Interview Cheatsheet 🧠

> [!IMPORTANT]
> **Work in Progress:** This interview hub is currently being seeded with community-contributed questions. We are actively looking for "Silicon Realist" scenarios from recent interviews at top AI labs. [Contribute your experience here](#-community-contribute-intelligence).

The world's most comprehensive community-driven vault of **AI Systems Design** interview questions—from silicon physics to global fleets.

> **Our Mission:** Curate the high-signal questions actually asked at frontier labs (OpenAI, Anthropic, Meta, Google) and provide the "Silicon Realist" answers that LLMs hallucinate.

---

## 🏗️ The Curriculum: Silicon to Fleet Scale

Follow the natural evolution of an AI system. Each guide features interactive "Flashcards" for active recall.

> **[📄 View the 22 Walls Cheatsheet](WALLS_CHEATSHEET.md)**: A quick-reference guide to the equations and constraints.

**Phase 1: The Single Machine (Basics)**
1.  **[Foundations & Hardware](01_Foundations.md)**: `[VRAM Math]` `[Ridge Points]` `[HBM Physics]`
2.  **[Models & Architectures](02_Architectures.md)**: `[Attention Bottlenecks]` `[CNN vs MLP]`
3.  **[Data & Pipelines](03_Data_Pipelines.md)**: `[Storage I/O]` `[CPU Starvation]`

**Phase 2: The Global Infrastructure (Fleet Scale)**
4.  **[Distributed ML & Scaling](04_Distributed_ML.md)**: `[AllReduce]` `[3D Parallelism]` `[Bisection BW]`
5.  **[Serving & Operations](05_Serving_Ops.md)**: `[KV-Cache]` `[Tail Latency]` `[TCO]`

---

## 🖊️ The Whiteboard Challenge (Anti-Hallucination)

Interviewers at top labs focus on **Visual Logic and Precise Calculation**. These challenges test your ability to do napkin math on the fly—the one thing LLMs consistently get wrong.

<details>
<summary><b>WHITEBOARD: Draw the Roofline for a new "Mystery Chip"</b></summary>

### The Prompt
"I have a new accelerator with 500 TFLOPS of compute and 500 GB/s of HBM bandwidth. Draw the Roofline model and plot where a ResNet-50 (Intensity: 50) and a Llama-3 (Intensity: 0.5) would fall."

### The Realist Solution
1.  **Calculate the Ridge Point ($I^*$):** $I^* = \frac{500 \text{ TFLOPS}}{500 \text{ GB/s}} = 1.0 \text{ FLOPs/Byte}$.
2.  **Plot ResNet-50:** Since $50 > 1.0$, the model is **Compute Bound**. It sits on the flat part of the roof.
3.  **Plot Llama-3:** Since $0.5 < 1.0$, the model is **Memory Bound**. It sits on the slanted "memory" part of the roof.
</details>

<details>
<summary><b>WHITEBOARD: KV-Cache Scaling Logic</b></summary>

### The Prompt
"We are increasing the context window of our model from 8k to 128k tokens. What happens to our serving capacity? Give me the exact memory scaling logic."

### The Realist Solution
KV-cache grows linearly with sequence length ($S$) and batch size ($B$). At 128k tokens, the KV-cache for a 70B model can exceed 100GB per request. You cannot use static allocation; you must use PagedAttention (Wall 5) or you will waste 40%+ of your HBM to fragmentation.
</details>

---

## 🗳️ The Voting Arena (Community Choice)

Mastery is built by the community. We use **GitHub Issues** as a public "Drafting Room" for new questions. 

- **[Vote on Upcoming Questions](https://github.com/harvard-edge/cs249r_book/issues?q=is%3Aissue+is%3Aopen+label%3Ainterview-prep)**: Give a **👍 Reaction** to the questions you want to see added to the official hub.
- **The 10-Upvote Rule:** When a community question hits **10 upvotes**, we promote it to the official Phase 1/Phase 2 guides and add your avatar to the **Hall of Fame**.

### 🔥 Trending Community Scenarios
<!-- TRENDING-QUESTIONS-START -->
*No trending questions yet. Be the first to [submit a question from a recent interview](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml)!*
<!-- TRENDING-QUESTIONS-END -->

---

## 🤝 Community: Contribute Intelligence

Did you have an AI Systems interview recently? Help the community by sharing the scenario.

**How to contribute:**
1.  **Submit an Issue:** Use the **[💼 New Interview Question](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml)** template.
2.  **Tag the Company:** Add tags like `[OpenAI]` or `[Meta]` to the title.
3.  **PR to the Vault:** If your question hits 10 upvotes, we'll merge your PR to the markdown files using our **[Copy-Paste Templates](01_Foundations.md#clipboard-copy-paste-template)**.

---

> "The world is rushing to build AI systems. It is not engineering them."
> — *Machine Learning Systems: Principles and Practices*

## Contributors

Thanks goes to these wonderful people who have contributed to the Interview Hub!

**Legend:** 🪲 Bug Hunter · 🧑‍💻 Code Contributor · ✍️ Doc Wizard · 🎨 Design Artist · 🧠 Idea Spark · 🔎 Code Reviewer · 🧪 Test Tinkerer · 🛠️ Tool Builder

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
