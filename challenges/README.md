# The AI Systems Arena 🏆

This is the practice arena for AI Systems Engineering. While most software engineering challenges focus on algorithms, here we focus on **System Physics**.

Each challenge gives you a high-stakes design scenario and a set of constraints (SLAs). Your job is to architect a solution that doesn't just work in code, but survives the **22 Walls** of silicon reality.

---

## 🕹️ How to Play

1.  **Select a Challenge:** Pick a folder (e.g., [`01_the_memory_wall`](01_the_memory_wall/README.md)).
2.  **Architect your Solution:** Determine the GPU count, precision, and parallelism strategy.
3.  **Verify your Math:** Use **MLSys·im** to evaluate your design.
    ```bash
    mlsysim eval <MODEL> <HARDWARE> --batch-size <B> --precision <P> --nodes <N>
    ```
4.  **Beat the SLA:** If the output satisfies the prompt's constraints, you've engineered a viable system.

---

## 🤝 Community: Scale the Arena

We are building a collection of real-world AI systems challenges. Have you seen a unique system design problem?

- **[Submit a New Challenge](https://github.com/harvard-edge/cs249r_book/issues/new?template=new_challenge.yml)**
- **[Submit an Interview Question](https://github.com/harvard-edge/cs249r_book/issues/new?template=interview_question.yml)**

---

> "Theory is when you know everything but nothing works. Engineering is when everything works and you know why."
> — *Machine Learning Systems: Principles and Practices*
