# AI Systems Foundations & Hardware 🧱

Fundamental principles of ML systems engineering, hardware constraints, and the Software 2.0 paradigm.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Foundations.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>🟢 LEVEL 1: What is the primary difference between Software 1.0 and Software 2.0?</b></summary>

**Answer:** The shift from instruction-centric to data-centric computing.
**Realistic Solution:** In Software 1.0, engineers write explicit logic (`if x then y`). In Software 2.0, engineers provide data and optimization objectives, and the system "compiles" the logic into weights. Debugging shifts from tracing code paths to inspecting data distributions.
**📖 Deep Dive:** [Volume I: Introduction](https://mlsysbook.ai/vol1/introduction.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Explain the '95% Problem' in production ML systems.</b></summary>

**Answer:** The model code is only a tiny fraction of the overall system.
**Realistic Solution:** Based on Sculley et al. (2015), the actual ML code is ~5% of the total codebase. The other 95% is "Glue Code"—ingestion, monitoring, feature extraction, and resource management. Optimizing the model alone ignores the largest sources of technical debt and system friction.
**📖 Deep Dive:** [Volume I: Introduction](https://mlsysbook.ai/vol1/introduction.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: Why is it mathematically impossible to 'verify' an image classifier?</b></summary>

**Answer:** The high-dimensional input space creates a structural 'Verification Gap.'
**Realistic Solution:** A standard $224\times224$ RGB image has $256^{150,528}$ possible configurations. Even the largest test sets (ImageNet) sample a negligible fraction of this space. Unlike traditional software, you cannot guarantee correctness; you must engineer for **statistical reliability** and real-time observability.
**📖 Deep Dive:** [Volume I: Introduction](https://mlsysbook.ai/vol1/introduction.html)
</details>
