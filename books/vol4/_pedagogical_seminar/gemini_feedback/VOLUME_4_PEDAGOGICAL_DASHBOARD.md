# Volume IV: Physical AI Pedagogical Dashboard

**Book**: *Physical AI: Machine Learning Systems That Sense and Act*
**Author**: Prof. Vijay Janapa Reddi (Harvard University)
**Last Synchronized**: `2026-09-19T14:58:17.441242`
**Pipeline**: Autonomous Two-Pass Agentic Loop (Pass 1 Grounding + Pass 2 Pruning)

---

## Overall Progress

```text
Progress: ░░░░░░░░░░░░░░░░░░░░  0% (0/17 chapters completed)
```

- 🟢 **Completed Chapters**: 0 / 17
- 🟡 **In-Progress Chapters**: 4
- ⚪ **Pending Chapters**: 13
- 🛠️ **Total Pass 1 Scaffolding Edits Applied**: **1**
- ✂️ **Total Pass 2 Words Pruned**: **775 words**

---

## Chapter-by-Chapter Status Table

| Ch | Part | Title | Status | Sections Completed | Pass 1 Edits | Pass 2 Cuts | Words Trimmed |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **01** | Part I: Anatomy | **The Causal Boundary** | 🟡 In Progress | 6/22 | 0 | 15 | -260 w |
| **02** | Part I: Anatomy | **The Body** | 🟡 In Progress | 5/12 | 0 | 10 | -195 w |
| **03** | Part I: Anatomy | **The Brain** | 🟡 In Progress | 6/19 | 0 | 16 | -157 w |
| **04** | Part I: Anatomy | **The Nervous System** | 🟡 In Progress | 5/13 | 1 | 14 | -163 w |
| **05** | Part II: Teaching | **Data** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **06** | Part II: Teaching | **Training** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **07** | Part II: Teaching | **Evaluation** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **08** | Part III: Running | **Perception** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **09** | Part III: Running | **Memory** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **10** | Part III: Running | **Intent** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **11** | Part III: Running | **Planning** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **12** | Part III: Running | **Enforcement** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **13** | Part III: Running | **Placement** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **14** | Part IV: Governing | **Intervention** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **15** | Part IV: Governing | **Verification** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **16** | Part IV: Governing | **Release** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |
| **17** | Conclusion | **The Frontier** | ⚪ Pending | 0/-- | 0 | 0 | -0 w |

---

## Student Cohort Reviewers

| Evaluator | Persona & Background | Core Oversight Mandate |
|:---|:---|:---|
| **Alex Chen** | MSc in CS / Deep Learning | Grounds foundation models in physical latency and actuation limits. |
| **Priya Patel** | PhD in Computer Systems | Enforces real-time bus determinism, memory bandwidth, and silicon partitioning. |
| **Marcus Vance** | PhD in MechE / Control | Enforces Newton-Euler mechanics, stopping distance bounds, and actuator saturation. |
| **Elena Rostova** | Senior Undergrad in EECS | **Guardian of Progressive Disclosure**: Audits acronyms, cognitive load, and flow. |
| **Dr. Aris Thorne** | Lead TA & Moderator | Convenes weekly seminar discussions and synthesizes surgical rewrites. |

---

## How to Control the Pipeline

```bash
# Resume audit for the current chapter (batch of 5 sections):
.venv/bin/python -m scripts.pedagogy_sim.book_orchestrator --step

# Process next chapter specifically:
.venv/bin/python -m scripts.pedagogy_sim.book_orchestrator --chapter 2 --batch-size 5

# Run whole book in parallel (4 concurrent chapters):
.venv/bin/python -m scripts.pedagogy_sim.book_orchestrator --all --parallel 4
```
