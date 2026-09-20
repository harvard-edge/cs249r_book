# Volume IV: Physical AI Pedagogical Dashboard

**Book**: *Physical AI: Machine Learning Systems That Sense and Act*
**Author**: Prof. Vijay Janapa Reddi (Harvard University)
**Last Synchronized**: `2026-09-19T13:50:32.123336`
**Pipeline**: Autonomous Two-Pass Agentic Loop (Pass 1 Grounding + Pass 2 Pruning)

---

## Overall Progress

```text
Progress: ████████████████████  100% (17/17 chapters completed)
```

- 🟢 **Completed Chapters**: 17 / 17
- 🟡 **In-Progress Chapters**: 0
- ⚪ **Pending Chapters**: 0
- 🛠️ **Total Pass 1 Scaffolding Edits Applied**: **147**
- ✂️ **Total Pass 2 Words Pruned**: **4055 words**

---

## Chapter-by-Chapter Status Table

| Ch | Part | Title | Status | Sections Completed | Pass 1 Edits | Pass 2 Cuts | Words Trimmed |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **01** | Part I: Anatomy | **The Causal Boundary** | 🟢 Completed | 31/31 | 10 | 71 | -240 w |
| **02** | Part I: Anatomy | **The Body** | 🟢 Completed | 20/20 | 9 | 45 | -217 w |
| **03** | Part I: Anatomy | **The Brain** | 🟢 Completed | 28/28 | 11 | 68 | -306 w |
| **04** | Part I: Anatomy | **The Nervous System** | 🟢 Completed | 32/32 | 16 | 75 | -382 w |
| **05** | Part II: Teaching | **Data** | 🟢 Completed | 33/33 | 16 | 68 | -323 w |
| **06** | Part II: Teaching | **Training** | 🟢 Completed | 22/22 | 5 | 50 | -195 w |
| **07** | Part II: Teaching | **Evaluation** | 🟢 Completed | 21/21 | 8 | 48 | -234 w |
| **08** | Part III: Running | **Perception** | 🟢 Completed | 30/30 | 6 | 61 | -279 w |
| **09** | Part III: Running | **Memory** | 🟢 Completed | 14/14 | 1 | 28 | -107 w |
| **10** | Part III: Running | **Intent** | 🟢 Completed | 31/31 | 20 | 74 | -297 w |
| **11** | Part III: Running | **Planning** | 🟢 Completed | 19/19 | 7 | 46 | -241 w |
| **12** | Part III: Running | **Enforcement** | 🟢 Completed | 16/16 | 2 | 39 | -127 w |
| **13** | Part III: Running | **Placement** | 🟢 Completed | 13/13 | 4 | 35 | -153 w |
| **14** | Part IV: Governing | **Intervention** | 🟢 Completed | 28/28 | 15 | 69 | -366 w |
| **15** | Part IV: Governing | **Verification** | 🟢 Completed | 11/11 | 4 | 28 | -177 w |
| **16** | Part IV: Governing | **Release** | 🟢 Completed | 26/26 | 7 | 58 | -243 w |
| **17** | Conclusion | **The Frontier** | 🟢 Completed | 21/21 | 6 | 48 | -168 w |

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
