# Capstone Project Studio: Physical Release Defense

**Schedule:** Weeks 12–14 (3 Full Weeks Dedicated Studio)
**Required Textbook Reading:** Complete Volume IV Synthesis ([Chapters 1–17](../../../books/vol4/index.qmd))
**Target Competencies:** `[ ] D3 Physical Release Defense & Evidence Dossier` *(Full 12-Card Mastery)*
**Milestone Deliverable:** [Milestone 5](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 14) — *Capstone System Defense & Physical Release Dossier*

---

### 1. The Physical Question
How does an engineering team defend a formal, evidence-backed safety and release claim for an autonomous embodied machine? When operating under real-world physical variability, how do we prove where the system is dependable, where it safely abstains, and that it never violates its safety governor?

---

### 2. Studio Operating Rhythm (Weeks 12–14)

Formal lectures and new lab assignments have concluded. The final three weeks give student teams dedicated bench runway:

| Week | Focus | What the team does |
|:---:|:---|:---|
| **12** | Task build & policy hardening | Freeze the fixture and task (sorting, insertion, or bin picking). Collect a custom 50-episode LeRobot dataset with all four action taps. Fine-tune SmolVLA or ACT and export the quantized ONNX model to the UNO Q. |
| **13** | Adversarial peer exchange & dress rehearsal | Swap fault challenges with an assigned peer team, which introduces unannounced disturbances (lighting drops, obstacles). Harden the STM32 governor, replanning, and abstention boundaries. |
| **14** | Graduation trial & oral defense | Run 20 live, unedited trials (10 baseline + 10 disturbance). Submit the Physical Release Dossier. Each student defends the full causal loop individually. |

---

### 3. The 20-Trial Physical Benchmark Protocol

During the final exam session in Week 14, each team executes **20 repeated physical trials** on hardware before the teaching panel:

1. **Trials 1–10 (Baseline Operating Envelope):**
   * Target objects placed at 10 randomized, held-out coordinates across the characterized workspace.
   * Zero human intervention allowed.
   * Evaluate positioning precision, reach duration, and final object state.
2. **Trials 11–20 (Physical Disturbance Trials):**
   * The examination panel or peer team introduces live disturbances:
     * *Disturbance A (Mid-Reach Shift):* Object displaced by 50 mm during active trajectory.
     * *Disturbance B (Illumination Drop):* Overhead ambient lighting reduced by 70%.
     * *Disturbance C (Compliant Obstacle):* A soft foam barrier placed along the nominal approach path.
     * *Disturbance D (Missing Target):* Target block removed completely before approach.
   * The system must demonstrate adaptive recovery or justified, safe abstention. Zero unhandled collisions or hardware stalls are permitted.

---

### 4. The Physical Release Dossier Specification

Each team submits a formal, versioned technical engineering document containing:

1. **Claim-Argument-Evidence (CAE) Framework:**
   * *Claim:* The exact task success rate and safety guarantees the system provides.
   * *Operational Design Domain (ODD):* Precise physical boundaries (allowable lighting lux, object weight range, temperature, workspace volume).
   * *Evidence:* Quantitative data from the 20 physical trials, telemetry plots, and latency distributions.
2. **Baseline Comparative Analysis:**
   * Matched comparison against the deterministic scripted baseline. Where does the neural policy provide superior generalization, and where is the baseline faster or more predictable?
3. **Failure Mode Taxonomy & Abstention Policy:**
   * Complete inventory of observed failure modes. Documentation proving the system safely halts or abstains rather than executing uncontrolled motion.
4. **Complete Traceability Archive:**
   * Git commit hash of all software and firmware.
   * Hub link to the curated LeRobot evaluation dataset recording all 4 action taps for every trial.

---

### 5. Individual Oral Defense Protocol

Following the 20 physical trials, each team member is examined individually by the panel:
* **Student A (The Brain):** Traces a randomly selected episode from raw camera pixels through model preprocessing, SmolVLA/ACT inference, and action chunk proposal generation.
* **Student B (The Governor):** Explains the STM32 MCU permission log, boundary checks, velocity clamps, and why a specific proposal was permitted or clipped.
* **Student C (The Physics):** Interprets the physical discrepancy between commanded trajectory ($a_{\text{req}}$), motor response ($a_{\text{meas}}$), and final material outcome.

---

### 6. Sign-Off & Graduation Criteria
To achieve course graduation and Milestone 5 completion:
1. [ ] **20-Trial Execution:** Complete 20 witnessed physical trials with zero unhandled collisions or hardware crashes.
2. [ ] **Physical Release Dossier:** Submit an approved, auditable Claim-Argument-Evidence release dossier.
3. [ ] **Oral Defense:** Every team member successfully defends their individual causal trace.
4. [ ] **Complete Competency Mastery:** All 12 competencies (A1–A3, B1–B3, C1–C3, D1–D3) signed off on the [Competency Card](../curriculum/student-competencies.md).

> **A Note on Engineering Integrity:** A team that concludes their system **should NOT be released** for production due to measured timing or reliability limitations—and provides rigorous physical evidence supporting that conclusion—receives full honors. Proving where a physical system fails is the hallmark of a true systems engineer.
