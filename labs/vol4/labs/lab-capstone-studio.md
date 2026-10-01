# Capstone: Physical Release Defense

**Status:** `skeleton`\
**Weeks:** 12–14 (Parts C1, C2, C3), with the proposal due Week 8 and the task frozen Week 10\
**Reading:** [Chapter 17: The Epistemic Frontier](../../../books/vol4/17_frontier/17_frontier.qmd), plus the [whole volume](../../../books/vol4/index.qmd) as synthesis\
**Competencies:** `D3` (and the full card)\
**Milestone:** [Milestone 5](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 14)

---

## Overview

**The physical question.** How does a team defend an evidence-backed release claim for an autonomous machine? Under real physical variability, how do you show where the system is dependable, where it safely abstains, and that it never gets past its governor?

**Purpose.** Lectures and new labs are over. The final three weeks give each team bench time to run the whole pipeline on a task of its own, survive a peer team's attacks, and defend a written release decision. The thinking starts earlier: you propose the task in week 8 and freeze it in week 10, so the studio weeks go to building and testing, not deciding. A team that concludes its system should *not* be released, and backs that with rigorous evidence, can earn full marks. Showing where a physical system fails is the job.

**Comes in from Labs 1–6:** the full pipeline, from kinematics through your own governor.\
**Goes out:** the Physical Release Dossier and the oral defense.\
**If Lab 6 didn't finish:** staff provide reference governor firmware.

### Learning outcomes

By the end of the capstone, you will be able to:
- Define a manipulation task and the operating envelope in which you claim it works
- Evaluate a learned system against a baseline under held-out conditions and disturbances
- Build a claim–argument–evidence case from your own telemetry
- Explain what would change if the pipeline moved to a different robot
- Defend, individually, any trace from your system end to end

---

## 1. Before the Studio

| When | Deliverable | Content |
|:---:|:---|:---|
| End of Week 8 | **Proposal** | The task, the success measure, the baseline, and the main risk. TBD (staff) |
| End of Week 10 | **Frozen task** | Fixture, objects, operating envelope, and trial protocol. TBD (staff) |

<!-- STAFF: give feedback on proposals during the Lab 5 and Lab 6 sessions so
     teams arrive at Week 12 ready to build. -->

---

## 2. Protocol

### Part C1: Task Build (Week 12)

**Goal:** the frozen task runs on your pipeline.

1. **Custom dataset.** TBD (staff)
2. **Fine-tune and deploy.** TBD (staff)

**Checkpoint:** the task running end to end on your station.

### Part C2: Peer Fault Exchange (Week 13)

**Goal:** survive another team's attacks and harden the system.

1. **Fault exchange with a peer team.** TBD (staff)
2. **Hardening.** TBD (staff)

**Checkpoint:** a fault manifest and the fixes made in response.

### Part C3: Defense (Week 14)

**Goal:** run the final trials and defend the release decision.

1. **Final trials.** Baseline trials at held-out positions, then trials under live disturbances. Number of trials, disturbance types, and pass rules: TBD (staff).
2. **Oral defense.** TBD (staff)

**Checkpoint:** trial logs, the submitted dossier, and the defense.

---

## 3. Physical Release Dossier

The dossier contains:
1. **Claim, argument, and evidence**, including the operating envelope
2. **Baseline comparison**
3. **Failure modes and abstention policy**
4. **Transfer analysis:** what would change in the envelope, the data, and the governor if this pipeline moved to a different robot, such as a mobile base
5. **Traceability**: code and firmware versions, dataset, and trial logs

Format and length: TBD (staff).

---

## 4. Oral Defense

Each student is examined individually on one part of the loop:
- **The brain:** a randomly chosen episode from pixels to proposed action
- **The governor:** why a specific proposal was permitted or clipped
- **The physics:** the gap between commanded and measured motion, and the final outcome

Format and timing: TBD (staff).

---

## 5. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Final trials completed** | TBD (staff) |
| 2 | **Physical Release Dossier** | TBD (staff) |
| 3 | **Individual oral defense** | TBD (staff) |
| 4 | **Full competency card** | All twelve competencies signed off |

*Staff signs off `D3` on the team's [Competency Card](../curriculum/student-competencies.md).*
