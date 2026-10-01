# Capstone: Physical Release Defense

**Status:** `skeleton`\
**Weeks:** 12–14 (Parts C1, C2, C3), with the proposal due Week 8 and the task frozen Week 10\
**Reading:** [Chapter 16: Safe Release](../../../books/vol4/16_release/16_release.qmd), [Chapter 17: The Epistemic Frontier](../../../books/vol4/17_frontier/17_frontier.qmd)\
**Competencies:** `D3` (and the full card)\
**Milestone:** [Milestone 5](../curriculum/syllabus.md#7-assessment-milestones-and-grading) (End of Week 14)

---

## 1. Overview

**The physical question.** How does a team defend an evidence-backed release claim for an autonomous machine? Under real physical variability, how do you show where the system is dependable, where it safely abstains, and that it never gets past its governor?

**Purpose.** Lectures and new labs are over. The final three weeks give each team bench time to run the whole pipeline on a task of its own, survive a peer team's attacks, and defend a written release decision. The thinking starts earlier: you propose the task in week 8 and freeze it in week 10, so the studio weeks go to building and testing, not deciding. A team that concludes its system should *not* be released, and backs that with rigorous evidence, can earn full marks. Showing where a physical system fails is the job.

**Comes in from the previous labs:** the full pipeline, from kinematics through your own governor.\
**Goes out:** the Physical Release Dossier and the oral defense.\
**If the previous lab didn't finish:** staff provide reference governor firmware.

---

## 2. Learning Objectives

By the end of the capstone, you will be able to:
- Define a manipulation task and the operating envelope in which you claim it works
- Evaluate a learned system against a baseline under held-out conditions and disturbances
- Build a claim–argument–evidence case from your own telemetry
- Explain what would change if the pipeline moved to a different robot
- Defend, individually, any trace from your system end to end

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md), with your own governor from Lab 6.

**Before the studio:** a **proposal** at the end of Week 8 (task, success measure, baseline, main risk) and a **frozen task** at the end of Week 10 (fixture, objects, operating envelope, trial protocol). Format: TBD (staff)

<!-- STAFF: give feedback on proposals during the Lab 5 and Lab 6 sessions so
     teams arrive at Week 12 ready to build. -->

### Part C1: Task Build (Week 12)

**Goal:** the frozen task runs on your pipeline. **Checkpoint:** the task running end to end on your station.

TBD (staff)

<!-- STAFF: expected to cover the custom dataset, fine-tuning, and deployment. -->

### Part C2: Peer Fault Exchange (Week 13)

**Goal:** survive another team's attacks and harden the system. **Checkpoint:** a fault manifest and the fixes made in response.

TBD (staff)

<!-- STAFF: expected to cover how teams are paired, what attacks are allowed,
     and the safety rules for the exchange. -->

### Part C3: Defense (Week 14)

**Goal:** run the final trials and defend the release decision. **Checkpoint:** trial logs, the submitted dossier, and the defense.

TBD (staff)

<!-- STAFF: expected to cover the final trial protocol (baseline trials at
     held-out positions, then trials under live disturbances: number, types,
     pass rules) and the oral defense format. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Final trials** at held-out positions and under live disturbances | TBD (staff) |
| 2 | **Physical Release Dossier:** claim, argument, and evidence with the operating envelope; baseline comparison; failure modes and abstention policy; transfer analysis (what would change on a different robot, such as a mobile base); code, firmware, dataset, and trial-log versions | TBD (staff) |
| 3 | **Individual oral defense:** each student explains one part of the loop, the brain (pixels to proposed action), the governor (why a proposal was permitted or clipped), or the physics (commanded versus measured motion) | TBD (staff) |
| 4 | **Full competency card** | All twelve competencies signed off |

*Staff signs off `D3` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. What is the strongest claim your evidence supports, and what is the first thing that would make it false?
2. If you had one more week, would you spend it on more data, a better governor, or more trials? Defend the choice with your own measurements.
