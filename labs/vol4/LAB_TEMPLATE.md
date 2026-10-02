# Volume IV Lab Template

Every Volume IV lab has five sections in this order. Copy the block below to `labs/lab-NN-slug.md` to start a new lab.

| Section | Written by | Purpose |
|:---|:---|:---|
| **1. Overview** | Instructor | The physical question, why the lab exists, its core concepts, and where it sits in the course |
| **2. Learning Objectives** | Instructor | What a student can do afterward |
| **3. Instructions** | Staff | Setup, then one sub-section per weekly part: steps, tests, and pitfalls |
| **4. Deliverables** | Instructor (staff add thresholds) | The evidence each team hands in, and how it is signed off |
| **5. Reflection** | Instructor | Questions that connect the measurement back to the book |

**Labs and parts.** A lab teaches one core set of concepts. It is made of parts, one per week. Each part is one supervised studio session with a goal and a **checkpoint**, the artifact staff see before the team moves on. The goal and checkpoint are set by the instructor; the steps are written by staff.

**Status.** Each lab carries one status: `skeleton` (sections 1, 2, 4, 5 written; instructions open), `draft` (instructions written, not yet run on the bench), or `bench-verified` (a staff member has run the whole lab on the reference station, and every threshold comes from that run).

**Conventions.** `TBD (staff)` marks a slot staff fill. Guidance for staff goes in `<!-- STAFF: … -->` comments, which show in the source but not on the rendered page. Bench hardware, wiring, the four action taps, and the telemetry format are defined once in the [Station Reference Card](labs/station-reference.md); labs link there instead of repeating them.

---

```markdown
# Lab N: Title

**Status:** `skeleton`\
**Weeks:** X–Y (Parts NA, NB)\
**Reading:** [Chapter A: Title](../../../books/vol4/AA_slug/AA_slug.qmd)\
**Competencies:** `A1`, `D1`\
**Due:** end of Week W

---

## 1. Overview

**The physical question.** One or two sentences: the systems relationship this lab measures.

**Purpose.** One paragraph: why this lab exists, and what the student understands at the end that they did not before.

**Core concepts:**
- **Concept name:** the one idea, in a phrase. Three or four per lab; every part should serve at least one.

**Comes in from Lab N−1:** artifacts this lab uses.\
**Goes out to Lab N+1:** artifacts this lab produces.\
**If Lab N−1 didn't finish:** the staff-provided substitute.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Observable ability, verb first.

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md).

**Additional setup:** TBD (staff)

### Part NA: Name (Week X)

**Goal:** one sentence. **Checkpoint:** the artifact staff check.

TBD (staff)

<!-- STAFF: steps with real commands, what to record, the expected result,
     any disturbance or failure test, and pitfalls from the Student Zero run.
     One part must fit one 3-hour studio session; long compute runs between
     sessions. -->

### Part NB: Name (Week Y)

**Goal:** one sentence. **Checkpoint:** the artifact staff check.

TBD (staff)

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **What the team hands in** | TBD (staff) |

<!-- STAFF: thresholds come from the Student Zero run on the reference
     station, not from a spec sheet. -->

*Staff signs off `A1` and `D1` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. A question that asks what the measurement means.
2. A question that connects it back to the chapter.
```
