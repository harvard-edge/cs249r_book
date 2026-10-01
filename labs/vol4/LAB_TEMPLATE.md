# Volume IV Lab Template

Every Volume IV lab follows this template, section for section and in this order. Copy it to `labs/lab-NN-slug.md` to start a new lab.

**How to read this file.** Text in normal type is the structure students will see. Text inside `<!-- STAFF: … -->` comments is guidance for the person writing the lab; it is visible in the source but not on the rendered GitHub page. `TBD (staff)` marks a slot that has not been filled yet.

**Lab status.** Each lab carries one status in its header:

| Status | Meaning |
|:---|:---|
| `skeleton` | Purpose, question, readings, competencies, and phase outline are set. Procedures, commands, and thresholds are not. |
| `draft` | Every section is written, but at least one procedure or threshold has not been run on the bench. |
| `bench-verified` | A staff member has run the whole lab as Student Zero on the reference station, and every threshold comes from that run. |

**What lives elsewhere.** Bench hardware, wiring, the three engineering rules, the four action taps, and the telemetry CSV format are defined once in the [Station Reference Card](labs/station-reference.md). Labs link there and add only what is specific to the lab. Competency definitions live in the [Competency Card](curriculum/student-competencies.md); milestone weights live in the [syllabus](curriculum/syllabus.md).

---

```markdown
# Lab N: Title

**Status:** `skeleton`\
**Weeks:** X–Y\
**Reading:** [Chapter A: Title](../../../books/vol4/AA_slug/AA_slug.qmd), [Chapter B: Title](…)\
**Competencies:** `A1`, `D1`\
**Milestone:** Contributes to / Concludes [Milestone M](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week W)

---

## Overview

**The physical question.** One or two sentences: the systems relationship this lab measures or proves on hardware.

**Purpose.** One paragraph: why this lab exists in the sequence, and what the student understands at the end that they did not before.

**Comes in from Lab N−1:** the artifacts this lab consumes.\
**Goes out to Lab N+1:** the artifacts this lab produces.

### Learning outcomes

By the end of this lab, you will be able to:
- Outcome phrased as an observable ability (verb first).

<!-- STAFF: 3–6 outcomes. Each one should map to a sign-off criterion below. -->

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md).

Additional for this lab:
- TBD (staff)

<!-- STAFF: list only what differs from the standard station: extra hardware,
     fixtures, software, starter files, and the required initial state
     (e.g., motor power off, arm at rest). -->

---

## 2. Protocol

### Phase A: Name (Week X)

**Goal:** one sentence.

1. **Step name.** TBD (staff)
2. **Step name.** TBD (staff)

<!-- STAFF: for each step give the procedure, the exact commands or code entry
     points, what to record, and the expected result. Every command must exist
     in the starter repo and have been run on the reference station. -->

### Phase B: Name (Week Y)

**Goal:** one sentence.

1. **Step name.** TBD (staff)

---

## 3. Disturbance & Failure Tests

- **Test name.** What is perturbed or injected. Pass condition: TBD (staff).

<!-- STAFF: at least one deliberate physical disturbance or injected fault, with
     an observable pass condition. Simulation-only labs use an out-of-envelope
     or degenerate case instead. -->

---

## 4. Telemetry Trace

Which [action taps](station-reference.md#the-four-action-taps) to capture, during which event, and what the trace must show.

<!-- STAFF: name the plot or CSV the student submits and what it proves. -->

---

## 5. Common Pitfalls

TBD (staff)

<!-- STAFF: fill from the Student Zero run. Each pitfall: symptom, cause, fix. -->

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Criterion name** | TBD (staff) |

<!-- STAFF: 3–6 criteria, each observable at the bench. Thresholds come from the
     Student Zero run on the reference station, not from a spec sheet. -->

*Staff signs off `A1` and `D1` on the team's [Competency Card](../curriculum/student-competencies.md).*
```
