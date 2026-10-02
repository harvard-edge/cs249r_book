# Week 1: Getting Started

**Status:** `skeleton`\
**Weeks:** 1\
**Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), and before the first seminar, the [reader guide and prerequisites](../../../books/vol4/frontmatter/prerequisites.qmd)\
**Competencies:** first look at `A1` and `D1` (signed off later)\
**Milestone:** none; feeds [Milestone 1](../curriculum/syllabus.md#7-assessment-milestones-and-grading)

---

## 1. Overview

**The physical question.** What changes when a computer's output is not text on a screen but force on a motor?

**Purpose.** This week gets you up and running. You set up your tools, learn the two safety rules, and drive the robot on day one: teleoperate it, record a motion, replay it, and cut its power to see what happens. You meet the machine as a black box that already moves safely. The rest of the semester explains why it does: in Lab 1 you model its body, and in Lab 6 you replace the safety governor you are trusting today with one you wrote yourself.

**Core concepts:**
- **The causal boundary:** where computation stops being symbolic and becomes physical force
- **Safe operation:** arming, disarming, and cutting motor power before anything else
- **The authority path:** every motion you command passes through the STM32 before it reaches a motor

**Goes out to Lab 1:** a working station, your tools set up, and a recorded episode that replays.\
**If something didn't finish:** staff help you finish setup before Lab 1.

---

## 2. Learning Objectives

By the end of this week, you will be able to:
- Set up the course tools and connect to your station
- Arm the station, teleoperate the arm, and cut motor power safely
- Record a demonstration and replay it
- Trace the path a command takes from your computer to a motor

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

> **Two rules from day one.** If a servo chatters, buzzes, stalls, or overheats, switch off motor power within two seconds and call staff before re-arming. Never connect a computer directly to the servo bus, around the STM32; doing so fails the current milestone.

**Before the first lab:** set up accounts and software. TBD (staff)

<!-- STAFF: list the accounts (GitHub, Hugging Face), the Python environment,
     and how students reach their station. Students should arrive at the
     first lab with this done. -->

### Part 0: Meet the Robot (Week 1)

**Goal:** drive the arm, record a motion, replay it, and know how to stop it. **Checkpoint:** a recorded episode that replays, and a safe-state check.

TBD (staff)

<!-- STAFF: expected to cover a station tour and cable trace; teleoperation;
     record and replay a demonstration; cut motor power at a raised pose and
     confirm nothing moves until re-arm. This week must be fun and safe: low
     speed, soft targets, a staff member at the bench, and every motion
     through the STM32 baseline governor, never the stock USB path. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Tools set up and station connected** | TBD (staff) |
| 2 | **A recorded episode that replays on your station** | TBD (staff) |
| 3 | **Safe-state check:** motor power cut at a raised pose, no motion until re-arm | TBD (staff) |

*Nothing is signed off this week; the safe-state check counts toward `A1` in Lab 1.*

---

## 5. Reflection

1. When you cut motor power, the arm stopped safely. Which part of the system made that true, and what would happen if it were missing?
2. Your replayed motion was not exactly the one you recorded. Where could the difference have come from?
