# Week 1: Getting Started

**Status:** `skeleton`\
**Weeks:** 1\
**Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), and before the first seminar, the [reader guide and prerequisites](../../../books/vol4/frontmatter/prerequisites.qmd)\
**Competencies:** none signed off this week\
**Due:** nothing graded; Lab 1 starts in Week 2

---

## 1. Overview

**The physical question.** What changes when a computer's output is not text on a screen but force on a motor?

**Purpose.** This week makes sure everything works before the real work starts. You get your kit ready, learn how the pieces fit together and where each command goes, and check that your computer can talk to the robot. You also learn the two safety rules and how to cut motor power. By the end of the week your station is ready, and in Lab 1 you start using it.

**Core concepts:**
- **The causal boundary:** where computation stops being symbolic and becomes physical force
- **How the station fits together:** camera, Linux processor, STM32 governor, servos, and two separate power supplies
- **The authority path:** every motion passes through the STM32 before it reaches a motor

**Goes out to Lab 1:** a ready station, a working computer setup, and a team that can arm, move, and stop the robot safely.\
**If something didn't finish:** staff help you finish setup before Lab 1.

---

## 2. Learning Objectives

By the end of this week, you will be able to:
- Check your kit and get your station ready
- Explain the path a command takes from your computer to a motor, and what each piece does
- Set up the course software on your computer and connect to your station
- Arm the station, move a joint, and cut motor power safely

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

> **Two rules from day one.** If a servo chatters, buzzes, stalls, or overheats, switch off motor power within two seconds and call staff before re-arming. Never connect a computer directly to the servo bus, around the STM32; doing so fails the current lab.

### Part 0: Ready Your Station (Week 1)

**Goal:** a station, a computer, and a team that are ready for Lab 1. **Checkpoint:** a completed station check, signed by staff.

TBD (staff)

<!-- STAFF: expected to cover
     - the kit: check every part against the kit list and get the station ready;
     - how it works: a guided tour of the station, tracing a command from
       laptop to Linux to STM32 to servo, and the two power supplies;
     - the computer: install the course software, set up accounts (GitHub,
       Hugging Face), and connect to the station;
     - a first motion check: arm, move one joint at low speed, cut motor
       power, and confirm nothing moves until re-arm.
     Keep it low-pressure: a staff member at the bench, every motion through
     the STM32 baseline governor, never the stock USB path. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Kit checked and station ready** | TBD (staff) |
| 2 | **Computer set up and connected to the station** | TBD (staff) |
| 3 | **Command path sketch:** from your computer to a motor, with each piece labeled | TBD (staff) |
| 4 | **Safe-stop check:** motor power cut, no motion until re-arm | TBD (staff) |

*Nothing is graded this week; the station check must be complete before Lab 1.*

---

## 5. Reflection

1. When you cut motor power, the arm stopped. Which part of the system made that true, and what would happen if it were missing?
2. Your command crossed several pieces of hardware before a motor moved. Which one would you trust least, and why?
