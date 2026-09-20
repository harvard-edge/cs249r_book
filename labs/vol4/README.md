# Volume IV physical AI studio

**Status:** Curriculum and feasibility drafts for an ETH spring project seminar. The proposed primary station combines Hugging Face LeRobot, a Seeed SO-101 arm, and an Arduino UNO Q. The teaching team must qualify the arm command path, local learned policy, physical outcome measurement, and station capacity before releasing student assignments. The [instructor syllabus](../../instructors/vol4/README.md) lives outside the book front matter.

## The project students build

The course's unit of work is a **physical episode**: a versioned learned model on Qualcomm Linux proposes an action; the STM32 permits or refuses it; the actuator moves; the team measures the resulting tool and object state; and a fresh observation changes the next proposal or causes abstention. This satisfies the book's [three-part Physical AI scope test](../../books/vol4/01_boundary/01_boundary.qmd): learned decision, consequential physical feedback, and delegated actuation. A model that only classifies camera frames, or a model output that triggers a fixed arm script, cannot satisfy the capstone.

The common reference project is **disturbance-and-recovery manipulation**. The SO-101 reaches toward a soft target in a marked workspace; after its first action, staff safely shift the target or cause an incomplete move. The system must compare intended with measured state and decide again. Grasp-and-place, active inspection, and instruction-conditioned sorting are extensions once this smaller loop works. The capstone compares repeated held-out disturbance trials with a scripted baseline and includes a measured object or task outcome. Model score, latency, and memory are recorded as constraints on that result, not as the result itself.

The book's principles are visible in the same episode: commanded versus settled state exposes physical causality; the next observation exposes endogenous data; MCU refusal exposes proposal–permission privilege; and the frozen fault and outcome record supports a bounded release claim. The arm teaches the contact-dominated manipulation regime. Kinetic mobility and thermal/process regimes need separately qualified bodies for hands-on claims; seminar and simulator comparisons can introduce their different physical deadlines. The classroom station is a prototype within a measured envelope, not a certified safety system.

## Curriculum and build documents

- [Student competency matrix](student-competencies.md): platform-independent 2×2 checklist and evidence for each outcome.
- [Course operating plan](course-architecture.md): the physical episode, semester milestones, project choices, book-principle mapping, and assessment.
- [First-offering teaching plan](../../instructors/vol4/first-offering-plan.md): Vijay and Andrea's roles, four graded milestones, reading scope, and rehearsal buffer.
- [Fourteen lab blueprints](lab-blueprints.md): staff overview with links to 14 weekly studio prompts. The [course page](../../instructors/vol4/README.md) pairs each brief with its chapter source. Six common labs lead to eight project studios; the briefs are not fourteen separately graded assignments.
- [Feasibility plan](feasibility-plan.md): the proposed student exercise for each week, the question Andrea should test on the kit, and a simple report-back format.
- [One-station bench notes](station-pilot/README.md): optional technical detail for hardware investigation, with a [bench report](station-pilot/report-template.md) and [student bench card](station-pilot/bench-card-template.md).
- [SO-101/UNO Q course candidate](so101-uno-q-course.md): arm station and hardware tests.
- [Hugging Face curriculum spine](hugging-face-spine.md): LeRobot data, policy, simulation, correction, and re-evaluation workflow.
- [Candidate kit](kit-bom.md) and [staff build brief](staff-build-brief.md): parts, motor route, and integration risks.
- [Archetype proxy sketch](archetype-proxies.md): candidate mass, contact, and thermal systems.
- [One-axis fallback](lab-sequence.md): a simpler governed body if the SO-101's MCU motor route cannot be qualified. The arm can still teach LeRobot data and policy work in that case.

## What staff must test before enrollment

Stock SO-101 LeRobot operation uses a host USB motor-bus adapter and does not by itself place the STM32 between policy and motor. Staff must reproduce a sole live Qualcomm → Bridge → STM32 → servo-bus command route, with refusal and measured readback. They must also run a compact Hugging Face policy locally on the exact Q, show a changed-scene proposal after a physical move, and measure success independently of joint readback. SmolVLA on the Q is a separate experiment until its memory and timing are measured. The [feasibility plan](feasibility-plan.md) asks Andrea to test which student labs can use that station and what each one would require.
