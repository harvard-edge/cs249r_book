# Volume IV: Physical AI Studio Labs

These reusable labs accompany [MLSysBook Volume IV](https://mlsysbook.ai/vol4/).
Teams measure a robot's physical limits, compare simulation with real hardware,
collect demonstrations, deploy a learned policy, and test an independent governor.

**Status:** Lab skeletons; staff are preparing and qualifying the instructions.

The course overview, semester schedule, and assessment policies now live on the
[Physical AI course website](https://mlsysbook.ai/courses/physical-ai/eth-spring-2027/).
See the [course syllabus](https://mlsysbook.ai/courses/physical-ai/eth-spring-2027/syllabus.html)
for readings and deadlines.

## Lab Instructions

- [Lab 1: Forward Kinematics](labs/lab1.md): Derive the arm's kinematic model and track its pose on the microcontroller.
- [Lab 2: The Safety Governor](labs/lab2.md): Build the stop mechanism and the governor that bounds every command.
- [Lab 3: Inverse Kinematics](labs/lab3.md): Solve for joint angles, map singularities, and drive the arm to target poses.
- [Lab 4: VLA Policies in Simulation](labs/lab4.md): Reproduce the workbench in simulation and benchmark a learned policy.
- [Lab 5: The Sim2Real Gap](labs/lab5.md): Deploy the policy on the robot and measure how it differs from simulation.
- [Lab 6: Real Data and Fine-Tuning](labs/lab6.md): Collect demonstrations and measure how much fine-tuning closes the gap.
- [Lab 7: Robustness Testing](labs/lab7.md): Compare single-brain and dual-brain deployments under CPU stress.
- [Lab 8: Capstone Project](labs/capstone.md): Defend a release decision with physical evidence.

## Shared Lab Resources

- [Station Reference Card](labs/station-reference.md): hardware, wiring, command path, and telemetry.
- [Competency Card](curriculum/student-competencies.md): skills demonstrated at the bench.
- [Lab Template](LAB_TEMPLATE.md): structure used by the lab authors.

Staff qualification and implementation notes remain under `staff/`.

## Chapter companions: explore before the physical studio

The chapter-aligned browser companions complement this hardware studio. Start with
[`lab_00_orientation.py`](lab_00_orientation.py), then follow the chapter order
through the final transfer investigation. Each chapter uses four concept-owned
parts, visible goals and route, reactive reference evidence, saved comparisons,
and one downloadable discussion. The [chapter inventory](../config/lab-inventory.qmd)
connects every companion to the studio session it supports.

The browser project is the book's warehouse mobile manipulator. The studio uses
the SO-101 station. A warehouse reference calculation is not a measurement of
that station, and a constructed incident trace is not a physical trial. These
companions issue no hardware commands and do not alter the studio's status or
replace staff qualification. Physical instructions and approved station limits
remain in the existing studio and Station Reference Card.

Concepts and student-visible goals are authored once in `concepts/`; the packaged
learning contracts are generated from those maps. MLSysIM owns study cases and
calculations. `mlsysbook_labs.physical_companion` owns the shared interaction;
`build_companions.py` checks the thin chapter entrypoints. No chapter notebook
contains a private copy of the physics or narrative.
