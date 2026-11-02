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

- [Getting Started](labs/week-01-getting-started.md): Prepare the station and learn the operating rules.
- [Lab 1: The Digital Twin](labs/lab-01-digital-twin.md): Measure the body and build a kinematic model.
- [Lab 2: Simulation to Real Hardware](labs/lab-02-sim-to-real.md): Compare predicted and measured motion and timing.
- [Lab 3: Learned Control](labs/lab-03-learned-control.md): Compare a learned policy with the nonlearned baseline.
- [Lab 4: Real Data and Fine-Tuning](labs/lab-04-real-data-and-fine-tuning.md): Collect demonstrations, adapt the policy, and evaluate the loop.
- [Lab 5: Stress-Testing](labs/lab-05-stress-testing.md): Test action horizons and disturbance recovery.
- [Lab 6: Safety Governor](labs/lab-06-safety-governor.md): Build the governor and test its declared fault responses.
- [Capstone Studio](labs/lab-capstone-studio.md): Defend a release decision with physical evidence.

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
and one downloadable discussion. The [full narrative and chapter mapping](../plans/vol4/index.qmd)
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
