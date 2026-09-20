# Volume IV: fourteen lab blueprints for one physical AI project

**Status:** Teaching design, not released student handouts. These labs use the [course operating plan](course-architecture.md), [platform-independent competency checklist](student-competencies.md), and [SO-101/UNO Q station draft](so101-uno-q-course.md). The [feasibility plan](feasibility-plan.md) asks Andrea to try each proposed student exercise and report what works. Staff must test the hardware and time budgets before turning the blueprints into assignments.

For the first offering, these are fourteen weekly **studio prompts**, not fourteen graded assignments. Teams submit graded packets only at the week-2 charter, week-6 feedback loop, week-10 authority demonstration, and week-14 capstone. Weeks 8–9 offer a choice of deeper investigation, week 12 can be project repair, and week 13 is an ungraded rehearsal. Other evidence goes into one shared project notebook and receives formative bench feedback; the [first-offering teaching plan](../../instructors/vol4/first-offering-plan.md) assigns staff roles.

The [syllabus](../../instructors/vol4/README.md) pairs each Volume IV reading with a direct lab link. Individual draft briefs: [1](lab-01-boundary.md), [2](lab-02-body-and-sensors.md), [3](lab-03-action-permission.md), [4](lab-04-physical-episodes.md), [5](lab-05-baseline-and-learning.md), [6](lab-06-local-feedback-loop.md), [7](lab-07-simulation-gap.md), [8](lab-08-action-chunks.md), [9](lab-09-language-and-placement.md), [10](lab-10-authority-under-fault.md), [11](lab-11-incomplete-motion.md), [12](lab-12-human-correction.md), [13](lab-13-release-case.md), and [14](lab-14-capstone-trial.md). This page remains the staff design overview; keep the brief and blueprint for a week consistent when revising either.

## What students build

Each team carries one versioned system from week 1 to week 14. The shared first task is a **guarded visual reach with a changed scene**: observe a soft target in a marked workspace, propose a bounded arm movement using a local learned model on the UNO Q, let the STM32 permit or refuse it, measure actual movement, then safely move the target or interrupt the move. The next observation must reveal the actual arm and target state, and the learned component must influence a revised action or an abstention. The week-6 demonstration requires this loop, not a reliable grasp. Replaying the original action after merely capturing a second frame fails.

The semester project extends that loop to a **disturbance-and-recovery** manipulation outcome, such as placing a soft block in a marked tray after a moved target or incomplete pickup. Teams may instead investigate active inspection or partial placement if the station can measure those outcomes. A camera label followed by a fixed arm script does not meet the project requirement: teams must show a learned proposal whose next action depends on the measured state produced by prior motion or disturbance. Instruction-conditioned sorting is a capstone option only after staff demonstrate a task-matched local language-conditioned policy within the board's time and memory budget. An offboard SmolVLA result may be compared with the local policy but does not replace local learned inference in the required loop.

The common record is **observation ID and time; requested, mapped, enforced, and measured action; next observation; measured tool and object/task state; outcome; intervention**. A LeRobot episode, linked MCU event log, and independent camera or fixture outcome record may together provide this record. Servo readback alone cannot establish grasp or placement. Every team compares a nonlearned baseline and reports repeated physical trials with a scene change or incomplete action. The final defense follows a single episode from sensed state to revised decision and actual material outcome.

## Acceptance rule for every lab

Each weekly artifact contributes one link to the same causal episode: a physical state and prediction, a proposed action, a permission decision, actual motion or refusal, the resulting observation, and a task consequence. A lab may isolate one link, such as calibration or permission, but its hand-in must show how that link changes the continuing episode. Students must identify what would have gone wrong if the system had trusted the command instead of the measured world. Board inference speed, model score, and polished video are supporting measurements, never substitutes for this causal record. The final test must satisfy all three conditions in the book's scope test: a learned decision, consequential physical feedback, and delegated action inside the MCU permission boundary.

## Staff station contract before week 1

The teaching team provides an assembled SO-101 follower, calibrated camera and fixed lighting, guarded low-energy workspace, accessible motor-power cutoff, UNO Q image, and a tested Linux → STM32 → servo-bus command path with no parallel host command path. Staff also provide a LeRobot robot adapter or converter that retains the four action taps, a starter episode set, a compact Hub-sourced policy artifact that runs locally on the exact Q, a scripted baseline, two pinned simulation fixtures, fault scripts, and spare or replay stations. A leader arm is useful for teleoperation but is not a required student purchase if another tested input device works.

Before publishing an arm assignment, a second staff member must reproduce four demonstrations: native arm calibration and episode replay; one-joint MCU permission and refusal; whole-arm guarded command and measured readback; and a local learned reach whose action proposal changes after a measured scene change. The qualifying local policy consumes a current observation and robot/task state and proposes a bounded action for held-out target positions; a classifier label wired to one fixed motion does not qualify. Staff measure the physical envelope, loop timing, model memory, station capacity, and task success protocol. The [feasibility plan](feasibility-plan.md) defines the report. If the arm command path fails, the [one-axis reference station](lab-sequence.md) carries the governed loop while SO-101 remains a LeRobot data and policy station. If no compact action policy runs on the Q, the course cannot claim the proposed local-inference objective and the teaching team must revise the platform or objective before release.

## Common labs: establish the loop

### Lab 1 — Where is the physical AI boundary?

**Experiment:** With motor power isolated, inventory the camera, Qualcomm Linux, Bridge, STM32, servo bus, and cutoff. Trace one hypothetical proposal and identify every component that can cause motion. Observe boot, disconnect, and power-off behavior on the qualified station.

**Evidence and exit:** Submit a power and command diagram, safe-state observations, and one counterexample in which a model output would not establish physical AI. Pass when the team can identify the only live actuator command route and the measured safe state. This starts **C1, C6, C13**.

### Lab 2 — What can the body and sensors actually measure?

**Experiment:** Calibrate joint readback and camera coordinates, command the same slow move from repeated starts, and measure the difference between requested pose, settled pose, and visible tool position. Measure the reachable marked workspace, then conceal or disconnect one feedback source. State units, settling time, and uncertainty; use a staff fixture so no student must fabricate an arm mount.

**Evidence and exit:** Submit calibration and frame revisions, raw command-versus-settled readings, a measured operating envelope, and a sensor-loss trace. Pass when a teammate can reproduce a measured target position and explain why commanded motion is not measured motion. Freeze the week-2 task charter. This exercises **C1, C2**.

### Lab 3 — Who permits an action?

**Experiment:** Send synthetic one-joint proposals through the supplied STM32-to-servo interface: valid, duplicate, expired, disarmed, and out of range. Progress to a slow bounded pose only after the one-joint trace is correct. Do not ask students to design an untested servo-bus interface during term.

**Evidence and exit:** Submit the proposal schema and linked requested, mapped, enforced, and measured action traces. Pass when the MCU refuses every injected invalid request and no alternate host path can move the arm. This exercises **C3, C6, C13**.

### Lab 4 — Can an episode reconstruct what happened?

**Experiment:** Teleoperate guarded reaches and a few soft-block moves. Record images, joint readback, instruction or goal, four action taps, and observed outcomes. Replay one episode; deliberately introduce a missing frame or timing gap.

**Evidence and exit:** Submit a replayable LeRobot episode, linked MCU log, timing alignment rule, and dataset card. Pass when another team can reconstruct the action that followed a selected observation and identify the corrupted episode. This exercises **C3, C7**.

### Lab 5 — What does learning improve over a rule?

**Experiment:** Hold out complete capture sessions; run the scripted reach baseline on frozen start positions. Evaluate the supplied compact policy on the same observations, inspect its preprocessing and failure cases, and adapt it using team data if time permits. Training may use a workstation.

**Evidence and exit:** Submit split rationale, model and dataset revisions, baseline physical outcomes, held-out model errors, and a failure inventory. Pass when the comparison uses matched starts and a result cannot be explained by data leakage. This exercises **C4, C7, C8**.

### Lab 6 — Can the learned system close one physical loop on the Q?

**Experiment:** Run the compact policy locally on Qualcomm Linux. Compare its outputs with the workstation, send a bounded proposal through the MCU, and measure joint, tool, and target state. Staff then shift the soft target within the qualified workspace or cause a safe incomplete move. Capture a fresh observation and require the learned component to produce a state-dependent correction or abstention. Run the scripted baseline against the same disturbance. Use the slow reach task and a staff checkpoint if a team's model is not ready.

**Evidence and exit:** Demonstrate a witnessed model → permission → measured move → changed scene → new observation → revised decision. Submit the model revision, memory and latency trace, four action taps, target/tool state before and after, rule comparison, and a refusal or abstention. The second proposal must respond to measured state rather than execute a queued script. This **week-6 demonstration** exercises **C8, C10, C12, C15**.

## Project studios: explain and improve the system

### Lab 7 — What do the simulators miss?

**Experiment:** Apply identical initial states and actions to the staff-supplied kinematic model, a second pinned simulator, and the arm. Identify one dynamical parameter or fit a small next-state predictor using supplied training trajectories; evaluate the resulting prediction against held-out physical reaches. Students configure and compare the fixtures rather than build two engines.

**Evidence and exit:** Submit observation/action contracts, matched rollouts, one-step and multi-step errors, and one physical failure explained by model mismatch. Pass when the team identifies which prediction it will trust for its project. This exercises **C5, C9**.

### Lab 8 — When should a policy observe again?

**Experiment:** Start from a staff-provided ACT checkpoint and compare its sequence-action rollout with a stepwise policy on matched simulated starts and guarded physical trials. Perturb the target after the first action in one trial. Train or fine-tune ACT on workstation compute if that comparison serves the team's project question and fits the studio budget.

**Evidence and exit:** Submit checkpoint or policy revisions, action-chunk and stepwise traces, physical outcomes, and the point at which a fresh observation should invalidate a chunk. Pass when the team can explain one drift or recovery case. This exercises **C10, C11**.

### Lab 9 — Does language or model size earn its cost?

**Experiment:** Test a task-matched SmolVLA checkpoint or staff-provided rollout with two instructions applied to the same scene. Compare it with ACT and the compact Q policy. Profile local SmolVLA on the Q only if staff have shown a viable runtime; otherwise analyze the measured staff trace and keep the compact policy in the physical loop.

**Evidence and exit:** Submit paired-instruction action differences, matched **physical** task outcomes, memory and latency measurements, and a model placement decision. Pass when the team distinguishes a language-conditioned change in the physical plan from a visual coincidence and states whether the Q can meet its deadline. If the model cannot control the qualified station, label this an offboard comparison rather than a physical AI demonstration. This exercises **C11, C12**.

### Lab 10 — Can the authority boundary reject a plausible policy?

**Experiment:** Replay a valid-looking stale chunk, a duplicate, an oversized move, and a proposal sent while disarmed. Add Linux load or pause the proposer. Test a physical cutoff on the staff-qualified station.

**Evidence and exit:** Submit refusals with reasons, four action taps, measured no-motion or bounded-motion results, and a timing rule. Show that old intent cannot produce a new move after cutoff or timeout. This **authority demonstration** exercises **C13, C14**.

### Lab 11 — What happens after incomplete motion?

**Experiment:** Move the target or soft block, obstruct a reach with a compliant fixture, or induce a partial grasp within the approved envelope. Compare open-loop continuation with observing again and replanning or abstaining.

**Evidence and exit:** Submit video or object-state evidence, measured joint response, next observation, changed plan, and repeated outcomes. Pass when the team's policy detects at least one discrepancy between commanded and realized state rather than counting a command as success. This exercises **C10, C14**.

### Lab 12 — Does a correction improve physical outcomes?

**Experiment:** Record a human intervention or corrective demonstration, revise the dataset or preprocessing, retrain or fine-tune a compact policy, and repeat the same frozen physical trial set. Keep the previous policy as a control. A controller-rule change can be studied separately but does not replace the learned-policy revision.

**Evidence and exit:** Submit intervention log, changed artifact revision, and paired before/after physical results including failures. Pass when the team can attribute the observed change to a documented revision or report that the evidence is inconclusive. This exercises **C7, C15**.

### Lab 13 — What claim can the evidence support?

**Experiment:** Freeze code, model, data, simulator versions, calibration, fault script, and operating envelope. Exchange one safe fault case with another team and rehearse the final trial protocol without changing the scoring rule afterward.

**Evidence and exit:** Submit a release or no-release draft with raw traces, baseline comparison, success and abstention counts, unresolved limits, and an individual competency evidence index. Pass when another team can reproduce the trial and challenge the claim. This exercises **C6, C15**.

### Lab 14 — Does the system work on an unfamiliar physical trial?

**Experiment:** Staff select a held-out starting arrangement or instruction and, after the first permitted move, introduce a safe scene change or incomplete motion. The team runs repeated physical trials and defends the full observation → proposal → permission → measured action → changed world → new observation → revised proposal chain. Each student explains a selected trace and a decision to continue, correct, stop, or abstain.

**Evidence and exit:** Submit the frozen system packet, raw trial results, matched baseline, disturbance and fault responses, measured object/task outcomes, and bounded verdict. The defense checks evidence in all four competency quadrants; an impressive demonstration without a state-dependent revised decision, measured physical outcome, or MCU permission cannot pass. This is the capstone.

## What the handout author still has to specify

Each released lab needs a starting station state, exact student commands and diagrams, a bounded motion envelope, a time allowance, a reset procedure, a sample trace, a deliberate fault script, a pass/fail rubric, and a path for equipment outages. Staff fill in numerical limits only from the qualified station. The first six handouts can be written after the staff pilot; later project studios use the same station and evidence schema with team-specific task variants.
