# Three physical AI archetypes as teaching proxies

**Status:** Alternative project architecture for discussion, not a released kit or a tested hardware result. The [Volume IV introduction](../../../books/vol4/frontmatter/about.qmd) and [boundary chapter](../../../books/vol4/01_boundary/01_boundary.qmd) name three foundational machine classes by their binding physical constraints: mass and stopping clearance, contact force, and continuous energy or flow. The [student competency matrix](student-competencies.md) describes transferable engineering abilities. This page asks how small UNO Q stations could exercise each class without pretending that a tabletop proxy is a warehouse robot, industrial arm, or thermal plant.

## What the current draft has and what it misses

The current [lab sequence](../labs/lab-sequence.md) gives every team a guarded one-axis visual follower, then offers active inspection and retained-object routing. It establishes a common model → MCU permission → physical action → observation path. The syllabus has moved from book front matter to the [instructor materials](../../../instructors/vol4/README.md). The [kit sheet](../staff/kit-bom.md) and [feasibility plan](../staff/feasibility-plan.md) specify a candidate station and required tests. **No station, model runtime, sensor mount, simulator comparison, or SO-101 STM32-to-servo command path has yet been demonstrated by these documents.**

Those three project prompts mostly vary the task on the same low-force axis. Active inspection tests information gathering well, but turning a retained object without force measurement does not teach contact-dominated manipulation. A slow pointer does not reveal a useful stopping-distance budget merely because it moves. None of the three represents a continuous process. To teach the book's archetypes, each proxy must preserve the **dominant physical constraint and its failure mode**, rather than imitate the full machine's appearance.

## Three candidate body modules on one computing spine

Every module uses the same abstract cycle: observe physical state, run a versioned learned model on the UNO Q's Qualcomm/Linux side, propose a bounded action, let the STM32 check local state and permission, apply energy through a separate actuator supply, and measure the result. Training and simulation can run on a workstation. The shared trace links observation, model revision, proposal, permission, applied action, direct sensor reading, and next observation. The body-specific wiring, driver, protection, and simulation model differ.

| Book archetype | Bench proxy and learned decision | Physical limit to measure | Qualification test |
|:---|:---|:---|:---|
| **Class 1: mobility** | Guarded cart; vision proposes speed or stop. | Encoder and range sensor; stopping clearance. | Speed or inference delay consumes measurable clearance. |
| **Class 2: manipulation** | Compliant pusher; probe model proposes depth or withdrawal. | Position and contact force; force/travel limit. | Different sample compliance changes measured force. |
| **Class 3: process/energy** | Enclosed heater/fan cell; predictor proposes duty. | Two temperatures and power; thermal cutoff. | Stored heat causes measurable lag after duty changes. |

For **Class 1**, use a short marked rail or lane with physical end stops, a light cart, motor driver, and accessible cutoff. Compare coast and braking distance at several speeds and injected inference delays. The MCU uses local speed and obstacle distance to refuse a proposal that cannot stop within the remaining lane. If gearbox friction makes residual motion unmeasurable, change the carriage design or reject the proxy.

For **Class 2**, push replaceable samples of two safe compliance levels within a guarded contact zone. A model uses a first camera view or force/displacement probe to choose a second push or withdrawal. Calibrate a force sensor or a spring-deflection measurement and determine its delay before setting motion speed. The MCU refuses excessive travel or force and stops on sensor loss. A passive object turn without measured contact does not test this archetype.

For **Class 3**, enclose a low-voltage heater, fan, and modest thermal mass. Log temperatures at two locations and applied power. Compare a simple thermal model, learned predictor, and rule or PID controller while the MCU enforces a temperature ceiling and an independent thermal cutoff remains available. Reject the proxy if the thermal effect is too small or a trial takes too long for a class period.

These are **small proxies of the governing laws**, not claims that a rail cart covers legged locomotion, a slow compliant pusher reproduces high-rate industrial contact, or a warm box covers high-pressure fluid and power electronics. A [Pololu encoder gearmotor](https://www.pololu.com/product/5142), [SparkFun load-cell interface](https://learn.sparkfun.com/tutorials/load-cell-amplifier-hx711-breakout-hookup-guide/introduction), and [Adafruit low-voltage heater](https://www.adafruit.com/product/4308) show possible component families; no part is selected or safe to deploy merely because it has a product page. Staff must size, guard, power, instrument, and qualify each body. The load-cell interface linked here is a serial sensor, not SPI.

## How the three proxies use the competency matrix

The competency matrix says **what an engineer must be able to do**. The archetypes supply three different physical contexts in which to demonstrate it. A team need not build twelve projects; it should complete the four quadrants on one body and examine the other two through short instrumented labs or shared stations.

| Competency quadrant | Class 1: cart | Class 2: pusher | Class 3: thermal cell |
|:---|:---|:---|:---|
| **A. Measure the world** | Calibrate speed and clearance; measure coast and braking. | Calibrate position and force; measure force versus displacement. | Calibrate two temperatures and power; measure lag and stored heat. |
| **B. Characterize computation** | Evaluate visual obstacle or target inference and its latency. | Evaluate material or contact-state inference. | Evaluate next-temperature prediction and horizon error. |
| **C. Change the world** | Approach, brake, and replan after a missed stop. | Probe, push, withdraw, and replan after unexpected contact. | Heat, cool, hold, and replan after a disturbance. |
| **D. Integrate and govern** | Enforce a local stopping envelope. | Enforce a local force/travel envelope. | Enforce a local temperature/power envelope. |

Across all three, students version the model and data, compare a nonlearned baseline, run matched simulator and hardware trials, trace proposal versus permitted action, and explain a seeded fault. The physical limit and the sensor/actuator module change; the evidence contract stays the same.

The [UNO Q](https://docs.arduino.cc/hardware/uno-q/) provides a Qualcomm Linux processor, STM32 microcontroller, and Bridge communication for the shared compute/authority lesson. Its presence does not prove that any particular model, motor driver, force sensor, or thermal cell works together. A Hugging Face vision checkpoint is a candidate for the cart; a small learned probe or dynamics model could be versioned through the Hub for the pusher; [PatchTST](https://huggingface.co/docs/transformers/model_doc/patchtst) illustrates a time-series model family for the thermal cell. Model size and on-board runtime remain pilot questions. A simple, frozen model that demonstrably changes a physical decision is preferable to a larger model that misses the task deadline.

## Course structure and fallback

The one-axis follower can remain a **common bring-up exercise** for camera, Bridge, MCU permission, logging, and feedback. It need not be one of the three archetype capstones. Staff then build one qualified station of each archetype and provide recorded traces and simulator adapters so every student can compare all three constraints. Teams choose one archetype for a vertical capstone; the class can share the more expensive bodies instead of buying all three for every team. The [SO-101](https://huggingface.co/docs/lerobot/so101) remains a shared manipulation and imitation-policy extension until its motor-command authority is separately qualified.

If active visual inspection cannot meet its ambiguity or timing requirement, it can be dropped without losing the course sequence. The thermal cell is a plausible low-mechanical-complexity replacement because it can expose prediction, delayed physical response, and MCU authority on a slow time scale; thermal safety and session duration must still be proved. The pusher is the clearest contact proxy if force measurement is reliable. The cart is the clearest mobility proxy if its stopping distance is measurable within a guarded lane. No fallback should be declared ready until staff show the limiting physical effect, the on-board model path, MCU permission before actuation, and a repeated feedback trial.

**Decision before syllabus revision:** prototype one of each body, then choose whether all teams rotate through three short archetype labs and select one capstone, or whether only one archetype is practical for the first offering. The present syllabus still describes the earlier follower/inspection/routing options and should change only after that pilot decision.
