---
title: "Team Newsletter: Beyond the Spiral,  TinyML Pedagogy at the Hardware Edge"
date: "2026-05-22"
author: "MLSysBook Team"
description: "## Editor's Note  This month's team newsletter comes from Jeremy Ellis, High School Robotics and Machine Learning Educator in British Columbia, Canada, and Co-Chair of the AIEng4D Show &..."
categories: ["community"]
image: "https://assets.buttondown.email/images/24481222-3f5a-4f49-96a6-d7376b1ac6e6.png?w=960&fit=max"
---



## Editor's Note

This month's team newsletter comes from Jeremy Ellis, High School Robotics and Machine Learning Educator in British Columbia, Canada, and Co-Chair of the AIEng4D Show & Tell.

Jeremy brings a rare perspective to our community: more than a decade of teaching robotics, TinyML, and Edge AI in a secondary school classroom. In this reflection, he shares what AI engineering pedagogy looks like when students work directly with hardware, failure is part of the learning process, and the classroom functions like a small R&D lab rather than a traditional lecture course.

His piece is especially relevant as we think about how [MLSysBook.ai](https://mlsysbook.ai/), TinyTorch, kits, and Show & Tell support educators across very different teaching environments.

A previous newsletter introduced the Spiral Model as a framework for helping university faculty move from tool familiarity to process-driven AI integration. It is a powerful lens for institutional curriculum change.

---

*At the hardware edge, pedagogy stops being about curriculum design and becomes the disciplined management of failure in real time.*

What I can offer is something different: over a decade of running TinyML and Edge AI in a secondary school robotics classroom in British Columbia, Canada, where the constraints are not just computational but social, institutional, and deeply human.

## The Deployment Environment Is Different

University AI pedagogy often assumes stable multi-year curriculum cycles, predictable hardware procurement, and software stacks that survive semester boundaries. Secondary school robotics offers none of these guarantees. A single SDK deprecation or board revision can silently break a carefully designed assignment before the first student powers a device.

To function in this environment, I moved away from lecture-centric instruction toward a Zero-Lecture model, treating the classroom as a functioning R&D lab. The instructor's role shifts from lecturer to Lab Director: establishing protocols, enforcing verification gates, and shepherding failure into learning.

Educators capable of teaching embedded systems are in high demand and could earn substantially more in industry. This model is therefore designed so a capable generalist, not a specialist, can run the course. That constraint matters anywhere specialists are scarce, including many EdgeAI settings globally.

## The Analog Firewall

There is an upstream constraint, even before any TinyML pedagogy. Students' attention has been reshaped by years of passive, algorithmically-optimized screen consumption, and university educators may be starting to see the same thing in their own pipelines. Attention behaves like a limited system resource, similar to memory or power on an embedded device. Managing it is a system design problem, not a soft skill problem.

My response is what I call an Analog Firewall: rigorous, computer-free foundational work that runs in parallel with hands-on physical problem-solving. A robotics course built around machine learning is uniquely well-suited for this. The physical world provides grounding and constraint; the ML layer provides genuine intellectual challenge. Students must think before they type, explain before they deploy, and draw before they wire.

## Individual Ownership and the "Lead Three" Model

Most robotics programs default to group work. In practice, this produces task-splitting: one student owns the firmware, one wires, one documents. In an era where LLMs can scaffold any one of those tasks on demand, that division is pedagogically counterproductive.

In this model, each student works with their own board, their own sensors, and their own codebase. Students may work in pods on the same assignment, but there are no shared repositories to hide behind. Completing 40–60 individual assignments means every student who finishes the course has personally navigated the full embedded ML pipeline: data collection, model training, on-device inference, IoT communication, and sensor/actuator hardware integration.

Grading is explicitly non-competitive. I do not use a bell curve. The structure is:

- **Baseline:** Completion of 40–60 assignments covering the full pipeline
- **Pass:** A first final project integrating a unique sensor–actuator pair
- **Advanced Tier:** A second project combining multiple sensors and/or actuators, ML, and/or IoT connectivity

Progress is tracked on a master completion chart, a grid mapping every student against every assignment. Each cell is marked Checked (complete) or Lead (the student who first solved it and then walked 3 peers through it). A student must complete a sufficient number of assignments before being permitted to start final projects. (In early 2026, with a new board, the floor was 45.) The threshold is a structural gate; it prevents skill gaps from reaching the project phase, where they become expensive rather than instructive.

To eliminate knowledge hoarding, I use a "Lead Three" norm: the first student to solve a difficult problem is expected to walk at least three peers through it. Those three then assist others. Students become the primary vector of knowledge transfer, and a student's top grade (along with the class's collective top grade) is partly determined by how many peers they have led through the completion chart. The cooperative incentive is built into the ceiling, not just the floor.

## Hardware Verification as Pedagogical Protocol

Every assignment follows a fixed sequence designed to isolate variables and make failure informative rather than catastrophic:

1. **Pseudocode first:** short handwritten description of the algorithm before any tools are opened (new 2026)
2. **Software verification:** code must compile and upload before hardware is touched
3. **Hand-drawn circuit diagram:** students manually map 3V3, GND, and all GPIOs, sensors, and actuators
4. **Human gate:** diagram checked by the instructor or a qualified Lead student before wiring begins
5. **Wiring and pre-power inspection:** visual check of the physical board, before power is connected
6. **Video proof of completion:** asynchronous assessment proof after charting, via short student-submitted videos

This protocol exists for the same reason engineers write tests before deployment: to make errors survivable. Safety constraints follow the same logic. Parental awareness and permission are expected when high school students want to work above 9 volts, with water, or with drones. Without these, a single non-specialist teacher cannot run a functioning embedded systems lab with teenagers.

## LLMs as 24/7 TAs; GitHub as the Living Specification

LLMs are not treated as academic dishonesty in this course. They are the always-available teaching assistant no school budget can hire. Students use them the way a working embedded ML engineer does: as debuggers, explainers, and research tools.

The constraint is real. An LLM cannot reliably solve hardware-specific issues on a board released within the last year. This is where a well-maintained open-source GitHub `README.md` becomes load-bearing infrastructure: a living document of code skeletons and hardware-specific gotchas, updated as new failure modes are discovered. Students use the README as ground truth and the LLM as their research and reflection layer.

The asynchronous, self-paced structure also solves a hardware budget problem familiar to anyone equipping a lab on a constrained budget. Because students hit different milestones at different times, a full class set of any expensive sensor is rarely needed. Staggered progress is a feature, not a bug.

## What Success Looks Like at the Edge

By the time students reach their final projects, they have an embodied understanding of why on-device inference matters, where compute constraints actually bite, why data collection is usually harder than model architecture, and why running an actuator live with an ML model is harder than it looks. They know this last point in a particular way: actuators fail more dramatically than classifiers. A misclassification produces a wrong label; a miscalibrated servo at speed produces a broken robot and a lesson in mechanical consequence. That asymmetry is difficult to teach in simulation. It has to be experienced.

They have also rediscovered that mathematics, physics, chemistry, and biology are not abstract school subjects. They are the tools you reach for when the model is wrong and you need to know why. This cross-curricular integration is not designed; it emerges from failure.

In a hardware-first classroom, learning is a loop: build → break → debug → repeat, on hardware that does not care about your lesson plan.

![Beyond the Spiral.png](https://assets.buttondown.email/images/24481222-3f5a-4f49-96a6-d7376b1ac6e6.png?w=960&fit=max)

## A Note for University Educators

This newsletter sits alongside resources like [MLSysBook.ai](https://mlsysbook.ai/), a rigorous, open-source text with university-ready hardware labs that represents the kind of depth university instruction rightfully demands. What I describe here is something different in scale, intent, and context.

For undergraduate programs, this model might function as a kind of clinical residency in resilience, moving students from consumers of curriculum toward producers of technical solutions. The verification gates, completion chart, and Lead Three norm are not stylistic choices; they are load-bearing mechanisms. Removing any one of them changes the system's behaviour in ways that are difficult to predict without running the lab.

What I think generalizes beyond a high school classroom is a short list: verification gates before any commit of physical resources, individual ownership before collaboration, and a living spec maintained outside the LLM. The Analog Firewall may transfer too; that one I am less sure about. The rest is local.

I offer the piece as a data point from the edge, not a model to transplant: one account of what TinyML pedagogy looks like when the constraints are maximal, the instructor is a generalist, and the students are teenagers.

---

**Jeremy Ellis**
High School Robotics and Machine Learning Educator, MSS, Mission SD75, BC, Canada
Co-Chair, AIEng4D Show & Tell
LinkedIn: [Jeremy Ellis](https://www.linkedin.com/in/jeremy-ellis-4237a9bb)
Course GitHub: [The XIAO ML Kit Course](https://github.com/hpssjellis/maker100-xiaoML-kit)
![Screenshot 2026-05-20 at 11.25.10 AM.png](https://assets.buttondown.email/images/b5adfdcb-3487-4830-8619-5c375fabe1b5.png?w=960&fit=max)
