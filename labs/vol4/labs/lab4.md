# Lab 4 - Deploy Vision-Language-Action Policies in Simulation

Status: Skeleton \
When: Week 4 \
Due: end of Week 5 \
Readings: Part I: The Machine Anatomy - The Cognitive Brain; Part III: Running the Machine - Grounded Intent

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Reproduce real-world scenarios in simulation frameworks.
- Deploy learned VLA policies in simulation.
- Explain how a VLA combines vision, language, and proprioception into actions.
- Benchmark learned VLA policies in simulation.

---

## 2. Implementation Steps

1. Reproduce our workbench inside the simulation framework (We Do - 60 mins).
2. Deploy a learned VLA policy in the constructed simulation environment and measure its partial completion rate and collision count (We Do - 30 mins).
3. Inspect observation tensor shapes, locate the proprioception projection layer in the deployed VLA policy, and trace the concatenation of image, language, and state tokens (You Do - 15 mins).
4. Benchmark a learned VLA policy on your laptop (You Do - 75 mins).

---

## 3. Deliverables

1. A short screen recording (max 30 seconds) or 2–3 screenshots demonstrating the successfully reproduced workbench and the robot executing the VLA policy in simulation, along with the partial completion rate and collision count of the deployed policy in simulation.
2. A digital block diagram of the deployed VLA. You must explicitly highlight the observation tensor shapes, the proprioception projection layer, and the concatenation point of the image, language, and state tokens. Include a brief code snippet or terminal output proving you inspected the actual tensor shapes.
3. The partial completion rate, collision count, and average latency of the deployed VLA policy run on your laptop. Include a brief, 2–3 sentence analysis of your results (e.g., identify common failure modes, note your hardware specs, or hypothesize latency bottlenecks).

---