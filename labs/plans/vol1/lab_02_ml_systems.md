# 📐 Mission Plan: 02_ml_systems (The Iron Law)

## 1. Chapter Context
*   **Topic:** The Iron Law of ML Systems & Physical Constraints.
*   **Core Invariant:** $T = D/BW + O/R + L$.
*   **The Struggle:** Identifying the "Ghost Bottleneck." Students often assume more FLOPS ($R$) always equals more speed, but the **Memory Wall** ($BW$) or **Light Barrier** ($L$) often dominates.

---

## 2. The 4-Zone Dashboard Anatomy

### Zone 1: Command Header
*   **Title:** Lab 02: The Physics of Performance
*   **Persona Identity:** Current Role (e.g., Cloud Titan) and Global Scale.
*   **Constraint Badges:**
    *   `Latency < 100ms` (Red/Green)
    *   `Power < Budget` (Red/Green)
    *   `MFU > 50%` (Red/Green)

### Zone 2: Engineering Levers (Inputs)
*   **Hardware Selector:** H100 (Cloud), Orin (Edge), A17 (Mobile), M7 (Tiny).
*   **Batch Size Dial:** 1 to 512.
*   **Arithmetic Intensity Slider:** 1 to 200 FLOPs/Byte.
*   **Distance Slider:** 0 to 5000 km (Network offloading).

### Zone 3: Telemetry Center (Visuals)
*   **The System Ledger:** 4 Cards (Speed, Joules, TCO, Goodput).
*   **The Plot:** **Live Roofline Duel**. A dynamic chart showing the hardware's Ridge Point. As the student moves the Intensity slider, their "State Dot" moves along the roofline.

### Zone 4: Audit Trail & Justification
*   **Consequence Log:** "Warning: At Batch 1, your MFU is <5%. The system is Memory-Bound."
*   **Rationale Box:** Justify your hardware choice based on the **Bottleneck Principle**.

---

## 3. The 3-Act Narrative (The Lab Journey)

### Act I: Proving the Iron Law (15m)
*   **Scenario:** You have a fixed model ($AI = 50$). 
*   **Crisis:** It runs too slow on the CPU.
*   **Task:** Upgrade to a GPU. Observe that the latency *barely moves* because the data term ($D/BW$) is the bottleneck. Find the "Ridge Point" where the GPU actually becomes useful.

### Act II: The Light Barrier Race (15m)
*   **Scenario:** Move the workload to a Cloud Titan datacenter in another region.
*   **Crisis:** The user experience is "laggy" despite massive GPU power.
*   **Task:** Calculate the round-trip time ($L_{lat}$). Determine if edge deployment is mandatory to meet the <100ms SLA.

### Act III: The Thermal Wall (15m)
*   **Scenario:** Deploy to a Mobile Nomad device.
*   **Crisis:** After 60 seconds of inference, the FPS drops by 70%.
*   **Task:** The SoC is thermal throttling. Lower the **Precision** (FP32 -> INT8) to reduce Joules-per-inference and keep the clock speed stable.

---

## 4. Real-World Data Sources
*   **Hardware:** Official specs for NVIDIA H100, Jetson Orin NX, and Apple A17 Pro.
*   **Physics:** Speed of light in fiber (~200,000 km/s).
*   **Thermal:** Canonical 5W TDP limit for smartphone sustained loads.
