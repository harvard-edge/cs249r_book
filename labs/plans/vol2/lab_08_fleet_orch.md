# 📐 Mission Plan: 08_fleet_orch (The Fragmentation War)

## 1. Chapter Context
*   **Topic:** Cluster Scheduling, Resource Fragmentation, and Fault Tolerance.
*   **Core Invariant:** The Utilization Paradox (100% busy GPUs != 100% productive work).
*   **The Struggle:** Balancing "Fairness" (many small jobs) vs. "Throughput" (one giant gang-scheduled job).

---

## 2. The 4-Zone Dashboard Anatomy

### Zone 1: Command Header
*   **Title:** Lab 08: The Fleet Orchestrator
*   **Persona Identity:** Cloud Titan (Exaflop Scale)
*   **Constraint Badges:**
    *   `Fragmentation < 20%` (Red/Green)
    *   `Goodput > 80%` (Red/Green)
    *   `SLA: Met/Missed`

### Zone 2: Engineering Levers (Inputs)
*   **Scheduling Policy:** First-Come-First-Serve (FCFS) vs. Best-Fit vs. Preemption.
*   **Checkpoint Interval Slider:** 1 hour to 24 hours.
*   **Preemption Switch:** On/Off (Kill small jobs for the big one).
*   **Defragmentation Toggle:** "Stop-the-World" migration.

### Zone 3: Telemetry Center (Visuals)
*   **The System Ledger:** 4 Cards (Cluster MFU, Goodput %, Wait Time, TCO).
*   **The Plot:** **The Swiss Cheese Heatmap**. A grid of 10,000 GPUs. Green = Working, White = Idle, Red = Failed. Students see "holes" appear as fragmentation increases.

### Zone 4: Audit Trail & Justification
*   **Consequence Log:** "Incident: A node failure in Rack 4 wiped out 6 hours of work. Recovery initiated."
*   **Rationale Box:** Explain why you chose your checkpoint interval based on the **MTTF (Mean Time To Failure)**.

---

## 3. The 3-Act Narrative (The Lab Journey)

### Act I: The Fragmentation Audit (15m)
*   **Scenario:** Your cluster is 80% full with thousands of small "Student" jobs.
*   **Crisis:** A high-priority 1,000-GPU job arrives. It needs contiguous nodes for AllReduce.
*   **Task:** Observe that the job won't start because of the "Swiss Cheese" holes. Decide between killing the small jobs or migrating them (at a 10% performance tax).

### Act II: The Checkpoint Gamble (15m)
*   **Scenario:** You are training a 175B model on 16,384 GPUs.
*   **Crisis:** The cluster has a hardware failure every 3 hours (MTTF).
*   **Task:** Find the "Golden Interval." If you checkpoint too often, you waste compute on IO. If you don't, you lose days of work. Find the mathematical optimum.

### Act III: The Elastic Rescale (15m)
*   **Scenario:** 10 nodes suddenly fail due to a power spike.
*   **Crisis:** The training ring is broken.
*   **Task:** Implement **Elastic Rescaling**. Watch the system dynamically re-route around the dead nodes and continue training at 90% capacity rather than crashing.

---

## 4. Real-World Data Sources
*   **Reliability:** Meta's Llama-3 training logs (failure rates on 16k H100s).
*   **Orchestration:** Kubernetes/Slurm bin-packing logic.
*   **Economics:** Cost of idle H100 capacity (~$2-4/hour).
