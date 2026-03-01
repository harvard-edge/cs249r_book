# 📐 Mission Plan: 04_data_engr (Data Gravity & Drift)

## 1. Chapter Context
*   **Topic:** Dataset Compilation, Data Gravity, and Signal-to-Noise Engineering.
*   **Core Invariant:** The Energy-Movement Invariant (Moving data costs >100x more than compute).
*   **The Struggle:** Managing the "Feeding Tax." Students must keep the GPU ALUs busy despite low-bandwidth storage pipelines.

---

## 2. The 4-Zone Dashboard Anatomy

### Zone 1: Command Header
*   **Title:** Lab 04: The Data Factory
*   **Persona Identity:** Current Role (e.g., Tiny Pioneer) and Scale.
*   **Constraint Badges:**
    *   `Egress < $100k` (Red/Green)
    *   `GPU Hunger < 10%` (Red/Green) - Idle time
    *   `Drift Detected` (Alert Badge)

### Zone 2: Engineering Levers (Inputs)
*   **Storage Tier:** NVMe (Hot), S3 (Warm), Glacier (Cold).
*   **Transfer Method:** 10Gbps Fiber vs. AWS Snowball (Physical Truck).
*   **Deduplication Scrubber:** 0% to 50% removal of redundant data.
*   **Drift Sensitivity:** Alpha level for the K-S test.

### Zone 3: Telemetry Center (Visuals)
*   **The System Ledger:** 4 Cards (Ingestion Speed, Egress Cost, Data Entropy, Pipeline Health).
*   **The Plot:** **The Data Gravity Waterfall**. Shows the time breakdown of a training epoch (Disk IO vs. Network Transfer vs. GPU Math).

### Zone 4: Audit Trail & Justification
*   **Consequence Log:** "Alert: Egress fees for 1PB transfer exceed $90,000. Move compute to data?"
*   **Rationale Box:** Defend your storage tier choice using the **Feeding Tax** math.

---

## 3. The 3-Act Narrative (The Lab Journey)

### Act I: The Physics of Data Gravity (15m)
*   **Scenario:** You have 1 Petabyte of raw video in a warehouse. Your cluster is 3,000 miles away.
*   **Crisis:** Project deadline is in 10 days. 
*   **Task:** Calculate the transfer time over 1Gbps fiber. Realize it will take weeks. Toggle to "Snowball" (Sneakernet) and see the physical delivery time beat the fiber.

### Act II: The Feeding Tax (15m)
*   **Scenario:** Training ResNet-50 on a Cloud Titan cluster.
*   **Crisis:** GPU utilization is stuck at 15% (85% idle).
*   **Task:** Identify the bottleneck. It's the standard cloud disk ($250$ MB/s). Upgrade to local NVMe and watch the "Feeding Tax" drop to zero.

### Act III: The Drift Detector (15m)
*   **Scenario:** Smart Doorbell deployment.
*   **Crisis:** Detection accuracy is dropping in one city.
*   **Task:** Run a **Kolmogorov-Smirnov (K-S) test** on incoming image distributions. Discover that a firmware update changed the white balance, creating "Semantic Noise."

---

## 4. Real-World Data Sources
*   **Storage:** AWS S3, EBS, and Glacier pricing (2024).
*   **Logistics:** FedEx/AWS Snowball shipping durations and base fees.
*   **Bandwidth:** Standard egress fees ($0.09/GB).
