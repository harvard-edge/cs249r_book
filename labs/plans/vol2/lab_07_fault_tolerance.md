# 📐 Mission Plan: 07_fault_tolerance (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Fault Tolerance: The Reliability Tax.
*   **Core Invariant:** The Young-Daly Invariant ($	au_{opt} = \sqrt{2 \delta M}$) and **MTBF** (Mean Time Between Failures).
*   **The Struggle:** Understanding that at scale, "Hardware Failure is a Routine." Students must navigate the trade-off between **Checkpoint Overhead** ($\delta$) and **Wasted Compute** (work lost between failures), specifically focusing on how cluster size ($N$) collapses the fleet's MTBF.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Reliability Missions)

| Track | Persona | Fixed North Star Mission | The "Failure" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Checkpoint Storm.** Your 16,384-GPU cluster has an MTBF of 4 hours. If you checkpoint too often, you flood the storage; too little, and you lose days of work. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Silent Corruption (SDC).** A Cosmic Ray flipped a bit in the AV perception's weight buffer. The car is now hallucinating 'Green' lights. You must detect the 'Silent Failure'. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Preemption Wall.** The glasses OS keeps killing your model process to save background apps. You must implement 'Incremental Checkpointing' to resume in <100ms. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Brown-out Survival.** The hearable battery is spiking. You must use 'Non-Volatile Memory' (FeRAM) to save state before the power cuts out completely. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Young-Daly Audit (Exploration - 15 Mins)
*   **Objective:** Calculate the "Optimal Checkpoint Interval" for a given cluster size and failure rate.
*   **The "Lock" (Prediction):** "If you double the number of nodes ($N$) in your fleet, does the optimal checkpoint frequency increase, decrease, or stay the same?"
*   **The Workbench:**
    *   **Action:** Slide the **Number of Nodes** ($N$) and **Checkpoint Write Time** ($\delta$).
    *   **Observation:** The **Total Time Waterfall** (Useful Work vs. Checkpoint Overhead vs. Wasted Work). Watch the "Optimal" line move.
*   **Reflect:** "Patterson asks: 'Why is the Reliability Tax higher for the Cloud Titan than for the Tiny Pioneer?' (Reference the $P_{fleet} = P_{node}^N$ formula)."

### Part 2: Asynchronous Staging (Trade-off - 15 Mins)
*   **Objective:** Optimize checkpoint performance using Asynchronous Staging (DRAM -> SSD -> S3).
*   **The "Lock" (Prediction):** "Will staging a checkpoint to local NVMe instead of S3 reduce the 'Math-Stop' time? By what factor?"
*   **The Workbench:**
    *   **Interaction:** Toggle between **Blocking S3**, **Local SSD**, and **Async DRAM** checkpointing.
    *   **Instruments:** **Math-Stop Timer**. **HBM-to-Disk Throughput Gauge**.
    *   **The 10-Iteration Rule:** Students must find the exact "Staging Buffer Size" that prevents the "Checkpoint Storm" from crashing the network.
*   **Reflect:** "Jeff Dean observes: 'Your math-stop time is zero, but your SSD is 100% saturated.' Propose a 'Staggered Checkpoint' strategy to save the disk life."

### Part 3: The SDC Detective (Synthesis - 15 Mins)
*   **Objective:** Implement "Checksumming" and "Dual-Modular Redundancy" to detect Silent Data Corruption.
*   **The "Lock" (Prediction):** "If a weight bit flips from 1 to 0, will the model's 'Loss' function immediately alert you to the error?"
*   **The Workbench:**
    *   **Interaction:** **Inject Bit-Flip**. **Enable Checksums**. **Toggle DMR (Run model twice)**.
    *   **The "Stakeholder" Challenge:** The **Safety Director** (Edge) demands 100% SDC detection. You must prove that **Re-computing the Activation Hash** is 10x cheaper than running the whole model twice.
*   **Reflect (The Ledger):** "Defend your final 'Fault Tolerance Strategy.' Did you prioritize 'Throughput' or 'Integrity'? Justify why 'Availability' was your primary constraint."

---

## 4. Visual Layout Specification
*   **Primary:** `YoungDalyCostCurve` (Wasted Work vs. Checkpoint Interval).
*   **Secondary:** `FailureTopologyMap` (Visualizing which node failed and the 'Rollback' distance).
*   **Math Peek:** Toggle for `	au_{opt} = \sqrt{2 \delta M}` and `MTBF_{fleet} = \frac{MTBF_{node}}{N}`.
