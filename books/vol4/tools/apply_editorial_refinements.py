#!/usr/bin/env python3
"""
book/tools/apply_editorial_refinements.py
Applies high-priority editorial refinements recommended by the 4-Expert Red-Team Review Board.
"""

import re

def patch_ch01():
    path = "book/chapters/01-boundary/01-boundary.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 1. Update trip timing in Table 1-2
    text = text.replace(
        "sub-cycle inverter PWM lock ($\\tau_{\\text{trip}} < 100\\,\\mu\\text{s}$)",
        "sub-microsecond hardware desaturation trip ($\\tau_{\\text{trip}} < 2\\,\\mu\\text{s}$) & thermal lockout"
    )
    
    # 2. Add action chunking note in Sec 1.2
    chunk_note = (
        " Modern foundation policies frequently emit action chunks—sequences of $H$ horizon steps "
        "($H \\approx 16\\text{--}64$) predicting several hundred milliseconds into the future to amortize high neural "
        "inference latencies. An action chunk represents a multi-step open-loop temporal commitment across the causal boundary. "
        "If dynamic contact occurs or unmodeled obstacles appear mid-chunk, the machine will blindly execute the remaining slice "
        "unless an independent, deterministic permission gate continuously monitors invariants and preempts execution."
    )
    if "Modern foundation policies frequently emit action chunks" not in text:
        text = text.replace(
            "where $a_t$ is the action and $P$ is the transition dynamics of the real world.",
            "where $a_t$ is the action and $P$ is the transition dynamics of the real world." + chunk_note
        )

    # 3. Reference IEC 60204-1 stop categories in Sec 1.6
    stop_cat_note = (
        " This refusal hierarchy maps directly to international functional safety standards (IEC 60204-1): "
        "Category 2 (controlled stop with power maintained to hold position), Category 1 (controlled dynamic deceleration "
        "to standstill before removing power), and Category 0 (immediate electromechanical power cutoff and spring-brake engagement)."
    )
    if "This refusal hierarchy maps directly to international functional safety standards" not in text:
        text = text.replace(
            "the system transitions authority to a verified safe-stop controller or mechanical brake.",
            "the system transitions authority to a verified safe-stop controller or mechanical brake." + stop_cat_note
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 01 patched.")

def patch_ch04():
    path = "book/chapters/04-nervous/04-nervous.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Harmonize log-normal quantiles
    old_qn = "For this log-normal distribution, the $P_{99}$ latency is $88\\text{ ms}$ (well within the $100\\text{ ms}$ deadline), but the $P_{99.9}$ latency reaches $132\\text{ ms}$"
    new_qn = "For this log-normal distribution with median $50\\text{ ms}$ and $\\sigma_{\\ln} = 0.35$, the $P_{99}$ latency reaches $112.9\\text{ ms}$ (exceeding the $100\\text{ ms}$ deadline on $2.38\\%$ of cycles), while the $P_{99.9}$ tail extends to $147.5\\text{ ms}$"
    text = text.replace(old_qn, new_qn)

    # Memory barriers in Seqlock
    mem_bar = (
        " To guarantee that out-of-order execution units or optimizing compilers do not read payload data before "
        "validating the sequence counter, the reader and writer must execute hardware memory barriers "
        "(`DMB`/`DSB` on ARM Cortex-M/R, or `stdatomic` memory_order_acquire/release semantics)."
    )
    if "hardware memory barriers" not in text:
        text = text.replace(
            "the reader knows the data was overwritten during the read and retries.",
            "the reader knows the data was overwritten during the read and retries." + mem_bar
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 04 patched.")

def patch_ch07():
    path = "book/chapters/07-evaluation/07-evaluation.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Cite Butler-Finelli bound
    bf_text = (
        " This fundamental mathematical barrier is formalized in safety-critical systems literature as the "
        "**Butler & Finelli Infeasibility Bound** [@butler1993infeasibility]. Butler and Finelli proved that certifying life-critical software "
        "reliability ($10^{-9}\\text{ failures/h}$, as required in civil avionics DAL A and automotive ASIL D) purely through empirical testing "
        "requires over $3.0 \\times 10^9\\text{ hours}$ of continuous, failure-free operation—an exposure budget that exceeds hundreds of thousands of machine-years. "
        "Empirical testing alone cannot certify high-integrity autonomy; safety must be guaranteed by deterministic runtime architectural invariants."
    )
    if "Butler & Finelli Infeasibility Bound" not in text:
        text = text.replace(
            "This table exposes the physical cost of statistical confidence.",
            bf_text + "\n\nThis table exposes the physical cost of statistical confidence."
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 07 patched.")

def patch_ch09():
    path = "book/chapters/09-memory/09-memory.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Action chunking horizon clipping
    chunk_clip = (
        " In modern embodied foundation architectures (such as Diffusion Policies and Action Chunking with Transformers), "
        "policies predict trajectory action chunks spanning future horizons $H$ ($150\\text{--}500\\text{ ms}$). Under dynamic "
        "occlusions or sensor degradation, the open-loop execution of an action chunk must be dynamically clipped to "
        "$\\min(H \\cdot \\Delta t_{\\text{control}}, t_{\\text{exp}})$. This prevents the machine from executing unguided trajectory slices "
        "after the spatial belief state has crossed its invalidation horizon."
    )
    if "Action Chunking with Transformers" not in text:
        text = text.replace(
            "We now turn to the mathematical formalization of this expiry threshold.",
            chunk_clip + "\n\nWe now turn to the mathematical formalization of this expiry threshold."
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 09 patched.")

def patch_ch10():
    path = "book/chapters/10-intent/10-intent.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # IEEE 1588 note in 10.7
    ptp_note = (
        " In multi-core asymmetric architectures where the Brain (Linux MPU) and Nervous System (MCU) reside on separate silicon, "
        "absolute timestamp verification ($t_{\\text{expire}}$) assumes sub-microsecond **IEEE 1588 PTP (Precision Time Protocol)** synchronization "
        "across the real-time interconnect bus. In resource-constrained microcontrollers lacking hardware PTP, the lease contract should "
        "arm a local hardware monotonic down-counter initialized to $\\tau_{\\text{remain}} = \\tau$ upon packet arrival."
    )
    if "IEEE 1588 PTP" not in text:
        text = text.replace(
            "Every field in the schema serves a specific verification check.",
            ptp_note + "\n\nEvery field in the schema serves a specific verification check."
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 10 patched.")

def patch_ch11():
    path = "book/chapters/11-planning/11-planning.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Figure 11.4 caption current loop clamp note
    text = text.replace(
        "transmitting a destructive impulsive torque of $\\tau_{\\text{seam}} = 324\\text{ N}\\cdot\\text{m}$ through the reducer",
        "attempting to transmit $J_{\\text{eff}}\\ddot{q} = 1620\\text{ N}\\cdot\\text{m}$, which saturates the inverter current loop at its $324\\text{ N}\\cdot\\text{m}$ ceiling"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 11 patched.")

def patch_ch12():
    path = "book/chapters/12-enforcement/12-enforcement.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean up any malformed artifacts from previous patch
    text = re.sub(r'::: \{\.callout-lab[\s\S]*?:::', '', text)
    
    # 1. Update line 319 watchdog phrasing
    text = text.replace(
        "when a physical emergency stop pushbutton is depressed, an over-current comparator trips, or a hardware watchdog timer exceeds its 50 ms timeout,",
        "when a physical emergency stop pushbutton is depressed, an over-current comparator trips, the upstream intent lease expires ($\\tau_{\\text{lease}} > 40\\text{ ms}$), or the bare-metal microcontroller task watchdog detects a loop stall ($t_{\\text{timeout}} \\ge 3.0\\text{ ms}$),"
    )

    # 2. Add fallacy 1
    fallacy1 = """
::: {.callout-fallacy title="The Fallacy of Uncoordinated Scalar Clipping"}
**The Belief:** Clamping each actuator's torque command independently to its maximum datasheet limit is sufficient to keep an articulated robot safe.

**The Reality:** In multi-link articulated mechanisms, joint accelerations are coupled through the off-diagonal terms of the mass matrix $\\mathbf{M}(\\mathbf{q})$. Clamping individual joint torques independently rotates the net Cartesian wrench vector away from its intended line of action, exerting unmodeled lateral forces and destructive torsional shock loads on gearboxes. Real-time safety enforcement requires constrained quadratic optimization across all actuation degrees of freedom simultaneously.
:::
"""
    if "The Fallacy of Uncoordinated Scalar Clipping" not in text:
        text = text.replace("(@tbl-12-cbf-parameters):", "(@tbl-12-cbf-parameters):\n" + fallacy1)

    # 3. Add data endogeneity paragraph in 12.8
    endogeneity = """
When an enforcer modifies an unverified proposal ($\\Delta \\mathbf{u} = \\mathbf{u}^* - \\mathbf{u}_{\\text{nom}} \\neq \\mathbf{0}$), the resulting state trajectory diverges from the policy's unconstrained intent. In data collection pipelines for behavioral cloning or offline reinforcement learning, logging this intervened trajectory as a normal expert demonstration induces **data endogeneity**. The neural policy inadvertently learns that it can command unsafe actions because downstream filters will intervene, causing policy collapse near boundary states. To preserve dataset integrity, the nervous system must emit an explicit `truncated` flag on any cycle where the enforcer projects commands or escalates down the fallback ladder, allowing downstream training pipelines to truncate the episode horizon during imitation learning.
"""
    if "data endogeneity" not in text:
        text = text.replace("### Data Ingestion and the Zero-Allocation Rule", "### Data Ingestion and the Zero-Allocation Rule\n\n" + endogeneity)

    # 4. Add fallacy 2
    fallacy2 = """
::: {.callout-fallacy title="The Fallacy of Redundant Silicon"}
**The Belief:** Placing safety monitoring software on a secondary microcontroller provides independent safety enforcement.

**The Reality:** If the secondary microcontroller shares a power supply regulator, a crystal oscillator, a printed circuit board ground plane, or an un-preemptible communication bus with the primary application processor, the two chips remain tightly coupled. A brownout transient, ground bounce, or bus DMA priority inversion will fault both processors simultaneously. Physical safety requires galvanic, temporal, and electrical isolation down to the silicon floor.
:::
"""
    if "The Fallacy of Redundant Silicon" not in text:
        text = text.replace("## Incident Autopsies: Enforcement Failures", fallacy2 + "\n\n## Incident Autopsies: Enforcement Failures")

    # 5. Add callout-lab at the end
    callout_lab = """
::: {.callout-lab title="Lab 08: The 1 kHz MCU Safety Enforcer"}
Test the boundary between unverified learned proposals and deterministic physical authority on the dual-brain kit:

1. **Bare-Metal CBF-QP Implementation:** Deploy an active-set Control Barrier Function safety filter on the ARM Cortex-M4 microcontroller running at $1000\\text{ Hz}$ in zero-allocation static SRAM.
2. **Proposal Filtering on the Bench:** Stream unconstrained, out-of-envelope trajectory proposals from the Linux MPU over shared memory and verify that the MCU orthogonally projects commands onto the safe set boundary $\\partial \\mathcal{C}$ without distorting net Cartesian direction.
3. **Seeded Crash Injection:** Trigger a synthetic Linux kernel panic during a high-speed motion sweep. Measure the time required for the MCU hardware watchdog to detect heartbeat cessation ($t_{\\text{timeout}} \\le 3.0\\text{ ms}$) and verify autonomous Level 2 deceleration to position hold before carriage travel reaches physical end-stops.

*Hardware bench guide and starter firmware:* [`labs/08-mcu-enforcer/`](file:///Users/VJ/GitHub/PhysicalAI-draft/labs/08-mcu-enforcer/)
:::
"""
    text = text.strip() + "\n\n" + callout_lab

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 12 patched.")

def patch_ch14():
    path = "book/chapters/14-intervention/14-intervention.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Fix blending duration factor from 1.0 to 1.5 in Sec 14.8
    text = text.replace(
        "\\tau_{\\text{blend}} \\ge 25 / 500 = 50\\text{ ms}",
        "\\tau_{\\text{blend}} \\ge \\frac{1.5 \\times 25\\text{ N}\\cdot\\text{m}}{500\\text{ N}\\cdot\\text{m/s}} = 75\\text{ ms}"
    )
    text = text.replace(
        "1000\\text{ ms} + 20\\text{ ms} + 50\\text{ ms} = 1070\\text{ ms}",
        "1000\\text{ ms} + 20\\text{ ms} + 75\\text{ ms} = 1095\\text{ ms}"
    )
    text = text.replace(
        "22\\text{ m/s} \\times 1.07\\text{ s} = 23.54\\text{ m}",
        "22\\text{ m/s} \\times 1.095\\text{ s} = 24.09\\text{ m}"
    )
    text = text.replace(
        "leaving a defended clearance margin of $65.0\\text{ m} - (23.54\\text{ m} + 35.0\\text{ m}) = 6.46\\text{ m}$",
        "leaving a defended clearance margin of $65.0\\text{ m} - (24.09\\text{ m} + 35.0\\text{ m}) = 5.91\\text{ m}$"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 14 patched.")

def patch_ch15():
    path = "book/chapters/15-verification/15-verification.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Align qualification ladder prose in 15.4
    old_ladder = "offline trace replay, closed-loop software simulation, hardware-in-the-loop test benches, and live shadow deployment"
    new_ladder = "Software-in-the-Loop (SIL), Processor-in-the-Loop (PIL), Hardware-in-the-Loop (HIL), and In-Situ Physical Fault Injection"
    text = text.replace(old_ladder, new_ladder)

    pil_prose = (
        " In the **Processor-in-the-Loop (PIL)** stage, compiled firmware binaries execute directly on the target microcontroller ISA "
        "(e.g., ARM Cortex-R52 or TI TMS570) coupled to an emulated plant model. PIL qualification isolates target-compiler optimization bugs, "
        "register-level latching timing, and instruction cache evictions before physical actuators and power fieldbuses are connected."
    )
    if "Processor-in-the-Loop (PIL) stage" not in text:
        text = text.replace(
            "Software simulation validates policy behavior across millions of randomized environmental variations.",
            "Software simulation (SIL) validates policy behavior across millions of randomized environmental variations." + pil_prose
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 15 patched.")

def patch_ch16():
    path = "book/chapters/16-release/16-release.qmd"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Stopping distance total calculation with detection reaction in 16.2
    text = text.replace(
        "the stopping distance is $d_{\\text{stop}} = 36\\text{ mm}$. If the spatial clearance to the rigid partition is less than $36\\text{ mm}$",
        "the braking displacement is $d_{\\text{brake}} = 36.0\\text{ mm}$. Accounting for sensor detection latency ($t_{\\text{detect}} \\le 2.0\\text{ ms}$ traversing $d_{\\text{react}} = 3.6\\text{ mm}$ at $1.8\\text{ m/s}$), the total required clearance is $d_{\\text{total}} = d_{\\text{react}} + d_{\\text{brake}} = 39.6\\text{ mm}$. If the spatial clearance to the rigid partition is less than $39.6\\text{ mm}$"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Ch 16 patched.")

def main():
    patch_ch01()
    patch_ch04()
    patch_ch07()
    patch_ch09()
    patch_ch10()
    patch_ch11()
    patch_ch12()
    patch_ch14()
    patch_ch15()
    patch_ch16()
    print("All editorial board patches applied successfully.")

if __name__ == "__main__":
    main()
