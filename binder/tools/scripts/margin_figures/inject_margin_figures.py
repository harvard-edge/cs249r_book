#!/usr/bin/env python3
"""
inject_margin_figures.py

Performs exact, verified injections of all 34 approved Volume IV margin visual figures
and necessary locator / pointer adjustments across the 17 chapters of MLSysBook-vol4.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
VOL4_ROOT = REPO_ROOT / "books" / "vol4"

def replace_exact(file_path: Path, target: str, replacement: str, description: str):
    content = file_path.read_text(encoding="utf-8")
    count = content.count(target)
    if count == 0:
        raise ValueError(f"[{description}] Target string not found in {file_path.name}:\n{target[:120]}...")
    elif count > 1:
        raise ValueError(f"[{description}] Target string matched {count} times (must be unique) in {file_path.name}")
    
    new_content = content.replace(target, replacement, 1)
    file_path.write_text(new_content, encoding="utf-8")
    print(f"✓ Injected: {description} in {file_path.name}")

def main():
    print(f"Injecting margin figures into Volume IV chapters under {VOL4_ROOT}...")

    # =========================================================================
    # Ch 01: 01_boundary.qmd
    # =========================================================================
    ch01 = VOL4_ROOT / "01_boundary" / "01_boundary.qmd"
    
    # Fig 1: margin_timescale_separation_ladder.svg
    target_01_1 = (
        "As introduced in @fig-01-locator, every physical AI system—whether a humanoid biped, "
        "an autonomous tractor, or an automated surgical console—is structured across four "
        "architectural tiers: the **Dynamical Body** (Layer 1), the **Real-Time Nervous System** (Layer 2), "
        "the **Cognitive Brain** (Layer 3), and the **Governance Envelope** (Layer 4).\n\n"
        "Rather than treating these tiers as an undifferentiated software stack, systems engineering requires organizing them around **timescale separation**, **hardware interfaces**, and **contract boundaries**:"
    )
    repl_01_1 = (
        "As introduced in @fig-01-locator, every physical AI system—whether a humanoid biped, "
        "an autonomous tractor, or an automated surgical console—is structured across four "
        "architectural tiers: the **Dynamical Body** (Layer 1), the **Real-Time Nervous System** (Layer 2), "
        "the **Cognitive Brain** (Layer 3), and the **Governance Envelope** (Layer 4).\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_timescale_separation_ladder.svg){width=\"100%\" fig-alt=\"Five-tier physical AI hierarchy ladder from twenty kilohertz motor control to sub-hertz cognitive models.\"}\n\n"
        "*Control loops span five orders of magnitude between cognitive deliberation and inverter switching.*\n"
        ":::\n\n"
        "Rather than treating these tiers as an undifferentiated software stack, systems engineering requires organizing them around **timescale separation**, **hardware interfaces**, and **contract boundaries**:"
    )
    replace_exact(ch01, target_01_1, repl_01_1, "Ch 01 timescale separation ladder")

    # Fig 2: margin_covariate_shift_divergence.svg
    target_01_2 = (
        "For an ML systems engineer, this means that an embodied policy with 99 percent single-step "
        "offline validation accuracy can suffer a 100 percent task failure rate after only 50 closed-loop physical steps.\n\n"
        "[^fn-ml-covariate-drift]:"
    )
    repl_01_2 = (
        "For an ML systems engineer, this means that an embodied policy with 99 percent single-step "
        "offline validation accuracy can suffer a 100 percent task failure rate after only 50 closed-loop physical steps.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_covariate_shift_divergence.svg){width=\"100%\" fig-alt=\"Quadratic error divergence curve showing 99 percent accurate policy collapsing within 50 steps.\"}\n\n"
        "*Closed-loop interaction compounds policy errors quadratically, collapsing 99 percent accuracy within 50 steps.*\n"
        ":::\n\n"
        "[^fn-ml-covariate-drift]:"
    )
    replace_exact(ch01, target_01_2, repl_01_2, "Ch 01 covariate shift divergence")

    # =========================================================================
    # Ch 02: 02_body.qmd
    # =========================================================================
    ch02 = VOL4_ROOT / "02_body" / "02_body.qmd"

    # Relocate spa_locator_a.svg up from line 383 to after line 368
    old_locator_02 = (
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_a.svg){width=\"100%\" fig-alt=\"S·P·A locator triad with the Act node highlighted.\"}\n\n"
        "*The Act axis: non-smooth contact mechanics, reflected inertia, and thermal dissipation.*\n"
        ":::\n\n"
        "Transmissions introduce a severe mechanical asymmetry that software planners routinely ignore."
    )
    repl_remove_locator_02 = "Transmissions introduce a severe mechanical asymmetry that software planners routinely ignore."
    replace_exact(ch02, old_locator_02, repl_remove_locator_02, "Ch 02 remove old spa_locator_a")

    target_02_reloc = (
        "A digital setpoint is merely a software request; physical action is bounded by winding inductance, "
        "bus voltage ceilings, magnetic saturation, and transmission elasticity.\n\n"
        "The delay between commanding a torque and delivering it at the motor shaft begins in the stator windings."
    )
    repl_02_reloc = (
        "A digital setpoint is merely a software request; physical action is bounded by winding inductance, "
        "bus voltage ceilings, magnetic saturation, and transmission elasticity.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_a.svg){width=\"100%\" fig-alt=\"S·P·A locator triad with the Act node highlighted.\"}\n\n"
        "*The Act axis: non-smooth contact mechanics, reflected inertia, and thermal dissipation.*\n"
        ":::\n\n"
        "The delay between commanding a torque and delivering it at the motor shaft begins in the stator windings."
    )
    replace_exact(ch02, target_02_reloc, repl_02_reloc, "Ch 02 relocate spa_locator_a")

    # Fig 3: margin_reflected_rotor_inertia_knee.svg
    target_02_knee = (
        "When $N < N^*$, joint acceleration is torque-limited. When $N > N^*$, counter-intuitively, increasing the gear ratio *decreases* delivered joint acceleration, because the motor spends the majority of its electromagnetic torque accelerating its own spinning rotor rather than moving the robot's physical link.\n\n"
        "Because the actuator cannot instantly deliver the massive torque needed to spin up this amplified inertia"
    )
    repl_02_knee = (
        "When $N < N^*$, joint acceleration is torque-limited. When $N > N^*$, counter-intuitively, increasing the gear ratio *decreases* delivered joint acceleration, because the motor spends the majority of its electromagnetic torque accelerating its own spinning rotor rather than moving the robot's physical link.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_reflected_rotor_inertia_knee.svg){width=\"100%\" fig-alt=\"Two-tone curve of joint acceleration versus gear ratio peaking at thirty-nine then dropping sharply into a shaded red rotor-inertia-dominated regime.\"}\n\n"
        "*Past the inertia-matching ratio, gearing chokes delivered joint acceleration.*\n"
        ":::\n\n"
        "Because the actuator cannot instantly deliver the massive torque needed to spin up this amplified inertia"
    )
    replace_exact(ch02, target_02_knee, repl_02_knee, "Ch 02 reflected rotor inertia knee")

    # Fig 4: margin_pdn_voltage_droop_envelope.svg
    target_02_droop = (
        "Characterizing these dynamic electrical and mechanical failure boundaries cannot rely on nominal catalog specs or idealized CAD models; it demands disciplined bench measurement.\n\n"
        "To quantify the coupled dynamics of rail collapse and regenerative energy feedback"
    )
    repl_02_droop = (
        "Characterizing these dynamic electrical and mechanical failure boundaries cannot rely on nominal catalog specs or idealized CAD models; it demands disciplined bench measurement.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_pdn_voltage_droop_envelope.svg){width=\"100%\" fig-alt=\"Horizontal budget envelope showing nominal 48V rail, 41V brownout floor, and a 12.8V transient droop plunging the rail into an undervoltage reset.\"}\n\n"
        "*Transient current surges burn through voltage headroom into unrecoverable brownout resets.*\n"
        ":::\n\n"
        "To quantify the coupled dynamics of rail collapse and regenerative energy feedback"
    )
    replace_exact(ch02, target_02_droop, repl_02_droop, "Ch 02 PDN voltage droop envelope")

    # =========================================================================
    # Ch 03: 03_brain.qmd
    # =========================================================================
    ch03 = VOL4_ROOT / "03_brain" / "03_brain.qmd"

    # Relocate spa_locator_p.svg from line 404 up to after line 64
    old_locator_03 = (
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_p.svg){width=\"100%\" fig-alt=\"S·P·A locator triad with the Plan node highlighted.\"}\n\n"
        "*The Plan axis: high-capacity neural deliberation emitting unprivileged action chunks.*\n"
        ":::\n\n"
        "For example, when a canonical bimanual manipulator threads a flexible USB cable"
    )
    repl_remove_locator_03 = "For example, when a canonical bimanual manipulator threads a flexible USB cable"
    replace_exact(ch03, old_locator_03, repl_remove_locator_03, "Ch 03 remove old spa_locator_p")

    target_03_reloc = (
        "This asymmetric boundary ensures that cognitive exploration and semantic generalization never jeopardize the mechanical integrity of the machine.\n\n"
        "[^fn-brain-reflex-governor]:"
    )
    repl_03_reloc = (
        "This asymmetric boundary ensures that cognitive exploration and semantic generalization never jeopardize the mechanical integrity of the machine.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_p.svg){width=\"100%\" fig-alt=\"S·P·A locator triad with the Plan node highlighted.\"}\n\n"
        "*The Plan axis: high-capacity neural deliberation emitting unprivileged action chunks.*\n"
        ":::\n\n"
        "[^fn-brain-reflex-governor]:"
    )
    replace_exact(ch03, target_03_reloc, repl_03_reloc, "Ch 03 relocate spa_locator_p")

    # Fig 5: margin_action_chunk_windows_strip.svg
    target_03_strip = (
        "For example, when a canonical bimanual manipulator threads a flexible USB cable (@fig-aloha-manipulator), it executes a 32-step action chunk. The policy emits coordinated 14-DoF joint setpoints that smoothly align both grippers, flex the cable, and insert the connector without pausing between individual control cycles.\n\n"
        "However, action chunking introduces the critical systems challenge of **chunk seam continuity**."
    )
    repl_03_strip = (
        "For example, when a canonical bimanual manipulator threads a flexible USB cable (@fig-aloha-manipulator), it executes a 32-step action chunk. The policy emits coordinated 14-DoF joint setpoints that smoothly align both grippers, flex the cable, and insert the connector without pausing between individual control cycles.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_action_chunk_windows_strip.svg){width=\"100%\" fig-alt=\"Sequence strip illustrating overlapping action chunk horizons of 32 steps with 12-step replanning and temporal ensembling.\"}\n\n"
        "*Overlapping action chunks blend predictions across receding temporal execution windows.*\n"
        ":::\n\n"
        "However, action chunking introduces the critical systems challenge of **chunk seam continuity**."
    )
    replace_exact(ch03, target_03_strip, repl_03_strip, "Ch 03 action chunk windows strip")

    # =========================================================================
    # Ch 04: 04_nervous.qmd
    # =========================================================================
    ch04 = VOL4_ROOT / "04_nervous" / "04_nervous.qmd"

    # Fig 6: margin_fieldbus_cycle_budget_envelope.svg
    target_04_fb = (
        "Read as a sequence rather than a table, these protocols trace one trajectory. @Fig-04-fieldbus-evolution plots bandwidth against achievable cycle time from Modbus RTU to TSN Ethernet,[^fn-net-tsn-fieldbus] and each generation moves down and to the right, which is why a modern multi-axis machine can carry sensor payloads that would have been impossible to schedule deterministically a decade earlier.\n\n"
        "[^fn-net-tsn-fieldbus]:"
    )
    repl_04_fb = (
        "Read as a sequence rather than a table, these protocols trace one trajectory. @Fig-04-fieldbus-evolution plots bandwidth against achievable cycle time from Modbus RTU to TSN Ethernet,[^fn-net-tsn-fieldbus] and each generation moves down and to the right, which is why a modern multi-axis machine can carry sensor payloads that would have been impossible to schedule deterministically a decade earlier.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_fieldbus_cycle_budget_envelope.svg){width=\"100%\" fig-alt=\"Horizontal budget envelope showing eight-axis CAN-FD overrunning a one-millisecond cycle budget while EtherCAT uses thirty microseconds.\"}\n\n"
        "*Eight-axis polling overruns CAN-FD cycle budgets while EtherCAT leaves 97 percent headroom.*\n"
        ":::\n\n"
        "[^fn-net-tsn-fieldbus]:"
    )
    replace_exact(ch04, target_04_fb, repl_04_fb, "Ch 04 fieldbus cycle budget envelope")

    # Fig 7: margin_substrate_blast_radius.svg
    target_04_blast = (
        "Even a common clock oscillator creates an unbudgeted single point of failure, where thermal drift, mechanical vibration, or crystal cracking desynchronizes both domains at the same instant.\n\n"
        "A deterministic nervous system must distinguish between two fundamentally distinct failure modes in the brain: unresponsiveness and corruption."
    )
    repl_04_blast = (
        "Even a common clock oscillator creates an unbudgeted single point of failure, where thermal drift, mechanical vibration, or crystal cracking desynchronizes both domains at the same instant.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_substrate_blast_radius.svg){width=\"100%\" fig-alt=\"Blast radius diagram showing an NPU 50A transient current spike propagating into voltage droop, thermal sag, DMA starvation, and clock jitter.\"}\n\n"
        "*Unpartitioned physical substrates allow cognitive compute spikes to crash real-time safety controllers.*\n"
        ":::\n\n"
        "A deterministic nervous system must distinguish between two fundamentally distinct failure modes in the brain: unresponsiveness and corruption."
    )
    replace_exact(ch04, target_04_blast, repl_04_blast, "Ch 04 substrate blast radius")

    # =========================================================================
    # Ch 05: 05_data.qmd
    # =========================================================================
    ch05 = VOL4_ROOT / "05_data" / "05_data.qmd"

    # Fig 8: margin_sensory_ingestion_bandwidth_ladder.svg
    target_05_bw = (
        "prevent OS buffer overruns and silent frame drops.\n"
        ":::\n\n"
        "Sustained multi-camera ingestion directly bypasses operating-system page caches to maintain predictable write throughput.[^fn-data-nvme-direct]"
    )
    repl_05_bw = (
        "prevent OS buffer overruns and silent frame drops.\n"
        ":::\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_sensory_ingestion_bandwidth_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder comparing camera raw bandwidth against USB, SATA, and NVMe bus ceilings.\"}\n\n"
        "*Multi-camera raw telemetry overwhelms legacy storage interfaces, requiring direct PCIe NVMe logging.*\n"
        ":::\n\n"
        "Sustained multi-camera ingestion directly bypasses operating-system page caches to maintain predictable write throughput.[^fn-data-nvme-direct]"
    )
    replace_exact(ch05, target_05_bw, repl_05_bw, "Ch 05 sensory ingestion bandwidth ladder")

    # Fig 9: margin_facility_yield_burndown_envelope.svg
    target_05_yield = (
        "Understanding the safety and performance limits of a learned policy therefore requires analyzing how the collecting policy's operational biases shape its mathematical coverage across the state space—the focus of @sec-data-policy-coverage.\n\n"
        "## Collection Policy Coverage {#sec-data-policy-coverage}"
    )
    repl_05_yield = (
        "Understanding the safety and performance limits of a learned policy therefore requires analyzing how the collecting policy's operational biases shape its mathematical coverage across the state space—the focus of @sec-data-policy-coverage.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_facility_yield_burndown_envelope.svg){width=\"100%\" fig-alt=\"Budget envelope showing scheduled sixty-minute recording shift burning down to 14.7 minutes of usable data.\"}\n\n"
        "*Hardware resets and teleoperation failures reduce scheduled recording hours to 25 percent usable yield.*\n"
        ":::\n\n"
        "## Collection Policy Coverage {#sec-data-policy-coverage}"
    )
    replace_exact(ch05, target_05_yield, repl_05_yield, "Ch 05 facility yield burndown envelope")

    # =========================================================================
    # Ch 06: 06_training.qmd
    # =========================================================================
    ch06 = VOL4_ROOT / "06_training" / "06_training.qmd"

    # Relocate spa_locator_pa.svg from line 348 up to after line 84
    old_locator_06 = (
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_pa.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the Plan-Act motion edge.\"}\n\n"
        "*The Plan $\\cap$ Act motion interface: grounding learned policy tokens into continuous physical trajectories.*\n"
        ":::\n\n"
        "To see how diffusion denoising interacts with real-time execution deadlines"
    )
    repl_remove_locator_06 = "To see how diffusion denoising interacts with real-time execution deadlines"
    replace_exact(ch06, old_locator_06, repl_remove_locator_06, "Ch 06 remove old spa_locator_pa")

    target_06_reloc = (
        "[^fn-training-proposal-permission]: **Proposal-permission architecture**\\index{Proposal-permission architecture!enforcer}: Decoupling unconstrained policy inference from physical execution guarantees safety invariants without retraining neural weights. The deterministic filter evaluates policy commands against control barrier certificates and dynamic friction limits before writing to actuator registers. If a proposed action violates kinematic bounds, the enforcer projects the command onto the boundary of the safe set within a single control tick.\n\n"
        "Navigating these trades requires structuring policy synthesis as an end-to-end systems engineering workflow."
    )
    repl_06_reloc = (
        "[^fn-training-proposal-permission]: **Proposal-permission architecture**\\index{Proposal-permission architecture!enforcer}: Decoupling unconstrained policy inference from physical execution guarantees safety invariants without retraining neural weights. The deterministic filter evaluates policy commands against control barrier certificates and dynamic friction limits before writing to actuator registers. If a proposed action violates kinematic bounds, the enforcer projects the command onto the boundary of the safe set within a single control tick.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_pa.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the Plan-Act motion edge.\"}\n\n"
        "*The Plan $\\cap$ Act motion interface: grounding learned policy tokens into continuous physical trajectories.*\n"
        ":::\n\n"
        "Navigating these trades requires structuring policy synthesis as an end-to-end systems engineering workflow."
    )
    replace_exact(ch06, target_06_reloc, repl_06_reloc, "Ch 06 relocate spa_locator_pa")

    # Fig 10: margin_generative_inference_deadlines_envelope.svg
    target_06_diff = (
        "The edge SoC provides unified memory bandwidth $BW_{\\text{mem}} = 204.8\\text{ GB/s}$ and dense compute $C_{\\text{peak}} = 40\\text{ TFLOPs}$. Analyzing this compute pipeline reveals three distinct timing phases:\n\n"
        "1. **Perception backbone latency**:"
    )
    repl_06_diff = (
        "The edge SoC provides unified memory bandwidth $BW_{\\text{mem}} = 204.8\\text{ GB/s}$ and dense compute $C_{\\text{peak}} = 40\\text{ TFLOPs}$. Analyzing this compute pipeline reveals three distinct timing phases:\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_generative_inference_deadlines_envelope.svg){width=\"100%\" fig-alt=\"Budget envelope comparing flow matching and DDIM inference against a twenty-millisecond real-time deadline.\"}\n\n"
        "*Diffusion denoising overruns real-time feedback deadlines while two-step flow matching preserves timing slack.*\n"
        ":::\n\n"
        "1. **Perception backbone latency**:"
    )
    replace_exact(ch06, target_06_diff, repl_06_diff, "Ch 06 generative inference deadlines envelope")

    # Fig 11: margin_mechanical_fatigue_ladder.svg
    target_06_fatigue = (
        "At an average electrical power demand of $350\\text{ W}$ per test bench, executing $1.2 \\times 10^7\\text{ s}$ of automated trials draws $1167\\text{ kWh}$ of electrical energy, transforming policy training into an industrial-scale manufacturing operation.\n\n"
        "[^fn-hw-bearing-life]:"
    )
    repl_06_fatigue = (
        "At an average electrical power demand of $350\\text{ W}$ per test bench, executing $1.2 \\times 10^7\\text{ s}$ of automated trials draws $1167\\text{ kWh}$ of electrical energy, transforming policy training into an industrial-scale manufacturing operation.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_mechanical_fatigue_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder comparing RL trial cycles against mechanical gearhead and tendon cable fatigue lives.\"}\n\n"
        "*Real-world reinforcement learning exploration rapidly exhausts mechanical gearhead and cable fatigue lifespans.*\n"
        ":::\n\n"
        "[^fn-hw-bearing-life]:"
    )
    replace_exact(ch06, target_06_fatigue, repl_06_fatigue, "Ch 06 mechanical fatigue ladder")

    # Fig 12: margin_contact_stiffness_cliff_knee.svg + relocate pointer
    target_06_knee = (
        "Preventing this destruction requires independent nervous system safeguards: a real-time MCU approach velocity governor enforcing $v_{\\text{max}}(z) \\le \\sqrt{2 a_{\\text{max}} z}$, and a $1000\\text{ Hz}$ force-derivative tripwire detecting $dF/dt > 5.0\\times 10^3\\text{ N/s}$ to clamp dynamic brakes within $1.0\\text{ ms}$.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "⇄ **Contrast:** Simulated penalty spring deformation ($2.4\\text{ mm}$) contrasts with the non-smooth Coulomb friction cones analyzed in @sec-body-actuator-limits.\n"
        ":::\n\n"
        "::: {#chk-training-sim2real-contact-impulse"
    )
    repl_06_knee = (
        "Preventing this destruction requires independent nervous system safeguards: a real-time MCU approach velocity governor enforcing $v_{\\text{max}}(z) \\le \\sqrt{2 a_{\\text{max}} z}$, and a $1000\\text{ Hz}$ force-derivative tripwire detecting $dF/dt > 5.0\\times 10^3\\text{ N/s}$ to clamp dynamic brakes within $1.0\\text{ ms}$.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_contact_stiffness_cliff_knee.svg){width=\"100%\" fig-alt=\"Knee curve showing simulation penalty spring force capping at 94 Newtons while physical rigid contact spikes to 3770 Newtons.\"}\n\n"
        "*Rigid steel contact spikes real impact force 40-fold higher than compliant simulator springs.*\n"
        ":::\n\n"
        "::: {#chk-training-sim2real-contact-impulse"
    )
    replace_exact(ch06, target_06_knee, repl_06_knee, "Ch 06 contact stiffness cliff knee")

    target_06_ptr_new = (
        "Conversely, a policy that regulates force while holding a static contact will enter uncontrolled limit-cycle oscillations if static friction, stick-slip transitions, and actuator deadbands are omitted from the simulator.\n\n"
        "Establishing the validity of a simulator component requires **paired probing**"
    )
    repl_06_ptr_new = (
        "Conversely, a policy that regulates force while holding a static contact will enter uncontrolled limit-cycle oscillations if static friction, stick-slip transitions, and actuator deadbands are omitted from the simulator.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "⇄ **Contrast:** Simulated penalty spring deformation ($2.4\\text{ mm}$) contrasts with the non-smooth Coulomb friction cones analyzed in @sec-body-actuator-limits.\n"
        ":::\n\n"
        "Establishing the validity of a simulator component requires **paired probing**"
    )
    replace_exact(ch06, target_06_ptr_new, repl_06_ptr_new, "Ch 06 relocate contrast pointer")

    # =========================================================================
    # Ch 07: 07_evaluation.qmd
    # =========================================================================
    ch07 = VOL4_ROOT / "07_evaluation" / "07_evaluation.qmd"

    # Fig 13: margin_nonasymptotic_reliability_ladder.svg
    target_07_cp = (
        "A physical robot with a flawed policy that fails once in every seven deployments ($p = 87.5$ percent) will successfully complete twenty consecutive test runs more than 6.9 percent of the time ($0.875^{20} \\approx 0.0692$) by pure random chance.\n\n"
        "::: {#psp-evaluation-twenty-clean-runs"
    )
    repl_07_cp = (
        "A physical robot with a flawed policy that fails once in every seven deployments ($p = 87.5$ percent) will successfully complete twenty consecutive test runs more than 6.9 percent of the time ($0.875^{20} \\approx 0.0692$) by pure random chance.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_nonasymptotic_reliability_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder comparing trial sample sizes against achievable statistical reliability bounds.\"}\n\n"
        "*Twenty successful trials prove only an 86 percent reliability bound, requiring thousands of runs.*\n"
        ":::\n\n"
        "::: {#psp-evaluation-twenty-clean-runs"
    )
    replace_exact(ch07, target_07_cp, repl_07_cp, "Ch 07 nonasymptotic reliability ladder")

    # Fig 14: margin_extreme_value_shock_sparkline.svg
    target_07_evt = (
        "Protecting the structure requires physical series elastic actuators or elastomeric sole pads to lengthen deceleration from $1.5\\text{ ms}$ to $22.0\\text{ ms}$, alongside a $1\\text{ kHz}$ real-time MCU force-derivative clamp ($dF/dt \\le 1.2\\times 10^4\\text{ N/s}$) that commands active knee flexion impedance control within $1.0\\text{ ms}$.\n\n"
        "::: {#chk-eval-evt-contact-force"
    )
    repl_07_evt = (
        "Protecting the structure requires physical series elastic actuators or elastomeric sole pads to lengthen deceleration from $1.5\\text{ ms}$ to $22.0\\text{ ms}$, alongside a $1\\text{ kHz}$ real-time MCU force-derivative clamp ($dF/dt \\le 1.2\\times 10^4\\text{ N/s}$) that commands active knee flexion impedance control within $1.0\\text{ ms}$.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_extreme_value_shock_sparkline.svg){width=\"100%\" fig-alt=\"Sparkline comparing Gaussian distribution tail against Fréchet extreme value heavy tail.\"}\n\n"
        "*Heavy-tailed physical shocks turn Gaussian-impossible events into failures occurring every few minutes.*\n"
        ":::\n\n"
        "::: {#chk-eval-evt-contact-force"
    )
    replace_exact(ch07, target_07_evt, repl_07_evt, "Ch 07 extreme value shock sparkline")

    # =========================================================================
    # Ch 08: 08_perception.qmd
    # =========================================================================
    ch08 = VOL4_ROOT / "08_perception" / "08_perception.qmd"

    # Move prerequisite pointer up to line 83 and inject phase margin knee after line 105
    target_08_ptr_move = (
        "Downstream software cannot safely consume a perception output that does not explicitly declare when it was captured, what coordinate system it references, and how wide its error distribution remains.\n\n"
        "::: {#dfn-perception-spatial-information-age"
    )
    repl_08_ptr_move = (
        "Downstream software cannot safely consume a perception output that does not explicitly declare when it was captured, what coordinate system it references, and how wide its error distribution remains.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "↰ **Prerequisite:** Transduction freshness and sensor measurement aging are derived in @sec-body-measurement-freshness.\n"
        ":::\n\n"
        "::: {#dfn-perception-spatial-information-age"
    )
    replace_exact(ch08, target_08_ptr_move, repl_08_ptr_move, "Ch 08 relocate prerequisite pointer up")

    # Fig 15: margin_phase_margin_erosion_knee.svg (removing old pointer from line 107)
    target_08_knee = (
        "A machine learning engineer who reduces perception latency from $80\\text{ ms}$ to $40\\text{ ms}$ is not merely speeding up software; they are restoring $0.040 \\cdot \\omega_c$ radians of phase margin to the physical control plant.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "↰ **Prerequisite:** Transduction freshness and sensor measurement aging are derived in @sec-body-measurement-freshness.\n"
        ":::\n\n"
        "Resolving this physical dilemma requires transforming raw electrical transduction into grounded, verifiable spatial claims."
    )
    repl_08_knee = (
        "A machine learning engineer who reduces perception latency from $80\\text{ ms}$ to $40\\text{ ms}$ is not merely speeding up software; they are restoring $0.040 \\cdot \\omega_c$ radians of phase margin to the physical control plant.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_phase_margin_erosion_knee.svg){width=\"100%\" fig-alt=\"Knee curve showing phase margin eroding linearly with perception age until reaching instability at seventy-eight milliseconds.\"}\n\n"
        "*Past critical perception age, phase margin erodes to zero, triggering closed-loop instability.*\n"
        ":::\n\n"
        "Resolving this physical dilemma requires transforming raw electrical transduction into grounded, verifiable spatial claims."
    )
    replace_exact(ch08, target_08_knee, repl_08_knee, "Ch 08 phase margin erosion knee")

    # Fig 16: margin_dma_ingress_slack_envelope.svg
    target_08_dma = (
        "@Fig-perception-bandwidth-scaling traces two decades of sensor ingress, and the raw data rate rose by nearly two orders of magnitude, which is why a resolution increase justified on perception grounds must still be paid for out of the safety path's timing margin.\n\n"
        "The competition for memory bandwidth also creates a reverse failure mode that corrupts perception itself."
    )
    repl_08_dma = (
        "@Fig-perception-bandwidth-scaling traces two decades of sensor ingress, and the raw data rate rose by nearly two orders of magnitude, which is why a resolution increase justified on perception grounds must still be paid for out of the safety path's timing margin.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_dma_ingress_slack_envelope.svg){width=\"100%\" fig-alt=\"Budget envelope showing four-camera 4K DMA burst exhausting 850-microsecond timing slack and stalling the memory controller.\"}\n\n"
        "*Upgrading cameras from 1080p to 4K burns through safety timing slack, dropping control cycles.*\n"
        ":::\n\n"
        "The competition for memory bandwidth also creates a reverse failure mode that corrupts perception itself."
    )
    replace_exact(ch08, target_08_dma, repl_08_dma, "Ch 08 DMA ingress slack envelope")

    # =========================================================================
    # Ch 09: 09_memory.qmd
    # =========================================================================
    ch09 = VOL4_ROOT / "09_memory" / "09_memory.qmd"

    # Relocate spa_locator_sp.svg from line 200 up to after line 64
    old_locator_09 = (
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_sp.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the Sense-Plan belief edge.\"}\n\n"
        "*The Sense $\\cap$ Plan belief interface: preserving spatial object state under line-of-sight occlusions.*\n"
        ":::\n\n"
        "To understand how spatial memory fuses streaming depth frames into a persistent geometric model"
    )
    repl_remove_locator_09 = "To understand how spatial memory fuses streaming depth frames into a persistent geometric model"
    replace_exact(ch09, old_locator_09, repl_remove_locator_09, "Ch 09 remove old spa_locator_sp")

    target_09_reloc = (
        "The moment an unrefreshed belief breaches task safety margins, the nervous system revokes execution permission unilaterally.\n\n"
        "[^fn-mem-dual-brain]:"
    )
    repl_09_reloc = (
        "The moment an unrefreshed belief breaches task safety margins, the nervous system revokes execution permission unilaterally.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_sp.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the Sense-Plan belief edge.\"}\n\n"
        "*The Sense $\\cap$ Plan belief interface: preserving spatial object state under line-of-sight occlusions.*\n"
        ":::\n\n"
        "[^fn-mem-dual-brain]:"
    )
    replace_exact(ch09, target_09_reloc, repl_09_reloc, "Ch 09 relocate spa_locator_sp")

    # Fig 17: margin_active_negative_evidence_chain.svg
    target_09_ray = (
        "This active assertion of negative evidence along sensor sightlines is the exact computational mechanism that invalidates phantom obstacle tracks when dynamic objects leave a region, pruning ghost tracks without waiting for passive timeout decay.\n\n"
        "[^fn-math-voxel-raycast]:"
    )
    repl_09_ray = (
        "This active assertion of negative evidence along sensor sightlines is the exact computational mechanism that invalidates phantom obstacle tracks when dynamic objects leave a region, pruning ghost tracks without waiting for passive timeout decay.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_active_negative_evidence_chain.svg){width=\"100%\" fig-alt=\"Causal chain showing raycast through free space to surface hit, evicting ghost obstacles.\"}\n\n"
        "*Active negative evidence along sightlines clears phantom obstacles without waiting for timeout decay.*\n"
        ":::\n\n"
        "[^fn-math-voxel-raycast]:"
    )
    replace_exact(ch09, target_09_ray, repl_09_ray, "Ch 09 active negative evidence chain")

    # Fig 18: margin_volumetric_raycasting_bandwidth_ladder.svg
    target_09_vdb = (
        "To integrate incoming sensory point clouds while actively evicting stale obstacle geometry, the spatial memory engine executes the structured raycast update detailed in @algo-volumetric-tsdf-raycast.\n\n"
        "```pseudocode"
    )
    repl_09_vdb = (
        "To integrate incoming sensory point clouds while actively evicting stale obstacle geometry, the spatial memory engine executes the structured raycast update detailed in @algo-volumetric-tsdf-raycast.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_volumetric_raycasting_bandwidth_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder comparing SoC bus limit against dense TSDF and sparse VDB memory bandwidth.\"}\n\n"
        "*Dense TSDF raycasting consumes 86 percent of SoC bandwidth, while sparse VDB cuts bus traffic 1000-fold.*\n"
        ":::\n\n"
        "```pseudocode"
    )
    replace_exact(ch09, target_09_vdb, repl_09_vdb, "Ch 09 volumetric raycasting bandwidth ladder")

    # =========================================================================
    # Ch 10: 10_intent.qmd
    # =========================================================================
    ch10 = VOL4_ROOT / "10_intent" / "10_intent.qmd"

    # Fig 19: margin_operating_frequency_ladder.svg
    target_10_freq = (
        "A validity window sized only to accommodate slow inference would allow physical motion to outlast the evidence supporting its goal.\n\n"
        "::: {#fig-intent-frequency-gap"
    )
    repl_10_freq = (
        "A validity window sized only to accommodate slow inference would allow physical motion to outlast the evidence supporting its goal.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_operating_frequency_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder comparing operating frequencies from twenty kilohertz motor drives to half-hertz cognitive models.\"}\n\n"
        "*Operating frequencies span five orders of magnitude between inner motor drives and cognitive models.*\n"
        ":::\n\n"
        "::: {#fig-intent-frequency-gap"
    )
    replace_exact(ch10, target_10_freq, repl_10_freq, "Ch 10 operating frequency ladder")

    # Fig 20: margin_dynamic_reachability_shortfall_envelope.svg
    target_10_reach = (
        "Admitting an unachievable goal guarantees that the lease will expire mid-motion, forcing a sudden deceleration that wastes mechanical energy and leaves the arm stranded mid-path.\n\n"
        "::: {#chk-intent-trapezoidal-reachability-gate"
    )
    repl_10_reach = (
        "Admitting an unachievable goal guarantees that the lease will expire mid-motion, forcing a sudden deceleration that wastes mechanical energy and leaves the arm stranded mid-path.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_dynamic_reachability_shortfall_envelope.svg){width=\"100%\" fig-alt=\"Budget envelope showing 900-millisecond kinematic transit time overrunning 450-millisecond lease horizon.\"}\n\n"
        "*Kinematic transit time burns past offered lease horizons, triggering instant ingestion rejection.*\n"
        ":::\n\n"
        "::: {#chk-intent-trapezoidal-reachability-gate"
    )
    replace_exact(ch10, target_10_reach, repl_10_reach, "Ch 10 dynamic reachability shortfall envelope")

    # =========================================================================
    # Ch 11: 11_planning.qmd
    # =========================================================================
    ch11 = VOL4_ROOT / "11_planning" / "11_planning.qmd"

    # Fig 21: margin_action_chunk_latency_tail_strip.svg
    target_11_tail = (
        "To achieve an acceptable mission failure budget of $\\epsilon_{\\mathrm{mission}} = 10^{-3}$ across $3000$ handoffs, the allowable per-plan exhaustion probability simplifies to $\\epsilon \\approx \\epsilon_{\\mathrm{mission}} / K \\approx 3.3 \\times 10^{-7}$, requiring the system to size its horizon against the extreme tail quantile $Q_{1-\\epsilon}(L) \\approx P_{99.99997}$.\n\n"
        "Determining the length of each trajectory chunk requires scheduling when the replacement request is dispatched relative to the start of the current chunk."
    )
    repl_11_tail = (
        "To achieve an acceptable mission failure budget of $\\epsilon_{\\mathrm{mission}} = 10^{-3}$ across $3000$ handoffs, the allowable per-plan exhaustion probability simplifies to $\\epsilon \\approx \\epsilon_{\\mathrm{mission}} / K \\approx 3.3 \\times 10^{-7}$, requiring the system to size its horizon against the extreme tail quantile $Q_{1-\\epsilon}(L) \\approx P_{99.99997}$.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_action_chunk_latency_tail_strip.svg){width=\"100%\" fig-alt=\"Sequence strip showing chunk horizon replenishment across P99 and extreme P99.99997 latency tails.\"}\n\n"
        "*Sizing chunk horizons to P99.99997 tail latency prevents periodic starvation across thousands of handoffs.*\n"
        ":::\n\n"
        "Determining the length of each trajectory chunk requires scheduling when the replacement request is dispatched relative to the start of the current chunk."
    )
    replace_exact(ch11, target_11_tail, repl_11_tail, "Ch 11 action chunk latency tail strip")

    # Fig 22: margin_reflected_rotor_inertia_torque_knee.svg
    target_11_c2 = (
        "::: {#psp-planning-c2-continuity .callout-perspective title=\"The C² continuity invariant\"}\n"
        "A trajectory without continuous acceleration ($\\mathcal{C}^2$) is not a motion plan; unblended velocity steps across replanning seams are converted by driveline inertia into destructive torque spikes.\n"
        ":::\n\n"
        "A related physical failure arises from output chatter in learned policy proposals."
    )
    repl_11_c2 = (
        "::: {#psp-planning-c2-continuity .callout-perspective title=\"The C² continuity invariant\"}\n"
        "A trajectory without continuous acceleration ($\\mathcal{C}^2$) is not a motion plan; unblended velocity steps across replanning seams are converted by driveline inertia into destructive torque spikes.\n"
        ":::\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_reflected_rotor_inertia_torque_knee.svg){width=\"100%\" fig-alt=\"Knee curve showing 50 Newton-meter unblended torque shock collapsing to 1 Newton-meter with 50-millisecond spline blend.\"}\n\n"
        "*Unblended seam velocity steps provoke a 50-fold torque spike, destroying mechanical transmissions.*\n"
        ":::\n\n"
        "A related physical failure arises from output chatter in learned policy proposals."
    )
    replace_exact(ch11, target_11_c2, repl_11_c2, "Ch 11 reflected rotor inertia torque knee")

    # =========================================================================
    # Ch 12: 12_enforcement.qmd
    # =========================================================================
    ch12 = VOL4_ROOT / "12_enforcement" / "12_enforcement.qmd"

    # Fig 23: margin_quadratic_stopping_clearance_envelope.svg
    target_12_stop = (
        "While velocity increased by 100 percent ($2.0\\times$), the required stopping clearance expanded by 238 percent ($3.38\\times$), demonstrating how the quadratic scaling of kinetic braking energy ($E_k \\propto v^2$) dominates spatial clearance demands.\n\n"
        "Furthermore, if the robot approaches a blind warehouse rack aisle"
    )
    repl_12_stop = (
        "While velocity increased by 100 percent ($2.0\\times$), the required stopping clearance expanded by 238 percent ($3.38\\times$), demonstrating how the quadratic scaling of kinetic braking energy ($E_k \\propto v^2$) dominates spatial clearance demands.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_quadratic_stopping_clearance_envelope.svg){width=\"100%\" fig-alt=\"Budget envelope showing vehicle stopping clearance expanding from 388mm to 1312mm and exhausting a 1.2-meter blind corner budget.\"}\n\n"
        "*Doubling velocity quadruples braking distance, burning completely through the blind-corner clearance envelope.*\n"
        ":::\n\n"
        "Furthermore, if the robot approaches a blind warehouse rack aisle"
    )
    replace_exact(ch12, target_12_stop, repl_12_stop, "Ch 12 quadratic stopping clearance envelope")

    # Fig 24: margin_cbf_qp_cycle_budget_envelope.svg
    target_12_cbf = (
        "When physical saturation, conflicting obstacle boundaries, or numerical divergence render the admissible control set empty ($\\mathcal{U}_{\\text{safe}} = \\emptyset$), the mathematical premise of the projection fails, tripping the Simplex switch to escalate to the second rung.\n\n"
        "The second rung is an active position hold, corresponding to an IEC 60204-1 Category 2 stop (Safe Stop 2)"
    )
    repl_12_cbf = (
        "When physical saturation, conflicting obstacle boundaries, or numerical divergence render the admissible control set empty ($\\mathcal{U}_{\\text{safe}} = \\emptyset$), the mathematical premise of the projection fails, tripping the Simplex switch to escalate to the second rung.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_cbf_qp_cycle_budget_envelope.svg){width=\"100%\" fig-alt=\"Budget envelope showing active-set CBF-QP solve finishing in 175 microseconds within a 1000-microsecond control cycle.\"}\n\n"
        "*Embedded active-set CBF-QP solves finish in 175 microseconds, well within the 1000-microsecond control tick.*\n"
        ":::\n\n"
        "The second rung is an active position hold, corresponding to an IEC 60204-1 Category 2 stop (Safe Stop 2)"
    )
    replace_exact(ch12, target_12_cbf, repl_12_cbf, "Ch 12 CBF-QP cycle budget envelope")

    # =========================================================================
    # Ch 13: 13_placement.qmd
    # =========================================================================
    ch13 = VOL4_ROOT / "13_placement" / "13_placement.qmd"

    # Replace old locator at line 110 with margin_execution_determinism_ladder.svg
    target_13_ladder = (
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_core.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the central causal core.\"}\n\n"
        "*The silicon substrate: co-locating Sense, Plan, and Act across shared heterogeneous System-on-Chip crossbars.*\n"
        ":::"
    )
    repl_13_ladder = (
        "::: {.column-margin}\n"
        "![](images/svg/margin_execution_determinism_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder comparing execution jitter from fifty nanoseconds in lockstep microcontrollers to over twenty milliseconds in Linux.\"}\n\n"
        "*Execution jitter spans five orders of magnitude between lockstep microcontrollers and preemptible Linux.*\n"
        ":::"
    )
    replace_exact(ch13, target_13_ladder, repl_13_ladder, "Ch 13 execution determinism ladder (replacing old locator)")

    # Relocate spa_locator_core.svg to after line 204
    target_13_reloc = (
        "fig.tight_layout()\n"
        "```\n\n"
        ":::\n\n"
        "The latency experienced by the enforcement path depends directly on the longest uninterrupted burst permitted on the shared interconnect"
    )
    repl_13_reloc = (
        "fig.tight_layout()\n"
        "```\n\n"
        ":::\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_core.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the central causal core.\"}\n\n"
        "*The silicon substrate: co-locating Sense, Plan, and Act across shared heterogeneous System-on-Chip crossbars.*\n"
        ":::\n\n"
        "The latency experienced by the enforcement path depends directly on the longest uninterrupted burst permitted on the shared interconnect"
    )
    replace_exact(ch13, target_13_reloc, repl_13_reloc, "Ch 13 relocate spa_locator_core")

    # Fig 25: margin_memory_contention_tail_hockey_stick_knee.svg
    target_13_knee = (
        "Reporting average execution time or passing percentiles obscures the reality that one out of every thousand cycles violates the physical deadline and risks fluid overpressure.\n\n"
        "Completing millions of test cycles without a timing violation does not prove absolute independence."
    )
    repl_13_knee = (
        "Reporting average execution time or passing percentiles obscures the reality that one out of every thousand cycles violates the physical deadline and risks fluid overpressure.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_memory_contention_tail_hockey_stick_knee.svg){width=\"100%\" fig-alt=\"Hockey stick knee curve showing P99.9 tail latency breaching the 9.0-millisecond threshold under memory contention.\"}\n\n"
        "*Under memory bus contention, P99.9 tail latency breaches the 9.0-millisecond threshold, refuting independence.*\n"
        ":::\n\n"
        "Completing millions of test cycles without a timing violation does not prove absolute independence."
    )
    replace_exact(ch13, target_13_knee, repl_13_knee, "Ch 13 memory contention tail hockey stick knee")

    # =========================================================================
    # Ch 14: 14_intervention.qmd
    # =========================================================================
    ch14 = VOL4_ROOT / "14_intervention" / "14_intervention.qmd"

    # Move prerequisite pointer from line 85 down to line 97 and inject supervisory escalation ladder after line 83
    target_14_ptr_ladder = (
        "A system with one mechanism and not the other is incomplete in a way no amount of the other fixes.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "↰ **Prerequisite:** Actuator torque limits and back-EMF dynamics governing safe handovers originate in @sec-body-actuator-limits.\n"
        ":::\n\n"
        "When an operator attempts an intervention, human biology imposes a hard real-time latency budget before any physical corrective action can occur."
    )
    repl_14_ptr_ladder = (
        "A system with one mechanism and not the other is incomplete in a way no amount of the other fixes.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_supervisory_escalation_ladder.svg){width=\"100%\" fig-alt=\"Vertical staircase ladder showing four tiers of supervisory escalation from advisory haptic chime to galvanic cutoff.\"}\n\n"
        "*Authority escalates monotonically from advisory haptic feedback to galvanic torque cutoff.*\n"
        ":::\n\n"
        "When an operator attempts an intervention, human biology imposes a hard real-time latency budget before any physical corrective action can occur."
    )
    replace_exact(ch14, target_14_ptr_ladder, repl_14_ptr_ladder, "Ch 14 supervisory escalation ladder")

    target_14_ptr_new = (
        "Because the reaction distance scales linearly with velocity while the deceleration distance scales quadratically, human intervention becomes physically incapable of averting close-range collisions as operational speed increases.\n\n"
        "To see the direct physical consequences of @eq-intervention-reaction-floor"
    )
    repl_14_ptr_new = (
        "Because the reaction distance scales linearly with velocity while the deceleration distance scales quadratically, human intervention becomes physically incapable of averting close-range collisions as operational speed increases.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "↰ **Prerequisite:** Actuator torque limits and back-EMF dynamics governing safe handovers originate in @sec-body-actuator-limits.\n"
        ":::\n\n"
        "To see the direct physical consequences of @eq-intervention-reaction-floor"
    )
    replace_exact(ch14, target_14_ptr_new, repl_14_ptr_new, "Ch 14 relocate prerequisite pointer down")

    # Fig 26: margin_four_phase_handshake_strip.svg
    target_14_strip = (
        "In the final phase, the *Confirm* phase, the nervous system broadcasts a state confirmation packet across the network and triggers physical feedback, such as haptic vibration[^fn-bridge-haptic-authority] or optical indicators, to verify to the operator that the transfer of authority is complete.\n\n"
        "[^fn-bridge-haptic-authority]:"
    )
    repl_14_strip = (
        "In the final phase, the *Confirm* phase, the nervous system broadcasts a state confirmation packet across the network and triggers physical feedback, such as haptic vibration[^fn-bridge-haptic-authority] or optical indicators, to verify to the operator that the transfer of authority is complete.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_four_phase_handshake_strip.svg){width=\"100%\" fig-alt=\"Sequence strip showing four-phase handshake timing over a 31-millisecond total budget.\"}\n\n"
        "*Four-phase synchronous handshake guarantees mutual exclusion across a thirty-one millisecond deadline.*\n"
        ":::\n\n"
        "[^fn-bridge-haptic-authority]:"
    )
    replace_exact(ch14, target_14_strip, repl_14_strip, "Ch 14 four phase handshake strip")

    # Move spa_locator_core.svg from line 317 down to after line 331
    old_locator_14 = (
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_core.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the central causal core.\"}\n\n"
        "*Supervisory arbitration: dynamic authority transitions across the physical proposal-permission boundary.*\n"
        ":::\n\n"
        "When multiple control sources generate commands simultaneously"
    )
    repl_remove_locator_14 = "When multiple control sources generate commands simultaneously"
    replace_exact(ch14, old_locator_14, repl_remove_locator_14, "Ch 14 remove old spa_locator_core")

    target_14_reloc = (
        "The second half requires executing the switch across control boundaries without injecting discontinuous forces into the moving body, maintaining the kinodynamic feasibility established in @sec-planning-continuous-trajectories.\n\n"
        "::: {#chk-intervention-authority-transitions-and-timeout-bounds"
    )
    repl_14_reloc = (
        "The second half requires executing the switch across control boundaries without injecting discontinuous forces into the moving body, maintaining the kinodynamic feasibility established in @sec-planning-continuous-trajectories.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/spa_locator_core.svg){width=\"100%\" fig-alt=\"S·P·A locator triad highlighting the central causal core.\"}\n\n"
        "*Supervisory arbitration: dynamic authority transitions across the physical proposal-permission boundary.*\n"
        ":::\n\n"
        "::: {#chk-intervention-authority-transitions-and-timeout-bounds"
    )
    replace_exact(ch14, target_14_reloc, repl_14_reloc, "Ch 14 relocate spa_locator_core down")

    # =========================================================================
    # Ch 15: 15_verification.qmd
    # =========================================================================
    ch15 = VOL4_ROOT / "15_verification" / "15_verification.qmd"

    # Fig 27: margin_fault_injection_coverage_ladder.svg
    target_15_ladder = (
        "operating entirely outside the digital abstractions of the nervous system—a reality formalized in Leveson's **Systems-Theoretic Accident Model and Processes (STAMP)**\\index{Systems-Theoretic Accident Model and Processes!definition} framework[^fn-std-stpa-leveson] [@leveson2011engineering].\n\n"
        "[^fn-std-stpa-leveson]:"
    )
    repl_15_ladder = (
        "operating entirely outside the digital abstractions of the nervous system—a reality formalized in Leveson's **Systems-Theoretic Accident Model and Processes (STAMP)**\\index{Systems-Theoretic Accident Model and Processes!definition} framework[^fn-std-stpa-leveson] [@leveson2011engineering].\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_fault_injection_coverage_ladder.svg){width=\"100%\" fig-alt=\"Vertical ladder showing fault injection climbing from SRAM bit-flips up to physical dyno shaft jamming.\"}\n\n"
        "*Fault injection climbs from software bit-flips to physical dyno shaft jamming.*\n"
        ":::\n\n"
        "[^fn-std-stpa-leveson]:"
    )
    replace_exact(ch15, target_15_ladder, repl_15_ladder, "Ch 15 fault injection coverage ladder")

    # Fig 28: margin_poisson_exposure_wall_scale_anchor_knee.svg
    target_15_knee = (
        "Consequently, physical AI systems cannot be validated solely by accumulating millions of unperturbed operational fleet miles; safety assurance must instead be established through deterministic fault injection, layered runtime invariant enforcement, and formal architectural partitioning.\n\n"
        "| Target Safety Integrity Level & Standard"
    )
    repl_15_knee = (
        "Consequently, physical AI systems cannot be validated solely by accumulating millions of unperturbed operational fleet miles; safety assurance must instead be established through deterministic fault injection, layered runtime invariant enforcement, and formal architectural partitioning.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_poisson_exposure_wall_scale_anchor_knee.svg){width=\"100%\" fig-alt=\"Knee curve illustrating Butler and Finelli testing exposure wall scaling from hours to cosmic timescales.\"}\n\n"
        "*Certifying life-critical reliability through operational exposure hits an astronomical time barrier.*\n"
        ":::\n\n"
        "| Target Safety Integrity Level & Standard"
    )
    replace_exact(ch15, target_15_knee, repl_15_knee, "Ch 15 Poisson exposure wall scale anchor knee")

    # =========================================================================
    # Ch 16: 16_release.qmd
    # =========================================================================
    ch16 = VOL4_ROOT / "16_release" / "16_release.qmd"

    # Fig 29: margin_asil_decomposition_taxonomy.svg
    target_16_tax = (
        "The deterministic safety enforcer at Layer 2, executing on an independent, lockstep microcontroller or safety-rated field-programmable gate array (FPGA) at $1000\\text{ Hz}$, is certified to **ASIL D(D)**.\n\n"
        "Crucially, under ISO 26262 Part 9 Clause 5, an ASIL decomposition is valid **only if the decomposed elements achieve strict independence**."
    )
    repl_16_tax = (
        "The deterministic safety enforcer at Layer 2, executing on an independent, lockstep microcontroller or safety-rated field-programmable gate array (FPGA) at $1000\\text{ Hz}$, is certified to **ASIL D(D)**.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_asil_decomposition_taxonomy.svg){width=\"100%\" fig-alt=\"Taxonomy diagram showing ASIL decomposition isolating uncertified QM neural policy from certified ASIL D safety enforcer.\"}\n\n"
        "*ASIL decomposition isolates unverified neural planners behind a certified hardware safety enforcer.*\n"
        ":::\n\n"
        "Crucially, under ISO 26262 Part 9 Clause 5, an ASIL decomposition is valid **only if the decomposed elements achieve strict independence**."
    )
    replace_exact(ch16, target_16_tax, repl_16_tax, "Ch 16 ASIL decomposition taxonomy")

    # Fig 30: margin_cryptographic_seal_strip.svg
    target_16_seal = (
        "These additions transform the notebook from a retrospective log of development tests into an active operational contract governing physical execution.\n\n"
        "| Field Identifier           | Data / Representation Type"
    )
    repl_16_seal = (
        "These additions transform the notebook from a retrospective log of development tests into an active operational contract governing physical execution.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_cryptographic_seal_strip.svg){width=\"100%\" fig-alt=\"Sequence strip showing cryptographic seal progression through eFuse, Ed25519, SHA384, audit, and DC bus pre-charge.\"}\n\n"
        "*Hardware Root of Trust validates code hashes before commanding DC bus pre-charge.*\n"
        ":::\n\n"
        "| Field Identifier           | Data / Representation Type"
    )
    replace_exact(ch16, target_16_seal, repl_16_seal, "Ch 16 cryptographic seal strip")

    # =========================================================================
    # Ch 17: 17_frontier.qmd
    # =========================================================================
    ch17 = VOL4_ROOT / "17_frontier" / "17_frontier.qmd"

    # Fig 31: margin_observational_indistinguishability_chain.svg + relocate pointer
    target_17_chain = (
        "::: {.column-margin .margin-pointer}\n"
        "↰ **Prerequisite:** Failures undetectable by sensory telemetry challenge the observation contracts formulated in @sec-perception-observation-contract.\n"
        ":::\n\n"
        "The fundamental limit of runtime detection arises from observational indistinguishability. Consider two physical states of the world, $S_A$ and $S_B$. In state $S_A$, a commanded actuator trajectory $u(t)$ is safe. In state $S_B$, the identical trajectory $u(t)$ causes structural damage or human injury, requiring the nervous system to intervene and execute a stopping reflex. The dynamics of the body impose a hard deadline $t_{\\text{deadline}}$, determined by actuator response time and system momentum, after which physical contact or mechanical yield cannot be prevented. If the telemetry history $y(t)$ provided by onboard instrumentation is identical for $S_A$ and $S_B$ across the entire interval $t \\le t_{\\text{deadline}}$, no algorithm operating on those observations can reliably distinguish the benign state from the hazardous one. When two physical realities produce identical sensor traces up to the action deadline, the probability of safe intervention cannot exceed the unconditioned prior probability of the hazard, regardless of model scale or compute capacity.\n\n"
        "::: {#dfn-frontier-observational-indistinguishability"
    )
    repl_17_chain = (
        "The fundamental limit of runtime detection arises from observational indistinguishability. Consider two physical states of the world, $S_A$ and $S_B$. In state $S_A$, a commanded actuator trajectory $u(t)$ is safe. In state $S_B$, the identical trajectory $u(t)$ causes structural damage or human injury, requiring the nervous system to intervene and execute a stopping reflex. The dynamics of the body impose a hard deadline $t_{\\text{deadline}}$, determined by actuator response time and system momentum, after which physical contact or mechanical yield cannot be prevented. If the telemetry history $y(t)$ provided by onboard instrumentation is identical for $S_A$ and $S_B$ across the entire interval $t \\le t_{\\text{deadline}}$, no algorithm operating on those observations can reliably distinguish the benign state from the hazardous one. When two physical realities produce identical sensor traces up to the action deadline, the probability of safe intervention cannot exceed the unconditioned prior probability of the hazard, regardless of model scale or compute capacity.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_observational_indistinguishability_chain.svg){width=\"100%\" fig-alt=\"Causal chain showing state split producing identical telemetry until deadline passes and fracture occurs.\"}\n\n"
        "*Identical pre-contact telemetry prevents software intervention before irreversible physical fracture occurs.*\n"
        ":::\n\n"
        "::: {#dfn-frontier-observational-indistinguishability"
    )
    replace_exact(ch17, target_17_chain, repl_17_chain, "Ch 17 observational indistinguishability chain")

    target_17_ptr_new = (
        "yielding a total closed-loop reaction delay of $\\tau_{\\text{reaction}} = 15.0 + 2.0 + 1.0 + 24.0 = 42.0\\text{ ms}$.\n\n"
        "Because $\\tau_{\\text{reaction}} = 42.0\\text{ ms} > t_{\\text{harm}} = 37.5\\text{ ms}$"
    )
    repl_17_ptr_new = (
        "yielding a total closed-loop reaction delay of $\\tau_{\\text{reaction}} = 15.0 + 2.0 + 1.0 + 24.0 = 42.0\\text{ ms}$.\n\n"
        "::: {.column-margin .margin-pointer}\n"
        "↰ **Prerequisite:** Failures undetectable by sensory telemetry challenge the observation contracts formulated in @sec-perception-observation-contract.\n"
        ":::\n\n"
        "Because $\\tau_{\\text{reaction}} = 42.0\\text{ ms} > t_{\\text{harm}} = 37.5\\text{ ms}$"
    )
    replace_exact(ch17, target_17_ptr_new, repl_17_ptr_new, "Ch 17 relocate prerequisite pointer down")

    # Fig 32: margin_battery_diminishing_returns_knee.svg
    target_17_bat = (
        "Physical AI cannot scale purely by scaling parameters; it demands radical thermodynamic and architectural efficiency.\n\n"
        "### Neuromorphic sensing and physics-informed representations"
    )
    repl_17_bat = (
        "Physical AI cannot scale purely by scaling parameters; it demands radical thermodynamic and architectural efficiency.\n\n"
        "::: {.column-margin}\n"
        "![](images/svg/margin_battery_diminishing_returns_knee.svg){width=\"100%\" fig-alt=\"Scale anchor curve showing vehicle mission endurance hitting an asymptotic ceiling as battery mass increases.\"}\n\n"
        "*Added battery mass yields diminishing mission endurance as locomotion overhead consumes capacity.*\n"
        ":::\n\n"
        "### Neuromorphic sensing and physics-informed representations"
    )
    replace_exact(ch17, target_17_bat, repl_17_bat, "Ch 17 battery diminishing returns knee")

    print("\n🎉 All margin figures and locators successfully injected and relocated!")

if __name__ == "__main__":
    main()
