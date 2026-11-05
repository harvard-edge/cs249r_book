"""Named robotics scenarios for policy training.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....embodied.registry import Embodied as _Embodied
from ... import scenario_types as _st


# --- Policy training: discounting, decoders, trial campaigns, contact exploits (added 2026-10-20) ---
LatchOpeningDiscount = _st.DiscountHorizonScenario(
    name="The discounted return over one latch opening at the warehouse manipulator's setpoint step",
    description="Decisions per 8 s opening, the door-open reward's weight at the first decision, the "
    "effective horizon of a 0.99 discount, and the discount that keeps half the weight.",
    step=_Embodied.MobileManipulator.WarehouseMobileManipulator.chunk_step,
    rollout=Q_(8.0, "second"),  # scenario assumption: autonomous contact phase of one opening
    discount=0.99,  # scenario assumption: a common discount factor
    retained_weight=0.5,  # scenario assumption: keep half the terminal weight
)


LatchDecoderBudget = _st.IterativeDecoderBudgetScenario(
    name="A 16-step diffusion head and a 4-step flow head against the warehouse manipulator's ACT decoder",
    description="Per-evaluation cost, renewal slack, and the overrun under a jitter-sized stretch, "
    "on the 25 ms setpoint step and the 50 ms renewal period.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    denoising_steps=16,  # scenario assumption: DDIM denoising steps
    flow_steps=4,  # scenario assumption: flow-matching evaluations
    encode=Q_(8.0, "ms"),  # scenario assumption: visual encoding, once per proposal
    denoising_latency=Q_(48.0, "ms"),  # scenario assumption: stipulated total DDIM inference
    renewal_jitter=Q_(5.0, "ms"),  # scenario assumption: the lease rule's renewal-jitter allowance
)


LeggedGymQuadrupedRun = _st.ParallelSimRunScenario(
    name="4,096 simulated ANYmal quadrupeds trained with PPO on one GPU (Rudin et al. 2022)",
    description="Batch, total policy steps, lower-bound throughput, and simulated days of walking.",
    robots=4096,  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
    steps_per_update=24,  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
    updates=1500,  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
    training_time=Q_(20.0, "minute"),  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
    policy_rate=Q_(50.0, "Hz"),  # Rudin, N., Hoeller, D., Reist, P., and Hutter, M. (2022), Learning ...
)


LatchTrialCampaign = _st.TrialCampaignScenario(
    name="A from-scratch reinforcement learning campaign of one million latch openings",
    description="Execution, automated-reset, manual-reset, and 14-rig calendars; drivetrain load "
    "reversals; the trials a 2.5 h seeded campaign allows; and the same openings in a "
    "GPU-parallel simulator at a published throughput.",
    trials=1_000_000,  # scenario assumption: order of magnitude for learning from scratch
    rollout=LatchOpeningDiscount.rollout,
    decision_step=LatchOpeningDiscount.step,
    automated_reset=Q_(4.0, "second"),  # scenario assumption: dedicated RL re-latch fixture
    manual_reset=Q_(30.0, "second"),  # scenario assumption: manual operator reset
    farm_rigs=14,  # Levine, S., Pastor, P., Krizhevsky, A., Ibarz, J., and Quillen, D. ...
    reversals_per_trial=4,  # scenario assumption: joint load reversals per opening
    seeded_budget=Q_(2.5, "hour"),  # Luo, J., Xu, C., Wu, J., and Levine, S. (2025), Precise and Dextero...
    simulator=LeggedGymQuadrupedRun,
)


LatchContactDeadlines = _st.ContactDeadlineScenario(
    name="Constant-speed drives into the cage-door latch plate at the guarded and fast approaches",
    description="Contact yield deadline, tripwire crossing, brake onset, and the time left before the latch limit.",
    site=_Embodied.Scenario.WarehouseAisle,
)


LatchSoftSpringExploit = _st.SoftContactExploitScenario(
    name="A soft simulated strike plate against the real plate under an unbraked fast approach",
    description="Penalty-spring and rigid-arm peak forces and the arm stiffness at which the "
    "series chain reaches the latch limit.",
    machine=_Embodied.MobileManipulator.WarehouseMobileManipulator,
    site=_Embodied.Scenario.WarehouseAisle,
    penalty_penetration=Q_(2.4, "mm"),  # scenario assumption: penetration at the fast approach
)


LinearDriveRealityGap = _st.ForceRiseGapScenario(
    name="An illustrative linear drive regulating 10 N against a surface, in simulation and on hardware",
    description="One stated discrepancy per reality-gap channel: dynamics, timing, sensing, appearance.",
    approach_speed=Q_(0.20, "m/s"),  # scenario assumption
    target_force=Q_(10.0, "N"),  # scenario assumption
    simulated_rise_rate=Q_(500.0, "N/s"),  # scenario assumption
    physical_rise_rate=Q_(250.0, "N/s"),  # scenario assumption
    p99_delay=Q_(49.5, "ms"),  # scenario assumption: p99 end-to-end actuation delay
    simulated_delay=Q_(0.0, "ms"),  # scenario assumption: the simulator applies commands at once
    simulator_step=Q_(20.0, "ms"),  # scenario assumption: integration period
    drive_delay=Q_(8.0, "ms"),  # scenario assumption: unmodeled drive and bus delay
    overtravel=Q_(3.0, "mm"),  # scenario assumption: overtravel into the current limit
    range_bias=Q_(3.2, "mm"),  # scenario assumption: range-sensor bias
    dropout_frames=4,  # scenario assumption: specular dropout burst
    pixel_difference=0.055,  # scenario assumption: fraction of the image
    patch_shift=Q_(6.0, "mm"),  # scenario assumption: contact-patch estimate shift
    chamfer_tolerance=Q_(2.0, "mm"),  # scenario assumption
)


LatchResidualFineTune = _st.OnRobotFineTuneScenario(
    name="On-robot fine-tuning of a gated residual at the cage-door latch",
    description="Active contact time and above-tripwire samples of 20,000 interaction steps.",
    step=_Embodied.MobileManipulator.WarehouseMobileManipulator.chunk_step,
    interaction_steps=20_000,  # scenario assumption
    exceed_fraction=0.015,  # scenario assumption: exploratory steps past the tripwire
    plate_offset=Q_(1.5, "mm"),  # scenario assumption: real plate against its simulated placement
)
