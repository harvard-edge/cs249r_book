"""Named training scenarios for reinforcement.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st


# --- Reinforcement learning from verifiable rewards (added 2026-10-20) ---
VerifierFalseAcceptsAtTenPercentPass = _st.RewardCompositionScenario(
    name="One update of 8 rollouts on each of 1,024 tasks, scored by a verifier that accepts 2.5% of failures",
    description="Ten percent of rollouts truly pass; the verifier accepts every pass and wrongly accepts "
    "2.5 percent of failures. Expected counts, no false rejections.",
    tasks=1024,  # scenario assumption
    group=8,  # scenario assumption
    pass_rate=0.10,  # scenario assumption
    false_accept_rate=0.025,  # scenario assumption
    valid_accept_rate=1.0,  # scenario assumption: no false rejections
)


VerifierFalseAcceptsAtFivePercentPass = VerifierFalseAcceptsAtTenPercentPass.variant(
    name="The same verifier on a policy that passes 5 percent",
    description="Only the pass rate changes.",
    pass_rate=0.05,  # scenario assumption: a harder task mix
)


BinaryRewardFixedBaseline = _st.CenteredRewardScenario(
    name="Binary terminal reward centered on a fixed baseline of 0.25",
    description="Reward 1 for a pass and 0 for a failure.",
    baseline=0.25,  # scenario assumption
    passing_reward=1.0,
    failing_reward=0.0,
)


DebuggerTurnForkValue = _st.ForkValueScenario(
    name="Twenty continuations forked before and after a repair's debugger turn",
    description="One of 20 continuations passes from the bug report; 12 of 20 pass after the debugger turn.",
    continuations=20,  # scenario assumption
    passes_before=1,  # scenario assumption
    passes_after=12,  # scenario assumption
)


ThirtyTurnForksAtEveryBoundary = _st.ContinuationForkCostScenario(
    name="A 30-turn trajectory of 512 tokens per turn with 8 continuations forked at every boundary",
    description="Every one of the 29 intermediate boundaries reruns the remaining turns 8 times.",
    turns=30,  # scenario assumption
    tokens_per_turn=512,  # scenario assumption
    branches=8,  # scenario assumption
)


ThirtyTurnForksAfterTurnsTenAndTwenty = ThirtyTurnForksAtEveryBoundary.variant(
    name="The same trajectory with forks only after turns 10 and 20",
    description="Two chosen boundaries instead of all 29.",
    boundaries=[10, 20],  # scenario assumption
)


RepairGroupOnePassOfThree = _st.GroupCoefficientsScenario(
    name="Three repair rollouts, the middle one passing, under population dispersion",
    description="Rewards [0, 1, 0]; the dispersion divides by G and a 1e-8 stabilizer is added.",
    rewards=[0.0, 1.0, 0.0],  # scenario assumption
    stabilizer=1e-8,  # scenario assumption
)


LonePassInGroupOfSixteen = _st.GroupCoefficientsScenario(
    name="One pass among 16 rollouts of a hard task",
    description="Population dispersion with no stabilizer; the lone pass gets sqrt(15).",
    rewards=[1.0] + [0.0] * 15,  # scenario assumption
    stabilizer=0.0,
)


LoneFailureInGroupOfSixteen = LonePassInGroupOfSixteen.variant(
    name="One failure among 16 rollouts of an easy task",
    description="The mirror image of the lone pass.",
    rewards=[0.0] + [1.0] * 15,  # scenario assumption
)


ShortAndLongFailureLengths = _st.LengthNormalizationScenario(
    name="A 128-token and a 4,096-token failure with the same group coefficient",
    description="Per-rollout averaging divides each by its own length.",
    short_tokens=128,  # scenario assumption
    long_tokens=4_096,  # scenario assumption
)


HardTaskRolloutStep = _st.RolloutStepBudgetScenario(
    name="One update of 512 tasks with groups of 8 on a mix the policy passes 5 percent",
    description="Each rollout is the 30-turn, 512-token trajectory of the fork inventory.",
    tasks=512,  # scenario assumption
    group=8,  # scenario assumption
    turns=ThirtyTurnForksAtEveryBoundary.turns,
    tokens_per_turn=ThirtyTurnForksAtEveryBoundary.tokens_per_turn,
    pass_rate=0.05,  # scenario assumption: untargeted hard-task mix
)


HardTaskRolloutStepDoubledGroup = HardTaskRolloutStep.variant(
    name="The same update with groups of 16",
    description="Only the group size changes.",
    group=16,  # scenario assumption: double the group
)


TargetedTaskRolloutStep = HardTaskRolloutStep.variant(
    name="The same update on tasks the policy passes about half the time",
    description="Difficulty-targeted selection; only the pass rate changes.",
    pass_rate=0.50,  # scenario assumption
)


ModerateTaskRolloutStep = HardTaskRolloutStep.variant(
    name="The same update on a mix the policy passes 20 percent",
    description="Only the pass rate changes.",
    pass_rate=0.20,  # scenario assumption
)


PairGroupsAtTenPercentPass = HardTaskRolloutStep.variant(
    name="Groups of 2 rollouts on a task the policy passes 10 percent",
    description="The smallest group that can carry contrast.",
    group=2,  # scenario assumption
    pass_rate=0.10,  # scenario assumption
)


FlakyRaceTestAllPassRule = _st.RepeatedVerificationScenario(
    name="A timing-sensitive race test repeated three times on a candidate patch",
    description="An incorrect patch passes one trial with probability 0.20 and a correct one with 0.98; "
    "a group of 16 incorrect patches graded once each.",
    incorrect_pass=0.20,  # scenario assumption
    correct_pass=0.98,  # scenario assumption
    trials=3,  # scenario assumption
    group=16,  # scenario assumption
)


DominantTokenRatioClip = _st.RatioClipScenario(
    name="A token at probability 0.9 under a ratio clip of 0.2",
    description="The ceiling (1 + epsilon) p exceeds 1, so the clip does not bind.",
    epsilon=0.2,  # scenario assumption: PPO's common clip width
    probability=0.9,  # scenario assumption
)


RareTokenRatioClip = DominantTokenRatioClip.variant(
    name="A token at probability 0.01 under a ratio clip of 0.2",
    description="The token may rise only epsilon p in one update.",
    probability=0.01,  # scenario assumption
)


RepairRolloutSlotResets = _st.RolloutSlotCycleScenario(
    name="256 sandbox slots running 20-decision repair episodes, cold container reset against snapshot",
    description="120 ms of model service and 80 ms of tool execution per decision, 0.5 s of assessment, "
    "then a 3.5 s cold reset or a 15 ms snapshot restore. Constant times, no contention.",
    slots=256,  # scenario assumption
    decisions=20,  # scenario assumption
    model_time=Q_(120, "ms"),  # scenario assumption
    tool_time=Q_(80, "ms"),  # scenario assumption
    assessment_time=Q_(0.5, "s"),  # scenario assumption
    cold_reset=Q_(3.5, "s"),  # scenario assumption
    snapshot_reset=Q_(15, "ms"),  # scenario assumption
)


RolloutGroupOfFourBarrier = _st.GroupBarrierScenario(
    name="A group of four rollouts finishing between 0.2 s and 16.8 s",
    description="The group-relative update waits for the slowest member.",
    latencies=[Q_(0.2, "s"), Q_(2.1, "s"), Q_(4.2, "s"), Q_(16.8, "s")],  # scenario assumption
)


RepairVerifierPool = _st.VerifierPoolScenario(
    name="512 completed trajectories every 16 s, three 8 s verification trials each, 80 percent target",
    description="Stationary arrivals; every trial executes in this sizing case.",
    trajectories=512,  # scenario assumption
    interval=Q_(16, "s"),  # scenario assumption
    trials=3,  # scenario assumption
    trial_time=Q_(8, "s"),  # scenario assumption
    target_utilization=0.8,  # scenario assumption: illustrative planning target
)
