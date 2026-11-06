"""Named agents scenarios for security.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st
from ..agents.agents import RepairAttemptCostLevers
from ..robotics.demonstrations import InvariantPrefixRebuildDgxH100


# --- Agent security: injection exposure and defense evaluation (added 2026-10-20) ---
InjectionExposureByReadVolume = _st.InjectionExposureScenario(
    name="Injection exposure of four agent workloads at a 1 percent payload rate",
    description="Per-task chance of reading at least one injected payload for repair (20 items), "
    "research, support, and remediation (their workload profiles), unscreened and "
    "behind a screen that misses 5 percent of payloads; residual research exposure "
    "per 1,000 tasks.",
    payload_fraction=0.01,  # scenario assumption: share of retrievable items carrying a payload
    reads={
        "repair": 20,  # scenario assumption: READMEs, issues, and search results per repair task
        "research": _Agents.Research.OrchestratorWorker.untrusted_reads_per_task,
        "support": _Agents.Support.CustomerOps.untrusted_reads_per_task,
        "remediation": _Agents.Remediation.OnCall.untrusted_reads_per_task,
    },
    miss_rate=0.05,  # scenario assumption: share of payloads a screening classifier misses
    cohort=1000,  # scenario assumption: research tasks in the per-cohort count
)


FixtureReleasePolicyComparison = _st.PolicyComparisonScenario(
    name="Name-and-destination checks against labels plus a release check, 100 attacked and "
    "100 benign repair episodes each",
    description="Hypothetical outcome counts for two policies on equal, separate cohorts; attempts "
    "priced at the repair attempt cost.",
    attacked=100,  # scenario assumption
    benign=100,  # scenario assumption
    baseline={
        "confirmed": 20,
        "unresolved": 5,
        "accepted_attacked": 65,  # scenario assumption
        "accepted_benign": 90,
        "false_block": 5,
    },
    candidate={
        "confirmed": 5,
        "unresolved": 10,
        "accepted_attacked": 55,  # scenario assumption
        "accepted_benign": 80,
        "false_block": 15,
    },
    attempt_cost=RepairAttemptCostLevers.result.attempt_cost,
)


CleanAttackSuite50 = _st.ZeroFailureScenario(
    name="50 attacked episodes with no attack success",
    description="The attack-success rate a clean run supports at 95 percent one-sided confidence, "
    "and the clean episodes a 1 percent tolerance needs.",
    trials=50,  # scenario assumption
    confidence=0.95,
    target_rate=0.01,  # scenario assumption: tolerable attack-success rate
)


CleanAttackSuite100 = CleanAttackSuite50.variant(
    name="100 attacked episodes with no attack success",
    trials=100,  # scenario assumption
)


PrefixCacheProbe2K = InvariantPrefixRebuildDgxH100.variant(
    name="A 2,048-token probed prefix: prefill on a miss against a weight-read pass on a hit",
    prefix_tokens=2_048,  # scenario assumption: probed system-prompt prefix
)
