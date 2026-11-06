"""Named agent scenarios for execution reliability.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st


# --- Agent execution reliability ---------------------------------------
# One on-call remediation attempt priced on its workload profile, a lost
# reply settled by status rather than retried, and the closing arithmetic:
# residual violations, a passing set's invalid share, a shortened health
# window, blame search over a horizon, the specification boundary, a
# passing-patch audit, and METR's 50 against 80 percent horizons.
RemediationIncidentAttempt = _st.AcceptedTaskCostScenario(
    name="One on-call remediation attempt, priced on its workload profile",
    description="Model calls against the environment, health-check, and approver charges of the "
    "on-call remediation profile, at its acceptance; then twice the model calls.",
    model_cost=_Agents.Remediation.OnCall.model_cost_per_attempt,
    other_cost=(
        _Agents.Remediation.OnCall.environment_cost_per_attempt
        + _Agents.Remediation.OnCall.assessment_cost_per_attempt
        + _Agents.Remediation.OnCall.approval_cost_per_attempt
    ),
    acceptance=_Agents.Remediation.OnCall.acceptance_probability,
    model_cost_growth=1.0,  # scenario assumption: double the model calls for more diagnosis
)


BlindRetryAgainstStatusLookup = _st.BlindRetryScenario(
    name="A lost reply on 25,000 mutating calls a day: blind retry against a status lookup",
    description="Requests lost with probability 0.001 and replies with 0.004; a duplicated change "
    "costs $2,000 and a read-only status query $0.01.",
    request_loss=0.001,  # scenario assumption: hypothetical, not a measured rate
    reply_loss=0.004,  # scenario assumption: hypothetical, not a measured rate
    workers=1,  # one fleet counted as a whole
    calls_per_worker=25_000,  # scenario assumption: mutating calls per day
    duplicate_cost=Q_(2_000, "USD"),  # scenario assumption: cost of one duplicated change
    status_lookup_cost=Q_(0.01, "USD"),  # scenario assumption: cost of one status query
)


ResidualViolationsFiftyTurns = _st.ResidualViolationScenario(
    name="A 0.1 percent residual violation rate per step over 50-turn trajectories, 10,000 a day",
    description="Training lowers the rate of violating proposals without driving it to zero.",
    step_violation=0.001,  # scenario assumption: residual per-step rate after training
    horizon=50,  # scenario assumption: turns per trajectory
    trajectories_per_day=10_000,  # scenario assumption
)


ThirtyPercentValidPassingSet = _st.CheckerPosteriorScenario(
    name="A check over candidates of which 30 percent are valid",
    description="The check misses 10 percent of valid candidates and passes 5 percent of invalid ones.",
    defect_rate=0.70,  # scenario assumption: 30 percent of candidates valid
    false_accept_rate=0.05,  # scenario assumption: P(PASS | invalid)
    valid_accept_rate=0.90,  # scenario assumption: P(PASS | valid), 10 percent missed
)


HalvedHealthWindow = _st.ShortenedWindowScenario(
    name="A health window shortened to half its adopted length",
    description="Half the samples; the latency interval widens by the square root of two.",
    window_share=0.5,  # scenario assumption
)


CheckedWindowBlameSearch = _st.CounterfactualSearchScenario(
    name="Blame for a failure found only at the end of a 50-step task, or within one checked window",
    description="Four candidate actions per step; intermediate checks every five steps; 1.2 s per replayed check.",
    branching=4,  # scenario assumption: candidate actions per step
    horizon=50,  # scenario assumption: steps in the task
    window=5,  # scenario assumption: steps between intermediate checks
    replay_time=Q_(1.2, "s"),  # scenario assumption: time per replayed check
)
