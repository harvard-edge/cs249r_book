"""Named agents scenarios for agents.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st


# --- Agent trajectories (added 2026-10-20) ------------------------------
UncheckedRepairHorizon = _st.HorizonCompoundingScenario(
    name="Unchecked repair trajectory at 99 percent per step",
    description="Independent steps, no checks, no recovery: success at 10, 100, and 200 turns, "
    "and the step error that holds 90 percent at 100 turns.",
    step_success=0.99,  # scenario assumption: hypothetical per-step accuracy
    short_horizon=10,  # scenario assumption: turns
    long_horizon=100,  # scenario assumption: turns
    target_success=0.90,  # scenario assumption: success to hold at the long horizon
    extended_horizon=200,  # twice the long horizon: success squares
)


HorizonCompounding30Steps = UncheckedRepairHorizon.variant(
    name="Unchecked repair trajectory at the coding-agent step budget",
    description="The same 99 percent per step, priced at the harness's trajectory step budget.",
    short_horizon=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
)


RepairAttemptCostLevers = _st.AcceptedTaskCostScenario(
    name="Repair attempt: cheaper tokens or more acceptances",
    description="$1.30 of model tokens and $0.80 of sandbox, tests, and review per attempt, "
    "accepted 40 percent of the time; halve the token price, or raise acceptance "
    "to 60 percent; halve the $0.80, or add 10 points of acceptance; five "
    "independent attempts.",
    model_cost=Q_(1.30, "USD"),  # scenario assumption: tokens per attempt
    other_cost=Q_(0.80, "USD"),  # scenario assumption: sandbox, tests, and review per attempt
    acceptance=0.40,  # scenario assumption
    token_price_cut=0.5,  # scenario assumption: a tariff at half the token price
    improved_acceptance=0.60,  # scenario assumption: after a context or tool improvement
    other_cost_cut=0.5,  # scenario assumption: halve the sandbox, test, and review cost
    acceptance_gain=0.10,  # scenario assumption: a tool change adds 10 points
    independent_attempts=5,  # scenario assumption: retry budget
)


AppendOnlyRepairContext = _st.AppendOnlyContextScenario(
    name="Append-only repair transcript over the reference harness's turn budget",
    description="An 8,000-token first prompt plus 1,500 tokens per turn, resent every turn, "
    "over the SWE-bench harness's step budget and window.",
    initial_tokens=Q_(8000, "token"),  # scenario assumption: first serialized prompt
    tokens_per_turn=Q_(1500, "token"),  # scenario assumption: model output plus observation
    turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
    context_window=_Agents.Coding.SWE_Bench_Runner.context_window,
)
