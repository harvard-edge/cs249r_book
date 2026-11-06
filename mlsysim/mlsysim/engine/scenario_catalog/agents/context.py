"""Named agents scenarios for context.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st


# --- Context engineering (added 2026-10-20) ------------------------------
# The reference repair harness's request, budgeted by role; the layout,
# compaction, and invalidation scenarios reuse its sizes. Token counts are
# serialized-request stipulations, not measurements of any product.
RepairContextBudget = _st.ContextBudgetScenario(
    name="Repair-harness request budgeted by role, appending against selecting",
    description="The SWE-bench harness's window and step budget; 8,000 tokens reserved for the "
    "reply, 3,000 of instructions, 12,000 of tool declarations, 24,000 of task state; "
    "a 4,000-token tool result per turn appended, or 8,000 tokens of selected "
    "observations on every call; a 2,000-token digest.",
    window=int(_Agents.Coding.SWE_Bench_Runner.context_window.to("token").magnitude),
    output_reserve=8000,  # scenario assumption: K_max, tokens
    system_tokens=3000,  # scenario assumption: application instructions
    tool_tokens=12000,  # scenario assumption: tool declarations
    state_tokens=24000,  # scenario assumption: constraints, progress record, code excerpts
    observation_per_turn=4000,  # scenario assumption: tool result appended per turn
    selected_observation=8000,  # scenario assumption: observations a selecting policy keeps
    turns=_Agents.Coding.SWE_Bench_Runner.max_trajectory_steps,
    digest_tokens=2000,  # scenario assumption: a compacted per-turn result
)


ToyHeadAttentionDilution = _st.AttentionDilutionScenario(
    name="One attention head with fixed logits at 100 and 100,000 keys",
    description="Target logit 6, every other key at -2; only the number of keys changes.",
    target_logit=6.0,  # scenario assumption: toy scaled logit of the target key
    background_logit=-2.0,  # scenario assumption: toy scaled logit of every other key
    short_keys=100,  # scenario assumption: total keys, short context
    long_keys=100_000,  # scenario assumption: total keys, long context
)


ToyHeadLookalikeExcerpt = ToyHeadAttentionDilution.variant(
    name="The same head with a superseded look-alike excerpt among unrelated keys",
    description="40,000 unrelated keys at -2 and an 850-token stale excerpt whose keys share the "
    "evidence's identifiers, at logit 2.",
    unrelated_keys=40_000,  # scenario assumption
    lookalike_keys=850,  # scenario assumption: the superseded excerpt, one key per token
    lookalike_logit=2.0,  # scenario assumption
)


RepairPrefixLayoutBill = _st.PrefixLayoutBillScenario(
    name="Repair trajectory's input bill, volatile first against root, trunk, leaf",
    description="The selecting-policy request split into a root of instructions and tools, a "
    "trunk of task state, and a leaf of selected observations; two trunk rewrites; "
    "a hypothetical $3 per million input tokens, cached reads at 10 percent.",
    turns=RepairContextBudget.turns,
    root_tokens=RepairContextBudget.system_tokens + RepairContextBudget.tool_tokens,
    trunk_tokens=RepairContextBudget.state_tokens,
    leaf_tokens=RepairContextBudget.selected_observation,
    trunk_rewrites=2,  # scenario assumption: one per compaction pass (RepairCompactionWatermarks)
    input_price=Q_(3.00, "USD / Mtoken"),  # scenario assumption: hypothetical, not a provider schedule
    cached_fraction=0.10,  # scenario assumption: cached reads billed at this share
)


RepairCompactionWatermarks = _st.CompactionCadenceScenario(
    name="Repair request compacted between a 92,000-token trigger and the selecting request",
    description="Each 4,000-token result is appended; a pass resets the request to the "
    "selecting policy's size; an alternate 81,000-token target narrows the gap.",
    window=RepairContextBudget.window,
    output_reserve=RepairContextBudget.output_reserve,
    target=(
        RepairContextBudget.system_tokens
        + RepairContextBudget.tool_tokens
        + RepairContextBudget.state_tokens
        + RepairContextBudget.selected_observation
    ),
    trigger=92_000,  # scenario assumption: hypothetical high watermark
    tokens_per_turn=RepairContextBudget.observation_per_turn,
    turns=RepairContextBudget.turns,
    alternate_target=81_000,  # scenario assumption: a narrower gap, same trigger
)


SupersededExcerptInvalidation = _st.ContextInvalidationScenario(
    name="A superseded 850-token source excerpt opening the repair request's trunk",
    description="Remove it, replace it with a 24-token marker, or append a 128-token notice; "
    "the reused prefix is the root.",
    request_tokens=RepairPrefixLayoutBill.root_tokens
    + RepairPrefixLayoutBill.trunk_tokens
    + RepairPrefixLayoutBill.leaf_tokens,
    prefix_tokens=RepairPrefixLayoutBill.root_tokens,
    superseded_tokens=ToyHeadLookalikeExcerpt.lookalike_keys,
    marker_tokens=24,  # scenario assumption: in-place tombstone
    notice_tokens=128,  # scenario assumption: appended notice with role and message framing
)


CompactionConstraintRetention = _st.HorizonCompoundingScenario(
    name="One constraint carried through repeated compactions at 5 percent loss per round",
    description="Independent rounds; survival after 10 and 40 rounds and the first round below even odds.",
    step_success=0.95,  # scenario assumption: hypothetical per-round survival of one constraint
    short_horizon=10,  # scenario assumption: rounds
    long_horizon=40,  # scenario assumption: rounds
    target_success=0.5,  # even odds
)


ProtectedFieldCheckPayback = _st.ProtectedFieldCheckScenario(
    name="Twelve constraints over three compactions, with and without a protected-field check",
    description="3 percent loss per constraint per round; acceptance 0.6 when all survive; a $2 "
    "attempt; a check that catches 90 percent of losses for $0.05 per round.",
    constraints=12,  # scenario assumption
    loss_per_round=0.03,  # scenario assumption
    rounds=3,  # scenario assumption
    acceptance_if_intact=0.6,  # scenario assumption
    attempt_cost=Q_(2.00, "USD"),  # scenario assumption
    detection=0.90,  # scenario assumption: share of losses the check detects and repairs
    check_cost_per_round=Q_(0.05, "USD"),  # scenario assumption: one correction call per round
    survival_floor=0.5,  # even odds
    target_survival=0.95,  # scenario assumption: survival to hold without a check
)
