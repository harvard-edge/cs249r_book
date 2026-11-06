"""Named agents scenarios for deliberation.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ... import scenario_types as _st
from ..agents.model_calls import Llama3_70B_Int8DecodeH100


# --- Test-time deliberation (added 2026-10-20) ---------------------------
# One repair decision spent several ways with the weights fixed. Token
# counts, pass rates, check error rates, and prices are teaching
# assumptions; times use the Llama-3 70B 8-bit decode floor on one H100.
ProbeVsGeneratedExplanation = _st.ProbeVsAnswerScenario(
    name="A 1,024-token explanation against a prepared 40 ms test",
    description="Generation at the 8-bit decode floor and a hypothetical $15 per million "
    "output tokens, against a test that observes the running program.",
    floors=Llama3_70B_Int8DecodeH100,
    answer_tokens=1024,  # scenario assumption: explanation length, output tokens
    probe_time=Q_(40, "ms"),  # scenario assumption: an already-prepared diagnostic test
    output_price=Q_(15.00, "USD / Mtoken"),  # scenario assumption: hypothetical output price
)


OutputCeilingThreeAllocations = _st.AllocationAxesScenario(
    name="One 4,096-token output ceiling as depth, breadth, or feedback",
    description="A 4,096-token prompt; depth spends the ceiling in one chain; breadth runs four "
    "concurrent 1,024-token candidates, each billed its own prompt; feedback runs "
    "three 1,024-token rounds, each followed by a 1.5 s check returning 340 "
    "tokens; $3 and $15 per million input and output tokens, no cache discount.",
    floors=Llama3_70B_Int8DecodeH100,
    prompt=4096,  # scenario assumption: serialized initial input, tokens
    ceiling=4096,  # scenario assumption: common generated-token ceiling
    candidates=4,  # scenario assumption: breadth concurrency and candidate count
    candidate_tokens=1024,  # scenario assumption: generated tokens per short candidate
    rounds=3,  # scenario assumption: feedback rounds used
    report_tokens=340,  # scenario assumption: tokens appended per check report
    check_time=Q_(1.5, "second"),  # scenario assumption: per feedback check
    input_price=Q_(3.00, "USD / Mtoken"),  # scenario assumption: hypothetical input price
    output_price=ProbeVsGeneratedExplanation.output_price,
    context_growth=8,  # scenario assumption: repository context grows eightfold
)


IndependentCandidatePools = _st.BestOfNPoolScenario(
    name="Pools of 1, 2, 4, and 8 independent candidates at a 30 percent pass rate",
    description="IID candidates for one task, a selector that finds any passing member; and a "
    "mixed workload, half its tasks at 60 percent and half never passing.",
    pass_rate=0.30,  # scenario assumption: per-candidate pass probability for a fixed task
    pool_sizes=[1, 2, 4, 8],  # scenario assumption: pool sizes compared
    easy_pass_rate=0.60,  # scenario assumption: the easy half of a mixed workload
    hard_pass_rate=0.0,  # scenario assumption: the hard half never passes
)


SparseValidPoolFirstPass = _st.VerifiedPoolScenario(
    name="64 candidates at 5 percent valid, first passing one returned",
    description="A check that rejects 10 percent of valid and passes 2 percent of invalid "
    "candidates; 1,024 output tokens each at $15 per million; a defective deploy "
    "costs $500.",
    pool_size=64,  # scenario assumption
    valid_rate=0.05,  # scenario assumption: valid-candidate rate
    false_reject_rate=0.10,  # scenario assumption: epsilon_FR
    false_accept_rate=0.02,  # scenario assumption: epsilon_FA
    tokens_per_candidate=Q_(1024, "token"),  # scenario assumption
    output_price=ProbeVsGeneratedExplanation.output_price,
    incident_loss=Q_(500, "USD"),  # scenario assumption: loss per defective deploy
)


SparseValidPoolHalvedFalseAccept = SparseValidPoolFirstPass.variant(
    name="The same pool with the false-acceptance rate halved",
    false_accept_rate=SparseValidPoolFirstPass.false_accept_rate / 2,
)


SparseValidPoolDoubledValidRate = SparseValidPoolFirstPass.variant(
    name="The same pool with the valid-candidate rate doubled", valid_rate=2 * SparseValidPoolFirstPass.valid_rate
)


ArgmaxScoreNoise = _st.ArgmaxNoiseScenario(
    name="Argmax over equally good candidates scored with Gaussian noise",
    description="Expected winner's scoring error at 1 to 64 candidates, and a loophole in "
    "0.5 percent of single samples in a pool of 64.",
    pool_sizes=[1, 2, 4, 8, 16, 32, 64],  # scenario assumption: pool sizes plotted
    small_pool=4,  # scenario assumption
    large_pool=SparseValidPoolFirstPass.pool_size,
    exploit_rate=0.005,  # scenario assumption: share of samples exploiting a check gap
)


ScopedPlanRevision = _st.ScopedRevisionScenario(
    name="A 12-node candidate plan, 3 nodes invalidated",
    description="Equal 150-token nodes; regenerate only the refuted premise's descendants.",
    total_nodes=12,  # scenario assumption
    replaced_nodes=3,  # scenario assumption
    tokens_per_node=150,  # scenario assumption: generated tokens per node
)


StepCheckedSearchTree = _st.PrunedSearchTreeScenario(
    name="A depth-3 search tree, 5 children per node, step check rejecting 68 percent",
    description="Independent rejection of each child; and an unpruned tree of 4 alternatives over 10 steps for scale.",
    branching=5,  # scenario assumption
    rejection_rate=0.68,  # scenario assumption
    depth=3,  # scenario assumption
    wide_branching=4,  # scenario assumption: unpruned illustration
    wide_depth=10,  # scenario assumption: unpruned illustration
)


FlatVsBeamSearch = _st.SearchAccountsScenario(
    name="16 flat candidates against a width-4 beam with 3 extensions per prefix",
    description="Four segments of 128 tokens over a 4,096-token prompt; 8 concurrent streams "
    "and 8 test slots; 15 ms per full-prefix score and 0.5 s per final test; "
    "$3, $15, and $0.50 per million input, output, and scorer tokens.",
    floors=Llama3_70B_Int8DecodeH100,
    segments=4,  # scenario assumption
    segment_tokens=128,  # scenario assumption
    prompt_tokens=OutputCeilingThreeAllocations.prompt,
    flat_candidates=16,  # scenario assumption
    beam_width=4,  # scenario assumption
    extensions=3,  # scenario assumption
    generation_slots=8,  # scenario assumption: concurrent streams at an unchanged rate
    test_slots=8,  # scenario assumption: concurrent isolated test runs
    score_time=Q_(15, "ms"),  # scenario assumption
    test_time=Q_(0.5, "second"),  # scenario assumption
    input_price=OutputCeilingThreeAllocations.input_price,
    output_price=OutputCeilingThreeAllocations.output_price,
    scorer_price=Q_(0.50, "USD / Mtoken"),  # scenario assumption: scorer input price
)


DeadlineBoundRepairPolicies = _st.DeadlinePolicyScenario(
    name="One candidate, 8 in parallel, or feedback under a 75 s deadline",
    description="A 2,048-token prompt, 1,000-token candidates, 10 s checks; the first candidate "
    "passes 40 percent of the time and a revision fed a 340-token failure report "
    "passes 35 percent; at most three feedback rounds requested.",
    floors=Llama3_70B_Int8DecodeH100,
    prompt=2048,  # scenario assumption
    output=1000,  # scenario assumption
    diagnostic=OutputCeilingThreeAllocations.report_tokens,
    breadth=8,  # scenario assumption
    max_rounds=3,  # scenario assumption: requested feedback cap
    deadline=Q_(75, "second"),  # scenario assumption
    check_time=Q_(10, "second"),  # scenario assumption
    initial_pass=0.40,  # scenario assumption
    revision_pass=0.35,  # scenario assumption: conditional on a failed first attempt
    input_price=OutputCeilingThreeAllocations.input_price,
    output_price=OutputCeilingThreeAllocations.output_price,
    input_ceiling=32_768,  # scenario assumption
    output_ceiling=8_192,  # scenario assumption
    check_ceiling=8,  # scenario assumption
    concurrency=8,  # scenario assumption: generator and check capacity
)
