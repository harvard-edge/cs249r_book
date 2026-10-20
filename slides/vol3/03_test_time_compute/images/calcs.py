#!/usr/bin/env python3
"""Every number on the Volume III Lecture 3 slides and notes, read from MLSysIM.

MLSysIM holds the inputs (registry, Literature) and the calculations
(Scenarios.*, each backed by a solver in engine/solvers/deliberation.py); the
chapter reads the same scenarios. The one other source is the deck's own
measurement, pool_trace.txt, written by pool_reproducer.py (one run on a
laptop); this file reads it as recorded. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex.

Sources behind the scenarios
  Napkin Math 3.1   Scenarios.ProbeVsGeneratedExplanation
  Napkin Math 3.2   Scenarios.OutputCeilingThreeAllocations (Checkpoint 3.1: context_growth)
  Napkin Math 3.3   Scenarios.IndependentCandidatePools (and the mixed-workload pitfall)
  Exercise 2(d)     Scenarios.PassAtKSubsetEstimate
  Brown et al.      Scenarios.RepeatedSamplingCoverageVsVote
  Napkin Math 3.4   Scenarios.SparseValidPoolFirstPass and its two variants
  Equation 4        Scenarios.ArgmaxScoreNoise
  Section 3.6.2     Scenarios.StepCheckedSearchTree
  Napkin Math 3.5   Scenarios.ScopedPlanRevision
  Napkin Math 3.6   Scenarios.FlatVsBeamSearch
  Napkin Math 3.7   Scenarios.DeadlineBoundRepairPolicies
  Lecture 1 recall  Scenarios.UncheckedRepairHorizon
  Public runs       Scenarios.SweAgentClaude35SonnetRunCost (Literature.SweBenchTrajectories)
  Snell, PRM800K    Literature.TestTimeCompute

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
Q_ = mlsysim.core.units.Q_

HOOK = Scenarios.ProbeVsGeneratedExplanation
AXES = Scenarios.OutputCeilingThreeAllocations
POOL = Scenarios.IndependentCandidatePools
EST = Scenarios.PassAtKSubsetEstimate
COV = Scenarios.RepeatedSamplingCoverageVsVote
VER = Scenarios.SparseValidPoolFirstPass
VER_FA = Scenarios.SparseValidPoolHalvedFalseAccept
VER_V = Scenarios.SparseValidPoolDoubledValidRate
SEL = Scenarios.ArgmaxScoreNoise
TREE = Scenarios.StepCheckedSearchTree
REV = Scenarios.ScopedPlanRevision
BEAM = Scenarios.FlatVsBeamSearch
POL = Scenarios.DeadlineBoundRepairPolicies
HORIZON = Scenarios.UncheckedRepairHorizon
RUN = Scenarios.SweAgentClaude35SonnetRunCost
TTC = Literature.TestTimeCompute


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    return 100 * x


def per_million(price):
    """Dollars per million tokens."""
    return m(price, "USD/Mtoken")


print("== Prices and the decode floor (scenario assumptions)")
v("price_in", per_million(AXES.input_price), "{:.0f}", "input, $ per million tokens")
v("price_out", per_million(AXES.output_price), "{:.0f}", "output, $ per million tokens")
v("price_score", per_million(BEAM.scorer_price), "{:.2f}", "scorer input, $ per million tokens")
v("tok_ms", m(HOOK.result.token_time, "ms"), "{:.3f}", "decode floor per output token, ms")

print("== Hook (Napkin Math 3.1)")
h = HOOK.result
v("hook_tokens", HOOK.answer_tokens, "{:,.0f}", "explanation, output tokens")
v("hook_probe_ms", m(HOOK.probe_time, "ms"), "{:.0f}", "prepared test, ms")
v("hook_gen_s", m(h.answer_time, "s"), "{:.2f}", "explanation, s")
v("hook_ratio", h.time_ratio, "{:.1f}", "explanation / test")
v("hook_bill", m(h.answer_cost, "USD"), "{:.3f}", "explanation bill, $")

print("== Real reproducer run (pool_trace.txt, the deck's own measurement)")
runs = [blk.strip().splitlines() for blk in
        (Path(__file__).parent / "pool_trace.txt").read_text().strip().split("\n\n")]


def at(lines, words):
    """Time (ms) of the first trace event that contains ``words``."""
    return float(next(ln for ln in lines if words in ln).split()[0])


def blocked(lines):
    return float(next(ln for ln in lines if ln.startswith("B blocked")).split()[2])


def wall(lines):
    return float(next(ln for ln in lines if ln.startswith("probe")).split(",")[1].split()[0])


bug, t60, fix = runs
v("tr_return_bug", at(bug, "A returns"), "{:.1f}", "connection returned, ms")
v("tr_acquire_bug", at(bug, "B acquires"), "{:.1f}", "B acquires, ms")
v("tr_blocked_bug", blocked(bug), "{:.1f}", "B blocked after the return, ms")
v("tr_wall_bug", wall(bug), "{:.1f}", "probe wall clock, ms")
v("tr_blocked_t60", blocked(t60), "{:.1f}", "timeout patch: B blocked, ms")
v("tr_wall_t60", wall(t60), "{:.1f}", "timeout patch: probe, ms")
v("tr_blocked_fix", blocked(fix), "{:.1f}", "notify patch: B blocked, ms")
v("tr_wall_fix", wall(fix), "{:.1f}", "notify patch: probe, ms")

print("== Recall: horizon compounding (Lecture 1)")
v("rec_h", HORIZON.short_horizon, "{:.0f}", "unchecked turns")
v("rec_step_pct", pct(HORIZON.step_success), "{:.0f}", "per-turn success, %")
v("rec_success_pct", pct(HORIZON.result.success[HORIZON.short_horizon]), "{:.1f}", "all turns succeed, %")

print("== Three allocations (Napkin Math 3.2, Checkpoint 3.1)")
a = AXES.result
v("ax_prompt", AXES.prompt, "{:,.0f}", "prompt, tokens")
v("ax_ceiling", AXES.ceiling, "{:,.0f}", "output ceiling, tokens")
v("ax_n", AXES.candidates, "{:.0f}", "breadth streams")
v("ax_cand", AXES.candidate_tokens, "{:,.0f}", "tokens per candidate")
v("ax_rounds", AXES.rounds, "{:.0f}", "feedback rounds")
v("ax_rounds_word", AXES.rounds, "words", "feedback rounds, in prose")
v("ax_check_s", m(AXES.check_time, "s"), "{:.1f}", "check, s")
v("ax_depth_s", m(a.depth_time, "s"), "{:.1f}", "depth, s")
v("ax_breadth_s", m(a.breadth_time, "s"), "{:.1f}", "breadth, s")
v("ax_fb_s", m(a.feedback_time, "s"), "{:.1f}", "feedback, s")
v("ax_fb_s2", m(a.feedback_time, "s"), "{:.2f}", "feedback, s (two places)")
v("ax_depth_bill", m(a.depth_cost, "USD"), "{:.3f}", "depth bill, $")
v("ax_breadth_bill", m(a.breadth_cost, "USD"), "{:.3f}", "breadth bill, $")
v("ax_fb_bill", m(a.feedback_cost, "USD"), "{:.3f}", "feedback bill, $")
v("ax_fb_input", a.feedback_input, "{:,.0f}", "feedback input by the last round, tokens")
v("ax_bill_ratio", a.bill_ratio, "{:.1f}", "breadth bill / depth bill")
v("ax_extra_copies", AXES.candidates - 1, "{:.0f}", "extra prompt copies, N - 1")
v("ax_premium", a.breadth_premium, "{:,.0f}", "breadth premium, tokens")
v("ax_growth", AXES.context_growth, "{:.0f}", "repository context growth")
v("ax_grown_prompt", AXES.prompt * AXES.context_growth, "{:,.0f}", "grown prompt, tokens")
v("ax_grown_premium", a.grown_premium, "{:,.0f}", "premium at the grown prompt, tokens")
v("ax_grown_cost", m(a.grown_premium_cost, "USD"), "{:.2f}", "premium at the grown prompt, $")
v("ax_serial_s", m(a.serial_breadth_time, "s"), "{:.2f}", "breadth one stream at a time, s")

print("== Snell et al. 2024 (Literature.TestTimeCompute)")
v("snell_bins", TTC.SnellDifficultyBins, "{:.0f}", "difficulty bins")
v("snell_bins_word", TTC.SnellDifficultyBins, "words", "difficulty bins, in prose")
v("snell_samples", TTC.SnellDifficultySamples, "{:,.0f}", "samples per question for binning")
v("snell_beam", TTC.SnellBeamGenerations, "{:.0f}", "beam generations")
v("snell_bon", TTC.SnellBestOfNGenerations, "{:.0f}", "best-of-N generations matched")
v("snell_saving", TTC.SnellBestOfNGenerations / TTC.SnellBeamGenerations, "{:.0f}", "compute saving")
v("snell_ratio", TTC.SnellLargerModelRatio, "{:.0f}", "larger model beaten, parameter ratio")

print("== Pools (Napkin Math 3.3) and the mixed-workload pitfall")
p = POOL.result
n_max = POOL.pool_sizes[-1]
v("pool_p_pct", pct(POOL.pass_rate), "{:.0f}", "pass rate per candidate, %")
v("pool_n", n_max, "{:.0f}", "largest pool")
for k in POOL.pool_sizes[:-1]:
    v(f"pool_k{k}", pct(p.pass_at[k]), "{:.0f}", f"pass@{k}, %")
v(f"pool_k{n_max}", pct(p.pass_at[n_max]), "{:.1f}", f"pass@{n_max}, %")
v("pool_last_gain", pct(p.last_gain), "{:.1f}", "last candidate adds, points")
v("pool_par", p.parallel_per_success, "{:.1f}", "parallel candidates per accepted repair")
v("pool_seq_cand", p.sequential_candidates, "{:.2f}", "sequential candidates on average")
v("pool_seq", p.sequential_per_success, "{:.1f}", "sequential candidates per accepted repair")
v("pool_cost_ratio", p.cost_ratio, "{:.1f}", "parallel / sequential")
v("mix_easy_pct", pct(POOL.easy_pass_rate), "{:.0f}", "easy half pass rate, %")
v("mix_hard", POOL.hard_pass_rate, "{:g}", "hard half pass rate")
v("mix_hard_pct", pct(POOL.hard_pass_rate), "{:.0f}", "hard half pass rate, %")
v("mix_easy_fail", 1 - POOL.easy_pass_rate, "{:.1f}", "easy half failure rate")
v("mix_easy_pass", pct(p.easy_pass_at_max), "{:.1f}", "easy half pass@8, %")
v("mix_plug", pct(p.mixed_plug_in), "{:.1f}", "averaged rate, pass@8, %")
v("mix_true", pct(p.mixed_true), "{:.1f}", "per task then averaged, %")

print("== pass@k estimator (Exercise 2(d))")
e = EST.result
v("est_n", EST.samples, "{:.0f}", "samples")
v("est_c", EST.passing, "{:.0f}", "passing")
v("est_k", EST.k, "{:.0f}", "k")
v("est_rate", EST.passing / EST.samples, "{:.1f}", "observed rate")
v("est_subset", e.subset_estimate, "{:.3f}", "subset estimate")
v("est_plug", e.plug_in, "{:.3f}", "plug-in estimate")

print("== Coverage against majority vote (Brown et al. 2024)")
c = COV.result
v("cov_kmax", COV.k_max, "{:,.0f}", "largest pool")
v("cov_knee", COV.k_knee, "{:.0f}", "voting knee")
v("cov_repair_k", COV.repair_k, "{:.0f}", "repair pool")
v("cov_small_1", pct(c.coverage_small_first), "{:.0f}", "8B coverage at one sample, %")
v("cov_large_1", pct(c.coverage_large_first), "{:.0f}", "70B coverage at one sample, %")
v("cov_small", pct(c.coverage_small), "{:.0f}", "8B coverage at the largest pool, %")
v("cov_large", pct(c.coverage_large), "{:.0f}", "70B coverage at the largest pool, %")
v("vote_small", pct(c.vote_small), "{:.0f}", "8B majority vote, %")
v("vote_large", pct(c.vote_large), "{:.0f}", "70B majority vote, %")
v("vote_gain", pct(c.vote_gain_small), "{:.1f}", "8B vote gain after the knee, points")
v("cov_gain", pct(c.coverage_gain_small), "{:.0f}", "8B coverage gain after the knee, points")
v("repair_1", pct(c.repair_first), "{:.0f}", "repair coverage at one sample, %")
v("repair_k", pct(c.repair_max), "{:.0f}", "repair coverage at its pool, %")

print("== PRM800K annotated line (Literature.TestTimeCompute)")
v("prm_labels", TTC.Prm800kStepLabels, "{:,.0f}", "step labels, about")
v("prm_steps", TTC.Prm800kExampleSteps, "{:.0f}", "steps in the line")
v("prm_first_error", TTC.Prm800kExampleFirstErrorStep, "{:.0f}", "first wrong step")
v("prm_after", TTC.Prm800kExampleSteps - TTC.Prm800kExampleFirstErrorStep, "{:.0f}", "steps after it")
v("prm_answer", TTC.Prm800kExampleAnswer, "{:,.0f}", "answer reached")
v("prm_truth", TTC.Prm800kExampleGroundTruth, "{:,.0f}", "true answer")
v("prm_score", TTC.Prm800kExampleVerifierScore, "{:.3f}", "process scorer's score")
v("prm_minutes", m(Q_(float(TTC.Prm800kExampleLabelTimeMs), "ms"), "minute"), "{:.1f}", "labeling time, min")

print("== Verifier precision and one attempt's risk (Napkin Math 3.4)")
r = VER.result
v("vp_n", VER.pool_size, "{:.0f}", "candidates")
v("vp_tokens", m(VER.tokens_per_candidate, "token"), "{:,.0f}", "tokens per candidate")
v("vp_v_pct", pct(VER.valid_rate), "{:.0f}", "valid rate, %")
v("vp_fa_pct", pct(VER.false_accept_rate), "{:.0f}", "false acceptance, %")
v("vp_fa_half_pct", pct(VER_FA.false_accept_rate), "{:.0f}", "false acceptance halved, %")
v("vp_fr_pct", pct(VER.false_reject_rate), "{:.0f}", "false rejection, %")
v("vp_loss", m(VER.incident_loss, "USD"), "{:.0f}", "loss per defective deploy, $")
v("vp_q", r.pass_probability, "{:.3f}", "pass probability per candidate")
v("vp_valid_passes", r.expected_valid_passes, "{:.2f}", "expected valid passes")
v("vp_invalid_passes", r.expected_invalid_passes, "{:.3f}", "expected invalid passes")
v("vp_prec1", pct(r.precision), "{:.1f}", "precision, %")
v("vp_prec2", pct(r.precision), "{:.2f}", "precision, % (two places)")
v("vp_defect0", pct(r.defect_given_selection), "{:.0f}", "passes broken, % (whole)")
v("vp_defect1", pct(r.defect_given_selection), "{:.1f}", "passes broken, %")
v("vp_defect2", pct(r.defect_given_selection), "{:.2f}", "passes broken, % (two places)")
v("vp_defect_frac", r.defect_given_selection, "{:.4f}", "passes broken, fraction")
v("vp_any2", pct(r.any_selection), "{:.2f}", "pool selects, %")
v("vp_any_frac", r.any_selection, "{:.4f}", "pool selects, fraction")
v("vp_abstain2", pct(r.no_selection), "{:.2f}", "abstains, %")
v("vp_wrong2", pct(r.wrong_selection), "{:.2f}", "defective per attempt, %")
v("vp_task_loss", m(r.task_loss, "USD"), "{:.2f}", "expected loss per attempt, $")
v("vp_bill", m(r.output_bill, "USD"), "{:.2f}", "output bill, $")
v("vp_loss_ratio", r.loss_to_bill, "{:.0f}", "loss / bill")
v("vp_false_any", pct(r.any_false_pass), "{:.2f}", "batch holds a false pass, %")
v("vp_half_fa", pct(VER_FA.result.precision), "{:.1f}", "precision, false acceptance halved, %")
v("vp_double_v", pct(VER_V.result.precision), "{:.1f}", "precision, valid rate doubled, %")

print("== Selection pressure (equation 4)")
s = SEL.result
v("sel_n_small", SEL.small_pool, "{:.0f}", "small pool")
v("sel_n_large", SEL.large_pool, "{:.0f}", "large pool")
v("sel_shift_small", s.expected_max[SEL.small_pool], "{:.1f}", "winner's error, small pool, sigma")
v("sel_shift_large", s.expected_max[SEL.large_pool], "{:.1f}", "winner's error, large pool, sigma")
v("sel_gap_pct", pct(SEL.exploit_rate), "{:.1f}", "loophole rate per sample, %")
v("sel_gap_present", pct(s.exploit_presence), "{:.1f}", "loophole in the pool, %")

print("== Pruning (section 3.6.2)")
t = TREE.result
v("tree_b", TREE.branching, "{:.0f}", "children per node")
v("tree_depth", TREE.depth, "{:.0f}", "depth")
v("tree_r", TREE.rejection_rate, "{:.2f}", "rejection rate")
v("tree_r_pct", pct(TREE.rejection_rate), "{:.0f}", "rejection rate, %")
v("tree_keep", 1 - TREE.rejection_rate, "{:.2f}", "kept share")
v("tree_beff", t.effective_branching, "{:.1f}", "effective branching factor")
for d in range(1, TREE.depth + 1):
    v(f"tree_l{d}", TREE.branching ** d, "{:,.0f}", f"unpruned nodes at level {d}")
v("tree_unpruned", t.unpruned_nodes, "{:,.0f}", "unpruned nodes")
v("tree_flat", t.flat_cut_nodes, "{:.1f}", "flat cut reference, nodes")
v("tree_pruned", t.pruned_nodes, "{:.1f}", "pruned nodes")
v("tree_leaves", t.pruned_leaves, "{:.1f}", "pruned leaves")
v("tree_cut", pct(t.node_saving), "{:.1f}", "node saving, %")
v("tree_crit", pct(t.critical_rejection), "{:.0f}", "critical rejection, %")

print("== Scoped revision (Napkin Math 3.5)")
rv = REV.result
v("rv_nodes", REV.total_nodes, "{:.0f}", "plan nodes")
v("rv_per", REV.tokens_per_node, "{:.0f}", "tokens per node")
v("rv_bad", REV.replaced_nodes, "{:.0f}", "nodes replaced")
v("rv_scoped", rv.scoped_output, "{:,.0f}", "scoped output, tokens")
v("rv_full", rv.full_output, "{:,.0f}", "full output, tokens")
v("rv_avoided", pct(rv.avoided_share), "{:.0f}", "output avoided, %")

print("== Flat sampling against beam search (Napkin Math 3.6)")
b = BEAM.result
v("bm_flat_n", BEAM.flat_candidates, "{:.0f}", "flat candidates")
v("bm_segs", BEAM.segments, "{:.0f}", "segments")
v("bm_seg", BEAM.segment_tokens, "{:.0f}", "tokens per segment")
v("bm_width", BEAM.beam_width, "{:.0f}", "beam width")
v("bm_ext", BEAM.extensions, "{:.0f}", "extensions per retained prefix")
for i, n in enumerate(b.frontier_counts):
    v(f"bm_s{i + 1}", n, "{:.0f}", f"stage {i + 1} extensions")
v("bm_prompt", BEAM.prompt_tokens, "{:,.0f}", "task prompt, tokens")
v("bm_beam_out", b.beam_output_tokens, "{:,.0f}", "beam output, tokens")
v("bm_flat_out", b.flat_output_tokens, "{:,.0f}", "flat output, tokens")
v("bm_out_saved", pct(b.output_saving), "{:.2f}", "output saved, %")
v("bm_out_saved_usd", m(b.output_saving_cost, "USD"), "{:.3f}", "output saved, $")
v("bm_beam_in", b.beam_prompt_tokens, "{:,.0f}", "beam input, tokens")
v("bm_flat_in", b.flat_prompt_tokens, "{:,.0f}", "flat input, tokens")
v("bm_in_extra_usd", m(b.input_premium_cost, "USD"), "{:.3f}", "extra input, $")
v("bm_beam_bill", m(b.beam_token_cost, "USD"), "{:.3f}", "beam bill, $")
v("bm_flat_bill", m(b.flat_token_cost, "USD"), "{:.3f}", "flat bill, $")
v("bm_beam_s", m(b.beam_elapsed, "s"), "{:.1f}", "beam time, s")
v("bm_flat_s", m(b.flat_elapsed, "s"), "{:.1f}", "flat time, s")
v("bm_breakeven", pct(b.breakeven_input_share), "{:.1f}", "break-even input price, % of list")

print("== A public repair-agent run (SWE-bench Verified, 2024)")
w = RUN.result
v("swe_per_resolved", w.cost_per_resolved, "{:.2f}", "spend per resolved task, $")
v("swe_unresolved_pct", pct(w.unresolved_share), "{:.0f}", "spend on failed attempts, %")

print("== Three policies under one deadline (Napkin Math 3.7)")
po = POL.result
v("pol_prompt", POL.prompt, "{:,.0f}", "prompt, tokens")
v("pol_out", POL.output, "{:,.0f}", "candidate, tokens")
v("pol_check", m(POL.check_time, "s"), "{:.0f}", "test, s")
v("pol_deadline", m(POL.deadline, "s"), "{:.0f}", "deadline, s")
v("pol_n", POL.breadth, "{:.0f}", "parallel candidates")
v("pol_workers", POL.concurrency, "{:.0f}", "workers")
v("pol_diag", POL.diagnostic, "{:.0f}", "failure report, tokens")
v("pol_p_pct", pct(POL.initial_pass), "{:.0f}", "first candidate passes, %")
v("pol_q_pct", pct(POL.revision_pass), "{:.0f}", "revision passes, %")
v("pol_round", m(po.round_time, "s"), "{:.2f}", "one round, s")
v("pol_requested", m(po.requested_time, "s"), "{:.2f}", "requested rounds end at, s")
v("pol_rounds_word", po.rounds, "words", "rounds that fit, in prose")
v("pol_max_word", POL.max_rounds, "words", "requested rounds, in prose")
v("pol_a_acc", m(po.a_cost_per_pass, "USD"), "{:.3f}", "A per accepted repair, $")
v("pol_b_pass", pct(po.b_pass), "{:.2f}", "B passes, %")
v("pol_b_acc", m(po.b_cost_per_pass, "USD"), "{:.3f}", "B per accepted repair, $")
v("pol_b_over_a", po.b_over_a, "{:.1f}", "B / A per accepted repair")
v("pol_c_pass", pct(po.c_pass), "{:.0f}", "C passes, %")
v("pol_c_acc", m(po.c_cost_per_pass, "USD"), "{:.3f}", "C per accepted repair, $")
v("pol_c_s", m(po.c_time, "s"), "{:.2f}", "C expected time, s")

if __name__ == "__main__":
    write(__file__)
