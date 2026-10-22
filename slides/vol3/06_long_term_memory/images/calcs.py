#!/usr/bin/env python3
"""Every number on the Volume III Lecture 6 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the chapter's Napkin Math 6.1
to 6.5, Example 6.1, and Checkpoint 6.4 are Scenarios.* backed by the solvers
in engine/solvers/retrieval.py (the chapter reads the same scenarios); the
cited results are Literature entries (Moffatt v. Air Canada, SWE-bench,
SWE-agent, PoisonedRAG, Lost in the Middle); the DDR5-4800 module is
Hardware.Tech.Memory. The one other source is the deck's own measurement,
scan_bench.csv (written by scan_bench.py), read as recorded and fitted with
MLSysIM's bandwidth fit. This file only selects values, converts units, and
formats them; it types no numbers. Each value is recorded with v() under the
key the deck prints as \\val{key}; running this file writes images/numbers.tex.

Run from this folder: python3 calcs.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402

from mlsysim.core.units import BYTES_FP32, Q_  # noqa: E402
from mlsysim.physics import (  # noqa: E402
    calc_bandwidth_fit_through_origin, calc_interface_bandwidth, calc_vectors_in_budget)

Hardware, Literature, Scenarios = mlsysim.Hardware, mlsysim.Literature, mlsysim.Scenarios
HERE = Path(__file__).resolve().parent

CASE = Literature.IncidentReports
RETR = Literature.AgentRetrieval
SWEB = load_series("slides/vol3/06_long_term_memory/images/data/swe_bench_bm25_context.csv").entries
POISON = load_series("slides/vol3/06_long_term_memory/images/data/poisonedrag_black_box_palm2.csv").entries
LIU = load_series("slides/vol3/04_context_engineering/images/data/lost_in_the_middle_gpt35_turbo_20docs.csv").entries
PAY = Scenarios.RetrievalRecordsInAllowance
BM25 = Scenarios.Bm25ShortAgainstLongFile
DOCS = Scenarios.ConditionNotifyDocstrings
SCAN = Scenarios.ExactScanTenMillionVectors
ANN = Scenarios.AnnIndexesTenMillionVectors
TREE = Scenarios.TypedTraversalDepthThree
PREF = Scenarios.MeetingPreferenceRevision
MAINT = Scenarios.RepairEditIndexMaintenance
STORE = Scenarios.LexicalGraphAgainstDenseStore
COST = Scenarios.MaintainedMemoryPolicyCost
DIMM = Hardware.Tech.Memory.DDR5_4800

K, M = 10 ** 3, 10 ** 6


def mag(q, unit):
    return q.to(unit).magnitude


print("== Hook: Moffatt v. Air Canada (Literature.IncidentReports)")
v("mof_days", CASE.MoffattRefundWindowDays, "{:.0f}", "refund window the chatbot stated, days")
v("mof_damages", CASE.MoffattDamagesCad, "{:.2f}", "damages, CA$")
v("mof_interest", CASE.MoffattInterestCad, "{:.2f}", "pre-judgment interest, CA$")
v("mof_fees", CASE.MoffattFeesCad, "{:.0f}", "tribunal fees, CA$")
v("mof_total", CASE.MoffattAwardTotalCad, "{:.2f}", "total ordered, CA$ (para 44)")
assert f"{CASE.MoffattDamagesCad + CASE.MoffattInterestCad + CASE.MoffattFeesCad:.2f}" == \
    f"{CASE.MoffattAwardTotalCad:.2f}", "the three parts sum to the ordered total"
assert f"{CASE.MoffattFarePaidCad - CASE.MoffattBereavementFareCad:.2f}" == \
    f"{CASE.MoffattDamagesCad:.2f}", "damages are the fare difference (para 40)"

print("== Napkin Math 6.1 and Checkpoint 6.1 (Scenarios.RetrievalRecordsInAllowance)")
p = PAY.result
v("rp_window", PAY.window, "{:,.0f}", "call limit, tokens")
v("rp_reserve", PAY.output_reserve, "{:,.0f}", "output reserve, tokens")
v("rp_other", PAY.other_input, "{:,.0f}", "prompt, tools, history, tokens")
v("rp_envelope", PAY.envelope, "{:,.0f}", "envelope, tokens")
v("rp_body", PAY.body, "{:,.0f}", "body per record, tokens")
v("rp_meta", PAY.metadata, "{:,.0f}", "metadata per record, tokens")
v("rp_allowance", p.allowance, "{:,.0f}", "retrieval allowance, tokens")
v("rp_record", p.record, "{:,.0f}", "serialized record, tokens")
v("rp_records", p.records, "{:.0f}", "whole records that fit")
v("rp_used", p.used, "{:,.0f}", "staged payload, tokens")
v("rp_remaining", p.remaining, "{:,.0f}", "left over, tokens")
v("rp_body_only", p.body_only, "{:.0f}", "records a body-only count admits")
v("rp_body_only_used", p.body_only_used, "{:,.0f}", "those records serialized, tokens")
v("rp_overshoot", p.overshoot, "{:,.0f}", "over the allowance, tokens")
v("rp_meta2", p.scaled_metadata, "{:,.0f}", "metadata doubled, tokens")
v("rp_meta2_records", p.scaled_metadata_records, "{:.0f}", "whole records at doubled metadata")

print("== SWE-bench BM25 by context limit (slides/vol3/06_long_term_memory/images/data/swe_bench_bm25_context.csv)")
for i, row in enumerate(SWEB, start=1):
    v(f"swb_ctx{i}", row["max_context_k"] * K, "{:,.0f}", "maximum retrieved context, tokens")
    v(f"swb_rec{i}", row["bm25_recall_avg"], "{:.2f}", "BM25 recall of edited files, %")
    v(f"swb_res{i}", row["claude2_resolved"], "{:.2f}", "Claude 2 issues resolved, %")

print("== SWE-agent against pre-staging (Literature.AgentRetrieval)")
v("swa_rag", RETR.SweBenchLiteRagResolved, "{:.2f}", "Lite resolved, BM25 pre-staged, %")
v("swa_shell", RETR.SweBenchLiteShellResolved, "{:.2f}", "Lite resolved, shell commands, %")
v("swa_agent", RETR.SweBenchLiteSweAgentResolved, "{:.2f}", "Lite resolved, SWE-agent, %")
v("swa_ratio", RETR.SweBenchLiteSweAgentResolved / RETR.SweBenchLiteRagResolved, "{:.1f}",
  "SWE-agent / pre-staged")
v("swa_cost_lo", RETR.SweAgentCostMultipleLow, "{:.0f}", "cost per resolved issue, low multiple")
v("swa_cost_hi", RETR.SweAgentCostMultipleHigh, "{:.0f}", "cost per resolved issue, high multiple")

print("== Example 6.1 (Scenarios.Bm25ShortAgainstLongFile)")
v("bm_short", BM25.result.short_score, "{:.2f}", "short file, BM25 score")
v("bm_long", BM25.result.long_score, "{:.2f}", "long file, BM25 score")
v("bm_short_tf_words", BM25.short_frequency, "words", "matches in the short file")
v("bm_long_tf_words", BM25.long_frequency, "words", "matches in the long file")

print("== CPython Condition docstrings (Scenarios.ConditionNotifyDocstrings)")
v("doc_shared", DOCS.result.shared, "{:.0f}", "distinct words shared")
v("doc_union", DOCS.result.union, "{:.0f}", "distinct words in both")

print("== Napkin Math 6.2 (Scenarios.ExactScanTenMillionVectors)")
s = SCAN.result
v("sc_n_m", SCAN.vectors / M, "{:.0f}", "vectors, millions")
v("sc_dim", SCAN.dimension, "{:.0f}", "dimension")
v("sc_bytes", mag(SCAN.bytes_per_value, "byte"), "{:.0f}", "bytes per value (FP16)")
v("sc_bw", mag(SCAN.host.memory.bandwidth, "GB/s"), "{:.0f}", "host bandwidth, GB/s")
v("sc_peak", mag(SCAN.host.compute.peak_flops, "TFLOP/s"), "{:.0f}", "host peak, TFLOP/s")
v("sc_gb", mag(s.footprint, "GB"), "{:.2f}", "bytes per scan, GB")
v("sc_gflop", mag(s.flops, "GFLOP"), "{:.2f}", "work per scan, GFLOP")
v("sc_ms", mag(s.stream_time, "ms"), "{:.1f}", "bandwidth floor, ms")
v("sc_math_ms", mag(s.math_time, "ms"), "{:.2f}", "arithmetic alone, ms")
v("sc_intensity", mag(s.intensity, "flop/byte"), "{:.0f}", "intensity, FLOP per byte")
v("sc_ridge", mag(s.ridge, "flop/byte"), "{:.0f}", "host ridge, FLOP per byte")
v("sc_traj_s", mag(s.trajectory_time, "s"), "{:.1f}", "searches in one trajectory, s")
v("sc_budget_ms", mag(SCAN.budget, "ms"), "{:.0f}", "per-search budget, ms")
v("sc_budget_vec", s.budget_vectors, "{:,.0f}", "vectors the budget streams")
v("sc_budget_vec_k", round(s.budget_vectors, -3), "{:,.0f}", "the same, to the thousand")
v("sc_filtered", SCAN.filtered_vectors, "{:,.0f}", "vectors after a scope filter")
v("sc_filtered_ms", mag(s.filtered_time, "ms"), "{:.2f}", "their floor, ms")

print("== Measured scan on one laptop (scan_bench.csv, the deck's own run)")
with open(HERE / "scan_bench.csv") as f:
    reader = csv.reader(f)
    meta = next(reader)
    next(reader)
    rows = [(int(n), int(b), float(t)) for n, b, t in reader]
MACHINE = meta[1]
LAP_BYTES = mag(BYTES_FP32, "byte")
FLOP_PER_VALUE = mag(s.flops, "flop") / (SCAN.vectors * SCAN.dimension)  # one multiply-add
gflops = [FLOP_PER_VALUE * n * SCAN.dimension / t / K ** 3 for n, b, t in rows]
bw_fit = calc_bandwidth_fit_through_origin([b for _, b, _ in rows], [t for _, _, t in rows])
lap_budget = calc_vectors_in_budget(SCAN.budget, Q_(bw_fit, "byte/s"),
                                    SCAN.dimension, BYTES_FP32)
v("lap_bw", bw_fit / K ** 3, "{:.0f}", "fitted streamed bandwidth, GB/s")
v("lap_gflops_lo", min(gflops), "{:.0f}", "lowest arithmetic rate, GFLOP/s")
v("lap_gflops_hi", max(gflops), "{:.0f}", "highest arithmetic rate, GFLOP/s")
v("lap_budget_k", round(lap_budget, -3), "{:,.0f}", "FP32 vectors in the budget, to the thousand")
v("lap_bytes", LAP_BYTES, "{:.0f}", "bytes per FP32 value")
v("lap_top_m", rows[-1][0] / M, "{:.0f}", "largest run, million vectors")
v("lap_top_ms", rows[-1][2] * K, "{:.0f}", "largest run, ms")
v("ddr_rate", mag(DIMM.pin_data_rate, "Mbit/s"), "{:,.0f}", "DDR5-4800, million transfers per second")
v("ddr_bytes", mag(DIMM.interface_width, "byte"), "{:.0f}", "DIMM channel width, bytes")
v("ddr_bw", mag(calc_interface_bandwidth(DIMM.interface_width, DIMM.pin_data_rate), "GB/s"),
  "{:.1f}", "DDR5-4800 module peak, GB/s")

print("== Approximate-index bytes (Scenarios.AnnIndexesTenMillionVectors)")
a = ANN.result
v("ann_links", ANN.links_per_node, "{:.0f}", "links per node")
v("ann_link_bytes_words", mag(ANN.link_bytes, "byte"), "words", "bytes per link")
v("ann_graph_gb", mag(a.graph, "GB"), "{:.2f}", "graph links, GB")
v("ann_total_gb", mag(a.graph_total, "GB"), "{:.2f}", "vectors plus links, GB")
v("ann_sub", ANN.subvectors, "{:.0f}", "PQ subvectors (codes per vector)")
v("ann_code_bytes_words", mag(ANN.code_bytes, "byte"), "words", "bytes per code")
v("ann_codes_gb", mag(a.codes, "GB"), "{:.2f}", "PQ codes, GB")
v("ann_ratio", a.compression, "{:.0f}", "vectors / codes")
v("ann_lut_kb", mag(a.lookup_table, "KB"), "{:.1f}", "lookup table, KB")
v("ann_vec_bytes", mag(a.bytes_per_vector, "byte"), "{:,.0f}", "bytes per FP16 vector")

print("== Napkin Math 6.3 (Scenarios.TypedTraversalDepthThree)")
g = TREE.result
v("gn_allowance", g.allowance, "{:,.0f}", "retrieval allowance, tokens")
v("gn_depth", TREE.depth, "{:.0f}", "depth")
v("gn_shallow_depth", TREE.depth - 1, "{:.0f}", "one level less")
v("gn_children", TREE.children, "{:.0f}", "children per node")
v("gn_node", TREE.node_tokens, "{:.0f}", "tokens per node")
v("gn_env", TREE.envelope_tokens, "{:.0f}", "edges and envelope, tokens")
for d, (n, t) in enumerate(zip(g.nodes_by_depth, g.tokens_by_depth)):
    v(f"gn_nodes{d}", n, "{:.0f}", f"nodes through depth {d}")
    v(f"gn_tok{d}", t, "{:,.0f}", f"tokens through depth {d}")
    v(f"gn_level{d}", n - (g.nodes_by_depth[d - 1] if d else 0), "{:.0f}", f"nodes at depth {d}")
v("gn_fit", g.fit, "{:.0f}", "most whole nodes in the allowance")

print("== PoisonedRAG (slides/vol3/06_long_term_memory/images/data/poisonedrag_black_box_palm2.csv, AgentRetrieval)")
n_poison = RETR.PoisonedRagTextsPerQuestion
v("pr_n", n_poison, "{:.0f}", "planted texts per target question")
for i, row in enumerate(POISON, start=1):
    v(f"pr_kb{i}", row["knowledge_base_texts"] / M, "{:.1f}", f"{row['dataset']} texts, millions")
    v(f"pr_asr{i}", row["attack_success_rate"] * 100, "{:.0f}", f"{row['dataset']} attack success, %")
per_million = n_poison / POISON[0]["knowledge_base_texts"] * M
v("pr_per_m", per_million, "{:.2f}", "planted texts per million, smallest corpus")
v("pr_per_m_round", per_million, "{:.0f}", "the same, whole")

print("== Write path: the 10:00 and 11:00 directives (Scenarios.MeetingPreferenceRevision)")
v("pf_shared", PREF.result.shared, "{:.0f}", "words shared")
v("pf_words", PREF.result.first_words, "{:.0f}", "words in a directive")

print("== Napkin Math 6.4 (Scenarios.RepairEditIndexMaintenance)")
x = MAINT.result
v("ix_files", MAINT.files, "{:.0f}", "files changed")
v("ix_chunks", x.chunks, "{:.0f}", "chunks changed")
v("ix_cands", MAINT.candidates, "{:.0f}", "candidates checked per query")
v("ix_local", mag(x.local_time, "ms"), "{:.0f}", "re-index, local encoder, ms")
v("ix_remote", mag(x.remote_batch_time, "ms"), "{:.0f}", "re-index, one remote batch, ms")
v("ix_serial", mag(x.remote_per_file_time, "s"), "{:.2f}", "re-index, remote per file, s")
v("ix_write_us", mag(x.exclusion_write_time, "us"), "{:.1f}", "exclusion entries, us")
v("ix_check_us", mag(x.exclusion_check_time, "us"), "{:.1f}", "exclusion checks per query, us")
v("ix_ratio", x.write_path_ratio, "{:,.0f}", "local re-index / exclusion write")

print("== Napkin Math 6.5 (Scenarios.LexicalGraphAgainstDenseStore)")
v("st_ratio_a", STORE.result.ratio_lexical_graph, "{:.2f}", "lexical + graph, auxiliary / source")
v("st_ratio_b", STORE.result.ratio_lexical_dense, "{:.2f}", "lexical + dense, auxiliary / source")

print("== Checkpoint 6.4 (Scenarios.MaintainedMemoryPolicyCost)")
c = COST.result
v("pc_pa_pct", COST.baseline_acceptance * 100, "{:.0f}", "policy A acceptance, %")
v("pc_pb_pct", COST.maintained_acceptance * 100, "{:.0f}", "policy B acceptance, %")
v("pc_pa", COST.baseline_acceptance, "{:.1f}", "policy A acceptance")
v("pc_pb", COST.maintained_acceptance, "{:.1f}", "policy B acceptance")
v("pc_maint", COST.maintenance_share, "{:.1f}", "maintenance, share of A's attempt")
v("pc_attempt_b", 1 + COST.maintenance_share, "{:.1f}", "policy B attempt, units of A's")
v("pc_ceff_a", c.baseline_cost, "{:.2f}", "policy A cost per accepted task")
v("pc_ceff_b", c.maintained_cost, "{:.2f}", "policy B cost per accepted task")
v("pc_change", -c.change * 100, "{:.1f}", "policy B lower by, %")
v("pc_breakeven", c.break_even_acceptance, "{:.2f}", "break-even acceptance")

print("== Lost in the middle (slides/vol3/04_context_engineering/images/data/lost_in_the_middle_gpt35_turbo_20docs.csv)")
v("liu_docs", LIU[0]["documents"], "{:.0f}", "documents per question")

if __name__ == "__main__":
    write(__file__)
