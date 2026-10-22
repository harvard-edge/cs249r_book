#!/usr/bin/env python3
"""Every number on the Volume III Lecture 7 slides and notes, read from MLSysIM.

MLSysIM holds the inputs and the calculations: the chapter's Napkin Math 7.1
to 7.4 and Exercise 9 are Scenarios.* backed by the solvers in
engine/solvers/tool_calls.py (the chapter reads the same scenarios); the
adapter count is Scenarios.ThreeHostsFourToolServices; the 500 public repair
runs are Scenarios.MiniSweAgentSonnet4Run over Literature.SweBenchMiniSonnet4Run,
with the harness's staging policy; SWE-agent's results are
Literature.AgentRetrieval. The other sources are the deck's own measurements
in data/, read as recorded: the retry harness (retry_results.json), the
GitHub MCP descriptors (github_mcp_tools.json), the per-command outputs of
the 500 runs (swebench_observations.csv), the pytest log (pytest_log.txt),
the pipe capacity (pipecap.txt), and the orphaned workers (orphans.txt).
This file only selects values, converts units, and formats them; it types no
numbers. Each value is recorded with v() under the key the deck prints as
\\val{key}; running this file writes images/numbers.tex. make_figs.py imports
the data below so figures and slides agree.

Run from this folder: python3 calcs.py
"""

import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402

from mlsysim.core.units import Q_  # noqa: E402
from mlsysim.physics import (  # noqa: E402
    calc_blind_retry_duplicate_probability, calc_timeout_probability)

Literature, Scenarios = mlsysim.Literature, mlsysim.Scenarios
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

RETRY = Scenarios.BlindRetryTwoPercentLoss
LOG = Scenarios.TwentyFourMegabyteLogObservation
POLL = Scenarios.ModelPollingEveryFiveSeconds
POLL_SWEEP = [POLL, Scenarios.ModelPollingEveryTenSeconds, Scenarios.ModelPollingEveryTwentySeconds,
              Scenarios.ModelPollingEveryThirtySeconds, Scenarios.ModelPollingEverySixtySeconds]
RUNTIME_POLL = Scenarios.ModelPollingEveryTenSeconds  # runtime polling at this interval costs no tokens
CATALOG = Scenarios.FiftyToolCatalogSixSelected
TOOLKIT = Scenarios.ShellAgainstTwelveStructuredTools
ADAPTERS = Scenarios.ThreeHostsFourToolServices
RUN = Scenarios.MiniSweAgentSonnet4Run
SWE = Literature.AgentRetrieval

WINDOW = LOG.window
STAGE_CAP = RUN.truncation_chars


def usd(q):
    return q.to("USD").magnitude


def per_million(q):
    return q.to("USD / Mtoken").magnitude


def recorded(path, pattern):
    """The integer a measurement file records after ``pattern``."""
    return int(re.search(pattern + r"\D*(\d+)", (DATA / path).read_text()).group(1))


# The deck's measurements, read as recorded (make_figs.py plots these).
HARNESS = json.loads((DATA / "retry_results.json").read_text())
for r in HARNESS:
    r["predicted"] = 0.0 if r["dedup"] else calc_blind_retry_duplicate_probability(r["p_req"], r["p_rep"]) * r["n"]
    r["pred_timeouts"] = calc_timeout_probability(r["p_req"], r["p_rep"]) * r["n"]
GH_ROWS = json.loads((DATA / "github_mcp_tools.json").read_text())
OBS_ROWS = list(csv.DictReader(open(DATA / "swebench_observations.csv")))
OBS_RAW = [int(r["raw_chars"]) for r in OBS_ROWS]
OBS_CUT = sum(1 for r in OBS_ROWS if r["truncated"] == "1")

print("== Adapters, N x M against N + M (Scenarios.ThreeHostsFourToolServices)")
v("ad_hosts", ADAPTERS.hosts, "{:.0f}", "agent hosts")
v("ad_services", ADAPTERS.services, "{:.0f}", "tool services")
v("ad_pair", ADAPTERS.result.pairwise, "{:.0f}", "pairwise adapters")
v("ad_shared", ADAPTERS.result.shared, "{:.0f}", "adapters through one protocol")

print("== GitHub MCP server descriptors (data/github_mcp_tools.json, measured)")
tok = [r["tokens"] for r in GH_ROWS]
biggest = max(GH_ROWS, key=lambda r: r["tokens"])
assert biggest["name"] == "projects_write", "the slide names the largest descriptor"
v("gh_n", len(GH_ROWS), "{:.0f}", "tools listed")
v("gh_mut", sum(not r["read_only"] for r in GH_ROWS), "{:.0f}", "mutating tools")
v("gh_repo", sum(r["repo_free_string"] for r in GH_ROWS), "{:.0f}", "tools taking the repository as any string")
v("gh_destr", sum(r["destructive"] for r in GH_ROWS), "{:.0f}", "declared destructive")
v("gh_idem", sum(r["idempotent"] for r in GH_ROWS), "{:.0f}", "declared idempotent")
v("gh_total", sum(tok), "{:,.0f}", "descriptor tokens, whole catalog")
v("gh_median", statistics.median_low(tok), "{:,.0f}", "median descriptor, tokens")
v("gh_max", biggest["tokens"], "{:,.0f}", "largest descriptor, tokens")
v("gh_share", sum(tok) / WINDOW * 100, "{:.1f}", "catalog share of the window, %")
v("win", WINDOW, "{:,.0f}", "context window, tokens (repair harness)")

print("== Napkin Math 7.1 (Scenarios.BlindRetryTwoPercentLoss)")
b = RETRY.result
assert RETRY.request_loss == RETRY.reply_loss, "the slide states one loss rate for both directions"
v("rt_loss_pct", RETRY.reply_loss * 100, "{:.0f}", "request and reply loss, %")
v("rt_loss", RETRY.reply_loss, "{:.2f}", "loss probability")
v("rt_keep", 1 - RETRY.request_loss, "{:.2f}", "delivery probability")
v("rt_workers", RETRY.workers, "{:.0f}", "workers")
v("rt_per_worker", RETRY.calls_per_worker, "{:,.0f}", "mutating calls per worker per day")
v("rt_calls", b.calls, "{:,.0f}", "mutating calls per day")
v("rt_timeout_pct", b.timeout_probability * 100, "{:.2f}", "timeout probability, %")
v("rt_dup_pct", b.duplicate_probability * 100, "{:.2f}", "duplicate probability, %")
v("rt_dup_pct4", b.duplicate_probability * 100, "{:.4f}", "duplicate probability, %, four places")
v("rt_dups", b.duplicates, "{:,.0f}", "duplicate effects per day")

print("== Retry harness (data/retry_results.json, measured)")
n_ops = {r["n"] for r in HARNESS}
assert len(n_ops) == 1, "every setting ran the same number of operations"
v("hx_n", n_ops.pop(), "{:,.0f}", "operations per setting")
req = {r["p_req"] for r in HARNESS}
assert len(req) == 1, "request loss is held fixed"
v("hx_req", req.pop() * 100, "{:.0f}", "request loss, %")
for r in HARNESS:
    pct = round(r["p_rep"] * 100)
    if r["dedup"]:
        v("hx_dedup", r["duplicates"], "{:.0f}", f"duplicates, {pct}% reply loss, deduplicated")
        continue
    v(f"hx_rep{pct}", r["p_rep"] * 100, "{:.0f}", "reply loss, %")
    v(f"hx_dup{pct}", r["duplicates"], "{:.0f}", f"measured duplicates, {pct}% reply loss")
    v(f"hx_pred{pct}", r["predicted"], "{:.0f}", f"predicted duplicates, {pct}% reply loss")

print("== Real pytest log (data/pytest_log.txt, measured)")
log_bytes = recorded("pytest_log.txt", "bytes of -v output")
log_tokens = recorded("pytest_log.txt", r"tokens under cl100k_base \(tiktoken\)")
v("pl_bytes", log_bytes, "{:,.0f}", "bytes of -v output")
v("pl_tokens", log_tokens, "{:,.0f}", "cl100k_base tokens")
v("pl_bpt", log_bytes / log_tokens, "{:.1f}", "bytes per token")
v("pl_windows", log_tokens / WINDOW, "{:.1f}", "windows")

print("== Napkin Math 7.2 (Scenarios.TwentyFourMegabyteLogObservation)")
o = LOG.result
v("ob_mb", LOG.raw_bytes.to("MB").magnitude, "{:.0f}", "raw log, MB")
v("ob_bpt", LOG.bytes_per_token.to("byte / token").magnitude, "{:.0f}", "assumed bytes per token")
v("ob_tokens", o.raw_tokens.to("token").magnitude, "{:,.0f}", "raw tokens")
v("ob_windows", o.raw_windows, "{:.1f}", "raw windows")
v("ob_shaped", LOG.shaped_tokens, "{:,.0f}", "shaped observation, tokens")
v("ob_share", o.shaped_share * 100, "{:.1f}", "shaped share of the window, %")
v("ob_calls", LOG.carry_calls, "{:.0f}", "later calls that carry it")
v("ob_carried", o.carried_tokens.to("token").magnitude, "{:,.0f}", "carried input tokens")
v("ob_cost", usd(o.carried_cost_uncached), "{:.2f}", "carried, uncached, $")
v("ob_cost_cached", usd(o.carried_cost_cached), "{:.3f}", "carried, cached, $")

print("== Pipe capacity (data/pipecap.txt, measured)")
pipe = recorded("pipecap.txt", "bytes written before the pipe blocked")
v("pipe_bytes", pipe, "{:,.0f}", "bytes a pipe held")
v("pipe_kib", Q_(pipe, "byte").to("KiB").magnitude, "{:.0f}", "the same, KiB")

print("== 500 repair runs (Scenarios.MiniSweAgentSonnet4Run; data/swebench_observations.csv)")
run = RUN.result
assert len(OBS_ROWS) == run.observations and OBS_CUT == run.truncated_outputs, \
    "the per-command file matches the run's recorded counts"
assert OBS_CUT == sum(1 for x in OBS_RAW if x >= STAGE_CAP), "staged exactly the outputs at the cap"
assert len({r["instance_id"] for r in OBS_ROWS}) == run.tasks and \
    len({r["instance_id"] for r in OBS_ROWS if r["truncated"] == "1"}) == run.truncated_tasks, \
    "the per-command file matches the run's recorded task counts"
big = max(OBS_ROWS, key=lambda r: int(r["raw_chars"]))
v("so_n", run.tasks, "{:.0f}", "runs (tasks)")
v("so_obs", run.observations, "{:,.0f}", "command outputs")
v("so_cut", run.truncated_outputs, "{:.0f}", "outputs cut")
v("so_cut_pct", run.truncated_outputs / run.observations * 100, "{:.1f}", "outputs cut, %")
v("so_runs_cut", run.truncated_tasks, "{:.0f}", "runs with at least one cut")
v("so_cap", STAGE_CAP, "{:,.0f}", "staging cap, characters")
v("so_head", RUN.head_chars, "{:,.0f}", "head kept, characters")
v("so_tail", RUN.tail_chars, "{:,.0f}", "tail kept, characters")
v("so_max", int(big["raw_chars"]), "{:,.0f}", "largest output, characters")
v("so_max_staged", int(big["staged_chars"]), "{:,.0f}", "the same, staged")

print("== SWE-agent (Literature.AgentRetrieval; Yang et al. 2024, Tables 1 and 3)")
v("swa_shell", SWE.SweBenchLiteShellResolved, "{:.1f}", "shell-only agent resolved, %")
v("swa_agent", SWE.SweBenchLiteSweAgentResolved, "{:.1f}", "SWE-agent (100-line viewer) resolved, %")
v("swa_full", SWE.SweAgentFullFileViewerResolved, "{:.1f}", "whole-file viewer resolved, %")
v("swa_noedit", SWE.SweAgentNoEditCommandResolved, "{:.1f}", "no edit command resolved, %")
v("swa_ratio", SWE.SweBenchLiteSweAgentResolved / SWE.SweBenchLiteShellResolved, "{:.2f}",
  "SWE-agent / shell only")

print("== Orphaned workers (data/orphans.txt, measured)")
v("orph_left", recorded("orphans.txt", r"SIGTERM to shell only: \d+ sleep workers before,"), "{:.0f}",
  "workers still running after SIGTERM to the shell")

print("== Napkin Math 7.3 (Scenarios.ModelPollingEveryFiveSeconds and variants)")
p = POLL.result
v("pc_job", POLL.job_duration.to("s").magnitude, "{:.0f}", "job, s")
v("pc_job_min", POLL.job_duration.to("minute").magnitude, "words", "job, minutes")
v("pc_int", POLL.poll_interval.to("s").magnitude, "{:.0f}", "poll interval, s")
v("pc_in", POLL.context_tokens, "{:,.0f}", "input per status call, tokens")
v("pc_out", POLL.output_tokens, "{:.0f}", "output per status call, tokens")
v("pc_cache", POLL.cache_hit * 100, "{:.0f}", "input cached, %")
v("pc_price_cached", per_million(p.cached_price), "{:.2f}", "cached input, $ per million")
v("pc_price_in", per_million(POLL.input_price), "{:.0f}", "uncached input, $ per million")
v("pc_price_out", per_million(POLL.output_price), "{:.0f}", "output, $ per million")
v("pc_polls", p.polls, "{:.0f}", "pending status calls")
v("pc_tin", p.input_tokens, "{:,.0f}", "input tokens")
v("pc_tout", p.output_tokens, "{:,.0f}", "output tokens")
v("pc_cin", usd(p.input_cost), "{:.4f}", "input cost, $")
v("pc_cout", usd(p.output_cost), "{:.4f}", "output cost, $")
v("pc_cost", usd(p.cost), "{:.4f}", "total per job, $")
v("pc_cost2", usd(p.cost), "{:.2f}", "total per job, $, cents")
for s in POLL_SWEEP[1:]:
    t = round(s.poll_interval.to("s").magnitude)
    v(f"pc_i{t}", t, "{:.0f}", "poll interval, s")
    v(f"pc_cost{t}s", usd(s.result.cost), "{:.2f}", f"total per job at {t} s, $")

print("== Napkin Math 7.4 (Scenarios.FiftyToolCatalogSixSelected)")
c = CATALOG.result
v("cat_reg", CATALOG.registered, "{:.0f}", "registered tools")
v("cat_desc", CATALOG.descriptor_tokens, "{:.0f}", "tokens per descriptor")
v("cat_sel", CATALOG.selected, "{:.0f}", "selected tools")
v("cat_over", CATALOG.selection_overhead, "{:.0f}", "selection overhead, tokens")
v("cat_full", c.full_tokens, "{:,.0f}", "full catalog, tokens")
v("cat_selected", c.selected_tokens, "{:,.0f}", "selected plus overhead, tokens")
v("cat_released", c.released_tokens, "{:,.0f}", "released per call, tokens")
v("cat_unused", c.unused_descriptor_tokens, "{:,.0f}", "descriptors left out, tokens")
v("cat_be", c.break_even_unused, "{:.1f}", "break-even unused descriptors")

print("== Exercise 9 (Scenarios.ShellAgainstTwelveStructuredTools)")
k = TOOLKIT.result
v("tk_tools", TOOLKIT.tools, "{:.0f}", "structured tools")
v("tk_tool_tok", TOOLKIT.tool_tokens, "{:.0f}", "tokens per structured descriptor")
v("tk_shell_tok", TOOLKIT.shell_tokens, "{:.0f}", "shell descriptor, tokens")
v("tk_catalog", k.structured_descriptor_tokens, "{:,.0f}", "structured descriptors per call, tokens")
v("tk_calls", TOOLKIT.calls, "{:.0f}", "model calls per trajectory")
v("tk_price", per_million(TOOLKIT.input_price), "{:.0f}", "input, $ per million")
v("tk_attempt", usd(TOOLKIT.structured_attempt), "{:.2f}", "structured attempt, $")
v("tk_acc", TOOLKIT.structured_acceptance, "{:.2f}", "structured acceptance")
v("tk_acc_shell", TOOLKIT.shell_acceptance, "{:.2f}", "shell acceptance")
v("tk_chg", TOOLKIT.structured_change, "{:.3f}", "structured unconfined-change probability")
v("tk_chg_shell", TOOLKIT.shell_change, "{:.3f}", "shell unconfined-change probability")
v("tk_chg_cost", usd(TOOLKIT.change_cost), "{:,.0f}", "cost of one unconfined change, $")
v("tk_saved", k.saved_tokens, "{:,.0f}", "tokens saved per trajectory")
v("tk_saved_usd", usd(k.saved_cost), "{:.2f}", "saved, $")
v("tk_saved_usd3", usd(k.saved_cost), "{:.3f}", "saved, $, three places")
v("tk_shell_attempt", usd(k.shell_attempt_cost), "{:.2f}", "shell attempt, $")
v("tk_shell_attempt3", usd(k.shell_attempt_cost), "{:.3f}", "shell attempt, $, three places")
v("tk_ceff", usd(k.structured_cost_per_accepted), "{:.2f}", "structured C_eff, tokens only, $")
v("tk_ceff_shell", usd(k.shell_cost_per_accepted), "{:.2f}", "shell C_eff, tokens only, $")
v("tk_loss", usd(k.structured_cost_per_accepted_with_loss), "{:.2f}", "structured C_eff with losses, $")
v("tk_loss_shell", usd(k.shell_cost_per_accepted_with_loss), "{:.2f}", "shell C_eff with losses, $")
v("tk_be", k.break_even_change_probability, "{:.4f}", "shell break-even change probability")

if __name__ == "__main__":
    write(__file__)
