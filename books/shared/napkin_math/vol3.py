"""Napkin Math, Volume III: the trajectory (an agent's sequence of model calls, tool calls, and state).

Every number below is computed here. Worked examples reuse the book's own
scenarios (mlsysim Scenarios.*), so the sheet and the chapters agree; their
inputs are hypothetical scenario assumptions, not measurements, and the sheet
tags the leading example input of each card with assumed(). Hardware and model
values come from the mlsysim registry. Each result is recomputed from the
scenario inputs with the formula printed on the sheet and checked against the
scenario's own solver, so a drift in either shows up as a build error.
"""
from __future__ import annotations

import math
from statistics import NormalDist

from mlsysim import Agents, Hardware as HW, Models as M
from mlsysim.engine.scenarios import Scenarios as SC

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, sig, v)

EXTERNAL: dict = {}  # nothing outside the registry
# Chart-only choices; each per-step rate is itself a book scenario input
# (ResidualViolationsFiftyTurns, UncheckedRepairHorizon, CompactionConstraintRetention).
SCENARIO = dict(chart_horizon=(1, 1000), chart_workers=16)


def _check(mine: float, book: float, what: str, rel: float = 1e-6) -> float:
    """Fail the build if the sheet's formula and the book's solver disagree."""
    if not math.isclose(mine, book, rel_tol=rel, abs_tol=1e-12):
        raise AssertionError(f"{what}: sheet {mine!r} vs book {book!r}")
    return mine


def _usd(q) -> float:
    return float(q.to("USD").magnitude)


def _per_tok(q) -> float:
    """A price in USD per million tokens, as USD per token."""
    return float(q.to("USD / Mtoken").magnitude) / 1e6


def _pct(x: float, d: int = 0) -> str:
    return f"{x * 100:,.{d}f}%"


def _pct_up(x: float, d: int = 1) -> str:
    """An upper bound, rounded up so the printed bound still holds."""
    m = 10 ** d
    return f"{math.ceil(x * 100 * m - 1e-9) / m:,.{d}f}%"


def _pct_dn(x: float, d: int = 1) -> str:
    """A lower bound or a ceiling on a rate, rounded down so it still holds."""
    m = 10 ** d
    return f"{math.floor(x * 100 * m + 1e-9) / m:,.{d}f}%"


def build() -> Sheet:
    A = TAG = assumed()

    # ================================================================ the band: two fundamentals (Ch 1)
    hz = SC.UncheckedRepairHorizon
    hz_r = hz.result
    eps = 1 - hz.step_success
    succ = {h: _check((1 - eps) ** h, hz_r.success[h], f"(1-eps)^{h}") for h in hz.horizons}
    eps_req = _check(1 - hz.target_success ** (1 / hz.long_horizon), hz_r.required_step_error, "required eps")


    # ================================================================ 1 The model call (Ch 2, 3)
    fl = SC.Llama3_70B_Int8DecodeH100
    fl_r = fl.result
    L70 = M.Language.Llama3_70B
    H100 = HW.Cloud.H100
    P = L70.parameters.magnitude
    R_PEAK = H100.compute.precision_flops[fl.peak_precision].to("flop/s").magnitude
    BW = H100.memory.bandwidth.to("B/s").magnitude
    s_elem = 1  # 8-bit weights, the scenario's precision
    t_pre = _check(2 * P / R_PEAK, fl_r.compute_time.to("s").magnitude, "t_prefill")
    t_dec = _check(P * s_elem / BW, fl_r.memory_time.to("s").magnitude, "t_decode")
    dec_ratio = t_dec / t_pre

    call = SC.CodingAgentFirstTurnCall
    call_r = call.result
    call_pre = _check(call.prompt_tokens * t_pre, call_r.prefill_time.to("s").magnitude, "call prefill")
    call_dec = _check(call.output_tokens * t_dec, call_r.decode_time.to("s").magnitude, "call decode")
    call_out_share = call_dec / (call_pre + call_dec)
    call_tok_share = call.output_tokens / (call.prompt_tokens + call.output_tokens)

    rs = SC.CodingAgentResend30Turns
    rs_r = rs.result
    H_rs, S0, dS, K_rs = rs.turns, rs.first_prompt, rs.appended, rs.output_per_turn

    def sent(h):
        return h * S0 + dS * h * (h - 1) // 2

    sent30 = _check(sent(H_rs), rs_r.resent_tokens, "S_sent")
    fresh30 = _check(S0 + (H_rs - 1) * dS, rs_r.reused_prefix_tokens, "fresh tokens")
    sent60 = _check(sent(2 * H_rs), SC.CodingAgentResend60Turns.result.resent_tokens, "S_sent 60")

    pools = SC.IndependentCandidatePools
    pools_r = pools.result
    pass_at = {k: _check(1 - (1 - pools.pass_rate) ** k, pools_r.pass_at[k], f"pass@{k}") for k in pools.pool_sizes}
    k_max_pool = max(pools.pool_sizes)
    mixed = _check(0.5 * (1 - (1 - pools.easy_pass_rate) ** k_max_pool) + 0.5 * (1 - (1 - pools.hard_pass_rate) ** k_max_pool),
                   pools_r.mixed_true, "mixed pass@8")

    vp = SC.SparseValidPoolFirstPass
    vp_r = vp.result
    tp = vp.valid_rate * (1 - vp.false_reject_rate)
    fp = (1 - vp.valid_rate) * vp.false_accept_rate
    prec = _check(tp / (tp + fp), vp_r.precision, "verifier precision")
    prec_half = SC.SparseValidPoolHalvedFalseAccept.result.precision
    tp2 = 2 * vp.valid_rate * (1 - vp.false_reject_rate)
    prec_dbl = _check(tp2 / (tp2 + (1 - 2 * vp.valid_rate) * vp.false_accept_rate),
                      SC.SparseValidPoolDoubledValidRate.result.precision, "precision, doubled valid rate")

    # ================================================================ 2 Agent memory (Ch 4, 5, 6)
    cb = SC.RepairContextBudget
    cb_r = cb.result
    input_limit = _check(cb.window - cb.output_reserve, cb_r.input_limit, "S_max - K_max")
    fixed = cb.system_tokens + cb.tool_tokens + cb.state_tokens
    obs_budget = _check(input_limit - fixed, cb_r.observation_budget, "observation budget")
    turns_fit = _check(obs_budget // cb.observation_per_turn, cb_r.turns_fit, "turns that fit")

    bill = SC.RepairResendCacheBill
    bill_r = bill.result
    c_in, c_out = _per_tok(bill.input_price), _per_tok(bill.output_price)
    c_cache = c_in * bill.cached_fraction
    prompts = [S0 + t * dS for t in range(H_rs)]
    turn_fresh = [c_in * s + c_out * K_rs for s in prompts]
    turn_cached = [turn_fresh[0]] + [c_cache * prompts[t - 1] + c_in * dS + c_out * K_rs for t in range(1, H_rs)]
    bill_fresh = _check(sum(turn_fresh), _usd(bill_r.total_uncached), "fresh bill")
    bill_cached = _check(sum(turn_cached), _usd(bill_r.total_cached), "cached bill")

    kv = SC.Llama3_70BLongContextAdmissionDgxH100
    kv_r = kv.result
    NL, HKV = L70.layers, L70.kv_heads
    DH = L70.hidden_dim // L70.heads
    s_kv = kv.kv_bytes_per_elem.to("B").magnitude
    m_tok = _check(2 * NL * HKV * DH * s_kv, kv_r.kv_per_token.to("B").magnitude, "m_token")
    S_ctx = kv.context_tokens
    m_req = _check(S_ctx * m_tok, kv_r.kv_per_request.to("B").magnitude, "M_KV")
    hbm = kv.accelerators * kv.hbm_per_accelerator.to("B").magnitude
    w_bytes = P * kv.weight_bytes_per_param.to("B").magnitude
    kv_pool = _check(hbm - w_bytes - kv.reserve.to("B").magnitude, kv_r.kv_budget.to("B").magnitude, "KV pool")
    max_req = _check(math.floor(kv_pool / m_req), kv_r.max_requests, "requests that fit")

    vs = SC.ExactScanTenMillionVectors
    vs_r = vs.result
    cpu_bw = vs.host.memory.bandwidth.to("B/s").magnitude
    vs_bytes = _check(vs.vectors * vs.dimension * vs.bytes_per_value.to("B").magnitude, vs_r.footprint.to("B").magnitude, "scan bytes")
    vs_t = _check(vs_bytes / cpu_bw, vs_r.stream_time.to("s").magnitude, "scan time")
    vs_traj = _check(vs.searches * vs_t, vs_r.trajectory_time.to("s").magnitude, "scan per trajectory")
    vs_filt = _check(vs.filtered_vectors * vs.dimension * vs.bytes_per_value.to("B").magnitude / cpu_bw,
                     vs_r.filtered_time.to("s").magnitude, "filtered scan")

    # ================================================================ 3 Tool use (Ch 7, 8, 9)
    br = SC.BlindRetryTwoPercentLoss
    br_r = br.result
    qreq, qrep = br.request_loss, br.reply_loss
    p_to = _check(1 - (1 - qreq) * (1 - qrep), br_r.timeout_probability, "P(timeout)")
    p_dup = _check((1 - qreq) * qrep * (1 - qreq), br_r.duplicate_probability, "p_dup")
    br_calls = br.workers * br.calls_per_worker
    dups_day = _check(br_calls * p_dup, br_r.duplicates, "duplicates per day")

    ob = SC.TwentyFourMegabyteLogObservation
    ob_r = ob.result
    raw_tok = ob.raw_bytes.to("B").magnitude / ob.bytes_per_token.to("B/token").magnitude
    raw_win = _check(raw_tok / ob.window, ob_r.raw_windows, "raw log windows")
    carried = _check(ob.shaped_tokens * ob.carry_calls, ob_r.carried_tokens.magnitude, "carried tokens")
    carried_usd = _check(carried * _per_tok(ob.input_price), _usd(ob_r.carried_cost_uncached), "carried cost")

    sp = SC.CodingSandboxPoolTwoPerSecond
    sp_r = sp.result
    lam = sp.arrival_rate.to("1/s").magnitude
    t_hold = sp.hold.to("s").magnitude
    t_init = sp.restore.to("s").magnitude
    n_active = _check(lam * t_hold, sp_r.active, "N_active")
    lt = lam * t_init

    def p_short(r):  # Poisson P(at least r restores outstanding)
        return 1 - sum(math.exp(-lt) * lt ** i / math.factorial(i) for i in range(r))

    reserve = next(r for r in range(1, 64) if p_short(r) <= sp.target_miss)
    _check(reserve, sp_r.reserve, "N_reserve")
    priv_mem = _check(n_active * sp.guest_memory.to("B").magnitude, sp_r.private_memory.to("B").magnitude, "private memory")
    pool_mem = _check(sp.guest_memory.to("B").magnitude + n_active * sp.dirty.to("B").magnitude + reserve * sp.standby.to("B").magnitude,
                      sp_r.pool_memory.to("B").magnitude, "shared pool memory")

    inj = SC.InjectionExposureByReadVolume
    inj_r = inj.result
    q_pay = inj.payload_fraction
    expo = {k: _check(1 - (1 - q_pay) ** r, inj_r.exposure[k], f"exposure {k}") for k, r in inj.reads.items()}
    expo_scr = _check(1 - (1 - q_pay * inj.miss_rate) ** inj.reads["research"], inj_r.screened["research"], "screened research")

    za = SC.CleanAttackSuite50
    za_r = za.result
    rule = -math.log(1 - za.confidence)
    za_up = _check(1 - (1 - za.confidence) ** (1 / za.trials), za_r.rate_upper_exact, "exact zero-failure bound")
    za_need = _check(math.ceil(math.log(1 - za.confidence) / math.log(1 - za.target_rate)), za_r.trials_for_target, "clean trials needed")

    # ================================================================ 4 The agent runtime (Ch 10, 11, 12)
    ad = SC.RepairNextCallAdmission
    ad_r = ad.result
    a_in, a_out = _per_tok(ad.input_price), _per_tok(ad.output_price)
    liab = _check(a_in * ad.input_tokens + a_out * ad.output_allowance, ad_r.liability, "liability")
    settled, reserved, fin, allow = _usd(ad.settled), _usd(ad.reserved), _usd(ad.finalization_reserve), _usd(ad.allowance)
    committed = settled + reserved + liab + fin
    max_in = _check((allow - settled - reserved - fin - a_out * ad.output_allowance) / a_in, ad_r.max_input_tokens, "max input")

    ck = SC.ReexecutedWorkCheckpointInterval
    ck_r = ck.result
    t_w, mtbf = ck.write_time.to("s").magnitude, ck.mtbf.to("s").magnitude
    t_star = _check(math.sqrt(2 * t_w * mtbf), ck_r.interval.to("s").magnitude, "Young interval")
    waste = _check(t_w / t_star + t_star / (2 * mtbf), ck_r.waste, "checkpoint waste")

    rv = SC.TestConfigRepairVsRebuild
    rv_r = rv.result
    t_fb = _check((rv.unwind_time + rv.rebuild_time).to("s").magnitude, rv_r.fallback_time.to("s").magnitude, "fallback time")
    t_rep = _check(rv.startup_time.to("s").magnitude + rv.repair_output / rv.output_rate.to("token/s").magnitude + rv.validation_time.to("s").magnitude,
                   rv_r.repair_time.to("s").magnitude, "repair time")
    r_in, r_out = _per_tok(rv.input_price), _per_tok(rv.output_price)
    c_fb = _check(r_in * rv.rebuild_input + r_out * rv.rebuild_output, _usd(rv_r.fallback_cost), "fallback cost")
    c_rep = _check(r_in * rv.repair_input + r_out * rv.repair_output, _usd(rv_r.repair_cost), "repair cost")
    pstar_t, pstar_c = t_rep / t_fb, c_rep / c_fb
    t_exp = _check(t_rep + (1 - rv.assumed_success) * t_fb, rv_r.expected_time.to("s").magnitude, "expected time")

    pv = SC.RemediationCanaryPivotLoss
    pv_r = pv.result
    wk = pv.workload
    impact = _usd(wk.wrong_change_impact)
    l0 = _check(wk.wrong_change_probability * impact, _usd(pv_r.loss_without_check), "loss without canary")
    l1 = _check(l0 * (wk.canary_catch_rate * (1 - wk.canary_recovery_fraction) + (1 - wk.canary_catch_rate)),
                _usd(pv_r.loss_with_check), "loss with canary")
    pv_att = _usd(pv.attempt_cost)

    ra = SC.ThreeLayerRetryAmplification
    ra_r = ra.result
    layers = len(ra.layer_names)
    amp = _check(ra.attempts_per_layer ** layers, ra_r.layered, "retry amplification")

    lp = SC.RepairLoopToTurnCeiling
    lp_r = lp.result
    lp_d = lp.output_per_turn + lp.observation_per_turn

    def loop_cost(h):
        inp = h * lp.first_input + lp_d * h * (h - 1) // 2
        return inp * _per_tok(lp.input_price) + h * lp.output_per_turn * _per_tok(lp.output_price)

    lp_ceiling = _check(loop_cost(lp.ceiling), _usd(lp_r.ceiling_cost), "loop to ceiling")
    lp_stop = _check(loop_cost(lp.stop_after), _usd(lp_r.stop_cost), "loop stopped")

    # ================================================================ 5 Evaluation and learning (Ch 13-17)
    wi = SC.FiftyTaskIntervals
    wi_r = wi.result
    z = _check(NormalDist().inv_cdf(0.975), wi_r.z_critical, "z")
    Nt = wi.trials

    def wilson(x):
        c = (x + z * z / 2) / (Nt + z * z)
        h = z / (Nt + z * z) * math.sqrt(x * (Nt - x) / Nt + z * z / 4)
        return c - h, c + h

    wil = {x: wilson(x) for x in wi.successes}
    for x in wi.successes:
        _check(wil[x][1], wi_r.wilson[x][1], f"Wilson upper {x}/{Nt}")
    wald_lo2 = wi_r.wald[2][0]
    x2 = wi.successes[1]
    _check(x2 / Nt - z * math.sqrt(x2 / Nt * (1 - x2 / Nt) / Nt), wald_lo2, "Wald lower 2/50")

    pl = SC.AcceptanceRatePlanning
    pl_r = pl.result
    var = pl.rate * (1 - pl.rate)
    per_arm = {d: _check(2 * (pl.z_alpha + pl.z_beta) ** 2 * var / d ** 2, pl_r.per_arm[d], f"N_arm at {d}") for d in pl.differences}
    one_rate = _check(math.ceil(z ** 2 * var / pl.target_half_width ** 2), pl_r.samples_for_target, "N for one rate")

    rc = SC.RecurringTaskConsistency
    rc_r = rc.result
    pk = _check(rc.step_success ** rc.long_horizon, rc_r.success[rc.long_horizon], "pass^k")
    p_need = _check(rc.target_success ** (1 / rc.long_horizon), 1 - rc_r.required_step_error, "single-run rate needed")

    hs = SC.RareFailureUnderHeadSampling
    hs_miss = _check((1 - hs.retention_probability) ** hs.occurrences, hs.miss, "head-sampling miss")
    hs50 = SC.RecurringFailureUnderHeadSampling
    hs50_miss = _check((1 - hs50.retention_probability) ** hs50.occurrences, hs50.miss, "head-sampling miss 50")

    gr = SC.HardTaskRolloutStep
    gr_r = gr.result

    def contrast(p, g):
        return 1 - p ** g - (1 - p) ** g

    g_c = _check(contrast(gr.pass_rate, gr.group), gr_r.contrast, "group contrast")
    g_per = _check(gr.group / g_c, gr_r.rollouts_per_informative_group, "rollouts per informative group")
    tg = SC.TargetedTaskRolloutStep
    g_c_t = _check(contrast(tg.pass_rate, tg.group), tg.result.contrast, "targeted contrast")
    gr_tok = _check(gr.tasks * gr.group * gr.turns * gr.tokens_per_turn, gr_r.step_tokens, "tokens per update")

    lo = SC.Rank16AdaptersOnHypothetical8B
    lo_P = lo.workload.parameters.magnitude
    per_mat = lo.adapter_rank * (lo.adapter_in_dim + lo.adapter_out_dim)
    lo_n = _check(per_mat * lo.adapted_matrices * lo.workload.layers, lo.adapter_count, "adapter parameters")
    b_train = lo.bytes_per_value.to("B").magnitude * 2 + lo.master_bytes_per_parameter.to("B").magnitude + lo.optimizer_bytes_per_parameter.to("B").magnitude
    lo_state = _check(lo_n * b_train, lo.adapter_state.to("B").magnitude, "adapter training state")
    full_state = _check(lo_P * b_train, lo.full_state.to("B").magnitude, "full training state")
    frozen = lo_P * lo.bytes_per_value.to("B").magnitude

    # ================================================================ 6 Agents at scale (Ch 18, 19)
    ps = SC.RepairPartitionScreen
    f_par, h_w = ps.parallel_fraction, ps.overhead_per_extra_worker

    def t_rel(nw, h=h_w):
        return (1 - f_par) + f_par / nw + h * (nw - 1)

    for nw, el in ps.elapsed.items():
        _check(t_rel(nw), el, f"elapsed at {nw} workers")
    best_n = min(range(1, ps.max_workers + 1), key=t_rel)
    _check(best_n, ps.best_workers, "best worker count")
    best_s = 1 / t_rel(best_n)
    us = SC.RepairPartitionScalability
    n_star = _check(math.sqrt((1 - us.contention) / us.coherency), us.continuous_peak, "USL peak")
    rf = SC.ResearchFanoutScalability
    n_star_rf = _check(math.sqrt((1 - rf.contention) / rf.coherency), rf.continuous_peak, "research USL peak")

    dl = SC.ResearchSystemTokenBreakEven
    m_tok_mult = dl.multi_agent_tokens_vs_chat / dl.agent_tokens_vs_chat
    max_single = _check(1 / m_tok_mult, dl.max_single_acceptance, "max single acceptance")
    gain_mult = 1 + dl.reported_gain
    ratio_at_gain = _check(m_tok_mult / gain_mult, dl.cost_ratio_if_gain_is_acceptance, "cost ratio at reported gain")

    tc = SC.RepairCohortTokenPriceComparison
    tc_r = tc.result

    def cohort(c):
        turn_tok = c.input_tokens_per_turn * _per_tok(c.input_price) + c.output_tokens_per_turn * _per_tok(c.output_price)
        att = c.turns * (turn_tok + _usd(c.environment_per_turn)) + _usd(c.assessment_per_attempt)
        tot = c.tasks * att
        tri = (c.tasks - c.accepted) * _usd(c.triage_per_rejected)
        return att, tot / c.accepted, (tot + tri) / c.accepted

    fr_att, fr_eff, fr_tri = cohort(tc.baseline)
    lw_att, lw_eff, lw_tri = cohort(tc.alternative)
    _check(lw_eff / fr_eff, tc_r.cost_ratio, "token-price ratio")
    _check(lw_tri / fr_tri, tc_r.cost_ratio_with_triage, "token-price ratio with triage")
    tok_cut = 1 - _per_tok(tc.alternative.input_price) / _per_tok(tc.baseline.input_price)

    qa = SC.RepairTrajectoryQuotaAdmission
    qa_r = qa.result
    lam_t = qa.arrival.to("1/minute").magnitude
    T_t = qa.residence.to("minute").magnitude
    n_traj = _check(lam_t * T_t, qa_r.concurrent, "N_traj")
    tpm = _check(n_traj * qa.calls / T_t * qa.tokens_per_call, qa_r.demand.to("1/minute").magnitude, "tokens per minute")
    quota_share = tpm / qa.quota.to("1/minute").magnitude

    de = SC.DedicatedCapacityBreakEven
    de_r = de.result
    p_in_w = de.cached_fraction * _per_tok(de.cached_price) + (1 - de.cached_fraction) * _per_tok(de.input_price)
    api_out = de.input_per_output * p_in_w + _per_tok(de.output_price)  # USD per output token, all-in
    rate = de.reference_rate.to("1/s").magnitude
    fixed_h = de.fixed_per_hour.to("USD/hour").magnitude
    u_star = _check(fixed_h / (3600 * rate * (api_out - _per_tok(de.variable_price))), de_r.crossing, "break-even utilization")

    ap = SC.CodeReviewAdaptationPayback
    ap_r = ap.result

    def task_cost(d):
        return d["steps"] * (d["input_tokens"] * _per_tok(d["input_price"]) + d["output_tokens"] * _per_tok(d["output_price"]))

    c_base, c_adapt = task_cost(ap.base), task_cost(ap.adapted)
    invest = _usd(ap.data_cost) + _usd(ap.training_cost) + _usd(ap.evaluation_cost)
    n_be = _check(invest / (c_base - c_adapt), ap_r.breakeven_tasks, "break-even tasks")
    months = n_be / ap.monthly_volume

    # ---- reference workloads (bottom table): the four lighthouses
    lh = SC.LighthouseWorkloadLevers
    lh_r = lh.result.rows
    W = lh.workloads
    rows_cost = {}
    rp = W["repair"]
    rows_cost["repair"] = dict(model=round(bill_cached, 2),  # printed cents, so the row reproduces
                                env=rp["environment_per_turn"] * rp["turns"], assess=rp["assessment"], human=rp["human"], acc=rp["acceptance"])
    rr = W["research"]
    chat = rr["chat_input"] * c_in + rr["chat_output"] * c_out
    rows_cost["research"] = dict(model=chat * rr["token_multiple"], env=rr["environment"], assess=rr["assessment"], human=rr["human"], acc=rr["acceptance"])
    rm = W["remediation"]
    rows_cost["remediation"] = dict(model=rm["model"], env=rm["environment"], assess=rm["assessment"], human=rm["human"], acc=rm["acceptance"])
    su = W["support"]
    rows_cost["support"] = dict(model=su["model"], env=su["environment"], assess=su["assessment"], human=su["handoff"] * (1 - su["acceptance"]), acc=su["acceptance"])
    for k, d in rows_cost.items():
        d["att"] = _check(d["model"] + d["env"] + d["assess"] + d["human"], lh_r[k].attempt, f"{k} attempt", rel=0.01)
        d["eff"] = _check(d["att"] / d["acc"], lh_r[k].cost_per_accepted, f"{k} C_eff", rel=0.01)
        terms = {"model tokens": d["model"], "sandbox and tools": d["env"], "checks": d["assess"], "human review": d["human"]}
        d["largest"] = max(terms, key=terms.get)
        d["share"] = terms[d["largest"]] / d["att"]
    horizon = {"repair": Agents.Coding.SWE_Bench_Runner.max_trajectory_steps, "research": Agents.Research.OrchestratorWorker.horizon_turns,
               "remediation": Agents.Remediation.OnCall.horizon_turns, "support": Agents.Support.CustomerOps.horizon_turns}
    archetype = {"repair": "long-horizon worker", "research": "untrusted reader", "remediation": "irreversible actor", "support": "interactive responder"}

    # band: the repair lighthouse row, so band, prefix-cache card, and table share one attempt
    rep = rows_cost["repair"]
    lever_gain = lh.acceptance_gain
    eff_half_tok = (rep["model"] / 2 + rep["env"] + rep["assess"] + rep["human"]) / rep["acc"]
    _check(1 - eff_half_tok / rep["eff"], lh_r["repair"].halve_tokens, "repair: halve token price", rel=0.01)
    eff_more_acc = rep["att"] / (rep["acc"] + lever_gain)
    _check(1 - eff_more_acc / rep["eff"], lh_r["repair"].acceptance_gain, "repair: +10 points acceptance", rel=0.01)

    tail = SC.ThirtyCallTailExposure
    tail_any = _check(1 - (1 - tail.tail_probability) ** tail.calls, tail.any_tail, "any p99 call")



    # ---- printed inputs that let a reader reproduce each result
    def usd_m(price) -> str:
        x = price.to("USD/Mtoken").magnitude
        return f"${x:g}" if x == int(x) else f"${x:.2f}"

    def money(x: float, d: int = 2) -> str:
        """Dollars, rounded half up so a printed sum reproduces."""
        return f"${x + 1e-9:,.{d}f}"

    rep_eff = rep["eff"]
    fb, al = tc.baseline, tc.alternative
    rv_rate = rv.output_rate.to("token/s").magnitude
    tok_in_s, tok_out_s = 1 / t_pre, 1 / t_dec
    price_out_ratio = c_out / c_in
    price_cache_ratio = c_cache / c_in
    e_inv = math.exp(-1)
    _check(succ[hz.long_horizon], e_inv, "0.99^100 ≈ 1/e", rel=0.01)
    hbm_gib = kv.hbm_per_accelerator.to("GiB").magnitude
    lw_att_cost = al.turns * (al.input_tokens_per_turn * _per_tok(al.input_price) + al.output_tokens_per_turn * _per_tok(al.output_price)
                              + _usd(al.environment_per_turn)) + _usd(al.assessment_per_attempt)
    _check(lw_att_cost, lw_att, "cheap-model attempt")
    fb_turn = fb.input_tokens_per_turn * _per_tok(fb.input_price) + fb.output_tokens_per_turn * _per_tok(fb.output_price) + _usd(fb.environment_per_turn)
    al_turn = al.input_tokens_per_turn * _per_tok(al.input_price) + al.output_tokens_per_turn * _per_tok(al.output_price) + _usd(al.environment_per_turn)
    # printed per-turn costs reproduce the printed attempt costs
    _check(round(fb.turns * round(fb_turn, 2) + _usd(fb.assessment_per_attempt), 2), round(fr_att, 2), "frontier printed sum")
    _check(round(al.turns * round(al_turn, 2) + _usd(al.assessment_per_attempt), 2), round(lw_att, 2), "small-model printed sum")
    hbm_gb = hbm / 1e9
    young_write, young_rework = t_w / t_star, t_star / (2 * mtbf)

    # ================================================================ page 3 and 4 extras
    rt = SC.CheapRetryAgainstStrongCall
    rt_r = rt.result
    rt_cheap = _check(_usd(rt.attempt_cost) / rt.pass_probability, _usd(rt_r.sequential_per_accepted), "cheap per accepted")
    rt_strong = _check(_usd(rt.strong_attempt_cost) / rt.strong_pass_probability, _usd(rt_r.strong_per_accepted), "strong per accepted")
    rt_cover = _check(1 - (1 - rt.pass_probability) ** rt.max_attempts, rt_r.coverage, "retry coverage")

    cs = SC.CodeEditThreeTierCascade
    cs_r = cs.result
    tier_cost = [_usd(c) + _usd(cs.gate_cost) for c in cs.call_costs]
    cs_task = _check(sum(r * c for r, c in zip(cs_r.reach_shares, tier_cost)), cs_r.per_initiated, "cascade per task")
    cs_acc = sum(cs.accepted_by_tier) / cs.tasks
    cs_eff = _check(cs_task / cs_acc, cs_r.per_accepted, "cascade per accepted")
    cs_direct = _check(tier_cost[-1] * cs.tasks / cs.direct_accepted, cs_r.direct_per_accepted, "frontier alone per accepted")

    p_api_out = api_out  # USD per output token, input included
    sd = SC.SerialRepairEpisodeSpeculation
    sd_r = sd.result
    sd_den = sd.draft_length * sd.draft_cost_ratio + 1
    sd_hi = _check((sd.accepted_high + 1) / sd_den, sd_r.speedup_high, "spec speedup high")
    sd_lo = _check((sd.accepted_low + 1) / sd_den, sd_r.speedup_low, "spec speedup low")
    ep = sd.episode
    ep_r = ep.result
    ep_model, ep_total = ep_r.model_time.to("s").magnitude, ep_r.total_time.to("s").magnitude
    ep_bound = _check(1 / (1 - ep_model / ep_total), sd_r.episode_speedup_bound, "episode bound")

    cw = SC.RepairCompactionWatermarks
    cw_r = cw.result
    cyc = _check(math.ceil((cw.trigger - cw.target) / cw.tokens_per_turn), cw_r.cycle_turns, "compaction cycle")
    cyc_alt = _check(math.ceil((cw.trigger - cw.alternate_target) / cw.tokens_per_turn), cw_r.alternate_cycle_turns, "narrow-gap cycle")

    tcat = SC.FiftyToolCatalogSixSelected
    tcat_r = tcat.result
    tc_full = _check(tcat.registered * tcat.descriptor_tokens, tcat_r.full_tokens, "full catalog")
    tc_sel = _check(tcat.selected * tcat.descriptor_tokens + tcat.selection_overhead, tcat_r.selected_tokens, "selected catalog")

    an = SC.AnnIndexesTenMillionVectors
    an_r = an.result
    an_raw = an.vectors * an.dimension * an.bytes_per_value.to("B").magnitude
    an_graph = _check(an.vectors * an.links_per_node * an.link_bytes.to("B").magnitude, an_r.graph.to("B").magnitude, "HNSW links")
    an_codes = _check(an.vectors * an.subvectors * an.code_bytes.to("B").magnitude, an_r.codes.to("B").magnitude, "PQ codes")

    dg = SC.InterfaceMigrationSchedule
    dg_d = {k: q.to("s").magnitude for k, q in dg.durations.items()}
    dg_work = _check(sum(dg_d.values()), dg.work.to("s").magnitude, "DAG work")
    finish: dict = {}
    for node in sorted(dg_d):
        finish[node] = dg_d[node] + max((finish[p] for p in dg.predecessors[node]), default=0.0)
    dg_cp = _check(max(finish.values()), dg.makespan.to("s").magnitude, "critical path")
    path, node = [], max(finish, key=finish.get)
    while True:
        path.append(node)
        preds = dg.predecessors[node]
        if not preds:
            break
        node = max(preds, key=lambda q: finish[q])
    path.reverse()

    kq = SC.BurstyModelServerQueuePrice
    kq_r = kq.result
    kq_v = (kq.arrival_scv + kq.service_scv) / 2
    t_s1 = kq.service.to("s").magnitude
    kq_lo = _check(kq_v * kq.low_utilization / (1 - kq.low_utilization) * t_s1, kq_r.wait_low.to("s").magnitude, "Kingman low")
    kq_hi = _check(kq_v * kq.high_utilization / (1 - kq.high_utilization) * t_s1, kq_r.wait_high.to("s").magnitude, "Kingman high")
    mm1_lo = kq.low_utilization / (1 - kq.low_utilization) * t_s1

    jt = SC.RetryCohortJitter
    jt_burst = jt.retries / jt.burst_window.to("s").magnitude
    jt_spread = jt.retries / jt.jitter_span.to("s").magnitude

    pc = SC.RepairCandidatePairedComparison
    pc_r = pc.result
    pc_n = pc.both + pc.candidate_only + pc.baseline_only + pc.neither
    pc_d = _check((pc.candidate_only - pc.baseline_only) / pc_n, pc_r.difference, "paired difference")
    pc_disc = (pc.candidate_only + pc.baseline_only) / pc_n
    pc_hw = _check(z * math.sqrt((pc_disc - pc_d ** 2) / pc_n), pc_r.paired_half_width, "paired half-width")

    sq = SC.AcceptanceCanaryMonitor
    sq_r = sq.result
    sq_A = _check(math.log((1 - sq.beta) / sq.alpha), sq_r.upper, "SPRT upper")
    sq_fail = _check(math.log((1 - sq.p_degraded) / (1 - sq.p_base)), sq_r.failure_step, "SPRT failure step")
    sq_succ = _check(math.log(sq.p_degraded / sq.p_base), sq_r.success_step, "SPRT success step")
    sq_rate = sq.arrival_rate.to("1/hour").magnitude * sq.fraction
    sq_hours = _check(sq_r.expected_samples / sq_rate + sq.label_delay.to("hour").magnitude, sq_r.decision_time.to("hour").magnitude, "SPRT decision time")
    sq2_hours = SC.AcceptanceCanaryMonitorDoubled.result.decision_time.to("hour").magnitude

    rs_ = SC.EightAttemptRunSplit
    rs_r = rs_.result
    rs_whole = _check(rs_.train_fraction ** rs_.attempts + (1 - rs_.train_fraction) ** rs_.attempts, rs_r.whole, "run split whole")

    mh = SC.TrajectoryMinHashScreen
    mh_p = {j: _check(1 - (1 - j ** mh.rows) ** mh.bands, mh.candidate_probability(j), f"MinHash {j}") for j in mh.probe_similarities}
    mh_bytes = _check(mh.bands * mh.rows * mh.hash_bytes.to("B").magnitude * mh.records, mh.index_bytes.to("B").magnitude, "MinHash index")

    ao = SC.DemonstrationAdmissionOrder
    ao_r = ao.result

    def check_time(order):
        tot, reach = 0.0, 1.0
        for name in order:
            pp, tt = ao.checks[name]
            tot += reach * tt.to("s").magnitude
            reach *= pp
        return tot

    ao_t = [_check(check_time(o), ao_r.time_per_attempted[k].to("s").magnitude, f"admission order {k}") for k, o in enumerate(ao.orders)]

    cd = SC.CurationCohortDispositions
    cd_emit = cd.dispositions[cd.emitted]
    cd_cost = _usd(cd.cohort_cost) / cd_emit
    cd_total = sum(cd.dispositions.values())

    pk_ = SC.EightExamplePackingBatch
    pk_r = pk_.result
    pk_store = _check(sum(pk_.lengths), pk_r.stored, "packed stored tokens")
    pk_pad = _check(pk_store / (len(pk_.lengths) * pk_.capacity), pk_r.padded_occupancy, "padded occupancy")
    pk_pack = _check(pk_store / (pk_r.buffers * pk_.capacity), pk_r.packed_occupancy, "packed occupancy")

    dp = SC.RepairPreferencePair
    dp_r = dp.result
    dp_w = dp.policy_preferred - dp.reference_preferred
    dp_l = dp.policy_dispreferred - dp.reference_dispreferred
    dp_m = _check(dp.beta * (dp_w - dp_l), dp_r.margin, "DPO margin")
    dp_loss = _check(-math.log(1 / (1 + math.exp(-dp_m))), dp_r.loss, "DPO loss")
    dp_g = _check(dp.beta / (1 + math.exp(dp_m)), dp_r.gradient_coefficient, "DPO gradient coefficient")

    fa = SC.VerifierFalseAcceptsAtTenPercentPass
    fa_r = fa.result
    fa_share = _check((1 - fa.pass_rate) * fa.false_accept_rate / (fa.pass_rate + (1 - fa.pass_rate) * fa.false_accept_rate), fa_r.false_share, "false-accept share")

    lp16 = SC.LonePassInGroupOfSixteen
    lp16_r = lp16.result
    g16 = len(lp16.rewards)
    mu16 = sum(lp16.rewards) / g16
    sd16 = math.sqrt(sum((r - mu16) ** 2 for r in lp16.rewards) / g16)
    adv16 = _check((1 - mu16) / sd16, lp16_r.coefficients[0], "lone-pass advantage")
    advf16 = (0 - mu16) / sd16

    cr = SC.FiveCorrelatedReviewers
    maj = lambda m: sum(math.comb(m, j) * cr.false_approval ** j * (1 - cr.false_approval) ** (m - j) for j in range(m // 2 + 1, m + 1))  # noqa: E731
    cr_i5 = _check(maj(cr.reviewers), cr.independent_majority(cr.reviewers), "independent majority 5")
    cr_i15 = _check(maj(cr.large_ensemble), cr.independent_majority(cr.large_ensemble), "independent majority 15")
    cr_c5, cr_c15 = cr.correlated_majority(cr.reviewers), cr.correlated_majority(cr.large_ensemble)

    # ================================================================ charts
    hp = Plot((1, 1000), (0, 1), xlog=True, W=262, H=82, left=32, right=10, top=8, bottom=21)
    hp.grid([1, 10, 100, 1000], [0, 0.5, 1.0], yfmt=lambda y: f"{y * 100:g}%")
    eps_lines = [(SC.ResidualViolationsFiftyTurns.step_violation, "line muted"), (eps, "line"),
                 (1 - SC.CompactionConstraintRetention.step_success, "line cat3")]
    for e, cls in eps_lines:
        hp.line([(10 ** (i / 40), (1 - e) ** (10 ** (i / 40))) for i in range(121)], cls)
    hp.text(hp.X(250), hp.Y((1 - eps_lines[0][0]) ** 250) - 6, f"ε = {eps_lines[0][0] * 100:g}%", "start")
    hp.text(hp.X(150), hp.Y(0.55), f"ε = {eps * 100:g}%", "start")
    hp.text(hp.X(1.6), hp.Y(0.62), f"ε = {eps_lines[2][0] * 100:g}%", "start")
    for h in (hz.short_horizon, hz.long_horizon):
        hp.point(h, succ[h], _pct(succ[h]), "end", -5, 10)
    hp.axes("turns H (log scale)", "Pr(success)")

    wmax = SCENARIO["chart_workers"]
    sp_plot = Plot((1, wmax), (1, 6), W=226, H=96, left=34, right=8, top=8, bottom=21)
    sp_plot.grid([1, 4, 8, 12, 16], [1, 2, 3, 4, 5], yfmt=lambda y: f"{y:g}×")
    sp_plot.line([(nw, 1 / t_rel(nw, 0)) for nw in range(1, wmax + 1)], "line alt")
    sp_plot.line([(nw, 1 / t_rel(nw)) for nw in range(1, wmax + 1)], "line")
    sp_plot.text(sp_plot.X(wmax) - 2, sp_plot.Y(1 / t_rel(wmax, 0)) - 5, "Amdahl, no coordination cost", "end")
    sp_plot.point(best_n, best_s, f"peak {best_s:.1f}× at {best_n}", "start", 4, 13)
    sp_plot.text(sp_plot.X(wmax) - 2, sp_plot.Y(1 / t_rel(wmax)) + 11, f"h = {h_w:g}", "end")
    sp_plot.axes("workers", "speedup")

    kp = Plot((0, 1), (0, 40), W=226, H=80, left=40, right=8, top=8, bottom=21)
    kp.grid([0, 0.25, 0.5, 0.75, 1.0], [0, 20, 40], yfmt=lambda y: f"{y:g} s")
    for mult, cls in ((1.0, "line alt"), (kq_v, "line")):
        pts, r = [], 0.0
        while r < 0.995:
            w = mult * r / (1 - r) * t_s1
            pts.append((r, w))
            if w > 40:
                break
            r += 0.005
        kp.line(pts, cls)
    kp.point(kq.low_utilization, kq_lo, f"{fmt(kq_lo)} s", "end", -5, -3)
    kp.point(kq.high_utilization, kq_hi, f"{fmt(kq_hi)} s", "end", -5, 3)
    kp.axes("utilization ρ", "wait")

    cwp = Plot((0, cw.turns), (40_000, 100_000), W=226, H=80, left=40, right=8, top=8, bottom=21)
    cwp.grid([0, 10, 20, 30], [40_000, 70_000, 100_000], yfmt=lambda y: f"{y / 1000:g}K")
    cwp.line(list(enumerate(cw_r.lengths, 1)), "line")
    cwp.line(list(enumerate(cw_r.alternate_lengths, 1)), "line cat3")
    cwp.line([(0, cw.trigger), (cw.turns, cw.trigger)], "line alt")
    cwp.axes("turn", "tokens")

    # ================================================================ page 1
    band = (f'<div class="eqbig">{v("C", "eff")} = {frac("E[" + v("C", "attempt") + "]", v("p", "acc"))}'
            f'<span class="lbl">cost per accepted task (Ch 1, 19)</span></div>'
            f'<div class="txt"><div><b>Price the acceptance, not the token.</b> A repair attempt: ${rep["model"]:.2f} tokens (the 30-turn cached bill, card 2·2) '
            f'+ {A}${rep["env"]:.2f} sandbox + ${rep["assess"]:.2f} checks = ${rep["att"]:.2f}; ÷ {v("p", "acc")} {rep["acc"]:g} → {v("C", "eff")} = {n(f"${rep_eff:.2f}")}.</div>'
            f'<div>Half-price tokens → {money(eff_half_tok)}; {v("p", "acc")} + {fmt(lever_gain * 100)} points → {n(money(eff_more_acc))}. <b>So</b> acceptance is the larger lever.</div>'
            f'<div><b>Horizon compounding (Ch 1).</b> Pr(success) = (1 − {v("ε")})<sup>{v("H")}</sup> ≈ e<sup>−{v("ε")}{v("H")}</sup>, '
            f'{v("ε")} error per unchecked turn, {v("H")} turns. {A}{v("ε")} = {_pct(eps)}: {_pct(succ[hz.short_horizon])} at {v("H")} = {hz.short_horizon}, '
            f'{n(_pct(succ[hz.long_horizon]), True)} at {hz.long_horizon} (target {_pct(hz.target_success)} needs {v("ε")} ≤ {_pct_dn(eps_req, 3)}).</div></div>'
            f'<div class="side" style="width:2.75in">{hp.svg("Probability of success against turns for three per-turn error rates")}</div>')

    p1 = Page(
        question="Will it finish within budget?",
        sub="Read left to right: where a call's time and money go, what the context holds, what a tool call risks. The band prices one attempt.",
        meta="reference: Llama 3 70B on NVIDIA H100",
        notation=[
            ("Symbols", f'{v("H")} turns · {v("S")}, {v("K")} input, output tokens · {v("P")} parameters · {v("s", "elem")} bytes/value · '
                        f'{v("c", "in")}, {v("c", "out")}, {v("c", "cache")} $/token in, out, cached · Δ{v("S", "turn")} appended/turn · {v("p", "acc")} acceptance · {v("C", "eff")} $/accepted task'),
        ],
        know=[
            Know("0.99<sup>100</sup> ≈ 1/e", f"(1 − ε)<sup>H</sup> ≈ e<sup>−εH</sup>: 100 turns at 99% ≈ {_pct(e_inv)}"),
            Know(f"{fmt(rule)} / n", "95% bound on a failure rate after n clean trials"),
            Know("≈ <i>n</i> · <i>q</i>", "1 − (1 − <i>q</i>)<sup><i>n</i></sup> when <i>nq</i> ≪ 1 (exposure, coverage)"),
            Know(f"{fmt(m_tok / 1024)} KiB", "KV state per token, Llama 3 70B, FP16"),
            Know(f"${c_in * 1e6:g} · ${c_out * 1e6:g}", f"$ per M in, out tokens (book's hypothetical); cached ${c_cache * 1e6:.2f}"),
            Know(f"{fmt(price_out_ratio)}× · {price_cache_ratio:g}×", "output, cached-input price ÷ input price"),
            Know(f"{H100.memory.bandwidth.to('TB/s').magnitude:.2f} · {fmt(R_PEAK / 1e12)}", "H100: HBM TB/s · FP8 peak TFLOP/s"),
        ],
        band=band,
        cols=[
            Col("1 Where do a call's time and money go?", [
                Card("Call latency floor",
                     f'{v("T")} ≥ {v("S")} · {frac("2" + v("P"), v("R", "peak"))} + {v("K")} · {frac(v("P") + " · " + v("s", "elem"), "BW")}',
                     chapter="Ch 2", holds=f'{v("R", "peak")} peak FLOP/s, BW bytes/s; batch 1, before queueing.',
                     ex=f'70B 8-bit, H100, {A}{v("S")} = {fmt(call.prompt_tokens)}, {v("K")} = {call.output_tokens}: '
                        f'2 · {P / 1e9:.1f}G · {fmt(call.prompt_tokens)} / {fmt(R_PEAK / 1e12)}T = {call_pre:.2f} s; '
                        f'{call.output_tokens} · {P / 1e9:.1f} GB / {BW / 1e12:.2f} TB/s = {call_dec:.2f} s.',
                     so=f'decode: {n(_pct(call_out_share))} of the time, {_pct(call_tok_share, 1)} of the tokens. Shorten outputs.'),
                Card("Append-only history is resent every turn",
                     f'{v("S", "sent")} = {v("H")} · {v("S", "base")} + Δ{v("S", "turn")} · {frac(v("H") + "(" + v("H") + " − 1)", "2")}',
                     chapter="Ch 2", holds=f'{v("S", "base")} first prompt; {v("S", "sent")} input tokens sent over {v("H")} calls.',
                     ex=f'{A}{H_rs} · {fmt(S0)} + {fmt(dS)} · {H_rs}·{H_rs - 1}/2 = {n(fmt(sent30))}; {fmt(fresh30)} new.',
                     so=f'grows as {v("H", sup="2")}: {2 * H_rs} turns send {sent60 / sent30:.1f}×. Cache the prefix (card 2·2).'),
                Card("pass@k: one of k independent tries passes", f'pass@{v("k")} = E<sub><i>i</i></sub>[1 − (1 − {v("p", "i")})<sup>{v("k")}</sup>]', chapter="Ch 3, 13",
                     holds=f'{v("p", "i")} pass rate of task <i>i</i>; {v("k")} tries.',
                     ex=f'{A}{v("p", "i")} = {_pct(pools.pass_rate)}, {v("k")} = ' + ", ".join(str(k) for k in pools.pool_sizes) + " → " + ", ".join(_pct(pass_at[k]) for k in pools.pool_sizes) +
                        f'. Half at {_pct(pools.easy_pass_rate)}, half at 0 → {n(_pct(mixed))} at 8.',
                     so='samples cannot fix a task that never passes.'),
            ], "Ch 2–3"),
            Col("2 Will the context fit, and at what price?", [
                Card("Context budget", 'fixed + evidence ≤ window − output reserve', chapter="Ch 4",
                     ex=f'{fmt(cb.window)} window − {A}{fmt(cb.output_reserve)} output reserve − {fmt(fixed)} fixed (instructions, tools, state) = {n(fmt(obs_budget))} for evidence.',
                     so=f'appending {A}{fmt(cb.observation_per_turn)} per turn fills it at turn {turns_fit + 1} of {cb.turns}; select evidence instead.'),
                Card("Prefix caching prices the resend",
                     f'$/turn = {v("c", "cache")}({v("S")} − Δ{v("S", "turn")}) + {v("c", "in")}Δ{v("S", "turn")} + {v("c", "out")}{v("K")}', chapter="Ch 4, 19",
                     holds="From turn 2, when the prior request is a byte-identical prefix.",
                     ex=f'Summed over the {H_rs} turns of card 1·2, {v("K")} = {K_rs}, at {A}{usd_m(bill.input_price)}/{usd_m(bill.input_price * bill.cached_fraction)}/{usd_m(bill.output_price)} per M: '
                        f'${bill_fresh:.2f} fresh, {n(f"${bill_cached:.2f}")} cached.',
                     so=f'{bill_fresh / bill_cached:.1f}× cheaper: keep stable content first so the prefix stays identical.'),
                Card("KV state of one trajectory",
                     f'{v("M", "KV")} = 2 · {v("N", "L")} · {v("H", "KV")} · {v("d", "head")} · {v("s", "elem")} · {v("S")}', chapter="Ch 5",
                     holds=f'{v("N", "L")} layers, {v("H", "KV")} KV heads, {v("d", "head")} head dimension.',
                     ex=f'70B FP16: 2 · {NL} · {HKV} · {DH} · 2 B = {fmt(m_tok / 1024)} KiB/token; × {fmt(S_ctx)} = {n(f"{m_req / 1e9:.1f} GB")}. '
                        f'{kv.accelerators} H100s: {kv.accelerators} × {hbm_gib:g} GiB = {hbm_gb:.0f} GB − {w_bytes / 1e9:.0f} weights − {A}{kv.reserve.to("GB").magnitude:g} reserve = {kv_pool / 1e9:.0f} GB ÷ {m_req / 1e9:.1f} → {n(f"{max_req}")}.',
                     so='a trajectory waiting on a tool still holds its share.'),
            ], "Ch 4–5"),
            Col("3 What does a tool call risk?", [
                Card("Blind retry duplicates effects",
                     f'{v("p", "dup")} = (1 − {v("q", "req")})<sup>2</sup> · {v("q", "rep")}',
                     chapter="Ch 7", holds=f'{v("q", "req")}, {v("q", "rep")} request, reply loss; {v("p", "dup")} chance a retry repeats the effect.',
                     ex=f'{A}0.98² · 0.02 = {_pct(p_dup, 2)} × {fmt(br_calls)} calls/day = {n(fmt(dups_day))}.',
                     so='settle before retrying: an idempotency key or a status probe.'),
                Card("Sandbox pool: mean leases plus a ready reserve",
                     [f'{v("N", "active")} = {v("λ")} · {v("T", "hold")} &nbsp;(mean)',
                      f'Pr(Poisson({v("λ")}{v("T", "init")}) ≥ {v("N", "reserve")}) ≤ miss'],
                     chapter="Ch 8",
                     holds=f'{v("λ")} arrivals/s; {v("T", "hold")} lease, {v("T", "init")} restore; miss: none free.',
                     ex=f'{A}{lam:g}/s × {fmt(t_hold)} s = {n(fmt(n_active))} busy; {lam:g} × {t_init:g} s = {lt:.2f} → miss ≤ {_pct(sp.target_miss, 1)} needs {n(str(reserve))}.',
                     so='size for the mean plus a small reserve for the tail.'),
                Card("Injection exposure grows with reads", f'Pr(exposed) = 1 − (1 − {v("q", "inj")})<sup>{v("n", "read")}</sup>', chapter="Ch 9",
                     holds=f'{v("q", "inj")} payload share of read items; {v("n", "read")} untrusted reads per task.',
                     ex=f'{A}{v("q", "inj")} = {_pct(q_pay)}: 1 − 0.99<sup>{inj.reads["repair"]}</sup> = {n(_pct(expo["repair"]))}; 1 − 0.99<sup>{inj.reads["research"]}</sup> = {n(_pct(expo["research"]))} ({_pct(expo_scr, 1)} screened).',
                     so='screens lower the rate; only containment bounds the effect.'),
            ], "Ch 7–9"),
        ],
        bottom=Table(["Llama 3 70B", f"{v('P')} = {P / 1e9:.1f}B", f"{v('N', 'L')} = {NL}", f"{v('H', 'KV')} = {HKV}", f"{v('d', 'head')} = {DH}", f"context {fmt(S_ctx)}"], [
            ["NVIDIA H100", f"{v('R', 'peak')} = {fmt(R_PEAK / 1e12)} TFLOP/s FP8", f"BW = {BW / 1e12:.2f} TB/s", f"{hbm_gib:g} GiB HBM",
             f"in: ≥ {math.floor(t_pre * 1e6) / 1e3:.3f} ms, ≤ {fmt(math.ceil(tok_in_s))}/s", f"out: ≥ {math.floor(t_dec * 1e4) / 10:.1f} ms, ≤ {fmt(math.ceil(tok_out_s))}/s"],
            [f"{A}Prices, $ per M tokens", f"{v('c', 'in')} = ${c_in * 1e6:g}", f"{v('c', 'cache')} = ${c_cache * 1e6:.2f}", f"{v('c', 'out')} = ${c_out * 1e6:g}",
             f"cached ÷ input = {price_cache_ratio:g}×", f"output ÷ input = {fmt(price_out_ratio)}×"],
        ], caption=f'Per token: 70B 8-bit weights, one H100, batch 1 (time floors, rate ceilings). GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B. '
                   f'{TAG}= example input; examples are the book\'s hypothetical scenarios.'),
    )

    # ================================================================ page 2
    p2 = Page(
        question="Is it worth running?",
        sub="Read left to right: containing failures, evidence to ship, and whether more agents pay. The table prices the book's four workloads.",
        meta="the four lighthouse workloads",
        notation=[
            ("Symbols", f'{v("N")} tasks · {v("p")}, {v("p", "i")} pass rate of one run, of task <i>i</i> · {v("k")} runs · {v("z")} = 1.96 (95%, two-sided) · '
                        f'{v("p", "acc")} acceptance per attempt · {v("C", "eff")} $ per accepted task'),
        ],
        cols=[
            Col("4 Can the runtime contain a failure?", [
                Card("Admit a call only if its worst case fits",
                     [f'committed + liability ≤ allowance', f'liability = {v("c", "in")} · {v("S")} + {v("c", "out")} · {v("K", "max")}'],
                     chapter="Ch 10",
                     holds=f'Committed = settled + reserved + final-check reserve; {v("c", "in")}, {v("c", "out")} $/token; {v("S")} input, {v("K", "max")} output-cap tokens.',
                     ex=f'{A}{fmt(ad.input_tokens)} · {usd_m(ad.input_price)}/M + {fmt(ad.output_allowance)} · {usd_m(ad.output_price)}/M = ${liab:.2f}; '
                        f'${settled:.2f} + ${reserved:.2f} + ${fin:.2f} + ${liab:.2f} = ${committed:.2f} &gt; ${allow:.2f} → {n("deny", True)}.',
                     so='a spend-only check would admit it.'),
                Card("Checkpoint interval (Young)", [f'{v("T", "ckpt", "*")} ≈ √(2 · {v("T", "write")} · MTBF)', f'overhead ≈ {frac(v("T", "write"), v("T", "ckpt"))} + {frac(v("T", "ckpt"), "2 MTBF")}'], chapter="Ch 11",
                     holds=f'{v("T", "write")} pause per checkpoint; MTBF between work-losing failures.',
                     ex=f'{A}√(2 · {t_w:g} s · {fmt(mtbf)} s) = {n(f"{fmt(t_star)} s")}; overhead {_pct(young_write, 1)} + {_pct(young_rework, 1)} = {_pct(waste, 1)}.',
                     so='checkpoint more often only if writes get cheaper.'),
                Card("Repair first, or unwind?",
                     f'E[cost] = {v("C", "fix")} + (1 − {v("p", "fix")}) · {v("C", "fallback")}',
                     chapter="Ch 12",
                     holds=f'{v("p", "fix")} success; {v("C", "fix")}, {v("C", "fallback")} repair, unwind-and-rebuild cost.',
                     ex=f'{A}{v("C", "fix")} = {rv.startup_time.to("s").magnitude:g} + {rv.repair_output}/{rv_rate:g} + {rv.validation_time.to("s").magnitude:g} = {t_rep:.2f} s; '
                        f'{v("C", "fallback")} = {rv.unwind_time.to("s").magnitude:g} + {rv.rebuild_time.to("s").magnitude:g} = {t_fb:.1f} s; break-even {v("p", "fix")} = {t_rep:.2f}/{t_fb:.1f} = {n(_pct_up(pstar_t, 2))}.',
                     so='repair first whenever it succeeds more often.'),
                Card("Expected irreversible loss", f'{v("L", "A")} = Σ<sub><i>a</i></sub> Pr(bad <i>a</i>) · impact(<i>a</i>) · (1 − recovered(<i>a</i>))', chapter="Ch 12",
                     ex=f'{A}{_pct(wk.wrong_change_probability)} × ${fmt(impact)} = ${fmt(l0)}; canary catches {_pct(wk.canary_catch_rate)}, reverts {_pct(wk.canary_recovery_fraction)}: '
                        f'${fmt(l0)} × ({wk.canary_catch_rate:g} × {1 - wk.canary_recovery_fraction:.1f} + {1 - wk.canary_catch_rate:.1f}) = {n(f"${l1:.0f}")}.',
                     so=f'still {l1 / pv_att:.1f}× the ${pv_att:.2f} attempt: loss prices it.'),
            ], "Ch 10–12"),
            Col("5 Is the evidence enough to ship?", [
                Card("Wilson interval on an acceptance rate",
                     [f'{frac(v("X") + " + " + v("z", sup="2") + "/2", v("N") + " + " + v("z", sup="2"))} ± {frac(v("z"), v("N") + " + " + v("z", sup="2"))}√({frac(v("X") + "(" + v("N") + " − " + v("X") + ")", v("N"))} + {frac(v("z", sup="2"), "4")})'],
                     chapter="Ch 13", holds=f'{v("X")} accepted of {v("N")}.',
                     ex=f'{A}{v("N")} = {Nt}: {v("X")} = 0 → [0, {_pct_up(wil[0][1])}]; {v("X")} = 2 → [{_pct_dn(wil[2][0])}, {_pct_up(wil[2][1])}]; {v("X")} = {wi.successes[2]} → '
                        f'±{(wil[wi.successes[2]][1] - wil[wi.successes[2]][0]) / 2 * 100:.1f} points.',
                     so=f'{Nt} tasks cannot tell 60% from 70%.'),
                Card("Tasks to detect a change", f'{v("N", "arm")} ≈ {frac("2(" + v("z") + " + " + v("z", "β") + ")<sup>2</sup> " + v("p") + "(1 − " + v("p") + ")", v("δ", sup="2"))}',
                     chapter="Ch 13",
                     holds=f'{v("N", "arm")} tasks per independent arm; {v("z", "β")} = {pl.z_beta:g} for 80% power; {v("δ")} difference to detect.',
                     ex=f'{A}{v("p")} = {pl.rate:g}, {v("δ")} = {pl.differences[1]:g}: 2 · 2.8² · 0.25 / {pl.differences[1]:g}² = {n(fmt(per_arm[pl.differences[1]]))}; {v("δ")} = {pl.differences[0]:g}: {fmt(per_arm[pl.differences[0]])}.',
                     so='halving δ quadruples the tasks; pair tasks across arms to need fewer.'),
                Card("Verifier precision: share of passes that are valid",
                     f'Π = {frac(v("p", "valid") + "(1 − " + v("ε", "FR") + ")", v("p", "valid") + "(1 − " + v("ε", "FR") + ") + (1 − " + v("p", "valid") + ")" + v("ε", "FA"))}',
                     chapter="Ch 3",
                     holds=f'{v("p", "valid")} valid share of candidates; {v("ε", "FA")}, {v("ε", "FR")} false-accept, false-reject rates.',
                     ex=f'{A}{_pct(vp.valid_rate)} · 0.9 / ({_pct(vp.valid_rate)} · 0.9 + 95% · {_pct(vp.false_accept_rate)}) = {n(_pct(prec))}; {v("ε", "FA")} = {_pct(vp.false_accept_rate / 2)} → {_pct(prec_half)}.',
                     so='when valid answers are rare, the false-accept rate dominates.'),
                Card("pass^k: all k runs pass", f'pass<sup>{v("k")}</sup> = E<sub><i>i</i></sub>[{v("p", "i", v("k"))}] &nbsp; vs &nbsp; pass@{v("k")} = 1 − E<sub><i>i</i></sub>[(1 − {v("p", "i")})<sup>{v("k")}</sup>]', chapter="Ch 13",
                     ex=f'{A}{v("p", "i")} = {_pct(rc.step_success)}, {v("k")} = {rc.long_horizon}: 0.9<sup>8</sup> = {n(_pct(pk), True)}, below a {_pct(rc.target_success)} target, which needs {v("p", "i")} ≥ 0.9<sup>1/8</sup> = {_pct_up(p_need)}.',
                     so='recurring work needs reliability per run, not one lucky pass.'),
            ], "Ch 3, 13"),
            Col("6 Does adding agents pay?", [
                Chart(sp_plot.svg("Speedup against workers with and without coordination overhead")),
                Card("Delegation speedup with coordination cost",
                     f'speedup = 1 / ((1 − {v("f")}) + {v("f")}/{v("N", "w")} + {v("h")}({v("N", "w")} − 1))',
                     chapter="Ch 18", holds=f'{v("N", "w")} workers, {v("f")} parallel fraction, {v("h")} overhead per added worker.',
                     ex=f'{A}{v("f")} = {f_par:g}, {v("h")} = {h_w:g}, {v("N", "w")} = {best_n}: 1 / ({1 - f_par:.1f} + {f_par:g}/{best_n} + {h_w:g} · {best_n - 1}) = {n(f"{best_s:.2f}×")}.',
                     so='past the peak, each added agent slows the task.'),
                Card("A token multiple must buy acceptance",
                     f'{frac(v("C", "eff") + " multi", v("C", "eff") + " single")} = token multiple × {frac(v("p", "acc") + " single", v("p", "acc") + " multi")}', chapter="Ch 18",
                     holds="With cost proportional to tokens.",
                     ex=f'{fmt(dl.multi_agent_tokens_vs_chat)}× vs {fmt(dl.agent_tokens_vs_chat)}× chat tokens (Hadfield et al., 2025) → {v("p", "acc")} must rise {m_tok_mult:.2f}×. '
                        f'As {v("p", "acc")} multi ≤ 100%, a single agent above 1/{m_tok_mult:.2f} = {_pct_up(max_single)} cannot be beaten.',
                     so='delegate only tasks a single agent usually fails.'),
                Card("The token-price fallacy", f'{v("C", "eff")} = cohort cost / accepted tasks', chapter="Ch 19",
                     ex=f'{A}Frontier: {fb.turns} turns × ${fb_turn:.2f} + ${_usd(fb.assessment_per_attempt):.2f} check = ${fr_att:.2f} ÷ {fb.accepted / fb.tasks:.2f} → ${fr_eff:.2f}. '
                        f'Small model ({usd_m(al.input_price)} in, {usd_m(al.output_price)} out per M): {al.turns} × ${al_turn:.2f} + ${_usd(al.assessment_per_attempt):.2f} = ${lw_att:.2f} ÷ {al.accepted / al.tasks:.2f} → ${lw_eff:.2f}.',
                     so=f'tokens {_pct(tok_cut)} cheaper, accepted task {lw_eff / fr_eff:.1f}× dearer.'),
            ], "Ch 18–19"),
        ],
        bottom=Table(["Reference workloads (Ch 1, 19)", "turns", "untrusted reads", "Pr(exposed) at 1%", "C<sub>attempt</sub>", "p<sub>acc</sub>", "C<sub>eff</sub>", "largest cost term"], [
            [f"{k.capitalize()} agent ({archetype[k]})", str(horizon[k]), str(inj.reads[k]), _pct(expo[k]),
             f"${rows_cost[k]['att']:.2f}", _pct(rows_cost[k]["acc"]), f"${rows_cost[k]['eff']:.2f}",
             f"{rows_cost[k]['largest']} ({_pct(rows_cost[k]['share'])})"]
            for k in ("repair", "research", "remediation", "support")]),
    )

    # ================================================================ page 3
    p3 = Page(
        question="Can it be cheaper and faster?",
        sub="Read each column top to bottom: levers on the model call, on the context and capacity, and on the critical path.",
        meta="levers on cost and time",
        notation=[
            ("Symbols", f'{v("p")} pass rate of one attempt · {v("T")} time · {v("ρ")} = {v("λ")}{v("t", "s")} server utilization, {v("λ")} arrivals/s, {v("t", "s")} service time · '
                        f'{v("c", "in")}, {v("c", "out")} $ per input, output token'),
        ],
        cols=[
            Col("7 Can each call cost less?", [
                Card("Retry a cheap model, or call a strong one?", f'{v("C", "eff")} = {frac("$ per attempt", v("p"))}, &nbsp; coverage = 1 − (1 − {v("p")})<sup>{v("k")}</sup>', chapter="Ch 19",
                     holds=f'Independent sequential attempts, at most {v("k")}.',
                     ex=f'{A}Cheap ${_usd(rt.attempt_cost):.2f} / {rt.pass_probability:g} = {n(money(rt_cheap))}; strong ${_usd(rt.strong_attempt_cost):.2f} / {rt.strong_pass_probability:g} = {money(rt_strong)}. '
                        f'Cap {v("k")} = {rt.max_attempts}: 1 − 0.7<sup>5</sup> = {_pct(rt_cover)} covered.',
                     so='cheap retries win on price; the cap, not the price, limits coverage.'),
                Card("Model cascade", 'E[$ per task] = Σ<sub><i>j</i></sub> Pr(reach tier <i>j</i>) · (call<sub><i>j</i></sub> + gate)', chapter="Ch 19",
                     holds='A task reaches tier <i>j</i> only if the gate rejected every cheaper tier.',
                     ex=f'{A}Reach ' + ", ".join(_pct(r) for r in cs_r.reach_shares) + ' × (' + ", ".join(f"${c:.4f}" for c in tier_cost) +
                        f') = ${cs_task:.5f}; ÷ {_pct(cs_acc, 1)} accepted = {n(f"${cs_eff:.4f}")} vs frontier alone ${tier_cost[-1]:.3f} ÷ {_pct(cs.direct_accepted / cs.tasks)} = ${cs_direct:.4f}.',
                     so=f'{cs_direct / cs_eff:.1f}× cheaper, if every exit applies the same gate.'),
                Card("Own or rent: break-even utilization", f'{v("U", sup="*")} = {frac("owned $/h", "3600 · tokens/s · Δ$ per token")}', chapter="Ch 19",
                     holds='Δ$ = API price − owned marginal cost, per output token; tokens/s at full load.',
                     ex=f'{A}API per output token, {de.input_per_output:g} input tokens each ({_pct(de.cached_fraction)} cached): {de.input_per_output:g} × ${p_in_w * 1e6:.2f} + ${_per_tok(de.output_price) * 1e6:g} = ${api_out * 1e6:.2f}/M. '
                        f'${fixed_h:g} / (3600 · {fmt(rate)} · (${api_out * 1e6:.2f} − ${_per_tok(de.variable_price) * 1e6:g})/M) = {n(_pct_up(u_star, 1))}.',
                     so='below that load, renting is cheaper.'),
                Card("Speculative decoding", f'speedup = {frac(v("a") + " + 1", v("γ") + v("c", "d") + " + 1")}', chapter="Ch 19",
                     holds=f'{v("γ")} drafted tokens per round, {v("c", "d")} draft cost per target step, {v("a")} drafts kept per round.',
                     ex=f'{A}{v("γ")} = {sd.draft_length}, {v("c", "d")} = {sd.draft_cost_ratio:g}: {v("a")} = {sd.accepted_high:g} → {n(f"{sd_hi:.2f}×")}; {v("a")} = {sd.accepted_low:g} → {sd_lo:.2f}× (slower). '
                        f'Model calls are {ep_model:.1f} of {ep_total:.1f} s, so the episode gains ≤ {ep_bound:.2f}×.',
                     so='speed up the term that dominates; here it is the tests.'),
                Note("Price and latency are different accounts.", 'batching evens out input and output work, but a trajectory still waits on each output token (Ch 2).'),
            ], "Ch 19"),
            Col("8 Can the context shrink?", [
                Chart(cwp.svg("Request length over turns with compaction at a high watermark"), caption=f"Dashed: {cw.trigger / 1000:g}K trigger. Accent: target {cw.target / 1000:g}K, a pass every {cyc} turns. Brown: target {cw.alternate_target / 1000:g}K, every {cyc_alt}."),
                Card("Compaction cadence", 'cycle = ⌈(trigger − target) / appended per turn⌉', chapter="Ch 4",
                     ex=f'{A}(({fmt(cw.trigger)} − {fmt(cw.target)}) / {fmt(cw.tokens_per_turn)}) → {n(str(cyc) + " turns")}; target {fmt(cw.alternate_target)}: {cyc_alt} turns.',
                     so=f'a narrow gap compacts {cyc / cyc_alt:.0f}× as often, and each pass can drop a constraint.'),
                Card("Select tools per call", f'select if unused tools × tokens each &gt; selector', chapter="Ch 7",
                     ex=f'{A}{tcat.registered} × {tcat.descriptor_tokens} = {fmt(tc_full)} tokens vs {tcat.selected} × {tcat.descriptor_tokens} + {tcat.selection_overhead} = {fmt(tc_sel)}.',
                     so=f'{fmt(tc_full - tc_sel)} fewer input tokens on every call.'),
                Card("Shape observations before they enter context", 'carried tokens = shaped tokens × later calls', chapter="Ch 7",
                     ex=f'{A}{ob.raw_bytes.to("MB").magnitude:g} MB log / {ob.bytes_per_token.magnitude:g} B per token = {fmt(raw_tok / 1e6)}M tokens = {raw_win:g} windows; '
                        f'a {fmt(ob.shaped_tokens)}-token view × {ob.carry_calls} calls = {fmt(carried)} tokens × {usd_m(ob.input_price)}/M = {n(f"${carried_usd:.2f}")}.',
                     so='the raw log never fits; even the shaped view is paid on every later call.'),
                Card("Vector index memory", 'bytes = vectors × bytes per vector', chapter="Ch 6",
                     holds='HNSW adds neighbor links per vector; IVF-PQ keeps one byte per subvector.',
                     ex=f'{A}{fmt(an.vectors / 1e6)}M × {an.dimension} × 2 B = {an_raw / 1e9:.2f} GB raw; + {an.links_per_node} links × 8 B = {an_graph / 1e9:.2f} GB (HNSW); '
                        f'{an.subvectors} one-byte codes = {n(f"{an_codes / 1e9:.2f} GB")} (IVF-PQ). An exact scan of the raw vectors at {fmt(cpu_bw / 1e9)} GB/s: {fmt(vs_t * 1e3)} ms.',
                     so=f'PQ codes are {an_raw / an_codes:.0f}× smaller than raw vectors and fit in RAM; the graph adds memory.'),
            ], "Ch 4, 6, 7"),
            Col("9 Can it finish sooner?", [
                Card("Dependency graph: makespan", f'{v("T")} ≥ max(work / slots, critical path)', chapter="Ch 18",
                     ex=f'{A}Nodes ' + ", ".join(f"{dg_d[k]:g}" for k in sorted(dg_d)) + f' s on {dg.slots} slots: work {dg_work:g} / {dg.slots} = {dg_work / dg.slots:.1f} s; '
                        f'path ' + " → ".join(str(k) for k in path) + ' = ' + " + ".join(f"{dg_d[k]:g}" for k in path) + f' = {n(f"{dg_cp:g} s")}.',
                     so=f'only shortening the critical path helps; {dg_work / dg_cp:.2f}× over serial.'),
                Chart(kp.svg("Mean queue wait against utilization, Poisson against bursty traffic"), caption=f"Solid: bursty, (c²ₐ + c²ₛ)/2 = {kq_v:g}. Dashed: M/M/1. {v('t', 's')} = {t_s1:g} s."),
                Card("Queueing with bursty traffic (Kingman)", f'E[wait] ≈ {frac("c²ₐ + c²ₛ", "2")} · {frac(v("ρ"), "1 − " + v("ρ"))} · {v("t", "s")}', chapter="Ch 19",
                     holds='c²ₐ, c²ₛ squared variation of arrivals and service (1 for M/M/1).',
                     ex=f'{A}c²ₐ = {kq.arrival_scv:g}, c²ₛ = {kq.service_scv:g}, {v("t", "s")} = {t_s1:g} s: {kq_v:g} × {kq.low_utilization:g}/{1 - kq.low_utilization:.1f} = {n(f"{fmt(kq_lo)} s")} at {v("ρ")} = {kq.low_utilization:g}; '
                        f'{kq_v:g} × {kq.high_utilization:g}/{1 - kq.high_utilization:.1f} = {fmt(kq_hi)} s at {kq.high_utilization:g}.',
                     so=f'bursts multiply the M/M/1 wait by {kq_v:g}; keep {v("ρ")} low.'),
                Card("Stop an undetected loop", f'cost = {v("c", "in")}[{v("H")}{v("S", "base")} + Δ{v("S", "turn")}{frac(v("H") + "(" + v("H") + " − 1)", "2")}] + {v("c", "out")}{v("H")}{v("K")}', chapter="Ch 12",
                     holds=f'{v("H")} turns, {v("S", "base")} first input, Δ{v("S", "turn")} appended per turn, {v("K")} output per turn.',
                     ex=f'{A}{v("S", "base")} = {fmt(lp.first_input)}, Δ{v("S", "turn")} = {fmt(lp_d)}, {v("K")} = {lp.output_per_turn}, {usd_m(lp.input_price)}/{usd_m(lp.output_price)} per M: '
                        f'{v("H")} = {lp.ceiling} → {n(f"${lp_ceiling:.2f}")}; {v("H")} = {lp.stop_after} → ${lp_stop:.2f}.',
                     so=f'a detector at turn {lp.stop_after} saves {lp_ceiling / lp_stop:.0f}×.'),
                Card("Retry storms", 'calls = (tries per layer)<sup>layers</sup>', chapter="Ch 12",
                     ex=f'{A}{ra.attempts_per_layer}<sup>{layers}</sup> = {n(fmt(amp))} calls per failure. Rate = retries / window: {jt.retries} in {jt.burst_window.to("ms").magnitude:g} ms = {fmt(jt_burst)}/s; jittered over {jt.jitter_span.to("s").magnitude:g} s, {fmt(jt_spread)}/s.',
                     so='retry in one layer only, with jitter.'),
            ], "Ch 12, 18, 19"),
        ],
    )

    # ================================================================ page 4
    p4 = Page(
        question="Can it be trusted and improved?",
        sub="Read left to right: release evidence, a clean training signal, then what training on trajectories costs.",
        meta="evidence, data, and training",
        notation=[
            ("Symbols", f'{v("N")} tasks or pairs · {v("p")} pass rate of one rollout · {v("z")} = 1.96 (95%, two-sided) · {v("α")}, {v("β")} false-hold, missed-degradation rates'),
        ],
        cols=[
            Col("10 Is the release safe?", [
                Card("Zero failures is not zero risk", f'rate ≤ 1 − 0.05<sup>1/{v("n")}</sup> ≈ {fmt(rule)}/{v("n")}', chapter="Ch 9",
                     holds=f'95% one-sided bound after {v("n")} trials with no failure.',
                     ex=f'{A}{v("n")} = {za.trials}: 1 − 0.05<sup>1/{za.trials}</sup> = {n(_pct_up(za_up, 2))} ({fmt(rule)}/{za.trials} = {_pct_up(rule / za.trials, 1)}). ≤ {_pct(za.target_rate)} needs {fmt(za_need)} clean runs.',
                     so='a clean suite of 50 only rules out rates above 6%.'),
                Card("Paired comparison", f'Δ = {frac(v("b") + " − " + v("c"), v("N"))} ± {v("z")}√({frac("(" + v("b") + " + " + v("c") + ")/" + v("N") + " − Δ²", v("N"))})', chapter="Ch 13",
                     holds=f'{v("b")}, {v("c")} tasks only the candidate, only the baseline accepts.',
                     ex=f'{A}{v("N")} = {pc_n}, {v("b")} = {pc.candidate_only}, {v("c")} = {pc.baseline_only}: Δ = {pc_d * 100:.1f} ± {math.ceil(pc_hw * 1000) / 10:.1f} points.',
                     so='the interval spans 0: inconclusive; only the b + c split tasks carry signal.'),
                Card("Sequential canary (SPRT)", [f'hold when Σ log-ratios ≥ ln{frac("1 − " + v("β"), v("α"))}',
                                                  'time ≈ E[samples] / canary task rate + label delay'], chapter="Ch 14",
                     ex=f'{A}{v("α")} = {sq.alpha:g}, {v("β")} = {sq.beta:g}: bound {sq_A:.2f}; a failure adds ln(0.25/0.10) = {sq_fail:.2f}, a pass {sq_succ:.2f}. '
                        f'{sq_r.expected_samples:.0f} samples at {_pct(sq.fraction)} of {sq.arrival_rate.to("1/hour").magnitude:g}/h + {sq.label_delay.to("hour").magnitude:g} h = {n(f"{sq_hours:.1f} h")}.',
                     so=f'doubling the canary share cuts it to {sq2_hours:.1f} h; evidence takes calendar time.'),
                Card("Head sampling misses rare failures", f'Pr(no trace kept) = (1 − {v("θ")})<sup>{v("n", "f")}</sup>', chapter="Ch 14",
                     holds=f'{v("θ")} share of traces kept at random; {v("n", "f")} occurrences.',
                     ex=f'{A}{v("θ")} = {_pct(hs.retention_probability)}, {v("n", "f")} = {hs.occurrences}: 0.9<sup>5</sup> = {n(_pct(hs_miss))}.',
                     so='keep traces by verdict, not at random.'),
                Note("Before you ship on a number:", f'is its interval narrower than the change? Were runs split by task? Are rare passes verified? Is a clean suite big enough ({fmt(rule)}/<i>n</i>)?'),
            ], "Ch 9, 13, 14"),
            Col("11 Is the training signal clean?", [
                Card("Split by task, not by run", f'Pr(all {v("k")} runs on one side) = {v("f", "tr", v("k"))} + (1 − {v("f", "tr")})<sup>{v("k")}</sup>', chapter="Ch 15",
                     holds=f'{v("f", "tr")} chance a run goes to training; {v("k")} runs per task.',
                     ex=f'{A}{v("f", "tr")} = {rs_.train_fraction:g}, {v("k")} = {rs_.attempts}: 0.8<sup>8</sup> + 0.2<sup>8</sup> = {_pct(rs_whole, 1)}.',
                     so=f'{n(_pct(1 - rs_whole))} of tasks leak across the split.'),
                Card("MinHash near-duplicate screen", f'Pr(candidate) = 1 − (1 − {v("J", sup=v("r"))})<sup>{v("b")}</sup>', chapter="Ch 15",
                     holds=f'{v("J")} Jaccard similarity; {v("b")} bands of {v("r")} rows.',
                     ex=f'{A}{v("b")} = {mh.bands}, {v("r")} = {mh.rows}: {v("J")} = 0.5 → {_pct(mh_p[0.5])}; {v("J")} = 0.8 → {n(_pct_dn(mh_p[0.8], 1))}. '
                        f'Index: {mh.bands * mh.rows} × {mh.hash_bytes.magnitude:g} B × {fmt(mh.records)} = {mh_bytes / 1e6:.1f} MB.',
                     so='near-duplicates are caught almost surely; raise r to pass loose pairs.'),
                Card("Order admission checks", 'E[time] = Σ<sub><i>i</i></sub> <i>t</i><sub><i>i</i></sub> Π<sub><i>j</i>&lt;<i>i</i></sub> <i>p</i><sub><i>j</i></sub>; run lower <i>t</i>/(1 − <i>p</i>) first', chapter="Ch 15",
                     holds='<i>t</i> slot time and <i>p</i> pass rate of each check.',
                     ex=f'{A}' + ", ".join(f"{k} {tt.to('s').magnitude:g} s/{_pct(pp)}" for k, (pp, tt) in ao.checks.items()) +
                        f'. Audit before tests: {n(f"{ao_t[0]:.2f} s")}; after: {ao_t[1]:.2f} s.',
                     so='cheap, selective checks go first.'),
                Card("Cost per emitted artifact", 'cost = cohort cost / artifacts emitted', chapter="Ch 15",
                     ex=f'{A}${_usd(cd.cohort_cost):g} for {cd_total} attempts, {cd_emit} emitted: {n(money(cd_cost))} each.',
                     so='yield, not generation, sets the price of data.'),
                Card("False accepts in the reward", f'share = {frac("(1 − " + v("p") + ")" + v("ε", "FA"), v("p") + " + (1 − " + v("p") + ")" + v("ε", "FA"))}', chapter="Ch 17",
                     holds=f'{v("ε", "FA")} verifier false-accept rate.',
                     ex=f'{A}{v("p")} = {_pct(fa.pass_rate)}, {v("ε", "FA")} = {_pct(fa.false_accept_rate, 1)}: 0.9 × 0.025 / (0.1 + 0.0225) = {n(_pct(fa_share))} of rewards go to failures.',
                     so='at low pass rates a small verifier error becomes a large reward error.'),
            ], "Ch 15, 17"),
            Col("12 What does training cost?", [
                Card("Sequence packing", 'occupancy = stored tokens / buffer positions', chapter="Ch 16",
                     holds='Padding positions cost compute and train nothing.',
                     ex=f'{A}{len(pk_.lengths)} examples, {fmt(pk_store)} tokens: padded to {len(pk_.lengths)} × {fmt(pk_.capacity)} → {_pct(pk_pad)}; '
                        f'packed into {pk_r.buffers} × {fmt(pk_.capacity)} → {n(_pct(pk_pack))}.',
                     so=f'{pk_r.position_ratio:g}× fewer positions; mask attention per example.'),
                Card("Low-rank adapters", f'trained parameters = {v("r")}({v("d", "in")} + {v("d", "out")}) per matrix', chapter="Ch 16",
                     holds=f'{v("r")} adapter rank; {v("d", "in")}, {v("d", "out")} input, output width of an adapted matrix.',
                     ex=f'{A}{v("r")} = {lo.adapter_rank}, {lo.adapted_matrices} {fmt(lo.adapter_in_dim)}² matrices × {lo.workload.layers} layers: {lo.adapter_rank} · {fmt(lo.adapter_in_dim * 2)} · {lo.adapted_matrices * lo.workload.layers} = {fmt(lo_n / 1e6, 1)}M; '
                        f'× {b_train:g} B = {n(f"{lo_state / 1e6:.0f} MB")} vs {full_state / 1e9:.0f} GB full.',
                     so=f'adapters remove the optimizer state; the {frozen / 1e9:.0f} GB of FP16 weights stay.'),
                Card("DPO: how hard one pair pulls", f'margin = {v("β")}[(log π − log π<sub>ref</sub>)<sub>w</sub> − (…)<sub>l</sub>]',
                     chapter="Ch 16", holds=f'w, l preferred, dispreferred; gradient weight = {v("β")} σ(−margin), σ logistic.',
                     ex=f'{A}{v("β")} = {dp.beta:g}: {dp.beta:g} × ({dp_w:g} − ({dp_l:g})) = {dp_m:.1f}; weight {n(f"{dp_g:.3f}")} vs 0.1 at margin 0.',
                     so='a pair the policy already prefers pulls less.'),
                Card("Group-relative advantage", 'Â = (reward − group mean) / group std', chapter="Ch 17",
                     holds=f'{v("G")} rollouts per task; Pr(mixed group) = 1 − {v("p", sup=v("G"))} − (1 − {v("p")})<sup>{v("G")}</sup>.',
                     ex=f'{A}1 pass in {g16}: (1 − {mu16:.4f}) / {sd16:.3f} = {n(f"{adv16:.2f}")}. {v("G")} = {gr.group}, {v("p")} = {_pct(gr.pass_rate)}: {_pct(g_c)} of groups mixed.',
                     so='target tasks near 50% pass; verify rare passes well.'),
                Card("Adaptation payback", 'break-even tasks = investment / saving per task', chapter="Ch 19",
                     holds='Saving per task = model cost per task before − after adapting.',
                     ex=f'{A}${fmt(invest)} / (${c_base:.4f} − ${c_adapt:.4f}) = {n(fmt(n_be) + " tasks")}, {months:.1f} months at {fmt(ap.monthly_volume)}/month.',
                     so='adapt only for steady, high-volume work.'),
            ], "Ch 16–17, 19"),
        ],
    )
    return Sheet(volume=3, book="Agentic Machine Learning Systems", scope="The Trajectory", pages=[p1, p2, p3, p4],
                 sources=f'{TAG}= example input · GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · examples are the book\'s mlsysim scenarios: hypothetical inputs, not measurements · prices are the book\'s hypothetical $3 / $0.30 / $15 per M tokens')
