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
    rows_cost["repair"] = dict(model=bill_cached, env=rp["environment_per_turn"] * rp["turns"], assess=rp["assessment"], human=rp["human"], acc=rp["acceptance"])
    rr = W["research"]
    chat = rr["chat_input"] * c_in + rr["chat_output"] * c_out
    rows_cost["research"] = dict(model=chat * rr["token_multiple"], env=rr["environment"], assess=rr["assessment"], human=rr["human"], acc=rr["acceptance"])
    rm = W["remediation"]
    rows_cost["remediation"] = dict(model=rm["model"], env=rm["environment"], assess=rm["assessment"], human=rm["human"], acc=rm["acceptance"])
    su = W["support"]
    rows_cost["support"] = dict(model=su["model"], env=su["environment"], assess=su["assessment"], human=su["handoff"] * (1 - su["acceptance"]), acc=su["acceptance"])
    for k, d in rows_cost.items():
        d["att"] = _check(d["model"] + d["env"] + d["assess"] + d["human"], lh_r[k].attempt, f"{k} attempt")
        d["eff"] = _check(d["att"] / d["acc"], lh_r[k].cost_per_accepted, f"{k} C_eff")
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
    _check(1 - eff_half_tok / rep["eff"], lh_r["repair"].halve_tokens, "repair: halve token price")
    eff_more_acc = rep["att"] / (rep["acc"] + lever_gain)
    _check(1 - eff_more_acc / rep["eff"], lh_r["repair"].acceptance_gain, "repair: +10 points acceptance")

    tail = SC.ThirtyCallTailExposure
    tail_any = _check(1 - (1 - tail.tail_probability) ** tail.calls, tail.any_tail, "any p99 call")



    # ---- printed inputs that let a reader reproduce each result
    def usd_m(price) -> str:
        x = price.to("USD/Mtoken").magnitude
        return f"${x:g}" if x == int(x) else f"${x:.2f}"

    rep_eff = rep["eff"]
    fb, al = tc.baseline, tc.alternative
    rv_rate = rv.output_rate.to("token/s").magnitude
    tok_in_s, tok_out_s = 1 / t_pre, 1 / t_dec
    price_out_ratio = c_out / c_in
    price_cache_ratio = c_cache / c_in
    e_inv = math.exp(-1)
    _check(succ[hz.long_horizon], e_inv, "0.99^100 ≈ 1/e", rel=0.01)
    hbm_gib = kv.hbm_per_accelerator.to("GiB").magnitude
    log_bpt = ob.bytes_per_token.to("B/token").magnitude
    lw_att_cost = al.turns * (al.input_tokens_per_turn * _per_tok(al.input_price) + al.output_tokens_per_turn * _per_tok(al.output_price)
                              + _usd(al.environment_per_turn)) + _usd(al.assessment_per_attempt)
    _check(lw_att_cost, lw_att, "cheap-model attempt")

    # ================================================================ charts
    hp = Plot((1, 1000), (0, 1), xlog=True, W=300, H=92, left=30, right=10, top=8, bottom=21)
    hp.grid([1, 10, 100, 1000], [0, 0.5, 1.0], yfmt=lambda y: f"{y * 100:g}%")
    eps_lines = [(SC.ResidualViolationsFiftyTurns.step_violation, "line muted"), (eps, "line"),
                 (1 - SC.CompactionConstraintRetention.step_success, "line warn")]
    for e, cls in eps_lines:
        hp.line([(10 ** (i / 40), (1 - e) ** (10 ** (i / 40))) for i in range(121)], cls)
    hp.text(hp.X(250), hp.Y((1 - eps_lines[0][0]) ** 250) - 6, f"ε = {eps_lines[0][0] * 100:g}%", "start")
    hp.text(hp.X(150), hp.Y(0.55), f"ε = {eps * 100:g}%", "start")
    hp.text(hp.X(1.6), hp.Y(0.62), f"ε = {eps_lines[2][0] * 100:g}%", "start")
    for h in (hz.short_horizon, hz.long_horizon):
        hp.point(h, succ[h], _pct(succ[h]), "end", -5, 10)
    hp.axes("turns H (log scale)", "Pr(success)")

    wmax = SCENARIO["chart_workers"]
    sp_plot = Plot((1, wmax), (1, 6), W=250, H=70, left=24, right=8, top=8, bottom=21)
    sp_plot.grid([1, 4, 8, 12, 16], [1, 3, 5], yfmt=lambda y: f"{y:g}×")
    sp_plot.line([(nw, 1 / t_rel(nw, 0)) for nw in range(1, wmax + 1)], "line alt")
    sp_plot.line([(nw, 1 / t_rel(nw)) for nw in range(1, wmax + 1)], "line")
    sp_plot.text(sp_plot.X(wmax) - 2, sp_plot.Y(1 / t_rel(wmax, 0)) - 5, "Amdahl, no coordination cost", "end")
    sp_plot.point(best_n, best_s, f"peak {best_s:.1f}× at {best_n}", "start", 4, 13)
    sp_plot.text(sp_plot.X(wmax) - 2, sp_plot.Y(1 / t_rel(wmax)) + 11, f"h = {h_w:g}", "end")
    sp_plot.axes("workers", "speedup")

    # ================================================================ page 1
    band = (f'<div class="eqbig">{v("C", "eff")} = {frac("E[" + v("C", "attempt") + "]", v("p", "acc"))}'
            f'<span class="lbl">cost per accepted task (Ch 1, 19)</span></div>'
            f'<div class="txt"><div><b>Price the acceptance, not the token.</b> A repair attempt (table, p. 2): ${rep["model"]:.2f} tokens (cached bill, col. 2) '
            f'+ {A}${rep["env"]:.2f} sandbox + ${rep["assess"]:.2f} checks = ${rep["att"]:.2f}; {v("p", "acc")} = {_pct(rep["acc"])} → {v("C", "eff")} = {n(f"${rep_eff:.2f}")}.</div>'
            f'<div>Half-price tokens → ${eff_half_tok:.2f}. {v("p", "acc")} + {fmt(lever_gain * 100)} points → {n(f"${eff_more_acc:.2f}")}.</div>'
            f'<div><b>Horizon compounding (Ch 1).</b> Pr(success) = (1 − {v("ε")})<sup>{v("H")}</sup> ≈ e<sup>−{v("ε")}{v("H")}</sup>. '
            f'{A}{v("ε")} = {_pct(eps)}: 90% at {v("H")} = {hz.short_horizon}, {n(_pct(succ[hz.long_horizon]), True)} at {hz.long_horizon}. '
            f'{_pct(hz.target_success)} at {hz.long_horizon} turns needs {v("ε")} ≤ {_pct_dn(eps_req, 3)}.</div></div>'
            f'<div class="side" style="width:2.75in">{hp.svg("Probability of success against turns for three per-turn error rates")}</div>')

    p1 = Page(
        question="Will it finish within budget?",
        sub="Read left to right: where a model call's time and money go, what the context and memory hold, and what each tool call risks.",
        meta="reference: Llama 3 70B on NVIDIA H100",
        notation=[
            ("Trajectory", f'{v("H")} turns · {v("ε")} error per unchecked turn · {v("p", "acc")} acceptance per attempt · {v("C", "attempt")}, {v("C", "eff")} $ per attempt, per accepted task · '
                           f'{v("p")} pass rate of one try · {A}= example input'),
            ("Tokens", f'{v("P")} parameters · {v("S")} input tokens (sequence length) · {v("K")} output tokens · Δ{v("S", "turn")} tokens appended per turn · '
                       f'{v("s", "elem")} bytes per value · BW bytes/s · {v("c", "in")}, {v("c", "out")}, {v("c", "cache")} $ per input, output, cached-input token'),
        ],
        know=[
            Know(f"0.99<sup>100</sup> ≈ 1/e", f"(1 − ε)<sup>H</sup> ≈ e<sup>−εH</sup>: 100 turns at 99% ≈ {_pct(e_inv)}"),
            Know(f"{fmt(rule)} / n", "95% bound on a failure rate after n clean trials"),
            Know(f"{fmt(m_tok / 1024)} KiB", "KV state per token, Llama 3 70B, FP16"),
            Know(f"${c_in * 1e6:g} · ${c_out * 1e6:g}", f"$ per M in, out tokens (book's hypothetical); cached ${c_cache * 1e6:.2f}"),
            Know(f"{fmt(price_out_ratio)}× · {price_cache_ratio:g}×", "output, cached-input price ÷ input price"),
            Know(f"{fmt(tok_out_s)} · {fmt(round(tok_in_s, -3))}", "≈ tokens/s out, in: 70B 8-bit, one H100, batch 1 (floors)"),
            Know(f"{H100.memory.bandwidth.to('TB/s').magnitude:.2f} · {fmt(R_PEAK / 1e12)}", "H100: HBM TB/s · FP8 peak TFLOP/s"),
        ],
        band=band,
        cols=[
            Col("1 Where do a call's time and money go?", [
                Card("Call latency floor: output-bound",
                     f'{v("T")} ≥ {v("S")} · {frac("2" + v("P"), v("R", "peak"))} + {v("K")} · {frac(v("P") + " · " + v("s", "elem"), "BW")}',
                     chapter="Ch 2", holds=f'Batch 1 at peak FLOP/s {v("R", "peak")}; queueing and preparation add more.',
                     ex=f'70B 8-bit on H100, {A}{v("S")} = {fmt(call.prompt_tokens)}, {v("K")} = {call.output_tokens} → {call_pre:.2f} s + {call_dec:.2f} s: '
                        f'{n(_pct(call_out_share))} of the time on {_pct(call_tok_share, 1)} of the tokens.'),
                Card("Append-only history is resent every turn",
                     f'{v("S", "sent")} = {v("H")} · {v("S", "base")} + Δ{v("S", "turn")} · {frac(v("H") + "(" + v("H") + " − 1)", "2")}',
                     chapter="Ch 2", holds=f'{v("S", "base")} first prompt; {v("S", "sent")} input tokens sent over {v("H")} calls.',
                     ex=f'{A}{v("S", "base")} = {fmt(S0)}, Δ{v("S", "turn")} = {fmt(dS)}, {v("H")} = {H_rs} → {n(fmt(sent30))} ({fmt(fresh30)} new); {v("H")} = {2 * H_rs} → {sent60 / sent30:.1f}×.'),
                Card("pass@k: one of k independent tries passes", f'pass@{v("k")} = E<sub><i>i</i></sub>[1 − (1 − {v("p", "i")})<sup>{v("k")}</sup>]', chapter="Ch 3, 13",
                     holds=f'{v("p", "i")} pass rate of task <i>i</i>; {v("k")} tries.',
                     ex=f'{A}{v("p", "i")} = {_pct(pools.pass_rate)}, {v("k")} = ' + ", ".join(str(k) for k in pools.pool_sizes) + " → " + ", ".join(_pct(pass_at[k]) for k in pools.pool_sizes) +
                        f'. Half at {_pct(pools.easy_pass_rate)}, half at 0 → pass@{k_max_pool} = {n(_pct(mixed), True)}: more samples cannot fix a task that never passes.'),
            ], "Ch 2–3"),
            Col("2 Will the context fit, and at what price?", [
                Card("Context budget", 'fixed + evidence ≤ window − output reserve', chapter="Ch 4",
                     ex=f'{fmt(cb.window)} − {A}{fmt(cb.output_reserve)} − {fmt(fixed)} fixed → {n(fmt(obs_budget))}; {A}+{fmt(cb.observation_per_turn)} per turn → full at turn {turns_fit + 1} of {cb.turns}.'),
                Card("Prefix caching prices the resend",
                     f'$/turn = {v("c", "cache")}({v("S")} − Δ{v("S", "turn")}) + {v("c", "in")}Δ{v("S", "turn")} + {v("c", "out")}{v("K")}', chapter="Ch 4, 19",
                     holds="From turn 2, when the prior request is a byte-identical prefix.",
                     ex=f'The 30-turn resend, {v("K")} = {K_rs}, {A}{usd_m(bill.input_price)}/{usd_m(bill.input_price * bill.cached_fraction)}/{usd_m(bill.output_price)} per M → '
                        f'${bill_fresh:.2f} fresh, {n(f"${bill_cached:.2f}")} cached ({bill_fresh / bill_cached:.1f}×).'),
                Card("KV state of one trajectory",
                     f'{v("M", "KV")} = 2 · {v("N", "L")} · {v("H", "KV")} · {v("d", "head")} · {v("s", "elem")} · {v("S")}', chapter="Ch 5",
                     holds=f'{v("N", "L")} layers, {v("H", "KV")} KV heads, {v("d", "head")} head dimension.',
                     ex=f'Llama 3 70B FP16, {v("S")} = {fmt(S_ctx)} → {fmt(m_tok / 1024)} KiB/token, {n(f"{m_req / 1e9:.1f} GB")}. '
                        f'{kv.accelerators} × {hbm_gib:g} GiB − {w_bytes / 1e9:.0f} GB weights − {A}{kv.reserve.to("GB").magnitude:g} GB → {n(f"{max_req} contexts")}.'),
                Card("Exact vector scan is bandwidth-bound", f'{v("T")} ≈ index bytes / BW', chapter="Ch 6",
                     ex=f'{A}{fmt(vs.vectors / 1e6)}M × {vs.dimension} × {vs.bytes_per_value.to("B").magnitude:g} B = {vs_bytes / 1e9:.1f} GB at {fmt(cpu_bw / 1e9)} GB/s → {n(f"{fmt(vs_t * 1e3)} ms")}; {fmt(vs.filtered_vectors)} vectors → {vs_filt * 1e3:.2f} ms.'),
            ], "Ch 4–6"),
            Col("3 What does a tool call risk?", [
                Card("Blind retry duplicates effects: settle first",
                     [f'Pr(timeout) = 1 − (1 − {v("q", "req")})(1 − {v("q", "rep")})', f'{v("p", "dup")} = (1 − {v("q", "req")})<sup>2</sup> · {v("q", "rep")}'],
                     chapter="Ch 7", holds=f'{v("q", "req")}, {v("q", "rep")} loss rates; {v("p", "dup")} chance a retry repeats the effect.',
                     ex=f'{A}{v("q", "req")} = {v("q", "rep")} = {_pct(qreq)}, {fmt(br_calls)} calls/day → {n(fmt(dups_day) + " duplicates/day", True)}.'),
                Card("Sandbox pool: mean leases plus a ready reserve",
                     [f'{v("N", "active")} = {v("λ")} · {v("T", "hold")} &nbsp;(mean)',
                      f'Pr(Poisson({v("λ")}{v("T", "init")}) ≥ {v("N", "reserve")}) ≤ miss'],
                     chapter="Ch 8",
                     holds=f'{v("λ")} arrivals/s; {v("T", "hold")} lease, {v("T", "init")} restore; miss: no standby free.',
                     ex=f'{A}{v("λ")} = {lam:g}/s, {v("T", "hold")} = {fmt(t_hold)} s → {n(fmt(n_active))} busy (mean); {v("T", "init")} = {t_init * 1e3:g} ms, miss ≤ {_pct(sp.target_miss, 1)} → {n(str(reserve) + " standby")} for the tail.'),
                Card("Injection exposure grows with reads", f'Pr(exposed) = 1 − (1 − {v("q", "inj")})<sup>{v("n", "read")}</sup>', chapter="Ch 9",
                     holds=f'{v("q", "inj")} payload share of read items; {v("n", "read")} untrusted reads.',
                     ex=f'{A}{v("q", "inj")} = {_pct(q_pay)}: {v("n", "read")} = {inj.reads["repair"]} → {n(_pct(expo["repair"]))}, {inj.reads["research"]} → {n(_pct(expo["research"]), True)}; '
                        f'{_pct(inj.miss_rate)}-miss screen → {_pct(expo_scr, 1)}.'),
            ], "Ch 7–9"),
        ],
        bottom=Table(["Llama 3 70B", f"{v('P')} = {P / 1e9:.1f}B", f"{v('N', 'L')} = {NL}", f"{v('H', 'KV')} = {HKV}", f"{v('d', 'head')} = {DH}", f"context {fmt(S_ctx)}"], [
            ["NVIDIA H100", f"{v('R', 'peak')} = {fmt(R_PEAK / 1e12)} TFLOP/s FP8", f"BW = {BW / 1e12:.2f} TB/s", f"{hbm_gib:g} GiB HBM",
             f"{t_pre * 1e3:.3f} ms per input token", f"{t_dec * 1e3:.1f} ms per output token"],
            [f"{A}Prices per M tokens (book's)", f"{v('c', 'in')} = ${c_in * 1e6:g}", f"{v('c', 'cache')} = ${c_cache * 1e6:.2f} ({price_cache_ratio:g}×)",
             f"{v('c', 'out')} = ${c_out * 1e6:g} ({fmt(price_out_ratio)}×)", f"{A}logs ≈ {log_bpt:g} B/token", "GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B"],
        ], caption=f'Per-token times: 70B 8-bit weights, batch 1, at the peak and bandwidth above (floors, no queueing). {TAG}= example input. '
                   'Examples are the book\'s mlsysim scenarios: hypothetical inputs, not measurements.'),
    )

    # ================================================================ page 2
    p2 = Page(
        question="Is it worth running?",
        sub="Recovery, evidence, learning, and scale: what each costs, and how many accepted tasks it buys.",
        meta="the four lighthouse workloads",
        notation=[
            ("Shared", f'{v("p")} pass rate of one run · {v("k")} runs · {v("N")} tasks · {v("p", "acc")} acceptance per attempt · {v("C", "eff")} $ per accepted task · '
                       f'{v("c", "in")}, {v("c", "out")} $ per input, output token · {v("z")} = 1.96 (95%, two-sided) · {A}= example input'),
        ],
        cols=[
            Col("4 Can the runtime contain a failure?", [
                Card("Admit a call only if its worst case fits",
                     [f'committed + liability ≤ allowance', f'liability = {v("c", "in")} · {v("S")} + {v("c", "out")} · {v("K", "max")}'],
                     chapter="Ch 10",
                     holds=f'Committed = settled + reserved + final reserve; {v("S")}, {v("K", "max")} input, output-cap tokens.',
                     ex=f'{A}${settled:.2f} + ${reserved:.2f} + ${fin:.2f}; {v("S")} = {fmt(ad.input_tokens)}, {v("K", "max")} = {fmt(ad.output_allowance)} at {usd_m(ad.input_price)}/{usd_m(ad.output_price)} per M → '
                        f'${liab:.2f}; ${committed:.2f} &gt; ${allow:.2f}: {n("deny", True)}.'),
                Card("Checkpoint interval (Young)", f'{v("T", "ckpt", "*")} ≈ √(2 · {v("T", "write")} · MTBF)', chapter="Ch 11",
                     holds=f'{v("T", "write")} pause per checkpoint; MTBF mean time between work-losing failures.',
                     ex=f'{A}{v("T", "write")} = {t_w:g} s, MTBF = {mtbf / 60:g} min → {n(f"{fmt(t_star)} s")}; pauses plus rework ≈ {_pct(waste, 1)}.'),
                Card("Repair first, or unwind?",
                     f'E[cost] = {v("C", "fix")} + (1 − {v("p", "fix")}) · {v("C", "fallback")}',
                     chapter="Ch 12",
                     holds=f'{v("p", "fix")} repair success; {v("C", "fix")}, {v("C", "fallback")} cost of repairing, of unwinding and rebuilding. Repair pays if {v("p", "fix")} &gt; {v("C", "fix")}/{v("C", "fallback")}.',
                     ex=f'{A}{v("C", "fix")} = {rv.startup_time.to("s").magnitude:g} s + {rv.repair_output} tokens at {rv_rate:g}/s + {rv.validation_time.to("s").magnitude:g} s = {t_rep:.2f} s; '
                        f'{v("C", "fallback")} = {rv.unwind_time.to("s").magnitude:g} + {rv.rebuild_time.to("s").magnitude:g} s → repair pays above {n(_pct_up(pstar_t, 2))}.'),
                Card("Expected irreversible loss", f'{v("L", "A")} = Σ<sub><i>a</i></sub> Pr(bad <i>a</i>) · impact(<i>a</i>) · (1 − recovered(<i>a</i>))', chapter="Ch 12",
                     ex=f'One change, {A}Pr(bad) = {_pct(wk.wrong_change_probability)}, impact ${fmt(impact)} → {n(f"${fmt(l0)}", True)}; '
                        f'canary catches {_pct(wk.canary_catch_rate)}, reverts {_pct(wk.canary_recovery_fraction)} → {n(f"${l1:.0f}")}.'),
                Card("Head sampling misses rare failures", f'Pr(no trace kept) = (1 − {v("θ")})<sup>{v("n", "f")}</sup>', chapter="Ch 14",
                     holds=f'{v("θ")} share of traces kept at random; {v("n", "f")} occurrences.',
                     ex=f'{A}{v("θ")} = {_pct(hs.retention_probability)}, {v("n", "f")} = {hs.occurrences} → {n(_pct(hs_miss), True)}.'),
            ], "Ch 10–12"),
            Col("5 Is the evidence strong enough?", [
                Card("Wilson interval on an acceptance rate",
                     [f'{frac(v("X") + " + " + v("z", sup="2") + "/2", v("N") + " + " + v("z", sup="2"))} ± {frac(v("z"), v("N") + " + " + v("z", sup="2"))}√({frac(v("X") + "(" + v("N") + " − " + v("X") + ")", v("N"))} + {frac(v("z", sup="2"), "4")})'],
                     chapter="Ch 13", holds=f'{v("X")} accepted of {v("N")}.',
                     ex=f'{A}{v("N")} = {Nt}: {v("X")} = 0 → [0, {_pct_up(wil[0][1])}]; 2 → [{_pct_dn(wil[2][0])}, {_pct_up(wil[2][1])}]; '
                        f'{wi.successes[2]} → {n(f"±{(wil[wi.successes[2]][1] - wil[wi.successes[2]][0]) / 2 * 100:.1f} points")}.'),
                Card("Tasks to detect a change", f'{v("N", "arm")} ≈ {frac("2(" + v("z") + " + " + v("z", "β") + ")<sup>2</sup> " + v("p") + "(1 − " + v("p") + ")", v("δ", sup="2"))}',
                     chapter="Ch 13",
                     holds=f'Per independent arm; {v("z", "β")} = {pl.z_beta:g} for 80% power; {v("δ")} difference to detect.',
                     ex=f'{v("p")} = {pl.rate:g}, {A}{v("δ")} = {fmt(pl.differences[0] * 100)} points → {n(fmt(per_arm[pl.differences[0]]))}; {fmt(pl.differences[1] * 100)} points → {n(fmt(per_arm[pl.differences[1]]), True)}.'),
                Card("Verifier precision: share of passes that are valid",
                     f'Π = {frac(v("p", "valid") + "(1 − " + v("ε", "FR") + ")", v("p", "valid") + "(1 − " + v("ε", "FR") + ") + (1 − " + v("p", "valid") + ")" + v("ε", "FA"))}',
                     chapter="Ch 3",
                     holds=f'{v("p", "valid")} valid share of candidates; {v("ε", "FA")}, {v("ε", "FR")} false-accept, false-reject rates.',
                     ex=f'{A}{v("p", "valid")} = {_pct(vp.valid_rate)}, {v("ε", "FR")} = {_pct(vp.false_reject_rate)}, {v("ε", "FA")} = {_pct(vp.false_accept_rate)} → Π = {n(_pct(prec))}; '
                        f'{v("ε", "FA")} = {_pct(vp.false_accept_rate / 2)} → {_pct(prec_half)}.'),
                Card("pass^k: all k runs pass", f'pass<sup>{v("k")}</sup> = E<sub><i>i</i></sub>[{v("p", "i", v("k"))}] &nbsp; vs &nbsp; pass@{v("k")} = 1 − E<sub><i>i</i></sub>[(1 − {v("p", "i")})<sup>{v("k")}</sup>]', chapter="Ch 13",
                     holds=f'{v("p", "i")} pass rate of task <i>i</i>.',
                     ex=f'{A}{v("p", "i")} = {_pct(rc.step_success)}, {v("k")} = {rc.long_horizon} → {n(_pct(pk), True)}; holding {_pct(rc.target_success)} over {rc.long_horizon} → {v("p", "i")} ≥ {_pct_up(p_need)}.'),
                Card("Group contrast: which rollouts carry gradient", f'Pr(mixed group) = 1 − {v("p", sup=v("G"))} − (1 − {v("p")})<sup>{v("G")}</sup>', chapter="Ch 15, 17",
                     holds=f'{v("G")} rollouts of one task; an all-pass or all-fail group has zero advantage, so zero gradient.',
                     ex=f'{A}{v("G")} = {gr.group}: {v("p")} = {_pct(gr.pass_rate)} → {n(_pct(g_c))} ({g_per:.0f} rollouts per useful group); {v("p")} = {_pct(tg.pass_rate)} → {_pct(g_c_t)}.'),
            ], "Ch 13–17"),
            Col("6 Does scale pay?", [
                Chart(sp_plot.svg("Speedup against workers with and without coordination overhead")),
                Card("Delegation speedup with coordination cost",
                     f'speedup = 1 / ((1 − {v("f")}) + {v("f")}/{v("N", "w")} + {v("h")}({v("N", "w")} − 1))',
                     chapter="Ch 18", holds=f'{v("N", "w")} workers, {v("f")} parallel fraction, {v("h")} overhead per added worker.',
                     ex=f'{A}{v("f")} = {f_par:g}, {v("h")} = {h_w:g} → peak {n(f"{best_s:.2f}×")} at {best_n} workers.'),
                Card("A token multiple must buy acceptance",
                     f'{frac(v("C", "eff") + " multi", v("C", "eff") + " single")} = token multiple × {frac(v("p", "acc") + " single", v("p", "acc") + " multi")}', chapter="Ch 18",
                     holds="With cost proportional to tokens.",
                     ex=f'{fmt(dl.multi_agent_tokens_vs_chat)}× vs {fmt(dl.agent_tokens_vs_chat)}× chat tokens (reported, Hadfield et al., 2025) → {v("p", "acc")} must rise {m_tok_mult:.2f}×; impossible above {_pct_dn(max_single)}.'),
                Card("The token-price fallacy", f'{v("C", "eff")} = cohort cost / accepted tasks', chapter="Ch 19",
                     ex=f'{A}{fb.turns} turns × ({fmt(fb.input_tokens_per_turn)} in, {fmt(fb.output_tokens_per_turn)} out at {usd_m(fb.input_price)}/{usd_m(fb.output_price)} per M + ${_usd(fb.environment_per_turn):.2f} tools) '
                        f'+ ${_usd(fb.assessment_per_attempt):.2f} check = ${fr_att:.2f}, {fb.accepted}/100 → ${fr_eff:.2f}. '
                        f'{al.turns} × ({fmt(al.input_tokens_per_turn)}, {fmt(al.output_tokens_per_turn)} at {usd_m(al.input_price)}/{usd_m(al.output_price)} + ${_usd(al.environment_per_turn):.2f} tools) + ${_usd(al.assessment_per_attempt):.2f} = ${lw_att:.2f}, '
                        f'{al.accepted}/100 → ${lw_eff:.2f} ({n(f"{lw_eff / fr_eff:.1f}×", True)}).'),
                Card("Adaptation payback", f'{v("N", sup="*")} = investment / ({v("C", "base")} − {v("C", "adapted")})', chapter="Ch 19",
                     holds=f'{v("N", sup="*")} break-even tasks; {v("C", "base")}, {v("C", "adapted")} cost per task before, after.',
                     ex=f'{A}${fmt(invest)}; {ap.base["steps"]} × ({fmt(ap.base["input_tokens"])} in, {ap.base["output_tokens"]} out) at {usd_m(ap.base["input_price"])}/{usd_m(ap.base["output_price"])} = ${c_base:.4f} → '
                        f'{ap.adapted["steps"]} × ({fmt(ap.adapted["input_tokens"])}, {ap.adapted["output_tokens"]}) at {usd_m(ap.adapted["input_price"])}/{usd_m(ap.adapted["output_price"])} = ${c_adapt:.4f} → '
                        f'{n(fmt(n_be) + " tasks")}; {fmt(ap.monthly_volume)}/month → {months:.1f} months.'),
            ], "Ch 18–19"),
        ],
        bottom=Table(["Reference workloads (Ch 1, 19)", "turns", "untrusted reads", "Pr(exposed) at 1%", "C<sub>attempt</sub>", "p<sub>acc</sub>", "C<sub>eff</sub>", "largest cost term"], [
            [f"{k.capitalize()} agent ({archetype[k]})", str(horizon[k]), str(inj.reads[k]), _pct(expo[k]),
             f"${rows_cost[k]['att']:.2f}", _pct(rows_cost[k]["acc"]), f"${rows_cost[k]['eff']:.2f}",
             f"{rows_cost[k]['largest']} ({_pct(rows_cost[k]['share'])})"]
            for k in ("repair", "research", "remediation", "support")]),
    )
    return Sheet(volume=3, book="Agentic Machine Learning Systems", scope="The Trajectory", pages=[p1, p2],
                 sources=f'{TAG}= example input · GB = 10<sup>9</sup> B · examples are the book\'s mlsysim scenarios: hypothetical inputs, not measurements · prices are the book\'s hypothetical $3 / $0.30 / $15 per M tokens')
