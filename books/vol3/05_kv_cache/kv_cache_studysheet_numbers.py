"""Numbers for the KV Cache Management study sheet (Volume III, Chapter 5).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(KvGeometry, StaticReservation, PagedWaste, PrefixHit, PausedStateChoice, AgentCapacity,
ResidencyByArchetype), and written as \\setnum{key}{value} to kv_cache_studysheet_numbers.tex.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value. Accelerator memory is read from the registry in GiB and
printed with its decimal GB value.
"""
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from mlsysim import Scenarios as SC


def fx(x, nd):
    """Round half up, as a reader would, and format with nd decimals."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(x, 9))).quantize(q, rounding=ROUND_HALF_UP))


def r(x, nd):
    return float(fx(x, nd))


def pct(x, nd=0):
    return fx(x * 100, nd)


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}

    # ---- Footprint and fit (chapter: KvGeometry, Scenarios.Llama3_70B*ContextAdmissionDgxH100)
    ag = SC.Llama3_70BAgentContextAdmissionDgxH100
    ar = ag.result
    m = ag.workload
    layers, kvheads, dhead = m.layers, m.kv_heads, ar.head_dim
    elem = ag.kv_bytes_per_elem.to("byte").magnitude
    tok_bytes = 2 * layers * kvheads * dhead * elem
    near(tok_bytes, ar.kv_per_token.to("byte").magnitude, 0)
    acc_gib = ag.hbm_per_accelerator.to("GiB").magnitude
    acc_gb = ag.hbm_per_accelerator.to("GB").magnitude
    n_acc = ag.accelerators
    hbm_gb = r(ar.hbm_total.to("GB").magnitude, 1)
    weights_gb = r(ar.weights.to("GB").magnitude, 1)
    reserve_gb = ag.reserve.to("GB").magnitude
    pool_gb = r(hbm_gb - weights_gb - reserve_gb, 1)
    near(pool_gb, ar.kv_budget.to("GB").magnitude, 0.05)
    ctx = ag.context_tokens
    kv_gb = r(ctx * tok_bytes / 1e9, 2)
    fit = math.floor(pool_gb / kv_gb)                                 # from the printed values
    assert fit == ar.max_requests == 48, fit
    n.update(layers=f"{layers}", kvheads=f"{kvheads}", headdim=f"{dhead}", elembytes=f"{elem:g}",
             elembits=f"{elem * 8:g}", tokbytes=f"{tok_bytes:,.0f}", nodecount=f"{n_acc}",
             accgib=f"{acc_gib:.0f}", accgb=fx(acc_gb, 1), hbmgb=fx(hbm_gb, 1), weightsgb=fx(weights_gb, 1),
             reservegb=f"{reserve_gb:.0f}", poolgb=fx(pool_gb, 1), ctx=f"{ctx:,}", kvgb=fx(kv_gb, 2), fit=f"{fit}")

    # ---- Residency (chapter: AgentCapacity timings, Scenarios.RepairBlockSharingResidency)
    rs = SC.RepairBlockSharingResidency
    gen = rs.generation.to("second").magnitude
    wait = rs.wait.to("second").magnitude
    cycle = gen + wait
    kept = fit / cycle
    freed = fit / gen
    tax = 1 + wait / gen
    near(freed / kept, tax, 1e-12)
    near(tax, rs.result.residency_factor, 1e-12)
    n.update(gen=f"{gen:g}", wait=f"{wait:g}", cycle=f"{cycle:g}", keptrate=fx(kept, 2), freedrate=fx(freed, 1),
             tax=f"{tax:g}")
    # chapter's own fit (nbk-05-residency-tax, AgentCapacity): each request also holds its output allowance,
    # the pool keeps headroom, and allocation is in whole blocks. Reproduced from the printed values.
    cr = rs.result
    req_tok = rs.context + rs.output
    assert req_tok % rs.block_tokens == 0                              # whole blocks: rounding adds nothing
    req_gb = r(req_tok * tok_bytes / 1e9, 2)
    use_gb = r((1 - rs.headroom) * pool_gb, 1)
    chapfit = math.floor(use_gb / req_gb)
    assert chapfit == cr.fit_unshared == 41, chapfit
    chapkept, chapfreed = chapfit / cycle, chapfit / gen
    near(chapkept, cr.retained_rate.magnitude, 1e-9)
    near(chapfreed, cr.released_rate.magnitude, 1e-9)
    n.update(chapfit=f"{chapfit}", chapkept=fx(chapkept, 2), chapfreed=fx(chapfreed, 1),
             outtok=f"{rs.output:,}", reqtok=f"{req_tok:,}", reqgb=fx(req_gb, 2), headpct=pct(rs.headroom),
             headkeep=fx(1 - rs.headroom, 1), usegb=fx(use_gb, 1), blocktok=f"{rs.block_tokens}")

    # ---- Chunked prefill (chapter: ChunkedIteration, Scenarios.ChunkedPrefillAmongDecodes, eq-kv-token-budget)
    cp = SC.ChunkedPrefillAmongDecodes
    budget = cp.chunk_tokens + cp.decodes                              # the chapter's iteration size
    assert cp.prompt_tokens // cp.chunk_tokens == cp.result.chunks
    q_dec = 2 * cp.decodes                                             # ASSUMED: twice the chapter's decodes
    q_chunk = budget - q_dec
    q_iters = math.ceil(cp.prompt_tokens / q_chunk)
    n.update(cpbudget=f"{budget}", cpdecodes=f"{cp.decodes}", cpchunk=f"{cp.chunk_tokens}",
             cpprompt=f"{cp.prompt_tokens:,}", cpchunks=f"{cp.result.chunks}",
             cpqdec=f"{q_dec}", cpqchunk=f"{q_chunk}", cpqiters=f"{q_iters}",
             cpqratio=fx(cp.prompt_tokens / q_chunk, 2))

    # ---- Your turn: support agent (chapter: ResidencyByArchetype, Scenarios.SupportAgentResidency)
    sp = SC.SupportAgentResidency
    sres = sp.result
    sctx = sp.context_tokens
    skv = r(sctx * tok_bytes / 1e9, 3)
    sfit = math.floor(pool_gb / skv)
    assert sfit == sres.fit_unshared == 195, sfit
    sgen = sp.generation.to("second").magnitude
    swait = sp.wait.to("second").magnitude
    scycle = sgen + swait
    stax = 1 + swait / sgen
    near(stax, sres.residency_factor, 1e-12)
    n.update(supctx=f"{sctx:,}", supkvgb=fx(skv, 3), supfit=f"{sfit}", supgen=f"{sgen:g}", supwait=f"{swait:g}",
             supcycle=f"{scycle:g}", supkept=fx(sfit / scycle, 2), supfreed=fx(sfit / sgen, 1), suptax=f"{stax:g}",
             supreload=f"{sres.reload.to('ms').magnitude:.0f}", suprecompute=fx(sres.recompute.to('second').magnitude, 2))

    # ---- Paused state (chapter: PausedStateChoice, Scenarios.RepairToolWaitPausedState)
    ps = SC.RepairToolWaitPausedState
    pr = ps.result
    state_gb = pr.state.to("GB").magnitude
    near(state_gb, ctx * tok_bytes / 1e9, 1e-9)
    link = pr.link.to("GB/second").magnitude
    reload_ms = state_gb / link * 1000
    near(reload_ms, pr.copy_in.to("ms").magnitude, 1e-9)
    near(2 * reload_ms, pr.round_trip.to("ms").magnitude, 1e-9)
    n.update(linkgb=f"{link:.0f}", reload=f"{reload_ms:.0f}", roundtrip=f"{2 * reload_ms:.0f}",
             recompute=fx(pr.recompute.to("second").magnitude, 1), ratio=f"{pr.recompute_over_reload:.0f}",
             freedwait=fx(pr.released_on_demand.to("second").magnitude, 2))

    # ---- Mistake: full-cap reservation (chapter: StaticReservation, PagedWaste)
    cp = SC.CappedRequestsPagedAllocation
    cres = cp.result
    near(cres.final_length_efficiency, cp.mean_final / cp.cap, 1e-12)
    n.update(cap=f"{cp.cap:,}", meanfinal=f"{cp.mean_final:,}", efficiency=pct(cres.final_length_efficiency, 2),
             reservations=f"{cres.reservations}", pagedfit=f"{cres.paged_fit}", pagedgain=f"{round(cres.paged_gain)}")
    assert cres.reservations == fit

    # ---- Mistake: a hit is not freshness (chapter: PrefixHit)
    ph = SC.InvariantPrefixRebuildDgxH100
    hres = ph.result
    n.update(hitprefix=f"{ph.prefix_tokens:,}", rebuild=f"{hres.rebuild_time.to('ms').magnitude:.0f}",
             hitpass=f"{hres.weight_read_time.to('ms').magnitude:.0f}")

    # ---- Q1 ASSUMED: repair requests of 32,768 input + 2,048 output tokens in 16-token blocks,
    #      sharing a 20,480-token workspace prefix (block-aligned)
    q_in, q_out, q_blk, q_shared = rs.context, rs.output, rs.block_tokens, 20_480
    assert q_shared % q_blk == 0
    tot_blocks = math.ceil((q_in + q_out) / q_blk)
    sh_blocks = q_shared // q_blk
    f = r(sh_blocks / tot_blocks, 3)
    gain = 1 / (1 - f)
    n.update(qin=f"{q_in:,}", qout=f"{q_out:,}", qblk=f"{q_blk}", qshared=f"{q_shared:,}",
             qtotblocks=f"{tot_blocks:,}", qshblocks=f"{sh_blocks:,}", qf=fx(f, 3), qgain=fx(gain, 1))

    # ---- Q2: chunked prefill, computed above with the chapter's ChunkedIteration scenario

    # ---- Q3 ASSUMED: three 5,000-token requests, first 4,100 identical, full 64-token blocks only;
    #      in request three the harness replaces the stale log from token 3,000 onward
    q_req, q_prompt, q_same, q_b, q_cut = 3, 5_000, 4_100, 64, 3_000
    reuse = (q_same // q_b) * q_b
    reuse_cut = (q_cut // q_b) * q_b
    rho = (q_req - 1) * reuse / (q_req * q_prompt)
    rho_cut = (reuse + reuse_cut) / (q_req * q_prompt)
    n.update(qreq=f"{q_req}", qprompt=f"{q_prompt:,}", qsame=f"{q_same:,}", qb=f"{q_b}", qcut=f"{q_cut:,}",
             qreuse=f"{reuse:,}", qreusecut=f"{reuse_cut:,}", qtotal=f"{q_req * q_prompt:,}",
             qreusesum=f"{(q_req - 1) * reuse:,}", qreusesumcut=f"{reuse + reuse_cut:,}",
             qrho=pct(rho, 1), qrhocut=pct(rho_cut, 1))
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
