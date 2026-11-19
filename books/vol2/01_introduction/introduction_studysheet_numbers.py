"""Numbers for the Introduction study sheet (Volume II, Chapter 1).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to introduction_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value.
"""
import math
from pathlib import Path

from mlsysim import Literature, Models, Systems
from mlsysim.engine.scenarios import Scenarios as SC


def r(x, nd):
    return round(x, nd)


def fx(x, nd):
    return f"{x:.{nd}f}"


def sci(x, nd=1):
    """LaTeX math for x in scientific notation, e.g. 8.0\\times10^{-5}."""
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, nd) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{nd}f}\\times10^{{{e}}}"


def pct(x):
    return f"{x * 100:.0f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    n = {}

    # ---- Worked example: the 175B reference step on 1,024 and on 8,192 GPUs (chapter: gpt3-sync-tax)
    small, large = SC.DenseLLMPretrainingStep, SC.DenseLLMPretrainingStep8K
    rs, rl = small.result, large.result

    def parts(res):
        comp = res.node_profile.latency.to("s").magnitude
        tp = res.tp_communication_latency.to("s").magnitude
        bub = res.pipeline_bubble_latency.to("s").magnitude
        step = res.step_latency_total.to("s").magnitude
        return comp, tp, step - comp - tp - bub, bub, step   # exposed DP = step - compute - TP - bubble

    sc, stp, sdp, sbub, sstep = parts(rs)
    lc, ltp, ldp, lbub, lstep = parts(rl)
    # printed (rounded) inputs: two decimals on the small cluster, three on the large
    sc_, stp_, sdp_, sbub_ = r(sc, 2), r(stp, 2), r(sdp, 2), r(sbub, 2)
    lc_, ltp_, ldp_, lbub_ = r(lc, 3), r(ltp, 3), r(ldp, 3), r(lbub, 3)
    sstep_, lstep_ = sc_ + stp_ + sdp_ + sbub_, lc_ + ltp_ + ldp_ + lbub_
    near(sstep_, sstep, 0.006)
    near(lstep_, lstep, 0.0006)
    speedup = sstep_ / lstep_
    sfrac, lfrac = sc_ / sstep_, lc_ / lstep_
    near(sfrac, rs.scaling_efficiency, 0.005)          # chapter: compute-time fraction = scaling efficiency
    near(lfrac, rl.scaling_efficiency, 0.005)
    n.update(
        smallgpus=f"{small.fleet.total_accelerators:,}", largegpus=f"{large.fleet.total_accelerators:,}",
        gpugrowth=f"{large.fleet.total_accelerators // small.fleet.total_accelerators}",
        batchseqs=f"{small.global_batch_sequences:,}",
        params=f"{small.replica.workload.parameters.to('param').magnitude / 1e9:.0f}",
        smallcompute=fx(sc_, 2), smalltensor=fx(stp_, 2), smallgrad=fx(sdp_, 2), smallbubble=fx(sbub_, 2),
        smallstep=fx(sstep_, 2),
        largecompute=fx(lc_, 3), largetensor=fx(ltp_, 3), largegrad=fx(ldp_, 3), largebubble=fx(lbub_, 3),
        largestep=fx(lstep_, 3),
        stepspeedup=f"{speedup:.1f}", smallfraction=pct(sfrac), largefraction=pct(lfrac),
        largebubbleshare=pct(lbub_ / lstep_),
    )

    # ---- Problem to finish: accelerators twice as fast on 1,024 GPUs (chapter: FasterAcceleratorFallacy)
    accel = 2                                                     # chapter's scenario assumption
    comm_ = r(stp_ + sdp_, 2)                                      # exposed TP + DP exchange
    halved = r((sc_ + sbub_) / accel, 2)
    after = halved + comm_
    newcompute = r(sc_ / accel, 3)
    near(after, (sc + sbub) / accel + (stp + sdp), 0.01)
    smallgpusec = small.fleet.total_accelerators * sstep_
    largegpusec = large.fleet.total_accelerators * lstep_
    n.update(smallgpuseconds=f"{smallgpusec:,.0f}", largegpuseconds=f"{largegpusec:,.0f}",
             gpusecondsratio=f"{largegpusec / smallgpusec:.2f}")
    n.update(accel=f"{accel}", commexposed=fx(comm_, 2), halfbubble=fx(sbub_ / accel, 3), halved=fx(halved, 2), afterstep=fx(after, 2),
             afterspeedup=f"{sstep_ / after:.1f}", newcompute=fx(newcompute, 3),
             afterfraction=pct(newcompute / after))

    # ---- Mistakes: reliability (chapter: Gpt4TrainingScenario, ReliabilityGap) and linear scaling
    mttf = Systems.Reliability.Gpu.mttf_hours
    mttf = float(getattr(mttf, "magnitude", mttf))
    gpus = Models.Language.GPT4.training_accelerators_ref.to("count").magnitude
    mtbf = mttf / gpus
    n.update(mttf=f"{mttf:,.0f}", fleetgpus=f"{gpus:,.0f}", mtbf=f"{mtbf:.0f}", perday=f"{24 / mtbf:.0f}")
    a1k, a10k, a4n = SC.NodeAvailability1K, SC.NodeAvailability10K, SC.FourNinesAvailability10K
    n.update(threenines=f"{a1k.availability * 100:.1f}", fournines=f"{a4n.availability * 100:.2f}",
             thousand=f"{a1k.n_components:,}", tenthousand=f"{a4n.n_components:,}",
             allupthree=f"{a1k.all_up * 100:.1f}", allupfour=f"{a4n.all_up * 100:.1f}",
             allupten=f"{a10k.all_up * 100:.3f}",
             # for the front plot: per-node availabilities as fractions
             lnthree=f"{math.log(a1k.availability):.7f}", lnfour=f"{math.log(a4n.availability):.7f}")
    plan = SC.CoordinationOverheadExample
    n.update(planned=f"{plan.planned_speedup:.0f}", overhead=pct(plan.overhead_fraction),
             keepshare=f"{1 - plan.overhead_fraction:.1f}", realized=f"{plan.realized_speedup:.1f}")

    # ---- Front CI ladder: the reference job's CI on both clusters vs fabric balance (chapter: gpt3-ci-ratio)
    from mlsysim.physics import calc_ring_allreduce_data_factor
    params = small.replica.workload.parameters.to("param").magnitude
    rg = small.replica.replica_gpus
    grad = params * 2 / rg                                         # 16-bit gradient slice per GPU, bytes
    link = small.fleet.fabric.bandwidth.to("GB/s").magnitude
    peak = small.fleet.node.accelerator.compute.peak_flops.to("TFLOP/s").magnitude
    balance = link * 1e9 / (peak * 1e12)

    def ci(s):
        tokens = s.global_batch_tokens // s.data_parallel
        return calc_ring_allreduce_data_factor(s.data_parallel) * grad / (6 * params * tokens / rg)
    cis, cil = ci(small), ci(large)
    n.update(link=f"{link:.0f}", peak=f"{peak:.0f}", balance=sci(balance), cismall=sci(cis), cilarge=sci(cil),
             balancelog=f"{math.log10(balance):.3f}", cismalllog=f"{math.log10(cis):.3f}",
             cilargelog=f"{math.log10(cil):.3f}", cigrowth=f"{cil / cis:.1f}",
             smallofbalance=pct(cis / balance), largeofbalance=pct(cil / balance))

    # ---- Check yourself
    # Q1 ASSUMED: a job needs 4,000 nodes at 99.98 percent each; per-node MTTF is the chapter's 50,000 h
    qnodes, qavail = 4_000, 0.9998
    n.update(qnodes=f"{qnodes:,}", qavail=f"{qavail * 100:.2f}", qallup=pct(qavail ** qnodes),
             qmtbf=f"{mttf / qnodes:.1f}")

    # Q2 ASSUMED: a worker moves 8 GB per step and does 1.0e14 FLOPs, on the chapter's link and peak
    qbytes, qflop = 8, 1.0e14
    qci = qbytes * 1e9 / qflop
    qt_link = qbytes / link
    qt_math = qflop / (peak * 1e12)
    qci_half = qci / 2
    n.update(qbytes=f"{qbytes}", qbytesfull=f"{qbytes}\\times10^{{9}}", qflop=sci(qflop), qci=sci(qci), qlink=f"{qt_link:.2f}", qmath=f"{qt_math:.2f}",
             qhalfbytes=f"{qbytes // 2}", qcihalf=sci(qci_half), qlinkhalf=f"{qt_link / 2:.2f}")
    assert qci > balance > qci_half

    # Q4 ASSUMED: scaling exponent 0.1; budget 1.2e24 FLOPs; chapter's 6 FLOPs per parameter-token and
    # 20 tokens per parameter (Literature.Chinchilla); only 500 billion tokens of data exist
    alpha, budget, havetok = 0.1, 1.2e24, 500e9
    k = float(Literature.Chinchilla.ComputeConstant)
    tpp = float(Literature.Chinchilla.TokensPerParam)
    p_opt = math.sqrt(budget / (k * tpp))
    d_opt = tpp * p_opt
    n.update(qalpha=f"{alpha}", qmult=f"{2 ** (1 / alpha):,.0f}", qinvalpha=f"{1 / alpha:.0f}", qpow=f"2^{{{1 / alpha:.0f}}}",
             qbudget=sci(budget), flopsper=f"{k:.0f}", tokensper=f"{tpp:.0f}",
             qparams=f"{p_opt / 1e9:.0f}", qtokens=f"{d_opt / 1e12:.0f}", qhave=f"{havetok / 1e9:.0f}",
             qkt=f"{k * tpp:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
