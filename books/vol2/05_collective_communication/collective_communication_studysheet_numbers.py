"""Numbers for the Collective Communication study sheet (Volume II, Chapter 5).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to collective_communication_studysheet_numbers.tex next to
this file. Inputs the sheet chooses for its own questions are marked ASSUMED (built as variants
of the chapter's scenarios). A printed result is computed from the rounded values the sheet prints
beside it, and checked against the unrounded model value.

Units: KB, MB, GB are decimal (10^3, 10^6, 10^9 bytes), as in the chapter; link rates are GB/s
(bytes); the NDR port is 400 Gb/s (bits) = 50 GB/s.
"""
import math
from pathlib import Path

from mlsysim import Scenarios as SC
from mlsysim.core.units import Q_


def fx(x, nd):
    return f"{x:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def mag(q, unit):
    return q.to(unit).magnitude


def numbers() -> dict:
    n = {}

    # ---- Ring card + worked example: one 25 MB bucket on 64 and 1,024 NDR ranks (chapter: chunk-regime-calc)
    d = SC.DdpBucketRingNdr
    r = d.result
    small, large = d.group_sizes
    alpha = mag(d.alpha, "microsecond")                 # 1.5 us, canonical effective NDR startup
    beta = mag(d.bandwidth, "GB/s")                     # 50 GB/s per port
    gbit = mag(d.bandwidth, "Gbit/s")                   # 400 Gb/s
    bucket = mag(d.message, "MB")
    nstar = mag(SC.CriticalSizeByInterconnect.result.crossovers["NDR"], "KB")
    near(alpha * beta, nstar, 1e-9)                     # us * GB/s = KB
    chunk_s, chunk_l = mag(r.chunk[small], "KB"), mag(r.chunk[large], "KB")
    assert chunk_s > nstar > chunk_l
    steps_s, steps_l = 2 * (small - 1), 2 * (large - 1)
    startup_s = steps_s * alpha / 1000                  # ms
    startup_l = steps_l * alpha / 1000
    bytes_s, bytes_l = mag(r.ring_byte_time[small], "ms"), mag(r.ring_byte_time[large], "ms")
    ring_s, ring_l = mag(r.ring_time[small], "ms"), mag(r.ring_time[large], "ms")
    near(startup_s, mag(r.ring_startup_time[small], "ms"), 1e-9)
    near(startup_l, mag(r.ring_startup_time[large], "ms"), 1e-9)
    # printed: startup to 2 dp, bytes to 2 dp; totals are their sums
    ps, pl = round(startup_s, 2), round(startup_l, 2)
    bs, bl = round(bytes_s, 2), round(bytes_l, 2)
    near(ps + bs, ring_s, 0.006)
    near(pl + bl, ring_l, 0.006)
    n.update(alpha=fx(alpha, 1), beta=f"{beta:.0f}", gbit=f"{gbit:.0f}", nstar=f"{nstar:.0f}",
             bucket=f"{bucket:.0f}", small=f"{small}", large=f"{large:,}",
             chunksmall=fx(chunk_s, 1), chunklarge=fx(chunk_l, 1),
             stepssmall=f"{steps_s}", stepslarge=f"{steps_l:,}",
             startupsmall=fx(ps, 2), startuplarge=fx(pl, 2), bytessmall=fx(bs, 2), byteslarge=fx(bl, 2),
             ringsmall=fx(ps + bs, 2), ringlarge=fx(pl + bl, 2),
             ringslowdown=f"{(pl + bl) / (ps + bs):.1f}",
             # log10 of sizes in KB, for the worked example's ladder
             logchunklarge=fx(math.log10(chunk_l), 3), lognstar=fx(math.log10(nstar), 3),
             logchunksmall=fx(math.log10(chunk_s), 3), logbucket=fx(math.log10(mag(d.message, "KB")), 3),
             bucketkb=f"{mag(d.message, 'KB'):,.0f}")

    # ---- Problem to finish: the same bucket on 1,024 ranks with the simple tree (chapter: crossover
    # estimate N alpha beta / log2 N; tree model 2 log2 N (alpha + M/beta))
    log2n = int(math.log2(large))
    cross = mag(r.crossover_estimate[large], "MB")
    near(cross, large * alpha * beta / log2n / 1000, 1e-9)
    tree_rounds = 2 * log2n
    tree_start = mag(r.tree_startup_time[large], "ms")
    tree_bytes = mag(r.tree_byte_time[large], "ms")
    tree_total = mag(r.tree_time[large], "ms")
    near(tree_start + tree_bytes, tree_total, 1e-9)
    half_ms = bucket / beta                              # M / beta in ms (MB / (GB/s))
    near(tree_rounds * half_ms, tree_bytes, 1e-9)
    # double binary tree / halving-doubling estimate: log rounds with the ring's byte term
    dbt = round(tree_start, 2) + bl
    n.update(log2large=f"{log2n}", crossover=fx(cross, 2), treerounds=f"{tree_rounds}",
             treestartup=fx(tree_start, 2), perround=fx(half_ms, 1), treebytes=fx(tree_bytes, 0),
             treetotal=fx(round(tree_start, 2) + round(tree_bytes, 2), 2),
             treevsring=f"{(round(tree_start, 2) + round(tree_bytes, 2)) / (pl + bl):.1f}",
             doubletree=fx(dbt, 2), doubletreegain=f"{(pl + bl) / dbt:.1f}")

    # ---- Front picture: hierarchical vs flat, 1 GB on eight DGX H100 nodes (chapter: HierarchicalAllreduceCalc)
    h = SC.HierarchicalAllReduce8x8DgxH100
    hr = h.result
    flat = mag(hr.flat_bound_time, "ms")
    rs, inter, ag = mag(hr.reduce_scatter_time, "ms"), mag(hr.inter_time, "ms"), mag(hr.all_gather_time, "ms")
    tot = mag(hr.total_time, "ms")
    prs, pint, pag = round(rs, 2), round(inter, 2), round(ag, 2)
    near(prs + pint + pag, tot, 0.006)
    n.update(hiergb=f"{mag(h.message, 'GB'):.0f}", hiernodes=f"{h.n_nodes}", hiergpus=f"{h.gpus_per_node}",
             nvlink=f"{mag(h.intra_bandwidth, 'GB/s'):.0f}", shard=f"{mag(hr.shard, 'MB'):.0f}",
             flat=f"{flat:.0f}", hierrs=fx(prs, 2), hierinter=fx(pint, 2), hierag=fx(pag, 2),
             hiertotal=fx(prs + pint + pag, 2), hiergain=f"{flat / (prs + pint + pag):.1f}",
             # unrounded for drawing
             drawflat=fx(flat, 3), drawrs=fx(rs, 3), drawinter=fx(inter, 3), drawag=fx(ag, 3))
    near(flat / (prs + pint + pag), hr.speedup_vs_flat_bound, 0.05)

    # ---- Mistake: a hot expert rank sets the AllToAll (chapter: MoeAllToAll, MoeSkew)
    m = SC.MoeDispatch8Gpu64Experts
    mr = m.result
    rx_u, rx_h = mag(mr.uniform_receive_bytes, "MB"), mag(mr.hot_receive_bytes, "MB")
    bu, bh = mag(mr.uniform_receive_bound, "microsecond"), mag(mr.hot_receive_bound, "microsecond")
    near(round(rx_u, 3) / beta * 1000, bu, 0.2)
    near(round(round(rx_u, 3) * m.receive_skew, 2), round(rx_h, 2), 1e-9)   # printed hot volume from printed inputs
    near(round(rx_h, 2) / beta * 1000, bh, 0.2)
    n.update(moegpus=f"{m.n_devices}", moeexperts=f"{m.n_experts}", skew=fx(m.receive_skew, 1),
             rxuniform=fx(rx_u, 3), rxhot=fx(rx_h, 2), bounduniform=f"{bu:.0f}", boundhot=f"{bh:.0f}")

    # ---- Mistake: overlap leaves a backlog (chapter: OverlapBudgetCalc)
    o = SC.OverlapBudget7B64Gpu
    orr = o.result
    layers = orr.layers
    bwd = mag(o.backward_per_layer, "ms")
    comm = mag(orr.communication_per_layer, "ms")
    exposed = mag(orr.exposed_per_layer, "ms")
    seq, ovl = mag(orr.sequential_time, "ms"), mag(orr.overlapped_time, "ms")
    near(layers * bwd + comm + (layers - 1) * exposed, ovl, 1e-9)
    near(layers * bwd + layers * comm, seq, 1e-9)
    n.update(layers=f"{layers}", bwdlayer=f"{bwd:.0f}", commlayer=f"{comm:.0f}", exposed=f"{exposed:.0f}",
             bwdtotal=f"{layers * bwd:.0f}", layersless=f"{layers - 1}",
             sequential=f"{seq:,.0f}", overlapped=f"{ovl:,.0f}")

    # ---- Mistake: compression without error feedback (chapter: ErrorFeedbackFiveSteps)
    e = SC.ErrorFeedbackFiveSteps
    er = e.result
    n.update(efgrads=", ".join(f"{g:.1f}" for g in e.gradients), eftrue=fx(er.true_total, 1),
             efnaive=f"{er.naive_total_sent}", efsent=f"{er.total_sent}", efresidual=fx(er.final_error, 1))
    near(er.total_sent + er.final_error, er.true_total, 1e-9)

    # ---- Check yourself
    # Q2 ASSUMED: a 2 GB buffer on 16 DGX H100 nodes of eight (same NVLink and NDR ports as the chapter)
    q2 = h.variant(name="ASSUMED: 2 GB on 16 nodes", description="study sheet question",
                   message=Q_(2, "GB"), n_nodes=16)
    q2r = q2.result
    q2flat = mag(q2r.flat_bound_time, "ms")
    q2shard = mag(q2r.shard, "MB")
    q2inter = mag(q2r.inter_time, "ms")
    q2rs, q2ag = mag(q2r.reduce_scatter_time, "ms"), mag(q2r.all_gather_time, "ms")
    q2tot = round(q2rs, 1) + round(q2inter, 1) + round(q2ag, 1)
    near(q2tot, mag(q2r.total_time, "ms"), 0.06)
    q2shared = round(q2inter, 1) * q2.gpus_per_node      # shard streams serialize on one shared uplink
    n.update(qgb=f"{mag(q2.message, 'GB'):.0f}", qnodes=f"{q2.n_nodes}", qflat=f"{q2flat:.0f}",
             qshard=f"{q2shard:.0f}", qinter=fx(q2inter, 1), qrs=fx(q2rs, 1), qag=fx(q2ag, 1),
             qtotal=fx(q2tot, 1), qgain=f"{q2flat / q2tot:.1f}", qshared=f"{q2shared:.0f}",
             qsharedtotal=f"{q2shared + round(q2rs, 1) + round(q2ag, 1):.0f}",
             qringfactor=f"{2 * (q2.n_nodes - 1)}/{q2.n_nodes}",
             qnvfrac=f"{q2.gpus_per_node - 1}/{q2.gpus_per_node}")

    # Q3 ASSUMED: a 12 ms synchronization, effective 4x gain, 2 ms codec; 1 ms on a fast NVLink domain
    q3 = SC.CompressionPayback8x.variant(name="ASSUMED: 12 ms at 4x", description="study sheet question",
                                         communication_time=Q_(12, "ms"), compression_ratio=4,
                                         codec_time=Q_(2, "ms"), fast_communication_time=Q_(1, "ms"))
    t, rr, c, f = (mag(q3.communication_time, "ms"), q3.compression_ratio, mag(q3.codec_time, "ms"),
                   mag(q3.fast_communication_time, "ms"))
    assert c < t * (1 - 1 / rr) and c > f * (1 - 1 / rr)
    n.update(qcomm=f"{t:.0f}", qratio=f"{rr:.0f}", qcodec=f"{c:.0f}", qfast=f"{f:.0f}",
             qcompressed=f"{t / rr:.0f}", qnew=f"{t / rr + c:.0f}", qspeedup=f"{t / (t / rr + c):.1f}",
             qremoved=f"{t * (1 - 1 / rr):.0f}", qfastnew=f"{f / rr + c:.2f}",
             qfastremoved=f"{f * (1 - 1 / rr):.2f}", qfastslow=f"{(f / rr + c) / f:.2f}",
             qfastcompressed=f"{f / rr:.2f}")
    near(t / (t / rr + c), mag(q3.communication_time, "ms") / mag(q3.total_time, "ms"), 1e-9)
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
