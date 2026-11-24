"""Numbers for the Network Fabrics study sheet (Volume II, Chapter 3).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to network_fabrics_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
Rates: link rates are quoted in Gb/s (bits) and converted to GB/s (bytes) here; KB, MB, GB are decimal.
"""
import math
from pathlib import Path

from mlsysim import Systems
from mlsysim.engine.scenarios import Scenarios


def c(x):
    return f"{x:,.0f}"


def g(x, nd=3):
    return f"{x:.{nd}g}"


def numbers() -> dict:
    n = {}
    ndr = Systems.Fabrics.InfiniBand_NDR.bandwidth        # 400 Gb/s per port, per direction
    hdr = Systems.Fabrics.InfiniBand_HDR.bandwidth        # 200 Gb/s
    beta = ndr.to("GB/s").magnitude                         # bytes per second, decimal GB
    n.update(ndrgbit=f"{ndr.to('Gbps').magnitude:.0f}", ndrbytes=f"{beta:.0f}",
             hdrgbit=f"{hdr.to('Gbps').magnitude:.0f}")

    # ---- alpha-beta scenario of the chapter (Scenarios.AlphaBetaNdrRoce, cell NetworkAlphaBeta)
    ab = Scenarios.AlphaBetaNdrRoce
    alpha = ab.alphas["NDR"].to("microsecond").magnitude    # 1.5 us
    ctrl = ab.messages["control"].to("byte").magnitude      # 4 KB control message
    shard = ab.messages["shard"].to("byte").magnitude       # 350 MB gradient shard
    cross = alpha * 1e-6 * beta * 1e9                       # n* = alpha * beta, bytes

    def t_us(nbytes, a_us, b_gbs):
        return a_us + nbytes / (b_gbs * 1e9) * 1e6

    t_ctrl = t_us(ctrl, alpha, beta)
    t_shard = t_us(shard, alpha, beta)
    ctrl_xfer = ctrl / (beta * 1e9) * 1e6
    n.update(alpha=f"{alpha:.1f}", crossover=f"{cross / 1e3:.0f}", ctrlkb=f"{ctrl / 1e3:.0f}",
             shardmb=f"{shard / 1e6:.0f}", ctrlxfer=f"{ctrl_xfer:.2f}", ctrltime=f"{t_ctrl:.2f}",
             ctrlshare=f"{ctrl_xfer / t_ctrl * 100:.0f}", shardxfer=f"{shard / (beta * 1e9) * 1e3:.1f}",
             shardtime=f"{t_shard / 1e3:.1f}", kneetime=f"{2 * alpha:.0f}")
    # Doubled beta (chapter: "doubling bandwidth to 100 GB/s approximately halves the transfer time")
    beta2 = 2 * beta
    t_ctrl2 = t_us(ctrl, alpha, beta2)
    t_shard2 = t_us(shard, alpha, beta2)
    n.update(betadouble=f"{beta2:.0f}", ctrltimetwo=f"{t_ctrl2:.2f}", shardtimetwo=f"{t_shard2 / 1e3:.2f}",
             ctrlsaving=f"{(1 - t_ctrl2 / t_ctrl) * 100:.1f}", shardspeedup=f"{t_shard / t_shard2:.2f}")

    # Front picture: T(n) on log-log axes, drawn from the same alpha and beta.
    # x: 10^2 .. 10^9 bytes over 196 pt; y: 10^0 .. 10^4.5 us over 92 pt.
    x0, xs, y0, ys = 24.0, 28.0, 0.0, 20.0
    def px(nb):
        return x0 + (math.log10(nb) - 2) * xs
    def py(tu):
        return y0 + math.log10(tu) * ys
    pts = []
    for i in range(0, 71):
        nb = 10 ** (2 + i / 10)
        tu = t_us(nb, alpha, beta)
        if math.log10(tu) <= 4.5:
            pts.append(f"({px(nb):.1f},{py(tu):.1f})")
    n["abcurve"] = " ".join(pts)
    n.update(xctrl=f"{px(ctrl):.1f}", yctrl=f"{py(t_ctrl):.1f}", xshard=f"{px(shard):.1f}", yshard=f"{py(t_shard):.1f}",
             xknee=f"{px(cross):.1f}", yknee=f"{py(2 * alpha):.1f}", yalpha=f"{py(alpha):.1f}")

    # ---- Your turn: ASSUMED RoCE path, chapter's RoCE alpha (5 us) on the same 400 Gb/s line rate;
    # ASSUMED 32 KB pipeline activation handoff and the chapter's 350 MB shard.
    alpha_roce = ab.alphas["RoCE 400G"].to("microsecond").magnitude
    assert ab.bandwidths["RoCE 400G"] == ab.bandwidths["NDR"]
    hand = 32e3
    n.update(roceal=f"{alpha_roce:.0f}", rocecross=f"{alpha_roce * 1e-6 * beta * 1e9 / 1e3:.0f}", handkb=f"{hand / 1e3:.0f}",
             handxfer=f"{hand / (beta * 1e9) * 1e6:.2f}", handroce=f"{t_us(hand, alpha_roce, beta):.2f}",
             handndr=f"{t_us(hand, alpha, beta):.2f}",
             handratio=f"{t_us(hand, alpha_roce, beta) / t_us(hand, alpha, beta):.1f}",
             sharddiff=f"{t_us(shard, alpha_roce, beta) - t_shard:.1f}",
             sharddiffpct=f"{(t_us(shard, alpha_roce, beta) / t_shard - 1) * 100:.2f}")

    # ---- Mistake 1: HDR -> NDR for a 10 KB message (Scenarios.AlphaBetaHdrNdrSmallMessage, cell BandwidthFallacy)
    fb = Scenarios.AlphaBetaHdrNdrSmallMessage
    t_h = fb.result.times["HDR"]["small"].to("microsecond").magnitude
    t_n = fb.result.times["NDR"]["small"].to("microsecond").magnitude
    n.update(fallkb=f"{fb.messages['small'].to('KB').magnitude:.0f}", fallhdr=f"{t_h:.2f}", fallndr=f"{t_n:.2f}",
             fallspeed=f"{t_h / t_n:.2f}", fallless=f"{(1 - t_n / t_h) * 100:.1f}")

    # ---- Cut card: chapter's 1,024-GPU fat-tree, 4:1 spine (Scenarios.RingAllReduceAcross4to1Spine)
    cut = Scenarios.RingAllReduceAcross4to1Spine
    r = cut.result
    n.update(cutgpus=c(cut.fleet.total_accelerators), cutside=c(r.devices_per_side),
             cutfull=f"{r.bisection_full.to('TB/s').magnitude:.1f}",
             cutover=f"{r.bisection_oversubscribed.to('TB/s').magnitude:.1f}", cutratio=f"{cut.oversubscription_ratio:.0f}")
    # 2:1 spine at 30 percent communication (Scenarios.RingAllReduceAcross2to1Spine, cell NonBlockingOversubscription)
    two = Scenarios.RingAllReduceAcross2to1Spine
    n.update(tworatio=f"{two.oversubscription_ratio:.0f}", twocomm=f"{two.communication_fraction * 100:.0f}",
             twoloss=f"{two.result.throughput_loss * 100:.1f}")

    # ---- Sizing table: fat-tree of radix-64 NDR switches (Scenarios.FatTreeNdrSwitch)
    ft = Scenarios.FatTreeNdrSwitch
    n.update(radix=c(ft.radix), twotier=c(ft.two_tier_hosts), threetier=c(ft.three_tier_hosts))
    # Incast worst case (Scenarios.IncastSuperPodUnit, cell IncastBurst)
    inc = Scenarios.IncastSuperPodUnit
    ir = inc.result
    n.update(incsend=c(inc.n_senders), incbuf=f"{inc.buffer.to('MB').magnitude:.0f}",
             incfill=f"{ir.overflow_time.to('microsecond').magnitude:.1f}",
             incrtt=f"{inc.feedback_rtt.to('microsecond').magnitude:.0f}", incload=f"{ir.oversubscription:.0f}")

    # ---- Check yourself (ASSUMED inputs)
    # Q1: ASSUMED 2,048-GPU fat-tree, NDR links, 2:1 spine, job spends 25 percent of each step on cut traffic
    q1n, q1ratio, q1f = 2048, 2, 0.25
    full = q1n / 2 * beta / 1e3                       # TB/s per direction
    rel = 1 / ((1 - q1f) + q1f * q1ratio)
    n.update(qonegpus=c(q1n), qoneratio=c(q1ratio), qonecomm=f"{q1f * 100:.0f}", qoneside=c(q1n / 2),
             qonefull=f"{full:.1f}", qoneover=f"{full / q1ratio:.1f}", qonedenom=f"{(1 - q1f) + q1f * q1ratio:.2f}",
             qonerel=f"{rel:.2f}", qoneloss=f"{(1 - rel) * 100:.0f}", qonekeep=f"{(1 - q1f):.2f}",
             qonestretch=f"{q1f * q1ratio:.2f}")
    # Q2: ASSUMED 6,000-GPU cluster; radix-64 vs radix-128 switches, two-tier non-blocking fat-tree
    q2n, kbig = 6000, 128
    n.update(qtwogpus=c(q2n), qtwobig=c(kbig), qtwosmallhosts=c(ft.radix ** 2 / 2), qtwobighosts=c(kbig ** 2 / 2))
    assert ft.radix ** 2 / 2 == ft.two_tier_hosts
    # Q3: ASSUMED 8 GPUs each at the NDR rate into one NDR storage port with a 16 MB buffer; ECN round trip 5 us
    # (8 senders make the answer decisive: the buffer lasts many feedback round trips, unlike the chapter's incast)
    q3s, q3buf = 8, 16e6
    lam = q3s * beta * 1e9
    mu = beta * 1e9
    fill = q3buf / (lam - mu) * 1e6
    n.update(qthreesend=c(q3s), qthreebuf=f"{q3buf / 1e6:.0f}", qthreeload=f"{lam / 1e12:.1f}",
             qthreenet=f"{(lam - mu) / 1e12:.2f}", qthreefill=f"{fill:.1f}", qthreertts=f"{fill / inc.feedback_rtt.to('microsecond').magnitude:.0f}")
    assert fill / inc.feedback_rtt.to('microsecond').magnitude > 5      # decisive: many round trips
    # printed fill reproduces from the printed buffer and net rate
    assert abs(q3buf / 1e6 / float(f"{(lam - mu) / 1e12:.2f}") - float(f"{fill:.1f}")) < 0.06
    # Q4: one node pair measures half the NDR rate (chapter's debugging checkpoint uses the same halving)
    n.update(qfourmeas=f"{beta / 2:.0f}", qfourhdr=f"{hdr.to('GB/s').magnitude:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
