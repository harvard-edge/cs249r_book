"""Numbers for the Conclusion study sheet (Volume II, Chapter 17).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells
(conclusion-scale-facts, conclusion-expansion, conclusion-expansion-overhead,
conclusion-expansion-scaling-gain), and written as \\setnum{key}{value} to
conclusion_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own questions are marked ASSUMED (built as variants of the
chapter's scenarios). A printed result is computed from the rounded values the sheet prints
beside it, and checked against the unrounded model value.
"""
import math
from pathlib import Path

from mlsysim import HOURS_PER_DAY, Literature, Models, Systems
from mlsysim.core.units import TB, hour
from mlsysim.engine.scenarios import Scenarios as SC


def fx(x, nd):
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}"


def near(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


SECONDS_PER_HOUR = hour.to("s").magnitude if hasattr(hour, "to") else 3600.0


def numbers() -> dict:
    n = {}

    # ---- Worked example: doubling the Archetype A reference cluster (chapter: conclusion-expansion,
    #      conclusion-expansion-overhead). GPU-only failures, Young-Daly overhead sqrt(2 T_write / MTBF).
    ex = SC.Gpt3ClusterDoubling1K
    mttf = ex.component_mtbf.to("hour").magnitude
    t_write = ex.checkpoint.t_write.to("s").magnitude
    ckpt_tb = ex.checkpoint.checkpoint_bytes.to(TB).magnitude
    ref_mtbf_, exp_mtbf_ = round(ex.mtbf.to("hour").magnitude, 1), round(ex.expanded_mtbf.to("hour").magnitude, 1)
    ref_mtbf_s = ref_mtbf_ * SECONDS_PER_HOUR
    o_ref = math.sqrt(2 * t_write / ref_mtbf_s)                      # from the printed interval
    o_exp = math.sqrt(2 * t_write / (exp_mtbf_ * SECONDS_PER_HOUR))
    near(o_ref, ex.overhead, 0.0005)
    near(o_exp, ex.expanded_overhead, 0.0005)
    o_ref_, o_exp_ = round(o_ref, 3), round(o_exp, 3)                # printed as 1.7 and 2.4 percent
    breakeven = (1 - o_ref_) / (1 - o_exp_)
    near(breakeven, ex.breakeven_speedup, 0.001)
    hours_change = (ex.factor / ex.factor) * breakeven - 1           # at the ideal step speedup = factor
    near(hours_change, ex.accelerator_hours_change(ex.factor), 0.001)
    n.update(
        refgpus=f"{ex.n_devices:,}", expgpus=f"{ex.expanded_devices:,.0f}", factor=f"{ex.factor:.0f}",
        mttf=f"{mttf:,.0f}", writes=fx(t_write, 1), ckpttb=fx(ckpt_tb, 2),
        bytesperparam=f"{ex.checkpoint.bytes_per_param.magnitude:.0f}",
        params=f"{ex.checkpoint.workload.parameters.to('param').magnitude / 1e9:.0f}",
        writebw=f"{ex.checkpoint.write_bandwidth.to('GB/s').magnitude:.0f}",
        refmtbf=fx(ref_mtbf_, 1), expmtbf=fx(exp_mtbf_, 1), refmtbfsec=f"{ref_mtbf_s:,.0f}",
        refover=pct(o_ref_), expover=pct(o_exp_), overgrowth=fx(ex.overhead_growth, 2),
        refkeep=fx(1 - o_ref_, 3), expkeep=fx(1 - o_exp_, 3),
        youngdaly="2",  # the Young-Daly overhead coefficient, sqrt(2 T_write / MTBF), as in the chapter cell
        breakeven=fx(breakeven, 3), hoursup=pct(hours_change),
        # curve on the back: overhead (percent) against GPU count, as sqrt(2 T_write N / MTTF)
        curvecoef=f"{math.sqrt(2 * t_write / (mttf * SECONDS_PER_HOUR)) * 100:.6f}",
        refoverraw=f"{ex.overhead * 100:.4f}", expoverraw=f"{ex.expanded_overhead * 100:.4f}",
    )

    # ---- Your turn ASSUMED: the same job quadrupled to 4,096 GPUs (chapter scenario, factor 4)
    q = ex.variant(factor=4.0)
    q_mtbf_ = round(q.expanded_mtbf.to("hour").magnitude, 1)
    q_o = math.sqrt(2 * t_write / (q_mtbf_ * SECONDS_PER_HOUR))
    near(q_o, q.expanded_overhead, 0.0005)
    q_o_ = round(q_o, 3)
    q_be = (1 - o_ref_) / (1 - q_o_)
    near(q_be, q.breakeven_speedup, 0.001)
    k = SC.KaplanComputeDoubling
    k_factor_ = round(k.multiple_factor, 3)
    k4 = k.variant(resource_multiple=4.0)
    four_factor = k_factor_ ** 2                                     # two doublings, from the printed factor
    near(four_factor, k4.multiple_factor, 0.001)
    n.update(quadfactor=f"{q.factor:.0f}", quadgpus=f"{q.expanded_devices:,.0f}", quadmtbf=fx(q_mtbf_, 1),
             quadmtbfsec=f"{q_mtbf_ * SECONDS_PER_HOUR:,.0f}", quadover=pct(q_o_), quadovergrowth=fx(q.overhead_growth, 0),
             quadkeep=fx(1 - q_o_, 3), quadbreakeven=fx(q_be, 3), quadhoursup=pct(q_be - 1),
             fourdoublings=f"{math.log2(k4.resource_multiple):.0f}", fourlossfactor=fx(four_factor, 3), fourlosscut=pct(1 - four_factor))

    # ---- Scaling card: Kaplan compute exponent, loss factor per doubling (conclusion-expansion-scaling-gain)
    n.update(kaplanalpha=fx(k.exponent, 3), lossfactor=fx(k_factor_, 3), losscut=pct(1 - k_factor_),
             doubling=f"{k.resource_multiple:.0f}")

    # ---- Mistakes (conclusion-scale-facts): fan-out tail and the straggler
    fan = SC.ServingFanout100Leaves
    hit = 1 - fan.availability ** fan.n_components
    near(hit, 1 - fan.all_up, 1e-12)
    slow = SC.SlowWorker8K
    n.update(fanservers=f"{fan.n_components}", fanpct=f"{fan.availability * 100:.0f}",
             fanprob=fx(fan.availability, 2), fanhit=pct(hit), slowspeed=f"{slow.slow_speed * 100:.0f}",
             slowdrop=f"{(1 - slow.throughput) * 100:.0f}", slowworkers=f"{slow.n_devices:,}")

    # ---- Front anchor: Llama 3 interruptions (conclusion-scale-facts)
    tr = Literature.TrainingRuns
    llama_gpus = Models.Language.Llama3_405B.training_accelerators_ref.magnitude
    per = tr.Llama3_405B_SnapshotDays * float(getattr(HOURS_PER_DAY, 'magnitude', HOURS_PER_DAY)) / tr.Llama3_405B_UnexpectedInterruptions
    n.update(llamagpus=f"{llama_gpus:,.0f}", llamastops=f"{tr.Llama3_405B_UnexpectedInterruptions:.0f}",
             llamadays=f"{tr.Llama3_405B_SnapshotDays:.0f}", llamahours=fx(per, 1),
             llamakept=f"{tr.Llama3_405B_EffectiveTrainingTimeFraction * 100:.0f}", hoursperday=f"{float(getattr(HOURS_PER_DAY, 'magnitude', HOURS_PER_DAY)):.0f}")
    assert Systems.Clusters.Training_1K.total_accelerators == ex.n_devices

    # ---- Check yourself
    # Q2 ASSUMED: fan-out to 50 servers at their p99, and to 100 servers at their p99.9
    f50 = fan.variant(n_components=50)
    f100 = fan.variant(availability=0.999)
    n.update(qfanfew=f"{f50.n_components}", qfanfewhit=pct(1 - f50.all_up), qfanmany=f"{f100.n_components}",
             qfantight=fx(f100.availability * 100, 1), qfantightprob=fx(f100.availability, 3),
             qfantighthit=pct(1 - f100.all_up), qfanfewall=fx(f50.all_up, 3), qfantightall=fx(f100.all_up, 3), qfanfewprob=fx(f50.availability, 2))

    # Q2 ASSUMED: the doubled cluster's measured uninterrupted step speedup is 1.6x (above break-even)
    measured = 1.6
    assert measured > breakeven
    hours_measured = ex.factor / measured * breakeven - 1
    n.update(qspeedup=fx(measured, 1), qhoursratio=fx(ex.factor / measured * breakeven, 2), qhoursup=f"{hours_measured * 100:.0f}")

    # Q3 ASSUMED: fabric links draw 12 percent of fleet power; new optics halve energy per bit at fixed traffic
    link_share, per_bit_cut = 0.12, 0.5
    n.update(qlinkshare=f"{link_share * 100:.0f}", qlinkcut=f"{per_bit_cut * 100:.0f}",
             qfleetsave=f"{link_share * per_bit_cut * 100:.0f}")
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
